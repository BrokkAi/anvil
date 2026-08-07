use anyhow::Context;
use futures::StreamExt;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{Mutex, RwLock, oneshot};
use tokio_util::sync::CancellationToken;

const PROTOCOL_VERSION: &str = "2025-11-25";
pub const BUNDLED_BIFROST_VERSION: &str = "0.8.21";
const BIFROST_RELEASE_BASE: &str = "https://github.com/BrokkAi/bifrost/releases/download";
/// Budget for setup RPCs: the initialize handshake, `tools/list`, and SSE
/// endpoint discovery. These are expected to be fast; a long stall here
/// usually means the server (or its transport) is broken.
const MCP_STARTUP_TIMEOUT: Duration = Duration::from_secs(60);
/// Budget for `tools/call`. Kept generous because MCP tools can run
/// long-lived work server-side (e.g. Mjolnir's `code_agent`, which the
/// server itself may hold open for up to 240s). See
/// https://github.com/BrokkAi/anvil/issues/292 for the fuller adaptive-timeout
/// design this is standing in for.
const MCP_TOOL_CALL_TIMEOUT: Duration = Duration::from_secs(300);

#[cfg(target_os = "macos")]
const BIFROST_TARGET_TRIPLE: &str = "universal-apple-darwin";
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const BIFROST_TARGET_TRIPLE: &str = "x86_64-unknown-linux-musl";
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
const BIFROST_TARGET_TRIPLE: &str = "aarch64-unknown-linux-gnu";
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
const BIFROST_TARGET_TRIPLE: &str = "x86_64-pc-windows-msvc";
#[cfg(all(target_os = "windows", target_arch = "aarch64"))]
const BIFROST_TARGET_TRIPLE: &str = "aarch64-pc-windows-msvc";
#[cfg(all(target_os = "android", target_arch = "aarch64"))]
const BIFROST_TARGET_TRIPLE: &str = "aarch64-linux-android";

#[cfg(not(any(
    target_os = "macos",
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "linux", target_arch = "aarch64"),
    all(target_os = "windows", target_arch = "x86_64"),
    all(target_os = "windows", target_arch = "aarch64"),
    all(target_os = "android", target_arch = "aarch64"),
)))]
compile_error!(
    "bifrost releases only ship a universal macOS binary, x86_64/aarch64 Linux, \
     x86_64/aarch64 Windows, and aarch64 Android; this build cannot bundle Bifrost on other targets"
);

#[cfg(target_os = "windows")]
const BIFROST_ARCHIVE_EXT: &str = "zip";
#[cfg(not(target_os = "windows"))]
const BIFROST_ARCHIVE_EXT: &str = "tar.gz";

#[cfg(target_os = "windows")]
const BIFROST_BINARY_NAME: &str = "bifrost.exe";
#[cfg(not(target_os = "windows"))]
const BIFROST_BINARY_NAME: &str = "bifrost";

static PREPARED_BIFROST_PATH: OnceLock<PathBuf> = OnceLock::new();
static PREPARE_BIFROST_LOCK: Mutex<()> = Mutex::const_new(());

/// `Clone` so a single terminal transport failure (the stdio reader task
/// stopping) can be delivered to every caller that was waiting on that
/// connection.
#[derive(Debug, Clone)]
pub enum McpError {
    Spawn(String),
    Io(String),
    Protocol(String),
    /// HTTP 404 response to a streamable-HTTP request carrying an
    /// `Mcp-Session-Id`: the server has expired or evicted this session
    /// (e.g. rmcp's `LocalSessionManager` default 5-minute idle timeout --
    /// see `transport-streamable-http-server`'s `keep_alive`) and, per the
    /// MCP spec, expects the client to reinitialize. Carries the same
    /// formatted message text as `Protocol` (so logs/errors read
    /// identically) but is kept distinct so `call_tool_with_timeout` can
    /// reinitialize the session and retry once instead of failing the tool
    /// call outright.
    HttpSessionNotFound(String),
    JsonRpc {
        code: i64,
        message: String,
    },
    Timeout {
        tool: String,
        timeout: Duration,
    },
    Cancelled {
        tool: String,
    },
}

impl fmt::Display for McpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            McpError::Spawn(s) => write!(f, "spawn failed: {s}"),
            McpError::Io(s) => write!(f, "io error: {s}"),
            McpError::Protocol(s) => write!(f, "protocol error: {s}"),
            McpError::HttpSessionNotFound(s) => write!(f, "protocol error: {s}"),
            McpError::JsonRpc { code, message } => {
                write!(f, "jsonrpc error {code}: {message}")
            }
            McpError::Timeout { tool, timeout } => {
                write!(f, "tool '{tool}' timed out after {}s", timeout.as_secs())
            }
            McpError::Cancelled { tool } => write!(f, "tool '{tool}' was cancelled"),
        }
    }
}

impl std::error::Error for McpError {}

#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum McpTransport {
    #[default]
    Stdio,
    Http,
    Sse,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpServerConfig {
    pub name: String,
    #[serde(default)]
    pub transport: McpTransport,
    #[serde(default)]
    pub command: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub headers: Vec<McpEnvVar>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub env: Vec<McpEnvVar>,
    #[serde(default)]
    pub framing: McpFraming,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpEnvVar {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum McpFraming {
    #[default]
    ContentLength,
    Line,
}

impl McpFraming {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "content-length" | "contentlength" | "framed" | "standard" => Some(Self::ContentLength),
            "line" | "line-delimited" | "ndjson" => Some(Self::Line),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::ContentLength => "content-length",
            Self::Line => "line",
        }
    }
}

fn default_enabled() -> bool {
    true
}

const DEFAULT_BIFROST_TOOLSET: &str = "core";

fn bifrost_args(flag: &str, toolset: &str) -> Vec<String> {
    vec![
        "--root".to_string(),
        "{cwd}".to_string(),
        flag.to_string(),
        toolset.to_string(),
        "--no-line-numbers".to_string(),
    ]
}

fn default_bifrost_args() -> Vec<String> {
    bifrost_args("--mcp", DEFAULT_BIFROST_TOOLSET)
}

fn managed_bifrost_cache_dir() -> anyhow::Result<PathBuf> {
    Ok(crate::setup_state::config_home()?
        .join("bifrost")
        .join(BUNDLED_BIFROST_VERSION)
        .join(BIFROST_TARGET_TRIPLE))
}

fn managed_bifrost_binary_path() -> anyhow::Result<PathBuf> {
    Ok(managed_bifrost_cache_dir()?.join(BIFROST_BINARY_NAME))
}

// Not cached: config_home() can return different values depending on the
// environment (test thread-local vs. env-var override vs. OS default), so a
// process-wide OnceLock here would break test isolation.  The call is cheap
// (env-var lookup or OS config-dir resolution) and is made only during session
// startup, not on any hot path.
fn managed_bifrost_command() -> String {
    PREPARED_BIFROST_PATH
        .get()
        .cloned()
        .or_else(|| managed_bifrost_binary_path().ok())
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "bifrost".to_string())
}

fn is_default_or_managed_bifrost_command(command: &str) -> bool {
    command == "bifrost" || command == managed_bifrost_command()
}

/// Normalises a stored Bifrost server entry so it always uses Anvil's managed
/// local binary with the correct line framing.
///
/// The function matches on name, the current default args, and the default
/// (`"bifrost"`) or managed-path command. When all three match the **entire**
/// `McpServerConfig` is replaced by [`McpServerConfig::bifrost()`]; only
/// `enabled` is preserved. Any other customisation (e.g. a manually-set
/// framing override) is intentionally discarded — Bifrost's wire protocol
/// requires line framing and the command must point to the pinned managed
/// binary.
/// Argument sets that earlier Anvil versions shipped as the managed Bifrost
/// default. A persisted entry still carrying one of these (with the managed
/// command) is an unmodified prior default, so it follows the current default
/// on load. Custom toolset combinations are deliberately not included.
fn legacy_default_bifrost_arg_sets() -> Vec<Vec<String>> {
    vec![
        bifrost_args("--server", "core"),
        bifrost_args("--server", "searchtools"),
        bifrost_args("--mcp", "searchtools"),
    ]
}

/// True if `args` is the current managed default or a recognized prior default.
fn is_managed_default_bifrost_args(args: &[String]) -> bool {
    args == default_bifrost_args().as_slice()
        || legacy_default_bifrost_arg_sets()
            .iter()
            .any(|legacy| args == legacy.as_slice())
}

pub fn normalize_preinstalled_bifrost_server(server: &mut McpServerConfig) {
    if server.name != "bifrost"
        || !is_managed_default_bifrost_args(&server.args)
        || !is_default_or_managed_bifrost_command(&server.command)
    {
        return;
    }
    let enabled = server.enabled;
    *server = McpServerConfig::bifrost();
    server.enabled = enabled;
}

pub async fn ensure_bundled_bifrost() -> anyhow::Result<PathBuf> {
    if let Some(path) = PREPARED_BIFROST_PATH.get() {
        return Ok(path.clone());
    }

    let _guard = PREPARE_BIFROST_LOCK.lock().await;
    if let Some(path) = PREPARED_BIFROST_PATH.get() {
        return Ok(path.clone());
    }

    let cache_dir = managed_bifrost_cache_dir()?;
    let binary = managed_bifrost_binary_path()?;
    if !binary.is_file() {
        download_and_extract_bifrost(&cache_dir).await?;
    }
    anyhow::ensure!(
        binary.is_file(),
        "expected bundled bifrost at {} after preparation",
        binary.display()
    );
    let _ = PREPARED_BIFROST_PATH.set(binary.clone());
    Ok(binary)
}

async fn download_and_extract_bifrost(cache_dir: &Path) -> anyhow::Result<()> {
    tokio::fs::create_dir_all(cache_dir)
        .await
        .with_context(|| format!("creating bifrost cache dir {}", cache_dir.display()))?;

    let _install_lock = acquire_bifrost_install_lock(cache_dir).await?;

    let target = cache_dir.join(BIFROST_BINARY_NAME);
    if target.is_file() {
        return Ok(());
    }

    let asset =
        format!("bifrost-v{BUNDLED_BIFROST_VERSION}-{BIFROST_TARGET_TRIPLE}.{BIFROST_ARCHIVE_EXT}");
    let url = format!("{BIFROST_RELEASE_BASE}/v{BUNDLED_BIFROST_VERSION}/{asset}");
    let sha256_url = format!("{url}.sha256");

    // A single client with an explicit timeout shared across both requests so a
    // slow or dropped CDN connection does not stall startup indefinitely.
    let client = crate::llm_client::OpenAiClient::apply_runtime_tls_workarounds(
        reqwest::Client::builder().timeout(std::time::Duration::from_secs(120)),
        &url,
    )
    .build()
    .context("building reqwest client for bifrost download")?;

    tracing::info!(%url, version = BUNDLED_BIFROST_VERSION, "downloading bundled bifrost");
    let bytes = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("downloading bundled bifrost archive from {url}"))?
        .error_for_status()
        .with_context(|| format!("bundled bifrost archive request failed for {url}"))?
        .bytes()
        .await
        .context("reading bundled bifrost archive bytes")?;

    tracing::info!(
        %sha256_url,
        version = BUNDLED_BIFROST_VERSION,
        "verifying bundled bifrost archive"
    );
    let sidecar = client
        .get(&sha256_url)
        .send()
        .await
        .with_context(|| format!("downloading bundled bifrost checksum from {sha256_url}"))?
        .error_for_status()
        .with_context(|| format!("bundled bifrost checksum request failed for {sha256_url}"))?
        .text()
        .await
        .context("reading bundled bifrost checksum text")?;
    let expected_hex = sidecar
        .split_whitespace()
        .next()
        .context("bundled bifrost checksum sidecar is empty")?
        .to_lowercase();
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let actual_hex = hasher
        .finalize()
        .as_slice()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    anyhow::ensure!(
        actual_hex == expected_hex,
        "bundled bifrost sha256 mismatch for {url}: got {actual_hex}, expected {expected_hex}"
    );

    let staging_dir = cache_dir.join(format!(
        ".extract-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    tokio::fs::create_dir(&staging_dir).await.with_context(|| {
        format!(
            "creating bundled bifrost staging dir {}",
            staging_dir.display()
        )
    })?;

    let archive_path = staging_dir.join(&asset);
    tokio::fs::write(&archive_path, &bytes)
        .await
        .with_context(|| format!("writing bundled bifrost archive {}", archive_path.display()))?;

    // `tar -xf` auto-detects the format; on Windows 10+ (build 17063) the
    // inbox BSD tar handles both `.tar.gz` and `.zip`, so a single invocation
    // covers all supported targets without an extra dependency.
    let status = Command::new("tar")
        .arg("-xf")
        .arg(&archive_path)
        .arg("-C")
        .arg(&staging_dir)
        .status()
        .await
        .with_context(|| format!("invoking tar to extract {}", archive_path.display()))?;
    anyhow::ensure!(
        status.success(),
        "tar extraction failed for {} with status {status}",
        archive_path.display()
    );

    let inner_dir = staging_dir.join(format!(
        "bifrost-v{BUNDLED_BIFROST_VERSION}-{BIFROST_TARGET_TRIPLE}"
    ));
    let inner_binary = inner_dir.join(BIFROST_BINARY_NAME);
    anyhow::ensure!(
        inner_binary.is_file(),
        "expected extracted bifrost binary at {}",
        inner_binary.display()
    );

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = tokio::fs::metadata(&inner_binary)
            .await
            .with_context(|| format!("stat bundled bifrost binary {}", inner_binary.display()))?
            .permissions();
        perms.set_mode(0o755);
        tokio::fs::set_permissions(&inner_binary, perms)
            .await
            .with_context(|| format!("chmod 755 {}", inner_binary.display()))?;
    }

    tokio::fs::rename(&inner_binary, &target)
        .await
        .with_context(|| {
            format!(
                "atomically installing bundled bifrost from {} to {}",
                inner_binary.display(),
                target.display()
            )
        })?;

    let _ = tokio::fs::remove_dir_all(&staging_dir).await;

    Ok(())
}

struct BifrostInstallLock {
    #[cfg(unix)]
    file: std::fs::File,
    #[cfg(not(unix))]
    path: PathBuf,
}

impl Drop for BifrostInstallLock {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            // Unlock best-effort on drop. The OS also releases the advisory
            // lock when the file descriptor closes, so failure here cannot
            // leave a held lock behind.
            // SAFETY: `as_raw_fd` returns a valid descriptor owned by `self.file`
            // for the duration of this call, and `flock` does not retain it.
            let _ = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
        }

        #[cfg(not(unix))]
        {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

async fn acquire_bifrost_install_lock(cache_dir: &Path) -> anyhow::Result<BifrostInstallLock> {
    let lock_path = cache_dir.join(".install.lock");

    #[cfg(unix)]
    {
        let file = tokio::task::spawn_blocking({
            let lock_path = lock_path.clone();
            move || -> anyhow::Result<std::fs::File> {
                use std::os::fd::AsRawFd;

                let file = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .open(&lock_path)
                    .with_context(|| {
                        format!("opening bifrost install lock {}", lock_path.display())
                    })?;
                // SAFETY: `as_raw_fd` returns a valid descriptor owned by
                // `file` for the duration of this call, and `flock` does not
                // retain it.
                let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
                anyhow::ensure!(
                    rc == 0,
                    "locking bifrost install lock {} failed: {}",
                    lock_path.display(),
                    std::io::Error::last_os_error()
                );
                Ok(file)
            }
        })
        .await
        .context("joining bifrost install lock task")??;
        Ok(BifrostInstallLock { file })
    }

    #[cfg(not(unix))]
    {
        loop {
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
            {
                Ok(_) => return Ok(BifrostInstallLock { path: lock_path }),
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                Err(err) => {
                    return Err(err).with_context(|| {
                        format!("creating bifrost install lock {}", lock_path.display())
                    });
                }
            }
        }
    }
}

impl McpServerConfig {
    pub fn bifrost() -> Self {
        Self {
            name: "bifrost".to_string(),
            transport: McpTransport::Stdio,
            command: managed_bifrost_command(),
            url: None,
            headers: Vec::new(),
            args: default_bifrost_args(),
            env: Vec::new(),
            framing: McpFraming::Line,
            enabled: true,
        }
    }

    pub fn rendered_args(&self, cwd: &Path) -> Vec<String> {
        let cwd = cwd.display().to_string();
        self.args
            .iter()
            .map(|arg| arg.replace("{cwd}", &cwd))
            .collect()
    }
}

pub fn default_servers() -> Vec<McpServerConfig> {
    vec![McpServerConfig::bifrost()]
}

#[derive(Debug, Clone)]
pub struct McpToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub annotations: McpToolAnnotations,
}

#[derive(Debug, Clone, Default)]
pub struct McpToolAnnotations {
    pub read_only_hint: Option<bool>,
}

/// JSON-RPC client for a long-lived MCP server connection.
///
/// For the stdio transport the client holds the subprocess for its own
/// lifetime; the process is killed when the client is dropped
/// (`kill_on_drop(true)`). The stdio protocol carries JSON-RPC messages either
/// `Content-Length`-framed or line-delimited (see [`McpFraming`]).
///
/// `state` is the *connection identity*: which subprocess or HTTP session this
/// client currently talks to. It is taken only to check liveness, to respawn a
/// dead stdio subprocess, and to pick up the connection handle. It is **not**
/// held across a request/response round trip on the stdio path -- see
/// [`StdioConn`], whose reader task demultiplexes responses so that the
/// parallel tool batches issued by `execute_parallel_safe_calls` overlap
/// instead of queueing behind each other.
pub struct McpClient {
    name: String,
    config: McpServerConfig,
    cwd: PathBuf,
    state: Mutex<McpClientState>,
    next_id: AtomicI64,
    tools: Vec<McpToolDef>,
    instructions: RwLock<Option<String>>,
}

enum McpClientState {
    Stdio(Arc<StdioConn>),
    Http {
        client: reqwest::Client,
        url: String,
        headers: reqwest::header::HeaderMap,
        session_id: Option<reqwest::header::HeaderValue>,
    },
    Sse {
        client: reqwest::Client,
        endpoint: String,
        headers: reqwest::header::HeaderMap,
        responses: tokio::sync::mpsc::UnboundedReceiver<Value>,
        _reader: tokio::task::JoinHandle<()>,
    },
}

/// One live stdio MCP subprocess, shared by every in-flight call to it.
///
/// A single JSON-RPC stream is multiplexed: a caller takes `writer` only for
/// as long as it takes to serialize and flush one message, and then awaits a
/// per-request oneshot registered in `shared.pending`. The dedicated reader
/// task spawned by [`StdioConn::new`] is the only reader of the subprocess's
/// stdout and routes each response to its caller by JSON-RPC id.
struct StdioConn {
    /// Only taken to kill the subprocess when the connection is retired.
    child: Mutex<Child>,
    writer: Mutex<StdioWriter>,
    shared: Arc<StdioShared>,
    /// Stops the reader task when this connection is dropped (on respawn, or
    /// when the client goes away).
    _reader: AbortOnDrop,
}

/// The part of a [`StdioConn`] that the reader task shares with the callers it
/// serves. Separate from `StdioConn` so the task holds no reference to the
/// subprocess it is reading from and cannot keep it alive.
struct StdioShared {
    /// Cleared by the first error that leaves the transport unusable, and by
    /// the reader task when it stops. The next `tools/call` respawns the
    /// subprocess.
    healthy: AtomicBool,
    pending: StdMutex<PendingResponses>,
}

/// Callers waiting for a stdio response, keyed by the JSON-RPC id of their
/// request.
enum PendingResponses {
    Open(HashMap<i64, oneshot::Sender<Result<Value, McpError>>>),
    /// The reader task has stopped, so nothing further can ever arrive on this
    /// connection. Registering fails immediately with the error that killed
    /// the reader instead of leaving the caller to wait out its own timeout.
    Closed(McpError),
}

struct StdioWriter {
    writer: ChildStdin,
    framing: McpFraming,
}

struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

impl StdioConn {
    /// Take ownership of a subprocess whose handshake has already completed on
    /// `reader`/`writer`, and start demultiplexing everything that follows.
    fn new(
        child: Child,
        writer: StdioWriter,
        reader: BufReader<ChildStdout>,
        framing: McpFraming,
        server: String,
    ) -> Self {
        let shared = Arc::new(StdioShared {
            healthy: AtomicBool::new(true),
            pending: StdMutex::new(PendingResponses::Open(HashMap::new())),
        });
        let task = tokio::spawn(stdio_reader_loop(
            reader,
            framing,
            Arc::clone(&shared),
            server,
        ));
        Self {
            child: Mutex::new(child),
            writer: Mutex::new(writer),
            shared,
            _reader: AbortOnDrop(task),
        }
    }

    fn healthy(&self) -> bool {
        self.shared.healthy.load(Ordering::SeqCst)
    }

    /// Register a waiter for `id`. Must happen before the request is written:
    /// otherwise a fast server could answer before the reader task has anyone
    /// to hand the response to.
    fn register(&self, id: i64) -> Result<oneshot::Receiver<Result<Value, McpError>>, McpError> {
        let (sender, receiver) = oneshot::channel();
        match &mut *self.pending() {
            PendingResponses::Open(waiters) => {
                let displaced = waiters.insert(id, sender);
                debug_assert!(
                    displaced.is_none(),
                    "duplicate in-flight MCP request id {id}"
                );
                Ok(receiver)
            }
            PendingResponses::Closed(err) => Err(err.clone()),
        }
    }

    /// Drop the waiter for `id` after abandoning the request (timeout or
    /// cancellation), so a late response is discarded rather than routed.
    fn deregister(&self, id: i64) {
        if let PendingResponses::Open(waiters) = &mut *self.pending() {
            waiters.remove(&id);
        }
    }

    fn pending(&self) -> std::sync::MutexGuard<'_, PendingResponses> {
        self.shared
            .pending
            .lock()
            .expect("mcp pending-response map poisoned")
    }

    async fn write_request(&self, id: i64, method: &str, params: Value) -> Result<(), McpError> {
        write_request(&mut *self.writer.lock().await, id, method, params).await
    }

    async fn write_notification(&self, method: &str, params: Value) -> Result<(), McpError> {
        write_notification(&mut *self.writer.lock().await, method, params).await
    }

    /// Retire the whole connection when one call fails in a way that leaves
    /// the transport unusable: kill the subprocess so the next `tools/call`
    /// respawns it.
    async fn mark_unhealthy(&self, err: &McpError) {
        if !err.leaves_client_unhealthy() {
            return;
        }
        self.shared.healthy.store(false, Ordering::SeqCst);
        let _ = self.child.lock().await.kill().await;
    }
}

/// Sole reader of a stdio MCP server's stdout.
///
/// Routes each JSON-RPC response to the caller registered under its id. On
/// stream error or EOF it fails every pending caller with that error, so a
/// server that dies mid-flight does not leave a batch of calls waiting out
/// their individual timeouts.
async fn stdio_reader_loop(
    mut reader: BufReader<ChildStdout>,
    framing: McpFraming,
    shared: Arc<StdioShared>,
    server: String,
) {
    loop {
        let message = match read_message(&mut reader, framing).await {
            Ok(message) => message,
            Err(err) => {
                let waiters = {
                    let mut pending = shared
                        .pending
                        .lock()
                        .expect("mcp pending-response map poisoned");
                    match std::mem::replace(&mut *pending, PendingResponses::Closed(err.clone())) {
                        PendingResponses::Open(waiters) => waiters,
                        PendingResponses::Closed(_) => return,
                    }
                };
                shared.healthy.store(false, Ordering::SeqCst);
                tracing::debug!(
                    server = %server,
                    %err,
                    pending = waiters.len(),
                    "mcp stdio reader stopped; failing pending calls"
                );
                for (_, waiter) in waiters {
                    let _ = waiter.send(Err(err.clone()));
                }
                return;
            }
        };
        // Server-initiated requests and notifications carry `method`. Any id
        // on them belongs to the server's own id space and must never be
        // matched against our pending requests. We advertise no client
        // capabilities, so there is nothing here to answer.
        if message.get("method").is_some() {
            tracing::debug!(server = %server, ?message, "ignoring server-initiated mcp message");
            continue;
        }
        let Some(id) = message.get("id").and_then(Value::as_i64) else {
            tracing::debug!(server = %server, ?message, "ignoring mcp message without a request id");
            continue;
        };
        let waiter = {
            let mut pending = shared
                .pending
                .lock()
                .expect("mcp pending-response map poisoned");
            match &mut *pending {
                PendingResponses::Open(waiters) => waiters.remove(&id),
                PendingResponses::Closed(_) => return,
            }
        };
        match waiter {
            Some(waiter) => {
                let _ = waiter.send(jsonrpc_result(message));
            }
            // Abandoned by a timed-out or cancelled caller.
            None => tracing::debug!(server = %server, id, "no waiter for mcp response"),
        }
    }
}

impl McpClient {
    pub async fn spawn(config: &McpServerConfig, cwd: &Path) -> Result<Self, McpError> {
        let next_id = AtomicI64::new(1);
        let (state, tools, instructions) = Self::spawn_connected(config, cwd, &next_id).await?;

        Ok(Self {
            name: config.name.clone(),
            config: config.clone(),
            cwd: cwd.to_path_buf(),
            state: Mutex::new(state),
            next_id,
            tools,
            instructions: RwLock::new(instructions),
        })
    }

    async fn spawn_connected(
        config: &McpServerConfig,
        cwd: &Path,
        next_id: &AtomicI64,
    ) -> Result<(McpClientState, Vec<McpToolDef>, Option<String>), McpError> {
        match config.transport {
            McpTransport::Http => return Self::connect_http(config, next_id).await,
            McpTransport::Sse => return Self::connect_sse(config, next_id).await,
            McpTransport::Stdio => {}
        }

        let rendered_args = config.rendered_args(cwd);
        let mut child = Command::new(&config.command)
            .args(&rendered_args)
            .envs(config.env.iter().map(|var| (&var.name, &var.value)))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| McpError::Spawn(format!("{}: {e}", config.command)))?;

        let writer = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Spawn("missing stdin pipe".into()))?;
        let mut reader = BufReader::new(
            child
                .stdout
                .take()
                .ok_or_else(|| McpError::Spawn("missing stdout pipe".into()))?,
        );

        let mut writer = StdioWriter {
            writer,
            framing: config.framing,
        };

        // The handshake is inherently sequential, so it reads the stream
        // directly; the demultiplexing reader task takes over once it is done.
        let init_id = next_id.fetch_add(1, Ordering::SeqCst);
        write_request(
            &mut writer,
            init_id,
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "brokk-acp-rust",
                    "version": env!("CARGO_PKG_VERSION"),
                },
            }),
        )
        .await?;
        let init = read_response(&mut reader, config.framing, init_id).await?;
        let instructions = parse_server_instructions(&init);

        write_notification(&mut writer, "notifications/initialized", json!({})).await?;

        let list_id = next_id.fetch_add(1, Ordering::SeqCst);
        write_request(&mut writer, list_id, "tools/list", json!({})).await?;
        let list = read_response(&mut reader, config.framing, list_id).await?;
        let tools = parse_tool_list(list, Some(config.name.as_str()))?;
        let read_only_hint_count = tools
            .iter()
            .filter(|tool| tool.annotations.read_only_hint == Some(true))
            .count();

        tracing::info!(
            server = %config.name,
            command = %config.command,
            args = ?rendered_args,
            framing = %config.framing.as_str(),
            cwd = %cwd.display(),
            tool_count = tools.len(),
            read_only_hint_count,
            "mcp server ready"
        );

        Ok((
            McpClientState::Stdio(Arc::new(StdioConn::new(
                child,
                writer,
                reader,
                config.framing,
                config.name.clone(),
            ))),
            tools,
            instructions,
        ))
    }

    async fn connect_http(
        config: &McpServerConfig,
        next_id: &AtomicI64,
    ) -> Result<(McpClientState, Vec<McpToolDef>, Option<String>), McpError> {
        let url = config
            .url
            .as_deref()
            .ok_or_else(|| McpError::Protocol("HTTP MCP server missing URL".into()))?
            .to_string();
        let mut headers = reqwest::header::HeaderMap::new();
        for header in &config.headers {
            let name = reqwest::header::HeaderName::from_bytes(header.name.as_bytes())
                .map_err(|e| McpError::Protocol(format!("invalid HTTP header name: {e}")))?;
            let value = reqwest::header::HeaderValue::from_str(&header.value)
                .map_err(|e| McpError::Protocol(format!("invalid HTTP header value: {e}")))?;
            headers.append(name, value);
        }
        let client = build_mcp_http_client(&url)?;
        let init_id = next_id.fetch_add(1, Ordering::SeqCst);
        let (init, session_id) = http_request_with_session(
            &client,
            &url,
            &headers,
            None,
            init_id,
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": { "name": "brokk-acp-rust", "version": env!("CARGO_PKG_VERSION") },
            }),
        )
        .await?;
        let instructions = parse_server_instructions(&init);
        http_notification(
            &client,
            &url,
            &headers,
            session_id.as_ref(),
            "notifications/initialized",
            json!({}),
        )
        .await?;
        let list_id = next_id.fetch_add(1, Ordering::SeqCst);
        let list = http_request(
            &client,
            &url,
            &headers,
            session_id.as_ref(),
            list_id,
            "tools/list",
            json!({}),
        )
        .await?;
        let tools = parse_tool_list(list, Some(config.name.as_str()))?;
        tracing::info!(server = %config.name, url = %url, tool_count = tools.len(), "HTTP MCP server ready");
        Ok((
            McpClientState::Http {
                client,
                url,
                headers,
                session_id,
            },
            tools,
            instructions,
        ))
    }

    async fn connect_sse(
        config: &McpServerConfig,
        next_id: &AtomicI64,
    ) -> Result<(McpClientState, Vec<McpToolDef>, Option<String>), McpError> {
        let url = config
            .url
            .as_deref()
            .ok_or_else(|| McpError::Protocol("SSE MCP server missing URL".into()))?;
        let headers = build_http_headers(&config.headers)?;
        let client = build_mcp_http_client(url)?;
        let response = client
            .get(url)
            .headers(headers.clone())
            .header(reqwest::header::ACCEPT, "text/event-stream")
            .send()
            .await
            .map_err(|e| McpError::Io(format!("open SSE stream: {e}")))?;
        if !response.status().is_success() {
            return Err(McpError::Protocol(format!(
                "SSE endpoint returned HTTP {}",
                response.status()
            )));
        }
        let base_url = reqwest::Url::parse(url)
            .map_err(|e| McpError::Protocol(format!("invalid SSE URL: {e}")))?;
        let (endpoint_tx, endpoint_rx) = tokio::sync::oneshot::channel();
        let (response_tx, response_rx) = tokio::sync::mpsc::unbounded_channel();
        let reader = tokio::spawn(read_sse_stream(
            response.bytes_stream(),
            base_url,
            endpoint_tx,
            response_tx,
        ));
        let endpoint = tokio::time::timeout(MCP_STARTUP_TIMEOUT, endpoint_rx)
            .await
            .map_err(|_| McpError::Timeout {
                tool: "SSE endpoint discovery".into(),
                timeout: MCP_STARTUP_TIMEOUT,
            })?
            .map_err(|_| McpError::Protocol("SSE stream closed before endpoint event".into()))??;
        let mut state = McpClientState::Sse {
            client,
            endpoint,
            headers,
            responses: response_rx,
            _reader: reader,
        };
        let init_id = next_id.fetch_add(1, Ordering::SeqCst);
        let init = sse_request(
            &mut state,
            init_id,
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": { "name": "brokk-acp-rust", "version": env!("CARGO_PKG_VERSION") },
            }),
        )
        .await?;
        let instructions = parse_server_instructions(&init);
        sse_notification(&state, "notifications/initialized", json!({})).await?;
        let list_id = next_id.fetch_add(1, Ordering::SeqCst);
        let list = sse_request(&mut state, list_id, "tools/list", json!({})).await?;
        let tools = parse_tool_list(list, Some(config.name.as_str()))?;
        tracing::info!(server = %config.name, url, tool_count = tools.len(), "SSE MCP server ready");
        Ok((state, tools, instructions))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn tools(&self) -> &[McpToolDef] {
        &self.tools
    }

    /// Instructions advertised by this server's most recent successful
    /// initialize handshake. An unhealthy stdio client is omitted until its
    /// next tool call reconnects it successfully.
    pub async fn instructions(&self) -> Option<String> {
        let state = self.state.lock().await;
        if let McpClientState::Stdio(conn) = &*state
            && !conn.healthy()
        {
            return None;
        }
        self.instructions.read().await.clone()
    }

    pub async fn call_tool(&self, name: &str, args: Value) -> Result<Value, McpError> {
        self.call_tool_with_timeout(name, args, MCP_TOOL_CALL_TIMEOUT, None)
            .await
    }

    pub async fn call_tool_cancellable(
        &self,
        name: &str,
        args: Value,
        cancel: Option<&CancellationToken>,
    ) -> Result<Value, McpError> {
        self.call_tool_with_timeout(name, args, MCP_TOOL_CALL_TIMEOUT, cancel)
            .await
    }

    pub(crate) async fn call_tool_with_timeout(
        &self,
        name: &str,
        args: Value,
        timeout: Duration,
        cancel: Option<&CancellationToken>,
    ) -> Result<Value, McpError> {
        let mut state = match cancel {
            Some(cancel) => {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        return Err(McpError::Cancelled { tool: name.to_string() });
                    }
                    state = self.state.lock() => state,
                }
            }
            None => self.state.lock().await,
        };

        if matches!(&*state, McpClientState::Stdio(conn) if !conn.healthy()) {
            let (new_state, _, new_instructions) =
                Self::spawn_connected(&self.config, &self.cwd, &self.next_id).await?;
            *state = new_state;
            *self.instructions.write().await = new_instructions;
        }

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        if matches!(&*state, McpClientState::Sse { .. }) {
            let call = sse_request(
                &mut state,
                id,
                "tools/call",
                json!({ "name": name, "arguments": args }),
            );
            let outcome = match cancel {
                Some(cancel) => tokio::select! {
                    biased;
                    _ = cancel.cancelled() => ToolCallOutcome::Cancelled,
                    result = tokio::time::timeout(timeout, call) => ToolCallOutcome::from(result),
                },
                None => ToolCallOutcome::from(tokio::time::timeout(timeout, call).await),
            };
            let result = match outcome {
                ToolCallOutcome::Done(result) => result,
                ToolCallOutcome::Cancelled => Err(McpError::Cancelled {
                    tool: name.to_string(),
                }),
                ToolCallOutcome::TimedOut => {
                    notify_cancelled_best_effort(sse_notification(
                        &state,
                        "notifications/cancelled",
                        cancelled_notification_params(id),
                    ))
                    .await;
                    Err(McpError::Timeout {
                        tool: name.to_string(),
                        timeout,
                    })
                }
            }?;
            return parse_tool_result(result);
        }

        if matches!(&*state, McpClientState::Http { .. }) {
            let params = json!({ "name": name, "arguments": args });
            let first = {
                let McpClientState::Http {
                    client,
                    url,
                    headers,
                    session_id,
                } = &*state
                else {
                    unreachable!()
                };
                call_http_tools_call(
                    client,
                    url,
                    headers,
                    session_id.as_ref(),
                    id,
                    name,
                    params.clone(),
                    timeout,
                    cancel,
                )
                .await
            };
            let result = match first {
                // Per the MCP streamable-HTTP spec, a 404 to a
                // session-bearing request means the server considers the
                // session gone (rmcp's `LocalSessionManager` evicts it after
                // `keep_alive`, default 5 minutes idle -- see
                // `code_agent.rs`'s `HttpServer::start`, which does not
                // override it). The client is expected to reinitialize
                // rather than keep presenting the dead session id forever.
                Err(McpError::HttpSessionNotFound(detail)) => {
                    tracing::info!(
                        server = %self.name,
                        detail = %detail,
                        "HTTP MCP session not found; reinitializing session and retrying tool call once"
                    );
                    match reinit_http_session(&mut state, &self.next_id).await {
                        Ok(new_instructions) => {
                            *self.instructions.write().await = new_instructions;
                            let retry_id = self.next_id.fetch_add(1, Ordering::SeqCst);
                            let McpClientState::Http {
                                client,
                                url,
                                headers,
                                session_id,
                            } = &*state
                            else {
                                unreachable!()
                            };
                            call_http_tools_call(
                                client,
                                url,
                                headers,
                                session_id.as_ref(),
                                retry_id,
                                name,
                                params,
                                timeout,
                                cancel,
                            )
                            .await
                        }
                        Err(reinit_err) => Err(reinit_err),
                    }
                }
                other => other,
            }?;
            return parse_tool_result(result);
        }

        let McpClientState::Stdio(conn) = &*state else {
            unreachable!()
        };
        // Everything below runs *without* the state lock. Holding it across
        // the response wait is what made a parallel tool batch serialize
        // client-side; the reader task inside `conn` is what makes releasing
        // it safe, because it -- not this caller -- owns the read half.
        let conn = Arc::clone(conn);
        drop(state);

        let response = match conn.register(id) {
            Ok(response) => response,
            Err(err) => {
                conn.mark_unhealthy(&err).await;
                return Err(err);
            }
        };
        if let Err(err) = conn
            .write_request(id, "tools/call", json!({ "name": name, "arguments": args }))
            .await
        {
            conn.deregister(id);
            conn.mark_unhealthy(&err).await;
            return Err(err);
        }

        let response = async {
            response.await.unwrap_or_else(|_| {
                Err(McpError::Io(
                    "mcp reader dropped the response channel".into(),
                ))
            })
        };
        let outcome = match cancel {
            Some(cancel) => {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => ToolCallOutcome::Cancelled,
                    result = tokio::time::timeout(timeout, response) => ToolCallOutcome::from(result),
                }
            }
            None => ToolCallOutcome::from(tokio::time::timeout(timeout, response).await),
        };
        let result = match outcome {
            ToolCallOutcome::Done(result) => result,
            ToolCallOutcome::Cancelled => {
                conn.deregister(id);
                Err(McpError::Cancelled {
                    tool: name.to_string(),
                })
            }
            ToolCallOutcome::TimedOut => {
                conn.deregister(id);
                notify_cancelled_best_effort(conn.write_notification(
                    "notifications/cancelled",
                    cancelled_notification_params(id),
                ))
                .await;
                Err(McpError::Timeout {
                    tool: name.to_string(),
                    timeout,
                })
            }
        };
        let result = match result {
            Ok(result) => result,
            Err(err) => {
                conn.mark_unhealthy(&err).await;
                return Err(err);
            }
        };

        parse_tool_result(result)
    }
}

/// Outcome of racing a `tools/call` future against a timeout and (optionally)
/// a cancellation token.
enum ToolCallOutcome {
    Done(Result<Value, McpError>),
    Cancelled,
    TimedOut,
}

impl From<Result<Result<Value, McpError>, tokio::time::error::Elapsed>> for ToolCallOutcome {
    fn from(result: Result<Result<Value, McpError>, tokio::time::error::Elapsed>) -> Self {
        match result {
            Ok(result) => ToolCallOutcome::Done(result),
            Err(_) => ToolCallOutcome::TimedOut,
        }
    }
}

/// Params for a `notifications/cancelled` notification, per the MCP spec's
/// recommendation to inform the server when a client gives up on a request
/// (here: a `tools/call` that blew through its timeout budget).
fn cancelled_notification_params(id: i64) -> Value {
    json!({ "requestId": id, "reason": "Request timed out" })
}

/// Send a `notifications/cancelled` notification best-effort. Failures are
/// logged, never surfaced -- they must not mask the timeout error that
/// triggered the cancellation.
async fn notify_cancelled_best_effort(
    send: impl std::future::Future<Output = Result<(), McpError>>,
) {
    if let Err(err) = send.await {
        tracing::debug!(
            ?err,
            "failed to send notifications/cancelled after tool-call timeout"
        );
    }
}

fn parse_tool_result(result: Value) -> Result<Value, McpError> {
    if result
        .get("isError")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        let msg = result
            .get("content")
            .and_then(|c| c.get(0))
            .and_then(|m| m.get("text"))
            .and_then(Value::as_str)
            .unwrap_or("Unknown MCP tool error")
            .to_string();
        return Err(McpError::Protocol(msg));
    }
    if let Some(structured) = result.get("structuredContent") {
        return Ok(structured.clone());
    }
    if let Some(text) = result
        .get("content")
        .and_then(|c| c.get(0))
        .and_then(|m| m.get("text"))
    {
        return Ok(text.clone());
    }
    Ok(result)
}

impl McpError {
    fn leaves_client_unhealthy(&self) -> bool {
        matches!(
            self,
            McpError::Io(_) | McpError::Timeout { .. } | McpError::Cancelled { .. }
        )
    }
}

fn parse_server_instructions(initialize_result: &Value) -> Option<String> {
    initialize_result
        .get("instructions")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|instructions| !instructions.is_empty())
        .map(str::to_string)
}

fn build_http_headers(headers: &[McpEnvVar]) -> Result<reqwest::header::HeaderMap, McpError> {
    let mut result = reqwest::header::HeaderMap::new();
    for header in headers {
        let name = reqwest::header::HeaderName::from_bytes(header.name.as_bytes())
            .map_err(|e| McpError::Protocol(format!("invalid HTTP header name: {e}")))?;
        let value = reqwest::header::HeaderValue::from_str(&header.value)
            .map_err(|e| McpError::Protocol(format!("invalid HTTP header value: {e}")))?;
        result.append(name, value);
    }
    Ok(result)
}

fn build_mcp_http_client(url: &str) -> Result<reqwest::Client, McpError> {
    crate::llm_client::OpenAiClient::apply_runtime_tls_workarounds(
        reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()),
        url,
    )
    .build()
    .map_err(|e| McpError::Io(format!("build HTTP client: {e}")))
}

async fn sse_request(
    state: &mut McpClientState,
    id: i64,
    method: &str,
    params: Value,
) -> Result<Value, McpError> {
    let McpClientState::Sse {
        client,
        endpoint,
        headers,
        responses,
        ..
    } = state
    else {
        return Err(McpError::Protocol("not an SSE transport".into()));
    };
    let response = client
        .post(endpoint.as_str())
        .headers(headers.clone())
        .json(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))
        .send()
        .await
        .map_err(|e| McpError::Io(format!("SSE request: {e}")))?;
    if !response.status().is_success() {
        return Err(McpError::Protocol(format!("HTTP {}", response.status())));
    }
    loop {
        let value = responses
            .recv()
            .await
            .ok_or_else(|| McpError::Io("SSE stream closed".into()))?;
        if value.get("id").and_then(Value::as_i64) == Some(id) {
            return parse_jsonrpc_response(value, id);
        }
        tracing::debug!(?value, "skipping SSE message with unexpected id");
    }
}

async fn sse_notification(
    state: &McpClientState,
    method: &str,
    params: Value,
) -> Result<(), McpError> {
    let McpClientState::Sse {
        client,
        endpoint,
        headers,
        ..
    } = state
    else {
        return Err(McpError::Protocol("not an SSE transport".into()));
    };
    let response = client
        .post(endpoint.as_str())
        .headers(headers.clone())
        .json(&json!({ "jsonrpc": "2.0", "method": method, "params": params }))
        .send()
        .await
        .map_err(|e| McpError::Io(format!("SSE notification: {e}")))?;
    if response.status().is_success() {
        Ok(())
    } else {
        Err(McpError::Protocol(format!("HTTP {}", response.status())))
    }
}

async fn read_sse_stream<S>(
    mut stream: S,
    base_url: reqwest::Url,
    endpoint_tx: tokio::sync::oneshot::Sender<Result<String, McpError>>,
    response_tx: tokio::sync::mpsc::UnboundedSender<Value>,
) where
    S: futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    let mut buffer = String::new();
    let mut endpoint_tx = Some(endpoint_tx);
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(e) => {
                if let Some(tx) = endpoint_tx.take() {
                    let _ = tx.send(Err(McpError::Io(format!("read SSE stream: {e}"))));
                }
                break;
            }
        };
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(index) = buffer.find("\n\n").or_else(|| buffer.find("\r\n\r\n")) {
            let separator_len = if buffer[index..].starts_with("\r\n\r\n") {
                4
            } else {
                2
            };
            let event = buffer[..index].to_string();
            buffer.drain(..index + separator_len);
            let mut event_name = "message";
            let mut data = Vec::new();
            for line in event.lines() {
                let line = line.trim_end_matches('\r');
                if let Some(value) = line.strip_prefix("event:") {
                    event_name = value.trim();
                } else if let Some(value) = line.strip_prefix("data:") {
                    data.push(value.trim_start());
                }
            }
            let data = data.join("\n");
            if event_name == "endpoint" {
                if let Some(tx) = endpoint_tx.take() {
                    let endpoint = reqwest::Url::parse(&data)
                        .or_else(|_| base_url.join(&data))
                        .map_err(|e| McpError::Protocol(format!("invalid SSE endpoint: {e}")))
                        .and_then(|url| {
                            let same_origin = url.scheme() == base_url.scheme()
                                && url.host_str() == base_url.host_str()
                                && url.port_or_known_default() == base_url.port_or_known_default();
                            if same_origin {
                                Ok(url.to_string())
                            } else {
                                Err(McpError::Protocol(
                                    "SSE message endpoint must use the configured server origin"
                                        .into(),
                                ))
                            }
                        });
                    let _ = tx.send(endpoint);
                }
            } else if event_name == "message"
                && let Ok(value) = serde_json::from_str(&data)
            {
                let _ = response_tx.send(value);
            }
        }
    }
}

async fn http_request(
    client: &reqwest::Client,
    url: &str,
    headers: &reqwest::header::HeaderMap,
    session_id: Option<&reqwest::header::HeaderValue>,
    id: i64,
    method: &str,
    params: Value,
) -> Result<Value, McpError> {
    Ok(
        http_request_with_session(client, url, headers, session_id, id, method, params)
            .await?
            .0,
    )
}

async fn http_request_with_session(
    client: &reqwest::Client,
    url: &str,
    headers: &reqwest::header::HeaderMap,
    session_id: Option<&reqwest::header::HeaderValue>,
    id: i64,
    method: &str,
    params: Value,
) -> Result<(Value, Option<reqwest::header::HeaderValue>), McpError> {
    let mut request = client.post(url).headers(headers.clone()).header(
        reqwest::header::ACCEPT,
        "application/json, text/event-stream",
    );
    if let Some(session_id) = session_id {
        request = request.header("Mcp-Session-Id", session_id.clone());
    }
    let response = request
        .json(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))
        .send()
        .await
        .map_err(|e| McpError::Io(format!("HTTP request: {e}")))?;
    let status = response.status();
    let response_session_id = response.headers().get("Mcp-Session-Id").cloned();
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let body = response
        .text()
        .await
        .map_err(|e| McpError::Io(format!("read HTTP response: {e}")))?;
    if !status.is_success() {
        let message = format!("HTTP {status}: {body}");
        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(McpError::HttpSessionNotFound(message));
        }
        return Err(McpError::Protocol(message));
    }
    let value = if content_type.starts_with("text/event-stream") {
        parse_sse_json(&body)?
    } else {
        serde_json::from_str(&body)
            .map_err(|e| McpError::Protocol(format!("parse HTTP response: {e}")))?
    };
    Ok((parse_jsonrpc_response(value, id)?, response_session_id))
}

async fn http_notification(
    client: &reqwest::Client,
    url: &str,
    headers: &reqwest::header::HeaderMap,
    session_id: Option<&reqwest::header::HeaderValue>,
    method: &str,
    params: Value,
) -> Result<(), McpError> {
    let mut request = client.post(url).headers(headers.clone()).header(
        reqwest::header::ACCEPT,
        "application/json, text/event-stream",
    );
    if let Some(session_id) = session_id {
        request = request.header("Mcp-Session-Id", session_id.clone());
    }
    let response = request
        .json(&json!({ "jsonrpc": "2.0", "method": method, "params": params }))
        .send()
        .await
        .map_err(|e| McpError::Io(format!("HTTP notification: {e}")))?;
    if response.status().is_success() {
        Ok(())
    } else {
        Err(McpError::Protocol(format!("HTTP {}", response.status())))
    }
}

/// Issue one `tools/call` over an established HTTP MCP session, racing it
/// against `timeout` and (if given) `cancel`, and best-effort notifying the
/// server via `notifications/cancelled` if the call is abandoned to a
/// timeout. Mirrors the SSE/stdio call paths' cancellation handling; kept as
/// a free function so `call_tool_with_timeout`'s HTTP branch can invoke it
/// twice (initial attempt, then once more after `reinit_http_session`)
/// without holding two conflicting borrows of the client's session state.
#[allow(clippy::too_many_arguments)]
async fn call_http_tools_call(
    client: &reqwest::Client,
    url: &str,
    headers: &reqwest::header::HeaderMap,
    session_id: Option<&reqwest::header::HeaderValue>,
    id: i64,
    name: &str,
    params: Value,
    timeout: Duration,
    cancel: Option<&CancellationToken>,
) -> Result<Value, McpError> {
    let call = http_request(client, url, headers, session_id, id, "tools/call", params);
    let outcome = match cancel {
        Some(cancel) => tokio::select! {
            biased;
            _ = cancel.cancelled() => ToolCallOutcome::Cancelled,
            result = tokio::time::timeout(timeout, call) => ToolCallOutcome::from(result),
        },
        None => ToolCallOutcome::from(tokio::time::timeout(timeout, call).await),
    };
    match outcome {
        ToolCallOutcome::Done(result) => result,
        ToolCallOutcome::Cancelled => Err(McpError::Cancelled {
            tool: name.to_string(),
        }),
        ToolCallOutcome::TimedOut => {
            notify_cancelled_best_effort(http_notification(
                client,
                url,
                headers,
                session_id,
                "notifications/cancelled",
                cancelled_notification_params(id),
            ))
            .await;
            Err(McpError::Timeout {
                tool: name.to_string(),
                timeout,
            })
        }
    }
}

/// Re-run the initialize handshake on an already-connected HTTP MCP
/// transport and install the fresh session id it returns, without
/// rebuilding the `reqwest::Client` or re-fetching `tools/list` (the tool
/// set does not change across a same-server reinitialization; only the
/// server-side session state does, e.g. after rmcp's `LocalSessionManager`
/// idle-timeout eviction). Called from `call_tool_with_timeout` only after
/// an HTTP 404 ("session not found") on a `tools/call`.
async fn reinit_http_session(
    state: &mut McpClientState,
    next_id: &AtomicI64,
) -> Result<Option<String>, McpError> {
    let McpClientState::Http {
        client,
        url,
        headers,
        session_id,
    } = state
    else {
        unreachable!()
    };
    let init_id = next_id.fetch_add(1, Ordering::SeqCst);
    let (init, new_session_id) = http_request_with_session(
        client,
        url,
        headers,
        None,
        init_id,
        "initialize",
        json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": { "name": "brokk-acp-rust", "version": env!("CARGO_PKG_VERSION") },
        }),
    )
    .await?;
    let instructions = parse_server_instructions(&init);
    http_notification(
        client,
        url,
        headers,
        new_session_id.as_ref(),
        "notifications/initialized",
        json!({}),
    )
    .await?;
    *session_id = new_session_id;
    Ok(instructions)
}

fn parse_sse_json(body: &str) -> Result<Value, McpError> {
    let data = body
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("\n");
    serde_json::from_str(&data).map_err(|e| McpError::Protocol(format!("parse SSE response: {e}")))
}

fn parse_jsonrpc_response(value: Value, expected_id: i64) -> Result<Value, McpError> {
    if value.get("id").and_then(Value::as_i64) != Some(expected_id) {
        return Err(McpError::Protocol("HTTP response has unexpected id".into()));
    }
    jsonrpc_result(value)
}

/// Split an already-id-matched JSON-RPC response object into its `result` or
/// its `error`.
fn jsonrpc_result(value: Value) -> Result<Value, McpError> {
    if let Some(error) = value.get("error") {
        return Err(McpError::JsonRpc {
            code: error.get("code").and_then(Value::as_i64).unwrap_or(0),
            message: error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        });
    }
    value
        .get("result")
        .cloned()
        .ok_or_else(|| McpError::Protocol("response missing result".into()))
}

#[cfg(test)]
mod http_tests {
    use super::*;
    use std::sync::{Arc, Mutex as StdMutex};

    #[tokio::test]
    async fn sse_transport_discovers_endpoint_and_calls_tools() {
        let server = tiny_http::Server::http("127.0.0.1:0").expect("start server");
        let base = format!("http://{}", server.server_addr());
        let sse_url = format!("{base}/sse");
        let endpoint = format!("{base}/messages");
        let thread = std::thread::spawn(move || {
            let request = server.recv().expect("SSE request");
            assert_eq!(request.method(), &tiny_http::Method::Get);
            let events = format!(
                "event: endpoint\ndata: {endpoint}\n\n\
                 event: message\ndata: {{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{{}}}}\n\n\
                 event: message\ndata: {{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{{\"tools\":[{{\"name\":\"echo\",\"description\":\"echo\",\"inputSchema\":{{\"type\":\"object\"}}}}]}}}}\n\n\
                 event: message\ndata: {{\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{{\"structuredContent\":{{\"ok\":true}}}}}}\n\n"
            );
            request
                .respond(tiny_http::Response::from_string(events).with_header(
                    tiny_http::Header::from_bytes("Content-Type", "text/event-stream").unwrap(),
                ))
                .expect("respond SSE");
            for _ in 0..4 {
                let mut request = server.recv().expect("message request");
                assert_eq!(request.method(), &tiny_http::Method::Post);
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).expect("body");
                serde_json::from_str::<Value>(&body).expect("JSON-RPC body");
                request
                    .respond(tiny_http::Response::empty(202))
                    .expect("respond message");
            }
        });
        let config = McpServerConfig {
            name: "events".into(),
            transport: McpTransport::Sse,
            command: String::new(),
            url: Some(sse_url),
            headers: vec![],
            args: vec![],
            env: vec![],
            framing: McpFraming::ContentLength,
            enabled: true,
        };
        let client = McpClient::spawn(&config, Path::new("."))
            .await
            .expect("connect");
        assert_eq!(client.tools()[0].name, "echo");
        assert_eq!(
            client.call_tool("echo", json!({})).await.unwrap(),
            json!({"ok": true})
        );
        thread.join().unwrap();
    }

    #[tokio::test]
    async fn http_transport_initializes_lists_and_calls_tools() {
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let seen = requests.clone();
        let server = tiny_http::Server::http("127.0.0.1:0").expect("start server");
        let url = format!("http://{}/mcp", server.server_addr());
        let thread = std::thread::spawn(move || {
            for _ in 0..4 {
                let mut request = server.recv().expect("request");
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).expect("body");
                let value: Value = serde_json::from_str(&body).expect("json");
                seen.lock().unwrap().push(value.clone());
                let response = match value.get("method").and_then(Value::as_str) {
                    Some("initialize") => json!({
                        "jsonrpc":"2.0",
                        "id":value["id"],
                        "result":{"instructions":"Follow remote coordination policy."}
                    }),
                    Some("tools/list") => {
                        json!({"jsonrpc":"2.0","id":value["id"],"result":{"tools":[{"name":"echo","description":"echo","inputSchema":{"type":"object"}}]}})
                    }
                    Some("tools/call") => {
                        json!({"jsonrpc":"2.0","id":value["id"],"result":{"structuredContent":{"ok":true}}})
                    }
                    _ => {
                        request
                            .respond(tiny_http::Response::empty(202))
                            .expect("respond");
                        continue;
                    }
                };
                request
                    .respond(
                        tiny_http::Response::from_string(response.to_string()).with_header(
                            tiny_http::Header::from_bytes("Content-Type", "application/json")
                                .unwrap(),
                        ),
                    )
                    .expect("respond");
            }
        });
        let config = McpServerConfig {
            name: "remote".into(),
            transport: McpTransport::Http,
            command: String::new(),
            url: Some(url),
            headers: vec![McpEnvVar {
                name: "X-Test".into(),
                value: "yes".into(),
            }],
            args: vec![],
            env: vec![],
            framing: McpFraming::ContentLength,
            enabled: true,
        };
        let client = McpClient::spawn(&config, Path::new("."))
            .await
            .expect("connect");
        assert_eq!(
            client.instructions().await.as_deref(),
            Some("Follow remote coordination policy.")
        );
        assert_eq!(client.tools()[0].name, "echo");
        assert_eq!(
            client.call_tool("echo", json!({})).await.unwrap(),
            json!({"ok":true})
        );
        thread.join().unwrap();
        assert_eq!(requests.lock().unwrap().len(), 4);
    }

    /// A `tools/call` that blows through its (parameterized, short-for-test)
    /// timeout budget should get a best-effort `notifications/cancelled`
    /// over the same HTTP transport, per the MCP spec's SHOULD.
    ///
    /// The fake server dispatches by JSON-RPC `method` (like
    /// `http_transport_initializes_lists_and_calls_tools` above) rather than
    /// by request position, because `connect_http` also fires a
    /// `notifications/initialized` notification between `initialize` and
    /// `tools/list` that a purely positional server would misattribute.
    /// `tools/call` is answered, but only well after the client's short test
    /// timeout has already elapsed and it's moved on to sending the
    /// cancellation notification: this keeps that connection's
    /// request/response cycle intact instead of leaving a reused (or
    /// pipelined) HTTP/1.1 connection resolving a response nobody's
    /// listening for, which otherwise made the *next* request on that
    /// connection (the cancellation notification, or even the next test's
    /// requests) resolve against a stray, previously-completed response.
    #[tokio::test]
    async fn http_tool_call_timeout_sends_cancelled_notification() {
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let seen = requests.clone();
        let server = tiny_http::Server::http("127.0.0.1:0").expect("start server");
        let url = format!("http://{}/mcp", server.server_addr());
        let thread = std::thread::spawn(move || {
            loop {
                let mut request = server.recv().expect("request");
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).expect("body");
                let value: Value = serde_json::from_str(&body).expect("json");
                seen.lock().unwrap().push(value.clone());
                match value.get("method").and_then(Value::as_str) {
                    Some("initialize") => {
                        request
                            .respond(tiny_http::Response::from_string(
                                json!({"jsonrpc":"2.0","id":value["id"],"result":{}}).to_string(),
                            ))
                            .expect("respond initialize");
                    }
                    Some("notifications/initialized") => {
                        request
                            .respond(tiny_http::Response::empty(202))
                            .expect("respond notifications/initialized");
                    }
                    Some("tools/list") => {
                        request
                            .respond(tiny_http::Response::from_string(
                                json!({
                                    "jsonrpc":"2.0",
                                    "id":value["id"],
                                    "result":{"tools":[{"name":"fake_tool","description":"fake","inputSchema":{"type":"object"}}]}
                                })
                                .to_string(),
                            ))
                            .expect("respond tools/list");
                    }
                    Some("tools/call") => {
                        // Stall well past the client's timeout before
                        // responding -- see the doc comment above for why.
                        std::thread::sleep(Duration::from_millis(400));
                        request
                            .respond(tiny_http::Response::from_string(
                                json!({"jsonrpc":"2.0","id":value["id"],"result":{"structuredContent":{"ok":true}}}).to_string(),
                            ))
                            .expect("respond tools/call (late, past the client's timeout)");
                    }
                    Some("notifications/cancelled") => {
                        request
                            .respond(tiny_http::Response::empty(202))
                            .expect("respond notifications/cancelled");
                        break;
                    }
                    other => panic!("unexpected method in fake HTTP server: {other:?}"),
                }
            }
        });
        let config = McpServerConfig {
            name: "remote".into(),
            transport: McpTransport::Http,
            command: String::new(),
            url: Some(url),
            headers: vec![],
            args: vec![],
            env: vec![],
            framing: McpFraming::ContentLength,
            enabled: true,
        };
        let client = McpClient::spawn(&config, Path::new("."))
            .await
            .expect("connect");

        let err = client
            .call_tool_with_timeout("fake_tool", json!({}), Duration::from_millis(200), None)
            .await
            .expect_err("tools/call should time out");
        assert!(
            matches!(err, McpError::Timeout { .. }),
            "expected timeout, got {err}"
        );

        thread.join().unwrap();
        let seen = requests.lock().unwrap();
        let call_index = seen
            .iter()
            .position(|msg| msg["method"] == "tools/call")
            .expect("tools/call should have been sent");
        let cancel_index = seen
            .iter()
            .position(|msg| msg["method"] == "notifications/cancelled")
            .expect("notifications/cancelled should have been sent");
        assert!(
            cancel_index > call_index,
            "cancellation notification should follow the timed-out tools/call"
        );
        assert_eq!(
            seen[cancel_index]["params"]["reason"].as_str(),
            Some("Request timed out")
        );
        assert_eq!(
            seen[cancel_index]["params"]["requestId"].as_i64(),
            seen[call_index]["id"].as_i64()
        );
        assert!(
            seen[cancel_index].get("id").is_none(),
            "notifications must not carry an id"
        );
    }

    /// If the server responds 404 to a `tools/call` -- as rmcp's
    /// `LocalSessionManager` does once it has evicted a session past its
    /// idle timeout (see `code_agent.rs`'s `HttpServer::start`, which uses
    /// that default) -- the client should reinitialize the HTTP MCP session
    /// and retry the call once, rather than surfacing the dead session as a
    /// permanent tool-call failure for the rest of the client's lifetime.
    /// See `reinit_http_session` and the HTTP branch of
    /// `call_tool_with_timeout`.
    #[tokio::test]
    async fn http_session_not_found_reinitializes_and_retries_tool_call() {
        let requests = Arc::new(StdMutex::new(Vec::new()));
        let seen = requests.clone();
        let server = tiny_http::Server::http("127.0.0.1:0").expect("start server");
        let url = format!("http://{}/mcp", server.server_addr());
        let thread = std::thread::spawn(move || {
            let mut initializes = 0;
            let mut calls = 0;
            loop {
                let mut request = server.recv().expect("request");
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).expect("body");
                let value: Value = serde_json::from_str(&body).expect("json");
                seen.lock().unwrap().push(value.clone());
                match value.get("method").and_then(Value::as_str) {
                    Some("initialize") => {
                        initializes += 1;
                        let session_header = format!("session-{initializes}");
                        request
                            .respond(
                                tiny_http::Response::from_string(
                                    json!({"jsonrpc":"2.0","id":value["id"],"result":{}})
                                        .to_string(),
                                )
                                .with_header(
                                    tiny_http::Header::from_bytes(
                                        "Mcp-Session-Id",
                                        session_header.as_bytes(),
                                    )
                                    .unwrap(),
                                ),
                            )
                            .expect("respond initialize");
                    }
                    Some("notifications/initialized") => {
                        request
                            .respond(tiny_http::Response::empty(202))
                            .expect("respond notifications/initialized");
                    }
                    Some("tools/list") => {
                        request
                            .respond(tiny_http::Response::from_string(
                                json!({
                                    "jsonrpc":"2.0",
                                    "id":value["id"],
                                    "result":{"tools":[{"name":"fake_tool","description":"fake","inputSchema":{"type":"object"}}]}
                                })
                                .to_string(),
                            ))
                            .expect("respond tools/list");
                    }
                    Some("tools/call") => {
                        calls += 1;
                        if calls == 1 {
                            request
                                .respond(
                                    tiny_http::Response::from_string(
                                        "Not Found: Session not found",
                                    )
                                    .with_status_code(404),
                                )
                                .expect("respond tools/call (session not found)");
                        } else {
                            request
                                .respond(tiny_http::Response::from_string(
                                    json!({"jsonrpc":"2.0","id":value["id"],"result":{"structuredContent":{"ok":true}}}).to_string(),
                                ))
                                .expect("respond tools/call (retry succeeds)");
                            break;
                        }
                    }
                    other => panic!("unexpected method in fake HTTP server: {other:?}"),
                }
            }
        });
        let config = McpServerConfig {
            name: "remote".into(),
            transport: McpTransport::Http,
            command: String::new(),
            url: Some(url),
            headers: vec![],
            args: vec![],
            env: vec![],
            framing: McpFraming::ContentLength,
            enabled: true,
        };
        let client = McpClient::spawn(&config, Path::new("."))
            .await
            .expect("connect");

        let result = client
            .call_tool("fake_tool", json!({}))
            .await
            .expect("the client should transparently reinitialize and retry once");
        assert_eq!(result, json!({"ok": true}));

        thread.join().unwrap();
        let methods = requests
            .lock()
            .unwrap()
            .iter()
            .map(|msg| msg["method"].as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            methods,
            vec![
                "initialize",
                "notifications/initialized",
                "tools/list",
                "tools/call",
                "initialize",
                "notifications/initialized",
                "tools/call",
            ],
            "expected: connect handshake, a 404'd tools/call, a reinitialize \
             handshake, then the retried tools/call"
        );
    }
}

async fn write_request(
    io: &mut StdioWriter,
    id: i64,
    method: &str,
    params: Value,
) -> Result<(), McpError> {
    write_message(
        io,
        &json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }),
    )
    .await
}

async fn write_notification(
    io: &mut StdioWriter,
    method: &str,
    params: Value,
) -> Result<(), McpError> {
    write_message(
        io,
        &json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }),
    )
    .await
}

async fn write_message(io: &mut StdioWriter, msg: &Value) -> Result<(), McpError> {
    let bytes = serde_json::to_vec(msg).map_err(|e| McpError::Io(format!("serialize: {e}")))?;
    match io.framing {
        McpFraming::ContentLength => {
            let header = format!("Content-Length: {}\r\n\r\n", bytes.len());
            io.writer
                .write_all(header.as_bytes())
                .await
                .map_err(|e| McpError::Io(format!("write header: {e}")))?;
            io.writer
                .write_all(&bytes)
                .await
                .map_err(|e| McpError::Io(format!("write body: {e}")))?;
        }
        McpFraming::Line => {
            io.writer
                .write_all(&bytes)
                .await
                .map_err(|e| McpError::Io(format!("write body: {e}")))?;
            io.writer
                .write_all(b"\n")
                .await
                .map_err(|e| McpError::Io(format!("write newline: {e}")))?;
        }
    }
    io.writer
        .flush()
        .await
        .map_err(|e| McpError::Io(format!("flush: {e}")))?;
    Ok(())
}

/// Read the response to `expected_id` straight off the stream. Only valid
/// before a [`StdioConn`] takes ownership of the reader -- that is, during the
/// initialize/`tools/list` handshake, which has no concurrency to demultiplex.
async fn read_response(
    reader: &mut BufReader<ChildStdout>,
    framing: McpFraming,
    expected_id: i64,
) -> Result<Value, McpError> {
    loop {
        let value = read_message(reader, framing).await?;
        if value.get("id").and_then(Value::as_i64) != Some(expected_id) {
            tracing::debug!(?value, "skipping mcp message with unexpected id");
            continue;
        }
        return jsonrpc_result(value);
    }
}

async fn read_message(
    reader: &mut BufReader<ChildStdout>,
    framing: McpFraming,
) -> Result<Value, McpError> {
    if framing == McpFraming::Line {
        return read_line_message(reader).await;
    }

    let mut content_length = None;
    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .await
            .map_err(|e| McpError::Io(format!("read header: {e}")))?;
        if n == 0 {
            return Err(McpError::Io("mcp server closed stdout".into()));
        }
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            break;
        }
        let Some((name, value)) = line.split_once(':') else {
            return Err(McpError::Protocol(format!("malformed MCP header: {line}")));
        };
        if name.eq_ignore_ascii_case("content-length") {
            let len = value.trim().parse::<usize>().map_err(|e| {
                McpError::Protocol(format!("invalid Content-Length `{}`: {e}", value.trim()))
            })?;
            content_length = Some(len);
        }
    }

    let len =
        content_length.ok_or_else(|| McpError::Protocol("missing Content-Length header".into()))?;
    let mut body = vec![0; len];
    reader
        .read_exact(&mut body)
        .await
        .map_err(|e| McpError::Io(format!("read body: {e}")))?;
    serde_json::from_slice(&body).map_err(|e| McpError::Protocol(format!("parse body: {e}")))
}

async fn read_line_message(reader: &mut BufReader<ChildStdout>) -> Result<Value, McpError> {
    let mut line = String::new();
    let n = reader
        .read_line(&mut line)
        .await
        .map_err(|e| McpError::Io(format!("read line: {e}")))?;
    if n == 0 {
        return Err(McpError::Io("mcp server closed stdout".into()));
    }
    let trimmed = line.trim();
    serde_json::from_str(trimmed)
        .map_err(|e| McpError::Protocol(format!("parse line: {e} (line: {trimmed})")))
}

fn parse_tool_list(result: Value, server: Option<&str>) -> Result<Vec<McpToolDef>, McpError> {
    let tools_array = result
        .get("tools")
        .and_then(Value::as_array)
        .ok_or_else(|| McpError::Protocol("tools/list missing 'tools' array".into()))?;
    let mut tools = Vec::new();
    for tool in tools_array {
        let name = tool
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| McpError::Protocol("tool missing name".into()))?
            .to_string();
        let description = tool
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let input_schema = tool
            .get("inputSchema")
            .cloned()
            .unwrap_or_else(|| json!({ "type": "object" }));
        let input_schema = match normalize_tool_input_schema(input_schema) {
            Ok(input_schema) => input_schema,
            Err(reason) => {
                tracing::warn!(
                    server,
                    tool = %name,
                    reason = %reason,
                    "skipping mcp tool with invalid input schema"
                );
                continue;
            }
        };
        let annotations = parse_tool_annotations(tool.get("annotations"));
        tools.push(McpToolDef {
            name,
            description,
            input_schema,
            annotations,
        });
    }
    Ok(tools)
}

fn normalize_tool_input_schema(mut schema: Value) -> Result<Value, String> {
    let Some(object) = schema.as_object_mut() else {
        return Err("inputSchema must be a JSON object".to_string());
    };

    match object.get("type") {
        Some(Value::String(schema_type)) if schema_type == "object" => {}
        Some(_) => return Err("inputSchema top-level type must be \"object\"".to_string()),
        None => {
            object.insert("type".to_string(), Value::String("object".to_string()));
        }
    }

    match object.get("properties") {
        Some(Value::Object(_)) => {}
        Some(Value::Null) | None => {
            object.insert("properties".to_string(), json!({}));
        }
        Some(_) => return Err("inputSchema properties must be an object".to_string()),
    }

    let property_names: HashSet<String> = object
        .get("properties")
        .and_then(Value::as_object)
        .expect("properties was normalized to an object")
        .keys()
        .cloned()
        .collect();

    let Some(required) = object.get_mut("required") else {
        return Ok(schema);
    };
    let Some(required_array) = required.as_array() else {
        object.remove("required");
        tracing::warn!(
            dropped_required = "non-array required",
            "normalized mcp tool input schema required field"
        );
        return Ok(schema);
    };

    let mut seen = HashSet::new();
    let mut normalized = Vec::new();
    let mut dropped = Vec::new();
    for entry in required_array {
        let Some(name) = entry.as_str() else {
            dropped.push(entry.to_string());
            continue;
        };
        if name.is_empty() {
            dropped.push(name.to_string());
            continue;
        }
        if !property_names.contains(name) {
            dropped.push(name.to_string());
            continue;
        }
        if !seen.insert(name.to_string()) {
            dropped.push(name.to_string());
            continue;
        }
        normalized.push(Value::String(name.to_string()));
    }

    if !dropped.is_empty() {
        *required = Value::Array(normalized);
        tracing::warn!(
            dropped_required = ?dropped,
            "normalized mcp tool input schema required field"
        );
    }
    Ok(schema)
}

fn parse_tool_annotations(value: Option<&Value>) -> McpToolAnnotations {
    let Some(annotations) = value.and_then(Value::as_object) else {
        return McpToolAnnotations::default();
    };
    McpToolAnnotations {
        read_only_hint: annotations.get("readOnlyHint").and_then(Value::as_bool),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Pins the interim-fix timeout budgets: `tools/call` gets a generous
    /// 300s (long-running server-side tools like Mjolnir's `code_agent` can
    /// run up to 240s), while setup RPCs (initialize, `tools/list`, SSE
    /// endpoint discovery) keep the original 60s. See
    /// https://github.com/BrokkAi/anvil/issues/292 for the fuller design.
    #[test]
    fn startup_and_tool_call_timeouts_have_expected_budgets() {
        assert_eq!(MCP_STARTUP_TIMEOUT, Duration::from_secs(60));
        assert_eq!(MCP_TOOL_CALL_TIMEOUT, Duration::from_secs(300));
        assert!(MCP_TOOL_CALL_TIMEOUT > MCP_STARTUP_TIMEOUT);
    }

    /// Callers/tests match on the "timed out after {n}s" message, so it must
    /// survive the timeout value being parameterized per call site.
    #[test]
    fn timeout_error_message_format_is_stable_across_timeout_values() {
        let startup_err = McpError::Timeout {
            tool: "SSE endpoint discovery".to_string(),
            timeout: MCP_STARTUP_TIMEOUT,
        };
        assert_eq!(
            startup_err.to_string(),
            "tool 'SSE endpoint discovery' timed out after 60s"
        );

        let tool_call_err = McpError::Timeout {
            tool: "fake_tool".to_string(),
            timeout: MCP_TOOL_CALL_TIMEOUT,
        };
        assert_eq!(
            tool_call_err.to_string(),
            "tool 'fake_tool' timed out after 300s"
        );
    }

    #[test]
    fn server_instructions_omit_absent_empty_and_whitespace_values() {
        assert_eq!(parse_server_instructions(&json!({})), None);
        assert_eq!(
            parse_server_instructions(&json!({"instructions": null})),
            None
        );
        assert_eq!(
            parse_server_instructions(&json!({"instructions": "  \n"})),
            None
        );
        assert_eq!(
            parse_server_instructions(&json!({"instructions": "  coordinate carefully  "})),
            Some("coordinate carefully".to_string())
        );
    }

    /// Persisted managed defaults follow the current default across both the
    /// deprecated flag migration and past toolset changes.
    #[test]
    fn legacy_default_bifrost_args_follow_current_default() {
        for legacy_args in legacy_default_bifrost_arg_sets() {
            let mut server = McpServerConfig {
                name: "bifrost".to_string(),
                transport: crate::mcp::McpTransport::Stdio,
                url: None,
                headers: Vec::new(),
                command: "bifrost".to_string(),
                args: legacy_args,
                env: Vec::new(),
                framing: McpFraming::Line,
                enabled: false,
            };
            normalize_preinstalled_bifrost_server(&mut server);
            assert_eq!(server.args, default_bifrost_args());
            assert!(!server.enabled, "the stored enabled flag must be preserved");
        }
    }

    /// A user-customized bifrost surface (neither the current nor a prior
    /// managed default) is left untouched.
    #[test]
    fn customized_bifrost_args_are_not_upgraded() {
        let custom = vec![
            "--root".to_string(),
            "{cwd}".to_string(),
            "--server".to_string(),
            "symbol".to_string(),
            "--no-line-numbers".to_string(),
        ];
        let mut server = McpServerConfig {
            name: "bifrost".to_string(),
            transport: crate::mcp::McpTransport::Stdio,
            url: None,
            headers: Vec::new(),
            command: "bifrost".to_string(),
            args: custom.clone(),
            env: Vec::new(),
            framing: McpFraming::Line,
            enabled: true,
        };
        normalize_preinstalled_bifrost_server(&mut server);
        assert_eq!(server.args, custom, "a custom surface must be left as-is");
    }

    #[test]
    fn tool_schema_missing_type_and_properties_are_inserted() {
        let tools = parse_tool_list(
            json!({
                "tools": [{
                    "name": "needs_defaults",
                    "inputSchema": {}
                }]
            }),
            Some("test-server"),
        )
        .expect("tools/list should parse");

        assert_eq!(tools.len(), 1);
        assert_eq!(
            tools[0].input_schema,
            json!({
                "type": "object",
                "properties": {}
            })
        );
    }

    #[test]
    fn tool_schema_with_non_object_top_level_type_is_skipped() {
        let tools = parse_tool_list(
            json!({
                "tools": [
                    {
                        "name": "bad",
                        "inputSchema": {
                            "type": "string"
                        }
                    },
                    {
                        "name": "good",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": { "type": "string" }
                            }
                        }
                    }
                ]
            }),
            Some("test-server"),
        )
        .expect("tools/list should parse");

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "good");
    }

    #[test]
    fn tool_schema_required_entries_are_cleaned() {
        let tools = parse_tool_list(
            json!({
                "tools": [{
                    "name": "clean_required",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": { "type": "string" },
                            "limit": { "type": "number" }
                        },
                        "required": ["path", "path", 7, "missing", "", "limit"]
                    }
                }]
            }),
            Some("test-server"),
        )
        .expect("tools/list should parse");

        assert_eq!(tools[0].input_schema["required"], json!(["path", "limit"]));
    }

    #[test]
    fn tool_schema_required_non_array_is_removed() {
        let tools = parse_tool_list(
            json!({
                "tools": [{
                    "name": "bad_required",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": { "type": "string" }
                        },
                        "required": "path"
                    }
                }]
            }),
            Some("test-server"),
        )
        .expect("tools/list should parse");

        assert!(tools[0].input_schema.get("required").is_none());
    }

    #[test]
    fn tool_schema_json_string_is_skipped() {
        let tools = parse_tool_list(
            json!({
                "tools": [{
                    "name": "bad",
                    "inputSchema": "not a schema"
                }]
            }),
            Some("test-server"),
        )
        .expect("tools/list should parse");

        assert!(tools.is_empty());
    }

    #[test]
    fn absent_input_schema_uses_normalized_object_default() {
        let tools = parse_tool_list(
            json!({
                "tools": [{
                    "name": "defaulted"
                }]
            }),
            Some("test-server"),
        )
        .expect("tools/list should parse");

        assert_eq!(
            tools[0].input_schema,
            json!({
                "type": "object",
                "properties": {}
            })
        );
    }

    /// Resolve the bifrost binary used by the handshake test.
    ///
    /// Resolution order:
    /// 1. `BROKK_BIFROST_BINARY` env var (override for testing against an
    ///    in-tree bifrost build).
    /// 2. The cached pinned-version binary under
    ///    `target/test-fixtures/bifrost/<version>/<triple>/`.
    /// 3. Download the pinned release into the cache, then return its path.
    ///
    /// We deliberately do NOT consult `which bifrost`: that coupled the test
    /// to whatever happened to be installed locally, which dragged the test
    /// behavior into "depends on which version of bifrost the contributor
    /// happens to have on PATH" -- the bug this helper exists to remove.
    async fn ensure_test_bifrost_binary() -> PathBuf {
        if let Ok(override_path) = std::env::var("BROKK_BIFROST_BINARY") {
            let p = PathBuf::from(&override_path);
            assert!(
                p.is_file(),
                "BROKK_BIFROST_BINARY={override_path} is not a regular file"
            );
            return p;
        }

        let cache_dir = test_fixture_cache_dir();
        let binary = cache_dir.join(BIFROST_BINARY_NAME);
        if binary.is_file() {
            return binary;
        }

        download_and_extract_bifrost(&cache_dir)
            .await
            .expect("download bifrost test fixture");
        assert!(
            binary.is_file(),
            "expected bifrost at {binary:?} after download+extract"
        );
        binary
    }

    fn test_fixture_cache_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-fixtures")
            .join("bifrost")
            .join(BUNDLED_BIFROST_VERSION)
            .join(BIFROST_TARGET_TRIPLE)
    }

    /// Smoke test: spawn the real bifrost subprocess (pinned release,
    /// downloaded into `target/test-fixtures/`), run the MCP handshake,
    /// confirm a stable subset of the default tools is exposed, and round-trip
    /// two distinct tool calls. We deliberately do NOT pin the exact tool
    /// count or full tool list -- bifrost adds tools faster than this test
    /// gets updated, and the handshake's job is to verify the protocol
    /// path works, not to enumerate the surface.
    #[tokio::test]
    async fn handshake_and_call_default_tools() {
        let binary = ensure_test_bifrost_binary().await;
        let cwd = std::env::current_dir()
            .expect("cwd")
            .canonicalize()
            .expect("canonicalize");

        let config = McpServerConfig {
            name: "bifrost".to_string(),
            transport: crate::mcp::McpTransport::Stdio,
            url: None,
            headers: Vec::new(),
            command: binary.display().to_string(),
            args: McpServerConfig::bifrost().args,
            env: Vec::new(),
            framing: McpFraming::Line,
            enabled: true,
        };
        let client = McpClient::spawn(&config, &cwd)
            .await
            .expect("bifrost subprocess should start");

        let names: Vec<&str> = client.tools().iter().map(|t| t.name.as_str()).collect();

        // Floor on total tool count -- catches a wholesale regression where
        // bifrost drops most of its tools (e.g. a misconfigured server arg).
        // The exact count drifts as bifrost adds tools, so we don't pin it.
        assert!(
            client.tools().len() >= 5,
            "expected at least 5 tools, got {} -- {names:?}",
            client.tools().len()
        );

        for expected in [
            "search_symbols",
            "get_symbol_sources",
            "get_summaries",
            "usage_graph",
            "activate_workspace",
            "get_active_workspace",
        ] {
            assert!(
                names.contains(&expected),
                "missing tool {expected} in {names:?}"
            );
        }
        assert!(
            names.contains(&"scan_usages_by_reference") || names.contains(&"scan_usages"),
            "missing reference usage-scanning tool in {names:?}"
        );

        // Anti-drift: every tool bifrost advertises must have a row in
        // `tools::TOOLS`. Without one, `tool_kind` falls back to
        // `Other` (refused in `readOnly`, prompts unnecessarily in
        // `default`) and `display_name` falls back to "Executing
        // tool" in the UI. If this assertion fires, bifrost likely
        // added or renamed a tool -- update `TOOLS` in
        // `tools/mod.rs` to match.
        for tool in client.tools() {
            if tool.name == "list_symbols" {
                continue;
            }
            assert!(
                crate::tools::is_known_tool(&tool.name),
                "bifrost advertises '{}' but it is not in the TOOLS metadata table; \
                 add a ToolMeta row in tools/mod.rs (current bifrost surface: {names:?})",
                tool.name
            );

            let kind = crate::tools::ToolRegistry::tool_kind(&tool.name);
            if matches!(
                kind,
                agent_client_protocol::schema::v1::ToolKind::Read
                    | agent_client_protocol::schema::v1::ToolKind::Search
                    | agent_client_protocol::schema::v1::ToolKind::Fetch
            ) {
                assert_eq!(
                    tool.annotations.read_only_hint,
                    Some(true),
                    "bifrost advertises '{}' as {kind:?} in Anvil, but MCP readOnlyHint is {:?}",
                    tool.name,
                    tool.annotations.read_only_hint
                );
            }
        }

        // Round-trip two distinct tool calls so we exercise back-to-back use
        // of the demultiplexed JSON-RPC transport (id correlation, waiter
        // registration, response-shape branching) -- not just one-shot
        // dispatch.
        let result = client
            .call_tool("search_symbols", json!({ "patterns": ["McpClient"] }))
            .await
            .expect("search_symbols call should succeed");
        eprintln!(
            "search_symbols result: {}",
            serde_json::to_string_pretty(&result).unwrap_or_default()
        );

        let result = client
            .call_tool(
                "get_summaries",
                json!({ "targets": ["brokk-acp-rust/src/mcp.rs"] }),
            )
            .await
            .expect("get_summaries call should succeed");
        eprintln!(
            "get_summaries result: {}",
            serde_json::to_string_pretty(&result).unwrap_or_default()
        );
    }

    /// SlopCop opts into its dedicated Bifrost surface. Advertisement and
    /// permission classification are independent: the MCP surface supplies
    /// the definitions, while Anvil's global metadata makes the reporters
    /// callable in read-only sessions.
    #[tokio::test]
    async fn slopcop_surface_is_read_only_permission_compatible() {
        let binary = ensure_test_bifrost_binary().await;
        let cwd = std::env::current_dir()
            .expect("cwd")
            .canonicalize()
            .expect("canonicalize");
        let config = McpServerConfig {
            name: "bifrost".to_string(),
            transport: crate::mcp::McpTransport::Stdio,
            url: None,
            headers: Vec::new(),
            command: binary.display().to_string(),
            args: bifrost_args("--mcp", "slopcop"),
            env: Vec::new(),
            framing: McpFraming::Line,
            enabled: true,
        };
        let client = McpClient::spawn(&config, &cwd)
            .await
            .expect("SlopCop Bifrost surface should start");
        let names: Vec<&str> = client
            .tools()
            .iter()
            .map(|tool| tool.name.as_str())
            .collect();

        for reporter in crate::tools::SLOPCOP_BIFROST_READ_ONLY_TOOLS {
            assert!(
                names.contains(reporter),
                "SlopCop surface is missing reporter '{reporter}'; got {names:?}"
            );
            assert_eq!(
                crate::tools::ToolRegistry::tool_kind(reporter),
                agent_client_protocol::schema::v1::ToolKind::Read,
                "SlopCop reporter '{reporter}' must pass the read-only permission gate"
            );
            let tool = client
                .tools()
                .iter()
                .find(|tool| tool.name == *reporter)
                .expect("reporter was present above");
            assert_eq!(
                tool.annotations.read_only_hint,
                Some(true),
                "SlopCop reporter '{reporter}' must be annotated read-only by Bifrost"
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn spawn_passes_configured_env_vars() {
        let tmp = tempfile::tempdir().expect("tmp");
        let script_path = tmp.path().join("fake-mcp.sh");
        let env_log = tmp.path().join("env.log");
        let script = format!(
            r#"#!/bin/sh
printf '%s\n' "$ANVIL_MCP_TEST_TOKEN" > "{}"
while IFS= read -r line; do
  case "$line" in
    *'"method":"initialize"'* )
      printf '%s\n' '{{"jsonrpc":"2.0","id":1,"result":{{"capabilities":{{}}}}}}'
      ;;
    *'"method":"tools/list"'* )
      printf '%s\n' '{{"jsonrpc":"2.0","id":2,"result":{{"tools":[]}}}}'
      exit 0
      ;;
  esac
done
"#,
            env_log.display()
        );
        std::fs::write(&script_path, script).expect("write fake MCP script");
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&script_path)
            .expect("stat fake MCP script")
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script_path, perms).expect("chmod fake MCP script");

        let config = McpServerConfig {
            name: "fake".to_string(),
            transport: crate::mcp::McpTransport::Stdio,
            url: None,
            headers: Vec::new(),
            command: script_path.display().to_string(),
            args: Vec::new(),
            env: vec![McpEnvVar {
                name: "ANVIL_MCP_TEST_TOKEN".to_string(),
                value: "expected-token".to_string(),
            }],
            framing: McpFraming::Line,
            enabled: true,
        };

        let _client = McpClient::spawn(&config, tmp.path())
            .await
            .expect("fake MCP subprocess should start");

        assert_eq!(
            std::fs::read_to_string(&env_log).expect("read env log"),
            "expected-token\n"
        );
    }

    #[cfg(unix)]
    fn write_executable_script(path: &std::path::Path, script: &str) {
        std::fs::write(path, script).expect("write fake MCP script");
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(path)
            .expect("stat fake MCP script")
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(path, perms).expect("chmod fake MCP script");
    }

    #[cfg(unix)]
    fn fake_mcp_config(script_path: &std::path::Path) -> McpServerConfig {
        McpServerConfig {
            name: "fake".to_string(),
            transport: crate::mcp::McpTransport::Stdio,
            url: None,
            headers: Vec::new(),
            command: script_path.display().to_string(),
            args: Vec::new(),
            env: Vec::new(),
            framing: McpFraming::Line,
            enabled: true,
        }
    }

    #[cfg(unix)]
    fn fake_mcp_script(call_arm: &str) -> String {
        format!(
            r#"#!/bin/sh
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
  case "$line" in
    *'"method":"initialize"'* )
      printf '%s\n' "{{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{{\"capabilities\":{{}}}}}}"
      ;;
    *'"method":"tools/list"'* )
      printf '%s\n' "{{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{{\"tools\":[{{\"name\":\"fake_tool\",\"description\":\"Fake\",\"inputSchema\":{{\"type\":\"object\"}}}}]}}}}"
      ;;
    *'"method":"tools/call"'* )
{call_arm}
      ;;
  esac
done
"#
        )
    }

    #[cfg(unix)]
    async fn wait_for_path(path: &std::path::Path) {
        for _ in 0..50 {
            if path.exists() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("timed out waiting for {}", path.display());
    }

    #[cfg(unix)]
    async fn wait_for_log_line(path: &std::path::Path, needle: &str) {
        for _ in 0..50 {
            if let Ok(contents) = std::fs::read_to_string(path)
                && contents.contains(needle)
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("timed out waiting for {needle:?} in {}", path.display());
    }

    /// The MCP spec SHOULDs a `notifications/cancelled` when a client gives
    /// up on a request. `call_tool_with_timeout` sends that notification and
    /// then immediately hands the timeout to `mark_unhealthy`, which SIGKILLs
    /// the stdio child -- there is no guarantee the child's `read` loop is
    /// ever scheduled again before it dies, so it cannot be trusted to
    /// observe (and log) the notification itself. Instead the script `tee`s
    /// all stdin to a log file *before* the read loop consumes it: `tee` runs
    /// as its own process in the pipeline, so it keeps draining and logging
    /// stdin even after `sh script.sh` (the immediate, killed child) is gone.
    #[cfg(unix)]
    #[tokio::test]
    async fn timed_out_stdio_call_sends_cancelled_notification() {
        let tmp = tempfile::tempdir().expect("tmp");
        let script_path = tmp.path().join("fake-mcp.sh");
        let log_path = tmp.path().join("received.log");
        let script = format!(
            r#"#!/bin/sh
tee -a '{log}' | while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9][0-9]*\).*/\1/p')
  case "$line" in
    *'"method":"initialize"'* )
      printf '%s\n' "{{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{{\"capabilities\":{{}}}}}}"
      ;;
    *'"method":"tools/list"'* )
      printf '%s\n' "{{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{{\"tools\":[{{\"name\":\"fake_tool\",\"description\":\"Fake\",\"inputSchema\":{{\"type\":\"object\"}}}}]}}}}"
      ;;
    *'"method":"tools/call"'* )
      sleep 60
      ;;
  esac
done
"#,
            log = log_path.display()
        );
        write_executable_script(&script_path, &script);

        let client = McpClient::spawn(&fake_mcp_config(&script_path), tmp.path())
            .await
            .expect("fake MCP subprocess should start");

        let err = client
            .call_tool_with_timeout("fake_tool", json!({}), Duration::from_millis(100), None)
            .await
            .expect_err("call should time out");
        assert!(
            matches!(err, McpError::Timeout { .. }),
            "expected timeout, got {err}"
        );

        wait_for_log_line(&log_path, "notifications/cancelled").await;
        let log = std::fs::read_to_string(&log_path).expect("read log");
        let cancelled_line = log
            .lines()
            .find(|line| line.contains("notifications/cancelled"))
            .expect("cancellation notification should have been logged");
        let cancelled: Value =
            serde_json::from_str(cancelled_line).expect("parse cancellation notification");
        assert_eq!(
            cancelled["method"].as_str(),
            Some("notifications/cancelled")
        );
        assert_eq!(
            cancelled["params"]["reason"].as_str(),
            Some("Request timed out")
        );
        // initialize=1, tools/list=2, tools/call=3 for a freshly spawned client.
        assert_eq!(cancelled["params"]["requestId"].as_i64(), Some(3));
        assert!(
            cancelled.get("id").is_none(),
            "notifications must not carry an id"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timed_out_call_marks_unhealthy_and_next_call_respawns() {
        let tmp = tempfile::tempdir().expect("tmp");
        let script_path = tmp.path().join("fake-mcp.sh");
        let marker = tmp.path().join("first-call");
        let script = fake_mcp_script(&format!(
            r#"      if [ ! -f '{marker}' ]; then
        : > '{marker}'
        sleep 60
      else
        printf '%s\n' "{{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{{\"content\":[{{\"type\":\"text\",\"text\":\"ok\"}}]}}}}"
      fi"#,
            marker = marker.display()
        ));
        write_executable_script(&script_path, &script);

        let client = McpClient::spawn(&fake_mcp_config(&script_path), tmp.path())
            .await
            .expect("fake MCP subprocess should start");

        let err = client
            .call_tool_with_timeout("fake_tool", json!({}), Duration::from_millis(100), None)
            .await
            .expect_err("first call should time out");
        assert!(
            matches!(err, McpError::Timeout { .. }),
            "expected timeout, got {err}"
        );
        wait_for_path(&marker).await;

        let value = client
            .call_tool_with_timeout("fake_tool", json!({}), Duration::from_secs(2), None)
            .await
            .expect("next call should respawn and succeed");
        assert_eq!(value, json!("ok"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn closed_subprocess_marks_unhealthy_and_next_call_respawns() {
        let tmp = tempfile::tempdir().expect("tmp");
        let script_path = tmp.path().join("fake-mcp.sh");
        let marker = tmp.path().join("first-call");
        let script = fake_mcp_script(&format!(
            r#"      if [ ! -f '{marker}' ]; then
        : > '{marker}'
        exit 0
      else
        printf '%s\n' "{{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{{\"content\":[{{\"type\":\"text\",\"text\":\"ok\"}}]}}}}"
      fi"#,
            marker = marker.display()
        ));
        write_executable_script(&script_path, &script);

        let client = McpClient::spawn(&fake_mcp_config(&script_path), tmp.path())
            .await
            .expect("fake MCP subprocess should start");

        let err = client
            .call_tool_with_timeout("fake_tool", json!({}), Duration::from_secs(2), None)
            .await
            .expect_err("first call should fail when subprocess closes stdout");
        assert!(
            matches!(err, McpError::Io(_)),
            "expected io error, got {err}"
        );

        let value = client
            .call_tool_with_timeout("fake_tool", json!({}), Duration::from_secs(2), None)
            .await
            .expect("next call should respawn and succeed");
        assert_eq!(value, json!("ok"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelled_call_marks_unhealthy_and_next_call_respawns() {
        let tmp = tempfile::tempdir().expect("tmp");
        let script_path = tmp.path().join("fake-mcp.sh");
        let marker = tmp.path().join("first-call");
        let script = fake_mcp_script(&format!(
            r#"      if [ ! -f '{marker}' ]; then
        : > '{marker}'
        sleep 60
      else
        printf '%s\n' "{{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{{\"content\":[{{\"type\":\"text\",\"text\":\"ok\"}}]}}}}"
      fi"#,
            marker = marker.display()
        ));
        write_executable_script(&script_path, &script);

        let client = McpClient::spawn(&fake_mcp_config(&script_path), tmp.path())
            .await
            .expect("fake MCP subprocess should start");
        let cancel = CancellationToken::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            trigger.cancel();
        });

        let err = client
            .call_tool_with_timeout(
                "fake_tool",
                json!({}),
                Duration::from_secs(30),
                Some(&cancel),
            )
            .await
            .expect_err("first call should be cancelled");
        assert!(
            matches!(err, McpError::Cancelled { .. }),
            "expected cancellation, got {err}"
        );

        let value = client
            .call_tool_with_timeout("fake_tool", json!({}), Duration::from_secs(2), None)
            .await
            .expect("next call should respawn and succeed");
        assert_eq!(value, json!("ok"));
    }

    /// Two `tools/call` requests issued concurrently must overlap on the
    /// server instead of queueing client-side.
    ///
    /// Regression guard for the head-of-line block that
    /// `call_tool_with_timeout` used to have: it held the client-state mutex
    /// across both the request write *and* the response read, so a parallel
    /// safe-tool batch (`execute_parallel_safe_calls`) serialized inside the
    /// MCP client. Measured on the CodeScaleBench grep-hard symbols arm
    /// (2026-08-07): 8029s of client-observed latency against 4158s of
    /// server-side execution across 378 aligned calls.
    ///
    /// The fake server answers every call from a background subshell after
    /// `SERVER_DELAY`, so two overlapping calls cost one delay and two
    /// serialized calls cost two.
    #[cfg(unix)]
    #[tokio::test]
    async fn concurrent_tool_calls_overlap_instead_of_serializing() {
        const SERVER_DELAY: Duration = Duration::from_secs(2);

        let tmp = tempfile::tempdir().expect("tmp");
        let script_path = tmp.path().join("fake-mcp.sh");
        let script = fake_mcp_script(
            r#"      ( sleep 2
        printf '%s\n' "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"ok\"}]}}"
      ) &"#,
        );
        write_executable_script(&script_path, &script);

        let client = McpClient::spawn(&fake_mcp_config(&script_path), tmp.path())
            .await
            .expect("fake MCP subprocess should start");

        let started = std::time::Instant::now();
        let (first, second) = tokio::join!(
            client.call_tool_with_timeout("fake_tool", json!({}), Duration::from_secs(60), None),
            client.call_tool_with_timeout("fake_tool", json!({}), Duration::from_secs(60), None),
        );
        let elapsed = started.elapsed();

        assert_eq!(first.expect("first call should succeed"), json!("ok"));
        assert_eq!(second.expect("second call should succeed"), json!("ok"));
        assert!(
            elapsed >= SERVER_DELAY,
            "sanity: the fake server should have delayed both calls, took {elapsed:?}"
        );
        assert!(
            elapsed < SERVER_DELAY + SERVER_DELAY / 2,
            "two concurrent MCP calls of {SERVER_DELAY:?} each took {elapsed:?}; \
             they are serializing client-side instead of overlapping"
        );
    }

    /// Responses that come back out of order are routed by JSON-RPC id, not by
    /// arrival order: the slow call keeps waiting and each caller gets its own
    /// result.
    #[cfg(unix)]
    #[tokio::test]
    async fn out_of_order_responses_route_to_their_own_callers() {
        let tmp = tempfile::tempdir().expect("tmp");
        let script_path = tmp.path().join("fake-mcp.sh");
        // Echo the requested tool name back in the result so a mis-routed
        // response is visible to the caller, and answer `slow_tool` late so
        // the second request's response overtakes the first's.
        let script = fake_mcp_script(
            r#"      tool=$(printf '%s' "$line" | sed -n 's/.*"name":"\([a-z_]*\)".*/\1/p')
      case "$tool" in
        slow_tool )
          ( sleep 2
            printf '%s\n' "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"$tool\"}]}}"
          ) &
          ;;
        * )
          printf '%s\n' "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"$tool\"}]}}"
          ;;
      esac"#,
        );
        write_executable_script(&script_path, &script);

        let client = McpClient::spawn(&fake_mcp_config(&script_path), tmp.path())
            .await
            .expect("fake MCP subprocess should start");

        let started = std::time::Instant::now();
        let slow = async {
            let value = client
                .call_tool_with_timeout("slow_tool", json!({}), Duration::from_secs(60), None)
                .await
                .expect("slow call should succeed");
            (value, started.elapsed())
        };
        let fast = async {
            // Give the slow request a head start so its id is issued first.
            tokio::time::sleep(Duration::from_millis(200)).await;
            let value = client
                .call_tool_with_timeout("fast_tool", json!({}), Duration::from_secs(60), None)
                .await
                .expect("fast call should succeed");
            (value, started.elapsed())
        };
        let ((slow_value, slow_at), (fast_value, fast_at)) = tokio::join!(slow, fast);

        assert_eq!(slow_value, json!("slow_tool"));
        assert_eq!(fast_value, json!("fast_tool"));
        assert!(
            fast_at < slow_at,
            "the later request's response arrived first, so it must also \
             complete first: fast at {fast_at:?}, slow at {slow_at:?}"
        );
    }

    /// When the server closes its stdout while requests are in flight, every
    /// pending caller fails immediately with the transport error instead of
    /// waiting out its own timeout.
    #[cfg(unix)]
    #[tokio::test]
    async fn reader_death_fails_all_pending_calls_promptly() {
        let tmp = tempfile::tempdir().expect("tmp");
        let script_path = tmp.path().join("fake-mcp.sh");
        let pending_marker = tmp.path().join("pending");
        let script = fake_mcp_script(&format!(
            r#"      case "$line" in
        *hang_tool* )
          : > '{marker}'
          ;;
        * )
          exit 0
          ;;
      esac"#,
            marker = pending_marker.display()
        ));
        write_executable_script(&script_path, &script);

        let client = McpClient::spawn(&fake_mcp_config(&script_path), tmp.path())
            .await
            .expect("fake MCP subprocess should start");

        // `hang_tool` is accepted and never answered; `kill_tool` then makes
        // the server exit, closing the stream with one call still pending.
        let hanging = async {
            client
                .call_tool_with_timeout("hang_tool", json!({}), Duration::from_secs(120), None)
                .await
        };
        let killing = async {
            wait_for_path(&pending_marker).await;
            client
                .call_tool_with_timeout("kill_tool", json!({}), Duration::from_secs(120), None)
                .await
        };

        let (hanging, killing) = tokio::time::timeout(Duration::from_secs(10), async {
            tokio::join!(hanging, killing)
        })
        .await
        .expect("pending MCP calls must fail when the server closes the stream");

        let hanging = hanging.expect_err("the unanswered call must fail");
        assert!(
            matches!(hanging, McpError::Io(_)),
            "expected an io error for the pending call, got {hanging}"
        );
        let killing = killing.expect_err("the call that killed the server must fail");
        assert!(
            matches!(killing, McpError::Io(_)),
            "expected an io error for the second call, got {killing}"
        );
    }
}
