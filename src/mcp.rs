use anyhow::Context;
use futures::StreamExt;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

const PROTOCOL_VERSION: &str = "2025-11-25";
pub const BUNDLED_BIFROST_VERSION: &str = "0.7.4";
const BIFROST_RELEASE_BASE: &str = "https://github.com/BrokkAi/bifrost/releases/download";
const MCP_CALL_TIMEOUT: Duration = Duration::from_secs(60);

#[cfg(target_os = "macos")]
const BIFROST_TARGET_TRIPLE: &str = "universal-apple-darwin";
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const BIFROST_TARGET_TRIPLE: &str = "x86_64-unknown-linux-gnu";
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

#[derive(Debug)]
pub enum McpError {
    Spawn(String),
    Io(String),
    Protocol(String),
    JsonRpc { code: i64, message: String },
    Timeout { tool: String, timeout: Duration },
    Cancelled { tool: String },
}

impl fmt::Display for McpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            McpError::Spawn(s) => write!(f, "spawn failed: {s}"),
            McpError::Io(s) => write!(f, "io error: {s}"),
            McpError::Protocol(s) => write!(f, "protocol error: {s}"),
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

fn default_bifrost_args() -> Vec<String> {
    vec![
        "--root".to_string(),
        "{cwd}".to_string(),
        "--mcp".to_string(),
        // `searchtools` is Bifrost's full read-only surface: the `core`
        // symbol/nlp/workspace tools plus the extended/text search tools and
        // the SlopCop code-quality reporters. SlopCop ACP read-only sessions
        // need the reporters (and tools like find_filenames/get_file_contents)
        // exposed; `core` advertised neither (#121). Every tool `searchtools`
        // advertises has a read-classified `ToolMeta` row, which the
        // Bifrost handshake/anti-drift test asserts.
        "searchtools".to_string(),
        "--no-line-numbers".to_string(),
    ]
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
/// command) is an unmodified prior default, so it is upgraded to the current
/// default on load. Without this, existing installs would keep their old
/// surface and never pick up changes short of a manual `/mcp reset`: the
/// `core` -> `searchtools` switch (#121), and the deprecated `--server` flag
/// -> `--mcp`.
fn legacy_default_bifrost_arg_sets() -> Vec<Vec<String>> {
    vec![
        // `--server core` (pre-#121 surface).
        vec![
            "--root".to_string(),
            "{cwd}".to_string(),
            "--server".to_string(),
            "core".to_string(),
            "--no-line-numbers".to_string(),
        ],
        // `--server searchtools` (current surface, but on the deprecated flag).
        vec![
            "--root".to_string(),
            "{cwd}".to_string(),
            "--server".to_string(),
            "searchtools".to_string(),
            "--no-line-numbers".to_string(),
        ],
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

    let archive_path = cache_dir.join(&asset);
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
        .arg(cache_dir)
        .status()
        .await
        .with_context(|| format!("invoking tar to extract {}", archive_path.display()))?;
    anyhow::ensure!(
        status.success(),
        "tar extraction failed for {} with status {status}",
        archive_path.display()
    );

    let inner_dir = cache_dir.join(format!(
        "bifrost-v{BUNDLED_BIFROST_VERSION}-{BIFROST_TARGET_TRIPLE}"
    ));
    let inner_binary = inner_dir.join(BIFROST_BINARY_NAME);
    anyhow::ensure!(
        inner_binary.is_file(),
        "expected extracted bifrost binary at {}",
        inner_binary.display()
    );

    let target = cache_dir.join(BIFROST_BINARY_NAME);
    if tokio::fs::rename(&inner_binary, &target).await.is_err() {
        tokio::fs::copy(&inner_binary, &target)
            .await
            .with_context(|| {
                format!(
                    "moving bundled bifrost from {} to {}",
                    inner_binary.display(),
                    target.display()
                )
            })?;
    }

    let _ = tokio::fs::remove_file(&archive_path).await;
    let _ = tokio::fs::remove_dir_all(&inner_dir).await;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = tokio::fs::metadata(&target)
            .await
            .with_context(|| format!("stat bundled bifrost binary {}", target.display()))?
            .permissions();
        perms.set_mode(0o755);
        tokio::fs::set_permissions(&target, perms)
            .await
            .with_context(|| format!("chmod 755 {}", target.display()))?;
    }

    Ok(())
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

/// JSON-RPC client for a long-lived stdio MCP subprocess.
///
/// Holds the child process for the lifetime of the client; the process is
/// killed when the client is dropped (`kill_on_drop(true)`).
///
/// The MCP stdio protocol uses `Content-Length` framed JSON-RPC messages.
/// Reads and writes are serialized through a single mutex because the existing
/// tool loop dispatches tool calls sequentially within a session.
pub struct McpClient {
    name: String,
    config: McpServerConfig,
    cwd: PathBuf,
    state: Mutex<McpClientState>,
    next_id: AtomicI64,
    tools: Vec<McpToolDef>,
}

enum McpClientState {
    Stdio {
        child: Box<Child>,
        io: McpIo,
        healthy: bool,
    },
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

struct McpIo {
    writer: ChildStdin,
    reader: BufReader<ChildStdout>,
    framing: McpFraming,
}

impl McpClient {
    pub async fn spawn(config: &McpServerConfig, cwd: &Path) -> Result<Self, McpError> {
        let next_id = AtomicI64::new(1);
        let (state, tools) = Self::spawn_connected(config, cwd, &next_id).await?;

        Ok(Self {
            name: config.name.clone(),
            config: config.clone(),
            cwd: cwd.to_path_buf(),
            state: Mutex::new(state),
            next_id,
            tools,
        })
    }

    async fn spawn_connected(
        config: &McpServerConfig,
        cwd: &Path,
        next_id: &AtomicI64,
    ) -> Result<(McpClientState, Vec<McpToolDef>), McpError> {
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
        let reader = BufReader::new(
            child
                .stdout
                .take()
                .ok_or_else(|| McpError::Spawn("missing stdout pipe".into()))?,
        );

        let mut io = McpIo {
            writer,
            reader,
            framing: config.framing,
        };

        let init_id = next_id.fetch_add(1, Ordering::SeqCst);
        write_request(
            &mut io,
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
        let _init = read_response(&mut io, init_id).await?;

        write_notification(&mut io, "notifications/initialized", json!({})).await?;

        let list_id = next_id.fetch_add(1, Ordering::SeqCst);
        write_request(&mut io, list_id, "tools/list", json!({})).await?;
        let list = read_response(&mut io, list_id).await?;
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
            McpClientState::Stdio {
                child: Box::new(child),
                io,
                healthy: true,
            },
            tools,
        ))
    }

    async fn connect_http(
        config: &McpServerConfig,
        next_id: &AtomicI64,
    ) -> Result<(McpClientState, Vec<McpToolDef>), McpError> {
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
        let (_, session_id) = http_request_with_session(
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
        ))
    }

    async fn connect_sse(
        config: &McpServerConfig,
        next_id: &AtomicI64,
    ) -> Result<(McpClientState, Vec<McpToolDef>), McpError> {
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
        let endpoint = tokio::time::timeout(MCP_CALL_TIMEOUT, endpoint_rx)
            .await
            .map_err(|_| McpError::Timeout {
                tool: "SSE endpoint discovery".into(),
                timeout: MCP_CALL_TIMEOUT,
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
        let _ = sse_request(
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
        sse_notification(&state, "notifications/initialized", json!({})).await?;
        let list_id = next_id.fetch_add(1, Ordering::SeqCst);
        let list = sse_request(&mut state, list_id, "tools/list", json!({})).await?;
        let tools = parse_tool_list(list, Some(config.name.as_str()))?;
        tracing::info!(server = %config.name, url, tool_count = tools.len(), "SSE MCP server ready");
        Ok((state, tools))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn tools(&self) -> &[McpToolDef] {
        &self.tools
    }

    pub async fn call_tool(&self, name: &str, args: Value) -> Result<Value, McpError> {
        self.call_tool_with_timeout(name, args, MCP_CALL_TIMEOUT, None)
            .await
    }

    pub async fn call_tool_cancellable(
        &self,
        name: &str,
        args: Value,
        cancel: Option<&CancellationToken>,
    ) -> Result<Value, McpError> {
        self.call_tool_with_timeout(name, args, MCP_CALL_TIMEOUT, cancel)
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

        if matches!(&*state, McpClientState::Stdio { healthy: false, .. }) {
            let (new_state, _) =
                Self::spawn_connected(&self.config, &self.cwd, &self.next_id).await?;
            *state = new_state;
        }

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        if matches!(&*state, McpClientState::Sse { .. }) {
            let call = sse_request(
                &mut state,
                id,
                "tools/call",
                json!({ "name": name, "arguments": args }),
            );
            let result = match cancel {
                Some(cancel) => tokio::select! {
                    biased;
                    _ = cancel.cancelled() => Err(McpError::Cancelled { tool: name.to_string() }),
                    result = tokio::time::timeout(timeout, call) => result.unwrap_or_else(|_| Err(McpError::Timeout { tool: name.to_string(), timeout })),
                },
                None => tokio::time::timeout(timeout, call)
                    .await
                    .unwrap_or_else(|_| {
                        Err(McpError::Timeout {
                            tool: name.to_string(),
                            timeout,
                        })
                    }),
            }?;
            return parse_tool_result(result);
        }

        if let McpClientState::Http {
            client,
            url,
            headers,
            session_id,
        } = &*state
        {
            let call = http_request(
                client,
                url,
                headers,
                session_id.as_ref(),
                id,
                "tools/call",
                json!({ "name": name, "arguments": args }),
            );
            let result = match cancel {
                Some(cancel) => tokio::select! {
                    biased;
                    _ = cancel.cancelled() => Err(McpError::Cancelled { tool: name.to_string() }),
                    result = tokio::time::timeout(timeout, call) => result.unwrap_or_else(|_| Err(McpError::Timeout { tool: name.to_string(), timeout })),
                },
                None => tokio::time::timeout(timeout, call)
                    .await
                    .unwrap_or_else(|_| {
                        Err(McpError::Timeout {
                            tool: name.to_string(),
                            timeout,
                        })
                    }),
            }?;
            return parse_tool_result(result);
        }

        let McpClientState::Stdio { io, .. } = &mut *state else {
            unreachable!()
        };
        if let Err(err) = write_request(
            io,
            id,
            "tools/call",
            json!({ "name": name, "arguments": args }),
        )
        .await
        {
            mark_unhealthy(&mut state, &err).await;
            return Err(err);
        }

        let McpClientState::Stdio { io, .. } = &mut *state else {
            unreachable!()
        };
        let read = read_response(io, id);
        let result = match cancel {
            Some(cancel) => {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => Err(McpError::Cancelled { tool: name.to_string() }),
                    result = tokio::time::timeout(timeout, read) => match result {
                        Ok(result) => result,
                        Err(_) => Err(McpError::Timeout {
                            tool: name.to_string(),
                            timeout,
                        }),
                    },
                }
            }
            None => match tokio::time::timeout(timeout, read).await {
                Ok(result) => result,
                Err(_) => Err(McpError::Timeout {
                    tool: name.to_string(),
                    timeout,
                }),
            },
        };
        let result = match result {
            Ok(result) => result,
            Err(err) => {
                mark_unhealthy(&mut state, &err).await;
                return Err(err);
            }
        };

        parse_tool_result(result)
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

async fn mark_unhealthy(state: &mut McpClientState, err: &McpError) {
    if err.leaves_client_unhealthy()
        && let McpClientState::Stdio { child, healthy, .. } = state
    {
        *healthy = false;
        let _ = child.kill().await;
    }
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
        return Err(McpError::Protocol(format!("HTTP {status}: {body}")));
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
                    Some("initialize") => json!({"jsonrpc":"2.0","id":value["id"],"result":{}}),
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
        assert_eq!(client.tools()[0].name, "echo");
        assert_eq!(
            client.call_tool("echo", json!({})).await.unwrap(),
            json!({"ok":true})
        );
        thread.join().unwrap();
        assert_eq!(requests.lock().unwrap().len(), 4);
    }
}

async fn write_request(
    io: &mut McpIo,
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

async fn write_notification(io: &mut McpIo, method: &str, params: Value) -> Result<(), McpError> {
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

async fn write_message(io: &mut McpIo, msg: &Value) -> Result<(), McpError> {
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

async fn read_response(io: &mut McpIo, expected_id: i64) -> Result<Value, McpError> {
    loop {
        let value = read_message(io).await?;
        if value.get("id").and_then(Value::as_i64) != Some(expected_id) {
            tracing::debug!(?value, "skipping mcp message with unexpected id");
            continue;
        }
        if let Some(error) = value.get("error") {
            let code = error.get("code").and_then(Value::as_i64).unwrap_or(0);
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            return Err(McpError::JsonRpc { code, message });
        }
        return value
            .get("result")
            .cloned()
            .ok_or_else(|| McpError::Protocol("response missing result".into()));
    }
}

async fn read_message(io: &mut McpIo) -> Result<Value, McpError> {
    if io.framing == McpFraming::Line {
        return read_line_message(io).await;
    }

    let mut content_length = None;
    loop {
        let mut line = String::new();
        let n = io
            .reader
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
    io.reader
        .read_exact(&mut body)
        .await
        .map_err(|e| McpError::Io(format!("read body: {e}")))?;
    serde_json::from_slice(&body).map_err(|e| McpError::Protocol(format!("parse body: {e}")))
}

async fn read_line_message(io: &mut McpIo) -> Result<Value, McpError> {
    let mut line = String::new();
    let n = io
        .reader
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

    /// A persisted bifrost entry carrying the prior managed default
    /// (`--server core`) is upgraded to the current default surface on load, so
    /// existing installs pick up the searchtools switch without `/mcp reset`
    /// (#121).
    #[test]
    fn legacy_default_bifrost_args_are_upgraded_to_current_default() {
        let mut server = McpServerConfig {
            name: "bifrost".to_string(),
            transport: crate::mcp::McpTransport::Stdio,
            url: None,
            headers: Vec::new(),
            command: "bifrost".to_string(),
            args: vec![
                "--root".to_string(),
                "{cwd}".to_string(),
                "--server".to_string(),
                "core".to_string(),
                "--no-line-numbers".to_string(),
            ],
            env: Vec::new(),
            framing: McpFraming::Line,
            enabled: false,
        };
        normalize_preinstalled_bifrost_server(&mut server);
        assert_eq!(
            server.args,
            default_bifrost_args(),
            "a stored prior-default bifrost entry should be upgraded to the current default"
        );
        assert!(
            server.args.iter().any(|a| a == "searchtools"),
            "upgraded entry should use the searchtools surface"
        );
        // Non-default fields are preserved.
        assert!(!server.enabled, "the stored enabled flag must be preserved");
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
    /// confirm a stable subset of search tools is exposed, and round-trip
    /// two distinct tool calls. We deliberately do NOT pin the exact tool
    /// count or full tool list -- bifrost adds tools faster than this test
    /// gets updated, and the handshake's job is to verify the protocol
    /// path works, not to enumerate the surface.
    #[tokio::test]
    async fn handshake_and_call_search_tools() {
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
            // Extended/text search tools the SlopCop workflow relies on, now
            // exposed by the default `searchtools` surface (#121).
            "find_filenames",
            "get_file_contents",
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

        // The SlopCop code-quality reporters must be advertised by the default
        // surface AND known to Anvil's permission metadata, so read-only ACP
        // sessions can call them instead of having them fall back to
        // `ToolKind::Other` and be blocked (#121).
        for reporter in crate::tools::SLOPCOP_BIFROST_READ_ONLY_TOOLS {
            assert!(
                names.contains(reporter),
                "SlopCop reporter '{reporter}' not advertised by default Bifrost surface; \
                 got {names:?}"
            );
            assert!(
                crate::tools::is_known_tool(reporter),
                "SlopCop reporter '{reporter}' is advertised but missing from the TOOLS table"
            );
        }

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
        // of the JSON-RPC reader/writer mutex (id correlation, sequential
        // dispatch, response-shape branching) -- not just one-shot dispatch.
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
}
