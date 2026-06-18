use anyhow::Context;
use sha2::{Digest, Sha256};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, Ordering};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

const PROTOCOL_VERSION: &str = "2025-11-25";
pub const BUNDLED_BIFROST_VERSION: &str = "0.6.5";
const BIFROST_RELEASE_BASE: &str = "https://github.com/BrokkAi/bifrost/releases/download";

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
        }
    }
}

impl std::error::Error for McpError {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
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
        "--server".to_string(),
        "core".to_string(),
    ]
}

fn legacy_core_bifrost_args() -> Vec<String> {
    vec![
        "--root".to_string(),
        "{cwd}".to_string(),
        "--server".to_string(),
        "core".to_string(),
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

fn is_legacy_or_managed_bifrost_command(command: &str) -> bool {
    command == "bifrost" || command == managed_bifrost_command()
}

/// Normalises a stored Bifrost server entry so it always uses Anvil's managed
/// local binary with the correct line framing.
///
/// The function matches on name, current-or-legacy default args, and a legacy (`"bifrost"`) or
/// managed-path command.  When all three match the **entire** `McpServerConfig`
/// is replaced by [`McpServerConfig::bifrost()`]; only `enabled` is preserved.
/// Any other customisation (e.g. a manually-set framing override) is
/// intentionally discarded — Bifrost's wire protocol requires line framing and
/// the command must point to the pinned managed binary.
pub fn normalize_preinstalled_bifrost_server(server: &mut McpServerConfig) {
    if server.name != "bifrost"
        || !matches!(
            server.args.as_slice(),
            args if args == default_bifrost_args().as_slice()
                || args == legacy_core_bifrost_args().as_slice()
        )
        || !is_legacy_or_managed_bifrost_command(&server.command)
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
            command: managed_bifrost_command(),
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
    _child: Mutex<Child>,
    io: Mutex<McpIo>,
    next_id: AtomicI64,
    tools: Vec<McpToolDef>,
}

struct McpIo {
    writer: ChildStdin,
    reader: BufReader<ChildStdout>,
    framing: McpFraming,
}

impl McpClient {
    pub async fn spawn(config: &McpServerConfig, cwd: &Path) -> Result<Self, McpError> {
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
        let next_id = AtomicI64::new(1);

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
        let tools = parse_tool_list(list)?;
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

        Ok(Self {
            name: config.name.clone(),
            _child: Mutex::new(child),
            io: Mutex::new(io),
            next_id,
            tools,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn tools(&self) -> &[McpToolDef] {
        &self.tools
    }

    pub async fn call_tool(&self, name: &str, args: Value) -> Result<Value, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let mut io = self.io.lock().await;
        write_request(
            &mut io,
            id,
            "tools/call",
            json!({ "name": name, "arguments": args }),
        )
        .await?;
        let result = read_response(&mut io, id).await?;

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

fn parse_tool_list(result: Value) -> Result<Vec<McpToolDef>, McpError> {
    let tools_array = result
        .get("tools")
        .and_then(Value::as_array)
        .ok_or_else(|| McpError::Protocol("tools/list missing 'tools' array".into()))?;
    tools_array
        .iter()
        .map(|tool| {
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
            let annotations = parse_tool_annotations(tool.get("annotations"));
            Ok(McpToolDef {
                name,
                description,
                input_schema,
                annotations,
            })
        })
        .collect()
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

        for expected in ["search_symbols", "get_summaries"] {
            assert!(
                names.contains(&expected),
                "missing tool {expected} in {names:?}"
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
                agent_client_protocol::schema::ToolKind::Read
                    | agent_client_protocol::schema::ToolKind::Search
                    | agent_client_protocol::schema::ToolKind::Fetch
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
}
