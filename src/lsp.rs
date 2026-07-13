use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, Command};
use tokio::sync::{Mutex, RwLock, oneshot};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LspServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LspSettings {
    #[serde(default)]
    pub diagnostics_on_read: bool,
    #[serde(default)]
    pub diagnostics_on_write: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub servers: Vec<LspServerConfig>,
}

impl Default for LspSettings {
    fn default() -> Self {
        Self {
            diagnostics_on_read: false,
            diagnostics_on_write: true,
            servers: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub path: PathBuf,
    pub severity: DiagnosticSeverity,
    pub line: u64,
    pub character: u64,
    pub source: String,
    pub code: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Information,
    Hint,
}

impl DiagnosticSeverity {
    fn from_lsp(value: Option<u64>) -> Self {
        match value.unwrap_or(3) {
            1 => Self::Error,
            2 => Self::Warning,
            4 => Self::Hint,
            _ => Self::Information,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Error => "Error",
            Self::Warning => "Warn",
            Self::Information => "Info",
            Self::Hint => "Hint",
        }
    }
}

pub struct LspManager {
    settings: LspSettings,
    clients: Vec<Arc<LspClient>>,
}

impl LspManager {
    pub async fn start(cwd: PathBuf, settings: LspSettings) -> Arc<Self> {
        let mut clients = Vec::new();
        for config in settings.servers.iter().filter(|server| server.enabled) {
            match LspClient::start(cwd.clone(), config.clone()).await {
                Ok(client) => clients.push(Arc::new(client)),
                Err(err) => tracing::warn!(
                    server = %config.name,
                    command = %config.command,
                    %err,
                    "LSP server failed to start; diagnostics disabled for this server"
                ),
            }
        }
        Arc::new(Self { settings, clients })
    }

    pub fn diagnostics_on_read(&self) -> bool {
        self.settings.diagnostics_on_read && !self.clients.is_empty()
    }

    pub fn diagnostics_on_write(&self) -> bool {
        self.settings.diagnostics_on_write && !self.clients.is_empty()
    }

    pub fn has_clients(&self) -> bool {
        !self.clients.is_empty()
    }

    pub async fn open_file(&self, path: &Path) {
        for client in &self.clients {
            if let Err(err) = client.open_file(path).await {
                tracing::debug!(server = %client.name, path = %path.display(), %err, "LSP didOpen failed");
            }
        }
    }

    pub async fn change_file(&self, path: &Path) {
        for client in &self.clients {
            if let Err(err) = client.change_file(path).await {
                tracing::debug!(server = %client.name, path = %path.display(), %err, "LSP didChange failed");
            }
        }
    }

    pub async fn diagnostics_for_file(&self, path: &Path) -> Vec<Diagnostic> {
        let mut out = Vec::new();
        for client in &self.clients {
            out.extend(client.diagnostics_for_file(path).await);
        }
        sort_diagnostics(&mut out);
        out
    }

    pub async fn all_diagnostics(&self) -> Vec<Diagnostic> {
        let mut out = Vec::new();
        for client in &self.clients {
            out.extend(client.all_diagnostics().await);
        }
        sort_diagnostics(&mut out);
        out
    }

    pub async fn wait_for_file_diagnostics(
        &self,
        path: &Path,
        timeout: Duration,
    ) -> Vec<Diagnostic> {
        let deadline = Instant::now() + timeout;
        loop {
            let diagnostics = self.diagnostics_for_file(path).await;
            if !diagnostics.is_empty() || Instant::now() >= deadline {
                return diagnostics;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

impl Drop for LspManager {
    fn drop(&mut self) {
        for client in &self.clients {
            client.abort();
        }
    }
}

fn sort_diagnostics(diagnostics: &mut [Diagnostic]) {
    diagnostics.sort_by(|a, b| {
        a.severity
            .cmp(&b.severity)
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.line.cmp(&b.line))
            .then_with(|| a.character.cmp(&b.character))
            .then_with(|| a.message.cmp(&b.message))
    });
}

pub fn format_diagnostics(file_path: Option<&Path>, diagnostics: &[Diagnostic]) -> String {
    if diagnostics.is_empty() {
        return String::new();
    }
    let mut file_diags = Vec::new();
    let mut project_diags = Vec::new();
    for diagnostic in diagnostics {
        let line = format_diagnostic(diagnostic);
        if file_path.is_some_and(|path| path == diagnostic.path) {
            file_diags.push(line);
        } else {
            project_diags.push(line);
        }
    }

    let mut out = String::new();
    if !file_diags.is_empty() {
        out.push_str("\n<file_diagnostics>\n");
        append_limited(&mut out, &file_diags);
        out.push_str("\n</file_diagnostics>\n");
    }
    if !project_diags.is_empty() {
        out.push_str("\n<project_diagnostics>\n");
        append_limited(&mut out, &project_diags);
        out.push_str("\n</project_diagnostics>\n");
    }
    let file_errors = count_severity(&file_diags, "Error");
    let file_warnings = count_severity(&file_diags, "Warn");
    let project_errors = count_severity(&project_diags, "Error");
    let project_warnings = count_severity(&project_diags, "Warn");
    out.push_str("\n<diagnostic_summary>\n");
    if file_path.is_some() {
        out.push_str(&format!(
            "Current file: {file_errors} errors, {file_warnings} warnings\n"
        ));
    }
    out.push_str(&format!(
        "Project: {project_errors} errors, {project_warnings} warnings\n"
    ));
    out.push_str("</diagnostic_summary>\n");
    out
}

fn append_limited(out: &mut String, lines: &[String]) {
    let shown = lines.len().min(10);
    out.push_str(&lines[..shown].join("\n"));
    if lines.len() > shown {
        out.push_str(&format!(
            "\n... and {} more diagnostics",
            lines.len() - shown
        ));
    }
}

fn count_severity(lines: &[String], severity: &str) -> usize {
    lines
        .iter()
        .filter(|line| line.starts_with(severity))
        .count()
}

fn format_diagnostic(diagnostic: &Diagnostic) -> String {
    let source = if diagnostic.source.is_empty() {
        "lsp".to_string()
    } else {
        diagnostic.source.clone()
    };
    let code = diagnostic
        .code
        .as_ref()
        .map(|code| format!("[{code}]"))
        .unwrap_or_default();
    format!(
        "{}: {}:{}:{} [{}]{} {}",
        diagnostic.severity.label(),
        diagnostic.path.display(),
        diagnostic.line + 1,
        diagnostic.character + 1,
        source,
        code,
        diagnostic.message.replace('\n', " ")
    )
}

struct LspClient {
    name: String,
    root: PathBuf,
    stdin: Arc<Mutex<ChildStdin>>,
    next_id: AtomicI64,
    pending: Arc<Mutex<HashMap<i64, oneshot::Sender<Value>>>>,
    diagnostics: Arc<RwLock<HashMap<PathBuf, Vec<Diagnostic>>>>,
    open_files: Mutex<HashMap<PathBuf, i32>>,
    child: Mutex<tokio::process::Child>,
}

impl LspClient {
    async fn start(root: PathBuf, config: LspServerConfig) -> anyhow::Result<Self> {
        let mut child = Command::new(&config.command)
            .args(&config.args)
            .current_dir(&root)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("missing LSP stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("missing LSP stdout"))?;
        if let Some(stderr) = child.stderr.take() {
            let name = config.name.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!(server = %name, "LSP stderr: {line}");
                }
            });
        }

        let client = Self {
            name: config.name,
            root,
            stdin: Arc::new(Mutex::new(stdin)),
            next_id: AtomicI64::new(1),
            pending: Arc::new(Mutex::new(HashMap::new())),
            diagnostics: Arc::new(RwLock::new(HashMap::new())),
            open_files: Mutex::new(HashMap::new()),
            child: Mutex::new(child),
        };
        client.spawn_reader(stdout);
        client.initialize().await?;
        Ok(client)
    }

    fn spawn_reader(&self, stdout: tokio::process::ChildStdout) {
        let pending = self.pending.clone();
        let diagnostics = self.diagnostics.clone();
        let name = self.name.clone();
        let stdin = self.stdin.clone();
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            loop {
                let msg = match read_message(&mut reader).await {
                    Ok(msg) => msg,
                    Err(err) => {
                        tracing::debug!(server = %name, %err, "LSP reader stopped");
                        break;
                    }
                };
                if let Some(id) = msg.get("id").and_then(Value::as_i64)
                    && msg.get("method").is_none()
                {
                    if let Some(tx) = pending.lock().await.remove(&id) {
                        let _ = tx.send(msg.get("result").cloned().unwrap_or(Value::Null));
                    }
                    continue;
                }
                if msg.get("method").and_then(Value::as_str)
                    == Some("textDocument/publishDiagnostics")
                {
                    if let Some((path, diags)) = parse_publish_diagnostics(&name, &msg) {
                        diagnostics.write().await.insert(path, diags);
                    }
                    continue;
                }
                if let (Some(id), Some(method)) = (
                    msg.get("id").and_then(Value::as_i64),
                    msg.get("method").and_then(Value::as_str),
                ) {
                    let result = match method {
                        "workspace/configuration" => json!([]),
                        "client/registerCapability" => Value::Null,
                        _ => Value::Null,
                    };
                    let response = json!({"jsonrpc":"2.0","id":id,"result":result});
                    let _ = write_message(&mut *stdin.lock().await, &response).await;
                }
            }
        });
    }

    async fn initialize(&self) -> anyhow::Result<()> {
        let root_uri = file_uri(&self.root);
        let params = json!({
            "processId": std::process::id(),
            "clientInfo": {"name":"anvil", "version": env!("CARGO_PKG_VERSION")},
            "rootUri": root_uri,
            "rootPath": self.root,
            "workspaceFolders": [{"uri": root_uri, "name": self.root.file_name().and_then(|n| n.to_str()).unwrap_or("workspace")}],
            "capabilities": {
                "workspace": {
                    "configuration": true,
                    "didChangeWatchedFiles": {"dynamicRegistration": true}
                },
                "textDocument": {
                    "synchronization": {"dynamicRegistration": true, "didSave": true},
                    "publishDiagnostics": {"relatedInformation": true, "versionSupport": true}
                }
            }
        });
        let _ = self
            .call("initialize", params, Duration::from_secs(30))
            .await?;
        self.notify("initialized", json!({})).await?;
        Ok(())
    }

    async fn open_file(&self, path: &Path) -> anyhow::Result<()> {
        let path = path.to_path_buf();
        if self.open_files.lock().await.contains_key(&path) {
            return Ok(());
        }
        let text = tokio::fs::read_to_string(&path).await?;
        let params = json!({
            "textDocument": {
                "uri": file_uri(&path),
                "languageId": language_id(&path),
                "version": 1,
                "text": text
            }
        });
        self.notify("textDocument/didOpen", params).await?;
        self.open_files.lock().await.insert(path, 1);
        Ok(())
    }

    async fn change_file(&self, path: &Path) -> anyhow::Result<()> {
        let path = path.to_path_buf();
        if !self.open_files.lock().await.contains_key(&path) {
            return self.open_file(&path).await;
        }
        let text = tokio::fs::read_to_string(&path).await?;
        let version = {
            let mut open = self.open_files.lock().await;
            let version = open.entry(path.clone()).or_insert(1);
            *version += 1;
            *version
        };
        let params = json!({
            "textDocument": {"uri": file_uri(&path), "version": version},
            "contentChanges": [{"text": text}]
        });
        self.notify("textDocument/didChange", params).await
    }

    async fn diagnostics_for_file(&self, path: &Path) -> Vec<Diagnostic> {
        self.diagnostics
            .read()
            .await
            .get(path)
            .cloned()
            .unwrap_or_default()
    }

    async fn all_diagnostics(&self) -> Vec<Diagnostic> {
        self.diagnostics
            .read()
            .await
            .values()
            .flat_map(|v| v.iter().cloned())
            .collect()
    }

    async fn call(&self, method: &str, params: Value, timeout: Duration) -> anyhow::Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);
        let msg = json!({"jsonrpc":"2.0","id":id,"method":method,"params":params});
        if let Err(err) = write_message(&mut *self.stdin.lock().await, &msg).await {
            self.pending.lock().await.remove(&id);
            return Err(err);
        }
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(_)) => Err(anyhow::anyhow!("LSP response channel closed")),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err(anyhow::anyhow!("LSP request timed out: {method}"))
            }
        }
    }

    async fn notify(&self, method: &str, params: Value) -> anyhow::Result<()> {
        let msg = json!({"jsonrpc":"2.0","method":method,"params":params});
        write_message(&mut *self.stdin.lock().await, &msg).await
    }

    fn abort(&self) {
        if let Ok(mut child) = self.child.try_lock() {
            let _ = child.start_kill();
        }
    }
}

async fn write_message(writer: &mut ChildStdin, msg: &Value) -> anyhow::Result<()> {
    let bytes = serde_json::to_vec(msg)?;
    writer
        .write_all(format!("Content-Length: {}\r\n\r\n", bytes.len()).as_bytes())
        .await?;
    writer.write_all(&bytes).await?;
    writer.flush().await?;
    Ok(())
}

async fn read_message(
    reader: &mut BufReader<tokio::process::ChildStdout>,
) -> anyhow::Result<Value> {
    let mut content_length = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).await? == 0 {
            return Err(anyhow::anyhow!("LSP stdout closed"));
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some(value) = trimmed.strip_prefix("Content-Length:") {
            content_length = Some(value.trim().parse::<usize>()?);
        }
    }
    let len = content_length.ok_or_else(|| anyhow::anyhow!("missing LSP Content-Length"))?;
    let mut buf = vec![0; len];
    reader.read_exact(&mut buf).await?;
    Ok(serde_json::from_slice(&buf)?)
}

fn parse_publish_diagnostics(server_name: &str, msg: &Value) -> Option<(PathBuf, Vec<Diagnostic>)> {
    let params = msg.get("params")?;
    let path = path_from_file_uri(params.get("uri")?.as_str()?)?;
    let diagnostics = params
        .get("diagnostics")?
        .as_array()?
        .iter()
        .map(|diag| {
            let start = diag.get("range").and_then(|r| r.get("start"));
            let line = start
                .and_then(|s| s.get("line"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let character = start
                .and_then(|s| s.get("character"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let source = diag
                .get("source")
                .and_then(Value::as_str)
                .unwrap_or(server_name)
                .to_string();
            let code = diag.get("code").map(|code| {
                code.as_str()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| code.to_string())
            });
            Diagnostic {
                path: path.clone(),
                severity: DiagnosticSeverity::from_lsp(
                    diag.get("severity").and_then(Value::as_u64),
                ),
                line,
                character,
                source,
                code,
                message: diag
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            }
        })
        .collect();
    Some((path, diagnostics))
}

fn file_uri(path: &Path) -> String {
    format!("file://{}", path.display())
}

fn path_from_file_uri(uri: &str) -> Option<PathBuf> {
    uri.strip_prefix("file://").map(PathBuf::from)
}

fn language_id(path: &Path) -> &'static str {
    match path.extension().and_then(|ext| ext.to_str()).unwrap_or("") {
        "rs" => "rust",
        "go" => "go",
        "ts" => "typescript",
        "tsx" => "typescriptreact",
        "js" => "javascript",
        "jsx" => "javascriptreact",
        "py" => "python",
        "java" => "java",
        "c" => "c",
        "cc" | "cpp" | "cxx" => "cpp",
        "h" | "hpp" => "cpp",
        "cs" => "csharp",
        "rb" => "ruby",
        "php" => "php",
        "swift" => "swift",
        "kt" | "kts" => "kotlin",
        "lua" => "lua",
        "dart" => "dart",
        "ex" | "exs" => "elixir",
        "erl" | "hrl" => "erlang",
        "scala" => "scala",
        "sh" | "bash" | "zsh" => "shellscript",
        "json" => "json",
        "yaml" | "yml" => "yaml",
        "toml" => "toml",
        "md" => "markdown",
        _ => "plaintext",
    }
}
