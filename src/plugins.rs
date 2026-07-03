//! Claude Code-format plugin discovery and native plugin management.
//!
//! A plugin is a directory with a `.claude-plugin/plugin.json` manifest
//! that can provide skills (`skills/<name>/SKILL.md`), subagents
//! (`agents/<name>.md`), slash commands (`commands/<name>.md`), hooks
//! (`hooks/hooks.json`), and MCP servers (`.mcp.json`). Anvil consumes
//! plugins from two sources:
//!
//!   1. **Claude Code installs** -- `~/.claude/plugins/installed_plugins.json`
//!      (version 2), filtered by the `enabledPlugins` map in
//!      `~/.claude/settings.json`. Anything installed with
//!      `claude plugin install` works in Anvil with no extra steps.
//!   2. **Native installs** -- recorded in `<config_home>/plugins.json`
//!      and managed by the `/plugin` slash command. Git installs are
//!      cloned under `<config_home>/plugins/`; local-path installs are
//!      referenced in place.
//!
//! Enabled state for Claude Code plugins can be overridden on the Anvil
//! side (`claudeOverrides` in `plugins.json`) without touching Claude
//! Code's own settings file, which Anvil never writes.
//!
//! Discovery is pure filesystem reads (a few small JSON files), cheap
//! enough to re-run on every registry build. Consumers integrate the
//! catalog at three points: `skills::discover` scans plugin skill roots
//! (lowest precedence, so user/project skills override), `agents::discover`
//! loads plugin agent files, and `session::effective_mcp_servers` merges
//! plugin MCP servers (user-configured servers and Anvil's managed
//! bifrost win on name collision).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const MANIFEST_DIR: &str = ".claude-plugin";
const MANIFEST_FILE: &str = "plugin.json";
const CLAUDE_DIR: &str = ".claude";
const PLUGINS_SUBDIR: &str = "plugins";
const INSTALLED_PLUGINS_FILE: &str = "installed_plugins.json";
const SETTINGS_FILE: &str = "settings.json";
const NATIVE_REGISTRY_FILE: &str = "plugins.json";
const PLUGIN_ROOT_VAR: &str = "${CLAUDE_PLUGIN_ROOT}";

/// Max size for any plugin JSON file we parse. These are small manifests;
/// reject pathological files instead of buffering them.
const MAX_JSON_BYTES: u64 = 1024 * 1024;

static NATIVE_WRITE_LOCK: Mutex<()> = Mutex::new(());

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

/// Parsed `.claude-plugin/plugin.json`. Only the fields Anvil consumes
/// are modelled; unknown fields (author, keywords, marketplace display
/// metadata) are ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct PluginManifest {
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    skills: Option<StringOrList>,
    #[serde(default)]
    agents: Option<StringOrList>,
    #[serde(default)]
    commands: Option<StringOrList>,
    #[serde(default)]
    hooks: Option<HooksSpec>,
    #[serde(default, rename = "mcpServers")]
    mcp_servers: Option<McpServersSpec>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum StringOrList {
    One(String),
    Many(Vec<String>),
}

impl StringOrList {
    fn iter(&self) -> impl Iterator<Item = &str> {
        match self {
            Self::One(s) => std::slice::from_ref(s),
            Self::Many(v) => v.as_slice(),
        }
        .iter()
        .map(String::as_str)
    }
}

/// `mcpServers` in the manifest is either a path to an `.mcp.json` file
/// or an inline `{ name: config }` map.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum McpServersSpec {
    Path(String),
    Inline(serde_json::Map<String, serde_json::Value>),
}

/// `hooks` in the manifest is either a path to a `hooks.json` file or
/// the inline equivalent of its contents.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum HooksSpec {
    Path(String),
    Inline(HooksFile),
}

/// Claude Code `hooks/hooks.json` schema: event name -> matcher groups.
#[derive(Debug, Clone, Default, Deserialize)]
struct HooksFile {
    #[serde(default)]
    hooks: HashMap<String, Vec<HookMatcherGroup>>,
}

#[derive(Debug, Clone, Deserialize)]
struct HookMatcherGroup {
    #[serde(default)]
    matcher: Option<String>,
    #[serde(default)]
    hooks: Vec<HookDef>,
}

#[derive(Debug, Clone, Deserialize)]
struct HookDef {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    command: String,
    /// Seconds; Claude Code's default is 60.
    #[serde(default)]
    timeout: Option<u64>,
}

/// One server entry in Claude Code `.mcp.json` format. Fields Anvil does
/// not support (`startup_timeout_sec`, `headers`, ...) are ignored.
#[derive(Debug, Deserialize)]
struct McpServerJson {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: HashMap<String, String>,
}

/// Load and validate a plugin manifest from a plugin root directory.
pub fn load_manifest(root: &Path) -> Result<PluginManifest> {
    let path = root.join(MANIFEST_DIR).join(MANIFEST_FILE);
    let raw = read_small_file(&path)?;
    let manifest: PluginManifest = serde_json::from_str(&raw)
        .with_context(|| format!("invalid plugin manifest at '{}'", path.display()))?;
    if manifest.name.trim().is_empty() {
        anyhow::bail!("plugin manifest at '{}' has empty `name`", path.display());
    }
    Ok(manifest)
}

fn read_small_file(path: &Path) -> Result<String> {
    let meta = std::fs::metadata(path)
        .with_context(|| format!("missing plugin file '{}'", path.display()))?;
    if meta.len() > MAX_JSON_BYTES {
        anyhow::bail!(
            "plugin file '{}' exceeds {MAX_JSON_BYTES} bytes",
            path.display()
        );
    }
    std::fs::read_to_string(path).with_context(|| format!("reading '{}'", path.display()))
}

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginSource {
    /// Installed by Claude Code; read-only from Anvil's perspective
    /// (enable/disable goes through `claudeOverrides`).
    ClaudeCode,
    /// Installed/registered by Anvil's `/plugin` command.
    Native,
}

#[derive(Debug, Clone)]
pub struct InstalledPlugin {
    /// Stable identifier: `name@marketplace` for Claude Code installs,
    /// the registered name for native installs.
    pub key: String,
    pub root: PathBuf,
    pub manifest: PluginManifest,
    pub source: PluginSource,
    pub enabled: bool,
}

#[derive(Debug, Default)]
pub struct PluginCatalog {
    pub plugins: Vec<InstalledPlugin>,
    pub diagnostics: Vec<String>,
}

impl PluginCatalog {
    pub fn enabled(&self) -> impl Iterator<Item = &InstalledPlugin> {
        self.plugins.iter().filter(|p| p.enabled)
    }

    /// All MCP servers provided by enabled plugins, in catalog order.
    /// Translation problems are logged (they also surface as skill/agent
    /// registry diagnostics via `discover`'s catalog diagnostics).
    pub fn mcp_servers(&self) -> Vec<crate::mcp::McpServerConfig> {
        let mut diagnostics = Vec::new();
        let servers = self
            .enabled()
            .flat_map(|p| p.mcp_servers(&mut diagnostics))
            .collect();
        for msg in diagnostics {
            tracing::warn!("{msg}");
        }
        servers
    }

    /// All hook commands from enabled plugins, in catalog order.
    pub fn hooks(&self) -> Vec<HookCommand> {
        let mut diagnostics = Vec::new();
        let hooks: Vec<HookCommand> = self
            .enabled()
            .flat_map(|p| p.hooks(&mut diagnostics))
            .collect();
        for msg in diagnostics {
            tracing::warn!("{msg}");
        }
        hooks
    }

    fn push_diagnostic(&mut self, msg: String) {
        tracing::warn!("{msg}");
        self.diagnostics.push(msg);
    }
}

impl InstalledPlugin {
    /// Directories to scan for `SKILL.md` files. Defaults to `skills/`
    /// under the plugin root when the manifest has no `skills` field.
    pub fn skill_roots(&self) -> Vec<PathBuf> {
        let roots: Vec<PathBuf> = match &self.manifest.skills {
            Some(spec) => spec.iter().filter_map(|p| self.resolve(p).ok()).collect(),
            None => self.resolve("skills").ok().into_iter().collect(),
        };
        roots.into_iter().filter(|p| p.is_dir()).collect()
    }

    /// Agent sources: individual `.md` files or directories of them.
    /// Defaults to `agents/` under the plugin root.
    pub fn agent_sources(&self) -> Vec<PathBuf> {
        let sources: Vec<PathBuf> = match &self.manifest.agents {
            Some(spec) => spec.iter().filter_map(|p| self.resolve(p).ok()).collect(),
            None => self.resolve("agents").ok().into_iter().collect(),
        };
        sources.into_iter().filter(|p| p.exists()).collect()
    }

    /// Slash-command prompt files: `(name, path)` pairs. Defaults to
    /// `commands/` under the plugin root; a file in a subdirectory is
    /// namespaced `subdir:name`, matching Claude Code's convention.
    pub fn command_files(&self) -> Vec<(String, PathBuf)> {
        let sources: Vec<PathBuf> = match &self.manifest.commands {
            Some(spec) => spec.iter().filter_map(|p| self.resolve(p).ok()).collect(),
            None => self.resolve("commands").ok().into_iter().collect(),
        };
        let mut out = Vec::new();
        for source in sources {
            if source.is_file() {
                if let Some(name) = command_name(&source, source.parent()) {
                    out.push((name, source));
                }
                continue;
            }
            if !source.is_dir() {
                continue;
            }
            let walker = walkdir::WalkDir::new(&source)
                .max_depth(3)
                .follow_links(false)
                .sort_by_file_name();
            for entry in walker.into_iter().filter_map(|e| e.ok()) {
                if !entry.file_type().is_file() {
                    continue;
                }
                if let Some(name) = command_name(entry.path(), Some(&source)) {
                    out.push((name, entry.path().to_path_buf()));
                }
            }
        }
        out
    }

    /// Hook commands declared by this plugin. The manifest `hooks` field
    /// (path or inline) wins; otherwise `hooks/hooks.json` is auto-loaded
    /// when present, mirroring the other component defaults.
    pub fn hooks(&self, diagnostics: &mut Vec<String>) -> Vec<HookCommand> {
        let file = match &self.manifest.hooks {
            Some(HooksSpec::Inline(file)) => file.clone(),
            Some(HooksSpec::Path(rel)) => {
                let path = match self.resolve(rel) {
                    Ok(path) => path,
                    Err(e) => {
                        diagnostics.push(format!("plugin '{}': {e}", self.key));
                        return Vec::new();
                    }
                };
                match self.read_hooks_file(&path) {
                    Ok(file) => file,
                    Err(e) => {
                        diagnostics.push(format!("plugin '{}': {e:#}", self.key));
                        return Vec::new();
                    }
                }
            }
            None => {
                let path = match self.resolve("hooks/hooks.json") {
                    Ok(path) => path,
                    Err(e) => {
                        diagnostics.push(format!("plugin '{}': {e}", self.key));
                        return Vec::new();
                    }
                };
                if !path.is_file() {
                    return Vec::new();
                }
                match self.read_hooks_file(&path) {
                    Ok(file) => file,
                    Err(e) => {
                        diagnostics.push(format!("plugin '{}': {e:#}", self.key));
                        return Vec::new();
                    }
                }
            }
        };

        let mut out = Vec::new();
        let mut events: Vec<&String> = file.hooks.keys().collect();
        events.sort();
        for event_name in events {
            let Some(event) = HookEvent::parse(event_name) else {
                diagnostics.push(format!(
                    "plugin '{}': hook event '{event_name}' is not supported by Anvil; ignoring",
                    self.key
                ));
                continue;
            };
            for group in &file.hooks[event_name] {
                for def in &group.hooks {
                    match def.kind.as_deref() {
                        None | Some("command") => {}
                        Some(other) => {
                            diagnostics.push(format!(
                                "plugin '{}': hook type '{other}' is not supported; ignoring",
                                self.key
                            ));
                            continue;
                        }
                    }
                    out.push(HookCommand {
                        plugin: self.key.clone(),
                        event,
                        matcher: group.matcher.clone().filter(|m| !m.trim().is_empty()),
                        command: self.substitute(&def.command),
                        timeout: std::time::Duration::from_secs(
                            def.timeout.unwrap_or(DEFAULT_HOOK_TIMEOUT_SECS),
                        ),
                    });
                }
            }
        }
        out
    }

    fn read_hooks_file(&self, path: &Path) -> Result<HooksFile> {
        let raw = read_small_file(path)?;
        serde_json::from_str(&raw)
            .with_context(|| format!("invalid hooks config at '{}'", path.display()))
    }

    /// Translate the plugin's MCP servers into Anvil's config model.
    /// Untranslatable entries are skipped with a message pushed to
    /// `diagnostics`.
    pub fn mcp_servers(&self, diagnostics: &mut Vec<String>) -> Vec<crate::mcp::McpServerConfig> {
        let map = match &self.manifest.mcp_servers {
            Some(McpServersSpec::Inline(map)) => map.clone(),
            Some(McpServersSpec::Path(rel)) => {
                let path = match self.resolve(rel) {
                    Ok(path) => path,
                    Err(e) => {
                        diagnostics.push(format!("plugin '{}': {e}", self.key));
                        return Vec::new();
                    }
                };
                match self.read_mcp_file(&path) {
                    Ok(map) => map,
                    Err(e) => {
                        diagnostics.push(format!("plugin '{}': {e:#}", self.key));
                        return Vec::new();
                    }
                }
            }
            // Claude Code auto-loads a root `.mcp.json` even without a
            // manifest field; mirror that.
            None => {
                let path = match self.resolve(".mcp.json") {
                    Ok(path) => path,
                    Err(e) => {
                        diagnostics.push(format!("plugin '{}': {e}", self.key));
                        return Vec::new();
                    }
                };
                if !path.is_file() {
                    return Vec::new();
                }
                match self.read_mcp_file(&path) {
                    Ok(map) => map,
                    Err(e) => {
                        diagnostics.push(format!("plugin '{}': {e:#}", self.key));
                        return Vec::new();
                    }
                }
            }
        };

        let mut out = Vec::new();
        for (name, value) in map {
            match self.translate_mcp_server(&name, value) {
                Ok(server) => out.push(server),
                Err(msg) => diagnostics.push(format!(
                    "plugin '{}': skipping MCP server '{name}': {msg}",
                    self.key
                )),
            }
        }
        out
    }

    fn read_mcp_file(&self, path: &Path) -> Result<serde_json::Map<String, serde_json::Value>> {
        let raw = read_small_file(path)?;
        let value: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("invalid MCP config at '{}'", path.display()))?;
        // Standard `.mcp.json` wraps servers in an `mcpServers` key; an
        // unwrapped `{ name: config }` map is accepted too.
        let map = match value {
            serde_json::Value::Object(mut obj) => match obj.remove("mcpServers") {
                Some(serde_json::Value::Object(inner)) => inner,
                Some(_) => anyhow::bail!(
                    "MCP config at '{}' has non-object `mcpServers`",
                    path.display()
                ),
                None => obj,
            },
            _ => anyhow::bail!("MCP config at '{}' is not a JSON object", path.display()),
        };
        Ok(map)
    }

    fn translate_mcp_server(
        &self,
        name: &str,
        value: serde_json::Value,
    ) -> std::result::Result<crate::mcp::McpServerConfig, String> {
        if !valid_server_name(name) {
            return Err("name may contain only letters, numbers, `_`, `-`, and `.`".into());
        }
        let parsed: McpServerJson =
            serde_json::from_value(value).map_err(|e| format!("invalid entry: {e}"))?;
        match parsed.kind.as_deref() {
            None | Some("stdio") => {}
            Some(other) => {
                return Err(format!("transport '{other}' is not supported (stdio only)"));
            }
        }
        let Some(command) = parsed.command.filter(|c| !c.trim().is_empty()) else {
            return Err("missing `command`".into());
        };
        let command = self.resolve_command(&self.substitute(&command));
        let args = parsed.args.iter().map(|a| self.substitute(a)).collect();
        let env = parsed
            .env
            .into_iter()
            .map(|(name, value)| crate::mcp::McpEnvVar {
                name,
                value: self.substitute(&value),
            })
            .collect();
        Ok(crate::mcp::McpServerConfig {
            name: name.to_string(),
            command,
            args,
            env,
            // Claude Code's MCP stdio transport is newline-delimited
            // JSON-RPC, so plugins written for it expect line framing.
            framing: crate::mcp::McpFraming::Line,
            enabled: true,
        })
    }

    /// Replace `${CLAUDE_PLUGIN_ROOT}` with the plugin root. This is the
    /// spec's portability variable for plugin-relative paths in MCP
    /// configs; other `${VAR}` forms pass through untouched.
    fn substitute(&self, value: &str) -> String {
        value.replace(PLUGIN_ROOT_VAR, &self.root.display().to_string())
    }

    /// Commands with an explicit path shape (`./bin/x`, `bin/x`) resolve
    /// against the plugin root; bare names (`node`, `npx`) resolve on
    /// PATH; absolute paths pass through.
    fn resolve_command(&self, command: &str) -> String {
        let path = Path::new(command);
        if path.is_absolute() || path.components().count() <= 1 {
            return command.to_string();
        }
        join_normalized(&self.root, path).display().to_string()
    }

    /// Resolve a manifest path against the plugin root. Relative paths
    /// must stay inside the root, and absolute paths are accepted only
    /// after `${CLAUDE_PLUGIN_ROOT}` substitution keeps them inside it.
    fn resolve(&self, rel: &str) -> std::result::Result<PathBuf, String> {
        let rel = self.substitute(rel);
        resolve_manifest_path(&self.root, &rel)
    }
}

fn resolve_manifest_path(root: &Path, rel: &str) -> std::result::Result<PathBuf, String> {
    let rel = rel.trim();
    if rel.is_empty() {
        return Err("manifest path must not be empty".to_string());
    }
    let path = Path::new(rel);
    if path
        .components()
        .any(|comp| matches!(comp, std::path::Component::ParentDir))
    {
        return Err(format!("manifest path `{rel}` must not contain `..`"));
    }

    let root = canonicalize_with_missing_tail(root)?;
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        join_normalized(&root, path)
    };
    let resolved = canonicalize_with_missing_tail(&candidate)?;
    if !resolved.starts_with(&root) {
        return Err(format!(
            "manifest path `{rel}` resolves outside '{}'",
            root.display()
        ));
    }
    Ok(resolved)
}

fn canonicalize_with_missing_tail(path: &Path) -> std::result::Result<PathBuf, String> {
    if path.exists() {
        return std::fs::canonicalize(path)
            .map_err(|e| format!("cannot resolve '{}': {e}", path.display()));
    }

    let mut ancestor = path;
    let mut tail = Vec::new();
    while !ancestor.exists() {
        let Some(name) = ancestor.file_name() else {
            return Err(format!(
                "path '{}' has no existing ancestor",
                path.display()
            ));
        };
        tail.push(name.to_owned());
        ancestor = ancestor
            .parent()
            .ok_or_else(|| format!("path '{}' has no existing ancestor", path.display()))?;
    }

    let mut resolved = std::fs::canonicalize(ancestor)
        .map_err(|e| format!("cannot resolve '{}': {e}", ancestor.display()))?;
    for name in tail.iter().rev() {
        resolved.push(name);
    }
    Ok(resolved)
}

/// Join dropping `.` segments (`root` + `./bin/x` -> `root/bin/x`) so
/// resolved paths render cleanly in configs and diagnostics. `..` is
/// preserved untouched.
fn join_normalized(root: &Path, rel: &Path) -> PathBuf {
    let mut out = root.to_path_buf();
    for comp in rel.components() {
        if !matches!(comp, std::path::Component::CurDir) {
            out.push(comp.as_os_str());
        }
    }
    out
}

/// Derive a slash-command name from a command file path: the `.md` stem,
/// prefixed with `subdir:` segments relative to the commands root
/// (`commands/git/commit.md` -> `git:commit`), matching Claude Code's
/// namespacing convention.
fn command_name(path: &Path, root: Option<&Path>) -> Option<String> {
    let is_md = path
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("md"));
    if !is_md {
        return None;
    }
    let stem = path.file_stem()?.to_str()?;
    if stem.is_empty() || stem.starts_with('.') {
        return None;
    }
    let mut segments: Vec<&str> = Vec::new();
    if let Some(rel) = root.and_then(|root| path.strip_prefix(root).ok())
        && let Some(parent) = rel.parent()
    {
        for comp in parent.components() {
            segments.push(comp.as_os_str().to_str()?);
        }
    }
    segments.push(stem);
    Some(segments.join(":"))
}

fn valid_server_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

const DEFAULT_HOOK_TIMEOUT_SECS: u64 = 60;

/// Hook events Anvil executes. Claude Code defines more (SessionStart,
/// Stop, PreCompact, ...); unsupported ones surface a diagnostic at
/// discovery so plugin authors aren't silently ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookEvent {
    PreToolUse,
    PostToolUse,
    UserPromptSubmit,
}

impl HookEvent {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "PreToolUse" => Some(Self::PreToolUse),
            "PostToolUse" => Some(Self::PostToolUse),
            "UserPromptSubmit" => Some(Self::UserPromptSubmit),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::PreToolUse => "PreToolUse",
            Self::PostToolUse => "PostToolUse",
            Self::UserPromptSubmit => "UserPromptSubmit",
        }
    }
}

/// One executable hook, flattened from a plugin's hooks config with
/// `${CLAUDE_PLUGIN_ROOT}` already substituted.
#[derive(Debug, Clone)]
pub struct HookCommand {
    pub plugin: String,
    pub event: HookEvent,
    /// Regex matched against the tool name for tool events; `None`
    /// matches everything. Prompt events ignore matchers.
    pub matcher: Option<String>,
    pub command: String,
    pub timeout: std::time::Duration,
}

/// Aggregate outcome of running the hooks for one event occurrence.
#[derive(Debug, Default)]
pub struct HookDecision {
    /// A hook exited with code 2: the operation should be blocked (for
    /// PreToolUse/UserPromptSubmit) or the feedback fed back to the
    /// model (PostToolUse). `reasons` carries the hooks' stderr.
    pub blocked: bool,
    pub reasons: Vec<String>,
    /// stdout from successful (exit 0) hooks, used as added context for
    /// UserPromptSubmit.
    pub context: Vec<String>,
}

/// Run every hook registered for `event` whose matcher accepts
/// `matcher_input`, feeding each the JSON `payload` on stdin. The
/// exit-code protocol is Claude Code's: 0 = success (stdout may add
/// context), 2 = block (stderr is the reason), anything else = warn and
/// continue. Hooks run sequentially; a blocking result does not stop
/// later hooks (their reasons aggregate).
pub async fn run_hooks(
    hooks: &[HookCommand],
    event: HookEvent,
    matcher_input: Option<&str>,
    payload: &serde_json::Value,
    cwd: &Path,
) -> HookDecision {
    let mut decision = HookDecision::default();
    let payload_bytes = payload.to_string();
    for hook in hooks.iter().filter(|h| h.event == event) {
        if let Some(matcher) = &hook.matcher {
            let Some(input) = matcher_input else { continue };
            match regex::Regex::new(matcher) {
                Ok(re) => {
                    if !re.is_match(input) {
                        continue;
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        plugin = %hook.plugin,
                        matcher = %matcher,
                        "invalid hook matcher regex; skipping hook: {e}"
                    );
                    continue;
                }
            }
        }
        match run_one_hook(hook, &payload_bytes, cwd).await {
            HookResult::Ok(stdout) => {
                if !stdout.trim().is_empty() {
                    decision.context.push(stdout.trim().to_string());
                }
            }
            HookResult::Block(stderr) => {
                decision.blocked = true;
                let reason = if stderr.trim().is_empty() {
                    format!("blocked by plugin '{}' hook", hook.plugin)
                } else {
                    stderr.trim().to_string()
                };
                decision.reasons.push(reason);
            }
            HookResult::Error(msg) => {
                tracing::warn!(plugin = %hook.plugin, event = event.name(), "{msg}");
            }
        }
    }
    decision
}

enum HookResult {
    Ok(String),
    Block(String),
    Error(String),
}

async fn run_one_hook(hook: &HookCommand, payload: &str, cwd: &Path) -> HookResult {
    use tokio::io::AsyncWriteExt;

    #[cfg(not(windows))]
    let mut command = {
        let mut c = tokio::process::Command::new("sh");
        c.arg("-c").arg(&hook.command);
        c
    };
    #[cfg(windows)]
    let mut command = {
        let mut c = tokio::process::Command::new("cmd");
        c.arg("/C").arg(&hook.command);
        c
    };

    let child = command
        .current_dir(cwd)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn();
    let mut child = match child {
        Ok(child) => child,
        Err(e) => return HookResult::Error(format!("hook failed to spawn: {e}")),
    };
    if let Some(mut stdin) = child.stdin.take() {
        // A hook that never reads stdin closes the pipe; that's fine.
        let _ = stdin.write_all(payload.as_bytes()).await;
        drop(stdin);
    }
    let output = match tokio::time::timeout(hook.timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => return HookResult::Error(format!("hook failed: {e}")),
        Err(_) => {
            return HookResult::Error(format!("hook timed out after {}s", hook.timeout.as_secs()));
        }
    };
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    match output.status.code() {
        Some(0) => HookResult::Ok(stdout),
        Some(2) => HookResult::Block(stderr),
        code => HookResult::Error(format!("hook exited with {:?}: {}", code, stderr.trim())),
    }
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Discover all installed plugins (Claude Code + native). Disabled
/// plugins are included with `enabled: false` so `/plugin list` can show
/// them; skill/agent/MCP consumers filter via [`PluginCatalog::enabled`].
///
/// `cwd` locates project-scope `enabledPlugins` overrides in the git
/// root's `.claude/settings.json` / `.claude/settings.local.json`;
/// `None` skips the project scope.
pub fn discover(cwd: Option<&Path>, home: Option<&Path>) -> PluginCatalog {
    let mut catalog = PluginCatalog::default();
    let native = read_native_registry();
    if let Some(home) = home {
        discover_claude_installs(cwd, home, &native.claude_overrides, &mut catalog);
    }
    discover_native(&native, &mut catalog);
    catalog
}

/// Schema of `~/.claude/plugins/installed_plugins.json` (version 2).
#[derive(Debug, Deserialize)]
struct InstalledPluginsFile {
    #[serde(default)]
    version: u32,
    #[serde(default)]
    plugins: HashMap<String, Vec<InstallRecord>>,
}

#[derive(Debug, Deserialize)]
struct InstallRecord {
    #[serde(rename = "installPath")]
    install_path: PathBuf,
}

fn discover_claude_installs(
    cwd: Option<&Path>,
    home: &Path,
    overrides: &HashMap<String, bool>,
    catalog: &mut PluginCatalog,
) {
    let plugins_dir = home.join(CLAUDE_DIR).join(PLUGINS_SUBDIR);
    let registry_path = plugins_dir.join(INSTALLED_PLUGINS_FILE);
    if !registry_path.is_file() {
        return;
    }
    let installed: InstalledPluginsFile = match read_small_file(&registry_path)
        .and_then(|raw| serde_json::from_str(&raw).map_err(anyhow::Error::from))
    {
        Ok(f) => f,
        Err(e) => {
            catalog.push_diagnostic(format!(
                "unreadable Claude Code plugin registry '{}': {e:#}",
                registry_path.display()
            ));
            return;
        }
    };
    if installed.version != 2 {
        catalog.push_diagnostic(format!(
            "Claude Code plugin registry '{}' has unsupported version {}; attempting to load anyway",
            registry_path.display(),
            installed.version
        ));
    }

    // Claude Code precedence: project settings.local.json > project
    // settings.json > user settings.json. Later reads overwrite earlier
    // keys.
    let mut enabled_map = read_enabled_plugins(&home.join(CLAUDE_DIR).join(SETTINGS_FILE));
    if let Some(project_root) = cwd.and_then(find_git_root) {
        let claude_dir = project_root.join(CLAUDE_DIR);
        for file in [SETTINGS_FILE, "settings.local.json"] {
            enabled_map.extend(read_enabled_plugins(&claude_dir.join(file)));
        }
    }

    let mut keys: Vec<&String> = installed.plugins.keys().collect();
    keys.sort();
    for key in keys {
        let records = &installed.plugins[key];
        let Some(record) = records.iter().find(|r| r.install_path.is_dir()) else {
            catalog.push_diagnostic(format!(
                "Claude Code plugin '{key}' has no existing install path; skipping"
            ));
            continue;
        };
        // A manifest is optional for Claude Code plugins (e.g. LSP-only
        // plugins ship none); the default directory conventions still
        // apply, and a plugin with none of them simply contributes
        // nothing Anvil consumes. Only a *broken* manifest is worth a
        // diagnostic.
        let manifest_file = record.install_path.join(MANIFEST_DIR).join(MANIFEST_FILE);
        let manifest = if manifest_file.is_file() {
            match load_manifest(&record.install_path) {
                Ok(m) => m,
                Err(e) => {
                    catalog.push_diagnostic(format!("Claude Code plugin '{key}': {e:#}"));
                    continue;
                }
            }
        } else {
            PluginManifest {
                name: key.split('@').next().unwrap_or(key).to_string(),
                version: None,
                description: None,
                skills: None,
                agents: None,
                commands: None,
                hooks: None,
                mcp_servers: None,
            }
        };
        // Anvil-side override beats Claude Code's own setting; a plugin
        // absent from `enabledPlugins` counts as enabled (installing it
        // was the opt-in).
        let enabled = overrides
            .get(key)
            .or_else(|| enabled_map.get(key))
            .copied()
            .unwrap_or(true);
        catalog.plugins.push(InstalledPlugin {
            key: key.clone(),
            root: record.install_path.clone(),
            manifest,
            source: PluginSource::ClaudeCode,
            enabled,
        });
    }
}

/// Locate the enclosing git root, mirroring the same helper in
/// `skills`/`agents` discovery (handles both `.git/` dirs and worktree
/// `gitdir:` files).
fn find_git_root(start: &Path) -> Option<PathBuf> {
    let mut cur = Some(start);
    while let Some(p) = cur {
        let marker = p.join(".git");
        let is_root = if marker.is_dir() {
            marker.join("HEAD").is_file()
        } else {
            marker.is_file()
                && std::fs::read_to_string(&marker).is_ok_and(|content| {
                    content
                        .lines()
                        .next()
                        .is_some_and(|line| line.starts_with("gitdir:"))
                })
        };
        if is_root {
            return Some(p.to_path_buf());
        }
        cur = p.parent();
    }
    None
}

/// Read the `enabledPlugins` map out of a Claude Code `settings.json`.
/// Missing file or field yields an empty map (all installed plugins
/// enabled).
fn read_enabled_plugins(settings_path: &Path) -> HashMap<String, bool> {
    let Ok(raw) = read_small_file(settings_path) else {
        return HashMap::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) else {
        tracing::warn!(
            path = %settings_path.display(),
            "unparseable Claude Code settings.json; treating all plugins as enabled"
        );
        return HashMap::new();
    };
    let Some(map) = value.get("enabledPlugins").and_then(|v| v.as_object()) else {
        return HashMap::new();
    };
    map.iter()
        .filter_map(|(k, v)| v.as_bool().map(|b| (k.clone(), b)))
        .collect()
}

fn discover_native(registry: &NativeRegistry, catalog: &mut PluginCatalog) {
    for entry in &registry.plugins {
        let manifest = match load_manifest(&entry.path) {
            Ok(m) => m,
            Err(e) => {
                catalog.push_diagnostic(format!("plugin '{}': {e:#}", entry.name));
                continue;
            }
        };
        catalog.plugins.push(InstalledPlugin {
            key: entry.name.clone(),
            root: entry.path.clone(),
            manifest,
            source: PluginSource::Native,
            enabled: entry.enabled,
        });
    }
}

// ---------------------------------------------------------------------------
// Native registry (`<config_home>/plugins.json`, managed by `/plugin`)
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct NativeRegistry {
    #[serde(default)]
    pub plugins: Vec<NativePluginEntry>,
    /// Anvil-side enable/disable overrides for Claude Code plugins,
    /// keyed by `name@marketplace`. Kept here so Anvil never writes to
    /// Claude Code's settings.json.
    #[serde(default, rename = "claudeOverrides")]
    pub claude_overrides: HashMap<String, bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativePluginEntry {
    pub name: String,
    /// What the user asked to install: a git URL or a local path.
    pub source: String,
    /// The plugin root (where `.claude-plugin/plugin.json` lives).
    pub path: PathBuf,
    /// Directory Anvil created for this install (a git clone); `path`
    /// may be a subdirectory of it for marketplace installs. `None` for
    /// local-path registrations, which are never deleted.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "installRoot"
    )]
    pub install_root: Option<PathBuf>,
    pub enabled: bool,
}

fn native_registry_path() -> Result<PathBuf> {
    Ok(crate::setup_state::config_home()?.join(NATIVE_REGISTRY_FILE))
}

/// Directory that `/plugin add <git-url>` clones into.
pub fn native_plugins_dir() -> Result<PathBuf> {
    Ok(crate::setup_state::config_home()?.join(PLUGINS_SUBDIR))
}

/// Missing or unreadable registry degrades to empty: plugin management
/// is a convenience layer, never a startup blocker.
pub fn read_native_registry() -> NativeRegistry {
    let Ok(path) = native_registry_path() else {
        return NativeRegistry::default();
    };
    let Ok(raw) = read_small_file(&path) else {
        return NativeRegistry::default();
    };
    match serde_json::from_str(&raw) {
        Ok(reg) => reg,
        Err(e) => {
            tracing::warn!(path = %path.display(), "unparseable native plugin registry: {e}");
            NativeRegistry::default()
        }
    }
}

pub fn write_native_registry(registry: &NativeRegistry) -> Result<()> {
    let _guard = NATIVE_WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let path = native_registry_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating '{}'", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(registry)?;
    std::fs::write(&path, json).with_context(|| format!("writing '{}'", path.display()))
}

/// Register a plugin rooted at `path`. Validates the manifest and
/// rejects name collisions with existing native entries. Returns the
/// registered plugin's manifest name. `install_root` is the clone
/// directory Anvil owns for this install, when there is one.
pub fn register_native(source: &str, path: &Path, install_root: Option<&Path>) -> Result<String> {
    let manifest = load_manifest(path)?;
    let name = manifest.name.clone();
    let mut registry = read_native_registry();
    if registry.plugins.iter().any(|p| p.name == name) {
        anyhow::bail!(
            "a native plugin named '{name}' is already registered; remove it first with `/plugin remove {name}`"
        );
    }
    registry.plugins.push(NativePluginEntry {
        name: name.clone(),
        source: source.to_string(),
        path: path.to_path_buf(),
        install_root: install_root.map(Path::to_path_buf),
        enabled: true,
    });
    write_native_registry(&registry)?;
    Ok(name)
}

/// Remove a native plugin. Returns its entry so the caller can clean up
/// a managed clone directory.
pub fn remove_native(name: &str) -> Result<Option<NativePluginEntry>> {
    let mut registry = read_native_registry();
    let Some(idx) = registry.plugins.iter().position(|p| p.name == name) else {
        return Ok(None);
    };
    let entry = registry.plugins.remove(idx);
    write_native_registry(&registry)?;
    Ok(Some(entry))
}

pub fn set_native_enabled(name: &str, enabled: bool) -> Result<bool> {
    let mut registry = read_native_registry();
    let Some(entry) = registry.plugins.iter_mut().find(|p| p.name == name) else {
        return Ok(false);
    };
    entry.enabled = enabled;
    write_native_registry(&registry)?;
    Ok(true)
}

pub fn set_claude_override(key: &str, enabled: bool) -> Result<()> {
    let mut registry = read_native_registry();
    registry.claude_overrides.insert(key.to_string(), enabled);
    write_native_registry(&registry)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Marketplaces
// ---------------------------------------------------------------------------

/// Parsed `.claude-plugin/marketplace.json`: a repository that lists
/// plugins rather than being one.
#[derive(Debug, Deserialize)]
pub struct Marketplace {
    pub name: String,
    #[serde(default)]
    pub plugins: Vec<MarketplaceEntry>,
}

#[derive(Debug, Deserialize)]
pub struct MarketplaceEntry {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub source: MarketplaceSource,
}

/// A marketplace entry's source: a path relative to the marketplace
/// repo, or a detailed object pointing at an external git location.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MarketplaceSource {
    Path(String),
    Detailed(MarketplaceSourceDetail),
}

#[derive(Debug, Deserialize)]
pub struct MarketplaceSourceDetail {
    /// `github`, `git-subdir`, `url`, ...
    pub source: String,
    #[serde(default)]
    pub url: Option<String>,
    /// `owner/repo`, for `source: github`.
    #[serde(default)]
    pub repo: Option<String>,
    /// Subdirectory within the repo, for `source: git-subdir`.
    #[serde(default)]
    pub path: Option<String>,
}

/// Load a repo's marketplace listing, if it has one.
pub fn load_marketplace(root: &Path) -> Result<Option<Marketplace>> {
    let path = root.join(MANIFEST_DIR).join("marketplace.json");
    if !path.is_file() {
        return Ok(None);
    }
    let raw = read_small_file(&path)?;
    let marketplace = serde_json::from_str(&raw)
        .with_context(|| format!("invalid marketplace listing at '{}'", path.display()))?;
    Ok(Some(marketplace))
}

/// Whether a repo root is a plugin (has a plugin manifest).
pub fn is_plugin_root(root: &Path) -> bool {
    root.join(MANIFEST_DIR).join(MANIFEST_FILE).is_file()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }

    fn plugin_at(root: &Path, manifest: &str) {
        write(&root.join(MANIFEST_DIR).join(MANIFEST_FILE), manifest);
    }

    /// A home dir with one Claude Code-installed plugin at `root`.
    fn claude_home(plugin_key: &str, root: &Path) -> TempDir {
        let home = TempDir::new().unwrap();
        write(
            &home
                .path()
                .join(CLAUDE_DIR)
                .join(PLUGINS_SUBDIR)
                .join(INSTALLED_PLUGINS_FILE),
            &format!(
                r#"{{"version":2,"plugins":{{"{plugin_key}":[{{"scope":"user","installPath":{},"version":"1.0.0"}}]}}}}"#,
                serde_json::to_string(&root.display().to_string()).unwrap()
            ),
        );
        home
    }

    #[test]
    fn manifest_parses_string_and_list_forms() {
        let m: PluginManifest = serde_json::from_str(
            r#"{"name":"p","skills":"./skills/","agents":["./agents/a.md"],"mcpServers":"./.mcp.json"}"#,
        )
        .unwrap();
        assert_eq!(m.skills.as_ref().unwrap().iter().count(), 1);
        assert_eq!(
            m.agents.as_ref().unwrap().iter().next().unwrap(),
            "./agents/a.md"
        );
        assert!(matches!(m.mcp_servers, Some(McpServersSpec::Path(_))));

        let m: PluginManifest = serde_json::from_str(
            r#"{"name":"p","skills":["./a/","./b/"],"mcpServers":{"srv":{"command":"x"}}}"#,
        )
        .unwrap();
        assert_eq!(m.skills.as_ref().unwrap().iter().count(), 2);
        assert!(matches!(m.mcp_servers, Some(McpServersSpec::Inline(_))));
    }

    #[test]
    fn discover_reads_claude_installs_and_respects_enabled_map() {
        let plugin = TempDir::new().unwrap();
        plugin_at(plugin.path(), r#"{"name":"demo","version":"1.0.0"}"#);
        let home = claude_home("demo@mkt", plugin.path());

        let catalog = discover(None, Some(home.path()));
        assert_eq!(catalog.plugins.len(), 1);
        let p = &catalog.plugins[0];
        assert_eq!(p.key, "demo@mkt");
        assert_eq!(p.manifest.name, "demo");
        assert!(p.enabled, "absent from enabledPlugins means enabled");
        assert_eq!(p.source, PluginSource::ClaudeCode);

        // Explicitly disabled in Claude Code settings.
        write(
            &home.path().join(CLAUDE_DIR).join(SETTINGS_FILE),
            r#"{"enabledPlugins":{"demo@mkt":false}}"#,
        );
        let catalog = discover(None, Some(home.path()));
        assert!(!catalog.plugins[0].enabled);
        assert_eq!(catalog.enabled().count(), 0);
    }

    #[test]
    fn discover_skips_broken_manifests_with_diagnostic() {
        let plugin = TempDir::new().unwrap();
        plugin_at(plugin.path(), "{not json");
        let home = claude_home("bad@mkt", plugin.path());

        let catalog = discover(None, Some(home.path()));
        assert!(catalog.plugins.is_empty());
        assert!(
            catalog.diagnostics.iter().any(|d| d.contains("bad@mkt")),
            "expected diagnostic, got {:?}",
            catalog.diagnostics
        );
    }

    #[test]
    fn manifest_less_claude_plugin_loads_quietly_and_provides_nothing() {
        let plugin = TempDir::new().unwrap();
        let home = claude_home("lsp-only@mkt", plugin.path());

        let catalog = discover(None, Some(home.path()));
        assert!(catalog.diagnostics.is_empty(), "{:?}", catalog.diagnostics);
        assert_eq!(catalog.plugins.len(), 1);
        let p = &catalog.plugins[0];
        assert_eq!(p.manifest.name, "lsp-only");
        assert!(p.skill_roots().is_empty());
        assert!(p.agent_sources().is_empty());
        let mut diags = Vec::new();
        assert!(p.mcp_servers(&mut diags).is_empty());
    }

    #[test]
    fn skill_roots_and_agent_sources_default_and_explicit() {
        let plugin = TempDir::new().unwrap();
        plugin_at(plugin.path(), r#"{"name":"demo"}"#);
        fs::create_dir_all(plugin.path().join("skills").join("s1")).unwrap();
        fs::create_dir_all(plugin.path().join("agents")).unwrap();
        write(&plugin.path().join("agents").join("a.md"), "x");

        let p = InstalledPlugin {
            key: "demo".into(),
            root: plugin.path().to_path_buf(),
            manifest: load_manifest(plugin.path()).unwrap(),
            source: PluginSource::Native,
            enabled: true,
        };
        assert_eq!(
            p.skill_roots(),
            vec![plugin.path().join("skills").canonicalize().unwrap()]
        );
        assert_eq!(
            p.agent_sources(),
            vec![plugin.path().join("agents").canonicalize().unwrap()]
        );

        // Explicit manifest entries, one missing on disk.
        plugin_at(
            plugin.path(),
            r#"{"name":"demo","skills":"./skills/","agents":["./agents/a.md","./agents/missing.md"]}"#,
        );
        let p = InstalledPlugin {
            manifest: load_manifest(plugin.path()).unwrap(),
            ..p
        };
        assert_eq!(
            p.agent_sources(),
            vec![plugin.path().join("agents/a.md").canonicalize().unwrap()]
        );
    }

    #[test]
    fn manifest_paths_do_not_escape_plugin_root() {
        let plugin = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        fs::create_dir_all(plugin.path().join("skills")).unwrap();
        fs::create_dir_all(outside.path().join("skills")).unwrap();
        plugin_at(
            plugin.path(),
            &format!(
                r#"{{"name":"demo","skills":["../outside",{}],"mcpServers":"../mcp.json"}}"#,
                serde_json::to_string(&outside.path().join("skills").display().to_string())
                    .unwrap()
            ),
        );

        let p = InstalledPlugin {
            key: "demo".into(),
            root: plugin.path().to_path_buf(),
            manifest: load_manifest(plugin.path()).unwrap(),
            source: PluginSource::Native,
            enabled: true,
        };
        assert!(p.skill_roots().is_empty());

        let mut diags = Vec::new();
        assert!(p.mcp_servers(&mut diags).is_empty());
        assert!(
            diags.iter().any(|d| d.contains("must not contain `..`")),
            "diags: {diags:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn manifest_paths_reject_symlink_escape() {
        let plugin = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        fs::create_dir_all(outside.path().join("skills")).unwrap();
        std::os::unix::fs::symlink(outside.path(), plugin.path().join("escape")).unwrap();
        plugin_at(plugin.path(), r#"{"name":"demo","skills":"escape/skills"}"#);

        let p = InstalledPlugin {
            key: "demo".into(),
            root: plugin.path().to_path_buf(),
            manifest: load_manifest(plugin.path()).unwrap(),
            source: PluginSource::Native,
            enabled: true,
        };
        assert!(p.skill_roots().is_empty());
    }

    #[test]
    fn mcp_translation_resolves_paths_and_substitutes_root() {
        let plugin = TempDir::new().unwrap();
        plugin_at(
            plugin.path(),
            r#"{"name":"demo","mcpServers":"./.mcp.json"}"#,
        );
        write(
            &plugin.path().join(".mcp.json"),
            r#"{"mcpServers":{"srv":{
                "type":"stdio",
                "command":"./bin/launch.sh",
                "args":["--root","${CLAUDE_PLUGIN_ROOT}/data","{cwd}"],
                "env":{"PLUGIN_HOME":"${CLAUDE_PLUGIN_ROOT}"},
                "startup_timeout_sec":60
            }}}"#,
        );
        let p = InstalledPlugin {
            key: "demo".into(),
            root: plugin.path().to_path_buf(),
            manifest: load_manifest(plugin.path()).unwrap(),
            source: PluginSource::Native,
            enabled: true,
        };
        let mut diags = Vec::new();
        let servers = p.mcp_servers(&mut diags);
        assert!(diags.is_empty(), "diags: {diags:?}");
        assert_eq!(servers.len(), 1);
        let s = &servers[0];
        assert_eq!(s.name, "srv");
        assert_eq!(
            s.command,
            plugin.path().join("bin/launch.sh").display().to_string()
        );
        assert_eq!(s.args[1], format!("{}/data", plugin.path().display()));
        // Anvil's own `{cwd}` placeholder must survive untouched.
        assert_eq!(s.args[2], "{cwd}");
        assert_eq!(s.env[0].value, plugin.path().display().to_string());
        assert_eq!(s.framing, crate::mcp::McpFraming::Line);
        assert!(s.enabled);
    }

    #[test]
    fn mcp_translation_skips_unsupported_transports_and_bad_names() {
        let plugin = TempDir::new().unwrap();
        plugin_at(
            plugin.path(),
            r#"{"name":"demo","mcpServers":{
                "http-srv":{"type":"http","url":"https://example.com"},
                "bad name!":{"command":"x"},
                "bare":{"command":"node","args":["server.js"]}
            }}"#,
        );
        let p = InstalledPlugin {
            key: "demo".into(),
            root: plugin.path().to_path_buf(),
            manifest: load_manifest(plugin.path()).unwrap(),
            source: PluginSource::Native,
            enabled: true,
        };
        let mut diags = Vec::new();
        let servers = p.mcp_servers(&mut diags);
        assert_eq!(servers.len(), 1);
        // Bare command names resolve on PATH, not the plugin root.
        assert_eq!(servers[0].command, "node");
        assert_eq!(diags.len(), 2, "diags: {diags:?}");
    }

    #[test]
    fn root_mcp_json_autoloads_without_manifest_field() {
        let plugin = TempDir::new().unwrap();
        plugin_at(plugin.path(), r#"{"name":"demo"}"#);
        write(
            &plugin.path().join(".mcp.json"),
            r#"{"mcpServers":{"srv":{"command":"tool"}}}"#,
        );
        let p = InstalledPlugin {
            key: "demo".into(),
            root: plugin.path().to_path_buf(),
            manifest: load_manifest(plugin.path()).unwrap(),
            source: PluginSource::Native,
            enabled: true,
        };
        let mut diags = Vec::new();
        assert_eq!(p.mcp_servers(&mut diags).len(), 1);
    }

    #[test]
    fn native_registry_roundtrip_and_management() {
        let config = TempDir::new().unwrap();
        let _scope = crate::setup_state::TestConfigHomeScope::set(config.path().to_path_buf());

        let plugin = TempDir::new().unwrap();
        plugin_at(plugin.path(), r#"{"name":"local-demo"}"#);

        let name = register_native("/some/source", plugin.path(), None).unwrap();
        assert_eq!(name, "local-demo");
        // Duplicate registration is rejected.
        assert!(register_native("/some/source", plugin.path(), None).is_err());

        let catalog = discover(None, None);
        assert_eq!(catalog.plugins.len(), 1);
        assert_eq!(catalog.plugins[0].source, PluginSource::Native);
        assert!(catalog.plugins[0].enabled);

        assert!(set_native_enabled("local-demo", false).unwrap());
        assert!(!discover(None, None).plugins[0].enabled);
        assert!(!set_native_enabled("nope", true).unwrap());

        let removed = remove_native("local-demo").unwrap().unwrap();
        assert_eq!(removed.name, "local-demo");
        assert!(discover(None, None).plugins.is_empty());
        assert!(remove_native("local-demo").unwrap().is_none());
    }

    #[test]
    fn command_files_default_dir_and_nested_namespacing() {
        let plugin = TempDir::new().unwrap();
        plugin_at(plugin.path(), r#"{"name":"demo"}"#);
        write(&plugin.path().join("commands").join("deploy.md"), "Deploy");
        write(
            &plugin.path().join("commands").join("git").join("commit.md"),
            "Commit",
        );
        write(&plugin.path().join("commands").join("notes.txt"), "x");

        let p = InstalledPlugin {
            key: "demo".into(),
            root: plugin.path().to_path_buf(),
            manifest: load_manifest(plugin.path()).unwrap(),
            source: PluginSource::Native,
            enabled: true,
        };
        let names: Vec<String> = p.command_files().into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, vec!["deploy".to_string(), "git:commit".to_string()]);
    }

    #[test]
    fn hooks_autoload_parse_and_diagnostics() {
        let plugin = TempDir::new().unwrap();
        plugin_at(plugin.path(), r#"{"name":"demo"}"#);
        write(
            &plugin.path().join("hooks").join("hooks.json"),
            r#"{"hooks":{
                "PreToolUse":[{"matcher":"run_shell_command","hooks":[
                    {"type":"command","command":"${CLAUDE_PLUGIN_ROOT}/check.sh","timeout":5},
                    {"type":"prompt","command":"unsupported-kind"}
                ]}],
                "SessionStart":[{"hooks":[{"command":"echo hi"}]}],
                "UserPromptSubmit":[{"hooks":[{"command":"validate"}]}]
            }}"#,
        );
        let p = InstalledPlugin {
            key: "demo".into(),
            root: plugin.path().to_path_buf(),
            manifest: load_manifest(plugin.path()).unwrap(),
            source: PluginSource::Native,
            enabled: true,
        };
        let mut diags = Vec::new();
        let hooks = p.hooks(&mut diags);
        assert_eq!(hooks.len(), 2, "hooks: {hooks:?}");
        let pre = hooks
            .iter()
            .find(|h| h.event == HookEvent::PreToolUse)
            .unwrap();
        assert_eq!(pre.matcher.as_deref(), Some("run_shell_command"));
        assert_eq!(pre.command, format!("{}/check.sh", plugin.path().display()));
        assert_eq!(pre.timeout, std::time::Duration::from_secs(5));
        let prompt = hooks
            .iter()
            .find(|h| h.event == HookEvent::UserPromptSubmit)
            .unwrap();
        assert_eq!(prompt.timeout, std::time::Duration::from_secs(60));
        assert!(
            diags.iter().any(|d| d.contains("SessionStart")),
            "unsupported event diagnostic missing: {diags:?}"
        );
        assert!(
            diags.iter().any(|d| d.contains("'prompt'")),
            "unsupported hook type diagnostic missing: {diags:?}"
        );
    }

    #[cfg(unix)]
    fn hook(event: HookEvent, matcher: Option<&str>, command: &str) -> HookCommand {
        HookCommand {
            plugin: "demo".into(),
            event,
            matcher: matcher.map(str::to_string),
            command: command.to_string(),
            timeout: std::time::Duration::from_secs(10),
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_hooks_exit_codes_and_matchers() {
        let cwd = TempDir::new().unwrap();
        let hooks = vec![
            // Matches: echoes context from stdin payload.
            hook(
                HookEvent::PreToolUse,
                Some("run_shell.*"),
                "cat > /dev/null; echo seen-by-hook",
            ),
            // Doesn't match this tool name.
            hook(HookEvent::PreToolUse, Some("^edit_file$"), "exit 2"),
            // Wrong event.
            hook(HookEvent::PostToolUse, None, "exit 2"),
        ];
        let payload = serde_json::json!({"tool_name": "run_shell_command"});
        let decision = run_hooks(
            &hooks,
            HookEvent::PreToolUse,
            Some("run_shell_command"),
            &payload,
            cwd.path(),
        )
        .await;
        assert!(!decision.blocked);
        assert_eq!(decision.context, vec!["seen-by-hook".to_string()]);

        // Blocking hook: exit 2 with stderr reason.
        let hooks = vec![hook(
            HookEvent::PreToolUse,
            None,
            "echo 'nope, not allowed' >&2; exit 2",
        )];
        let decision = run_hooks(
            &hooks,
            HookEvent::PreToolUse,
            Some("anything"),
            &payload,
            cwd.path(),
        )
        .await;
        assert!(decision.blocked);
        assert_eq!(decision.reasons, vec!["nope, not allowed".to_string()]);

        // Non-0/2 exit codes are non-blocking errors.
        let hooks = vec![hook(HookEvent::PreToolUse, None, "exit 1")];
        let decision = run_hooks(
            &hooks,
            HookEvent::PreToolUse,
            Some("anything"),
            &payload,
            cwd.path(),
        )
        .await;
        assert!(!decision.blocked);
        assert!(decision.context.is_empty());

        // Hooks with matchers are skipped for events with no matcher
        // input (prompt events); matcher-less hooks still run.
        let hooks = vec![
            hook(HookEvent::UserPromptSubmit, Some(".*"), "exit 2"),
            hook(HookEvent::UserPromptSubmit, None, "echo prompt-ctx"),
        ];
        let decision = run_hooks(
            &hooks,
            HookEvent::UserPromptSubmit,
            None,
            &payload,
            cwd.path(),
        )
        .await;
        assert!(!decision.blocked);
        assert_eq!(decision.context, vec!["prompt-ctx".to_string()]);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_hooks_timeout_is_nonblocking() {
        let cwd = TempDir::new().unwrap();
        let hooks = vec![HookCommand {
            plugin: "demo".into(),
            event: HookEvent::PreToolUse,
            matcher: None,
            command: "sleep 5".into(),
            timeout: std::time::Duration::from_millis(50),
        }];
        let payload = serde_json::json!({});
        let decision = run_hooks(
            &hooks,
            HookEvent::PreToolUse,
            Some("x"),
            &payload,
            cwd.path(),
        )
        .await;
        assert!(!decision.blocked);
    }

    #[test]
    fn project_settings_override_user_enabled_map() {
        let plugin = TempDir::new().unwrap();
        plugin_at(plugin.path(), r#"{"name":"demo"}"#);
        let home = claude_home("demo@mkt", plugin.path());
        write(
            &home.path().join(CLAUDE_DIR).join(SETTINGS_FILE),
            r#"{"enabledPlugins":{"demo@mkt":true}}"#,
        );

        let project = TempDir::new().unwrap();
        fs::create_dir_all(project.path().join(".git")).unwrap();
        fs::write(
            project.path().join(".git").join("HEAD"),
            "ref: refs/heads/main\n",
        )
        .unwrap();
        write(
            &project.path().join(CLAUDE_DIR).join(SETTINGS_FILE),
            r#"{"enabledPlugins":{"demo@mkt":false}}"#,
        );

        let catalog = discover(Some(project.path()), Some(home.path()));
        assert!(!catalog.plugins[0].enabled, "project setting should win");

        // settings.local.json wins over settings.json.
        write(
            &project.path().join(CLAUDE_DIR).join("settings.local.json"),
            r#"{"enabledPlugins":{"demo@mkt":true}}"#,
        );
        let catalog = discover(Some(project.path()), Some(home.path()));
        assert!(catalog.plugins[0].enabled, "local setting should win");
    }

    #[test]
    fn marketplace_listing_parses_both_source_forms() {
        let repo = TempDir::new().unwrap();
        write(
            &repo.path().join(MANIFEST_DIR).join("marketplace.json"),
            r#"{"name":"mkt","plugins":[
                {"name":"local","source":"./plugins/local","description":"in-repo"},
                {"name":"external","source":{"source":"git-subdir","url":"https://example.com/r.git","path":"plugins/x"}},
                {"name":"gh","source":{"source":"github","repo":"owner/repo"}}
            ]}"#,
        );
        assert!(!is_plugin_root(repo.path()));
        let marketplace = load_marketplace(repo.path()).unwrap().unwrap();
        assert_eq!(marketplace.name, "mkt");
        assert_eq!(marketplace.plugins.len(), 3);
        assert!(matches!(
            marketplace.plugins[0].source,
            MarketplaceSource::Path(_)
        ));
        match &marketplace.plugins[1].source {
            MarketplaceSource::Detailed(d) => {
                assert_eq!(d.source, "git-subdir");
                assert_eq!(d.path.as_deref(), Some("plugins/x"));
            }
            other => panic!("expected detailed source, got {other:?}"),
        }
        // A plugin repo is not a marketplace.
        let plugin = TempDir::new().unwrap();
        plugin_at(plugin.path(), r#"{"name":"p"}"#);
        assert!(is_plugin_root(plugin.path()));
        assert!(load_marketplace(plugin.path()).unwrap().is_none());
    }

    #[test]
    fn claude_override_beats_settings_enabled_map() {
        let config = TempDir::new().unwrap();
        let _scope = crate::setup_state::TestConfigHomeScope::set(config.path().to_path_buf());

        let plugin = TempDir::new().unwrap();
        plugin_at(plugin.path(), r#"{"name":"demo"}"#);
        let home = claude_home("demo@mkt", plugin.path());
        write(
            &home.path().join(CLAUDE_DIR).join(SETTINGS_FILE),
            r#"{"enabledPlugins":{"demo@mkt":true}}"#,
        );

        set_claude_override("demo@mkt", false).unwrap();
        let catalog = discover(None, Some(home.path()));
        assert!(!catalog.plugins[0].enabled);
    }
}
