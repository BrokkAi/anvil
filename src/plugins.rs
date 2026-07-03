//! Claude Code-format plugin discovery and native plugin management.
//!
//! A plugin is a directory with a `.claude-plugin/plugin.json` manifest
//! that can provide skills (`skills/<name>/SKILL.md`), subagents
//! (`agents/<name>.md`), and MCP servers (`.mcp.json`). Anvil consumes
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
/// are modelled; unknown fields (hooks, commands, marketplace metadata)
/// are ignored.
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
            Some(spec) => spec.iter().map(|p| self.resolve(p)).collect(),
            None => vec![self.root.join("skills")],
        };
        roots.into_iter().filter(|p| p.is_dir()).collect()
    }

    /// Agent sources: individual `.md` files or directories of them.
    /// Defaults to `agents/` under the plugin root.
    pub fn agent_sources(&self) -> Vec<PathBuf> {
        let sources: Vec<PathBuf> = match &self.manifest.agents {
            Some(spec) => spec.iter().map(|p| self.resolve(p)).collect(),
            None => vec![self.root.join("agents")],
        };
        sources.into_iter().filter(|p| p.exists()).collect()
    }

    /// Translate the plugin's MCP servers into Anvil's config model.
    /// Untranslatable entries are skipped with a message pushed to
    /// `diagnostics`.
    pub fn mcp_servers(&self, diagnostics: &mut Vec<String>) -> Vec<crate::mcp::McpServerConfig> {
        let map = match &self.manifest.mcp_servers {
            Some(McpServersSpec::Inline(map)) => map.clone(),
            Some(McpServersSpec::Path(rel)) => {
                let path = self.resolve(rel);
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
                let path = self.root.join(".mcp.json");
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
            Some(other) => return Err(format!("transport '{other}' is not supported (stdio only)")),
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

    /// Resolve a manifest-relative path against the plugin root.
    fn resolve(&self, rel: &str) -> PathBuf {
        let rel = self.substitute(rel);
        let path = Path::new(&rel);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            join_normalized(&self.root, path)
        }
    }
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

fn valid_server_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Discover all installed plugins (Claude Code + native). Disabled
/// plugins are included with `enabled: false` so `/plugin list` can show
/// them; skill/agent/MCP consumers filter via [`PluginCatalog::enabled`].
pub fn discover(home: Option<&Path>) -> PluginCatalog {
    let mut catalog = PluginCatalog::default();
    let native = read_native_registry();
    if let Some(home) = home {
        discover_claude_installs(home, &native.claude_overrides, &mut catalog);
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

    let enabled_map = read_enabled_plugins(&home.join(CLAUDE_DIR).join(SETTINGS_FILE));

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
    pub path: PathBuf,
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
/// registered plugin's manifest name.
pub fn register_native(source: &str, path: &Path) -> Result<String> {
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

        let catalog = discover(Some(home.path()));
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
        let catalog = discover(Some(home.path()));
        assert!(!catalog.plugins[0].enabled);
        assert_eq!(catalog.enabled().count(), 0);
    }

    #[test]
    fn discover_skips_broken_manifests_with_diagnostic() {
        let plugin = TempDir::new().unwrap();
        plugin_at(plugin.path(), "{not json");
        let home = claude_home("bad@mkt", plugin.path());

        let catalog = discover(Some(home.path()));
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

        let catalog = discover(Some(home.path()));
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
        assert_eq!(p.skill_roots(), vec![plugin.path().join("skills")]);
        assert_eq!(p.agent_sources(), vec![plugin.path().join("agents")]);

        // Explicit manifest entries, one missing on disk.
        plugin_at(
            plugin.path(),
            r#"{"name":"demo","skills":"./skills/","agents":["./agents/a.md","./agents/missing.md"]}"#,
        );
        let p = InstalledPlugin {
            manifest: load_manifest(plugin.path()).unwrap(),
            ..p
        };
        assert_eq!(p.agent_sources(), vec![plugin.path().join("agents/a.md")]);
    }

    #[test]
    fn mcp_translation_resolves_paths_and_substitutes_root() {
        let plugin = TempDir::new().unwrap();
        plugin_at(plugin.path(), r#"{"name":"demo","mcpServers":"./.mcp.json"}"#);
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

        let name = register_native("/some/source", plugin.path()).unwrap();
        assert_eq!(name, "local-demo");
        // Duplicate registration is rejected.
        assert!(register_native("/some/source", plugin.path()).is_err());

        let catalog = discover(None);
        assert_eq!(catalog.plugins.len(), 1);
        assert_eq!(catalog.plugins[0].source, PluginSource::Native);
        assert!(catalog.plugins[0].enabled);

        assert!(set_native_enabled("local-demo", false).unwrap());
        assert!(!discover(None).plugins[0].enabled);
        assert!(!set_native_enabled("nope", true).unwrap());

        let removed = remove_native("local-demo").unwrap().unwrap();
        assert_eq!(removed.name, "local-demo");
        assert!(discover(None).plugins.is_empty());
        assert!(remove_native("local-demo").unwrap().is_none());
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
        let catalog = discover(Some(home.path()));
        assert!(!catalog.plugins[0].enabled);
    }
}
