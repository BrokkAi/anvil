//! Subagent (`<name>.md`) discovery for the `task` meta-tool.
//!
//! A subagent is a markdown file with YAML frontmatter (`name`,
//! `description`) that the parent LLM can delegate a focused task to via
//! the `task` tool. The subagent runs in an isolated `tool_loop::run`
//! invocation (silent notifications, fresh `Vec<ChatMessage>`) and only
//! its final assistant text is returned to the parent.
//!
//! Discovery mirrors [`crate::skills`] but with two differences:
//!
//!   * Layout is **flat**: subagents live as `<root>/<name>.md`, not as
//!     `<root>/<name>/SKILL.md`. This matches the Claude Code convention
//!     so existing `.claude/agents/foo.md` files work as-is.
//!   * Scan depth is 1 (one `read_dir` per root, no recursion). Subagents
//!     do not bundle resources.
//!
//! Scan order, **last-wins** like skills:
//!
//!   1. `~/.claude/agents/`                       (user, Claude compat)
//!   2. `~/.agents/agents/`                       (user, cross-client)
//!   3. `<git-root walk down to cwd>/.claude/agents/` (project, Claude compat)
//!   4. `<git-root walk down to cwd>/.agents/agents/` (project, cross-client)
//!
//! Pure module; no LLM/session deps.

use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use brokk_acp_sandbox::split_frontmatter;

/// Per-root cap on directory entries scanned. Mirrors skills.rs.
const MAX_ENTRIES_PER_ROOT: usize = 2000;

/// Hard cap on body size for a single agent file. Generous headroom over
/// the spec's "< 5000 tokens" suggestion.
const MAX_BODY_BYTES: usize = 256 * 1024;

const AGENT_FILE_EXT: &str = "md";
const AGENTS_DIR: &str = ".agents";
const CLAUDE_DIR: &str = ".claude";
const AGENTS_SUBDIR: &str = "agents";

/// Discovered subagent metadata. The body is loaded on demand by the
/// `task` dispatch path, not eagerly, so a session with 30 subagents
/// doesn't pay the I/O cost upfront.
#[derive(Debug, Clone)]
pub struct AgentMeta {
    pub name: String,
    pub description: String,
    pub location: PathBuf,
    #[allow(dead_code)]
    pub scope: AgentScope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentScope {
    User,
    Project,
}

/// In-memory registry keyed by `name`; last-wins on insert. Diagnostics
/// are stashed for `/context` to surface without spamming the LLM
/// catalog.
#[derive(Debug, Default, Clone)]
pub struct AgentRegistry {
    by_name: HashMap<String, AgentMeta>,
    diagnostics: Vec<String>,
}

impl AgentRegistry {
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    pub fn get(&self, name: &str) -> Option<&AgentMeta> {
        self.by_name.get(name)
    }

    /// Stable-ordered iterator over discovered subagents (sorted by name)
    /// so the catalog presented to the LLM is deterministic.
    pub fn iter_sorted(&self) -> impl Iterator<Item = &AgentMeta> {
        let mut v: Vec<&AgentMeta> = self.by_name.values().collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v.into_iter()
    }

    #[cfg(test)]
    pub fn diagnostics(&self) -> &[String] {
        &self.diagnostics
    }

    #[cfg(test)]
    pub fn insert_for_test(&mut self, meta: AgentMeta) {
        self.by_name.insert(meta.name.clone(), meta);
    }

    fn add(&mut self, meta: AgentMeta) {
        if let Some(prev) = self.by_name.get(&meta.name) {
            let msg = format!(
                "duplicate subagent '{}': '{}' shadowed by '{}'",
                meta.name,
                prev.location.display(),
                meta.location.display(),
            );
            tracing::warn!("{msg}");
            self.diagnostics.push(msg);
        }
        self.by_name.insert(meta.name.clone(), meta);
    }

    fn push_diagnostic(&mut self, msg: String) {
        tracing::warn!("{msg}");
        self.diagnostics.push(msg);
    }
}

/// Discover all subagent `*.md` files reachable from `cwd` and the user
/// home directory. Returns an empty registry when nothing is found.
pub fn discover(cwd: &Path) -> AgentRegistry {
    let home = dirs::home_dir();
    discover_with_backend(cwd, home.as_deref(), crate::sandbox_backend::global())
}

pub fn discover_with_sandbox_mode(
    cwd: &Path,
    mode: Option<crate::sandbox_backend::SandboxMode>,
) -> AgentRegistry {
    let home = dirs::home_dir();
    match crate::sandbox_backend::backend_for_mode(mode) {
        Ok(backend) => discover_with_backend(cwd, home.as_deref(), &backend),
        Err(e) => {
            let mut reg = AgentRegistry::default();
            reg.push_diagnostic(format!(
                "failed to initialize sandbox backend for subagent discovery: {e}"
            ));
            reg
        }
    }
}

#[cfg(test)]
fn discover_inner(cwd: &Path, home: Option<&Path>) -> AgentRegistry {
    discover_with_backend(cwd, home, crate::sandbox_backend::global())
}

fn discover_with_backend(
    cwd: &Path,
    home: Option<&Path>,
    backend: &crate::sandbox_backend::SandboxBackend,
) -> AgentRegistry {
    let cwd = normalize_path(cwd);
    let mut reg = AgentRegistry::default();

    // 1+2. User scope.
    if let Some(h) = home {
        scan_root(
            &h.join(CLAUDE_DIR).join(AGENTS_SUBDIR),
            AgentScope::User,
            &mut reg,
            backend,
        );
        scan_root(
            &h.join(AGENTS_DIR).join(AGENTS_SUBDIR),
            AgentScope::User,
            &mut reg,
            backend,
        );
    }

    // 3+4. Project scope: walk from git root down to cwd.
    let git_root = find_git_root(&cwd);
    for dir in build_dir_chain(&cwd, git_root.as_deref()) {
        scan_root(
            &dir.join(CLAUDE_DIR).join(AGENTS_SUBDIR),
            AgentScope::Project,
            &mut reg,
            backend,
        );
        scan_root(
            &dir.join(AGENTS_DIR).join(AGENTS_SUBDIR),
            AgentScope::Project,
            &mut reg,
            backend,
        );
    }

    if !reg.is_empty() {
        let names: Vec<&str> = reg.by_name.keys().map(|s| s.as_str()).collect();
        tracing::info!(subagents = ?names, "subagent discovery");
    }
    reg
}

fn scan_root(
    root: &Path,
    scope: AgentScope,
    reg: &mut AgentRegistry,
    backend: &crate::sandbox_backend::SandboxBackend,
) {
    let entries = match std::fs::read_dir(root) {
        Ok(it) => it,
        Err(e) if e.kind() == ErrorKind::NotFound => return,
        Err(e) => {
            reg.push_diagnostic(format!(
                "subagents root '{}' is unreadable: {e}",
                root.display()
            ));
            return;
        }
    };

    let mut scanned = 0usize;
    for entry in entries {
        scanned += 1;
        if scanned > MAX_ENTRIES_PER_ROOT {
            reg.push_diagnostic(format!(
                "subagents scan under '{}' exceeded {MAX_ENTRIES_PER_ROOT} entries; stopping",
                root.display()
            ));
            break;
        }
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                reg.push_diagnostic(format!(
                    "subagent entry error under '{}': {e}",
                    root.display()
                ));
                continue;
            }
        };
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };
        if !file_type.is_file() {
            continue;
        }
        let path = entry.path();
        let is_md = path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case(AGENT_FILE_EXT));
        if !is_md {
            continue;
        }
        load_agent(&path, scope, reg, backend);
    }
}

fn load_agent(
    path: &Path,
    scope: AgentScope,
    reg: &mut AgentRegistry,
    backend: &crate::sandbox_backend::SandboxBackend,
) {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            reg.push_diagnostic(format!("subagent unreadable at '{}': {e}", path.display()));
            return;
        }
    };
    if raw.len() > MAX_BODY_BYTES {
        reg.push_diagnostic(format!(
            "subagent at '{}' exceeds {MAX_BODY_BYTES} bytes; skipping",
            path.display()
        ));
        return;
    }

    let (front, _body) = match split_frontmatter(&raw) {
        Ok(p) => p,
        Err(e) => {
            reg.push_diagnostic(format!(
                "subagent at '{}' missing or unterminated frontmatter: {e}",
                path.display()
            ));
            return;
        }
    };

    // Reuse the skills frontmatter parser: it extracts `{name, description}`,
    // which is exactly what we need. Anything beyond (tools, model) is ignored
    // in this v1.
    let parsed = match backend.parse_skill_frontmatter(front) {
        Ok(p) => p,
        Err(e) => {
            reg.push_diagnostic(format!(
                "subagent at '{}' has invalid YAML frontmatter: {e}",
                path.display()
            ));
            return;
        }
    };

    let file_stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    let name = match parsed.name {
        Some(n) if n.trim().is_empty() => {
            if file_stem.is_empty() {
                reg.push_diagnostic(format!(
                    "subagent at '{}' has empty `name` and no usable filename; skipping",
                    path.display()
                ));
                return;
            }
            reg.push_diagnostic(format!(
                "subagent at '{}' has empty `name`; using filename '{file_stem}'",
                path.display()
            ));
            file_stem.clone()
        }
        Some(n) => {
            if !file_stem.is_empty() && n != file_stem {
                reg.push_diagnostic(format!(
                    "subagent at '{}' has name '{n}' that does not match filename '{file_stem}'; loading anyway",
                    path.display()
                ));
            }
            if n.chars().count() > 64 {
                reg.push_diagnostic(format!(
                    "subagent at '{}' has name longer than 64 chars; loading anyway",
                    path.display()
                ));
            }
            n
        }
        None => {
            if file_stem.is_empty() {
                reg.push_diagnostic(format!(
                    "subagent at '{}' has no `name` and no usable filename; skipping",
                    path.display()
                ));
                return;
            }
            file_stem.clone()
        }
    };

    let description = match parsed.description {
        Some(d) if !d.trim().is_empty() => d.trim().to_string(),
        _ => {
            reg.push_diagnostic(format!(
                "subagent at '{}' is missing or has empty `description`; skipping",
                path.display()
            ));
            return;
        }
    };

    reg.add(AgentMeta {
        name,
        description,
        location: path.to_path_buf(),
        scope,
    });
}

/// Read just the body of a subagent file (frontmatter stripped). On any
/// I/O or parse error returns the raw file contents so the activation
/// path always has something to feed the LLM.
pub fn read_agent_body(path: &Path) -> Result<String, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read subagent '{}': {e}", path.display()))?;
    let body = match split_frontmatter(&raw) {
        Ok((_, body)) => body.trim_start_matches('\n').trim_end().to_string(),
        Err(_) => raw.trim().to_string(),
    };
    Ok(body)
}

fn normalize_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn find_git_root(start: &Path) -> Option<PathBuf> {
    let mut cur = Some(start);
    while let Some(p) = cur {
        if p.join(".git").exists() {
            return Some(p.to_path_buf());
        }
        cur = p.parent();
    }
    None
}

fn build_dir_chain(cwd: &Path, git_root: Option<&Path>) -> Vec<PathBuf> {
    let Some(root) = git_root else {
        return vec![cwd.to_path_buf()];
    };
    if root == cwd {
        return vec![cwd.to_path_buf()];
    }
    let Ok(rel) = cwd.strip_prefix(root) else {
        return vec![cwd.to_path_buf()];
    };
    let mut chain = vec![root.to_path_buf()];
    let mut acc = root.to_path_buf();
    for part in rel.iter() {
        acc.push(part);
        chain.push(acc.clone());
    }
    chain
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

    fn touch_git(root: &Path) {
        fs::create_dir_all(root.join(".git")).unwrap();
    }

    fn agent_md(name: &str, description: &str, body: &str) -> String {
        format!("---\nname: {name}\ndescription: {description}\n---\n\n{body}\n")
    }

    #[test]
    fn discover_empty_when_nothing_present() {
        let tmp = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let reg = discover_inner(tmp.path(), Some(home.path()));
        assert!(reg.is_empty(), "expected empty registry, got {}", reg.len());
        assert!(reg.diagnostics().is_empty(), "{:?}", reg.diagnostics());
    }

    #[test]
    fn discover_user_scope_claude_dir() {
        let tmp = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        write(
            &home.path().join(".claude").join("agents").join("hunter.md"),
            &agent_md("hunter", "Hunt for bugs", "Be thorough."),
        );
        let reg = discover_inner(tmp.path(), Some(home.path()));
        assert_eq!(reg.len(), 1);
        let meta = reg.get("hunter").unwrap();
        assert_eq!(meta.description, "Hunt for bugs");
        assert_eq!(meta.scope, AgentScope::User);
    }

    #[test]
    fn project_scope_overrides_user_scope() {
        let tmp = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        touch_git(tmp.path());

        // User-scope version
        write(
            &home.path().join(".claude").join("agents").join("hunter.md"),
            &agent_md("hunter", "User version", "user"),
        );
        // Project-scope version (should win)
        write(
            &tmp.path().join(".claude").join("agents").join("hunter.md"),
            &agent_md("hunter", "Project version", "project"),
        );

        let reg = discover_inner(tmp.path(), Some(home.path()));
        assert_eq!(reg.len(), 1);
        let meta = reg.get("hunter").unwrap();
        assert_eq!(meta.description, "Project version");
        assert_eq!(meta.scope, AgentScope::Project);
        // Should have logged the collision
        assert!(
            reg.diagnostics().iter().any(|d| d.contains("duplicate")),
            "expected duplicate diagnostic, got {:?}",
            reg.diagnostics()
        );
    }

    #[test]
    fn agents_dir_wins_over_claude_dir_in_same_scope() {
        let tmp = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        write(
            &home.path().join(".claude").join("agents").join("h.md"),
            &agent_md("h", "claude version", "x"),
        );
        write(
            &home.path().join(".agents").join("agents").join("h.md"),
            &agent_md("h", "agents version", "y"),
        );
        let reg = discover_inner(tmp.path(), Some(home.path()));
        assert_eq!(reg.get("h").unwrap().description, "agents version");
    }

    #[test]
    fn missing_description_is_skipped_with_diagnostic() {
        let tmp = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        write(
            &home.path().join(".claude").join("agents").join("noop.md"),
            "---\nname: noop\n---\n\nbody\n",
        );
        let reg = discover_inner(tmp.path(), Some(home.path()));
        assert!(reg.is_empty());
        assert!(
            reg.diagnostics().iter().any(|d| d.contains("description")),
            "{:?}",
            reg.diagnostics()
        );
    }

    #[test]
    fn malformed_frontmatter_is_skipped_with_diagnostic() {
        let tmp = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        write(
            &home.path().join(".claude").join("agents").join("bad.md"),
            "no frontmatter here\n",
        );
        let reg = discover_inner(tmp.path(), Some(home.path()));
        assert!(reg.is_empty());
        assert!(
            reg.diagnostics().iter().any(|d| d.contains("frontmatter")),
            "{:?}",
            reg.diagnostics()
        );
    }

    #[test]
    fn missing_name_falls_back_to_filename() {
        let tmp = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        write(
            &home
                .path()
                .join(".claude")
                .join("agents")
                .join("from-file.md"),
            "---\ndescription: A subagent\n---\n\nbody\n",
        );
        let reg = discover_inner(tmp.path(), Some(home.path()));
        assert_eq!(reg.len(), 1);
        assert!(reg.get("from-file").is_some());
    }

    #[test]
    fn name_filename_mismatch_loads_with_diagnostic() {
        let tmp = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        write(
            &home.path().join(".claude").join("agents").join("a.md"),
            &agent_md("b", "Mismatched", "x"),
        );
        let reg = discover_inner(tmp.path(), Some(home.path()));
        // Loaded under the frontmatter name, not the filename.
        assert!(reg.get("b").is_some());
        assert!(
            reg.diagnostics()
                .iter()
                .any(|d| d.contains("does not match")),
            "{:?}",
            reg.diagnostics()
        );
    }

    #[test]
    fn non_md_files_ignored() {
        let tmp = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        write(
            &home
                .path()
                .join(".claude")
                .join("agents")
                .join("readme.txt"),
            "not a subagent",
        );
        let reg = discover_inner(tmp.path(), Some(home.path()));
        assert!(reg.is_empty());
        assert!(reg.diagnostics().is_empty());
    }

    #[test]
    fn read_agent_body_strips_frontmatter() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("x.md");
        fs::write(
            &path,
            "---\nname: x\ndescription: y\n---\n\nThe body lines.\nSecond line.\n",
        )
        .unwrap();
        let body = read_agent_body(&path).unwrap();
        assert_eq!(body, "The body lines.\nSecond line.");
    }
}
