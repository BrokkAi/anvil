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

#[derive(Clone, Copy)]
struct BundledAgent {
    path: &'static str,
    content: &'static str,
}

const BUNDLED_AGENTS: &[BundledAgent] = &[
    BundledAgent {
        path: "architect-reviewer.md",
        content: include_str!("../bundled/brokk-agents/architect-reviewer.md"),
    },
    BundledAgent {
        path: "devops-reviewer.md",
        content: include_str!("../bundled/brokk-agents/devops-reviewer.md"),
    },
    BundledAgent {
        path: "dry-reviewer.md",
        content: include_str!("../bundled/brokk-agents/dry-reviewer.md"),
    },
    BundledAgent {
        path: "issue-diagnostician.md",
        content: include_str!("../bundled/brokk-agents/issue-diagnostician.md"),
    },
    BundledAgent {
        path: "issue-enhancer.md",
        content: include_str!("../bundled/brokk-agents/issue-enhancer.md"),
    },
    BundledAgent {
        path: "issue-planner.md",
        content: include_str!("../bundled/brokk-agents/issue-planner.md"),
    },
    BundledAgent {
        path: "security-reviewer.md",
        content: include_str!("../bundled/brokk-agents/security-reviewer.md"),
    },
    BundledAgent {
        path: "senior-dev-reviewer.md",
        content: include_str!("../bundled/brokk-agents/senior-dev-reviewer.md"),
    },
];

/// Discovered subagent metadata. The body is loaded on demand by the
/// `task` dispatch path, not eagerly, so a session with 30 subagents
/// doesn't pay the I/O cost upfront.
#[derive(Debug, Clone)]
pub struct AgentMeta {
    pub name: String,
    pub description: String,
    pub max_turns: Option<usize>,
    pub allowed_tools: Option<Vec<String>>,
    pub location: PathBuf,
    pub scope: AgentScope,
    pub bundled_body: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentScope {
    BuiltIn,
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

    load_bundled_agents(&mut reg, backend);

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
        let built_in_count = reg
            .by_name
            .values()
            .filter(|meta| meta.scope == AgentScope::BuiltIn)
            .count();
        tracing::info!(subagents = ?names, built_in_count, "subagent discovery");
    }
    reg
}

fn load_bundled_agents(reg: &mut AgentRegistry, backend: &crate::sandbox_backend::SandboxBackend) {
    for agent in BUNDLED_AGENTS {
        load_bundled_agent(*agent, reg, backend);
    }
}

fn load_bundled_agent(
    agent: BundledAgent,
    reg: &mut AgentRegistry,
    backend: &crate::sandbox_backend::SandboxBackend,
) {
    let path = Path::new(agent.path);
    let (front, body) = match split_frontmatter(agent.content) {
        Ok(p) => p,
        Err(e) => {
            reg.push_diagnostic(format!(
                "bundled subagent at '{}' missing or unterminated frontmatter: {e}",
                path.display()
            ));
            return;
        }
    };
    let parsed = match backend.parse_skill_frontmatter(front) {
        Ok(p) => p,
        Err(e) => {
            reg.push_diagnostic(format!(
                "bundled subagent at '{}' has invalid YAML frontmatter: {e}",
                path.display()
            ));
            return;
        }
    };
    let file_stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let Some(name) = parsed.name.filter(|n| !n.trim().is_empty()) else {
        reg.push_diagnostic(format!(
            "bundled subagent at '{}' has no usable `name`; skipping",
            path.display()
        ));
        return;
    };
    if !file_stem.is_empty() && name != file_stem {
        reg.push_diagnostic(format!(
            "bundled subagent at '{}' has name '{name}' that does not match filename '{file_stem}'; loading anyway",
            path.display()
        ));
    }
    let description = match parsed.description {
        Some(d) if !d.trim().is_empty() => d.trim().to_string(),
        _ => {
            reg.push_diagnostic(format!(
                "bundled subagent at '{}' is missing or has empty `description`; skipping",
                path.display()
            ));
            return;
        }
    };
    let max_turns = match parse_max_turns(front) {
        Ok(v) => v,
        Err(e) => {
            reg.push_diagnostic(format!(
                "bundled subagent at '{}' has invalid `max_turns`/`maxTurns`: {e}",
                path.display()
            ));
            return;
        }
    };
    let allowed_tools = match parse_allowed_tools(front) {
        Ok(v) => v,
        Err(e) => {
            reg.push_diagnostic(format!(
                "bundled subagent at '{}' has invalid `tools`: {e}",
                path.display()
            ));
            return;
        }
    };
    let location = PathBuf::from("<anvil>").join("brokk-agents").join(path);
    reg.add(AgentMeta {
        name,
        description,
        max_turns,
        allowed_tools,
        location,
        scope: AgentScope::BuiltIn,
        bundled_body: Some(body.trim_start_matches('\n').trim_end()),
    });
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

    // Reuse the skills frontmatter parser for the required fields. Optional
    // subagent-specific fields are parsed locally below so older sandbox
    // parser releases remain compatible.
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

    let max_turns = match parse_max_turns(front) {
        Ok(v) => v,
        Err(e) => {
            reg.push_diagnostic(format!(
                "subagent at '{}' has invalid `max_turns`/`maxTurns`: {e}",
                path.display()
            ));
            return;
        }
    };

    let allowed_tools = match parse_allowed_tools(front) {
        Ok(v) => v,
        Err(e) => {
            reg.push_diagnostic(format!(
                "subagent at '{}' has invalid `tools`: {e}",
                path.display()
            ));
            return;
        }
    };

    reg.add(AgentMeta {
        name,
        description,
        max_turns,
        allowed_tools,
        location: path.to_path_buf(),
        scope,
        bundled_body: None,
    });
}

fn parse_max_turns(frontmatter: &str) -> Result<Option<usize>, String> {
    let lines: Vec<&str> = frontmatter.lines().collect();
    let Some((_, raw_value)) = find_top_level_scalar(&lines, "max_turns")
        .or_else(|| find_top_level_scalar(&lines, "maxTurns"))
    else {
        return Ok(None);
    };
    let value = strip_inline_comment(raw_value).trim();
    let value = value
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .or_else(|| value.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
        .unwrap_or(value)
        .trim();
    if value.is_empty() {
        return Err("must be a positive integer".to_string());
    }
    let turns = value
        .parse::<usize>()
        .map_err(|_| "must be a positive integer".to_string())?;
    if turns == 0 {
        return Err("must be greater than zero".to_string());
    }
    Ok(Some(turns))
}

fn parse_allowed_tools(frontmatter: &str) -> Result<Option<Vec<String>>, String> {
    let lines: Vec<&str> = frontmatter.lines().collect();
    let Some((line_index, raw_value)) = find_top_level_scalar(&lines, "tools") else {
        return Ok(None);
    };

    let mut tools = if strip_inline_comment(raw_value).trim().is_empty() {
        parse_block_tool_list(&lines[line_index + 1..])?
    } else {
        parse_inline_tool_list(raw_value)?
    };
    for tool in &mut tools {
        *tool = normalize_tool_name(tool);
    }
    tools.sort();
    tools.dedup();
    Ok(Some(tools))
}

fn find_top_level_scalar<'a>(lines: &'a [&str], key: &str) -> Option<(usize, &'a str)> {
    let prefix = format!("{key}:");
    lines.iter().enumerate().find_map(|(index, line)| {
        let trimmed = line.trim_start();
        if trimmed.len() != line.len() || trimmed.starts_with('#') {
            return None;
        }
        trimmed.strip_prefix(&prefix).map(|value| (index, value))
    })
}

fn parse_block_tool_list(lines: &[&str]) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for line in lines {
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.len() == line.len() {
            break;
        }
        let Some(item) = trimmed.strip_prefix("- ") else {
            return Err("block list entries must start with `-`".to_string());
        };
        out.push(clean_tool_item(item)?);
    }
    if out.is_empty() {
        return Err("must be a comma-separated string or non-empty list".to_string());
    }
    Ok(out)
}

fn parse_inline_tool_list(value: &str) -> Result<Vec<String>, String> {
    let value = strip_inline_comment(value).trim();
    if value.starts_with('[') || value.ends_with(']') {
        let inner = value
            .strip_prefix('[')
            .and_then(|v| v.strip_suffix(']'))
            .ok_or_else(|| "inline list must be enclosed in `[` and `]`".to_string())?;
        if inner.trim().is_empty() {
            return Ok(Vec::new());
        }
        return split_tool_items(inner);
    }
    split_tool_items(unquote_tool_scalar(value))
}

fn split_tool_items(value: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut in_single = false;
    let mut in_double = false;
    let mut prev_was_backslash = false;
    for (idx, ch) in value.char_indices() {
        match ch {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single && !prev_was_backslash => in_double = !in_double,
            ',' if !in_single && !in_double => {
                out.push(clean_tool_item(&value[start..idx])?);
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
        prev_was_backslash = ch == '\\' && !prev_was_backslash;
        if ch != '\\' {
            prev_was_backslash = false;
        }
    }
    if in_single || in_double {
        return Err("unterminated quoted tool name".to_string());
    }
    out.push(clean_tool_item(&value[start..])?);
    Ok(out)
}

fn clean_tool_item(value: &str) -> Result<String, String> {
    let value = strip_inline_comment(value).trim();
    let value = unquote_tool_scalar(value).trim();
    if value.is_empty() {
        return Err("tool names must not be empty".to_string());
    }
    Ok(value.to_string())
}

fn unquote_tool_scalar(value: &str) -> &str {
    value
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .or_else(|| value.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
        .unwrap_or(value)
}

fn normalize_tool_name(tool: &str) -> String {
    match tool {
        "Read" => "read_file",
        "Write" => "write_file",
        "Edit" | "MultiEdit" | "NotebookEdit" => "edit",
        "Bash" => "run_shell_command",
        "Grep" | "Glob" => "grep_search",
        "WebSearch" | "WebFetch" => "web_search",
        "LS" => "list_directory",
        _ => tool,
    }
    .to_string()
}

fn strip_inline_comment(value: &str) -> &str {
    let mut in_single = false;
    let mut in_double = false;
    let mut prev_was_backslash = false;
    for (idx, ch) in value.char_indices() {
        match ch {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single && !prev_was_backslash => in_double = !in_double,
            '#' if !in_single && !in_double => return &value[..idx],
            _ => {}
        }
        prev_was_backslash = ch == '\\' && !prev_was_backslash;
        if ch != '\\' {
            prev_was_backslash = false;
        }
    }
    value
}

/// Read just the body of a subagent file (frontmatter stripped). On any
/// I/O or parse error returns the raw file contents so the activation
/// path always has something to feed the LLM.
pub fn read_agent_body(meta: &AgentMeta) -> Result<String, String> {
    if let Some(body) = meta.bundled_body {
        return Ok(body.to_string());
    }
    let raw = std::fs::read_to_string(&meta.location)
        .map_err(|e| format!("failed to read subagent '{}': {e}", meta.location.display()))?;
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
        if is_git_marker(&p.join(".git")) {
            return Some(p.to_path_buf());
        }
        cur = p.parent();
    }
    None
}

fn is_git_marker(path: &Path) -> bool {
    if path.is_dir() {
        return path.join("HEAD").is_file();
    }
    path.is_file()
        && std::fs::read_to_string(path).is_ok_and(|content| {
            content
                .lines()
                .next()
                .is_some_and(|line| line.starts_with("gitdir:"))
        })
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
        fs::write(root.join(".git").join("HEAD"), "ref: refs/heads/main\n").unwrap();
    }

    fn agent_md(name: &str, description: &str, body: &str) -> String {
        format!("---\nname: {name}\ndescription: {description}\n---\n\n{body}\n")
    }

    #[test]
    fn bundled_agents_are_discovered_when_nothing_present() {
        let tmp = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let reg = discover_inner(tmp.path(), Some(home.path()));
        let meta = reg
            .get("architect-reviewer")
            .expect("bundled reviewer should be present");
        assert_eq!(meta.scope, AgentScope::BuiltIn);
        assert!(reg.get("security-reviewer").is_some());
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
        let meta = reg.get("hunter").unwrap();
        assert_eq!(meta.description, "Hunt for bugs");
        assert_eq!(meta.scope, AgentScope::User);
        assert_eq!(meta.max_turns, None);
        assert_eq!(meta.allowed_tools, None);
    }

    #[test]
    fn max_turns_frontmatter_is_loaded() {
        let tmp = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        write(
            &home
                .path()
                .join(".claude")
                .join("agents")
                .join("bounded.md"),
            "---\nname: bounded\ndescription: Bounded agent\nmax_turns: 7 # tight cap\n---\n\nbody\n",
        );
        let reg = discover_inner(tmp.path(), Some(home.path()));
        let meta = reg.get("bounded").unwrap();
        assert_eq!(meta.max_turns, Some(7));
    }

    #[test]
    fn max_turns_camel_case_alias_is_loaded() {
        let tmp = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        write(
            &home
                .path()
                .join(".claude")
                .join("agents")
                .join("bounded.md"),
            "---\nname: bounded\ndescription: Bounded agent\nmaxTurns: '6'\n---\n\nbody\n",
        );
        let reg = discover_inner(tmp.path(), Some(home.path()));
        let meta = reg.get("bounded").unwrap();
        assert_eq!(meta.max_turns, Some(6));
    }

    #[test]
    fn tools_frontmatter_comma_list_is_loaded() {
        let tmp = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        write(
            &home.path().join(".claude").join("agents").join("reader.md"),
            "---\nname: reader\ndescription: Read things\ntools: Read, Glob, Grep, WebSearch\n---\n\nbody\n",
        );
        let reg = discover_inner(tmp.path(), Some(home.path()));
        let meta = reg.get("reader").unwrap();
        assert_eq!(
            meta.allowed_tools.as_deref(),
            Some(
                &[
                    "grep_search".to_string(),
                    "read_file".to_string(),
                    "web_search".to_string()
                ][..]
            )
        );
    }

    #[test]
    fn tools_frontmatter_quoted_comma_list_is_loaded() {
        let tmp = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        write(
            &home.path().join(".claude").join("agents").join("reader.md"),
            "---\nname: reader\ndescription: Read things\ntools: \"Read, Grep\"\n---\n\nbody\n",
        );
        let reg = discover_inner(tmp.path(), Some(home.path()));
        let meta = reg.get("reader").unwrap();
        assert_eq!(
            meta.allowed_tools.as_deref(),
            Some(&["grep_search".to_string(), "read_file".to_string()][..])
        );
    }

    #[test]
    fn tools_frontmatter_block_list_is_loaded() {
        let tmp = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        write(
            &home.path().join(".claude").join("agents").join("writer.md"),
            "---\nname: writer\ndescription: Write things\ntools:\n  - Write\n  - Edit\n  - search_symbols\n---\n\nbody\n",
        );
        let reg = discover_inner(tmp.path(), Some(home.path()));
        let meta = reg.get("writer").unwrap();
        assert_eq!(
            meta.allowed_tools.as_deref(),
            Some(
                &[
                    "edit".to_string(),
                    "search_symbols".to_string(),
                    "write_file".to_string()
                ][..]
            )
        );
    }

    #[test]
    fn tools_frontmatter_empty_inline_list_is_loaded() {
        let tmp = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        write(
            &home
                .path()
                .join(".claude")
                .join("agents")
                .join("text-only.md"),
            "---\nname: text-only\ndescription: Text only\ntools: []\n---\n\nbody\n",
        );
        let reg = discover_inner(tmp.path(), Some(home.path()));
        let meta = reg.get("text-only").unwrap();
        assert_eq!(meta.allowed_tools.as_deref(), Some(&[][..]));
    }

    #[test]
    fn invalid_max_turns_is_skipped_with_diagnostic() {
        let tmp = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        write(
            &home
                .path()
                .join(".claude")
                .join("agents")
                .join("bad-cap.md"),
            "---\nname: bad-cap\ndescription: Bad cap\nmax_turns: none\n---\n\nbody\n",
        );
        let reg = discover_inner(tmp.path(), Some(home.path()));
        assert!(
            reg.diagnostics().iter().any(|d| d.contains("max_turns")),
            "{:?}",
            reg.diagnostics()
        );
    }

    #[test]
    fn invalid_tools_frontmatter_is_skipped_with_diagnostic() {
        let tmp = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        write(
            &home.path().join(".claude").join("agents").join("bad.md"),
            "---\nname: bad\ndescription: Bad tools\ntools:\n  Read\n---\n\nbody\n",
        );
        let reg = discover_inner(tmp.path(), Some(home.path()));
        assert!(
            reg.diagnostics().iter().any(|d| d.contains("tools")),
            "{:?}",
            reg.diagnostics()
        );
    }

    #[test]
    fn unbalanced_quoted_tool_name_is_rejected() {
        let err = parse_allowed_tools("name: bad\ndescription: Bad tools\ntools: [Read, \"Grep]\n")
            .unwrap_err();
        assert!(err.contains("unterminated quoted tool name"), "{err}");
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
        let meta = AgentMeta {
            name: "x".into(),
            description: "y".into(),
            max_turns: None,
            allowed_tools: None,
            location: path,
            scope: AgentScope::Project,
            bundled_body: None,
        };
        let body = read_agent_body(&meta).unwrap();
        assert_eq!(body, "The body lines.\nSecond line.");
    }
}
