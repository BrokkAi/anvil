use std::collections::{HashMap, HashSet, hash_map::Entry};
use std::convert::TryFrom;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::llm_client::ModelMetadata;
use crate::mcp::{McpEnvVar, McpFraming, McpServerConfig};
use crate::structured_output::StructuredOutputResult;
use crate::tools::ToolRegistry;

// ---------------------------------------------------------------------------
// Sandbox-bounded read limits
// ---------------------------------------------------------------------------

/// Upper bound on the size of a session zip we will read off disk. Any
/// archive larger than this is rejected before bytes flow through
/// `SandboxBackend::read_zip_entry_text`, so a corrupted or hostile
/// `~/.brokk/sessions/<id>.zip` cannot OOM the agent. 256 MiB is well
/// above what `write_new_session_zip` produces in practice (sessions
/// dominated by conversation text rarely cross a few MB).
const MAX_SESSION_ARCHIVE_BYTES: u64 = 256 * 1024 * 1024;
/// Upper bound on the decompressed `manifest.json` payload. The schema
/// is tiny (id, name, timestamps, mode, model); 1 MiB is loose enough
/// to absorb future fields while still rejecting absurd values.
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
/// Upper bound on the decompressed `fragments-v4.json` payload.
/// Fragments hold one node per turn (no message bodies -- those live in
/// `content/<id>.txt`), so 16 MiB is large enough for any plausible
/// session and small enough to fail fast on a crafted archive.
const MAX_FRAGMENTS_BYTES: u64 = 16 * 1024 * 1024;
/// Upper bound on the decompressed `contexts.jsonl` payload. One JSON
/// line per turn, each line is small (~few KB at most).
const MAX_CONTEXTS_BYTES: u64 = 16 * 1024 * 1024;
/// Per-entry cap when reading `content/*.txt`. A single conversation
/// turn or tool exchange should never approach this size; the cap is
/// there to bound a single hostile entry without dropping legitimate
/// turns that carry, say, a large file dump.
const MAX_CONTENT_ENTRY_BYTES: u64 = 32 * 1024 * 1024;
/// Total budget across all `content/*.txt` entries pulled out of a
/// session zip in one read. A swarm of small bomb entries cannot
/// collectively exceed this, even when each is below the per-entry
/// cap.
const MAX_CONTENT_TOTAL_BYTES: u64 = 256 * 1024 * 1024;
const DEFAULT_SESSION_NAME: &str = "New Session";

// ---------------------------------------------------------------------------
// Store limits
// ---------------------------------------------------------------------------

/// Bounds the in-memory `SessionStore` so a long-running server doesn't
/// accumulate sessions or per-session conversation history without limit.
/// Disk persistence is unaffected: evicted sessions can be re-loaded from
/// their on-disk zip on demand, and history is trimmed only in memory.
#[derive(Debug, Clone, Copy)]
pub struct SessionLimits {
    /// Maximum number of sessions kept resident in memory. When the cap is
    /// exceeded, the least-recently-used session(s) are dropped from memory
    /// (but remain on disk). `0` disables the cap.
    pub max_sessions: usize,
    /// Maximum number of conversation turns retained per session in memory.
    /// When the cap is exceeded, the oldest turns are dropped (sliding
    /// window). `0` disables the cap.
    ///
    /// **Disk is not pruned.** The on-disk session zip retains every turn
    /// ever appended, and `add_turn` rewrites the entire zip per call
    /// (copying every prior `content/*.txt` entry through, plus 2N new ones
    /// for an N-tool turn from #3409). Persistence latency therefore scales
    /// with the *full* on-disk session length, not with `max_history_turns`.
    /// A long-running session with heavy tool use will see write times grow
    /// super-linearly until either an explicit on-disk cap or an
    /// incremental/streaming append path is added. Tracked as follow-up to
    /// #3409 -- documented here so the next person hitting it doesn't
    /// attribute it to a leak.
    pub max_history_turns: usize,
}

impl Default for SessionLimits {
    fn default() -> Self {
        Self {
            max_sessions: 50,
            max_history_turns: 50,
        }
    }
}

/// Drop oldest turns until `history.len() <= max`. `max == 0` disables.
fn trim_history(history: &mut Vec<ConversationTurn>, max: usize) {
    if max > 0 && history.len() > max {
        let drain_to = history.len() - max;
        history.drain(0..drain_to);
    }
}

fn current_timestamp_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn normalize_session_title(name: &str) -> Option<String> {
    let trimmed = name.trim();
    if trimmed.is_empty() || trimmed == DEFAULT_SESSION_NAME {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn derive_session_title(prompt: &str) -> Option<String> {
    let first_line = prompt.lines().find(|line| !line.trim().is_empty())?.trim();
    let collapsed = first_line.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = collapsed.trim_matches(|ch: char| {
        ch.is_whitespace() || matches!(ch, '"' | '\'' | '`' | '.' | ',' | ':' | ';' | '!' | '?')
    });
    if trimmed.is_empty() {
        return None;
    }

    const MAX_TITLE_CHARS: usize = 80;
    let mut title = trimmed.chars().take(MAX_TITLE_CHARS).collect::<String>();
    if title.is_empty() {
        None
    } else {
        if trimmed.chars().count() > MAX_TITLE_CHARS {
            title.push_str("...");
        }
        if title == DEFAULT_SESSION_NAME {
            None
        } else {
            Some(title)
        }
    }
}

pub(crate) fn rfc3339_from_millis(millis: u64) -> Option<String> {
    let millis = i64::try_from(millis).ok()?;
    DateTime::<Utc>::from_timestamp_millis(millis).map(|ts| ts.to_rfc3339())
}

fn effective_mcp_servers(extra_servers: Option<Vec<McpServerConfig>>) -> Vec<McpServerConfig> {
    let mut servers = crate::setup_state::read_mcp_servers();
    for server in extra_servers.into_iter().flatten() {
        if server.name == "bifrost" {
            tracing::warn!(
                command = %server.command,
                "ignoring additive ACP bifrost MCP server; use /mcp setup for canonical bifrost config"
            );
            continue;
        }
        servers.push(server);
    }
    servers
}

/// Error returned when a session already has a prompt in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptStartError {
    AlreadyInFlight,
}

impl std::fmt::Display for PromptStartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyInFlight => write!(f, "prompt already in flight"),
        }
    }
}

impl std::error::Error for PromptStartError {}

fn select_session_model(
    persisted_model: Option<String>,
    default_model: String,
    catalog: &[ModelMetadata],
) -> String {
    let Some(candidate) = persisted_model
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return default_model;
    };

    if catalog.is_empty() || catalog.iter().any(|meta| meta.id == candidate) {
        candidate.to_string()
    } else {
        default_model
    }
}

fn select_session_reasoning_effort(
    model: &str,
    persisted_reasoning_effort: Option<String>,
    catalog: &[ModelMetadata],
) -> Option<String> {
    let requested = persisted_reasoning_effort?.trim().to_string();
    if requested.is_empty() || model.is_empty() {
        return None;
    }

    let Some(meta) = catalog.iter().find(|meta| meta.id == model) else {
        // No catalog metadata means we cannot prove the pick is invalid, so
        // preserve the user's last explicit choice for the new session.
        return Some(requested);
    };

    if meta
        .supported_reasoning_levels
        .iter()
        .any(|preset| preset.effort == requested)
    {
        Some(requested)
    } else {
        // The model knows about reasoning presets but does not support this
        // one, so leave the session unpinned and let snapshot() fall back to
        // the model's default_reasoning_level.
        None
    }
}

fn usable_sandbox_mode_preference(
    mode: Option<crate::sandbox_backend::SandboxMode>,
) -> Option<crate::sandbox_backend::SandboxMode> {
    let mode = mode?;
    match crate::sandbox_backend::backend_for_mode(Some(mode)) {
        Ok(_) => Some(mode),
        Err(e) => {
            tracing::warn!(
                sandbox_mode = mode.as_str(),
                "ignoring persisted sandbox preference because the backend is unavailable: {e}"
            );
            None
        }
    }
}

fn discover_session_context(
    cwd: &Path,
    sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
) -> (
    String,
    Arc<crate::skills::SkillRegistry>,
    Arc<crate::agents::AgentRegistry>,
) {
    match sandbox_mode {
        Some(mode) => (
            crate::agents_md::discover_with_sandbox_mode(cwd, Some(mode)),
            Arc::new(crate::skills::discover_with_sandbox_mode(cwd, Some(mode))),
            Arc::new(crate::agents::discover_with_sandbox_mode(cwd, Some(mode))),
        ),
        None => (
            crate::agents_md::discover(cwd),
            Arc::new(crate::skills::discover(cwd)),
            Arc::new(crate::agents::discover(cwd)),
        ),
    }
}

pub(crate) fn acp_mcp_servers_to_configs(
    servers: Vec<agent_client_protocol::schema::McpServer>,
) -> Vec<McpServerConfig> {
    servers
        .into_iter()
        .filter_map(|server| match server {
            agent_client_protocol::schema::McpServer::Stdio(stdio) => {
                Some(McpServerConfig {
                    name: stdio.name,
                    command: stdio.command.display().to_string(),
                    args: stdio.args,
                    env: stdio
                        .env
                        .into_iter()
                        .map(|var| McpEnvVar {
                            name: var.name,
                            value: var.value,
                        })
                        .collect(),
                    framing: McpFraming::ContentLength,
                    enabled: true,
                })
            }
            agent_client_protocol::schema::McpServer::Http(http) => {
                tracing::warn!(
                    server = %http.name,
                    "ACP HTTP MCP server skipped; Anvil only supports stdio MCP subprocesses"
                );
                None
            }
            agent_client_protocol::schema::McpServer::Sse(sse) => {
                tracing::warn!(
                    server = %sse.name,
                    "ACP SSE MCP server skipped; Anvil only supports stdio MCP subprocesses"
                );
                None
            }
            _ => {
                tracing::warn!(
                    "unsupported ACP MCP server transport skipped; Anvil only supports stdio MCP subprocesses"
                );
                None
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Session modes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionMode {
    Lutz,
    Code,
    Ask,
    Plan,
}

impl SessionMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Lutz => "LUTZ",
            Self::Code => "CODE",
            Self::Ask => "ASK",
            Self::Plan => "PLAN",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "LUTZ" => Some(Self::Lutz),
            "CODE" => Some(Self::Code),
            "ASK" => Some(Self::Ask),
            "PLAN" => Some(Self::Plan),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Permission mode
// ---------------------------------------------------------------------------

/// Per-session permission policy, mirroring the four reference modes that
/// `claude-agent-acp` exposes (default / acceptEdits / plan / bypassPermissions).
/// Surfaced to clients as a `SessionConfigOption` (its own dropdown), independent
/// of `SessionMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionMode {
    Default,
    AcceptEdits,
    /// Hard read-only: refuses every Edit / Delete / Move / Execute tool call.
    /// Renamed from the reference's "plan" to avoid colliding with Brokk's
    /// PLAN behavior mode (LUTZ/CODE/ASK/PLAN), which is a separate dropdown.
    ReadOnly,
    BypassPermissions,
}

impl PermissionMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::AcceptEdits => "acceptEdits",
            Self::ReadOnly => "readOnly",
            Self::BypassPermissions => "bypassPermissions",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "default" => Some(Self::Default),
            "acceptEdits" => Some(Self::AcceptEdits),
            "readOnly" => Some(Self::ReadOnly),
            "bypassPermissions" => Some(Self::BypassPermissions),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Conversation history
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct ConversationTurn {
    pub user_prompt: String,
    pub agent_response: String,
    /// Tool calls executed during this turn, paired 1:1 with their results
    /// in chronological order. Empty for text-only turns.
    ///
    /// Persisted to and read back from the session zip so a `session/load`
    /// reconstructs the same context the LLM had when it produced
    /// `agent_response` -- without this, the model sees only the final
    /// answer and may repeat searches/reads/writes it already performed
    /// in the prior turn (#3409).
    pub tool_exchanges: Vec<ToolExchange>,
    /// Structured-output validation result for this turn's final
    /// assistant response, when requested through ACP `_meta`.
    pub structured_output: Option<StructuredOutputResult>,
    /// LLM-produced summary of this turn, when one has been generated
    /// by the compression engine. When set, `build_prompt_messages`
    /// substitutes the summary for this turn's verbatim
    /// user/tool/assistant messages -- the original log is preserved
    /// on disk (via `summaryContentId` in the zip) so a session reload
    /// reproduces the same compressed prompt. Mirrors Brokk's
    /// `TaskEntry.summary` on the Java side.
    pub summary: Option<String>,
    /// Stable identifier matching the fragment id under which this
    /// turn was persisted in the session zip (the `task.<id>` key in
    /// `fragments-v4.json` plus the `logId` value in
    /// `contexts.jsonl`). Populated by the load path from disk and
    /// by `add_turn` on persist; `None` for turns constructed
    /// in-process before they've been persisted, and for the test
    /// fixtures that build turns directly. Used by
    /// `set_turn_summary` to locate the right `summaryContentId` to
    /// rewrite. Not persisted as a separate field -- it's just the
    /// in-memory shadow of the zip's existing id.
    pub fragment_id: Option<String>,
}

/// One tool invocation and its result, captured during a turn so the next
/// `session/prompt` after a load can re-feed the LLM the full context.
///
/// Kept transport-agnostic (no `ChatMessage` dependency) so `session.rs`
/// stays decoupled from the LLM client. Conversion to `ChatMessage`
/// happens at the agent's prompt-replay call site.
#[derive(Debug, Clone, Default)]
pub struct ToolExchange {
    /// Provider-supplied id (`call.id` in the OpenAI tool-calling shape)
    /// used to pair the assistant's tool_call with its tool_result message
    /// when replaying. Required for OpenAI-compat clients.
    pub call_id: String,
    pub tool_name: String,
    /// Raw JSON arguments string the LLM emitted, kept verbatim so a
    /// re-fed assistant_tool_calls message reproduces the original turn.
    pub arguments: String,
    /// Output the tool returned, post-truncation to `MAX_TOOL_RESULT_BYTES`,
    /// matching what was fed to the LLM during the original turn.
    pub result: String,
}

// ---------------------------------------------------------------------------
// Executor-compatible manifest (manifest.json inside the zip)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionManifest {
    pub id: String,
    pub name: String,
    pub created: u64,
    pub modified: u64,
    #[serde(default = "default_version")]
    pub version: String,
    /// Brokk ACP-specific: session mode, persisted so it survives reload.
    /// Absent in manifests produced by the Java executor.
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "brokkMode")]
    pub mode: Option<String>,
    /// Brokk ACP-specific: last selected model for this session.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "brokkModel"
    )]
    pub model: Option<String>,
    /// Brokk ACP-specific: additional MCP server configuration for this
    /// session, supplied by ACP `session/new`.
    ///
    /// These servers are additive to Anvil's canonical Bifrost setup from
    /// install-level `/mcp` config or the built-in default.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "brokkMcpServers"
    )]
    pub brokk_mcp_servers: Option<Vec<McpServerConfig>>,
}

impl SessionManifest {
    /// Human-readable title to expose to ACP clients.
    pub fn title(&self) -> Option<String> {
        normalize_session_title(&self.name)
    }

    /// Last activity timestamp formatted for ACP clients.
    pub fn updated_at(&self) -> Option<String> {
        rfc3339_from_millis(self.modified)
    }
}

fn default_version() -> String {
    "4.0".to_string()
}

// ---------------------------------------------------------------------------
// Per-session state (in-memory)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub cwd: PathBuf,
    permission_scope_root: PathBuf,
    pub mode: SessionMode,
    pub model: String,
    pub history: Vec<ConversationTurn>,
    pub manifest: SessionManifest,
    pub permission_mode: PermissionMode,
    /// Effective sandbox mode override for this session. Controls both
    /// shell command wrapping (OS sandbox: bwrap / seatbelt) and parsing
    /// backend (wasm vs native). `None` means use the global default
    /// detected at startup.
    ///
    /// Set via `/setup sandbox os|wasm|off` and seeded from the
    /// install-level setup preference for new or reloaded sessions. It is
    /// intentionally not persisted in the session zip, so a tampered or
    /// stale zip cannot silently impose a sandbox policy.
    ///
    /// The `sandbox_mode_explicitly_set` flag tracks whether this value
    /// was set explicitly in the current session (via `/setup sandbox`)
    /// vs inherited from setup state. Only inherited modes are subject
    /// to auto-sync from external setup state changes.
    pub sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
    sandbox_mode_explicitly_set: bool,
    /// Approval keys the user has chosen "Always allow" for this repo.
    /// Most tools use the tool name; shell commands use a scoped key.
    /// Hydrated from trusted repo-local permission state, not from workspace
    /// session zips.
    pub always_allow_tools: HashSet<String>,
    /// Approval keys in the order they were first remembered. This keeps
    /// `/permissions list` and `revoke <number>` stable without
    /// weakening the fast membership check above.
    always_allow_order: Vec<String>,
    /// User's explicit pick from the reasoning-effort dropdown, if any.
    /// `None` means "use the active model's `default_reasoning_level`";
    /// `Some(_)` means "honor this exact level, fail if unsupported".
    /// In-memory only -- the issue scope explicitly excludes
    /// workspace-level persistence for this knob.
    pub selected_reasoning_effort: Option<String>,
    /// Per-session override of the LLM SSE idle timeout (seconds).
    /// `None` means "use the binary-wide default" (CLI flag
    /// `--llm-idle-timeout-secs` / env `BROKK_ACP_LLM_IDLE_TIMEOUT_SECS`).
    /// Set via `/idle-timeout <secs>`, cleared via `/idle-timeout default`.
    /// In-memory only -- does not survive a reload.
    pub idle_timeout_secs: Option<u64>,
    /// Additional per-session MCP servers supplied by ACP `session/new`.
    /// These are additive to Anvil's canonical Bifrost setup from the
    /// install-level `/mcp` configuration or the built-in default.
    pub mcp_servers: Option<Vec<McpServerConfig>>,
    /// Concatenated AGENTS.md / CLAUDE.md content discovered under `cwd`
    /// (and the user's global config slot) at session creation or on the
    /// most recent `update_cwd`. Empty string means none found. Not
    /// persisted -- re-discovered on load so the prompt sees the current
    /// file on disk rather than a stale snapshot from when the session
    /// was first created.
    pub project_instructions: String,
    /// Agent Skills discovered for this session's cwd plus the user's
    /// home dir. Wrapped in `Arc` so `SessionSnapshot::clone()` stays
    /// cheap (the registry can hold dozens of skills with their
    /// descriptions). Refreshed on cwd change just like
    /// `project_instructions`.
    pub skills: Arc<crate::skills::SkillRegistry>,
    /// Subagent catalog (`<name>.md` files) discovered alongside skills.
    /// Drives the `task` meta-tool's `subagent_type` enum and supplies
    /// the body the subagent runs with as a system prompt.
    pub agents: Arc<crate::agents::AgentRegistry>,
    /// Names of skills the harness has already injected into the
    /// conversation context during this session. Used by the
    /// `activate_skill` tool to skip re-injection (the spec's "Deduplicate
    /// activations" recommendation). In-memory only; intentionally not
    /// persisted -- on reload the model re-reads the catalog and decides
    /// fresh which skills to activate.
    pub activated_skills: HashSet<String>,
    /// Cumulative provider-reported token usage for this session.
    /// Populated from each `tool_loop::run` and surfaced on the ACP
    /// `PromptResponse.usage` payload per the `session/usage` RFD
    /// ("Sum of all token types across session", "Total input tokens
    /// across all turns", ...). In-memory only: a session reload
    /// starts the counters fresh because we don't persist per-call
    /// usage on disk yet.
    pub usage: crate::llm_client::TokenUsage,
}

impl Session {
    pub fn new(id: String, cwd: PathBuf, model: String, name: String) -> Self {
        Self::new_with_sandbox_mode(id, cwd, model, name, None)
    }

    fn new_with_sandbox_mode(
        id: String,
        cwd: PathBuf,
        model: String,
        name: String,
        sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
    ) -> Self {
        let now = current_timestamp_millis();
        let mode = SessionMode::Lutz;
        let permission_mode = PermissionMode::Default;
        let manifest = SessionManifest {
            id: id.clone(),
            name,
            created: now,
            modified: now,
            version: "4.0".to_string(),
            mode: Some(mode.as_str().to_string()),
            model: if model.is_empty() {
                None
            } else {
                Some(model.clone())
            },
            brokk_mcp_servers: None,
        };
        let (project_instructions, skills, agents) = discover_session_context(&cwd, sandbox_mode);
        let permission_scope_root = permission_scope_root(&cwd);
        Self {
            id,
            cwd,
            permission_scope_root,
            mode,
            model,
            history: Vec::new(),
            manifest,
            permission_mode,
            sandbox_mode,
            sandbox_mode_explicitly_set: sandbox_mode.is_some(),
            always_allow_tools: HashSet::new(),
            always_allow_order: Vec::new(),
            selected_reasoning_effort: None,
            idle_timeout_secs: None,
            mcp_servers: None,
            project_instructions,
            skills,
            agents,
            activated_skills: HashSet::new(),
            usage: crate::llm_client::TokenUsage::default(),
        }
    }

    /// Construct a `Session` from data loaded off disk.
    ///
    /// SECURITY: transient fields that intentionally do NOT come from the
    /// workspace session zip (`permission_mode`, `always_allow_tools`,
    /// `always_allow_order`) are reset here. `SessionStore` rehydrates
    /// remembered approvals from trusted repo-local permission state after
    /// construction, so a stale or tampered zip still cannot silently
    /// auto-allow tool calls on launch.
    ///
    /// Also rejects a mismatch between `id` (the caller's requested id, used
    /// to locate the zip and to key the in-memory map) and `manifest.id`
    /// (read from inside the zip). A mismatch indicates either a stale or
    /// tampered zip, or a logic error mapping ids to zip paths -- continuing
    /// silently would route subsequent writes against a different zip than
    /// the one we resumed from.
    pub fn from_persisted(
        id: String,
        cwd: PathBuf,
        mode: SessionMode,
        model: String,
        history: Vec<ConversationTurn>,
        manifest: SessionManifest,
    ) -> Result<Self, SessionIdMismatch> {
        Self::from_persisted_with_sandbox_mode(id, cwd, mode, model, history, manifest, None)
    }

    fn from_persisted_with_sandbox_mode(
        id: String,
        cwd: PathBuf,
        mode: SessionMode,
        model: String,
        history: Vec<ConversationTurn>,
        manifest: SessionManifest,
        sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
    ) -> Result<Self, SessionIdMismatch> {
        if manifest.id != id {
            return Err(SessionIdMismatch {
                requested: id,
                loaded: manifest.id,
            });
        }
        let (project_instructions, skills, agents) = discover_session_context(&cwd, sandbox_mode);
        let permission_scope_root = permission_scope_root(&cwd);
        let mcp_servers = manifest.brokk_mcp_servers.clone();
        Ok(Self {
            id,
            cwd,
            permission_scope_root,
            mode,
            model,
            history,
            manifest,
            permission_mode: PermissionMode::Default,
            sandbox_mode,
            sandbox_mode_explicitly_set: sandbox_mode.is_some(),
            always_allow_tools: HashSet::new(),
            always_allow_order: Vec::new(),
            // Reset reasoning effort on load -- it's a transient
            // per-session preference (per issue scope), so a
            // reloaded zip starts at "use model default" until the
            // user picks again.
            selected_reasoning_effort: None,
            // Same rationale: idle timeout override is in-memory only.
            idle_timeout_secs: None,
            mcp_servers,
            project_instructions,
            skills,
            agents,
            activated_skills: HashSet::new(),
            usage: crate::llm_client::TokenUsage::default(),
        })
    }

    fn set_always_allow_keys(&mut self, keys: impl IntoIterator<Item = String>) {
        self.always_allow_tools.clear();
        self.always_allow_order.clear();
        for key in keys {
            if !key.is_empty() && self.always_allow_tools.insert(key.clone()) {
                self.always_allow_order.push(key);
            }
        }
    }
}

/// Returned by `Session::from_persisted` when the requested session id
/// doesn't match the id stored in the manifest read from disk.
#[derive(Debug)]
pub struct SessionIdMismatch {
    pub requested: String,
    pub loaded: String,
}

impl std::fmt::Display for SessionIdMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "session id mismatch: caller requested '{}' but manifest in zip says '{}'",
            self.requested, self.loaded
        )
    }
}

impl std::error::Error for SessionIdMismatch {}

/// Snapshot of the per-session data needed to start a prompt turn. The
/// conversation history is cloned exactly once under the read lock; callers
/// consume it (via `.into_iter()`) when constructing protocol-specific
/// message types, so no further string clones happen on the prompt path.
///
/// This intentionally exposes raw `ConversationTurn` rather than a
/// pre-built `Vec<ChatMessage>` so `session.rs` doesn't depend on the LLM
/// transport layer — message assembly belongs at the call site.
#[derive(Debug)]
pub struct SessionSnapshot {
    pub cwd: PathBuf,
    pub mode: SessionMode,
    pub model: String,
    pub history: Vec<ConversationTurn>,
    /// Reasoning-effort level to send on this turn, already resolved
    /// against the user's pick and the active model's
    /// `default_reasoning_level`. `None` means the model exposes no
    /// effort presets, so the backend will simply omit the field.
    pub reasoning_effort: Option<String>,
    /// Per-session override of the LLM SSE idle timeout (seconds).
    /// `None` means the caller should fall back to the binary-wide default.
    pub idle_timeout_secs: Option<u64>,
    /// AGENTS.md / CLAUDE.md content discovered for this session,
    /// concatenated general -> specific. Empty when nothing is found.
    pub project_instructions: String,
    /// Agent Skills (`SKILL.md`) discovered for this session. Wrapped
    /// in `Arc` so cloning the snapshot doesn't copy the registry.
    pub skills: Arc<crate::skills::SkillRegistry>,
}

/// Human-readable session metadata for ACP notifications and lists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SessionMetadata {
    pub title: Option<String>,
    pub updated_at: Option<String>,
}

// ---------------------------------------------------------------------------
// Zip I/O: read/write executor-compatible session zips
// ---------------------------------------------------------------------------

fn sessions_dir(cwd: &Path) -> PathBuf {
    cwd.join(".brokk").join("sessions")
}

fn session_zip_path(cwd: &Path, id: &str) -> PathBuf {
    sessions_dir(cwd).join(format!("{id}.zip"))
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct RepoPermissionState {
    #[serde(default, skip_serializing_if = "Vec::is_empty", rename = "alwaysAllow")]
    always_allow: Vec<String>,
    #[serde(default, rename = "alwaysAllowShellPrefixes", skip_serializing)]
    legacy_shell_prefixes: Vec<String>,
}

impl RepoPermissionState {
    fn merged_approvals(&self) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for key in self
            .always_allow
            .iter()
            .chain(self.legacy_shell_prefixes.iter())
        {
            if !key.is_empty() && seen.insert(key.clone()) {
                out.push(key.clone());
            }
        }
        out
    }

    fn migrate_legacy(&mut self) {
        self.always_allow = self.merged_approvals();
        self.legacy_shell_prefixes.clear();
    }
}

static REPO_PERMISSION_WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn permission_scope_root(cwd: &Path) -> PathBuf {
    let start = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());
    for ancestor in start.ancestors() {
        if ancestor.join(".git").exists() {
            return ancestor.to_path_buf();
        }
    }
    start
}

fn repo_permission_path(scope_root: &Path) -> PathBuf {
    scope_root.join(".brokk").join("permissions.json")
}

fn read_repo_permission_state(scope_root: &Path) -> RepoPermissionState {
    let path = repo_permission_path(scope_root);
    let Ok(bytes) = std::fs::read(&path) else {
        return RepoPermissionState::default();
    };
    serde_json::from_slice(&bytes).unwrap_or_default()
}

fn write_repo_permission_state(
    scope_root: &Path,
    state: &RepoPermissionState,
) -> anyhow::Result<()> {
    use anyhow::Context;

    let path = repo_permission_path(scope_root);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating repo permission dir {}", parent.display()))?;
    }
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("permissions.json");
    let tmp = path.with_file_name(format!(".{file_name}.tmp.{}", uuid::Uuid::new_v4()));
    let bytes = serde_json::to_vec_pretty(state).context("serializing repo permission state")?;
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &path)
        .with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))?;
    Ok(())
}

fn update_repo_permission_state(
    scope_root: &Path,
    mutator: impl FnOnce(&mut RepoPermissionState),
) -> anyhow::Result<()> {
    let _guard = REPO_PERMISSION_WRITE_LOCK
        .lock()
        .expect("repo permission write mutex poisoned");
    let mut state = read_repo_permission_state(scope_root);
    state.migrate_legacy();
    mutator(&mut state);
    write_repo_permission_state(scope_root, &state)
}

fn read_repo_always_allow_keys(scope_root: &Path) -> Vec<String> {
    read_repo_permission_state(scope_root).merged_approvals()
}

fn remember_repo_always_allow_key(scope_root: &Path, key: &str) -> anyhow::Result<()> {
    if key.is_empty() {
        return Ok(());
    }
    update_repo_permission_state(scope_root, |state| {
        if !state.always_allow.iter().any(|existing| existing == key) {
            state.always_allow.push(key.to_string());
        }
    })
}

fn forget_repo_always_allow_key(scope_root: &Path, key: &str) -> anyhow::Result<()> {
    update_repo_permission_state(scope_root, |state| {
        state.always_allow.retain(|existing| existing != key);
    })
}

fn clear_repo_always_allow_keys(scope_root: &Path) -> anyhow::Result<()> {
    update_repo_permission_state(scope_root, |state| {
        state.always_allow.clear();
    })
}

/// Read manifest.json from a session zip. Returns None if the zip or manifest is unreadable.
///
/// Routed through `SandboxBackend::read_zip_entry_text` so the parser
/// runs inside the wasm sandbox by default: a malformed or hostile
/// session zip on disk cannot OOM the agent (the archive read is
/// bounded by `MAX_SESSION_ARCHIVE_BYTES` and the manifest entry by
/// `MAX_MANIFEST_BYTES`, and the wasm linear-memory limit catches
/// anything those size pre-checks miss).
fn read_manifest_from_zip(zip_path: &Path) -> Option<SessionManifest> {
    let body = match crate::sandbox_backend::global().read_zip_entry_text(
        zip_path,
        "manifest.json",
        MAX_SESSION_ARCHIVE_BYTES,
        MAX_MANIFEST_BYTES,
    ) {
        Ok(Some(s)) => s,
        Ok(None) => return None,
        Err(e) => {
            tracing::warn!(
                path = %zip_path.display(),
                "session manifest unreadable: {e}"
            );
            return None;
        }
    };
    serde_json::from_str(&body).ok()
}

/// Read conversation history from a session zip.
/// Reads TaskFragmentDto entries from fragments-v4.json and resolves their
/// markdownContentId / messages[].contentId against content/*.txt files.
///
/// Three sandboxed pulls per zip: one prefix-scan to grab every
/// `content/<id>.txt` in a single sandbox boot, one named fetch for
/// `fragments-v4.json`, one for `contexts.jsonl`. The wasm memory cap
/// (`StoreLimits::memory_size`) and the per-entry / per-total byte
/// limits above keep a crafted archive from OOM-ing the host.
fn read_history_from_zip(zip_path: &Path) -> Vec<ConversationTurn> {
    let backend = crate::sandbox_backend::global();

    // 1. Read all content/*.txt files into a map: content_id -> text
    let content_entries = match backend.read_zip_entries_with_prefix(
        zip_path,
        "content/",
        MAX_SESSION_ARCHIVE_BYTES,
        MAX_CONTENT_ENTRY_BYTES,
        MAX_CONTENT_TOTAL_BYTES,
    ) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(
                path = %zip_path.display(),
                "session content entries unreadable: {e}"
            );
            return vec![];
        }
    };
    let mut content_map: HashMap<String, String> = HashMap::with_capacity(content_entries.len());
    for (name, body) in content_entries {
        if let Some(content_id) = name
            .strip_prefix("content/")
            .and_then(|s| s.strip_suffix(".txt"))
        {
            content_map.insert(content_id.to_string(), body);
        }
    }

    // 2. Read fragments-v4.json
    let fragments_json = match backend.read_zip_entry_text(
        zip_path,
        "fragments-v4.json",
        MAX_SESSION_ARCHIVE_BYTES,
        MAX_FRAGMENTS_BYTES,
    ) {
        Ok(Some(s)) => s,
        Ok(None) => return vec![],
        Err(e) => {
            tracing::warn!(
                path = %zip_path.display(),
                "fragments-v4.json unreadable: {e}"
            );
            return vec![];
        }
    };

    let fragments: serde_json::Value = match serde_json::from_str(&fragments_json) {
        Ok(v) => v,
        Err(_) => return vec![],
    };

    // 2.5. Recover chronological turn order from contexts.jsonl.
    //      `fragments.task` is a `serde_json::Map` backed by `BTreeMap`
    //      (the `preserve_order` feature is intentionally not enabled), so
    //      iterating it yields lexicographic UUID order -- effectively
    //      shuffled with respect to the order turns were appended. That's
    //      fine for a one-turn round-trip but actively misleads the LLM on
    //      replay across multiple turns: the user prompt of turn 0 ends up
    //      paired with the tool exchanges of turn 2, etc.
    //
    //      `contexts.jsonl` is append-only and one line per turn (plus the
    //      initial empty context), with `virtuals: [task_fragment_id]`
    //      pointing to the new fragment. Walking it line-by-line and
    //      collecting `virtuals` in first-seen order gives the chronological
    //      sequence of task fragments. Works for both newly-written zips and
    //      older ones that already followed this convention -- no migration.
    let chronological_ids = read_task_fragment_order_from_zip(zip_path);
    let summary_content_ids = read_summary_content_ids_from_zip(zip_path);

    // 3. Extract conversation from task fragments, in chronological order
    //    where recoverable. Each task fragment may have:
    //    - messages: [{role, contentId, ...}] -- includes "tool_call" and
    //      "tool_result" entries when the turn used tools (#3409)
    //    - markdownContentId: points to the rendered final response
    let mut turns = Vec::new();
    if let Some(tasks) = fragments.get("task").and_then(|t| t.as_object()) {
        // Build the visit order: ids known to contexts.jsonl first (in
        // chronological order), then any orphan task fragments not
        // referenced by any context (defensive -- preserves the prior
        // behavior of not dropping turns just because contexts.jsonl is
        // missing or malformed).
        let mut emitted: HashSet<String> = HashSet::new();
        let mut visit: Vec<&serde_json::Value> = Vec::with_capacity(tasks.len());
        for id in &chronological_ids {
            if let Some(task) = tasks.get(id) {
                emitted.insert(id.clone());
                visit.push(task);
            }
        }
        for (id, task) in tasks {
            if !emitted.contains(id) {
                visit.push(task);
            }
        }

        for task in visit {
            // Walk the messages array first to pick up any tool exchanges,
            // even when markdownContentId is the source of agent_response.
            // Tool exchanges are persisted in `messages` regardless of the
            // markdown shortcut (which only stores the final text).
            let exchanges = read_tool_exchanges_from_messages(task, &content_map);

            // Resolve this fragment's persisted summary, if any. The
            // fragment self-identifies via its `id` field (which equals
            // its outer `task.*` key); we look that up in the
            // `logId -> summaryContentId` mapping pulled from
            // contexts.jsonl, then dereference against the content
            // blob map.
            let fragment_id = task.get("id").and_then(|v| v.as_str()).map(str::to_string);
            let summary = fragment_id
                .as_ref()
                .and_then(|fragment_id| summary_content_ids.get(fragment_id))
                .and_then(|sid| content_map.get(sid))
                .cloned();
            let structured_output = task
                .get("structuredOutputContentId")
                .and_then(|v| v.as_str())
                .and_then(|sid| content_map.get(sid))
                .and_then(|raw| serde_json::from_str::<StructuredOutputResult>(raw).ok());

            // Try markdownContentId first (newer format: pre-rendered markdown)
            if let Some(content_id) = task.get("markdownContentId").and_then(|v| v.as_str())
                && let Some(text) = content_map.get(content_id)
            {
                let description = task
                    .get("taskDescription")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let user_prompt = if description.is_empty() {
                    // Older zips put the user text under a `role: "user"` entry
                    // instead of taskDescription -- pull it from there as a
                    // last resort so the user side isn't lost.
                    read_first_role_text(task, "user", &content_map)
                } else {
                    description.to_string()
                };
                turns.push(ConversationTurn {
                    user_prompt,
                    agent_response: text.clone(),
                    tool_exchanges: exchanges,
                    structured_output,
                    summary,
                    fragment_id,
                });
                continue;
            }

            // Fall back to messages array (older format)
            if let Some(messages) = task.get("messages").and_then(|v| v.as_array()) {
                let mut user_text = String::new();
                let mut assistant_text = String::new();
                for msg in messages {
                    let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
                    let content_id = msg.get("contentId").and_then(|v| v.as_str()).unwrap_or("");
                    let text = content_map.get(content_id).cloned().unwrap_or_default();
                    match role {
                        "user" => user_text.push_str(&text),
                        "ai" => assistant_text.push_str(&text),
                        // tool_call / tool_result entries are folded into
                        // `exchanges` above; ignore them here so they don't
                        // contaminate user_text / assistant_text.
                        _ => {}
                    }
                }
                // Keep the turn whenever we recovered any AI-side activity --
                // either rendered text or recorded tool exchanges. Dropping a
                // turn that produced only tool calls (e.g. a future write
                // path that omits the trailing `role: "ai"` entry) would
                // also drop its `tool_exchanges`, defeating the replay.
                if !assistant_text.is_empty() || !exchanges.is_empty() {
                    turns.push(ConversationTurn {
                        user_prompt: user_text,
                        agent_response: assistant_text,
                        tool_exchanges: exchanges,
                        structured_output: structured_output.clone(),
                        summary: summary.clone(),
                        fragment_id: fragment_id.clone(),
                    });
                }
            }
        }
    }

    turns
}

/// Read `contexts.jsonl` from `archive` and return the task-fragment ids in
/// the chronological order they were appended (each turn's context entry
/// references its new fragment via `virtuals: [task_fragment_id]`).
///
/// Returns an empty vec if `contexts.jsonl` is missing or unreadable -- the
/// caller falls back to the BTreeMap iteration order on the fragments map,
/// which keeps the prior shuffled-but-best-effort behavior for malformed
/// zips rather than dropping turns outright.
fn read_task_fragment_order_from_zip(zip_path: &Path) -> Vec<String> {
    let buf = match crate::sandbox_backend::global().read_zip_entry_text(
        zip_path,
        "contexts.jsonl",
        MAX_SESSION_ARCHIVE_BYTES,
        MAX_CONTEXTS_BYTES,
    ) {
        Ok(Some(s)) => s,
        Ok(None) => return Vec::new(),
        Err(e) => {
            tracing::warn!(
                path = %zip_path.display(),
                "contexts.jsonl unreadable: {e}"
            );
            return Vec::new();
        }
    };

    let mut ids: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for line in buf.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(ctx) = serde_json::from_str::<serde_json::Value>(trimmed) else {
            continue;
        };
        let Some(virtuals) = ctx.get("virtuals").and_then(|v| v.as_array()) else {
            continue;
        };
        for v in virtuals {
            if let Some(id) = v.as_str()
                && seen.insert(id.to_string())
            {
                ids.push(id.to_string());
            }
        }
    }
    ids
}

/// Walk `contexts.jsonl` and extract the `logId -> summaryContentId`
/// mapping out of each context's `tasks[]` array. Mirrors the Brokk
/// Java side's `TaskEntry.summary` storage, where each task references
/// a `content/<id>.txt` entry holding its compressed summary text.
///
/// Returns an empty map if `contexts.jsonl` is missing, unreadable, or
/// contains no task with a non-null `summaryContentId` -- consistent
/// with `read_task_fragment_order_from_zip`'s degradation policy.
fn read_summary_content_ids_from_zip(zip_path: &Path) -> HashMap<String, String> {
    let buf = match crate::sandbox_backend::global().read_zip_entry_text(
        zip_path,
        "contexts.jsonl",
        MAX_SESSION_ARCHIVE_BYTES,
        MAX_CONTEXTS_BYTES,
    ) {
        Ok(Some(s)) => s,
        _ => return HashMap::new(),
    };
    let mut out = HashMap::new();
    for line in buf.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(ctx) = serde_json::from_str::<serde_json::Value>(trimmed) else {
            continue;
        };
        let Some(tasks) = ctx.get("tasks").and_then(|v| v.as_array()) else {
            continue;
        };
        for task in tasks {
            let log_id = task.get("logId").and_then(|v| v.as_str());
            let summary_id = task.get("summaryContentId").and_then(|v| v.as_str());
            if let (Some(log_id), Some(summary_id)) = (log_id, summary_id) {
                // Later contexts win -- a turn that was recompressed
                // points to a new content id, and we want the latest.
                out.insert(log_id.to_string(), summary_id.to_string());
            }
        }
    }
    out
}

/// Walk a task fragment's `messages` array and pair `tool_call` entries
/// with the matching `tool_result` by `toolCallId`. Order in the returned
/// `Vec` follows the order of the `tool_call` entries; results without a
/// matching call are dropped (and vice versa) since a half-recorded
/// exchange would mislead the model on replay more than omitting it.
fn read_tool_exchanges_from_messages(
    task: &serde_json::Value,
    content_map: &HashMap<String, String>,
) -> Vec<ToolExchange> {
    let Some(messages) = task.get("messages").and_then(|v| v.as_array()) else {
        return Vec::new();
    };

    // First pass: collect tool_results keyed by toolCallId so we can pair
    // them up regardless of the order calls and results appear in (the
    // OpenAI shape interleaves: call, call, result, result).
    let mut results: HashMap<String, String> = HashMap::new();
    for msg in messages {
        if msg.get("role").and_then(|v| v.as_str()) != Some("tool_result") {
            continue;
        }
        let Some(call_id) = msg.get("toolCallId").and_then(|v| v.as_str()) else {
            continue;
        };
        let content_id = msg.get("contentId").and_then(|v| v.as_str()).unwrap_or("");
        let result = content_map.get(content_id).cloned().unwrap_or_default();
        if let Some(prev) = results.insert(call_id.to_string(), result) {
            // Two `tool_result` entries with the same `toolCallId` is not
            // something our write path produces (call_ids come from the
            // provider and are unique within a turn), but a corrupted or
            // third-party-generated zip could carry duplicates. Surface it
            // as a warning so the operator sees corrupted-zip evidence
            // rather than chasing model-quality regressions: the second
            // result silently overwrote the first in this map.
            tracing::warn!(
                call_id = %call_id,
                prev_result_len = prev.len(),
                "duplicate toolCallId in persisted messages; previous tool_result discarded on read"
            );
        }
    }

    // Second pass: emit exchanges in the order of the tool_call entries.
    let mut exchanges = Vec::new();
    for msg in messages {
        if msg.get("role").and_then(|v| v.as_str()) != Some("tool_call") {
            continue;
        }
        let call_id = msg
            .get("toolCallId")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let tool_name = msg
            .get("toolName")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let arguments_id = msg
            .get("argumentsContentId")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let arguments = content_map.get(arguments_id).cloned().unwrap_or_default();
        let Some(result) = results.remove(call_id) else {
            // No result recorded for this call; skip rather than feed the
            // LLM a tool_call without a matching tool_result (the OpenAI
            // wire format requires the pair).
            continue;
        };
        exchanges.push(ToolExchange {
            call_id: call_id.to_string(),
            tool_name: tool_name.to_string(),
            arguments,
            result,
        });
    }
    exchanges
}

/// Look up the first message in `task.messages` whose `role` matches and
/// return its dereferenced content text. Used as a fallback when
/// `taskDescription` is null but the user prompt is still recoverable from
/// the messages array.
fn read_first_role_text(
    task: &serde_json::Value,
    role: &str,
    content_map: &HashMap<String, String>,
) -> String {
    task.get("messages")
        .and_then(|v| v.as_array())
        .and_then(|msgs| {
            msgs.iter().find_map(|m| {
                if m.get("role").and_then(|v| v.as_str()) == Some(role) {
                    let cid = m.get("contentId").and_then(|v| v.as_str())?;
                    content_map.get(cid).cloned()
                } else {
                    None
                }
            })
        })
        .unwrap_or_default()
}

/// Run `populate` against a fresh `<zip_path>.tmp`, finalize the writer, and atomically
/// rename it over `zip_path`. On any failure the temp file is cleaned up and the original
/// zip (if any) is left untouched, so callers get all-or-nothing semantics.
fn with_temp_zip_writer<F>(zip_path: &Path, populate: F) -> anyhow::Result<()>
where
    F: FnOnce(
        &mut zip::ZipWriter<std::fs::File>,
        zip::write::SimpleFileOptions,
    ) -> anyhow::Result<()>,
{
    use anyhow::Context;

    let dir = zip_path.parent().with_context(|| {
        format!(
            "session zip path has no parent directory: {}",
            zip_path.display()
        )
    })?;
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating sessions dir {}", dir.display()))?;

    let tmp = zip_path.with_extension("tmp");
    let file = std::fs::File::create(&tmp)
        .with_context(|| format!("creating temp zip {}", tmp.display()))?;

    let mut writer = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    if let Err(e) = populate(&mut writer, options) {
        // Drop the writer (closes its file handle) before unlinking the temp.
        drop(writer);
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }

    writer
        .finish()
        .with_context(|| format!("finalizing temp zip {}", tmp.display()))?;
    std::fs::rename(&tmp, zip_path)
        .with_context(|| format!("renaming {} to {}", tmp.display(), zip_path.display()))?;
    Ok(())
}

/// Copy every entry from `archive` into `writer` whose name does not match `skip`.
/// Per-entry I/O failures bubble up; callers in `with_temp_zip_writer` will discard
/// the half-written temp zip.
/// Stream the entries of an existing session zip through the sandbox,
/// writing each one into `writer` as soon as it comes back. We list
/// names first, then fetch each entry individually so the host's peak
/// memory is bounded by the per-entry cap (`MAX_CONTENT_ENTRY_BYTES`,
/// 32 MiB) rather than the full decompressed archive. This is the
/// load-bearing piece that lets `append_turn_to_zip` and
/// `rewrite_manifest_in_zip` rebuild a session zip without ever
/// re-parsing it with the host's `zip` crate, closing the TOCTOU
/// window between session load and the next turn append.
fn copy_zip_entries_via_sandbox<F>(
    zip_path: &Path,
    writer: &mut zip::ZipWriter<std::fs::File>,
    options: zip::write::SimpleFileOptions,
    skip: F,
) -> anyhow::Result<()>
where
    F: Fn(&str) -> bool,
{
    use anyhow::Context;
    // The sandbox primitives translate a missing archive into
    // "no entries" so most callers can stay silent. The rewrite path
    // is different: silently producing an archive that does not
    // contain the prior session's entries would corrupt the on-disk
    // state. Match the prior `File::open(...).with_context(...)`
    // behaviour and fail loudly so `with_temp_zip_writer` rolls back.
    if !zip_path.exists() {
        anyhow::bail!(
            "opening session zip {} for update: file not found",
            zip_path.display()
        );
    }
    let backend = crate::sandbox_backend::global();
    let names = backend
        .list_zip_entry_names(zip_path, MAX_SESSION_ARCHIVE_BYTES)
        .with_context(|| format!("listing entries in {}", zip_path.display()))?;
    for name in names {
        if skip(&name) {
            continue;
        }
        let body = match backend.read_zip_entry_text(
            zip_path,
            &name,
            MAX_SESSION_ARCHIVE_BYTES,
            MAX_CONTENT_ENTRY_BYTES,
        ) {
            Ok(Some(s)) => s,
            // Race: an entry disappeared between list and read. Treat
            // as a transient zip mutation and abort the rewrite -- the
            // atomic temp-then-rename in `with_temp_zip_writer` keeps
            // the on-disk zip unchanged.
            Ok(None) => anyhow::bail!(
                "zip entry {name} disappeared between list and read in {}",
                zip_path.display()
            ),
            Err(e) => {
                return Err(anyhow::anyhow!(e))
                    .with_context(|| format!("reading zip entry {name}"));
            }
        };
        writer
            .start_file(&name, options)
            .with_context(|| format!("starting zip entry {name}"))?;
        writer
            .write_all(body.as_bytes())
            .with_context(|| format!("writing zip entry {name}"))?;
    }
    Ok(())
}

/// Write a new empty session zip compatible with the executor.
fn write_new_session_zip(zip_path: &Path, manifest: &SessionManifest) -> anyhow::Result<()> {
    use anyhow::Context;

    with_temp_zip_writer(zip_path, |zip, options| {
        // manifest.json
        let manifest_json =
            serde_json::to_string_pretty(manifest).context("serializing session manifest")?;
        zip.start_file("manifest.json", options)?;
        zip.write_all(manifest_json.as_bytes())?;

        // Empty context (one initial context entry)
        let ctx_id = uuid::Uuid::new_v4().to_string();
        let context_line = serde_json::json!({
            "id": ctx_id,
            "editable": [],
            "readonly": [],
            "virtuals": [],
            "pinned": [],
            "tasks": [],
            "parsedOutputId": null
        });
        zip.start_file("contexts.jsonl", options)?;
        zip.write_all(serde_json::to_string(&context_line)?.as_bytes())?;
        zip.write_all(b"\n")?;

        // Empty fragments
        let fragments =
            serde_json::json!({"version": 4, "referenced": {}, "virtual": {}, "task": {}});
        zip.start_file("fragments-v4.json", options)?;
        zip.write_all(serde_json::to_string(&fragments)?.as_bytes())?;

        // Empty content metadata
        zip.start_file("content_metadata.json", options)?;
        zip.write_all(b"{}")?;

        // Empty group info
        let group_info = serde_json::json!({"contextToGroupId": {}, "groupLabels": {}});
        zip.start_file("group_info.json", options)?;
        zip.write_all(serde_json::to_string(&group_info)?.as_bytes())?;

        Ok(())
    })
}

/// Update manifest.json and add a conversation turn to an existing session zip.
///
/// All framing failures (open, archive read, per-entry copy, finalize, rename) propagate
/// to the caller so `add_turn` can roll back and keep `memory == disk`. The atomic
/// temp-then-rename in `with_temp_zip_writer` guarantees the on-disk zip is unchanged
/// on any failure path.
/// Persist a new turn into the session zip and return the fragment id
/// it was stored under. The caller (`add_turn`) writes that id back
/// onto the in-memory `ConversationTurn` so subsequent
/// `set_turn_summary` calls have a stable handle to the on-disk task.
fn append_turn_to_zip(
    zip_path: &Path,
    manifest: &SessionManifest,
    turn: &ConversationTurn,
) -> anyhow::Result<String> {
    use anyhow::Context;

    // Pre-read the entries we plan to rewrite through the sandbox so a
    // mutated zip on disk (an attacker swapping the file between the
    // initial load and this append) cannot panic the host's `zip`
    // crate.
    //
    // Error policy here is load-bearing: only an *explicitly missing*
    // entry (`Ok(None)`) or a legacy-malformed entry (`Ok(Some(_))` that
    // does not parse as JSON) is allowed to fall back to the empty
    // default. Any real read failure -- `FileTooLarge`, sandbox crash,
    // transient I/O, an entry that exceeds the per-entry cap -- must
    // propagate so `add_turn` rolls back the in-memory state instead of
    // silently rewriting the on-disk zip with empty fragments/contexts
    // (which would erase prior history).
    let backend = crate::sandbox_backend::global();
    let mut existing_fragments: serde_json::Value =
        serde_json::json!({"version": 4, "referenced": {}, "virtual": {}, "task": {}});
    if let Some(buf) = backend
        .read_zip_entry_text(
            zip_path,
            "fragments-v4.json",
            MAX_SESSION_ARCHIVE_BYTES,
            MAX_FRAGMENTS_BYTES,
        )
        .context("reading fragments-v4.json from session zip")?
        && let Ok(v) = serde_json::from_str(&buf)
    {
        existing_fragments = v;
    }
    let existing_contexts = backend
        .read_zip_entry_text(
            zip_path,
            "contexts.jsonl",
            MAX_SESSION_ARCHIVE_BYTES,
            MAX_CONTEXTS_BYTES,
        )
        .context("reading contexts.jsonl from session zip")?
        .unwrap_or_default();
    let mut existing_content_metadata: serde_json::Value = serde_json::json!({});
    if let Some(buf) = backend
        .read_zip_entry_text(
            zip_path,
            "content_metadata.json",
            MAX_SESSION_ARCHIVE_BYTES,
            MAX_MANIFEST_BYTES,
        )
        .context("reading content_metadata.json from session zip")?
        && let Ok(v) = serde_json::from_str(&buf)
    {
        existing_content_metadata = v;
    }
    let mut existing_group_info: serde_json::Value =
        serde_json::json!({"contextToGroupId": {}, "groupLabels": {}});
    if let Some(buf) = backend
        .read_zip_entry_text(
            zip_path,
            "group_info.json",
            MAX_SESSION_ARCHIVE_BYTES,
            MAX_MANIFEST_BYTES,
        )
        .context("reading group_info.json from session zip")?
        && let Ok(v) = serde_json::from_str(&buf)
    {
        existing_group_info = v;
    }

    let user_content_id = uuid::Uuid::new_v4().to_string();
    let response_content_id = uuid::Uuid::new_v4().to_string();
    let task_fragment_id = uuid::Uuid::new_v4().to_string();
    let new_context_id = uuid::Uuid::new_v4().to_string();
    // Persist any pre-existing summary into a content blob and
    // reference it from the task's `summaryContentId`. Usually `None`
    // on first append (compression runs later via `set_turn_summary`);
    // populated when an in-flight summary is being saved alongside
    // the turn.
    let summary_content_id: Option<String> = turn
        .summary
        .as_ref()
        .map(|_| uuid::Uuid::new_v4().to_string());
    let structured_output_content_id: Option<String> = turn
        .structured_output
        .as_ref()
        .map(|_| uuid::Uuid::new_v4().to_string());

    // Build the messages array: user message, then tool_call/tool_result
    // pairs (one of each per exchange) interleaved per the OpenAI wire
    // shape, then the final ai message. Each tool_call's argument string
    // and each tool_result's output get their own content/*.txt entry so
    // the messages JSON stays compact even for large outputs.
    //
    // `tool_exchange_content` collects (content_id, text) pairs to write
    // after the manifest/fragments rewrite below; we accumulate now so
    // we don't have to walk the exchanges twice.
    let mut messages_json = vec![serde_json::json!(
        {"role": "user", "contentId": user_content_id}
    )];
    let mut tool_exchange_content: Vec<(String, String)> = Vec::new();
    for exchange in &turn.tool_exchanges {
        let arguments_content_id = uuid::Uuid::new_v4().to_string();
        let result_content_id = uuid::Uuid::new_v4().to_string();
        messages_json.push(serde_json::json!({
            "role": "tool_call",
            "toolCallId": exchange.call_id,
            "toolName": exchange.tool_name,
            "argumentsContentId": arguments_content_id,
        }));
        messages_json.push(serde_json::json!({
            "role": "tool_result",
            "toolCallId": exchange.call_id,
            "toolName": exchange.tool_name,
            "contentId": result_content_id,
        }));
        tool_exchange_content.push((arguments_content_id, exchange.arguments.clone()));
        tool_exchange_content.push((result_content_id, exchange.result.clone()));
    }
    messages_json.push(serde_json::json!(
        {"role": "ai", "contentId": response_content_id}
    ));

    if let Some(tasks) = existing_fragments
        .get_mut("task")
        .and_then(|t| t.as_object_mut())
    {
        tasks.insert(
            task_fragment_id.clone(),
            serde_json::json!({
                "id": task_fragment_id,
                "messages": messages_json,
                "taskDescription": null,
                "markdownContentId": response_content_id,
                "structuredOutputContentId": structured_output_content_id,
                "escapeHtml": false
            }),
        );
    }

    let new_context = serde_json::json!({
        "id": new_context_id,
        "editable": [],
        "readonly": [],
        "virtuals": [task_fragment_id],
        "pinned": [],
        "tasks": [{
            "sequence": 0,
            "description": null,
            "logId": task_fragment_id,
            "llmLogId": null,
            "summaryContentId": summary_content_id,
            "taskType": null,
            "primaryModelName": null,
            "primaryModelReasoning": null
        }],
        "parsedOutputId": null
    });
    let mut contexts = existing_contexts;
    contexts.push_str(&serde_json::to_string(&new_context).unwrap_or_default());
    contexts.push('\n');

    const REWRITTEN: &[&str] = &[
        "manifest.json",
        "fragments-v4.json",
        "contexts.jsonl",
        "content_metadata.json",
        "group_info.json",
    ];

    with_temp_zip_writer(zip_path, |writer, options| {
        copy_zip_entries_via_sandbox(zip_path, writer, options, |n| REWRITTEN.contains(&n))?;

        let manifest_json =
            serde_json::to_string_pretty(manifest).context("serializing session manifest")?;
        writer.start_file("manifest.json", options)?;
        writer.write_all(manifest_json.as_bytes())?;

        writer.start_file("fragments-v4.json", options)?;
        writer.write_all(serde_json::to_string(&existing_fragments)?.as_bytes())?;

        writer.start_file("contexts.jsonl", options)?;
        writer.write_all(contexts.as_bytes())?;

        writer.start_file("content_metadata.json", options)?;
        writer.write_all(serde_json::to_string(&existing_content_metadata)?.as_bytes())?;

        writer.start_file("group_info.json", options)?;
        writer.write_all(serde_json::to_string(&existing_group_info)?.as_bytes())?;

        writer.start_file(format!("content/{user_content_id}.txt"), options)?;
        writer.write_all(turn.user_prompt.as_bytes())?;

        writer.start_file(format!("content/{response_content_id}.txt"), options)?;
        writer.write_all(turn.agent_response.as_bytes())?;

        // Tool-exchange args + results: one content/*.txt file per blob,
        // keyed by the ids we assigned in `messages_json` above. Failures
        // here propagate up and the half-written temp zip is discarded.
        for (content_id, text) in &tool_exchange_content {
            writer.start_file(format!("content/{content_id}.txt"), options)?;
            writer.write_all(text.as_bytes())?;
        }

        // Per-turn summary blob, referenced by the context's
        // `summaryContentId`. Brokk-compatible: the Java side reads
        // the same slot to render the "compressed" indicator.
        if let (Some(sid), Some(summary_text)) =
            (summary_content_id.as_ref(), turn.summary.as_ref())
        {
            writer.start_file(format!("content/{sid}.txt"), options)?;
            writer.write_all(summary_text.as_bytes())?;
        }
        if let (Some(sid), Some(structured_output)) = (
            structured_output_content_id.as_ref(),
            turn.structured_output.as_ref(),
        ) {
            writer.start_file(format!("content/{sid}.txt"), options)?;
            writer.write_all(serde_json::to_string(structured_output)?.as_bytes())?;
        }

        Ok(())
    })?;
    Ok(task_fragment_id)
}

/// Mutate the `summaryContentId` of the task whose `logId == fragment_id`
/// in `contexts.jsonl`, and write the new summary blob to
/// `content/<new_content_id>.txt`. Used by `set_turn_summary` to land
/// an LLM-produced summary onto a previously-persisted turn without
/// re-writing every other entry.
///
/// Atomic via `with_temp_zip_writer`: any failure leaves the on-disk
/// zip unchanged, so the caller can roll back its in-memory mutation
/// and keep `memory == disk`.
fn rewrite_turn_summary_in_zip(
    zip_path: &Path,
    fragment_id: &str,
    summary_text: &str,
) -> anyhow::Result<()> {
    use anyhow::Context;

    // Read current contexts.jsonl through the sandbox so a hostile zip
    // on disk can't panic the host parser.
    let backend = crate::sandbox_backend::global();
    let existing = backend
        .read_zip_entry_text(
            zip_path,
            "contexts.jsonl",
            MAX_SESSION_ARCHIVE_BYTES,
            MAX_CONTEXTS_BYTES,
        )
        .context("reading contexts.jsonl for summary rewrite")?
        .unwrap_or_default();

    let new_content_id = uuid::Uuid::new_v4().to_string();
    let mut rewritten = String::with_capacity(existing.len() + 128);
    let mut hit = false;
    for line in existing.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            rewritten.push('\n');
            continue;
        }
        let mut ctx: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => {
                // Pass malformed lines through unchanged -- losing them
                // would silently rewrite history; the next reload would
                // miss task fragments referenced only there.
                rewritten.push_str(line);
                rewritten.push('\n');
                continue;
            }
        };
        if let Some(tasks) = ctx.get_mut("tasks").and_then(|v| v.as_array_mut()) {
            for task in tasks {
                let matches = task
                    .get("logId")
                    .and_then(|v| v.as_str())
                    .map(|v| v == fragment_id)
                    .unwrap_or(false);
                if matches && let Some(obj) = task.as_object_mut() {
                    obj.insert(
                        "summaryContentId".to_string(),
                        serde_json::Value::String(new_content_id.clone()),
                    );
                    hit = true;
                }
            }
        }
        rewritten.push_str(&serde_json::to_string(&ctx).unwrap_or_else(|_| line.to_string()));
        rewritten.push('\n');
    }
    if !hit {
        anyhow::bail!("no context entry references fragment_id `{fragment_id}` in contexts.jsonl");
    }

    with_temp_zip_writer(zip_path, |writer, options| {
        copy_zip_entries_via_sandbox(zip_path, writer, options, |n| n == "contexts.jsonl")?;
        writer.start_file("contexts.jsonl", options)?;
        writer.write_all(rewritten.as_bytes())?;
        writer.start_file(format!("content/{new_content_id}.txt"), options)?;
        writer.write_all(summary_text.as_bytes())?;
        Ok(())
    })
}

/// Replace manifest.json in an existing session zip, copying all other entries as-is.
///
/// Atomic: any failure leaves the on-disk zip untouched, so callers can roll back
/// in-memory mutations and keep `memory == disk`.
fn rewrite_manifest_in_zip(zip_path: &Path, manifest: &SessionManifest) -> anyhow::Result<()> {
    use anyhow::Context;

    with_temp_zip_writer(zip_path, |writer, options| {
        // Stream every existing entry except `manifest.json` through
        // the sandbox so a mutated zip on disk cannot panic the
        // host's `zip` parser during the rewrite.
        copy_zip_entries_via_sandbox(zip_path, writer, options, |n| n == "manifest.json")?;
        let manifest_json =
            serde_json::to_string_pretty(manifest).context("serializing session manifest")?;
        writer.start_file("manifest.json", options)?;
        writer.write_all(manifest_json.as_bytes())?;
        Ok(())
    })
}

/// List all session manifests from the executor's sessions directory.
fn list_manifests_from_disk(cwd: &Path) -> Vec<SessionManifest> {
    let dir = sessions_dir(cwd);
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return vec![],
    };
    entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("zip") {
                return None;
            }
            read_manifest_from_zip(&path)
        })
        .collect()
}

/// Normalize a cwd before comparing or rooting a per-session registry.
///
/// We canonicalize when possible so equivalent spellings of the same workspace
/// (`.`/`..`, symlinks) reuse the same cached registry and Bifrost subprocess.
/// If the path no longer exists, fall back to the lexical path so callers with
/// a stale cwd still get a deterministic comparison.
fn normalize_cwd(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

// ---------------------------------------------------------------------------
// Thread-safe session store
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SessionStore {
    sessions: Arc<RwLock<HashMap<String, Session>>>,
    default_model: Arc<RwLock<String>>,
    default_reasoning_effort: Arc<RwLock<Option<String>>>,
    /// Process-local replacement for the install-level setup preference
    /// file. When present, model/reasoning/sandbox choices still seed later
    /// sessions in this server process, but they are never read from or
    /// written to `setup.json`. This is useful for scripts that pass an
    /// explicit model/sandbox choice and must not mutate the user's global
    /// Brokk configuration.
    transient_setup_state: Option<Arc<std::sync::Mutex<crate::setup_state::SetupState>>>,
    /// Last-known catalog of models, populated by `set_available_models`
    /// from the LLM endpoint. Used to fulfil `session/new` without
    /// re-fetching on every call, and to look up per-model reasoning
    /// presets when the agent needs to resolve "no user pick" to the
    /// model's `default_reasoning_level`. Stored as full
    /// `ModelMetadata` records so the picker and the reasoning resolver
    /// share one source of truth -- string ids alone would force a
    /// parallel cache to avoid drift.
    available_models: Arc<RwLock<Vec<ModelMetadata>>>,
    cancel_tokens: Arc<RwLock<HashMap<String, CancellationToken>>>,
    /// One ToolRegistry per session, kept warm across turns so any MCP
    /// subprocesses survive. Populated lazily on first prompt.
    registries: Arc<RwLock<HashMap<String, Arc<ToolRegistry>>>>,
    /// Per-session monotonic access counter used for LRU eviction. Held
    /// behind a sync `Mutex` because every touch is a fast in-memory bump
    /// and must not require holding a tokio lock across `.await` points.
    last_accessed: Arc<std::sync::Mutex<HashMap<String, u64>>>,
    next_access: Arc<AtomicU64>,
    limits: SessionLimits,
}

impl SessionStore {
    /// Build a `SessionStore` with `SessionLimits::default()`. Test-only:
    /// production code goes through `with_limits` so CLI flags drive the
    /// caps. Gated on `cfg(test)` because in a binary crate `dead_code` is
    /// evaluated separately per build target, and the bin target never
    /// calls this constructor.
    #[cfg(test)]
    pub fn new(default_model: String) -> Self {
        Self::with_limits(default_model, SessionLimits::default())
    }

    #[cfg(test)]
    pub fn with_limits(default_model: String, limits: SessionLimits) -> Self {
        Self::with_limits_and_transient_setup(default_model, limits, false)
    }

    pub fn with_limits_and_transient_setup(
        default_model: String,
        limits: SessionLimits,
        transient_setup: bool,
    ) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            default_model: Arc::new(RwLock::new(default_model)),
            default_reasoning_effort: Arc::new(RwLock::new(None)),
            transient_setup_state: transient_setup.then(|| {
                Arc::new(std::sync::Mutex::new(
                    crate::setup_state::SetupState::default(),
                ))
            }),
            available_models: Arc::new(RwLock::new(Vec::new())),
            cancel_tokens: Arc::new(RwLock::new(HashMap::new())),
            registries: Arc::new(RwLock::new(HashMap::new())),
            last_accessed: Arc::new(std::sync::Mutex::new(HashMap::new())),
            next_access: Arc::new(AtomicU64::new(0)),
            limits,
        }
    }

    /// Bump the LRU "last accessed" counter for `id`. Cheap: a single
    /// `AtomicU64::fetch_add` + a `HashMap::insert` under a sync mutex held
    /// for one statement.
    fn touch(&self, id: &str) {
        let counter = self.next_access.fetch_add(1, Ordering::Relaxed);
        self.last_accessed
            .lock()
            .expect("last_accessed mutex poisoned")
            .insert(id.to_string(), counter);
    }

    pub(crate) fn setup_state_snapshot(&self) -> crate::setup_state::SetupState {
        match &self.transient_setup_state {
            Some(state) => state
                .lock()
                .expect("transient setup state mutex poisoned")
                .clone(),
            None => crate::setup_state::read(),
        }
    }

    pub(crate) fn remember_first_run_seen(&self) -> anyhow::Result<()> {
        match &self.transient_setup_state {
            Some(state) => {
                state
                    .lock()
                    .expect("transient setup state mutex poisoned")
                    .first_run_seen = true;
                Ok(())
            }
            None => crate::setup_state::mark_first_run_seen(),
        }
    }

    fn read_sandbox_mode_preference(&self) -> Option<Option<crate::sandbox_backend::SandboxMode>> {
        match &self.transient_setup_state {
            Some(state) => Some(
                state
                    .lock()
                    .expect("transient setup state mutex poisoned")
                    .last_sandbox_mode,
            ),
            None => crate::setup_state::read_sandbox_mode_preference(),
        }
    }

    fn remember_last_selection(
        &self,
        model: Option<String>,
        reasoning_effort: Option<String>,
    ) -> anyhow::Result<()> {
        match &self.transient_setup_state {
            Some(state) => {
                let mut state = state.lock().expect("transient setup state mutex poisoned");
                state.last_model = model;
                state.last_reasoning_effort = reasoning_effort;
                Ok(())
            }
            None => crate::setup_state::remember_last_selection(model, reasoning_effort),
        }
    }

    fn remember_last_reasoning_effort(
        &self,
        reasoning_effort: Option<String>,
    ) -> anyhow::Result<()> {
        match &self.transient_setup_state {
            Some(state) => {
                state
                    .lock()
                    .expect("transient setup state mutex poisoned")
                    .last_reasoning_effort = reasoning_effort;
                Ok(())
            }
            None => crate::setup_state::remember_last_reasoning_effort(reasoning_effort),
        }
    }

    fn remember_sandbox_mode(
        &self,
        mode: Option<crate::sandbox_backend::SandboxMode>,
    ) -> anyhow::Result<()> {
        match &self.transient_setup_state {
            Some(state) => {
                state
                    .lock()
                    .expect("transient setup state mutex poisoned")
                    .last_sandbox_mode = mode;
                Ok(())
            }
            None => crate::setup_state::remember_sandbox_mode(mode),
        }
    }

    /// If `sessions.len()` exceeds `limits.max_sessions`, evict the least
    /// recently used session(s) from memory. Sessions with an in-flight
    /// prompt (cancel token registered via `start_prompt`) are skipped --
    /// evicting them would remove the in-memory state the running prompt is
    /// still mutating.
    ///
    /// Eviction is in-memory only. The on-disk zip is untouched, so a
    /// subsequent `session/load` re-hydrates the session unchanged.
    async fn enforce_session_cap(&self) {
        let max = self.limits.max_sessions;
        if max == 0 {
            return;
        }
        let to_evict: Vec<String> = {
            let sessions = self.sessions.read().await;
            if sessions.len() <= max {
                return;
            }
            let in_flight = self.cancel_tokens.read().await;
            let last_accessed = self
                .last_accessed
                .lock()
                .expect("last_accessed mutex poisoned");
            let excess = sessions.len() - max;
            let mut candidates: Vec<(String, u64)> = sessions
                .keys()
                .filter(|id| !in_flight.contains_key(id.as_str()))
                .map(|id| (id.clone(), last_accessed.get(id).copied().unwrap_or(0)))
                .collect();
            candidates.sort_by_key(|(_, c)| *c);
            candidates
                .into_iter()
                .take(excess)
                .map(|(id, _)| id)
                .collect()
        };
        if to_evict.is_empty() {
            return;
        }
        {
            let mut sessions = self.sessions.write().await;
            let mut registries = self.registries.write().await;
            let mut last_accessed = self
                .last_accessed
                .lock()
                .expect("last_accessed mutex poisoned");
            for id in &to_evict {
                sessions.remove(id);
                registries.remove(id);
                last_accessed.remove(id);
            }
        }
        tracing::info!(
            evicted = to_evict.len(),
            "evicted least-recently-used sessions from memory: {to_evict:?}"
        );
    }

    /// Return the cached ToolRegistry for `session_id`, or build one and cache it.
    /// MCP server config is consulted only when the registry is built. If
    /// config changes, callers should invalidate the registry first.
    ///
    /// If the session cwd is unchanged, refresh the cached `SkillRegistry`
    /// in place so `activate_skill` sees the latest on-disk skills without
    /// respawning Bifrost.
    /// If the cwd changed, drop the stale registry and rebuild it so the
    /// registry's own cwd and Bifrost subprocess are rooted in the new
    /// workspace.
    pub async fn get_or_create_registry(
        &self,
        session_id: &str,
        cwd: PathBuf,
    ) -> Arc<ToolRegistry> {
        let normalized_cwd = normalize_cwd(&cwd);
        let (skills, agents) = {
            let sessions = self.sessions.read().await;
            match sessions.get(session_id) {
                Some(s) => (s.skills.clone(), s.agents.clone()),
                None => (
                    Arc::new(crate::skills::SkillRegistry::default()),
                    Arc::new(crate::agents::AgentRegistry::default()),
                ),
            }
        };

        if let Some(existing) = self.registries.read().await.get(session_id).cloned()
            && existing.cwd() == normalized_cwd.as_path()
        {
            existing.set_skills(skills).await;
            existing.set_agents(agents).await;
            return existing;
        }
        let mcp_servers = {
            let sessions = self.sessions.read().await;
            let extra_servers = sessions
                .get(session_id)
                .and_then(|session| session.mcp_servers.clone());
            effective_mcp_servers(extra_servers)
        };
        let registry =
            Arc::new(ToolRegistry::new(normalized_cwd, mcp_servers, skills, agents).await);
        self.registries
            .write()
            .await
            .insert(session_id.to_string(), registry.clone());
        registry
    }

    pub async fn invalidate_registry(&self, session_id: &str) {
        self.registries.write().await.remove(session_id);
    }

    pub async fn set_available_models(&self, models: Vec<ModelMetadata>) {
        *self.available_models.write().await = models;
    }

    /// Bare ids only, in display order. Existing callers (the picker
    /// builder, the model-validation arm of `set_model`) just want the
    /// catalog of ids; metadata-aware callers reach for
    /// `available_model_metadata` instead.
    pub async fn available_models(&self) -> Vec<String> {
        self.available_models
            .read()
            .await
            .iter()
            .map(|m| m.id.clone())
            .collect()
    }

    /// Full catalog snapshot, including per-model reasoning presets.
    /// Used by the agent to render the reasoning-effort picker and to
    /// resolve "user has no pick" to the model's
    /// `default_reasoning_level` before issuing a request.
    pub async fn available_model_metadata(&self) -> Vec<ModelMetadata> {
        self.available_models.read().await.clone()
    }

    #[cfg(test)]
    pub async fn create_session(&self, cwd: PathBuf) -> Session {
        self.create_session_with_mcp_servers(cwd, None).await
    }

    /// Create a new session and write it to disk as a zip.
    pub async fn create_session_with_mcp_servers(
        &self,
        cwd: PathBuf,
        mcp_servers: Option<Vec<McpServerConfig>>,
    ) -> Session {
        let id = uuid::Uuid::new_v4().to_string();
        let prefs = self.setup_state_snapshot();
        let default_model = self.default_model.read().await.clone();
        let default_reasoning_effort = self.default_reasoning_effort.read().await.clone();
        let catalog = self.available_models.read().await.clone();
        let model = select_session_model(prefs.last_model, default_model, &catalog);
        let reasoning_effort = select_session_reasoning_effort(
            &model,
            default_reasoning_effort.or(prefs.last_reasoning_effort),
            &catalog,
        );
        let sandbox_mode = usable_sandbox_mode_preference(prefs.last_sandbox_mode);
        let mut session = match sandbox_mode {
            Some(mode) => Session::new_with_sandbox_mode(
                id.clone(),
                cwd.clone(),
                model,
                DEFAULT_SESSION_NAME.to_string(),
                Some(mode),
            ),
            None => Session::new(
                id.clone(),
                cwd.clone(),
                model,
                DEFAULT_SESSION_NAME.to_string(),
            ),
        };
        session.mcp_servers = mcp_servers.clone();
        session.manifest.brokk_mcp_servers = mcp_servers;
        session.selected_reasoning_effort = reasoning_effort;
        session.set_always_allow_keys(read_repo_always_allow_keys(&session.permission_scope_root));

        // Write to disk on a blocking worker so we don't stall the tokio runtime.
        // Persistence failures are logged but not surfaced: `create_session` returns
        // an in-memory session today, and changing the API to `Result` would ripple
        // through every `session/new` caller. Reload after a write failure simply
        // returns no manifest from disk.
        let zip_path = session_zip_path(&cwd, &id);
        let manifest = session.manifest.clone();
        match tokio::task::spawn_blocking(move || write_new_session_zip(&zip_path, &manifest)).await
        {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                tracing::warn!(session_id = %id, "failed to write new session zip: {e:#}")
            }
            Err(e) => tracing::warn!(session_id = %id, "session zip writer task panicked: {e}"),
        }

        self.sessions
            .write()
            .await
            .insert(id.clone(), session.clone());
        self.touch(&id);
        self.enforce_session_cap().await;
        session
    }

    /// Get session from memory, or load from disk if it exists.
    ///
    /// Note: this clones the full `Session` (including the conversation history
    /// vec). Callers on the hot prompt path should prefer `build_prompt_state`,
    /// which constructs the message list under the read lock without copying
    /// the history twice.
    pub async fn get_session(&self, id: &str, cwd: &Path) -> Option<Session> {
        if !self.load_into_memory_if_cold(id, cwd).await {
            return None;
        }
        let cloned = self.sessions.read().await.get(id).cloned();
        if cloned.is_some() {
            self.touch(id);
        }
        cloned
    }

    /// Ensure the session is in memory, loading it from disk if needed.
    /// Returns true iff the session ends up loaded.
    ///
    /// Note: if a session is already in memory, this is a no-op and the
    /// disk copy is NOT re-read. Live in-memory state (e.g. an updated
    /// `permission_mode` or pushed history turn) takes precedence over
    /// whatever is on disk.
    async fn load_into_memory_if_cold(&self, id: &str, cwd: &Path) -> bool {
        if self.sessions.read().await.contains_key(id) {
            return true;
        }
        let zip_path = session_zip_path(cwd, id);
        let loaded = tokio::task::spawn_blocking(move || {
            let manifest = read_manifest_from_zip(&zip_path)?;
            let history = read_history_from_zip(&zip_path);
            Some((manifest, history))
        })
        .await
        .ok()
        .flatten();
        let Some((manifest, mut history)) = loaded else {
            return false;
        };

        // Apply the in-memory sliding window before constructing the session,
        // so `Session.history.len()` never exceeds `max_history_turns`.
        // Disk is unaffected: the persisted zip still contains every turn.
        trim_history(&mut history, self.limits.max_history_turns);

        // Prefer persisted mode/model; fall back to server defaults.
        let session_mode = manifest
            .mode
            .as_deref()
            .and_then(SessionMode::parse)
            .unwrap_or(SessionMode::Lutz);
        let model = match manifest.model.clone() {
            Some(m) if !m.is_empty() => m,
            _ => self.default_model.read().await.clone(),
        };

        let prefs = self.setup_state_snapshot();
        let sandbox_mode = usable_sandbox_mode_preference(prefs.last_sandbox_mode);
        let loaded_session = match sandbox_mode {
            Some(sandbox_mode) => Session::from_persisted_with_sandbox_mode(
                id.to_string(),
                cwd.to_path_buf(),
                session_mode,
                model,
                history,
                manifest,
                Some(sandbox_mode),
            ),
            None => Session::from_persisted(
                id.to_string(),
                cwd.to_path_buf(),
                session_mode,
                model,
                history,
                manifest,
            ),
        };
        let mut session = match loaded_session {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "rejecting persisted session");
                return false;
            }
        };
        session.set_always_allow_keys(read_repo_always_allow_keys(&session.permission_scope_root));
        let inserted = {
            let mut sessions = self.sessions.write().await;
            // Race window: another task may have inserted under the same id while
            // we read from disk. `or_insert` keeps the existing in-memory entry
            // (which may carry mutations not yet persisted, like a newer
            // `permission_mode` or pushed turn) and silently drops our freshly
            // loaded copy. The on-disk zip is read-only on this path so no
            // information is lost.
            let len_before = sessions.len();
            sessions.entry(id.to_string()).or_insert(session);
            sessions.len() > len_before
        };
        if inserted {
            self.touch(id);
            self.enforce_session_cap().await;
        }
        true
    }

    /// Snapshot the per-session data needed to start a prompt turn,
    /// cloning the conversation history exactly once (under the read lock).
    /// Callers consume `history` to build protocol-specific message types
    /// without further string copies.
    pub async fn snapshot(&self, id: &str, fallback_cwd: &Path) -> Option<SessionSnapshot> {
        if !self.load_into_memory_if_cold(id, fallback_cwd).await {
            return None;
        }
        // Build the snapshot under the sessions read lock, then resolve
        // reasoning effort under the available_models lock outside it.
        // Holding both at once is unnecessary and would invite a lock
        // ordering hazard with set_model (which writes sessions then
        // reads available_models on auto-fallback).
        let (snap_base, selected_effort, idle_timeout_secs, project_instructions, skills) = {
            let sessions = self.sessions.read().await;
            let s = sessions.get(id)?;
            (
                (s.cwd.clone(), s.mode, s.model.clone(), s.history.clone()),
                s.selected_reasoning_effort.clone(),
                s.idle_timeout_secs,
                s.project_instructions.clone(),
                s.skills.clone(),
            )
        };
        let (cwd, mode, model, history) = snap_base;
        // Resolve "user has no pick" to the model's
        // default_reasoning_level so the backend gets a concrete
        // intent. Models that publish no presets resolve to None and
        // the backend omits the field entirely.
        let reasoning_effort = match selected_effort {
            Some(eff) => Some(eff),
            None => self
                .available_models
                .read()
                .await
                .iter()
                .find(|m| m.id == model)
                .and_then(|m| m.default_reasoning_level.clone()),
        };
        self.touch(id);
        Some(SessionSnapshot {
            cwd,
            mode,
            model,
            history,
            reasoning_effort,
            idle_timeout_secs,
            project_instructions,
            skills,
        })
    }

    /// Return session metadata formatted for ACP clients.
    pub(crate) async fn session_metadata(&self, id: &str) -> Option<SessionMetadata> {
        let sessions = self.sessions.read().await;
        let session = sessions.get(id)?;
        Some(SessionMetadata {
            title: session.manifest.title(),
            updated_at: session.manifest.updated_at(),
        })
    }

    /// If the session still has the placeholder title, derive a better
    /// one from the provided prompt seed and persist it alongside the
    /// turn history.
    pub(crate) async fn maybe_rename_from_prompt(
        &self,
        id: &str,
        title_seed: &str,
    ) -> anyhow::Result<Option<String>> {
        let snapshot = {
            let mut sessions = self.sessions.write().await;
            match sessions.get_mut(id) {
                Some(session) => {
                    if session.manifest.title().is_some() {
                        return Ok(None);
                    }
                    let Some(title) = derive_session_title(title_seed) else {
                        return Ok(None);
                    };
                    let prev_name = session.manifest.name.clone();
                    let prev_modified = session.manifest.modified;
                    session.manifest.name = title.clone();
                    session.manifest.modified = current_timestamp_millis();
                    Some((
                        session.cwd.clone(),
                        session.manifest.clone(),
                        prev_name,
                        prev_modified,
                        title,
                    ))
                }
                None => None,
            }
        };
        let Some((cwd, manifest, prev_name, prev_modified, title)) = snapshot else {
            return Ok(None);
        };

        let zip_path = session_zip_path(&cwd, id);
        let join_result =
            tokio::task::spawn_blocking(move || rewrite_manifest_in_zip(&zip_path, &manifest))
                .await;

        let persist_result = match join_result {
            Ok(r) => r,
            Err(join_err) => Err(anyhow::anyhow!(
                "session persistence task panicked: {join_err}"
            )),
        };

        if let Err(e) = persist_result {
            tracing::error!(
                session_id = %id,
                "failed to persist session title; rolling back in-memory state: {e:#}"
            );
            if let Some(session) = self.sessions.write().await.get_mut(id) {
                session.manifest.name = prev_name;
                session.manifest.modified = prev_modified;
            }
            return Err(e);
        }

        self.touch(id);
        Ok(Some(title))
    }

    pub async fn update_cwd(&self, id: &str, cwd: PathBuf) {
        // Re-discover AGENTS.md/CLAUDE.md, SKILL.md, and subagent `*.md`
        // files against the new cwd before taking the write lock: file
        // I/O off the lock keeps prompt turns and other session
        // mutations unblocked. The discovery results are then swapped
        // in atomically alongside the cwd.
        let sandbox_mode = {
            let sessions = self.sessions.read().await;
            let Some(session) = sessions.get(id) else {
                return;
            };
            session.sandbox_mode
        };
        let project_instructions = crate::agents_md::discover_with_sandbox_mode(&cwd, sandbox_mode);
        let skills = Arc::new(crate::skills::discover_with_sandbox_mode(
            &cwd,
            sandbox_mode,
        ));
        let agents = Arc::new(crate::agents::discover_with_sandbox_mode(
            &cwd,
            sandbox_mode,
        ));
        let permission_scope_root = permission_scope_root(&cwd);
        let repo_always_allow = read_repo_always_allow_keys(&permission_scope_root);
        if let Some(session) = self.sessions.write().await.get_mut(id) {
            session.cwd = cwd;
            session.permission_scope_root = permission_scope_root;
            session.project_instructions = project_instructions;
            session.skills = skills;
            session.agents = agents;
            session.set_always_allow_keys(repo_always_allow);
            // cwd changed -> previously-activated skills may no longer
            // be relevant. Clear so the model can re-activate against
            // the new catalog without stale dedup entries.
            session.activated_skills.clear();
        }
    }

    /// Mark a skill as activated for this session so the
    /// `activate_skill` tool can short-circuit a re-injection. Returns
    /// `true` if the name was inserted, `false` if it was already
    /// present or the session is unknown.
    pub async fn mark_skill_activated(&self, id: &str, name: &str) -> bool {
        let mut sessions = self.sessions.write().await;
        match sessions.get_mut(id) {
            Some(s) => s.activated_skills.insert(name.to_string()),
            None => false,
        }
    }

    /// Update the session's permission mode. Returns false if the session is unknown.
    /// Permission mode is intentionally session-only (not persisted to the manifest).
    pub async fn set_permission_mode(&self, id: &str, permission_mode: PermissionMode) -> bool {
        let mut sessions = self.sessions.write().await;
        match sessions.get_mut(id) {
            Some(session) => {
                session.permission_mode = permission_mode;
                true
            }
            None => false,
        }
    }

    /// Read the current permission_mode for a session. Returns None if unknown.
    pub async fn permission_mode(&self, id: &str) -> Option<PermissionMode> {
        self.sessions
            .read()
            .await
            .get(id)
            .map(|s| s.permission_mode)
    }

    /// Update the session's sandbox mode override. Returns false if the
    /// session is unknown. The choice is also saved as an install-level
    /// setup preference for future new/reloaded sessions, but it is not
    /// persisted to the session manifest so a stale zip can never impose a
    /// sandbox policy on reload.
    ///
    /// `None` clears the override (revert to global default).
    pub async fn set_sandbox_mode(
        &self,
        id: &str,
        mode: Option<crate::sandbox_backend::SandboxMode>,
    ) -> bool {
        let cwd = {
            let sessions = self.sessions.read().await;
            let Some(session) = sessions.get(id) else {
                return false;
            };
            session.cwd.clone()
        };

        // Re-discover per-session context under the new parser backend so
        // `/setup sandbox wasm|off|os` takes effect immediately for future
        // prompts, not just after a cwd change.
        let project_instructions = crate::agents_md::discover_with_sandbox_mode(&cwd, mode);
        let skills = Arc::new(crate::skills::discover_with_sandbox_mode(&cwd, mode));
        let agents = Arc::new(crate::agents::discover_with_sandbox_mode(&cwd, mode));

        let mut sessions = self.sessions.write().await;
        let updated = match sessions.get_mut(id) {
            Some(session) => {
                session.sandbox_mode = mode;
                session.sandbox_mode_explicitly_set = true;
                session.project_instructions = project_instructions;
                session.skills = skills;
                session.agents = agents;
                session.activated_skills.clear();
                true
            }
            None => false,
        };
        drop(sessions);
        if updated && let Err(e) = self.remember_sandbox_mode(mode) {
            tracing::warn!(
                session_id = %id,
                "failed to persist sandbox preference: {e:#}"
            );
        }
        updated
    }

    async fn sync_sandbox_mode_from_setup_state(
        &self,
        id: &str,
    ) -> Option<Option<crate::sandbox_backend::SandboxMode>> {
        // If the session explicitly set its sandbox mode in this session,
        // respect that choice and don't auto-sync from external setup state.
        let sessions = self.sessions.read().await;
        if let Some(session) = sessions.get(id)
            && session.sandbox_mode_explicitly_set
        {
            let mode = session.sandbox_mode;
            drop(sessions);
            return Some(mode);
        }
        drop(sessions);

        let Some(persisted_mode) = self.read_sandbox_mode_preference() else {
            return self.sessions.read().await.get(id).map(|s| s.sandbox_mode);
        };
        let mode = usable_sandbox_mode_preference(persisted_mode);
        let cwd = {
            let sessions = self.sessions.read().await;
            let session = sessions.get(id)?;
            if session.sandbox_mode == mode {
                return Some(mode);
            }
            session.cwd.clone()
        };

        let (project_instructions, skills, agents) = discover_session_context(&cwd, mode);

        let mut sessions = self.sessions.write().await;
        let session = sessions.get_mut(id)?;
        if session.sandbox_mode != mode {
            session.sandbox_mode = mode;
            session.project_instructions = project_instructions;
            session.skills = skills;
            session.agents = agents;
            session.activated_skills.clear();
        }
        Some(session.sandbox_mode)
    }

    /// Read the current sandbox mode override for a session, first
    /// syncing from the install-level setup preference if it is readable.
    /// Returns `None` if the session is unknown, and `Some(None)` if it is
    /// known and using the process default.
    pub async fn sandbox_mode(
        &self,
        id: &str,
    ) -> Option<Option<crate::sandbox_backend::SandboxMode>> {
        self.sync_sandbox_mode_from_setup_state(id).await
    }

    /// True if any of the candidate approval keys is remembered for this session.
    pub async fn is_any_always_allowed(&self, id: &str, approval_keys: &[String]) -> bool {
        self.sessions
            .read()
            .await
            .get(id)
            .map(|s| {
                approval_keys
                    .iter()
                    .any(|key| s.always_allow_tools.contains(key))
            })
            .unwrap_or(false)
    }

    /// Add `approval_key` to the current repo's remembered approval set.
    pub async fn add_always_allow(&self, id: &str, approval_key: &str) {
        if approval_key.is_empty() {
            return;
        }
        let Some(scope_root) = ({
            let sessions = self.sessions.read().await;
            sessions
                .get(id)
                .map(|session| session.permission_scope_root.clone())
        }) else {
            return;
        };

        let changed = {
            let mut sessions = self.sessions.write().await;
            // Re-validate: the originating session must still exist with the
            // same scope_root read before acquiring the write lock.  Without
            // this check a session that is closed and replaced by a new one in
            // a different repo between the two locks could receive an approval
            // that was scoped to the original repo.
            let scope_matches = sessions
                .get(id)
                .map(|s| s.permission_scope_root == scope_root)
                .unwrap_or(false);
            if !scope_matches {
                return;
            }
            let mut changed = false;
            for session in sessions.values_mut() {
                if session.permission_scope_root == scope_root
                    && session.always_allow_tools.insert(approval_key.to_string())
                {
                    session.always_allow_order.push(approval_key.to_string());
                    changed = true;
                }
            }
            changed
        };
        if changed && let Err(e) = remember_repo_always_allow_key(&scope_root, approval_key) {
            tracing::warn!(
                session_id = %id,
                repo_root = %scope_root.display(),
                "failed to persist repo Always allow approval: {e:#}"
            );
        }
    }

    /// Return remembered always-allow keys in approval order.
    pub async fn always_allow_keys(&self, id: &str) -> Option<Vec<String>> {
        let sessions = self.sessions.read().await;
        let session = sessions.get(id)?;
        Some(session.always_allow_order.clone())
    }

    /// Remove one remembered always-allow key. Returns `None` if the session is unknown.
    pub async fn remove_always_allow(&self, id: &str, approval_key: &str) -> Option<bool> {
        let (scope_root, remove_repo) = {
            let sessions = self.sessions.read().await;
            let session = sessions.get(id)?;
            (
                session.permission_scope_root.clone(),
                session.always_allow_tools.contains(approval_key),
            )
        };
        if !remove_repo {
            return Some(false);
        }

        {
            let mut sessions = self.sessions.write().await;
            for session in sessions.values_mut() {
                if session.permission_scope_root == scope_root {
                    session.always_allow_tools.remove(approval_key);
                    session.always_allow_order.retain(|key| key != approval_key);
                }
            }
        }
        if remove_repo && let Err(e) = forget_repo_always_allow_key(&scope_root, approval_key) {
            tracing::warn!(
                session_id = %id,
                repo_root = %scope_root.display(),
                "failed to persist repo Always allow revocation: {e:#}"
            );
        }
        Some(true)
    }

    /// Clear all remembered always-allow keys.
    pub async fn clear_always_allow(&self, id: &str) -> Option<usize> {
        let (scope_root, count) = {
            let sessions = self.sessions.read().await;
            let session = sessions.get(id)?;
            (
                session.permission_scope_root.clone(),
                session.always_allow_tools.len(),
            )
        };
        if count == 0 {
            return Some(0);
        }

        {
            let mut sessions = self.sessions.write().await;
            for session in sessions.values_mut() {
                if session.permission_scope_root == scope_root {
                    session.always_allow_tools.clear();
                    session.always_allow_order.clear();
                }
            }
        }
        if let Err(e) = clear_repo_always_allow_keys(&scope_root) {
            tracing::warn!(
                session_id = %id,
                repo_root = %scope_root.display(),
                "failed to persist repo Always allow clear: {e:#}"
            );
        }
        Some(count)
    }

    /// Fold a turn's token usage into the session running total and
    /// return the new cumulative figure. Used by the prompt handler to
    /// populate `PromptResponse.usage` after `tool_loop::run` reports
    /// what the LLM(s) burned for this prompt. Returns `None` if the
    /// session doesn't exist (e.g. raced with a delete).
    pub async fn record_usage(
        &self,
        id: &str,
        delta: crate::llm_client::TokenUsage,
    ) -> Option<crate::llm_client::TokenUsage> {
        let mut sessions = self.sessions.write().await;
        let session = sessions.get_mut(id)?;
        session.usage.add(delta);
        Some(session.usage)
    }

    /// Update the session's behavior mode and persist the new manifest.
    ///
    /// Returns `Ok(true)` on success, `Ok(false)` if the session is unknown
    /// (no-op), or `Err` if persistence failed -- in that case the in-memory
    /// mutation is rolled back so `memory == disk`, mirroring `add_turn`.
    pub async fn set_mode(&self, id: &str, mode: SessionMode) -> anyhow::Result<bool> {
        // Update in-memory state and snapshot what we need for persistence.
        // Capture the pre-mutation mode so we can reverse it on failure.
        let snapshot = {
            let mut sessions = self.sessions.write().await;
            match sessions.get_mut(id) {
                Some(session) => {
                    let prev_mode = session.mode;
                    let prev_manifest_mode = session.manifest.mode.clone();
                    session.mode = mode;
                    session.manifest.mode = Some(mode.as_str().to_string());
                    Some((
                        session.cwd.clone(),
                        session.manifest.clone(),
                        prev_mode,
                        prev_manifest_mode,
                    ))
                }
                None => None,
            }
        };
        let Some((cwd, manifest, prev_mode, prev_manifest_mode)) = snapshot else {
            return Ok(false);
        };

        let zip_path = session_zip_path(&cwd, id);
        let join_result =
            tokio::task::spawn_blocking(move || rewrite_manifest_in_zip(&zip_path, &manifest))
                .await;

        let persist_result = match join_result {
            Ok(r) => r,
            Err(join_err) => Err(anyhow::anyhow!(
                "session persistence task panicked: {join_err}"
            )),
        };

        if let Err(e) = persist_result {
            tracing::error!(
                session_id = %id,
                "failed to persist session mode; rolling back in-memory state: {e:#}"
            );
            if let Some(session) = self.sessions.write().await.get_mut(id) {
                session.mode = prev_mode;
                session.manifest.mode = prev_manifest_mode;
            }
            return Err(e);
        }
        Ok(true)
    }

    /// Store an LLM-produced summary onto a previously-persisted turn,
    /// reachable both by index into the in-memory history and via the
    /// turn's stable fragment id. Mirrors Brokk's
    /// `ContextManager.compressHistory(TaskEntry)`: the original log
    /// stays on disk untouched; only `summaryContentId` flips so the
    /// next prompt build substitutes the summary in place of the
    /// verbatim turn.
    ///
    /// Returns:
    /// - `Ok(true)` on success.
    /// - `Ok(false)` for an unknown session, an out-of-range
    ///   `turn_index`, or a turn whose `fragment_id` is `None` (a turn
    ///   that hasn't completed its initial persistence yet -- a
    ///   programming error rather than a runtime condition).
    /// - `Err` only when the disk rewrite fails. The in-memory
    ///   mutation is rolled back so `memory == disk`.
    pub async fn set_turn_summary(
        &self,
        id: &str,
        turn_index: usize,
        summary: String,
    ) -> anyhow::Result<bool> {
        let snapshot = {
            let mut sessions = self.sessions.write().await;
            let Some(session) = sessions.get_mut(id) else {
                return Ok(false);
            };
            if turn_index >= session.history.len() {
                tracing::warn!(
                    session_id = %id,
                    turn_index,
                    history_len = session.history.len(),
                    "rejecting set_turn_summary: turn_index out of range"
                );
                return Ok(false);
            }
            let Some(fragment_id) = session.history[turn_index].fragment_id.clone() else {
                tracing::warn!(
                    session_id = %id,
                    turn_index,
                    "rejecting set_turn_summary: turn has no fragment_id (was it persisted?)"
                );
                return Ok(false);
            };
            let prev_summary = session.history[turn_index].summary.clone();
            session.history[turn_index].summary = Some(summary.clone());
            Some((session.cwd.clone(), fragment_id, prev_summary))
        };
        let Some((cwd, fragment_id, prev_summary)) = snapshot else {
            return Ok(false);
        };

        let zip_path = session_zip_path(&cwd, id);
        let summary_for_zip = summary.clone();
        let join_result = tokio::task::spawn_blocking(move || {
            rewrite_turn_summary_in_zip(&zip_path, &fragment_id, &summary_for_zip)
        })
        .await;
        let persist_result = match join_result {
            Ok(r) => r,
            Err(join_err) => Err(anyhow::anyhow!(
                "session persistence task panicked: {join_err}"
            )),
        };
        if let Err(e) = persist_result {
            tracing::error!(
                session_id = %id,
                turn_index,
                "failed to persist turn summary; rolling back in-memory state: {e:#}"
            );
            if let Some(session) = self.sessions.write().await.get_mut(id)
                && let Some(turn) = session.history.get_mut(turn_index)
            {
                turn.summary = prev_summary;
            }
            return Err(e);
        }
        Ok(true)
    }

    /// Update the per-session LLM model. Returns false if the session is
    /// unknown. Persists the new model into the session manifest so it
    /// survives a reload, mirroring `set_mode`.
    ///
    /// When the previously-selected reasoning effort isn't in the new
    /// model's supported set, the selection is auto-cleared so the next
    /// turn falls back to the new model's `default_reasoning_level`.
    /// Returns the cleared value (if any) so the caller can notify the
    /// user, since silently dropping the pick would look like a bug
    /// next time they wonder why thoughts shortened.
    pub async fn set_model(&self, id: &str, model: String) -> (bool, Option<String>) {
        // Pull the supported-effort set for the new model BEFORE
        // acquiring the sessions write lock -- the available_models
        // store and the sessions store are separate locks, and reading
        // available_models while holding sessions in write would
        // invert the lock order taken by `snapshot()`.
        let supported_effort: Option<Vec<String>> = {
            let catalog = self.available_models.read().await;
            catalog.iter().find(|m| m.id == model).map(|m| {
                m.supported_reasoning_levels
                    .iter()
                    .map(|p| p.effort.clone())
                    .collect()
            })
        };

        let (snapshot, cleared_effort) = {
            let mut sessions = self.sessions.write().await;
            match sessions.get_mut(id) {
                Some(session) => {
                    session.model = model.clone();
                    session.manifest.model = if model.is_empty() {
                        None
                    } else {
                        Some(model.clone())
                    };
                    // Auto-fallback: if the user had a pick but the new
                    // model doesn't advertise it, drop the pick. The
                    // next snapshot resolves to the new model's default.
                    let cleared = match (&session.selected_reasoning_effort, &supported_effort) {
                        (Some(eff), Some(supported)) if !supported.iter().any(|s| s == eff) => {
                            session.selected_reasoning_effort.take()
                        }
                        // No catalog metadata for the new model (e.g.
                        // Ollama, or model discovery hasn't run yet):
                        // leave the pick as-is. We don't have evidence
                        // the pick is invalid; let the server be the
                        // arbiter rather than dropping silently.
                        _ => None,
                    };
                    if let Err(e) = self.remember_last_selection(
                        if model.is_empty() {
                            None
                        } else {
                            Some(model.clone())
                        },
                        session.selected_reasoning_effort.clone(),
                    ) {
                        tracing::warn!(
                            session_id = %id,
                            "failed to persist last model/reasoning preference: {e:#}"
                        );
                    }
                    (
                        Some((session.cwd.clone(), session.manifest.clone())),
                        cleared,
                    )
                }
                None => (None, None),
            }
        };
        match snapshot {
            Some((cwd, manifest)) => {
                let zip_path = session_zip_path(&cwd, id);
                let _ = tokio::task::spawn_blocking(move || {
                    rewrite_manifest_in_zip(&zip_path, &manifest)
                })
                .await;
                (true, cleared_effort)
            }
            None => (false, None),
        }
    }

    /// Record the user's reasoning-effort pick for this session.
    /// `None` clears it (back to "use model default"). Returns false
    /// if the session is unknown. The session manifest still does not
    /// persist this; the install-wide preference file is updated
    /// separately so new sessions can inherit the last explicit pick.
    pub async fn set_reasoning_effort(&self, id: &str, effort: Option<String>) -> bool {
        let mut sessions = self.sessions.write().await;
        match sessions.get_mut(id) {
            Some(session) => {
                session.selected_reasoning_effort = effort;
                if let Err(e) =
                    self.remember_last_reasoning_effort(session.selected_reasoning_effort.clone())
                {
                    tracing::warn!(
                        session_id = %id,
                        "failed to persist last reasoning preference: {e:#}"
                    );
                }
                true
            }
            None => false,
        }
    }

    /// Record the user's per-session LLM idle-timeout override.
    /// `None` clears it (back to the binary-wide default). Returns false
    /// if the session is unknown. In-memory only -- not persisted.
    pub async fn set_idle_timeout_secs(&self, id: &str, secs: Option<u64>) -> bool {
        let mut sessions = self.sessions.write().await;
        match sessions.get_mut(id) {
            Some(session) => {
                session.idle_timeout_secs = secs;
                true
            }
            None => false,
        }
    }

    pub async fn set_default_model(&self, model: String) {
        *self.default_model.write().await = model;
    }

    pub async fn set_default_reasoning_effort(&self, effort: Option<String>) {
        *self.default_reasoning_effort.write().await = effort;
    }

    pub async fn default_model(&self) -> String {
        self.default_model.read().await.clone()
    }

    /// Add a conversation turn and persist it to the session zip.
    ///
    /// Returns `Ok(())` on successful persistence, or `Err` if the zip
    /// rewrite failed. On error the in-memory turn is rolled back so memory
    /// and disk stay in sync: the next prompt either succeeds (with both
    /// written) or surfaces the same error again deterministically. The
    /// atomic temp-then-rename in `append_turn_to_zip` guarantees the
    /// on-disk zip is unchanged on any failure path.
    ///
    /// Concurrency: callers must serialize `add_turn` per session. The agent
    /// achieves this by calling `start_prompt`, which refuses a second
    /// in-flight prompt for the same session, and keeping that token alive
    /// via `finish_prompt` for the entire prompt-plus-persistence window --
    /// `finish_prompt` runs *after* `add_turn` returns. Without that
    /// ordering, a second `session/prompt` could push a new turn between
    /// this turn's push and its rollback `pop()`, removing the wrong entry.
    pub async fn add_turn(&self, id: &str, turn: ConversationTurn) -> anyhow::Result<()> {
        // Mutate in-memory state first, then release the lock BEFORE blocking I/O.
        // Capture pre-mutation `modified` so we can reverse the bump on failure.
        let snapshot = {
            let mut sessions = self.sessions.write().await;
            match sessions.get_mut(id) {
                Some(session) => {
                    let prev_modified = session.manifest.modified;
                    session.history.push(turn.clone());
                    let now = current_timestamp_millis();
                    session.manifest.modified = now;
                    Some((
                        session_zip_path(&session.cwd, &session.id),
                        session.manifest.clone(),
                        prev_modified,
                    ))
                }
                None => None,
            }
        };
        let Some((zip_path, manifest, prev_modified)) = snapshot else {
            // Session is unknown -- no in-memory state to roll back, nothing
            // to persist. Treat as a no-op success.
            return Ok(());
        };

        let turn_for_zip = turn.clone();
        let join_result = tokio::task::spawn_blocking(move || {
            append_turn_to_zip(&zip_path, &manifest, &turn_for_zip)
        })
        .await;

        let persist_result: anyhow::Result<String> = match join_result {
            Ok(r) => r,
            Err(join_err) => Err(anyhow::anyhow!(
                "session persistence task panicked: {join_err}"
            )),
        };

        let assigned_fragment_id = match persist_result {
            Ok(id) => id,
            Err(e) => {
                tracing::error!(
                    session_id = %id,
                    "failed to persist conversation turn; rolling back in-memory state: {e:#}"
                );
                if let Some(session) = self.sessions.write().await.get_mut(id) {
                    session.history.pop();
                    session.manifest.modified = prev_modified;
                }
                return Err(e);
            }
        };
        // Stamp the persisted fragment id back onto the in-memory
        // turn so subsequent `set_turn_summary` calls can locate the
        // right `summaryContentId` to rewrite without re-reading the
        // zip. Trim eviction (`max_history_turns`) may have dropped
        // the pushed turn between persist and now -- it stays consistent
        // with disk either way.
        if let Some(session) = self.sessions.write().await.get_mut(id)
            && let Some(last) = session.history.last_mut()
        {
            last.fragment_id = Some(assigned_fragment_id);
        }

        // Persistence succeeded. Apply the in-memory sliding window so
        // history length stays bounded. Trimming AFTER successful persist
        // keeps the on-disk zip authoritative -- a future reload re-reads
        // the full history and trims it again on its way into memory.
        // The trim has no rollback partner because we've already committed
        // to disk; if trimming pushed something out of memory, the only
        // copy now lives in the zip, which is the design intent.
        let max_history = self.limits.max_history_turns;
        if max_history > 0
            && let Some(session) = self.sessions.write().await.get_mut(id)
        {
            trim_history(&mut session.history, max_history);
        }

        self.touch(id);
        Ok(())
    }

    /// List sessions from disk, filtered by cwd.
    pub async fn list_sessions_from_disk(&self, cwd: &Path) -> Vec<SessionManifest> {
        let cwd = cwd.to_path_buf();
        tokio::task::spawn_blocking(move || list_manifests_from_disk(&cwd))
            .await
            .unwrap_or_default()
    }

    pub async fn start_prompt(
        &self,
        session_id: &str,
    ) -> Result<CancellationToken, PromptStartError> {
        let token = CancellationToken::new();
        let mut cancel_tokens = self.cancel_tokens.write().await;
        match cancel_tokens.entry(session_id.to_string()) {
            Entry::Occupied(_) => Err(PromptStartError::AlreadyInFlight),
            Entry::Vacant(entry) => {
                entry.insert(token.clone());
                Ok(token)
            }
        }
    }

    pub async fn cancel_prompt(&self, session_id: &str) {
        if let Some(token) = self.cancel_tokens.read().await.get(session_id) {
            token.cancel();
        }
    }

    pub async fn finish_prompt(&self, session_id: &str) {
        self.cancel_tokens.write().await.remove(session_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup_state::TestConfigHomeScope;

    /// Build an in-memory session that has NEVER been written to disk, then
    /// call `add_turn`. The append step must fail (because the zip file
    /// doesn't exist), and the in-memory state must be rolled back so
    /// `history` is empty and `manifest.modified` matches the pre-call value.
    #[tokio::test]
    async fn add_turn_rolls_back_on_persistence_failure() {
        let store = SessionStore::with_limits("test-model".to_string(), SessionLimits::default());

        // Hand-construct a session and inject it into the in-memory map without
        // ever calling `create_session` (which would write a zip on disk).
        let id = "rollback-test".to_string();
        let cwd = std::env::temp_dir().join(format!("brokk-acp-rust-rollback-{}", id));
        let session = Session::new(id.clone(), cwd, "test-model".to_string(), "test".into());
        let pre_modified = session.manifest.modified;
        store.sessions.write().await.insert(id.clone(), session);

        let result = store
            .add_turn(
                &id,
                ConversationTurn {
                    user_prompt: "hello".into(),
                    agent_response: "world".into(),
                    ..Default::default()
                },
            )
            .await;

        assert!(
            result.is_err(),
            "add_turn should fail when the session zip doesn't exist on disk"
        );

        let sessions = store.sessions.read().await;
        let s = sessions.get(&id).expect("session still in memory");
        assert!(
            s.history.is_empty(),
            "rollback should have popped the optimistically-pushed turn, got {:?}",
            s.history
        );
        assert_eq!(
            s.manifest.modified, pre_modified,
            "rollback should restore the pre-call manifest.modified timestamp"
        );
    }

    /// `set_mode` must roll back the in-memory mode change when the zip rewrite
    /// fails, mirroring `add_turn`. Otherwise memory drifts away from disk and
    /// the next reload silently downgrades the user's selection.
    #[tokio::test]
    async fn set_mode_rolls_back_on_persistence_failure() {
        let store = SessionStore::new("test-model".to_string());

        let id = "set-mode-rollback".to_string();
        let cwd = std::env::temp_dir().join(format!("brokk-acp-rust-set-mode-{}", id));
        let session = Session::new(id.clone(), cwd, "test-model".to_string(), "test".into());
        let pre_mode = session.mode;
        let pre_manifest_mode = session.manifest.mode.clone();
        store.sessions.write().await.insert(id.clone(), session);

        let result = store.set_mode(&id, SessionMode::Plan).await;
        assert!(
            result.is_err(),
            "set_mode should fail when the session zip doesn't exist on disk"
        );

        let sessions = store.sessions.read().await;
        let s = sessions.get(&id).expect("session still in memory");
        assert_eq!(
            s.mode, pre_mode,
            "rollback should restore the previous in-memory mode"
        );
        assert_eq!(
            s.manifest.mode, pre_manifest_mode,
            "rollback should restore the previous manifest.mode"
        );
    }

    /// `set_mode` on an unknown session id is `Ok(false)` (no-op), distinct
    /// from `Err`, so callers can return a precise "unknown session" error.
    #[tokio::test]
    async fn set_mode_unknown_session_returns_ok_false() {
        let store = SessionStore::new("test-model".to_string());
        let result = store.set_mode("no-such-session", SessionMode::Code).await;
        assert!(matches!(result, Ok(false)));
    }

    /// Sanity check that `add_turn` on an unknown session id is a no-op
    /// success rather than an error -- callers may race between session
    /// removal (none today, but reserved) and turn persistence.
    #[tokio::test]
    async fn add_turn_unknown_session_is_noop_success() {
        let store = SessionStore::with_limits("test-model".to_string(), SessionLimits::default());
        let result = store
            .add_turn(
                "no-such-session",
                ConversationTurn {
                    user_prompt: "x".into(),
                    agent_response: "y".into(),
                    ..Default::default()
                },
            )
            .await;
        assert!(result.is_ok());
    }

    /// `Session::from_persisted` must always reset the transient security
    /// fields (`permission_mode`, `always_allow_tools`, `always_allow_order`),
    /// even when the caller's data was reconstructed from a manifest that may
    /// be stale or tampered. Persisted-side fields
    /// (id/cwd/mode/model/history/manifest) must round-trip unchanged when ids
    /// match.
    #[test]
    fn from_persisted_resets_transient_security_fields() {
        let manifest = SessionManifest {
            id: "abc".into(),
            name: "n".into(),
            created: 1,
            modified: 2,
            version: "4.0".into(),
            mode: Some("CODE".into()),
            model: Some("m".into()),
            brokk_mcp_servers: None,
        };
        let history = vec![ConversationTurn {
            user_prompt: "u".into(),
            agent_response: "a".into(),
            ..Default::default()
        }];

        let session = Session::from_persisted(
            "abc".into(),
            PathBuf::from("/tmp/x"),
            SessionMode::Code,
            "m".into(),
            history.clone(),
            manifest.clone(),
        )
        .expect("matching ids should succeed");

        assert_eq!(session.permission_mode, PermissionMode::Default);
        assert!(session.always_allow_tools.is_empty());
        assert!(session.always_allow_order.is_empty());

        assert_eq!(session.id, "abc");
        assert_eq!(session.cwd, PathBuf::from("/tmp/x"));
        assert_eq!(session.mode, SessionMode::Code);
        assert_eq!(session.model, "m");
        assert_eq!(session.history.len(), 1);
        assert_eq!(session.manifest.id, manifest.id);
    }

    /// `trim_history` is a sliding window: when the cap is exceeded, the
    /// oldest entries go and the most-recent `max` are kept. `0` is a
    /// sentinel for "unbounded".
    #[test]
    fn trim_history_keeps_most_recent_when_capped() {
        let mut h: Vec<ConversationTurn> = (0..5)
            .map(|i| ConversationTurn {
                user_prompt: format!("u{i}"),
                agent_response: format!("a{i}"),
                ..Default::default()
            })
            .collect();

        trim_history(&mut h, 0);
        assert_eq!(h.len(), 5, "max=0 must disable the cap");

        trim_history(&mut h, 3);
        assert_eq!(h.len(), 3);
        assert_eq!(h[0].user_prompt, "u2");
        assert_eq!(h[2].user_prompt, "u4");

        trim_history(&mut h, 10);
        assert_eq!(h.len(), 3, "below the cap is a no-op");
    }

    /// `add_turn` must enforce `max_history_turns` in memory after a
    /// successful persist. We exercise the full path: a real zip on disk
    /// (so `append_turn_to_zip` succeeds), four sequential `add_turn`
    /// calls, and a memory cap of 2 -- the in-memory history must be the
    /// last two turns, while the zip on disk retains everything.
    #[tokio::test]
    async fn add_turn_enforces_history_window_in_memory() {
        let store = SessionStore::with_limits(
            "test-model".to_string(),
            SessionLimits {
                max_sessions: 0,
                max_history_turns: 2,
            },
        );
        let cwd = std::env::temp_dir().join(format!(
            "brokk-acp-rust-history-window-{}",
            uuid::Uuid::new_v4()
        ));
        let session = store.create_session(cwd.clone()).await;
        let id = session.id.clone();

        for i in 0..4 {
            store
                .add_turn(
                    &id,
                    ConversationTurn {
                        user_prompt: format!("u{i}"),
                        agent_response: format!("a{i}"),
                        ..Default::default()
                    },
                )
                .await
                .expect("persist should succeed");
        }

        let in_mem = store.sessions.read().await.get(&id).cloned().unwrap();
        assert_eq!(in_mem.history.len(), 2, "memory must respect the cap");
        assert_eq!(in_mem.history[0].user_prompt, "u2");
        assert_eq!(in_mem.history[1].user_prompt, "u3");

        // Disk-side: the zip carries every turn we appended, regardless of
        // what we kept in memory.
        let on_disk = read_history_from_zip(&session_zip_path(&cwd, &id));
        assert_eq!(
            on_disk.len(),
            4,
            "disk history must be untouched by the in-memory window"
        );

        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// When `sessions.len()` exceeds `max_sessions`, the LRU sessions are
    /// evicted from memory. The session whose access counter was bumped most
    /// recently must survive; the oldest must be dropped.
    #[tokio::test]
    async fn lru_eviction_drops_oldest_session() {
        let store = SessionStore::with_limits(
            "test-model".to_string(),
            SessionLimits {
                max_sessions: 2,
                max_history_turns: 0,
            },
        );

        // Inject three sessions directly into the map without going through
        // create_session (which would write zips). Touch each one so the
        // last_accessed counter reflects insertion order: a < b < c.
        for id in ["a", "b", "c"] {
            let s = Session::new(
                id.into(),
                std::env::temp_dir().join(format!("brokk-acp-rust-lru-{id}")),
                "test-model".into(),
                "t".into(),
            );
            store.sessions.write().await.insert(id.into(), s);
            store.touch(id);
        }

        // Bump "b" so it's now most-recent: order becomes a (oldest), c, b.
        store.touch("b");

        store.enforce_session_cap().await;

        let sessions = store.sessions.read().await;
        assert_eq!(sessions.len(), 2);
        assert!(!sessions.contains_key("a"), "oldest LRU must be evicted");
        assert!(sessions.contains_key("b"), "recently-touched must survive");
        assert!(sessions.contains_key("c"));
    }

    /// In-flight prompts (those holding a cancellation token via
    /// `start_prompt`) must not be evicted: the running task is mutating the
    /// in-memory state and dropping it would lose the partially-completed
    /// turn.
    #[tokio::test]
    async fn lru_eviction_skips_in_flight_sessions() {
        let store = SessionStore::with_limits(
            "test-model".to_string(),
            SessionLimits {
                max_sessions: 1,
                max_history_turns: 0,
            },
        );

        for id in ["old", "fresh"] {
            let s = Session::new(
                id.into(),
                std::env::temp_dir().join(format!("brokk-acp-rust-inflight-{id}")),
                "test-model".into(),
                "t".into(),
            );
            store.sessions.write().await.insert(id.into(), s);
            store.touch(id);
        }
        // "old" is the LRU candidate, but mark it in-flight so it's pinned.
        let _token = store
            .start_prompt("old")
            .await
            .expect("in-flight session should register once");

        store.enforce_session_cap().await;

        let sessions = store.sessions.read().await;
        // We can only evict non-in-flight sessions, so "fresh" goes even
        // though it was the most-recent. "old" stays pinned.
        assert!(sessions.contains_key("old"), "in-flight session is pinned");
        assert!(
            !sessions.contains_key("fresh"),
            "the only evictable session must be dropped"
        );
    }

    /// `max_sessions = 0` must disable the cap entirely: no eviction runs
    /// even when the in-memory map grows large.
    #[tokio::test]
    async fn lru_eviction_disabled_when_max_is_zero() {
        let store = SessionStore::with_limits(
            "test-model".to_string(),
            SessionLimits {
                max_sessions: 0,
                max_history_turns: 0,
            },
        );
        for i in 0..5 {
            let id = format!("s{i}");
            let s = Session::new(
                id.clone(),
                std::env::temp_dir().join("brokk-acp-rust-uncapped"),
                "test-model".into(),
                "t".into(),
            );
            store.sessions.write().await.insert(id.clone(), s);
            store.touch(&id);
        }
        store.enforce_session_cap().await;
        assert_eq!(store.sessions.read().await.len(), 5);
    }

    /// A zip whose manifest reports a different id than the one the caller
    /// asked for must be rejected: continuing would let the in-memory map
    /// key drift away from `Session.id`, so subsequent writes would target
    /// a different zip than the one we resumed from.
    #[test]
    fn from_persisted_rejects_id_mismatch() {
        let manifest = SessionManifest {
            id: "loaded-id".into(),
            name: "n".into(),
            created: 1,
            modified: 2,
            version: "4.0".into(),
            mode: None,
            model: None,
            brokk_mcp_servers: None,
        };

        let err = Session::from_persisted(
            "requested-id".into(),
            PathBuf::from("/tmp/x"),
            SessionMode::Lutz,
            "m".into(),
            Vec::new(),
            manifest,
        )
        .expect_err("mismatched ids must be rejected");

        assert_eq!(err.requested, "requested-id");
        assert_eq!(err.loaded, "loaded-id");
    }

    /// `set_model` should update the in-memory `Session.model` and
    /// `manifest.model` so a subsequent reload from disk picks up the new
    /// value. Persistence itself is exercised by `rewrite_manifest_in_zip`
    /// (shared with `set_mode`); this test verifies wiring only.
    #[tokio::test]
    async fn set_model_updates_session_and_manifest() {
        let store = SessionStore::new("initial-model".to_string());

        // Use a unique tmp cwd so concurrent test runs don't clobber.
        let cwd =
            std::env::temp_dir().join(format!("brokk-acp-rust-set-model-{}", uuid::Uuid::new_v4()));
        let session = store.create_session(cwd).await;
        let id = session.id.clone();

        let (ok, cleared) = store.set_model(&id, "next-model".to_string()).await;
        assert!(ok);
        assert!(
            cleared.is_none(),
            "no reasoning effort was selected, nothing to clear"
        );
        let sessions = store.sessions.read().await;
        let s = sessions.get(&id).expect("session still in memory");
        assert_eq!(s.model, "next-model");
        assert_eq!(s.manifest.model.as_deref(), Some("next-model"));
    }

    #[tokio::test]
    async fn set_model_returns_false_for_unknown_session() {
        let store = SessionStore::new("initial-model".to_string());
        let (ok, cleared) = store
            .set_model("no-such-session", "next-model".into())
            .await;
        assert!(!ok);
        assert!(cleared.is_none());
    }

    /// `SessionMode::parse` round-trips every variant via its wire id, with
    /// the canonical (uppercase) spelling.
    #[test]
    fn session_mode_parse_round_trip() {
        for mode in [
            SessionMode::Lutz,
            SessionMode::Code,
            SessionMode::Ask,
            SessionMode::Plan,
        ] {
            assert_eq!(
                SessionMode::parse(mode.as_str()),
                Some(mode),
                "round-trip failed for {mode:?}"
            );
        }
    }

    /// Clients sometimes lowercase or mixed-case the wire id; `parse` must
    /// accept any casing because the wire form is documented as
    /// case-insensitive.
    #[test]
    fn session_mode_parse_is_case_insensitive() {
        assert_eq!(SessionMode::parse("lutz"), Some(SessionMode::Lutz));
        assert_eq!(SessionMode::parse("Code"), Some(SessionMode::Code));
        assert_eq!(SessionMode::parse("aSk"), Some(SessionMode::Ask));
        assert_eq!(SessionMode::parse("plan"), Some(SessionMode::Plan));
    }

    /// Unknown / empty / whitespace-only ids must return `None` rather than
    /// silently mapping to a default mode.
    #[test]
    fn session_mode_parse_rejects_unknown() {
        assert_eq!(SessionMode::parse(""), None);
        assert_eq!(SessionMode::parse("EDIT"), None);
        assert_eq!(SessionMode::parse("LUT"), None);
        assert_eq!(SessionMode::parse(" LUTZ "), None);
    }

    /// `PermissionMode::parse` round-trips every variant. Wire ids are
    /// camelCase here (mirrors `claude-agent-acp`) and case-sensitive,
    /// unlike `SessionMode`.
    #[test]
    fn permission_mode_parse_round_trip_all_variants() {
        for mode in [
            PermissionMode::Default,
            PermissionMode::AcceptEdits,
            PermissionMode::ReadOnly,
            PermissionMode::BypassPermissions,
        ] {
            assert_eq!(
                PermissionMode::parse(mode.as_str()),
                Some(mode),
                "round-trip failed for {mode:?}"
            );
        }
        assert_eq!(PermissionMode::parse(""), None);
        assert_eq!(PermissionMode::parse("ACCEPTEDITS"), None);
        assert_eq!(PermissionMode::parse("accept-edits"), None);
    }

    /// `create_session` writes a manifest.json that round-trips through
    /// `read_manifest_from_zip` -- the disk persistence format must stay
    /// stable, otherwise a session created today won't load tomorrow.
    #[tokio::test]
    async fn create_session_persists_manifest_to_disk() {
        let store = SessionStore::new("initial-model".to_string());
        let cwd =
            std::env::temp_dir().join(format!("brokk-acp-rust-create-{}", uuid::Uuid::new_v4()));
        let session = store.create_session(cwd.clone()).await;

        let zip_path = session_zip_path(&cwd, &session.id);
        assert!(zip_path.exists(), "session zip must be written to disk");

        let manifest = read_manifest_from_zip(&zip_path).expect("manifest must round-trip");
        assert_eq!(manifest.id, session.id);
        assert_eq!(manifest.mode.as_deref(), Some("LUTZ"));
        assert_eq!(manifest.model.as_deref(), Some("initial-model"));
        assert!(manifest.title().is_none(), "new sessions start untitled");
        assert!(
            manifest.updated_at().is_some(),
            "manifest.modified must be serializable as ACP updatedAt"
        );

        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// A placeholder-titled session should rename itself from the first
    /// prompt seed and persist the new title in the manifest zip.
    #[tokio::test]
    async fn maybe_rename_from_prompt_persists_title() {
        let store = SessionStore::new("initial-model".to_string());
        let cwd =
            std::env::temp_dir().join(format!("brokk-acp-rust-rename-{}", uuid::Uuid::new_v4()));
        let session = store.create_session(cwd.clone()).await;
        let id = session.id.clone();

        let title = store
            .maybe_rename_from_prompt(&id, "Investigate session names")
            .await
            .expect("title rename should persist");

        assert_eq!(title.as_deref(), Some("Investigate session names"));

        let in_memory = store.sessions.read().await;
        let session = in_memory.get(&id).expect("session still in memory");
        assert_eq!(session.manifest.name, "Investigate session names");
        assert_eq!(
            session.manifest.title().as_deref(),
            Some("Investigate session names")
        );
        assert!(session.manifest.updated_at().is_some());

        let manifest =
            read_manifest_from_zip(&session_zip_path(&cwd, &id)).expect("manifest must round-trip");
        assert_eq!(manifest.name, "Investigate session names");
        assert_eq!(
            manifest.title().as_deref(),
            Some("Investigate session names")
        );
        assert!(manifest.updated_at().is_some());

        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// A persisted install-level preference should seed the next new session
    /// with both the last model and the last supported reasoning effort.
    #[tokio::test(flavor = "current_thread")]
    async fn create_session_reuses_persisted_model_and_reasoning_effort() {
        use crate::llm_client::ReasoningLevelPreset;

        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = TestConfigHomeScope::set(config_dir.path().to_path_buf());
        crate::setup_state::remember_last_selection(
            Some("model-b".to_string()),
            Some("high".to_string()),
        )
        .expect("persist preferences");

        let store = SessionStore::new("model-a".to_string());
        store
            .set_available_models(vec![
                ModelMetadata {
                    id: "model-a".to_string(),
                    default_reasoning_level: Some("medium".to_string()),
                    supported_reasoning_levels: vec![ReasoningLevelPreset {
                        effort: "medium".to_string(),
                        description: "".to_string(),
                    }],
                    supports_images: None,
                    context_length: None,
                },
                ModelMetadata {
                    id: "model-b".to_string(),
                    default_reasoning_level: Some("medium".to_string()),
                    supported_reasoning_levels: vec![
                        ReasoningLevelPreset {
                            effort: "low".to_string(),
                            description: "".to_string(),
                        },
                        ReasoningLevelPreset {
                            effort: "medium".to_string(),
                            description: "".to_string(),
                        },
                        ReasoningLevelPreset {
                            effort: "high".to_string(),
                            description: "".to_string(),
                        },
                    ],
                    supports_images: None,
                    context_length: None,
                },
            ])
            .await;
        let cwd = tempfile::tempdir().expect("cwd");
        let session = store.create_session(cwd.path().to_path_buf()).await;

        assert_eq!(session.model, "model-b");
        assert_eq!(session.selected_reasoning_effort.as_deref(), Some("high"));
        let snap = store
            .snapshot(&session.id, cwd.path())
            .await
            .expect("session still loadable");
        assert_eq!(snap.reasoning_effort.as_deref(), Some("high"));
    }

    /// If the saved model disappeared from the catalog, new sessions should
    /// fall back to the current default/preferred model rather than failing.
    #[tokio::test(flavor = "current_thread")]
    async fn create_session_falls_back_to_default_model_when_persisted_model_is_missing() {
        use crate::llm_client::ReasoningLevelPreset;

        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = TestConfigHomeScope::set(config_dir.path().to_path_buf());
        crate::setup_state::remember_last_selection(
            Some("ghost-model".to_string()),
            Some("high".to_string()),
        )
        .expect("persist preferences");

        let store = SessionStore::new("model-a".to_string());
        store
            .set_available_models(vec![ModelMetadata {
                id: "model-a".to_string(),
                default_reasoning_level: Some("medium".to_string()),
                supported_reasoning_levels: vec![
                    ReasoningLevelPreset {
                        effort: "low".to_string(),
                        description: "".to_string(),
                    },
                    ReasoningLevelPreset {
                        effort: "medium".to_string(),
                        description: "".to_string(),
                    },
                    ReasoningLevelPreset {
                        effort: "high".to_string(),
                        description: "".to_string(),
                    },
                ],
                supports_images: None,
                context_length: None,
            }])
            .await;
        let cwd = tempfile::tempdir().expect("cwd");
        let session = store.create_session(cwd.path().to_path_buf()).await;

        assert_eq!(session.model, "model-a");
        assert_eq!(session.selected_reasoning_effort.as_deref(), Some("high"));
        let snap = store
            .snapshot(&session.id, cwd.path())
            .await
            .expect("session still loadable");
        assert_eq!(snap.reasoning_effort.as_deref(), Some("high"));
    }

    /// Unsupported saved reasoning should fall back to the chosen model's
    /// default reasoning level on the first prompt.
    #[tokio::test(flavor = "current_thread")]
    async fn create_session_falls_back_to_model_default_when_reasoning_is_unsupported() {
        use crate::llm_client::ReasoningLevelPreset;

        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = TestConfigHomeScope::set(config_dir.path().to_path_buf());
        crate::setup_state::remember_last_selection(
            Some("model-b".to_string()),
            Some("xhigh".to_string()),
        )
        .expect("persist preferences");

        let store = SessionStore::new("model-a".to_string());
        store
            .set_available_models(vec![
                ModelMetadata {
                    id: "model-a".to_string(),
                    default_reasoning_level: Some("medium".to_string()),
                    supported_reasoning_levels: vec![ReasoningLevelPreset {
                        effort: "medium".to_string(),
                        description: "".to_string(),
                    }],
                    supports_images: None,
                    context_length: None,
                },
                ModelMetadata {
                    id: "model-b".to_string(),
                    default_reasoning_level: Some("medium".to_string()),
                    supported_reasoning_levels: vec![
                        ReasoningLevelPreset {
                            effort: "low".to_string(),
                            description: "".to_string(),
                        },
                        ReasoningLevelPreset {
                            effort: "medium".to_string(),
                            description: "".to_string(),
                        },
                        ReasoningLevelPreset {
                            effort: "high".to_string(),
                            description: "".to_string(),
                        },
                    ],
                    supports_images: None,
                    context_length: None,
                },
            ])
            .await;
        let cwd = tempfile::tempdir().expect("cwd");
        let session = store.create_session(cwd.path().to_path_buf()).await;

        assert_eq!(session.model, "model-b");
        assert!(
            session.selected_reasoning_effort.is_none(),
            "unsupported reasoning should not be seeded explicitly"
        );
        let snap = store
            .snapshot(&session.id, cwd.path())
            .await
            .expect("session still loadable");
        assert_eq!(snap.reasoning_effort.as_deref(), Some("medium"));
    }

    /// A persisted install-level sandbox preference should seed the next
    /// new session instead of always falling back to the process default.
    #[tokio::test(flavor = "current_thread")]
    async fn create_session_reuses_persisted_sandbox_mode() {
        use crate::sandbox_backend::SandboxMode;

        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = TestConfigHomeScope::set(config_dir.path().to_path_buf());
        crate::setup_state::remember_sandbox_mode(Some(SandboxMode::Off))
            .expect("persist sandbox preference");

        let store = SessionStore::new("model-a".to_string());
        let cwd = tempfile::tempdir().expect("cwd");
        let session = store.create_session(cwd.path().to_path_buf()).await;

        assert_eq!(session.sandbox_mode, Some(SandboxMode::Off));
        assert_eq!(
            store.sandbox_mode(&session.id).await,
            Some(Some(SandboxMode::Off))
        );
    }

    /// `get_session` for an unknown id (with no on-disk zip either) must
    /// return None, not panic or allocate a session under the wrong id.
    #[tokio::test]
    async fn get_session_unknown_returns_none() {
        let store = SessionStore::new("m".to_string());
        let cwd = std::env::temp_dir().join(format!(
            "brokk-acp-rust-get-unknown-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&cwd).ok();
        assert!(store.get_session("does-not-exist", &cwd).await.is_none());
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// Cold path: a session that's been evicted from memory must reload
    /// from its on-disk zip on the next `get_session`. Mode and model
    /// survive the round-trip via the manifest.
    #[tokio::test]
    async fn get_session_loads_from_disk_when_cold() {
        let store = SessionStore::new("m".to_string());
        let cwd =
            std::env::temp_dir().join(format!("brokk-acp-rust-cold-{}", uuid::Uuid::new_v4()));
        let created = store.create_session(cwd.clone()).await;
        let id = created.id.clone();

        // Persist a non-default mode so we can verify it survives the reload.
        store
            .set_mode(&id, SessionMode::Plan)
            .await
            .expect("set_mode persists");

        // Evict from memory; the on-disk zip is unaffected.
        store.sessions.write().await.remove(&id);
        store.registries.write().await.remove(&id);

        let reloaded = store
            .get_session(&id, &cwd)
            .await
            .expect("session must reload from disk");
        assert_eq!(reloaded.id, id);
        assert_eq!(reloaded.mode, SessionMode::Plan);
        assert_eq!(reloaded.permission_mode, PermissionMode::Default);
        assert!(reloaded.sandbox_mode.is_none());
        assert!(reloaded.always_allow_tools.is_empty());
        assert!(reloaded.always_allow_order.is_empty());

        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// `sandbox_mode` round-trips through the setter, defaults to
    /// `None`, and reports `false` from the setter on an unknown session
    /// (matching the `set_permission_mode` contract).
    #[tokio::test]
    async fn sandbox_mode_setter_and_getter_round_trip() {
        use crate::sandbox_backend::SandboxMode;
        let store = SessionStore::new("m".to_string());
        let cwd =
            std::env::temp_dir().join(format!("brokk-acp-rust-sandbox-{}", uuid::Uuid::new_v4()));
        let session = store.create_session(cwd.clone()).await;
        let id = session.id.clone();

        assert_eq!(store.sandbox_mode(&id).await, Some(None));

        assert!(store.set_sandbox_mode(&id, Some(SandboxMode::Off)).await);
        assert_eq!(store.sandbox_mode(&id).await, Some(Some(SandboxMode::Off)));

        assert!(store.set_sandbox_mode(&id, Some(SandboxMode::Wasm)).await);
        assert_eq!(store.sandbox_mode(&id).await, Some(Some(SandboxMode::Wasm)));

        assert!(store.set_sandbox_mode(&id, None).await);
        assert_eq!(store.sandbox_mode(&id).await, Some(None));

        assert!(
            !store
                .set_sandbox_mode("no-such", Some(SandboxMode::Off))
                .await
        );
        assert_eq!(store.sandbox_mode("no-such").await, None);

        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// The `/setup sandbox` setter persists an install-level preference:
    /// future sessions and cold reloads should inherit it even though the
    /// session zip itself does not contain sandbox state.
    #[tokio::test(flavor = "current_thread")]
    async fn sandbox_mode_setter_seeds_future_sessions_and_cold_reloads() {
        use crate::sandbox_backend::SandboxMode;

        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = TestConfigHomeScope::set(config_dir.path().to_path_buf());
        let store = SessionStore::new("m".to_string());

        let cwd = tempfile::tempdir().expect("cwd");
        let first = store.create_session(cwd.path().to_path_buf()).await;
        let id = first.id.clone();
        assert_eq!(store.sandbox_mode(&id).await, Some(None));

        assert!(store.set_sandbox_mode(&id, Some(SandboxMode::Off)).await);
        assert_eq!(
            crate::setup_state::read().last_sandbox_mode,
            Some(SandboxMode::Off)
        );

        let next_cwd = tempfile::tempdir().expect("next cwd");
        let next = store.create_session(next_cwd.path().to_path_buf()).await;
        assert_eq!(next.sandbox_mode, Some(SandboxMode::Off));

        store.sessions.write().await.remove(&id);
        store.registries.write().await.remove(&id);

        let reloaded = store
            .get_session(&id, cwd.path())
            .await
            .expect("session must reload from disk");
        assert_eq!(reloaded.sandbox_mode, Some(SandboxMode::Off));
    }

    /// Transient setup mode keeps model/reasoning/sandbox choices process-local:
    /// they seed later sessions and cold reloads in this `SessionStore`, but the
    /// on-disk setup file is neither read nor written for those preferences.
    #[tokio::test(flavor = "current_thread")]
    async fn transient_setup_keeps_model_and_sandbox_preferences_in_memory_only() {
        use crate::llm_client::ReasoningLevelPreset;
        use crate::sandbox_backend::SandboxMode;

        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = TestConfigHomeScope::set(config_dir.path().to_path_buf());
        crate::setup_state::remember_last_selection(
            Some("persisted-model".to_string()),
            Some("high".to_string()),
        )
        .expect("seed persistent model preference");
        crate::setup_state::remember_sandbox_mode(Some(SandboxMode::Wasm))
            .expect("seed persistent sandbox preference");

        let store = SessionStore::with_limits_and_transient_setup(
            "default-model".to_string(),
            SessionLimits::default(),
            true,
        );
        store
            .set_available_models(vec![
                ModelMetadata {
                    id: "default-model".to_string(),
                    default_reasoning_level: Some("medium".to_string()),
                    supported_reasoning_levels: vec![
                        ReasoningLevelPreset {
                            effort: "medium".to_string(),
                            description: "".to_string(),
                        },
                        ReasoningLevelPreset {
                            effort: "high".to_string(),
                            description: "".to_string(),
                        },
                    ],
                    supports_images: None,
                    context_length: None,
                },
                ModelMetadata {
                    id: "runtime-model".to_string(),
                    default_reasoning_level: Some("medium".to_string()),
                    supported_reasoning_levels: vec![ReasoningLevelPreset {
                        effort: "high".to_string(),
                        description: "".to_string(),
                    }],
                    supports_images: None,
                    context_length: None,
                },
                ModelMetadata::id_only("persisted-model"),
            ])
            .await;

        let cwd = tempfile::tempdir().expect("cwd");
        let first = store.create_session(cwd.path().to_path_buf()).await;
        assert_eq!(
            first.model, "default-model",
            "transient stores must ignore persisted model preferences"
        );
        assert_eq!(
            first.sandbox_mode, None,
            "transient stores must ignore persisted sandbox preferences"
        );

        assert!(
            store
                .set_sandbox_mode(&first.id, Some(SandboxMode::Off))
                .await
        );
        let (ok, cleared) = store
            .set_model(&first.id, "runtime-model".to_string())
            .await;
        assert!(ok);
        assert!(cleared.is_none());
        assert!(
            store
                .set_reasoning_effort(&first.id, Some("high".to_string()))
                .await
        );

        let next_cwd = tempfile::tempdir().expect("next cwd");
        let next = store.create_session(next_cwd.path().to_path_buf()).await;
        assert_eq!(next.model, "runtime-model");
        assert_eq!(next.selected_reasoning_effort.as_deref(), Some("high"));
        assert_eq!(next.sandbox_mode, Some(SandboxMode::Off));

        assert_eq!(
            crate::setup_state::read().last_model.as_deref(),
            Some("persisted-model"),
            "transient choices must not overwrite setup.json model preference"
        );
        assert_eq!(
            crate::setup_state::read().last_sandbox_mode,
            Some(SandboxMode::Wasm),
            "transient choices must not overwrite setup.json sandbox preference"
        );

        store.sessions.write().await.remove(&first.id);
        store.registries.write().await.remove(&first.id);
        let reloaded = store
            .get_session(&first.id, cwd.path())
            .await
            .expect("session must reload from disk");
        assert_eq!(reloaded.sandbox_mode, Some(SandboxMode::Off));
    }

    /// Existing sessions should pick up sandbox changes written by another
    /// Anvil process on the next sandbox-mode read. That read happens on
    /// every tool execution path, so cross-process changes become effective
    /// without waiting for session reload.
    #[tokio::test(flavor = "current_thread")]
    async fn sandbox_mode_reads_external_setup_state_changes() {
        use crate::sandbox_backend::SandboxMode;

        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = TestConfigHomeScope::set(config_dir.path().to_path_buf());
        let store = SessionStore::new("m".to_string());

        let cwd = tempfile::tempdir().expect("cwd");
        let session = store.create_session(cwd.path().to_path_buf()).await;
        let id = session.id.clone();
        assert_eq!(store.sandbox_mode(&id).await, Some(None));

        crate::setup_state::remember_sandbox_mode(Some(SandboxMode::Off))
            .expect("external sandbox preference write");
        assert_eq!(store.sandbox_mode(&id).await, Some(Some(SandboxMode::Off)));

        crate::setup_state::remember_sandbox_mode(None).expect("external sandbox preference clear");
        assert_eq!(store.sandbox_mode(&id).await, Some(None));
    }

    /// Permission-mode setters/getters round-trip, default to `Default`,
    /// and report `false` for unknown sessions instead of silently
    /// inserting an entry.
    #[tokio::test]
    async fn permission_mode_setter_and_getter_round_trip() {
        let store = SessionStore::new("m".to_string());
        let cwd =
            std::env::temp_dir().join(format!("brokk-acp-rust-perm-{}", uuid::Uuid::new_v4()));
        let session = store.create_session(cwd.clone()).await;
        let id = session.id.clone();

        // Defaults out of the box.
        assert_eq!(
            store.permission_mode(&id).await,
            Some(PermissionMode::Default)
        );

        assert!(
            store
                .set_permission_mode(&id, PermissionMode::AcceptEdits)
                .await
        );
        assert_eq!(
            store.permission_mode(&id).await,
            Some(PermissionMode::AcceptEdits)
        );

        // Unknown session: false, not Ok with a phantom insert.
        assert!(
            !store
                .set_permission_mode("no-such", PermissionMode::ReadOnly)
                .await
        );
        assert_eq!(store.permission_mode("no-such").await, None);

        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// Remembered approvals persist per repo and survive new sessions in that repo.
    #[tokio::test]
    async fn always_allow_set_is_persisted_for_new_sessions() {
        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = crate::setup_state::TestConfigHomeScope::set(config_dir.path().to_path_buf());
        let store = SessionStore::new("m".to_string());
        let repo = tempfile::tempdir().expect("repo");
        let other_repo = tempfile::tempdir().expect("other repo");
        std::fs::create_dir_all(repo.path().join(".git")).expect("git dir");
        std::fs::create_dir_all(other_repo.path().join(".git")).expect("other git dir");
        let s = store.create_session(repo.path().to_path_buf()).await;

        assert!(
            !store
                .is_any_always_allowed(&s.id, &["write_file".to_string()])
                .await
        );
        store.add_always_allow(&s.id, "write_file").await;
        assert!(
            store
                .is_any_always_allowed(&s.id, &["write_file".to_string()])
                .await
        );
        // Different tool name, same session: still false.
        assert!(
            !store
                .is_any_always_allowed(&s.id, &["run_shell_command".to_string()])
                .await
        );
        // Unknown session never reports allowed.
        assert!(
            !store
                .is_any_always_allowed("no-such", &["write_file".to_string()])
                .await
        );

        let next = store.create_session(repo.path().to_path_buf()).await;
        assert!(
            store
                .is_any_always_allowed(&next.id, &["write_file".to_string()])
                .await
        );
        assert_eq!(
            store.always_allow_keys(&next.id).await,
            Some(vec!["write_file".to_string()])
        );

        let different_repo = store.create_session(other_repo.path().to_path_buf()).await;
        assert!(
            !store
                .is_any_always_allowed(&different_repo.id, &["write_file".to_string()])
                .await
        );
        assert_eq!(
            read_repo_permission_state(repo.path()).always_allow,
            vec!["write_file".to_string()]
        );
    }

    #[tokio::test]
    async fn repo_shell_permissions_are_persisted_per_repo() {
        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = crate::setup_state::TestConfigHomeScope::set(config_dir.path().to_path_buf());
        let store = SessionStore::new("m".to_string());
        let repo1 = tempfile::tempdir().expect("repo1");
        let repo2 = tempfile::tempdir().expect("repo2");
        std::fs::create_dir_all(repo1.path().join(".git")).expect("git dir 1");
        std::fs::create_dir_all(repo2.path().join(".git")).expect("git dir 2");

        let repo_key = serde_json::json!({
            "tool": "run_shell_command",
            "rule": "prefix",
            "argvPrefix": ["cargo", "test"],
            "shellSandboxed": true,
        })
        .to_string();

        let first = store.create_session(repo1.path().to_path_buf()).await;
        store.add_always_allow(&first.id, &repo_key).await;
        assert!(
            store
                .is_any_always_allowed(&first.id, std::slice::from_ref(&repo_key))
                .await
        );

        let second_same_repo = store.create_session(repo1.path().to_path_buf()).await;
        assert!(
            store
                .is_any_always_allowed(&second_same_repo.id, std::slice::from_ref(&repo_key))
                .await
        );

        let other_repo = store.create_session(repo2.path().to_path_buf()).await;
        assert!(
            !store
                .is_any_always_allowed(&other_repo.id, std::slice::from_ref(&repo_key))
                .await
        );

        assert_eq!(
            read_repo_permission_state(repo1.path()).always_allow,
            vec![repo_key]
        );
        assert!(
            read_repo_permission_state(repo2.path())
                .always_allow
                .is_empty()
        );
    }

    #[tokio::test]
    async fn always_allow_keys_can_be_listed_removed_and_cleared() {
        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = crate::setup_state::TestConfigHomeScope::set(config_dir.path().to_path_buf());
        let store = SessionStore::new("m".to_string());
        let cwd = tempfile::tempdir().expect("repo");
        std::fs::create_dir_all(cwd.path().join(".git")).expect("git dir");
        let s = store.create_session(cwd.path().to_path_buf()).await;
        let repo_key = serde_json::json!({
            "tool": "run_shell_command",
            "rule": "prefix",
            "argvPrefix": ["cargo", "test"],
            "shellSandboxed": true,
        })
        .to_string();

        store.add_always_allow(&s.id, "write_file").await;
        store.add_always_allow(&s.id, &repo_key).await;
        store.add_always_allow(&s.id, "write_file").await;

        assert_eq!(
            store.always_allow_keys(&s.id).await,
            Some(vec!["write_file".to_string(), repo_key.clone()])
        );
        assert_eq!(
            store.remove_always_allow(&s.id, "write_file").await,
            Some(true)
        );
        assert_eq!(
            store.remove_always_allow(&s.id, "write_file").await,
            Some(false)
        );
        assert_eq!(store.clear_always_allow(&s.id).await, Some(1));
        assert_eq!(store.always_allow_keys(&s.id).await, Some(Vec::new()));

        assert_eq!(store.always_allow_keys("no-such").await, None);
        assert_eq!(store.remove_always_allow("no-such", "x").await, None);
        assert_eq!(store.clear_always_allow("no-such").await, None);
    }

    /// `start_prompt` registers a token, `cancel_prompt` flips it,
    /// `finish_prompt` clears the registry. Together this is the gate
    /// `add_turn` relies on to serialize per-session writes.
    #[tokio::test]
    async fn cancel_prompt_lifecycle() {
        let store = SessionStore::new("m".to_string());
        let token = store
            .start_prompt("s1")
            .await
            .expect("first prompt should start");
        assert!(!token.is_cancelled());

        store.cancel_prompt("s1").await;
        assert!(token.is_cancelled(), "cancel_prompt should fire the token");

        store.finish_prompt("s1").await;
        // After finish, cancel becomes a no-op (no token registered).
        store.cancel_prompt("s1").await;
    }

    /// `cancel_prompt` for a session with no in-flight prompt must be a
    /// silent no-op rather than a panic.
    #[tokio::test]
    async fn cancel_prompt_unknown_session_is_noop() {
        let store = SessionStore::new("m".to_string());
        store.cancel_prompt("never-started").await;
    }

    /// `start_prompt` must reject a second in-flight prompt for the same
    /// session instead of overwriting the original cancellation token.
    #[tokio::test]
    async fn start_prompt_rejects_concurrent_prompt_for_same_session() {
        let store = SessionStore::new("m".to_string());
        let token = store
            .start_prompt("s1")
            .await
            .expect("first prompt should start");

        let err = store
            .start_prompt("s1")
            .await
            .expect_err("second prompt should be rejected");
        assert_eq!(err, PromptStartError::AlreadyInFlight);
        assert!(
            !token.is_cancelled(),
            "rejected concurrent prompt must not affect the active token"
        );

        store.cancel_prompt("s1").await;
        assert!(
            token.is_cancelled(),
            "cancel_prompt should still target the active prompt token"
        );

        store.finish_prompt("s1").await;
        let next = store
            .start_prompt("s1")
            .await
            .expect("prompt should be restartable after finish");
        assert!(!next.is_cancelled());
    }

    /// `set_default_model` flows into the next `create_session`. The
    /// previous default (set by the constructor) must NOT be sticky.
    #[tokio::test]
    async fn set_default_model_drives_subsequent_create_session() {
        let store = SessionStore::new("first".to_string());
        store.set_default_model("second".into()).await;
        let cwd =
            std::env::temp_dir().join(format!("brokk-acp-rust-default-{}", uuid::Uuid::new_v4()));
        let s = store.create_session(cwd.clone()).await;
        assert_eq!(s.model, "second");
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// A configured default reasoning effort should seed later
    /// `create_session` calls without polluting persisted setup state.
    #[tokio::test]
    async fn set_default_reasoning_effort_drives_subsequent_create_session() {
        use crate::llm_client::ReasoningLevelPreset;

        let store = SessionStore::new("gpt-mini".to_string());
        store
            .set_available_models(vec![ModelMetadata {
                id: "gpt-mini".to_string(),
                default_reasoning_level: Some("medium".to_string()),
                supported_reasoning_levels: vec![
                    ReasoningLevelPreset {
                        effort: "low".to_string(),
                        description: "".to_string(),
                    },
                    ReasoningLevelPreset {
                        effort: "medium".to_string(),
                        description: "".to_string(),
                    },
                    ReasoningLevelPreset {
                        effort: "high".to_string(),
                        description: "".to_string(),
                    },
                ],
                supports_images: None,
                context_length: None,
            }])
            .await;
        store
            .set_default_reasoning_effort(Some("high".to_string()))
            .await;
        let cwd = std::env::temp_dir().join(format!(
            "brokk-acp-rust-default-reasoning-{}",
            uuid::Uuid::new_v4()
        ));
        let s = store.create_session(cwd.clone()).await;
        assert_eq!(s.selected_reasoning_effort.as_deref(), Some("high"));
        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// `set_available_models` round-trips through `available_models` --
    /// this is the cache the agent populates on init and reuses for every
    /// `session/new` and `session/set_config_option` validation.
    #[tokio::test]
    async fn available_models_round_trip() {
        let store = SessionStore::new("m".to_string());
        assert!(store.available_models().await.is_empty());

        let models = vec![ModelMetadata::id_only("a"), ModelMetadata::id_only("b")];
        store.set_available_models(models).await;
        assert_eq!(
            store.available_models().await,
            vec!["a".to_string(), "b".to_string()]
        );

        // Setting an empty list is allowed (model discovery cleared).
        store.set_available_models(vec![]).await;
        assert!(store.available_models().await.is_empty());
    }

    /// When the user picks a reasoning effort that the new model
    /// doesn't advertise, `set_model` clears the pick and surfaces the
    /// cleared value so the caller can notify the user. Auto-fallback
    /// to the new model's default happens at the next `snapshot()`.
    #[tokio::test]
    async fn set_model_clears_reasoning_effort_when_unsupported() {
        use crate::llm_client::ReasoningLevelPreset;

        let store = SessionStore::new("gpt-big".to_string());
        store
            .set_available_models(vec![
                ModelMetadata {
                    id: "gpt-big".to_string(),
                    default_reasoning_level: Some("medium".to_string()),
                    supported_reasoning_levels: vec![
                        ReasoningLevelPreset {
                            effort: "low".to_string(),
                            description: "".to_string(),
                        },
                        ReasoningLevelPreset {
                            effort: "medium".to_string(),
                            description: "".to_string(),
                        },
                        ReasoningLevelPreset {
                            effort: "high".to_string(),
                            description: "".to_string(),
                        },
                        ReasoningLevelPreset {
                            effort: "xhigh".to_string(),
                            description: "".to_string(),
                        },
                    ],
                    supports_images: None,
                    context_length: None,
                },
                ModelMetadata {
                    id: "gpt-mini".to_string(),
                    default_reasoning_level: Some("medium".to_string()),
                    supported_reasoning_levels: vec![
                        ReasoningLevelPreset {
                            effort: "low".to_string(),
                            description: "".to_string(),
                        },
                        ReasoningLevelPreset {
                            effort: "medium".to_string(),
                            description: "".to_string(),
                        },
                        ReasoningLevelPreset {
                            effort: "high".to_string(),
                            description: "".to_string(),
                        },
                    ],
                    supports_images: None,
                    context_length: None,
                },
            ])
            .await;
        let cwd = std::env::temp_dir().join(format!(
            "brokk-acp-rust-set-model-clears-{}",
            uuid::Uuid::new_v4()
        ));
        let session = store.create_session(cwd.clone()).await;
        let id = session.id.clone();
        assert!(
            store
                .set_reasoning_effort(&id, Some("xhigh".to_string()))
                .await
        );

        // Switch to gpt-mini, which doesn't advertise xhigh.
        let (ok, cleared) = store.set_model(&id, "gpt-mini".to_string()).await;
        assert!(ok);
        assert_eq!(cleared.as_deref(), Some("xhigh"));

        // The snapshot now resolves to gpt-mini's default.
        let snap = store
            .snapshot(&id, &cwd)
            .await
            .expect("session still loadable");
        assert_eq!(snap.reasoning_effort.as_deref(), Some("medium"));

        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// Picking a level that *is* still supported by the new model
    /// must NOT be cleared -- this protects the user's intent across
    /// a slug bump within the same family (e.g. gpt-5.4 -> gpt-5.5).
    #[tokio::test]
    async fn set_model_preserves_reasoning_effort_when_supported() {
        use crate::llm_client::ReasoningLevelPreset;

        let supported = vec![
            ReasoningLevelPreset {
                effort: "low".to_string(),
                description: "".to_string(),
            },
            ReasoningLevelPreset {
                effort: "medium".to_string(),
                description: "".to_string(),
            },
            ReasoningLevelPreset {
                effort: "high".to_string(),
                description: "".to_string(),
            },
        ];
        let store = SessionStore::new("gpt-a".to_string());
        store
            .set_available_models(vec![
                ModelMetadata {
                    id: "gpt-a".to_string(),
                    default_reasoning_level: Some("medium".to_string()),
                    supported_reasoning_levels: supported.clone(),
                    supports_images: None,
                    context_length: None,
                },
                ModelMetadata {
                    id: "gpt-b".to_string(),
                    default_reasoning_level: Some("medium".to_string()),
                    supported_reasoning_levels: supported,
                    supports_images: None,
                    context_length: None,
                },
            ])
            .await;
        let cwd = std::env::temp_dir().join(format!(
            "brokk-acp-rust-set-model-preserves-{}",
            uuid::Uuid::new_v4()
        ));
        let session = store.create_session(cwd.clone()).await;
        let id = session.id.clone();
        assert!(
            store
                .set_reasoning_effort(&id, Some("high".to_string()))
                .await
        );

        let (ok, cleared) = store.set_model(&id, "gpt-b".to_string()).await;
        assert!(ok);
        assert!(cleared.is_none(), "high is still supported by gpt-b");
        let snap = store
            .snapshot(&id, &cwd)
            .await
            .expect("session still loadable");
        assert_eq!(snap.reasoning_effort.as_deref(), Some("high"));

        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// `list_sessions_from_disk` ignores non-zip files in the sessions
    /// directory (e.g. half-written `.tmp` files from a crashed write,
    /// or stray editor backups). Otherwise the list could surface entries
    /// that fail to load on resume.
    #[tokio::test]
    async fn list_sessions_from_disk_filters_non_zip() {
        let store = SessionStore::new("m".to_string());
        let cwd =
            std::env::temp_dir().join(format!("brokk-acp-rust-list-{}", uuid::Uuid::new_v4()));
        let s = store.create_session(cwd.clone()).await;

        // Drop a non-zip file alongside the real session zip.
        let dir = sessions_dir(&cwd);
        std::fs::write(dir.join("not-a-session.txt"), "hello").unwrap();
        // And a stray .tmp from a crashed write.
        std::fs::write(dir.join("garbage.tmp"), "junk").unwrap();

        let listed = store.list_sessions_from_disk(&cwd).await;
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, s.id);

        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// Concurrent `load_into_memory_if_cold` calls for the same id may
    /// both read the zip from disk; the `or_insert` race-keeper must keep
    /// exactly one entry in `sessions`. Regression coverage for the
    /// "race window" comment in `load_into_memory_if_cold`.
    #[tokio::test]
    async fn concurrent_cold_loads_dedupe() {
        let store = SessionStore::new("m".to_string());
        let cwd =
            std::env::temp_dir().join(format!("brokk-acp-rust-race-{}", uuid::Uuid::new_v4()));
        let created = store.create_session(cwd.clone()).await;
        let id = created.id.clone();

        // Evict the in-memory entry so both calls take the cold path.
        store.sessions.write().await.remove(&id);

        let store2 = store.clone();
        let id2 = id.clone();
        let cwd2 = cwd.clone();
        let h1 = tokio::spawn(async move { store2.get_session(&id2, &cwd2).await });
        let store3 = store.clone();
        let id3 = id.clone();
        let cwd3 = cwd.clone();
        let h2 = tokio::spawn(async move { store3.get_session(&id3, &cwd3).await });

        let r1 = h1.await.unwrap();
        let r2 = h2.await.unwrap();
        assert!(r1.is_some() && r2.is_some());

        let len = store.sessions.read().await.len();
        assert_eq!(len, 1, "both racers must collapse to a single entry");

        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// `update_cwd` rewrites the in-memory cwd; subsequent prompts use the
    /// new directory when computing zip paths. Persistence side is tested
    /// implicitly via `add_turn`/`set_mode`. Cached registries are left in
    /// place here and will be rebuilt lazily on the next prompt if their cwd
    /// no longer matches.
    #[tokio::test]
    async fn update_cwd_changes_in_memory_cwd() {
        let store = SessionStore::new("m".to_string());
        let cwd1 =
            std::env::temp_dir().join(format!("brokk-acp-rust-cwd1-{}", uuid::Uuid::new_v4()));
        let s = store.create_session(cwd1.clone()).await;

        let cwd2 = std::env::temp_dir().join("brokk-acp-rust-cwd2-replacement");
        store.update_cwd(&s.id, cwd2.clone()).await;
        let after = store.sessions.read().await.get(&s.id).cloned().unwrap();
        assert_eq!(after.cwd, cwd2);

        let _ = std::fs::remove_dir_all(&cwd1);
    }

    /// `update_cwd` must also re-discover subagents so the `task` tool's
    /// catalog reflects the new project. A session opened in a cwd with
    /// no `.claude/agents/` and then moved to a cwd that has one should
    /// pick up the new agents.
    #[tokio::test]
    async fn update_cwd_refreshes_agents() {
        let store = SessionStore::new("m".to_string());

        // First cwd: no agents.
        let cwd1 = tempfile::tempdir().expect("cwd1");
        std::fs::create_dir_all(cwd1.path().join(".git")).expect("touch git1");
        let s = store.create_session(cwd1.path().to_path_buf()).await;
        let agent_name = format!("hunter-{}", uuid::Uuid::new_v4());
        let before = store.sessions.read().await.get(&s.id).cloned().unwrap();
        assert!(
            before.agents.get(&agent_name).is_none(),
            "fresh session in empty cwd should not have the project test agent"
        );

        // Second cwd: one project-scope agent.
        let cwd2 = tempfile::tempdir().expect("cwd2");
        std::fs::create_dir_all(cwd2.path().join(".git")).expect("touch git2");
        let agents_dir = cwd2.path().join(".claude").join("agents");
        std::fs::create_dir_all(&agents_dir).expect("agents dir");
        std::fs::write(
            agents_dir.join(format!("{agent_name}.md")),
            format!("---\nname: {agent_name}\ndescription: Hunt bugs\n---\n\nbody\n"),
        )
        .expect("write agent");

        store.update_cwd(&s.id, cwd2.path().to_path_buf()).await;

        let after = store.sessions.read().await.get(&s.id).cloned().unwrap();
        assert!(
            after.agents.get(&agent_name).is_some(),
            "update_cwd should re-discover subagents in the new cwd"
        );
    }

    /// `get_or_create_registry` should reuse the cached registry when the
    /// cwd is only spelled differently (e.g. `path/.`). That keeps the
    /// existing Bifrost subprocess alive and only refreshes skills in place.
    #[tokio::test]
    async fn get_or_create_registry_reuses_cached_registry_for_equivalent_cwd_paths() {
        let store = SessionStore::new("m".to_string());
        let cwd = tempfile::tempdir().expect("cwd");
        let session = store.create_session(cwd.path().to_path_buf()).await;
        let cwd_alias = cwd.path().join(".");
        let canonical_cwd = normalize_cwd(cwd.path());

        let registry1 = store
            .get_or_create_registry(&session.id, cwd.path().to_path_buf())
            .await;
        let registry2 = store.get_or_create_registry(&session.id, cwd_alias).await;

        assert!(
            Arc::ptr_eq(&registry1, &registry2),
            "equivalent cwd spellings should reuse the cached registry"
        );
        assert_eq!(registry1.cwd(), canonical_cwd.as_path());
    }

    #[cfg(unix)]
    fn make_fake_bifrost_binary(script_dir: &Path, log_path: &Path) -> PathBuf {
        let script_path = script_dir.join("fake-bifrost.sh");
        let script = format!(
            r#"#!/bin/sh
log_path="{}"
printf '%s\n' "$@" >> "$log_path"
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
            log_path.display()
        );
        std::fs::write(&script_path, script).expect("write fake bifrost binary");
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&script_path)
            .expect("stat fake bifrost binary")
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script_path, perms).expect("chmod fake bifrost binary");
        script_path
    }

    #[cfg(unix)]
    fn bifrost_spawn_args(cwd: &Path) -> Vec<String> {
        vec![
            "--root".to_string(),
            cwd.display().to_string(),
            "--server".to_string(),
            "core".to_string(),
        ]
    }

    #[cfg(unix)]
    fn read_log_lines(path: &Path) -> Vec<String> {
        std::fs::read_to_string(path)
            .expect("read fake bifrost log")
            .lines()
            .map(|line| line.to_string())
            .collect()
    }

    #[test]
    fn acp_stdio_mcp_servers_convert_to_anvil_configs() {
        let configs =
            acp_mcp_servers_to_configs(vec![agent_client_protocol::schema::McpServer::Stdio(
                agent_client_protocol::schema::McpServerStdio::new("local", "/usr/bin/mcp")
                    .args(vec!["--flag".to_string(), "value".to_string()])
                    .env(vec![agent_client_protocol::schema::EnvVariable::new(
                        "TOKEN", "secret",
                    )]),
            )]);

        assert_eq!(
            configs,
            vec![crate::mcp::McpServerConfig {
                name: "local".to_string(),
                command: "/usr/bin/mcp".to_string(),
                args: vec!["--flag".to_string(), "value".to_string()],
                env: vec![crate::mcp::McpEnvVar {
                    name: "TOKEN".to_string(),
                    value: "secret".to_string(),
                }],
                framing: crate::mcp::McpFraming::ContentLength,
                enabled: true,
            }]
        );
    }

    /// An explicit ACP `mcpServers: []` means "no extra MCP servers";
    /// canonical Bifrost should still spawn from setup/default config.
    #[cfg(unix)]
    #[tokio::test]
    async fn explicit_empty_acp_mcp_servers_still_spawn_default_bifrost() {
        let store = SessionStore::new("m".to_string());
        let cwd = tempfile::tempdir().expect("cwd");
        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = crate::setup_state::TestConfigHomeScope::set(config_dir.path().to_path_buf());
        let fake_bifrost_dir = tempfile::tempdir().expect("fake bifrost dir");
        let bifrost_log = fake_bifrost_dir.path().join("bifrost-argv.log");
        let fake_bifrost = make_fake_bifrost_binary(fake_bifrost_dir.path(), &bifrost_log);
        crate::setup_state::remember_mcp_servers(vec![crate::mcp::McpServerConfig {
            name: "bifrost".to_string(),
            command: fake_bifrost.display().to_string(),
            args: crate::mcp::McpServerConfig::bifrost().args,
            env: Vec::new(),
            framing: crate::mcp::McpFraming::Line,
            enabled: true,
        }])
        .expect("remember mcp config");
        let session = store
            .create_session_with_mcp_servers(cwd.path().to_path_buf(), Some(Vec::new()))
            .await;

        let registry = store
            .get_or_create_registry(&session.id, cwd.path().to_path_buf())
            .await;

        assert_eq!(registry.cwd(), normalize_cwd(cwd.path()).as_path());
        assert!(
            bifrost_log.exists(),
            "explicit empty ACP MCP list should still spawn persisted default bifrost"
        );
        assert_eq!(read_log_lines(&bifrost_log), bifrost_spawn_args(cwd.path()));
    }

    /// The ACP extra MCP list is persisted in the session manifest, but
    /// canonical Bifrost is still reconstructed from setup/default config
    /// after reload.
    #[cfg(unix)]
    #[tokio::test]
    async fn explicit_empty_acp_mcp_servers_survive_cold_reload_with_default_bifrost() {
        let store = SessionStore::new("m".to_string());
        let cwd = tempfile::tempdir().expect("cwd");
        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = crate::setup_state::TestConfigHomeScope::set(config_dir.path().to_path_buf());
        let fake_bifrost_dir = tempfile::tempdir().expect("fake bifrost dir");
        let bifrost_log = fake_bifrost_dir.path().join("bifrost-argv.log");
        let fake_bifrost = make_fake_bifrost_binary(fake_bifrost_dir.path(), &bifrost_log);
        crate::setup_state::remember_mcp_servers(vec![crate::mcp::McpServerConfig {
            name: "bifrost".to_string(),
            command: fake_bifrost.display().to_string(),
            args: crate::mcp::McpServerConfig::bifrost().args,
            env: Vec::new(),
            framing: crate::mcp::McpFraming::Line,
            enabled: true,
        }])
        .expect("remember mcp config");
        let session = store
            .create_session_with_mcp_servers(cwd.path().to_path_buf(), Some(Vec::new()))
            .await;
        store.sessions.write().await.remove(&session.id);
        store.registries.write().await.remove(&session.id);

        let reloaded = store
            .get_session(&session.id, cwd.path())
            .await
            .expect("session should cold-load from zip");
        assert_eq!(reloaded.mcp_servers, Some(Vec::new()));

        let registry = store
            .get_or_create_registry(&session.id, cwd.path().to_path_buf())
            .await;

        assert_eq!(registry.cwd(), normalize_cwd(cwd.path()).as_path());
        assert!(
            bifrost_log.exists(),
            "cold-loaded ACP empty MCP list should still spawn persisted default bifrost"
        );
        assert_eq!(read_log_lines(&bifrost_log), bifrost_spawn_args(cwd.path()));
    }

    /// ACP-provided MCP servers are additive to canonical Bifrost.
    #[cfg(unix)]
    #[tokio::test]
    async fn acp_extra_mcp_servers_spawn_alongside_default_bifrost() {
        let store = SessionStore::new("m".to_string());
        let cwd = tempfile::tempdir().expect("cwd");
        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = crate::setup_state::TestConfigHomeScope::set(config_dir.path().to_path_buf());
        let fake_bifrost_dir = tempfile::tempdir().expect("fake bifrost dir");
        let bifrost_log = fake_bifrost_dir.path().join("bifrost-argv.log");
        let fake_bifrost = make_fake_bifrost_binary(fake_bifrost_dir.path(), &bifrost_log);
        crate::setup_state::remember_mcp_servers(vec![crate::mcp::McpServerConfig {
            name: "bifrost".to_string(),
            command: fake_bifrost.display().to_string(),
            args: crate::mcp::McpServerConfig::bifrost().args,
            env: Vec::new(),
            framing: crate::mcp::McpFraming::Line,
            enabled: true,
        }])
        .expect("remember mcp config");
        let extra_mcp_dir = tempfile::tempdir().expect("extra mcp dir");
        let extra_log = extra_mcp_dir.path().join("extra-mcp-argv.log");
        let extra_mcp = make_fake_bifrost_binary(extra_mcp_dir.path(), &extra_log);
        let session = store
            .create_session_with_mcp_servers(
                cwd.path().to_path_buf(),
                Some(vec![crate::mcp::McpServerConfig {
                    name: "local".to_string(),
                    command: extra_mcp.display().to_string(),
                    args: Vec::new(),
                    env: Vec::new(),
                    framing: crate::mcp::McpFraming::Line,
                    enabled: true,
                }]),
            )
            .await;

        let registry = store
            .get_or_create_registry(&session.id, cwd.path().to_path_buf())
            .await;

        assert_eq!(registry.cwd(), normalize_cwd(cwd.path()).as_path());
        assert_eq!(read_log_lines(&bifrost_log), bifrost_spawn_args(cwd.path()));
        assert!(extra_log.exists(), "extra ACP MCP server should also spawn");
        assert!(
            !read_log_lines(&extra_log).is_empty(),
            "extra ACP MCP server should receive its startup argv"
        );
    }

    /// ACP must not be able to add a duplicate Bifrost server on top of the
    /// canonical setup/default instance.
    #[cfg(unix)]
    #[tokio::test]
    async fn additive_acp_bifrost_server_is_ignored() {
        let store = SessionStore::new("m".to_string());
        let cwd = tempfile::tempdir().expect("cwd");
        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = crate::setup_state::TestConfigHomeScope::set(config_dir.path().to_path_buf());
        let fake_bifrost_dir = tempfile::tempdir().expect("fake bifrost dir");
        let bifrost_log = fake_bifrost_dir.path().join("bifrost-argv.log");
        let fake_bifrost = make_fake_bifrost_binary(fake_bifrost_dir.path(), &bifrost_log);
        crate::setup_state::remember_mcp_servers(vec![crate::mcp::McpServerConfig {
            name: "bifrost".to_string(),
            command: fake_bifrost.display().to_string(),
            args: crate::mcp::McpServerConfig::bifrost().args,
            env: Vec::new(),
            framing: crate::mcp::McpFraming::Line,
            enabled: true,
        }])
        .expect("remember mcp config");
        let extra_bifrost_dir = tempfile::tempdir().expect("extra bifrost dir");
        let extra_log = extra_bifrost_dir.path().join("extra-bifrost-argv.log");
        let extra_bifrost = make_fake_bifrost_binary(extra_bifrost_dir.path(), &extra_log);
        let session = store
            .create_session_with_mcp_servers(
                cwd.path().to_path_buf(),
                Some(vec![crate::mcp::McpServerConfig {
                    name: "bifrost".to_string(),
                    command: extra_bifrost.display().to_string(),
                    args: vec![
                        "--root".to_string(),
                        "{cwd}".to_string(),
                        "--server".to_string(),
                        "symbol".to_string(),
                    ],
                    env: Vec::new(),
                    framing: crate::mcp::McpFraming::Line,
                    enabled: true,
                }]),
            )
            .await;

        let registry = store
            .get_or_create_registry(&session.id, cwd.path().to_path_buf())
            .await;

        assert_eq!(registry.cwd(), normalize_cwd(cwd.path()).as_path());
        assert_eq!(read_log_lines(&bifrost_log), bifrost_spawn_args(cwd.path()));
        assert!(
            !extra_log.exists(),
            "duplicate ACP bifrost server should be ignored"
        );
    }

    /// A cwd change must invalidate the cached registry so the next
    /// prompt runs against the new workspace and respawns Bifrost with the
    /// updated root.
    #[cfg(unix)]
    #[tokio::test]
    async fn get_or_create_registry_rebuilds_and_respawns_bifrost_when_cwd_changes() {
        let store = SessionStore::new("m".to_string());
        let cwd1 = tempfile::tempdir().expect("cwd1");
        let cwd2 = tempfile::tempdir().expect("cwd2");
        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = crate::setup_state::TestConfigHomeScope::set(config_dir.path().to_path_buf());
        let fake_bifrost_dir = tempfile::tempdir().expect("fake bifrost dir");
        let bifrost_log = fake_bifrost_dir.path().join("bifrost-argv.log");
        let fake_bifrost = make_fake_bifrost_binary(fake_bifrost_dir.path(), &bifrost_log);
        crate::setup_state::remember_mcp_servers(vec![crate::mcp::McpServerConfig {
            name: "bifrost".to_string(),
            command: fake_bifrost.display().to_string(),
            args: crate::mcp::McpServerConfig::bifrost().args,
            env: Vec::new(),
            framing: crate::mcp::McpFraming::Line,
            enabled: true,
        }])
        .expect("remember mcp config");
        let session = store.create_session(cwd1.path().to_path_buf()).await;
        let canonical_cwd1 = normalize_cwd(cwd1.path());
        let canonical_cwd2 = normalize_cwd(cwd2.path());

        let registry1 = store
            .get_or_create_registry(&session.id, cwd1.path().to_path_buf())
            .await;
        assert_eq!(registry1.cwd(), canonical_cwd1.as_path());
        assert_eq!(
            read_log_lines(&bifrost_log),
            bifrost_spawn_args(&canonical_cwd1)
        );

        store
            .update_cwd(&session.id, cwd2.path().to_path_buf())
            .await;

        let registry2 = store
            .get_or_create_registry(&session.id, cwd2.path().to_path_buf())
            .await;

        assert!(
            !Arc::ptr_eq(&registry1, &registry2),
            "changed cwd should rebuild the registry"
        );
        assert_eq!(registry2.cwd(), canonical_cwd2.as_path());
        assert_eq!(
            read_log_lines(&bifrost_log),
            [
                bifrost_spawn_args(&canonical_cwd1),
                bifrost_spawn_args(&canonical_cwd2)
            ]
            .concat()
        );
    }

    /// Round-trip a full conversation history through the zip: write four
    /// turns with `add_turn`, then re-read with `read_history_from_zip`
    /// and verify both the user prompt and the agent response survive.
    #[tokio::test]
    async fn add_turn_round_trips_history_through_zip() {
        let store = SessionStore::with_limits(
            "m".to_string(),
            SessionLimits {
                max_sessions: 0,
                max_history_turns: 0,
            },
        );
        let cwd =
            std::env::temp_dir().join(format!("brokk-acp-rust-history-{}", uuid::Uuid::new_v4()));
        let s = store.create_session(cwd.clone()).await;

        for i in 0..4 {
            store
                .add_turn(
                    &s.id,
                    ConversationTurn {
                        user_prompt: format!("user-{i}"),
                        agent_response: format!("agent-{i}"),
                        ..Default::default()
                    },
                )
                .await
                .expect("persist must succeed");
        }

        let on_disk = read_history_from_zip(&session_zip_path(&cwd, &s.id));
        assert_eq!(on_disk.len(), 4);
        // Persisted format uses `markdownContentId` so the user prompt is
        // re-derived from `taskDescription` (null) -- only the agent
        // response is round-tripped on this path. Order is recovered from
        // contexts.jsonl, so the comparison is positional.
        let actual: Vec<String> = on_disk.iter().map(|t| t.agent_response.clone()).collect();
        let expected: Vec<String> = (0..4).map(|i| format!("agent-{i}")).collect();
        assert_eq!(actual, expected);

        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// Multi-turn replay must reconstruct turns in the order they were
    /// appended -- not in lexicographic UUID order, which is what falling
    /// back to `serde_json::Map` (BTreeMap) iteration would produce.
    ///
    /// Regression coverage for the failure mode the PR review caught
    /// (#3409 review HIGH): without ordering recovery from `contexts.jsonl`,
    /// `session/load` paired turn 0's user prompt with turn 2's
    /// tool_exchanges and turn 1's text on replay -- silently misleading
    /// the LLM rather than helping it. We use 6 turns so the probability
    /// of the BTreeMap ordering accidentally matching insertion order is
    /// negligible (≈ 1/6! ≈ 0.14%).
    #[tokio::test]
    async fn read_history_from_zip_preserves_chronological_turn_order() {
        let store = SessionStore::with_limits(
            "m".to_string(),
            SessionLimits {
                max_sessions: 0,
                max_history_turns: 0,
            },
        );
        let cwd = std::env::temp_dir().join(format!(
            "brokk-acp-rust-history-order-{}",
            uuid::Uuid::new_v4()
        ));
        let s = store.create_session(cwd.clone()).await;

        for i in 0..6 {
            store
                .add_turn(
                    &s.id,
                    ConversationTurn {
                        user_prompt: format!("u-{i}"),
                        agent_response: format!("a-{i}"),
                        tool_exchanges: vec![ToolExchange {
                            call_id: format!("call-{i}"),
                            tool_name: "noop".into(),
                            arguments: format!(r#"{{"i":{i}}}"#),
                            result: format!("r-{i}"),
                        }],
                        structured_output: None,
                        summary: None,
                        fragment_id: None,
                    },
                )
                .await
                .expect("persist must succeed");
        }

        let on_disk = read_history_from_zip(&session_zip_path(&cwd, &s.id));
        assert_eq!(on_disk.len(), 6);

        // Each turn's tool_exchange's call_id encodes its sequence index.
        // If ordering were lex-by-UUID, this would almost certainly fail.
        let order: Vec<String> = on_disk
            .iter()
            .map(|t| t.tool_exchanges[0].call_id.clone())
            .collect();
        let expected: Vec<String> = (0..6).map(|i| format!("call-{i}")).collect();
        assert_eq!(order, expected, "turns must replay in append order");

        // And the agent_response side too, for clarity on failure.
        let agent_order: Vec<String> = on_disk.iter().map(|t| t.agent_response.clone()).collect();
        let expected_agents: Vec<String> = (0..6).map(|i| format!("a-{i}")).collect();
        assert_eq!(agent_order, expected_agents);

        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// Tool calls and results must round-trip through the session zip
    /// (#3409). Without this, `session/load` fed the LLM only the final
    /// agent_response and the model would repeat searches/reads it had
    /// already performed in the prior turn.
    #[tokio::test]
    async fn add_turn_round_trips_tool_exchanges_through_zip() {
        let store = SessionStore::with_limits(
            "m".to_string(),
            SessionLimits {
                max_sessions: 0,
                max_history_turns: 0,
            },
        );
        let cwd = std::env::temp_dir().join(format!(
            "brokk-acp-rust-tool-exchanges-{}",
            uuid::Uuid::new_v4()
        ));
        let s = store.create_session(cwd.clone()).await;

        let exchanges = vec![
            ToolExchange {
                call_id: "call_abc".into(),
                tool_name: "read_file".into(),
                arguments: r#"{"file_path":"src/lib.rs"}"#.into(),
                result: "fn main() {}\n".into(),
            },
            ToolExchange {
                call_id: "call_xyz".into(),
                tool_name: "grep_search".into(),
                arguments: r#"{"pattern":"TODO"}"#.into(),
                result: "no matches".into(),
            },
        ];

        store
            .add_turn(
                &s.id,
                ConversationTurn {
                    user_prompt: "explore the repo".into(),
                    agent_response: "found one file".into(),
                    tool_exchanges: exchanges.clone(),
                    structured_output: None,
                    summary: None,
                    fragment_id: None,
                },
            )
            .await
            .expect("persist must succeed");

        let on_disk = read_history_from_zip(&session_zip_path(&cwd, &s.id));
        assert_eq!(on_disk.len(), 1);
        let turn = &on_disk[0];
        assert_eq!(turn.agent_response, "found one file");
        assert_eq!(turn.tool_exchanges.len(), 2);

        // Pair-up by call_id since the messages JSON ordering inside the
        // task fragment is preserved but we don't want the test to depend
        // on iteration order if that ever changes.
        let by_id: std::collections::HashMap<&str, &ToolExchange> = turn
            .tool_exchanges
            .iter()
            .map(|e| (e.call_id.as_str(), e))
            .collect();

        let abc = by_id.get("call_abc").expect("read_file exchange present");
        assert_eq!(abc.tool_name, "read_file");
        assert_eq!(abc.arguments, r#"{"file_path":"src/lib.rs"}"#);
        assert_eq!(abc.result, "fn main() {}\n");

        let xyz = by_id.get("call_xyz").expect("search exchange present");
        assert_eq!(xyz.tool_name, "grep_search");
        assert_eq!(xyz.arguments, r#"{"pattern":"TODO"}"#);
        assert_eq!(xyz.result, "no matches");

        let _ = std::fs::remove_dir_all(&cwd);
    }

    #[tokio::test]
    async fn add_turn_round_trips_structured_output_through_zip() {
        let store = SessionStore::with_limits(
            "m".to_string(),
            SessionLimits {
                max_sessions: 0,
                max_history_turns: 0,
            },
        );
        let cwd = std::env::temp_dir().join(format!(
            "brokk-acp-rust-structured-output-{}",
            uuid::Uuid::new_v4()
        ));
        let s = store.create_session(cwd.clone()).await;

        store
            .add_turn(
                &s.id,
                ConversationTurn {
                    user_prompt: "emit json".into(),
                    agent_response: r#"{"answer":"ok"}"#.into(),
                    tool_exchanges: Vec::new(),
                    structured_output: Some(StructuredOutputResult::CoercedSuccess(
                        crate::structured_output::StructuredOutputCoercedSuccess {
                            schema_name: "audit_result".into(),
                            validated_output: serde_json::json!({"answer":"one\ntwo"}),
                            coercions: vec!["response.answer array -> string".into()],
                            coercion_requested: true,
                        },
                    )),
                    summary: None,
                    fragment_id: None,
                },
            )
            .await
            .expect("persist must succeed");

        let on_disk = read_history_from_zip(&session_zip_path(&cwd, &s.id));
        assert_eq!(on_disk.len(), 1);
        match on_disk[0]
            .structured_output
            .as_ref()
            .expect("structured output present")
        {
            StructuredOutputResult::CoercedSuccess(success) => {
                assert_eq!(success.schema_name, "audit_result");
                assert_eq!(success.validated_output["answer"], "one\ntwo");
                assert_eq!(success.coercions, vec!["response.answer array -> string"]);
                assert!(success.coercion_requested);
            }
            other => panic!("expected coerced success, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// Reading an old-format zip (no tool_call / tool_result messages)
    /// must yield turns with empty `tool_exchanges` -- not panic, not
    /// drop the turn entirely. Backward-compat guarantee for sessions
    /// written before #3409 landed.
    #[tokio::test]
    async fn read_history_returns_empty_exchanges_for_legacy_zips() {
        let store = SessionStore::with_limits(
            "m".to_string(),
            SessionLimits {
                max_sessions: 0,
                max_history_turns: 0,
            },
        );
        let cwd = std::env::temp_dir().join(format!(
            "brokk-acp-rust-legacy-zip-{}",
            uuid::Uuid::new_v4()
        ));
        let s = store.create_session(cwd.clone()).await;

        // No tool exchanges -- mimics every turn ever written before
        // #3409: messages array contains only user + ai entries.
        store
            .add_turn(
                &s.id,
                ConversationTurn {
                    user_prompt: "hi".into(),
                    agent_response: "hello".into(),
                    ..Default::default()
                },
            )
            .await
            .expect("persist must succeed");

        let on_disk = read_history_from_zip(&session_zip_path(&cwd, &s.id));
        assert_eq!(on_disk.len(), 1);
        assert!(on_disk[0].tool_exchanges.is_empty());

        let _ = std::fs::remove_dir_all(&cwd);
    }

    // -----------------------------------------------------------------------
    // Per-turn summary persistence (mirrors Brokk's TaskEntry.summary)
    // -----------------------------------------------------------------------

    /// `set_turn_summary` updates a single turn's `summary` in memory
    /// and persists it via the existing `summaryContentId` slot. A
    /// reload must recover the same summary -- without this, the
    /// next prompt build would re-send the verbatim turn, defeating
    /// the compression.
    #[tokio::test]
    async fn set_turn_summary_round_trips_through_zip() {
        let store = SessionStore::with_limits(
            "m".to_string(),
            SessionLimits {
                max_sessions: 0,
                max_history_turns: 0,
            },
        );
        let cwd = std::env::temp_dir().join(format!(
            "brokk-acp-rust-turn-summary-{}",
            uuid::Uuid::new_v4()
        ));
        let s = store.create_session(cwd.clone()).await;
        store
            .add_turn(
                &s.id,
                ConversationTurn {
                    user_prompt: "user 0".into(),
                    agent_response: "agent 0".into(),
                    ..Default::default()
                },
            )
            .await
            .expect("turn persists");
        store
            .add_turn(
                &s.id,
                ConversationTurn {
                    user_prompt: "user 1".into(),
                    agent_response: "agent 1".into(),
                    ..Default::default()
                },
            )
            .await
            .expect("turn persists");

        // Compress turn 0.
        store
            .set_turn_summary(&s.id, 0, "- key bullet for turn 0".into())
            .await
            .expect("persist succeeds");

        // Drop the in-memory copy and reload from disk.
        store.sessions.write().await.remove(&s.id);
        let snap = store.snapshot(&s.id, &cwd).await.expect("reload succeeds");
        assert_eq!(snap.history.len(), 2);
        assert_eq!(
            snap.history[0].summary.as_deref(),
            Some("- key bullet for turn 0"),
            "turn 0's summary must round-trip"
        );
        assert!(
            snap.history[1].summary.is_none(),
            "turn 1 was never compressed"
        );

        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// `set_turn_summary` must reject an out-of-range turn index and
    /// leave the session untouched, mirroring the guard `set_mode`
    /// uses for unknown sessions.
    #[tokio::test]
    async fn set_turn_summary_rejects_out_of_range_turn_index() {
        let store = SessionStore::with_limits(
            "m".to_string(),
            SessionLimits {
                max_sessions: 0,
                max_history_turns: 0,
            },
        );
        let cwd = std::env::temp_dir().join(format!(
            "brokk-acp-rust-turn-summary-guard-{}",
            uuid::Uuid::new_v4()
        ));
        let s = store.create_session(cwd.clone()).await;
        store
            .add_turn(
                &s.id,
                ConversationTurn {
                    user_prompt: "u".into(),
                    agent_response: "a".into(),
                    ..Default::default()
                },
            )
            .await
            .expect("turn persists");

        let ok = store
            .set_turn_summary(&s.id, 99, "- nope".into())
            .await
            .expect("no error");
        assert!(!ok, "out-of-range index must return false");

        let _ = std::fs::remove_dir_all(&cwd);
    }

    /// A persisted summary must survive subsequent `add_turn` calls
    /// (which rewrite the zip's manifest, fragments, and contexts).
    /// Without this guarantee a single post-compression turn would
    /// silently erase the summary and the next reload would build a
    /// different prompt.
    #[tokio::test]
    async fn add_turn_preserves_existing_turn_summary() {
        let store = SessionStore::with_limits(
            "m".to_string(),
            SessionLimits {
                max_sessions: 0,
                max_history_turns: 0,
            },
        );
        let cwd = std::env::temp_dir().join(format!(
            "brokk-acp-rust-summary-survives-add-{}",
            uuid::Uuid::new_v4()
        ));
        let s = store.create_session(cwd.clone()).await;
        store
            .add_turn(
                &s.id,
                ConversationTurn {
                    user_prompt: "u-old".into(),
                    agent_response: "a-old".into(),
                    ..Default::default()
                },
            )
            .await
            .expect("turn persists");
        store
            .set_turn_summary(&s.id, 0, "- preserved bullet".into())
            .await
            .expect("summary persists");

        // Append another turn. The first turn's summary must survive.
        store
            .add_turn(
                &s.id,
                ConversationTurn {
                    user_prompt: "u-new".into(),
                    agent_response: "a-new".into(),
                    ..Default::default()
                },
            )
            .await
            .expect("turn appends");

        store.sessions.write().await.remove(&s.id);
        let snap = store.snapshot(&s.id, &cwd).await.expect("reload succeeds");
        assert_eq!(snap.history.len(), 2);
        assert_eq!(
            snap.history[0].summary.as_deref(),
            Some("- preserved bullet")
        );
        assert!(snap.history[1].summary.is_none());

        let _ = std::fs::remove_dir_all(&cwd);
    }
}
