mod announce;

use std::env;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use agent_client_protocol::schema::{
    Diff, PermissionOption, PermissionOptionId, PermissionOptionKind, RequestPermissionOutcome,
    RequestPermissionRequest, SessionNotification, SessionUpdate, ToolCallId, ToolCallStatus,
    ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use agent_client_protocol::{Client, ConnectionTo};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::llm_client::{
    ChatMessage, LlmBackend, LlmResponse, StreamChatRequest, TokenUsage, ToolCall, ToolDefinition,
    messages_include_images, rewrite_image_prompt_provider_error,
};
use crate::session::{PermissionMode, SessionStore, ToolExchange};
use crate::structured_output::StructuredOutputRequest;
use crate::terminal_notifications::{
    TerminalNotificationEvent, emit as emit_terminal_notification,
};
use crate::tools::sandbox::SandboxPolicy;
use crate::tools::{ToolRegistry, ToolStatus, safe_resolve_for_write};
use crate::trace_logging::append_trace_record;
use crate::train_bifrost::{self, TrainingPacket};

const MAX_TOOL_RESULT_BYTES: usize = 50_000;
const TRAIN_BIFROST_ENV: &str = "BRK_TRAIN_BIFROST";

fn train_bifrost_enabled() -> bool {
    env::var(TRAIN_BIFROST_ENV).ok().is_some_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn train_bifrost_initial_builtin_tools() -> std::collections::HashSet<String> {
    ["think", "write_file", "edit", "list_directory"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

fn train_bifrost_post_edit_builtin_tools() -> std::collections::HashSet<String> {
    let mut tools = train_bifrost_initial_builtin_tools();
    tools.insert("run_shell_command".to_string());
    tools
}

fn advertised_tool_names(tools: Option<&Vec<ToolDefinition>>) -> std::collections::HashSet<String> {
    tools
        .into_iter()
        .flat_map(|defs| defs.iter().map(|tool| tool.function.name.clone()))
        .collect()
}

fn tool_unavailable_message(tool_name: &str) -> String {
    format!(
        "Error: tool '{tool_name}' is unavailable in the current tool catalog. Retry using a currently advertised tool."
    )
}

fn is_retryable_llm_error(error: &anyhow::Error) -> bool {
    let error = format!("{error:#}");
    error.contains("stream read error")
        || error.contains("no meaningful progress")
        || error.contains("Responses stream failed: server_error")
        || error.contains("Responses stream failed: server_is_overloaded")
        || error.contains("Responses stream failed: rate_limit_exceeded")
        || error.contains("server_is_overloaded")
        || error.contains("server_error")
        || error.contains("rate_limit_exceeded")
}

#[allow(clippy::too_many_arguments)]
async fn stream_chat_with_transient_retry(
    llm: &Arc<dyn LlmBackend>,
    turn: usize,
    model: &str,
    messages: &[ChatMessage],
    tools: Option<Vec<ToolDefinition>>,
    reasoning_effort: Option<&str>,
    structured_output: Option<&StructuredOutputRequest>,
    on_text: &TextSink,
    on_thought: &TextSink,
    cancel: &CancellationToken,
    idle_timeout: Duration,
) -> anyhow::Result<LlmResponse> {
    let mut attempt = 1u64;
    loop {
        let emitted_output = Arc::new(AtomicBool::new(false));

        let token_sink = on_text.clone();
        let token_emitted = emitted_output.clone();
        let on_token: Box<dyn FnMut(&str) + Send> = Box::new(move |token: &str| {
            if !token.is_empty() {
                token_emitted.store(true, Ordering::SeqCst);
            }
            if let Ok(mut cb) = token_sink.lock() {
                cb(token);
            }
        });

        let thought_sink = on_thought.clone();
        let thought_emitted = emitted_output.clone();
        let on_thought_cb: Box<dyn FnMut(&str) + Send> = Box::new(move |token: &str| {
            if !token.is_empty() {
                thought_emitted.store(true, Ordering::SeqCst);
            }
            if let Ok(mut cb) = thought_sink.lock() {
                cb(token);
            }
        });

        let response = llm
            .stream_chat(StreamChatRequest {
                model: model.to_string(),
                messages: messages.to_vec(),
                tools: tools.clone(),
                reasoning_effort: reasoning_effort.map(str::to_string),
                structured_output: structured_output.cloned(),
                on_token,
                on_thought: on_thought_cb,
                cancel: cancel.clone(),
                idle_timeout,
            })
            .await;

        match response {
            Ok(response) => return Ok(response),
            Err(error)
                if attempt < crate::http_retry::LLM_MAX_ATTEMPTS
                    && !cancel.is_cancelled()
                    && !emitted_output.load(Ordering::SeqCst)
                    && is_retryable_llm_error(&error) =>
            {
                let delay = crate::http_retry::retry_backoff(attempt);
                append_trace_record(serde_json::json!({
                    "type": "llm_retry",
                    "turn": turn,
                    "attempt": attempt,
                    "max_attempts": crate::http_retry::LLM_MAX_ATTEMPTS,
                    "phase": "stream",
                    "reason": format!("{error:#}"),
                    "delay_ms": delay.as_millis(),
                }));
                tracing::warn!(
                    turn,
                    attempt,
                    max_attempts = crate::http_retry::LLM_MAX_ATTEMPTS,
                    "retrying transient LLM stream failure before any output was emitted"
                );
                crate::http_retry::sleep_before_retry(
                    "streaming LLM response",
                    attempt,
                    format!("{error:#}"),
                    Some(cancel),
                )
                .await?;
                attempt += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

/// Result of approving a permission request.
///
/// Shell commands can be approved for the session when they run under the
/// normal sandbox. A one-time outside-sandbox approval is intentionally never
/// persisted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PermissionGrant {
    allow_always: bool,
    sandbox_policy_override: Option<SandboxPolicy>,
}

fn trace_llm_request(
    turn: usize,
    model: &str,
    reasoning_effort: Option<&str>,
    messages: &[ChatMessage],
    tools: Option<&Vec<ToolDefinition>>,
) {
    append_trace_record(serde_json::json!({
        "type": "llm_request",
        "turn": turn,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "messages": messages,
        "tools": tools,
    }));
}

fn trace_llm_text_response(turn: usize, text: &str, usage: TokenUsage) {
    append_trace_record(serde_json::json!({
        "type": "llm_response",
        "turn": turn,
        "response": {
            "kind": "text",
            "text": text,
        },
        "usage": trace_usage(usage),
    }));
}

fn trace_llm_tool_response(
    turn: usize,
    text: &str,
    calls: &[crate::llm_client::ToolCall],
    usage: TokenUsage,
) {
    append_trace_record(serde_json::json!({
        "type": "llm_response",
        "turn": turn,
        "response": {
            "kind": "tool_calls",
            "text": text,
            "tool_calls": calls,
        },
        "usage": trace_usage(usage),
    }));
}

fn trace_llm_error(turn: usize, error: &anyhow::Error) {
    append_trace_record(serde_json::json!({
        "type": "llm_error",
        "turn": turn,
        "error": format!("{error:#}"),
    }));
}

fn trace_usage(usage: TokenUsage) -> serde_json::Value {
    serde_json::json!({
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "thought_tokens": usage.thought_tokens,
        "cached_read_tokens": usage.cached_read_tokens,
        "cached_write_tokens": usage.cached_write_tokens,
        "total_tokens": usage.total_tokens(),
    })
}

fn permission_options(tool_name: &str, shell_sandboxed: bool) -> Vec<PermissionOption> {
    let mut options = Vec::with_capacity(4);
    if tool_name == "run_shell_command" {
        if shell_sandboxed {
            options.push(PermissionOption::new(
                PermissionOptionId::new("allow"),
                "Allow in sandbox",
                PermissionOptionKind::AllowOnce,
            ));
            options.push(PermissionOption::new(
                PermissionOptionId::new("allow_always"),
                "Always allow this command in sandbox",
                PermissionOptionKind::AllowAlways,
            ));
            options.push(PermissionOption::new(
                PermissionOptionId::new("allow_outside_sandbox"),
                "Run outside sandbox once",
                PermissionOptionKind::AllowOnce,
            ));
        } else {
            options.push(PermissionOption::new(
                PermissionOptionId::new("allow"),
                "Allow",
                PermissionOptionKind::AllowOnce,
            ));
            options.push(PermissionOption::new(
                PermissionOptionId::new("allow_always"),
                "Always allow this command",
                PermissionOptionKind::AllowAlways,
            ));
        }
    } else {
        options.push(PermissionOption::new(
            PermissionOptionId::new("allow_always"),
            format!("Always allow {tool_name}"),
            PermissionOptionKind::AllowAlways,
        ));
        options.push(PermissionOption::new(
            PermissionOptionId::new("allow"),
            "Allow",
            PermissionOptionKind::AllowOnce,
        ));
    }
    options.push(PermissionOption::new(
        PermissionOptionId::new("reject"),
        "Reject",
        PermissionOptionKind::RejectOnce,
    ));
    options
}

fn permission_grant_for_selection(
    tool_name: &str,
    option_id: &str,
    shell_sandboxed: bool,
) -> Result<PermissionGrant, String> {
    match option_id {
        "allow_always" => Ok(PermissionGrant {
            allow_always: true,
            sandbox_policy_override: None,
        }),
        "allow" => Ok(PermissionGrant {
            allow_always: false,
            sandbox_policy_override: None,
        }),
        "allow_outside_sandbox" if tool_name == "run_shell_command" && shell_sandboxed => {
            Ok(PermissionGrant {
                allow_always: false,
                sandbox_policy_override: Some(SandboxPolicy::None),
            })
        }
        "reject" => Err("Tool use denied by user.".to_string()),
        other => {
            tracing::warn!(
                "request_permission returned unknown option id '{other}'; treating as reject"
            );
            Err("Tool use denied (unknown option selected).".to_string())
        }
    }
}

/// Shared text-emit callback. Held behind `Arc<Mutex<>>` so it can be cloned
/// into each streaming turn's `Box<dyn FnMut>` without being consumed.
pub type TextSink = Arc<Mutex<dyn FnMut(&str) + Send>>;

/// Whether `run()` emits per-tool `SessionUpdate` notifications to the
/// ACP client.
///
/// `Live` is the default for top-level runs: the user sees each tool
/// card appear, transition Pending -> InProgress -> Completed/Failed.
///
/// `Silent` is used for nested subagent runs invoked by the `task`
/// meta-tool: the subagent's tool-call noise stays out of the parent
/// conversation. Permission prompts still fire (`Silent` does not relax
/// the gate); only the progress cards are suppressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NotificationMode {
    Live,
    Silent,
}

/// Cap on subagent nesting. A depth of 1 means top-level agents can
/// invoke `task`, but the subagent they spawn cannot in turn invoke
/// another `task`. Prevents runaway recursion and keeps the catalog
/// from leaking into nested system prompts.
pub(crate) const MAX_SUBAGENT_DEPTH: usize = 1;

/// Per-subagent turn ceiling, applied as `parent_max_turns.min(...)` in
/// `execute_subagent`. Bounds the cost of a single delegation so a
/// parent run with `--max-turns 200` doesn't hand 200 turns to each
/// `task` invocation.
pub(crate) const MAX_SUBAGENT_TURNS: usize = 25;

/// Outcome of consulting the permission gate before executing a tool.
enum GateDecision {
    /// Run the tool without prompting.
    Allow {
        sandbox_policy_override: Option<SandboxPolicy>,
        sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
    },
    /// Refuse the call; feed the LLM the given denial message instead.
    Reject(String),
}

/// Witness type proving the holder is executing inside a `cx.spawn(...)` body.
///
/// `block_task()` (used by `request_user_permission`) deadlocks the dispatch
/// loop if invoked directly from a request handler -- it must run on a task
/// spawned via `ConnectionTo::spawn`. By threading `SpawnedCx<'_>` through
/// every `block_task` caller, we move the rule from scattered SAFETY comments
/// to a single named constructor that anyone violating it has to read first.
///
/// The constructor is `pub(crate)` and intentionally undocumented as
/// constructable elsewhere -- there is no compile-time check that it is only
/// called inside `cx.spawn`, but the doc here is the choke point for review.
pub(crate) struct SpawnedCx<'a> {
    cx: &'a ConnectionTo<Client>,
}

impl<'a> SpawnedCx<'a> {
    /// Construct only inside a `cx.spawn(async move { ... })` future.
    /// Calling this from a request handler and then invoking `block_task`
    /// downstream will deadlock the dispatch loop.
    pub(crate) fn new(cx: &'a ConnectionTo<Client>) -> Self {
        Self { cx }
    }

    fn cx(&self) -> &ConnectionTo<Client> {
        self.cx
    }
}

/// Result of the non-prompting portion of the gate. Pure (no I/O) so the
/// state-machine matrix can be unit-tested without a live ACP `cx` or store.
#[derive(Debug, PartialEq, Eq)]
enum PureGateDecision {
    Allow,
    Reject(String),
    Prompt,
}

/// Pure permission-gate logic. Given the snapshot of mode + kind + name +
/// always-allow membership, decide whether to allow, reject, or escalate to
/// the user. Kept separate from `consult_gate` so it can be tested in
/// isolation.
fn pure_gate_decision(
    mode: PermissionMode,
    kind: ToolKind,
    tool_name: &str,
    is_always_allowed: bool,
    shell_auto_allow: bool,
) -> PureGateDecision {
    // bypassPermissions: trust everything. Explicit user opt-out of the gate.
    if matches!(mode, PermissionMode::BypassPermissions) {
        return PureGateDecision::Allow;
    }

    // read-only: only allow strictly informational kinds, regardless of the
    // always-allow set. `Other` (Bifrost-loaded tools we haven't classified)
    // is refused so the user-visible "Refuse every edit, deletion, move, or
    // shell command" promise actually holds.
    if matches!(mode, PermissionMode::ReadOnly)
        && !matches!(
            kind,
            ToolKind::Read | ToolKind::Search | ToolKind::Think | ToolKind::Fetch
        )
    {
        return PureGateDecision::Reject(
            "Tool use denied: read-only mode forbids edits, deletions, moves, shell execution, \
             and any tool not classified as read/search/think/fetch. \
             Switch the Permission menu to 'default' or 'acceptEdits' to run this tool."
                .to_string(),
        );
    }

    // Mode-independent auto-allow: pure-info kinds never mutate. In addition,
    // sandboxed shell commands that fit a conservative read-only safelist may
    // run without a prompt in the editable modes; the OS sandbox remains the
    // hard boundary for filesystem writes.
    let auto_allow = match kind {
        ToolKind::Read | ToolKind::Search | ToolKind::Think | ToolKind::Fetch => true,
        ToolKind::Edit if matches!(mode, PermissionMode::AcceptEdits) => true,
        ToolKind::Execute
            if tool_name == "run_shell_command"
                && matches!(mode, PermissionMode::Default | PermissionMode::AcceptEdits)
                && shell_auto_allow =>
        {
            true
        }
        _ => false,
    };
    if auto_allow {
        return PureGateDecision::Allow;
    }

    // Remembered "Always allow". `consult_gate` chooses the cache key; shell
    // commands use repo-scoped prefix or exact-command keys, while regular
    // tools use the tool name.
    if is_always_allowed {
        return PureGateDecision::Allow;
    }

    PureGateDecision::Prompt
}

fn should_auto_allow_shell_command(
    raw_input: &Value,
    mode: PermissionMode,
    sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
    shell_sandboxed: bool,
) -> bool {
    if !matches!(mode, PermissionMode::Default | PermissionMode::AcceptEdits) {
        return false;
    }
    if !shell_sandboxed {
        return false;
    }
    if crate::sandbox_backend::resolve_mode(sandbox_mode) != crate::sandbox_backend::SandboxMode::Os
    {
        return false;
    }

    raw_input
        .get("command")
        .and_then(Value::as_str)
        .is_some_and(is_auto_approvable_sandboxed_shell_command)
}

fn is_auto_approvable_sandboxed_shell_command(command: &str) -> bool {
    let Some(tokens) = tokenize_simple_shell_command(command) else {
        return false;
    };

    let Some(program) = tokens.first().map(String::as_str) else {
        return false;
    };

    match program {
        "pwd" | "id" | "whoami" | "uname" | "echo" | "ls" | "cat" | "head" | "tail" | "wc"
        | "cut" | "tr" | "sort" | "uniq" | "nl" | "stat" | "which" | "grep" | "rg" => {
            !tokens_request_file_write(&tokens)
        }
        "find" => is_safe_find_command(&tokens),
        "sed" => is_safe_sed_command(&tokens),
        "awk" => is_safe_awk_command(&tokens),
        "git" => is_safe_git_command(&tokens),
        _ => false,
    }
}

fn tokenize_simple_shell_command(command: &str) -> Option<Vec<String>> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum QuoteState {
        Plain,
        Single,
        Double,
    }

    let mut state = QuoteState::Plain;
    let mut current = String::new();
    let mut tokens = Vec::new();
    let mut chars = command.chars().peekable();

    while let Some(ch) = chars.next() {
        match state {
            QuoteState::Plain => match ch {
                '\n' | '\r' => return None,
                ch if ch.is_whitespace() => {
                    if !current.is_empty() {
                        tokens.push(std::mem::take(&mut current));
                    }
                }
                '\'' => state = QuoteState::Single,
                '"' => state = QuoteState::Double,
                '\\' => {
                    let escaped = chars.next()?;
                    if escaped == '\n' || escaped == '\r' {
                        return None;
                    }
                    current.push(escaped);
                }
                '|' | '&' | ';' | '<' | '>' | '(' | ')' | '{' | '}' | '[' | ']' | '*' | '?'
                | '!' | '`' | '$' => return None,
                ch if ch.is_control() => return None,
                _ => current.push(ch),
            },
            QuoteState::Single => match ch {
                '\'' => state = QuoteState::Plain,
                ch if ch.is_control() && ch != '\t' => return None,
                _ => current.push(ch),
            },
            QuoteState::Double => match ch {
                '"' => state = QuoteState::Plain,
                '$' | '`' => return None,
                '\\' => {
                    let escaped = chars.next()?;
                    if escaped == '\n' || escaped == '\r' {
                        return None;
                    }
                    current.push(escaped);
                }
                ch if ch.is_control() && ch != '\t' => return None,
                _ => current.push(ch),
            },
        }
    }

    if state != QuoteState::Plain {
        return None;
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    if tokens.is_empty() {
        None
    } else {
        Some(tokens)
    }
}

fn tokens_request_file_write(tokens: &[String]) -> bool {
    tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "-i" | "-o" | "-f" | "--in-place" | "--output"
        ) || token.starts_with("--in-place=")
            || token.starts_with("--output=")
    })
}

fn is_safe_find_command(tokens: &[String]) -> bool {
    if tokens_request_file_write(tokens) {
        return false;
    }
    !tokens.iter().skip(1).any(|token| {
        matches!(
            token.as_str(),
            "-delete"
                | "-exec"
                | "-execdir"
                | "-ok"
                | "-okdir"
                | "-fls"
                | "-fprint"
                | "-fprint0"
                | "-fprintf"
                | "-ls"
                | "-print"
                | "-print0"
                | "-printf"
                | "-prune"
                | "-quit"
        )
    })
}

fn is_safe_sed_command(tokens: &[String]) -> bool {
    !tokens_request_file_write(tokens)
}

fn is_safe_awk_command(tokens: &[String]) -> bool {
    !tokens_request_file_write(tokens)
}

fn is_safe_git_command(tokens: &[String]) -> bool {
    if tokens.len() < 2 || tokens[1].starts_with('-') {
        return false;
    }

    matches!(
        tokens[1].as_str(),
        "status" | "diff" | "log" | "show" | "branch" | "rev-parse"
    )
}

fn shell_command_will_run_sandboxed(
    permission_mode: PermissionMode,
    sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
) -> bool {
    crate::tools::sandbox::shell_command_will_run_sandboxed(permission_mode, sandbox_mode)
}

fn always_allow_key(
    tool_name: &str,
    raw_input: &Value,
    cwd: &Path,
    shell_sandboxed: bool,
) -> String {
    if tool_name == "run_shell_command"
        && let Some(command) = raw_input.get("command").and_then(Value::as_str)
    {
        return serde_json::json!({
            "tool": tool_name,
            "cwd": cwd.display().to_string(),
            "command": command,
            "shellSandboxed": shell_sandboxed,
        })
        .to_string();
    }

    tool_name.to_string()
}

fn shell_always_allow_rule_key(raw_input: &Value, shell_sandboxed: bool) -> Option<String> {
    let command = raw_input.get("command").and_then(Value::as_str)?;
    if let Some(tokens) = tokenize_simple_shell_command(command) {
        let argv_prefix: Vec<String> = tokens.into_iter().take(3).collect();
        if !argv_prefix.is_empty() {
            return Some(
                serde_json::json!({
                    "tool": "run_shell_command",
                    "rule": "prefix",
                    "argvPrefix": argv_prefix,
                    "shellSandboxed": shell_sandboxed,
                })
                .to_string(),
            );
        }
    }

    Some(
        serde_json::json!({
            "tool": "run_shell_command",
            "rule": "exact",
            "command": command,
            "shellSandboxed": shell_sandboxed,
        })
        .to_string(),
    )
}

fn always_allow_lookup_keys(
    tool_name: &str,
    raw_input: &Value,
    cwd: &Path,
    shell_sandboxed: bool,
) -> Vec<String> {
    if tool_name == "run_shell_command" {
        let mut keys = Vec::with_capacity(2);
        if let Some(rule_key) = shell_always_allow_rule_key(raw_input, shell_sandboxed) {
            keys.push(rule_key);
        }
        let legacy_key = always_allow_key(tool_name, raw_input, cwd, shell_sandboxed);
        if !legacy_key.is_empty() {
            keys.push(legacy_key);
        }
        return keys;
    }

    vec![tool_name.to_string()]
}

/// Decide which sandbox policy to use for an approved tool call.
///
/// A per-call shell override should only survive if the session still exists;
/// if the session disappears between approval and execution, we fall back to
/// `ReadOnly` and drop the override.
fn resolve_execution_policy(
    permission_mode: Option<PermissionMode>,
    sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
    sandbox_policy_override: Option<SandboxPolicy>,
) -> (SandboxPolicy, bool) {
    match (sandbox_policy_override, permission_mode) {
        (Some(policy), Some(_)) => (policy, true),
        (Some(_), None) => (SandboxPolicy::ReadOnly, false),
        (None, Some(mode)) => (SandboxPolicy::resolve(mode, sandbox_mode), false),
        (None, None) => (SandboxPolicy::ReadOnly, false),
    }
}

/// Run the agentic tool-calling loop.
///
/// Sends messages to the LLM with tool definitions. If the LLM responds with
/// tool calls, executes them, appends results, and loops. Stops when the LLM
/// responds with text only or the turn limit is reached.
///
/// `on_text` is invoked for each text token streamed from the LLM, in real time.
/// Tool-call lifecycle is reported to the client via `SessionUpdate::ToolCall`
/// and `SessionUpdate::ToolCallUpdate` notifications (Pending -> InProgress ->
/// Completed/Failed).
///
/// Each tool call is gated through the session's permission policy: depending on
/// the session's `PermissionMode` and the tool's `ToolKind`, a call is auto-allowed,
/// auto-rejected, or escalated to the client via `session/request_permission`.
///
/// SAFETY: this function calls `SentRequest::block_task().await`, which is only
/// safe inside `ConnectionTo::spawn`. The `SpawnedCx<'_>` parameter encodes
/// that requirement -- callers must construct it inside a spawned task.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run(
    llm: &Arc<dyn LlmBackend>,
    registry: &ToolRegistry,
    model: &str,
    reasoning_effort: Option<&str>,
    structured_output: Option<&StructuredOutputRequest>,
    mut messages: Vec<ChatMessage>,
    max_turns: usize,
    idle_timeout: Duration,
    cancel: CancellationToken,
    on_text: TextSink,
    on_thought: TextSink,
    spawned_cx: SpawnedCx<'_>,
    session_id: String,
    sessions: SessionStore,
    notifications: NotificationMode,
    depth: usize,
) -> (String, Vec<ToolExchange>, TokenUsage) {
    let train_bifrost = train_bifrost_enabled();
    let training_packet = if train_bifrost {
        match train_bifrost::load_packet_from_env() {
            Ok(packet) => Some(packet),
            Err(error) => {
                append_trace_record(serde_json::json!({
                    "type": "train_bifrost_config_error",
                    "error": format!("{error:#}"),
                }));
                return (
                    format!("BRK_TRAIN_BIFROST is misconfigured: {error:#}"),
                    Vec::new(),
                    TokenUsage::default(),
                );
            }
        }
    } else {
        None
    };
    if train_bifrost {
        registry
            .set_builtin_tools(train_bifrost_initial_builtin_tools())
            .await;
    }
    let mut tools: Vec<ToolDefinition> = registry.tool_definitions().await;
    // Nested runs (subagents) must not see the `task` tool themselves --
    // capping depth at `MAX_SUBAGENT_DEPTH` and stripping `task` from the
    // catalog at deeper levels prevents an unbounded recursion of
    // subagents-spawning-subagents.
    if depth >= MAX_SUBAGENT_DEPTH {
        tools.retain(|t| t.function.name != "task");
    }
    let mut full_response = String::new();
    // Captured per-call so the caller can persist them with the turn (#3409),
    // letting a `session/load` re-feed the LLM the same tool context the
    // model had when it produced `full_response`.
    let mut tool_exchanges: Vec<ToolExchange> = Vec::new();
    // Aggregate the per-call usage reported by the LLM across every
    // turn of this tool loop (one prompt may issue many `stream_chat`
    // calls as it dispatches tools). The caller adds this to the
    // session-wide running total before emitting `PromptResponse.usage`.
    let mut turn_usage = TokenUsage::default();
    let mut no_edit_progress_nudge_count = 0usize;
    'outer: for turn in 0..max_turns {
        if cancel.is_cancelled() {
            break;
        }
        let permission_mode = sessions
            .permission_mode(&session_id)
            .await
            .unwrap_or(PermissionMode::ReadOnly);
        if train_bifrost
            && should_emit_no_edit_progress_nudge(
                permission_mode,
                turn,
                max_turns,
                &tool_exchanges,
                no_edit_progress_nudge_count,
            )
        {
            no_edit_progress_nudge_count += 1;
            match build_train_bifrost_nudge(
                llm,
                turn,
                &messages,
                &tool_exchanges,
                training_packet.as_ref(),
                &cancel,
                idle_timeout,
            )
            .await
            {
                Some((nudge, usage)) => {
                    turn_usage.add(usage);
                    append_trace_record(serde_json::json!({
                        "type": "no_edit_progress_nudge",
                        "turn": turn,
                        "nudge_count": no_edit_progress_nudge_count,
                        "executed_tool_counts": executed_tool_counts(&tool_exchanges),
                        "message": nudge,
                    }));
                    messages.push(ChatMessage::user(nudge));
                }
                None => {
                    append_trace_record(serde_json::json!({
                        "type": "no_edit_progress_nudge_skipped",
                        "turn": turn,
                        "nudge_count": no_edit_progress_nudge_count,
                        "executed_tool_counts": executed_tool_counts(&tool_exchanges),
                    }));
                }
            }
        }

        // For the last turn, normally force a text response. If no file
        // change has succeeded yet, keep tools available so a hard task does
        // not end with a false "I cannot edit" answer solely because the
        // harness withheld edit/write tools on the final turn.
        let force_text_response =
            turn >= max_turns - 1 && has_successful_file_change(&tool_exchanges);
        let turn_tools = if !force_text_response {
            Some(tools.clone())
        } else {
            None
        };

        // Wall-clock bound on this stream is enforced by the reqwest client's
        // own `.timeout(...)` (see `OpenAiClient::new`). Per-chunk idle
        // inactivity (the case in #3366 / #3453: streams that drip
        // occasional bytes and would defeat wall-clock) is enforced inside
        // the SSE driver via `idle_timeout`, threaded here from
        // `--llm-idle-timeout-secs` and the per-session `/idle-timeout`
        // override.
        let request_tools = turn_tools.clone();
        let advertised_this_request = advertised_tool_names(request_tools.as_ref());
        trace_llm_request(
            turn,
            model,
            reasoning_effort,
            &messages,
            request_tools.as_ref(),
        );

        let response = stream_chat_with_transient_retry(
            llm,
            turn,
            model,
            &messages,
            request_tools,
            reasoning_effort,
            structured_output,
            &on_text,
            &on_thought,
            &cancel,
            idle_timeout,
        )
        .await;

        match response {
            Ok(LlmResponse::Text { text, usage }) => {
                trace_llm_text_response(turn, &text, usage);
                turn_usage.add(usage);
                if train_bifrost
                    && should_reject_no_edit_final_answer(
                        permission_mode,
                        turn,
                        max_turns,
                        &tool_exchanges,
                    )
                {
                    match build_train_bifrost_nudge(
                        llm,
                        turn,
                        &messages,
                        &tool_exchanges,
                        training_packet.as_ref(),
                        &cancel,
                        idle_timeout,
                    )
                    .await
                    {
                        Some((nudge, hint_usage)) => {
                            turn_usage.add(hint_usage);
                            append_trace_record(serde_json::json!({
                                "type": "no_edit_final_answer_guard",
                                "turn": turn,
                                "executed_tool_counts": executed_tool_counts(&tool_exchanges),
                                "text": text,
                                "message": nudge,
                            }));
                            messages.push(ChatMessage::assistant(text));
                            messages.push(ChatMessage::user(nudge));
                            continue;
                        }
                        None => {
                            append_trace_record(serde_json::json!({
                                "type": "no_edit_final_answer_guard_skipped",
                                "turn": turn,
                                "executed_tool_counts": executed_tool_counts(&tool_exchanges),
                                "text": text,
                            }));
                        }
                    }
                }
                full_response.push_str(&text);
                // Final text response -- we're done
                break;
            }
            Ok(LlmResponse::ToolCalls { text, calls, usage }) => {
                trace_llm_tool_response(turn, &text, &calls, usage);
                turn_usage.add(usage);
                // Any text emitted before tool calls
                if !text.is_empty() {
                    full_response.push_str(&text);
                }

                // Record the assistant message with tool_calls
                messages.push(ChatMessage::assistant_tool_calls(calls.clone()));

                // Execute each tool call
                for call_index in
                    ordered_tool_call_indices(&calls, |name| registry.is_bifrost_tool(name))
                {
                    let call = &calls[call_index];
                    if cancel.is_cancelled() {
                        // The user cancelled mid-batch; stop issuing more
                        // permission prompts and tool executions.
                        break 'outer;
                    }

                    let tool_name = call.function.name.clone();
                    let kind = ToolRegistry::tool_kind(&tool_name);

                    // Parse the LLM's serialized arguments up front so the
                    // tool-call card can pull `path` / `command` / `pattern`
                    // out for the title, and so an arg-parse failure becomes
                    // a Failed card rather than a silent fallback.
                    let parsed_input = match serde_json::from_str::<Value>(&call.function.arguments)
                    {
                        Ok(v) => v,
                        Err(e) => {
                            let reason = format!(
                                "Error: tool arguments are not valid JSON ({e}). \
                                 Please retry with a valid JSON object matching the tool schema."
                            );
                            // Render the card anyway so the user sees what
                            // the agent tried to invoke -- raw_input falls
                            // back to the unparsed string.
                            maybe_send_session_update(
                                notifications,
                                spawned_cx.cx(),
                                &session_id,
                                SessionUpdate::ToolCall(announce::initial_tool_call(
                                    &call.id,
                                    &tool_name,
                                    kind,
                                    &Value::String(call.function.arguments.clone()),
                                )),
                            );
                            maybe_send_session_update(
                                notifications,
                                spawned_cx.cx(),
                                &session_id,
                                SessionUpdate::ToolCallUpdate(announce::update_failed(
                                    &call.id,
                                    &reason,
                                    Some(Value::String(reason.clone())),
                                )),
                            );
                            messages.push(ChatMessage::tool_result(&call.id, &tool_name, &reason));
                            // Record the failed exchange so a session/load
                            // sees that the model attempted this call (with
                            // unparseable args) and got rejected; without it
                            // the model might re-emit the same broken call.
                            tool_exchanges.push(ToolExchange {
                                call_id: call.id.clone(),
                                tool_name: tool_name.clone(),
                                arguments: call.function.arguments.clone(),
                                result: reason,
                            });
                            continue;
                        }
                    };

                    // Refuse outright if the permission card would hide input.
                    // Two separate gates apply:
                    //   • Non-shell tools: title length capped at
                    //     MAX_TOOL_TITLE_CHARS (1024 chars).
                    //   • Shell commands: full command text capped at
                    //     MAX_INLINE_OUTPUT_BYTES (50 000 bytes). The modal
                    //     title carries the full command for clients that only
                    //     render the title, so the content-size bound is the
                    //     right limit for shell.
                    // Reject rather than truncating; the LLM can retry with
                    // smaller arguments.
                    if let Some(reason) = announce::rejection_for_oversized_title(
                        &tool_name,
                        &parsed_input,
                    )
                    .or_else(|| {
                        announce::rejection_for_oversized_input_content(&tool_name, &parsed_input)
                    }) {
                        tracing::warn!(
                            session_id = %session_id,
                            tool_name = %tool_name,
                            title_chars = announce::permission_prompt_title(&tool_name, &parsed_input)
                                .chars()
                                .count(),
                            "rejecting tool call: rendered permission card would hide input",
                        );
                        maybe_send_session_update(
                            notifications,
                            spawned_cx.cx(),
                            &session_id,
                            SessionUpdate::ToolCall(announce::rejected_initial_tool_call(
                                &call.id,
                                &tool_name,
                                kind,
                                &parsed_input,
                            )),
                        );
                        maybe_send_session_update(
                            notifications,
                            spawned_cx.cx(),
                            &session_id,
                            SessionUpdate::ToolCallUpdate(announce::update_failed(
                                &call.id,
                                &reason,
                                Some(Value::String(reason.clone())),
                            )),
                        );
                        messages.push(ChatMessage::tool_result(&call.id, &tool_name, &reason));
                        tool_exchanges.push(ToolExchange {
                            call_id: call.id.clone(),
                            tool_name: tool_name.clone(),
                            arguments: call.function.arguments.clone(),
                            result: reason,
                        });
                        continue;
                    }

                    if let Some(message) = deterministic_gate_rejection(
                        &sessions,
                        &session_id,
                        &tool_name,
                        kind,
                        &parsed_input,
                        registry.cwd(),
                    )
                    .await
                    {
                        let (blocked_call, failed_update) = blocked_tool_call_updates(
                            &call.id,
                            &tool_name,
                            kind,
                            &parsed_input,
                            &message,
                        );
                        maybe_send_session_update(
                            notifications,
                            spawned_cx.cx(),
                            &session_id,
                            SessionUpdate::ToolCall(blocked_call),
                        );
                        maybe_send_session_update(
                            notifications,
                            spawned_cx.cx(),
                            &session_id,
                            SessionUpdate::ToolCallUpdate(failed_update),
                        );
                        messages.push(ChatMessage::tool_result(&call.id, &tool_name, &message));
                        tool_exchanges.push(ToolExchange {
                            call_id: call.id.clone(),
                            tool_name: tool_name.clone(),
                            arguments: call.function.arguments.clone(),
                            result: message,
                        });
                        continue;
                    }

                    // Pending -- emit the card before the gate runs so the
                    // permission modal (which reuses this id) renders against
                    // a card that already shows path / command / etc.
                    maybe_send_session_update(
                        notifications,
                        spawned_cx.cx(),
                        &session_id,
                        SessionUpdate::ToolCall(announce::initial_tool_call(
                            &call.id,
                            &tool_name,
                            kind,
                            &parsed_input,
                        )),
                    );

                    if !advertised_this_request.contains(tool_name.as_str()) {
                        let message = tool_unavailable_message(&tool_name);
                        maybe_send_session_update(
                            notifications,
                            spawned_cx.cx(),
                            &session_id,
                            SessionUpdate::ToolCallUpdate(announce::update_failed(
                                &call.id,
                                &message,
                                Some(Value::String(message.clone())),
                            )),
                        );
                        messages.push(ChatMessage::tool_result(&call.id, &tool_name, &message));
                        tool_exchanges.push(ToolExchange {
                            call_id: call.id.clone(),
                            tool_name: tool_name.clone(),
                            arguments: call.function.arguments.clone(),
                            result: message,
                        });
                        continue;
                    }

                    // Consult the gate before announcing or executing the call.
                    let decision = consult_gate(
                        &sessions,
                        &spawned_cx,
                        &cancel,
                        GateCheck {
                            session_id: &session_id,
                            tool_name: &tool_name,
                            kind,
                            tool_call_id: &call.id,
                            raw_input: &parsed_input,
                            cwd: registry.cwd(),
                        },
                    )
                    .await;

                    let output = match decision {
                        GateDecision::Reject(message) => {
                            // Failed terminal update so the card reflects the
                            // denial and doesn't sit at Pending forever.
                            maybe_send_session_update(
                                notifications,
                                spawned_cx.cx(),
                                &session_id,
                                SessionUpdate::ToolCallUpdate(announce::update_failed(
                                    &call.id,
                                    &message,
                                    Some(Value::String(message.clone())),
                                )),
                            );
                            message
                        }
                        GateDecision::Allow {
                            sandbox_policy_override,
                            sandbox_mode,
                        } => {
                            maybe_send_session_update(
                                notifications,
                                spawned_cx.cx(),
                                &session_id,
                                SessionUpdate::ToolCallUpdate(announce::update_in_progress(
                                    &call.id,
                                )),
                            );

                            // Capture pre-write content so write/edit tools get a
                            // real Diff card. Outer None == not an edit tool,
                            // or prior content unavailable (binary, missing
                            // parent dir we can't resolve, etc) -- in either
                            // case we fall back to text content. Inner None
                            // (per ACP `Diff.old_text` schema) == new file.
                            let pre_write: Option<Option<String>> =
                                if matches!(tool_name.as_str(), "write_file" | "edit") {
                                    capture_pre_write_text(registry.cwd(), &parsed_input)
                                } else {
                                    None
                                };

                            // Resolve the sandbox tier from the session's permission mode.
                            // If the session disappeared between gate-accept and exec
                            // (race), fail safe to ReadOnly: the gate already cleared
                            // the call but we no longer trust the mode lookup.
                            let permission_mode = sessions.permission_mode(&session_id).await;
                            if permission_mode.is_none() {
                                tracing::warn!(
                                    session_id,
                                    tool_name,
                                    outside_sandbox_once = sandbox_policy_override.is_some(),
                                    "session vanished between gate-accept and exec; falling back to ReadOnly sandbox"
                                );
                            }
                            let (policy, outside_sandbox_once) = resolve_execution_policy(
                                permission_mode,
                                sandbox_mode,
                                sandbox_policy_override,
                            );

                            tracing::info!(
                                "executing tool {} with args: {} (sandbox={:?}, outside_sandbox_once={})",
                                tool_name,
                                call.function.arguments,
                                policy,
                                outside_sandbox_once
                            );

                            // `task` short-circuits the registry: it needs
                            // `llm`/`spawned_cx`/`sessions` to spin up a
                            // nested `run()` for the subagent, none of
                            // which the registry sees. The gate has
                            // already cleared this call (kind=Other =>
                            // prompted in `default`, refused in
                            // `readOnly`), so by the time we get here we
                            // know the user has authorized the dispatch.
                            let exec = if tool_name == "task" {
                                let (exec, nested_usage) = execute_subagent(
                                    llm,
                                    registry,
                                    model,
                                    reasoning_effort,
                                    structured_output,
                                    &parsed_input,
                                    max_turns,
                                    idle_timeout,
                                    cancel.clone(),
                                    &spawned_cx,
                                    &session_id,
                                    &sessions,
                                    depth + 1,
                                )
                                .await;
                                // A subagent burns its own tokens against
                                // the same upstream account; surface them
                                // in the parent's `PromptResponse.usage`
                                // so the client sees the true cost of the
                                // turn, not just the parent's own calls.
                                turn_usage.add(nested_usage);
                                exec
                            } else {
                                trace_bifrost_context_shadow(
                                    &tool_name,
                                    &parsed_input,
                                    &tool_exchanges,
                                );
                                execute_tool(
                                    registry,
                                    &tool_name,
                                    parsed_input.clone(),
                                    policy,
                                    outside_sandbox_once,
                                    sandbox_mode,
                                )
                                .await
                            };

                            // Build the terminal update -- Completed (with a
                            // Diff for write/edit tools when we have prior content)
                            // or Failed (for tool-reported errors).
                            let update = if exec.failed {
                                announce::update_failed_with_input(
                                    &call.id,
                                    &tool_name,
                                    &parsed_input,
                                    &exec.output,
                                    Some(Value::String(exec.output.clone())),
                                )
                            } else {
                                let diff = pre_write.and_then(|prior| {
                                    build_editing_diff(&tool_name, &parsed_input, prior)
                                });
                                announce::update_completed(
                                    &call.id,
                                    &tool_name,
                                    &parsed_input,
                                    &exec.output,
                                    diff,
                                )
                            };
                            maybe_send_session_update(
                                notifications,
                                spawned_cx.cx(),
                                &session_id,
                                SessionUpdate::ToolCallUpdate(update),
                            );
                            exec.output
                        }
                    };

                    messages.push(ChatMessage::tool_result(&call.id, &tool_name, &output));
                    tool_exchanges.push(ToolExchange {
                        call_id: call.id.clone(),
                        tool_name: tool_name.clone(),
                        arguments: call.function.arguments.clone(),
                        result: output,
                    });
                }
                if train_bifrost
                    && has_successful_file_change(&tool_exchanges)
                    && !registry
                        .is_builtin_tool_advertised("run_shell_command")
                        .await
                {
                    registry
                        .set_builtin_tools(train_bifrost_post_edit_builtin_tools())
                        .await;
                    tools = registry.tool_definitions().await;
                    if depth >= MAX_SUBAGENT_DEPTH {
                        tools.retain(|t| t.function.name != "task");
                    }
                }
            }
            Err(e) => {
                trace_llm_error(turn, &e);
                let friendly = messages_include_images(&messages)
                    .then(|| rewrite_image_prompt_provider_error(&e.to_string()))
                    .flatten();
                let err_msg = if let Some(message) = friendly {
                    format!("\n**Error:** {message}\n")
                } else {
                    format!("\n**Error:** LLM request failed: {e}\n")
                };
                if let Ok(mut cb) = on_text.lock() {
                    cb(&err_msg);
                }
                full_response.push_str(&err_msg);
                break;
            }
        }
    }

    (full_response, tool_exchanges, turn_usage)
}

fn blocked_tool_call_updates(
    tool_call_id: &str,
    tool_name: &str,
    kind: ToolKind,
    raw_input: &Value,
    reason: &str,
) -> (agent_client_protocol::schema::ToolCall, ToolCallUpdate) {
    (
        announce::blocked_tool_call(tool_call_id, tool_name, kind, raw_input, reason),
        announce::update_failed(
            tool_call_id,
            reason,
            Some(Value::String(reason.to_string())),
        ),
    )
}

struct PureGateEvaluation {
    decision: PureGateDecision,
    sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
    shell_sandboxed: bool,
}

async fn evaluate_pure_gate(
    sessions: &SessionStore,
    session_id: &str,
    tool_name: &str,
    kind: ToolKind,
    raw_input: &Value,
    cwd: &Path,
) -> Result<PureGateEvaluation, String> {
    let mode = match sessions.permission_mode(session_id).await {
        Some(m) => m,
        None => {
            tracing::warn!(
                session_id,
                tool_name,
                "permission gate: session not found; refusing tool"
            );
            return Err("Tool use denied: session is no longer registered. \
                 Start a new prompt to continue."
                .to_string());
        }
    };
    let sandbox_mode = sessions.sandbox_mode(session_id).await.flatten();
    let shell_sandboxed =
        tool_name == "run_shell_command" && shell_command_will_run_sandboxed(mode, sandbox_mode);
    let shell_auto_allow = tool_name == "run_shell_command"
        && should_auto_allow_shell_command(raw_input, mode, sandbox_mode, shell_sandboxed);
    let always_allow_keys = always_allow_lookup_keys(tool_name, raw_input, cwd, shell_sandboxed);
    let is_always_allowed = sessions
        .is_any_always_allowed(session_id, &always_allow_keys)
        .await;
    let decision = pure_gate_decision(mode, kind, tool_name, is_always_allowed, shell_auto_allow);

    Ok(PureGateEvaluation {
        decision,
        sandbox_mode,
        shell_sandboxed,
    })
}

/// Return a deterministic permission denial, if one exists before any user
/// prompt is needed. Promptable calls still go through `consult_gate` after
/// the pending card is emitted so the permission modal has a matching card id.
async fn deterministic_gate_rejection(
    sessions: &SessionStore,
    session_id: &str,
    tool_name: &str,
    kind: ToolKind,
    raw_input: &Value,
    cwd: &Path,
) -> Option<String> {
    match evaluate_pure_gate(sessions, session_id, tool_name, kind, raw_input, cwd).await {
        Err(msg) => Some(msg),
        Ok(PureGateEvaluation {
            decision: PureGateDecision::Reject(msg),
            ..
        }) => Some(msg),
        Ok(PureGateEvaluation {
            decision: PureGateDecision::Allow | PureGateDecision::Prompt,
            ..
        }) => None,
    }
}

/// Apply the per-call permission policy. Returns `Allow` if the tool should
/// execute, or `Reject(msg)` to feed the LLM a denial message instead.
async fn consult_gate(
    sessions: &SessionStore,
    spawned_cx: &SpawnedCx<'_>,
    cancel: &CancellationToken,
    request: GateCheck<'_>,
) -> GateDecision {
    let evaluation = match evaluate_pure_gate(
        sessions,
        request.session_id,
        request.tool_name,
        request.kind,
        request.raw_input,
        request.cwd,
    )
    .await
    {
        Ok(evaluation) => evaluation,
        Err(reason) => return GateDecision::Reject(reason),
    };

    match evaluation.decision {
        PureGateDecision::Allow => GateDecision::Allow {
            sandbox_policy_override: None,
            sandbox_mode: evaluation.sandbox_mode,
        },
        PureGateDecision::Reject(msg) => GateDecision::Reject(msg),
        PureGateDecision::Prompt => {
            match request_user_permission(
                spawned_cx,
                cancel,
                PermissionRequest {
                    session_id: request.session_id,
                    tool_name: request.tool_name,
                    kind: request.kind,
                    tool_call_id: request.tool_call_id,
                    raw_input: request.raw_input,
                    shell_sandboxed: evaluation.shell_sandboxed,
                },
            )
            .await
            {
                Ok(grant) => {
                    // Awaited inline so the next tool call in the same batch
                    // sees the updated set without re-prompting.
                    if grant.allow_always && grant.sandbox_policy_override.is_none() {
                        if request.tool_name == "run_shell_command" {
                            if let Some(rule_key) = shell_always_allow_rule_key(
                                request.raw_input,
                                evaluation.shell_sandboxed,
                            ) {
                                sessions
                                    .add_always_allow(request.session_id, &rule_key)
                                    .await;
                            }
                        } else {
                            sessions
                                .add_always_allow(request.session_id, request.tool_name)
                                .await;
                        }
                    }
                    GateDecision::Allow {
                        sandbox_policy_override: grant.sandbox_policy_override,
                        sandbox_mode: evaluation.sandbox_mode,
                    }
                }
                Err(reason) => GateDecision::Reject(reason),
            }
        }
    }
}

/// Send `session/request_permission` to the client and await the outcome.
/// Returns `Ok(grant)` if the user approved (with or without remembering),
/// or `Err(reason)` describing the rejection or transport failure.
struct GateCheck<'a> {
    session_id: &'a str,
    tool_name: &'a str,
    kind: ToolKind,
    tool_call_id: &'a str,
    raw_input: &'a Value,
    cwd: &'a Path,
}

struct PermissionRequest<'a> {
    session_id: &'a str,
    tool_name: &'a str,
    kind: ToolKind,
    tool_call_id: &'a str,
    raw_input: &'a Value,
    shell_sandboxed: bool,
}

async fn request_user_permission(
    spawned_cx: &SpawnedCx<'_>,
    cancel: &CancellationToken,
    request: PermissionRequest<'_>,
) -> Result<PermissionGrant, String> {
    let PermissionRequest {
        session_id,
        tool_name,
        kind,
        tool_call_id,
        raw_input,
        shell_sandboxed,
    } = request;

    // The permission modal needs to show *what* is being approved, not just
    // the tool kind. Shell commands use a dedicated builder that includes the
    // full command text because some clients only surface the modal title.
    //
    // Assumes the caller has already filtered oversized titles via
    // `announce::rejection_for_oversized_title` in `run`; the debug assert
    // catches any future path that reaches the modal without that gate.
    let title = announce::permission_prompt_title(tool_name, raw_input);
    // Shell titles carry the full command and are bounded at MAX_INLINE_OUTPUT_BYTES
    // by rejection_for_oversized_input_content; non-shell titles are bounded at
    // MAX_TOOL_TITLE_CHARS by rejection_for_oversized_title.
    let max_title_chars = if tool_name == "run_shell_command" {
        announce::MAX_INLINE_OUTPUT_BYTES
    } else {
        announce::MAX_TOOL_TITLE_CHARS
    };
    debug_assert!(
        title.chars().count() <= max_title_chars,
        "request_user_permission: oversized title bypassed the pre-gate check \
         (tool={tool_name}, chars={})",
        title.chars().count()
    );
    let fields = ToolCallUpdateFields::new()
        .kind(kind)
        .status(ToolCallStatus::Pending)
        .title(title)
        .content(announce::tool_input_content(tool_name, raw_input))
        .raw_input(raw_input.clone());
    let tool_call = ToolCallUpdate::new(ToolCallId::new(tool_call_id.to_string()), fields);

    let options = permission_options(tool_name, shell_sandboxed);

    let request = RequestPermissionRequest::new(session_id.to_string(), tool_call, options);
    emit_terminal_notification(TerminalNotificationEvent::Prompt);

    // block_task() is only safe inside ConnectionTo::spawn; see the SAFETY note
    // on `run` above. We deliberately do not apply a local timeout here: ACP
    // has no per-request cancel API, and dropping an in-flight SentRequest can
    // leave the client free to answer a request whose receiver no longer
    // exists. A user-visible permission prompt is allowed to wait indefinitely
    // until the user either chooses an option or cancels the prompt/session.
    let response = tokio::select! {
        biased;
        _ = cancel.cancelled() => {
            tracing::warn!(
                session_id,
                tool_name,
                "permission request abandoned due to session cancel; client should dismiss the modal"
            );
            return Err("Tool use denied: the prompt was cancelled before the user responded.".to_string());
        }
        r = spawned_cx.cx().send_request(request).block_task() => r,
    };

    match response {
        Ok(resp) => match resp.outcome {
            RequestPermissionOutcome::Selected(selected) => {
                let id: &str = &selected.option_id.0;
                permission_grant_for_selection(tool_name, id, shell_sandboxed)
            }
            RequestPermissionOutcome::Cancelled => Err(
                "Tool use denied: the prompt was cancelled before the user responded.".to_string(),
            ),
            // Future-proof: schema is #[non_exhaustive].
            _ => Err("Tool use denied: unknown permission outcome.".to_string()),
        },
        Err(err) => {
            tracing::warn!("request_permission transport error: {err}");
            Err(format!(
                "Tool use denied: permission request failed ({err})."
            ))
        }
    }
}

/// Outcome of executing a tool, formatted for both the LLM (via `output`)
/// and the client card (`failed` -> `ToolCallStatus::Failed`).
#[derive(Clone)]
struct ToolExecution {
    output: String,
    failed: bool,
}

#[cfg(test)]
fn has_tool(tools: &[ToolDefinition], name: &str) -> bool {
    tools.iter().any(|tool| tool.function.name == name)
}

#[cfg(test)]
fn is_text_navigation_tool(name: &str) -> bool {
    matches!(name, "read_file" | "grep_search" | "list_directory")
}

fn executed_tool_counts(tool_exchanges: &[ToolExchange]) -> Value {
    serde_json::json!({
        "read_file": tool_exchanges.iter().filter(|exchange| exchange.tool_name == "read_file").count(),
        "grep_search": tool_exchanges.iter().filter(|exchange| exchange.tool_name == "grep_search").count(),
        "list_directory": tool_exchanges.iter().filter(|exchange| exchange.tool_name == "list_directory").count(),
        "run_shell_command": tool_exchanges.iter().filter(|exchange| exchange.tool_name == "run_shell_command").count(),
        "search_symbols": tool_exchanges.iter().filter(|exchange| exchange.tool_name == "search_symbols").count(),
        "get_symbol_sources": tool_exchanges.iter().filter(|exchange| exchange.tool_name == "get_symbol_sources").count(),
        "scan_usages": tool_exchanges.iter().filter(|exchange| exchange.tool_name == "scan_usages").count(),
        "get_summaries": tool_exchanges.iter().filter(|exchange| exchange.tool_name == "get_summaries").count(),
    })
}

fn has_successful_file_change(tool_exchanges: &[ToolExchange]) -> bool {
    tool_exchanges.iter().any(|exchange| {
        matches!(exchange.tool_name.as_str(), "edit" | "write_file")
            && !tool_result_failed(&exchange.result)
    })
}

fn tool_result_failed(result: &str) -> bool {
    result.starts_with("Error:") || result.starts_with("Internal error:")
}

fn is_bifrost_context_tool(name: &str) -> bool {
    matches!(
        name,
        "search_symbols"
            | "get_symbol_sources"
            | "scan_usages"
            | "get_summaries"
            | "list_symbols"
            | "get_symbol_locations"
    )
}

fn trace_bifrost_context_shadow(
    tool_name: &str,
    parsed_input: &Value,
    tool_exchanges: &[ToolExchange],
) {
    if !is_bifrost_context_tool(tool_name) {
        return;
    }
    let mut prior_same_tool_count = 0usize;
    let mut prior_exact_args_count = 0usize;
    for exchange in tool_exchanges {
        if exchange.tool_name != tool_name {
            continue;
        }
        prior_same_tool_count += 1;
        if !tool_result_failed(&exchange.result)
            && serde_json::from_str::<Value>(&exchange.arguments)
                .is_ok_and(|prior_args| prior_args == *parsed_input)
        {
            prior_exact_args_count += 1;
        }
    }
    append_trace_record(serde_json::json!({
        "type": "bifrost_context_shadow",
        "tool": tool_name,
        "args": parsed_input,
        "prior_same_tool_count": prior_same_tool_count,
        "prior_exact_args_count": prior_exact_args_count,
        "has_successful_file_change": has_successful_file_change(tool_exchanges),
        "exact_source_count": exact_source_tool_count(tool_exchanges),
        "executed_tool_counts": executed_tool_counts(tool_exchanges),
    }));
}

fn should_reject_no_edit_final_answer(
    permission_mode: PermissionMode,
    turn: usize,
    max_turns: usize,
    tool_exchanges: &[ToolExchange],
) -> bool {
    if matches!(permission_mode, PermissionMode::ReadOnly) {
        return false;
    }
    if turn >= max_turns - 1 || has_successful_file_change(tool_exchanges) {
        return false;
    }
    tool_exchanges.iter().any(|exchange| {
        matches!(
            exchange.tool_name.as_str(),
            "search_symbols"
                | "get_symbol_sources"
                | "scan_usages"
                | "get_summaries"
                | "read_file"
                | "grep_search"
                | "list_directory"
        )
    })
}

async fn build_train_bifrost_nudge(
    llm: &Arc<dyn LlmBackend>,
    turn: usize,
    messages: &[ChatMessage],
    tool_exchanges: &[ToolExchange],
    training_packet: Option<&TrainingPacket>,
    cancel: &CancellationToken,
    idle_timeout: Duration,
) -> Option<(String, TokenUsage)> {
    let packet = training_packet?;
    train_bifrost::compose_no_edit_nudge(
        llm,
        turn,
        messages,
        tool_exchanges,
        packet,
        cancel,
        idle_timeout,
    )
    .await
}

fn should_emit_no_edit_progress_nudge(
    permission_mode: PermissionMode,
    turn: usize,
    max_turns: usize,
    tool_exchanges: &[ToolExchange],
    nudge_count: usize,
) -> bool {
    if matches!(permission_mode, PermissionMode::ReadOnly) {
        return false;
    }
    if nudge_count >= 2 || has_successful_file_change(tool_exchanges) {
        return false;
    }
    let first_nudge_turn = (max_turns / 3).clamp(6, 10);
    let next_nudge_turn = first_nudge_turn + 4 * nudge_count;
    turn >= next_nudge_turn
}

fn exact_source_tool_count(tool_exchanges: &[ToolExchange]) -> usize {
    tool_exchanges
        .iter()
        .filter(|exchange| exchange.tool_name == "get_symbol_sources")
        .count()
}

#[cfg(test)]
fn maybe_text_navigation_gate(
    tool_name: &str,
    tool_exchanges: &[ToolExchange],
    tools: &[ToolDefinition],
    gate_count: u8,
) -> Option<String> {
    if gate_count >= 2 {
        return None;
    }
    if !is_text_navigation_tool(tool_name)
        || !has_tool(tools, "get_summaries")
        || !has_tool(tools, "scan_usages")
    {
        return None;
    }

    let text_navigation_count = tool_exchanges
        .iter()
        .filter(|exchange| is_text_navigation_tool(&exchange.tool_name))
        .count()
        + 1;
    match (gate_count, text_navigation_count) {
        (0, 4) => Some(
            "Navigation gate: you have used text/file navigation several times in this turn. \
             Before another read_file/grep_search call, choose one: call `get_summaries` for the \
             relevant module, package, class, API, or file glob if you are still orienting; call \
             `scan_usages` if you are looking for callers, references, or related tests for a known \
             symbol; or retry the text-navigation call if the needed context is already localized. \
             Do not call Bifrost ceremonially -- use it only if it answers the current context question."
                .to_string(),
        ),
        (1, 8) => Some(
            "Summary gate: you are still using text/file navigation after the earlier navigation \
             gate. If this is still orientation across files or modules, call `get_summaries` now \
             with the relevant file glob, module, class, or API target. If the remaining question is \
             already localized to exact lines, retry the text-navigation call."
                .to_string(),
        ),
        _ => None,
    }
}

/// Run the tool against the registry and format the result for the LLM.
/// Arg-parse failure is handled in the caller so it can render a Failed
/// card; this function only sees already-parsed inputs.
async fn execute_tool(
    registry: &ToolRegistry,
    tool_name: &str,
    args: Value,
    policy: SandboxPolicy,
    outside_sandbox_once: bool,
    sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
) -> ToolExecution {
    let result = registry
        .execute_with_sandbox_mode(tool_name, args, policy, outside_sandbox_once, sandbox_mode)
        .await;
    tool_result_to_execution(result)
}

fn tool_result_to_execution(result: crate::tools::ToolResult) -> ToolExecution {
    let (status_prefix, failed) = match result.status {
        ToolStatus::Success => ("", false),
        ToolStatus::RequestError => ("Error: ", true),
        ToolStatus::InternalError => ("Internal error: ", true),
    };
    let mut output = format!("{}{}", status_prefix, result.output);
    if output.len() > MAX_TOOL_RESULT_BYTES {
        // Truncate on a UTF-8 char boundary; otherwise an emoji or accented
        // byte sequence could leave the slice mid-codepoint.
        let mut cut = MAX_TOOL_RESULT_BYTES;
        while !output.is_char_boundary(cut) {
            cut -= 1;
        }
        output.truncate(cut);
        output.push_str("\n... output truncated");
    }
    ToolExecution { output, failed }
}

/// Dispatch the `task` meta-tool: look up the named subagent, build a
/// fresh `Vec<ChatMessage>` from its body + the caller-provided prompt,
/// and drive a nested `run()` to completion with notifications
/// suppressed. Only the subagent's final assistant text is returned to
/// the parent loop as the tool result.
///
/// The nested call shares:
///   * `llm`, `registry`, `model`, `reasoning_effort`: same backend, same
///     tool catalog (minus `task` itself at depth >= MAX_SUBAGENT_DEPTH),
///     same model and effort as the parent.
///   * `cancel`: a parent cancellation propagates to the subagent.
///   * `session_id` and `sessions`: the permission gate stays active
///     against the parent session's mode and always-allow set, so a
///     subagent cannot escape `readOnly` or skip a prompt the parent
///     would have to clear.
///
/// What is NOT shared:
///   * `messages`: the subagent gets a fresh transcript built from its
///     own system prompt + the dispatcher's `prompt` argument. It does
///     not see the parent conversation.
///   * Text/thought sinks: a no-op sink swallows the subagent's
///     streamed tokens. The parent only sees the final result string.
///   * Progress notifications: `NotificationMode::Silent` suppresses
///     per-tool `SessionUpdate`s. Permission prompts still fire when
///     the gate escalates.
#[allow(clippy::too_many_arguments)]
async fn execute_subagent(
    llm: &Arc<dyn LlmBackend>,
    registry: &ToolRegistry,
    model: &str,
    reasoning_effort: Option<&str>,
    _structured_output: Option<&StructuredOutputRequest>,
    args: &Value,
    max_turns: usize,
    idle_timeout: Duration,
    cancel: CancellationToken,
    spawned_cx: &SpawnedCx<'_>,
    session_id: &str,
    sessions: &SessionStore,
    depth: usize,
) -> (ToolExecution, TokenUsage) {
    let subagent_name = match args.get("subagent_type").and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => s,
        _ => {
            return (
                ToolExecution {
                    output: "Error: `task` requires a non-empty `subagent_type`.".to_string(),
                    failed: true,
                },
                TokenUsage::default(),
            );
        }
    };
    let prompt = match args.get("prompt").and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => s,
        _ => {
            return (
                ToolExecution {
                    output: "Error: `task` requires a non-empty `prompt`.".to_string(),
                    failed: true,
                },
                TokenUsage::default(),
            );
        }
    };

    // Snapshot the agent metadata under the registry lock, then drop the
    // guard before recursing into `run()` -- the nested call also
    // touches `registry.tool_definitions().await`, which takes the same
    // RwLock for read and would deadlock if we held it across the await.
    let meta = {
        let agents = registry.agents_snapshot().await;
        match agents.get(subagent_name) {
            Some(m) => m.clone(),
            None => {
                let available: Vec<&str> = agents.iter_sorted().map(|m| m.name.as_str()).collect();
                return (
                    ToolExecution {
                        output: format!(
                            "Error: unknown subagent '{subagent_name}'. Available: {}",
                            available.join(", ")
                        ),
                        failed: true,
                    },
                    TokenUsage::default(),
                );
            }
        }
    };

    let body = match crate::agents::read_agent_body(&meta.location) {
        Ok(b) => b,
        Err(e) => {
            return (
                ToolExecution {
                    output: format!("Error: failed to load subagent '{subagent_name}': {e}"),
                    failed: true,
                },
                TokenUsage::default(),
            );
        }
    };

    let system = format!(
        "{body}\n\n---\n\nYou are running as a subagent invoked via the `task` tool. \
         The parent agent will receive your final assistant message as your \
         result -- be self-contained and end with the answer."
    );
    let messages = vec![ChatMessage::system(system), ChatMessage::user(prompt)];

    // No-op sinks: the subagent's streamed tokens stay internal. We
    // collect the final string from `run()`'s return value.
    let noop_text: TextSink = Arc::new(Mutex::new(|_: &str| {}));
    let noop_thought: TextSink = Arc::new(Mutex::new(|_: &str| {}));

    // Cap the subagent's turn budget so a runaway delegation can't
    // burn the parent's entire allowance (which can be 200+ turns).
    // 25 is enough for a well-scoped focused task; if the subagent
    // needs more, that's a sign the parent should have done the work
    // itself or split it differently.
    let nested_max_turns = max_turns.min(MAX_SUBAGENT_TURNS);

    // `Box::pin` is required because `run` is recursive via this
    // function and Rust async fns can't be directly recursive (the
    // future type would have infinite size).
    let nested = Box::pin(run(
        llm,
        registry,
        model,
        reasoning_effort,
        None,
        messages,
        nested_max_turns,
        idle_timeout,
        cancel,
        noop_text,
        noop_thought,
        SpawnedCx::new(spawned_cx.cx()),
        session_id.to_string(),
        sessions.clone(),
        NotificationMode::Silent,
        depth,
    ))
    .await;

    let (text, _exchanges, nested_usage) = nested;
    let exec = if text.trim().is_empty() {
        ToolExecution {
            output: format!("Error: subagent '{subagent_name}' returned an empty response."),
            failed: true,
        }
    } else {
        ToolExecution {
            output: text,
            failed: false,
        }
    };
    (exec, nested_usage)
}

/// Send a `SessionNotification` and log on failure -- there is nothing
/// useful we can do if the channel to the client is broken.
/// Emit a `SessionUpdate` notification only when `mode == Live`. Used so
/// subagent runs (`mode == Silent`) don't push their internal tool-call
/// progress cards back to the ACP client -- the parent only sees the
/// subagent's final answer as the `task` tool's result.
fn maybe_send_session_update(
    mode: NotificationMode,
    cx: &ConnectionTo<Client>,
    session_id: &str,
    update: SessionUpdate,
) {
    if mode == NotificationMode::Live {
        send_session_update(cx, session_id, update);
    }
}

fn send_session_update(cx: &ConnectionTo<Client>, session_id: &str, update: SessionUpdate) {
    let notification = SessionNotification::new(session_id.to_string(), update);
    if let Err(e) = cx.send_notification(notification) {
        tracing::warn!("failed to send session update: {e}");
    }
}

/// Read the existing file content for a file-editing call so the diff
/// card can show before/after text.
///
/// Returns `Some(Some(text))` when we read the prior content, `Some(None)`
/// when the file is new (per the ACP `Diff.old_text` schema, `None` is
/// the "no prior content" sentinel), and `None` when prior content is
/// unavailable -- e.g. binary file, unreadable, or path can't be resolved
/// against cwd. The outer `None` tells the caller to fall back to text
/// content for the card.
fn capture_pre_write_text(cwd: &Path, parsed_input: &Value) -> Option<Option<String>> {
    let path = parsed_input
        .get("file_path")
        .or_else(|| parsed_input.get("path"))
        .and_then(Value::as_str)?;
    let resolved = safe_resolve_for_write(cwd, path).ok()?;
    if !resolved.exists() {
        return Some(None);
    }
    match std::fs::read_to_string(&resolved) {
        Ok(text) => Some(Some(text)),
        Err(_) => None,
    }
}

/// Assemble a `Diff` block for a successful write/edit call from the parsed
/// args plus the captured prior content. Returns `None` if we couldn't
/// pull the path/content (in which case the caller falls back to text).
fn build_editing_diff(
    tool_name: &str,
    parsed_input: &Value,
    prior: Option<String>,
) -> Option<Diff> {
    let path = parsed_input
        .get("file_path")
        .or_else(|| parsed_input.get("path"))
        .and_then(Value::as_str)?;
    let new_text = match tool_name {
        "write_file" => parsed_input
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        "edit" => {
            let prior_text = prior.as_ref()?;
            let old = parsed_input.get("old_string").and_then(Value::as_str)?;
            let new = parsed_input.get("new_string").and_then(Value::as_str)?;
            let replace_all = parsed_input
                .get("replace_all")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if replace_all {
                prior_text.replace(old, new)
            } else {
                prior_text.replacen(old, new, 1)
            }
        }
        _ => return None,
    };
    let mut diff = Diff::new(PathBuf::from(path), new_text);
    diff.old_text = prior;
    Some(diff)
}

fn ordered_tool_call_indices(
    calls: &[ToolCall],
    is_bifrost_tool: impl Fn(&str) -> bool,
) -> Vec<usize> {
    let mut builtin_or_other = Vec::new();
    let mut bifrost = Vec::new();
    for (index, call) in calls.iter().enumerate() {
        if is_bifrost_tool(&call.function.name) {
            bifrost.push(index);
        } else {
            builtin_or_other.push(index);
        }
    }
    builtin_or_other.extend(bifrost);
    builtin_or_other
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_client::{FunctionCall, FunctionDef};
    use futures::future::{BoxFuture, FutureExt};
    use std::sync::atomic::AtomicUsize;

    fn tool_def_for_test(name: &str) -> ToolDefinition {
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: name.to_string(),
                description: String::new(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            },
        }
    }

    fn tool_call_for_test(name: &str) -> ToolCall {
        ToolCall {
            id: format!("call-{name}"),
            r#type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: "{}".to_string(),
            },
        }
    }

    fn exchange_for_test(tool_name: &str) -> ToolExchange {
        ToolExchange {
            call_id: format!("call-{tool_name}"),
            tool_name: tool_name.to_string(),
            arguments: "{}".to_string(),
            result: String::new(),
        }
    }

    fn ordered_names_for_test(calls: &[ToolCall], bifrost_names: &[&str]) -> Vec<String> {
        ordered_tool_call_indices(calls, |name| bifrost_names.contains(&name))
            .into_iter()
            .map(|index| calls[index].function.name.clone())
            .collect()
    }

    struct RetryBackend {
        attempts: Arc<AtomicUsize>,
        emit_before_error: bool,
        first_error: &'static str,
    }

    impl LlmBackend for RetryBackend {
        fn list_models(&self) -> BoxFuture<'_, anyhow::Result<Vec<String>>> {
            async { Ok(Vec::new()) }.boxed()
        }

        fn stream_chat(
            &self,
            mut request: StreamChatRequest,
        ) -> BoxFuture<'_, anyhow::Result<LlmResponse>> {
            let attempts = self.attempts.clone();
            let emit_before_error = self.emit_before_error;
            let first_error = self.first_error;
            async move {
                let attempt = attempts.fetch_add(1, Ordering::SeqCst) + 1;
                if attempt == 1 {
                    if emit_before_error {
                        (request.on_token)("partial");
                    }
                    anyhow::bail!(first_error);
                }
                Ok(LlmResponse::Text {
                    text: "ok".to_string(),
                    usage: TokenUsage::default(),
                })
            }
            .boxed()
        }
    }

    fn text_sink_for_test(buffer: Arc<Mutex<String>>) -> TextSink {
        Arc::new(Mutex::new(move |text: &str| {
            buffer.lock().unwrap().push_str(text);
        }))
    }

    fn decide(
        mode: PermissionMode,
        kind: ToolKind,
        tool_name: &str,
        allowed: bool,
        shell_auto_allow: bool,
    ) -> PureGateDecision {
        pure_gate_decision(mode, kind, tool_name, allowed, shell_auto_allow)
    }

    #[test]
    fn tool_call_order_runs_non_bifrost_before_bifrost() {
        let calls = vec![
            tool_call_for_test("search_symbols"),
            tool_call_for_test("read_file"),
            tool_call_for_test("get_summaries"),
            tool_call_for_test("run_shell_command"),
        ];

        let names = ordered_names_for_test(&calls, &["search_symbols", "get_summaries"]);

        assert_eq!(
            names,
            vec![
                "read_file",
                "run_shell_command",
                "search_symbols",
                "get_summaries"
            ]
        );
    }

    #[test]
    fn tool_call_order_preserves_relative_order_within_groups() {
        let calls = vec![
            tool_call_for_test("get_summaries"),
            tool_call_for_test("grep_search"),
            tool_call_for_test("search_symbols"),
            tool_call_for_test("read_file"),
            tool_call_for_test("scan_usages"),
        ];

        let names =
            ordered_names_for_test(&calls, &["get_summaries", "search_symbols", "scan_usages"]);

        assert_eq!(
            names,
            vec![
                "grep_search",
                "read_file",
                "get_summaries",
                "search_symbols",
                "scan_usages"
            ]
        );
    }

    #[test]
    fn tool_call_order_leaves_non_bifrost_batch_unchanged() {
        let calls = vec![
            tool_call_for_test("think"),
            tool_call_for_test("read_file"),
            tool_call_for_test("run_shell_command"),
        ];

        let names = ordered_names_for_test(&calls, &["search_symbols"]);

        assert_eq!(names, vec!["think", "read_file", "run_shell_command"]);
    }

    #[test]
    fn advertised_tool_names_match_current_request_catalog() {
        let tools = vec![tool_def_for_test("think"), tool_def_for_test("edit")];
        let names = advertised_tool_names(Some(&tools));

        assert!(names.contains("think"));
        assert!(names.contains("edit"));
        assert!(!names.contains("run_shell_command"));
    }

    #[test]
    fn hidden_tool_calls_are_marked_unavailable() {
        let message = tool_unavailable_message("run_shell_command");

        assert!(message.contains("run_shell_command"));
        assert!(message.contains("unavailable in the current tool catalog"));
    }

    #[test]
    fn train_bifrost_post_edit_policy_only_adds_shell() {
        let initial = train_bifrost_initial_builtin_tools();
        let post_edit = train_bifrost_post_edit_builtin_tools();

        assert!(initial.contains("think"));
        assert!(initial.contains("edit"));
        assert!(initial.contains("write_file"));
        assert!(initial.contains("list_directory"));
        assert!(!initial.contains("read_file"));
        assert!(!initial.contains("grep_search"));
        assert!(!initial.contains("run_shell_command"));

        assert!(post_edit.contains("run_shell_command"));
        assert!(!post_edit.contains("read_file"));
        assert!(!post_edit.contains("grep_search"));
    }

    #[test]
    fn train_bifrost_policy_is_env_controlled() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();

        let _scope = crate::openrouter_auth::test_support::EnvScope::remove(TRAIN_BIFROST_ENV);
        assert!(!train_bifrost_enabled());
        drop(_scope);

        let _scope = crate::openrouter_auth::test_support::EnvScope::set(TRAIN_BIFROST_ENV, "1");
        assert!(train_bifrost_enabled());
        drop(_scope);

        let _scope =
            crate::openrouter_auth::test_support::EnvScope::set(TRAIN_BIFROST_ENV, "false");
        assert!(!train_bifrost_enabled());
    }

    #[tokio::test]
    async fn stream_chat_retries_transient_stream_error_before_output() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let backend: Arc<dyn LlmBackend> = Arc::new(RetryBackend {
            attempts: attempts.clone(),
            emit_before_error: false,
            first_error: "Codex stream read error: simulated disconnect",
        });
        let output = Arc::new(Mutex::new(String::new()));
        let thoughts = Arc::new(Mutex::new(String::new()));

        let response = stream_chat_with_transient_retry(
            &backend,
            0,
            "codex::test",
            &[ChatMessage::user("hello")],
            None,
            None,
            None,
            &text_sink_for_test(output.clone()),
            &text_sink_for_test(thoughts),
            &CancellationToken::new(),
            Duration::from_secs(30),
        )
        .await
        .expect("retry should recover");

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert!(matches!(response, LlmResponse::Text { text, .. } if text == "ok"));
        assert_eq!(output.lock().unwrap().as_str(), "");
    }

    #[tokio::test]
    async fn stream_chat_does_not_retry_after_partial_output() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let backend: Arc<dyn LlmBackend> = Arc::new(RetryBackend {
            attempts: attempts.clone(),
            emit_before_error: true,
            first_error: "Codex stream read error: simulated disconnect",
        });
        let output = Arc::new(Mutex::new(String::new()));
        let thoughts = Arc::new(Mutex::new(String::new()));

        let error = stream_chat_with_transient_retry(
            &backend,
            0,
            "codex::test",
            &[ChatMessage::user("hello")],
            None,
            None,
            None,
            &text_sink_for_test(output.clone()),
            &text_sink_for_test(thoughts),
            &CancellationToken::new(),
            Duration::from_secs(30),
        )
        .await
        .expect_err("partial output makes retry unsafe");

        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert!(format!("{error:#}").contains("Codex stream read error"));
        assert_eq!(output.lock().unwrap().as_str(), "partial");
    }

    #[tokio::test]
    async fn stream_chat_retries_server_overloaded_before_output() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let backend: Arc<dyn LlmBackend> = Arc::new(RetryBackend {
            attempts: attempts.clone(),
            emit_before_error: false,
            first_error: "Codex Responses stream failed: server_is_overloaded: Our servers are currently overloaded. Please try again later.",
        });
        let output = Arc::new(Mutex::new(String::new()));
        let thoughts = Arc::new(Mutex::new(String::new()));

        let response = stream_chat_with_transient_retry(
            &backend,
            0,
            "codex::test",
            &[ChatMessage::user("hello")],
            None,
            None,
            None,
            &text_sink_for_test(output.clone()),
            &text_sink_for_test(thoughts),
            &CancellationToken::new(),
            Duration::from_secs(30),
        )
        .await
        .expect("overload should be retried before output");

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert!(matches!(response, LlmResponse::Text { text, .. } if text == "ok"));
        assert_eq!(output.lock().unwrap().as_str(), "");
    }

    #[tokio::test]
    async fn stream_chat_retries_server_error_before_output() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let backend: Arc<dyn LlmBackend> = Arc::new(RetryBackend {
            attempts: attempts.clone(),
            emit_before_error: false,
            first_error: "Responses stream failed: server_error: The server had an error while processing your request.",
        });
        let output = Arc::new(Mutex::new(String::new()));
        let thoughts = Arc::new(Mutex::new(String::new()));

        let response = stream_chat_with_transient_retry(
            &backend,
            0,
            "bedrock::openai.gpt-5.4",
            &[ChatMessage::user("hello")],
            None,
            None,
            None,
            &text_sink_for_test(output.clone()),
            &text_sink_for_test(thoughts),
            &CancellationToken::new(),
            Duration::from_secs(30),
        )
        .await
        .expect("server_error should be retried before output");

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert!(matches!(response, LlmResponse::Text { text, .. } if text == "ok"));
        assert_eq!(output.lock().unwrap().as_str(), "");
    }

    #[tokio::test]
    async fn stream_chat_retries_rate_limit_before_output() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let backend: Arc<dyn LlmBackend> = Arc::new(RetryBackend {
            attempts: attempts.clone(),
            emit_before_error: false,
            first_error: "Responses stream failed: rate_limit_exceeded: slow down",
        });
        let output = Arc::new(Mutex::new(String::new()));
        let thoughts = Arc::new(Mutex::new(String::new()));

        let response = stream_chat_with_transient_retry(
            &backend,
            0,
            "bedrock::openai.gpt-5.4",
            &[ChatMessage::user("hello")],
            None,
            None,
            None,
            &text_sink_for_test(output.clone()),
            &text_sink_for_test(thoughts),
            &CancellationToken::new(),
            Duration::from_secs(30),
        )
        .await
        .expect("rate_limit_exceeded should be retried before output");

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert!(matches!(response, LlmResponse::Text { text, .. } if text == "ok"));
        assert_eq!(output.lock().unwrap().as_str(), "");
    }

    #[test]
    fn text_navigation_gate_triggers_on_fourth_text_navigation_call() {
        let tools = vec![
            tool_def_for_test("read_file"),
            tool_def_for_test("grep_search"),
            tool_def_for_test("get_summaries"),
            tool_def_for_test("scan_usages"),
        ];
        let prior = vec![
            exchange_for_test("read_file"),
            exchange_for_test("grep_search"),
            exchange_for_test("read_file"),
        ];

        let output = maybe_text_navigation_gate("read_file", &prior, &tools, 0)
            .expect("fourth text-navigation call should trip the gate");

        assert!(output.contains("Navigation gate:"));
        assert!(output.contains("get_summaries"));
        assert!(output.contains("scan_usages"));
    }

    #[test]
    fn text_navigation_gate_does_not_repeat_after_trigger_point() {
        let tools = vec![
            tool_def_for_test("read_file"),
            tool_def_for_test("grep_search"),
            tool_def_for_test("get_summaries"),
            tool_def_for_test("scan_usages"),
        ];
        let prior = vec![
            exchange_for_test("read_file"),
            exchange_for_test("grep_search"),
            exchange_for_test("read_file"),
        ];

        let output = maybe_text_navigation_gate("grep_search", &prior, &tools, 2);

        assert!(output.is_none());
    }

    #[test]
    fn text_navigation_gate_triggers_summary_followup_after_persistent_navigation() {
        let tools = vec![
            tool_def_for_test("read_file"),
            tool_def_for_test("grep_search"),
            tool_def_for_test("get_summaries"),
            tool_def_for_test("scan_usages"),
        ];
        let prior = vec![
            exchange_for_test("read_file"),
            exchange_for_test("grep_search"),
            exchange_for_test("read_file"),
            exchange_for_test("grep_search"),
            exchange_for_test("read_file"),
            exchange_for_test("grep_search"),
            exchange_for_test("read_file"),
        ];

        let output = maybe_text_navigation_gate("grep_search", &prior, &tools, 1)
            .expect("eighth text-navigation call should trip the summary follow-up");

        assert!(output.contains("Summary gate:"));
        assert!(output.contains("get_summaries"));
    }

    #[test]
    fn text_navigation_gate_requires_bifrost_tools() {
        let tools = vec![
            tool_def_for_test("read_file"),
            tool_def_for_test("grep_search"),
        ];
        let prior = vec![
            exchange_for_test("read_file"),
            exchange_for_test("grep_search"),
            exchange_for_test("read_file"),
        ];

        let output = maybe_text_navigation_gate("read_file", &prior, &tools, 0);

        assert!(output.is_none());
    }

    #[test]
    fn no_edit_final_guard_rejects_navigation_only_final_before_last_turn() {
        let prior = vec![exchange_for_test("search_symbols")];

        assert!(should_reject_no_edit_final_answer(
            PermissionMode::Default,
            3,
            10,
            &prior
        ));
        assert!(!should_reject_no_edit_final_answer(
            PermissionMode::ReadOnly,
            3,
            10,
            &prior
        ));
    }

    #[test]
    fn no_edit_final_guard_allows_after_successful_edit() {
        let prior = vec![ToolExchange {
            call_id: "edit".to_string(),
            tool_name: "edit".to_string(),
            arguments: "{}".to_string(),
            result: "Edited 'src/lib.rs' (1 replacement)".to_string(),
        }];

        assert!(!should_reject_no_edit_final_answer(
            PermissionMode::Default,
            3,
            10,
            &prior
        ));
        assert!(has_successful_file_change(&prior));
    }

    #[test]
    fn no_edit_final_guard_does_not_reject_on_last_turn() {
        let prior = vec![exchange_for_test("search_symbols")];

        assert!(!should_reject_no_edit_final_answer(
            PermissionMode::Default,
            9,
            10,
            &prior
        ));
    }

    #[test]
    fn no_edit_progress_nudge_triggers_after_enough_context() {
        let prior = vec![
            exchange_for_test("search_symbols"),
            exchange_for_test("get_symbol_sources"),
            exchange_for_test("scan_usages"),
            exchange_for_test("get_summaries"),
            exchange_for_test("read_file"),
        ];

        assert!(should_emit_no_edit_progress_nudge(
            PermissionMode::Default,
            8,
            25,
            &prior,
            0
        ));
        assert!(!should_emit_no_edit_progress_nudge(
            PermissionMode::ReadOnly,
            8,
            25,
            &prior,
            0
        ));
        assert!(!should_emit_no_edit_progress_nudge(
            PermissionMode::Default,
            7,
            25,
            &prior,
            0
        ));
        assert!(!should_emit_no_edit_progress_nudge(
            PermissionMode::Default,
            8,
            25,
            &prior,
            2
        ));
    }

    #[test]
    fn no_edit_progress_nudge_uses_turn_threshold_without_context_gate() {
        let prior = vec![
            exchange_for_test("search_symbols"),
            exchange_for_test("scan_usages"),
            exchange_for_test("get_summaries"),
            exchange_for_test("search_symbols"),
            exchange_for_test("read_file"),
        ];

        assert!(!should_emit_no_edit_progress_nudge(
            PermissionMode::Default,
            7,
            25,
            &prior,
            0
        ));
        assert!(should_emit_no_edit_progress_nudge(
            PermissionMode::Default,
            8,
            25,
            &prior,
            0
        ));
    }

    #[test]
    fn no_edit_progress_nudge_allows_after_successful_edit() {
        let prior = vec![
            exchange_for_test("search_symbols"),
            exchange_for_test("get_symbol_sources"),
            exchange_for_test("scan_usages"),
            exchange_for_test("get_summaries"),
            exchange_for_test("search_symbols"),
            exchange_for_test("get_symbol_sources"),
            exchange_for_test("read_file"),
            ToolExchange {
                call_id: "edit".to_string(),
                tool_name: "edit".to_string(),
                arguments: "{}".to_string(),
                result: "Edited 'src/lib.rs' (1 replacement)".to_string(),
            },
        ];

        assert!(!should_emit_no_edit_progress_nudge(
            PermissionMode::Default,
            12,
            25,
            &prior,
            0
        ));
    }

    #[test]
    fn bypass_allows_everything_without_consulting_always_allow() {
        for kind in [
            ToolKind::Read,
            ToolKind::Edit,
            ToolKind::Delete,
            ToolKind::Move,
            ToolKind::Execute,
            ToolKind::Other,
        ] {
            assert_eq!(
                decide(
                    PermissionMode::BypassPermissions,
                    kind,
                    "anything",
                    false,
                    false
                ),
                PureGateDecision::Allow,
                "bypass should allow {:?} regardless of always-allow",
                kind
            );
        }
    }

    #[test]
    fn read_only_allows_only_info_kinds() {
        for kind in [
            ToolKind::Read,
            ToolKind::Search,
            ToolKind::Think,
            ToolKind::Fetch,
        ] {
            assert_eq!(
                decide(PermissionMode::ReadOnly, kind, "anything", false, false),
                PureGateDecision::Allow,
                "read-only should allow info kind {:?}",
                kind
            );
        }
    }

    #[test]
    fn read_only_rejects_mutating_kinds_even_when_always_allowed() {
        // This is the regression we just fixed: ReadOnly must override
        // a prior "Always allow" for any non-info kind, including `Other`.
        for kind in [
            ToolKind::Edit,
            ToolKind::Delete,
            ToolKind::Move,
            ToolKind::Execute,
            ToolKind::Other,
        ] {
            assert!(
                matches!(
                    decide(PermissionMode::ReadOnly, kind, "any", true, false),
                    PureGateDecision::Reject(_)
                ),
                "read-only should reject {:?} even when always-allowed",
                kind
            );
        }
    }

    #[tokio::test]
    async fn read_only_preflight_rejects_mutation_tools_before_execution() {
        let cwd = tempfile::tempdir().expect("temp cwd");
        let store = SessionStore::new("m".to_string());
        let session = store.create_session(cwd.path().to_path_buf()).await;
        assert!(
            store
                .set_permission_mode(&session.id, PermissionMode::ReadOnly)
                .await
        );

        let cases = [
            (
                "write_file",
                ToolRegistry::tool_kind("write_file"),
                serde_json::json!({"file_path": "app.js", "content": "x"}),
            ),
            (
                "edit",
                ToolRegistry::tool_kind("edit"),
                serde_json::json!({"file_path": "app.js", "old_string": "x", "new_string": "y"}),
            ),
            (
                "run_shell_command",
                ToolRegistry::tool_kind("run_shell_command"),
                serde_json::json!({"command": "touch app.js"}),
            ),
            (
                "task",
                ToolRegistry::tool_kind("task"),
                serde_json::json!({"subagent_type": "reviewer", "prompt": "edit app.js"}),
            ),
        ];

        for (tool_name, kind, input) in cases {
            let rejection = deterministic_gate_rejection(
                &store,
                &session.id,
                tool_name,
                kind,
                &input,
                cwd.path(),
            )
            .await
            .unwrap_or_else(|| panic!("{tool_name} should be rejected before execution"));

            assert!(
                rejection.contains("read-only mode forbids"),
                "unexpected rejection for {tool_name}: {rejection}"
            );
        }
    }

    #[tokio::test]
    async fn preflight_does_not_block_promptable_or_read_only_safe_tools() {
        let cwd = tempfile::tempdir().expect("temp cwd");
        let store = SessionStore::new("m".to_string());
        let session = store.create_session(cwd.path().to_path_buf()).await;

        let default_edit = deterministic_gate_rejection(
            &store,
            &session.id,
            "write_file",
            ToolRegistry::tool_kind("write_file"),
            &serde_json::json!({"file_path": "app.js", "content": "x"}),
            cwd.path(),
        )
        .await;
        assert!(
            default_edit.is_none(),
            "default edit should proceed to the permission prompt"
        );

        assert!(
            store
                .set_permission_mode(&session.id, PermissionMode::ReadOnly)
                .await
        );
        let read = deterministic_gate_rejection(
            &store,
            &session.id,
            "read_file",
            ToolRegistry::tool_kind("read_file"),
            &serde_json::json!({"file_path": "app.js"}),
            cwd.path(),
        )
        .await;
        assert!(read.is_none(), "read-only should allow read tools");
    }

    #[test]
    fn blocked_tool_call_sequence_has_failed_card_and_terminal_update() {
        let reason = "Tool use denied: read-only mode forbids edits";
        let (card, update) = blocked_tool_call_updates(
            "call-write",
            "write_file",
            ToolRegistry::tool_kind("write_file"),
            &serde_json::json!({"file_path": "app.js", "content": "x"}),
            reason,
        );

        assert_eq!(card.tool_call_id.0.as_ref(), "call-write");
        assert_eq!(card.title, "Blocked Writing file");
        assert_eq!(card.status, ToolCallStatus::Failed);
        assert_eq!(update.tool_call_id.0.as_ref(), "call-write");
        assert_eq!(update.fields.status, Some(ToolCallStatus::Failed));
        assert!(card.raw_input.is_some());
        assert_eq!(
            update.fields.raw_output,
            Some(Value::String(reason.to_string()))
        );
    }

    #[test]
    fn default_auto_allows_info_kinds_without_always_allow() {
        for kind in [
            ToolKind::Read,
            ToolKind::Search,
            ToolKind::Think,
            ToolKind::Fetch,
        ] {
            assert_eq!(
                decide(PermissionMode::Default, kind, "anything", false, false),
                PureGateDecision::Allow
            );
        }
    }

    #[test]
    fn default_prompts_for_edit_without_always_allow() {
        assert_eq!(
            decide(
                PermissionMode::Default,
                ToolKind::Edit,
                "write_file",
                false,
                false
            ),
            PureGateDecision::Prompt
        );
    }

    #[test]
    fn default_uses_always_allow_for_edit() {
        assert_eq!(
            decide(
                PermissionMode::Default,
                ToolKind::Edit,
                "write_file",
                true,
                false
            ),
            PureGateDecision::Allow
        );
    }

    #[test]
    fn accept_edits_auto_allows_edit_without_prior_approval() {
        assert_eq!(
            decide(
                PermissionMode::AcceptEdits,
                ToolKind::Edit,
                "write_file",
                false,
                false
            ),
            PureGateDecision::Allow
        );
    }

    #[test]
    fn accept_edits_still_prompts_for_execute() {
        assert_eq!(
            decide(
                PermissionMode::AcceptEdits,
                ToolKind::Execute,
                "run_shell_command",
                false,
                false
            ),
            PureGateDecision::Prompt
        );
    }

    #[test]
    fn shell_command_uses_scoped_always_allow() {
        // The cache key is command-scoped for shell calls, so the pure gate
        // may trust a positive lookup without granting every shell command.
        for mode in [PermissionMode::Default, PermissionMode::AcceptEdits] {
            assert_eq!(
                decide(mode, ToolKind::Execute, "run_shell_command", true, false),
                PureGateDecision::Allow,
                "run_shell_command should honor scoped approval in {:?}",
                mode
            );
        }
    }

    #[test]
    fn default_auto_allows_conservative_sandboxed_shell_commands() {
        assert_eq!(
            decide(
                PermissionMode::Default,
                ToolKind::Execute,
                "run_shell_command",
                false,
                true
            ),
            PureGateDecision::Allow
        );
    }

    #[test]
    fn accept_edits_auto_allows_conservative_sandboxed_shell_commands() {
        assert_eq!(
            decide(
                PermissionMode::AcceptEdits,
                ToolKind::Execute,
                "run_shell_command",
                false,
                true
            ),
            PureGateDecision::Allow
        );
    }

    #[test]
    fn shell_auto_allow_does_not_bypass_read_only_mode() {
        assert!(matches!(
            decide(
                PermissionMode::ReadOnly,
                ToolKind::Execute,
                "run_shell_command",
                false,
                true
            ),
            PureGateDecision::Reject(_)
        ));
    }

    #[test]
    fn shell_auto_allow_does_not_bypass_non_shell_execute_tools() {
        assert_eq!(
            decide(
                PermissionMode::Default,
                ToolKind::Execute,
                "task",
                false,
                true
            ),
            PureGateDecision::Prompt
        );
    }

    #[test]
    fn safe_shell_classifier_accepts_basic_read_only_commands() {
        assert!(is_auto_approvable_sandboxed_shell_command("pwd"));
        assert!(is_auto_approvable_sandboxed_shell_command("git status"));
        assert!(is_auto_approvable_sandboxed_shell_command(
            "rg PermissionMode src"
        ));
    }

    #[test]
    fn safe_shell_classifier_rejects_writes_and_shell_metacharacters() {
        assert!(!is_auto_approvable_sandboxed_shell_command(
            "sed -i 's/a/b/' src/main.rs"
        ));
        assert!(!is_auto_approvable_sandboxed_shell_command(
            "python3 -c 'print(1)'"
        ));
        assert!(!is_auto_approvable_sandboxed_shell_command("pwd && ls"));
    }

    #[test]
    fn should_auto_allow_shell_command_requires_os_sandboxed_shell() {
        use crate::sandbox_backend::SandboxMode;

        assert!(should_auto_allow_shell_command(
            &serde_json::json!({"command": "pwd"}),
            PermissionMode::Default,
            Some(SandboxMode::Os),
            true,
        ));
        assert!(!should_auto_allow_shell_command(
            &serde_json::json!({"command": "pwd"}),
            PermissionMode::Default,
            Some(SandboxMode::Wasm),
            true,
        ));
        assert!(!should_auto_allow_shell_command(
            &serde_json::json!({"command": "pwd"}),
            PermissionMode::Default,
            Some(SandboxMode::Os),
            false,
        ));
        assert!(!should_auto_allow_shell_command(
            &serde_json::json!({"command": "pwd"}),
            PermissionMode::ReadOnly,
            Some(SandboxMode::Os),
            true,
        ));
    }

    #[test]
    fn should_auto_allow_shell_command_rejects_missing_or_unsupported_commands() {
        use crate::sandbox_backend::SandboxMode;

        assert!(!should_auto_allow_shell_command(
            &serde_json::json!({}),
            PermissionMode::Default,
            Some(SandboxMode::Os),
            true,
        ));
        assert!(!should_auto_allow_shell_command(
            &serde_json::json!({"command": 7}),
            PermissionMode::Default,
            Some(SandboxMode::Os),
            true,
        ));
        assert!(!should_auto_allow_shell_command(
            &serde_json::json!({"command": ""}),
            PermissionMode::Default,
            Some(SandboxMode::Os),
            true,
        ));
        assert!(!should_auto_allow_shell_command(
            &serde_json::json!({"command": "touch /tmp/x"}),
            PermissionMode::Default,
            Some(SandboxMode::Os),
            true,
        ));
    }

    #[test]
    fn tokenizer_rejects_untrusted_shell_forms() {
        assert!(tokenize_simple_shell_command("git status").is_some());
        assert!(tokenize_simple_shell_command("grep \"foo bar\" README.md").is_some());
        assert!(tokenize_simple_shell_command("pwd && ls").is_none());
        assert!(tokenize_simple_shell_command("echo $HOME").is_none());
        assert!(tokenize_simple_shell_command("python3 -c 'print(1)' | cat").is_none());
    }

    #[test]
    fn shell_permission_prompt_includes_explicit_outside_sandbox_choice() {
        let options = permission_options("run_shell_command", true);
        let labels: Vec<_> = options
            .iter()
            .map(|option| (option.option_id.0.as_ref(), option.name.as_str()))
            .collect();

        assert_eq!(
            labels,
            vec![
                ("allow", "Allow in sandbox"),
                ("allow_always", "Always allow this command in sandbox"),
                ("allow_outside_sandbox", "Run outside sandbox once"),
                ("reject", "Reject"),
            ]
        );
    }

    #[test]
    fn shell_permission_prompt_omits_sandbox_language_when_disabled() {
        let options = permission_options("run_shell_command", false);
        let labels: Vec<_> = options
            .iter()
            .map(|option| (option.option_id.0.as_ref(), option.name.as_str()))
            .collect();

        assert_eq!(
            labels,
            vec![
                ("allow", "Allow"),
                ("allow_always", "Always allow this command"),
                ("reject", "Reject"),
            ]
        );
    }

    #[test]
    fn shell_allow_always_choice_maps_to_sandboxed_session_approval() {
        let grant = permission_grant_for_selection("run_shell_command", "allow_always", true)
            .expect("shell sticky sandbox approval should be accepted");

        assert_eq!(
            grant,
            PermissionGrant {
                allow_always: true,
                sandbox_policy_override: None,
            }
        );
    }

    #[test]
    fn shell_always_allow_key_is_command_and_cwd_scoped() {
        let cwd = Path::new("/tmp/project");
        let first = always_allow_key(
            "run_shell_command",
            &serde_json::json!({"command": "cargo test", "timeout": 60}),
            cwd,
            true,
        );
        let same_without_timeout = always_allow_key(
            "run_shell_command",
            &serde_json::json!({"command": "cargo test"}),
            cwd,
            true,
        );
        let different_command = always_allow_key(
            "run_shell_command",
            &serde_json::json!({"command": "cargo check"}),
            cwd,
            true,
        );
        let different_cwd = always_allow_key(
            "run_shell_command",
            &serde_json::json!({"command": "cargo test"}),
            Path::new("/tmp/other"),
            true,
        );
        let different_sandbox_mode = always_allow_key(
            "run_shell_command",
            &serde_json::json!({"command": "cargo test"}),
            cwd,
            false,
        );

        assert_eq!(first, same_without_timeout);
        assert_ne!(first, different_command);
        assert_ne!(first, different_cwd);
        assert_ne!(first, different_sandbox_mode);
    }

    #[test]
    fn shell_repo_always_allow_rule_uses_first_three_tokens() {
        let key = shell_always_allow_rule_key(
            &serde_json::json!({"command": "cargo test --workspace --lib"}),
            true,
        )
        .expect("shell rule key");

        assert_eq!(
            key,
            serde_json::json!({
                "tool": "run_shell_command",
                "rule": "prefix",
                "argvPrefix": ["cargo", "test", "--workspace"],
                "shellSandboxed": true,
            })
            .to_string()
        );
    }

    #[test]
    fn shell_outside_sandbox_choice_maps_to_policy_override() {
        let grant =
            permission_grant_for_selection("run_shell_command", "allow_outside_sandbox", true)
                .expect("shell override should be accepted");

        assert_eq!(
            grant,
            PermissionGrant {
                allow_always: false,
                sandbox_policy_override: Some(SandboxPolicy::None),
            }
        );
    }

    #[test]
    fn shell_outside_sandbox_choice_is_rejected_when_shell_sandbox_disabled() {
        let err =
            permission_grant_for_selection("run_shell_command", "allow_outside_sandbox", false)
                .expect_err(
                    "outside-sandbox option is not valid when shell sandboxing is disabled",
                );
        assert!(err.contains("unknown option"), "got: {err}");
    }

    #[test]
    fn shell_outside_sandbox_choice_is_dropped_if_session_is_missing() {
        assert_eq!(
            resolve_execution_policy(None, None, Some(SandboxPolicy::None)),
            (SandboxPolicy::ReadOnly, false)
        );
    }

    #[test]
    fn shell_outside_sandbox_choice_is_kept_when_session_is_present() {
        assert_eq!(
            resolve_execution_policy(
                Some(PermissionMode::Default),
                None,
                Some(SandboxPolicy::None)
            ),
            (SandboxPolicy::None, true)
        );
    }

    #[test]
    fn sandbox_mode_off_collapses_policy_to_none() {
        use crate::sandbox_backend::SandboxMode;
        // Without an override, `sandbox_mode=Some(Off)` returns `None`
        // regardless of the permission mode -- the per-call prompt still
        // fires upstream, but the OS sandbox is skipped.
        assert_eq!(
            resolve_execution_policy(Some(PermissionMode::Default), Some(SandboxMode::Off), None),
            (SandboxPolicy::None, false)
        );
        assert_eq!(
            resolve_execution_policy(
                Some(PermissionMode::AcceptEdits),
                Some(SandboxMode::Off),
                None
            ),
            (SandboxPolicy::None, false)
        );
        assert_eq!(
            resolve_execution_policy(Some(PermissionMode::ReadOnly), Some(SandboxMode::Off), None),
            (SandboxPolicy::None, false)
        );
    }

    #[test]
    fn sandbox_mode_is_ignored_when_override_present() {
        use crate::sandbox_backend::SandboxMode;
        // A per-call override (the "Allow outside sandbox" choice) is
        // narrower than the session-wide flag, so it wins -- the override
        // path already carries `outside_sandbox_once = true`.
        assert_eq!(
            resolve_execution_policy(
                Some(PermissionMode::Default),
                Some(SandboxMode::Off),
                Some(SandboxPolicy::WorkspaceWrite)
            ),
            (SandboxPolicy::WorkspaceWrite, true)
        );
    }

    #[test]
    fn non_shell_permission_prompt_keeps_sticky_allow_and_no_override() {
        let options = permission_options("write_file", false);
        let labels: Vec<_> = options
            .iter()
            .map(|option| (option.option_id.0.as_ref(), option.name.as_str()))
            .collect();

        assert_eq!(
            labels,
            vec![
                ("allow_always", "Always allow write_file"),
                ("allow", "Allow"),
                ("reject", "Reject"),
            ]
        );

        let grant = permission_grant_for_selection("write_file", "allow_always", false)
            .expect("non-shell sticky allow should still be accepted");
        assert_eq!(
            grant,
            PermissionGrant {
                allow_always: true,
                sandbox_policy_override: None,
            }
        );
    }

    #[test]
    fn permission_mode_round_trip() {
        for mode in [
            PermissionMode::Default,
            PermissionMode::AcceptEdits,
            PermissionMode::ReadOnly,
            PermissionMode::BypassPermissions,
        ] {
            assert_eq!(
                PermissionMode::parse(mode.as_str()),
                Some(mode),
                "round trip failed for {:?}",
                mode
            );
        }
    }
}
