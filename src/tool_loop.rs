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
use serde::Deserialize;
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
const AUTO_PERMISSION_CLASSIFIER_IDLE_TIMEOUT: Duration = Duration::from_secs(45);
const AUTO_PERMISSION_CLASSIFIER_MAX_CHARS: usize = 8_000;

fn train_bifrost_enabled() -> bool {
    env::var(TRAIN_BIFROST_ENV).ok().is_some_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn train_bifrost_initial_builtin_tools() -> std::collections::HashSet<String> {
    ["write_file", "edit", "list_directory"]
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
/// normal sandbox. A one-time outside-sandbox approval is available only when
/// the model explicitly requests sandbox escalation for a shell command; it is
/// intentionally never persisted.
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

#[cfg(test)]
fn permission_options(
    tool_name: &str,
    shell_sandboxed: bool,
    always_allow_label: Option<&str>,
) -> Vec<PermissionOption> {
    permission_options_for_request(tool_name, shell_sandboxed, false, always_allow_label)
}

/// Build the permission-modal options.
///
/// For shell commands the prompt never mentions the sandbox (the title already
/// shows the full command) and the "Always allow" choice is offered only when
/// `always_allow_label` is `Some` -- i.e. the first sub-command has an
/// extractable argv prefix that isn't already remembered. The label is the
/// prefix itself (e.g. `cargo fmt --check`), making clear that approving a
/// compound command only ever remembers its first command's prefix.
fn permission_options_for_request(
    tool_name: &str,
    shell_sandboxed: bool,
    sandbox_escalation_requested: bool,
    always_allow_label: Option<&str>,
) -> Vec<PermissionOption> {
    let mut options = Vec::with_capacity(4);
    if tool_name == "run_shell_command" {
        if shell_sandboxed && sandbox_escalation_requested {
            options.push(PermissionOption::new(
                PermissionOptionId::new("allow_outside_sandbox"),
                "Run outside sandbox",
                PermissionOptionKind::AllowOnce,
            ));
            options.push(PermissionOption::new(
                PermissionOptionId::new("reject"),
                "No",
                PermissionOptionKind::RejectOnce,
            ));
            return options;
        }
        options.push(PermissionOption::new(
            PermissionOptionId::new("allow"),
            "Allow",
            PermissionOptionKind::AllowOnce,
        ));
        if let Some(label) = always_allow_label {
            options.push(PermissionOption::new(
                PermissionOptionId::new("allow_always"),
                format!("Always allow {label}"),
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
    sandbox_escalation_requested: bool,
    always_allow_label: Option<&str>,
) -> Result<PermissionGrant, String> {
    let valid_options = permission_options_for_request(
        tool_name,
        shell_sandboxed,
        sandbox_escalation_requested,
        always_allow_label,
    );
    if !valid_options
        .iter()
        .any(|option| option.option_id.0.as_ref() == option_id)
    {
        tracing::warn!(
            "request_permission returned unknown option id '{option_id}'; treating as reject"
        );
        return Err("Tool use denied (unknown option selected).".to_string());
    }

    match option_id {
        "allow_always" => Ok(PermissionGrant {
            allow_always: true,
            sandbox_policy_override: None,
        }),
        "allow" => Ok(PermissionGrant {
            allow_always: false,
            sandbox_policy_override: None,
        }),
        "allow_outside_sandbox" => Ok(PermissionGrant {
            allow_always: false,
            sandbox_policy_override: Some(SandboxPolicy::None),
        }),
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
        shell_sandboxed: bool,
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

struct GateOutcome {
    decision: GateDecision,
    usage: TokenUsage,
}

impl GateOutcome {
    fn without_usage(decision: GateDecision) -> Self {
        Self {
            decision,
            usage: TokenUsage::default(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct PermissionScopeClassification {
    allow: bool,
    rationale: String,
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
        && !matches!(kind, ToolKind::Read | ToolKind::Search | ToolKind::Fetch)
    {
        return PureGateDecision::Reject(
            "Tool use denied: read-only mode forbids edits, deletions, moves, shell execution, \
             and any tool not classified as read/search/fetch. \
             Switch the Permission menu to 'default' or 'acceptEdits' to run this tool."
                .to_string(),
        );
    }

    // Mode-independent auto-allow: pure-info kinds never mutate. In addition,
    // sandboxed shell commands that fit a conservative read-only safelist may
    // run without a prompt in the editable modes; the OS sandbox remains the
    // hard boundary for filesystem writes.
    let auto_allow = match kind {
        ToolKind::Read | ToolKind::Search | ToolKind::Fetch => true,
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
    // commands use repo-scoped argv-prefix keys (one per sub-command), while
    // regular tools use the tool name.
    if is_always_allowed {
        return PureGateDecision::Allow;
    }

    PureGateDecision::Prompt
}

/// Whether the OS-sandbox read-only safelist applies in this context: an
/// editable mode running shell commands under a real OS sandbox. Both the
/// whole-command auto-allow and the per-sub-command safelist credit gate on it.
fn shell_safelist_context(
    mode: PermissionMode,
    sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
    shell_sandboxed: bool,
) -> bool {
    matches!(mode, PermissionMode::Default | PermissionMode::AcceptEdits)
        && shell_sandboxed
        && crate::sandbox_backend::resolve_mode(sandbox_mode)
            == crate::sandbox_backend::SandboxMode::Os
}

fn should_auto_allow_shell_command(
    raw_input: &Value,
    mode: PermissionMode,
    sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
    shell_sandboxed: bool,
) -> bool {
    shell_safelist_context(mode, sandbox_mode, shell_sandboxed)
        && raw_input
            .get("command")
            .and_then(Value::as_str)
            .is_some_and(is_auto_approvable_sandboxed_shell_command)
}

fn is_auto_approvable_sandboxed_shell_command(command: &str) -> bool {
    let Some(commands) = split_simple_shell_command_sequence(command) else {
        return false;
    };

    commands.iter().all(|command| {
        let Some(tokens) = tokenize_simple_shell_command(command) else {
            return false;
        };
        is_auto_approvable_sandboxed_shell_tokens(&tokens)
    })
}

fn is_auto_approvable_sandboxed_shell_tokens(tokens: &[String]) -> bool {
    let Some(program) = tokens.first().map(String::as_str) else {
        return false;
    };

    match program {
        "pwd" | "id" | "whoami" | "uname" | "echo" | "true" | "false" | "ls" | "cat" | "head"
        | "tail" | "wc" | "cut" | "tr" | "uniq" | "nl" | "stat" | "which" | "grep" => {
            !tokens_request_file_write(tokens)
        }
        "sort" => is_safe_sort_command(tokens),
        "rg" => is_safe_rg_command(tokens),
        "find" => is_safe_find_command(tokens),
        "git" => is_safe_git_command(tokens),
        _ => false,
    }
}

fn split_simple_shell_command_sequence(command: &str) -> Option<Vec<String>> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum QuoteState {
        Plain,
        Single,
        Double,
    }

    fn push_segment(commands: &mut Vec<String>, current: &mut String) -> Option<()> {
        let segment = current.trim();
        if segment.is_empty() {
            return None;
        }
        commands.push(segment.to_string());
        current.clear();
        Some(())
    }

    let mut state = QuoteState::Plain;
    let mut current = String::new();
    let mut commands = Vec::new();
    let mut chars = command.chars().peekable();

    while let Some(ch) = chars.next() {
        match state {
            QuoteState::Plain => match ch {
                '\n' | '\r' => return None,
                '\'' => {
                    current.push(ch);
                    state = QuoteState::Single;
                }
                '"' => {
                    current.push(ch);
                    state = QuoteState::Double;
                }
                '\\' => {
                    let escaped = chars.next()?;
                    if escaped == '\n' || escaped == '\r' {
                        return None;
                    }
                    current.push(ch);
                    current.push(escaped);
                }
                '|' => {
                    if chars.peek() == Some(&'|') {
                        chars.next();
                    }
                    push_segment(&mut commands, &mut current)?;
                }
                '&' => {
                    if chars.peek() != Some(&'&') {
                        return None;
                    }
                    chars.next();
                    push_segment(&mut commands, &mut current)?;
                }
                ch if ch.is_control() => return None,
                _ => current.push(ch),
            },
            QuoteState::Single => match ch {
                '\'' => {
                    current.push(ch);
                    state = QuoteState::Plain;
                }
                ch if ch.is_control() && ch != '\t' => return None,
                _ => current.push(ch),
            },
            QuoteState::Double => match ch {
                '"' => {
                    current.push(ch);
                    state = QuoteState::Plain;
                }
                '\\' => {
                    let escaped = chars.next()?;
                    if escaped == '\n' || escaped == '\r' {
                        return None;
                    }
                    current.push(ch);
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
    push_segment(&mut commands, &mut current)?;
    Some(commands)
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
            || is_short_option_with_payload(token, 'o')
    })
}

fn token_is_option(token: &str) -> bool {
    token.starts_with('-') && token != "-"
}

fn is_short_option_with_payload(token: &str, option: char) -> bool {
    let mut chars = token.chars();
    chars.next() == Some('-') && chars.next() == Some(option) && chars.next().is_some()
}

fn has_forbidden_long_option(tokens: &[String], names: &[&str]) -> bool {
    tokens.iter().any(|token| {
        names.iter().any(|name| {
            token == name
                || token
                    .strip_prefix(name)
                    .is_some_and(|suffix| suffix.starts_with('='))
        })
    })
}

fn is_safe_sort_command(tokens: &[String]) -> bool {
    if tokens_request_file_write(tokens) || has_forbidden_long_option(tokens, &["--output"]) {
        return false;
    }
    !tokens
        .iter()
        .any(|token| is_short_option_with_payload(token, 'o'))
}

fn is_safe_rg_command(tokens: &[String]) -> bool {
    if tokens_request_file_write(tokens)
        || has_forbidden_long_option(
            tokens,
            &[
                "--files-with-matches",
                "--files-without-match",
                "--generate",
                "--pre",
                "--pre-glob",
                "--sort",
                "--sortr",
            ],
        )
    {
        return false;
    }

    !tokens.iter().any(|token| {
        matches!(token.as_str(), "--files" | "-l" | "-L")
            || is_short_option_with_payload(token, 'l')
            || is_short_option_with_payload(token, 'L')
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

fn is_safe_git_command(tokens: &[String]) -> bool {
    if tokens.len() < 2 || tokens[1].starts_with('-') {
        return false;
    }

    match tokens[1].as_str() {
        "status" | "diff" | "log" | "show" | "branch" | "rev-parse" => {}
        _ => return false,
    }

    let mut end_of_options = false;
    let mut iter = tokens.iter().skip(2).map(String::as_str);
    while let Some(token) = iter.next() {
        if end_of_options {
            continue;
        }
        if token == "--" {
            end_of_options = true;
            continue;
        }
        if !token_is_option(token) {
            continue;
        }
        if token == "-c" || token == "--config" {
            return false;
        }
        if token.starts_with("--config=") || is_short_option_with_payload(token, 'c') {
            return false;
        }
        if matches!(token, "-C" | "--git-dir" | "--work-tree") {
            if iter.next().is_none() {
                return false;
            }
            continue;
        }
        if token.starts_with("--git-dir=") || token.starts_with("--work-tree=") {
            continue;
        }
    }

    true
}

fn shell_command_will_run_sandboxed(
    permission_mode: PermissionMode,
    sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
) -> bool {
    crate::tools::sandbox::shell_command_will_run_sandboxed(permission_mode, sandbox_mode)
}

const SANDBOX_ESCALATION_COMMAND_FIELD: &str = "sandbox_permissions";
const SANDBOX_ESCALATION_REQUEST_VALUE: &str = "require_escalated";

#[derive(Clone, Debug, Eq, PartialEq)]
struct ShellSandboxRetryState {
    cwd: String,
    command: String,
    effective_directory: String,
}

impl ShellSandboxRetryState {
    fn from_raw_input(raw_input: &Value, cwd: &Path) -> Option<Self> {
        let command = raw_input.get("command")?.as_str()?.to_string();
        let cwd = cwd.canonicalize().ok()?;
        let effective_directory = match raw_input.get("directory").and_then(Value::as_str) {
            Some(directory) if !directory.trim().is_empty() => {
                crate::tools::safe_resolve(&cwd, directory).ok()?
            }
            _ => cwd.clone(),
        };
        Some(Self {
            cwd: cwd.display().to_string(),
            command,
            effective_directory: effective_directory.display().to_string(),
        })
    }

    fn matches_raw_input(&self, raw_input: &Value, cwd: &Path) -> bool {
        Self::from_raw_input(raw_input, cwd).is_some_and(|candidate| candidate == *self)
    }
}

fn shell_sandbox_escalation_requested(raw_input: &Value) -> bool {
    raw_input
        .get(SANDBOX_ESCALATION_COMMAND_FIELD)
        .and_then(Value::as_str)
        .is_some_and(|value| value == SANDBOX_ESCALATION_REQUEST_VALUE)
}

fn shell_sandbox_retry_state_index(
    states: &[ShellSandboxRetryState],
    raw_input: &Value,
    cwd: &Path,
) -> Option<usize> {
    states
        .iter()
        .position(|state| state.matches_raw_input(raw_input, cwd))
}

fn push_unique_shell_sandbox_retry_state(
    states: &mut Vec<ShellSandboxRetryState>,
    state: ShellSandboxRetryState,
) {
    if !states.contains(&state) {
        states.push(state);
    }
}

/// Number of leading argv tokens kept as a shell "Always allow" prefix.
const SHELL_PREFIX_TOKENS: usize = 3;

/// One top-level sub-command of a shell command line.
struct ShellSegment {
    /// Leading literal argv tokens, capped at [`SHELL_PREFIX_TOKENS`]; the basis
    /// for this sub-command's always-allow key.
    prefix: Vec<String>,
    /// All of the sub-command's literal argv tokens, present only when the whole
    /// sub-command is literal (no redirection, glob, or expansion). Used to test
    /// the built-in read-only safelist; `None` means "not eligible".
    safe_tokens: Option<Vec<String>>,
}

/// Decompose a shell command into its top-level sub-commands -- split on `;`,
/// `&&`, `||`, and `|`.
///
/// A "literal" token carries no redirection, glob, brace, tilde, or parameter
/// expansion, so each sub-command's `prefix` is the leading run of literal
/// tokens (capped at [`SHELL_PREFIX_TOKENS`]): trailing `2>&1` redirections and
/// `$?`-style expansions are excluded rather than stored verbatim. The whole
/// command is rejected (returns `None`) when it uses command or process
/// substitution (`` ` ``, `$(`, `<(`, `>(`) anywhere, opens a subshell
/// `( … )`, has unbalanced quotes, or contains an empty sub-command -- none of
/// those can be reduced to a prefix we are willing to vouch for, so the caller
/// must prompt instead of auto-allowing.
fn shell_command_segments(command: &str) -> Option<Vec<ShellSegment>> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum QuoteState {
        Plain,
        Single,
        Double,
    }

    // Fold the current token into the sub-command. Literal tokens extend both
    // the capped `prefix` run and the full `all` list; a non-literal token
    // (redirection/glob/expansion) closes the prefix run and marks the
    // sub-command "dirty" so it is excluded from the read-only safelist.
    #[allow(clippy::too_many_arguments)]
    fn absorb_token(
        seg_prefix: &mut Vec<String>,
        seg_all: &mut Vec<String>,
        seg_closed: &mut bool,
        seg_dirty: &mut bool,
        cur: &mut String,
        cur_started: &mut bool,
        cur_literal: &mut bool,
    ) {
        if *cur_started {
            if *cur_literal {
                let token = std::mem::take(cur);
                if !*seg_closed {
                    seg_prefix.push(token.clone());
                    if seg_prefix.len() >= SHELL_PREFIX_TOKENS {
                        *seg_closed = true;
                    }
                }
                seg_all.push(token);
            } else {
                *seg_closed = true;
                *seg_dirty = true;
            }
        }
        cur.clear();
        *cur_started = false;
        *cur_literal = true;
    }

    let mut state = QuoteState::Plain;
    let mut chars = command.chars().peekable();

    let mut segments: Vec<ShellSegment> = Vec::new();
    let mut seg_prefix: Vec<String> = Vec::new();
    let mut seg_all: Vec<String> = Vec::new();
    let mut seg_closed = false;
    let mut seg_dirty = false;
    let mut cur = String::new();
    let mut cur_started = false;
    let mut cur_literal = true;

    // Close the current sub-command. `$is_final` is true only for the flush
    // after the last char, where a trailing-empty segment (e.g. `cargo build
    // &&`) is tolerated rather than rejected. Operator arms that continue into a
    // new segment reset `seg_closed`/`seg_dirty` themselves.
    macro_rules! end_segment {
        ($is_final:expr) => {{
            absorb_token(
                &mut seg_prefix,
                &mut seg_all,
                &mut seg_closed,
                &mut seg_dirty,
                &mut cur,
                &mut cur_started,
                &mut cur_literal,
            );
            if seg_prefix.is_empty() {
                if !$is_final {
                    return None; // empty sub-command between operators
                }
            } else {
                let safe_tokens = if seg_dirty {
                    None
                } else {
                    Some(std::mem::take(&mut seg_all))
                };
                segments.push(ShellSegment {
                    prefix: std::mem::take(&mut seg_prefix),
                    safe_tokens,
                });
            }
            seg_all.clear();
        }};
    }
    macro_rules! non_literal_push {
        ($ch:expr) => {{
            cur_literal = false;
            cur_started = true;
            cur.push($ch);
        }};
    }
    macro_rules! start_next_segment {
        () => {{
            seg_closed = false;
            seg_dirty = false;
        }};
    }

    while let Some(ch) = chars.next() {
        match state {
            QuoteState::Plain => match ch {
                '\n' | '\r' => return None,
                '`' => return None,
                '$' => {
                    if chars.peek() == Some(&'(') {
                        return None; // command substitution
                    }
                    non_literal_push!('$');
                }
                '<' | '>' => {
                    if chars.peek() == Some(&'(') {
                        return None; // process substitution
                    }
                    non_literal_push!(ch);
                }
                '(' | ')' => return None, // subshell / grouping
                ';' => {
                    end_segment!(false);
                    start_next_segment!();
                }
                '|' => {
                    if chars.peek() == Some(&'|') {
                        chars.next();
                    }
                    end_segment!(false);
                    start_next_segment!();
                }
                '&' => {
                    if chars.peek() == Some(&'&') {
                        chars.next();
                        end_segment!(false);
                        start_next_segment!();
                    } else if chars.peek() == Some(&'>') || cur.ends_with('>') || cur.ends_with('<')
                    {
                        // Part of a redirection (`2>&1`, `>&2`, `&>file`): the `&`
                        // stays inside the current (dirty) token and never starts
                        // a new command.
                        non_literal_push!('&');
                    } else {
                        // A bare `&` backgrounds the preceding command and runs
                        // whatever follows as a *separate* command. Decomposing
                        // would drop that trailing command from the per-sub-command
                        // analysis while the shell still runs it, so refuse to
                        // decompose -- the caller then prompts.
                        return None;
                    }
                }
                '\'' => {
                    cur_started = true;
                    state = QuoteState::Single;
                }
                '"' => {
                    cur_started = true;
                    state = QuoteState::Double;
                }
                '\\' => {
                    let escaped = chars.next()?;
                    if escaped == '\n' || escaped == '\r' {
                        return None;
                    }
                    cur_started = true;
                    cur.push(escaped);
                }
                '*' | '?' | '[' | ']' | '{' | '}' | '~' | '!' => non_literal_push!(ch),
                ch if ch.is_whitespace() => absorb_token(
                    &mut seg_prefix,
                    &mut seg_all,
                    &mut seg_closed,
                    &mut seg_dirty,
                    &mut cur,
                    &mut cur_started,
                    &mut cur_literal,
                ),
                ch if ch.is_control() => return None,
                _ => {
                    cur_started = true;
                    cur.push(ch);
                }
            },
            QuoteState::Single => match ch {
                '\'' => state = QuoteState::Plain,
                ch if ch.is_control() && ch != '\t' => return None,
                _ => cur.push(ch),
            },
            QuoteState::Double => match ch {
                '"' => state = QuoteState::Plain,
                '`' => return None,
                '$' => {
                    if chars.peek() == Some(&'(') {
                        return None;
                    }
                    cur_literal = false;
                    cur.push('$');
                }
                '\\' => {
                    let escaped = chars.next()?;
                    if escaped == '\n' || escaped == '\r' {
                        return None;
                    }
                    cur.push(escaped);
                }
                ch if ch.is_control() && ch != '\t' => return None,
                _ => cur.push(ch),
            },
        }
    }

    if state != QuoteState::Plain {
        return None; // unbalanced quotes
    }
    end_segment!(true);

    if segments.is_empty() {
        None
    } else {
        Some(segments)
    }
}

/// Whether a sub-command is covered by the built-in read-only safelist (`head`,
/// `tail`, `grep`, `ls`, read-only `git`, …) and therefore needs no remembered
/// approval. Only fully-literal sub-commands are eligible.
fn shell_segment_is_safe(segment: &ShellSegment) -> bool {
    segment
        .safe_tokens
        .as_deref()
        .is_some_and(is_auto_approvable_sandboxed_shell_tokens)
}

/// Build the repo-scoped always-allow key for one argv prefix. Prefixes are the
/// only shape ever stored for shell commands -- exact command lines are never
/// persisted.
fn shell_prefix_key(argv_prefix: &[String], shell_sandboxed: bool) -> String {
    serde_json::json!({
        "tool": "run_shell_command",
        "rule": "prefix",
        "argvPrefix": argv_prefix,
        "shellSandboxed": shell_sandboxed,
    })
    .to_string()
}

/// How a shell command relates to the always-allow list.
struct ShellAlwaysAllowPlan {
    /// Prefix keys of every sub-command that still needs an explicit approval --
    /// i.e. not covered by the built-in read-only safelist. The command skips
    /// the prompt only when all of these are remembered.
    required_keys: Vec<String>,
    /// The first such sub-command's prefix: what "Always allow" stores and
    /// displays. `None` when every sub-command is already safelist-covered (the
    /// "Always allow" option is then withheld -- there is nothing to remember).
    first_required_prefix: Option<Vec<String>>,
}

/// Build the always-allow plan for a shell command. `safelist_credit` enables
/// crediting read-only safelist sub-commands (`head`, `tail`, …) as already
/// allowed; it should mirror the OS-sandbox auto-allow context. `None` if the
/// command can't be decomposed into prefixes, in which case the caller prompts.
fn shell_always_allow_plan(
    raw_input: &Value,
    shell_sandboxed: bool,
    safelist_credit: bool,
) -> Option<ShellAlwaysAllowPlan> {
    let command = raw_input.get("command").and_then(Value::as_str)?;
    let segments = shell_command_segments(command)?;

    let mut required_keys = Vec::new();
    let mut first_required_prefix = None;
    for segment in &segments {
        if safelist_credit && shell_segment_is_safe(segment) {
            continue;
        }
        if first_required_prefix.is_none() {
            first_required_prefix = Some(segment.prefix.clone());
        }
        required_keys.push(shell_prefix_key(&segment.prefix, shell_sandboxed));
    }

    Some(ShellAlwaysAllowPlan {
        required_keys,
        first_required_prefix,
    })
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
    original_user_request: String,
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
    let mut shell_sandbox_retry_states: Vec<ShellSandboxRetryState> = Vec::new();
    let mut tools: Vec<ToolDefinition> = if !shell_sandbox_retry_states.is_empty() {
        registry.tool_definitions_with_shell_escalation(true).await
    } else {
        registry.tool_definitions().await
    };
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
    let mut no_edit_completion_retry_count = 0usize;
    let turn_limit = if train_bifrost {
        max_turns.saturating_add(1)
    } else {
        max_turns
    };
    'outer: for turn in 0..turn_limit {
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
                training_packet
                    .as_ref()
                    .expect("BRK_TRAIN_BIFROST requires a training packet"),
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
        let mut available_shell_sandbox_retry_states = shell_sandbox_retry_states.clone();
        let mut next_shell_sandbox_retry_states = Vec::new();
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
                        training_packet
                            .as_ref()
                            .expect("BRK_TRAIN_BIFROST requires a training packet"),
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
                if train_bifrost
                    && should_retry_no_edit_completion(
                        permission_mode,
                        &tool_exchanges,
                        no_edit_completion_retry_count,
                        training_packet
                            .as_ref()
                            .expect("BRK_TRAIN_BIFROST requires a training packet"),
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
                            no_edit_completion_retry_count += 1;
                            turn_usage.add(hint_usage);
                            append_trace_record(serde_json::json!({
                                "type": "no_edit_completion_retry",
                                "reason": "final_text",
                                "turn": turn,
                                "retry_count": no_edit_completion_retry_count,
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
                                "type": "no_edit_completion_retry_skipped",
                                "reason": "final_text",
                                "turn": turn,
                                "retry_count": no_edit_completion_retry_count,
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
                        &available_shell_sandbox_retry_states,
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
                    let consume_shell_sandbox_retry_state =
                        shell_sandbox_escalation_requested(&parsed_input)
                            .then(|| {
                                shell_sandbox_retry_state_index(
                                    &available_shell_sandbox_retry_states,
                                    &parsed_input,
                                    registry.cwd(),
                                )
                            })
                            .flatten();

                    let decision = consult_gate(
                        &sessions,
                        &spawned_cx,
                        &cancel,
                        GateCheck {
                            llm,
                            model,
                            reasoning_effort,
                            original_user_request: &original_user_request,
                            idle_timeout,
                            session_id: &session_id,
                            tool_name: &tool_name,
                            kind,
                            tool_call_id: &call.id,
                            raw_input: &parsed_input,
                            cwd: registry.cwd(),
                            shell_sandbox_retry_states: &available_shell_sandbox_retry_states,
                        },
                    )
                    .await;
                    turn_usage.add(decision.usage);
                    let decision = decision.decision;

                    if let Some(index) = consume_shell_sandbox_retry_state {
                        available_shell_sandbox_retry_states.remove(index);
                    }

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
                            shell_sandboxed,
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
                                    ToolExecRequest {
                                        tool_name: &tool_name,
                                        args: parsed_input.clone(),
                                        policy,
                                        outside_sandbox_once,
                                        sandbox_mode,
                                        shell_sandboxed: sandbox_policy_override.is_none()
                                            && shell_sandboxed,
                                        cancel: &cancel,
                                    },
                                )
                                .await
                            };

                            if exec.sandbox_retry_available
                                && let Some(state) = ShellSandboxRetryState::from_raw_input(
                                    &parsed_input,
                                    registry.cwd(),
                                )
                            {
                                push_unique_shell_sandbox_retry_state(
                                    &mut next_shell_sandbox_retry_states,
                                    state,
                                );
                            }

                            // Build the terminal update -- Completed (with a
                            // Diff for write/edit tools when we have prior content)
                            // or Failed (for tool-reported errors).
                            let update = if exec.failed {
                                let clean = strip_sandbox_escalation_hint(&exec.output);
                                announce::update_failed_with_input(
                                    &call.id,
                                    &tool_name,
                                    &parsed_input,
                                    clean,
                                    Some(Value::String(clean.to_string())),
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
                let previous_shell_sandbox_retry_states = shell_sandbox_retry_states.clone();
                shell_sandbox_retry_states = available_shell_sandbox_retry_states;
                for state in next_shell_sandbox_retry_states {
                    push_unique_shell_sandbox_retry_state(&mut shell_sandbox_retry_states, state);
                }
                if shell_sandbox_retry_states != previous_shell_sandbox_retry_states {
                    tools = registry
                        .tool_definitions_with_shell_escalation(
                            !shell_sandbox_retry_states.is_empty(),
                        )
                        .await;
                    if depth >= MAX_SUBAGENT_DEPTH {
                        tools.retain(|t| t.function.name != "task");
                    }
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
                    tools = registry
                        .tool_definitions_with_shell_escalation(
                            !shell_sandbox_retry_states.is_empty(),
                        )
                        .await;
                    if depth >= MAX_SUBAGENT_DEPTH {
                        tools.retain(|t| t.function.name != "task");
                    }
                }
                if train_bifrost
                    && should_retry_no_edit_turn_limit_completion(
                        permission_mode,
                        turn,
                        max_turns,
                        &tool_exchanges,
                        no_edit_completion_retry_count,
                        training_packet
                            .as_ref()
                            .expect("BRK_TRAIN_BIFROST requires a training packet"),
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
                            no_edit_completion_retry_count += 1;
                            turn_usage.add(hint_usage);
                            append_trace_record(serde_json::json!({
                                "type": "no_edit_completion_retry",
                                "reason": "turn_limit",
                                "turn": turn,
                                "retry_count": no_edit_completion_retry_count,
                                "executed_tool_counts": executed_tool_counts(&tool_exchanges),
                                "message": nudge,
                            }));
                            messages.push(ChatMessage::user(nudge));
                            continue;
                        }
                        None => {
                            append_trace_record(serde_json::json!({
                                "type": "no_edit_completion_retry_skipped",
                                "reason": "turn_limit",
                                "turn": turn,
                                "retry_count": no_edit_completion_retry_count,
                                "executed_tool_counts": executed_tool_counts(&tool_exchanges),
                            }));
                        }
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
    /// Whether read-only safelist sub-commands count as already-allowed in this
    /// context. Threaded to the prompt path so the "Always allow" key/label is
    /// computed the same way the gate decided to prompt.
    safelist_credit: bool,
}

async fn evaluate_pure_gate(
    sessions: &SessionStore,
    session_id: &str,
    tool_name: &str,
    kind: ToolKind,
    raw_input: &Value,
    cwd: &Path,
    shell_sandbox_retry_states: &[ShellSandboxRetryState],
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
    let shell_sandbox_escalation_requested =
        tool_name == "run_shell_command" && shell_sandbox_escalation_requested(raw_input);
    if shell_sandbox_escalation_requested && shell_sandbox_retry_states.is_empty() {
        return Err("Tool use denied: outside-sandbox permission can only be requested after a sandboxed shell command fails and reports that retry option. Retry the command in the sandbox first."
            .to_string());
    }
    if shell_sandbox_escalation_requested
        && shell_sandbox_retry_state_index(shell_sandbox_retry_states, raw_input, cwd).is_none()
    {
        return Err("Tool use denied: outside-sandbox permission can only be requested when retrying the same shell command that failed under the sandbox. Retry this command in the sandbox first."
            .to_string());
    }
    if shell_sandbox_escalation_requested && !shell_sandboxed {
        return Err("Tool use denied: outside-sandbox permission was requested, but this shell command is not running under an active OS sandbox. Retry without `sandbox_permissions`."
            .to_string());
    }
    let sandbox_escalation_requested = shell_sandboxed && shell_sandbox_escalation_requested;
    let shell_auto_allow = tool_name == "run_shell_command"
        && !sandbox_escalation_requested
        && should_auto_allow_shell_command(raw_input, mode, sandbox_mode, shell_sandboxed);
    let safelist_credit = tool_name == "run_shell_command"
        && shell_safelist_context(mode, sandbox_mode, shell_sandboxed);
    let is_always_allowed = if sandbox_escalation_requested {
        false
    } else if tool_name == "run_shell_command" {
        // Each sub-command must be either remembered or covered by the read-only
        // safelist; otherwise we prompt. A command we can't decompose into
        // prefixes (substitution, subshell, …) always prompts.
        match shell_always_allow_plan(raw_input, shell_sandboxed, safelist_credit) {
            Some(plan) => {
                plan.required_keys.is_empty()
                    || sessions
                        .are_all_always_allowed(session_id, &plan.required_keys)
                        .await
            }
            None => false,
        }
    } else {
        sessions
            .is_any_always_allowed(session_id, &[tool_name.to_string()])
            .await
    };
    let decision = pure_gate_decision(mode, kind, tool_name, is_always_allowed, shell_auto_allow);

    Ok(PureGateEvaluation {
        decision,
        sandbox_mode,
        shell_sandboxed,
        safelist_credit,
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
    shell_sandbox_retry_states: &[ShellSandboxRetryState],
) -> Option<String> {
    match evaluate_pure_gate(
        sessions,
        session_id,
        tool_name,
        kind,
        raw_input,
        cwd,
        shell_sandbox_retry_states,
    )
    .await
    {
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
) -> GateOutcome {
    let evaluation = match evaluate_pure_gate(
        sessions,
        request.session_id,
        request.tool_name,
        request.kind,
        request.raw_input,
        request.cwd,
        request.shell_sandbox_retry_states,
    )
    .await
    {
        Ok(evaluation) => evaluation,
        Err(reason) => return GateOutcome::without_usage(GateDecision::Reject(reason)),
    };

    match evaluation.decision {
        PureGateDecision::Allow => GateOutcome::without_usage(GateDecision::Allow {
            sandbox_policy_override: None,
            sandbox_mode: evaluation.sandbox_mode,
            shell_sandboxed: evaluation.shell_sandboxed,
        }),
        PureGateDecision::Reject(msg) => GateOutcome::without_usage(GateDecision::Reject(msg)),
        PureGateDecision::Prompt => {
            let escalation_requested = shell_sandbox_escalation_requested(request.raw_input);
            if !escalation_requested
                && (request.tool_name != "run_shell_command" || evaluation.shell_sandboxed)
                && let Some((classification, usage)) =
                    classify_permission_scope_with_model(&request, cancel).await
            {
                if classification.allow {
                    tracing::info!(
                        session_id = request.session_id,
                        tool_name = request.tool_name,
                        rationale = %classification.rationale,
                        "permission gate: auto-classifier approved tool call for this turn"
                    );
                    return GateOutcome {
                        decision: GateDecision::Allow {
                            sandbox_policy_override: None,
                            sandbox_mode: evaluation.sandbox_mode,
                            shell_sandboxed: evaluation.shell_sandboxed,
                        },
                        usage,
                    };
                }
                tracing::info!(
                    session_id = request.session_id,
                    tool_name = request.tool_name,
                    rationale = %classification.rationale,
                    "permission gate: auto-classifier declined to approve tool call; prompting user"
                );
                // Preserve token accounting while falling back to the human prompt.
                return GateOutcome {
                    decision: match request_user_permission_with_evaluation(
                        sessions,
                        spawned_cx,
                        cancel,
                        request,
                        evaluation,
                        escalation_requested,
                    )
                    .await
                    {
                        Ok(decision) => decision,
                        Err(reason) => GateDecision::Reject(reason),
                    },
                    usage,
                };
            }
            GateOutcome::without_usage(
                match request_user_permission_with_evaluation(
                    sessions,
                    spawned_cx,
                    cancel,
                    request,
                    evaluation,
                    escalation_requested,
                )
                .await
                {
                    Ok(decision) => decision,
                    Err(reason) => GateDecision::Reject(reason),
                },
            )
        }
    }
}

async fn request_user_permission_with_evaluation(
    sessions: &SessionStore,
    spawned_cx: &SpawnedCx<'_>,
    cancel: &CancellationToken,
    request: GateCheck<'_>,
    evaluation: PureGateEvaluation,
    escalation_requested: bool,
) -> Result<GateDecision, String> {
    // "Always allow" remembers the first sub-command that actually needs
    // remembering (safelist sub-commands like `tail` are skipped).
    let shell_always_allow_prefix =
        if request.tool_name == "run_shell_command" && !escalation_requested {
            shell_always_allow_plan(
                request.raw_input,
                evaluation.shell_sandboxed,
                evaluation.safelist_credit,
            )
            .and_then(|plan| plan.first_required_prefix)
        } else {
            None
        };
    // Offer it only when that prefix isn't already remembered: if it is,
    // a *different* sub-command is forcing the prompt, so remembering the
    // first prefix again wouldn't help.
    let always_allow_label = match &shell_always_allow_prefix {
        Some(prefix) => {
            let key = shell_prefix_key(prefix, evaluation.shell_sandboxed);
            if sessions
                .is_any_always_allowed(request.session_id, std::slice::from_ref(&key))
                .await
            {
                None
            } else {
                Some(prefix.join(" "))
            }
        }
        None => None,
    };
    let grant = request_user_permission(
        spawned_cx,
        cancel,
        PermissionRequest {
            session_id: request.session_id,
            tool_name: request.tool_name,
            kind: request.kind,
            tool_call_id: request.tool_call_id,
            raw_input: request.raw_input,
            shell_sandboxed: evaluation.shell_sandboxed,
            sandbox_escalation_requested: escalation_requested,
            always_allow_label,
        },
    )
    .await?;

    // Awaited inline so the next tool call in the same batch sees the updated
    // set without re-prompting.
    if grant.allow_always && grant.sandbox_policy_override.is_none() {
        if request.tool_name == "run_shell_command" {
            if let Some(prefix) = &shell_always_allow_prefix {
                let key = shell_prefix_key(prefix, evaluation.shell_sandboxed);
                sessions.add_always_allow(request.session_id, &key).await;
            }
        } else {
            sessions
                .add_always_allow(request.session_id, request.tool_name)
                .await;
        }
    }
    Ok(GateDecision::Allow {
        sandbox_policy_override: grant.sandbox_policy_override,
        sandbox_mode: evaluation.sandbox_mode,
        shell_sandboxed: evaluation.shell_sandboxed,
    })
}

async fn classify_permission_scope_with_model(
    request: &GateCheck<'_>,
    cancel: &CancellationToken,
) -> Option<(PermissionScopeClassification, TokenUsage)> {
    if request.original_user_request.trim().is_empty() {
        return None;
    }

    let raw_input = truncate_for_permission_classifier(
        &serde_json::to_string_pretty(request.raw_input)
            .unwrap_or_else(|_| request.raw_input.to_string()),
    );
    let user_request = truncate_for_permission_classifier(request.original_user_request);
    let action_title = truncate_for_permission_classifier(&announce::permission_prompt_title(
        request.tool_name,
        request.raw_input,
    ));
    let prompt = format!(
        "Original user request:\n{user_request}\n\n\
         Proposed tool call:\n\
         - tool: {tool}\n\
         - kind: {kind:?}\n\
         - cwd: {cwd}\n\
         - title: {action_title}\n\
         - input JSON:\n{raw_input}\n\n\
         Decide whether the proposed tool call is clearly within the scope of \
         the original user request.",
        tool = request.tool_name,
        kind = request.kind,
        cwd = request.cwd.display(),
    );
    let messages = vec![
        ChatMessage::system(AUTO_PERMISSION_CLASSIFIER_SYSTEM_PROMPT),
        ChatMessage::user(prompt),
    ];
    let result = request
        .llm
        .stream_chat(StreamChatRequest {
            model: request.model.to_string(),
            messages,
            tools: None,
            reasoning_effort: request.reasoning_effort.map(str::to_string),
            structured_output: Some(permission_classifier_schema().clone()),
            on_token: Box::new(|_| {}),
            on_thought: Box::new(|_| {}),
            cancel: cancel.clone(),
            idle_timeout: request
                .idle_timeout
                .min(AUTO_PERMISSION_CLASSIFIER_IDLE_TIMEOUT),
        })
        .await;

    let response = match result {
        Ok(response) => response,
        Err(error) => {
            tracing::warn!(
                session_id = request.session_id,
                tool_name = request.tool_name,
                "permission auto-classifier failed; falling back to user prompt: {error:#}"
            );
            return None;
        }
    };
    let usage = response.usage();
    let text = match response {
        LlmResponse::Text { text, .. } => text,
        LlmResponse::ToolCalls { .. } => {
            tracing::warn!(
                session_id = request.session_id,
                tool_name = request.tool_name,
                "permission auto-classifier returned tool calls; falling back to user prompt"
            );
            return None;
        }
    };
    match parse_permission_scope_classification(&text) {
        Some(classification) => Some((classification, usage)),
        None => {
            tracing::warn!(
                session_id = request.session_id,
                tool_name = request.tool_name,
                output = %truncate_for_permission_classifier(&text),
                "permission auto-classifier returned invalid JSON; falling back to user prompt"
            );
            None
        }
    }
}

const AUTO_PERMISSION_CLASSIFIER_SYSTEM_PROMPT: &str = "\
You are a conservative permission scope classifier for a coding agent.\n\
Return JSON only.\n\
\n\
Set allow=true only when the proposed tool call is a direct, ordinary, and \
reasonably necessary step toward the user's original request. Examples include \
editing files the user asked to change, running focused tests, inspecting \
nearby code, or using a helper tool whose purpose matches the task.\n\
\n\
Set allow=false when the action starts a new task, broadens the request, \
changes credentials or secrets, performs unrelated destructive work, asks for \
outside-sandbox execution, spends money, publishes externally, or is ambiguous.\n\
\n\
The classifier only grants one tool call. It must not approve based on what \
would be convenient for the agent; it must be clearly covered by the user.";

fn permission_classifier_schema() -> &'static StructuredOutputRequest {
    static SCHEMA: std::sync::OnceLock<StructuredOutputRequest> = std::sync::OnceLock::new();
    SCHEMA.get_or_init(|| StructuredOutputRequest {
        schema_name: "permission_scope_classification".to_string(),
        allow_coercion: false,
        schema: serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["allow", "rationale"],
            "properties": {
                "allow": {
                    "type": "boolean",
                    "description": "True only when the tool call is clearly within the original user request."
                },
                "rationale": {
                    "type": "string",
                    "description": "A short reason for the decision."
                }
            }
        }),
    })
}

fn parse_permission_scope_classification(text: &str) -> Option<PermissionScopeClassification> {
    let classification: PermissionScopeClassification = serde_json::from_str(text.trim()).ok()?;
    if classification.rationale.trim().is_empty() {
        return None;
    }
    Some(classification)
}

fn truncate_for_permission_classifier(text: &str) -> String {
    if text.len() <= AUTO_PERMISSION_CLASSIFIER_MAX_CHARS {
        return text.to_string();
    }
    let mut end = AUTO_PERMISSION_CLASSIFIER_MAX_CHARS;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n... truncated", &text[..end])
}

/// Send `session/request_permission` to the client and await the outcome.
/// Returns `Ok(grant)` if the user approved (with or without remembering),
/// or `Err(reason)` describing the rejection or transport failure.
struct GateCheck<'a> {
    llm: &'a Arc<dyn LlmBackend>,
    model: &'a str,
    reasoning_effort: Option<&'a str>,
    original_user_request: &'a str,
    idle_timeout: Duration,
    session_id: &'a str,
    tool_name: &'a str,
    kind: ToolKind,
    tool_call_id: &'a str,
    raw_input: &'a Value,
    cwd: &'a Path,
    shell_sandbox_retry_states: &'a [ShellSandboxRetryState],
}

struct PermissionRequest<'a> {
    session_id: &'a str,
    tool_name: &'a str,
    kind: ToolKind,
    tool_call_id: &'a str,
    raw_input: &'a Value,
    shell_sandboxed: bool,
    sandbox_escalation_requested: bool,
    /// `Some(prefix)` offers a shell "Always allow <prefix>" choice; `None`
    /// withholds it (non-shell tools ignore this and always offer their own).
    always_allow_label: Option<String>,
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
        sandbox_escalation_requested,
        always_allow_label,
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

    let options = permission_options_for_request(
        tool_name,
        shell_sandboxed,
        sandbox_escalation_requested,
        always_allow_label.as_deref(),
    );

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
                permission_grant_for_selection(
                    tool_name,
                    id,
                    shell_sandboxed,
                    sandbox_escalation_requested,
                    always_allow_label.as_deref(),
                )
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
    sandbox_retry_available: bool,
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

fn has_successful_training_file_change(
    tool_exchanges: &[ToolExchange],
    packet: &TrainingPacket,
) -> bool {
    tool_exchanges.iter().any(|exchange| {
        if tool_result_failed(&exchange.result) {
            return false;
        }
        let Some(path) = file_change_target_path(exchange) else {
            return false;
        };
        packet.files.iter().any(|file| file.path == path)
    })
}

fn file_change_target_path(exchange: &ToolExchange) -> Option<String> {
    if !matches!(exchange.tool_name.as_str(), "edit" | "write_file") {
        return None;
    }
    let args = serde_json::from_str::<Value>(&exchange.arguments).ok()?;
    let path = args.get("file_path")?.as_str()?;
    normalize_tool_path(path)
}

fn normalize_tool_path(path: &str) -> Option<String> {
    let path = path.trim().replace('\\', "/");
    if path.is_empty() || path.starts_with('/') {
        return None;
    }
    Some(
        path.strip_prefix("a/")
            .or_else(|| path.strip_prefix("b/"))
            .unwrap_or(path.as_str())
            .trim_start_matches("./")
            .to_string(),
    )
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
    packet: &TrainingPacket,
) -> bool {
    if matches!(permission_mode, PermissionMode::ReadOnly) {
        return false;
    }
    if turn >= max_turns - 1 || has_successful_training_file_change(tool_exchanges, packet) {
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

fn should_retry_no_edit_completion(
    permission_mode: PermissionMode,
    tool_exchanges: &[ToolExchange],
    retry_count: usize,
    packet: &TrainingPacket,
) -> bool {
    if matches!(permission_mode, PermissionMode::ReadOnly) {
        return false;
    }
    retry_count == 0 && !has_successful_training_file_change(tool_exchanges, packet)
}

fn should_retry_no_edit_turn_limit_completion(
    permission_mode: PermissionMode,
    turn: usize,
    max_turns: usize,
    tool_exchanges: &[ToolExchange],
    retry_count: usize,
    packet: &TrainingPacket,
) -> bool {
    turn >= max_turns.saturating_sub(1)
        && should_retry_no_edit_completion(permission_mode, tool_exchanges, retry_count, packet)
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
    packet: &TrainingPacket,
) -> bool {
    if matches!(permission_mode, PermissionMode::ReadOnly) {
        return false;
    }
    if nudge_count >= 2 || has_successful_training_file_change(tool_exchanges, packet) {
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
struct ToolExecRequest<'a> {
    tool_name: &'a str,
    args: Value,
    policy: SandboxPolicy,
    outside_sandbox_once: bool,
    sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
    shell_sandboxed: bool,
    cancel: &'a CancellationToken,
}

async fn execute_tool(registry: &ToolRegistry, request: ToolExecRequest<'_>) -> ToolExecution {
    let result = registry
        .execute_with_sandbox_mode_cancellable(
            request.tool_name,
            request.args,
            request.policy,
            request.outside_sandbox_once,
            request.sandbox_mode,
            Some(request.cancel),
        )
        .await;
    tool_result_to_execution(
        request.tool_name,
        request.shell_sandboxed,
        request.outside_sandbox_once,
        result,
    )
}

fn tool_result_to_execution(
    tool_name: &str,
    shell_sandboxed: bool,
    outside_sandbox_once: bool,
    result: crate::tools::ToolResult,
) -> ToolExecution {
    let (status_prefix, failed) = match result.status {
        ToolStatus::Success => ("", false),
        ToolStatus::RequestError => ("Error: ", true),
        ToolStatus::InternalError => ("Internal error: ", true),
    };
    let mut output = format!("{}{}", status_prefix, result.output);
    let sandbox_retry_available = tool_name == "run_shell_command"
        && failed
        && !outside_sandbox_once
        && shell_sandboxed
        && is_likely_sandbox_limitation(&output);
    if sandbox_retry_available {
        output.push_str(SANDBOX_FAILURE_ESCALATION_HINT);
    }
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
    ToolExecution {
        output,
        failed,
        sandbox_retry_available,
    }
}

const SANDBOX_FAILURE_ESCALATION_HINT: &str = "\n\n⚠️  This command was blocked by the OS sandbox. Retry it with `sandbox_permissions: \"require_escalated\"` to run outside the sandbox.";

/// Strip the sandbox-escalation hint from output so client-facing
/// tool-call cards show a clean message. The full hint still flows
/// to the LLM via `ToolExecution.output`.
fn strip_sandbox_escalation_hint(output: &str) -> &str {
    output
        .trim_end()
        .strip_suffix(SANDBOX_FAILURE_ESCALATION_HINT)
        .unwrap_or(output)
}

fn is_likely_sandbox_limitation(output: &str) -> bool {
    let lower = output.to_ascii_lowercase();
    [
        "permission denied",
        "operation not permitted",
        "read-only file system",
        "readonly file system",
        "read-only filesystem",
        "read only file system",
        "access denied",
        "not permitted",
        "eperm",
        "eacces",
        "namespace",
        "seccomp",
        "cannot create directory",
        "cannot touch",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
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
                    sandbox_retry_available: false,
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
                    sandbox_retry_available: false,
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
                        sandbox_retry_available: false,
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
                    sandbox_retry_available: false,
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
        prompt.to_string(),
        NotificationMode::Silent,
        depth,
    ))
    .await;

    let (text, _exchanges, nested_usage) = nested;
    let exec = if text.trim().is_empty() {
        ToolExecution {
            output: format!("Error: subagent '{subagent_name}' returned an empty response."),
            failed: true,
            sandbox_retry_available: false,
        }
    } else {
        ToolExecution {
            output: text,
            failed: false,
            sandbox_retry_available: false,
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

    fn training_packet_for_test(path: &str) -> TrainingPacket {
        TrainingPacket {
            files: vec![train_bifrost::TrainingFile {
                path: path.to_string(),
                diff: "diff --git a/src/lib.rs b/src/lib.rs\n".to_string(),
            }],
            related_files: Vec::new(),
        }
    }

    fn file_exchange_for_test(tool_name: &str, path: &str) -> ToolExchange {
        ToolExchange {
            call_id: format!("call-{tool_name}"),
            tool_name: tool_name.to_string(),
            arguments: serde_json::json!({ "file_path": path }).to_string(),
            result: format!("Edited '{path}'"),
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

    struct StaticClassifierBackend {
        response: &'static str,
        calls: Arc<AtomicUsize>,
    }

    impl LlmBackend for StaticClassifierBackend {
        fn list_models(&self) -> BoxFuture<'_, anyhow::Result<Vec<String>>> {
            async { Ok(Vec::new()) }.boxed()
        }

        fn stream_chat(
            &self,
            request: StreamChatRequest,
        ) -> BoxFuture<'_, anyhow::Result<LlmResponse>> {
            let response = self.response.to_string();
            let calls = self.calls.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                assert!(request.tools.is_none());
                assert!(request.structured_output.is_some());
                assert!(
                    request.messages[1]
                        .content_text()
                        .contains("Original user request:")
                );
                Ok(LlmResponse::Text {
                    text: response,
                    usage: TokenUsage {
                        input_tokens: 3,
                        output_tokens: 2,
                        thought_tokens: 1,
                        cached_read_tokens: 0,
                        cached_write_tokens: 0,
                    },
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

    #[test]
    fn permission_scope_classification_requires_valid_json_and_rationale() {
        assert!(parse_permission_scope_classification("not json").is_none());
        assert!(
            parse_permission_scope_classification(r#"{"allow":true,"rationale":""}"#).is_none()
        );

        let parsed =
            parse_permission_scope_classification(r#"{"allow":false,"rationale":"too broad"}"#)
                .expect("valid classifier JSON should parse");
        assert!(!parsed.allow);
        assert_eq!(parsed.rationale, "too broad");
    }

    #[test]
    fn permission_classifier_truncation_preserves_utf8_boundary() {
        let text = "é".repeat(AUTO_PERMISSION_CLASSIFIER_MAX_CHARS);
        let truncated = truncate_for_permission_classifier(&text);
        assert!(truncated.ends_with("\n... truncated"));
        assert!(truncated.is_char_boundary(truncated.len()));
    }

    #[tokio::test]
    async fn permission_auto_classifier_uses_model_and_returns_usage() {
        let calls = Arc::new(AtomicUsize::new(0));
        let llm: Arc<dyn LlmBackend> = Arc::new(StaticClassifierBackend {
            response: r#"{"allow":true,"rationale":"focused test command"}"#,
            calls: calls.clone(),
        });
        let raw_input = serde_json::json!({"command": "cargo test"});
        let request = GateCheck {
            llm: &llm,
            model: "test-model",
            reasoning_effort: None,
            original_user_request: "fix the failing tests",
            idle_timeout: Duration::from_secs(300),
            session_id: "session",
            tool_name: "run_shell_command",
            kind: ToolKind::Execute,
            tool_call_id: "call",
            raw_input: &raw_input,
            cwd: Path::new("/tmp/project"),
            shell_sandbox_retry_states: &[],
        };

        let (classification, usage) =
            classify_permission_scope_with_model(&request, &CancellationToken::new())
                .await
                .expect("classifier should parse valid model output");

        assert!(classification.allow);
        assert_eq!(classification.rationale, "focused test command");
        assert_eq!(usage.total_tokens(), 6);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
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
            tool_call_for_test("edit"),
            tool_call_for_test("read_file"),
            tool_call_for_test("run_shell_command"),
        ];

        let names = ordered_names_for_test(&calls, &["search_symbols"]);

        assert_eq!(names, vec!["edit", "read_file", "run_shell_command"]);
    }

    #[test]
    fn advertised_tool_names_match_current_request_catalog() {
        let tools = vec![tool_def_for_test("read_file"), tool_def_for_test("edit")];
        let names = advertised_tool_names(Some(&tools));

        assert!(names.contains("read_file"));
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
        let packet = training_packet_for_test("src/lib.rs");

        assert!(should_reject_no_edit_final_answer(
            PermissionMode::Default,
            3,
            10,
            &prior,
            &packet
        ));
        assert!(!should_reject_no_edit_final_answer(
            PermissionMode::ReadOnly,
            3,
            10,
            &prior,
            &packet
        ));
    }

    #[test]
    fn no_edit_final_guard_allows_after_successful_gold_path_edit() {
        let prior = vec![file_exchange_for_test("edit", "src/lib.rs")];
        let packet = training_packet_for_test("src/lib.rs");

        assert!(!should_reject_no_edit_final_answer(
            PermissionMode::Default,
            3,
            10,
            &prior,
            &packet
        ));
        assert!(has_successful_file_change(&prior));
        assert!(has_successful_training_file_change(&prior, &packet));
    }

    #[test]
    fn no_edit_gate_ignores_successful_scratch_writes() {
        let prior = vec![file_exchange_for_test("write_file", ".tmp")];
        let packet = training_packet_for_test("src/lib.rs");

        assert!(has_successful_file_change(&prior));
        assert!(!has_successful_training_file_change(&prior, &packet));
        assert!(should_retry_no_edit_completion(
            PermissionMode::Default,
            &prior,
            0,
            &packet
        ));
    }

    #[test]
    fn no_edit_final_guard_does_not_reject_on_last_turn() {
        let prior = vec![exchange_for_test("search_symbols")];
        let packet = training_packet_for_test("src/lib.rs");

        assert!(!should_reject_no_edit_final_answer(
            PermissionMode::Default,
            9,
            10,
            &prior,
            &packet
        ));
    }

    #[test]
    fn no_edit_completion_retry_triggers_without_prior_edit() {
        let prior = vec![exchange_for_test("search_symbols")];
        let packet = training_packet_for_test("src/lib.rs");

        assert!(should_retry_no_edit_completion(
            PermissionMode::Default,
            &prior,
            0,
            &packet
        ));
        assert!(!should_retry_no_edit_completion(
            PermissionMode::ReadOnly,
            &prior,
            0,
            &packet
        ));
    }

    #[test]
    fn no_edit_completion_retry_is_one_shot_and_allows_successful_gold_path_edits() {
        let edited = vec![file_exchange_for_test("edit", "src/lib.rs")];
        let searched = vec![exchange_for_test("search_symbols")];
        let packet = training_packet_for_test("src/lib.rs");

        assert!(!should_retry_no_edit_completion(
            PermissionMode::Default,
            &edited,
            0,
            &packet
        ));
        assert!(!should_retry_no_edit_completion(
            PermissionMode::Default,
            &searched,
            1,
            &packet
        ));
    }

    #[test]
    fn no_edit_turn_limit_completion_retry_triggers_at_limit_only() {
        let searched = vec![exchange_for_test("search_symbols")];
        let packet = training_packet_for_test("src/lib.rs");

        assert!(!should_retry_no_edit_turn_limit_completion(
            PermissionMode::Default,
            8,
            10,
            &searched,
            0,
            &packet
        ));
        assert!(should_retry_no_edit_turn_limit_completion(
            PermissionMode::Default,
            9,
            10,
            &searched,
            0,
            &packet
        ));
        assert!(!should_retry_no_edit_turn_limit_completion(
            PermissionMode::Default,
            9,
            10,
            &searched,
            1,
            &packet
        ));
    }

    #[test]
    fn no_edit_turn_limit_completion_retry_is_independent_of_progress_nudge_cap() {
        let searched = vec![exchange_for_test("search_symbols")];
        let packet = training_packet_for_test("src/lib.rs");

        assert!(!should_emit_no_edit_progress_nudge(
            PermissionMode::Default,
            12,
            25,
            &searched,
            2,
            &packet
        ));
        assert!(should_retry_no_edit_turn_limit_completion(
            PermissionMode::Default,
            24,
            25,
            &searched,
            0,
            &packet
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
        let packet = training_packet_for_test("src/lib.rs");

        assert!(should_emit_no_edit_progress_nudge(
            PermissionMode::Default,
            8,
            25,
            &prior,
            0,
            &packet
        ));
        assert!(!should_emit_no_edit_progress_nudge(
            PermissionMode::ReadOnly,
            8,
            25,
            &prior,
            0,
            &packet
        ));
        assert!(!should_emit_no_edit_progress_nudge(
            PermissionMode::Default,
            7,
            25,
            &prior,
            0,
            &packet
        ));
        assert!(!should_emit_no_edit_progress_nudge(
            PermissionMode::Default,
            8,
            25,
            &prior,
            2,
            &packet
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
        let packet = training_packet_for_test("src/lib.rs");

        assert!(!should_emit_no_edit_progress_nudge(
            PermissionMode::Default,
            7,
            25,
            &prior,
            0,
            &packet
        ));
        assert!(should_emit_no_edit_progress_nudge(
            PermissionMode::Default,
            8,
            25,
            &prior,
            0,
            &packet
        ));
    }

    #[test]
    fn no_edit_progress_nudge_allows_after_successful_gold_path_edit() {
        let prior = vec![
            exchange_for_test("search_symbols"),
            exchange_for_test("get_symbol_sources"),
            exchange_for_test("scan_usages"),
            exchange_for_test("get_summaries"),
            exchange_for_test("search_symbols"),
            exchange_for_test("get_symbol_sources"),
            exchange_for_test("read_file"),
            file_exchange_for_test("edit", "src/lib.rs"),
        ];
        let packet = training_packet_for_test("src/lib.rs");

        assert!(!should_emit_no_edit_progress_nudge(
            PermissionMode::Default,
            12,
            25,
            &prior,
            0,
            &packet
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
        for kind in [ToolKind::Read, ToolKind::Search, ToolKind::Fetch] {
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
                &[],
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
            &[],
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
            &[],
        )
        .await;
        assert!(read.is_none(), "read-only should allow read tools");
    }

    #[tokio::test]
    async fn preflight_rejects_shell_sandbox_escalation_before_retry_field_enabled() {
        let cwd = tempfile::tempdir().expect("temp cwd");
        let store = SessionStore::new("m".to_string());
        let session = store.create_session(cwd.path().to_path_buf()).await;

        let rejection = deterministic_gate_rejection(
            &store,
            &session.id,
            "run_shell_command",
            ToolRegistry::tool_kind("run_shell_command"),
            &serde_json::json!({
                "command": "echo ok",
                "sandbox_permissions": "require_escalated",
            }),
            cwd.path(),
            &[],
        )
        .await
        .expect("premature outside-sandbox retry must be rejected before prompting");

        assert!(
            rejection.contains("outside-sandbox permission can only be requested"),
            "unexpected rejection: {rejection}"
        );
    }

    #[tokio::test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    async fn preflight_allows_matching_shell_sandbox_retry() {
        use crate::sandbox_backend::SandboxMode;

        let cwd = tempfile::tempdir().expect("temp cwd");
        std::fs::create_dir_all(cwd.path().join("crates/app")).expect("test directory");
        let store = SessionStore::new("m".to_string());
        let session = store.create_session(cwd.path().to_path_buf()).await;
        assert!(
            store
                .set_sandbox_mode(&session.id, Some(SandboxMode::Os))
                .await
        );
        let original = serde_json::json!({"command": "cargo test", "directory": "crates/app"});
        let retry_state =
            ShellSandboxRetryState::from_raw_input(&original, cwd.path()).expect("retry state");

        let rejection = deterministic_gate_rejection(
            &store,
            &session.id,
            "run_shell_command",
            ToolRegistry::tool_kind("run_shell_command"),
            &serde_json::json!({
                "command": "cargo test",
                "directory": "crates/app",
                "sandbox_permissions": "require_escalated",
            }),
            cwd.path(),
            std::slice::from_ref(&retry_state),
        )
        .await;

        assert!(
            rejection.is_none(),
            "matching retry should reach the prompt"
        );
    }

    #[tokio::test]
    async fn preflight_rejects_shell_sandbox_retry_for_different_command() {
        use crate::sandbox_backend::SandboxMode;

        let cwd = tempfile::tempdir().expect("temp cwd");
        let store = SessionStore::new("m".to_string());
        let session = store.create_session(cwd.path().to_path_buf()).await;
        assert!(
            store
                .set_sandbox_mode(&session.id, Some(SandboxMode::Os))
                .await
        );
        let original = serde_json::json!({"command": "cargo test"});
        let retry_state =
            ShellSandboxRetryState::from_raw_input(&original, cwd.path()).expect("retry state");

        let rejection = deterministic_gate_rejection(
            &store,
            &session.id,
            "run_shell_command",
            ToolRegistry::tool_kind("run_shell_command"),
            &serde_json::json!({
                "command": "cargo check",
                "sandbox_permissions": "require_escalated",
            }),
            cwd.path(),
            std::slice::from_ref(&retry_state),
        )
        .await
        .expect("different command should be rejected");

        assert!(
            rejection.contains("retrying the same shell command"),
            "unexpected rejection: {rejection}"
        );
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
        for kind in [ToolKind::Read, ToolKind::Search, ToolKind::Fetch] {
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
            "git diff --stat"
        ));
        assert!(is_auto_approvable_sandboxed_shell_command(
            "git show HEAD~1"
        ));
        assert!(is_auto_approvable_sandboxed_shell_command(
            "rg PermissionMode src"
        ));
    }

    #[test]
    fn safe_shell_classifier_accepts_pipelines_and_conditionals_of_read_only_commands() {
        assert!(is_auto_approvable_sandboxed_shell_command(
            "grep PermissionMode src/tool_loop.rs | head -n 5"
        ));
        assert!(is_auto_approvable_sandboxed_shell_command(
            "rg PermissionMode src | sort | uniq | head"
        ));
        assert!(is_auto_approvable_sandboxed_shell_command(
            "grep 'PermissionMode|ToolKind' src/tool_loop.rs | wc -l"
        ));
        assert!(is_auto_approvable_sandboxed_shell_command(
            "grep PermissionMode src/tool_loop.rs && git status"
        ));
        assert!(is_auto_approvable_sandboxed_shell_command("false || true"));
    }

    #[test]
    fn safe_shell_classifier_rejects_pipelines_with_unsafe_segments() {
        assert!(!is_auto_approvable_sandboxed_shell_command(
            "grep PermissionMode src/tool_loop.rs | python3 -c 'print(1)'"
        ));
        assert!(!is_auto_approvable_sandboxed_shell_command(
            "grep PermissionMode src/tool_loop.rs | sed -i 's/a/b/'"
        ));
        assert!(!is_auto_approvable_sandboxed_shell_command(
            "grep PermissionMode src/tool_loop.rs |"
        ));
    }

    #[test]
    fn command_sequence_splitter_splits_unquoted_connectors() {
        assert_eq!(
            split_simple_shell_command_sequence("grep 'a|b' file | head")
                .expect("pipeline should split"),
            vec!["grep 'a|b' file", "head"]
        );
        assert_eq!(
            split_simple_shell_command_sequence("rg foo src && git status || pwd")
                .expect("connectors should split"),
            vec!["rg foo src", "git status", "pwd"]
        );
        assert!(split_simple_shell_command_sequence("| head").is_none());
        assert!(split_simple_shell_command_sequence("grep a &&").is_none());
        assert!(split_simple_shell_command_sequence("grep a & head").is_none());
    }
    #[test]
    fn safe_shell_classifier_rejects_writes_and_shell_metacharacters() {
        assert!(!is_auto_approvable_sandboxed_shell_command(
            "sed -i 's/a/b/' src/main.rs"
        ));
        assert!(!is_auto_approvable_sandboxed_shell_command(
            "python3 -c 'print(1)'"
        ));
        assert!(is_auto_approvable_sandboxed_shell_command("pwd && ls"));
        assert!(!is_auto_approvable_sandboxed_shell_command("pwd; ls"));
    }

    #[test]
    fn safe_shell_classifier_rejects_command_options_with_side_effects() {
        assert!(!is_auto_approvable_sandboxed_shell_command(
            "sed '1w out.txt' src/main.rs"
        ));
        assert!(!is_auto_approvable_sandboxed_shell_command(
            "awk 'BEGIN { system(\"touch out.txt\") }'"
        ));
        assert!(!is_auto_approvable_sandboxed_shell_command(
            "sort -oout.txt Cargo.toml"
        ));
        assert!(!is_auto_approvable_sandboxed_shell_command(
            "sort --output=out.txt Cargo.toml"
        ));
        assert!(!is_auto_approvable_sandboxed_shell_command(
            "rg --pre cat needle src"
        ));
        assert!(!is_auto_approvable_sandboxed_shell_command(
            "rg --sort path needle src"
        ));
    }

    #[test]
    fn safe_git_classifier_rejects_global_flags_and_mutating_subcommands() {
        assert!(!is_auto_approvable_sandboxed_shell_command(
            "git -C /tmp status"
        ));
        assert!(!is_auto_approvable_sandboxed_shell_command(
            "git commit -m nope"
        ));
        assert!(!is_auto_approvable_sandboxed_shell_command(
            "git apply patch.diff"
        ));
        assert!(!is_auto_approvable_sandboxed_shell_command(
            "git status -c core.pager='sh -c echo'"
        ));
        assert!(!is_auto_approvable_sandboxed_shell_command(
            "git log --config=core.pager=cat"
        ));
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
    fn shell_permission_prompt_omits_sandbox_language_and_uses_prefix_label() {
        // Sandboxed and unsandboxed prompts read identically now: no sandbox
        // text, and "Always allow" carries the first sub-command's prefix.
        for shell_sandboxed in [true, false] {
            let options = permission_options(
                "run_shell_command",
                shell_sandboxed,
                Some("cargo fmt --check"),
            );
            let labels: Vec<_> = options
                .iter()
                .map(|option| (option.option_id.0.as_ref(), option.name.as_str()))
                .collect();

            assert_eq!(
                labels,
                vec![
                    ("allow", "Allow"),
                    ("allow_always", "Always allow cargo fmt --check"),
                    ("reject", "Reject"),
                ],
                "shell_sandboxed={shell_sandboxed}"
            );
        }
    }

    #[test]
    fn shell_permission_prompt_hides_always_allow_without_prefix() {
        // No extractable/offerable prefix -> no "Always allow" choice.
        let options = permission_options("run_shell_command", true, None);
        let ids: Vec<_> = options
            .iter()
            .map(|option| option.option_id.0.as_ref())
            .collect();

        assert_eq!(ids, vec!["allow", "reject"]);
    }

    #[test]
    fn shell_permission_prompt_includes_explicit_outside_sandbox_choice_when_requested() {
        let options = permission_options_for_request("run_shell_command", true, true, None);
        let labels: Vec<_> = options
            .iter()
            .map(|option| (option.option_id.0.as_ref(), option.name.as_str()))
            .collect();

        assert_eq!(
            labels,
            vec![
                ("allow_outside_sandbox", "Run outside sandbox"),
                ("reject", "No"),
            ]
        );
    }

    #[test]
    fn shell_allow_always_choice_is_rejected_when_not_offered() {
        // Selecting "allow_always" when no prefix label was offered must not
        // smuggle in an approval.
        let err =
            permission_grant_for_selection("run_shell_command", "allow_always", true, false, None)
                .expect_err("allow_always must be rejected when it was not offered");
        assert!(err.contains("unknown option"), "got: {err}");
    }

    #[test]
    fn shell_allow_always_choice_maps_to_sandboxed_session_approval() {
        let grant = permission_grant_for_selection(
            "run_shell_command",
            "allow_always",
            true,
            false,
            Some("cargo fmt --check"),
        )
        .expect("shell sticky sandbox approval should be accepted");

        assert_eq!(
            grant,
            PermissionGrant {
                allow_always: true,
                sandbox_policy_override: None,
            }
        );
    }

    // Convenience: the leading argv prefix of each sub-command.
    fn segment_prefixes(command: &str) -> Option<Vec<Vec<String>>> {
        Some(
            shell_command_segments(command)?
                .into_iter()
                .map(|segment| segment.prefix)
                .collect(),
        )
    }

    #[test]
    fn shell_first_command_prefix_strips_redirections_and_pipes() {
        let cases: [(&str, &[&str]); 6] = [
            // Trailing `2>&1 | …` is excluded; the first command's prefix stands.
            (
                "cargo fmt --check 2>&1 | tail -5",
                &["cargo", "fmt", "--check"],
            ),
            (
                "cargo test --workspace --lib",
                &["cargo", "test", "--workspace"],
            ),
            (
                "git status --short && git diff -- x",
                &["git", "status", "--short"],
            ),
            ("cargo fmt && cargo clippy", &["cargo", "fmt"]),
            ("tail -5 file", &["tail", "-5", "file"]),
            // `$?` keeps the literal head; expansion just closes the prefix.
            ("echo done $?", &["echo", "done"]),
        ];
        for (command, want) in cases {
            let got = segment_prefixes(command).expect(command);
            assert_eq!(got[0], want, "command={command}");
        }
    }

    #[test]
    fn shell_command_segments_split_each_subcommand() {
        assert_eq!(
            segment_prefixes("cargo fmt && cargo clippy --all-targets | tail -8")
                .expect("prefixes"),
            vec![
                vec!["cargo".to_string(), "fmt".to_string()],
                vec![
                    "cargo".to_string(),
                    "clippy".to_string(),
                    "--all-targets".to_string()
                ],
                vec!["tail".to_string(), "-8".to_string()],
            ]
        );
    }

    #[test]
    fn shell_command_segments_reject_unsafe_or_malformed() {
        for command in [
            "echo $(rm -rf /)",       // command substitution
            "echo `whoami`",          // backtick substitution
            "diff <(a) <(b)",         // process substitution
            "(cd /tmp && ls)",        // subshell
            "a || | b",               // empty middle sub-command
            "   ",                    // no command
            "\"unbalanced",           // unbalanced quote
            "cargo build & rm -rf ~", // background `&` hides a second command
            "true & true & rm -rf /", // chained backgrounding
            "ls &",                   // trailing background
            "a |& b",                 // pipe-both then background
        ] {
            assert!(
                shell_command_segments(command).is_none(),
                "expected None for {command:?}"
            );
        }
    }

    #[test]
    fn shell_background_ampersand_never_rides_a_remembered_prefix() {
        // A bare `&` backgrounds the head and runs the rest as a separate
        // command the shell still executes. Decomposing would drop it from the
        // analysis, so the whole line must refuse to decompose (-> prompt) even
        // when the first sub-command's prefix is remembered.
        let bg = serde_json::json!({"command": "cargo build & rm -rf ~"});
        assert!(shell_always_allow_plan(&bg, true, true).is_none());
        assert!(shell_always_allow_plan(&bg, true, false).is_none());

        // But `&` inside a redirection (`2>&1`) is preserved, not rejected.
        let redir = serde_json::json!({"command": "cargo test --lib 2>&1 | tail -40"});
        let plan = shell_always_allow_plan(&redir, true, true).expect("redirection plan");
        assert_eq!(
            plan.required_keys,
            vec![shell_prefix_key(
                &["cargo".into(), "test".into(), "--lib".into()],
                true
            )]
        );
    }

    #[test]
    fn shell_plan_required_keys_skip_safelisted_subcommands_only_with_credit() {
        // `cargo test … | tail` mixes a non-safe head with a safe tail.
        let raw = serde_json::json!({"command": "cargo test --lib 2>&1 | tail -40"});
        let cargo_key = shell_prefix_key(&["cargo".into(), "test".into(), "--lib".into()], true);
        let tail_key = shell_prefix_key(&["tail".into(), "-40".into()], true);

        // No credit: every sub-command must be remembered.
        let no_credit = shell_always_allow_plan(&raw, true, false).expect("plan");
        assert_eq!(no_credit.required_keys, vec![cargo_key.clone(), tail_key]);

        // With credit: `tail` is covered by the built-in safelist, so only the
        // cargo prefix needs remembering -- and that's what "Always allow" stores.
        let with_credit = shell_always_allow_plan(&raw, true, true).expect("plan");
        assert_eq!(with_credit.required_keys, vec![cargo_key]);
        assert_eq!(
            with_credit.first_required_prefix,
            Some(vec![
                "cargo".to_string(),
                "test".to_string(),
                "--lib".to_string()
            ])
        );
    }

    #[test]
    fn shell_plan_is_empty_when_every_subcommand_is_safelisted() {
        // `grep … | head` is entirely read-only: nothing to remember, and no
        // "Always allow" prefix to offer.
        let raw = serde_json::json!({"command": "grep foo file | head -5"});
        let plan = shell_always_allow_plan(&raw, true, true).expect("plan");
        assert!(plan.required_keys.is_empty());
        assert!(plan.first_required_prefix.is_none());
    }

    #[test]
    fn shell_outside_sandbox_choice_maps_to_policy_override() {
        let grant = permission_grant_for_selection(
            "run_shell_command",
            "allow_outside_sandbox",
            true,
            true,
            None,
        )
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
    fn shell_escalation_prompt_rejects_sticky_sandbox_options() {
        for option_id in ["allow", "allow_always"] {
            let err =
                permission_grant_for_selection("run_shell_command", option_id, true, true, None)
                    .expect_err("escalation prompt must reject options it did not offer");
            assert!(err.contains("unknown option"), "got: {err}");
        }
    }

    #[test]
    fn shell_outside_sandbox_choice_is_rejected_when_shell_sandbox_disabled() {
        let err = permission_grant_for_selection(
            "run_shell_command",
            "allow_outside_sandbox",
            false,
            true,
            None,
        )
        .expect_err("outside-sandbox option is not valid when shell sandboxing is disabled");
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
        let options = permission_options("write_file", false, None);
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

        let grant =
            permission_grant_for_selection("write_file", "allow_always", false, false, None)
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
    fn sandbox_escalation_hint_requires_actual_shell_sandbox() {
        let result = crate::tools::ToolResult {
            status: ToolStatus::RequestError,
            output: "permission denied".to_string(),
        };

        let exec = tool_result_to_execution("run_shell_command", false, false, result);

        assert!(!exec.output.contains(SANDBOX_FAILURE_ESCALATION_HINT));
        assert!(!exec.sandbox_retry_available);
    }

    #[test]
    fn sandbox_escalation_hint_is_added_for_sandboxed_shell_failures() {
        let result = crate::tools::ToolResult {
            status: ToolStatus::RequestError,
            output: "permission denied".to_string(),
        };

        let exec = tool_result_to_execution("run_shell_command", true, false, result);

        assert!(exec.output.contains(SANDBOX_FAILURE_ESCALATION_HINT));
        assert!(exec.sandbox_retry_available);
    }

    #[test]
    fn sandbox_retry_is_not_unlocked_by_spoofed_hint_text() {
        let result = crate::tools::ToolResult {
            status: ToolStatus::RequestError,
            output: format!("script printed this text:{SANDBOX_FAILURE_ESCALATION_HINT}"),
        };

        let exec = tool_result_to_execution("run_shell_command", true, false, result);

        assert!(exec.output.contains(SANDBOX_FAILURE_ESCALATION_HINT));
        assert!(!exec.sandbox_retry_available);
    }

    #[test]
    fn shell_sandbox_retry_state_normalizes_equivalent_directories() {
        let cwd = tempfile::tempdir().expect("temp cwd");
        std::fs::create_dir_all(cwd.path().join("crates/app")).expect("test directory");
        let original = serde_json::json!({
            "command": "cargo test",
            "directory": "crates/app",
        });
        let retry_state =
            ShellSandboxRetryState::from_raw_input(&original, cwd.path()).expect("retry state");

        assert!(retry_state.matches_raw_input(
            &serde_json::json!({
                "command": "cargo test",
                "directory": "./crates/app",
            }),
            cwd.path(),
        ));
    }

    #[test]
    fn shell_sandbox_retry_state_treats_dot_directory_as_cwd() {
        let cwd = tempfile::tempdir().expect("temp cwd");
        let original = serde_json::json!({"command": "cargo test"});
        let retry_state =
            ShellSandboxRetryState::from_raw_input(&original, cwd.path()).expect("retry state");

        assert!(retry_state.matches_raw_input(
            &serde_json::json!({
                "command": "cargo test",
                "directory": ".",
            }),
            cwd.path(),
        ));
    }

    #[test]
    fn shell_sandbox_retry_state_index_matches_normalized_retry() {
        let cwd = tempfile::tempdir().expect("temp cwd");
        std::fs::create_dir_all(cwd.path().join("crates/app")).expect("test directory");
        let original = serde_json::json!({
            "command": "cargo test",
            "directory": "crates/app",
        });
        let retry_state =
            ShellSandboxRetryState::from_raw_input(&original, cwd.path()).expect("retry state");
        let states = vec![retry_state];

        assert_eq!(
            shell_sandbox_retry_state_index(
                &states,
                &serde_json::json!({
                    "command": "cargo test",
                    "directory": "./crates/app",
                    "sandbox_permissions": "require_escalated",
                }),
                cwd.path(),
            ),
            Some(0)
        );
        assert_eq!(
            shell_sandbox_retry_state_index(
                &states,
                &serde_json::json!({
                    "command": "cargo check",
                    "directory": "./crates/app",
                    "sandbox_permissions": "require_escalated",
                }),
                cwd.path(),
            ),
            None
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
