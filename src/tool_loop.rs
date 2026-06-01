mod announce;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_client_protocol::schema::{
    Diff, PermissionOption, PermissionOptionId, PermissionOptionKind, RequestPermissionOutcome,
    RequestPermissionRequest, SessionNotification, SessionUpdate, ToolCallId, ToolCallStatus,
    ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use agent_client_protocol::{Client, ConnectionTo};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::bifrost_gate::{
    GateClassifierDecision, GateConfidence, GateContext, RecommendedTool, ShellClassifierDecision,
    ShellStaticRoute, TextStaticRoute, classify_shell_tool_call, classify_text_tool_call,
    encourage_bifrost_enabled, gate_message, shell_gate_message,
    should_skip_for_static_text_target, static_shell_route, static_shell_route_output,
    static_text_route,
};
use crate::llm_client::{
    ChatMessage, LlmBackend, LlmResponse, StreamChatRequest, TokenUsage, ToolCall, ToolDefinition,
};
use crate::session::{PermissionMode, SessionStore, ToolExchange};
use crate::tools::sandbox::SandboxPolicy;
use crate::tools::{ToolRegistry, ToolStatus, safe_resolve_for_write};
use crate::trace_logging::append_trace_record;

const MAX_TOOL_RESULT_BYTES: usize = 50_000;

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
    _tool_name: &str,
    is_always_allowed: bool,
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

    // Mode-independent auto-allow: pure-info kinds never mutate.
    let auto_allow = match kind {
        ToolKind::Read | ToolKind::Search | ToolKind::Think | ToolKind::Fetch => true,
        ToolKind::Edit if matches!(mode, PermissionMode::AcceptEdits) => true,
        _ => false,
    };
    if auto_allow {
        return PureGateDecision::Allow;
    }

    // In-session "Always allow". `consult_gate` chooses the cache key; shell
    // commands use the exact command plus cwd, while regular tools use the
    // tool name.
    if is_always_allowed {
        return PureGateDecision::Allow;
    }

    PureGateDecision::Prompt
}

fn shell_command_will_run_sandboxed(
    permission_mode: PermissionMode,
    sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
) -> bool {
    !matches!(
        SandboxPolicy::resolve(permission_mode, sandbox_mode),
        SandboxPolicy::None
    )
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
    let mut skip_bifrost_classifier_for_next_tool_batch = false;
    let mut no_edit_progress_nudge_count = 0usize;
    'outer: for turn in 0..max_turns {
        if cancel.is_cancelled() {
            break;
        }
        if should_emit_no_edit_progress_nudge(
            turn,
            max_turns,
            &tool_exchanges,
            no_edit_progress_nudge_count,
        ) {
            no_edit_progress_nudge_count += 1;
            append_trace_record(serde_json::json!({
                "type": "no_edit_progress_nudge",
                "turn": turn,
                "nudge_count": no_edit_progress_nudge_count,
                "tool_counts": bifrost_gate_tool_counts(&tool_exchanges),
            }));
            messages.push(ChatMessage::user(
                "You have already retrieved exact source context and no successful file edit/write has happened yet. Stop broad exploration and repeated code retrieval. In this turn, make the smallest plausible code change that addresses the task. Use edit/write_file before any further validation or additional investigation.",
            ));
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

        // Forward tokens straight through to the caller; no buffering.
        let text_clone = on_text.clone();
        let on_token: Box<dyn FnMut(&str) + Send> = Box::new(move |token: &str| {
            if let Ok(mut cb) = text_clone.lock() {
                cb(token);
            }
        });
        // Reasoning deltas route to a parallel sink so the agent layer
        // can emit them as ACP AgentThoughtChunk events; backends that
        // don't surface reasoning simply never invoke this closure.
        let thought_clone = on_thought.clone();
        let on_thought_cb: Box<dyn FnMut(&str) + Send> = Box::new(move |token: &str| {
            if let Ok(mut cb) = thought_clone.lock() {
                cb(token);
            }
        });

        // Wall-clock bound on this stream is enforced by the reqwest client's
        // own `.timeout(...)` (see `OpenAiClient::new`). Per-chunk idle
        // inactivity (the case in #3366 / #3453: streams that drip
        // occasional bytes and would defeat wall-clock) is enforced inside
        // the SSE driver via `idle_timeout`, threaded here from
        // `--llm-idle-timeout-secs` and the per-session `/idle-timeout`
        // override.
        let request_tools = turn_tools.clone();
        trace_llm_request(
            turn,
            model,
            reasoning_effort,
            &messages,
            request_tools.as_ref(),
        );

        let response = llm
            .stream_chat(StreamChatRequest {
                model: model.to_string(),
                messages: messages.clone(),
                tools: request_tools,
                reasoning_effort: reasoning_effort.map(str::to_string),
                on_token,
                on_thought: on_thought_cb,
                cancel: cancel.clone(),
                idle_timeout,
            })
            .await;

        match response {
            Ok(LlmResponse::Text { text, usage }) => {
                trace_llm_text_response(turn, &text, usage);
                turn_usage.add(usage);
                if should_reject_no_edit_final_answer(turn, max_turns, &tool_exchanges) {
                    append_trace_record(serde_json::json!({
                        "type": "no_edit_final_answer_guard",
                        "turn": turn,
                        "tool_counts": bifrost_gate_tool_counts(&tool_exchanges),
                        "text": text,
                    }));
                    messages.push(ChatMessage::assistant(text));
                    messages.push(ChatMessage::user(
                        "The task is not complete: no successful file edit/write has been executed yet. Do not give a final answer. Use the available edit/write tools to implement the required code change, then run focused validation.",
                    ));
                    continue;
                }
                full_response.push_str(&text);
                // Final text response -- we're done
                break;
            }
            Ok(LlmResponse::ToolCalls { text, calls, usage }) => {
                trace_llm_tool_response(turn, &text, &calls, usage);
                turn_usage.add(usage);
                let skip_bifrost_classifier_for_this_tool_batch =
                    skip_bifrost_classifier_for_next_tool_batch;
                let mut current_tool_batch_triggered_bifrost_gate = false;
                let mut bifrost_refreshed_this_tool_batch = false;
                let mut bifrost_refresh_failure: Option<ToolExecution> = None;
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

                    // Refuse outright if the permission card would hide
                    // input. Oversized titles wrap the approval modal and
                    // oversized multiline shell content gets truncated, so in
                    // both cases the user could authorize a call they can't
                    // fully read. Reject instead of hiding details; the LLM can
                    // retry with smaller arguments.
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
                            title_chars = announce::tool_title(&tool_name, &parsed_input)
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

                    // Consult the gate before announcing or executing the call.
                    let decision = consult_gate(
                        &sessions,
                        &session_id,
                        &spawned_cx,
                        &cancel,
                        &tool_name,
                        kind,
                        &call.id,
                        &parsed_input,
                        registry.cwd(),
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
                            } else if let Some(message) = maybe_bifrost_classifier_gate(
                                &tool_name,
                                &parsed_input,
                                &messages,
                                &tools,
                                &tool_exchanges,
                                skip_bifrost_classifier_for_this_tool_batch,
                                &cancel,
                            )
                            .await
                            {
                                current_tool_batch_triggered_bifrost_gate = true;
                                ToolExecution {
                                    output: message,
                                    failed: false,
                                }
                            } else {
                                trace_bifrost_context_shadow(
                                    &tool_name,
                                    &parsed_input,
                                    &tool_exchanges,
                                );
                                if registry.is_bifrost_tool(&tool_name)
                                    && !bifrost_refreshed_this_tool_batch
                                {
                                    tracing::info!(
                                        session_id = %session_id,
                                        tool_name = %tool_name,
                                        "refreshing bifrost before first bifrost tool in batch"
                                    );
                                    append_trace_record(serde_json::json!({
                                        "type": "bifrost_refresh",
                                        "tool": tool_name,
                                        "status": "started",
                                    }));
                                    let refresh = registry.refresh_bifrost().await;
                                    let refresh_status = match refresh.status {
                                        ToolStatus::Success => "success",
                                        ToolStatus::RequestError => "request_error",
                                        ToolStatus::InternalError => "internal_error",
                                    };
                                    tracing::info!(
                                        session_id = %session_id,
                                        tool_name = %tool_name,
                                        status = refresh_status,
                                        "bifrost refresh completed before tool batch"
                                    );
                                    append_trace_record(serde_json::json!({
                                        "type": "bifrost_refresh",
                                        "tool": tool_name,
                                        "status": refresh_status,
                                        "output": refresh.output.clone(),
                                    }));
                                    bifrost_refreshed_this_tool_batch = true;
                                    if !matches!(refresh.status, ToolStatus::Success) {
                                        let exec = tool_result_to_execution(refresh);
                                        bifrost_refresh_failure = Some(exec.clone());
                                        exec
                                    } else {
                                        execute_tool(
                                            registry,
                                            &tool_name,
                                            parsed_input.clone(),
                                            policy,
                                            outside_sandbox_once,
                                            sandbox_mode,
                                        )
                                        .await
                                    }
                                } else if registry.is_bifrost_tool(&tool_name) {
                                    if let Some(exec) = &bifrost_refresh_failure {
                                        exec.clone()
                                    } else {
                                        execute_tool(
                                            registry,
                                            &tool_name,
                                            parsed_input.clone(),
                                            policy,
                                            outside_sandbox_once,
                                            sandbox_mode,
                                        )
                                        .await
                                    }
                                } else {
                                    execute_tool(
                                        registry,
                                        &tool_name,
                                        parsed_input.clone(),
                                        policy,
                                        outside_sandbox_once,
                                        sandbox_mode,
                                    )
                                    .await
                                }
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
                skip_bifrost_classifier_for_next_tool_batch =
                    current_tool_batch_triggered_bifrost_gate;
            }
            Err(e) => {
                trace_llm_error(turn, &e);
                let err_msg = format!("\n**Error:** LLM request failed: {e}\n");
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

/// Apply the per-call permission policy. Returns `Allow` if the tool should
/// execute, or `Reject(msg)` to feed the LLM a denial message instead.
#[allow(clippy::too_many_arguments)]
async fn consult_gate(
    sessions: &SessionStore,
    session_id: &str,
    spawned_cx: &SpawnedCx<'_>,
    cancel: &CancellationToken,
    tool_name: &str,
    kind: ToolKind,
    tool_call_id: &str,
    raw_input: &Value,
    cwd: &Path,
) -> GateDecision {
    let mode = match sessions.permission_mode(session_id).await {
        Some(m) => m,
        None => {
            tracing::warn!(
                session_id,
                tool_name,
                "permission gate: session not found; refusing tool"
            );
            return GateDecision::Reject(
                "Tool use denied: session is no longer registered. \
                 Start a new prompt to continue."
                    .to_string(),
            );
        }
    };
    let sandbox_mode = sessions.sandbox_mode(session_id).await.flatten();
    let shell_sandboxed =
        tool_name == "run_shell_command" && shell_command_will_run_sandboxed(mode, sandbox_mode);
    let always_allow_key = always_allow_key(tool_name, raw_input, cwd, shell_sandboxed);
    let is_always_allowed = sessions
        .is_always_allowed(session_id, &always_allow_key)
        .await;

    match pure_gate_decision(mode, kind, tool_name, is_always_allowed) {
        PureGateDecision::Allow => GateDecision::Allow {
            sandbox_policy_override: None,
            sandbox_mode,
        },
        PureGateDecision::Reject(msg) => GateDecision::Reject(msg),
        PureGateDecision::Prompt => {
            match request_user_permission(
                spawned_cx,
                cancel,
                PermissionRequest {
                    session_id,
                    tool_name,
                    kind,
                    tool_call_id,
                    raw_input,
                    shell_sandboxed,
                },
            )
            .await
            {
                Ok(grant) => {
                    // Awaited inline so the next tool call in the same batch
                    // sees the updated set without re-prompting.
                    if grant.allow_always && grant.sandbox_policy_override.is_none() {
                        sessions
                            .add_always_allow(session_id, &always_allow_key)
                            .await;
                    }
                    GateDecision::Allow {
                        sandbox_policy_override: grant.sandbox_policy_override,
                        sandbox_mode,
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
    // the tool kind. Reuse the same title-builder the standalone tool-call
    // card uses so e.g. ``Run `cargo test` `` appears in the prompt.
    //
    // Assumes the caller has already filtered oversized titles via
    // `announce::rejection_for_oversized_title` in `run`; the debug assert
    // catches any future path that reaches the modal without that gate.
    let title = announce::tool_title(tool_name, raw_input);
    debug_assert!(
        title.chars().count() <= announce::MAX_TOOL_TITLE_CHARS,
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

fn has_tool(tools: &[ToolDefinition], name: &str) -> bool {
    tools.iter().any(|tool| tool.function.name == name)
}

fn is_text_navigation_tool(name: &str) -> bool {
    matches!(name, "read_file" | "grep_search" | "list_directory")
}

async fn maybe_bifrost_classifier_gate(
    tool_name: &str,
    parsed_input: &Value,
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    tool_exchanges: &[ToolExchange],
    skip_after_prior_gate: bool,
    cancel: &CancellationToken,
) -> Option<String> {
    if let Some(reason) = bifrost_classifier_skip_reason(
        tool_name,
        parsed_input,
        tools,
        tool_exchanges,
        skip_after_prior_gate,
    ) {
        let trace_type = if tool_name == "run_shell_command" {
            "shell_gate_classifier_skipped"
        } else {
            "bifrost_gate_classifier_skipped"
        };
        append_trace_record(serde_json::json!({
            "type": trace_type,
            "tool": tool_name,
            "args": parsed_input,
            "reason": reason,
            "prior_tool_counts": bifrost_gate_tool_counts(tool_exchanges),
        }));
        return None;
    }

    let context = GateContext {
        tool_name: tool_name.to_string(),
        args: parsed_input.clone(),
        messages: messages.to_vec(),
        tools: tools.to_vec(),
        tool_exchanges: tool_exchanges.to_vec(),
    };
    if tool_name == "run_shell_command" {
        return maybe_shell_classifier_gate(parsed_input, tools, tool_exchanges, context, cancel)
            .await;
    }

    if let Some(route) = static_text_route(tool_name, parsed_input, tool_exchanges) {
        match route {
            TextStaticRoute::AllowText(reason) => {
                append_trace_record(serde_json::json!({
                    "type": "bifrost_gate_static_route",
                    "tool": tool_name,
                    "args": parsed_input,
                    "route": "allow_text",
                    "reason": reason,
                    "prior_tool_counts": bifrost_gate_tool_counts(tool_exchanges),
                }));
                return None;
            }
        }
    }

    append_trace_record(serde_json::json!({
        "type": "bifrost_gate_classifier_call",
        "tool": tool_name,
        "args": parsed_input,
        "prior_tool_counts": bifrost_gate_tool_counts(tool_exchanges),
    }));

    match classify_text_tool_call(context, cancel).await {
        Ok(output)
            if output.decision == GateClassifierDecision::GateToSymbolTool
                && output.confidence == GateConfidence::High
                && matches!(
                    output.recommended_tool,
                    RecommendedTool::SearchSymbols
                        | RecommendedTool::ScanUsages
                        | RecommendedTool::GetSummaries
                        | RecommendedTool::GetSymbolSources
                ) =>
        {
            let message = gate_message(&output, tools);
            append_trace_record(serde_json::json!({
                "type": "bifrost_gate_classifier",
                "tool": tool_name,
                "args": parsed_input,
                "decision": output,
                "gated": true,
            }));
            Some(message)
        }
        Ok(mut output) => {
            let normalized_recommended_tool = output.decision == GateClassifierDecision::AllowText
                && output.recommended_tool != RecommendedTool::None;
            if normalized_recommended_tool {
                output.recommended_tool = RecommendedTool::None;
                output.suggested_args = serde_json::json!({});
            }
            append_trace_record(serde_json::json!({
                "type": "bifrost_gate_classifier",
                "tool": tool_name,
                "args": parsed_input,
                "decision": output,
                "gated": false,
                "normalized_recommended_tool": normalized_recommended_tool,
            }));
            None
        }
        Err(err) => {
            let error = format!("{err:#}");
            tracing::warn!(tool_name, "Bifrost gate classifier failed open: {error}");
            append_trace_record(serde_json::json!({
                "type": "bifrost_gate_classifier_error",
                "tool": tool_name,
                "args": parsed_input,
                "category": classifier_error_category(&error),
                "error": error,
            }));
            None
        }
    }
}

async fn maybe_shell_classifier_gate(
    parsed_input: &Value,
    tools: &[ToolDefinition],
    tool_exchanges: &[ToolExchange],
    context: GateContext,
    cancel: &CancellationToken,
) -> Option<String> {
    if let Some(route) = static_shell_route(parsed_input) {
        match route {
            ShellStaticRoute::AllowShell(reason) => {
                append_trace_record(serde_json::json!({
                    "type": "shell_gate_static_route",
                    "tool": "run_shell_command",
                    "args": parsed_input,
                    "route": "allow_shell",
                    "reason": reason,
                    "prior_tool_counts": bifrost_gate_tool_counts(tool_exchanges),
                }));
                return None;
            }
            ShellStaticRoute::UseBuiltin(reason, recommended_tool) => {
                let output = static_shell_route_output(
                    reason,
                    ShellClassifierDecision::UseBuiltinTool,
                    recommended_tool,
                );
                let message = shell_gate_message(&output, tools);
                append_trace_record(serde_json::json!({
                    "type": "shell_gate_static_route",
                    "tool": "run_shell_command",
                    "args": parsed_input,
                    "route": "use_builtin_tool",
                    "reason": reason,
                    "decision": output,
                    "prior_tool_counts": bifrost_gate_tool_counts(tool_exchanges),
                }));
                return Some(message);
            }
        }
    }

    append_trace_record(serde_json::json!({
        "type": "shell_gate_classifier_call",
        "tool": "run_shell_command",
        "args": parsed_input,
        "prior_tool_counts": bifrost_gate_tool_counts(tool_exchanges),
    }));

    match classify_shell_tool_call(context, cancel).await {
        Ok(output)
            if matches!(
                output.decision,
                ShellClassifierDecision::UseBuiltinTool | ShellClassifierDecision::UseBifrostTool
            ) && output.confidence == GateConfidence::High =>
        {
            let message = shell_gate_message(&output, tools);
            append_trace_record(serde_json::json!({
                "type": "shell_gate_classifier",
                "tool": "run_shell_command",
                "args": parsed_input,
                "decision": output,
                "gated": true,
            }));
            Some(message)
        }
        Ok(mut output) => {
            let normalized_recommended_tool = output.decision
                == ShellClassifierDecision::AllowShell
                && output.recommended_tool != RecommendedTool::None;
            if normalized_recommended_tool {
                output.recommended_tool = RecommendedTool::None;
                output.suggested_args = serde_json::json!({});
            }
            append_trace_record(serde_json::json!({
                "type": "shell_gate_classifier",
                "tool": "run_shell_command",
                "args": parsed_input,
                "decision": output,
                "gated": false,
                "normalized_recommended_tool": normalized_recommended_tool,
            }));
            None
        }
        Err(err) => {
            let error = format!("{err:#}");
            tracing::warn!("Shell routing classifier failed open: {error}");
            append_trace_record(serde_json::json!({
                "type": "shell_gate_classifier_error",
                "tool": "run_shell_command",
                "args": parsed_input,
                "category": classifier_error_category(&error),
                "error": error,
            }));
            None
        }
    }
}

fn classifier_error_category(error: &str) -> &'static str {
    let lower = error.to_ascii_lowercase();
    if lower.contains("http") {
        "http"
    } else if lower.contains("timeout") || lower.contains("timed out") {
        "timeout"
    } else if lower.contains("missing choices") || lower.contains("missing content") {
        "missing_content"
    } else if lower.contains("parsing") || lower.contains("json") || lower.contains("schema") {
        "parse_or_schema"
    } else if lower.contains("cancelled") {
        "cancelled"
    } else if lower.contains("credential") || lower.contains("api key") {
        "auth"
    } else if lower.contains("sending classifier request")
        || lower.contains("dns")
        || lower.contains("connect")
    {
        "transport"
    } else {
        "other"
    }
}

#[cfg(test)]
fn should_consult_bifrost_classifier(
    tool_name: &str,
    parsed_input: &Value,
    tools: &[ToolDefinition],
    tool_exchanges: &[ToolExchange],
    skip_after_prior_gate: bool,
) -> bool {
    bifrost_classifier_skip_reason(
        tool_name,
        parsed_input,
        tools,
        tool_exchanges,
        skip_after_prior_gate,
    )
    .is_none()
}

fn bifrost_classifier_skip_reason(
    tool_name: &str,
    parsed_input: &Value,
    tools: &[ToolDefinition],
    tool_exchanges: &[ToolExchange],
    skip_after_prior_gate: bool,
) -> Option<&'static str> {
    if !encourage_bifrost_enabled() {
        return Some("bifrost_encouragement_disabled");
    }
    if tool_name == "run_shell_command" {
        return shell_classifier_skip_reason(tool_name, tools, skip_after_prior_gate);
    } else if !is_text_navigation_tool(tool_name) {
        return Some("not_text_navigation_tool");
    }
    if skip_after_prior_gate {
        return Some("post_gate_tool_batch");
    }
    if tool_name == "list_directory" {
        return Some("list_directory_default_allow");
    }
    if is_text_navigation_tool(tool_name)
        && should_skip_for_static_text_target(tool_name, parsed_input)
    {
        return Some("static_text_target");
    }
    if tool_name == "read_file" && has_targeted_recent_bifrost_miss(parsed_input, tool_exchanges) {
        return Some("targeted_bifrost_miss_fallback");
    }
    if tool_name == "grep_search"
        && has_targeted_recent_bifrost_miss_for_grep(parsed_input, tool_exchanges)
    {
        return Some("targeted_bifrost_miss_fallback");
    }
    if !has_tool(tools, "search_symbols")
        || !has_tool(tools, "scan_usages")
        || !has_tool(tools, "get_summaries")
    {
        return Some("missing_required_bifrost_tools");
    }
    None
}

fn has_targeted_recent_bifrost_miss(parsed_input: &Value, tool_exchanges: &[ToolExchange]) -> bool {
    let Some(path) = parsed_input.get("file_path").and_then(Value::as_str) else {
        return false;
    };
    if path.len() < 3 {
        return false;
    }
    tool_exchanges.iter().rev().take(8).any(|exchange| {
        matches!(
            exchange.tool_name.as_str(),
            "search_symbols" | "scan_usages" | "get_summaries" | "get_symbol_sources"
        ) && tool_result_looks_like_bifrost_miss(&exchange.result)
            && (exchange.arguments.contains(path) || exchange.result.contains(path))
    })
}

fn has_targeted_recent_bifrost_miss_for_grep(
    parsed_input: &Value,
    tool_exchanges: &[ToolExchange],
) -> bool {
    let Some(pattern) = parsed_input.get("pattern").and_then(Value::as_str) else {
        return false;
    };
    let tokens = identifier_terms(pattern);
    if tokens.is_empty() {
        return false;
    }
    tool_exchanges.iter().rev().take(8).any(|exchange| {
        matches!(
            exchange.tool_name.as_str(),
            "search_symbols" | "scan_usages" | "get_summaries" | "get_symbol_sources"
        ) && tool_result_looks_like_bifrost_miss(&exchange.result)
            && tokens
                .iter()
                .any(|token| exchange.arguments.contains(token) || exchange.result.contains(token))
    })
}

fn identifier_terms(pattern: &str) -> Vec<String> {
    pattern
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .filter(|token| token.len() >= 3)
        .take(12)
        .map(str::to_string)
        .collect()
}

fn tool_result_looks_like_bifrost_miss(result: &str) -> bool {
    let trimmed = result.trim();
    if trimmed.is_empty() || trimmed == "[]" || trimmed == "{}" {
        return true;
    }
    let lower = trimmed.to_ascii_lowercase();
    [
        "not_found",
        "not found",
        "no match",
        "no matches",
        "no symbol",
        "no symbols",
        "no usages",
        "no references",
        "empty result",
        "returned empty",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn shell_classifier_skip_reason(
    tool_name: &str,
    _tools: &[ToolDefinition],
    skip_after_prior_gate: bool,
) -> Option<&'static str> {
    if tool_name != "run_shell_command" {
        return Some("not_shell_tool");
    }
    if skip_after_prior_gate {
        return Some("post_gate_tool_batch");
    }
    None
}

fn bifrost_gate_tool_counts(tool_exchanges: &[ToolExchange]) -> Value {
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
        "tool_counts": bifrost_gate_tool_counts(tool_exchanges),
    }));
}

fn should_reject_no_edit_final_answer(
    turn: usize,
    max_turns: usize,
    tool_exchanges: &[ToolExchange],
) -> bool {
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

fn should_emit_no_edit_progress_nudge(
    turn: usize,
    max_turns: usize,
    tool_exchanges: &[ToolExchange],
    nudge_count: usize,
) -> bool {
    if nudge_count >= 2 || has_successful_file_change(tool_exchanges) {
        return false;
    }
    let first_nudge_turn = (max_turns / 3).clamp(6, 10);
    let next_nudge_turn = first_nudge_turn + 4 * nudge_count;
    turn >= next_nudge_turn
        && exact_source_tool_count(tool_exchanges) >= 1
        && code_context_tool_count(tool_exchanges) >= 5
}

fn code_context_tool_count(tool_exchanges: &[ToolExchange]) -> usize {
    tool_exchanges
        .iter()
        .filter(|exchange| {
            matches!(
                exchange.tool_name.as_str(),
                "search_symbols"
                    | "get_symbol_sources"
                    | "scan_usages"
                    | "get_summaries"
                    | "list_symbols"
                    | "get_symbol_locations"
                    | "read_file"
                    | "grep_search"
                    | "list_directory"
            )
        })
        .count()
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

    fn decide(
        mode: PermissionMode,
        kind: ToolKind,
        tool_name: &str,
        allowed: bool,
    ) -> PureGateDecision {
        pure_gate_decision(mode, kind, tool_name, allowed)
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
    fn bifrost_classifier_is_skipped_immediately_after_gate_prompt() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let _scope = crate::openrouter_auth::test_support::EnvScope::set(
            crate::bifrost_gate::ENCOURAGE_BIFROST_ENV,
            "1",
        );
        let tools = vec![
            tool_def_for_test("read_file"),
            tool_def_for_test("grep_search"),
            tool_def_for_test("search_symbols"),
            tool_def_for_test("scan_usages"),
            tool_def_for_test("get_summaries"),
        ];

        let should_consult = should_consult_bifrost_classifier(
            "read_file",
            &serde_json::json!({"file_path": "src/lib.rs"}),
            &tools,
            &[],
            true,
        );

        assert!(!should_consult);
        assert_eq!(
            bifrost_classifier_skip_reason(
                "read_file",
                &serde_json::json!({"file_path": "src/lib.rs"}),
                &tools,
                &[],
                true,
            ),
            Some("post_gate_tool_batch")
        );
    }

    #[test]
    fn bifrost_classifier_is_disabled_by_default_without_env_var() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let _scope = crate::openrouter_auth::test_support::EnvScope::remove(
            crate::bifrost_gate::ENCOURAGE_BIFROST_ENV,
        );
        let tools = vec![
            tool_def_for_test("read_file"),
            tool_def_for_test("grep_search"),
            tool_def_for_test("search_symbols"),
            tool_def_for_test("scan_usages"),
            tool_def_for_test("get_summaries"),
        ];

        assert_eq!(
            bifrost_classifier_skip_reason(
                "read_file",
                &serde_json::json!({"file_path": "src/lib.rs"}),
                &tools,
                &[],
                false,
            ),
            Some("bifrost_encouragement_disabled")
        );
        assert!(!should_consult_bifrost_classifier(
            "run_shell_command",
            &serde_json::json!({"command": "cargo test -q"}),
            &tools,
            &[],
            false,
        ));
    }

    #[test]
    fn bifrost_classifier_can_be_enabled_with_env_var() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let _scope = crate::openrouter_auth::test_support::EnvScope::set(
            crate::bifrost_gate::ENCOURAGE_BIFROST_ENV,
            "1",
        );
        let tools = vec![
            tool_def_for_test("read_file"),
            tool_def_for_test("grep_search"),
            tool_def_for_test("search_symbols"),
            tool_def_for_test("scan_usages"),
            tool_def_for_test("get_summaries"),
        ];

        assert!(should_consult_bifrost_classifier(
            "read_file",
            &serde_json::json!({"file_path": "src/lib.rs"}),
            &tools,
            &[],
            false,
        ));
    }

    #[test]
    fn list_directory_always_skips_bifrost_classifier() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let _scope = crate::openrouter_auth::test_support::EnvScope::set(
            crate::bifrost_gate::ENCOURAGE_BIFROST_ENV,
            "1",
        );
        let tools = vec![
            tool_def_for_test("list_directory"),
            tool_def_for_test("search_symbols"),
            tool_def_for_test("scan_usages"),
            tool_def_for_test("get_summaries"),
        ];

        assert_eq!(
            bifrost_classifier_skip_reason(
                "list_directory",
                &serde_json::json!({"path": "src"}),
                &tools,
                &[],
                false,
            ),
            Some("list_directory_default_allow")
        );
    }

    #[test]
    fn read_file_allows_targeted_bifrost_miss_fallback() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let _scope = crate::openrouter_auth::test_support::EnvScope::set(
            crate::bifrost_gate::ENCOURAGE_BIFROST_ENV,
            "1",
        );
        let tools = vec![
            tool_def_for_test("read_file"),
            tool_def_for_test("search_symbols"),
            tool_def_for_test("scan_usages"),
            tool_def_for_test("get_summaries"),
        ];
        let mut miss = exchange_for_test("search_symbols");
        miss.arguments = "{\"patterns\":[\"src/lib.rs\"]}".to_string();
        miss.result = "No symbols found for src/lib.rs".to_string();

        assert_eq!(
            bifrost_classifier_skip_reason(
                "read_file",
                &serde_json::json!({"file_path": "src/lib.rs"}),
                &tools,
                &[miss],
                false,
            ),
            Some("targeted_bifrost_miss_fallback")
        );
    }

    #[test]
    fn grep_search_allows_targeted_bifrost_miss_fallback() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let _scope = crate::openrouter_auth::test_support::EnvScope::set(
            crate::bifrost_gate::ENCOURAGE_BIFROST_ENV,
            "1",
        );
        let tools = vec![
            tool_def_for_test("grep_search"),
            tool_def_for_test("search_symbols"),
            tool_def_for_test("scan_usages"),
            tool_def_for_test("get_summaries"),
        ];
        let mut miss = exchange_for_test("search_symbols");
        miss.arguments = "{\"patterns\":[\"RenameFileAsync\"]}".to_string();
        miss.result = "No symbols found for RenameFileAsync".to_string();

        assert_eq!(
            bifrost_classifier_skip_reason(
                "grep_search",
                &serde_json::json!({"glob": "**/*.cs", "pattern": "RenameFileAsync"}),
                &tools,
                &[miss],
                false,
            ),
            Some("targeted_bifrost_miss_fallback")
        );
    }

    #[test]
    fn no_edit_final_guard_rejects_navigation_only_final_before_last_turn() {
        let prior = vec![exchange_for_test("search_symbols")];

        assert!(should_reject_no_edit_final_answer(3, 10, &prior));
    }

    #[test]
    fn no_edit_final_guard_allows_after_successful_edit() {
        let prior = vec![ToolExchange {
            call_id: "edit".to_string(),
            tool_name: "edit".to_string(),
            arguments: "{}".to_string(),
            result: "Edited 'src/lib.rs' (1 replacement)".to_string(),
        }];

        assert!(!should_reject_no_edit_final_answer(3, 10, &prior));
        assert!(has_successful_file_change(&prior));
    }

    #[test]
    fn no_edit_final_guard_does_not_reject_on_last_turn() {
        let prior = vec![exchange_for_test("search_symbols")];

        assert!(!should_reject_no_edit_final_answer(9, 10, &prior));
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

        assert!(should_emit_no_edit_progress_nudge(8, 25, &prior, 0));
        assert!(!should_emit_no_edit_progress_nudge(7, 25, &prior, 0));
        assert!(!should_emit_no_edit_progress_nudge(8, 25, &prior, 2));
    }

    #[test]
    fn no_edit_progress_nudge_waits_for_exact_source() {
        let prior = vec![
            exchange_for_test("search_symbols"),
            exchange_for_test("scan_usages"),
            exchange_for_test("get_summaries"),
            exchange_for_test("search_symbols"),
            exchange_for_test("read_file"),
        ];

        assert!(!should_emit_no_edit_progress_nudge(8, 25, &prior, 0));
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

        assert!(!should_emit_no_edit_progress_nudge(12, 25, &prior, 0));
    }

    #[test]
    fn bifrost_classifier_runs_when_not_in_post_gate_batch() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let _scope = crate::openrouter_auth::test_support::EnvScope::set(
            crate::bifrost_gate::ENCOURAGE_BIFROST_ENV,
            "1",
        );
        let tools = vec![
            tool_def_for_test("read_file"),
            tool_def_for_test("grep_search"),
            tool_def_for_test("search_symbols"),
            tool_def_for_test("scan_usages"),
            tool_def_for_test("get_summaries"),
        ];

        let should_consult = should_consult_bifrost_classifier(
            "read_file",
            &serde_json::json!({"file_path": "src/lib.rs"}),
            &tools,
            &[],
            false,
        );

        assert!(should_consult);
    }

    #[test]
    fn shell_classifier_runs_for_all_shell_commands() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let _scope = crate::openrouter_auth::test_support::EnvScope::set(
            crate::bifrost_gate::ENCOURAGE_BIFROST_ENV,
            "1",
        );
        let tools = vec![
            tool_def_for_test("read_file"),
            tool_def_for_test("grep_search"),
            tool_def_for_test("run_shell_command"),
            tool_def_for_test("search_symbols"),
            tool_def_for_test("scan_usages"),
            tool_def_for_test("get_summaries"),
        ];

        assert!(should_consult_bifrost_classifier(
            "run_shell_command",
            &serde_json::json!({"command": "nl -ba src/main.rs | sed -n '1,80p'"}),
            &tools,
            &[],
            false,
        ));
        assert!(should_consult_bifrost_classifier(
            "run_shell_command",
            &serde_json::json!({"command": "cargo test -q"}),
            &tools,
            &[],
            false,
        ));
        assert!(should_consult_bifrost_classifier(
            "run_shell_command",
            &serde_json::json!({"command": "python3 - <<'PY'\nfrom pathlib import Path\nPath('src/Foo.php').write_text('x')\nPY"}),
            &tools,
            &[],
            false,
        ));
        assert!(should_consult_bifrost_classifier(
            "run_shell_command",
            &serde_json::json!({"command": "sed -i 's/a/b/' src/Foo.php"}),
            &tools,
            &[],
            false,
        ));
        assert!(!should_consult_bifrost_classifier(
            "run_shell_command",
            &serde_json::json!({"command": "cargo test -q"}),
            &tools,
            &[],
            true,
        ));
        assert_eq!(
            bifrost_classifier_skip_reason(
                "run_shell_command",
                &serde_json::json!({"command": "cargo test -q"}),
                &tools,
                &[],
                true,
            ),
            Some("post_gate_tool_batch")
        );
    }

    #[test]
    fn bifrost_classifier_skip_reasons_are_specific() {
        let _lock = crate::openrouter_auth::test_support::ENV_GUARD.blocking_lock();
        let _scope = crate::openrouter_auth::test_support::EnvScope::set(
            crate::bifrost_gate::ENCOURAGE_BIFROST_ENV,
            "1",
        );
        let tools = vec![
            tool_def_for_test("read_file"),
            tool_def_for_test("grep_search"),
            tool_def_for_test("search_symbols"),
            tool_def_for_test("scan_usages"),
            tool_def_for_test("get_summaries"),
        ];

        assert_eq!(
            bifrost_classifier_skip_reason(
                "edit",
                &serde_json::json!({"file_path": "src/lib.rs"}),
                &tools,
                &[],
                false,
            ),
            Some("not_text_navigation_tool")
        );
        assert_eq!(
            bifrost_classifier_skip_reason(
                "read_file",
                &serde_json::json!({"file_path": ".harness/build.sh"}),
                &tools,
                &[],
                false,
            ),
            Some("static_text_target")
        );
        assert_eq!(
            bifrost_classifier_skip_reason(
                "read_file",
                &serde_json::json!({"file_path": "src/lib.rs"}),
                &tools,
                &[
                    exchange_for_test("search_symbols"),
                    exchange_for_test("scan_usages"),
                    exchange_for_test("get_summaries"),
                ],
                false,
            ),
            None
        );
        assert_eq!(
            bifrost_classifier_skip_reason(
                "read_file",
                &serde_json::json!({"file_path": "src/lib.rs"}),
                &[tool_def_for_test("read_file")],
                &[],
                false,
            ),
            Some("missing_required_bifrost_tools")
        );
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
                decide(PermissionMode::BypassPermissions, kind, "anything", false),
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
                decide(PermissionMode::ReadOnly, kind, "anything", false),
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
                    decide(PermissionMode::ReadOnly, kind, "any", true),
                    PureGateDecision::Reject(_)
                ),
                "read-only should reject {:?} even when always-allowed",
                kind
            );
        }
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
                decide(PermissionMode::Default, kind, "anything", false),
                PureGateDecision::Allow
            );
        }
    }

    #[test]
    fn default_prompts_for_edit_without_always_allow() {
        assert_eq!(
            decide(PermissionMode::Default, ToolKind::Edit, "write_file", false),
            PureGateDecision::Prompt
        );
    }

    #[test]
    fn default_uses_always_allow_for_edit() {
        assert_eq!(
            decide(PermissionMode::Default, ToolKind::Edit, "write_file", true),
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
                decide(mode, ToolKind::Execute, "run_shell_command", true),
                PureGateDecision::Allow,
                "run_shell_command should honor scoped approval in {:?}",
                mode
            );
        }
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
