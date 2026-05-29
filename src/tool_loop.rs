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

use crate::llm_client::{
    ChatMessage, LlmBackend, LlmResponse, StreamChatRequest, TokenUsage, ToolDefinition,
};
use crate::session::{PermissionMode, SessionStore, ToolExchange};
use crate::tools::sandbox::SandboxPolicy;
use crate::tools::{ToolRegistry, ToolStatus, safe_resolve_for_write};

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

    'outer: for turn in 0..max_turns {
        if cancel.is_cancelled() {
            break;
        }

        // For the last turn, don't offer tools -- force a text response
        let turn_tools = if turn < max_turns - 1 {
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
        let response = llm
            .stream_chat(StreamChatRequest {
                model: model.to_string(),
                messages: messages.clone(),
                tools: turn_tools,
                reasoning_effort: reasoning_effort.map(str::to_string),
                on_token,
                on_thought: on_thought_cb,
                cancel: cancel.clone(),
                idle_timeout,
            })
            .await;

        match response {
            Ok(LlmResponse::Text { text, usage }) => {
                turn_usage.add(usage);
                full_response.push_str(&text);
                // Final text response -- we're done
                break;
            }
            Ok(LlmResponse::ToolCalls { text, calls, usage }) => {
                turn_usage.add(usage);
                // Any text emitted before tool calls
                if !text.is_empty() {
                    full_response.push_str(&text);
                }

                // Record the assistant message with tool_calls
                messages.push(ChatMessage::assistant_tool_calls(calls.clone()));

                // Execute each tool call
                for call in &calls {
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
            }
            Err(e) => {
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
struct ToolExecution {
    output: String,
    failed: bool,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn decide(
        mode: PermissionMode,
        kind: ToolKind,
        tool_name: &str,
        allowed: bool,
    ) -> PureGateDecision {
        pure_gate_decision(mode, kind, tool_name, allowed)
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
