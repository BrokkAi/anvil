//! Minimal headless ACP client behind `anvil --print` (#356).
//!
//! Connects an in-process one-shot ACP client to the same agent component
//! `run_agent` serves over stdio, runs exactly one prompt, prints the result
//! in one of three formats, and exits. The CLI surface and output contract
//! deliberately mirror Mjolnir's `mj --print` so existing headless scripts
//! port over with a binary-name change.
//!
//! This is a *client*, not a second orchestrator: no TUI, no session picker,
//! no interactive permission prompts, no subagent supervision. Permission
//! requests are answered from a fixed policy so a run can never hang on a
//! human.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    ClientCapabilities, ContentBlock, FileSystemCapabilities, Implementation, InitializeRequest,
    Meta, NewSessionRequest, PermissionOption, PermissionOptionKind, PromptRequest,
    RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    ResumeSessionRequest, SelectedPermissionOutcome, SessionNotification, SessionUpdate,
    SetSessionConfigOptionRequest, StopReason, TextContent, ToolCallStatus, ToolKind,
    Usage as AcpUsage,
};
use agent_client_protocol::{Agent, Client, ConnectionTo};
use anyhow::{Result, anyhow, bail};
use serde::Serialize;

use crate::multi_backend::MultiBackend;
use crate::session::SessionStore;

/// Effort suffixes accepted in `--model MODEL[+EFFORT]`. Matches Mjolnir's
/// list; `none` canonicalizes to `off` (Anvil's spelling for "no provider
/// reasoning controls").
const KNOWN_REASONING_EFFORTS: &[&str] = &[
    "off", "none", "minimal", "low", "medium", "high", "xhigh", "max",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    /// Final assistant message on stdout, nothing else.
    Text,
    /// One JSON object at exit.
    Json,
    /// Newline-delimited JSON records as they happen, `result` last.
    StreamJson,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum PermissionMode {
    /// Reject every permission request while honoring agent-side read-only
    /// auto-approvals and remembered repo-scoped Always allow grants.
    #[value(alias = "default")]
    Manual,
    /// Accept edit/delete/move requests; reject shell execution.
    #[value(alias = "acceptEdits", alias = "accept-edits")]
    Auto,
    /// Accept every permission request.
    #[value(alias = "bypassPermissions", alias = "bypass-permissions")]
    Yolo,
}

impl PermissionMode {
    /// Value pushed to the agent-side `permission_mode` session config.
    /// Always applied: Anvil's own session default is `auto`, whose LLM
    /// classifier decides permissions without ever consulting the client, so
    /// leaving it in place would break the documented headless table.
    ///
    /// - `manual` → `default`: the agent honors read-only auto-approvals and
    ///   remembered Always allow grants; the client-side policy rejects every
    ///   request that reaches it.
    /// - `auto` → `acceptEdits`: edits are auto-approved agent-side;
    ///   delete/move prompts are approved and shell execution rejected by
    ///   the client-side policy — together exactly "accept edit/delete/move,
    ///   reject shell".
    /// - `yolo` → `bypassPermissions`: nothing prompts.
    ///
    /// The client-side answering policy in [`answer_permission`] stays active
    /// as the backstop that guarantees headless runs never block.
    fn agent_config_value(self) -> &'static str {
        match self {
            PermissionMode::Manual => "default",
            PermissionMode::Auto => "acceptEdits",
            PermissionMode::Yolo => "bypassPermissions",
        }
    }
}

/// Parse `--model MODEL[+EFFORT]`. Only the last `+` is considered, and only
/// when the suffix is a known effort token — so model ids containing `+`
/// still round-trip unless their tail collides with an effort name.
pub fn parse_model_override(raw: &str) -> Result<(String, Option<String>), String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err("--model requires a non-empty model id".to_string());
    }
    let (model, effort) = split_model_effort(raw);
    if model.is_empty() {
        return Err(format!(
            "--model '{raw}' has an effort suffix but no model id"
        ));
    }
    Ok((model.to_string(), effort))
}

fn split_model_effort(value: &str) -> (&str, Option<String>) {
    let Some(idx) = value.rfind('+') else {
        return (value, None);
    };
    let (model, suffix) = value.split_at(idx);
    let suffix = &suffix[1..];
    let lower = suffix.to_ascii_lowercase();
    if !KNOWN_REASONING_EFFORTS.contains(&lower.as_str()) {
        return (value, None);
    }
    let effort = if lower == "none" {
        "off".to_string()
    } else {
        lower
    };
    (model, Some(effort))
}

pub struct RunConfig {
    /// Raw `--print` value; `-` (the default when the flag has no value)
    /// reads the prompt from stdin.
    pub prompt_arg: String,
    pub output_format: OutputFormat,
    pub permission_mode: PermissionMode,
    pub cwd: Option<PathBuf>,
    pub model: Option<(String, Option<String>)>,
    pub resume: Option<String>,
}

/// Everything the final payload needs, accumulated across notification
/// handlers and the driver task.
#[derive(Default)]
struct RunState {
    session_id: Option<String>,
    resumed: bool,
    /// Assistant prose after the last tool-call/plan boundary — the "final
    /// assistant message". Chunks before the prompt is sent (delayed setup
    /// notices from session/new) are dropped.
    final_text: String,
    prompt_sent: bool,
    stop_reason: Option<StopReason>,
    usage: Option<AcpUsage>,
    /// `_meta.anvil.turnFailure.message` from the usage update: Anvil
    /// reports LLM turn failures there while still ending the turn with
    /// `end_turn`, so this is the only reliable failure signal.
    turn_failure: Option<String>,
    error: Option<String>,
}

/// Newline-delimited records for `--output-format stream-json`. Internally
/// tagged with `type`; variant names snake_case. Field shapes follow
/// Mjolnir's `StreamRecord` (the reference contract from #356), with the
/// `result` record carrying exactly the same payload as `--output-format
/// json` plus the tag.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StreamRecord<'a> {
    Connected {
        agent_name: Option<&'a str>,
        agent_version: Option<&'a str>,
    },
    SessionStarted {
        session_id: &'a str,
        resumed: bool,
    },
    AgentMessage {
        actor: &'a str,
        text: &'a str,
    },
    AgentThought {
        actor: &'a str,
        text: &'a str,
    },
    ToolCall {
        actor: &'a str,
        id: &'a str,
        title: &'a str,
        kind: &'static str,
        status: &'static str,
    },
    ToolCallUpdate {
        actor: &'a str,
        id: &'a str,
        title: Option<&'a str>,
        kind: Option<&'static str>,
        status: Option<&'static str>,
    },
    /// `decision` is the selected permission option id (`allow`,
    /// `allow_outside_sandbox`, `reject`, ...) or `cancelled` when no
    /// matching option existed and the request was cancelled instead.
    Permission {
        actor: &'a str,
        tool_call_id: &'a str,
        decision: &'a str,
    },
    Warning {
        message: &'a str,
    },
    Error {
        message: &'a str,
    },
    Result {
        session_id: Option<&'a str>,
        resumed: bool,
        result: &'a str,
        stop_reason: &'a str,
        usage: Option<&'a AcpUsage>,
        error: Option<&'a str>,
    },
}

/// The single object `--output-format json` prints at exit (#356 shape).
#[derive(Serialize)]
struct JsonResult<'a> {
    session_id: Option<&'a str>,
    resumed: bool,
    result: &'a str,
    stop_reason: &'a str,
    usage: Option<&'a AcpUsage>,
    error: Option<&'a str>,
}

const PRIMARY_ACTOR: &str = "primary";

fn emit_json<T: Serialize>(value: &T) {
    match serde_json::to_string(value) {
        Ok(line) => println!("{line}"),
        Err(err) => eprintln!("warning: failed to serialize stream record: {err}"),
    }
}

/// Warn on stderr for `text`/`json` (stdout is reserved for the answer /
/// the single object), as a `warning` record on stdout for `stream-json`.
fn emit_warning(format: OutputFormat, message: &str) {
    if format == OutputFormat::StreamJson {
        emit_json(&StreamRecord::Warning { message });
    } else {
        eprintln!("warning: {message}");
    }
}

fn stop_reason_label(reason: StopReason) -> &'static str {
    match reason {
        StopReason::EndTurn => "end_turn",
        StopReason::MaxTokens => "max_tokens",
        StopReason::MaxTurnRequests => "max_turn_requests",
        StopReason::Refusal => "refusal",
        StopReason::Cancelled => "cancelled",
        _ => "other",
    }
}

fn tool_kind_label(kind: ToolKind) -> &'static str {
    match kind {
        ToolKind::Read => "read",
        ToolKind::Edit => "edit",
        ToolKind::Delete => "delete",
        ToolKind::Move => "move",
        ToolKind::Search => "search",
        ToolKind::Execute => "execute",
        ToolKind::Think => "think",
        ToolKind::Fetch => "fetch",
        ToolKind::SwitchMode => "switch_mode",
        _ => "other",
    }
}

fn tool_status_label(status: ToolCallStatus) -> &'static str {
    match status {
        ToolCallStatus::Pending => "pending",
        ToolCallStatus::InProgress => "in_progress",
        ToolCallStatus::Completed => "completed",
        ToolCallStatus::Failed => "failed",
        _ => "other",
    }
}

fn content_block_text(block: &ContentBlock) -> String {
    match block {
        ContentBlock::Text(text) => text.text.clone(),
        ContentBlock::Image(_) => "[image]".to_string(),
        ContentBlock::Audio(_) => "[audio]".to_string(),
        ContentBlock::ResourceLink(link) => format!("[link {}]", link.uri),
        ContentBlock::Resource(_) => "[resource]".to_string(),
        _ => "[unknown content]".to_string(),
    }
}

/// Read the prompt: a literal argument, or all of stdin for `-` (which is
/// also the `default_missing_value` for a bare `--print`).
fn resolve_prompt(prompt_arg: &str) -> Result<String> {
    if prompt_arg != "-" {
        return Ok(prompt_arg.to_string());
    }
    let mut prompt = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut prompt)
        .map_err(|err| anyhow!("failed to read prompt from stdin: {err}"))?;
    Ok(prompt)
}

fn resolve_cwd(cwd: Option<PathBuf>) -> Result<PathBuf> {
    match cwd {
        Some(path) => std::path::absolute(&path)
            .map_err(|err| anyhow!("failed to resolve --cwd {}: {err}", path.display())),
        None => std::env::current_dir().map_err(|err| anyhow!("failed to resolve cwd: {err}")),
    }
}

/// Decide a permission request from the fixed headless policy. Returns the
/// selected option id, or `None` to cancel the request (the agent treats a
/// cancelled prompt as a denial and reports it to the model as the tool
/// result, so the turn continues either way).
fn permission_decision(
    mode: PermissionMode,
    kind: Option<ToolKind>,
    options: &[PermissionOption],
) -> Option<&PermissionOption> {
    let allow = match mode {
        PermissionMode::Manual => false,
        PermissionMode::Yolo => true,
        PermissionMode::Auto => matches!(
            kind,
            Some(ToolKind::Edit | ToolKind::Delete | ToolKind::Move)
        ),
    };
    if allow {
        // Prefer AllowOnce over AllowAlways: Anvil persists AllowAlways
        // approvals into the repo's .brokk/permissions.json, and a one-shot
        // headless run must not leave durable grants behind.
        options
            .iter()
            .find(|option| option.kind == PermissionOptionKind::AllowOnce)
            .or_else(|| {
                options
                    .iter()
                    .find(|option| option.kind == PermissionOptionKind::AllowAlways)
            })
    } else {
        // Prefer an explicit reject option ("Tool use denied by user.") over
        // cancelling; RejectAlways is avoided for the same durability reason.
        options
            .iter()
            .find(|option| option.kind == PermissionOptionKind::RejectOnce)
    }
}

fn answer_permission(
    request: &RequestPermissionRequest,
    mode: PermissionMode,
    format: OutputFormat,
) -> RequestPermissionResponse {
    let selected = permission_decision(mode, request.tool_call.fields.kind, &request.options);
    if format == OutputFormat::StreamJson {
        let tool_call_id = request.tool_call.tool_call_id.to_string();
        let decision = selected
            .map(|option| option.option_id.to_string())
            .unwrap_or_else(|| "cancelled".to_string());
        emit_json(&StreamRecord::Permission {
            actor: PRIMARY_ACTOR,
            tool_call_id: &tool_call_id,
            decision: &decision,
        });
    }
    match selected {
        Some(option) => RequestPermissionResponse::new(RequestPermissionOutcome::Selected(
            SelectedPermissionOutcome::new(option.option_id.clone()),
        )),
        None => RequestPermissionResponse::new(RequestPermissionOutcome::Cancelled),
    }
}

fn turn_failure_message(meta: Option<&Meta>) -> Option<String> {
    meta?
        .get(crate::structured_output::ACP_META_NAMESPACE)?
        .get("turnFailure")?
        .get("message")?
        .as_str()
        .map(str::to_string)
}

fn handle_session_update(update: SessionUpdate, state: &Mutex<RunState>, format: OutputFormat) {
    let stream = format == OutputFormat::StreamJson;
    match update {
        SessionUpdate::AgentMessageChunk(chunk) => {
            let text = content_block_text(&chunk.content);
            if stream {
                emit_json(&StreamRecord::AgentMessage {
                    actor: PRIMARY_ACTOR,
                    text: &text,
                });
            }
            let mut state = state.lock().expect("headless state lock poisoned");
            if state.prompt_sent {
                state.final_text.push_str(&text);
            }
        }
        SessionUpdate::AgentThoughtChunk(chunk) if stream => {
            let text = content_block_text(&chunk.content);
            emit_json(&StreamRecord::AgentThought {
                actor: PRIMARY_ACTOR,
                text: &text,
            });
        }
        SessionUpdate::ToolCall(call) => {
            if stream {
                emit_json(&StreamRecord::ToolCall {
                    actor: PRIMARY_ACTOR,
                    id: &call.tool_call_id.to_string(),
                    title: &call.title,
                    kind: tool_kind_label(call.kind),
                    status: tool_status_label(call.status),
                });
            }
            // A tool call starts a new agent-message segment: everything
            // streamed before it was preamble, and the final assistant
            // message is whatever prose follows the last tool activity.
            state
                .lock()
                .expect("headless state lock poisoned")
                .final_text
                .clear();
        }
        SessionUpdate::ToolCallUpdate(update) if stream => {
            emit_json(&StreamRecord::ToolCallUpdate {
                actor: PRIMARY_ACTOR,
                id: &update.tool_call_id.to_string(),
                title: update.fields.title.as_deref(),
                kind: update.fields.kind.map(tool_kind_label),
                status: update.fields.status.map(tool_status_label),
            });
        }
        SessionUpdate::Plan(_) => {
            state
                .lock()
                .expect("headless state lock poisoned")
                .final_text
                .clear();
        }
        SessionUpdate::UsageUpdate(usage) => {
            let mut state = state.lock().expect("headless state lock poisoned");
            if state.prompt_sent
                && let Some(message) = turn_failure_message(usage.meta.as_ref())
            {
                state.turn_failure = Some(message);
            }
        }
        _ => {}
    }
}

async fn set_config(
    connection: &ConnectionTo<Agent>,
    session_id: &str,
    key: &str,
    value: &str,
) -> agent_client_protocol::Result<()> {
    connection
        .send_request(SetSessionConfigOptionRequest::new(
            session_id.to_string(),
            key.to_string(),
            value,
        ))
        .block_task()
        .await?;
    Ok(())
}

/// The client's main task: initialize, create or resume the session, apply
/// config overrides, run the prompt, and record the outcome.
#[allow(clippy::too_many_arguments)]
async fn drive(
    connection: ConnectionTo<Agent>,
    state: Arc<Mutex<RunState>>,
    prompt: String,
    cwd: PathBuf,
    resume: Option<String>,
    model: Option<(String, Option<String>)>,
    permission_config: &'static str,
    format: OutputFormat,
) -> agent_client_protocol::Result<()> {
    let init = connection
        .send_request(
            InitializeRequest::new(ProtocolVersion::V1)
                .client_info(
                    Implementation::new("anvil-headless", env!("CARGO_PKG_VERSION"))
                        .title("Anvil --print"),
                )
                .client_capabilities(
                    ClientCapabilities::new()
                        .fs(FileSystemCapabilities::new()
                            .read_text_file(false)
                            .write_text_file(false))
                        .terminal(false),
                ),
        )
        .block_task()
        .await?;
    if format == OutputFormat::StreamJson {
        let (name, version) = init
            .agent_info
            .as_ref()
            .map(|info| (Some(info.name.as_str()), Some(info.version.as_str())))
            .unwrap_or((None, None));
        emit_json(&StreamRecord::Connected {
            agent_name: name,
            agent_version: version,
        });
    }

    let (session_id, resumed) = match resume {
        Some(id) => {
            connection
                .send_request(ResumeSessionRequest::new(id.clone(), cwd.clone()))
                .block_task()
                .await?;
            (id, true)
        }
        None => {
            let response = connection
                .send_request(NewSessionRequest::new(cwd.clone()))
                .block_task()
                .await?;
            (response.session_id.to_string(), false)
        }
    };
    {
        let mut state = state.lock().expect("headless state lock poisoned");
        state.session_id = Some(session_id.clone());
        state.resumed = resumed;
    }
    if format == OutputFormat::StreamJson {
        emit_json(&StreamRecord::SessionStarted {
            session_id: &session_id,
            resumed,
        });
    }

    // A failure to apply the agent-side permission mode downgrades to a
    // warning because the client-side policy still guarantees the run cannot
    // hang — but the agent may then be in its `auto` default, so say so.
    if let Err(err) = set_config(
        &connection,
        &session_id,
        "permission_mode",
        permission_config,
    )
    .await
    {
        emit_warning(
            format,
            &format!("permission mode '{permission_config}' was not applied: {err}"),
        );
    }
    if let Some((model, effort)) = model {
        // A bad model id is fatal (the run would silently use the wrong
        // model); a rejected effort only warns and falls back to the model's
        // default, matching `mj --print`.
        set_config(&connection, &session_id, "model_selection", &model).await?;
        if let Some(effort) = effort
            && let Err(err) =
                set_config(&connection, &session_id, "reasoning_effort", &effort).await
        {
            emit_warning(
                format,
                &format!("reasoning effort '{effort}' was not applied: {err}"),
            );
        }
    }

    state
        .lock()
        .expect("headless state lock poisoned")
        .prompt_sent = true;
    let response = connection
        .send_request(PromptRequest::new(
            session_id.clone(),
            vec![ContentBlock::Text(TextContent::new(prompt))],
        ))
        .block_task()
        .await?;

    let mut state = state.lock().expect("headless state lock poisoned");
    state.stop_reason = Some(response.stop_reason);
    state.usage = response.usage;
    Ok(())
}

/// Run one headless prompt against an in-process Anvil agent. Once the run
/// starts, the promised output payload is emitted even on agent/transport
/// failure; only pre-run validation (empty prompt, unreadable stdin, bad
/// `--cwd`) exits with just a stderr message. A non-`Ok` return carries the
/// failure for the exit code and stderr.
pub async fn run(
    config: RunConfig,
    llm: Arc<MultiBackend>,
    sessions: SessionStore,
    max_turns: usize,
    default_idle_timeout_secs: u64,
    default_stall_timeout_secs: u64,
) -> Result<()> {
    let prompt = resolve_prompt(&config.prompt_arg)?;
    if prompt.trim().is_empty() {
        bail!("empty prompt");
    }
    let cwd = resolve_cwd(config.cwd)?;
    let format = config.output_format;
    let mode = config.permission_mode;
    let permission_config = mode.agent_config_value();

    // Headless stdout carries the answer (or JSON records); the interactive
    // chatter the agent streams into sessions (setup notices, catalog-refresh
    // progress, end-of-turn recaps) would corrupt it.
    crate::acp::suppress_session_chatter();

    let agent = crate::acp::agent_component(
        llm,
        sessions,
        max_turns,
        default_idle_timeout_secs,
        default_stall_timeout_secs,
    );

    let state = Arc::new(Mutex::new(RunState::default()));
    let notification_state = state.clone();
    let driver_state = state.clone();
    let resume = config.resume.clone();
    let model = config.model.clone();

    let client = Client
        .builder()
        .on_receive_notification(
            async move |notification: SessionNotification, _cx| {
                // Only the one session this client created (or resumed) feeds
                // the output; anything else would silently mix into the
                // final text if the agent ever notified across sessions.
                {
                    let state = notification_state
                        .lock()
                        .expect("headless state lock poisoned");
                    if let Some(session_id) = state.session_id.as_deref()
                        && session_id != notification.session_id.to_string()
                    {
                        return Ok(());
                    }
                }
                handle_session_update(notification.update, &notification_state, format);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            async move |request: RequestPermissionRequest, responder, _cx| {
                responder.respond(answer_permission(&request, mode, format))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(agent, move |connection: ConnectionTo<Agent>| {
            drive(
                connection,
                driver_state,
                prompt,
                cwd,
                resume,
                model,
                permission_config,
                format,
            )
        });

    // First Ctrl-C ends the run gracefully: the connection future is dropped
    // (tearing down the in-process agent) and the promised output payload is
    // still emitted, with stop_reason `cancelled` and exit code 0 — the same
    // contract as `mj --print`.
    let mut interrupted = false;
    tokio::select! {
        result = client => {
            if let Err(err) = result {
                let mut state = state.lock().expect("headless state lock poisoned");
                if state.error.is_none() {
                    state.error = Some(err.to_string());
                }
            }
        }
        _ = tokio::signal::ctrl_c() => {
            interrupted = true;
            // Polling ctrl_c() replaced SIGINT's default terminate
            // disposition for the process lifetime, so restore an escape
            // hatch: a second Ctrl-C hard-exits even if payload emission or
            // teardown wedges (e.g. a blocked stdout pipe).
            tokio::spawn(async {
                let _ = tokio::signal::ctrl_c().await;
                std::process::exit(130);
            });
        }
    }

    let state = state.lock().expect("headless state lock poisoned");
    let error = state.error.clone().or_else(|| state.turn_failure.clone());
    let stop_reason = if interrupted {
        "cancelled"
    } else if state.turn_failure.is_some() {
        // Anvil ends failed turns with `end_turn` (ACP has no errored stop
        // reason) and reports the failure via usage-update meta; surface it
        // as `error` so scripts don't have to scrape the streamed text.
        "error"
    } else {
        match state.stop_reason {
            Some(reason) => stop_reason_label(reason),
            None if error.is_some() => "error",
            None => "cancelled",
        }
    };

    match format {
        OutputFormat::Text => {
            print!("{}", state.final_text);
            if !state.final_text.ends_with('\n') {
                println!();
            }
        }
        OutputFormat::Json => {
            emit_json(&JsonResult {
                session_id: state.session_id.as_deref(),
                resumed: state.resumed,
                result: &state.final_text,
                stop_reason,
                usage: state.usage.as_ref(),
                error: error.as_deref(),
            });
        }
        OutputFormat::StreamJson => {
            if let Some(message) = error.as_deref() {
                emit_json(&StreamRecord::Error { message });
            }
            emit_json(&StreamRecord::Result {
                session_id: state.session_id.as_deref(),
                resumed: state.resumed,
                result: &state.final_text,
                stop_reason,
                usage: state.usage.as_ref(),
                error: error.as_deref(),
            });
        }
    }

    if interrupted {
        return Ok(());
    }
    if let Some(message) = error {
        return Err(anyhow!(message));
    }
    match state.stop_reason {
        Some(StopReason::EndTurn | StopReason::MaxTokens | StopReason::MaxTurnRequests) => Ok(()),
        _ => Err(anyhow!("prompt stopped with {stop_reason}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn option(id: &str, kind: PermissionOptionKind) -> PermissionOption {
        PermissionOption::new(id.to_string(), id.to_string(), kind)
    }

    fn anvil_prompt_options() -> Vec<PermissionOption> {
        vec![
            option("allow_always", PermissionOptionKind::AllowAlways),
            option("allow", PermissionOptionKind::AllowOnce),
            option("reject", PermissionOptionKind::RejectOnce),
        ]
    }

    #[test]
    fn model_override_splits_known_effort_suffix() {
        assert_eq!(
            parse_model_override("codex::gpt-5-codex+high"),
            Ok(("codex::gpt-5-codex".to_string(), Some("high".to_string())))
        );
        assert_eq!(
            parse_model_override("ollama::llama3:latest"),
            Ok(("ollama::llama3:latest".to_string(), None))
        );
    }

    #[test]
    fn model_override_canonicalizes_none_to_off() {
        assert_eq!(
            parse_model_override("m+none"),
            Ok(("m".to_string(), Some("off".to_string())))
        );
    }

    #[test]
    fn model_override_keeps_unknown_suffix_in_model_id() {
        assert_eq!(
            parse_model_override("provider::model+tag"),
            Ok(("provider::model+tag".to_string(), None))
        );
    }

    #[test]
    fn model_override_rejects_empty() {
        assert!(parse_model_override("").is_err());
        assert!(parse_model_override("   ").is_err());
        assert!(parse_model_override("+high").is_err());
    }

    #[test]
    fn manual_mode_rejects_every_permission_request() {
        let options = anvil_prompt_options();
        let decision = permission_decision(PermissionMode::Manual, Some(ToolKind::Edit), &options);
        assert_eq!(decision.unwrap().option_id.to_string(), "reject");
    }

    #[test]
    fn auto_mode_allows_edits_rejects_execution() {
        let options = anvil_prompt_options();
        for kind in [ToolKind::Edit, ToolKind::Delete, ToolKind::Move] {
            let decision = permission_decision(PermissionMode::Auto, Some(kind), &options);
            assert_eq!(decision.unwrap().option_id.to_string(), "allow");
        }
        for kind in [ToolKind::Execute, ToolKind::Other] {
            let decision = permission_decision(PermissionMode::Auto, Some(kind), &options);
            assert_eq!(decision.unwrap().option_id.to_string(), "reject");
        }
        // An absent kind must not be treated as an edit.
        let decision = permission_decision(PermissionMode::Auto, None, &options);
        assert_eq!(decision.unwrap().option_id.to_string(), "reject");
    }

    #[test]
    fn yolo_mode_allows_everything_preferring_allow_once() {
        let options = anvil_prompt_options();
        let decision = permission_decision(PermissionMode::Yolo, Some(ToolKind::Execute), &options);
        // AllowOnce preferred so a one-shot run leaves no durable grants.
        assert_eq!(decision.unwrap().option_id.to_string(), "allow");

        let escalation = vec![
            option("allow_outside_sandbox", PermissionOptionKind::AllowOnce),
            option("reject", PermissionOptionKind::RejectOnce),
        ];
        let decision =
            permission_decision(PermissionMode::Yolo, Some(ToolKind::Execute), &escalation);
        assert_eq!(
            decision.unwrap().option_id.to_string(),
            "allow_outside_sandbox"
        );
    }

    #[test]
    fn rejection_cancels_when_no_reject_option_exists() {
        let options = vec![option("allow", PermissionOptionKind::AllowOnce)];
        assert!(
            permission_decision(PermissionMode::Manual, Some(ToolKind::Execute), &options)
                .is_none()
        );
    }

    #[test]
    fn stream_records_serialize_with_type_tags() {
        let record = StreamRecord::SessionStarted {
            session_id: "s1",
            resumed: false,
        };
        assert_eq!(
            serde_json::to_string(&record).unwrap(),
            r#"{"type":"session_started","session_id":"s1","resumed":false}"#
        );

        let record = StreamRecord::Permission {
            actor: PRIMARY_ACTOR,
            tool_call_id: "call-1",
            decision: "reject",
        };
        assert_eq!(
            serde_json::to_string(&record).unwrap(),
            r#"{"type":"permission","actor":"primary","tool_call_id":"call-1","decision":"reject"}"#
        );
    }

    #[test]
    fn json_result_shape_matches_issue_contract() {
        let payload = JsonResult {
            session_id: Some("s1"),
            resumed: true,
            result: "done",
            stop_reason: "end_turn",
            usage: None,
            error: None,
        };
        assert_eq!(
            serde_json::to_string(&payload).unwrap(),
            r#"{"session_id":"s1","resumed":true,"result":"done","stop_reason":"end_turn","usage":null,"error":null}"#
        );
    }

    #[test]
    fn turn_failure_message_reads_anvil_meta() {
        let meta: Meta = serde_json::from_value(serde_json::json!({
            "anvil": { "turnFailure": { "retryable": false, "message": "boom" } }
        }))
        .unwrap();
        assert_eq!(turn_failure_message(Some(&meta)), Some("boom".to_string()));
        assert_eq!(turn_failure_message(None), None);
    }
}
