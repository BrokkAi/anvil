use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol::schema::{
    AgentCapabilities, AvailableCommand, AvailableCommandsUpdate, CancelNotification,
    CloseSessionRequest, CloseSessionResponse, ConfigOptionUpdate, ContentBlock, ContentChunk,
    Cost, CurrentModeUpdate, DeleteSessionRequest, DeleteSessionResponse, EmbeddedResource,
    EmbeddedResourceResource, ForkSessionRequest, ForkSessionResponse, InitializeRequest,
    InitializeResponse, ListSessionsRequest, ListSessionsResponse, LoadSessionRequest,
    LoadSessionResponse, McpCapabilities, NewSessionRequest, NewSessionResponse,
    PromptCapabilities, PromptRequest, PromptResponse, ProtocolVersion, ResourceLink,
    ResumeSessionRequest, ResumeSessionResponse, SessionCapabilities, SessionCloseCapabilities,
    SessionConfigOption, SessionConfigOptionCategory, SessionConfigOptionValue,
    SessionConfigSelectOption, SessionDeleteCapabilities, SessionForkCapabilities, SessionInfo,
    SessionInfoUpdate, SessionListCapabilities, SessionMode as AcpSessionMode, SessionModeState,
    SessionNotification, SessionResumeCapabilities, SessionUpdate, SetSessionConfigOptionRequest,
    SetSessionConfigOptionResponse, SetSessionModeRequest, SetSessionModeResponse, StopReason,
    TextContent, Usage as AcpUsage, UsageUpdate,
};
use agent_client_protocol::{
    Agent, ByteStreams, Client, ConnectionTo, Dispatch, Handled, Responder, on_receive_dispatch,
    on_receive_notification, on_receive_request,
};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use crate::discovery::{ModelSource, split_wire_id};
use crate::llm_client::{ChatContentPart, ChatMessage, ModelMetadata, ResolvedModelInfo};
use crate::multi_backend::MultiBackend;
use crate::session::{
    CloseSessionResult, ConversationTurn, ForkOutcome, LifecycleReopen, PermissionMode,
    PromptStartError, REASONING_EFFORT_OFF_VALUE, Session, SessionManifest, SessionMode,
    SessionSnapshot, SessionStore, UnsupportedMcpTransport, acp_mcp_servers_to_configs,
};
use crate::structured_output::{
    StructuredOutputRequest, StructuredOutputResult, build_structured_output_meta,
    parse_structured_output_request, validate_response,
};
use crate::terminal_notifications::{
    TerminalNotificationEvent, emit as emit_terminal_notification,
};

/// Stable ids for our `SessionConfigOption` selectors. We expose both
/// dropdowns via configOptions because the ACP spec says clients SHOULD
/// ignore the legacy `modes` channel when configOptions is present (Zed
/// does), so once we expose any configOption we have to expose all of them.
const PERMISSION_CONFIG_ID: &str = "permission_mode";
const BEHAVIOR_CONFIG_ID: &str = "behavior_mode";
const SUPPORTED_ACP_PROTOCOL_VERSION: ProtocolVersion = ProtocolVersion::V1;
/// Mirrors the Java executor's wire id so cross-implementation clients
/// (Zed, brokk-code) can drive model selection through one canonical name.
const MODEL_CONFIG_ID: &str = "model_selection";
/// Per-session reasoning-effort knob.
/// Empty string in the wire payload clears the user's pick (back to the
/// model's `default_reasoning_level`). The `off` option explicitly omits
/// reasoning controls even when the model advertises a default.
const REASONING_EFFORT_CONFIG_ID: &str = "reasoning_effort";
/// Sentinel value the client sends to clear the user's pick. We accept
/// either an empty string or this token so editor implementations that
/// strip-trim selection ids still work.
const REASONING_EFFORT_DEFAULT_VALUE: &str = "(default)";

fn negotiate_protocol_version(requested: ProtocolVersion) -> ProtocolVersion {
    if requested == SUPPORTED_ACP_PROTOCOL_VERSION {
        requested
    } else {
        SUPPORTED_ACP_PROTOCOL_VERSION
    }
}

fn parse_prompt_structured_output_request(
    req: &PromptRequest,
) -> Result<Option<StructuredOutputRequest>, String> {
    parse_structured_output_request(req.meta.as_ref()).map_err(|err| err.to_string())
}

fn invalid_lifecycle_cwd_error(method: &str, cwd: &Path) -> agent_client_protocol::Error {
    agent_client_protocol::Error::invalid_params().data(serde_json::json!({
        "reason": format!("{method} cwd must be absolute: {}", cwd.display()),
    }))
}

/// Build the protocol error returned when a request names a session Anvil
/// does not know. Shared by the `session/prompt` sites (cold-miss,
/// closed-mid-request, registry rebuild) and the `session/load` /
/// `session/resume` lifecycle handlers so the wording stays identical and
/// unknown sessions surface as protocol errors rather than synthetic agent
/// messages plus a successful response.
fn unknown_session_error(session_id: &str) -> agent_client_protocol::Error {
    agent_client_protocol::Error::invalid_params().data(serde_json::json!({
        "reason": format!("unknown session '{session_id}'"),
    }))
}

/// Build the protocol error returned when a lifecycle request's `cwd` does not
/// match the cwd an existing in-memory session was created/loaded under. ACP
/// treats `cwd` as the session working directory; silently moving a warm
/// session to a different root would change project instructions, skills,
/// permission scope, and sandbox assumptions, so Anvil rejects the move.
fn lifecycle_cwd_mismatch_error(
    method: &str,
    session_cwd: &Path,
    requested_cwd: &Path,
) -> agent_client_protocol::Error {
    agent_client_protocol::Error::invalid_params().data(serde_json::json!({
        "reason": format!(
            "{method} cwd '{}' does not match the session's cwd '{}'; moving a \
             session between working directories is not supported",
            requested_cwd.display(),
            session_cwd.display(),
        ),
    }))
}

/// Build the protocol error returned when a lifecycle request references an
/// MCP server transport Anvil does not support. Anvil advertises
/// `mcpCapabilities` with http/sse disabled, so an HTTP/SSE server is rejected
/// rather than silently skipped (which would leave the session looking
/// configured while the requested tools were missing).
fn unsupported_mcp_transport_error(
    method: &str,
    err: &UnsupportedMcpTransport,
) -> agent_client_protocol::Error {
    agent_client_protocol::Error::invalid_params().data(serde_json::json!({
        "reason": format!(
            "{method} MCP server '{}' uses the unsupported '{}' transport; Anvil only \
             supports stdio MCP servers",
            err.server, err.transport
        ),
    }))
}

/// Build the protocol error returned for non-empty `additionalDirectories` on
/// a lifecycle request. Anvil does not advertise
/// `sessionCapabilities.additionalDirectories`, so rather than silently
/// dropping requested roots -- a footgun where the session looks configured
/// but tools cannot reach those roots -- it rejects them until multi-root
/// support exists.
fn unsupported_additional_directories_error(method: &str) -> agent_client_protocol::Error {
    agent_client_protocol::Error::invalid_params().data(serde_json::json!({
        "reason": format!(
            "{method} additionalDirectories is not supported: Anvil does not advertise \
             sessionCapabilities.additionalDirectories"
        ),
    }))
}

fn prompt_response_meta(
    result: Option<&StructuredOutputResult>,
    orchestration_model: Option<&ResolvedModelInfo>,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    let mut meta = build_structured_output_meta(result).unwrap_or_default();
    if let Some(model) = orchestration_model {
        let mut namespace = meta
            .remove(crate::structured_output::ACP_META_NAMESPACE)
            .and_then(|value| match value {
                serde_json::Value::Object(map) => Some(map),
                _ => None,
            })
            .unwrap_or_default();
        namespace.insert(
            "modelSelection".to_string(),
            serde_json::json!({
                "orchestration": {
                    "configured_model": model.configured_model.clone(),
                    "resolved_provider": model.resolved_provider.clone(),
                    "resolved_model": model.resolved_model.clone(),
                    "actual_model": model.resolved_model.clone(),
                },
                "internal_specialist": {
                    "separate_model_selection_supported": false,
                    "configured_model": null,
                    "resolved_provider": model.resolved_provider.clone(),
                    "resolved_model": model.resolved_model.clone(),
                    "actual_model": model.resolved_model.clone(),
                    "selection_source": "inherits_orchestration",
                    "reason": "ACP session/prompt does not support a separate internal specialist model; task subagents inherit the orchestration model.",
                }
            }),
        );
        meta.insert(
            crate::structured_output::ACP_META_NAMESPACE.to_string(),
            serde_json::Value::Object(namespace),
        );
    }
    if meta.is_empty() { None } else { Some(meta) }
}

/// Build the terminal `PromptResponse` for a finished turn, choosing the
/// stop reason from whether the prompt's cancellation token fired.
///
/// ACP requires that a turn cancelled via `session/cancel` resolves its
/// original `session/prompt` with `StopReason::Cancelled` -- even when the
/// cancellation aborted underlying LLM/tool work -- so the client can
/// distinguish a cancelled prompt from a normal completion. Callers pass
/// `cancel.is_cancelled()` after the turn settles; the tool loop already
/// catches cancellation internally and returns normally, so the token is
/// the authoritative signal here.
fn prompt_stop_response(cancelled: bool) -> PromptResponse {
    emit_terminal_notification(TerminalNotificationEvent::TurnEnded);
    let stop_reason = if cancelled {
        StopReason::Cancelled
    } else {
        StopReason::EndTurn
    };
    PromptResponse::new(stop_reason)
}

/// Convenience wrapper for the non-cancellable, synchronous prompt paths
/// (slash commands, validation short-circuits) that always end the turn.
fn prompt_end_turn_response() -> PromptResponse {
    prompt_stop_response(false)
}

/// Available session modes exposed to ACP clients.
fn available_modes() -> Vec<AcpSessionMode> {
    vec![
        AcpSessionMode::new("LUTZ", "LUTZ").description("Agentic loop with task list"),
        AcpSessionMode::new("CODE", "CODE").description("Code changes only"),
        AcpSessionMode::new("ASK", "ASK").description("Question answering"),
        AcpSessionMode::new("PLAN", "PLAN").description("Planning only"),
    ]
}

fn mode_state(current: &str) -> SessionModeState {
    SessionModeState::new(current.to_string(), available_modes())
}

/// Build the permission-mode `SessionConfigOption` reflecting `current`.
fn permission_config_option(current: PermissionMode) -> SessionConfigOption {
    let options = vec![
        SessionConfigSelectOption::new("default", "Default")
            .description("Ask for permission before each tool call"),
        SessionConfigSelectOption::new("auto", "Auto")
            .description("Use the permission classifier only for promptable tool calls"),
        SessionConfigSelectOption::new("acceptEdits", "Accept Edits")
            .description("Auto-allow edits; ask for everything else"),
        SessionConfigSelectOption::new("readOnly", "Read-only")
            .description("Refuse every edit, deletion, move, or shell command"),
        SessionConfigSelectOption::new("bypassPermissions", "Bypass Permissions")
            .description("Allow all tool calls without prompting (use with care)"),
    ];
    SessionConfigOption::select(
        PERMISSION_CONFIG_ID,
        "Permission",
        current.as_str(),
        options,
    )
    .description("Controls which tool calls require user approval.")
    .category(SessionConfigOptionCategory::Mode)
}

/// Build the behavior-mode `SessionConfigOption` reflecting `current`. This
/// is the configOptions-channel counterpart to the legacy `SessionMode` menu
/// and drives system-prompt selection.
fn behavior_config_option(current: SessionMode) -> SessionConfigOption {
    let options = vec![
        SessionConfigSelectOption::new("LUTZ", "LUTZ").description("Agentic loop with task list"),
        SessionConfigSelectOption::new("CODE", "CODE").description("Code changes only"),
        SessionConfigSelectOption::new("ASK", "ASK").description("Question answering"),
        SessionConfigSelectOption::new("PLAN", "PLAN").description("Planning only"),
    ];
    SessionConfigOption::select(BEHAVIOR_CONFIG_ID, "Mode", current.as_str(), options)
        .description("Controls Brokk's overall behavior style for this session.")
        .category(SessionConfigOptionCategory::Mode)
}

/// Build the model `SessionConfigOption` reflecting `current` against the
/// cached `available_models` catalog. Returns `None` when the catalog is
/// empty, in which case the dropdown is omitted entirely (per ACP, a select
/// with zero options is not useful and some clients reject it).
fn model_config_option(current: &str, available_models: &[String]) -> Option<SessionConfigOption> {
    if available_models.is_empty() {
        return None;
    }
    // `SessionConfigSelectOption::new` stores its arguments owned, so the
    // closure must hand it owned Strings -- borrowing from `available_models`
    // would tie the option's lifetime to the slice and fail E0521.
    let options: Vec<SessionConfigSelectOption> = available_models
        .iter()
        .map(|m| SessionConfigSelectOption::new(m.clone(), m.clone()))
        .collect();
    // Fall back to the first catalog entry when `current` is empty or has
    // drifted out of the catalog -- otherwise some clients refuse to render
    // a select whose value is not in `options`.
    let current_value = if !current.is_empty() && available_models.iter().any(|m| m == current) {
        current.to_string()
    } else {
        available_models[0].clone()
    };
    Some(
        SessionConfigOption::select(MODEL_CONFIG_ID, "Model", current_value, options)
            .description("Selects the LLM model used for this session.")
            .category(SessionConfigOptionCategory::Model),
    )
}

/// Build the reasoning-effort `SessionConfigOption` for the active model.
/// Returns `None` when the model exposes no presets -- the dropdown is
/// omitted entirely in that case rather than shown empty.
///
/// Layout: an explicit "(default)" entry at the head represents "no user
/// pick, server uses `default_reasoning_level`". The following "off" entry
/// represents an explicit user pick to omit reasoning controls even for models
/// that default to reasoning. The user's stored pick (`current`) selects
/// whichever option matches; when no pick exists, the default entry is selected
/// so the picker reflects actual intent.
fn reasoning_effort_config_option(
    current: Option<&str>,
    catalog: &[ModelMetadata],
    current_model: &str,
) -> Option<SessionConfigOption> {
    let model = catalog.iter().find(|m| m.id == current_model)?;
    if model.supported_reasoning_levels.is_empty() {
        return None;
    }
    let default_label = match &model.default_reasoning_level {
        Some(d) => format!("Default ({d})"),
        None => "Default".to_string(),
    };
    let mut options = vec![
        SessionConfigSelectOption::new(REASONING_EFFORT_DEFAULT_VALUE, default_label)
            .description("Use the model's default reasoning effort."),
        SessionConfigSelectOption::new(REASONING_EFFORT_OFF_VALUE, "Off")
            .description("Do not send reasoning controls for this session."),
    ];
    options.extend(model.supported_reasoning_levels.iter().map(|preset| {
        let opt = SessionConfigSelectOption::new(preset.effort.clone(), preset.effort.clone());
        if preset.description.is_empty() {
            opt
        } else {
            opt.description(preset.description.clone())
        }
    }));
    // Coerce out-of-catalog picks (e.g. stale from before a slug bump)
    // to the default sentinel so the picker always renders against an
    // entry it advertises.
    let current_value = match current {
        Some(eff) if eff == REASONING_EFFORT_OFF_VALUE => REASONING_EFFORT_OFF_VALUE.to_string(),
        Some(eff)
            if model
                .supported_reasoning_levels
                .iter()
                .any(|p| p.effort == eff) =>
        {
            eff.to_string()
        }
        _ => REASONING_EFFORT_DEFAULT_VALUE.to_string(),
    };
    Some(
        SessionConfigOption::select(
            REASONING_EFFORT_CONFIG_ID,
            "Reasoning effort",
            current_value,
            options,
        )
        .description(
            "Controls how much chain-of-thought the model spends on each turn. \
             Higher levels are deeper but slower and cost more against your plan's quota.",
        )
        .category(SessionConfigOptionCategory::Model),
    )
}

/// All configOption selectors we expose, in display order. The model
/// selector is appended only when the LLM catalog is known; clients that
/// drive model selection through the meta extension still see the current
/// model via `meta.brokk.modelId`. The reasoning-effort selector is appended
/// only when the active model publishes presets.
fn all_config_options(
    behavior: SessionMode,
    permission: PermissionMode,
    current_model: &str,
    available_models: &[ModelMetadata],
    current_reasoning_effort: Option<&str>,
) -> Vec<SessionConfigOption> {
    let model_ids: Vec<String> = available_models.iter().map(|m| m.id.clone()).collect();
    let mut opts = vec![
        behavior_config_option(behavior),
        permission_config_option(permission),
    ];
    if let Some(model_opt) = model_config_option(current_model, &model_ids) {
        opts.push(model_opt);
    }
    if let Some(re_opt) =
        reasoning_effort_config_option(current_reasoning_effort, available_models, current_model)
    {
        opts.push(re_opt);
    }
    opts
}

/// Wire ids accepted by `apply_config_option`. Kept in a single slice so
/// the ACP `setSessionConfigOption` request handler and `/setup advanced`
/// report identical supported-key lists.
const CONFIGURE_KNOWN_KEYS: &[&str] = &[
    BEHAVIOR_CONFIG_ID,
    PERMISSION_CONFIG_ID,
    MODEL_CONFIG_ID,
    REASONING_EFFORT_CONFIG_ID,
];

/// Outcome of a successful `apply_config_option` call. Carries the full
/// re-derived option list so the caller can re-emit a `ConfigOptionUpdate`
/// notification with the spec-required complete state.
#[derive(Debug)]
struct ConfigApplyOutcome {
    updated_options: Vec<SessionConfigOption>,
    /// Set only by the `model` arm when the previous reasoning_effort pick
    /// is not in the new model's supported set and the store dropped it.
    /// Both callers surface this to the user.
    cleared_reasoning: Option<String>,
}

/// Validation / dispatch errors from `apply_config_option`. The request
/// handler maps these into JSON error data; the slash command formats them
/// into a one-line user message via `human_message`.
#[derive(Debug)]
enum ConfigApplyError {
    UnknownConfigId,
    InvalidValue {
        reason: String,
        supported: Vec<String>,
    },
    UnknownSession,
    PersistFailed {
        details: String,
    },
}

impl ConfigApplyError {
    fn human_message(&self) -> String {
        match self {
            ConfigApplyError::UnknownConfigId => format!(
                "unknown config key. Supported: {}",
                CONFIGURE_KNOWN_KEYS.join(", ")
            ),
            ConfigApplyError::InvalidValue { reason, supported } => {
                if supported.is_empty() {
                    reason.clone()
                } else {
                    format!("{reason}. Supported: {}", supported.join(", "))
                }
            }
            ConfigApplyError::UnknownSession => "unknown session".to_string(),
            ConfigApplyError::PersistFailed { details } => {
                format!("failed to persist setting: {details}")
            }
        }
    }
}

/// Apply a single `configOptions` change. Single source of truth shared by
/// the `setSessionConfigOption` ACP request and `/setup`: validates the
/// value, mutates session state, and returns the full re-derived options
/// list so the caller can emit a `ConfigOptionUpdate` notification with
/// the spec-required complete state.
/// Re-fetch the session and build the complete current `SessionConfigOption`
/// list. ACP config-option responses and `config_option_update` notifications
/// carry the full set (not just the changed selector), so both the
/// `session/set_config_option` and `session/set_mode` paths use this. Returns
/// `None` if the session is unknown.
async fn current_config_options(
    sessions: &SessionStore,
    session_id: &str,
) -> Option<Vec<SessionConfigOption>> {
    let fallback_cwd = std::env::current_dir().unwrap_or_default();
    let session = sessions.get_session(session_id, &fallback_cwd).await?;
    let catalog = sessions.available_model_metadata().await;
    Some(all_config_options(
        session.mode,
        session.permission_mode,
        &session.model,
        &catalog,
        session.selected_reasoning_effort.as_deref(),
    ))
}

/// Emit the ACP updates for a config-option change: a `config_option_update`
/// with the full current set, plus -- when the change was the behavior-mode
/// selector -- a `current_mode_update` so the legacy modes surface stays in
/// sync (#157). Every path that mutates a config option (the
/// `session/set_config_option` request and the `/setup` slash commands) routes
/// through this, so the two surfaces cannot drift apart.
fn send_config_option_change_updates(
    cx: &ConnectionTo<Client>,
    session_id: &str,
    config_id: &str,
    config_value: &str,
    updated_options: Vec<SessionConfigOption>,
) {
    let notification = SessionNotification::new(
        session_id.to_string(),
        SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate::new(updated_options)),
    );
    if let Err(e) = cx.send_notification(notification) {
        tracing::warn!("failed to send config_option_update: {e}");
    }

    if config_id == BEHAVIOR_CONFIG_ID
        && let Some(mode) = SessionMode::parse(config_value)
    {
        let mode_notification = SessionNotification::new(
            session_id.to_string(),
            SessionUpdate::CurrentModeUpdate(CurrentModeUpdate::new(mode.as_str())),
        );
        if let Err(e) = cx.send_notification(mode_notification) {
            tracing::warn!("failed to send current_mode_update: {e}");
        }
    }
}

async fn apply_config_option(
    sessions: &SessionStore,
    session_id: &str,
    config_id: &str,
    value: &str,
) -> Result<ConfigApplyOutcome, ConfigApplyError> {
    let mut cleared_reasoning: Option<String> = None;

    match config_id {
        PERMISSION_CONFIG_ID => {
            let Some(permission_mode) = PermissionMode::parse(value) else {
                return Err(ConfigApplyError::InvalidValue {
                    reason: format!("unknown permission mode '{value}'"),
                    supported: vec![
                        "default".to_string(),
                        "auto".to_string(),
                        "acceptEdits".to_string(),
                        "readOnly".to_string(),
                        "bypassPermissions".to_string(),
                    ],
                });
            };
            if !sessions
                .set_permission_mode(session_id, permission_mode)
                .await
            {
                return Err(ConfigApplyError::UnknownSession);
            }
        }
        BEHAVIOR_CONFIG_ID => {
            let Some(behavior_mode) = SessionMode::parse(value) else {
                return Err(ConfigApplyError::InvalidValue {
                    reason: format!("unknown behavior mode '{value}'"),
                    supported: vec![
                        "LUTZ".to_string(),
                        "CODE".to_string(),
                        "ASK".to_string(),
                        "PLAN".to_string(),
                    ],
                });
            };
            match sessions.set_mode(session_id, behavior_mode).await {
                Ok(true) => {}
                Ok(false) => return Err(ConfigApplyError::UnknownSession),
                Err(e) => {
                    return Err(ConfigApplyError::PersistFailed {
                        details: format!("{e:#}"),
                    });
                }
            }
        }
        MODEL_CONFIG_ID => {
            if value.is_empty() {
                return Err(ConfigApplyError::InvalidValue {
                    reason: "model id must be a non-empty string".to_string(),
                    supported: Vec::new(),
                });
            }
            // Reject ids that drift out of the catalog when one is known.
            // An empty catalog means model discovery never succeeded;
            // accept anything in that case so the user can still drive
            // the agent against a manually-configured backend.
            let known = sessions.available_models().await;
            if !known.is_empty() && !known.iter().any(|m| m == value) {
                return Err(ConfigApplyError::InvalidValue {
                    reason: format!("unknown model '{value}'"),
                    supported: known,
                });
            }
            match sessions.set_model(session_id, value.to_string()).await {
                Ok((true, cleared)) => {
                    cleared_reasoning = cleared;
                }
                Ok((false, _)) => return Err(ConfigApplyError::UnknownSession),
                Err(e) => {
                    return Err(ConfigApplyError::PersistFailed {
                        details: format!("{e:#}"),
                    });
                }
            }
        }
        REASONING_EFFORT_CONFIG_ID => {
            // Empty string or the "(default)" sentinel both mean "clear my
            // pick, use the model default". The explicit "off" selection is
            // stored as a real pick; snapshot() interprets it as "omit
            // reasoning controls" rather than falling back to the model
            // default.
            let effort = if value.is_empty() || value == REASONING_EFFORT_DEFAULT_VALUE {
                None
            } else {
                Some(value.to_string())
            };
            // Validate against the active model's published levels when
            // one is known. An unknown catalog (e.g. discovery never
            // finished) accepts any string so a manually-configured
            // backend still works. "off" sends no provider reasoning
            // parameter, so it is harmless and always accepted even if the
            // current model has no configurable reasoning presets.
            if let Some(eff) = &effort
                && eff != REASONING_EFFORT_OFF_VALUE
            {
                let fallback_cwd = std::env::current_dir().unwrap_or_default();
                let active_model = sessions
                    .get_session(session_id, &fallback_cwd)
                    .await
                    .map(|s| s.model);
                let catalog = sessions.available_model_metadata().await;
                if let Some(model_id) = active_model
                    && let Some(meta) = catalog.iter().find(|m| m.id == model_id)
                {
                    if meta.supported_reasoning_levels.is_empty() {
                        return Err(ConfigApplyError::InvalidValue {
                            reason: format!(
                                "model '{model_id}' does not support configurable reasoning effort"
                            ),
                            supported: Vec::new(),
                        });
                    }
                    if !meta
                        .supported_reasoning_levels
                        .iter()
                        .any(|p| &p.effort == eff)
                    {
                        let supported: Vec<String> = meta
                            .supported_reasoning_levels
                            .iter()
                            .map(|p| p.effort.clone())
                            .collect();
                        return Err(ConfigApplyError::InvalidValue {
                            reason: format!(
                                "reasoning effort '{eff}' is not supported by model '{model_id}'"
                            ),
                            supported,
                        });
                    }
                }
            }
            if !sessions.set_reasoning_effort(session_id, effort).await {
                return Err(ConfigApplyError::UnknownSession);
            }
        }
        _ => return Err(ConfigApplyError::UnknownConfigId),
    }

    // Re-fetch the session so the returned options reflect the latest
    // values for *all* selectors. The spec says the response carries the
    // full updated set, not just the one we changed.
    let updated_options = current_config_options(sessions, session_id)
        .await
        .ok_or(ConfigApplyError::UnknownSession)?;

    Ok(ConfigApplyOutcome {
        updated_options,
        cleared_reasoning,
    })
}

/// `/setup` remains the model/provider and advanced configuration entry point.
/// Permission mode is exposed through the ACP session config selector; the
/// `/permissions` slash command only manages remembered Always allow entries.
fn builtin_commands() -> Vec<AvailableCommand> {
    vec![
        AvailableCommand::new("context", "Show current session context snapshot"),
        AvailableCommand::new(
            "loop",
            "Repeat a slash command or prompt on an interval until cancelled",
        ),
        AvailableCommand::new(
            "goal",
            "Work autonomously across turns until an objective is verifiably met \
             (e.g. `/goal make `cargo test` pass`)",
        ),
        AvailableCommand::new(
            "setup",
            "Set up models, login, behavior, sandboxing, and advanced options",
        ),
        AvailableCommand::new(
            "permissions",
            "List and clear remembered Always allow entries",
        ),
        AvailableCommand::new(
            "compress",
            "Summarize uncompressed turns to free up context window",
        ),
        AvailableCommand::new("mcp", "List and configure MCP servers"),
        AvailableCommand::new(
            "pr-create",
            "Create a GitHub pull request from the current branch (e.g. `/pr-create [title]`)",
        ),
        AvailableCommand::new(
            "usage",
            "Show session token totals, USD cost, and OpenRouter credit balance",
        ),
    ]
}

/// Set of built-in slash command names, used to detect collisions with
/// skill names so the built-in always wins (matches the spec's "Hide
/// filtered skills entirely" guidance: don't expose a slash that won't
/// actually dispatch to the skill).
fn builtin_command_names() -> std::collections::HashSet<&'static str> {
    [
        "context",
        "loop",
        "goal",
        "setup",
        "permissions",
        "compress",
        "mcp",
        "pr-create",
        "usage",
    ]
    .into_iter()
    .collect()
}

/// Build the full command list advertised to the client: built-ins plus
/// one entry per discovered skill. Skill commands whose names collide
/// with a built-in are dropped (with a warning) so the user doesn't see
/// ambiguous autocomplete -- the skill remains reachable to the model
/// via the `activate_skill` tool.
fn available_commands(registry: &crate::skills::SkillRegistry) -> Vec<AvailableCommand> {
    let mut commands = builtin_commands();
    if registry.is_empty() {
        return commands;
    }
    let builtins = builtin_command_names();
    for meta in registry.iter_sorted() {
        let command_name = meta.name.to_ascii_lowercase();
        if builtins.contains(command_name.as_str()) {
            tracing::warn!(
                skill = %meta.name,
                location = %meta.location.display(),
                "skill name collides with a built-in slash command; hiding from autocomplete"
            );
            continue;
        }
        match registry.get_for_slash_command(&command_name) {
            Some(resolved) if resolved.name == meta.name => {}
            Some(resolved) => {
                tracing::warn!(
                    skill = %meta.name,
                    resolved_skill = %resolved.name,
                    location = %meta.location.display(),
                    "skill name collides with another skill after slash-command case normalization; hiding from autocomplete"
                );
                continue;
            }
            None => {
                tracing::warn!(
                    skill = %meta.name,
                    location = %meta.location.display(),
                    "skill name is ambiguous after slash-command case normalization; hiding from autocomplete"
                );
                continue;
            }
        }
        commands.push(AvailableCommand::new(
            meta.name.clone(),
            shorten_for_autocomplete(&meta.description),
        ));
    }
    commands
}

/// Editor autocomplete widgets render the command description inline,
/// so wrap long descriptions to keep the dropdown legible. The spec
/// caps descriptions at 1024 chars; ~200 chars is plenty for a tooltip.
fn shorten_for_autocomplete(s: &str) -> String {
    const MAX: usize = 200;
    let trimmed = s.trim();
    if trimmed.chars().count() <= MAX {
        return trimmed.to_string();
    }
    let mut acc = String::with_capacity(MAX + 3);
    for (i, ch) in trimmed.chars().enumerate() {
        if i >= MAX - 1 {
            break;
        }
        acc.push(ch);
    }
    acc.push('…');
    acc
}

fn send_available_commands_update(
    cx: &ConnectionTo<Client>,
    session_id: &str,
    registry: &crate::skills::SkillRegistry,
) {
    let update = SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate::new(
        available_commands(registry),
    ));
    let notification = SessionNotification::new(session_id.to_string(), update);
    if let Err(e) = cx.send_notification(notification) {
        tracing::warn!("failed to send available_commands_update: {e}");
    }
}

fn session_info_from_manifest(manifest: &SessionManifest, cwd: &Path) -> SessionInfo {
    SessionInfo::new(manifest.id.clone(), cwd.to_path_buf())
        .title(manifest.title())
        .updated_at(manifest.updated_at())
}

/// Page size for ACP `session/list` cursor pagination.
const SESSION_LIST_PAGE_SIZE: usize = 50;
/// Prefix for our opaque `session/list` cursor token. Namespacing the token
/// lets us reject foreign or hand-crafted cursors instead of silently treating
/// them as offset 0, satisfying ACP's "invalid cursor SHOULD error" guidance.
const SESSION_LIST_CURSOR_PREFIX: &str = "anvil:";

/// Fingerprint of the list context (the `cwd` filter) a `session/list` cursor
/// was issued for. Cursors are offsets into a specific ordered list; binding
/// them to the cwd context lets the handler reject a cursor replayed against a
/// different filter (e.g. a cwd-list cursor resent without `cwd`), which would
/// otherwise silently skip or duplicate entries. `DefaultHasher` is
/// deterministic within a process, which is all cursor round-trips require.
fn session_list_context_tag(cwd: Option<&Path>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    match cwd {
        Some(cwd) => {
            true.hash(&mut hasher);
            cwd.hash(&mut hasher);
        }
        None => false.hash(&mut hasher),
    }
    hasher.finish()
}

/// Encode a page offset (for the given list context) into an opaque
/// `session/list` cursor token.
fn encode_session_list_cursor(context_tag: u64, offset: usize) -> String {
    format!("{SESSION_LIST_CURSOR_PREFIX}{context_tag:x}:{offset}")
}

/// Decode an opaque `session/list` cursor token back to its page offset.
/// Returns `None` for any cursor Anvil did not issue for this same list
/// context -- foreign, malformed, or minted against a different `cwd` filter --
/// so the handler can surface an invalid-params error rather than silently
/// restarting at 0 or paging the wrong list.
fn parse_session_list_cursor(cursor: &str, context_tag: u64) -> Option<usize> {
    let rest = cursor.strip_prefix(SESSION_LIST_CURSOR_PREFIX)?;
    let (tag_hex, offset) = rest.split_once(':')?;
    if u64::from_str_radix(tag_hex, 16).ok()? != context_tag {
        return None;
    }
    offset.parse::<usize>().ok()
}

/// Compute the half-open page bounds `[start, end)` and the next-page cursor
/// for a `session/list` response covering `total` ordered sessions starting at
/// `offset`. An `offset` past the end yields an empty page and no next cursor
/// (end-of-results), never an error.
fn paginate_session_list(
    total: usize,
    offset: usize,
    context_tag: u64,
) -> (usize, usize, Option<String>) {
    let start = offset.min(total);
    let end = start.saturating_add(SESSION_LIST_PAGE_SIZE).min(total);
    let next_cursor = (end < total).then(|| encode_session_list_cursor(context_tag, end));
    (start, end, next_cursor)
}

fn send_session_info_update(
    cx: &ConnectionTo<Client>,
    session_id: &str,
    title: Option<String>,
    updated_at: Option<String>,
) {
    if title.is_none() && updated_at.is_none() {
        return;
    }
    let mut update = SessionInfoUpdate::new();
    if let Some(title) = title {
        update = update.title(title);
    }
    if let Some(updated_at) = updated_at {
        update = update.updated_at(updated_at);
    }
    let notification = SessionNotification::new(
        session_id.to_string(),
        SessionUpdate::SessionInfoUpdate(update),
    );
    if let Err(e) = cx.send_notification(notification) {
        tracing::warn!("failed to send session_info_update: {e}");
    }
}

fn session_usage_update(
    snap: &SessionSnapshot,
    available_models: &[crate::llm_client::ModelMetadata],
    cost_usd: Option<f64>,
) -> UsageUpdate {
    let messages = build_prompt_messages_with_parts(snap, "", &[]);
    let used = crate::tokens::approximate_tokens_messages(&messages) as u64;
    let size = available_models
        .iter()
        .find(|m| m.id == snap.model)
        .and_then(|m| m.context_length)
        .unwrap_or(crate::context_manager::FALLBACK_CONTEXT_LENGTH) as u64;
    let mut update = UsageUpdate::new(used, size);
    if let Some(amount) = cost_usd {
        update = update.cost(Some(Cost::new(amount, "USD")));
    }
    update
}

async fn send_session_usage_update(
    cx: &ConnectionTo<Client>,
    sessions: &SessionStore,
    session_id: &str,
    fallback_cwd: &Path,
) {
    let Some(snap) = sessions.snapshot(session_id, fallback_cwd).await else {
        return;
    };
    let cost_usd = sessions.exact_usage_cost_usd(session_id).await;
    let update = session_usage_update(&snap, &sessions.available_model_metadata().await, cost_usd);
    let notification =
        SessionNotification::new(session_id.to_string(), SessionUpdate::UsageUpdate(update));
    if let Err(e) = cx.send_notification(notification) {
        tracing::warn!("failed to send usage_update: {e}");
    }
}

/// Defer the `available_commands_update` notification so the client has
/// time to register the freshly-issued session id before the
/// notification references it.
///
/// History: #3611 fixed the same symptom by responding to `session/new`
/// *before* sending this notification, relying on the
/// agent-client-protocol crate's single FIFO outbound channel. That
/// ordered the two messages correctly on the wire. The bug has come
/// back because of how Zed dispatches incoming traffic: its
/// `new_session` handler (zed `crates/agent_servers/src/acp.rs`) inserts
/// the session into `self.sessions` only *after* the `session/new`
/// response future resolves and follow-up work runs (default
/// `SetSessionMode` / `SetSessionModel` RPCs, default-config-option
/// application, `AcpThread::new`). The response and any notification on
/// the same session arrive on Zed as two independent dispatch tasks;
/// the notification handler can be polled in the window between the
/// response future resolving and `sessions.borrow_mut().insert(...)`,
/// and is dropped with `Received session notification for unknown
/// session`. Symptom: the command palette stays empty even though the
/// wire order matches #3611.
///
/// `session/load` and `session/resume` go through Zed's
/// `open_or_create_session`, which *pre*-registers the session id before
/// awaiting the RPC (the client knows the id up front on those paths).
/// `session/new` cannot pre-register because the id is issued by the
/// server in the response, so it stays exposed to this race.
///
/// Wire-order alone is not enough. We send the notification from a
/// short-delay tokio task so it lands on Zed *after* Zed's post-response
/// bookkeeping has run and the session id is in the map. ~100ms is
/// invisible to a human at the command palette and well above the
/// post-response sync work measured locally. Applied symmetrically to
/// new/load/resume so a future Zed refactor that reshapes a
/// load/resume path can't silently re-introduce the regression.
fn spawn_delayed_available_commands_update(
    cx: ConnectionTo<Client>,
    session_id: String,
    skills: Arc<crate::skills::SkillRegistry>,
) {
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        send_available_commands_update(&cx, &session_id, &skills);
    });
}

fn spawn_delayed_session_usage_update(
    cx: ConnectionTo<Client>,
    sessions: SessionStore,
    session_id: String,
    fallback_cwd: std::path::PathBuf,
) {
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        send_session_usage_update(&cx, &sessions, &session_id, &fallback_cwd).await;
    });
}

/// Spawn a background discovery refresh that updates the session
/// store's cached model catalog. Background callers queue on the shared
/// refresh lock instead of skipping work when another refresh is in
/// flight. Shared by `session/new` and provider login/logout flows.
fn spawn_background_refresh(
    refresh_lock: Arc<tokio::sync::Mutex<()>>,
    llm: Arc<MultiBackend>,
    sessions: SessionStore,
    transcript: Option<(ConnectionTo<Client>, String, &'static str)>,
    initial_delay: Option<Duration>,
) {
    tokio::spawn(async move {
        if let Some(delay) = initial_delay {
            tokio::time::sleep(delay).await;
        }

        if let Some((cx, session_id, intro)) = &transcript {
            trace_openrouter_refresh(intro.trim_end());
            send_message(cx, session_id, &format!("{intro}\n"));
            trace_openrouter_refresh("Waiting for model refresh lock...");
            send_message(cx, session_id, "Waiting for model refresh lock...\n");
        }

        let _refresh_guard = refresh_lock.lock().await;
        if let Some((cx, session_id, _)) = &transcript {
            trace_openrouter_refresh("Refresh lock acquired.");
            send_message(cx, session_id, "Refresh lock acquired.\n");
        }

        let result = match &transcript {
            Some((cx, session_id, _)) => {
                refresh_model_catalog_after_lock(Some(cx), Some(session_id), &llm, &sessions).await
            }
            None => refresh_model_catalog_after_lock(None, None, &llm, &sessions).await,
        };

        if let Err(e) = result {
            tracing::debug!("background model-catalog refresh failed: {e}");
            if let Some((cx, session_id, _)) = &transcript {
                send_message(
                    cx,
                    session_id,
                    &format!("Model catalog refresh failed: {e}\n"),
                );
            }
        }
    });
}

fn spawn_delayed_setup_notice(
    cx: ConnectionTo<Client>,
    session: Session,
    catalog: Vec<ModelMetadata>,
    sessions: SessionStore,
) {
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(150)).await;
        let state = sessions.setup_state_snapshot();
        let message = render_session_start_setup_notice(&session, &catalog, state.first_run_seen);
        send_message(&cx, &session.id, &message);
        if !state.first_run_seen
            && let Err(e) = sessions.remember_first_run_seen()
        {
            tracing::warn!("failed to persist first-run setup state: {e:#}");
        }
    });
}

fn render_session_start_setup_notice(
    session: &Session,
    catalog: &[ModelMetadata],
    first_run_seen: bool,
) -> String {
    if session.model.is_empty() {
        let mut out = String::from("No model is ready yet. Starting setup.\n\n");
        out.push_str(&render_setup_home(session, catalog));
        return out;
    }

    if !first_run_seen {
        let mut out = String::from("Anvil found a working model setup and is ready to use.\n\n");
        out.push_str("Run `/setup` anytime to change or repair model setup.");
        return out;
    }

    "Anvil is ready. Run `/setup` anytime to change or repair model setup.".to_string()
}

fn source_count(catalog: &[ModelMetadata], source: ModelSource) -> usize {
    catalog
        .iter()
        .filter(|m| split_wire_id(&m.id).is_some_and(|(s, _)| s == source))
        .count()
}

const MODEL_REFRESH_LOCK_WAIT: Duration = Duration::from_secs(2);

fn preferred_model(catalog: &[ModelMetadata]) -> Option<String> {
    [
        ModelSource::Bedrock,
        ModelSource::Codex,
        ModelSource::Ollama,
        ModelSource::Ds4,
        ModelSource::DeepSeek,
        ModelSource::OpenRouter,
    ]
    .into_iter()
    .find_map(|source| {
        catalog
            .iter()
            .find(|m| split_wire_id(&m.id).is_some_and(|(s, _)| s == source))
            .map(|m| m.id.clone())
    })
}

fn render_setup_home(session: &Session, catalog: &[ModelMetadata]) -> String {
    render_setup_home_for_model(&session.model, catalog)
}

fn render_setup_home_from_snapshot(snap: &SessionSnapshot, catalog: &[ModelMetadata]) -> String {
    let mut out =
        String::from("No model is ready yet. Start setup before asking Anvil to work.\n\n");
    out.push_str(&render_setup_home_for_model(&snap.model, catalog));
    out
}

fn render_setup_home_for_model(model: &str, catalog: &[ModelMetadata]) -> String {
    let bedrock_count = source_count(catalog, ModelSource::Bedrock);
    let codex_count = source_count(catalog, ModelSource::Codex);
    let ollama_count = source_count(catalog, ModelSource::Ollama);
    let deepseek_count = source_count(catalog, ModelSource::DeepSeek);
    let openrouter_count = source_count(catalog, ModelSource::OpenRouter);
    let openrouter_state = crate::openrouter_auth::CredentialState::snapshot();
    let deepseek_env_ready = std::env::var(crate::discovery::DEEPSEEK_API_KEY_ENV)
        .ok()
        .is_some_and(|raw| !raw.trim().is_empty());
    let ready = if model.is_empty() {
        "No model selected yet.".to_string()
    } else {
        "A model is selected.".to_string()
    };

    format!(
        "**Anvil setup**\n\n\
         {ready}\n\n\
         Pick one:\n\
         - `/setup choose` - Choose for me.\n\
         - `/setup codex` - Use Codex or ChatGPT sign-in.\n\
         - `/setup bedrock` - Use AWS Bedrock.\n\
         - `/setup local` - Use free local models on this computer.\n\
         - Set `DEEPSEEK_API_KEY` - Use hosted DeepSeek.\n\
         - `/setup openrouter` - Use OpenRouter.\n\
         - `/setup advanced` - Show model ids and extra settings.\n\n\
         Found now:\n\
         - Bedrock: {bedrock_status}\n\
         - Codex: {codex_status}\n\
         - Local models: {local_status}\n\
         - DeepSeek: {deepseek_status}\n\
         - OpenRouter: {openrouter_status}\n\n\
         You can run `/setup` anytime.",
        bedrock_status = if bedrock_count > 0 {
            "ready".to_string()
        } else {
            "not connected".to_string()
        },
        codex_status = if codex_count > 0 {
            "ready".to_string()
        } else {
            "not signed in".to_string()
        },
        local_status = if ollama_count > 0 {
            "ready".to_string()
        } else {
            "not found".to_string()
        },
        deepseek_status = if deepseek_count > 0 {
            "ready".to_string()
        } else if deepseek_env_ready {
            "connected, no models found yet".to_string()
        } else {
            "not connected".to_string()
        },
        openrouter_status = if openrouter_count > 0 {
            "ready".to_string()
        } else if openrouter_state.active_source() == "none" {
            "not connected".to_string()
        } else {
            "connected, no models found yet".to_string()
        },
    )
}

/// Build and run the ACP agent over stdio.
pub async fn run_agent(
    llm: Arc<MultiBackend>,
    sessions: SessionStore,
    max_turns: usize,
    default_idle_timeout_secs: u64,
) -> agent_client_protocol::Result<()> {
    let llm_init = llm.clone();
    let sessions_init = sessions.clone();

    let llm_new = llm.clone();
    let sessions_new = sessions.clone();
    // Throttle background discovery refreshes so a burst of session/new
    // calls (e.g. an editor reconnecting and re-creating sessions) doesn't
    // pile up redundant probes against /v1/models and /codex/models. We
    // hold this owned Mutex via try_lock_owned: when a refresh is already
    // in flight, the next try_lock returns None and we skip the spawn.
    //
    // The same lock is shared with the `/codex-login` post-install
    // refresh below so an immediate session/new after login doesn't race
    // a second probe through the discovery path.
    let refresh_lock = Arc::new(tokio::sync::Mutex::new(()));
    let refresh_lock_new = refresh_lock.clone();
    let refresh_lock_login = refresh_lock.clone();

    let sessions_load = sessions.clone();
    let sessions_resume = sessions.clone();
    let sessions_fork = sessions.clone();
    let sessions_list = sessions.clone();

    let llm_prompt = llm.clone();
    let llm_login = llm.clone();
    let sessions_prompt = sessions.clone();
    let sessions_login = sessions.clone();

    let sessions_cancel = sessions.clone();
    let sessions_close = sessions.clone();
    let sessions_delete = sessions.clone();
    let sessions_mode = sessions.clone();
    let sessions_perm = sessions.clone();

    Agent
        .builder()
        .name("brokk-acp-rust")
        // Handle initialize
        .on_receive_request(
            async move |req: InitializeRequest,
                        responder: Responder<InitializeResponse>,
                        _cx: ConnectionTo<Client>| {
                tracing::info!("ACP initialize");

                // Try to discover models at startup and cache them for session/new.
                let models = match llm_init.list_model_metadata_with_progress(None).await {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::warn!("model discovery failed during init: {e}");
                        vec![]
                    }
                };
                let current_default_model = sessions_init.default_model().await;
                if current_default_model.trim().is_empty()
                    && let Some(first) = models.first()
                {
                    sessions_init.set_default_model(first.id.clone()).await;
                }
                sessions_init.set_available_models(models).await;

                let capabilities = AgentCapabilities::new()
                    .load_session(true)
                    .prompt_capabilities(
                        PromptCapabilities::new().embedded_context(true).image(true),
                    )
                    // Anvil only speaks to stdio MCP subprocesses, so advertise
                    // http/sse as unsupported. Lifecycle requests carrying other
                    // transports are rejected rather than silently dropped (#159).
                    .mcp_capabilities(McpCapabilities::new())
                    .session_capabilities(
                        SessionCapabilities::new()
                            .list(SessionListCapabilities::new())
                            .resume(SessionResumeCapabilities::new())
                            .close(SessionCloseCapabilities::new())
                            .delete(SessionDeleteCapabilities::new())
                            .fork(SessionForkCapabilities::new()),
                    );

                let protocol_version = negotiate_protocol_version(req.protocol_version);
                responder.respond(
                    InitializeResponse::new(protocol_version).agent_capabilities(capabilities),
                )
            },
            on_receive_request!(),
        )
        // Handle session/new
        .on_receive_request(
            async move |req: NewSessionRequest,
                        responder: Responder<NewSessionResponse>,
                        cx: ConnectionTo<Client>| {
                let cwd = req.cwd.clone();
                tracing::info!("ACP session/new, cwd={}", cwd.display());
                if !cwd.is_absolute() {
                    tracing::warn!("session/new rejected relative cwd={}", cwd.display());
                    return responder.respond_with_error(invalid_lifecycle_cwd_error(
                        "session/new",
                        &cwd,
                    ));
                }
                if !req.additional_directories.is_empty() {
                    tracing::warn!(
                        "session/new rejected {} additionalDirectories (unsupported)",
                        req.additional_directories.len()
                    );
                    return responder
                        .respond_with_error(unsupported_additional_directories_error("session/new"));
                }
                let session_mcp_servers = match acp_mcp_servers_to_configs(req.mcp_servers) {
                    Ok(servers) => servers,
                    Err(err) => {
                        tracing::warn!(
                            "session/new rejected unsupported MCP transport '{}' for server '{}'",
                            err.transport,
                            err.server
                        );
                        return responder
                            .respond_with_error(unsupported_mcp_transport_error("session/new", &err));
                    }
                };
                let session = sessions_new
                    .create_session_with_mcp_servers(cwd, Some(session_mcp_servers))
                    .await;

                // Use the cached catalog populated at init; fall back to a
                // single-entry catalog from the session's own model so the
                // dropdown still renders something on a fresh discovery miss.
                let mut catalog = sessions_new.available_model_metadata().await;
                let should_stream_refresh = catalog.is_empty() || session.model.is_empty();
                // Re-discover in the background so the next `session/new` picks up
                // models the user added/removed since startup (e.g. they ran
                // `ollama pull` or signed into Codex). When this session starts
                // without a usable cached catalog, stream the same progress into
                // its transcript after the client finishes registering the id.
                spawn_background_refresh(
                    refresh_lock_new.clone(),
                    llm_new.clone(),
                    sessions_new.clone(),
                    should_stream_refresh.then(|| {
                        (
                            cx.clone(),
                            session.id.clone(),
                            "Checking model providers for this session...",
                        )
                    }),
                    should_stream_refresh.then_some(Duration::from_millis(150)),
                );
                if catalog.is_empty() && !session.model.is_empty() {
                    catalog = vec![ModelMetadata::id_only(&session.model)];
                }
                let model_ids: Vec<String> = catalog.iter().map(|m| m.id.clone()).collect();
                let setup_session = session.clone();
                let setup_catalog = catalog.clone();

                let meta_value = serde_json::json!({
                    "brokk": {
                        "modelId": session.model,
                        "availableModels": model_ids,
                    }
                });
                let meta_map = match meta_value {
                    serde_json::Value::Object(m) => m,
                    _ => serde_json::Map::new(),
                };

                let response = NewSessionResponse::new(session.id.clone())
                    .modes(mode_state(session.mode.as_str()))
                    .config_options(all_config_options(
                        session.mode,
                        session.permission_mode,
                        &session.model,
                        &catalog,
                        session.selected_reasoning_effort.as_deref(),
                    ))
                    .meta(meta_map);

                // Respond first so the client receives the session id, then
                // schedule the available-commands notification on a short
                // delay so it lands on Zed *after* its `new_session` handler
                // has inserted the session id into its sessions map. See
                // `spawn_delayed_available_commands_update` for the full
                // rationale (FIFO wire order alone is not enough on the
                // session/new path).
                let result = responder.respond(response);
                spawn_delayed_available_commands_update(
                    cx.clone(),
                    session.id.clone(),
                    session.skills.clone(),
                );
                spawn_delayed_session_usage_update(
                    cx.clone(),
                    sessions_new.clone(),
                    session.id.clone(),
                    session.cwd.clone(),
                );
                spawn_delayed_setup_notice(
                    cx.clone(),
                    setup_session,
                    setup_catalog,
                    sessions_new.clone(),
                );
                result
            },
            on_receive_request!(),
        )
        // Handle session/load
        .on_receive_request(
            async move |req: LoadSessionRequest,
                        responder: Responder<LoadSessionResponse>,
                        cx: ConnectionTo<Client>| {
                let session_id = req.session_id.to_string();
                let cwd = req.cwd.clone();
                tracing::info!(
                    "ACP session/load session={session_id}, cwd={}",
                    cwd.display()
                );
                if !cwd.is_absolute() {
                    tracing::warn!(
                        "session/load rejected relative cwd={} for session={session_id}",
                        cwd.display()
                    );
                    return responder.respond_with_error(invalid_lifecycle_cwd_error(
                        "session/load",
                        &cwd,
                    ));
                }
                if !req.additional_directories.is_empty() {
                    tracing::warn!(
                        "session/load rejected {} additionalDirectories (unsupported) for session={session_id}",
                        req.additional_directories.len()
                    );
                    return responder.respond_with_error(
                        unsupported_additional_directories_error("session/load"),
                    );
                }
                // Convert (and validate) the requested MCP servers before any
                // session work, so an unsupported transport is rejected early
                // (#159). The converted set is applied after the session loads
                // (#145).
                let requested_mcp_servers = match acp_mcp_servers_to_configs(req.mcp_servers) {
                    Ok(servers) => servers,
                    Err(err) => {
                        tracing::warn!(
                            "session/load rejected unsupported MCP transport '{}' for server '{}'",
                            err.transport,
                            err.server
                        );
                        return responder.respond_with_error(unsupported_mcp_transport_error(
                            "session/load",
                            &err,
                        ));
                    }
                };

                // Look up the session from memory or disk, validating that the
                // request cwd matches the session's original cwd (#147). Unknown
                // ids are a protocol error, not a successful empty load (#154).
                let session = match sessions_load.reopen_session_checked(&session_id, &cwd).await {
                    LifecycleReopen::Reopened(session) => *session,
                    LifecycleReopen::CwdMismatch { session_cwd } => {
                        tracing::warn!(
                            "session/load cwd mismatch session={session_id}: session cwd={} request cwd={}",
                            session_cwd.display(),
                            cwd.display()
                        );
                        return responder.respond_with_error(lifecycle_cwd_mismatch_error(
                            "session/load",
                            &session_cwd,
                            &cwd,
                        ));
                    }
                    LifecycleReopen::Unknown => {
                        tracing::warn!("session/load: unknown session {session_id}");
                        return responder.respond_with_error(unknown_session_error(&session_id));
                    }
                };
                sessions_load.update_cwd(&session_id, cwd).await;
                // Apply the client-supplied MCP servers for this load, dropping
                // any cached registry so the next prompt rebuilds with them (#145).
                sessions_load
                    .apply_lifecycle_mcp_servers(&session_id, requested_mcp_servers)
                    .await;

                // Replay conversation history as session updates (both sides).
                for turn in &session.history {
                    if !turn.user_prompt.is_empty() {
                        send_user_message(&cx, &session_id, &turn.user_prompt);
                    }
                    if !turn.agent_response.is_empty() {
                        send_message(&cx, &session_id, &turn.agent_response);
                    }
                }

                let catalog = sessions_load.available_model_metadata().await;
                let setup_session = session.clone();
                let setup_catalog = catalog.clone();
                let result = responder.respond(
                    LoadSessionResponse::new()
                        .modes(mode_state(session.mode.as_str()))
                        .config_options(all_config_options(
                            session.mode,
                            session.permission_mode,
                            &session.model,
                            &catalog,
                            session.selected_reasoning_effort.as_deref(),
                        )),
                );
                spawn_delayed_available_commands_update(
                    cx.clone(),
                    session_id.clone(),
                    session.skills.clone(),
                );
                spawn_delayed_session_usage_update(
                    cx.clone(),
                    sessions_load.clone(),
                    session_id.clone(),
                    session.cwd.clone(),
                );
                spawn_delayed_setup_notice(
                    cx.clone(),
                    setup_session,
                    setup_catalog,
                    sessions_load.clone(),
                );
                result
            },
            on_receive_request!(),
        )
        // Handle session/resume
        .on_receive_request(
            async move |req: ResumeSessionRequest,
                        responder: Responder<ResumeSessionResponse>,
                        cx: ConnectionTo<Client>| {
                let session_id = req.session_id.to_string();
                let cwd = req.cwd.clone();
                tracing::info!(
                    "ACP session/resume session={session_id}, cwd={}",
                    cwd.display()
                );
                if !cwd.is_absolute() {
                    tracing::warn!(
                        "session/resume rejected relative cwd={} for session={session_id}",
                        cwd.display()
                    );
                    return responder.respond_with_error(invalid_lifecycle_cwd_error(
                        "session/resume",
                        &cwd,
                    ));
                }
                if !req.additional_directories.is_empty() {
                    tracing::warn!(
                        "session/resume rejected {} additionalDirectories (unsupported) for session={session_id}",
                        req.additional_directories.len()
                    );
                    return responder.respond_with_error(
                        unsupported_additional_directories_error("session/resume"),
                    );
                }
                // Reject unsupported MCP transports before any session work (#159);
                // apply the converted set after the session loads (#146).
                let requested_mcp_servers = match acp_mcp_servers_to_configs(req.mcp_servers) {
                    Ok(servers) => servers,
                    Err(err) => {
                        tracing::warn!(
                            "session/resume rejected unsupported MCP transport '{}' for server '{}'",
                            err.transport,
                            err.server
                        );
                        return responder.respond_with_error(unsupported_mcp_transport_error(
                            "session/resume",
                            &err,
                        ));
                    }
                };

                // Validate cwd consistency (#147); unknown ids are a protocol
                // error, not a successful empty resume (#154).
                let session = match sessions_resume.reopen_session_checked(&session_id, &cwd).await {
                    LifecycleReopen::Reopened(session) => *session,
                    LifecycleReopen::CwdMismatch { session_cwd } => {
                        tracing::warn!(
                            "session/resume cwd mismatch session={session_id}: session cwd={} request cwd={}",
                            session_cwd.display(),
                            cwd.display()
                        );
                        return responder.respond_with_error(lifecycle_cwd_mismatch_error(
                            "session/resume",
                            &session_cwd,
                            &cwd,
                        ));
                    }
                    LifecycleReopen::Unknown => {
                        tracing::warn!("session/resume: unknown session {session_id}");
                        return responder.respond_with_error(unknown_session_error(&session_id));
                    }
                };
                sessions_resume.update_cwd(&session_id, cwd).await;
                // Apply the client-supplied MCP servers for this resume,
                // dropping any cached registry so the next prompt rebuilds with
                // them (#146).
                sessions_resume
                    .apply_lifecycle_mcp_servers(&session_id, requested_mcp_servers)
                    .await;
                let catalog = sessions_resume.available_model_metadata().await;
                let setup_session = session.clone();
                let setup_catalog = catalog.clone();
                let result = responder.respond(
                    ResumeSessionResponse::new()
                        .modes(mode_state(session.mode.as_str()))
                        .config_options(all_config_options(
                            session.mode,
                            session.permission_mode,
                            &session.model,
                            &catalog,
                            session.selected_reasoning_effort.as_deref(),
                        )),
                );
                spawn_delayed_available_commands_update(
                    cx.clone(),
                    session_id.clone(),
                    session.skills.clone(),
                );
                spawn_delayed_session_usage_update(
                    cx.clone(),
                    sessions_resume.clone(),
                    session_id.clone(),
                    session.cwd.clone(),
                );
                spawn_delayed_setup_notice(
                    cx.clone(),
                    setup_session,
                    setup_catalog,
                    sessions_resume.clone(),
                );
                result
            },
            on_receive_request!(),
        )
        // Handle session/fork
        .on_receive_request(
            async move |req: ForkSessionRequest,
                        responder: Responder<ForkSessionResponse>,
                        cx: ConnectionTo<Client>| {
                let source_id = req.session_id.to_string();
                let cwd = req.cwd.clone();
                tracing::info!(
                    "ACP session/fork source={source_id}, cwd={}",
                    cwd.display()
                );
                if !cwd.is_absolute() {
                    return responder
                        .respond_with_error(invalid_lifecycle_cwd_error("session/fork", &cwd));
                }
                if !req.additional_directories.is_empty() {
                    return responder.respond_with_error(
                        unsupported_additional_directories_error("session/fork"),
                    );
                }
                let requested_mcp_servers = match acp_mcp_servers_to_configs(req.mcp_servers) {
                    Ok(servers) => servers,
                    Err(err) => {
                        return responder
                            .respond_with_error(unsupported_mcp_transport_error("session/fork", &err));
                    }
                };

                // Fork copies the source's full persisted history into a new,
                // independent session id; the request cwd must match the
                // source's cwd (#147).
                let forked = match sessions_fork.fork_session(&source_id, &cwd).await {
                    ForkOutcome::Forked(session) => *session,
                    ForkOutcome::CwdMismatch { session_cwd } => {
                        return responder.respond_with_error(lifecycle_cwd_mismatch_error(
                            "session/fork",
                            &session_cwd,
                            &cwd,
                        ));
                    }
                    ForkOutcome::Unknown => {
                        return responder.respond_with_error(unknown_session_error(&source_id));
                    }
                    ForkOutcome::Failed(reason) => {
                        return responder.respond_with_error(
                            agent_client_protocol::Error::internal_error().data(
                                serde_json::json!({
                                    "reason": "failed to fork session",
                                    "details": reason,
                                }),
                            ),
                        );
                    }
                };
                let new_id = forked.id.clone();
                // Apply the request's MCP servers (replace) when supplied; an
                // empty set inherits the source's copied MCP config (#145/#146
                // semantics, but fork defaults to the source's config).
                if !requested_mcp_servers.is_empty() {
                    sessions_fork
                        .apply_lifecycle_mcp_servers(&new_id, requested_mcp_servers)
                        .await;
                }

                let catalog = sessions_fork.available_model_metadata().await;
                let setup_session = forked.clone();
                let setup_catalog = catalog.clone();
                let result = responder.respond(
                    ForkSessionResponse::new(new_id.clone())
                        .modes(mode_state(forked.mode.as_str()))
                        .config_options(all_config_options(
                            forked.mode,
                            forked.permission_mode,
                            &forked.model,
                            &catalog,
                            forked.selected_reasoning_effort.as_deref(),
                        )),
                );
                spawn_delayed_available_commands_update(
                    cx.clone(),
                    new_id.clone(),
                    forked.skills.clone(),
                );
                spawn_delayed_session_usage_update(
                    cx.clone(),
                    sessions_fork.clone(),
                    new_id.clone(),
                    forked.cwd.clone(),
                );
                spawn_delayed_setup_notice(
                    cx.clone(),
                    setup_session,
                    setup_catalog,
                    sessions_fork.clone(),
                );
                result
            },
            on_receive_request!(),
        )
        // Handle session/list
        .on_receive_request(
            async move |req: ListSessionsRequest,
                        responder: Responder<ListSessionsResponse>,
                        _cx: ConnectionTo<Client>| {
                tracing::info!(
                    "ACP session/list, cwd filter={:?}, cursor={:?}",
                    req.cwd,
                    req.cursor
                );

                // A supplied cwd filter must be absolute, matching the other
                // cwd-bearing lifecycle handlers (#143 keeps cwd optional).
                if let Some(cwd) = &req.cwd
                    && !cwd.is_absolute()
                {
                    tracing::warn!("session/list rejected relative cwd={}", cwd.display());
                    return responder
                        .respond_with_error(invalid_lifecycle_cwd_error("session/list", cwd));
                }

                // Resolve the page offset from the opaque cursor first; an
                // unrecognized cursor -- including one minted for a different
                // cwd context -- is a protocol error, not a silent restart at
                // the first page (#144).
                let context_tag = session_list_context_tag(req.cwd.as_deref());
                let offset = match req.cursor.as_deref() {
                    None => 0,
                    Some(cursor) => match parse_session_list_cursor(cursor, context_tag) {
                        Some(offset) => offset,
                        None => {
                            return responder.respond_with_error(
                                agent_client_protocol::Error::invalid_params().data(
                                    serde_json::json!({
                                        "reason": format!(
                                            "invalid session/list cursor: '{cursor}'"
                                        ),
                                    }),
                                ),
                            );
                        }
                    },
                };

                // With a cwd, list that workspace's persisted sessions; without
                // one, return the process's resident known sessions (#143).
                let entries: Vec<(SessionManifest, PathBuf)> = if let Some(cwd) = &req.cwd {
                    sessions_list
                        .list_sessions_from_disk(cwd)
                        .await
                        .into_iter()
                        .map(|manifest| (manifest, cwd.clone()))
                        .collect()
                } else {
                    sessions_list.resident_session_manifests().await
                };

                let (start, end, next_cursor) =
                    paginate_session_list(entries.len(), offset, context_tag);
                let infos: Vec<SessionInfo> = entries[start..end]
                    .iter()
                    .map(|(manifest, cwd)| session_info_from_manifest(manifest, cwd))
                    .collect();

                responder.respond(ListSessionsResponse::new(infos).next_cursor(next_cursor))
            },
            on_receive_request!(),
        )
        // Handle session/prompt
        .on_receive_request(
            async move |req: PromptRequest,
                        responder: Responder<PromptResponse>,
                        cx: ConnectionTo<Client>| {
                let session_id = req.session_id.to_string();
                tracing::info!("ACP session/prompt session={session_id}");

                // Extract prompt content from ACP blocks. Text drives slash-command
                // parsing and session titles; images are preserved for the LLM turn.
                let raw_prompt_text = extract_prompt_text(&req.prompt);
                let raw_prompt_parts = extract_prompt_parts(&req.prompt);
                if raw_prompt_parts.is_empty() {
                    // An empty prompt is an invalid request, not a completed
                    // turn: report it at the protocol layer so clients don't
                    // mistake it for a normal end-turn.
                    return responder.respond_with_error(
                        agent_client_protocol::Error::invalid_params().data(serde_json::json!({
                            "reason": "session/prompt requires at least one text, image, \
                                       resource link, or embedded resource content block",
                        })),
                    );
                }
                let structured_output_request = match parse_prompt_structured_output_request(&req) {
                    Ok(request) => request,
                    Err(reason) => {
                        return responder.respond_with_error(
                            agent_client_protocol::Error::invalid_params().data(
                                serde_json::json!({
                                    "reason": format!(
                                        "invalid structured output request metadata: {reason}"
                                    ),
                                }),
                            ),
                        );
                    }
                };

                // Get session state (prompt doesn't carry cwd, so use current dir as fallback).
                // The snapshot clones the conversation history exactly once under the
                // read lock; we then consume it via `.into_iter()` to build ChatMessages
                // without further string copies.
                let fallback_cwd = std::env::current_dir().unwrap_or_default();
                let mut snap = match sessions_prompt.snapshot(&session_id, &fallback_cwd).await {
                    Some(s) => s,
                    None => {
                        // Unknown session is a protocol-level invalid request,
                        // not a successful end-turn.
                        return responder.respond_with_error(unknown_session_error(&session_id));
                    }
                };

                // Slash commands run locally and short-circuit the LLM round-trip.
                // They are not persisted as conversation turns -- the response is
                // purely informational and replaying it on the next session load
                // would mislead the model about prior dialog. Mirrors the Java
                // executor's `handleSlashCommand` path.
                if is_slash_command(&raw_prompt_text, "context") {
                    let permission_mode = sessions_prompt
                        .permission_mode(&session_id)
                        .await
                        .unwrap_or_default();
                    let available_models = sessions_prompt.available_model_metadata().await;
                    let report = render_context_report(&snap, permission_mode, &available_models);
                    send_message(&cx, &session_id, &report);
                    return responder.respond(prompt_end_turn_response());
                }

                if is_slash_command(&raw_prompt_text, "usage") {
                    let usage = sessions_prompt
                        .cumulative_token_usage(&session_id)
                        .await
                        .unwrap_or_default();
                    let cost_usd = sessions_prompt.exact_usage_cost_usd(&session_id).await;
                    let (credits, codex_usage) = tokio::join!(
                        fetch_openrouter_credits_for_usage(&snap.model),
                        fetch_codex_credits_for_usage(&snap.model),
                    );
                    let report =
                        render_usage_report(&snap, usage, cost_usd, credits, codex_usage);
                    send_message(&cx, &session_id, &report);
                    return responder.respond(prompt_end_turn_response());
                }

                let loop_spec = if is_slash_command(&raw_prompt_text, "loop") {
                    match parse_loop_command(&raw_prompt_text) {
                        Ok(spec) => Some(spec),
                        Err(report) => {
                            send_message(&cx, &session_id, &report);
                            return responder.respond(prompt_end_turn_response());
                        }
                    }
                } else {
                    None
                };

                // `/goal <objective>` drives the agent autonomously across
                // turns until the objective is verifiably met (the model
                // emits the completion sentinel), it is blocked, or the
                // session is cancelled. Unbounded by default; an optional
                // `--max-turns` ceiling can cap it. Parsed here so a malformed
                // invocation prints usage and short-circuits, mirroring
                // `/loop`; the spawn that actually runs the goal loop is
                // dispatched further down, after model validation.
                let goal_spec = if is_slash_command(&raw_prompt_text, "goal") {
                    match parse_goal_command(&raw_prompt_text) {
                        Ok(spec) => Some(spec),
                        Err(report) => {
                            send_message(&cx, &session_id, &report);
                            return responder.respond(prompt_end_turn_response());
                        }
                    }
                } else {
                    None
                };

                let stream_setup_openrouter_refresh =
                    is_streamed_setup_openrouter_refresh(&raw_prompt_text);

                if is_slash_command(&raw_prompt_text, "setup") && !stream_setup_openrouter_refresh {
                    let setup_ctx = SetupContext {
                        cx: &cx,
                        sessions: &sessions_prompt,
                        llm: &llm_login,
                        login_sessions: &sessions_login,
                        refresh_lock: &refresh_lock_login,
                        default_idle_timeout_secs,
                        current_session_idle_timeout: snap.idle_timeout_secs,
                    };
                    let report = handle_setup(&setup_ctx, &raw_prompt_text, &session_id).await;
                    send_message(&cx, &session_id, &report);
                    return responder.respond(prompt_end_turn_response());
                }

                if is_slash_command(&raw_prompt_text, "permissions") {
                    let report =
                        handle_permissions(&sessions_prompt, &session_id, &raw_prompt_text).await;
                    send_message(&cx, &session_id, &report);
                    return responder.respond(prompt_end_turn_response());
                }

                if is_slash_command(&raw_prompt_text, "mcp") {
                    let report = handle_mcp(&raw_prompt_text, &sessions_prompt, &session_id).await;
                    send_message(&cx, &session_id, &report);
                    return responder.respond(prompt_end_turn_response());
                }

                if is_slash_command(&raw_prompt_text, "pr-create") {
                    let permission_mode = sessions_prompt
                        .permission_mode(&session_id)
                        .await
                        .unwrap_or_default();
                    let sandbox_mode = sessions_prompt.sandbox_mode(&session_id).await.flatten();
                    // Reuse the per-session ToolRegistry so shell calls
                    // route through the same `run_shell_command` dispatch
                    // (env scrub, sandbox, rlimits) the LLM tool path
                    // uses. The registry is created on demand if this is
                    // the session's first prompt.
                    let registry = sessions_prompt
                        .get_or_create_registry(&session_id, snap.cwd.clone())
                        .await;
                    let Some(registry) = registry else {
                        send_message(&cx, &session_id, "Error: unknown session");
                        return responder.respond(prompt_end_turn_response());
                    };
                    let report = handle_pr_create(
                        &raw_prompt_text,
                        &registry,
                        permission_mode,
                        sandbox_mode,
                    )
                    .await;
                    send_message(&cx, &session_id, &report);
                    return responder.respond(prompt_end_turn_response());
                }

                // User-explicit skill activation. Unlike the built-in
                // short-circuit commands above, a skill slash IS the LLM
                // round-trip: the SKILL.md body becomes the user's
                // message for this turn (with any args after the command
                // appended), so it persists into history and replays
                // correctly. Built-ins are checked first so a skill
                // that happens to name itself e.g. `context` or
                // `setup` can never shadow them.
                let slash_command = parse_slash_command(&raw_prompt_text);
                let title_seed = snap
                    .history
                    .first()
                    .map(|turn| turn.user_prompt.clone())
                    .filter(|text| !text.trim().is_empty())
                    .unwrap_or_else(|| prompt_text_for_title(&raw_prompt_text, &raw_prompt_parts));

                // Rename the session from its first prompt *before* any LLM
                // work starts. The title depends only on the user's text, not
                // on the model response, so there is no reason to defer it
                // past the spawn below.
                if should_auto_rename_session_from_prompt(&raw_prompt_text) {
                    match sessions_prompt
                        .maybe_rename_from_prompt(&session_id, &title_seed)
                        .await
                    {
                        Ok(renamed_title) => {
                            if renamed_title.is_some()
                                && let Some(metadata) =
                                    sessions_prompt.session_metadata(&session_id).await
                            {
                                send_session_info_update(
                                    &cx,
                                    &session_id,
                                    renamed_title,
                                    metadata.updated_at,
                                );
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                session_id = %session_id,
                                "failed to update session title: {e:#}"
                            );
                        }
                    }
                }

                let prompt_text = if let Some((name, args)) = slash_command.as_ref()
                    && let Some(meta) = snap.skills.get_for_slash_command(name)
                {
                    tracing::info!(skill = %meta.name, "slash-command activating skill");
                    sessions_prompt
                        .mark_skill_activated(&session_id, &meta.name)
                        .await;
                    let body = build_skill_payload(meta);
                    if args.is_empty() {
                        body
                    } else {
                        format!("{body}\n\nUser input: {args}")
                    }
                } else {
                    raw_prompt_text.clone()
                };
                let prompt_parts = if prompt_text == raw_prompt_text {
                    raw_prompt_parts
                } else {
                    vec![ChatContentPart::text(prompt_text.clone())]
                };

                if let Some(spec) = loop_spec.as_ref()
                    && snap.model.is_empty()
                    && !loop_target_runs_without_model(&spec.target)
                {
                    let catalog = sessions_prompt.available_model_metadata().await;
                    send_message(
                        &cx,
                        &session_id,
                        &render_setup_home_from_snapshot(&snap, &catalog),
                    );
                    return responder.respond(prompt_end_turn_response());
                }

                // Validate model is configured
                if snap.model.is_empty()
                    && !stream_setup_openrouter_refresh
                    && loop_spec.is_none()
                {
                    let catalog = sessions_prompt.available_model_metadata().await;
                    send_message(
                        &cx,
                        &session_id,
                        &render_setup_home_from_snapshot(&snap, &catalog),
                    );
                    return responder.respond(prompt_end_turn_response());
                }

                let available_models = sessions_prompt.available_model_metadata().await;
                if let Some(message) =
                    image_prompt_rejection(&snap.model, &prompt_parts, &available_models)
                {
                    send_message(&cx, &session_id, &format!("Error: {message}\n"));
                    return responder.respond(prompt_end_turn_response());
                }

                // Create a cancellation token for this prompt. Reject a
                // second in-flight prompt for the same session before we
                // spawn any background work.
                let cancel = match sessions_prompt.start_prompt(&session_id).await {
                    Ok(cancel) => cancel,
                    Err(PromptStartError::AlreadyInFlight) => {
                        tracing::warn!(
                            "rejecting concurrent ACP session/prompt session={session_id}"
                        );
                        return responder.respond_with_error(
                            agent_client_protocol::Error::invalid_params().data(
                                serde_json::json!({
                                    "reason": format!(
                                        "prompt already in flight for session '{session_id}'"
                                    ),
                                }),
                            ),
                        );
                    }
                    Err(PromptStartError::UnknownSession) => {
                        return responder.respond_with_error(unknown_session_error(&session_id));
                    }
                };

                // Resolve the model's declared context window once, here, so
                // the compression budget calc has it. Codex/Ollama models
                // typically don't publish one and fall through to the
                // per-backend default inside `context_budget`.
                let context_length = available_models
                    .iter()
                    .find(|m| m.id == snap.model)
                    .and_then(|m| m.context_length);
                // Idle timeout for the summarization LLM call mirrors the
                // resolution used for the main chat call below.
                let compression_idle_timeout = Duration::from_secs(
                    snap.idle_timeout_secs
                        .unwrap_or(default_idle_timeout_secs)
                        .max(1),
                );

                // `/compress` runs synchronously here (not via the
                // spawn task below) because it's a slash command that
                // produces a final report rather than a streamed LLM
                // turn. Dispatch *after* `start_prompt` so the user
                // can `session/cancel` mid-compress -- the cancel
                // token threads into `run_summarization`, aborting
                // any in-flight summarization stream, and the loop in
                // `handle_compress` checks the token between turns.
                // `finish_prompt` releases the session reservation
                // before we respond so a subsequent prompt isn't
                // rejected as AlreadyInFlight.
                if is_slash_command(&prompt_text, "compress") {
                    let report = handle_compress(
                        &snap,
                        llm_prompt.as_ref(),
                        &sessions_prompt,
                        &session_id,
                        cancel.clone(),
                        compression_idle_timeout,
                        context_length,
                        &cx,
                    )
                    .await;
                    send_message(&cx, &session_id, &report);
                    send_session_usage_update(&cx, &sessions_prompt, &session_id, &snap.cwd).await;
                    sessions_prompt.finish_prompt(&session_id).await;
                    // `/compress` threads the cancel token into summarization;
                    // a mid-compress `session/cancel` resolves as cancelled.
                    return responder.respond(prompt_stop_response(cancel.is_cancelled()));
                }

                if let Some(loop_spec) = loop_spec {
                    let llm_for_loop_turns: Arc<dyn crate::llm_client::LlmBackend> =
                        llm_prompt.clone();
                    let orchestration_model_for_response =
                        llm_for_loop_turns.resolve_model_info(&snap.model);
                    let llm_for_setup = llm_login.clone();
                    let sessions_for_loop = sessions_prompt.clone();
                    let cx_for_loop = cx.clone();
                    let session_id_for_loop = session_id.clone();
                    let fallback_cwd_for_loop = fallback_cwd.clone();
                    let refresh_lock_for_loop = refresh_lock_login.clone();
                    let structured_output_request_for_loop = structured_output_request.clone();

                    let spawn_result = cx.spawn(async move {
                        use futures::FutureExt;
                        use std::panic::AssertUnwindSafe;

                        let loop_result = AssertUnwindSafe(async {
                            send_message(
                                &cx_for_loop,
                                &session_id_for_loop,
                                &format!(
                                    "Starting `/loop`: every {}s run this target:\n{}\n\
                                     Cancel the session to stop.\n",
                                    loop_spec.interval_secs, loop_spec.target
                                ),
                            );

                            let mut iteration = 0u64;
                            let mut last_structured_output_result = None;
                            let mut last_cumulative_usage = None;

                            loop {
                                if cancel.is_cancelled() {
                                    send_message(
                                        &cx_for_loop,
                                        &session_id_for_loop,
                                        "Cancelled.\n",
                                    );
                                    break;
                                }

                                iteration += 1;
                                send_thought(
                                    &cx_for_loop,
                                    &session_id_for_loop,
                                    &format!(
                                        "\n[loop iteration {iteration} | every {}s]\n",
                                        loop_spec.interval_secs
                                    ),
                                );
                                send_user_message(
                                    &cx_for_loop,
                                    &session_id_for_loop,
                                    &loop_spec.target,
                                );

                                match run_loop_iteration(
                                    &cx_for_loop,
                                    &sessions_for_loop,
                                    &session_id_for_loop,
                                    &fallback_cwd_for_loop,
                                    llm_for_loop_turns.clone(),
                                    llm_for_setup.clone(),
                                    &refresh_lock_for_loop,
                                    &loop_spec.target,
                                    structured_output_request_for_loop.as_ref(),
                                    default_idle_timeout_secs,
                                    max_turns,
                                    cancel.clone(),
                                )
                                .await
                                {
                                    Ok(outcome) => {
                                        last_structured_output_result =
                                            outcome.structured_output_result;
                                        last_cumulative_usage = Some(outcome.cumulative_usage);
                                    }
                                    Err(LoopIterationError::Terminal(err)) => {
                                        send_message(
                                            &cx_for_loop,
                                            &session_id_for_loop,
                                            &format!(
                                                "Loop iteration {iteration} stopped: {err}\n"
                                            ),
                                        );
                                        break;
                                    }
                                }

                                tokio::select! {
                                    _ = cancel.cancelled() => {
                                        send_message(&cx_for_loop, &session_id_for_loop, "Cancelled.\n");
                                        break;
                                    }
                                    _ = tokio::time::sleep(Duration::from_secs(loop_spec.interval_secs)) => {}
                                }
                            }

                            (last_structured_output_result, last_cumulative_usage)
                        })
                        .catch_unwind()
                        .await;

                        let (structured_output_result, cumulative_usage) = match loop_result {
                            Ok(state) => state,
                            Err(panic) => {
                                tracing::error!(
                                    session_id = %session_id_for_loop,
                                    "loop dispatcher panicked: {:?}",
                                    panic
                                );
                                send_message(
                                    &cx_for_loop,
                                    &session_id_for_loop,
                                    "Error: loop dispatcher panicked. See server logs.\n",
                                );
                                (None, None)
                            }
                        };

                        sessions_for_loop.finish_prompt(&session_id_for_loop).await;
                        // `/loop` exits its iteration loop when the cancel
                        // token fires; report that as a cancelled turn.
                        let cancelled = cancel.is_cancelled();
                        let response = if let Some(cumulative_usage) = cumulative_usage {
                            let acp_usage = AcpUsage::new(
                                cumulative_usage.total_tokens(),
                                cumulative_usage.input_tokens,
                                cumulative_usage.output_tokens,
                            )
                            .thought_tokens(cumulative_usage.thought_tokens)
                            .cached_read_tokens(cumulative_usage.cached_read_tokens)
                            .cached_write_tokens(cumulative_usage.cached_write_tokens);
                            prompt_stop_response(cancelled).usage(Some(acp_usage))
                        } else {
                            prompt_stop_response(cancelled)
                        };
                        let response = response.meta(prompt_response_meta(
                            structured_output_result.as_ref(),
                            Some(&orchestration_model_for_response),
                        ));
                        if let Err(e) = responder.respond(response) {
                            tracing::warn!(
                                session_id = %session_id_for_loop,
                                "failed to deliver PromptResponse: {e}"
                            );
                        }
                        Ok(())
                    });

                    if let Err(e) = spawn_result {
                        sessions_prompt.finish_prompt(&session_id).await;
                        return Err(e);
                    }

                    return Ok(());
                }

                // `/goal` dispatch. Like `/loop`, the goal runs in a spawned
                // task that holds the session's in-flight slot until the
                // objective is met, blocked, or the optional ceiling is hit;
                // `session/cancel` stops it early. Reaching this point means
                // the model is configured (the empty-model guard above only
                // skips for `loop_spec`, and a `/goal` prompt has none).
                if let Some(goal_spec) = goal_spec {
                    let llm_for_goal: Arc<dyn crate::llm_client::LlmBackend> = llm_prompt.clone();
                    let orchestration_model_for_response =
                        llm_for_goal.resolve_model_info(&snap.model);
                    let sessions_for_goal = sessions_prompt.clone();
                    let cx_for_goal = cx.clone();
                    let session_id_for_goal = session_id.clone();
                    let fallback_cwd_for_goal = fallback_cwd.clone();

                    let spawn_result = cx.spawn(async move {
                        use futures::FutureExt;
                        use std::panic::AssertUnwindSafe;

                        let goal_result = AssertUnwindSafe(run_goal_loop(
                            &cx_for_goal,
                            &sessions_for_goal,
                            &session_id_for_goal,
                            &fallback_cwd_for_goal,
                            llm_for_goal,
                            &goal_spec,
                            default_idle_timeout_secs,
                            max_turns,
                            cancel.clone(),
                        ))
                        .catch_unwind()
                        .await;

                        let cumulative_usage = match goal_result {
                            Ok(usage) => usage,
                            Err(panic) => {
                                tracing::error!(
                                    session_id = %session_id_for_goal,
                                    "goal dispatcher panicked: {:?}",
                                    panic
                                );
                                send_message(
                                    &cx_for_goal,
                                    &session_id_for_goal,
                                    "Error: goal dispatcher panicked. See server logs.\n",
                                );
                                crate::llm_client::TokenUsage::default()
                            }
                        };

                        sessions_for_goal.finish_prompt(&session_id_for_goal).await;
                        // `run_goal_loop` returns when the goal is met, blocked,
                        // or cancelled; the token distinguishes cancellation.
                        let cancelled = cancel.is_cancelled();
                        let acp_usage = AcpUsage::new(
                            cumulative_usage.total_tokens(),
                            cumulative_usage.input_tokens,
                            cumulative_usage.output_tokens,
                        )
                        .thought_tokens(cumulative_usage.thought_tokens)
                        .cached_read_tokens(cumulative_usage.cached_read_tokens)
                        .cached_write_tokens(cumulative_usage.cached_write_tokens);
                        let response = prompt_stop_response(cancelled)
                            .usage(Some(acp_usage))
                            .meta(prompt_response_meta(
                                None,
                                Some(&orchestration_model_for_response),
                            ));
                        if let Err(e) = responder.respond(response) {
                            tracing::warn!(
                                session_id = %session_id_for_goal,
                                "failed to deliver PromptResponse: {e}"
                            );
                        }
                        Ok(())
                    });

                    if let Err(e) = spawn_result {
                        sessions_prompt.finish_prompt(&session_id).await;
                        return Err(e);
                    }

                    return Ok(());
                }

                if stream_setup_openrouter_refresh {
                    let llm_for_refresh = llm_login.clone();
                    let sessions_for_refresh = sessions_prompt.clone();
                    let cx_for_refresh = cx.clone();
                    let session_id_for_refresh = session_id.clone();
                    let refresh_lock_for_refresh = refresh_lock_login.clone();
                    let cancel_for_refresh = cancel.clone();

                    let spawn_result = cx.spawn(async move {
                        let report = match refresh_model_catalog_now(
                            Some(&cx_for_refresh),
                            Some(&session_id_for_refresh),
                            &llm_for_refresh,
                            &sessions_for_refresh,
                            &refresh_lock_for_refresh,
                        )
                        .await
                        {
                            Ok(catalog) => {
                                let count = source_count(&catalog, ModelSource::OpenRouter);
                                if count > 0 {
                                    "OpenRouter models are ready. Run `/setup choose`, or use `/setup model` for advanced selection.".to_string()
                                } else {
                                    format!(
                                        "OpenRouter is not showing models yet.\n\n{}",
                                        render_openrouter_setup_help()
                                    )
                                }
                            }
                            Err(e) => format!(
                                "Could not check OpenRouter yet: {e}\n\n{}",
                                render_openrouter_setup_help()
                            ),
                        };
                        send_message(&cx_for_refresh, &session_id_for_refresh, &report);
                        sessions_for_refresh.finish_prompt(&session_id_for_refresh).await;
                        if let Err(e) = responder
                            .respond(prompt_stop_response(cancel_for_refresh.is_cancelled()))
                        {
                            tracing::warn!(
                                session_id = %session_id_for_refresh,
                                "failed to deliver PromptResponse: {e}"
                            );
                        }
                        Ok(())
                    });

                    if let Err(e) = spawn_result {
                        sessions_prompt.finish_prompt(&session_id).await;
                        return Err(e);
                    }

                    return Ok(());
                }

                let messages = build_prompt_messages_with_compression(
                    &mut snap,
                    &prompt_text,
                    &prompt_parts,
                    llm_prompt.as_ref(),
                    &sessions_prompt,
                    &session_id,
                    cancel.clone(),
                    compression_idle_timeout,
                    context_length,
                )
                .await;

                // Build the tool registry up-front so we don't pay for it inside the spawn.
                let Some(registry) = sessions_prompt
                    .get_or_create_registry(&session_id, snap.cwd)
                    .await
                else {
                    sessions_prompt.finish_prompt(&session_id).await;
                    return responder.respond_with_error(unknown_session_error(&session_id));
                };

                // Capture everything the spawned task needs before we move into it.
                // The tool loop calls `block_task()` to await `session/request_permission`,
                // which is only safe when run inside `cx.spawn` (per the ACP SDK docs --
                // calling it directly from a request handler can deadlock the dispatch loop).
                //
                // The tool loop only needs the trait, so coerce the
                // concrete `Arc<MultiBackend>` here -- keeping the
                // multi-backend specific surface (e.g. `install_codex`)
                // out of the generic chat path.
                let llm_for_loop: Arc<dyn crate::llm_client::LlmBackend> = llm_prompt.clone();
                let sessions_for_loop = sessions_prompt.clone();
                let cx_for_loop = cx.clone();
                let session_id_for_loop = session_id.clone();
                let fallback_cwd_for_loop = fallback_cwd.clone();
                let prompt_text_for_turn = prompt_text;
                let model_for_loop = snap.model;
                let orchestration_model_for_response =
                    llm_for_loop.resolve_model_info(&model_for_loop);
                let reasoning_effort_for_loop = snap.reasoning_effort;
                // Resolve per-turn idle timeout: the session override wins,
                // otherwise fall back to the binary-wide default from
                // `--llm-idle-timeout-secs` / `BROKK_ACP_LLM_IDLE_TIMEOUT_SECS`.
                let idle_timeout_for_loop = Duration::from_secs(
                    snap.idle_timeout_secs
                        .unwrap_or(default_idle_timeout_secs)
                        .max(1),
                );
                // `cancel` is moved into the tool loop below, so keep a clone to
                // detect after the turn whether the prompt was cancelled.
                let cancel_status = cancel.clone();

                let spawn_result = cx.spawn(async move {
                    // The normal prompt path uses only the structured output and
                    // usage; `response`/`failure` are for autonomous drivers and
                    // are ignored here (errors were already streamed to the user).
                    let turn_result = run_model_turn_in_spawn(
                        &cx_for_loop,
                        &sessions_for_loop,
                        &session_id_for_loop,
                        &fallback_cwd_for_loop,
                        &llm_for_loop,
                        &registry,
                        &model_for_loop,
                        reasoning_effort_for_loop.as_deref(),
                        structured_output_request.as_ref(),
                        messages,
                        max_turns,
                        idle_timeout_for_loop,
                        cancel,
                        prompt_text_for_turn,
                    )
                    .await;
                    let structured_output_result = turn_result.structured_output;
                    let cumulative_usage = turn_result.cumulative_usage;

                    // Clean up cancellation token even on panic / persistence failure.
                    sessions_for_loop.finish_prompt(&session_id_for_loop).await;

                    // ACP `session/usage` RFD: PromptResponse.usage
                    // carries cumulative session totals. Field mapping:
                    // `total_tokens` is the sum of all categories
                    // (matches the spec example), `input_tokens` /
                    // `output_tokens` are uncached input and visible
                    // output respectively, with reasoning and cached
                    // reads split out so they aren't double-counted.
                    // `Usage` is `#[non_exhaustive]`, so we go through
                    // the builder API rather than struct literal syntax.
                    let acp_usage = AcpUsage::new(
                        cumulative_usage.total_tokens(),
                        cumulative_usage.input_tokens,
                        cumulative_usage.output_tokens,
                    )
                    .thought_tokens(cumulative_usage.thought_tokens)
                    .cached_read_tokens(cumulative_usage.cached_read_tokens)
                    .cached_write_tokens(cumulative_usage.cached_write_tokens);
                    // ACP: a turn aborted by `session/cancel` MUST resolve its
                    // prompt with the cancelled stop reason, even though the
                    // tool loop swallowed the cancellation and returned normally.
                    let response =
                        prompt_stop_response(cancel_status.is_cancelled()).usage(Some(acp_usage));
                    let response = response.meta(prompt_response_meta(
                        structured_output_result.as_ref(),
                        Some(&orchestration_model_for_response),
                    ));
                    if let Err(e) = responder.respond(response) {
                        tracing::warn!(
                            session_id = %session_id_for_loop,
                            "failed to deliver PromptResponse: {e}"
                        );
                    }
                    Ok(())
                });

                if let Err(e) = spawn_result {
                    // `start_prompt` already registered the in-flight token.
                    // If spawning fails, clear it here so the session does not
                    // stay permanently blocked on a prompt that never started.
                    sessions_prompt.finish_prompt(&session_id).await;
                    return Err(e);
                }

                Ok(())
            },
            on_receive_request!(),
        )
        // Handle session/cancel
        .on_receive_notification(
            async move |notification: CancelNotification,
                        _cx: ConnectionTo<Client>|
                        -> agent_client_protocol::Result<()> {
                let session_id = notification.session_id.to_string();
                tracing::info!("ACP cancel session={session_id}");
                sessions_cancel.cancel_prompt(&session_id).await;
                Ok(())
            },
            on_receive_notification!(),
        )
        // Handle session/close
        .on_receive_request(
            async move |req: CloseSessionRequest,
                        responder: Responder<CloseSessionResponse>,
                        _cx: ConnectionTo<Client>| {
                let session_id = req.session_id.to_string();
                tracing::info!("ACP close session={session_id}");

                match sessions_close.close_session(&session_id).await {
                    CloseSessionResult::Closed => responder.respond(CloseSessionResponse::new()),
                    CloseSessionResult::Unknown => responder.respond_with_error(
                        agent_client_protocol::Error::invalid_params().data(serde_json::json!({
                            "reason": format!("unknown session '{session_id}'"),
                        })),
                    ),
                    CloseSessionResult::AlreadyClosed => responder.respond_with_error(
                        agent_client_protocol::Error::invalid_params().data(serde_json::json!({
                            "reason": format!("session '{session_id}' is already closed"),
                        })),
                    ),
                }
            },
            on_receive_request!(),
        )
        // Handle session/delete
        .on_receive_request(
            async move |req: DeleteSessionRequest,
                        responder: Responder<DeleteSessionResponse>,
                        _cx: ConnectionTo<Client>| {
                let session_id = req.session_id.to_string();
                tracing::info!("ACP delete session={session_id}");

                // ACP `session/delete` is idempotent: it cancels any in-flight
                // prompt, drops per-session resources, removes the persisted
                // session from `session/list`, and succeeds even for unknown or
                // already-deleted sessions.
                let removed = sessions_delete.delete_session(&session_id).await;
                tracing::info!("ACP delete session={session_id} removed_archive={removed}");
                responder.respond(DeleteSessionResponse::new())
            },
            on_receive_request!(),
        )
        // Handle session/set_mode
        .on_receive_request(
            async move |req: SetSessionModeRequest,
                        responder: Responder<SetSessionModeResponse>,
                        cx: ConnectionTo<Client>| {
                let session_id = req.session_id.to_string();
                let mode_id = req.mode_id.to_string();
                tracing::info!("ACP set_mode session={session_id} mode={mode_id}");

                let Some(mode) = SessionMode::parse(&mode_id) else {
                    return responder.respond_with_error(
                        agent_client_protocol::Error::invalid_params().data(serde_json::json!({
                            "reason": format!("unknown mode '{mode_id}'"),
                            "supported": available_modes()
                                .iter()
                                .map(|m| m.id.to_string())
                                .collect::<Vec<_>>(),
                        })),
                    );
                };

                match sessions_mode.set_mode(&session_id, mode).await {
                    Ok(true) => {
                        // Config options supersede legacy modes, but Anvil
                        // exposes both. Keep clients on the config-options
                        // surface in sync by emitting a config_option_update
                        // with the complete current set after a mode change
                        // through the legacy modes API (#156).
                        if let Some(options) =
                            current_config_options(&sessions_mode, &session_id).await
                        {
                            let notification = SessionNotification::new(
                                session_id.clone(),
                                SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate::new(options)),
                            );
                            if let Err(e) = cx.send_notification(notification) {
                                tracing::warn!(
                                    "failed to send config_option_update after set_mode: {e}"
                                );
                            }
                        }
                        responder.respond(SetSessionModeResponse::new())
                    }
                    Ok(false) => responder.respond_with_error(
                        agent_client_protocol::Error::invalid_params().data(serde_json::json!({
                            "reason": format!("unknown session '{session_id}'"),
                        })),
                    ),
                    Err(e) => responder.respond_with_error(
                        agent_client_protocol::Error::internal_error().data(serde_json::json!({
                            "reason": "failed to persist session mode",
                            "details": format!("{e:#}"),
                        })),
                    ),
                }
            },
            on_receive_request!(),
        )
        // Handle session/set_config_option
        .on_receive_request(
            async move |req: SetSessionConfigOptionRequest,
                        responder: Responder<SetSessionConfigOptionResponse>,
                        cx: ConnectionTo<Client>| {
                let session_id = req.session_id.to_string();
                let config_id = req.config_id.to_string();
                let value = match &req.value {
                    SessionConfigOptionValue::ValueId { value } => value.to_string(),
                    SessionConfigOptionValue::Boolean { value } => value.to_string(),
                    _ => serde_json::to_string(&req.value)
                        .unwrap_or_else(|_| "<unsupported config value>".to_string()),
                };
                tracing::info!(
                    "ACP set_config_option session={session_id} config={config_id} value={value}"
                );

                let outcome = match apply_config_option(
                    &sessions_perm,
                    &session_id,
                    &config_id,
                    &value,
                )
                .await
                {
                    Ok(out) => out,
                    Err(ConfigApplyError::UnknownConfigId) => {
                        return responder.respond_with_error(
                            agent_client_protocol::Error::invalid_params().data(
                                serde_json::json!({
                                    "reason": format!("unknown configOption '{config_id}'"),
                                    "supported": CONFIGURE_KNOWN_KEYS,
                                }),
                            ),
                        );
                    }
                    Err(ConfigApplyError::InvalidValue { reason, supported }) => {
                        return responder.respond_with_error(
                            agent_client_protocol::Error::invalid_params().data(
                                serde_json::json!({
                                    "reason": reason,
                                    "supported": supported,
                                }),
                            ),
                        );
                    }
                    Err(ConfigApplyError::UnknownSession) => {
                        return responder.respond_with_error(
                            agent_client_protocol::Error::invalid_params().data(
                                serde_json::json!({
                                    "reason": format!("unknown session '{session_id}'"),
                                }),
                            ),
                        );
                    }
                    Err(ConfigApplyError::PersistFailed { details }) => {
                        return responder.respond_with_error(
                            agent_client_protocol::Error::internal_error().data(
                                serde_json::json!({
                                    "reason": "failed to persist session mode",
                                    "details": details,
                                }),
                            ),
                        );
                    }
                };

                // Auto-fallback notice: when changing the model dropped a
                // now-unsupported reasoning_effort pick, surface a
                // one-line system note so the silent change isn't
                // mysterious next time the user wonders why thoughts
                // shortened.
                if let Some(prev) = &outcome.cleared_reasoning {
                    send_message(
                        &cx,
                        &session_id,
                        &format!(
                            "Reasoning effort reset: `{prev}` is not supported by `{value}`. \
                             Using model default until you pick a level."
                        ),
                    );
                }

                send_config_option_change_updates(
                    &cx,
                    &session_id,
                    &config_id,
                    &value,
                    outcome.updated_options.clone(),
                );

                let fallback_cwd = std::env::current_dir().unwrap_or_default();
                send_session_usage_update(&cx, &sessions_perm, &session_id, &fallback_cwd).await;

                responder.respond(SetSessionConfigOptionResponse::new(outcome.updated_options))
            },
            on_receive_request!(),
        )
        // Fallback: return unhandled for unknown messages
        .on_receive_dispatch(
            async move |message: Dispatch, _cx: ConnectionTo<Client>| {
                tracing::debug!("unhandled dispatch: {}", message.method());
                Ok(Handled::No {
                    message,
                    retry: false,
                })
            },
            on_receive_dispatch!(),
        )
        .connect_to(ByteStreams::new(
            tokio::io::stdout().compat_write(),
            tokio::io::stdin().compat(),
        ))
        .await
}

/// Extract text content from ACP content blocks.
fn extract_prompt_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Convert ACP prompt blocks into the internal multimodal chat content
/// representation. Baseline text is preserved verbatim; images are
/// forwarded as either data URLs (for inline base64) or URLs (when the
/// client supplied a URI without inline bytes).
///
/// ACP requires baseline agents to support resource links, and Anvil
/// advertises `embeddedContext`, so both are handled here rather than
/// silently dropped: resource links become explicit textual references and
/// embedded resources become inline text (or an image part for image
/// blobs). Audio is still ignored because the agent has not advertised
/// audio support.
fn extract_prompt_parts(blocks: &[ContentBlock]) -> Vec<ChatContentPart> {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(t) if !t.text.is_empty() => Some(ChatContentPart::text(&t.text)),
            ContentBlock::Image(image) if !image.data.is_empty() => Some(
                ChatContentPart::image_data(image.data.clone(), image.mime_type.as_str()),
            ),
            ContentBlock::Image(image) => image
                .uri
                .as_ref()
                .map(|uri| ChatContentPart::image_url(uri.clone())),
            ContentBlock::ResourceLink(link) => {
                Some(ChatContentPart::text(resource_link_to_text(link)))
            }
            ContentBlock::Resource(resource) => Some(embedded_resource_to_part(resource)),
            _ => None,
        })
        .collect()
}

/// Render an ACP `ResourceLink` as textual context for the model.
///
/// ACP baseline prompt support requires agents to accept resource links;
/// Anvil does not resolve the referenced bytes (that would require client
/// filesystem round-trips), so it surfaces the reference -- name, uri, and
/// any human-readable hints -- as text. This keeps the link visible to the
/// model and ensures a resource-link-only prompt is not mistaken for an
/// empty prompt.
fn resource_link_to_text(link: &ResourceLink) -> String {
    let mut out = format!("[resource link: {}", link.name);
    if let Some(title) = link.title.as_deref()
        && title != link.name
    {
        out.push_str(&format!(" ({title})"));
    }
    out.push_str(&format!("; uri: {}", link.uri));
    if let Some(mime) = link.mime_type.as_deref() {
        out.push_str(&format!("; mimeType: {mime}"));
    }
    if let Some(desc) = link.description.as_deref() {
        out.push_str(&format!("; {desc}"));
    }
    out.push(']');
    out
}

/// Convert an ACP embedded `Resource` block into a chat content part.
///
/// Text resources are surfaced inline (tagged with their uri) so embedded
/// context reaches the model, satisfying the advertised `embeddedContext`
/// capability. Image blobs are forwarded as image parts for vision models;
/// any other binary blob becomes a textual placeholder so the context is
/// acknowledged rather than silently dropped.
fn embedded_resource_to_part(resource: &EmbeddedResource) -> ChatContentPart {
    match &resource.resource {
        EmbeddedResourceResource::TextResourceContents(text) => {
            ChatContentPart::text(format!("[embedded resource: {}]\n{}", text.uri, text.text))
        }
        EmbeddedResourceResource::BlobResourceContents(blob) => match blob.mime_type.as_deref() {
            Some(mime) if mime.starts_with("image/") && !blob.blob.is_empty() => {
                ChatContentPart::image_data(blob.blob.clone(), mime)
            }
            mime => ChatContentPart::text(format!(
                "[embedded binary resource: {} ({})]",
                blob.uri,
                mime.unwrap_or("application/octet-stream")
            )),
        },
        // `EmbeddedResourceResource` is `#[non_exhaustive]`; surface
        // unrecognized future variants as text rather than dropping them.
        _ => ChatContentPart::text("[unsupported embedded resource]".to_string()),
    }
}

fn prompt_parts_include_images(parts: &[ChatContentPart]) -> bool {
    parts
        .iter()
        .any(|part| matches!(part, ChatContentPart::Image { .. }))
}

fn image_prompt_rejection(
    model: &str,
    prompt_parts: &[ChatContentPart],
    catalog: &[ModelMetadata],
) -> Option<String> {
    if !prompt_parts_include_images(prompt_parts) {
        return None;
    }
    let supports_images = catalog
        .iter()
        .find(|meta| meta.id == model)
        .and_then(|meta| meta.supports_images);
    (supports_images == Some(false)).then_some(
        "The selected model does not advertise image input support. Choose a vision-capable model to use image prompts.".to_string(),
    )
}

fn prompt_text_for_title(text: &str, parts: &[ChatContentPart]) -> String {
    if !text.trim().is_empty() {
        return text.to_string();
    }
    let image_count = parts
        .iter()
        .filter(|part| matches!(part, ChatContentPart::Image { .. }))
        .count();
    if image_count == 1 {
        "[image]".to_string()
    } else if image_count > 1 {
        format!("[{image_count} images]")
    } else {
        String::new()
    }
}

/// Send an agent_message_chunk session update to the client.
fn send_message(cx: &ConnectionTo<Client>, session_id: &str, text: &str) {
    let chunk = ContentChunk::new(ContentBlock::Text(TextContent::new(text)));
    let update = SessionUpdate::AgentMessageChunk(chunk);
    let notification = SessionNotification::new(session_id.to_string(), update);
    if let Err(e) = cx.send_notification(notification) {
        tracing::warn!("failed to send session update: {e}");
    }
}

fn trace_openrouter_refresh(line: &str) {
    crate::openrouter_auth::append_refresh_log(line);
}

/// Send a user_message_chunk session update to the client (used when replaying history).
fn send_user_message(cx: &ConnectionTo<Client>, session_id: &str, text: &str) {
    let chunk = ContentChunk::new(ContentBlock::Text(TextContent::new(text)));
    let update = SessionUpdate::UserMessageChunk(chunk);
    let notification = SessionNotification::new(session_id.to_string(), update);
    if let Err(e) = cx.send_notification(notification) {
        tracing::warn!("failed to send user session update: {e}");
    }
}

/// Send an agent_thought_chunk session update to the client. Mirrors
/// `send_message` but routes through ACP 0.12's `AgentThoughtChunk`
/// variant so the client renders reasoning text as a distinct,
/// typically-collapsible block instead of interleaving it with the
/// final answer.
fn send_thought(cx: &ConnectionTo<Client>, session_id: &str, text: &str) {
    let chunk = ContentChunk::new(ContentBlock::Text(TextContent::new(text)));
    let update = SessionUpdate::AgentThoughtChunk(chunk);
    let notification = SessionNotification::new(session_id.to_string(), update);
    if let Err(e) = cx.send_notification(notification) {
        tracing::warn!("failed to send thought session update: {e}");
    }
}

#[derive(Debug)]
enum LoopIterationError {
    Terminal(String),
}

struct LoopIterationOutcome {
    structured_output_result: Option<StructuredOutputResult>,
    cumulative_usage: crate::llm_client::TokenUsage,
}

impl LoopIterationOutcome {
    fn without_usage() -> Self {
        Self {
            structured_output_result: None,
            cumulative_usage: crate::llm_client::TokenUsage::default(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_loop_iteration(
    cx: &ConnectionTo<Client>,
    sessions: &SessionStore,
    session_id: &str,
    fallback_cwd: &Path,
    llm: Arc<dyn crate::llm_client::LlmBackend>,
    llm_setup: Arc<MultiBackend>,
    refresh_lock: &Arc<tokio::sync::Mutex<()>>,
    target: &str,
    structured_output_request: Option<&StructuredOutputRequest>,
    default_idle_timeout_secs: u64,
    max_turns: usize,
    cancel: tokio_util::sync::CancellationToken,
) -> Result<LoopIterationOutcome, LoopIterationError> {
    let mut snap = sessions
        .snapshot(session_id, fallback_cwd)
        .await
        .ok_or_else(|| LoopIterationError::Terminal("unknown session".to_string()))?;

    if is_slash_command(target, "context") {
        let permission_mode = sessions
            .permission_mode(session_id)
            .await
            .unwrap_or_default();
        let available_models = sessions.available_model_metadata().await;
        send_message(
            cx,
            session_id,
            &render_context_report(&snap, permission_mode, &available_models),
        );
        return Ok(LoopIterationOutcome::without_usage());
    }

    if is_slash_command(target, "usage") {
        let usage = sessions
            .cumulative_token_usage(session_id)
            .await
            .unwrap_or_default();
        let cost_usd = sessions.exact_usage_cost_usd(session_id).await;
        let (credits, codex_usage) = tokio::join!(
            fetch_openrouter_credits_for_usage(&snap.model),
            fetch_codex_credits_for_usage(&snap.model),
        );
        send_message(
            cx,
            session_id,
            &render_usage_report(&snap, usage, cost_usd, credits, codex_usage),
        );
        return Ok(LoopIterationOutcome::without_usage());
    }

    if is_slash_command(target, "setup") {
        let setup_ctx = SetupContext {
            cx,
            sessions,
            llm: &llm_setup,
            login_sessions: sessions,
            refresh_lock,
            default_idle_timeout_secs,
            current_session_idle_timeout: snap.idle_timeout_secs,
        };
        send_message(
            cx,
            session_id,
            &handle_setup(&setup_ctx, target, session_id).await,
        );
        return Ok(LoopIterationOutcome::without_usage());
    }

    if is_slash_command(target, "permissions") {
        send_message(
            cx,
            session_id,
            &handle_permissions(sessions, session_id, target).await,
        );
        return Ok(LoopIterationOutcome::without_usage());
    }

    if is_slash_command(target, "mcp") {
        send_message(
            cx,
            session_id,
            &handle_mcp(target, sessions, session_id).await,
        );
        return Ok(LoopIterationOutcome::without_usage());
    }

    if is_slash_command(target, "pr-create") {
        let permission_mode = sessions
            .permission_mode(session_id)
            .await
            .unwrap_or_default();
        let sandbox_mode = sessions.sandbox_mode(session_id).await.flatten();
        let Some(registry) = sessions
            .get_or_create_registry(session_id, snap.cwd.clone())
            .await
        else {
            return Err(LoopIterationError::Terminal("unknown session".to_string()));
        };
        send_message(
            cx,
            session_id,
            &handle_pr_create(target, &registry, permission_mode, sandbox_mode).await,
        );
        return Ok(LoopIterationOutcome::without_usage());
    }

    if is_slash_command(target, "compress") {
        let context_length = sessions
            .available_model_metadata()
            .await
            .iter()
            .find(|m| m.id == snap.model)
            .and_then(|m| m.context_length);
        let idle_timeout = Duration::from_secs(
            snap.idle_timeout_secs
                .unwrap_or(default_idle_timeout_secs)
                .max(1),
        );
        let report = handle_compress(
            &snap,
            llm.as_ref(),
            sessions,
            session_id,
            cancel,
            idle_timeout,
            context_length,
            cx,
        )
        .await;
        send_message(cx, session_id, &report);
        return Ok(LoopIterationOutcome::without_usage());
    }

    let raw_prompt_text = target.to_string();
    let raw_prompt_parts = vec![ChatContentPart::text(raw_prompt_text.clone())];
    let slash_command = parse_slash_command(&raw_prompt_text);
    let prompt_text = if let Some((name, args)) = slash_command.as_ref()
        && let Some(meta) = snap.skills.get_for_slash_command(name)
    {
        tracing::info!(skill = %meta.name, "loop activating skill");
        // `mark_skill_activated` writes into the session's HashSet of
        // activated skills, so repeated loop iterations are idempotent.
        sessions.mark_skill_activated(session_id, &meta.name).await;
        let body = build_skill_payload(meta);
        if args.is_empty() {
            body
        } else {
            format!("{body}\n\nUser input: {args}")
        }
    } else {
        raw_prompt_text.clone()
    };
    let prompt_parts = if prompt_text == raw_prompt_text {
        raw_prompt_parts
    } else {
        vec![ChatContentPart::text(prompt_text.clone())]
    };

    if snap.model.is_empty() {
        return Err(LoopIterationError::Terminal(
            "model not configured".to_string(),
        ));
    }

    let available_models = sessions.available_model_metadata().await;
    if let Some(message) = image_prompt_rejection(&snap.model, &prompt_parts, &available_models) {
        send_message(cx, session_id, &format!("Error: {message}\n"));
        return Ok(LoopIterationOutcome::without_usage());
    }

    let turn = run_prepared_model_turn(
        cx,
        sessions,
        session_id,
        fallback_cwd,
        &llm,
        &mut snap,
        &prompt_text,
        &prompt_parts,
        structured_output_request,
        default_idle_timeout_secs,
        max_turns,
        cancel,
    )
    .await?;
    Ok(LoopIterationOutcome {
        structured_output_result: turn.structured_output,
        cumulative_usage: turn.cumulative_usage,
    })
}

/// Outcome of a single autonomous goal turn: the assistant's final text
/// (scanned for the completion/blocked sentinel), the cumulative session
/// usage after the turn was accounted, and -- when the turn ended in an LLM
/// error or panic instead of a real completion -- the classified failure so
/// the loop can back off (transient) or stop (fatal).
struct GoalTurnOutcome {
    response: String,
    cumulative_usage: crate::llm_client::TokenUsage,
    failure: Option<crate::tool_loop::TurnFailure>,
}

/// Run one model turn for an active goal: inject `prompt_text` as the turn's
/// user message and run the shared per-turn pipeline to completion. Returns
/// the assistant text directly (for the sentinel scan) plus any failure
/// classification. The goal stop condition is the sentinel rather than schema
/// validation, so no structured-output request is threaded.
#[allow(clippy::too_many_arguments)]
async fn run_goal_turn(
    cx: &ConnectionTo<Client>,
    sessions: &SessionStore,
    session_id: &str,
    fallback_cwd: &Path,
    llm: Arc<dyn crate::llm_client::LlmBackend>,
    prompt_text: &str,
    default_idle_timeout_secs: u64,
    max_turns: usize,
    cancel: tokio_util::sync::CancellationToken,
) -> Result<GoalTurnOutcome, LoopIterationError> {
    let mut snap = sessions
        .snapshot(session_id, fallback_cwd)
        .await
        .ok_or_else(|| LoopIterationError::Terminal("unknown session".to_string()))?;

    let prompt_parts = vec![ChatContentPart::text(prompt_text.to_string())];
    let turn = run_prepared_model_turn(
        cx,
        sessions,
        session_id,
        fallback_cwd,
        &llm,
        &mut snap,
        prompt_text,
        &prompt_parts,
        // A goal stops on the completion sentinel, not on a schema, so it
        // never forces structured output on its turns.
        None,
        default_idle_timeout_secs,
        max_turns,
        cancel,
    )
    .await?;

    Ok(GoalTurnOutcome {
        response: turn.response,
        cumulative_usage: turn.cumulative_usage,
        failure: turn.failure,
    })
}

/// Drive a goal to completion across multiple autonomous turns.
///
/// Each iteration injects a continuation prompt (objective + completion
/// audit + sentinel protocol), runs a model turn, then inspects the
/// assistant's final text:
/// - [`GoalSignal::Complete`] → the objective is verifiably met; stop.
/// - [`GoalSignal::Blocked`] → count it; stop only once a blocker has been
///   reported for [`GOAL_BLOCKED_THRESHOLD`] consecutive turns (mirrors
///   Codex's "don't surrender on the first blocker" rule). The reasons need
///   not match -- any blocked report extends the streak.
/// - [`GoalSignal::Continue`] → keep going.
///
/// By default the goal is unbounded -- it runs until one of those signals
/// fires or the session is cancelled. The optional `--max-turns` ceiling on
/// the [`GoalSpec`] is a user opt-in: when set, the final allowed turn uses a
/// wrap-up framing so the agent leaves clean state before stopping.
/// Returns the cumulative session usage for the `PromptResponse`.
#[allow(clippy::too_many_arguments)]
async fn run_goal_loop(
    cx: &ConnectionTo<Client>,
    sessions: &SessionStore,
    session_id: &str,
    fallback_cwd: &Path,
    llm: Arc<dyn crate::llm_client::LlmBackend>,
    spec: &GoalSpec,
    default_idle_timeout_secs: u64,
    max_turns: usize,
    cancel: tokio_util::sync::CancellationToken,
) -> crate::llm_client::TokenUsage {
    let budget_note = match spec.max_turns {
        Some(max) => format!(" (optional ceiling: {max} turns)"),
        None => String::new(),
    };
    send_message(
        cx,
        session_id,
        &format!(
            "Starting `/goal`{budget_note}. I'll keep working across turns until the \
             objective is verifiably met or I'm blocked. Cancel the session to stop \
             early.\n\nObjective:\n{}\n",
            spec.objective.trim()
        ),
    );

    let mut cumulative = crate::llm_client::TokenUsage::default();
    let mut consecutive_blocked = 0u32;
    // Consecutive transient LLM failures (outage). Drives a capped backoff so
    // the goal survives an outage and resumes when it clears, instead of
    // spinning. Reset by any turn that produced a real model response.
    let mut consecutive_failures = 0u32;
    let mut turn = 0u32;

    loop {
        if cancel.is_cancelled() {
            send_message(cx, session_id, "Goal cancelled.\n");
            break;
        }

        turn += 1;
        // The ceiling only fires when the user opted into one; an unbounded
        // goal never treats a turn as "final" and runs until it completes,
        // blocks, or is cancelled.
        let final_turn = spec.max_turns.is_some_and(|max| turn >= max);
        let phase = if final_turn {
            GoalPhase::FinalWrapUp
        } else {
            GoalPhase::Continue
        };
        let prompt = build_goal_prompt(&spec.objective, turn, spec.max_turns, phase);

        let turn_label = match spec.max_turns {
            Some(max) => format!("\n[goal turn {turn}/{max}]\n"),
            None => format!("\n[goal turn {turn}]\n"),
        };
        send_thought(cx, session_id, &turn_label);

        match run_goal_turn(
            cx,
            sessions,
            session_id,
            fallback_cwd,
            llm.clone(),
            &prompt,
            default_idle_timeout_secs,
            max_turns,
            cancel.clone(),
        )
        .await
        {
            Ok(outcome) => {
                cumulative = outcome.cumulative_usage;

                // A turn that ended in an LLM failure produced no real
                // assistant response to scan for a sentinel. Classify it:
                // transient outages back off and retry (surviving the outage),
                // fatal errors stop and hand back to the user. Handled before
                // the sentinel scan so the error text can't be mistaken for a
                // signal.
                if let Some(failure) = outcome.failure {
                    match decide_after_goal_failure(&failure, consecutive_failures) {
                        GoalFailureAction::Stop => {
                            send_message(
                                cx,
                                session_id,
                                &format!(
                                    "\n⛔ Goal stopped after {turn} turn(s): the model request \
                                     failed and cannot be retried.\nReason: {}\n",
                                    failure.message
                                ),
                            );
                            break;
                        }
                        GoalFailureAction::Backoff {
                            consecutive_failures: updated,
                        } => {
                            consecutive_failures = updated;
                            let delay = goal_failure_backoff(consecutive_failures);
                            send_thought(
                                cx,
                                session_id,
                                &format!(
                                    "[goal: transient failure (attempt {consecutive_failures}): \
                                     {}; backing off {:.1}s and retrying]\n",
                                    failure.message,
                                    delay.as_secs_f64()
                                ),
                            );
                            // A failed turn doesn't consume the opt-in ceiling;
                            // retry reuses this turn number once the backoff
                            // (cancellable) elapses.
                            turn -= 1;
                            tokio::select! {
                                _ = cancel.cancelled() => {
                                    send_message(cx, session_id, "Goal cancelled.\n");
                                    break;
                                }
                                _ = tokio::time::sleep(delay) => {}
                            }
                            continue;
                        }
                    }
                }

                // A productive turn clears the outage streak.
                consecutive_failures = 0;
                let signal = detect_goal_signal(&outcome.response);
                match decide_after_goal_turn(signal, turn, spec.max_turns, consecutive_blocked) {
                    GoalStep::Stop(GoalStop::Completed) => {
                        send_message(
                            cx,
                            session_id,
                            &format!(
                                "\n✅ Goal achieved in {turn} turn(s): the agent reported the \
                                 objective verifiably complete.\n"
                            ),
                        );
                        break;
                    }
                    GoalStep::Stop(GoalStop::Blocked(reason)) => {
                        send_message(
                            cx,
                            session_id,
                            &format!(
                                "\n⛔ Goal blocked after {turn} turn(s) \
                                 ({GOAL_BLOCKED_THRESHOLD} consecutive blocked reports). \
                                 Stopping for user input.\nReason: {reason}\n"
                            ),
                        );
                        break;
                    }
                    GoalStep::Stop(GoalStop::CeilingReached) => {
                        send_message(
                            cx,
                            session_id,
                            &format!(
                                "\n🛑 Goal stopped: reached the opt-in {}-turn ceiling without a \
                                 completion signal. Review the progress above and re-run `/goal` \
                                 (raise or drop `--max-turns`) to keep going.\n",
                                spec.max_turns.unwrap_or(turn)
                            ),
                        );
                        break;
                    }
                    GoalStep::Continue {
                        consecutive_blocked: updated,
                    } => {
                        // A non-zero counter means this turn reported a blocker
                        // that has not yet reached the threshold.
                        if updated > 0 {
                            send_thought(
                                cx,
                                session_id,
                                &format!(
                                    "[goal: blocked report {updated}/{GOAL_BLOCKED_THRESHOLD}; \
                                     retrying]\n"
                                ),
                            );
                        }
                        consecutive_blocked = updated;
                    }
                }
            }
            Err(LoopIterationError::Terminal(err)) => {
                send_message(cx, session_id, &format!("\nGoal stopped: {err}\n"));
                break;
            }
        }

        if cancel.is_cancelled() {
            send_message(cx, session_id, "Goal cancelled.\n");
            break;
        }
    }

    cumulative
}

/// Build the `Vec<ChatMessage>` to send to the LLM for a fresh prompt.
///
/// Layout:
/// 1. System prompt (mode + cwd).
/// 2. AGENTS.md content, when present.
/// 3. Skills catalog, when the registry has entries.
/// 4. For each turn in `snap.history`:
///    - If the turn has a `summary`, emit a single `user` message
///      wrapping the summary in `<conversation_summary>` tags. The
///      original user prompt / tool exchanges / assistant text are
///      *not* re-emitted -- the summary replaces them in the prompt.
///    - Otherwise, replay the turn verbatim: user prompt, optional
///      `assistant_tool_calls` + `tool_result` pairs, optional
///      assistant text.
/// 5. The user's new prompt.
///
/// Pure -- exposed for unit testing the replay shape without spinning
/// up an LLM.
#[cfg(test)]
fn build_prompt_messages(snap: &SessionSnapshot, new_prompt: &str) -> Vec<ChatMessage> {
    build_prompt_messages_with_parts(snap, new_prompt, &[ChatContentPart::text(new_prompt)])
}

fn build_prompt_messages_with_parts(
    snap: &SessionSnapshot,
    new_prompt_text: &str,
    new_prompt_parts: &[ChatContentPart],
) -> Vec<ChatMessage> {
    let mut messages = Vec::with_capacity(snap.history.len() * 2 + 4);
    messages.push(ChatMessage::system(build_system_prompt(
        &snap.mode, &snap.cwd,
    )));
    if !snap.project_instructions.is_empty() {
        messages.push(ChatMessage::user(format!(
            "# AGENTS.md instructions for {}\n\n<INSTRUCTIONS>\n{}\n</INSTRUCTIONS>",
            snap.cwd.display(),
            snap.project_instructions
        )));
    }
    // Tier-1 disclosure: list each discovered skill's name+description
    // so the model can decide to auto-activate via `activate_skill`.
    // Skipped entirely when the registry is empty -- per the spec, an
    // empty `<available_skills/>` block would just confuse the model.
    if let Some(catalog) = build_skills_catalog(&snap.skills) {
        messages.push(ChatMessage::user(catalog));
    }
    for turn in &snap.history {
        // Per-turn summarization (mirrors Brokk's `TaskEntry.summary`):
        // when a summary is present, replace the entire turn (user
        // prompt + tool exchanges + assistant response) with one
        // `<conversation_summary>` block. The full log stays on disk
        // for replay determinism but never goes back to the LLM.
        if let Some(summary_text) = turn.summary.as_deref() {
            let trimmed = summary_text.trim();
            if !trimmed.is_empty() {
                messages.push(ChatMessage::user(format!(
                    "<conversation_summary>\n{trimmed}\n</conversation_summary>"
                )));
                continue;
            }
        }

        messages.push(ChatMessage::user(turn.user_prompt.clone()));

        // If the prior turn used tools, replay them as a single
        // assistant_tool_calls message followed by one tool_result per call
        // -- enough for the LLM to see the calls it made and what came back,
        // so it doesn't redo the same searches or writes (#3409).
        //
        // FIXME(#3409 follow-up): multi-round tool sequences within the same
        // turn (text₀ + calls₀ → results → text₁ + calls₁ → results → final)
        // collapse into a single `assistant_tool_calls` batch here, with all
        // intermediate text concatenated into `agent_response` and replayed
        // *after* the tool_results. For models that condition heavily on
        // order-of-reasoning this is a faithfulness loss compared to the
        // original turn. Acceptable today (the LLM still sees the calls and
        // their results) but worth revisiting if we observe model-quality
        // regressions on resumed multi-round turns. A faithful replay would
        // require persisting `Vec<Vec<ToolExchange>>` plus per-round
        // assistant text, doubling the on-disk schema cost.
        if !turn.tool_exchanges.is_empty() {
            let calls: Vec<crate::llm_client::ToolCall> = turn
                .tool_exchanges
                .iter()
                .map(|e| crate::llm_client::ToolCall {
                    id: e.call_id.clone(),
                    r#type: "function".to_string(),
                    function: crate::llm_client::FunctionCall {
                        name: e.tool_name.clone(),
                        arguments: e.arguments.clone(),
                    },
                })
                .collect();
            messages.push(ChatMessage::assistant_tool_calls(calls));
            for exchange in &turn.tool_exchanges {
                messages.push(ChatMessage::tool_result(
                    &exchange.call_id,
                    &exchange.tool_name,
                    &exchange.result,
                ));
            }
        }

        // Skip the trailing assistant message when the turn ended without
        // any final text (e.g. tool_loop exhausted max_turns, or the last
        // LLM call failed/was cancelled): `agent_response == ""`. Several
        // OpenAI-compatible providers (Mistral, some local-LLM proxies,
        // Anthropic's tool-use shape) reject an `assistant` message that
        // is both empty-content and non-tool_calls; even when accepted it
        // wastes a slot and may confuse the model on long replays. If the
        // turn used tools, the tool_results above already terminate it
        // coherently. If it didn't use tools and produced no text either,
        // there is nothing to replay -- emitting "" would be misleading.
        if !turn.agent_response.is_empty() {
            messages.push(ChatMessage::assistant(turn.agent_response.clone()));
        }
    }
    if new_prompt_parts.is_empty() {
        messages.push(ChatMessage::user(new_prompt_text.to_string()));
    } else {
        messages.push(ChatMessage::user_parts(new_prompt_parts.to_vec()));
    }
    messages
}

/// Wrap `build_prompt_messages` with per-turn LLM summarization.
///
/// When the projected prompt exceeds the model's budget, walk the
/// history from the oldest *uncompressed* turn forward, asking the
/// LLM to summarize each one in turn (Brokk's pattern from
/// `ContextManager.compressHistory(TaskEntry)`). Each successful
/// summary is persisted via [`SessionStore::set_turn_summary`] so a
/// reload reproduces the same compressed prompt, and the in-memory
/// `snap` is mutated so the rebuilt prompt sees the new state.
///
/// Stops when the prompt fits, when every turn already carries a
/// summary, or when a summarization call fails. Persistence failures
/// are logged but non-fatal -- the summary lives in memory for the
/// current turn and the next session reload will recompress.
///
/// Turns that fail to summarize stay uncompressed; we never silently
/// drop history. The prompt may still overrun budget after this runs
/// -- the LLM/server is the final arbiter -- but we've done what we
/// can without losing information.
#[allow(clippy::too_many_arguments)]
async fn build_prompt_messages_with_compression(
    snap: &mut SessionSnapshot,
    prompt_text: &str,
    prompt_parts: &[ChatContentPart],
    llm: &dyn crate::llm_client::LlmBackend,
    sessions: &SessionStore,
    session_id: &str,
    cancel: tokio_util::sync::CancellationToken,
    idle_timeout: Duration,
    context_length: Option<u32>,
) -> Vec<ChatMessage> {
    use crate::context_manager::{context_budget, summarize_turn};

    let budget = context_budget(context_length);
    let mut messages = build_prompt_messages_with_parts(snap, prompt_text, prompt_parts);

    loop {
        let projected = crate::tokens::approximate_tokens_messages(&messages);
        if projected <= budget {
            return messages;
        }
        // Find the oldest uncompressed turn -- compressing in order
        // mirrors how Brokk's `compressHistoryAsync(Context)` walks
        // entries, and keeps the most recent (most semantically
        // important) turns verbatim for the longest.
        let Some(idx) = snap.history.iter().position(|t| t.summary.is_none()) else {
            tracing::warn!(
                session_id = %session_id,
                projected_tokens = projected,
                budget,
                "prompt exceeds context budget but every turn is already summarized"
            );
            return messages;
        };
        let turn_to_summarize = snap.history[idx].clone();
        match summarize_turn(
            llm,
            &snap.model,
            &turn_to_summarize,
            context_length,
            idle_timeout,
            cancel.clone(),
        )
        .await
        {
            Ok(new_summary) => {
                if let Err(e) = sessions
                    .set_turn_summary(session_id, idx, new_summary.clone())
                    .await
                {
                    // Persistence failure isn't fatal -- we still want
                    // this turn's prompt to benefit from the summary.
                    // The next reload will see the uncompressed turn
                    // and try again.
                    tracing::warn!(
                        session_id = %session_id,
                        turn_index = idx,
                        "failed to persist turn summary, continuing with in-memory copy: {e:#}"
                    );
                }
                snap.history[idx].summary = Some(new_summary);
                messages = build_prompt_messages_with_parts(snap, prompt_text, prompt_parts);
            }
            Err(e) => {
                tracing::warn!(
                    session_id = %session_id,
                    turn_index = idx,
                    "summarization failed, leaving turn uncompressed: {e:#}"
                );
                // Brokk's `ContextManager.compressHistory` returns the
                // original on failure -- we mirror that by leaving the
                // turn uncompressed and giving up further attempts on
                // this prompt rather than retrying earlier turns and
                // racking up cost.
                return messages;
            }
        }
    }
}

/// Everything a single model turn produced, threaded back to the caller.
///
/// `response` is the assistant's final text (returned directly so callers no
/// longer have to re-read it from persisted history), and `failure` is set
/// only when the turn ended in an LLM error or panic rather than a real
/// completion. The normal-prompt and `/loop` callers use just
/// `structured_output` + `cumulative_usage`; `/goal` additionally inspects
/// `response` (for the sentinel) and `failure` (to back off or stop).
struct ModelTurnResult {
    structured_output: Option<StructuredOutputResult>,
    cumulative_usage: crate::llm_client::TokenUsage,
    response: String,
    failure: Option<crate::tool_loop::TurnFailure>,
}

#[allow(clippy::too_many_arguments)]
async fn run_model_turn_in_spawn(
    cx: &ConnectionTo<Client>,
    sessions: &SessionStore,
    session_id: &str,
    fallback_cwd: &Path,
    llm: &Arc<dyn crate::llm_client::LlmBackend>,
    registry: &Arc<crate::tools::ToolRegistry>,
    model: &str,
    reasoning_effort: Option<&str>,
    structured_output_request: Option<&StructuredOutputRequest>,
    messages: Vec<ChatMessage>,
    max_turns: usize,
    idle_timeout: Duration,
    cancel: tokio_util::sync::CancellationToken,
    prompt_text_for_turn: String,
) -> ModelTurnResult {
    use futures::FutureExt;
    use std::panic::AssertUnwindSafe;

    let cx_text = cx.clone();
    let sid_text = session_id.to_string();
    let cx_thought = cx.clone();
    let sid_thought = session_id.to_string();

    let text_sink: crate::tool_loop::TextSink =
        std::sync::Arc::new(std::sync::Mutex::new(move |token: &str| {
            send_message(&cx_text, &sid_text, token);
        }));
    let thought_sink: crate::tool_loop::TextSink =
        std::sync::Arc::new(std::sync::Mutex::new(move |token: &str| {
            send_thought(&cx_thought, &sid_thought, token);
        }));

    let cx_for_gate = cx.clone();
    let spawned_cx = crate::tool_loop::SpawnedCx::new(&cx_for_gate);
    let loop_result = AssertUnwindSafe(crate::tool_loop::run(
        llm,
        registry,
        model,
        reasoning_effort,
        structured_output_request,
        messages,
        max_turns,
        idle_timeout,
        cancel,
        text_sink,
        thought_sink,
        spawned_cx,
        session_id.to_string(),
        sessions.clone(),
        prompt_text_for_turn.clone(),
        crate::tool_loop::NotificationMode::Live,
        0,
    ))
    .catch_unwind()
    .await;

    let (response_text, tool_exchanges, turn_usage, failure) = match loop_result {
        Ok((text, exchanges, usage, failure)) => (text, exchanges, usage, failure),
        Err(panic) => {
            tracing::error!(session_id = %session_id, "tool loop panicked: {:?}", panic);
            // A panic is treated as fatal (non-retryable): retrying a
            // deterministic crash would just spin, so an autonomous driver
            // should stop and surface it rather than back off and retry.
            (
                "Error: agent loop panicked. See server logs.".to_string(),
                Vec::new(),
                crate::llm_client::TokenUsage::default(),
                Some(crate::tool_loop::TurnFailure {
                    retryable: false,
                    message: "agent loop panicked".to_string(),
                }),
            )
        }
    };

    let cost_delta_usd = sessions
        .available_model_metadata()
        .await
        .iter()
        .find(|meta| meta.id == model)
        .and_then(|meta| meta.estimate_cost_usd(turn_usage));
    let cumulative_usage = sessions
        .record_usage(session_id, turn_usage, cost_delta_usd)
        .await
        .unwrap_or(turn_usage);
    let structured_output_result =
        structured_output_request.map(|request| validate_response(request, &response_text));

    if let Err(e) = sessions
        .add_turn(
            session_id,
            ConversationTurn {
                user_prompt: prompt_text_for_turn,
                agent_response: response_text.clone(),
                tool_exchanges,
                structured_output: structured_output_result.clone(),
                summary: None,
                fragment_id: None,
            },
        )
        .await
    {
        send_message(
            cx,
            session_id,
            &format!(
                "\n**Warning:** failed to save this conversation turn to disk; \
                 it will not survive a session reload: {e}\n"
            ),
        );
    }

    send_session_usage_update(cx, sessions, session_id, fallback_cwd).await;
    ModelTurnResult {
        structured_output: structured_output_result,
        cumulative_usage,
        response: response_text,
        failure,
    }
}

/// Shared "run one model turn" pipeline behind both `/loop` and `/goal`:
/// validate the model, resolve the context window, compress history to fit,
/// snapshot the tool registry, then run the turn. Returns the threaded
/// [`ModelTurnResult`] (assistant text + usage + structured output + failure
/// classification). Callers keep only their own pre/post steps -- `/loop`'s
/// image-prompt rejection, `/goal`'s sentinel scan -- so the per-turn pipeline
/// lives in exactly one place. `snap` is taken by `&mut` because compression
/// rewrites its in-memory history to fit the context budget.
#[allow(clippy::too_many_arguments)]
async fn run_prepared_model_turn(
    cx: &ConnectionTo<Client>,
    sessions: &SessionStore,
    session_id: &str,
    fallback_cwd: &Path,
    llm: &Arc<dyn crate::llm_client::LlmBackend>,
    snap: &mut SessionSnapshot,
    prompt_text: &str,
    prompt_parts: &[ChatContentPart],
    structured_output_request: Option<&StructuredOutputRequest>,
    default_idle_timeout_secs: u64,
    max_turns: usize,
    cancel: tokio_util::sync::CancellationToken,
) -> Result<ModelTurnResult, LoopIterationError> {
    if snap.model.is_empty() {
        return Err(LoopIterationError::Terminal(
            "model not configured".to_string(),
        ));
    }

    let context_length = sessions
        .available_model_metadata()
        .await
        .iter()
        .find(|m| m.id == snap.model)
        .and_then(|m| m.context_length);
    // The compression and chat calls share one idle timeout (the previous
    // inline copies computed this same value twice).
    let idle_timeout = Duration::from_secs(
        snap.idle_timeout_secs
            .unwrap_or(default_idle_timeout_secs)
            .max(1),
    );
    let messages = build_prompt_messages_with_compression(
        snap,
        prompt_text,
        prompt_parts,
        llm.as_ref(),
        sessions,
        session_id,
        cancel.clone(),
        idle_timeout,
        context_length,
    )
    .await;
    let Some(registry) = sessions
        .get_or_create_registry(session_id, snap.cwd.clone())
        .await
    else {
        return Err(LoopIterationError::Terminal("unknown session".to_string()));
    };

    Ok(run_model_turn_in_spawn(
        cx,
        sessions,
        session_id,
        fallback_cwd,
        llm,
        &registry,
        &snap.model,
        snap.reasoning_effort.as_deref(),
        structured_output_request,
        messages,
        max_turns,
        idle_timeout,
        cancel,
        prompt_text.to_string(),
    )
    .await)
}

fn build_system_prompt(mode: &SessionMode, cwd: &Path) -> String {
    let cwd_context = format!(
        "The user's working directory is: {}\n\
         All file paths should be interpreted relative to this directory.\n\n",
        cwd.display()
    );

    // The identity line is intentionally general-purpose: Anvil is often
    // driven by hosts (e.g. `mj`) that mix coding and non-coding prompts,
    // and "AI coding assistant" wording was enough for some models to
    // refuse off-topic questions. We still name software engineering as
    // the specialty so coding behavior is unchanged.
    let mode_prompt = match mode {
        SessionMode::Lutz => {
            "You are Brokk, an AI assistant running in a terminal environment. You specialize in \
             software engineering — code analysis, generation, refactoring, debugging, and \
             architecture — but you can help with any task the user brings to you. You work \
             using an agentic approach: break complex tasks into steps, execute them, and report \
             results. When appropriate, create a task list to track progress."
        }
        SessionMode::Code => {
            "You are Brokk, an AI assistant running in a terminal environment. You specialize in \
             software engineering — code analysis, generation, refactoring, debugging, and \
             architecture — but you can help with any task the user brings to you. In this mode, \
             focus on code changes: generate modifications, refactors, and implementations. Be \
             concise and focus on the code."
        }
        SessionMode::Ask => {
            "You are Brokk, an AI assistant running in a terminal environment. You specialize in \
             software engineering — code analysis, generation, refactoring, debugging, and \
             architecture — but you can help with any task the user brings to you. Answer \
             questions about code, architecture, and software engineering concepts thoroughly \
             but concisely."
        }
        SessionMode::Plan => {
            "You are Brokk, an AI assistant running in a terminal environment. You specialize in \
             software engineering — code analysis, generation, refactoring, debugging, and \
             architecture — but you can help with any task the user brings to you. In this mode, \
             focus on planning: analyze requirements, design solutions, and create implementation \
             plans. Do not write code directly."
        }
    };

    format!("{cwd_context}{mode_prompt}")
}

/// Returns true when `prompt_text` invokes the slash command `name`,
/// matching `/name` exactly or `/name <args>`. Whitespace and case are
/// normalized so clients that uppercase auto-complete entries still hit.
fn is_slash_command(prompt_text: &str, name: &str) -> bool {
    let stripped = prompt_text.trim();
    let Some(rest) = stripped.strip_prefix('/') else {
        return false;
    };
    let head = rest
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    head == name
}

/// Parse `/<name> <args...>` out of a prompt. Returns `None` when the
/// prompt isn't a slash command. The `name` is lowercased for
/// case-insensitive lookup; the `args` slice preserves the original
/// casing/whitespace after the command head.
fn parse_slash_command(prompt_text: &str) -> Option<(String, String)> {
    let stripped = prompt_text.trim();
    let rest = stripped.strip_prefix('/')?;
    if rest.is_empty() {
        return None;
    }
    let (head, tail) = match rest.find(char::is_whitespace) {
        Some(i) => (&rest[..i], rest[i..].trim_start()),
        None => (rest, ""),
    };
    if head.is_empty() {
        return None;
    }
    Some((head.to_ascii_lowercase(), tail.to_string()))
}

/// Only plain prompts should auto-title the session. Any slash command,
/// including skill activations, is an operational turn rather than a
/// good title seed and should leave the placeholder title alone.
fn should_auto_rename_session_from_prompt(prompt_text: &str) -> bool {
    parse_slash_command(prompt_text).is_none()
}

/// Build the `<available_skills>` tier-1 disclosure block for the system
/// prompt. Returns `None` when the registry is empty so the caller can
/// skip the injection entirely (per the spec's "When no skills are
/// available" guidance: never emit an empty block).
fn build_skills_catalog(registry: &crate::skills::SkillRegistry) -> Option<String> {
    if registry.is_empty() {
        return None;
    }
    let mut out = String::from("<available_skills>\n");
    for meta in registry.iter_sorted() {
        out.push_str("  <skill>\n");
        out.push_str(&format!("    <name>{}</name>\n", xml_escape(&meta.name)));
        out.push_str(&format!(
            "    <description>{}</description>\n",
            xml_escape(&meta.description)
        ));
        out.push_str(&format!(
            "    <location>{}</location>\n",
            xml_escape(&meta.location.display().to_string())
        ));
        out.push_str("  </skill>\n");
    }
    out.push_str("</available_skills>\n\n");
    out.push_str(
        "The skills above provide specialized instructions for specific tasks. \
        When a task matches a skill's description, call the `activate_skill` tool \
        with the skill's name to load its full instructions. Users can also invoke \
        a skill directly by typing `/<skill-name>` as a slash command.",
    );
    Some(out)
}

/// Build the structured-wrapping payload sent to the LLM when a skill is
/// activated (whether via slash command or the `activate_skill` tool).
/// Format follows the spec's recommended "Structured wrapping" example:
/// the skill body inside `<skill_content name="...">` tags, with the
/// skill directory and a `<skill_resources>` listing so the model can
/// pull bundled scripts/references with its existing file-read tool.
pub(crate) fn build_skill_payload(meta: &crate::skills::SkillMeta) -> String {
    let body = match crate::skills::read_skill_body(&meta.location) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(
                path = %meta.location.display(),
                "SKILL.md became unreadable between discovery and activation: {e}"
            );
            return format!(
                "<skill_content name=\"{}\">\n[skill file {} could not be read: {e}]\n</skill_content>",
                xml_escape(&meta.name),
                meta.location.display()
            );
        }
    };
    let resources = crate::skills::list_bundled_resources(&meta.skill_dir);
    let mut out = format!("<skill_content name=\"{}\">\n", xml_escape(&meta.name));
    out.push_str(&body);
    if !body.ends_with('\n') {
        out.push('\n');
    }
    out.push('\n');
    out.push_str(&format!("Skill directory: {}\n", meta.skill_dir.display()));
    out.push_str("Relative paths inside this skill resolve against the skill directory.\n");
    if !resources.is_empty() {
        out.push_str("\n<skill_resources>\n");
        for rel in &resources {
            out.push_str(&format!("  <file>{}</file>\n", xml_escape(rel)));
        }
        out.push_str("</skill_resources>\n");
    }
    out.push_str("</skill_content>");
    out
}

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

/// Handle the `/codex-login` slash command and its subcommands.
/// Subcommands: bare = start interactive login, `status` = report what's
/// stored, `disconnect` = wipe the local credentials.
///
/// On a successful bare login we install the freshly-built Codex
/// backend into `MultiBackend` so the next `session/new` (and any
/// subsequent `codex::*` route) picks it up without a server restart.
/// Without this, the empty-at-startup `Option` captured at
/// construction would remain `None` forever and the new credentials
/// would be unreachable until restart -- the behaviour issue #3555
/// reported.
async fn handle_codex_login(
    prompt_text: &str,
    llm: &Arc<MultiBackend>,
    sessions: &SessionStore,
    refresh_lock: &Arc<tokio::sync::Mutex<()>>,
    cx: Option<&ConnectionTo<Client>>,
    session_id: Option<&str>,
) -> String {
    let arg = prompt_text
        .trim()
        .strip_prefix('/')
        .unwrap_or("")
        .split_whitespace()
        .nth(1)
        .unwrap_or("")
        .to_ascii_lowercase();

    match arg.as_str() {
        "status" => match crate::codex_auth::read_auth_dot_json() {
            Ok(Some(auth)) => {
                let mode = auth.auth_mode.as_deref().unwrap_or("(unset)");
                let has_key = auth.openai_api_key.is_some();
                let acct = auth
                    .tokens
                    .as_ref()
                    .map(|t| t.account_id.as_str())
                    .unwrap_or("(none)");
                let last = auth
                    .last_refresh
                    .map(|ts| ts.to_rfc3339())
                    .unwrap_or_else(|| "(unknown)".to_string());
                let routing = match mode {
                    "chatgpt" => "ChatGPT subscription (Responses API on chatgpt.com)",
                    "apikey" => "OPENAI_API_KEY (api.openai.com, billed as API usage)",
                    _ => "unknown",
                };
                // ChatGPT-only accounts don't get an OPENAI_API_KEY
                // because they have no API organization to mint one
                // against. Surface that explicitly so users don't read
                // "MISSING" as a broken login.
                let api_key_label = match (mode, has_key) {
                    (_, true) => "present",
                    ("chatgpt", false) => {
                        "n/a (ChatGPT-only account; subscription routing does not need one)"
                    }
                    (_, false) => "MISSING",
                };
                format!(
                    "Codex login status:\n  auth_mode: {mode}\n  routing: {routing}\n  api_key: {api_key_label}\n  account_id: {acct}\n  last_refresh: {last}"
                )
            }
            Ok(None) => {
                "No Codex credentials found. Run `/setup codex` to authenticate.".to_string()
            }
            Err(e) => format!("Failed to read ~/.codex/auth.json: {e:#}"),
        },
        "disconnect" => match crate::codex_auth::logout() {
            Ok(()) => {
                // Drop the in-memory backend so subsequent `codex::*`
                // routes fail loudly (and identically to a no-auth
                // startup) instead of firing requests against now-missing
                // credentials. Refresh the cached catalog so the picker
                // stops offering Codex models.
                llm.uninstall_codex();
                spawn_background_refresh(
                    refresh_lock.clone(),
                    llm.clone(),
                    sessions.clone(),
                    cx.zip(session_id).map(|(cx, session_id)| {
                        (
                            cx.clone(),
                            session_id.to_string(),
                            "Refreshing model catalog after Codex disconnect...",
                        )
                    }),
                    None,
                );
                "Codex credentials cleared and the in-memory backend was unloaded; \
                 the picker will only show local models until you re-run `/setup codex`."
                    .to_string()
            }
            Err(e) => format!("Failed to remove ~/.codex/auth.json: {e:#}"),
        },
        "" => match crate::codex_auth::interactive_login().await {
            Ok(auth) => {
                let acct = auth
                    .tokens
                    .as_ref()
                    .map(|t| t.account_id.as_str())
                    .unwrap_or("(unknown)");
                // Install the new backend so this session (and any
                // future ones) can route `codex::*` and bare model ids
                // immediately. We only install when the auth payload
                // resolves to a usable backend -- a malformed auth.json
                // (e.g. apikey mode with no key) leaves the slot empty
                // and the user-facing message stays honest about it.
                match crate::codex_backend_from_auth(&auth) {
                    Some(backend) => {
                        llm.install_codex(backend);
                        // Refresh the cached model catalog in the
                        // background so the picker picks Codex up on
                        // the next `session/new` without waiting for
                        // an unrelated discovery trigger. Shares the
                        // same throttle as `session/new` so an
                        // immediate session creation right after login
                        // doesn't race a second probe.
                        spawn_background_refresh(
                            refresh_lock.clone(),
                            llm.clone(),
                            sessions.clone(),
                            cx.zip(session_id).map(|(cx, session_id)| {
                                (
                                    cx.clone(),
                                    session_id.to_string(),
                                    "Refreshing model catalog after Codex login...",
                                )
                            }),
                            None,
                        );
                        format!(
                            "Codex login complete (account_id: {acct}). \
                             Codex is now active -- create a new session \
                             (or wait for the next discovery refresh) and \
                             pick a `codex::*` model from the picker; \
                             prompts route through your ChatGPT subscription \
                             via https://chatgpt.com/backend-api/codex/responses."
                        )
                    }
                    None => format!(
                        "Codex login completed but the saved credentials are not usable \
                         (auth_mode={:?}, no OPENAI_API_KEY). Re-run `/setup codex` or \
                         inspect ~/.codex/auth.json.",
                        auth.auth_mode
                    ),
                }
            }
            Err(e) => format!("Codex login failed: {e:#}"),
        },
        other => format!(
            "Unknown subcommand `{other}`. Try: /setup codex | /setup codex status | /setup codex disconnect"
        ),
    }
}

/// User-facing explanation returned when `OPENROUTER_API_KEY` is set
/// in the process environment. Single source of truth for the message
/// so the setup handler, future status surfaces, and tests stay in
/// agreement on the wording.
fn openrouter_env_owned_explanation() -> String {
    let state = crate::openrouter_auth::CredentialState::snapshot();
    format!(
        "OpenRouter credentials are owned by the OPENROUTER_API_KEY environment \
         variable. Anvil reads that value at startup; unset it and restart the \
         server if you want `/setup openrouter key <key>` to manage credentials.\n\n\
         Credential state:\n\
         - active_source: `{}`\n\
         - env_set: `{}`\n\
         - file_present: `{}`",
        state.active_source(),
        state.env_set,
        state.file_present
    )
}

/// Handle the `/openrouter-login` slash command and its subcommands.
/// Subcommands: bare = help text (no OAuth flow), `<key>` = save key and
/// install backend, `status` = report what's stored and where it came
/// from, `disconnect` = wipe the local credentials.
///
/// Unlike Codex, OpenRouter has no browser flow -- the user pastes a
/// static `sk-or-...` key inline. That key lands in the session
/// transcript, so the help text and the success message both warn the
/// user to rotate the key if the transcript is shared.
///
/// **Credential-ownership contract**: when `OPENROUTER_API_KEY` is set
/// in the process environment, the env owns the credential lifecycle
/// and this handler short-circuits with an explanation for every
/// subcommand. The slash is hidden from autocomplete in that mode too
/// (see `builtin_commands`), but the handler still runs when typed
/// manually so users can't get "command not found" with no hint.
/// Diagnostic state (env_set, file_present, active_source) stays
/// available via `/setup openrouter status` regardless of which mode is
/// active.
async fn handle_openrouter_login(
    prompt_text: &str,
    llm: &Arc<MultiBackend>,
    sessions: &SessionStore,
    refresh_lock: &Arc<tokio::sync::Mutex<()>>,
    cx: Option<&ConnectionTo<Client>>,
    session_id: Option<&str>,
) -> String {
    if crate::openrouter_auth::CredentialState::snapshot().env_owns() {
        return openrouter_env_owned_explanation();
    }
    // Take the entire argument tail (everything after the command), not
    // just the first whitespace-delimited token: OpenRouter keys are
    // ASCII with no spaces in practice, but we trim defensively so a
    // user who pasted with trailing spaces doesn't see a "key was empty"
    // bounce. `status` and `disconnect` are case-insensitive to match
    // the `/codex-login` ergonomics.
    let after_cmd = prompt_text
        .trim()
        .strip_prefix('/')
        .unwrap_or("")
        .split_once(char::is_whitespace)
        .map(|(_, tail)| tail)
        .unwrap_or("")
        .trim();

    let lowered = after_cmd.to_ascii_lowercase();
    match lowered.as_str() {
        "" => format!(
            "Usage: `/setup openrouter key <key>` | `/setup openrouter status` | \
             `/setup openrouter disconnect`. Get a key at \
             https://openrouter.ai/keys. Note: the key appears in this session's \
             transcript, so rotate it at openrouter.ai if you share the log. \
             Credentials are persisted to {}.",
            crate::openrouter_auth::auth_path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "the OS config directory".to_string())
        ),
        "status" => {
            // env_owns short-circuits the whole handler at the top, so
            // we only reach this arm when the env var is unset. The
            // snapshot's env_set is therefore always false here -- we
            // include it in the output anyway for self-contained
            // diagnostics so users can confirm the env is clear from
            // `/setup openrouter status`.
            let state = crate::openrouter_auth::CredentialState::snapshot();
            let file_key = match crate::openrouter_auth::read() {
                Ok(Some(auth)) => Some(auth.api_key.trim().to_string()).filter(|s| !s.is_empty()),
                Ok(None) => None,
                Err(e) => {
                    return format!("Failed to read OpenRouter credential file: {e:#}");
                }
            };
            let active_len = file_key
                .as_deref()
                .map(str::len)
                .map(|n| n.to_string())
                .unwrap_or_else(|| "n/a".to_string());
            let path = crate::openrouter_auth::auth_path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "<unresolved>".to_string());
            format!(
                "OpenRouter login status:\n  active_source: {}\n  \
                 active_key_length: {active_len}\n  base_url: {}\n  \
                 credential_file: {path}\n  file_present: {}\n  env_set: {}",
                state.active_source(),
                crate::discovery::OPENROUTER_BASE_URL,
                state.file_present,
                state.env_set,
            )
        }
        "disconnect" => match crate::openrouter_auth::logout() {
            Ok(()) => {
                llm.uninstall_openrouter();
                spawn_background_refresh(
                    refresh_lock.clone(),
                    llm.clone(),
                    sessions.clone(),
                    cx.zip(session_id).map(|(cx, session_id)| {
                        (
                            cx.clone(),
                            session_id.to_string(),
                            "Refreshing model catalog after OpenRouter disconnect...",
                        )
                    }),
                    None,
                );
                "OpenRouter credentials cleared and the in-memory backend was unloaded; \
                 the picker will only show models from other configured backends until \
                 you re-run `/setup openrouter key <key>`."
                    .to_string()
            }
            Err(e) => format!("Failed to remove OpenRouter credential file: {e:#}"),
        },
        _ => {
            // Anything else is treated as a candidate API key. Reject
            // obvious junk (whitespace-only after trim is handled above;
            // empty is the "" arm); accept everything else and let the
            // first request 401 if the key is malformed. We don't gate
            // on the `sk-or-` prefix because OpenRouter has historically
            // issued keys with other shapes and we'd rather not hardcode
            // a check that ages out.
            let key = after_cmd.to_string();
            match crate::openrouter_auth::write(&crate::openrouter_auth::OpenRouterAuth {
                api_key: key.clone(),
            }) {
                Ok(()) => match crate::openrouter_backend_from_key(&key) {
                    Some(backend) => {
                        llm.install_openrouter(backend);
                        spawn_background_refresh(
                            refresh_lock.clone(),
                            llm.clone(),
                            sessions.clone(),
                            cx.zip(session_id).map(|(cx, session_id)| {
                                (
                                    cx.clone(),
                                    session_id.to_string(),
                                    "Refreshing model catalog after OpenRouter login...",
                                )
                            }),
                            None,
                        );
                        let path = crate::openrouter_auth::auth_path()
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|_| "<unresolved>".to_string());
                        format!(
                            "OpenRouter login complete (key length: {}). \
                             Credentials saved to {} (chmod 0600). The picker will \
                             show `openrouter::*` models after the next discovery \
                             refresh; create a new session or wait briefly. \
                             Reminder: the key was sent inline and is recorded in \
                             this session's transcript -- rotate it at \
                             https://openrouter.ai/keys if the transcript is shared.",
                            key.len(),
                            path,
                        )
                    }
                    None => {
                        // Defensive: write() rejects empty input upstream
                        // via the "" arm, so reaching None here means the
                        // key became empty after trim somewhere -- still
                        // surface a clear error rather than installing a
                        // broken backend.
                        let _ = crate::openrouter_auth::logout();
                        "OpenRouter login failed: provided key was empty after trimming".to_string()
                    }
                },
                Err(e) => format!("OpenRouter login failed: could not save key: {e:#}"),
            }
        }
    }
}

/// Pure parser for `/idle-timeout` arguments. Returns either a successful
/// action to apply, or a user-facing error string. Factored out from
/// `handle_idle_timeout` so it can be unit-tested without standing up a
/// real `SessionStore`. Bounds are shared with the `--llm-idle-timeout-secs`
/// CLI flag (see `llm_client::{MIN,MAX}_IDLE_CHUNK_TIMEOUT_SECS`).
#[derive(Debug, PartialEq, Eq)]
enum IdleTimeoutAction {
    /// `/idle-timeout` -- caller should render the current value.
    Show,
    /// `/idle-timeout default` -- clear the session override.
    Clear,
    /// `/idle-timeout <secs>` with a valid value.
    Set(u64),
}

fn parse_idle_timeout_arg(prompt_text: &str) -> Result<IdleTimeoutAction, String> {
    let arg = prompt_text
        .trim()
        .strip_prefix('/')
        .unwrap_or("")
        .split_whitespace()
        .nth(1)
        .unwrap_or("")
        .to_ascii_lowercase();

    let min = crate::llm_client::MIN_IDLE_CHUNK_TIMEOUT_SECS;
    let max = crate::llm_client::MAX_IDLE_CHUNK_TIMEOUT_SECS;

    match arg.as_str() {
        "" => Ok(IdleTimeoutAction::Show),
        "default" => Ok(IdleTimeoutAction::Clear),
        other => match other.parse::<u64>() {
            Ok(secs) if (min..=max).contains(&secs) => Ok(IdleTimeoutAction::Set(secs)),
            Ok(out_of_range) => Err(format!(
                "Value `{out_of_range}` is out of range. Pick a value between \
                 {min}s and {max}s, or use `default` to clear the override."
            )),
            Err(_) => Err(format!(
                "Unknown subcommand `{other}`. Try: /setup timeout | \
                 /setup timeout <seconds> | /setup timeout default"
            )),
        },
    }
}

fn parse_shell_words(input: &str) -> Result<Vec<String>, String> {
    #[derive(Copy, Clone, PartialEq, Eq)]
    enum Quote {
        None,
        Single,
        Double,
    }

    let mut words = Vec::new();
    let mut current = String::new();
    let mut quote = Quote::None;
    let mut current_started = false;
    let mut chars = input.chars();

    while let Some(ch) = chars.next() {
        match quote {
            Quote::None => match ch {
                c if c.is_whitespace() => {
                    if current_started {
                        words.push(std::mem::take(&mut current));
                        current_started = false;
                    }
                }
                '\'' => {
                    quote = Quote::Single;
                    current_started = true;
                }
                '"' => {
                    quote = Quote::Double;
                    current_started = true;
                }
                '\\' => {
                    let Some(next) = chars.next() else {
                        return Err("Trailing backslash in MCP command.".to_string());
                    };
                    current.push(next);
                    current_started = true;
                }
                _ => {
                    current.push(ch);
                    current_started = true;
                }
            },
            Quote::Single => {
                if ch == '\'' {
                    quote = Quote::None;
                } else {
                    current.push(ch);
                }
            }
            Quote::Double => match ch {
                '"' => quote = Quote::None,
                '\\' => {
                    let Some(next) = chars.next() else {
                        return Err("Trailing backslash in MCP command.".to_string());
                    };
                    if matches!(next, '"' | '\\' | '$' | '`' | '\n') {
                        current.push(next);
                    } else {
                        current.push('\\');
                        current.push(next);
                    }
                }
                _ => current.push(ch),
            },
        }
    }

    match quote {
        Quote::Single => return Err("Unclosed single quote in MCP command.".to_string()),
        Quote::Double => return Err("Unclosed double quote in MCP command.".to_string()),
        Quote::None => {}
    }
    if current_started {
        words.push(current);
    }
    Ok(words)
}

async fn handle_mcp(prompt_text: &str, sessions: &SessionStore, session_id: &str) -> String {
    let trimmed = slash_command_args(prompt_text);
    if trimmed.is_empty() {
        return render_mcp_servers();
    }

    let words = match parse_shell_words(&trimmed) {
        Ok(words) => words,
        Err(e) => return format!("Error: {e}"),
    };
    let command = words
        .first()
        .map(|word| word.to_ascii_lowercase())
        .unwrap_or_default();
    if command == "list" {
        return render_mcp_servers();
    }
    let result = match command.as_str() {
        "add" | "set" => {
            let mut framing = crate::mcp::McpFraming::ContentLength;
            let mut idx = 1;
            if words.get(idx).is_some_and(|word| word == "--framing") {
                let Some(raw_framing) = words.get(idx + 1) else {
                    return mcp_usage();
                };
                let Some(parsed) = crate::mcp::McpFraming::parse(raw_framing) else {
                    return "Unknown MCP framing. Use `content-length` or `line`.".to_string();
                };
                framing = parsed;
                idx += 2;
            }
            if words.len() < idx + 2 {
                return mcp_usage();
            }
            let name = &words[idx];
            let server_command = &words[idx + 1];
            if !valid_mcp_name(name) {
                return "MCP server names may contain only letters, numbers, `_`, `-`, and `.`."
                    .to_string();
            }
            let mut servers = crate::setup_state::read_mcp_servers();
            let server = crate::mcp::McpServerConfig {
                name: name.to_string(),
                command: server_command.to_string(),
                args: words[idx + 2..].to_vec(),
                env: Vec::new(),
                framing,
                enabled: true,
            };
            if let Some(existing) = servers.iter_mut().find(|s| s.name == *name) {
                *existing = server;
            } else {
                servers.push(server);
            }
            crate::setup_state::remember_mcp_servers(servers)
                .map(|_| format!("MCP server `{name}` saved and enabled."))
        }
        "remove" | "delete" | "rm" => {
            let Some(name) = words.get(1) else {
                return mcp_usage();
            };
            let mut servers = crate::setup_state::read_mcp_servers();
            let before = servers.len();
            servers.retain(|s| s.name != *name);
            if servers.len() == before {
                return format!("No MCP server named `{name}` is configured.");
            }
            crate::setup_state::remember_mcp_servers(servers)
                .map(|_| format!("MCP server `{name}` removed."))
        }
        "enable" | "disable" => {
            let Some(name) = words.get(1) else {
                return mcp_usage();
            };
            let enabled = command == "enable";
            let mut servers = crate::setup_state::read_mcp_servers();
            let Some(server) = servers.iter_mut().find(|s| s.name == *name) else {
                return format!("No MCP server named `{name}` is configured.");
            };
            server.enabled = enabled;
            crate::setup_state::remember_mcp_servers(servers).map(|_| {
                format!(
                    "MCP server `{name}` {}.",
                    if enabled { "enabled" } else { "disabled" }
                )
            })
        }
        "reset" => crate::setup_state::remember_mcp_servers(crate::mcp::default_servers())
            .map(|_| "MCP servers reset to Anvil defaults.".to_string()),
        "help" => return mcp_usage(),
        _ => return format!("Unknown MCP command `{command}`.\n\n{}", mcp_usage()),
    };

    match result {
        Ok(message) => {
            sessions.invalidate_registry(session_id).await;
            format!("{message}\n\nChanges take effect on the next tool-capable prompt.")
        }
        Err(e) => format!("Error: failed to save MCP configuration: {e:#}"),
    }
}

fn valid_mcp_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

fn render_mcp_servers() -> String {
    let servers = crate::setup_state::read_mcp_servers();
    let mut out = String::from("MCP servers\n\n");
    if servers.is_empty() {
        out.push_str("No MCP servers are configured.\n\n");
    } else {
        for server in servers {
            let status = if server.enabled {
                "enabled"
            } else {
                "disabled"
            };
            let args = if server.args.is_empty() {
                String::new()
            } else {
                format!(
                    " {}",
                    server
                        .args
                        .iter()
                        .map(|arg| shell_quote(arg))
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            };
            out.push_str(&format!(
                "- `{}` ({status}, {}): `{}{args}`\n",
                server.name,
                server.framing.as_str(),
                shell_quote(&server.command)
            ));
        }
        out.push('\n');
    }
    out.push_str(&mcp_usage());
    out
}

fn mcp_usage() -> String {
    "Commands:\n\
     - `/mcp list`\n\
     - `/mcp add [--framing content-length|line] <name> <command> [args...]`\n\
     - `/mcp enable <name>`\n\
     - `/mcp disable <name>`\n\
     - `/mcp remove <name>`\n\
     - `/mcp reset`\n\n\
     `content-length` is the standard MCP stdio framing and is the default for new \
     servers. Use `line` only for NDJSON-speaking servers. Use shell-style quoting \
     for commands or args that contain spaces, and use `{cwd}` in args to pass the \
     current workspace root. Bifrost is preinstalled as Anvil's managed local \
     binary with the equivalent args `--root '{cwd}' --server searchtools --no-line-numbers`."
        .to_string()
}

fn shell_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".to_string();
    }
    if value.chars().all(|c| {
        c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | ':' | '=' | '{' | '}')
    }) {
        return value.to_string();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// Handle the `/idle-timeout` slash command. Reads/sets the per-session
/// LLM SSE idle timeout (in seconds). The session override is in-memory
/// only -- it does not survive a session reload or a server restart.
///
/// Subcommands:
///   `/idle-timeout`           -> report the active value and where it came from
///   `/idle-timeout <secs>`    -> set the session override (1..=86_400)
///   `/idle-timeout default`   -> clear the session override
async fn handle_idle_timeout(
    prompt_text: &str,
    session_id: &str,
    sessions: &SessionStore,
    current_session_override: Option<u64>,
    default_secs: u64,
) -> String {
    let action = match parse_idle_timeout_arg(prompt_text) {
        Ok(action) => action,
        Err(msg) => return msg,
    };
    match action {
        IdleTimeoutAction::Show => match current_session_override {
            Some(secs) => format!(
                "LLM idle timeout: {secs}s (session override).\n\
                 Server default is {default_secs}s. Use `/setup timeout default` to clear, \
                 or `/setup timeout <seconds>` to change."
            ),
            None => format!(
                "LLM idle timeout: {default_secs}s (server default).\n\
                 Use `/setup timeout <seconds>` to override for this session only, \
                 or restart with `--llm-idle-timeout-secs` / `BROKK_ACP_LLM_IDLE_TIMEOUT_SECS` \
                 to change the default."
            ),
        },
        IdleTimeoutAction::Clear => {
            if sessions.set_idle_timeout_secs(session_id, None).await {
                format!(
                    "Cleared session override. LLM idle timeout is back to the server \
                     default ({default_secs}s)."
                )
            } else {
                "Error: unknown session.".to_string()
            }
        }
        IdleTimeoutAction::Set(secs) => {
            if sessions.set_idle_timeout_secs(session_id, Some(secs)).await {
                format!(
                    "LLM idle timeout set to {secs}s for this session. \
                     In-memory only -- reload or restart resets to the server \
                     default ({default_secs}s)."
                )
            } else {
                "Error: unknown session.".to_string()
            }
        }
    }
}

/// Infrastructure shared by the `/setup` command family.
struct SetupContext<'a> {
    cx: &'a ConnectionTo<Client>,
    sessions: &'a SessionStore,
    llm: &'a Arc<MultiBackend>,
    login_sessions: &'a SessionStore,
    refresh_lock: &'a Arc<tokio::sync::Mutex<()>>,
    default_idle_timeout_secs: u64,
    current_session_idle_timeout: Option<u64>,
}

/// Handle `/setup`, the model/provider and advanced configuration surface.
/// The command is intentionally task-oriented: it offers "choose for me",
/// Codex sign-in, local models, OpenRouter, sandbox/behavior settings, and an
/// advanced page. Permission mode lives in the ACP session config selector.
/// Internal config ids stay hidden unless the user explicitly enters `advanced`.
async fn handle_setup(ctx: &SetupContext<'_>, prompt_text: &str, session_id: &str) -> String {
    let trimmed = slash_command_args(prompt_text);
    if trimmed.is_empty() {
        return render_current_setup(ctx.sessions, session_id).await;
    }

    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let command = parts.next().unwrap_or("").to_ascii_lowercase();
    let rest = parts.next().unwrap_or("").trim();

    match command.as_str() {
        "choose" | "choose-for-me" | "chooseforme" => {
            handle_setup_choose(ctx.cx, ctx.sessions, session_id, ctx.llm, ctx.refresh_lock).await
        }
        "refresh" | "try-again" => {
            match refresh_model_catalog_now(
                Some(ctx.cx),
                Some(session_id),
                ctx.llm,
                ctx.sessions,
                ctx.refresh_lock,
            )
            .await
            {
                Ok(_) => render_current_setup(ctx.sessions, session_id).await,
                Err(e) => format!(
                    "Setup could not refresh models yet: {e}\n\n{}",
                    render_current_setup(ctx.sessions, session_id).await
                ),
            }
        }
        "codex" => {
            let codex_prompt = if rest.is_empty() || rest == "login" {
                "/codex-login".to_string()
            } else {
                format!("/codex-login {rest}")
            };
            let mut out = handle_codex_login(
                &codex_prompt,
                ctx.llm,
                ctx.login_sessions,
                ctx.refresh_lock,
                Some(ctx.cx),
                Some(session_id),
            )
            .await;
            out.push_str("\n\nRun `/setup choose` after sign-in completes.");
            out
        }
        "local" | "ollama" => {
            handle_setup_local(
                ctx.cx,
                ctx.sessions,
                session_id,
                ctx.llm,
                ctx.refresh_lock,
                rest,
            )
            .await
        }
        "bedrock" => {
            handle_setup_bedrock(
                ctx.cx,
                ctx.sessions,
                session_id,
                ctx.llm,
                ctx.refresh_lock,
                rest,
            )
            .await
        }
        "openrouter" => {
            handle_setup_openrouter(
                ctx.cx,
                session_id,
                rest,
                ctx.llm,
                ctx.login_sessions,
                ctx.refresh_lock,
            )
            .await
        }
        "sandbox" => handle_setup_sandbox(ctx.sessions, session_id, rest).await,
        "mode" | "behavior" => handle_setup_mode(ctx.cx, ctx.sessions, session_id, rest).await,
        "timeout" => {
            let prompt = if rest.is_empty() {
                "/idle-timeout".to_string()
            } else {
                format!("/idle-timeout {rest}")
            };
            handle_idle_timeout(
                &prompt,
                session_id,
                ctx.sessions,
                ctx.current_session_idle_timeout,
                ctx.default_idle_timeout_secs,
            )
            .await
        }
        "model" => {
            if rest.is_empty() {
                render_setup_models(ctx.sessions.available_model_metadata().await.as_slice())
            } else {
                apply_setup_config(ctx.cx, ctx.sessions, session_id, MODEL_CONFIG_ID, rest).await
            }
        }
        "reasoning" | "reasoning-effort" => {
            if rest.is_empty() {
                "Use `/setup reasoning default`, `/setup reasoning off`, or `/setup reasoning <level>`.\n\
                 This is an advanced setting; most users should leave it alone."
                    .to_string()
            } else {
                let value = if rest.eq_ignore_ascii_case("default") {
                    REASONING_EFFORT_DEFAULT_VALUE
                } else if rest.eq_ignore_ascii_case(REASONING_EFFORT_OFF_VALUE) {
                    REASONING_EFFORT_OFF_VALUE
                } else {
                    rest
                };
                apply_setup_config(
                    ctx.cx,
                    ctx.sessions,
                    session_id,
                    REASONING_EFFORT_CONFIG_ID,
                    value,
                )
                .await
            }
        }
        "advanced" => render_setup_advanced(ctx.sessions, session_id).await,
        other => format!(
            "Unknown setup option `{other}`.\n\n{}",
            render_current_setup(ctx.sessions, session_id).await
        ),
    }
}

fn is_streamed_setup_openrouter_refresh(prompt_text: &str) -> bool {
    if !is_slash_command(prompt_text, "setup") {
        return false;
    }
    let trimmed = slash_command_args(prompt_text);
    let (action, rest) = split_setup_action(&trimmed);
    action.eq_ignore_ascii_case("openrouter")
        && matches!(rest.to_ascii_lowercase().as_str(), "refresh" | "try-again")
}

async fn render_current_setup(sessions: &SessionStore, session_id: &str) -> String {
    let fallback_cwd = std::env::current_dir().unwrap_or_default();
    let Some(session) = sessions.get_session(session_id, &fallback_cwd).await else {
        return "Error: unknown session.".to_string();
    };
    let catalog = sessions.available_model_metadata().await;
    render_setup_home(&session, &catalog)
}

async fn refresh_model_catalog_now(
    cx: Option<&ConnectionTo<Client>>,
    session_id: Option<&str>,
    llm: &Arc<MultiBackend>,
    sessions: &SessionStore,
    refresh_lock: &Arc<tokio::sync::Mutex<()>>,
) -> Result<Vec<ModelMetadata>, String> {
    if let Some((cx, session_id)) = cx.zip(session_id) {
        trace_openrouter_refresh("OpenRouter refresh requested.");
        send_message(cx, session_id, "OpenRouter refresh requested.\n");
        trace_openrouter_refresh("Waiting for model refresh lock...");
        send_message(cx, session_id, "Waiting for model refresh lock...\n");
    }

    let _guard = tokio::time::timeout(MODEL_REFRESH_LOCK_WAIT, refresh_lock.lock())
        .await
        .map_err(|_| {
            "another model refresh is already running; if it is wedged, wait a moment and try again"
                .to_string()
        })?;

    if let Some((cx, session_id)) = cx.zip(session_id) {
        trace_openrouter_refresh("Refresh lock acquired.");
        send_message(cx, session_id, "Refresh lock acquired.\n");
    }

    refresh_model_catalog_after_lock(cx, session_id, llm, sessions).await
}

async fn refresh_model_catalog_after_lock(
    cx: Option<&ConnectionTo<Client>>,
    session_id: Option<&str>,
    llm: &Arc<MultiBackend>,
    sessions: &SessionStore,
) -> Result<Vec<ModelMetadata>, String> {
    let models = if let Some((cx, session_id)) = cx.zip(session_id) {
        trace_openrouter_refresh("Refreshing model catalog...");
        send_message(cx, session_id, "Refreshing model catalog...\n");
        trace_openrouter_refresh("Preparing provider discovery...");
        send_message(cx, session_id, "Preparing provider discovery...\n");
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let list_future = llm.list_model_metadata_with_progress(Some(tx));
        tokio::pin!(list_future);

        let models = loop {
            tokio::select! {
                maybe_chunk = rx.recv() => {
                    if let Some(chunk) = maybe_chunk {
                        trace_openrouter_refresh(chunk.trim_end());
                        send_message(cx, session_id, &chunk);
                    }
                }
                result = &mut list_future => {
                    break result.map_err(|e| format!("{e:#}"))?;
                }
            }
        };

        while let Ok(chunk) = rx.try_recv() {
            trace_openrouter_refresh(chunk.trim_end());
            send_message(cx, session_id, &chunk);
        }
        models
    } else {
        llm.list_model_metadata_with_progress(None)
            .await
            .map_err(|e| format!("{e:#}"))?
    };
    if let Some(model) = preferred_model(&models) {
        sessions.set_default_model(model).await;
    }
    sessions.set_available_models(models.clone()).await;
    if let Some((cx, session_id)) = cx.zip(session_id) {
        trace_openrouter_refresh(&format!(
            "Catalog refresh complete: {} model(s) total.",
            models.len()
        ));
        send_message(
            cx,
            session_id,
            &format!(
                "Catalog refresh complete: {} model(s) total.\n",
                models.len()
            ),
        );
    }
    Ok(models)
}

async fn handle_setup_choose(
    cx: &ConnectionTo<Client>,
    sessions: &SessionStore,
    session_id: &str,
    llm: &Arc<MultiBackend>,
    refresh_lock: &Arc<tokio::sync::Mutex<()>>,
) -> String {
    let catalog =
        match refresh_model_catalog_now(Some(cx), Some(session_id), llm, sessions, refresh_lock)
            .await
        {
            Ok(models) => models,
            Err(e) => {
                return format!(
                    "Anvil could not find a working model yet: {e}\n\n\
                 Try `/setup codex` if you use Codex, or `/setup local` for free local models."
                );
            }
        };
    let Some(model) = preferred_model(&catalog) else {
        return format!(
            "Anvil could not find a working model yet.\n\n{}",
            render_setup_home_for_model("", &catalog)
        );
    };
    match apply_setup_config(cx, sessions, session_id, MODEL_CONFIG_ID, &model).await {
        msg if msg.starts_with("Error:") => msg,
        _ => "Anvil is ready. Run `/setup` anytime to change or repair model setup.".to_string(),
    }
}

async fn handle_setup_local(
    cx: &ConnectionTo<Client>,
    sessions: &SessionStore,
    session_id: &str,
    llm: &Arc<MultiBackend>,
    refresh_lock: &Arc<tokio::sync::Mutex<()>>,
    rest: &str,
) -> String {
    match rest.to_ascii_lowercase().as_str() {
        "use" | "choose" => {
            let catalog = sessions.available_model_metadata().await;
            if let Some(model) = catalog
                .iter()
                .find(|m| split_wire_id(&m.id).is_some_and(|(s, _)| s == ModelSource::Ollama))
                .map(|m| m.id.clone())
            {
                return apply_setup_config(cx, sessions, session_id, MODEL_CONFIG_ID, &model).await;
            }
            "No local model is ready yet. Install Ollama, start it, then run `/setup local refresh`."
                .to_string()
        }
        "refresh" | "try-again" => {
            match refresh_model_catalog_now(Some(cx), Some(session_id), llm, sessions, refresh_lock)
                .await
            {
                Ok(catalog) => {
                    let local_count = source_count(&catalog, ModelSource::Ollama);
                    if local_count > 0 {
                        "Local models are ready. Run `/setup local use` to use them, or `/setup choose` to let Anvil pick.".to_string()
                    } else {
                        render_local_setup_help()
                    }
                }
                Err(e) => format!(
                    "Could not check local models yet: {e}\n\n{}",
                    render_local_setup_help()
                ),
            }
        }
        _ => render_local_setup_help(),
    }
}

fn render_local_setup_help() -> String {
    "Use free local models\n\n\
     Anvil looks for Ollama automatically. You do not need to know ports or model ids.\n\n\
     1. Install Ollama from https://ollama.com\n\
     2. Start Ollama.\n\
     3. Run `/setup local refresh`.\n\
     4. Run `/setup local use`.\n\n\
     llama.cpp and custom local servers belong in `/setup advanced`."
        .to_string()
}

async fn handle_setup_bedrock(
    cx: &ConnectionTo<Client>,
    sessions: &SessionStore,
    session_id: &str,
    llm: &Arc<MultiBackend>,
    refresh_lock: &Arc<tokio::sync::Mutex<()>>,
    rest: &str,
) -> String {
    use crate::bedrock_client::BEDROCK_DEFAULT_MODEL;

    if rest.is_empty() {
        return render_bedrock_setup_help();
    }
    let lower = rest.to_ascii_lowercase();
    if matches!(lower.as_str(), "refresh" | "try-again") {
        return match refresh_model_catalog_now(
            Some(cx),
            Some(session_id),
            llm,
            sessions,
            refresh_lock,
        )
        .await
        {
            Ok(catalog) => {
                let count = source_count(&catalog, ModelSource::Bedrock);
                if count > 0 {
                    format!(
                        "Bedrock models are ready ({count} found). Run `/setup choose`, or use `/setup model` for advanced selection."
                    )
                } else {
                    format!(
                        "Bedrock is not showing models yet.\n\n{}",
                        render_bedrock_setup_help()
                    )
                }
            }
            Err(e) => format!(
                "Could not check Bedrock yet: {e}\n\n{}",
                render_bedrock_setup_help()
            ),
        };
    }

    if let Some(key) = rest.strip_prefix("key ") {
        let state = crate::bedrock_auth::CredentialState::snapshot();
        if state.env_owns() {
            return format!(
                "Bedrock credentials are managed by the {} environment variable. \
                 Unset it and restart before using `/setup bedrock key`.",
                crate::bedrock_client::BEDROCK_API_KEY_ENV
            );
        }
        let key = key.trim();
        if key.is_empty() {
            return "Provide a bearer token: `/setup bedrock key <token>`.".to_string();
        }

        let existing = crate::bedrock_auth::read().unwrap_or(None);
        let region = existing
            .as_ref()
            .and_then(|a| a.region.clone())
            .unwrap_or_else(crate::bedrock_auth::region_from_any_source);
        let default_model = existing
            .as_ref()
            .and_then(|a| a.default_model.clone())
            .unwrap_or_else(crate::bedrock_auth::model_from_any_source);
        let auth = crate::bedrock_auth::BedrockAuth {
            bearer_token: key.to_string(),
            region: Some(region.clone()),
            default_model: Some(default_model.clone()),
        };
        match crate::bedrock_auth::write(&auth) {
            Ok(()) => {
                let backend: Arc<dyn crate::llm_client::LlmBackend> =
                    Arc::new(crate::bedrock_client::BedrockClient::new(
                        key.to_string(),
                        region.clone(),
                        default_model.clone(),
                    ));
                llm.install_bedrock(backend);
                spawn_background_refresh(
                    refresh_lock.clone(),
                    llm.clone(),
                    sessions.clone(),
                    Some((
                        cx.clone(),
                        session_id.to_string(),
                        "Refreshing model catalog after Bedrock setup...",
                    )),
                    None,
                );
                format!(
                    "Bedrock credentials saved.\n\
                     Token: saved (length {})\n\
                     Region: {region}\n\
                     Model: {default_model}\n\n\
                     Run `/setup choose` or `/setup model` to pick a Bedrock model.\n\n\
                     Tip: change region with `/setup bedrock region <region>`\n\
                     Tip: change model with `/setup bedrock model <model_id>`",
                    key.len()
                )
            }
            Err(e) => format!("Failed to save Bedrock credentials: {e:#}"),
        }
    } else if let Some(region) = rest.strip_prefix("region ") {
        let region = region.trim();
        if region.is_empty() {
            return "Provide a region: `/setup bedrock region <region>` (e.g. us-east-1)."
                .to_string();
        }
        let mut auth = match crate::bedrock_auth::read() {
            Ok(Some(a)) => a,
            _ => {
                return "No Bedrock credentials saved yet. Run `/setup bedrock key <token>` first."
                    .to_string();
            }
        };
        auth.region = Some(region.to_string());
        match crate::bedrock_auth::write(&auth) {
            Ok(()) => {
                let backend: Arc<dyn crate::llm_client::LlmBackend> =
                    Arc::new(crate::bedrock_client::BedrockClient::new(
                        auth.bearer_token.clone(),
                        region.to_string(),
                        auth.default_model
                            .clone()
                            .unwrap_or_else(|| BEDROCK_DEFAULT_MODEL.to_string()),
                    ));
                llm.install_bedrock(backend);
                spawn_background_refresh(
                    refresh_lock.clone(),
                    llm.clone(),
                    sessions.clone(),
                    Some((
                        cx.clone(),
                        session_id.to_string(),
                        "Refreshing model catalog after Bedrock region change...",
                    )),
                    None,
                );
                format!("Bedrock region set to {region}.")
            }
            Err(e) => format!("Failed to save Bedrock region: {e:#}"),
        }
    } else if let Some(model) = rest.strip_prefix("model ") {
        let model = model.trim();
        if model.is_empty() {
            return "Provide a model id: `/setup bedrock model <model_id>` (e.g. us.anthropic.claude-sonnet-4-6).".to_string();
        }
        let mut auth = match crate::bedrock_auth::read() {
            Ok(Some(a)) => a,
            _ => {
                return "No Bedrock credentials saved yet. Run `/setup bedrock key <token>` first."
                    .to_string();
            }
        };
        auth.default_model = Some(model.to_string());
        match crate::bedrock_auth::write(&auth) {
            Ok(()) => {
                let backend: Arc<dyn crate::llm_client::LlmBackend> =
                    Arc::new(crate::bedrock_client::BedrockClient::new(
                        auth.bearer_token.clone(),
                        auth.region
                            .clone()
                            .unwrap_or_else(crate::bedrock_auth::region_from_any_source),
                        model.to_string(),
                    ));
                llm.install_bedrock(backend);
                spawn_background_refresh(
                    refresh_lock.clone(),
                    llm.clone(),
                    sessions.clone(),
                    Some((
                        cx.clone(),
                        session_id.to_string(),
                        "Refreshing model catalog after Bedrock model change...",
                    )),
                    None,
                );
                format!("Bedrock default model set to {model}.")
            }
            Err(e) => format!("Failed to save Bedrock model: {e:#}"),
        }
    } else {
        match lower.as_str() {
            "status" => {
                let state = crate::bedrock_auth::CredentialState::snapshot();
                if state.env_set {
                    format!(
                        "Bedrock is configured via {} environment variable.\n\
                         Region: {}\n\
                         Model: {}",
                        crate::bedrock_client::BEDROCK_API_KEY_ENV,
                        crate::bedrock_auth::region_from_any_source(),
                        crate::bedrock_auth::model_from_any_source(),
                    )
                } else {
                    match crate::bedrock_auth::read() {
                        Ok(Some(auth)) => {
                            let region = auth.region.as_deref().unwrap_or("(default)");
                            let model = auth.default_model.as_deref().unwrap_or("(default)");
                            format!(
                                "Bedrock credentials:\n  Token: saved (length {})\n  Region: {region}\n  Model: {model}",
                                auth.bearer_token.len()
                            )
                        }
                        Ok(None) => {
                            "No Bedrock credentials found. Run `/setup bedrock key <token>`."
                                .to_string()
                        }
                        Err(e) => format!("Failed to read Bedrock credentials: {e:#}"),
                    }
                }
            }
            "disconnect" if crate::bedrock_auth::CredentialState::snapshot().env_owns() => {
                format!(
                    "Bedrock credentials are managed by the {} environment variable. \
                     Unset it and restart to disconnect Bedrock.",
                    crate::bedrock_client::BEDROCK_API_KEY_ENV
                )
            }
            "disconnect" => match crate::bedrock_auth::logout() {
                Ok(()) => {
                    llm.uninstall_bedrock();
                    spawn_background_refresh(
                        refresh_lock.clone(),
                        llm.clone(),
                        sessions.clone(),
                        Some((
                            cx.clone(),
                            session_id.to_string(),
                            "Refreshing model catalog after Bedrock disconnect...",
                        )),
                        None,
                    );
                    "Bedrock credentials cleared. Run `/setup bedrock key <token>` to reconnect."
                        .to_string()
                }
                Err(e) => format!("Failed to remove Bedrock credentials: {e:#}"),
            },
            _ => format!(
                "Unknown Bedrock setup option `{rest}`.\n\n{}",
                render_bedrock_setup_help()
            ),
        }
    }
}

fn render_bedrock_setup_help() -> String {
    let state = crate::bedrock_auth::CredentialState::snapshot();
    let status = match state.active_source() {
        "env" => format!(
            "Bedrock is connected from the {} environment variable.",
            crate::bedrock_client::BEDROCK_API_KEY_ENV
        ),
        "file" => "Bedrock is connected from saved credentials.".to_string(),
        _ => "Bedrock is not connected.".to_string(),
    };
    let key_help = if state.env_owns() {
        "Credentials are managed by the environment variable. Unset it and restart to use `/setup bedrock key`."
            .to_string()
    } else {
        "If you have a Bedrock bearer token, run:\n`/setup bedrock key <token>`".to_string()
    };
    format!(
        "Use AWS Bedrock\n\n\
         {status}\n\n\
         {key_help}\n\n\
         You also need:\n\
         - A region (default: us-east-1): `/setup bedrock region <region>`\n\
         - A model (default: us.anthropic.claude-sonnet-4-6): `/setup bedrock model <id>`\n\n\
         Other commands:\n\
         - `/setup bedrock status`\n\
         - `/setup bedrock disconnect`\n\
         - `/setup bedrock refresh`\n\n\
         Choose for me: `/setup choose`."
    )
}

async fn handle_setup_openrouter(
    cx: &ConnectionTo<Client>,
    session_id: &str,
    rest: &str,
    llm: &Arc<MultiBackend>,
    sessions: &SessionStore,
    refresh_lock: &Arc<tokio::sync::Mutex<()>>,
) -> String {
    if rest.is_empty() {
        return render_openrouter_setup_help();
    }
    let lower = rest.to_ascii_lowercase();
    if matches!(lower.as_str(), "refresh" | "try-again") {
        return match refresh_model_catalog_now(
            Some(cx),
            Some(session_id),
            llm,
            sessions,
            refresh_lock,
        )
        .await
        {
            Ok(catalog) => {
                let count = source_count(&catalog, ModelSource::OpenRouter);
                if count > 0 {
                    "OpenRouter models are ready. Run `/setup choose`, or use `/setup model` for advanced selection.".to_string()
                } else {
                    format!(
                        "OpenRouter is not showing models yet.\n\n{}",
                        render_openrouter_setup_help()
                    )
                }
            }
            Err(e) => format!(
                "Could not check OpenRouter yet: {e}\n\n{}",
                render_openrouter_setup_help()
            ),
        };
    }

    let prompt = match rest.split_once(char::is_whitespace) {
        Some((cmd, key)) if cmd.eq_ignore_ascii_case("key") && !key.trim().is_empty() => {
            format!("/openrouter-login {}", key.trim())
        }
        _ if matches!(lower.as_str(), "status" | "disconnect") => {
            format!("/openrouter-login {rest}")
        }
        _ if rest.starts_with("sk-") => format!("/openrouter-login {rest}"),
        _ => {
            return format!(
                "Unknown OpenRouter setup option `{rest}`.\n\n{}",
                render_openrouter_setup_help()
            );
        }
    };
    let mut out = handle_openrouter_login(
        &prompt,
        llm,
        sessions,
        refresh_lock,
        Some(cx),
        Some(session_id),
    )
    .await;
    out.push_str("\n\nRun `/setup choose` after OpenRouter is connected.");
    out
}

fn render_openrouter_setup_help() -> String {
    let state = crate::openrouter_auth::CredentialState::snapshot();
    let status = match state.active_source() {
        "env" => "OpenRouter is connected from the OPENROUTER_API_KEY environment variable.",
        "file" => "OpenRouter is connected from saved credentials.",
        _ => "OpenRouter is not connected.",
    };
    format!(
        "Use OpenRouter\n\n\
         {status}\n\n\
         If you already know OpenRouter and have a key, run:\n\
         `/setup openrouter key <your key>`\n\n\
         Other useful commands:\n\
         - `/setup openrouter status`\n\
         - `/setup openrouter disconnect`\n\
         - `/setup openrouter refresh`\n\n\
         Choose for me: `/setup choose`."
    )
}

async fn handle_permissions(
    sessions: &SessionStore,
    session_id: &str,
    prompt_text: &str,
) -> String {
    let rest = slash_command_args(prompt_text);
    if rest.is_empty() {
        return "Remembered Always allow approvals:\n\n\
                - `/permissions list` - Show remembered approvals for this repo.\n\
                - `/permissions revoke <number-or-key>` - Forget one remembered approval.\n\
                - `/permissions clear` - Forget all remembered approvals."
            .to_string();
    }
    let (action, arg) = split_setup_action(&rest);
    match action.to_ascii_lowercase().as_str() {
        "list" | "show" | "always" | "remembered" => {
            return render_always_allowed_permissions(sessions, session_id).await;
        }
        "revoke" | "remove" | "forget" => {
            return revoke_always_allowed_permission(sessions, session_id, arg).await;
        }
        "clear" | "reset" => return clear_always_allowed_permissions(sessions, session_id).await,
        _ => "Unknown permissions command. Try `/permissions list`, \
                    `revoke`, or `clear`. Permission mode is changed through \
                    the session Permission selector."
            .to_string(),
    }
}

fn split_setup_action(input: &str) -> (&str, &str) {
    let trimmed = input.trim();
    match trimmed.find(char::is_whitespace) {
        Some(idx) => {
            let (action, rest) = trimmed.split_at(idx);
            (action, rest.trim())
        }
        None => (trimmed, ""),
    }
}

async fn render_always_allowed_permissions(sessions: &SessionStore, session_id: &str) -> String {
    let Some(keys) = sessions.always_allow_keys(session_id).await else {
        return "Error: unknown session.".to_string();
    };
    if keys.is_empty() {
        return "No remembered Always allow approvals.".to_string();
    }

    let mut out = String::from("Remembered Always allow approvals for this repo:\n\n");
    for (idx, key) in keys.iter().enumerate() {
        out.push_str(&format!(
            "{}. {}\n",
            idx + 1,
            describe_always_allow_key(key)
        ));
        out.push_str(&format!("   Key: `{key}`\n"));
    }
    out.push_str(
        "\nUse `/permissions revoke <number>` to forget one, or \
         `/permissions clear` to forget all.",
    );
    out
}

async fn revoke_always_allowed_permission(
    sessions: &SessionStore,
    session_id: &str,
    arg: &str,
) -> String {
    if arg.is_empty() {
        return "Usage: `/permissions revoke <number-or-key>`.\n\
                Run `/permissions list` to see remembered approvals."
            .to_string();
    }

    let Some(keys) = sessions.always_allow_keys(session_id).await else {
        return "Error: unknown session.".to_string();
    };
    let key = match arg.parse::<usize>() {
        Ok(index) if (1..=keys.len()).contains(&index) => keys[index - 1].clone(),
        Ok(_) => {
            return format!(
                "No remembered Always allow approval numbered `{arg}`. \
                 Run `/permissions list` to see valid numbers."
            );
        }
        Err(_) => arg.to_string(),
    };

    match sessions.remove_always_allow(session_id, &key).await {
        Some(true) => format!(
            "Forgot Always allow approval: {}",
            describe_always_allow_key(&key)
        ),
        Some(false) => "No matching remembered Always allow approval was found.".to_string(),
        None => "Error: unknown session.".to_string(),
    }
}

async fn clear_always_allowed_permissions(sessions: &SessionStore, session_id: &str) -> String {
    match sessions.clear_always_allow(session_id).await {
        Some(0) => "No remembered Always allow approvals to clear.".to_string(),
        Some(1) => "Forgot 1 remembered Always allow approval.".to_string(),
        Some(count) => format!("Forgot {count} remembered Always allow approvals."),
        None => "Error: unknown session.".to_string(),
    }
}

fn describe_always_allow_key(key: &str) -> String {
    let parsed = serde_json::from_str::<serde_json::Value>(key).ok();
    if let Some(value) = parsed
        && value.get("tool").and_then(serde_json::Value::as_str) == Some("run_shell_command")
    {
        if value.get("rule").and_then(serde_json::Value::as_str) == Some("prefix") {
            let prefix = value
                .get("argvPrefix")
                .and_then(serde_json::Value::as_array)
                .map(|argv| {
                    argv.iter()
                        .filter_map(serde_json::Value::as_str)
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .filter(|joined| !joined.is_empty())
                .unwrap_or_else(|| "(unknown prefix)".to_string());
            return format!("run_shell_command prefix `{prefix}` in this repo");
        }
        // Legacy exact-command keys are no longer stored (they are purged on
        // load), but describe any straggler passed in by `/permissions revoke`.
        let command = value
            .get("command")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("(unknown command)");
        return format!("run_shell_command `{command}` in this repo");
    }

    format!("tool `{key}`")
}

/// Configure the session's effective sandbox mode. Separate from permission
/// mode: this controls the sandbox boundary and parser backend, not whether
/// the user is prompted before each tool call.
///
/// The choice is saved as an install-level setup preference and seeds new
/// sessions and cold reloads. It is still kept out of session manifests so
/// an old zip cannot impose a sandbox policy.
async fn handle_setup_sandbox(sessions: &SessionStore, session_id: &str, rest: &str) -> String {
    use crate::sandbox_backend::SandboxMode;

    if rest.is_empty() {
        let current = sessions.sandbox_mode(session_id).await;
        let (state, suffix) = match current {
            Some(mode) => (
                crate::sandbox_backend::resolve_mode(mode).as_str(),
                if mode.is_none() { " (default)" } else { "" },
            ),
            None => return "Error: unknown session.".to_string(),
        };
        let wasm_line = if crate::sandbox_backend::wasm_sandbox_compiled() {
            "- `/setup sandbox wasm`   - wasm parsing, no OS sandbox for shell commands."
        } else {
            "- `/setup sandbox wasm`   - unavailable in this build."
        };
        return format!(
            "Sandbox is currently `{state}`{suffix}.\n\n\
             - `/setup sandbox default` - use the process default.\n\
             - `/setup sandbox os`     - OS sandbox + native parsing.\n\
             {wasm_line}\n\
             - `/setup sandbox off`    - no sandbox at all.\n\
             - `/setup sandbox status` - report current mode."
        );
    }
    let mode = match rest.to_ascii_lowercase().as_str() {
        "default" | "on" | "enable" | "enabled" | "true" | "yes" => {
            None // clear override -> use global default
        }
        "os" => Some(SandboxMode::Os),
        "wasm" => Some(SandboxMode::Wasm),
        "off" | "disable" | "disabled" | "false" | "no" => Some(SandboxMode::Off),
        "status" => {
            let current = sessions.sandbox_mode(session_id).await;
            let Some(mode) = current else {
                return "Error: unknown session.".to_string();
            };
            return describe_sandbox_mode(
                crate::sandbox_backend::resolve_mode(mode),
                mode.is_none(),
            );
        }
        _ => {
            return "Unknown choice. Try `/setup sandbox`, `/setup sandbox default`, `/setup sandbox os`, `/setup sandbox wasm`, `/setup sandbox off`, or `/setup sandbox status`.".to_string();
        }
    };
    if let Err(e) = crate::sandbox_backend::backend_for_mode(mode) {
        return format!("Error: failed to initialize requested sandbox backend: {e}");
    }
    if !sessions.set_sandbox_mode(session_id, mode).await {
        return "Error: unknown session.".to_string();
    }
    match mode {
        Some(SandboxMode::Os) => "Sandbox set to `os`. Shell commands use the OS sandbox; parsing runs natively. Per-call permission prompts are unchanged. This preference will apply to future sessions.".to_string(),
        Some(SandboxMode::Wasm) => "Sandbox set to `wasm`. Parsing goes through WASM sandbox; shell commands will run without OS sandbox. Per-call permission prompts are unchanged. This preference will apply to future sessions.".to_string(),
        Some(SandboxMode::Off) => "Sandbox set to `off`. No sandboxing at all. Per-call permission prompts are unchanged. This preference will apply to future sessions.".to_string(),
        _ => {
            let default = crate::sandbox_backend::default_mode();
            format!(
                "Sandbox reset to default (`{}`). This preference will apply to future sessions.",
                default.as_str()
            )
        }
    }
}

fn describe_sandbox_mode(mode: crate::sandbox_backend::SandboxMode, is_default: bool) -> String {
    let suffix = if is_default { " (default)" } else { "" };
    match mode {
        crate::sandbox_backend::SandboxMode::Os => {
            format!(
                "Sandbox is `os`{suffix}. Shell commands use the OS sandbox; parsing is native."
            )
        }
        crate::sandbox_backend::SandboxMode::Wasm => {
            format!(
                "Sandbox is `wasm`{suffix}. Parsing goes through WASM; shell commands have no OS sandbox."
            )
        }
        crate::sandbox_backend::SandboxMode::Off => {
            format!("Sandbox is `off`{suffix}. No sandboxing at all.")
        }
    }
}

async fn handle_setup_mode(
    cx: &ConnectionTo<Client>,
    sessions: &SessionStore,
    session_id: &str,
    rest: &str,
) -> String {
    if rest.is_empty() {
        return "How should Anvil behave?\n\n\
                - `/setup mode agent` - General coding assistant.\n\
                - `/setup mode code` - Focus on code changes.\n\
                - `/setup mode ask` - Answer questions.\n\
                - `/setup mode plan` - Plan only."
            .to_string();
    }
    let value = match rest.to_ascii_lowercase().as_str() {
        "agent" | "default" | "lutz" => "LUTZ",
        "code" => "CODE",
        "ask" => "ASK",
        "plan" => "PLAN",
        _ => return "Unknown mode. Try `/setup mode agent`, `code`, `ask`, or `plan`.".to_string(),
    };
    apply_setup_config(cx, sessions, session_id, BEHAVIOR_CONFIG_ID, value).await
}

async fn apply_setup_config(
    cx: &ConnectionTo<Client>,
    sessions: &SessionStore,
    session_id: &str,
    key: &str,
    value: &str,
) -> String {
    match apply_config_option(sessions, session_id, key, value).await {
        Ok(outcome) => {
            // Route through the shared helper so a `/setup mode` change also
            // emits current_mode_update for the legacy modes surface (#157),
            // matching the session/set_config_option request path.
            send_config_option_change_updates(cx, session_id, key, value, outcome.updated_options);
            let fallback_cwd = std::env::current_dir().unwrap_or_default();
            send_session_usage_update(cx, sessions, session_id, &fallback_cwd).await;
            let mut msg = match key {
                MODEL_CONFIG_ID => "Model setup updated.".to_string(),
                PERMISSION_CONFIG_ID => "Permission mode updated.".to_string(),
                BEHAVIOR_CONFIG_ID => "Behavior setup updated.".to_string(),
                REASONING_EFFORT_CONFIG_ID => "Advanced reasoning setup updated.".to_string(),
                _ => "Setup updated.".to_string(),
            };
            if let Some(prev) = outcome.cleared_reasoning {
                msg.push_str(&format!(
                    "\nReasoning effort reset: `{prev}` is not supported by the new model."
                ));
            }
            msg
        }
        Err(e) => format!("Error: {}", e.human_message()),
    }
}

fn render_setup_models(catalog: &[ModelMetadata]) -> String {
    if catalog.is_empty() {
        return "No models are in the catalog yet. Run `/setup refresh`.".to_string();
    }
    let mut out = String::from("Advanced model selection\n\nUse `/setup model <model id>`.\n\n");

    {
        let mut write_group = |title: &str, models: Vec<String>, empty: &str| {
            out.push_str(title);
            out.push('\n');
            if models.is_empty() {
                out.push_str(empty);
                out.push('\n');
            } else {
                for id in models {
                    out.push_str(&format!("- `{id}`\n"));
                }
            }
            out.push('\n');
        };

        write_group(
            "Bedrock",
            source_model_ids(catalog, ModelSource::Bedrock, 12),
            "No Bedrock models found. Run `/setup bedrock` to configure your token and region.",
        );
        write_group(
            "Codex",
            source_model_ids(catalog, ModelSource::Codex, 12),
            "No Codex models found. Run `/setup codex`.",
        );
        write_group(
            "Local models",
            source_model_ids(catalog, ModelSource::Ollama, 12),
            "No local models found. Run `/setup local`.",
        );
        write_group(
            "ds4 (DeepSeek V4)",
            source_model_ids(catalog, ModelSource::Ds4, 12),
            "No ds4 models found. Start `ds4-server` (antirez/ds4), or set DS4_BASE_URL.",
        );
        write_group(
            "DeepSeek",
            source_model_ids(catalog, ModelSource::DeepSeek, 12),
            "No hosted DeepSeek models found. Export DEEPSEEK_API_KEY and refresh.",
        );
        write_group(
            "OpenRouter",
            filtered_openrouter_models(catalog),
            "No OpenRouter coding candidates found. Run `/setup openrouter`.",
        );
    }

    let openrouter_total = source_count(catalog, ModelSource::OpenRouter);
    if openrouter_total > 0 {
        out.push_str(&format!(
            "OpenRouter list is filtered for chat and coding models ({openrouter_total} total in the raw catalog).\n"
        ));
    }
    out
}

fn source_model_ids(catalog: &[ModelMetadata], source: ModelSource, limit: usize) -> Vec<String> {
    catalog
        .iter()
        .filter(|m| split_wire_id(&m.id).is_some_and(|(s, _)| s == source))
        .take(limit)
        .map(|m| m.id.clone())
        .collect()
}

fn filtered_openrouter_models(catalog: &[ModelMetadata]) -> Vec<String> {
    const EXCLUDE: &[&str] = &[
        "image",
        "vision",
        "audio",
        "tts",
        "embedding",
        "moderation",
        "free",
    ];
    const INCLUDE: &[&str] = &[
        "claude",
        "gpt",
        "gemini",
        "qwen",
        "deepseek",
        "codestral",
        "kimi",
        "mistral",
        "llama",
    ];
    catalog
        .iter()
        .filter(|m| split_wire_id(&m.id).is_some_and(|(s, _)| s == ModelSource::OpenRouter))
        .filter(|m| {
            let id = m.id.to_ascii_lowercase();
            INCLUDE.iter().any(|needle| id.contains(needle))
                && !EXCLUDE.iter().any(|needle| id.contains(needle))
        })
        .take(8)
        .map(|m| m.id.clone())
        .collect()
}

async fn render_setup_advanced(sessions: &SessionStore, session_id: &str) -> String {
    let fallback_cwd = std::env::current_dir().unwrap_or_default();
    let Some(session) = sessions.get_session(session_id, &fallback_cwd).await else {
        return "Error: unknown session.".to_string();
    };
    let catalog = sessions.available_model_metadata().await;
    let openrouter_picks = filtered_openrouter_models(&catalog);
    let mut out = String::from("Advanced setup\n\n");
    out.push_str(&format!(
        "- Selected model: `{}`\n",
        if session.model.is_empty() {
            "(none)"
        } else {
            &session.model
        }
    ));
    out.push_str(&format!(
        "- Permission mode: `{}`\n",
        session.permission_mode.as_str()
    ));
    out.push_str(&format!(
        "- Sandbox mode: `{}`\n",
        crate::sandbox_backend::resolve_mode(session.sandbox_mode).as_str()
    ));
    out.push_str(&format!("- Behavior mode: `{}`\n", session.mode.as_str()));
    out.push_str(&format!(
        "- Reasoning effort: `{}`\n",
        session
            .selected_reasoning_effort
            .as_deref()
            .unwrap_or(REASONING_EFFORT_DEFAULT_VALUE)
    ));
    out.push_str(&format!(
        "- LLM idle timeout: `{}`\n\n",
        session
            .idle_timeout_secs
            .map(|s| format!("{s}s"))
            .unwrap_or_else(|| "server default".to_string())
    ));
    out.push_str("Commands:\n");
    out.push_str("- `/setup model` - list model ids.\n");
    out.push_str("- `/setup model <model id>` - choose a specific model.\n");
    out.push_str("- Permission selector - change edit/command approval mode.\n");
    out.push_str("- `/permissions` - list or revoke remembered Always allow approvals.\n");
    out.push_str(
        "- `/setup sandbox default|os|wasm|off` - choose the sandbox strategy for this and future sessions.\n",
    );
    out.push_str("- `/setup mode` - change assistant behavior.\n");
    out.push_str("- `/setup timeout <seconds>` - change stream idle timeout.\n");
    out.push_str("- `/setup reasoning default|off|<level>` - advanced reasoning setting.\n");
    if !openrouter_picks.is_empty() {
        out.push_str("\nFiltered OpenRouter coding candidates:\n");
        for id in openrouter_picks {
            out.push_str(&format!("- `{id}`\n"));
        }
    }
    out
}

/// Trimmed args for a slash command. Returns the empty string when the
/// prompt is not a slash command at all, or when the command has no
/// trailing args. Shared by setup and `/pr-create` -- both want "args
/// after the command name, trimmed of surrounding whitespace".
fn slash_command_args(prompt_text: &str) -> String {
    parse_slash_command(prompt_text)
        .map(|(_, a)| a)
        .unwrap_or_default()
        .trim()
        .to_string()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LoopSpec {
    interval_secs: u64,
    target: String,
}

fn loop_target_runs_without_model(target: &str) -> bool {
    is_slash_command(target, "context")
        || is_slash_command(target, "setup")
        || is_slash_command(target, "permissions")
        || is_slash_command(target, "mcp")
        || is_slash_command(target, "pr-create")
        || is_slash_command(target, "usage")
}

fn parse_loop_command(prompt_text: &str) -> Result<LoopSpec, String> {
    let trimmed = slash_command_args(prompt_text);
    if trimmed.is_empty() {
        return Err("Usage: `/loop <seconds> <slash-command-or-prompt>`\n\
             Example: `/loop 30 /context`\n\
             Example: `/loop 300 check CI status`\n\n\
             The loop runs until you cancel the session."
            .to_string());
    }

    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let raw_secs = parts.next().unwrap_or("");
    let target = parts.next().unwrap_or("").trim();
    if target.is_empty() {
        return Err("Usage: `/loop <seconds> <slash-command-or-prompt>`\n\
             Missing command or prompt after the interval."
            .to_string());
    }
    if is_slash_command(target, "loop") {
        return Err("Nested `/loop` is not supported.".to_string());
    }

    let interval_secs = match raw_secs.parse::<u64>() {
        Ok(secs) if (1..=86_400).contains(&secs) => secs,
        Ok(other) => {
            return Err(format!(
                "Interval `{other}` is out of range. Pick a value between 1 and 86400 seconds."
            ));
        }
        Err(_) => {
            return Err(format!(
                "Invalid interval `{raw_secs}`. Usage: `/loop <seconds> <slash-command-or-prompt>`"
            ));
        }
    };

    Ok(LoopSpec {
        interval_secs,
        target: target.to_string(),
    })
}

/// Bounds for the *optional* `--max-turns` guardrail. A goal is unbounded by
/// default: the stopping condition is the model's verified completion or a
/// genuine block, not an arbitrary turn count -- a turn cap that fired on its
/// own would stop the agent before the goal is met, defeating the purpose.
/// (This matches Codex, whose token budget is `Option` and defaults to none.)
/// `--max-turns` only applies when the user explicitly opts into a ceiling,
/// and then must fall in this range.
const GOAL_MIN_MAX_TURNS: u32 = 1;
const GOAL_MAX_MAX_TURNS: u32 = 10_000;

/// Sentinel the model emits, alone on the final line, once it has verified
/// the objective is complete. Detected by [`detect_goal_signal`].
const GOAL_COMPLETE_SENTINEL: &str = "GOAL_COMPLETE";
/// Sentinel prefix the model emits when genuinely at an impasse.
const GOAL_BLOCKED_SENTINEL: &str = "GOAL_BLOCKED";
/// How many consecutive blocked reports are required before the loop stops
/// and hands back to the user. Mirrors Codex's three-turn blocked rule so
/// the agent doesn't surrender on a transient blocker.
const GOAL_BLOCKED_THRESHOLD: u32 = 3;

/// A parsed `/goal` invocation. `max_turns` is `None` for an unbounded goal
/// (the default) and `Some(n)` only when the user opts into a ceiling.
#[derive(Debug, Clone, PartialEq, Eq)]
struct GoalSpec {
    objective: String,
    max_turns: Option<u32>,
}

/// Which framing the continuation prompt uses for a given turn.
#[derive(Clone, Copy)]
enum GoalPhase {
    /// A normal continuation turn: make verifiable progress.
    Continue,
    /// The last turn of an opt-in `--max-turns` ceiling: wrap up cleanly and
    /// summarize. Never used for an unbounded goal.
    FinalWrapUp,
}

/// The stop signal (if any) parsed from a goal turn's assistant text.
#[derive(Debug, PartialEq, Eq)]
enum GoalSignal {
    Complete,
    Blocked(String),
    Continue,
}

fn goal_usage() -> String {
    "Usage: `/goal [--max-turns N] <objective>`\n\
     Example: `/goal make `cargo test` pass`\n\
     Example: `/goal --max-turns 40 migrate the config loader to serde`\n\n\
     Anvil works autonomously across turns until the objective is verifiably met or \
     it is blocked -- there is no turn limit by default. Cancel the session to stop \
     early, or pass `--max-turns N` to set an optional ceiling."
        .to_string()
}

/// Parse `/goal [--max-turns N] <objective>`.
///
/// `--max-turns` (also `--max-turns=N`) is optional and, when present, must
/// lead; without it the goal is unbounded (`max_turns: None`). The remainder
/// is the free-text objective. An empty objective or an out-of-range ceiling
/// is a user error returned as a usage string.
fn parse_goal_command(prompt_text: &str) -> Result<GoalSpec, String> {
    let args = slash_command_args(prompt_text);
    let trimmed = args.trim();
    if trimmed.is_empty() {
        return Err(goal_usage());
    }

    let mut max_turns: Option<u32> = None;
    let mut rest = trimmed;
    loop {
        let head = rest.trim_start();
        let Some(after) = head.strip_prefix("--max-turns") else {
            break;
        };
        // Only treat this as the flag when `--max-turns` is a whole token
        // (followed by `=`, whitespace, or end) -- otherwise it's the start
        // of the objective and we leave it alone.
        if !(after.is_empty() || after.starts_with('=') || after.starts_with(char::is_whitespace)) {
            break;
        }
        let after = after.trim_start_matches('=').trim_start();
        let mut parts = after.splitn(2, char::is_whitespace);
        let raw = parts.next().unwrap_or("");
        let remainder = parts.next().unwrap_or("");
        let n = raw.parse::<u32>().map_err(|_| {
            format!(
                "Invalid `--max-turns` value `{raw}`. Pick an integer between \
                 {GOAL_MIN_MAX_TURNS} and {GOAL_MAX_MAX_TURNS}."
            )
        })?;
        if !(GOAL_MIN_MAX_TURNS..=GOAL_MAX_MAX_TURNS).contains(&n) {
            return Err(format!(
                "`--max-turns` {n} is out of range. Pick a value between \
                 {GOAL_MIN_MAX_TURNS} and {GOAL_MAX_MAX_TURNS}."
            ));
        }
        max_turns = Some(n);
        rest = remainder;
    }

    let objective = rest.trim().to_string();
    if objective.is_empty() {
        return Err(goal_usage());
    }

    Ok(GoalSpec {
        objective,
        max_turns,
    })
}

/// Strip surrounding markdown emphasis, quoting, and trailing punctuation
/// from a single line so a sentinel the model lightly decorated (e.g.
/// `` `GOAL_COMPLETE` `` or `**GOAL_COMPLETE.**`) still matches.
fn normalize_sentinel_line(raw: &str) -> String {
    raw.trim()
        .trim_matches(|c: char| {
            matches!(
                c,
                '`' | '*' | '_' | '#' | '"' | '\'' | '.' | '!' | ':' | ' '
            )
        })
        .trim()
        .to_string()
}

/// Inspect a goal turn's assistant text for a stop signal.
///
/// A standalone line equal to [`GOAL_COMPLETE_SENTINEL`] wins outright. A
/// standalone line starting with [`GOAL_BLOCKED_SENTINEL`] reports a
/// blocker (the trailing text, if any, is the reason). Anything else means
/// keep going. Only whole-line matches count, so the agent can discuss the
/// sentinels in prose without accidentally tripping a stop.
fn detect_goal_signal(response: &str) -> GoalSignal {
    let mut blocked: Option<String> = None;
    for raw in response.lines() {
        let line = normalize_sentinel_line(raw);
        if line == GOAL_COMPLETE_SENTINEL {
            return GoalSignal::Complete;
        }
        if let Some(reason) = line.strip_prefix(GOAL_BLOCKED_SENTINEL) {
            let reason = reason.trim_start_matches([':', '-', ' ']).trim();
            blocked = Some(reason.to_string());
        }
    }
    match blocked {
        Some(reason) => GoalSignal::Blocked(reason),
        None => GoalSignal::Continue,
    }
}

/// Why a goal loop stopped. Carried out of [`decide_after_goal_turn`] so the
/// caller can render the right user-facing message.
#[derive(Debug, PartialEq, Eq)]
enum GoalStop {
    /// The model verified the objective and emitted the completion sentinel.
    Completed,
    /// The model reported a blocker on [`GOAL_BLOCKED_THRESHOLD`] consecutive
    /// turns (reasons need not match); carries the latest reason.
    Blocked(String),
    /// The user's opt-in `--max-turns` ceiling was reached without a
    /// completion signal. Never produced for an unbounded goal.
    CeilingReached,
}

/// What the goal loop should do after a turn completes.
#[derive(Debug, PartialEq, Eq)]
enum GoalStep {
    /// Run another turn. `consecutive_blocked` is the updated counter to
    /// carry forward (non-zero means the just-finished turn reported a
    /// blocker that has not yet hit the threshold).
    Continue { consecutive_blocked: u32 },
    /// Stop the loop with this disposition.
    Stop(GoalStop),
}

/// Pure control logic for the goal loop: given the signal parsed from a
/// turn and the loop's counters, decide whether to continue or stop.
///
/// Cancellation and terminal errors are handled by the caller; they
/// pre-empt this decision. Kept side-effect-free (no `cx`, no LLM) so the
/// branching -- completion-wins-over-ceiling, the consecutive-blocked
/// threshold and its reset, and the optional ceiling -- is unit-testable
/// and the runtime loop and tests share one source of truth.
fn decide_after_goal_turn(
    signal: GoalSignal,
    turn: u32,
    max_turns: Option<u32>,
    consecutive_blocked: u32,
) -> GoalStep {
    let ceiling_reached = max_turns.is_some_and(|max| turn >= max);
    match signal {
        // A verified completion wins even on the final allowed turn.
        GoalSignal::Complete => GoalStep::Stop(GoalStop::Completed),
        GoalSignal::Blocked(reason) => {
            let blocked = consecutive_blocked + 1;
            if blocked >= GOAL_BLOCKED_THRESHOLD {
                GoalStep::Stop(GoalStop::Blocked(reason))
            } else if ceiling_reached {
                GoalStep::Stop(GoalStop::CeilingReached)
            } else {
                GoalStep::Continue {
                    consecutive_blocked: blocked,
                }
            }
        }
        GoalSignal::Continue => {
            if ceiling_reached {
                GoalStep::Stop(GoalStop::CeilingReached)
            } else {
                GoalStep::Continue {
                    consecutive_blocked: 0,
                }
            }
        }
    }
}

/// What the goal loop should do about a turn that ended in an LLM failure
/// (vs. a real model response). Kept side-effect-free so the
/// transient-vs-fatal branch is unit-testable, like [`decide_after_goal_turn`].
#[derive(Debug, PartialEq, Eq)]
enum GoalFailureAction {
    /// Transient outage (server overload, rate limit, stream/connection drop):
    /// wait a backoff scaled by `consecutive_failures` and retry the turn.
    /// Unbounded by design so the goal survives a long outage and resumes when
    /// it clears -- the delay is capped instead of the retry count.
    Backoff { consecutive_failures: u32 },
    /// Fatal (auth, invalid request, panic): retrying would not help, so stop
    /// the goal and hand back to the user.
    Stop,
}

/// Classify a failed goal turn. Transient failures back off and retry; fatal
/// ones stop. Mirrors Codex's retryable/fatal split (the underlying predicate
/// is [`crate::llm_client::is_retryable_llm_error`], applied in
/// [`crate::tool_loop::run`]); the divergence is deliberate -- Codex bounds
/// transient retries by a fixed count then aborts the turn, whereas an
/// unbounded goal keeps retrying (with a capped delay) to survive the outage.
fn decide_after_goal_failure(
    failure: &crate::tool_loop::TurnFailure,
    consecutive_failures: u32,
) -> GoalFailureAction {
    if failure.retryable {
        GoalFailureAction::Backoff {
            consecutive_failures: consecutive_failures.saturating_add(1),
        }
    } else {
        GoalFailureAction::Stop
    }
}

/// Upper bound on the inter-turn backoff for a goal surviving an outage. The
/// base schedule is the codex-compatible [`crate::http_retry::retry_backoff`]
/// (200ms * 2^(n-1) + jitter); because a goal retries an unbounded number of
/// times, the delay is capped here so a long outage settles into a steady
/// ~1-minute poll rather than growing without limit.
const GOAL_FAILURE_BACKOFF_CAP: Duration = Duration::from_secs(60);

/// Capped exponential backoff for the `consecutive_failures`-th transient
/// failure in a row (1-based).
fn goal_failure_backoff(consecutive_failures: u32) -> Duration {
    crate::http_retry::retry_backoff(u64::from(consecutive_failures)).min(GOAL_FAILURE_BACKOFF_CAP)
}

/// Build the continuation prompt injected as the user message for one goal
/// turn. Adapts Codex's `continuation.md` (objective framing + completion
/// audit + blocked discipline) to Anvil's sentinel-based stop signal.
fn build_goal_prompt(
    objective: &str,
    turn: u32,
    max_turns: Option<u32>,
    phase: GoalPhase,
) -> String {
    let header = match max_turns {
        Some(max) => {
            format!("You are operating in autonomous goal mode (turn {turn} of at most {max}).")
        }
        None => format!("You are operating in autonomous goal mode (turn {turn})."),
    };
    let objective_block = format!("<objective>\n{}\n</objective>", objective.trim());

    let completion_protocol = format!(
        "Completion protocol:\n\
         - Treat completion as unproven. Before claiming success, derive concrete \
         requirements from the objective and verify each against the ACTUAL current state \
         of the worktree and any commands/tests it implies -- inspect file contents, command \
         output, and test results rather than relying on intent, memory, or a plausible answer.\n\
         - Keep the full objective intact; do not redefine success around a smaller or easier \
         task just to finish.\n\
         - Only when every requirement is satisfied and verified, end your message with a line \
         containing exactly:\n\
         {GOAL_COMPLETE_SENTINEL}\n\
         Put it alone on the final line, with no surrounding text, quotes, or formatting. \
         Emitting it is a claim that the full objective is done and can withstand \
         requirement-by-requirement scrutiny. If any requirement is missing, weak, indirect, \
         or unverified, do NOT emit it -- keep working."
    );

    let blocked_protocol = format!(
        "If you are genuinely at an impasse and cannot make progress without user input or an \
         external change, end your message with a line:\n\
         {GOAL_BLOCKED_SENTINEL}: <one-line reason>\n\
         Use this only when truly stuck -- never because the work is merely hard, slow, or \
         incomplete. If the same blocker persists for {GOAL_BLOCKED_THRESHOLD} consecutive \
         turns, the goal stops and hands back to the user."
    );

    match phase {
        GoalPhase::Continue => format!(
            "{header}\n\n\
             Continue working toward the objective below. This goal persists across turns, so \
             you do not need to shrink it to what fits in one turn -- make concrete, verifiable \
             progress toward the real end state.\n\n\
             {objective_block}\n\n\
             Work from evidence: treat the current worktree and command output as authoritative \
             before relying on earlier conversation. Use your tools to actually make the changes \
             -- do not just describe them. If the next work is meaningfully multi-step, keep a \
             short task list.\n\n\
             {completion_protocol}\n\n\
             {blocked_protocol}"
        ),
        GoalPhase::FinalWrapUp => format!(
            "{header}\n\n\
             This is the FINAL turn of the goal's opt-in turn ceiling. Do not start new large work. \
             Bring the current work to a safe, coherent stopping point, then summarize what was \
             accomplished, what remains, and the clear next step for the user.\n\n\
             {objective_block}\n\n\
             If -- and only if -- the objective is actually complete and verified, end with a \
             line containing exactly {GOAL_COMPLETE_SENTINEL}. Otherwise do not emit it; just \
             summarize."
        ),
    }
}

/// Parse the optional title from `/pr-create [title]`. Whitespace-only
/// arguments collapse to `None` so `gh pr create --fill` derives the title
/// from commit messages instead.
fn parse_pr_create_arg(prompt_text: &str) -> Option<String> {
    let trimmed = slash_command_args(prompt_text);
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

/// Quote a string for `sh -c` by wrapping in single quotes and
/// escaping any embedded single quote via the standard `'\''` trick.
/// `run_shell_command` invokes `sh -c` with a single argv element, so
/// command parts that come from user input (PR title) or external
/// lookups (default branch name) need shell-safe quoting.
fn shell_single_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Per-shell-call timeout for slash-command-driven `run_shell_command`
/// invocations. Generous enough for `gh pr create` over a slow link
/// without leaving a stuck child for minutes.
const HANDLER_SHELL_TIMEOUT_SECS: u64 = 60;

/// Run `cmd` via `run_shell_command` on the per-session `ToolRegistry`
/// and return its stdout/stderr blob on success, or a pre-formatted
/// `Error: ...` string on failure. `label` is the short command name
/// shown in the error message.
async fn run_or_report(
    registry: &crate::tools::ToolRegistry,
    cmd: &str,
    label: &str,
    policy: crate::tools::sandbox::SandboxPolicy,
) -> Result<String, String> {
    let result = registry
        .execute(
            "run_shell_command",
            serde_json::json!({ "command": cmd, "timeout": HANDLER_SHELL_TIMEOUT_SECS * 1000 }),
            policy,
        )
        .await;
    if matches!(result.status, crate::tools::ToolStatus::Success) {
        Ok(result.output)
    } else {
        Err(format!("Error: `{label}` failed.\n\n{}", result.output))
    }
}

/// Handle the `/pr-create` slash command. Creates a GitHub pull request
/// from the current branch by shelling out to `gh pr create`.
///
/// Flow (each step short-circuits with a user-facing error on failure):
///   1. Refuse on `PermissionMode::ReadOnly` -- git push won't be allowed
///      under the resulting sandbox tier.
///   2. Refuse if `git status --porcelain` is non-empty so we never push
///      with uncommitted state.
///   3. Refuse if the branch has no upstream and instruct the user to
///      push manually. We deliberately do NOT auto-push: the choice of
///      which remote to push to is meaningful in fork-based workflows
///      (`origin` may be the user's personal fork OR the upstream repo)
///      and a server-side handler should not make that call silently.
///   4. Detect the repository's default branch via `gh repo view` and
///      pass it explicitly to `--base`.
///   5. Invoke `gh pr create --base <default> --fill [--title <user-arg>]`
///      and surface the resulting PR URL.
///
/// All shell calls go through `ToolRegistry::execute("run_shell_command")`
/// so they share the LLM tool path's env scrubbing, sandbox policy,
/// rlimits, and output truncation. The user typed `/pr-create`, so the
/// `consult_gate` step the LLM path requires is unnecessary -- the
/// slash command itself is the user's consent.
///
/// Notes:
///   - `gh` falls back to `~/.config/gh/hosts.yml` for auth; `GH_TOKEN`
///     and `GITHUB_TOKEN` are scrubbed from the child env, so users who
///     rely on env-var auth must `gh auth login` first.
async fn handle_pr_create(
    prompt_text: &str,
    registry: &crate::tools::ToolRegistry,
    permission_mode: PermissionMode,
    sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
) -> String {
    if matches!(permission_mode, PermissionMode::ReadOnly) {
        return "Error: `/pr-create` is disabled in read-only permission mode. \
                Change the session Permission selector to a non-read-only mode to create PRs."
            .to_string();
    }

    let policy = crate::tools::sandbox::SandboxPolicy::resolve(permission_mode, sandbox_mode);

    let status = match run_or_report(
        registry,
        "git status --porcelain",
        "git status --porcelain",
        policy,
    )
    .await
    {
        Ok(o) => o,
        Err(e) => return e,
    };
    let dirty = status.trim();
    if !dirty.is_empty() {
        return format!(
            "Error: working tree is dirty. Commit or stash these paths before \
             running `/pr-create`:\n\n{dirty}"
        );
    }

    // No-upstream check. Failure of `git rev-parse @{u}` is the trigger
    // for the "no upstream" branch -- it can also fire for unrelated
    // git errors (detached HEAD, corrupt refs), but the user-facing
    // remediation is the same: push manually and re-run.
    let upstream = registry
        .execute(
            "run_shell_command",
            serde_json::json!({
                "command": "git rev-parse --abbrev-ref --symbolic-full-name @{u}",
                "timeout": HANDLER_SHELL_TIMEOUT_SECS * 1000,
            }),
            policy,
        )
        .await;
    if !matches!(upstream.status, crate::tools::ToolStatus::Success) {
        let remotes = run_or_report(registry, "git remote -v", "git remote -v", policy)
            .await
            .unwrap_or_else(|e| e);
        return format!(
            "Error: this branch has no upstream. Push it manually and re-run \
             `/pr-create` -- the choice of remote is yours, not the server's.\n\n\
             Try: `git push -u <remote> HEAD`\n\n\
             Detected remotes:\n{remotes}"
        );
    }

    let base = match run_or_report(
        registry,
        "gh repo view --json defaultBranchRef --jq .defaultBranchRef.name",
        "gh repo view",
        policy,
    )
    .await
    {
        Ok(o) => o,
        Err(e) => {
            return format!("{e}\n\nIs `gh` installed and authenticated (`gh auth login`)?");
        }
    };
    let base_branch = base.trim();
    if base_branch.is_empty() {
        return "Error: `gh repo view` returned an empty default branch name.".to_string();
    }

    let title_arg = match parse_pr_create_arg(prompt_text) {
        Some(t) => format!(" --title {}", shell_single_quote(&t)),
        None => String::new(),
    };
    let cmd = format!(
        "gh pr create --base {} --fill{title_arg}",
        shell_single_quote(base_branch)
    );
    match run_or_report(registry, &cmd, "gh pr create", policy).await {
        Ok(output) => {
            // `gh pr create` prints the PR URL on stdout. Surface it
            // prominently; combined output may also contain a "Creating
            // pull request..." line on stderr that we keep below.
            let url = output
                .lines()
                .map(str::trim)
                .find(|l| l.starts_with("https://") && l.contains("/pull/"))
                .unwrap_or("");
            if url.is_empty() {
                format!(
                    "Pull request created against `{base_branch}`, but the URL \
                     could not be parsed from `gh`'s output. Raw output:\n\n{output}"
                )
            } else {
                format!("Pull request created against `{base_branch}`:\n\n{url}")
            }
        }
        Err(e) => e,
    }
}

/// Run the `/compress` slash command: summarize every uncompressed
/// turn in the session, one at a time, persisting each summary
/// through `set_turn_summary` so a reload reproduces the same state.
///
/// Mirrors Brokk's user-triggered "Compress History" UI button
/// (`ContextManager.compressHistoryAsync(Context)`), with two
/// deliberate differences:
///
/// 1. Sequential rather than parallel. Anvil's tool loop already runs
///    one prompt at a time per session (gated by `start_prompt`), so
///    fanning out N parallel LLM calls here only adds rate-limit
///    pressure without saving meaningful wall time -- the user is
///    waiting on this command interactively.
/// 2. Errors are non-fatal per turn. If summarizing turn 3 errors,
///    turns 1, 2, 4... still get summarized; the failed turn just
///    stays verbatim. Mirrors `ContextManager.compressHistory`
///    returning the original on failure.
///
/// Streams per-turn progress notifications via `send_message` so the
/// user sees what's happening on long sessions, and finishes with a
/// summary tally.
/// Indexes of turns that need compressing on a `/compress` run.
/// Extracted as a pure helper so the planning logic is unit-testable
/// without standing up a mock `ConnectionTo<Client>`.
struct CompressPlan {
    total: usize,
    uncompressed: Vec<usize>,
}

fn plan_compress(snap: &SessionSnapshot) -> CompressPlan {
    CompressPlan {
        total: snap.history.len(),
        uncompressed: snap
            .history
            .iter()
            .enumerate()
            .filter_map(|(i, t)| t.summary.is_none().then_some(i))
            .collect(),
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_compress(
    snap: &SessionSnapshot,
    llm: &dyn crate::llm_client::LlmBackend,
    sessions: &SessionStore,
    session_id: &str,
    cancel: tokio_util::sync::CancellationToken,
    idle_timeout: Duration,
    context_length: Option<u32>,
    cx: &ConnectionTo<Client>,
) -> String {
    let plan = plan_compress(snap);
    let total_turns = plan.total;
    let uncompressed = plan.uncompressed;
    if uncompressed.is_empty() {
        return format!(
            "Nothing to compress: {total_turns} turn(s) in history, all already summarized."
        );
    }

    send_message(
        cx,
        session_id,
        &format!(
            "Compressing {} of {} turn(s)...\n",
            uncompressed.len(),
            total_turns
        ),
    );

    // Track aggregate token impact for the final report. Per-turn we
    // measure verbatim cost (what the next prompt would have charged)
    // vs. the produced summary's cost (what it'll charge after
    // compression).
    let mut verbatim_tokens_total = 0usize;
    let mut summary_tokens_total = 0usize;
    let mut succeeded = 0usize;
    let mut failed: Vec<(usize, String)> = Vec::new();

    for idx in uncompressed.iter().copied() {
        if cancel.is_cancelled() {
            send_message(cx, session_id, "Cancelled.\n");
            break;
        }
        let turn = snap.history[idx].clone();
        let display_idx = idx + 1;
        let verbatim_cost = approximate_turn_tokens(&turn);
        match crate::context_manager::summarize_turn(
            llm,
            &snap.model,
            &turn,
            context_length,
            idle_timeout,
            cancel.clone(),
        )
        .await
        {
            Ok(summary) => {
                let summary_cost = crate::tokens::approximate_tokens(&summary);
                match sessions.set_turn_summary(session_id, idx, summary).await {
                    Ok(true) => {
                        succeeded += 1;
                        verbatim_tokens_total += verbatim_cost;
                        summary_tokens_total += summary_cost;
                        send_message(
                            cx,
                            session_id,
                            &format!(
                                "- Turn {display_idx}: compressed (~{verbatim_cost} -> ~{summary_cost} tokens)\n"
                            ),
                        );
                    }
                    Ok(false) => {
                        // Setter refused (unknown session, out-of-range
                        // index, or missing fragment_id). Treat as a
                        // soft failure and keep going.
                        failed.push((
                            display_idx,
                            "setter refused (turn not persisted?)".to_string(),
                        ));
                        send_message(
                            cx,
                            session_id,
                            &format!("- Turn {display_idx}: persist refused -- skipped\n"),
                        );
                    }
                    Err(e) => {
                        failed.push((display_idx, format!("persist failed: {e}")));
                        send_message(
                            cx,
                            session_id,
                            &format!("- Turn {display_idx}: persist failed -- {e}\n"),
                        );
                    }
                }
            }
            Err(e) => {
                failed.push((display_idx, e.to_string()));
                send_message(
                    cx,
                    session_id,
                    &format!("- Turn {display_idx}: summarization failed -- {e}\n"),
                );
            }
        }
    }

    let mut out = String::new();
    out.push_str("\n**Done.**\n\n");
    out.push_str(&format!(
        "- Compressed {succeeded}/{} turn(s).\n",
        uncompressed.len()
    ));
    if succeeded > 0 {
        let saved = verbatim_tokens_total.saturating_sub(summary_tokens_total);
        out.push_str(&format!(
            "- Approx tokens: {verbatim_tokens_total} (verbatim) -> {summary_tokens_total} (summary). \
             Saved ~{saved}.\n"
        ));
    }
    if !failed.is_empty() {
        out.push_str(&format!("- Failed {}: \n", failed.len()));
        for (turn, msg) in &failed {
            out.push_str(&format!("  - Turn {turn}: {msg}\n"));
        }
    }
    out
}

/// Approximate per-turn token cost when replayed verbatim (user
/// prompt + assistant response + tool exchanges). Used by
/// `handle_compress` to report "before vs. after" savings.
fn approximate_turn_tokens(turn: &crate::session::ConversationTurn) -> usize {
    let mut sum = crate::tokens::approximate_tokens(&turn.user_prompt);
    sum += crate::tokens::approximate_tokens(&turn.agent_response);
    for exchange in &turn.tool_exchanges {
        sum += crate::tokens::approximate_tokens(&exchange.tool_name);
        sum += crate::tokens::approximate_tokens(&exchange.arguments);
        sum += crate::tokens::approximate_tokens(&exchange.result);
    }
    sum
}

/// Render the `/context` snapshot. Mirrors the Java executor's report at a
/// coarser granularity -- the Rust agent does not yet model
/// editable/readonly/virtual fragments, so the table reports the
/// conversation history instead, which is what actually drives token
/// pressure on the LLM today.
fn render_context_report(
    snap: &crate::session::SessionSnapshot,
    permission_mode: PermissionMode,
    available_models: &[crate::llm_client::ModelMetadata],
) -> String {
    // Sum tokens via the o200k_base encoder so this report matches the
    // numbers the compression layer will see at the threshold. Tool
    // exchanges count too -- they round-trip back to the LLM on every
    // replay via build_prompt_messages, so omitting them would
    // understate real pressure on long sessions.
    let mut user_tokens = 0usize;
    let mut agent_tokens = 0usize;
    let mut tool_tokens = 0usize;
    for turn in &snap.history {
        user_tokens += crate::tokens::approximate_tokens(&turn.user_prompt);
        agent_tokens += crate::tokens::approximate_tokens(&turn.agent_response);
        for exchange in &turn.tool_exchanges {
            tool_tokens += crate::tokens::approximate_tokens(&exchange.tool_name);
            tool_tokens += crate::tokens::approximate_tokens(&exchange.arguments);
            tool_tokens += crate::tokens::approximate_tokens(&exchange.result);
        }
    }
    let total_tokens = user_tokens + agent_tokens + tool_tokens;
    let model_display = if snap.model.is_empty() {
        "(none)".to_string()
    } else {
        snap.model.clone()
    };
    let catalog_size = available_models.len();
    let context_length = available_models
        .iter()
        .find(|m| m.id == snap.model)
        .and_then(|m| m.context_length);

    let mut out = String::new();
    out.push_str("**Session context**\n\n");
    out.push_str(&format!("- Working directory: `{}`\n", snap.cwd.display()));
    out.push_str(&format!("- Mode: `{}`\n", snap.mode.as_str()));
    out.push_str(&format!(
        "- Permission mode: `{}`\n",
        permission_mode.as_str()
    ));
    out.push_str(&format!(
        "- Model: `{model_display}` ({catalog_size} known in catalog)\n"
    ));
    if let Some(ctx) = context_length {
        let pct = if ctx > 0 {
            (total_tokens as f64 / ctx as f64 * 100.0).round() as u32
        } else {
            0
        };
        out.push_str(&format!(
            "- Context window: {total_tokens} / {ctx} tokens (~{pct}% used)\n"
        ));
    } else {
        out.push_str(&format!(
            "- Context window: {total_tokens} tokens used (model max unknown)\n"
        ));
    }
    out.push_str(&format!(
        "- Conversation turns: {} (~{} user / ~{} agent / ~{} tool exchanges)\n",
        snap.history.len(),
        user_tokens,
        agent_tokens,
        tool_tokens
    ));
    out
}

/// Outcome of the OpenRouter `/credits` lookup performed by `/usage`.
/// Modelled as a 3-state enum (rather than `Result<Option<...>>`) so
/// `render_usage_report` can render a distinct line for "no credential
/// found" vs. "credential found but the upstream call failed" without
/// the call site re-shaping a nested type.
#[derive(Debug, Clone)]
enum OpenRouterCreditsOutcome {
    /// No credential resolved (no env var, no on-disk file). Either the
    /// user hasn't logged in, or the active model isn't an OpenRouter
    /// one and we deliberately skipped the lookup.
    Skipped,
    Fetched(crate::openrouter_credits::Credits),
    Failed(String),
}

/// Outcome of the ChatGPT `wham/usage` lookup performed by `/usage`.
/// Carries more skip reasons than the OpenRouter sibling because Codex
/// has two distinct "logged in but the endpoint doesn't apply to you"
/// states: api-key mode (billing through OPENAI_API_KEY) vs no auth at
/// all. Conflating them would tell an api-key user to "run /setup
/// codex" when their setup is actually fine.
#[derive(Debug, Clone)]
enum CodexCreditsOutcome {
    /// Active model isn't a Codex one; no Codex line in the report.
    NotApplicable,
    /// On a Codex model but `~/.codex/auth.json` is missing.
    NoAuth,
    /// On a Codex model and authenticated, but with an api-key billing
    /// path that the subscription `wham/usage` endpoint doesn't cover.
    ApiKeyMode,
    Fetched(crate::codex_credits::CodexUsage),
    Failed(String),
}

/// Hard wall-clock ceiling for each `/usage` credits fetch. The inner
/// reqwest client already has a 5s request budget, but `codex_credits`
/// runs `refresh_if_stale` first against a *different* client that has
/// no timeout configured -- without an outer cancel the slash command
/// can hang on a stuck token refresh for as long as the OS waits on a
/// dead TCP connection. Sized to match the inner request budget so a
/// clean reqwest error normally surfaces first; the outer timeout only
/// fires when something upstream of the credits request itself stalls.
const USAGE_CREDITS_FETCH_TIMEOUT: Duration = Duration::from_secs(5);

/// Run the `/credits` lookup only when it makes sense:
/// - The active model is an OpenRouter one (`openrouter::<id>`), so the
///   balance is relevant to the bill the user is racking up right now;
///   AND
/// - A credential is resolvable via env or on-disk file (same
///   precedence as `build_openrouter_backend`).
///
/// Returns `Skipped` for any other configuration so the report stays
/// quiet on non-OpenRouter sessions instead of dragging in an
/// irrelevant network call.
async fn fetch_openrouter_credits_for_usage(model_wire_id: &str) -> OpenRouterCreditsOutcome {
    let active_source = split_wire_id(model_wire_id).map(|(src, _)| src);
    if active_source != Some(ModelSource::OpenRouter) {
        return OpenRouterCreditsOutcome::Skipped;
    }
    let Some(key) = crate::openrouter_credits::active_api_key() else {
        return OpenRouterCreditsOutcome::Skipped;
    };
    match tokio::time::timeout(
        USAGE_CREDITS_FETCH_TIMEOUT,
        crate::openrouter_credits::fetch(&key),
    )
    .await
    {
        Ok(Ok(credits)) => OpenRouterCreditsOutcome::Fetched(credits),
        Ok(Err(e)) => OpenRouterCreditsOutcome::Failed(format!("{e:#}")),
        Err(_) => OpenRouterCreditsOutcome::Failed(format!(
            "timed out after {}s",
            USAGE_CREDITS_FETCH_TIMEOUT.as_secs()
        )),
    }
}

/// Run the Codex `wham/usage` lookup only when:
/// - The active model is a Codex one (`codex::<id>`), so the credits
///   are relevant to the bill the user is racking up right now; AND
/// - The on-disk `auth.json` is in `chatgpt` mode (api-key mode means
///   spend goes through the OpenAI billing dashboard, not the ChatGPT
///   subscription credits this endpoint reports).
///
/// Inspects `auth_status` before the network call so the report can
/// distinguish "no auth at all" from "api-key billing" -- the previous
/// implementation conflated them and told api-key users to re-run
/// `/setup codex` when their setup was already correct.
async fn fetch_codex_credits_for_usage(model_wire_id: &str) -> CodexCreditsOutcome {
    let active_source = split_wire_id(model_wire_id).map(|(src, _)| src);
    if active_source != Some(ModelSource::Codex) {
        return CodexCreditsOutcome::NotApplicable;
    }
    match crate::codex_credits::auth_status() {
        Ok(crate::codex_credits::AuthStatus::Missing) => return CodexCreditsOutcome::NoAuth,
        Ok(crate::codex_credits::AuthStatus::ApiKeyMode) => {
            return CodexCreditsOutcome::ApiKeyMode;
        }
        Ok(crate::codex_credits::AuthStatus::ChatGptMode) => {}
        Err(e) => return CodexCreditsOutcome::Failed(format!("{e:#}")),
    }
    match tokio::time::timeout(USAGE_CREDITS_FETCH_TIMEOUT, crate::codex_credits::fetch()).await {
        Ok(Ok(Some(usage))) => CodexCreditsOutcome::Fetched(usage),
        // `fetch` only returns Ok(None) for the same conditions
        // `auth_status` already classified, but auth.json could have
        // been deleted between the two calls. Treat as no-auth in
        // that race.
        Ok(Ok(None)) => CodexCreditsOutcome::NoAuth,
        Ok(Err(e)) => CodexCreditsOutcome::Failed(format!("{e:#}")),
        Err(_) => CodexCreditsOutcome::Failed(format!(
            "timed out after {}s",
            USAGE_CREDITS_FETCH_TIMEOUT.as_secs()
        )),
    }
}

/// Capitalize a plan slug for human display. The wire form is
/// snake_case (`"free_workspace"`, `"self_serve_business_usage_based"`)
/// which reads poorly inline; we replace underscores with spaces and
/// title-case each word. Unknown slugs pass through unchanged because
/// the server may add new plan tokens we haven't seen.
fn humanize_plan_type(plan: &str) -> String {
    plan.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().chain(chars).collect::<String>(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Render `reset_after_seconds` as a short human-friendly string.
/// Returns `"soon"` for sub-minute values to avoid a noisy "0m" line.
fn format_reset_after(seconds: i32) -> String {
    if seconds <= 60 {
        return "soon".to_string();
    }
    let total = seconds as u64;
    let days = total / 86_400;
    let hours = (total % 86_400) / 3_600;
    let minutes = (total % 3_600) / 60;
    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {minutes}m")
    } else {
        format!("{minutes}m")
    }
}

/// `/usage` body: session token totals + USD cost + (for OpenRouter
/// sessions) the live credit balance. Kept pure so it's unit-testable
/// without standing up a session store or a mock HTTP server -- the
/// network call sits in `fetch_openrouter_credits_for_usage` above and
/// is invoked by the slash dispatch sites.
fn render_usage_report(
    snap: &crate::session::SessionSnapshot,
    usage: crate::llm_client::TokenUsage,
    cost_usd: Option<f64>,
    openrouter_credits: OpenRouterCreditsOutcome,
    codex_credits: CodexCreditsOutcome,
) -> String {
    let (model_display, source_display) = match split_wire_id(&snap.model) {
        Some((source, bare)) => (bare.to_string(), source.as_str().to_string()),
        None if snap.model.is_empty() => ("(none)".to_string(), "(unset)".to_string()),
        None => (snap.model.clone(), "(unknown)".to_string()),
    };

    let mut out = String::new();
    out.push_str("**Session usage**\n\n");
    out.push_str(&format!("- Model: `{model_display}` ({source_display})\n"));
    if usage.is_zero() {
        out.push_str("- Tokens: no LLM turns recorded yet this session\n");
    } else {
        out.push_str(&format!(
            "- Tokens: {} total (input {}, output {}, reasoning {}, cached read {}, cached write {})\n",
            usage.total_tokens(),
            usage.input_tokens,
            usage.output_tokens,
            usage.thought_tokens,
            usage.cached_read_tokens,
            usage.cached_write_tokens,
        ));
    }
    match cost_usd {
        Some(amount) => out.push_str(&format!("- Session cost: ${amount:.4} USD\n")),
        None if usage.is_zero() => {
            out.push_str("- Session cost: $0.0000 USD (no billable turns yet)\n")
        }
        None => out
            .push_str("- Session cost: unavailable (at least one turn lacked pricing metadata)\n"),
    }

    match openrouter_credits {
        OpenRouterCreditsOutcome::Fetched(credits) => {
            out.push_str(&format!(
                "- OpenRouter balance: ${:.4} remaining (${:.4} purchased − ${:.4} used)\n",
                credits.balance(),
                credits.total_credits,
                credits.total_usage,
            ));
        }
        OpenRouterCreditsOutcome::Failed(msg) => {
            out.push_str(&format!("- OpenRouter balance: lookup failed ({msg})\n"));
        }
        OpenRouterCreditsOutcome::Skipped => {
            // Two distinct skip reasons, distinguished by inspecting
            // the active model id. We keep the message short and
            // actionable rather than dumping internal state.
            if split_wire_id(&snap.model).map(|(src, _)| src) == Some(ModelSource::OpenRouter) {
                out.push_str(
                    "- OpenRouter balance: no credential configured. Set \
                     OPENROUTER_API_KEY or run `/setup openrouter key <key>`.\n",
                );
            } else {
                out.push_str(
                    "- OpenRouter balance: not applicable (active model is not OpenRouter)\n",
                );
            }
        }
    }

    match codex_credits {
        CodexCreditsOutcome::Fetched(usage) => {
            let plan = usage
                .plan_type
                .as_deref()
                .filter(|s| !s.is_empty())
                .map(humanize_plan_type)
                .unwrap_or_else(|| "Unknown".to_string());
            out.push_str(&format!("- ChatGPT plan: {plan}\n"));
            match usage.credits {
                Some(c) if c.unlimited => {
                    out.push_str("- Codex credits: unlimited on this plan\n");
                }
                Some(c) if c.has_credits => match c.balance.as_deref() {
                    Some(balance) if !balance.is_empty() => {
                        out.push_str(&format!("- Codex credits: ${balance} remaining\n"));
                    }
                    _ => out.push_str("- Codex credits: available\n"),
                },
                Some(_) => {
                    out.push_str(
                        "- Codex credits: depleted (subscription rate-limit still applies)\n",
                    );
                }
                None => {
                    // No credits block -- typical for Pro/Team plans
                    // where access is gated by rate-limits rather than
                    // metered credits. Silent here; the rate-limit
                    // line below carries the actionable info.
                }
            }
            if let Some(window) = usage.rate_limit.and_then(|r| r.primary_window) {
                let reset = format_reset_after(window.reset_after_seconds);
                out.push_str(&format!(
                    "- Codex rate limit: {}% of primary window used (resets in {reset})\n",
                    window.used_percent
                ));
            }
        }
        CodexCreditsOutcome::Failed(msg) => {
            out.push_str(&format!(
                "- Codex (ChatGPT) status: lookup failed ({msg})\n"
            ));
        }
        CodexCreditsOutcome::NoAuth => {
            out.push_str(
                "- Codex (ChatGPT) status: no ChatGPT credentials. Run `/setup codex` \
                 to authenticate, or set `OPENAI_API_KEY` to bill against the API.\n",
            );
        }
        CodexCreditsOutcome::ApiKeyMode => {
            // The wham/usage endpoint only reports ChatGPT-subscription
            // state. Calls in api-key mode bill the OpenAI account
            // dashboard instead, so point the user there rather than
            // surfacing an irrelevant "balance: unknown" line.
            out.push_str(
                "- Codex billing: OPENAI_API_KEY (api-key mode); subscription credit \
                 balance not applicable. See platform.openai.com/usage for spend.\n",
            );
        }
        CodexCreditsOutcome::NotApplicable => {
            // Stay quiet on non-Codex sessions so the `/usage` report
            // doesn't grow a line per unconfigured provider.
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn negotiate_protocol_version_accepts_supported_version() {
        assert_eq!(
            negotiate_protocol_version(ProtocolVersion::V1),
            ProtocolVersion::V1
        );
    }

    #[test]
    fn negotiate_protocol_version_downgrades_future_version() {
        assert_eq!(
            negotiate_protocol_version(ProtocolVersion::from(2_u16)),
            ProtocolVersion::V1
        );
    }

    #[test]
    fn is_slash_command_matches_bare_and_with_args() {
        assert!(is_slash_command("/context", "context"));
        assert!(is_slash_command("  /context  ", "context"));
        assert!(is_slash_command("/context with extra args", "context"));
        // Case-insensitive: clients sometimes uppercase auto-complete entries.
        assert!(is_slash_command("/Context", "context"));
        assert!(is_slash_command("/CONTEXT", "context"));
    }

    #[test]
    fn parse_idle_timeout_arg_routes_to_show_when_bare() {
        assert_eq!(
            parse_idle_timeout_arg("/idle-timeout"),
            Ok(IdleTimeoutAction::Show)
        );
        assert_eq!(
            parse_idle_timeout_arg("  /idle-timeout  "),
            Ok(IdleTimeoutAction::Show)
        );
    }

    #[test]
    fn parse_idle_timeout_arg_clears_on_default_keyword() {
        assert_eq!(
            parse_idle_timeout_arg("/idle-timeout default"),
            Ok(IdleTimeoutAction::Clear)
        );
        // Case-insensitive keyword.
        assert_eq!(
            parse_idle_timeout_arg("/idle-timeout DEFAULT"),
            Ok(IdleTimeoutAction::Clear)
        );
    }

    #[test]
    fn parse_idle_timeout_arg_accepts_numeric_in_range() {
        assert_eq!(
            parse_idle_timeout_arg("/idle-timeout 600"),
            Ok(IdleTimeoutAction::Set(600))
        );
        // Bounds inclusive.
        assert_eq!(
            parse_idle_timeout_arg("/idle-timeout 1"),
            Ok(IdleTimeoutAction::Set(1))
        );
        assert_eq!(
            parse_idle_timeout_arg("/idle-timeout 86400"),
            Ok(IdleTimeoutAction::Set(86_400))
        );
    }

    #[test]
    fn parse_idle_timeout_arg_rejects_out_of_range() {
        // 0 would mean "abort instantly" -- the lower bound is 1.
        let err = parse_idle_timeout_arg("/idle-timeout 0").expect_err("zero must reject");
        assert!(err.contains("out of range"), "got: {err}");
        // Above the 24h ceiling.
        let err = parse_idle_timeout_arg("/idle-timeout 999999").expect_err("huge must reject");
        assert!(err.contains("out of range"), "got: {err}");
    }

    #[test]
    fn parse_idle_timeout_arg_rejects_non_numeric_junk() {
        let err = parse_idle_timeout_arg("/idle-timeout banana").expect_err("junk must reject");
        assert!(err.contains("Unknown subcommand"), "got: {err}");
    }

    #[test]
    fn parse_shell_words_supports_quotes_and_escapes() {
        assert_eq!(
            parse_shell_words(
                r#"add --framing content-length win "C:\Program Files\server.exe" --arg '{"k":"v v"}'"#
            )
            .unwrap(),
            vec![
                "add",
                "--framing",
                "content-length",
                "win",
                r#"C:\Program Files\server.exe"#,
                "--arg",
                r#"{"k":"v v"}"#
            ]
        );
        assert_eq!(
            parse_shell_words(r#"add local command\ with\ spaces --flag"#).unwrap(),
            vec!["add", "local", "command with spaces", "--flag"]
        );
    }

    #[test]
    fn parse_shell_words_rejects_unclosed_quotes() {
        let err = parse_shell_words(r#"add bad "unterminated"#).expect_err("must reject");
        assert!(err.contains("Unclosed double quote"), "got: {err}");
    }

    #[test]
    fn is_slash_command_rejects_non_matches() {
        // Plain text is never a command, even if the word "context" appears.
        assert!(!is_slash_command("context please", "context"));
        // Missing leading slash.
        assert!(!is_slash_command("context", "context"));
        // Different command sharing a prefix must not match.
        assert!(!is_slash_command("/contextual", "context"));
        // Empty input.
        assert!(!is_slash_command("", "context"));
        assert!(!is_slash_command("/", "context"));
    }

    #[test]
    fn parse_pr_create_arg_returns_none_when_bare() {
        assert_eq!(parse_pr_create_arg("/pr-create"), None);
        assert_eq!(parse_pr_create_arg("  /pr-create  "), None);
        assert_eq!(parse_pr_create_arg("/pr-create   "), None);
    }

    #[test]
    fn parse_pr_create_arg_returns_title_when_present() {
        assert_eq!(
            parse_pr_create_arg("/pr-create Fix the thing"),
            Some("Fix the thing".to_string())
        );
    }

    #[test]
    fn parse_pr_create_arg_trims_surrounding_whitespace() {
        assert_eq!(
            parse_pr_create_arg("/pr-create   Fix the thing   "),
            Some("Fix the thing".to_string())
        );
    }

    #[test]
    fn parse_pr_create_arg_preserves_internal_punctuation_and_case() {
        // Conventional-commit prefixes, parens, colons and mixed case
        // must round-trip verbatim into the title.
        assert_eq!(
            parse_pr_create_arg("/pr-create feat(api): Add NewThing"),
            Some("feat(api): Add NewThing".to_string())
        );
    }

    #[test]
    fn is_slash_command_matches_pr_create_variants() {
        assert!(is_slash_command("/pr-create", "pr-create"));
        assert!(is_slash_command("  /pr-create  ", "pr-create"));
        assert!(is_slash_command("/pr-create my title", "pr-create"));
        // Case-insensitive matching, like other slash commands.
        assert!(is_slash_command("/PR-Create", "pr-create"));
        assert!(is_slash_command("/PR-CREATE", "pr-create"));
        // Hyphen-prefix collisions must not match.
        assert!(!is_slash_command("/pr-create-extra", "pr-create"));
    }

    #[test]
    fn builtin_commands_include_setup_permissions_and_pr_create() {
        // `/setup` owns model/provider configuration, `/permissions` owns
        // remembered approval management, and `/pr-create` remains an explicit workflow command.
        let cmds = builtin_commands();
        assert!(
            cmds.iter().any(|c| c.name == "setup"),
            "builtin_commands() missing setup; got: {:?}",
            cmds.iter().map(|c| &c.name).collect::<Vec<_>>()
        );
        assert!(
            cmds.iter().any(|c| c.name == "permissions"),
            "builtin_commands() missing permissions; got: {:?}",
            cmds.iter().map(|c| &c.name).collect::<Vec<_>>()
        );
        assert!(
            cmds.iter().any(|c| c.name == "pr-create"),
            "builtin_commands() missing pr-create; got: {:?}",
            cmds.iter().map(|c| &c.name).collect::<Vec<_>>()
        );
        assert!(
            builtin_command_names().contains("setup"),
            "builtin_command_names() missing setup"
        );
        assert!(
            builtin_command_names().contains("permissions"),
            "builtin_command_names() missing permissions"
        );
        assert!(
            builtin_command_names().contains("pr-create"),
            "builtin_command_names() missing pr-create"
        );
        assert!(!builtin_command_names().contains("configure"));
    }

    /// `/compress` must appear in autocomplete (`builtin_commands`)
    /// and in the collision set (`builtin_command_names`) so a skill
    /// named "compress" can't shadow the built-in.
    #[test]
    fn builtin_commands_include_compress() {
        let cmds = builtin_commands();
        assert!(
            cmds.iter().any(|c| c.name == "compress"),
            "builtin_commands() missing compress; got: {:?}",
            cmds.iter().map(|c| &c.name).collect::<Vec<_>>()
        );
        assert!(
            builtin_command_names().contains("compress"),
            "builtin_command_names() missing compress"
        );
    }

    #[test]
    fn builtin_commands_include_loop() {
        let cmds = builtin_commands();
        assert!(
            cmds.iter().any(|c| c.name == "loop"),
            "builtin_commands() missing loop; got: {:?}",
            cmds.iter().map(|c| &c.name).collect::<Vec<_>>()
        );
        assert!(
            builtin_command_names().contains("loop"),
            "builtin_command_names() missing loop"
        );
    }

    /// `/usage` must surface in autocomplete and the collision set
    /// (so a skill named "usage" can't shadow it) and must be allowed
    /// as a `/loop` target without a configured model (the report is
    /// generated locally and doesn't need an LLM round-trip).
    #[test]
    fn builtin_commands_include_usage() {
        let cmds = builtin_commands();
        assert!(
            cmds.iter().any(|c| c.name == "usage"),
            "builtin_commands() missing usage; got: {:?}",
            cmds.iter().map(|c| &c.name).collect::<Vec<_>>()
        );
        assert!(
            builtin_command_names().contains("usage"),
            "builtin_command_names() missing usage"
        );
        assert!(
            loop_target_runs_without_model("/usage"),
            "/usage must be runnable in a /loop without a configured model"
        );
    }

    /// `/compress` parses via the same slash-command dispatcher used
    /// by `/context` and `/setup`, including case-insensitive and
    /// args-tolerant forms.
    #[test]
    fn is_slash_command_matches_compress_variants() {
        assert!(is_slash_command("/compress", "compress"));
        assert!(is_slash_command("  /compress  ", "compress"));
        assert!(is_slash_command("/compress now", "compress"));
        assert!(is_slash_command("/COMPRESS", "compress"));
        // The dispatcher must not confuse `/compress` with `/context`.
        assert!(!is_slash_command("/context", "compress"));
    }

    #[test]
    fn parse_loop_command_parses_interval_and_target() {
        assert_eq!(
            parse_loop_command("/loop 30 /context"),
            Ok(LoopSpec {
                interval_secs: 30,
                target: "/context".to_string(),
            })
        );
    }

    #[test]
    fn parse_loop_command_rejects_missing_target() {
        let err = parse_loop_command("/loop 30").expect_err("missing target must reject");
        assert!(err.contains("Missing command or prompt"), "got: {err}");
    }

    #[test]
    fn parse_loop_command_rejects_invalid_interval() {
        let err = parse_loop_command("/loop soon /context").expect_err("junk interval must reject");
        assert!(err.contains("Invalid interval"), "got: {err}");
    }

    #[test]
    fn parse_loop_command_rejects_out_of_range() {
        let err = parse_loop_command("/loop 0 /context").expect_err("zero must reject");
        assert!(err.contains("out of range"), "got: {err}");

        let err = parse_loop_command("/loop 86401 /context").expect_err("too large must reject");
        assert!(err.contains("out of range"), "got: {err}");
    }

    #[test]
    fn parse_loop_command_rejects_nested_loop() {
        let err = parse_loop_command("/loop 30 /loop 60 hi").expect_err("nested loop must reject");
        assert!(err.contains("Nested `/loop`"), "got: {err}");
    }

    #[test]
    fn parse_goal_command_is_unbounded_by_default() {
        // No `--max-turns` means no ceiling: the goal runs until it is
        // verifiably complete, blocked, or cancelled.
        assert_eq!(
            parse_goal_command("/goal make cargo test pass"),
            Ok(GoalSpec {
                objective: "make cargo test pass".to_string(),
                max_turns: None,
            })
        );
    }

    #[test]
    fn parse_goal_command_parses_max_turns_flag() {
        assert_eq!(
            parse_goal_command("/goal --max-turns 40 migrate the loader"),
            Ok(GoalSpec {
                objective: "migrate the loader".to_string(),
                max_turns: Some(40),
            })
        );
        // `=` form is equivalent.
        assert_eq!(
            parse_goal_command("/goal --max-turns=7 do the thing"),
            Ok(GoalSpec {
                objective: "do the thing".to_string(),
                max_turns: Some(7),
            })
        );
    }

    #[test]
    fn parse_goal_command_requires_objective() {
        let err = parse_goal_command("/goal").expect_err("bare /goal must reject");
        assert!(err.contains("Usage:"), "got: {err}");
        // A flag with no objective after it is still a usage error.
        let err =
            parse_goal_command("/goal --max-turns 5").expect_err("flag-only /goal must reject");
        assert!(err.contains("Usage:"), "got: {err}");
    }

    #[test]
    fn parse_goal_command_rejects_bad_max_turns() {
        let err = parse_goal_command("/goal --max-turns soon do it")
            .expect_err("junk budget must reject");
        assert!(err.contains("Invalid `--max-turns`"), "got: {err}");

        let err = parse_goal_command("/goal --max-turns 0 do it").expect_err("zero must reject");
        assert!(err.contains("out of range"), "got: {err}");

        let err =
            parse_goal_command("/goal --max-turns 99999 do it").expect_err("too large must reject");
        assert!(err.contains("out of range"), "got: {err}");
    }

    #[test]
    fn parse_goal_command_treats_lookalike_flag_as_objective() {
        // `--max-turnsy` is not the flag, so it stays part of the objective
        // and the goal stays unbounded.
        assert_eq!(
            parse_goal_command("/goal --max-turnsy is a weird objective"),
            Ok(GoalSpec {
                objective: "--max-turnsy is a weird objective".to_string(),
                max_turns: None,
            })
        );
    }

    #[test]
    fn detect_goal_signal_recognizes_complete() {
        assert_eq!(
            detect_goal_signal("All tests pass now.\n\nGOAL_COMPLETE"),
            GoalSignal::Complete
        );
        // Lightly decorated / trailing punctuation still matches.
        assert_eq!(
            detect_goal_signal("done\n`GOAL_COMPLETE`"),
            GoalSignal::Complete
        );
        assert_eq!(
            detect_goal_signal("**GOAL_COMPLETE.**"),
            GoalSignal::Complete
        );
    }

    #[test]
    fn detect_goal_signal_recognizes_blocked_with_reason() {
        assert_eq!(
            detect_goal_signal("I cannot proceed.\nGOAL_BLOCKED: missing API credentials"),
            GoalSignal::Blocked("missing API credentials".to_string())
        );
    }

    #[test]
    fn detect_goal_signal_continue_when_no_sentinel() {
        assert_eq!(
            detect_goal_signal("Made progress: refactored the parser, two tests still red."),
            GoalSignal::Continue
        );
    }

    #[test]
    fn detect_goal_signal_ignores_sentinel_discussed_in_prose() {
        // The model mentioning the sentinel mid-sentence must NOT trip a
        // stop -- only a standalone line counts.
        assert_eq!(
            detect_goal_signal("I will emit GOAL_COMPLETE once the suite is green."),
            GoalSignal::Continue
        );
    }

    #[test]
    fn detect_goal_signal_complete_wins_over_blocked() {
        assert_eq!(
            detect_goal_signal("GOAL_BLOCKED: earlier note\nGOAL_COMPLETE"),
            GoalSignal::Complete
        );
    }

    #[test]
    fn decide_continue_runs_forever_when_unbounded() {
        // No ceiling + no signal => keep going, counter stays reset.
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Continue, 1_000, None, 0),
            GoalStep::Continue {
                consecutive_blocked: 0
            }
        );
    }

    #[test]
    fn decide_complete_stops_immediately() {
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Complete, 1, None, 0),
            GoalStep::Stop(GoalStop::Completed)
        );
    }

    #[test]
    fn decide_complete_wins_on_the_final_turn() {
        // Even when the ceiling is reached, a verified completion reports
        // success rather than a budget stop.
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Complete, 5, Some(5), 0),
            GoalStep::Stop(GoalStop::Completed)
        );
    }

    #[test]
    fn decide_blocked_needs_three_consecutive_turns() {
        // First two blocked reports keep going with an incrementing counter.
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Blocked("x".into()), 1, None, 0),
            GoalStep::Continue {
                consecutive_blocked: 1
            }
        );
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Blocked("x".into()), 2, None, 1),
            GoalStep::Continue {
                consecutive_blocked: 2
            }
        );
        // The third consecutive blocked report stops the loop.
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Blocked("stuck".into()), 3, None, 2),
            GoalStep::Stop(GoalStop::Blocked("stuck".into()))
        );
    }

    #[test]
    fn decide_continue_resets_the_blocked_counter() {
        // A productive turn after some blocked reports clears the counter,
        // so a later transient blocker starts counting from scratch.
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Continue, 4, None, 2),
            GoalStep::Continue {
                consecutive_blocked: 0
            }
        );
    }

    #[test]
    fn decide_ceiling_stops_only_when_opted_in() {
        // Unbounded: never a ceiling stop.
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Continue, 9_999, None, 0),
            GoalStep::Continue {
                consecutive_blocked: 0
            }
        );
        // Opt-in ceiling reached with no completion => budget stop.
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Continue, 25, Some(25), 0),
            GoalStep::Stop(GoalStop::CeilingReached)
        );
        // A sub-threshold blocker on the final allowed turn also yields a
        // ceiling stop (it can't keep retrying past the budget).
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Blocked("y".into()), 25, Some(25), 0),
            GoalStep::Stop(GoalStop::CeilingReached)
        );
    }

    #[test]
    fn decide_after_goal_failure_backs_off_on_transient() {
        // A retryable (transient) failure backs off and retries, incrementing
        // the outage streak -- the goal survives the outage rather than stopping.
        let transient = crate::tool_loop::TurnFailure {
            retryable: true,
            message: "server_is_overloaded".to_string(),
        };
        assert_eq!(
            decide_after_goal_failure(&transient, 0),
            GoalFailureAction::Backoff {
                consecutive_failures: 1
            }
        );
        assert_eq!(
            decide_after_goal_failure(&transient, 4),
            GoalFailureAction::Backoff {
                consecutive_failures: 5
            }
        );
    }

    #[test]
    fn decide_after_goal_failure_stops_on_fatal() {
        // A non-retryable failure (auth, invalid request, panic) stops the goal:
        // retrying would not help.
        let fatal = crate::tool_loop::TurnFailure {
            retryable: false,
            message: "agent loop panicked".to_string(),
        };
        assert_eq!(
            decide_after_goal_failure(&fatal, 0),
            GoalFailureAction::Stop
        );
        assert_eq!(
            decide_after_goal_failure(&fatal, 9),
            GoalFailureAction::Stop
        );
    }

    #[test]
    fn goal_failure_backoff_grows_then_caps() {
        // First failure ~200ms (codex base, jittered); the delay grows
        // exponentially but never exceeds the cap, so a long outage settles
        // into a steady poll instead of growing without bound.
        let first = goal_failure_backoff(1);
        assert!(
            (180..=220).contains(&first.as_millis()),
            "first backoff should jitter around 200ms, got {first:?}"
        );
        assert!(first <= GOAL_FAILURE_BACKOFF_CAP);
        // A large streak is clamped to the cap (and must not overflow/panic).
        assert_eq!(goal_failure_backoff(1_000), GOAL_FAILURE_BACKOFF_CAP);
    }

    #[test]
    fn build_goal_prompt_embeds_objective_and_sentinels() {
        // Unbounded goal: header carries the turn number but no ceiling.
        let p = build_goal_prompt("ship the feature", 1, None, GoalPhase::Continue);
        assert!(p.contains("ship the feature"), "objective missing");
        assert!(
            p.contains(GOAL_COMPLETE_SENTINEL),
            "complete sentinel missing"
        );
        assert!(
            p.contains(GOAL_BLOCKED_SENTINEL),
            "blocked sentinel missing"
        );
        assert!(p.contains("turn 1)"), "unbounded turn header missing");
        assert!(
            !p.contains("of at most"),
            "unbounded goal must not advertise a ceiling"
        );

        // Capped goal: header advertises the ceiling.
        let capped = build_goal_prompt("ship it", 3, Some(25), GoalPhase::Continue);
        assert!(
            capped.contains("turn 3 of at most 25"),
            "capped turn header missing"
        );

        let wrap = build_goal_prompt("ship it", 25, Some(25), GoalPhase::FinalWrapUp);
        assert!(wrap.contains("FINAL turn"), "wrap-up framing missing");
    }

    /// `plan_compress` returns the indexes of every turn whose
    /// `summary` is `None`, in chronological order. Already-summarized
    /// turns must NOT appear -- `/compress` should be idempotent.
    #[test]
    fn plan_compress_returns_uncompressed_turn_indexes() {
        use crate::session::{ConversationTurn, SessionSnapshot};
        let snap = SessionSnapshot {
            cwd: std::path::PathBuf::from("/tmp/cwd"),
            mode: SessionMode::Code,
            model: "m".into(),
            history: vec![
                ConversationTurn {
                    user_prompt: "u0".into(),
                    summary: Some("already done".into()),
                    ..Default::default()
                },
                ConversationTurn {
                    user_prompt: "u1".into(),
                    ..Default::default()
                },
                ConversationTurn {
                    user_prompt: "u2".into(),
                    summary: Some("also done".into()),
                    ..Default::default()
                },
                ConversationTurn {
                    user_prompt: "u3".into(),
                    ..Default::default()
                },
            ],
            reasoning_effort: None,
            idle_timeout_secs: None,
            project_instructions: String::new(),
            skills: std::sync::Arc::new(crate::skills::SkillRegistry::default()),
        };
        let plan = plan_compress(&snap);
        assert_eq!(plan.total, 4);
        assert_eq!(plan.uncompressed, vec![1, 3]);
    }

    /// When every turn already carries a summary, `plan_compress`
    /// must report zero work to do so `handle_compress` can short-
    /// circuit before making any LLM calls. Idempotent re-runs are
    /// the property the user relies on.
    #[test]
    fn plan_compress_reports_empty_when_all_summarized() {
        use crate::session::{ConversationTurn, SessionSnapshot};
        let snap = SessionSnapshot {
            cwd: std::path::PathBuf::from("/tmp/cwd"),
            mode: SessionMode::Code,
            model: "m".into(),
            history: vec![ConversationTurn {
                user_prompt: "u".into(),
                summary: Some("done".into()),
                ..Default::default()
            }],
            reasoning_effort: None,
            idle_timeout_secs: None,
            project_instructions: String::new(),
            skills: std::sync::Arc::new(crate::skills::SkillRegistry::default()),
        };
        assert!(plan_compress(&snap).uncompressed.is_empty());
    }

    /// `approximate_turn_tokens` charges tool exchanges as well as
    /// user/assistant text, so a tool-heavy turn correctly looks
    /// expensive when the user reads the "verbatim -> summary"
    /// savings line in the report.
    #[test]
    fn approximate_turn_tokens_includes_tool_exchanges() {
        use crate::session::{ConversationTurn, ToolExchange};
        let plain = ConversationTurn {
            user_prompt: "u".into(),
            agent_response: "a".into(),
            ..Default::default()
        };
        let toolful = ConversationTurn {
            user_prompt: "u".into(),
            agent_response: "a".into(),
            tool_exchanges: vec![ToolExchange {
                call_id: "c".into(),
                tool_name: "search".into(),
                arguments: r#"{"q":"x"}"#.into(),
                result: "z".repeat(5_000),
            }],
            ..Default::default()
        };
        assert!(approximate_turn_tokens(&toolful) > approximate_turn_tokens(&plain));
    }

    #[test]
    fn shell_single_quote_escapes_embedded_quote() {
        assert_eq!(shell_single_quote("hello"), "'hello'");
        assert_eq!(shell_single_quote(""), "''");
        // The standard `'\''` escape: close, escaped quote, reopen.
        assert_eq!(shell_single_quote("it's"), "'it'\\''s'");
        // Backticks/$/" are harmless inside single quotes -- preserved as-is.
        assert_eq!(shell_single_quote("$x `y` \"z\""), "'$x `y` \"z\"'");
    }

    #[test]
    fn model_config_option_omitted_when_catalog_empty() {
        // No discovery results means we can't offer a meaningful dropdown.
        assert!(model_config_option("anything", &[]).is_none());
    }

    #[test]
    fn model_config_option_present_when_catalog_known() {
        let models = vec!["model-a".to_string(), "model-b".to_string()];
        // Spot-check that the option is actually built. Field shapes are
        // covered by the `agent-client-protocol` crate; we just need to know
        // the helper produced *something*.
        assert!(model_config_option("model-a", &models).is_some());
        // Out-of-catalog current value still produces an option (we fall
        // back to the first catalog entry); tested implicitly via the
        // is_some assertion plus the no-panic contract.
        assert!(model_config_option("model-zzz", &models).is_some());
        assert!(model_config_option("", &models).is_some());
    }

    /// `extract_prompt_text` joins text blocks with newlines and silently
    /// drops blocks that are not text -- images, embedded resources, etc.
    /// don't get fed to the chat-completions endpoint.
    #[test]
    fn extract_prompt_text_joins_text_blocks_with_newlines() {
        let blocks = vec![
            ContentBlock::Text(TextContent::new("hello")),
            ContentBlock::Text(TextContent::new("world")),
        ];
        assert_eq!(extract_prompt_text(&blocks), "hello\nworld");
    }

    #[test]
    fn extract_prompt_text_returns_empty_for_no_text_blocks() {
        // Empty input is the simplest case `session/prompt` rejects with
        // "Error: empty prompt" -- the helper itself just yields "".
        assert_eq!(extract_prompt_text(&[]), "");
    }

    /// A prompt with mixed blocks (e.g. text plus an image) must keep the
    /// text and silently drop the rest. Today the agent doesn't advertise
    /// image support, but well-behaved clients can still send mixed prompts
    /// when speaking to multiple agents through a single session.
    #[test]
    fn extract_prompt_text_filters_non_text_blocks() {
        use agent_client_protocol::schema::ImageContent;
        let blocks = vec![
            ContentBlock::Text(TextContent::new("before")),
            ContentBlock::Image(ImageContent::new("base64data", "image/png")),
            ContentBlock::Text(TextContent::new("after")),
        ];
        assert_eq!(extract_prompt_text(&blocks), "before\nafter");
    }

    /// ACP requires baseline agents to accept resource links. Anvil surfaces
    /// them as textual references so a link is never silently dropped (#150).
    #[test]
    fn extract_prompt_parts_renders_resource_link_as_text() {
        use agent_client_protocol::schema::ResourceLink;
        let link = ResourceLink::new("notes.md", "file:///repo/notes.md")
            .description("design notes")
            .mime_type("text/markdown");
        let parts = extract_prompt_parts(&[ContentBlock::ResourceLink(link)]);
        assert_eq!(parts.len(), 1, "resource link must produce one part");
        let ChatContentPart::Text { text } = &parts[0] else {
            panic!("resource link should become a text part: {parts:?}");
        };
        assert!(text.contains("notes.md"), "missing name: {text}");
        assert!(
            text.contains("file:///repo/notes.md"),
            "missing uri: {text}"
        );
        assert!(text.contains("design notes"), "missing description: {text}");
    }

    /// A resource-link-only prompt must not be mistaken for an empty prompt:
    /// `extract_prompt_parts` yields a non-empty part list (#150).
    #[test]
    fn extract_prompt_parts_resource_link_only_is_not_empty() {
        use agent_client_protocol::schema::ResourceLink;
        let link = ResourceLink::new("a.rs", "file:///repo/a.rs");
        let parts = extract_prompt_parts(&[ContentBlock::ResourceLink(link)]);
        assert!(
            !parts.is_empty(),
            "resource-link-only prompt should not be empty"
        );
    }

    /// Anvil advertises `embeddedContext`, so embedded text resources must
    /// reach prompt construction rather than being dropped (#151).
    #[test]
    fn extract_prompt_parts_inlines_embedded_text_resource() {
        use agent_client_protocol::schema::{
            EmbeddedResource, EmbeddedResourceResource, TextResourceContents,
        };
        let resource = EmbeddedResource::new(EmbeddedResourceResource::TextResourceContents(
            TextResourceContents::new("fn main() {}", "file:///repo/main.rs"),
        ));
        let parts = extract_prompt_parts(&[ContentBlock::Resource(resource)]);
        assert_eq!(parts.len(), 1);
        let ChatContentPart::Text { text } = &parts[0] else {
            panic!("embedded text resource should become a text part: {parts:?}");
        };
        assert!(text.contains("fn main() {}"), "missing body: {text}");
        assert!(
            text.contains("file:///repo/main.rs"),
            "missing uri tag: {text}"
        );
    }

    /// An embedded image blob is forwarded as an image part for vision
    /// models (#151).
    #[test]
    fn extract_prompt_parts_forwards_embedded_image_blob() {
        use agent_client_protocol::schema::{
            BlobResourceContents, EmbeddedResource, EmbeddedResourceResource,
        };
        let resource = EmbeddedResource::new(EmbeddedResourceResource::BlobResourceContents(
            BlobResourceContents::new("AAAA", "file:///repo/pic.png").mime_type("image/png"),
        ));
        let parts = extract_prompt_parts(&[ContentBlock::Resource(resource)]);
        assert_eq!(parts.len(), 1);
        assert!(
            matches!(&parts[0], ChatContentPart::Image { .. }),
            "image blob should become an image part: {parts:?}"
        );
    }

    /// A non-image embedded blob is surfaced as a textual placeholder rather
    /// than silently dropped (#151).
    #[test]
    fn extract_prompt_parts_placeholders_embedded_binary_blob() {
        use agent_client_protocol::schema::{
            BlobResourceContents, EmbeddedResource, EmbeddedResourceResource,
        };
        let resource = EmbeddedResource::new(EmbeddedResourceResource::BlobResourceContents(
            BlobResourceContents::new("AAAA", "file:///repo/data.bin")
                .mime_type("application/octet-stream"),
        ));
        let parts = extract_prompt_parts(&[ContentBlock::Resource(resource)]);
        assert_eq!(parts.len(), 1);
        let ChatContentPart::Text { text } = &parts[0] else {
            panic!("binary blob should become a text placeholder: {parts:?}");
        };
        assert!(
            text.contains("file:///repo/data.bin"),
            "missing uri: {text}"
        );
    }

    /// A cancelled turn resolves with `StopReason::Cancelled`; a normal turn
    /// stays `EndTurn` (#152).
    #[test]
    fn prompt_stop_response_maps_cancellation_to_stop_reason() {
        assert_eq!(
            prompt_stop_response(true).stop_reason,
            StopReason::Cancelled
        );
        assert_eq!(prompt_stop_response(false).stop_reason, StopReason::EndTurn);
        assert_eq!(
            prompt_end_turn_response().stop_reason,
            StopReason::EndTurn,
            "the non-cancellable convenience wrapper always ends the turn"
        );
    }

    #[test]
    fn image_prompt_rejection_blocks_known_text_only_models() {
        let prompt_parts = vec![ChatContentPart::image_url("https://example.com/cat.png")];
        let catalog = vec![ModelMetadata {
            id: "text-only".into(),
            default_reasoning_level: None,
            supported_reasoning_levels: Vec::new(),
            supports_images: Some(false),
            context_length: None,
            pricing: None,
        }];

        let message =
            image_prompt_rejection("text-only", &prompt_parts, &catalog).expect("must reject");
        assert!(message.contains("does not advertise image input support"));
    }

    #[test]
    fn image_prompt_rejection_allows_unknown_support_models() {
        let prompt_parts = vec![ChatContentPart::image_url("https://example.com/cat.png")];
        let catalog = vec![ModelMetadata::id_only("unknown")];
        assert!(image_prompt_rejection("unknown", &prompt_parts, &catalog).is_none());
    }

    #[test]
    fn prompt_response_meta_includes_structured_output_success() {
        let result =
            StructuredOutputResult::Success(crate::structured_output::StructuredOutputSuccess {
                schema_name: "audit_result".into(),
                validated_output: serde_json::json!({"answer":"ok"}),
                coercion_requested: false,
            });
        let meta = prompt_response_meta(Some(&result), None).expect("meta present");
        assert_eq!(
            meta["anvil"]["structuredOutput"]["status"],
            serde_json::Value::String("success".into())
        );
        assert_eq!(
            meta["anvil"]["structuredOutput"]["validated_output"]["answer"],
            "ok"
        );
        assert_eq!(
            meta["anvil"]["structuredOutput"]["coercion_requested"],
            serde_json::Value::Bool(false)
        );
    }

    #[test]
    fn prompt_response_meta_includes_structured_output_coerced_success() {
        let result = StructuredOutputResult::CoercedSuccess(
            crate::structured_output::StructuredOutputCoercedSuccess {
                schema_name: "audit_result".into(),
                validated_output: serde_json::json!({"answer":"one\ntwo"}),
                coercions: vec!["response.answer array -> string".into()],
                coercion_requested: true,
            },
        );
        let meta = prompt_response_meta(Some(&result), None).expect("meta present");
        assert_eq!(
            meta["anvil"]["structuredOutput"]["status"],
            serde_json::Value::String("coerced_success".into())
        );
        assert_eq!(
            meta["anvil"]["structuredOutput"]["validated_output"]["answer"],
            "one\ntwo"
        );
        assert_eq!(
            meta["anvil"]["structuredOutput"]["coercions"][0],
            "response.answer array -> string"
        );
        assert_eq!(
            meta["anvil"]["structuredOutput"]["coercion_requested"],
            serde_json::Value::Bool(true)
        );
    }

    #[test]
    fn prompt_response_meta_includes_structured_output_validation_error_coercion_flag() {
        let result = StructuredOutputResult::ValidationError(
            crate::structured_output::StructuredOutputValidationError {
                schema_name: "audit_result".into(),
                errors: vec![],
                invalid_excerpt: "{\"answer\":null}".into(),
                coercion_requested: true,
            },
        );
        let meta = prompt_response_meta(Some(&result), None).expect("meta present");
        assert_eq!(
            meta["anvil"]["structuredOutput"]["status"],
            serde_json::Value::String("validation_error".into())
        );
        assert_eq!(
            meta["anvil"]["structuredOutput"]["coercion_requested"],
            serde_json::Value::Bool(true)
        );
    }

    #[test]
    fn prompt_response_meta_is_absent_without_structured_output() {
        assert!(prompt_response_meta(None, None).is_none());
    }

    #[test]
    fn prompt_response_meta_includes_model_selection_contract() {
        let model = ResolvedModelInfo {
            configured_model: "openrouter::google/gemini-3.1-pro-preview".into(),
            resolved_provider: Some("openrouter".into()),
            resolved_model: "google/gemini-3.1-pro-preview".into(),
        };
        let meta = prompt_response_meta(None, Some(&model)).expect("meta present");

        assert_eq!(
            meta["anvil"]["modelSelection"]["orchestration"]["configured_model"],
            "openrouter::google/gemini-3.1-pro-preview"
        );
        assert_eq!(
            meta["anvil"]["modelSelection"]["orchestration"]["resolved_provider"],
            "openrouter"
        );
        assert_eq!(
            meta["anvil"]["modelSelection"]["orchestration"]["resolved_model"],
            "google/gemini-3.1-pro-preview"
        );
        assert_eq!(
            meta["anvil"]["modelSelection"]["orchestration"]["actual_model"],
            "google/gemini-3.1-pro-preview"
        );
        assert_eq!(
            meta["anvil"]["modelSelection"]["internal_specialist"]["separate_model_selection_supported"],
            serde_json::Value::Bool(false)
        );
        assert_eq!(
            meta["anvil"]["modelSelection"]["internal_specialist"]["actual_model"],
            "google/gemini-3.1-pro-preview"
        );
        assert_eq!(
            meta["anvil"]["modelSelection"]["internal_specialist"]["selection_source"],
            "inherits_orchestration"
        );
    }

    /// All four behavior modes embed the cwd into the system prompt and
    /// open with the shared general-purpose identity line, while still
    /// carrying a distinct mode-specific paragraph. The "AI coding
    /// assistant" wording must stay gone -- some models refuse non-coding
    /// prompts when it's present, which is the regression this guards.
    #[test]
    fn build_system_prompt_includes_cwd_and_mode_specific_text() {
        let cwd = std::path::Path::new("/tmp/some-cwd");
        for (mode, marker) in [
            (SessionMode::Lutz, "agentic approach"),
            (SessionMode::Code, "focus on code changes"),
            (SessionMode::Ask, "Answer questions about code"),
            (SessionMode::Plan, "focus on planning"),
        ] {
            let prompt = build_system_prompt(&mode, cwd);
            assert!(
                prompt.contains("/tmp/some-cwd") || prompt.contains("\\tmp\\some-cwd"),
                "system prompt for {mode:?} must embed the cwd, got: {prompt}"
            );
            assert!(
                prompt.contains(marker),
                "system prompt for {mode:?} must mention '{marker}', got: {prompt}"
            );
            assert!(
                prompt.contains("any task the user brings to you"),
                "system prompt for {mode:?} must use the general-purpose identity opening, got: {prompt}"
            );
            assert!(
                !prompt.contains("AI coding assistant"),
                "system prompt for {mode:?} must not revive the 'AI coding assistant' wording, got: {prompt}"
            );
        }
    }

    /// `render_context_report` is the body of the `/context` slash command.
    /// It should surface the mode, permission mode, model, conversation
    /// turn count, and token estimate -- enough that the user can debug
    /// "why does the model think X" without a separate inspector.
    #[test]
    fn render_context_report_lists_session_facts() {
        use crate::llm_client::ModelMetadata;
        use crate::session::{ConversationTurn, SessionSnapshot};
        let snap = SessionSnapshot {
            cwd: std::path::PathBuf::from("/tmp/cwd"),
            mode: SessionMode::Code,
            model: "gpt-99".into(),
            history: vec![ConversationTurn {
                user_prompt: "hi".repeat(8),
                agent_response: "ok".repeat(8),
                ..Default::default()
            }],
            reasoning_effort: None,
            idle_timeout_secs: None,
            project_instructions: String::new(),
            skills: std::sync::Arc::new(crate::skills::SkillRegistry::default()),
        };
        let catalog = vec![ModelMetadata {
            id: "gpt-99".into(),
            default_reasoning_level: None,
            supported_reasoning_levels: Vec::new(),
            supports_images: None,
            context_length: Some(200_000),
            pricing: None,
        }];
        let report = render_context_report(&snap, PermissionMode::AcceptEdits, &catalog);

        assert!(report.contains("Mode: `CODE`"));
        assert!(report.contains("Permission mode: `acceptEdits`"));
        assert!(report.contains("Model: `gpt-99`"));
        assert!(report.contains("(1 known in catalog)"));
        assert!(report.contains("Conversation turns: 1"));
        // Context-window line must surface both the count and the cap
        // when the catalog publishes one.
        assert!(report.contains("/ 200000 tokens"));
        assert!(report.contains("% used"));
    }

    /// When no model is set, `/context` shows `(none)` rather than the
    /// empty string so the user notices the misconfig.
    #[test]
    fn render_context_report_shows_none_when_model_empty() {
        use crate::session::SessionSnapshot;
        let snap = SessionSnapshot {
            cwd: std::path::PathBuf::from("/tmp/cwd"),
            mode: SessionMode::Lutz,
            model: String::new(),
            history: vec![],
            reasoning_effort: None,
            idle_timeout_secs: None,
            project_instructions: String::new(),
            skills: std::sync::Arc::new(crate::skills::SkillRegistry::default()),
        };
        let report = render_context_report(&snap, PermissionMode::Default, &[]);
        assert!(report.contains("Model: `(none)`"));
        assert!(report.contains("(0 known in catalog)"));
        assert!(report.contains("Conversation turns: 0"));
        // No catalog entry for the (empty) model id -> falls back to
        // the "model max unknown" line rather than crashing.
        assert!(report.contains("model max unknown"));
    }

    fn usage_snapshot(model: &str) -> crate::session::SessionSnapshot {
        crate::session::SessionSnapshot {
            cwd: std::path::PathBuf::from("/tmp/cwd"),
            mode: SessionMode::Code,
            model: model.into(),
            history: vec![],
            reasoning_effort: None,
            idle_timeout_secs: None,
            project_instructions: String::new(),
            skills: std::sync::Arc::new(crate::skills::SkillRegistry::default()),
        }
    }

    #[test]
    fn render_usage_report_with_openrouter_balance_shows_all_lines() {
        let snap = usage_snapshot("openrouter::anthropic/claude-sonnet-4.5");
        let usage = crate::llm_client::TokenUsage {
            input_tokens: 1_000,
            output_tokens: 500,
            thought_tokens: 200,
            cached_read_tokens: 50,
            cached_write_tokens: 25,
        };
        let credits = crate::openrouter_credits::Credits {
            total_credits: 50.0,
            total_usage: 7.5,
        };
        let report = render_usage_report(
            &snap,
            usage,
            Some(0.1234),
            OpenRouterCreditsOutcome::Fetched(credits),
            CodexCreditsOutcome::NotApplicable,
        );

        assert!(report.contains("Model: `anthropic/claude-sonnet-4.5` (openrouter)"));
        // 1000+500+200+50+25 = 1775 across the five buckets.
        assert!(report.contains("1775 total"));
        assert!(report.contains("input 1000"));
        assert!(report.contains("reasoning 200"));
        assert!(report.contains("Session cost: $0.1234 USD"));
        assert!(
            report.contains("$42.5000 remaining"),
            "expected balance line in: {report}"
        );
        assert!(report.contains("$50.0000 purchased"));
        assert!(report.contains("$7.5000 used"));
    }

    #[test]
    fn render_usage_report_skips_openrouter_when_model_is_codex() {
        let snap = usage_snapshot("codex::gpt-5-codex");
        let report = render_usage_report(
            &snap,
            crate::llm_client::TokenUsage::default(),
            None,
            OpenRouterCreditsOutcome::Skipped,
            CodexCreditsOutcome::NoAuth,
        );
        assert!(report.contains("Model: `gpt-5-codex` (codex)"));
        assert!(report.contains("no LLM turns recorded yet this session"));
        // No billable turns yet should show $0 rather than the
        // "pricing unavailable" wording.
        assert!(report.contains("$0.0000 USD (no billable turns yet)"));
        assert!(report.contains("not applicable (active model is not OpenRouter)"));
        // NoAuth on a Codex session must surface the actionable hint --
        // the user is clearly trying to use ChatGPT-subscription routing.
        assert!(report.contains("Codex (ChatGPT) status: no ChatGPT credentials"));
        assert!(report.contains("/setup codex"));
    }

    #[test]
    fn render_usage_report_api_key_mode_on_codex_session_points_at_dashboard() {
        // api-key-mode Codex users (auth.json with OPENAI_API_KEY but
        // no chatgpt tokens) must not be told to run `/setup codex` --
        // their setup is correct, the subscription endpoint just
        // doesn't apply. Verify the renderer points them at the
        // OpenAI dashboard instead.
        let snap = usage_snapshot("codex::gpt-5-codex");
        let report = render_usage_report(
            &snap,
            crate::llm_client::TokenUsage::default(),
            None,
            OpenRouterCreditsOutcome::Skipped,
            CodexCreditsOutcome::ApiKeyMode,
        );
        assert!(report.contains("Codex billing: OPENAI_API_KEY"));
        assert!(report.contains("platform.openai.com/usage"));
        // Critically: must NOT tell them to re-authenticate.
        assert!(
            !report.contains("/setup codex"),
            "api-key-mode users should not be told to re-auth: {report}"
        );
    }

    #[test]
    fn render_usage_report_distinguishes_no_credential_for_openrouter_model() {
        let snap = usage_snapshot("openrouter::openai/gpt-4o");
        let report = render_usage_report(
            &snap,
            crate::llm_client::TokenUsage::default(),
            None,
            OpenRouterCreditsOutcome::Skipped,
            CodexCreditsOutcome::NotApplicable,
        );
        // The "no credential" branch must fire for OpenRouter models,
        // distinct from "not applicable" for non-OpenRouter ones.
        assert!(report.contains("no credential configured"));
        assert!(report.contains("/setup openrouter key"));
        // Skipped Codex on a non-Codex session stays silent so the
        // report doesn't grow a line per unconfigured provider.
        assert!(
            !report.contains("Codex"),
            "Codex line must be silent on non-Codex session: {report}"
        );
    }

    #[test]
    fn render_usage_report_surfaces_upstream_failure() {
        let snap = usage_snapshot("openrouter::openai/gpt-4o");
        let report = render_usage_report(
            &snap,
            crate::llm_client::TokenUsage::default(),
            None,
            OpenRouterCreditsOutcome::Failed("HTTP 401: invalid api key".into()),
            CodexCreditsOutcome::NotApplicable,
        );
        assert!(report.contains("OpenRouter balance: lookup failed"));
        assert!(report.contains("HTTP 401: invalid api key"));
    }

    #[test]
    fn render_usage_report_marks_partial_pricing_as_unavailable() {
        let snap = usage_snapshot("openrouter::openai/gpt-4o");
        let usage = crate::llm_client::TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        };
        // cost_usd = None *with* non-zero usage means at least one turn
        // ran without pricing metadata. Distinct from the no-LLM-yet
        // wording (covered above).
        let report = render_usage_report(
            &snap,
            usage,
            None,
            OpenRouterCreditsOutcome::Skipped,
            CodexCreditsOutcome::NotApplicable,
        );
        assert!(report.contains("Session cost: unavailable"));
        assert!(report.contains("at least one turn lacked pricing metadata"));
    }

    #[test]
    fn render_usage_report_handles_bare_model_id_without_panic() {
        // A bare id (no `source::` prefix) is legal on the wire when
        // the user typed a default model that routes to a backend.
        let snap = usage_snapshot("llama3:latest");
        let report = render_usage_report(
            &snap,
            crate::llm_client::TokenUsage::default(),
            None,
            OpenRouterCreditsOutcome::Skipped,
            CodexCreditsOutcome::NotApplicable,
        );
        assert!(report.contains("Model: `llama3:latest` ((unknown))"));
        assert!(report.contains("not applicable"));
    }

    #[test]
    fn render_usage_report_shows_codex_metered_credits() {
        let snap = usage_snapshot("codex::gpt-5-codex");
        let usage = crate::codex_credits::CodexUsage {
            plan_type: Some("plus".to_string()),
            credits: Some(crate::codex_credits::CreditStatus {
                has_credits: true,
                unlimited: false,
                balance: Some("12.50".to_string()),
            }),
            rate_limit: Some(crate::codex_credits::RateLimitStatus {
                primary_window: Some(crate::codex_credits::RateLimitWindow {
                    used_percent: 42,
                    reset_after_seconds: 7_200,
                }),
            }),
        };
        let report = render_usage_report(
            &snap,
            crate::llm_client::TokenUsage::default(),
            None,
            OpenRouterCreditsOutcome::Skipped,
            CodexCreditsOutcome::Fetched(usage),
        );
        assert!(report.contains("ChatGPT plan: Plus"));
        assert!(report.contains("Codex credits: $12.50 remaining"));
        assert!(report.contains("Codex rate limit: 42% of primary window used"));
        assert!(report.contains("resets in 2h 0m"));
    }

    #[test]
    fn render_usage_report_shows_codex_unlimited_plan() {
        let snap = usage_snapshot("codex::gpt-5-codex");
        let usage = crate::codex_credits::CodexUsage {
            plan_type: Some("pro".to_string()),
            credits: Some(crate::codex_credits::CreditStatus {
                has_credits: true,
                unlimited: true,
                balance: None,
            }),
            rate_limit: None,
        };
        let report = render_usage_report(
            &snap,
            crate::llm_client::TokenUsage::default(),
            None,
            OpenRouterCreditsOutcome::Skipped,
            CodexCreditsOutcome::Fetched(usage),
        );
        assert!(report.contains("ChatGPT plan: Pro"));
        assert!(report.contains("Codex credits: unlimited on this plan"));
        // No rate-limit info on the wire -> renderer omits the line
        // rather than guessing.
        assert!(!report.contains("Codex rate limit"));
    }

    #[test]
    fn render_usage_report_codex_failure_includes_diagnostic() {
        let snap = usage_snapshot("codex::gpt-5-codex");
        let report = render_usage_report(
            &snap,
            crate::llm_client::TokenUsage::default(),
            None,
            OpenRouterCreditsOutcome::Skipped,
            CodexCreditsOutcome::Failed(
                "chatgpt /wham/usage returned HTTP 401: invalid_token".into(),
            ),
        );
        assert!(report.contains("Codex (ChatGPT) status: lookup failed"));
        assert!(report.contains("HTTP 401"));
        // The actual bearer token must never appear in the failure
        // line -- we only echo upstream body excerpts, not local state.
        assert!(!report.contains("Bearer"));
    }

    #[test]
    fn humanize_plan_type_title_cases_known_snakes() {
        assert_eq!(humanize_plan_type("plus"), "Plus");
        assert_eq!(humanize_plan_type("free_workspace"), "Free Workspace");
        assert_eq!(
            humanize_plan_type("self_serve_business_usage_based"),
            "Self Serve Business Usage Based"
        );
        // Unknown slugs pass through with whatever casing the server sent.
        assert_eq!(humanize_plan_type("hyperspace"), "Hyperspace");
    }

    #[test]
    fn format_reset_after_renders_sane_durations() {
        assert_eq!(format_reset_after(0), "soon");
        assert_eq!(format_reset_after(30), "soon");
        assert_eq!(format_reset_after(60), "soon");
        assert_eq!(format_reset_after(125), "2m");
        assert_eq!(format_reset_after(3_600), "1h 0m");
        assert_eq!(format_reset_after(7_320), "2h 2m");
        assert_eq!(format_reset_after(90_000), "1d 1h");
    }

    #[test]
    fn session_usage_update_reports_replayed_prompt_tokens() {
        use crate::llm_client::ModelMetadata;
        use crate::session::{ConversationTurn, SessionSnapshot};

        let snap = SessionSnapshot {
            cwd: std::path::PathBuf::from("/tmp/cwd"),
            mode: SessionMode::Code,
            model: "gpt-99".into(),
            history: vec![ConversationTurn {
                user_prompt: "investigate context accounting".into(),
                agent_response: "count the replayed prompt, not cumulative billing".into(),
                ..Default::default()
            }],
            reasoning_effort: None,
            idle_timeout_secs: None,
            project_instructions: "Use the local style.".into(),
            skills: std::sync::Arc::new(crate::skills::SkillRegistry::default()),
        };
        let catalog = vec![ModelMetadata {
            id: "gpt-99".into(),
            default_reasoning_level: None,
            supported_reasoning_levels: Vec::new(),
            supports_images: None,
            context_length: Some(200_000),
            pricing: None,
        }];

        let update = session_usage_update(&snap, &catalog, None);
        let expected_used = crate::tokens::approximate_tokens_messages(
            &build_prompt_messages_with_parts(&snap, "", &[]),
        ) as u64;

        assert_eq!(update.used, expected_used);
        assert_eq!(update.size, 200_000);
    }

    #[test]
    fn session_usage_update_falls_back_when_model_window_unknown() {
        use crate::llm_client::ModelMetadata;
        use crate::session::SessionSnapshot;

        let snap = SessionSnapshot {
            cwd: std::path::PathBuf::from("/tmp/cwd"),
            mode: SessionMode::Ask,
            model: "codex::gpt-5-codex".into(),
            history: vec![],
            reasoning_effort: None,
            idle_timeout_secs: None,
            project_instructions: String::new(),
            skills: std::sync::Arc::new(crate::skills::SkillRegistry::default()),
        };
        let catalog = vec![ModelMetadata {
            id: "codex::gpt-5-codex".into(),
            default_reasoning_level: None,
            supported_reasoning_levels: Vec::new(),
            supports_images: None,
            context_length: None,
            pricing: None,
        }];

        let update = session_usage_update(&snap, &catalog, None);

        assert_eq!(
            update.size,
            crate::context_manager::FALLBACK_CONTEXT_LENGTH as u64
        );
    }

    #[test]
    fn session_usage_update_includes_cost_when_available() {
        use crate::llm_client::ModelMetadata;
        use crate::session::SessionSnapshot;

        let snap = SessionSnapshot {
            cwd: std::path::PathBuf::from("/tmp/cwd"),
            mode: SessionMode::Ask,
            model: "openrouter::openai/gpt-4o".into(),
            history: vec![],
            reasoning_effort: None,
            idle_timeout_secs: None,
            project_instructions: String::new(),
            skills: std::sync::Arc::new(crate::skills::SkillRegistry::default()),
        };
        let catalog = vec![ModelMetadata {
            id: "openrouter::openai/gpt-4o".into(),
            default_reasoning_level: None,
            supported_reasoning_levels: Vec::new(),
            supports_images: None,
            context_length: Some(128_000),
            pricing: None,
        }];

        let update = session_usage_update(&snap, &catalog, Some(1.25));

        assert_eq!(update.cost.as_ref().map(|cost| cost.amount), Some(1.25));
        assert_eq!(
            update.cost.as_ref().map(|cost| cost.currency.as_str()),
            Some("USD")
        );
    }

    /// `session/list` should expose the persisted title and updatedAt
    /// fields so the client can render the thread name and sort order.
    #[test]
    fn session_info_from_manifest_populates_title_and_updated_at() {
        use crate::session::SessionManifest;

        let manifest = SessionManifest {
            id: "session-1".into(),
            name: "Investigate session names".into(),
            created: 1,
            modified: 1_706_000_000_000,
            version: "4.0".into(),
            mode: None,
            model: None,
            brokk_mcp_servers: None,
            cwd: None,
        };
        let info = session_info_from_manifest(&manifest, &PathBuf::from("/tmp/cwd"));

        assert_eq!(info.session_id.to_string(), "session-1");
        assert_eq!(info.cwd, PathBuf::from("/tmp/cwd"));
        assert_eq!(info.title.as_deref(), Some("Investigate session names"));
        assert_eq!(info.updated_at, manifest.updated_at());
    }

    /// An issued `session/list` cursor round-trips to its offset; foreign or
    /// malformed cursors decode to `None` so the handler can reject them (#144).
    #[test]
    fn session_list_cursor_round_trips_and_rejects_foreign() {
        let tag = session_list_context_tag(None);
        assert_eq!(
            parse_session_list_cursor(&encode_session_list_cursor(tag, 0), tag),
            Some(0)
        );
        assert_eq!(
            parse_session_list_cursor(&encode_session_list_cursor(tag, 137), tag),
            Some(137)
        );
        // A cursor minted for a *different* cwd context must not validate here.
        let other_tag = session_list_context_tag(Some(Path::new("/repo")));
        assert_ne!(tag, other_tag, "cwd vs no-cwd contexts must differ");
        assert_eq!(
            parse_session_list_cursor(&encode_session_list_cursor(other_tag, 50), tag),
            None
        );
        // No namespace prefix -> not one of ours.
        assert_eq!(parse_session_list_cursor("137", tag), None);
        // Right prefix, non-numeric tag/offset.
        assert_eq!(parse_session_list_cursor("anvil:zz:5", tag), None);
        assert_eq!(
            parse_session_list_cursor(&format!("anvil:{tag:x}:abc"), tag),
            None
        );
        // Arbitrary garbage.
        assert_eq!(parse_session_list_cursor("garbage", tag), None);
    }

    /// Pagination yields full pages with a follow-up cursor until the final
    /// page, which omits the cursor (end-of-results). An offset past the end is
    /// an empty page, not an error (#144).
    #[test]
    fn paginate_session_list_pages_and_terminates() {
        let tag = session_list_context_tag(Some(Path::new("/repo")));
        let total = SESSION_LIST_PAGE_SIZE * 2 + 10;

        let (start, end, next) = paginate_session_list(total, 0, tag);
        assert_eq!((start, end), (0, SESSION_LIST_PAGE_SIZE));
        assert_eq!(
            next.and_then(|c| parse_session_list_cursor(&c, tag)),
            Some(SESSION_LIST_PAGE_SIZE)
        );

        let (start, end, next) = paginate_session_list(total, SESSION_LIST_PAGE_SIZE, tag);
        assert_eq!(
            (start, end),
            (SESSION_LIST_PAGE_SIZE, SESSION_LIST_PAGE_SIZE * 2)
        );
        assert_eq!(
            next.and_then(|c| parse_session_list_cursor(&c, tag)),
            Some(SESSION_LIST_PAGE_SIZE * 2)
        );

        // Final partial page: no next cursor.
        let (start, end, next) = paginate_session_list(total, SESSION_LIST_PAGE_SIZE * 2, tag);
        assert_eq!((start, end), (SESSION_LIST_PAGE_SIZE * 2, total));
        assert!(next.is_none(), "final page must omit nextCursor");

        // Offset past the end: empty page, end-of-results.
        let (start, end, next) = paginate_session_list(total, total + 100, tag);
        assert_eq!((start, end), (total, total));
        assert!(next.is_none());
    }

    /// `build_prompt_messages` for a turn that used no tools must produce
    /// the historical user/assistant pair plus the new user prompt -- no
    /// tool_call or tool messages snuck in.
    #[test]
    fn build_prompt_messages_text_only_history() {
        use crate::session::{ConversationTurn, SessionSnapshot};
        let snap = SessionSnapshot {
            cwd: std::path::PathBuf::from("/tmp/cwd"),
            mode: SessionMode::Code,
            model: "m".into(),
            history: vec![ConversationTurn {
                user_prompt: "what is rust?".into(),
                agent_response: "a language".into(),
                ..Default::default()
            }],
            reasoning_effort: None,
            idle_timeout_secs: None,
            project_instructions: String::new(),
            skills: std::sync::Arc::new(crate::skills::SkillRegistry::default()),
        };
        let msgs = build_prompt_messages(&snap, "follow up");
        // system + user(history) + assistant(history) + user(new)
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[1].role, "user");
        assert_eq!(msgs[1].text_content(), Some("what is rust?"));
        assert_eq!(msgs[2].role, "assistant");
        assert_eq!(msgs[2].text_content(), Some("a language"));
        assert_eq!(msgs[3].role, "user");
        assert_eq!(msgs[3].text_content(), Some("follow up"));
    }

    /// History with tool_exchanges must replay as user → assistant_tool_calls
    /// → N tool_results → final assistant text → new user. This is the
    /// regression #3409 fixes: without it, a session/load fed the LLM
    /// only the final answer and the model would repeat searches/reads.
    #[test]
    fn build_prompt_messages_replays_tool_exchanges() {
        use crate::session::{ConversationTurn, SessionSnapshot, ToolExchange};
        let snap = SessionSnapshot {
            cwd: std::path::PathBuf::from("/tmp/cwd"),
            mode: SessionMode::Code,
            model: "m".into(),
            history: vec![ConversationTurn {
                user_prompt: "find TODOs".into(),
                agent_response: "found 3 in src/lib.rs".into(),
                tool_exchanges: vec![
                    ToolExchange {
                        call_id: "c1".into(),
                        tool_name: "grep_search".into(),
                        arguments: r#"{"pattern":"TODO"}"#.into(),
                        result: "src/lib.rs:42: // TODO".into(),
                    },
                    ToolExchange {
                        call_id: "c2".into(),
                        tool_name: "read_file".into(),
                        arguments: r#"{"file_path":"src/lib.rs"}"#.into(),
                        result: "fn main() {}".into(),
                    },
                ],
                structured_output: None,
                summary: None,
                fragment_id: None,
            }],
            reasoning_effort: None,
            idle_timeout_secs: None,
            project_instructions: String::new(),
            skills: std::sync::Arc::new(crate::skills::SkillRegistry::default()),
        };
        let msgs = build_prompt_messages(&snap, "now fix them");

        // Expected flow: system, user, assistant(tool_calls), tool, tool,
        // assistant(text), user.
        assert_eq!(msgs.len(), 7);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[1].role, "user");
        assert_eq!(msgs[1].text_content(), Some("find TODOs"));

        // assistant_tool_calls: no content, tool_calls present, both calls
        // bundled into a single batch (the conservative collapse).
        assert_eq!(msgs[2].role, "assistant");
        assert!(msgs[2].content.is_empty());
        let calls = msgs[2].tool_calls.as_ref().expect("tool_calls present");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "c1");
        assert_eq!(calls[0].function.name, "grep_search");
        assert_eq!(calls[1].id, "c2");
        assert_eq!(calls[1].function.name, "read_file");

        // tool_result messages, paired by call_id and in original order.
        assert_eq!(msgs[3].role, "tool");
        assert_eq!(msgs[3].tool_call_id.as_deref(), Some("c1"));
        assert_eq!(msgs[3].text_content(), Some("src/lib.rs:42: // TODO"));
        assert_eq!(msgs[4].role, "tool");
        assert_eq!(msgs[4].tool_call_id.as_deref(), Some("c2"));
        assert_eq!(msgs[4].text_content(), Some("fn main() {}"));

        // Final assistant text and new user prompt.
        assert_eq!(msgs[5].role, "assistant");
        assert_eq!(msgs[5].text_content(), Some("found 3 in src/lib.rs"));
        assert_eq!(msgs[6].role, "user");
        assert_eq!(msgs[6].text_content(), Some("now fix them"));
    }

    /// Empty history: just system + the new user prompt. Establishes the
    /// `with_capacity(history.len() * 2 + 2)` lower bound.
    #[test]
    fn build_prompt_messages_empty_history() {
        use crate::session::SessionSnapshot;
        let snap = SessionSnapshot {
            cwd: std::path::PathBuf::from("/tmp/cwd"),
            mode: SessionMode::Lutz,
            model: "m".into(),
            history: vec![],
            reasoning_effort: None,
            idle_timeout_secs: None,
            project_instructions: String::new(),
            skills: std::sync::Arc::new(crate::skills::SkillRegistry::default()),
        };
        let msgs = build_prompt_messages(&snap, "hi");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[1].role, "user");
        assert_eq!(msgs[1].text_content(), Some("hi"));
    }

    #[test]
    fn build_prompt_messages_puts_project_instructions_in_user_context() {
        use crate::session::SessionSnapshot;
        let snap = SessionSnapshot {
            cwd: std::path::PathBuf::from("/tmp/cwd"),
            mode: SessionMode::Code,
            model: "m".into(),
            history: vec![],
            reasoning_effort: None,
            idle_timeout_secs: None,
            project_instructions: "Use the local style.".into(),
            skills: std::sync::Arc::new(crate::skills::SkillRegistry::default()),
        };

        let msgs = build_prompt_messages(&snap, "hi");

        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].role, "system");
        assert!(
            !msgs[0]
                .text_content()
                .expect("system prompt")
                .contains("Use the local style."),
            "project-controlled AGENTS.md content must not be system instructions"
        );
        assert_eq!(msgs[1].role, "user");
        let project_context = msgs[1].text_content().expect("project context");
        assert!(project_context.starts_with("# AGENTS.md instructions for "));
        assert!(project_context.contains("<INSTRUCTIONS>\nUse the local style.\n</INSTRUCTIONS>"));
        assert_eq!(msgs[2].role, "user");
        assert_eq!(msgs[2].text_content(), Some("hi"));
    }

    /// A turn that ended without final assistant text (e.g. tool_loop hit
    /// max_turns mid-tools, or the final LLM call was cancelled) must NOT
    /// emit an empty `assistant("")` message on replay -- several
    /// providers reject an assistant message that is both empty-content
    /// and not a tool_calls message, and even when accepted it wastes a
    /// slot. The tool_results from this turn already terminate it
    /// coherently for the LLM (#3409 review MED).
    #[test]
    fn build_prompt_messages_skips_empty_assistant_after_tools() {
        use crate::session::{ConversationTurn, SessionSnapshot, ToolExchange};
        let snap = SessionSnapshot {
            cwd: std::path::PathBuf::from("/tmp/cwd"),
            mode: SessionMode::Code,
            model: "m".into(),
            history: vec![ConversationTurn {
                user_prompt: "search".into(),
                // Empty: turn ended without final assistant text.
                agent_response: String::new(),
                tool_exchanges: vec![ToolExchange {
                    call_id: "c1".into(),
                    tool_name: "grep_search".into(),
                    arguments: r#"{"pattern":"x"}"#.into(),
                    result: "no matches".into(),
                }],
                structured_output: None,
                summary: None,
                fragment_id: None,
            }],
            reasoning_effort: None,
            idle_timeout_secs: None,
            project_instructions: String::new(),
            skills: std::sync::Arc::new(crate::skills::SkillRegistry::default()),
        };
        let msgs = build_prompt_messages(&snap, "next");

        // Expected: system, user, assistant_tool_calls, tool, user(new).
        // No trailing `assistant("")`.
        assert_eq!(msgs.len(), 5);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[1].role, "user");
        assert_eq!(msgs[2].role, "assistant");
        assert!(msgs[2].content.is_empty());
        assert!(msgs[2].tool_calls.is_some());
        assert_eq!(msgs[3].role, "tool");
        assert_eq!(msgs[4].role, "user");
        assert_eq!(msgs[4].text_content(), Some("next"));
    }

    // ---------------------------------------------------------------
    // Agent Skills integration (catalog injection, slash dispatch,
    // built-in collision precedence, command merging, payload format).
    // ---------------------------------------------------------------

    use crate::skills::{SkillMeta, SkillRegistry, SkillScope};
    use std::path::PathBuf as TestPathBuf;

    fn make_registry(skills: Vec<(&str, &str)>) -> std::sync::Arc<SkillRegistry> {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut reg = SkillRegistry::default();
        for (name, description) in skills {
            // Write a real SKILL.md so `build_skill_payload` can read it.
            let skill_dir = tmp.path().join(name);
            std::fs::create_dir_all(&skill_dir).unwrap();
            let location = skill_dir.join("SKILL.md");
            std::fs::write(
                &location,
                format!("---\nname: {name}\ndescription: {description}\n---\nBody for {name}"),
            )
            .unwrap();
            reg.insert_for_test(SkillMeta {
                name: name.to_string(),
                description: description.to_string(),
                location: location.clone(),
                skill_dir: skill_dir.clone(),
                scope: SkillScope::Project,
            });
        }
        // Leak the TempDir so files survive the test (we don't manage
        // lifetime here; the worker thread cleans up the system tmpdir).
        std::mem::forget(tmp);
        std::sync::Arc::new(reg)
    }

    #[test]
    fn build_prompt_messages_injects_catalog_when_skills_present() {
        use crate::session::SessionSnapshot;
        let snap = SessionSnapshot {
            cwd: TestPathBuf::from("/tmp/cwd"),
            mode: SessionMode::Code,
            model: "m".into(),
            history: vec![],
            reasoning_effort: None,
            idle_timeout_secs: None,
            project_instructions: String::new(),
            skills: make_registry(vec![
                ("hello-world", "Greet the user with a single short line."),
                ("pdf-processing", "Extract text from PDFs."),
            ]),
        };
        let msgs = build_prompt_messages(&snap, "hi");
        // system, catalog (user context), user(new) -> 3
        assert_eq!(msgs.len(), 3);
        let catalog = msgs[1].text_content().expect("catalog message has content");
        assert!(catalog.contains("<available_skills>"));
        assert!(catalog.contains("<name>hello-world</name>"));
        assert!(catalog.contains("<name>pdf-processing</name>"));
        // Sorted: hello-world before pdf-processing.
        let hw = catalog.find("<name>hello-world</name>").unwrap();
        let pdf = catalog.find("<name>pdf-processing</name>").unwrap();
        assert!(hw < pdf, "catalog must be alphabetically sorted");
        // Behavioral instruction tells the model to call activate_skill.
        assert!(catalog.contains("activate_skill"));
    }

    #[test]
    fn build_prompt_messages_skips_catalog_when_empty() {
        use crate::session::SessionSnapshot;
        let snap = SessionSnapshot {
            cwd: TestPathBuf::from("/tmp/cwd"),
            mode: SessionMode::Code,
            model: "m".into(),
            history: vec![],
            reasoning_effort: None,
            idle_timeout_secs: None,
            project_instructions: String::new(),
            skills: std::sync::Arc::new(SkillRegistry::default()),
        };
        let msgs = build_prompt_messages(&snap, "hi");
        // Just system + the user prompt -- no catalog message.
        assert_eq!(msgs.len(), 2);
        for m in &msgs {
            if let Some(c) = m.text_content() {
                assert!(
                    !c.contains("<available_skills>"),
                    "empty registry must not emit an empty catalog block"
                );
            }
        }
    }

    #[test]
    fn available_commands_merges_builtins_and_skills() {
        let registry = make_registry(vec![("zebra", "Z skill"), ("apple", "A skill")]);
        let cmds = available_commands(&registry);
        let names: Vec<&str> = cmds.iter().map(|c| c.name.as_str()).collect();
        // Built-ins come first in their declared order; skills follow,
        // sorted alphabetically.
        assert_eq!(
            names,
            vec![
                "context",
                "loop",
                "goal",
                "setup",
                "permissions",
                "compress",
                "mcp",
                "pr-create",
                "usage",
                "apple",
                "zebra"
            ]
        );
    }

    #[test]
    fn available_commands_hide_case_ambiguous_skill_slashes() {
        let mut reg = SkillRegistry::default();
        for name in ["Review", "REVIEW"] {
            reg.insert_for_test(SkillMeta {
                name: name.to_string(),
                description: format!("{name} skill"),
                location: TestPathBuf::from(format!("/tmp/{name}/SKILL.md")),
                skill_dir: TestPathBuf::from(format!("/tmp/{name}")),
                scope: SkillScope::Project,
            });
        }

        let cmds = available_commands(&reg);
        let names: Vec<&str> = cmds.iter().map(|c| c.name.as_str()).collect();
        assert!(!names.contains(&"Review"));
        assert!(!names.contains(&"REVIEW"));
    }

    #[test]
    fn slash_collision_with_builtin_keeps_builtin_warns() {
        // A skill named `context` must NOT shadow the `/context` builtin
        // in autocomplete (the dispatcher checks built-ins first, so the
        // slash still hits the builtin, but the duplicate command entry
        // would confuse the user).
        let registry = make_registry(vec![
            ("context", "this should be hidden"),
            ("ok-skill", "this should show"),
        ]);
        let cmds = available_commands(&registry);
        let names: Vec<&str> = cmds.iter().map(|c| c.name.as_str()).collect();
        // Built-in `context` exactly once; skill `context` dropped.
        assert_eq!(names.iter().filter(|n| **n == "context").count(), 1);
        // Non-colliding skill still appears.
        assert!(names.contains(&"ok-skill"));
    }

    #[test]
    fn slash_collision_with_builtin_is_case_insensitive() {
        let registry = make_registry(vec![
            ("Context", "this should be hidden"),
            ("ok-skill", "this should show"),
        ]);
        let cmds = available_commands(&registry);
        let names: Vec<&str> = cmds.iter().map(|c| c.name.as_str()).collect();

        assert_eq!(names.iter().filter(|n| **n == "context").count(), 1);
        assert!(!names.contains(&"Context"));
        assert!(names.contains(&"ok-skill"));
    }

    #[test]
    fn available_commands_expose_public_configuration_slashes() {
        use crate::openrouter_auth::test_support::{ENV_GUARD, EnvScope};
        let _lock = ENV_GUARD.blocking_lock();
        let _env = EnvScope::set("OPENROUTER_API_KEY", "sk-or-from-env");

        let registry = make_registry(vec![]);
        let cmds = available_commands(&registry);
        let names: Vec<&str> = cmds.iter().map(|c| c.name.as_str()).collect();

        assert!(names.contains(&"setup"));
        assert!(names.contains(&"permissions"));
        assert!(!names.contains(&"codex-login"));
        assert!(!names.contains(&"openrouter-login"));
        assert!(!names.contains(&"idle-timeout"));
        assert!(!names.contains(&"configure"));
    }

    #[tokio::test]
    async fn openrouter_setup_reports_env_owned_credentials() {
        use crate::openrouter_auth::test_support::{ENV_GUARD, EnvScope};
        let _lock = ENV_GUARD.lock().await;
        let tmp_cfg = tempfile::tempdir().unwrap();
        let _brokk = EnvScope::set("BROKK_CONFIG_HOME", tmp_cfg.path());
        let _env = EnvScope::set("OPENROUTER_API_KEY", "sk-or-from-env");

        let dump = render_openrouter_setup_help();

        assert!(
            dump.contains("OPENROUTER_API_KEY"),
            "dump must report env as active source; got:\n{dump}"
        );
        assert!(dump.contains("/setup openrouter key <your key>"));
    }

    #[tokio::test]
    async fn openrouter_setup_reports_file_owned_credentials() {
        use crate::openrouter_auth::test_support::{ENV_GUARD, EnvScope};
        let _lock = ENV_GUARD.lock().await;
        let tmp_cfg = tempfile::tempdir().unwrap();
        let _brokk = EnvScope::set("BROKK_CONFIG_HOME", tmp_cfg.path());
        let _env = EnvScope::remove("OPENROUTER_API_KEY");
        crate::openrouter_auth::write(&crate::openrouter_auth::OpenRouterAuth {
            api_key: "sk-or-on-disk".to_string(),
        })
        .unwrap();

        let dump = render_openrouter_setup_help();

        assert!(
            dump.contains("saved credentials"),
            "dump must report file as active source; got:\n{dump}"
        );
        assert!(dump.contains("/setup openrouter status"));
    }

    #[tokio::test]
    async fn openrouter_setup_reports_no_credentials() {
        use crate::openrouter_auth::test_support::{ENV_GUARD, EnvScope};
        let _lock = ENV_GUARD.lock().await;
        let tmp_cfg = tempfile::tempdir().unwrap();
        let _brokk = EnvScope::set("BROKK_CONFIG_HOME", tmp_cfg.path());
        let _env = EnvScope::remove("OPENROUTER_API_KEY");

        let dump = render_openrouter_setup_help();

        assert!(
            dump.contains("OpenRouter is not connected"),
            "dump:\n{dump}"
        );
        assert!(dump.contains("/setup openrouter key <your key>"));
    }

    #[tokio::test]
    async fn bedrock_setup_reports_env_owned_credentials() {
        use crate::openrouter_auth::test_support::{ENV_GUARD, EnvScope};
        let _lock = ENV_GUARD.lock().await;
        let tmp_cfg = tempfile::tempdir().unwrap();
        let _brokk = EnvScope::set("BROKK_CONFIG_HOME", tmp_cfg.path());
        let _env = EnvScope::set("AWS_BEARER_TOKEN_BEDROCK", "bedrock-from-env");

        let dump = render_bedrock_setup_help();

        assert!(
            dump.contains("AWS_BEARER_TOKEN_BEDROCK"),
            "dump must report env as active source; got:\n{dump}"
        );
        assert!(
            dump.contains("Unset it and restart"),
            "env-owned setup should not invite file writes; got:\n{dump}"
        );
    }

    #[tokio::test]
    async fn bedrock_setup_reports_file_owned_credentials() {
        use crate::openrouter_auth::test_support::{ENV_GUARD, EnvScope};
        let _lock = ENV_GUARD.lock().await;
        let tmp_cfg = tempfile::tempdir().unwrap();
        let _brokk = EnvScope::set("BROKK_CONFIG_HOME", tmp_cfg.path());
        let _env = EnvScope::remove("AWS_BEARER_TOKEN_BEDROCK");
        crate::bedrock_auth::write(&crate::bedrock_auth::BedrockAuth {
            bearer_token: "bedrock-on-disk".to_string(),
            region: Some("eu-west-1".to_string()),
            default_model: Some("us.anthropic.claude-sonnet-4-6".to_string()),
        })
        .unwrap();

        let dump = render_bedrock_setup_help();

        assert!(
            dump.contains("saved credentials"),
            "dump must report file as active source; got:\n{dump}"
        );
        assert!(dump.contains("/setup bedrock status"));
    }

    #[tokio::test]
    async fn bedrock_setup_reports_no_credentials() {
        use crate::openrouter_auth::test_support::{ENV_GUARD, EnvScope};
        let _lock = ENV_GUARD.lock().await;
        let tmp_cfg = tempfile::tempdir().unwrap();
        let _brokk = EnvScope::set("BROKK_CONFIG_HOME", tmp_cfg.path());
        let _env = EnvScope::remove("AWS_BEARER_TOKEN_BEDROCK");

        let dump = render_bedrock_setup_help();

        assert!(dump.contains("Bedrock is not connected"), "dump:\n{dump}");
        assert!(dump.contains("/setup bedrock key <token>"));
    }

    /// The handler short-circuits with the env-owned explanation for
    /// every subcommand when `OPENROUTER_API_KEY` is set. We assert the
    /// bare and `<key>` paths -- they're the ones that would mutate
    /// state if the early-return ever regressed. Status/disconnect are
    /// covered transitively by the same short-circuit.
    #[tokio::test]
    async fn handle_openrouter_login_short_circuits_when_env_owns() {
        use crate::openrouter_auth::test_support::{ENV_GUARD, EnvScope};
        let _lock = ENV_GUARD.lock().await;
        let tmp_cfg = tempfile::tempdir().unwrap();
        let _brokk = EnvScope::set("BROKK_CONFIG_HOME", tmp_cfg.path());
        let _env = EnvScope::set("OPENROUTER_API_KEY", "sk-or-from-env");

        let store = SessionStore::new("m".into());
        let llm = std::sync::Arc::new(crate::multi_backend::MultiBackend::new(
            None, None, None, None, None,
        ));
        let refresh = std::sync::Arc::new(tokio::sync::Mutex::new(()));

        let bare =
            handle_openrouter_login("/openrouter-login", &llm, &store, &refresh, None, None).await;
        let with_key = handle_openrouter_login(
            "/openrouter-login sk-or-rotated",
            &llm,
            &store,
            &refresh,
            None,
            None,
        )
        .await;

        for (label, msg) in [("bare", bare), ("with key", with_key)] {
            assert!(
                msg.contains("OPENROUTER_API_KEY"),
                "{label} response must explain env ownership: {msg}"
            );
            assert!(
                msg.contains("active_source: `env`"),
                "{label} response must include credential diagnostics: {msg}"
            );
        }
        // And critically: no file was written despite the candidate key.
        let path = crate::openrouter_auth::auth_path().unwrap();
        assert!(
            !path.exists(),
            "env-owned mode must not persist a key on disk; file at {path:?} should not exist"
        );
    }

    #[test]
    fn render_setup_models_filters_openrouter_catalog() {
        let catalog = vec![
            ModelMetadata::id_only("codex::chatgpt-latest"),
            ModelMetadata::id_only("ollama::llama3.1:latest"),
            ModelMetadata::id_only("openrouter::anthropic/claude-sonnet-4.5"),
            ModelMetadata::id_only("openrouter::openai/text-embedding-3-large"),
            ModelMetadata::id_only("openrouter::black-forest-labs/flux-image"),
        ];

        let out = render_setup_models(&catalog);
        assert!(out.contains("codex::chatgpt-latest"));
        assert!(out.contains("ollama::llama3.1:latest"));
        assert!(out.contains("openrouter::anthropic/claude-sonnet-4.5"));
        assert!(!out.contains("text-embedding"));
        assert!(!out.contains("flux-image"));
        assert!(out.contains("OpenRouter list is filtered"));
    }

    #[test]
    fn build_skill_payload_wraps_body_with_resources_listing() {
        let tmp = tempfile::TempDir::new().unwrap();
        let skill_dir = tmp.path().join("demo");
        std::fs::create_dir_all(skill_dir.join("scripts")).unwrap();
        std::fs::create_dir_all(skill_dir.join("references")).unwrap();
        let location = skill_dir.join("SKILL.md");
        std::fs::write(
            &location,
            "---\nname: demo\ndescription: demo skill\n---\nDo a thing.\n",
        )
        .unwrap();
        std::fs::write(skill_dir.join("scripts").join("run.sh"), "#!/bin/sh\n").unwrap();
        std::fs::write(skill_dir.join("references").join("notes.md"), "n").unwrap();

        let meta = SkillMeta {
            name: "demo".into(),
            description: "demo skill".into(),
            location,
            skill_dir: skill_dir.clone(),
            scope: SkillScope::Project,
        };
        let payload = build_skill_payload(&meta);
        assert!(payload.starts_with("<skill_content name=\"demo\">"));
        assert!(payload.contains("Do a thing."));
        // Frontmatter must be stripped.
        assert!(!payload.contains("---\nname:"));
        // Resources listed.
        assert!(payload.contains("<file>scripts/run.sh</file>"));
        assert!(payload.contains("<file>references/notes.md</file>"));
        // Skill directory + relative-path hint present.
        assert!(payload.contains(&format!("Skill directory: {}", skill_dir.display())));
        assert!(payload.ends_with("</skill_content>"));
    }

    #[test]
    fn parse_slash_command_splits_name_and_args() {
        assert_eq!(
            parse_slash_command("/hello world"),
            Some(("hello".into(), "world".into()))
        );
        assert_eq!(
            parse_slash_command("/hello"),
            Some(("hello".into(), String::new()))
        );
        assert_eq!(
            parse_slash_command("/Hello   foo bar"),
            Some(("hello".into(), "foo bar".into()))
        );
        assert_eq!(parse_slash_command("hello"), None);
        assert_eq!(parse_slash_command("/"), None);
        assert_eq!(parse_slash_command(""), None);
    }

    #[test]
    fn slash_commands_do_not_auto_rename_sessions() {
        assert!(should_auto_rename_session_from_prompt(
            "Investigate session names"
        ));
        assert!(should_auto_rename_session_from_prompt(
            "  Explain the diff  "
        ));
        assert!(!should_auto_rename_session_from_prompt(
            "/setup openrouter refresh"
        ));
        assert!(!should_auto_rename_session_from_prompt(
            "  /my-skill with args  "
        ));
    }

    /// Build a `SessionStore` with one session for the apply/render tests
    /// below. The cwd is randomized so concurrent test runs don't clobber.
    async fn make_store_with_session(default_model: &str) -> (SessionStore, String) {
        let (store, id, _cwd) = make_store_with_session_and_cwd(default_model).await;
        (store, id)
    }

    async fn make_store_with_session_and_cwd(
        default_model: &str,
    ) -> (SessionStore, String, PathBuf) {
        let store = SessionStore::new(default_model.to_string());
        let cwd =
            std::env::temp_dir().join(format!("brokk-acp-configure-{}", uuid::Uuid::new_v4()));
        let session = store.create_session(cwd.clone()).await;
        (store, session.id, cwd)
    }

    #[test]
    fn describe_always_allow_key_formats_shell_keys() {
        let repo_prefix_key = serde_json::json!({
            "tool": "run_shell_command",
            "rule": "prefix",
            "argvPrefix": ["cargo", "test"],
            "shellSandboxed": true,
        })
        .to_string();
        let legacy_key = serde_json::json!({
            "tool": "run_shell_command",
            "cwd": "/work/repo",
            "command": "cargo test",
            "shellSandboxed": true,
        })
        .to_string();

        assert_eq!(
            describe_always_allow_key(&repo_prefix_key),
            "run_shell_command prefix `cargo test` in this repo"
        );
        assert_eq!(
            describe_always_allow_key(&legacy_key),
            "run_shell_command `cargo test` in this repo"
        );
        assert_eq!(describe_always_allow_key("write_file"), "tool `write_file`");
    }

    #[tokio::test]
    async fn remembered_permissions_can_be_listed_revoked_and_cleared() {
        let (store, id) = make_store_with_session("m").await;
        let repo_key = serde_json::json!({
            "tool": "run_shell_command",
            "rule": "prefix",
            "argvPrefix": ["cargo", "test"],
            "shellSandboxed": true,
        })
        .to_string();
        store.add_always_allow(&id, "write_file").await;
        store.add_always_allow(&id, &repo_key).await;

        let listed = render_always_allowed_permissions(&store, &id).await;
        assert!(listed.contains("1. tool `write_file`"), "{listed}");
        assert!(
            listed.contains("2. run_shell_command prefix `cargo test` in this repo"),
            "{listed}"
        );

        let revoked = revoke_always_allowed_permission(&store, &id, "1").await;
        assert_eq!(revoked, "Forgot Always allow approval: tool `write_file`");
        assert!(
            !store
                .is_any_always_allowed(&id, &["write_file".to_string()])
                .await
        );
        assert!(
            store
                .is_any_always_allowed(&id, std::slice::from_ref(&repo_key))
                .await
        );

        let missing = revoke_always_allowed_permission(&store, &id, "99").await;
        assert!(missing.contains("No remembered Always allow approval numbered `99`"));

        let cleared = clear_always_allowed_permissions(&store, &id).await;
        assert_eq!(cleared, "Forgot 1 remembered Always allow approval.");
        assert_eq!(
            render_always_allowed_permissions(&store, &id).await,
            "No remembered Always allow approvals."
        );
    }

    #[tokio::test]
    async fn apply_config_option_sets_permission_mode() {
        use agent_client_protocol::schema::{SessionConfigKind, SessionConfigSelectOptions};

        let (store, id) = make_store_with_session("m").await;
        let outcome = apply_config_option(&store, &id, PERMISSION_CONFIG_ID, "auto")
            .await
            .expect("permission mode update");
        assert!(outcome.cleared_reasoning.is_none());
        let pm = store.permission_mode(&id).await.expect("session present");
        assert_eq!(pm, PermissionMode::Auto);

        let permission_option = outcome
            .updated_options
            .iter()
            .find(|opt| opt.id.to_string() == PERMISSION_CONFIG_ID)
            .expect("permission option advertised");
        match &permission_option.kind {
            SessionConfigKind::Select(select) => {
                assert_eq!(select.current_value.to_string(), "auto");
                match &select.options {
                    SessionConfigSelectOptions::Ungrouped(options) => {
                        assert!(options.iter().any(|opt| opt.value.to_string() == "auto"));
                    }
                    other => panic!("expected ungrouped permission options, got {other:?}"),
                }
            }
            other => panic!("expected select permission option, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn apply_config_option_sets_behavior_mode() {
        let (store, id) = make_store_with_session("m").await;
        apply_config_option(&store, &id, BEHAVIOR_CONFIG_ID, "PLAN")
            .await
            .expect("behavior mode update");
        let snap = store
            .snapshot(&id, &std::env::temp_dir())
            .await
            .expect("session present");
        assert_eq!(snap.mode, SessionMode::Plan);
    }

    #[tokio::test]
    async fn apply_config_option_sets_model_when_catalog_empty() {
        let (store, id) = make_store_with_session("initial").await;
        // Empty catalog must accept any id so a manually-configured
        // backend still works.
        apply_config_option(&store, &id, MODEL_CONFIG_ID, "custom/model")
            .await
            .expect("model update");
        let snap = store
            .snapshot(&id, &std::env::temp_dir())
            .await
            .expect("session present");
        assert_eq!(snap.model, "custom/model");
    }

    #[tokio::test]
    async fn apply_config_option_reports_model_persistence_failure() {
        let (store, id, cwd) = make_store_with_session_and_cwd("initial").await;
        std::fs::remove_dir_all(&cwd).expect("remove persisted session zip parent");

        let err = apply_config_option(&store, &id, MODEL_CONFIG_ID, "custom/model")
            .await
            .expect_err("missing session zip should surface as persistence failure");

        match err {
            ConfigApplyError::PersistFailed { details } => {
                let lower = details.to_lowercase();
                assert!(
                    lower.contains("cannot resolve")
                        || lower.contains("failed")
                        || lower.contains("no such file")
                        || lower.contains("file not found"),
                    "unexpected persistence details: {details}"
                );
            }
            other => panic!("expected PersistFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn apply_config_option_rejects_unknown_model_when_catalog_known() {
        let (store, id) = make_store_with_session("initial").await;
        store
            .set_available_models(vec![
                ModelMetadata::id_only("known-1"),
                ModelMetadata::id_only("known-2"),
            ])
            .await;
        let err = apply_config_option(&store, &id, MODEL_CONFIG_ID, "ghost")
            .await
            .expect_err("ghost model is not in the catalog");
        match err {
            ConfigApplyError::InvalidValue { supported, .. } => {
                assert_eq!(
                    supported,
                    vec!["known-1".to_string(), "known-2".to_string()]
                );
            }
            other => panic!("expected InvalidValue, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn apply_config_option_clears_reasoning_when_model_drops_it() {
        use crate::llm_client::ReasoningLevelPreset;
        let (store, id) = make_store_with_session("model-a").await;
        // model-a publishes a "high" preset; model-b publishes nothing,
        // so swapping to it forces the store to drop the user's pick.
        store
            .set_available_models(vec![
                ModelMetadata {
                    id: "model-a".into(),
                    default_reasoning_level: Some("high".into()),
                    supported_reasoning_levels: vec![ReasoningLevelPreset {
                        effort: "high".into(),
                        description: "High".into(),
                    }],
                    supports_images: None,
                    context_length: None,
                    pricing: None,
                },
                ModelMetadata::id_only("model-b"),
            ])
            .await;
        apply_config_option(&store, &id, REASONING_EFFORT_CONFIG_ID, "high")
            .await
            .expect("set reasoning effort");
        let outcome = apply_config_option(&store, &id, MODEL_CONFIG_ID, "model-b")
            .await
            .expect("swap model");
        assert_eq!(outcome.cleared_reasoning.as_deref(), Some("high"));
    }

    #[tokio::test]
    async fn apply_config_option_sets_reasoning_off_and_omits_default() {
        use crate::llm_client::ReasoningLevelPreset;
        use agent_client_protocol::schema::{SessionConfigKind, SessionConfigSelectOptions};

        let (store, id) = make_store_with_session("model-a").await;
        store
            .set_available_models(vec![ModelMetadata {
                id: "model-a".into(),
                default_reasoning_level: Some("medium".into()),
                supported_reasoning_levels: vec![
                    ReasoningLevelPreset {
                        effort: "low".into(),
                        description: "Low".into(),
                    },
                    ReasoningLevelPreset {
                        effort: "medium".into(),
                        description: "Medium".into(),
                    },
                    ReasoningLevelPreset {
                        effort: "high".into(),
                        description: "High".into(),
                    },
                ],
                supports_images: None,
                context_length: None,
                pricing: None,
            }])
            .await;

        let outcome = apply_config_option(&store, &id, REASONING_EFFORT_CONFIG_ID, "off")
            .await
            .expect("off is a valid reasoning selection");
        let session = store
            .get_session(&id, &std::env::temp_dir())
            .await
            .expect("session present");
        assert_eq!(
            session.selected_reasoning_effort.as_deref(),
            Some(REASONING_EFFORT_OFF_VALUE)
        );
        let snap = store
            .snapshot(&id, &std::env::temp_dir())
            .await
            .expect("session present");
        assert_eq!(
            snap.reasoning_effort, None,
            "explicit off must not fall back to model default"
        );

        let reasoning_option = outcome
            .updated_options
            .iter()
            .find(|opt| opt.id.to_string() == REASONING_EFFORT_CONFIG_ID)
            .expect("reasoning option still advertised");
        match &reasoning_option.kind {
            SessionConfigKind::Select(select) => {
                assert_eq!(select.current_value.to_string(), REASONING_EFFORT_OFF_VALUE);
                match &select.options {
                    SessionConfigSelectOptions::Ungrouped(options) => {
                        assert!(
                            options
                                .iter()
                                .any(|opt| opt.value.to_string() == "(default)")
                        );
                        assert!(options.iter().any(|opt| opt.value.to_string() == "off"));
                        assert!(options.iter().any(|opt| opt.value.to_string() == "high"));
                    }
                    other => panic!("expected ungrouped reasoning options, got {other:?}"),
                }
            }
            other => panic!("expected select reasoning option, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn apply_config_option_accepts_reasoning_off_for_model_without_presets() {
        let (store, id) = make_store_with_session("plain-model").await;
        store
            .set_available_models(vec![ModelMetadata::id_only("plain-model")])
            .await;

        apply_config_option(&store, &id, REASONING_EFFORT_CONFIG_ID, "off")
            .await
            .expect("off sends no provider parameter and should always be valid");
        let session = store
            .get_session(&id, &std::env::temp_dir())
            .await
            .expect("session present");
        assert_eq!(
            session.selected_reasoning_effort.as_deref(),
            Some(REASONING_EFFORT_OFF_VALUE)
        );
        let snap = store
            .snapshot(&id, &std::env::temp_dir())
            .await
            .expect("session present");
        assert_eq!(snap.reasoning_effort, None);
    }

    #[tokio::test]
    async fn apply_config_option_rejects_reasoning_effort_for_model_without_presets() {
        let (store, id) = make_store_with_session("plain-model").await;
        store
            .set_available_models(vec![ModelMetadata::id_only("plain-model")])
            .await;

        let err = apply_config_option(&store, &id, REASONING_EFFORT_CONFIG_ID, "high")
            .await
            .expect_err("known model without presets cannot accept reasoning effort");

        match err {
            ConfigApplyError::InvalidValue { reason, supported } => {
                assert!(reason.contains("plain-model"));
                assert!(supported.is_empty());
            }
            other => panic!("expected InvalidValue, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn apply_config_option_rejects_invalid_permission_mode() {
        let (store, id) = make_store_with_session("m").await;
        let err = apply_config_option(&store, &id, PERMISSION_CONFIG_ID, "bogus")
            .await
            .expect_err("bogus is not a permission mode");
        match err {
            ConfigApplyError::InvalidValue { reason, supported } => {
                assert!(reason.contains("bogus"));
                assert!(supported.contains(&"acceptEdits".to_string()));
            }
            other => panic!("expected InvalidValue, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn apply_config_option_rejects_unknown_key() {
        let (store, id) = make_store_with_session("m").await;
        let err = apply_config_option(&store, &id, "no_such_knob", "value")
            .await
            .expect_err("unknown key");
        assert!(matches!(err, ConfigApplyError::UnknownConfigId));
    }

    #[tokio::test]
    async fn apply_config_option_reports_unknown_session() {
        let store = SessionStore::new("m".into());
        let err = apply_config_option(&store, "no-session", PERMISSION_CONFIG_ID, "default")
            .await
            .expect_err("session does not exist");
        assert!(matches!(err, ConfigApplyError::UnknownSession));
    }

    #[test]
    fn setup_unknown_config_key_error_lists_supported_ids() {
        let out = ConfigApplyError::UnknownConfigId.human_message();
        assert!(out.contains("unknown config key"));
        for key in CONFIGURE_KNOWN_KEYS {
            assert!(out.contains(key), "missing key `{key}` in error: {out}");
        }
    }

    // -----------------------------------------------------------------------
    // Per-turn summary substitution in build_prompt_messages
    // -----------------------------------------------------------------------

    /// When a `ConversationTurn` has a `summary`, the prompt must
    /// contain that summary wrapped in `<conversation_summary>` tags
    /// in place of the verbatim user/tool/assistant messages for that
    /// turn. Mirrors how Brokk's `TaskEntry.summary` substitutes into
    /// the next prompt.
    #[test]
    fn build_prompt_messages_substitutes_summary_for_turn() {
        use crate::session::{ConversationTurn, SessionSnapshot};
        let snap = SessionSnapshot {
            cwd: std::path::PathBuf::from("/tmp/cwd"),
            mode: SessionMode::Code,
            model: "m".into(),
            history: vec![
                ConversationTurn {
                    user_prompt: "OLD user".into(),
                    agent_response: "OLD agent".into(),
                    summary: Some("- file foo.rs touched\n- decision X".into()),
                    ..Default::default()
                },
                ConversationTurn {
                    user_prompt: "RECENT user".into(),
                    agent_response: "RECENT agent".into(),
                    ..Default::default()
                },
            ],
            reasoning_effort: None,
            idle_timeout_secs: None,
            project_instructions: String::new(),
            skills: std::sync::Arc::new(crate::skills::SkillRegistry::default()),
        };
        let msgs = build_prompt_messages(&snap, "next");
        // system + summary(turn 0) + user(turn 1) + assistant(turn 1) + user(new) = 5
        assert_eq!(msgs.len(), 5);
        let bodies: Vec<&str> = msgs.iter().filter_map(|m| m.text_content()).collect();
        // Verbatim OLD content must NOT appear -- the summary replaces it.
        assert!(
            bodies.iter().all(|b| !b.contains("OLD user")),
            "verbatim old user prompt leaked: {bodies:?}"
        );
        assert!(
            bodies.iter().all(|b| !b.contains("OLD agent")),
            "verbatim old agent response leaked: {bodies:?}"
        );
        // Summary block must appear with the tags.
        assert!(
            bodies
                .iter()
                .any(|b| b.contains("<conversation_summary>") && b.contains("- file foo.rs"))
        );
        // The unsummarized recent turn must still come through verbatim.
        assert!(bodies.iter().any(|b| b.contains("RECENT user")));
        assert!(bodies.iter().any(|b| b.contains("RECENT agent")));
    }

    /// An empty / whitespace-only summary must not produce an empty
    /// `<conversation_summary>` message -- the turn should be replayed
    /// verbatim instead. Otherwise a corrupted summary could silently
    /// drop the turn from the prompt.
    #[test]
    fn build_prompt_messages_falls_back_to_verbatim_when_summary_blank() {
        use crate::session::{ConversationTurn, SessionSnapshot};
        let snap = SessionSnapshot {
            cwd: std::path::PathBuf::from("/tmp/cwd"),
            mode: SessionMode::Code,
            model: "m".into(),
            history: vec![ConversationTurn {
                user_prompt: "verbatim user".into(),
                agent_response: "verbatim agent".into(),
                summary: Some("   \n  ".into()),
                ..Default::default()
            }],
            reasoning_effort: None,
            idle_timeout_secs: None,
            project_instructions: String::new(),
            skills: std::sync::Arc::new(crate::skills::SkillRegistry::default()),
        };
        let msgs = build_prompt_messages(&snap, "next");
        let bodies: Vec<&str> = msgs.iter().filter_map(|m| m.text_content()).collect();
        assert!(bodies.iter().any(|b| b.contains("verbatim user")));
        assert!(bodies.iter().any(|b| b.contains("verbatim agent")));
        // No empty summary block leaked through.
        assert!(bodies.iter().all(|b| !b.contains("<conversation_summary>")));
    }

    /// `/setup sandbox` round-trip: bare reports current state, `off`
    /// flips the flag, `on` flips it back, and an unknown choice neither
    /// mutates state nor panics. Asserts both the user-facing string and
    /// the store's observable side effect so a future refactor that
    /// drops one without the other gets caught.
    #[tokio::test]
    async fn handle_setup_sandbox_round_trip() {
        use crate::sandbox_backend::SandboxMode;
        let (store, id) = make_store_with_session("m").await;

        // Bare: reports the effective default and surfaces the usage hints.
        let bare = handle_setup_sandbox(&store, &id, "").await;
        assert!(bare.contains("currently `os`"), "got: {bare}");
        assert!(bare.contains("/setup sandbox default"), "got: {bare}");
        assert!(bare.contains("/setup sandbox wasm"), "got: {bare}");
        assert_eq!(store.sandbox_mode(&id).await, Some(None));

        // `off` flips the flag and confirms the per-call prompt is
        // still in play -- the message wording is part of the contract.
        let off = handle_setup_sandbox(&store, &id, "off").await;
        assert!(
            off.contains("set to `off`") || off.contains("No sandboxing"),
            "got: {off}"
        );
        assert!(off.contains("permission prompts"), "got: {off}");
        assert_eq!(store.sandbox_mode(&id).await, Some(Some(SandboxMode::Off)));

        // `os` is a real override, distinct from clearing to default.
        let os = handle_setup_sandbox(&store, &id, "os").await;
        assert!(os.contains("set to `os`"), "got: {os}");
        assert_eq!(store.sandbox_mode(&id).await, Some(Some(SandboxMode::Os)));

        // `on` flips it back.
        let on = handle_setup_sandbox(&store, &id, "on").await;
        assert!(
            on.contains("reset to default") || on.contains("default"),
            "got: {on}"
        );
        assert_eq!(store.sandbox_mode(&id).await, Some(None));

        // `status` reports without mutating.
        assert!(store.set_sandbox_mode(&id, Some(SandboxMode::Off)).await);
        let status = handle_setup_sandbox(&store, &id, "status").await;
        assert!(status.contains("`off`"), "got: {status}");
        assert_eq!(store.sandbox_mode(&id).await, Some(Some(SandboxMode::Off)));

        // `wasm` either sets sandbox mode or reports that the build was
        // compiled without wasm support.
        let wasm = handle_setup_sandbox(&store, &id, "wasm").await;
        if crate::sandbox_backend::wasm_sandbox_compiled() {
            assert!(
                wasm.contains("set to `wasm`") || wasm.contains("WASM sandbox"),
                "got: {wasm}"
            );
            assert_eq!(store.sandbox_mode(&id).await, Some(Some(SandboxMode::Wasm)));
        } else {
            assert!(wasm.contains("not compiled into this build"), "got: {wasm}");
            assert_eq!(store.sandbox_mode(&id).await, Some(Some(SandboxMode::Off)));
        }

        // Unknown choice is rejected and leaves state untouched.
        let bad = handle_setup_sandbox(&store, &id, "maybe").await;
        assert!(
            bad.contains("Unknown choice") || bad.contains("Unknown sandbox choice"),
            "got: {bad}"
        );
        assert_eq!(
            store.sandbox_mode(&id).await,
            Some(Some(if crate::sandbox_backend::wasm_sandbox_compiled() {
                SandboxMode::Wasm
            } else {
                SandboxMode::Off
            }))
        );

        // Unknown session id is surfaced rather than silently noop'd.
        let missing = handle_setup_sandbox(&store, "no-such", "off").await;
        assert!(missing.contains("unknown session"), "got: {missing}");
    }
}
