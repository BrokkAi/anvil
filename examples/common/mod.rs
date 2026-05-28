use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::{
    ClientCapabilities, ConfigOptionUpdate, ContentBlock, FileSystemCapabilities,
    InitializeRequest, NewSessionRequest, PermissionOptionKind, PromptRequest, ProtocolVersion,
    RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    SelectedPermissionOutcome, SessionNotification, SessionUpdate, SetSessionConfigOptionRequest,
    StopReason, TextContent,
};
use agent_client_protocol::{AcpAgent, Agent, Client, ConnectionTo};
use anyhow::{Context, Result};

const PERMISSION_CONFIG_ID: &str = "permission_mode";
const BEHAVIOR_CONFIG_ID: &str = "behavior_mode";

#[derive(Debug, Clone, Copy)]
pub enum PermissionPolicy {
    ReadOnly,
}

#[derive(Debug, Clone)]
pub struct AnvilRun {
    pub agent: String,
    pub cwd: PathBuf,
    pub behavior_mode: Option<&'static str>,
    pub permission_policy: PermissionPolicy,
    pub echo: bool,
}

impl AnvilRun {
    pub fn read_only(agent: String, cwd: PathBuf) -> Self {
        Self {
            agent,
            cwd,
            behavior_mode: Some("ASK"),
            permission_policy: PermissionPolicy::ReadOnly,
            echo: true,
        }
    }
}

pub fn default_agent_command() -> String {
    if let Ok(command) = std::env::var("ANVIL_AGENT") {
        return command;
    }

    let debug_binary = PathBuf::from("target/debug/anvil");
    if debug_binary.exists() {
        return debug_binary.display().to_string();
    }

    "cargo run --quiet --bin anvil --".to_string()
}

pub async fn run_prompt(config: AnvilRun, prompt: String) -> Result<String> {
    let output = Arc::new(Mutex::new(String::new()));
    let output_for_notifications = output.clone();
    let echo = config.echo;
    let permission_policy = config.permission_policy;
    let agent = AcpAgent::from_str(&config.agent).context("parse ACP agent command")?;

    Client
        .builder()
        .on_receive_notification(
            async move |notification: SessionNotification, _cx| {
                handle_update(notification.update, &output_for_notifications, echo);
                Ok(())
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            async move |request: RequestPermissionRequest, responder, _cx| {
                responder.respond(permission_response(request, permission_policy))
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(agent, |connection: ConnectionTo<Agent>| async move {
            initialize(&connection).await?;
            let session = connection
                .send_request(NewSessionRequest::new(config.cwd.clone()))
                .block_task()
                .await?;

            if let Some(mode) = config.behavior_mode {
                set_session_config(
                    &connection,
                    session.session_id.clone(),
                    BEHAVIOR_CONFIG_ID,
                    mode,
                )
                .await?;
            }

            let permission = match config.permission_policy {
                PermissionPolicy::ReadOnly => "readOnly",
            };
            set_session_config(
                &connection,
                session.session_id.clone(),
                PERMISSION_CONFIG_ID,
                permission,
            )
            .await?;

            let prompt_response = connection
                .send_request(PromptRequest::new(
                    session.session_id,
                    vec![ContentBlock::Text(TextContent::new(prompt))],
                ))
                .block_task()
                .await?;
            if !matches!(prompt_response.stop_reason, StopReason::EndTurn) {
                return Err(agent_client_protocol::Error::new(
                    -32603,
                    format!(
                        "agent stopped before completing the turn: {:?}",
                        prompt_response.stop_reason
                    ),
                ));
            }

            Ok(())
        })
        .await
        .context("run ACP client")?;

    let text = output.lock().expect("output lock poisoned").clone();
    Ok(text.trim().to_string())
}

async fn initialize(connection: &ConnectionTo<Agent>) -> agent_client_protocol::Result<()> {
    let request = InitializeRequest::new(ProtocolVersion::V1).client_capabilities(
        ClientCapabilities::new()
            .fs(FileSystemCapabilities::new()
                .read_text_file(false)
                .write_text_file(false))
            .terminal(false),
    );
    connection.send_request(request).block_task().await?;
    Ok(())
}

async fn set_session_config(
    connection: &ConnectionTo<Agent>,
    session_id: agent_client_protocol::schema::SessionId,
    key: &'static str,
    value: &'static str,
) -> agent_client_protocol::Result<()> {
    connection
        .send_request(SetSessionConfigOptionRequest::new(session_id, key, value))
        .block_task()
        .await?;
    Ok(())
}

fn handle_update(update: SessionUpdate, output: &Arc<Mutex<String>>, echo: bool) {
    match update {
        SessionUpdate::AgentMessageChunk(chunk) => {
            if let ContentBlock::Text(text) = chunk.content {
                if echo {
                    print!("{}", text.text);
                }
                output
                    .lock()
                    .expect("output lock poisoned")
                    .push_str(&text.text);
            }
        }
        SessionUpdate::AgentThoughtChunk(_) => {}
        _ if !echo => {}
        SessionUpdate::ToolCall(call) => {
            eprintln!("tool: {}", call.title);
        }
        SessionUpdate::ToolCallUpdate(update) => {
            if let Some(title) = update.fields.title {
                eprintln!("tool update: {title}");
            }
        }
        SessionUpdate::Plan(plan) => {
            eprintln!("plan: {}", plan.entries.len());
        }
        SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate { .. })
        | SessionUpdate::UserMessageChunk(_)
        | SessionUpdate::AvailableCommandsUpdate(_)
        | SessionUpdate::CurrentModeUpdate(_)
        | SessionUpdate::SessionInfoUpdate(_)
        | SessionUpdate::UsageUpdate(_) => {}
        _ => {}
    }
}

fn permission_response(
    request: RequestPermissionRequest,
    policy: PermissionPolicy,
) -> RequestPermissionResponse {
    let allow = match policy {
        PermissionPolicy::ReadOnly => false,
    };
    let preferred = request.options.iter().find(|option| {
        matches!(
            (allow, option.kind),
            (
                true,
                PermissionOptionKind::AllowOnce | PermissionOptionKind::AllowAlways
            ) | (
                false,
                PermissionOptionKind::RejectOnce | PermissionOptionKind::RejectAlways
            )
        )
    });
    match preferred {
        Some(option) => RequestPermissionResponse::new(RequestPermissionOutcome::Selected(
            SelectedPermissionOutcome::new(option.option_id.clone()),
        )),
        None => RequestPermissionResponse::new(RequestPermissionOutcome::Cancelled),
    }
}

pub fn run_gh(args: &[&str]) -> Result<String> {
    run_command("gh", args)
}

fn run_command(program: &str, args: &[&str]) -> Result<String> {
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("run {program} {}", args.join(" ")))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("{program} {} failed: {}", args.join(" "), stderr.trim());
    }
}
