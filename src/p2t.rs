use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::llm_client::{ChatContentPart, ChatMessage, FunctionCall, ToolCall};

pub(crate) const PATCHES_TO_TRACES_ENV: &str = "BRK_PATCHES_TO_TRACES";
const P2T_CONFIG_ENV: &str = "BRK_P2T_CONFIG";

pub(crate) fn env_var_truthy(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct P2tConfig {
    pub prefix_steps: Option<PathBuf>,
    pub forced_first_step: Option<ForcedStep>,
    pub max_steps: usize,
    pub snapshot_dir: Option<PathBuf>,
    pub temperature: Option<f64>,
    pub step_trace_out: PathBuf,
}

#[derive(Debug, Deserialize)]
struct P2tConfigFile {
    prefix_steps: Option<PathBuf>,
    forced_first_step: Option<ForcedStep>,
    max_steps: usize,
    snapshot_dir: Option<PathBuf>,
    temperature: Option<f64>,
    step_trace_out: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub(crate) struct PrefixStep {
    #[serde(default)]
    pub assistant_text: String,
    #[serde(default)]
    pub tool_calls: Vec<PrefixToolCall>,
    #[serde(default)]
    pub results: Vec<PrefixToolResult>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub(crate) struct ForcedStep {
    #[serde(default)]
    pub assistant_text: String,
    #[serde(default)]
    pub tool_calls: Vec<PrefixToolCall>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub(crate) struct PrefixToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub(crate) struct PrefixToolResult {
    pub call_id: String,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum P2tStopReason {
    WindowEnd,
    Finished,
}

impl P2tStopReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::WindowEnd => "p2t_window_end",
            Self::Finished => "p2t_finished",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct StepTraceRecord {
    #[serde(rename = "type")]
    pub record_type: &'static str,
    pub step: usize,
    pub forced: bool,
    pub assistant_text: String,
    pub tool_calls: Vec<PrefixToolCall>,
    pub results: Vec<PrefixToolResult>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SnapshotPlan {
    pub dest: PathBuf,
    pub link_dest: Option<PathBuf>,
}

pub(crate) fn load_config_from_env(train_bifrost_enabled: bool) -> Result<Option<P2tConfig>> {
    let enabled = env_var_truthy(PATCHES_TO_TRACES_ENV);
    if !enabled {
        return Ok(None);
    }
    if train_bifrost_enabled {
        bail!("BRK_PATCHES_TO_TRACES and BRK_TRAIN_BIFROST cannot both be enabled");
    }
    let path = std::env::var(P2T_CONFIG_ENV)
        .with_context(|| format!("{P2T_CONFIG_ENV} must be set when {PATCHES_TO_TRACES_ENV}=1"))?;
    load_config(Path::new(&path)).map(Some)
}

fn load_config(path: &Path) -> Result<P2tConfig> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let parsed: P2tConfigFile = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    if parsed
        .snapshot_dir
        .as_ref()
        .is_some_and(|path| !path.is_absolute())
    {
        bail!("snapshot_dir must be an absolute path when set");
    }
    Ok(P2tConfig {
        prefix_steps: parsed.prefix_steps,
        forced_first_step: parsed.forced_first_step,
        max_steps: parsed.max_steps,
        snapshot_dir: parsed.snapshot_dir,
        temperature: parsed.temperature,
        step_trace_out: parsed.step_trace_out,
    })
}

pub(crate) fn load_prefix_steps(path: &Path) -> Result<Vec<PrefixStep>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let mut steps = Vec::new();
    for (line_no, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let step: PrefixStep = serde_json::from_str(trimmed).with_context(|| {
            format!(
                "failed to parse prefix step JSON on line {} of {}",
                line_no + 1,
                path.display()
            )
        })?;
        steps.push(step);
    }
    Ok(steps)
}

pub(crate) fn append_prefix_messages(messages: &mut Vec<ChatMessage>, steps: &[PrefixStep]) {
    messages.extend(prefix_steps_to_messages(steps));
}

pub(crate) fn prefix_steps_to_messages(steps: &[PrefixStep]) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    for step in steps {
        if !step.tool_calls.is_empty() {
            messages.push(assistant_message_with_tool_calls(
                &step.assistant_text,
                &step.tool_calls,
            ));
        } else if !step.assistant_text.is_empty() {
            messages.push(ChatMessage::assistant(step.assistant_text.clone()));
        }

        for result in &step.results {
            let tool_name = step
                .tool_calls
                .iter()
                .find(|call| call.id == result.call_id)
                .map(|call| call.name.clone())
                .unwrap_or_default();
            messages.push(ChatMessage::tool_result(
                &result.call_id,
                tool_name,
                &result.content,
            ));
        }
    }
    messages
}

pub(crate) fn forced_step_to_message(step: &ForcedStep) -> ChatMessage {
    if step.tool_calls.is_empty() {
        ChatMessage::assistant(step.assistant_text.clone())
    } else {
        assistant_message_with_tool_calls(&step.assistant_text, &step.tool_calls)
    }
}

pub(crate) fn forced_step_to_tool_calls(step: &ForcedStep) -> Vec<ToolCall> {
    tool_calls_from_prefix(&step.tool_calls)
}

fn assistant_message_with_tool_calls(
    assistant_text: &str,
    tool_calls: &[PrefixToolCall],
) -> ChatMessage {
    ChatMessage {
        role: "assistant".to_string(),
        content: if !assistant_text.is_empty() {
            vec![ChatContentPart::text(assistant_text.to_string())]
        } else {
            Vec::new()
        },
        tool_calls: Some(tool_calls_from_prefix(tool_calls)),
        tool_call_id: None,
        name: None,
    }
}

fn tool_calls_from_prefix(tool_calls: &[PrefixToolCall]) -> Vec<ToolCall> {
    tool_calls
        .iter()
        .map(|call| ToolCall {
            id: call.id.clone(),
            r#type: "function".to_string(),
            function: FunctionCall {
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            },
        })
        .collect()
}

pub(crate) fn p2t_initial_builtin_tools() -> HashSet<String> {
    ["write_file", "edit", "list_directory"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

pub(crate) fn p2t_post_edit_builtin_tools() -> HashSet<String> {
    let mut tools = p2t_initial_builtin_tools();
    tools.insert("run_shell_command".to_string());
    tools
}

pub(crate) fn prefix_unlocks_shell(steps: &[PrefixStep]) -> bool {
    steps.iter().any(prefix_step_has_successful_file_change)
}

fn prefix_step_has_successful_file_change(step: &PrefixStep) -> bool {
    step.tool_calls.iter().any(|call| {
        matches!(call.name.as_str(), "edit" | "write_file")
            && step
                .results
                .iter()
                .any(|result| result.call_id == call.id && !tool_result_failed(&result.content))
    })
}

pub(crate) fn tool_result_failed(result: &str) -> bool {
    result.starts_with("Error:") || result.starts_with("Internal error:")
}

pub(crate) fn stop_reason_after_step(
    steps_completed: usize,
    max_steps: usize,
    tool_calls_len: usize,
) -> Option<P2tStopReason> {
    if tool_calls_len == 0 {
        Some(P2tStopReason::Finished)
    } else if steps_completed >= max_steps {
        Some(P2tStopReason::WindowEnd)
    } else {
        None
    }
}

pub(crate) fn append_step_trace(path: &Path, record: &StepTraceRecord) {
    append_jsonl(path, record);
}

pub(crate) fn append_snapshot_error_trace(path: &Path, step: usize, error: &str) {
    #[derive(Serialize)]
    struct SnapshotErrorRecord<'a> {
        #[serde(rename = "type")]
        record_type: &'a str,
        step: usize,
        error: &'a str,
    }

    append_jsonl(
        path,
        &SnapshotErrorRecord {
            record_type: "snapshot_error",
            step,
            error,
        },
    );
}

pub(crate) fn append_window_start_trace(
    path: &Path,
    messages: &[ChatMessage],
    tools: &[crate::llm_client::ToolDefinition],
) {
    #[derive(Serialize)]
    struct WindowStartRecord<'a> {
        #[serde(rename = "type")]
        record_type: &'a str,
        messages: &'a [ChatMessage],
        tool_names: Vec<&'a str>,
    }

    append_jsonl(
        path,
        &WindowStartRecord {
            record_type: "window_start",
            messages,
            tool_names: tools
                .iter()
                .map(|tool| tool.function.name.as_str())
                .collect(),
        },
    );
}

pub(crate) fn append_window_end_trace(path: &Path, reason: P2tStopReason, steps: usize) {
    #[derive(Serialize)]
    struct WindowEndRecord<'a> {
        #[serde(rename = "type")]
        record_type: &'a str,
        reason: &'a str,
        steps: usize,
    }

    append_jsonl(
        path,
        &WindowEndRecord {
            record_type: "window_end",
            reason: reason.as_str(),
            steps,
        },
    );
}

pub(crate) fn snapshot_plan(
    snapshot_dir: &Path,
    step: usize,
    previous_exists: bool,
) -> SnapshotPlan {
    SnapshotPlan {
        dest: snapshot_dir.join(format!("step-{step}")),
        link_dest: (step > 0 && previous_exists)
            .then(|| snapshot_dir.join(format!("step-{}", step - 1))),
    }
}

pub(crate) fn snapshot_workspace(cwd: &Path, snapshot_dir: &Path, step: usize) -> Result<()> {
    std::fs::create_dir_all(snapshot_dir)
        .with_context(|| format!("failed to create {}", snapshot_dir.display()))?;
    let previous_exists = step > 0 && snapshot_dir.join(format!("step-{}", step - 1)).exists();
    let plan = snapshot_plan(snapshot_dir, step, previous_exists);

    let mut command = Command::new("rsync");
    command
        .arg("-a")
        .arg("--delete")
        .arg("--exclude")
        .arg("/.git");
    if let Some(link_dest) = plan.link_dest.as_ref() {
        command.arg(format!("--link-dest={}", link_dest.display()));
    }
    command.arg(format!("{}/", cwd.display()));
    command.arg(format!("{}/", plan.dest.display()));

    let output = command
        .output()
        .with_context(|| format!("failed to spawn rsync for step {step}"))?;
    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        "rsync produced no output".to_string()
    };
    bail!(
        "rsync step {step} failed with status {}: {detail}",
        output
            .status
            .code()
            .map_or_else(|| "signal".to_string(), |code| code.to_string())
    );
}

fn append_jsonl(path: &Path, record: &impl Serialize) {
    let line = match serde_json::to_string(record) {
        Ok(line) => line,
        Err(e) => {
            tracing::warn!("failed to serialize P2T trace record: {e:#}");
            return;
        }
    };

    let result = (|| -> std::io::Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        writeln!(file, "{line}")?;
        Ok(())
    })();

    if let Err(e) = result {
        tracing::warn!(
            "failed to append P2T trace record to {}: {e:#}",
            path.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openrouter_auth::test_support::{ENV_GUARD, EnvScope};

    #[test]
    fn load_config_accepts_forced_first_step() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            serde_json::json!({
                "prefix_steps": null,
                "forced_first_step": {
                    "assistant_text": "plan",
                    "tool_calls": [{"id":"c1","name":"edit","arguments":"{}"}]
                },
                "max_steps": 10,
                "snapshot_dir": "/tmp/p2t-snaps",
                "temperature": 0.6,
                "step_trace_out": "/tmp/out.jsonl"
            })
            .to_string(),
        )
        .unwrap();

        let config = load_config(file.path()).unwrap();
        assert_eq!(
            config.forced_first_step,
            Some(ForcedStep {
                assistant_text: "plan".to_string(),
                tool_calls: vec![PrefixToolCall {
                    id: "c1".to_string(),
                    name: "edit".to_string(),
                    arguments: "{}".to_string(),
                }],
            })
        );
        assert_eq!(config.snapshot_dir, Some(PathBuf::from("/tmp/p2t-snaps")));
    }

    #[test]
    fn load_config_accepts_null_forced_first_step() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            serde_json::json!({
                "prefix_steps": null,
                "forced_first_step": null,
                "max_steps": 10,
                "snapshot_dir": null,
                "temperature": 0.6,
                "step_trace_out": "/tmp/out.jsonl"
            })
            .to_string(),
        )
        .unwrap();

        let config = load_config(file.path()).unwrap();
        assert_eq!(config.forced_first_step, None);
        assert_eq!(config.snapshot_dir, None);
    }

    #[test]
    fn load_config_rejects_relative_snapshot_dir() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            serde_json::json!({
                "prefix_steps": null,
                "forced_first_step": null,
                "max_steps": 10,
                "snapshot_dir": "relative/snaps",
                "temperature": 0.6,
                "step_trace_out": "/tmp/out.jsonl"
            })
            .to_string(),
        )
        .unwrap();

        let err = load_config(file.path()).unwrap_err().to_string();
        assert!(err.contains("snapshot_dir"));
    }

    #[test]
    fn load_config_from_env_rejects_mutual_exclusion() {
        let _lock = ENV_GUARD.blocking_lock();
        let _scope = EnvScope::set(PATCHES_TO_TRACES_ENV, "1");
        let err = load_config_from_env(true).unwrap_err().to_string();
        assert!(err.contains("cannot both be enabled"));
    }

    #[test]
    fn load_prefix_steps_parses_jsonl() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            concat!(
                "{\"assistant_text\":\"a\",\"tool_calls\":[],\"results\":[]}\n",
                "{\"assistant_text\":\"b\",\"tool_calls\":[{\"id\":\"c1\",\"name\":\"edit\",\"arguments\":\"{}\"}],\"results\":[{\"call_id\":\"c1\",\"content\":\"ok\"}]}\n"
            ),
        )
        .unwrap();

        let steps = load_prefix_steps(file.path()).unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[1].tool_calls[0].name, "edit");
        assert_eq!(steps[1].results[0].content, "ok");
    }

    #[test]
    fn prefix_steps_convert_to_message_sequence() {
        let steps = vec![PrefixStep {
            assistant_text: "thinking".to_string(),
            tool_calls: vec![PrefixToolCall {
                id: "call-1".to_string(),
                name: "edit".to_string(),
                arguments: "{\"file_path\":\"src/main.rs\"}".to_string(),
            }],
            results: vec![PrefixToolResult {
                call_id: "call-1".to_string(),
                content: "Edited 'src/main.rs'".to_string(),
            }],
        }];

        let messages = prefix_steps_to_messages(&steps);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "assistant");
        assert_eq!(messages[0].content_text(), "thinking");
        assert_eq!(
            messages[0].tool_calls.as_ref().unwrap()[0].function.name,
            "edit"
        );
        assert_eq!(messages[1].role, "tool");
        assert_eq!(messages[1].tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(messages[1].name.as_deref(), Some("edit"));
    }

    #[test]
    fn forced_step_message_preserves_text_and_tool_calls() {
        let message = forced_step_to_message(&ForcedStep {
            assistant_text: "planning".to_string(),
            tool_calls: vec![PrefixToolCall {
                id: "call-1".to_string(),
                name: "edit".to_string(),
                arguments: "{\"file_path\":\"src/main.rs\"}".to_string(),
            }],
        });

        assert_eq!(message.role, "assistant");
        assert_eq!(message.content_text(), "planning");
        assert_eq!(
            message.tool_calls.as_ref().unwrap()[0].function.arguments,
            "{\"file_path\":\"src/main.rs\"}"
        );
    }

    #[test]
    fn stop_reason_logic_distinguishes_finish_from_window_end() {
        assert_eq!(
            stop_reason_after_step(1, 3, 0),
            Some(P2tStopReason::Finished)
        );
        assert_eq!(
            stop_reason_after_step(3, 3, 1),
            Some(P2tStopReason::WindowEnd)
        );
        assert_eq!(stop_reason_after_step(2, 3, 1), None);
    }

    #[test]
    fn prefix_successful_edit_unlocks_shell() {
        let steps = vec![PrefixStep {
            assistant_text: String::new(),
            tool_calls: vec![PrefixToolCall {
                id: "call-1".to_string(),
                name: "write_file".to_string(),
                arguments: "{}".to_string(),
            }],
            results: vec![PrefixToolResult {
                call_id: "call-1".to_string(),
                content: "wrote file".to_string(),
            }],
        }];

        assert!(prefix_unlocks_shell(&steps));
    }

    #[test]
    fn snapshot_plan_uses_previous_step_as_link_dest() {
        let root = Path::new("/tmp/p2t-snapshots");
        let plan = snapshot_plan(root, 3, true);

        assert_eq!(plan.dest, root.join("step-3"));
        assert_eq!(plan.link_dest, Some(root.join("step-2")));
    }

    #[test]
    fn snapshot_plan_skips_link_dest_without_previous_snapshot() {
        let root = Path::new("/tmp/p2t-snapshots");

        assert_eq!(
            snapshot_plan(root, 0, false),
            SnapshotPlan {
                dest: root.join("step-0"),
                link_dest: None,
            }
        );
        assert_eq!(
            snapshot_plan(root, 1, false),
            SnapshotPlan {
                dest: root.join("step-1"),
                link_dest: None,
            }
        );
    }
}
