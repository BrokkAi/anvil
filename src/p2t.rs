use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Deserializer, Serialize};

use crate::llm_client::{ChatContentPart, ChatMessage, FunctionCall, ToolCall};
use crate::tools::tool_result_failed;

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
    /// Base for the FIRST step's snapshot link-dest: the workspace's source
    /// (the overlay lower / canonical the caller branched from). Lets step-0's
    /// snapshot hardlink unchanged files to that base instead of full-copying,
    /// so it costs no disk. Later steps already link against the previous step.
    pub link_base: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct P2tConfigFile {
    prefix_steps: Option<PathBuf>,
    forced_first_step: Option<ForcedStep>,
    max_steps: usize,
    snapshot_dir: Option<PathBuf>,
    temperature: Option<f64>,
    step_trace_out: PathBuf,
    link_base: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub(crate) struct PrefixStep {
    #[serde(default)]
    pub assistant_text: String,
    #[serde(default)]
    pub tool_calls: Vec<PrefixToolCall>,
    #[serde(default)]
    pub results: Vec<PrefixToolResult>,
    #[serde(default, deserialize_with = "deserialize_prefix_messages")]
    pub messages: Vec<ChatMessage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub(crate) struct ForcedStep {
    #[serde(default)]
    pub assistant_text: String,
    #[serde(default)]
    pub tool_calls: Vec<PrefixToolCall>,
    #[serde(default)]
    pub message: Option<ChatMessage>,
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
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<ChatMessage>,
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
    if parsed
        .link_base
        .as_ref()
        .is_some_and(|path| !path.is_absolute())
    {
        bail!("link_base must be an absolute path when set");
    }
    Ok(P2tConfig {
        prefix_steps: parsed.prefix_steps,
        forced_first_step: parsed.forced_first_step,
        max_steps: parsed.max_steps,
        snapshot_dir: parsed.snapshot_dir,
        temperature: parsed.temperature,
        step_trace_out: parsed.step_trace_out,
        link_base: parsed.link_base,
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

fn deserialize_prefix_messages<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<ChatMessage>, D::Error>
where
    D: Deserializer<'de>,
{
    let values = Vec::<serde_json::Value>::deserialize(deserializer)?;
    values
        .into_iter()
        .map(normalize_prefix_message)
        .map(|value| serde_json::from_value(value).map_err(serde::de::Error::custom))
        .collect()
}

fn normalize_prefix_message(mut value: serde_json::Value) -> serde_json::Value {
    if let Some(content) = value.get_mut("content")
        && let Some(text) = content.as_str()
    {
        *content = serde_json::json!([{ "type": "text", "text": text }]);
    }
    value
}

pub(crate) fn append_prefix_messages(messages: &mut Vec<ChatMessage>, steps: &[PrefixStep]) {
    messages.extend(prefix_steps_to_messages(steps));
}

pub(crate) fn prefix_steps_to_messages(steps: &[PrefixStep]) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    for step in steps {
        if !step.messages.is_empty() {
            messages.extend(step.messages.clone());
            continue;
        }
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
    if let Some(message) = &step.message {
        return message.clone();
    }
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
        // Present-but-empty, not None: DeepSeek thinking-mode rejects an
        // assistant turn with no reasoning_content ("must be passed back"),
        // and None omits the field entirely (llm_client serialize). An injected
        // PrefixStep tool-call has no real reasoning; an empty string satisfies
        // the field-presence contract without fabricating rationale.
        reasoning_content: Some(String::new()),
    }
}

pub(crate) fn tool_result_messages(
    tool_calls: &[PrefixToolCall],
    results: &[PrefixToolResult],
) -> Vec<ChatMessage> {
    results
        .iter()
        .map(|result| {
            let tool_name = tool_calls
                .iter()
                .find(|call| call.id == result.call_id)
                .map(|call| call.name.clone())
                .unwrap_or_default();
            ChatMessage::tool_result(&result.call_id, tool_name, &result.content)
        })
        .collect()
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
    ["write_file", "edit"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

pub(crate) fn p2t_post_edit_builtin_tools() -> HashSet<String> {
    let mut tools = p2t_initial_builtin_tools();
    tools.insert("run_shell_command".to_string());
    // Native ranged file read unlocks with shell: pre-unlock the agent must
    // investigate through the symbol tools only (no file-level reads); once it
    // has edited, raw reads reveal nothing the edit phase shouldn't see.
    tools.insert("read_file".to_string());
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

pub(crate) fn reset_window_session_if_stale(
    step_trace_out: &Path,
    snapshot_dir: Option<&Path>,
) -> Result<bool> {
    if !step_trace_has_records(step_trace_out)? {
        return Ok(false);
    }

    std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(step_trace_out)
        .with_context(|| format!("failed to truncate {}", step_trace_out.display()))?;

    if let Some(snapshot_dir) = snapshot_dir {
        clear_step_snapshots(snapshot_dir)?;
    }

    Ok(true)
}

fn step_trace_has_records(path: &Path) -> Result<bool> {
    match std::fs::metadata(path) {
        Ok(metadata) => Ok(metadata.len() > 0),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).with_context(|| format!("failed to stat {}", path.display())),
    }
}

fn clear_step_snapshots(snapshot_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(snapshot_dir)
        .with_context(|| format!("failed to create {}", snapshot_dir.display()))?;
    for entry in std::fs::read_dir(snapshot_dir)
        .with_context(|| format!("failed to read {}", snapshot_dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read {}", snapshot_dir.display()))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !name.starts_with("step-") {
            continue;
        }

        let path = entry.path();
        if entry
            .file_type()
            .with_context(|| format!("failed to stat {}", path.display()))?
            .is_dir()
        {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        }
        .with_context(|| format!("failed to remove {}", path.display()))?;
    }
    Ok(())
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

pub(crate) fn append_debug_trace(path: &Path, event: &str, details: serde_json::Value) {
    #[derive(Serialize)]
    struct DebugRecord<'a> {
        #[serde(rename = "type")]
        record_type: &'a str,
        event: &'a str,
        details: serde_json::Value,
    }

    append_jsonl(
        path,
        &DebugRecord {
            record_type: "debug",
            event,
            details,
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
    link_base: Option<&Path>,
) -> SnapshotPlan {
    // Snapshots are copy-on-write via rsync `--link-dest`: unchanged files are
    // hardlinked from the link-dest into the new snapshot (only changed files
    // cost disk). Normally each step links against the previous step; the first
    // step (no previous) links against `link_base` — the workspace's source
    // (overlay lower / canonical) — so even step-0 hardlinks unchanged files
    // instead of full-copying the whole workspace.
    let link_dest = if step > 0 && previous_exists {
        Some(snapshot_dir.join(format!("step-{}", step - 1)))
    } else {
        link_base.map(Path::to_path_buf)
    };
    SnapshotPlan {
        dest: snapshot_dir.join(format!("step-{step}")),
        link_dest,
    }
}

pub(crate) fn snapshot_workspace(
    cwd: &Path,
    snapshot_dir: &Path,
    step: usize,
    link_base: Option<&Path>,
) -> Result<()> {
    std::fs::create_dir_all(snapshot_dir)
        .with_context(|| format!("failed to create {}", snapshot_dir.display()))?;
    let previous_exists = step > 0 && snapshot_dir.join(format!("step-{}", step - 1)).exists();
    let plan = snapshot_plan(snapshot_dir, step, previous_exists, link_base);

    let mut command = Command::new("rsync");
    command.args(snapshot_rsync_args(cwd, &plan));

    let output = command
        .output()
        .with_context(|| format!("failed to spawn rsync for step {step}"))?;
    if output.status.success() {
        return Ok(());
    }

    bail!("{}", rsync_failure_detail(step, &output));
}

fn snapshot_rsync_args(cwd: &Path, plan: &SnapshotPlan) -> Vec<String> {
    let mut args = vec!["-a".to_string(), "--delete".to_string()];
    // Hard excludes FIRST (rsync applies rules in order, first match wins): these
    // runtime caches never belong in a snapshot, regardless of any .gitignore.
    for exclude in ["/.git", "/.brokk", "/.bifrost"] {
        args.push("--exclude".to_string());
        args.push(exclude.to_string());
    }
    // Then drop whatever the repo ITSELF declares regenerable via .gitignore
    // (build output: target/, node_modules/, vendor/, __pycache__/, dist/, ...).
    // `:- .gitignore` is a per-directory merge with git's own semantics -- no
    // hardcoded per-language list. Snapshots are CoW copies used only for scoring
    // and for the commit that re-materializes canonical; the testsome/build
    // command regenerates these artifacts, so excluding them is safe. Crucially it
    // stops per-step rebuilds (which rewrite build trees and so DEFEAT --link-dest
    // hardlinking) from ballooning disk across steps x seeds.
    args.push("--filter".to_string());
    args.push(":- .gitignore".to_string());
    if let Some(link_dest) = plan.link_dest.as_ref() {
        args.push(format!("--link-dest={}", link_dest.display()));
    }
    args.push(format!("{}/", cwd.display()));
    args.push(format!("{}/", plan.dest.display()));
    args
}

fn rsync_failure_detail(step: usize, output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        "rsync produced no output".to_string()
    };
    format!(
        "rsync step {step} failed with status {}: {detail}",
        output
            .status
            .code()
            .map_or_else(|| "signal".to_string(), |code| code.to_string())
    )
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
        let tempdir = tempfile::tempdir().unwrap();
        let snapshot_dir = tempdir.path().join("p2t-snaps");
        let step_trace_out = tempdir.path().join("out.jsonl");
        std::fs::write(
            file.path(),
            serde_json::json!({
                "prefix_steps": null,
                "forced_first_step": {
                    "assistant_text": "plan",
                    "tool_calls": [{"id":"c1","name":"edit","arguments":"{}"}]
                },
                "max_steps": 10,
                "snapshot_dir": snapshot_dir,
                "temperature": 0.6,
                "step_trace_out": step_trace_out
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
                message: None,
            })
        );
        assert_eq!(config.snapshot_dir, Some(snapshot_dir));
    }

    #[test]
    fn load_config_accepts_null_forced_first_step() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let tempdir = tempfile::tempdir().unwrap();
        let step_trace_out = tempdir.path().join("out.jsonl");
        std::fs::write(
            file.path(),
            serde_json::json!({
                "prefix_steps": null,
                "forced_first_step": null,
                "max_steps": 10,
                "snapshot_dir": null,
                "temperature": 0.6,
                "step_trace_out": step_trace_out
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
        let tempdir = tempfile::tempdir().unwrap();
        let step_trace_out = tempdir.path().join("out.jsonl");
        std::fs::write(
            file.path(),
            serde_json::json!({
                "prefix_steps": null,
                "forced_first_step": null,
                "max_steps": 10,
                "snapshot_dir": "relative/snaps",
                "temperature": 0.6,
                "step_trace_out": step_trace_out
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
            messages: Vec::new(),
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
    fn prefix_steps_prefer_exact_message_sequence() {
        let exact = ChatMessage::assistant_with_reasoning(
            "exact visible".to_string(),
            Some("native reasoning".to_string()),
        );
        let steps = vec![PrefixStep {
            assistant_text: "lossy fallback".to_string(),
            tool_calls: Vec::new(),
            results: Vec::new(),
            messages: vec![exact.clone()],
        }];

        let messages = prefix_steps_to_messages(&steps);
        assert_eq!(messages, vec![exact]);
    }

    #[test]
    fn load_prefix_steps_accepts_raw_text_message_content() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            r#"{"assistant_text":"","tool_calls":[],"results":[],"messages":[{"role":"user","content":"Still reproduces."}]}"#,
        )
        .unwrap();

        let steps = load_prefix_steps(file.path()).unwrap();

        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].messages.len(), 1);
        assert_eq!(steps[0].messages[0].role, "user");
        assert_eq!(steps[0].messages[0].content_text(), "Still reproduces.");
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
            message: None,
        });

        assert_eq!(message.role, "assistant");
        assert_eq!(message.content_text(), "planning");
        assert_eq!(
            message.tool_calls.as_ref().unwrap()[0].function.arguments,
            "{\"file_path\":\"src/main.rs\"}"
        );
    }

    #[test]
    fn forced_step_prefers_exact_message_with_reasoning_content() {
        let exact = ChatMessage::assistant_tool_calls_with_content_and_reasoning(
            "synthetic visible",
            vec![ToolCall {
                id: "call-1".to_string(),
                r#type: "function".to_string(),
                function: FunctionCall {
                    name: "get_summaries".to_string(),
                    arguments: "{\"targets\":[\"src/main.rs\"]}".to_string(),
                },
            }],
            Some("synthetic visible-prefix reasoning".to_string()),
        );

        let message = forced_step_to_message(&ForcedStep {
            assistant_text: "fallback visible".to_string(),
            tool_calls: Vec::new(),
            message: Some(exact.clone()),
        });

        assert_eq!(message, exact);
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
    fn stale_window_session_is_rotated_before_window_start() {
        let tempdir = tempfile::tempdir().unwrap();
        let trace = tempdir.path().join("steps.jsonl");
        let snapshot_dir = tempdir.path().join("snapshots");
        std::fs::create_dir_all(snapshot_dir.join("step-0")).unwrap();
        std::fs::create_dir_all(snapshot_dir.join("step-1")).unwrap();
        std::fs::write(snapshot_dir.join("step-1").join("stale.txt"), "old").unwrap();
        std::fs::write(snapshot_dir.join("step-note"), "old").unwrap();
        std::fs::create_dir_all(snapshot_dir.join("keep")).unwrap();
        std::fs::write(
            &trace,
            concat!(
                "{\"type\":\"window_start\",\"messages\":[],\"tool_names\":[]}\n",
                "{\"type\":\"step\",\"step\":1}\n",
                "{\"type\":\"window_end\",\"reason\":\"p2t_window_end\",\"steps\":1}\n"
            ),
        )
        .unwrap();

        assert!(reset_window_session_if_stale(&trace, Some(&snapshot_dir)).unwrap());
        append_window_start_trace(&trace, &[], &[]);

        let records: Vec<serde_json::Value> = std::fs::read_to_string(&trace)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["type"], "window_start");
        assert!(snapshot_dir.exists());
        assert!(!snapshot_dir.join("step-0").exists());
        assert!(!snapshot_dir.join("step-1").exists());
        assert!(!snapshot_dir.join("step-note").exists());
        assert!(snapshot_dir.join("keep").exists());
    }

    #[test]
    fn fresh_window_session_does_not_touch_snapshots() {
        let tempdir = tempfile::tempdir().unwrap();
        let trace = tempdir.path().join("steps.jsonl");
        let snapshot_dir = tempdir.path().join("snapshots");

        assert!(!reset_window_session_if_stale(&trace, Some(&snapshot_dir)).unwrap());

        assert!(!trace.exists());
        assert!(!snapshot_dir.exists());
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
            messages: Vec::new(),
        }];

        assert!(prefix_unlocks_shell(&steps));
    }

    #[test]
    fn snapshot_plan_uses_previous_step_as_link_dest() {
        let root = Path::new("/tmp/p2t-snapshots");
        // Previous step wins over link_base.
        let plan = snapshot_plan(root, 3, true, Some(Path::new("/tmp/canonical")));

        assert_eq!(plan.dest, root.join("step-3"));
        assert_eq!(plan.link_dest, Some(root.join("step-2")));
    }

    #[test]
    fn snapshot_plan_first_step_links_against_base() {
        let root = Path::new("/tmp/p2t-snapshots");
        let base = Path::new("/tmp/canonical");
        // No previous step -> link against the workspace source so step-0 is
        // hardlinked (CoW), not a full copy.
        assert_eq!(
            snapshot_plan(root, 0, false, Some(base)),
            SnapshotPlan {
                dest: root.join("step-0"),
                link_dest: Some(base.to_path_buf()),
            }
        );
    }

    #[test]
    fn snapshot_plan_no_link_dest_without_previous_or_base() {
        let root = Path::new("/tmp/p2t-snapshots");

        assert_eq!(
            snapshot_plan(root, 0, false, None),
            SnapshotPlan {
                dest: root.join("step-0"),
                link_dest: None,
            }
        );
        assert_eq!(
            snapshot_plan(root, 1, false, None),
            SnapshotPlan {
                dest: root.join("step-1"),
                link_dest: None,
            }
        );
    }

    #[test]
    fn snapshot_rsync_args_exclude_runtime_artifact_dirs() {
        let plan = SnapshotPlan {
            dest: PathBuf::from("/tmp/p2t-snapshots/step-0"),
            link_dest: Some(PathBuf::from("/tmp/canonical")),
        };
        let args = snapshot_rsync_args(Path::new("/tmp/worktree"), &plan);

        for excluded in ["/.git", "/.brokk", "/.bifrost"] {
            assert!(
                args.windows(2).any(|pair| pair == ["--exclude", excluded]),
                "missing rsync exclude for {excluded}: {args:?}"
            );
        }
        // Build output is dropped via the repo's own .gitignore (per-dir merge),
        // not a hardcoded list. This is what keeps per-step snapshots from
        // capturing regenerated target/ | node_modules/ | vendor/ trees.
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--filter", ":- .gitignore"]),
            "missing gitignore dir-merge filter: {args:?}"
        );
        // ...and the hard excludes precede the .gitignore merge so a stray
        // negation in a repo .gitignore can never re-include a runtime cache.
        let filter_at = args.iter().position(|a| a == "--filter").unwrap();
        let last_hard_exclude = args
            .iter()
            .rposition(|a| a == "/.bifrost" || a == "/.brokk" || a == "/.git")
            .unwrap();
        assert!(
            last_hard_exclude < filter_at,
            "hard excludes must precede .gitignore merge: {args:?}"
        );
        assert!(args.contains(&"--link-dest=/tmp/canonical".to_string()));
        assert!(args.contains(&"/tmp/worktree/".to_string()));
        assert!(args.contains(&"/tmp/p2t-snapshots/step-0/".to_string()));
    }
}
