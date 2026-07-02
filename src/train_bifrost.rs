use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(test)]
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::llm_client::{
    ChatMessage, IdleTimeouts, LlmBackend, LlmResponse, StreamChatRequest, TokenUsage,
    is_retryable_llm_error, stream_chat_no_visible_output_with_retry,
};
use crate::session::ToolExchange;
use crate::trace_logging::append_trace_record;

pub(crate) const TRAIN_BIFROST_PACKET_ENV: &str = "BRK_TRAIN_BIFROST_PACKET";
pub(crate) const TRAIN_BIFROST_HINT_MODEL: &str = "openrouter::deepseek/deepseek-v4-flash";
const HINT_MAX_ATTEMPTS: usize = 3;
const HINT_SYSTEM_PROMPT: &str = r#"You generate non-spoiling discovery hints for another coding model.

You are shown the active model's conversation and one golden-patch file diff.
Your job is to suggest the next source-discovery direction, not the fix.

Rules:
- Do not include file paths from the golden diff.
- Do not include new symbol names, method names, test names, constants, exact code, or string literals from the golden diff unless they already appear in the conversation/tool history.
- Even when terms already appeared, do not assemble them into implementation steps.
- Do not reveal the golden diff's specific diagnosis, such as a particular missing condition, constant, inserted character, ordering change, or exact API choice.
- Phrase hints as source areas, behaviors, relationships, or broad searchable terms to inspect.
- Avoid edit verbs like add, create, implement, apply, change, replace, or set.
- The active model must still use source-context tools to discover the actual fix.

Examples:
Bad: Focus on the leading character inserted before each text segment.
Good: Consider inspecting how multimodal text blocks are combined into the final message content.

Bad: Check whether GWL_EXSTYLE is available for modifying the overlay window's extended style.
Good: Consider inspecting existing overlay-window style handling and related native window-style helpers.

Bad: Add a custom load context that resolves assemblies from each plugin directory.
Good: Consider inspecting how plugin directories are discovered and how assemblies are loaded from them.

Return strict JSON only: {"hint":"..."}."#;

#[derive(Debug, Clone)]
pub(crate) struct TrainingPacket {
    pub files: Vec<TrainingFile>,
    pub related_files: Vec<TrainingRelatedFile>,
}

#[derive(Debug, Clone)]
pub(crate) struct TrainingFile {
    pub path: String,
    pub diff: String,
}

#[derive(Debug, Clone)]
pub(crate) struct TrainingRelatedFile {
    pub path: String,
    pub summary: String,
}

#[derive(Debug, Deserialize)]
struct PacketManifest {
    files: Vec<PacketManifestFile>,
    #[serde(default)]
    related_files: Vec<PacketManifestRelatedFile>,
}

#[derive(Debug, Deserialize)]
struct PacketManifestFile {
    path: String,
    diff: String,
}

#[derive(Debug, Deserialize)]
struct PacketManifestRelatedFile {
    path: String,
    summary: String,
}

#[derive(Debug, Deserialize)]
struct HintResponse {
    hint: String,
}

pub(crate) fn load_packet_from_env() -> Result<TrainingPacket> {
    let manifest_path = env::var(TRAIN_BIFROST_PACKET_ENV).with_context(|| {
        format!("{TRAIN_BIFROST_PACKET_ENV} must be set when BRK_TRAIN_BIFROST=1")
    })?;
    load_packet(Path::new(&manifest_path))
}

fn load_packet(manifest_path: &Path) -> Result<TrainingPacket> {
    let manifest_text = std::fs::read_to_string(manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;
    let manifest: PacketManifest = serde_json::from_str(&manifest_text)
        .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
    let base = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let mut files = Vec::with_capacity(manifest.files.len());
    for entry in manifest.files {
        let path = normalize_project_path(&entry.path);
        if path.is_empty() {
            continue;
        }
        let diff_path = resolve_packet_child(base, &entry.diff)?;
        let diff = std::fs::read_to_string(&diff_path)
            .with_context(|| format!("failed to read {}", diff_path.display()))?;
        if diff.trim().is_empty() {
            continue;
        }
        files.push(TrainingFile { path, diff });
    }
    if files.is_empty() {
        bail!("training packet contains no usable files");
    }
    let related_files = manifest
        .related_files
        .into_iter()
        .filter_map(|entry| {
            let path = normalize_project_path(&entry.path);
            let summary = entry.summary.trim().to_string();
            if path.is_empty() || summary.is_empty() {
                None
            } else {
                Some(TrainingRelatedFile { path, summary })
            }
        })
        .collect();
    Ok(TrainingPacket {
        files,
        related_files,
    })
}

fn resolve_packet_child(base: &Path, child: &str) -> Result<PathBuf> {
    let child_path = Path::new(child);
    if child_path.is_absolute() {
        bail!("training packet diff path must be relative: {child}");
    }
    if child_path.components().any(|component| {
        matches!(
            component,
            std::path::Component::ParentDir | std::path::Component::RootDir
        )
    }) {
        bail!("training packet diff path must stay inside packet directory: {child}");
    }
    Ok(base.join(child_path))
}

fn normalize_project_path(path: &str) -> String {
    let path = path.trim().replace('\\', "/");
    path.strip_prefix("a/")
        .or_else(|| path.strip_prefix("b/"))
        .unwrap_or(path.as_str())
        .trim_start_matches("./")
        .to_string()
}

pub(crate) fn unread_files<'a>(
    packet: &'a TrainingPacket,
    tool_exchanges: &[ToolExchange],
) -> Vec<&'a TrainingFile> {
    let inspected = inspected_training_files(
        tool_exchanges,
        packet.files.iter().map(|file| file.path.as_str()),
    );
    packet
        .files
        .iter()
        .filter(|file| !inspected.contains(&file.path))
        .collect()
}

fn inspected_training_files<'a>(
    tool_exchanges: &[ToolExchange],
    candidate_paths: impl Iterator<Item = &'a str>,
) -> BTreeSet<String> {
    let candidates: Vec<String> = candidate_paths.map(normalize_project_path).collect();
    let mut inspected = BTreeSet::new();
    for exchange in tool_exchanges {
        if !is_successful_bifrost_content_exchange(exchange) {
            continue;
        }
        let haystack = format!("{}\n{}", exchange.arguments, exchange.result).replace('\\', "/");
        for path in &candidates {
            if path_appears_in_text(path, &haystack) {
                inspected.insert(path.clone());
            }
        }
    }
    inspected
}

fn is_successful_bifrost_content_exchange(exchange: &ToolExchange) -> bool {
    matches!(
        exchange.tool_name.as_str(),
        "get_symbol_sources" | "get_summaries" | "scan_usages"
    ) && !exchange.result.starts_with("Error:")
        && !exchange.result.starts_with("Internal error:")
}

fn path_appears_in_text(path: &str, text: &str) -> bool {
    text.contains(path)
        || text.contains(&format!("a/{path}"))
        || text.contains(&format!("b/{path}"))
}

fn deterministic_summary_nudges(
    packet: &TrainingPacket,
    tool_exchanges: &[ToolExchange],
) -> Vec<String> {
    packet
        .related_files
        .iter()
        .filter(|file| {
            related_file_discovered(&file.path, tool_exchanges)
                && !related_file_summarized(&file.path, tool_exchanges)
        })
        .map(|file| {
            format!(
                "You have found {} but have not summarized it yet. Call get_summaries on {} before deciding the edit.",
                file.path, file.path
            )
        })
        .collect()
}

fn related_file_discovered(path: &str, tool_exchanges: &[ToolExchange]) -> bool {
    tool_exchanges.iter().any(|exchange| {
        let haystack = format!("{}\n{}", exchange.arguments, exchange.result).replace('\\', "/");
        path_appears_in_text(path, &haystack) || listed_directory_contains_path(path, exchange)
    })
}

fn related_file_summarized(path: &str, tool_exchanges: &[ToolExchange]) -> bool {
    tool_exchanges.iter().any(|exchange| {
        exchange.tool_name == "get_summaries"
            && !tool_result_failed(&exchange.result)
            && path_appears_in_text(
                path,
                &format!("{}\n{}", exchange.arguments, exchange.result).replace('\\', "/"),
            )
    })
}

fn tool_result_failed(result: &str) -> bool {
    result.starts_with("Error:") || result.starts_with("Internal error:")
}

fn listed_directory_contains_path(path: &str, exchange: &ToolExchange) -> bool {
    if exchange.tool_name != "list_directory" || tool_result_failed(&exchange.result) {
        return false;
    }
    let Some((parent, basename)) = path_parent_and_basename(path) else {
        return false;
    };
    let Ok(args) = serde_json::from_str::<serde_json::Value>(&exchange.arguments) else {
        return false;
    };
    let listed = args
        .get("path")
        .and_then(|path| path.as_str())
        .map(normalize_project_path)
        .unwrap_or_else(|| ".".to_string());
    normalize_dir_path(&listed) == normalize_dir_path(parent)
        && exchange
            .result
            .replace('\\', "/")
            .lines()
            .any(|line| line.trim() == basename)
}

fn path_parent_and_basename(path: &str) -> Option<(&str, &str)> {
    let path = path.trim_matches('/');
    if path.is_empty() {
        return None;
    }
    match path.rsplit_once('/') {
        Some((parent, basename)) if !basename.is_empty() => Some((parent, basename)),
        None => Some((".", path)),
        _ => None,
    }
}

fn normalize_dir_path(path: &str) -> String {
    let normalized = normalize_project_path(path).trim_matches('/').to_string();
    if normalized.is_empty() || normalized == "." {
        ".".to_string()
    } else {
        normalized
    }
}

fn sidecar_summaries_for_hint(packet: &TrainingPacket) -> String {
    if packet.related_files.is_empty() {
        return "(none)".to_string();
    }
    packet
        .related_files
        .iter()
        .map(|file| {
            format!(
                "## {}\n{}\n",
                file.path,
                truncate_chars(&file.summary, 12_000)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn deterministic_nudges_for_hint(nudges: &[String]) -> String {
    if nudges.is_empty() {
        return "(none)".to_string();
    }
    nudges
        .iter()
        .map(|nudge| format!("- {}", nudge.replace('\n', " ")))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) async fn compose_no_edit_nudge(
    llm: &Arc<dyn LlmBackend>,
    turn: usize,
    messages: &[ChatMessage],
    tool_exchanges: &[ToolExchange],
    packet: &TrainingPacket,
    cancel: &CancellationToken,
    idle_timeout: IdleTimeouts,
) -> Option<(String, TokenUsage)> {
    let unread = unread_files(packet, tool_exchanges);
    let deterministic_nudges = deterministic_summary_nudges(packet, tool_exchanges);
    if unread.is_empty() {
        if !deterministic_nudges.is_empty() {
            let mut nudge = String::from(
                "You have not made a successful edit/write yet. Based only on areas already surfaced in this conversation, consider these next search directions before continuing:\n",
            );
            for hint in &deterministic_nudges {
                nudge.push_str("- ");
                nudge.push_str(&hint.replace('\n', " "));
                nudge.push('\n');
            }
            nudge.push_str(
                "Use Bifrost/source-context tools to verify any lead, then make the smallest plausible edit/write_file change.",
            );
            return Some((nudge, TokenUsage::default()));
        }
        return Some((
            "You appear to have inspected the relevant source areas for this task. Stop gathering more context and craft the smallest plausible solution now using edit/write_file.".to_string(),
            TokenUsage::default(),
        ));
    }

    let mut hints = Vec::new();
    let mut usage = TokenUsage::default();
    for file in unread {
        match request_hint_with_retries(HintRequest {
            llm,
            turn,
            context: HintRequestContext {
                messages,
                tool_exchanges,
                file,
                packet,
                deterministic_nudges: &deterministic_nudges,
            },
            cancel,
            idle_timeout,
        })
        .await
        {
            Ok((hint, hint_usage)) => {
                usage.add(hint_usage);
                if !hint.trim().is_empty() {
                    hints.push(hint.trim().to_string());
                }
            }
            Err(error) => {
                append_trace_record(serde_json::json!({
                    "type": "train_bifrost_hint_skipped",
                    "turn": turn,
                    "file": file.path,
                    "error": format!("{error:#}"),
                }));
            }
        }
    }

    if deterministic_nudges.is_empty() && hints.is_empty() {
        return None;
    }

    let mut nudge = String::from(
        "You have not made a successful edit/write yet. Based only on areas already surfaced in this conversation, consider these next search directions before continuing:\n",
    );
    for hint in &deterministic_nudges {
        nudge.push_str("- ");
        nudge.push_str(&hint.replace('\n', " "));
        nudge.push('\n');
    }
    for hint in hints {
        nudge.push_str("- ");
        nudge.push_str(&hint.replace('\n', " "));
        nudge.push('\n');
    }
    nudge.push_str("Use Bifrost/source-context tools to verify any lead, then make the smallest plausible edit/write_file change.");
    Some((nudge, usage))
}

struct HintRequest<'a> {
    llm: &'a Arc<dyn LlmBackend>,
    turn: usize,
    context: HintRequestContext<'a>,
    cancel: &'a CancellationToken,
    idle_timeout: IdleTimeouts,
}

async fn request_hint_with_retries(req: HintRequest<'_>) -> Result<(String, TokenUsage)> {
    let mut last_error = None;
    for attempt in 1..=HINT_MAX_ATTEMPTS {
        match request_hint(&req, attempt).await {
            Ok(result) => return Ok(result),
            Err(error) => {
                append_trace_record(serde_json::json!({
                    "type": "train_bifrost_hint_retry",
                    "turn": req.turn,
                    "file": req.context.file.path,
                    "attempt": attempt,
                    "max_attempts": HINT_MAX_ATTEMPTS,
                    "error": format!("{error:#}"),
                }));
                if is_retryable_llm_error(&error) {
                    return Err(error);
                }
                last_error = Some(error);
            }
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("hint request failed")))
}

async fn request_hint(req: &HintRequest<'_>, attempt: usize) -> Result<(String, TokenUsage)> {
    append_trace_record(serde_json::json!({
        "type": "train_bifrost_hint_request",
        "turn": req.turn,
        "attempt": attempt,
        "file": req.context.file.path,
        "model": TRAIN_BIFROST_HINT_MODEL,
    }));

    let prompt_messages = vec![
        ChatMessage::system(HINT_SYSTEM_PROMPT),
        ChatMessage::user(format!(
            "<conversation>\n{}\n</conversation>\n\n<sidecar_file_summaries>\n{}\n</sidecar_file_summaries>\n\n<deterministic_nudges_already_planned>\n{}\n</deterministic_nudges_already_planned>\n\n<golden_file_diff>\n{}\n</golden_file_diff>\n\nReturn one concise source-discovery hint. It should help the active model decide what to inspect next, not what code to write. Do not repeat any deterministic nudge already planned above.",
            conversation_for_hint(req.context.messages, req.context.tool_exchanges),
            sidecar_summaries_for_hint(req.context.packet),
            deterministic_nudges_for_hint(req.context.deterministic_nudges),
            req.context.file.diff
        )),
    ];

    let response = stream_chat_no_visible_output_with_retry(
        req.llm.as_ref(),
        "requesting train-bifrost hint",
        req.cancel,
        || StreamChatRequest {
            model: TRAIN_BIFROST_HINT_MODEL.to_string(),
            messages: prompt_messages.clone(),
            tools: None,
            reasoning_effort: None,
            temperature: None,
            structured_output: None,
            on_token: Box::new(|_| {}),
            on_thought: Box::new(|_| {}),
            cancel: req.cancel.clone(),
            idle_timeouts: req.idle_timeout,
        },
    )
    .await
    .with_context(|| format!("hint model request failed for {}", req.context.file.path))?;

    let usage = response.usage();
    let text = match response {
        LlmResponse::Text { text, .. } => text,
        LlmResponse::ToolCalls { text, .. } => text,
    };
    let parsed: HintResponse = serde_json::from_str(text.trim())
        .with_context(|| format!("hint model returned malformed JSON: {}", text.trim()))?;
    append_trace_record(serde_json::json!({
        "type": "train_bifrost_hint_response",
        "turn": req.turn,
        "attempt": attempt,
        "file": req.context.file.path,
        "hint": parsed.hint,
    }));
    Ok((parsed.hint, usage))
}

#[derive(Clone, Copy)]
struct HintRequestContext<'a> {
    messages: &'a [ChatMessage],
    tool_exchanges: &'a [ToolExchange],
    file: &'a TrainingFile,
    packet: &'a TrainingPacket,
    deterministic_nudges: &'a [String],
}

fn conversation_for_hint(messages: &[ChatMessage], tool_exchanges: &[ToolExchange]) -> String {
    const MAX_CHARS: usize = 60_000;
    const MAX_TOOL_RESULT_CHARS: usize = 4_000;
    let mut out = String::new();
    for message in messages {
        out.push_str(&format!("[{}]\n", message.role));
        for part in &message.content {
            out.push_str(&format!("{part:?}\n"));
        }
        if let Some(calls) = &message.tool_calls {
            out.push_str(&format!("tool_calls: {calls:?}\n"));
        }
    }
    out.push_str("\n[tool_exchanges]\n");
    for exchange in tool_exchanges {
        out.push_str(&format!(
            "tool: {}\nargs: {}\nresult: {}\n\n",
            exchange.tool_name,
            exchange.arguments,
            truncate_chars(&exchange.result, MAX_TOOL_RESULT_CHARS)
        ));
    }
    truncate_chars(&out, MAX_CHARS)
}

fn truncate_chars(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...[truncated]", &text[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_client::{ChatContentPart, LlmResponse};
    use futures::FutureExt;
    use futures::future::BoxFuture;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn packet_load_rejects_empty_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = dir.path().join("manifest.json");
        std::fs::write(&manifest, r#"{"files":[]}"#).unwrap();

        assert!(load_packet(&manifest).is_err());
    }

    #[test]
    fn packet_loads_relative_diff_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("files")).unwrap();
        std::fs::write(
            dir.path().join("files/0.diff"),
            "diff --git a/src/lib.rs b/src/lib.rs\n+hi\n",
        )
        .unwrap();
        let manifest = dir.path().join("manifest.json");
        std::fs::write(
            &manifest,
            r#"{"files":[{"path":"b/src/lib.rs","diff":"files/0.diff"}]}"#,
        )
        .unwrap();

        let packet = load_packet(&manifest).unwrap();

        assert_eq!(packet.files[0].path, "src/lib.rs");
        assert!(packet.files[0].diff.contains("+hi"));
    }

    #[test]
    fn packet_loads_related_file_summaries() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("files")).unwrap();
        std::fs::write(
            dir.path().join("files/0.diff"),
            "diff --git a/src/lib.rs b/src/lib.rs\n+hi\n",
        )
        .unwrap();
        let manifest = dir.path().join("manifest.json");
        std::fs::write(
            &manifest,
            r#"{
                "files":[{"path":"src/lib.rs","diff":"files/0.diff"}],
                "related_files":[{"path":"b/src/Related.rs","summary":"related summary"}]
            }"#,
        )
        .unwrap();

        let packet = load_packet(&manifest).unwrap();

        assert_eq!(packet.related_files[0].path, "src/Related.rs");
        assert_eq!(packet.related_files[0].summary, "related summary");
    }

    #[test]
    fn packet_rejects_parent_diff_paths() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = dir.path().join("manifest.json");
        std::fs::write(
            &manifest,
            r#"{"files":[{"path":"src/lib.rs","diff":"../leak.diff"}]}"#,
        )
        .unwrap();

        assert!(load_packet(&manifest).is_err());
    }

    #[test]
    fn unread_files_ignores_read_file_and_counts_bifrost_source_path() {
        let packet = TrainingPacket {
            files: vec![TrainingFile {
                path: "src/lib.rs".to_string(),
                diff: "diff".to_string(),
            }],
            related_files: Vec::new(),
        };
        let read_file = ToolExchange {
            call_id: "read".to_string(),
            tool_name: "read_file".to_string(),
            arguments: r#"{"file_path":"src/lib.rs"}"#.to_string(),
            result: "contents".to_string(),
            ..ToolExchange::default()
        };

        assert_eq!(unread_files(&packet, &[read_file]).len(), 1);

        let source = ToolExchange {
            call_id: "source".to_string(),
            tool_name: "get_symbol_sources".to_string(),
            arguments: "{}".to_string(),
            result: "File: src/lib.rs\nfn main() {}".to_string(),
            ..ToolExchange::default()
        };

        assert!(unread_files(&packet, &[source]).is_empty());
    }

    #[test]
    fn deterministic_summary_nudge_triggers_after_path_discovery() {
        let packet = TrainingPacket {
            files: vec![TrainingFile {
                path: "src/lib.rs".to_string(),
                diff: "diff".to_string(),
            }],
            related_files: vec![TrainingRelatedFile {
                path: "src/Related.rs".to_string(),
                summary: "related summary".to_string(),
            }],
        };
        let source = ToolExchange {
            call_id: "source".to_string(),
            tool_name: "get_symbol_sources".to_string(),
            arguments: "{}".to_string(),
            result: "File: src/Related.rs\nstruct Related;".to_string(),
            ..ToolExchange::default()
        };

        let nudges = deterministic_summary_nudges(&packet, &[source]);

        assert_eq!(nudges.len(), 1);
        assert!(nudges[0].contains("Call get_summaries on src/Related.rs"));
    }

    #[test]
    fn deterministic_summary_nudge_triggers_after_parent_listing() {
        let packet = TrainingPacket {
            files: vec![TrainingFile {
                path: "src/lib.rs".to_string(),
                diff: "diff".to_string(),
            }],
            related_files: vec![TrainingRelatedFile {
                path: "src/Related.rs".to_string(),
                summary: "related summary".to_string(),
            }],
        };
        let listing = ToolExchange {
            call_id: "list".to_string(),
            tool_name: "list_directory".to_string(),
            arguments: r#"{"path":"src"}"#.to_string(),
            result: "lib.rs\nRelated.rs\n".to_string(),
            ..ToolExchange::default()
        };

        let nudges = deterministic_summary_nudges(&packet, &[listing]);

        assert_eq!(nudges.len(), 1);
        assert!(nudges[0].contains("src/Related.rs"));
    }

    #[test]
    fn deterministic_summary_nudge_suppressed_after_get_summaries() {
        let packet = TrainingPacket {
            files: vec![TrainingFile {
                path: "src/lib.rs".to_string(),
                diff: "diff".to_string(),
            }],
            related_files: vec![TrainingRelatedFile {
                path: "src/Related.rs".to_string(),
                summary: "related summary".to_string(),
            }],
        };
        let summary = ToolExchange {
            call_id: "summary".to_string(),
            tool_name: "get_summaries".to_string(),
            arguments: r#"{"targets":["src/Related.rs"]}"#.to_string(),
            result: "Summary for src/Related.rs".to_string(),
            ..ToolExchange::default()
        };

        assert!(deterministic_summary_nudges(&packet, &[summary]).is_empty());
    }

    struct HintBackend {
        attempts: Arc<AtomicUsize>,
        fail_until: usize,
        fail_first_incomplete: bool,
        last_prompt: Arc<Mutex<Option<String>>>,
    }

    impl LlmBackend for HintBackend {
        fn list_models(&self) -> BoxFuture<'_, anyhow::Result<Vec<String>>> {
            async { Ok(Vec::new()) }.boxed()
        }

        fn stream_chat(
            &self,
            _request: StreamChatRequest,
        ) -> BoxFuture<'_, anyhow::Result<LlmResponse>> {
            let attempts = self.attempts.clone();
            let fail_until = self.fail_until;
            let fail_first_incomplete = self.fail_first_incomplete;
            let last_prompt = self.last_prompt.clone();
            async move {
                let attempt = attempts.fetch_add(1, Ordering::SeqCst) + 1;
                if fail_first_incomplete && attempt == 1 {
                    return Err(anyhow::Error::new(
                        crate::llm_client::IncompleteStreamError::new(
                            "test SSE",
                            "response.completed",
                        ),
                    ));
                }
                if attempt <= fail_until {
                    anyhow::bail!("temporary failure");
                }
                let prompt = _request
                    .messages
                    .iter()
                    .flat_map(|message| message.content.iter())
                    .map(|part| format!("{part:?}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                *last_prompt.lock().unwrap() = Some(prompt);
                Ok(LlmResponse::Text {
                    text: r#"{"hint":"Consider searching for account."}"#.to_string(),
                    reasoning_content: None,
                    usage: TokenUsage::default(),
                })
            }
            .boxed()
        }
    }

    #[tokio::test]
    async fn hint_request_retries_three_times() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let backend: Arc<dyn LlmBackend> = Arc::new(HintBackend {
            attempts: attempts.clone(),
            fail_until: 2,
            fail_first_incomplete: false,
            last_prompt: Arc::new(Mutex::new(None)),
        });
        let packet = TrainingPacket {
            files: vec![TrainingFile {
                path: "src/lib.rs".to_string(),
                diff: "diff".to_string(),
            }],
            related_files: Vec::new(),
        };

        let nudge = compose_no_edit_nudge(
            &backend,
            8,
            &[ChatMessage {
                role: "user".to_string(),
                content: vec![ChatContentPart::text("fix account bug")],
                tool_calls: None,
                tool_call_id: None,
                name: None,
                reasoning_content: None,
            }],
            &[],
            &packet,
            &CancellationToken::new(),
            IdleTimeouts::uniform(Duration::from_secs(30)),
        )
        .await
        .expect("third attempt should produce a nudge");

        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        assert!(nudge.0.contains("Consider searching for account."));
    }

    #[tokio::test]
    async fn hint_request_retries_incomplete_stream_without_visible_output() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let backend: Arc<dyn LlmBackend> = Arc::new(HintBackend {
            attempts: attempts.clone(),
            fail_until: 0,
            fail_first_incomplete: true,
            last_prompt: Arc::new(Mutex::new(None)),
        });
        let packet = TrainingPacket {
            files: vec![TrainingFile {
                path: "src/lib.rs".to_string(),
                diff: "diff".to_string(),
            }],
            related_files: Vec::new(),
        };

        let nudge = compose_no_edit_nudge(
            &backend,
            8,
            &[ChatMessage {
                role: "user".to_string(),
                content: vec![ChatContentPart::text("fix account bug")],
                tool_calls: None,
                tool_call_id: None,
                name: None,
                reasoning_content: None,
            }],
            &[],
            &packet,
            &CancellationToken::new(),
            IdleTimeouts::uniform(Duration::from_secs(30)),
        )
        .await
        .expect("transport retry should recover hint request");

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert!(nudge.0.contains("Consider searching for account."));
    }

    #[tokio::test]
    async fn failed_hint_requests_skip_nudge() {
        let backend: Arc<dyn LlmBackend> = Arc::new(HintBackend {
            attempts: Arc::new(AtomicUsize::new(0)),
            fail_until: 3,
            fail_first_incomplete: false,
            last_prompt: Arc::new(Mutex::new(None)),
        });
        let packet = TrainingPacket {
            files: vec![TrainingFile {
                path: "src/lib.rs".to_string(),
                diff: "diff".to_string(),
            }],
            related_files: Vec::new(),
        };

        let nudge = compose_no_edit_nudge(
            &backend,
            8,
            &[],
            &[],
            &packet,
            &CancellationToken::new(),
            IdleTimeouts::uniform(Duration::from_secs(30)),
        )
        .await;

        assert!(nudge.is_none());
    }

    #[tokio::test]
    async fn all_files_read_emits_proceed_nudge_without_hint_call() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let backend: Arc<dyn LlmBackend> = Arc::new(HintBackend {
            attempts: attempts.clone(),
            fail_until: 0,
            fail_first_incomplete: false,
            last_prompt: Arc::new(Mutex::new(None)),
        });
        let packet = TrainingPacket {
            files: vec![TrainingFile {
                path: "src/lib.rs".to_string(),
                diff: "diff".to_string(),
            }],
            related_files: Vec::new(),
        };
        let source = ToolExchange {
            call_id: "source".to_string(),
            tool_name: "get_symbol_sources".to_string(),
            arguments: "{}".to_string(),
            result: "File: src/lib.rs\nfn main() {}".to_string(),
            ..ToolExchange::default()
        };

        let nudge = compose_no_edit_nudge(
            &backend,
            8,
            &[],
            &[source],
            &packet,
            &CancellationToken::new(),
            IdleTimeouts::uniform(Duration::from_secs(30)),
        )
        .await
        .expect("all-read state should nudge");

        assert_eq!(attempts.load(Ordering::SeqCst), 0);
        assert!(nudge.0.contains("smallest plausible solution"));
    }

    #[tokio::test]
    async fn hint_prompt_includes_sidecar_summaries_and_planned_nudges() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let last_prompt = Arc::new(Mutex::new(None));
        let backend: Arc<dyn LlmBackend> = Arc::new(HintBackend {
            attempts: attempts.clone(),
            fail_until: 0,
            fail_first_incomplete: false,
            last_prompt: last_prompt.clone(),
        });
        let packet = TrainingPacket {
            files: vec![TrainingFile {
                path: "src/lib.rs".to_string(),
                diff: "diff".to_string(),
            }],
            related_files: vec![TrainingRelatedFile {
                path: "src/Related.rs".to_string(),
                summary: "base-revision summary".to_string(),
            }],
        };
        let discovered = ToolExchange {
            call_id: "list".to_string(),
            tool_name: "list_directory".to_string(),
            arguments: r#"{"path":"src"}"#.to_string(),
            result: "lib.rs\nRelated.rs\n".to_string(),
            ..ToolExchange::default()
        };

        let nudge = compose_no_edit_nudge(
            &backend,
            8,
            &[],
            &[discovered],
            &packet,
            &CancellationToken::new(),
            IdleTimeouts::uniform(Duration::from_secs(30)),
        )
        .await
        .expect("nudge should include deterministic and flash hints");

        assert!(nudge.0.contains("Call get_summaries on src/Related.rs"));
        assert!(nudge.0.contains("Consider searching for account."));
        let user_prompt = last_prompt
            .lock()
            .unwrap()
            .clone()
            .expect("hint request should be captured");
        assert!(user_prompt.contains("<sidecar_file_summaries>"));
        assert!(user_prompt.contains("base-revision summary"));
        assert!(user_prompt.contains("<deterministic_nudges_already_planned>"));
        assert!(user_prompt.contains("Call get_summaries on src/Related.rs"));
    }

    #[test]
    fn hint_system_prompt_teaches_discovery_examples() {
        assert!(HINT_SYSTEM_PROMPT.contains("Do not reveal the golden diff's specific diagnosis"));
        assert!(HINT_SYSTEM_PROMPT.contains("Bad: Focus on the leading character"));
        assert!(
            HINT_SYSTEM_PROMPT.contains("Good: Consider inspecting how multimodal text blocks")
        );
        assert!(HINT_SYSTEM_PROMPT.contains("Return strict JSON only"));
    }
}
