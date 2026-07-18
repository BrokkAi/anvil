//! Compact, deterministic rendering of a candidate work window for the Asgard
//! supervisor.
//!
//! The supervisor cannot afford to read every candidate's full trajectory, but
//! anything it cannot read it must be able to fetch. This module renders each
//! window entry to a minimal line and stamps every tool result with a stable
//! handle; `view_tool_call` resolves a handle back to the original arguments
//! and the complete untruncated result.
//!
//! The design follows Slate 1.0.42's compact trace renderer, adapted to Anvil's
//! tool set. Two deliberate departures:
//!
//! * Handles encode window and lane. Slate derives them from session and
//!   message identifiers, which are unique because one session writes them.
//!   Asgard runs candidates concurrently and each lane is an independent LLM
//!   stream, so provider tool-call identifiers can collide between lanes.
//! * Anvil tool results are plain strings, not typed envelopes, so the
//!   per-tool summaries are built from the call arguments plus cheap structural
//!   facts about the result (length, exit code, error prefix) rather than from
//!   result fields.
//!
//! Nothing here consults an LLM. Every rendering is a pure function of the
//! window, which is the point: a summarizer that infers what a window
//! accomplished can report work that never happened, and the supervisor has no
//! way to tell.

use crate::llm_client::{ChatContentPart, ChatMessage};

/// Cap for inline prose (reasoning, assistant text, user and system messages).
pub(crate) const COMPACT_TEXT_LIMIT: usize = 120;

/// Cap for a rendered tool-result summary.
pub(crate) const COMPACT_TOOL_SUMMARY_LIMIT: usize = 200;

/// Cap for an argument value quoted inside a tool summary.
const COMPACT_ARGUMENT_LIMIT: usize = 80;

pub(crate) struct CompactText {
    pub(crate) text: String,
    pub(crate) len: usize,
    pub(crate) truncated: bool,
}

impl CompactText {
    fn render(&self) -> String {
        if self.truncated {
            format!("{}…", self.text)
        } else {
            self.text.clone()
        }
    }
}

/// Reduces a value to its first line, capped at `limit` characters, while
/// reporting the original length so the supervisor knows how much it is not
/// seeing.
pub(crate) fn compact_text(value: &str, limit: usize) -> CompactText {
    let cleaned = value
        .replace(['\u{200d}', '\u{200b}', '\u{feff}'], "")
        .trim()
        .to_string();
    let len = cleaned.chars().count();
    let first_line = cleaned.lines().next().unwrap_or_default();
    let mut truncated = first_line.chars().count() < len;
    let text = if first_line.chars().count() > limit {
        truncated = true;
        first_line.chars().take(limit).collect::<String>()
    } else {
        first_line.to_string()
    };
    CompactText {
        text,
        len,
        truncated,
    }
}

fn truncate_summary(text: &str, max_len: usize) -> String {
    if text.chars().count() <= max_len {
        return text.to_string();
    }
    let kept = max_len.saturating_sub(24);
    let head = text.chars().take(kept).collect::<String>();
    let dropped = text.chars().count() - max_len;
    format!("{head}…[{dropped} more chars]")
}

/// Escapes element content. Quotes are left alone: summaries quote commands and
/// patterns, and `&quot;`-ing them makes shell lines unreadable for no benefit,
/// since content is not attribute-delimited.
fn escape_content(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Mints the handle for the tool result at `index` in lane `lane`'s window.
///
/// Collision-free across concurrent candidates and across windows by
/// construction, and self-describing: the supervisor can tell which lane a
/// handle belongs to before spending a call on it.
pub(crate) fn asgard_tool_handle(window: usize, lane: usize, index: usize) -> String {
    format!("w{window}l{lane}m{index}")
}

/// Inverse of [`asgard_tool_handle`], returning `(window, lane, index)`.
pub(crate) fn parse_asgard_tool_handle(handle: &str) -> Option<(usize, usize, usize)> {
    let rest = handle.strip_prefix('w')?;
    let (window, rest) = rest.split_once('l')?;
    let (lane, index) = rest.split_once('m')?;
    Some((
        window.parse().ok()?,
        lane.parse().ok()?,
        index.parse().ok()?,
    ))
}

fn argument_str<'a>(arguments: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    arguments.get(key).and_then(serde_json::Value::as_str)
}

fn quoted_argument(arguments: &serde_json::Value, key: &str) -> Option<String> {
    argument_str(arguments, key)
        .map(|value| format!("{:?}", truncate_summary(value, COMPACT_ARGUMENT_LIMIT)))
}

/// Anvil surfaces tool failures as an `Error:`-prefixed result string rather
/// than a typed failure, so failure detection is textual.
fn result_error(result: &str) -> Option<String> {
    let trimmed = result.trim_start();
    trimmed
        .strip_prefix("Error:")
        .map(|detail| compact_text(detail, COMPACT_ARGUMENT_LIMIT).render())
}

fn shell_exit_code(result: &str) -> Option<i32> {
    let regex = regex::Regex::new(r"Exit code: (-?\d+)|Command completed with exit code (-?\d+)")
        .expect("valid exit-code regex");
    regex
        .captures_iter(result)
        .last()
        .and_then(|captures| captures.get(1).or_else(|| captures.get(2)))
        .and_then(|capture| capture.as_str().parse::<i32>().ok())
}

fn omitted(result: &str) -> String {
    if result.is_empty() {
        String::new()
    } else {
        format!(" ({} chars omitted)", result.len())
    }
}

/// Renders a deterministic one-line summary of a tool result.
///
/// Every branch reports what the call *was* and how large the result is; none
/// of them characterizes what the result means. Judging significance is the
/// supervisor's job, and it can fetch the full result by handle.
pub(crate) fn summarize_tool_result(
    tool: &str,
    arguments: &serde_json::Value,
    result: &str,
    max_len: usize,
) -> String {
    if let Some(error) = result_error(result) {
        return truncate_summary(&format!("failed: {error}"), max_len);
    }
    let summary = match tool {
        "read_file" => {
            let path = argument_str(arguments, "file_path").unwrap_or("?");
            let range = match (
                arguments.get("offset").and_then(serde_json::Value::as_u64),
                arguments.get("limit").and_then(serde_json::Value::as_u64),
            ) {
                (Some(offset), Some(limit)) => format!(" lines {offset}-{}", offset + limit),
                (Some(offset), None) => format!(" from line {offset}"),
                _ => String::new(),
            };
            format!("read {path}{range}{}", omitted(result))
        }
        "write_file" => {
            let path = argument_str(arguments, "file_path").unwrap_or("?");
            let bytes = argument_str(arguments, "content").map_or(0, str::len);
            format!("write {path} ({bytes} chars written)")
        }
        "edit" => {
            let path = argument_str(arguments, "file_path").unwrap_or("?");
            let replace_all = arguments
                .get("replace_all")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            let scope = if replace_all {
                " (all occurrences)"
            } else {
                ""
            };
            format!("edit {path}{scope}: ok")
        }
        "list_directory" => {
            let path = argument_str(arguments, "path").unwrap_or(".");
            format!("listed {path} ({} entries)", result.lines().count())
        }
        "grep_search" => {
            let pattern = quoted_argument(arguments, "pattern").unwrap_or_else(|| "?".to_string());
            let scope = argument_str(arguments, "path")
                .map(|path| format!(" in {path}"))
                .unwrap_or_default();
            let glob = argument_str(arguments, "glob")
                .map(|glob| format!(" matching {glob}"))
                .unwrap_or_default();
            format!(
                "grep {pattern}{scope}{glob} ({} lines omitted)",
                result.lines().count()
            )
        }
        "run_shell_command" => {
            let command = quoted_argument(arguments, "command").unwrap_or_else(|| "?".to_string());
            let exit = shell_exit_code(result)
                .map(|code| format!(" (exit {code})"))
                .unwrap_or_default();
            format!("$ {command}{exit}{}", omitted(result))
        }
        "web_search" => {
            let query = quoted_argument(arguments, "query").unwrap_or_else(|| "?".to_string());
            format!("websearch {query}{}", omitted(result))
        }
        "update_plan" => {
            let steps = arguments
                .get("plan")
                .and_then(serde_json::Value::as_array)
                .map_or(0, Vec::len);
            format!("plan updated ({steps} steps)")
        }
        _ => {
            let first = arguments
                .as_object()
                .and_then(|object| object.iter().next())
                .map(|(key, value)| match value.as_str() {
                    Some(text) => {
                        format!("{key}={:?}", truncate_summary(text, COMPACT_ARGUMENT_LIMIT))
                    }
                    None => format!(
                        "{key}={}",
                        truncate_summary(&value.to_string(), COMPACT_ARGUMENT_LIMIT)
                    ),
                })
                .map(|argument| format!(" {argument}"))
                .unwrap_or_default();
            format!("{tool}{first}{}", omitted(result))
        }
    };
    truncate_summary(&summary, max_len)
}

fn message_text(message: &ChatMessage) -> String {
    message
        .content
        .iter()
        .filter_map(|part| match part {
            ChatContentPart::Text { text } => Some(text.as_str()),
            ChatContentPart::Image { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Resolves the tool call that produced the result at `index`, matching Slate's
/// reverse walk from the result back to the assistant message that issued it.
pub(crate) fn originating_tool_call(
    messages: &[ChatMessage],
    index: usize,
) -> Option<&crate::llm_client::ToolCall> {
    let result = messages.get(index)?;
    let tool_call_id = result.tool_call_id.as_deref()?;
    messages[..index]
        .iter()
        .rev()
        .filter(|message| message.role == "assistant")
        .flat_map(|message| message.tool_calls.iter().flatten())
        .find(|call| call.id == tool_call_id)
}

/// Renders one candidate window compactly, stamping each tool result with a
/// handle and collapsing byte-identical repeated results to a back-reference.
///
/// The dedup matters more under compaction, not less: two identical compact
/// `<tool>` lines read as two independent confirmations of the same fact, which
/// is exactly the double-counting a supervisor weighing evidence must not do.
pub(crate) fn render_window_compact(
    window: usize,
    lane: usize,
    messages: &[ChatMessage],
) -> String {
    let mut rendered = String::new();
    let mut seen_results: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    for (index, message) in messages.iter().enumerate() {
        if let Some(reasoning) = &message.reasoning_content {
            let compact = compact_text(reasoning, COMPACT_TEXT_LIMIT);
            if compact.len > 0 {
                rendered.push_str(&format!(
                    "<thinking len=\"{}\">{}</thinking>\n",
                    compact.len,
                    escape_content(&compact.render())
                ));
            }
        }

        match message.role.as_str() {
            "tool" => {
                let raw = message_text(message);
                let handle = asgard_tool_handle(window, lane, index);
                let call = originating_tool_call(messages, index);
                let tool = call
                    .map(|call| call.function.name.as_str())
                    .or(message.name.as_deref())
                    .unwrap_or("tool");
                if let Some(first_handle) = seen_results.get(&raw) {
                    rendered.push_str(&format!(
                        "<tool name=\"{tool}\" id=\"{handle}\" exact_duplicate_of=\"{first_handle}\" />\n"
                    ));
                    continue;
                }
                seen_results.insert(raw.clone(), handle.clone());
                let arguments = call
                    .and_then(|call| {
                        serde_json::from_str::<serde_json::Value>(&call.function.arguments).ok()
                    })
                    .unwrap_or_else(|| serde_json::json!({}));
                let summary =
                    summarize_tool_result(tool, &arguments, &raw, COMPACT_TOOL_SUMMARY_LIMIT);
                rendered.push_str(&format!(
                    "<tool name=\"{tool}\" id=\"{handle}\">{}</tool>\n",
                    escape_content(&summary)
                ));
            }
            role => {
                let text = message_text(message);
                let compact = compact_text(&text, COMPACT_TEXT_LIMIT);
                if compact.len == 0 {
                    continue;
                }
                let tag = match role {
                    "assistant" => "output",
                    "user" => "user",
                    "system" => "system",
                    other => other,
                };
                rendered.push_str(&format!(
                    "<{tag} len=\"{}\">{}</{tag}>\n",
                    compact.len,
                    escape_content(&compact.render())
                ));
            }
        }
    }
    rendered
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_client::{FunctionCall, ToolCall};

    fn call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            r#type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    fn assistant_call(call: ToolCall) -> ChatMessage {
        ChatMessage::assistant_tool_calls_with_content_and_reasoning(
            String::new(),
            vec![call],
            None,
        )
    }

    #[test]
    fn handles_round_trip() {
        let handle = asgard_tool_handle(3, 1, 7);
        assert_eq!(handle, "w3l1m7");
        assert_eq!(parse_asgard_tool_handle(&handle), Some((3, 1, 7)));
        assert_eq!(parse_asgard_tool_handle("nonsense"), None);
    }

    #[test]
    fn handles_are_unique_across_concurrent_lanes_and_windows() {
        // Two lanes running the same window at the same message index, and the
        // same lane in a later window, must never share a handle.
        let mut handles = std::collections::HashSet::new();
        for window in 0..3 {
            for lane in 0..4 {
                for index in 0..5 {
                    assert!(handles.insert(asgard_tool_handle(window, lane, index)));
                }
            }
        }
        assert_eq!(handles.len(), 60);
    }

    #[test]
    fn tool_results_render_with_handles_and_deterministic_summaries() {
        let messages = vec![
            assistant_call(call("c1", "read_file", r#"{"file_path":"src/main.rs"}"#)),
            ChatMessage::tool_result("c1", "read_file", "x".repeat(9_000)),
            assistant_call(call(
                "c2",
                "run_shell_command",
                r#"{"command":"cargo test"}"#,
            )),
            ChatMessage::tool_result("c2", "run_shell_command", "ok\nExit code: 0".to_string()),
        ];
        let rendered = render_window_compact(2, 1, &messages);
        assert!(rendered.contains(
            r#"<tool name="read_file" id="w2l1m1">read src/main.rs (9000 chars omitted)</tool>"#
        ));
        assert!(rendered.contains(r#"id="w2l1m3">$ "cargo test" (exit 0)"#));
        // The 9,000-char result never reaches the supervisor verbatim.
        assert!(!rendered.contains(&"x".repeat(100)));
    }

    #[test]
    fn identical_results_collapse_to_a_back_reference() {
        let messages = vec![
            assistant_call(call("c1", "run_shell_command", r#"{"command":"ls"}"#)),
            ChatMessage::tool_result("c1", "run_shell_command", "same output".to_string()),
            assistant_call(call("c2", "run_shell_command", r#"{"command":"ls"}"#)),
            ChatMessage::tool_result("c2", "run_shell_command", "same output".to_string()),
        ];
        let rendered = render_window_compact(0, 0, &messages);
        assert!(rendered.contains(r#"id="w0l0m3" exact_duplicate_of="w0l0m1""#));
    }

    #[test]
    fn reasoning_is_compacted_to_a_length_marker() {
        let mut message = ChatMessage::assistant(String::new());
        message.reasoning_content = Some("a".repeat(4_000));
        let rendered = render_window_compact(0, 0, &[message]);
        assert!(rendered.contains(r#"<thinking len="4000">"#));
        assert!(rendered.contains('…'));
        assert!(!rendered.contains(&"a".repeat(200)));
    }

    #[test]
    fn failures_are_reported_as_failures() {
        let summary = summarize_tool_result(
            "read_file",
            &serde_json::json!({"file_path": "missing.rs"}),
            "Error: no such file",
            COMPACT_TOOL_SUMMARY_LIMIT,
        );
        assert_eq!(summary, "failed: no such file");
    }

    #[test]
    fn every_builtin_tool_has_a_specific_summary() {
        // Guards against a tool silently falling through to the generic branch,
        // which would tell the supervisor nothing it did not already see on the
        // call itself.
        for (tool, arguments, expected) in [
            (
                "read_file",
                serde_json::json!({"file_path": "a.rs"}),
                "read a.rs",
            ),
            (
                "write_file",
                serde_json::json!({"file_path": "a.rs", "content": "hi"}),
                "write a.rs (2 chars written)",
            ),
            (
                "edit",
                serde_json::json!({"file_path": "a.rs"}),
                "edit a.rs: ok",
            ),
            (
                "list_directory",
                serde_json::json!({"path": "src"}),
                "listed src",
            ),
            (
                "grep_search",
                serde_json::json!({"pattern": "fn main"}),
                r#"grep "fn main""#,
            ),
            (
                "web_search",
                serde_json::json!({"query": "rust"}),
                r#"websearch "rust""#,
            ),
            (
                "run_shell_command",
                serde_json::json!({"command": "ls"}),
                r#"$ "ls""#,
            ),
            (
                "update_plan",
                serde_json::json!({"plan": []}),
                "plan updated (0 steps)",
            ),
        ] {
            let summary = summarize_tool_result(tool, &arguments, "output", 200);
            assert!(
                summary.contains(expected),
                "{tool} summary {summary:?} does not contain {expected:?}"
            );
        }
    }

    #[test]
    fn originating_call_is_found_across_interleaved_messages() {
        let messages = vec![
            assistant_call(call("c1", "read_file", r#"{"file_path":"a.rs"}"#)),
            assistant_call(call("c2", "read_file", r#"{"file_path":"b.rs"}"#)),
            ChatMessage::tool_result("c2", "read_file", "b".to_string()),
            ChatMessage::tool_result("c1", "read_file", "a".to_string()),
        ];
        assert_eq!(
            originating_tool_call(&messages, 2).map(|call| call.function.arguments.as_str()),
            Some(r#"{"file_path":"b.rs"}"#)
        );
        assert_eq!(
            originating_tool_call(&messages, 3).map(|call| call.function.arguments.as_str()),
            Some(r#"{"file_path":"a.rs"}"#)
        );
    }
}
