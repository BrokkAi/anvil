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

/// Cap on a cited result rendered verbatim. A citation says "this result is
/// why I believe what I claim", so the supervisor should not have to fetch it
/// again - but a runaway result must not swallow the review either, and the
/// handle still resolves the untruncated original through `view_tool_call`.
const COMPACT_CITED_RESULT_LIMIT: usize = 8_000;

/// Tail budget for the trailing results of a window: the last few lines, and
/// never more than this many characters of them.
const COMPACT_TAIL_LINES: usize = 10;
const COMPACT_TAIL_CHARS: usize = 800;

/// How many trailing tool results of a window keep a verbatim tail. Evidence
/// produced immediately before a forced report is exactly what a summary line
/// cannot stand in for.
const COMPACT_TAIL_RESULTS: usize = 2;

/// Cap on the first non-empty line of a failed result kept verbatim.
const COMPACT_FAILURE_LINE_LIMIT: usize = 200;

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

/// Mints the v2 worker-scoped handle for the tool result at `index`.
pub(crate) fn worker_tool_handle(worker: usize, index: usize) -> String {
    format!("w{worker}m{index}")
}

/// Inverse of [`worker_tool_handle`], returning `(worker, index)`.
pub(crate) fn parse_worker_tool_handle(handle: &str) -> Option<(usize, usize)> {
    let rest = handle.strip_prefix('w')?;
    let (worker, index) = rest.split_once('m')?;
    Some((worker.parse().ok()?, index.parse().ok()?))
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

/// Whether a tool result reports a failure. Anvil has no typed failures, so
/// this is the same textual evidence `summarize_tool_result` keys on: an
/// `Error:` prefix, or a shell command that reported a non-zero exit code.
fn result_failed(result: &str) -> bool {
    result_error(result).is_some() || shell_exit_code(result).is_some_and(|code| code != 0)
}

/// The first non-empty line of a failed result, capped. Modeled on mjolnir's
/// review renderer, which appends the first error line to a failed tool's
/// delta: the size floor elides the line that says *what* broke, and the
/// supervisor cannot judge a failure it cannot see.
fn failure_first_line(result: &str) -> Option<String> {
    if !result_failed(result) {
        return None;
    }
    let line = result
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())?;
    Some(truncate_summary(line, COMPACT_FAILURE_LINE_LIMIT))
}

/// The last [`COMPACT_TAIL_LINES`] lines of a result, capped at
/// [`COMPACT_TAIL_CHARS`] characters.
fn result_tail(result: &str) -> String {
    let trimmed = result.trim_end();
    let lines = trimmed.lines().collect::<Vec<_>>();
    let start = lines.len().saturating_sub(COMPACT_TAIL_LINES);
    let tail = lines[start..].join("\n");
    let length = tail.chars().count();
    if length <= COMPACT_TAIL_CHARS {
        tail
    } else {
        tail.chars().skip(length - COMPACT_TAIL_CHARS).collect()
    }
}

/// Whether `report` cites `handle` as an exact token. Purely mechanical: the
/// handle must appear with non-alphanumeric neighbours, so "w3m1" in a report
/// does not match handle "w3m12" and prose can never be parsed for intent.
pub(crate) fn report_cites_handle(report: &str, handle: &str) -> bool {
    if handle.is_empty() {
        return false;
    }
    let bytes = report.as_bytes();
    report.match_indices(handle).any(|(start, _)| {
        let end = start + handle.len();
        let before_ok = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
        let after_ok = end >= bytes.len() || !bytes[end].is_ascii_alphanumeric();
        before_ok && after_ok
    })
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

/// Renders one v2 worker window compactly.
pub(crate) fn render_window_compact_for_worker(worker: usize, messages: &[ChatMessage]) -> String {
    let report = crate::asgard::extract_worker_final_response(messages);
    render_window_compact_with_handle(messages, &report, |index| worker_tool_handle(worker, index))
}

/// Indexes of the last [`COMPACT_TAIL_RESULTS`] tool results in the window.
fn trailing_result_indexes(messages: &[ChatMessage]) -> std::collections::HashSet<usize> {
    let results = messages
        .iter()
        .enumerate()
        .filter(|(_, message)| message.role == "tool")
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    results
        .iter()
        .rev()
        .take(COMPACT_TAIL_RESULTS)
        .copied()
        .collect()
}

/// `report` is the worker's own final message, scanned only for exact result
/// handles: a cited result is rendered verbatim instead of elided. Nothing
/// else about the report is interpreted.
fn render_window_compact_with_handle(
    messages: &[ChatMessage],
    report: &str,
    handle_for_index: impl Fn(usize) -> String,
) -> String {
    let mut rendered = String::new();
    let mut seen_results: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    let trailing = trailing_result_indexes(messages);

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
                let handle = handle_for_index(index);
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
                    "<tool name=\"{tool}\" id=\"{handle}\">{}",
                    escape_content(&summary)
                ));
                // Retention, in precedence order. A cited result is already
                // whole, so neither of the partial retentions adds anything
                // to it.
                if report_cites_handle(report, &handle) {
                    rendered.push_str(&format!(
                        "\n<cited_result>{}</cited_result>",
                        escape_content(&truncate_summary(&raw, COMPACT_CITED_RESULT_LIMIT))
                    ));
                } else {
                    if let Some(line) = failure_first_line(&raw) {
                        rendered.push_str(&format!(
                            "\n<failure_line>{}</failure_line>",
                            escape_content(&line)
                        ));
                    }
                    if trailing.contains(&index) && !raw.trim().is_empty() {
                        rendered.push_str(&format!(
                            "\n<result_tail>{}</result_tail>",
                            escape_content(&result_tail(&raw))
                        ));
                    }
                }
                rendered.push_str("</tool>\n");
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
    fn worker_handles_round_trip() {
        let handle = worker_tool_handle(7, 11);
        assert_eq!(handle, "w7m11");
        assert_eq!(parse_worker_tool_handle(&handle), Some((7, 11)));
        assert_eq!(parse_worker_tool_handle("nonsense"), None);
    }

    #[test]
    fn worker_handles_reject_v1_lane_handles() {
        assert_eq!(parse_worker_tool_handle("w3l1m7"), None);
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
            assistant_call(call("c3", "list_directory", r#"{"path":"src"}"#)),
            ChatMessage::tool_result("c3", "list_directory", "main.rs".to_string()),
            assistant_call(call("c4", "list_directory", r#"{"path":"docs"}"#)),
            ChatMessage::tool_result("c4", "list_directory", "readme.md".to_string()),
        ];
        let rendered = render_window_compact_for_worker(2, &messages);
        assert!(rendered.contains(
            r#"<tool name="read_file" id="w2m1">read src/main.rs (9000 chars omitted)</tool>"#
        ));
        assert!(rendered.contains(r#"id="w2m3">$ "cargo test" (exit 0)"#));
        // A big, uncited, mid-window success result never reaches the
        // supervisor verbatim - that size floor is the whole point.
        assert!(!rendered.contains(&"x".repeat(100)));
    }

    #[test]
    fn failed_results_keep_their_first_line_however_large() {
        let mut noise = "error[E0308]: mismatched types\n".to_string();
        noise.push_str(&"context line\n".repeat(2_000));
        let messages = vec![
            assistant_call(call(
                "c1",
                "run_shell_command",
                r#"{"command":"cargo build"}"#,
            )),
            ChatMessage::tool_result("c1", "run_shell_command", format!("{noise}Exit code: 101")),
            assistant_call(call("c2", "read_file", r#"{"file_path":"missing.rs"}"#)),
            ChatMessage::tool_result("c2", "read_file", "Error: no such file".to_string()),
            assistant_call(call("c3", "list_directory", r#"{"path":"src"}"#)),
            ChatMessage::tool_result("c3", "list_directory", "main.rs".to_string()),
            assistant_call(call("c4", "list_directory", r#"{"path":"docs"}"#)),
            ChatMessage::tool_result("c4", "list_directory", "readme.md".to_string()),
        ];
        let rendered = render_window_compact_for_worker(3, &messages);
        assert!(
            rendered.contains("<failure_line>error[E0308]: mismatched types</failure_line>"),
            "failed shell result should keep its first line:\n{rendered}"
        );
        assert!(
            rendered.contains("<failure_line>Error: no such file</failure_line>"),
            "failed tool result should keep its first line:\n{rendered}"
        );
        // The rest of the failure is still elided.
        assert!(!rendered.contains(&"context line\n".repeat(4)));
    }

    #[test]
    fn a_result_the_report_cites_by_handle_renders_whole() {
        let cited = format!("SUITE OUTPUT\n{}\n42 passed", "detail line\n".repeat(60));
        let messages = vec![
            assistant_call(call(
                "c1",
                "run_shell_command",
                r#"{"command":"cargo test"}"#,
            )),
            ChatMessage::tool_result("c1", "run_shell_command", cited.clone()),
            assistant_call(call("c2", "list_directory", r#"{"path":"src"}"#)),
            ChatMessage::tool_result("c2", "list_directory", "main.rs".to_string()),
            assistant_call(call("c3", "list_directory", r#"{"path":"docs"}"#)),
            ChatMessage::tool_result("c3", "list_directory", "readme.md".to_string()),
            ChatMessage::assistant("Suite is green; see w5m1 for the full run.".to_string()),
        ];
        let rendered = render_window_compact_for_worker(5, &messages);
        assert!(
            rendered.contains(&format!("<cited_result>{}</cited_result>", cited)),
            "cited result should render whole:\n{rendered}"
        );
    }

    #[test]
    fn an_uncited_handle_prefix_does_not_count_as_a_citation() {
        let big = "payload line\n".repeat(200);
        let messages = vec![
            assistant_call(call("c1", "read_file", r#"{"file_path":"a.rs"}"#)),
            ChatMessage::tool_result("c1", "read_file", big.clone()),
            assistant_call(call("c2", "list_directory", r#"{"path":"src"}"#)),
            ChatMessage::tool_result("c2", "list_directory", "main.rs".to_string()),
            assistant_call(call("c3", "list_directory", r#"{"path":"docs"}"#)),
            ChatMessage::tool_result("c3", "list_directory", "readme.md".to_string()),
            // w5m11 is a different handle; the substring w5m1 must not match.
            ChatMessage::assistant("see w5m11 for details".to_string()),
        ];
        let rendered = render_window_compact_for_worker(5, &messages);
        assert!(!rendered.contains("<cited_result>"));
        assert!(!rendered.contains(&"payload line\n".repeat(4)));
    }

    #[test]
    fn the_last_two_results_keep_a_verbatim_tail() {
        let long = |file: &str| {
            (0..40)
                .map(|index| format!("{file} line {index}"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let messages = vec![
            assistant_call(call("c1", "read_file", r#"{"file_path":"a.rs"}"#)),
            ChatMessage::tool_result("c1", "read_file", long("a")),
            assistant_call(call("c2", "read_file", r#"{"file_path":"b.rs"}"#)),
            ChatMessage::tool_result("c2", "read_file", long("b")),
            assistant_call(call("c3", "read_file", r#"{"file_path":"c.rs"}"#)),
            ChatMessage::tool_result("c3", "read_file", long("c")),
        ];
        let rendered = render_window_compact_for_worker(6, &messages);
        // The last two results keep their last 10 lines; the first does not.
        assert_eq!(rendered.matches("<result_tail>").count(), 2);
        assert!(rendered.contains("c line 30\nc line 31"));
        assert!(rendered.contains("b line 39"));
        // The first result stays elided: its tail never appears.
        assert!(!rendered.contains("a line 39"));
    }

    #[test]
    fn handle_citation_detection_is_exact_token_matching() {
        assert!(report_cites_handle("evidence: w3m5 is the run", "w3m5"));
        assert!(report_cites_handle("(w3m5)", "w3m5"));
        assert!(report_cites_handle("w3m5", "w3m5"));
        assert!(!report_cites_handle("w3m50", "w3m5"));
        assert!(!report_cites_handle("aw3m5", "w3m5"));
        assert!(!report_cites_handle("nothing here", "w3m5"));
    }

    #[test]
    fn worker_render_stamps_worker_ids_and_deduplicates_results() {
        let messages = vec![
            assistant_call(call("c1", "run_shell_command", r#"{"command":"ls"}"#)),
            ChatMessage::tool_result("c1", "run_shell_command", "same output".to_string()),
            assistant_call(call("c2", "run_shell_command", r#"{"command":"pwd"}"#)),
            ChatMessage::tool_result("c2", "run_shell_command", "same output".to_string()),
        ];
        let rendered = render_window_compact_for_worker(7, &messages);
        assert!(rendered.contains(r#"id="w7m1""#));
        assert!(rendered.contains(r#"id="w7m3" exact_duplicate_of="w7m1""#));
        assert!(!rendered.contains("w7l"));
    }

    #[test]
    fn identical_results_collapse_to_a_back_reference() {
        let messages = vec![
            assistant_call(call("c1", "run_shell_command", r#"{"command":"ls"}"#)),
            ChatMessage::tool_result("c1", "run_shell_command", "same output".to_string()),
            assistant_call(call("c2", "run_shell_command", r#"{"command":"ls"}"#)),
            ChatMessage::tool_result("c2", "run_shell_command", "same output".to_string()),
        ];
        let rendered = render_window_compact_for_worker(4, &messages);
        assert!(rendered.contains(r#"id="w4m3" exact_duplicate_of="w4m1""#));
    }

    #[test]
    fn reasoning_is_compacted_to_a_length_marker() {
        let mut message = ChatMessage::assistant(String::new());
        message.reasoning_content = Some("a".repeat(4_000));
        let rendered = render_window_compact_for_worker(4, &[message]);
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
