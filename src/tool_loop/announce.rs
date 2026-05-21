//! Builders that turn a tool name + parsed JSON args into the ACP
//! `ToolCall` / `ToolCallUpdate` payloads our tool loop sends to the client.
//!
//! Kept side-effect-free so the title/location logic is unit-testable
//! without an ACP connection. The actual `cx.send_notification(...)` calls
//! live in `tool_loop.rs`, which decides *when* to emit each lifecycle
//! event (Pending -> InProgress -> Completed/Failed).

use std::path::PathBuf;

use agent_client_protocol::schema::{
    Content, ContentBlock, Diff, TextContent, ToolCall, ToolCallContent, ToolCallId,
    ToolCallLocation, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use serde_json::Value;

use crate::tools::ToolRegistry;

/// Cap for inline text content we put on a Completed/Failed update. Keeps
/// the wire payload bounded; the LLM-facing `raw_output` is bounded
/// separately by `tool_loop::MAX_TOOL_RESULT_BYTES`.
const MAX_INLINE_OUTPUT_BYTES: usize = 50_000;

/// Cap on the rendered tool-call title's count of Unicode scalars
/// (`str::chars`). Clients render this title at the top of the permission
/// dialog with a fixed slot; past this width the title wraps onto
/// multiple lines and pushes the Approve/Reject buttons (or even
/// content) off-screen, leading users to authorize calls they can't
/// fully see. Observed identically across multiple ACP clients, since
/// they all rely on the title we send.
///
/// Counting scalars (rather than grapheme clusters or terminal display
/// width) is an approximation: a CJK fullwidth string at exactly the cap
/// renders ~2x wider than an ASCII string at the cap. Adequate for the
/// dominant case of LLM-emitted shell commands and paths, which are
/// nearly all ASCII; a future tightening could switch to
/// `unicode_width::UnicodeWidthStr` if we observe CJK-only blowups.
///
/// We deliberately *reject* over-cap titles rather than truncating: a
/// truncated title hides information from the approver, which is the
/// exact failure mode we're trying to prevent. The LLM gets a clear
/// error and is expected to retry with smaller arguments (shorter
/// command, narrower pattern, etc.).
pub(super) const MAX_TOOL_TITLE_CHARS: usize = 1024;

/// Build the canned rejection text fed back to the LLM (and surfaced on
/// the Failed tool-call card) when a title would exceed
/// `MAX_TOOL_TITLE_CHARS`. Derived from the constant so the number
/// quoted to the model never drifts from the enforced cap.
fn title_too_long_reason() -> String {
    format!(
        "Tool use denied: the rendered tool-call title would exceed {MAX_TOOL_TITLE_CHARS} \
         characters, which clips the approval dialog and would hide what's being \
         authorized. Retry with smaller arguments (shorter command, narrower pattern, etc.)."
    )
}

/// Build the initial `Pending` tool call -- the card the client renders
/// before we run the permission gate.
///
/// Assumes the caller has already gated the title length via
/// `rejection_for_oversized_title`; a debug assertion catches any future
/// path that emits an oversized title without going through the gate.
/// In release builds the assertion compiles out, so we degrade to the
/// pre-cap behavior rather than panicking on user input.
pub(super) fn initial_tool_call(
    tool_call_id: &str,
    tool_name: &str,
    kind: ToolKind,
    raw_input: &Value,
) -> ToolCall {
    let title = tool_title(tool_name, raw_input);
    debug_assert!(
        title.chars().count() <= MAX_TOOL_TITLE_CHARS,
        "initial_tool_call: oversized title bypassed the pre-gate check \
         (tool={tool_name}, chars={})",
        title.chars().count()
    );
    ToolCall::new(ToolCallId::new(tool_call_id.to_string()), title)
        .kind(kind)
        .status(ToolCallStatus::Pending)
        .raw_input(raw_input.clone())
        .locations(tool_locations(tool_name, raw_input))
}

/// If the title we'd render for this call exceeds `MAX_TOOL_TITLE_CHARS`,
/// return the rejection message. `None` means the call's title fits and
/// it can proceed to the normal Pending → gate flow.
pub(super) fn rejection_for_oversized_title(tool_name: &str, raw_input: &Value) -> Option<String> {
    if tool_title(tool_name, raw_input).chars().count() > MAX_TOOL_TITLE_CHARS {
        Some(title_too_long_reason())
    } else {
        None
    }
}

/// Pending card for a call we're about to refuse because its full title
/// would be too long. Uses only the static display name as the title --
/// no user-controlled input -- so the rejection card itself can't trigger
/// the same dialog-clipping problem. `raw_input` is still attached so the
/// user sees *what* was rejected.
pub(super) fn rejected_initial_tool_call(
    tool_call_id: &str,
    tool_name: &str,
    kind: ToolKind,
    raw_input: &Value,
) -> ToolCall {
    ToolCall::new(
        ToolCallId::new(tool_call_id.to_string()),
        ToolRegistry::display_name(tool_name).to_string(),
    )
    .kind(kind)
    .status(ToolCallStatus::Pending)
    .raw_input(raw_input.clone())
    .locations(tool_locations(tool_name, raw_input))
}

/// Mark the tool as actively running. Sent once the gate clears.
pub(super) fn update_in_progress(tool_call_id: &str) -> ToolCallUpdate {
    let fields = ToolCallUpdateFields::new().status(ToolCallStatus::InProgress);
    ToolCallUpdate::new(ToolCallId::new(tool_call_id.to_string()), fields)
}

/// Terminal `Failed` update -- denial messages, internal errors, etc.
/// `reason` is shown inline in the card; `raw_output` (when available)
/// preserves the full tool error for clients that surface it.
pub(super) fn update_failed(
    tool_call_id: &str,
    reason: &str,
    raw_output: Option<Value>,
) -> ToolCallUpdate {
    let mut fields = ToolCallUpdateFields::new()
        .status(ToolCallStatus::Failed)
        .content(vec![text_content(reason)]);
    if let Some(raw) = raw_output {
        fields = fields.raw_output(raw);
    }
    ToolCallUpdate::new(ToolCallId::new(tool_call_id.to_string()), fields)
}

/// Terminal `Completed` update. Pass `Some(diff)` for `writeFile` to
/// render an inline diff; otherwise the `output` is shown as text.
pub(super) fn update_completed(
    tool_call_id: &str,
    output: &str,
    diff: Option<Diff>,
) -> ToolCallUpdate {
    let content = match diff {
        Some(diff) => vec![ToolCallContent::Diff(diff)],
        None => vec![text_content(&truncate(output))],
    };
    let fields = ToolCallUpdateFields::new()
        .status(ToolCallStatus::Completed)
        .content(content)
        .raw_output(Value::String(output.to_string()));
    ToolCallUpdate::new(ToolCallId::new(tool_call_id.to_string()), fields)
}

/// Human-friendly card title that shows *what* the tool is doing,
/// not just *which* tool it is. Falls back to the static display name
/// for tools we don't introspect (Bifrost, unknown).
pub(super) fn tool_title(tool_name: &str, raw_input: &Value) -> String {
    let display = ToolRegistry::display_name(tool_name);
    let path = raw_input.get("path").and_then(Value::as_str);
    let pattern = raw_input.get("pattern").and_then(Value::as_str);
    let command = raw_input.get("command").and_then(Value::as_str);

    match tool_name {
        "readFile" => path
            .map(|p| format!("Read `{p}`"))
            .unwrap_or_else(|| display.to_string()),
        "writeFile" => path
            .map(|p| format!("Write `{p}`"))
            .unwrap_or_else(|| display.to_string()),
        "listDirectory" => path
            .map(|p| format!("List `{p}`"))
            .unwrap_or_else(|| display.to_string()),
        "searchFileContents" => pattern
            .map(|p| format!("Search `{p}`"))
            .unwrap_or_else(|| display.to_string()),
        "runShellCommand" => command
            .map(|c| format!("Run `{}`", first_line(c)))
            .unwrap_or_else(|| display.to_string()),
        "think" => "Think".to_string(),
        _ => {
            // Bifrost or anything we don't special-case: append a brief
            // input summary so the user sees more than just "Searching for
            // symbols". Falls back to the bare display name when no input
            // string can be picked.
            let summary = brief_input_summary(raw_input);
            if summary.is_empty() {
                display.to_string()
            } else {
                format!("{display}: {summary}")
            }
        }
    }
}

/// File locations affected by this call, used by clients for follow-along.
/// v1: only the obvious `path` arg on filesystem tools. Bifrost JSON
/// outputs may carry locations, but parsing them is out of scope here.
pub(super) fn tool_locations(tool_name: &str, raw_input: &Value) -> Vec<ToolCallLocation> {
    if matches!(tool_name, "readFile" | "writeFile" | "listDirectory")
        && let Some(path) = raw_input.get("path").and_then(Value::as_str)
    {
        return vec![ToolCallLocation::new(PathBuf::from(path))];
    }
    Vec::new()
}

fn truncate(s: &str) -> String {
    if s.len() <= MAX_INLINE_OUTPUT_BYTES {
        s.to_string()
    } else {
        // Truncate on a UTF-8 char boundary so we don't ship invalid
        // strings; floor-search backwards from the cap.
        let mut cut = MAX_INLINE_OUTPUT_BYTES;
        while !s.is_char_boundary(cut) {
            cut -= 1;
        }
        let mut out = s[..cut].to_string();
        out.push_str("\n... output truncated");
        out
    }
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or(s)
}

fn text_content(s: &str) -> ToolCallContent {
    ToolCallContent::Content(Content::new(ContentBlock::Text(TextContent::new(s))))
}

/// Pull the first non-empty string value out of a JSON object, capped
/// at ~80 chars. Used to give Bifrost calls a "where am I" hint in the
/// card title (e.g. `search_symbols` argv -> "main").
fn brief_input_summary(raw_input: &Value) -> String {
    let Some(obj) = raw_input.as_object() else {
        return String::new();
    };
    for (_, v) in obj {
        if let Some(s) = v.as_str() {
            let s = s.trim();
            if !s.is_empty() {
                let mut out = s.to_string();
                if out.len() > 80 {
                    out.truncate(80);
                    out.push_str("...");
                }
                return out;
            }
        }
        if let Some(arr) = v.as_array()
            && let Some(first) = arr.first().and_then(Value::as_str)
        {
            let s = first.trim();
            if !s.is_empty() {
                let mut out = s.to_string();
                if out.len() > 80 {
                    out.truncate(80);
                    out.push_str("...");
                }
                return out;
            }
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn read_file_title_shows_path() {
        let title = tool_title("readFile", &json!({"path": "src/lib.rs"}));
        assert_eq!(title, "Read `src/lib.rs`");
    }

    #[test]
    fn write_file_title_shows_path() {
        let title = tool_title("writeFile", &json!({"path": "a/b.txt", "content": "x"}));
        assert_eq!(title, "Write `a/b.txt`");
    }

    #[test]
    fn list_directory_title_shows_path() {
        let title = tool_title("listDirectory", &json!({"path": "src"}));
        assert_eq!(title, "List `src`");
    }

    #[test]
    fn search_title_shows_pattern() {
        let title = tool_title("searchFileContents", &json!({"pattern": "TODO"}));
        assert_eq!(title, "Search `TODO`");
    }

    #[test]
    fn run_shell_title_shows_first_line() {
        let title = tool_title(
            "runShellCommand",
            &json!({"command": "cargo test\n# extra junk"}),
        );
        assert_eq!(title, "Run `cargo test`");
    }

    #[test]
    fn think_title_is_constant() {
        let title = tool_title("think", &json!({"thought": "..."}));
        assert_eq!(title, "Think");
    }

    #[test]
    fn unknown_tool_falls_back_to_display_name_with_summary() {
        // search_symbols isn't special-cased here; it goes through
        // ToolRegistry::display_name + brief_input_summary.
        let title = tool_title("search_symbols", &json!({"patterns": ["main"]}));
        assert!(title.starts_with("Searching for symbols"));
        assert!(title.ends_with("main"));
    }

    #[test]
    fn unknown_tool_without_string_input_uses_display_name() {
        let title = tool_title("search_symbols", &json!({"limit": 10}));
        assert_eq!(title, "Searching for symbols");
    }

    #[test]
    fn missing_path_uses_display_name() {
        let title = tool_title("readFile", &json!({}));
        assert_eq!(title, "Reading file");
    }

    #[test]
    fn locations_include_path_for_filesystem_tools() {
        let locs = tool_locations("readFile", &json!({"path": "src/lib.rs"}));
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0].path, PathBuf::from("src/lib.rs"));

        let locs = tool_locations("writeFile", &json!({"path": "a.txt", "content": ""}));
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0].path, PathBuf::from("a.txt"));

        let locs = tool_locations("listDirectory", &json!({"path": "."}));
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0].path, PathBuf::from("."));
    }

    #[test]
    fn locations_empty_for_non_filesystem_tools() {
        assert!(tool_locations("runShellCommand", &json!({"command": "ls"})).is_empty());
        assert!(tool_locations("searchFileContents", &json!({"pattern": "x"})).is_empty());
        assert!(tool_locations("think", &json!({"thought": "..."})).is_empty());
        assert!(tool_locations("search_symbols", &json!({"patterns": ["x"]})).is_empty());
    }

    #[test]
    fn oversized_title_rejected_for_long_shell_command() {
        // Single-line shell command long enough to push the rendered
        // `Run \`<cmd>\`` title past 1024 chars. first_line() only strips
        // \n, not length, so the long single-line case is what trips the
        // cap in practice.
        let cmd = "echo ".to_string() + &"a".repeat(MAX_TOOL_TITLE_CHARS);
        let reason =
            rejection_for_oversized_title("runShellCommand", &json!({"command": cmd.clone()}));
        assert!(reason.is_some(), "title >1024 chars must be rejected");

        let cmd_short = "echo hello".to_string();
        let reason_short =
            rejection_for_oversized_title("runShellCommand", &json!({"command": cmd_short}));
        assert!(reason_short.is_none(), "short title must pass");
    }

    #[test]
    fn oversized_title_boundary_is_inclusive() {
        // Boundary check: title of exactly MAX chars passes, MAX+1 fails.
        // Title is `Read \`<path>\`` -> 7 chars of chrome ("Read `" + "`").
        let chrome = "Read ``".chars().count();
        let path_len = MAX_TOOL_TITLE_CHARS - chrome;
        let path = "a".repeat(path_len);
        let title = tool_title("readFile", &json!({"path": path.clone()}));
        assert_eq!(title.chars().count(), MAX_TOOL_TITLE_CHARS);
        assert!(rejection_for_oversized_title("readFile", &json!({"path": path})).is_none());

        let path_over = "a".repeat(path_len + 1);
        assert!(
            rejection_for_oversized_title("readFile", &json!({"path": path_over})).is_some(),
            "title MAX+1 chars must be rejected"
        );
    }

    #[test]
    fn rejection_text_quotes_the_actual_cap() {
        // Drift guard: the message handed to the LLM must mention the same
        // number the gate enforces. If MAX_TOOL_TITLE_CHARS changes, the
        // message has to follow.
        let cmd = "echo ".to_string() + &"a".repeat(MAX_TOOL_TITLE_CHARS);
        let reason =
            rejection_for_oversized_title("runShellCommand", &json!({"command": cmd})).unwrap();
        assert!(
            reason.contains(&MAX_TOOL_TITLE_CHARS.to_string()),
            "rejection message must quote the cap; got: {reason}"
        );
    }

    #[test]
    fn rejected_card_uses_static_display_name_not_user_input() {
        // Sanity: the rejection card's title is the static display name,
        // never the user-controlled args -- otherwise the rejection card
        // itself could trigger the same dialog-clip we're protecting
        // against.
        let huge = "x".repeat(MAX_TOOL_TITLE_CHARS * 2);
        let card = rejected_initial_tool_call(
            "tc1",
            "runShellCommand",
            ToolKind::Execute,
            &json!({"command": huge}),
        );
        assert_eq!(card.title, "Running shell command");
        // raw_input is still attached so the user can inspect what was
        // attempted; it lives in a scrollable region client-side.
        assert!(card.raw_input.is_some());
    }

    #[test]
    fn truncate_respects_char_boundary() {
        // Build a string just past the cap that ends in a multi-byte char,
        // and verify we don't slice through the middle of it.
        let mut s = "a".repeat(MAX_INLINE_OUTPUT_BYTES - 1);
        s.push('\u{1F600}'); // 4-byte UTF-8
        s.push_str("tail");
        let out = truncate(&s);
        assert!(out.ends_with("output truncated"));
        // No panic and the output is still valid UTF-8 (truncate already
        // returned a String).
        assert!(out.is_char_boundary(out.len()));
    }
}
