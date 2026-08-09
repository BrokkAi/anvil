/// Returns true when `prompt_text` invokes the slash command `name`,
/// matching `/name` exactly or `/name <args>`. Whitespace and case are
/// normalized so clients that uppercase auto-complete entries still hit.
pub(crate) fn is_slash_command(prompt_text: &str, name: &str) -> bool {
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
pub(crate) fn parse_slash_command(prompt_text: &str) -> Option<(String, String)> {
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

/// Trimmed args for a slash command. Returns the empty string when the
/// prompt is not a slash command at all, or when the command has no
/// trailing args. Shared by setup and `/pr-create` -- both want "args
/// after the command name, trimmed of surrounding whitespace".
pub(crate) fn slash_command_args(prompt_text: &str) -> String {
    parse_slash_command(prompt_text)
        .map(|(_, a)| a)
        .unwrap_or_default()
        .trim()
        .to_string()
}
