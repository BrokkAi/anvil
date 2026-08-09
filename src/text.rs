/// Returns a prefix of at most `max_bytes` without splitting a UTF-8 character.
pub(crate) fn truncate_utf8(value: &str, max_bytes: usize) -> &str {
    if value.len() <= max_bytes {
        return value;
    }

    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

/// Returns `value` unchanged when it fits, or a UTF-8-safe prefix followed by
/// `suffix` when it exceeds `max_bytes`. The byte cap applies to the source
/// text; callers may use a visible truncation marker without risking a direct
/// `String::truncate` at a non-character boundary.
pub(crate) fn truncate_utf8_with_suffix(value: &str, max_bytes: usize, suffix: &str) -> String {
    let prefix = truncate_utf8(value, max_bytes);
    if prefix.len() == value.len() {
        return value.to_string();
    }

    let mut output = String::with_capacity(prefix.len() + suffix.len());
    output.push_str(prefix);
    output.push_str(suffix);
    output
}

/// Returns `value` unchanged when it fits, or a UTF-8-safe head and tail joined
/// by `marker` when it exceeds `max_bytes`. The head keeps up to `max_bytes / 2`
/// bytes and the tail keeps the rest of the budget from the end of the string,
/// so both the beginning and the end of long output survive. `marker` receives
/// the number of source bytes elided; like `truncate_utf8_with_suffix`, the
/// marker text itself is not counted against `max_bytes`.
pub(crate) fn truncate_middle_utf8(
    value: &str,
    max_bytes: usize,
    marker: impl FnOnce(usize) -> String,
) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }

    let head = truncate_utf8(value, max_bytes / 2);
    // The head never exceeds `max_bytes / 2`, so the tail budget is at least as
    // large; because `value` is longer than `max_bytes`, the tail always starts
    // after the head and the two halves cannot overlap.
    let mut start = value.len() - (max_bytes - head.len());
    while !value.is_char_boundary(start) {
        start += 1;
    }
    let tail = &value[start..];

    let elided = value.len() - head.len() - tail.len();
    let marker = marker(elided);

    let mut output = String::with_capacity(head.len() + marker.len() + tail.len());
    output.push_str(head);
    output.push_str(&marker);
    output.push_str(tail);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_at_previous_utf8_boundary() {
        let value = "abc\u{25cf}tail";

        assert_eq!(truncate_utf8(value, 4), "abc");
    }

    #[test]
    fn leaves_short_strings_unchanged() {
        let value = "\u{25cf}";

        assert_eq!(truncate_utf8(value, 10), "\u{25cf}");
    }

    #[test]
    fn suffix_helper_handles_reported_unicode_boundaries() {
        assert_eq!(
            truncate_utf8_with_suffix("abc\u{2715}tail", 4, "..."),
            "abc..."
        );
        assert_eq!(
            truncate_utf8_with_suffix("abc\u{25cf}tail", 6, "..."),
            "abc\u{25cf}..."
        );
        assert_eq!(truncate_utf8_with_suffix("\u{25cf}", 0, "..."), "...");
    }

    #[test]
    fn middle_helper_keeps_head_and_tail_for_ascii() {
        let out = truncate_middle_utf8("abcdefghij", 6, |n| format!("<{n}>"));

        assert_eq!(out, "abc<4>hij");
        assert!(out.starts_with("abc"));
        assert!(out.ends_with("hij"));
    }

    #[test]
    fn middle_helper_handles_multibyte_at_both_cut_points() {
        // Ten 3-byte characters (30 bytes) with a budget whose halves both land
        // in the middle of a character.
        let value = "\u{25cf}".repeat(10);

        let out = truncate_middle_utf8(&value, 11, |n| format!("<{n}>"));

        // Head keeps one character (3 bytes), tail keeps two (6 bytes).
        assert_eq!(out, "\u{25cf}<21>\u{25cf}\u{25cf}");
        assert!(std::str::from_utf8(out.as_bytes()).is_ok());
        assert_eq!(value.len() - 3 - 6, 21);
    }

    #[test]
    fn middle_helper_leaves_exact_fit_unchanged() {
        let value = "abc\u{25cf}";

        let out = truncate_middle_utf8(value, value.len(), |_| unreachable!("no marker expected"));

        assert_eq!(out, value);
    }

    #[test]
    fn middle_helper_elides_everything_with_zero_budget() {
        let value = "abc\u{25cf}";

        let out = truncate_middle_utf8(value, 0, |n| format!("<{n}>"));

        assert_eq!(out, "<6>");
    }
}
