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
}
