use serde_json::Value;
use serde_json::error::Category;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone)]
pub(crate) struct NormalizedToolArguments {
    pub(crate) value: Value,
    pub(crate) arguments: String,
    pub(crate) repaired: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolArgumentErrorKind {
    Invalid,
    Incomplete,
}

#[derive(Debug, Clone)]
pub(crate) struct ToolArgumentParseError {
    kind: ToolArgumentErrorKind,
    message: String,
}

impl ToolArgumentParseError {
    pub(crate) fn kind(&self) -> ToolArgumentErrorKind {
        self.kind
    }
}

impl fmt::Display for ToolArgumentParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for ToolArgumentParseError {}

pub(crate) fn normalize_tool_arguments(
    raw: &str,
) -> Result<NormalizedToolArguments, ToolArgumentParseError> {
    match serde_json::from_str::<Value>(raw) {
        Ok(value) => Ok(NormalizedToolArguments {
            value,
            arguments: raw.to_string(),
            repaired: false,
        }),
        Err(first_error) => {
            let first_message = first_error.to_string();
            let raw_trimmed = raw.trim();
            let mut saw_eof = first_error.classify() == Category::Eof && !raw_trimmed.is_empty();
            let mut candidates = Vec::new();

            push_candidate(&mut candidates, raw_trimmed);
            if let Some(fenced) = strip_json_code_fence(raw_trimmed) {
                push_candidate(&mut candidates, &fenced);
            }

            let bases = candidates.clone();
            for base in bases {
                let mut repaired = quote_unquoted_object_keys(&base);
                repaired = convert_single_quoted_strings(&repaired);
                repaired = remove_trailing_commas(&repaired);
                push_candidate(&mut candidates, &repaired);
                if let Some(completed) = complete_structural_tail(&repaired) {
                    push_candidate(&mut candidates, &completed);
                }
            }

            for candidate in candidates {
                match serde_json::from_str::<Value>(&candidate) {
                    Ok(value) => {
                        return Ok(NormalizedToolArguments {
                            arguments: value.to_string(),
                            value,
                            repaired: true,
                        });
                    }
                    Err(err) => {
                        saw_eof |= err.classify() == Category::Eof;
                    }
                }
            }

            let kind = if saw_eof {
                ToolArgumentErrorKind::Incomplete
            } else {
                ToolArgumentErrorKind::Invalid
            };
            Err(ToolArgumentParseError {
                kind,
                message: format!("parse tool arguments as JSON: {first_message}"),
            })
        }
    }
}

fn push_candidate(candidates: &mut Vec<String>, candidate: &str) {
    let trimmed = candidate.trim();
    if trimmed.is_empty() {
        return;
    }
    if !candidates.iter().any(|existing| existing == trimmed) {
        candidates.push(trimmed.to_string());
    }
}

fn strip_json_code_fence(input: &str) -> Option<String> {
    let rest = input.strip_prefix("```")?;
    let (_, body) = rest.split_once('\n')?;
    let body = body.trim();
    let body = body.strip_suffix("```")?.trim();
    Some(body.to_string())
}

fn convert_single_quoted_strings(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars();
    let mut in_double = false;
    let mut double_escaped = false;

    while let Some(ch) = chars.next() {
        if in_double {
            out.push(ch);
            if double_escaped {
                double_escaped = false;
            } else if ch == '\\' {
                double_escaped = true;
            } else if ch == '"' {
                in_double = false;
            }
            continue;
        }

        match ch {
            '"' => {
                in_double = true;
                out.push(ch);
            }
            '\'' => {
                out.push('"');
                let mut escaped = false;
                for next in chars.by_ref() {
                    if escaped {
                        match next {
                            '\'' => out.push('\''),
                            '"' => out.push_str("\\\""),
                            '\\' => out.push_str("\\\\"),
                            'n' => out.push_str("\\n"),
                            'r' => out.push_str("\\r"),
                            't' => out.push_str("\\t"),
                            other => {
                                out.push('\\');
                                out.push(other);
                            }
                        }
                        escaped = false;
                    } else if next == '\\' {
                        escaped = true;
                    } else if next == '\'' {
                        out.push('"');
                        break;
                    } else if next == '"' {
                        out.push_str("\\\"");
                    } else if next == '\n' {
                        out.push_str("\\n");
                    } else if next == '\r' {
                        out.push_str("\\r");
                    } else if next == '\t' {
                        out.push_str("\\t");
                    } else {
                        out.push(next);
                    }
                }
            }
            other => out.push(other),
        }
    }

    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Container {
    ObjectExpectKey,
    ObjectExpectValue,
    Array,
}

fn quote_unquoted_object_keys(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());
    let mut stack: Vec<Container> = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        if matches!(ch, '"' | '\'') {
            i = copy_string_literal(&chars, i, &mut out);
            continue;
        }

        if stack.last() == Some(&Container::ObjectExpectKey) && is_key_start(ch) {
            let key_start = i;
            i += 1;
            while i < chars.len() && is_key_char(chars[i]) {
                i += 1;
            }
            let key_end = i;
            let mut colon = i;
            while colon < chars.len() && chars[colon].is_whitespace() {
                colon += 1;
            }
            if colon < chars.len() && chars[colon] == ':' {
                out.push('"');
                for key_ch in &chars[key_start..key_end] {
                    out.push(*key_ch);
                }
                out.push('"');
                for ws in &chars[key_end..colon] {
                    out.push(*ws);
                }
                out.push(':');
                if let Some(top) = stack.last_mut() {
                    *top = Container::ObjectExpectValue;
                }
                i = colon + 1;
                continue;
            }

            for copied in &chars[key_start..key_end] {
                out.push(*copied);
            }
            continue;
        }

        match ch {
            '{' => stack.push(Container::ObjectExpectKey),
            '[' => stack.push(Container::Array),
            '}' => {
                if matches!(
                    stack.last(),
                    Some(Container::ObjectExpectKey | Container::ObjectExpectValue)
                ) {
                    stack.pop();
                }
            }
            ']' if stack.last() == Some(&Container::Array) => {
                stack.pop();
            }
            ':' => {
                if let Some(Container::ObjectExpectKey) = stack.last_mut() {
                    *stack.last_mut().expect("checked above") = Container::ObjectExpectValue;
                }
            }
            ',' => {
                if let Some(Container::ObjectExpectValue) = stack.last_mut() {
                    *stack.last_mut().expect("checked above") = Container::ObjectExpectKey;
                }
            }
            _ => {}
        }
        out.push(ch);
        i += 1;
    }

    out
}

fn copy_string_literal(chars: &[char], start: usize, out: &mut String) -> usize {
    let quote = chars[start];
    // Emit the opening quote, then scan the body. Starting the loop at the
    // opening quote would let it match itself as the closing quote and bail
    // after a single character, leaving the string body exposed to the
    // structural scanners (which would then mangle commas/keys inside values).
    out.push(quote);
    let mut i = start + 1;
    let mut escaped = false;
    while i < chars.len() {
        let ch = chars[i];
        out.push(ch);
        i += 1;
        if escaped {
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == quote {
            break;
        }
    }
    i
}

fn is_key_start(ch: char) -> bool {
    ch == '_' || ch == '$' || ch.is_ascii_alphabetic()
}

fn is_key_char(ch: char) -> bool {
    ch == '_' || ch == '$' || ch == '-' || ch.is_ascii_alphanumeric()
}

fn remove_trailing_commas(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        if matches!(ch, '"' | '\'') {
            i = copy_string_literal(&chars, i, &mut out);
            continue;
        }
        if ch == ',' {
            let mut next = i + 1;
            while next < chars.len() && chars[next].is_whitespace() {
                next += 1;
            }
            if next < chars.len() && matches!(chars[next], '}' | ']') {
                i += 1;
                continue;
            }
        }
        out.push(ch);
        i += 1;
    }

    out
}

fn complete_structural_tail(input: &str) -> Option<String> {
    let tail = analyze_json_tail(input);
    // Refuse to close a buffer that ends in a dangling bare number. Any prefix
    // of a number is itself a valid number, so appending closers would fabricate
    // a wrong value (e.g. a truncated `30000` becomes `30`) and dispatch it
    // instead of letting the truncation be classified Incomplete and retried.
    // Terminated values (strings, `true`/`false`/`null`, closed containers) are
    // safe to complete and a truncated keyword fails to parse on its own, so
    // only bare numbers need this guard.
    if tail.in_string || tail.ends_in_number || tail.closers.is_empty() {
        return None;
    }

    let mut completed = input.trim_end().to_string();
    for closer in tail.closers.iter().rev() {
        completed.push(*closer);
    }
    Some(completed)
}

#[derive(Default)]
struct JsonTail {
    closers: Vec<char>,
    in_string: bool,
    /// True when the buffer ends inside an unterminated bare number (the last
    /// significant character is a digit or `.` with no delimiter after it).
    ends_in_number: bool,
}

fn analyze_json_tail(input: &str) -> JsonTail {
    let mut tail = JsonTail::default();
    let mut escaped = false;

    for ch in input.chars() {
        if tail.in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                tail.in_string = false;
            }
            continue;
        }

        match ch {
            '"' => {
                tail.in_string = true;
                tail.ends_in_number = false;
            }
            '{' => {
                tail.closers.push('}');
                tail.ends_in_number = false;
            }
            '[' => {
                tail.closers.push(']');
                tail.ends_in_number = false;
            }
            '}' | ']' if tail.closers.last() == Some(&ch) => {
                tail.closers.pop();
                tail.ends_in_number = false;
            }
            // A digit or decimal point keeps (or starts) a trailing number run;
            // any delimiter, whitespace, sign or exponent marker terminates it.
            '0'..='9' | '.' => tail.ends_in_number = true,
            _ => tail.ends_in_number = false,
        }
    }

    tail
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_json_is_returned_without_repair() {
        let parsed = normalize_tool_arguments(r#"{"file_path":"README.md"}"#).unwrap();

        assert!(!parsed.repaired);
        assert_eq!(parsed.arguments, r#"{"file_path":"README.md"}"#);
        assert_eq!(parsed.value["file_path"], "README.md");
    }

    #[test]
    fn repairs_common_model_argument_json() {
        let parsed = normalize_tool_arguments("{file_path:'README.md',}").unwrap();

        assert!(parsed.repaired);
        assert_eq!(parsed.arguments, r#"{"file_path":"README.md"}"#);
        assert_eq!(parsed.value["file_path"], "README.md");
    }

    #[test]
    fn repairs_missing_closing_object() {
        let parsed = normalize_tool_arguments(r#"{"file_path":"README.md""#).unwrap();

        assert!(parsed.repaired);
        assert_eq!(parsed.arguments, r#"{"file_path":"README.md"}"#);
    }

    #[test]
    fn strips_json_code_fences() {
        let parsed =
            normalize_tool_arguments("```json\n{\"file_path\":\"README.md\"}\n```").unwrap();

        assert!(parsed.repaired);
        assert_eq!(parsed.value["file_path"], "README.md");
    }

    #[test]
    fn reports_unterminated_string_as_incomplete() {
        let err = normalize_tool_arguments(r#"{"file_path":"README"#).unwrap_err();

        assert_eq!(err.kind(), ToolArgumentErrorKind::Incomplete);
    }

    #[test]
    fn preserves_comma_inside_string_value_during_repair() {
        // The unquoted key forces the repair path; the comma lives INSIDE the
        // value and must not be deleted as if it were a trailing comma.
        let parsed = normalize_tool_arguments(r#"{cmd:"a,}"}"#).unwrap();

        assert!(parsed.repaired);
        assert_eq!(parsed.value["cmd"], "a,}");
    }

    #[test]
    fn does_not_inject_quotes_into_string_values_during_repair() {
        // Structural-looking characters (commas, colons) inside a quoted value
        // must survive key-quoting untouched.
        let parsed = normalize_tool_arguments(r#"{msg:"a, b: c"}"#).unwrap();

        assert!(parsed.repaired);
        assert_eq!(parsed.value["msg"], "a, b: c");
    }

    #[test]
    fn repairs_object_with_numeric_value() {
        // A number that is followed by structure is fully known and must still
        // repair (the truncation guard only fires on a dangling trailing number).
        let parsed = normalize_tool_arguments("{count:5,enabled:true}").unwrap();

        assert!(parsed.repaired);
        assert_eq!(parsed.value["count"], 5);
        assert_eq!(parsed.value["enabled"], true);
    }

    #[test]
    fn reports_truncated_number_as_incomplete() {
        // A stream cut mid-number must be retried, not silently completed into a
        // smaller valid value (30 here could be a truncated 30000).
        let err = normalize_tool_arguments(r#"{"timeout_ms":30"#).unwrap_err();

        assert_eq!(err.kind(), ToolArgumentErrorKind::Incomplete);
    }

    #[test]
    fn reports_truncated_number_in_array_as_incomplete() {
        let err = normalize_tool_arguments(r#"{"lines":[1,2,3"#).unwrap_err();

        assert_eq!(err.kind(), ToolArgumentErrorKind::Incomplete);
    }

    #[test]
    fn reports_empty_arguments_as_invalid() {
        let err = normalize_tool_arguments("").unwrap_err();

        assert_eq!(err.kind(), ToolArgumentErrorKind::Invalid);
    }
}
