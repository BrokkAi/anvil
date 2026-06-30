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
    let mut i = start;
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
    if tail.in_string || tail.closers.is_empty() {
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
            '"' => tail.in_string = true,
            '{' => tail.closers.push('}'),
            '[' => tail.closers.push(']'),
            '}' | ']' if tail.closers.last() == Some(&ch) => {
                tail.closers.pop();
            }
            _ => {}
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
    fn reports_empty_arguments_as_invalid() {
        let err = normalize_tool_arguments("").unwrap_err();

        assert_eq!(err.kind(), ToolArgumentErrorKind::Invalid);
    }
}
