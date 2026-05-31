use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::discovery::{OPENROUTER_API_KEY_ENV, OPENROUTER_BASE_URL};
use crate::llm_client::{ChatMessage, ToolDefinition};
use crate::openrouter_auth;
use crate::session::ToolExchange;

const DEFAULT_CLASSIFIER_MODEL: &str = "deepseek/deepseek-v4-flash";
const CLASSIFIER_MODEL_ENV: &str = "BROKK_BIFROST_GATE_CLASSIFIER_MODEL";
const CLASSIFIER_DISABLE_ENV: &str = "BROKK_BIFROST_GATE_CLASSIFIER_DISABLE";
const CLASSIFIER_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_TEXT: usize = 4_000;
const RECENT_EXCHANGES: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateClassifierDecision {
    AllowText,
    GateToSymbolTool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecommendedTool {
    SearchSymbols,
    ScanUsages,
    GetSummaries,
    GetSymbolSources,
    None,
}

impl RecommendedTool {
    pub fn as_tool_name(&self) -> &'static str {
        match self {
            Self::SearchSymbols => "search_symbols",
            Self::ScanUsages => "scan_usages",
            Self::GetSummaries => "get_summaries",
            Self::GetSymbolSources => "get_symbol_sources",
            Self::None => "none",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateConfidence {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateClassifierOutput {
    // Keep this first. The request schema also serializes this first so the
    // classifier commits the rationale before emitting its decision.
    pub reason: String,
    pub decision: GateClassifierDecision,
    pub recommended_tool: RecommendedTool,
    pub suggested_args: Value,
    pub confidence: GateConfidence,
}

#[derive(Debug, Clone)]
pub struct GateContext {
    pub tool_name: String,
    pub args: Value,
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ToolDefinition>,
    pub tool_exchanges: Vec<ToolExchange>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StaticTextTarget {
    TextLike,
    UnknownOrCodeLike,
}

pub async fn classify_text_tool_call(
    context: GateContext,
    cancel: &CancellationToken,
) -> Result<GateClassifierOutput> {
    if classifier_disabled() {
        return Err(anyhow!(
            "{CLASSIFIER_DISABLE_ENV} disables Bifrost gate classifier"
        ));
    }
    let api_key = openrouter_api_key().context("OpenRouter API key unavailable")?;
    let model = classifier_model();
    let body = build_request_body(&model, &context)?;
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(CLASSIFIER_TIMEOUT)
        .build()
        .context("building OpenRouter classifier client")?;

    let mut last_err: Option<anyhow::Error> = None;
    for attempt in 0..2 {
        if cancel.is_cancelled() {
            return Err(anyhow!("classification cancelled"));
        }
        match send_classifier_request(&client, &api_key, &body, cancel).await {
            Ok(output) => return Ok(output),
            Err(err) => {
                last_err = Some(err.context(format!("classifier attempt {}", attempt + 1)));
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow!("classifier failed without error")))
}

pub fn should_skip_for_static_text_target(tool_name: &str, args: &Value) -> bool {
    classify_static_text_target(tool_name, args) == StaticTextTarget::TextLike
}

pub fn classify_static_text_target(tool_name: &str, args: &Value) -> StaticTextTarget {
    match tool_name {
        "read_file" => args
            .get("file_path")
            .and_then(Value::as_str)
            .map(classify_path_or_glob)
            .unwrap_or(StaticTextTarget::UnknownOrCodeLike),
        "grep_search" => {
            let glob = args.get("glob").and_then(Value::as_str);
            let path = args.get("path").and_then(Value::as_str);
            if glob.map(classify_path_or_glob) == Some(StaticTextTarget::TextLike)
                || path.map(classify_path_or_glob) == Some(StaticTextTarget::TextLike)
            {
                StaticTextTarget::TextLike
            } else {
                StaticTextTarget::UnknownOrCodeLike
            }
        }
        "list_directory" => args
            .get("path")
            .and_then(Value::as_str)
            .map(classify_path_or_glob)
            .unwrap_or(StaticTextTarget::UnknownOrCodeLike),
        _ => StaticTextTarget::UnknownOrCodeLike,
    }
}

#[cfg(test)]
pub fn all_priority_symbol_tools_called(exchanges: &[ToolExchange]) -> bool {
    ["search_symbols", "scan_usages", "get_summaries"]
        .iter()
        .all(|name| exchanges.iter().any(|exchange| exchange.tool_name == *name))
}

pub fn gate_message(output: &GateClassifierOutput, tools: &[ToolDefinition]) -> String {
    let tool_name = output.recommended_tool.as_tool_name();
    let tool = tools.iter().find(|tool| tool.function.name == tool_name);
    let description = tool
        .map(|tool| tool.function.description.as_str())
        .unwrap_or("Use the recommended Bifrost symbol tool for this code-navigation question.");
    let schema = tool
        .map(|tool| serde_json::to_string(&tool.function.parameters).unwrap_or_default())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "{}".to_string());
    format!(
        "Bifrost gate: {reason}\n\nThe original text-navigation tool call was not executed. This is a navigation hint, not a final decision. If this hint is wrong and you need exact text, raw bytes, docs/config/log content, localized lines, or recovery after an edit/build/Bifrost miss, retry the original text tool call in your next tool batch; it will be allowed.\n\nRecommended next tool: `{tool_name}`.\n\nDescription: {description}\n\nSchema: {schema}\n\nIf you are looking for declarations or definitions by name, use `search_symbols`. If you are looking for callers, references, or related tests for a known symbol, use `scan_usages`. If you are orienting across a module, package, class, API, or file glob, use `get_summaries`. If you already know the relevant symbol names and need implementation source, use `get_symbol_sources`.",
        reason = output.reason,
    )
}

fn classifier_disabled() -> bool {
    std::env::var(CLASSIFIER_DISABLE_ENV)
        .ok()
        .map(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

fn classifier_model() -> String {
    std::env::var(CLASSIFIER_MODEL_ENV)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_CLASSIFIER_MODEL.to_string())
}

fn openrouter_api_key() -> Result<String> {
    if let Ok(raw) = std::env::var(OPENROUTER_API_KEY_ENV) {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }
    if let Some(auth) = openrouter_auth::read()? {
        let trimmed = auth.api_key.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }
    Err(anyhow!("no OpenRouter credential configured"))
}

async fn send_classifier_request(
    client: &reqwest::Client,
    api_key: &str,
    body: &str,
    cancel: &CancellationToken,
) -> Result<GateClassifierOutput> {
    let request = client
        .post(format!("{OPENROUTER_BASE_URL}/chat/completions"))
        .bearer_auth(api_key)
        .header("content-type", "application/json")
        .header("http-referer", "https://github.com/BrokkAi/brokk")
        .header("x-title", "anvil");
    let response = tokio::select! {
        _ = cancel.cancelled() => return Err(anyhow!("classification cancelled")),
        result = request.body(body.to_string()).send() => result.context("sending classifier request")?,
    };
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    if status != StatusCode::OK {
        return Err(anyhow!("classifier HTTP {status}: {text}"));
    }
    parse_openrouter_response(&text)
}

fn parse_openrouter_response(text: &str) -> Result<GateClassifierOutput> {
    let value: Value =
        serde_json::from_str(text).context("parsing classifier response envelope")?;
    let content = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("classifier response missing choices[0].message.content"))?;
    let mut output: GateClassifierOutput = serde_json::from_str(content)
        .with_context(|| format!("parsing classifier JSON: {content}"))?;
    normalize_classifier_consistency(&mut output);
    Ok(output)
}

fn build_request_body(model: &str, context: &GateContext) -> Result<String> {
    let system = "You are a routing classifier for an AI coding agent. Think privately before answering. You are advising a stronger coding model, not commanding it; gate only when a Bifrost symbol tool is clearly a better first step than the pending text call. Decide whether the pending read_file/grep_search/list_directory/run_shell_command-as-source-reader call should run as text navigation or should be replaced by a Bifrost symbol tool. Prefer Bifrost for code declarations, definitions, implementations, references, callers, related tests, and broad code orientation. Gate if the text call is searching or browsing source code for symbol-shaped patterns, function names, type names, method names, call sites, declarations, definitions, directory structure, or API usage. Gate repeated source reads, broad source globs/directories, and shell file-reader commands after exact symbol source has already been retrieved unless there was an edit/build failure, a prior Bifrost miss, or a need for raw literal/config/macro text. Recommend search_symbols for unknown declarations/definitions by name, scan_usages for callers/references/related tests, get_summaries for broad module/package/file-glob orientation, and get_symbol_sources when the relevant symbol names are already known and the agent needs implementation source. Allow text for docs, config, logs, build scripts, exact localized line checks after the relevant code symbol/file/line is already known, literal string searches not represented as symbols, non-code files, verification of recent edits, recovery from failed edits, and targeted fallback after Bifrost returned empty/not_found. When uncertain, allow text. The reason field must commit to one of these forms: 'ALLOW_TEXT because ...' or 'GATE_TO_SYMBOL_TOOL because ...'. If your reason says or implies that Bifrost/search_symbols/scan_usages/get_summaries/get_symbol_sources would be better, more direct, more appropriate, should be used, or should replace the pending call, decision must be gate_to_symbol_tool. If decision is allow_text, recommended_tool must be none. Output only JSON matching the schema.";
    let user = render_context(context)?;
    let model_json = serde_json::to_string(model)?;
    let system_json = serde_json::to_string(system)?;
    let user_json = serde_json::to_string(&user)?;
    Ok(format!(
        r#"{{
  "model": {model_json},
  "messages": [
    {{"role":"system","content":{system_json}}},
    {{"role":"user","content":{user_json}}}
  ],
  "response_format": {{
    "type": "json_schema",
    "json_schema": {{
      "name": "bifrost_gate_decision",
      "strict": true,
      "schema": {{
        "type": "object",
        "additionalProperties": false,
        "properties": {{
          "reason": {{"type":"string"}},
          "decision": {{"type":"string","enum":["allow_text","gate_to_symbol_tool"]}},
          "recommended_tool": {{"type":"string","enum":["search_symbols","scan_usages","get_summaries","get_symbol_sources","none"]}},
          "suggested_args": {{"type":"object"}},
          "confidence": {{"type":"string","enum":["low","medium","high"]}}
        }},
        "required": ["reason","decision","recommended_tool","suggested_args","confidence"]
      }}
    }}
  }},
  "temperature": 0,
  "max_tokens": 512
}}"#
    ))
}

fn normalize_classifier_consistency(output: &mut GateClassifierOutput) {
    if output.decision != GateClassifierDecision::AllowText {
        return;
    }
    let reason = output.reason.to_ascii_lowercase();
    let gate_cues = [
        "gate_to_symbol_tool",
        "better served by bifrost",
        "better answered by bifrost",
        "should use bifrost",
        "use bifrost",
        "more appropriate",
        "replace the pending",
        "replaced by",
        "directly return the declaration",
        "directly find",
    ];
    if !gate_cues.iter().any(|cue| reason.contains(cue)) {
        return;
    }
    output.decision = GateClassifierDecision::GateToSymbolTool;
    if output.recommended_tool == RecommendedTool::None {
        output.recommended_tool = infer_recommended_tool_from_reason(&reason);
    }
}

fn infer_recommended_tool_from_reason(reason: &str) -> RecommendedTool {
    if reason.contains("scan_usages")
        || reason.contains("caller")
        || reason.contains("call site")
        || reason.contains("callers")
        || reason.contains("references")
        || reason.contains("usages")
    {
        RecommendedTool::ScanUsages
    } else if reason.contains("get_symbol_sources")
        || reason.contains("implementation source")
        || reason.contains("source code")
        || reason.contains("exact source")
        || reason.contains("symbol source")
        || reason.contains("line-by-line source")
        || reason.contains("line ranges")
    {
        RecommendedTool::GetSymbolSources
    } else if reason.contains("get_summaries")
        || reason.contains("orientation")
        || reason.contains("orienting")
        || reason.contains("overview")
        || reason.contains("module")
        || reason.contains("api usage")
    {
        RecommendedTool::GetSummaries
    } else {
        RecommendedTool::SearchSymbols
    }
}

fn render_context(context: &GateContext) -> Result<String> {
    let user_prompt = context
        .messages
        .iter()
        .find(|message| message.role == "user")
        .and_then(|message| message.content.as_deref())
        .unwrap_or("");
    let bifrost_tools: Vec<_> = context
        .tools
        .iter()
        .filter(|tool| {
            matches!(
                tool.function.name.as_str(),
                "search_symbols" | "scan_usages" | "get_summaries"
            )
        })
        .map(|tool| {
            json!({
                "name": tool.function.name,
                "description": tool.function.description,
                "schema": tool.function.parameters,
            })
        })
        .collect();
    let recent: Vec<_> = context
        .tool_exchanges
        .iter()
        .rev()
        .take(RECENT_EXCHANGES)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|exchange| {
            json!({
                "tool": exchange.tool_name,
                "args": exchange.arguments,
                "result_excerpt": truncate(&exchange.result),
            })
        })
        .collect();
    let prior_counts = json!({
        "read_file": count_tool(&context.tool_exchanges, "read_file"),
        "grep_search": count_tool(&context.tool_exchanges, "grep_search"),
        "run_shell_command": count_tool(&context.tool_exchanges, "run_shell_command"),
        "search_symbols": count_tool(&context.tool_exchanges, "search_symbols"),
        "scan_usages": count_tool(&context.tool_exchanges, "scan_usages"),
        "get_summaries": count_tool(&context.tool_exchanges, "get_summaries"),
    });
    let payload = json!({
        "task_prompt_excerpt": truncate(user_prompt),
        "pending_text_call": {
            "tool": context.tool_name,
            "args": context.args,
        },
        "prior_tool_counts": prior_counts,
        "recent_tool_calls": recent,
        "bifrost_tools": bifrost_tools,
        "static_text_target": classify_static_text_target(&context.tool_name, &context.args),
    });
    serde_json::to_string_pretty(&payload).context("rendering classifier context")
}

fn count_tool(exchanges: &[ToolExchange], name: &str) -> usize {
    exchanges
        .iter()
        .filter(|exchange| exchange.tool_name == name)
        .count()
}

fn truncate(text: &str) -> String {
    if text.len() <= MAX_TEXT {
        return text.to_string();
    }
    let mut end = MAX_TEXT;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…[truncated]", &text[..end])
}

fn classify_path_or_glob(input: &str) -> StaticTextTarget {
    let lower = input.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return StaticTextTarget::UnknownOrCodeLike;
    }
    let file_name = lower.rsplit('/').next().unwrap_or(&lower);
    if matches!(
        file_name,
        "readme" | "readme.md" | "changelog" | "changelog.md" | "license" | "copying"
    ) || lower.contains("/docs/")
        || lower.starts_with("docs/")
        || lower.contains("/.github/")
        || lower.starts_with(".github/")
        || lower == ".harness/build.sh"
        || lower.ends_with("/.harness/build.sh")
    {
        return StaticTextTarget::TextLike;
    }
    if text_like_extension(&lower) {
        StaticTextTarget::TextLike
    } else {
        StaticTextTarget::UnknownOrCodeLike
    }
}

fn text_like_extension(input: &str) -> bool {
    const TEXT_EXTENSIONS: &[&str] = &[
        ".md",
        ".markdown",
        ".txt",
        ".rst",
        ".adoc",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".toml",
        ".lock",
        ".ini",
        ".cfg",
        ".conf",
        ".sbt",
        ".gradle",
        ".props",
        ".targets",
        ".log",
        ".csv",
        ".tsv",
        ".xml",
        ".html",
        ".css",
    ];
    TEXT_EXTENSIONS.iter().any(|ext| input.ends_with(ext))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exchange(tool_name: &str) -> ToolExchange {
        ToolExchange {
            call_id: String::new(),
            tool_name: tool_name.to_string(),
            arguments: "{}".to_string(),
            result: String::new(),
        }
    }

    #[test]
    fn static_text_target_skips_docs_and_configs() {
        assert!(should_skip_for_static_text_target(
            "read_file",
            &json!({"file_path": "README.md"})
        ));
        assert!(should_skip_for_static_text_target(
            "grep_search",
            &json!({"pattern": "foo", "glob": "**/*.md"})
        ));
        assert!(should_skip_for_static_text_target(
            "read_file",
            &json!({"file_path": ".harness/build.sh"})
        ));
        assert!(should_skip_for_static_text_target(
            "read_file",
            &json!({"file_path": "build.sbt"})
        ));
    }

    #[test]
    fn static_text_target_keeps_source_like_inputs() {
        assert!(!should_skip_for_static_text_target(
            "read_file",
            &json!({"file_path": "src/lib.rs"})
        ));
        assert!(!should_skip_for_static_text_target(
            "grep_search",
            &json!({"pattern": "Foo", "glob": "**/*.java"})
        ));
    }

    #[test]
    fn priority_symbol_tool_stop_condition_requires_all_three() {
        assert!(!all_priority_symbol_tools_called(&[
            exchange("search_symbols"),
            exchange("scan_usages"),
        ]));
        assert!(all_priority_symbol_tools_called(&[
            exchange("search_symbols"),
            exchange("scan_usages"),
            exchange("get_summaries"),
        ]));
    }

    #[test]
    fn request_schema_puts_reason_before_decision() {
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({"pattern": "Foo"}),
            messages: vec![ChatMessage::user("Find Foo")],
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };
        let body = build_request_body("deepseek/deepseek-v4-flash", &context).unwrap();
        let reason_pos = body.find(r#""reason""#).unwrap();
        let decision_pos = body.find(r#""decision""#).unwrap();
        assert!(reason_pos < decision_pos, "{body}");
    }

    #[test]
    fn classifier_prompt_frames_flash_as_advisory() {
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({"pattern": "Foo"}),
            messages: vec![ChatMessage::user("Find Foo")],
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };
        let body = build_request_body("deepseek/deepseek-v4-flash", &context).unwrap();

        assert!(body.contains("advising a stronger coding model"));
        assert!(body.contains("gate only when a Bifrost symbol tool is clearly a better first step"));
        assert!(body.contains("When uncertain, allow text"));
    }

    #[test]
    fn gate_message_explains_retry_override() {
        let output = GateClassifierOutput {
            reason: "GATE_TO_SYMBOL_TOOL because Foo looks like a declaration.".to_string(),
            decision: GateClassifierDecision::GateToSymbolTool,
            recommended_tool: RecommendedTool::SearchSymbols,
            suggested_args: json!({"patterns": ["Foo"]}),
            confidence: GateConfidence::High,
        };
        let message = gate_message(&output, &[]);

        assert!(message.contains("navigation hint, not a final decision"));
        assert!(message.contains("retry the original text tool call in your next tool batch"));
        assert!(message.contains("it will be allowed"));
    }

    #[test]
    fn parses_classifier_content() {
        let envelope = json!({
            "choices": [{
                "message": {
                    "content": "{\"reason\":\"symbol lookup\",\"decision\":\"gate_to_symbol_tool\",\"recommended_tool\":\"search_symbols\",\"suggested_args\":{\"patterns\":[\"Foo\"]},\"confidence\":\"high\"}"
                }
            }]
        });
        let parsed = parse_openrouter_response(&envelope.to_string()).unwrap();
        assert_eq!(parsed.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(parsed.recommended_tool, RecommendedTool::SearchSymbols);
    }

    #[test]
    fn parses_get_symbol_sources_recommendation() {
        let envelope = json!({
            "choices": [{
                "message": {
                    "content": "{\"reason\":\"GATE_TO_SYMBOL_TOOL because known symbols need implementation source.\",\"decision\":\"gate_to_symbol_tool\",\"recommended_tool\":\"get_symbol_sources\",\"suggested_args\":{\"symbols\":[\"Foo.bar\"]},\"confidence\":\"high\"}"
                }
            }]
        });
        let parsed = parse_openrouter_response(&envelope.to_string()).unwrap();
        assert_eq!(parsed.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(parsed.recommended_tool, RecommendedTool::GetSymbolSources);
    }

    #[test]
    fn contradictory_allow_text_reason_is_normalized_to_gate() {
        let envelope = json!({
            "choices": [{
                "message": {
                    "content": "{\"reason\":\"The grep is looking for callers and would be better served by Bifrost scan_usages.\",\"decision\":\"allow_text\",\"recommended_tool\":\"none\",\"suggested_args\":{},\"confidence\":\"high\"}"
                }
            }]
        });
        let parsed = parse_openrouter_response(&envelope.to_string()).unwrap();
        assert_eq!(parsed.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(parsed.recommended_tool, RecommendedTool::ScanUsages);
    }

    #[test]
    fn contradictory_source_reason_infers_get_symbol_sources() {
        let envelope = json!({
            "choices": [{
                "message": {
                    "content": "{\"reason\":\"The read_file asks for line-by-line source for known symbols and would be better served by Bifrost get_symbol_sources.\",\"decision\":\"allow_text\",\"recommended_tool\":\"none\",\"suggested_args\":{},\"confidence\":\"high\"}"
                }
            }]
        });
        let parsed = parse_openrouter_response(&envelope.to_string()).unwrap();
        assert_eq!(parsed.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(parsed.recommended_tool, RecommendedTool::GetSymbolSources);
    }

    #[test]
    fn localized_allow_text_reason_stays_allowed() {
        let envelope = json!({
            "choices": [{
                "message": {
                    "content": "{\"reason\":\"ALLOW_TEXT because this reads exact localized lines in a known source file after symbol discovery.\",\"decision\":\"allow_text\",\"recommended_tool\":\"none\",\"suggested_args\":{},\"confidence\":\"high\"}"
                }
            }]
        });
        let parsed = parse_openrouter_response(&envelope.to_string()).unwrap();
        assert_eq!(parsed.decision, GateClassifierDecision::AllowText);
        assert_eq!(parsed.recommended_tool, RecommendedTool::None);
    }
}
