use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::discovery::{OPENROUTER_API_KEY_ENV, OPENROUTER_BASE_URL};
use crate::llm_client::{ChatMessage, OpenAiClient, ToolDefinition};
use crate::openrouter_auth;
use crate::session::ToolExchange;

const DEFAULT_CLASSIFIER_MODEL: &str = "deepseek/deepseek-v4-flash";
pub const ENCOURAGE_BIFROST_ENV: &str = "BRK_ENCOURAGE_BIFROST";
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
    ReadFile,
    GrepSearch,
    ListDirectory,
    Edit,
    WriteFile,
    SearchSymbols,
    ScanUsages,
    GetSummaries,
    GetSymbolSources,
    None,
}

impl RecommendedTool {
    pub fn as_tool_name(&self) -> &'static str {
        match self {
            Self::ReadFile => "read_file",
            Self::GrepSearch => "grep_search",
            Self::ListDirectory => "list_directory",
            Self::Edit => "edit",
            Self::WriteFile => "write_file",
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
#[serde(rename_all = "snake_case")]
pub enum ShellClassifierDecision {
    AllowShell,
    UseBuiltinTool,
    UseBifrostTool,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShellClassifierOutput {
    // Keep this first so the model commits reasoning before the routing decision.
    pub reason: String,
    pub decision: ShellClassifierDecision,
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

pub fn encourage_bifrost_enabled() -> bool {
    env_var_is_truthy(ENCOURAGE_BIFROST_ENV)
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
    let client = OpenAiClient::apply_runtime_tls_workarounds(
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(CLASSIFIER_TIMEOUT),
        OPENROUTER_BASE_URL,
    )
    .build()
    .context("building OpenRouter classifier client")?;

    let mut last_err: Option<anyhow::Error> = None;
    for attempt in 0..2 {
        if cancel.is_cancelled() {
            return Err(anyhow!("classification cancelled"));
        }
        match send_classifier_request(&client, &api_key, &body, parse_openrouter_response, cancel)
            .await
        {
            Ok(output) => return Ok(output),
            Err(err) => {
                last_err = Some(err.context(format!("classifier attempt {}", attempt + 1)));
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow!("classifier failed without error")))
}

pub async fn classify_shell_tool_call(
    context: GateContext,
    cancel: &CancellationToken,
) -> Result<ShellClassifierOutput> {
    if classifier_disabled() {
        return Err(anyhow!(
            "{CLASSIFIER_DISABLE_ENV} disables Bifrost gate classifier"
        ));
    }
    let api_key = openrouter_api_key().context("OpenRouter API key unavailable")?;
    let model = classifier_model();
    let body = build_shell_request_body(&model, &context)?;
    let client = OpenAiClient::apply_runtime_tls_workarounds(
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(CLASSIFIER_TIMEOUT),
        OPENROUTER_BASE_URL,
    )
    .build()
    .context("building OpenRouter shell classifier client")?;

    let mut last_err: Option<anyhow::Error> = None;
    for attempt in 0..2 {
        if cancel.is_cancelled() {
            return Err(anyhow!("classification cancelled"));
        }
        match send_classifier_request(
            &client,
            &api_key,
            &body,
            parse_shell_openrouter_response,
            cancel,
        )
        .await
        {
            Ok(output) => return Ok(output),
            Err(err) => {
                last_err = Some(err.context(format!("shell classifier attempt {}", attempt + 1)));
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow!("shell classifier failed without error")))
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

pub fn shell_gate_message(output: &ShellClassifierOutput, tools: &[ToolDefinition]) -> String {
    let tool_name = output.recommended_tool.as_tool_name();
    let tool = tools.iter().find(|tool| tool.function.name == tool_name);
    let description = tool
        .map(|tool| tool.function.description.as_str())
        .unwrap_or(
            "Use the recommended built-in or Bifrost tool instead of shell for this operation.",
        );
    let schema = tool
        .map(|tool| serde_json::to_string(&tool.function.parameters).unwrap_or_default())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "{}".to_string());
    format!(
        "Shell routing hint: {reason}\n\nThe original shell command was not executed. This is a routing hint, not a final decision. If this hint is wrong and you intentionally need shell semantics, a pipeline, raw bytes, build/test/git/package-manager behavior, or fallback after a Bifrost miss, retry the original shell command in your next tool batch; it will be allowed.\n\nRecommended next tool: `{tool_name}`.\n\nDescription: {description}\n\nSchema: {schema}\n\nUse built-in tools for ordinary file reads, file searches, directory listings, edits, and writes. Use Bifrost tools for source declarations, definitions, callers, references, related tests, and broad code orientation. Use shell for commands whose shell behavior is the point.",
        reason = output.reason,
    )
}

fn classifier_disabled() -> bool {
    env_var_is_truthy(CLASSIFIER_DISABLE_ENV)
}

fn classifier_model() -> String {
    std::env::var(CLASSIFIER_MODEL_ENV)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_CLASSIFIER_MODEL.to_string())
}

fn env_var_is_truthy(var: &str) -> bool {
    std::env::var(var)
        .ok()
        .map(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
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

async fn send_classifier_request<T>(
    client: &reqwest::Client,
    api_key: &str,
    body: &str,
    parse: fn(&str) -> Result<T>,
    cancel: &CancellationToken,
) -> Result<T> {
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
    parse(&text)
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

fn parse_shell_openrouter_response(text: &str) -> Result<ShellClassifierOutput> {
    let value: Value =
        serde_json::from_str(text).context("parsing shell classifier response envelope")?;
    let content = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("shell classifier response missing choices[0].message.content"))?;
    let mut output: ShellClassifierOutput = serde_json::from_str(content)
        .with_context(|| format!("parsing shell classifier JSON: {content}"))?;
    normalize_shell_classifier_consistency(&mut output);
    Ok(output)
}

fn build_request_body(model: &str, context: &GateContext) -> Result<String> {
    let system = "You are a routing classifier for an AI coding agent. Think privately before answering. You are advising a stronger coding model, not commanding it; gate only when a Bifrost symbol tool is clearly a better first step than the pending text call. Decide whether the pending read_file/grep_search/list_directory call should run as text navigation or should be replaced by a Bifrost symbol tool. Prefer Bifrost for code declarations, definitions, implementations, references, callers, related tests, and broad code orientation. Gate if the text call is searching or browsing source code for symbol-shaped patterns, function names, type names, method names, call sites, declarations, definitions, directory structure, or API usage. Gate repeated source reads and broad source globs/directories after exact symbol source has already been retrieved unless there was an edit/build failure, a prior Bifrost miss, or a need for raw literal/config/macro text. Recommend search_symbols for unknown declarations/definitions by name, scan_usages for callers/references/related tests, get_summaries for broad module/package/file-glob orientation, and get_symbol_sources when the relevant symbol names are already known and the agent needs implementation source. Allow text for docs, config, logs, build scripts, exact localized line checks after the relevant code symbol/file/line is already known, literal string searches not represented as symbols, non-code files, verification of recent edits, recovery from failed edits, and targeted fallback after Bifrost returned empty/not_found. Use recent_bifrost_misses to recognize targeted fallbacks for the same unresolved token/path. When uncertain, allow text. The reason field must commit to one of these forms: 'ALLOW_TEXT because ...' or 'GATE_TO_SYMBOL_TOOL because ...'. If your reason says or implies that Bifrost/search_symbols/scan_usages/get_summaries/get_symbol_sources would be better, more direct, more appropriate, should be used, or should replace the pending call, decision must be gate_to_symbol_tool. If decision is allow_text, recommended_tool must be none. Output only JSON matching the schema.";
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

fn build_shell_request_body(model: &str, context: &GateContext) -> Result<String> {
    let system = "You are a shell routing classifier for an AI coding agent. Think privately before answering. You are advising a stronger coding model, not commanding it; gate only when a built-in tool or Bifrost symbol tool is clearly better than the pending run_shell_command. Decide whether the command should run as shell, should be replaced by a built-in tool, or should be replaced by a Bifrost symbol tool. Allow shell for build, test, git, package manager, project CLI, environment inspection, raw-byte/format probes such as cat -A, xxd, od, hexdump, pipeline-specific debugging, command substitution behavior, and cases where shell semantics are the point. Prefer built-ins for ordinary file reads, directory listings, file-content searches, edits, and writes, including docs/config/harness files. Prefer Bifrost for source declarations, definitions, implementations, references, callers, related tests, and broad source orientation. Use recent_bifrost_misses to allow targeted shell fallback for the same unresolved token/path after a real Bifrost miss. Do not overrule the pending command just because a different route might also work; gate only when the alternative is clearly more appropriate for high-quality coding-agent behavior. When uncertain, allow shell. The reason field must commit to one of these forms: 'ALLOW_SHELL because ...', 'USE_BUILTIN_TOOL because ...', or 'USE_BIFROST_TOOL because ...'. If decision is allow_shell, recommended_tool must be none. If decision is use_builtin_tool, recommended_tool must be one of read_file, grep_search, list_directory, edit, write_file. If decision is use_bifrost_tool, recommended_tool must be one of search_symbols, scan_usages, get_summaries, get_symbol_sources. Output only JSON matching the schema.";
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
      "name": "shell_routing_decision",
      "strict": true,
      "schema": {{
        "type": "object",
        "additionalProperties": false,
        "properties": {{
          "reason": {{"type":"string"}},
          "decision": {{"type":"string","enum":["allow_shell","use_builtin_tool","use_bifrost_tool"]}},
          "recommended_tool": {{"type":"string","enum":["read_file","grep_search","list_directory","edit","write_file","search_symbols","scan_usages","get_summaries","get_symbol_sources","none"]}},
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

fn normalize_shell_classifier_consistency(output: &mut ShellClassifierOutput) {
    match output.decision {
        ShellClassifierDecision::AllowShell => {
            output.recommended_tool = RecommendedTool::None;
            output.suggested_args = json!({});
        }
        ShellClassifierDecision::UseBuiltinTool
            if !is_builtin_recommendation(&output.recommended_tool) =>
        {
            output.recommended_tool =
                infer_shell_recommended_tool_from_reason(&output.reason, true);
        }
        ShellClassifierDecision::UseBifrostTool
            if !is_bifrost_recommendation(&output.recommended_tool) =>
        {
            output.recommended_tool =
                infer_shell_recommended_tool_from_reason(&output.reason, false);
        }
        _ => {}
    }
}

fn is_builtin_recommendation(tool: &RecommendedTool) -> bool {
    matches!(
        tool,
        RecommendedTool::ReadFile
            | RecommendedTool::GrepSearch
            | RecommendedTool::ListDirectory
            | RecommendedTool::Edit
            | RecommendedTool::WriteFile
    )
}

fn is_bifrost_recommendation(tool: &RecommendedTool) -> bool {
    matches!(
        tool,
        RecommendedTool::SearchSymbols
            | RecommendedTool::ScanUsages
            | RecommendedTool::GetSummaries
            | RecommendedTool::GetSymbolSources
    )
}

fn infer_shell_recommended_tool_from_reason(reason: &str, builtin: bool) -> RecommendedTool {
    let reason = reason.to_ascii_lowercase();
    if builtin {
        if reason.contains("grep") || reason.contains("search") {
            RecommendedTool::GrepSearch
        } else if reason.contains("list") || reason.contains("directory") || reason.contains("ls") {
            RecommendedTool::ListDirectory
        } else if reason.contains("write") || reason.contains("create") {
            RecommendedTool::WriteFile
        } else if reason.contains("edit") || reason.contains("replace") || reason.contains("patch")
        {
            RecommendedTool::Edit
        } else {
            RecommendedTool::ReadFile
        }
    } else {
        infer_recommended_tool_from_reason(&reason)
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
        .map(ChatMessage::content_text)
        .unwrap_or_default();
    let bifrost_tools: Vec<_> = context
        .tools
        .iter()
        .filter(|tool| {
            matches!(
                tool.function.name.as_str(),
                "search_symbols" | "scan_usages" | "get_summaries" | "get_symbol_sources"
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
    let recent_bifrost_misses = recent_bifrost_misses(&context.tool_exchanges);
    let prior_counts = json!({
        "read_file": count_tool(&context.tool_exchanges, "read_file"),
        "grep_search": count_tool(&context.tool_exchanges, "grep_search"),
        "run_shell_command": count_tool(&context.tool_exchanges, "run_shell_command"),
        "search_symbols": count_tool(&context.tool_exchanges, "search_symbols"),
        "scan_usages": count_tool(&context.tool_exchanges, "scan_usages"),
        "get_summaries": count_tool(&context.tool_exchanges, "get_summaries"),
    });
    let payload = json!({
        "task_prompt_excerpt": truncate(&user_prompt),
        "pending_text_call": {
            "tool": context.tool_name,
            "args": context.args,
        },
        "prior_tool_counts": prior_counts,
        "recent_tool_calls": recent,
        "recent_bifrost_misses": recent_bifrost_misses,
        "bifrost_tools": bifrost_tools,
        "static_text_target": classify_static_text_target(&context.tool_name, &context.args),
    });
    serde_json::to_string_pretty(&payload).context("rendering classifier context")
}

fn recent_bifrost_misses(exchanges: &[ToolExchange]) -> Vec<Value> {
    exchanges
        .iter()
        .rev()
        .filter(|exchange| {
            is_bifrost_tool(&exchange.tool_name) && looks_like_bifrost_miss(&exchange.result)
        })
        .take(5)
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
        .collect()
}

fn is_bifrost_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "search_symbols" | "scan_usages" | "get_summaries" | "get_symbol_sources"
    )
}

fn looks_like_bifrost_miss(result: &str) -> bool {
    let trimmed = result.trim();
    if trimmed.is_empty() || trimmed == "[]" || trimmed == "{}" {
        return true;
    }
    let lower = trimmed.to_ascii_lowercase();
    [
        "not_found",
        "not found",
        "no match",
        "no matches",
        "no symbol",
        "no symbols",
        "no usages",
        "no references",
        "empty result",
        "returned empty",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
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
        assert!(
            body.contains("gate only when a Bifrost symbol tool is clearly a better first step")
        );
        assert!(body.contains("When uncertain, allow text"));
    }

    #[test]
    fn shell_request_schema_puts_reason_before_decision() {
        let context = GateContext {
            tool_name: "run_shell_command".to_string(),
            args: json!({"command": "cat src/lib.rs"}),
            messages: vec![ChatMessage::user("Read lib.rs")],
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };
        let body = build_shell_request_body("deepseek/deepseek-v4-flash", &context).unwrap();
        let reason_pos = body.find(r#""reason""#).unwrap();
        let decision_pos = body.find(r#""decision""#).unwrap();
        assert!(reason_pos < decision_pos, "{body}");
        assert!(body.contains("shell routing classifier"));
        assert!(body.contains("When uncertain, allow shell"));
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
    fn shell_gate_message_explains_retry_override() {
        let output = ShellClassifierOutput {
            reason: "USE_BUILTIN_TOOL because cat is an ordinary file read.".to_string(),
            decision: ShellClassifierDecision::UseBuiltinTool,
            recommended_tool: RecommendedTool::ReadFile,
            suggested_args: json!({"file_path": "src/lib.rs"}),
            confidence: GateConfidence::High,
        };
        let message = shell_gate_message(&output, &[]);

        assert!(message.contains("routing hint, not a final decision"));
        assert!(message.contains("retry the original shell command in your next tool batch"));
        assert!(message.contains("it will be allowed"));
        assert!(message.contains("Recommended next tool: `read_file`"));
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
    fn parses_shell_classifier_builtin_content() {
        let envelope = json!({
            "choices": [{
                "message": {
                    "content": "{\"reason\":\"USE_BUILTIN_TOOL because cat is an ordinary file read.\",\"decision\":\"use_builtin_tool\",\"recommended_tool\":\"read_file\",\"suggested_args\":{\"file_path\":\"src/lib.rs\"},\"confidence\":\"high\"}"
                }
            }]
        });
        let parsed = parse_shell_openrouter_response(&envelope.to_string()).unwrap();
        assert_eq!(parsed.decision, ShellClassifierDecision::UseBuiltinTool);
        assert_eq!(parsed.recommended_tool, RecommendedTool::ReadFile);
    }

    #[test]
    fn parses_shell_classifier_bifrost_content() {
        let envelope = json!({
            "choices": [{
                "message": {
                    "content": "{\"reason\":\"USE_BIFROST_TOOL because this grep is looking for callers.\",\"decision\":\"use_bifrost_tool\",\"recommended_tool\":\"scan_usages\",\"suggested_args\":{\"symbols\":[\"Foo\"]},\"confidence\":\"high\"}"
                }
            }]
        });
        let parsed = parse_shell_openrouter_response(&envelope.to_string()).unwrap();
        assert_eq!(parsed.decision, ShellClassifierDecision::UseBifrostTool);
        assert_eq!(parsed.recommended_tool, RecommendedTool::ScanUsages);
    }

    #[test]
    fn shell_allow_normalizes_recommended_tool_to_none() {
        let envelope = json!({
            "choices": [{
                "message": {
                    "content": "{\"reason\":\"ALLOW_SHELL because cargo test needs CLI semantics.\",\"decision\":\"allow_shell\",\"recommended_tool\":\"read_file\",\"suggested_args\":{\"file_path\":\"Cargo.toml\"},\"confidence\":\"high\"}"
                }
            }]
        });
        let parsed = parse_shell_openrouter_response(&envelope.to_string()).unwrap();
        assert_eq!(parsed.decision, ShellClassifierDecision::AllowShell);
        assert_eq!(parsed.recommended_tool, RecommendedTool::None);
        assert_eq!(parsed.suggested_args, json!({}));
    }

    #[test]
    fn rendered_context_includes_recent_bifrost_misses() {
        let mut miss = exchange("search_symbols");
        miss.arguments = "{\"patterns\":[\"FOO_MACRO\"]}".to_string();
        miss.result = "No symbols found for FOO_MACRO".to_string();
        let mut hit = exchange("scan_usages");
        hit.arguments = "{\"symbols\":[\"Bar\"]}".to_string();
        hit.result = "Found usages for Bar".to_string();
        let context = GateContext {
            tool_name: "run_shell_command".to_string(),
            args: json!({"command": "rg FOO_MACRO src"}),
            messages: vec![ChatMessage::user("Find FOO_MACRO")],
            tools: Vec::new(),
            tool_exchanges: vec![miss, hit],
        };

        let rendered: Value = serde_json::from_str(&render_context(&context).unwrap()).unwrap();
        let misses = rendered
            .get("recent_bifrost_misses")
            .and_then(Value::as_array)
            .unwrap();
        assert_eq!(misses.len(), 1);
        assert!(misses[0].to_string().contains("FOO_MACRO"));
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
