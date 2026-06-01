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
#[serde(rename_all = "snake_case")]
pub enum TextRouteAction {
    AllowOriginal,
    UseSearchSymbols,
    UseScanUsages,
    UseGetSummaries,
    UseGetSymbolSources,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShellRouteAction {
    AllowOriginal,
    UseReadFile,
    UseGrepSearch,
    UseListDirectory,
    UseEdit,
    UseWriteFile,
    UseSearchSymbols,
    UseScanUsages,
    UseGetSummaries,
    UseGetSymbolSources,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextIntent {
    ExactTextOrLocalizedRead,
    SequentialFileRead,
    WholeFileOrTopOfFileOrientation,
    KnownSymbolSource,
    SymbolDefinitionLookup,
    SymbolReferenceLookup,
    BroadSemanticOrientation,
    LiteralOrRegexSearch,
    PostEditOrValidationVerification,
    Unknown,
}

impl Default for TextIntent {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BifrostFit {
    SameOrMoreDirect,
    LessDirect,
    NotApplicable,
    Unknown,
}

impl Default for BifrostFit {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextAllowException {
    None,
    NonSourceText,
    ExactLiteralOrRegex,
    LocalizedOrSequentialRead,
    WholeFileOrTopOfFileOrientation,
    TestHeaderMacroOrEntrypointContext,
    PostEditOrBuildOrTestVerification,
    SameTokenOrPathBifrostMiss,
    Uncertain,
}

impl Default for TextAllowException {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextEvidence {
    pub symbol_tokens: Vec<String>,
    pub same_token_or_path_bifrost_miss: bool,
    pub same_path_recent_edit_or_write: bool,
    pub same_path_recent_bifrost_hit: bool,
    pub exact_text_or_regex_needed: bool,
}

impl Default for TextEvidence {
    fn default() -> Self {
        Self {
            symbol_tokens: Vec::new(),
            same_token_or_path_bifrost_miss: false,
            same_path_recent_edit_or_write: false,
            same_path_recent_bifrost_hit: false,
            exact_text_or_regex_needed: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShellIntent {
    BuildTestGitPackageOrProjectCli,
    EnvironmentOrPermissionProbe,
    RawByteOrFormatProbe,
    MutationOrWrite,
    OrdinaryFileRead,
    DirectoryOrFileDiscovery,
    LiteralTextSearch,
    SymbolDefinitionLookup,
    SymbolReferenceLookup,
    BroadSemanticOrientation,
    PipelineTransformationOrExitBehavior,
    Unknown,
}

impl Default for ShellIntent {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShellAllowException {
    None,
    ShellSemanticsRequired,
    RawByteOrHiddenWhitespace,
    BuildTestGitPackageOrProjectCli,
    MutationOrWriteOrDelete,
    HeredocOrCommandSubstitution,
    GeneratedArtifactOrExitBehavior,
    SameTokenOrPathBifrostMiss,
    Uncertain,
}

impl Default for ShellAllowException {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShellReplacementClass {
    AllowShellShellSemantics,
    AllowShellUncertain,
    UseBuiltinInspection,
    UseBifrostSymbol,
}

impl Default for ShellReplacementClass {
    fn default() -> Self {
        Self::AllowShellUncertain
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShellStaticRoute {
    AllowShell(&'static str),
    UseBuiltin(&'static str, RecommendedTool),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextStaticRoute {
    AllowText(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateClassifierOutput {
    // Keep this first. The request schema also serializes this first so the
    // classifier commits the rationale before emitting its decision.
    pub reason: String,
    #[serde(default)]
    pub intent: TextIntent,
    #[serde(default)]
    pub bifrost_fit: BifrostFit,
    #[serde(default)]
    pub allow_exception: TextAllowException,
    #[serde(default)]
    pub evidence: TextEvidence,
    pub decision: GateClassifierDecision,
    pub recommended_tool: RecommendedTool,
    #[serde(default = "empty_object")]
    pub suggested_args: Value,
    pub confidence: GateConfidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawGateClassifierOutput {
    pub reason: String,
    pub action: TextRouteAction,
    #[serde(default)]
    pub intent: TextIntent,
    #[serde(default)]
    pub bifrost_fit: BifrostFit,
    #[serde(default)]
    pub allow_exception: TextAllowException,
    #[serde(default)]
    pub evidence: TextEvidence,
    #[serde(default = "empty_object")]
    pub suggested_args: Value,
    pub confidence: GateConfidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShellClassifierOutput {
    // Keep this first so the model commits reasoning before the routing decision.
    pub reason: String,
    #[serde(default)]
    pub intent: ShellIntent,
    #[serde(default)]
    pub shell_semantics_required: bool,
    #[serde(default)]
    pub builtin_preserves_intent: bool,
    #[serde(default)]
    pub bifrost_fit: BifrostFit,
    #[serde(default)]
    pub allow_exception: ShellAllowException,
    #[serde(default)]
    pub replacement_class: ShellReplacementClass,
    pub decision: ShellClassifierDecision,
    pub recommended_tool: RecommendedTool,
    #[serde(default = "empty_object")]
    pub suggested_args: Value,
    pub confidence: GateConfidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawShellClassifierOutput {
    pub reason: String,
    pub action: ShellRouteAction,
    #[serde(default)]
    pub intent: ShellIntent,
    #[serde(default)]
    pub shell_semantics_required: bool,
    #[serde(default)]
    pub builtin_preserves_intent: bool,
    #[serde(default)]
    pub bifrost_fit: BifrostFit,
    #[serde(default)]
    pub allow_exception: ShellAllowException,
    #[serde(default)]
    pub replacement_class: ShellReplacementClass,
    #[serde(default = "empty_object")]
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

impl From<RawGateClassifierOutput> for GateClassifierOutput {
    fn from(raw: RawGateClassifierOutput) -> Self {
        let recommended_tool = match raw.action {
            TextRouteAction::AllowOriginal => RecommendedTool::None,
            TextRouteAction::UseSearchSymbols => RecommendedTool::SearchSymbols,
            TextRouteAction::UseScanUsages => RecommendedTool::ScanUsages,
            TextRouteAction::UseGetSummaries => RecommendedTool::GetSummaries,
            TextRouteAction::UseGetSymbolSources => RecommendedTool::GetSymbolSources,
        };
        let decision = if recommended_tool == RecommendedTool::None {
            GateClassifierDecision::AllowText
        } else {
            GateClassifierDecision::GateToSymbolTool
        };
        Self {
            reason: raw.reason,
            intent: raw.intent,
            bifrost_fit: raw.bifrost_fit,
            allow_exception: raw.allow_exception,
            evidence: raw.evidence,
            decision,
            recommended_tool,
            suggested_args: raw.suggested_args,
            confidence: raw.confidence,
        }
    }
}

impl From<RawShellClassifierOutput> for ShellClassifierOutput {
    fn from(raw: RawShellClassifierOutput) -> Self {
        let (decision, recommended_tool) = match raw.action {
            ShellRouteAction::AllowOriginal => {
                (ShellClassifierDecision::AllowShell, RecommendedTool::None)
            }
            ShellRouteAction::UseReadFile => (
                ShellClassifierDecision::UseBuiltinTool,
                RecommendedTool::ReadFile,
            ),
            ShellRouteAction::UseGrepSearch => (
                ShellClassifierDecision::UseBuiltinTool,
                RecommendedTool::GrepSearch,
            ),
            ShellRouteAction::UseListDirectory => (
                ShellClassifierDecision::UseBuiltinTool,
                RecommendedTool::ListDirectory,
            ),
            ShellRouteAction::UseEdit => (
                ShellClassifierDecision::UseBuiltinTool,
                RecommendedTool::Edit,
            ),
            ShellRouteAction::UseWriteFile => (
                ShellClassifierDecision::UseBuiltinTool,
                RecommendedTool::WriteFile,
            ),
            ShellRouteAction::UseSearchSymbols => (
                ShellClassifierDecision::UseBifrostTool,
                RecommendedTool::SearchSymbols,
            ),
            ShellRouteAction::UseScanUsages => (
                ShellClassifierDecision::UseBifrostTool,
                RecommendedTool::ScanUsages,
            ),
            ShellRouteAction::UseGetSummaries => (
                ShellClassifierDecision::UseBifrostTool,
                RecommendedTool::GetSummaries,
            ),
            ShellRouteAction::UseGetSymbolSources => (
                ShellClassifierDecision::UseBifrostTool,
                RecommendedTool::GetSymbolSources,
            ),
        };
        Self {
            reason: raw.reason,
            intent: raw.intent,
            shell_semantics_required: raw.shell_semantics_required,
            builtin_preserves_intent: raw.builtin_preserves_intent,
            bifrost_fit: raw.bifrost_fit,
            allow_exception: raw.allow_exception,
            replacement_class: raw.replacement_class,
            decision,
            recommended_tool,
            suggested_args: raw.suggested_args,
            confidence: raw.confidence,
        }
    }
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
    let retry_body = build_request_retry_body(&model, &context)?;
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
        let request_body = if attempt == 0 { &body } else { &retry_body };
        match send_classifier_request(
            &client,
            &api_key,
            request_body,
            parse_openrouter_response,
            cancel,
        )
            .await
        {
            Ok(mut output) => {
                enforce_text_classifier_policy(&mut output, &context);
                return Ok(output);
            }
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
    let retry_body = build_shell_request_retry_body(&model, &context)?;
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(CLASSIFIER_TIMEOUT)
        .build()
        .context("building OpenRouter shell classifier client")?;

    let mut last_err: Option<anyhow::Error> = None;
    for attempt in 0..2 {
        if cancel.is_cancelled() {
            return Err(anyhow!("classification cancelled"));
        }
        let request_body = if attempt == 0 { &body } else { &retry_body };
        match send_classifier_request(
            &client,
            &api_key,
            request_body,
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

pub fn static_text_route(
    tool_name: &str,
    args: &Value,
    exchanges: &[ToolExchange],
) -> Option<TextStaticRoute> {
    match tool_name {
        "read_file" if exact_localized_read_file(args, exchanges) => {
            Some(TextStaticRoute::AllowText("static_exact_localized_read"))
        }
        "grep_search" => {
            let glob = args.get("glob").and_then(Value::as_str).unwrap_or("");
            let path = args.get("path").and_then(Value::as_str).unwrap_or("");
            let file_path = args.get("file_path").and_then(Value::as_str).unwrap_or("");
            if grep_scope_granularity(glob, path, file_path) == "exact_file" {
                Some(TextStaticRoute::AllowText("static_exact_file_grep"))
            } else {
                None
            }
        }
        _ => None,
    }
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
    let suggested_args = suggested_args_block(&output.suggested_args);
    format!(
        "Bifrost gate: {reason}\n\nThe original text-navigation tool call was not executed. This is a navigation hint, not a final decision. If this hint is wrong and you need exact text, raw bytes, docs/config/log content, localized lines, or recovery after an edit/build/Bifrost miss, retry the original text tool call in your next tool batch; it will be allowed.\n\nRecommended next tool: `{tool_name}`.\n\nDescription: {description}\n\nSchema: {schema}{suggested_args}\n\nIf you are looking for declarations or definitions by name, use `search_symbols`. If you are looking for callers, references, or related tests for a known symbol, use `scan_usages`. If you are orienting across a module, package, class, API, or file glob, use `get_summaries`. If you already know the relevant symbol names and need implementation source, use `get_symbol_sources`.",
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
    let suggested_args = suggested_args_block(&output.suggested_args);
    format!(
        "Shell routing hint: {reason}\n\nThe original shell command was not executed. This is a routing hint, not a final decision. If this hint is wrong and you intentionally need shell semantics, a pipeline, raw bytes, build/test/git/package-manager behavior, or fallback after a Bifrost miss, retry the original shell command in your next tool batch; it will be allowed.\n\nRecommended next tool: `{tool_name}`.\n\nDescription: {description}\n\nSchema: {schema}{suggested_args}\n\nUse built-in tools for ordinary text reads, searches, listings, and simple text edits/writes. Use Bifrost tools for source declarations, definitions, callers, references, related tests, and broad code orientation. Use shell for build/test/git/package-manager behavior, byte/encoding/newline-sensitive inspection, generated artifacts, scripted transformations, and writes where shell/script behavior is the point.",
        reason = output.reason,
    )
}

pub fn static_shell_route_output(
    reason: &str,
    decision: ShellClassifierDecision,
    recommended_tool: RecommendedTool,
) -> ShellClassifierOutput {
    let (intent, replacement_class, bifrost_fit, builtin_preserves_intent) = match decision {
        ShellClassifierDecision::AllowShell => (
            ShellIntent::BuildTestGitPackageOrProjectCli,
            ShellReplacementClass::AllowShellShellSemantics,
            BifrostFit::NotApplicable,
            false,
        ),
        ShellClassifierDecision::UseBuiltinTool => (
            ShellIntent::LiteralTextSearch,
            ShellReplacementClass::UseBuiltinInspection,
            BifrostFit::NotApplicable,
            true,
        ),
        ShellClassifierDecision::UseBifrostTool => (
            ShellIntent::SymbolDefinitionLookup,
            ShellReplacementClass::UseBifrostSymbol,
            BifrostFit::SameOrMoreDirect,
            false,
        ),
    };
    ShellClassifierOutput {
        reason: format!(
            "{} because {reason}.",
            match decision {
                ShellClassifierDecision::AllowShell => "ALLOW_SHELL",
                ShellClassifierDecision::UseBuiltinTool => "USE_BUILTIN_TOOL",
                ShellClassifierDecision::UseBifrostTool => "USE_BIFROST_TOOL",
            }
        ),
        intent,
        shell_semantics_required: decision == ShellClassifierDecision::AllowShell,
        builtin_preserves_intent,
        bifrost_fit,
        allow_exception: if decision == ShellClassifierDecision::AllowShell {
            ShellAllowException::ShellSemanticsRequired
        } else {
            ShellAllowException::None
        },
        replacement_class,
        decision,
        recommended_tool,
        suggested_args: json!({}),
        confidence: GateConfidence::High,
    }
}

fn suggested_args_block(args: &Value) -> String {
    if args.as_object().map(|obj| obj.is_empty()).unwrap_or(false) {
        return String::new();
    }
    serde_json::to_string_pretty(args)
        .ok()
        .filter(|text| text != "{}")
        .map(|text| format!("\n\nSuggested args:\n```json\n{text}\n```"))
        .unwrap_or_default()
}

fn empty_object() -> Value {
    json!({})
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
        .ok_or_else(|| {
            anyhow!(
                "classifier response missing choices[0].message.content; envelope={}",
                truncate_to(text, 2_000)
            )
        })?;
    let raw: RawGateClassifierOutput = serde_json::from_str(content)
        .with_context(|| format!("parsing classifier JSON: {content}"))?;
    let mut output = GateClassifierOutput::from(raw);
    normalize_classifier_consistency(&mut output);
    Ok(output)
}

fn parse_shell_openrouter_response(text: &str) -> Result<ShellClassifierOutput> {
    let value: Value =
        serde_json::from_str(text).context("parsing shell classifier response envelope")?;
    let content = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            anyhow!(
                "shell classifier response missing choices[0].message.content; envelope={}",
                truncate_to(text, 2_000)
            )
        })?;
    let raw: RawShellClassifierOutput = serde_json::from_str(content)
        .with_context(|| format!("parsing shell classifier JSON: {content}"))?;
    let mut output = ShellClassifierOutput::from(raw);
    normalize_shell_classifier_consistency(&mut output);
    Ok(output)
}

fn build_request_body(model: &str, context: &GateContext) -> Result<String> {
    build_request_body_with_mode(model, context, false)
}

fn build_request_retry_body(model: &str, context: &GateContext) -> Result<String> {
    build_request_body_with_mode(model, context, true)
}

fn build_request_body_with_mode(
    model: &str,
    context: &GateContext,
    json_only_retry: bool,
) -> Result<String> {
    let system = if json_only_retry {
        "Return only a valid JSON object matching the schema. No prose, markdown, or hidden commentary. Classify the pending read_file/grep_search call from pending_call_features and compact_evidence. Gate broad source/test grep with identifier-like terms to Bifrost unless an exact/localized scope, post-edit/build/test verification, exact literal non-symbol text, or an observed empty/no-match Bifrost result supports allow_original. Infrastructure/protocol/refresh errors are not Bifrost misses. If uncertain, choose allow_original with confidence low."
    } else {
        "You are a routing classifier for an AI coding agent. You are advising a stronger coding model, not commanding it. Decide whether the pending read_file/grep_search call should run as text navigation or should be replaced by a Bifrost symbol tool. Use pending_call_features and compact_evidence first, prose excerpts second. Target policy: gate to Bifrost only when it is clearly the same-or-more-direct answer to a specific source-code symbol/navigation intent; otherwise allow exact text. Do not allow text merely because the agent wants to double-check or distrust Bifrost. A previous Bifrost call being skipped, filtered, or marked not_text_navigation_tool is not evidence that Bifrost failed. Bifrost internal/protocol/refresh errors are infrastructure failures to fix, not targeted symbol misses; do not allow text solely because of them. Allow exact/localized file reads, bounded reads, top-of-file orientation, post edit/build/test verification, non-source text, exact literal non-symbol searches, and observed targeted fallback for the same unresolved token/path after Bifrost returned an empty/no-match result. For grep_search, broad source/test glob or repo-wide scope plus identifier-like terms and no observed targeted Bifrost empty/no-match is usually a Bifrost action, even if the pattern uses alternation, C signatures, macros, or test identifiers. Exact-file scope, post-edit verification, exact literal non-symbol text, or fallback after observed Bifrost empty/no-match is allow_original. Use use_search_symbols for unknown declarations/definitions by name, use_scan_usages for callers/references/related tests, use_get_summaries for broad module/package/API orientation, and use_get_symbol_sources when relevant symbol names are already known and implementation source is needed. The action field is the only route the gate will execute. If Bifrost is preferred, choose the corresponding Bifrost action; never choose allow_original while saying Bifrost is preferred. The reason field must commit to one of these forms: 'ALLOW_TEXT because ...' or 'GATE_TO_SYMBOL_TOOL because ...'. Keep reason concise. Output only JSON matching the schema."
    };
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
          "action": {{"type":"string","enum":["allow_original","use_search_symbols","use_scan_usages","use_get_summaries","use_get_symbol_sources"]}},
          "intent": {{"type":"string","enum":["exact_text_or_localized_read","sequential_file_read","whole_file_or_top_of_file_orientation","known_symbol_source","symbol_definition_lookup","symbol_reference_lookup","broad_semantic_orientation","literal_or_regex_search","post_edit_or_validation_verification","unknown"]}},
          "bifrost_fit": {{"type":"string","enum":["same_or_more_direct","less_direct","not_applicable","unknown"]}},
          "allow_exception": {{"type":"string","enum":["none","non_source_text","exact_literal_or_regex","localized_or_sequential_read","whole_file_or_top_of_file_orientation","test_header_macro_or_entrypoint_context","post_edit_or_build_or_test_verification","same_token_or_path_bifrost_miss","uncertain"]}},
          "evidence": {{
            "type":"object",
            "additionalProperties": false,
            "properties": {{
              "symbol_tokens": {{"type":"array","items":{{"type":"string"}}}},
              "same_token_or_path_bifrost_miss": {{"type":"boolean"}},
              "same_path_recent_edit_or_write": {{"type":"boolean"}},
              "same_path_recent_bifrost_hit": {{"type":"boolean"}},
              "exact_text_or_regex_needed": {{"type":"boolean"}}
            }},
            "required": ["symbol_tokens","same_token_or_path_bifrost_miss","same_path_recent_edit_or_write","same_path_recent_bifrost_hit","exact_text_or_regex_needed"]
          }},
          "suggested_args": {{"type":"object"}},
          "confidence": {{"type":"string","enum":["low","medium","high"]}}
        }},
        "required": ["reason","action","intent","bifrost_fit","allow_exception","evidence","suggested_args","confidence"]
      }}
    }}
  }},
  "temperature": 0,
  "max_tokens": 2048
}}"#
    ))
}

fn build_shell_request_body(model: &str, context: &GateContext) -> Result<String> {
    build_shell_request_body_with_mode(model, context, false)
}

fn build_shell_request_retry_body(model: &str, context: &GateContext) -> Result<String> {
    build_shell_request_body_with_mode(model, context, true)
}

fn build_shell_request_body_with_mode(
    model: &str,
    context: &GateContext,
    json_only_retry: bool,
) -> Result<String> {
    let system = if json_only_retry {
        "Return only a valid JSON object matching the schema. No prose, markdown, or hidden commentary. Classify by purpose from pending_call_features and compact_evidence. Use builtins for ordinary file reads/searches/listings when they preserve intent. Use Bifrost only for clear source-code symbol discovery. Allow shell only for concrete shell semantics such as build/test/git/package commands, env/path probes, raw bytes, mutation, generated artifacts, or meaningful pipeline behavior. If uncertain, choose allow_original with confidence low."
    } else {
        "You are a shell routing classifier for an AI coding agent. You are advising a stronger coding model, not commanding it. Classify by purpose, not by syntax. Use pending_call_features and compact_evidence first, prose excerpts second. Allow shell when shell semantics are materially part of the task: build/test/git/package/project CLI, env/permission/path probing, raw bytes or hidden whitespace, generated artifacts, command substitution, mutation/write/delete behavior, or a pipeline whose transformation or exit behavior matters. Use builtins when the command only reads, searches, lists, or prints bounded ranges, and shell syntax is only being used to limit, count, or pretty-print inspection output. A filename/path search with find, ls, or path globs is builtin inspection, not Bifrost. Grep/rg/git-grep-like shell commands should usually use use_grep_search first; choose a Bifrost action only when the command's primary purpose is clearly source-code symbol discovery and builtin grep_search would be a worse route. Do not allow shell merely because the user used shell syntax, heredoc, Python, grep, sed, head, tail, xargs, or a pipe. A previous Bifrost call being skipped, filtered, or marked not_text_navigation_tool is not evidence that Bifrost failed. When uncertain between builtin and shell, allow shell only if you can identify a concrete shell-specific semantic that may matter. When uncertain between Bifrost and builtin/text, prefer builtin/text unless the command is clearly source symbol discovery. The action field is the only route the gate will execute. If a builtin or Bifrost tool is preferred, choose that action; never choose allow_original while recommending another tool. The reason field must commit to one of these forms: 'ALLOW_SHELL because ...', 'USE_BUILTIN_TOOL because ...', or 'USE_BIFROST_TOOL because ...'. Keep reason concise. Output only JSON matching the schema."
    };
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
          "action": {{"type":"string","enum":["allow_original","use_read_file","use_grep_search","use_list_directory","use_edit","use_write_file","use_search_symbols","use_scan_usages","use_get_summaries","use_get_symbol_sources"]}},
          "intent": {{"type":"string","enum":["build_test_git_package_or_project_cli","environment_or_permission_probe","raw_byte_or_format_probe","mutation_or_write","ordinary_file_read","directory_or_file_discovery","literal_text_search","symbol_definition_lookup","symbol_reference_lookup","broad_semantic_orientation","pipeline_transformation_or_exit_behavior","unknown"]}},
          "shell_semantics_required": {{"type":"boolean"}},
          "builtin_preserves_intent": {{"type":"boolean"}},
          "bifrost_fit": {{"type":"string","enum":["same_or_more_direct","less_direct","not_applicable","unknown"]}},
          "allow_exception": {{"type":"string","enum":["none","shell_semantics_required","raw_byte_or_hidden_whitespace","build_test_git_package_or_project_cli","mutation_or_write_or_delete","heredoc_or_command_substitution","generated_artifact_or_exit_behavior","same_token_or_path_bifrost_miss","uncertain"]}},
          "replacement_class": {{"type":"string","enum":["allow_shell_shell_semantics","allow_shell_uncertain","use_builtin_inspection","use_bifrost_symbol"]}},
          "suggested_args": {{"type":"object"}},
          "confidence": {{"type":"string","enum":["low","medium","high"]}}
        }},
        "required": ["reason","action","intent","shell_semantics_required","builtin_preserves_intent","bifrost_fit","allow_exception","replacement_class","suggested_args","confidence"]
      }}
    }}
  }},
  "temperature": 0,
  "max_tokens": 2048
}}"#
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoutePrefix {
    AllowOriginal,
    UseBuiltin,
    UseBifrost,
}

fn normalize_classifier_consistency(output: &mut GateClassifierOutput) {
    if output.decision == GateClassifierDecision::AllowText
        && route_prefix(&output.reason) == Some(RoutePrefix::UseBifrost)
        && output.confidence == GateConfidence::High
    {
        output.decision = GateClassifierDecision::GateToSymbolTool;
        output.recommended_tool = bifrost_tool_for_text_intent(&output.intent);
        return;
    }

    let must_allow = output.confidence != GateConfidence::High;
    if output.decision == GateClassifierDecision::AllowText
        || must_allow
        || !is_bifrost_recommendation(&output.recommended_tool)
    {
        output.decision = GateClassifierDecision::AllowText;
        output.recommended_tool = RecommendedTool::None;
        output.suggested_args = json!({});
    }
}

fn enforce_text_classifier_policy(output: &mut GateClassifierOutput, context: &GateContext) {
    if context.tool_name != "grep_search"
        || output.decision != GateClassifierDecision::AllowText
        || output.confidence != GateConfidence::High
    {
        return;
    }
    let pattern = context
        .args
        .get("pattern")
        .and_then(Value::as_str)
        .unwrap_or("");
    if !symbol_like_pattern(pattern) {
        return;
    }
    let glob = context.args.get("glob").and_then(Value::as_str).unwrap_or("");
    let path = context.args.get("path").and_then(Value::as_str).unwrap_or("");
    let file_path = context
        .args
        .get("file_path")
        .and_then(Value::as_str)
        .unwrap_or("");
    if grep_scope_granularity(glob, path, file_path) == "exact_file"
        || !broad_source_or_test_scope(glob, path)
    {
        return;
    }
    if supported_broad_symbol_grep_allow(output, pattern, glob, path, file_path, context) {
        return;
    }
    output.decision = GateClassifierDecision::GateToSymbolTool;
    output.recommended_tool = bifrost_tool_for_text_intent(&output.intent);
    output.suggested_args = json!({});
    output.reason = format!(
        "GATE_TO_SYMBOL_TOOL because broad source/test grep for symbol-like pattern `{}` lacks a supported allow exception.",
        truncate_to(pattern, 120)
    );
}

fn supported_broad_symbol_grep_allow(
    output: &GateClassifierOutput,
    pattern: &str,
    glob: &str,
    path: &str,
    file_path: &str,
    context: &GateContext,
) -> bool {
    let scope_target = grep_scope_target(glob, path, file_path);
    if output.evidence.same_token_or_path_bifrost_miss
        || overlaps_recent_bifrost_miss(pattern, &context.tool_exchanges)
        || overlaps_recent_bifrost_miss(&scope_target, &context.tool_exchanges)
    {
        return true;
    }
    if output.evidence.same_path_recent_edit_or_write
        || recent_edit_or_build_failure(&context.tool_exchanges)
        || same_path_recent_tool(&scope_target, &context.tool_exchanges, &["edit", "write_file"])
    {
        return true;
    }
    if matches!(
        output.allow_exception,
        TextAllowException::LocalizedOrSequentialRead
            | TextAllowException::WholeFileOrTopOfFileOrientation
    ) && (same_path_recent_tool(&scope_target, &context.tool_exchanges, &["read_file"])
        || !scope_target.trim().is_empty() && scope_target != ".")
    {
        return true;
    }
    matches!(output.allow_exception, TextAllowException::ExactLiteralOrRegex)
        && literal_like_pattern(pattern)
        && !symbol_like_pattern(pattern)
}

fn normalize_shell_classifier_consistency(output: &mut ShellClassifierOutput) {
    if output.confidence != GateConfidence::High {
        output.decision = ShellClassifierDecision::AllowShell;
        output.recommended_tool = RecommendedTool::None;
        output.suggested_args = json!({});
        return;
    }

    match route_prefix(&output.reason) {
        Some(RoutePrefix::UseBifrost) => {
            output.decision = ShellClassifierDecision::UseBifrostTool;
            output.recommended_tool = bifrost_tool_for_shell_intent(&output.intent);
            return;
        }
        Some(RoutePrefix::UseBuiltin) => {
            output.decision = ShellClassifierDecision::UseBuiltinTool;
            output.recommended_tool = builtin_tool_for_shell_intent(&output.intent);
            return;
        }
        _ => {}
    }

    match output.decision {
        ShellClassifierDecision::UseBifrostTool
            if is_bifrost_recommendation(&output.recommended_tool) => {}
        ShellClassifierDecision::UseBuiltinTool
            if is_builtin_recommendation(&output.recommended_tool) => {}
        ShellClassifierDecision::AllowShell => {
            output.recommended_tool = RecommendedTool::None;
            output.suggested_args = json!({});
        }
        _ => {
            output.decision = ShellClassifierDecision::AllowShell;
            output.recommended_tool = RecommendedTool::None;
            output.suggested_args = json!({});
        }
    }
}

fn route_prefix(reason: &str) -> Option<RoutePrefix> {
    let first = reason
        .trim_start()
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches([':', ',', '.']);
    match first {
        "GATE_TO_SYMBOL_TOOL" | "USE_BIFROST_TOOL" | "ROUTE_TO_BIFROST" => {
            Some(RoutePrefix::UseBifrost)
        }
        "USE_BUILTIN_TOOL" => Some(RoutePrefix::UseBuiltin),
        "ALLOW_TEXT" | "ALLOW_SHELL" | "ALLOW_ORIGINAL" => Some(RoutePrefix::AllowOriginal),
        _ => None,
    }
}

fn builtin_tool_for_shell_intent(intent: &ShellIntent) -> RecommendedTool {
    match intent {
        ShellIntent::DirectoryOrFileDiscovery => RecommendedTool::ListDirectory,
        ShellIntent::LiteralTextSearch => RecommendedTool::GrepSearch,
        ShellIntent::MutationOrWrite => RecommendedTool::Edit,
        _ => RecommendedTool::ReadFile,
    }
}

fn bifrost_tool_for_text_intent(intent: &TextIntent) -> RecommendedTool {
    match intent {
        TextIntent::KnownSymbolSource => RecommendedTool::GetSymbolSources,
        TextIntent::SymbolReferenceLookup => RecommendedTool::ScanUsages,
        TextIntent::BroadSemanticOrientation => RecommendedTool::GetSummaries,
        TextIntent::SymbolDefinitionLookup => RecommendedTool::SearchSymbols,
        _ => RecommendedTool::SearchSymbols,
    }
}

fn bifrost_tool_for_shell_intent(intent: &ShellIntent) -> RecommendedTool {
    match intent {
        ShellIntent::SymbolReferenceLookup => RecommendedTool::ScanUsages,
        ShellIntent::BroadSemanticOrientation => RecommendedTool::GetSummaries,
        ShellIntent::SymbolDefinitionLookup => RecommendedTool::SearchSymbols,
        _ => RecommendedTool::SearchSymbols,
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

fn render_context(context: &GateContext) -> Result<String> {
    let first_user_prompt = context
        .messages
        .iter()
        .find(|message| message.role == "user")
        .map(ChatMessage::content_text)
        .unwrap_or_default();
    let latest_user_prompt = context
        .messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .map(ChatMessage::content_text)
        .unwrap_or_default();
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
                "args_excerpt": truncate_to(&exchange.arguments, 600),
                "result_kind": result_kind(&exchange.result),
            })
        })
        .collect();
    let features =
        pending_call_features(&context.tool_name, &context.args, &context.tool_exchanges);
    let compact_evidence =
        compact_classifier_evidence(&context.tool_name, &context.args, &context.tool_exchanges);
    let payload = json!({
        "task_prompt_excerpt": truncate(&first_user_prompt),
        "latest_user_message_excerpt": truncate(&latest_user_prompt),
        "pending_call": {
            "tool": context.tool_name,
            "args": context.args,
        },
        "pending_call_features": features,
        "compact_evidence": compact_evidence,
        "recent_tool_summary": recent,
        "available_bifrost_tools": available_bifrost_tool_names(&context.tools),
        "bifrost_tool_purposes": bifrost_tool_purposes(),
        "static_text_target": classify_static_text_target(&context.tool_name, &context.args),
    });
    serde_json::to_string_pretty(&payload).context("rendering classifier context")
}

fn available_bifrost_tool_names(tools: &[ToolDefinition]) -> Vec<&str> {
    tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .filter(|name| is_bifrost_tool(name))
        .collect()
}

fn pending_call_features(tool_name: &str, args: &Value, exchanges: &[ToolExchange]) -> Value {
    match tool_name {
        "read_file" => read_file_features(args, exchanges),
        "grep_search" => grep_search_features(args, exchanges),
        "run_shell_command" => shell_command_features(args),
        _ => json!({ "tool": tool_name }),
    }
}

fn compact_classifier_evidence(tool_name: &str, args: &Value, exchanges: &[ToolExchange]) -> Value {
    let target = primary_target(tool_name, args);
    let symbols = symbol_tokens_for_call(tool_name, args);
    json!({
        "intent_hints": intent_hints(tool_name, args),
        "same_path_recent_read": same_path_recent_tool(&target, exchanges, &["read_file"]),
        "sequential_read_from_same_file": sequential_read_from_same_file(args, exchanges),
        "same_path_recent_edit_or_write": same_path_recent_tool(&target, exchanges, &["edit", "write_file"]),
        "after_build_or_test_failure": recent_edit_or_build_failure(exchanges),
        "same_token_bifrost_miss": symbols.iter().any(|token| overlaps_recent_bifrost_miss(token, exchanges)),
        "same_path_bifrost_miss": overlaps_recent_bifrost_miss(&target, exchanges),
        "same_symbol_bifrost_hit": symbols.iter().any(|token| overlaps_recent_bifrost_hit(token, exchanges)),
        "simple_builtin_equivalent": simple_builtin_equivalent(tool_name, args),
        "shell_semantics_required_by_features": tool_name == "run_shell_command" && shell_semantics_required_by_features(args),
        "symbol_tokens": symbols,
    })
}

fn bifrost_tool_purposes() -> Value {
    json!([
        {
            "name": "search_symbols",
            "purpose": "Find source symbols, declarations, or definitions by name/pattern."
        },
        {
            "name": "scan_usages",
            "purpose": "Find callers, references, usages, and related tests for known symbols."
        },
        {
            "name": "get_summaries",
            "purpose": "Orient across source files, modules, packages, classes, APIs, or globs."
        },
        {
            "name": "get_symbol_sources",
            "purpose": "Retrieve implementation source for known symbols."
        }
    ])
}

fn read_file_features(args: &Value, exchanges: &[ToolExchange]) -> Value {
    let path = args.get("file_path").and_then(Value::as_str).unwrap_or("");
    json!({
        "path": path,
        "extension": extension(path),
        "static_text_target": classify_static_text_target("read_file", args),
        "has_offset": args.get("offset").is_some(),
        "has_limit": args.get("limit").is_some(),
        "source_like": is_source_like_path(path),
        "recent_bifrost_miss_overlap": overlaps_recent_bifrost_miss(path, exchanges),
        "recent_bifrost_hit_overlap": overlaps_recent_bifrost_hit(path, exchanges),
        "after_successful_edit": recent_tool_success(exchanges, &["edit", "write_file"]),
        "after_failed_edit_or_build": recent_edit_or_build_failure(exchanges),
    })
}

fn grep_search_features(args: &Value, exchanges: &[ToolExchange]) -> Value {
    let pattern = args.get("pattern").and_then(Value::as_str).unwrap_or("");
    let glob = args.get("glob").and_then(Value::as_str).unwrap_or("");
    let path = args.get("path").and_then(Value::as_str).unwrap_or("");
    let file_path = args.get("file_path").and_then(Value::as_str).unwrap_or("");
    let scope_target = grep_scope_target(glob, path, file_path);
    let scope_granularity = grep_scope_granularity(glob, path, file_path);
    json!({
        "pattern": pattern,
        "glob": glob,
        "path": path,
        "file_path": file_path,
        "scope_granularity": scope_granularity,
        "exact_file_scope": scope_granularity == "exact_file",
        "broad_source_or_test_scope": broad_source_or_test_scope(glob, path),
        "grep_scope_recent_edit_or_write": same_path_recent_tool(&scope_target, exchanges, &["edit", "write_file"]),
        "grep_scope_recent_read": same_path_recent_tool(&scope_target, exchanges, &["read_file"]),
        "source_like_scope": is_source_like_path(glob) || is_source_like_path(path) || (glob.is_empty() && path.is_empty()),
        "broad_repo_scope": glob.is_empty() && path.is_empty(),
        "symbol_like_pattern": symbol_like_pattern(pattern),
        "literal_like_pattern": literal_like_pattern(pattern),
        "recent_bifrost_miss_overlap": overlaps_recent_bifrost_miss(pattern, exchanges),
        "search_scope": search_scope(glob, path),
        "pattern_shape": pattern_shape(pattern),
        "prior_bifrost_result_status": prior_bifrost_result_status(pattern, exchanges),
    })
}

fn exact_localized_read_file(args: &Value, exchanges: &[ToolExchange]) -> bool {
    let Some(path) = args.get("file_path").and_then(Value::as_str) else {
        return false;
    };
    if path.trim().is_empty() || path.contains('*') {
        return false;
    }
    if classify_path_or_glob(path) == StaticTextTarget::TextLike {
        return true;
    }
    if args.get("offset").is_some() || args.get("limit").is_some() {
        return true;
    }
    if recent_edit_or_build_failure(exchanges)
        || same_path_recent_tool(path, exchanges, &["read_file", "edit", "write_file"])
        || path_localized_by_recent_tool(path, exchanges)
    {
        return true;
    }
    if source_or_test_path(path) && recent_distinct_source_reads(exchanges) < 2 {
        return true;
    }
    false
}

fn path_localized_by_recent_tool(path: &str, exchanges: &[ToolExchange]) -> bool {
    if path.len() < 3 {
        return false;
    }
    let file_name = path.rsplit('/').next().unwrap_or(path);
    exchanges.iter().rev().take(8).any(|exchange| {
        matches!(
            exchange.tool_name.as_str(),
            "grep_search"
                | "list_directory"
                | "search_symbols"
                | "scan_usages"
                | "get_summaries"
                | "get_symbol_sources"
                | "run_shell_command"
        ) && (exchange.arguments.contains(path)
            || exchange.result.contains(path)
            || (file_name.len() >= 3 && exchange.result.contains(file_name)))
    })
}

fn recent_distinct_source_reads(exchanges: &[ToolExchange]) -> usize {
    let mut paths: Vec<String> = Vec::new();
    for exchange in exchanges.iter().rev().take(8) {
        if exchange.tool_name != "read_file" {
            continue;
        }
        let Ok(args) = serde_json::from_str::<Value>(&exchange.arguments) else {
            continue;
        };
        let Some(path) = args.get("file_path").and_then(Value::as_str) else {
            continue;
        };
        if source_or_test_path(path) && !paths.iter().any(|prior| prior == path) {
            paths.push(path.to_string());
        }
    }
    paths.len()
}

fn source_or_test_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    is_source_like_path(path)
        || lower.contains("/test")
        || lower.contains("tests/")
        || lower.contains("test/")
        || lower.contains("spec/")
}

fn shell_command_features(args: &Value) -> Value {
    let command = args.get("command").and_then(Value::as_str).unwrap_or("");
    json!({
        "command_name": command_name(command),
        "has_pipeline": command.contains('|'),
        "has_redirection": has_redirection(command),
        "has_heredoc": command.contains("<<"),
        "mutates_files": mutates_files(command),
        "build_test_git_package_like": build_test_git_package_like(command),
        "raw_byte_probe": raw_byte_probe(command),
        "simple_read_like": simple_shell_read_like(command),
        "simple_search_like": simple_shell_search_like(command),
        "symbol_search_like": shell_symbol_search_like(command),
        "script_raw_byte_or_format_probe": script_raw_byte_or_format_probe(command),
        "script_mutates_files": script_mutates_files(command),
        "shell_semantics_required_by_features": shell_semantics_required_by_features(args),
    })
}

fn primary_target(tool_name: &str, args: &Value) -> String {
    match tool_name {
        "read_file" => args
            .get("file_path")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        "grep_search" => args
            .get("pattern")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        "run_shell_command" => args
            .get("command")
            .and_then(Value::as_str)
            .and_then(shell_search_pattern)
            .unwrap_or_else(|| {
                args.get("command")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string()
            }),
        _ => String::new(),
    }
}

fn symbol_tokens_for_call(tool_name: &str, args: &Value) -> Vec<String> {
    let raw = match tool_name {
        "grep_search" => args
            .get("pattern")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        "read_file" => args
            .get("file_path")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        "run_shell_command" => args
            .get("command")
            .and_then(Value::as_str)
            .and_then(shell_search_pattern)
            .unwrap_or_default(),
        _ => String::new(),
    };
    raw.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .filter(|token| token.len() >= 3 && symbol_like_pattern(token))
        .take(8)
        .map(str::to_string)
        .collect()
}

fn identifier_tokens(pattern: &str) -> Vec<String> {
    pattern
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .filter(|token| token.len() >= 3)
        .take(12)
        .map(str::to_string)
        .collect()
}

fn pattern_shape(pattern: &str) -> Value {
    let identifiers = identifier_tokens(pattern);
    let symbol_terms: Vec<_> = identifiers
        .iter()
        .filter(|token| symbol_like_pattern(token))
        .cloned()
        .collect();
    let macro_or_test_terms: Vec<_> = identifiers
        .iter()
        .filter(|token| {
            token.chars().all(|ch| ch.is_ascii_uppercase() || ch == '_')
                || token.to_ascii_lowercase().contains("test")
        })
        .cloned()
        .collect();
    let non_identifier_terms: Vec<_> = pattern
        .split(['|', ',', ';'])
        .map(str::trim)
        .filter(|term| !term.is_empty() && !identifier_tokens(term).iter().any(|t| t == *term))
        .take(8)
        .map(str::to_string)
        .collect();
    json!({
        "has_alternation": pattern.contains('|'),
        "identifier_terms": symbol_terms,
        "macro_or_test_terms": macro_or_test_terms,
        "non_identifier_terms": non_identifier_terms,
        "string_literal_or_raw_text_cues": literal_like_pattern(pattern),
        "regex_semantics_likely_material": regex_semantics_likely_material(pattern),
    })
}

fn regex_semantics_likely_material(pattern: &str) -> bool {
    pattern.contains(".*")
        || pattern.contains("[^")
        || pattern.contains("\\s")
        || pattern.contains("\\d")
        || pattern.contains('^')
        || pattern.contains('$')
}

fn search_scope(glob: &str, path: &str) -> &'static str {
    let combined = format!("{glob} {path}").to_ascii_lowercase();
    if combined.contains("doc")
        || matches!(classify_path_or_glob(&combined), StaticTextTarget::TextLike)
    {
        "docs"
    } else if combined.contains("test") || combined.contains("spec") {
        "tests"
    } else if combined.is_empty() || is_source_like_path(&combined) {
        "source_code"
    } else {
        "unknown"
    }
}

fn grep_scope_target(glob: &str, path: &str, file_path: &str) -> String {
    if !file_path.trim().is_empty() {
        file_path.trim().to_string()
    } else if !glob.trim().is_empty()
        && (path.trim().is_empty() || path.trim() == ".")
        && !glob.contains('*')
        && is_source_like_path(glob)
        && glob.contains('.')
    {
        glob.trim().to_string()
    } else if !path.trim().is_empty() {
        path.trim().to_string()
    } else {
        glob.trim().to_string()
    }
}

fn grep_scope_granularity(glob: &str, path: &str, file_path: &str) -> &'static str {
    let target = grep_scope_target(glob, path, file_path);
    if target != "."
        && !target.contains('*')
        && is_source_like_path(&target)
        && target.contains('.')
    {
        return "exact_file";
    }
    if target.is_empty() || target == "." {
        return "repo_wide";
    }
    if target.contains('*') && is_source_like_path(&target) {
        return "source_glob";
    }
    if target.ends_with('/') || (!target.contains('*') && !target.contains('.')) {
        return "directory";
    }
    "unknown"
}

fn broad_source_or_test_scope(glob: &str, path: &str) -> bool {
    let scope = grep_scope_granularity(glob, path, "");
    let combined = format!("{glob} {path}").to_ascii_lowercase();
    matches!(scope, "repo_wide" | "source_glob")
        || matches!(path, "." | "src" | "source" | "test" | "tests")
        || combined.contains("test/")
        || combined.contains("/test")
        || combined.contains("tests/")
        || combined.contains("/tests")
}

fn prior_bifrost_result_status(needle: &str, exchanges: &[ToolExchange]) -> &'static str {
    let needle = needle.trim();
    let mut saw_bifrost = false;
    for exchange in exchanges.iter().rev().take(8) {
        if !is_bifrost_tool(&exchange.tool_name) {
            continue;
        }
        saw_bifrost = true;
        if needle.len() >= 3
            && !(exchange.arguments.contains(needle) || exchange.result.contains(needle))
        {
            continue;
        }
        if looks_like_failure(&exchange.result) {
            return "observed_inadequate";
        }
        if looks_like_bifrost_miss(&exchange.result) {
            return "observed_empty";
        }
        return "observed_hit";
    }
    if saw_bifrost { "unknown" } else { "none" }
}

fn intent_hints(tool_name: &str, args: &Value) -> Vec<&'static str> {
    let mut hints = Vec::new();
    match tool_name {
        "read_file" => {
            let path = args.get("file_path").and_then(Value::as_str).unwrap_or("");
            if args.get("offset").is_some() || args.get("limit").is_some() {
                hints.push("localized_read");
            }
            if is_source_like_path(path) {
                hints.push("source_path");
            } else {
                hints.push("non_source_text");
            }
        }
        "grep_search" => {
            let pattern = args.get("pattern").and_then(Value::as_str).unwrap_or("");
            if symbol_like_pattern(pattern) {
                hints.push("symbol_like_pattern");
            }
            if literal_like_pattern(pattern) {
                hints.push("literal_like_pattern");
            }
        }
        "run_shell_command" => {
            let command = args.get("command").and_then(Value::as_str).unwrap_or("");
            if simple_shell_read_like(command) {
                hints.push("simple_file_inspection");
            }
            if shell_symbol_search_like(command) {
                hints.push("symbol_search_like");
            }
            if build_test_git_package_like(command) {
                hints.push("build_test_git_package_like");
            }
            if raw_byte_probe(command) {
                hints.push("raw_byte_probe");
            }
        }
        _ => {}
    }
    hints
}

fn simple_builtin_equivalent(tool_name: &str, args: &Value) -> bool {
    match tool_name {
        "read_file" | "grep_search" | "list_directory" => true,
        "run_shell_command" => args
            .get("command")
            .and_then(Value::as_str)
            .map(|command| {
                (simple_shell_read_like(command) || simple_shell_search_like(command))
                    && !shell_semantics_required(command)
            })
            .unwrap_or(false),
        _ => false,
    }
}

fn shell_semantics_required_by_features(args: &Value) -> bool {
    args.get("command")
        .and_then(Value::as_str)
        .map(shell_semantics_required)
        .unwrap_or(false)
}

fn shell_semantics_required(command: &str) -> bool {
    command_segments(command).iter().any(|segment| {
        build_test_git_package_like(segment)
            || raw_byte_probe(segment)
            || shell_mutates_files(segment)
            || script_raw_byte_or_format_probe(segment)
            || script_mutates_files(segment)
            || inline_runtime_execution(segment)
    }) || script_raw_byte_or_format_probe(command)
        || script_mutates_files(command)
        || command.contains("$(")
}

fn same_path_recent_tool(path: &str, exchanges: &[ToolExchange], tools: &[&str]) -> bool {
    if path.len() < 3 {
        return false;
    }
    exchanges.iter().rev().take(8).any(|exchange| {
        tools.contains(&exchange.tool_name.as_str()) && exchange.arguments.contains(path)
    })
}

fn sequential_read_from_same_file(args: &Value, exchanges: &[ToolExchange]) -> bool {
    let Some(path) = args.get("file_path").and_then(Value::as_str) else {
        return false;
    };
    args.get("offset").is_some() && same_path_recent_tool(path, exchanges, &["read_file"])
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

pub fn static_shell_route(args: &Value) -> Option<ShellStaticRoute> {
    let command = args.get("command").and_then(Value::as_str)?.trim();
    if command.is_empty() {
        return None;
    }
    if shell_semantics_required(command) {
        return Some(ShellStaticRoute::AllowShell("static_shell_semantics"));
    }
    if simple_shell_read_like(command) || simple_shell_search_like(command) {
        return Some(ShellStaticRoute::UseBuiltin(
            "static_shell_builtin_inspection",
            builtin_tool_for_shell_command(command),
        ));
    }
    None
}

fn truncate(text: &str) -> String {
    truncate_to(text, MAX_TEXT)
}

fn truncate_to(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        return text.to_string();
    }
    let mut end = max_len;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…[truncated]", &text[..end])
}

fn result_kind(result: &str) -> &'static str {
    if looks_like_failure(result) {
        "failure"
    } else if looks_like_bifrost_miss(result) {
        "empty_or_miss"
    } else {
        "success"
    }
}

fn command_name(command: &str) -> String {
    command
        .trim()
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_start_matches("env")
        .to_string()
}

fn build_test_git_package_like(command: &str) -> bool {
    let normalized = strip_shell_wrappers(command);
    let trimmed = normalized.trim();
    let lower = trimmed.to_ascii_lowercase();
    const PREFIXES: &[&str] = &[
        "cargo ",
        "composer ",
        "go test",
        "go build",
        "go fmt",
        "go vet",
        "gofmt ",
        "apt-get ",
        "sudo apt-get ",
        "npm ",
        "pnpm ",
        "yarn ",
        "mvn ",
        "gradle ",
        "./gradlew",
        "pytest",
        "phpunit",
        "vendor/bin/phpunit",
        "phpcs",
        "vendor/bin/phpcs",
        "php -l",
        "python -m pytest",
        "uv run pytest",
        "uv run python -m pytest",
        "dotnet build",
        "dotnet test",
        "sbt ",
        "bash .harness/",
        "make ",
        "cmake ",
        "ctest ",
        "git ",
        "./scripts/",
        "./run",
        ".harness/",
        "./.harness/",
    ];
    PREFIXES.iter().any(|prefix| lower.starts_with(prefix))
        || lower.contains("/go test")
        || lower.contains("/go build")
        || lower.contains("/go fmt")
        || lower.contains("/go vet")
        || lower.contains("/phpunit")
        || lower.contains("/phpcs")
        || lower.contains("/sbt")
        || lower.contains("bin/sbt")
        || lower.contains("uv_run_tests")
        || lower.contains(" testonly")
}

fn raw_byte_probe(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    lower.starts_with("xxd ")
        || lower.starts_with("od ")
        || lower.starts_with("hexdump ")
        || lower.contains(" cat -a ")
        || lower.starts_with("cat -a ")
        || lower.contains(" cat -a ")
        || lower.starts_with("cat -a ")
}

fn has_redirection(command: &str) -> bool {
    command.contains('>') || command.contains('<')
}

fn mutates_files(command: &str) -> bool {
    shell_mutates_files(command)
}

fn shell_mutates_files(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    has_output_redirection(command)
        || lower.contains("sed -i")
        || lower.starts_with("rm ")
        || lower.starts_with("mv ")
        || lower.starts_with("cp ")
        || lower.starts_with("mkdir ")
        || lower.starts_with("touch ")
}

fn script_raw_byte_or_format_probe(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    if !scriptish(&lower) {
        return false;
    }
    lower.contains(".read_bytes(")
        || (lower.contains("open(") && (lower.contains("'rb'") || lower.contains("\"rb\"")))
        || lower.contains("utf-8-sig")
        || lower.contains("bom")
        || (lower.contains("repr(") && lower.contains("read"))
        || lower.contains("split(b")
        || lower.contains("\\r\\n")
}

fn script_mutates_files(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    if !scriptish(&lower) {
        return false;
    }
    lower.contains(".write_text(")
        || lower.contains(".write_bytes(")
        || (lower.contains("open(")
            && (lower.contains("'w'")
                || lower.contains("\"w\"")
                || lower.contains("'wb'")
                || lower.contains("\"wb\"")
                || lower.contains("'a'")
                || lower.contains("\"a\"")
                || lower.contains("'r+'")
                || lower.contains("\"r+\"")))
}

fn inline_runtime_execution(command: &str) -> bool {
    let lower = strip_shell_wrappers(command).to_ascii_lowercase();
    lower.starts_with("php -r ")
        || lower.starts_with("node -e ")
        || lower.starts_with("ruby -e ")
        || lower.starts_with("python -c ")
        || lower.starts_with("python3 -c ")
}

fn scriptish(lower_command: &str) -> bool {
    lower_command.starts_with("python ")
        || lower_command.starts_with("python3 ")
        || lower_command.contains("python -c")
        || lower_command.contains("python3 -c")
        || lower_command.contains("<<")
}

fn command_segments(command: &str) -> Vec<String> {
    command
        .split([';', '\n'])
        .flat_map(|segment| segment.split("&&"))
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .map(str::to_string)
        .collect()
}

fn strip_shell_wrappers(command: &str) -> String {
    let mut parts: Vec<&str> = command.split_whitespace().collect();
    if parts.is_empty() {
        return String::new();
    }
    while let Some(first) = parts.first().copied() {
        if first == "env" || first == "command" {
            parts.remove(0);
            continue;
        }
        if first.contains('=') && !first.starts_with('-') {
            parts.remove(0);
            continue;
        }
        if first == "timeout" && parts.len() >= 2 {
            parts.drain(0..2);
            continue;
        }
        break;
    }
    parts.join(" ")
}

fn strip_harmless_shell_prefixes(command: &str) -> String {
    let mut text = strip_shell_wrappers(command);
    loop {
        let trimmed = text.trim_start();
        if let Some(rest) = trimmed.strip_prefix("pwd &&") {
            text = rest.trim_start().to_string();
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("cd ") {
            if let Some((_, after)) = rest.split_once("&&") {
                text = after.trim_start().to_string();
                continue;
            }
        }
        return trimmed.to_string();
    }
}

fn has_output_redirection(command: &str) -> bool {
    command.contains(">>") || command.contains(" >")
}

fn simple_shell_read_like(command: &str) -> bool {
    command_segments(command).iter().any(|segment| {
        let lower = strip_harmless_shell_prefixes(segment).to_ascii_lowercase();
        [
            "cat ", "head ", "tail ", "nl ", "sed -n", "ls ", "find ", "wc ",
        ]
        .iter()
        .any(|prefix| lower.starts_with(prefix))
    }) || simple_shell_search_like(command)
}

fn simple_shell_search_like(command: &str) -> bool {
    command_segments(command).iter().any(|segment| {
        let lower = strip_harmless_shell_prefixes(segment).to_ascii_lowercase();
        lower.starts_with("rg ")
            || lower.starts_with("grep ")
            || lower.starts_with("git grep ")
            || lower.contains(" xargs grep ")
            || lower.contains(" xargs rg ")
    })
}

fn shell_symbol_search_like(command: &str) -> bool {
    let Some(pattern) = shell_search_pattern(command) else {
        return false;
    };
    symbol_like_pattern(&pattern) && command_has_source_scope(command)
}

const SOURCE_EXTENSIONS: &[&str] = &[
    ".c", ".h", ".cc", ".cpp", ".hpp", ".go", ".java", ".cs", ".php", ".scala", ".rs",
];

fn builtin_tool_for_shell_command(command: &str) -> RecommendedTool {
    let lower = command.to_ascii_lowercase();
    if lower.contains("rg ") || lower.contains("grep ") || lower.contains("git grep ") {
        RecommendedTool::GrepSearch
    } else if lower.contains("ls ") || lower.contains("find ") {
        RecommendedTool::ListDirectory
    } else {
        RecommendedTool::ReadFile
    }
}

fn command_has_source_scope(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    lower.contains(" src")
        || lower.contains("/src")
        || lower.contains("grep -r")
        || lower.contains("rg ")
        || SOURCE_EXTENSIONS
            .iter()
            .any(|ext| lower.contains(&format!("*{ext}")))
}

fn shell_search_pattern(command: &str) -> Option<String> {
    let mut tokens = command.split_whitespace().peekable();
    while let Some(token) = tokens.next() {
        let cmd = token.rsplit('/').next().unwrap_or(token);
        if matches!(cmd, "rg" | "grep") {
            for next in tokens.by_ref() {
                if next.starts_with('-') {
                    continue;
                }
                return Some(next.trim_matches(['"', '\'']).to_string());
            }
        }
    }
    None
}

fn extension(path: &str) -> String {
    PathLike(path)
        .extension()
        .map(str::to_string)
        .unwrap_or_default()
}

struct PathLike<'a>(&'a str);

impl<'a> PathLike<'a> {
    fn extension(&self) -> Option<&'a str> {
        self.0.rsplit_once('.').map(|(_, ext)| ext)
    }
}

fn is_source_like_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    if lower.is_empty() {
        return true;
    }
    !matches!(classify_path_or_glob(&lower), StaticTextTarget::TextLike)
}

fn symbol_like_pattern(pattern: &str) -> bool {
    let trimmed = pattern.trim_matches(['"', '\'']);
    if trimmed.contains("::") || trimmed.contains("->") || trimmed.contains('.') {
        return true;
    }
    let alnum: String = trimmed
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
        .collect();
    if alnum.len() < 3 {
        return false;
    }
    let has_lower = alnum.chars().any(|ch| ch.is_ascii_lowercase());
    let has_upper = alnum.chars().any(|ch| ch.is_ascii_uppercase());
    has_upper || (has_lower && alnum.contains('_'))
}

fn literal_like_pattern(pattern: &str) -> bool {
    let lower = pattern.to_ascii_lowercase();
    lower.contains(' ')
        || lower.contains('=')
        || lower.contains(':')
        || lower.contains('"')
        || lower.contains('\'')
        || lower.contains("error")
        || lower.contains("assert")
}

fn overlaps_recent_bifrost_miss(needle: &str, exchanges: &[ToolExchange]) -> bool {
    overlaps_recent_bifrost_result(needle, exchanges, true)
}

fn overlaps_recent_bifrost_hit(needle: &str, exchanges: &[ToolExchange]) -> bool {
    overlaps_recent_bifrost_result(needle, exchanges, false)
}

fn overlaps_recent_bifrost_result(needle: &str, exchanges: &[ToolExchange], miss: bool) -> bool {
    let needle = needle.trim();
    if needle.len() < 3 {
        return false;
    }
    exchanges.iter().rev().take(8).any(|exchange| {
        is_bifrost_tool(&exchange.tool_name)
            && looks_like_bifrost_miss(&exchange.result) == miss
            && (exchange.arguments.contains(needle) || exchange.result.contains(needle))
    })
}

fn recent_tool_success(exchanges: &[ToolExchange], names: &[&str]) -> bool {
    exchanges.iter().rev().take(5).any(|exchange| {
        names.contains(&exchange.tool_name.as_str()) && !looks_like_failure(&exchange.result)
    })
}

fn recent_edit_or_build_failure(exchanges: &[ToolExchange]) -> bool {
    exchanges.iter().rev().take(5).any(|exchange| {
        matches!(
            exchange.tool_name.as_str(),
            "edit" | "write_file" | "run_shell_command"
        ) && looks_like_failure(&exchange.result)
    })
}

fn looks_like_failure(result: &str) -> bool {
    let lower = result.to_ascii_lowercase();
    lower.starts_with("error:")
        || lower.starts_with("internal error:")
        || lower.contains("failed")
        || lower.contains("failure")
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

    fn text_output_json(
        reason: &str,
        intent: &str,
        fit: &str,
        exception: &str,
        decision: &str,
        tool: &str,
    ) -> String {
        let action = match (decision, tool) {
            ("allow_text", _) => "allow_original",
            (_, "search_symbols") => "use_search_symbols",
            (_, "scan_usages") => "use_scan_usages",
            (_, "get_summaries") => "use_get_summaries",
            (_, "get_symbol_sources") => "use_get_symbol_sources",
            _ => "allow_original",
        };
        json!({
            "reason": reason,
            "action": action,
            "intent": intent,
            "bifrost_fit": fit,
            "allow_exception": exception,
            "evidence": {
                "symbol_tokens": ["Foo"],
                "same_token_or_path_bifrost_miss": false,
                "same_path_recent_edit_or_write": false,
                "same_path_recent_bifrost_hit": false,
                "exact_text_or_regex_needed": false
            },
            "suggested_args": if tool == "none" { json!({}) } else { json!({"patterns":["Foo"]}) },
            "confidence": "high",
        })
        .to_string()
    }

    fn shell_output_json(
        reason: &str,
        intent: &str,
        shell_required: bool,
        builtin_preserves: bool,
        fit: &str,
        exception: &str,
        replacement_class: &str,
        decision: &str,
        tool: &str,
    ) -> String {
        let action = match (decision, tool) {
            ("allow_shell", _) => "allow_original",
            (_, "read_file") => "use_read_file",
            (_, "grep_search") => "use_grep_search",
            (_, "list_directory") => "use_list_directory",
            (_, "edit") => "use_edit",
            (_, "write_file") => "use_write_file",
            (_, "search_symbols") => "use_search_symbols",
            (_, "scan_usages") => "use_scan_usages",
            (_, "get_summaries") => "use_get_summaries",
            (_, "get_symbol_sources") => "use_get_symbol_sources",
            _ => "allow_original",
        };
        json!({
            "reason": reason,
            "action": action,
            "intent": intent,
            "shell_semantics_required": shell_required,
            "builtin_preserves_intent": builtin_preserves,
            "bifrost_fit": fit,
            "allow_exception": exception,
            "replacement_class": replacement_class,
            "suggested_args": if tool == "none" { json!({}) } else { json!({"file_path":"src/lib.rs"}) },
            "confidence": "high",
        })
        .to_string()
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
    fn request_schema_puts_reason_before_action() {
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({"pattern": "Foo"}),
            messages: vec![ChatMessage::user("Find Foo")],
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };
        let body = build_request_body("deepseek/deepseek-v4-flash", &context).unwrap();
        let reason_pos = body.find(r#""reason""#).unwrap();
        let action_pos = body.find(r#""action""#).unwrap();
        assert!(reason_pos < action_pos, "{body}");
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
        assert!(body.contains("broad source/test glob or repo-wide scope"));
        assert!(body.contains("Do not allow text merely because the agent wants to double-check"));
        assert!(body.contains("Keep reason concise"));
    }

    #[test]
    fn shell_request_schema_puts_reason_before_action() {
        let context = GateContext {
            tool_name: "run_shell_command".to_string(),
            args: json!({"command": "cat src/lib.rs"}),
            messages: vec![ChatMessage::user("Read lib.rs")],
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };
        let body = build_shell_request_body("deepseek/deepseek-v4-flash", &context).unwrap();
        let reason_pos = body.find(r#""reason""#).unwrap();
        let action_pos = body.find(r#""action""#).unwrap();
        assert!(reason_pos < action_pos, "{body}");
        assert!(body.contains("shell routing classifier"));
        assert!(body.contains("When uncertain between builtin and shell, allow shell"));
        assert!(body.contains("Keep reason concise"));
        assert!(body.contains("replacement_class"));
    }

    #[test]
    fn gate_message_explains_retry_override() {
        let output = GateClassifierOutput {
            reason: "GATE_TO_SYMBOL_TOOL because Foo looks like a declaration.".to_string(),
            intent: TextIntent::SymbolDefinitionLookup,
            bifrost_fit: BifrostFit::SameOrMoreDirect,
            allow_exception: TextAllowException::None,
            evidence: TextEvidence {
                symbol_tokens: vec!["Foo".to_string()],
                same_token_or_path_bifrost_miss: false,
                same_path_recent_edit_or_write: false,
                same_path_recent_bifrost_hit: false,
                exact_text_or_regex_needed: false,
            },
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
            intent: ShellIntent::OrdinaryFileRead,
            shell_semantics_required: false,
            builtin_preserves_intent: true,
            bifrost_fit: BifrostFit::NotApplicable,
            allow_exception: ShellAllowException::None,
            replacement_class: ShellReplacementClass::UseBuiltinInspection,
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
        assert!(message.contains("Suggested args:"));
    }

    #[test]
    fn parses_classifier_content() {
        let envelope = json!({
            "choices": [{
                "message": {
                    "content": text_output_json("GATE_TO_SYMBOL_TOOL because symbol lookup", "symbol_definition_lookup", "same_or_more_direct", "none", "gate_to_symbol_tool", "search_symbols")
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
                    "content": shell_output_json("USE_BUILTIN_TOOL because cat is an ordinary file read.", "ordinary_file_read", false, true, "not_applicable", "none", "use_builtin_inspection", "use_builtin_tool", "read_file")
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
                    "content": shell_output_json("USE_BIFROST_TOOL because this grep is looking for callers.", "symbol_reference_lookup", false, false, "same_or_more_direct", "none", "use_bifrost_symbol", "use_bifrost_tool", "scan_usages")
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
                    "content": shell_output_json("ALLOW_SHELL because cargo test needs CLI semantics.", "build_test_git_package_or_project_cli", true, false, "not_applicable", "build_test_git_package_or_project_cli", "allow_shell_shell_semantics", "allow_shell", "read_file")
                }
            }]
        });
        let parsed = parse_shell_openrouter_response(&envelope.to_string()).unwrap();
        assert_eq!(parsed.decision, ShellClassifierDecision::AllowShell);
        assert_eq!(parsed.recommended_tool, RecommendedTool::None);
        assert_eq!(parsed.suggested_args, json!({}));
    }

    #[test]
    fn rendered_context_includes_compact_bifrost_miss_evidence() {
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
        assert_eq!(
            rendered
                .pointer("/compact_evidence/same_token_bifrost_miss")
                .and_then(Value::as_bool),
            Some(true)
        );
    }

    #[test]
    fn parses_get_symbol_sources_recommendation() {
        let envelope = json!({
            "choices": [{
                "message": {
                    "content": text_output_json("GATE_TO_SYMBOL_TOOL because known symbols need implementation source.", "known_symbol_source", "same_or_more_direct", "none", "gate_to_symbol_tool", "get_symbol_sources")
                }
            }]
        });
        let parsed = parse_openrouter_response(&envelope.to_string()).unwrap();
        assert_eq!(parsed.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(parsed.recommended_tool, RecommendedTool::GetSymbolSources);
    }

    #[test]
    fn route_action_is_operational_source_of_truth() {
        let envelope = json!({
            "choices": [{
                "message": {
                    "content": text_output_json("GATE_TO_SYMBOL_TOOL because this grep is looking for callers.", "symbol_reference_lookup", "not_applicable", "non_source_text", "gate_to_symbol_tool", "scan_usages")
                }
            }]
        });
        let parsed = parse_openrouter_response(&envelope.to_string()).unwrap();
        assert_eq!(parsed.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(parsed.recommended_tool, RecommendedTool::ScanUsages);
    }

    #[test]
    fn reason_only_bifrost_claim_does_not_override_allow_text_decision() {
        let envelope = json!({
            "choices": [{
                "message": {
                    "content": text_output_json("ALLOW_TEXT because exact regex semantics are needed, although Bifrost might find related symbols.", "literal_or_regex_search", "unknown", "exact_literal_or_regex", "allow_text", "search_symbols")
                }
            }]
        });
        let parsed = parse_openrouter_response(&envelope.to_string()).unwrap();
        assert_eq!(parsed.decision, GateClassifierDecision::AllowText);
        assert_eq!(parsed.recommended_tool, RecommendedTool::None);
    }

    #[test]
    fn low_confidence_text_route_fails_open() {
        let envelope = json!({
            "choices": [{
                "message": {
                    "content": json!({
                        "reason": "GATE_TO_SYMBOL_TOOL because source should use symbols.",
                        "action": "use_search_symbols",
                        "intent": "symbol_definition_lookup",
                        "bifrost_fit": "same_or_more_direct",
                        "allow_exception": "none",
                        "evidence": {
                            "symbol_tokens": ["Foo"],
                            "same_token_or_path_bifrost_miss": false,
                            "same_path_recent_edit_or_write": false,
                            "same_path_recent_bifrost_hit": false,
                            "exact_text_or_regex_needed": false
                        },
                        "suggested_args": {"patterns":["Foo"]},
                        "confidence": "low"
                    }).to_string()
                }
            }]
        });
        let parsed = parse_openrouter_response(&envelope.to_string()).unwrap();
        assert_eq!(parsed.decision, GateClassifierDecision::AllowText);
        assert_eq!(parsed.recommended_tool, RecommendedTool::None);
    }

    #[test]
    fn localized_allow_text_reason_stays_allowed() {
        let envelope = json!({
            "choices": [{
                "message": {
                    "content": text_output_json("ALLOW_TEXT because this reads exact localized lines in a known source file after symbol discovery.", "exact_text_or_localized_read", "same_or_more_direct", "localized_or_sequential_read", "allow_text", "none")
                }
            }]
        });
        let parsed = parse_openrouter_response(&envelope.to_string()).unwrap();
        assert_eq!(parsed.decision, GateClassifierDecision::AllowText);
        assert_eq!(parsed.recommended_tool, RecommendedTool::None);
    }

    #[test]
    fn broad_symbol_grep_allow_without_supported_exception_is_gated() {
        let mut output = GateClassifierOutput {
            reason: "ALLOW_TEXT because regex can find the identifier.".to_string(),
            intent: TextIntent::SymbolDefinitionLookup,
            bifrost_fit: BifrostFit::Unknown,
            allow_exception: TextAllowException::ExactLiteralOrRegex,
            evidence: TextEvidence {
                symbol_tokens: vec!["RichEditBoxDefaultLineEnding".to_string()],
                same_token_or_path_bifrost_miss: false,
                same_path_recent_edit_or_write: false,
                same_path_recent_bifrost_hit: false,
                exact_text_or_regex_needed: true,
            },
            decision: GateClassifierDecision::AllowText,
            recommended_tool: RecommendedTool::None,
            suggested_args: json!({}),
            confidence: GateConfidence::High,
        };
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({"pattern": "RichEditBoxDefaultLineEnding", "path": "src", "glob": "**/*.cs"}),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(output.recommended_tool, RecommendedTool::SearchSymbols);
    }

    #[test]
    fn exact_file_symbol_grep_allow_stays_allowed() {
        let mut output = GateClassifierOutput {
            reason: "ALLOW_TEXT because exact-file grep is localized.".to_string(),
            intent: TextIntent::SymbolDefinitionLookup,
            bifrost_fit: BifrostFit::Unknown,
            allow_exception: TextAllowException::LocalizedOrSequentialRead,
            evidence: TextEvidence::default(),
            decision: GateClassifierDecision::AllowText,
            recommended_tool: RecommendedTool::None,
            suggested_args: json!({}),
            confidence: GateConfidence::High,
        };
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({"pattern": "RichEditBoxDefaultLineEnding", "path": ".", "glob": "src/Notepads/App.xaml.cs"}),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::AllowText);
        assert_eq!(output.recommended_tool, RecommendedTool::None);
    }

    #[test]
    fn rendered_context_uses_pending_call_and_features() {
        let context = GateContext {
            tool_name: "read_file".to_string(),
            args: json!({"file_path": "src/lib.rs", "offset": 10, "limit": 20}),
            messages: vec![ChatMessage::user("Read lib.rs")],
            tools: Vec::new(),
            tool_exchanges: vec![exchange("get_symbol_sources"), exchange("edit")],
        };

        let rendered: Value = serde_json::from_str(&render_context(&context).unwrap()).unwrap();
        assert!(rendered.get("pending_text_call").is_none());
        assert_eq!(
            rendered
                .pointer("/pending_call/tool")
                .and_then(Value::as_str),
            Some("read_file")
        );
        assert_eq!(
            rendered
                .pointer("/pending_call_features/has_offset")
                .and_then(Value::as_bool),
            Some(true)
        );
        assert!(rendered.pointer("/compact_evidence").is_some());
        assert!(rendered.pointer("/bifrost_tool_purposes").is_some());
        assert!(rendered.pointer("/prior_tool_counts").is_none());
    }

    #[test]
    fn grep_features_include_scope_granularity() {
        let exact = grep_search_features(
            &json!({"path": "src/Foo.cs", "pattern": "FlowDirection"}),
            &[exchange("read_file")],
        );
        assert_eq!(
            exact.get("scope_granularity").and_then(Value::as_str),
            Some("exact_file")
        );
        assert_eq!(
            exact.get("exact_file_scope").and_then(Value::as_bool),
            Some(true)
        );

        let glob_exact = grep_search_features(
            &json!({"path": ".", "glob": "src/win/udp.c", "pattern": "uv_udp_try_send"}),
            &[],
        );
        assert_eq!(
            glob_exact
                .get("scope_granularity")
                .and_then(Value::as_str),
            Some("exact_file")
        );

        let broad =
            grep_search_features(&json!({"glob": "**/*.c", "pattern": "uv_pipe_bind2"}), &[]);
        assert_eq!(
            broad.get("scope_granularity").and_then(Value::as_str),
            Some("source_glob")
        );
        assert_eq!(
            broad
                .get("broad_source_or_test_scope")
                .and_then(Value::as_bool),
            Some(true)
        );
    }

    #[test]
    fn static_text_route_allows_exact_localized_reads() {
        assert!(matches!(
            static_text_route(
                "read_file",
                &json!({"file_path": "src/FileExtension.php"}),
                &[]
            ),
            Some(TextStaticRoute::AllowText("static_exact_localized_read"))
        ));
        assert!(matches!(
            static_text_route(
                "read_file",
                &json!({"file_path": "src/lib.rs", "offset": 10, "limit": 20}),
                &[]
            ),
            Some(TextStaticRoute::AllowText("static_exact_localized_read"))
        ));
    }

    #[test]
    fn static_text_route_leaves_broad_source_symbol_grep_for_classifier() {
        assert!(
            static_text_route(
                "grep_search",
                &json!({"path": ".", "glob": "**/*.[ch]", "pattern": "uv_pipe_bind2"}),
                &[]
            )
            .is_none()
        );
        assert!(
            static_text_route(
                "grep_search",
                &json!({"path": "src", "glob": "*.c", "pattern": "int uv_os_getenv\\("}),
                &[]
            )
            .is_none()
        );
        assert!(matches!(
            static_text_route(
                "grep_search",
                &json!({"path": ".", "glob": "src/win/udp.c", "pattern": "uv_udp_try_send"}),
                &[]
            ),
            Some(TextStaticRoute::AllowText("static_exact_file_grep"))
        ));
        assert!(
            static_text_route(
                "grep_search",
                &json!({"path": "README.md", "pattern": "error: file not found"}),
                &[]
            )
            .is_none()
        );
    }

    #[test]
    fn static_shell_route_handles_project_execution_and_source_search() {
        assert!(matches!(
            static_shell_route(&json!({"command": "composer install --no-interaction"})),
            Some(ShellStaticRoute::AllowShell("static_shell_semantics"))
        ));
        assert!(matches!(
            static_shell_route(&json!({"command": "vendor/bin/phpunit tests/FooTest.php"})),
            Some(ShellStaticRoute::AllowShell("static_shell_semantics"))
        ));
        assert!(matches!(
            static_shell_route(&json!({"command": "./tool/go test -count=1 ./util/osuser"})),
            Some(ShellStaticRoute::AllowShell("static_shell_semantics"))
        ));
        assert!(matches!(
            static_shell_route(
                &json!({"command": "cat build.sbt | grep -E \"project|tests\" | head -30"})
            ),
            Some(ShellStaticRoute::UseBuiltin(
                "static_shell_builtin_inspection",
                RecommendedTool::GrepSearch
            ))
        ));
        assert!(matches!(
            static_shell_route(
                &json!({"command": "grep -RIn \"EnsureCertLoops|TerminateTLS|type TCPPortHandler\" kube ipn . | head -200"})
            ),
            Some(ShellStaticRoute::UseBuiltin(
                "static_shell_builtin_inspection",
                RecommendedTool::GrepSearch
            ))
        ));
    }

    #[test]
    fn static_shell_route_only_allows_obvious_shell_semantics() {
        assert!(matches!(
            static_shell_route(&json!({"command": "cargo test -q"})),
            Some(ShellStaticRoute::AllowShell("static_shell_semantics"))
        ));
        assert!(matches!(
            static_shell_route(&json!({"command": "nl -ba src/main.rs | sed -n '1,80p'"})),
            Some(ShellStaticRoute::UseBuiltin(
                "static_shell_builtin_inspection",
                RecommendedTool::ReadFile
            ))
        ));
        assert!(matches!(
            static_shell_route(&json!({"command": "rg FooService src"})),
            Some(ShellStaticRoute::UseBuiltin(
                "static_shell_builtin_inspection",
                RecommendedTool::GrepSearch
            ))
        ));
    }

    #[test]
    fn static_shell_route_allows_scripted_raw_byte_and_mutation_semantics() {
        assert!(matches!(
            static_shell_route(
                &json!({"command": "python3 - <<'PY'\nfrom pathlib import Path\nprint(repr(Path('src/lib.rs').read_bytes()[:20]))\nPY"})
            ),
            Some(ShellStaticRoute::AllowShell("static_shell_semantics"))
        ));
        assert!(matches!(
            static_shell_route(
                &json!({"command": "python3 - <<'PY'\nfrom pathlib import Path\nPath('src/lib.rs').write_bytes(b'abc')\nPY"})
            ),
            Some(ShellStaticRoute::AllowShell("static_shell_semantics"))
        ));
    }

    #[test]
    fn shell_symbol_search_features_distinguish_path_find() {
        let source_grep = shell_command_features(
            &json!({"command": "grep -RIn --include='*.go' -E 'Foo|Bar' src | head -20"}),
        );
        assert_eq!(
            source_grep
                .get("symbol_search_like")
                .and_then(Value::as_bool),
            Some(true)
        );

        let path_find = shell_command_features(
            &json!({"command": "find . -path '*Volume.java' -o -path '*VolumeTest.java'"}),
        );
        assert_eq!(
            path_find.get("symbol_search_like").and_then(Value::as_bool),
            Some(false)
        );
        assert_eq!(
            path_find.get("simple_read_like").and_then(Value::as_bool),
            Some(true)
        );
    }
}
