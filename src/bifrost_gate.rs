use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use reqwest::StatusCode;
use serde::de::Deserializer;
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
    SymbolUsageLookup,
    BroadSemanticOrientation,
    LiteralOrRegexSearch,
    RegexTextSearch,
    PathOrFilenameSearch,
    PostEditOrValidationVerification,
    Other,
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
    ExternalApiOrAnnotationText,
    CompoundIdiomRegex,
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
#[serde(rename_all = "snake_case")]
pub enum TextPatternClass {
    IdentifierLike,
    SymbolGlob,
    MixedSymbolIdentifiers,
    ExternalApiOrAnnotationNames,
    CompoundCodeIdiomRegex,
    LiteralExact,
    RegexText,
    PathLike,
    NaturalLanguage,
    Mixed,
    Unknown,
}

impl Default for TextPatternClass {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextScopeClass {
    ExactFile,
    NarrowedFileSet,
    DirectoryOrGlob,
    BroadSourceScope,
    MultiFileSourceScope,
    RepositoryWide,
    Unknown,
}

impl Default for TextScopeClass {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BifrostCandidate {
    #[serde(default, deserialize_with = "deserialize_optional_recommended_tool")]
    pub tool: Option<RecommendedTool>,
    #[serde(default = "empty_object")]
    pub args: Value,
}

fn deserialize_optional_recommended_tool<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<RecommendedTool>, D::Error>
where
    D: Deserializer<'de>,
{
    let Some(value) = Option::<Value>::deserialize(deserializer)? else {
        return Ok(None);
    };
    Ok(RecommendedTool::deserialize(value).ok())
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
    pub pattern_class: TextPatternClass,
    #[serde(default)]
    pub scope_class: TextScopeClass,
    #[serde(default)]
    pub bifrost_fit: BifrostFit,
    #[serde(default, alias = "fallback_exception")]
    pub allow_exception: TextAllowException,
    #[serde(default)]
    pub evidence: TextEvidence,
    #[serde(default)]
    pub bifrost_candidate: Option<BifrostCandidate>,
    pub decision: GateClassifierDecision,
    pub recommended_tool: RecommendedTool,
    #[serde(default = "empty_object")]
    pub suggested_args: Value,
    pub confidence: GateConfidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawGateClassifierOutput {
    pub reason: String,
    #[serde(default)]
    pub intent: TextIntent,
    #[serde(default)]
    pub pattern_class: TextPatternClass,
    #[serde(default)]
    pub scope_class: TextScopeClass,
    #[serde(default)]
    pub bifrost_fit: BifrostFit,
    #[serde(default, alias = "fallback_exception")]
    pub allow_exception: TextAllowException,
    #[serde(default)]
    pub evidence: TextEvidence,
    #[serde(default)]
    pub bifrost_candidate: Option<BifrostCandidate>,
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
        let should_gate = text_policy_facts_gate(&raw);
        let recommended_tool = if should_gate {
            raw.bifrost_candidate
                .as_ref()
                .and_then(|candidate| candidate.tool.clone())
                .filter(is_bifrost_recommendation)
                .unwrap_or_else(|| bifrost_tool_for_text_intent(&raw.intent))
        } else {
            RecommendedTool::None
        };
        let decision = if should_gate && is_bifrost_recommendation(&recommended_tool) {
            GateClassifierDecision::GateToSymbolTool
        } else {
            GateClassifierDecision::AllowText
        };
        let suggested_args = raw
            .bifrost_candidate
            .as_ref()
            .map(|candidate| candidate.args.clone())
            .filter(|args| args.as_object().is_some_and(|object| !object.is_empty()))
            .unwrap_or(raw.suggested_args);
        Self {
            reason: raw.reason,
            intent: raw.intent,
            pattern_class: raw.pattern_class,
            scope_class: raw.scope_class,
            bifrost_fit: raw.bifrost_fit,
            allow_exception: raw.allow_exception,
            evidence: raw.evidence,
            bifrost_candidate: raw.bifrost_candidate,
            decision,
            recommended_tool,
            suggested_args,
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

fn text_policy_facts_gate(raw: &RawGateClassifierOutput) -> bool {
    if !matches!(
        raw.confidence,
        GateConfidence::High | GateConfidence::Medium
    ) {
        return false;
    }
    if raw.allow_exception != TextAllowException::None || raw.evidence.exact_text_or_regex_needed {
        return false;
    }
    if !matches!(
        raw.intent,
        TextIntent::SymbolDefinitionLookup
            | TextIntent::SymbolReferenceLookup
            | TextIntent::SymbolUsageLookup
            | TextIntent::KnownSymbolSource
            | TextIntent::BroadSemanticOrientation
    ) {
        return false;
    }
    let same_direct_candidate = raw.bifrost_fit == BifrostFit::SameOrMoreDirect
        && raw.allow_exception == TextAllowException::None
        && !raw.evidence.exact_text_or_regex_needed
        && raw
            .bifrost_candidate
            .as_ref()
            .and_then(|candidate| candidate.tool.as_ref())
            .is_some_and(is_bifrost_recommendation);
    if !matches!(
        raw.pattern_class,
        TextPatternClass::IdentifierLike
            | TextPatternClass::SymbolGlob
            | TextPatternClass::MixedSymbolIdentifiers
    ) && !(raw.pattern_class == TextPatternClass::RegexText && same_direct_candidate)
    {
        return false;
    }
    if !matches!(
        raw.scope_class,
        TextScopeClass::DirectoryOrGlob
            | TextScopeClass::BroadSourceScope
            | TextScopeClass::MultiFileSourceScope
            | TextScopeClass::RepositoryWide
    ) {
        return false;
    }
    raw.bifrost_fit == BifrostFit::SameOrMoreDirect
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
            Ok(mut output) => {
                enforce_shell_classifier_policy(&mut output, &context);
                return Ok(output);
            }
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
        "list_directory" => Some(TextStaticRoute::AllowText("static_list_directory_allow")),
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
        "Return only a valid JSON object matching the schema. No prose, markdown, hidden commentary, action field, or final decision field. Classify the pending read_file/grep_search call as policy facts from pending_call_features and compact_evidence. The system will compute the executable route from those facts."
    } else {
        "You are a policy-fact extractor for routing an AI coding agent's text navigation. You are advising a stronger coding model, not commanding it. Return policy facts only; the system will compute the final routing decision from these facts. Do not include an action field or final decision field. The first field must be reason. The reason should explain the policy facts, not announce an executable route; avoid route labels like GATE_TO_SYMBOL_TOOL or ALLOW_TEXT unless quoting prior context. Use pending_call_features and compact_evidence first, prose excerpts second. Always allow list_directory. For read_file, exact file/path/line/range inspection is allowed text; only broad redundant source reading after Bifrost found exact source is a possible symbol-navigation fact. Before classifying grep_search, determine the object of search. Object = source relationship when the pattern is a textual encoding of source structure: declaration, definition, constructor, call site, member/static/qualified access, reference, caller/callee, implementation, override, or related-test navigation. Set bifrost_fit=same_or_more_direct only when the target is a repo-indexed source-symbol relationship Bifrost can preserve: declaration, definition, implementation source, references, callers, related tests, or broad orientation over known project symbols. Do not set same_or_more_direct merely because tokens are identifier-shaped. If the target is an external/library/framework API name, annotation or attribute name, toolchain-provided symbol, import/package line, or compound regex/idiom whose exact alternation, qualifier, argument literal, or co-occurrence semantics matter, use literal_or_regex_search or regex_text_search facts, bifrost_fit=less_direct or not_applicable, and no Bifrost candidate. In broad source/test scope, route these to Bifrost even when the grep pattern is exact regex; regex exactness is not an allow-text reason if stripping regex syntax leaves recoverable repo source symbols plus source-relationship shape. For regex-like patterns, normalize semantic intent first: ignore wrappers/escapes such as \\b, ^, $, \\(, \\), \\s*, simple groups, and escaping that only protects code punctuation. Call-site shapes include foo\\(, \\bfoo\\s*\\(, .foo\\(, ->foo\\(, ::foo\\(, receiver.foo\\(, and Foo::bar\\(; these are usage/reference searches and should prefer scan_usages unless the task is explicitly about exact emitted text. Object = exact text when the requested target is the characters themselves: wire/schema field, config key, serialized/emitted output, API payload field, DB column, fixture/golden/snapshot, diagnostic/log/UI text, docs, import/include/package lines, build text, assertion text, external annotation/API names, or localized post-edit verification. Allow grep for these even if the token is identifier-shaped and even if the files are source files. Bare identifier guard: do not route a single identifier, lower_snake token, generic word, or word-boundary regex to Bifrost solely because it looks like code. Bare identifiers require source-navigation context, a known existing repo symbol target, or source-relationship shape; otherwise prefer text grep. Concrete code shape is not, by itself, an allow-text reason. First decide whether concrete syntax is the user's target, or merely a way to name a source symbol or source relationship. Treat concrete syntax as Bifrost symbol navigation when it encodes a declaration, definition, constructor, method call, static/member access, reference, caller/callee query, related-test query, or small set of known project symbols over broad source/test scope. Examples include class Foo, function foo, def foo, new Foo(, Foo::, Foo., foo(, and Foo|Bar|Baz when these are project symbols. Treat concrete syntax as allow-text only when the exact characters, literal value, diagnostic text, config/build text, import/include/package line, operator syntax, assertion text, serialization/snapshot text, external API/annotation names, localized post-edit verification text, or compound idiom regex is the object of the search. A test scope does not make a symbol-looking query allow-text. Do not use exact regex as an allow-text reason when stripping regex wrappers leaves recoverable repo source symbols and the search is over broad source/test scope. A previous Bifrost call being skipped, filtered, or marked not_text_navigation_tool is not evidence that Bifrost failed. Bifrost internal/protocol/refresh errors are infrastructure failures to fix, not targeted symbol misses. Fill bifrost_candidate only when a symbol tool would be same-or-more-direct and preserve the exact intent: search_symbols for declarations/definitions, scan_usages for calls/references/related tests, get_summaries for broad orientation, and get_symbol_sources only when exact repo symbol names are already known. When bifrost_candidate is not null, it must contain both `tool` and `args`; `args` must be a JSON object; `tool` must exactly be one of search_symbols, scan_usages, get_summaries, or get_symbol_sources. Keep reason concise. Output only JSON matching the schema."
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
          "intent": {{"type":"string","enum":["exact_text_or_localized_read","sequential_file_read","whole_file_or_top_of_file_orientation","known_symbol_source","symbol_definition_lookup","symbol_reference_lookup","symbol_usage_lookup","broad_semantic_orientation","literal_or_regex_search","regex_text_search","path_or_filename_search","post_edit_or_validation_verification","other","unknown"]}},
          "pattern_class": {{"type":"string","enum":["identifier_like","symbol_glob","mixed_symbol_identifiers","external_api_or_annotation_names","compound_code_idiom_regex","literal_exact","regex_text","path_like","natural_language","mixed","unknown"]}},
          "scope_class": {{"type":"string","enum":["exact_file","narrowed_file_set","directory_or_glob","broad_source_scope","multi_file_source_scope","repository_wide","unknown"]}},
          "bifrost_fit": {{"type":"string","enum":["same_or_more_direct","less_direct","not_applicable","unknown"]}},
          "fallback_exception": {{"type":"string","enum":["none","non_source_text","exact_literal_or_regex","external_api_or_annotation_text","compound_idiom_regex","localized_or_sequential_read","whole_file_or_top_of_file_orientation","test_header_macro_or_entrypoint_context","post_edit_or_build_or_test_verification","same_token_or_path_bifrost_miss","uncertain"]}},
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
          "bifrost_candidate": {{
            "type":["object","null"],
            "additionalProperties": false,
            "properties": {{
              "tool": {{"type":"string","enum":["search_symbols","scan_usages","get_summaries","get_symbol_sources","none"]}},
              "args": {{"type":"object"}}
            }},
            "required": ["tool","args"]
          }},
          "suggested_args": {{"type":"object"}},
          "confidence": {{"type":"string","enum":["low","medium","high"]}}
        }},
        "required": ["reason","intent","pattern_class","scope_class","bifrost_fit","fallback_exception","evidence","bifrost_candidate","suggested_args","confidence"]
      }}
    }}
  }},
  "temperature": 0,
  "max_tokens": 4096
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
        "Return only a valid JSON object matching the schema. No prose, markdown, or hidden commentary. Classify by purpose from pending_call_features and compact_evidence. Use builtins for ordinary file reads/searches/listings when they preserve intent. Use Bifrost only for clear source-code symbol discovery. Allow shell only for concrete shell semantics such as build/test/git/package commands, env/path probes, compiler/preprocessor transformations, raw bytes, mutation, generated artifacts, or meaningful pipeline behavior. If uncertain, choose allow_original with confidence low."
    } else {
        "You are a shell routing classifier for an AI coding agent. You are advising a stronger coding model, not commanding it. Classify by purpose, not by syntax. Use pending_call_features and compact_evidence first, prose excerpts second. Distinguish grep over file contents from grep filtering file paths: rg, git grep, grep -R, and find ... | xargs grep search contents; find ... | grep PATTERN without xargs/-exec grep filters paths. Allow shell when shell semantics are materially part of the task: build/test/git/package/project CLI, env/permission/path probing, compiler/preprocessor transformations, raw bytes or hidden whitespace, generated artifacts, command substitution, mutation/write/delete behavior, path-only filtering, or a pipeline whose transformation or exit behavior matters. Do not treat a pipeline as shell semantics when it only limits, counts, sorts, or pretty-prints inspection output. Use builtins when the command only reads, searches, lists, or prints bounded ranges, and shell syntax is only being used to limit, count, or pretty-print inspection output. A filename/path search with find, ls, or path globs is path inspection, not Bifrost and not content grep_search. For grep/rg/git-grep/find-xargs-grep commands over broad source contents, choose Bifrost when the pattern is bare source-symbol discovery: declarations, definitions, call sites, references, related tests, or broad code orientation. Choose builtin grep_search for exact text, exact-file scope, config/docs/log/error/header/import/wire/literal searches, counts, or paired literal token checks. Tie breakers: broad source/root symbol or declaration/call/reference search uses Bifrost; bounded config/test/exact-file/concrete-code-shape content search uses builtin; real execution behavior uses shell. Do not allow shell unless concrete shell execution semantics are required. A previous Bifrost call being skipped, filtered, or marked not_text_navigation_tool is not evidence that Bifrost failed. When uncertain between builtin and shell, allow shell only if you can identify a concrete shell-specific semantic that may matter. The action field is the only route the gate will execute. If a builtin or Bifrost tool is preferred, choose that action; never choose allow_original while recommending another tool. The reason field must commit to one of these forms: 'ALLOW_SHELL because ...', 'USE_BUILTIN_TOOL because ...', or 'USE_BIFROST_TOOL because ...'. Keep reason concise. Output only JSON matching the schema."
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
    if !output.suggested_args.is_object() {
        output.suggested_args = json!({});
    }
    if let Some(candidate) = output.bifrost_candidate.as_mut() {
        if !candidate.args.is_object() {
            candidate.args = json!({});
        }
        if !candidate
            .tool
            .as_ref()
            .is_some_and(is_bifrost_recommendation)
        {
            output.bifrost_candidate = None;
        }
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
    if context.tool_name != "grep_search" {
        return;
    }
    let pattern = context
        .args
        .get("pattern")
        .and_then(Value::as_str)
        .unwrap_or("");
    let glob = context
        .args
        .get("glob")
        .and_then(Value::as_str)
        .unwrap_or("");
    let path = context
        .args
        .get("path")
        .and_then(Value::as_str)
        .unwrap_or("");
    let file_path = context
        .args
        .get("file_path")
        .and_then(Value::as_str)
        .unwrap_or("");

    let real_bifrost_miss =
        recent_bifrost_miss_for_grep(pattern, glob, path, file_path, &context.tool_exchanges)
            && !very_strong_deterministic_symbol_navigation(pattern, glob, path, file_path);
    let same_symbol_bifrost_hit =
        same_symbol_bifrost_hit_for_grep(pattern, &context.tool_exchanges);
    let lower_snake_text_probe = lower_snake_literal_token_count(pattern) >= 2
        && !identifier_tokens(pattern)
            .iter()
            .any(|token| short_prefixed_source_symbol_like(token))
        && !declaration_search_pattern(pattern)
        && !symbol_call_or_member(pattern)
        && !source_static_or_member_access(pattern)
        && !source_call_navigation_pattern(pattern)
        && !same_symbol_bifrost_hit;
    let repair_tool =
        deterministic_source_symbol_repair_tool(pattern, glob, path, file_path, context);
    if text_negative_facts_veto(output)
        && !very_strong_deterministic_symbol_navigation(pattern, glob, path, file_path)
        && repair_tool.is_none()
        && !same_direct_bifrost_candidate(output)
    {
        output.decision = GateClassifierDecision::AllowText;
        output.recommended_tool = RecommendedTool::None;
        output.suggested_args = json!({});
        output.reason = format!(
            "ALLOW_TEXT because `{}` has exact-text or non-Bifrost policy facts.",
            truncate_to(pattern, 120)
        );
        return;
    }
    if (repair_tool.is_none()
        && repair_blocked_text_grep(pattern, glob, path, file_path, context)
        && !same_symbol_bifrost_hit)
        || ((output.allow_exception == TextAllowException::NonSourceText
            || (matches!(output.intent, TextIntent::ExactTextOrLocalizedRead)
                && output.bifrost_fit == BifrostFit::NotApplicable
                && output.bifrost_candidate.is_none()))
            && exact_bare_identifier_text_exception(
                pattern,
                output.evidence.exact_text_or_regex_needed,
            ))
        || lower_snake_text_probe
        || (repair_tool.is_none()
            && !same_direct_bifrost_candidate(output)
            && strong_text_grep_allow_exception(
                pattern,
                glob,
                path,
                file_path,
                &output.bifrost_fit,
                output.evidence.exact_text_or_regex_needed,
                context,
            ))
        || exact_error_constant_search(pattern, path, file_path)
        || real_bifrost_miss
    {
        output.decision = GateClassifierDecision::AllowText;
        output.recommended_tool = RecommendedTool::None;
        output.suggested_args = json!({});
        output.reason = format!(
            "ALLOW_TEXT because `{}` is an exact text/regex search whose intent would be lost by symbol routing.",
            truncate_to(pattern, 120)
        );
        return;
    }

    if let Some((tool, relation)) = repair_tool {
        output.decision = GateClassifierDecision::GateToSymbolTool;
        output.recommended_tool = tool;
        output.suggested_args = json!({});
        output.reason = format!(
            "GATE_TO_SYMBOL_TOOL because broad source/test grep for `{}` is source-symbol {relation} navigation, not exact text/value search.",
            truncate_to(pattern, 120)
        );
        output.confidence = GateConfidence::High;
        return;
    }

    if same_direct_bifrost_candidate(output)
        && output.allow_exception == TextAllowException::None
        && !output.evidence.exact_text_or_regex_needed
    {
        output.decision = GateClassifierDecision::GateToSymbolTool;
        if let Some(tool) = output
            .bifrost_candidate
            .as_ref()
            .and_then(|candidate| candidate.tool.clone())
        {
            output.recommended_tool = tool;
        }
        output.suggested_args = output
            .bifrost_candidate
            .as_ref()
            .map(|candidate| candidate.args.clone())
            .unwrap_or_else(|| json!({}));
        return;
    }

    if output.decision == GateClassifierDecision::AllowText {
        return;
    }
    if !is_bifrost_recommendation(&output.recommended_tool) {
        output.recommended_tool =
            deterministic_text_tool_for_grep(&output.intent, pattern, glob, path);
    }
    if !is_bifrost_recommendation(&output.recommended_tool) {
        output.decision = GateClassifierDecision::AllowText;
        output.recommended_tool = RecommendedTool::None;
        output.suggested_args = json!({});
    }
}

fn same_direct_bifrost_candidate(output: &GateClassifierOutput) -> bool {
    output.bifrost_fit == BifrostFit::SameOrMoreDirect
        && output.allow_exception == TextAllowException::None
        && !output.evidence.exact_text_or_regex_needed
        && output
            .bifrost_candidate
            .as_ref()
            .and_then(|candidate| candidate.tool.as_ref())
            .is_some_and(is_bifrost_recommendation)
}

fn text_negative_facts_veto(output: &GateClassifierOutput) -> bool {
    output.allow_exception != TextAllowException::None
        || output.evidence.exact_text_or_regex_needed
        || matches!(
            output.bifrost_fit,
            BifrostFit::LessDirect | BifrostFit::NotApplicable
        )
        || matches!(
            output.pattern_class,
            TextPatternClass::ExternalApiOrAnnotationNames
                | TextPatternClass::CompoundCodeIdiomRegex
                | TextPatternClass::LiteralExact
                | TextPatternClass::RegexText
        )
}

fn very_strong_deterministic_symbol_navigation(
    pattern: &str,
    glob: &str,
    path: &str,
    file_path: &str,
) -> bool {
    bifrost_navigation_scope(glob, path, file_path)
        && !compound_code_idiom_regex(pattern)
        && !external_api_or_annotation_names(pattern)
        && (declaration_search_pattern(pattern)
            || constructor_or_static_call_usage_pattern(pattern)
            || source_static_or_member_access(pattern)
            || source_call_navigation_pattern(pattern)
            || source_relationship_call_site_pattern(pattern)
            || source_symbol_family_navigation(pattern))
}

fn source_symbol_family_navigation(pattern: &str) -> bool {
    if natural_language_or_path_literal(pattern) {
        return false;
    }
    let symbols = recoverable_source_symbols(pattern);
    if symbols.len() == 1 {
        return high_confidence_source_symbol(&symbols[0]);
    }
    symbols.len() >= 2
        && symbols
            .iter()
            .all(|symbol| short_prefixed_source_symbol_like(symbol))
}

fn must_allow_text_grep(pattern: &str, glob: &str, path: &str, file_path: &str) -> bool {
    if grep_scope_granularity(glob, path, file_path) == "exact_file" {
        return true;
    }
    if filename_or_resource_regex(pattern)
        || inline_regex_flag_text_search(pattern)
        || compound_code_idiom_regex(pattern)
        || external_api_or_annotation_names(pattern)
    {
        return true;
    }
    matches!(
        grep_pattern_kind(pattern, glob, path),
        GrepPatternKind::ExactImportOrInclude
            | GrepPatternKind::ErrorMessageOrLogText
            | GrepPatternKind::WireKeyOrSerializationKey
            | GrepPatternKind::HeaderOrProtocolLiteral
            | GrepPatternKind::NumericOrCodepointLiteral
            | GrepPatternKind::TestTextSearch
            | GrepPatternKind::PackageDeclarationLiteral
            | GrepPatternKind::CodeIdiom
    )
}

fn strong_text_grep_allow_exception(
    pattern: &str,
    glob: &str,
    path: &str,
    file_path: &str,
    bifrost_fit: &BifrostFit,
    exact_text_or_regex_needed: bool,
    context: &GateContext,
) -> bool {
    let kind = grep_pattern_kind(pattern, glob, path);
    if package_declaration_literal(pattern)
        || uppercase_error_constant_set(pattern)
        || mixed_error_constant_text_search(pattern)
        || external_api_word_boundary_text_search(pattern)
        || external_api_or_annotation_names(pattern)
        || compound_code_idiom_regex(pattern)
        || config_or_build_text_scope(glob, path, file_path)
        || diagnostic_or_status_constant_text(pattern)
    {
        return true;
    }
    if post_edit_scope_verification(pattern, glob, path, file_path, &context.tool_exchanges) {
        return true;
    }
    if natural_language_or_path_literal(pattern) && !declaration_search_pattern(pattern) {
        return true;
    }
    if import_api_or_operator_regex_text(pattern) && !declaration_search_pattern(pattern) {
        return true;
    }
    if exact_member_chain_or_call_text(pattern) {
        return true;
    }
    if assertion_or_serialization_text(pattern)
        && (narrow_component_scope(glob, path, file_path)
            || bounded_test_scope(glob, path, file_path)
            || bifrost_fit == &BifrostFit::NotApplicable
            || exact_text_or_regex_needed)
    {
        return true;
    }
    if assertion_or_build_string_regex_text(pattern) {
        return true;
    }
    if regex_text_semantics_material(pattern)
        && narrow_component_scope(glob, path, file_path)
        && !declaration_search_pattern(pattern)
    {
        return true;
    }
    if bounded_test_scope(glob, path, file_path)
        && (matches!(
            kind,
            GrepPatternKind::TestTextSearch
                | GrepPatternKind::CodeIdiom
                | GrepPatternKind::ExactMemberChainOrCallText
                | GrepPatternKind::AssertionOrSerializationText
                | GrepPatternKind::NaturalLanguageOrPathLiteral
                | GrepPatternKind::UppercaseErrorConstantSet
        ) || (pattern.contains("\\(") && !declaration_search_pattern(pattern)))
    {
        return true;
    }
    if test_only_literal_scope(glob, path, file_path)
        && (lower_snake_literal_token_count(pattern) > 0
            || uppercase_constant_token_count(pattern) > 0
            || matches!(
                kind,
                GrepPatternKind::TestTextSearch
                    | GrepPatternKind::CodeIdiom
                    | GrepPatternKind::HeaderOrProtocolLiteral
                    | GrepPatternKind::NumericOrCodepointLiteral
                    | GrepPatternKind::UppercaseErrorConstantSet
            ))
        && !declaration_search_pattern(pattern)
        && !symbol_call_or_member(pattern)
    {
        return true;
    }
    if bounded_test_scope(glob, path, file_path) && lower_snake_literal_token_count(pattern) > 0 {
        return true;
    }
    if lower_snake_literal_token_count(pattern) >= 2
        && !identifier_tokens(pattern)
            .iter()
            .any(|token| short_prefixed_source_symbol_like(token))
        && !declaration_search_pattern(pattern)
        && !symbol_call_or_member(pattern)
        && !source_static_or_member_access(pattern)
        && !source_call_navigation_pattern(pattern)
    {
        return true;
    }
    if narrow_component_scope(glob, path, file_path)
        && matches!(
            kind,
            GrepPatternKind::ExactMemberChainOrCallText
                | GrepPatternKind::AssertionOrSerializationText
                | GrepPatternKind::NaturalLanguageOrPathLiteral
                | GrepPatternKind::UppercaseErrorConstantSet
        )
    {
        return true;
    }
    if narrow_component_scope(glob, path, file_path)
        && lower_snake_literal_token_count(pattern) >= 2
    {
        return true;
    }
    if narrow_component_scope(glob, path, file_path)
        && identifier_tokens(pattern).len() == 1
        && !pattern.contains('|')
        && !pattern.contains(".*")
        && !pattern.contains("\\(")
    {
        return !identifier_tokens(pattern)
            .first()
            .is_some_and(|token| high_confidence_source_symbol(token));
    }
    false
}

fn recent_bifrost_miss_for_grep(
    pattern: &str,
    glob: &str,
    path: &str,
    file_path: &str,
    exchanges: &[ToolExchange],
) -> bool {
    let scope_target = grep_scope_target(glob, path, file_path);
    if overlaps_recent_bifrost_miss(pattern, exchanges)
        || overlaps_recent_bifrost_miss(&scope_target, exchanges)
    {
        return true;
    }

    let symbols = recoverable_source_symbols(pattern);
    if symbols.is_empty() || declaration_search_pattern(pattern) {
        return false;
    }
    let token_misses = symbols
        .iter()
        .filter(|symbol| overlaps_recent_bifrost_miss(symbol, exchanges))
        .count();
    let token_hits = symbols
        .iter()
        .filter(|symbol| overlaps_recent_bifrost_hit(symbol, exchanges))
        .count();
    token_misses > 0 && token_hits == 0
}

fn same_symbol_bifrost_hit_for_grep(pattern: &str, exchanges: &[ToolExchange]) -> bool {
    recoverable_source_symbols(pattern)
        .iter()
        .any(|symbol| overlaps_recent_bifrost_hit(symbol, exchanges))
}

fn hard_text_grep_allow_before_force(
    pattern: &str,
    glob: &str,
    path: &str,
    file_path: &str,
    context: &GateContext,
) -> bool {
    let kind = grep_pattern_kind(pattern, glob, path);
    config_or_build_text_scope(glob, path, file_path)
        || diagnostic_or_status_constant_text(pattern)
        || external_api_word_boundary_text_search(pattern)
        || post_edit_scope_verification(pattern, glob, path, file_path, &context.tool_exchanges)
        || (test_only_literal_scope(glob, path, file_path)
            && (lower_snake_literal_token_count(pattern) > 0
                || uppercase_constant_token_count(pattern) > 0
                || matches!(
                    kind,
                    GrepPatternKind::TestTextSearch
                        | GrepPatternKind::CodeIdiom
                        | GrepPatternKind::HeaderOrProtocolLiteral
                        | GrepPatternKind::NumericOrCodepointLiteral
                        | GrepPatternKind::UppercaseErrorConstantSet
                ))
            && !declaration_search_pattern(pattern)
            && !symbol_call_or_member(pattern))
}

fn repair_blocked_text_grep(
    pattern: &str,
    glob: &str,
    path: &str,
    file_path: &str,
    context: &GateContext,
) -> bool {
    must_allow_text_grep(pattern, glob, path, file_path)
        || hard_text_grep_allow_before_force(pattern, glob, path, file_path, context)
        || config_or_build_text_scope(glob, path, file_path)
        || exact_file_or_narrow_local_probe(pattern, glob, path, file_path)
        || bounded_test_literal_probe(pattern, glob, path, file_path)
        || test_scope_single_identifier_text_probe(pattern, glob, path, file_path)
        || env_or_errno_constant_text(pattern)
        || regex_wildcard_text_probe(pattern)
        || exact_member_or_call_expression_text_probe(pattern)
        || ambiguous_bare_identifier_text_probe(pattern)
        || recent_naming_variant_text_search(pattern, glob, path, file_path, &context.tool_exchanges)
        || path_or_protocol_construction_text_probe(pattern)
        || boolean_assignment_value_text_probe(pattern)
        || filename_or_resource_regex(pattern)
        || inline_regex_flag_text_search(pattern)
}

fn exact_file_or_narrow_local_probe(
    pattern: &str,
    glob: &str,
    path: &str,
    file_path: &str,
) -> bool {
    if grep_scope_granularity(glob, path, file_path) == "exact_file" {
        return true;
    }
    if !narrow_component_scope(glob, path, file_path) || declaration_search_pattern(pattern) {
        return false;
    }
    let tokens = identifier_tokens(pattern);
    (single_identifier_pattern(pattern)
        && !tokens
            .first()
            .is_some_and(|token| high_confidence_source_symbol(token)))
        || source_call_navigation_pattern(pattern)
        || exact_member_chain_or_call_text(pattern)
        || filename_or_resource_regex(pattern)
        || inline_regex_flag_text_search(pattern)
        || (regex_text_semantics_material(pattern) && !source_static_or_member_access(pattern))
}

fn bounded_test_literal_probe(_pattern: &str, glob: &str, path: &str, file_path: &str) -> bool {
    test_only_source_scope(glob, path, file_path)
        && !constructor_or_static_call_usage_pattern(_pattern)
}

fn test_only_source_scope(glob: &str, path: &str, file_path: &str) -> bool {
    let scope = normalized_grep_scope(glob, path, file_path).to_ascii_lowercase();
    if scope.is_empty() {
        return false;
    }
    if scope
        .split(['/', '\\'])
        .any(|part| matches!(part, "test" | "tests" | "spec" | "specs"))
    {
        return true;
    }
    let name = scope.rsplit('/').next().unwrap_or(&scope);
    name.contains("_test.")
        || name.starts_with("test_")
        || name.contains("test*.")
        || name.contains("*test.")
        || name.contains("*tests.")
        || name.contains("*test*.")
        || name.contains("*spec.")
        || name.contains("*spec*.")
}

fn test_scope_single_identifier_text_probe(
    pattern: &str,
    glob: &str,
    path: &str,
    file_path: &str,
) -> bool {
    bounded_test_scope(glob, path, file_path)
        && single_identifier_pattern(pattern)
        && !declaration_search_pattern(pattern)
        && !source_call_navigation_pattern(pattern)
}

fn env_or_errno_constant_text(pattern: &str) -> bool {
    let tokens = identifier_tokens(pattern);
    if tokens.len() != 1 {
        return false;
    }
    let original = &tokens[0];
    let token = original.to_ascii_uppercase();
    if original.len() >= 5
        && original.starts_with('E')
        && original
            .chars()
            .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_')
    {
        return true;
    }
    if !token.contains('_')
        || token.starts_with("UV_")
        || token.starts_with("ERR_")
        || token.starts_with("ERROR_")
    {
        return false;
    }
    token.split('_').any(|part| {
        matches!(
            part,
            "ADDR" | "PORT" | "POD" | "HOST" | "IP" | "IPS" | "URL" | "KEY" | "SECRET" | "TOKEN"
        )
    })
}

fn regex_wildcard_text_probe(pattern: &str) -> bool {
    pattern.contains(".*")
        && !declaration_search_pattern(pattern)
        && !constructor_or_static_call_usage_pattern(pattern)
        && !source_call_navigation_pattern(pattern)
}

fn exact_member_or_call_expression_text_probe(pattern: &str) -> bool {
    if declaration_search_pattern(pattern) || pattern.contains("::") || pattern.contains("->") {
        return false;
    }
    if exact_member_chain_or_call_text(pattern) {
        return true;
    }
    single_identifier_pattern(&pattern.replace("\\.", "")) && pattern.ends_with("\\.")
}

fn ambiguous_bare_identifier_text_probe(pattern: &str) -> bool {
    if pattern.contains('(')
        || pattern.contains("\\(")
        || declaration_search_pattern(pattern)
        || source_static_or_member_access(pattern)
    {
        return false;
    }
    let Some(token) = regex_wrapped_single_identifier(pattern) else {
        return false;
    };
    if high_confidence_source_symbol(&token) {
        return false;
    }
    lowercase_wire_key_like(&token) || token.chars().all(|ch| ch.is_ascii_lowercase())
}

fn regex_wrapped_single_identifier(pattern: &str) -> Option<String> {
    let mut normalized = pattern.trim().to_string();
    for wrapper in ["^", "$", "\\b"] {
        normalized = normalized.replace(wrapper, "");
    }
    normalized = normalized
        .trim_matches(['(', ')', '"', '\''])
        .trim()
        .to_string();
    let tokens = identifier_tokens(&normalized);
    if tokens.len() == 1 && normalized == tokens[0] {
        Some(tokens[0].clone())
    } else {
        None
    }
}

fn recent_naming_variant_text_search(
    pattern: &str,
    glob: &str,
    path: &str,
    file_path: &str,
    exchanges: &[ToolExchange],
) -> bool {
    let current_keys = naming_variant_keys(pattern);
    if current_keys.is_empty() {
        return false;
    }
    let scope = normalized_grep_scope(glob, path, file_path);
    let mut variants = std::collections::HashSet::new();
    for exchange in exchanges.iter().rev().take(10) {
        if exchange.tool_name != "grep_search" {
            continue;
        }
        if !scope.is_empty() && !exchange.arguments.contains(&scope) {
            continue;
        }
        for key in current_keys.iter() {
            for token in identifier_tokens(&exchange.arguments) {
                let token_key = naming_variant_key(&token);
                if token_key.as_deref() == Some(key.as_str()) {
                    variants.insert(token);
                }
            }
        }
    }
    variants.len() >= 2
}

fn naming_variant_keys(input: &str) -> std::collections::HashSet<String> {
    identifier_tokens(input)
        .into_iter()
        .filter_map(|token| naming_variant_key(&token))
        .collect()
}

fn naming_variant_key(token: &str) -> Option<String> {
    let key: String = token
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect();
    (key.len() >= 3).then_some(key)
}

fn path_or_protocol_construction_text_probe(pattern: &str) -> bool {
    let lower = pattern.to_ascii_lowercase();
    (pattern.contains("\\.") || pattern.contains('.'))
        && ["uri", "path", "encoded", "render", "scheme", "protocol"]
            .iter()
            .any(|marker| lower.contains(marker))
}

fn boolean_assignment_value_text_probe(pattern: &str) -> bool {
    let lower = pattern.to_ascii_lowercase();
    lower.contains("\\s*=\\s*")
        && (lower.contains("true") || lower.contains("false") || lower.contains("null"))
}

fn inline_regex_flag_text_search(pattern: &str) -> bool {
    ["(?i)", "(?-i)", "(?m)", "(?s)"]
        .iter()
        .any(|flag| pattern.contains(flag))
}

fn filename_or_resource_regex(pattern: &str) -> bool {
    let lower = pattern.to_ascii_lowercase();
    if !lower.contains("\\.") {
        return false;
    }
    let has_suffix = [
        "\\.scala", "\\.java", "\\.cs", "\\.php", "\\.go", "\\.c", "\\.h", "\\.m3u", "\\.json",
        "\\.xml", "\\.yaml", "\\.yml", "\\.sbt", "\\.md",
    ]
    .iter()
    .any(|suffix| lower.contains(suffix))
        || lower.ends_with('$');
    has_suffix && !pattern.contains("\\(") && !declaration_search_pattern(pattern)
}

fn constructor_or_static_call_usage_pattern(pattern: &str) -> bool {
    (pattern.contains("new ") || pattern.contains("::"))
        && (pattern.contains("\\(") || pattern.contains('('))
        && identifier_tokens(pattern)
            .iter()
            .any(|token| symbol_like_pattern(token))
}

fn source_relationship_call_site_pattern(pattern: &str) -> bool {
    if declaration_search_pattern(pattern) {
        return false;
    }
    let Some(identifier) = identifier_before_call_delimiter(pattern) else {
        return false;
    };
    identifier.len() >= 3 && identifier.chars().any(|ch| ch.is_ascii_alphabetic())
}

fn identifier_before_call_delimiter(pattern: &str) -> Option<String> {
    let normalized = pattern
        .replace("\\b", "")
        .replace("\\s*", "")
        .replace("\\(", "(")
        .replace("\\)", ")")
        .replace("\\{", "{")
        .replace("\\.", ".")
        .replace("\\-", "-");
    let call_pos = normalized.find('(').or_else(|| normalized.find('{'))?;
    let before = &normalized[..call_pos];
    let token: String = before
        .chars()
        .rev()
        .take_while(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    if token.is_empty() { None } else { Some(token) }
}

fn symbol_assignment_or_subscription_pattern(pattern: &str) -> bool {
    let tokens = identifier_tokens(pattern);
    tokens.len() == 1
        && high_confidence_source_symbol(&tokens[0])
        && (pattern.contains(" =") || pattern.contains("\\s*="))
        && !pattern.contains("+=")
        && !pattern.contains("\\+=")
}

fn deterministic_source_symbol_repair_tool(
    pattern: &str,
    glob: &str,
    path: &str,
    file_path: &str,
    context: &GateContext,
) -> Option<(RecommendedTool, &'static str)> {
    let source_call_site = source_relationship_call_site_pattern(pattern);
    let same_symbol_bifrost_hit =
        same_symbol_bifrost_hit_for_grep(pattern, &context.tool_exchanges);
    if (repair_blocked_text_grep(pattern, glob, path, file_path, context)
        && !same_symbol_bifrost_hit)
        || classify_path_or_glob(&normalized_grep_scope(glob, path, file_path))
            == StaticTextTarget::TextLike
        || exact_error_constant_search(pattern, path, file_path)
        || diagnostic_or_status_constant_text(pattern)
        || (natural_language_or_path_literal(pattern)
            && !declaration_search_pattern(pattern)
            && !constructor_or_static_call_usage_pattern(pattern)
            && !source_static_or_member_access(pattern)
            && !source_call_navigation_pattern(pattern)
            && !source_call_site)
        || (literal_value_text_target(pattern)
            && !same_symbol_bifrost_hit
            && !(declaration_search_pattern(pattern)
                || constructor_or_static_call_usage_pattern(pattern)
                || source_static_or_member_access(pattern)
                || source_call_navigation_pattern(pattern)
                || source_call_site
                || recoverable_source_symbols(pattern).len() >= 3))
        || import_include_or_package_text_target(pattern)
        || assertion_or_serialization_text(pattern)
        || (recent_bifrost_miss_for_grep(pattern, glob, path, file_path, &context.tool_exchanges)
            && !very_strong_deterministic_symbol_navigation(pattern, glob, path, file_path))
    {
        return None;
    }

    let scope = normalized_grep_scope(glob, path, file_path);
    let broad_nav_scope = bifrost_navigation_scope(glob, path, file_path);
    let declaration_component_scope = declaration_search_pattern(pattern)
        && is_source_like_path(&scope)
        && !bounded_test_scope(glob, path, file_path)
        && !broad_test_scope(glob, path, file_path);
    let high_conf_dir_symbol =
        source_directory_high_confidence_symbol(pattern, glob, path, file_path);
    if declaration_keyword_survey_pattern(pattern) && broad_nav_scope {
        return Some((RecommendedTool::GetSummaries, "declaration-keyword survey"));
    }
    if !broad_nav_scope && !declaration_component_scope && !high_conf_dir_symbol {
        return None;
    }
    let symbols = recoverable_source_symbols(pattern);
    if symbols.is_empty() && !source_call_site {
        return None;
    }

    if broad_nav_scope && declaration_search_pattern(pattern) && symbols.len() >= 3 {
        return Some((RecommendedTool::GetSummaries, "multi-symbol survey"));
    }
    if declaration_search_pattern(pattern) {
        return Some((RecommendedTool::SearchSymbols, "declaration"));
    }
    if broad_nav_scope
        && (constructor_or_static_call_usage_pattern(pattern)
            || source_static_or_member_access(pattern)
            || source_call_navigation_pattern(pattern)
            || source_call_site
            || symbol_assignment_or_subscription_pattern(pattern))
    {
        return Some((RecommendedTool::ScanUsages, "call/reference"));
    }
    if broad_nav_scope
        && symbols.len() >= 2
        && symbols
            .iter()
            .all(|symbol| symbol.contains('_') && !symbol.chars().all(|ch| ch.is_ascii_uppercase()))
    {
        return Some((RecommendedTool::SearchSymbols, "symbol discovery"));
    }
    if broad_nav_scope && uppercase_constant_token_count(pattern) >= 2 {
        return Some((RecommendedTool::SearchSymbols, "constant symbol discovery"));
    }
    if broad_nav_scope
        && same_symbol_bifrost_hit
        && symbols.len() == 1
        && symbols
            .iter()
            .any(|symbol| symbol.contains('_') && !uppercase_constant_like(symbol))
    {
        return Some((RecommendedTool::SearchSymbols, "known symbol discovery"));
    }
    if broad_nav_scope
        && symbols.len() == 1
        && symbols
            .iter()
            .any(|symbol| overlaps_recent_bifrost_hit(symbol, &context.tool_exchanges))
    {
        return Some((
            RecommendedTool::GetSymbolSources,
            "known symbol source lookup",
        ));
    }
    if broad_nav_scope
        && uppercase_constant_token_count(pattern) == 1
        && symbols
            .iter()
            .any(|symbol| overlaps_recent_bifrost_hit(symbol, &context.tool_exchanges))
    {
        return Some((
            RecommendedTool::GetSymbolSources,
            "known constant source lookup",
        ));
    }
    if broad_nav_scope && symbols.len() >= 3 {
        return Some((RecommendedTool::GetSummaries, "multi-symbol survey"));
    }
    if broad_nav_scope && symbols.len() >= 2 {
        return Some((RecommendedTool::SearchSymbols, "symbol discovery"));
    }
    if high_conf_dir_symbol {
        return Some((RecommendedTool::SearchSymbols, "symbol discovery"));
    }
    None
}

fn source_directory_high_confidence_symbol(
    pattern: &str,
    glob: &str,
    path: &str,
    file_path: &str,
) -> bool {
    let kind = grep_pattern_kind(pattern, glob, path);
    matches!(
        grep_scope_granularity(glob, path, file_path),
        "directory" | "source_glob"
    ) && is_source_like_path(&normalized_grep_scope(glob, path, file_path))
        && (matches!(
            kind,
            GrepPatternKind::SymbolFamilyOrRelationship | GrepPatternKind::DeclarationForm
        ) || symbol_like_pattern(pattern))
        && single_identifier_pattern(pattern)
        && identifier_tokens(pattern)
            .first()
            .is_some_and(|token| high_confidence_source_symbol(token))
        && !regex_text_semantics_material(pattern)
}

fn recoverable_source_symbols(pattern: &str) -> Vec<String> {
    const KEYWORDS: &[&str] = &[
        "abstract",
        "case",
        "class",
        "def",
        "enum",
        "false",
        "func",
        "function",
        "interface",
        "new",
        "object",
        "static",
        "struct",
        "trait",
        "true",
        "type",
        "virtual",
        "void",
    ];
    let mut symbols = Vec::new();
    for token in identifier_tokens(pattern) {
        let lower = token.to_ascii_lowercase();
        if KEYWORDS.contains(&lower.as_str()) {
            continue;
        }
        if token.len() >= 3
            && (symbol_like_pattern(&token)
                || (token.contains('_') && short_prefixed_source_symbol_like(&token)))
            && !symbols.iter().any(|prior| prior == &token)
        {
            symbols.push(token);
        }
    }
    symbols
}

fn high_confidence_source_symbol(token: &str) -> bool {
    if token.contains('_') {
        return false;
    }
    let Some(first) = token.chars().next() else {
        return false;
    };
    first.is_ascii_uppercase()
        && !token.chars().all(|ch| ch.is_ascii_uppercase())
        && (token.len() >= 12 || token.chars().filter(|ch| ch.is_ascii_uppercase()).count() >= 3)
}

fn source_static_or_member_access(pattern: &str) -> bool {
    let symbols = recoverable_source_symbols(pattern);
    !symbols.is_empty()
        && (pattern.contains("::")
            || pattern.contains("->")
            || pattern.contains("\\.")
            || pattern.contains('.'))
}

fn source_call_navigation_pattern(pattern: &str) -> bool {
    let symbols = recoverable_source_symbols(pattern);
    !symbols.is_empty() && (pattern.contains("\\(") || pattern.contains("\\s*\\("))
}

fn literal_value_text_target(pattern: &str) -> bool {
    let lower = pattern.to_ascii_lowercase();
    numeric_or_codepoint_literal(pattern)
        || header_or_protocol_literal(pattern, &lower)
        || wire_key_or_serialization_key(pattern)
        || natural_language_or_path_literal(pattern)
        || mixed_lower_snake_serialization_or_config_search(pattern)
        || assertion_or_build_string_regex_text(pattern)
}

fn declaration_keyword_survey_pattern(pattern: &str) -> bool {
    let parts: Vec<String> = pattern
        .split('|')
        .map(|part| part.trim().trim_matches(['"', '\'']).to_ascii_lowercase())
        .filter(|part| !part.is_empty())
        .collect();
    parts.len() >= 2
        && parts.iter().all(|part| {
            matches!(
                part.as_str(),
                "class"
                    | "object"
                    | "trait"
                    | "interface"
                    | "enum"
                    | "def"
                    | "func"
                    | "function"
                    | "struct"
                    | "type"
            )
        })
}

fn import_include_or_package_text_target(pattern: &str) -> bool {
    let lower = pattern.to_ascii_lowercase();
    lower.contains("#include")
        || lower.contains("import ")
        || lower.contains("package ")
        || lower.contains("using ")
        || pattern.contains('/')
        || pattern.contains("\\/")
        || pattern.contains("\\+")
        || (pattern.contains("\\.") && pattern.contains("\\+"))
        || pattern.contains("\\\"")
        || pattern.contains('"')
}

fn single_identifier_pattern(pattern: &str) -> bool {
    let tokens = identifier_tokens(pattern);
    tokens.len() == 1 && tokens[0] == pattern.trim()
}

fn exact_bare_identifier_text_exception(pattern: &str, exact_text_or_regex_needed: bool) -> bool {
    exact_text_or_regex_needed
        && single_identifier_pattern(pattern)
        && !declaration_search_pattern(pattern)
        && !symbol_call_or_member(pattern)
        && !source_static_or_member_access(pattern)
        && !source_call_navigation_pattern(pattern)
}

fn exact_error_constant_search(pattern: &str, path: &str, file_path: &str) -> bool {
    if path != "src/win" && file_path != "src/win" {
        return false;
    }
    let tokens = identifier_tokens(pattern);
    !tokens.is_empty()
        && tokens
            .iter()
            .all(|token| token.chars().all(|ch| ch.is_ascii_uppercase()) && token.contains("ERR"))
}

fn uppercase_constant_token_count(pattern: &str) -> usize {
    identifier_tokens(pattern)
        .into_iter()
        .filter(|token| uppercase_constant_like(token))
        .count()
}

fn lower_snake_literal_token_count(pattern: &str) -> usize {
    identifier_tokens(pattern)
        .into_iter()
        .filter(|token| lowercase_wire_key_like(token) && !short_prefixed_source_symbol_like(token))
        .count()
}

fn external_api_word_boundary_text_search(pattern: &str) -> bool {
    if !pattern.contains('|') || !pattern.contains("\\b") {
        return false;
    }
    let tokens = identifier_tokens(pattern);
    tokens.len() >= 2
        && tokens.iter().all(|token| {
            token.len() >= 3
                && token.chars().all(|ch| ch.is_ascii_lowercase())
                && !token.contains('_')
        })
}

fn external_api_or_annotation_names(pattern: &str) -> bool {
    if declaration_search_pattern(pattern) {
        return false;
    }
    let tokens = identifier_tokens(pattern);
    if tokens.len() == 1
        && external_annotation_or_test_api_token(&tokens[0])
        && !symbol_call_or_member(pattern)
    {
        return true;
    }
    if !pattern.contains('|') {
        return false;
    }
    tokens.len() >= 3
        && tokens.iter().all(|token| {
            token.starts_with("Json")
                || token.ends_with("Attribute")
                || token.ends_with("Annotation")
                || token.ends_with("Property")
                || token.ends_with("Creator")
                || token.ends_with("Include")
                || token.ends_with("Setter")
                || token.ends_with("IgnoreProperties")
        })
        && !symbol_call_or_member(pattern)
}

fn external_annotation_or_test_api_token(token: &str) -> bool {
    token.ends_with("Attribute")
        || token.ends_with("Annotation")
        || token.ends_with("VisibleTo")
        || (token.starts_with("Test")
            && ["Case", "Fixture", "Method", "Source"]
                .iter()
                .any(|suffix| token.ends_with(suffix)))
}

fn compound_code_idiom_regex(pattern: &str) -> bool {
    if !pattern.contains('|') || declaration_search_pattern(pattern) {
        return false;
    }
    if constructor_or_static_call_usage_pattern(pattern) {
        return false;
    }
    let has_call_fragment = pattern.contains("\\(") || pattern.contains('(');
    let has_qualified_fragment = pattern.contains("\\.") || pattern.contains('.');
    let has_literal_arg = regex_has_literal_call_arg(pattern);
    let has_mixed_terms = pattern
        .split('|')
        .filter(|term| !term.trim().is_empty())
        .count()
        >= 2;
    has_mixed_terms && has_call_fragment && (has_literal_arg || has_qualified_fragment)
}

fn regex_has_literal_call_arg(pattern: &str) -> bool {
    pattern.contains("\\(2\\)")
        || pattern.contains("(2)")
        || pattern.contains("\\(true\\)")
        || pattern.contains("(true)")
        || pattern.contains("\\(false\\)")
        || pattern.contains("(false)")
        || pattern.contains("\\\"")
        || pattern.contains('"')
}

fn normalize_shell_classifier_consistency(output: &mut ShellClassifierOutput) {
    if output.shell_semantics_required
        || matches!(
            output.allow_exception,
            ShellAllowException::BuildTestGitPackageOrProjectCli
                | ShellAllowException::ShellSemanticsRequired
                | ShellAllowException::RawByteOrHiddenWhitespace
                | ShellAllowException::MutationOrWriteOrDelete
                | ShellAllowException::HeredocOrCommandSubstitution
                | ShellAllowException::GeneratedArtifactOrExitBehavior
        )
    {
        output.decision = ShellClassifierDecision::AllowShell;
        output.recommended_tool = RecommendedTool::None;
        output.suggested_args = json!({});
        return;
    }
    if shell_suggested_args_contradict_route(output) {
        output.decision = ShellClassifierDecision::AllowShell;
        output.recommended_tool = RecommendedTool::None;
        output.suggested_args = json!({});
        output.confidence = GateConfidence::Low;
        return;
    }
    if output.confidence != GateConfidence::High {
        if output.confidence == GateConfidence::Medium
            && !output.shell_semantics_required
            && output.builtin_preserves_intent
            && (output.decision == ShellClassifierDecision::UseBuiltinTool
                || route_prefix(&output.reason) == Some(RoutePrefix::UseBuiltin))
        {
            output.decision = ShellClassifierDecision::UseBuiltinTool;
            output.recommended_tool = builtin_tool_for_shell_intent(&output.intent);
            return;
        }
        output.decision = ShellClassifierDecision::AllowShell;
        output.recommended_tool = RecommendedTool::None;
        output.suggested_args = json!({});
        return;
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

fn shell_suggested_args_contradict_route(output: &ShellClassifierOutput) -> bool {
    output
        .suggested_args
        .get("tool")
        .and_then(Value::as_str)
        .is_some_and(|tool| tool != output.recommended_tool.as_tool_name())
}

fn enforce_shell_classifier_policy(output: &mut ShellClassifierOutput, context: &GateContext) {
    if context.tool_name != "run_shell_command" {
        return;
    }
    let command = context
        .args
        .get("command")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    if command.is_empty() {
        return;
    }
    if shell_semantics_required(command) {
        output.decision = ShellClassifierDecision::AllowShell;
        output.recommended_tool = RecommendedTool::None;
        output.suggested_args = json!({});
        output.reason = format!(
            "ALLOW_SHELL because `{}` requires process, shell, or execution semantics.",
            truncate_to(command, 160)
        );
        output.shell_semantics_required = true;
        output.builtin_preserves_intent = false;
        output.bifrost_fit = BifrostFit::NotApplicable;
        output.allow_exception = ShellAllowException::ShellSemanticsRequired;
        output.replacement_class = ShellReplacementClass::AllowShellUncertain;
        output.confidence = GateConfidence::High;
        return;
    }
    if shell_search_stream(command) == ShellSearchStream::FilePaths {
        output.decision = ShellClassifierDecision::AllowShell;
        output.recommended_tool = RecommendedTool::None;
        output.suggested_args = json!({});
        output.reason = format!(
            "ALLOW_SHELL because `{}` filters file paths, not source contents; Bifrost would not preserve the intent.",
            truncate_to(command, 160)
        );
        output.intent = ShellIntent::DirectoryOrFileDiscovery;
        output.allow_exception = ShellAllowException::None;
        output.replacement_class = ShellReplacementClass::AllowShellUncertain;
        output.bifrost_fit = BifrostFit::NotApplicable;
        output.builtin_preserves_intent = false;
        output.confidence = GateConfidence::High;
        return;
    }
    if output.decision == ShellClassifierDecision::UseBifrostTool
        && shell_search_pattern(command).is_some_and(|pattern| {
            shell_source_search_builtin_exception(
                &pattern,
                grep_pattern_kind(&pattern, "", command),
            )
        })
    {
        output.decision = ShellClassifierDecision::UseBuiltinTool;
        output.recommended_tool = builtin_tool_for_shell_command(command);
        output.suggested_args = json!({});
        output.reason = format!(
            "USE_BUILTIN_TOOL because `{}` is exact/literal shell search and no concrete shell semantic is required.",
            truncate_to(command, 160)
        );
        output.intent = ShellIntent::LiteralTextSearch;
        output.allow_exception = ShellAllowException::None;
        output.replacement_class = ShellReplacementClass::UseBuiltinInspection;
        output.bifrost_fit = BifrostFit::NotApplicable;
        output.builtin_preserves_intent = true;
        output.confidence = GateConfidence::High;
        return;
    }
    if matches!(
        output.decision,
        ShellClassifierDecision::AllowShell | ShellClassifierDecision::UseBuiltinTool
    ) {
        if let Some(tool) = strong_shell_source_symbol_search(command) {
            output.decision = ShellClassifierDecision::UseBifrostTool;
            let intent = if tool == RecommendedTool::ScanUsages {
                ShellIntent::SymbolReferenceLookup
            } else {
                ShellIntent::SymbolDefinitionLookup
            };
            output.recommended_tool = tool;
            output.suggested_args = json!({});
            output.reason = format!(
                "USE_BIFROST_TOOL because `{}` is broad source-content symbol search and no concrete shell semantic is required.",
                truncate_to(command, 160)
            );
            output.intent = intent;
            output.allow_exception = ShellAllowException::None;
            output.replacement_class = ShellReplacementClass::UseBifrostSymbol;
            output.bifrost_fit = BifrostFit::SameOrMoreDirect;
            output.builtin_preserves_intent = false;
            output.confidence = GateConfidence::High;
            return;
        }
    }
    if output.decision == ShellClassifierDecision::AllowShell
        && route_prefix(&output.reason) == Some(RoutePrefix::UseBuiltin)
        && conservative_shell_read_search_inspection(command)
    {
        output.decision = ShellClassifierDecision::UseBuiltinTool;
        output.recommended_tool = builtin_tool_for_shell_command(command);
        output.suggested_args = json!({});
        output.reason = format!(
            "USE_BUILTIN_TOOL because `{}` is conservative file/search/list inspection and the classifier's allow_original route contradicted its builtin policy fields.",
            truncate_to(command, 160)
        );
        let lower = command.to_ascii_lowercase();
        output.intent = if simple_shell_search_like(command) {
            ShellIntent::LiteralTextSearch
        } else if lower.contains("find ") || lower.contains("ls ") {
            ShellIntent::DirectoryOrFileDiscovery
        } else {
            ShellIntent::OrdinaryFileRead
        };
        output.allow_exception = ShellAllowException::None;
        output.replacement_class = ShellReplacementClass::UseBuiltinInspection;
        output.bifrost_fit = BifrostFit::NotApplicable;
        output.confidence = GateConfidence::High;
        return;
    }
    if output.decision == ShellClassifierDecision::UseBuiltinTool && xargs_content_grep(command) {
        if let Some(tool) = strong_shell_source_symbol_search(command) {
            output.decision = ShellClassifierDecision::UseBifrostTool;
            output.intent = if tool == RecommendedTool::ScanUsages {
                ShellIntent::SymbolReferenceLookup
            } else {
                ShellIntent::SymbolDefinitionLookup
            };
            output.recommended_tool = tool;
            output.suggested_args = json!({});
            output.reason = format!(
                "USE_BIFROST_TOOL because `{}` is find-xargs source-content symbol search and no concrete shell semantic is required.",
                truncate_to(command, 160)
            );
            output.allow_exception = ShellAllowException::None;
            output.replacement_class = ShellReplacementClass::UseBifrostSymbol;
            output.bifrost_fit = BifrostFit::SameOrMoreDirect;
            output.builtin_preserves_intent = false;
            output.confidence = GateConfidence::High;
            return;
        }
    }
    if output.decision == ShellClassifierDecision::AllowShell
        && (simple_shell_read_like(command)
            || script_exact_file_text_inspection(command)
            || script_bounded_file_text_inspection(command))
    {
        output.decision = ShellClassifierDecision::UseBuiltinTool;
        output.recommended_tool = builtin_tool_for_shell_command(command);
        output.suggested_args = json!({});
        output.reason = format!(
            "USE_BUILTIN_TOOL because `{}` is file/search/list inspection and no concrete shell semantic is required.",
            truncate_to(command, 160)
        );
        let lower = command.to_ascii_lowercase();
        output.intent = if simple_shell_search_like(command) {
            ShellIntent::LiteralTextSearch
        } else if lower.contains("find ") || lower.contains("ls ") {
            ShellIntent::DirectoryOrFileDiscovery
        } else {
            ShellIntent::OrdinaryFileRead
        };
        output.allow_exception = ShellAllowException::None;
        output.replacement_class = ShellReplacementClass::UseBuiltinInspection;
        output.bifrost_fit = BifrostFit::NotApplicable;
        output.builtin_preserves_intent = true;
        output.confidence = GateConfidence::High;
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
        TextIntent::SymbolReferenceLookup | TextIntent::SymbolUsageLookup => {
            RecommendedTool::ScanUsages
        }
        TextIntent::BroadSemanticOrientation => RecommendedTool::GetSummaries,
        TextIntent::SymbolDefinitionLookup => RecommendedTool::SearchSymbols,
        _ => RecommendedTool::SearchSymbols,
    }
}

fn deterministic_text_tool_for_grep(
    intent: &TextIntent,
    pattern: &str,
    glob: &str,
    path: &str,
) -> RecommendedTool {
    if constructor_or_static_call_usage_pattern(pattern) {
        return RecommendedTool::ScanUsages;
    }
    match grep_pattern_kind(pattern, glob, path) {
        GrepPatternKind::SymbolCallOrMember => RecommendedTool::ScanUsages,
        GrepPatternKind::DeclarationForm => RecommendedTool::SearchSymbols,
        _ => bifrost_tool_for_text_intent(intent),
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

pub fn classifier_trace_context(context: &GateContext) -> Value {
    json!({
        "pending_call_features": pending_call_features(&context.tool_name, &context.args, &context.tool_exchanges),
        "compact_evidence": compact_classifier_evidence(&context.tool_name, &context.args, &context.tool_exchanges),
        "static_text_target": classify_static_text_target(&context.tool_name, &context.args),
    })
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
        "bifrost_navigation_scope": bifrost_navigation_scope(glob, path, file_path),
        "bounded_test_scope": bounded_test_scope(glob, path, file_path),
        "narrow_component_scope": narrow_component_scope(glob, path, file_path),
        "grep_scope_recent_edit_or_write": same_scope_recent_edit_or_write(glob, path, file_path, exchanges)
            || same_path_recent_tool(&scope_target, exchanges, &["edit", "write_file"]),
        "grep_scope_recent_read": same_path_recent_tool(&scope_target, exchanges, &["read_file"]),
        "source_like_scope": is_source_like_path(glob) || is_source_like_path(path) || (glob.is_empty() && path.is_empty()),
        "broad_repo_scope": glob.is_empty() && path.is_empty(),
        "symbol_like_pattern": symbol_like_pattern(pattern),
        "literal_like_pattern": literal_like_pattern(pattern),
        "recent_bifrost_miss_overlap": overlaps_recent_bifrost_miss(pattern, exchanges),
        "search_scope": search_scope(glob, path),
        "pattern_shape": pattern_shape(pattern),
        "scope_kind": grep_scope_kind(glob, path, file_path),
        "is_source_root_wide": is_source_root_wide_grep_scope(glob, path, file_path),
        "is_test_scope": bounded_test_scope(glob, path, file_path),
        "is_config_or_build_scope": config_or_build_text_scope(glob, path, file_path),
        "is_exact_file_scope": scope_granularity == "exact_file",
        "looks_like_single_identifier": single_identifier_pattern(pattern),
        "looks_like_call_pattern": symbol_call_or_member(pattern) || constructor_or_static_call_usage_pattern(pattern),
        "looks_like_declaration_pattern": declaration_search_pattern(pattern),
        "looks_like_concrete_code_shape": concrete_code_shape_pattern(pattern),
        "post_edit_or_verification_context": post_edit_scope_verification(pattern, glob, path, file_path, exchanges),
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
        "shell_search_stream": shell_search_stream(command),
        "shell_search_pattern": shell_search_pattern(command),
        "source_scope": command_has_source_scope(command),
        "source_root_wide": command_has_source_scope(command) && !shell_search_has_exact_file_scope(command),
        "exact_file_scope": shell_search_has_exact_file_scope(command),
        "config_or_build_scope": shell_search_targets_config_or_docs(&command.to_ascii_lowercase()),
        "looks_like_single_identifier": shell_search_pattern(command).is_some_and(|pattern| single_identifier_pattern(&pattern)),
        "looks_like_call_pattern": shell_search_pattern(command).is_some_and(|pattern| shell_call_or_member_search(&pattern) || constructor_or_static_call_usage_pattern(&pattern)),
        "looks_like_declaration_pattern": shell_search_pattern(command).is_some_and(|pattern| declaration_search_pattern(&pattern)),
        "looks_like_concrete_code_shape": shell_search_pattern(command).is_some_and(|pattern| concrete_code_shape_pattern(&pattern)),
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
    let glob = glob.trim();
    let path = path.trim();
    let file_path = file_path.trim();
    if !file_path.is_empty() {
        return file_path.to_string();
    }
    if !glob.is_empty() && !path.is_empty() && path != "." {
        if glob.contains('/') {
            return glob.to_string();
        }
        return format!("{}/{}", path.trim_end_matches('/'), glob);
    }
    if !glob.is_empty() && !glob.contains('*') && glob.contains('.') {
        if path.is_empty() || path == "." || glob.contains('/') {
            return glob.to_string();
        }
        return format!("{}/{}", path.trim_end_matches('/'), glob);
    }
    if !path.is_empty() {
        path.to_string()
    } else {
        glob.to_string()
    }
}

fn normalized_grep_scope(glob: &str, path: &str, file_path: &str) -> String {
    if !file_path.trim().is_empty() {
        return file_path.trim().to_string();
    }
    let glob = glob.trim();
    let path = path.trim();
    if !glob.is_empty() && (path.is_empty() || path == ".") {
        return glob.to_string();
    }
    if !glob.is_empty() && !path.is_empty() {
        if glob.contains('/') {
            glob.to_string()
        } else {
            format!("{}/{}", path.trim_end_matches('/'), glob)
        }
    } else if !path.is_empty() {
        path.to_string()
    } else {
        glob.to_string()
    }
}

fn grep_scope_granularity(glob: &str, path: &str, file_path: &str) -> &'static str {
    let target = grep_scope_target(glob, path, file_path);
    if target != "." && !target.contains('*') && target.contains('.') {
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

fn grep_scope_kind(glob: &str, path: &str, file_path: &str) -> &'static str {
    let granularity = grep_scope_granularity(glob, path, file_path);
    if granularity == "exact_file" {
        "exact_file"
    } else if config_or_build_text_scope(glob, path, file_path) {
        "config_or_build"
    } else if bounded_test_scope(glob, path, file_path) {
        "bounded_test"
    } else if broad_test_scope(glob, path, file_path) {
        "broad_test"
    } else if matches!(granularity, "repo_wide" | "source_glob")
        || source_root_scope(glob, path, file_path)
    {
        "source_root_wide"
    } else if narrow_component_scope(glob, path, file_path) {
        "narrow_component"
    } else if shallow_component_scope(glob, path, file_path) {
        "shallow_component"
    } else {
        "unknown"
    }
}

fn is_source_root_wide_grep_scope(glob: &str, path: &str, file_path: &str) -> bool {
    matches!(
        grep_scope_kind(glob, path, file_path),
        "source_root_wide" | "broad_test"
    )
}

fn path_is_repo_root(path: &str) -> bool {
    let trimmed = path.trim();
    trimmed.is_empty() || trimmed == "."
}

fn path_component_count(path: &str) -> usize {
    path.split('/')
        .filter(|part| {
            let part = part.trim();
            !part.is_empty() && part != "." && !part.contains('*')
        })
        .count()
}

fn recursive_source_glob(glob: &str, path: &str, file_path: &str) -> bool {
    let scope = normalized_grep_scope(glob, path, file_path);
    scope.contains("**") && is_source_like_path(&scope)
}

fn source_root_scope(glob: &str, path: &str, file_path: &str) -> bool {
    let scope = normalized_grep_scope(glob, path, file_path);
    matches!(
        scope.as_str(),
        "" | "." | "src" | "source" | "test" | "tests"
    )
}

fn broad_test_scope(glob: &str, path: &str, file_path: &str) -> bool {
    let scope = normalized_grep_scope(glob, path, file_path).to_ascii_lowercase();
    (scope.contains("test") || scope.contains("spec"))
        && (scope.contains("**")
            || matches!(path.trim(), "." | "test" | "tests" | "spec" | "specs"))
}

fn bounded_test_scope(glob: &str, path: &str, file_path: &str) -> bool {
    let scope = normalized_grep_scope(glob, path, file_path).to_ascii_lowercase();
    (scope.contains("test") || scope.contains("spec"))
        && !scope.contains("**")
        && !path_is_repo_root(&scope)
}

fn test_only_literal_scope(glob: &str, path: &str, file_path: &str) -> bool {
    let scope = normalized_grep_scope(glob, path, file_path).to_ascii_lowercase();
    (scope.contains("test") || scope.contains("spec"))
        && !scope.contains("/src/")
        && !scope.starts_with("src/")
}

fn shallow_component_scope(glob: &str, path: &str, file_path: &str) -> bool {
    let scope = normalized_grep_scope(glob, path, file_path);
    if path_is_repo_root(&scope) || scope.contains("**") {
        return false;
    }
    let components = path_component_count(&scope);
    scope.starts_with("src/") && components <= 2
}

fn narrow_component_scope(glob: &str, path: &str, file_path: &str) -> bool {
    let scope = normalized_grep_scope(glob, path, file_path);
    if path_is_repo_root(&scope) || scope.contains("**") {
        return false;
    }
    path_component_count(&scope) >= 3
}

fn bifrost_navigation_scope(glob: &str, path: &str, file_path: &str) -> bool {
    matches!(
        grep_scope_granularity(glob, path, file_path),
        "repo_wide" | "source_glob"
    ) || recursive_source_glob(glob, path, file_path)
        || source_root_scope(glob, path, file_path)
        || shallow_component_scope(glob, path, file_path)
        || broad_test_scope(glob, path, file_path)
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
        go_doc_like(segment)
            || build_test_git_package_like(segment)
            || environment_probe(segment)
            || shell_parameter_expansion_probe(segment)
            || process_control_or_inspection(segment)
            || shell_conditional_probe_like(
                &strip_harmless_shell_prefixes(segment).to_ascii_lowercase(),
            )
            || compiler_preprocessor_probe(segment)
            || raw_byte_probe(segment)
            || shell_mutates_files(segment)
            || script_raw_byte_or_format_probe(segment)
            || script_mutates_files(segment)
            || inline_runtime_execution(segment)
    }) || script_raw_byte_or_format_probe(command)
        || script_mutates_files(command)
        || environment_probe(command)
        || shell_parameter_expansion_probe(command)
        || command.contains("$(")
}

fn process_control_or_inspection(command: &str) -> bool {
    let lower = strip_harmless_shell_prefixes(command).to_ascii_lowercase();
    matches!(
        lower.split_whitespace().next().unwrap_or(""),
        "sleep" | "ps" | "pgrep" | "pkill" | "kill" | "jobs"
    )
}

fn environment_probe(command: &str) -> bool {
    let lower = command.trim_start().to_ascii_lowercase();
    matches!(
        lower.split_whitespace().next().unwrap_or(""),
        "env" | "printenv"
    )
}

fn shell_parameter_expansion_probe(command: &str) -> bool {
    let stripped = strip_harmless_shell_prefixes(command);
    let first = stripped.split_whitespace().next().unwrap_or("");
    matches!(first, "echo" | "printf") && contains_shell_variable_expansion(&stripped)
}

fn contains_shell_variable_expansion(command: &str) -> bool {
    if command.contains("${") || command.contains("$(") {
        return true;
    }
    let mut chars = command.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '$' {
            continue;
        }
        if chars
            .peek()
            .is_some_and(|next| next.is_ascii_alphabetic() || *next == '_')
        {
            return true;
        }
    }
    false
}

fn same_path_recent_tool(path: &str, exchanges: &[ToolExchange], tools: &[&str]) -> bool {
    if path.len() < 3 {
        return false;
    }
    exchanges.iter().rev().take(8).any(|exchange| {
        tools.contains(&exchange.tool_name.as_str()) && exchange.arguments.contains(path)
    })
}

fn post_edit_scope_verification(
    pattern: &str,
    glob: &str,
    path: &str,
    file_path: &str,
    exchanges: &[ToolExchange],
) -> bool {
    same_scope_recent_edit_or_write(glob, path, file_path, exchanges)
        && (pattern.contains("\\(")
            || pattern.contains("\\s*")
            || exact_member_chain_or_call_text(pattern)
            || uppercase_error_constant_set(pattern)
            || identifier_tokens(pattern).len() == 1)
}

fn same_scope_recent_edit_or_write(
    glob: &str,
    path: &str,
    file_path: &str,
    exchanges: &[ToolExchange],
) -> bool {
    let scope = normalized_grep_scope(glob, path, file_path);
    if scope.len() < 3 {
        return false;
    }
    exchanges.iter().rev().take(8).any(|exchange| {
        matches!(exchange.tool_name.as_str(), "edit" | "write_file")
            && edited_path_matches_scope(&exchange.arguments, &scope)
    })
}

fn edited_path_matches_scope(arguments: &str, scope: &str) -> bool {
    let paths = serde_json::from_str::<Value>(arguments)
        .ok()
        .map(|value| {
            ["file_path", "path"]
                .iter()
                .filter_map(|key| value.get(*key).and_then(Value::as_str))
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .filter(|paths| !paths.is_empty())
        .unwrap_or_else(|| vec![arguments.to_string()]);
    paths
        .iter()
        .any(|path| path_matches_grep_scope(path, scope))
}

fn path_matches_grep_scope(path: &str, scope: &str) -> bool {
    let path = path.trim().trim_start_matches("./");
    let scope = scope.trim().trim_start_matches("./");
    if scope.contains('*') {
        let prefix = scope.split('*').next().unwrap_or("").trim_end_matches('/');
        return prefix.len() >= 3 && path.starts_with(prefix);
    }
    path == scope || path.starts_with(&format!("{}/", scope.trim_end_matches('/')))
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
    if script_exact_file_text_inspection(command) || script_bounded_file_text_inspection(command) {
        return Some(ShellStaticRoute::UseBuiltin(
            "static_shell_exact_file_inspection",
            RecommendedTool::ReadFile,
        ));
    }
    if shell_semantics_required(command) {
        return Some(ShellStaticRoute::AllowShell("static_shell_semantics"));
    }
    if shell_search_stream(command) == ShellSearchStream::FilePaths {
        return Some(ShellStaticRoute::AllowShell("static_shell_path_filter"));
    }
    if shell_search_pattern(command).is_some() {
        return None;
    }
    if simple_shell_read_like(command) && !simple_shell_search_like(command) {
        return Some(ShellStaticRoute::UseBuiltin(
            "static_shell_builtin_inspection",
            builtin_tool_for_shell_command(command),
        ));
    }
    if ambiguous_source_shell_search(command) {
        return None;
    }
    if shell_search_stream(command) == ShellSearchStream::FileContents {
        return None;
    }
    if simple_shell_read_like(command) {
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
    if env_probe_like(&lower) {
        return true;
    }
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
        "python -m py_compile",
        "python -m pytest",
        "uv run pytest",
        "uv run python -m py_compile",
        "uv run python -m pytest",
        "dotnet build",
        "dotnet test",
        "dotnet --info",
        "dotnet --version",
        "sbt ",
        "bash .harness/",
        "bash ./.harness/",
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
        || lower.contains("sbt_bin")
        || lower.contains("-batch \"projects\"")
        || lower.contains("-batch projects")
        || lower.contains("uv_run_tests")
        || lower.contains(" testonly")
}

fn go_doc_like(command: &str) -> bool {
    let stripped = strip_harmless_shell_prefixes(command);
    let mut words = stripped.split_whitespace();
    let Some(cmd) = words.next() else {
        return false;
    };
    let basename = cmd.rsplit('/').next().unwrap_or(cmd);
    basename == "go" && words.next() == Some("doc")
}

fn env_probe_like(lower: &str) -> bool {
    let normalized = lower.replace("~/", "").replace("./", "");
    normalized.contains("dotnet --info")
        || normalized.contains("dotnet --version")
        || echo_env_probe_like(&normalized)
        || normalized.starts_with("which ")
        || normalized.starts_with("command -v ")
        || normalized.starts_with("go version")
        || normalized.starts_with("java -version")
        || shell_conditional_probe_like(&normalized)
}

fn echo_env_probe_like(lower: &str) -> bool {
    let trimmed = lower.trim();
    trimmed.starts_with("echo ") && contains_shell_variable_expansion(trimmed)
}

fn shell_conditional_probe_like(lower: &str) -> bool {
    (lower.starts_with("if [[") || lower.starts_with("if [") || lower.starts_with("test "))
        && (lower.contains(" -x ")
            || lower.contains(" -e ")
            || lower.contains(" -f ")
            || lower.contains("command -v")
            || lower.contains("which "))
}

fn compiler_preprocessor_probe(command: &str) -> bool {
    let lower = strip_harmless_shell_prefixes(command).to_ascii_lowercase();
    let mut words = lower.split_whitespace();
    let name = words.next().unwrap_or("");
    matches!(
        name,
        "cc" | "gcc" | "clang" | "cpp" | "c++" | "g++" | "clang++"
    ) && words.any(|word| word == "-e" || word.starts_with("-e"))
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
        || lower.contains("perl -pi")
        || lower.contains("perl -i")
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

fn script_exact_file_text_inspection(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    if !scriptish(&lower)
        || script_mutates_files(command)
        || true_script_raw_byte_or_format_probe(command)
        || recursive_source_traversal(&lower)
    {
        return false;
    }
    (lower.contains(".read_text(") || lower.contains("open("))
        && source_path_literal_count(&lower) == 1
}

fn script_bounded_file_text_inspection(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    if !scriptish(&lower)
        || script_mutates_files(command)
        || true_script_raw_byte_or_format_probe(command)
        || recursive_source_traversal(&lower)
    {
        return false;
    }
    (lower.contains(".read_text(") || lower.contains("open("))
        && source_path_literal_count(&lower) >= 1
}

fn recursive_source_traversal(lower_command: &str) -> bool {
    (lower_command.contains(".rglob(")
        || lower_command.contains(".glob(")
        || lower_command.contains("os.walk")
        || lower_command.contains("walkdir")
        || lower_command.contains("find "))
        && SOURCE_EXTENSIONS
            .iter()
            .any(|ext| lower_command.contains(ext))
}

fn source_path_literal_count(lower_command: &str) -> usize {
    lower_command
        .split(|ch: char| ch.is_whitespace() || matches!(ch, '"' | '\'' | ',' | ')' | '('))
        .map(|token| token.trim_matches(|ch: char| matches!(ch, ';' | ':' | '[' | ']')))
        .filter(|token| {
            token.contains('.')
                && SOURCE_EXTENSIONS
                    .iter()
                    .any(|ext| token.ends_with(ext) || token.contains(&format!("{ext}:")))
        })
        .collect::<std::collections::HashSet<_>>()
        .len()
}

fn true_script_raw_byte_or_format_probe(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    script_raw_byte_or_format_probe(command)
        && !(lower.contains("decode(")
            && (lower.contains(".read()") || lower.contains("open("))
            && (lower.contains(" in text")
                || lower.contains(" in content")
                || lower.contains(".count(")))
}

fn script_recursive_constructor_or_call_search(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    scriptish(&lower)
        && !script_mutates_files(command)
        && !script_raw_byte_or_format_probe(command)
        && recursive_source_traversal(&lower)
        && command.split(['"', '\'']).enumerate().any(|(index, part)| {
            index % 2 == 1
                && (part.contains('(') || part.contains("::") || part.contains("->"))
                && identifier_tokens(part).iter().any(|token| {
                    token.len() >= 8
                        && symbol_like_pattern(token)
                        && !lowercase_wire_key_like(token)
                })
        })
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
            || xargs_content_grep(&lower)
    })
}

fn conservative_shell_read_search_inspection(command: &str) -> bool {
    if shell_semantics_required(command)
        || shell_search_stream(command) == ShellSearchStream::FilePaths
    {
        return false;
    }
    let mut saw_part = false;
    for part in command_segments(command)
        .iter()
        .flat_map(|segment| segment.split(" | "))
    {
        let name = shell_segment_executable_name(part);
        if name.is_empty() {
            continue;
        }
        saw_part = true;
        if !matches!(
            name.as_str(),
            "grep"
                | "egrep"
                | "fgrep"
                | "rg"
                | "ag"
                | "ack"
                | "cat"
                | "head"
                | "tail"
                | "wc"
                | "ls"
                | "nl"
                | "sed"
        ) {
            return false;
        }
    }
    saw_part
}

fn shell_segment_executable_name(segment: &str) -> String {
    strip_harmless_shell_prefixes(segment)
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches(['"', '\''])
        .rsplit('/')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn xargs_content_grep(command: &str) -> bool {
    let tokens: Vec<&str> = command.split_whitespace().collect();
    for (index, token) in tokens.iter().enumerate() {
        if token.trim_matches(['"', '\'']).rsplit('/').next() != Some("xargs") {
            continue;
        }
        let mut probe = index + 1;
        while probe < tokens.len() && tokens[probe].starts_with('-') {
            probe += 1;
        }
        if probe < tokens.len() {
            let cmd = tokens[probe]
                .trim_matches(['"', '\''])
                .rsplit('/')
                .next()
                .unwrap_or(tokens[probe]);
            if matches!(cmd, "grep" | "rg") {
                return true;
            }
        }
    }
    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ShellSearchStream {
    FileContents,
    FilePaths,
    CommandOutput,
    None,
}

fn shell_search_stream(command: &str) -> ShellSearchStream {
    let lower = command.to_ascii_lowercase();
    if xargs_content_grep(&lower)
        || lower.contains(" -exec grep")
        || lower.contains(" -exec rg")
        || lower.contains("git grep")
        || lower.contains("grep -r")
        || lower
            .split([';', '\n', '|'])
            .any(|segment| strip_harmless_shell_prefixes(segment).starts_with("rg "))
    {
        return ShellSearchStream::FileContents;
    }
    if lower.contains("find ") && lower.contains("| grep") {
        return ShellSearchStream::FilePaths;
    }
    if lower
        .split([';', '\n', '|'])
        .any(|segment| strip_harmless_shell_prefixes(segment).starts_with("grep "))
    {
        return ShellSearchStream::FileContents;
    }
    if lower.contains("| grep") || lower.contains("| rg") {
        return ShellSearchStream::CommandOutput;
    }
    ShellSearchStream::None
}

fn shell_builtin_inspection_guard(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    grep_count_like(&lower)
        || shell_search_has_exact_file_scope(command)
        || shell_search_targets_config_or_docs(&lower)
}

fn grep_count_like(lower_command: &str) -> bool {
    lower_command
        .split([';', '\n'])
        .any(|segment| segment.contains("grep -c ") || segment.contains("rg -c "))
}

fn shell_search_has_exact_file_scope(command: &str) -> bool {
    command.split_whitespace().any(|token| {
        let cleaned =
            token.trim_matches(|ch: char| matches!(ch, '"' | '\'' | ';' | '|' | ')' | '(' | ','));
        !cleaned.contains('*') && is_source_like_file(cleaned)
    })
}

fn shell_search_targets_config_or_docs(lower_command: &str) -> bool {
    lower_command
        .split_whitespace()
        .map(clean_shell_path_token)
        .any(|token| shell_path_token_targets_config_or_docs(&token))
}

fn clean_shell_path_token(token: &str) -> String {
    token
        .trim_matches(|ch: char| {
            matches!(
                ch,
                '"' | '\'' | ';' | '|' | ')' | '(' | ',' | ':' | '[' | ']'
            )
        })
        .to_string()
}

fn shell_path_token_targets_config_or_docs(token: &str) -> bool {
    if token.is_empty() {
        return false;
    }
    let file_name = token.rsplit('/').next().unwrap_or(token);
    file_name == "readme"
        || file_name.starts_with("readme.")
        || (file_name.starts_with('.') && file_name.ends_with("opts"))
        || text_like_extension(file_name)
}

fn shell_symbol_search_like(command: &str) -> bool {
    let Some(pattern) = shell_search_pattern(command) else {
        return false;
    };
    if !command_has_source_scope(command) {
        return false;
    }
    !matches!(
        grep_pattern_kind(&pattern, "", command),
        GrepPatternKind::ExactImportOrInclude
            | GrepPatternKind::ErrorMessageOrLogText
            | GrepPatternKind::WireKeyOrSerializationKey
            | GrepPatternKind::HeaderOrProtocolLiteral
            | GrepPatternKind::NumericOrCodepointLiteral
            | GrepPatternKind::TestTextSearch
            | GrepPatternKind::CodeIdiom
    ) && (symbol_like_pattern(&pattern)
        || matches!(
            grep_pattern_kind(&pattern, "", command),
            GrepPatternKind::DeclarationForm
                | GrepPatternKind::SymbolCallOrMember
                | GrepPatternKind::SymbolFamilyOrRelationship
        ))
}

fn strong_shell_source_symbol_search(command: &str) -> Option<RecommendedTool> {
    if shell_semantics_required(command) {
        return None;
    }
    if script_recursive_constructor_or_call_search(command) {
        return Some(RecommendedTool::ScanUsages);
    }
    if shell_search_stream(command) != ShellSearchStream::FileContents {
        return None;
    }
    let pattern = shell_search_pattern(command)?;
    if !command_has_source_scope(command) || shell_builtin_inspection_guard(command) {
        return None;
    }
    let kind = grep_pattern_kind(&pattern, "", command);
    if shell_source_search_builtin_exception(&pattern, kind) {
        return None;
    }
    if declaration_search_pattern(&pattern) {
        return Some(RecommendedTool::SearchSymbols);
    }
    if shell_call_or_member_search(&pattern) {
        return Some(RecommendedTool::ScanUsages);
    }
    if strong_symbol_discovery_pattern(&pattern) {
        return Some(RecommendedTool::SearchSymbols);
    }
    None
}

fn shell_source_search_builtin_exception(pattern: &str, kind: GrepPatternKind) -> bool {
    let tokens = identifier_tokens(pattern);
    if kind == GrepPatternKind::AssertionOrSerializationText
        && tokens
            .iter()
            .all(|token| short_prefixed_source_symbol_like(token))
    {
        return false;
    }
    matches!(
        kind,
        GrepPatternKind::ExactImportOrInclude
            | GrepPatternKind::ErrorMessageOrLogText
            | GrepPatternKind::WireKeyOrSerializationKey
            | GrepPatternKind::HeaderOrProtocolLiteral
            | GrepPatternKind::NumericOrCodepointLiteral
            | GrepPatternKind::TestTextSearch
            | GrepPatternKind::CodeIdiom
            | GrepPatternKind::PackageDeclarationLiteral
            | GrepPatternKind::NaturalLanguageOrPathLiteral
            | GrepPatternKind::AssertionOrSerializationText
    ) || mixed_env_literal_search(pattern)
}

fn shell_call_or_member_search(pattern: &str) -> bool {
    (pattern.contains("\\(")
        || pattern.contains('(')
        || pattern.contains("::")
        || pattern.contains("->"))
        && symbol_like_pattern(pattern)
}

fn strong_symbol_discovery_pattern(pattern: &str) -> bool {
    let tokens = identifier_tokens(pattern);
    if tokens.len() == 1 {
        return symbol_like_pattern(pattern);
    }
    if tokens.len() >= 3 && tokens.iter().any(|token| symbol_like_pattern(token)) {
        return true;
    }
    if tokens.len() >= 2 && tokens.iter().all(|token| symbol_like_pattern(token)) {
        return true;
    }
    tokens.len() >= 2
        && tokens
            .iter()
            .all(|token| short_prefixed_source_symbol_like(token))
}

fn mixed_env_literal_search(pattern: &str) -> bool {
    pattern.contains("USERPROFILE") || pattern.contains("TMP")
}

fn ambiguous_source_shell_search(command: &str) -> bool {
    let Some(pattern) = shell_search_pattern(command) else {
        return false;
    };
    if !command_has_source_scope(command) || shell_builtin_inspection_guard(command) {
        return false;
    }
    let kind = grep_pattern_kind(&pattern, "", command);
    if matches!(
        kind,
        GrepPatternKind::ExactImportOrInclude
            | GrepPatternKind::ErrorMessageOrLogText
            | GrepPatternKind::WireKeyOrSerializationKey
            | GrepPatternKind::HeaderOrProtocolLiteral
            | GrepPatternKind::NumericOrCodepointLiteral
            | GrepPatternKind::NamingVariantTextSearch
            | GrepPatternKind::TestTextSearch
            | GrepPatternKind::CodeIdiom
    ) {
        return false;
    }
    symbol_like_pattern(&pattern)
        || matches!(
            kind,
            GrepPatternKind::SymbolCallOrMember | GrepPatternKind::SymbolFamilyOrRelationship
        )
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

fn is_source_like_file(path: &str) -> bool {
    SOURCE_EXTENSIONS.iter().any(|ext| path.ends_with(ext))
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GrepPatternKind {
    DeclarationForm,
    SymbolCallOrMember,
    SymbolFamilyOrRelationship,
    CodeIdiom,
    ExactMemberChainOrCallText,
    AssertionOrSerializationText,
    NaturalLanguageOrPathLiteral,
    UppercaseErrorConstantSet,
    PackageDeclarationLiteral,
    ExactImportOrInclude,
    ErrorMessageOrLogText,
    WireKeyOrSerializationKey,
    HeaderOrProtocolLiteral,
    NumericOrCodepointLiteral,
    NamingVariantTextSearch,
    TestTextSearch,
    Unknown,
}

fn grep_pattern_kind(pattern: &str, glob: &str, path: &str) -> GrepPatternKind {
    let trimmed = pattern.trim_matches(['"', '\'']).trim();
    if trimmed.is_empty() {
        return GrepPatternKind::Unknown;
    }
    if declaration_search_pattern(trimmed) {
        return GrepPatternKind::DeclarationForm;
    }

    let lower = trimmed.to_ascii_lowercase();
    if exact_import_or_include_pattern(&lower) {
        return GrepPatternKind::ExactImportOrInclude;
    }
    if package_declaration_literal(trimmed) {
        return GrepPatternKind::PackageDeclarationLiteral;
    }
    if uppercase_error_constant_set(trimmed) {
        return GrepPatternKind::UppercaseErrorConstantSet;
    }
    if natural_language_or_path_literal(trimmed) {
        return GrepPatternKind::NaturalLanguageOrPathLiteral;
    }
    if assertion_or_serialization_text(trimmed) {
        return GrepPatternKind::AssertionOrSerializationText;
    }
    if exact_member_chain_or_call_text(trimmed) {
        return GrepPatternKind::ExactMemberChainOrCallText;
    }
    if numeric_or_codepoint_literal(trimmed) {
        return GrepPatternKind::NumericOrCodepointLiteral;
    }
    if header_or_protocol_literal(trimmed, &lower) {
        return GrepPatternKind::HeaderOrProtocolLiteral;
    }
    if error_or_log_text_pattern(&lower, trimmed) {
        return GrepPatternKind::ErrorMessageOrLogText;
    }
    if naming_variant_search(trimmed) {
        return GrepPatternKind::NamingVariantTextSearch;
    }
    if wire_key_or_serialization_key(trimmed) {
        return GrepPatternKind::WireKeyOrSerializationKey;
    }
    if localized_code_idiom(trimmed) {
        return GrepPatternKind::CodeIdiom;
    }
    if test_scope(glob, path) && test_text_search(trimmed) {
        return GrepPatternKind::TestTextSearch;
    }
    if symbol_call_or_member(trimmed) {
        return GrepPatternKind::SymbolCallOrMember;
    }
    if symbol_family_or_relationship(trimmed) {
        return GrepPatternKind::SymbolFamilyOrRelationship;
    }
    if test_scope(glob, path) && !symbol_like_pattern(trimmed) {
        return GrepPatternKind::TestTextSearch;
    }
    GrepPatternKind::Unknown
}

fn exact_import_or_include_pattern(lower: &str) -> bool {
    lower.starts_with("import")
        || lower.contains(" import")
        || lower.starts_with("#include")
        || lower.contains("#include")
}

fn numeric_or_codepoint_literal(pattern: &str) -> bool {
    let stripped: String = pattern
        .chars()
        .filter(|ch| ch.is_ascii_hexdigit() || *ch == 'x' || *ch == 'X')
        .collect();
    stripped.len() >= 3
        && stripped.len() == pattern.chars().filter(|ch| !ch.is_whitespace()).count()
}

fn header_or_protocol_literal(pattern: &str, lower: &str) -> bool {
    ["csrf", "sec-fetch", "content-type", "accept-"]
        .iter()
        .any(|needle| lower.contains(needle))
        || (lower.contains("http") && pattern.chars().all(|ch| !ch.is_ascii_uppercase()))
        || (lower.contains("header") && pattern.contains('-'))
}

fn error_or_log_text_pattern(lower: &str, pattern: &str) -> bool {
    if all_same_prefixed_error_constants(pattern) {
        return false;
    }
    contains_text_word(lower)
        && (pattern.contains(' ')
            || pattern.contains('|')
            || pattern.contains('"')
            || pattern.contains('\'')
            || pattern.contains('[')
            || pattern.contains(']'))
}

fn wire_key_or_serialization_key(pattern: &str) -> bool {
    let tokens = identifier_tokens(pattern);
    !tokens.is_empty()
        && tokens.iter().all(|token| {
            lowercase_wire_key_like(token)
                && !short_prefixed_source_symbol_like(token)
                && !token.contains("__")
                && !token.ends_with("_t")
        })
}

fn naming_variant_search(pattern: &str) -> bool {
    if pattern.contains("[_ ]") || pattern.contains("[_\\s]") {
        return true;
    }
    let parts: Vec<_> = pattern
        .split('|')
        .map(|part| part.trim_matches(['"', '\'']).trim())
        .filter(|part| !part.is_empty())
        .collect();
    if parts.len() < 2 {
        return false;
    }
    let mut normalized = std::collections::HashSet::new();
    parts.iter().any(|part| {
        let key: String = part
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .flat_map(|ch| ch.to_lowercase())
            .collect();
        !key.is_empty() && !normalized.insert(key)
    })
}

fn localized_code_idiom(pattern: &str) -> bool {
    let lower = pattern.to_ascii_lowercase();
    (pattern.contains("\\.") && (pattern.contains("\\(") || pattern.contains("\\.(")))
        && (lower.contains("\\(async") || lower.contains("(async") || lower.contains("\\.("))
}

fn regex_text_semantics_material(pattern: &str) -> bool {
    [".*", "[^", "\\s", "\\d", "^", "$"]
        .iter()
        .any(|needle| pattern.contains(needle))
}

fn package_declaration_literal(pattern: &str) -> bool {
    let lower = pattern.to_ascii_lowercase();
    (lower.starts_with("package ") || lower.contains("|package ") || lower.contains("package\\s"))
        && (pattern.contains("\\.") || pattern.contains('.'))
}

fn uppercase_error_constant_set(pattern: &str) -> bool {
    let tokens = identifier_tokens(pattern);
    tokens.len() >= 2 && tokens.iter().all(|token| uppercase_constant_like(token))
}

fn mixed_error_constant_text_search(pattern: &str) -> bool {
    let tokens = identifier_tokens(pattern);
    let uppercase_count = tokens
        .iter()
        .filter(|token| uppercase_constant_like(token))
        .count();
    uppercase_count >= 2
        && tokens.iter().any(|token| {
            let lower = token.to_ascii_lowercase();
            lower.contains("err") || lower.contains("error") || lower.contains("errno")
        })
}

fn natural_language_or_path_literal(pattern: &str) -> bool {
    let lower = pattern.to_ascii_lowercase();
    let tokens = identifier_tokens(pattern);
    lower.contains("drive root")
        || lower.contains("absolute")
        || lower.contains("\\\\\\\\")
        || (pattern.contains(' ') && tokens.len() >= 3)
        || (pattern.contains(' ') && tokens.iter().all(|token| !symbol_like_pattern(token)))
}

fn config_or_build_text_scope(glob: &str, path: &str, file_path: &str) -> bool {
    let scope = normalized_grep_scope(glob, path, file_path).to_ascii_lowercase();
    [
        "makefile",
        "dockerfile",
        "build.sbt",
        "pom.xml",
        "package.json",
        "composer.json",
        "requirements.txt",
        "go.mod",
        "cargo.toml",
        ".github/",
        ".yaml",
        ".yml",
        ".toml",
        ".json",
        ".md",
        ".txt",
        ".csproj",
        ".fsproj",
        ".vbproj",
        ".sln",
        ".proj",
        "project/",
    ]
    .iter()
    .any(|needle| scope.contains(needle))
}

fn diagnostic_or_status_constant_text(pattern: &str) -> bool {
    let tokens = identifier_tokens(pattern);
    !tokens.is_empty()
        && tokens.iter().any(|token| {
            let upper = token.to_ascii_uppercase();
            upper.starts_with("ERROR_")
                || upper.starts_with("ERR_")
                || upper.starts_with("UV_E")
                || (token.len() >= 5
                    && token.starts_with('E')
                    && token
                        .chars()
                        .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_'))
                || upper.contains("_ERROR")
                || upper.contains("EACCES")
                || upper.contains("EINVAL")
        })
        && tokens.iter().all(|token| {
            uppercase_constant_like(token)
                || token
                    .chars()
                    .all(|ch| ch.is_ascii_lowercase() || ch == '_' || ch.is_ascii_digit())
        })
}

fn import_api_or_operator_regex_text(pattern: &str) -> bool {
    let lower = pattern.to_ascii_lowercase();
    lower.contains('/')
        || pattern.contains("\\.")
        || pattern.contains("\\+")
        || pattern.contains("\\\"")
        || pattern.contains('"')
        || pattern.contains("\\s*\"")
        || pattern.contains("::")
        || pattern.contains("->")
}

fn concrete_code_shape_pattern(pattern: &str) -> bool {
    let lower = pattern.to_ascii_lowercase();
    import_api_or_operator_regex_text(pattern)
        || assertion_or_build_string_regex_text(pattern)
        || lower.contains("projectreference")
        || lower.contains("methodsource")
        || lower.contains("scheme.")
        || lower.contains("startobject")
        || pattern.contains("\\s*=")
        || pattern.contains(".*=")
        || pattern.contains("\"")
        || pattern.contains("'")
}

fn assertion_or_serialization_text(pattern: &str) -> bool {
    let lower = pattern.to_ascii_lowercase();
    lower.contains("shouldbe")
        || lower.contains("should be")
        || lower.contains(" should")
        || lower.contains("\\.string")
        || lower.contains(".string")
        || mixed_lower_snake_serialization_or_config_search(pattern)
}

fn assertion_or_build_string_regex_text(pattern: &str) -> bool {
    let lower = pattern.to_ascii_lowercase();
    assertion_or_serialization_text(pattern)
        && regex_text_semantics_material(pattern)
        && (lower.contains("should")
            || lower.contains("\\.string")
            || lower.contains(".string")
            || lower.contains("\\.build")
            || lower.contains(".build"))
}

fn mixed_lower_snake_serialization_or_config_search(pattern: &str) -> bool {
    let lower = pattern.to_ascii_lowercase();
    let has_serialization_context = [
        "json",
        "yaml",
        "toml",
        "xml",
        "config",
        "setting",
        "field",
        "key",
        "value",
        "param",
        "string",
        "serde",
        "serializ",
        "deserializ",
    ]
    .iter()
    .any(|needle| lower.contains(needle));
    if !has_serialization_context {
        return false;
    }
    let tokens = identifier_tokens(pattern);
    let lower_snake_count = tokens
        .iter()
        .filter(|token| lowercase_wire_key_like(token) && !short_prefixed_source_symbol_like(token))
        .count();
    lower_snake_count > 0 && tokens.len() >= 2
}

fn exact_member_chain_or_call_text(pattern: &str) -> bool {
    if pattern.contains(".*") || pattern.contains('|') || declaration_search_pattern(pattern) {
        return false;
    }
    let tokens = identifier_tokens(pattern);
    tokens.len() >= 2
        && (pattern.contains("\\.") || pattern.contains('.'))
        && tokens.iter().all(|token| symbol_like_pattern(token))
}

fn test_text_search(pattern: &str) -> bool {
    let lower = pattern.to_ascii_lowercase();
    if contains_text_word(&lower)
        || header_or_protocol_literal(pattern, &lower)
        || numeric_or_codepoint_literal(pattern)
        || naming_variant_search(pattern)
        || wire_key_or_serialization_key(pattern)
    {
        return true;
    }
    let tokens = identifier_tokens(pattern);
    if tokens.len() >= 2 && tokens.iter().all(|token| !symbol_like_pattern(token)) {
        return true;
    }
    tokens.len() >= 2
        && tokens
            .iter()
            .all(|token| !token.chars().any(|ch| ch.is_ascii_uppercase()))
        && tokens.iter().any(|token| !symbol_like_pattern(token))
        && (pattern.contains('|') || pattern.contains(' '))
}

fn symbol_call_or_member(pattern: &str) -> bool {
    if localized_code_idiom(pattern) {
        return false;
    }
    let tokens = identifier_tokens(pattern);
    (pattern.contains("\\(") || pattern.contains("\\s*\\(") || pattern.contains("::"))
        && tokens
            .iter()
            .any(|token| symbol_like_pattern(token) && !lowercase_wire_key_like(token))
}

fn symbol_family_or_relationship(pattern: &str) -> bool {
    let tokens = identifier_tokens(pattern);
    if tokens.is_empty() {
        return false;
    }
    if pattern.contains(".*") && tokens.iter().any(|token| symbol_like_pattern(token)) {
        return true;
    }
    if pattern.contains('|') {
        let parts: Vec<_> = pattern
            .split('|')
            .map(|part| identifier_tokens(part))
            .filter(|part_tokens| !part_tokens.is_empty())
            .collect();
        if parts.len() >= 2
            && parts.iter().all(|part_tokens| {
                part_tokens
                    .iter()
                    .all(|token| symbol_like_pattern(token) && !uppercase_constant_like(token))
            })
            && !naming_variant_search(pattern)
        {
            return true;
        }
    }
    tokens.len() == 1 && symbol_like_pattern(&tokens[0]) && !lowercase_wire_key_like(&tokens[0])
}

fn test_scope(glob: &str, path: &str) -> bool {
    let lower = format!("{} {}", glob, path).to_ascii_lowercase();
    lower.contains("test") || lower.contains("spec")
}

fn contains_text_word(lower: &str) -> bool {
    [
        "failed",
        "exception",
        "warning",
        "message",
        "invalid",
        "panic",
        "recover",
        "url",
    ]
    .iter()
    .any(|word| lower.contains(word))
        || lower
            .split(|ch: char| !ch.is_ascii_alphanumeric())
            .any(|word| matches!(word, "error" | "assert" | "log"))
}

fn declaration_search_pattern(pattern: &str) -> bool {
    let lower = pattern.to_ascii_lowercase();
    if lower.starts_with("#define ") || lower.contains("|#define ") {
        return true;
    }
    [
        "class ",
        "static class ",
        "case class ",
        "object ",
        "trait ",
        "sealed trait ",
        "interface ",
        "enum ",
        "typedef",
        "def ",
        "func ",
        "function ",
    ]
    .iter()
    .any(|needle| lower.starts_with(needle) || lower.contains(&format!("|{needle}")))
        || [
            "class\\s+",
            "static\\s+class\\s+",
            "case\\s+class\\s+",
            "object\\s+",
            "trait\\s+",
            "sealed\\s+trait\\s+",
            "interface\\s+",
            "enum\\s+",
            "def\\s+",
            "func\\s+",
            "function\\s+",
        ]
        .iter()
        .any(|needle| lower.starts_with(needle) || lower.contains(&format!("|{needle}")))
}

fn uppercase_constant_like(token: &str) -> bool {
    token.len() >= 3
        && token
            .chars()
            .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_')
        && token.chars().any(|ch| ch.is_ascii_uppercase())
}

fn lowercase_wire_key_like(token: &str) -> bool {
    token.len() >= 3
        && token.contains('_')
        && token
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
        && token.chars().any(|ch| ch.is_ascii_lowercase())
}

fn short_prefixed_source_symbol_like(token: &str) -> bool {
    token
        .split_once('_')
        .map(|(prefix, _)| prefix.len() <= 3)
        .unwrap_or(false)
}

fn all_same_prefixed_error_constants(pattern: &str) -> bool {
    let tokens = identifier_tokens(pattern);
    tokens.len() >= 2
        && tokens
            .iter()
            .all(|token| token.starts_with("ERROR_") && uppercase_constant_like(token))
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
        ".csproj",
        ".fsproj",
        ".vbproj",
        ".sln",
        ".proj",
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
        let gate = decision != "allow_text" && tool != "none";
        json!({
            "reason": reason,
            "intent": intent,
            "pattern_class": if gate { "identifier_like" } else { "literal_exact" },
            "scope_class": if gate { "broad_source_scope" } else { "narrowed_file_set" },
            "bifrost_fit": fit,
            "fallback_exception": exception,
            "evidence": {
                "symbol_tokens": ["Foo"],
                "same_token_or_path_bifrost_miss": false,
                "same_path_recent_edit_or_write": false,
                "same_path_recent_bifrost_hit": false,
                "exact_text_or_regex_needed": false
            },
            "bifrost_candidate": if gate { json!({"tool": tool, "args": {"patterns":["Foo"]}}) } else { Value::Null },
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
    fn request_schema_puts_reason_before_policy_facts() {
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({"pattern": "Foo"}),
            messages: vec![ChatMessage::user("Find Foo")],
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };
        let body = build_request_body("deepseek/deepseek-v4-flash", &context).unwrap();
        let reason_pos = body.find(r#""reason""#).unwrap();
        let intent_pos = body.find(r#""intent""#).unwrap();
        assert!(reason_pos < intent_pos, "{body}");
        assert!(!body.contains(r#""action": {"#), "{body}");
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
        assert!(body.contains("policy-fact extractor"));
        assert!(body.contains("Do not include an action field or final decision field"));
        assert!(body.contains("Concrete code shape is not, by itself, an allow-text reason"));
        assert!(body.contains("Treat concrete syntax as Bifrost symbol navigation"));
        assert!(body.contains("Do not use exact regex as an allow-text reason"));
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
            pattern_class: TextPatternClass::IdentifierLike,
            scope_class: TextScopeClass::BroadSourceScope,
            bifrost_fit: BifrostFit::SameOrMoreDirect,
            allow_exception: TextAllowException::None,
            evidence: TextEvidence {
                symbol_tokens: vec!["Foo".to_string()],
                same_token_or_path_bifrost_miss: false,
                same_path_recent_edit_or_write: false,
                same_path_recent_bifrost_hit: false,
                exact_text_or_regex_needed: false,
            },
            bifrost_candidate: Some(BifrostCandidate {
                tool: Some(RecommendedTool::SearchSymbols),
                args: json!({"patterns": ["Foo"]}),
            }),
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
    fn malformed_bifrost_candidate_falls_back_to_intent_tool() {
        let envelope = json!({
            "choices": [{
                "message": {
                    "content": json!({
                        "reason": "Broad source symbol grep should use Bifrost.",
                        "intent": "symbol_definition_lookup",
                        "pattern_class": "identifier_like",
                        "scope_class": "broad_source_scope",
                        "bifrost_fit": "same_or_more_direct",
                        "fallback_exception": "none",
                        "evidence": {
                            "symbol_tokens": ["Foo"],
                            "same_token_or_path_bifrost_miss": false,
                            "same_path_recent_edit_or_write": false,
                            "same_path_recent_bifrost_hit": false,
                            "exact_text_or_regex_needed": false
                        },
                        "bifrost_candidate": {"patterns": ["Foo"]},
                        "suggested_args": {},
                        "confidence": "high"
                    }).to_string()
                }
            }]
        });
        let parsed = parse_openrouter_response(&envelope.to_string()).unwrap();
        assert_eq!(parsed.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(parsed.recommended_tool, RecommendedTool::SearchSymbols);
    }

    #[test]
    fn invalid_bifrost_candidate_tool_falls_back_to_intent_tool() {
        let envelope = json!({
            "choices": [{
                "message": {
                    "content": json!({
                        "reason": "Broad source symbol grep should use Bifrost.",
                        "intent": "symbol_definition_lookup",
                        "pattern_class": "identifier_like",
                        "scope_class": "broad_source_scope",
                        "bifrost_fit": "same_or_more_direct",
                        "fallback_exception": "none",
                        "evidence": {
                            "symbol_tokens": ["Foo"],
                            "same_token_or_path_bifrost_miss": false,
                            "same_path_recent_edit_or_write": false,
                            "same_path_recent_bifrost_hit": false,
                            "exact_text_or_regex_needed": false
                        },
                        "bifrost_candidate": {"tool": "searsh_symbols", "args": {"patterns": ["Foo"]}},
                        "suggested_args": {},
                        "confidence": "high"
                    }).to_string()
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
    fn shell_action_wins_over_contradictory_reason_prefix() {
        let envelope = json!({
            "choices": [{
                "message": {
                    "content": shell_output_json("USE_BUILTIN_TOOL because prose is contradictory.", "build_test_git_package_or_project_cli", false, false, "not_applicable", "build_test_git_package_or_project_cli", "allow_shell_shell_semantics", "allow_shell", "read_file")
                }
            }]
        });
        let parsed = parse_shell_openrouter_response(&envelope.to_string()).unwrap();
        assert_eq!(parsed.decision, ShellClassifierDecision::AllowShell);
        assert_eq!(parsed.recommended_tool, RecommendedTool::None);
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
    fn policy_facts_are_operational_source_of_truth() {
        let envelope = json!({
            "choices": [{
                "message": {
                    "content": text_output_json("GATE_TO_SYMBOL_TOOL because this grep is looking for callers.", "symbol_reference_lookup", "not_applicable", "non_source_text", "gate_to_symbol_tool", "scan_usages")
                }
            }]
        });
        let parsed = parse_openrouter_response(&envelope.to_string()).unwrap();
        assert_eq!(parsed.decision, GateClassifierDecision::AllowText);
        assert_eq!(parsed.recommended_tool, RecommendedTool::None);
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
                        "intent": "symbol_definition_lookup",
                        "pattern_class": "identifier_like",
                        "scope_class": "broad_source_scope",
                        "bifrost_fit": "same_or_more_direct",
                        "fallback_exception": "none",
                        "evidence": {
                            "symbol_tokens": ["Foo"],
                            "same_token_or_path_bifrost_miss": false,
                            "same_path_recent_edit_or_write": false,
                            "same_path_recent_bifrost_hit": false,
                            "exact_text_or_regex_needed": false
                        },
                        "bifrost_candidate": {"tool": "search_symbols", "args": {"patterns":["Foo"]}},
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
    fn broad_symbol_grep_with_invalid_allow_facts_is_not_repaired_to_gate() {
        let mut output = GateClassifierOutput {
            reason: "ALLOW_TEXT because regex can find the identifier.".to_string(),
            intent: TextIntent::SymbolDefinitionLookup,
            pattern_class: TextPatternClass::IdentifierLike,
            scope_class: TextScopeClass::BroadSourceScope,
            bifrost_fit: BifrostFit::Unknown,
            allow_exception: TextAllowException::ExactLiteralOrRegex,
            evidence: TextEvidence {
                symbol_tokens: vec!["RichEditBoxDefaultLineEnding".to_string()],
                same_token_or_path_bifrost_miss: false,
                same_path_recent_edit_or_write: false,
                same_path_recent_bifrost_hit: false,
                exact_text_or_regex_needed: true,
            },
            bifrost_candidate: None,
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
    fn symbol_usage_intent_recommends_scan_usages() {
        assert_eq!(
            bifrost_tool_for_text_intent(&TextIntent::SymbolUsageLookup),
            RecommendedTool::ScanUsages
        );
    }

    #[test]
    fn scala_block_call_grep_repairs_to_scan_usages() {
        let mut output = low_confidence_allow_output(TextIntent::SymbolUsageLookup);
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({"pattern": "effectAsync\\s*\\{", "path": ".", "glob": "*.scala"}),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(output.recommended_tool, RecommendedTool::ScanUsages);
    }

    fn low_confidence_allow_output(intent: TextIntent) -> GateClassifierOutput {
        GateClassifierOutput {
            reason: "ALLOW_TEXT because classifier was uncertain.".to_string(),
            intent,
            pattern_class: TextPatternClass::SymbolGlob,
            scope_class: TextScopeClass::BroadSourceScope,
            bifrost_fit: BifrostFit::SameOrMoreDirect,
            allow_exception: TextAllowException::None,
            evidence: TextEvidence {
                symbol_tokens: vec!["Foo".to_string()],
                same_token_or_path_bifrost_miss: false,
                same_path_recent_edit_or_write: false,
                same_path_recent_bifrost_hit: false,
                exact_text_or_regex_needed: false,
            },
            bifrost_candidate: None,
            decision: GateClassifierDecision::AllowText,
            recommended_tool: RecommendedTool::None,
            suggested_args: json!({}),
            confidence: GateConfidence::Low,
        }
    }

    #[test]
    fn low_confidence_symbol_usage_grep_repairs_to_scan_usages() {
        let mut output = low_confidence_allow_output(TextIntent::SymbolUsageLookup);
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({"pattern": "RenameFileAsync\\s*\\(", "path": "src/Notepads", "glob": "*.cs"}),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(output.recommended_tool, RecommendedTool::ScanUsages);
    }

    #[test]
    fn low_confidence_broad_declaration_grep_repairs_to_bifrost() {
        let mut output = low_confidence_allow_output(TextIntent::SymbolDefinitionLookup);
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({
                "pattern": "case class ElisionTokenFilter|object ElisionTokenFilter|trait ElisionTokenFilter|ElisionTokenFilter\\(",
                "path": "elastic4s-core",
                "glob": "**/*.scala"
            }),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(output.recommended_tool, RecommendedTool::SearchSymbols);
    }

    #[test]
    fn broad_source_scope_repairs_surviving_literal_claims() {
        let cases = [
            (
                "class\\s+Plugin|abstract class\\s+Plugin|virtual\\s+void\\s+OnGameSelected|OnControllerButtonStateChanged|OnFullscreenViewChanged|FullscreenView|GameDetailsVisible|FullscreenAppViewModel",
                "source",
                "*.cs",
                TextIntent::BroadSemanticOrientation,
                RecommendedTool::GetSummaries,
            ),
            (
                "GameDetailsVisible =",
                "source",
                "*.cs",
                TextIntent::SymbolUsageLookup,
                RecommendedTool::ScanUsages,
            ),
            (
                "object|class|def",
                "elastic4s-client-http4s/src",
                "*.scala",
                TextIntent::BroadSemanticOrientation,
                RecommendedTool::GetSummaries,
            ),
        ];
        for (pattern, path, glob, intent, expected_tool) in cases {
            let mut output = low_confidence_allow_output(intent);
            let context = GateContext {
                tool_name: "grep_search".to_string(),
                args: json!({"pattern": pattern, "path": path, "glob": glob}),
                messages: Vec::new(),
                tools: Vec::new(),
                tool_exchanges: Vec::new(),
            };

            enforce_text_classifier_policy(&mut output, &context);

            assert_eq!(output.decision, GateClassifierDecision::GateToSymbolTool);
            assert_eq!(output.recommended_tool, expected_tool);
        }
    }

    #[test]
    fn low_confidence_exact_file_grep_stays_allowed() {
        let mut output = low_confidence_allow_output(TextIntent::SymbolUsageLookup);
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({"pattern": "RenameFileAsync\\s*\\(", "path": ".", "glob": "src/Notepads/App.xaml.cs"}),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::AllowText);
        assert_eq!(output.recommended_tool, RecommendedTool::None);

        let mut output = GateClassifierOutput {
            reason: "Bifrost can find this C# attribute symbol.".to_string(),
            intent: TextIntent::SymbolDefinitionLookup,
            pattern_class: TextPatternClass::SymbolGlob,
            scope_class: TextScopeClass::BroadSourceScope,
            bifrost_fit: BifrostFit::SameOrMoreDirect,
            allow_exception: TextAllowException::None,
            evidence: TextEvidence {
                symbol_tokens: vec!["InternalsVisibleTo".to_string()],
                same_token_or_path_bifrost_miss: false,
                same_path_recent_edit_or_write: false,
                same_path_recent_bifrost_hit: false,
                exact_text_or_regex_needed: false,
            },
            bifrost_candidate: Some(BifrostCandidate {
                tool: Some(RecommendedTool::SearchSymbols),
                args: json!({"patterns":["InternalsVisibleTo"]}),
            }),
            decision: GateClassifierDecision::GateToSymbolTool,
            recommended_tool: RecommendedTool::SearchSymbols,
            suggested_args: json!({"patterns":["InternalsVisibleTo"]}),
            confidence: GateConfidence::High,
        };
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({
                "path": "source",
                "glob": "source/**/*.cs",
                "pattern": "InternalsVisibleTo"
            }),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::AllowText);
        assert_eq!(output.recommended_tool, RecommendedTool::None);
    }

    #[test]
    fn low_confidence_text_literal_grep_stays_allowed() {
        let mut output = low_confidence_allow_output(TextIntent::SymbolDefinitionLookup);
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({"pattern": "import java.util.List", "path": "src", "glob": "*.java"}),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::AllowText);
        assert_eq!(output.recommended_tool, RecommendedTool::None);
    }

    #[test]
    fn mixed_phrase_and_symbol_grep_stays_text() {
        let mut output = low_confidence_allow_output(TextIntent::SymbolDefinitionLookup);
        let mut hit = exchange("search_symbols");
        hit.arguments = r#"{"patterns":["checkWordOccurrence"]}"#.to_string();
        hit.result = "checkWordOccurrence src/words.js".to_string();
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({
                "pattern": "word occurrence|checkWordOccurrence|CheckWordOccurrence|occurrence",
                "path": ".",
                "glob": "**/*.{js,ts}"
            }),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: vec![hit],
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::AllowText);
        assert_eq!(output.recommended_tool, RecommendedTool::None);
    }

    #[test]
    fn same_direct_regex_candidate_routes_to_bifrost() {
        let mut output = GateClassifierOutput {
            reason: "Source call-site regex has a same-direct Bifrost candidate.".to_string(),
            intent: TextIntent::SymbolUsageLookup,
            pattern_class: TextPatternClass::RegexText,
            scope_class: TextScopeClass::BroadSourceScope,
            bifrost_fit: BifrostFit::SameOrMoreDirect,
            allow_exception: TextAllowException::None,
            evidence: TextEvidence {
                symbol_tokens: vec!["effectAsync".to_string()],
                same_token_or_path_bifrost_miss: false,
                same_path_recent_edit_or_write: false,
                same_path_recent_bifrost_hit: false,
                exact_text_or_regex_needed: false,
            },
            bifrost_candidate: Some(BifrostCandidate {
                tool: Some(RecommendedTool::ScanUsages),
                args: json!({"symbols":["effectAsync"]}),
            }),
            decision: GateClassifierDecision::AllowText,
            recommended_tool: RecommendedTool::None,
            suggested_args: json!({}),
            confidence: GateConfidence::High,
        };
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({"pattern": "effectAsync\\s*\\{", "path": ".", "glob": "*.scala"}),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(output.recommended_tool, RecommendedTool::ScanUsages);
    }

    #[test]
    fn same_direct_candidate_with_allow_exception_stays_text() {
        let mut output = GateClassifierOutput {
            reason: "Bare variant probe has an explicit allow exception.".to_string(),
            intent: TextIntent::BroadSemanticOrientation,
            pattern_class: TextPatternClass::SymbolGlob,
            scope_class: TextScopeClass::BroadSourceScope,
            bifrost_fit: BifrostFit::SameOrMoreDirect,
            allow_exception: TextAllowException::NonSourceText,
            evidence: TextEvidence {
                symbol_tokens: vec!["effectAsyncM".to_string()],
                same_token_or_path_bifrost_miss: false,
                same_path_recent_edit_or_write: false,
                same_path_recent_bifrost_hit: false,
                exact_text_or_regex_needed: false,
            },
            bifrost_candidate: Some(BifrostCandidate {
                tool: Some(RecommendedTool::ScanUsages),
                args: json!({"symbols":["effectAsyncM"]}),
            }),
            decision: GateClassifierDecision::AllowText,
            recommended_tool: RecommendedTool::None,
            suggested_args: json!({}),
            confidence: GateConfidence::High,
        };
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({"pattern": "effectAsyncM", "path": ".", "glob": "*.scala"}),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::AllowText);
        assert_eq!(output.recommended_tool, RecommendedTool::None);
    }

    #[test]
    fn low_confidence_post_edit_verification_grep_stays_allowed() {
        let mut output = low_confidence_allow_output(TextIntent::SymbolUsageLookup);
        let mut edit = exchange("edit");
        edit.arguments = r#"{"file_path":"src/Notepads/Main.cs"}"#.to_string();
        edit.result = "ok".to_string();
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({"pattern": "Handle\\(", "path": "src/Notepads", "glob": "*.cs"}),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: vec![edit],
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::AllowText);
        assert_eq!(output.recommended_tool, RecommendedTool::None);
    }

    #[test]
    fn low_confidence_bifrost_miss_grep_stays_allowed() {
        let mut output = low_confidence_allow_output(TextIntent::SymbolDefinitionLookup);
        let mut miss = exchange("search_symbols");
        miss.arguments = r#"{"patterns":["FooSymbol"]}"#.to_string();
        miss.result = "No symbols found for FooSymbol".to_string();
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({"pattern": "FooSymbol", "path": "src", "glob": "**/*.rs"}),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: vec![miss],
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::AllowText);
        assert_eq!(output.recommended_tool, RecommendedTool::None);
    }

    #[test]
    fn recent_naming_variant_grep_stays_text() {
        let mut output = GateClassifierOutput {
            reason: "GATE_TO_SYMBOL_TOOL because property references are source navigation.".to_string(),
            intent: TextIntent::SymbolReferenceLookup,
            pattern_class: TextPatternClass::IdentifierLike,
            scope_class: TextScopeClass::BroadSourceScope,
            bifrost_fit: BifrostFit::SameOrMoreDirect,
            allow_exception: TextAllowException::None,
            evidence: TextEvidence::default(),
            bifrost_candidate: Some(BifrostCandidate {
                tool: Some(RecommendedTool::ScanUsages),
                args: json!({"symbols":["callID"]}),
            }),
            decision: GateClassifierDecision::GateToSymbolTool,
            recommended_tool: RecommendedTool::ScanUsages,
            suggested_args: json!({}),
            confidence: GateConfidence::High,
        };
        let mut first = exchange("grep_search");
        first.arguments =
            r#"{"pattern":"call(ID|Id|_id)","glob":"src/tools/delegate-task/**/*.ts","path":"."}"#
                .to_string();
        let mut second = exchange("grep_search");
        second.arguments =
            r#"{"pattern":"callId","glob":"src/tools/delegate-task/**/*.ts","path":"."}"#
                .to_string();
        let mut third = exchange("grep_search");
        third.arguments =
            r#"{"pattern":"callID","glob":"src/tools/delegate-task/**/*.ts","path":"."}"#
                .to_string();
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({
                "pattern": "callID\\W",
                "glob": "src/tools/delegate-task/**/*.ts",
                "path": "."
            }),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: vec![first, second, third],
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::AllowText);
        assert_eq!(output.recommended_tool, RecommendedTool::None);
    }

    #[test]
    fn lower_snake_same_symbol_hit_repairs_to_bifrost() {
        let mut output = low_confidence_allow_output(TextIntent::SymbolDefinitionLookup);
        output.confidence = GateConfidence::High;
        let mut hit = exchange("search_symbols");
        hit.arguments = r#"{"patterns":["mount_info_from_path"]}"#.to_string();
        hit.result = r#"{"files":[{"functions":[{"signature":"fn mount_info_from_path"}]}]}"#
            .to_string();
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({"pattern": "mount_info_from_path", "path": ".", "glob": "*.rs"}),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: vec![hit],
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(output.recommended_tool, RecommendedTool::SearchSymbols);
    }

    #[test]
    fn canary_text_hard_allow_patterns_stay_allowed() {
        let cases = [
            (
                "Unable to find encoder for unknown file extension",
                "src/**/*.php",
                ".",
            ),
            (
                "strings\\.|\\+\\s*\"|atomic\\.",
                "**/*.go",
                "modules/actions",
            ),
            ("^lint-go|golangci-lint", "Makefile", "."),
            ("TEST_PIPENAME", "*.h", "test"),
            ("UV_EINVAL|udp_try_send|try_send", "test/**/*.c", "."),
            ("ERROR_NOACCESS", "**/*", "."),
            ("ERROR_BUFFER_OVERFLOW", "**/*", "."),
            ("ENAMETOOLONG", "src/win/*.c", "."),
            ("TS_LOCAL_ADDR_PORT", "*.go", "."),
            ("POD_IPS", "*.go", "."),
            (
                "sync/atomic|atomic\\.AddInt64|atomic\\.LoadInt64\"",
                "**/*.go",
                "modules",
            ),
            (
                "permission_parser\\|parseRawPermissions\\|ExtractJobPermissions",
                "**/*_test.go",
                ".",
            ),
            ("Path.*Without|FileInfo.*M3U", "*.cs", "source"),
            ("OnesComplement", "**/*Test.java", "src/test/java"),
            ("MethodSource", "src/test/java/**/*.java", "."),
            ("InnerHit", "**/*Test*.scala", "."),
            (
                "canSkipStatus|status queue|QueueStatus|skip status|LoginFinished|LoggedIn",
                "**/*_test.go",
                ".",
            ),
        ];
        for (pattern, glob, path) in cases {
            let context = GateContext {
                tool_name: "grep_search".to_string(),
                args: json!({"pattern": pattern, "glob": glob, "path": path}),
                messages: Vec::new(),
                tools: Vec::new(),
                tool_exchanges: Vec::new(),
            };
            let mut output = GateClassifierOutput {
                reason: "GATE_TO_SYMBOL_TOOL because identifiers were present.".to_string(),
                intent: TextIntent::SymbolDefinitionLookup,
                pattern_class: TextPatternClass::SymbolGlob,
                scope_class: TextScopeClass::BroadSourceScope,
                bifrost_fit: BifrostFit::SameOrMoreDirect,
                allow_exception: TextAllowException::None,
                evidence: TextEvidence::default(),
                bifrost_candidate: None,
                decision: GateClassifierDecision::GateToSymbolTool,
                recommended_tool: RecommendedTool::SearchSymbols,
                suggested_args: json!({}),
                confidence: GateConfidence::High,
            };

            enforce_text_classifier_policy(&mut output, &context);

            assert_eq!(
                output.decision,
                GateClassifierDecision::AllowText,
                "{pattern}"
            );
            assert_eq!(output.recommended_tool, RecommendedTool::None, "{pattern}");
        }
    }

    #[test]
    fn broad_source_naming_variants_repair_symbol_navigation() {
        let cases = [
            (
                json!({"pattern": "uv_udp_try_send\\|uv__udp_try_send", "glob": "**/*.c"}),
                GateClassifierDecision::GateToSymbolTool,
                RecommendedTool::SearchSymbols,
            ),
            (
                json!({"pattern": "Exiting\\s*\\+=", "glob": "*.cs", "path": "src/Notepads"}),
                GateClassifierDecision::AllowText,
                RecommendedTool::None,
            ),
            (
                json!({"pattern": "new Size\\(|Size::create\\(", "glob": "tests/**/*.php", "path": "."}),
                GateClassifierDecision::GateToSymbolTool,
                RecommendedTool::ScanUsages,
            ),
            (
                json!({"pattern": "NewInvalidArgumentError", "glob": "*.go", "path": "modules/util"}),
                GateClassifierDecision::GateToSymbolTool,
                RecommendedTool::SearchSymbols,
            ),
        ];
        for (args, expected_decision, expected_tool) in cases {
            let context = GateContext {
                tool_name: "grep_search".to_string(),
                args,
                messages: Vec::new(),
                tools: Vec::new(),
                tool_exchanges: Vec::new(),
            };
            let mut output = low_confidence_allow_output(TextIntent::SymbolDefinitionLookup);
            output.confidence = GateConfidence::High;

            enforce_text_classifier_policy(&mut output, &context);

            assert_eq!(output.decision, expected_decision);
            assert_eq!(output.recommended_tool, expected_tool);
        }
    }

    #[test]
    fn external_annotation_family_grep_stays_text() {
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({
                "pattern": "JsonProperty|JsonCreator|JsonIgnoreProperties|JsonInclude|JsonSetter",
                "glob": "*.scala",
                "path": "."
            }),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };
        let mut output = GateClassifierOutput {
            reason: "GATE_TO_SYMBOL_TOOL because annotations are source symbols.".to_string(),
            intent: TextIntent::SymbolUsageLookup,
            pattern_class: TextPatternClass::SymbolGlob,
            scope_class: TextScopeClass::BroadSourceScope,
            bifrost_fit: BifrostFit::SameOrMoreDirect,
            allow_exception: TextAllowException::None,
            evidence: TextEvidence {
                symbol_tokens: vec![
                    "JsonProperty".to_string(),
                    "JsonCreator".to_string(),
                    "JsonIgnoreProperties".to_string(),
                    "JsonInclude".to_string(),
                    "JsonSetter".to_string(),
                ],
                ..TextEvidence::default()
            },
            bifrost_candidate: Some(BifrostCandidate {
                tool: Some(RecommendedTool::GetSummaries),
                args: json!({"query":"JsonProperty|JsonCreator|JsonIgnoreProperties|JsonInclude|JsonSetter"}),
            }),
            decision: GateClassifierDecision::GateToSymbolTool,
            recommended_tool: RecommendedTool::GetSummaries,
            suggested_args: json!({}),
            confidence: GateConfidence::High,
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::AllowText);
        assert_eq!(output.recommended_tool, RecommendedTool::None);
    }

    #[test]
    fn compound_code_idiom_regex_stays_text() {
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({
                "pattern": "SetLen\\(unix\\.CmsgLen|CmsgSpace\\(2\\)|Cmsghdr",
                "glob": "*.go",
                "path": "."
            }),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };
        let mut output = GateClassifierOutput {
            reason: "GATE_TO_SYMBOL_TOOL because this includes call-shaped source symbols."
                .to_string(),
            intent: TextIntent::SymbolReferenceLookup,
            pattern_class: TextPatternClass::SymbolGlob,
            scope_class: TextScopeClass::RepositoryWide,
            bifrost_fit: BifrostFit::NotApplicable,
            allow_exception: TextAllowException::NonSourceText,
            evidence: TextEvidence {
                symbol_tokens: vec![
                    "SetLen".to_string(),
                    "CmsgLen".to_string(),
                    "CmsgSpace".to_string(),
                    "Cmsghdr".to_string(),
                ],
                ..TextEvidence::default()
            },
            bifrost_candidate: None,
            decision: GateClassifierDecision::GateToSymbolTool,
            recommended_tool: RecommendedTool::ScanUsages,
            suggested_args: json!({}),
            confidence: GateConfidence::High,
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::AllowText);
        assert_eq!(output.recommended_tool, RecommendedTool::None);
    }

    #[test]
    fn source_directory_symbol_rule_avoids_weak_literals() {
        for (pattern, path) in [
            ("Error", "modules/util"),
            ("Context", "modules/util"),
            ("String", "modules/util"),
            ("Size", "modules/util"),
            ("TODO", "modules/util"),
            ("Status", "modules/util"),
        ] {
            let context = GateContext {
                tool_name: "grep_search".to_string(),
                args: json!({"pattern": pattern, "glob": "", "path": path}),
                messages: Vec::new(),
                tools: Vec::new(),
                tool_exchanges: Vec::new(),
            };
            let mut output = GateClassifierOutput {
                reason: "ALLOW_TEXT because this is an exact local lookup.".to_string(),
                intent: TextIntent::ExactTextOrLocalizedRead,
                pattern_class: TextPatternClass::LiteralExact,
                scope_class: TextScopeClass::DirectoryOrGlob,
                bifrost_fit: BifrostFit::NotApplicable,
                allow_exception: TextAllowException::None,
                evidence: TextEvidence::default(),
                bifrost_candidate: None,
                decision: GateClassifierDecision::AllowText,
                recommended_tool: RecommendedTool::None,
                suggested_args: json!({}),
                confidence: GateConfidence::High,
            };

            enforce_text_classifier_policy(&mut output, &context);

            assert_eq!(
                output.decision,
                GateClassifierDecision::AllowText,
                "{pattern}"
            );
            assert_eq!(output.recommended_tool, RecommendedTool::None, "{pattern}");
        }

        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({"pattern": "RichEditBoxDefaultLineEnding", "glob": "", "path": "src/Notepads/Controls/TextEditor"}),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };
        let mut output = low_confidence_allow_output(TextIntent::SymbolDefinitionLookup);
        output.confidence = GateConfidence::High;
        enforce_text_classifier_policy(&mut output, &context);
        assert_eq!(output.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(output.recommended_tool, RecommendedTool::SearchSymbols);
    }

    #[test]
    fn shell_content_symbol_search_repairs_builtin_route_to_bifrost() {
        let mut output = ShellClassifierOutput {
            reason: "USE_BUILTIN_TOOL because grep_search can search text.".to_string(),
            intent: ShellIntent::SymbolDefinitionLookup,
            shell_semantics_required: false,
            builtin_preserves_intent: true,
            bifrost_fit: BifrostFit::NotApplicable,
            allow_exception: ShellAllowException::None,
            replacement_class: ShellReplacementClass::UseBuiltinInspection,
            decision: ShellClassifierDecision::UseBuiltinTool,
            recommended_tool: RecommendedTool::GrepSearch,
            suggested_args: json!({}),
            confidence: GateConfidence::High,
        };
        let context = GateContext {
            tool_name: "run_shell_command".to_string(),
            args: json!({"command": "rg -n \"RemoveExecutableNameOrPathFromCommandLineArgs|FileSystemUtility\" src"}),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_shell_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, ShellClassifierDecision::UseBifrostTool);
        assert_eq!(output.recommended_tool, RecommendedTool::SearchSymbols);
    }

    #[test]
    fn shell_php_method_search_repairs_builtin_route_to_scan_usages() {
        let mut output = ShellClassifierOutput {
            reason: "USE_BUILTIN_TOOL because rg can search text.".to_string(),
            intent: ShellIntent::LiteralTextSearch,
            shell_semantics_required: false,
            builtin_preserves_intent: true,
            bifrost_fit: BifrostFit::NotApplicable,
            allow_exception: ShellAllowException::None,
            replacement_class: ShellReplacementClass::UseBuiltinInspection,
            decision: ShellClassifierDecision::UseBuiltinTool,
            recommended_tool: RecommendedTool::GrepSearch,
            suggested_args: json!({}),
            confidence: GateConfidence::High,
        };
        let context = GateContext {
            tool_name: "run_shell_command".to_string(),
            args: json!({"command": "rg -n \"->alignment\\(|->setAlignment\\(|horizontalAlignment\\(|setHorizontalAlignment\\(\" src tests"}),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_shell_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, ShellClassifierDecision::UseBifrostTool);
        assert_eq!(output.recommended_tool, RecommendedTool::ScanUsages);
    }

    #[test]
    fn shell_sbt_config_probe_does_not_repair_builtin_to_bifrost() {
        let command = "sed -n '1,220p' .sbtopts 2>/dev/null || true; printf '\\n---\\n'; sed -n '1,220p' project/build.properties 2>/dev/null || true; printf '\\n---\\n'; grep -RIn 'CMSClassUnloadingEnabled' . 2>/dev/null | head -20";
        assert!(shell_search_targets_config_or_docs(
            &command.to_ascii_lowercase()
        ));
        assert_eq!(strong_shell_source_symbol_search(command), None);

        let mut output = ShellClassifierOutput {
            reason: "USE_BUILTIN_TOOL because this reads config files and greps an exact JVM flag."
                .to_string(),
            intent: ShellIntent::LiteralTextSearch,
            shell_semantics_required: false,
            builtin_preserves_intent: true,
            bifrost_fit: BifrostFit::NotApplicable,
            allow_exception: ShellAllowException::None,
            replacement_class: ShellReplacementClass::UseBifrostSymbol,
            decision: ShellClassifierDecision::UseBuiltinTool,
            recommended_tool: RecommendedTool::GrepSearch,
            suggested_args: json!({}),
            confidence: GateConfidence::High,
        };
        let context = GateContext {
            tool_name: "run_shell_command".to_string(),
            args: json!({"command": command}),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_shell_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, ShellClassifierDecision::UseBuiltinTool);
        assert_eq!(output.recommended_tool, RecommendedTool::GrepSearch);

        let mut contradictory_output = ShellClassifierOutput {
            reason: "USE_BUILTIN_TOOL because this reads config files and greps an exact JVM flag."
                .to_string(),
            intent: ShellIntent::BuildTestGitPackageOrProjectCli,
            shell_semantics_required: true,
            builtin_preserves_intent: false,
            bifrost_fit: BifrostFit::NotApplicable,
            allow_exception: ShellAllowException::BuildTestGitPackageOrProjectCli,
            replacement_class: ShellReplacementClass::AllowShellUncertain,
            decision: ShellClassifierDecision::AllowShell,
            recommended_tool: RecommendedTool::None,
            suggested_args: json!({}),
            confidence: GateConfidence::High,
        };

        enforce_shell_classifier_policy(&mut contradictory_output, &context);

        assert_eq!(
            contradictory_output.decision,
            ShellClassifierDecision::UseBuiltinTool
        );
        assert_eq!(
            contradictory_output.recommended_tool,
            RecommendedTool::GrepSearch
        );
    }

    #[test]
    fn shell_allow_with_builtin_policy_quorum_repairs_to_builtin() {
        let mut output = ShellClassifierOutput {
            reason:
                "USE_BUILTIN_TOOL because grep on build.sbt is config-scoped literal text search."
                    .to_string(),
            intent: ShellIntent::BuildTestGitPackageOrProjectCli,
            shell_semantics_required: false,
            builtin_preserves_intent: true,
            bifrost_fit: BifrostFit::NotApplicable,
            allow_exception: ShellAllowException::BuildTestGitPackageOrProjectCli,
            replacement_class: ShellReplacementClass::AllowShellUncertain,
            decision: ShellClassifierDecision::AllowShell,
            recommended_tool: RecommendedTool::None,
            suggested_args: json!({}),
            confidence: GateConfidence::High,
        };
        let context = GateContext {
            tool_name: "run_shell_command".to_string(),
            args: json!({"command": "grep -R -n \"lazy val .*tests\\|lazy val .*core\\|lazy val .*domain\" build.sbt project/*.sbt"}),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        assert!(conservative_shell_read_search_inspection(
            "grep -R -n \"lazy val .*tests\\|lazy val .*core\\|lazy val .*domain\" build.sbt project/*.sbt"
        ));

        enforce_shell_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, ShellClassifierDecision::UseBuiltinTool);
        assert_eq!(output.recommended_tool, RecommendedTool::GrepSearch);
        assert_eq!(output.allow_exception, ShellAllowException::None);
    }

    #[test]
    fn shell_process_inspection_stays_shell() {
        assert!(matches!(
            static_shell_route(
                &json!({"command": "sleep 10; ps -ef | grep -v grep | grep golangci-lint || true"})
            ),
            Some(ShellStaticRoute::AllowShell("static_shell_semantics"))
        ));
    }

    #[test]
    fn shell_toolchain_doc_and_env_probes_stay_shell() {
        assert!(matches!(
            static_shell_route(&json!({"command": "./tool/go doc golang.org/x/sys/unix.Cmsghdr"})),
            Some(ShellStaticRoute::AllowShell("static_shell_semantics"))
        ));
        assert!(matches!(
            static_shell_route(&json!({"command": "echo HOME=$HOME"})),
            Some(ShellStaticRoute::AllowShell("static_shell_semantics"))
        ));
    }

    #[test]
    fn shell_contradictory_suggested_args_fail_open() {
        let mut output = ShellClassifierOutput {
            reason: "USE_BUILTIN_TOOL because this is a documentation lookup.".to_string(),
            intent: ShellIntent::BuildTestGitPackageOrProjectCli,
            shell_semantics_required: false,
            builtin_preserves_intent: true,
            bifrost_fit: BifrostFit::NotApplicable,
            allow_exception: ShellAllowException::None,
            replacement_class: ShellReplacementClass::UseBuiltinInspection,
            decision: ShellClassifierDecision::UseBuiltinTool,
            recommended_tool: RecommendedTool::ReadFile,
            suggested_args: json!({"tool":"get_symbol_sources","args":{"symbols":["unix.Cmsghdr"]}}),
            confidence: GateConfidence::High,
        };

        normalize_shell_classifier_consistency(&mut output);

        assert_eq!(output.decision, ShellClassifierDecision::AllowShell);
        assert_eq!(output.recommended_tool, RecommendedTool::None);
        assert_eq!(output.confidence, GateConfidence::Low);
    }

    #[test]
    fn shell_path_filter_does_not_route_to_bifrost_or_content_grep() {
        let command = "find . -path '*/src/main/scala/*' -name '*.scala' | grep 'PatternReplace\\|TokenFilter' | sort";
        assert_eq!(shell_search_stream(command), ShellSearchStream::FilePaths);
        assert_eq!(
            static_shell_route(&json!({"command": command})),
            Some(ShellStaticRoute::AllowShell("static_shell_path_filter"))
        );
        let mut output = ShellClassifierOutput {
            reason: "USE_BIFROST_TOOL because identifiers were present.".to_string(),
            intent: ShellIntent::SymbolDefinitionLookup,
            shell_semantics_required: false,
            builtin_preserves_intent: false,
            bifrost_fit: BifrostFit::SameOrMoreDirect,
            allow_exception: ShellAllowException::None,
            replacement_class: ShellReplacementClass::UseBifrostSymbol,
            decision: ShellClassifierDecision::UseBifrostTool,
            recommended_tool: RecommendedTool::SearchSymbols,
            suggested_args: json!({}),
            confidence: GateConfidence::High,
        };
        let context = GateContext {
            tool_name: "run_shell_command".to_string(),
            args: json!({"command": command}),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_shell_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, ShellClassifierDecision::AllowShell);
        assert_eq!(output.recommended_tool, RecommendedTool::None);
    }

    #[test]
    fn shell_find_xargs_grep_content_search_routes_to_bifrost() {
        let command = "find /tmp/repo -type f -name \"*.scala\" | xargs grep -l \"ElisionTokenFilter\\|KeywordMarkerTokenFilter\" 2>/dev/null | head -20";
        assert_eq!(
            shell_search_stream(command),
            ShellSearchStream::FileContents
        );
        assert_eq!(
            strong_shell_source_symbol_search(command),
            Some(RecommendedTool::SearchSymbols)
        );

        let mut output = ShellClassifierOutput {
            reason: "ALLOW_SHELL because classifier was uncertain.".to_string(),
            intent: ShellIntent::SymbolReferenceLookup,
            shell_semantics_required: false,
            builtin_preserves_intent: true,
            bifrost_fit: BifrostFit::NotApplicable,
            allow_exception: ShellAllowException::BuildTestGitPackageOrProjectCli,
            replacement_class: ShellReplacementClass::AllowShellUncertain,
            decision: ShellClassifierDecision::AllowShell,
            recommended_tool: RecommendedTool::None,
            suggested_args: json!({}),
            confidence: GateConfidence::Low,
        };
        let context = GateContext {
            tool_name: "run_shell_command".to_string(),
            args: json!({"command": "find source -name '*.cs' -print0 | xargs -0 grep -n \"GameControllerManager\""}),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_shell_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, ShellClassifierDecision::UseBifrostTool);
        assert_eq!(output.recommended_tool, RecommendedTool::SearchSymbols);
    }

    #[test]
    fn exact_file_symbol_grep_allow_stays_allowed() {
        let mut output = GateClassifierOutput {
            reason: "ALLOW_TEXT because exact-file grep is localized.".to_string(),
            intent: TextIntent::SymbolDefinitionLookup,
            pattern_class: TextPatternClass::IdentifierLike,
            scope_class: TextScopeClass::ExactFile,
            bifrost_fit: BifrostFit::Unknown,
            allow_exception: TextAllowException::LocalizedOrSequentialRead,
            evidence: TextEvidence::default(),
            bifrost_candidate: None,
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
    fn assertion_build_string_regex_grep_stays_allowed() {
        let mut output = GateClassifierOutput {
            reason: "GATE_TO_SYMBOL_TOOL because broad source grep mentions a class.".to_string(),
            intent: TextIntent::SymbolDefinitionLookup,
            pattern_class: TextPatternClass::SymbolGlob,
            scope_class: TextScopeClass::BroadSourceScope,
            bifrost_fit: BifrostFit::SameOrMoreDirect,
            allow_exception: TextAllowException::None,
            evidence: TextEvidence {
                symbol_tokens: vec!["PatternReplaceTokenFilter".to_string()],
                same_token_or_path_bifrost_miss: false,
                same_path_recent_edit_or_write: false,
                same_path_recent_bifrost_hit: false,
                exact_text_or_regex_needed: false,
            },
            bifrost_candidate: Some(BifrostCandidate {
                tool: Some(RecommendedTool::SearchSymbols),
                args: json!({"patterns":["PatternReplaceTokenFilter"]}),
            }),
            decision: GateClassifierDecision::GateToSymbolTool,
            recommended_tool: RecommendedTool::SearchSymbols,
            suggested_args: json!({"patterns":["PatternReplaceTokenFilter"]}),
            confidence: GateConfidence::High,
        };
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({
                "path": ".",
                "glob": "*.scala",
                "pattern": "\\.build\\.string|\\.string shouldBe .*pattern_replace|PatternReplaceTokenFilter\\(.*\\)\\.build"
            }),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::AllowText);
        assert_eq!(output.recommended_tool, RecommendedTool::None);
    }

    #[test]
    fn c_api_lower_snake_symbol_grep_repairs_to_bifrost() {
        let mut output = GateClassifierOutput {
            reason: "ALLOW_TEXT because exact regex text was requested.".to_string(),
            intent: TextIntent::SymbolDefinitionLookup,
            pattern_class: TextPatternClass::IdentifierLike,
            scope_class: TextScopeClass::BroadSourceScope,
            bifrost_fit: BifrostFit::SameOrMoreDirect,
            allow_exception: TextAllowException::ExactLiteralOrRegex,
            evidence: TextEvidence {
                symbol_tokens: vec!["uv_pipe_bind2".to_string(), "uv_pipe_connect2".to_string()],
                same_token_or_path_bifrost_miss: false,
                same_path_recent_edit_or_write: false,
                same_path_recent_bifrost_hit: false,
                exact_text_or_regex_needed: true,
            },
            bifrost_candidate: Some(BifrostCandidate {
                tool: Some(RecommendedTool::SearchSymbols),
                args: json!({"patterns":["uv_pipe_bind2", "uv_pipe_connect2"]}),
            }),
            decision: GateClassifierDecision::AllowText,
            recommended_tool: RecommendedTool::None,
            suggested_args: json!({}),
            confidence: GateConfidence::Medium,
        };
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({
                "path": ".",
                "glob": "src/**/*.c",
                "pattern": "uv_pipe_bind2|uv_pipe_connect2|pipe_bind2|pipe_connect2"
            }),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(output.recommended_tool, RecommendedTool::SearchSymbols);
    }

    #[test]
    fn external_api_word_boundary_grep_stays_allowed() {
        let mut output = GateClassifierOutput {
            reason: "Bifrost scan_usages would find call sites.".to_string(),
            intent: TextIntent::SymbolUsageLookup,
            pattern_class: TextPatternClass::SymbolGlob,
            scope_class: TextScopeClass::BroadSourceScope,
            bifrost_fit: BifrostFit::SameOrMoreDirect,
            allow_exception: TextAllowException::None,
            evidence: TextEvidence {
                symbol_tokens: vec![
                    "futimens".to_string(),
                    "fchown".to_string(),
                    "fchmod".to_string(),
                ],
                same_token_or_path_bifrost_miss: false,
                same_path_recent_edit_or_write: false,
                same_path_recent_bifrost_hit: false,
                exact_text_or_regex_needed: false,
            },
            bifrost_candidate: Some(BifrostCandidate {
                tool: Some(RecommendedTool::ScanUsages),
                args: json!({"symbols":["futimens", "fchown", "fchmod"]}),
            }),
            decision: GateClassifierDecision::GateToSymbolTool,
            recommended_tool: RecommendedTool::ScanUsages,
            suggested_args: json!({"symbols":["futimens", "fchown", "fchmod"]}),
            confidence: GateConfidence::High,
        };
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({
                "path": ".",
                "glob": "src/unix/*.c",
                "pattern": "\\bfutimens\\b|\\bfchown\\b|\\bfchmod\\b"
            }),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::AllowText);
        assert_eq!(output.recommended_tool, RecommendedTool::None);
    }

    #[test]
    fn uppercase_constant_symbol_glob_uses_search_symbols() {
        let mut output = GateClassifierOutput {
            reason: "Bifrost should find constant symbols.".to_string(),
            intent: TextIntent::SymbolDefinitionLookup,
            pattern_class: TextPatternClass::SymbolGlob,
            scope_class: TextScopeClass::BroadSourceScope,
            bifrost_fit: BifrostFit::SameOrMoreDirect,
            allow_exception: TextAllowException::None,
            evidence: TextEvidence {
                symbol_tokens: vec![
                    "CHANNEL_RGB".to_string(),
                    "CHANNEL_ALPHA".to_string(),
                    "CHANNEL_ALL".to_string(),
                ],
                same_token_or_path_bifrost_miss: false,
                same_path_recent_edit_or_write: false,
                same_path_recent_bifrost_hit: false,
                exact_text_or_regex_needed: false,
            },
            bifrost_candidate: Some(BifrostCandidate {
                tool: Some(RecommendedTool::SearchSymbols),
                args: json!({"patterns":["CHANNEL_RGB", "CHANNEL_ALPHA", "CHANNEL_ALL"]}),
            }),
            decision: GateClassifierDecision::GateToSymbolTool,
            recommended_tool: RecommendedTool::SearchSymbols,
            suggested_args: json!({"patterns":["CHANNEL_RGB", "CHANNEL_ALPHA", "CHANNEL_ALL"]}),
            confidence: GateConfidence::High,
        };
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({
                "path": ".",
                "glob": "src/**/*.php",
                "pattern": "CHANNEL_RGB|CHANNEL_ALPHA|CHANNEL_ALL"
            }),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(output.recommended_tool, RecommendedTool::SearchSymbols);
    }

    #[test]
    fn declaration_regex_search_uses_search_symbols() {
        let mut output = GateClassifierOutput {
            reason: "ALLOW_TEXT because exact regex was requested.".to_string(),
            intent: TextIntent::ExactTextOrLocalizedRead,
            pattern_class: TextPatternClass::LiteralExact,
            scope_class: TextScopeClass::DirectoryOrGlob,
            bifrost_fit: BifrostFit::NotApplicable,
            allow_exception: TextAllowException::None,
            evidence: TextEvidence::default(),
            bifrost_candidate: None,
            decision: GateClassifierDecision::AllowText,
            recommended_tool: RecommendedTool::None,
            suggested_args: json!({}),
            confidence: GateConfidence::High,
        };
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({
                "path": "src",
                "glob": "*.php",
                "pattern": "enum\\s+Alignment"
            }),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(output.recommended_tool, RecommendedTool::SearchSymbols);
    }

    #[test]
    fn single_constant_with_recent_bifrost_hit_uses_get_symbol_sources() {
        let mut output = GateClassifierOutput {
            reason: "ALLOW_TEXT because this is an exact constant string.".to_string(),
            intent: TextIntent::ExactTextOrLocalizedRead,
            pattern_class: TextPatternClass::LiteralExact,
            scope_class: TextScopeClass::DirectoryOrGlob,
            bifrost_fit: BifrostFit::NotApplicable,
            allow_exception: TextAllowException::None,
            evidence: TextEvidence::default(),
            bifrost_candidate: None,
            decision: GateClassifierDecision::AllowText,
            recommended_tool: RecommendedTool::None,
            suggested_args: json!({}),
            confidence: GateConfidence::High,
        };
        let mut prior = exchange("search_symbols");
        prior.arguments = r#"{"patterns":["TOP_LEFT"]}"#.to_string();
        prior.result = "TOP_LEFT enum case found in src/Geometry/Alignment.php".to_string();
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({
                "path": "src",
                "glob": "*.php",
                "pattern": "TOP_LEFT"
            }),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: vec![prior],
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(output.recommended_tool, RecommendedTool::GetSymbolSources);
    }

    #[test]
    fn single_camel_symbol_with_recent_bifrost_hit_uses_get_symbol_sources() {
        let mut output = GateClassifierOutput {
            reason: "Bifrost can find references for this known symbol.".to_string(),
            intent: TextIntent::SymbolDefinitionLookup,
            pattern_class: TextPatternClass::SymbolGlob,
            scope_class: TextScopeClass::BroadSourceScope,
            bifrost_fit: BifrostFit::SameOrMoreDirect,
            allow_exception: TextAllowException::None,
            evidence: TextEvidence {
                symbol_tokens: vec!["ChildWindowStyle".to_string()],
                same_token_or_path_bifrost_miss: false,
                same_path_recent_edit_or_write: false,
                same_path_recent_bifrost_hit: true,
                exact_text_or_regex_needed: false,
            },
            bifrost_candidate: Some(BifrostCandidate {
                tool: Some(RecommendedTool::ScanUsages),
                args: json!({"symbols":["ChildWindowStyle"]}),
            }),
            decision: GateClassifierDecision::GateToSymbolTool,
            recommended_tool: RecommendedTool::ScanUsages,
            suggested_args: json!({"symbols":["ChildWindowStyle"]}),
            confidence: GateConfidence::High,
        };
        let mut prior = exchange("search_symbols");
        prior.arguments = r#"{"patterns":["ChildWindowStyle"]}"#.to_string();
        prior.result = "ChildWindowStyle found in source/Toolbox/Styles.xaml".to_string();
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({
                "path": ".",
                "glob": "**/*",
                "pattern": "ChildWindowStyle"
            }),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: vec![prior],
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(output.recommended_tool, RecommendedTool::GetSymbolSources);
    }

    #[test]
    fn broad_callsite_grep_ignores_prior_empty_bifrost_and_uses_scan_usages() {
        let mut output = GateClassifierOutput {
            reason: "ALLOW_TEXT because Bifrost already returned empty.".to_string(),
            intent: TextIntent::SymbolUsageLookup,
            pattern_class: TextPatternClass::RegexText,
            scope_class: TextScopeClass::BroadSourceScope,
            bifrost_fit: BifrostFit::SameOrMoreDirect,
            allow_exception: TextAllowException::None,
            evidence: TextEvidence {
                symbol_tokens: vec!["rawField".to_string(), "rawValue".to_string()],
                same_token_or_path_bifrost_miss: true,
                same_path_recent_edit_or_write: false,
                same_path_recent_bifrost_hit: false,
                exact_text_or_regex_needed: false,
            },
            bifrost_candidate: Some(BifrostCandidate {
                tool: Some(RecommendedTool::ScanUsages),
                args: json!({"symbols":["rawField", "rawValue"]}),
            }),
            decision: GateClassifierDecision::AllowText,
            recommended_tool: RecommendedTool::None,
            suggested_args: json!({}),
            confidence: GateConfidence::High,
        };
        let mut prior = exchange("scan_usages");
        prior.arguments = r#"{"symbols":["rawField","rawValue"]}"#.to_string();
        prior.result = r#"{"usages":[{"symbol":"rawField","total_hits":0},{"symbol":"rawValue","total_hits":0}]}"#.to_string();
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({
                "path": ".",
                "glob": "**/*.scala",
                "pattern": "rawField\\(|rawValue\\(|raw\\("
            }),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: vec![prior],
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(output.recommended_tool, RecommendedTool::ScanUsages);
    }

    #[test]
    fn exact_bare_identifier_grep_with_text_evidence_stays_allowed() {
        let mut output = GateClassifierOutput {
            reason: "Bifrost could find the attribute symbol.".to_string(),
            intent: TextIntent::SymbolUsageLookup,
            pattern_class: TextPatternClass::SymbolGlob,
            scope_class: TextScopeClass::BroadSourceScope,
            bifrost_fit: BifrostFit::SameOrMoreDirect,
            allow_exception: TextAllowException::NonSourceText,
            evidence: TextEvidence {
                symbol_tokens: vec!["InternalsVisibleTo".to_string()],
                same_token_or_path_bifrost_miss: false,
                same_path_recent_edit_or_write: false,
                same_path_recent_bifrost_hit: false,
                exact_text_or_regex_needed: true,
            },
            bifrost_candidate: Some(BifrostCandidate {
                tool: Some(RecommendedTool::SearchSymbols),
                args: json!({"patterns":["InternalsVisibleTo"]}),
            }),
            decision: GateClassifierDecision::GateToSymbolTool,
            recommended_tool: RecommendedTool::SearchSymbols,
            suggested_args: json!({"patterns":["InternalsVisibleTo"]}),
            confidence: GateConfidence::High,
        };
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({
                "path": "source",
                "glob": "**/*.cs",
                "pattern": "InternalsVisibleTo"
            }),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::AllowText);
        assert_eq!(output.recommended_tool, RecommendedTool::None);
    }

    #[test]
    fn broad_member_call_regex_repairs_to_scan_usages() {
        let mut output = low_confidence_allow_output(TextIntent::LiteralOrRegexSearch);
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({
                "path": ".",
                "glob": "src/**/*.php",
                "pattern": "->destroy\\("
            }),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::GateToSymbolTool);
        assert_eq!(output.recommended_tool, RecommendedTool::ScanUsages);
    }

    #[test]
    fn lower_snake_wire_key_grep_stays_allowed() {
        let mut output = low_confidence_allow_output(TextIntent::SymbolDefinitionLookup);
        output.decision = GateClassifierDecision::GateToSymbolTool;
        output.recommended_tool = RecommendedTool::SearchSymbols;
        output.confidence = GateConfidence::High;
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({
                "path": ".",
                "glob": "*.scala",
                "pattern": "on_disk_rescore"
            }),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::AllowText);
        assert_eq!(output.recommended_tool, RecommendedTool::None);

        let mut output = low_confidence_allow_output(TextIntent::SymbolUsageLookup);
        output.decision = GateClassifierDecision::GateToSymbolTool;
        output.recommended_tool = RecommendedTool::SearchSymbols;
        output.confidence = GateConfidence::High;
        output.bifrost_fit = BifrostFit::SameOrMoreDirect;
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({
                "path": ".",
                "glob": "**/*.scala",
                "pattern": "throttled_millis|version_conflicts|noops|retries"
            }),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_exchanges: Vec::new(),
        };

        enforce_text_classifier_policy(&mut output, &context);

        assert_eq!(output.decision, GateClassifierDecision::AllowText);
        assert_eq!(output.recommended_tool, RecommendedTool::None);
    }

    #[test]
    fn word_boundary_bare_identifier_grep_stays_allowed() {
        let mut output = low_confidence_allow_output(TextIntent::SymbolDefinitionLookup);
        output.decision = GateClassifierDecision::GateToSymbolTool;
        output.recommended_tool = RecommendedTool::SearchSymbols;
        output.confidence = GateConfidence::High;
        let context = GateContext {
            tool_name: "grep_search".to_string(),
            args: json!({
                "path": ".",
                "glob": "**/*.scala",
                "pattern": "\\bstrategy\\b"
            }),
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
            glob_exact.get("scope_granularity").and_then(Value::as_str),
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
        assert!(matches!(
            static_text_route(
                "grep_search",
                &json!({"path": "README.md", "pattern": "error: file not found"}),
                &[]
            ),
            Some(TextStaticRoute::AllowText("static_exact_file_grep"))
        ));
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
            static_shell_route(&json!({"command": "python -m py_compile openhands/cli/tui.py tests/unit/test_cli_tui.py"})),
            Some(ShellStaticRoute::AllowShell("static_shell_semantics"))
        ));
        assert!(matches!(
            static_shell_route(
                &json!({"command": "SBT_BIN=/tmp/sbt && \"$SBT_BIN\" -batch \"projects\" | grep -E \"(project|tests)\""})
            ),
            Some(ShellStaticRoute::AllowShell("static_shell_semantics"))
        ));
        assert!(matches!(
            static_shell_route(&json!({"command": "./tool/go test -count=1 ./util/osuser"})),
            Some(ShellStaticRoute::AllowShell("static_shell_semantics"))
        ));
        assert!(matches!(
            static_shell_route(
                &json!({"command": "if [[ -x \"$HOME/.local/share/TheAlgorithms__Java/jdk/current/bin/java\" ]]; then \"$HOME/.local/share/TheAlgorithms__Java/jdk/current/bin/java\" -version; else echo missing; fi"})
            ),
            Some(ShellStaticRoute::AllowShell("static_shell_semantics"))
        ));
        assert!(matches!(
            static_shell_route(
                &json!({"command": "cc -E -P -Iinclude -Isrc src/unix/pipe.c | grep -n \"int uv_pipe_chmod\\|int uv_pipe_getsockname\\|int uv_pipe_getpeername\" | head -20"})
            ),
            Some(ShellStaticRoute::AllowShell("static_shell_semantics"))
        ));
        assert!(matches!(
            static_shell_route(
                &json!({"command": "cat build.sbt | grep -E \"project|tests\" | head -30"})
            ),
            None
        ));
        assert!(matches!(
            static_shell_route(
                &json!({"command": "grep -RIn \"EnsureCertLoops|TerminateTLS|type TCPPortHandler\" kube ipn . | head -200"})
            ),
            None
        ));
        assert!(matches!(
            static_shell_route(
                &json!({"command": "grep -rn \"uv_os_homedir\\|uv_os_tmpdir\" /tmp/work --include=\"*.c\" -l"})
            ),
            None
        ));
    }

    #[test]
    fn static_shell_route_only_allows_obvious_shell_semantics() {
        assert!(matches!(
            static_shell_route(&json!({"command": "cargo test -q"})),
            Some(ShellStaticRoute::AllowShell("static_shell_semantics"))
        ));
        assert!(matches!(
            static_shell_route(&json!({"command": "env | grep -E 'JAVA|SBT' | sort"})),
            Some(ShellStaticRoute::AllowShell("static_shell_semantics"))
        ));
        assert!(matches!(
            static_shell_route(&json!({"command": "printenv JAVA_HOME"})),
            Some(ShellStaticRoute::AllowShell("static_shell_semantics"))
        ));
        assert!(matches!(
            static_shell_route(
                &json!({"command": "printf 'A=%s\\nB=%s\\n' \"${JAVA_TOOL_OPTIONS-}\" \"$SBT_OPTS\""})
            ),
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
            None
        ));
    }

    #[test]
    fn static_shell_route_splits_symbol_search_from_literal_inspection() {
        assert!(matches!(
            static_shell_route(
                &json!({"command": "rg -n \"RichEditBoxDefaultLineEnding|ApplyTabAndLineEndingFix\" src/Notepads/Controls/TextEditor"})
            ),
            None
        ));
        assert!(matches!(
            static_shell_route(
                &json!({"command": "find src -name '*.cs' | xargs grep -n \"RemoveExecutableNameOrPathFromCommandLineArgs|Notepads-Dev\""})
            ),
            None
        ));
        assert!(matches!(
            static_shell_route(
                &json!({"command": "grep -R -n \"StopTokenFilter(\" elastic4s-*/src/main/scala 2>/dev/null | head -100"})
            ),
            None
        ));
        assert!(matches!(
            static_shell_route(
                &json!({"command": "grep -RIn \"failed to encode|invalid path\" tests -g '*.php' | head"})
            ),
            None
        ));
    }

    #[test]
    fn static_shell_route_keeps_literal_recursive_scripts_builtin() {
        assert!(matches!(
            static_shell_route(
                &json!({"command": "python - <<'PY'\nfrom pathlib import Path\nfor p in Path('.').rglob('*.scala'):\n    txt = p.read_text(errors='ignore')\n    if 'StopTokenFilter' in txt:\n        print(p)\nPY"})
            ),
            None
        ));
        assert!(matches!(
            static_shell_route(
                &json!({"command": "python - <<'PY'\nimport pathlib\nfor path in pathlib.Path('.').rglob('*.scala'):\n    text = path.read_text()\n    if 'ChunkingSettings(' in text:\n        print(path)\nPY"})
            ),
            None
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
        assert!(matches!(
            static_shell_route(
                &json!({"command": "perl -pi -e 's/\\r$//' source/Playnite/WebView/OffscreenWebView.cs"})
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
