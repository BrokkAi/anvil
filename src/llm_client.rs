use anyhow::{Context, Result};
use futures::Stream;
use futures::StreamExt;
use futures::future::BoxFuture;
use serde::ser::{SerializeSeq, SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::http_retry::{LlmRetryTier, RetryableLlmError};
use crate::structured_output::StructuredOutputRequest;

/// Default value for the `--llm-idle-timeout-secs` CLI flag (and the
/// env var `ANVIL_LLM_IDLE_TIMEOUT_SECS`). The actual value used
/// per request is a required parameter on `LlmBackend::stream_chat` --
/// callers cannot fall back to this implicitly.
///
/// Stream timeout semantics are split into two phases: a long
/// first-progress timeout for models that spend time reasoning before
/// emitting the first chunk, and a shorter inter-chunk timeout for
/// detecting stalls after meaningful SSE progress has begun.
/// "Meaningful progress" is a parsed `data:` event that contributed
/// content, tool-call deltas, or a completion marker. Comments
/// (`:keepalive\n`), blank lines, and partial bytes that don't advance
/// the parser do NOT reset the inter-chunk timer -- otherwise a server
/// or proxy could keep us alive forever by drip-feeding pings.
///
/// Distinct from the reqwest client's overall `.timeout()` (wall-clock).
pub const DEFAULT_IDLE_CHUNK_TIMEOUT_SECS: u64 = 300;

/// Default value for `--llm-stall-timeout-secs`: the maximum gap between
/// meaningful chunks after the first progress event has arrived.
pub const DEFAULT_INTER_CHUNK_TIMEOUT_SECS: u64 = 60;

/// Lower bound for the LLM stream timeout CLI flags and the `/idle-timeout`
/// slash command. 0 would mean "abort instantly", which is never useful.
pub const MIN_IDLE_CHUNK_TIMEOUT_SECS: u64 = 1;

/// Upper bound for the LLM stream timeout CLI flags and the `/idle-timeout`
/// slash command. 24h is well above any realistic local
/// LLM prompt processing on consumer hardware and stops a typo'd huge
/// number from effectively disabling the stall detector.
pub const MAX_IDLE_CHUNK_TIMEOUT_SECS: u64 = 86_400;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdleTimeouts {
    pub first_progress: Duration,
    pub inter_chunk: Duration,
}

impl IdleTimeouts {
    pub fn uniform(timeout: Duration) -> Self {
        Self {
            first_progress: timeout,
            inter_chunk: timeout,
        }
    }

    pub fn min(self, timeout: Duration) -> Self {
        Self {
            first_progress: self.first_progress.min(timeout),
            inter_chunk: self.inter_chunk.min(timeout),
        }
    }
}

/// `/models` discovery should fail fast. Chat requests can legitimately
/// run for minutes; setup refreshes should not inherit that wall-clock.
const MODEL_DISCOVERY_TIMEOUT_SECS: u64 = 12;
const MODEL_DISCOVERY_TASK_TIMEOUT_SECS: u64 = 20;

#[derive(Debug)]
pub(crate) struct IncompleteStreamError {
    protocol: &'static str,
    expected_marker: &'static str,
}

impl IncompleteStreamError {
    pub(crate) fn new(protocol: &'static str, expected_marker: &'static str) -> Self {
        Self {
            protocol,
            expected_marker,
        }
    }
}

impl fmt::Display for IncompleteStreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} stream ended before completion marker {}",
            self.protocol, self.expected_marker
        )
    }
}

impl std::error::Error for IncompleteStreamError {}

pub(crate) fn is_incomplete_stream_error(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.downcast_ref::<IncompleteStreamError>().is_some())
}

#[derive(Debug)]
pub(crate) struct OutputBudgetExhaustedError;

impl fmt::Display for OutputBudgetExhaustedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            "model exhausted its output token budget before emitting any text or tool calls; \
             not retrying (the same request would hit the same cap)",
        )
    }
}

impl std::error::Error for OutputBudgetExhaustedError {}

pub(crate) fn is_output_budget_exhausted_error(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.downcast_ref::<OutputBudgetExhaustedError>().is_some())
}

pub(crate) fn is_retryable_llm_error(error: &anyhow::Error) -> bool {
    llm_retry_tier(error).is_some()
}

pub(crate) fn llm_retry_tier(error: &anyhow::Error) -> Option<LlmRetryTier> {
    if is_output_budget_exhausted_error(error) {
        return None;
    }

    if is_incomplete_stream_error(error) {
        return Some(LlmRetryTier::Fast);
    }

    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<RetryableLlmError>())
        .map(RetryableLlmError::tier)
}

pub(crate) const EMPTY_COMPLETION_RETRY_REASON: &str =
    "no meaningful progress: empty completion (no text, no tool calls)";

pub(crate) fn is_degenerate_empty_completion(response: &LlmResponse) -> bool {
    match response {
        LlmResponse::Text { text, .. } => text.trim().is_empty(),
        LlmResponse::ToolCalls { text, calls, .. } => text.trim().is_empty() && calls.is_empty(),
    }
}

/// Retry a streamed LLM request when the caller guarantees streamed
/// deltas are not user-visible. Use the main tool-loop wrapper instead
/// when callbacks can emit text or thought updates to the client.
pub(crate) async fn stream_chat_no_visible_output_with_retry<F>(
    llm: &dyn LlmBackend,
    operation: &str,
    cancel: &CancellationToken,
    mut build_request: F,
) -> Result<LlmResponse>
where
    F: FnMut() -> StreamChatRequest,
{
    let mut attempt = 1u64;
    loop {
        match llm.stream_chat(build_request()).await {
            Ok(response)
                if !cancel.is_cancelled()
                    && is_degenerate_empty_completion(&response)
                    && attempt < LlmRetryTier::Fast.max_attempts() =>
            {
                tracing::warn!(
                    attempt,
                    max_attempts = LlmRetryTier::Fast.max_attempts(),
                    operation,
                    "retrying empty LLM completion with no visible output"
                );
                crate::http_retry::sleep_before_retry_for_tier(
                    operation,
                    LlmRetryTier::Fast,
                    attempt,
                    EMPTY_COMPLETION_RETRY_REASON.to_string(),
                    Some(cancel),
                )
                .await?;
                attempt += 1;
            }
            Ok(response) => return Ok(response),
            Err(error)
                if !cancel.is_cancelled()
                    && llm_retry_tier(&error).is_some_and(|tier| attempt < tier.max_attempts()) =>
            {
                let tier = llm_retry_tier(&error).expect("guard checked retry tier");
                tracing::warn!(
                    attempt,
                    max_attempts = tier.max_attempts(),
                    operation,
                    "retrying transient LLM stream failure with no visible output"
                );
                crate::http_retry::sleep_before_retry_for_tier(
                    operation,
                    tier,
                    attempt,
                    format!("{error:#}"),
                    Some(cancel),
                )
                .await?;
                attempt += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

/// Owning callback handed token deltas as the LLM streams them.
pub type TokenSink = Box<dyn FnMut(&str) + Send>;

// ---------------------------------------------------------------------------
// Tool calling types (OpenAI-compatible)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: FunctionDef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Provider-reported token counts for a single `stream_chat` call.
///
/// Field names mirror the ACP `session/usage` RFD so they map 1:1 into
/// `agent_client_protocol::Usage`. Counts are populated from the
/// upstream LLM's own accounting (OpenAI's `usage` SSE chunk, the Codex
/// Responses API `response.completed` event); backends that don't
/// surface usage (e.g. Ollama under some configurations) leave the
/// fields at zero. Using `u64` matches the wire type expected by the
/// ACP `Usage` struct.
#[derive(Debug, Default, Clone, Copy)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub thought_tokens: u64,
    pub cached_read_tokens: u64,
    pub cached_write_tokens: u64,
}

impl TokenUsage {
    /// Whether every token bucket is zero.
    pub fn is_zero(&self) -> bool {
        self.input_tokens == 0
            && self.output_tokens == 0
            && self.thought_tokens == 0
            && self.cached_read_tokens == 0
            && self.cached_write_tokens == 0
    }

    /// Sum of all categories. Matches the ACP `total_tokens` semantics:
    /// "Sum of all token types across session." Cached counts are
    /// included because the spec example treats them as additive
    /// (35k + 12k + 5k + 5k + 1k = 53k).
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens
            .saturating_add(self.output_tokens)
            .saturating_add(self.thought_tokens)
            .saturating_add(self.cached_read_tokens)
            .saturating_add(self.cached_write_tokens)
    }

    /// Merge another reading into this one. Used to roll up per-call
    /// usage into a per-turn or per-session total.
    pub fn add(&mut self, other: TokenUsage) {
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.thought_tokens = self.thought_tokens.saturating_add(other.thought_tokens);
        self.cached_read_tokens = self
            .cached_read_tokens
            .saturating_add(other.cached_read_tokens);
        self.cached_write_tokens = self
            .cached_write_tokens
            .saturating_add(other.cached_write_tokens);
    }
}

/// What the LLM returned: either final text or tool calls to execute,
/// plus the provider-reported token usage for the call.
#[derive(Debug)]
pub enum LlmResponse {
    Text {
        text: String,
        reasoning_content: Option<String>,
        usage: TokenUsage,
    },
    ToolCalls {
        text: String,
        reasoning_content: Option<String>,
        calls: Vec<ToolCall>,
        usage: TokenUsage,
    },
}

impl LlmResponse {
    pub fn usage(&self) -> TokenUsage {
        match self {
            LlmResponse::Text { usage, .. } | LlmResponse::ToolCalls { usage, .. } => *usage,
        }
    }
}

/// Parameters for a single streamed chat request.
///
/// Grouping the transport knobs into one object keeps backend APIs below
/// Clippy's argument-count threshold without hiding the semantics in an
/// opaque tuple.
pub struct StreamChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub reasoning_effort: Option<String>,
    pub temperature: Option<f64>,
    pub structured_output: Option<StructuredOutputRequest>,
    pub on_token: TokenSink,
    pub on_thought: TokenSink,
    pub cancel: CancellationToken,
    pub idle_timeouts: IdleTimeouts,
}

// ---------------------------------------------------------------------------
// Chat message (extended for tool calling)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub content: Vec<ChatContentPart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl Serialize for ChatMessage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut len = 1;
        if !self.content.is_empty() {
            len += 1;
        }
        if self.tool_calls.is_some() {
            len += 1;
        }
        if self.tool_call_id.is_some() {
            len += 1;
        }
        if self.name.is_some() {
            len += 1;
        }
        if self.reasoning_content.is_some() {
            len += 2;
        }

        let mut state = serializer.serialize_struct("ChatMessage", len)?;
        state.serialize_field("role", &self.role)?;
        if !self.content.is_empty() {
            state.serialize_field("content", &self.content)?;
        }
        if let Some(tool_calls) = &self.tool_calls {
            state.serialize_field("tool_calls", tool_calls)?;
        }
        if let Some(tool_call_id) = &self.tool_call_id {
            state.serialize_field("tool_call_id", tool_call_id)?;
        }
        if let Some(name) = &self.name {
            state.serialize_field("name", name)?;
        }
        if let Some(reasoning_content) = &self.reasoning_content {
            state.serialize_field("reasoning_content", reasoning_content)?;
            state.serialize_field("reasoning", reasoning_content)?;
        }
        state.end()
    }
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: vec![ChatContentPart::text(content)],
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: vec![ChatContentPart::text(content)],
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
        }
    }

    pub fn user_parts(content: Vec<ChatContentPart>) -> Self {
        Self {
            role: "user".to_string(),
            content,
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: vec![ChatContentPart::text(content)],
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
        }
    }

    pub fn assistant_with_reasoning(
        content: impl Into<String>,
        reasoning_content: Option<String>,
    ) -> Self {
        Self {
            role: "assistant".to_string(),
            content: vec![ChatContentPart::text(content)],
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content,
        }
    }

    pub fn assistant_tool_calls(calls: Vec<ToolCall>) -> Self {
        Self::assistant_tool_calls_with_content_and_reasoning("", calls, None)
    }

    pub fn assistant_tool_calls_with_content_and_reasoning(
        content: impl Into<String>,
        calls: Vec<ToolCall>,
        reasoning_content: Option<String>,
    ) -> Self {
        let content = content.into();
        Self {
            role: "assistant".to_string(),
            content: if content.is_empty() {
                Vec::new()
            } else {
                vec![ChatContentPart::text(content)]
            },
            tool_calls: Some(calls),
            tool_call_id: None,
            name: None,
            reasoning_content,
        }
    }

    pub fn tool_result(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            role: "tool".to_string(),
            content: vec![ChatContentPart::text(content)],
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name: Some(name.into()),
            reasoning_content: None,
        }
    }

    pub fn text_content(&self) -> Option<&str> {
        self.content.iter().find_map(|part| match part {
            ChatContentPart::Text { text } => Some(text.as_str()),
            ChatContentPart::Image { .. } => None,
        })
    }

    pub fn content_text(&self) -> String {
        self.content
            .iter()
            .filter_map(|part| match part {
                ChatContentPart::Text { text } => Some(text.as_str()),
                ChatContentPart::Image { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn has_content(&self) -> bool {
        !self.content.is_empty()
    }
}

pub fn messages_include_images(messages: &[ChatMessage]) -> bool {
    messages.iter().any(|message| {
        message
            .content
            .iter()
            .any(|part| matches!(part, ChatContentPart::Image { .. }))
    })
}

pub fn rewrite_image_prompt_provider_error(error: &str) -> Option<&'static str> {
    let normalized = error.to_ascii_lowercase();
    let image_like = normalized.contains("image")
        || normalized.contains("vision")
        || normalized.contains("multimodal")
        || normalized.contains("modality");
    let unsupported_like = normalized.contains("not support")
        || normalized.contains("unsupported")
        || normalized.contains("invalid input type")
        || normalized.contains("input modality")
        || normalized.contains("only text")
        || normalized.contains("text-only");
    if image_like && unsupported_like {
        Some(
            "The selected model does not accept image prompts. Choose a vision-capable model and try again.",
        )
    } else {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContentPart {
    Text { text: String },
    Image { image_url: String },
}

impl ChatContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub fn image_data(data: impl Into<String>, mime_type: impl AsRef<str>) -> Self {
        let data = data.into();
        let mime_type = mime_type.as_ref();
        let image_url = if data.starts_with("data:") {
            data
        } else {
            format!("data:{mime_type};base64,{data}")
        };
        Self::Image { image_url }
    }

    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image {
            image_url: url.into(),
        }
    }
}

impl Serialize for ChatContentPart {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            ChatContentPart::Text { text } => {
                let mut state = serializer.serialize_struct("ChatContentPart", 2)?;
                state.serialize_field("type", "text")?;
                state.serialize_field("text", text)?;
                state.end()
            }
            ChatContentPart::Image { image_url } => {
                #[derive(Serialize)]
                struct ImageUrl<'a> {
                    url: &'a str,
                }

                let mut state = serializer.serialize_struct("ChatContentPart", 2)?;
                state.serialize_field("type", "image_url")?;
                state.serialize_field("image_url", &ImageUrl { url: image_url })?;
                state.end()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Model metadata
// ---------------------------------------------------------------------------

/// One reasoning-effort preset advertised by the server for a given model.
/// Mirrors the `supported_reasoning_levels[]` entries returned by the
/// ChatGPT `/models` endpoint. Backends without per-effort presets (Ollama,
/// OpenAI `/v1/models`) simply leave the per-model vec empty.
#[derive(Debug, Clone)]
pub struct ReasoningLevelPreset {
    pub effort: String,
    pub description: String,
}

/// Per-token USD pricing published by a provider's model catalog.
///
/// Today this is populated only from OpenRouter's `/models` endpoint,
/// which publishes `pricing.prompt` and `pricing.completion` values as
/// decimal USD-per-token strings.
#[derive(Debug, Clone, Copy)]
pub struct ModelPricing {
    pub input_cost_per_token_usd: f64,
    pub output_cost_per_token_usd: f64,
}

impl ModelPricing {
    /// Estimate the provider-billed cost for one call using the token
    /// accounting shape Anvil already stores.
    ///
    /// Input-side cached tokens still occupy the provider's prompt-token
    /// bucket, so they are billed at the input rate when a backend does
    /// not publish a separate cache tariff. Thought tokens are part of
    /// the completion-side total and are billed at the output rate.
    pub fn estimate_cost_usd(&self, usage: TokenUsage) -> f64 {
        let billed_input_tokens =
            usage.input_tokens + usage.cached_read_tokens + usage.cached_write_tokens;
        let billed_output_tokens = usage.output_tokens + usage.thought_tokens;
        billed_input_tokens as f64 * self.input_cost_per_token_usd
            + billed_output_tokens as f64 * self.output_cost_per_token_usd
    }
}

/// Richer model descriptor surfaced through `LlmBackend::list_model_metadata`.
/// The `id` is the wire identifier the backend expects in `stream_chat`;
/// the optional reasoning fields are populated only for backends whose
/// catalog publishes them (today: `CodexClient`).
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub id: String,
    pub default_reasoning_level: Option<String>,
    pub supported_reasoning_levels: Vec<ReasoningLevelPreset>,
    /// Tri-state image-input support as published by the provider.
    /// `None` means the backend does not expose reliable modality info.
    pub supports_images: Option<bool>,
    /// Maximum context window in tokens, as published by the provider.
    /// `None` when the backend doesn't expose it (Codex, Ollama); the
    /// compression layer falls back to a per-backend default in that
    /// case.
    pub context_length: Option<u32>,
    /// Provider-published per-token pricing, when available.
    pub pricing: Option<ModelPricing>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModelInfo {
    pub configured_model: String,
    pub resolved_provider: Option<String>,
    pub resolved_model: String,
}

impl ModelMetadata {
    /// Lift a bare model id to a metadata record with no reasoning data.
    /// Used by backends that don't publish reasoning presets, and by the
    /// default impl of `list_model_metadata` so existing `list_models`
    /// impls don't have to be rewritten.
    pub fn id_only(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            default_reasoning_level: None,
            supported_reasoning_levels: Vec::new(),
            supports_images: None,
            context_length: None,
            pricing: None,
        }
    }

    /// Estimate this model's cost for one provider-reported usage sample.
    pub fn estimate_cost_usd(&self, usage: TokenUsage) -> Option<f64> {
        self.pricing.map(|pricing| pricing.estimate_cost_usd(usage))
    }
}

// ---------------------------------------------------------------------------
// LLM backend trait
// ---------------------------------------------------------------------------

pub trait LlmBackend: Send + Sync {
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>>;

    fn resolve_model_info(&self, configured_model: &str) -> ResolvedModelInfo {
        ResolvedModelInfo {
            configured_model: configured_model.to_string(),
            resolved_provider: None,
            resolved_model: configured_model.to_string(),
        }
    }

    /// Same catalog as `list_models`, but carrying any per-model
    /// reasoning-effort presets the backend's discovery endpoint
    /// publishes. The default impl lifts each id to `ModelMetadata::id_only`,
    /// so backends that don't expose reasoning data don't need to override.
    fn list_model_metadata(&self) -> BoxFuture<'_, Result<Vec<ModelMetadata>>> {
        let fut = self.list_models();
        Box::pin(async move { Ok(fut.await?.into_iter().map(ModelMetadata::id_only).collect()) })
    }

    /// Stream a chat completion. `reasoning_effort` is honored only by
    /// backends that route to a reasoning-capable endpoint (today,
    /// `CodexClient` via the ChatGPT Responses API); other backends
    /// ignore it. `on_thought` receives chain-of-thought / reasoning
    /// text deltas separate from the assistant text on `on_token`;
    /// backends that don't surface reasoning never invoke it.
    ///
    /// `idle_timeouts` contains the first-progress timeout and the
    /// post-progress inter-chunk stall timeout before the backend aborts
    /// the stream. Threaded from `--llm-idle-timeout-secs`,
    /// `--llm-stall-timeout-secs`, and the per-session `/idle-timeout`
    /// override.
    fn stream_chat(&self, request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>>;
}

// ---------------------------------------------------------------------------
// Request/response types for the OpenAI-compatible API
// ---------------------------------------------------------------------------

fn serialize_chat_completion_messages<S>(
    messages: &[ChatMessage],
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut seq = serializer.serialize_seq(Some(messages.len()))?;
    for message in messages {
        seq.serialize_element(&ChatCompletionMessage(message))?;
    }
    seq.end()
}

struct ChatCompletionMessage<'a>(&'a ChatMessage);

impl Serialize for ChatCompletionMessage<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let message = self.0;
        let mut len = 1;
        if !message.content.is_empty() {
            len += 1;
        }
        if message.tool_calls.is_some() {
            len += 1;
        }
        if message.tool_call_id.is_some() {
            len += 1;
        }
        if message.name.is_some() {
            len += 1;
        }
        if message.reasoning_content.is_some() {
            len += 2;
        }

        let mut state = serializer.serialize_struct("ChatCompletionMessage", len)?;
        state.serialize_field("role", &message.role)?;
        if !message.content.is_empty() {
            if message.content.len() == 1 {
                if let ChatContentPart::Text { text } = &message.content[0] {
                    state.serialize_field("content", text)?;
                } else {
                    state.serialize_field("content", &message.content)?;
                }
            } else {
                state.serialize_field("content", &message.content)?;
            }
        }
        if let Some(tool_calls) = &message.tool_calls {
            state.serialize_field("tool_calls", tool_calls)?;
        }
        if let Some(tool_call_id) = &message.tool_call_id {
            state.serialize_field("tool_call_id", tool_call_id)?;
        }
        if let Some(name) = &message.name {
            state.serialize_field("name", name)?;
        }
        if let Some(reasoning_content) = &message.reasoning_content {
            state.serialize_field("reasoning_content", reasoning_content)?;
            state.serialize_field("reasoning", reasoning_content)?;
        }
        state.end()
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    #[serde(serialize_with = "serialize_chat_completion_messages")]
    messages: Vec<ChatMessage>,
    stream: bool,
    /// OpenAI's opt-in to receive a final `usage` payload on a streamed
    /// response. Without this, the upstream `usage` field is `null` on
    /// every chunk and we'd have to fall back to local tiktoken
    /// estimates. OpenRouter and Ollama both honor this flag (it's a
    /// no-op on servers that don't, since they were ignoring the
    /// trailing usage chunk anyway).
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    /// OpenRouter/Ollama's unified `reasoning: { effort }` object.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningConfig>,
    /// DeepSeek's thinking toggle, e.g. `{"type": "enabled"}`, paired with
    /// the top-level `reasoning_effort` below.
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
    /// DeepSeek's (OpenAI-compatible) top-level `reasoning_effort` string,
    /// `"high"` or `"max"`. Distinct from the unified `reasoning.effort`.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ChatCompletionResponseFormat>,
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Debug, Serialize)]
struct ReasoningConfig {
    effort: String,
}

/// DeepSeek's `thinking` object. We only ever send the `type` toggle.
#[derive(Debug, Serialize)]
struct ThinkingConfig {
    r#type: &'static str,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ChatCompletionResponseFormat {
    #[serde(rename = "json_schema")]
    JsonSchema { json_schema: NativeJsonSchemaFormat },
    #[serde(rename = "json_object")]
    JsonObject,
}

#[derive(Debug, Serialize)]
struct NativeJsonSchemaFormat {
    name: String,
    schema: serde_json::Value,
    strict: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ModelEntry {
    pub(crate) id: String,
    #[serde(default)]
    pub(crate) supported_parameters: Vec<String>,
    #[serde(default)]
    pub(crate) default_parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) architecture: Option<ModelArchitecture>,
    #[serde(default)]
    pub(crate) context_length: Option<u32>,
    #[serde(default)]
    pub(crate) pricing: Option<ModelPricingEntry>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ModelArchitecture {
    #[serde(default)]
    pub(crate) input_modalities: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ModelsResponse {
    #[serde(default)]
    pub(crate) data: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ModelPricingEntry {
    #[serde(default)]
    pub(crate) prompt: String,
    #[serde(default)]
    pub(crate) completion: String,
}

const OPENROUTER_REASONING_PRESETS: &[(&str, &str)] = &[
    (
        "minimal",
        "Basic reasoning with minimal computational effort",
    ),
    ("low", "Light reasoning for simple problems"),
    ("medium", "Balanced reasoning for moderate complexity"),
    ("high", "Deep reasoning for complex problems"),
    (
        "xhigh",
        "Extra-high reasoning for the most difficult problems",
    ),
];

fn openrouter_reasoning_presets() -> Vec<ReasoningLevelPreset> {
    OPENROUTER_REASONING_PRESETS
        .iter()
        .map(|(effort, description)| ReasoningLevelPreset {
            effort: (*effort).to_string(),
            description: (*description).to_string(),
        })
        .collect()
}

/// How a client spells per-request reasoning effort on the wire. Different
/// OpenAI-compatible servers differ here, so the client carries the dialect
/// rather than sniffing the base URL at send time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReasoningWire {
    /// Don't forward reasoning effort (plain OpenAI / Codex / Ollama without
    /// reasoning). `stream_chat` strips any incoming effort.
    Off,
    /// OpenRouter / Ollama unified `reasoning: { effort }` object.
    Unified,
    /// DeepSeek: `thinking: { type: "enabled" }` + a top-level
    /// `reasoning_effort` on DeepSeek's `high`/`max` scale.
    DeepSeek,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StructuredOutputWire {
    JsonSchema,
    JsonObject,
}

/// DeepSeek's two documented reasoning levels. DeepSeek's `/v1/models` is
/// id-only (no capability fields), so -- like Bedrock and Ollama -- we
/// declare the levels here and let the shared picker/selection code in
/// `agent.rs` and `session.rs` map any requested effort onto them.
const DEEPSEEK_REASONING_PRESETS: &[(&str, &str)] = &[
    ("high", "Deep reasoning (DeepSeek default)."),
    ("max", "Maximum reasoning for the hardest problems."),
];

fn deepseek_reasoning_presets() -> Vec<ReasoningLevelPreset> {
    DEEPSEEK_REASONING_PRESETS
        .iter()
        .map(|(effort, description)| ReasoningLevelPreset {
            effort: (*effort).to_string(),
            description: (*description).to_string(),
        })
        .collect()
}

/// Clamp an effort string to a value DeepSeek actually accepts. The picker
/// and `select_session_reasoning_effort` already constrain selections to the
/// declared presets, so this only guards the edge where an effort reaches
/// the client before discovery has populated DeepSeek's metadata (e.g. a
/// global `--reasoning-effort` default on the very first turn). Per
/// DeepSeek's docs, `xhigh`/`max` -> `max`; everything else floors to
/// `high`, DeepSeek's default.
fn deepseek_reasoning_effort(effort: &str) -> &'static str {
    match effort.trim().to_ascii_lowercase().as_str() {
        "max" | "xhigh" => "max",
        _ => "high",
    }
}

fn reasoning_effort_from_default_parameters(
    default_parameters: Option<&serde_json::Value>,
) -> Option<String> {
    let reasoning = default_parameters?.get("reasoning")?;
    if let Some(effort) = reasoning.get("effort").and_then(|v| v.as_str()) {
        return Some(effort.to_string());
    }
    if reasoning
        .get("enabled")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return Some("medium".to_string());
    }
    None
}

impl ModelEntry {
    pub(crate) fn pricing(&self) -> Option<ModelPricing> {
        let pricing = self.pricing.as_ref()?;
        let input_cost_per_token_usd = pricing.prompt.parse::<f64>().ok()?;
        let output_cost_per_token_usd = pricing.completion.parse::<f64>().ok()?;
        Some(ModelPricing {
            input_cost_per_token_usd,
            output_cost_per_token_usd,
        })
    }

    pub(crate) fn supports_images(&self) -> Option<bool> {
        let modalities = self.architecture.as_ref()?.input_modalities.as_ref()?;
        Some(
            modalities
                .iter()
                .any(|modality| modality.eq_ignore_ascii_case("image")),
        )
    }

    pub(crate) fn supports_reasoning(&self) -> bool {
        self.supported_parameters
            .iter()
            .any(|param| param == "reasoning" || param == "include_reasoning")
            || self
                .default_parameters
                .as_ref()
                .and_then(|params| params.get("reasoning"))
                .is_some()
    }

    pub(crate) fn to_model_metadata(&self) -> ModelMetadata {
        if !self.supports_reasoning() {
            return ModelMetadata {
                id: self.id.clone(),
                default_reasoning_level: None,
                supported_reasoning_levels: Vec::new(),
                supports_images: self.supports_images(),
                context_length: self.context_length,
                pricing: self.pricing(),
            };
        }

        ModelMetadata {
            id: self.id.clone(),
            default_reasoning_level: reasoning_effort_from_default_parameters(
                self.default_parameters.as_ref(),
            )
            .or_else(|| Some("medium".to_string())),
            supported_reasoning_levels: openrouter_reasoning_presets(),
            supports_images: self.supports_images(),
            context_length: self.context_length,
            pricing: self.pricing(),
        }
    }
}

// SSE chunk types for streaming (extended for tool calls)

#[derive(Debug, Deserialize)]
struct ChatCompletionChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    /// Per OpenAI's `stream_options.include_usage`, the final SSE chunk
    /// before `[DONE]` carries a populated `usage` block with empty
    /// `choices`. Earlier chunks omit it (or send `null`). Optional so
    /// servers that don't support the flag still deserialize cleanly.
    #[serde(default)]
    usage: Option<UsageChunk>,
}

/// OpenAI / OpenRouter / Ollama trailing usage block. Field names
/// match the wire format (`prompt_tokens`, `completion_tokens`,
/// `prompt_tokens_details.cached_tokens`, etc.); we translate to the
/// ACP/anvil-internal `TokenUsage` shape at parse time so the rest of
/// the codebase only sees one vocabulary.
#[derive(Debug, Deserialize)]
struct UsageChunk {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct CompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: u64,
}

impl UsageChunk {
    /// Project OpenAI's wire fields onto the ACP-aligned `TokenUsage`.
    ///
    /// `prompt_tokens` is OpenAI's total input (cached + uncached). The
    /// ACP spec splits these into `input_tokens` (uncached input) and
    /// `cached_read_tokens`. We subtract so the two don't double-count
    /// in `total_tokens`. Similarly, `completion_tokens` is total
    /// output (reasoning + visible); we surface reasoning separately
    /// as `thought_tokens` and subtract from `output_tokens`.
    fn into_usage(self) -> TokenUsage {
        let cached_read = self
            .prompt_tokens_details
            .as_ref()
            .map(|d| d.cached_tokens)
            .unwrap_or(0);
        let reasoning = self
            .completion_tokens_details
            .as_ref()
            .map(|d| d.reasoning_tokens)
            .unwrap_or(0);
        TokenUsage {
            input_tokens: self.prompt_tokens.saturating_sub(cached_read),
            output_tokens: self.completion_tokens.saturating_sub(reasoning),
            thought_tokens: reasoning,
            cached_read_tokens: cached_read,
            // OpenAI's chat completions API doesn't separate cache
            // writes; that concept comes from Anthropic's prompt
            // caching. Leave at zero rather than guess.
            cached_write_tokens: 0,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    delta: ChunkDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug)]
struct ChunkDelta {
    content: Option<String>,
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<ChunkToolCall>>,
}

impl<'de> Deserialize<'de> for ChunkDelta {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawChunkDelta {
            #[serde(default)]
            content: Option<String>,
            #[serde(default)]
            reasoning_content: Option<String>,
            #[serde(default)]
            reasoning: Option<String>,
            #[serde(default)]
            tool_calls: Option<Vec<ChunkToolCall>>,
        }

        let raw = RawChunkDelta::deserialize(deserializer)?;
        Ok(Self {
            content: raw.content,
            reasoning_content: raw.reasoning_content.or(raw.reasoning),
            tool_calls: raw.tool_calls,
        })
    }
}

#[derive(Debug, Deserialize)]
struct ChunkToolCall {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<ChunkFunctionCall>,
}

#[derive(Debug, Deserialize)]
struct ChunkFunctionCall {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// Accumulator for tool calls arriving as SSE fragments.
#[derive(Default)]
struct ToolCallAccumulator {
    calls: HashMap<usize, (String, String, String)>, // index -> (id, name, arguments)
}

impl ToolCallAccumulator {
    fn push(&mut self, chunk: &ChunkToolCall) {
        let entry = self
            .calls
            .entry(chunk.index)
            .or_insert_with(|| (String::new(), String::new(), String::new()));
        if let Some(id) = &chunk.id {
            entry.0 = id.clone();
        }
        if let Some(func) = &chunk.function {
            if let Some(name) = &func.name {
                entry.1 = name.clone();
            }
            if let Some(args) = &func.arguments {
                entry.2.push_str(args);
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    fn into_tool_calls(self) -> Vec<ToolCall> {
        let mut entries: Vec<_> = self.calls.into_iter().collect();
        entries.sort_by_key(|(idx, _)| *idx);
        entries
            .into_iter()
            .map(|(_, (id, name, arguments))| ToolCall {
                id,
                r#type: "function".to_string(),
                function: FunctionCall { name, arguments },
            })
            .collect()
    }
}

fn normalize_stream_tool_calls(
    calls: Vec<ToolCall>,
    protocol: &'static str,
) -> Result<Vec<ToolCall>> {
    let mut normalized_calls = Vec::with_capacity(calls.len());
    for mut call in calls {
        match crate::tool_arguments::normalize_tool_arguments(&call.function.arguments) {
            Ok(normalized) => {
                if normalized.repaired {
                    tracing::warn!(
                        tool_call_id = %call.id,
                        tool_name = %call.function.name,
                        "repaired malformed streamed tool-call arguments"
                    );
                    call.function.arguments = normalized.arguments;
                }
                normalized_calls.push(call);
            }
            Err(err) if err.kind() == crate::tool_arguments::ToolArgumentErrorKind::Incomplete => {
                let error = anyhow::Error::new(IncompleteStreamError::new(
                    protocol,
                    "complete tool-call arguments",
                ));
                return Err(error.context(format!(
                    "incomplete tool-call arguments for {}",
                    call.function.name
                )));
            }
            Err(err) => {
                tracing::debug!(
                    tool_call_id = %call.id,
                    tool_name = %call.function.name,
                    error = %err,
                    "leaving unrepaired malformed streamed tool-call arguments"
                );
                normalized_calls.push(call);
            }
        }
    }
    Ok(normalized_calls)
}

// ---------------------------------------------------------------------------
// OpenAI-compatible client
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct OpenAiClient {
    base_url: String,
    api_key: Option<String>,
    http: reqwest::Client,
    /// How this client forwards reasoning effort on the wire (off, the
    /// unified object, or DeepSeek's thinking + top-level reasoning_effort).
    reasoning_wire: ReasoningWire,
    structured_output_wire: StructuredOutputWire,
}

impl std::fmt::Debug for OpenAiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiClient")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("reasoning_wire", &self.reasoning_wire)
            .field("structured_output_wire", &self.structured_output_wire)
            .finish()
    }
}

impl OpenAiClient {
    #[cfg(target_os = "android")]
    pub fn apply_runtime_tls_workarounds(
        mut builder: reqwest::ClientBuilder,
        target: &str,
    ) -> reqwest::ClientBuilder {
        let native = rustls_native_certs::load_native_certs();
        tracing::info!(
            target,
            certs = native.certs.len(),
            errors = native.errors.len(),
            "using native certs with tls_certs_only() for Android HTTP(S) client"
        );
        if target.contains("openrouter.ai") {
            crate::openrouter_auth::append_refresh_log(&format!(
                "Android HTTPS workaround for {target}: {} cert(s), {} error(s)",
                native.certs.len(),
                native.errors.len()
            ));
        }
        let certs = native
            .certs
            .into_iter()
            .filter_map(|cert| match reqwest::Certificate::from_der(cert.as_ref()) {
                Ok(cert) => Some(cert),
                Err(e) => {
                    tracing::warn!(
                        target,
                        "skipping native cert during reqwest conversion: {e}"
                    );
                    if target.contains("openrouter.ai") {
                        crate::openrouter_auth::append_refresh_log(&format!(
                            "OpenRouter native cert conversion skipped one cert: {e}"
                        ));
                    }
                    None
                }
            })
            .collect::<Vec<_>>();
        builder = builder.tls_certs_only(certs);
        builder
    }

    #[cfg(not(target_os = "android"))]
    pub fn apply_runtime_tls_workarounds(
        builder: reqwest::ClientBuilder,
        _target: &str,
    ) -> reqwest::ClientBuilder {
        builder
    }

    pub fn new(base_url: String, api_key: Option<String>) -> Self {
        Self::with_default_headers(base_url, api_key, reqwest::header::HeaderMap::new())
    }

    /// Like `new`, but attaches `default_headers` to every request the
    /// resulting `reqwest::Client` makes. Used by providers that require
    /// out-of-band attribution headers on every call (today, OpenRouter's
    /// optional `HTTP-Referer` / `X-Title` leaderboard headers). Plain
    /// OpenAI / Ollama callers should keep using `new`.
    pub fn with_default_headers(
        base_url: String,
        api_key: Option<String>,
        default_headers: reqwest::header::HeaderMap,
    ) -> Self {
        let base_url = base_url.trim_end_matches('/').to_string();
        let openrouter_mode = base_url.contains("openrouter.ai");
        let mut builder = Self::apply_runtime_tls_workarounds(
            reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(600))
                .default_headers(default_headers),
            &base_url,
        );
        if openrouter_mode {
            crate::openrouter_auth::append_refresh_log(
                "Configuring OpenRouter reqwest client: native certs + tls_certs_only + http1_only + connection_verbose + no idle pool",
            );
            builder = builder
                .http1_only()
                .connection_verbose(true)
                .pool_max_idle_per_host(0);
        }
        let http = builder.build().expect("failed to build HTTP client");
        Self {
            base_url,
            api_key,
            http,
            reasoning_wire: ReasoningWire::Off,
            // Default to OpenAI's strict JSON Schema response_format for
            // OpenAI-compatible servers. Known exceptions can override the
            // dialect in their constructor; hosted DeepSeek currently supports
            // JSON mode only.
            structured_output_wire: StructuredOutputWire::JsonSchema,
        }
    }

    /// Construct an `OpenAiClient` that forwards reasoning effort via the
    /// OpenRouter/Ollama unified `reasoning: { effort }` object.
    pub fn with_reasoning_support(
        base_url: String,
        api_key: Option<String>,
        default_headers: reqwest::header::HeaderMap,
    ) -> Self {
        let mut client = Self::with_default_headers(base_url, api_key, default_headers);
        client.reasoning_wire = ReasoningWire::Unified;
        client
    }

    /// Construct an `OpenAiClient` for the hosted DeepSeek API: it advertises
    /// DeepSeek's `high`/`max` reasoning levels (see
    /// [`deepseek_reasoning_presets`]) and forwards effort in DeepSeek's
    /// dialect (`thinking` + a top-level `reasoning_effort`).
    pub fn with_deepseek_reasoning_support(
        base_url: String,
        api_key: Option<String>,
        default_headers: reqwest::header::HeaderMap,
    ) -> Self {
        let mut client = Self::with_default_headers(base_url, api_key, default_headers);
        client.reasoning_wire = ReasoningWire::DeepSeek;
        client.structured_output_wire = StructuredOutputWire::JsonObject;
        client
    }

    fn api_url(&self, path: &str) -> String {
        if self.base_url.ends_with("/v1") {
            format!("{}{}", self.base_url, path)
        } else {
            format!("{}/v1{}", self.base_url, path)
        }
    }
}

impl LlmBackend for OpenAiClient {
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
        Box::pin(self.list_models_impl())
    }

    fn list_model_metadata(&self) -> BoxFuture<'_, Result<Vec<ModelMetadata>>> {
        Box::pin(self.list_model_metadata_impl())
    }

    fn stream_chat(&self, request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
        let request = if self.reasoning_wire == ReasoningWire::Off {
            StreamChatRequest {
                reasoning_effort: None,
                ..request
            }
        } else {
            request
        };
        Box::pin(self.stream_chat_impl(request))
    }
}

impl OpenAiClient {
    async fn fetch_models_response(&self) -> Result<ModelsResponse> {
        let url = self.api_url("/models");
        let trace_openrouter = self.base_url.contains("openrouter.ai");
        if trace_openrouter {
            crate::openrouter_auth::append_refresh_log(&format!(
                "OpenRouter fetch_models_response: building GET {url}"
            ));
        }
        let http = self.http.clone();
        let api_key = self.api_key.clone();
        let url_for_task = url.clone();
        let trace_for_task = trace_openrouter;
        let mut send_task = tokio::spawn(async move {
            let mut req = http
                .get(&url_for_task)
                .timeout(Duration::from_secs(MODEL_DISCOVERY_TIMEOUT_SECS));
            if let Some(key) = &api_key {
                req = req.bearer_auth(key);
            }
            if trace_for_task {
                crate::openrouter_auth::append_refresh_log(
                    "OpenRouter fetch_models_response: calling req.send()",
                );
            }
            req.send()
                .await
                .with_context(|| format!("GET {url_for_task}"))
        });
        let resp = tokio::select! {
            joined = &mut send_task => {
                match joined {
                    Ok(Ok(resp)) => resp,
                    Ok(Err(e)) => return Err(e),
                    Err(e) => {
                        if trace_openrouter {
                            crate::openrouter_auth::append_refresh_log(&format!(
                                "OpenRouter fetch_models_response: send task join error: {e}"
                            ));
                        }
                        anyhow::bail!("OpenRouter /models send task failed: {e}");
                    }
                }
            }
            _ = tokio::time::sleep(Duration::from_secs(MODEL_DISCOVERY_TASK_TIMEOUT_SECS)) => {
                if trace_openrouter {
                    crate::openrouter_auth::append_refresh_log(
                        "OpenRouter fetch_models_response: send task timed out; aborting task",
                    );
                }
                send_task.abort();
                anyhow::bail!(
                    "GET {url} timed out after {}s waiting for req.send task",
                    MODEL_DISCOVERY_TASK_TIMEOUT_SECS
                );
            }
        };
        if trace_openrouter {
            crate::openrouter_auth::append_refresh_log(
                "OpenRouter fetch_models_response: req.send() returned",
            );
        }
        let status = resp.status();
        if trace_openrouter {
            crate::openrouter_auth::append_refresh_log(&format!(
                "OpenRouter fetch_models_response: HTTP {status}"
            ));
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            if trace_openrouter {
                crate::openrouter_auth::append_refresh_log(&format!(
                    "OpenRouter fetch_models_response: error body length {}",
                    body.len()
                ));
            }
            tracing::warn!("model discovery failed (HTTP {status}): {body}");
            anyhow::bail!("model discovery failed (HTTP {status})");
        }

        if trace_openrouter {
            crate::openrouter_auth::append_refresh_log(
                "OpenRouter fetch_models_response: parsing JSON body",
            );
        }
        let parsed: ModelsResponse = resp
            .json()
            .await
            .context("failed to parse models response")?;
        if trace_openrouter {
            crate::openrouter_auth::append_refresh_log(&format!(
                "OpenRouter fetch_models_response: parsed {} model entries",
                parsed.data.len()
            ));
        }
        Ok(parsed)
    }

    async fn list_models_impl(&self) -> Result<Vec<String>> {
        let models = self.fetch_models_response().await?;
        Ok(models.data.into_iter().map(|m| m.id).collect())
    }

    async fn list_model_metadata_impl(&self) -> Result<Vec<ModelMetadata>> {
        if self.base_url.contains("openrouter.ai") {
            crate::openrouter_auth::append_refresh_log(
                "OpenRouter list_model_metadata_impl: start",
            );
        }
        let models = self.fetch_models_response().await?;
        if self.base_url.contains("openrouter.ai") {
            crate::openrouter_auth::append_refresh_log(
                "OpenRouter list_model_metadata_impl: fetched models response",
            );
        }
        if self.reasoning_wire == ReasoningWire::Off {
            return Ok(models
                .data
                .into_iter()
                .map(|model| model.to_model_metadata())
                .collect());
        }
        if self.reasoning_wire == ReasoningWire::DeepSeek {
            // DeepSeek's `/models` is id-only, so `to_model_metadata` would
            // yield no reasoning info. Declare DeepSeek's levels here (default
            // `high`) so the shared picker/selection code can map requests to
            // availability, exactly as it does for Bedrock/Ollama.
            let presets = deepseek_reasoning_presets();
            return Ok(models
                .data
                .into_iter()
                .map(|model| ModelMetadata {
                    id: model.id.clone(),
                    default_reasoning_level: Some("high".to_string()),
                    supported_reasoning_levels: presets.clone(),
                    supports_images: model.supports_images(),
                    context_length: model.context_length,
                    pricing: model.pricing(),
                })
                .collect());
        }
        let metadata = models
            .data
            .into_iter()
            .map(|model| model.to_model_metadata())
            .collect::<Vec<_>>();
        if self.base_url.contains("openrouter.ai") {
            crate::openrouter_auth::append_refresh_log(&format!(
                "OpenRouter list_model_metadata_impl: built {} metadata entries",
                metadata.len()
            ));
        }
        Ok(metadata)
    }

    async fn stream_chat_impl(&self, request: StreamChatRequest) -> Result<LlmResponse> {
        let StreamChatRequest {
            model,
            messages,
            tools,
            reasoning_effort,
            temperature,
            structured_output,
            on_token,
            on_thought,
            cancel,
            idle_timeouts,
        } = request;
        let url = self.api_url("/chat/completions");

        let tool_choice = tools.as_ref().map(|_| "auto".to_string());

        // Spell reasoning effort in this client's dialect. `stream_chat` has
        // already cleared the effort for `ReasoningWire::Off`.
        let (reasoning, thinking, reasoning_effort_field) = match self.reasoning_wire {
            ReasoningWire::Off => (None, None, None),
            ReasoningWire::Unified => (
                reasoning_effort.map(|effort| ReasoningConfig { effort }),
                None,
                None,
            ),
            ReasoningWire::DeepSeek => match reasoning_effort {
                Some(effort) => (
                    None,
                    Some(ThinkingConfig { r#type: "enabled" }),
                    Some(deepseek_reasoning_effort(&effort).to_string()),
                ),
                // No effort: send nothing; DeepSeek defaults to thinking
                // enabled at `high`.
                None => (None, None, None),
            },
        };
        let response_format = structured_output.as_ref().map(|request| {
            // A request can opt down to basic JSON mode regardless of the
            // client's default wire (e.g. the permission classifier, to stay
            // compatible with OpenRouter providers that reject strict schema).
            let wire = if request.prefer_json_object {
                StructuredOutputWire::JsonObject
            } else {
                self.structured_output_wire
            };
            let format = crate::structured_output::native_response_format(request);
            match wire {
                StructuredOutputWire::JsonSchema => ChatCompletionResponseFormat::JsonSchema {
                    json_schema: NativeJsonSchemaFormat {
                        name: format.name,
                        schema: format.schema,
                        strict: format.strict,
                    },
                },
                StructuredOutputWire::JsonObject => ChatCompletionResponseFormat::JsonObject,
            }
        });

        let body = ChatCompletionRequest {
            model,
            messages,
            stream: true,
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
            temperature,
            max_tokens: None,
            tools,
            tool_choice,
            reasoning,
            thinking,
            reasoning_effort: reasoning_effort_field,
            response_format,
        };
        let resp = match crate::http_retry::send_with_retries(
            "sending chat completion request",
            || {
                let mut req = self.http.post(&url).json(&body);
                if let Some(key) = &self.api_key {
                    req = req.bearer_auth(key);
                }
                req
            },
            Some(&cancel),
        )
        .await
        {
            Ok(resp) => resp,
            Err(e) => return Err(e),
        };
        let status = resp.status();
        if !status.is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(crate::http_retry::retryable_llm_error_for_body(
                format!("chat completion failed (HTTP {status}): {body_text}"),
                &body_text,
            ));
        }

        let stream = resp
            .bytes_stream()
            .map(|r| r.map(|b| b.to_vec()).map_err(anyhow::Error::from));

        drive_sse_stream(stream, on_token, on_thought, cancel, idle_timeouts).await
    }
}

/// Drive an SSE byte stream until the LLM emits `[DONE]` or the
/// cancellation token fires. Aborts with a clear error if the stream
/// ends before `[DONE]`, or if
/// no *meaningful progress* (parsed `data:` event contributing content
/// or tool-call deltas, or `[DONE]`) is observed within the current
/// first-progress or inter-chunk timeout. SSE
/// keepalive comments (`:\n`), blank lines, and partial bytes that
/// don't advance the parser do NOT reset the deadline, so a server or
/// proxy that drip-feeds pings every <90s can no longer hold the
/// request open indefinitely.
async fn drive_sse_stream<S>(
    mut stream: S,
    mut on_token: TokenSink,
    mut on_thought: TokenSink,
    cancel: CancellationToken,
    idle: IdleTimeouts,
) -> Result<LlmResponse>
where
    S: Stream<Item = Result<Vec<u8>>> + Unpin,
{
    let mut full_text = String::new();
    let mut full_reasoning = String::new();
    let mut tool_acc = ToolCallAccumulator::default();
    let mut raw_buf: Vec<u8> = Vec::new();
    let mut deadline = tokio::time::Instant::now() + idle.first_progress;
    let mut saw_progress = false;
    // Captured from the trailing `usage` chunk emitted when
    // `stream_options.include_usage = true`. The usage chunk arrives
    // *before* `[DONE]` with an empty `choices` array, so we stash it
    // here and attach to whichever variant we return.
    let mut usage = TokenUsage::default();
    let mut last_finish_reason: Option<String> = None;

    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                tracing::info!("streaming cancelled by client");
                break;
            }
            chunk_or_timeout = tokio::time::timeout_at(deadline, stream.next()) => {
                let chunk_opt = match chunk_or_timeout {
                    Ok(opt) => opt,
                    Err(_elapsed) => {
                        if saw_progress {
                            return Err(crate::http_retry::retryable_llm_error(
                                format!(
                                "LLM stream stalled mid-stream for {}s; aborting (server-side hang or keepalive-only flood)",
                                idle.inter_chunk.as_secs()
                                ),
                                RetryableLlmError::fast("stream stalled mid-stream"),
                            ));
                        }
                        return Err(crate::http_retry::retryable_llm_error(
                            format!(
                            "LLM stream made no first token for {}s; aborting (server-side hang or keepalive-only flood)",
                            idle.first_progress.as_secs()
                            ),
                            RetryableLlmError::fast("stream made no first token"),
                        ));
                    }
                };
                let eof_after_buffer = if let Some(chunk) = chunk_opt {
                    let chunk = chunk.map_err(|err| {
                        crate::http_retry::retryable_llm_context(
                            err,
                            "stream read error",
                            RetryableLlmError::fast("stream read error"),
                        )
                    })?;
                    raw_buf.extend_from_slice(&chunk);
                    false
                } else if raw_buf.is_empty() {
                    break;
                } else {
                    raw_buf.push(b'\n');
                    true
                };

                let mut made_progress = false;

                while let Some(pos) = raw_buf.iter().position(|&b| b == b'\n') {
                    let line_bytes = raw_buf.drain(..=pos).collect::<Vec<_>>();
                    let line = String::from_utf8_lossy(&line_bytes).trim().to_string();

                    if line.is_empty() || line.starts_with(':') {
                        continue;
                    }

                    let data = if let Some(stripped) = line.strip_prefix("data: ") {
                        stripped.trim()
                    } else {
                        continue;
                    };

                    if data == "[DONE]" {
                        let output_budget_exhausted =
                            last_finish_reason.as_deref() == Some("length");
                        let reasoning_content = (!full_reasoning.is_empty()).then_some(full_reasoning);
                        if tool_acc.is_empty() {
                            if output_budget_exhausted {
                                if full_text.trim().is_empty() {
                                    return Err(anyhow::Error::new(OutputBudgetExhaustedError));
                                }
                                tracing::warn!(
                                    "chat completions stream ended with finish_reason=length after emitting text; returning truncated content"
                                );
                            }
                            return Ok(LlmResponse::Text {
                                text: full_text,
                                reasoning_content,
                                usage,
                            });
                        }
                        if output_budget_exhausted {
                            tracing::warn!(
                                "chat completions stream ended with finish_reason=length after emitting tool calls; returning truncated content"
                            );
                        }
                        return Ok(LlmResponse::ToolCalls {
                            text: full_text,
                            reasoning_content,
                            calls: normalize_stream_tool_calls(
                                tool_acc.into_tool_calls(),
                                "chat completions SSE",
                            )?,
                            usage,
                        });
                    }

                    match serde_json::from_str::<ChatCompletionChunk>(data) {
                        Ok(chunk) => {
                            for choice in &chunk.choices {
                                if let Some(reason) = choice
                                    .finish_reason
                                    .as_deref()
                                    .map(str::trim)
                                    .filter(|reason| !reason.is_empty())
                                {
                                    last_finish_reason = Some(reason.to_string());
                                }
                                // Accumulate text content
                                if let Some(content) = &choice.delta.content {
                                    made_progress = true;
                                    on_token(content);
                                    full_text.push_str(content);
                                }
                                if let Some(reasoning_content) = &choice.delta.reasoning_content {
                                    made_progress = true;
                                    on_thought(reasoning_content);
                                    full_reasoning.push_str(reasoning_content);
                                }
                                // Accumulate tool call fragments
                                if let Some(tc_chunks) = &choice.delta.tool_calls {
                                    if !tc_chunks.is_empty() {
                                        made_progress = true;
                                    }
                                    for tc in tc_chunks {
                                        tool_acc.push(tc);
                                    }
                                }
                            }
                            // Last chunk before [DONE]: trailing usage
                            // block. `choices` is empty here; we just
                            // record the totals.
                            if let Some(u) = chunk.usage {
                                usage = u.into_usage();
                                made_progress = true;
                            }
                        }
                        Err(e) => {
                            tracing::debug!("skipping unparseable SSE chunk: {e}");
                        }
                    }
                }

                if made_progress {
                    saw_progress = true;
                    deadline = tokio::time::Instant::now() + idle.inter_chunk;
                }
                if eof_after_buffer {
                    break;
                }
            }
        }
    }

    // If we exited the loop via cancellation, tool call fragments may be incomplete
    // (arguments JSON truncated mid-stream). Return only the text we've already
    // streamed to the caller to avoid dispatching malformed tool calls.
    if cancel.is_cancelled() {
        return Ok(LlmResponse::Text {
            text: full_text,
            reasoning_content: (!full_reasoning.is_empty()).then_some(full_reasoning),
            usage,
        });
    }

    Err(anyhow::Error::new(IncompleteStreamError::new(
        "chat completions SSE",
        "[DONE]",
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structured_output::StructuredOutputRequest;
    use futures::StreamExt;
    use futures::stream;
    use futures::stream::BoxStream;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn collect_tokens() -> (TokenSink, Arc<Mutex<Vec<String>>>) {
        let collected = Arc::new(Mutex::new(Vec::<String>::new()));
        let inner = collected.clone();
        let cb: TokenSink = Box::new(move |t| {
            inner.lock().unwrap().push(t.to_string());
        });
        (cb, collected)
    }

    fn delayed_chunks(
        chunks: Vec<(Duration, &'static str)>,
    ) -> BoxStream<'static, Result<Vec<u8>>> {
        stream::unfold(chunks.into_iter(), |mut chunks| async move {
            let (delay, chunk) = chunks.next()?;
            tokio::time::sleep(delay).await;
            Some((Ok(chunk.as_bytes().to_vec()), chunks))
        })
        .boxed()
    }

    struct EmptyThenOkBackend {
        attempts: Arc<AtomicUsize>,
    }

    impl LlmBackend for EmptyThenOkBackend {
        fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
            Box::pin(async { Ok(Vec::new()) })
        }

        fn stream_chat(
            &self,
            mut request: StreamChatRequest,
        ) -> BoxFuture<'_, Result<LlmResponse>> {
            let attempts = self.attempts.clone();
            Box::pin(async move {
                let attempt = attempts.fetch_add(1, Ordering::SeqCst) + 1;
                if attempt == 1 {
                    return Ok(LlmResponse::Text {
                        text: String::new(),
                        reasoning_content: None,
                        usage: TokenUsage::default(),
                    });
                }
                (request.on_token)("ok");
                Ok(LlmResponse::Text {
                    text: "ok".to_string(),
                    reasoning_content: None,
                    usage: TokenUsage::default(),
                })
            })
        }
    }

    struct OutputBudgetErrorBackend {
        attempts: Arc<AtomicUsize>,
    }

    impl LlmBackend for OutputBudgetErrorBackend {
        fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
            Box::pin(async { Ok(Vec::new()) })
        }

        fn stream_chat(&self, _request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
            let attempts = self.attempts.clone();
            Box::pin(async move {
                attempts.fetch_add(1, Ordering::SeqCst);
                Err(anyhow::Error::new(OutputBudgetExhaustedError))
            })
        }
    }

    struct RetryThenSseResponder {
        calls: Arc<AtomicUsize>,
    }

    impl wiremock::Respond for RetryThenSseResponder {
        fn respond(&self, _request: &wiremock::Request) -> ResponseTemplate {
            let attempt = self.calls.fetch_add(1, Ordering::SeqCst);
            if attempt == 0 {
                ResponseTemplate::new(500).set_body_string("temporary overload")
            } else {
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(
                        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\
                         \n\
                         data: [DONE]\n\n",
                    )
            }
        }
    }

    #[test]
    fn retryable_llm_error_classifies_gateway_markers_only_as_long_tier() {
        let gateway = crate::http_retry::retryable_llm_error_for_body(
            "chat completion failed (HTTP 400): JSON-RPC error -32602: Job registration failed",
            "JSON-RPC error -32602: Job registration failed",
        )
        .context("teacher rollout request failed");
        assert_eq!(
            llm_retry_tier(&gateway),
            Some(crate::http_retry::LlmRetryTier::GatewayTransient)
        );
        assert!(is_retryable_llm_error(&gateway));

        let standard = crate::http_retry::retryable_llm_error_for_responses_failure(
            "Responses stream failed: server_error: overloaded",
            "server_error: overloaded",
        );
        assert_eq!(
            llm_retry_tier(&standard),
            Some(crate::http_retry::LlmRetryTier::Fast)
        );

        let validation =
            anyhow::anyhow!("chat completion failed (HTTP 400): missing required field messages");
        assert_eq!(llm_retry_tier(&validation), None);
        assert!(!is_retryable_llm_error(&validation));

        let quoted_upstream_text =
            anyhow::anyhow!("tool output mentioned server_error but the request was valid");
        assert_eq!(llm_retry_tier(&quoted_upstream_text), None);
    }

    #[tokio::test]
    async fn no_visible_output_retry_recovers_empty_completion() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let backend = EmptyThenOkBackend {
            attempts: attempts.clone(),
        };
        let cancel = CancellationToken::new();

        let response = stream_chat_no_visible_output_with_retry(
            &backend,
            "empty completion test",
            &cancel,
            || StreamChatRequest {
                model: "test-model".to_string(),
                messages: vec![ChatMessage::user("hello")],
                tools: None,
                reasoning_effort: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: cancel.clone(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(30)),
            },
        )
        .await
        .expect("empty completion should retry and recover");

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert!(matches!(response, LlmResponse::Text { text, .. } if text == "ok"));
    }

    #[tokio::test]
    async fn no_visible_output_retry_does_not_retry_output_budget_exhaustion() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let backend = OutputBudgetErrorBackend {
            attempts: attempts.clone(),
        };
        let cancel = CancellationToken::new();

        let err = stream_chat_no_visible_output_with_retry(
            &backend,
            "output budget test",
            &cancel,
            || StreamChatRequest {
                model: "test-model".to_string(),
                messages: vec![ChatMessage::user("hello")],
                tools: None,
                reasoning_effort: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: cancel.clone(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(30)),
            },
        )
        .await
        .expect_err("output budget exhaustion should not retry");

        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert!(is_output_budget_exhausted_error(&err));
        assert!(!is_retryable_llm_error(&err));
    }

    #[tokio::test]
    async fn openai_client_retries_transient_5xx_before_streaming() {
        let server = MockServer::start().await;
        let calls = Arc::new(AtomicUsize::new(0));

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(RetryThenSseResponder {
                calls: calls.clone(),
            })
            .mount(&server)
            .await;

        let client = OpenAiClient::new(server.uri(), None);
        let (on_token, tokens) = collect_tokens();
        let result = client
            .stream_chat(StreamChatRequest {
                model: "test-model".to_string(),
                messages: vec![ChatMessage::user("hello")],
                tools: None,
                reasoning_effort: None,
                temperature: None,
                structured_output: None,
                on_token,
                on_thought: Box::new(|_| {}),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(30)),
            })
            .await
            .expect("retry should recover");

        match result {
            LlmResponse::Text { text, .. } => assert_eq!(text, "ok"),
            other => panic!("expected text response, got {other:?}"),
        }
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(*tokens.lock().unwrap(), vec!["ok".to_string()]);
    }

    #[tokio::test]
    async fn openai_client_non_success_error_includes_response_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(400)
                    .set_body_string("JSON-RPC error -32602: Task submission failed"),
            )
            .mount(&server)
            .await;

        let client = OpenAiClient::new(server.uri(), None);
        let (on_token, _) = collect_tokens();
        let err = client
            .stream_chat(StreamChatRequest {
                model: "test-model".to_string(),
                messages: vec![ChatMessage::user("hello")],
                tools: None,
                reasoning_effort: None,
                temperature: None,
                structured_output: None,
                on_token,
                on_thought: Box::new(|_| {}),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(30)),
            })
            .await
            .expect_err("HTTP 400 should fail");

        let message = format!("{err:#}");
        assert!(message.contains("HTTP 400"), "{message}");
        assert!(message.contains("Task submission failed"), "{message}");
        assert_eq!(
            llm_retry_tier(&err),
            Some(crate::http_retry::LlmRetryTier::GatewayTransient)
        );
    }

    #[test]
    fn deepseek_reasoning_effort_clamps_to_high_or_max() {
        // Only `high`/`max` are valid for DeepSeek. xhigh/max -> max;
        // everything else (low/medium/minimal/unknown) floors to high.
        assert_eq!(deepseek_reasoning_effort("low"), "high");
        assert_eq!(deepseek_reasoning_effort("medium"), "high");
        assert_eq!(deepseek_reasoning_effort("high"), "high");
        assert_eq!(deepseek_reasoning_effort("xhigh"), "max");
        assert_eq!(deepseek_reasoning_effort("max"), "max");
        assert_eq!(deepseek_reasoning_effort("minimal"), "high");
        assert_eq!(deepseek_reasoning_effort("  MAX "), "max");
    }

    /// DeepSeek `/models` is id-only, so the client declares the levels
    /// itself: every model gets `[high, max]` with `high` as the default.
    /// This is what lets the shared picker/selection map requests to
    /// availability.
    #[tokio::test]
    async fn deepseek_wire_declares_high_max_reasoning_levels() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [
                    {"id": "deepseek-reasoner", "object": "model", "owned_by": "deepseek"},
                    {"id": "deepseek-chat", "object": "model", "owned_by": "deepseek"}
                ]
            })))
            .mount(&server)
            .await;
        let client = OpenAiClient::with_deepseek_reasoning_support(
            server.uri(),
            None,
            reqwest::header::HeaderMap::new(),
        );

        let meta = client.list_model_metadata().await.expect("metadata");
        assert_eq!(meta.len(), 2);
        for m in &meta {
            assert_eq!(m.default_reasoning_level.as_deref(), Some("high"), "{m:?}");
            let levels: Vec<&str> = m
                .supported_reasoning_levels
                .iter()
                .map(|p| p.effort.as_str())
                .collect();
            assert_eq!(levels, vec!["high", "max"], "{m:?}");
        }
    }

    /// Capture the JSON body a client sends for one chat request via a mock
    /// server returning an empty SSE stream.
    async fn capture_request_body(wire: ReasoningWire, effort: Option<&str>) -> serde_json::Value {
        capture_request_body_with_structured_output(wire, effort, None).await
    }

    async fn capture_request_body_with_structured_output(
        wire: ReasoningWire,
        effort: Option<&str>,
        structured_output: Option<StructuredOutputRequest>,
    ) -> serde_json::Value {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string("data: [DONE]\n"))
            .mount(&server)
            .await;
        let client = match wire {
            ReasoningWire::DeepSeek => OpenAiClient::with_deepseek_reasoning_support(
                server.uri(),
                None,
                reqwest::header::HeaderMap::new(),
            ),
            ReasoningWire::Unified => OpenAiClient::with_reasoning_support(
                server.uri(),
                None,
                reqwest::header::HeaderMap::new(),
            ),
            ReasoningWire::Off => OpenAiClient::new(server.uri(), None),
        };
        let (on_token, _) = collect_tokens();
        client
            .stream_chat(StreamChatRequest {
                model: "deepseek-reasoner".to_string(),
                messages: vec![ChatMessage::user("hi")],
                tools: None,
                reasoning_effort: effort.map(str::to_string),
                temperature: None,
                structured_output,
                on_token,
                on_thought: Box::new(|_| {}),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(30)),
            })
            .await
            .expect("mock stream completes");
        let requests = server.received_requests().await.expect("recorded requests");
        assert_eq!(requests.len(), 1);
        serde_json::from_slice(&requests[0].body).expect("body is JSON")
    }

    fn test_structured_output_request() -> StructuredOutputRequest {
        StructuredOutputRequest {
            schema_name: "audit_result".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"]
            }),
            allow_coercion: false,
            prefer_json_object: false,
        }
    }

    /// DeepSeek wire sends `thinking: {type: enabled}` + a top-level
    /// `reasoning_effort` (clamped to DeepSeek's scale), and NOT the unified
    /// `reasoning` object.
    #[tokio::test]
    async fn deepseek_wire_sends_thinking_and_top_level_effort() {
        let body = capture_request_body(ReasoningWire::DeepSeek, Some("medium")).await;
        assert_eq!(body["thinking"]["type"], "enabled", "{body}");
        assert_eq!(body["reasoning_effort"], "high", "{body}");
        assert!(body.get("reasoning").is_none(), "{body}");
    }

    /// With no effort, the DeepSeek wire sends none of the reasoning fields
    /// and lets DeepSeek apply its default.
    #[tokio::test]
    async fn deepseek_wire_omits_reasoning_when_no_effort() {
        let body = capture_request_body(ReasoningWire::DeepSeek, None).await;
        assert!(body.get("thinking").is_none(), "{body}");
        assert!(body.get("reasoning_effort").is_none(), "{body}");
        assert!(body.get("reasoning").is_none(), "{body}");
    }

    #[tokio::test]
    async fn generic_openai_compatible_wire_sends_strict_json_schema_response_format() {
        let body = capture_request_body_with_structured_output(
            ReasoningWire::Off,
            None,
            Some(test_structured_output_request()),
        )
        .await;
        assert_eq!(body["response_format"]["type"], "json_schema", "{body}");
        assert_eq!(
            body["response_format"]["json_schema"]["name"], "audit_result",
            "{body}"
        );
        assert_eq!(
            body["response_format"]["json_schema"]["schema"]["type"], "object",
            "{body}"
        );
        assert_eq!(
            body["response_format"]["json_schema"]["strict"], true,
            "{body}"
        );
    }

    #[tokio::test]
    async fn deepseek_wire_sends_json_object_response_format_for_structured_output() {
        let body = capture_request_body_with_structured_output(
            ReasoningWire::DeepSeek,
            None,
            Some(test_structured_output_request()),
        )
        .await;
        assert_eq!(body["response_format"]["type"], "json_object", "{body}");
        assert!(
            body["response_format"].get("json_schema").is_none(),
            "{body}"
        );
    }

    /// A request opting into `prefer_json_object` downgrades to basic JSON
    /// mode even on a strict-json_schema client (the OpenRouter default), so
    /// the permission classifier stays compatible with providers that reject
    /// strict schema. The json_schema block must be absent.
    #[tokio::test]
    async fn prefer_json_object_downgrades_strict_schema_client_to_json_object() {
        let mut request = test_structured_output_request();
        request.prefer_json_object = true;
        let body = capture_request_body_with_structured_output(
            ReasoningWire::Unified, // strict json_schema wire by default
            None,
            Some(request),
        )
        .await;
        assert_eq!(body["response_format"]["type"], "json_object", "{body}");
        assert!(
            body["response_format"].get("json_schema").is_none(),
            "{body}"
        );
    }

    /// The unified wire still sends `reasoning: {effort}` verbatim and none
    /// of DeepSeek's fields -- guards against dialect leakage.
    #[tokio::test]
    async fn unified_wire_sends_reasoning_object() {
        let body = capture_request_body(ReasoningWire::Unified, Some("high")).await;
        assert_eq!(body["reasoning"]["effort"], "high", "{body}");
        assert!(body.get("thinking").is_none(), "{body}");
        assert!(body.get("reasoning_effort").is_none(), "{body}");
    }

    /// A stream that emits only SSE keepalive comments (`:\n`) must trip
    /// the deadline -- otherwise a server can hold the request open
    /// forever by drip-feeding pings. Regression for the failure mode
    /// flagged in PR #3510 review.
    #[tokio::test(start_paused = true)]
    async fn drive_sse_stream_bails_on_keepalive_only_chunks() {
        // One keepalive, then permanently pending. Auto-advance under
        // `start_paused` will roll forward to the deadline since no task
        // is ready -- the keepalive line does NOT count as progress.
        let chunks: Vec<Result<Vec<u8>>> = vec![Ok(b":keepalive\n".to_vec())];
        let s = stream::iter(chunks).chain(stream::pending());

        let (on_token, _) = collect_tokens();
        let cancel = CancellationToken::new();
        let result = drive_sse_stream(
            s,
            on_token,
            Box::new(|_| {}),
            cancel,
            IdleTimeouts::uniform(Duration::from_secs(90)),
        )
        .await;

        let err = result.expect_err("keepalive-only stream should bail");
        let msg = err.to_string();
        assert!(
            msg.contains("no first token for 90s"),
            "error should mention the first-token timeout, got: {msg}"
        );
        assert!(is_retryable_llm_error(&err));
    }

    #[tokio::test(start_paused = true)]
    async fn drive_sse_stream_allows_slow_first_progress() {
        let stream = delayed_chunks(vec![
            (
                Duration::from_secs(5),
                "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n",
            ),
            (Duration::ZERO, "data: [DONE]\n"),
        ]);

        let (on_token, collected) = collect_tokens();
        let result = drive_sse_stream(
            stream,
            on_token,
            Box::new(|_| {}),
            CancellationToken::new(),
            IdleTimeouts {
                first_progress: Duration::from_secs(10),
                inter_chunk: Duration::from_secs(2),
            },
        )
        .await;

        match result.expect("first chunk may arrive after the stall timeout") {
            LlmResponse::Text { text, .. } => assert_eq!(text, "ok"),
            other => panic!("expected text response, got {other:?}"),
        }
        assert_eq!(*collected.lock().unwrap(), vec!["ok"]);
    }

    #[tokio::test(start_paused = true)]
    async fn drive_sse_stream_times_out_on_mid_stream_stall() {
        let stream = delayed_chunks(vec![
            (
                Duration::ZERO,
                "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n",
            ),
            (Duration::from_secs(3), "data: [DONE]\n"),
        ]);

        let (on_token, _) = collect_tokens();
        let err = drive_sse_stream(
            stream,
            on_token,
            Box::new(|_| {}),
            CancellationToken::new(),
            IdleTimeouts {
                first_progress: Duration::from_secs(10),
                inter_chunk: Duration::from_secs(2),
            },
        )
        .await
        .expect_err("mid-stream stall should abort");
        let msg = format!("{err:#}");

        assert!(msg.contains("stalled mid-stream for 2s"), "got: {msg}");
        assert!(is_retryable_llm_error(&err));
    }

    #[tokio::test(start_paused = true)]
    async fn drive_sse_stream_times_out_waiting_for_first_progress() {
        let stream = delayed_chunks(vec![(
            Duration::from_secs(3),
            "data: {\"choices\":[{\"delta\":{\"content\":\"late\"}}]}\n",
        )]);

        let (on_token, _) = collect_tokens();
        let err = drive_sse_stream(
            stream,
            on_token,
            Box::new(|_| {}),
            CancellationToken::new(),
            IdleTimeouts {
                first_progress: Duration::from_secs(2),
                inter_chunk: Duration::from_secs(1),
            },
        )
        .await
        .expect_err("missing first progress should abort");
        let msg = format!("{err:#}");

        assert!(msg.contains("no first token for 2s"), "got: {msg}");
        assert!(is_retryable_llm_error(&err));
    }

    /// A stream that emits content data resets the deadline. Mixed with
    /// many keepalives, the helper must still complete normally.
    #[tokio::test(start_paused = true)]
    async fn drive_sse_stream_resets_deadline_on_content_chunks() {
        let chunks: Vec<Result<Vec<u8>>> = vec![
            Ok(b":keepalive\n".to_vec()),
            Ok(b"data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n".to_vec()),
            Ok(b":keepalive\n".to_vec()),
            Ok(b"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n".to_vec()),
            Ok(b"data: [DONE]\n".to_vec()),
        ];
        let s = stream::iter(chunks);

        let (on_token, collected) = collect_tokens();
        let cancel = CancellationToken::new();
        let result = drive_sse_stream(
            s,
            on_token,
            Box::new(|_| {}),
            cancel,
            IdleTimeouts::uniform(Duration::from_secs(90)),
        )
        .await;

        match result.expect("should complete") {
            LlmResponse::Text { text: t, .. } => assert_eq!(t, "hello"),
            other => panic!("expected text response, got {other:?}"),
        }
        assert_eq!(*collected.lock().unwrap(), vec!["hel", "lo"]);
    }

    /// `[DONE]` ends the stream cleanly with whatever has been accumulated.
    #[tokio::test]
    async fn drive_sse_stream_returns_text_on_done() {
        let chunks: Vec<Result<Vec<u8>>> = vec![
            Ok(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n".to_vec()),
            Ok(b"data: [DONE]\n".to_vec()),
        ];
        let s = stream::iter(chunks);

        let (on_token, collected) = collect_tokens();
        let cancel = CancellationToken::new();
        let result = drive_sse_stream(
            s,
            on_token,
            Box::new(|_| {}),
            cancel,
            IdleTimeouts::uniform(Duration::from_secs(90)),
        )
        .await;

        match result.expect("should complete") {
            LlmResponse::Text { text: t, .. } => assert_eq!(t, "hi"),
            other => panic!("expected text response, got {other:?}"),
        }
        assert_eq!(*collected.lock().unwrap(), vec!["hi"]);
    }

    #[tokio::test]
    async fn drive_sse_stream_errors_on_empty_length_finish_reason() {
        let chunks: Vec<Result<Vec<u8>>> = vec![
            Ok(b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking\"},\"finish_reason\":\"length\"}]}\n".to_vec()),
            Ok(b"data: [DONE]\n".to_vec()),
        ];
        let s = stream::iter(chunks);

        let (on_token, _) = collect_tokens();
        let thoughts = Arc::new(Mutex::new(Vec::<String>::new()));
        let thoughts_for_cb = Arc::clone(&thoughts);
        let on_thought: TokenSink = Box::new(move |s: &str| {
            thoughts_for_cb.lock().unwrap().push(s.to_string());
        });
        let err = drive_sse_stream(
            s,
            on_token,
            on_thought,
            CancellationToken::new(),
            IdleTimeouts::uniform(Duration::from_secs(90)),
        )
        .await
        .expect_err("empty length-limited completion should be classified");

        assert!(is_output_budget_exhausted_error(&err));
        assert!(!is_retryable_llm_error(&err));
        assert_eq!(*thoughts.lock().unwrap(), vec!["thinking".to_string()]);
    }

    #[tokio::test]
    async fn drive_sse_stream_returns_truncated_text_on_length_finish_reason() {
        let chunks: Vec<Result<Vec<u8>>> = vec![
            Ok(b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":\"length\"}]}\n".to_vec()),
            Ok(b"data: [DONE]\n".to_vec()),
        ];
        let s = stream::iter(chunks);

        let (on_token, collected) = collect_tokens();
        let result = drive_sse_stream(
            s,
            on_token,
            Box::new(|_| {}),
            CancellationToken::new(),
            IdleTimeouts::uniform(Duration::from_secs(90)),
        )
        .await
        .expect("length after text should return truncated content");

        match result {
            LlmResponse::Text { text, .. } => assert_eq!(text, "partial"),
            other => panic!("expected text response, got {other:?}"),
        }
        assert_eq!(*collected.lock().unwrap(), vec!["partial".to_string()]);
    }

    #[tokio::test]
    async fn drive_sse_stream_ignores_empty_finish_reason_and_allows_empty_stop() {
        for finish_reason in ["", "stop"] {
            let raw = format!(
                "data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"{finish_reason}\"}}]}}\n"
            );
            let chunks: Vec<Result<Vec<u8>>> =
                vec![Ok(raw.into_bytes()), Ok(b"data: [DONE]\n".to_vec())];
            let s = stream::iter(chunks);

            let (on_token, _) = collect_tokens();
            let result = drive_sse_stream(
                s,
                on_token,
                Box::new(|_| {}),
                CancellationToken::new(),
                IdleTimeouts::uniform(Duration::from_secs(90)),
            )
            .await
            .expect("non-length empty completions keep existing behavior");

            match result {
                LlmResponse::Text { text, .. } => assert!(text.is_empty()),
                other => panic!("expected text response, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn drive_sse_stream_accepts_final_done_without_newline() {
        let chunks: Vec<Result<Vec<u8>>> = vec![
            Ok(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n".to_vec()),
            Ok(b"data: [DONE]".to_vec()),
        ];
        let s = stream::iter(chunks);

        let (on_token, collected) = collect_tokens();
        let cancel = CancellationToken::new();
        let result = drive_sse_stream(
            s,
            on_token,
            Box::new(|_| {}),
            cancel,
            IdleTimeouts::uniform(Duration::from_secs(90)),
        )
        .await;

        match result.expect("final buffered [DONE] should complete") {
            LlmResponse::Text { text: t, .. } => assert_eq!(t, "hi"),
            other => panic!("expected text response, got {other:?}"),
        }
        assert_eq!(*collected.lock().unwrap(), vec!["hi"]);
    }

    /// A pre-cancelled token routes through the cancel arm of `select!`
    /// and returns an empty `Text`, never `ToolCalls` (which would be
    /// malformed if the stream cut mid-arguments). Sanity check that the
    /// cancel arm exists and produces text.
    #[tokio::test]
    async fn drive_sse_stream_cancellation_returns_text() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let s = stream::pending::<Result<Vec<u8>>>();

        let (on_token, _) = collect_tokens();
        let result = drive_sse_stream(
            s,
            on_token,
            Box::new(|_| {}),
            cancel,
            IdleTimeouts::uniform(Duration::from_secs(90)),
        )
        .await;

        match result.expect("should complete via cancel") {
            LlmResponse::Text { text: t, .. } => assert_eq!(t, ""),
            other => panic!("expected text response, got {other:?}"),
        }
    }

    /// `[DONE]` after tool-call deltas (no text) returns `ToolCalls`,
    /// preserving id/name and the concatenated arguments JSON. The SSE
    /// stream only delivers complete tool calls when `[DONE]` arrives;
    /// truncating mid-arguments would leave malformed JSON for the LLM.
    #[tokio::test]
    async fn drive_sse_stream_returns_tool_calls_on_done() {
        let chunks: Vec<Result<Vec<u8>>> = vec![
            Ok(
                b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"file_pa\"}}]}}]}\n"
                    .to_vec(),
            ),
            Ok(
                b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"th\\\":\\\"a.txt\\\"}\"}}]}}]}\n"
                    .to_vec(),
            ),
            Ok(b"data: [DONE]\n".to_vec()),
        ];
        let s = stream::iter(chunks);

        let (on_token, _) = collect_tokens();
        let cancel = CancellationToken::new();
        let result = drive_sse_stream(
            s,
            on_token,
            Box::new(|_| {}),
            cancel,
            IdleTimeouts::uniform(Duration::from_secs(90)),
        )
        .await;

        match result.expect("should complete") {
            LlmResponse::ToolCalls { text, calls, .. } => {
                assert!(text.is_empty());
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "call_1");
                assert_eq!(calls[0].function.name, "read_file");
                assert_eq!(calls[0].function.arguments, r#"{"file_path":"a.txt"}"#);
            }
            other => panic!("expected tool calls, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn drive_sse_stream_repairs_malformed_tool_call_arguments_on_done() {
        let chunks: Vec<Result<Vec<u8>>> = vec![
            Ok(
                b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{file_path:'a.txt',}\"}}]}}]}\n"
                    .to_vec(),
            ),
            Ok(b"data: [DONE]\n".to_vec()),
        ];
        let s = stream::iter(chunks);

        let (on_token, _) = collect_tokens();
        let result = drive_sse_stream(
            s,
            on_token,
            Box::new(|_| {}),
            CancellationToken::new(),
            IdleTimeouts::uniform(Duration::from_secs(90)),
        )
        .await;

        match result.expect("repairable tool-call arguments should complete") {
            LlmResponse::ToolCalls { calls, .. } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.arguments, r#"{"file_path":"a.txt"}"#);
            }
            other => panic!("expected tool calls, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn drive_sse_stream_treats_unterminated_tool_arguments_as_incomplete() {
        let chunks: Vec<Result<Vec<u8>>> = vec![
            Ok(
                b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"file_path\\\":\\\"unterminated\"}}]}}]}\n"
                    .to_vec(),
            ),
            Ok(b"data: [DONE]\n".to_vec()),
        ];
        let s = stream::iter(chunks);

        let (on_token, _) = collect_tokens();
        let err = drive_sse_stream(
            s,
            on_token,
            Box::new(|_| {}),
            CancellationToken::new(),
            IdleTimeouts::uniform(Duration::from_secs(90)),
        )
        .await
        .expect_err("unterminated streamed arguments should be retryable truncation");

        assert!(is_incomplete_stream_error(&err));
    }

    /// Unparseable SSE chunks are skipped (logged at debug). They must
    /// not abort the stream or count as progress -- the deadline keeps
    /// ticking, but valid downstream chunks still parse normally.
    #[tokio::test]
    async fn drive_sse_stream_skips_unparseable_data_chunks() {
        let chunks: Vec<Result<Vec<u8>>> = vec![
            Ok(b"data: {not-json}\n".to_vec()),
            Ok(b"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n".to_vec()),
            Ok(b"data: [DONE]\n".to_vec()),
        ];
        let s = stream::iter(chunks);

        let (on_token, collected) = collect_tokens();
        let cancel = CancellationToken::new();
        let result = drive_sse_stream(
            s,
            on_token,
            Box::new(|_| {}),
            cancel,
            IdleTimeouts::uniform(Duration::from_secs(90)),
        )
        .await;

        match result.expect("should complete") {
            LlmResponse::Text { text: t, .. } => assert_eq!(t, "ok"),
            other => panic!("expected text response, got {other:?}"),
        }
        assert_eq!(*collected.lock().unwrap(), vec!["ok"]);
    }

    /// The trailing usage chunk (empty `choices`, populated `usage`)
    /// emitted by OpenAI / OpenRouter when `stream_options.include_usage`
    /// is set must populate `LlmResponse.usage` with the ACP-aligned
    /// projection: `prompt_tokens - cached` → `input_tokens`,
    /// `completion_tokens - reasoning` → `output_tokens`, and so on.
    #[tokio::test]
    async fn drive_sse_stream_captures_trailing_usage_chunk() {
        let chunks: Vec<Result<Vec<u8>>> = vec![
            Ok(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n".to_vec()),
            Ok(b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":120,\"completion_tokens\":30,\"prompt_tokens_details\":{\"cached_tokens\":40},\"completion_tokens_details\":{\"reasoning_tokens\":10}}}\n".to_vec()),
            Ok(b"data: [DONE]\n".to_vec()),
        ];
        let s = stream::iter(chunks);

        let (on_token, _) = collect_tokens();
        let cancel = CancellationToken::new();
        let result = drive_sse_stream(
            s,
            on_token,
            Box::new(|_| {}),
            cancel,
            IdleTimeouts::uniform(Duration::from_secs(90)),
        )
        .await;

        match result.expect("should complete") {
            LlmResponse::Text { text, usage, .. } => {
                assert_eq!(text, "hi");
                assert_eq!(usage.input_tokens, 80); // 120 - 40 cached
                assert_eq!(usage.output_tokens, 20); // 30 - 10 reasoning
                assert_eq!(usage.thought_tokens, 10);
                assert_eq!(usage.cached_read_tokens, 40);
                assert_eq!(usage.cached_write_tokens, 0);
                assert_eq!(usage.total_tokens(), 150);
            }
            other => panic!("expected text response, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn drive_sse_stream_preserves_reasoning_content() {
        let chunks: Vec<Result<Vec<u8>>> = vec![
            Ok(b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think \"}}]}\n".to_vec()),
            Ok(b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hard\",\"content\":\"ok\"}}]}\n".to_vec()),
            Ok(b"data: [DONE]\n".to_vec()),
        ];
        let s = stream::iter(chunks);

        let (on_token, tokens) = collect_tokens();
        let thoughts = Arc::new(Mutex::new(Vec::<String>::new()));
        let thoughts_for_cb = Arc::clone(&thoughts);
        let on_thought: TokenSink = Box::new(move |s: &str| {
            thoughts_for_cb.lock().unwrap().push(s.to_string());
        });
        let cancel = CancellationToken::new();
        let result = drive_sse_stream(
            s,
            on_token,
            on_thought,
            cancel,
            IdleTimeouts::uniform(Duration::from_secs(90)),
        )
        .await
        .expect("should complete");

        match result {
            LlmResponse::Text {
                text,
                reasoning_content,
                ..
            } => {
                assert_eq!(text, "ok");
                assert_eq!(reasoning_content.as_deref(), Some("think hard"));
            }
            other => panic!("expected text response, got {other:?}"),
        }
        assert_eq!(*tokens.lock().unwrap(), vec!["ok"]);
        assert_eq!(*thoughts.lock().unwrap(), vec!["think ", "hard"]);
    }

    #[test]
    fn chat_completion_chunk_accepts_vllm_reasoning_delta() {
        let chunk: ChatCompletionChunk =
            serde_json::from_str(r#"{"choices":[{"delta":{"reasoning":"thinking"}}]}"#).unwrap();
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(
            chunk.choices[0].delta.reasoning_content.as_deref(),
            Some("thinking")
        );

        let chunk: ChatCompletionChunk = serde_json::from_str(
            r#"{"choices":[{"delta":{"reasoning_content":"deepseek","reasoning":"vllm"}}]}"#,
        )
        .unwrap();
        assert_eq!(
            chunk.choices[0].delta.reasoning_content.as_deref(),
            Some("deepseek")
        );
    }

    /// Stream EOF without `[DONE]` is incomplete, even if the server
    /// already emitted text. The caller may only retry this safely when
    /// no user-visible output escaped.
    #[tokio::test]
    async fn drive_sse_stream_errors_on_eof_without_done() {
        let chunks: Vec<Result<Vec<u8>>> = vec![Ok(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n".to_vec(),
        )];
        let s = stream::iter(chunks);

        let (on_token, collected) = collect_tokens();
        let cancel = CancellationToken::new();
        let err = drive_sse_stream(
            s,
            on_token,
            Box::new(|_| {}),
            cancel,
            IdleTimeouts::uniform(Duration::from_secs(90)),
        )
        .await
        .expect_err("EOF before [DONE] must be an incomplete stream");

        assert!(is_incomplete_stream_error(&err));
        assert_eq!(*collected.lock().unwrap(), vec!["partial"]);
    }

    #[tokio::test]
    async fn drive_sse_stream_errors_on_empty_eof_without_done() {
        let s = stream::iter(Vec::<Result<Vec<u8>>>::new());

        let (on_token, collected) = collect_tokens();
        let cancel = CancellationToken::new();
        let err = drive_sse_stream(
            s,
            on_token,
            Box::new(|_| {}),
            cancel,
            IdleTimeouts::uniform(Duration::from_secs(90)),
        )
        .await
        .expect_err("empty EOF before [DONE] must be incomplete");

        assert!(is_incomplete_stream_error(&err));
        assert!(collected.lock().unwrap().is_empty());
    }

    /// The base URL is normalized to drop a trailing slash, and `/v1` is
    /// appended only when not already present. This is what lets the
    /// CLI default `http://localhost:11434/v1` and a bare endpoint URL
    /// like `https://api.example.com` both work.
    #[test]
    fn api_url_appends_v1_when_missing() {
        let client = OpenAiClient::new("http://localhost:11434".into(), None);
        assert_eq!(
            client.api_url("/chat/completions"),
            "http://localhost:11434/v1/chat/completions"
        );
        assert_eq!(
            client.api_url("/models"),
            "http://localhost:11434/v1/models"
        );
    }

    #[test]
    fn api_url_does_not_double_v1() {
        let client = OpenAiClient::new("http://localhost:11434/v1".into(), None);
        assert_eq!(
            client.api_url("/chat/completions"),
            "http://localhost:11434/v1/chat/completions"
        );
    }

    /// Trailing slash is trimmed at construction, so the final URL never
    /// has a `//` between the base and the path.
    #[test]
    fn api_url_strips_trailing_slash() {
        let client = OpenAiClient::new("http://localhost:11434/".into(), None);
        assert_eq!(
            client.api_url("/models"),
            "http://localhost:11434/v1/models"
        );

        let client = OpenAiClient::new("http://localhost:11434/v1/".into(), None);
        assert_eq!(
            client.api_url("/models"),
            "http://localhost:11434/v1/models"
        );
    }

    /// HTTPS endpoints (typical for hosted OpenAI-compatible providers)
    /// receive the same normalization as plain HTTP.
    #[test]
    fn api_url_handles_https_endpoint() {
        let client = OpenAiClient::new("https://api.example.com".into(), None);
        assert_eq!(
            client.api_url("/chat/completions"),
            "https://api.example.com/v1/chat/completions"
        );
    }

    /// `Debug` for `OpenAiClient` must redact the api key so it does not
    /// leak into log output via `{:?}`.
    #[test]
    fn debug_redacts_api_key() {
        let client = OpenAiClient::new("http://x".into(), Some("sk-secret-123".into()));
        let dbg = format!("{client:?}");
        assert!(!dbg.contains("sk-secret-123"));
        assert!(dbg.contains("REDACTED"));
    }

    /// `ChatMessage` constructors set the role correctly and serialize
    /// only the fields they own. The `#[serde(skip_serializing_if = "Option::is_none")]`
    /// attributes are load-bearing for endpoints that reject extra keys.
    #[test]
    fn chat_message_constructors_round_trip_through_json() {
        let m = ChatMessage::system("you are helpful");
        let v: serde_json::Value = serde_json::to_value(&m).unwrap();
        assert_eq!(v["role"], "system");
        assert_eq!(
            v["content"],
            serde_json::json!([{ "type": "text", "text": "you are helpful" }])
        );
        assert!(v.get("tool_calls").is_none(), "system has no tool_calls");
        assert!(v.get("tool_call_id").is_none());

        let m = ChatMessage::user("hi");
        assert_eq!(serde_json::to_value(&m).unwrap()["role"], "user");

        let m = ChatMessage::assistant("ok");
        assert_eq!(serde_json::to_value(&m).unwrap()["role"], "assistant");

        let m = ChatMessage::tool_result("call_1", "read_file", "contents");
        let v: serde_json::Value = serde_json::to_value(&m).unwrap();
        assert_eq!(v["role"], "tool");
        assert_eq!(v["tool_call_id"], "call_1");
        assert_eq!(v["name"], "read_file");
        assert_eq!(
            v["content"],
            serde_json::json!([{ "type": "text", "text": "contents" }])
        );
    }

    #[test]
    fn chat_message_reasoning_serializes_with_deepseek_and_vllm_keys() {
        let m = ChatMessage::assistant_with_reasoning("answer", Some("thinking".into()));
        let v: serde_json::Value = serde_json::to_value(&m).unwrap();
        assert_eq!(v["reasoning_content"], "thinking");
        assert_eq!(v["reasoning"], "thinking");

        let body = ChatCompletionRequest {
            model: "openai-compatible-model".into(),
            messages: vec![m],
            stream: true,
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
            temperature: None,
            max_tokens: None,
            tools: None,
            tool_choice: None,
            reasoning: None,
            thinking: None,
            reasoning_effort: None,
            response_format: None,
        };
        let body: serde_json::Value = serde_json::to_value(&body).unwrap();
        let message = &body["messages"][0];
        assert_eq!(message["reasoning_content"], "thinking");
        assert_eq!(message["reasoning"], "thinking");
    }

    /// `assistant_tool_calls` omits `content` entirely (not `null`),
    /// because some providers reject `content: null` alongside
    /// `tool_calls`.
    #[test]
    fn assistant_tool_calls_omits_content_field() {
        let calls = vec![ToolCall {
            id: "id_0".into(),
            r#type: "function".into(),
            function: FunctionCall {
                name: "read_file".into(),
                arguments: r#"{"file_path":"x"}"#.into(),
            },
        }];
        let m = ChatMessage::assistant_tool_calls(calls);
        let v: serde_json::Value = serde_json::to_value(&m).unwrap();
        assert_eq!(v["role"], "assistant");
        assert!(
            v.get("content").is_none(),
            "content key must be skipped when None, got: {v}"
        );
        assert!(v.get("tool_calls").is_some());
    }

    /// `ToolCallAccumulator` merges fragments by index. The OpenAI API
    /// streams tool-call arguments across many SSE chunks; we must
    /// concatenate them in order without duplicating the id or name.
    #[test]
    fn tool_call_accumulator_concatenates_fragments_per_index() {
        let mut acc = ToolCallAccumulator::default();
        // Index 0: id arrives first, then name, then arguments split into two.
        acc.push(&ChunkToolCall {
            index: 0,
            id: Some("call_0".into()),
            function: None,
        });
        acc.push(&ChunkToolCall {
            index: 0,
            id: None,
            function: Some(ChunkFunctionCall {
                name: Some("read_file".into()),
                arguments: Some(r#"{"file_pa"#.into()),
            }),
        });
        acc.push(&ChunkToolCall {
            index: 0,
            id: None,
            function: Some(ChunkFunctionCall {
                name: None,
                arguments: Some(r#"th":"x.txt"}"#.into()),
            }),
        });
        // Index 1 in parallel; sort_by_key in into_tool_calls puts it second.
        acc.push(&ChunkToolCall {
            index: 1,
            id: Some("call_1".into()),
            function: Some(ChunkFunctionCall {
                name: Some("write_file".into()),
                arguments: Some(r#"{"file_path":"y.txt","content":""}"#.into()),
            }),
        });

        let calls = acc.into_tool_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[0].function.arguments, r#"{"file_path":"x.txt"}"#);
        assert_eq!(calls[1].id, "call_1");
        assert_eq!(calls[1].function.name, "write_file");
    }

    /// `OpenAiClient::with_default_headers` produces an instance that
    /// `LlmBackend` can be cast over (no panic on construction, headers
    /// accepted as-is). Wire path is exercised by the integration with
    /// `MultiBackend` -- this test pins the constructor contract that
    /// OpenRouter's `HTTP-Referer` / `X-Title` attribution headers rely
    /// on (`main.rs::build_openrouter_backend`).
    #[test]
    fn with_default_headers_constructs_with_attribution_headers() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::HeaderName::from_static("http-referer"),
            reqwest::header::HeaderValue::from_static("https://example.test"),
        );
        headers.insert(
            reqwest::header::HeaderName::from_static("x-title"),
            reqwest::header::HeaderValue::from_static("brokk-acp-rust"),
        );
        let client = OpenAiClient::with_default_headers(
            "https://openrouter.ai/api/v1".to_string(),
            Some("sk-test-key".to_string()),
            headers,
        );
        // Trailing slashes are stripped by both constructors so callers
        // can interchange `.../v1` and `.../v1/` without double-slashes
        // showing up in the request URL.
        let debug = format!("{client:?}");
        assert!(debug.contains("openrouter.ai"), "got {debug}");
        assert!(
            debug.contains("[REDACTED]"),
            "api_key must be redacted from Debug output: {debug}"
        );
    }

    #[test]
    fn chat_completion_request_serializes_response_format_when_present() {
        let format = crate::structured_output::native_response_format(&StructuredOutputRequest {
            schema_name: "audit_result".into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"]
            }),
            allow_coercion: false,
            prefer_json_object: false,
        });
        let body = ChatCompletionRequest {
            model: "gpt-4.1".into(),
            messages: vec![ChatMessage::user("hi")],
            stream: true,
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
            temperature: None,
            max_tokens: None,
            tools: None,
            tool_choice: None,
            reasoning: None,
            thinking: None,
            reasoning_effort: None,
            response_format: Some(ChatCompletionResponseFormat::JsonSchema {
                json_schema: NativeJsonSchemaFormat {
                    name: format.name,
                    schema: format.schema,
                    strict: format.strict,
                },
            }),
        };
        let serialized = serde_json::to_value(&body).unwrap();
        assert_eq!(serialized["response_format"]["type"], "json_schema");
        assert_eq!(
            serialized["response_format"]["json_schema"]["name"],
            "audit_result"
        );
        assert_eq!(
            serialized["response_format"]["json_schema"]["schema"]["type"],
            "object"
        );
        assert_eq!(serialized["response_format"]["json_schema"]["strict"], true);
    }

    /// OpenRouter's `/v1/models` response carries strictly more fields
    /// than OpenAI's (`name`, `canonical_slug`, `pricing`, `architecture`,
    /// etc.). The shared `ModelsResponse` deserializer must round-trip
    /// the catalog without choking on the extra fields, leaving the
    /// caller with the model ids and any reasoning metadata the routing
    /// layer can use.
    /// Sample shape distilled from a live `GET https://openrouter.ai/api/v1/models`
    /// response (vendor/model ids, with nested `pricing` and
    /// `architecture` objects) so a future serde-rename regression is
    /// caught here rather than at runtime on the user's first session.
    #[test]
    fn models_response_parses_openrouter_shape_ignoring_extra_fields() {
        let raw = r#"{
            "data": [
                {
                    "id": "anthropic/claude-3.5-sonnet",
                    "name": "Anthropic: Claude 3.5 Sonnet",
                    "canonical_slug": "anthropic/claude-3.5-sonnet",
                    "context_length": 200000,
                    "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                    "architecture": {"input_modalities": ["text", "image"]},
                    "top_provider": {"context_length": 200000}
                },
                {
                    "id": "openai/gpt-4o",
                    "name": "OpenAI: GPT-4o",
                    "context_length": 128000,
                    "pricing": {"prompt": "0.0000025", "completion": "0.00001"}
                }
            ]
        }"#;
        let parsed: ModelsResponse = serde_json::from_str(raw).expect("OpenRouter /models parses");
        let entries: Vec<(String, Option<u32>)> = parsed
            .data
            .iter()
            .map(|m| (m.id.clone(), m.context_length))
            .collect();
        assert_eq!(
            entries,
            vec![
                ("anthropic/claude-3.5-sonnet".to_string(), Some(200_000)),
                ("openai/gpt-4o".to_string(), Some(128_000)),
            ]
        );
        // `to_model_metadata` must surface context_length for both
        // reasoning-capable and plain entries (compression layer needs
        // it for the threshold regardless of reasoning support).
        let plain = parsed.data[1].to_model_metadata();
        assert_eq!(plain.context_length, Some(128_000));
        assert_eq!(
            parsed.data[0].to_model_metadata().supports_images,
            Some(true)
        );
        assert_eq!(plain.supports_images, None);
    }

    /// Missing `context_length` (some OpenRouter providers omit it, and
    /// every Codex/Ollama entry will) must deserialize cleanly as
    /// `None` rather than failing the whole catalog fetch.
    #[test]
    fn model_entry_without_context_length_is_none() {
        let raw = r#"{ "id": "local/model" }"#;
        let entry: ModelEntry = serde_json::from_str(raw).expect("entry parses");
        assert!(entry.context_length.is_none());
        assert!(entry.to_model_metadata().context_length.is_none());
        assert!(entry.to_model_metadata().supports_images.is_none());
    }

    #[test]
    fn openrouter_text_only_model_surfaces_supports_images_false() {
        let raw = r#"{
            "id": "openai/gpt-4.1-mini",
            "architecture": {"input_modalities": ["text"]}
        }"#;
        let entry: ModelEntry = serde_json::from_str(raw).expect("entry parses");
        assert_eq!(entry.to_model_metadata().supports_images, Some(false));
    }

    #[test]
    fn image_prompt_provider_errors_are_rewritten() {
        let err = "chat completion failed (HTTP 400): model is text-only and does not support image input";
        assert!(rewrite_image_prompt_provider_error(err).is_some());
        assert!(rewrite_image_prompt_provider_error("timeout waiting for stream").is_none());
    }

    /// OpenRouter reasoning-capable models should surface a default
    /// reasoning effort and the picker presets so the session UI can
    /// offer them instead of silently collapsing to "no reasoning."
    #[test]
    fn openrouter_reasoning_metadata_is_preserved_when_supported() {
        let raw = r#"{
            "data": [
                {
                    "id": "openai/gpt-5.2",
                    "supported_parameters": ["temperature", "reasoning"],
                    "default_parameters": {
                        "reasoning": {
                            "enabled": true,
                            "effort": "high"
                        }
                    }
                },
                {
                    "id": "openai/gpt-4o",
                    "supported_parameters": ["temperature", "top_p"]
                }
            ]
        }"#;
        let parsed: ModelsResponse = serde_json::from_str(raw).expect("OpenRouter /models parses");
        let reasoning_model = parsed.data[0].to_model_metadata();
        assert_eq!(reasoning_model.id, "openai/gpt-5.2");
        assert_eq!(
            reasoning_model.default_reasoning_level.as_deref(),
            Some("high")
        );
        assert_eq!(
            reasoning_model
                .supported_reasoning_levels
                .iter()
                .map(|preset| preset.effort.as_str())
                .collect::<Vec<_>>(),
            vec!["minimal", "low", "medium", "high", "xhigh"]
        );

        let plain_model = parsed.data[1].to_model_metadata();
        assert_eq!(plain_model.id, "openai/gpt-4o");
        assert!(plain_model.default_reasoning_level.is_none());
        assert!(plain_model.supported_reasoning_levels.is_empty());
    }

    #[test]
    fn openrouter_pricing_is_preserved_and_estimates_cost() {
        let raw = r#"{
            "data": [
                {
                    "id": "anthropic/claude-3.5-sonnet",
                    "pricing": {"prompt": "0.000003", "completion": "0.000015"}
                }
            ]
        }"#;
        let parsed: ModelsResponse = serde_json::from_str(raw).expect("OpenRouter /models parses");
        let model = parsed.data[0].to_model_metadata();
        let cost = model.estimate_cost_usd(TokenUsage {
            input_tokens: 10,
            output_tokens: 4,
            thought_tokens: 1,
            cached_read_tokens: 2,
            cached_write_tokens: 0,
        });
        assert_eq!(model.id, "anthropic/claude-3.5-sonnet");
        let cost = cost.expect("pricing must produce a cost");
        assert!((cost - 0.000111).abs() < 1e-12, "got {cost}");
    }
}
