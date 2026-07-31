use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::http_retry::RetryableLlmError;
use crate::llm_client::{
    ChatContentPart, ChatMessage, FunctionCall, IdleTimeouts, IncompleteStreamError, LlmBackend,
    LlmResponse, ModelMetadata, ModelsResponse, OpenAiClient, OutputBudgetExhaustedError,
    ReasoningLevelPreset, StreamChatRequest, TokenSink, TokenUsage, ToolCall, ToolDefinition,
};
use crate::responses_api::{build_responses_request, drive_responses_sse_stream};
use crate::trace_logging::append_trace_record;
use anyhow::{Context, Result};
use futures::StreamExt;
use futures::future::BoxFuture;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// Returns true when the model supports Anthropic-style prompt caching
/// and needs explicit `cache_control` breakpoints in the request body.
/// All Bedrock-hosted Claude models have "anthropic" in their id
/// (e.g. `us.anthropic.claude-sonnet-4-6`).
fn requires_explicit_caching(model: &str) -> bool {
    model.contains("anthropic")
}

const OPENAI_GPT_REASONING_PRESETS: &[(&str, &str)] = &[
    ("none", "No reasoning effort."),
    ("low", "Light reasoning for simpler problems."),
    ("medium", "Balanced reasoning for moderate complexity."),
    ("high", "Deep reasoning for complex problems."),
    ("xhigh", "Extra-high reasoning for the hardest problems."),
];

const OPENAI_GPT_OSS_REASONING_PRESETS: &[(&str, &str)] = &[
    ("low", "Light reasoning for low-latency responses."),
    ("medium", "Balanced reasoning for general use."),
    ("high", "Deep reasoning for harder tasks."),
];

const ANTHROPIC_BASIC_REASONING_PRESETS: &[(&str, &str)] = &[
    ("low", "Most efficient; significant token savings."),
    ("medium", "Balanced speed, cost, and performance."),
    ("high", "High capability; provider default."),
];

const ANTHROPIC_MAX_REASONING_PRESETS: &[(&str, &str)] = &[
    ("low", "Most efficient; significant token savings."),
    ("medium", "Balanced speed, cost, and performance."),
    ("high", "High capability; provider default."),
    (
        "max",
        "Absolute maximum capability with no token-spend constraint.",
    ),
];

const ANTHROPIC_XHIGH_REASONING_PRESETS: &[(&str, &str)] = &[
    ("low", "Most efficient; significant token savings."),
    ("medium", "Balanced speed, cost, and performance."),
    ("high", "High capability; provider default."),
    ("xhigh", "Extended capability for long-horizon work."),
    (
        "max",
        "Absolute maximum capability with no token-spend constraint.",
    ),
];

const ANTHROPIC_MANUAL_THINKING_PRESETS: &[(&str, &str)] = &[
    ("low", "Manual extended thinking with a 2K token budget."),
    ("medium", "Manual extended thinking with a 4K token budget."),
    ("high", "Manual extended thinking with an 8K token budget."),
];

const BEDROCK_LEGACY_REASONING_PRESETS: &[(&str, &str)] = &[
    ("low", "Light reasoning for shorter problems."),
    ("medium", "Balanced reasoning for moderate complexity."),
    ("high", "Deep reasoning for harder problems."),
];

#[derive(Debug, Clone, Copy)]
struct BedrockReasoningSpec {
    default_level: Option<&'static str>,
    presets: &'static [(&'static str, &'static str)],
    anthropic_thinking_shape: Option<ThinkingShape>,
    send_output_effort: bool,
}

impl BedrockReasoningSpec {
    fn supports_reasoning_level(self, effort: &str) -> bool {
        self.presets
            .iter()
            .any(|(level, _)| *level == effort.trim())
    }

    fn to_presets(self) -> Vec<ReasoningLevelPreset> {
        self.presets
            .iter()
            .map(|(effort, description)| ReasoningLevelPreset {
                effort: (*effort).to_string(),
                description: (*description).to_string(),
            })
            .collect()
    }
}

fn is_prefixed_bedrock_model(model: &str, bases: &[&str]) -> bool {
    bases.iter().any(|base| match model.strip_prefix(base) {
        Some(rest) => rest.is_empty() || rest.starts_with('-'),
        None => false,
    })
}

fn bedrock_reasoning_spec_for_model(model: &str) -> Option<BedrockReasoningSpec> {
    let model = model.trim().to_ascii_lowercase();
    if is_prefixed_bedrock_model(&model, &["openai.gpt-oss-120b", "openai.gpt-oss-20b"]) {
        return Some(BedrockReasoningSpec {
            default_level: Some("medium"),
            presets: OPENAI_GPT_OSS_REASONING_PRESETS,
            anthropic_thinking_shape: None,
            send_output_effort: false,
        });
    }
    if is_prefixed_bedrock_model(
        &model,
        &["openai.gpt-5.6", "openai.gpt-5.5", "openai.gpt-5.4"],
    ) {
        return Some(BedrockReasoningSpec {
            default_level: Some("medium"),
            presets: OPENAI_GPT_REASONING_PRESETS,
            anthropic_thinking_shape: None,
            send_output_effort: false,
        });
    }

    let anthropic = |needle: &str| model.contains(needle);
    if anthropic("claude-fable-5")
        || anthropic("claude-mythos-5")
        || anthropic("claude-opus-5")
        || anthropic("claude-opus-4-8")
        || anthropic("claude-opus-4-7")
        || anthropic("claude-sonnet-5")
    {
        return Some(BedrockReasoningSpec {
            default_level: Some("high"),
            presets: ANTHROPIC_XHIGH_REASONING_PRESETS,
            anthropic_thinking_shape: Some(ThinkingShape::Adaptive),
            send_output_effort: true,
        });
    }
    if anthropic("claude-mythos-preview")
        || anthropic("claude-opus-4-6")
        || anthropic("claude-sonnet-4-6")
    {
        return Some(BedrockReasoningSpec {
            default_level: Some("high"),
            presets: ANTHROPIC_MAX_REASONING_PRESETS,
            anthropic_thinking_shape: Some(ThinkingShape::Adaptive),
            send_output_effort: true,
        });
    }
    if anthropic("claude-opus-4-5") {
        return Some(BedrockReasoningSpec {
            default_level: Some("high"),
            presets: ANTHROPIC_BASIC_REASONING_PRESETS,
            anthropic_thinking_shape: Some(ThinkingShape::Enabled),
            send_output_effort: true,
        });
    }
    if anthropic("claude-haiku-4-5")
        || anthropic("claude-sonnet-4-5")
        || anthropic("claude-opus-4-1")
        || anthropic("claude-sonnet-4-2025")
        || model.ends_with("claude-sonnet-4")
    {
        return Some(BedrockReasoningSpec {
            default_level: None,
            presets: ANTHROPIC_MANUAL_THINKING_PRESETS,
            anthropic_thinking_shape: Some(ThinkingShape::Enabled),
            send_output_effort: false,
        });
    }
    None
}

fn fallback_bedrock_reasoning_spec(model: &str) -> BedrockReasoningSpec {
    BedrockReasoningSpec {
        default_level: Some("medium"),
        presets: BEDROCK_LEGACY_REASONING_PRESETS,
        anthropic_thinking_shape: (!uses_responses_api(model)).then_some(ThinkingShape::Enabled),
        send_output_effort: false,
    }
}

fn bedrock_reasoning_spec_or_fallback(model: &str) -> BedrockReasoningSpec {
    bedrock_reasoning_spec_for_model(model)
        .unwrap_or_else(|| fallback_bedrock_reasoning_spec(model))
}

/// Map a reasoning-effort preset to an Anthropic extended-thinking
/// `budget_tokens` value. Returns `None` for unknown/empty efforts so the
/// caller omits the `thinking` block entirely.
///
/// Budgets must be >= 1024 and strictly less than the request's
/// `max_tokens`; `thinking_max_tokens` keeps that invariant.
fn thinking_budget_for_effort(effort: &str) -> Option<u32> {
    match effort.trim() {
        "low" => Some(2_048),
        "medium" => Some(4_096),
        "high" => Some(8_192),
        _ => None,
    }
}

/// Output cap to send alongside a `thinking` budget. Anthropic requires
/// `max_tokens > budget_tokens`; we add headroom for the visible answer on
/// top of the reasoning budget.
fn thinking_max_tokens(budget_tokens: u32) -> u32 {
    budget_tokens.saturating_add(MAX_TOKENS)
}

/// Build a native Anthropic request, attaching the reasoning controls for the
/// given wire `shape` when a validated reasoning level is present.
#[derive(Debug, Clone, Copy)]
struct AnthropicReasoningControls<'a> {
    effort: Option<&'a str>,
    shape: ThinkingShape,
    send_output_effort: bool,
}

fn build_anthropic_request(
    anthropic_version: &'static str,
    system: Option<Vec<BedrockTextBlock>>,
    messages: Vec<BedrockMessage>,
    tools: Option<Vec<BedrockTool>>,
    temperature: Option<f64>,
    reasoning: AnthropicReasoningControls<'_>,
) -> BedrockAnthropicRequest {
    let mut request = BedrockAnthropicRequest {
        anthropic_version,
        system,
        messages,
        tools,
        max_tokens: MAX_TOKENS,
        temperature,
        thinking: None,
        output_config: None,
    };
    let Some(effort) = reasoning.effort else {
        return request;
    };
    match reasoning.shape {
        ThinkingShape::Enabled => {
            if let Some(budget_tokens) = thinking_budget_for_effort(effort) {
                request.max_tokens = thinking_max_tokens(budget_tokens);
                request.temperature = None;
                request.thinking = Some(BedrockThinking::Enabled { budget_tokens });
                if reasoning.send_output_effort {
                    request.output_config = Some(BedrockOutputConfig {
                        effort: effort.to_string(),
                    });
                }
            }
        }
        ThinkingShape::Adaptive => {
            // Adaptive thinking has no budget_tokens to derive a ceiling
            // from -- the model decides how much to think, and the legacy
            // budget arithmetic does not apply (budget_tokens is rejected
            // outright by the models that use this shape). With only the
            // flat MAX_TOKENS ceiling, the model can spend the entire
            // budget thinking and return thinking-only content with no
            // text or tool_use, which reads as a degenerate empty
            // completion and gets retried. Confirmed by replay: at 8192,
            // stop_reason=max_tokens with 8192/8192 thinking tokens and
            // zero visible output.
            //
            // Anthropic's guidance for high-and-above effort with adaptive
            // thinking is a ceiling of at least ADAPTIVE_MAX_TOKENS, well
            // above what budget-derived arithmetic would produce.
            request.max_tokens = ADAPTIVE_MAX_TOKENS;
            request.temperature = None;
            request.thinking = Some(BedrockThinking::Adaptive {
                display: "summarized",
            });
            request.output_config = Some(BedrockOutputConfig {
                effort: effort.to_string(),
            });
        }
    }
    request
}

/// True when a failed native invoke is the documented "this model does not
/// support `thinking.type.enabled`; use `thinking.type.adaptive` and
/// `output_config.effort`" rejection. Detection is on the API's own error
/// contract (authoritative) rather than a model-id version allowlist.
fn error_requires_adaptive_thinking(err: &anyhow::Error) -> bool {
    let msg = err.to_string();
    msg.contains("thinking.type.enabled")
        && (msg.contains("adaptive") || msg.contains("output_config"))
}

/// Attach Bedrock/provider-specific reasoning presets in the merged catalog.
fn apply_bedrock_reasoning_presets(models: &mut [ModelMetadata]) {
    for model in models.iter_mut() {
        let spec = bedrock_reasoning_spec_or_fallback(&model.id);
        model.supported_reasoning_levels = spec.to_presets();
        model.default_reasoning_level = spec.default_level.map(str::to_string);
    }
}

const CACHE_CONTROL: CacheControl = CacheControl {
    r#type: "ephemeral",
    // 1-hour TTL (GA on Bedrock, no beta header). Agent sessions reuse the
    // prompt prefix across long inter-turn gaps — Thor pauses while Eitri runs
    // sliced code_agent and Loki reviews interleave — which routinely exceeds
    // the 5-minute default and evicts the prefix before the next turn reads it.
    ttl: Some("1h"),
};

fn trace_bedrock_request(body: &BedrockAnthropicRequest) {
    let Ok(path) = std::env::var("ANVIL_TRACE_JSONL") else {
        return;
    };
    let path = path.trim();
    if path.is_empty() {
        return;
    }
    let Ok(body_json) = serde_json::to_value(body) else {
        tracing::warn!("failed to serialize Bedrock trace request body");
        return;
    };
    let system_cache = body_json
        .get("system")
        .and_then(|s| s.as_array())
        .and_then(|blocks| blocks.last())
        .and_then(|b| b.get("cache_control"))
        .is_some();
    let tool_cache = body_json
        .get("tools")
        .and_then(|t| t.as_array())
        .and_then(|tools| tools.last())
        .and_then(|t| t.get("cache_control"))
        .is_some();
    let user_cache = body_json
        .get("messages")
        .and_then(|m| m.as_array())
        .and_then(|msgs| {
            msgs.iter()
                .rev()
                .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        })
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_array())
        .and_then(|blocks| blocks.last())
        .and_then(|b| b.get("cache_control"))
        .is_some();
    let record = serde_json::json!({
        "type": "bedrock_request_body",
        "enable_cache": system_cache || tool_cache || user_cache,
        "system_cache_control": system_cache,
        "tool_cache_control": tool_cache,
        "user_cache_control": user_cache,
        "system_blocks": body_json.get("system").and_then(|s| s.as_array()).map(|a| a.len()).unwrap_or(0),
        "message_count": body_json.get("messages").and_then(|m| m.as_array()).map(|a| a.len()).unwrap_or(0),
        "tool_count": body_json.get("tools").and_then(|t| t.as_array()).map(|a| a.len()).unwrap_or(0),
        "full_body": body_json,
    });
    append_trace_record(record);
}

pub const BEDROCK_API_KEY_ENV: &str = "AWS_BEARER_TOKEN_BEDROCK";
pub const BEDROCK_REGION_ENV: &str = "BEDROCK_REGION";
pub const BEDROCK_MODEL_ENV: &str = "ANVIL_BEDROCK_MODEL";
pub const BEDROCK_DEFAULT_REGION: &str = "us-east-1";
pub const BEDROCK_DEFAULT_MODEL: &str = "us.anthropic.claude-sonnet-4-6";
const BEDROCK_RUNTIME_BASE_URL: &str = "https://bedrock-runtime";
const BEDROCK_CONTROL_BASE_URL: &str = "https://bedrock";

const ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";
const MAX_TOKENS: u32 = 8192;

/// Output ceiling for adaptive-thinking requests. Adaptive thinking spends
/// from the same budget as the visible answer, so the flat `MAX_TOKENS` is
/// far too tight: the model exhausts it thinking and returns nothing else.
/// Anthropic's guidance is at least this much at `high` effort and above.
const ADAPTIVE_MAX_TOKENS: u32 = 64_000;

/// Wall-clock budget for one native Anthropic invoke. This path does not
/// stream -- the caller blocks until the entire response body is ready --
/// so the ceiling above translates directly into request duration, and the
/// client-wide 600s timeout cuts long generations off mid-flight. Observed
/// as `operation timed out` with workers dying at clean multiples of that
/// timeout. Sized to let an `ADAPTIVE_MAX_TOKENS` generation finish.
const NATIVE_INVOKE_TIMEOUT: Duration = Duration::from_secs(1_800);

/// Idle floor for native Anthropic streaming. Adaptive thinking on a real
/// supervisor request -- large system prompt, full tool schemas, accumulated
/// history -- goes quiet for minutes at a time while the model reasons, and
/// nothing is emitted during that window even with `display: "summarized"`.
/// The general-purpose 60s stall timeout reads that as a hung stream and
/// aborts a turn that was progressing normally, which is how a whole sweep
/// came back with empty patches. Keep a backstop, but well above the gaps
/// this workload actually produces, and inside NATIVE_INVOKE_TIMEOUT so a
/// genuinely wedged stream is still bounded.
const NATIVE_STREAM_IDLE_FLOOR: Duration = Duration::from_secs(600);

/// Raise `idle` to the native-streaming floor, leaving anything already
/// more generous alone.
fn native_stream_idle_timeouts(idle: IdleTimeouts) -> IdleTimeouts {
    IdleTimeouts {
        first_progress: idle.first_progress.max(NATIVE_STREAM_IDLE_FLOOR),
        inter_chunk: idle.inter_chunk.max(NATIVE_STREAM_IDLE_FLOOR),
    }
}
const BEDROCK_DISCOVERY_MAX_ATTEMPTS: u64 = 4;
const BEDROCK_DISCOVERY_RETRY_BASE_DELAY: Duration = Duration::from_millis(500);

#[derive(Clone)]
pub struct BedrockClient {
    bearer_token: String,
    region: String,
    default_model: String,
    http: reqwest::Client,
    runtime_base_url: String,
    mantle_base_url: String,
    control_base_url: String,
    catalog_mode: crate::setup_state::BedrockCatalogMode,
    /// Per-resolved-model record of which Anthropic thinking wire shape the
    /// model actually accepts. Most shapes are selected from the documented
    /// model family, but this cache preserves the provider-directed adaptive
    /// fallback if a manual-thinking model rejects `enabled`.
    thinking_shape_cache: Arc<RwLock<HashMap<String, ThinkingShape>>>,
    /// Content-keyed cache mapping a hash of a Responses API message context
    /// to the `response_id` it produced, so the next turn on the same
    /// conversation can chain onto it via `previous_response_id` and send
    /// only the delta instead of resending the whole conversation. See
    /// `invoke_responses_model`. Only touched by the Responses API (OpenAI
    /// Bedrock Mantle) path; irrelevant to the native Anthropic/converse
    /// path.
    responses_chain: Arc<std::sync::Mutex<ResponsesChainCache>>,
    discovery_cache: Arc<BedrockDiscoveryCache>,
}

struct BedrockDiscoveryCache {
    mantle_models: LastGoodDiscovery<Vec<ModelMetadata>>,
    foundation_models: LastGoodDiscovery<Vec<ModelMetadata>>,
    inference_profiles: LastGoodDiscovery<Vec<BedrockInferenceProfileSummary>>,
}

impl BedrockDiscoveryCache {
    fn new() -> Self {
        Self {
            mantle_models: LastGoodDiscovery::new("Bedrock Mantle model discovery"),
            foundation_models: LastGoodDiscovery::new("Bedrock foundation-model discovery"),
            inference_profiles: LastGoodDiscovery::new("Bedrock inference profile discovery"),
        }
    }
}

struct LastGoodDiscovery<T> {
    name: &'static str,
    value: std::sync::Mutex<Option<LastGoodValue<T>>>,
}

impl<T> LastGoodDiscovery<T> {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            value: std::sync::Mutex::new(None),
        }
    }
}

impl<T: Clone> LastGoodDiscovery<T> {
    fn record(&self, value: T) -> T {
        *self.value.lock().expect("discovery cache mutex poisoned") = Some(LastGoodValue {
            value: value.clone(),
            updated_at: std::time::Instant::now(),
        });
        value
    }

    fn get_stale(&self) -> Option<(T, Duration)> {
        self.value
            .lock()
            .expect("discovery cache mutex poisoned")
            .as_ref()
            .map(|cached| (cached.value.clone(), cached.updated_at.elapsed()))
    }
}

struct LastGoodValue<T> {
    value: T,
    updated_at: std::time::Instant,
}

/// How many message-context -> response_id entries `responses_chain` keeps
/// before evicting the oldest. Bounds memory for long-lived processes with
/// many/large conversations without needing a `lru` crate dependency.
const RESPONSES_CHAIN_CACHE_CAP: usize = 512;

/// Bounded content-keyed cache backing `BedrockClient::responses_chain`.
/// Plain `HashMap` + FIFO insertion-order eviction -- no promote-on-read --
/// which is enough to bound memory; see `RESPONSES_CHAIN_CACHE_CAP`.
#[derive(Debug, Default)]
struct ResponsesChainCache {
    entries: HashMap<u64, String>,
    order: std::collections::VecDeque<u64>,
    cap: usize,
}

impl ResponsesChainCache {
    fn new(cap: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: std::collections::VecDeque::new(),
            cap,
        }
    }

    fn get(&self, key: u64) -> Option<String> {
        self.entries.get(&key).cloned()
    }

    fn insert(&mut self, key: u64, value: String) {
        if !self.entries.contains_key(&key) {
            self.order.push_back(key);
            while self.order.len() > self.cap {
                if let Some(oldest) = self.order.pop_front() {
                    self.entries.remove(&oldest);
                }
            }
        }
        self.entries.insert(key, value);
    }

    fn evict(&mut self, key: u64) {
        self.entries.remove(&key);
        self.order.retain(|k| *k != key);
    }

    fn evict_context_prefixes(&mut self, messages: &[ChatMessage]) -> usize {
        let mut evicted = 0;
        for a in (0..messages.len()).rev() {
            if messages[a].role != "assistant" {
                continue;
            }
            let key = hash_responses_context(&messages[..a]);
            if self.entries.contains_key(&key) {
                self.evict(key);
                evicted += 1;
            }
        }
        evicted
    }
}

/// Hashes an ordered Responses API message context (role + normalized
/// content + tool-call/tool-result fields, in order) to a stable 64-bit key
/// for `ResponsesChainCache`. Order-sensitive and content-sensitive: any
/// difference in message order, text, or tool-call identity/arguments
/// produces a different hash.
fn hash_responses_context(messages: &[ChatMessage]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    messages.len().hash(&mut hasher);
    for msg in messages {
        hash_responses_message(msg, &mut hasher);
    }
    hasher.finish()
}

fn hash_responses_message(msg: &ChatMessage, hasher: &mut impl std::hash::Hasher) {
    use std::hash::Hash;
    msg.role.hash(hasher);
    for part in &msg.content {
        match part {
            ChatContentPart::Text { text } => {
                0u8.hash(hasher);
                text.hash(hasher);
            }
            ChatContentPart::Image { image_url } => {
                1u8.hash(hasher);
                image_url.hash(hasher);
            }
        }
    }
    match &msg.tool_calls {
        Some(calls) => {
            calls.len().hash(hasher);
            for call in calls {
                call.id.hash(hasher);
                call.r#type.hash(hasher);
                call.function.name.hash(hasher);
                call.function.arguments.hash(hasher);
            }
        }
        None => 0usize.hash(hasher),
    }
    msg.tool_call_id.hash(hasher);
    msg.name.hash(hasher);
}

/// Finds the largest cached prefix of `messages` that matches a previously
/// stored Responses API context, so the delta sent as the next turn's input
/// is as small as possible.
///
/// Walks assistant-message boundary indices from the *end* of `messages`
/// toward the start. For each assistant index `a`, looks up
/// `hash(messages[0..a])`: a hit means messages `0..a` were exactly the
/// input of a previously stored response, and `messages[a]` is that
/// response's assistant turn now echoed back into history. The first
/// (largest-`a`) hit wins. Returns `(a, previous_response_id)`; the caller
/// sends `previous_response_id` plus `messages[a+1..]` as the new input.
///
/// Returns `None` on a fresh conversation, an evicted/never-seen prefix, or
/// a first turn -- callers should fall back to sending the full `messages`
/// as today.
fn find_responses_continuation(
    messages: &[ChatMessage],
    lookup: impl Fn(u64) -> Option<String>,
) -> Option<(usize, String)> {
    for a in (0..messages.len()).rev() {
        if messages[a].role != "assistant" {
            continue;
        }
        let hash = hash_responses_context(&messages[..a]);
        if let Some(id) = lookup(hash) {
            return Some((a, id));
        }
    }
    None
}

/// Heuristic match for a Responses API error indicating `previous_response_id`
/// was unknown, expired, or otherwise invalid, so the caller can evict the
/// cache entry and retry once with the full input. The exact error-body
/// shape isn't documented; this defensively matches a client-error status
/// plus the field name and a rejection-ish word in the body rather than a
/// single exact message.
fn looks_like_expired_previous_response_id(status: reqwest::StatusCode, body: &str) -> bool {
    if !status.is_client_error() {
        return false;
    }
    let lower = body.to_ascii_lowercase();
    (lower.contains("previous_response_id")
        && (lower.contains("not found")
            || lower.contains("not exist")
            || lower.contains("expired")
            || lower.contains("invalid")
            || lower.contains("unknown")
            || lower.contains("no longer")))
        || lower.contains("not_found_error")
        || (lower.contains("response") && lower.contains("not found"))
}

/// Which Anthropic extended-thinking request shape a Bedrock model accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkingShape {
    /// Legacy extended thinking: `thinking: {type: "enabled", budget_tokens}`.
    Enabled,
    /// Newer effort-based control: `thinking: {type: "adaptive"}` plus
    /// `output_config: {effort}`.
    Adaptive,
}

impl std::fmt::Debug for BedrockClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BedrockClient")
            .field("bearer_token", &"[REDACTED]")
            .field("region", &self.region)
            .field("default_model", &self.default_model)
            .finish()
    }
}

#[derive(Debug)]
struct DiscoveryFailure {
    err: anyhow::Error,
    attempts: u64,
}

async fn retry_bedrock_discovery<T, Fut>(
    source: &'static str,
    mut operation: impl FnMut() -> Fut,
) -> std::result::Result<T, DiscoveryFailure>
where
    Fut: std::future::Future<Output = Result<T>>,
{
    for attempt in 1..=BEDROCK_DISCOVERY_MAX_ATTEMPTS {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(err) if attempt == BEDROCK_DISCOVERY_MAX_ATTEMPTS => {
                return Err(DiscoveryFailure {
                    err,
                    attempts: attempt,
                });
            }
            Err(_) => {
                let delay = bedrock_discovery_retry_backoff(attempt);
                tracing::debug!(
                    "{source} attempt {attempt}/{BEDROCK_DISCOVERY_MAX_ATTEMPTS} failed; retrying in {delay:?}"
                );
                tokio::time::sleep(delay).await;
            }
        }
    }
    unreachable!("retry loop returns on success or final failure")
}

fn bedrock_discovery_retry_backoff(attempt: u64) -> Duration {
    let exp = 2u64.saturating_pow(attempt.saturating_sub(1) as u32);
    let raw = BEDROCK_DISCOVERY_RETRY_BASE_DELAY
        .as_millis()
        .saturating_mul(u128::from(exp));
    let jitter = rand::rng().random_range(0.5..1.5);
    Duration::from_millis((raw as f64 * jitter) as u64)
}

fn discovery_result_or_last_good<T: Clone>(
    cache: &LastGoodDiscovery<T>,
    result: std::result::Result<T, DiscoveryFailure>,
    empty: impl FnOnce() -> T,
) -> T {
    match result {
        Ok(value) => cache.record(value),
        Err(failure) => {
            tracing::info!(
                "{} skipped after {} attempts: {:#}",
                cache.name,
                failure.attempts,
                failure.err
            );
            if let Some((value, age)) = cache.get_stale() {
                tracing::warn!(
                    "{} serving stale catalog data from last successful discovery (age: {:?})",
                    cache.name,
                    age
                );
                value
            } else {
                empty()
            }
        }
    }
}

impl BedrockClient {
    pub fn new(bearer_token: String, region: String, default_model: String) -> Self {
        Self::new_with_catalog_mode(
            bearer_token,
            region,
            default_model,
            crate::setup_state::bedrock_catalog_mode(),
        )
    }

    pub fn new_with_catalog_mode(
        bearer_token: String,
        region: String,
        default_model: String,
        catalog_mode: crate::setup_state::BedrockCatalogMode,
    ) -> Self {
        let http = OpenAiClient::apply_runtime_tls_workarounds(
            reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(600)),
            BEDROCK_RUNTIME_BASE_URL,
        )
        .build()
        .expect("failed to build Bedrock HTTP client");
        Self {
            bearer_token,
            region: region.clone(),
            default_model,
            http,
            runtime_base_url: BEDROCK_RUNTIME_BASE_URL.to_string(),
            mantle_base_url: mantle_base_url(&region),
            control_base_url: BEDROCK_CONTROL_BASE_URL.to_string(),
            catalog_mode,
            thinking_shape_cache: Arc::new(RwLock::new(HashMap::new())),
            responses_chain: Arc::new(std::sync::Mutex::new(ResponsesChainCache::new(
                RESPONSES_CHAIN_CACHE_CAP,
            ))),
            discovery_cache: Arc::new(BedrockDiscoveryCache::new()),
        }
    }

    #[cfg(test)]
    fn with_base_urls(
        bearer_token: String,
        region: String,
        default_model: String,
        runtime_base_url: String,
        mantle_base_url: String,
        control_base_url: String,
    ) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(600))
            .build()
            .expect("failed to build test Bedrock HTTP client");
        Self {
            bearer_token,
            region,
            default_model,
            http,
            runtime_base_url,
            mantle_base_url,
            control_base_url,
            catalog_mode: crate::setup_state::BedrockCatalogMode::MantlePreferred,
            thinking_shape_cache: Arc::new(RwLock::new(HashMap::new())),
            responses_chain: Arc::new(std::sync::Mutex::new(ResponsesChainCache::new(
                RESPONSES_CHAIN_CACHE_CAP,
            ))),
            discovery_cache: Arc::new(BedrockDiscoveryCache::new()),
        }
    }

    /// URL for the native Anthropic streaming endpoint. The non-streaming
    /// `/invoke` sibling returns nothing until generation completes, which
    /// leaves an `ADAPTIVE_MAX_TOKENS` run with zero observability; this one
    /// emits `application/vnd.amazon.eventstream` frames as they are produced.
    fn invoke_stream_url(&self, model: &str) -> String {
        let encoded = percent_encode_path_segment(model);
        if self.runtime_base_url.starts_with("http://")
            || self.runtime_base_url.starts_with("https://")
        {
            if self.runtime_base_url == "https://bedrock-runtime"
                || self.runtime_base_url == "http://bedrock-runtime"
            {
                return format!(
                    "{}.{}.amazonaws.com/model/{encoded}/invoke-with-response-stream",
                    self.runtime_base_url.trim_end_matches('.'),
                    self.region,
                );
            }
            if self.runtime_base_url.contains("amazonaws.com") {
                return format!(
                    "{}/model/{encoded}/invoke-with-response-stream",
                    self.runtime_base_url.trim_end_matches('/')
                );
            }
            return format!(
                "{}/model/{encoded}/invoke-with-response-stream",
                self.runtime_base_url.trim_end_matches('/')
            );
        }
        format!(
            "{}.{}.amazonaws.com/model/{encoded}/invoke-with-response-stream",
            self.runtime_base_url, self.region
        )
    }

    fn catalog_url(&self) -> String {
        if (self.runtime_base_url.starts_with("http://")
            || self.runtime_base_url.starts_with("https://"))
            && self.runtime_base_url != "https://bedrock-runtime"
            && self.runtime_base_url != "http://bedrock-runtime"
            && !self.runtime_base_url.contains("amazonaws.com")
        {
            return format!(
                "{}/foundation-models",
                self.runtime_base_url.trim_end_matches('/')
            );
        }
        format!(
            "https://bedrock.{}.amazonaws.com/foundation-models",
            self.region
        )
    }

    fn inference_profiles_url(&self) -> String {
        if (self.control_base_url.starts_with("http://")
            || self.control_base_url.starts_with("https://"))
            && self.control_base_url != "https://bedrock"
            && self.control_base_url != "http://bedrock"
            && !self.control_base_url.contains("amazonaws.com")
        {
            return format!(
                "{}/inference-profiles?typeEquals=SYSTEM_DEFINED",
                self.control_base_url.trim_end_matches('/')
            );
        }
        format!(
            "{}.{}.amazonaws.com/inference-profiles?typeEquals=SYSTEM_DEFINED",
            self.control_base_url.trim_end_matches('.'),
            self.region
        )
    }

    async fn discover_model_metadata(&self) -> Result<Vec<ModelMetadata>> {
        discover_model_metadata_with_http(
            &self.http,
            &self.catalog_url(),
            &self.bearer_token,
            &self.default_model,
        )
        .await
    }

    async fn invoke_model(&self, request: StreamChatRequest) -> Result<LlmResponse> {
        if uses_responses_api(&request.model) {
            return self.invoke_responses_model(request).await;
        }
        let StreamChatRequest {
            model,
            messages,
            tools,
            reasoning_effort,
            service_tier: _service_tier,
            temperature,
            structured_output: _,
            mut on_token,
            mut on_thought,
            cancel,
            idle_timeouts,
        } = request;
        let resolved_model = self.resolve_invocable_model_id(&model).await?;
        let enable_cache = requires_explicit_caching(&resolved_model);
        let (system_blocks, messages) = convert_messages(messages, enable_cache)?;
        let system = if system_blocks.is_empty() {
            None
        } else {
            Some(system_blocks)
        };
        let tools = tools.map(|t| convert_tools(t, enable_cache));

        // Bedrock does not publish a single uniform reasoning schema. Validate
        // against the model family we advertise, then emit that family's native
        // Anthropic thinking shape below.
        let reasoning_spec = bedrock_reasoning_spec_or_fallback(&resolved_model);
        let effort = reasoning_effort
            .as_deref()
            .filter(|e| reasoning_spec.supports_reasoning_level(e))
            .map(str::to_string);
        let send_output_effort = reasoning_spec.send_output_effort;
        let preferred_shape = reasoning_spec
            .anthropic_thinking_shape
            .unwrap_or(ThinkingShape::Enabled);

        // Start from the documented shape for this family, unless a prior
        // request learned a different accepted shape from the provider.
        let mut shape = match &effort {
            Some(_) => self
                .thinking_shape_cache
                .read()
                .await
                .get(&resolved_model)
                .copied()
                .unwrap_or(preferred_shape),
            None => preferred_shape,
        };

        let url = self.invoke_stream_url(&resolved_model);
        let outcome = loop {
            let body = build_anthropic_request(
                ANTHROPIC_VERSION,
                system.clone(),
                messages.clone(),
                tools.clone(),
                temperature,
                AnthropicReasoningControls {
                    effort: effort.as_deref(),
                    shape,
                    send_output_effort,
                },
            );
            trace_bedrock_request(&body);
            match self
                .invoke_native_anthropic_with_fallback(
                    &resolved_model,
                    &url,
                    &body,
                    &cancel,
                    idle_timeouts,
                    &mut on_token,
                    &mut on_thought,
                )
                .await
            {
                Ok(outcome) => {
                    if effort.is_some() {
                        self.thinking_shape_cache
                            .write()
                            .await
                            .insert(resolved_model.clone(), shape);
                    }
                    break outcome;
                }
                // The model rejected the legacy `enabled` shape and told us to
                // use `adaptive` + `output_config.effort`. Switch shape once
                // and retry; the cache (set on success above) prevents this
                // probe from recurring on later turns.
                Err(err)
                    if shape == ThinkingShape::Enabled
                        && effort.is_some()
                        && error_requires_adaptive_thinking(&err) =>
                {
                    tracing::info!(
                        "Bedrock model {} rejected enabled-thinking; retrying with adaptive effort control",
                        resolved_model
                    );
                    shape = ThinkingShape::Adaptive;
                }
                Err(err) => return Err(err),
            }
        };
        let NativeStreamOutcome {
            text,
            thoughts,
            calls,
            usage,
            stop_reason,
        } = outcome;
        if cancel.is_cancelled() {
            return Ok(LlmResponse::Text {
                text,
                reasoning_content: (!thoughts.is_empty()).then_some(thoughts),
                usage,
            });
        }
        if calls.is_empty() && stop_reason.as_deref() == Some("max_tokens") {
            // Hit the output ceiling with nothing to show for it. Under
            // adaptive thinking this is the budget-exhaustion case: the
            // whole allowance went to thinking, so the stream carried only
            // thinking deltas. Retrying re-sends an identical request that
            // hits the identical cap, so classify it as non-retryable
            // rather than letting it read as a generic empty completion.
            if text.trim().is_empty() {
                return Err(anyhow::Error::new(OutputBudgetExhaustedError));
            }
            tracing::warn!(
                "Bedrock response hit max_tokens after emitting text; returning truncated content"
            );
        }
        if calls.is_empty() {
            Ok(LlmResponse::Text {
                text,
                reasoning_content: (!thoughts.is_empty()).then_some(thoughts),
                usage,
            })
        } else {
            Ok(LlmResponse::ToolCalls {
                text,
                reasoning_content: (!thoughts.is_empty()).then_some(thoughts),
                calls,
                usage,
            })
        }
    }

    /// Posts a single Responses API request and returns the raw HTTP
    /// response (success or not -- callers decide what to do with a
    /// non-2xx status). Shared by the fresh-input and delta-continuation
    /// attempts in `invoke_responses_model` so both retry transiently via
    /// `send_with_retries` the same way.
    async fn post_responses_request(
        &self,
        url: &str,
        body: &crate::responses_api::ResponsesRequest,
        cancel: &tokio_util::sync::CancellationToken,
        first_progress_timeout: Duration,
    ) -> Result<reqwest::Response> {
        crate::http_retry::send_with_retries(
            "posting Bedrock Responses API request",
            || {
                self.http
                    .post(url)
                    .header("Accept", "text/event-stream")
                    .bearer_auth(&self.bearer_token)
                    .json(body)
            },
            Some(cancel),
            Some(first_progress_timeout),
        )
        .await
    }

    /// Drives one turn of the OpenAI Responses API on Bedrock Mantle.
    ///
    /// `LlmBackend::stream_chat` receives the *full* conversation on every
    /// call with no conversation id, so resending it all every turn with
    /// `store: false` (the historical behavior) only lets OpenAI's
    /// prompt-cache match the first-turn prefix -- it never advances as the
    /// conversation grows. Instead: chain turns server-side via
    /// `store: true` + `previous_response_id`, keyed off a content hash of
    /// the message context (`responses_chain`), and send only the new
    /// messages since the cached response as `input`. See
    /// `find_responses_continuation` / `hash_responses_context` for the
    /// matching logic and `looks_like_expired_previous_response_id` for the
    /// self-healing fallback below.
    async fn invoke_responses_model(&self, request: StreamChatRequest) -> Result<LlmResponse> {
        let StreamChatRequest {
            model,
            messages,
            tools,
            reasoning_effort,
            service_tier: _service_tier,
            temperature: _temperature,
            structured_output,
            on_token,
            on_thought,
            cancel,
            idle_timeouts,
        } = request;
        let reasoning_spec = bedrock_reasoning_spec_or_fallback(&model);
        let reasoning_effort = reasoning_effort
            .as_deref()
            .filter(|e| reasoning_spec.supports_reasoning_level(e))
            .map(str::to_string);
        let url = format!(
            "{}/responses",
            mantle_base_url_for_model(&self.mantle_base_url, &model)
        );

        // Look for the largest cached prefix of `messages` we can chain
        // onto. Locking is synchronous and never held across the network
        // call below.
        let continuation = {
            let cache = self
                .responses_chain
                .lock()
                .expect("responses_chain mutex poisoned");
            find_responses_continuation(&messages, |h| cache.get(h))
        };

        let build_fresh_body = || {
            build_responses_request(
                &model,
                &messages,
                tools.as_deref(),
                reasoning_effort.as_deref(),
                structured_output.as_ref(),
                true,
                None,
            )
        };

        let (body, continuation_cache_key) = match &continuation {
            Some((boundary, previous_response_id)) => {
                let delta = &messages[boundary + 1..];
                let body = build_responses_request(
                    &model,
                    delta,
                    tools.as_deref(),
                    reasoning_effort.as_deref(),
                    structured_output.as_ref(),
                    true,
                    Some(previous_response_id.as_str()),
                );
                (body, Some(hash_responses_context(&messages[..*boundary])))
            }
            None => (build_fresh_body(), None),
        };

        let resp = self
            .post_responses_request(&url, &body, &cancel, idle_timeouts.first_progress)
            .await?;
        let status = resp.status();
        let resp = if status.is_success() {
            resp
        } else {
            let body_text = resp.text().await.unwrap_or_default();
            if continuation.is_some() && looks_like_expired_previous_response_id(status, &body_text)
            {
                // Self-healing: the chained response_id is no longer valid
                // server-side (expired/evicted upstream). Evict it locally
                // so we don't keep retrying it, then retry once with the
                // full conversation and no previous_response_id -- exactly
                // like a fresh/no-match turn. No tokens have been emitted
                // yet (we haven't touched the SSE body), so replaying here
                // cannot duplicate output.
                if let Some(key) = continuation_cache_key {
                    self.responses_chain
                        .lock()
                        .expect("responses_chain mutex poisoned")
                        .evict(key);
                }
                tracing::warn!(
                    %status,
                    "Bedrock Responses API rejected previous_response_id as expired/unknown; evicting and retrying with full input"
                );
                let retry_body = build_fresh_body();
                let retry_resp = self
                    .post_responses_request(
                        &url,
                        &retry_body,
                        &cancel,
                        idle_timeouts.first_progress,
                    )
                    .await?;
                let retry_status = retry_resp.status();
                if !retry_status.is_success() {
                    let retry_body_text = retry_resp.text().await.unwrap_or_default();
                    return Err(crate::http_retry::retryable_llm_error_for_status_and_body(
                        format!(
                            "Bedrock Responses API failed (HTTP {retry_status}) after retrying without previous_response_id: {retry_body_text}"
                        ),
                        retry_status,
                        &retry_body_text,
                    ));
                }
                retry_resp
            } else {
                return Err(crate::http_retry::retryable_llm_error_for_status_and_body(
                    format!("Bedrock Responses API failed (HTTP {status}): {body_text}"),
                    status,
                    &body_text,
                ));
            }
        };

        let stream = resp
            .bytes_stream()
            .map(|r| r.map(|b| b.to_vec()).map_err(anyhow::Error::from));
        let outcome = match drive_responses_sse_stream(
            stream,
            on_token,
            on_thought,
            cancel,
            idle_timeouts,
        )
        .await
        {
            Ok(outcome) => outcome,
            Err(err) => {
                if continuation.is_some() {
                    let evicted = self
                        .responses_chain
                        .lock()
                        .expect("responses_chain mutex poisoned")
                        .evict_context_prefixes(&messages);
                    tracing::warn!(
                        evicted,
                        "Bedrock Responses API stream failed on chained call; evicting cached prefixes before outer retry"
                    );
                }
                return Err(err);
            }
        };

        if let Some(response_id) = outcome.response_id {
            let key = hash_responses_context(&messages);
            self.responses_chain
                .lock()
                .expect("responses_chain mutex poisoned")
                .insert(key, response_id);
        }

        Ok(outcome.response)
    }

    async fn list_mantle_model_metadata(&self) -> Result<Vec<ModelMetadata>> {
        let url = format!("{}/models", self.mantle_base_url.trim_end_matches('/'));
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.bearer_token)
            .send()
            .await
            .with_context(|| format!("GET {url}"))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Bedrock Mantle /models failed (HTTP {status}): {body}");
        }
        let parsed: ModelsResponse = resp
            .json()
            .await
            .context("parsing Bedrock Mantle /models")?;
        Ok(parsed
            .data
            .into_iter()
            .map(|model| model.to_model_metadata())
            .collect())
    }

    async fn list_mantle_model_metadata_with_fallback(&self) -> Vec<ModelMetadata> {
        let result = retry_bedrock_discovery("Bedrock Mantle model discovery", || {
            self.list_mantle_model_metadata()
        })
        .await;
        discovery_result_or_last_good(&self.discovery_cache.mantle_models, result, Vec::new)
    }

    async fn discover_model_metadata_with_fallback(&self) -> Vec<ModelMetadata> {
        let result = retry_bedrock_discovery("Bedrock foundation-model discovery", || {
            self.discover_model_metadata()
        })
        .await;
        discovery_result_or_last_good(&self.discovery_cache.foundation_models, result, Vec::new)
    }

    /// Opens the native Anthropic stream, retrying through inference profile
    /// ids when the plain model id is rejected as on-demand-unsupported.
    ///
    /// The profile retry is only sound because it happens before any frame is
    /// consumed: a rejected request is a 4xx with a JSON error body, so no
    /// token has reached `on_token`/`on_thought` yet and replaying cannot
    /// duplicate visible output.
    #[allow(clippy::too_many_arguments)]
    async fn invoke_native_anthropic_with_fallback(
        &self,
        model: &str,
        url: &str,
        body: &BedrockAnthropicRequest,
        cancel: &tokio_util::sync::CancellationToken,
        idle: IdleTimeouts,
        on_token: &mut TokenSink,
        on_thought: &mut TokenSink,
    ) -> Result<NativeStreamOutcome> {
        let (status, body_text) = match self.post_native_invoke(url, body, cancel).await? {
            NativeInvokeAttempt::Cancelled => return Ok(NativeStreamOutcome::default()),
            NativeInvokeAttempt::Streaming(resp) => {
                return drive_native_anthropic_stream(
                    resp,
                    on_token,
                    on_thought,
                    cancel,
                    native_stream_idle_timeouts(idle),
                )
                .await;
            }
            NativeInvokeAttempt::Failed { status, body } => (status, body),
        };

        if !needs_inference_profile_retry(status, &body_text)
            || looks_like_inference_profile_identifier(model)
        {
            return Err(crate::http_retry::retryable_llm_error_for_status_and_body(
                format!("Bedrock request failed (HTTP {status}): {body_text}"),
                status,
                &body_text,
            ));
        }

        let mut last_retry_error = None;
        for profile_id in self.inference_profile_candidates_for_model(model).await {
            tracing::info!(
                "retrying Bedrock invoke for model {} with inference profile {}",
                model,
                profile_id
            );
            let retry_url = self.invoke_stream_url(&profile_id);
            let (retry_status, retry_body) =
                match self.post_native_invoke(&retry_url, body, cancel).await? {
                    NativeInvokeAttempt::Cancelled => return Ok(NativeStreamOutcome::default()),
                    NativeInvokeAttempt::Streaming(resp) => {
                        return drive_native_anthropic_stream(
                            resp,
                            on_token,
                            on_thought,
                            cancel,
                            native_stream_idle_timeouts(idle),
                        )
                        .await;
                    }
                    NativeInvokeAttempt::Failed { status, body } => (status, body),
                };
            last_retry_error = Some((profile_id, retry_status, retry_body));
        }

        if let Some((profile_id, retry_status, retry_body)) = last_retry_error {
            let message = format!(
                "Bedrock request failed (HTTP {status}): {body_text}; retry with inference profile {} also failed (HTTP {retry_status}): {retry_body}",
                profile_id
            );
            let bodies = format!("{body_text}\n{retry_body}");
            return Err(crate::http_retry::retryable_llm_error_for_status_and_body(
                message,
                retry_status,
                &bodies,
            ));
        }

        Err(crate::http_retry::retryable_llm_error_for_status_and_body(
            format!("Bedrock request failed (HTTP {status}): {body_text}"),
            status,
            &body_text,
        ))
    }

    async fn post_native_invoke(
        &self,
        url: &str,
        body: &BedrockAnthropicRequest,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> Result<NativeInvokeAttempt> {
        // Ceiling on the whole exchange, headers plus streamed body. Now that
        // frames arrive continuously the per-chunk idle timeouts below are the
        // load-bearing stall detector, so this could be relaxed toward the
        // client-wide default; left as-is until the streaming path has run in
        // anger. Overridden per request rather than globally so discovery
        // calls keep failing fast.
        let send = self
            .http
            .post(url)
            .timeout(NATIVE_INVOKE_TIMEOUT)
            .bearer_auth(&self.bearer_token)
            .json(body)
            .send();

        let resp = tokio::select! {
            _ = cancel.cancelled() => {
                return Ok(NativeInvokeAttempt::Cancelled);
            }
            resp = send => resp.context("failed to send Bedrock request")?,
        };

        let status = resp.status();
        if status.is_success() {
            return Ok(NativeInvokeAttempt::Streaming(resp));
        }
        Ok(NativeInvokeAttempt::Failed {
            status,
            body: resp.text().await.unwrap_or_default(),
        })
    }

    async fn inference_profile_candidates_for_model(&self, model: &str) -> Vec<String> {
        let profiles = self.list_inference_profiles_with_fallback().await;
        let mut candidates = match_inference_profiles_to_model(model, &profiles, &self.region);
        candidates.extend(guessed_inference_profile_candidates(model, &self.region));
        dedup_preserve_order(candidates)
    }

    async fn resolve_invocable_model_id(&self, model: &str) -> Result<String> {
        if looks_like_inference_profile_identifier(model) || uses_responses_api(model) {
            return Ok(model.to_string());
        }

        let profiles = self.list_inference_profiles_with_fallback().await;
        if let Some(invocable_id) =
            preferred_invocable_bedrock_model_id(model, &profiles, &self.region)
        {
            return Ok(invocable_id);
        }

        Ok(model.to_string())
    }

    async fn list_inference_profiles(&self) -> Result<Vec<BedrockInferenceProfileSummary>> {
        let url = self.inference_profiles_url();
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.bearer_token)
            .send()
            .await
            .with_context(|| format!("GET {url}"))?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            anyhow::bail!("Bedrock inference profile discovery failed (HTTP {status}): {body}");
        }
        let parsed: BedrockListInferenceProfilesResponse =
            serde_json::from_str(&body).context("parse Bedrock inference profile response")?;
        Ok(parsed.inference_profile_summaries)
    }

    async fn list_inference_profiles_with_fallback(&self) -> Vec<BedrockInferenceProfileSummary> {
        let result = retry_bedrock_discovery("Bedrock inference profile discovery", || {
            self.list_inference_profiles()
        })
        .await;
        discovery_result_or_last_good(&self.discovery_cache.inference_profiles, result, Vec::new)
    }
}

impl LlmBackend for BedrockClient {
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
        Box::pin(async move {
            Ok(self
                .list_model_metadata()
                .await?
                .into_iter()
                .map(|m| m.id)
                .collect())
        })
    }

    fn list_model_metadata(&self) -> BoxFuture<'_, Result<Vec<ModelMetadata>>> {
        Box::pin(async move {
            use crate::setup_state::BedrockCatalogMode;

            let discover_mantle =
                || async { self.list_mantle_model_metadata_with_fallback().await };
            let discover_native = || async { self.discover_model_metadata_with_fallback().await };

            let (mut models, uses_native) = match self.catalog_mode {
                BedrockCatalogMode::MantleOnly => (discover_mantle().await, false),
                BedrockCatalogMode::NativeOnly => (discover_native().await, true),
                BedrockCatalogMode::MantlePreferred => {
                    let mut models = discover_mantle().await;
                    models.extend(discover_native().await);
                    (models, true)
                }
                BedrockCatalogMode::NativePreferred => {
                    let mut models = discover_native().await;
                    models.extend(discover_mantle().await);
                    (models, true)
                }
            };
            let inference_profiles = if uses_native {
                self.list_inference_profiles_with_fallback().await
            } else {
                Vec::new()
            };
            let default_model = normalize_default_bedrock_model(
                &self.default_model,
                &inference_profiles,
                &self.region,
            );
            models = normalize_bedrock_model_ids(models, &inference_profiles, &self.region);

            // Enrich AFTER merge + normalization: the discovery sources can
            // disagree or omit per-effort details. The final picker must match
            // the real Bedrock/provider model family, not generic catalog
            // defaults.
            apply_bedrock_reasoning_presets(&mut models);

            if !models.iter().any(|m| m.id == default_model) {
                let reasoning_spec = bedrock_reasoning_spec_or_fallback(&default_model);
                models.push(ModelMetadata {
                    id: default_model.clone(),
                    default_reasoning_level: reasoning_spec.default_level.map(str::to_string),
                    supported_reasoning_levels: reasoning_spec.to_presets(),
                    service_tiers: Vec::new(),
                    supports_images: None,
                    context_length: Some(200_000),
                    pricing: None,
                });
            }
            models.sort_by(|a, b| {
                let a_default = a.id == default_model;
                let b_default = b.id == default_model;
                b_default.cmp(&a_default).then_with(|| a.id.cmp(&b.id))
            });
            models.dedup_by(|a, b| a.id == b.id);
            Ok(models)
        })
    }

    fn stream_chat(&self, request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
        Box::pin(self.invoke_model(request))
    }
}

fn uses_responses_api(model: &str) -> bool {
    model.starts_with("openai.")
}

/// The gpt-5.4, gpt-5.5, and gpt-5.6 model families are served from a dedicated
/// `/openai/v1` path on Bedrock Mantle rather than the shared `/v1` path every
/// other Mantle model uses (per their Bedrock model cards). Match the bare ids
/// and any suffixed derivative (e.g. `openai.gpt-5.5-codex`, `openai.gpt-5.6-sol`),
/// but not unrelated ids that merely share the numeric prefix (e.g. a future
/// `openai.gpt-5.40`).
fn uses_openai_mantle_path(model: &str) -> bool {
    ["openai.gpt-5.4", "openai.gpt-5.5", "openai.gpt-5.6"]
        .iter()
        .any(|base| match model.strip_prefix(base) {
            Some(rest) => rest.is_empty() || rest.starts_with('-'),
            None => false,
        })
}

/// Resolve the Mantle base URL for a specific model: the shared `.../v1` base
/// for most models, rewritten to `.../openai/v1` for the families that require
/// the dedicated path. Falls back to the base unchanged if it lacks the
/// expected `/v1` suffix so an unusual override can never produce a broken URL.
fn mantle_base_url_for_model(base: &str, model: &str) -> String {
    let base = base.trim_end_matches('/');
    match base.strip_suffix("/v1") {
        Some(prefix) if uses_openai_mantle_path(model) => format!("{prefix}/openai/v1"),
        _ => base.to_string(),
    }
}

fn mantle_base_url(region: &str) -> String {
    format!("https://bedrock-mantle.{region}.api.aws/v1")
}

/// Resolve bearer token, region, and model from all available sources.
/// Precedence for each: env > brokk config file > ~/.secrets/ > default.
/// Kept as thin wrappers for backward compat; the authoritative
/// resolution lives in `bedrock_auth` so the setup handlers and startup
/// path share the same precedence logic.
pub fn bearer_token_from_env_or_secrets() -> Result<Option<String>> {
    if let Ok(raw) = std::env::var(BEDROCK_API_KEY_ENV) {
        let token = raw.trim();
        if !token.is_empty() {
            return Ok(Some(token.to_string()));
        }
    }

    if let Some(token) = bearer_token_from_brokk_config()? {
        return Ok(Some(token));
    }

    bearer_token_from_secrets()
}

/// Read the bearer token from the legacy `~/.secrets/` fallback files
/// (lowest precedence). Kept so users who configured Bedrock before the
/// managed credential file existed keep working. Detection
/// (`CredentialState::snapshot`) reads the same files through this
/// function so the setup/status UI never disagrees with the token the
/// backend actually resolves.
pub(crate) const BEDROCK_SECRET_FILE_NAMES: [&str; 2] =
    ["aws_bearer_token_bedrock", "bedrock_api_key"];

pub fn bearer_token_from_secrets() -> Result<Option<String>> {
    for name in BEDROCK_SECRET_FILE_NAMES {
        if let Some(token) = read_secret_file(name)? {
            return Ok(Some(token));
        }
    }
    Ok(None)
}

pub fn bearer_token_from_brokk_config() -> Result<Option<String>> {
    Ok(crate::bedrock_auth::read()?.and_then(|auth| {
        let token = auth.bearer_token.trim();
        (!token.is_empty()).then(|| token.to_string())
    }))
}

pub fn region_from_env() -> String {
    std::env::var("AWS_REGION")
        .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
        .or_else(|_| std::env::var(BEDROCK_REGION_ENV))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(region_from_config)
        .unwrap_or_else(|| BEDROCK_DEFAULT_REGION.to_string())
}

pub fn region_from_config() -> Option<String> {
    crate::bedrock_auth::read().ok().flatten().and_then(|auth| {
        auth.region.and_then(|region| {
            let region = region.trim();
            (!region.is_empty()).then(|| region.to_string())
        })
    })
}

pub fn model_from_env() -> String {
    std::env::var(BEDROCK_MODEL_ENV)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(model_from_config)
        .unwrap_or_else(|| BEDROCK_DEFAULT_MODEL.to_string())
}

pub fn model_from_config() -> Option<String> {
    crate::bedrock_auth::read().ok().flatten().and_then(|auth| {
        auth.default_model.and_then(|model| {
            let model = model.trim();
            (!model.is_empty()).then(|| model.to_string())
        })
    })
}

pub fn backend_config() -> Result<Option<(String, String, String)>> {
    let Some(token) = bearer_token_from_env_or_secrets()? else {
        return Ok(None);
    };
    Ok(Some((token, region_from_env(), model_from_env())))
}

fn needs_inference_profile_retry(status: reqwest::StatusCode, body: &str) -> bool {
    status == reqwest::StatusCode::BAD_REQUEST
        && body.contains("on-demand throughput isn")
        && body.contains("inference profile")
}

fn looks_like_inference_profile_identifier(model: &str) -> bool {
    model.starts_with("global.")
        || model.starts_with("us.")
        || model.starts_with("eu.")
        || model.starts_with("jp.")
        || model.starts_with("au.")
        || model.contains(":inference-profile/")
}

fn guessed_inference_profile_candidates(model: &str, region: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    if let Some(prefix) = preferred_geo_prefix(region) {
        candidates.push(format!("{prefix}{model}"));
    }
    candidates.push(format!("global.{model}"));
    candidates
}

fn normalize_default_bedrock_model(
    model: &str,
    profiles: &[BedrockInferenceProfileSummary],
    region: &str,
) -> String {
    preferred_invocable_bedrock_model_id(model, profiles, region)
        .unwrap_or_else(|| model.to_string())
}

fn normalize_bedrock_model_ids(
    models: Vec<ModelMetadata>,
    profiles: &[BedrockInferenceProfileSummary],
    region: &str,
) -> Vec<ModelMetadata> {
    let mut by_id = HashMap::new();
    for mut model in models {
        if let Some(invocable_id) =
            preferred_invocable_bedrock_model_id(&model.id, profiles, region)
        {
            model.id = invocable_id;
        }
        by_id.entry(model.id.clone()).or_insert(model);
    }
    let mut normalized: Vec<ModelMetadata> = by_id.into_values().collect();
    normalized.sort_by(|a, b| a.id.cmp(&b.id));
    normalized
}

fn preferred_invocable_bedrock_model_id(
    model: &str,
    profiles: &[BedrockInferenceProfileSummary],
    region: &str,
) -> Option<String> {
    if looks_like_inference_profile_identifier(model) || uses_responses_api(model) {
        return None;
    }
    match_inference_profiles_to_model(model, profiles, region)
        .into_iter()
        .next()
}

fn preferred_geo_prefix(region: &str) -> Option<&'static str> {
    if region.starts_with("us-")
        || region.starts_with("ca-")
        || region.starts_with("sa-")
        || region.starts_with("mx-")
    {
        Some("us.")
    } else if region.starts_with("eu-") {
        Some("eu.")
    } else if matches!(region, "ap-northeast-1" | "ap-northeast-3") {
        Some("jp.")
    } else if matches!(region, "ap-southeast-2" | "ap-southeast-4") {
        Some("au.")
    } else {
        None
    }
}

fn match_inference_profiles_to_model(
    model: &str,
    profiles: &[BedrockInferenceProfileSummary],
    region: &str,
) -> Vec<String> {
    let mut exact_geo = Vec::new();
    let mut global = Vec::new();
    let mut other = Vec::new();
    let preferred_geo = preferred_geo_prefix(region);

    for profile in profiles {
        if !profile_matches_model(profile, model) {
            continue;
        }
        if let Some(prefix) = preferred_geo
            && profile.inference_profile_id.starts_with(prefix)
        {
            exact_geo.push(profile.inference_profile_id.clone());
        } else if profile.inference_profile_id.starts_with("global.") {
            global.push(profile.inference_profile_id.clone());
        } else {
            other.push(profile.inference_profile_id.clone());
        }
    }

    exact_geo.extend(global);
    exact_geo.extend(other);
    exact_geo
}

fn profile_matches_model(profile: &BedrockInferenceProfileSummary, model: &str) -> bool {
    profile
        .models
        .iter()
        .filter_map(|entry| entry.model_arn.rsplit('/').next())
        .any(|model_id| model_id == model)
}

fn dedup_preserve_order(items: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for item in items {
        if seen.insert(item.clone()) {
            out.push(item);
        }
    }
    out
}

/// Resolve the path to a legacy secrets file. Honours `BROKK_SECRETS_HOME`
/// (symmetric with `BROKK_CONFIG_HOME`) so tests and power users can
/// redirect the directory; otherwise `~/.secrets/<name>`.
pub(crate) fn secret_file_path(name: &str) -> Option<PathBuf> {
    if let Ok(custom) = std::env::var("BROKK_SECRETS_HOME") {
        let custom = custom.trim();
        if !custom.is_empty() {
            return Some(PathBuf::from(custom).join(name));
        }
    }
    dirs::home_dir().map(|home| home.join(".secrets").join(name))
}

fn read_secret_file(name: &str) -> Result<Option<String>> {
    let Some(path) = secret_file_path(name) else {
        return Ok(None);
    };
    if !path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let token = raw.trim();
    if token.is_empty() {
        Ok(None)
    } else {
        Ok(Some(token.to_string()))
    }
}

#[derive(Debug, Serialize)]
struct BedrockAnthropicRequest {
    anthropic_version: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<BedrockTextBlock>>,
    messages: Vec<BedrockMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<BedrockTool>>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<BedrockThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<BedrockOutputConfig>,
}

/// Anthropic extended-thinking control block. Two wire shapes exist:
///
/// - `Enabled { budget_tokens }` — legacy extended thinking. Requires
///   `temperature` unset and `max_tokens > budget_tokens`.
/// - `Adaptive` — newer effort-based control, paired with
///   `output_config: { effort }` on the request.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum BedrockThinking {
    #[serde(rename = "enabled")]
    Enabled { budget_tokens: u32 },
    /// `display` defaults to `omitted` on the models that use this shape,
    /// which streams thinking blocks whose text is an empty string. Ask for
    /// summaries instead so the reasoning reaches `on_thought` and the trace
    /// as `thinking_delta` events -- otherwise a long generation streams
    /// nothing at all until the answer starts.
    #[serde(rename = "adaptive")]
    Adaptive { display: &'static str },
}

/// `output_config.effort` companion for Anthropic effort-aware thinking.
/// Carries the user's chosen model-supported effort verbatim.
#[derive(Debug, Serialize)]
struct BedrockOutputConfig {
    effort: String,
}

#[derive(Debug, Serialize, Clone)]
struct CacheControl {
    r#type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl: Option<&'static str>,
}

#[derive(Debug, Serialize, Clone)]
struct BedrockTextBlock {
    #[serde(rename = "type")]
    block_type: &'static str,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Debug, Serialize, Clone)]
struct BedrockMessage {
    role: String,
    content: Vec<BedrockContentOut>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")]
enum BedrockContentOut {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "image")]
    Image { source: BedrockImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

#[derive(Debug, Serialize, Clone)]
struct BedrockImageSource {
    #[serde(rename = "type")]
    source_type: &'static str,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize, Clone)]
struct BedrockTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

/// One Anthropic streaming event, as carried inside an eventstream frame.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: AnthropicStreamMessage },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: u64,
        content_block: AnthropicStreamBlockStart,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: u64,
        delta: AnthropicStreamDelta,
    },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: AnthropicStreamMessageDelta,
        #[serde(default)]
        usage: Option<BedrockUsage>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    /// `error` events are terminal: the model stopped mid-generation.
    #[serde(rename = "error")]
    Error {
        #[serde(default)]
        error: serde_json::Value,
    },
    /// `content_block_stop`, `ping`, and anything the provider adds later.
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamMessage {
    #[serde(default)]
    usage: Option<BedrockUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicStreamBlockStart {
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicStreamDelta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
    /// `signature_delta` and friends carry no user-visible content.
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamMessageDelta {
    /// `max_tokens` here means the model hit the output ceiling. With
    /// adaptive thinking that regularly means it spent the whole budget
    /// thinking and emitted nothing else -- see `invoke_model`, which turns
    /// that case into `OutputBudgetExhaustedError` so it is not retried.
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct BedrockUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
}

impl BedrockUsage {
    fn into_usage(self) -> TokenUsage {
        TokenUsage {
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            thought_tokens: 0,
            cached_read_tokens: self.cache_read_input_tokens,
            cached_write_tokens: self.cache_creation_input_tokens,
        }
    }
}

/// Streaming splits usage across two events: `message_start` carries the
/// input and cache counts (with a placeholder output count), `message_delta`
/// carries the final output count. Merge by non-zero field so neither event
/// can blank out what the other reported.
fn merge_stream_usage(total: &mut TokenUsage, part: TokenUsage) {
    for (slot, value) in [
        (&mut total.input_tokens, part.input_tokens),
        (&mut total.output_tokens, part.output_tokens),
        (&mut total.thought_tokens, part.thought_tokens),
        (&mut total.cached_read_tokens, part.cached_read_tokens),
        (&mut total.cached_write_tokens, part.cached_write_tokens),
    ] {
        if value != 0 {
            *slot = value;
        }
    }
}

/// What one native streaming POST produced. A non-2xx never reaches the
/// stream, so the caller can still fall back to an inference profile.
enum NativeInvokeAttempt {
    Cancelled,
    Streaming(reqwest::Response),
    Failed {
        status: reqwest::StatusCode,
        body: String,
    },
}

/// Everything a completed native Anthropic stream accumulated. Text and
/// thoughts have already been forwarded to the sinks as they arrived; these
/// copies are what the caller returns in `LlmResponse`.
#[derive(Debug, Default)]
struct NativeStreamOutcome {
    text: String,
    thoughts: String,
    calls: Vec<ToolCall>,
    usage: TokenUsage,
    stop_reason: Option<String>,
}

/// A `tool_use` block being assembled from `input_json_delta` fragments.
struct PartialToolUse {
    id: String,
    name: String,
    arguments: String,
}

/// Bedrock reports mid-stream failures as eventstream exception frames or
/// Anthropic `error` events rather than an HTTP status, so the retry tier has
/// to be recovered from the failure name. These are the transient ones
/// documented for `InvokeModelWithResponseStream` plus Anthropic's own
/// overload/api error kinds.
const RETRYABLE_STREAM_FAILURES: &[&str] = &[
    "internalServerException",
    "modelStreamErrorException",
    "modelTimeoutException",
    "serviceUnavailableException",
    "throttlingException",
    "overloaded_error",
    "api_error",
];

fn stream_failure_error(kind: &str, body: &str) -> anyhow::Error {
    let message = format!("Bedrock stream failed ({kind}): {body}");
    // A daily quota reported as a `throttlingException` is still a wall;
    // the exception kind must not out-vote the quota id in the body.
    if let Some(error) = crate::http_retry::fatal_daily_quota_error(&message, body) {
        return error;
    }
    if RETRYABLE_STREAM_FAILURES
        .iter()
        .any(|known| known.eq_ignore_ascii_case(kind))
    {
        return crate::http_retry::retryable_llm_error(
            message,
            RetryableLlmError::fast("Bedrock stream failure"),
        );
    }
    crate::http_retry::retryable_llm_error_for_body(message, body)
}

/// Consumes an `invoke-with-response-stream` body, forwarding text and
/// summarized thinking to the sinks as it arrives.
///
/// Unlike every other backend here the wire format is not SSE: Bedrock frames
/// the response as `application/vnd.amazon.eventstream` (see
/// `EventStreamDecoder`), and each frame's payload is a base64-wrapped
/// Anthropic streaming event.
async fn drive_native_anthropic_stream(
    resp: reqwest::Response,
    on_token: &mut TokenSink,
    on_thought: &mut TokenSink,
    cancel: &tokio_util::sync::CancellationToken,
    idle: IdleTimeouts,
) -> Result<NativeStreamOutcome> {
    let mut stream = std::pin::pin!(resp.bytes_stream());
    let mut decoder = EventStreamDecoder::default();
    let mut outcome = NativeStreamOutcome::default();
    let mut tool_blocks: BTreeMap<u64, PartialToolUse> = BTreeMap::new();
    let mut deadline = tokio::time::Instant::now() + idle.first_progress;
    let mut saw_progress = false;
    let mut completed = false;
    let mut failure: Option<anyhow::Error> = None;

    loop {
        tokio::select! {
            _ = cancel.cancelled() => break,
            chunk_or_timeout = tokio::time::timeout_at(deadline, stream.next()) => {
                let Ok(chunk_opt) = chunk_or_timeout else {
                    if saw_progress {
                        return Err(crate::http_retry::retryable_llm_error(
                            format!(
                                "Bedrock stream stalled mid-stream for {}s; aborting",
                                idle.inter_chunk.as_secs()
                            ),
                            RetryableLlmError::fast("Bedrock stream stalled mid-stream"),
                        ));
                    }
                    return Err(crate::http_retry::retryable_llm_error(
                        format!(
                            "Bedrock stream made no first token for {}s; aborting",
                            idle.first_progress.as_secs()
                        ),
                        RetryableLlmError::fast("Bedrock stream made no first token"),
                    ));
                };
                let Some(chunk) = chunk_opt else { break };
                let chunk = chunk.map_err(|err| {
                    crate::http_retry::retryable_llm_context(
                        anyhow::Error::new(err),
                        "Bedrock stream read error",
                        RetryableLlmError::fast("Bedrock stream read error"),
                    )
                })?;
                decoder.push(&chunk);

                let mut made_progress = false;
                while let Some(frame) = decoder.next_frame()? {
                    made_progress = true;
                    let Some(event) = decode_anthropic_event(&frame)? else {
                        continue;
                    };
                    match event {
                        AnthropicStreamEvent::MessageStart { message } => {
                            if let Some(usage) = message.usage {
                                merge_stream_usage(&mut outcome.usage, usage.into_usage());
                            }
                        }
                        AnthropicStreamEvent::ContentBlockStart { index, content_block } => {
                            if let AnthropicStreamBlockStart::ToolUse { id, name } = content_block {
                                tool_blocks.insert(
                                    index,
                                    PartialToolUse { id, name, arguments: String::new() },
                                );
                            }
                        }
                        AnthropicStreamEvent::ContentBlockDelta { index, delta } => match delta {
                            AnthropicStreamDelta::Text { text } => {
                                on_token(&text);
                                outcome.text.push_str(&text);
                            }
                            AnthropicStreamDelta::Thinking { thinking } => {
                                on_thought(&thinking);
                                outcome.thoughts.push_str(&thinking);
                            }
                            AnthropicStreamDelta::InputJson { partial_json } => {
                                if let Some(block) = tool_blocks.get_mut(&index) {
                                    block.arguments.push_str(&partial_json);
                                }
                            }
                            AnthropicStreamDelta::Other => {}
                        },
                        AnthropicStreamEvent::MessageDelta { delta, usage } => {
                            if let Some(stop_reason) = delta.stop_reason {
                                outcome.stop_reason = Some(stop_reason);
                            }
                            if let Some(usage) = usage {
                                merge_stream_usage(&mut outcome.usage, usage.into_usage());
                            }
                        }
                        AnthropicStreamEvent::MessageStop => {
                            completed = true;
                            break;
                        }
                        AnthropicStreamEvent::Error { error } => {
                            let kind = error
                                .get("type")
                                .and_then(|t| t.as_str())
                                .unwrap_or("unknown")
                                .to_string();
                            failure = Some(stream_failure_error(&kind, &error.to_string()));
                            completed = true;
                            break;
                        }
                        AnthropicStreamEvent::Other => {}
                    }
                }

                if made_progress {
                    saw_progress = true;
                    deadline = tokio::time::Instant::now() + idle.inter_chunk;
                }
                if completed {
                    break;
                }
            }
        }
    }

    if let Some(err) = failure {
        return Err(err);
    }
    // Cancellation is a caller decision, not a provider failure: hand back
    // whatever streamed before the abort rather than reporting a truncated
    // stream. Half-built tool calls are dropped -- their arguments are
    // partial JSON and there is nothing to execute them for.
    if cancel.is_cancelled() {
        return Ok(outcome);
    }
    if !completed {
        return Err(anyhow::Error::new(IncompleteStreamError::new(
            "Bedrock eventstream",
            "message_stop",
        )));
    }
    for (_, block) in std::mem::take(&mut tool_blocks) {
        let arguments = if block.arguments.trim().is_empty() {
            // Anthropic sends no `input_json_delta` at all for a no-argument
            // tool call.
            "{}".to_string()
        } else {
            block.arguments
        };
        let arguments = crate::tool_arguments::normalize_streamed_tool_arguments(
            &block.id,
            &block.name,
            arguments,
            "Bedrock eventstream",
        )?;
        outcome.calls.push(ToolCall {
            id: block.id,
            r#type: "function".to_string(),
            function: FunctionCall {
                name: block.name,
                arguments,
            },
        });
    }
    Ok(outcome)
}

/// Turns one eventstream frame into an Anthropic streaming event, or `None`
/// for frames that carry no model output (unknown message/event types).
fn decode_anthropic_event(frame: &EventStreamFrame) -> Result<Option<AnthropicStreamEvent>> {
    match frame.header(":message-type").unwrap_or("event") {
        "event" => {}
        message_type @ ("exception" | "error") => {
            let kind = frame
                .header(":exception-type")
                .or_else(|| frame.header(":error-code"))
                .unwrap_or(message_type);
            let body = String::from_utf8_lossy(&frame.payload).into_owned();
            return Err(stream_failure_error(kind, &body));
        }
        other => {
            tracing::debug!(
                message_type = other,
                "ignoring unknown Bedrock eventstream message type"
            );
            return Ok(None);
        }
    }
    if frame
        .header(":event-type")
        .is_some_and(|kind| kind != "chunk")
    {
        return Ok(None);
    }

    #[derive(Deserialize)]
    struct ChunkEnvelope {
        bytes: String,
    }

    use base64::Engine as _;
    let envelope: ChunkEnvelope = serde_json::from_slice(&frame.payload)
        .context("parse Bedrock eventstream chunk envelope")?;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(envelope.bytes.as_bytes())
        .context("base64-decode Bedrock eventstream chunk")?;
    let event = serde_json::from_slice(&decoded).with_context(|| {
        format!(
            "parse Anthropic stream event: {}",
            String::from_utf8_lossy(&decoded)
        )
    })?;
    Ok(Some(event))
}

/// AWS caps an eventstream message at 16 MiB. A larger length prefix means we
/// lost frame sync, so fail instead of reserving on a bogus number.
const EVENTSTREAM_MAX_FRAME_LEN: usize = 16 * 1024 * 1024;
/// `total_length` (4) + `headers_length` (4) + prelude CRC (4).
const EVENTSTREAM_PRELUDE_LEN: usize = 12;
/// Trailing message CRC.
const EVENTSTREAM_TRAILER_LEN: usize = 4;

#[derive(Debug)]
struct EventStreamFrame {
    /// String-valued headers only; the binary/numeric header types the spec
    /// defines never appear on this API's frames.
    headers: Vec<(String, String)>,
    payload: Vec<u8>,
}

impl EventStreamFrame {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }
}

/// Incremental decoder for AWS `application/vnd.amazon.eventstream` framing:
///
/// ```text
/// [total len u32][headers len u32][prelude CRC u32][headers][payload][CRC u32]
/// ```
///
/// Frames are split arbitrarily across HTTP chunks, so bytes accumulate here
/// until a whole frame is available. Both CRCs are ignored: the transport is
/// TLS, which already detects corruption, and verifying them would mean
/// carrying a CRC32 implementation for no additional protection.
#[derive(Default)]
struct EventStreamDecoder {
    buf: Vec<u8>,
}

impl EventStreamDecoder {
    fn push(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
    }

    fn next_frame(&mut self) -> Result<Option<EventStreamFrame>> {
        if self.buf.len() < EVENTSTREAM_PRELUDE_LEN {
            return Ok(None);
        }
        let total_len =
            u32::from_be_bytes(self.buf[0..4].try_into().expect("4-byte slice")) as usize;
        let headers_len =
            u32::from_be_bytes(self.buf[4..8].try_into().expect("4-byte slice")) as usize;
        let overhead = EVENTSTREAM_PRELUDE_LEN + EVENTSTREAM_TRAILER_LEN;
        if total_len < overhead
            || total_len > EVENTSTREAM_MAX_FRAME_LEN
            || headers_len > total_len - overhead
        {
            anyhow::bail!(
                "malformed Bedrock eventstream frame (total {total_len} bytes, headers {headers_len} bytes)"
            );
        }
        if self.buf.len() < total_len {
            return Ok(None);
        }
        let frame: Vec<u8> = self.buf.drain(..total_len).collect();
        let headers_end = EVENTSTREAM_PRELUDE_LEN + headers_len;
        Ok(Some(EventStreamFrame {
            headers: parse_eventstream_headers(&frame[EVENTSTREAM_PRELUDE_LEN..headers_end])?,
            payload: frame[headers_end..total_len - EVENTSTREAM_TRAILER_LEN].to_vec(),
        }))
    }
}

fn parse_eventstream_headers(mut region: &[u8]) -> Result<Vec<(String, String)>> {
    const STRING_VALUE_TYPE: u8 = 7;
    let mut headers = Vec::new();
    while let Some((&name_len, rest)) = region.split_first() {
        let name_len = usize::from(name_len);
        if rest.len() < name_len + 1 {
            anyhow::bail!("truncated Bedrock eventstream header name");
        }
        let name = String::from_utf8_lossy(&rest[..name_len]).into_owned();
        let value_type = rest[name_len];
        region = &rest[name_len + 1..];

        // Value widths per the eventstream spec. Only strings are retained,
        // but every type has to be skipped by the right width to stay in sync.
        let value_len = match value_type {
            0 | 1 => 0,
            2 => 1,
            3 => 2,
            4 => 4,
            5 | 8 => 8,
            9 => 16,
            6 | STRING_VALUE_TYPE => {
                if region.len() < 2 {
                    anyhow::bail!("truncated Bedrock eventstream header length");
                }
                let len = usize::from(u16::from_be_bytes([region[0], region[1]]));
                region = &region[2..];
                len
            }
            other => anyhow::bail!("unsupported Bedrock eventstream header value type {other}"),
        };
        if region.len() < value_len {
            anyhow::bail!("truncated Bedrock eventstream header value");
        }
        if value_type == STRING_VALUE_TYPE {
            headers.push((
                name,
                String::from_utf8_lossy(&region[..value_len]).into_owned(),
            ));
        }
        region = &region[value_len..];
    }
    Ok(headers)
}

fn convert_messages(
    messages: Vec<ChatMessage>,
    enable_cache: bool,
) -> Result<(Vec<BedrockTextBlock>, Vec<BedrockMessage>)> {
    let mut system_texts = Vec::new();
    let mut converted = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                reject_images_for_role(&msg, "system")?;
                let content = msg.content_text();
                if !content.trim().is_empty() {
                    system_texts.push(content);
                }
            }
            "user" => {
                let content = convert_content_parts(msg.content)?;
                if !content.is_empty() {
                    converted.push(BedrockMessage {
                        role: "user".to_string(),
                        content,
                    });
                }
            }
            "assistant" => {
                let mut content = Vec::new();
                reject_images_for_role(&msg, "assistant")?;
                let text = msg.content_text();
                if !text.is_empty() {
                    content.push(BedrockContentOut::Text {
                        text,
                        cache_control: None,
                    });
                }
                if let Some(calls) = msg.tool_calls {
                    for call in calls {
                        content.push(BedrockContentOut::ToolUse {
                            id: call.id,
                            name: call.function.name,
                            input: parse_tool_arguments(&call.function.arguments)?,
                        });
                    }
                }
                if !content.is_empty() {
                    converted.push(BedrockMessage {
                        role: "assistant".to_string(),
                        content,
                    });
                }
            }
            "tool" => {
                let tool_output = msg.content_text();
                let tool_use_id = msg
                    .tool_call_id
                    .context("tool result missing tool_call_id")?;
                converted.push(BedrockMessage {
                    role: "user".to_string(),
                    content: vec![BedrockContentOut::ToolResult {
                        tool_use_id,
                        content: tool_output,
                        cache_control: None,
                    }],
                });
            }
            _ => {}
        }
    }

    // Build system text blocks and attach cache_control to the last one
    // when caching is enabled.
    let mut system_blocks: Vec<BedrockTextBlock> = system_texts
        .into_iter()
        .map(|text| BedrockTextBlock {
            block_type: "text",
            text,
            cache_control: None,
        })
        .collect();
    if enable_cache && let Some(last) = system_blocks.last_mut() {
        last.cache_control = Some(CACHE_CONTROL);
    }

    // When caching is enabled, attach cache_control to the last content
    // block of the last user message (text or tool_result).
    if enable_cache {
        for msg in converted.iter_mut().rev() {
            if msg.role == "user" {
                // The last user block before an assistant turn is almost always
                // a tool_result in agentic loops; caching only the Text case
                // left the entire growing message history uncached every turn.
                if let Some(
                    BedrockContentOut::Text { cache_control, .. }
                    | BedrockContentOut::ToolResult { cache_control, .. },
                ) = msg.content.last_mut()
                {
                    *cache_control = Some(CACHE_CONTROL);
                }
                break;
            }
        }
    }

    Ok((system_blocks, converted))
}

fn convert_content_parts(parts: Vec<ChatContentPart>) -> Result<Vec<BedrockContentOut>> {
    parts
        .into_iter()
        .map(|part| match part {
            ChatContentPart::Text { text } => Ok(BedrockContentOut::Text {
                text,
                cache_control: None,
            }),
            ChatContentPart::Image { image_url } => Ok(BedrockContentOut::Image {
                source: parse_bedrock_image_source(&image_url)?,
            }),
        })
        .collect()
}

fn reject_images_for_role(msg: &ChatMessage, role: &str) -> Result<()> {
    if msg
        .content
        .iter()
        .any(|part| matches!(part, ChatContentPart::Image { .. }))
    {
        anyhow::bail!("Bedrock {role} messages do not support image content");
    }
    Ok(())
}

fn parse_bedrock_image_source(image_url: &str) -> Result<BedrockImageSource> {
    let data_url = image_url.strip_prefix("data:").context(
        "Bedrock image inputs must be inline data URLs; remote image URLs are not supported",
    )?;
    let (metadata, data) = data_url
        .split_once(',')
        .context("Bedrock image data URL is missing a base64 payload")?;
    let mut metadata_parts = metadata.split(';');
    let media_type = metadata_parts
        .next()
        .filter(|media_type| media_type.starts_with("image/"))
        .context("Bedrock image data URL must include an image/* media type")?;
    if !metadata.split(';').any(|part| part == "base64") {
        anyhow::bail!("Bedrock image data URL must be base64 encoded");
    }
    if data.is_empty() {
        anyhow::bail!("Bedrock image data URL has an empty payload");
    }

    Ok(BedrockImageSource {
        source_type: "base64",
        media_type: media_type.to_string(),
        data: data.to_string(),
    })
}

fn convert_tools(tools: Vec<ToolDefinition>, enable_cache: bool) -> Vec<BedrockTool> {
    let mut converted: Vec<BedrockTool> = tools
        .into_iter()
        .map(|tool| BedrockTool {
            name: tool.function.name,
            description: tool.function.description,
            input_schema: bedrock_tool_input_schema(tool.function.parameters),
            cache_control: None,
        })
        .collect();
    if enable_cache && let Some(last) = converted.last_mut() {
        last.cache_control = Some(CACHE_CONTROL);
    }
    converted
}

/// Bedrock's native Anthropic tools endpoint rejects schemas whose root object
/// is a JSON Schema combiner (`oneOf`, `anyOf`, or `allOf`). Structured-output
/// JSON schemas and strict tool use have their own broader compatibility rules;
/// this adapter is intentionally narrower and only removes the top-level tool
/// `input_schema` combiner rejected by Bedrock request validation. MCP servers
/// are allowed to advertise such schemas, so normalize only the top-level
/// wrapper before sending tools to Bedrock while preserving the usable object
/// fields the model needs to call the tool.
fn bedrock_tool_input_schema(schema: serde_json::Value) -> serde_json::Value {
    let Some(map) = schema.as_object() else {
        return schema;
    };
    let Some((combiner_key, variants)) = top_level_combiner_variants(map) else {
        return schema;
    };

    let merged = merge_top_level_combiner_schema(map, combiner_key, variants);
    tracing::debug!(
        combiner = combiner_key,
        "flattened top-level JSON Schema combiner for Bedrock tool input schema"
    );
    serde_json::Value::Object(merged)
}

fn top_level_combiner_variants(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Option<(&'static str, &[serde_json::Value])> {
    for key in ["oneOf", "anyOf", "allOf"] {
        if let Some(values) = map.get(key).and_then(serde_json::Value::as_array) {
            return Some((key, values.as_slice()));
        }
    }
    None
}

fn merge_top_level_combiner_schema(
    map: &serde_json::Map<String, serde_json::Value>,
    combiner_key: &str,
    variants: &[serde_json::Value],
) -> serde_json::Map<String, serde_json::Value> {
    let mut merged = map.clone();
    merged.remove("oneOf");
    merged.remove("anyOf");
    merged.remove("allOf");
    merged
        .entry("type".to_string())
        .or_insert_with(|| serde_json::Value::String("object".to_string()));

    let mut properties = merged
        .get("properties")
        .and_then(serde_json::Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut required: Option<HashSet<String>> = merged
        .get("required")
        .and_then(serde_json::Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect()
        });

    for variant in variants {
        let Some(variant) = variant.as_object() else {
            continue;
        };
        if let Some(variant_properties) = variant
            .get("properties")
            .and_then(serde_json::Value::as_object)
        {
            for (name, schema) in variant_properties {
                properties
                    .entry(name.clone())
                    .or_insert_with(|| schema.clone());
            }
        }
        if combiner_key == "allOf"
            && let Some(variant_required) = variant
                .get("required")
                .and_then(serde_json::Value::as_array)
        {
            required.get_or_insert_with(HashSet::new).extend(
                variant_required
                    .iter()
                    .filter_map(|value| value.as_str().map(str::to_string)),
            );
        }
    }

    if !properties.is_empty() {
        merged.insert(
            "properties".to_string(),
            serde_json::Value::Object(properties),
        );
    }
    if let Some(required) = required.filter(|required| !required.is_empty()) {
        let mut required: Vec<serde_json::Value> = required
            .into_iter()
            .map(serde_json::Value::String)
            .collect();
        required.sort_by(|a, b| a.as_str().cmp(&b.as_str()));
        merged.insert("required".to_string(), serde_json::Value::Array(required));
    }

    merged
}

fn parse_tool_arguments(raw: &str) -> Result<serde_json::Value> {
    match crate::tool_arguments::normalize_tool_arguments(raw) {
        Ok(normalized) => {
            if normalized.repaired {
                tracing::warn!("repaired malformed Bedrock tool-call arguments");
            }
            Ok(normalized.value)
        }
        Err(err) => Err(anyhow::Error::new(err))
            .with_context(|| format!("parse tool arguments as JSON: {raw}")),
    }
}

async fn discover_model_metadata_with_http(
    http: &reqwest::Client,
    catalog_url: &str,
    bearer_token: &str,
    default_model: &str,
) -> Result<Vec<ModelMetadata>> {
    let response = http
        .get(catalog_url)
        .bearer_auth(bearer_token)
        .send()
        .await
        .context("failed to send Bedrock model discovery request")?;
    let status = response.status();
    let body = response
        .text()
        .await
        .context("failed to read Bedrock model discovery response body")?;
    if !status.is_success() {
        anyhow::bail!("Bedrock model discovery failed (HTTP {status}): {body}");
    }

    let parsed: BedrockListModelsResponse =
        serde_json::from_str(&body).context("parse Bedrock model discovery response")?;
    let mut models: Vec<ModelMetadata> = parsed
        .model_summaries
        .into_iter()
        .filter(is_supported_bedrock_model)
        .map(BedrockFoundationModelSummary::into_model_metadata)
        .collect();

    if models.is_empty() {
        tracing::warn!(
            "Bedrock model discovery returned no compatible models; falling back to default model {}",
            default_model
        );
        models.push(ModelMetadata::id_only(default_model.to_string()));
        return Ok(models);
    }

    models.sort_by(|a, b| {
        let a_default = a.id == default_model;
        let b_default = b.id == default_model;
        b_default.cmp(&a_default).then_with(|| a.id.cmp(&b.id))
    });
    models.dedup_by(|a, b| a.id == b.id);
    Ok(models)
}

fn is_supported_bedrock_model(summary: &BedrockFoundationModelSummary) -> bool {
    summary.response_streaming_supported.unwrap_or(false)
        && summary.model_id.contains("anthropic")
        && summary
            .output_modalities
            .iter()
            .any(|modality| modality == "TEXT")
        && summary
            .input_modalities
            .iter()
            .any(|modality| modality == "TEXT")
}

fn percent_encode_path_segment(input: &str) -> String {
    let mut out = String::new();
    for b in input.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(char::from(b));
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[derive(Debug, Deserialize)]
struct BedrockListModelsResponse {
    #[serde(rename = "modelSummaries")]
    model_summaries: Vec<BedrockFoundationModelSummary>,
}

#[derive(Debug, Deserialize)]
struct BedrockFoundationModelSummary {
    #[serde(rename = "modelId")]
    model_id: String,
    #[serde(rename = "inputModalities")]
    input_modalities: Vec<String>,
    #[serde(rename = "outputModalities")]
    output_modalities: Vec<String>,
    // Some newer/preview models return null for this field; treat null as
    // false (i.e. exclude them from the supported set) so the response
    // can still be parsed for the models that do report a boolean.
    #[serde(rename = "responseStreamingSupported", default)]
    response_streaming_supported: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct BedrockListInferenceProfilesResponse {
    #[serde(rename = "inferenceProfileSummaries", default)]
    inference_profile_summaries: Vec<BedrockInferenceProfileSummary>,
}

#[derive(Debug, Deserialize, Clone)]
struct BedrockInferenceProfileSummary {
    #[serde(rename = "inferenceProfileId")]
    inference_profile_id: String,
    #[serde(default)]
    models: Vec<BedrockInferenceProfileModel>,
}

#[derive(Debug, Deserialize, Clone)]
struct BedrockInferenceProfileModel {
    #[serde(rename = "modelArn")]
    model_arn: String,
}

impl BedrockFoundationModelSummary {
    fn into_model_metadata(self) -> ModelMetadata {
        let supports_images = self
            .input_modalities
            .iter()
            .any(|modality| modality == "IMAGE");
        let reasoning_spec = bedrock_reasoning_spec_or_fallback(&self.model_id);
        ModelMetadata {
            id: self.model_id,
            default_reasoning_level: reasoning_spec.default_level.map(str::to_string),
            supported_reasoning_levels: reasoning_spec.to_presets(),
            service_tiers: Vec::new(),
            supports_images: Some(supports_images),
            context_length: None,
            pricing: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::llm_client::{ChatContentPart, FunctionDef};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;
    use tokio_util::sync::CancellationToken;
    use wiremock::matchers::{header, method, path, path_regex};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    const EVENTSTREAM_CONTENT_TYPE: &str = "application/vnd.amazon.eventstream";

    /// One content block in a mocked streaming response.
    enum StreamBlock<'a> {
        Text(&'a str),
        Thinking(&'a str),
        ToolUse {
            id: &'a str,
            name: &'a str,
            /// Sent as a single `input_json_delta`; use `""` for a tool call
            /// the model emits with no arguments at all.
            input_json: &'a str,
        },
    }

    /// Expands content blocks into the Anthropic event sequence the streaming
    /// endpoint emits: usage split across `message_start`/`message_delta`,
    /// one start/delta/stop trio per block, then `message_stop`.
    fn anthropic_stream_events(
        blocks: &[StreamBlock<'_>],
        stop_reason: Option<&str>,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Vec<serde_json::Value> {
        let mut events = vec![serde_json::json!({
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": input_tokens, "output_tokens": 1}
            }
        })];
        for (index, block) in blocks.iter().enumerate() {
            let (start, delta) = match block {
                StreamBlock::Text(text) => (
                    serde_json::json!({"type": "text", "text": ""}),
                    Some(serde_json::json!({"type": "text_delta", "text": text})),
                ),
                StreamBlock::Thinking(thinking) => (
                    serde_json::json!({"type": "thinking", "thinking": ""}),
                    Some(serde_json::json!({"type": "thinking_delta", "thinking": thinking})),
                ),
                StreamBlock::ToolUse {
                    id,
                    name,
                    input_json,
                } => (
                    serde_json::json!({"type": "tool_use", "id": id, "name": name, "input": {}}),
                    (!input_json.is_empty()).then(|| {
                        serde_json::json!({"type": "input_json_delta", "partial_json": input_json})
                    }),
                ),
            };
            events.push(serde_json::json!({
                "type": "content_block_start",
                "index": index,
                "content_block": start
            }));
            if let Some(delta) = delta {
                events.push(serde_json::json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": delta
                }));
            }
            events.push(serde_json::json!({"type": "content_block_stop", "index": index}));
        }
        events.push(serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
            "usage": {"output_tokens": output_tokens}
        }));
        events.push(serde_json::json!({"type": "message_stop"}));
        events
    }

    /// Frames arbitrary events the way `invoke-with-response-stream` does:
    /// a 12-byte prelude, string headers, a base64 `{"bytes":...}` payload,
    /// and a trailing CRC slot. Both CRC slots are zero -- the decoder does
    /// not verify them.
    fn eventstream_body(events: &[serde_json::Value]) -> Vec<u8> {
        use base64::Engine as _;
        let mut body = Vec::new();
        for event in events {
            let payload = serde_json::json!({
                "bytes": base64::engine::general_purpose::STANDARD.encode(event.to_string())
            })
            .to_string();
            body.extend_from_slice(&eventstream_frame(
                &[
                    (":message-type", "event"),
                    (":event-type", "chunk"),
                    (":content-type", "application/json"),
                ],
                payload.as_bytes(),
            ));
        }
        body
    }

    fn eventstream_frame(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
        let mut headers_bytes = Vec::new();
        for (name, value) in headers {
            headers_bytes.push(u8::try_from(name.len()).expect("header name fits in a byte"));
            headers_bytes.extend_from_slice(name.as_bytes());
            headers_bytes.push(7);
            headers_bytes.extend_from_slice(
                &u16::try_from(value.len())
                    .expect("header value fits in u16")
                    .to_be_bytes(),
            );
            headers_bytes.extend_from_slice(value.as_bytes());
        }
        let total =
            EVENTSTREAM_PRELUDE_LEN + headers_bytes.len() + payload.len() + EVENTSTREAM_TRAILER_LEN;
        let mut frame = Vec::with_capacity(total);
        frame.extend_from_slice(
            &u32::try_from(total)
                .expect("frame fits in u32")
                .to_be_bytes(),
        );
        frame.extend_from_slice(
            &u32::try_from(headers_bytes.len())
                .expect("headers fit in u32")
                .to_be_bytes(),
        );
        frame.extend_from_slice(&0u32.to_be_bytes());
        frame.extend_from_slice(&headers_bytes);
        frame.extend_from_slice(payload);
        frame.extend_from_slice(&0u32.to_be_bytes());
        frame
    }

    /// Wraps a mocked streaming body in the response shape Bedrock returns.
    fn eventstream_response(events: &[serde_json::Value]) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_raw(eventstream_body(events), EVENTSTREAM_CONTENT_TYPE)
    }

    /// The common case: one text block, no thinking, no tool calls.
    fn text_stream_response(text: &str, input_tokens: u64, output_tokens: u64) -> ResponseTemplate {
        eventstream_response(&anthropic_stream_events(
            &[StreamBlock::Text(text)],
            Some("end_turn"),
            input_tokens,
            output_tokens,
        ))
    }

    #[test]
    fn percent_encode_model_id() {
        assert_eq!(
            percent_encode_path_segment("us.anthropic.claude-sonnet-4-6"),
            "us.anthropic.claude-sonnet-4-6"
        );
        assert_eq!(percent_encode_path_segment("a/b c"), "a%2Fb%20c");
    }

    #[test]
    fn reasoning_specs_match_bedrock_model_families() {
        let efforts = |model: &str| {
            bedrock_reasoning_spec_for_model(model)
                .expect("reasoning spec")
                .presets
                .iter()
                .map(|(effort, _)| *effort)
                .collect::<Vec<_>>()
        };

        assert_eq!(
            efforts("openai.gpt-5.5"),
            ["none", "low", "medium", "high", "xhigh"]
        );
        assert_eq!(efforts("openai.gpt-oss-120b"), ["low", "medium", "high"]);
        assert_eq!(
            efforts("openai.gpt-oss-120b-1:0"),
            ["low", "medium", "high"]
        );
        assert_eq!(
            efforts("global.anthropic.claude-opus-4-8"),
            ["low", "medium", "high", "xhigh", "max"]
        );
        assert_eq!(
            efforts("us.anthropic.claude-sonnet-4-6"),
            ["low", "medium", "high", "max"]
        );
        assert_eq!(
            efforts("anthropic.claude-opus-4-5"),
            ["low", "medium", "high"]
        );
        assert_eq!(
            efforts("global.anthropic.claude-sonnet-4-5"),
            ["low", "medium", "high"]
        );
        assert_eq!(
            efforts("anthropic.claude-haiku-4-5"),
            ["low", "medium", "high"]
        );
        assert_eq!(
            efforts("anthropic.claude-opus-4-1"),
            ["low", "medium", "high"]
        );
        assert_eq!(
            efforts("anthropic.claude-sonnet-4-20250514-v1:0"),
            ["low", "medium", "high"]
        );

        assert!(bedrock_reasoning_spec_for_model("anthropic.claude-3-5-sonnet").is_none());
        assert!(bedrock_reasoning_spec_for_model("openai.gpt-oss-safeguard-120b").is_none());
        assert!(bedrock_reasoning_spec_for_model("amazon.titan-text").is_none());
    }

    #[test]
    fn thinking_budget_maps_known_efforts_only() {
        assert_eq!(thinking_budget_for_effort("low"), Some(2_048));
        assert_eq!(thinking_budget_for_effort("medium"), Some(4_096));
        assert_eq!(thinking_budget_for_effort("high"), Some(8_192));
        assert_eq!(thinking_budget_for_effort(" high "), Some(8_192));
        assert_eq!(thinking_budget_for_effort("none"), None);
        assert_eq!(thinking_budget_for_effort(""), None);
    }

    #[test]
    fn thinking_max_tokens_exceeds_budget() {
        for effort in ["low", "medium", "high"] {
            let budget = thinking_budget_for_effort(effort).expect("known effort");
            assert!(thinking_max_tokens(budget) > budget);
        }
    }

    #[test]
    fn foundation_models_advertise_family_reasoning_presets() {
        let summary = BedrockFoundationModelSummary {
            model_id: "anthropic.claude-sonnet-4-6".to_string(),
            input_modalities: vec!["TEXT".to_string()],
            output_modalities: vec!["TEXT".to_string()],
            response_streaming_supported: Some(true),
        };
        let meta = summary.into_model_metadata();
        assert_eq!(meta.default_reasoning_level.as_deref(), Some("high"));
        let efforts: Vec<_> = meta
            .supported_reasoning_levels
            .iter()
            .map(|p| p.effort.as_str())
            .collect();
        assert_eq!(efforts, ["low", "medium", "high", "max"]);

        let plain = BedrockFoundationModelSummary {
            model_id: "amazon.titan-text".to_string(),
            input_modalities: vec!["TEXT".to_string()],
            output_modalities: vec!["TEXT".to_string()],
            response_streaming_supported: Some(true),
        };
        let plain_meta = plain.into_model_metadata();
        assert_eq!(
            plain_meta.default_reasoning_level.as_deref(),
            Some("medium")
        );
        let plain_efforts: Vec<_> = plain_meta
            .supported_reasoning_levels
            .iter()
            .map(|p| p.effort.as_str())
            .collect();
        assert_eq!(plain_efforts, ["low", "medium", "high"]);

        let manual = BedrockFoundationModelSummary {
            model_id: "anthropic.claude-sonnet-4-5".to_string(),
            input_modalities: vec!["TEXT".to_string()],
            output_modalities: vec!["TEXT".to_string()],
            response_streaming_supported: Some(true),
        };
        let manual_meta = manual.into_model_metadata();
        assert_eq!(manual_meta.default_reasoning_level, None);
        let manual_efforts: Vec<_> = manual_meta
            .supported_reasoning_levels
            .iter()
            .map(|p| p.effort.as_str())
            .collect();
        assert_eq!(manual_efforts, ["low", "medium", "high"]);
    }

    #[test]
    fn enrichment_attaches_family_specific_presets() {
        let mut models = vec![
            ModelMetadata {
                id: "global.anthropic.claude-opus-4-8".to_string(),
                default_reasoning_level: None,
                supported_reasoning_levels: Vec::new(),
                service_tiers: Vec::new(),
                supports_images: Some(true),
                context_length: Some(200_000),
                pricing: None,
            },
            ModelMetadata {
                id: "us.anthropic.claude-3-5-sonnet".to_string(),
                default_reasoning_level: None,
                supported_reasoning_levels: Vec::new(),
                service_tiers: Vec::new(),
                supports_images: Some(true),
                context_length: Some(200_000),
                pricing: None,
            },
            ModelMetadata {
                id: "openai.gpt-5.4".to_string(),
                default_reasoning_level: None,
                supported_reasoning_levels: Vec::new(),
                service_tiers: Vec::new(),
                supports_images: Some(true),
                context_length: Some(200_000),
                pricing: None,
            },
            ModelMetadata {
                id: "openai.gpt-oss-20b".to_string(),
                default_reasoning_level: None,
                supported_reasoning_levels: Vec::new(),
                service_tiers: Vec::new(),
                supports_images: Some(false),
                context_length: Some(128_000),
                pricing: None,
            },
            ModelMetadata {
                id: "global.anthropic.claude-sonnet-4-5".to_string(),
                default_reasoning_level: Some("medium".to_string()),
                supported_reasoning_levels: Vec::new(),
                service_tiers: Vec::new(),
                supports_images: Some(true),
                context_length: Some(200_000),
                pricing: None,
            },
        ];
        apply_bedrock_reasoning_presets(&mut models);

        let by_id = |id: &str| models.iter().find(|model| model.id == id).unwrap();
        assert_eq!(
            by_id("global.anthropic.claude-opus-4-8")
                .supported_reasoning_levels
                .iter()
                .map(|p| p.effort.as_str())
                .collect::<Vec<_>>(),
            ["low", "medium", "high", "xhigh", "max"]
        );
        assert_eq!(
            by_id("us.anthropic.claude-3-5-sonnet")
                .supported_reasoning_levels
                .iter()
                .map(|p| p.effort.as_str())
                .collect::<Vec<_>>(),
            ["low", "medium", "high"]
        );
        assert_eq!(
            by_id("us.anthropic.claude-3-5-sonnet")
                .default_reasoning_level
                .as_deref(),
            Some("medium")
        );
        assert_eq!(
            by_id("openai.gpt-5.4")
                .supported_reasoning_levels
                .iter()
                .map(|p| p.effort.as_str())
                .collect::<Vec<_>>(),
            ["none", "low", "medium", "high", "xhigh"]
        );
        assert_eq!(
            by_id("openai.gpt-oss-20b")
                .supported_reasoning_levels
                .iter()
                .map(|p| p.effort.as_str())
                .collect::<Vec<_>>(),
            ["low", "medium", "high"]
        );
        assert_eq!(
            by_id("global.anthropic.claude-sonnet-4-5")
                .supported_reasoning_levels
                .iter()
                .map(|p| p.effort.as_str())
                .collect::<Vec<_>>(),
            ["low", "medium", "high"]
        );
        assert_eq!(
            by_id("global.anthropic.claude-sonnet-4-5").default_reasoning_level,
            None
        );
    }

    #[test]
    fn enrichment_overrides_stale_presets_with_model_card_values() {
        let mut models = vec![ModelMetadata {
            id: "us.anthropic.claude-sonnet-4-6".to_string(),
            default_reasoning_level: Some("medium".to_string()),
            supported_reasoning_levels: vec![ReasoningLevelPreset {
                effort: "high".to_string(),
                description: "preset".to_string(),
            }],
            service_tiers: Vec::new(),
            supports_images: Some(true),
            context_length: Some(200_000),
            pricing: None,
        }];
        apply_bedrock_reasoning_presets(&mut models);
        assert_eq!(
            models[0]
                .supported_reasoning_levels
                .iter()
                .map(|p| p.effort.as_str())
                .collect::<Vec<_>>(),
            ["low", "medium", "high", "max"]
        );
        assert_eq!(models[0].default_reasoning_level.as_deref(), Some("high"));
    }

    #[test]
    fn default_bedrock_runtime_url_includes_region_and_aws_host() {
        let client = BedrockClient::new(
            "token".to_string(),
            "us-east-1".to_string(),
            "us.anthropic.claude-sonnet-4-6".to_string(),
        );

        assert_eq!(
            client.invoke_stream_url("us.anthropic.claude-sonnet-4-6"),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4-6/invoke-with-response-stream"
        );
    }

    #[test]
    fn full_bedrock_runtime_url_is_not_region_expanded_twice() {
        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-1".to_string(),
            "us.anthropic.claude-sonnet-4-6".to_string(),
            "https://bedrock-runtime.us-east-1.amazonaws.com".to_string(),
            "https://bedrock-mantle.us-east-1.api.aws/v1".to_string(),
            "https://bedrock".to_string(),
        );

        assert_eq!(
            client.invoke_stream_url("a/b c"),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/a%2Fb%20c/invoke-with-response-stream"
        );
    }

    #[test]
    fn converts_tool_round_trip_messages() {
        let calls = vec![ToolCall {
            id: "toolu_1".to_string(),
            r#type: "function".to_string(),
            function: FunctionCall {
                name: "readFile".to_string(),
                arguments: r#"{"path":"src/main.rs"}"#.to_string(),
            },
        }];
        let (_, messages) = convert_messages(
            vec![
                ChatMessage::assistant_tool_calls(calls),
                ChatMessage::tool_result("toolu_1", "readFile", "contents"),
            ],
            false,
        )
        .expect("convert");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "assistant");
        assert_eq!(messages[1].role, "user");
    }

    #[test]
    fn converts_inline_image_user_parts() {
        let (_, messages) = convert_messages(
            vec![ChatMessage::user_parts(vec![
                ChatContentPart::text("What is in this image?"),
                ChatContentPart::image_data("aW1hZ2U=", "image/png"),
            ])],
            false,
        )
        .expect("convert");

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content.len(), 2);
        match &messages[0].content[1] {
            BedrockContentOut::Image { source } => {
                assert_eq!(source.source_type, "base64");
                assert_eq!(source.media_type, "image/png");
                assert_eq!(source.data, "aW1hZ2U=");
            }
            other => panic!("expected Image, got {other:?}"),
        }
    }

    #[test]
    fn rejects_remote_image_urls() {
        let err = convert_messages(
            vec![ChatMessage::user_parts(vec![ChatContentPart::image_url(
                "https://example.com/image.png",
            )])],
            false,
        )
        .expect_err("remote image URLs should not be converted for Bedrock");

        assert!(
            format!("{err:#}").contains("remote image URLs are not supported"),
            "got {err:#}"
        );
    }

    #[test]
    fn cache_control_on_claude_requests() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there"),
            ChatMessage::user("What's the weather?"),
        ];
        let (system_blocks, converted) =
            convert_messages(messages, true).expect("convert with cache");

        // Last system block has cache_control
        assert_eq!(system_blocks.len(), 1);
        assert!(system_blocks[0].cache_control.is_some());

        // Last user message's content has cache_control
        let last_user = converted
            .iter()
            .rfind(|m| m.role == "user")
            .expect("last user msg");
        match last_user.content.last().expect("content") {
            BedrockContentOut::Text { cache_control, .. } => {
                assert!(cache_control.is_some());
            }
            other => panic!("expected Text, got {other:?}"),
        }

        // Earlier user message does NOT have cache_control
        let first_user = converted
            .iter()
            .find(|m| m.role == "user")
            .expect("first user msg");
        match first_user.content.last().expect("content") {
            BedrockContentOut::Text { cache_control, .. } => {
                assert!(cache_control.is_none());
            }
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn cache_control_on_trailing_tool_result() {
        // The common agentic-loop shape: the last user-role message is a
        // tool_result, not text. The breakpoint must still land on it, or the
        // entire message-history prefix is billed uncached every turn.
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("What's the weather?"),
            ChatMessage::assistant("Let me check."),
            ChatMessage::tool_result("call_1", "get_weather", "72F and sunny"),
        ];
        let (_system_blocks, converted) =
            convert_messages(messages, true).expect("convert with cache");

        let last_user = converted
            .iter()
            .rfind(|m| m.role == "user")
            .expect("last user msg");
        match last_user.content.last().expect("content") {
            BedrockContentOut::ToolResult { cache_control, .. } => {
                assert!(
                    cache_control.is_some(),
                    "trailing tool_result must carry a cache breakpoint"
                );
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn cache_control_serializes_one_hour_ttl() {
        let json = serde_json::to_value(CACHE_CONTROL).expect("serialize");
        assert_eq!(json["type"], "ephemeral");
        assert_eq!(json["ttl"], "1h");
    }

    #[test]
    fn bedrock_tool_schema_flattens_top_level_oneof() {
        let schema = serde_json::json!({
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["symbols"]
                },
                {
                    "type": "object",
                    "properties": {
                        "targets": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"}
                                }
                            }
                        }
                    },
                    "required": ["targets"]
                }
            ]
        });

        let flattened = bedrock_tool_input_schema(schema);

        assert!(flattened.get("oneOf").is_none());
        assert!(flattened.get("anyOf").is_none());
        assert!(flattened.get("allOf").is_none());
        assert_eq!(flattened["type"], "object");
        assert_eq!(flattened["properties"]["symbols"]["type"], "array");
        assert_eq!(flattened["properties"]["targets"]["type"], "array");
        assert!(flattened.get("required").is_none());
    }

    #[test]
    fn bedrock_tool_schema_flattens_top_level_anyof() {
        let schema = serde_json::json!({
            "description": "Either a symbol lookup or a source-location lookup.",
            "additionalProperties": false,
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"symbols": {"type": "array"}},
                    "required": ["symbols"]
                },
                {
                    "type": "object",
                    "properties": {"targets": {"type": "array"}},
                    "required": ["targets"]
                }
            ]
        });

        let flattened = bedrock_tool_input_schema(schema);

        assert!(flattened.get("anyOf").is_none());
        assert_eq!(flattened["type"], "object");
        assert_eq!(
            flattened["description"],
            "Either a symbol lookup or a source-location lookup."
        );
        assert_eq!(flattened["additionalProperties"], false);
        assert_eq!(flattened["properties"]["symbols"]["type"], "array");
        assert_eq!(flattened["properties"]["targets"]["type"], "array");
        assert!(flattened.get("required").is_none());
    }

    #[test]
    fn bedrock_tool_schema_preserves_allof_required_fields() {
        let schema = serde_json::json!({
            "allOf": [
                {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                },
                {
                    "type": "object",
                    "properties": {"line": {"type": "integer"}},
                    "required": ["line"]
                }
            ]
        });

        let flattened = bedrock_tool_input_schema(schema);

        assert!(flattened.get("allOf").is_none());
        assert_eq!(flattened["properties"]["path"]["type"], "string");
        assert_eq!(flattened["properties"]["line"]["type"], "integer");
        let required = flattened["required"].as_array().expect("required array");
        assert!(required.contains(&serde_json::json!("path")));
        assert!(required.contains(&serde_json::json!("line")));
    }

    #[test]
    fn bedrock_tool_schema_leaves_nested_combiner_untouched() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "one_of_value": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "integer"}
                    ]
                },
                "any_of_value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"}
                    ]
                },
                "all_of_value": {
                    "allOf": [
                        {"type": "object", "properties": {"path": {"type": "string"}}},
                        {"type": "object", "properties": {"line": {"type": "integer"}}}
                    ]
                }
            }
        });

        assert_eq!(bedrock_tool_input_schema(schema.clone()), schema);
    }

    #[test]
    fn bedrock_tool_schema_anthropic_request_serializes_without_top_level_combiner() {
        let tools = vec![ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: "scan_usages_by_reference".to_string(),
                description: "Find references by symbol or source location".to_string(),
                parameters: serde_json::json!({
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {"symbols": {"type": "array"}},
                            "required": ["symbols"]
                        },
                        {
                            "type": "object",
                            "properties": {"targets": {"type": "array"}},
                            "required": ["targets"]
                        }
                    ]
                }),
            },
        }];

        let converted = convert_tools(tools, false);
        let request = build_anthropic_request(
            ANTHROPIC_VERSION,
            None,
            vec![BedrockMessage {
                role: "user".to_string(),
                content: vec![BedrockContentOut::Text {
                    text: "find references".to_string(),
                    cache_control: None,
                }],
            }],
            Some(converted),
            None,
            AnthropicReasoningControls {
                effort: None,
                shape: ThinkingShape::Enabled,
                send_output_effort: false,
            },
        );
        let serialized = serde_json::to_value(&request).expect("serialize Bedrock request");
        let input_schema = &serialized["tools"][0]["input_schema"];

        assert!(input_schema.get("oneOf").is_none());
        assert!(input_schema.get("anyOf").is_none());
        assert!(input_schema.get("allOf").is_none());
        assert_eq!(input_schema["type"], "object");
        assert_eq!(input_schema["properties"]["symbols"]["type"], "array");
        assert_eq!(input_schema["properties"]["targets"]["type"], "array");
    }

    #[test]
    fn cache_control_on_last_tool() {
        let tools = vec![
            ToolDefinition {
                r#type: "function".to_string(),
                function: FunctionDef {
                    name: "read_file".to_string(),
                    description: "Read a file".to_string(),
                    parameters: serde_json::json!({}),
                },
            },
            ToolDefinition {
                r#type: "function".to_string(),
                function: FunctionDef {
                    name: "write_file".to_string(),
                    description: "Write a file".to_string(),
                    parameters: serde_json::json!({}),
                },
            },
        ];
        let converted = convert_tools(tools, true);

        // Last tool has cache_control
        assert!(converted[1].cache_control.is_some());
        // First tool does NOT
        assert!(converted[0].cache_control.is_none());
    }

    #[test]
    fn no_cache_control_for_non_claude() {
        let messages = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hello"),
        ];
        let (system_blocks, converted) =
            convert_messages(messages, false).expect("convert without cache");

        // No system blocks have cache_control
        for block in &system_blocks {
            assert!(block.cache_control.is_none());
        }

        // No user message content has cache_control
        for msg in &converted {
            if msg.role == "user" {
                for block in &msg.content {
                    if let BedrockContentOut::Text { cache_control, .. } = block {
                        assert!(cache_control.is_none());
                    }
                }
            }
        }
    }

    #[test]
    fn openai_models_use_responses_api_path() {
        assert!(uses_responses_api("openai.gpt-5.4"));
        assert!(!uses_responses_api("us.anthropic.claude-sonnet-4-6"));
    }

    #[tokio::test]
    async fn reasoning_effort_adds_adaptive_thinking_and_surfaces_thoughts() {
        use std::sync::{Arc, Mutex};
        use wiremock::matchers::body_partial_json;

        let server = MockServer::start().await;
        // Claude 4.6+ uses adaptive thinking plus output_config.effort, not
        // the legacy budget_tokens shape.
        Mock::given(method("POST"))
            .and(path(
                "/model/us.anthropic.claude-sonnet-4-6/invoke-with-response-stream",
            ))
            .and(body_partial_json(serde_json::json!({
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "medium"}
            })))
            .respond_with(eventstream_response(&anthropic_stream_events(
                &[
                    StreamBlock::Thinking("let me reason"),
                    StreamBlock::Text("done"),
                ],
                Some("end_turn"),
                5,
                3,
            )))
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-4-6".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let thoughts = Arc::new(Mutex::new(String::new()));
        let captured = thoughts.clone();
        let response = client
            .stream_chat(StreamChatRequest {
                model: "us.anthropic.claude-sonnet-4-6".to_string(),
                messages: vec![ChatMessage::user("hi")],
                tools: None,
                reasoning_effort: Some("medium".to_string()),
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(move |t| captured.lock().unwrap().push_str(t)),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
            })
            .await
            .expect("native request with thinking should succeed");
        match response {
            LlmResponse::Text { text, .. } => assert_eq!(text, "done"),
            other => panic!("expected text response, got {other:?}"),
        }
        assert_eq!(thoughts.lock().unwrap().as_str(), "let me reason");
    }

    #[tokio::test]
    async fn unrecognized_anthropic_models_keep_generic_thinking_budget() {
        use wiremock::{Match, Request};

        struct GenericThinkingBlock;
        impl Match for GenericThinkingBlock {
            fn matches(&self, request: &Request) -> bool {
                let Ok(v) = serde_json::from_slice::<serde_json::Value>(&request.body) else {
                    return false;
                };
                v.get("thinking")
                    == Some(&serde_json::json!({"type": "enabled", "budget_tokens": 8192}))
                    && v.get("output_config").is_none()
            }
        }

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(
                "/model/us.anthropic.claude-3-5-sonnet/invoke-with-response-stream",
            ))
            .and(GenericThinkingBlock)
            .respond_with(text_stream_response("plain", 2, 1))
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-3-5-sonnet".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let response = client
            .stream_chat(StreamChatRequest {
                model: "us.anthropic.claude-3-5-sonnet".to_string(),
                messages: vec![ChatMessage::user("hi")],
                tools: None,
                reasoning_effort: Some("high".to_string()),
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
            })
            .await
            .expect("native request should keep generic thinking controls");
        match response {
            LlmResponse::Text { text, .. } => assert_eq!(text, "plain"),
            other => panic!("expected text response, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn manual_anthropic_reasoning_models_send_budget_without_effort_field() {
        use wiremock::{Match, Request};

        struct ManualThinkingOnly;
        impl Match for ManualThinkingOnly {
            fn matches(&self, request: &Request) -> bool {
                let Ok(v) = serde_json::from_slice::<serde_json::Value>(&request.body) else {
                    return false;
                };
                v.get("thinking")
                    == Some(&serde_json::json!({"type": "enabled", "budget_tokens": 8192}))
                    && v.get("output_config").is_none()
            }
        }

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(
                "/model/global.anthropic.claude-sonnet-4-5/invoke-with-response-stream",
            ))
            .and(ManualThinkingOnly)
            .respond_with(text_stream_response("manual ok", 2, 1))
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "global.anthropic.claude-sonnet-4-5".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let response = client
            .stream_chat(StreamChatRequest {
                model: "global.anthropic.claude-sonnet-4-5".to_string(),
                messages: vec![ChatMessage::user("hi")],
                tools: None,
                reasoning_effort: Some("high".to_string()),
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
            })
            .await
            .expect("manual thinking request should succeed");
        match response {
            LlmResponse::Text { text, .. } => assert_eq!(text, "manual ok"),
            other => panic!("expected text response, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn adaptive_anthropic_models_forward_xhigh_effort() {
        use wiremock::matchers::body_partial_json;

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(
                "/model/global.anthropic.claude-opus-4-8/invoke-with-response-stream",
            ))
            .and(body_partial_json(serde_json::json!({
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "xhigh"}
            })))
            .respond_with(text_stream_response("xhigh ok", 2, 1))
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "global.anthropic.claude-opus-4-8".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let response = client
            .stream_chat(StreamChatRequest {
                model: "global.anthropic.claude-opus-4-8".to_string(),
                messages: vec![ChatMessage::user("hi")],
                tools: None,
                reasoning_effort: Some("xhigh".to_string()),
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
            })
            .await
            .expect("adaptive request should accept xhigh");
        match response {
            LlmResponse::Text { text, .. } => assert_eq!(text, "xhigh ok"),
            other => panic!("expected text response, got {other:?}"),
        }
    }

    #[test]
    fn native_stream_idle_floor_survives_a_short_session_timeout() {
        // The session default is 60s inter-chunk; adaptive thinking on a
        // real supervisor request goes quiet far longer than that, so the
        // floor must win or normal turns get aborted as "stalled".
        let raised = native_stream_idle_timeouts(IdleTimeouts::uniform(Duration::from_secs(60)));
        assert_eq!(raised.inter_chunk, NATIVE_STREAM_IDLE_FLOOR);
        assert_eq!(raised.first_progress, NATIVE_STREAM_IDLE_FLOOR);

        // An operator who configured something more generous keeps it.
        let generous = IdleTimeouts::uniform(Duration::from_secs(1_200));
        let kept = native_stream_idle_timeouts(generous);
        assert_eq!(kept.inter_chunk, Duration::from_secs(1_200));

        // The backstop stays inside the whole-request ceiling.
        assert!(NATIVE_STREAM_IDLE_FLOOR < NATIVE_INVOKE_TIMEOUT);
    }

    #[tokio::test]
    async fn adaptive_requests_reserve_headroom_beyond_the_flat_ceiling() {
        use wiremock::matchers::body_partial_json;

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(
                "/model/us.anthropic.claude-sonnet-5/invoke-with-response-stream",
            ))
            .and(body_partial_json(serde_json::json!({
                // `display` must be explicit: the provider default is
                // `omitted`, which streams thinking blocks with empty text
                // and leaves a long generation emitting nothing until the
                // answer starts.
                "thinking": {"type": "adaptive", "display": "summarized"},
                "max_tokens": ADAPTIVE_MAX_TOKENS,
            })))
            .respond_with(eventstream_response(&anthropic_stream_events(
                &[
                    StreamBlock::Thinking("weighing options"),
                    StreamBlock::Text("roomy"),
                ],
                Some("end_turn"),
                2,
                1,
            )))
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-5".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let thoughts = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let captured = thoughts.clone();
        let response = client
            .stream_chat(StreamChatRequest {
                model: "us.anthropic.claude-sonnet-5".to_string(),
                messages: vec![ChatMessage::user("hi")],
                tools: None,
                reasoning_effort: Some("high".to_string()),
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(move |t| captured.lock().unwrap().push_str(t)),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
            })
            .await
            .expect("adaptive request should carry the adaptive ceiling");
        match response {
            LlmResponse::Text { text, .. } => assert_eq!(text, "roomy"),
            other => panic!("expected text response, got {other:?}"),
        }
        // Summarized thinking must reach `on_thought` as it streams.
        assert_eq!(thoughts.lock().unwrap().as_str(), "weighing options");
    }

    #[tokio::test]
    async fn thinking_only_max_tokens_response_is_not_retryable() {
        let server = MockServer::start().await;
        // The budget-exhaustion shape: the whole allowance went to thinking,
        // so there is no text and no tool_use to return.
        Mock::given(method("POST"))
            .and(path(
                "/model/us.anthropic.claude-sonnet-5/invoke-with-response-stream",
            ))
            .respond_with(eventstream_response(&anthropic_stream_events(
                &[StreamBlock::Thinking("deliberating")],
                Some("max_tokens"),
                5,
                64_000,
            )))
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-5".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let error = client
            .stream_chat(StreamChatRequest {
                model: "us.anthropic.claude-sonnet-5".to_string(),
                messages: vec![ChatMessage::user("hi")],
                tools: None,
                reasoning_effort: Some("high".to_string()),
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
            })
            .await
            .expect_err("thinking-only max_tokens must not look like a usable response");
        assert!(
            crate::llm_client::is_output_budget_exhausted_error(&error),
            "expected budget exhaustion, got {error:#}"
        );
        // Non-retryable: re-sending would hit the identical cap.
        assert!(
            crate::llm_client::llm_retry_tier(&error).is_none(),
            "budget exhaustion must not be retried"
        );
    }

    #[tokio::test]
    async fn truncated_response_with_text_is_still_returned() {
        let server = MockServer::start().await;
        // Same ceiling hit, but the model did emit something usable --
        // return the truncated content instead of erroring.
        Mock::given(method("POST"))
            .and(path(
                "/model/us.anthropic.claude-sonnet-5/invoke-with-response-stream",
            ))
            .respond_with(eventstream_response(&anthropic_stream_events(
                &[
                    StreamBlock::Thinking("deliberating"),
                    StreamBlock::Text("partial answer"),
                ],
                Some("max_tokens"),
                5,
                64_000,
            )))
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-5".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let response = client
            .stream_chat(StreamChatRequest {
                model: "us.anthropic.claude-sonnet-5".to_string(),
                messages: vec![ChatMessage::user("hi")],
                tools: None,
                reasoning_effort: Some("high".to_string()),
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
            })
            .await
            .expect("truncated-but-non-empty content should still be returned");
        match response {
            LlmResponse::Text { text, .. } => assert_eq!(text, "partial answer"),
            other => panic!("expected text response, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn reasoning_effort_falls_back_to_adaptive_shape_on_400() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Mutex};
        use wiremock::matchers::body_partial_json;
        use wiremock::{Match, Request};

        // First attempt: manual `enabled` shape -> provider rejects it with
        // the documented "use adaptive + output_config.effort" 400.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/model/anthropic.claude-opus-4-5/invoke-with-response-stream"))
            .and(body_partial_json(serde_json::json!({
                "thinking": {"type": "enabled"}
            })))
            .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "message": "\"thinking.type.enabled\" is not supported for this model. Use \"thinking.type.adaptive\" and \"output_config.effort\" to control thinking behavior."
            })))
            .mount(&server)
            .await;

        // Count adaptive-shaped requests so we can prove the cache prevents a
        // second probe on the follow-up turn.
        let adaptive_hits = Arc::new(AtomicUsize::new(0));
        struct AdaptiveShape(Arc<AtomicUsize>);
        impl Match for AdaptiveShape {
            fn matches(&self, request: &Request) -> bool {
                let Ok(v) = serde_json::from_slice::<serde_json::Value>(&request.body) else {
                    return false;
                };
                let ok = v
                    .get("thinking")
                    .and_then(|t| t.get("type"))
                    .and_then(|t| t.as_str())
                    == Some("adaptive")
                    && v.get("output_config")
                        .and_then(|o| o.get("effort"))
                        .and_then(|e| e.as_str())
                        == Some("high");
                if ok {
                    self.0.fetch_add(1, Ordering::SeqCst);
                }
                ok
            }
        }
        Mock::given(method("POST"))
            .and(path(
                "/model/anthropic.claude-opus-4-5/invoke-with-response-stream",
            ))
            .and(AdaptiveShape(adaptive_hits.clone()))
            .respond_with(eventstream_response(&anthropic_stream_events(
                &[
                    StreamBlock::Thinking("weighing"),
                    StreamBlock::Text("adaptive ok"),
                ],
                Some("end_turn"),
                6,
                4,
            )))
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "anthropic.claude-opus-4-5".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );

        let make_request = || {
            let thoughts = Arc::new(Mutex::new(String::new()));
            let captured = thoughts.clone();
            (
                StreamChatRequest {
                    model: "anthropic.claude-opus-4-5".to_string(),
                    messages: vec![ChatMessage::user("hi")],
                    tools: None,
                    reasoning_effort: Some("high".to_string()),
                    service_tier: None,
                    temperature: None,
                    structured_output: None,
                    on_token: Box::new(|_| {}),
                    on_thought: Box::new(move |t| captured.lock().unwrap().push_str(t)),
                    cancel: CancellationToken::new(),
                    idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
                },
                thoughts,
            )
        };

        let (req, thoughts) = make_request();
        let response = client
            .stream_chat(req)
            .await
            .expect("should recover by switching to adaptive shape");
        match response {
            LlmResponse::Text { text, .. } => assert_eq!(text, "adaptive ok"),
            other => panic!("expected text response, got {other:?}"),
        }
        assert_eq!(thoughts.lock().unwrap().as_str(), "weighing");
        assert_eq!(adaptive_hits.load(Ordering::SeqCst), 1);

        // Second turn: the learned shape is cached, so it goes straight to
        // adaptive (no enabled probe) -- adaptive hit count rises to 2.
        let (req2, _) = make_request();
        client
            .stream_chat(req2)
            .await
            .expect("cached adaptive shape should succeed directly");
        assert_eq!(adaptive_hits.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn gpt5_families_use_dedicated_openai_mantle_path() {
        // gpt-5.4 / gpt-5.5 and their suffixed derivatives route to /openai/v1.
        assert!(uses_openai_mantle_path("openai.gpt-5.4"));
        assert!(uses_openai_mantle_path("openai.gpt-5.5"));
        assert!(uses_openai_mantle_path("openai.gpt-5.5-codex"));
        assert!(uses_openai_mantle_path("openai.gpt-5.4-2026-01-01"));
        assert!(uses_openai_mantle_path("openai.gpt-5.6-sol"));
        assert!(uses_openai_mantle_path("openai.gpt-5.6-luna"));
        assert!(uses_openai_mantle_path("openai.gpt-5.6"));
        // Other Mantle models -- including a hypothetical id that merely shares
        // the numeric prefix -- stay on the shared /v1 path.
        assert!(!uses_openai_mantle_path("openai.gpt-oss-120b"));
        assert!(!uses_openai_mantle_path("openai.gpt-5.40"));
        assert!(!uses_openai_mantle_path("openai.gpt-5.3"));

        let base = "https://bedrock-mantle.us-east-2.api.aws/v1";
        assert_eq!(
            mantle_base_url_for_model(base, "openai.gpt-5.5"),
            "https://bedrock-mantle.us-east-2.api.aws/openai/v1"
        );
        assert_eq!(mantle_base_url_for_model(base, "openai.gpt-oss-120b"), base);
        // A base without the expected /v1 suffix is returned unchanged.
        let odd = "https://example.test/custom";
        assert_eq!(mantle_base_url_for_model(odd, "openai.gpt-5.5"), odd);
    }

    #[tokio::test]
    async fn gpt5_responses_models_route_to_openai_mantle_path() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/v1/responses"))
            .and(header("authorization", "Bearer token"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(
                        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n\
                         data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
                    ),
            )
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-4-6".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let response = client
            .stream_chat(StreamChatRequest {
                model: "openai.gpt-5.5".to_string(),
                messages: vec![ChatMessage::user("hi")],
                tools: None,
                reasoning_effort: None,
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
            })
            .await
            .expect("responses request should succeed");
        match response {
            LlmResponse::Text { text, usage, .. } => {
                assert_eq!(text, "ok");
                assert_eq!(usage.input_tokens, 3);
            }
            other => panic!("expected text response, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn responses_api_chains_second_turn_via_previous_response_id() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(
                        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_turn1\"}}\n\n\
                         data: {\"type\":\"response.output_text.delta\",\"delta\":\"turn1 answer\"}\n\n\
                         data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_turn1\",\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
                    ),
            )
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-4-6".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );

        let request = |messages: Vec<ChatMessage>| StreamChatRequest {
            model: "openai.gpt-5.5".to_string(),
            messages,
            tools: None,
            reasoning_effort: None,
            service_tier: None,
            temperature: None,
            structured_output: None,
            on_token: Box::new(|_| {}),
            on_thought: Box::new(|_| {}),
            cancel: CancellationToken::new(),
            idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
        };

        let turn1_messages = vec![ChatMessage::user("turn1 question")];
        let response1 = client
            .stream_chat(request(turn1_messages.clone()))
            .await
            .expect("turn 1 should succeed");
        assert!(matches!(response1, LlmResponse::Text { .. }));

        // Simulate the caller re-sending the full conversation as required
        // by `LlmBackend::stream_chat`'s stateless-caller contract: the
        // prior assistant turn is echoed back, plus a new user turn.
        let mut turn2_messages = turn1_messages.clone();
        turn2_messages.push(ChatMessage::assistant("turn1 answer"));
        turn2_messages.push(ChatMessage::user("turn2 question"));

        let response2 = client
            .stream_chat(request(turn2_messages))
            .await
            .expect("turn 2 should succeed");
        assert!(matches!(response2, LlmResponse::Text { .. }));

        let requests = server.received_requests().await.expect("recorded requests");
        assert_eq!(requests.len(), 2);

        let turn1_body: serde_json::Value =
            serde_json::from_slice(&requests[0].body).expect("turn 1 body is JSON");
        assert_eq!(turn1_body["store"], serde_json::json!(true));
        assert!(
            turn1_body.get("previous_response_id").is_none(),
            "first turn has nothing to chain onto: {turn1_body}"
        );
        assert_eq!(turn1_body["input"].as_array().unwrap().len(), 1);

        let turn2_body: serde_json::Value =
            serde_json::from_slice(&requests[1].body).expect("turn 2 body is JSON");
        assert_eq!(turn2_body["store"], serde_json::json!(true));
        assert_eq!(
            turn2_body["previous_response_id"],
            serde_json::json!("resp_turn1"),
            "turn 2 must chain onto turn 1's response id: {turn2_body}"
        );
        let turn2_input = turn2_body["input"].as_array().expect("input array");
        assert_eq!(
            turn2_input.len(),
            1,
            "continuation must send only the delta, not the full history: {turn2_input:?}"
        );
        let turn2_body_str = turn2_body.to_string();
        assert!(
            !turn2_body_str.contains("turn1 question") && !turn2_body_str.contains("turn1 answer"),
            "continuation input must not resend turn 1 history: {turn2_body_str}"
        );
        assert!(turn2_body_str.contains("turn2 question"));
    }

    #[tokio::test]
    async fn responses_api_stream_failure_evicts_all_cached_prefixes_for_context() {
        struct FailingChainedResponsesResponder {
            calls: Arc<AtomicUsize>,
        }

        fn successful_responses_sse(response_id: &str, text: &str) -> ResponseTemplate {
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!(
                    "data: {{\"type\":\"response.created\",\"response\":{{\"id\":\"{response_id}\"}}}}\n\n\
                     data: {{\"type\":\"response.output_text.delta\",\"delta\":\"{text}\"}}\n\n\
                     data: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"{response_id}\"}}}}\n\n",
                ))
        }

        impl Respond for FailingChainedResponsesResponder {
            fn respond(&self, request: &Request) -> ResponseTemplate {
                let attempt = self.calls.fetch_add(1, Ordering::SeqCst);
                let body: serde_json::Value =
                    serde_json::from_slice(&request.body).expect("request body is JSON");
                let previous_response_id = body
                    .get("previous_response_id")
                    .and_then(serde_json::Value::as_str);
                let body_str = body.to_string();

                match attempt {
                    0 => {
                        assert_eq!(previous_response_id, None);
                        assert!(body_str.contains("turn1 question"), "{body_str}");
                        successful_responses_sse("resp_turn0", "turn1 answer")
                    }
                    1 => {
                        assert_eq!(previous_response_id, Some("resp_turn0"));
                        assert!(!body_str.contains("turn1 question"), "{body_str}");
                        assert!(body_str.contains("turn2 question"), "{body_str}");
                        successful_responses_sse("resp_turn1", "turn2 answer")
                    }
                    2 => {
                        assert_eq!(previous_response_id, Some("resp_turn1"));
                        ResponseTemplate::new(200)
                            .insert_header("content-type", "text/event-stream")
                            .set_body_string(
                                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n\
                                 data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"code\":\"server_error\",\"message\":\"overloaded\"}}}\n\n",
                            )
                    }
                    3 => {
                        assert_eq!(
                            previous_response_id, None,
                            "after stream failure, every cached prefix for this context must be evicted"
                        );
                        assert!(body_str.contains("turn1 question"), "{body_str}");
                        assert!(body_str.contains("turn2 question"), "{body_str}");
                        assert!(body_str.contains("turn3 question"), "{body_str}");
                        successful_responses_sse("resp_recovered", "recovered")
                    }
                    other => panic!("unexpected Responses request #{other}: {body_str}"),
                }
            }
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/v1/responses"))
            .respond_with(FailingChainedResponsesResponder {
                calls: calls.clone(),
            })
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-4-6".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );

        let request = |messages: Vec<ChatMessage>| StreamChatRequest {
            model: "openai.gpt-5.5".to_string(),
            messages,
            tools: None,
            reasoning_effort: None,
            service_tier: None,
            temperature: None,
            structured_output: None,
            on_token: Box::new(|_| {}),
            on_thought: Box::new(|_| {}),
            cancel: CancellationToken::new(),
            idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
        };

        let turn0_messages = vec![ChatMessage::user("turn1 question")];
        client
            .stream_chat(request(turn0_messages.clone()))
            .await
            .expect("turn 1 should seed the first chain link");

        let mut turn1_messages = turn0_messages.clone();
        turn1_messages.push(ChatMessage::assistant("turn1 answer"));
        turn1_messages.push(ChatMessage::user("turn2 question"));
        client
            .stream_chat(request(turn1_messages.clone()))
            .await
            .expect("turn 2 should seed the second chain link");

        let mut turn2_messages = turn1_messages.clone();
        turn2_messages.push(ChatMessage::assistant("turn2 answer"));
        turn2_messages.push(ChatMessage::user("turn3 question"));

        let err = client
            .stream_chat(request(turn2_messages.clone()))
            .await
            .expect_err("stream-borne response.failed should propagate to the outer retry loop");
        assert!(
            format!("{err:#}").contains("Responses stream failed: server_error: overloaded"),
            "{err:#}"
        );

        {
            let cache = client
                .responses_chain
                .lock()
                .expect("responses_chain mutex poisoned");
            assert!(
                find_responses_continuation(&turn2_messages, |h| cache.get(h)).is_none(),
                "stream failure on a chained call must leave no usable cached prefix for the same context"
            );
        }

        client
            .stream_chat(request(turn2_messages))
            .await
            .expect("after eviction, the next identical invocation should send full input");

        let requests = server.received_requests().await.expect("recorded requests");
        assert_eq!(requests.len(), 4);
        let retry_body: serde_json::Value =
            serde_json::from_slice(&requests[3].body).expect("retry body is JSON");
        assert!(
            retry_body.get("previous_response_id").is_none(),
            "post-eviction retry must not fall back to an older cached prefix: {retry_body}"
        );
    }

    #[tokio::test]
    async fn responses_api_self_heals_on_not_found_previous_response_id() {
        use wiremock::{Match, Request};

        struct HasPreviousResponseId;
        impl Match for HasPreviousResponseId {
            fn matches(&self, request: &Request) -> bool {
                let Ok(v) = serde_json::from_slice::<serde_json::Value>(&request.body) else {
                    return false;
                };
                v.get("previous_response_id").is_some()
            }
        }
        struct NoPreviousResponseId;
        impl Match for NoPreviousResponseId {
            fn matches(&self, request: &Request) -> bool {
                let Ok(v) = serde_json::from_slice::<serde_json::Value>(&request.body) else {
                    return false;
                };
                v.get("previous_response_id").is_none()
            }
        }

        let server = MockServer::start().await;
        // Any request WITHOUT previous_response_id (the initial turn, and
        // the post-eviction retry) succeeds.
        Mock::given(method("POST"))
            .and(path("/openai/v1/responses"))
            .and(NoPreviousResponseId)
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(
                        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_turn1\"}}\n\n\
                         data: {\"type\":\"response.output_text.delta\",\"delta\":\"answer\"}\n\n\
                         data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_turn1\"}}\n\n",
                    ),
            )
            .mount(&server)
            .await;
        // Any request WITH previous_response_id is rejected as expired/unknown.
        Mock::given(method("POST"))
            .and(path("/openai/v1/responses"))
            .and(HasPreviousResponseId)
            .respond_with(ResponseTemplate::new(404).set_body_string(
                "{\"error\":{\"code\":\"not_found_error\",\"message\":\"Response 'resp_akkecf7gb36b3mffg2eifdikhj3cmi6fn7p3mh3muslnh6fbryxa' not found.\",\"param\":null,\"type\":\"invalid_request_error\"}}",
            ))
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-4-6".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );

        let request = |messages: Vec<ChatMessage>| StreamChatRequest {
            model: "openai.gpt-5.5".to_string(),
            messages,
            tools: None,
            reasoning_effort: None,
            service_tier: None,
            temperature: None,
            structured_output: None,
            on_token: Box::new(|_| {}),
            on_thought: Box::new(|_| {}),
            cancel: CancellationToken::new(),
            idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
        };

        let turn1_messages = vec![ChatMessage::user("turn1 question")];
        client
            .stream_chat(request(turn1_messages.clone()))
            .await
            .expect("turn 1 should succeed and seed the chain cache");

        let mut turn2_messages = turn1_messages.clone();
        turn2_messages.push(ChatMessage::assistant("answer"));
        turn2_messages.push(ChatMessage::user("turn2 question"));

        // Turn 2 detects a (now-stale) continuation, gets rejected, and must
        // self-heal by evicting the bad id and retrying with the full
        // input rather than surfacing the error or corrupting the run.
        let response2 = client
            .stream_chat(request(turn2_messages.clone()))
            .await
            .expect("turn 2 should self-heal and succeed despite the expired id");
        assert!(matches!(response2, LlmResponse::Text { .. }));

        let requests = server.received_requests().await.expect("recorded requests");
        assert_eq!(
            requests.len(),
            3,
            "turn 1, the rejected continuation attempt, and the full-input retry"
        );

        let retry_body: serde_json::Value =
            serde_json::from_slice(&requests[2].body).expect("retry body is JSON");
        assert!(
            retry_body.get("previous_response_id").is_none(),
            "retry after eviction must not resend the stale id: {retry_body}"
        );
        let retry_body_str = retry_body.to_string();
        assert!(
            retry_body_str.contains("turn1 question") && retry_body_str.contains("turn2 question"),
            "retry after eviction must resend the full conversation: {retry_body_str}"
        );

        // The stale entry should actually be gone from the cache (not just
        // papered over by the retry): confirm eviction directly.
        let stale_key = hash_responses_context(&turn1_messages);
        let cache = client
            .responses_chain
            .lock()
            .expect("responses_chain mutex poisoned");
        assert!(
            cache.get(stale_key).is_none(),
            "expired previous_response_id entry must be evicted, not just bypassed"
        );
    }

    #[tokio::test]
    async fn gpt5_responses_models_filter_reasoning_effort_to_model_card_values() {
        use wiremock::matchers::body_partial_json;
        use wiremock::{Match, Request};

        struct NoReasoningObject;
        impl Match for NoReasoningObject {
            fn matches(&self, request: &Request) -> bool {
                let Ok(v) = serde_json::from_slice::<serde_json::Value>(&request.body) else {
                    return false;
                };
                v.get("reasoning").is_none()
            }
        }

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/v1/responses"))
            .and(body_partial_json(serde_json::json!({
                "reasoning": {"effort": "xhigh"}
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(
                        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"xhigh\"}\n\n\
                         data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
                    ),
            )
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/openai/v1/responses"))
            .and(NoReasoningObject)
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(
                        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"filtered\"}\n\n\
                         data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
                    ),
            )
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-4-6".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let supported = client
            .stream_chat(StreamChatRequest {
                model: "openai.gpt-5.5".to_string(),
                messages: vec![ChatMessage::user("hi")],
                tools: None,
                reasoning_effort: Some("xhigh".to_string()),
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
            })
            .await
            .expect("supported reasoning request should succeed");
        assert!(matches!(supported, LlmResponse::Text { ref text, .. } if text == "xhigh"));

        let unsupported = client
            .stream_chat(StreamChatRequest {
                model: "openai.gpt-5.5".to_string(),
                messages: vec![ChatMessage::user("hi")],
                tools: None,
                reasoning_effort: Some("minimal".to_string()),
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
            })
            .await
            .expect("unsupported reasoning request should omit reasoning");
        assert!(matches!(unsupported, LlmResponse::Text { ref text, .. } if text == "filtered"));
    }

    #[tokio::test]
    async fn other_responses_models_use_shared_mantle_path() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer token"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(
                        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n\
                         data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
                    ),
            )
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-4-6".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let response = client
            .stream_chat(StreamChatRequest {
                model: "openai.gpt-oss-120b".to_string(),
                messages: vec![ChatMessage::user("hi")],
                tools: None,
                reasoning_effort: None,
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
            })
            .await
            .expect("responses request should succeed");
        match response {
            LlmResponse::Text { text, usage, .. } => {
                assert_eq!(text, "ok");
                assert_eq!(usage.input_tokens, 3);
            }
            other => panic!("expected text response, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn anthropic_models_still_use_native_invoke() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"/model/.+/invoke-with-response-stream$"))
            .and(header("authorization", "Bearer token"))
            .respond_with(text_stream_response("native ok", 2, 9))
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-4-6".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let response = client
            .stream_chat(StreamChatRequest {
                model: "us.anthropic.claude-sonnet-4-6".to_string(),
                messages: vec![ChatMessage::user("hi")],
                tools: None,
                reasoning_effort: None,
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
            })
            .await
            .expect("native request should succeed");
        match response {
            LlmResponse::Text { text, usage, .. } => {
                assert_eq!(text, "native ok");
                // Streaming splits usage: inputs arrive in `message_start`,
                // the real output count only in `message_delta`.
                assert_eq!(usage.input_tokens, 2);
                assert_eq!(usage.output_tokens, 9);
            }
            other => panic!("expected text response, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn native_tool_calls_are_assembled_from_input_json_deltas() {
        let server = MockServer::start().await;
        // Split the arguments across two `input_json_delta` events: the
        // provider fragments them arbitrarily and only the concatenation is
        // valid JSON.
        let mut events = vec![
            serde_json::json!({
                "type": "message_start",
                "message": {"usage": {"input_tokens": 11, "output_tokens": 1}}
            }),
            serde_json::json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {}}
            }),
        ];
        for fragment in ["{\"path\":", "\"/tmp/x\"}"] {
            events.push(serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": fragment}
            }));
        }
        events.push(serde_json::json!({"type": "content_block_stop", "index": 0}));
        events.push(serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 12}
        }));
        events.push(serde_json::json!({"type": "message_stop"}));

        Mock::given(method("POST"))
            .and(path(
                "/model/us.anthropic.claude-sonnet-5/invoke-with-response-stream",
            ))
            .respond_with(eventstream_response(&events))
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-5".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let response = client
            .stream_chat(StreamChatRequest {
                model: "us.anthropic.claude-sonnet-5".to_string(),
                messages: vec![ChatMessage::user("hi")],
                tools: None,
                reasoning_effort: None,
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
            })
            .await
            .expect("streamed tool call should succeed");
        match response {
            LlmResponse::ToolCalls { calls, usage, .. } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "call_1");
                assert_eq!(calls[0].function.name, "read_file");
                assert_eq!(calls[0].function.arguments, "{\"path\":\"/tmp/x\"}");
                assert_eq!(usage.input_tokens, 11);
                assert_eq!(usage.output_tokens, 12);
            }
            other => panic!("expected tool calls, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn streamed_tool_calls_keep_leading_text_and_default_empty_arguments() {
        let server = MockServer::start().await;
        // A no-argument tool call streams no `input_json_delta` at all, and
        // any text emitted before it has to survive alongside the call.
        Mock::given(method("POST"))
            .and(path(
                "/model/us.anthropic.claude-sonnet-5/invoke-with-response-stream",
            ))
            .respond_with(eventstream_response(&anthropic_stream_events(
                &[
                    StreamBlock::Text("checking"),
                    StreamBlock::ToolUse {
                        id: "call_0",
                        name: "list_files",
                        input_json: "",
                    },
                ],
                Some("tool_use"),
                3,
                4,
            )))
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-5".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let response = client
            .stream_chat(StreamChatRequest {
                model: "us.anthropic.claude-sonnet-5".to_string(),
                messages: vec![ChatMessage::user("hi")],
                tools: None,
                reasoning_effort: None,
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
            })
            .await
            .expect("streamed tool call should succeed");
        match response {
            LlmResponse::ToolCalls { text, calls, .. } => {
                assert_eq!(text, "checking");
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "list_files");
                assert_eq!(calls[0].function.arguments, "{}");
            }
            other => panic!("expected tool calls, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_exception_frames_stay_retryable() {
        let server = MockServer::start().await;
        let mut body = eventstream_body(&[serde_json::json!({
            "type": "message_start",
            "message": {"usage": {"input_tokens": 1, "output_tokens": 1}}
        })]);
        // Bedrock reports mid-stream failures as exception frames, not as an
        // HTTP status, so the retry tier has to come from the frame headers.
        body.extend_from_slice(&eventstream_frame(
            &[
                (":message-type", "exception"),
                (":exception-type", "throttlingException"),
            ],
            br#"{"message":"Too many requests"}"#,
        ));
        Mock::given(method("POST"))
            .and(path(
                "/model/us.anthropic.claude-sonnet-5/invoke-with-response-stream",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, EVENTSTREAM_CONTENT_TYPE))
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-5".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let error = client
            .stream_chat(StreamChatRequest {
                model: "us.anthropic.claude-sonnet-5".to_string(),
                messages: vec![ChatMessage::user("hi")],
                tools: None,
                reasoning_effort: None,
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
            })
            .await
            .expect_err("a stream exception must not read as a successful turn");
        assert_eq!(
            crate::llm_client::llm_retry_tier(&error),
            Some(crate::http_retry::LlmRetryTier::Fast),
            "throttling must stay retryable: {error:#}"
        );
    }

    #[tokio::test]
    async fn truncated_stream_without_message_stop_is_retryable() {
        let server = MockServer::start().await;
        // Everything but the terminating `message_stop`: a connection that
        // died mid-generation must not be mistaken for a complete answer.
        let mut events =
            anthropic_stream_events(&[StreamBlock::Text("half an answ")], Some("end_turn"), 1, 2);
        events.pop();
        Mock::given(method("POST"))
            .and(path(
                "/model/us.anthropic.claude-sonnet-5/invoke-with-response-stream",
            ))
            .respond_with(eventstream_response(&events))
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-5".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let error = client
            .stream_chat(StreamChatRequest {
                model: "us.anthropic.claude-sonnet-5".to_string(),
                messages: vec![ChatMessage::user("hi")],
                tools: None,
                reasoning_effort: None,
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
            })
            .await
            .expect_err("a stream cut short must not read as a complete turn");
        assert!(
            crate::llm_client::is_incomplete_stream_error(&error),
            "expected an incomplete-stream error, got {error:#}"
        );
    }

    #[tokio::test]
    async fn native_stream_emits_tokens_before_completion_and_honors_cancellation() {
        // wiremock can only delay a whole response, which cannot show that
        // frames reach the sinks while the request is still open. Serve the
        // prefix by hand over a chunked connection that then stalls forever:
        // the only way this test finishes is if tokens are delivered
        // mid-stream and cancellation aborts the still-open body.
        let prefix = eventstream_body(&[
            serde_json::json!({
                "type": "message_start",
                "message": {"usage": {"input_tokens": 4, "output_tokens": 1}}
            }),
            serde_json::json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""}
            }),
            serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "streamed early"}
            }),
        ]);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind stalling stream server");
        let addr = listener.local_addr().expect("listener address");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept streaming client");
            let mut scratch = [0u8; 4096];
            let _ = tokio::io::AsyncReadExt::read(&mut socket, &mut scratch).await;
            let mut response = b"HTTP/1.1 200 OK\r\nContent-Type: application/vnd.amazon.eventstream\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
            response.extend_from_slice(format!("{:x}\r\n", prefix.len()).as_bytes());
            response.extend_from_slice(&prefix);
            response.extend_from_slice(b"\r\n");
            tokio::io::AsyncWriteExt::write_all(&mut socket, &response)
                .await
                .expect("write streamed prefix");
            tokio::io::AsyncWriteExt::flush(&mut socket)
                .await
                .expect("flush streamed prefix");
            // Never terminate the chunked body.
            std::future::pending::<()>().await;
        });

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-5".to_string(),
            format!("http://{addr}"),
            format!("http://{addr}/v1"),
            format!("http://{addr}"),
        );
        let cancel = CancellationToken::new();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let cancel_for_request = cancel.clone();
        let request = client.stream_chat(StreamChatRequest {
            model: "us.anthropic.claude-sonnet-5".to_string(),
            messages: vec![ChatMessage::user("hi")],
            tools: None,
            reasoning_effort: None,
            service_tier: None,
            temperature: None,
            structured_output: None,
            on_token: Box::new(move |t| {
                tx.send(t.to_string()).expect("token receiver alive");
            }),
            on_thought: Box::new(|_| {}),
            cancel: cancel_for_request,
            // Generous enough that only cancellation can end this stream.
            idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(120)),
        });

        let waiter = tokio::spawn(async move {
            let token = rx.recv().await.expect("a token before the stream finishes");
            cancel.cancel();
            token
        });

        let response = tokio::time::timeout(Duration::from_secs(10), request)
            .await
            .expect("cancellation must abort the open stream")
            .expect("cancelled stream returns what it already produced");
        let token = waiter.await.expect("token waiter");
        assert_eq!(token, "streamed early");
        match response {
            LlmResponse::Text { text, usage, .. } => {
                assert_eq!(text, "streamed early");
                assert_eq!(usage.input_tokens, 4);
            }
            other => panic!("expected text response, got {other:?}"),
        }
        server.abort();
    }

    #[test]
    fn eventstream_decoder_reassembles_frames_split_across_chunks() {
        let body = eventstream_body(&[
            serde_json::json!({
                "type": "message_start",
                "message": {"usage": {"input_tokens": 7, "output_tokens": 1}}
            }),
            serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"}
            }),
        ]);

        // One byte at a time: frames straddle HTTP chunk boundaries in
        // practice, so a partial frame must decode to nothing, not an error.
        let mut decoder = EventStreamDecoder::default();
        let mut events = Vec::new();
        for byte in &body {
            decoder.push(std::slice::from_ref(byte));
            while let Some(frame) = decoder.next_frame().expect("frame decodes") {
                events.push(
                    decode_anthropic_event(&frame)
                        .expect("event decodes")
                        .expect("chunk frame carries an event"),
                );
            }
        }

        assert_eq!(events.len(), 2);
        match &events[0] {
            AnthropicStreamEvent::MessageStart { message } => {
                assert_eq!(
                    message.usage.as_ref().map(|u| u.input_tokens),
                    Some(7),
                    "message_start must carry the input token count"
                );
            }
            other => panic!("expected message_start, got {other:?}"),
        }
        match &events[1] {
            AnthropicStreamEvent::ContentBlockDelta {
                delta: AnthropicStreamDelta::Text { text },
                ..
            } => assert_eq!(text, "hi"),
            other => panic!("expected a text delta, got {other:?}"),
        }
    }

    #[test]
    fn eventstream_decoder_rejects_impossible_frame_lengths() {
        let mut decoder = EventStreamDecoder::default();
        // total_length smaller than the fixed prelude + trailer: a bogus
        // length prefix means frame sync is lost, not a short read.
        decoder.push(&[0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0]);
        let err = decoder
            .next_frame()
            .expect_err("an impossible frame length must fail loudly");
        assert!(
            format!("{err:#}").contains("malformed Bedrock eventstream frame"),
            "unexpected error: {err:#}"
        );
    }

    #[tokio::test]
    async fn on_demand_throughput_errors_retry_with_inference_profile() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/inference-profiles"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "inferenceProfileSummaries": [
                    {
                        "inferenceProfileId": "global.anthropic.claude-opus-4-8",
                        "models": [
                            {
                                "modelArn": "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-opus-4-8"
                            }
                        ]
                    }
                ]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/model/anthropic.claude-opus-4-8/invoke-with-response-stream"))
            .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "message": "Invocation of model ID anthropic.claude-opus-4-8 with on-demand throughput isn’t supported. Retry your request with the ID or ARN of an inference profile that contains this model."
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(
                "/model/global.anthropic.claude-opus-4-8/invoke-with-response-stream",
            ))
            .respond_with(text_stream_response("profile ok", 4, 2))
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-4-6".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let response = client
            .stream_chat(StreamChatRequest {
                model: "anthropic.claude-opus-4-8".to_string(),
                messages: vec![ChatMessage::user("hi")],
                tools: None,
                reasoning_effort: None,
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
            })
            .await
            .expect("request should retry with inference profile");
        match response {
            LlmResponse::Text { text, usage, .. } => {
                assert_eq!(text, "profile ok");
                assert_eq!(usage.input_tokens, 4);
            }
            other => panic!("expected text response, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn invoke_resolves_prefixed_profile_before_first_request() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/inference-profiles"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "inferenceProfileSummaries": [
                    {
                        "inferenceProfileId": "us.anthropic.claude-opus-4-8",
                        "models": [
                            {
                                "modelArn": "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-opus-4-8"
                            }
                        ]
                    }
                ]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(
                "/model/us.anthropic.claude-opus-4-8/invoke-with-response-stream",
            ))
            .respond_with(text_stream_response("resolved first", 3, 2))
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-4-6".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let response = client
            .stream_chat(StreamChatRequest {
                model: "anthropic.claude-opus-4-8".to_string(),
                messages: vec![ChatMessage::user("hi")],
                tools: None,
                reasoning_effort: None,
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
            })
            .await
            .expect("request should resolve directly to prefixed profile");
        match response {
            LlmResponse::Text { text, usage, .. } => {
                assert_eq!(text, "resolved first");
                assert_eq!(usage.input_tokens, 3);
            }
            other => panic!("expected text response, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn mantle_models_are_discovered_and_default_is_preserved() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {
                        "id": "openai.gpt-5.4",
                        "supported_parameters": ["reasoning"],
                        "default_parameters": {"reasoning": {"effort": "medium"}},
                        "context_length": 400000
                    }
                ]
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/foundation-models"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "modelSummaries": [
                            {
                                "modelId": "anthropic.claude-sonnet-4-20250514-v1:0",
                                "inputModalities": ["TEXT", "IMAGE"],
                                "outputModalities": ["TEXT"],
                                "responseStreamingSupported": true
                            }
                        ]
                    })
                    .to_string(),
                    "application/json",
                ),
            )
            .mount(&server)
            .await;
        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-4-6".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let models = client
            .list_model_metadata()
            .await
            .expect("discovery should succeed");
        assert!(models.iter().any(|m| m.id == "openai.gpt-5.4"));
        assert!(
            models
                .iter()
                .any(|m| m.id == "anthropic.claude-sonnet-4-20250514-v1:0")
        );
        assert_eq!(
            models
                .iter()
                .find(|m| m.id == "anthropic.claude-sonnet-4-20250514-v1:0")
                .and_then(|m| m.supports_images),
            Some(true)
        );
    }

    #[tokio::test]
    async fn catalog_mode_controls_sources_and_duplicate_precedence() {
        use crate::setup_state::BedrockCatalogMode;

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{
                    "id": "anthropic.shared-model",
                    "context_length": 400000
                }]
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/foundation-models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "modelSummaries": [{
                    "modelId": "anthropic.shared-model",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": true
                }]
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/inference-profiles"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "inferenceProfileSummaries": []
            })))
            .mount(&server)
            .await;

        let mut client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "anthropic.shared-model".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );

        for (mode, expected_context) in [
            (BedrockCatalogMode::MantleOnly, Some(400_000)),
            (BedrockCatalogMode::NativeOnly, None),
            (BedrockCatalogMode::MantlePreferred, Some(400_000)),
            (BedrockCatalogMode::NativePreferred, None),
        ] {
            client.catalog_mode = mode;
            let models = client.list_model_metadata().await.expect("list models");
            let shared = models
                .iter()
                .find(|model| model.id == "anthropic.shared-model")
                .expect("shared model");
            assert_eq!(shared.context_length, expected_context, "mode: {mode:?}");
            assert_eq!(
                models
                    .iter()
                    .filter(|model| model.id == "anthropic.shared-model")
                    .count(),
                1,
                "mode: {mode:?}"
            );
        }
    }

    #[tokio::test]
    async fn bedrock_discovery_retry_retries_and_succeeds_on_later_attempt() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let result = retry_bedrock_discovery("test Bedrock discovery", || {
            let attempts = Arc::clone(&attempts);
            async move {
                if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                    anyhow::bail!("temporary discovery failure");
                }
                Ok("ok")
            }
        })
        .await
        .expect("retry should eventually succeed");

        assert_eq!(result, "ok");
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[derive(Clone)]
    struct FoundationModelsThenThrottle {
        calls: Arc<AtomicUsize>,
    }

    impl Respond for FoundationModelsThenThrottle {
        fn respond(&self, _request: &Request) -> ResponseTemplate {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "modelSummaries": [{
                        "modelId": "anthropic.claude-stable",
                        "inputModalities": ["TEXT"],
                        "outputModalities": ["TEXT"],
                        "responseStreamingSupported": true
                    }]
                }))
            } else {
                ResponseTemplate::new(429).set_body_string("ThrottlingException")
            }
        }
    }

    #[tokio::test]
    async fn bedrock_discovery_uses_last_good_after_later_total_failure() {
        use crate::setup_state::BedrockCatalogMode;

        let server = MockServer::start().await;
        let foundation_calls = Arc::new(AtomicUsize::new(0));
        Mock::given(method("GET"))
            .and(path("/foundation-models"))
            .respond_with(FoundationModelsThenThrottle {
                calls: Arc::clone(&foundation_calls),
            })
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/inference-profiles"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "inferenceProfileSummaries": []
            })))
            .mount(&server)
            .await;

        let mut client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-4-6".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        client.catalog_mode = BedrockCatalogMode::NativeOnly;

        let first = client
            .list_model_metadata()
            .await
            .expect("initial discovery should succeed");
        assert!(first.iter().any(|m| m.id == "anthropic.claude-stable"));

        let second = client
            .list_model_metadata()
            .await
            .expect("stale fallback should still succeed");
        assert!(
            second.iter().any(|m| m.id == "anthropic.claude-stable"),
            "stale last-good model should remain advertised"
        );
        assert_eq!(
            foundation_calls.load(Ordering::SeqCst),
            1 + BEDROCK_DISCOVERY_MAX_ATTEMPTS as usize
        );
    }

    #[test]
    fn bedrock_discovery_without_last_good_returns_empty_after_failure() {
        let cache = LastGoodDiscovery::<Vec<ModelMetadata>>::new("test Bedrock discovery");
        let models = discovery_result_or_last_good(
            &cache,
            Err(DiscoveryFailure {
                err: anyhow::anyhow!("permanent discovery failure"),
                attempts: BEDROCK_DISCOVERY_MAX_ATTEMPTS,
            }),
            Vec::new,
        );

        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn foundation_models_are_rewritten_to_invocable_profile_ids() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/foundation-models"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "modelSummaries": [
                            {
                                "modelId": "anthropic.claude-opus-4-8",
                                "inputModalities": ["TEXT", "IMAGE"],
                                "outputModalities": ["TEXT"],
                                "responseStreamingSupported": true
                            }
                        ]
                    })
                    .to_string(),
                    "application/json",
                ),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/inference-profiles"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "inferenceProfileSummaries": [
                    {
                        "inferenceProfileId": "global.anthropic.claude-opus-4-8",
                        "models": [
                            {
                                "modelArn": "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-opus-4-8"
                            }
                        ]
                    },
                    {
                        "inferenceProfileId": "us.anthropic.claude-opus-4-8",
                        "models": [
                            {
                                "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-opus-4-8"
                            }
                        ]
                    }
                ]
            })))
            .mount(&server)
            .await;

        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "anthropic.claude-opus-4-8".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let models = client
            .list_model_metadata()
            .await
            .expect("discovery should succeed");

        assert!(
            models
                .iter()
                .any(|m| m.id == "us.anthropic.claude-opus-4-8")
        );
        assert!(!models.iter().any(|m| m.id == "anthropic.claude-opus-4-8"));
        assert_eq!(
            models
                .iter()
                .find(|m| m.id == "us.anthropic.claude-opus-4-8")
                .and_then(|m| m.supports_images),
            Some(true)
        );
        assert_eq!(
            models.first().map(|m| m.id.as_str()),
            Some("us.anthropic.claude-opus-4-8")
        );
    }

    #[tokio::test]
    async fn mantle_discovery_failure_still_returns_discovered_native_models() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/foundation-models"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "modelSummaries": [
                            {
                                "modelId": "anthropic.claude-sonnet-4-20250514-v1:0",
                                "inputModalities": ["TEXT"],
                                "outputModalities": ["TEXT"],
                                "responseStreamingSupported": true
                            }
                        ]
                    })
                    .to_string(),
                    "application/json",
                ),
            )
            .mount(&server)
            .await;
        let client = BedrockClient::with_base_urls(
            "token".to_string(),
            "us-east-2".to_string(),
            "us.anthropic.claude-sonnet-4-6".to_string(),
            server.uri(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let models = client
            .list_model_metadata()
            .await
            .expect("discovery should still succeed");
        assert!(
            models
                .iter()
                .any(|m| m.id == "anthropic.claude-sonnet-4-20250514-v1:0")
        );
        assert!(
            models
                .iter()
                .any(|m| m.id == "us.anthropic.claude-sonnet-4-6")
        );
    }

    #[tokio::test]
    async fn discovers_compatible_bedrock_models_and_prioritizes_default() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/foundation-models"))
            .and(header("authorization", "Bearer test-token"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "modelSummaries": [
                            {
                                "modelId": "amazon.nova-micro-v1:0",
                                "inputModalities": ["TEXT"],
                                "outputModalities": ["TEXT"],
                                "responseStreamingSupported": true
                            },
                            {
                                "modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                                "inputModalities": ["TEXT", "IMAGE"],
                                "outputModalities": ["TEXT"],
                                "responseStreamingSupported": true
                            },
                            {
                                "modelId": "anthropic.claude-sonnet-4-20250514-v1:0",
                                "inputModalities": ["TEXT"],
                                "outputModalities": ["TEXT"],
                                "responseStreamingSupported": true
                            },
                            {
                                "modelId": "anthropic.claude-image-v1:0",
                                "inputModalities": ["IMAGE"],
                                "outputModalities": ["IMAGE"],
                                "responseStreamingSupported": false
                            }
                        ]
                    })
                    .to_string(),
                    "application/json",
                ),
            )
            .mount(&server)
            .await;

        let http = reqwest::Client::new();
        let models = discover_model_metadata_with_http(
            &http,
            &format!("{}/foundation-models", server.uri()),
            "test-token",
            "anthropic.claude-sonnet-4-20250514-v1:0",
        )
        .await
        .expect("discovery should succeed");

        let ids: Vec<String> = models.iter().map(|m| m.id.clone()).collect();
        assert_eq!(
            ids,
            vec![
                "anthropic.claude-sonnet-4-20250514-v1:0",
                "anthropic.claude-3-5-sonnet-20240620-v1:0",
            ]
        );
        assert_eq!(
            models
                .iter()
                .find(|m| m.id == "anthropic.claude-3-5-sonnet-20240620-v1:0")
                .and_then(|m| m.supports_images),
            Some(true)
        );
    }

    #[tokio::test]
    async fn falls_back_to_default_model_when_no_compatible_models_exist() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/foundation-models"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(
                    serde_json::json!({
                        "modelSummaries": [
                            {
                                "modelId": "amazon.nova-canvas-v1:0",
                                "inputModalities": ["TEXT"],
                                "outputModalities": ["IMAGE"],
                                "responseStreamingSupported": false
                            }
                        ]
                    })
                    .to_string(),
                    "application/json",
                ),
            )
            .mount(&server)
            .await;

        let http = reqwest::Client::new();
        let models = discover_model_metadata_with_http(
            &http,
            &format!("{}/foundation-models", server.uri()),
            "test-token",
            "us.anthropic.claude-sonnet-4-6",
        )
        .await
        .expect("fallback should succeed");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "us.anthropic.claude-sonnet-4-6");
    }

    #[test]
    fn inference_profile_matching_prefers_geo_then_global() {
        let profiles = vec![
            BedrockInferenceProfileSummary {
                inference_profile_id: "global.anthropic.claude-opus-4-8".to_string(),
                models: vec![BedrockInferenceProfileModel {
                    model_arn:
                        "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-opus-4-8"
                            .to_string(),
                }],
            },
            BedrockInferenceProfileSummary {
                inference_profile_id: "us.anthropic.claude-opus-4-8".to_string(),
                models: vec![BedrockInferenceProfileModel {
                    model_arn:
                        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-opus-4-8"
                            .to_string(),
                }],
            },
            BedrockInferenceProfileSummary {
                inference_profile_id: "eu.anthropic.claude-opus-4-8".to_string(),
                models: vec![BedrockInferenceProfileModel {
                    model_arn:
                        "arn:aws:bedrock:eu-west-1::foundation-model/anthropic.claude-opus-4-8"
                            .to_string(),
                }],
            },
        ];
        let matched =
            match_inference_profiles_to_model("anthropic.claude-opus-4-8", &profiles, "us-east-2");

        assert_eq!(
            matched,
            vec![
                "us.anthropic.claude-opus-4-8".to_string(),
                "global.anthropic.claude-opus-4-8".to_string(),
                "eu.anthropic.claude-opus-4-8".to_string(),
            ]
        );
    }

    #[test]
    fn default_model_is_normalized_to_invocable_profile_id() {
        let profiles = vec![BedrockInferenceProfileSummary {
            inference_profile_id: "global.anthropic.claude-opus-4-8".to_string(),
            models: vec![BedrockInferenceProfileModel {
                model_arn: "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-opus-4-8"
                    .to_string(),
            }],
        }];

        assert_eq!(
            normalize_default_bedrock_model("anthropic.claude-opus-4-8", &profiles, "us-east-2",),
            "global.anthropic.claude-opus-4-8"
        );
    }

    // ---- Responses API continuation (`previous_response_id` chaining) ----
    //
    // Pure-logic tests for `hash_responses_context` / `find_responses_continuation`
    // / `looks_like_expired_previous_response_id`. These don't touch the
    // network, `BedrockClient`, or `responses_chain` directly -- they exercise
    // exactly the matching/hashing logic `invoke_responses_model` relies on to
    // decide what to send. The wiremock round-trip and expiry-fallback tests
    // further down exercise the full `invoke_responses_model` path.

    fn tool_call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            r#type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    #[test]
    fn responses_context_hash_is_content_and_order_sensitive() {
        let a = vec![ChatMessage::user("hello"), ChatMessage::assistant("hi")];
        let a_again = vec![ChatMessage::user("hello"), ChatMessage::assistant("hi")];
        assert_eq!(hash_responses_context(&a), hash_responses_context(&a_again));

        let different_text = vec![ChatMessage::user("hello!"), ChatMessage::assistant("hi")];
        assert_ne!(
            hash_responses_context(&a),
            hash_responses_context(&different_text)
        );

        let reordered = vec![ChatMessage::assistant("hi"), ChatMessage::user("hello")];
        assert_ne!(
            hash_responses_context(&a),
            hash_responses_context(&reordered)
        );

        let shorter = vec![ChatMessage::user("hello")];
        assert_ne!(hash_responses_context(&a), hash_responses_context(&shorter));

        let with_call_1 = vec![
            ChatMessage::user("hello"),
            ChatMessage::assistant_tool_calls(vec![tool_call("call_1", "read_file", "{}")]),
        ];
        let with_call_2 = vec![
            ChatMessage::user("hello"),
            ChatMessage::assistant_tool_calls(vec![tool_call("call_2", "read_file", "{}")]),
        ];
        assert_ne!(
            hash_responses_context(&with_call_1),
            hash_responses_context(&with_call_2),
            "differing tool_call id must change the hash"
        );

        let with_call_1_again = vec![
            ChatMessage::user("hello"),
            ChatMessage::assistant_tool_calls(vec![tool_call("call_1", "read_file", "{}")]),
        ];
        assert_eq!(
            hash_responses_context(&with_call_1),
            hash_responses_context(&with_call_1_again)
        );

        let tool_output = vec![
            ChatMessage::user("hello"),
            ChatMessage::assistant_tool_calls(vec![tool_call("call_1", "read_file", "{}")]),
            ChatMessage::tool_result("call_1", "read_file", "contents-a"),
        ];
        let tool_output_different = vec![
            ChatMessage::user("hello"),
            ChatMessage::assistant_tool_calls(vec![tool_call("call_1", "read_file", "{}")]),
            ChatMessage::tool_result("call_1", "read_file", "contents-b"),
        ];
        assert_ne!(
            hash_responses_context(&tool_output),
            hash_responses_context(&tool_output_different),
            "differing tool_call_id output content must change the hash"
        );
    }

    #[test]
    fn responses_continuation_no_match_returns_none_for_fresh_conversation() {
        let messages = vec![ChatMessage::user("q1")];
        let cache: HashMap<u64, String> = HashMap::new();
        assert!(find_responses_continuation(&messages, |h| cache.get(&h).cloned()).is_none());

        // An assistant turn is present, but nothing about this conversation
        // was ever cached (e.g. evicted, or the process just started).
        let messages_with_assistant = vec![ChatMessage::user("q1"), ChatMessage::assistant("a1")];
        assert!(
            find_responses_continuation(&messages_with_assistant, |h| cache.get(&h).cloned())
                .is_none()
        );
    }

    #[test]
    fn responses_continuation_matches_next_turn_after_storing_previous_turn() {
        // Turn N: this is the full input that produced response A.
        let turn_n = vec![
            ChatMessage::user("q1"),
            ChatMessage::assistant_tool_calls(vec![tool_call("call_1", "read_file", "{}")]),
            ChatMessage::tool_result("call_1", "read_file", "file contents"),
        ];
        let mut cache: HashMap<u64, String> = HashMap::new();
        cache.insert(hash_responses_context(&turn_n), "resp_A".to_string());

        // Turn N+1 = turn-N messages + [assistant echo] + [new tool msg].
        let new_tool_msg = ChatMessage::tool_result("call_2", "write_file", "ok");
        let mut turn_n_plus_1 = turn_n.clone();
        turn_n_plus_1.push(ChatMessage::assistant_tool_calls(vec![tool_call(
            "call_2",
            "write_file",
            "{\"path\":\"a.txt\"}",
        )]));
        turn_n_plus_1.push(new_tool_msg.clone());

        let (boundary, previous_response_id) =
            find_responses_continuation(&turn_n_plus_1, |h| cache.get(&h).cloned())
                .expect("turn N+1 should be detected as a continuation of turn N");

        assert_eq!(previous_response_id, "resp_A");
        assert_eq!(boundary, turn_n.len());
        let delta = &turn_n_plus_1[boundary + 1..];
        assert_eq!(delta, [new_tool_msg]);
    }

    #[test]
    fn responses_continuation_prefers_largest_matching_prefix() {
        let turn0 = vec![ChatMessage::user("q1")];
        let turn1 = {
            let mut m = turn0.clone();
            m.push(ChatMessage::assistant("a1"));
            m.push(ChatMessage::user("q2"));
            m
        };
        let turn2 = {
            let mut m = turn1.clone();
            m.push(ChatMessage::assistant("a2"));
            m.push(ChatMessage::user("q3"));
            m
        };

        // Both turn0 and turn1 are cached; the larger (more recent) prefix
        // should win so the delta we send is as small as possible.
        let mut cache: HashMap<u64, String> = HashMap::new();
        cache.insert(
            hash_responses_context(&turn0),
            "resp_after_turn0".to_string(),
        );
        cache.insert(
            hash_responses_context(&turn1),
            "resp_after_turn1".to_string(),
        );

        let (boundary, previous_response_id) =
            find_responses_continuation(&turn2, |h| cache.get(&h).cloned())
                .expect("should match the larger cached prefix");

        assert_eq!(previous_response_id, "resp_after_turn1");
        assert_eq!(boundary, turn1.len());
    }

    #[test]
    fn responses_continuation_falls_back_to_older_prefix_when_newer_is_evicted() {
        let turn0 = vec![ChatMessage::user("q1")];
        let turn1 = {
            let mut m = turn0.clone();
            m.push(ChatMessage::assistant("a1"));
            m.push(ChatMessage::user("q2"));
            m
        };
        let turn2 = {
            let mut m = turn1.clone();
            m.push(ChatMessage::assistant("a2"));
            m.push(ChatMessage::user("q3"));
            m
        };

        // turn1's entry was evicted (or never stored); only turn0 remains.
        let mut cache: HashMap<u64, String> = HashMap::new();
        cache.insert(
            hash_responses_context(&turn0),
            "resp_after_turn0".to_string(),
        );

        let (boundary, previous_response_id) =
            find_responses_continuation(&turn2, |h| cache.get(&h).cloned())
                .expect("should fall back to the older cached prefix");

        assert_eq!(previous_response_id, "resp_after_turn0");
        assert_eq!(boundary, turn0.len());
    }

    #[test]
    fn responses_chain_cache_evicts_oldest_beyond_cap() {
        let mut cache = ResponsesChainCache::new(2);
        cache.insert(1, "a".to_string());
        cache.insert(2, "b".to_string());
        cache.insert(3, "c".to_string());

        assert_eq!(cache.get(1), None, "oldest entry should have been evicted");
        assert_eq!(cache.get(2), Some("b".to_string()));
        assert_eq!(cache.get(3), Some("c".to_string()));
    }

    #[test]
    fn responses_chain_cache_evict_removes_entry() {
        let mut cache = ResponsesChainCache::new(8);
        cache.insert(1, "a".to_string());
        cache.evict(1);
        assert_eq!(cache.get(1), None);
    }

    #[test]
    fn responses_chain_cache_evict_context_prefixes_removes_lookup_keys() {
        let turn0 = vec![ChatMessage::user("q1")];
        let turn1 = {
            let mut m = turn0.clone();
            m.push(ChatMessage::assistant("a1"));
            m.push(ChatMessage::user("q2"));
            m
        };
        let turn2 = {
            let mut m = turn1.clone();
            m.push(ChatMessage::assistant("a2"));
            m.push(ChatMessage::user("q3"));
            m
        };

        let mut cache = ResponsesChainCache::new(8);
        cache.insert(
            hash_responses_context(&turn0),
            "resp_after_turn0".to_string(),
        );
        cache.insert(
            hash_responses_context(&turn1),
            "resp_after_turn1".to_string(),
        );
        cache.insert(42, "unrelated".to_string());

        let (boundary, previous_response_id) =
            find_responses_continuation(&turn2, |h| cache.get(h))
                .expect("largest cached prefix should be usable before eviction");
        assert_eq!(previous_response_id, "resp_after_turn1");

        let lookup_key = hash_responses_context(&turn2[..boundary]);
        assert_eq!(cache.get(lookup_key), Some("resp_after_turn1".to_string()));

        assert_eq!(cache.evict_context_prefixes(&turn2), 2);
        assert_eq!(cache.get(lookup_key), None);
        assert_eq!(cache.get(42), Some("unrelated".to_string()));
        assert!(
            find_responses_continuation(&turn2, |h| cache.get(h)).is_none(),
            "all usable cached prefixes for this context should be gone"
        );
    }

    #[test]
    fn looks_like_expired_previous_response_id_matches_defensively() {
        assert!(looks_like_expired_previous_response_id(
            reqwest::StatusCode::BAD_REQUEST,
            "{\"error\":{\"message\":\"previous_response_id 'resp_123' not found\"}}"
        ));
        assert!(looks_like_expired_previous_response_id(
            reqwest::StatusCode::NOT_FOUND,
            "{\"error\":{\"message\":\"Unknown previous_response_id\"}}"
        ));
        assert!(looks_like_expired_previous_response_id(
            reqwest::StatusCode::BAD_REQUEST,
            "{\"error\":{\"message\":\"previous_response_id has expired\"}}"
        ));
        assert!(looks_like_expired_previous_response_id(
            reqwest::StatusCode::NOT_FOUND,
            "{\"error\":{\"code\":\"not_found_error\",\"message\":\"Response 'resp_akkecf7gb36b3mffg2eifdikhj3cmi6fn7p3mh3muslnh6fbryxa' not found.\",\"param\":null,\"type\":\"invalid_request_error\"}}"
        ));

        // Not client errors, or no mention of previous_response_id, or no
        // rejection wording -- must not match.
        assert!(!looks_like_expired_previous_response_id(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            "previous_response_id not found"
        ));
        assert!(!looks_like_expired_previous_response_id(
            reqwest::StatusCode::BAD_REQUEST,
            "{\"error\":{\"message\":\"model not found\"}}"
        ));
        assert!(!looks_like_expired_previous_response_id(
            reqwest::StatusCode::BAD_REQUEST,
            "{\"error\":{\"message\":\"previous_response_id is required\"}}"
        ));
    }
}
