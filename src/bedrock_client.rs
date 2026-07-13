use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

#[cfg(test)]
use crate::llm_client::IdleTimeouts;
use crate::llm_client::{
    ChatContentPart, ChatMessage, FunctionCall, LlmBackend, LlmResponse, ModelMetadata,
    ModelsResponse, OpenAiClient, ReasoningLevelPreset, StreamChatRequest, TokenUsage, ToolCall,
    ToolDefinition,
};
use crate::responses_api::{build_responses_request, drive_responses_sse_stream};
use crate::trace_logging::append_trace_record;
use anyhow::{Context, Result};
use futures::StreamExt;
use futures::future::BoxFuture;
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
    if is_prefixed_bedrock_model(&model, &["openai.gpt-5.5", "openai.gpt-5.4"]) {
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
            request.temperature = None;
            request.thinking = Some(BedrockThinking::Adaptive);
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

impl BedrockClient {
    pub fn new(bearer_token: String, region: String, default_model: String) -> Self {
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
            catalog_mode: crate::setup_state::bedrock_catalog_mode(),
            thinking_shape_cache: Arc::new(RwLock::new(HashMap::new())),
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
        }
    }

    fn invoke_url(&self, model: &str) -> String {
        let encoded = percent_encode_path_segment(model);
        if self.runtime_base_url.starts_with("http://")
            || self.runtime_base_url.starts_with("https://")
        {
            if self.runtime_base_url == "https://bedrock-runtime"
                || self.runtime_base_url == "http://bedrock-runtime"
            {
                return format!(
                    "{}.{}.amazonaws.com/model/{encoded}/invoke",
                    self.runtime_base_url.trim_end_matches('.'),
                    self.region,
                );
            }
            if self.runtime_base_url.contains("amazonaws.com") {
                return format!(
                    "{}/model/{encoded}/invoke",
                    self.runtime_base_url.trim_end_matches('/')
                );
            }
            return format!(
                "{}/model/{encoded}/invoke",
                self.runtime_base_url.trim_end_matches('/')
            );
        }
        format!(
            "{}.{}.amazonaws.com/model/{encoded}/invoke",
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
            idle_timeouts: _,
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

        let url = self.invoke_url(&resolved_model);
        let body_text = loop {
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
                .invoke_native_anthropic_with_fallback(&resolved_model, &url, &body, &cancel)
                .await
            {
                Ok(text) => {
                    if effort.is_some() {
                        self.thinking_shape_cache
                            .write()
                            .await
                            .insert(resolved_model.clone(), shape);
                    }
                    break text;
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
        if body_text.is_empty() && cancel.is_cancelled() {
            return Ok(LlmResponse::Text {
                text: String::new(),
                reasoning_content: None,
                usage: TokenUsage::default(),
            });
        }
        let parsed: BedrockAnthropicResponse =
            serde_json::from_str(&body_text).context("parse Bedrock response")?;
        let mut text = String::new();
        let mut thoughts = String::new();
        let mut calls = Vec::new();
        for block in parsed.content {
            match block {
                BedrockContentBlock::Text { text: part } => text.push_str(&part),
                BedrockContentBlock::Thinking { thinking: part } => thoughts.push_str(&part),
                BedrockContentBlock::ToolUse { id, name, input } => {
                    calls.push(ToolCall {
                        id,
                        r#type: "function".to_string(),
                        function: FunctionCall {
                            name,
                            arguments: input.to_string(),
                        },
                    });
                }
                BedrockContentBlock::Other => {}
            }
        }
        if !thoughts.is_empty() {
            on_thought(&thoughts);
        }
        if !text.is_empty() {
            on_token(&text);
        }
        let usage = parsed.usage.into_usage();
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
        let body = build_responses_request(
            &model,
            &messages,
            tools.as_deref(),
            reasoning_effort.as_deref(),
            structured_output.as_ref(),
        );
        let url = format!(
            "{}/responses",
            mantle_base_url_for_model(&self.mantle_base_url, &model)
        );
        let resp = crate::http_retry::send_with_retries(
            "posting Bedrock Responses API request",
            || {
                self.http
                    .post(&url)
                    .header("Accept", "text/event-stream")
                    .bearer_auth(&self.bearer_token)
                    .json(&body)
            },
            Some(&cancel),
        )
        .await?;
        let status = resp.status();
        if !status.is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(crate::http_retry::retryable_llm_error_for_body(
                format!("Bedrock Responses API failed (HTTP {status}): {body_text}"),
                &body_text,
            ));
        }
        let stream = resp
            .bytes_stream()
            .map(|r| r.map(|b| b.to_vec()).map_err(anyhow::Error::from));
        drive_responses_sse_stream(stream, on_token, on_thought, cancel, idle_timeouts).await
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

    async fn invoke_native_anthropic_with_fallback(
        &self,
        model: &str,
        url: &str,
        body: &BedrockAnthropicRequest,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> Result<String> {
        let Some((status, body_text)) = self.post_native_invoke(url, body, cancel).await? else {
            return Ok(String::new());
        };
        if status.is_success() {
            return Ok(body_text);
        }

        if !needs_inference_profile_retry(status, &body_text)
            || looks_like_inference_profile_identifier(model)
        {
            return Err(crate::http_retry::retryable_llm_error_for_body(
                format!("Bedrock request failed (HTTP {status}): {body_text}"),
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
            let retry_url = self.invoke_url(&profile_id);
            let Some((retry_status, retry_body)) =
                self.post_native_invoke(&retry_url, body, cancel).await?
            else {
                return Ok(String::new());
            };
            if retry_status.is_success() {
                return Ok(retry_body);
            }
            last_retry_error = Some((profile_id, retry_status, retry_body));
        }

        if let Some((profile_id, retry_status, retry_body)) = last_retry_error {
            let message = format!(
                "Bedrock request failed (HTTP {status}): {body_text}; retry with inference profile {} also failed (HTTP {retry_status}): {retry_body}",
                profile_id
            );
            let bodies = format!("{body_text}\n{retry_body}");
            return Err(crate::http_retry::retryable_llm_error_for_body(
                message, &bodies,
            ));
        }

        Err(crate::http_retry::retryable_llm_error_for_body(
            format!("Bedrock request failed (HTTP {status}): {body_text}"),
            &body_text,
        ))
    }

    async fn post_native_invoke(
        &self,
        url: &str,
        body: &BedrockAnthropicRequest,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> Result<Option<(reqwest::StatusCode, String)>> {
        let send = self
            .http
            .post(url)
            .bearer_auth(&self.bearer_token)
            .json(body)
            .send();

        let resp = tokio::select! {
            _ = cancel.cancelled() => {
                return Ok(None);
            }
            resp = send => resp.context("failed to send Bedrock request")?,
        };

        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        Ok(Some((status, body_text)))
    }

    async fn inference_profile_candidates_for_model(&self, model: &str) -> Vec<String> {
        let mut candidates = self
            .list_inference_profiles()
            .await
            .map(|profiles| match_inference_profiles_to_model(model, &profiles, &self.region))
            .unwrap_or_else(|err| {
                tracing::info!("Bedrock inference profile discovery skipped: {err:#}");
                Vec::new()
            });
        candidates.extend(guessed_inference_profile_candidates(model, &self.region));
        dedup_preserve_order(candidates)
    }

    async fn resolve_invocable_model_id(&self, model: &str) -> Result<String> {
        if looks_like_inference_profile_identifier(model) || uses_responses_api(model) {
            return Ok(model.to_string());
        }

        let profiles = self.list_inference_profiles().await.unwrap_or_else(|err| {
            tracing::info!("Bedrock inference profile discovery skipped: {err:#}");
            Vec::new()
        });
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

            let discover_mantle = || async {
                match self.list_mantle_model_metadata().await {
                    Ok(models) => models,
                    Err(err) => {
                        tracing::info!("Bedrock Mantle model discovery skipped: {err:#}");
                        Vec::new()
                    }
                }
            };
            let discover_native = || async {
                match self.discover_model_metadata().await {
                    Ok(models) => models,
                    Err(err) => {
                        tracing::info!("Bedrock foundation-model discovery skipped: {err:#}");
                        Vec::new()
                    }
                }
            };

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
                match self.list_inference_profiles().await {
                    Ok(profiles) => profiles,
                    Err(err) => {
                        tracing::info!("Bedrock inference profile discovery skipped: {err:#}");
                        Vec::new()
                    }
                }
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

/// The gpt-5.4 and gpt-5.5 model families are served from a dedicated
/// `/openai/v1` path on Bedrock Mantle rather than the shared `/v1` path every
/// other Mantle model uses (per their Bedrock model cards). Match the bare ids
/// and any suffixed derivative (e.g. `openai.gpt-5.5-codex`), but not unrelated
/// ids that merely share the numeric prefix (e.g. a future `openai.gpt-5.40`).
fn uses_openai_mantle_path(model: &str) -> bool {
    ["openai.gpt-5.4", "openai.gpt-5.5"]
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

pub fn build_backend_from_config() -> Result<Option<Arc<dyn LlmBackend>>> {
    let Some(token) = bearer_token_from_env_or_secrets()? else {
        return Ok(None);
    };
    Ok(Some(Arc::new(BedrockClient::new(
        token,
        region_from_env(),
        model_from_env(),
    ))))
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
    #[serde(rename = "adaptive")]
    Adaptive,
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

#[derive(Debug, Deserialize)]
struct BedrockAnthropicResponse {
    #[serde(default)]
    content: Vec<BedrockContentBlock>,
    #[serde(default)]
    usage: BedrockUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum BedrockContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(other)]
    Other,
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
                if let Some(BedrockContentOut::Text { cache_control, .. }) = msg.content.last_mut()
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

#[derive(Debug, Deserialize)]
struct BedrockInferenceProfileSummary {
    #[serde(rename = "inferenceProfileId")]
    inference_profile_id: String,
    #[serde(default)]
    models: Vec<BedrockInferenceProfileModel>,
}

#[derive(Debug, Deserialize)]
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
    use std::time::Duration;

    use super::*;
    use tokio_util::sync::CancellationToken;
    use wiremock::matchers::{header, method, path, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

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
            client.invoke_url("us.anthropic.claude-sonnet-4-6"),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4-6/invoke"
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
            client.invoke_url("a/b c"),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/a%2Fb%20c/invoke"
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
            .and(path("/model/us.anthropic.claude-sonnet-4-6/invoke"))
            .and(body_partial_json(serde_json::json!({
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "medium"}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [
                    {"type": "thinking", "thinking": "let me reason"},
                    {"type": "text", "text": "done"}
                ],
                "usage": {"input_tokens": 5, "output_tokens": 3}
            })))
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
            .and(path("/model/us.anthropic.claude-3-5-sonnet/invoke"))
            .and(GenericThinkingBlock)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [{"type": "text", "text": "plain"}],
                "usage": {"input_tokens": 2, "output_tokens": 1}
            })))
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
            .and(path("/model/global.anthropic.claude-sonnet-4-5/invoke"))
            .and(ManualThinkingOnly)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [{"type": "text", "text": "manual ok"}],
                "usage": {"input_tokens": 2, "output_tokens": 1}
            })))
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
            .and(path("/model/global.anthropic.claude-opus-4-8/invoke"))
            .and(body_partial_json(serde_json::json!({
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "xhigh"}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [{"type": "text", "text": "xhigh ok"}],
                "usage": {"input_tokens": 2, "output_tokens": 1}
            })))
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
            .and(path("/model/anthropic.claude-opus-4-5/invoke"))
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
            .and(path("/model/anthropic.claude-opus-4-5/invoke"))
            .and(AdaptiveShape(adaptive_hits.clone()))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [
                    {"type": "thinking", "thinking": "weighing"},
                    {"type": "text", "text": "adaptive ok"}
                ],
                "usage": {"input_tokens": 6, "output_tokens": 4}
            })))
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
            .and(path_regex(r"/model/.+/invoke$"))
            .and(header("authorization", "Bearer token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [{"type": "text", "text": "native ok"}],
                "usage": {"input_tokens": 2, "output_tokens": 1}
            })))
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
                assert_eq!(usage.input_tokens, 2);
            }
            other => panic!("expected text response, got {other:?}"),
        }
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
            .and(path("/model/anthropic.claude-opus-4-8/invoke"))
            .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "message": "Invocation of model ID anthropic.claude-opus-4-8 with on-demand throughput isn’t supported. Retry your request with the ID or ARN of an inference profile that contains this model."
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/model/global.anthropic.claude-opus-4-8/invoke"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [{"type": "text", "text": "profile ok"}],
                "usage": {"input_tokens": 4, "output_tokens": 2}
            })))
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
            .and(path("/model/us.anthropic.claude-opus-4-8/invoke"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [{"type": "text", "text": "resolved first"}],
                "usage": {"input_tokens": 3, "output_tokens": 2}
            })))
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
}
