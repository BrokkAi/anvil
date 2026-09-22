//! Xiaomi MiMo's OpenAI-compatible Responses APIs.
//!
//! Token Plan and pay-as-you-go deliberately use different base URLs and key
//! prefixes. Keeping them as explicit plan routes prevents subscription traffic
//! from accidentally hitting the billed endpoint (or vice versa).

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use futures::{StreamExt, future::BoxFuture};

use crate::llm_client::{LlmBackend, LlmResponse, ModelMetadata, StreamChatRequest};
use crate::responses_api::{build_responses_request, drive_responses_sse_stream};

pub const MIMO_TOKEN_PLAN_API_KEY_ENV: &str = "MIMO_TOKEN_PLAN_API_KEY";
/// Xiaomi's integration examples use this name for both MiMo plans. Anvil
/// accepts it as a fallback, but the dedicated variable above makes plan
/// selection unambiguous when both credentials exist in the same shell.
pub const MIMO_API_KEY_ENV: &str = "MIMO_API_KEY";
pub const MIMO_TOKEN_PLAN_BASE_URL_ENV: &str = "MIMO_TOKEN_PLAN_BASE_URL";
pub const MIMO_TOKEN_PLAN_BASE_URL: &str = "https://token-plan-cn.xiaomimimo.com/v1";
pub const MIMO_PAY_AS_YOU_GO_API_KEY_ENV: &str = "MIMO_PAY_AS_YOU_GO_API_KEY";
pub const MIMO_PAY_AS_YOU_GO_BASE_URL_ENV: &str = "MIMO_PAY_AS_YOU_GO_BASE_URL";
pub const MIMO_PAY_AS_YOU_GO_BASE_URL: &str = "https://api.xiaomimimo.com/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MimoPlan {
    TokenPlan,
    PayAsYouGo,
}

impl MimoPlan {
    fn label(self) -> &'static str {
        match self {
            Self::TokenPlan => "Xiaomi MiMo Token Plan",
            Self::PayAsYouGo => "Xiaomi MiMo pay-as-you-go",
        }
    }

    fn default_base_url(self) -> &'static str {
        match self {
            Self::TokenPlan => MIMO_TOKEN_PLAN_BASE_URL,
            Self::PayAsYouGo => MIMO_PAY_AS_YOU_GO_BASE_URL,
        }
    }

    fn base_url_env(self) -> &'static str {
        match self {
            Self::TokenPlan => MIMO_TOKEN_PLAN_BASE_URL_ENV,
            Self::PayAsYouGo => MIMO_PAY_AS_YOU_GO_BASE_URL_ENV,
        }
    }

    fn dedicated_api_key_env(self) -> &'static str {
        match self {
            Self::TokenPlan => MIMO_TOKEN_PLAN_API_KEY_ENV,
            Self::PayAsYouGo => MIMO_PAY_AS_YOU_GO_API_KEY_ENV,
        }
    }

    fn generic_api_key_matches(self, key: &str) -> bool {
        match self {
            Self::TokenPlan => key.starts_with("tp-"),
            Self::PayAsYouGo => key.starts_with("sk-"),
        }
    }
}

pub struct MimoClient {
    plan: MimoPlan,
    http: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl MimoClient {
    /// Load a dedicated plan variable first. Xiaomi's generic `MIMO_API_KEY`
    /// is accepted only when its documented key prefix identifies the plan.
    pub fn load(plan: MimoPlan) -> Result<Option<Arc<dyn LlmBackend>>> {
        let key = select_api_key(plan, |name| std::env::var(name).ok());
        let Some(key) = key else {
            return Ok(None);
        };

        let base_url = match std::env::var(plan.base_url_env()) {
            Ok(value) => {
                let value = value.trim().to_string();
                if value.is_empty() {
                    bail!("{} is set but empty", plan.base_url_env());
                }
                value
            }
            Err(_) => plan.default_base_url().to_string(),
        };
        Ok(Some(
            Arc::new(Self::new(plan, base_url, key)?) as Arc<dyn LlmBackend>
        ))
    }

    /// Explicit endpoint construction also supports local wire-level tests.
    pub fn new(
        plan: MimoPlan,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self {
            plan,
            http: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(20))
                .build()
                .context("building Xiaomi MiMo Responses client")?,
            api_key: api_key.into().trim().to_string(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
        })
    }

    fn responses_url(&self) -> String {
        if self.base_url.ends_with("/v1") {
            format!("{}/responses", self.base_url)
        } else {
            format!("{}/v1/responses", self.base_url)
        }
    }

    async fn invoke(&self, request: StreamChatRequest) -> Result<LlmResponse> {
        let StreamChatRequest {
            model,
            messages,
            tools,
            reasoning_effort,
            structured_output,
            on_token,
            on_thought,
            cancel,
            idle_timeouts,
            ..
        } = request;
        let effort = reasoning_effort.as_deref().map(mimo_reasoning_effort);
        let mut body = build_responses_request(
            &model,
            &messages,
            tools.as_deref(),
            effort,
            structured_output.as_ref(),
            false,
            None,
        );
        // Xiaomi's model catalog explicitly marks parallel tool calls
        // unsupported. Keep the wire request sequential even when the agent
        // loop is willing to execute a returned batch concurrently.
        body.parallel_tool_calls = false;

        let response = crate::http_retry::send_with_retries(
            "posting Xiaomi MiMo Responses request",
            || {
                self.http
                    .post(self.responses_url())
                    .bearer_auth(&self.api_key)
                    .header("Accept", "text/event-stream")
                    .json(&body)
            },
            Some(&cancel),
            Some(idle_timeouts.first_progress),
        )
        .await?;
        let status = response.status();
        if !status.is_success() {
            let body_text = read_limited_error_body(response).await;
            return Err(crate::http_retry::retryable_llm_error_for_status_and_body(
                format!("{} Responses API failed (HTTP {status})", self.plan.label()),
                status,
                &body_text,
            ));
        }

        let stream = response
            .bytes_stream()
            .map(|chunk| chunk.map(|b| b.to_vec()).map_err(anyhow::Error::from));
        let outcome =
            drive_responses_sse_stream(stream, on_token, on_thought, cancel.clone(), idle_timeouts)
                .await?;
        if cancel.is_cancelled() {
            bail!("{} request cancelled", self.plan.label());
        }
        if outcome.incomplete {
            bail!("{} output was incomplete", self.plan.label());
        }
        Ok(outcome.response)
    }
}

fn select_api_key(plan: MimoPlan, lookup: impl Fn(&str) -> Option<String>) -> Option<String> {
    if let Some(key) = lookup(plan.dedicated_api_key_env())
        .map(|key| key.trim().to_string())
        .filter(|key| !key.is_empty())
    {
        return Some(key);
    }

    let generic = lookup(MIMO_API_KEY_ENV)
        .map(|key| key.trim().to_string())
        .filter(|key| !key.is_empty())?;
    if plan.generic_api_key_matches(&generic) {
        Some(generic)
    } else {
        tracing::info!(
            "{} is set but its key prefix does not identify {}; backend skipped",
            MIMO_API_KEY_ENV,
            plan.label()
        );
        None
    }
}

async fn read_limited_error_body(response: reqwest::Response) -> String {
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    while let Some(Ok(chunk)) = stream.next().await {
        let remaining = 64 * 1024 - bytes.len();
        bytes.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
        if bytes.len() == 64 * 1024 {
            break;
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// MiMo advertises `none`, `low`, `medium`, and `high`. Map the larger
/// Anvil effort vocabulary onto that set before the first request.
fn mimo_reasoning_effort(effort: &str) -> &'static str {
    match effort.trim().to_ascii_lowercase().as_str() {
        "none" => "none",
        "minimal" | "low" => "low",
        "medium" => "medium",
        _ => "high",
    }
}

const MIMO_REASONING_PRESETS: &[(&str, &str)] = &[
    ("none", "No extra reasoning for faster responses"),
    ("low", "Fast responses with lighter reasoning"),
    (
        "medium",
        "Balances speed and reasoning depth for everyday tasks",
    ),
    ("high", "Greater reasoning depth for complex problems"),
];

fn mimo_reasoning_presets() -> Vec<crate::llm_client::ReasoningLevelPreset> {
    MIMO_REASONING_PRESETS
        .iter()
        .map(
            |(effort, description)| crate::llm_client::ReasoningLevelPreset {
                effort: (*effort).to_string(),
                description: (*description).to_string(),
            },
        )
        .collect()
}

/// Xiaomi's `/models` response is OpenAI-shaped and may not expose all of
/// the capabilities published in its Codex model catalog. Enrich known IDs
/// while preserving newly introduced models from the live endpoint.
fn enrich_mimo_metadata(mut metadata: ModelMetadata, plan: MimoPlan) -> ModelMetadata {
    let known = matches!(
        metadata.id.as_str(),
        "mimo-v2.6-pro"
            | "mimo-v2.6-flash"
            | "mimo-v2.6-pro-ultraspeed"
            | "mimo-v2.5-pro"
            | "mimo-v2.5"
    );
    if known {
        metadata.default_reasoning_level = Some("low".to_string());
        metadata.supported_reasoning_levels = mimo_reasoning_presets();
        metadata.context_length = Some(1_048_576);
        if plan == MimoPlan::TokenPlan {
            metadata.pricing = None;
        }
        if metadata.id != "mimo-v2.5-pro" {
            metadata.supports_images = Some(true);
        } else {
            metadata.supports_images = Some(false);
        }
    }
    metadata
}

impl LlmBackend for MimoClient {
    fn supports_native_structured_output(&self) -> bool {
        true
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
        Box::pin(async move {
            crate::llm_client::OpenAiClient::new(self.base_url.clone(), Some(self.api_key.clone()))
                .list_models()
                .await
        })
    }

    fn list_model_metadata(&self) -> BoxFuture<'_, Result<Vec<ModelMetadata>>> {
        Box::pin(async move {
            let models = crate::llm_client::OpenAiClient::new(
                self.base_url.clone(),
                Some(self.api_key.clone()),
            )
            .list_model_metadata()
            .await?;
            Ok(models
                .into_iter()
                .map(|metadata| enrich_mimo_metadata(metadata, self.plan))
                .collect())
        })
    }

    fn stream_chat(&self, request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
        Box::pin(self.invoke(request))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_client::{ChatMessage, IdleTimeouts};
    use serde_json::json;
    use tokio_util::sync::CancellationToken;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn request(cancel: CancellationToken) -> StreamChatRequest {
        StreamChatRequest {
            model: "mimo-v2.6-pro".to_string(),
            messages: vec![
                ChatMessage::system("stable rules"),
                ChatMessage::user("summarize the issue"),
            ],
            tools: None,
            reasoning_effort: Some("max".to_string()),
            service_tier: None,
            temperature: None,
            structured_output: None,
            on_token: Box::new(|_| {}),
            on_thought: Box::new(|_| {}),
            cancel,
            idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(2)),
        }
    }

    fn completed() -> String {
        format!(
            "data: {}\n\ndata: {}\n\n",
            json!({"type":"response.output_text.delta","delta":"done"}),
            json!({"type":"response.completed","response":{"id":"resp_test","usage":{"input_tokens":20,"output_tokens":5}}})
        )
    }

    #[test]
    fn mimo_efforts_are_clamped_to_documented_levels() {
        for (requested, expected) in [
            ("none", "none"),
            ("minimal", "low"),
            ("medium", "medium"),
            ("xhigh", "high"),
            ("max", "high"),
        ] {
            assert_eq!(mimo_reasoning_effort(requested), expected);
        }
    }

    #[test]
    fn plan_urls_stay_separate() {
        let token_plan =
            MimoClient::new(MimoPlan::TokenPlan, MIMO_TOKEN_PLAN_BASE_URL, "tp-test").unwrap();
        assert_eq!(
            token_plan.responses_url(),
            "https://token-plan-cn.xiaomimimo.com/v1/responses"
        );

        let pay_as_you_go =
            MimoClient::new(MimoPlan::PayAsYouGo, MIMO_PAY_AS_YOU_GO_BASE_URL, "sk-test").unwrap();
        assert_eq!(
            pay_as_you_go.responses_url(),
            "https://api.xiaomimimo.com/v1/responses"
        );

        let custom_origin = MimoClient::new(
            MimoPlan::TokenPlan,
            "https://token-plan.example.test",
            "tp-test",
        )
        .unwrap();
        assert_eq!(
            custom_origin.responses_url(),
            "https://token-plan.example.test/v1/responses"
        );
    }

    #[test]
    fn generic_credentials_select_only_their_documented_plan() {
        let lookup = |name: &str| match name {
            MIMO_TOKEN_PLAN_API_KEY_ENV => None,
            MIMO_PAY_AS_YOU_GO_API_KEY_ENV => None,
            MIMO_API_KEY_ENV => Some("tp-token-plan".to_string()),
            _ => None,
        };
        assert_eq!(
            select_api_key(MimoPlan::TokenPlan, lookup),
            Some("tp-token-plan".to_string())
        );
        assert_eq!(select_api_key(MimoPlan::PayAsYouGo, lookup), None);

        let lookup = |name: &str| match name {
            MIMO_TOKEN_PLAN_API_KEY_ENV => None,
            MIMO_PAY_AS_YOU_GO_API_KEY_ENV => None,
            MIMO_API_KEY_ENV => Some("sk-payg".to_string()),
            _ => None,
        };
        assert_eq!(select_api_key(MimoPlan::TokenPlan, lookup), None);
        assert_eq!(
            select_api_key(MimoPlan::PayAsYouGo, lookup),
            Some("sk-payg".to_string())
        );
    }

    #[test]
    fn dedicated_credentials_take_precedence_over_generic_keys() {
        let lookup = |name: &str| match name {
            MIMO_TOKEN_PLAN_API_KEY_ENV => Some("tp-dedicated".to_string()),
            MIMO_PAY_AS_YOU_GO_API_KEY_ENV => Some("sk-dedicated".to_string()),
            MIMO_API_KEY_ENV => Some("sk-generic".to_string()),
            _ => None,
        };
        assert_eq!(
            select_api_key(MimoPlan::TokenPlan, lookup),
            Some("tp-dedicated".to_string())
        );
        assert_eq!(
            select_api_key(MimoPlan::PayAsYouGo, lookup),
            Some("sk-dedicated".to_string())
        );
    }

    #[tokio::test]
    async fn responses_request_targets_token_plan_and_disables_parallel_tools() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("Authorization", "Bearer tp-test-key"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(completed()),
            )
            .mount(&server)
            .await;

        let client = MimoClient::new(MimoPlan::TokenPlan, server.uri(), "tp-test-key").unwrap();
        client
            .stream_chat(request(CancellationToken::new()))
            .await
            .unwrap();

        let requests = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(body["reasoning"]["effort"], "high");
        assert_eq!(body["store"], false);
    }

    #[tokio::test]
    async fn model_discovery_is_enriched_without_pay_as_you_go_pricing() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .and(header("Authorization", "Bearer tp-test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [
                    {"id": "mimo-v2.6-pro", "object": "model", "pricing": {"prompt": "0.0000036", "completion": "0.00000087"}},
                    {"id": "future-mimo", "object": "model"}
                ]
            })))
            .mount(&server)
            .await;

        let client = MimoClient::new(MimoPlan::TokenPlan, server.uri(), "tp-test-key").unwrap();
        let models = client.list_model_metadata().await.unwrap();
        assert_eq!(models.len(), 2);

        let known = &models[0];
        assert_eq!(known.default_reasoning_level.as_deref(), Some("low"));
        assert_eq!(known.context_length, Some(1_048_576));
        assert_eq!(known.supports_images, Some(true));
        assert!(known.pricing.is_none());
        assert_eq!(
            known
                .supported_reasoning_levels
                .iter()
                .map(|preset| preset.effort.as_str())
                .collect::<Vec<_>>(),
            vec!["none", "low", "medium", "high"]
        );

        let unknown = &models[1];
        assert_eq!(unknown.id, "future-mimo");
        assert!(unknown.default_reasoning_level.is_none());
    }

    #[tokio::test]
    async fn pay_as_you_go_discovery_preserves_provider_pricing() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .and(header("Authorization", "Bearer sk-test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [
                    {"id": "mimo-v2.6-pro", "object": "model", "pricing": {"prompt": "0.0000036", "completion": "0.00000087"}}
                ]
            })))
            .mount(&server)
            .await;

        let client = MimoClient::new(MimoPlan::PayAsYouGo, server.uri(), "sk-test-key").unwrap();
        let models = client.list_model_metadata().await.unwrap();
        assert_eq!(
            models[0].pricing.map(|pricing| (
                pricing.input_cost_per_token_usd,
                pricing.output_cost_per_token_usd
            )),
            Some((0.0000036, 0.00000087))
        );
    }
}
