use std::sync::Arc;

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use futures::future::BoxFuture;
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::Value;

use crate::grok_auth::{GrokAuthManager, GrokCredential};
use crate::llm_client::{
    LlmBackend, LlmResponse, ModelMetadata, ReasoningLevelPreset, ResolvedModelInfo,
    StreamChatRequest,
};
use crate::responses_api::{ReasoningConfig, build_responses_request, drive_responses_sse_stream};

const GROK_API_BASE_URL: &str = "https://cli-chat-proxy.grok.com/v1";

pub(crate) struct GrokClient {
    auth: Arc<GrokAuthManager>,
    http: reqwest::Client,
    base_url: String,
    client_version: String,
}

impl GrokClient {
    pub(crate) fn load() -> Result<Option<Arc<dyn LlmBackend>>> {
        let Some(auth) = GrokAuthManager::load()? else {
            return Ok(None);
        };
        Ok(Some(Arc::new(Self {
            auth,
            http: reqwest::Client::new(),
            base_url: GROK_API_BASE_URL.to_string(),
            client_version: crate::grok_auth::client_version(),
        })))
    }

    fn request_headers(
        &self,
        request: reqwest::RequestBuilder,
        credential: &GrokCredential,
    ) -> reqwest::RequestBuilder {
        let mut request = request
            .bearer_auth(&credential.access_token)
            .header("X-XAI-Token-Auth", "xai-grok-cli")
            .header("x-authenticateresponse", "authenticate-response")
            .header("x-grok-client-version", &self.client_version)
            .header("x-grok-client-identifier", "anvil")
            .header("x-grok-client-mode", "headless")
            .header("User-Agent", format!("anvil/{}", env!("CARGO_PKG_VERSION")));
        request = request.header("x-userid", &credential.user_id);
        if let Some(email) = credential
            .email
            .as_deref()
            .filter(|email| !email.is_empty())
        {
            request = request.header("x-email", email);
        }
        request
    }

    async fn models_response(&self) -> Result<reqwest::Response> {
        let credential = self.auth.credential().await?;
        let response = self
            .request_headers(
                self.http.get(format!("{}/models", self.base_url)),
                &credential,
            )
            .send()
            .await
            .context("discovering Grok models")?;
        if response.status() != StatusCode::UNAUTHORIZED {
            return Ok(response);
        }
        let refreshed = self.auth.refresh_rejected(&credential.access_token).await?;
        self.request_headers(
            self.http.get(format!("{}/models", self.base_url)),
            &refreshed,
        )
        .send()
        .await
        .context("discovering Grok models after OAuth refresh")
    }

    async fn discover_models(&self) -> Result<Vec<ModelMetadata>> {
        let response = self.models_response().await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            bail!("Grok model discovery failed (HTTP {status}): {body}")
        }
        let payload = response
            .json::<ModelsResponse>()
            .await
            .context("parsing Grok model catalog")?;
        Ok(payload
            .data
            .into_iter()
            .filter_map(model_metadata)
            .collect())
    }

    async fn post_responses(
        &self,
        body: &crate::responses_api::ResponsesRequest,
        credential: &GrokCredential,
        cancel: &tokio_util::sync::CancellationToken,
        first_progress: std::time::Duration,
    ) -> Result<reqwest::Response> {
        let conversation_id = uuid::Uuid::new_v4().to_string();
        let request_id = uuid::Uuid::new_v4().to_string();
        crate::http_retry::send_with_retries(
            "posting Grok Responses API request",
            || {
                self.request_headers(
                    self.http
                        .post(format!("{}/responses", self.base_url))
                        .header("Accept", "text/event-stream")
                        .header("x-grok-conv-id", &conversation_id)
                        .header("x-grok-req-id", &request_id)
                        .header("x-grok-model-override", &body.model)
                        .header("x-grok-session-id", &conversation_id)
                        .header("x-grok-agent-id", "anvil")
                        .json(body),
                    credential,
                )
            },
            Some(cancel),
            Some(first_progress),
        )
        .await
    }

    async fn invoke(&self, request: StreamChatRequest) -> Result<LlmResponse> {
        let StreamChatRequest {
            model,
            messages,
            tools,
            reasoning_effort,
            service_tier: _,
            temperature: _,
            structured_output,
            on_token,
            on_thought,
            cancel,
            idle_timeouts,
        } = request;
        let mut body = build_responses_request(
            &model,
            &messages,
            tools.as_deref(),
            reasoning_effort.as_deref(),
            structured_output.as_ref(),
            false,
            None,
        );
        match body.reasoning.as_mut() {
            Some(reasoning) => reasoning.summary = Some("concise".to_string()),
            None => {
                body.reasoning = Some(ReasoningConfig {
                    effort: None,
                    summary: Some("concise".to_string()),
                });
            }
        }
        body.include = Some(vec!["reasoning.encrypted_content".to_string()]);
        let credential = self.auth.credential().await?;
        let response = self
            .post_responses(&body, &credential, &cancel, idle_timeouts.first_progress)
            .await?;
        let status = response.status();
        let response = if status == StatusCode::UNAUTHORIZED {
            let refreshed = self.auth.refresh_rejected(&credential.access_token).await?;
            self.post_responses(&body, &refreshed, &cancel, idle_timeouts.first_progress)
                .await?
        } else {
            response
        };
        let status = response.status();
        if !status.is_success() {
            let response_body = response.text().await.unwrap_or_default();
            return Err(crate::http_retry::retryable_llm_error_for_status_and_body(
                format!("Grok Responses API failed (HTTP {status}): {response_body}"),
                status,
                &response_body,
            ));
        }
        let stream = response.bytes_stream().map(|chunk| {
            chunk
                .map(|bytes| bytes.to_vec())
                .map_err(anyhow::Error::from)
        });
        Ok(
            drive_responses_sse_stream(stream, on_token, on_thought, cancel, idle_timeouts)
                .await?
                .response,
        )
    }
}

impl LlmBackend for GrokClient {
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
        Box::pin(async move {
            Ok(self
                .discover_models()
                .await?
                .into_iter()
                .map(|model| model.id)
                .collect())
        })
    }

    fn list_model_metadata(&self) -> BoxFuture<'_, Result<Vec<ModelMetadata>>> {
        Box::pin(self.discover_models())
    }

    fn resolve_model_info(&self, configured_model: &str) -> ResolvedModelInfo {
        ResolvedModelInfo {
            configured_model: configured_model.to_string(),
            resolved_provider: Some("grok".to_string()),
            resolved_model: configured_model.to_string(),
        }
    }

    fn stream_chat(&self, request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
        Box::pin(self.invoke(request))
    }
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<Value>,
}

fn model_metadata(value: Value) -> Option<ModelMetadata> {
    let meta = value.get("_meta");
    let field = |names: &[&str]| {
        names
            .iter()
            .find_map(|name| value.get(*name))
            .or_else(|| names.iter().find_map(|name| meta?.get(*name)))
    };
    let id = value
        .get("id")
        .or_else(|| value.get("model"))
        .or_else(|| value.get("modelId"))
        .or_else(|| meta.and_then(|meta| meta.get("model")))
        .or_else(|| meta.and_then(|meta| meta.get("modelId")))
        .and_then(Value::as_str)?
        .to_string();
    let context_length = field(&[
        "context_window",
        "context_length",
        "contextWindow",
        "totalContextTokens",
    ])
    .and_then(Value::as_u64)
    .and_then(|length| u32::try_from(length).ok());
    let reasoning = field(&[
        "reasoning_efforts",
        "reasoningEfforts",
        "supported_reasoning_levels",
    ])
    .and_then(Value::as_array);
    let mut default_reasoning_level = field(&[
        "reasoning_effort",
        "reasoningEffort",
        "default_reasoning_effort",
        "default_reasoning_level",
    ])
    .and_then(Value::as_str)
    .map(str::to_string);
    let supported_reasoning_levels = reasoning
        .into_iter()
        .flatten()
        .filter_map(|preset| {
            let effort = preset
                .get("value")
                .or_else(|| preset.get("effort"))
                .and_then(Value::as_str)?;
            if preset.get("default").and_then(Value::as_bool) == Some(true) {
                default_reasoning_level = Some(effort.to_string());
            }
            Some(ReasoningLevelPreset {
                effort: effort.to_string(),
                description: preset
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            })
        })
        .collect();
    let supports_images = field(&["supports_images", "supportsImages"])
        .and_then(Value::as_bool)
        .or_else(|| {
            field(&["input_modalities", "inputModalities"])
                .and_then(Value::as_array)
                .map(|modalities| modalities.iter().any(|item| item.as_str() == Some("image")))
        });
    Some(ModelMetadata {
        id,
        default_reasoning_level,
        supported_reasoning_levels,
        service_tiers: Vec::new(),
        supports_images,
        context_length,
        pricing: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_remote_reasoning_and_modality_metadata() {
        let model = model_metadata(json!({
            "id": "grok-4.6",
            "context_window": 500000,
            "input_modalities": ["text", "image"],
            "reasoning_efforts": [
                {"value": "low", "description": "Fast"},
                {"value": "high", "description": "Deep", "default": true}
            ]
        }))
        .unwrap();
        assert_eq!(model.id, "grok-4.6");
        assert_eq!(model.context_length, Some(500000));
        assert_eq!(model.supports_images, Some(true));
        assert_eq!(model.default_reasoning_level.as_deref(), Some("high"));
        assert_eq!(model.supported_reasoning_levels.len(), 2);
    }

    #[test]
    fn parses_proxy_camel_case_and_meta_fields() {
        let model = model_metadata(json!({
            "id": "grok-next",
            "contextWindow": 256000,
            "_meta": {
                "reasoningEfforts": [{"value": "xhigh", "default": true}],
                "supportsImages": false
            }
        }))
        .unwrap();
        assert_eq!(model.context_length, Some(256000));
        assert_eq!(model.default_reasoning_level.as_deref(), Some("xhigh"));
        assert_eq!(model.supports_images, Some(false));
    }

    #[test]
    fn ignores_catalog_entries_without_an_id() {
        assert!(model_metadata(json!({"context_window": 1})).is_none());
    }

    #[test]
    fn proxy_requests_carry_required_oauth_headers() {
        let dir = tempfile::tempdir().unwrap();
        let client = GrokClient {
            auth: GrokAuthManager::for_test(dir.path().join("auth.json")),
            http: reqwest::Client::new(),
            base_url: "https://example.invalid/v1".into(),
            client_version: "1.2.3".into(),
        };
        let credential = GrokCredential {
            access_token: "oauth-token".into(),
            user_id: "user-1".into(),
            email: Some("user@example.com".into()),
        };
        let request = client
            .request_headers(
                client.http.get("https://example.invalid/v1/models"),
                &credential,
            )
            .build()
            .unwrap();
        let headers = request.headers();
        assert_eq!(headers["authorization"], "Bearer oauth-token");
        assert_eq!(headers["x-xai-token-auth"], "xai-grok-cli");
        assert_eq!(headers["x-authenticateresponse"], "authenticate-response");
        assert_eq!(headers["x-grok-client-version"], "1.2.3");
        assert_eq!(headers["x-grok-client-identifier"], "anvil");
        assert_eq!(headers["x-grok-client-mode"], "headless");
        assert_eq!(headers["x-userid"], "user-1");
        assert_eq!(headers["x-email"], "user@example.com");
    }
}
