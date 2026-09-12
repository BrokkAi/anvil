//! DeepSeek's stateless Responses API for native structured inference.
//!
//! Agent chat continues to use the Chat Completions backend. This client is
//! deliberately routed only by HostedClient, where a native JSON Schema request is
//! useful and partial completions must never be accepted as valid output.
//! DeepSeek may violate its requested schema even with strict=true, so the
//! shared inference layer must still validate every result locally.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use futures::{StreamExt, future::BoxFuture};

use crate::llm_client::{LlmBackend, LlmResponse, StreamChatRequest};
use crate::responses_api::{build_responses_request, drive_responses_sse_stream};

pub struct DeepSeekClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl DeepSeekClient {
    /// Env credentials take precedence over the existing consolidated store.
    pub fn load() -> Result<Option<Arc<dyn LlmBackend>>> {
        let key = match std::env::var(crate::discovery::DEEPSEEK_API_KEY_ENV)
            .ok()
            .filter(|key| !key.trim().is_empty())
        {
            Some(key) => Some(key),
            None => crate::deepseek_auth::read()?.map(|auth| auth.api_key),
        };
        key.filter(|key| !key.trim().is_empty())
            .map(|key| {
                Self::new(crate::discovery::DEEPSEEK_BASE_URL, key)
                    .map(|client| Arc::new(client) as Arc<dyn LlmBackend>)
            })
            .transpose()
    }

    /// Explicit endpoint construction also supports local wire-level tests.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Result<Self> {
        Ok(Self {
            http: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(20))
                .build()
                .context("building DeepSeek Responses client")?,
            api_key: api_key.into().trim().to_string(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
        })
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
        let effort = reasoning_effort.as_deref().map(|effort| {
            match effort.trim().to_ascii_lowercase().as_str() {
                "none" => "none",
                "minimal" | "low" => "low",
                "max" => "max",
                _ => "high",
            }
        });
        let body = build_responses_request(
            &model,
            &messages,
            tools.as_deref(),
            effort,
            structured_output.as_ref(),
            false,
            None,
        );
        let response = crate::http_retry::send_with_retries(
            "posting DeepSeek Responses request",
            || {
                self.http
                    .post(format!("{}/responses", self.base_url))
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
            // Read only enough to classify retryable/provider errors. Never
            // expose an error body that might echo credentials or prompt data.
            let read = async {
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
            };
            let body = tokio::select! {
                _ = cancel.cancelled() => bail!("DeepSeek Responses request cancelled"),
                body = tokio::time::timeout(Duration::from_secs(3), read) => body.unwrap_or_default(),
            };
            return Err(crate::http_retry::retryable_llm_error_for_status_and_body(
                format!("DeepSeek Responses API failed (HTTP {status})"),
                status,
                &body,
            ));
        }
        let stream = response
            .bytes_stream()
            .map(|chunk| chunk.map(|b| b.to_vec()).map_err(anyhow::Error::from));
        let outcome =
            drive_responses_sse_stream(stream, on_token, on_thought, cancel.clone(), idle_timeouts)
                .await?;
        if cancel.is_cancelled() {
            bail!("DeepSeek Responses request cancelled");
        }
        if outcome.incomplete {
            bail!("DeepSeek Responses output was incomplete; structured output cannot be accepted");
        }
        Ok(outcome.response)
    }
}

impl LlmBackend for DeepSeekClient {
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

    fn stream_chat(&self, request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
        Box::pin(self.invoke(request))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::{InferMessage, InferOptions, StructuredInferRequest, infer_structured};
    use crate::llm_client::{ChatMessage, IdleTimeouts};
    use crate::structured_output::StructuredOutputRequest;
    use serde_json::json;
    use tokio_util::sync::CancellationToken;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn schema() -> serde_json::Value {
        json!({"type":"object", "properties":{"slot0":{"type":"boolean"}},
               "required":["slot0"], "additionalProperties":false})
    }

    fn request(cancel: CancellationToken) -> StreamChatRequest {
        StreamChatRequest {
            model: "deepseek-v4-flash".to_string(),
            messages: vec![
                ChatMessage::system("stable rules"),
                ChatMessage::user("stable articles then candidates"),
            ],
            tools: None,
            reasoning_effort: Some("max".to_string()),
            service_tier: None,
            temperature: None,
            structured_output: Some(StructuredOutputRequest {
                schema_name: "coverage".to_string(),
                schema: schema(),
                allow_coercion: false,
                prefer_json_object: false,
            }),
            on_token: Box::new(|_| {}),
            on_thought: Box::new(|_| {}),
            cancel,
            idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(2)),
        }
    }

    fn completed() -> String {
        format!(
            "data: {}\n\ndata: {}\n\n",
            json!({"type":"response.output_text.delta","delta":"{\"slot0\":true}"}),
            json!({"type":"response.completed","response":{"id":"resp_test","usage":{"input_tokens":100,"output_tokens":15,
                "input_tokens_details":{"cached_tokens":80},"output_tokens_details":{"reasoning_tokens":10}}}})
        )
    }

    #[tokio::test]
    async fn structured_wire_preserves_prefix_and_requests_native_schema() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/responses"))
            .and(header("Authorization", "Bearer test-key"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(completed()),
            )
            .mount(&server)
            .await;
        let client = DeepSeekClient::new(server.uri(), "test-key").unwrap();
        let result = infer_structured(
            &client,
            "deepseek-v4-flash",
            StructuredInferRequest {
                messages: vec![
                    InferMessage::system("stable rules"),
                    InferMessage::user("stable articles then candidates"),
                ],
                schema_name: "coverage".to_string(),
                schema: schema(),
            },
            InferOptions {
                reasoning_effort: Some("low".to_string()),
                ..InferOptions::default()
            },
            CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(result.output, json!({"slot0":true}));
        assert_eq!(result.usage.input_tokens, 20);
        assert_eq!(result.usage.cached_read_tokens, 80);
        assert_eq!(result.usage.thought_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);
        let requests = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["instructions"], "stable rules");
        assert_eq!(body["input"].as_array().unwrap().len(), 1);
        assert_eq!(
            body["input"][0]["content"][0]["text"],
            "stable articles then candidates"
        );
        assert_eq!(body["text"]["format"]["type"], "json_schema");
        assert_eq!(body["text"]["format"]["schema"], schema());
        assert_eq!(body["reasoning"]["effort"], "low");
        assert_eq!(body["store"], false);
        assert!(body.get("previous_response_id").is_none());
    }

    #[tokio::test]
    async fn native_format_still_rejects_provider_schema_violations_locally() {
        let server = MockServer::start().await;
        let body = completed().replace(r#"\"slot0\":true"#, r#"\"slot0\":true,\"no_match\":true"#);
        Mock::given(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_string(body))
            .mount(&server)
            .await;
        let client = DeepSeekClient::new(server.uri(), "test").unwrap();
        let error = infer_structured(
            &client,
            "deepseek-v4-flash",
            StructuredInferRequest {
                messages: vec![InferMessage::user("Classify this")],
                schema_name: "coverage".to_string(),
                schema: schema(),
            },
            InferOptions {
                validation_retries: 0,
                ..InferOptions::default()
            },
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind(), crate::infer::InferErrorKind::StructuredOutput);
        assert!(error.to_string().contains("no_match"));
    }

    #[tokio::test]
    async fn incomplete_response_rejected_even_when_partial_text_is_valid_json() {
        let server = MockServer::start().await;
        let body = format!(
            "data: {}\n\ndata: {}\n\n",
            json!({"type":"response.output_text.delta","delta":"{\"slot0\":true}"}),
            json!({"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"}}})
        );
        Mock::given(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_string(body))
            .mount(&server)
            .await;
        let client = DeepSeekClient::new(server.uri(), "test").unwrap();
        let error = client
            .stream_chat(request(CancellationToken::new()))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("incomplete"));
    }

    #[tokio::test]
    async fn cancellation_interrupts_pending_response() {
        let server = MockServer::start().await;
        Mock::given(path("/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_secs(10))
                    .set_body_string(completed()),
            )
            .mount(&server)
            .await;
        let client = DeepSeekClient::new(server.uri(), "test").unwrap();
        let cancel = CancellationToken::new();
        let task = client.stream_chat(request(cancel.clone()));
        let trigger = async {
            tokio::time::sleep(Duration::from_millis(30)).await;
            cancel.cancel();
        };
        let (result, ()) = tokio::time::timeout(Duration::from_secs(1), async {
            tokio::join!(task, trigger)
        })
        .await
        .unwrap();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn reasoning_modes_follow_responses_dialect() {
        let server = MockServer::start().await;
        Mock::given(path("/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_string(completed()))
            .mount(&server)
            .await;
        let client = DeepSeekClient::new(server.uri(), "test").unwrap();
        for (requested, expected) in [
            ("none", "none"),
            ("minimal", "low"),
            ("medium", "high"),
            ("xhigh", "high"),
            ("max", "max"),
        ] {
            let mut req = request(CancellationToken::new());
            req.reasoning_effort = Some(requested.to_string());
            client.stream_chat(req).await.unwrap();
            let requests = server.received_requests().await.unwrap();
            let body: serde_json::Value =
                serde_json::from_slice(&requests.last().unwrap().body).unwrap();
            assert_eq!(body["reasoning"]["effort"], expected);
        }
    }
}
