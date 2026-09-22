//! Xiaomi MiMo's OpenAI-compatible Responses APIs.
//!
//! Token Plan and pay-as-you-go deliberately use different base URLs and key
//! prefixes. Keeping them as explicit plan routes prevents subscription traffic
//! from accidentally hitting the billed endpoint (or vice versa).

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use futures::{StreamExt, future::BoxFuture};

use crate::llm_client::{ChatMessage, LlmBackend, LlmResponse, ModelMetadata, StreamChatRequest};
use crate::responses_api::{
    ResponsesRequestOptions, ResponsesTextConfig, ResponsesTextFormat, build_responses_request,
    drive_responses_sse_stream,
};

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
            Self::TokenPlan => key.starts_with("tp-") || key.starts_with("ttp-"),
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
            mut messages,
            tools,
            reasoning_effort,
            structured_output,
            on_token,
            on_thought,
            cancel,
            idle_timeouts,
            ..
        } = request;
        // MiMo supports JSON-object mode, not native JSON Schema. Include
        // the schema after the stable caller prefix, including direct ACP
        // callers that do not go through infer_structured's fallback.
        if let Some(output) = &structured_output {
            let instruction = crate::structured_output::json_schema_instruction(output);
            if !messages
                .iter()
                .any(|message| message.role == "user" && message.content_text() == instruction)
            {
                messages.push(ChatMessage::user(instruction));
            }
        }
        let effort = reasoning_effort.as_deref().map(mimo_reasoning_effort);
        let mut body = build_responses_request(
            &model,
            &messages,
            tools.as_deref(),
            effort,
            None,
            ResponsesRequestOptions {
                replay_reasoning: true,
                ..Default::default()
            },
        );
        body.text = structured_output.as_ref().map(|_| ResponsesTextConfig {
            format: ResponsesTextFormat::JsonObject,
        });
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
            let body_text = tokio::select! {
                biased;
                _ = cancel.cancelled() => bail!("{} request cancelled", self.plan.label()),
                body = tokio::time::timeout(
                    Duration::from_secs(3).min(idle_timeouts.inter_chunk),
                    read_limited_error_body(response),
                ) => body.unwrap_or_default(),
            };
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
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
        Box::pin(async move {
            Ok(self
                .list_model_metadata()
                .await?
                .into_iter()
                .map(|model| model.id)
                .collect())
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
                .filter(|metadata| !is_audio_only_model(&metadata.id))
                .map(|metadata| enrich_mimo_metadata(metadata, self.plan))
                .collect())
        })
    }

    fn stream_chat(&self, request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
        Box::pin(self.invoke(request))
    }
}

fn is_audio_only_model(id: &str) -> bool {
    matches!(
        id,
        "mimo-v2.5-asr"
            | "mimo-v2.5-tts"
            | "mimo-v2.5-tts-voiceclone"
            | "mimo-v2.5-tts-voicedesign"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::{
        InferErrorKind, InferMessage, InferOptions, StructuredInferRequest, infer_structured,
    };
    use crate::llm_client::IdleTimeouts;
    use crate::structured_output::StructuredOutputRequest;
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

    #[test]
    fn shared_team_credentials_select_token_plan_only() {
        let lookup = |name: &str| (name == MIMO_API_KEY_ENV).then(|| "  ttp-team-key  ".into());
        assert_eq!(
            select_api_key(MimoPlan::TokenPlan, lookup),
            Some("ttp-team-key".into())
        );
        assert_eq!(select_api_key(MimoPlan::PayAsYouGo, lookup), None);
    }

    fn output_schema() -> serde_json::Value {
        json!({"type":"object", "properties":{"ok":{"type":"boolean"}},
            "required":["ok"], "additionalProperties":false})
    }

    #[tokio::test]
    async fn structured_inference_uses_json_object_and_validates_locally() {
        for (output, valid) in [(r#"{"ok":true}"#, true), (r#"{"ok":"wrong type"}"#, false)] {
            let server = MockServer::start().await;
            let body = format!(
                "data: {}\n\ndata: {}\n\n",
                json!({"type":"response.output_text.delta","delta":output}),
                json!({"type":"response.completed","response":{"id":"resp_test"}})
            );
            Mock::given(path("/v1/responses"))
                .respond_with(ResponseTemplate::new(200).set_body_string(body))
                .mount(&server)
                .await;
            let client = MimoClient::new(MimoPlan::PayAsYouGo, server.uri(), "sk-test").unwrap();
            let result = infer_structured(
                &client,
                "mimo-v2.6-pro",
                StructuredInferRequest {
                    messages: vec![
                        InferMessage::system("stable rules"),
                        InferMessage::user("stable input"),
                    ],
                    schema_name: "result".into(),
                    schema: output_schema(),
                },
                InferOptions {
                    validation_retries: 0,
                    ..Default::default()
                },
                CancellationToken::new(),
            )
            .await;
            if valid {
                assert_eq!(result.unwrap().output, json!({"ok":true}));
            } else {
                assert_eq!(result.unwrap_err().kind(), InferErrorKind::StructuredOutput);
            }
            let requests = server.received_requests().await.unwrap();
            let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
            assert_eq!(body["text"]["format"], json!({"type":"json_object"}));
            assert_eq!(body["instructions"], "stable rules");
            let input = body["input"].as_array().unwrap();
            assert_eq!(input.len(), 2, "schema must be included exactly once");
            assert_eq!(input[0]["content"][0]["text"], "stable input");
            assert_eq!(
                input[1]["content"][0]["text"],
                format!(
                    "Return only JSON matching this JSON Schema: {}",
                    output_schema()
                )
            );
        }
    }

    #[tokio::test]
    async fn direct_structured_chat_also_includes_schema_after_caller_input() {
        let server = MockServer::start().await;
        Mock::given(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_string(completed()))
            .mount(&server)
            .await;
        let client = MimoClient::new(MimoPlan::TokenPlan, server.uri(), "tp-test").unwrap();
        let mut req = request(CancellationToken::new());
        req.structured_output = Some(StructuredOutputRequest {
            schema_name: "result".into(),
            schema: output_schema(),
            allow_coercion: false,
            prefer_json_object: false,
        });
        client.stream_chat(req).await.unwrap();
        let requests = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["text"]["format"], json!({"type":"json_object"}));
        assert_eq!(body["instructions"], "stable rules");
        assert_eq!(
            body["input"][0]["content"][0]["text"],
            "summarize the issue"
        );
        assert_eq!(
            body["input"][1]["content"][0]["text"],
            format!(
                "Return only JSON matching this JSON Schema: {}",
                output_schema()
            )
        );
    }

    #[tokio::test]
    async fn error_body_obeys_cancellation_and_deadline() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        for cancel_request in [true, false] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            let (headers_sent, headers_received) = tokio::sync::oneshot::channel();
            let server = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut buffer = [0; 4096];
                let mut request_headers = Vec::new();
                while !request_headers.windows(4).any(|part| part == b"\r\n\r\n") {
                    let received = socket.read(&mut buffer).await.unwrap();
                    assert_ne!(received, 0, "request ended before its headers");
                    request_headers.extend_from_slice(&buffer[..received]);
                }
                socket
                    .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 100\r\n\r\n")
                    .await
                    .unwrap();
                headers_sent.send(()).unwrap();
                std::future::pending::<()>().await;
            });
            let client = MimoClient::new(MimoPlan::TokenPlan, endpoint, "tp-test").unwrap();
            let cancel = CancellationToken::new();
            let mut req = request(cancel.clone());
            if !cancel_request {
                req.idle_timeouts.inter_chunk = Duration::from_millis(100);
            }
            let mut call = tokio::spawn(async move { client.stream_chat(req).await });
            headers_received.await.unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
            if cancel_request {
                cancel.cancel();
            }
            let result = tokio::time::timeout(Duration::from_secs(1), &mut call).await;
            call.abort();
            server.abort();
            let error = result
                .expect("stalled error body must terminate promptly")
                .unwrap()
                .unwrap_err();
            assert!(
                error.to_string().contains(if cancel_request {
                    "cancelled"
                } else {
                    "HTTP 400"
                }),
                "{error:#}"
            );
        }
    }

    #[tokio::test]
    async fn tool_round_trip_replays_reasoning_without_duplication() {
        for (send_delta, send_done) in [(true, true), (true, false), (false, true)] {
            let server = MockServer::start().await;
            let mut events = vec![];
            if send_delta {
                events.push(json!({"type":"response.reasoning_text.delta","delta":"Inspect the file first"}));
            }
            if send_done {
                events.push(json!({"type":"response.output_item.done","item":{"type":"reasoning","id":"r1","content":[{"type":"reasoning_text","text":"Inspect the file first"}]}}));
            }
            events.extend([
                json!({"type":"response.output_item.done","item":{"type":"function_call","call_id":"call1","name":"read_file","arguments":"{}"}}),
                json!({"type":"response.completed","response":{"id":"resp1"}}),
            ]);
            let body = events
                .iter()
                .map(|event| format!("data: {event}\n\n"))
                .collect::<String>();
            Mock::given(path("/v1/responses"))
                .respond_with(ResponseTemplate::new(200).set_body_string(body))
                .mount(&server)
                .await;
            let client = MimoClient::new(MimoPlan::TokenPlan, server.uri(), "tp-test").unwrap();
            let mut req = request(CancellationToken::new());
            let thoughts = Arc::new(std::sync::Mutex::new(String::new()));
            let captured = thoughts.clone();
            req.on_thought = Box::new(move |text| captured.lock().unwrap().push_str(text));
            let response = client.stream_chat(req).await.unwrap();
            let LlmResponse::ToolCalls {
                text,
                reasoning_content,
                calls,
                ..
            } = response
            else {
                panic!("expected tool response");
            };
            assert_eq!(reasoning_content.as_deref(), Some("Inspect the file first"));
            assert_eq!(*thoughts.lock().unwrap(), "Inspect the file first");
            let mut next = request(CancellationToken::new());
            next.messages.push(
                ChatMessage::assistant_tool_calls_with_content_and_reasoning(
                    text,
                    calls,
                    reasoning_content,
                ),
            );
            next.messages.push(ChatMessage::tool_result(
                "call1",
                "read_file",
                "file contents",
            ));
            client.stream_chat(next).await.unwrap();
            let requests = server.received_requests().await.unwrap();
            let body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
            let input = body["input"].as_array().unwrap();
            assert_eq!(
                input
                    .iter()
                    .map(|item| item["type"].as_str().unwrap())
                    .collect::<Vec<_>>(),
                vec![
                    "message",
                    "reasoning",
                    "function_call",
                    "function_call_output"
                ]
            );
            assert!(input[1]["id"].as_str().is_some_and(|id| !id.is_empty()));
            assert_eq!(
                input[1]["content"],
                json!([{"type":"reasoning_text","text":"Inspect the file first"}])
            );
            assert_eq!(input[2]["call_id"], input[3]["call_id"]);
            assert_eq!(input[3]["output"], "file contents");
        }
    }

    #[tokio::test]
    async fn both_discovery_methods_exclude_audio_and_preserve_unknown_models() {
        let server = MockServer::start().await;
        Mock::given(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data":[
                {"id":"mimo-v2.5-asr"}, {"id":"mimo-v2.5-tts"},
                {"id":"mimo-v2.5-tts-voiceclone"}, {"id":"mimo-v2.5-tts-voicedesign"},
                {"id":"mimo-v2.6-pro"}, {"id":"future-mimo"}
            ]})))
            .mount(&server)
            .await;
        for plan in [MimoPlan::TokenPlan, MimoPlan::PayAsYouGo] {
            let client = MimoClient::new(plan, server.uri(), "test-key").unwrap();
            let metadata_ids: Vec<_> = client
                .list_model_metadata()
                .await
                .unwrap()
                .into_iter()
                .map(|model| model.id)
                .collect();
            assert_eq!(metadata_ids, vec!["mimo-v2.6-pro", "future-mimo"]);
            assert_eq!(client.list_models().await.unwrap(), metadata_ids);
        }
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
