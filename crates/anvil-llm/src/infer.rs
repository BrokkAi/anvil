//! One-shot, tool-free structured inference over an existing LLM backend.

use anyhow::{Context, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::http_retry::RetryableLlmError;
use crate::llm_client::{
    ChatMessage, IdleTimeouts, LlmBackend, LlmResponse, StreamChatRequest, TokenUsage,
    stream_chat_no_visible_output_with_retry,
};
use crate::structured_output::{
    StructuredOutputRequest, StructuredOutputResult, validate_response, validation_retry_prompt,
};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InferRole {
    System,
    User,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InferMessage {
    pub role: InferRole,
    pub content: String,
}

impl InferMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: InferRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: InferRole::User,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredInferRequest {
    pub messages: Vec<InferMessage>,
    pub schema_name: String,
    pub schema: Value,
}

#[derive(Debug, Clone)]
pub struct InferOptions {
    pub reasoning_effort: Option<String>,
    pub service_tier: Option<String>,
    pub idle_timeouts: IdleTimeouts,
    pub validation_retries: usize,
}

impl Default for InferOptions {
    fn default() -> Self {
        Self {
            reasoning_effort: None,
            service_tier: None,
            idle_timeouts: IdleTimeouts {
                first_progress: std::time::Duration::from_secs(
                    crate::llm_client::DEFAULT_IDLE_CHUNK_TIMEOUT_SECS,
                ),
                inter_chunk: std::time::Duration::from_secs(
                    crate::llm_client::DEFAULT_INTER_CHUNK_TIMEOUT_SECS,
                ),
            },
            validation_retries: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StructuredInferResponse {
    pub model: String,
    pub reasoning_effort: Option<String>,
    pub service_tier: Option<String>,
    pub output: Value,
    pub usage: InferUsage,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct InferUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub thought_tokens: u64,
    pub cached_read_tokens: u64,
    pub cached_write_tokens: u64,
}

impl From<TokenUsage> for InferUsage {
    fn from(value: TokenUsage) -> Self {
        Self {
            input_tokens: value.input_tokens,
            output_tokens: value.output_tokens,
            thought_tokens: value.thought_tokens,
            cached_read_tokens: value.cached_read_tokens,
            cached_write_tokens: value.cached_write_tokens,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferErrorKind {
    InvalidRequest,
    Cancelled,
    Authentication,
    RateLimited,
    ContextLength,
    Transport,
    StructuredOutput,
    Provider,
}

#[derive(Debug)]
pub struct InferError {
    kind: InferErrorKind,
    source: anyhow::Error,
}

impl InferError {
    pub fn kind(&self) -> InferErrorKind {
        self.kind
    }

    pub fn into_inner(self) -> anyhow::Error {
        self.source
    }

    fn new(kind: InferErrorKind, source: impl Into<anyhow::Error>) -> Self {
        Self {
            kind,
            source: source.into(),
        }
    }
}

impl std::fmt::Display for InferError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.source.fmt(formatter)
    }
}

impl std::error::Error for InferError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.source()
    }
}

pub async fn infer_structured(
    backend: &dyn LlmBackend,
    model: impl Into<String>,
    request: StructuredInferRequest,
    options: InferOptions,
    cancel: CancellationToken,
) -> Result<StructuredInferResponse, InferError> {
    let model = model.into();
    if model.trim().is_empty() {
        return Err(InferError::new(
            InferErrorKind::InvalidRequest,
            anyhow!("model must be non-empty"),
        ));
    }
    if request.messages.is_empty() {
        return Err(InferError::new(
            InferErrorKind::InvalidRequest,
            anyhow!("messages must contain at least one system or user message"),
        ));
    }
    let messages = request
        .messages
        .into_iter()
        .enumerate()
        .map(|(index, message)| {
            if message.content.is_empty() {
                return Err(InferError::new(
                    InferErrorKind::InvalidRequest,
                    anyhow!("messages[{index}].content must be non-empty"),
                ));
            }
            Ok(match message.role {
                InferRole::System => ChatMessage::system(message.content),
                InferRole::User => ChatMessage::user(message.content),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if request.schema_name.trim().is_empty() || !request.schema.is_object() {
        return Err(InferError::new(
            InferErrorKind::InvalidRequest,
            anyhow!("schema_name must be non-empty and schema must be an object"),
        ));
    }
    jsonschema::validator_for(&request.schema)
        .context("compiling structured-output schema")
        .map_err(|error| InferError::new(InferErrorKind::InvalidRequest, error))?;
    let structured_output = StructuredOutputRequest {
        schema_name: request.schema_name,
        schema: request.schema,
        allow_coercion: false,
        prefer_json_object: false,
    };
    let mut messages = messages;
    let mut total_usage = TokenUsage::default();
    let mut validation_attempt = 0;
    let output = loop {
        let response = stream_chat_no_visible_output_with_retry(
            backend,
            "bare structured inference",
            &cancel,
            || StreamChatRequest {
                model: model.clone(),
                messages: messages.clone(),
                tools: None,
                reasoning_effort: options.reasoning_effort.clone(),
                service_tier: options.service_tier.clone(),
                temperature: None,
                structured_output: Some(structured_output.clone()),
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: cancel.clone(),
                idle_timeouts: options.idle_timeouts,
            },
        )
        .await
        .map_err(|error| classify_error(error, &cancel))?;
        let (text, usage) = match response {
            LlmResponse::Text { text, usage, .. } => (text, usage),
            LlmResponse::ToolCalls { .. } => {
                return Err(InferError::new(
                    InferErrorKind::Provider,
                    anyhow!(
                        "inference backend returned tool calls even though no tools were supplied"
                    ),
                ));
            }
        };
        total_usage.add(usage);
        match validate_response(&structured_output, &text) {
            StructuredOutputResult::Success(success) => break success.validated_output,
            StructuredOutputResult::CoercedSuccess(_) => {
                unreachable!("structured inference disables coercion")
            }
            StructuredOutputResult::ValidationError(error)
                if validation_attempt < options.validation_retries =>
            {
                validation_attempt += 1;
                messages.push(ChatMessage::assistant(text));
                messages.push(ChatMessage::user(validation_retry_prompt(&error)));
            }
            StructuredOutputResult::ValidationError(error) => {
                return Err(InferError::new(
                    InferErrorKind::StructuredOutput,
                    anyhow!("structured output validation failed: {:?}", error.errors),
                ));
            }
        }
    };
    Ok(StructuredInferResponse {
        model,
        reasoning_effort: options.reasoning_effort,
        service_tier: options.service_tier,
        output,
        usage: total_usage.into(),
    })
}

fn classify_error(error: anyhow::Error, cancel: &CancellationToken) -> InferError {
    if cancel.is_cancelled() {
        return InferError::new(InferErrorKind::Cancelled, error);
    }
    let message = format!("{error:#}").to_ascii_lowercase();
    let kind = if message.contains("context length")
        || message.contains("maximum context")
        || message.contains("too many tokens")
        || message.contains("request too large")
    {
        InferErrorKind::ContextLength
    } else if message.contains("unauthorized")
        || message.contains("http 401")
        || message.contains("credentials")
        || message.contains("login")
        || message.contains("auth.json")
    {
        InferErrorKind::Authentication
    } else if message.contains("rate limit")
        || message.contains("http 429")
        || message.contains("quota exhausted")
    {
        InferErrorKind::RateLimited
    } else if error
        .chain()
        .any(|cause| cause.downcast_ref::<RetryableLlmError>().is_some())
    {
        InferErrorKind::Transport
    } else {
        InferErrorKind::Provider
    };
    InferError::new(kind, error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::future::BoxFuture;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ObservedRequest {
        model: String,
        roles: Vec<String>,
        tools_are_none: bool,
        has_structured_output: bool,
        reasoning_effort: Option<String>,
    }

    struct RecordingBackend {
        observed: Arc<Mutex<Vec<ObservedRequest>>>,
        response: &'static str,
    }

    impl LlmBackend for RecordingBackend {
        fn list_models(&self) -> BoxFuture<'_, anyhow::Result<Vec<String>>> {
            Box::pin(async { Ok(vec!["utility-model".to_string()]) })
        }

        fn stream_chat(
            &self,
            request: StreamChatRequest,
        ) -> BoxFuture<'_, anyhow::Result<LlmResponse>> {
            self.observed.lock().unwrap().push(ObservedRequest {
                model: request.model,
                roles: request
                    .messages
                    .iter()
                    .map(|message| message.role.clone())
                    .collect(),
                tools_are_none: request.tools.is_none(),
                has_structured_output: request.structured_output.is_some(),
                reasoning_effort: request.reasoning_effort,
            });
            let response = self.response.to_string();
            Box::pin(async move {
                Ok(LlmResponse::Text {
                    text: response,
                    reasoning_content: None,
                    usage: TokenUsage {
                        input_tokens: 3,
                        output_tokens: 2,
                        ..TokenUsage::default()
                    },
                    codex_reasoning: None,
                })
            })
        }
    }

    #[test]
    fn request_messages_have_no_assistant_or_tool_role() {
        let error = serde_json::from_value::<InferMessage>(serde_json::json!({
            "role": "assistant", "content": "not allowed"
        }))
        .unwrap_err();
        assert!(error.to_string().contains("unknown variant"));
    }

    #[test]
    fn defaults_match_cli_inference_policy() {
        let options = InferOptions::default();
        assert_eq!(options.validation_retries, 1);
        assert_eq!(
            options.idle_timeouts.first_progress,
            Duration::from_secs(crate::llm_client::DEFAULT_IDLE_CHUNK_TIMEOUT_SECS)
        );
    }

    #[tokio::test]
    async fn structured_inference_never_supplies_tools_or_session_messages() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let backend = RecordingBackend {
            observed: observed.clone(),
            response: r#"{"answer":"ok"}"#,
        };
        let response = infer_structured(
            &backend,
            "utility-model",
            StructuredInferRequest {
                messages: vec![
                    InferMessage::system("follow the schema"),
                    InferMessage::user("summarize this"),
                ],
                schema_name: "answer".to_string(),
                schema: serde_json::json!({
                    "type": "object",
                    "properties": { "answer": { "type": "string" } },
                    "required": ["answer"],
                    "additionalProperties": false
                }),
            },
            InferOptions {
                reasoning_effort: Some("low".to_string()),
                ..InferOptions::default()
            },
            CancellationToken::new(),
        )
        .await
        .unwrap();

        assert_eq!(response.output, serde_json::json!({"answer": "ok"}));
        assert_eq!(response.usage.input_tokens, 3);
        assert_eq!(response.usage.output_tokens, 2);
        assert_eq!(
            observed.lock().unwrap().as_slice(),
            [ObservedRequest {
                model: "utility-model".to_string(),
                roles: vec!["system".to_string(), "user".to_string()],
                tools_are_none: true,
                has_structured_output: true,
                reasoning_effort: Some("low".to_string()),
            }]
        );
    }
}
