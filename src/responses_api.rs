use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use futures::Stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::llm_client::{
    ChatContentPart, ChatMessage, FunctionCall, IncompleteStreamError, LlmResponse, TokenSink,
    TokenUsage, ToolCall, ToolDefinition,
};
use crate::structured_output::{
    NativeResponseFormat, StructuredOutputRequest, native_response_format,
};

#[derive(Debug, Serialize)]
pub(crate) struct ResponsesRequest {
    pub(crate) model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) instructions: Option<String>,
    pub(crate) input: Vec<ResponsesInputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<ResponsesToolDef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_choice: Option<String>,
    pub(crate) parallel_tool_calls: bool,
    pub(crate) stream: bool,
    pub(crate) store: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning: Option<ReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) text: Option<ResponsesTextConfig>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ReasoningConfig {
    pub(crate) effort: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct ResponsesTextConfig {
    pub(crate) format: ResponsesTextFormat,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ResponsesTextFormat {
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        strict: bool,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ResponsesInputItem {
    Message {
        role: String,
        content: Vec<ResponsesContent>,
    },
    FunctionCall {
        name: String,
        arguments: String,
        call_id: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ResponsesContent {
    InputText { text: String },
    InputImage { image_url: String },
    OutputText { text: String },
}

#[derive(Debug, Serialize)]
pub(crate) struct ResponsesToolDef {
    pub(crate) r#type: String,
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) parameters: serde_json::Value,
}

pub(crate) fn build_responses_request(
    model: &str,
    messages: &[ChatMessage],
    tools: Option<&[ToolDefinition]>,
    reasoning_effort: Option<&str>,
    structured_output: Option<&StructuredOutputRequest>,
) -> ResponsesRequest {
    let mut instructions_parts: Vec<String> = Vec::new();
    let mut input: Vec<ResponsesInputItem> = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                let text = msg.content_text();
                if !text.is_empty() {
                    instructions_parts.push(text);
                }
            }
            "user" if !msg.content.is_empty() => {
                input.push(ResponsesInputItem::Message {
                    role: "user".to_string(),
                    content: msg
                        .content
                        .iter()
                        .map(|part| match part {
                            ChatContentPart::Text { text } => {
                                ResponsesContent::InputText { text: text.clone() }
                            }
                            ChatContentPart::Image { image_url } => ResponsesContent::InputImage {
                                image_url: image_url.clone(),
                            },
                        })
                        .collect(),
                });
            }
            "assistant" => {
                if let Some(calls) = &msg.tool_calls {
                    for call in calls {
                        input.push(ResponsesInputItem::FunctionCall {
                            name: call.function.name.clone(),
                            arguments: crate::tool_arguments::normalize_request_tool_arguments(
                                &call.function.arguments,
                                &call.function.name,
                            ),
                            call_id: call.id.clone(),
                        });
                    }
                } else if msg.has_content() {
                    input.push(ResponsesInputItem::Message {
                        role: "assistant".to_string(),
                        content: vec![ResponsesContent::OutputText {
                            text: msg.content_text(),
                        }],
                    });
                }
            }
            "tool" => {
                if let Some(call_id) = &msg.tool_call_id {
                    input.push(ResponsesInputItem::FunctionCallOutput {
                        call_id: call_id.clone(),
                        output: msg.content_text(),
                    });
                }
            }
            _ => {}
        }
    }

    let tools = tools.map(|defs| {
        defs.iter()
            .map(|d| ResponsesToolDef {
                r#type: "function".to_string(),
                name: d.function.name.clone(),
                description: d.function.description.clone(),
                parameters: d.function.parameters.clone(),
            })
            .collect()
    });

    let text = structured_output
        .map(native_response_format)
        .map(|format: NativeResponseFormat| ResponsesTextConfig {
            format: ResponsesTextFormat::JsonSchema {
                name: format.name,
                schema: format.schema,
                strict: format.strict,
            },
        });

    ResponsesRequest {
        model: model.to_string(),
        instructions: (!instructions_parts.is_empty()).then(|| instructions_parts.join("\n\n")),
        input,
        tool_choice: tools.as_ref().map(|_| "auto".to_string()),
        tools,
        parallel_tool_calls: true,
        stream: true,
        store: false,
        reasoning: reasoning_effort.map(|effort| ReasoningConfig {
            effort: effort.to_string(),
        }),
        text,
    }
}

#[derive(Debug, Deserialize)]
struct StreamEvent {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    delta: Option<String>,
    #[serde(default)]
    item: Option<serde_json::Value>,
    #[serde(default)]
    response: Option<ResponseFinal>,
}

#[derive(Debug, Deserialize)]
struct ResponseFinal {
    #[serde(default)]
    error: Option<ResponseError>,
    #[serde(default)]
    usage: Option<ResponseUsage>,
}

#[derive(Debug, Deserialize)]
struct ResponseUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    input_tokens_details: Option<InputTokensDetails>,
    #[serde(default)]
    output_tokens_details: Option<OutputTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct InputTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct OutputTokensDetails {
    #[serde(default)]
    reasoning_tokens: u64,
}

impl ResponseUsage {
    fn into_usage(self) -> TokenUsage {
        let cached = self
            .input_tokens_details
            .as_ref()
            .map(|d| d.cached_tokens)
            .unwrap_or(0);
        let reasoning = self
            .output_tokens_details
            .as_ref()
            .map(|d| d.reasoning_tokens)
            .unwrap_or(0);
        TokenUsage {
            input_tokens: self.input_tokens.saturating_sub(cached),
            output_tokens: self.output_tokens.saturating_sub(reasoning),
            thought_tokens: reasoning,
            cached_read_tokens: cached,
            cached_write_tokens: 0,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ResponseError {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutputItem {
    Message {
        #[serde(default)]
        role: Option<String>,
        #[serde(default)]
        content: Vec<OutputItemContent>,
    },
    FunctionCall {
        #[serde(default)]
        id: Option<String>,
        name: String,
        arguments: String,
        #[serde(default)]
        call_id: Option<String>,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutputItemContent {
    OutputText {
        #[serde(default)]
        text: String,
    },
    #[serde(other)]
    Other,
}

pub(crate) async fn drive_responses_sse_stream<S>(
    mut stream: S,
    mut on_token: TokenSink,
    mut on_thought: TokenSink,
    cancel: CancellationToken,
    idle: Duration,
) -> Result<LlmResponse>
where
    S: Stream<Item = Result<Vec<u8>>> + Unpin,
{
    let mut full_text = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut raw_buf: Vec<u8> = Vec::new();
    let mut deadline = tokio::time::Instant::now() + idle;
    let mut completed = false;
    let mut failure: Option<anyhow::Error> = None;
    let mut usage = TokenUsage::default();
    let mut deltas_received = false;

    loop {
        tokio::select! {
            _ = cancel.cancelled() => break,
            chunk_or_timeout = tokio::time::timeout_at(deadline, stream.next()) => {
                let chunk_opt = match chunk_or_timeout {
                    Ok(opt) => opt,
                    Err(_elapsed) => anyhow::bail!(
                        "Responses stream made no meaningful progress for {}s; aborting",
                        idle.as_secs()
                    ),
                };
                let eof_after_buffer = if let Some(chunk) = chunk_opt {
                    // Tag mid-stream transport failures so `is_retryable_llm_error`
                    // classifies a dropped SSE body as retryable (matches the
                    // codex_client path, which adds the same context).
                    let chunk = chunk.context("Responses stream read error")?;
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
                    if line.is_empty() || line.starts_with(':') || line.starts_with("event:") {
                        continue;
                    }
                    let data = if let Some(stripped) = line.strip_prefix("data: ") {
                        stripped.trim()
                    } else if let Some(stripped) = line.strip_prefix("data:") {
                        stripped.trim()
                    } else {
                        continue;
                    };
                    if data == "[DONE]" {
                        continue;
                    }

                    let Ok(event) = serde_json::from_str::<StreamEvent>(data) else {
                        continue;
                    };
                    match event.kind.as_str() {
                        "response.output_text.delta" => {
                            if let Some(delta) = event.delta {
                                on_token(&delta);
                                full_text.push_str(&delta);
                                deltas_received = true;
                                made_progress = true;
                            }
                        }
                        "response.output_item.done" => {
                            if let Some(item_val) = event.item
                                && let Ok(item) = serde_json::from_value::<OutputItem>(item_val)
                            {
                                match item {
                                    OutputItem::Message { role, content } => {
                                        if role.as_deref() == Some("assistant") && !deltas_received {
                                            for c in content {
                                                if let OutputItemContent::OutputText { text } = c {
                                                    on_token(&text);
                                                    full_text.push_str(&text);
                                                }
                                            }
                                        }
                                        made_progress = true;
                                    }
                                    OutputItem::FunctionCall { id, name, arguments, call_id } => {
                                        let resolved_id = call_id
                                            .or(id)
                                            .unwrap_or_else(|| format!("call_{}", tool_calls.len()));
                                        let arguments =
                                            crate::tool_arguments::normalize_streamed_tool_arguments(
                                                &resolved_id,
                                                &name,
                                                arguments,
                                                "Responses SSE",
                                            )?;
                                        tool_calls.push(ToolCall {
                                            id: resolved_id,
                                            r#type: "function".to_string(),
                                            function: FunctionCall { name, arguments },
                                        });
                                        made_progress = true;
                                    }
                                    OutputItem::Other => {}
                                }
                            }
                        }
                        "response.completed" => {
                            if let Some(final_body) = event.response
                                && let Some(u) = final_body.usage
                            {
                                usage = u.into_usage();
                            }
                            completed = true;
                            break;
                        }
                        "response.failed" => {
                            let msg = event
                                .response
                                .and_then(|r| r.error)
                                .map(|e| {
                                    let code = e.code.unwrap_or_else(|| "unknown".to_string());
                                    let body = e.message.unwrap_or_default();
                                    format!("{code}: {body}")
                                })
                                .unwrap_or_else(|| "unknown error".to_string());
                            failure = Some(anyhow!("Responses stream failed: {msg}"));
                            completed = true;
                            break;
                        }
                        "response.reasoning_text.delta"
                        | "response.reasoning_summary_text.delta" => {
                            if let Some(delta) = event.delta {
                                on_thought(&delta);
                                made_progress = true;
                            }
                        }
                        _ => {
                            made_progress = true;
                        }
                    }
                }

                if completed {
                    break;
                }
                if made_progress {
                    deadline = tokio::time::Instant::now() + idle;
                }
                if eof_after_buffer {
                    break;
                }
            }
        }
    }

    if let Some(err) = failure {
        return Err(err);
    }
    if cancel.is_cancelled() {
        return Ok(LlmResponse::Text {
            text: full_text,
            reasoning_content: None,
            usage,
        });
    }
    if !completed {
        return Err(anyhow::Error::new(IncompleteStreamError::new(
            "Responses SSE",
            "response.completed",
        )));
    }
    if tool_calls.is_empty() {
        Ok(LlmResponse::Text {
            text: full_text,
            reasoning_content: None,
            usage,
        })
    } else {
        Ok(LlmResponse::ToolCalls {
            text: full_text,
            reasoning_content: None,
            calls: tool_calls,
            usage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use futures::stream;
    use std::sync::{Arc, Mutex};

    fn collect_tokens() -> (TokenSink, Arc<Mutex<String>>) {
        let collected = Arc::new(Mutex::new(String::new()));
        let inner = collected.clone();
        let cb: TokenSink = Box::new(move |t| {
            inner.lock().unwrap().push_str(t);
        });
        (cb, collected)
    }

    fn noop_sink() -> TokenSink {
        Box::new(|_| {})
    }

    #[tokio::test]
    async fn shared_responses_stream_requires_response_completed() {
        let raw = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n";
        let stream = stream::iter(vec![Ok(raw.as_bytes().to_vec())]);
        let (on_token, collected) = collect_tokens();

        let err = drive_responses_sse_stream(
            stream,
            on_token,
            noop_sink(),
            CancellationToken::new(),
            Duration::from_secs(5),
        )
        .await
        .expect_err("EOF before response.completed must be incomplete");

        assert!(crate::llm_client::is_incomplete_stream_error(&err));
        assert_eq!(collected.lock().unwrap().as_str(), "partial");
    }

    #[tokio::test]
    async fn shared_responses_stream_does_not_accept_done_as_completion() {
        let raw = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n",
            "data: [DONE]\n\n",
        );
        let stream = stream::iter(vec![Ok(raw.as_bytes().to_vec())]);
        let (on_token, collected) = collect_tokens();

        let err = drive_responses_sse_stream(
            stream,
            on_token,
            noop_sink(),
            CancellationToken::new(),
            Duration::from_secs(5),
        )
        .await
        .expect_err("[DONE] is not the Responses completion marker");

        assert!(crate::llm_client::is_incomplete_stream_error(&err));
        assert_eq!(collected.lock().unwrap().as_str(), "partial");
    }

    #[tokio::test(start_paused = true)]
    async fn shared_responses_stream_done_does_not_reset_idle_deadline() {
        let stream = stream::iter(vec![Ok(b"data: [DONE]\n".to_vec())]).chain(stream::pending());
        let (on_token, _) = collect_tokens();

        let err = drive_responses_sse_stream(
            stream,
            on_token,
            noop_sink(),
            CancellationToken::new(),
            Duration::from_secs(5),
        )
        .await
        .expect_err("[DONE] should not keep a Responses stream alive");
        let msg = format!("{err:#}");

        assert!(msg.contains("no meaningful progress"), "got: {msg}");
    }

    #[tokio::test]
    async fn shared_responses_stream_returns_text_on_response_completed() {
        let raw = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{}}\n\n",
        );
        let stream = stream::iter(vec![Ok(raw.as_bytes().to_vec())]);
        let (on_token, collected) = collect_tokens();

        let resp = drive_responses_sse_stream(
            stream,
            on_token,
            noop_sink(),
            CancellationToken::new(),
            Duration::from_secs(5),
        )
        .await
        .expect("response.completed should finish the stream");

        match resp {
            LlmResponse::Text { text, .. } => assert_eq!(text, "ok"),
            other => panic!("expected Text, got {other:?}"),
        }
        assert_eq!(collected.lock().unwrap().as_str(), "ok");
    }

    #[tokio::test]
    async fn shared_responses_stream_accepts_final_completed_without_newline() {
        let raw = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{}}",
        );
        let stream = stream::iter(vec![Ok(raw.as_bytes().to_vec())]);
        let (on_token, collected) = collect_tokens();

        let resp = drive_responses_sse_stream(
            stream,
            on_token,
            noop_sink(),
            CancellationToken::new(),
            Duration::from_secs(5),
        )
        .await
        .expect("final buffered response.completed should complete");

        match resp {
            LlmResponse::Text { text, .. } => assert_eq!(text, "ok"),
            other => panic!("expected Text, got {other:?}"),
        }
        assert_eq!(collected.lock().unwrap().as_str(), "ok");
    }

    #[tokio::test]
    async fn shared_responses_stream_repairs_malformed_tool_call_arguments() {
        let raw = concat!(
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"name\":\"read_file\",\"arguments\":\"{file_path:'a.txt',}\",\"call_id\":\"fc_1\"}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{}}\n\n",
        );
        let stream = stream::iter(vec![Ok(raw.as_bytes().to_vec())]);
        let (on_token, _) = collect_tokens();

        let resp = drive_responses_sse_stream(
            stream,
            on_token,
            noop_sink(),
            CancellationToken::new(),
            Duration::from_secs(5),
        )
        .await
        .expect("repairable tool-call arguments should complete");

        match resp {
            LlmResponse::ToolCalls { calls, .. } => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.arguments, r#"{"file_path":"a.txt"}"#);
            }
            other => panic!("expected ToolCalls, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn shared_responses_stream_treats_unterminated_tool_arguments_as_incomplete() {
        let raw = concat!(
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"name\":\"read_file\",\"arguments\":\"{\\\"file_path\\\":\\\"unterminated\",\"call_id\":\"fc_1\"}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{}}\n\n",
        );
        let stream = stream::iter(vec![Ok(raw.as_bytes().to_vec())]);
        let (on_token, _) = collect_tokens();

        let err = drive_responses_sse_stream(
            stream,
            on_token,
            noop_sink(),
            CancellationToken::new(),
            Duration::from_secs(5),
        )
        .await
        .expect_err("unterminated streamed arguments should be retryable truncation");

        assert!(crate::llm_client::is_incomplete_stream_error(&err));
    }

    #[tokio::test]
    async fn shared_responses_stream_cancellation_is_not_incomplete_eof() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let stream = stream::pending::<Result<Vec<u8>>>();
        let (on_token, _) = collect_tokens();

        let resp = drive_responses_sse_stream(
            stream,
            on_token,
            noop_sink(),
            cancel,
            Duration::from_secs(5),
        )
        .await
        .expect("cancellation should return normally");

        match resp {
            LlmResponse::Text { text, .. } => assert_eq!(text, ""),
            other => panic!("expected Text, got {other:?}"),
        }
    }
}
