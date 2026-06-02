use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};

use crate::llm_client::{
    ChatContentPart, ChatMessage, FunctionCall, LlmBackend, LlmResponse, ModelMetadata,
    OpenAiClient, StreamChatRequest, TokenUsage, ToolCall, ToolDefinition,
};
use crate::trace_logging::append_trace_record;

/// Returns true when the model supports Anthropic-style prompt caching
/// and needs explicit `cache_control` breakpoints in the request body.
/// All Bedrock-hosted Claude models have "anthropic" in their id
/// (e.g. `us.anthropic.claude-sonnet-4-6`).
fn requires_explicit_caching(model: &str) -> bool {
    model.contains("anthropic")
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

const ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";
const MAX_TOKENS: u32 = 8192;

#[derive(Clone)]
pub struct BedrockClient {
    bearer_token: String,
    region: String,
    default_model: String,
    http: reqwest::Client,
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
            "https://bedrock-runtime.amazonaws.com",
        )
        .build()
        .expect("failed to build Bedrock HTTP client");
        Self {
            bearer_token,
            region,
            default_model,
            http,
        }
    }

    fn invoke_url(&self, model: &str) -> String {
        let encoded = percent_encode_path_segment(model);
        format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{encoded}/invoke",
            self.region
        )
    }

    async fn invoke_model(&self, request: StreamChatRequest) -> Result<LlmResponse> {
        let StreamChatRequest {
            model,
            messages,
            tools,
            reasoning_effort: _,
            structured_output: _,
            mut on_token,
            on_thought: _,
            cancel,
            idle_timeout: _,
        } = request;
        let enable_cache = requires_explicit_caching(&model);
        let (system_blocks, messages) = convert_messages(messages, enable_cache)?;
        let system = if system_blocks.is_empty() {
            None
        } else {
            Some(system_blocks)
        };
        let body = BedrockAnthropicRequest {
            anthropic_version: ANTHROPIC_VERSION,
            system,
            messages,
            tools: tools.map(|t| convert_tools(t, enable_cache)),
            max_tokens: MAX_TOKENS,
            temperature: None,
        };
        let url = self.invoke_url(&model);
        trace_bedrock_request(&body);
        let send = self
            .http
            .post(&url)
            .bearer_auth(&self.bearer_token)
            .json(&body)
            .send();

        let resp = tokio::select! {
            _ = cancel.cancelled() => {
                return Ok(LlmResponse::Text {
                    text: String::new(),
                    usage: TokenUsage::default(),
                });
            }
            resp = send => resp.context("failed to send Bedrock request")?,
        };

        let status = resp.status();
        let body_text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            anyhow::bail!("Bedrock request failed (HTTP {status}): {body_text}");
        }

        let parsed: BedrockAnthropicResponse =
            serde_json::from_str(&body_text).context("parse Bedrock response")?;
        let mut text = String::new();
        let mut calls = Vec::new();
        for block in parsed.content {
            match block {
                BedrockContentBlock::Text { text: part } => text.push_str(&part),
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
        if !text.is_empty() {
            on_token(&text);
        }
        let usage = parsed.usage.into_usage();
        if calls.is_empty() {
            Ok(LlmResponse::Text { text, usage })
        } else {
            Ok(LlmResponse::ToolCalls { text, calls, usage })
        }
    }
}

impl LlmBackend for BedrockClient {
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
        let model = self.default_model.clone();
        Box::pin(async move { Ok(vec![model]) })
    }

    fn list_model_metadata(&self) -> BoxFuture<'_, Result<Vec<ModelMetadata>>> {
        let model = self.default_model.clone();
        Box::pin(async move {
            Ok(vec![ModelMetadata {
                id: model,
                default_reasoning_level: None,
                supported_reasoning_levels: Vec::new(),
                context_length: Some(200_000),
            }])
        })
    }

    fn stream_chat(&self, request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
        Box::pin(self.invoke_model(request))
    }
}

pub fn bearer_token_from_env_or_secrets() -> Result<Option<String>> {
    if let Ok(raw) = std::env::var(BEDROCK_API_KEY_ENV) {
        let token = raw.trim();
        if !token.is_empty() {
            return Ok(Some(token.to_string()));
        }
    }

    for name in ["aws_bearer_token_bedrock", "bedrock_api_key"] {
        if let Some(token) = read_secret_file(name)? {
            return Ok(Some(token));
        }
    }
    Ok(None)
}

pub fn region_from_env() -> String {
    std::env::var("AWS_REGION")
        .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
        .or_else(|_| std::env::var(BEDROCK_REGION_ENV))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| BEDROCK_DEFAULT_REGION.to_string())
}

pub fn model_from_env() -> String {
    std::env::var(BEDROCK_MODEL_ENV)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| BEDROCK_DEFAULT_MODEL.to_string())
}

fn read_secret_file(name: &str) -> Result<Option<String>> {
    let Some(home) = dirs::home_dir() else {
        return Ok(None);
    };
    let path: PathBuf = home.join(".secrets").join(name);
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
}

#[derive(Debug, Serialize, Clone)]
struct CacheControl {
    r#type: &'static str,
}

#[derive(Debug, Serialize)]
struct BedrockTextBlock {
    #[serde(rename = "type")]
    block_type: &'static str,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Debug, Serialize)]
struct BedrockMessage {
    role: String,
    content: Vec<BedrockContentOut>,
}

#[derive(Debug, Serialize)]
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

#[derive(Debug, Serialize)]
struct BedrockImageSource {
    #[serde(rename = "type")]
    source_type: &'static str,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
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
            input_schema: tool.function.parameters,
            cache_control: None,
        })
        .collect();
    if enable_cache && let Some(last) = converted.last_mut() {
        last.cache_control = Some(CACHE_CONTROL);
    }
    converted
}

fn parse_tool_arguments(raw: &str) -> Result<serde_json::Value> {
    serde_json::from_str(raw).with_context(|| format!("parse tool arguments as JSON: {raw}"))
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

#[cfg(test)]
mod tests {
    use crate::llm_client::{ChatContentPart, FunctionDef};

    use super::*;

    #[test]
    fn percent_encode_model_id() {
        assert_eq!(
            percent_encode_path_segment("us.anthropic.claude-sonnet-4-6"),
            "us.anthropic.claude-sonnet-4-6"
        );
        assert_eq!(percent_encode_path_segment("a/b c"), "a%2Fb%20c");
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
}
