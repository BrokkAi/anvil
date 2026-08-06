//! One-shot, tool-free structured inference over Anvil's Codex-auth backend.

use std::io::{Read, Write};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Args;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use std::sync::Arc;

use crate::codex_client::CodexClient;
use crate::llm_client::{
    ChatMessage, IdleTimeouts, LlmBackend, LlmResponse, StreamChatRequest, TokenUsage,
    stream_chat_no_visible_output_with_retry,
};
use crate::structured_output::{
    StructuredOutputRequest, StructuredOutputResult, validate_response,
};

const CODEX_MODEL_PREFIX: &str = "codex::";
const KIMI_MODEL_PREFIX: &str = "kimi::";

#[derive(Args, Debug)]
pub(crate) struct InferArgs {
    /// Codex wire model id. The codex:: prefix is required to prevent provider fallback.
    #[arg(long)]
    model: String,

    /// Reasoning effort forwarded to the Codex Responses API.
    #[arg(long)]
    reasoning_effort: Option<String>,

    /// Optional service tier. Omit this flag to use the provider default.
    #[arg(long)]
    service_tier: Option<String>,

    /// Seconds to wait for the first meaningful response event.
    #[arg(long, default_value_t = crate::llm_client::DEFAULT_IDLE_CHUNK_TIMEOUT_SECS)]
    idle_timeout_secs: u64,

    /// Seconds to wait between meaningful response events.
    #[arg(long, default_value_t = crate::llm_client::DEFAULT_INTER_CHUNK_TIMEOUT_SECS)]
    stall_timeout_secs: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InferRequest {
    messages: Vec<InferMessage>,
    schema_name: String,
    schema: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InferMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct InferResponse {
    model: String,
    reasoning_effort: Option<String>,
    service_tier: Option<String>,
    output: Value,
    usage: InferUsage,
}

#[derive(Debug, Serialize)]
struct InferUsage {
    input_tokens: u64,
    output_tokens: u64,
    thought_tokens: u64,
    cached_read_tokens: u64,
    cached_write_tokens: u64,
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

pub(crate) async fn run(args: &InferArgs) -> Result<()> {
    // Route on the provider prefix so the same tool-free, schema-constrained
    // path can judge with either the Codex-auth backend or the Kimi Code
    // backend. The prefix is required in both cases to prevent provider
    // fallback picking a different model than the caller pinned.
    let (backend, wire_model): (Arc<dyn LlmBackend>, String) =
        if let Some(model) = args.model.strip_prefix(CODEX_MODEL_PREFIX) {
            if model.trim().is_empty() {
                bail!("--model must name a model after the codex:: prefix");
            }
            (Arc::new(CodexClient::new()), model.to_string())
        } else if let Some(model) = args.model.strip_prefix(KIMI_MODEL_PREFIX) {
            if model.trim().is_empty() {
                bail!("--model must name a model after the kimi:: prefix");
            }
            let backend = crate::build_kimi_backend().ok_or_else(|| {
                anyhow::anyhow!(
                    "Kimi backend is not configured; set KIMI_API_KEY or sign in with the Kimi CLI"
                )
            })?;
            (backend, model.to_string())
        } else {
            bail!("--model must use the codex::<model-id> or kimi::<model-id> wire form");
        };

    let mut raw = String::new();
    std::io::stdin()
        .read_to_string(&mut raw)
        .context("reading inference request from stdin")?;
    let input: InferRequest =
        serde_json::from_str(&raw).context("parsing inference request JSON")?;
    let messages = validate_messages(input.messages)?;
    if input.schema_name.trim().is_empty() {
        bail!("schema_name must be non-empty");
    }
    if !input.schema.is_object() {
        bail!("schema must be a JSON object");
    }
    jsonschema::validator_for(&input.schema).context("compiling structured-output schema")?;
    let structured_output = StructuredOutputRequest {
        schema_name: input.schema_name,
        schema: input.schema,
        allow_coercion: false,
        prefer_json_object: false,
    };

    let cancel = CancellationToken::new();
    let response = stream_chat_no_visible_output_with_retry(
        backend.as_ref(),
        "bare structured inference",
        &cancel,
        || {
            StreamChatRequest {
                model: wire_model.clone(),
                messages: messages.clone(),
                tools: None,
                reasoning_effort: args.reasoning_effort.clone(),
                service_tier: args.service_tier.clone(),
                temperature: None,
                structured_output: Some(structured_output.clone()),
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: cancel.clone(),
                idle_timeouts: IdleTimeouts {
                    first_progress: Duration::from_secs(args.idle_timeout_secs),
                    inter_chunk: Duration::from_secs(args.stall_timeout_secs),
                },
            }
        })
        .await
        .context("running bare structured inference")?;

    let (text, usage) = match response {
        LlmResponse::Text { text, usage, .. } => (text, usage),
        LlmResponse::ToolCalls { .. } => {
            bail!("Codex returned tool calls even though no tools were supplied")
        }
    };
    let output = match validate_response(&structured_output, &text) {
        StructuredOutputResult::Success(success) => success.validated_output,
        StructuredOutputResult::CoercedSuccess(_) => {
            unreachable!("bare inference disables structured-output coercion")
        }
        StructuredOutputResult::ValidationError(error) => {
            bail!("structured output validation failed: {:?}", error.errors)
        }
    };
    let result = InferResponse {
        model: args.model.clone(),
        reasoning_effort: args.reasoning_effort.clone(),
        service_tier: args.service_tier.clone(),
        output,
        usage: usage.into(),
    };
    serde_json::to_writer(std::io::stdout().lock(), &result)
        .context("writing inference response JSON")?;
    std::io::stdout().lock().write_all(b"\n")?;
    Ok(())
}

fn validate_messages(messages: Vec<InferMessage>) -> Result<Vec<ChatMessage>> {
    if messages.is_empty() {
        bail!("messages must contain at least one system or user message");
    }
    messages
        .into_iter()
        .enumerate()
        .map(|(index, message)| {
            if message.content.is_empty() {
                bail!("messages[{index}].content must be non-empty");
            }
            match message.role.as_str() {
                "system" => Ok(ChatMessage::system(message.content)),
                "user" => Ok(ChatMessage::user(message.content)),
                role => bail!("messages[{index}].role must be system or user, got {role:?}"),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_reject_agent_and_tool_history() {
        let error = validate_messages(vec![InferMessage {
            role: "assistant".into(),
            content: "prior answer".into(),
        }])
        .unwrap_err();
        assert!(error.to_string().contains("must be system or user"));
    }

    #[test]
    fn messages_accept_system_and_user_text() {
        let messages = validate_messages(vec![
            InferMessage {
                role: "system".into(),
                content: "judge carefully".into(),
            },
            InferMessage {
                role: "user".into(),
                content: "item".into(),
            },
        ])
        .unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
    }
}
