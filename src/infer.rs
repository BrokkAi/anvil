//! One-shot, tool-free structured inference over Anvil's hosted backends.

use std::io::{Read, Write};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Args;
use tokio_util::sync::CancellationToken;

use anvil_client::infer::{HostedClient, InferOptions, StructuredInferRequest};
use anvil_client::llm_client::IdleTimeouts;

#[derive(Args, Debug)]
pub(crate) struct InferArgs {
    /// Provider-qualified wire model id (codex::, meta::, kimi::, grok::, or deepseek::).
    #[arg(long)]
    model: String,

    /// Reasoning effort forwarded in the selected provider's dialect.
    #[arg(long)]
    reasoning_effort: Option<String>,

    /// Optional service tier. Omit this flag to use the provider default.
    #[arg(long)]
    service_tier: Option<String>,

    /// Seconds to wait for the first meaningful response event.
    #[arg(long, default_value_t = anvil_client::llm_client::DEFAULT_IDLE_CHUNK_TIMEOUT_SECS)]
    idle_timeout_secs: u64,

    /// Seconds to wait between meaningful response events.
    #[arg(long, default_value_t = anvil_client::llm_client::DEFAULT_INTER_CHUNK_TIMEOUT_SECS)]
    stall_timeout_secs: u64,

    /// Additional attempts after local structured-output validation fails.
    #[arg(long, default_value_t = 1)]
    validation_retries: usize,
}

pub(crate) async fn run(args: &InferArgs) -> Result<()> {
    let mut raw = String::new();
    std::io::stdin()
        .read_to_string(&mut raw)
        .context("reading inference request from stdin")?;
    let input: StructuredInferRequest =
        serde_json::from_str(&raw).context("parsing inference request JSON")?;
    let cancel = CancellationToken::new();
    let result = HostedClient::default()
        .infer(
            &args.model,
            input,
            InferOptions {
                reasoning_effort: args.reasoning_effort.clone(),
                service_tier: args.service_tier.clone(),
                idle_timeouts: IdleTimeouts {
                    first_progress: Duration::from_secs(args.idle_timeout_secs),
                    inter_chunk: Duration::from_secs(args.stall_timeout_secs),
                },
                validation_retries: args.validation_retries,
            },
            cancel,
        )
        .await
        .map_err(InferErrorContext::from)?;
    serde_json::to_writer(std::io::stdout().lock(), &result)
        .context("writing inference response JSON")?;
    std::io::stdout().lock().write_all(b"\n")?;
    Ok(())
}

struct InferErrorContext(anvil_client::infer::InferError);

impl From<anvil_client::infer::InferError> for InferErrorContext {
    fn from(error: anvil_client::infer::InferError) -> Self {
        Self(error)
    }
}

impl std::fmt::Debug for InferErrorContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::fmt::Display for InferErrorContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::error::Error for InferErrorContext {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_reject_agent_and_tool_history() {
        let error = serde_json::from_value::<StructuredInferRequest>(serde_json::json!({
            "messages": [{"role": "assistant", "content": "prior answer"}],
            "schema_name": "answer",
            "schema": {"type": "object"}
        }))
        .unwrap_err();
        assert!(error.to_string().contains("unknown variant"));
    }

    #[test]
    fn messages_accept_system_and_user_text() {
        let request = serde_json::from_value::<StructuredInferRequest>(serde_json::json!({
            "messages": [
                {"role": "system", "content": "judge carefully"},
                {"role": "user", "content": "item"}
            ],
            "schema_name": "answer",
            "schema": {"type": "object"}
        }))
        .unwrap();
        assert_eq!(request.messages.len(), 2);
    }
}
