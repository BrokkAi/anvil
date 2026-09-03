//! One-shot, tool-free structured inference over Anvil's hosted backends.

use std::io::{Read, Write};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Args;
use tokio_util::sync::CancellationToken;

use std::sync::Arc;

use anvil_llm::codex_client::CodexClient;
use anvil_llm::infer::{InferOptions, StructuredInferRequest, infer_structured};
use anvil_llm::llm_client::{IdleTimeouts, LlmBackend};

const CODEX_MODEL_PREFIX: &str = "codex::";
const KIMI_MODEL_PREFIX: &str = "kimi::";
const GROK_MODEL_PREFIX: &str = "grok::";
const DEEPSEEK_MODEL_PREFIX: &str = "deepseek::";

#[derive(Args, Debug)]
pub(crate) struct InferArgs {
    /// Provider-qualified wire model id (codex::, kimi::, grok::, or deepseek::).
    #[arg(long)]
    model: String,

    /// Reasoning effort forwarded in the selected provider's dialect.
    #[arg(long)]
    reasoning_effort: Option<String>,

    /// Optional service tier. Omit this flag to use the provider default.
    #[arg(long)]
    service_tier: Option<String>,

    /// Seconds to wait for the first meaningful response event.
    #[arg(long, default_value_t = anvil_llm::llm_client::DEFAULT_IDLE_CHUNK_TIMEOUT_SECS)]
    idle_timeout_secs: u64,

    /// Seconds to wait between meaningful response events.
    #[arg(long, default_value_t = anvil_llm::llm_client::DEFAULT_INTER_CHUNK_TIMEOUT_SECS)]
    stall_timeout_secs: u64,

    /// Additional attempts after local structured-output validation fails.
    #[arg(long, default_value_t = 1)]
    validation_retries: usize,
}

pub(crate) async fn run(args: &InferArgs) -> Result<()> {
    // Route on the provider prefix so the same tool-free, schema-constrained
    // path can judge with any explicitly supported hosted backend. The
    // prefix is required in every case to prevent provider
    // fallback picking a different model than the caller pinned.
    let (backend, wire_model): (Arc<dyn LlmBackend>, String) = if let Some(model) =
        args.model.strip_prefix(CODEX_MODEL_PREFIX)
    {
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
    } else if let Some(model) = args.model.strip_prefix(GROK_MODEL_PREFIX) {
        if model.trim().is_empty() {
            bail!("--model must name a model after the grok:: prefix");
        }
        let backend = crate::build_grok_backend().ok_or_else(|| {
            anyhow::anyhow!(
                "Grok backend is not configured; install Grok Build and run `grok login --oauth`"
            )
        })?;
        (backend, model.to_string())
    } else if let Some(model) = args.model.strip_prefix(DEEPSEEK_MODEL_PREFIX) {
        if model.trim().is_empty() {
            bail!("--model must name a model after the deepseek:: prefix");
        }
        let backend = crate::build_deepseek_backend().ok_or_else(|| {
                anyhow::anyhow!(
                    "DeepSeek backend is not configured; set DEEPSEEK_API_KEY or run `/setup deepseek key <key>`"
                )
            })?;
        (backend, model.to_string())
    } else {
        bail!(
            "--model must use a codex::<model-id>, kimi::<model-id>, grok::<model-id>, or deepseek::<model-id> wire form"
        );
    };

    let mut raw = String::new();
    std::io::stdin()
        .read_to_string(&mut raw)
        .context("reading inference request from stdin")?;
    let input: StructuredInferRequest =
        serde_json::from_str(&raw).context("parsing inference request JSON")?;
    let cancel = CancellationToken::new();
    let mut result = infer_structured(
        backend.as_ref(),
        wire_model,
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
    result.model.clone_from(&args.model);
    serde_json::to_writer(std::io::stdout().lock(), &result)
        .context("writing inference response JSON")?;
    std::io::stdout().lock().write_all(b"\n")?;
    Ok(())
}

struct InferErrorContext(anvil_llm::infer::InferError);

impl From<anvil_llm::infer::InferError> for InferErrorContext {
    fn from(error: anvil_llm::infer::InferError) -> Self {
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
