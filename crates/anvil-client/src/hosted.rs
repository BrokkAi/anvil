//! Shared hosted-provider construction for Anvil and language bindings.

use crate::llm_client::LlmBackend;
use crate::{deepseek_auth, discovery, grok_client, kimi_auth, llm_client};
use std::sync::Arc;

/// Build a hosted DeepSeek chat backend from a raw API key. DeepSeek's API
/// is OpenAI-compatible at `https://api.deepseek.com`, but its reasoning knob
/// is spelled in DeepSeek's own dialect (`thinking` + a top-level
/// `reasoning_effort` on the `high`/`max` scale), so we build the client with
/// the DeepSeek reasoning wire rather than the unified one.
pub fn deepseek_backend_from_key(raw: &str) -> Option<Arc<dyn LlmBackend>> {
    let key = raw.trim();
    if key.is_empty() {
        return None;
    }
    Some(Arc::new(
        llm_client::OpenAiClient::with_deepseek_reasoning_support(
            discovery::DEEPSEEK_BASE_URL.to_string(),
            Some(key.to_string()),
            reqwest::header::HeaderMap::new(),
        ),
    ))
}

/// Build the hosted DeepSeek backend from `DEEPSEEK_API_KEY`, falling back
/// to the consolidated secrets store (written by `/setup deepseek key`).
/// Precedence matches OpenRouter and Bedrock: env > file > nothing.
pub fn build_deepseek_backend() -> Option<Arc<dyn LlmBackend>> {
    if let Ok(raw) = std::env::var(discovery::DEEPSEEK_API_KEY_ENV) {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            tracing::info!(
                "{} is set but empty; falling back to the secrets store",
                discovery::DEEPSEEK_API_KEY_ENV
            );
        } else {
            tracing::info!(
                "DeepSeek backend wired from {} at {} (chat + discovery); key length={}",
                discovery::DEEPSEEK_API_KEY_ENV,
                discovery::DEEPSEEK_BASE_URL,
                trimmed.len()
            );
            return deepseek_backend_from_key(trimmed);
        }
    }

    match deepseek_auth::read() {
        Ok(Some(auth)) => {
            let trimmed = auth.api_key.trim();
            if trimmed.is_empty() {
                tracing::info!(
                    "DeepSeek entry in the secrets store has an empty key; backend skipped"
                );
                return None;
            }
            tracing::info!(
                "DeepSeek backend wired from the secrets store at {} (chat + discovery); key length={}",
                discovery::DEEPSEEK_BASE_URL,
                trimmed.len()
            );
            deepseek_backend_from_key(trimmed)
        }
        Ok(None) => None,
        Err(e) => {
            tracing::warn!("failed to read the secrets store for DeepSeek: {e:#}");
            None
        }
    }
}

pub fn build_kimi_backend() -> Option<Arc<dyn LlmBackend>> {
    let auth = match kimi_auth::load_provider() {
        Ok(auth) => auth,
        Err(error) => {
            tracing::warn!("failed to configure Kimi authentication: {error:#}");
            return None;
        }
    }?;
    let headers = match kimi_auth::default_headers() {
        Ok(headers) => headers,
        Err(error) => {
            tracing::warn!("failed to configure Kimi request headers: {error:#}");
            return None;
        }
    };
    let base_url = kimi_auth::base_url();
    tracing::info!(
        base_url,
        "Kimi backend wired from KIMI_API_KEY or Kimi Code credentials"
    );
    Some(Arc::new(llm_client::OpenAiClient::with_kimi_support(
        base_url, auth, headers,
    )))
}

pub fn build_grok_backend() -> Option<Arc<dyn LlmBackend>> {
    match grok_client::GrokClient::load() {
        Ok(backend) => backend,
        Err(error) => {
            tracing::warn!("failed to configure Grok OAuth authentication: {error:#}");
            None
        }
    }
}
