use std::time::Duration;

use anyhow::{Context, Result};
use rand::RngExt;
use tokio_util::sync::CancellationToken;

pub(crate) const LLM_MAX_ATTEMPTS: u64 = 4;
pub(crate) const LLM_GATEWAY_TRANSIENT_MAX_ATTEMPTS: u64 = 12;
const REQUEST_RETRY_BASE_DELAY: Duration = Duration::from_millis(200);
const GATEWAY_TRANSIENT_RETRY_BASE_DELAY: Duration = Duration::from_secs(1);
const GATEWAY_TRANSIENT_RETRY_MAX_DELAY: Duration = Duration::from_secs(45);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LlmRetryTier {
    Fast,
    GatewayTransient,
}

impl LlmRetryTier {
    pub(crate) fn max_attempts(self) -> u64 {
        match self {
            Self::Fast => LLM_MAX_ATTEMPTS,
            Self::GatewayTransient => LLM_GATEWAY_TRANSIENT_MAX_ATTEMPTS,
        }
    }
}

#[derive(Debug)]
pub(crate) struct RetryableLlmError {
    tier: LlmRetryTier,
    reason: &'static str,
}

impl RetryableLlmError {
    pub(crate) fn new(tier: LlmRetryTier, reason: &'static str) -> Self {
        Self { tier, reason }
    }

    pub(crate) fn fast(reason: &'static str) -> Self {
        Self::new(LlmRetryTier::Fast, reason)
    }

    pub(crate) fn gateway_transient(reason: &'static str) -> Self {
        Self::new(LlmRetryTier::GatewayTransient, reason)
    }

    pub(crate) fn tier(&self) -> LlmRetryTier {
        self.tier
    }
}

impl std::fmt::Display for RetryableLlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "retryable LLM error ({:?}): {}", self.tier, self.reason)
    }
}

impl std::error::Error for RetryableLlmError {}

pub(crate) fn retryable_llm_error(
    message: impl Into<String>,
    marker: RetryableLlmError,
) -> anyhow::Error {
    anyhow::Error::new(marker).context(message.into())
}

pub(crate) fn retryable_llm_context(
    error: anyhow::Error,
    context: &'static str,
    marker: RetryableLlmError,
) -> anyhow::Error {
    error.context(marker).context(context)
}

pub(crate) fn retryable_llm_error_for_body(
    message: impl Into<String>,
    body: &str,
) -> anyhow::Error {
    let message = message.into();
    if contains_gateway_transient_marker(body) {
        return retryable_llm_error(
            message,
            RetryableLlmError::gateway_transient("gateway transient response body"),
        );
    }
    if contains_standard_transient_marker(body) {
        return retryable_llm_error(
            message,
            RetryableLlmError::fast("standard transient response body"),
        );
    }
    anyhow::anyhow!(message)
}

pub(crate) fn retryable_llm_error_for_responses_failure(
    message: impl Into<String>,
    failure: &str,
) -> anyhow::Error {
    let message = message.into();
    if contains_standard_transient_marker(failure) {
        return retryable_llm_error(
            message,
            RetryableLlmError::fast("Responses stream transient failure"),
        );
    }
    anyhow::anyhow!(message)
}

pub(crate) fn contains_gateway_transient_marker(message: &str) -> bool {
    [
        "JSON-RPC error -32602",
        "Job registration failed",
        "Task submission failed",
        "Engine not found",
    ]
    .iter()
    .any(|marker| message.contains(marker))
}

pub(crate) fn contains_standard_transient_marker(message: &str) -> bool {
    [
        "server_error",
        "server_is_overloaded",
        "rate_limit_exceeded",
    ]
    .iter()
    .any(|marker| message.contains(marker))
}

/// Retry a request until it either succeeds, fails deterministically, or
/// exhausts Codex-compatible retry budget. This is intentionally limited
/// to the pre-stream HTTP exchange: once an SSE body starts producing
/// tokens, replaying here could duplicate user-visible output.
pub(crate) async fn send_with_retries(
    operation: &str,
    mut make_request: impl FnMut() -> reqwest::RequestBuilder,
    cancel: Option<&CancellationToken>,
) -> Result<reqwest::Response> {
    for attempt in 1..=LLM_MAX_ATTEMPTS {
        let send = make_request().send();
        let response = if let Some(cancel) = cancel {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    anyhow::bail!("{operation} was cancelled while sending request");
                }
                response = send => response,
            }
        } else {
            send.await
        };
        match response {
            Ok(resp) if is_retryable_status(resp.status()) && attempt < LLM_MAX_ATTEMPTS => {
                let status = resp.status();
                sleep_before_retry(operation, attempt, format!("HTTP {status}"), cancel).await?;
            }
            Ok(resp) => return Ok(resp),
            Err(err) if is_retryable_reqwest_error(&err) && attempt < LLM_MAX_ATTEMPTS => {
                let reason = err.to_string();
                sleep_before_retry(operation, attempt, reason, cancel).await?;
            }
            Err(err) => return Err(err).with_context(|| operation.to_string()),
        }
    }

    unreachable!("retry loop always returns on the last attempt")
}

fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::REQUEST_TIMEOUT
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status.is_server_error()
}

fn is_retryable_reqwest_error(err: &reqwest::Error) -> bool {
    !err.is_builder() && !err.is_redirect() && !err.is_status() && !err.is_decode()
}

pub(crate) async fn sleep_before_retry(
    operation: &str,
    attempt: u64,
    reason: String,
    cancel: Option<&CancellationToken>,
) -> Result<()> {
    sleep_before_retry_for_tier(operation, LlmRetryTier::Fast, attempt, reason, cancel).await
}

pub(crate) async fn sleep_before_retry_for_tier(
    operation: &str,
    tier: LlmRetryTier,
    attempt: u64,
    reason: String,
    cancel: Option<&CancellationToken>,
) -> Result<()> {
    let delay = retry_backoff_for_tier(tier, attempt);
    let max_attempts = tier.max_attempts();
    tracing::warn!(
        "{operation} failed ({reason}); retrying request ({attempt}/{max_attempts}) in {delay:?}"
    );

    if let Some(cancel) = cancel {
        tokio::select! {
            _ = cancel.cancelled() => {
                anyhow::bail!("{operation} cancelled while waiting to retry");
            }
            _ = tokio::time::sleep(delay) => {}
        }
    } else {
        tokio::time::sleep(delay).await;
    }

    Ok(())
}

pub(crate) fn retry_backoff_for_tier(tier: LlmRetryTier, attempt: u64) -> Duration {
    match tier {
        LlmRetryTier::Fast => retry_backoff(attempt),
        LlmRetryTier::GatewayTransient => gateway_transient_retry_backoff(attempt),
    }
}

pub(crate) fn retry_backoff(attempt: u64) -> Duration {
    let exp = 2u64.saturating_pow(attempt.saturating_sub(1) as u32);
    let raw = REQUEST_RETRY_BASE_DELAY
        .as_millis()
        .saturating_mul(u128::from(exp));
    let jitter = rand::rng().random_range(0.9..1.1);
    Duration::from_millis((raw as f64 * jitter) as u64)
}

pub(crate) fn gateway_transient_retry_backoff(attempt: u64) -> Duration {
    let exp = 2u64.saturating_pow(attempt.saturating_sub(1) as u32);
    let raw = GATEWAY_TRANSIENT_RETRY_BASE_DELAY
        .as_millis()
        .saturating_mul(u128::from(exp))
        .min(GATEWAY_TRANSIENT_RETRY_MAX_DELAY.as_millis());
    let jitter = rand::rng().random_range(0.9..1.1);
    Duration::from_millis((raw as f64 * jitter) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn backoff_uses_codex_base_delay() {
        let first = retry_backoff(1);
        assert!(
            (180..=220).contains(&first.as_millis()),
            "first retry should jitter around 200ms, got {first:?}"
        );

        let second = retry_backoff(2);
        assert!(
            (360..=440).contains(&second.as_millis()),
            "second retry should jitter around 400ms, got {second:?}"
        );
    }

    #[test]
    fn gateway_transient_backoff_uses_long_capped_schedule() {
        assert_eq!(LlmRetryTier::Fast.max_attempts(), LLM_MAX_ATTEMPTS);
        assert_eq!(
            LlmRetryTier::GatewayTransient.max_attempts(),
            LLM_GATEWAY_TRANSIENT_MAX_ATTEMPTS
        );

        let first = gateway_transient_retry_backoff(1);
        assert!(
            (900..=1100).contains(&first.as_millis()),
            "first gateway retry should jitter around 1s, got {first:?}"
        );

        let capped = gateway_transient_retry_backoff(10);
        assert!(
            (40_500..=49_500).contains(&capped.as_millis()),
            "gateway retry should cap around 45s with jitter, got {capped:?}"
        );
    }

    #[test]
    fn gateway_transient_marker_detection_is_narrow() {
        assert!(contains_gateway_transient_marker(
            "chat completion failed (HTTP 400): JSON-RPC error -32602: Job registration failed"
        ));
        assert!(contains_gateway_transient_marker(
            "chat completion failed (HTTP 400): Engine not found"
        ));
        assert!(!contains_gateway_transient_marker(
            "chat completion failed (HTTP 400): invalid request: missing messages"
        ));
    }

    #[test]
    fn status_retry_policy_includes_transient_failures() {
        assert!(is_retryable_status(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
        ));
        assert!(is_retryable_status(reqwest::StatusCode::REQUEST_TIMEOUT));
        assert!(is_retryable_status(reqwest::StatusCode::TOO_MANY_REQUESTS));
        assert!(!is_retryable_status(reqwest::StatusCode::BAD_REQUEST));
    }

    #[test]
    fn stream_error_context_preserves_retry_marker() {
        let err = retryable_llm_context(
            anyhow::anyhow!("connection reset"),
            "Codex stream read error",
            RetryableLlmError::fast("Codex stream read error"),
        );

        assert!(
            crate::llm_client::is_retryable_llm_error(&err),
            "retry marker was lost from error chain: {err:#}"
        );
    }

    #[test]
    fn shared_attempt_budget_uses_four_total_attempts() {
        assert_eq!(LLM_MAX_ATTEMPTS, 4);
    }

    #[tokio::test]
    async fn send_with_retries_honors_cancellation_during_send() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/slow"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string("eventually")
                    .set_delay(Duration::from_secs(30)),
            )
            .mount(&server)
            .await;

        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("build reqwest client");
        let url = format!("{}/slow", server.uri());
        let cancel = CancellationToken::new();
        let cancel_from_task = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            cancel_from_task.cancel();
        });

        let started = std::time::Instant::now();
        let err = tokio::time::timeout(
            Duration::from_secs(5),
            send_with_retries("slow HTTP test", || http.get(&url), Some(&cancel)),
        )
        .await
        .expect("cancelled HTTP request should return before test timeout")
        .expect_err("cancelled HTTP request should fail");

        assert!(
            format!("{err:#}").contains("cancelled while sending request"),
            "unexpected cancellation error: {err:#}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "cancelled HTTP request waited too long"
        );
    }
}
