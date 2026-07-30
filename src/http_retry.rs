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

/// Substring that identifies a *per-day* provider quota in a rejection body.
/// Bedrock/mantle quota ids are `<metric>-<window>:<account>:<model>`, so a
/// daily input-token quota reads `input-tpd:842609633142:openai.gpt-5.6-sol`.
/// The per-minute sibling is `-tpm:`, which is exactly what the retry tiers
/// exist for and must keep its retryable classification.
const DAILY_QUOTA_MARKER: &str = "-tpd:";

/// A provider quota that cannot clear before the run ends.
///
/// A per-day token quota does not reset on any timescale a retry loop can
/// wait out, so treating its 429 as "rate limited, back off and try again"
/// burns the whole retry budget and then hands the caller a failure that is
/// indistinguishable from a transient one. That is how a 2x113-task sweep
/// laundered 186 dead supervisor turns into ordinary-looking `TESTS_FAILED`
/// completions: the Asgard fallback finalized an arbitrary checkpoint
/// because nothing in the error said "this is fatal". This marker says it.
#[derive(Debug)]
pub(crate) struct FatalLlmQuotaError {
    quota: String,
}

impl std::fmt::Display for FatalLlmQuotaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "provider daily token quota exhausted ({}); this quota resets on a day boundary, \
             so retrying cannot recover it -- failing the run instead of continuing without a model",
            self.quota
        )
    }
}

impl std::error::Error for FatalLlmQuotaError {}

/// True when `error` carries a [`FatalLlmQuotaError`] anywhere in its chain.
pub(crate) fn is_fatal_llm_quota_error(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.downcast_ref::<FatalLlmQuotaError>().is_some())
}

/// Extract the daily-quota identifier from a provider rejection body, if it
/// names one. Returns e.g. `input-tpd:842609633142:openai.gpt-5.6-sol`.
pub(crate) fn daily_quota_id(body: &str) -> Option<String> {
    let marker = body.find(DAILY_QUOTA_MARKER)?;
    let is_id_char = |c: char| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':');
    // Walk back over the metric prefix (`input`, `output`, ...) and forward
    // over the account/model suffix, stopping at whatever delimiter the
    // provider used (space, quote, comma).
    let start = body[..marker]
        .char_indices()
        .rev()
        .take_while(|(_, c)| is_id_char(*c))
        .last()
        .map(|(index, _)| index)
        .unwrap_or(marker);
    let end = body[marker..]
        .char_indices()
        .find(|(_, c)| !is_id_char(*c))
        .map(|(offset, _)| marker + offset)
        .unwrap_or(body.len());
    Some(body[start..end].to_string())
}

/// Classify a rejection body as a fatal daily-quota exhaustion, or `None` if
/// it names no per-day quota. Checked ahead of every retryable classification
/// so no status code or body marker can launder it back into a retry.
pub(crate) fn fatal_daily_quota_error(message: &str, body: &str) -> Option<anyhow::Error> {
    let quota = daily_quota_id(body)?;
    Some(anyhow::Error::new(FatalLlmQuotaError { quota }).context(message.to_string()))
}

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
    if let Some(error) = fatal_daily_quota_error(&message, body) {
        return error;
    }
    if contains_gateway_transient_marker(body) {
        return retryable_llm_error(
            message,
            RetryableLlmError::gateway_transient("gateway transient response body"),
        );
    }
    if contains_standard_transient_marker(body) {
        return retryable_llm_error(
            message,
            RetryableLlmError::gateway_transient("standard transient response body"),
        );
    }
    anyhow::anyhow!(message)
}

/// Classify a failed HTTP exchange, preferring the status code over the
/// response body. A 5xx or 429 is retryable by definition, whatever prose
/// the provider puts in the body -- Bedrock, for instance, returns
/// `{"message":"The system encountered an unexpected error during
/// processing. Try your request again."}` on a 500, which matches none of
/// the body markers and would otherwise be treated as terminal. Body
/// markers still apply when the status alone does not settle it (notably
/// providers that report failures inside a 200).
///
/// The one thing that outranks the status is a per-day quota id in the body:
/// a 429 that says `input-tpd:...` is a wall, not a throttle, and no amount
/// of backoff gets past it. See [`FatalLlmQuotaError`].
///
/// A 5xx gets the same patient tier as a 429. `Fast` spends its four
/// attempts inside about 1.4 seconds, which is less patience than a routine
/// provider blip needs: a roughly two-second Bedrock 500 storm killed a
/// supervisor's first turn four minutes into a 90-minute run, and the run
/// delivered a zero-byte patch. `GatewayTransient`'s ~3.5-minute envelope
/// is the right patience for unattended long-running work, and it is still
/// bounded, so a permanent rejection wearing a 500 fails in minutes rather
/// than hanging.
pub(crate) fn retryable_llm_error_for_status_and_body(
    message: impl Into<String>,
    status: reqwest::StatusCode,
    body: &str,
) -> anyhow::Error {
    let message = message.into();
    if let Some(error) = fatal_daily_quota_error(&message, body) {
        return error;
    }
    if status.is_server_error() {
        return retryable_llm_error(
            message,
            RetryableLlmError::gateway_transient("server error status"),
        );
    }
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return retryable_llm_error(
            message,
            RetryableLlmError::gateway_transient("rate limited status"),
        );
    }
    retryable_llm_error_for_body(message, body)
}

pub(crate) fn retryable_llm_error_for_responses_failure(
    message: impl Into<String>,
    failure: &str,
) -> anyhow::Error {
    let message = message.into();
    if let Some(error) = fatal_daily_quota_error(&message, failure) {
        return error;
    }
    if contains_standard_transient_marker(failure) {
        // These markers are the body-borne equivalent of a 5xx/429 status
        // (see `contains_standard_transient_marker`), so they earn the same
        // patient tier the status path already gets. Measured 2026-07-30 on a
        // sol-primary sweep: transient `server_error` streams exhausted the
        // Fast budget (~1.4s across 4 attempts), killed attempts nine turns
        // deep, and surfaced as harness-level infra restarts on 16 of 24
        // tasks — a provider blip should not discard an hour of work.
        return retryable_llm_error(
            message,
            RetryableLlmError::gateway_transient("Responses stream transient failure"),
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

/// Body markers that mean the same thing a 5xx or a 429 status means, and
/// so earn the same patient retry tier: every one of them names an overload
/// or a throttle on the provider's side, never a defect in the request.
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
    request_timeout: Option<Duration>,
) -> Result<reqwest::Response> {
    for attempt in 1..=LLM_MAX_ATTEMPTS {
        let send = make_request().send();
        let response = if let Some(timeout) = request_timeout {
            if let Some(cancel) = cancel {
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        anyhow::bail!("{operation} was cancelled while sending request");
                    }
                    _ = tokio::time::sleep(timeout) => {
                        Err(anyhow::anyhow!(
                            "request produced no response headers within {timeout:?}"
                        ))
                    }
                    response = send => response.with_context(|| operation.to_string()),
                }
            } else {
                match tokio::time::timeout(timeout, send).await {
                    Ok(response) => response.with_context(|| operation.to_string()),
                    Err(_) => Err(anyhow::anyhow!(
                        "request produced no response headers within {timeout:?}"
                    )),
                }
            }
        } else if let Some(cancel) = cancel {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    anyhow::bail!("{operation} was cancelled while sending request");
                }
                response = send => response.with_context(|| operation.to_string()),
            }
        } else {
            send.await.with_context(|| operation.to_string())
        };
        match response {
            Ok(resp) if is_retryable_status(resp.status()) && attempt < LLM_MAX_ATTEMPTS => {
                let status = resp.status();
                sleep_before_retry(operation, attempt, format!("HTTP {status}"), cancel).await?;
            }
            Ok(resp) => return Ok(resp),
            Err(err) if is_retryable_send_error(&err) && attempt < LLM_MAX_ATTEMPTS => {
                let reason = format!("{err:#}");
                sleep_before_retry(operation, attempt, reason, cancel).await?;
            }
            Err(err) => return Err(err),
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

fn is_retryable_send_error(err: &anyhow::Error) -> bool {
    err.chain()
        .filter_map(|cause| cause.downcast_ref::<reqwest::Error>())
        .any(is_retryable_reqwest_error)
        || err
            .to_string()
            .contains("request produced no response headers within")
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
            send_with_retries("slow HTTP test", || http.get(&url), Some(&cancel), None),
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

    #[tokio::test]
    async fn send_with_retries_honors_explicit_request_timeout() {
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

        let started = std::time::Instant::now();
        let err = tokio::time::timeout(
            Duration::from_secs(5),
            send_with_retries(
                "slow HTTP test",
                || http.get(&url),
                None,
                Some(Duration::from_millis(10)),
            ),
        )
        .await
        .expect("explicit request timeout should return before test timeout")
        .expect_err("slow request should fail");

        assert!(
            format!("{err:#}").contains("request produced no response headers within 10ms"),
            "unexpected timeout error: {err:#}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "explicit request timeout waited too long"
        );
    }

    #[test]
    fn bedrock_500_is_retryable_despite_an_unrecognized_body() {
        // Verbatim body Bedrock returns on a 500. It matches none of the
        // body markers, so body-only classification called it terminal and
        // a transient blip killed an entire task on the supervisor's first
        // turn.
        let body = r#"{"message":"The system encountered an unexpected error during processing. Try your request again."}"#;
        assert!(
            !contains_standard_transient_marker(body),
            "body markers must not be what makes this retryable"
        );
        let error = retryable_llm_error_for_status_and_body(
            "Bedrock request failed",
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            body,
        );
        assert_eq!(
            crate::llm_client::llm_retry_tier(&error),
            Some(LlmRetryTier::GatewayTransient)
        );
    }

    /// A 5xx must outlast a provider blip that is longer than the fast
    /// tier's whole 1.4-second budget: the same two-second Bedrock storm
    /// that used to kill a 90-minute run on its first supervisor turn.
    #[test]
    fn server_errors_get_the_patient_tier_from_status_or_body() {
        for status in [
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            reqwest::StatusCode::BAD_GATEWAY,
            reqwest::StatusCode::SERVICE_UNAVAILABLE,
            reqwest::StatusCode::GATEWAY_TIMEOUT,
        ] {
            let error = retryable_llm_error_for_status_and_body("upstream failed", status, "{}");
            assert_eq!(
                crate::llm_client::llm_retry_tier(&error),
                Some(LlmRetryTier::GatewayTransient),
                "{status} should get the patient tier"
            );
        }

        // The body-marker path says the same thing about the same
        // conditions, so it earns the same patience.
        for body in [
            r#"{"error":{"type":"server_error"}}"#,
            r#"{"error":{"type":"server_is_overloaded"}}"#,
            r#"{"error":{"code":"rate_limit_exceeded","message":"Too many requests"}}"#,
        ] {
            let error = retryable_llm_error_for_body("chat completion failed", body);
            assert_eq!(
                crate::llm_client::llm_retry_tier(&error),
                Some(LlmRetryTier::GatewayTransient),
                "{body} should get the patient tier"
            );
        }

        // The patient tier is still bounded: a permanent rejection wearing
        // a 500 fails in minutes rather than retrying forever.
        assert_eq!(
            LlmRetryTier::GatewayTransient.max_attempts(),
            LLM_GATEWAY_TRANSIENT_MAX_ATTEMPTS
        );
    }

    #[test]
    fn rate_limited_status_gets_the_patient_tier() {
        let error = retryable_llm_error_for_status_and_body(
            "rate limited",
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            "{}",
        );
        assert_eq!(
            crate::llm_client::llm_retry_tier(&error),
            Some(LlmRetryTier::GatewayTransient)
        );
    }

    /// Verbatim body the Bedrock mantle endpoint returned on every request
    /// for the rest of the day once the daily input-token quota blew.
    const DAILY_QUOTA_BODY: &str = r#"{"error":{"code":"rate_limit_exceeded","message":"quota input-tpd:842609633142:openai.gpt-5.6-sol (InputTokens) exceeded by 88030.29629390592","param":null,"type":"rate_limit_error"}}"#;

    #[test]
    fn daily_quota_429_is_fatal_not_retryable() {
        let error = retryable_llm_error_for_status_and_body(
            "Bedrock request failed",
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            DAILY_QUOTA_BODY,
        );
        assert!(
            is_fatal_llm_quota_error(&error),
            "daily quota body must carry the fatal marker: {error:#}"
        );
        assert_eq!(
            crate::llm_client::llm_retry_tier(&error),
            None,
            "daily quota must not be retryable: {error:#}"
        );
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("input-tpd:842609633142:openai.gpt-5.6-sol"),
            "fatal quota error should name the quota: {rendered}"
        );
        assert!(
            rendered.contains("daily token quota exhausted"),
            "fatal quota error should say why it is not retryable: {rendered}"
        );
    }

    #[test]
    fn daily_quota_body_stays_fatal_through_the_body_only_path() {
        // `retryable_llm_error_for_body` is the classifier the OpenAI-compatible
        // chat path uses; the body's `rate_limit_exceeded` marker would
        // otherwise re-classify this as a fast retry.
        assert!(contains_standard_transient_marker(DAILY_QUOTA_BODY));
        let error = retryable_llm_error_for_body("chat completion failed", DAILY_QUOTA_BODY);
        assert!(is_fatal_llm_quota_error(&error));
        assert_eq!(crate::llm_client::llm_retry_tier(&error), None);
    }

    #[test]
    fn per_minute_throttle_429_stays_retryable() {
        let body = r#"{"error":{"code":"rate_limit_exceeded","message":"quota input-tpm:842609633142:openai.gpt-5.6-sol (InputTokens) exceeded by 512.5","param":null,"type":"rate_limit_error"}}"#;
        let error = retryable_llm_error_for_status_and_body(
            "Bedrock request failed",
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            body,
        );
        assert!(!is_fatal_llm_quota_error(&error));
        assert_eq!(
            crate::llm_client::llm_retry_tier(&error),
            Some(LlmRetryTier::GatewayTransient),
            "per-minute throttles are exactly what the patient tier is for"
        );
    }

    #[test]
    fn generic_429_stays_retryable() {
        let error = retryable_llm_error_for_status_and_body(
            "rate limited",
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":{"code":"rate_limit_exceeded","message":"Too many requests"}}"#,
        );
        assert!(!is_fatal_llm_quota_error(&error));
        assert_eq!(
            crate::llm_client::llm_retry_tier(&error),
            Some(LlmRetryTier::GatewayTransient)
        );
    }

    #[test]
    fn daily_quota_id_extraction_is_bounded_by_delimiters() {
        assert_eq!(
            daily_quota_id(DAILY_QUOTA_BODY).as_deref(),
            Some("input-tpd:842609633142:openai.gpt-5.6-sol")
        );
        assert_eq!(
            daily_quota_id(r#"{"message":"quota output-tpd:1:m exceeded"}"#).as_deref(),
            Some("output-tpd:1:m")
        );
        assert_eq!(daily_quota_id(r#"{"message":"Too many requests"}"#), None);
        assert_eq!(daily_quota_id("quota input-tpm:1:m exceeded"), None);
    }

    #[test]
    fn client_errors_stay_terminal() {
        // A 400 is a real rejection -- retrying re-sends the same bad
        // request. Body markers still get their say.
        let error = retryable_llm_error_for_status_and_body(
            "bad request",
            reqwest::StatusCode::BAD_REQUEST,
            r#"{"message":"The provided model identifier is invalid."}"#,
        );
        assert!(crate::llm_client::llm_retry_tier(&error).is_none());
    }
}
