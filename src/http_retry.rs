use std::time::Duration;

use anyhow::{Context, Result};
use rand::RngExt;
use tokio_util::sync::CancellationToken;

pub(crate) const LLM_MAX_ATTEMPTS: u64 = 4;
const REQUEST_RETRY_BASE_DELAY: Duration = Duration::from_millis(200);

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
                    anyhow::bail!("{operation} cancelled while sending request");
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
    let delay = retry_backoff(attempt);
    tracing::warn!(
        "{operation} failed ({reason}); retrying request ({attempt}/{LLM_MAX_ATTEMPTS}) in {delay:?}"
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

pub(crate) fn retry_backoff(attempt: u64) -> Duration {
    let exp = 2u64.saturating_pow(attempt.saturating_sub(1) as u32);
    let raw = REQUEST_RETRY_BASE_DELAY
        .as_millis()
        .saturating_mul(u128::from(exp));
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
    fn status_retry_policy_includes_transient_failures() {
        assert!(is_retryable_status(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
        ));
        assert!(is_retryable_status(reqwest::StatusCode::REQUEST_TIMEOUT));
        assert!(is_retryable_status(reqwest::StatusCode::TOO_MANY_REQUESTS));
        assert!(!is_retryable_status(reqwest::StatusCode::BAD_REQUEST));
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
