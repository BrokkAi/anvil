use std::time::Duration;

use anyhow::{Context, Result};
use rand::Rng;
use tokio_util::sync::CancellationToken;

const DEFAULT_REQUEST_MAX_RETRIES: u64 = 4;
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
    for attempt in 0..=DEFAULT_REQUEST_MAX_RETRIES {
        match make_request().send().await {
            Ok(resp)
                if is_retryable_status(resp.status()) && attempt < DEFAULT_REQUEST_MAX_RETRIES =>
            {
                let status = resp.status();
                sleep_before_retry(operation, attempt, format!("HTTP {status}"), cancel).await?;
            }
            Ok(resp) => return Ok(resp),
            Err(err)
                if is_retryable_reqwest_error(&err) && attempt < DEFAULT_REQUEST_MAX_RETRIES =>
            {
                let reason = err.to_string();
                sleep_before_retry(operation, attempt, reason, cancel).await?;
            }
            Err(err) => return Err(err).with_context(|| operation.to_string()),
        }
    }

    unreachable!("retry loop always returns on the last attempt")
}

fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::REQUEST_TIMEOUT || status.is_server_error()
}

fn is_retryable_reqwest_error(err: &reqwest::Error) -> bool {
    !err.is_builder() && !err.is_redirect() && !err.is_status() && !err.is_decode()
}

async fn sleep_before_retry(
    operation: &str,
    attempt: u64,
    reason: String,
    cancel: Option<&CancellationToken>,
) -> Result<()> {
    let retry_number = attempt + 1;
    let delay = backoff(retry_number);
    tracing::warn!(
        "{operation} failed ({reason}); retrying request ({retry_number}/{DEFAULT_REQUEST_MAX_RETRIES}) in {delay:?}"
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

fn backoff(attempt: u64) -> Duration {
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

    #[test]
    fn backoff_uses_codex_base_delay() {
        let first = backoff(1);
        assert!(
            (180..=220).contains(&first.as_millis()),
            "first retry should jitter around 200ms, got {first:?}"
        );

        let second = backoff(2);
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
        assert!(!is_retryable_status(reqwest::StatusCode::TOO_MANY_REQUESTS));
        assert!(!is_retryable_status(reqwest::StatusCode::BAD_REQUEST));
    }
}
