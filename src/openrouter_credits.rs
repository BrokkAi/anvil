//! `GET /api/v1/credits` query for the `/usage` slash command.
//!
//! OpenRouter returns the absolute lifetime numbers (`total_credits`
//! purchased, `total_usage` consumed) and leaves the subtraction to the
//! client; `Credits::balance` does that once so the report path doesn't
//! repeat the arithmetic. Best-effort by design: callers handle the
//! `Result` and degrade the report rather than failing the whole
//! command.
//!
//! Auth uses the same precedence as `build_openrouter_backend` (env
//! wins over the on-disk credential file) so users see a balance
//! regardless of which credential source is currently active.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::discovery::{OPENROUTER_API_KEY_ENV, OPENROUTER_BASE_URL};

/// `data` wrapper on the wire; flatten it so callers see plain fields.
#[derive(Debug, Deserialize)]
struct CreditsEnvelope {
    data: Credits,
}

/// Numbers returned by `GET /api/v1/credits`. Both are USD-denominated
/// floats per the OpenRouter docs ("total credits purchased" / "total
/// credits used"); the remaining balance is `total_credits -
/// total_usage`.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
pub struct Credits {
    pub total_credits: f64,
    pub total_usage: f64,
}

impl Credits {
    pub fn balance(&self) -> f64 {
        self.total_credits - self.total_usage
    }
}

/// Resolve the active OpenRouter API key with the same env-then-file
/// precedence as `main::build_openrouter_backend`. Whitespace-only
/// values are treated as unset so an accidentally exported
/// `OPENROUTER_API_KEY=""` doesn't shadow the on-disk credential file.
pub fn active_api_key() -> Option<String> {
    if let Ok(raw) = std::env::var(OPENROUTER_API_KEY_ENV) {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    match crate::openrouter_auth::read() {
        Ok(Some(auth)) => {
            let trimmed = auth.api_key.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        _ => None,
    }
}

/// Build a short-timeout HTTP client tuned for the `/usage` slash
/// command: the user is staring at the prompt waiting for the report,
/// so we'd rather fail in a few seconds than block on a stuck TLS
/// handshake.
fn credits_http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(3))
        .timeout(Duration::from_secs(8))
        .build()
        .context("building OpenRouter credits HTTP client")
}

/// `GET /api/v1/credits` against the live OpenRouter API using `api_key`
/// as a bearer token. Returns the parsed numbers on 2xx; otherwise
/// surfaces the upstream status with a short body excerpt so callers
/// can render a useful diagnostic in the slash-command report (a bare
/// "request failed" doesn't help a user who pasted a bad key).
pub async fn fetch(api_key: &str) -> Result<Credits> {
    let http = credits_http_client()?;
    fetch_with(&http, OPENROUTER_BASE_URL, api_key).await
}

/// Same as `fetch` but lets callers inject the HTTP client and base
/// URL so the unit tests can point at a `wiremock` server.
pub async fn fetch_with(http: &reqwest::Client, base_url: &str, api_key: &str) -> Result<Credits> {
    let url = format!("{}/credits", base_url.trim_end_matches('/'));
    let resp = http
        .get(&url)
        .bearer_auth(api_key)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        let excerpt: String = body.chars().take(200).collect();
        bail!("openrouter /credits returned HTTP {status}: {excerpt}");
    }
    let parsed: CreditsEnvelope = resp.json().await.context("parsing /credits JSON")?;
    Ok(parsed.data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn balance_is_credits_minus_usage() {
        let c = Credits {
            total_credits: 100.5,
            total_usage: 25.75,
        };
        assert!((c.balance() - 74.75).abs() < 1e-9);
    }

    #[tokio::test]
    async fn fetch_parses_documented_shape() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/credits"))
            .and(header("authorization", "Bearer sk-or-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {"total_credits": 100.5, "total_usage": 25.75}
            })))
            .mount(&server)
            .await;

        let http = reqwest::Client::new();
        let credits = fetch_with(&http, &server.uri(), "sk-or-test")
            .await
            .expect("ok");
        assert_eq!(credits.total_credits, 100.5);
        assert_eq!(credits.total_usage, 25.75);
        assert!((credits.balance() - 74.75).abs() < 1e-9);
    }

    /// Trailing-slash base URLs must not produce `//credits` -- the
    /// upstream tolerates it today but pinning the join behaviour here
    /// keeps a future regression from sending double slashes.
    #[tokio::test]
    async fn fetch_strips_trailing_slash_on_base_url() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/credits"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {"total_credits": 5.0, "total_usage": 1.0}
            })))
            .mount(&server)
            .await;

        let http = reqwest::Client::new();
        let with_slash = format!("{}/", server.uri());
        let credits = fetch_with(&http, &with_slash, "sk-or-test")
            .await
            .expect("ok");
        assert_eq!(credits.total_credits, 5.0);
    }

    /// Non-2xx responses surface the status and a body excerpt so the
    /// slash-command report can show a useful diagnostic instead of
    /// the generic "request failed".
    #[tokio::test]
    async fn fetch_surfaces_upstream_status_and_body_excerpt() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/credits"))
            .respond_with(ResponseTemplate::new(401).set_body_string("No auth credentials found"))
            .mount(&server)
            .await;

        let http = reqwest::Client::new();
        let err = fetch_with(&http, &server.uri(), "sk-or-bad")
            .await
            .expect_err("must propagate upstream failure");
        let msg = format!("{err:#}");
        assert!(msg.contains("401"), "missing status: {msg}");
        assert!(
            msg.contains("No auth credentials found"),
            "missing body: {msg}"
        );
    }
}
