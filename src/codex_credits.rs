//! `GET chatgpt.com/backend-api/codex/wham/usage` query for the `/usage`
//! slash command.
//!
//! Surfaces the same numbers Codex CLI shows in `/status`: which
//! ChatGPT plan the OAuth token is attached to, whether the account
//! has Codex credits remaining, and how much of the primary rate-limit
//! window has been consumed. Best-effort by design: callers handle the
//! `Result` and degrade the report rather than failing the whole
//! slash command.
//!
//! Auth mirrors `codex_client`: Bearer access token, `ChatGPT-Account-ID`
//! header, `originator: codex_cli_rs`. We refresh past-staleness tokens
//! proactively before the call -- failing to do so means `/usage` 401s
//! every time on accounts whose tokens are >8 days old until the next
//! agent prompt happens to refresh them. Refresh failures are logged
//! and we still attempt the fetch with whatever's on disk; the 401 then
//! surfaces in the report so the user can re-authenticate.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::codex_auth::{AuthDotJson, read_auth_dot_json, refresh_if_stale};
use crate::llm_client::OpenAiClient;

/// Endpoint Codex CLI hits for plan + rate-limit info. Spelled out
/// rather than concatenated from a shared base so the URL is greppable
/// verbatim alongside `codex_client::CHATGPT_RESPONSES_URL`.
const CHATGPT_WHAM_USAGE_URL: &str = "https://chatgpt.com/backend-api/codex/wham/usage";

/// Originator header Codex CLI sends. The server gates ChatGPT-
/// subscription usage on this identity (alongside the OAuth token) so
/// matching it is the difference between getting JSON back and being
/// rejected as an unrecognized client. Duplicated from `codex_client`
/// rather than re-exported so this module compiles independently.
const ORIGINATOR: &str = "codex_cli_rs";

/// Top-level shape of `GET /wham/usage`. Mirrors codex-rs's
/// `RateLimitStatusPayload` but only models the fields we render.
/// `#[serde(default)]` on every optional sub-struct keeps the
/// deserializer permissive: a server-side schema addition (new
/// `spend_control` variant, new `rate_limit_reached_type` token)
/// deserializes successfully and our render path just doesn't surface
/// it, rather than failing the whole `/usage` command.
#[derive(Debug, Clone, Deserialize)]
pub struct CodexUsage {
    /// `plan_type` -- e.g. `"plus"`, `"pro"`, `"team"`, `"enterprise"`.
    /// Free-form on the wire so a new plan slug doesn't break parsing;
    /// we round-trip whatever the server says.
    #[serde(default)]
    pub plan_type: Option<String>,
    #[serde(default)]
    pub credits: Option<CreditStatus>,
    #[serde(default)]
    pub rate_limit: Option<RateLimitStatus>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct CreditStatus {
    #[serde(default)]
    pub has_credits: bool,
    #[serde(default)]
    pub unlimited: bool,
    /// USD-denominated balance as a server-formatted string (e.g. `"12.34"`).
    /// Kept as `String` rather than `f64` because the server omits the
    /// field on plans without metered credits (Pro/Team/etc.) and we
    /// want to preserve whatever precision/format it sent.
    #[serde(default)]
    pub balance: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct RateLimitStatus {
    #[serde(default)]
    pub primary_window: Option<RateLimitWindow>,
}

/// One window snapshot. `used_percent` is the headline number we
/// render. `reset_after_seconds` lets us format a human-friendly "resets
/// in 2h 15m" without dragging a clock into the renderer.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RateLimitWindow {
    #[serde(default)]
    pub used_percent: i32,
    #[serde(default)]
    pub reset_after_seconds: i32,
}

/// Credentials lifted out of `auth.json`. Kept private so the
/// access token never escapes this module's public surface.
#[derive(Debug, Clone)]
struct ChatGptCredentials {
    access_token: String,
    account_id: String,
}

impl ChatGptCredentials {
    fn from_auth(auth: &AuthDotJson) -> Result<Self> {
        let tokens = auth
            .tokens
            .as_ref()
            .context("auth.json has no `tokens` block; run /setup codex again")?;
        if tokens.access_token.is_empty() {
            bail!("auth.json `tokens.access_token` is empty");
        }
        if tokens.account_id.is_empty() {
            bail!("auth.json `tokens.account_id` is empty");
        }
        Ok(Self {
            access_token: tokens.access_token.clone(),
            account_id: tokens.account_id.clone(),
        })
    }
}

fn is_chatgpt_mode(auth: &AuthDotJson) -> bool {
    matches!(auth.auth_mode.as_deref(), Some("chatgpt"))
}

/// Classification of `~/.codex/auth.json` for the `/usage` renderer.
/// The renderer needs to tell three skip reasons apart so it can show
/// an actionable hint:
/// - `Missing`: no auth.json at all -> "run /setup codex"
/// - `ApiKeyMode`: auth.json present but billing through OPENAI_API_KEY
///   -> tell the user `wham/usage` doesn't apply and they should check
///   their OpenAI dashboard instead
/// - `ChatGptMode`: ready to hit the endpoint
///
/// Returned by `auth_status` so the agent layer doesn't have to read
/// auth.json a second time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthStatus {
    Missing,
    ApiKeyMode,
    ChatGptMode,
}

/// Inspect `~/.codex/auth.json` and report which auth path is active.
/// Cheap; meant to be called by `/usage` rendering before deciding
/// whether to hit the network.
pub fn auth_status() -> Result<AuthStatus> {
    match read_auth_dot_json()? {
        None => Ok(AuthStatus::Missing),
        Some(auth) if is_chatgpt_mode(&auth) => Ok(AuthStatus::ChatGptMode),
        Some(_) => Ok(AuthStatus::ApiKeyMode),
    }
}

/// Short-timeout HTTP client tuned for the `/usage` slash command: the
/// user is staring at the prompt waiting for the report, so we'd rather
/// fail in a few seconds than block on a stuck TLS handshake. The 5s
/// `timeout` is the total request budget (connect + response);
/// `connect_timeout` is the tighter inner cap so a stuck handshake
/// fails fast without burning the full window. The cookie jar matches
/// `CodexClient`'s posture -- the ChatGPT backend sits behind
/// Cloudflare's bot manager, which gets unhappy when we drop the
/// `__cf_bm` cookie it sets on the first response.
fn credits_http_client() -> Result<reqwest::Client> {
    let user_agent = format!(
        "{ORIGINATOR}/{ver} (brokk-acp; {os})",
        ver = env!("CARGO_PKG_VERSION"),
        os = std::env::consts::OS,
    );
    OpenAiClient::apply_runtime_tls_workarounds(
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(5))
            .cookie_store(true)
            .user_agent(user_agent),
        CHATGPT_WHAM_USAGE_URL,
    )
    .build()
    .context("building ChatGPT credits HTTP client")
}

/// Hit the live ChatGPT backend using whatever credentials are in
/// `~/.codex/auth.json`. Returns `Ok(None)` when the user isn't on the
/// ChatGPT auth path (no auth.json, or apikey mode) so the slash-command
/// renderer can show a clean "not applicable" line. Refreshes stale
/// tokens proactively before the call -- without it `/usage` 401s on
/// accounts whose tokens have aged past Codex's 8-day window until
/// the next agent prompt happens to refresh them.
pub async fn fetch() -> Result<Option<CodexUsage>> {
    let Some(mut auth) = read_auth_dot_json()? else {
        return Ok(None);
    };
    if !is_chatgpt_mode(&auth) {
        return Ok(None);
    }
    if let Err(e) = refresh_if_stale(&mut auth).await {
        // Best-effort: a refresh failure here doesn't fail the whole
        // `/usage` lookup. Either the tokens are still valid (the
        // staleness check is conservative), in which case the request
        // succeeds and the user gets their report, or they really are
        // expired and the upstream 401 will surface in the diagnostic.
        tracing::warn!("codex token refresh during /usage failed: {e:#}");
    }
    let creds = ChatGptCredentials::from_auth(&auth)?;
    let http = credits_http_client()?;
    Ok(Some(
        fetch_with(&http, CHATGPT_WHAM_USAGE_URL, &creds).await?,
    ))
}

/// Same as `fetch` but lets callers inject the HTTP client and full
/// endpoint URL so the unit tests can point at a `wiremock` server.
/// Credentials remain private to the module; tests construct them via
/// the `mock_credentials` helper below.
async fn fetch_with(
    http: &reqwest::Client,
    url: &str,
    creds: &ChatGptCredentials,
) -> Result<CodexUsage> {
    let resp = http
        .get(url)
        .header("Authorization", format!("Bearer {}", creds.access_token))
        .header("ChatGPT-Account-ID", &creds.account_id)
        .header("originator", ORIGINATOR)
        .header("Accept", "application/json")
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        // Cap the body excerpt and strip newlines so a Cloudflare
        // challenge HTML page doesn't drown the slash-command output.
        let excerpt: String = body
            .chars()
            .take(200)
            .collect::<String>()
            .replace('\n', " ");
        bail!("chatgpt /wham/usage returned HTTP {status}: {excerpt}");
    }
    serde_json::from_str::<CodexUsage>(&body).context("parsing /wham/usage JSON")
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn mock_credentials() -> ChatGptCredentials {
        ChatGptCredentials {
            access_token: "sk-chatgpt-test".to_string(),
            account_id: "acc_test".to_string(),
        }
    }

    #[tokio::test]
    async fn fetch_parses_documented_shape() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/wham/usage"))
            .and(header("authorization", "Bearer sk-chatgpt-test"))
            .and(header("chatgpt-account-id", "acc_test"))
            .and(header("originator", "codex_cli_rs"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "plan_type": "plus",
                "credits": {
                    "has_credits": true,
                    "unlimited": false,
                    "balance": "12.50"
                },
                "rate_limit": {
                    "primary_window": {
                        "used_percent": 42,
                        "limit_window_seconds": 86400,
                        "reset_after_seconds": 7200,
                        "reset_at": 0
                    }
                }
            })))
            .mount(&server)
            .await;

        let http = reqwest::Client::new();
        let url = format!("{}/wham/usage", server.uri());
        let usage = fetch_with(&http, &url, &mock_credentials())
            .await
            .expect("ok");
        assert_eq!(usage.plan_type.as_deref(), Some("plus"));
        let credits = usage.credits.expect("credits present");
        assert!(credits.has_credits);
        assert!(!credits.unlimited);
        assert_eq!(credits.balance.as_deref(), Some("12.50"));
        let window = usage
            .rate_limit
            .and_then(|r| r.primary_window)
            .expect("primary window present");
        assert_eq!(window.used_percent, 42);
        assert_eq!(window.reset_after_seconds, 7200);
    }

    /// Pro/Team plans don't include a `credits` block. The deserializer
    /// must accept that quietly -- this is the most common shape in
    /// production traffic and a strict requirement would break /usage
    /// for any subscription-tier account.
    #[tokio::test]
    async fn fetch_tolerates_missing_credits_block() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/wham/usage"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "plan_type": "pro",
                "rate_limit": {
                    "primary_window": {
                        "used_percent": 5,
                        "limit_window_seconds": 86400,
                        "reset_after_seconds": 60,
                        "reset_at": 0
                    }
                }
            })))
            .mount(&server)
            .await;

        let http = reqwest::Client::new();
        let url = format!("{}/wham/usage", server.uri());
        let usage = fetch_with(&http, &url, &mock_credentials())
            .await
            .expect("ok");
        assert_eq!(usage.plan_type.as_deref(), Some("pro"));
        assert!(usage.credits.is_none());
        assert_eq!(
            usage
                .rate_limit
                .and_then(|r| r.primary_window)
                .map(|w| w.used_percent),
            Some(5)
        );
    }

    /// New plan strings (`"go"`, `"prolite"`, ...) must round-trip
    /// without falling into a hardcoded enum -- we model `plan_type`
    /// as a free-form String for exactly this reason.
    #[tokio::test]
    async fn fetch_round_trips_unknown_plan_string() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/wham/usage"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "plan_type": "experimental_new_tier"
            })))
            .mount(&server)
            .await;

        let http = reqwest::Client::new();
        let url = format!("{}/wham/usage", server.uri());
        let usage = fetch_with(&http, &url, &mock_credentials())
            .await
            .expect("ok");
        assert_eq!(usage.plan_type.as_deref(), Some("experimental_new_tier"));
    }

    /// A slow upstream must NOT hang the `/usage` slash command for the
    /// 30+ seconds reqwest waits by default -- the `credits_http_client`
    /// caps the request at 5s so a stuck Cloudflare/ChatGPT response
    /// fails fast and the renderer can show a "lookup failed" line
    /// instead of leaving the user staring at a blank prompt.
    /// Regression test for the original /usage hang where the client
    /// had no enforced wall-clock budget.
    #[tokio::test]
    async fn credits_http_client_enforces_total_timeout() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/wham/usage"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({}))
                    .set_delay(Duration::from_secs(30)),
            )
            .mount(&server)
            .await;

        let http = credits_http_client().expect("client builds");
        let url = format!("{}/wham/usage", server.uri());
        let start = std::time::Instant::now();
        let err = fetch_with(&http, &url, &mock_credentials())
            .await
            .expect_err("must time out before the 30s server delay");
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(8),
            "timeout took too long: {elapsed:?}; client budget is meant to be ~5s"
        );
        let msg = format!("{err:#}").to_lowercase();
        assert!(
            msg.contains("timed out") || msg.contains("timeout"),
            "expected timeout wording in: {msg}"
        );
    }

    /// Non-2xx responses surface the status and a short body excerpt so
    /// the slash-command report can show a useful diagnostic instead of
    /// the generic "request failed". A 401 here usually means the OAuth
    /// token went stale between agent calls -- the message should hint
    /// at re-auth without spilling the bearer token into the log.
    #[tokio::test]
    async fn fetch_surfaces_upstream_status_and_body_excerpt() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/wham/usage"))
            .respond_with(ResponseTemplate::new(401).set_body_string("invalid_token: refresh me"))
            .mount(&server)
            .await;

        let http = reqwest::Client::new();
        let url = format!("{}/wham/usage", server.uri());
        let err = fetch_with(&http, &url, &mock_credentials())
            .await
            .expect_err("must propagate upstream failure");
        let msg = format!("{err:#}");
        assert!(msg.contains("401"), "missing status: {msg}");
        assert!(msg.contains("invalid_token"), "missing body: {msg}");
        // The bearer token must never appear in user-facing diagnostics.
        assert!(
            !msg.contains("sk-chatgpt-test"),
            "bearer token leaked: {msg}"
        );
    }

    /// A long HTML body (e.g. a Cloudflare challenge page) must be
    /// truncated and de-newlined so the slash-command output stays
    /// readable instead of dumping kilobytes of markup into the
    /// terminal.
    #[tokio::test]
    async fn fetch_truncates_long_html_bodies_in_error() {
        let server = MockServer::start().await;
        let html = "<html>\n".to_string() + &"x".repeat(5_000) + "\n</html>";
        Mock::given(method("GET"))
            .and(path("/wham/usage"))
            .respond_with(ResponseTemplate::new(403).set_body_string(html))
            .mount(&server)
            .await;

        let http = reqwest::Client::new();
        let url = format!("{}/wham/usage", server.uri());
        let err = fetch_with(&http, &url, &mock_credentials())
            .await
            .expect_err("must propagate upstream failure");
        let msg = format!("{err:#}");
        assert!(msg.contains("403"), "missing status: {msg}");
        // Excerpt is capped at 200 chars; total error string should be
        // well under 1 KiB even with the boilerplate framing.
        assert!(
            msg.len() < 1_024,
            "error message too long: {} bytes",
            msg.len()
        );
        assert!(!msg.contains('\n'), "newlines not stripped: {msg}");
    }
}
