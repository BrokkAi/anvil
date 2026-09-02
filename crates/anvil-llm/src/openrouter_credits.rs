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

use std::sync::{OnceLock, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::discovery::{OPENROUTER_API_KEY_ENV, OPENROUTER_BASE_URL};
use crate::llm_client::OpenAiClient;

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

const CACHE_TTL: Duration = Duration::from_secs(300);

/// Provider-owned OpenRouter balance telemetry for ACP usage updates.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum Status {
    Available {
        #[serde(rename = "remainingUsd")]
        remaining_usd: f64,
        #[serde(rename = "totalCreditsUsd")]
        total_credits_usd: f64,
        #[serde(rename = "totalUsageUsd")]
        total_usage_usd: f64,
        #[serde(rename = "asOf")]
        as_of: String,
    },
    Unavailable {
        reason: String,
        #[serde(rename = "asOf")]
        as_of: String,
    },
}

#[derive(Default)]
struct Cache {
    status: Option<Status>,
    refreshed: Option<Instant>,
    refresh_in_flight: bool,
    credential_fingerprint: Option<[u8; 32]>,
}

static CACHE: OnceLock<RwLock<Cache>> = OnceLock::new();

fn cache() -> &'static RwLock<Cache> {
    CACHE.get_or_init(|| RwLock::new(Cache::default()))
}

/// Return cached OpenRouter balance telemetry immediately, refreshing stale
/// values in the background. This deliberately never waits for HTTP I/O.
pub fn status() -> Status {
    let api_key = active_api_key();
    let fingerprint = api_key.as_deref().map(credential_fingerprint);
    let (status, start_refresh) = {
        let mut cache = cache().write().expect("OpenRouter credit cache poisoned");
        status_from_cache(&mut cache, fingerprint, Instant::now(), as_of())
    };

    if start_refresh {
        let api_key = api_key.expect("a refresh requires an OpenRouter credential");
        let fingerprint = fingerprint.expect("a refresh requires a credential fingerprint");
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(async move {
                    let status = refreshed_status(&api_key).await;
                    let mut cache = cache().write().expect("OpenRouter credit cache poisoned");
                    // A rotated credential starts its own refresh. Do not let the old
                    // request overwrite the newer credential's cache entry.
                    if cache.credential_fingerprint == Some(fingerprint) {
                        cache.status = Some(status);
                        cache.refreshed = Some(Instant::now());
                        cache.refresh_in_flight = false;
                    }
                });
            }
            Err(_) => {
                // Do not leave the cache permanently in-flight during shutdown or
                // synchronous test use; a later runtime-backed call can retry.
                let mut cache = cache().write().expect("OpenRouter credit cache poisoned");
                if cache.credential_fingerprint == Some(fingerprint) {
                    cache.refresh_in_flight = false;
                }
            }
        }
    }

    status
}

fn status_from_cache(
    cache: &mut Cache,
    credential_fingerprint: Option<[u8; 32]>,
    now: Instant,
    as_of: String,
) -> (Status, bool) {
    let Some(credential_fingerprint) = credential_fingerprint else {
        // Do not retain a no-credential outcome: adding a credential must start a
        // fresh lookup immediately rather than waiting for a stale TTL.
        *cache = Cache::default();
        return (
            Status::Unavailable {
                reason: "OpenRouter credentials are unavailable".to_string(),
                as_of,
            },
            false,
        );
    };

    if cache.credential_fingerprint != Some(credential_fingerprint) {
        *cache = Cache {
            credential_fingerprint: Some(credential_fingerprint),
            ..Cache::default()
        };
    }

    let fresh = cache
        .refreshed
        .is_some_and(|refreshed| now.duration_since(refreshed) < CACHE_TTL);
    if let Some(status) = cache.status.clone() {
        if fresh {
            return (status, false);
        }
        let start_refresh = !cache.refresh_in_flight;
        cache.refresh_in_flight = true;
        return (status, start_refresh);
    }

    let start_refresh = !cache.refresh_in_flight;
    cache.refresh_in_flight = true;
    (
        Status::Unavailable {
            reason: "refresh pending".to_string(),
            as_of,
        },
        start_refresh,
    )
}

fn credential_fingerprint(api_key: &str) -> [u8; 32] {
    Sha256::digest(api_key.as_bytes()).into()
}

fn as_of() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

async fn refreshed_status(api_key: &str) -> Status {
    match query(api_key).await.and_then(status_from_credits) {
        Ok(status) => status,
        Err(error) => {
            tracing::debug!(
                reason = error.user_reason(),
                "OpenRouter balance refresh unavailable"
            );
            Status::Unavailable {
                reason: error.user_reason().to_string(),
                as_of: as_of(),
            }
        }
    }
}

fn status_from_credits(credits: Credits) -> Result<Status, QueryError> {
    let remaining_usd = credits.balance();
    if !credits.total_credits.is_finite()
        || !credits.total_usage.is_finite()
        || !remaining_usd.is_finite()
    {
        return Err(QueryError::ResponseUnavailable);
    }
    Ok(Status::Available {
        remaining_usd,
        total_credits_usd: credits.total_credits,
        total_usage_usd: credits.total_usage,
        as_of: as_of(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueryError {
    Unauthorized,
    RateLimited,
    TimedOut,
    ResponseUnavailable,
}

impl QueryError {
    fn user_reason(self) -> &'static str {
        match self {
            Self::Unauthorized => "OpenRouter credentials were rejected",
            Self::RateLimited => "OpenRouter request was rate limited",
            Self::TimedOut => "OpenRouter request timed out",
            Self::ResponseUnavailable => "OpenRouter balance response unavailable",
        }
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
/// handshake. The 5s `timeout` is the total request budget
/// (connect + response); `connect_timeout` is the tighter inner cap so
/// a stuck handshake fails fast without burning the full window.
fn credits_http_client() -> Result<reqwest::Client> {
    OpenAiClient::apply_runtime_tls_workarounds(
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(5)),
        OPENROUTER_BASE_URL,
    )
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
    fetch_impl(http, base_url, api_key)
        .await
        .map_err(FetchError::into_anyhow)
}

async fn query(api_key: &str) -> Result<Credits, QueryError> {
    let http = credits_http_client().map_err(|_| QueryError::ResponseUnavailable)?;
    query_with(&http, OPENROUTER_BASE_URL, api_key).await
}

async fn query_with(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
) -> Result<Credits, QueryError> {
    fetch_impl(http, base_url, api_key)
        .await
        .map_err(QueryError::from)
}

enum FetchError {
    Transport {
        error: reqwest::Error,
        url: String,
    },
    Response {
        status: reqwest::StatusCode,
        excerpt: String,
    },
    Parse(reqwest::Error),
}

impl FetchError {
    fn into_anyhow(self) -> anyhow::Error {
        match self {
            Self::Transport { error, url } => anyhow!(error).context(format!("GET {url}")),
            Self::Response { status, excerpt } => {
                anyhow!("openrouter /credits returned HTTP {status}: {excerpt}")
            }
            Self::Parse(error) => anyhow!(error).context("parsing /credits JSON"),
        }
    }
}

impl From<FetchError> for QueryError {
    fn from(error: FetchError) -> Self {
        match error {
            FetchError::Response { status, .. }
                if status.as_u16() == 401 || status.as_u16() == 403 =>
            {
                Self::Unauthorized
            }
            FetchError::Response { status, .. } if status.as_u16() == 429 => Self::RateLimited,
            FetchError::Transport { error, .. } if error.is_timeout() => Self::TimedOut,
            FetchError::Transport { .. } | FetchError::Response { .. } | FetchError::Parse(_) => {
                Self::ResponseUnavailable
            }
        }
    }
}

async fn fetch_impl(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
) -> std::result::Result<Credits, FetchError> {
    let url = format!("{}/credits", base_url.trim_end_matches('/'));
    let resp = http
        .get(&url)
        .bearer_auth(api_key)
        .send()
        .await
        .map_err(|error| FetchError::Transport {
            error,
            url: url.clone(),
        })?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        let excerpt: String = body.chars().take(200).collect();
        return Err(FetchError::Response { status, excerpt });
    }
    let parsed: CreditsEnvelope = resp.json().await.map_err(FetchError::Parse)?;
    Ok(parsed.data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openrouter_auth::test_support::{ENV_GUARD, EnvScope};
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

    /// A slow upstream must NOT hang the `/usage` slash command for the
    /// 30+ seconds reqwest waits by default -- the `credits_http_client`
    /// caps the request at 5s so a stuck server fails fast and the
    /// renderer can show a "lookup failed" line instead of leaving the
    /// user staring at a blank prompt. Regression test for the original
    /// /usage hang where the client had no enforced wall-clock budget.
    #[tokio::test]
    async fn credits_http_client_enforces_total_timeout() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/credits"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({
                        "data": {"total_credits": 1.0, "total_usage": 0.5}
                    }))
                    .set_delay(Duration::from_secs(30)),
            )
            .mount(&server)
            .await;

        let http = credits_http_client().expect("client builds");
        let start = std::time::Instant::now();
        let err = fetch_with(&http, &server.uri(), "sk-or-test")
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

    #[test]
    fn status_emits_documented_shape_and_accepts_zero_balance() {
        let status = status_from_credits(Credits {
            total_credits: 12.5,
            total_usage: 12.5,
        })
        .expect("zero is an available balance");
        let json = serde_json::to_value(status).expect("serializable status");
        assert_eq!(json["status"], "available");
        assert_eq!(json["remainingUsd"], 0.0);
        assert_eq!(json["totalCreditsUsd"], 12.5);
        assert_eq!(json["totalUsageUsd"], 12.5);
        assert!(json["asOf"].is_string());
    }

    #[test]
    fn non_finite_numbers_cannot_become_available() {
        for credits in [
            Credits {
                total_credits: f64::NAN,
                total_usage: 1.0,
            },
            Credits {
                total_credits: f64::INFINITY,
                total_usage: 1.0,
            },
            Credits {
                total_credits: f64::MAX,
                total_usage: -f64::MAX,
            },
        ] {
            assert_eq!(
                status_from_credits(credits).expect_err("non-finite values are unavailable"),
                QueryError::ResponseUnavailable
            );
        }
    }

    #[tokio::test]
    async fn malformed_response_is_unavailable_for_telemetry() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/credits"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not JSON"))
            .mount(&server)
            .await;

        let error = query_with(&reqwest::Client::new(), &server.uri(), "sk-or-test")
            .await
            .expect_err("malformed JSON must not produce telemetry");
        assert_eq!(error, QueryError::ResponseUnavailable);
    }

    #[tokio::test]
    async fn query_classifies_safe_upstream_failures() {
        for (status, expected) in [
            (401, QueryError::Unauthorized),
            (403, QueryError::Unauthorized),
            (429, QueryError::RateLimited),
        ] {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/credits"))
                .respond_with(ResponseTemplate::new(status).set_body_string("secret provider body"))
                .mount(&server)
                .await;
            let error = query_with(&reqwest::Client::new(), &server.uri(), "sk-or-test")
                .await
                .expect_err("non-success response");
            assert_eq!(error, expected);
            assert!(!error.user_reason().contains("secret provider body"));
        }

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/credits"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(1)))
            .mount(&server)
            .await;
        let short_client = reqwest::Client::builder()
            .timeout(Duration::from_millis(10))
            .build()
            .expect("client builds");
        let error = query_with(&short_client, &server.uri(), "sk-or-test")
            .await
            .expect_err("request must time out");
        assert_eq!(error, QueryError::TimedOut);
        assert_eq!(error.user_reason(), "OpenRouter request timed out");
    }

    #[test]
    fn cache_cold_fresh_stale_and_single_flight_behaviour() {
        let now = Instant::now();
        let fingerprint = credential_fingerprint("first credential");
        let mut cache = Cache::default();
        let (cold, start_refresh) =
            status_from_cache(&mut cache, Some(fingerprint), now, "cold".into());
        assert_eq!(cold, unavailable("refresh pending", "cold"));
        assert!(start_refresh);
        assert!(cache.refresh_in_flight);

        let (_, second_refresh) =
            status_from_cache(&mut cache, Some(fingerprint), now, "again".into());
        assert!(!second_refresh, "only one cold refresh may run");

        let fresh = available(5.0, "fresh");
        cache.status = Some(fresh.clone());
        cache.refreshed = Some(now);
        cache.refresh_in_flight = false;
        let (status, start_refresh) =
            status_from_cache(&mut cache, Some(fingerprint), now, "new".into());
        assert_eq!(status, fresh);
        assert!(!start_refresh);

        cache.refreshed = Some(now - CACHE_TTL);
        let (status, start_refresh) =
            status_from_cache(&mut cache, Some(fingerprint), now, "stale".into());
        assert_eq!(status, fresh, "stale value remains immediately usable");
        assert!(start_refresh);
    }

    #[test]
    fn missing_or_rotated_credential_resets_cache_without_exposing_key() {
        let now = Instant::now();
        let first = credential_fingerprint("first credential");
        let second = credential_fingerprint("rotated credential");
        let mut cache = Cache {
            status: Some(available(9.0, "old")),
            refreshed: Some(now),
            refresh_in_flight: false,
            credential_fingerprint: Some(first),
        };

        let (rotated, start_refresh) =
            status_from_cache(&mut cache, Some(second), now, "rotated".into());
        assert_eq!(rotated, unavailable("refresh pending", "rotated"));
        assert!(
            start_refresh,
            "a replacement credential starts a fresh request"
        );
        assert_eq!(cache.credential_fingerprint, Some(second));
        assert!(cache.status.is_none(), "old credential data is discarded");

        let (missing, start_refresh) = status_from_cache(&mut cache, None, now, "missing".into());
        assert_eq!(
            missing,
            unavailable("OpenRouter credentials are unavailable", "missing")
        );
        assert!(!start_refresh, "missing credentials never start a request");
        assert!(cache.status.is_none());
        assert!(cache.credential_fingerprint.is_none());

        let (_, start_refresh) =
            status_from_cache(&mut cache, Some(second), now, "re-added".into());
        assert!(
            start_refresh,
            "a newly added credential starts a fresh request"
        );
        assert_eq!(cache.credential_fingerprint, Some(second));
        let debug = format!("{:?}", cache.credential_fingerprint);
        assert!(!debug.contains("rotated credential"));
    }

    #[test]
    fn synchronous_status_call_clears_in_flight_refresh() {
        let _env_lock = ENV_GUARD.blocking_lock();
        let _key = EnvScope::set(OPENROUTER_API_KEY_ENV, "test credential");
        *cache().write().expect("cache lock") = Cache::default();

        let status = status();

        assert!(
            matches!(status, Status::Unavailable { ref reason, .. } if reason == "refresh pending")
        );
        assert!(
            !cache().read().expect("cache lock").refresh_in_flight,
            "without a Tokio runtime the next call must be able to retry"
        );
    }

    fn available(remaining_usd: f64, as_of: &str) -> Status {
        Status::Available {
            remaining_usd,
            total_credits_usd: remaining_usd,
            total_usage_usd: 0.0,
            as_of: as_of.into(),
        }
    }

    fn unavailable(reason: &str, as_of: &str) -> Status {
        Status::Unavailable {
            reason: reason.into(),
            as_of: as_of.into(),
        }
    }
}
