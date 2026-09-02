//! Best-effort `GET /user/balance` telemetry for hosted DeepSeek sessions.
//!
//! DeepSeek returns money values as decimal strings. Keep those strings opaque:
//! the ACP payload must preserve precision, formatting, and record order.

use std::sync::{OnceLock, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::discovery::{DEEPSEEK_API_KEY_ENV, DEEPSEEK_BASE_URL};
use crate::llm_client::OpenAiClient;

const CACHE_TTL: Duration = Duration::from_secs(300);

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Balance {
    pub currency: String,
    #[serde(rename(deserialize = "total_balance", serialize = "totalBalance"))]
    pub total_balance: String,
    #[serde(rename(deserialize = "granted_balance", serialize = "grantedBalance"))]
    pub granted_balance: String,
    #[serde(rename(deserialize = "topped_up_balance", serialize = "toppedUpBalance"))]
    pub topped_up_balance: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct BalanceResponse {
    pub is_available: bool,
    pub balance_infos: Vec<Balance>,
}

/// Provider-owned DeepSeek balance telemetry for ACP usage updates.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum Status {
    Available {
        balances: Vec<Balance>,
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

/// Return the cached balance immediately and refresh stale data in the
/// background. Usage updates must never wait on provider HTTP I/O.
pub fn status() -> Status {
    let api_key = active_api_key();
    let fingerprint = api_key.as_deref().map(credential_fingerprint);
    let (status, start_refresh) = {
        let mut cache = cache().write().expect("DeepSeek balance cache poisoned");
        status_from_cache(&mut cache, fingerprint, Instant::now(), as_of())
    };

    if start_refresh {
        let api_key = api_key.expect("a refresh requires a DeepSeek credential");
        let fingerprint = fingerprint.expect("a refresh requires a credential fingerprint");
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(async move {
                    let status = refreshed_status(&api_key).await;
                    let mut cache = cache().write().expect("DeepSeek balance cache poisoned");
                    if cache.credential_fingerprint == Some(fingerprint) {
                        cache.status = Some(status);
                        cache.refreshed = Some(Instant::now());
                        cache.refresh_in_flight = false;
                    }
                });
            }
            Err(_) => {
                let mut cache = cache().write().expect("DeepSeek balance cache poisoned");
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
    fingerprint: Option<[u8; 32]>,
    now: Instant,
    as_of: String,
) -> (Status, bool) {
    let Some(fingerprint) = fingerprint else {
        *cache = Cache::default();
        return (
            unavailable("DeepSeek credentials are unavailable", as_of),
            false,
        );
    };
    if cache.credential_fingerprint != Some(fingerprint) {
        *cache = Cache {
            credential_fingerprint: Some(fingerprint),
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
    (unavailable("refresh pending", as_of), start_refresh)
}

fn credential_fingerprint(api_key: &str) -> [u8; 32] {
    Sha256::digest(api_key.as_bytes()).into()
}

fn as_of() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn unavailable(reason: impl Into<String>, as_of: impl Into<String>) -> Status {
    Status::Unavailable {
        reason: reason.into(),
        as_of: as_of.into(),
    }
}

async fn refreshed_status(api_key: &str) -> Status {
    match query(api_key).await.and_then(status_from_response) {
        Ok(status) => status,
        Err(error) => {
            tracing::debug!(
                reason = error.user_reason(),
                "DeepSeek balance refresh unavailable"
            );
            unavailable(error.user_reason(), as_of())
        }
    }
}

fn status_from_response(response: BalanceResponse) -> Result<Status, QueryError> {
    if !response.is_available {
        return Err(QueryError::ProviderUnavailable);
    }
    if response.balance_infos.iter().any(|balance| {
        balance.currency.trim().is_empty()
            || balance.total_balance.trim().is_empty()
            || balance.granted_balance.trim().is_empty()
            || balance.topped_up_balance.trim().is_empty()
    }) {
        return Err(QueryError::ResponseUnavailable);
    }
    Ok(Status::Available {
        balances: response.balance_infos,
        as_of: as_of(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueryError {
    Unauthorized,
    RateLimited,
    TimedOut,
    ResponseUnavailable,
    ProviderUnavailable,
}

impl QueryError {
    fn user_reason(self) -> &'static str {
        match self {
            Self::Unauthorized => "DeepSeek credentials were rejected",
            Self::RateLimited => "DeepSeek request was rate limited",
            Self::TimedOut => "DeepSeek request timed out",
            Self::ResponseUnavailable => "DeepSeek balance response unavailable",
            Self::ProviderUnavailable => "DeepSeek balance is unavailable",
        }
    }
}

/// Resolve credentials with the same environment-first precedence as the
/// DeepSeek backend. Empty environment values fall through to saved secrets.
pub fn active_api_key() -> Option<String> {
    if let Ok(raw) = std::env::var(DEEPSEEK_API_KEY_ENV) {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    match crate::deepseek_auth::read() {
        Ok(Some(auth)) => {
            let trimmed = auth.api_key.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        _ => None,
    }
}

fn balance_http_client() -> Result<reqwest::Client> {
    OpenAiClient::apply_runtime_tls_workarounds(
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(5)),
        DEEPSEEK_BASE_URL,
    )
    .build()
    .context("building DeepSeek balance HTTP client")
}

/// Fetch the provider response with a short timeout.
#[allow(dead_code)] // Public injection seam for callers outside the refresh path.
pub async fn fetch(api_key: &str) -> Result<BalanceResponse> {
    let http = balance_http_client()?;
    fetch_with(&http, DEEPSEEK_BASE_URL, api_key).await
}

/// Injectable client/base URL seam for unit tests and alternate transports.
#[allow(dead_code)] // Used by tests and available to alternate transports.
pub async fn fetch_with(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
) -> Result<BalanceResponse> {
    fetch_impl(http, base_url, api_key)
        .await
        .map_err(FetchError::into_anyhow)
}

async fn query(api_key: &str) -> Result<BalanceResponse, QueryError> {
    let http = balance_http_client().map_err(|_| QueryError::ResponseUnavailable)?;
    query_with(&http, DEEPSEEK_BASE_URL, api_key).await
}

async fn query_with(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
) -> Result<BalanceResponse, QueryError> {
    fetch_impl(http, base_url, api_key)
        .await
        .map_err(QueryError::from)
}

enum FetchError {
    Transport(reqwest::Error),
    Response(reqwest::StatusCode),
    Parse,
}

impl FetchError {
    fn into_anyhow(self) -> anyhow::Error {
        match self {
            Self::Transport(error) => anyhow!(error).context("GET DeepSeek /user/balance"),
            Self::Response(status) => anyhow!("DeepSeek /user/balance returned HTTP {status}"),
            Self::Parse => anyhow!("parsing DeepSeek /user/balance JSON"),
        }
    }
}

impl From<FetchError> for QueryError {
    fn from(error: FetchError) -> Self {
        match error {
            FetchError::Response(status) if status.as_u16() == 401 || status.as_u16() == 403 => {
                Self::Unauthorized
            }
            FetchError::Response(status) if status.as_u16() == 429 => Self::RateLimited,
            FetchError::Transport(error) if error.is_timeout() => Self::TimedOut,
            FetchError::Transport(_) | FetchError::Response(_) | FetchError::Parse => {
                Self::ResponseUnavailable
            }
        }
    }
}

async fn fetch_impl(
    http: &reqwest::Client,
    base_url: &str,
    api_key: &str,
) -> std::result::Result<BalanceResponse, FetchError> {
    let url = format!("{}/user/balance", base_url.trim_end_matches('/'));
    let response = http
        .get(url)
        .bearer_auth(api_key)
        .send()
        .await
        .map_err(FetchError::Transport)?;
    if !response.status().is_success() {
        return Err(FetchError::Response(response.status()));
    }
    response.json().await.map_err(|_| FetchError::Parse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn official_response() -> serde_json::Value {
        serde_json::json!({"is_available": true, "balance_infos": [
            {"currency":"CNY","total_balance":"123.4500","granted_balance":"23.0000","topped_up_balance":"100.4500"},
            {"currency":"USD","total_balance":"0.010","granted_balance":"0.000","topped_up_balance":"0.010"}
        ]})
    }

    #[tokio::test]
    async fn fetch_preserves_official_multi_currency_strings_and_order() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/user/balance"))
            .and(header("authorization", "Bearer sk-ds-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(official_response()))
            .mount(&server)
            .await;
        let response = fetch_with(&reqwest::Client::new(), &server.uri(), "sk-ds-test")
            .await
            .unwrap();
        let status = status_from_response(response).unwrap();
        let json = serde_json::to_value(status).unwrap();
        assert_eq!(json["balances"][0]["currency"], "CNY");
        assert_eq!(json["balances"][0]["totalBalance"], "123.4500");
        assert_eq!(json["balances"][1]["currency"], "USD");
        assert_eq!(json["balances"][1]["totalBalance"], "0.010");
    }

    #[test]
    fn provider_unavailable_is_safe_unavailable_status() {
        let response: BalanceResponse =
            serde_json::from_value(serde_json::json!({"is_available": false, "balance_infos": []}))
                .unwrap();
        assert_eq!(
            status_from_response(response),
            Err(QueryError::ProviderUnavailable)
        );
    }

    #[tokio::test]
    async fn query_classifies_safe_failures() {
        for (status, expected) in [
            (401, QueryError::Unauthorized),
            (403, QueryError::Unauthorized),
            (429, QueryError::RateLimited),
        ] {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .respond_with(ResponseTemplate::new(status).set_body_string("do not disclose this"))
                .mount(&server)
                .await;
            assert_eq!(
                query_with(&reqwest::Client::new(), &server.uri(), "secret")
                    .await
                    .unwrap_err(),
                expected
            );
        }
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(1)))
            .mount(&server)
            .await;
        let http = reqwest::Client::builder()
            .timeout(Duration::from_millis(10))
            .build()
            .unwrap();
        assert_eq!(
            query_with(&http, &server.uri(), "secret")
                .await
                .unwrap_err(),
            QueryError::TimedOut
        );
    }

    #[tokio::test]
    async fn malformed_and_blank_responses_are_unavailable() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .mount(&server)
            .await;
        assert_eq!(
            query_with(&reqwest::Client::new(), &server.uri(), "secret")
                .await
                .unwrap_err(),
            QueryError::ResponseUnavailable
        );
        let response: BalanceResponse = serde_json::from_value(serde_json::json!({"is_available": true, "balance_infos": [{"currency":" ", "total_balance":"1", "granted_balance":"0", "topped_up_balance":"1"}]})).unwrap();
        assert_eq!(
            status_from_response(response),
            Err(QueryError::ResponseUnavailable)
        );
        let response: BalanceResponse = serde_json::from_value(serde_json::json!({"is_available": true, "balance_infos": [{"currency":"USD", "total_balance":" ", "granted_balance":"0", "topped_up_balance":"1"}]})).unwrap();
        assert_eq!(
            status_from_response(response),
            Err(QueryError::ResponseUnavailable)
        );
    }

    #[test]
    fn cache_is_single_flight_and_invalidates_rotated_credentials() {
        let now = Instant::now();
        let first = credential_fingerprint("first");
        let second = credential_fingerprint("second");
        let mut cache = Cache::default();
        assert!(status_from_cache(&mut cache, Some(first), now, "now".into()).1);
        assert!(!status_from_cache(&mut cache, Some(first), now, "now".into()).1);
        cache.status = Some(Status::Available {
            balances: vec![],
            as_of: "old".into(),
        });
        cache.refreshed = Some(now);
        cache.refresh_in_flight = false;
        assert!(!status_from_cache(&mut cache, Some(first), now, "now".into()).1);
        let (status, starts) = status_from_cache(&mut cache, Some(second), now, "now".into());
        assert!(starts);
        assert!(
            matches!(status, Status::Unavailable { reason, .. } if reason == "refresh pending")
        );
        let (status, starts) = status_from_cache(&mut cache, None, now, "now".into());
        assert!(!starts);
        assert!(
            matches!(status, Status::Unavailable { reason, .. } if reason == "DeepSeek credentials are unavailable")
        );
    }
}
