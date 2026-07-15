//! Cached AWS Billing `GetCredits` telemetry for Bedrock ACP sessions.

use std::collections::BTreeMap;
use std::sync::{OnceLock, RwLock};
use std::time::{Duration, Instant, SystemTime};

use aws_credential_types::Credentials;
use aws_credential_types::provider::ProvideCredentials;
use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(8);
const CACHE_TTL: Duration = Duration::from_secs(300);
const BILLING_ENDPOINT: &str = "https://billing.us-east-1.api.aws/";
const STS_ENDPOINT: &str = "https://sts.amazonaws.com/";

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum Status {
    Available {
        amounts: Vec<CreditAmount>,
        #[serde(rename = "earliestExpiration", skip_serializing_if = "Option::is_none")]
        earliest_expiration: Option<String>,
        #[serde(rename = "asOf")]
        as_of: String,
    },
    Unavailable {
        reason: String,
        #[serde(rename = "asOf")]
        as_of: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CreditAmount {
    pub currency: String,
    pub amount: f64,
}

#[derive(Debug)]
pub enum QueryError {
    MissingCredentials,
    AccessDenied,
    TimedOut,
    Unsupported,
    Protocol,
}

impl QueryError {
    pub fn user_reason(&self) -> &'static str {
        match self {
            Self::MissingCredentials => "billing credentials are unavailable",
            Self::AccessDenied => "access denied (requires billing:GetCredits)",
            Self::TimedOut => "request timed out",
            Self::Unsupported => "AWS Billing GetCredits is unavailable",
            Self::Protocol => "billing response unavailable",
        }
    }
}

#[derive(Default)]
struct Cache {
    status: Option<Status>,
    refreshed: Option<Instant>,
    refresh_in_flight: bool,
}

static CACHE: OnceLock<RwLock<Cache>> = OnceLock::new();

fn cache() -> &'static RwLock<Cache> {
    CACHE.get_or_init(|| RwLock::new(Cache::default()))
}

/// Return cached credit status immediately and refresh stale data in the background.
///
/// The prompt path must not wait for AWS credential resolution or Billing requests.
pub fn status() -> Status {
    let (status, start_refresh) = {
        let mut cache = cache().write().expect("bedrock credit cache poisoned");
        status_from_cache(&mut cache, Instant::now(), as_of())
    };
    if start_refresh {
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(async {
                    let status = refreshed_status().await;
                    let mut cache = cache().write().expect("bedrock credit cache poisoned");
                    cache.status = Some(status);
                    cache.refreshed = Some(Instant::now());
                    cache.refresh_in_flight = false;
                });
            }
            Err(_) => {
                // This API is normally called from the ACP Tokio runtime. Do not leave
                // the cache permanently in-flight if it is called during shutdown.
                cache()
                    .write()
                    .expect("bedrock credit cache poisoned")
                    .refresh_in_flight = false;
            }
        }
    }
    status
}

fn status_from_cache(cache: &mut Cache, now: Instant, as_of: String) -> (Status, bool) {
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

async fn refreshed_status() -> Status {
    match query().await {
        Ok(report) => Status::Available {
            amounts: report.amounts,
            earliest_expiration: report.earliest_expiration,
            as_of: as_of(),
        },
        Err(error) => {
            tracing::debug!(
                reason = error.user_reason(),
                "Bedrock credit refresh unavailable"
            );
            Status::Unavailable {
                reason: error.user_reason().to_string(),
                as_of: as_of(),
            }
        }
    }
}

fn as_of() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

struct Report {
    amounts: Vec<CreditAmount>,
    earliest_expiration: Option<String>,
}

async fn query() -> Result<Report, QueryError> {
    tokio::time::timeout(REQUEST_TIMEOUT, query_inner())
        .await
        .map_err(|_| QueryError::TimedOut)?
}

async fn query_inner() -> Result<Report, QueryError> {
    let credential_http_client = aws_smithy_http_client::Builder::new()
        .tls_provider(aws_smithy_http_client::tls::Provider::Rustls(
            aws_smithy_http_client::tls::rustls_provider::CryptoMode::Ring,
        ))
        .build_https();
    let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .http_client(credential_http_client)
        .load()
        .await;
    let provider = config
        .credentials_provider()
        .ok_or(QueryError::MissingCredentials)?;
    let credentials = provider
        .provide_credentials()
        .await
        .map_err(|_| QueryError::MissingCredentials)?;
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|_| QueryError::Protocol)?;
    let account_id = caller_account_id(&client, &credentials).await?;
    let start_date = (Utc::now() - ChronoDuration::days(364)).timestamp();
    let body =
        serde_json::to_vec(&serde_json::json!({"accountId": account_id, "startDate": start_date}))
            .map_err(|_| QueryError::Protocol)?;
    let response = send_signed(
        &client,
        &credentials,
        BILLING_ENDPOINT,
        "billing",
        "application/x-amz-json-1.0",
        Some("AWSBilling.GetCredits"),
        body,
    )
    .await?;
    parse_report(&response, Utc::now())
}

async fn caller_account_id(
    client: &reqwest::Client,
    credentials: &Credentials,
) -> Result<String, QueryError> {
    let response = send_signed(
        client,
        credentials,
        STS_ENDPOINT,
        "sts",
        "application/x-www-form-urlencoded; charset=utf-8",
        None,
        b"Action=GetCallerIdentity&Version=2011-06-15".to_vec(),
    )
    .await?;
    extract_xml_tag(&response, "Account").ok_or(QueryError::Protocol)
}

async fn send_signed(
    client: &reqwest::Client,
    credentials: &Credentials,
    endpoint: &str,
    service: &str,
    content_type: &str,
    target: Option<&str>,
    body: Vec<u8>,
) -> Result<Vec<u8>, QueryError> {
    let mut builder = http::Request::builder()
        .method("POST")
        .uri(endpoint)
        .header("content-type", content_type);
    if let Some(target) = target {
        builder = builder.header("x-amz-target", target);
    }
    let mut request = builder
        .body(body.clone())
        .map_err(|_| QueryError::Protocol)?;
    let headers = request
        .headers()
        .iter()
        .filter_map(|(name, value)| value.to_str().ok().map(|value| (name.as_str(), value)));
    let signable = SignableRequest::new("POST", endpoint, headers, SignableBody::Bytes(&body))
        .map_err(|_| QueryError::Protocol)?;
    let identity = Identity::from(credentials.clone());
    let params = v4::SigningParams::builder()
        .identity(&identity)
        .region("us-east-1")
        .name(service)
        .time(SystemTime::now())
        .settings(SigningSettings::default())
        .build()
        .map_err(|_| QueryError::Protocol)?
        .into();
    let (instructions, _) = sign(signable, &params)
        .map_err(|_| QueryError::Protocol)?
        .into_parts();
    instructions.apply_to_request_http1x(&mut request);
    let mut request_builder = client.post(endpoint).body(body);
    for (name, value) in request.headers() {
        request_builder = request_builder.header(name, value);
    }
    let response = request_builder.send().await.map_err(classify_transport)?;
    let http_status = response.status();
    let bytes = response.bytes().await.map_err(classify_transport)?;
    if http_status.is_success() {
        Ok(bytes.to_vec())
    } else {
        Err(classify_response(http_status.as_u16(), &bytes))
    }
}

fn classify_transport(error: reqwest::Error) -> QueryError {
    if error.is_timeout() {
        QueryError::TimedOut
    } else {
        QueryError::Protocol
    }
}
fn classify_response(status: u16, body: &[u8]) -> QueryError {
    let lower = String::from_utf8_lossy(body).to_ascii_lowercase();
    if status == 401
        || status == 403
        || lower.contains("accessdenied")
        || lower.contains("access denied")
    {
        QueryError::AccessDenied
    } else if status == 404
        || lower.contains("unknownoperation")
        || lower.contains("unsupportedoperation")
    {
        QueryError::Unsupported
    } else {
        QueryError::Protocol
    }
}
fn extract_xml_tag(body: &[u8], tag: &str) -> Option<String> {
    let body = std::str::from_utf8(body).ok()?;
    let value = body
        .split_once(&format!("<{tag}>"))?
        .1
        .split_once(&format!("</{tag}>"))?
        .0;
    (!value.is_empty()).then(|| value.to_string())
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GetCreditsResponse {
    #[serde(default)]
    credits: Vec<Credit>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Credit {
    remaining_amount: Option<Amount>,
    estimated_amount: Option<Amount>,
    #[serde(default)]
    applicable_product_names: Vec<String>,
    credit_status: Option<String>,
    end_date: Option<f64>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Amount {
    currency_amount: String,
    currency_code: String,
}

fn parse_report(body: &[u8], now: DateTime<Utc>) -> Result<Report, QueryError> {
    let response: GetCreditsResponse =
        serde_json::from_slice(body).map_err(|_| QueryError::Protocol)?;
    let mut totals = BTreeMap::<String, f64>::new();
    let mut earliest = None;
    for credit in response.credits {
        if !is_bedrock(&credit) || !is_active(&credit, now.timestamp() as f64) {
            continue;
        }
        let Some(amount) = credit
            .estimated_amount
            .as_ref()
            .or(credit.remaining_amount.as_ref())
        else {
            continue;
        };
        let Ok(value) = amount.currency_amount.parse::<f64>() else {
            continue;
        };
        let currency = amount.currency_code.trim();
        if currency.is_empty() || !value.is_finite() {
            continue;
        }
        *totals.entry(currency.to_string()).or_default() += value;
        if let Some(end) = credit.end_date.filter(|end| end.is_finite())
            && earliest.is_none_or(|old| end < old)
        {
            earliest = Some(end);
        }
    }
    Ok(Report {
        amounts: totals
            .into_iter()
            .map(|(currency, amount)| CreditAmount { currency, amount })
            .collect(),
        earliest_expiration: earliest.and_then(format_date),
    })
}
fn is_bedrock(credit: &Credit) -> bool {
    credit
        .applicable_product_names
        .iter()
        .any(|name| name.eq_ignore_ascii_case("Amazon Bedrock"))
}
fn is_active(credit: &Credit, now: f64) -> bool {
    let enabled = credit
        .credit_status
        .as_deref()
        .is_none_or(|status| status.eq_ignore_ascii_case("enabled"));
    enabled
        && credit
            .end_date
            .is_none_or(|end| end.is_finite() && end >= now)
}
fn format_date(timestamp: f64) -> Option<String> {
    chrono::DateTime::from_timestamp(timestamp as i64, 0)
        .map(|value| value.date_naive().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_aggregates_and_filters_documented_credits() {
        let now = DateTime::parse_from_rfc3339("2026-07-15T18:42:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let report = parse_report(br#"{"credits":[{"remainingAmount":{"currencyAmount":"2","currencyCode":"USD"},"estimatedAmount":{"currencyAmount":"3.5","currencyCode":"USD"},"applicableProductNames":["Amazon Bedrock"],"endDate":1796083200},{"remainingAmount":{"currencyAmount":"4","currencyCode":"EUR"},"applicableProductNames":["Amazon Bedrock"],"creditStatus":"DISABLED","endDate":1796083200},{"remainingAmount":{"currencyAmount":"4","currencyCode":"EUR"},"applicableProductNames":["Amazon Bedrock"],"creditStatus":"EXPIRED"},{"remainingAmount":{"currencyAmount":"2","currencyCode":"USD"},"applicableProductNames":["Amazon EC2"]}]}"#, now).unwrap();
        assert_eq!(
            report.amounts,
            vec![CreditAmount {
                currency: "USD".into(),
                amount: 3.5
            }]
        );
        assert_eq!(report.earliest_expiration.as_deref(), Some("2026-12-01"));
    }

    #[test]
    fn accepts_only_absent_or_enabled_credit_statuses() {
        let now = DateTime::parse_from_rfc3339("2026-07-15T18:42:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let report = parse_report(
            br#"{"credits":[
                {"remainingAmount":{"currencyAmount":"1","currencyCode":"USD"},"applicableProductNames":["Amazon Bedrock"]},
                {"remainingAmount":{"currencyAmount":"2","currencyCode":"USD"},"applicableProductNames":["Amazon Bedrock"],"creditStatus":"ENABLED"},
                {"remainingAmount":{"currencyAmount":"4","currencyCode":"USD"},"applicableProductNames":["Amazon Bedrock"],"creditStatus":"DISABLED"},
                {"remainingAmount":{"currencyAmount":"8","currencyCode":"USD"},"applicableProductNames":["Amazon Bedrock"],"creditStatus":"EXPIRED"}
            ]}"#,
            now,
        )
        .unwrap();
        assert_eq!(
            report.amounts,
            vec![CreditAmount {
                currency: "USD".into(),
                amount: 3.0,
            }]
        );
    }
    #[test]
    fn classifies_unavailable_errors() {
        assert_eq!(
            QueryError::MissingCredentials.user_reason(),
            "billing credentials are unavailable"
        );
        assert_eq!(
            classify_response(403, b"AccessDeniedException").user_reason(),
            "access denied (requires billing:GetCredits)"
        );
        assert_eq!(QueryError::TimedOut.user_reason(), "request timed out");
        assert_eq!(
            QueryError::Protocol.user_reason(),
            "billing response unavailable"
        );
    }
    #[test]
    fn serializes_available_and_unavailable_status() {
        let available = Status::Available {
            amounts: vec![CreditAmount {
                currency: "USD".into(),
                amount: 12.5,
            }],
            earliest_expiration: Some("2026-12-31".into()),
            as_of: "2026-07-15T18:42:00Z".into(),
        };
        assert_eq!(
            serde_json::to_value(&available).unwrap()["amounts"][0]["currency"],
            "USD"
        );
        let unavailable = Status::Unavailable {
            reason: "request timed out".into(),
            as_of: "2026-07-15T18:42:00Z".into(),
        };
        assert_eq!(
            serde_json::to_value(unavailable).unwrap()["status"],
            "unavailable"
        );
    }

    #[test]
    fn cold_cache_returns_pending_and_starts_one_refresh() {
        let now = Instant::now();
        let mut cache = Cache::default();

        let (status, start_refresh) =
            status_from_cache(&mut cache, now, "2026-07-15T18:42:00Z".into());

        assert_eq!(
            status,
            Status::Unavailable {
                reason: "refresh pending".into(),
                as_of: "2026-07-15T18:42:00Z".into(),
            }
        );
        assert!(start_refresh);
        assert!(cache.refresh_in_flight);
    }

    #[test]
    fn in_flight_refresh_is_not_requested_twice() {
        let now = Instant::now();
        let mut cache = Cache {
            refresh_in_flight: true,
            ..Cache::default()
        };

        let (_, start_refresh) = status_from_cache(&mut cache, now, "as-of".into());

        assert!(!start_refresh);
        assert!(cache.refresh_in_flight);
    }

    #[test]
    fn stale_cache_returns_status_while_starting_one_refresh() {
        let now = Instant::now();
        let stale_status = Status::Unavailable {
            reason: "request timed out".into(),
            as_of: "2026-07-15T18:00:00Z".into(),
        };
        let mut cache = Cache {
            status: Some(stale_status.clone()),
            refreshed: Some(now - CACHE_TTL),
            refresh_in_flight: false,
        };

        let (status, start_refresh) = status_from_cache(&mut cache, now, "new-as-of".into());

        assert_eq!(status, stale_status);
        assert!(start_refresh);
        assert!(cache.refresh_in_flight);
    }

    #[test]
    fn fresh_cache_returns_status_without_refresh() {
        let now = Instant::now();
        let fresh_status = Status::Available {
            amounts: vec![CreditAmount {
                currency: "USD".into(),
                amount: 1.0,
            }],
            earliest_expiration: None,
            as_of: "2026-07-15T18:42:00Z".into(),
        };
        let mut cache = Cache {
            status: Some(fresh_status.clone()),
            refreshed: Some(now),
            refresh_in_flight: false,
        };

        let (status, start_refresh) = status_from_cache(&mut cache, now, "new-as-of".into());

        assert_eq!(status, fresh_status);
        assert!(!start_refresh);
        assert!(!cache.refresh_in_flight);
    }
}
