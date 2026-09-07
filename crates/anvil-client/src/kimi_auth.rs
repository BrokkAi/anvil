//! Kimi Code authentication backed by `KIMI_API_KEY` or the Kimi CLI's
//! short-lived OAuth credential file.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};

use crate::llm_client::{BearerTokenProvider, LlmBackend, OpenAiClient};

pub const KIMI_API_KEY_ENV: &str = "KIMI_API_KEY";
pub const KIMI_CODE_BASE_URL_ENV: &str = "KIMI_CODE_BASE_URL";
pub const DEFAULT_KIMI_CODE_BASE_URL: &str = "https://api.kimi.com/coding/v1";
const KIMI_CODE_HOME_ENV: &str = "KIMI_CODE_HOME";
const KIMI_OAUTH_HOST_ENV: &str = "KIMI_CODE_OAUTH_HOST";
const KIMI_OAUTH_HOST_FALLBACK_ENV: &str = "KIMI_OAUTH_HOST";
const KIMI_CUSTOM_HEADERS_ENV: &str = "KIMI_CODE_CUSTOM_HEADERS";
const DEFAULT_OAUTH_HOST: &str = "https://auth.kimi.com";
const OAUTH_CLIENT_ID: &str = "17e5f671-d194-4dfb-9706-5516cb48c098";
const REFRESH_MARGIN_SECS: u64 = 60;

/// Explicit, process-environment-independent configuration for one Kimi profile.
#[derive(Clone)]
pub struct KimiBackendConfig {
    pub api_key: Option<String>,
    pub credentials_path: PathBuf,
    pub device_id_path: Option<PathBuf>,
    pub oauth_host: String,
    pub base_url: String,
    pub custom_headers: reqwest::header::HeaderMap,
}

impl KimiBackendConfig {
    pub fn from_home(home: impl AsRef<Path>) -> Self {
        let home = home.as_ref();
        Self {
            api_key: None,
            credentials_path: home.join("credentials").join("kimi-code.json"),
            device_id_path: Some(home.join("device_id")),
            oauth_host: DEFAULT_OAUTH_HOST.to_string(),
            base_url: DEFAULT_KIMI_CODE_BASE_URL.to_string(),
            custom_headers: reqwest::header::HeaderMap::new(),
        }
    }

    pub fn build(self) -> Result<Option<Arc<dyn LlmBackend>>> {
        let auth: Arc<dyn BearerTokenProvider> =
            if let Some(api_key) = self.api_key.filter(|value| !value.trim().is_empty()) {
                Arc::new(StaticToken { token: api_key })
            } else {
                if !self.credentials_path.is_file() {
                    return Ok(None);
                }
                Arc::new(KimiOAuthTokenProvider {
                    path: self.credentials_path,
                    oauth_host: self.oauth_host,
                    http: reqwest::Client::builder()
                        .connect_timeout(std::time::Duration::from_secs(10))
                        .timeout(std::time::Duration::from_secs(30))
                        .build()
                        .context("building Kimi OAuth HTTP client")?,
                    refresh_lock: tokio::sync::Mutex::new(()),
                })
            };
        let headers = profile_headers(self.device_id_path.as_deref(), self.custom_headers)?;
        Ok(Some(Arc::new(OpenAiClient::with_kimi_support(
            self.base_url.trim_end_matches('/').to_string(),
            auth,
            headers,
        ))))
    }
}

fn profile_headers(
    device_id_path: Option<&Path>,
    mut headers: reqwest::header::HeaderMap,
) -> Result<reqwest::header::HeaderMap> {
    use reqwest::header::{HeaderName, HeaderValue};
    headers.insert(
        reqwest::header::USER_AGENT,
        HeaderValue::from_str(concat!("anvil/", env!("CARGO_PKG_VERSION")))?,
    );
    headers.insert(
        HeaderName::from_static("x-msh-platform"),
        HeaderValue::from_static("kimi_code_cli"),
    );
    headers.insert(
        HeaderName::from_static("x-msh-version"),
        HeaderValue::from_static(env!("CARGO_PKG_VERSION")),
    );
    if let Some(device_id) = device_id_path
        .and_then(|path| std::fs::read_to_string(path).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        headers.insert(
            HeaderName::from_static("x-msh-device-id"),
            HeaderValue::from_str(&device_id)?,
        );
    }
    Ok(headers)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KimiCredentials {
    access_token: String,
    refresh_token: String,
    expires_at: u64,
    #[serde(default)]
    scope: String,
    #[serde(default = "default_token_type")]
    token_type: String,
    #[serde(default)]
    expires_in: u64,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

fn default_token_type() -> String {
    "Bearer".to_string()
}

#[derive(Debug, Deserialize)]
struct RefreshResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: u64,
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    token_type: Option<String>,
}

#[derive(Debug)]
struct StaticToken {
    token: String,
}

impl BearerTokenProvider for StaticToken {
    fn bearer_token(&self) -> BoxFuture<'_, Result<Option<String>>> {
        Box::pin(std::future::ready(Ok(Some(self.token.clone()))))
    }
}

struct KimiOAuthTokenProvider {
    path: PathBuf,
    oauth_host: String,
    http: reqwest::Client,
    refresh_lock: tokio::sync::Mutex<()>,
}

impl std::fmt::Debug for KimiOAuthTokenProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KimiOAuthTokenProvider")
            .field("path", &self.path)
            .field("oauth_host", &self.oauth_host)
            .finish_non_exhaustive()
    }
}

impl KimiOAuthTokenProvider {
    fn read_credentials(&self) -> Result<KimiCredentials> {
        let bytes = std::fs::read(&self.path)
            .with_context(|| format!("reading Kimi credentials {}", self.path.display()))?;
        serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing Kimi credentials {}", self.path.display()))
    }

    async fn token(&self) -> Result<String> {
        let credentials = self.read_credentials()?;
        if credentials.expires_at > unix_time().saturating_add(REFRESH_MARGIN_SECS) {
            return nonempty_access_token(credentials.access_token);
        }

        let _guard = self.refresh_lock.lock().await;
        let credentials = self.read_credentials()?;
        if credentials.expires_at > unix_time().saturating_add(REFRESH_MARGIN_SECS) {
            return nonempty_access_token(credentials.access_token);
        }
        self.refresh(credentials).await
    }

    async fn refresh(&self, mut credentials: KimiCredentials) -> Result<String> {
        if credentials.refresh_token.trim().is_empty() {
            anyhow::bail!("Kimi OAuth token is expired and has no refresh token; run `kimi login`");
        }
        let url = format!("{}/api/oauth/token", self.oauth_host.trim_end_matches('/'));
        let mut last_status = None;
        for attempt in 0..3u32 {
            let response = self
                .http
                .post(&url)
                .form(&[
                    ("client_id", OAUTH_CLIENT_ID),
                    ("grant_type", "refresh_token"),
                    ("refresh_token", credentials.refresh_token.as_str()),
                ])
                .send()
                .await
                .with_context(|| format!("refreshing Kimi OAuth token at {url}"))?;
            let status = response.status();
            if status.is_success() {
                let refreshed: RefreshResponse = response
                    .json()
                    .await
                    .context("parsing Kimi OAuth refresh response")?;
                if refreshed.access_token.trim().is_empty() || refreshed.expires_in == 0 {
                    anyhow::bail!("Kimi OAuth refresh returned incomplete credentials");
                }
                credentials.access_token = refreshed.access_token;
                if let Some(refresh_token) = refreshed.refresh_token {
                    credentials.refresh_token = refresh_token;
                }
                credentials.expires_in = refreshed.expires_in;
                credentials.expires_at = unix_time().saturating_add(refreshed.expires_in);
                if let Some(scope) = refreshed.scope {
                    credentials.scope = scope;
                }
                if let Some(token_type) = refreshed.token_type {
                    credentials.token_type = token_type;
                }
                write_credentials_atomic(&self.path, &credentials)?;
                return Ok(credentials.access_token);
            }

            last_status = Some(status);
            let retryable =
                status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
            if !retryable {
                anyhow::bail!("Kimi OAuth refresh was rejected (HTTP {status}); run `kimi login`");
            }
            if attempt < 2 {
                tokio::time::sleep(std::time::Duration::from_secs(1u64 << attempt)).await;
            }
        }
        anyhow::bail!(
            "Kimi OAuth refresh failed after retries (HTTP {}); run `kimi login`",
            last_status.expect("refresh loop ran")
        )
    }
}

impl BearerTokenProvider for KimiOAuthTokenProvider {
    fn bearer_token(&self) -> BoxFuture<'_, Result<Option<String>>> {
        Box::pin(async move { self.token().await.map(Some) })
    }
}

pub fn load_provider() -> Result<Option<Arc<dyn BearerTokenProvider>>> {
    if let Ok(raw) = std::env::var(KIMI_API_KEY_ENV) {
        let token = raw.trim();
        if !token.is_empty() {
            return Ok(Some(Arc::new(StaticToken {
                token: token.to_string(),
            })));
        }
    }

    let path = credentials_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let oauth_host = std::env::var(KIMI_OAUTH_HOST_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            std::env::var(KIMI_OAUTH_HOST_FALLBACK_ENV)
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        .unwrap_or_else(|| DEFAULT_OAUTH_HOST.to_string());
    Ok(Some(Arc::new(KimiOAuthTokenProvider {
        path,
        oauth_host,
        http: reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("building Kimi OAuth HTTP client")?,
        refresh_lock: tokio::sync::Mutex::new(()),
    })))
}

pub fn credentials_path() -> Result<PathBuf> {
    let home = std::env::var_os(KIMI_CODE_HOME_ENV)
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".kimi-code")))
        .context("could not determine Kimi Code home directory")?;
    Ok(home.join("credentials").join("kimi-code.json"))
}

pub fn base_url() -> String {
    std::env::var(KIMI_CODE_BASE_URL_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_KIMI_CODE_BASE_URL.to_string())
        .trim_end_matches('/')
        .to_string()
}

pub fn default_headers() -> Result<reqwest::header::HeaderMap> {
    use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

    let mut headers = HeaderMap::new();
    headers.insert(
        reqwest::header::USER_AGENT,
        HeaderValue::from_str(concat!("anvil/", env!("CARGO_PKG_VERSION")))?,
    );
    headers.insert(
        HeaderName::from_static("x-msh-platform"),
        HeaderValue::from_static("kimi_code_cli"),
    );
    headers.insert(
        HeaderName::from_static("x-msh-version"),
        HeaderValue::from_static(env!("CARGO_PKG_VERSION")),
    );
    let home = credentials_path()?
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf);
    if let Some(device_id) = home
        .and_then(|home| std::fs::read_to_string(home.join("device_id")).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        headers.insert(
            HeaderName::from_static("x-msh-device-id"),
            HeaderValue::from_str(&device_id)?,
        );
    }
    if let Ok(raw) = std::env::var(KIMI_CUSTOM_HEADERS_ENV) {
        for line in raw.lines() {
            let Some((name, value)) = line.split_once(':') else {
                continue;
            };
            let name = name.trim();
            if name.is_empty() {
                continue;
            }
            headers.insert(
                HeaderName::from_bytes(name.as_bytes())?,
                HeaderValue::from_str(value.trim())?,
            );
        }
    }
    Ok(headers)
}

fn nonempty_access_token(token: String) -> Result<String> {
    if token.trim().is_empty() {
        anyhow::bail!("Kimi credential file contains an empty access token; run `kimi login`");
    }
    Ok(token)
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn write_credentials_atomic(path: &Path, credentials: &KimiCredentials) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(credentials).context("serializing Kimi credentials")?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("kimi-code.json");
    let tmp = path.with_file_name(format!(".{file_name}.anvil-{}.tmp", std::process::id()));
    std::fs::write(&tmp, bytes)
        .with_context(|| format!("writing temporary Kimi credentials {}", tmp.display()))?;
    set_private_permissions(&tmp)?;
    std::fs::rename(&tmp, path).with_context(|| {
        format!(
            "replacing Kimi credentials {} with {}",
            path.display(),
            tmp.display()
        )
    })?;
    Ok(())
}

#[cfg(unix)]
fn set_private_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("setting permissions on {}", path.display()))
}

#[cfg(not(unix))]
fn set_private_permissions(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_client::{
        ChatMessage, IdleTimeouts, LlmBackend, LlmResponse, OpenAiClient, StreamChatRequest,
    };
    use wiremock::matchers::{body_string_contains, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn credentials(access_token: &str, expires_at: u64) -> KimiCredentials {
        KimiCredentials {
            access_token: access_token.to_string(),
            refresh_token: "refresh-old".to_string(),
            expires_at,
            scope: "kimi-code".to_string(),
            token_type: "Bearer".to_string(),
            expires_in: 900,
            extra: BTreeMap::new(),
        }
    }

    fn provider(path: PathBuf, oauth_host: String) -> KimiOAuthTokenProvider {
        KimiOAuthTokenProvider {
            path,
            oauth_host,
            http: reqwest::Client::new(),
            refresh_lock: tokio::sync::Mutex::new(()),
        }
    }

    #[tokio::test]
    async fn fresh_token_is_returned_without_refresh() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kimi-code.json");
        write_credentials_atomic(&path, &credentials("fresh", unix_time() + 600)).unwrap();

        let token = provider(path, "http://127.0.0.1:1".to_string())
            .bearer_token()
            .await
            .unwrap();

        assert_eq!(token.as_deref(), Some("fresh"));
    }

    #[tokio::test]
    async fn expired_token_refreshes_and_persists() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/oauth/token"))
            .and(body_string_contains("grant_type=refresh_token"))
            .and(body_string_contains("refresh_token=refresh-old"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "fresh-access",
                "refresh_token": "fresh-refresh",
                "expires_in": 900,
                "scope": "kimi-code",
                "token_type": "Bearer"
            })))
            .expect(1)
            .mount(&server)
            .await;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kimi-code.json");
        write_credentials_atomic(&path, &credentials("expired", unix_time() - 1)).unwrap();
        let provider = provider(path.clone(), server.uri());

        let (first, second) = tokio::join!(provider.bearer_token(), provider.bearer_token());

        assert_eq!(first.unwrap().as_deref(), Some("fresh-access"));
        assert_eq!(second.unwrap().as_deref(), Some("fresh-access"));
        let persisted: KimiCredentials =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(persisted.access_token, "fresh-access");
        assert_eq!(persisted.refresh_token, "fresh-refresh");
        assert!(persisted.expires_at > unix_time());
    }

    #[tokio::test]
    async fn rejected_refresh_requests_manual_login() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/oauth/token"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kimi-code.json");
        write_credentials_atomic(&path, &credentials("expired", unix_time() - 1)).unwrap();

        let error = provider(path, server.uri())
            .bearer_token()
            .await
            .unwrap_err();

        assert!(format!("{error:#}").contains("kimi login"));
    }

    #[tokio::test]
    #[ignore = "requires live Kimi credentials and network access"]
    async fn kimi_k3_live() {
        let auth = load_provider()
            .expect("load Kimi credentials")
            .expect("set KIMI_API_KEY or run `kimi login`");
        let client = OpenAiClient::with_kimi_support(
            base_url(),
            auth,
            default_headers().expect("build Kimi headers"),
        );

        let models = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            client.list_model_metadata(),
        )
        .await
        .expect("Kimi model discovery timed out")
        .expect("Kimi model discovery failed");
        assert!(models.iter().any(|model| model.id == "k3"), "{models:?}");

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(90),
            client.stream_chat(StreamChatRequest {
                model: "k3".to_string(),
                messages: vec![ChatMessage::user(
                    "Reply with exactly the lowercase word ok.",
                )],
                tools: None,
                reasoning_effort: Some("max".to_string()),
                service_tier: None,
                temperature: Some(0.0),
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: tokio_util::sync::CancellationToken::new(),
                idle_timeouts: IdleTimeouts::uniform(std::time::Duration::from_secs(60)),
            }),
        )
        .await
        .expect("Kimi K3 request timed out")
        .expect("Kimi K3 request failed");
        match response {
            LlmResponse::Text { text, .. } => assert!(!text.trim().is_empty()),
            LlmResponse::ToolCalls { .. } => panic!("K3 unexpectedly returned a tool call"),
        }
    }

    #[test]
    fn explicit_profile_config_derives_only_from_its_home() {
        let config = KimiBackendConfig::from_home("/profiles/kimi-two");
        assert_eq!(
            config.credentials_path,
            Path::new("/profiles/kimi-two/credentials/kimi-code.json")
        );
        assert_eq!(
            config.device_id_path.as_deref(),
            Some(Path::new("/profiles/kimi-two/device_id"))
        );
    }
}
