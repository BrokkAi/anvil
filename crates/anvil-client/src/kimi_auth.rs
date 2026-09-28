//! Kimi Code authentication backed by `KIMI_API_KEY` or the Kimi CLI's
//! short-lived OAuth credential file. OAuth renewal is owned by the installed
//! vendor CLI service, never by this crate. Automatic service launch currently
//! requires Unix; API keys and injected providers remain platform independent.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use futures::future::BoxFuture;

mod service;
pub use service::{KimiService, KimiServiceConfig};

use crate::llm_client::{BearerTokenProvider, LlmBackend, OpenAiClient};

pub const KIMI_API_KEY_ENV: &str = "KIMI_API_KEY";
pub const KIMI_CODE_BASE_URL_ENV: &str = "KIMI_CODE_BASE_URL";
pub const DEFAULT_KIMI_CODE_BASE_URL: &str = "https://api.kimi.com/coding/v1";
const KIMI_CODE_HOME_ENV: &str = "KIMI_CODE_HOME";
const KIMI_OAUTH_HOST_ENV: &str = "KIMI_CODE_OAUTH_HOST";
const KIMI_OAUTH_HOST_FALLBACK_ENV: &str = "KIMI_OAUTH_HOST";
const KIMI_CUSTOM_HEADERS_ENV: &str = "KIMI_CODE_CUSTOM_HEADERS";
const DEFAULT_OAUTH_HOST: &str = "https://auth.kimi.com";

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
        let auth: Arc<dyn BearerTokenProvider> = if let Some(api_key) = self
            .api_key
            .clone()
            .filter(|value| !value.trim().is_empty())
        {
            Arc::new(StaticToken { token: api_key })
        } else {
            let home = self
                .credentials_path
                .parent()
                .and_then(Path::parent)
                .context("Kimi credentials must belong to a profile home")?;
            if self.credentials_path != home.join("credentials").join("kimi-code.json") {
                anyhow::bail!("the Kimi vendor service requires home/credentials/kimi-code.json");
            }
            let mut config = KimiServiceConfig::from_home(home);
            config
                .environment
                .insert(KIMI_OAUTH_HOST_ENV.to_owned(), self.oauth_host.clone());
            config
                .environment
                .insert(KIMI_CODE_BASE_URL_ENV.to_owned(), self.base_url.clone());
            if !self.custom_headers.is_empty() {
                let lines = self
                    .custom_headers
                    .iter()
                    .map(|(name, value)| value.to_str().map(|value| format!("{name}: {value}")))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                config
                    .environment
                    .insert(KIMI_CUSTOM_HEADERS_ENV.to_owned(), lines.join("\n"));
            }
            if !config.credentials_path()?.is_file() {
                return Ok(None);
            }
            KimiService::new(config)?
        };
        self.build_with_token_provider(auth).map(Some)
    }

    /// Use the caller's sole credential owner. This path never reads credentials,
    /// creates another provider, or falls back after an authentication failure.
    pub fn build_with_token_provider(
        self,
        auth: Arc<dyn BearerTokenProvider>,
    ) -> Result<Arc<dyn LlmBackend>> {
        let headers = profile_headers(self.device_id_path.as_deref(), self.custom_headers)?;
        Ok(Arc::new(OpenAiClient::with_kimi_support(
            self.base_url.trim_end_matches('/').to_string(),
            auth,
            headers,
        )))
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

#[derive(Debug)]
struct StaticToken {
    token: String,
}

impl BearerTokenProvider for StaticToken {
    fn bearer_token(&self) -> BoxFuture<'_, Result<Option<String>>> {
        Box::pin(std::future::ready(Ok(Some(self.token.clone()))))
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
    let config = environment_config()?;
    if !config.credentials_path()?.is_file() {
        return Ok(None);
    }
    Ok(Some(KimiService::new(config)?))
}

fn environment_config() -> Result<KimiServiceConfig> {
    let home = std::env::var_os(KIMI_CODE_HOME_ENV)
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".kimi-code")))
        .context("could not determine Kimi Code home directory")?;
    let mut config = KimiServiceConfig::from_home(home);
    for name in [
        KIMI_OAUTH_HOST_ENV,
        KIMI_OAUTH_HOST_FALLBACK_ENV,
        KIMI_CODE_BASE_URL_ENV,
        KIMI_CUSTOM_HEADERS_ENV,
    ] {
        if let Ok(value) = std::env::var(name)
            && !value.trim().is_empty()
        {
            config.environment.insert(name.to_owned(), value);
        }
    }
    Ok(config)
}

pub fn credentials_path() -> Result<PathBuf> {
    environment_config()?.credentials_path()
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
