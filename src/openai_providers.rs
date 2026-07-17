//! Config-file backed OpenAI-compatible provider profiles.
//!
//! This is deliberately stricter than the built-in provider setup flows:
//! `providers.json` is process-start configuration, not an interactive
//! credential store. Malformed config fails startup so bad installs are
//! obvious; discovery failures for valid profiles remain nonfatal.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use futures::future::{BoxFuture, join_all};
use serde::Deserialize;

use crate::llm_client::{
    LlmBackend, LlmResponse, ModelDiscoveryNotice, ModelMetadata, ResolvedModelInfo,
    StreamChatRequest,
};

pub const PROVIDERS_FILE_NAME: &str = "providers.json";
pub const OPENAI_PROVIDER_SOURCE: &str = "openai";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiProviderProfile {
    pub name: String,
    pub base_url: String,
    pub api_key_env: Option<String>,
    pub api_key: Option<String>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct OpenAiProviderConfig {
    pub profiles: Vec<OpenAiProviderProfile>,
}

impl OpenAiProviderConfig {
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProvidersFile {
    #[serde(default)]
    openai: BTreeMap<String, RawOpenAiProviderProfile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOpenAiProviderProfile {
    base_url: String,
    #[serde(default)]
    api_key_env: Option<String>,
}

pub fn path() -> Result<PathBuf> {
    Ok(crate::setup_state::config_home()?.join(PROVIDERS_FILE_NAME))
}

pub fn read() -> Result<OpenAiProviderConfig> {
    let path = path()?;
    if !path.exists() {
        return Ok(OpenAiProviderConfig::default());
    }
    let bytes = std::fs::read(&path)
        .with_context(|| format!("reading OpenAI provider config {}", path.display()))?;
    parse(&bytes).with_context(|| format!("parsing OpenAI provider config {}", path.display()))
}

fn parse(bytes: &[u8]) -> Result<OpenAiProviderConfig> {
    let parsed: ProvidersFile = serde_json::from_slice(bytes)?;
    let mut profiles = Vec::with_capacity(parsed.openai.len());
    for (name, raw) in parsed.openai {
        validate_profile_name(&name)?;
        let base_url = normalize_base_url(&raw.base_url)
            .with_context(|| format!("invalid base_url for openai provider {name:?}"))?;
        let (api_key_env, api_key) = match raw.api_key_env {
            Some(env) => {
                validate_env_name(&env)
                    .with_context(|| format!("invalid api_key_env for openai provider {name:?}"))?;
                let value = std::env::var(&env).with_context(|| {
                    format!("openai provider {name:?} requires missing environment variable {env}")
                })?;
                let trimmed = value.trim();
                if trimmed.is_empty() {
                    anyhow::bail!(
                        "openai provider {name:?} requires non-empty environment variable {env}"
                    );
                }
                (Some(env), Some(trimmed.to_string()))
            }
            None => (None, None),
        };
        profiles.push(OpenAiProviderProfile {
            name,
            base_url,
            api_key_env,
            api_key,
        });
    }
    Ok(OpenAiProviderConfig { profiles })
}

fn validate_profile_name(name: &str) -> Result<()> {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes.len() > 32 {
        anyhow::bail!("profile name must be 1-32 characters");
    }
    if !bytes[0].is_ascii_lowercase() && !bytes[0].is_ascii_digit() {
        anyhow::bail!("profile name must start with [a-z0-9]");
    }
    if !bytes
        .iter()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'_' || *b == b'-')
    {
        anyhow::bail!("profile name must match [a-z0-9][a-z0-9_-]{{0,31}}");
    }
    Ok(())
}

fn validate_env_name(name: &str) -> Result<()> {
    let bytes = name.as_bytes();
    if bytes.is_empty() {
        anyhow::bail!("environment variable name is empty");
    }
    if !bytes[0].is_ascii_alphabetic() && bytes[0] != b'_' {
        anyhow::bail!("environment variable name must start with [A-Za-z_]");
    }
    if !bytes
        .iter()
        .all(|b| b.is_ascii_alphanumeric() || *b == b'_')
    {
        anyhow::bail!("environment variable name must match [A-Za-z_][A-Za-z0-9_]*");
    }
    Ok(())
}

fn normalize_base_url(raw: &str) -> Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        anyhow::bail!("base_url is empty");
    }
    let parsed = reqwest::Url::parse(trimmed)?;
    match parsed.scheme() {
        "http" | "https" => {}
        scheme => anyhow::bail!("unsupported scheme {scheme:?}; expected http or https"),
    }
    if parsed.host_str().is_none() {
        anyhow::bail!("base_url must include a host");
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        anyhow::bail!("base_url must not contain embedded credentials");
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        anyhow::bail!("base_url must not contain a query string or fragment");
    }

    let base = trimmed.trim_end_matches('/');
    if base.ends_with("/v1") {
        Ok(base.to_string())
    } else {
        Ok(format!("{base}/v1"))
    }
}

pub fn build_backend(config: OpenAiProviderConfig) -> Option<Arc<dyn LlmBackend>> {
    if config.is_empty() {
        return None;
    }
    let profiles = config
        .profiles
        .into_iter()
        .map(|profile| {
            let client = crate::llm_client::OpenAiClient::new(profile.base_url, profile.api_key);
            (profile.name, Arc::new(client) as Arc<dyn LlmBackend>)
        })
        .collect();
    Some(Arc::new(OpenAiProvidersBackend { profiles }))
}

struct OpenAiProvidersBackend {
    profiles: BTreeMap<String, Arc<dyn LlmBackend>>,
}

impl OpenAiProvidersBackend {
    fn split_profile_model(&self, model: &str) -> Result<(String, String)> {
        if let Some((profile, upstream)) = model.split_once('/')
            && self.profiles.contains_key(profile)
            && !upstream.is_empty()
        {
            return Ok((profile.to_string(), upstream.to_string()));
        }

        if self.profiles.len() == 1 {
            let profile = self.profiles.keys().next().expect("len checked").as_str();
            if !model.is_empty() {
                return Ok((profile.to_string(), model.to_string()));
            }
        }

        anyhow::bail!(
            "openai model {model:?} must be profile-qualified as openai::<profile>/<model>"
        );
    }

    fn profile(&self, profile: &str) -> Result<Arc<dyn LlmBackend>> {
        self.profiles
            .get(profile)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("openai provider profile {profile:?} is not configured"))
    }
}

impl LlmBackend for OpenAiProvidersBackend {
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
        Box::pin(async move {
            Ok(self
                .list_model_metadata()
                .await?
                .into_iter()
                .map(|m| m.id)
                .collect())
        })
    }

    fn list_model_metadata(&self) -> BoxFuture<'_, Result<Vec<ModelMetadata>>> {
        Box::pin(async move {
            let futures = self.profiles.iter().map(|(profile, backend)| {
                let profile = profile.clone();
                let backend = backend.clone();
                async move {
                    match backend.list_model_metadata().await {
                        Ok(models) => models
                            .into_iter()
                            .map(|model| ModelMetadata {
                                id: format!("{profile}/{}", model.id),
                                default_reasoning_level: None,
                                supported_reasoning_levels: Vec::new(),
                                service_tiers: model.service_tiers,
                                supports_images: model.supports_images,
                                context_length: model.context_length,
                                pricing: model.pricing,
                            })
                            .collect::<Vec<_>>(),
                        Err(e) => {
                            tracing::info!(
                                "openai provider profile {profile} model discovery skipped: {e:#}"
                            );
                            Vec::new()
                        }
                    }
                }
            });
            Ok(join_all(futures).await.into_iter().flatten().collect())
        })
    }

    fn resolve_model_info(&self, configured_model: &str) -> ResolvedModelInfo {
        match self.split_profile_model(configured_model) {
            Ok((profile, upstream)) => ResolvedModelInfo {
                configured_model: configured_model.to_string(),
                resolved_provider: Some(format!("{OPENAI_PROVIDER_SOURCE}/{profile}")),
                resolved_model: upstream,
            },
            Err(_) => ResolvedModelInfo {
                configured_model: configured_model.to_string(),
                resolved_provider: Some(OPENAI_PROVIDER_SOURCE.to_string()),
                resolved_model: configured_model.to_string(),
            },
        }
    }

    fn take_model_discovery_notices(&self) -> Vec<ModelDiscoveryNotice> {
        self.profiles
            .values()
            .flat_map(|backend| backend.take_model_discovery_notices())
            .collect()
    }

    fn stream_chat(&self, request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
        let resolution = self
            .split_profile_model(&request.model)
            .and_then(|(profile, upstream)| Ok((self.profile(&profile)?, upstream)));
        Box::pin(async move {
            let (backend, upstream) = resolution?;
            let mut request = request;
            request.model = upstream;
            backend.stream_chat(request).await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup_state::TestConfigHomeScope;
    use futures::FutureExt;
    use std::sync::Mutex;

    fn chat_request(model: &str) -> StreamChatRequest {
        StreamChatRequest {
            model: model.to_string(),
            messages: vec![],
            tools: None,
            reasoning_effort: Some("high".to_string()),
            service_tier: None,
            temperature: None,
            structured_output: None,
            on_token: Box::new(|_| {}),
            on_thought: Box::new(|_| {}),
            cancel: tokio_util::sync::CancellationToken::new(),
            idle_timeouts: crate::llm_client::IdleTimeouts::uniform(
                std::time::Duration::from_secs(60),
            ),
        }
    }

    struct RecordingBackend {
        last_model: Arc<Mutex<Option<String>>>,
        last_reasoning_effort: Arc<Mutex<Option<String>>>,
    }

    impl LlmBackend for RecordingBackend {
        fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
            async move { Ok(vec!["model-a".to_string()]) }.boxed()
        }

        fn stream_chat(&self, request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
            *self.last_model.lock().unwrap() = Some(request.model);
            *self.last_reasoning_effort.lock().unwrap() = request.reasoning_effort;
            async move {
                Ok(LlmResponse::Text {
                    text: "ok".to_string(),
                    reasoning_content: None,
                    usage: crate::llm_client::TokenUsage::default(),
                })
            }
            .boxed()
        }
    }

    struct RecordingHandles {
        backend: Arc<dyn LlmBackend>,
        last_model: Arc<Mutex<Option<String>>>,
        last_reasoning_effort: Arc<Mutex<Option<String>>>,
    }

    fn recording() -> RecordingHandles {
        let last_model = Arc::new(Mutex::new(None));
        let last_reasoning_effort = Arc::new(Mutex::new(None));
        RecordingHandles {
            backend: Arc::new(RecordingBackend {
                last_model: last_model.clone(),
                last_reasoning_effort: last_reasoning_effort.clone(),
            }),
            last_model,
            last_reasoning_effort,
        }
    }

    #[test]
    fn missing_file_is_empty_config() {
        let dir = tempfile::tempdir().expect("config dir");
        let _scope = TestConfigHomeScope::set(dir.path().to_path_buf());

        let config = read().expect("missing providers.json is ok");

        assert!(config.is_empty());
    }

    #[test]
    fn parses_profiles_and_appends_v1() {
        let dir = tempfile::tempdir().expect("config dir");
        let _scope = TestConfigHomeScope::set(dir.path().to_path_buf());
        std::fs::write(
            dir.path().join(PROVIDERS_FILE_NAME),
            r#"{
              "openai": {
                "deca": {
                  "base_url": "https://api.genlabs.dev/deca",
                  "api_key_env": "DECA_API_KEY"
                },
                "local-proxy": {
                  "base_url": "http://127.0.0.1:8000/v1/"
                }
              }
            }"#,
        )
        .expect("write config");
        let _env = EnvScope::set("DECA_API_KEY", "sk-test");

        let config = read().expect("valid config");

        assert_eq!(config.profiles.len(), 2);
        assert_eq!(config.profiles[0].name, "deca");
        assert_eq!(
            config.profiles[0].base_url,
            "https://api.genlabs.dev/deca/v1"
        );
        assert_eq!(config.profiles[0].api_key.as_deref(), Some("sk-test"));
        assert_eq!(config.profiles[1].name, "local-proxy");
        assert_eq!(config.profiles[1].base_url, "http://127.0.0.1:8000/v1");
        assert!(config.profiles[1].api_key.is_none());
    }

    #[test]
    fn rejects_unknown_fields() {
        let err = parse(
            br#"{
              "openai": {
                "deca": {
                  "base_url": "https://api.genlabs.dev/deca",
                  "api_key": "sk-inline-is-not-allowed"
                }
              }
            }"#,
        )
        .expect_err("unknown fields are fatal");

        assert!(format!("{err:#}").contains("unknown field"));
    }

    #[test]
    fn rejects_invalid_profile_and_url() {
        let err = parse(
            br#"{
              "openai": {
                "Bad": {"base_url": "https://api.example.com"}
              }
            }"#,
        )
        .expect_err("invalid profile");
        assert!(format!("{err:#}").contains("profile name"));

        let err = parse(
            br#"{
              "openai": {
                "ok": {"base_url": "https://user:pass@example.com/v1"}
              }
            }"#,
        )
        .expect_err("invalid url");
        assert!(format!("{err:#}").contains("embedded credentials"));
    }

    #[test]
    fn missing_configured_env_var_is_fatal() {
        let _env = EnvScope::unset("MISSING_OPENAI_PROVIDER_KEY");

        let err = parse(
            br#"{
              "openai": {
                "deca": {
                  "base_url": "https://api.genlabs.dev/deca",
                  "api_key_env": "MISSING_OPENAI_PROVIDER_KEY"
                }
              }
            }"#,
        )
        .expect_err("missing env var");

        assert!(format!("{err:#}").contains("MISSING_OPENAI_PROVIDER_KEY"));
    }

    #[tokio::test]
    async fn profile_wire_model_strips_profile_before_delegating() {
        let handles = recording();
        let profiles = BTreeMap::from([("deca".to_string(), handles.backend)]);
        let aggregate = OpenAiProvidersBackend { profiles };

        let _ = aggregate
            .stream_chat(chat_request("deca/foo/bar"))
            .await
            .expect("profile route");

        assert_eq!(
            handles.last_model.lock().unwrap().as_deref(),
            Some("foo/bar")
        );
        assert_eq!(
            handles.last_reasoning_effort.lock().unwrap().as_deref(),
            Some("high")
        );
    }

    #[tokio::test]
    async fn bare_model_routes_when_exactly_one_profile_exists() {
        let handles = recording();
        let profiles = BTreeMap::from([("deca".to_string(), handles.backend)]);
        let aggregate = OpenAiProvidersBackend { profiles };

        let _ = aggregate
            .stream_chat(chat_request("vendor/model-a"))
            .await
            .expect("single-profile bare route");

        assert_eq!(
            handles.last_model.lock().unwrap().as_deref(),
            Some("vendor/model-a")
        );
    }

    #[tokio::test]
    async fn bare_model_errors_when_multiple_profiles_exist() {
        let deca = recording();
        let other = recording();
        let profiles = BTreeMap::from([
            ("deca".to_string(), deca.backend),
            ("other".to_string(), other.backend),
        ]);
        let aggregate = OpenAiProvidersBackend { profiles };

        let err = aggregate
            .stream_chat(chat_request("model-a"))
            .await
            .expect_err("bare id is ambiguous");

        assert!(format!("{err:#}").contains("profile-qualified"));
    }

    struct EnvScope {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvScope {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            unsafe {
                std::env::set_var(key, value);
            }
            Self { key, previous }
        }

        fn unset(key: &'static str) -> Self {
            let previous = std::env::var(key).ok();
            unsafe {
                std::env::remove_var(key);
            }
            Self { key, previous }
        }
    }

    impl Drop for EnvScope {
        fn drop(&mut self) {
            unsafe {
                match &self.previous {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }
}
