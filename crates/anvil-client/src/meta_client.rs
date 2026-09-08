//! Meta's native Muse Code client.
//!
//! Muse Code stores an OAuth access token in `~/.config/muse/auth.json` and
//! exchanges that token for a short-lived API key before using the OpenAI
//! Responses-compatible API.  The exchanged key is kept in memory only; the
//! native auth file is never modified.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use futures::future::BoxFuture;
use reqwest::StatusCode;
use serde_json::Value;
use tokio::io::AsyncReadExt;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::llm_client::{
    LlmBackend, LlmResponse, ModelMetadata, ResolvedModelInfo, StreamChatRequest,
};
use crate::responses_api::{build_responses_request, drive_responses_sse_stream};

const META_API_BASE_URL: &str = "https://api.meta.ai/v1";
const META_MINT_BASE_URL: &str = "https://api.meta.ai";
const MUSE_MODEL_PREFIX: &str = "muse-spark-";
const AUTH_READ_LIMIT: usize = 1024 * 1024;
const AUTH_READ_TIMEOUT: Duration = Duration::from_secs(3);
const MINT_REQUEST_TIMEOUT: Duration = Duration::from_secs(20);
const ERROR_BODY_LIMIT: usize = 64 * 1024;

/// Configuration for a native Muse Code profile.
#[derive(Debug, Clone)]
pub struct MetaClientConfig {
    pub auth_path: PathBuf,
    pub base_url: String,
    pub mint_base_url: String,
}

impl MetaClientConfig {
    /// Build a profile-local configuration from a Muse Code home directory.
    pub fn from_home(home: impl AsRef<Path>) -> Self {
        Self {
            auth_path: home.as_ref().join("auth.json"),
            base_url: META_API_BASE_URL.to_string(),
            mint_base_url: META_MINT_BASE_URL.to_string(),
        }
    }
}

#[derive(Clone)]
struct CachedApiKey {
    access_token: String,
    api_key: String,
}

struct MetaAuth {
    path: PathBuf,
    mint_lock: Mutex<Option<CachedApiKey>>,
}

/// LLM backend for the native Meta Muse Code service.
pub struct MetaClient {
    auth: Arc<MetaAuth>,
    http: reqwest::Client,
    base_url: String,
    mint_base_url: String,
}

impl MetaClient {
    /// Load the native Muse Code profile from the user's config directory.
    pub fn load() -> Result<Option<Arc<dyn LlmBackend>>> {
        let Some(config_home) = native_config_home() else {
            return Ok(None);
        };
        let mut config = MetaClientConfig::from_home(config_home.join("muse"));
        if let Ok(url) = std::env::var("TBH_MINT_BASE_URL") {
            let url = url.trim();
            if !url.is_empty() {
                config.mint_base_url = url.to_string();
            }
        }
        Self::load_with_config(config)
    }

    /// Load a Muse Code backend from an explicit profile configuration.
    pub fn load_with_config(config: MetaClientConfig) -> Result<Option<Arc<dyn LlmBackend>>> {
        if !config.auth_path.is_file() {
            return Ok(None);
        }
        if !auth_file_has_access_token(&config.auth_path)? {
            return Ok(None);
        }
        Ok(Some(Arc::new(Self {
            auth: Arc::new(MetaAuth {
                path: config.auth_path,
                mint_lock: Mutex::new(None),
            }),
            http: build_http_client()?,
            base_url: trim_base_url(&config.base_url),
            mint_base_url: trim_base_url(&config.mint_base_url),
        })))
    }

    #[cfg(test)]
    fn new_for_test(auth_path: PathBuf, base_url: String, mint_base_url: String) -> Self {
        Self {
            auth: Arc::new(MetaAuth {
                path: auth_path,
                mint_lock: Mutex::new(None),
            }),
            http: build_http_client().expect("build test Meta HTTP client"),
            base_url: trim_base_url(&base_url),
            mint_base_url: trim_base_url(&mint_base_url),
        }
    }

    async fn access_token(&self) -> Result<String> {
        read_access_token_bounded(&self.auth.path).await
    }

    async fn mint_api_key(&self, access_token: &str) -> Result<String> {
        let response = tokio::time::timeout(
            MINT_REQUEST_TIMEOUT,
            self.http
                .post(format!("{}/muse-code/key", self.mint_base_url))
                .bearer_auth(access_token)
                .timeout(MINT_REQUEST_TIMEOUT)
                .json(&serde_json::json!({"onboard": false}))
                .send(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("timed out minting Meta API key"))?
        .context("minting Meta API key")?;
        let status = response.status();
        if !status.is_success() {
            // Do not include the response body: Meta may echo request or
            // account details, and this path must never expose credentials.
            let body = read_error_body(response).await;
            return Err(crate::http_retry::retryable_llm_error_for_status_and_body(
                format!("Meta API-key minting failed (HTTP {status})"),
                status,
                &body,
            ));
        }
        let payload = response
            .json::<MintResponse>()
            .await
            .context("parsing Meta API-key response")?;
        let api_key = payload
            .api_key
            .filter(|key| !key.trim().is_empty())
            .ok_or_else(|| anyhow::anyhow!("Meta API-key response did not contain api_key"))?;
        Ok(api_key)
    }

    /// Resolve the current API key while holding the mutex only across the
    /// cache check and mint request.  Callers clone the resulting key and do
    /// all inference I/O after releasing this lock.
    async fn api_key(&self) -> Result<String> {
        let mut cached = self.auth.mint_lock.lock().await;
        let access_token = self.access_token().await?;
        if let Some(entry) = cached.as_ref()
            && entry.access_token == access_token
        {
            return Ok(entry.api_key.clone());
        }
        let api_key = self.mint_api_key(&access_token).await?;
        *cached = Some(CachedApiKey {
            access_token,
            api_key: api_key.clone(),
        });
        Ok(api_key)
    }

    async fn api_key_for(&self, cancel: &CancellationToken) -> Result<String> {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => bail!("Meta Responses request was cancelled while acquiring an API key"),
            key = self.api_key() => key,
        }
    }

    /// Evict a key only if it is still the key rejected by the server.  A
    /// concurrent request may already have minted a newer key by the time a
    /// 401 is observed, and that newer key must remain cached.
    async fn invalidate_rejected_key(&self, rejected_key: &str) {
        let mut cached = self.auth.mint_lock.lock().await;
        if cached
            .as_ref()
            .is_some_and(|entry| entry.api_key == rejected_key)
        {
            *cached = None;
        }
    }

    async fn models_response(&self) -> Result<reqwest::Response> {
        let api_key = self.api_key().await?;
        let response = tokio::time::timeout(
            MINT_REQUEST_TIMEOUT,
            self.http
                .get(format!("{}/models", self.base_url))
                .bearer_auth(&api_key)
                .timeout(MINT_REQUEST_TIMEOUT)
                .send(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("timed out discovering Meta models"))?
        .context("discovering Meta models")?;
        if response.status() != StatusCode::UNAUTHORIZED {
            return Ok(response);
        }
        self.invalidate_rejected_key(&api_key).await;
        let api_key = self.api_key().await?;
        tokio::time::timeout(
            MINT_REQUEST_TIMEOUT,
            self.http
                .get(format!("{}/models", self.base_url))
                .bearer_auth(api_key)
                .timeout(MINT_REQUEST_TIMEOUT)
                .send(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("timed out discovering Meta models after API-key renewal"))?
        .context("discovering Meta models after API-key renewal")
    }

    async fn discover_models(&self) -> Result<Vec<ModelMetadata>> {
        let response = self.models_response().await?;
        let status = response.status();
        if !status.is_success() {
            let body = read_error_body(response).await;
            return Err(crate::http_retry::retryable_llm_error_for_status_and_body(
                format!("Meta model discovery failed (HTTP {status})"),
                status,
                &body,
            ));
        }
        let payload = response
            .json::<ModelsResponse>()
            .await
            .context("parsing Meta model catalog")?;
        Ok(payload
            .data
            .into_iter()
            .filter_map(|model| {
                model
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|id| id.starts_with(MUSE_MODEL_PREFIX))
                    .map(str::to_string)
            })
            .map(ModelMetadata::id_only)
            .collect())
    }

    async fn post_responses(
        &self,
        body: &crate::responses_api::ResponsesRequest,
        api_key: &str,
        cancel: &CancellationToken,
        first_progress: Duration,
    ) -> Result<reqwest::Response> {
        crate::http_retry::send_with_retries(
            "posting Meta Responses API request",
            || {
                self.http
                    .post(format!("{}/responses", self.base_url))
                    .bearer_auth(api_key)
                    .header("Accept", "text/event-stream")
                    .json(body)
            },
            Some(cancel),
            Some(first_progress),
        )
        .await
    }

    async fn invoke(&self, request: StreamChatRequest) -> Result<LlmResponse> {
        let StreamChatRequest {
            model,
            messages,
            tools,
            reasoning_effort,
            service_tier: _,
            temperature: _,
            structured_output,
            on_token,
            on_thought,
            cancel,
            idle_timeouts,
        } = request;
        let body = build_responses_request(
            &model,
            &messages,
            tools.as_deref(),
            reasoning_effort.as_deref(),
            structured_output.as_ref(),
            false,
            None,
        );
        let api_key = self.api_key_for(&cancel).await?;
        let response = self
            .post_responses(&body, &api_key, &cancel, idle_timeouts.first_progress)
            .await?;
        let response = if response.status() == StatusCode::UNAUTHORIZED {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => bail!("Meta Responses request was cancelled"),
                () = self.invalidate_rejected_key(&api_key) => {}
            }
            let renewed_key = self.api_key_for(&cancel).await?;
            self.post_responses(&body, &renewed_key, &cancel, idle_timeouts.first_progress)
                .await?
        } else {
            response
        };
        let status = response.status();
        if !status.is_success() {
            let body = read_error_body(response).await;
            return Err(crate::http_retry::retryable_llm_error_for_status_and_body(
                format!("Meta Responses API failed (HTTP {status})"),
                status,
                &body,
            ));
        }
        let stream = response.bytes_stream().map(|chunk| {
            chunk
                .map(|bytes| bytes.to_vec())
                .map_err(anyhow::Error::from)
        });
        Ok(
            drive_responses_sse_stream(stream, on_token, on_thought, cancel, idle_timeouts)
                .await?
                .response,
        )
    }
}

impl LlmBackend for MetaClient {
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
        Box::pin(async move {
            Ok(self
                .discover_models()
                .await?
                .into_iter()
                .map(|model| model.id)
                .collect())
        })
    }

    fn list_model_metadata(&self) -> BoxFuture<'_, Result<Vec<ModelMetadata>>> {
        Box::pin(self.discover_models())
    }

    fn resolve_model_info(&self, configured_model: &str) -> ResolvedModelInfo {
        ResolvedModelInfo {
            configured_model: configured_model.to_string(),
            resolved_provider: Some("meta".to_string()),
            resolved_model: configured_model.to_string(),
        }
    }

    fn stream_chat(&self, request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
        Box::pin(self.invoke(request))
    }
}

#[derive(serde::Deserialize)]
struct MintResponse {
    #[serde(default)]
    api_key: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<Value>,
}

fn build_http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(MINT_REQUEST_TIMEOUT)
        .build()
        .context("building Meta HTTP client")
}

async fn read_error_body(response: reqwest::Response) -> String {
    let read = async move {
        let mut stream = response.bytes_stream();
        let mut bytes = Vec::new();
        while let Some(Ok(chunk)) = stream.next().await {
            let remaining = ERROR_BODY_LIMIT - bytes.len();
            bytes.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
            if bytes.len() == ERROR_BODY_LIMIT {
                break;
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    };
    tokio::time::timeout(AUTH_READ_TIMEOUT, read)
        .await
        .unwrap_or_default()
}

fn trim_base_url(url: &str) -> String {
    url.trim_end_matches('/').to_string()
}

fn native_config_home() -> Option<PathBuf> {
    std::env::var_os("XDG_CONFIG_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".config")))
}

fn access_token_from_bytes(bytes: &[u8]) -> Result<Option<String>> {
    let root: Value = serde_json::from_slice(bytes).context("parsing Meta auth file")?;
    Ok(root
        .get("providers")
        .and_then(Value::as_object)
        .and_then(|providers| providers.get("meta"))
        .and_then(Value::as_object)
        .and_then(|meta| meta.get("access_token"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(str::to_string))
}

fn auth_file_has_access_token(path: &Path) -> Result<bool> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("reading Meta auth file {}", path.display()))?;
    let mut bytes = Vec::new();
    use std::io::Read;
    file.take((AUTH_READ_LIMIT + 1) as u64)
        .read_to_end(&mut bytes)
        .context("reading Meta auth file")?;
    if bytes.len() > AUTH_READ_LIMIT {
        bail!("Meta auth file is too large");
    }
    Ok(access_token_from_bytes(&bytes)?.is_some())
}

async fn read_access_token_bounded(path: &Path) -> Result<String> {
    let path = path.to_path_buf();
    let read = async move {
        let file = tokio::fs::File::open(&path)
            .await
            .with_context(|| format!("reading Meta auth file {}", path.display()))?;
        let mut bytes = Vec::new();
        file.take((AUTH_READ_LIMIT + 1) as u64)
            .read_to_end(&mut bytes)
            .await
            .context("reading Meta auth file")?;
        if bytes.len() > AUTH_READ_LIMIT {
            bail!("Meta auth file is too large");
        }
        access_token_from_bytes(&bytes)?
            .ok_or_else(|| anyhow::anyhow!("Meta auth file has no providers.meta.access_token"))
    };
    tokio::time::timeout(AUTH_READ_TIMEOUT, read)
        .await
        .map_err(|_| anyhow::anyhow!("timed out reading Meta auth file"))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_client::{ChatMessage, IdleTimeouts, TokenSink};
    use serde_json::json;
    use std::time::Duration;
    use tempfile::tempdir;
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn write_auth(path: &Path, token: &str) {
        std::fs::write(
            path,
            serde_json::to_vec(&json!({
                "providers": {"meta": {"access_token": token}},
                "other": {"access_token": "must-not-be-used"}
            }))
            .unwrap(),
        )
        .unwrap();
    }

    fn request(model: &str, cancel: CancellationToken) -> StreamChatRequest {
        fn noop(_: &str) {}

        StreamChatRequest {
            model: model.to_string(),
            messages: vec![ChatMessage::user("hello")],
            tools: None,
            reasoning_effort: Some("low".to_string()),
            service_tier: None,
            temperature: None,
            structured_output: None,
            on_token: Box::new(noop) as TokenSink,
            on_thought: Box::new(noop) as TokenSink,
            cancel,
            idle_timeouts: IdleTimeouts::uniform(Duration::from_secs(5)),
        }
    }

    #[test]
    fn config_is_profile_local_and_uses_native_defaults() {
        let config = MetaClientConfig::from_home("/profiles/muse-one");
        assert_eq!(config.auth_path, Path::new("/profiles/muse-one/auth.json"));
        assert_eq!(config.base_url, META_API_BASE_URL);
        assert_eq!(config.mint_base_url, META_MINT_BASE_URL);
    }

    #[test]
    fn auth_parser_selects_only_native_meta_provider() {
        let parsed = access_token_from_bytes(
            br#"{"providers":{"other":{"access_token":"wrong"},"meta":{"access_token":" right "}}}"#,
        )
        .unwrap();
        assert_eq!(parsed.as_deref(), Some("right"));
        assert_eq!(
            access_token_from_bytes(br#"{"access_token":"wrong"}"#).unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn mint_and_models_use_exchange_key_and_filter_chat_models() {
        let server = MockServer::start().await;
        let dir = tempdir().unwrap();
        let auth_path = dir.path().join("auth.json");
        write_auth(&auth_path, "oauth-token");
        Mock::given(method("POST"))
            .and(path("/muse-code/key"))
            .and(header("authorization", "Bearer oauth-token"))
            .and(body_json(json!({"onboard": false})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"api_key": "api-key"})))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .and(header("authorization", "Bearer api-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [
                    {"id": "muse-spark-1.3"},
                    {"id": "muse-spark-1.3-contributor"},
                    {"id": "muse-image-1"},
                    {"id": "voice-model"}
                ]
            })))
            .mount(&server)
            .await;
        let client =
            MetaClient::new_for_test(auth_path, format!("{}/v1", server.uri()), server.uri());
        assert_eq!(
            client.list_models().await.unwrap(),
            ["muse-spark-1.3", "muse-spark-1.3-contributor"]
        );
        assert_eq!(
            client.list_models().await.unwrap(),
            ["muse-spark-1.3", "muse-spark-1.3-contributor"]
        );
    }

    #[tokio::test]
    async fn responses_body_contains_native_schema_and_stream_is_parsed() {
        let server = MockServer::start().await;
        let dir = tempdir().unwrap();
        let auth_path = dir.path().join("auth.json");
        write_auth(&auth_path, "oauth-token");
        Mock::given(method("POST"))
            .and(path("/muse-code/key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"api_key": "api-key"})))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer api-key"))
            .and(wiremock::matchers::body_partial_json(json!({
                "model": "muse-spark-1.3",
                "stream": true,
                "store": false,
                "reasoning": {"effort": "low"},
                "text": {"format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
                    "strict": true
                }}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n\
                 data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n\
                 data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\"}}\n\n",
                "text/event-stream",
            ))
            .mount(&server)
            .await;
        let client =
            MetaClient::new_for_test(auth_path, format!("{}/v1", server.uri()), server.uri());
        let mut input = request("muse-spark-1.3", CancellationToken::new());
        input.structured_output = Some(crate::structured_output::StructuredOutputRequest {
            schema_name: "answer".into(),
            schema: json!({"type": "object", "properties": {"ok": {"type": "boolean"}}}),
            allow_coercion: false,
            prefer_json_object: false,
        });
        let response = client.stream_chat(input).await.unwrap();
        match response {
            LlmResponse::Text { text, .. } => assert_eq!(text, "hello"),
            LlmResponse::ToolCalls { .. } => panic!("expected text response"),
        }
    }

    #[tokio::test]
    async fn unauthorized_response_mints_once_more_and_retries() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let server = MockServer::start().await;
        let dir = tempdir().unwrap();
        let auth_path = dir.path().join("auth.json");
        write_auth(&auth_path, "oauth-token");
        let original = std::fs::read(&auth_path).unwrap();
        let mints = AtomicUsize::new(0);
        Mock::given(method("POST"))
            .and(path("/muse-code/key"))
            .respond_with(move |_: &wiremock::Request| {
                let n = mints.fetch_add(1, Ordering::SeqCst);
                ResponseTemplate::new(200).set_body_json(json!({"api_key": format!("key-{n}")}))
            })
            .expect(2)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer key-0"))
            .respond_with(ResponseTemplate::new(401))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer key-1"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"renewed\"}\n\ndata: {\"type\":\"response.completed\",\"response\":{}}\n\n",
                "text/event-stream",
            ))
            .expect(1).mount(&server).await;
        let client = MetaClient::new_for_test(
            auth_path.clone(),
            format!("{}/v1", server.uri()),
            server.uri(),
        );
        let response = client
            .stream_chat(request("muse-spark-1.3", CancellationToken::new()))
            .await
            .unwrap();
        assert!(matches!(response, LlmResponse::Text {text, ..} if text == "renewed"));
        assert_eq!(std::fs::read(auth_path).unwrap(), original);
    }

    #[tokio::test]
    async fn cancellation_interrupts_stalled_key_exchange() {
        let server = MockServer::start().await;
        let dir = tempdir().unwrap();
        let auth_path = dir.path().join("auth.json");
        write_auth(&auth_path, "oauth-token");
        Mock::given(method("POST"))
            .and(path("/muse-code/key"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({"api_key": "key"}))
                    .set_delay(Duration::from_secs(10)),
            )
            .mount(&server)
            .await;
        let client =
            MetaClient::new_for_test(auth_path, format!("{}/v1", server.uri()), server.uri());
        let cancel = CancellationToken::new();
        let operation = client.stream_chat(request("muse-spark-1.3", cancel.clone()));
        let cancellation = async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            cancel.cancel();
        };
        let (result, ()) = tokio::time::timeout(Duration::from_secs(2), async {
            tokio::join!(operation, cancellation)
        })
        .await
        .expect("cancellation must interrupt minting");
        assert!(result.unwrap_err().to_string().contains("cancelled"));
    }

    #[tokio::test]
    #[ignore = "requires native Muse credentials and explicit live opt-in"]
    async fn live_native_muse_structured_inference() {
        use crate::discovery::ModelSource;
        use crate::infer::{InferMessage, InferOptions, StructuredInferRequest, infer_structured};
        use crate::multi_backend::{BackendRegistration, MultiBackend};
        let auth_path = native_config_home().unwrap().join("muse/auth.json");
        let original = std::fs::read(&auth_path).unwrap();
        let backend = MetaClient::load()
            .unwrap()
            .expect("native Muse login required");
        assert!(
            backend
                .list_models()
                .await
                .unwrap()
                .iter()
                .any(|id| id == "muse-spark-1.3")
        );
        let backend = MultiBackend::new(vec![BackendRegistration::new(
            ModelSource::META,
            "Meta",
            Some(backend),
        )]);
        let response = infer_structured(
            &backend,
            "meta::muse-spark-1.3",
            StructuredInferRequest {
                messages: vec![InferMessage::user("Return an object with ok set to true.")],
                schema_name: "live_probe".into(),
                schema: json!({"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}),
            },
            InferOptions {reasoning_effort: Some("low".into()), ..Default::default()},
            CancellationToken::new(),
        ).await.unwrap();
        assert_eq!(response.output, json!({"ok":true}));
        assert_eq!(std::fs::read(auth_path).unwrap(), original);
        eprintln!(
            "{} returned valid structured JSON; native credentials unchanged",
            response.model
        );
    }
}
