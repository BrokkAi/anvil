//! Routing `LlmBackend` that fans `list_models` out to all configured
//! sources and dispatches `stream_chat` to the right one based on the
//! `<source>::<id>` wire prefix produced by `discovery.rs`.
//!
//! Why a separate type? `OpenAiClient` and `CodexClient` already implement
//! `LlmBackend` for one transport each. Wrapping the configured sources
//! (Bedrock, Codex, hosted DeepSeek, Kimi, Grok, generic OpenAI profiles,
//! OpenRouter, Ollama) in a single routing backend lets `agent.rs` stay
//! oblivious to which model it's talking to -- it just hands the wire id
//! back to the backend the same way it always has, and the backend strips
//! the prefix and routes.
//!
//! Bare ids (no `<source>::` prefix) fall back to a preferred
//! source so manually-typed model ids still route somewhere reasonable.
//! Without that fallback, a user typing `llama3:latest` directly into the
//! setup's advanced model picker would get a "no backend for model" error even
//! though the picker also offers `ollama::llama3:latest`.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use anyhow::Result;
use futures::future::{BoxFuture, join_all};
use tokio::sync::mpsc::UnboundedSender;

use crate::discovery::{
    ModelSource, OLLAMA_DEFAULT_URL, discover_ollama_model_metadata, discovery_http_client,
    split_wire_id,
};
#[cfg(test)]
use crate::llm_client::IdleTimeouts;
use crate::llm_client::{
    LlmBackend, LlmResponse, ModelDiscoveryNotice, ModelMetadata, ResolvedModelInfo,
    StreamChatRequest,
};

const PROVIDER_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(15);

/// LLM backend that routes by `<source>::<id>` prefix. Any inner backend
/// may be absent (e.g. no `auth.json`, no `DEEPSEEK_API_KEY`, no
/// `OPENROUTER_API_KEY`, or no
/// Ollama on the default port); calls for a source whose backend isn't
/// configured return a clear error rather than silently falling through.
///
/// Every registered source uses the same small `RwLock` slot. Runtime setup
/// flows replace selected slots, while startup-only providers simply leave
/// their slots unchanged. The lock is held only for a synchronous
/// `Option<Arc<...>>` clone, never across an `.await`.
///
/// ds4 is behind an `RwLock` like Codex/OpenRouter, but for a different
/// reason: ds4-server has no fixed port, so each discovery refresh
/// re-resolves the running server's port (see `discovery::ds4_base_url`)
/// and reinstalls a backend pointed at it. That keeps `ds4::*` chat
/// routing aimed at the same port discovery just found, and lets ds4 come
/// online when it's started after Anvil.
pub struct BackendRegistration {
    source: String,
    label: String,
    backend: Option<Arc<dyn LlmBackend>>,
}

impl BackendRegistration {
    pub fn new(
        source: impl Into<String>,
        label: impl Into<String>,
        backend: Option<Arc<dyn LlmBackend>>,
    ) -> Self {
        Self {
            source: source.into(),
            label: label.into(),
            backend,
        }
    }
}

struct BackendSlot {
    source: String,
    label: String,
    backend: RwLock<Option<Arc<dyn LlmBackend>>>,
}

pub struct MultiBackend {
    /// Ordered by discovery/default/fallback priority.
    backends: Vec<BackendSlot>,
}

impl MultiBackend {
    pub fn validate_explicit_model_route(&self, wire_model: &str) -> Result<()> {
        if split_wire_id(wire_model).is_none() {
            anyhow::bail!(
                "utility model {wire_model:?} must be provider-qualified as <source>::<id>"
            );
        }
        self.resolve(wire_model).map(|_| ())
    }

    pub fn new(registrations: Vec<BackendRegistration>) -> Self {
        let mut seen = std::collections::HashSet::new();
        let backends = registrations
            .into_iter()
            .map(|registration| {
                assert!(
                    seen.insert(registration.source.clone()),
                    "duplicate LLM backend source {}",
                    registration.source
                );
                BackendSlot {
                    source: registration.source,
                    label: registration.label,
                    backend: RwLock::new(registration.backend),
                }
            })
            .collect();
        Self { backends }
    }

    fn slot(&self, source: &str) -> Option<&BackendSlot> {
        self.backends.iter().find(|slot| slot.source == source)
    }

    fn install(&self, source: &str, backend: Arc<dyn LlmBackend>) {
        let slot = self
            .slot(source)
            .unwrap_or_else(|| panic!("LLM backend source {source} is not registered"));
        *slot.backend.write().unwrap() = Some(backend);
    }

    fn uninstall(&self, source: &str) {
        if let Some(slot) = self.slot(source) {
            *slot.backend.write().unwrap() = None;
        }
    }

    /// Install (or replace) the Bedrock backend at runtime. Called from
    /// `/setup bedrock key <token>` so a session that started without
    /// Bedrock credentials picks up the new key on the next discovery
    /// refresh.
    pub fn install_bedrock(&self, backend: Arc<dyn LlmBackend>) {
        self.install(ModelSource::BEDROCK, backend);
    }

    /// Drop the currently-installed Bedrock backend, if any. Called
    /// from `/setup bedrock disconnect` after the on-disk credentials
    /// are wiped.
    pub fn uninstall_bedrock(&self) {
        self.uninstall(ModelSource::BEDROCK);
    }

    /// Install (or replace) the Codex backend at runtime. Called from
    /// `/setup codex` so the next discovery refresh and any subsequent
    /// `codex::*` route picks it up without a server restart.
    ///
    /// Replacing an existing backend is safe: any in-flight request
    /// holding a clone of the old `Arc<CodexClient>` finishes against
    /// that instance and then drops it. Note that the new backend
    /// starts with an empty `reqwest` cookie jar and `refresh_lock`, so
    /// the first request after replacement may have to re-acquire any
    /// Cloudflare cookies (`__cf_bm`, `cf_clearance`) the previous
    /// instance had already accumulated; this is a one-request cost.
    pub fn install_codex(&self, backend: Arc<dyn LlmBackend>) {
        // unwrap: the only way the lock gets poisoned is a panic while
        // holding it, and the only sites that hold it are tiny clones of
        // an Option<Arc> -- not panickable in practice.
        self.install(ModelSource::CODEX, backend);
    }

    /// Drop the currently-installed Codex backend, if any. Called from
    /// `/setup codex disconnect` after the on-disk credentials are
    /// wiped so a subsequent `codex::*` request fails with the same
    /// "backend not configured" error a fresh-no-auth.json startup
    /// would give, instead of firing requests with credentials that
    /// will now 401. In-flight requests holding an `Arc` to the old
    /// backend complete against that captured instance.
    pub fn uninstall_codex(&self) {
        self.uninstall(ModelSource::CODEX);
    }

    /// Install (or replace) the hosted DeepSeek backend at runtime.
    /// Called from `/setup deepseek key <key>` so a session that started
    /// without `DEEPSEEK_API_KEY` or a stored key picks up the new key on
    /// the next discovery refresh.
    pub fn install_deepseek(&self, backend: Arc<dyn LlmBackend>) {
        self.install(ModelSource::DEEPSEEK, backend);
    }

    /// Drop the currently-installed DeepSeek backend, if any. Called from
    /// `/setup deepseek disconnect` after the stored credentials are
    /// wiped so a subsequent `deepseek::*` request fails with "backend
    /// not configured" instead of firing 401-bound requests.
    pub fn uninstall_deepseek(&self) {
        self.uninstall(ModelSource::DEEPSEEK);
    }

    /// Install or replace the Grok OAuth backend after the external Grok
    /// CLI credential file changes.
    pub fn install_grok(&self, backend: Arc<dyn LlmBackend>) {
        self.install(ModelSource::GROK, backend);
    }

    pub fn uninstall_grok(&self) {
        self.uninstall(ModelSource::GROK);
    }

    /// Install (or replace) the OpenRouter backend at runtime. Called
    /// from `/openrouter-login <key>` so a session that started without
    /// `OPENROUTER_API_KEY` or an on-disk credential file picks up the
    /// new key on the next discovery refresh.
    pub fn install_openrouter(&self, backend: Arc<dyn LlmBackend>) {
        self.install(ModelSource::OPENROUTER, backend);
    }

    /// Drop the currently-installed OpenRouter backend, if any. Called
    /// from `/openrouter-login disconnect` after the on-disk credential
    /// file is wiped so a subsequent `openrouter::*` request fails with
    /// "backend not configured" instead of firing 401-bound requests.
    pub fn uninstall_openrouter(&self) {
        self.uninstall(ModelSource::OPENROUTER);
    }

    /// Snapshot the current Bedrock backend, if any. Cloning the inner Arc
    /// lets callers release the read lock immediately; they can then
    /// `.await` the backend without holding a guard.
    fn snapshot(&self, source: &str) -> Option<Arc<dyn LlmBackend>> {
        self.slot(source)
            .and_then(|slot| slot.backend.read().unwrap().clone())
    }

    /// Install (or replace) the ds4 backend. Called from each discovery
    /// refresh with a backend pointed at the port the running ds4-server is
    /// currently listening on. In-flight requests holding the old `Arc`
    /// finish against it; new `ds4::*` routes pick up the new port.
    fn install_ds4(&self, backend: Arc<dyn LlmBackend>) {
        self.install(ModelSource::DS4, backend);
    }

    /// Drop the ds4 backend. Called from a discovery refresh that no longer
    /// sees a running ds4-server, so a subsequent `ds4::*` request fails
    /// with the standard "backend not configured" error instead of hitting
    /// a now-dead port.
    fn uninstall_ds4(&self) {
        self.uninstall(ModelSource::DS4);
    }

    /// Re-resolve the local ds4-server URL and (re)install or drop the ds4
    /// chat backend so `ds4::*` routes to whatever port ds4-server is on
    /// right now. Returns the resolved base URL for the discovery probe, or
    /// `None` when no ds4-server is detected. The process/port probe is
    /// blocking, so it runs off the async worker via `spawn_blocking`.
    async fn refresh_ds4_backend(&self) -> Option<String> {
        let url = tokio::task::spawn_blocking(crate::discovery::ds4_base_url)
            .await
            .unwrap_or(None);
        match &url {
            Some(u) => {
                self.install_ds4(build_ds4_backend(u));
                tracing::info!("ds4-server detected at {u}; ds4::* chat routes there");
            }
            None => self.uninstall_ds4(),
        }
        url
    }

    async fn list_model_metadata_inner(
        &self,
        progress: Option<UnboundedSender<String>>,
    ) -> Result<Vec<ModelMetadata>> {
        if let Some(tx) = &progress {
            crate::openrouter_auth::append_refresh_log("Snapshotting configured backends...");
            let _ = tx.send("Snapshotting configured backends...\n".to_string());
        }
        if let Some(tx) = &progress {
            crate::openrouter_auth::append_refresh_log("Building discovery HTTP client...");
            let _ = tx.send("Building discovery HTTP client...\n".to_string());
        }
        let http = discovery_http_client();

        // Re-resolve ds4-server's port and update its registry slot before
        // snapshotting providers for this discovery round.
        self.refresh_ds4_backend().await;

        if let Some(tx) = &progress {
            crate::openrouter_auth::append_refresh_log("Launching provider checks...");
            let _ = tx.send("Launching provider checks...\n".to_string());
        }

        let checks = self.backends.iter().map(|slot| {
            let source = slot.source.clone();
            let label = slot.label.clone();
            let backend = slot.backend.read().unwrap().clone();
            let progress = progress.clone();
            let http = http.clone();
            async move {
                let metadata = if source == ModelSource::OLLAMA {
                    discover_ollama_metadata(&http, progress)
                        .await
                        .into_values()
                        .collect()
                } else {
                    discover_backend_metadata(label, backend, progress).await
                };
                metadata
                    .into_iter()
                    .map(|mut model| {
                        model.id = format!("{source}::{}", model.id);
                        model
                    })
                    .collect::<Vec<_>>()
            }
        });
        let catalogs = join_all(checks).await;
        if let Some(tx) = &progress {
            crate::openrouter_auth::append_refresh_log(
                "Provider checks finished. Merging catalogs...",
            );
            let _ = tx.send("Provider checks finished. Merging catalogs...\n".to_string());
        }
        let discovered = catalogs.into_iter().flatten().collect::<Vec<_>>();
        if let Some(tx) = &progress {
            crate::openrouter_auth::append_refresh_log(&format!(
                "Merged discovery results: {} model(s).",
                discovered.len()
            ));
            let _ = tx.send(format!(
                "Merged discovery results: {} model(s).\n",
                discovered.len()
            ));
        }
        Ok(discovered)
    }

    pub async fn list_model_metadata_with_progress(
        &self,
        progress: Option<UnboundedSender<String>>,
    ) -> Result<Vec<ModelMetadata>> {
        self.list_model_metadata_inner(progress).await
    }

    pub fn take_model_discovery_notices(&self) -> Vec<ModelDiscoveryNotice> {
        let mut notices = Vec::new();
        for slot in &self.backends {
            if let Some(backend) = slot.backend.read().unwrap().clone() {
                notices.extend(backend.take_model_discovery_notices());
            }
        }
        notices
    }

    fn pick(&self, source: &str) -> Option<Arc<dyn LlmBackend>> {
        self.snapshot(source)
    }

    /// Source to use when a chat request arrives with no `<source>::` prefix.
    /// Computed on demand (rather than cached at construction) so a Codex
    /// login or Bedrock key paste mid-session promotes it to the preferred
    /// fallback.
    ///
    /// Priority follows registry order. Bedrock wins when configured because
    /// its model ids are otherwise easy to type
    /// bare from the environment-driven setup. ds4 sits after Ollama so an
    /// existing Ollama user's bare-id fallback is unchanged, and direct
    /// DeepSeek beats generic OpenAI-compatible profiles and OpenRouter
    /// when both expose the same family.
    fn fallback_source(&self) -> Option<&str> {
        self.backends.iter().find_map(|slot| {
            slot.backend
                .read()
                .unwrap()
                .as_ref()
                .map(|_| slot.source.as_str())
        })
    }

    /// Resolve a wire-form model id to (backend, bare id). Bare ids (no
    /// `<source>::` prefix) route to the fallback source.
    fn resolve(&self, wire_model: &str) -> Result<(Arc<dyn LlmBackend>, String)> {
        if let Some((source, bare)) = split_wire_id(wire_model) {
            let backend = self.pick(source).ok_or_else(|| {
                anyhow::anyhow!(
                    "model {wire_model} requires the {source} backend, which is not configured"
                )
            })?;
            return Ok((backend, bare.to_string()));
        }
        let source = self.fallback_source().ok_or_else(|| {
            anyhow::anyhow!(
                "no LLM backend is configured and no `<source>::<id>` wire prefix was provided"
            )
        })?;
        let backend = self
            .pick(source)
            .expect("fallback_source returns Some only when its backend exists");
        Ok((backend, wire_model.to_string()))
    }
}

impl LlmBackend for MultiBackend {
    fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
        // Thin adapter over `list_model_metadata` so the bare-id and
        // metadata paths can't drift -- e.g. forgetting to pick up a
        // freshly-installed Codex backend on only one of the two.
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
        Box::pin(self.list_model_metadata_inner(None))
    }

    fn resolve_model_info(&self, configured_model: &str) -> ResolvedModelInfo {
        match self.resolve(configured_model) {
            Ok((backend, bare)) => {
                let source = split_wire_id(configured_model)
                    .map(|(source, _)| source)
                    .or_else(|| self.fallback_source());
                let info = backend.resolve_model_info(&bare);
                ResolvedModelInfo {
                    configured_model: configured_model.to_string(),
                    resolved_provider: info
                        .resolved_provider
                        .or_else(|| source.map(str::to_string)),
                    resolved_model: info.resolved_model,
                }
            }
            Err(_) => ResolvedModelInfo {
                configured_model: configured_model.to_string(),
                resolved_provider: split_wire_id(configured_model)
                    .map(|(source, _)| source.to_string()),
                resolved_model: configured_model.to_string(),
            },
        }
    }

    fn stream_chat(&self, request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
        let resolution = self.resolve(&request.model);
        Box::pin(async move {
            let (backend, bare) = resolution?;
            let mut request = request;
            request.model = bare;
            backend.stream_chat(request).await
        })
    }
}

async fn discover_backend_metadata(
    label: String,
    backend: Option<Arc<dyn LlmBackend>>,
    progress: Option<UnboundedSender<String>>,
) -> Vec<ModelMetadata> {
    let Some(backend) = backend else {
        if let Some(tx) = &progress {
            let _ = tx.send(format!("{label}: not configured.\n"));
        }
        return Vec::new();
    };

    if let Some(tx) = &progress {
        let _ = tx.send(format!("{label}: checking...\n"));
    }

    match tokio::time::timeout(PROVIDER_DISCOVERY_TIMEOUT, backend.list_model_metadata()).await {
        Ok(Ok(metadata)) => {
            if let Some(tx) = &progress {
                let _ = tx.send(format!("{label}: {} model(s).\n", metadata.len()));
            }
            metadata
        }
        Ok(Err(e)) => {
            tracing::info!("{label} model discovery skipped: {e:#}");
            if let Some(tx) = &progress {
                let _ = tx.send(format!("{label}: unavailable.\n"));
            }
            Vec::new()
        }
        Err(_) => {
            tracing::warn!(
                "{label} model discovery timed out after {:?}",
                PROVIDER_DISCOVERY_TIMEOUT
            );
            if let Some(tx) = &progress {
                let _ = tx.send(format!("{label}: timed out.\n"));
            }
            Vec::new()
        }
    }
}

async fn discover_ollama_metadata(
    http: &reqwest::Client,
    progress: Option<UnboundedSender<String>>,
) -> HashMap<String, ModelMetadata> {
    if let Some(tx) = &progress {
        let _ = tx.send("Local models: checking...\n".to_string());
    }

    match tokio::time::timeout(
        PROVIDER_DISCOVERY_TIMEOUT,
        discover_ollama_model_metadata(http, OLLAMA_DEFAULT_URL),
    )
    .await
    {
        Ok(metadata) => {
            if let Some(tx) = &progress {
                let _ = tx.send(format!("Local models: {} model(s).\n", metadata.len()));
            }
            metadata
        }
        Err(_) => {
            tracing::warn!(
                "ollama model metadata discovery timed out after {:?}",
                PROVIDER_DISCOVERY_TIMEOUT
            );
            if let Some(tx) = &progress {
                let _ = tx.send("Local models: timed out.\n".to_string());
            }
            HashMap::new()
        }
    }
}

/// Build a ds4 chat backend pointed at `base_url` (already resolved to the
/// running ds4-server's port by `discovery::ds4_base_url`). ds4-server is
/// OpenAI-compatible, so this mirrors the Ollama backend: an `OpenAiClient`
/// against `{base}/v1` with no API key.
fn build_ds4_backend(base_url: &str) -> Arc<dyn LlmBackend> {
    let base = base_url.trim_end_matches('/');
    let chat_url = format!("{base}/v1");
    Arc::new(crate::llm_client::OpenAiClient::with_reasoning_support(
        chat_url,
        None,
        reqwest::header::HeaderMap::new(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::future::FutureExt;
    use std::sync::Mutex;

    fn test_multi(
        bedrock: Option<Arc<dyn LlmBackend>>,
        codex: Option<Arc<dyn LlmBackend>>,
        deepseek: Option<Arc<dyn LlmBackend>>,
        openai: Option<Arc<dyn LlmBackend>>,
        openrouter: Option<Arc<dyn LlmBackend>>,
        ollama: Option<Arc<dyn LlmBackend>>,
    ) -> MultiBackend {
        MultiBackend::new(vec![
            BackendRegistration::new(ModelSource::BEDROCK, "Bedrock", bedrock),
            BackendRegistration::new(ModelSource::CODEX, "Codex", codex),
            BackendRegistration::new(ModelSource::OLLAMA, "Local models", ollama),
            BackendRegistration::new(ModelSource::DS4, "ds4", None),
            BackendRegistration::new(ModelSource::DEEPSEEK, "DeepSeek", deepseek),
            BackendRegistration::new(ModelSource::KIMI, "Kimi", None),
            BackendRegistration::new(ModelSource::OPENAI, "OpenAI-compatible", openai),
            BackendRegistration::new(ModelSource::OPENROUTER, "OpenRouter", openrouter),
        ])
    }

    fn chat_request(model: &str, reasoning_effort: Option<&str>) -> StreamChatRequest {
        chat_request_with_service_tier(model, reasoning_effort, None)
    }

    fn chat_request_with_service_tier(
        model: &str,
        reasoning_effort: Option<&str>,
        service_tier: Option<&str>,
    ) -> StreamChatRequest {
        StreamChatRequest {
            model: model.to_string(),
            messages: vec![],
            tools: None,
            reasoning_effort: reasoning_effort.map(str::to_string),
            service_tier: service_tier.map(str::to_string),
            temperature: None,
            structured_output: None,
            on_token: Box::new(|_| {}),
            on_thought: Box::new(|_| {}),
            cancel: tokio_util::sync::CancellationToken::new(),
            idle_timeouts: IdleTimeouts::uniform(std::time::Duration::from_secs(60)),
        }
    }

    /// Test double that records the model id and reasoning effort it was
    /// called with. Lets us assert that `MultiBackend` strips the
    /// `<source>::` prefix before delegating, so the inner client
    /// receives the bare id Ollama or the Responses API actually expects,
    /// and that the per-session reasoning_effort threads all the way
    /// through the dispatcher unchanged.
    struct RecordingBackend {
        name: &'static str,
        last_model: Arc<Mutex<Option<String>>>,
        last_reasoning_effort: Arc<Mutex<Option<String>>>,
        last_service_tier: Arc<Mutex<Option<String>>>,
    }

    impl LlmBackend for RecordingBackend {
        fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
            let name = self.name;
            async move { Ok(vec![format!("{name}-stub")]) }.boxed()
        }

        fn stream_chat(&self, request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
            *self.last_model.lock().unwrap() = Some(request.model);
            *self.last_reasoning_effort.lock().unwrap() = request.reasoning_effort;
            *self.last_service_tier.lock().unwrap() = request.service_tier;
            let response = LlmResponse::Text {
                text: format!("hello from {}", self.name),
                reasoning_content: None,
                usage: crate::llm_client::TokenUsage::default(),
                codex_reasoning: None,
            };
            async move { Ok(response) }.boxed()
        }
    }

    /// Captured per-call state for assertions.
    struct RecordingHandles {
        last_model: Arc<Mutex<Option<String>>>,
        last_reasoning_effort: Arc<Mutex<Option<String>>>,
        last_service_tier: Arc<Mutex<Option<String>>>,
    }

    fn recording(name: &'static str) -> (Arc<dyn LlmBackend>, RecordingHandles) {
        let last_model = Arc::new(Mutex::new(None));
        let last_reasoning_effort = Arc::new(Mutex::new(None));
        let last_service_tier = Arc::new(Mutex::new(None));
        let backend = Arc::new(RecordingBackend {
            name,
            last_model: last_model.clone(),
            last_reasoning_effort: last_reasoning_effort.clone(),
            last_service_tier: last_service_tier.clone(),
        });
        (
            backend,
            RecordingHandles {
                last_model,
                last_reasoning_effort,
                last_service_tier,
            },
        )
    }

    struct HangingBackend;

    impl LlmBackend for HangingBackend {
        fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
            futures::future::pending().boxed()
        }

        fn list_model_metadata(&self) -> BoxFuture<'_, Result<Vec<ModelMetadata>>> {
            futures::future::pending().boxed()
        }

        fn stream_chat(&self, _request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
            async move { anyhow::bail!("stream_chat should not be called in this test") }.boxed()
        }
    }

    /// Wire ids tagged `codex::` route to the Codex backend with the bare
    /// id, while `ollama::` ids route to Ollama. Each backend records the
    /// model string it received so we can assert the prefix was stripped.
    #[tokio::test]
    async fn stream_chat_routes_by_wire_prefix() {
        let (codex_backend, codex_handles) = recording("codex");
        let (ollama_backend, ollama_handles) = recording("ollama");
        let multi = test_multi(
            None,
            Some(codex_backend),
            None,
            None,
            None,
            Some(ollama_backend),
        );

        let _ = multi
            .stream_chat(chat_request("codex::gpt-5-codex", None))
            .await
            .expect("codex route");
        assert_eq!(
            codex_handles.last_model.lock().unwrap().as_deref(),
            Some("gpt-5-codex")
        );
        assert!(ollama_handles.last_model.lock().unwrap().is_none());

        let _ = multi
            .stream_chat(chat_request("ollama::llama3:latest", None))
            .await
            .expect("ollama route");
        // The Ollama tag suffix must survive prefix stripping.
        assert_eq!(
            ollama_handles.last_model.lock().unwrap().as_deref(),
            Some("llama3:latest")
        );
    }

    /// A bare model id (no `<source>::` prefix) routes to the fallback
    /// source. With both backends configured, Codex wins -- it's the more
    /// capable choice and the more likely user intent for a bare id.
    #[tokio::test]
    async fn bare_id_routes_to_codex_fallback_when_both_configured() {
        let (codex_backend, codex_handles) = recording("codex");
        let (ollama_backend, ollama_handles) = recording("ollama");
        let multi = test_multi(
            None,
            Some(codex_backend),
            None,
            None,
            None,
            Some(ollama_backend),
        );

        let _ = multi
            .stream_chat(chat_request("gpt-5-codex", None))
            .await
            .expect("bare id falls back to codex");
        assert_eq!(
            codex_handles.last_model.lock().unwrap().as_deref(),
            Some("gpt-5-codex")
        );
        assert!(ollama_handles.last_model.lock().unwrap().is_none());
    }

    /// Bare id with only Ollama configured: falls through to Ollama
    /// rather than erroring. Lets users with no Codex login still type
    /// raw model ids into `/config`.
    #[tokio::test]
    async fn bare_id_routes_to_ollama_when_codex_absent() {
        let (ollama_backend, ollama_handles) = recording("ollama");
        let multi = test_multi(None, None, None, None, None, Some(ollama_backend));

        let _ = multi
            .stream_chat(chat_request("llama3", None))
            .await
            .expect("bare id falls back to ollama");
        assert_eq!(
            ollama_handles.last_model.lock().unwrap().as_deref(),
            Some("llama3")
        );
    }

    /// Wire id requesting an absent backend errors loudly instead of
    /// silently falling through to the other source -- if the user picked
    /// `codex::gpt-5` from the catalog and the Codex login expired, we
    /// must NOT route the request to Ollama under a different model name.
    #[tokio::test]
    async fn wire_id_for_absent_backend_returns_error() {
        // Only Ollama is configured; a `codex::` wire id must error.
        let (ollama_backend, _ollama_handles) = recording("ollama");
        let multi = test_multi(None, None, None, None, None, Some(ollama_backend));

        let err = multi
            .stream_chat(chat_request("codex::gpt-5", None))
            .await
            .expect_err("codex route must fail when codex backend is absent");
        let msg = format!("{err:#}");
        assert!(msg.contains("codex"), "error must mention codex: {msg}");
    }

    /// When neither backend is configured, every chat request errors
    /// rather than panics. `MultiBackend` is constructible empty (the
    /// server still starts -- the user can run `/setup codex` mid-session
    /// or start Ollama and re-discover) but no model can be routed.
    #[tokio::test]
    async fn empty_multi_backend_errors_on_chat() {
        let multi = test_multi(None, None, None, None, None, None);
        let err = multi
            .stream_chat(chat_request("anything", None))
            .await
            .expect_err("no backend means no route");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no LLM backend is configured"),
            "error must explain the empty-backend case: {msg}"
        );
    }

    /// Regression for the `/setup codex` lifecycle (issue #3555): the
    /// server starts with no Codex backend (auth.json absent), the user
    /// runs `/setup codex`, and the new backend is installed at
    /// runtime. Subsequent `codex::*` routing must succeed -- previously
    /// it kept returning the "backend not configured" error because the
    /// `None` was captured permanently at construction.
    #[tokio::test]
    async fn codex_installed_after_login_is_routable() {
        // Start with no Codex (mirrors the empty-auth.json startup path).
        let multi = test_multi(None, None, None, None, None, None);

        // Pre-install: a `codex::*` request must fail loudly.
        let err = multi
            .stream_chat(chat_request("codex::gpt-5-codex", None))
            .await
            .expect_err("codex route must fail before install");
        assert!(format!("{err:#}").contains("codex"));

        // User runs `/setup codex` successfully -- the handler installs
        // a freshly-built Codex backend.
        let (codex_backend, codex_handles) = recording("codex");
        multi.install_codex(codex_backend);

        // Now the same request routes through Codex with the prefix
        // stripped, exactly as if the credentials had been there at
        // startup.
        let _ = multi
            .stream_chat(chat_request("codex::gpt-5-codex", None))
            .await
            .expect("codex route must succeed after install");
        assert_eq!(
            codex_handles.last_model.lock().unwrap().as_deref(),
            Some("gpt-5-codex")
        );
    }

    /// Bare ids must also start routing to Codex once it's installed.
    /// Before the fix, `fallback_source` was frozen at construction --
    /// so even after `install_codex` a bare `gpt-5-codex` would error
    /// out with "no LLM backend is configured" because the cached
    /// fallback was still `None`.
    #[tokio::test]
    async fn bare_id_falls_back_to_codex_after_install() {
        let multi = test_multi(None, None, None, None, None, None);

        let (codex_backend, codex_handles) = recording("codex");
        multi.install_codex(codex_backend);

        let _ = multi
            .stream_chat(chat_request("gpt-5-codex", None))
            .await
            .expect("bare id must route to newly-installed codex");
        assert_eq!(
            codex_handles.last_model.lock().unwrap().as_deref(),
            Some("gpt-5-codex")
        );
    }

    /// `list_models` must consult the currently-installed Codex backend,
    /// not the one captured at construction. Without this, a successful
    /// `/setup codex` followed by a discovery refresh (e.g. on
    /// `session/new`) would keep returning an empty Codex list and the
    /// model picker would never show Codex models.
    #[tokio::test]
    async fn list_models_reflects_installed_codex() {
        let multi = test_multi(None, None, None, None, None, None);
        let (codex_backend, _codex_handles) = recording("codex");
        multi.install_codex(codex_backend);

        // RecordingBackend::list_models returns ["codex-stub"], so that id
        // surfacing proves the freshly-installed registry slot was consulted.
        let models = multi.list_models().await.expect("discovery must succeed");
        assert!(
            models.iter().any(|m| m.contains("codex-stub")),
            "installed codex backend must contribute to discovery: got {models:?}"
        );
    }

    /// `/setup codex disconnect` calls `uninstall_codex` after wiping
    /// auth.json. Subsequent `codex::*` routing must fail with the same
    /// "backend not configured" error a fresh-no-auth.json startup
    /// gives -- otherwise a wire id picked from a stale `availableModels`
    /// list would fire a request against credentials that no longer
    /// exist on disk.
    #[tokio::test]
    async fn codex_uninstall_unroutes_codex_requests() {
        let multi = test_multi(None, None, None, None, None, None);
        let (codex_backend, _codex_handles) = recording("codex");
        multi.install_codex(codex_backend);

        // Sanity check: routable while installed.
        let _ = multi
            .stream_chat(chat_request("codex::gpt-5-codex", None))
            .await
            .expect("codex route must succeed while installed");

        // Disconnect path drops the backend.
        multi.uninstall_codex();

        let err = multi
            .stream_chat(chat_request("codex::gpt-5-codex", None))
            .await
            .expect_err("codex route must fail after uninstall");
        assert!(
            format!("{err:#}").contains("codex"),
            "error must mention codex backend"
        );
    }

    #[tokio::test]
    async fn bedrock_installed_after_setup_is_routable() {
        let multi = test_multi(None, None, None, None, None, None);

        let err = multi
            .stream_chat(chat_request(
                "bedrock::us.anthropic.claude-sonnet-4-6",
                None,
            ))
            .await
            .expect_err("bedrock route must fail before install");
        assert!(format!("{err:#}").contains("bedrock"));

        let (bedrock_backend, bedrock_handles) = recording("bedrock");
        multi.install_bedrock(bedrock_backend);

        let _ = multi
            .stream_chat(chat_request(
                "bedrock::us.anthropic.claude-sonnet-4-6",
                None,
            ))
            .await
            .expect("bedrock route must succeed after install");
        assert_eq!(
            bedrock_handles.last_model.lock().unwrap().as_deref(),
            Some("us.anthropic.claude-sonnet-4-6")
        );
    }

    #[tokio::test]
    async fn bare_id_falls_back_to_bedrock_after_install() {
        let (codex_backend, codex_handles) = recording("codex");
        let multi = test_multi(None, Some(codex_backend), None, None, None, None);

        let (bedrock_backend, bedrock_handles) = recording("bedrock");
        multi.install_bedrock(bedrock_backend);

        let _ = multi
            .stream_chat(chat_request("us.anthropic.claude-sonnet-4-6", None))
            .await
            .expect("bare id must route to newly-installed bedrock");
        assert_eq!(
            bedrock_handles.last_model.lock().unwrap().as_deref(),
            Some("us.anthropic.claude-sonnet-4-6")
        );
        assert!(codex_handles.last_model.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn bedrock_uninstall_unroutes_bedrock_requests() {
        let multi = test_multi(None, None, None, None, None, None);
        let (bedrock_backend, _bedrock_handles) = recording("bedrock");
        multi.install_bedrock(bedrock_backend);

        let _ = multi
            .stream_chat(chat_request(
                "bedrock::us.anthropic.claude-sonnet-4-6",
                None,
            ))
            .await
            .expect("bedrock route must succeed while installed");

        multi.uninstall_bedrock();

        let err = multi
            .stream_chat(chat_request(
                "bedrock::us.anthropic.claude-sonnet-4-6",
                None,
            ))
            .await
            .expect_err("bedrock route must fail after uninstall");
        assert!(
            format!("{err:#}").contains("bedrock"),
            "error must mention bedrock backend"
        );
    }

    /// Wire ids tagged `openrouter::` route to the OpenRouter backend
    /// with the bare id (slash-separated `vendor/model`), and do NOT
    /// leak to Codex or Ollama when those are also configured.
    #[tokio::test]
    async fn openrouter_wire_id_routes_to_openrouter() {
        let (codex_backend, codex_handles) = recording("codex");
        let (openrouter_backend, openrouter_handles) = recording("openrouter");
        let (ollama_backend, ollama_handles) = recording("ollama");
        let multi = test_multi(
            None,
            Some(codex_backend),
            None,
            None,
            Some(openrouter_backend),
            Some(ollama_backend),
        );

        let _ = multi
            .stream_chat(chat_request(
                "openrouter::anthropic/claude-3.5-sonnet",
                None,
            ))
            .await
            .expect("openrouter route");
        // The inner slash in `vendor/model` must survive prefix
        // stripping; OpenRouter expects the slashed id verbatim.
        assert_eq!(
            openrouter_handles.last_model.lock().unwrap().as_deref(),
            Some("anthropic/claude-3.5-sonnet")
        );
        assert!(codex_handles.last_model.lock().unwrap().is_none());
        assert!(ollama_handles.last_model.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn deepseek_wire_id_routes_to_deepseek() {
        let (codex_backend, codex_handles) = recording("codex");
        let (deepseek_backend, deepseek_handles) = recording("deepseek");
        let (openrouter_backend, openrouter_handles) = recording("openrouter");
        let multi = test_multi(
            None,
            Some(codex_backend),
            Some(deepseek_backend),
            None,
            Some(openrouter_backend),
            None,
        );

        let _ = multi
            .stream_chat(chat_request("deepseek::deepseek-v4-pro", None))
            .await
            .expect("deepseek route");
        assert_eq!(
            deepseek_handles.last_model.lock().unwrap().as_deref(),
            Some("deepseek-v4-pro")
        );
        assert!(codex_handles.last_model.lock().unwrap().is_none());
        assert!(openrouter_handles.last_model.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn registry_routes_kimi_and_preserves_discovery_order() {
        let (deepseek_backend, _) = recording("deepseek");
        let (kimi_backend, kimi_handles) = recording("k3");
        let (openai_backend, _) = recording("openai");
        let multi = MultiBackend::new(vec![
            BackendRegistration::new(ModelSource::DEEPSEEK, "DeepSeek", Some(deepseek_backend)),
            BackendRegistration::new(ModelSource::KIMI, "Kimi", Some(kimi_backend)),
            BackendRegistration::new(ModelSource::OPENAI, "OpenAI", Some(openai_backend)),
        ]);

        multi
            .stream_chat(chat_request("kimi::k3", Some("max")))
            .await
            .expect("Kimi route");
        assert_eq!(
            kimi_handles.last_model.lock().unwrap().as_deref(),
            Some("k3")
        );
        assert_eq!(
            kimi_handles
                .last_reasoning_effort
                .lock()
                .unwrap()
                .as_deref(),
            Some("max")
        );

        assert_eq!(
            multi.list_models().await.unwrap(),
            vec![
                "deepseek::deepseek-stub",
                "kimi::k3-stub",
                "openai::openai-stub",
            ]
        );
    }

    #[tokio::test]
    async fn bare_id_routes_to_deepseek_when_only_deepseek_configured() {
        let (deepseek_backend, deepseek_handles) = recording("deepseek");
        let multi = test_multi(None, None, Some(deepseek_backend), None, None, None);

        let _ = multi
            .stream_chat(chat_request("deepseek-v4-flash", None))
            .await
            .expect("bare id falls back to deepseek");
        assert_eq!(
            deepseek_handles.last_model.lock().unwrap().as_deref(),
            Some("deepseek-v4-flash")
        );
    }

    #[tokio::test]
    async fn openai_wire_id_routes_to_profile_and_strips_profile_prefix() {
        let (openai_backend, openai_handles) = recording("openai");
        let multi = test_multi(None, None, None, Some(openai_backend), None, None);

        let _ = multi
            .stream_chat(chat_request("openai::deca/deca-model", None))
            .await
            .expect("openai profile route");
        assert_eq!(
            openai_handles.last_model.lock().unwrap().as_deref(),
            Some("deca/deca-model")
        );
    }

    #[tokio::test]
    async fn bare_id_routes_to_openai_before_openrouter() {
        let (openai_backend, openai_handles) = recording("openai");
        let (openrouter_backend, openrouter_handles) = recording("openrouter");
        let multi = test_multi(
            None,
            None,
            None,
            Some(openai_backend),
            Some(openrouter_backend),
            None,
        );

        let _ = multi
            .stream_chat(chat_request("some-model", None))
            .await
            .expect("bare id falls back to openai-compatible provider");
        assert_eq!(
            openai_handles.last_model.lock().unwrap().as_deref(),
            Some("some-model")
        );
        assert!(openrouter_handles.last_model.lock().unwrap().is_none());
    }

    #[test]
    fn resolve_model_info_reports_openai_profile_provider() {
        let openai_backend =
            crate::openai_providers::build_backend(crate::openai_providers::OpenAiProviderConfig {
                profiles: vec![crate::openai_providers::OpenAiProviderProfile {
                    name: "deca".to_string(),
                    base_url: "https://api.genlabs.dev/deca/v1".to_string(),
                    api_key_env: None,
                    api_key: None,
                }],
            })
            .expect("non-empty openai config builds a backend");
        let multi = test_multi(None, None, None, Some(openai_backend), None, None);

        let info = multi.resolve_model_info("openai::deca/deca-model");

        assert_eq!(info.configured_model, "openai::deca/deca-model");
        assert_eq!(info.resolved_provider.as_deref(), Some("openai/deca"));
        assert_eq!(info.resolved_model, "deca-model");
    }

    /// A bare id with only OpenRouter configured falls back to
    /// OpenRouter rather than erroring -- the same fallback contract
    /// Ollama gets when it's the only backend.
    #[tokio::test]
    async fn bare_id_routes_to_openrouter_when_only_openrouter_configured() {
        let (openrouter_backend, openrouter_handles) = recording("openrouter");
        let multi = test_multi(None, None, None, None, Some(openrouter_backend), None);

        let _ = multi
            .stream_chat(chat_request("anthropic/claude-3.5-sonnet", None))
            .await
            .expect("bare id falls back to openrouter");
        assert_eq!(
            openrouter_handles.last_model.lock().unwrap().as_deref(),
            Some("anthropic/claude-3.5-sonnet")
        );
    }

    #[test]
    fn resolve_model_info_reports_wire_provider_and_bare_model() {
        let (openrouter_backend, _openrouter_handles) = recording("openrouter");
        let multi = test_multi(None, None, None, None, Some(openrouter_backend), None);

        let info = multi.resolve_model_info("openrouter::google/gemini-3.1-pro-preview");

        assert_eq!(
            info.configured_model,
            "openrouter::google/gemini-3.1-pro-preview"
        );
        assert_eq!(info.resolved_provider.as_deref(), Some("openrouter"));
        assert_eq!(info.resolved_model, "google/gemini-3.1-pro-preview");
    }

    #[test]
    fn resolve_model_info_reports_bare_model_fallback_provider() {
        let (openrouter_backend, _openrouter_handles) = recording("openrouter");
        let multi = test_multi(None, None, None, None, Some(openrouter_backend), None);

        let info = multi.resolve_model_info("gemini-3.1-pro-preview");

        assert_eq!(info.configured_model, "gemini-3.1-pro-preview");
        assert_eq!(info.resolved_provider.as_deref(), Some("openrouter"));
        assert_eq!(info.resolved_model, "gemini-3.1-pro-preview");
    }

    /// Registry order defines fallback priority. With these providers
    /// configured, a bare id routes to Codex. With Codex absent, Ollama wins
    /// over OpenRouter so a free
    /// local daemon beats a paid cloud router unless the user explicitly
    /// chooses an `openrouter::` model.
    #[tokio::test]
    async fn bare_id_prefers_ollama_over_openrouter_when_codex_absent() {
        let (openrouter_backend, openrouter_handles) = recording("openrouter");
        let (ollama_backend, ollama_handles) = recording("ollama");
        let multi = test_multi(
            None,
            None,
            None,
            None,
            Some(openrouter_backend),
            Some(ollama_backend),
        );

        let _ = multi
            .stream_chat(chat_request("some-bare-id", None))
            .await
            .expect("bare id falls back to ollama");
        assert_eq!(
            ollama_handles.last_model.lock().unwrap().as_deref(),
            Some("some-bare-id")
        );
        assert!(openrouter_handles.last_model.lock().unwrap().is_none());
    }

    /// Wire id requesting an absent OpenRouter backend errors loudly
    /// rather than silently routing to a different source. Same contract
    /// as `wire_id_for_absent_backend_returns_error` for Codex -- if the
    /// user picks `openrouter::vendor/model` from a catalog snapshot and
    /// the key has since been unexported, we must NOT route the request
    /// to Codex or Ollama under a different (and probably nonexistent)
    /// model id.
    #[tokio::test]
    async fn openrouter_wire_id_for_absent_backend_returns_error() {
        let (codex_backend, _codex_handles) = recording("codex");
        let multi = test_multi(None, Some(codex_backend), None, None, None, None);

        let err = multi
            .stream_chat(chat_request(
                "openrouter::anthropic/claude-3.5-sonnet",
                None,
            ))
            .await
            .expect_err("openrouter route must fail when openrouter backend is absent");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("openrouter"),
            "error must mention openrouter: {msg}"
        );
    }

    /// `reasoning_effort` threads through the dispatcher unchanged --
    /// it must arrive at the resolved inner backend, not get swallowed
    /// or coerced. (This protects the Codex picker's per-session
    /// selection from being silently lost on its way through
    /// `MultiBackend`.)
    #[tokio::test]
    async fn stream_chat_forwards_reasoning_effort() {
        let (codex_backend, codex_handles) = recording("codex");
        let multi = test_multi(None, Some(codex_backend), None, None, None, None);

        let _ = multi
            .stream_chat(chat_request("codex::gpt-5.2", Some("xhigh")))
            .await
            .expect("codex route");
        assert_eq!(
            codex_handles
                .last_reasoning_effort
                .lock()
                .unwrap()
                .as_deref(),
            Some("xhigh"),
            "reasoning_effort must arrive at the inner backend unchanged"
        );
    }

    #[tokio::test]
    async fn stream_chat_forwards_service_tier() {
        let (codex_backend, codex_handles) = recording("codex");
        let multi = test_multi(None, Some(codex_backend), None, None, None, None);

        let _ = multi
            .stream_chat(chat_request_with_service_tier(
                "codex::gpt-5.5",
                None,
                Some("priority"),
            ))
            .await
            .expect("codex route");
        assert_eq!(
            codex_handles.last_service_tier.lock().unwrap().as_deref(),
            Some("priority"),
            "service_tier must arrive at the inner backend unchanged"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn list_models_times_out_stuck_provider_and_keeps_healthy_ones() {
        let hanging: Arc<dyn LlmBackend> = Arc::new(HangingBackend);
        let (openrouter_backend, _openrouter_handles) = recording("openrouter");
        let multi = test_multi(
            None,
            Some(hanging),
            None,
            None,
            Some(openrouter_backend),
            None,
        );

        let models = multi.list_models().await.expect("discovery must succeed");
        assert!(
            models.iter().any(|m| m == "openrouter::openrouter-stub"),
            "healthy provider should still contribute models: got {models:?}"
        );
    }

    #[test]
    fn explicit_utility_route_requires_qualified_available_provider() {
        let (deepseek_backend, _) = recording("deepseek");
        let multi = test_multi(None, None, Some(deepseek_backend), None, None, None);

        multi
            .validate_explicit_model_route("deepseek::deepseek-v4-flash")
            .expect("configured provider is accepted");
        assert!(
            multi
                .validate_explicit_model_route("deepseek-v4-flash")
                .unwrap_err()
                .to_string()
                .contains("provider-qualified")
        );
        assert!(
            multi
                .validate_explicit_model_route("bedrock::openai.gpt-5.6-luna")
                .unwrap_err()
                .to_string()
                .contains("not configured")
        );
    }
}
