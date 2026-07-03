//! Routing `LlmBackend` that fans `list_models` out to all configured
//! sources and dispatches `stream_chat` to the right one based on the
//! `<source>::<id>` wire prefix produced by `discovery.rs`.
//!
//! Why a separate type? `OpenAiClient` and `CodexClient` already implement
//! `LlmBackend` for one transport each. Wrapping the configured sources
//! (Bedrock, Codex, hosted DeepSeek, OpenRouter, Ollama) in a single routing
//! backend lets `agent.rs` stay
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
use futures::future::BoxFuture;
use tokio::sync::mpsc::UnboundedSender;

use crate::discovery::{
    DiscoveredModel, ModelSource, OLLAMA_DEFAULT_URL, discover_all, discover_ollama_model_metadata,
    discovery_http_client, split_wire_id,
};
#[cfg(test)]
use crate::llm_client::IdleTimeouts;
use crate::llm_client::{
    LlmBackend, LlmResponse, ModelMetadata, ResolvedModelInfo, StreamChatRequest,
};

const PROVIDER_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(15);

/// LLM backend that routes by `<source>::<id>` prefix. Any inner backend
/// may be absent (e.g. no `auth.json`, no `DEEPSEEK_API_KEY`, no
/// `OPENROUTER_API_KEY`, or no
/// Ollama on the default port); calls for a source whose backend isn't
/// configured return a clear error rather than silently falling through.
///
/// The Codex, hosted DeepSeek, and OpenRouter slots are held behind
/// `RwLock`s so a runtime install can update them without a server
/// restart. The lock is only ever held for the duration of a synchronous
/// `Option<Arc<...>>` clone -- we never hold it across an `.await`.
///
/// Ollama stays a plain `Option<Arc<...>>`: there's no login flow for
/// it (the daemon either listens on the default port or it doesn't),
/// so the slot is set once at construction and never mutated.
///
/// ds4 is behind an `RwLock` like Codex/OpenRouter, but for a different
/// reason: ds4-server has no fixed port, so each discovery refresh
/// re-resolves the running server's port (see `discovery::ds4_base_url`)
/// and reinstalls a backend pointed at it. That keeps `ds4::*` chat
/// routing aimed at the same port discovery just found, and lets ds4 come
/// online when it's started after Anvil.
pub struct MultiBackend {
    bedrock: RwLock<Option<Arc<dyn LlmBackend>>>,
    codex: RwLock<Option<Arc<dyn LlmBackend>>>,
    deepseek: RwLock<Option<Arc<dyn LlmBackend>>>,
    openrouter: RwLock<Option<Arc<dyn LlmBackend>>>,
    ollama: Option<Arc<dyn LlmBackend>>,
    ds4: RwLock<Option<Arc<dyn LlmBackend>>>,
}

impl MultiBackend {
    pub fn new(
        bedrock: Option<Arc<dyn LlmBackend>>,
        codex: Option<Arc<dyn LlmBackend>>,
        deepseek: Option<Arc<dyn LlmBackend>>,
        openrouter: Option<Arc<dyn LlmBackend>>,
        ollama: Option<Arc<dyn LlmBackend>>,
    ) -> Self {
        Self {
            bedrock: RwLock::new(bedrock),
            codex: RwLock::new(codex),
            deepseek: RwLock::new(deepseek),
            openrouter: RwLock::new(openrouter),
            ollama,
            // Resolved on the first discovery refresh (the eager startup
            // probe), then on every refresh thereafter.
            ds4: RwLock::new(None),
        }
    }

    /// Install (or replace) the Bedrock backend at runtime. Called from
    /// `/setup bedrock key <token>` so a session that started without
    /// Bedrock credentials picks up the new key on the next discovery
    /// refresh.
    pub fn install_bedrock(&self, backend: Arc<dyn LlmBackend>) {
        *self.bedrock.write().unwrap() = Some(backend);
    }

    /// Drop the currently-installed Bedrock backend, if any. Called
    /// from `/setup bedrock disconnect` after the on-disk credentials
    /// are wiped.
    pub fn uninstall_bedrock(&self) {
        *self.bedrock.write().unwrap() = None;
    }

    /// Install (or replace) the Codex backend at runtime. Called from
    /// the `/codex-login` handler so the next discovery refresh and any
    /// subsequent `codex::*` route picks it up without a server restart.
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
        *self.codex.write().unwrap() = Some(backend);
    }

    /// Drop the currently-installed Codex backend, if any. Called from
    /// `/codex-login disconnect` after the on-disk credentials are
    /// wiped so a subsequent `codex::*` request fails with the same
    /// "backend not configured" error a fresh-no-auth.json startup
    /// would give, instead of firing requests with credentials that
    /// will now 401. In-flight requests holding an `Arc` to the old
    /// backend complete against that captured instance.
    pub fn uninstall_codex(&self) {
        *self.codex.write().unwrap() = None;
    }

    /// Install (or replace) the hosted DeepSeek backend at runtime.
    /// Called from `/setup deepseek key <key>` so a session that started
    /// without `DEEPSEEK_API_KEY` or a stored key picks up the new key on
    /// the next discovery refresh.
    pub fn install_deepseek(&self, backend: Arc<dyn LlmBackend>) {
        *self.deepseek.write().unwrap() = Some(backend);
    }

    /// Drop the currently-installed DeepSeek backend, if any. Called from
    /// `/setup deepseek disconnect` after the stored credentials are
    /// wiped so a subsequent `deepseek::*` request fails with "backend
    /// not configured" instead of firing 401-bound requests.
    pub fn uninstall_deepseek(&self) {
        *self.deepseek.write().unwrap() = None;
    }

    /// Install (or replace) the OpenRouter backend at runtime. Called
    /// from `/openrouter-login <key>` so a session that started without
    /// `OPENROUTER_API_KEY` or an on-disk credential file picks up the
    /// new key on the next discovery refresh.
    pub fn install_openrouter(&self, backend: Arc<dyn LlmBackend>) {
        *self.openrouter.write().unwrap() = Some(backend);
    }

    /// Drop the currently-installed OpenRouter backend, if any. Called
    /// from `/openrouter-login disconnect` after the on-disk credential
    /// file is wiped so a subsequent `openrouter::*` request fails with
    /// "backend not configured" instead of firing 401-bound requests.
    pub fn uninstall_openrouter(&self) {
        *self.openrouter.write().unwrap() = None;
    }

    /// Snapshot the current Bedrock backend, if any. Cloning the inner Arc
    /// lets callers release the read lock immediately; they can then
    /// `.await` the backend without holding a guard.
    fn bedrock_snapshot(&self) -> Option<Arc<dyn LlmBackend>> {
        self.bedrock.read().unwrap().clone()
    }

    /// Snapshot the current Codex backend, if any. Cloning the inner Arc
    /// lets callers release the read lock immediately; they can then
    /// `.await` the backend without holding a guard.
    fn codex_snapshot(&self) -> Option<Arc<dyn LlmBackend>> {
        self.codex.read().unwrap().clone()
    }

    /// Snapshot the current hosted DeepSeek backend, if any.
    fn deepseek_snapshot(&self) -> Option<Arc<dyn LlmBackend>> {
        self.deepseek.read().unwrap().clone()
    }

    /// Snapshot the current OpenRouter backend, if any. Same shape as
    /// `codex_snapshot` -- callers release the read lock before awaiting.
    fn openrouter_snapshot(&self) -> Option<Arc<dyn LlmBackend>> {
        self.openrouter.read().unwrap().clone()
    }

    /// Install (or replace) the ds4 backend. Called from each discovery
    /// refresh with a backend pointed at the port the running ds4-server is
    /// currently listening on. In-flight requests holding the old `Arc`
    /// finish against it; new `ds4::*` routes pick up the new port.
    fn install_ds4(&self, backend: Arc<dyn LlmBackend>) {
        *self.ds4.write().unwrap() = Some(backend);
    }

    /// Drop the ds4 backend. Called from a discovery refresh that no longer
    /// sees a running ds4-server, so a subsequent `ds4::*` request fails
    /// with the standard "backend not configured" error instead of hitting
    /// a now-dead port.
    fn uninstall_ds4(&self) {
        *self.ds4.write().unwrap() = None;
    }

    /// Snapshot the current ds4 backend, if any. Same shape as
    /// `codex_snapshot` -- callers release the read lock before awaiting.
    fn ds4_snapshot(&self) -> Option<Arc<dyn LlmBackend>> {
        self.ds4.read().unwrap().clone()
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
        let bedrock = self.bedrock_snapshot();
        let codex = self.codex_snapshot();
        let deepseek = self.deepseek_snapshot();
        let openrouter = self.openrouter_snapshot();
        if let Some(tx) = &progress {
            crate::openrouter_auth::append_refresh_log("Building discovery HTTP client...");
            let _ = tx.send("Building discovery HTTP client...\n".to_string());
        }
        let http = discovery_http_client();
        if let Some(tx) = &progress {
            crate::openrouter_auth::append_refresh_log("Launching provider checks...");
            let _ = tx.send("Launching provider checks...\n".to_string());
        }

        let (
            bedrock_metadata,
            codex_metadata,
            deepseek_metadata,
            openrouter_metadata,
            ollama_metadata,
        ) = tokio::join!(
            discover_backend_metadata("Bedrock", bedrock, progress.clone()),
            discover_backend_metadata("Codex", codex, progress.clone()),
            discover_backend_metadata("DeepSeek", deepseek, progress.clone()),
            discover_backend_metadata("OpenRouter", openrouter, progress.clone()),
            discover_ollama_metadata(&http, progress.clone()),
        );
        if let Some(tx) = &progress {
            crate::openrouter_auth::append_refresh_log(
                "Provider checks finished. Merging catalogs...",
            );
            let _ = tx.send("Provider checks finished. Merging catalogs...\n".to_string());
        }

        let bedrock_by_id: HashMap<String, ModelMetadata> = bedrock_metadata
            .iter()
            .map(|m| (m.id.clone(), m.clone()))
            .collect();
        let bedrock_ids: Vec<String> = bedrock_metadata.iter().map(|m| m.id.clone()).collect();
        let bedrock_lookup = || async move { Ok::<_, anyhow::Error>(bedrock_ids) };

        let codex_by_id: HashMap<String, ModelMetadata> = codex_metadata
            .iter()
            .map(|m| (m.id.clone(), m.clone()))
            .collect();
        let codex_ids: Vec<String> = codex_metadata.iter().map(|m| m.id.clone()).collect();
        let codex_lookup = || async move { Ok::<_, anyhow::Error>(codex_ids) };

        let deepseek_by_id: HashMap<String, ModelMetadata> = deepseek_metadata
            .iter()
            .map(|m| (m.id.clone(), m.clone()))
            .collect();
        let deepseek_ids: Vec<String> = deepseek_metadata.iter().map(|m| m.id.clone()).collect();
        let deepseek_lookup = move || async move { Ok::<_, anyhow::Error>(deepseek_ids) };

        let openrouter_by_id: HashMap<String, ModelMetadata> = openrouter_metadata
            .iter()
            .map(|m| (m.id.clone(), m.clone()))
            .collect();
        let openrouter_ids: Vec<String> =
            openrouter_metadata.iter().map(|m| m.id.clone()).collect();
        let openrouter_lookup = move || async move { Ok::<_, anyhow::Error>(openrouter_ids) };

        // Re-resolve ds4-server's port and (re)install its chat backend, so
        // discovery and `ds4::*` routing agree on the same port this round.
        let ds4_url = self.refresh_ds4_backend().await;

        let discovered: Vec<DiscoveredModel> = discover_all(
            &http,
            OLLAMA_DEFAULT_URL,
            ds4_url.as_deref(),
            bedrock_lookup,
            codex_lookup,
            deepseek_lookup,
            openrouter_lookup,
        )
        .await;
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
        Ok(discovered
            .into_iter()
            .map(|m| {
                let wire = m.wire_id();
                match m.source {
                    ModelSource::Bedrock => bedrock_by_id
                        .get(&m.id)
                        .map(|meta| ModelMetadata {
                            id: wire.clone(),
                            default_reasoning_level: meta.default_reasoning_level.clone(),
                            supported_reasoning_levels: meta.supported_reasoning_levels.clone(),
                            supports_images: meta.supports_images,
                            context_length: meta.context_length,
                            pricing: meta.pricing,
                        })
                        .unwrap_or_else(|| ModelMetadata::id_only(wire)),
                    ModelSource::Codex => codex_by_id
                        .get(&m.id)
                        .map(|meta| ModelMetadata {
                            id: wire.clone(),
                            default_reasoning_level: meta.default_reasoning_level.clone(),
                            supported_reasoning_levels: meta.supported_reasoning_levels.clone(),
                            supports_images: meta.supports_images,
                            context_length: meta.context_length,
                            pricing: meta.pricing,
                        })
                        .unwrap_or_else(|| ModelMetadata::id_only(wire)),
                    ModelSource::DeepSeek => deepseek_by_id
                        .get(&m.id)
                        .map(|meta| ModelMetadata {
                            id: wire.clone(),
                            default_reasoning_level: meta.default_reasoning_level.clone(),
                            supported_reasoning_levels: meta.supported_reasoning_levels.clone(),
                            supports_images: meta.supports_images,
                            context_length: meta.context_length,
                            pricing: meta.pricing,
                        })
                        .unwrap_or_else(|| ModelMetadata::id_only(wire)),
                    ModelSource::Ollama => ollama_metadata
                        .get(&m.id)
                        .map(|meta| ModelMetadata {
                            id: wire.clone(),
                            default_reasoning_level: meta.default_reasoning_level.clone(),
                            supported_reasoning_levels: meta.supported_reasoning_levels.clone(),
                            supports_images: meta.supports_images,
                            context_length: meta.context_length,
                            pricing: meta.pricing,
                        })
                        .unwrap_or_else(|| ModelMetadata::id_only(wire)),
                    // ds4-server's OpenAI shim only reports model ids (no
                    // capability/context metadata like Ollama's /api/show),
                    // so we expose ids alone and let the compression layer
                    // fall back to its default context window.
                    ModelSource::Ds4 => ModelMetadata::id_only(wire),
                    ModelSource::OpenRouter => openrouter_by_id
                        .get(&m.id)
                        .map(|meta| ModelMetadata {
                            id: wire.clone(),
                            default_reasoning_level: meta.default_reasoning_level.clone(),
                            supported_reasoning_levels: meta.supported_reasoning_levels.clone(),
                            supports_images: meta.supports_images,
                            context_length: meta.context_length,
                            pricing: meta.pricing,
                        })
                        .unwrap_or_else(|| ModelMetadata::id_only(wire)),
                }
            })
            .collect())
    }

    pub async fn list_model_metadata_with_progress(
        &self,
        progress: Option<UnboundedSender<String>>,
    ) -> Result<Vec<ModelMetadata>> {
        self.list_model_metadata_inner(progress).await
    }

    fn pick(&self, source: ModelSource) -> Option<Arc<dyn LlmBackend>> {
        match source {
            ModelSource::Bedrock => self.bedrock_snapshot(),
            ModelSource::Codex => self.codex_snapshot(),
            ModelSource::DeepSeek => self.deepseek_snapshot(),
            ModelSource::OpenRouter => self.openrouter_snapshot(),
            ModelSource::Ollama => self.ollama.clone(),
            ModelSource::Ds4 => self.ds4_snapshot(),
        }
    }

    /// Source to use when a chat request arrives with no `<source>::` prefix.
    /// Computed on demand (rather than cached at construction) so a Codex
    /// login or Bedrock key paste mid-session promotes it to the preferred
    /// fallback.
    ///
    /// Priority is Bedrock > Codex > Ollama > ds4 > DeepSeek > OpenRouter. Bedrock wins
    /// when configured because its model ids are otherwise easy to type
    /// bare from the environment-driven setup. ds4 sits after Ollama so an
    /// existing Ollama user's bare-id fallback is unchanged, and direct
    /// DeepSeek beats OpenRouter when both expose the same family.
    fn fallback_source(&self) -> Option<ModelSource> {
        let bedrock_present = self.bedrock.read().unwrap().is_some();
        if bedrock_present {
            return Some(ModelSource::Bedrock);
        }
        let codex_present = self.codex.read().unwrap().is_some();
        if codex_present {
            return Some(ModelSource::Codex);
        }
        if self.ollama.is_some() {
            return Some(ModelSource::Ollama);
        }
        if self.ds4.read().unwrap().is_some() {
            return Some(ModelSource::Ds4);
        }
        let deepseek_present = self.deepseek.read().unwrap().is_some();
        if deepseek_present {
            return Some(ModelSource::DeepSeek);
        }
        let openrouter_present = self.openrouter.read().unwrap().is_some();
        if openrouter_present {
            return Some(ModelSource::OpenRouter);
        }
        None
    }

    /// Resolve a wire-form model id to (backend, bare id). Bare ids (no
    /// `<source>::` prefix) route to the fallback source.
    fn resolve(&self, wire_model: &str) -> Result<(Arc<dyn LlmBackend>, String)> {
        if let Some((source, bare)) = split_wire_id(wire_model) {
            let backend = self.pick(source).ok_or_else(|| {
                anyhow::anyhow!(
                    "model {wire_model} requires the {} backend, which is not configured",
                    source.as_str()
                )
            })?;
            return Ok((backend, bare.to_string()));
        }
        let source = self.fallback_source().ok_or_else(|| {
            anyhow::anyhow!(
                "no LLM backend is configured (none of Bedrock, Codex, DeepSeek, OpenRouter, or Ollama \
                 discovered any models, and no `<source>::<id>` wire prefix was provided)"
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
            Ok((_backend, bare)) => {
                let provider = split_wire_id(configured_model)
                    .map(|(source, _)| source)
                    .or_else(|| self.fallback_source())
                    .map(|source| source.as_str().to_string());
                ResolvedModelInfo {
                    configured_model: configured_model.to_string(),
                    resolved_provider: provider,
                    resolved_model: bare,
                }
            }
            Err(_) => ResolvedModelInfo {
                configured_model: configured_model.to_string(),
                resolved_provider: split_wire_id(configured_model)
                    .map(|(source, _)| source.as_str().to_string()),
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
    label: &'static str,
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

    fn chat_request(model: &str, reasoning_effort: Option<&str>) -> StreamChatRequest {
        StreamChatRequest {
            model: model.to_string(),
            messages: vec![],
            tools: None,
            reasoning_effort: reasoning_effort.map(str::to_string),
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
    }

    impl LlmBackend for RecordingBackend {
        fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
            let name = self.name;
            async move { Ok(vec![format!("{name}-stub")]) }.boxed()
        }

        fn stream_chat(&self, request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
            *self.last_model.lock().unwrap() = Some(request.model);
            *self.last_reasoning_effort.lock().unwrap() = request.reasoning_effort;
            let response = LlmResponse::Text {
                text: format!("hello from {}", self.name),
                reasoning_content: None,
                usage: crate::llm_client::TokenUsage::default(),
            };
            async move { Ok(response) }.boxed()
        }
    }

    /// Captured per-call state for assertions.
    struct RecordingHandles {
        last_model: Arc<Mutex<Option<String>>>,
        last_reasoning_effort: Arc<Mutex<Option<String>>>,
    }

    fn recording(name: &'static str) -> (Arc<dyn LlmBackend>, RecordingHandles) {
        let last_model = Arc::new(Mutex::new(None));
        let last_reasoning_effort = Arc::new(Mutex::new(None));
        let backend = Arc::new(RecordingBackend {
            name,
            last_model: last_model.clone(),
            last_reasoning_effort: last_reasoning_effort.clone(),
        });
        (
            backend,
            RecordingHandles {
                last_model,
                last_reasoning_effort,
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
        let multi = MultiBackend::new(None, Some(codex_backend), None, None, Some(ollama_backend));

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
        let multi = MultiBackend::new(None, Some(codex_backend), None, None, Some(ollama_backend));

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
        let multi = MultiBackend::new(None, None, None, None, Some(ollama_backend));

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
        let multi = MultiBackend::new(None, None, None, None, Some(ollama_backend));

        let err = multi
            .stream_chat(chat_request("codex::gpt-5", None))
            .await
            .expect_err("codex route must fail when codex backend is absent");
        let msg = format!("{err:#}");
        assert!(msg.contains("codex"), "error must mention codex: {msg}");
    }

    /// When neither backend is configured, every chat request errors
    /// rather than panics. `MultiBackend` is constructible empty (the
    /// server still starts -- the user can run `/codex-login` mid-session
    /// or start Ollama and re-discover) but no model can be routed.
    #[tokio::test]
    async fn empty_multi_backend_errors_on_chat() {
        let multi = MultiBackend::new(None, None, None, None, None);
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

    /// Regression for the `/codex-login` lifecycle (issue #3555): the
    /// server starts with no Codex backend (auth.json absent), the user
    /// runs `/codex-login`, and the new backend is installed at
    /// runtime. Subsequent `codex::*` routing must succeed -- previously
    /// it kept returning the "backend not configured" error because the
    /// `None` was captured permanently at construction.
    #[tokio::test]
    async fn codex_installed_after_login_is_routable() {
        // Start with no Codex (mirrors the empty-auth.json startup path).
        let multi = MultiBackend::new(None, None, None, None, None);

        // Pre-install: a `codex::*` request must fail loudly.
        let err = multi
            .stream_chat(chat_request("codex::gpt-5-codex", None))
            .await
            .expect_err("codex route must fail before install");
        assert!(format!("{err:#}").contains("codex"));

        // User runs `/codex-login` successfully -- the handler installs
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
        let multi = MultiBackend::new(None, None, None, None, None);

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
    /// `/codex-login` followed by a discovery refresh (e.g. on
    /// `session/new`) would keep returning an empty Codex list and the
    /// model picker would never show Codex models.
    #[tokio::test]
    async fn list_models_reflects_installed_codex() {
        let multi = MultiBackend::new(None, None, None, None, None);
        let (codex_backend, _codex_handles) = recording("codex");
        multi.install_codex(codex_backend);

        // RecordingBackend::list_models returns ["codex-stub"].
        // `discover_all` delegates Codex discovery entirely to the
        // closure (the only Codex source it has -- there is no separate
        // native probe), so the "codex-stub" id surfacing through it
        // proves the freshly-installed backend was consulted by the
        // refresh path rather than the empty `None` captured at
        // construction.
        let models = multi.list_models().await.expect("discovery must succeed");
        assert!(
            models.iter().any(|m| m.contains("codex-stub")),
            "installed codex backend must contribute to discovery: got {models:?}"
        );
    }

    /// `/codex-login disconnect` calls `uninstall_codex` after wiping
    /// auth.json. Subsequent `codex::*` routing must fail with the same
    /// "backend not configured" error a fresh-no-auth.json startup
    /// gives -- otherwise a wire id picked from a stale `availableModels`
    /// list would fire a request against credentials that no longer
    /// exist on disk.
    #[tokio::test]
    async fn codex_uninstall_unroutes_codex_requests() {
        let multi = MultiBackend::new(None, None, None, None, None);
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
        let multi = MultiBackend::new(None, None, None, None, None);

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
        let multi = MultiBackend::new(None, Some(codex_backend), None, None, None);

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
        let multi = MultiBackend::new(None, None, None, None, None);
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
        let multi = MultiBackend::new(
            None,
            Some(codex_backend),
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
        let multi = MultiBackend::new(
            None,
            Some(codex_backend),
            Some(deepseek_backend),
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
    async fn bare_id_routes_to_deepseek_when_only_deepseek_configured() {
        let (deepseek_backend, deepseek_handles) = recording("deepseek");
        let multi = MultiBackend::new(None, None, Some(deepseek_backend), None, None);

        let _ = multi
            .stream_chat(chat_request("deepseek-v4-flash", None))
            .await
            .expect("bare id falls back to deepseek");
        assert_eq!(
            deepseek_handles.last_model.lock().unwrap().as_deref(),
            Some("deepseek-v4-flash")
        );
    }

    /// A bare id with only OpenRouter configured falls back to
    /// OpenRouter rather than erroring -- the same fallback contract
    /// Ollama gets when it's the only backend.
    #[tokio::test]
    async fn bare_id_routes_to_openrouter_when_only_openrouter_configured() {
        let (openrouter_backend, openrouter_handles) = recording("openrouter");
        let multi = MultiBackend::new(None, None, None, Some(openrouter_backend), None);

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
        let multi = MultiBackend::new(None, None, None, Some(openrouter_backend), None);

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
        let multi = MultiBackend::new(None, None, None, Some(openrouter_backend), None);

        let info = multi.resolve_model_info("gemini-3.1-pro-preview");

        assert_eq!(info.configured_model, "gemini-3.1-pro-preview");
        assert_eq!(info.resolved_provider.as_deref(), Some("openrouter"));
        assert_eq!(info.resolved_model, "gemini-3.1-pro-preview");
    }

    /// Fallback priority: Codex > Ollama > OpenRouter. With all three
    /// configured, a bare id routes to Codex (most capable, most likely
    /// intent). With Codex absent, Ollama wins over OpenRouter so a free
    /// local daemon beats a paid cloud router unless the user explicitly
    /// chooses an `openrouter::` model.
    #[tokio::test]
    async fn bare_id_prefers_ollama_over_openrouter_when_codex_absent() {
        let (openrouter_backend, openrouter_handles) = recording("openrouter");
        let (ollama_backend, ollama_handles) = recording("ollama");
        let multi = MultiBackend::new(
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
        let multi = MultiBackend::new(None, Some(codex_backend), None, None, None);

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
        let multi = MultiBackend::new(None, Some(codex_backend), None, None, None);

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

    #[tokio::test(start_paused = true)]
    async fn list_models_times_out_stuck_provider_and_keeps_healthy_ones() {
        let hanging: Arc<dyn LlmBackend> = Arc::new(HangingBackend);
        let (openrouter_backend, _openrouter_handles) = recording("openrouter");
        let multi = MultiBackend::new(None, Some(hanging), None, Some(openrouter_backend), None);

        let models = multi.list_models().await.expect("discovery must succeed");
        assert!(
            models.iter().any(|m| m == "openrouter::openrouter-stub"),
            "healthy provider should still contribute models: got {models:?}"
        );
    }
}
