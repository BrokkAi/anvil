//! Auto-discover available LLM models from Bedrock, Codex
//! (`~/.codex/auth.json`), a local Ollama daemon
//! (`http://localhost:11434/v1/models`), a local ds4-server
//! (antirez/ds4, an OpenAI-compatible DeepSeek V4 inference engine), and
//! OpenRouter (`https://openrouter.ai/api/v1/models`, gated on the
//! `OPENROUTER_API_KEY` env var), and hosted DeepSeek
//! (`https://api.deepseek.com/v1/models`, gated on `DEEPSEEK_API_KEY`).
//!
//! Zero-config by design: the Ollama URL is fixed at the daemon's default
//! port. If your daemon listens elsewhere, the catalog will simply not
//! include Ollama models -- run `ollama serve` on `:11434` to make them
//! discoverable. OpenRouter and hosted DeepSeek are enabled only when
//! their API keys are set in the environment; absent the key, the catalog
//! simply omits their models with no warning.
//!
//! ds4 is the one source whose port is *not* fixed: `ds4-server` has no
//! standard port, so instead of probing a constant we detect a running
//! `ds4-server` process and resolve the TCP port it is actually listening
//! on (see [`ds4_base_url`]). It is therefore discovered only when ds4 is
//! genuinely running -- the running process is the opt-in. ds4 targets
//! macOS (Metal) and Linux (CUDA/ROCm) only, so the process probe is
//! `cfg(unix)`; elsewhere only the `DS4_BASE_URL` env override is honored.
//!
//! Each discovered model carries a `ModelSource` tag so the routing backend
//! (`MultiBackend`) can pick the right HTTP client at request time. The
//! catalog is presented to ACP clients as `<source>::<id>` wire ids, e.g.
//! `codex::gpt-5-codex`, `ollama::llama3:latest`, `deepseek::deepseek-v4-pro`, and
//! `openrouter::anthropic/claude-3.5-sonnet`. The double-colon separator
//! avoids collision with Ollama tags (`model:tag`) and with OpenRouter
//! ids (`vendor/model`).
//!
//! Failure posture: missing or unreachable sources are logged and skipped,
//! never propagated. A user with none of the three configured still gets a
//! working server -- they just see an empty model picker until one of the
//! sources comes online (and can re-run discovery via `session/new`, which
//! refreshes the cache).

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use futures::future::join_all;
use serde::Deserialize;

use crate::llm_client::{ModelMetadata, ReasoningLevelPreset};

/// Where a discovered model came from. The variant is encoded into the
/// wire id so the routing backend doesn't need a per-request map lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelSource {
    Bedrock,
    Codex,
    DeepSeek,
    Ds4,
    Ollama,
    OpenRouter,
}

impl ModelSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Bedrock => "bedrock",
            Self::Codex => "codex",
            Self::DeepSeek => "deepseek",
            Self::Ds4 => "ds4",
            Self::Ollama => "ollama",
            Self::OpenRouter => "openrouter",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiscoveredModel {
    /// Bare model id as the upstream returns it (e.g. `gpt-5-codex`,
    /// `llama3:latest`). Pass this verbatim to the source's chat endpoint.
    pub id: String,
    pub source: ModelSource,
}

impl DiscoveredModel {
    /// Wire form: `<source>::<id>`. Round-trips through the ACP model picker.
    pub fn wire_id(&self) -> String {
        format!("{}::{}", self.source.as_str(), self.id)
    }
}

/// Parse a wire-form model id back into `(source, bare_id)`. Returns `None`
/// if the input is missing the `<source>::` prefix or the prefix is unknown,
/// in which case callers should treat the input as already-bare and pick a
/// default backend.
pub fn split_wire_id(wire: &str) -> Option<(ModelSource, &str)> {
    let (prefix, rest) = wire.split_once("::")?;
    let source = match prefix {
        "bedrock" => ModelSource::Bedrock,
        "codex" => ModelSource::Codex,
        "deepseek" => ModelSource::DeepSeek,
        "ds4" => ModelSource::Ds4,
        "ollama" => ModelSource::Ollama,
        "openrouter" => ModelSource::OpenRouter,
        _ => return None,
    };
    Some((source, rest))
}

/// Default Ollama base URL. Hardcoded by design: the whole point of the
/// zero-config posture is that the user doesn't pick a port -- if the
/// daemon isn't here, it's not discoverable.
pub const OLLAMA_DEFAULT_URL: &str = "http://localhost:11434";

/// OpenRouter cloud base URL. Hardcoded; OpenRouter is a single SaaS
/// endpoint with no self-hosted variant to point at. `OpenAiClient` will
/// append `/v1/...` when it sees this base URL (already covered by the
/// existing `api_url` logic).
pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Environment variable name carrying the OpenRouter API key. Mirrors the
/// `OPENROUTER_API_KEY` convention used by OpenRouter's own SDK docs --
/// brokk-acp-rust deliberately does NOT introduce a `BROKK_OPENROUTER_*`
/// alias so the same shell that already works with `openrouter` / OpenAI
/// SDK / litellm works here too.
pub const OPENROUTER_API_KEY_ENV: &str = "OPENROUTER_API_KEY";

/// Hosted DeepSeek cloud base URL. The API is OpenAI-compatible and uses
/// this origin for both chat and model discovery.
pub const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";

/// Environment variable name carrying the hosted DeepSeek API key.
pub const DEEPSEEK_API_KEY_ENV: &str = "DEEPSEEK_API_KEY";

// ---------------------------------------------------------------------------
// ds4 (antirez/ds4) discovery
// ---------------------------------------------------------------------------

/// Fallback ds4-server base URL. `ds4-server` binds `127.0.0.1:8000` unless
/// launched with a different `--host`; we probe this only when a running
/// `ds4-server` process is detected but its actual listening port could not
/// be resolved from the OS.
#[cfg(unix)]
pub const DS4_DEFAULT_URL: &str = "http://127.0.0.1:8000";

/// Env override for the ds4-server base URL. The escape hatch for setups the
/// process probe can't see: a non-default `--host`, a remote box, or a
/// reverse proxy (e.g. `DS4_BASE_URL=http://127.0.0.1:9000`). When set it
/// short-circuits process/port autodetection entirely. Mirrors the
/// `OPENROUTER_API_KEY` convention of using the upstream-style env name with
/// no `BROKK_`/`ANVIL_` prefix.
pub const DS4_BASE_URL_ENV: &str = "DS4_BASE_URL";

/// Resolve the base URL to probe for a local `ds4-server`, or `None` when
/// ds4 should be skipped this round.
///
/// Priority:
///   1. `DS4_BASE_URL` env override (trimmed, trailing slash removed). When
///      set we always probe it, even if no local process is visible -- this
///      covers remote/proxied ds4 the process probe can't detect.
///   2. A running `ds4-server` process whose listening TCP port we resolve
///      from the OS, yielding `http://127.0.0.1:<port>`.
///   3. A running `ds4-server` process whose port could not be resolved,
///      falling back to the documented default `DS4_DEFAULT_URL`.
///
/// Returns `None` when neither the env override nor a running process is
/// present, so discovery omits ds4 entirely -- the running process *is* the
/// opt-in. This is re-evaluated on every discovery refresh, so starting
/// ds4-server after Anvil (and then creating/refreshing a session) brings it
/// online on whatever port it ended up on.
pub fn ds4_base_url() -> Option<String> {
    if let Ok(url) = std::env::var(DS4_BASE_URL_ENV) {
        let url = url.trim();
        if !url.is_empty() {
            return Some(url.trim_end_matches('/').to_string());
        }
    }
    detect_ds4_server_url()
}

/// Detect a running `ds4-server` and the URL it serves on.
///
/// ds4 (antirez/ds4) ships for macOS (Metal) and Linux (CUDA/ROCm) only,
/// so process/port detection is Unix-only. We shell out to `pgrep`/`lsof`
/// rather than take a process-listing crate dependency: both ship by
/// default on the two supported platforms, the calls are short-lived, and
/// this keeps the dependency surface flat. Any failure (tool missing,
/// non-zero exit, unparsable output) degrades to `None`/default, never an
/// error.
#[cfg(unix)]
fn detect_ds4_server_url() -> Option<String> {
    let pid = find_ds4_server_pid()?;
    match ds4_listen_port_for_pid(pid) {
        Some(port) => Some(format!("http://127.0.0.1:{port}")),
        None => {
            tracing::info!(
                "ds4-server process {pid} detected but its listening port could not be \
                 resolved; falling back to {DS4_DEFAULT_URL}"
            );
            Some(DS4_DEFAULT_URL.to_string())
        }
    }
}

/// On non-Unix targets ds4 has no native build, so there is nothing to
/// detect: only the `DS4_BASE_URL` override (handled by [`ds4_base_url`])
/// can bring it online.
#[cfg(not(unix))]
fn detect_ds4_server_url() -> Option<String> {
    None
}

/// PID of a running `ds4-server`, via `pgrep -x ds4-server`. `-x` matches
/// the process *name* exactly, so it ignores our own discovery commands and
/// unrelated processes that merely mention "ds4-server" on their command
/// line. Returns the first match; ds4 may fork workers but they all listen
/// behind the same accept socket, so any one PID resolves the same port.
#[cfg(unix)]
fn find_ds4_server_pid() -> Option<u32> {
    let output = std::process::Command::new("pgrep")
        .args(["-x", "ds4-server"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.split_whitespace().next()?.parse::<u32>().ok()
}

/// Listening TCP port for `pid`, via
/// `lsof -nP -iTCP -sTCP:LISTEN -a -p <pid>`. `-n`/`-P` skip DNS/port-name
/// lookups (faster, and keeps the NAME column numeric for parsing).
#[cfg(unix)]
fn ds4_listen_port_for_pid(pid: u32) -> Option<u16> {
    let output = std::process::Command::new("lsof")
        .args(["-nP", "-iTCP", "-sTCP:LISTEN", "-a", "-p", &pid.to_string()])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_lsof_listen_port(&stdout)
}

/// Extract the first listening port from `lsof` output. Each listen line
/// ends with the NAME column followed by `(LISTEN)`, e.g.
/// `... TCP 127.0.0.1:8000 (LISTEN)` or `... TCP [::1]:8000 (LISTEN)`; the
/// port is the segment after the final `:` of the address token preceding
/// `(LISTEN)`. Returns `None` if no listen line parses.
#[cfg(unix)]
fn parse_lsof_listen_port(output: &str) -> Option<u16> {
    for line in output.lines() {
        let mut prev: Option<&str> = None;
        for tok in line.split_whitespace() {
            if tok == "(LISTEN)"
                && let Some(port) = prev.and_then(|addr| addr.rsplit(':').next())
                && let Ok(port) = port.parse::<u16>()
            {
                return Some(port);
            }
            prev = Some(tok);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Ollama discovery (OpenAI-compatible /v1/models)
// ---------------------------------------------------------------------------

/// Build a short-timeout HTTP client tuned for discovery. Ollama is local;
/// if it's not running we want to fail fast, not block startup for 30s.
pub fn discovery_http_client() -> reqwest::Client {
    crate::llm_client::OpenAiClient::apply_runtime_tls_workarounds(
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .timeout(Duration::from_secs(5)),
        OLLAMA_DEFAULT_URL,
    )
    .build()
    .expect("failed to build discovery HTTP client")
}

/// Query Ollama's OpenAI-compatible `/v1/models` endpoint. Recent Ollama
/// versions preserve tag suffixes (`llama3:latest`) in the OpenAI shim,
/// so we use it directly instead of the native `/api/tags`. This also
/// keeps the parsing identical to any other OpenAI-compatible local
/// server a user might run on the same port.
pub async fn discover_ollama(
    http: &reqwest::Client,
    base_url: &str,
) -> Result<Vec<DiscoveredModel>> {
    discover_ollama_models(http, base_url).await
}

async fn discover_ollama_models(
    http: &reqwest::Client,
    base_url: &str,
) -> Result<Vec<DiscoveredModel>> {
    discover_openai_compatible_models(http, base_url, ModelSource::Ollama).await
}

/// Discover models from a local `ds4-server` (antirez/ds4). ds4 speaks the
/// same OpenAI-compatible `/v1/models` shape as Ollama, so this is the same
/// probe tagged with [`ModelSource::Ds4`]. `base_url` comes from
/// [`ds4_base_url`], which already resolved the running server's port.
pub async fn discover_ds4(http: &reqwest::Client, base_url: &str) -> Result<Vec<DiscoveredModel>> {
    discover_openai_compatible_models(http, base_url, ModelSource::Ds4).await
}

/// Shared `/v1/models` probe for any OpenAI-compatible local server. Tags
/// every returned id with `source` so the catalog routes it back to the
/// right backend.
async fn discover_openai_compatible_models(
    http: &reqwest::Client,
    base_url: &str,
    source: ModelSource,
) -> Result<Vec<DiscoveredModel>> {
    let base = base_url.trim_end_matches('/');
    let url = format!("{base}/v1/models");
    let resp = http
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("{} /v1/models returned HTTP {status}", source.as_str());
    }
    let parsed: crate::llm_client::ModelsResponse =
        resp.json().await.context("parsing /v1/models JSON")?;
    Ok(parsed
        .data
        .into_iter()
        .map(|m| DiscoveredModel { id: m.id, source })
        .collect())
}

#[derive(Debug, Deserialize)]
struct OllamaShowResponse {
    #[serde(default)]
    capabilities: Vec<String>,
}

fn ollama_thinking_presets() -> Vec<ReasoningLevelPreset> {
    [
        (
            "none",
            "Disable the reasoning trace while keeping the model available.",
        ),
        ("low", "Light reasoning for shorter problems."),
        ("medium", "Balanced reasoning for moderate complexity."),
        ("high", "Deep reasoning for harder problems."),
    ]
    .into_iter()
    .map(|(effort, description)| ReasoningLevelPreset {
        effort: effort.to_string(),
        description: description.to_string(),
    })
    .collect()
}

/// Discover Ollama models that advertise the `thinking` capability and
/// surface the reasoning presets the session UI should show for them.
///
/// The list endpoint gives us the installed model ids. `POST /api/show`
/// tells us whether a given model supports thinking, which is the signal
/// we need to decide whether to expose reasoning controls.
pub async fn discover_ollama_model_metadata(
    http: &reqwest::Client,
    base_url: &str,
) -> HashMap<String, ModelMetadata> {
    let models = match discover_ollama_models(http, base_url).await {
        Ok(models) => models,
        Err(e) => {
            tracing::info!("ollama model metadata discovery skipped at {base_url}: {e:#}");
            return HashMap::new();
        }
    };

    let base = base_url.trim_end_matches('/');
    let show_url = format!("{base}/api/show");
    let futures = models.into_iter().map(|model| {
        let http = http.clone();
        let show_url = show_url.clone();
        async move {
            let model_id = model.id;
            let body = serde_json::json!({
                "model": model_id,
                "verbose": false,
            });

            let resp = match http.post(&show_url).json(&body).send().await {
                Ok(resp) => resp,
                Err(e) => {
                    tracing::info!(
                        "ollama model details skipped for {model_id} at {show_url}: {e:#}"
                    );
                    return Some((model_id.clone(), ModelMetadata::id_only(model_id)));
                }
            };

            let status = resp.status();
            if !status.is_success() {
                let body_text = resp.text().await.unwrap_or_default();
                tracing::info!(
                    "ollama model details skipped for {model_id} at {show_url} (HTTP {status}): {body_text}"
                );
                return Some((model_id.clone(), ModelMetadata::id_only(model_id)));
            }

            let parsed: OllamaShowResponse = match resp.json().await {
                Ok(parsed) => parsed,
                Err(e) => {
                    tracing::info!(
                        "ollama model details skipped for {model_id} at {show_url}: {e:#}"
                    );
                    return Some((model_id.clone(), ModelMetadata::id_only(model_id)));
                }
            };

            if parsed.capabilities.iter().any(|cap| cap == "thinking") {
                return Some((
                    model_id.clone(),
                    ModelMetadata {
                        id: model_id,
                        default_reasoning_level: None,
                        supported_reasoning_levels: ollama_thinking_presets(),
                        service_tiers: Vec::new(),
                        supports_images: Some(
                            parsed.capabilities.iter().any(|cap| cap == "vision"),
                        ),
                        // Ollama's catalog doesn't publish a context window;
                        // the compression layer falls back to a default.
                        context_length: None,
                        pricing: None,
                    },
                ));
            }

            Some((
                model_id.clone(),
                ModelMetadata {
                    id: model_id,
                    default_reasoning_level: None,
                    supported_reasoning_levels: Vec::new(),
                    service_tiers: Vec::new(),
                    supports_images: Some(parsed.capabilities.iter().any(|cap| cap == "vision")),
                    context_length: None,
                    pricing: None,
                },
            ))
        }
    });

    join_all(futures).await.into_iter().flatten().collect()
}

// ---------------------------------------------------------------------------
// Top-level orchestrator
// ---------------------------------------------------------------------------

/// Run all configured discovery sources concurrently and merge results.
/// Failures are logged and treated as "no models from this source" so a
/// dead Ollama daemon doesn't shadow a working Codex login (or vice versa).
///
/// `ollama_url` is a parameter purely for test injection -- production
/// always passes `OLLAMA_DEFAULT_URL`. There's no CLI flag for this; the
/// zero-config posture means the user doesn't pick a port.
///
/// Codex, hosted DeepSeek, and OpenRouter discovery are delegated to caller-provided
/// closures. For Codex, the `CodexClient` lives in a sibling module that
/// already handles auth.json parsing, fallback slugs, and
/// `chatgpt`/`apikey` mode branching. For the hosted providers, each
/// closure lets the caller short-circuit cleanly when the matching API
/// key is absent without `discovery.rs` having to know about env vars.
/// Each closure keeps `discovery.rs` agnostic of those concerns while
/// still running all sources in parallel.
///
/// `ds4_url` is `Some` only when a local `ds4-server` was detected (or
/// `DS4_BASE_URL` is set); `None` skips ds4 entirely. See [`ds4_base_url`].
///
/// Output ordering is fixed: Bedrock, Codex, Ollama, ds4, hosted DeepSeek,
/// then OpenRouter.
/// The first model in the merged list is auto-selected as the session
/// default elsewhere -- keeping ordering deterministic stops users from
/// seeing a different default just because one source happened to be
/// slow on a given boot. ds4 sits after Ollama so an existing Ollama
/// user's default is unchanged. Hosted DeepSeek sits before OpenRouter so
/// a direct vendor backend wins over the aggregator when both expose the
/// same family of models.
pub async fn discover_all<FB, FutB, FC, FutC, FDS, FutDS, FOR, FutOR>(
    http: &reqwest::Client,
    ollama_url: &str,
    ds4_url: Option<&str>,
    bedrock_lookup: FB,
    codex_lookup: FC,
    deepseek_lookup: FDS,
    openrouter_lookup: FOR,
) -> Vec<DiscoveredModel>
where
    FB: FnOnce() -> FutB,
    FutB: std::future::Future<Output = Result<Vec<String>>>,
    FC: FnOnce() -> FutC,
    FutC: std::future::Future<Output = Result<Vec<String>>>,
    FDS: FnOnce() -> FutDS,
    FutDS: std::future::Future<Output = Result<Vec<String>>>,
    FOR: FnOnce() -> FutOR,
    FutOR: std::future::Future<Output = Result<Vec<String>>>,
{
    let bedrock_fut = async {
        match bedrock_lookup().await {
            Ok(ids) => ids
                .into_iter()
                .map(|id| DiscoveredModel {
                    id,
                    source: ModelSource::Bedrock,
                })
                .collect(),
            Err(e) => {
                tracing::info!("bedrock model discovery skipped: {e:#}");
                Vec::new()
            }
        }
    };

    let codex_fut = async {
        match codex_lookup().await {
            Ok(ids) => ids
                .into_iter()
                .map(|id| DiscoveredModel {
                    id,
                    source: ModelSource::Codex,
                })
                .collect(),
            Err(e) => {
                tracing::info!("codex model discovery skipped: {e:#}");
                Vec::new()
            }
        }
    };

    let deepseek_fut = async {
        match deepseek_lookup().await {
            Ok(ids) => ids
                .into_iter()
                .map(|id| DiscoveredModel {
                    id,
                    source: ModelSource::DeepSeek,
                })
                .collect(),
            Err(e) => {
                tracing::info!("deepseek model discovery skipped: {e:#}");
                Vec::new()
            }
        }
    };

    let openrouter_fut = async {
        match openrouter_lookup().await {
            Ok(ids) => ids
                .into_iter()
                .map(|id| DiscoveredModel {
                    id,
                    source: ModelSource::OpenRouter,
                })
                .collect(),
            Err(e) => {
                tracing::info!("openrouter model discovery skipped: {e:#}");
                Vec::new()
            }
        }
    };

    let ollama_fut = async {
        match discover_ollama(http, ollama_url).await {
            Ok(models) => models,
            Err(e) => {
                tracing::info!("ollama model discovery skipped at {ollama_url}: {e:#}");
                Vec::new()
            }
        }
    };

    let ds4_fut = async {
        match ds4_url {
            Some(url) => match discover_ds4(http, url).await {
                Ok(models) => models,
                Err(e) => {
                    tracing::info!("ds4 model discovery skipped at {url}: {e:#}");
                    Vec::new()
                }
            },
            None => Vec::new(),
        }
    };

    let (bedrock, codex, deepseek, openrouter, ollama, ds4) = tokio::join!(
        bedrock_fut,
        codex_fut,
        deepseek_fut,
        openrouter_fut,
        ollama_fut,
        ds4_fut
    );
    let mut all = Vec::with_capacity(
        bedrock.len() + codex.len() + ollama.len() + ds4.len() + deepseek.len() + openrouter.len(),
    );
    all.extend(bedrock);
    all.extend(codex);
    all.extend(ollama);
    all.extend(ds4);
    all.extend(deepseek);
    all.extend(openrouter);
    all
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// `discover_ollama` parses the OpenAI-shape `/v1/models` body and
    /// preserves tag suffixes (`llama3:latest`) verbatim through to
    /// `DiscoveredModel.id`. Without this test, the existing
    /// `discover_all_*` cases only hit the connect-refused error path
    /// via `TEST_DEAD_OLLAMA_URL`, so the JSON parsing path was never
    /// exercised.
    #[tokio::test]
    async fn discover_ollama_parses_v1_models_shape() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [
                    {"id": "llama3:latest", "object": "model"},
                    {"id": "qwen3.6:35b-a3b-coding-mxfp8", "object": "model"}
                ]
            })))
            .mount(&server)
            .await;

        let http = discovery_http_client();
        let models = discover_ollama(&http, &server.uri()).await.expect("ok");
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "llama3:latest");
        assert_eq!(models[0].source, ModelSource::Ollama);
        assert_eq!(models[1].id, "qwen3.6:35b-a3b-coding-mxfp8");
    }

    /// `discover_ollama_model_metadata` marks only models whose
    /// `POST /api/show` response advertises `thinking` and leaves the
    /// rest as plain `id_only` metadata.
    #[tokio::test]
    async fn discover_ollama_model_metadata_marks_thinking_models() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [
                    {"id": "gemma4:26b", "object": "model"},
                    {"id": "qwen3.5:4b", "object": "model"}
                ]
            })))
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/api/show"))
            .and(body_json(serde_json::json!({
                "model": "gemma4:26b",
                "verbose": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "capabilities": ["completion", "vision", "tools", "thinking"]
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/show"))
            .and(body_json(serde_json::json!({
                "model": "qwen3.5:4b",
                "verbose": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "capabilities": ["completion"]
            })))
            .mount(&server)
            .await;

        let http = discovery_http_client();
        let metadata = discover_ollama_model_metadata(&http, &server.uri()).await;
        let thinking = metadata.get("gemma4:26b").expect("thinking model metadata");
        assert!(thinking.default_reasoning_level.is_none());
        assert_eq!(thinking.supports_images, Some(true));
        assert_eq!(
            thinking
                .supported_reasoning_levels
                .iter()
                .map(|preset| preset.effort.as_str())
                .collect::<Vec<_>>(),
            vec!["none", "low", "medium", "high"]
        );

        let plain = metadata.get("qwen3.5:4b").expect("plain model metadata");
        assert!(plain.default_reasoning_level.is_none());
        assert!(plain.supported_reasoning_levels.is_empty());
        assert_eq!(plain.supports_images, Some(false));
    }

    /// `wire_id` round-trips through `split_wire_id` for every source,
    /// preserves Ollama tag suffixes (the inner colon in `llama3:latest`),
    /// and preserves OpenRouter vendor-prefixed ids (the inner slash in
    /// `anthropic/claude-3.5-sonnet`).
    #[test]
    fn wire_id_round_trips_for_every_source() {
        let bedrock = DiscoveredModel {
            id: "us.anthropic.claude-sonnet-4-6".into(),
            source: ModelSource::Bedrock,
        };
        let bedrock_wire = bedrock.wire_id();
        assert_eq!(bedrock_wire, "bedrock::us.anthropic.claude-sonnet-4-6");
        let (src, id) = split_wire_id(&bedrock_wire).expect("must parse");
        assert_eq!(src, ModelSource::Bedrock);
        assert_eq!(id, "us.anthropic.claude-sonnet-4-6");

        let codex = DiscoveredModel {
            id: "gpt-5-codex".into(),
            source: ModelSource::Codex,
        };
        let codex_wire = codex.wire_id();
        assert_eq!(codex_wire, "codex::gpt-5-codex");
        let (src, id) = split_wire_id(&codex_wire).expect("must parse");
        assert_eq!(src, ModelSource::Codex);
        assert_eq!(id, "gpt-5-codex");

        let deepseek = DiscoveredModel {
            id: "deepseek-v4-pro".into(),
            source: ModelSource::DeepSeek,
        };
        let deepseek_wire = deepseek.wire_id();
        assert_eq!(deepseek_wire, "deepseek::deepseek-v4-pro");
        let (src, id) = split_wire_id(&deepseek_wire).expect("must parse");
        assert_eq!(src, ModelSource::DeepSeek);
        assert_eq!(id, "deepseek-v4-pro");

        let ollama = DiscoveredModel {
            id: "llama3:latest".into(),
            source: ModelSource::Ollama,
        };
        let ollama_wire = ollama.wire_id();
        assert_eq!(ollama_wire, "ollama::llama3:latest");
        let (src, id) = split_wire_id(&ollama_wire).expect("must parse");
        assert_eq!(src, ModelSource::Ollama);
        // The inner `:` survives -- our separator is the double-colon, so
        // the bare id keeps its tag suffix and routes correctly to Ollama.
        assert_eq!(id, "llama3:latest");

        let ds4 = DiscoveredModel {
            id: "deepseek-v4-flash".into(),
            source: ModelSource::Ds4,
        };
        let ds4_wire = ds4.wire_id();
        assert_eq!(ds4_wire, "ds4::deepseek-v4-flash");
        let (src, id) = split_wire_id(&ds4_wire).expect("must parse");
        assert_eq!(src, ModelSource::Ds4);
        assert_eq!(id, "deepseek-v4-flash");

        let openrouter = DiscoveredModel {
            id: "anthropic/claude-3.5-sonnet".into(),
            source: ModelSource::OpenRouter,
        };
        let openrouter_wire = openrouter.wire_id();
        assert_eq!(openrouter_wire, "openrouter::anthropic/claude-3.5-sonnet");
        let (src, id) = split_wire_id(&openrouter_wire).expect("must parse");
        assert_eq!(src, ModelSource::OpenRouter);
        // The inner `/` survives -- OpenRouter's `vendor/model` format
        // round-trips through the wire id without escaping.
        assert_eq!(id, "anthropic/claude-3.5-sonnet");
    }

    /// Wire ids without the `::` separator or with an unknown source are
    /// rejected so callers can treat them as already-bare and route to a
    /// default backend rather than silently mis-tagging them.
    #[test]
    fn split_wire_id_rejects_bare_or_unknown_prefix() {
        assert!(split_wire_id("gpt-5-codex").is_none());
        assert!(split_wire_id("llama3:latest").is_none());
        assert!(split_wire_id("anthropic/claude-3.5-sonnet").is_none());
        assert!(split_wire_id("unknown::foo").is_none());
        // Single colon is not enough -- `llama3:latest` is a bare Ollama id,
        // not a wire id, so we must NOT treat it as `source=llama3`.
        assert!(split_wire_id("llama3:latest").is_none());
    }

    /// Closed-port URL used by tests to force `discover_ollama` to fail
    /// fast via `connect_timeout`, regardless of whether the test host
    /// happens to have a real Ollama running on the default port.
    const TEST_DEAD_OLLAMA_URL: &str = "http://127.0.0.1:1";

    /// `discover_all` returns Bedrock first, then Codex, Ollama, ds4,
    /// DeepSeek, then OpenRouter,
    /// regardless of which future resolves first. Stable ordering matters
    /// because the first model in the catalog is auto-selected as the
    /// session default; OpenRouter stays last because it is the explicit
    /// aggregator choice rather than the "choose for me" default.
    #[tokio::test]
    async fn discover_all_orders_codex_ollama_openrouter() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [
                    {"id": "llama3:latest", "object": "model"}
                ]
            })))
            .mount(&server)
            .await;

        let http = discovery_http_client();
        // Point ds4 at the same mock /v1/models so it also yields one model;
        // this exercises the fixed ordering with every source populated.
        let models = discover_all(
            &http,
            &server.uri(),
            Some(&server.uri()),
            || async { Ok(vec!["us.anthropic.claude-sonnet-4-6".to_string()]) },
            || async { Ok(vec!["gpt-5-codex".to_string(), "gpt-4o".to_string()]) },
            || async { Ok(vec!["deepseek-v4-pro".to_string()]) },
            || async {
                Ok(vec![
                    "anthropic/claude-3.5-sonnet".to_string(),
                    "openai/gpt-4o".to_string(),
                ])
            },
        )
        .await;
        // Bedrock 1; Codex 2; Ollama 1; ds4 1; DeepSeek 1; OpenRouter 2.
        assert_eq!(models.len(), 8);
        assert_eq!(models[0].source, ModelSource::Bedrock);
        assert_eq!(models[0].id, "us.anthropic.claude-sonnet-4-6");
        assert_eq!(models[1].source, ModelSource::Codex);
        assert_eq!(models[1].id, "gpt-5-codex");
        assert_eq!(models[2].source, ModelSource::Codex);
        assert_eq!(models[3].source, ModelSource::Ollama);
        assert_eq!(models[3].id, "llama3:latest");
        assert_eq!(models[4].source, ModelSource::Ds4);
        assert_eq!(models[4].id, "llama3:latest");
        assert_eq!(models[5].source, ModelSource::DeepSeek);
        assert_eq!(models[5].id, "deepseek-v4-pro");
        assert_eq!(models[6].source, ModelSource::OpenRouter);
        assert_eq!(models[6].id, "anthropic/claude-3.5-sonnet");
        assert_eq!(models[7].source, ModelSource::OpenRouter);
    }

    /// When every source fails, the merged vec is empty rather than an
    /// error -- the server still starts and the user can run /setup codex,
    /// start Ollama, or export `OPENROUTER_API_KEY`, then re-run discovery
    /// via the next session/new.
    #[tokio::test]
    async fn discover_all_returns_empty_when_every_source_fails() {
        let http = discovery_http_client();
        let models = discover_all(
            &http,
            TEST_DEAD_OLLAMA_URL,
            None,
            || async { anyhow::bail!("no Bedrock token") },
            || async { anyhow::bail!("no auth.json") },
            || async { anyhow::bail!("no DEEPSEEK_API_KEY") },
            || async { anyhow::bail!("no OPENROUTER_API_KEY") },
        )
        .await;
        assert!(models.is_empty());
    }

    /// OpenRouter discovery is treated independently from Codex: a Codex
    /// failure (no auth.json) must not suppress OpenRouter results, and
    /// vice versa. Regression for the "missing source shadows working
    /// source" failure mode the orchestrator is specifically designed to
    /// prevent.
    #[tokio::test]
    async fn discover_all_keeps_openrouter_when_codex_fails() {
        let http = discovery_http_client();
        let models = discover_all(
            &http,
            TEST_DEAD_OLLAMA_URL,
            None,
            || async { anyhow::bail!("no Bedrock token") },
            || async { anyhow::bail!("no auth.json") },
            || async { anyhow::bail!("no DEEPSEEK_API_KEY") },
            || async { Ok(vec!["anthropic/claude-3.5-sonnet".to_string()]) },
        )
        .await;
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].source, ModelSource::OpenRouter);
        assert_eq!(models[0].id, "anthropic/claude-3.5-sonnet");
    }

    /// `discover_ds4` hits the same OpenAI-compatible `/v1/models` shape as
    /// Ollama but tags the results [`ModelSource::Ds4`]. ds4 reports
    /// `deepseek-v4-flash` / `deepseek-v4-pro`, so verify both come back
    /// verbatim and routed to the ds4 source.
    #[tokio::test]
    async fn discover_ds4_parses_v1_models_shape() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [
                    {"id": "deepseek-v4-flash", "object": "model"},
                    {"id": "deepseek-v4-pro", "object": "model"}
                ]
            })))
            .mount(&server)
            .await;

        let http = discovery_http_client();
        let models = discover_ds4(&http, &server.uri()).await.expect("ok");
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "deepseek-v4-flash");
        assert_eq!(models[0].source, ModelSource::Ds4);
        assert_eq!(models[1].id, "deepseek-v4-pro");
    }

    /// `parse_lsof_listen_port` pulls the port out of the first `(LISTEN)`
    /// line, coping with the IPv4/IPv6 address forms `lsof -nP` emits and
    /// ignoring the header and any non-listen rows.
    #[cfg(unix)]
    #[test]
    fn parse_lsof_listen_port_extracts_first_listen_port() {
        let ipv4 = "COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME\n\
                    ds4-serve 123 user    6u  IPv4 0x1234      0t0  TCP 127.0.0.1:8000 (LISTEN)\n";
        assert_eq!(parse_lsof_listen_port(ipv4), Some(8000));

        let ipv6 = "ds4-serve 123 user    7u  IPv6 0x9abc      0t0  TCP [::1]:9000 (LISTEN)\n";
        assert_eq!(parse_lsof_listen_port(ipv6), Some(9000));

        // Wildcard bind form.
        let wildcard = "ds4-serve 123 user    8u  IPv4 0x1111      0t0  TCP *:7070 (LISTEN)\n";
        assert_eq!(parse_lsof_listen_port(wildcard), Some(7070));

        // No listen line -> None.
        let established = "ds4-serve 123 user    9u  IPv4 0x2222      0t0  TCP 127.0.0.1:55000->1.2.3.4:443 (ESTABLISHED)\n";
        assert_eq!(parse_lsof_listen_port(established), None);

        assert_eq!(parse_lsof_listen_port(""), None);
    }
}
