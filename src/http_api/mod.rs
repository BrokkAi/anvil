//! HTTP daemon mode (`anvil serve`) exposing the Anvil runtime as a
//! versioned REST API (#317).
//!
//! The daemon serves the same `SessionStore` used by the ACP adapter, so
//! HTTP-created sessions persist, reload, and configure exactly like ACP
//! sessions. Prompt runs, SSE event streaming, and interactive permissions
//! land in follow-up issues (#318, #319); this module covers readiness,
//! model/tool discovery, and the full session lifecycle:
//!
//! | Method | Path                          | Purpose                              |
//! |--------|-------------------------------|--------------------------------------|
//! | GET    | `/health`                     | Liveness + version + discovery state |
//! | GET    | `/v1/models`                  | Model catalog (`?refresh=true`)      |
//! | GET    | `/v1/tools`                   | Static harness tool catalog          |
//! | GET    | `/v1/sessions`                | Resident sessions (`?cwd=` for disk) |
//! | POST   | `/v1/sessions`                | Create + configure a session         |
//! | GET    | `/v1/sessions/{id}`           | Inspect (`?include_history=true`)    |
//! | PATCH  | `/v1/sessions/{id}`           | Reconfigure (model, modes, effort)   |
//! | DELETE | `/v1/sessions/{id}`           | Delete (idempotent)                  |
//! | POST   | `/v1/sessions/{id}/load`      | Reopen with history in the response  |
//! | POST   | `/v1/sessions/{id}/resume`    | Reopen without history               |
//! | POST   | `/v1/sessions/{id}/runs`      | Start an asynchronous prompt run     |
//! | GET    | `/v1/sessions/{id}/runs`      | List this session's runs             |
//! | GET    | `/v1/runs/{id}`               | Poll run state and result            |
//! | GET    | `/v1/runs/{id}/events`        | SSE event stream (`Last-Event-ID`)   |
//! | POST   | `/v1/runs/{id}/cancel`        | Cancel the active turn (idempotent)  |
//! | GET    | `/v1/runs/{id}/permissions`   | Pending interactive permissions      |
//! | GET    | `/v1/permissions/{id}`        | Inspect one permission request       |
//! | POST   | `/v1/permissions/{id}/respond`| Approve / reject / cancel it         |
//!
//! Error responses share one envelope: `{"error": {"code", "message",
//! "details"?}, "request_id"}` with the same id echoed in `x-request-id`.
//!
//! Security posture (#319): the listener is a remote-execution boundary
//! (sessions drive filesystem and shell tooling). Binding defaults to
//! loopback with optional bearer-token auth; non-loopback binding requires
//! a configured token. `--workspace-root` restricts session paths,
//! `bypassPermissions` is refused unless `--allow-bypass-permissions` is
//! set, and auth failures plus permission decisions are written to the
//! `audit` log target. Logs go to stderr; stdout carries exactly one
//! machine-readable `serve.ready` line.

use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Path, Query, Request, State};
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::acp::{
    BEHAVIOR_CONFIG_ID, ConfigApplyError, MODEL_CONFIG_ID, PERMISSION_CONFIG_ID,
    REASONING_EFFORT_CONFIG_ID, SERVICE_TIER_CONFIG_ID, apply_config_option,
    seed_default_model_if_empty,
};
use crate::llm_client::ModelMetadata;
use crate::mcp::McpServerConfig;
use crate::multi_backend::MultiBackend;
use crate::session::{LifecycleReopen, Session, SessionStore, validate_additional_directories};

#[derive(clap::Args, Debug)]
pub(crate) struct ServeArgs {
    /// Address to bind the HTTP listener to. Only loopback addresses are
    /// accepted until authenticated network access lands (#319).
    #[arg(long, default_value = "127.0.0.1")]
    pub(crate) host: String,

    /// Port to bind. `0` picks an ephemeral port; the bound address is
    /// reported on the stdout `serve.ready` line either way.
    /// The default, 26845, spells ANVIL on a phone keypad.
    #[arg(long, default_value_t = 26845)]
    pub(crate) port: u16,

    /// Bearer token required on every `/v1` endpoint (`Authorization:
    /// Bearer <token>`). `/health` stays open for liveness checks.
    /// Required to bind a non-loopback address.
    #[arg(long, env = "ANVIL_HTTP_TOKEN", hide_env_values = true)]
    pub(crate) auth_token: Option<String>,

    /// Read the bearer token from a file (surrounding whitespace trimmed).
    /// Preferred over --auth-token, which is visible in process listings.
    #[arg(long, conflicts_with = "auth_token")]
    pub(crate) auth_token_file: Option<PathBuf>,

    /// Generate a random bearer token at startup and report it on the
    /// stdout `serve.ready` line as `auth_token`.
    #[arg(long, conflicts_with_all = ["auth_token", "auth_token_file"])]
    pub(crate) generate_auth_token: bool,

    /// Restrict session working directories (and additional directories)
    /// to descendants of these roots. Repeatable. Without it, sessions may
    /// use any absolute path the daemon's user can access.
    #[arg(long = "workspace-root")]
    pub(crate) workspace_roots: Vec<PathBuf>,

    /// Allow HTTP clients to select `permission_mode:
    /// "bypassPermissions"`. Off by default: the bypass mode disables the
    /// permission gate entirely, which is not something a remote caller
    /// should be able to request unless the operator opted in.
    #[arg(long, default_value_t = false)]
    pub(crate) allow_bypass_permissions: bool,
}

/// Run the HTTP daemon until interrupted. Performs the same eager model
/// discovery the ACP `initialize` handler does so `/v1/models` and
/// session-creation model validation work immediately after readiness.
pub(crate) async fn serve(
    args: ServeArgs,
    llm: Arc<MultiBackend>,
    sessions: SessionStore,
    max_turns: usize,
    default_idle_timeout_secs: u64,
    default_stall_timeout_secs: u64,
) -> Result<()> {
    // Validate the security configuration before the (potentially slow)
    // provider discovery so a misconfigured daemon fails fast.
    let ip: IpAddr = if args.host == "localhost" {
        IpAddr::from([127, 0, 0, 1])
    } else {
        args.host
            .parse()
            .with_context(|| format!("invalid --host address '{}'", args.host))?
    };

    let (auth, generated_token) = resolve_auth(&args)?;
    if !ip.is_loopback() && auth.is_none() {
        bail!(
            "refusing to bind non-loopback address {ip} without authentication: the HTTP API \
             exposes model-driven filesystem and shell tooling. Pass --auth-token, \
             --auth-token-file, or --generate-auth-token (or set ANVIL_HTTP_TOKEN) to enable \
             authenticated network access."
        );
    }

    let mut workspace_roots = Vec::with_capacity(args.workspace_roots.len());
    for root in &args.workspace_roots {
        let canonical = root.canonicalize().with_context(|| {
            format!(
                "--workspace-root '{}' must be an existing directory",
                root.display()
            )
        })?;
        workspace_roots.push(canonical);
    }

    match llm.list_model_metadata_with_progress(None).await {
        Ok(models) => {
            tracing::info!("serve startup discovery: {} model(s) found", models.len());
            seed_default_model_if_empty(&sessions, &models).await;
            sessions.set_available_models(models).await;
        }
        Err(e) => tracing::warn!("serve startup discovery failed: {e:#}"),
    }

    let state = ApiState {
        sessions,
        llm,
        refresh_lock: Arc::new(tokio::sync::Mutex::new(())),
        runs: Arc::new(runs::RunManager::default()),
        permissions: Arc::new(permissions::PermissionRegistry::default()),
        auth: auth.map(Arc::new),
        workspace_roots: Arc::new(workspace_roots),
        allow_bypass_permissions: args.allow_bypass_permissions,
        max_turns,
        default_idle_timeout_secs,
        default_stall_timeout_secs,
    };
    let listener = tokio::net::TcpListener::bind((ip, args.port))
        .await
        .with_context(|| format!("failed to bind {ip}:{}", args.port))?;
    let addr = listener.local_addr()?;
    tracing::info!("anvil serve listening on http://{addr}");

    // The single intentional stdout line: machine-readable readiness with
    // the resolved address (required when --port 0 picks an ephemeral port).
    let mut ready = json!({
        "type": "serve.ready",
        "url": format!("http://{addr}"),
        "version": env!("CARGO_PKG_VERSION"),
        "auth_required": state.auth.is_some(),
    });
    if let Some(token) = generated_token {
        ready["auth_token"] = json!(token);
    }
    println!("{ready}");
    use std::io::Write as _;
    let _ = std::io::stdout().flush();

    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("http server error")
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(e) => {
                tracing::warn!("failed to install SIGTERM handler: {e}");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => tracing::info!("shutting down on ctrl-c"),
        _ = terminate => tracing::info!("shutting down on SIGTERM"),
    }
}

/// SHA-256 digest of the configured bearer token. Requests are checked by
/// comparing digests rather than raw bytes: an early-exit comparison over a
/// digest leaks nothing an attacker can extend into the token, because the
/// digest is not secret-prefix-continuable and cannot be inverted.
struct AuthToken {
    digest: [u8; 32],
}

impl AuthToken {
    fn new(token: &str) -> Self {
        use sha2::{Digest, Sha256};
        Self {
            digest: Sha256::digest(token.as_bytes()).into(),
        }
    }

    fn matches(&self, presented: &str) -> bool {
        use sha2::{Digest, Sha256};
        let presented: [u8; 32] = Sha256::digest(presented.as_bytes()).into();
        presented == self.digest
    }
}

/// Resolve the effective bearer token from the CLI/env inputs. Returns the
/// digest checker plus the plaintext token when it was generated here (so
/// `serve.ready` can report it exactly once).
fn resolve_auth(args: &ServeArgs) -> Result<(Option<AuthToken>, Option<String>)> {
    if args.generate_auth_token {
        let raw: [u8; 32] = rand::random();
        use base64::Engine as _;
        let token = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(raw);
        return Ok((Some(AuthToken::new(&token)), Some(token)));
    }
    if let Some(path) = &args.auth_token_file {
        let token = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read --auth-token-file {}", path.display()))?;
        let token = token.trim();
        if token.is_empty() {
            bail!("--auth-token-file {} is empty", path.display());
        }
        return Ok((Some(AuthToken::new(token)), None));
    }
    if let Some(token) = &args.auth_token {
        if token.is_empty() {
            bail!("--auth-token must not be empty");
        }
        return Ok((Some(AuthToken::new(token)), None));
    }
    Ok((None, None))
}

/// Bearer-token gate for every route except `/health` (liveness stays
/// unauthenticated). No-op when no token is configured — the documented
/// local policy for the loopback-only default.
async fn auth_middleware(State(state): State<ApiState>, request: Request, next: Next) -> Response {
    let Some(expected) = &state.auth else {
        return next.run(request).await;
    };
    if request.uri().path() == "/health" {
        return next.run(request).await;
    }
    let presented = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    match presented {
        Some(token) if expected.matches(token) => next.run(request).await,
        presented => {
            tracing::warn!(
                target: "audit",
                path = %request.uri().path(),
                method = %request.method(),
                header_present = presented.is_some(),
                "rejected unauthenticated http request"
            );
            ApiError {
                status: StatusCode::UNAUTHORIZED,
                code: "unauthorized",
                message: "missing or invalid bearer token".to_string(),
                details: None,
            }
            .into_response()
        }
    }
}

#[derive(Clone)]
struct ApiState {
    sessions: SessionStore,
    llm: Arc<MultiBackend>,
    /// Serializes `?refresh=true` model re-discovery so concurrent requests
    /// don't stack redundant provider probes (mirrors the ACP refresh lock).
    refresh_lock: Arc<tokio::sync::Mutex<()>>,
    /// Registry of prompt runs started over HTTP (#318).
    runs: Arc<runs::RunManager>,
    /// Pending interactive permission requests (#319).
    permissions: Arc<permissions::PermissionRegistry>,
    /// SHA-256 digest of the required bearer token; `None` disables auth
    /// (loopback-only default policy).
    auth: Option<Arc<AuthToken>>,
    /// Canonicalized roots session workspaces must live under; empty means
    /// unrestricted.
    workspace_roots: Arc<Vec<PathBuf>>,
    /// Whether HTTP clients may select `bypassPermissions`.
    allow_bypass_permissions: bool,
    /// Per-prompt tool-turn cap; `usize::MAX` means unbounded (`--max-turns 0`).
    max_turns: usize,
    /// Binary-wide LLM stream timeouts, overridable per session.
    default_idle_timeout_secs: u64,
    default_stall_timeout_secs: u64,
}

fn router(state: ApiState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/tools", get(list_tools))
        .route("/v1/sessions", get(list_sessions).post(create_session))
        .route(
            "/v1/sessions/{session_id}",
            get(get_session)
                .patch(configure_session)
                .delete(delete_session),
        )
        .route("/v1/sessions/{session_id}/load", post(load_session))
        .route("/v1/sessions/{session_id}/resume", post(resume_session))
        .route(
            "/v1/sessions/{session_id}/runs",
            get(runs::list_runs).post(runs::create_run),
        )
        .route("/v1/runs/{run_id}", get(runs::get_run))
        .route("/v1/runs/{run_id}/events", get(runs::run_events))
        .route("/v1/runs/{run_id}/cancel", post(runs::cancel_run))
        .route(
            "/v1/runs/{run_id}/permissions",
            get(permissions::list_run_permissions),
        )
        .route(
            "/v1/permissions/{permission_id}",
            get(permissions::get_permission),
        )
        .route(
            "/v1/permissions/{permission_id}/respond",
            post(permissions::respond_permission),
        )
        .fallback(fallback_not_found)
        // Explicit request-size cap (axum's default, stated as policy):
        // large payloads are structured-output schemas and prompts, both
        // comfortably under this.
        .layer(axum::extract::DefaultBodyLimit::max(2 * 1024 * 1024))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .layer(middleware::from_fn(request_id_middleware))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Request ids + error envelope
// ---------------------------------------------------------------------------

tokio::task_local! {
    /// Per-request id, scoped around each handler by `request_id_middleware`
    /// so `ApiError::into_response` can embed it without threading it
    /// through every handler signature.
    static REQUEST_ID: String;
}

async fn request_id_middleware(request: Request, next: Next) -> Response {
    let request_id = uuid::Uuid::new_v4().to_string();
    let mut response = REQUEST_ID
        .scope(request_id.clone(), next.run(request))
        .await;
    if let Ok(header) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert("x-request-id", header);
    }
    response
}

/// Stable JSON error envelope. Every non-2xx response the daemon produces
/// goes through this type so clients can rely on one shape.
#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
    details: Option<Value>,
}

impl ApiError {
    fn invalid_argument(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "invalid_argument",
            message: message.into(),
            details: None,
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "not_found",
            message: message.into(),
            details: None,
        }
    }

    fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            code: "conflict",
            message: message.into(),
            details: None,
        }
    }

    fn forbidden(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            code: "forbidden",
            message: message.into(),
            details: None,
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "internal",
            message: message.into(),
            details: None,
        }
    }

    fn details(mut self, details: Value) -> Self {
        self.details = Some(details);
        self
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let request_id = REQUEST_ID.try_with(|id| id.clone()).ok();
        let mut error = json!({
            "code": self.code,
            "message": self.message,
        });
        if let Some(details) = self.details {
            error["details"] = details;
        }
        let body = json!({
            "error": error,
            "request_id": request_id,
        });
        (self.status, Json(body)).into_response()
    }
}

async fn fallback_not_found() -> ApiError {
    ApiError::not_found("unknown route")
}

/// `axum::Json` wrapper whose rejection uses the shared error envelope
/// instead of axum's plain-text bodies.
struct ApiJson<T>(T);

impl<S, T> FromRequest<S> for ApiJson<T>
where
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(ApiJson(value)),
            Err(rejection) => Err(ApiError::invalid_argument(format!(
                "invalid JSON request body: {}",
                rejection.body_text()
            ))),
        }
    }
}

fn unknown_session_error(session_id: &str) -> ApiError {
    ApiError::not_found(format!("unknown session '{session_id}'"))
}

/// Map a shared `apply_config_option` failure onto the HTTP envelope,
/// naming the request field (not the internal config id) in the details.
fn config_apply_api_error(field: &'static str, err: ConfigApplyError) -> ApiError {
    match err {
        ConfigApplyError::UnknownSession => ApiError::not_found("unknown session"),
        ConfigApplyError::InvalidValue { reason, supported } => {
            let mut details = json!({ "field": field });
            if !supported.is_empty() {
                details["supported"] = json!(supported);
            }
            ApiError::invalid_argument(reason).details(details)
        }
        // Unreachable from this module: the field->config-id mapping below is
        // static. Kept non-panicking so a future mapping bug degrades to 500.
        ConfigApplyError::UnknownConfigId => ApiError::internal(err.human_message()),
    }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateSessionRequest {
    /// Absolute working directory the session operates in.
    cwd: String,
    #[serde(default)]
    additional_directories: Vec<String>,
    /// Additional per-session MCP servers, additive to Anvil's canonical
    /// Bifrost setup. Uses Anvil's internal MCP server config shape.
    #[serde(default)]
    mcp_servers: Vec<McpServerConfig>,
    #[serde(flatten)]
    config: SessionConfigPatch,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LifecycleRequest {
    /// Must match the cwd the session was created under.
    cwd: String,
    #[serde(default)]
    additional_directories: Vec<String>,
    #[serde(default)]
    mcp_servers: Vec<McpServerConfig>,
}

/// The client-owned per-session configuration selectors, mirroring the ACP
/// `SessionConfigOption` ids. Applied through the same shared
/// `apply_config_option` path as ACP so validation cannot drift.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionConfigPatch {
    /// Wire model id (`codex::gpt-5-codex`, `ollama::llama3:latest`).
    model: Option<String>,
    /// Reasoning effort (`low`/`medium`/`high`/... per model), `off` to omit
    /// reasoning controls, or empty string to clear back to the model default.
    reasoning_effort: Option<String>,
    /// Provider service tier id, or empty string to clear.
    service_tier: Option<String>,
    /// `LUTZ` or `PLAN`.
    behavior_mode: Option<String>,
    /// `default`, `auto`, `acceptEdits`, `readOnly`, or `bypassPermissions`.
    permission_mode: Option<String>,
}

impl SessionConfigPatch {
    fn is_empty(&self) -> bool {
        self.model.is_none()
            && self.reasoning_effort.is_none()
            && self.service_tier.is_none()
            && self.behavior_mode.is_none()
            && self.permission_mode.is_none()
    }

    /// Field-name/config-id/value triples in application order. Model goes
    /// first so a reasoning-effort or service-tier pick in the same request
    /// validates against the newly selected model.
    fn entries(&self) -> Vec<(&'static str, &'static str, &str)> {
        let mut entries = Vec::new();
        if let Some(value) = &self.model {
            entries.push(("model", MODEL_CONFIG_ID, value.as_str()));
        }
        if let Some(value) = &self.reasoning_effort {
            entries.push((
                "reasoning_effort",
                REASONING_EFFORT_CONFIG_ID,
                value.as_str(),
            ));
        }
        if let Some(value) = &self.service_tier {
            entries.push(("service_tier", SERVICE_TIER_CONFIG_ID, value.as_str()));
        }
        if let Some(value) = &self.behavior_mode {
            entries.push(("behavior_mode", BEHAVIOR_CONFIG_ID, value.as_str()));
        }
        if let Some(value) = &self.permission_mode {
            entries.push(("permission_mode", PERMISSION_CONFIG_ID, value.as_str()));
        }
        entries
    }
}

#[derive(Debug, Serialize)]
struct SessionResource {
    id: String,
    cwd: String,
    additional_directories: Vec<String>,
    model: String,
    behavior_mode: String,
    permission_mode: String,
    reasoning_effort: Option<String>,
    service_tier: Option<String>,
    title: Option<String>,
    created_at_ms: u64,
    modified_at_ms: u64,
    updated_at: Option<String>,
    history_turns: usize,
    usage: UsageResource,
    usage_cost_usd: Option<f64>,
    /// Full conversation history; present only on `load` responses and
    /// `GET ...?include_history=true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    history: Option<Value>,
}

#[derive(Debug, Serialize)]
struct UsageResource {
    input_tokens: u64,
    output_tokens: u64,
    thought_tokens: u64,
    cached_read_tokens: u64,
    cached_write_tokens: u64,
}

#[derive(Debug, Serialize)]
struct SessionListEntry {
    id: String,
    title: Option<String>,
    created_at_ms: u64,
    modified_at_ms: u64,
    updated_at: Option<String>,
    /// Present for resident sessions; disk listings echo the query `cwd`.
    cwd: Option<String>,
    resident: bool,
}

fn usage_resource(usage: crate::llm_client::TokenUsage) -> UsageResource {
    UsageResource {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        thought_tokens: usage.thought_tokens,
        cached_read_tokens: usage.cached_read_tokens,
        cached_write_tokens: usage.cached_write_tokens,
    }
}

/// Render conversation history for API consumers. A hand-built projection
/// (rather than `derive(Serialize)` on the session types) so the wire shape
/// is an explicit, stable contract independent of in-memory layout; the
/// full-fidelity event model arrives with runs/SSE in #318.
fn history_value(history: &[crate::session::ConversationTurn]) -> Value {
    let turns: Vec<Value> = history
        .iter()
        .map(|turn| {
            json!({
                "user_prompt": turn.user_prompt,
                "agent_response": turn.agent_response,
                "summary": turn.summary,
                "tool_exchanges": turn
                    .tool_exchanges
                    .iter()
                    .map(|exchange| json!({
                        "call_id": exchange.call_id,
                        "tool_name": exchange.tool_name,
                        "arguments": exchange.arguments,
                        "result": exchange.result,
                        "status": exchange.status.as_str(),
                        "permission_notice": exchange.permission_notice,
                    }))
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    Value::Array(turns)
}

async fn session_resource(
    store: &SessionStore,
    session: &Session,
    include_history: bool,
) -> SessionResource {
    let history = include_history.then(|| history_value(&session.history));
    SessionResource {
        id: session.id.clone(),
        cwd: session.cwd.display().to_string(),
        additional_directories: session
            .additional_directories
            .iter()
            .map(|p| p.display().to_string())
            .collect(),
        model: session.model.clone(),
        behavior_mode: session.mode.as_str().to_string(),
        permission_mode: session.permission_mode.as_str().to_string(),
        reasoning_effort: session.selected_reasoning_effort.clone(),
        service_tier: session.selected_service_tier.clone(),
        title: session.manifest.title(),
        created_at_ms: session.manifest.created,
        modified_at_ms: session.manifest.modified,
        updated_at: session.manifest.updated_at(),
        history_turns: session.history.len(),
        usage: usage_resource(session.usage),
        usage_cost_usd: store.exact_usage_cost_usd(&session.id).await,
        history,
    }
}

fn model_resource(meta: &ModelMetadata) -> Value {
    json!({
        "id": meta.id,
        "default_reasoning_level": meta.default_reasoning_level,
        "supported_reasoning_levels": meta
            .supported_reasoning_levels
            .iter()
            .map(|preset| json!({
                "effort": preset.effort,
                "description": preset.description,
            }))
            .collect::<Vec<_>>(),
        "service_tiers": meta
            .service_tiers
            .iter()
            .map(|tier| json!({
                "id": tier.id,
                "name": tier.name,
                "description": tier.description,
            }))
            .collect::<Vec<_>>(),
        "supports_images": meta.supports_images,
        "context_length": meta.context_length,
        "pricing": meta.pricing.map(|pricing| json!({
            "input_cost_per_token_usd": pricing.input_cost_per_token_usd,
            "output_cost_per_token_usd": pricing.output_cost_per_token_usd,
        })),
    })
}

/// Fallback cwd for store lookups that only need to find a *resident*
/// session (cold loads route through `/load`/`/resume`, which carry an
/// explicit cwd). Mirrors the ACP config handlers' use of the process cwd.
fn fallback_cwd(explicit: Option<&str>) -> PathBuf {
    match explicit {
        Some(cwd) => PathBuf::from(cwd),
        None => std::env::current_dir().unwrap_or_default(),
    }
}

/// Canonicalize a path for workspace-root policy checks, walking up to the
/// first existing ancestor when the tail does not exist yet (a session cwd
/// may be created later). Callers must reject `..` components first.
fn canonicalize_for_policy(path: &std::path::Path) -> PathBuf {
    let mut current = path.to_path_buf();
    let mut missing_tail: Vec<std::ffi::OsString> = Vec::new();
    loop {
        match current.canonicalize() {
            Ok(canonical) => {
                let mut resolved = canonical;
                for component in missing_tail.iter().rev() {
                    resolved.push(component);
                }
                return resolved;
            }
            Err(_) => match (current.parent(), current.file_name()) {
                (Some(parent), Some(name)) => {
                    missing_tail.push(name.to_os_string());
                    current = parent.to_path_buf();
                }
                _ => return path.to_path_buf(),
            },
        }
    }
}

/// Enforce the server's `--workspace-root` policy on a requested session
/// path. No-op when no roots are configured. Rejections are audited: a
/// caller probing outside the sandbox is exactly what the audit trail is
/// for.
fn enforce_workspace_roots(
    state: &ApiState,
    field: &'static str,
    path: &std::path::Path,
) -> Result<(), ApiError> {
    if state.workspace_roots.is_empty() {
        return Ok(());
    }
    let escapes_via_parent = path
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir));
    let canonical = canonicalize_for_policy(path);
    let allowed = !escapes_via_parent
        && state
            .workspace_roots
            .iter()
            .any(|root| canonical.starts_with(root));
    if allowed {
        return Ok(());
    }
    tracing::warn!(
        target: "audit",
        field,
        path = %path.display(),
        "rejected session path outside configured workspace roots"
    );
    Err(ApiError::forbidden(format!(
        "{field} '{}' is outside the server's configured workspace roots",
        path.display()
    ))
    .details(json!({
        "field": field,
        "workspace_roots": state
            .workspace_roots
            .iter()
            .map(|root| root.display().to_string())
            .collect::<Vec<_>>(),
    })))
}

/// Reject `bypassPermissions` from HTTP callers unless the operator opted
/// in with `--allow-bypass-permissions` (#319).
fn enforce_permission_mode_policy(
    state: &ApiState,
    patch: &SessionConfigPatch,
) -> Result<(), ApiError> {
    if state.allow_bypass_permissions {
        return Ok(());
    }
    if patch.permission_mode.as_deref() == Some("bypassPermissions") {
        tracing::warn!(
            target: "audit",
            "rejected bypassPermissions request from http client (server policy)"
        );
        return Err(ApiError::forbidden(
            "permission_mode 'bypassPermissions' is disabled for HTTP clients; start the              daemon with --allow-bypass-permissions to permit it",
        ));
    }
    Ok(())
}

fn require_absolute_cwd(cwd: &str) -> Result<PathBuf, ApiError> {
    let path = PathBuf::from(cwd);
    if !path.is_absolute() {
        return Err(
            ApiError::invalid_argument(format!("cwd must be an absolute path: {cwd}"))
                .details(json!({ "field": "cwd" })),
        );
    }
    Ok(path)
}

fn validated_additional_directories(directories: Vec<String>) -> Result<Vec<PathBuf>, ApiError> {
    validate_additional_directories(directories.into_iter().map(PathBuf::from).collect()).map_err(
        |err| {
            ApiError::invalid_argument(format!(
                "additional_directories[{}] ('{}') must be {}",
                err.index,
                err.path.display(),
                err.requirement
            ))
            .details(json!({
                "field": "additional_directories",
                "index": err.index,
            }))
        },
    )
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health(State(state): State<ApiState>) -> Json<Value> {
    let models_discovered = state.sessions.available_model_metadata().await.len();
    Json(json!({
        "status": "ok",
        "name": "anvil",
        "version": env!("CARGO_PKG_VERSION"),
        "models_discovered": models_discovered,
    }))
}

#[derive(Debug, Deserialize)]
struct ModelsQuery {
    #[serde(default)]
    refresh: bool,
}

async fn list_models(
    State(state): State<ApiState>,
    Query(query): Query<ModelsQuery>,
) -> Json<Value> {
    if query.refresh {
        let _guard = state.refresh_lock.lock().await;
        match state.llm.list_model_metadata_with_progress(None).await {
            Ok(models) => {
                seed_default_model_if_empty(&state.sessions, &models).await;
                state.sessions.set_available_models(models).await;
            }
            // Discovery failures are never fatal (dead Ollama, missing
            // auth); serve the cached catalog.
            Err(e) => tracing::warn!("model discovery refresh failed: {e:#}"),
        }
    }
    let catalog = state.sessions.available_model_metadata().await;
    let default_model = state.sessions.default_model().await;
    Json(json!({
        "models": catalog.iter().map(model_resource).collect::<Vec<_>>(),
        "default_model": (!default_model.is_empty()).then_some(default_model),
    }))
}

async fn list_tools() -> Json<Value> {
    let tools: Vec<Value> = crate::tools::tool_catalog()
        .iter()
        .map(|entry| {
            json!({
                "name": entry.name,
                "kind": serde_json::to_value(entry.kind).unwrap_or_else(|_| json!("other")),
                "display_name": entry.display_name,
                "concurrency_safe": entry.concurrency_safe,
                "source": if entry.builtin { "builtin" } else { "mcp" },
            })
        })
        .collect();
    Json(json!({ "tools": tools }))
}

#[derive(Debug, Deserialize)]
struct ListSessionsQuery {
    /// When set, also lists persisted sessions on disk under this
    /// workspace (like ACP `session/list`).
    cwd: Option<String>,
}

async fn list_sessions(
    State(state): State<ApiState>,
    Query(query): Query<ListSessionsQuery>,
) -> Result<Json<Value>, ApiError> {
    let mut entries = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (manifest, cwd) in state.sessions.resident_session_manifests().await {
        seen.insert(manifest.id.clone());
        entries.push(SessionListEntry {
            id: manifest.id.clone(),
            title: manifest.title(),
            created_at_ms: manifest.created,
            modified_at_ms: manifest.modified,
            updated_at: manifest.updated_at(),
            cwd: Some(cwd.display().to_string()),
            resident: true,
        });
    }
    if let Some(cwd) = &query.cwd {
        let cwd = require_absolute_cwd(cwd)?;
        enforce_workspace_roots(&state, "cwd", &cwd)?;
        for manifest in state.sessions.list_sessions_from_disk(&cwd).await {
            if seen.contains(&manifest.id) {
                continue;
            }
            entries.push(SessionListEntry {
                id: manifest.id.clone(),
                title: manifest.title(),
                created_at_ms: manifest.created,
                modified_at_ms: manifest.modified,
                updated_at: manifest.updated_at(),
                cwd: Some(cwd.display().to_string()),
                resident: false,
            });
        }
    }
    entries.sort_by_key(|entry| std::cmp::Reverse(entry.modified_at_ms));
    Ok(Json(json!({ "sessions": entries })))
}

async fn create_session(
    State(state): State<ApiState>,
    ApiJson(request): ApiJson<CreateSessionRequest>,
) -> Result<(StatusCode, Json<SessionResource>), ApiError> {
    let cwd = require_absolute_cwd(&request.cwd)?;
    let additional_directories = validated_additional_directories(request.additional_directories)?;
    enforce_workspace_roots(&state, "cwd", &cwd)?;
    for directory in &additional_directories {
        enforce_workspace_roots(&state, "additional_directories", directory)?;
    }
    enforce_permission_mode_policy(&state, &request.config)?;

    let session = state
        .sessions
        .create_session_with_mcp_servers_and_additional_directories(
            cwd.clone(),
            Some(request.mcp_servers),
            additional_directories,
        )
        .await;

    // Apply requested config through the shared ACP path. On the first
    // invalid value the half-configured session is deleted again so a
    // failed create leaves no state behind.
    for (field, config_id, value) in request.config.entries() {
        if let Err(err) = apply_config_option(&state.sessions, &session.id, config_id, value).await
        {
            state.sessions.delete_session(&session.id).await;
            return Err(config_apply_api_error(field, err));
        }
    }

    let session = state
        .sessions
        .get_session(&session.id, &cwd)
        .await
        .ok_or_else(|| ApiError::internal("session vanished during creation"))?;
    let resource = session_resource(&state.sessions, &session, false).await;
    Ok((StatusCode::CREATED, Json(resource)))
}

#[derive(Debug, Deserialize)]
struct GetSessionQuery {
    /// Workspace to cold-load the session zip from when it is not
    /// resident. Defaults to the daemon's process cwd.
    cwd: Option<String>,
    #[serde(default)]
    include_history: bool,
}

async fn get_session(
    State(state): State<ApiState>,
    Path(session_id): Path<String>,
    Query(query): Query<GetSessionQuery>,
) -> Result<Json<SessionResource>, ApiError> {
    let cwd = fallback_cwd(query.cwd.as_deref());
    if query.cwd.is_some() {
        enforce_workspace_roots(&state, "cwd", &cwd)?;
    }
    let session = state
        .sessions
        .get_session(&session_id, &cwd)
        .await
        .ok_or_else(|| unknown_session_error(&session_id))?;
    Ok(Json(
        session_resource(&state.sessions, &session, query.include_history).await,
    ))
}

#[derive(Debug, Serialize)]
struct ConfigureSessionResponse {
    session: SessionResource,
    /// Human-readable side effects of the change (e.g. a model switch
    /// dropping a reasoning-effort pick the new model doesn't support).
    warnings: Vec<String>,
}

async fn configure_session(
    State(state): State<ApiState>,
    Path(session_id): Path<String>,
    ApiJson(patch): ApiJson<SessionConfigPatch>,
) -> Result<Json<ConfigureSessionResponse>, ApiError> {
    if patch.is_empty() {
        return Err(ApiError::invalid_argument(
            "at least one of model, reasoning_effort, service_tier, behavior_mode, or \
             permission_mode is required",
        ));
    }
    enforce_permission_mode_policy(&state, &patch)?;
    let mut warnings = Vec::new();
    // Applied sequentially through the shared ACP path; on a validation
    // failure, selectors already applied stay applied (same as issuing the
    // equivalent ACP `session/set_config_option` calls one by one).
    for (field, config_id, value) in patch.entries() {
        match apply_config_option(&state.sessions, &session_id, config_id, value).await {
            Ok(outcome) => {
                if let Some(cleared) = outcome.cleared_reasoning {
                    warnings.push(format!(
                        "reasoning effort '{cleared}' is not supported by the new model and was reset to the model default"
                    ));
                }
                if let Some(cleared) = outcome.cleared_service_tier {
                    warnings.push(format!(
                        "service tier '{cleared}' is not supported by the new model and was reset to the provider default"
                    ));
                }
            }
            Err(err) => return Err(config_apply_api_error(field, err)),
        }
    }
    let session = state
        .sessions
        .get_session(&session_id, &fallback_cwd(None))
        .await
        .ok_or_else(|| unknown_session_error(&session_id))?;
    Ok(Json(ConfigureSessionResponse {
        session: session_resource(&state.sessions, &session, false).await,
        warnings,
    }))
}

async fn delete_session(
    State(state): State<ApiState>,
    Path(session_id): Path<String>,
) -> Json<Value> {
    let deleted = state.sessions.delete_session(&session_id).await;
    Json(json!({ "deleted": deleted }))
}

async fn load_session(
    State(state): State<ApiState>,
    Path(session_id): Path<String>,
    ApiJson(request): ApiJson<LifecycleRequest>,
) -> Result<Json<SessionResource>, ApiError> {
    reopen_session(state, session_id, request, true).await
}

async fn resume_session(
    State(state): State<ApiState>,
    Path(session_id): Path<String>,
    ApiJson(request): ApiJson<LifecycleRequest>,
) -> Result<Json<SessionResource>, ApiError> {
    reopen_session(state, session_id, request, false).await
}

/// Shared body of `/load` and `/resume`. Both reopen a persisted session
/// under its original cwd, reapply workspace roots and per-session MCP
/// servers, and return the session resource; `/load` additionally embeds
/// the conversation history (the HTTP analogue of ACP `session/load`'s
/// history replay).
async fn reopen_session(
    state: ApiState,
    session_id: String,
    request: LifecycleRequest,
    include_history: bool,
) -> Result<Json<SessionResource>, ApiError> {
    let cwd = require_absolute_cwd(&request.cwd)?;
    let additional_directories = validated_additional_directories(request.additional_directories)?;
    enforce_workspace_roots(&state, "cwd", &cwd)?;
    for directory in &additional_directories {
        enforce_workspace_roots(&state, "additional_directories", directory)?;
    }

    match state
        .sessions
        .reopen_session_checked(&session_id, &cwd)
        .await
    {
        LifecycleReopen::Reopened(_) => {}
        LifecycleReopen::CwdMismatch { session_cwd } => {
            return Err(ApiError::conflict(format!(
                "cwd '{}' does not match the session's cwd '{}'; moving a session between \
                 working directories is not supported",
                cwd.display(),
                session_cwd.display(),
            ))
            .details(json!({ "session_cwd": session_cwd.display().to_string() })));
        }
        LifecycleReopen::Unknown => return Err(unknown_session_error(&session_id)),
    }

    if let Err(err) = state
        .sessions
        .update_workspace_roots(&session_id, cwd.clone(), additional_directories)
        .await
    {
        return Err(ApiError::internal(format!(
            "failed to update session workspace roots: {err:#}"
        )));
    }
    state
        .sessions
        .apply_lifecycle_mcp_servers(&session_id, request.mcp_servers)
        .await;

    let session = state
        .sessions
        .get_session(&session_id, &cwd)
        .await
        .ok_or_else(|| unknown_session_error(&session_id))?;
    Ok(Json(
        session_resource(&state.sessions, &session, include_history).await,
    ))
}

mod permissions;
mod runs;

#[cfg(test)]
mod tests;
