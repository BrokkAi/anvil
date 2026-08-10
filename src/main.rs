use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::builder::RangedU64ValueParser;
use clap::{Parser, Subcommand};

mod acp;
mod agents;
mod agents_md;
mod bedrock_auth;
mod bedrock_client;
mod bedrock_credits;
mod codex_auth;
mod codex_client;
mod codex_credits;
mod context_manager;
mod deepseek_auth;
mod deepseek_balance;
mod discovery;
mod goal;
mod headless;
mod host_notice;
#[cfg(feature = "http-api")]
mod http_api;
mod http_retry;
mod infer;
mod installer;
mod kimi_auth;
mod llm_client;
mod lsp;
mod mcp;
mod multi_backend;
mod openai_providers;
mod openrouter_auth;
mod openrouter_credits;
mod p2t;
mod plan;
mod plugins;
mod responses_api;
mod runtime;
mod sandbox_backend;
mod secrets;
mod semantic_readiness;
mod semantic_rerank;
mod session;
mod setup_state;
mod skills;
mod slash;
mod structured_output;
mod terminal_notifications;
mod text;
mod tokens;
mod tool_arguments;
mod tool_loop;
mod tools;
mod trace_logging;
mod train_bifrost;
mod turn_runner;
mod usage_report;
mod workspace_delta;

use crate::llm_client::LlmBackend;
use crate::multi_backend::{BackendRegistration, MultiBackend};

/// Anvil -- Rust-based Agent Client Protocol (ACP) server with
/// first-run setup and zero-config auto-discovery: at startup we read
/// `~/.codex/auth.json` for Codex credentials, probe
/// `http://localhost:11434/v1/models` for Ollama, and include OpenRouter
/// when credentials are configured. No flags are required to point at a
/// different Ollama URL or restrict the picker -- if Ollama isn't on the
/// default port, it's simply not in the catalog.
#[derive(Parser)]
#[command(name = "anvil", version, about)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,

    /// Run one prompt headlessly and exit: connect a built-in ACP client to
    /// the in-process agent, stream or print the result, then quit. Omit the
    /// value, or pass `-`, to read the prompt from stdin. For a prompt that
    /// starts with `-`, use `--print=<prompt>` or stdin. Mirrors Mjolnir's
    /// `mj --print` contract (#356).
    #[arg(
        short = 'p',
        long = "print",
        value_name = "PROMPT",
        num_args = 0..=1,
        default_missing_value = "-"
    )]
    print: Option<String>,

    /// Output format for --print: `text` prints the final assistant message,
    /// `json` prints one object at exit, `stream-json` prints
    /// newline-delimited records as they happen with `result` last.
    #[arg(long, value_enum, default_value_t = headless::OutputFormat::Text, requires = "print")]
    output_format: headless::OutputFormat,

    /// Permission policy for --print. `manual` honors read-only auto-approvals
    /// and remembered repo-scoped Always allow grants, but rejects every
    /// permission request so a run can never hang; `auto` accepts
    /// edit/delete/move requests but rejects shell execution; `yolo` accepts
    /// everything.
    #[arg(long, value_enum, requires = "print")]
    permission_mode: Option<headless::PermissionMode>,

    /// Working directory for the --print session. Defaults to the current
    /// directory.
    #[arg(long, value_name = "PATH", requires = "print")]
    cwd: Option<PathBuf>,

    /// Model for the --print session, with an optional trailing `+<effort>`
    /// (off, none, minimal, low, medium, high, xhigh, max) to set the
    /// session's reasoning effort independent of the model default.
    #[arg(
        long,
        value_name = "MODEL[+EFFORT]",
        requires = "print",
        value_parser = headless::parse_model_override
    )]
    model: Option<(String, Option<String>)>,

    /// Resume an existing session by id instead of creating a new one. The
    /// session's original working directory must match.
    #[arg(long, value_name = "SESSION_ID", requires = "print")]
    resume: Option<String>,

    /// Override the default model id for new sessions. Accepts a wire
    /// form (`codex::<id>`, `ollama::llama3:latest`) or a bare id
    /// that routes to the preferred backend (Codex if available, else
    /// Ollama). When unset, the first discovered model wins.
    #[arg(long, default_value = "")]
    default_model: String,

    /// Seed new sessions with a reasoning effort such as `low`,
    /// `medium`, or `high`, or `off` to omit provider reasoning
    /// controls. Models that do not support configurable reasoning
    /// ignore unsupported effort levels and fall back to their default behavior.
    #[arg(long)]
    reasoning_effort: Option<String>,

    /// Optional cap on tool-calling turns per prompt. Defaults to `0` =
    /// unbounded: the loop runs until the model answers without a tool call
    /// (normal completion), with stalls caught earlier by the LLM stream
    /// timeouts and the no-progress nudges -- the same model-driven termination Codex
    /// uses and that `/goal` already uses here. A turn count is a poor work
    /// budget (you can't know up front how many tool rounds a task needs), so
    /// it is opt-in: pass a positive `--max-turns N` only to deliberately bound
    /// cost/time, in which case hitting N forces a final text response. The
    /// conversation context is preserved on that stop, so sending another
    /// message (e.g. "continue") resumes the task from where it stopped.
    #[arg(long, default_value_t = 0)]
    max_turns: usize,

    /// Maximum number of sessions to keep resident in memory before the
    /// least-recently-used session is evicted (the on-disk zip is unaffected
    /// and can be reloaded). Set to `0` to disable the cap.
    #[arg(long, default_value_t = 50)]
    max_sessions: usize,

    /// Maximum number of conversation turns retained per session in memory
    /// (sliding window). Older turns are dropped from memory once the cap is
    /// exceeded; the persisted zip retains the full history. Set to `0` to
    /// disable the cap.
    #[arg(long, default_value_t = 50)]
    max_history_turns: usize,

    /// DEPRECATED. MCP servers are configured with `/mcp`; Anvil now manages
    /// its own pinned local Bifrost binary for the built-in MCP server.
    #[arg(long, env = "BROKK_BIFROST_BINARY", hide = true)]
    bifrost_binary: Option<PathBuf>,

    /// Seconds to wait for the first meaningful SSE progress before aborting
    /// a streaming LLM response. Bump higher for models/providers that spend
    /// a long time reasoning before streaming any text. Overridable
    /// per-session via `/idle-timeout`, which sets both first-progress and
    /// mid-stream stall timeouts for back compatibility.
    #[arg(
        long,
        env = "ANVIL_LLM_IDLE_TIMEOUT_SECS",
        default_value_t = llm_client::DEFAULT_IDLE_CHUNK_TIMEOUT_SECS,
        value_parser = RangedU64ValueParser::<u64>::new()
            .range(llm_client::MIN_IDLE_CHUNK_TIMEOUT_SECS..=llm_client::MAX_IDLE_CHUNK_TIMEOUT_SECS),
    )]
    llm_idle_timeout_secs: u64,

    /// Seconds to wait between meaningful SSE chunks after streaming has
    /// started before aborting a stalled LLM response. Keepalive comments and
    /// unparseable chunks do not reset this timer.
    #[arg(
        long,
        env = "ANVIL_LLM_STALL_TIMEOUT_SECS",
        default_value_t = llm_client::DEFAULT_INTER_CHUNK_TIMEOUT_SECS,
        value_parser = RangedU64ValueParser::<u64>::new()
            .range(llm_client::MIN_IDLE_CHUNK_TIMEOUT_SECS..=llm_client::MAX_IDLE_CHUNK_TIMEOUT_SECS),
    )]
    llm_stall_timeout_secs: u64,

    /// Keep install-level setup preferences process-local. Sandbox, turn-recap,
    /// and first-run choices made during this process are not read from or
    /// written to setup.json. ACP session config options (model, reasoning,
    /// behavior, permission, and service tier) are already live-session-only.
    /// Provider credential commands, the `allowed_tools` tool allowlist, and
    /// `/mcp` are not made transient by this flag.
    #[arg(long, env = "ANVIL_TRANSIENT_SETUP", default_value_t = false)]
    transient_setup: bool,

    // ----- Deprecated backward-compat flags --------------------------------
    // Existing editor configs generated by older brokk-code (Python TUI)
    // hand these in. We accept them silently so an upgrade-in-place
    // doesn't break the user's IDE -- but they no longer drive routing,
    // because Codex and Ollama auto-discover unconditionally. A warning
    // is logged at startup so users see they can clean these up next
    // time they re-run `brokk install`.
    /// DEPRECATED and ignored. Ollama is probed at the default URL
    /// `http://localhost:11434`; if your daemon listens elsewhere, run
    /// `ollama serve` on that port.
    #[arg(long, hide = true)]
    endpoint_url: Option<String>,

    /// DEPRECATED and ignored. Codex auto-detects credentials from
    /// `~/.codex/auth.json`; Ollama doesn't use an API key.
    #[arg(long, env = "BROKK_ENDPOINT_API_KEY", hide = true)]
    api_key: Option<String>,

    /// DEPRECATED and ignored. Codex is auto-detected when
    /// `~/.codex/auth.json` is present.
    #[arg(long, hide = true)]
    use_codex: bool,

    /// Disable the wasmtime-hosted parser sandbox and run all parsing
    /// (SKILL.md YAML, AGENTS.md, session zip, regex search) natively
    /// in-process. Normally the wasm sandbox is used as a fallback
    /// when no OS-level sandbox (bwrap / seatbelt) is available;
    /// this flag forces native parsing regardless. On platforms
    /// without an OS sandbox, this also means `run_shell_command`
    /// runs without any sandbox of any kind.
    #[arg(long, env = "ANVIL_NO_WASM_SANDBOX", default_value_t = false)]
    no_wasm_sandbox: bool,

    /// Disable the built-in shell-output minimizer. By default,
    /// `run_shell_command` output from well-known tools (git, cargo,
    /// pytest, npm, ...) is condensed after capture, with the raw output
    /// preserved under `.brokk/shell-output/` in the workspace and
    /// referenced from the tool result.
    #[arg(long, env = "ANVIL_NO_SHELL_MINIMIZER", default_value_t = false)]
    no_shell_minimizer: bool,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Install Anvil as an ACP agent in a supported editor.
    Install(installer::InstallArgs),
    /// Run one tool-free, schema-constrained Codex inference from JSON on stdin.
    Infer(infer::InferArgs),
    /// Run Anvil as an HTTP daemon exposing the versioned REST API
    /// (sessions, models, tools, runs) on a loopback listener.
    #[cfg(feature = "http-api")]
    Serve(http_api::ServeArgs),
}

impl std::fmt::Debug for Args {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Args")
            .field("command", &self.command)
            .field("print", &self.print)
            .field("output_format", &self.output_format)
            .field("permission_mode", &self.permission_mode)
            .field("cwd", &self.cwd)
            .field("model", &self.model)
            .field("resume", &self.resume)
            .field("default_model", &self.default_model)
            .field("reasoning_effort", &self.reasoning_effort)
            .field("max_turns", &self.max_turns)
            .field("max_sessions", &self.max_sessions)
            .field("max_history_turns", &self.max_history_turns)
            .field("bifrost_binary", &self.bifrost_binary)
            .field("llm_idle_timeout_secs", &self.llm_idle_timeout_secs)
            .field("llm_stall_timeout_secs", &self.llm_stall_timeout_secs)
            .field("transient_setup", &self.transient_setup)
            .field("no_shell_minimizer", &self.no_shell_minimizer)
            // Deprecated flags omitted from Debug to avoid leaking api_key.
            .finish()
    }
}

/// Build a Codex backend from already-loaded credentials. Returns
/// `None` for the same "credentials are unusable" cases the startup
/// path treats as no-Codex (apikey mode without an API key, etc.) so
/// the picker stays honest.
///
/// Shared by the startup path (`build_codex_backend`) and the
/// post-`/setup codex` install path in `agent.rs`. Keeps the
/// `auth.auth_mode + tokens` decision tree in one place so the two
/// callers can't drift.
pub fn codex_backend_from_auth(auth: &codex_auth::AuthDotJson) -> Option<Arc<dyn LlmBackend>> {
    // ChatGPT-subscription routing requires both `auth_mode == "chatgpt"`
    // AND a usable `tokens` block. Anything else falls through to the
    // OPENAI_API_KEY path -- including `chatgpt` mode with no tokens
    // (which can happen if a refresh just blew them away), `apikey` mode
    // (the documented API-billed fallback), and any unrecognized mode
    // string from a future codex-cli version. If we hit the apikey path
    // with no key, the prompt would 401 -- skip in that case so the
    // picker is honest about what's available.
    if matches!(auth.auth_mode.as_deref(), Some("chatgpt")) && auth.tokens.is_some() {
        return Some(Arc::new(codex_client::CodexClient::new()));
    }
    let key = auth.openai_api_key.clone();
    key.map(|k| {
        Arc::new(llm_client::OpenAiClient::new(
            "https://api.openai.com/v1".to_string(),
            Some(k),
        )) as Arc<dyn LlmBackend>
    })
}

/// Build the Codex backend if `~/.codex/auth.json` is present. Returns
/// `None` when the file is missing or unreadable. Stale credentials
/// are refreshed proactively so the first prompt doesn't burn a 401
/// round-trip.
async fn build_codex_backend() -> Option<Arc<dyn LlmBackend>> {
    let mut auth = match codex_auth::read_auth_dot_json() {
        Ok(Some(a)) => a,
        Ok(None) => {
            tracing::info!(
                "no ~/.codex/auth.json found; Codex auto-discovery skipped. Run /setup codex from a session to authenticate."
            );
            return None;
        }
        Err(e) => {
            tracing::warn!("failed to read ~/.codex/auth.json: {e:#}");
            return None;
        }
    };
    if let Err(e) = codex_auth::refresh_if_stale(&mut auth).await {
        tracing::warn!("codex credential refresh failed: {e:#}");
    }
    let backend = codex_backend_from_auth(&auth);
    match (&backend, auth.auth_mode.as_deref(), auth.tokens.is_some()) {
        (Some(_), Some("chatgpt"), true) => {
            tracing::info!(
                "Codex backend enabled in ChatGPT subscription mode (Responses API on chatgpt.com)"
            );
        }
        (Some(_), mode, _) => {
            tracing::info!(
                "Codex backend enabled in OPENAI_API_KEY mode (api.openai.com), auth_mode={:?}",
                mode
            );
        }
        (None, mode, _) => {
            tracing::warn!(
                "~/.codex/auth.json is unusable (auth_mode={:?}, no OPENAI_API_KEY); \
                 skipping Codex backend. Run /setup codex to re-authenticate.",
                mode
            );
        }
    }
    backend
}

/// Build the Ollama chat backend. Always pointed at the default
/// `http://localhost:11434`; chat requests go through Ollama's
/// OpenAI-compatible `/v1/chat/completions` shim, while discovery
/// (handled by `discovery.rs`) hits the OpenAI-compatible `/v1/models`.
/// Ollama doesn't require an API key for local use.
fn build_ollama_backend() -> Arc<dyn LlmBackend> {
    let ollama_base_url =
        test_ollama_base_url().unwrap_or_else(|| discovery::OLLAMA_DEFAULT_URL.to_string());
    let ollama_base_url = ollama_base_url.trim_end_matches('/').to_string();
    let chat_url = format!("{ollama_base_url}/v1");
    tracing::info!(
        "Ollama backend wired at {chat_url} (chat) and {}/v1/models (discovery); \
         models become available if/when the daemon responds",
        ollama_base_url
    );
    Arc::new(llm_client::OpenAiClient::with_reasoning_support(
        chat_url,
        None,
        reqwest::header::HeaderMap::new(),
    ))
}

fn test_ollama_base_url() -> Option<String> {
    // Internal test hook for integration smoke tests. There is intentionally
    // no public CLI flag for this; normal production routing remains the
    // documented zero-config Ollama default unless this explicit test env is
    // set by a harness.
    std::env::var("ANVIL_TEST_OLLAMA_BASE_URL")
        .ok()
        .filter(|url| !url.trim().is_empty())
}

/// Build a hosted DeepSeek chat backend from a raw API key. DeepSeek's API
/// is OpenAI-compatible at `https://api.deepseek.com`, but its reasoning knob
/// is spelled in DeepSeek's own dialect (`thinking` + a top-level
/// `reasoning_effort` on the `high`/`max` scale), so we build the client with
/// the DeepSeek reasoning wire rather than the unified one.
pub fn deepseek_backend_from_key(raw: &str) -> Option<Arc<dyn LlmBackend>> {
    let key = raw.trim();
    if key.is_empty() {
        return None;
    }
    Some(Arc::new(
        llm_client::OpenAiClient::with_deepseek_reasoning_support(
            discovery::DEEPSEEK_BASE_URL.to_string(),
            Some(key.to_string()),
            reqwest::header::HeaderMap::new(),
        ),
    ))
}

/// Build the hosted DeepSeek backend from `DEEPSEEK_API_KEY`, falling back
/// to the consolidated secrets store (written by `/setup deepseek key`).
/// Precedence matches OpenRouter and Bedrock: env > file > nothing.
fn build_deepseek_backend() -> Option<Arc<dyn LlmBackend>> {
    if let Ok(raw) = std::env::var(discovery::DEEPSEEK_API_KEY_ENV) {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            tracing::info!(
                "{} is set but empty; falling back to the secrets store",
                discovery::DEEPSEEK_API_KEY_ENV
            );
        } else {
            tracing::info!(
                "DeepSeek backend wired from {} at {} (chat + discovery); key length={}",
                discovery::DEEPSEEK_API_KEY_ENV,
                discovery::DEEPSEEK_BASE_URL,
                trimmed.len()
            );
            return deepseek_backend_from_key(trimmed);
        }
    }

    match deepseek_auth::read() {
        Ok(Some(auth)) => {
            let trimmed = auth.api_key.trim();
            if trimmed.is_empty() {
                tracing::info!(
                    "DeepSeek entry in the secrets store has an empty key; backend skipped"
                );
                return None;
            }
            tracing::info!(
                "DeepSeek backend wired from the secrets store at {} (chat + discovery); key length={}",
                discovery::DEEPSEEK_BASE_URL,
                trimmed.len()
            );
            deepseek_backend_from_key(trimmed)
        }
        Ok(None) => None,
        Err(e) => {
            tracing::warn!("failed to read the secrets store for DeepSeek: {e:#}");
            None
        }
    }
}

fn build_kimi_backend() -> Option<Arc<dyn LlmBackend>> {
    let auth = match kimi_auth::load_provider() {
        Ok(auth) => auth,
        Err(error) => {
            tracing::warn!("failed to configure Kimi authentication: {error:#}");
            return None;
        }
    }?;
    let headers = match kimi_auth::default_headers() {
        Ok(headers) => headers,
        Err(error) => {
            tracing::warn!("failed to configure Kimi request headers: {error:#}");
            return None;
        }
    };
    let base_url = kimi_auth::base_url();
    tracing::info!(
        base_url,
        "Kimi backend wired from KIMI_API_KEY or Kimi Code credentials"
    );
    Some(Arc::new(llm_client::OpenAiClient::with_kimi_support(
        base_url, auth, headers,
    )))
}

/// Build an OpenRouter chat backend from a raw API key. OpenRouter speaks
/// the OpenAI Chat Completions wire format verbatim, so we reuse
/// `OpenAiClient` with the OpenRouter base URL and attach the optional
/// `HTTP-Referer` / `X-Title` attribution headers (these drive the
/// openrouter.ai leaderboard rankings; both are documented as optional
/// but we always set them so the app shows up consistently).
///
/// Whitespace is trimmed so accidental shell quoting
/// (`export OPENROUTER_API_KEY=" sk-..."`) doesn't 401 every request.
/// Returns `None` for an empty key so callers can distinguish "not
/// configured" from "configured but broken".
pub fn openrouter_backend_from_key(raw: &str) -> Option<Arc<dyn LlmBackend>> {
    let key = raw.trim();
    if key.is_empty() {
        return None;
    }

    let mut headers = reqwest::header::HeaderMap::new();
    // Both header values are well-known ASCII strings the API expects;
    // `from_static` panics only on invalid header bytes, which these
    // literals are not. Doing this once at startup means we don't pay
    // header-construction overhead per request.
    headers.insert(
        reqwest::header::HeaderName::from_static("http-referer"),
        reqwest::header::HeaderValue::from_static("https://github.com/BrokkAi/brokk"),
    );
    headers.insert(
        reqwest::header::HeaderName::from_static("x-title"),
        reqwest::header::HeaderValue::from_static("anvil"),
    );

    // OpenRouter supports the unified reasoning object, so enable it.
    Some(Arc::new(llm_client::OpenAiClient::with_openrouter_support(
        discovery::OPENROUTER_BASE_URL.to_string(),
        Some(key.to_string()),
        headers,
    )))
}

/// Build the OpenRouter chat backend from the first available credential
/// source. Auth posture mirrors Codex: zero-config -- if neither the env
/// var nor the on-disk file holds a usable key the backend is skipped
/// silently, no CLI flag required.
///
/// Precedence is env > file: an explicit `OPENROUTER_API_KEY=...` in the
/// shell that launched the server overrides a stale on-disk key, so a
/// user rotating their key in a shell session doesn't have to remember
/// to `/setup openrouter key <key>` first. The on-disk file (written by
/// setup mid-session) is the persistent fallback for
/// the common case of starting the server without env vars.
fn build_openrouter_backend() -> Option<Arc<dyn LlmBackend>> {
    if let Ok(raw) = std::env::var(discovery::OPENROUTER_API_KEY_ENV) {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            tracing::info!(
                "{} is set but empty; falling back to {}",
                discovery::OPENROUTER_API_KEY_ENV,
                openrouter_auth::auth_path()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|_| "on-disk credential file".to_string())
            );
        } else {
            tracing::info!(
                "OpenRouter backend wired from {} at {} (chat + discovery); key length={}",
                discovery::OPENROUTER_API_KEY_ENV,
                discovery::OPENROUTER_BASE_URL,
                trimmed.len()
            );
            return openrouter_backend_from_key(trimmed);
        }
    }

    match openrouter_auth::read() {
        Ok(Some(auth)) => {
            let trimmed = auth.api_key.trim();
            if trimmed.is_empty() {
                tracing::info!(
                    "OpenRouter credential file exists but contains an empty key; backend skipped"
                );
                return None;
            }
            tracing::info!(
                "OpenRouter backend wired from on-disk credentials at {} (chat + discovery); key length={}",
                discovery::OPENROUTER_BASE_URL,
                trimmed.len()
            );
            openrouter_backend_from_key(trimmed)
        }
        Ok(None) => None,
        Err(e) => {
            tracing::warn!("failed to read OpenRouter credential file: {e:#}");
            None
        }
    }
}

fn build_openai_compatible_backend() -> Result<Option<Arc<dyn LlmBackend>>> {
    let config = openai_providers::read()?;
    if config.is_empty() {
        tracing::info!(
            "No generic OpenAI-compatible providers configured at {}; create providers.json to enable them.",
            openai_providers::path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "the Brokk config directory".to_string())
        );
        return Ok(None);
    }
    for profile in &config.profiles {
        tracing::info!(
            profile = %profile.name,
            base_url = %profile.base_url,
            auth = profile.api_key_env.as_deref().unwrap_or("none"),
            "Generic OpenAI-compatible provider profile configured"
        );
    }
    Ok(openai_providers::build_backend(config))
}

fn build_bedrock_backend(
    catalog_mode: setup_state::BedrockCatalogMode,
) -> Option<Arc<dyn LlmBackend>> {
    let backend: Arc<dyn LlmBackend> = match bedrock_client::backend_config() {
        Ok(Some((token, region, model))) => {
            Arc::new(bedrock_client::BedrockClient::new_with_catalog_mode(
                token,
                region,
                model,
                catalog_mode,
            ))
        }
        Ok(None) => return None,
        Err(e) => {
            tracing::warn!("failed to read Bedrock credentials: {e:#}");
            return None;
        }
    };
    let region = bedrock_client::region_from_env();
    let model = bedrock_client::model_from_env();
    let state = bedrock_auth::CredentialState::snapshot();
    tracing::info!(
        "Bedrock backend wired from {} at region {region}; default model {model}",
        state.active_source()
    );
    Some(backend)
}

fn main() {
    // The ACP connection + model-discovery path recurses deeply enough to
    // overflow the 1MB main-thread stack Windows gives executables (the
    // macOS/Linux main-thread default is 8MB, which is why only Windows hit
    // the overflow). Host the tokio runtime on a dedicated 8MB-stack thread
    // so the whole agent run gets the same headroom everywhere.
    let handle = std::thread::Builder::new()
        .name("anvil-main".to_string())
        .stack_size(ANVIL_MAIN_STACK_BYTES)
        .spawn(anvil_run)
        .expect("failed to spawn anvil main thread");
    match handle.join() {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            eprintln!("Error: {error:?}");
            std::process::exit(1);
        }
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

/// Stack for the Anvil server thread, in bytes. 8MB matches the macOS/Linux
/// main-thread default so behavior is uniform across platforms.
const ANVIL_MAIN_STACK_BYTES: usize = 8 * 1024 * 1024;

fn anvil_run() -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(anvil_main())
}

async fn anvil_main() -> Result<()> {
    // Configure tracing to stderr only (stdout is reserved for JSON-RPC)
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    if args.print.is_some() && args.command.is_some() {
        anyhow::bail!("--print cannot be combined with a subcommand");
    }

    if let Some(Command::Install(install_args)) = &args.command {
        installer::install(install_args)?;
        return Ok(());
    }
    if let Some(Command::Infer(infer_args)) = &args.command {
        return infer::run(infer_args).await;
    }

    // Install the parser sandbox before any code that might load a SKILL.md
    // (or, eventually, parse AGENTS.md / session zips / regex queries) so
    // every parse goes through the chosen backend from the first call.
    // The OS sandbox is preferred when available. Otherwise wasm is the
    // parser-sandbox fallback unless `--no-wasm-sandbox` explicitly opts out.
    // Determine sandbox strategy: OS sandbox (preferred) or wasm fallback
    let os_available = tools::sandbox::is_os_sandbox_available();
    let strategy =
        crate::sandbox_backend::SandboxBackend::detect(os_available, args.no_wasm_sandbox);
    match &strategy {
        crate::sandbox_backend::SandboxBackend::OsNative if os_available => {
            tracing::info!("sandbox strategy: OsNative (OS sandbox + native parsing)");
        }
        crate::sandbox_backend::SandboxBackend::OsNative
            if !crate::sandbox_backend::wasm_sandbox_compiled() =>
        {
            tracing::info!(
                "sandbox strategy: OsNative (no OS sandbox available, wasm sandbox support not compiled into this build)"
            );
        }
        crate::sandbox_backend::SandboxBackend::OsNative => {
            tracing::info!(
                "sandbox strategy: OsNative (no OS sandbox available, wasm disabled by flag)"
            );
        }
        crate::sandbox_backend::SandboxBackend::WasmFallback(_) => {
            tracing::info!("sandbox strategy: WasmFallback (no OS sandbox; parsing through wasm)");
        }
    }
    sandbox_backend::install_global(strategy);

    if args.endpoint_url.is_some() {
        tracing::warn!(
            "--endpoint-url is deprecated and ignored. Ollama is probed at \
             http://localhost:11434; if your daemon listens elsewhere, run it on the \
             default port. Re-run `brokk install` to refresh your editor config."
        );
    }
    if args.api_key.is_some() {
        tracing::warn!(
            "--api-key (env BROKK_ENDPOINT_API_KEY) is deprecated and ignored. \
             Codex credentials are read from ~/.codex/auth.json; Ollama does not use a key."
        );
    }
    if args.use_codex {
        tracing::warn!(
            "--use-codex is deprecated and has no effect. Codex is auto-detected when \
             ~/.codex/auth.json exists, alongside Ollama."
        );
    }
    if args.bifrost_binary.is_some() {
        tracing::warn!(
            "--bifrost-binary is deprecated and ignored. Anvil now manages a pinned local \
             Bifrost MCP server; use `/mcp` to view or change MCP server configuration."
        );
    }

    {
        let legacy = crate::setup_state::read();
        if !legacy.always_allow.is_empty() {
            tracing::warn!(
                count = legacy.always_allow.len(),
                "setup.json contains install-wide Always allow approvals that are no longer \
                 used. Per-repo approvals are now stored in .brokk/permissions.json inside \
                 each repository. Re-approve the tools you want in each repository.",
            );
        }
    }

    match crate::mcp::ensure_bundled_bifrost().await {
        Ok(path) => tracing::info!(
            bifrost = %path.display(),
            version = crate::mcp::BUNDLED_BIFROST_VERSION,
            "bundled bifrost ready"
        ),
        Err(e) => tracing::warn!(
            version = crate::mcp::BUNDLED_BIFROST_VERSION,
            error = %e,
            "failed to prepare bundled bifrost; built-in Bifrost MCP tools may be unavailable"
        ),
    }

    // Fold any pre-consolidation per-provider credential files into the
    // single secrets store before the backends read their credentials.
    secrets::migrate_legacy_files();

    let bedrock_catalog_mode = if args.transient_setup {
        setup_state::BedrockCatalogMode::default()
    } else {
        setup_state::bedrock_catalog_mode()
    };
    let bedrock_backend = build_bedrock_backend(bedrock_catalog_mode);
    let codex_backend = build_codex_backend().await;
    let deepseek_backend = build_deepseek_backend();
    let kimi_backend = build_kimi_backend();
    let openai_backend = build_openai_compatible_backend()?;
    let openrouter_backend = build_openrouter_backend();
    let ollama_backend = Some(build_ollama_backend());

    if bedrock_backend.is_none() {
        tracing::info!(
            "Bedrock backend not available; set {} or run `/setup bedrock key <token>` from a session to enable it.",
            bedrock_client::BEDROCK_API_KEY_ENV
        );
    }
    if codex_backend.is_none() {
        tracing::info!(
            "Codex backend not available; the picker will fall back to Ollama \
             and hosted providers (if discovered). Run /setup codex from a session to add \
             Codex -- the new credentials are picked up on the next discovery \
             refresh, no restart required."
        );
    }
    if deepseek_backend.is_none() {
        tracing::info!(
            "DeepSeek backend not available; set {} or run `/setup deepseek key <key>` \
             from a session to enable hosted DeepSeek.",
            discovery::DEEPSEEK_API_KEY_ENV
        );
    }
    if kimi_backend.is_none() {
        tracing::info!(
            "Kimi backend not available; set {} or run `kimi login` to create {}.",
            kimi_auth::KIMI_API_KEY_ENV,
            kimi_auth::credentials_path()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|_| "the Kimi Code credential file".to_string())
        );
    }
    if openrouter_backend.is_none() {
        tracing::info!(
            "OpenRouter backend not available; set {} or run `/setup openrouter key <key>` \
             from a session to enable it.",
            discovery::OPENROUTER_API_KEY_ENV
        );
    }

    let llm: Arc<MultiBackend> = Arc::new(MultiBackend::new(vec![
        BackendRegistration::new(discovery::ModelSource::BEDROCK, "Bedrock", bedrock_backend),
        BackendRegistration::new(discovery::ModelSource::CODEX, "Codex", codex_backend),
        BackendRegistration::new(
            discovery::ModelSource::OLLAMA,
            "Local models",
            ollama_backend,
        ),
        BackendRegistration::new(discovery::ModelSource::DS4, "ds4", None),
        BackendRegistration::new(
            discovery::ModelSource::DEEPSEEK,
            "DeepSeek",
            deepseek_backend,
        ),
        BackendRegistration::new(discovery::ModelSource::KIMI, "Kimi", kimi_backend),
        BackendRegistration::new(
            discovery::ModelSource::OPENAI,
            "OpenAI-compatible",
            openai_backend,
        ),
        BackendRegistration::new(
            discovery::ModelSource::OPENROUTER,
            "OpenRouter",
            openrouter_backend,
        ),
    ]));

    // Kick off model discovery eagerly so any provider errors ("skipped"
    // log lines with HTTP status codes) appear immediately in the startup
    // log rather than waiting for the first client session to connect.
    {
        let llm = llm.clone();
        tokio::spawn(async move {
            match llm.list_model_metadata().await {
                Ok(models) => tracing::info!("startup discovery: {} model(s) found", models.len()),
                Err(e) => tracing::warn!("startup discovery failed: {e:#}"),
            }
        });
    }

    let limits = session::SessionLimits {
        max_sessions: args.max_sessions,
        max_history_turns: args.max_history_turns,
    };
    let sessions = session::SessionStore::with_limits_and_transient_setup(
        args.default_model,
        limits,
        args.transient_setup,
    )
    .with_shell_minimizer(!args.no_shell_minimizer);
    sessions
        .set_default_reasoning_effort(args.reasoning_effort)
        .await;

    // `0` means "no turn cap" (matching `--max-sessions`/`--max-history-turns`):
    // map it to the max so the `for turn in 0..turn_limit` loop is bounded only
    // by the model's own completion signal, the stream timeouts, and the nudges.
    let max_turns = if args.max_turns == 0 {
        usize::MAX
    } else {
        args.max_turns
    };

    // Headless one-shot mode (#356): drive the same agent component with the
    // built-in ACP client instead of serving stdio.
    if let Some(prompt_arg) = args.print {
        return headless::run(
            headless::RunConfig {
                prompt_arg,
                output_format: args.output_format,
                permission_mode: args
                    .permission_mode
                    .unwrap_or(headless::PermissionMode::Manual),
                cwd: args.cwd,
                model: args.model,
                resume: args.resume,
            },
            llm,
            sessions,
            max_turns,
            args.llm_idle_timeout_secs,
            args.llm_stall_timeout_secs,
        )
        .await;
    }

    // HTTP daemon mode shares the backend registrations, SessionStore, and
    // turn limits built above with the ACP path, so both transports use one
    // runtime and persistence implementation (#317, #318).
    #[cfg(feature = "http-api")]
    if let Some(Command::Serve(serve_args)) = args.command {
        return http_api::serve(
            serve_args,
            llm,
            sessions,
            max_turns,
            args.llm_idle_timeout_secs,
            args.llm_stall_timeout_secs,
        )
        .await;
    }
    // Bounds on the LLM timeout values are enforced by the clap
    // `value_parser`, so the values reach us already validated.
    acp::run_agent(
        llm,
        sessions,
        max_turns,
        args.llm_idle_timeout_secs,
        args.llm_stall_timeout_secs,
    )
    .await
    .map_err(|e| {
        tracing::error!("agent error: {e}");
        anyhow::anyhow!("agent error: {e}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore = "live network smoke test; requires DEEPSEEK_API_KEY"]
    async fn deepseek_backend_lists_models_live() {
        let key = std::env::var(discovery::DEEPSEEK_API_KEY_ENV)
            .expect("DEEPSEEK_API_KEY must be set for the live smoke test");
        let backend =
            deepseek_backend_from_key(&key).expect("non-empty DEEPSEEK_API_KEY should build");
        let models = backend
            .list_models()
            .await
            .expect("hosted DeepSeek list_models should succeed");
        assert!(
            models.iter().any(|id| id.contains("deepseek")),
            "expected at least one DeepSeek model id, got {models:?}"
        );
    }
}
