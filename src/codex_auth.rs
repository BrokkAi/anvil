//! "Sign in with ChatGPT" support using OpenAI's PKCE OAuth flow.
//!
//! On-disk format (`~/.codex/auth.json`) is intentionally compatible with
//! Codex CLI: a user who logs in here can use `codex` in the same terminal
//! and vice versa.
//!
//! Trust posture: we use `oauth2` (32M+ downloads, depended on by Servo
//! among others) for the standard PKCE + refresh state machine. The one
//! step `oauth2` doesn't cover -- RFC 8693 token exchange that converts
//! the ChatGPT id_token into a regular `OPENAI_API_KEY` -- is a single
//! POST written by hand below. JWT decoding for `chatgpt_account_id`
//! is also inline (~15 lines): we trust the token because we just
//! received it over HTTPS from auth.openai.com, and we only need an
//! unverified claim.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;
use std::{future::Future, time::Instant};

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use chrono::{DateTime, Utc};
// oauth2 supplies the PKCE + CSRF primitives only -- the actual token
// exchanges are issued via `reqwest` below so we don't have to dance
// with `oauth2::ExtraTokenFields` generics just to read `id_token`.
use oauth2::{CsrfToken, PkceCodeChallenge};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::llm_client::OpenAiClient;

const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const AUTH_ISSUER: &str = "https://auth.openai.com";
const AUTH_URL: &str = "https://auth.openai.com/oauth/authorize";
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CALLBACK_PORT: u16 = 1455;
const REDIRECT_URI: &str = "http://localhost:1455/auth/callback";

/// OAuth scopes Codex CLI requests during ChatGPT login. The two
/// `api.connectors.*` scopes plus `id_token_add_organizations` /
/// `codex_cli_simplified_flow` flags below are what flips the consent
/// page from the legacy "API organization access" screen (the one that
/// says "Codex will receive a token to generate API keys") to the
/// ChatGPT-subscription consent screen. Drop any of them and OpenAI
/// silently routes us back to the API-key flow.
const SCOPES: &[&str] = &[
    "openid",
    "profile",
    "email",
    "offline_access",
    "api.connectors.read",
    "api.connectors.invoke",
];

/// Identifies our requests as Codex CLI to OpenAI's auth and Responses
/// servers. The ChatGPT-subscription consent UI and the
/// `chatgpt.com/backend-api/codex/responses` gateway both gate on this
/// value -- using anything else (or omitting it) is what produced the
/// "API organization access" screen reported in #3540's review.
const CODEX_ORIGINATOR: &str = "codex_cli_rs";

/// Refresh proactively if the stored credentials are older than this.
/// Codex CLI uses an 8-day window; we follow suit.
const REFRESH_AFTER: chrono::Duration = chrono::Duration::days(8);

/// Wait at most this long for the user to complete sign-in in the browser.
const CALLBACK_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const AUTH_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const AUTH_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_AUTH_RESPONSE_BYTES: usize = 256 * 1024;
const MAX_DEVICE_INTERVAL_SECS: u64 = 60;

/// The device authorization issued by OpenAI is valid for 15 minutes. Keep
/// this in lockstep with Codex CLI so headless Anvil logins have the same
/// completion window.
const DEVICE_AUTH_TIMEOUT: Duration = Duration::from_secs(15 * 60);

#[derive(Debug, Clone)]
pub struct DeviceCodePrompt {
    pub verification_url: String,
    pub user_code: String,
}

#[derive(Debug)]
struct DeviceCode {
    prompt: DeviceCodePrompt,
    device_auth_id: String,
    interval: Duration,
}

#[derive(Debug, Deserialize)]
struct DeviceUserCodeResponse {
    device_auth_id: String,
    #[serde(alias = "user_code", alias = "usercode")]
    user_code: String,
    #[serde(default, deserialize_with = "deserialize_device_interval")]
    interval: u64,
}

#[derive(Debug, Deserialize)]
struct DeviceTokenResponse {
    authorization_code: String,
    code_verifier: String,
}

#[derive(Debug, Serialize)]
struct DeviceUserCodeRequest<'a> {
    client_id: &'a str,
}

#[derive(Debug, Serialize)]
struct DeviceTokenRequest<'a> {
    device_auth_id: &'a str,
    user_code: &'a str,
}

#[derive(Debug, Clone)]
struct DeviceAuthConfig {
    issuer: String,
    token_url: String,
    client_id: String,
    timeout: Duration,
}

impl DeviceAuthConfig {
    fn production() -> Self {
        Self {
            issuer: AUTH_ISSUER.to_string(),
            token_url: TOKEN_URL.to_string(),
            client_id: CLIENT_ID.to_string(),
            timeout: DEVICE_AUTH_TIMEOUT,
        }
    }

    fn api_base_url(&self) -> String {
        format!("{}/api/accounts", self.issuer.trim_end_matches('/'))
    }

    fn verification_url(&self) -> String {
        format!("{}/codex/device", self.issuer.trim_end_matches('/'))
    }

    fn redirect_uri(&self) -> String {
        format!("{}/deviceauth/callback", self.issuer.trim_end_matches('/'))
    }
}

fn deserialize_device_interval<'de, D>(deserializer: D) -> std::result::Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::String(value) => value.parse().map_err(serde::de::Error::custom),
        serde_json::Value::Number(value) => value
            .as_u64()
            .ok_or_else(|| serde::de::Error::custom("device interval must be an unsigned integer")),
        _ => Err(serde::de::Error::custom(
            "device interval must be a string or unsigned integer",
        )),
    }
}

#[cfg(test)]
thread_local! {
    static TEST_CODEX_HOME: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
struct TestCodexHomeScope {
    prev: Option<PathBuf>,
}

#[cfg(test)]
impl TestCodexHomeScope {
    fn set(path: PathBuf) -> Self {
        let prev = TEST_CODEX_HOME.with(|slot| slot.borrow().clone());
        TEST_CODEX_HOME.with(|slot| *slot.borrow_mut() = Some(path));
        Self { prev }
    }
}

#[cfg(test)]
impl Drop for TestCodexHomeScope {
    fn drop(&mut self) {
        TEST_CODEX_HOME.with(|slot| *slot.borrow_mut() = self.prev.take());
    }
}

/// Schema of `~/.codex/auth.json`. Field names (and the `OPENAI_API_KEY`
/// SHOUTING case) are dictated by Codex CLI's storage format -- do not
/// rename without breaking cross-compat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthDotJson {
    #[serde(default)]
    pub auth_mode: Option<String>,
    #[serde(
        rename = "OPENAI_API_KEY",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub openai_api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<TokenData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_refresh: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenData {
    pub id_token: String,
    pub access_token: String,
    pub refresh_token: String,
    pub account_id: String,
}

/// Resolve `~/.codex/auth.json`. Honours `$CODEX_HOME` if set, matching
/// Codex CLI's override convention.
pub fn auth_json_path() -> Result<PathBuf> {
    #[cfg(test)]
    if let Some(custom) = TEST_CODEX_HOME.with(|slot| slot.borrow().clone()) {
        return Ok(custom.join("auth.json"));
    }
    if let Ok(custom) = std::env::var("CODEX_HOME") {
        return Ok(PathBuf::from(custom).join("auth.json"));
    }
    let home = dirs::home_dir().ok_or_else(|| anyhow!("could not resolve home directory"))?;
    Ok(home.join(".codex").join("auth.json"))
}

pub fn read_auth_dot_json() -> Result<Option<AuthDotJson>> {
    let path = auth_json_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let parsed = serde_json::from_slice::<AuthDotJson>(&bytes)
        .with_context(|| format!("parsing {}", path.display()))?;
    Ok(Some(parsed))
}

/// Atomic write: stage to `auth.json.tmp` in the same directory, then
/// rename. Keeps the credential file from ever being half-written if the
/// process is interrupted mid-flush.
pub fn write_auth_dot_json(auth: &AuthDotJson) -> Result<()> {
    let path = auth_json_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let tmp = path.with_extension(format!("tmp.{}", uuid::Uuid::new_v4()));
    let bytes = serde_json::to_vec_pretty(auth).context("serializing AuthDotJson")?;
    std::fs::write(&tmp, &bytes).with_context(|| format!("writing {}", tmp.display()))?;
    set_user_only_perms(&tmp)?;
    std::fs::rename(&tmp, &path)
        .with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

#[cfg(unix)]
fn set_user_only_perms(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("chmod 600 {}", path.display()))
}

#[cfg(not(unix))]
fn set_user_only_perms(_path: &Path) -> Result<()> {
    Ok(())
}

/// Standard OAuth2 token endpoint response, with the OIDC `id_token`
/// extension that ChatGPT's auth server returns alongside the regular
/// access/refresh tokens.
#[derive(Debug, Deserialize)]
struct OidcTokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    id_token: Option<String>,
}

fn build_authorize_url(challenge: &PkceCodeChallenge, state: &str) -> String {
    // OAuth2 RFC 6749 + PKCE RFC 7636 plus three Codex-specific flags
    // that Codex CLI passes verbatim:
    //   * `id_token_add_organizations=true` -> the auth server includes
    //     the user's organization claim in the issued id_token, which
    //     downstream code uses to derive `chatgpt_account_id`.
    //   * `codex_cli_simplified_flow=true` -> tells OpenAI to render the
    //     ChatGPT-subscription consent screen ("Sign in with ChatGPT")
    //     instead of the legacy API-organization screen. Omitting this
    //     was the bug behind the "Codex requests access to your API
    //     organization" page users were seeing.
    //   * `originator=codex_cli_rs` -> identifies us as Codex CLI both
    //     to the consent UI and to the chatgpt.com/backend-api gateway.
    // Scopes already cover `api.connectors.*`; see SCOPES above for why
    // those are required for the simplified flow.
    let scopes_encoded = SCOPES
        .iter()
        .map(|s| urlencode(s))
        .collect::<Vec<_>>()
        .join("%20");
    format!(
        "{AUTH_URL}?response_type=code\
         &client_id={CLIENT_ID}\
         &redirect_uri={redirect}\
         &scope={scopes_encoded}\
         &code_challenge={code_challenge}\
         &code_challenge_method=S256\
         &id_token_add_organizations=true\
         &codex_cli_simplified_flow=true\
         &state={state}\
         &originator={originator}",
        redirect = urlencode(REDIRECT_URI),
        state = urlencode(state),
        code_challenge = challenge.as_str(),
        originator = urlencode(CODEX_ORIGINATOR),
    )
}

/// Minimal percent-encoder for the handful of characters that show up
/// in our query values (`/`, `:`, `=`, etc.). Avoids pulling in `url`
/// just to call `Url::parse`/`form_urlencoded::byte_serialize`.
pub(crate) fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn http_client() -> Result<reqwest::Client> {
    http_client_for(TOKEN_URL)
}

fn http_client_for(url: &str) -> Result<reqwest::Client> {
    OpenAiClient::apply_runtime_tls_workarounds(
        reqwest::Client::builder()
            .connect_timeout(AUTH_CONNECT_TIMEOUT)
            .timeout(AUTH_REQUEST_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none()),
        url,
    )
    .build()
    .context("building reqwest client")
}

async fn await_or_cancel<T, F>(cancel: Option<&CancellationToken>, future: F) -> Result<T>
where
    F: Future<Output = Result<T>>,
{
    match cancel {
        Some(cancel) => {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => bail!("sign-in was cancelled"),
                result = future => result,
            }
        }
        None => future.await,
    }
}

async fn exchange_code_for_tokens(
    http: &reqwest::Client,
    code: &str,
    verifier: &str,
) -> Result<OidcTokenResponse> {
    exchange_code_for_tokens_at(http, TOKEN_URL, CLIENT_ID, REDIRECT_URI, code, verifier).await
}

async fn exchange_code_for_tokens_at(
    http: &reqwest::Client,
    token_url: &str,
    client_id: &str,
    redirect_uri: &str,
    code: &str,
    verifier: &str,
) -> Result<OidcTokenResponse> {
    post_token_form_at(
        http,
        token_url,
        &[
            ("grant_type", "authorization_code"),
            ("client_id", client_id),
            ("code", code),
            ("code_verifier", verifier),
            ("redirect_uri", redirect_uri),
        ],
    )
    .await
}

async fn exchange_refresh_token(
    http: &reqwest::Client,
    refresh_token: &str,
) -> Result<OidcTokenResponse> {
    post_token_form(
        http,
        &[
            ("grant_type", "refresh_token"),
            ("client_id", CLIENT_ID),
            ("refresh_token", refresh_token),
            // Ask the server to keep issuing an id_token alongside the
            // new access/refresh pair so we can re-derive the API key.
            ("scope", &SCOPES.join(" ")),
        ],
    )
    .await
}

async fn post_token_form(
    http: &reqwest::Client,
    form: &[(&str, &str)],
) -> Result<OidcTokenResponse> {
    post_token_form_at(http, TOKEN_URL, form).await
}

async fn post_token_form_at(
    http: &reqwest::Client,
    token_url: &str,
    form: &[(&str, &str)],
) -> Result<OidcTokenResponse> {
    let resp = http
        .post(token_url)
        .form(form)
        .send()
        .await
        .context("token endpoint POST failed")?;
    let status = resp.status();
    if !status.is_success() {
        let body = read_response_text_bounded(resp).await.unwrap_or_default();
        bail!(
            "token endpoint returned HTTP {status}: {}",
            bounded_error_body(&body)
        );
    }
    let body = read_response_bytes_bounded(resp).await?;
    serde_json::from_slice::<OidcTokenResponse>(&body).context("parsing token endpoint response")
}

/// Run the full ChatGPT login flow using the default presenter (open the
/// system browser): PKCE in the browser, capture the callback locally,
/// exchange the `id_token` for an API key, and persist.
pub async fn interactive_login() -> Result<AuthDotJson> {
    interactive_login_with(|auth_url| async move {
        if let Err(err) = webbrowser::open(&auth_url) {
            eprintln!("Could not open a browser automatically ({err}). Visit:\n  {auth_url}");
        } else {
            eprintln!("Opened browser for ChatGPT sign-in. Waiting for callback...");
        }
        Ok(())
    })
    .await
}

/// Like [`interactive_login`], but the caller decides how to present the
/// authorize URL to the user (open a browser, hand it to a client UI via ACP
/// URL-mode elicitation, ...). `present` is awaited before the loopback
/// listener begins; returning `Err` aborts the login (e.g. the user declined)
/// and leaves stored credentials untouched.
pub async fn interactive_login_with<F, Fut>(present: F) -> Result<AuthDotJson>
where
    F: FnOnce(String) -> Fut,
    Fut: std::future::Future<Output = Result<()>>,
{
    let (challenge, verifier) = PkceCodeChallenge::new_random_sha256();
    let csrf = CsrfToken::new_random();
    let auth_url = build_authorize_url(&challenge, csrf.secret());

    present(auth_url).await?;

    let expected_state = csrf.secret().clone();
    let code = tokio::task::spawn_blocking(move || {
        loopback_capture_code(CALLBACK_PORT, expected_state, CALLBACK_TIMEOUT)
    })
    .await
    .context("loopback capture task panicked")??;

    let http = http_client()?;
    let token = exchange_code_for_tokens(&http, &code, verifier.secret()).await?;

    finalize_chatgpt_login(&http, TOKEN_URL, CLIENT_ID, token, None).await
}

/// Run the official Codex device authorization flow used for headless and
/// remote clients. The presenter receives a short verification URL and a
/// one-time code; no loopback listener or SSH port forwarding is required.
pub async fn interactive_device_login_with<F, Fut>(
    cancel: &CancellationToken,
    present: F,
) -> Result<AuthDotJson>
where
    F: FnOnce(DeviceCodePrompt) -> Fut,
    Fut: Future<Output = Result<()>>,
{
    interactive_device_login_with_config(DeviceAuthConfig::production(), cancel, present).await
}

async fn interactive_device_login_with_config<F, Fut>(
    config: DeviceAuthConfig,
    cancel: &CancellationToken,
    present: F,
) -> Result<AuthDotJson>
where
    F: FnOnce(DeviceCodePrompt) -> Fut,
    Fut: Future<Output = Result<()>>,
{
    let http = http_client_for(&config.token_url)?;
    let device_code = request_device_code(&http, &config, cancel).await?;
    await_or_cancel(Some(cancel), async {
        present(device_code.prompt.clone()).await
    })
    .await?;

    let authorized = poll_device_code(&http, &config, &device_code, cancel).await?;
    let redirect_uri = config.redirect_uri();
    let token = await_or_cancel(Some(cancel), async {
        exchange_code_for_tokens_at(
            &http,
            &config.token_url,
            &config.client_id,
            &redirect_uri,
            &authorized.authorization_code,
            &authorized.code_verifier,
        )
        .await
    })
    .await?;

    finalize_chatgpt_login(
        &http,
        &config.token_url,
        &config.client_id,
        token,
        Some(cancel),
    )
    .await
}

async fn request_device_code(
    http: &reqwest::Client,
    config: &DeviceAuthConfig,
    cancel: &CancellationToken,
) -> Result<DeviceCode> {
    let url = format!("{}/deviceauth/usercode", config.api_base_url());
    let response = await_or_cancel(Some(cancel), async {
        http.post(&url)
            .json(&DeviceUserCodeRequest {
                client_id: &config.client_id,
            })
            .send()
            .await
            .context("device-code request failed")
    })
    .await?;
    let status = response.status();
    if !status.is_success() {
        let body = await_or_cancel(Some(cancel), read_response_text_bounded(response))
            .await
            .unwrap_or_default();
        bail!(
            "device-code request returned HTTP {status}: {}",
            bounded_error_body(&body)
        );
    }
    let body = await_or_cancel(Some(cancel), read_response_bytes_bounded(response)).await?;
    let parsed = serde_json::from_slice::<DeviceUserCodeResponse>(&body)
        .context("parsing device-code response")?;
    if parsed.device_auth_id.trim().is_empty() || parsed.user_code.trim().is_empty() {
        bail!("device-code response omitted the authorization id or user code");
    }
    Ok(DeviceCode {
        prompt: DeviceCodePrompt {
            verification_url: config.verification_url(),
            user_code: parsed.user_code,
        },
        device_auth_id: parsed.device_auth_id,
        interval: Duration::from_secs(parsed.interval.clamp(1, MAX_DEVICE_INTERVAL_SECS)),
    })
}

async fn poll_device_code(
    http: &reqwest::Client,
    config: &DeviceAuthConfig,
    device_code: &DeviceCode,
    cancel: &CancellationToken,
) -> Result<DeviceTokenResponse> {
    let url = format!("{}/deviceauth/token", config.api_base_url());
    let started = Instant::now();
    loop {
        if started.elapsed() >= config.timeout {
            bail!("device authorization timed out after {:?}", config.timeout);
        }
        let response = await_or_cancel(Some(cancel), async {
            http.post(&url)
                .json(&DeviceTokenRequest {
                    device_auth_id: &device_code.device_auth_id,
                    user_code: &device_code.prompt.user_code,
                })
                .send()
                .await
                .context("device authorization poll failed")
        })
        .await?;
        let status = response.status();
        if status.is_success() {
            let body = await_or_cancel(Some(cancel), read_response_bytes_bounded(response)).await?;
            let parsed = serde_json::from_slice::<DeviceTokenResponse>(&body)
                .context("parsing device authorization response")?;
            if parsed.authorization_code.trim().is_empty() || parsed.code_verifier.trim().is_empty()
            {
                bail!("device authorization response omitted the authorization code or verifier");
            }
            return Ok(parsed);
        }
        if status != reqwest::StatusCode::FORBIDDEN && status != reqwest::StatusCode::NOT_FOUND {
            let body = read_response_text_bounded(response)
                .await
                .unwrap_or_default();
            bail!(
                "device authorization failed (HTTP {status}): {}",
                bounded_error_body(&body)
            );
        }

        let remaining = config.timeout.saturating_sub(started.elapsed());
        let delay = device_code.interval.min(remaining);
        await_or_cancel(Some(cancel), async {
            tokio::time::sleep(delay).await;
            Ok(())
        })
        .await?;
    }
}

fn bounded_error_body(body: &str) -> &str {
    const MAX_ERROR_BODY_BYTES: usize = 2048;
    if body.len() <= MAX_ERROR_BODY_BYTES {
        body
    } else {
        let mut end = MAX_ERROR_BODY_BYTES;
        while !body.is_char_boundary(end) {
            end -= 1;
        }
        &body[..end]
    }
}

async fn read_response_bytes_bounded(mut response: reqwest::Response) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.context("reading auth response")? {
        if body.len().saturating_add(chunk.len()) > MAX_AUTH_RESPONSE_BYTES {
            bail!("auth response exceeded {MAX_AUTH_RESPONSE_BYTES} bytes");
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

async fn read_response_text_bounded(response: reqwest::Response) -> Result<String> {
    let body = read_response_bytes_bounded(response).await?;
    String::from_utf8(body).context("auth response was not UTF-8")
}

async fn finalize_chatgpt_login(
    http: &reqwest::Client,
    token_url: &str,
    client_id: &str,
    token: OidcTokenResponse,
    cancel: Option<&CancellationToken>,
) -> Result<AuthDotJson> {
    let id_token = token
        .id_token
        .clone()
        .ok_or_else(|| anyhow!("OAuth response missing id_token; cannot derive API key"))?;
    let refresh_token = token
        .refresh_token
        .clone()
        .ok_or_else(|| anyhow!("OAuth response missing refresh_token"))?;

    // Token-exchange (RFC 8693 id_token -> sk-...) is best-effort: a
    // ChatGPT-subscription user with no associated API organization
    // gets `Invalid ID token: missing organization_id` here, which is
    // expected -- they don't *have* an API key to derive. Codex CLI's
    // own `obtain_api_key(...).await.ok()` does the same thing.
    // Subscription routing only needs the access_token + account_id we
    // already have; the API key is stored only as a convenience for
    // users who later run `codex` itself in apikey mode.
    let api_key = match await_or_cancel(
        cancel,
        token_exchange_id_token(http, token_url, client_id, &id_token),
    )
    .await
    {
        Ok(key) => Some(key),
        Err(e) => {
            if cancel.is_some_and(CancellationToken::is_cancelled) {
                return Err(e);
            }
            tracing::info!(
                "skipping API key derivation (typical for ChatGPT-only accounts): {e:#}"
            );
            None
        }
    };
    let account_id = extract_chatgpt_account_id(&token.access_token)?;

    let auth = AuthDotJson {
        auth_mode: Some("chatgpt".to_string()),
        openai_api_key: api_key,
        tokens: Some(TokenData {
            id_token,
            access_token: token.access_token,
            refresh_token,
            account_id,
        }),
        last_refresh: Some(Utc::now()),
    };
    if cancel.is_some_and(CancellationToken::is_cancelled) {
        bail!("sign-in was cancelled");
    }
    write_auth_dot_json(&auth)?;
    Ok(auth)
}

/// True when `auth.last_refresh` is older than `REFRESH_AFTER`. Mirrors
/// `refresh_if_stale`'s "no last_refresh => not stale" branch so callers
/// can cheaply skip the refresh lock when nothing needs to happen.
pub fn is_stale(auth: &AuthDotJson) -> bool {
    match auth.last_refresh {
        Some(ts) => Utc::now() - ts >= REFRESH_AFTER,
        None => false,
    }
}

/// Refresh the API key if the stored credentials are stale. Used on
/// startup so long-running ACP sessions don't 401 mid-flight.
pub async fn refresh_if_stale(auth: &mut AuthDotJson) -> Result<bool> {
    if !is_stale(auth) {
        return Ok(false);
    }
    let tokens = auth
        .tokens
        .as_ref()
        .ok_or_else(|| anyhow!("auth.json has no tokens to refresh"))?;
    let http = http_client()?;
    let refreshed = exchange_refresh_token(&http, &tokens.refresh_token).await?;

    let id_token = refreshed
        .id_token
        .clone()
        .ok_or_else(|| anyhow!("refresh response missing id_token"))?;
    let refresh_token = refreshed
        .refresh_token
        .clone()
        .unwrap_or_else(|| tokens.refresh_token.clone());

    // Same best-effort posture as `interactive_login`: refresh stays
    // successful even if the user's account can't mint an OPENAI_API_KEY.
    let api_key = match token_exchange_id_token(&http, TOKEN_URL, CLIENT_ID, &id_token).await {
        Ok(key) => Some(key),
        Err(e) => {
            tracing::debug!("token-exchange skipped during refresh: {e:#}");
            // Preserve any previously-stored key rather than wiping
            // it -- a transient failure shouldn't drop an apikey-mode
            // user's working credentials.
            auth.openai_api_key.clone()
        }
    };
    let account_id = extract_chatgpt_account_id(&refreshed.access_token)?;

    *auth = AuthDotJson {
        auth_mode: Some("chatgpt".to_string()),
        openai_api_key: api_key,
        tokens: Some(TokenData {
            id_token,
            access_token: refreshed.access_token,
            refresh_token,
            account_id,
        }),
        last_refresh: Some(Utc::now()),
    };
    write_auth_dot_json(auth)?;
    Ok(true)
}

/// Best-effort logout: delete the stored credentials. We do not (yet)
/// hit the revoke endpoint -- that's a follow-up.
pub fn logout() -> Result<()> {
    let path = auth_json_path()?;
    if path.exists() {
        std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
    }
    Ok(())
}

/// RFC 8693 token exchange: convert an OpenID id_token into a regular
/// `OPENAI_API_KEY`. This is the same call Codex CLI makes after OAuth
/// completes; the response's `access_token` is the `sk-...` we want.
async fn token_exchange_id_token(
    http: &reqwest::Client,
    token_url: &str,
    client_id: &str,
    id_token: &str,
) -> Result<String> {
    #[derive(Deserialize)]
    struct Resp {
        access_token: String,
    }
    let resp = http
        .post(token_url)
        .form(&[
            (
                "grant_type",
                "urn:ietf:params:oauth:grant-type:token-exchange",
            ),
            ("client_id", client_id),
            ("requested_token", "openai-api-key"),
            ("subject_token", id_token),
            (
                "subject_token_type",
                "urn:ietf:params:oauth:token-type:id_token",
            ),
        ])
        .send()
        .await
        .context("token-exchange POST failed")?;
    let status = resp.status();
    if !status.is_success() {
        let body = read_response_text_bounded(resp).await.unwrap_or_default();
        bail!(
            "token-exchange failed (HTTP {status}): {}",
            bounded_error_body(&body)
        );
    }
    let body = read_response_bytes_bounded(resp).await?;
    let parsed: Resp = serde_json::from_slice(&body).context("parsing token-exchange response")?;
    Ok(parsed.access_token)
}

/// Pull the `chatgpt_account_id` claim out of an access_token JWT
/// without verifying the signature. We trust the token because we just
/// received it over HTTPS from auth.openai.com; the only thing we need
/// is an opaque account identifier.
pub fn extract_chatgpt_account_id(access_token: &str) -> Result<String> {
    let payload = access_token
        .split('.')
        .nth(1)
        .ok_or_else(|| anyhow!("access_token is not a JWT"))?;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload.as_bytes())
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(payload.as_bytes()))
        .context("base64-decoding JWT payload")?;
    let claims: serde_json::Value =
        serde_json::from_slice(&bytes).context("parsing JWT payload as JSON")?;
    claims
        .get("https://api.openai.com/auth")
        .and_then(|v| v.get("chatgpt_account_id"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("JWT missing https://api.openai.com/auth.chatgpt_account_id claim"))
}

/// Spin up `tiny_http` on `port`, wait for `GET /auth/callback?code=...&state=...`,
/// validate `state`, return the `code`. Times out after `timeout`.
fn loopback_capture_code(port: u16, expected_state: String, timeout: Duration) -> Result<String> {
    let server = tiny_http::Server::http(("127.0.0.1", port))
        .map_err(|e| anyhow!("binding loopback port {port}: {e}"))?;
    let (tx, rx) = mpsc::channel::<Result<String>>();

    std::thread::spawn(move || {
        // One callback is all we expect; pull the first request only and
        // drop the server when the closure returns.
        if let Some(req) = server.incoming_requests().next() {
            let url = req.url().to_string();
            let result = parse_callback_url(&url, &expected_state);
            let body = match &result {
                Ok(_) => "Sign-in complete. You can close this tab and return to brokk.",
                Err(e) => {
                    tracing::warn!("codex-login callback rejected: {e:#}");
                    "Sign-in failed. Check the brokk console for details."
                }
            };
            let _ = req.respond(tiny_http::Response::from_string(body));
            let _ = tx.send(result);
        }
    });

    rx.recv_timeout(timeout)
        .map_err(|_| anyhow!("OAuth callback timed out after {timeout:?}"))?
}

/// Parse `/auth/callback?code=...&state=...` (or `?error=...`) by hand
/// to avoid pulling in the `url` crate just for query-string splitting.
fn parse_callback_url(url: &str, expected_state: &str) -> Result<String> {
    let query = url.split('?').nth(1).unwrap_or("");
    let mut code: Option<String> = None;
    let mut state: Option<String> = None;
    let mut error: Option<String> = None;
    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (raw_k, raw_v) = pair.split_once('=').unwrap_or((pair, ""));
        let v = url_decode(raw_v);
        match raw_k {
            "code" => code = Some(v),
            "state" => state = Some(v),
            "error" => error = Some(v),
            _ => {}
        }
    }
    if let Some(err) = error {
        bail!("authorization server returned error: {err}");
    }
    let state = state.ok_or_else(|| anyhow!("callback missing state param"))?;
    if state != expected_state {
        bail!("callback state did not match (CSRF guard)");
    }
    code.ok_or_else(|| anyhow!("callback missing code param"))
}

/// Inverse of `urlencode`: decode `%XX` escapes and `+` to space.
fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    out.push(byte);
                    i += 3;
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use wiremock::matchers::{body_json, body_string_contains, method, path};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    fn test_access_token(account_id: &str) -> String {
        let payload = serde_json::json!({
            "https://api.openai.com/auth": {
                "chatgpt_account_id": account_id,
            }
        });
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&payload).unwrap());
        format!("header.{payload}.signature")
    }

    #[derive(Clone)]
    struct PendingThenAuthorized {
        calls: Arc<AtomicUsize>,
    }

    impl Respond for PendingThenAuthorized {
        fn respond(&self, _request: &Request) -> ResponseTemplate {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                ResponseTemplate::new(403)
            } else {
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "authorization_code": "device-authorization-code",
                    "code_verifier": "device-verifier",
                }))
            }
        }
    }

    fn test_device_config(server: &MockServer, timeout: Duration) -> DeviceAuthConfig {
        DeviceAuthConfig {
            issuer: server.uri(),
            token_url: format!("{}/oauth/token", server.uri()),
            client_id: "test-client".to_string(),
            timeout,
        }
    }

    #[test]
    fn auth_json_round_trip_preserves_codex_field_names() {
        // Mirrors a real ~/.codex/auth.json from a logged-in Codex CLI install.
        let raw = r#"{
            "auth_mode": "chatgpt",
            "OPENAI_API_KEY": "sk-svcacct-test",
            "tokens": {
                "id_token": "eyJ.id",
                "access_token": "eyJ.acc",
                "refresh_token": "rt-test",
                "account_id": "acct_abc"
            },
            "last_refresh": "2026-05-05T12:00:00Z"
        }"#;
        let parsed: AuthDotJson = serde_json::from_str(raw).expect("parse auth.json");
        assert_eq!(parsed.auth_mode.as_deref(), Some("chatgpt"));
        assert_eq!(parsed.openai_api_key.as_deref(), Some("sk-svcacct-test"));
        let tokens = parsed.tokens.as_ref().expect("tokens");
        assert_eq!(tokens.account_id, "acct_abc");

        let reserialized = serde_json::to_string(&parsed).unwrap();
        // The shouting-snake-case key name is what Codex CLI expects;
        // serde's default would lower-case it.
        assert!(reserialized.contains("\"OPENAI_API_KEY\""));
        assert!(reserialized.contains("\"auth_mode\""));
        assert!(reserialized.contains("\"last_refresh\""));
    }

    #[test]
    fn auth_json_optional_fields_omit_cleanly() {
        let auth = AuthDotJson {
            auth_mode: Some("apikey".to_string()),
            openai_api_key: None,
            tokens: None,
            last_refresh: None,
        };
        let json = serde_json::to_string(&auth).unwrap();
        // Defaults that are None should not appear; auth_mode is always present.
        assert!(!json.contains("OPENAI_API_KEY"));
        assert!(!json.contains("tokens"));
        assert!(!json.contains("last_refresh"));
        assert!(json.contains("\"auth_mode\":\"apikey\""));
    }

    #[test]
    fn auth_json_path_honours_codex_home_override() {
        let _scope = TestCodexHomeScope::set(PathBuf::from("/tmp/codex-home-test-xyz"));
        let p = auth_json_path().unwrap();
        assert_eq!(
            p,
            std::path::PathBuf::from("/tmp/codex-home-test-xyz/auth.json")
        );
    }

    #[test]
    fn extract_chatgpt_account_id_pulls_nested_claim() {
        // Synthesize a JWT with the OpenAI namespace claim. We don't need a
        // valid signature -- the production code skips verification too.
        let payload = serde_json::json!({
            "sub": "user_xyz",
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct_test_123",
                "chatgpt_plan_type": "plus"
            }
        });
        let payload_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&payload).unwrap());
        let token = format!("eyJhbGciOiJSUzI1NiJ9.{payload_b64}.fake-sig");
        assert_eq!(extract_chatgpt_account_id(&token).unwrap(), "acct_test_123");
    }

    #[test]
    fn extract_chatgpt_account_id_errors_on_missing_claim() {
        let payload = serde_json::json!({"sub": "user_xyz"});
        let payload_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&payload).unwrap());
        let token = format!("eyJhbGciOiJSUzI1NiJ9.{payload_b64}.fake-sig");
        assert!(extract_chatgpt_account_id(&token).is_err());
    }

    #[test]
    fn parse_callback_url_validates_state_and_returns_code() {
        let url = "/auth/callback?code=abc123&state=xyz";
        assert_eq!(parse_callback_url(url, "xyz").unwrap(), "abc123");
    }

    #[test]
    fn parse_callback_url_rejects_state_mismatch() {
        let url = "/auth/callback?code=abc123&state=evil";
        let err = parse_callback_url(url, "expected").unwrap_err().to_string();
        assert!(err.contains("CSRF"));
    }

    #[test]
    fn parse_callback_url_surfaces_authorization_server_error() {
        let url = "/auth/callback?error=access_denied&state=xyz";
        let err = parse_callback_url(url, "xyz").unwrap_err().to_string();
        assert!(err.contains("access_denied"));
    }

    #[test]
    fn authorize_url_carries_codex_simplified_flow_flags() {
        // These three params + the connectors scopes are exactly what
        // flips OpenAI's consent screen from "API organization access"
        // to "Sign in with ChatGPT". Regression-guard them: dropping
        // any one silently routes the user back to the API-key flow.
        let (challenge, _verifier) = oauth2::PkceCodeChallenge::new_random_sha256();
        let url = build_authorize_url(&challenge, "test-state");
        assert!(
            url.contains("codex_cli_simplified_flow=true"),
            "missing codex_cli_simplified_flow flag in {url}"
        );
        assert!(
            url.contains("id_token_add_organizations=true"),
            "missing id_token_add_organizations flag in {url}"
        );
        assert!(
            url.contains("originator=codex_cli_rs"),
            "missing originator=codex_cli_rs in {url}"
        );
        assert!(
            url.contains("api.connectors.read"),
            "missing api.connectors.read scope in {url}"
        );
        assert!(
            url.contains("api.connectors.invoke"),
            "missing api.connectors.invoke scope in {url}"
        );
        // Sanity: still PKCE + CSRF.
        assert!(url.contains("code_challenge="));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("state=test-state"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn device_login_polls_and_persists_codex_compatible_credentials() {
        let server = MockServer::start().await;
        let codex_home = tempfile::tempdir().unwrap();
        let _scope = TestCodexHomeScope::set(codex_home.path().to_path_buf());
        let poll_calls = Arc::new(AtomicUsize::new(0));

        Mock::given(method("POST"))
            .and(path("/api/accounts/deviceauth/usercode"))
            .and(body_json(serde_json::json!({"client_id": "test-client"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "device_auth_id": "device-auth-id",
                "user_code": "ABCD-EFGH",
                "interval": "1",
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/accounts/deviceauth/token"))
            .and(body_json(serde_json::json!({
                "device_auth_id": "device-auth-id",
                "user_code": "ABCD-EFGH",
            })))
            .respond_with(PendingThenAuthorized {
                calls: poll_calls.clone(),
            })
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .and(body_string_contains("grant_type=authorization_code"))
            .and(body_string_contains("code=device-authorization-code"))
            .and(body_string_contains("code_verifier=device-verifier"))
            .and(body_string_contains("redirect_uri=http%3A%2F%2F127.0.0.1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": test_access_token("acct-device"),
                "refresh_token": "device-refresh-token",
                "id_token": "device-id-token",
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .and(body_string_contains("token-exchange"))
            .respond_with(ResponseTemplate::new(400).set_body_string("no API organization"))
            .expect(1)
            .mount(&server)
            .await;

        let config = test_device_config(&server, Duration::from_secs(3));
        let expected_url = config.verification_url();
        let cancel = CancellationToken::new();
        let auth = interactive_device_login_with_config(config, &cancel, |prompt| async move {
            assert_eq!(prompt.verification_url, expected_url);
            assert_eq!(prompt.user_code, "ABCD-EFGH");
            Ok(())
        })
        .await
        .unwrap();

        assert_eq!(poll_calls.load(Ordering::SeqCst), 2);
        assert_eq!(auth.auth_mode.as_deref(), Some("chatgpt"));
        assert_eq!(auth.openai_api_key, None);
        let tokens = auth.tokens.unwrap();
        assert_eq!(tokens.account_id, "acct-device");
        assert_eq!(tokens.refresh_token, "device-refresh-token");
        assert!(auth_json_path().unwrap().exists());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn device_login_cancellation_interrupts_poll_delay_without_writing_credentials() {
        let server = MockServer::start().await;
        let codex_home = tempfile::tempdir().unwrap();
        let _scope = TestCodexHomeScope::set(codex_home.path().to_path_buf());

        Mock::given(method("POST"))
            .and(path("/api/accounts/deviceauth/usercode"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "device_auth_id": "pending-device",
                "user_code": "WAIT-1234",
                "interval": "30",
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/accounts/deviceauth/token"))
            .respond_with(ResponseTemplate::new(403))
            .mount(&server)
            .await;

        let config = test_device_config(&server, Duration::from_secs(60));
        let cancel = CancellationToken::new();
        let cancel_after_present = cancel.clone();
        let result = tokio::time::timeout(
            Duration::from_secs(1),
            interactive_device_login_with_config(config, &cancel, move |_prompt| async move {
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    cancel_after_present.cancel();
                });
                Ok(())
            }),
        )
        .await
        .expect("cancellation must interrupt the long poll delay");

        assert!(result.unwrap_err().to_string().contains("cancelled"));
        assert!(!auth_json_path().unwrap().exists());
    }
}
