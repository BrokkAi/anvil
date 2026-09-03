use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Mutex;

const GROK_ISSUER: &str = "https://auth.x.ai";
const GROK_CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const REFRESH_MARGIN: Duration = Duration::from_secs(5 * 60);
const LOCK_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Debug)]
pub(crate) struct GrokCredential {
    pub(crate) access_token: String,
    pub(crate) user_id: String,
    pub(crate) email: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct AuthEntry {
    #[serde(default)]
    key: String,
    #[serde(default)]
    auth_mode: Option<String>,
    #[serde(default)]
    user_id: String,
    #[serde(default)]
    email: Option<String>,
    #[serde(default)]
    principal_type: Option<String>,
    #[serde(default)]
    principal_id: Option<String>,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    expires_at: Option<DateTime<Utc>>,
    #[serde(default)]
    oidc_issuer: Option<String>,
    #[serde(default)]
    oidc_client_id: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

type AuthStore = BTreeMap<String, Value>;

#[derive(Debug, Deserialize)]
struct OidcConfiguration {
    token_endpoint: String,
}

#[derive(Debug, Deserialize)]
struct RefreshResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    expires_in: Option<u64>,
}

pub(crate) struct GrokAuthManager {
    path: PathBuf,
    http: reqwest::Client,
    refresh_lock: Mutex<()>,
}

impl GrokAuthManager {
    pub(crate) fn load() -> Result<Option<Arc<Self>>> {
        let path = credentials_path()?;
        Self::load_from_path(path)
    }

    pub(crate) fn load_from_path(path: PathBuf) -> Result<Option<Arc<Self>>> {
        if !path.is_file() {
            return Ok(None);
        }
        let manager = Arc::new(Self {
            path,
            http: reqwest::Client::new(),
            refresh_lock: Mutex::new(()),
        });
        if manager.read_selected_sync()?.is_none() {
            return Ok(None);
        }
        Ok(Some(manager))
    }

    pub(crate) async fn credential(&self) -> Result<GrokCredential> {
        self.resolve(None).await
    }

    pub(crate) async fn refresh_rejected(&self, rejected_token: &str) -> Result<GrokCredential> {
        self.resolve(Some(rejected_token)).await
    }

    #[cfg(test)]
    pub(crate) fn for_test(path: PathBuf) -> Arc<Self> {
        Arc::new(Self {
            path,
            http: reqwest::Client::new(),
            refresh_lock: Mutex::new(()),
        })
    }

    async fn resolve(&self, rejected_token: Option<&str>) -> Result<GrokCredential> {
        let _process_guard = self.refresh_lock.lock().await;
        let selected = self.read_selected().await?.ok_or_else(login_error)?;
        if rejected_token.is_none() && is_fresh(&selected.1) {
            return Ok(to_credential(&selected.1));
        }

        let path = self.path.clone();
        let lock_file = tokio::task::spawn_blocking(move || acquire_lock(&path))
            .await
            .context("joining Grok credential-lock task")??;

        let mut store = self.read_store().await?;
        let (scope, mut entry) = select_entry(&store).ok_or_else(login_error)?;
        if rejected_token.is_some_and(|rejected| entry.key != rejected)
            || (rejected_token.is_none() && is_fresh(&entry))
        {
            drop(lock_file);
            return Ok(to_credential(&entry));
        }
        if !lock_is_current(&lock_file, &self.path)? {
            drop(lock_file);
            bail!("Grok credential lock was replaced; retry the request")
        }

        let refresh_token = entry
            .refresh_token
            .as_deref()
            .filter(|token| !token.is_empty())
            .ok_or_else(login_error)?;
        let issuer = entry
            .oidc_issuer
            .as_deref()
            .filter(|issuer| issuer.trim_end_matches('/') == GROK_ISSUER)
            .ok_or_else(login_error)?;
        let client_id = entry
            .oidc_client_id
            .as_deref()
            .filter(|id| !id.is_empty())
            .unwrap_or(GROK_CLIENT_ID);
        let discovery_url = format!(
            "{}/.well-known/openid-configuration",
            issuer.trim_end_matches('/')
        );
        let config = self
            .http
            .get(discovery_url)
            .send()
            .await
            .context("discovering the xAI OAuth token endpoint")?
            .error_for_status()
            .context("xAI OAuth discovery failed")?
            .json::<OidcConfiguration>()
            .await
            .context("parsing xAI OAuth discovery response")?;

        let mut form = vec![
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", client_id),
        ];
        if let Some(value) = entry.principal_type.as_deref().filter(|v| !v.is_empty()) {
            form.push(("principal_type", value));
        }
        if let Some(value) = entry.principal_id.as_deref().filter(|v| !v.is_empty()) {
            form.push(("principal_id", value));
        }
        let response = self
            .http
            .post(&config.token_endpoint)
            .form(&form)
            .send()
            .await
            .context("refreshing xAI OAuth credentials")?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            if body.contains("invalid_grant") || body.contains("invalid_client") {
                return Err(login_error());
            }
            bail!("xAI OAuth refresh failed (HTTP {status}): {body}")
        }
        let refreshed = response
            .json::<RefreshResponse>()
            .await
            .context("parsing xAI OAuth refresh response")?;
        entry.key = refreshed.access_token;
        if let Some(token) = refreshed.refresh_token.filter(|token| !token.is_empty()) {
            entry.refresh_token = Some(token);
        }
        entry.expires_at = refreshed
            .expires_in
            .map(|seconds| Utc::now() + chrono::Duration::seconds(seconds as i64));
        store.insert(
            scope,
            serde_json::to_value(&entry).context("serializing refreshed Grok OAuth credential")?,
        );
        self.write_store(store).await?;
        drop(lock_file);
        Ok(to_credential(&entry))
    }

    async fn read_selected(&self) -> Result<Option<(String, AuthEntry)>> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || read_store_sync(&path).map(|s| select_entry(&s)))
            .await
            .context("joining Grok credential-read task")?
    }

    fn read_selected_sync(&self) -> Result<Option<(String, AuthEntry)>> {
        read_store_sync(&self.path).map(|store| select_entry(&store))
    }

    async fn read_store(&self) -> Result<AuthStore> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || read_store_sync(&path))
            .await
            .context("joining Grok credential-read task")?
    }

    async fn write_store(&self, store: AuthStore) -> Result<()> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || write_store_sync(&path, &store))
            .await
            .context("joining Grok credential-write task")?
    }
}

pub(crate) fn credentials_path() -> Result<PathBuf> {
    if let Some(home) = std::env::var_os("GROK_HOME").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(home).join("auth.json"));
    }
    let home = dirs::home_dir().context("cannot locate the home directory for Grok OAuth")?;
    Ok(home.join(".grok").join("auth.json"))
}

pub(crate) fn client_version() -> String {
    let version_path = credentials_path()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join("version.json")));
    client_version_from_path(version_path.as_deref())
}

pub(crate) fn client_version_from_path(version_path: Option<&Path>) -> String {
    version_path
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .and_then(|value| value.get("version")?.as_str().map(str::to_owned))
        .filter(|version| !version.trim().is_empty())
        .unwrap_or_else(|| "1.0.5".to_string())
}

fn login_error() -> anyhow::Error {
    anyhow::anyhow!("Grok OAuth is unavailable or expired; run `grok login --oauth` and retry")
}

fn select_entry(store: &AuthStore) -> Option<(String, AuthEntry)> {
    store
        .iter()
        .filter_map(|(scope, value)| {
            serde_json::from_value::<AuthEntry>(value.clone())
                .ok()
                .map(|entry| (scope, entry))
        })
        .filter(|(_, entry)| {
            entry.auth_mode.as_deref() == Some("oidc")
                && entry
                    .oidc_issuer
                    .as_deref()
                    .map(|issuer| issuer.trim_end_matches('/'))
                    == Some(GROK_ISSUER)
                && !entry.key.trim().is_empty()
        })
        .max_by(|(scope_a, a), (scope_b, b)| {
            a.expires_at
                .cmp(&b.expires_at)
                .then_with(|| scope_b.cmp(scope_a))
        })
        .map(|(scope, entry)| (scope.clone(), entry))
}

fn is_fresh(entry: &AuthEntry) -> bool {
    entry.expires_at.is_some_and(|expires| {
        expires > Utc::now() + chrono::Duration::from_std(REFRESH_MARGIN).unwrap()
    })
}

fn to_credential(entry: &AuthEntry) -> GrokCredential {
    GrokCredential {
        access_token: entry.key.clone(),
        user_id: entry.user_id.clone(),
        email: entry.email.clone(),
    }
}

fn read_store_sync(path: &Path) -> Result<AuthStore> {
    let mut file = File::open(path)
        .with_context(|| format!("opening Grok OAuth credentials at {}", path.display()))?;
    let mut raw = String::new();
    file.read_to_string(&mut raw)
        .with_context(|| format!("reading Grok OAuth credentials at {}", path.display()))?;
    serde_json::from_str(&raw)
        .with_context(|| format!("parsing Grok OAuth credentials at {}", path.display()))
}

fn acquire_lock(path: &Path) -> Result<File> {
    let lock_path = path.with_file_name("auth.json.lock");
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("opening Grok credential lock at {}", lock_path.display()))?;
    let start = Instant::now();
    loop {
        match file.try_lock_exclusive() {
            Ok(()) => return Ok(file),
            Err(error) if start.elapsed() < LOCK_TIMEOUT => {
                std::thread::sleep(Duration::from_millis(50));
                if error.kind() != std::io::ErrorKind::WouldBlock {
                    tracing::debug!(%error, "waiting for Grok credential lock");
                }
            }
            Err(error) => return Err(error).context("timed out waiting for Grok credential lock"),
        }
    }
}

#[cfg(unix)]
fn lock_is_current(file: &File, auth_path: &Path) -> Result<bool> {
    use std::os::unix::fs::MetadataExt;
    let lock_path = auth_path.with_file_name("auth.json.lock");
    let held = file.metadata()?;
    let current = std::fs::metadata(lock_path)?;
    Ok(held.dev() == current.dev() && held.ino() == current.ino())
}

#[cfg(not(unix))]
fn lock_is_current(_file: &File, _auth_path: &Path) -> Result<bool> {
    Ok(true)
}

fn write_store_sync(path: &Path, store: &AuthStore) -> Result<()> {
    if path
        .symlink_metadata()
        .is_ok_and(|metadata| metadata.file_type().is_symlink())
    {
        bail!(
            "refusing to replace symlinked Grok credential file at {}",
            path.display()
        );
    }
    let parent = path
        .parent()
        .context("Grok credential path has no parent")?;
    let temp_path = parent.join(format!(".auth.json.{}.tmp", uuid::Uuid::new_v4()));
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut temp = options.open(&temp_path).with_context(|| {
        format!(
            "creating temporary Grok credentials at {}",
            temp_path.display()
        )
    })?;
    let result = (|| -> Result<()> {
        serde_json::to_writer_pretty(&mut temp, store)?;
        temp.write_all(b"\n")?;
        temp.sync_all()?;
        std::fs::rename(&temp_path, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result.with_context(|| {
        format!(
            "atomically writing Grok OAuth credentials at {}",
            path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(mode: &str, issuer: &str, key: &str, expires: DateTime<Utc>) -> AuthEntry {
        AuthEntry {
            key: key.into(),
            auth_mode: Some(mode.into()),
            user_id: "user".into(),
            email: None,
            principal_type: None,
            principal_id: None,
            refresh_token: Some("refresh".into()),
            expires_at: Some(expires),
            oidc_issuer: Some(issuer.into()),
            oidc_client_id: Some(GROK_CLIENT_ID.into()),
            extra: BTreeMap::new(),
        }
    }

    fn value(entry: AuthEntry) -> Value {
        serde_json::to_value(entry).unwrap()
    }

    #[test]
    fn selects_freshest_first_party_oidc_credential_only() {
        let now = Utc::now();
        let store = BTreeMap::from([
            (
                "api".into(),
                value(entry(
                    "api_key",
                    GROK_ISSUER,
                    "api",
                    now + chrono::Duration::hours(8),
                )),
            ),
            (
                "external".into(),
                value(entry(
                    "oidc",
                    "https://example.com",
                    "external",
                    now + chrono::Duration::hours(7),
                )),
            ),
            (
                "old".into(),
                value(entry(
                    "oidc",
                    GROK_ISSUER,
                    "old",
                    now + chrono::Duration::hours(1),
                )),
            ),
            (
                "new".into(),
                value(entry(
                    "oidc",
                    GROK_ISSUER,
                    "new",
                    now + chrono::Duration::hours(2),
                )),
            ),
            ("unknown".into(), Value::String("preserved".into())),
        ]);
        assert_eq!(select_entry(&store).unwrap().1.key, "new");
    }

    #[test]
    fn atomic_write_preserves_unknown_entry_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let mut selected = entry("oidc", GROK_ISSUER, "token", Utc::now());
        selected
            .extra
            .insert("future_field".into(), Value::String("kept".into()));
        let store = BTreeMap::from([("scope".into(), value(selected))]);
        write_store_sync(&path, &store).unwrap();
        let reread = read_store_sync(&path).unwrap();
        assert_eq!(reread["scope"]["future_field"], "kept");
    }

    #[test]
    fn sidecar_lock_serializes_independent_openers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        std::fs::write(&path, "{}").unwrap();
        let first = acquire_lock(&path).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let second_path = path.clone();
        let waiter = std::thread::spawn(move || {
            let lock = acquire_lock(&second_path).unwrap();
            tx.send(()).unwrap();
            drop(lock);
        });
        assert!(rx.recv_timeout(Duration::from_millis(150)).is_err());
        drop(first);
        rx.recv_timeout(Duration::from_secs(2)).unwrap();
        waiter.join().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_creates_owner_only_credentials() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        write_store_sync(&path, &BTreeMap::new()).unwrap();
        assert_eq!(
            std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}
