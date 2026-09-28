//! A profile's vendor process owns OAuth refresh and persistence. Anvil only
//! requests usage and reads the resulting access token. The process deliberately
//! outlives clients: stopping a caller must not interrupt credential rotation.
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use fs2::FileExt;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::{mpsc, oneshot};

use crate::llm_client::BearerTokenProvider;

/// Explicit launch configuration. Environment overrides are profile-local;
/// KIMI_CODE_HOME is always set from `home`, never inherited from another profile.
#[derive(Clone)]
pub struct KimiServiceConfig {
    pub home: PathBuf,
    pub executable: PathBuf,
    pub environment: BTreeMap<String, String>,
    /// Managed runtime lease inherited by the service on Unix.
    pub lease_path: Option<PathBuf>,
}

impl KimiServiceConfig {
    pub fn uses_api_key(&self) -> bool {
        self.api_key().is_some()
    }

    fn api_key(&self) -> Option<&str> {
        self.environment
            .get("KIMI_API_KEY")
            .map(String::as_str)
            .filter(|key| !key.trim().is_empty())
    }

    /// Match the vendor's read-only credential namespace for explicit endpoints.
    /// The service receives these same explicit defaults, so persisted regional
    /// configuration cannot select a different credential behind the caller.
    pub fn credentials_path(&self) -> Result<PathBuf> {
        let oauth_host = self
            .environment
            .get("KIMI_CODE_OAUTH_HOST")
            .or_else(|| self.environment.get("KIMI_OAUTH_HOST"))
            .map(String::as_str)
            .unwrap_or("https://auth.kimi.com")
            .trim()
            .trim_end_matches('/');
        let base_url = self
            .environment
            .get("KIMI_CODE_BASE_URL")
            .map(String::as_str)
            .unwrap_or(super::DEFAULT_KIMI_CODE_BASE_URL)
            .trim_end_matches('/');
        let name = if oauth_host == "https://auth.kimi.com"
            && base_url == super::DEFAULT_KIMI_CODE_BASE_URL
        {
            "kimi-code".to_owned()
        } else {
            #[derive(Serialize)]
            #[serde(rename_all = "camelCase")]
            struct Identity<'a> {
                oauth_host: &'a str,
                base_url: &'a str,
            }
            let hash = Sha256::digest(serde_json::to_vec(&Identity {
                oauth_host,
                base_url,
            })?);
            let suffix: String = hash[..8].iter().map(|byte| format!("{byte:02x}")).collect();
            format!("kimi-code-env-{suffix}")
        };
        Ok(self.home.join("credentials").join(format!("{name}.json")))
    }

    pub fn from_home(home: impl AsRef<Path>) -> Self {
        Self {
            home: home.as_ref().to_owned(),
            executable: PathBuf::from("kimi"),
            environment: BTreeMap::new(),
            lease_path: None,
        }
    }
}

pub struct KimiService {
    config: KimiServiceConfig,
    owner: tokio::sync::OnceCell<(Arc<Owner>, KimiServiceConfig)>,
}

impl KimiService {
    pub fn new(mut config: KimiServiceConfig) -> Result<Arc<Self>> {
        let oauth_host = config
            .environment
            .get("KIMI_CODE_OAUTH_HOST")
            .cloned()
            .or_else(|| config.environment.remove("KIMI_OAUTH_HOST"))
            .unwrap_or_else(|| "https://auth.kimi.com".into());
        config.environment.remove("KIMI_OAUTH_HOST");
        config
            .environment
            .insert("KIMI_CODE_OAUTH_HOST".into(), oauth_host);
        config
            .environment
            .entry("KIMI_CODE_BASE_URL".into())
            .or_insert_with(|| super::DEFAULT_KIMI_CODE_BASE_URL.into());
        Ok(Arc::new(Self {
            config,
            owner: tokio::sync::OnceCell::new(),
        }))
    }

    /// Fetch normalized vendor usage. Accepted requests run on an independent
    /// owner thread; dropping this future never cancels an accepted refresh.
    pub async fn usage(&self) -> Result<Value> {
        if self.config.uses_api_key() {
            return Ok(serde_json::json!({"kind":"api_key"}));
        }
        let (owner, config) = self
            .owner
            .get_or_try_init(|| async {
                let home = tokio::fs::canonicalize(&self.config.home)
                    .await
                    .context("resolve Kimi profile home")?;
                let mut config = self.config.clone();
                config.home = home;
                config.environment.remove("KIMI_CODE_HOME");
                Ok::<_, anyhow::Error>((owner(config.home.clone())?, config))
            })
            .await?;
        let (reply, response) = oneshot::channel();
        owner
            .sender
            .send(Request {
                owner: owner.clone(),
                config: config.clone(),
                reply,
            })
            .await
            .context("Kimi service owner stopped")?;
        response
            .await
            .context("Kimi service owner lost its response")?
    }

    async fn read_credentials(&self) -> Result<Credentials> {
        let path = self.config.credentials_path()?;
        let bytes = tokio::fs::read(path)
            .await
            .context("read vendor-owned Kimi credentials")?;
        let credentials: Credentials =
            serde_json::from_slice(&bytes).context("parse Kimi credentials")?;
        if credentials.access_token.trim().is_empty() {
            bail!("Kimi access token is missing; run `kimi login`");
        }
        Ok(credentials)
    }

    async fn read_token(&self) -> Result<String> {
        Ok(self.read_credentials().await?.access_token)
    }
}

#[derive(Deserialize)]
struct Credentials {
    access_token: String,
    #[serde(default)]
    expires_at: u64,
}

impl BearerTokenProvider for KimiService {
    fn bearer_token(&self) -> BoxFuture<'_, Result<Option<String>>> {
        Box::pin(async move {
            if let Some(key) = self.config.api_key() {
                return Ok(Some(key.to_owned()));
            }
            let credentials = self.read_credentials().await?;
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            if credentials.expires_at > now.saturating_add(60) {
                return Ok(Some(credentials.access_token));
            }
            self.usage().await?;
            self.read_token().await.map(Some)
        })
    }

    fn rejected_bearer_token<'a>(
        &'a self,
        rejected: &'a str,
    ) -> BoxFuture<'a, Result<Option<String>>> {
        Box::pin(async move {
            // There is no supported forced-refresh endpoint. Only a token
            // already changed by its owner authorizes another HTTP attempt.
            if self.config.uses_api_key() {
                return Ok(None);
            }
            let current = self.read_token().await?;
            Ok((current != rejected).then_some(current))
        })
    }
}

struct Request {
    // Keep the canonical owner discoverable until an accepted operation settles.
    owner: Arc<Owner>,
    config: KimiServiceConfig,
    reply: oneshot::Sender<Result<Value>>,
}
struct Owner {
    sender: mpsc::Sender<Request>,
}
static OWNERS: OnceLock<Mutex<HashMap<PathBuf, Weak<Owner>>>> = OnceLock::new();

fn fingerprint(config: &KimiServiceConfig) -> Result<String> {
    let bytes = serde_json::to_vec(&(&config.executable, &config.environment))?;
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn owner(home: PathBuf) -> Result<Arc<Owner>> {
    let mut owners = OWNERS
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| anyhow::anyhow!("Kimi service owner registry poisoned"))?;
    owners.retain(|_, owner| owner.strong_count() > 0);
    if let Some(owner) = owners.get(&home).and_then(Weak::upgrade) {
        return Ok(owner);
    }
    let (sender, mut receiver) = mpsc::channel::<Request>(64);
    let owner = Arc::new(Owner { sender });
    std::thread::Builder::new()
        .name("kimi-service-owner".into())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();
            let runtime = match runtime {
                Ok(runtime) => runtime,
                Err(error) => {
                    while let Some(request) = receiver.blocking_recv() {
                        let _ = request
                            .reply
                            .send(Err(anyhow::anyhow!("start Kimi service runtime: {error}")));
                    }
                    return;
                }
            };
            runtime.block_on(async move {
                let mut recent: Option<(String, Instant, Value)> = None;
                while let Some(Request {
                    owner,
                    config,
                    reply,
                }) = receiver.recv().await
                {
                    let result = match fingerprint(&config) {
                        Ok(key) => {
                            let result = match &recent {
                                Some((previous, at, value))
                                    if *previous == key
                                        && at.elapsed() < Duration::from_secs(1) =>
                                {
                                    Ok(value.clone())
                                }
                                _ => request_usage(&config, &key).await,
                            };
                            if let Ok(value) = &result {
                                recent = Some((key, Instant::now(), value.clone()));
                            }
                            result
                        }
                        Err(error) => Err(error),
                    };
                    if let Err(error) = &result {
                        tracing::warn!("Kimi vendor usage failed: {error:#}");
                    }
                    let _ = reply.send(result);
                    drop(owner);
                }
            });
        })
        .context("start Kimi service owner")?;
    owners.insert(home, Arc::downgrade(&owner));
    Ok(owner)
}

#[derive(Clone, Serialize, Deserialize)]
struct ServiceRecord {
    server_id: String,
    pid: u32,
    host: String,
    port: u16,
    #[serde(default)]
    fingerprint: String,
    #[serde(default)]
    meta_server_id: String,
    #[serde(default)]
    stopping: bool,
}

async fn acquire_lock(home: &Path) -> Result<std::fs::File> {
    let directory = home.join("server");
    tokio::fs::create_dir_all(&directory)
        .await
        .context("create Kimi service directory")?;
    let path = directory.join("anvil-service.lock");
    let file = tokio::task::spawn_blocking(move || {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
    })
    .await
    .context("join Kimi lock open")??;
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        match file.try_lock_exclusive() {
            Ok(()) => return Ok(file),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    bail!("timed out waiting for the Kimi service owner");
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            Err(error) => return Err(error).context("lock Kimi service owner"),
        }
    }
}

async fn request_usage(config: &KimiServiceConfig, fingerprint: &str) -> Result<Value> {
    // This lock coordinates our services, not the vendor's OAuth lock. Never
    // imitate or reclaim vendor lock directories.
    let launch_lock = acquire_lock(&config.home).await?;
    let http = reqwest::Client::builder()
        .no_proxy()
        .connect_timeout(Duration::from_secs(3))
        .timeout(Duration::from_secs(60))
        .build()?;
    let record_path = config.home.join("server").join("anvil-service.json");
    let existing = match tokio::fs::read(&record_path).await {
        Ok(bytes) => Some(
            serde_json::from_slice::<ServiceRecord>(&bytes)
                .context("parse Kimi service registration")?,
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error).context("read Kimi service registration"),
    };
    let record = if let Some(mut record) = existing {
        if record.port == 0 && record.server_id.is_empty() {
            record = await_registration(
                config,
                &record.fingerprint,
                &http,
                &record_path,
                record.pid,
                None,
            )
            .await?;
        }
        if record.stopping
            || validate_service(&http, &config.home, &record)
                .await?
                .is_some()
        {
            if record.stopping || record.fingerprint != fingerprint {
                retire_service(&http, &config.home, &mut record, &record_path).await?;
                launch(config, fingerprint, &http, &record_path, &launch_lock).await?
            } else {
                record
            }
        } else {
            launch(config, fingerprint, &http, &record_path, &launch_lock).await?
        }
    } else {
        launch(config, fingerprint, &http, &record_path, &launch_lock).await?
    };
    let token = server_token(&config.home).await?;
    let response = http
        .get(format!(
            "http://127.0.0.1:{}/api/v1/oauth/usage",
            record.port
        ))
        .bearer_auth(token)
        .send()
        .await
        .context("request Kimi vendor usage")?;
    let status = response.status();
    if !status.is_success() {
        bail!("Kimi vendor usage returned HTTP {status}");
    }
    let envelope: Value = response.json().await.context("decode Kimi vendor usage")?;
    let data = envelope
        .get("data")
        .context("Kimi vendor usage is missing data")?;
    if data.get("kind").and_then(Value::as_str) != Some("ok") {
        // Vendor error text is not trusted to be free of credentials.
        bail!(
            "Kimi vendor usage failed (HTTP status {:?}); inspect the profile login and endpoint configuration",
            data.get("status").and_then(Value::as_u64)
        );
    }
    Ok(data.clone())
}

async fn save_record(path: &Path, record: &ServiceRecord) -> Result<()> {
    let temp = path.with_extension("tmp");
    tokio::fs::write(&temp, serde_json::to_vec(record)?)
        .await
        .context("write Kimi service registration")?;
    tokio::fs::rename(temp, path)
        .await
        .context("publish Kimi service registration")
}

async fn retire_service(
    http: &reqwest::Client,
    home: &Path,
    record: &mut ServiceRecord,
    path: &Path,
) -> Result<()> {
    if !record.stopping {
        record.stopping = true;
        save_record(path, record).await?;
    }
    let registration = home
        .join("server")
        .join("instances")
        .join(format!("{}.json", record.server_id));
    if !tokio::fs::try_exists(&registration).await? {
        return Ok(());
    }
    let still_accepting = match validate_service(http, home, record).await {
        Ok(identity) => identity.is_some(),
        Err(error) => {
            tracing::debug!("Kimi retirement identity probe failed while draining: {error:#}");
            false
        }
    };
    if still_accepting {
        let response = http
            .post(format!("http://127.0.0.1:{}/api/v1/shutdown", record.port))
            .bearer_auth(server_token(home).await?)
            .send()
            .await
            .context("request Kimi service retirement")?;
        if !response.status().is_success()
            && response.status() != reqwest::StatusCode::SERVICE_UNAVAILABLE
        {
            bail!(
                "Kimi service retirement returned HTTP {}",
                response.status()
            );
        }
    }
    // Vendor removes its registration only after Fastify drains accepted HTTP
    // requests and core cleanup finishes. HTTP 200 alone is not settlement.
    let deadline = Instant::now() + Duration::from_secs(60);
    while tokio::fs::try_exists(&registration).await? {
        #[cfg(unix)]
        if process_is_gone(record.pid) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!("Kimi service is still draining accepted work; replacement deferred");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    Ok(())
}

async fn server_token(home: &Path) -> Result<String> {
    let token = tokio::fs::read_to_string(home.join("server.token"))
        .await
        .context("read Kimi service authentication token")?;
    let token = token.trim();
    if token.is_empty() {
        bail!("Kimi service authentication token is empty");
    }
    Ok(token.to_owned())
}

async fn validate_service(
    http: &reqwest::Client,
    home: &Path,
    record: &ServiceRecord,
) -> Result<Option<String>> {
    if record.host != "127.0.0.1" || record.port == 0 || record.server_id.contains(['/', '\\']) {
        bail!("invalid Kimi service identity");
    }
    let path = home
        .join("server")
        .join("instances")
        .join(format!("{}.json", record.server_id));
    let bytes = match tokio::fs::read(path).await {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).context("read vendor service identity"),
    };
    let registered: ServiceRecord =
        serde_json::from_slice(&bytes).context("parse vendor service identity")?;
    if registered.pid != record.pid
        || registered.port != record.port
        || registered.server_id != record.server_id
        || registered.host != record.host
    {
        return Ok(None);
    }
    let token = server_token(home).await?;
    let response = match http
        .get(format!("http://127.0.0.1:{}/api/v1/meta", record.port))
        .timeout(Duration::from_secs(3))
        .bearer_auth(token)
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) if error.is_connect() => return Ok(None),
        Err(error) => return Err(error).context("verify Kimi service identity"),
    };
    if !response.status().is_success() {
        bail!("Kimi service identity returned HTTP {}", response.status());
    }
    let meta: Value = response
        .json()
        .await
        .context("decode Kimi service identity")?;
    let actual = meta
        .pointer("/data/server_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .context("Kimi service metadata has no identity")?;
    if !record.meta_server_id.is_empty() && record.meta_server_id != actual {
        bail!("Kimi service metadata identity changed since launch");
    }
    Ok(Some(actual.to_owned()))
}

async fn launch(
    config: &KimiServiceConfig,
    fingerprint: &str,
    http: &reqwest::Client,
    record_path: &Path,
    launch_lock: &std::fs::File,
) -> Result<ServiceRecord> {
    let (pid, exited) = spawn_service(config, fingerprint, record_path, launch_lock)?;
    await_registration(config, fingerprint, http, record_path, pid, Some(exited)).await
}

async fn await_registration(
    config: &KimiServiceConfig,
    fingerprint: &str,
    http: &reqwest::Client,
    record_path: &Path,
    pid: u32,
    mut exited: Option<oneshot::Receiver<String>>,
) -> Result<ServiceRecord> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        #[cfg(unix)]
        if exited.is_none() && process_is_gone(pid) {
            tokio::fs::remove_file(record_path)
                .await
                .context("remove departed Kimi launch registration")?;
            bail!(
                "Kimi vendor process exited during daemon handoff before readiness; next request can start a new service"
            );
        }
        if let Some(exited) = &mut exited
            && let Ok(status) = exited.try_recv()
        {
            tokio::fs::remove_file(record_path)
                .await
                .context("remove exited Kimi launch registration")?;
            bail!("Kimi service exited before readiness: {status}");
        }
        let directory = config.home.join("server").join("instances");
        match tokio::fs::read_dir(&directory).await {
            Ok(mut entries) => {
                while let Some(entry) = entries
                    .next_entry()
                    .await
                    .context("scan Kimi registrations")?
                {
                    if entry.path().extension().is_none_or(|ext| ext != "json") {
                        continue;
                    }
                    let bytes = tokio::fs::read(entry.path())
                        .await
                        .context("read Kimi registration")?;
                    let Ok(mut record) = serde_json::from_slice::<ServiceRecord>(&bytes) else {
                        continue;
                    };
                    if record.pid != pid || record.port == 0 {
                        continue;
                    }
                    if let Some(meta_id) = validate_service(http, &config.home, &record).await? {
                        record.meta_server_id = meta_id;
                        record.fingerprint = fingerprint.to_owned();
                        save_record(record_path, &record).await?;
                        return Ok(record);
                    }
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).context("scan Kimi service directory"),
        }
        if Instant::now() >= deadline {
            bail!(
                "Kimi service did not publish an authenticated loopback registration within 30 seconds (pid {pid})"
            );
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// No pipes: the ready URL includes a secret even at silent log level. A reaper
/// thread owns wait(), while the vendor process survives the caller/runtime.
#[cfg(unix)]
fn spawn_service(
    config: &KimiServiceConfig,
    fingerprint: &str,
    record_path: &Path,
    launch_lock: &std::fs::File,
) -> Result<(u32, oneshot::Receiver<String>)> {
    let mut command = Command::new(&config.executable);
    command
        .args([
            "web",
            "--no-open",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--log-level",
            "silent",
        ])
        .envs(&config.environment)
        .env("KIMI_CODE_HOME", &config.home)
        .current_dir(&config.home)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    // Do not inherit a different profile's OAuth routing or static credential.
    for name in [
        "KIMI_API_KEY",
        "KIMI_CODE_BASE_URL",
        "KIMI_CODE_OAUTH_HOST",
        "KIMI_OAUTH_HOST",
        "KIMI_CODE_CUSTOM_HEADERS",
    ] {
        if !config.environment.contains_key(name) {
            command.env_remove(name);
        }
    }
    #[cfg(unix)]
    let lease = {
        use std::os::fd::AsRawFd;
        use std::os::unix::process::CommandExt;
        command.process_group(0);
        // Publish the child's identity before vendor exec. The inherited control
        // lock remains held until exec even if the parent dies during spawn, so
        // another daemon can adopt this pending launch instead of duplicating it.
        let marker_path = record_path.with_extension("launch.tmp");
        let marker = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&marker_path)
            .context("prepare durable Kimi launch registration")?;
        use std::os::unix::ffi::OsStrExt;
        let marker_name = std::ffi::CString::new(marker_path.as_os_str().as_bytes())?;
        let record_name = std::ffi::CString::new(record_path.as_os_str().as_bytes())?;
        let marker_fd = marker.as_raw_fd();
        let lock_fd = launch_lock.as_raw_fd();
        let suffix = format!(",\"host\":\"127.0.0.1\",\"port\":0,\"server_id\":\"\",\"fingerprint\":\"{fingerprint}\"}}").into_bytes();
        // SAFETY: only async-signal-safe syscalls and stack arithmetic execute
        // after fork. All descriptors/buffers are owned until spawn returns.
        unsafe {
            command.pre_exec(move || {
                let _ = lock_fd; // Keep the inherited open-file lock through exec.
                let mut digits = [0u8; 10];
                let mut offset = digits.len();
                let mut pid = libc::getpid() as u32;
                loop {
                    offset -= 1;
                    digits[offset] = b'0' + (pid % 10) as u8;
                    pid /= 10;
                    if pid == 0 {
                        break;
                    }
                }
                for part in [
                    b"{\"pid\":".as_slice(),
                    &digits[offset..],
                    suffix.as_slice(),
                ] {
                    let mut written = 0;
                    while written < part.len() {
                        let count = libc::write(
                            marker_fd,
                            part[written..].as_ptr().cast(),
                            part.len() - written,
                        );
                        if count < 0 {
                            return Err(std::io::Error::last_os_error());
                        }
                        if count == 0 {
                            return Err(std::io::ErrorKind::WriteZero.into());
                        }
                        written += count as usize;
                    }
                }
                if libc::fsync(marker_fd) != 0
                    || libc::rename(marker_name.as_ptr(), record_name.as_ptr()) != 0
                {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }

        let lease = config
            .lease_path
            .as_ref()
            .map(|path| -> Result<std::fs::File> {
                let file = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(path)
                    .context("open Kimi runtime lease")?;
                FileExt::lock_shared(&file).context("retain Kimi runtime lease")?;
                Ok(file)
            })
            .transpose()?;
        if let Some(file) = &lease {
            let fd = file.as_raw_fd();
            // SAFETY: fcntl is async-signal-safe; this closure only changes the
            // child's copy of the existing lease descriptor before exec.
            unsafe {
                command.pre_exec(move || {
                    if libc::fcntl(fd, libc::F_SETFD, 0) == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }
        (lease, marker)
    };
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            // spawn waited for the failed exec child; no vendor process exists.
            if let Err(cleanup) = std::fs::remove_file(record_path)
                && cleanup.kind() != std::io::ErrorKind::NotFound
            {
                tracing::warn!(%cleanup, "remove failed Kimi launch registration");
            }
            return Err(error).context("start Kimi vendor service");
        }
    };
    #[cfg(unix)]
    drop(lease);
    let pid = child.id();
    let (reply, response) = oneshot::channel();
    std::thread::Builder::new()
        .name("kimi-service-reaper".into())
        .spawn(move || {
            let status = match child.wait() {
                Ok(status) => {
                    if !status.success() {
                        tracing::warn!(%pid, %status, "Kimi vendor service exited");
                    }
                    status.to_string()
                }
                Err(error) => {
                    tracing::warn!(%pid, %error, "failed to reap Kimi vendor service");
                    error.to_string()
                }
            };
            let _ = reply.send(status);
        })
        .context("start Kimi vendor process reaper")?;
    Ok((pid, response))
}

#[cfg(not(unix))]
fn spawn_service(
    _config: &KimiServiceConfig,
    _fingerprint: &str,
    _record_path: &Path,
    _launch_lock: &std::fs::File,
) -> Result<(u32, oneshot::Receiver<String>)> {
    bail!(
        "Kimi vendor service auto-launch requires Unix durable process ownership on this version; configure KIMI_API_KEY on this platform"
    )
}

#[cfg(unix)]
fn process_is_gone(pid: u32) -> bool {
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    if pid <= 0 {
        return false;
    }
    // SAFETY: signal zero only queries existence, never signals a process.
    unsafe {
        libc::kill(pid, 0) == -1
            && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::kimi_auth::KimiBackendConfig;
    use std::io::{Read, Write};
    use std::os::unix::fs::PermissionsExt;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn fixture() -> (tempfile::TempDir, KimiServiceConfig) {
        let home = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(home.path().join("credentials")).unwrap();
        std::fs::write(
            home.path().join("credentials/kimi-code.json"),
            br#"{"access_token":"expired","refresh_token":"never-write-this","expires_at":1}"#,
        )
        .unwrap();
        let executable = home.path().join("kimi");
        let test_binary = std::env::current_exe().unwrap();
        // Paths come from the test environment, quoted as single shell words.
        let quoted = test_binary.to_str().unwrap().replace('\'', "'\\''");
        std::fs::write(&executable, format!("#!/bin/sh\nexec '{quoted}' --exact kimi_auth::service::tests::fake_vendor_service --ignored --nocapture\n")).unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
        let mut config = KimiServiceConfig::from_home(home.path());
        config.executable = executable;
        (home, config)
    }

    async fn stop(home: &Path) {
        let bytes = tokio::fs::read(home.join("server/anvil-service.json"))
            .await
            .unwrap();
        let record: ServiceRecord = serde_json::from_slice(&bytes).unwrap();
        reqwest::Client::new()
            .get(format!("http://127.0.0.1:{}/stop", record.port))
            .send()
            .await
            .unwrap();
    }

    #[test]
    #[ignore = "fake vendor process launched only by isolated service tests"]
    fn fake_vendor_service() {
        let home = PathBuf::from(std::env::var_os("KIMI_CODE_HOME").unwrap());
        assert!(home.join("kimi").is_file(), "requires isolated fixture");
        if let Ok(delay) = std::env::var("KIMI_TEST_STARTUP_DELAY_MS") {
            std::thread::sleep(Duration::from_millis(delay.parse().unwrap()));
        }
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let id = format!("fake-{}", std::process::id());
        std::fs::create_dir_all(home.join("server/instances")).unwrap();
        std::fs::write(home.join("server.token"), "test-service-secret").unwrap();
        let record = serde_json::json!({"server_id":id,"pid":std::process::id(),"host":"127.0.0.1","port":port,"heartbeat_at":0,"started_at":0});
        std::fs::write(
            home.join(format!("server/instances/{id}.json")),
            serde_json::to_vec(&record).unwrap(),
        )
        .unwrap();
        let mut calls = 0;
        for connection in listener.incoming() {
            let mut connection = connection.unwrap();
            let mut bytes = Vec::new();
            loop {
                let mut buffer = [0u8; 1024];
                let n = connection.read(&mut buffer).unwrap();
                if n == 0 {
                    break;
                }
                bytes.extend_from_slice(&buffer[..n]);
                if bytes.windows(4).any(|part| part == b"\r\n\r\n") {
                    break;
                }
            }
            let request = String::from_utf8(bytes).unwrap();
            let uri = request.split_whitespace().nth(1).unwrap_or("");
            let data = match uri {
                "/api/v1/meta" => serde_json::json!({"server_id":id}),
                "/api/v1/oauth/usage" => {
                    assert!(
                        request
                            .to_lowercase()
                            .contains("authorization: bearer test-service-secret")
                    );
                    calls += 1;
                    std::fs::write(home.join("usage-started"), calls.to_string()).unwrap();
                    std::thread::sleep(Duration::from_millis(150));
                    std::fs::write(home.join("credentials/kimi-code.json"), br#"{"access_token":"fresh","refresh_token":"vendor-rotated","expires_at":9999999999}"#).unwrap();
                    serde_json::json!({"kind":"ok","quota":{"usages":{"limit5h":{"usedRatio":calls as f64 / 10.0}},"extraUsage":null}})
                }
                "/stop" | "/api/v1/shutdown" => {
                    let _ = connection.write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                    std::fs::remove_file(home.join(format!("server/instances/{id}.json"))).unwrap();
                    return;
                }
                _ => panic!("unexpected vendor endpoint {uri}"),
            };
            let body = serde_json::to_vec(&serde_json::json!({"data":data})).unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = connection.write_all(response.as_bytes());
            let _ = connection.write_all(&body);
        }
    }

    #[tokio::test]
    async fn independent_clients_share_vendor_refresh_and_reuse_registration() {
        let (home, config) = fixture();
        let first = KimiService::new(config.clone()).unwrap();
        let alias = home.path().join("alias");
        std::os::unix::fs::symlink(home.path(), &alias).unwrap();
        let mut other = config.clone();
        other.home = alias;
        other
            .environment
            .insert("KIMI_CODE_HOME".into(), "ignored".into());
        let second = KimiService::new(other).unwrap();
        let (token, usage) = tokio::join!(first.bearer_token(), second.usage());
        assert_eq!(token.unwrap().as_deref(), Some("fresh"));
        assert_eq!(
            usage.unwrap()["quota"]["usages"]["limit5h"]["usedRatio"],
            0.1
        );
        let registration = tokio::fs::read(home.path().join("server/anvil-service.json"))
            .await
            .unwrap();
        drop(first);
        drop(second);
        let third = KimiService::new(config).unwrap();
        assert_eq!(
            third.bearer_token().await.unwrap().as_deref(),
            Some("fresh")
        );
        assert_eq!(
            registration,
            tokio::fs::read(home.path().join("server/anvil-service.json"))
                .await
                .unwrap()
        );
        stop(home.path()).await;
    }

    #[tokio::test]
    async fn cancellation_does_not_interrupt_vendor_persistence() {
        let (home, config) = fixture();
        let service = KimiService::new(config.clone()).unwrap();
        let request = tokio::spawn(async move { service.usage().await });
        tokio::time::timeout(Duration::from_secs(10), async {
            while !home.path().join("usage-started").exists() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        request.abort();
        let _ = request.await;
        let next = KimiService::new(config).unwrap();
        assert_eq!(next.bearer_token().await.unwrap().as_deref(), Some("fresh"));
        let credentials = tokio::fs::read_to_string(home.path().join("credentials/kimi-code.json"))
            .await
            .unwrap();
        assert!(credentials.contains("vendor-rotated"));
        stop(home.path()).await;
    }

    #[tokio::test]
    async fn explicit_provider_preserves_headers_without_touching_credentials() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .and(header("authorization", "Bearer supplied-key"))
            .and(header("x-msh-device-id", "fixture-device"))
            .and(header("x-custom", "preserved"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"data":[{"id":"k3"}]})),
            )
            .expect(1)
            .mount(&server)
            .await;
        let home = tempfile::tempdir().unwrap();
        std::fs::write(home.path().join("device_id"), "fixture-device").unwrap();
        let mut config = KimiBackendConfig::from_home(home.path());
        config.base_url = server.uri();
        config
            .custom_headers
            .insert("x-custom", "preserved".parse().unwrap());
        let provider = Arc::new(super::super::StaticToken {
            token: "supplied-key".into(),
        });
        let backend = config.build_with_token_provider(provider).unwrap();
        assert_eq!(backend.list_models().await.unwrap(), ["k3"]);
        assert!(!home.path().join("credentials").exists());
    }

    #[tokio::test]
    async fn static_key_and_rejected_token_never_start_refresh() {
        let home = tempfile::tempdir().unwrap();
        let mut config = KimiServiceConfig::from_home(home.path());
        config.executable = home.path().join("missing");
        config
            .environment
            .insert("KIMI_API_KEY".into(), "static".into());
        let service = KimiService::new(config).unwrap();
        assert_eq!(
            service.bearer_token().await.unwrap().as_deref(),
            Some("static")
        );
        assert_eq!(service.usage().await.unwrap()["kind"], "api_key");
        assert_eq!(service.rejected_bearer_token("static").await.unwrap(), None);
        assert!(!home.path().join("server").exists());
    }
    #[tokio::test]
    async fn kimi_auth_retry_requires_explicit_rejection_and_changed_token() {
        struct Replaced;
        impl BearerTokenProvider for Replaced {
            fn bearer_token(&self) -> BoxFuture<'_, Result<Option<String>>> {
                Box::pin(async { Ok(Some("old".into())) })
            }
            fn rejected_bearer_token<'a>(
                &'a self,
                rejected: &'a str,
            ) -> BoxFuture<'a, Result<Option<String>>> {
                assert_eq!(rejected, "old");
                Box::pin(async { Ok(Some("new".into())) })
            }
        }
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .and(header("authorization", "Bearer old"))
            .respond_with(ResponseTemplate::new(401))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .and(header("authorization", "Bearer new"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"data":[{"id":"k3"}]})),
            )
            .expect(1)
            .mount(&server)
            .await;
        let home = tempfile::tempdir().unwrap();
        let mut config = KimiBackendConfig::from_home(home.path());
        config.base_url = server.uri();
        let backend = config
            .build_with_token_provider(Arc::new(Replaced))
            .unwrap();
        assert_eq!(backend.list_models().await.unwrap(), ["k3"]);
    }

    #[tokio::test]
    async fn fresh_and_rejected_tokens_are_read_without_vendor_mutations() {
        let (home, mut config) = fixture();
        config.executable = home.path().join("not-an-executable");
        let credentials = home.path().join("credentials/kimi-code.json");
        let original =
            br#"{"access_token":"fresh","refresh_token":"untouched","expires_at":9999999999}"#;
        std::fs::write(&credentials, original).unwrap();
        let service = KimiService::new(config).unwrap();
        assert_eq!(
            service.bearer_token().await.unwrap().as_deref(),
            Some("fresh")
        );
        assert_eq!(service.rejected_bearer_token("fresh").await.unwrap(), None);
        assert_eq!(
            service
                .rejected_bearer_token("old")
                .await
                .unwrap()
                .as_deref(),
            Some("fresh")
        );
        assert_eq!(std::fs::read(credentials).unwrap(), original);
        assert!(!home.path().join("server").exists());
    }

    #[tokio::test]
    async fn vendor_process_inherits_runtime_lease() {
        let (home, mut config) = fixture();
        let lease_path = home.path().join("runtime.lease");
        std::fs::write(&lease_path, "").unwrap();
        config.lease_path = Some(lease_path.clone());
        let service = KimiService::new(config).unwrap();
        service.usage().await.unwrap();
        drop(service);
        let lease = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(lease_path)
            .unwrap();
        assert_eq!(
            lease.try_lock_exclusive().unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock
        );
        stop(home.path()).await;
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                match lease.try_lock_exclusive() {
                    Ok(()) => break,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        tokio::time::sleep(Duration::from_millis(10)).await
                    }
                    Err(error) => panic!("{error}"),
                }
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    #[ignore = "explicit isolated pinned-binary contract; set KIMI_VENDOR_CONTRACT_BINARY"]
    async fn pinned_kimi_service_contract_uses_only_fake_credentials() {
        let binary =
            std::env::var_os("KIMI_VENDOR_CONTRACT_BINARY").expect("explicit binary required");
        let (temporary, mut config) = fixture();
        let home = temporary.keep();
        eprintln!("isolated contract home: {}", home.display());
        config.executable = binary.into();
        let lease_path = home.join("runtime.lease");
        std::fs::write(&lease_path, "").unwrap();
        config.lease_path = Some(lease_path.clone());
        let server = MockServer::start().await;
        Mock::given(method("POST")).and(path("/api/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token":"contract-refreshed","refresh_token":"contract-rotated","expires_in":900,"token_type":"Bearer","scope":"kimi-code"
            }))).mount(&server).await;
        Mock::given(method("GET"))
            .and(path("/coding/v1/usages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "usages":{"limit_5h":{"used_ratio":0.1}}, "boosterWallet":null
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/coding/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"data":[]})))
            .mount(&server)
            .await;
        config
            .environment
            .insert("KIMI_CODE_OAUTH_HOST".into(), server.uri());
        config.environment.insert(
            "KIMI_CODE_BASE_URL".into(),
            format!("{}/coding/v1", server.uri()),
        );
        let scoped = config.credentials_path().unwrap();
        std::fs::copy(home.join("credentials/kimi-code.json"), &scoped).unwrap();
        let service = KimiService::new(config).unwrap();
        let result = service.usage().await;
        if let Err(error) = &result {
            eprintln!("vendor contract failure: {error:#}");
        }
        let record: ServiceRecord = serde_json::from_slice(
            &std::fs::read(home.as_path().join("server/anvil-service.json")).unwrap(),
        )
        .unwrap();
        let lease = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&lease_path)
            .unwrap();
        let lease_is_retained = lease.try_lock_exclusive().is_err();
        let token = server_token(home.as_path()).await.unwrap();
        let stop_result = reqwest::Client::new()
            .post(format!("http://127.0.0.1:{}/api/v1/shutdown", record.port))
            .bearer_auth(token)
            .send()
            .await;
        assert!(stop_result.unwrap().status().is_success());
        assert!(
            lease_is_retained,
            "vendor must retain its managed executable lease"
        );
        assert_eq!(result.unwrap()["kind"], "ok");
        assert_eq!(service.read_token().await.unwrap(), "contract-refreshed");
    }
    #[test]
    #[ignore = "isolated daemon-exit fixture"]
    fn fake_exiting_caller() {
        let home = PathBuf::from(std::env::var_os("KIMI_TEST_CALLER_HOME").unwrap());
        let mut config = KimiServiceConfig::from_home(&home);
        config.executable = home.join("kimi");
        config
            .environment
            .insert("KIMI_TEST_STARTUP_DELAY_MS".into(), "150".into());
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let service = KimiService::new(config).unwrap();
            tokio::spawn(async move { service.usage().await });
            let marker = if std::env::var_os("KIMI_TEST_EXIT_DURING_LAUNCH").is_some() {
                home.join("server/anvil-service.json")
            } else {
                home.join("usage-started")
            };
            tokio::time::timeout(Duration::from_secs(10), async {
                while !marker.is_file() {
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
            })
            .await
            .unwrap();
            std::process::exit(0);
        });
    }

    async fn check_caller_exit(during_launch: bool) {
        let (home, mut config) = fixture();
        config
            .environment
            .insert("KIMI_TEST_STARTUP_DELAY_MS".into(), "150".into());
        let caller_home = home.path().to_owned();
        let status = tokio::task::spawn_blocking(move || {
            let mut command = Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "--exact",
                    "kimi_auth::service::tests::fake_exiting_caller",
                    "--ignored",
                    "--nocapture",
                ])
                .env("KIMI_TEST_CALLER_HOME", caller_home)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            if during_launch {
                command.env("KIMI_TEST_EXIT_DURING_LAUNCH", "1");
            }
            command.status().unwrap()
        })
        .await
        .unwrap();
        assert!(status.success());
        let before: ServiceRecord = serde_json::from_slice(
            &tokio::fs::read(home.path().join("server/anvil-service.json"))
                .await
                .unwrap(),
        )
        .unwrap();
        let replacement = KimiService::new(config).unwrap();
        replacement.usage().await.unwrap();
        let after: ServiceRecord = serde_json::from_slice(
            &tokio::fs::read(home.path().join("server/anvil-service.json"))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            before.pid, after.pid,
            "handoff must adopt the same vendor process"
        );
        assert_eq!(replacement.read_token().await.unwrap(), "fresh");
        stop(home.path()).await;
    }

    #[tokio::test]
    async fn daemon_exit_during_launch_adopts_pending_vendor_process() {
        check_caller_exit(true).await;
    }

    #[tokio::test]
    async fn daemon_exit_during_refresh_preserves_vendor_persistence() {
        check_caller_exit(false).await;
    }
    #[tokio::test]
    async fn changed_service_configuration_drains_and_replaces_owned_vendor() {
        let (home, config) = fixture();
        let original = KimiService::new(config.clone()).unwrap();
        original.usage().await.unwrap();
        let before: ServiceRecord = serde_json::from_slice(
            &std::fs::read(home.path().join("server/anvil-service.json")).unwrap(),
        )
        .unwrap();
        let mut config = config;
        config
            .environment
            .insert("KIMI_CODE_CUSTOM_HEADERS".into(), "X-Test: changed".into());
        let changed = KimiService::new(config).unwrap();
        changed.usage().await.unwrap();
        let after: ServiceRecord = serde_json::from_slice(
            &std::fs::read(home.path().join("server/anvil-service.json")).unwrap(),
        )
        .unwrap();
        assert_ne!(before.server_id, after.server_id);
        assert_ne!(before.pid, after.pid);
        assert!(
            !home
                .path()
                .join(format!("server/instances/{}.json", before.server_id))
                .exists()
        );
        stop(home.path()).await;
    }

    #[tokio::test]
    async fn kimi_auth_retry_never_replays_unchanged_credentials() {
        struct Unchanged;
        impl BearerTokenProvider for Unchanged {
            fn bearer_token(&self) -> BoxFuture<'_, Result<Option<String>>> {
                Box::pin(async { Ok(Some("same".into())) })
            }
            fn rejected_bearer_token<'a>(
                &'a self,
                _: &'a str,
            ) -> BoxFuture<'a, Result<Option<String>>> {
                Box::pin(async { Ok(Some("same".into())) })
            }
        }
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(401))
            .expect(1)
            .mount(&server)
            .await;
        let home = tempfile::tempdir().unwrap();
        let mut config = KimiBackendConfig::from_home(home.path());
        config.base_url = server.uri();
        assert!(
            config
                .build_with_token_provider(Arc::new(Unchanged))
                .unwrap()
                .list_models()
                .await
                .is_err()
        );
    }
}
