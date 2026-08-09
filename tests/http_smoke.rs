#![cfg(feature = "http-api")]

//! Daemon-level smoke test for `anvil serve` (#317): spawns the real
//! binary, waits for the machine-readable `serve.ready` line on stdout,
//! and exercises the REST session lifecycle over a real localhost
//! listener. Handler-level coverage (validation, error envelopes, config
//! selectors) lives in `src/http_api/tests.rs`; this test proves the
//! subcommand wiring, loopback binding, ephemeral-port reporting, and
//! stdout/stderr discipline of the packaged daemon.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

struct ServeDaemon {
    child: Child,
    base_url: String,
}

impl Drop for ServeDaemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn anvil_bin() -> PathBuf {
    std::env::var_os("CARGO_BIN_EXE_anvil")
        .map(PathBuf::from)
        .or_else(|| option_env!("CARGO_BIN_EXE_anvil").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("target/debug/anvil"))
}

fn spawn_serve(home: &std::path::Path) -> ServeDaemon {
    spawn_serve_with(home, &[]).0
}

fn spawn_serve_with(home: &std::path::Path, extra_args: &[&str]) -> (ServeDaemon, Value) {
    let mut args = vec![
        "--no-wasm-sandbox",
        "--transient-setup",
        "--default-model",
        "smoke::model",
        "serve",
        "--port",
        "0",
    ];
    args.extend_from_slice(extra_args);
    let mut child = Command::new(anvil_bin())
        .args(&args)
        .env("HOME", home)
        .env("CODEX_HOME", home.join(".codex"))
        .env("BROKK_CONFIG_HOME", home.join("config"))
        .env_remove("OPENAI_API_KEY")
        .env_remove("OPENROUTER_API_KEY")
        .env_remove("BEDROCK_API_KEY")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn anvil serve");

    // The daemon promises exactly one machine-readable stdout line:
    // {"type":"serve.ready","url":...}. Startup includes provider discovery
    // probes with their own timeouts, so allow a generous deadline.
    let stdout = child.stdout.take().expect("child stdout");
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut lines = BufReader::new(stdout).lines();
        if let Some(Ok(line)) = lines.next() {
            let _ = sender.send(line);
        }
    });
    let deadline = Duration::from_secs(120);
    let ready_line = receiver
        .recv_timeout(deadline)
        .expect("serve.ready line on stdout before timeout");
    let ready: Value = serde_json::from_str(&ready_line).expect("serve.ready line is JSON");
    assert_eq!(ready["type"], "serve.ready");
    let base_url = ready["url"].as_str().expect("ready url").to_string();
    assert!(
        base_url.starts_with("http://127.0.0.1:"),
        "daemon must bind loopback, got {base_url}"
    );

    (ServeDaemon { child, base_url }, ready)
}

fn http_get(url: &str) -> (u16, Value) {
    let started = Instant::now();
    loop {
        match ureq_get(url) {
            Ok(result) => return result,
            Err(err) if started.elapsed() < Duration::from_secs(10) => {
                eprintln!("retrying {url}: {err}");
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(err) => panic!("GET {url} failed: {err}"),
        }
    }
}

/// Minimal blocking HTTP/1.1 client over std TcpStream: enough for
/// loopback JSON smoke checks without pulling an async runtime into this
/// test binary.
fn ureq_get(url: &str) -> Result<(u16, Value), String> {
    request("GET", url, None)
}

fn request(method: &str, url: &str, body: Option<&str>) -> Result<(u16, Value), String> {
    request_with_auth(method, url, body, None)
}

fn request_with_auth(
    method: &str,
    url: &str,
    body: Option<&str>,
    bearer: Option<&str>,
) -> Result<(u16, Value), String> {
    use std::io::{Read, Write};

    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| format!("unsupported url {url}"))?;
    let (host, path) = rest.split_once('/').unwrap_or((rest, ""));
    let path = format!("/{path}");
    let mut stream = std::net::TcpStream::connect(host).map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .map_err(|e| e.to_string())?;
    let payload = body.unwrap_or("");
    let auth_header = bearer
        .map(|token| format!("authorization: Bearer {token}\r\n"))
        .unwrap_or_default();
    let request = format!(
        "{method} {path} HTTP/1.1\r\nhost: {host}\r\n{auth_header}content-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
        payload.len(),
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|e| e.to_string())?;
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .map_err(|e| e.to_string())?;
    let (head, body) = response
        .split_once("\r\n\r\n")
        .ok_or_else(|| format!("malformed response: {response}"))?;
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("malformed status line: {head}"))?;
    // Responses may be chunked; both axum JSON bodies here are single-chunk,
    // so strip chunk framing when present.
    let body = if head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        body.lines()
            .skip(1)
            .take_while(|line| *line != "0")
            .collect::<Vec<_>>()
            .join("")
    } else {
        body.to_string()
    };
    let value =
        serde_json::from_str(body.trim()).map_err(|e| format!("non-JSON body ({e}): {body:?}"))?;
    Ok((status, value))
}

#[test]
fn serve_daemon_lifecycle_over_localhost() {
    let home = tempfile::tempdir().expect("temp home");
    let workspace = tempfile::tempdir().expect("workspace");
    let daemon = spawn_serve(home.path());
    let base = &daemon.base_url;

    // Readiness.
    let (status, health) = http_get(&format!("{base}/health"));
    assert_eq!(status, 200);
    assert_eq!(health["status"], "ok");

    // Models and tools respond (no providers are configured in the smoke
    // environment, so the catalog may be empty -- shape only).
    let (status, models) = http_get(&format!("{base}/v1/models"));
    assert_eq!(status, 200);
    assert!(models["models"].is_array());
    let (status, tools) = http_get(&format!("{base}/v1/tools"));
    assert_eq!(status, 200);
    assert!(
        tools["tools"]
            .as_array()
            .expect("tools array")
            .iter()
            .any(|t| t["name"] == "read_file")
    );

    // Session lifecycle: create -> inspect -> configure -> delete.
    let cwd = workspace.path().display().to_string();
    let (status, created) = request(
        "POST",
        &format!("{base}/v1/sessions"),
        Some(&serde_json::json!({ "cwd": cwd, "permission_mode": "readOnly" }).to_string()),
    )
    .expect("create session");
    assert_eq!(status, 201, "create failed: {created}");
    assert_eq!(created["permission_mode"], "readOnly");
    let session_id = created["id"].as_str().expect("session id").to_string();

    let (status, fetched) = http_get(&format!("{base}/v1/sessions/{session_id}"));
    assert_eq!(status, 200);
    assert_eq!(fetched["id"], session_id.as_str());

    let (status, patched) = request(
        "PATCH",
        &format!("{base}/v1/sessions/{session_id}"),
        Some(&serde_json::json!({ "behavior_mode": "PLAN" }).to_string()),
    )
    .expect("patch session");
    assert_eq!(status, 200, "patch failed: {patched}");
    assert_eq!(patched["session"]["behavior_mode"], "PLAN");

    // Unknown sessions surface the documented envelope.
    let (status, missing) = http_get(&format!("{base}/v1/sessions/no-such-session"));
    assert_eq!(status, 404);
    assert_eq!(missing["error"]["code"], "not_found");
    assert!(missing["request_id"].is_string());

    // Asynchronous run lifecycle (#318): the smoke environment has no LLM
    // providers, so the run is accepted, executes, and reports a failed
    // terminal state with the result retained for polling.
    let (status, run) = request(
        "POST",
        &format!("{base}/v1/sessions/{session_id}/runs"),
        Some(&serde_json::json!({ "prompt": "hello over http" }).to_string()),
    )
    .expect("create run");
    assert_eq!(status, 202, "run not accepted: {run}");
    let run_id = run["id"].as_str().expect("run id").to_string();
    assert_eq!(run["status"], "running");

    let deadline = Instant::now() + Duration::from_secs(60);
    let terminal = loop {
        let (status, run) = http_get(&format!("{base}/v1/runs/{run_id}"));
        assert_eq!(status, 200);
        if run["status"] != "running" {
            break run;
        }
        assert!(
            Instant::now() < deadline,
            "run did not reach a terminal state in time: {run}"
        );
        std::thread::sleep(Duration::from_millis(200));
    };
    assert_eq!(terminal["status"], "failed", "expected failure: {terminal}");
    assert_eq!(terminal["stop_reason"], "error");
    assert!(terminal["last_seq"].as_u64().expect("last_seq") >= 2);

    // Cancel stays idempotent once the run is terminal.
    let (status, _) = request("POST", &format!("{base}/v1/runs/{run_id}/cancel"), None)
        .expect("cancel terminal run");
    assert_eq!(status, 200, "cancel must be idempotent on terminal runs");

    let (status, deleted) = request("DELETE", &format!("{base}/v1/sessions/{session_id}"), None)
        .expect("delete session");
    assert_eq!(status, 200);
    assert_eq!(deleted["deleted"], true);
}

#[test]
fn serve_refuses_non_loopback_binding_without_auth() {
    let home = tempfile::tempdir().expect("temp home");
    let status = Command::new(anvil_bin())
        .args([
            "--no-wasm-sandbox",
            "--transient-setup",
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "0",
        ])
        .env("HOME", home.path())
        .env("BROKK_CONFIG_HOME", home.path().join("config"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("run anvil serve");
    assert!(
        !status.success(),
        "non-loopback binding without a token must fail startup"
    );
}

#[test]
fn serve_generated_token_gates_v1_endpoints() {
    let home = tempfile::tempdir().expect("temp home");
    let (daemon, ready) = spawn_serve_with(home.path(), &["--generate-auth-token"]);
    let base = &daemon.base_url;
    assert_eq!(ready["auth_required"], true);
    let token = ready["auth_token"].as_str().expect("generated token");

    // Liveness stays open; /v1 requires the bearer token.
    let (status, _) = http_get(&format!("{base}/health"));
    assert_eq!(status, 200);
    let (status, body) =
        request_with_auth("GET", &format!("{base}/v1/models"), None, None).expect("no token");
    assert_eq!(status, 401, "expected 401, got {body}");
    assert_eq!(body["error"]["code"], "unauthorized");
    let (status, _) = request_with_auth(
        "GET",
        &format!("{base}/v1/models"),
        None,
        Some("wrong-token"),
    )
    .expect("wrong token");
    assert_eq!(status, 401);
    let (status, body) = request_with_auth("GET", &format!("{base}/v1/models"), None, Some(token))
        .expect("valid token");
    assert_eq!(status, 200, "expected 200, got {body}");
}

#[test]
fn serve_rejects_non_loopback_host_header() {
    let home = tempfile::tempdir().expect("temp home");
    let daemon = spawn_serve(home.path());
    let base = &daemon.base_url;
    let host_port = base.strip_prefix("http://").expect("base url");

    // DNS-rebinding guard: same TCP destination, hostile Host header.
    let (status, body) = request_with_host("GET", host_port, "/v1/models", "evil.example")
        .expect("request with hostile host");
    assert_eq!(status, 403, "expected 403, got {body}");
    assert_eq!(body["error"]["code"], "forbidden");

    // The genuine loopback name keeps working.
    let (status, _) = request_with_host("GET", host_port, "/v1/models", "localhost")
        .expect("request with localhost host");
    assert_eq!(status, 200);
}

/// Raw request with an explicit Host header value (the shared helpers use
/// the connection address, which is exactly what the rebinding guard
/// accepts).
fn request_with_host(
    method: &str,
    connect_to: &str,
    path: &str,
    host_header: &str,
) -> Result<(u16, Value), String> {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect(connect_to).map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .map_err(|e| e.to_string())?;
    let request =
        format!("{method} {path} HTTP/1.1\r\nhost: {host_header}\r\nconnection: close\r\n\r\n");
    stream
        .write_all(request.as_bytes())
        .map_err(|e| e.to_string())?;
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .map_err(|e| e.to_string())?;
    let (head, body) = response
        .split_once("\r\n\r\n")
        .ok_or_else(|| format!("malformed response: {response}"))?;
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("malformed status line: {head}"))?;
    let body = if head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        body.lines()
            .skip(1)
            .take_while(|line| *line != "0")
            .collect::<Vec<_>>()
            .join("")
    } else {
        body.to_string()
    };
    let value =
        serde_json::from_str(body.trim()).map_err(|e| format!("non-JSON body ({e}): {body:?}"))?;
    Ok((status, value))
}
