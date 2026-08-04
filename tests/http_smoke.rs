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

fn spawn_serve(home: &std::path::Path) -> ServeDaemon {
    let bin = std::env::var_os("CARGO_BIN_EXE_anvil")
        .map(PathBuf::from)
        .or_else(|| option_env!("CARGO_BIN_EXE_anvil").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("target/debug/anvil"));
    let mut child = Command::new(bin)
        .args([
            "--no-wasm-sandbox",
            "--transient-setup",
            "--default-model",
            "smoke::model",
            "serve",
            "--port",
            "0",
        ])
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

    ServeDaemon { child, base_url }
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
    let request = format!(
        "{method} {path} HTTP/1.1\r\nhost: {host}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
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

    let (status, deleted) = request("DELETE", &format!("{base}/v1/sessions/{session_id}"), None)
        .expect("delete session");
    assert_eq!(status, 200);
    assert_eq!(deleted["deleted"], true);
}
