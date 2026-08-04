//! Black-box conformance suite for the Anvil HTTP API (#320).
//!
//! Starts the packaged `anvil serve` daemon and validates every JSON
//! response and every SSE event it produces against the authoritative
//! contract in `openapi/anvil.v1.yaml` and
//! `openapi/anvil.v1.events.schema.json`. CI runs this suite on every
//! change, so the implementation and the checked-in contract cannot drift
//! silently: a handler change that alters a wire shape fails here until
//! the contract (and its version) are updated in the same pull request.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

fn contract_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("openapi")
}

fn load_openapi() -> Value {
    let raw = std::fs::read_to_string(contract_dir().join("anvil.v1.yaml"))
        .expect("read openapi/anvil.v1.yaml");
    let yaml: serde_yaml::Value = serde_yaml::from_str(&raw).expect("parse OpenAPI YAML");
    serde_json::to_value(yaml).expect("OpenAPI YAML converts to JSON")
}

fn load_event_schema() -> Value {
    let raw = std::fs::read_to_string(contract_dir().join("anvil.v1.events.schema.json"))
        .expect("read openapi/anvil.v1.events.schema.json");
    serde_json::from_str(&raw).expect("parse event schema JSON")
}

/// Resolve one level of `$ref` indirection on a non-schema object (for
/// example a shared response component).
fn resolve_ref<'a>(value: &'a Value, root: &'a Value) -> &'a Value {
    if let Some(Value::String(reference)) = value.get("$ref") {
        let pointer = reference
            .strip_prefix('#')
            .unwrap_or_else(|| panic!("non-local $ref '{reference}'"));
        return root
            .pointer(pointer)
            .unwrap_or_else(|| panic!("dangling $ref '{reference}'"));
    }
    value
}

/// Resolve `#/components/schemas/...` references by inlining, producing a
/// self-contained JSON Schema for one response body. The contract's
/// schemas are acyclic, and the depth guard turns an accidental future
/// cycle into a loud failure instead of a hang.
fn inline_refs(value: &Value, root: &Value, depth: usize) -> Value {
    assert!(
        depth < 64,
        "$ref inlining exceeded depth 64 (cycle in contract?)"
    );
    match value {
        Value::Object(map) => {
            if let Some(Value::String(reference)) = map.get("$ref") {
                let pointer = reference
                    .strip_prefix("#")
                    .unwrap_or_else(|| panic!("non-local $ref '{reference}'"));
                let target = root
                    .pointer(pointer)
                    .unwrap_or_else(|| panic!("dangling $ref '{reference}'"));
                return inline_refs(target, root, depth + 1);
            }
            Value::Object(
                map.iter()
                    .map(|(key, entry)| (key.clone(), inline_refs(entry, root, depth + 1)))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|entry| inline_refs(entry, root, depth + 1))
                .collect(),
        ),
        other => other.clone(),
    }
}

struct Contract {
    openapi: Value,
    event_validator: jsonschema::Validator,
}

impl Contract {
    fn load() -> Self {
        let openapi = load_openapi();
        let event_validator =
            jsonschema::validator_for(&load_event_schema()).expect("compile event schema");
        Self {
            openapi,
            event_validator,
        }
    }

    /// Validate a JSON response body against the schema the contract
    /// declares for `method path -> status`. Response objects may be
    /// `$ref`s to shared components (the error envelope responses are).
    fn check_response(&self, method: &str, path: &str, status: u16, body: &Value) {
        let pointer = format!(
            "/paths/{}/{}/responses/{}",
            path.replace('~', "~0").replace('/', "~1"),
            method.to_ascii_lowercase(),
            status,
        );
        let response = self.openapi.pointer(&pointer).unwrap_or_else(|| {
            panic!("contract has no response for {method} {path} -> {status} ({pointer})")
        });
        let response = resolve_ref(response, &self.openapi);
        let schema = response
            .pointer("/content/application~1json/schema")
            .unwrap_or_else(|| {
                panic!("contract has no JSON schema for {method} {path} -> {status}")
            });
        let inlined = inline_refs(schema, &self.openapi, 0);
        let validator = jsonschema::validator_for(&inlined)
            .unwrap_or_else(|e| panic!("compile schema for {method} {path} {status}: {e}"));
        let errors: Vec<String> = validator
            .iter_errors(body)
            .map(|error| format!("{} at {}", error, error.instance_path()))
            .collect();
        assert!(
            errors.is_empty(),
            "{method} {path} -> {status} violates the contract:\n{}\nbody: {body}",
            errors.join("\n"),
        );
    }

    fn check_event(&self, payload: &Value) {
        let errors: Vec<String> = self
            .event_validator
            .iter_errors(payload)
            .map(|error| format!("{} at {}", error, error.instance_path()))
            .collect();
        assert!(
            errors.is_empty(),
            "SSE event violates the contract:\n{}\nevent: {payload}",
            errors.join("\n"),
        );
    }
}

// ---------------------------------------------------------------------------
// Daemon harness (mirrors tests/http_smoke.rs)
// ---------------------------------------------------------------------------

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
            "conformance::model",
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

    let stdout = child.stdout.take().expect("child stdout");
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut lines = BufReader::new(stdout).lines();
        if let Some(Ok(line)) = lines.next() {
            let _ = sender.send(line);
        }
    });
    let ready_line = receiver
        .recv_timeout(Duration::from_secs(120))
        .expect("serve.ready line on stdout before timeout");
    let ready: Value = serde_json::from_str(&ready_line).expect("serve.ready line is JSON");
    let base_url = ready["url"].as_str().expect("ready url").to_string();
    ServeDaemon { child, base_url }
}

/// Minimal blocking HTTP/1.1 client with a real chunked-transfer decoder,
/// so SSE bodies come back byte-accurate for frame parsing.
fn raw_request(method: &str, base: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let host = base.strip_prefix("http://").expect("base url");
    let mut stream = std::net::TcpStream::connect(host).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(60)))
        .expect("read timeout");
    let payload = body.unwrap_or("");
    let request = format!(
        "{method} {path} HTTP/1.1\r\nhost: {host}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
        payload.len(),
    );
    stream.write_all(request.as_bytes()).expect("write request");
    let mut response = Vec::new();
    stream.read_to_end(&mut response).expect("read response");
    let response = String::from_utf8_lossy(&response).into_owned();
    let (head, body) = response
        .split_once("\r\n\r\n")
        .unwrap_or_else(|| panic!("malformed response: {response}"));
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|token| token.parse().ok())
        .unwrap_or_else(|| panic!("malformed status line: {head}"));
    let body = if head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        decode_chunked(body)
    } else {
        body.to_string()
    };
    (status, body)
}

fn decode_chunked(raw: &str) -> String {
    let mut decoded = String::new();
    let mut rest = raw;
    while let Some((size_line, tail)) = rest.split_once("\r\n") {
        let size = usize::from_str_radix(size_line.trim(), 16).unwrap_or(0);
        if size == 0 {
            break;
        }
        if tail.len() < size {
            decoded.push_str(tail);
            break;
        }
        decoded.push_str(&tail[..size]);
        rest = tail[size..].strip_prefix("\r\n").unwrap_or(&tail[size..]);
    }
    decoded
}

fn get_json(contract: &Contract, base: &str, template: &str, actual: &str) -> Value {
    let (status, body) = raw_request("GET", base, actual, None);
    let body: Value =
        serde_json::from_str(&body).unwrap_or_else(|e| panic!("GET {actual}: non-JSON body ({e})"));
    contract.check_response("get", template, status, &body);
    body
}

/// One SSE frame parsed from the stream body.
struct SseFrame {
    id: Option<u64>,
    event: String,
    data: Value,
}

fn parse_sse(body: &str) -> Vec<SseFrame> {
    body.split("\n\n")
        .filter_map(|frame| {
            let mut id = None;
            let mut event = None;
            let mut data = None;
            for line in frame.lines() {
                if let Some(value) = line.strip_prefix("id: ") {
                    id = value.trim().parse().ok();
                } else if let Some(value) = line.strip_prefix("event: ") {
                    event = Some(value.trim().to_string());
                } else if let Some(value) = line.strip_prefix("data: ") {
                    data = Some(value.to_string());
                }
            }
            match (event, data) {
                (Some(event), Some(data)) => Some(SseFrame {
                    id,
                    event,
                    data: serde_json::from_str(&data).expect("SSE data is JSON"),
                }),
                _ => None,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The suite
// ---------------------------------------------------------------------------

#[test]
fn daemon_conforms_to_checked_in_contract() {
    let contract = Contract::load();
    let home = tempfile::tempdir().expect("temp home");
    let workspace = tempfile::tempdir().expect("workspace");
    let daemon = spawn_serve(home.path());
    let base = &daemon.base_url;
    let cwd = workspace.path().display().to_string();

    // Server + catalog endpoints.
    get_json(&contract, base, "/health", "/health");
    get_json(&contract, base, "/v1/models", "/v1/models");
    get_json(&contract, base, "/v1/tools", "/v1/tools");

    // Error envelopes.
    let (status, body) = raw_request("GET", base, "/v1/sessions/no-such-session", None);
    assert_eq!(status, 404);
    contract.check_response(
        "get",
        "/v1/sessions/{session_id}",
        404,
        &serde_json::from_str(&body).expect("error body JSON"),
    );
    let (status, body) = raw_request(
        "POST",
        base,
        "/v1/sessions",
        Some(&json!({ "cwd": "relative/path" }).to_string()),
    );
    assert_eq!(status, 400);
    contract.check_response(
        "post",
        "/v1/sessions",
        400,
        &serde_json::from_str(&body).expect("error body JSON"),
    );

    // Session lifecycle.
    let (status, body) = raw_request(
        "POST",
        base,
        "/v1/sessions",
        Some(&json!({ "cwd": cwd, "permission_mode": "readOnly" }).to_string()),
    );
    assert_eq!(status, 201, "create session failed: {body}");
    let created: Value = serde_json::from_str(&body).expect("session JSON");
    contract.check_response("post", "/v1/sessions", 201, &created);
    let session_id = created["id"].as_str().expect("session id").to_string();

    let session = get_json(
        &contract,
        base,
        "/v1/sessions/{session_id}",
        &format!("/v1/sessions/{session_id}?include_history=true"),
    );
    assert!(session["history"].is_array());
    get_json(&contract, base, "/v1/sessions", "/v1/sessions");

    let (status, body) = raw_request(
        "PATCH",
        base,
        &format!("/v1/sessions/{session_id}"),
        Some(&json!({ "behavior_mode": "PLAN" }).to_string()),
    );
    assert_eq!(status, 200, "patch failed: {body}");
    contract.check_response(
        "patch",
        "/v1/sessions/{session_id}",
        200,
        &serde_json::from_str(&body).expect("patch JSON"),
    );

    // Load and resume (history present on load).
    let (status, body) = raw_request(
        "POST",
        base,
        &format!("/v1/sessions/{session_id}/load"),
        Some(&json!({ "cwd": cwd }).to_string()),
    );
    assert_eq!(status, 200, "load failed: {body}");
    let loaded: Value = serde_json::from_str(&body).expect("load JSON");
    contract.check_response("post", "/v1/sessions/{session_id}/load", 200, &loaded);
    assert!(loaded["history"].is_array());
    let (status, body) = raw_request(
        "POST",
        base,
        &format!("/v1/sessions/{session_id}/resume"),
        Some(&json!({ "cwd": cwd }).to_string()),
    );
    assert_eq!(status, 200, "resume failed: {body}");
    contract.check_response(
        "post",
        "/v1/sessions/{session_id}/resume",
        200,
        &serde_json::from_str(&body).expect("resume JSON"),
    );

    // Runs: the conformance environment has no LLM providers, so the run
    // fails — which exercises the terminal contract just as well.
    let (status, body) = raw_request(
        "POST",
        base,
        &format!("/v1/sessions/{session_id}/runs"),
        Some(&json!({ "prompt": "conformance probe" }).to_string()),
    );
    assert_eq!(status, 202, "run not accepted: {body}");
    let run: Value = serde_json::from_str(&body).expect("run JSON");
    contract.check_response("post", "/v1/sessions/{session_id}/runs", 202, &run);
    let run_id = run["id"].as_str().expect("run id").to_string();

    let deadline = Instant::now() + Duration::from_secs(60);
    let terminal = loop {
        let run = get_json(
            &contract,
            base,
            "/v1/runs/{run_id}",
            &format!("/v1/runs/{run_id}"),
        );
        if run["status"] != "running" {
            break run;
        }
        assert!(Instant::now() < deadline, "run stuck: {run}");
        std::thread::sleep(Duration::from_millis(200));
    };
    assert_eq!(terminal["status"], "failed");

    get_json(
        &contract,
        base,
        "/v1/sessions/{session_id}/runs",
        &format!("/v1/sessions/{session_id}/runs"),
    );
    get_json(
        &contract,
        base,
        "/v1/runs/{run_id}/permissions",
        &format!("/v1/runs/{run_id}/permissions"),
    );
    let (status, body) = raw_request("POST", base, &format!("/v1/runs/{run_id}/cancel"), None);
    assert_eq!(status, 200);
    contract.check_response(
        "post",
        "/v1/runs/{run_id}/cancel",
        200,
        &serde_json::from_str(&body).expect("cancel JSON"),
    );

    // Every SSE event must satisfy the event schema; ids must equal the
    // enveloped sequence numbers and ascend; the stream must end on a
    // terminal event.
    let (status, events_body) =
        raw_request("GET", base, &format!("/v1/runs/{run_id}/events"), None);
    assert_eq!(status, 200);
    let frames = parse_sse(&events_body);
    assert!(
        frames.len() >= 2,
        "expected at least run.created + terminal, got: {events_body}"
    );
    let mut last_seq = 0;
    for frame in &frames {
        contract.check_event(&frame.data);
        assert_eq!(
            frame.data["type"].as_str().expect("event type"),
            frame.event,
            "SSE event name must equal the payload type"
        );
        if let Some(id) = frame.id {
            assert_eq!(frame.data["seq"].as_u64(), Some(id));
            assert!(id > last_seq, "sequence ids must ascend");
            last_seq = id;
        }
    }
    assert_eq!(frames.first().expect("first frame").event, "run.created");
    assert_eq!(frames.last().expect("last frame").event, "run.failed");

    // Permission endpoints reject unknown ids with the documented envelope.
    let (status, body) = raw_request("GET", base, "/v1/permissions/perm-nope", None);
    assert_eq!(status, 404);
    contract.check_response(
        "get",
        "/v1/permissions/{permission_id}",
        404,
        &serde_json::from_str(&body).expect("error JSON"),
    );

    // Deletion.
    let (status, body) = raw_request("DELETE", base, &format!("/v1/sessions/{session_id}"), None);
    assert_eq!(status, 200);
    contract.check_response(
        "delete",
        "/v1/sessions/{session_id}",
        200,
        &serde_json::from_str(&body).expect("delete JSON"),
    );
}

/// The contract artifacts themselves must stay well-formed: the OpenAPI
/// document parses, every `$ref` resolves, and the event schema compiles.
#[test]
fn contract_artifacts_are_well_formed() {
    let openapi = load_openapi();
    assert_eq!(openapi["openapi"], "3.1.0");
    assert!(openapi["info"]["version"].is_string());

    // Walk every declared JSON response schema and inline it, which fails
    // loudly on dangling refs; then verify it compiles.
    let paths = openapi["paths"].as_object().expect("paths object");
    let mut checked = 0;
    for (path, item) in paths {
        for (method, operation) in item.as_object().expect("path item") {
            if method == "parameters" {
                continue;
            }
            let Some(responses) = operation.get("responses").and_then(Value::as_object) else {
                continue;
            };
            for (status, response) in responses {
                let response = resolve_ref(response, &openapi);
                if let Some(schema) = response.pointer("/content/application~1json/schema") {
                    let inlined = inline_refs(schema, &openapi, 0);
                    jsonschema::validator_for(&inlined).unwrap_or_else(|e| {
                        panic!("schema for {method} {path} {status} does not compile: {e}")
                    });
                    checked += 1;
                }
            }
        }
    }
    assert!(
        checked > 20,
        "expected to compile many schemas, got {checked}"
    );

    jsonschema::validator_for(&load_event_schema()).expect("event schema compiles");
}
