#![cfg(feature = "http-api")]

//! Black-box conformance suite for the Anvil HTTP API (#320).
//!
//! Starts the packaged `anvil serve` daemon against a scripted
//! OpenAI-compatible mock provider (the `ANVIL_TEST_OLLAMA_BASE_URL` test
//! hook, same as `tests/acp_smoke.rs`) and validates every JSON response
//! and every SSE event against the authoritative contract in
//! `openapi/anvil.v1.yaml` and `openapi/anvil.v1.events.schema.json`.
//!
//! The scripted provider drives real turns through the daemon, so every
//! event type the contract defines is actually produced by the
//! implementation and schema-checked live: text and thought deltas, plan
//! updates, the full tool-call lifecycle (started, oversized/malformed
//! input, blocked, in-progress, failed, completed-with-diff), permission
//! request/resolution, and completed / cancelled / failed terminals. The
//! only exception is `events.gap`, which requires overflowing the 8192
//! event replay buffer; it stays schema-validated via the artifact test.
//!
//! CI runs this suite on every change, so a handler change that alters a
//! wire shape fails here until the contract (and its version) are updated
//! in the same pull request.

use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Contract loading and validation
// ---------------------------------------------------------------------------

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
                    .strip_prefix('#')
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
// Scripted OpenAI-compatible provider
// ---------------------------------------------------------------------------

/// One scripted provider connection: answer with a canned SSE body, fail
/// with a client error (for failed-terminal coverage), or hold the socket
/// open without responding (for cancellation coverage).
enum ProviderScript {
    Respond(String),
    Fail,
    Hang,
}

fn start_mock_provider(script: Vec<ProviderScript>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock provider");
    let base_url = format!("http://{}", listener.local_addr().expect("provider addr"));
    std::thread::spawn(move || {
        for (index, stream) in listener.incoming().enumerate() {
            let Ok(mut stream) = stream else { break };
            let Some(step) = script.get(index) else { break };
            read_provider_request(&mut stream);
            match step {
                ProviderScript::Respond(body) => {
                    let response = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(response.as_bytes());
                    let _ = stream.flush();
                }
                ProviderScript::Fail => {
                    // A 4xx is non-retryable, so the turn fails immediately
                    // and deterministically.
                    let body = r#"{"error":{"message":"scripted failure"}}"#;
                    let response = format!(
                        "HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(response.as_bytes());
                    let _ = stream.flush();
                }
                ProviderScript::Hang => {
                    // Hold the socket open past the test window on a side
                    // thread so later scripted connections are still
                    // accepted; the run is cancelled while the client waits
                    // on this stream.
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_secs(60));
                        drop(stream);
                    });
                }
            }
            if index + 1 == script.len() {
                break;
            }
        }
    });
    base_url
}

fn read_provider_request(stream: &mut TcpStream) {
    let mut raw = Vec::new();
    let mut buf = [0_u8; 1024];
    loop {
        let Ok(read) = stream.read(&mut buf) else {
            return;
        };
        if read == 0 {
            return;
        }
        raw.extend_from_slice(&buf[..read]);
        if let Some(header_end) = find_subslice(&raw, b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&raw[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    line.split_once(':').and_then(|(key, value)| {
                        key.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                })
                .unwrap_or(0);
            let body_start = header_end + 4;
            while raw.len().saturating_sub(body_start) < content_length {
                let Ok(read) = stream.read(&mut buf) else {
                    return;
                };
                if read == 0 {
                    return;
                }
                raw.extend_from_slice(&buf[..read]);
            }
            return;
        }
    }
}

fn sse_tool_call(call_id: &str, tool_name: &str, raw_args: &str) -> ProviderScript {
    let args = serde_json::to_string(raw_args).expect("encode args");
    ProviderScript::Respond(format!(
        "data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":\"{call_id}\",\"function\":{{\"name\":\"{tool_name}\",\"arguments\":{args}}}}}]}}}}]}}\n\
         \n\
         data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":12,\"completion_tokens\":4}}}}\n\
         \n\
         data: [DONE]\n\n"
    ))
}

fn sse_text(text: &str, reasoning: Option<&str>) -> ProviderScript {
    let text = serde_json::to_string(text).expect("encode text");
    let reasoning_frame = reasoning
        .map(|thought| {
            let thought = serde_json::to_string(thought).expect("encode thought");
            format!("data: {{\"choices\":[{{\"delta\":{{\"reasoning_content\":{thought}}}}}]}}\n\n")
        })
        .unwrap_or_default();
    ProviderScript::Respond(format!(
        "{reasoning_frame}data: {{\"choices\":[{{\"delta\":{{\"content\":{text}}}}}]}}\n\
         \n\
         data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":20,\"completion_tokens\":5}}}}\n\
         \n\
         data: [DONE]\n\n"
    ))
}

// ---------------------------------------------------------------------------
// Daemon harness
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

fn spawn_serve(home: &std::path::Path, provider_url: &str) -> ServeDaemon {
    let bin = std::env::var_os("CARGO_BIN_EXE_anvil")
        .map(PathBuf::from)
        .or_else(|| option_env!("CARGO_BIN_EXE_anvil").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("target/debug/anvil"));
    let mut child = Command::new(bin)
        .args([
            "--no-wasm-sandbox",
            "--transient-setup",
            "--default-model",
            "ollama::conformance",
            "serve",
            "--port",
            "0",
        ])
        .env("HOME", home)
        .env("CODEX_HOME", home.join(".codex"))
        .env("BROKK_CONFIG_HOME", home.join("config"))
        .env("ANVIL_TEST_OLLAMA_BASE_URL", provider_url)
        // Recaps would issue an extra summarizer LLM call after
        // file-changing turns and desync the scripted provider.
        .env("ANVIL_TEST_DISABLE_TURN_RECAP", "1")
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

// ---------------------------------------------------------------------------
// Byte-accurate HTTP client
// ---------------------------------------------------------------------------

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Minimal blocking HTTP/1.1 client. All framing — header split, chunked
/// transfer decoding — happens over raw bytes; UTF-8 decoding is applied
/// only to the fully dechunked body, so multi-byte characters crossing
/// chunk boundaries are handled correctly.
fn raw_request(method: &str, base: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let host = base.strip_prefix("http://").expect("base url");
    let mut stream = TcpStream::connect(host).expect("connect");
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

    let header_end = find_subslice(&response, b"\r\n\r\n")
        .unwrap_or_else(|| panic!("malformed response: {}", String::from_utf8_lossy(&response)));
    let head = String::from_utf8_lossy(&response[..header_end]).into_owned();
    let body_bytes = &response[header_end + 4..];
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|token| token.parse().ok())
        .unwrap_or_else(|| panic!("malformed status line: {head}"));
    let body_bytes = if head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        decode_chunked(body_bytes)
    } else {
        body_bytes.to_vec()
    };
    (status, String::from_utf8_lossy(&body_bytes).into_owned())
}

/// Decode HTTP/1.1 chunked transfer encoding over bytes. Chunk sizes are
/// byte counts, so slicing must happen before any UTF-8 interpretation.
fn decode_chunked(mut rest: &[u8]) -> Vec<u8> {
    let mut decoded = Vec::new();
    while let Some(line_end) = find_subslice(rest, b"\r\n") {
        let size_line = String::from_utf8_lossy(&rest[..line_end]);
        let size = usize::from_str_radix(size_line.trim(), 16).unwrap_or(0);
        if size == 0 {
            break;
        }
        let chunk_start = line_end + 2;
        let chunk_end = (chunk_start + size).min(rest.len());
        decoded.extend_from_slice(&rest[chunk_start..chunk_end]);
        if chunk_end >= rest.len() {
            break;
        }
        rest = rest[chunk_end..]
            .strip_prefix(b"\r\n")
            .unwrap_or(&rest[chunk_end..]);
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
// Flow helpers
// ---------------------------------------------------------------------------

struct Harness {
    contract: Contract,
    base: String,
}

impl Harness {
    fn create_session(&self, cwd: &str, permission_mode: Option<&str>) -> String {
        let mut request = json!({ "cwd": cwd });
        if let Some(mode) = permission_mode {
            request["permission_mode"] = json!(mode);
        }
        let (status, body) = raw_request(
            "POST",
            &self.base,
            "/v1/sessions",
            Some(&request.to_string()),
        );
        assert_eq!(status, 201, "create session failed: {body}");
        let created: Value = serde_json::from_str(&body).expect("session JSON");
        self.contract
            .check_response("post", "/v1/sessions", 201, &created);
        created["id"].as_str().expect("session id").to_string()
    }

    fn start_run(&self, session_id: &str, prompt: &str) -> String {
        let (status, body) = raw_request(
            "POST",
            &self.base,
            &format!("/v1/sessions/{session_id}/runs"),
            Some(&json!({ "prompt": prompt }).to_string()),
        );
        assert_eq!(status, 202, "run not accepted: {body}");
        let run: Value = serde_json::from_str(&body).expect("run JSON");
        self.contract
            .check_response("post", "/v1/sessions/{session_id}/runs", 202, &run);
        run["id"].as_str().expect("run id").to_string()
    }

    fn poll_run_terminal(&self, run_id: &str) -> Value {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let run = get_json(
                &self.contract,
                &self.base,
                "/v1/runs/{run_id}",
                &format!("/v1/runs/{run_id}"),
            );
            if run["status"] != "running" {
                return run;
            }
            assert!(Instant::now() < deadline, "run stuck: {run}");
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// Fetch, contract-check, and structurally check a terminal run's full
    /// event stream; returns the set of event types observed.
    fn validate_events(&self, run_id: &str, expect_terminal: &str) -> BTreeSet<String> {
        let (status, events_body) = raw_request(
            "GET",
            &self.base,
            &format!("/v1/runs/{run_id}/events"),
            None,
        );
        assert_eq!(status, 200);
        let frames = parse_sse(&events_body);
        assert!(!frames.is_empty(), "no events for run {run_id}");
        let mut last_seq = 0;
        for frame in &frames {
            self.contract.check_event(&frame.data);
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
        assert_eq!(frames.last().expect("last frame").event, expect_terminal);
        frames.iter().map(|frame| frame.event.clone()).collect()
    }

    fn wait_for_permission(&self, run_id: &str) -> Value {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let listing = get_json(
                &self.contract,
                &self.base,
                "/v1/runs/{run_id}/permissions",
                &format!("/v1/runs/{run_id}/permissions"),
            );
            let pending = listing["permissions"].as_array().expect("permissions");
            if let Some(first) = pending.first() {
                return first.clone();
            }
            assert!(
                Instant::now() < deadline,
                "no permission request appeared for run {run_id}"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

// ---------------------------------------------------------------------------
// The suite
// ---------------------------------------------------------------------------

#[test]
fn daemon_conforms_to_checked_in_contract() {
    let mut script = Vec::new();
    script.extend([
        // Run A (acceptEdits session): plan, write-with-diff, execution
        // failure, malformed arguments, then a final reasoning + text turn.
        sse_tool_call(
            "call_plan",
            "update_plan",
            r#"{"explanation":"drïve the plan","plan":[{"step":"write hello","status":"in_progress"}]}"#,
        ),
        sse_tool_call(
            "call_write",
            "write_file",
            r#"{"file_path":"hello.txt","content":"héllo wörld"}"#,
        ),
        sse_tool_call("call_read", "read_file", r#"{"file_path":"missing.txt"}"#),
        sse_tool_call("call_bad", "write_file", "{definitely not json"),
        sse_text("dône ✓", Some("pondering the ünïverse")),
        // Run B (default-mode session): permission approval flow.
        sse_tool_call(
            "call_perm",
            "write_file",
            r#"{"file_path":"approved.txt","content":"ok"}"#,
        ),
        sse_text("approved and done", None),
        // Run C (readOnly session): deterministic preflight block.
        sse_tool_call(
            "call_blocked",
            "write_file",
            r#"{"file_path":"nope.txt","content":"blocked"}"#,
        ),
        sse_text("acknowledged the block", None),
        // Run D: hang so cancellation wins.
        ProviderScript::Hang,
        // Run E: scripted provider failure -> failed terminal.
        ProviderScript::Fail,
    ]);
    let provider_url = start_mock_provider(script);

    let contract = Contract::load();
    let home = tempfile::tempdir().expect("temp home");
    let workspace_a = tempfile::tempdir().expect("workspace a");
    let workspace_b = tempfile::tempdir().expect("workspace b");
    let workspace_c = tempfile::tempdir().expect("workspace c");
    let daemon = spawn_serve(home.path(), &provider_url);
    let harness = Harness {
        contract,
        base: daemon.base_url.clone(),
    };
    let contract = &harness.contract;
    let base = &harness.base;

    // Server + catalog endpoints.
    get_json(contract, base, "/health", "/health");
    get_json(contract, base, "/v1/models", "/v1/models");
    get_json(contract, base, "/v1/tools", "/v1/tools");

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
    let (status, body) = raw_request("GET", base, "/v1/permissions/perm-nope", None);
    assert_eq!(status, 404);
    contract.check_response(
        "get",
        "/v1/permissions/{permission_id}",
        404,
        &serde_json::from_str(&body).expect("error JSON"),
    );

    // --- Run A: full tool lifecycle on an acceptEdits session -------------
    let cwd_a = workspace_a.path().display().to_string();
    let session_a = harness.create_session(&cwd_a, Some("acceptEdits"));
    let run_a = harness.start_run(&session_a, "exercise the tool lifecycle");
    let terminal = harness.poll_run_terminal(&run_a);
    assert_eq!(terminal["status"], "completed", "run A: {terminal}");
    let written =
        std::fs::read_to_string(workspace_a.path().join("hello.txt")).expect("write_file executed");
    assert_eq!(written, "héllo wörld");
    let events_a = harness.validate_events(&run_a, "run.completed");
    for expected in [
        "plan.updated",
        "tool_call.started",
        "tool_call.in_progress",
        "tool_call.completed",
        "tool_call.failed",
        "message.delta",
        "thought.delta",
    ] {
        assert!(
            events_a.contains(expected),
            "run A missing {expected}: {events_a:?}"
        );
    }

    // --- Run B: permission suspend/approve on a default-mode session ------
    let cwd_b = workspace_b.path().display().to_string();
    let session_b = harness.create_session(&cwd_b, Some("default"));
    let run_b = harness.start_run(&session_b, "ask before writing");
    let permission = harness.wait_for_permission(&run_b);
    let permission_id = permission["id"].as_str().expect("permission id");
    get_json(
        contract,
        base,
        "/v1/permissions/{permission_id}",
        &format!("/v1/permissions/{permission_id}"),
    );
    let (status, body) = raw_request(
        "POST",
        base,
        &format!("/v1/permissions/{permission_id}/respond"),
        Some(&json!({ "option_id": "allow" }).to_string()),
    );
    assert_eq!(status, 200, "respond failed: {body}");
    contract.check_response(
        "post",
        "/v1/permissions/{permission_id}/respond",
        200,
        &serde_json::from_str(&body).expect("respond JSON"),
    );
    let terminal = harness.poll_run_terminal(&run_b);
    assert_eq!(terminal["status"], "completed", "run B: {terminal}");
    let events_b = harness.validate_events(&run_b, "run.completed");
    for expected in ["permission.requested", "permission.resolved"] {
        assert!(
            events_b.contains(expected),
            "run B missing {expected}: {events_b:?}"
        );
    }

    // --- Run C: preflight block on a readOnly session ----------------------
    let cwd_c = workspace_c.path().display().to_string();
    let session_c = harness.create_session(&cwd_c, Some("readOnly"));
    let run_c = harness.start_run(&session_c, "try to write anyway");
    let terminal = harness.poll_run_terminal(&run_c);
    assert_eq!(terminal["status"], "completed", "run C: {terminal}");
    let events_c = harness.validate_events(&run_c, "run.completed");
    assert!(
        events_c.contains("tool_call.blocked"),
        "run C: {events_c:?}"
    );

    // --- Run D: cancellation while the provider hangs ----------------------
    let run_d = harness.start_run(&session_a, "hang until cancelled");
    std::thread::sleep(Duration::from_millis(300));
    let (status, body) = raw_request("POST", base, &format!("/v1/runs/{run_d}/cancel"), None);
    assert_eq!(status, 200);
    contract.check_response(
        "post",
        "/v1/runs/{run_id}/cancel",
        200,
        &serde_json::from_str(&body).expect("cancel JSON"),
    );
    let terminal = harness.poll_run_terminal(&run_d);
    assert_eq!(terminal["status"], "cancelled", "run D: {terminal}");
    harness.validate_events(&run_d, "run.cancelled");

    // --- Run E: scripted provider failure -> failed terminal ---------------
    // (Also exercises PATCH with a catalog-independent selector so the
    // response schema is validated regardless of what models the host's
    // real environment discovers.)
    let (status, body) = raw_request(
        "PATCH",
        base,
        &format!("/v1/sessions/{session_c}"),
        Some(&json!({ "behavior_mode": "LUTZ" }).to_string()),
    );
    assert_eq!(status, 200, "patch failed: {body}");
    contract.check_response(
        "patch",
        "/v1/sessions/{session_id}",
        200,
        &serde_json::from_str(&body).expect("patch JSON"),
    );
    let run_e = harness.start_run(&session_c, "fail please");
    let terminal = harness.poll_run_terminal(&run_e);
    assert_eq!(terminal["status"], "failed", "run E: {terminal}");
    harness.validate_events(&run_e, "run.failed");

    // --- Remaining lifecycle endpoints -------------------------------------
    let session = get_json(
        contract,
        base,
        "/v1/sessions/{session_id}",
        &format!("/v1/sessions/{session_a}?include_history=true"),
    );
    assert!(session["history"].is_array());
    get_json(contract, base, "/v1/sessions", "/v1/sessions");
    get_json(
        contract,
        base,
        "/v1/sessions/{session_id}/runs",
        &format!("/v1/sessions/{session_a}/runs"),
    );

    let (status, body) = raw_request(
        "POST",
        base,
        &format!("/v1/sessions/{session_a}/load"),
        Some(&json!({ "cwd": cwd_a }).to_string()),
    );
    assert_eq!(status, 200, "load failed: {body}");
    let loaded: Value = serde_json::from_str(&body).expect("load JSON");
    contract.check_response("post", "/v1/sessions/{session_id}/load", 200, &loaded);
    assert!(loaded["history"].is_array());
    let (status, body) = raw_request(
        "POST",
        base,
        &format!("/v1/sessions/{session_a}/resume"),
        Some(&json!({ "cwd": cwd_a }).to_string()),
    );
    assert_eq!(status, 200, "resume failed: {body}");
    contract.check_response(
        "post",
        "/v1/sessions/{session_id}/resume",
        200,
        &serde_json::from_str(&body).expect("resume JSON"),
    );

    let (status, body) = raw_request("DELETE", base, &format!("/v1/sessions/{session_a}"), None);
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
