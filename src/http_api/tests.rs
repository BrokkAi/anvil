//! In-process tests for the HTTP API: a real axum server bound to a
//! localhost ephemeral port, exercised with `reqwest`. The daemon-level
//! smoke test (spawning the `anvil serve` binary) lives in
//! `tests/http_smoke.rs`.

use std::net::SocketAddr;
use std::sync::Arc;

use serde_json::Value;

use super::{ApiState, router};
use crate::llm_client::{ModelMetadata, ReasoningLevelPreset};
use crate::multi_backend::MultiBackend;
use crate::session::SessionStore;

async fn start_server(sessions: SessionStore) -> SocketAddr {
    let state = ApiState {
        sessions,
        llm: Arc::new(MultiBackend::new(Vec::new())),
        refresh_lock: Arc::new(tokio::sync::Mutex::new(())),
        runs: Arc::new(super::runs::RunManager::default()),
        max_turns: usize::MAX,
        default_idle_timeout_secs: 30,
        default_stall_timeout_secs: 30,
    };
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("bind ephemeral test port");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, router(state)).await.expect("serve");
    });
    addr
}

fn test_model_catalog() -> Vec<ModelMetadata> {
    vec![
        ModelMetadata {
            id: "test::alpha".to_string(),
            default_reasoning_level: Some("medium".to_string()),
            supported_reasoning_levels: vec![
                ReasoningLevelPreset {
                    effort: "low".to_string(),
                    description: "Low".to_string(),
                },
                ReasoningLevelPreset {
                    effort: "medium".to_string(),
                    description: "Medium".to_string(),
                },
            ],
            service_tiers: Vec::new(),
            supports_images: Some(false),
            context_length: Some(128_000),
            pricing: None,
        },
        ModelMetadata::id_only("test::beta"),
    ]
}

async fn seeded_store() -> SessionStore {
    let store = SessionStore::new("test::alpha".to_string());
    store.set_available_models(test_model_catalog()).await;
    store
}

async fn get_json(addr: SocketAddr, path: &str) -> (reqwest::StatusCode, Value) {
    let response = reqwest::get(format!("http://{addr}{path}"))
        .await
        .expect("GET request");
    let status = response.status();
    let body = response.json::<Value>().await.expect("JSON body");
    (status, body)
}

#[tokio::test]
async fn health_reports_ok_and_discovery_state() {
    let addr = start_server(seeded_store().await).await;
    let (status, body) = get_json(addr, "/health").await;
    assert_eq!(status, reqwest::StatusCode::OK);
    assert_eq!(body["status"], "ok");
    assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(body["models_discovered"], 2);
}

#[tokio::test]
async fn models_lists_catalog_and_default() {
    let addr = start_server(seeded_store().await).await;
    let (status, body) = get_json(addr, "/v1/models").await;
    assert_eq!(status, reqwest::StatusCode::OK);
    assert_eq!(body["default_model"], "test::alpha");
    let models = body["models"].as_array().expect("models array");
    assert_eq!(models.len(), 2);
    assert_eq!(models[0]["id"], "test::alpha");
    assert_eq!(models[0]["default_reasoning_level"], "medium");
    assert_eq!(models[0]["supported_reasoning_levels"][0]["effort"], "low",);
    assert_eq!(models[0]["context_length"], 128_000);
    assert_eq!(models[1]["id"], "test::beta");
}

#[tokio::test]
async fn tools_lists_builtin_and_mcp_catalog() {
    let addr = start_server(seeded_store().await).await;
    let (status, body) = get_json(addr, "/v1/tools").await;
    assert_eq!(status, reqwest::StatusCode::OK);
    let tools = body["tools"].as_array().expect("tools array");
    let read_file = tools
        .iter()
        .find(|t| t["name"] == "read_file")
        .expect("read_file in catalog");
    assert_eq!(read_file["source"], "builtin");
    assert_eq!(read_file["concurrency_safe"], true);
    assert!(
        tools.iter().any(|t| t["source"] == "mcp"),
        "catalog should include MCP-loaded tools"
    );
}

#[tokio::test]
async fn unknown_route_uses_error_envelope_with_request_id() {
    let addr = start_server(seeded_store().await).await;
    let response = reqwest::get(format!("http://{addr}/v1/nope"))
        .await
        .expect("GET request");
    assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);
    let header_id = response
        .headers()
        .get("x-request-id")
        .expect("x-request-id header")
        .to_str()
        .expect("header utf8")
        .to_string();
    let body = response.json::<Value>().await.expect("JSON body");
    assert_eq!(body["error"]["code"], "not_found");
    assert_eq!(body["request_id"], Value::String(header_id));
}

#[tokio::test]
async fn session_lifecycle_create_inspect_configure_delete() {
    let workspace = tempfile::tempdir().expect("workspace");
    let cwd = workspace.path().display().to_string();
    let addr = start_server(seeded_store().await).await;
    let client = reqwest::Client::new();

    // Create with an explicit permission mode and reasoning effort.
    let response = client
        .post(format!("http://{addr}/v1/sessions"))
        .json(&serde_json::json!({
            "cwd": cwd,
            "permission_mode": "readOnly",
            "reasoning_effort": "low",
        }))
        .send()
        .await
        .expect("create session");
    assert_eq!(response.status(), reqwest::StatusCode::CREATED);
    let created = response.json::<Value>().await.expect("JSON body");
    let session_id = created["id"].as_str().expect("session id").to_string();
    assert_eq!(created["cwd"], cwd.as_str());
    assert_eq!(created["model"], "test::alpha");
    assert_eq!(created["permission_mode"], "readOnly");
    assert_eq!(created["reasoning_effort"], "low");
    assert_eq!(created["behavior_mode"], "LUTZ");
    assert_eq!(created["history_turns"], 0);

    // Inspect.
    let (status, fetched) = get_json(addr, &format!("/v1/sessions/{session_id}")).await;
    assert_eq!(status, reqwest::StatusCode::OK);
    assert_eq!(fetched["id"], session_id.as_str());
    assert!(fetched.get("history").is_none());

    // The session zip is persisted immediately, like an ACP session/new.
    let zip_path = workspace
        .path()
        .join(".brokk")
        .join("sessions")
        .join(format!("{session_id}.zip"));
    assert!(zip_path.exists(), "created session should persist a zip");

    // The listing endpoint responds; fresh sessions are omitted until they
    // have a title (first prompt), matching ACP session/list semantics.
    let (status, listing) = get_json(addr, "/v1/sessions").await;
    assert_eq!(status, reqwest::StatusCode::OK);
    assert!(listing["sessions"].is_array());

    // Reconfigure: switch model; the low reasoning pick isn't supported by
    // the schemaless beta model, so the store clears it and we get a warning.
    let response = client
        .patch(format!("http://{addr}/v1/sessions/{session_id}"))
        .json(&serde_json::json!({ "model": "test::beta" }))
        .send()
        .await
        .expect("patch session");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let patched = response.json::<Value>().await.expect("JSON body");
    assert_eq!(patched["session"]["model"], "test::beta");
    assert_eq!(patched["session"]["reasoning_effort"], Value::Null);
    assert!(
        !patched["warnings"].as_array().expect("warnings").is_empty(),
        "model switch that drops the reasoning pick should warn"
    );

    // Delete is idempotent and reported.
    let response = client
        .delete(format!("http://{addr}/v1/sessions/{session_id}"))
        .send()
        .await
        .expect("delete session");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    assert_eq!(
        response.json::<Value>().await.expect("JSON body")["deleted"],
        true
    );
    let response = client
        .delete(format!("http://{addr}/v1/sessions/{session_id}"))
        .send()
        .await
        .expect("second delete");
    assert_eq!(
        response.json::<Value>().await.expect("JSON body")["deleted"],
        false
    );
    let (status, _) = get_json(addr, &format!("/v1/sessions/{session_id}")).await;
    assert_eq!(status, reqwest::StatusCode::NOT_FOUND);
    assert!(!zip_path.exists(), "delete should remove the session zip");
}

#[tokio::test]
async fn create_rejects_relative_cwd() {
    let addr = start_server(seeded_store().await).await;
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions"))
        .json(&serde_json::json!({ "cwd": "relative/path" }))
        .send()
        .await
        .expect("create session");
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let body = response.json::<Value>().await.expect("JSON body");
    assert_eq!(body["error"]["code"], "invalid_argument");
    assert_eq!(body["error"]["details"]["field"], "cwd");
}

#[tokio::test]
async fn create_rejects_missing_additional_directory() {
    let workspace = tempfile::tempdir().expect("workspace");
    let addr = start_server(seeded_store().await).await;
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions"))
        .json(&serde_json::json!({
            "cwd": workspace.path().display().to_string(),
            "additional_directories": ["/definitely/not/a/real/dir"],
        }))
        .send()
        .await
        .expect("create session");
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let body = response.json::<Value>().await.expect("JSON body");
    assert_eq!(body["error"]["code"], "invalid_argument");
    assert_eq!(body["error"]["details"]["field"], "additional_directories");
    assert_eq!(body["error"]["details"]["index"], 0);
}

#[tokio::test]
async fn create_with_unknown_model_leaves_no_session_behind() {
    let workspace = tempfile::tempdir().expect("workspace");
    let addr = start_server(seeded_store().await).await;
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions"))
        .json(&serde_json::json!({
            "cwd": workspace.path().display().to_string(),
            "model": "test::does-not-exist",
        }))
        .send()
        .await
        .expect("create session");
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let body = response.json::<Value>().await.expect("JSON body");
    assert_eq!(body["error"]["code"], "invalid_argument");
    assert_eq!(body["error"]["details"]["field"], "model");
    let supported = body["error"]["details"]["supported"]
        .as_array()
        .expect("supported list");
    assert!(supported.iter().any(|m| m == "test::alpha"));

    // The rolled-back session must leave no zip behind in the workspace.
    let sessions_dir = workspace.path().join(".brokk").join("sessions");
    let leftover_zips = std::fs::read_dir(&sessions_dir)
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "zip"))
                .count()
        })
        .unwrap_or(0);
    assert_eq!(leftover_zips, 0, "failed create must roll the session back");
}

#[tokio::test]
async fn patch_rejects_invalid_permission_mode() {
    let workspace = tempfile::tempdir().expect("workspace");
    let addr = start_server(seeded_store().await).await;
    let client = reqwest::Client::new();
    let created = client
        .post(format!("http://{addr}/v1/sessions"))
        .json(&serde_json::json!({ "cwd": workspace.path().display().to_string() }))
        .send()
        .await
        .expect("create session")
        .json::<Value>()
        .await
        .expect("JSON body");
    let session_id = created["id"].as_str().expect("session id");

    let response = client
        .patch(format!("http://{addr}/v1/sessions/{session_id}"))
        .json(&serde_json::json!({ "permission_mode": "yolo" }))
        .send()
        .await
        .expect("patch session");
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let body = response.json::<Value>().await.expect("JSON body");
    assert_eq!(body["error"]["details"]["field"], "permission_mode");

    let response = client
        .patch(format!("http://{addr}/v1/sessions/{session_id}"))
        .json(&serde_json::json!({}))
        .send()
        .await
        .expect("empty patch");
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn load_and_resume_validate_cwd_and_return_history() {
    let workspace = tempfile::tempdir().expect("workspace");
    let other_workspace = tempfile::tempdir().expect("other workspace");
    let cwd = workspace.path().display().to_string();
    let addr = start_server(seeded_store().await).await;
    let client = reqwest::Client::new();

    let created = client
        .post(format!("http://{addr}/v1/sessions"))
        .json(&serde_json::json!({ "cwd": cwd }))
        .send()
        .await
        .expect("create session")
        .json::<Value>()
        .await
        .expect("JSON body");
    let session_id = created["id"].as_str().expect("session id");

    // Wrong cwd is a conflict, matching the ACP lifecycle rules.
    let response = client
        .post(format!("http://{addr}/v1/sessions/{session_id}/load"))
        .json(&serde_json::json!({ "cwd": other_workspace.path().display().to_string() }))
        .send()
        .await
        .expect("load with wrong cwd");
    assert_eq!(response.status(), reqwest::StatusCode::CONFLICT);
    let body = response.json::<Value>().await.expect("JSON body");
    assert_eq!(body["error"]["code"], "conflict");
    assert_eq!(body["error"]["details"]["session_cwd"], cwd.as_str());

    // Correct cwd loads and embeds (empty) history.
    let response = client
        .post(format!("http://{addr}/v1/sessions/{session_id}/load"))
        .json(&serde_json::json!({ "cwd": cwd }))
        .send()
        .await
        .expect("load session");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body = response.json::<Value>().await.expect("JSON body");
    assert_eq!(body["id"], session_id);
    assert_eq!(body["history"], serde_json::json!([]));

    // Resume succeeds without history.
    let response = client
        .post(format!("http://{addr}/v1/sessions/{session_id}/resume"))
        .json(&serde_json::json!({ "cwd": cwd }))
        .send()
        .await
        .expect("resume session");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body = response.json::<Value>().await.expect("JSON body");
    assert!(body.get("history").is_none());

    // Unknown session ids are 404s on both endpoints.
    let response = client
        .post(format!("http://{addr}/v1/sessions/no-such-session/load"))
        .json(&serde_json::json!({ "cwd": cwd }))
        .send()
        .await
        .expect("load unknown session");
    assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn malformed_json_body_uses_error_envelope() {
    let addr = start_server(seeded_store().await).await;
    let response = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions"))
        .header("content-type", "application/json")
        .body("{not json")
        .send()
        .await
        .expect("POST malformed body");
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    let body = response.json::<Value>().await.expect("JSON body");
    assert_eq!(body["error"]["code"], "invalid_argument");
}

// ---------------------------------------------------------------------------
// Prompt runs (#318)
// ---------------------------------------------------------------------------

use futures::FutureExt;
use futures::future::BoxFuture;

use crate::llm_client::{LlmBackend, LlmResponse, StreamChatRequest, TokenUsage};
use crate::multi_backend::BackendRegistration;

/// Scripted backend for the `test::` source: streams a fixed reply, or
/// hangs until cancellation to exercise in-flight semantics.
enum MockBehavior {
    Echo,
    HangUntilCancelled,
}

struct MockBackend {
    behavior: MockBehavior,
}

impl LlmBackend for MockBackend {
    fn list_models(&self) -> BoxFuture<'_, anyhow::Result<Vec<String>>> {
        async { Ok(vec!["alpha".to_string()]) }.boxed()
    }

    fn stream_chat(
        &self,
        request: StreamChatRequest,
    ) -> BoxFuture<'_, anyhow::Result<LlmResponse>> {
        match self.behavior {
            MockBehavior::Echo => {
                let mut on_token = request.on_token;
                async move {
                    on_token("Hello from mock");
                    Ok(LlmResponse::Text {
                        text: "Hello from mock".to_string(),
                        reasoning_content: None,
                        usage: TokenUsage {
                            input_tokens: 3,
                            output_tokens: 2,
                            thought_tokens: 0,
                            cached_read_tokens: 0,
                            cached_write_tokens: 0,
                        },
                    })
                }
                .boxed()
            }
            MockBehavior::HangUntilCancelled => async move {
                request.cancel.cancelled().await;
                anyhow::bail!("stream cancelled")
            }
            .boxed(),
        }
    }
}

async fn start_run_server(behavior: MockBehavior) -> (SocketAddr, SessionStore) {
    let sessions = seeded_store().await;
    let llm = Arc::new(MultiBackend::new(vec![BackendRegistration::new(
        "test",
        "Test",
        Some(Arc::new(MockBackend { behavior })),
    )]));
    let state = ApiState {
        sessions: sessions.clone(),
        llm,
        refresh_lock: Arc::new(tokio::sync::Mutex::new(())),
        runs: Arc::new(super::runs::RunManager::default()),
        max_turns: usize::MAX,
        default_idle_timeout_secs: 30,
        default_stall_timeout_secs: 30,
    };
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("bind ephemeral test port");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, router(state)).await.expect("serve");
    });
    (addr, sessions)
}

async fn create_test_session(addr: SocketAddr, cwd: &str) -> String {
    let created = reqwest::Client::new()
        .post(format!("http://{addr}/v1/sessions"))
        .json(&serde_json::json!({ "cwd": cwd }))
        .send()
        .await
        .expect("create session")
        .json::<Value>()
        .await
        .expect("JSON body");
    created["id"].as_str().expect("session id").to_string()
}

async fn poll_run_until_terminal(addr: SocketAddr, run_id: &str) -> Value {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let (status, run) = get_json(addr, &format!("/v1/runs/{run_id}")).await;
        assert_eq!(status, reqwest::StatusCode::OK);
        if run["status"] != "running" {
            return run;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "run did not reach a terminal state in time: {run}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
}

#[tokio::test]
async fn run_lifecycle_streams_events_and_persists_turn() {
    let workspace = tempfile::tempdir().expect("workspace");
    let cwd = workspace.path().display().to_string();
    let (addr, _) = start_run_server(MockBehavior::Echo).await;
    let client = reqwest::Client::new();
    let session_id = create_test_session(addr, &cwd).await;

    let response = client
        .post(format!("http://{addr}/v1/sessions/{session_id}/runs"))
        .json(&serde_json::json!({ "prompt": "say hello" }))
        .send()
        .await
        .expect("create run");
    assert_eq!(response.status(), reqwest::StatusCode::ACCEPTED);
    let created = response.json::<Value>().await.expect("JSON body");
    let run_id = created["id"].as_str().expect("run id").to_string();
    assert_eq!(created["session_id"], session_id.as_str());

    let run = poll_run_until_terminal(addr, &run_id).await;
    assert_eq!(run["status"], "completed", "run failed: {run}");
    assert_eq!(run["stop_reason"], "end_turn");
    assert_eq!(run["result_text"], "Hello from mock");
    assert_eq!(run["usage"]["input_tokens"], 3);
    assert_eq!(run["error"], Value::Null);

    // The event stream terminates after the terminal event, so the full
    // body is readable in one shot.
    let events_body = client
        .get(format!("http://{addr}/v1/runs/{run_id}/events"))
        .send()
        .await
        .expect("events request")
        .text()
        .await
        .expect("events body");
    assert!(events_body.contains("event: run.created"));
    assert!(events_body.contains("event: message.delta"));
    assert!(events_body.contains("Hello from mock"));
    assert!(events_body.contains("event: run.completed"));

    // Reconnecting from the last seen sequence id replays nothing new.
    let last_seq = run["last_seq"].as_u64().expect("last seq");
    let replay = client
        .get(format!("http://{addr}/v1/runs/{run_id}/events"))
        .header("last-event-id", last_seq.to_string())
        .send()
        .await
        .expect("replay request")
        .text()
        .await
        .expect("replay body");
    assert!(
        !replay.contains("event: run.completed"),
        "full replay after Last-Event-ID should skip already-seen events: {replay}"
    );

    // Resuming mid-stream replays only events after the cursor.
    let partial = client
        .get(format!("http://{addr}/v1/runs/{run_id}/events"))
        .header("last-event-id", (last_seq - 1).to_string())
        .send()
        .await
        .expect("partial replay request")
        .text()
        .await
        .expect("partial replay body");
    assert!(partial.contains("event: run.completed"));
    assert!(!partial.contains("event: run.created"));

    // The turn persisted through the same SessionStore path ACP uses.
    let (status, session) = get_json(
        addr,
        &format!("/v1/sessions/{session_id}?include_history=true"),
    )
    .await;
    assert_eq!(status, reqwest::StatusCode::OK);
    assert_eq!(session["history_turns"], 1);
    assert_eq!(session["history"][0]["user_prompt"], "say hello");
    assert_eq!(session["history"][0]["agent_response"], "Hello from mock");
    assert_eq!(session["usage"]["input_tokens"], 3);

    // The prompt slot was released: a second run is accepted.
    let response = client
        .post(format!("http://{addr}/v1/sessions/{session_id}/runs"))
        .json(&serde_json::json!({ "prompt": "again" }))
        .send()
        .await
        .expect("second run");
    assert_eq!(response.status(), reqwest::StatusCode::ACCEPTED);
    let second = response.json::<Value>().await.expect("JSON body");
    let second_run = poll_run_until_terminal(addr, second["id"].as_str().unwrap()).await;
    assert_eq!(second_run["status"], "completed");

    // Both runs are listed for the session, newest first.
    let (status, listing) = get_json(addr, &format!("/v1/sessions/{session_id}/runs")).await;
    assert_eq!(status, reqwest::StatusCode::OK);
    assert_eq!(listing["runs"].as_array().expect("runs array").len(), 2);
}

#[tokio::test]
async fn duplicate_run_is_rejected_and_cancel_is_idempotent() {
    let workspace = tempfile::tempdir().expect("workspace");
    let cwd = workspace.path().display().to_string();
    let (addr, _) = start_run_server(MockBehavior::HangUntilCancelled).await;
    let client = reqwest::Client::new();
    let session_id = create_test_session(addr, &cwd).await;

    let created = client
        .post(format!("http://{addr}/v1/sessions/{session_id}/runs"))
        .json(&serde_json::json!({ "prompt": "hang" }))
        .send()
        .await
        .expect("create run")
        .json::<Value>()
        .await
        .expect("JSON body");
    let run_id = created["id"].as_str().expect("run id").to_string();

    // One-in-flight-prompt-per-session is preserved across transports.
    let response = client
        .post(format!("http://{addr}/v1/sessions/{session_id}/runs"))
        .json(&serde_json::json!({ "prompt": "duplicate" }))
        .send()
        .await
        .expect("duplicate run");
    assert_eq!(response.status(), reqwest::StatusCode::CONFLICT);
    let body = response.json::<Value>().await.expect("JSON body");
    assert_eq!(body["error"]["code"], "conflict");

    // Cancel, and cancel again once terminal: both succeed.
    let response = client
        .post(format!("http://{addr}/v1/runs/{run_id}/cancel"))
        .send()
        .await
        .expect("cancel run");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let run = poll_run_until_terminal(addr, &run_id).await;
    assert_eq!(run["status"], "cancelled", "expected cancelled: {run}");
    assert_eq!(run["stop_reason"], "cancelled");
    let response = client
        .post(format!("http://{addr}/v1/runs/{run_id}/cancel"))
        .send()
        .await
        .expect("second cancel");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    assert_eq!(
        response.json::<Value>().await.expect("JSON body")["status"],
        "cancelled"
    );

    // The session accepts a fresh run afterwards (slot released).
    let response = client
        .post(format!("http://{addr}/v1/sessions/{session_id}/runs"))
        .json(&serde_json::json!({ "prompt": "after cancel" }))
        .send()
        .await
        .expect("run after cancel");
    assert_eq!(response.status(), reqwest::StatusCode::ACCEPTED);
}

#[tokio::test]
async fn run_validation_and_unknown_resources() {
    let workspace = tempfile::tempdir().expect("workspace");
    let cwd = workspace.path().display().to_string();
    let (addr, _) = start_run_server(MockBehavior::Echo).await;
    let client = reqwest::Client::new();

    let response = client
        .post(format!("http://{addr}/v1/sessions/no-such-session/runs"))
        .json(&serde_json::json!({ "prompt": "hello" }))
        .send()
        .await
        .expect("run on unknown session");
    assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);

    let session_id = create_test_session(addr, &cwd).await;
    let response = client
        .post(format!("http://{addr}/v1/sessions/{session_id}/runs"))
        .json(&serde_json::json!({ "prompt": "   " }))
        .send()
        .await
        .expect("empty prompt");
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);

    let (status, _) = get_json(addr, "/v1/runs/run_nope").await;
    assert_eq!(status, reqwest::StatusCode::NOT_FOUND);
    let response = client
        .post(format!("http://{addr}/v1/runs/run_nope/cancel"))
        .send()
        .await
        .expect("cancel unknown run");
    assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);
}
