//! Asynchronous prompt runs over HTTP (#318).
//!
//! A run is one prompt turn executed against a session, exposed as a
//! resource: `POST /v1/sessions/{id}/runs` returns a run id promptly and
//! executes the turn in a background task through the shared
//! `turn_runner::run_prompt_turn` pipeline (the same one ACP prompts use).
//! Progress is buffered per run and streamed as Server-Sent Events with
//! monotonically increasing sequence ids, so clients can poll
//! (`GET /v1/runs/{id}`), stream (`GET /v1/runs/{id}/events`), reconnect
//! with `Last-Event-ID`, and cancel (`POST /v1/runs/{id}/cancel`)
//! independently. Dropping an event stream never cancels the run, and the
//! terminal result is retained even when no consumer was connected.
//!
//! Interactive permission prompts are not yet answerable over HTTP (#319):
//! a `permission.requested` event is recorded and the request is
//! auto-rejected with a notice, so runs never hang waiting on an
//! unanswerable prompt. Sessions in `auto`, `acceptEdits`, or `readOnly`
//! permission mode never reach that path.

use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use futures::Stream;
use futures::future::BoxFuture;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::broadcast;

use crate::runtime::{
    EventSink, PermissionBroker, PermissionDecision, PermissionPrompt, RuntimeEvent, ToolCallPhase,
};
use crate::session::{PromptStartError, SessionSnapshot};
use crate::structured_output::StructuredOutputRequest;
use crate::tool_loop::{LoopStop, TextSink};

use super::{ApiError, ApiJson, ApiState, fallback_cwd, unknown_session_error};

/// Bounded per-run replay window. Events older than this are dropped from
/// the buffer (the SSE stream emits an `events.gap` notice when a reconnect
/// falls behind the window). 8k events comfortably covers typical turns.
const MAX_BUFFERED_EVENTS_PER_RUN: usize = 8192;
/// Completed/failed/cancelled runs retained for polling, oldest evicted
/// first. Active runs are never evicted.
const MAX_RETAINED_TERMINAL_RUNS: usize = 256;
const EVENT_BROADCAST_CAPACITY: usize = 1024;

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Run registry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RunStatus {
    Running,
    Completed,
    Cancelled,
    Failed,
}

impl RunStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
        }
    }

    fn is_terminal(self) -> bool {
        self != Self::Running
    }
}

#[derive(Debug, Default)]
struct RunState {
    stop_reason: Option<String>,
    error: Option<String>,
    result_text: Option<String>,
    structured_output: Option<Value>,
    usage: Option<Value>,
    finished_at_ms: Option<u64>,
}

#[derive(Debug, Clone)]
pub(super) struct RunEventRecord {
    seq: u64,
    event_type: &'static str,
    /// Fully rendered SSE `data` payload (JSON object as a string).
    data: String,
}

pub(super) struct RunHandle {
    pub(super) id: String,
    pub(super) session_id: String,
    created_at_ms: u64,
    status: RwLock<RunStatus>,
    state: RwLock<RunState>,
    events: Mutex<VecDeque<RunEventRecord>>,
    next_seq: AtomicU64,
    broadcast: broadcast::Sender<RunEventRecord>,
    cancel: tokio_util::sync::CancellationToken,
}

impl RunHandle {
    fn new(session_id: String, cancel: tokio_util::sync::CancellationToken) -> Self {
        let (broadcast, _) = broadcast::channel(EVENT_BROADCAST_CAPACITY);
        Self {
            id: format!("run_{}", uuid::Uuid::new_v4().simple()),
            session_id,
            created_at_ms: now_ms(),
            status: RwLock::new(RunStatus::Running),
            state: RwLock::new(RunState::default()),
            events: Mutex::new(VecDeque::new()),
            next_seq: AtomicU64::new(0),
            broadcast,
            cancel: cancel.clone(),
        }
    }

    pub(super) fn status(&self) -> RunStatus {
        *self.status.read().expect("run status lock")
    }

    fn last_seq(&self) -> u64 {
        self.next_seq.load(Ordering::SeqCst)
    }

    /// Record one event: assign the next sequence id, merge the envelope
    /// fields, buffer it for replay, and fan it out to live subscribers.
    pub(super) fn record(&self, event_type: &'static str, mut payload: Value) {
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst) + 1;
        let object = payload
            .as_object_mut()
            .expect("run event payloads are JSON objects");
        object.insert("type".into(), json!(event_type));
        object.insert("run_id".into(), json!(self.id));
        object.insert("session_id".into(), json!(self.session_id));
        object.insert("seq".into(), json!(seq));
        object.insert("ts_ms".into(), json!(now_ms()));
        let record = RunEventRecord {
            seq,
            event_type,
            data: payload.to_string(),
        };
        {
            let mut events = self.events.lock().expect("run events lock");
            if events.len() >= MAX_BUFFERED_EVENTS_PER_RUN {
                events.pop_front();
            }
            events.push_back(record.clone());
        }
        let _ = self.broadcast.send(record);
    }

    fn events_after(&self, seq: u64) -> VecDeque<RunEventRecord> {
        self.events
            .lock()
            .expect("run events lock")
            .iter()
            .filter(|record| record.seq > seq)
            .cloned()
            .collect()
    }

    /// Earliest sequence id still in the replay buffer, if any.
    fn earliest_buffered_seq(&self) -> Option<u64> {
        self.events
            .lock()
            .expect("run events lock")
            .front()
            .map(|record| record.seq)
    }

    fn finish(&self, status: RunStatus, state: RunState, terminal_payload: Value) {
        {
            let mut current = self.status.write().expect("run status lock");
            if current.is_terminal() {
                return;
            }
            *current = status;
        }
        let event_type = match status {
            RunStatus::Completed => "run.completed",
            RunStatus::Cancelled => "run.cancelled",
            RunStatus::Failed => "run.failed",
            RunStatus::Running => unreachable!("finish() requires a terminal status"),
        };
        *self.state.write().expect("run state lock") = state;
        self.record(event_type, terminal_payload);
    }

    pub(super) fn resource(&self) -> Value {
        let status = self.status();
        let state = self.state.read().expect("run state lock");
        json!({
            "id": self.id,
            "session_id": self.session_id,
            "status": status.as_str(),
            "stop_reason": state.stop_reason,
            "error": state.error,
            "result_text": state.result_text,
            "structured_output": state.structured_output,
            "usage": state.usage,
            "created_at_ms": self.created_at_ms,
            "finished_at_ms": state.finished_at_ms,
            "last_seq": self.last_seq(),
        })
    }
}

#[derive(Default)]
pub(super) struct RunManager {
    runs: RwLock<HashMap<String, Arc<RunHandle>>>,
    /// Terminal run ids in completion order, for retention eviction.
    terminal_order: Mutex<VecDeque<String>>,
}

impl RunManager {
    fn insert(&self, run: Arc<RunHandle>) {
        self.runs
            .write()
            .expect("runs lock")
            .insert(run.id.clone(), run);
    }

    pub(super) fn get(&self, run_id: &str) -> Option<Arc<RunHandle>> {
        self.runs.read().expect("runs lock").get(run_id).cloned()
    }

    fn list_for_session(&self, session_id: &str) -> Vec<Arc<RunHandle>> {
        let mut runs: Vec<Arc<RunHandle>> = self
            .runs
            .read()
            .expect("runs lock")
            .values()
            .filter(|run| run.session_id == session_id)
            .cloned()
            .collect();
        runs.sort_by_key(|run| std::cmp::Reverse(run.created_at_ms));
        runs
    }

    /// Retention: cap the number of terminal runs kept for polling.
    fn note_terminal(&self, run_id: &str) {
        let evict = {
            let mut order = self.terminal_order.lock().expect("terminal order lock");
            order.push_back(run_id.to_string());
            if order.len() > MAX_RETAINED_TERMINAL_RUNS {
                order.pop_front()
            } else {
                None
            }
        };
        if let Some(evicted) = evict {
            self.runs.write().expect("runs lock").remove(&evicted);
        }
    }
}

// ---------------------------------------------------------------------------
// Sinks: runtime events, streaming text, permissions
// ---------------------------------------------------------------------------

struct RunEventSink {
    run: Arc<RunHandle>,
}

impl EventSink for RunEventSink {
    fn emit(&self, _session_id: &str, event: RuntimeEvent) {
        match event {
            RuntimeEvent::Plan(plan) => {
                let plan = serde_json::to_value(&plan).unwrap_or(Value::Null);
                self.run.record("plan.updated", json!({ "plan": plan }));
            }
            RuntimeEvent::ToolCall {
                call_id,
                tool_name,
                phase,
            } => {
                let base = |extra: Value| {
                    let mut payload = json!({
                        "call_id": call_id,
                        "tool_name": tool_name,
                    });
                    if let (Some(target), Some(source)) =
                        (payload.as_object_mut(), extra.as_object())
                    {
                        for (key, value) in source {
                            target.insert(key.clone(), value.clone());
                        }
                    }
                    payload
                };
                match phase {
                    ToolCallPhase::Started { input } => self
                        .run
                        .record("tool_call.started", base(json!({ "input": input }))),
                    ToolCallPhase::StartedOversized { input } => self.run.record(
                        "tool_call.started",
                        base(json!({ "input": input, "oversized": true })),
                    ),
                    ToolCallPhase::Blocked { input, reason } => self.run.record(
                        "tool_call.blocked",
                        base(json!({ "input": input, "reason": reason })),
                    ),
                    ToolCallPhase::InProgress => {
                        self.run.record("tool_call.in_progress", base(json!({})))
                    }
                    ToolCallPhase::Failed {
                        reason,
                        permission_notice,
                        input,
                    } => self.run.record(
                        "tool_call.failed",
                        base(json!({
                            "reason": reason,
                            "permission_notice": permission_notice,
                            "input": input,
                        })),
                    ),
                    ToolCallPhase::Completed {
                        input,
                        output,
                        diff,
                        permission_notice,
                    } => self.run.record(
                        "tool_call.completed",
                        base(json!({
                            "input": input,
                            "output": output,
                            "diff": diff.map(|diff| json!({
                                "path": diff.path.display().to_string(),
                                "old_text": diff.old_text,
                                "new_text": diff.new_text,
                            })),
                            "permission_notice": permission_notice,
                        })),
                    ),
                }
            }
        }
    }
}

/// Permission broker for HTTP runs until #319 adds interactive responses:
/// surfaces the request as a run event, then rejects it so the turn keeps
/// moving instead of hanging on an unanswerable prompt.
struct AutoRejectingPermissionBroker {
    run: Arc<RunHandle>,
}

const HTTP_PERMISSION_UNSUPPORTED: &str = "interactive permission responses are not supported over the HTTP API yet; \
     the request was automatically rejected. Configure the session with \
     permission_mode \"auto\", \"acceptEdits\", or \"readOnly\" to avoid \
     interactive prompts.";

impl PermissionBroker for AutoRejectingPermissionBroker {
    fn request_permission(
        &self,
        prompt: PermissionPrompt,
    ) -> BoxFuture<'_, Result<PermissionDecision, String>> {
        self.run.record(
            "permission.requested",
            json!({
                "call_id": prompt.tool_call_id,
                "tool_name": prompt.tool_name,
                "input": prompt.raw_input,
                "permission_notice": prompt.permission_notice,
                "options": prompt
                    .options
                    .iter()
                    .map(|option| json!({ "id": option.id, "label": option.label }))
                    .collect::<Vec<_>>(),
                "auto_response": "rejected",
                "reason": HTTP_PERMISSION_UNSUPPORTED,
            }),
        );
        Box::pin(async { Err(HTTP_PERMISSION_UNSUPPORTED.to_string()) })
    }
}

fn delta_sink(run: Arc<RunHandle>, event_type: &'static str) -> TextSink {
    Arc::new(Mutex::new(move |token: &str| {
        run.record(event_type, json!({ "text": token }));
    }))
}

// ---------------------------------------------------------------------------
// Run execution
// ---------------------------------------------------------------------------

fn stop_reason_and_status(stop: &LoopStop, cancelled: bool) -> (RunStatus, &'static str) {
    match stop {
        LoopStop::Cancelled => (RunStatus::Cancelled, "cancelled"),
        _ if cancelled => (RunStatus::Cancelled, "cancelled"),
        LoopStop::Failed(_) => (RunStatus::Failed, "error"),
        LoopStop::MaxTurns { .. } => (RunStatus::Completed, "max_turns"),
        LoopStop::TimeLimit => (RunStatus::Completed, "time_limit"),
        LoopStop::Completed { .. } => (RunStatus::Completed, "end_turn"),
    }
}

fn usage_value(usage: crate::llm_client::TokenUsage) -> Value {
    json!({
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "thought_tokens": usage.thought_tokens,
        "cached_read_tokens": usage.cached_read_tokens,
        "cached_write_tokens": usage.cached_write_tokens,
    })
}

/// Drive one run to its terminal state. Owns `finish_prompt`: every exit
/// path releases the session's prompt slot before the terminal event is
/// observable, so a follow-up run is never rejected as already-in-flight
/// after its predecessor reported terminal.
async fn execute_run(
    state: ApiState,
    run: Arc<RunHandle>,
    mut snap: SessionSnapshot,
    prompt_text: String,
    structured_output: Option<StructuredOutputRequest>,
    cancel: tokio_util::sync::CancellationToken,
) {
    let session_id = run.session_id.clone();
    let sessions = state.sessions.clone();
    let llm: Arc<dyn crate::llm_client::LlmBackend> = state.llm.clone();

    let Some(registry) = sessions
        .get_or_create_registry(&session_id, snap.cwd.clone())
        .await
    else {
        sessions.finish_prompt(&session_id).await;
        state.runs.note_terminal(&run.id);
        run.finish(
            RunStatus::Failed,
            RunState {
                stop_reason: Some("error".to_string()),
                error: Some("session closed before the run started".to_string()),
                finished_at_ms: Some(now_ms()),
                ..RunState::default()
            },
            json!({
                "stop_reason": "error",
                "error": "session closed before the run started",
            }),
        );
        return;
    };

    let cwd = snap.cwd.clone();
    let model = snap.model.clone();
    let reasoning_effort = snap.reasoning_effort.clone();
    let service_tier = snap.service_tier.clone();
    let catalog = sessions.available_model_metadata().await;
    let context_length = catalog
        .iter()
        .find(|meta| meta.id == model)
        .and_then(|meta| meta.context_length);
    let idle_timeout = crate::acp::resolve_idle_timeouts(
        snap.idle_timeout_secs,
        state.default_idle_timeout_secs,
        state.default_stall_timeout_secs,
    );
    let turn_recap_enabled = sessions
        .turn_recap_enabled(&session_id)
        .await
        .unwrap_or(true);

    let tools_for_compression = registry.tool_definitions().await;
    let prepared = crate::acp::build_prompt_messages_with_compression(
        &mut snap,
        &prompt_text,
        &[],
        llm.as_ref(),
        &sessions,
        &session_id,
        cancel.clone(),
        idle_timeout,
        context_length,
        reasoning_effort.clone(),
        Some(&tools_for_compression),
    )
    .await;

    let outcome = crate::turn_runner::run_prompt_turn(crate::turn_runner::PromptTurnRequest {
        sessions: &sessions,
        session_id: &session_id,
        fallback_cwd: &cwd,
        llm: &llm,
        registry: &registry,
        model: &model,
        reasoning_effort: reasoning_effort.as_deref(),
        service_tier: service_tier.as_deref(),
        structured_output_request: structured_output.as_ref(),
        messages: prepared.messages,
        initial_usage: prepared.compaction_usage,
        context_length,
        context_prefix_len: prepared.prefix_len,
        initial_plan: prepared.current_plan,
        max_turns: state.max_turns,
        idle_timeout,
        cancel: cancel.clone(),
        turn_recap_enabled,
        prompt_text_for_turn: prompt_text,
        text_sink: delta_sink(run.clone(), "message.delta"),
        thought_sink: delta_sink(run.clone(), "thought.delta"),
        event_sink: &RunEventSink { run: run.clone() },
        permission_broker: &AutoRejectingPermissionBroker { run: run.clone() },
    })
    .await;

    sessions.finish_prompt(&session_id).await;

    let (status, stop_reason) = stop_reason_and_status(&outcome.stop, cancel.is_cancelled());
    let error = outcome
        .turn_failure()
        .map(|failure| failure.message.clone());
    let structured_output_value = outcome
        .structured_output
        .as_ref()
        .map(|result| serde_json::to_value(result).unwrap_or(Value::Null));
    let usage = usage_value(outcome.turn_usage);
    let cumulative_usage = usage_value(outcome.cumulative_usage);

    state.runs.note_terminal(&run.id);
    run.finish(
        status,
        RunState {
            stop_reason: Some(stop_reason.to_string()),
            error: error.clone(),
            result_text: Some(outcome.response.clone()),
            structured_output: structured_output_value.clone(),
            usage: Some(usage.clone()),
            finished_at_ms: Some(now_ms()),
        },
        json!({
            "stop_reason": stop_reason,
            "error": error,
            "result_text": outcome.response,
            "structured_output": structured_output_value,
            "usage": usage,
            "cumulative_usage": cumulative_usage,
        }),
    );
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CreateRunRequest {
    prompt: String,
    /// Optional structured-output constraint for the final response, in the
    /// same shape ACP carries in `_meta`.
    structured_output: Option<StructuredOutputSpec>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuredOutputSpec {
    schema: Value,
    schema_name: Option<String>,
    #[serde(default)]
    allow_coercion: bool,
    #[serde(default)]
    prefer_json_object: bool,
}

pub(super) async fn create_run(
    State(state): State<ApiState>,
    Path(session_id): Path<String>,
    ApiJson(request): ApiJson<CreateRunRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    if request.prompt.trim().is_empty() {
        return Err(ApiError::invalid_argument("prompt must not be empty")
            .details(json!({ "field": "prompt" })));
    }
    let structured_output = request
        .structured_output
        .map(|spec| StructuredOutputRequest {
            schema_name: spec
                .schema_name
                .unwrap_or_else(|| "structured_output".to_string()),
            schema: spec.schema,
            allow_coercion: spec.allow_coercion,
            prefer_json_object: spec.prefer_json_object,
        });

    let snap = state
        .sessions
        .snapshot(&session_id, &fallback_cwd(None))
        .await
        .ok_or_else(|| unknown_session_error(&session_id))?;
    if snap.model.is_empty() {
        return Err(ApiError::conflict(
            "session has no model configured; set one via PATCH /v1/sessions/{id} or restart \
             the daemon after model discovery succeeds",
        ));
    }

    let cancel = match state.sessions.start_prompt(&session_id).await {
        Ok(cancel) => cancel,
        Err(PromptStartError::AlreadyInFlight) => {
            return Err(ApiError::conflict(
                "a prompt is already in flight for this session; wait for the active run to \
                 reach a terminal state or cancel it",
            ));
        }
        Err(PromptStartError::UnknownSession) => {
            return Err(unknown_session_error(&session_id));
        }
    };

    let run = Arc::new(RunHandle::new(session_id.clone(), cancel.clone()));
    state.runs.insert(run.clone());
    run.record(
        "run.created",
        json!({ "prompt_chars": request.prompt.chars().count() }),
    );

    let resource = run.resource();
    tokio::spawn(execute_run(
        state,
        run,
        snap,
        request.prompt,
        structured_output,
        cancel,
    ));
    Ok((StatusCode::ACCEPTED, Json(resource)))
}

pub(super) async fn get_run(
    State(state): State<ApiState>,
    Path(run_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let run = state
        .runs
        .get(&run_id)
        .ok_or_else(|| ApiError::not_found(format!("unknown run '{run_id}'")))?;
    Ok(Json(run.resource()))
}

pub(super) async fn list_runs(
    State(state): State<ApiState>,
    Path(session_id): Path<String>,
) -> Json<Value> {
    let runs: Vec<Value> = state
        .runs
        .list_for_session(&session_id)
        .iter()
        .map(|run| run.resource())
        .collect();
    Json(json!({ "runs": runs }))
}

/// Cancel is idempotent and race-safe: it cancels through the same
/// `SessionStore` token the run started with, so a cancel that races the
/// turn's natural completion simply loses and the terminal state stands.
pub(super) async fn cancel_run(
    State(state): State<ApiState>,
    Path(run_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let run = state
        .runs
        .get(&run_id)
        .ok_or_else(|| ApiError::not_found(format!("unknown run '{run_id}'")))?;
    if !run.status().is_terminal() {
        run.cancel.cancel();
        state.sessions.cancel_prompt(&run.session_id).await;
    }
    Ok(Json(run.resource()))
}

#[derive(Debug, Deserialize)]
pub(super) struct RunEventsQuery {
    /// Resume after this sequence id (alternative to the standard
    /// `Last-Event-ID` header; the header wins when both are present).
    after_seq: Option<u64>,
}

struct SseStreamState {
    run: Arc<RunHandle>,
    backlog: VecDeque<RunEventRecord>,
    rx: broadcast::Receiver<RunEventRecord>,
    last_seq: u64,
    gap_notice: Option<u64>,
    done: bool,
}

fn sse_event(record: &RunEventRecord) -> SseEvent {
    SseEvent::default()
        .id(record.seq.to_string())
        .event(record.event_type)
        .data(record.data.clone())
}

fn is_terminal_event(record: &RunEventRecord) -> bool {
    matches!(
        record.event_type,
        "run.completed" | "run.cancelled" | "run.failed"
    )
}

pub(super) async fn run_events(
    State(state): State<ApiState>,
    Path(run_id): Path<String>,
    Query(query): Query<RunEventsQuery>,
    headers: HeaderMap,
) -> Result<Sse<impl Stream<Item = Result<SseEvent, Infallible>>>, ApiError> {
    let run = state
        .runs
        .get(&run_id)
        .ok_or_else(|| ApiError::not_found(format!("unknown run '{run_id}'")))?;

    let last_seq = headers
        .get("last-event-id")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .or(query.after_seq)
        .unwrap_or(0);

    // Subscribe before snapshotting the backlog so no event can fall between
    // the two; duplicates are dropped by the `seq <= last_seq` guard below.
    let rx = run.broadcast.subscribe();
    let backlog = run.events_after(last_seq);
    // A reconnect that fell behind the bounded replay window gets an explicit
    // gap notice instead of silently missing events.
    let gap_notice = run
        .earliest_buffered_seq()
        .and_then(|earliest| (last_seq > 0 && earliest > last_seq + 1).then_some(earliest));
    let done = run.status().is_terminal() && !backlog.iter().any(is_terminal_event);

    let stream_state = SseStreamState {
        run,
        backlog,
        rx,
        last_seq,
        gap_notice,
        done,
    };

    let stream = futures::stream::unfold(stream_state, |mut st| async move {
        if let Some(earliest) = st.gap_notice.take() {
            let event = SseEvent::default().event("events.gap").data(
                json!({
                    "type": "events.gap",
                    "run_id": st.run.id,
                    "missed_through_seq": earliest - 1,
                })
                .to_string(),
            );
            return Some((Ok::<_, Infallible>(event), st));
        }
        loop {
            if let Some(record) = st.backlog.pop_front() {
                if record.seq <= st.last_seq {
                    continue;
                }
                st.last_seq = record.seq;
                if is_terminal_event(&record) {
                    st.done = true;
                }
                return Some((Ok(sse_event(&record)), st));
            }
            if st.done {
                return None;
            }
            match st.rx.recv().await {
                Ok(record) => {
                    if record.seq <= st.last_seq {
                        continue;
                    }
                    st.last_seq = record.seq;
                    if is_terminal_event(&record) {
                        st.done = true;
                    }
                    return Some((Ok(sse_event(&record)), st));
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    // Fell behind the live channel: refill from the buffer.
                    st.backlog = st.run.events_after(st.last_seq);
                }
                Err(broadcast::error::RecvError::Closed) => return None,
            }
        }
    });

    Ok(Sse::new(stream).keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15))))
}
