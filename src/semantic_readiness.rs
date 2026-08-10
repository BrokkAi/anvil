//! Wait for bifrost's semantic index before the first agent turn.
//!
//! Bifrost opens and hydrates its semantic index asynchronously, by design
//! (bifrost AGENTS.md, "Index readiness design"): a tool call that needs the
//! index blocks until it is ready, and a client that would rather wait first
//! calls the readiness probe. A harness that measures tool latency must pick
//! one of those two patterns explicitly -- otherwise the one-time hydration of
//! a cold workspace lands inside whichever tool call happened to arrive first
//! and inflates it. On the r26 CodeScaleBench arms that was a 324 s charge
//! against a single `semantic_search` whose real retrievals ran 10-57 s.
//!
//! This module picks the first pattern: poll `semantic_search_status` (bifrost's
//! hidden, non-blocking probe) after the MCP connection is up and before the
//! first turn, then record the wait as session startup rather than tool time.
//! The indexer starts when bifrost assembles the session, so polling after
//! connect is valid -- there is nothing to kick off first.
//!
//! The wait is bounded. A broken or perpetually rebuilding index must not hang
//! a session forever, so on timeout the agent proceeds and the record says so.

use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::tools::ToolRegistry;
use crate::trace_logging::append_trace_record;

/// Bifrost's hidden readiness probe. Not advertised in `tools/list`, so it is
/// called on the client directly rather than through the model-facing dispatch.
const STATUS_TOOL: &str = "semantic_search_status";

/// Ceiling on the wait. r26 measured 324 s of hydration on firefox (170,889
/// files, 2,287,199 chunks), so the bound has to be minutes; past that the
/// index is presumed broken and the agent runs without waiting further.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);

/// Campaign override, in seconds. `0` disables the wait entirely.
const TIMEOUT_ENV: &str = "ANVIL_SEMANTIC_READY_TIMEOUT_SECS";

const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Why the wait ended. The value is the `phase` reported in the trace record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Readiness {
    /// The index reported `ready` with no pending batches.
    Ready,
    /// The index reported `failed` or `closed`; waiting longer cannot help.
    Stopped,
    /// The bound elapsed first. The agent proceeds and the first semantic call
    /// pays whatever hydration is left.
    TimedOut,
    /// The probe itself failed (older bifrost, nlp feature off, dead server).
    Unavailable,
    /// The turn was cancelled while waiting.
    Cancelled,
}

impl Readiness {
    fn label(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Stopped => "stopped",
            Self::TimedOut => "timed_out",
            Self::Unavailable => "unavailable",
            Self::Cancelled => "cancelled",
        }
    }
}

fn configured_timeout() -> Duration {
    match std::env::var(TIMEOUT_ENV) {
        Ok(value) => match value.trim().parse::<u64>() {
            Ok(secs) => Duration::from_secs(secs),
            Err(error) => {
                tracing::warn!(
                    value,
                    %error,
                    env_var = TIMEOUT_ENV,
                    "ignoring unparseable semantic readiness timeout"
                );
                DEFAULT_TIMEOUT
            }
        },
        Err(_) => DEFAULT_TIMEOUT,
    }
}

/// `semantic_search_status`, as far as this poll cares. Bifrost also reports
/// `materialized_files` / `materialize_total_files`, which ride along in the
/// trace record untouched.
fn ready_now(status: &Value) -> bool {
    status.get("phase").and_then(Value::as_str) == Some("ready")
        && status
            .get("pending_batches")
            .and_then(Value::as_u64)
            .is_some_and(|pending| pending == 0)
}

fn terminal(status: &Value) -> bool {
    matches!(
        status.get("phase").and_then(Value::as_str),
        Some("failed") | Some("closed")
    )
}

/// Block until bifrost's semantic index is live, the wait times out, or the
/// turn is cancelled. Returns `None` when there is nothing to wait for -- no
/// bifrost server, or semantic tools not enabled for this session -- in which
/// case no record is written either.
///
/// `workspace` labels the record. It is `None` for the session's opening wait,
/// where the active workspace is whatever bifrost started with, and the
/// activated path after an `activate_workspace` call. There is one wait per
/// workspace because bifrost keeps one index at a time: `activate_workspace`
/// assembles a replacement session and closes the previous indexer
/// (`searchtools_service.rs`, `handle_activate_workspace`), so every switch
/// hydrates again and would otherwise charge that hydration to whichever call
/// arrived first.
pub(crate) async fn wait_for_semantic_index(
    registry: &ToolRegistry,
    workspace: Option<&str>,
    cancel: &CancellationToken,
) -> Option<Readiness> {
    // `semantic_search` is advertised only when bifrost has the nlp feature,
    // a git workspace, and an accelerator. Its absence is the session-level
    // "semantic tools are off" signal, so there is nothing to hydrate.
    if !registry.is_bifrost_tool("semantic_search") {
        return None;
    }
    let client = registry.mcp_client("bifrost")?;
    let timeout = configured_timeout();
    if timeout.is_zero() {
        return None;
    }

    let started = Instant::now();
    let mut polls: u64 = 0;
    let mut last_status = Value::Null;
    let readiness = loop {
        if cancel.is_cancelled() {
            break Readiness::Cancelled;
        }
        polls += 1;
        match client
            .call_tool_cancellable(STATUS_TOOL, json!({}), Some(cancel))
            .await
        {
            Err(error) => {
                tracing::debug!(%error, "semantic readiness probe unavailable; not waiting");
                break Readiness::Unavailable;
            }
            Ok(status) => {
                let ready = ready_now(&status);
                let terminal = terminal(&status);
                last_status = status;
                if ready {
                    break Readiness::Ready;
                }
                if terminal {
                    break Readiness::Stopped;
                }
            }
        }
        if started.elapsed() >= timeout {
            break Readiness::TimedOut;
        }
        tokio::select! {
            _ = tokio::time::sleep(POLL_INTERVAL) => {}
            _ = cancel.cancelled() => break Readiness::Cancelled,
        }
    };

    let waited = started.elapsed();
    if readiness == Readiness::TimedOut {
        tracing::warn!(
            waited_secs = waited.as_secs(),
            status = %last_status,
            "semantic index still hydrating after the readiness bound; running anyway"
        );
    }
    append_trace_record(readiness_record(
        readiness,
        workspace,
        waited,
        polls,
        &last_status,
    ));
    Some(readiness)
}

/// Prehydration is session startup, not tool time, so it gets its own record
/// rather than riding inside a `tool_timing` duration.
///
/// The fields are named after bifrost's own `retrieval_timings` breakdown
/// (`wait_ready_ms`) so a consumer can fold them into the same hydration
/// accounting. They are deliberately *not* nested under a `retrieval_timings`
/// key: brokkbench's `_semantic_hydration` (f10a1bf7) charges every
/// `wait_ready_ms` it finds against `semantic_search` tool time, which was
/// right while the wait happened inside the first query and would now zero out
/// real retrieval time instead. A consumer that wants prehydration reads this
/// record's own `wait_ready_ms`.
fn readiness_record(
    readiness: Readiness,
    workspace: Option<&str>,
    waited: Duration,
    polls: u64,
    status: &Value,
) -> serde_json::Value {
    json!({
        "type": "semantic_index_prehydration",
        "phase": readiness.label(),
        "workspace": workspace,
        "wait_ready_ms": waited.as_millis(),
        "polls": polls,
        "status": status,
    })
}

/// A fake bifrost that advertises `semantic_search` and `activate_workspace`
/// and answers the hidden readiness probe: `starting` with a non-empty queue on
/// the first poll after a (re)build, `ready` afterwards. `activate_workspace`
/// resets that counter, the way the real server closes the old indexer and
/// hydrates the newly activated workspace from scratch. Line-framed so the
/// script stays readable.
#[cfg(test)]
#[cfg(unix)]
pub(crate) const FAKE_BIFROST: &str = r#"
import json, sys

polls = 0
def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    mid = msg.get("id")
    method = msg.get("method")
    if mid is None:
        continue
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "bifrost", "version": "0"}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [
            {"name": "semantic_search", "description": "fake",
             "inputSchema": {"type": "object", "properties": {}}},
            {"name": "activate_workspace", "description": "fake",
             "inputSchema": {"type": "object",
                             "properties": {"workspace_path": {"type": "string"}}}}]}})
    elif method == "tools/call":
        name = msg["params"]["name"]
        if name == "activate_workspace":
            polls = 0
            path = msg["params"].get("arguments", {}).get("workspace_path", "")
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "structuredContent": {"workspace_path": path},
                "content": [{"type": "text", "text": "activated " + path}]}})
            continue
        if name != "semantic_search_status":
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": "unexpected tool " + name}})
            continue
        polls += 1
        ready = polls >= 2
        status = {"phase": "ready" if ready else "starting",
                  "pending_batches": 0 if ready else 4,
                  "indexed_chunks": 7,
                  "materialized_files": 1,
                  "materialize_total_files": 1}
        send({"jsonrpc": "2.0", "id": mid, "result": {"structuredContent": status}})
    else:
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": method}})
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ready_needs_the_ready_phase_and_a_drained_queue() {
        assert!(ready_now(
            &json!({"phase": "ready", "pending_batches": 0, "indexed_chunks": 12})
        ));
        // Still ingesting: the phase flips to ready before the queue drains.
        assert!(!ready_now(&json!({"phase": "ready", "pending_batches": 3})));
        assert!(!ready_now(
            &json!({"phase": "starting", "pending_batches": 0})
        ));
        // An older bifrost that does not report the field is not ready.
        assert!(!ready_now(&json!({"phase": "ready"})));
    }

    #[test]
    fn failed_and_closed_end_the_wait() {
        assert!(terminal(&json!({"phase": "failed"})));
        assert!(terminal(&json!({"phase": "closed"})));
        assert!(!terminal(&json!({"phase": "starting"})));
        assert!(!terminal(&json!({"phase": "ready"})));
    }

    /// End to end over a real stdio MCP connection: the wait must poll until
    /// the index reports ready, and must leave the wait in the trace as its own
    /// prehydration record rather than inside any tool's time.
    #[cfg(unix)]
    #[tokio::test]
    async fn readiness_wait_polls_until_ready_and_records_prehydration() {
        use crate::mcp::{McpFraming, McpServerConfig};
        use std::sync::Arc;

        let cwd = tempfile::tempdir().expect("temp cwd");
        let script = cwd.path().join("fake_bifrost.py");
        std::fs::write(&script, FAKE_BIFROST).expect("write fake server");
        let trace = cwd.path().join("anvil-trace.jsonl");

        let registry = ToolRegistry::new(
            cwd.path().to_path_buf(),
            Vec::new(),
            vec![McpServerConfig {
                name: "bifrost".to_string(),
                transport: Default::default(),
                command: std::env::var("ANVIL_PYTHON").unwrap_or_else(|_| "python3".to_string()),
                url: None,
                headers: Vec::new(),
                args: vec![script.to_string_lossy().into_owned()],
                env: Vec::new(),
                framing: McpFraming::Line,
                enabled: true,
            }],
            Arc::new(crate::skills::SkillRegistry::default()),
            Arc::new(crate::agents::AgentRegistry::default()),
            Vec::new(),
            false,
        )
        .await;
        assert!(
            registry.is_bifrost_tool("semantic_search"),
            "the fake bifrost must advertise semantic_search for the wait to apply"
        );

        let readiness = crate::trace_logging::with_trace_path(
            &trace,
            wait_for_semantic_index(&registry, None, &CancellationToken::new()),
        )
        .await;
        assert_eq!(readiness, Some(Readiness::Ready));

        let lines = std::fs::read_to_string(&trace).unwrap_or_default();
        let records: Vec<Value> = lines
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .collect();
        let prehydration: Vec<&Value> = records
            .iter()
            .filter(|record| {
                record.get("type").and_then(Value::as_str) == Some("semantic_index_prehydration")
            })
            .collect();
        assert_eq!(
            prehydration.len(),
            1,
            "expected one prehydration record, got {records:?}"
        );
        let record = prehydration[0];
        assert_eq!(record["phase"], "ready");
        assert_eq!(
            record["polls"].as_u64(),
            Some(2),
            "the wait must poll again after `starting`: {record:?}"
        );
        assert!(record["wait_ready_ms"].is_number());
        assert_eq!(record["status"]["indexed_chunks"], 7);
        assert!(
            records
                .iter()
                .all(|record| record.get("type").and_then(Value::as_str) != Some("tool_timing")),
            "prehydration is session startup, not tool time: {records:?}"
        );
    }

    #[test]
    fn the_record_reports_the_wait_outside_any_retrieval_timings() {
        let record = readiness_record(
            Readiness::TimedOut,
            Some("/repos/kafka"),
            Duration::from_millis(1500),
            4,
            &json!({"phase": "starting", "pending_batches": 7}),
        );
        assert_eq!(record["type"], "semantic_index_prehydration");
        assert_eq!(record["phase"], "timed_out");
        assert_eq!(record["workspace"], "/repos/kafka");
        assert_eq!(record["wait_ready_ms"], 1500);
        assert_eq!(record["polls"], 4);
        assert_eq!(record["status"]["pending_batches"], 7);
        // Nesting this under `retrieval_timings` would make brokkbench charge
        // the wait against semantic_search tool time; see `readiness_record`.
        assert!(record.get("retrieval_timings").is_none());
    }
}
