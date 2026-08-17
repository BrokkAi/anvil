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
//! The probe has to name its workspace whenever the session has named ones.
//! Anvil runs bifrost in one of two shapes (`McpServerConfig::rendered_args`):
//! `--root <cwd>` for a single workspace, or `--workspace <name>=<path>` per
//! repository when the session configures analysis workspaces. In the second
//! shape bifrost routes every tool call through its named-workspace router,
//! which rejects any call without a `workspace` argument
//! (`rmcp_host.rs::prepare_named_tool_call`, "workspace must be one configured
//! name"). A nameless probe therefore fails on the first poll, and treating
//! that failure as "no index to wait for" skipped the wait entirely: in the
//! 2026-08-16 remeasure every task recorded a single `unavailable` poll and
//! firefox then paid 500-724 s of hydration inside its first `semantic_search`.
//! So the wait reads the session's configured workspaces and probes each one by
//! name, and every probe failure is logged rather than silently absorbed.
//!
//! The wait is bounded. A broken or perpetually rebuilding index must not hang
//! a session forever, so on timeout the agent proceeds and the record says so.
//! The bound covers the whole wait, not each workspace.

use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::mcp::McpClient;
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

/// `semantic_search_status`, as far as this poll cares: `phase` and
/// `pending_batches`, both untouched by bifrost c353c862. Bifrost also reports
/// a live-unit count -- `indexed_chunks` before that commit, `indexed_files`
/// after -- and `materialized_files` / `materialize_total_files`. None of them
/// gate readiness, so the whole status object rides into the trace record
/// verbatim and neither spelling needs a reader here.
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

/// One workspace to wait for: what the probe sends and what the record says.
enum ProbeTarget {
    /// Single-root bifrost (`--root <cwd>`). The probe carries no `workspace`
    /// argument, and the label is whatever the caller named -- `None` for the
    /// session's opening wait, the activated path after `activate_workspace`.
    Root(Option<String>),
    /// Named-workspace bifrost (`--workspace <name>=<path>`). The name is both
    /// the routing argument and the record's label.
    Named(String),
}

impl ProbeTarget {
    fn arguments(&self) -> Value {
        match self {
            Self::Root(_) => json!({}),
            Self::Named(name) => json!({ "workspace": name }),
        }
    }

    fn label(&self) -> Option<&str> {
        match self {
            Self::Root(label) => label.as_deref(),
            Self::Named(name) => Some(name),
        }
    }
}

/// The workspaces this wait must probe, read from the same configuration that
/// built bifrost's command line (`ToolRegistryOptions::analysis_workspaces` ->
/// `McpServerConfig::rendered_args`). With named workspaces every probe must
/// carry a configured name; without them the single-root probe stays nameless.
///
/// A caller that names one configured workspace -- the re-wait after switching
/// to it -- waits for that one alone. The session's opening wait names nothing
/// and therefore covers them all, because bifrost's router hydrates one index
/// per workspace and the first call into any of them would otherwise pay for it.
fn probe_targets(registry: &ToolRegistry, workspace: Option<&str>) -> Vec<ProbeTarget> {
    let configured = registry
        .analysis_workspaces()
        .filter(|workspaces| !workspaces.is_empty());
    let Some(configured) = configured else {
        return vec![ProbeTarget::Root(workspace.map(str::to_string))];
    };
    if let Some(selected) =
        workspace.filter(|name| configured.iter().any(|entry| entry.name == *name))
    {
        return vec![ProbeTarget::Named(selected.to_string())];
    }
    configured
        .iter()
        .map(|entry| ProbeTarget::Named(entry.name.clone()))
        .collect()
}

/// Block until bifrost's semantic index is live, the wait times out, or the
/// turn is cancelled. Returns `None` when there is nothing to wait for -- no
/// bifrost server, or semantic tools not enabled for this session -- in which
/// case no record is written either.
///
/// `workspace` selects and labels the wait. It is `None` for the session's
/// opening wait, where the active workspace is whatever bifrost started with,
/// and the activated workspace after an `activate_workspace` call. There is one
/// wait per workspace because bifrost keeps one index per workspace and, in the
/// single-root shape, one at a time: `activate_workspace` assembles a
/// replacement session and closes the previous indexer
/// (`searchtools_service.rs`, `handle_activate_workspace`), so every switch
/// hydrates again and would otherwise charge that hydration to whichever call
/// arrived first.
///
/// With named workspaces each one is probed by name and the returned value is
/// the first non-`Ready` outcome, so a caller sees `Ready` only when every
/// workspace is live. Each probe writes its own record.
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

    // One deadline for the whole wait: with four workspaces configured the
    // session still starts within the bound, it just may start with a
    // half-hydrated index and a record saying which one.
    let started = Instant::now();
    let mut outcome = Readiness::Ready;
    for target in probe_targets(registry, workspace) {
        let readiness = wait_for_workspace(client, &target, started, timeout, cancel).await;
        if outcome == Readiness::Ready {
            outcome = readiness;
        }
        if readiness == Readiness::Cancelled {
            break;
        }
    }
    Some(outcome)
}

/// Poll one workspace to readiness and write its prehydration record.
/// `started` and `timeout` are the shared bound, so a workspace that begins its
/// wait after the bound has passed still reports one status snapshot.
async fn wait_for_workspace(
    client: &McpClient,
    target: &ProbeTarget,
    started: Instant,
    timeout: Duration,
    cancel: &CancellationToken,
) -> Readiness {
    let probe_started = Instant::now();
    let mut polls: u64 = 0;
    let mut last_status = Value::Null;
    let readiness = loop {
        if cancel.is_cancelled() {
            break Readiness::Cancelled;
        }
        polls += 1;
        match client
            .call_tool_cancellable(STATUS_TOOL, target.arguments(), Some(cancel))
            .await
        {
            Err(error) => {
                // Warn, never debug: a probe that fails on every poll is how
                // the nameless-probe bug hid for a whole measurement campaign.
                // The genuine cases (older bifrost, nlp off, dead server) are
                // rare enough that one warning per session is cheap.
                tracing::warn!(
                    %error,
                    workspace = target.label().unwrap_or("<session default>"),
                    "semantic readiness probe failed; not waiting for this workspace"
                );
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

    let waited = probe_started.elapsed();
    if readiness == Readiness::TimedOut {
        tracing::warn!(
            waited_secs = waited.as_secs(),
            workspace = target.label().unwrap_or("<session default>"),
            status = %last_status,
            "semantic index still hydrating after the readiness bound; running anyway"
        );
    }
    append_trace_record(readiness_record(
        readiness,
        target.label(),
        waited,
        polls,
        &last_status,
    ));
    readiness
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

/// A fake bifrost that advertises `semantic_search` and answers the hidden
/// readiness probe: `starting` with a non-empty queue on the first poll after a
/// (re)build, `ready` afterwards. Line-framed so the script stays readable.
///
/// It takes both of the real server's shapes from its own command line, the way
/// `McpServerConfig::rendered_args` writes them:
///
/// - No `--workspace` argument: single-root mode. `activate_workspace` is
///   advertised and resets the poll counter, the way the real server closes the
///   old indexer and hydrates the newly activated workspace from scratch. A
///   `workspace` argument is rejected -- there is no router to route it.
/// - One or more `--workspace <name>=<path>`: named mode. `activate_workspace`
///   is withdrawn (`rmcp_host.rs` removes it), each workspace hydrates on its
///   own counter, and any call without a configured name is rejected with
///   bifrost's own `invalid_params` message. That rejection is the bug this
///   module exists to prevent: a nameless probe never sees `ready`.
#[cfg(test)]
#[cfg(unix)]
pub(crate) const FAKE_BIFROST: &str = r#"
import json, sys

named = {}
argv = sys.argv[1:]
i = 0
while i < len(argv):
    if argv[i] == "--workspace" and i + 1 < len(argv):
        name, _, path = argv[i + 1].partition("=")
        named[name] = path
        i += 2
    else:
        i += 1

polls = {}
def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def reject(mid, message):
    send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": message}})

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
        tools = [{"name": "semantic_search", "description": "fake",
                  "inputSchema": {"type": "object", "properties": {}}}]
        if not named:
            tools.append({"name": "activate_workspace", "description": "fake",
                          "inputSchema": {"type": "object",
                                          "properties": {"workspace_path": {"type": "string"}}}})
        send({"jsonrpc": "2.0", "id": mid, "result": {"tools": tools}})
    elif method == "tools/call":
        name = msg["params"]["name"]
        arguments = msg["params"].get("arguments") or {}
        workspace = arguments.get("workspace")
        if named:
            if not isinstance(workspace, str) or workspace not in named:
                reject(mid, "workspace must be one configured name")
                continue
        elif workspace is not None:
            reject(mid, "workspace is not accepted without configured workspaces")
            continue
        key = workspace if named else ""
        if name == "activate_workspace":
            polls[key] = 0
            path = arguments.get("workspace_path", "")
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "structuredContent": {"workspace_path": path},
                "content": [{"type": "text", "text": "activated " + path}]}})
            continue
        if name != "semantic_search_status":
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": "unexpected tool " + name}})
            continue
        polls[key] = polls.get(key, 0) + 1
        ready = polls[key] >= 2
        status = {"phase": "ready" if ready else "starting",
                  "pending_batches": 0 if ready else 4,
                  "indexed_files": 7,
                  "materialized_files": 1,
                  "materialize_total_files": 1,
                  "workspace": key}
        send({"jsonrpc": "2.0", "id": mid, "result": {"structuredContent": status}})
    else:
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": method}})
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// Bifrost c353c862 renamed the status field `indexed_chunks` to
    /// `indexed_files` (an exact live chunk count was the workspace-wide join
    /// that change deletes). Readiness reads neither, so both spellings -- and
    /// a server reporting neither -- must behave identically here.
    #[test]
    fn the_renamed_live_unit_count_does_not_gate_readiness() {
        let ready = json!({"phase": "ready", "pending_batches": 0});
        let old = json!({"phase": "ready", "pending_batches": 0, "indexed_chunks": 12});
        let new = json!({"phase": "ready", "pending_batches": 0, "indexed_files": 12});
        assert!(ready_now(&ready));
        assert!(ready_now(&old));
        assert!(ready_now(&new));
        // Whatever the server reported reaches the trace record untouched, so a
        // consumer can read either name without anvil translating it.
        for status in [&ready, &old, &new] {
            let record =
                readiness_record(Readiness::Ready, None, Duration::from_millis(3), 1, status);
            assert_eq!(record["status"], *status);
        }
    }

    #[test]
    fn ready_needs_the_ready_phase_and_a_drained_queue() {
        assert!(ready_now(
            &json!({"phase": "ready", "pending_batches": 0, "indexed_files": 12})
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

    /// A registry whose only MCP server is [`FAKE_BIFROST`], spawned through
    /// the real workspace-args expansion so the fake sees the same command line
    /// the real bifrost would: `--root <cwd>` without analysis workspaces, one
    /// `--workspace <name>=<path>` pair per workspace with them.
    #[cfg(unix)]
    async fn fake_bifrost_registry(
        cwd: &std::path::Path,
        analysis_workspaces: Option<Vec<crate::session::AnalysisWorkspace>>,
    ) -> ToolRegistry {
        use crate::mcp::{BIFROST_WORKSPACE_ARGS_PLACEHOLDER, McpFraming, McpServerConfig};
        use std::sync::Arc;

        let script = cwd.join("fake_bifrost.py");
        std::fs::write(&script, FAKE_BIFROST).expect("write fake server");
        ToolRegistry::new(
            cwd.to_path_buf(),
            Vec::new(),
            vec![McpServerConfig {
                name: "bifrost".to_string(),
                transport: Default::default(),
                command: std::env::var("ANVIL_PYTHON").unwrap_or_else(|_| "python3".to_string()),
                url: None,
                headers: Vec::new(),
                args: vec![
                    script.to_string_lossy().into_owned(),
                    BIFROST_WORKSPACE_ARGS_PLACEHOLDER.to_string(),
                ],
                env: Vec::new(),
                framing: McpFraming::Line,
                enabled: true,
            }],
            Arc::new(crate::skills::SkillRegistry::default()),
            Arc::new(crate::agents::AgentRegistry::default()),
            Vec::new(),
            crate::tools::ToolRegistryOptions {
                analysis_workspaces,
                lsp_settings: crate::lsp::LspSettings::default(),
                shell_minimizer_enabled: false,
            },
        )
        .await
    }

    #[cfg(unix)]
    fn workspaces(entries: &[(&str, &std::path::Path)]) -> Vec<crate::session::AnalysisWorkspace> {
        entries
            .iter()
            .map(|(name, path)| crate::session::AnalysisWorkspace {
                name: (*name).to_string(),
                path: path.to_path_buf(),
            })
            .collect()
    }

    #[cfg(unix)]
    fn trace_records(trace: &std::path::Path) -> Vec<Value> {
        std::fs::read_to_string(trace)
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .collect()
    }

    #[cfg(unix)]
    fn prehydration(records: &[Value]) -> Vec<&Value> {
        records
            .iter()
            .filter(|record| {
                record.get("type").and_then(Value::as_str) == Some("semantic_index_prehydration")
            })
            .collect()
    }

    /// End to end over a real stdio MCP connection: the wait must poll until
    /// the index reports ready, and must leave the wait in the trace as its own
    /// prehydration record rather than inside any tool's time.
    #[cfg(unix)]
    #[tokio::test]
    async fn readiness_wait_polls_until_ready_and_records_prehydration() {
        let cwd = tempfile::tempdir().expect("temp cwd");
        let trace = cwd.path().join("anvil-trace.jsonl");
        let registry = fake_bifrost_registry(cwd.path(), None).await;
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

        let records = trace_records(&trace);
        let prehydration = prehydration(&records);
        assert_eq!(
            prehydration.len(),
            1,
            "expected one prehydration record, got {records:?}"
        );
        let record = prehydration[0];
        assert_eq!(record["phase"], "ready");
        // A single-root session has no name to send, and the record has none
        // to report.
        assert!(record["workspace"].is_null(), "{record:?}");
        assert_eq!(
            record["polls"].as_u64(),
            Some(2),
            "the wait must poll again after `starting`: {record:?}"
        );
        assert!(record["wait_ready_ms"].is_number());
        assert_eq!(record["status"]["indexed_files"], 7);
        assert!(
            records
                .iter()
                .all(|record| record.get("type").and_then(Value::as_str) != Some("tool_timing")),
            "prehydration is session startup, not tool time: {records:?}"
        );
    }

    /// The bug this module was failing to prevent. Every CodeScale eval runs
    /// bifrost in named-workspace mode, where the router rejects a call that
    /// names no workspace. The wait must send each configured name, wait for
    /// every one of them, and attribute each wait in its own record -- not
    /// collect one `invalid_params` rejection and skip the wait.
    #[cfg(unix)]
    #[tokio::test]
    async fn named_workspaces_are_each_probed_by_name_and_all_waited_for() {
        let cwd = tempfile::tempdir().expect("temp cwd");
        let trace = cwd.path().join("anvil-trace.jsonl");
        let backend = cwd.path().join("backend");
        let frontend = cwd.path().join("frontend");
        let registry = fake_bifrost_registry(
            cwd.path(),
            Some(workspaces(&[
                ("backend", backend.as_path()),
                ("frontend", frontend.as_path()),
            ])),
        )
        .await;

        let readiness = crate::trace_logging::with_trace_path(
            &trace,
            wait_for_semantic_index(&registry, None, &CancellationToken::new()),
        )
        .await;
        assert_eq!(readiness, Some(Readiness::Ready));

        let records = trace_records(&trace);
        let prehydration = prehydration(&records);
        assert_eq!(
            prehydration.len(),
            2,
            "one record per configured workspace, got {records:?}"
        );
        let named: Vec<&str> = prehydration
            .iter()
            .filter_map(|record| record["workspace"].as_str())
            .collect();
        assert_eq!(named, vec!["backend", "frontend"], "{records:?}");
        for record in &prehydration {
            assert_eq!(record["phase"], "ready", "{record:?}");
            // Each workspace hydrates on its own counter, so each is polled
            // until it reports ready rather than riding on a sibling's status.
            assert_eq!(record["polls"].as_u64(), Some(2), "{record:?}");
            assert_eq!(record["status"]["workspace"], record["workspace"]);
        }
    }

    /// The rejection the old nameless probe collected, kept as the contrast
    /// case: with the same named bifrost but no configured names to send, the
    /// probe is refused and the wait ends after one poll. That is the exact
    /// signature the 2026-08-16 remeasure traces carried, and the reason the
    /// names have to come from the session configuration rather than be
    /// guessed here.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_nameless_probe_against_a_named_bifrost_is_refused() {
        use crate::mcp::{McpFraming, McpServerConfig};
        use std::sync::Arc;

        let cwd = tempfile::tempdir().expect("temp cwd");
        let trace = cwd.path().join("anvil-trace.jsonl");
        let script = cwd.path().join("fake_bifrost.py");
        std::fs::write(&script, FAKE_BIFROST).expect("write fake server");
        // Named on the wire, unnamed in the registry: the mismatch the fix
        // removes by reading the names from the same place the args come from.
        let registry = ToolRegistry::new(
            cwd.path().to_path_buf(),
            Vec::new(),
            vec![McpServerConfig {
                name: "bifrost".to_string(),
                transport: Default::default(),
                command: std::env::var("ANVIL_PYTHON").unwrap_or_else(|_| "python3".to_string()),
                url: None,
                headers: Vec::new(),
                args: vec![
                    script.to_string_lossy().into_owned(),
                    "--workspace".to_string(),
                    format!("backend={}", cwd.path().display()),
                ],
                env: Vec::new(),
                framing: McpFraming::Line,
                enabled: true,
            }],
            Arc::new(crate::skills::SkillRegistry::default()),
            Arc::new(crate::agents::AgentRegistry::default()),
            Vec::new(),
            crate::tools::ToolRegistryOptions {
                analysis_workspaces: None,
                lsp_settings: crate::lsp::LspSettings::default(),
                shell_minimizer_enabled: false,
            },
        )
        .await;

        let readiness = crate::trace_logging::with_trace_path(
            &trace,
            wait_for_semantic_index(&registry, None, &CancellationToken::new()),
        )
        .await;
        assert_eq!(readiness, Some(Readiness::Unavailable));
        let records = trace_records(&trace);
        let prehydration = prehydration(&records);
        assert_eq!(prehydration.len(), 1, "{records:?}");
        assert_eq!(prehydration[0]["phase"], "unavailable");
        assert_eq!(prehydration[0]["polls"].as_u64(), Some(1));
    }

    /// Which workspaces a wait covers. The opening wait covers every
    /// configured one; a wait that names a configured workspace -- the re-wait
    /// after activating it -- covers that one; a session with no configured
    /// workspaces keeps the nameless single-root probe, label and all.
    #[tokio::test]
    async fn probe_targets_follow_the_configured_workspaces() {
        let cwd = tempfile::tempdir().expect("temp cwd");
        let single_root = ToolRegistry::new(
            cwd.path().to_path_buf(),
            Vec::new(),
            Vec::new(),
            std::sync::Arc::new(crate::skills::SkillRegistry::default()),
            std::sync::Arc::new(crate::agents::AgentRegistry::default()),
            Vec::new(),
            crate::tools::ToolRegistryOptions {
                analysis_workspaces: None,
                lsp_settings: crate::lsp::LspSettings::default(),
                shell_minimizer_enabled: false,
            },
        )
        .await;
        let targets = probe_targets(&single_root, Some("/repos/kafka"));
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].label(), Some("/repos/kafka"));
        assert_eq!(
            targets[0].arguments(),
            json!({}),
            "a single-root bifrost has no router to name a workspace to"
        );

        let named = ToolRegistry::new(
            cwd.path().to_path_buf(),
            Vec::new(),
            Vec::new(),
            std::sync::Arc::new(crate::skills::SkillRegistry::default()),
            std::sync::Arc::new(crate::agents::AgentRegistry::default()),
            Vec::new(),
            crate::tools::ToolRegistryOptions {
                analysis_workspaces: Some(vec![
                    crate::session::AnalysisWorkspace {
                        name: "backend".to_string(),
                        path: cwd.path().join("backend"),
                    },
                    crate::session::AnalysisWorkspace {
                        name: "frontend".to_string(),
                        path: cwd.path().join("frontend"),
                    },
                ]),
                lsp_settings: crate::lsp::LspSettings::default(),
                shell_minimizer_enabled: false,
            },
        )
        .await;
        let opening = probe_targets(&named, None);
        assert_eq!(
            opening
                .iter()
                .map(|target| target.arguments())
                .collect::<Vec<_>>(),
            vec![
                json!({"workspace": "backend"}),
                json!({"workspace": "frontend"})
            ]
        );

        let after_activation = probe_targets(&named, Some("frontend"));
        assert_eq!(after_activation.len(), 1);
        assert_eq!(after_activation[0].label(), Some("frontend"));
        assert_eq!(
            after_activation[0].arguments(),
            json!({"workspace": "frontend"})
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
