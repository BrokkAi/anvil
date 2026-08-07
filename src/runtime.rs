use futures::future::BoxFuture;
use serde_json::Value;

/// A transport-independent permission request produced by the agent runtime.
#[derive(Debug, Clone)]
pub(crate) struct PermissionPrompt {
    pub session_id: String,
    pub tool_name: String,
    pub tool_call_id: String,
    pub raw_input: Value,
    pub permission_notice: Option<String>,
    pub options: Vec<PermissionOption>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PermissionOption {
    pub id: String,
    pub label: String,
    pub kind: PermissionOptionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PermissionOptionKind {
    AllowOnce,
    AllowAlways,
    RejectOnce,
}

/// The transport-independent outcome of asking a client for permission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PermissionDecision {
    Selected(String),
    Cancelled,
    Unsupported,
}

/// Boundary used by the runtime when a tool call needs an interactive decision.
///
/// ACP, HTTP, and in-process callers can implement this contract without the
/// tool loop knowing how the request is delivered or answered.
pub(crate) trait PermissionBroker: Send + Sync {
    fn request_permission(
        &self,
        prompt: PermissionPrompt,
    ) -> BoxFuture<'_, Result<PermissionDecision, String>>;
}

/// A transport-independent progress event emitted by the tool loop during a
/// prompt turn. The ACP adapter renders these into `SessionUpdate`
/// notifications (via `tool_loop::announce`); the HTTP adapter serializes
/// them onto a run's SSE stream.
///
/// Text and thought deltas are not events: they flow through the `TextSink`
/// callbacks `tool_loop::run` already takes, which are transport-neutral.
#[derive(Debug, Clone)]
pub(crate) enum RuntimeEvent {
    ToolCall {
        call_id: String,
        tool_name: String,
        phase: ToolCallPhase,
    },
    /// The model published or revised its plan (`update_plan` tool).
    Plan(crate::plan::UpdatePlanArgs),
}

/// Lifecycle phases of one tool call, in emission order. A call always opens
/// with exactly one of `Started`/`StartedOversized`/`Blocked` and, unless it
/// was `Blocked` (which is terminal), later reaches `Completed` or `Failed`,
/// optionally passing through `InProgress` when the permission gate allows
/// execution.
#[derive(Debug, Clone)]
pub(crate) enum ToolCallPhase {
    /// A new pending tool call announced with its parsed input.
    Started { input: Value },
    /// A new pending tool call whose rendered card would hide its input
    /// (oversized title/content); adapters use a static title instead.
    StartedOversized { input: Value },
    /// A call denied before execution by a deterministic preflight check.
    /// Terminal: adapters render both the (failed) card and the failure.
    Blocked { input: Value, reason: String },
    /// The permission gate allowed the call and execution began.
    InProgress,
    /// The call failed (invalid arguments, gate rejection, or execution
    /// failure). `input` is present for post-execution failures, where
    /// adapters re-render the call title and input alongside the reason.
    Failed {
        reason: String,
        permission_notice: Option<String>,
        input: Option<Value>,
    },
    /// The call executed successfully.
    Completed {
        input: Value,
        output: String,
        diff: Option<crate::session::ToolExchangeDiff>,
        permission_notice: Option<String>,
    },
}

/// Boundary through which the tool loop reports progress without knowing the
/// transport. Implementations must be cheap and non-blocking: events are
/// emitted inline on the tool-execution path.
pub(crate) trait EventSink: Send + Sync {
    fn emit(&self, session_id: &str, event: RuntimeEvent);
}
