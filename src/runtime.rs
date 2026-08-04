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
