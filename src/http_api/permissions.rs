//! Interactive permission requests over HTTP (#319).
//!
//! When a run's tool call needs an interactive decision, the broker
//! registers a pending permission resource, emits a `permission.requested`
//! run event carrying the permission id and options, and suspends the tool
//! loop until an authenticated client answers through
//! `POST /v1/permissions/{id}/respond` (or the run is cancelled, which
//! resolves the request as cancelled). Responses are validated against the
//! options the permission gate offered — the same option ids ACP clients
//! see (`allow`, `allow_always`, `allow_outside_sandbox`, `reject`).
//!
//! Race rules: a request can be resolved exactly once. Responses to an
//! unknown id are 404; responses after resolution (duplicate, or arriving
//! after run cancellation/completion) are 409. All decisions are written to
//! the `audit` log target.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::Json;
use axum::extract::{Path, State};
use futures::future::BoxFuture;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::oneshot;

use crate::runtime::{
    PermissionBroker, PermissionDecision, PermissionOption, PermissionOptionKind, PermissionPrompt,
};

use super::runs::RunHandle;
use super::{ApiError, ApiJson, ApiState};

fn option_kind_str(kind: PermissionOptionKind) -> &'static str {
    match kind {
        PermissionOptionKind::AllowOnce => "allow_once",
        PermissionOptionKind::AllowAlways => "allow_always",
        PermissionOptionKind::RejectOnce => "reject_once",
    }
}

pub(super) struct PendingPermission {
    pub(super) id: String,
    run: Arc<RunHandle>,
    tool_name: String,
    tool_call_id: String,
    raw_input: Value,
    permission_notice: Option<String>,
    options: Vec<PermissionOption>,
    created_at_ms: u64,
    responder: Mutex<Option<oneshot::Sender<PermissionDecision>>>,
}

impl PendingPermission {
    pub(super) fn resource(&self) -> Value {
        json!({
            "id": self.id,
            "run_id": self.run.id,
            "session_id": self.run.session_id,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "input": self.raw_input,
            "permission_notice": self.permission_notice,
            "options": self
                .options
                .iter()
                .map(|option| json!({
                    "id": option.id,
                    "label": option.label,
                    "kind": option_kind_str(option.kind),
                }))
                .collect::<Vec<_>>(),
            "created_at_ms": self.created_at_ms,
            "status": "pending",
        })
    }

    /// Deliver a decision exactly once. Returns false when the request was
    /// already resolved (duplicate response, or a response racing the run's
    /// cancellation/completion).
    fn resolve(&self, decision: PermissionDecision) -> bool {
        let sender = self
            .responder
            .lock()
            .expect("permission responder lock")
            .take();
        match sender {
            Some(sender) => sender.send(decision).is_ok(),
            None => false,
        }
    }
}

/// Pending permission requests across all runs, keyed by permission id.
#[derive(Default)]
pub(super) struct PermissionRegistry {
    pending: Mutex<HashMap<String, Arc<PendingPermission>>>,
}

impl PermissionRegistry {
    fn insert(&self, permission: Arc<PendingPermission>) {
        self.pending
            .lock()
            .expect("pending permissions lock")
            .insert(permission.id.clone(), permission);
    }

    fn remove(&self, permission_id: &str) {
        self.pending
            .lock()
            .expect("pending permissions lock")
            .remove(permission_id);
    }

    pub(super) fn get(&self, permission_id: &str) -> Option<Arc<PendingPermission>> {
        self.pending
            .lock()
            .expect("pending permissions lock")
            .get(permission_id)
            .cloned()
    }

    /// Synchronously expire every pending request for a run: each entry is
    /// resolved as cancelled through the same exactly-once transition
    /// `respond_permission` uses, and removed from the registry, before this
    /// returns. Called by run cancellation so a response arriving after the
    /// cancel endpoint returned deterministically finds nothing to approve.
    pub(super) fn expire_for_run(&self, run_id: &str) {
        let expired: Vec<Arc<PendingPermission>> = {
            let mut pending = self.pending.lock().expect("pending permissions lock");
            let ids: Vec<String> = pending
                .values()
                .filter(|permission| permission.run.id == run_id)
                .map(|permission| permission.id.clone())
                .collect();
            ids.iter().filter_map(|id| pending.remove(id)).collect()
        };
        for permission in expired {
            permission.resolve(PermissionDecision::Cancelled);
        }
    }

    pub(super) fn pending_for_run(&self, run_id: &str) -> Vec<Arc<PendingPermission>> {
        let mut pending: Vec<Arc<PendingPermission>> = self
            .pending
            .lock()
            .expect("pending permissions lock")
            .values()
            .filter(|permission| permission.run.id == run_id)
            .cloned()
            .collect();
        pending.sort_by_key(|permission| permission.created_at_ms);
        pending
    }
}

/// Broker handed to `turn_runner::run_prompt_turn` for HTTP runs: registers
/// the request, emits run events, and suspends until a client responds or
/// the run is cancelled.
pub(super) struct HttpPermissionBroker {
    pub(super) run: Arc<RunHandle>,
    pub(super) registry: Arc<PermissionRegistry>,
    pub(super) cancel: tokio_util::sync::CancellationToken,
}

impl PermissionBroker for HttpPermissionBroker {
    fn request_permission(
        &self,
        prompt: PermissionPrompt,
    ) -> BoxFuture<'_, Result<PermissionDecision, String>> {
        let (sender, receiver) = oneshot::channel();
        let permission = Arc::new(PendingPermission {
            id: format!("perm_{}", uuid::Uuid::new_v4().simple()),
            run: self.run.clone(),
            tool_name: prompt.tool_name.clone(),
            tool_call_id: prompt.tool_call_id.clone(),
            raw_input: prompt.raw_input.clone(),
            permission_notice: prompt.permission_notice.clone(),
            options: prompt.options.clone(),
            created_at_ms: super::runs::now_ms(),
            responder: Mutex::new(Some(sender)),
        });
        self.registry.insert(permission.clone());
        let mut requested = permission.resource();
        if let Some(object) = requested.as_object_mut() {
            object.remove("status");
            object.insert("permission_id".into(), json!(permission.id));
        }
        self.run.record("permission.requested", requested);
        tracing::info!(
            target: "audit",
            run_id = %self.run.id,
            session_id = %self.run.session_id,
            permission_id = %permission.id,
            tool_name = %permission.tool_name,
            "http permission requested"
        );

        Box::pin(async move {
            // `biased` so cancellation deterministically wins when a
            // response races it: a decision that lands after the cancel
            // token is set is discarded rather than approving a tool call
            // on a cancelled run.
            let decision = tokio::select! {
                biased;
                _ = self.cancel.cancelled() => {
                    // Run cancellation expires the pending request; a later
                    // response is rejected as already-resolved.
                    permission.resolve(PermissionDecision::Cancelled);
                    PermissionDecision::Cancelled
                }
                decision = receiver => decision.unwrap_or(PermissionDecision::Cancelled),
            };
            self.registry.remove(&permission.id);
            let decision_str = match &decision {
                PermissionDecision::Selected(option_id) => option_id.clone(),
                PermissionDecision::Cancelled => "cancelled".to_string(),
                PermissionDecision::Unsupported => "unsupported".to_string(),
            };
            self.run.record(
                "permission.resolved",
                json!({
                    "permission_id": permission.id,
                    "tool_call_id": permission.tool_call_id,
                    "tool_name": permission.tool_name,
                    "decision": decision_str,
                }),
            );
            tracing::info!(
                target: "audit",
                run_id = %self.run.id,
                session_id = %self.run.session_id,
                permission_id = %permission.id,
                tool_name = %permission.tool_name,
                decision = %decision_str,
                "http permission resolved"
            );
            Ok(decision)
        })
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

pub(super) async fn list_run_permissions(
    State(state): State<ApiState>,
    Path(run_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let run = state
        .runs
        .get(&run_id)
        .ok_or_else(|| ApiError::not_found(format!("unknown run '{run_id}'")))?;
    let permissions: Vec<Value> = state
        .permissions
        .pending_for_run(&run.id)
        .iter()
        .map(|permission| permission.resource())
        .collect();
    Ok(Json(json!({ "permissions": permissions })))
}

pub(super) async fn get_permission(
    State(state): State<ApiState>,
    Path(permission_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let permission = state.permissions.get(&permission_id).ok_or_else(|| {
        ApiError::not_found(format!(
            "unknown or already-resolved permission request '{permission_id}'"
        ))
    })?;
    Ok(Json(permission.resource()))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PermissionResponse {
    /// One of the option ids offered on the permission request
    /// (e.g. `allow`, `allow_always`, `reject`).
    option_id: Option<String>,
    /// Cancel the permission request without selecting an option; the tool
    /// call is denied as cancelled.
    #[serde(default)]
    cancel: bool,
}

pub(super) async fn respond_permission(
    State(state): State<ApiState>,
    Path(permission_id): Path<String>,
    ApiJson(response): ApiJson<PermissionResponse>,
) -> Result<Json<Value>, ApiError> {
    let permission = state.permissions.get(&permission_id).ok_or_else(|| {
        ApiError::not_found(format!(
            "unknown or already-resolved permission request '{permission_id}'"
        ))
    })?;

    if permission.run.cancel_requested() || permission.run.status().is_terminal() {
        return Err(ApiError::conflict(
            "permission request already resolved (duplicate response, or the run was \
             cancelled or completed first)",
        ));
    }

    let decision = match (&response.option_id, response.cancel) {
        (Some(_), true) => {
            return Err(ApiError::invalid_argument(
                "pass either option_id or cancel, not both",
            ));
        }
        (None, false) => {
            return Err(ApiError::invalid_argument(
                "option_id is required unless cancel is true",
            ));
        }
        (None, true) => PermissionDecision::Cancelled,
        (Some(option_id), false) => {
            if !permission
                .options
                .iter()
                .any(|option| &option.id == option_id)
            {
                let supported: Vec<String> = permission
                    .options
                    .iter()
                    .map(|option| option.id.clone())
                    .collect();
                return Err(ApiError::invalid_argument(format!(
                    "option '{option_id}' is not offered by this permission request"
                ))
                .details(json!({ "supported": supported })));
            }
            PermissionDecision::Selected(option_id.clone())
        }
    };

    if !permission.resolve(decision) {
        return Err(ApiError::conflict(
            "permission request already resolved (duplicate response, or the run was \
             cancelled or completed first)",
        ));
    }
    Ok(Json(json!({ "resolved": true, "id": permission.id })))
}
