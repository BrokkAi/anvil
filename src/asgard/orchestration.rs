use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use agent_client_protocol::schema::v1::{
    ContentBlock, ContentChunk, SessionNotification, SessionUpdate, TextContent,
};
use agent_client_protocol::{Client, ConnectionTo};
use anyhow::Context as _;
use sha2::{Digest, Sha256};

use crate::llm_client::{
    ChatContentPart, ChatMessage, FunctionDef, IdleTimeouts, LlmResponse, StreamChatRequest,
    ToolDefinition, stream_chat_no_visible_output_with_retry,
};
use crate::session::SessionStore;
use crate::structured_output::StructuredOutputRequest;

fn send_thought(cx: &ConnectionTo<Client>, session_id: &str, text: &str) {
    let chunk = ContentChunk::new(ContentBlock::Text(TextContent::new(text)));
    let update = SessionUpdate::AgentThoughtChunk(chunk);
    let notification = SessionNotification::new(session_id.to_string(), update);
    if let Err(e) = cx.send_notification(notification) {
        tracing::warn!("failed to send thought session update: {e}");
    }
}

#[derive(Clone)]
pub(crate) struct AsgardLiveOutput {
    pub(crate) cx: ConnectionTo<Client>,
    pub(crate) session_id: String,
    pub(crate) last_source: Arc<std::sync::Mutex<Option<String>>>,
}

impl AsgardLiveOutput {
    fn new(cx: &ConnectionTo<Client>, session_id: &str) -> Self {
        Self {
            cx: cx.clone(),
            session_id: session_id.to_string(),
            last_source: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    fn sink(&self, source: impl Into<String>) -> crate::tool_loop::TextSink {
        let output = self.clone();
        let source = source.into();
        Arc::new(std::sync::Mutex::new(move |text: &str| {
            if text.is_empty() {
                return;
            }
            let mut last_source = output
                .last_source
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if last_source.as_deref() != Some(source.as_str()) {
                send_thought(
                    &output.cx,
                    &output.session_id,
                    &format!("\n[Asgard {source}]\n"),
                );
                *last_source = Some(source.clone());
            }
            send_thought(&output.cx, &output.session_id, text);
        }))
    }
}

#[derive(Clone)]
pub(crate) struct AsgardStreamSinks {
    pub(crate) text: crate::tool_loop::TextSink,
    pub(crate) thought: crate::tool_loop::TextSink,
}

impl AsgardStreamSinks {
    fn new(output: &AsgardLiveOutput, source: &str) -> Self {
        Self {
            text: output.sink(format!("{source} output")),
            thought: output.sink(format!("{source} reasoning")),
        }
    }
}

pub(crate) struct AsgardCandidate {
    pub(crate) index: usize,
    pub(crate) model: String,
    pub(crate) outcome: crate::tool_loop::LoopOutcome,
    pub(crate) patch: Vec<u8>,
    pub(crate) delta_patch: Vec<u8>,
    pub(crate) window_messages: Vec<ChatMessage>,
    pub(crate) supervisor_window_messages: Vec<ChatMessage>,
    pub(crate) window_ledger: AsgardExecutionLedger,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AsgardCandidateDiffBase {
    LastSelectedDecision,
    CurrentWindowLane(usize),
}

impl AsgardCandidateDiffBase {
    fn label(&self) -> String {
        match self {
            Self::LastSelectedDecision => "last_selected_decision".to_string(),
            Self::CurrentWindowLane(lane) => format!("current_window_lane_{lane}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AsgardCandidateDiffView {
    pub(crate) candidate_index: usize,
    pub(crate) base: AsgardCandidateDiffBase,
    pub(crate) patch: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AsgardDiffPresentation {
    pub(crate) views: Vec<AsgardCandidateDiffView>,
    pub(crate) anchor_lane: Option<usize>,
    pub(crate) baseline_sum_bytes: usize,
    pub(crate) best_candidate_sum_bytes: Option<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct AsgardDiffCandidateInput {
    pub(crate) index: usize,
    pub(crate) patch: Vec<u8>,
    pub(crate) delta_patch: Vec<u8>,
    pub(crate) repository_root: PathBuf,
    pub(crate) base_commit: String,
}

pub(crate) struct AsgardCandidateDiffRow {
    pub(crate) anchor_lane: usize,
    pub(crate) sum_bytes: usize,
    pub(crate) patches: Vec<(usize, Vec<u8>)>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(crate) struct AsgardLedgerEntry {
    pub(crate) id: String,
    pub(crate) step: usize,
    pub(crate) command: String,
    pub(crate) exit_code: Option<i32>,
    pub(crate) output_bytes: usize,
    pub(crate) output_sha256: String,
    pub(crate) output_tail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(crate) struct AsgardLedgerEdit {
    pub(crate) step: usize,
    pub(crate) file: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, Default)]
pub(crate) struct AsgardExecutionLedger {
    pub(crate) entries: Vec<AsgardLedgerEntry>,
    pub(crate) edit_steps: Vec<AsgardLedgerEdit>,
    pub(crate) total_shell_commands: usize,
    pub(crate) entries_truncated: bool,
}

pub(crate) const ASGARD_CONTRACT_EXTRACTION_PROMPT: &str = "You are extracting the explicit contract checklist from a software task description, before any implementation exists. List every externally checkable requirement the task states: function and method signatures including argument order and curried versus non-curried forms, exact strings, separators, suffixes, and output formats, boundary values and their required handling, lifecycle and cancellation obligations (what must unblock, interrupt, close, or clean up what, including operations already in flight), concurrency and atomicity requirements, error types and exact error text, compatibility constraints, and explicit prohibitions. Quote the task's own words wherever possible. One requirement per entry; split compound sentences into separate entries. Do not invent requirements the task does not state and do not add generic quality goals. Tag each entry's kind: \"inspection\" when a reviewer can verify it by reading the final code (signatures, argument order, exact strings present in source, type shapes); \"execution\" when verification requires running a scenario (blocking and unblocking, timing, cancellation of in-flight operations, end-to-end produced output, formatting of emitted streams); \"delivery\" for repository-delivery obligations that do not affect runtime behavior (working on a named branch, committing the work, repository cleanliness). For each entry, if the requirement only holds meaning under a specific adverse condition — an operation already blocked or in flight when the triggering event fires, an exhausted window or resource, an externally stalled dependency, a boundary or zero value — record that condition verbatim in adverse_condition; otherwise set adverse_condition to null. When the task names classification values, categories, or enumerated labels (for example conflict types, error kinds, source values, status codes), add a separate contract that each reported classification carries the correct value for its specific scenario — distinct from, and in addition to, the contract that the item is detected or reported at all; verifying a count or presence does not verify a label. When a requirement combines, maps, constructs from, or dispatches over multiple positional inputs (combining N containers, building a record from several fields, constructor argument lists), add a separate contract, kind \"inspection\", that each input's value reaches its correct output position, with adverse_condition stating that verification requires either quoting the complete argument flow from construction to application, or execution with pairwise distinguishable inputs and a non-commutative operation — green tests with same-valued or commutative examples prove nothing about position. Success paths get adverse conditions too: when a requirement's happy path can be partially wrong (right for the first element, the common case, or homogeneous inputs), record the discriminating input shape as its adverse_condition rather than leaving it null. When the task requires emitting a well-known textual format or stream whose name implies standard structural conventions (for example a manifest stream with document separators and source annotations, a unified diff, a standard header block), also add one contract per structural convention the named format prescribes — including how the stream begins, how documents or sections are delimited, and how identifying annotations are formed and normalized — marked kind \"execution\" and quoting the format name from the task. Only do this for formats whose conventions you are certain of; do not invent conventions. When the task defines a typed public API whose signatures involve callbacks, generics, overloads, or container/element relationships — shapes where a misreading changes what external callers must write — add one contract per such signature stating its exact parameter and return types in the task's words, kind \"execution\", with adverse_condition: verification requires type-checking usage authored from this contract's text alone, not adapted from the implementation's own tests. Do not create type-shape contracts for simple scalar annotations; those are ordinary inspection contracts. If a requirement admits two materially different readings, record the contract once per reading and set each adverse_condition to the ambiguity it resolves; do not silently pick one reading. Call extract_task_contracts exactly once. Do not answer in prose.";
pub(crate) const ASGARD_CONTRACT_EXTRACTION_MAX_ATTEMPTS: usize = 3;
pub(crate) const ASGARD_AUDIT_BUILTIN_TOOLS: &[&str] =
    &["read_file", "grep_search", "list_directory"];
pub(crate) const ASGARD_AUDIT_BIFROST_TOOLS: &[&str] = &[
    "search_symbols",
    "get_symbol_sources",
    "get_symbol_locations",
    "get_summaries",
];
pub(crate) const ASGARD_AUDIT_MAX_ROUNDS: usize = 3;
pub(crate) const ASGARD_VIEW_TOOL_CALL_NAME: &str = "view_tool_call";
pub(crate) const ASGARD_VIEW_TOOL_CALL_MAX_HANDLES: usize = 16;
/// Retrieval rounds are deliberately not budgeted: expanding a compact line
/// costs no execution and the supervisor should never have to ration reading
/// evidence it was shown. This is a runaway guard so an unproductive loop
/// terminates, not a budget, and should never bind in practice.
pub(crate) const ASGARD_RETRIEVAL_MAX_ROUNDS: usize = 40;
pub(crate) const ASGARD_SELECTION_MAX_ATTEMPTS: usize = 3;
pub(crate) const ASGARD_REVIEW_SELECTION_MAX_ATTEMPTS: usize = 6;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(crate) struct AsgardCandidatePatchManifest {
    pub(crate) candidate_changed_production_files: Vec<String>,
    pub(crate) candidate_created_test_files: Vec<String>,
    pub(crate) candidate_modified_test_files: Vec<String>,
    pub(crate) patch_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AsgardSupervisorHistoryEntry {
    pub(crate) window: usize,
    pub(crate) winner: usize,
    pub(crate) state_summary: String,
}

#[derive(Debug, Default)]
pub(crate) struct AsgardSupervisorHistory {
    pub(crate) checkpointed: Vec<AsgardSupervisorHistoryEntry>,
    pub(crate) selected_windows: Vec<AsgardSupervisorHistoryEntry>,
}

impl AsgardSupervisorHistory {
    pub(crate) fn push(&mut self, window: usize, decision: &AsgardSupervisorDecision) {
        self.selected_windows.push(AsgardSupervisorHistoryEntry {
            window,
            winner: decision.winner,
            state_summary: decision.state_summary.clone(),
        });
    }

    pub(crate) fn checkpoint_selected_windows(&mut self) {
        self.checkpointed.append(&mut self.selected_windows);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(crate) struct AsgardSupervisorDecision {
    pub(crate) winner: usize,
    pub(crate) complete: bool,
    pub(crate) advices: Vec<Option<String>>,
    pub(crate) next_window_steps: Option<usize>,
    pub(crate) state_summary: String,
    pub(crate) contracts: Option<Vec<AsgardContractRow>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AsgardChallengeStats {
    pub(crate) issued: bool,
    pub(crate) flipped: bool,
    pub(crate) validation_rounds: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct AsgardPriorCompletionReview {
    pub(crate) window: usize,
    pub(crate) rows: Vec<AsgardContractRow>,
    pub(crate) decision: AsgardSupervisorDecision,
    pub(crate) reviewed_patch: Vec<u8>,
    pub(crate) evidence_fingerprints: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(crate) struct AsgardTaskContract {
    pub(crate) id: String,
    pub(crate) kind: String,
    pub(crate) text: String,
    pub(crate) adverse_condition: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct AsgardTaskContractChecklist {
    pub(crate) contracts: Vec<AsgardTaskContract>,
    /// Fully rendered `<task_contract_checklist ...>...</task_contract_checklist>`
    /// block, computed once, ready to insert verbatim as one assistant message.
    pub(crate) block: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(crate) struct AsgardContractRow {
    pub(crate) id: String,
    pub(crate) status: String,
    pub(crate) evidence: String,
    pub(crate) adverse_condition_evidence: Option<String>,
}

pub(crate) const ASGARD_INITIAL_ADVICE_MAX_ATTEMPTS: usize = 2;
pub(crate) const ASGARD_MIN_CANDIDATES: usize = 1;
pub(crate) const ASGARD_MAX_CANDIDATES: usize = 5;
pub(crate) const ASGARD_MIN_WINDOW_STEPS: usize = 1;
pub(crate) const ASGARD_MAX_WINDOW_STEPS: usize = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AsgardSupervisorInitialAdvice {
    pub(crate) advices: Vec<Option<String>>,
    pub(crate) next_window_steps: usize,
    pub(crate) state_summary: String,
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_asgard_trajectory_loop(
    cx: &ConnectionTo<Client>,
    sessions: &SessionStore,
    session_id: &str,
    llm: &Arc<dyn crate::llm_client::LlmBackend>,
    parent_registry: &Arc<crate::tools::ToolRegistry>,
    selected_model: &str,
    reasoning_effort: Option<&str>,
    service_tier: Option<&str>,
    structured_output: Option<&StructuredOutputRequest>,
    initial_messages: Vec<ChatMessage>,
    idle_timeout: IdleTimeouts,
    cancel: tokio_util::sync::CancellationToken,
    config: &crate::asgard::Config,
    context_length: Option<u32>,
    context_prefix_len: usize,
    initial_plan: Option<crate::plan::UpdatePlanArgs>,
) -> (
    crate::tool_loop::LoopOutcome,
    BTreeMap<String, crate::llm_client::TokenUsage>,
) {
    let mut usage_by_model = BTreeMap::new();
    if let Err(error) = crate::asgard::ensure_compatible_checkout(parent_registry.cwd()) {
        return (asgard_failure(error), usage_by_model);
    }
    if !(ASGARD_MIN_CANDIDATES..=ASGARD_MAX_CANDIDATES).contains(&config.candidate_models.len()) {
        return (
            asgard_failure(anyhow::anyhow!(
                "Asgard requires between {ASGARD_MIN_CANDIDATES} and \
                 {ASGARD_MAX_CANDIDATES} configured candidate models (got {})",
                config.candidate_models.len()
            )),
            usage_by_model,
        );
    }
    let mut repositories = Vec::with_capacity(config.candidate_models.len());
    for (index, model) in config.candidate_models.iter().enumerate() {
        match crate::asgard::create_candidate_repository(
            parent_registry.cwd(),
            &format!("{index}-{model}"),
        ) {
            Ok(repository) => repositories.push(repository),
            Err(error) => {
                for repository in &repositories {
                    crate::asgard::remove_candidate_repository(repository);
                }
                return (asgard_failure(error), usage_by_model);
            }
        }
    }
    let mut registries = Vec::with_capacity(repositories.len());
    for repository in &repositories {
        let Some(registry) = sessions
            .create_trajectory_registry(session_id, repository.session_cwd.clone())
            .await
        else {
            for repository in &repositories {
                crate::asgard::remove_candidate_repository(repository);
            }
            return (
                asgard_failure(anyhow::anyhow!("unknown Asgard parent session")),
                usage_by_model,
            );
        };
        registries.push(registry);
    }
    let mut common_messages = initial_messages;
    let original_task = asgard_original_task(&common_messages);
    let mut selected_trajectory_initial = common_messages.clone();
    let mut selected_trajectory_windows: Vec<Vec<ChatMessage>> = Vec::new();
    let mut canonical_ledger: Vec<(usize, AsgardExecutionLedger)> = Vec::new();
    // Winning lanes' full windows, kept so `view_tool_call` can expand handles
    // the dossier still cites from earlier windows. Only the winner is retained
    // — losing lanes' work is discarded with their checkouts. Measured at
    // ~0.7 MB for a whole task, so there is no eviction policy.
    let mut retained_windows: Vec<AsgardRetainedWindow> = Vec::new();
    let mut supervisor_history = AsgardSupervisorHistory::default();
    let mut prior_completion_review: Option<AsgardPriorCompletionReview> = None;
    let mut window_deltas_since_prior_review: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut common_patch = Vec::new();
    let mut aggregate_usage = crate::llm_client::TokenUsage::default();
    let mut selected_outcome = None;
    let live_output = AsgardLiveOutput::new(cx, session_id);
    let supervisor_model = config.supervisor_model.as_deref().unwrap_or(selected_model);
    let (task_contract_checklist, checklist_usage) = run_asgard_task_contract_extraction(
        llm,
        supervisor_model,
        idle_timeout,
        cancel.clone(),
        &original_task,
    )
    .await;
    if !task_contract_checklist.contracts.is_empty() {
        crate::trace_logging::append_trace_record(serde_json::json!({
            "type": "asgard_checklist",
            "contracts": &task_contract_checklist.contracts,
        }));
    }
    aggregate_usage.add(checklist_usage);
    trace_asgard_phase_usage(
        "task_contract",
        supervisor_model,
        None,
        None,
        checklist_usage,
    );
    usage_by_model
        .entry(supervisor_model.to_string())
        .or_default()
        .add(checklist_usage);
    let mut initial_advice = None;
    let mut initial_advice_error = None;
    for attempt in 1..=ASGARD_INITIAL_ADVICE_MAX_ATTEMPTS {
        let supervisor = run_asgard_initial_advice(
            llm,
            supervisor_model,
            idle_timeout,
            cancel.clone(),
            config.candidate_models.len(),
            &original_task,
            &live_output,
        )
        .await;
        trace_asgard_phase_usage("initial_advice", supervisor_model, None, None, supervisor.1);
        aggregate_usage.add(supervisor.1);
        usage_by_model
            .entry(supervisor_model.to_string())
            .or_default()
            .add(supervisor.1);
        match supervisor.0 {
            Ok(value) => {
                initial_advice = Some(value);
                break;
            }
            Err(error)
                if attempt < ASGARD_INITIAL_ADVICE_MAX_ATTEMPTS && !cancel.is_cancelled() =>
            {
                tracing::warn!(
                    attempt,
                    max_attempts = ASGARD_INITIAL_ADVICE_MAX_ATTEMPTS,
                    "retrying failed initial Asgard supervisor advice: {error:#}"
                );
            }
            Err(error) => {
                initial_advice_error = Some(error);
                break;
            }
        }
    }
    let initial_advice = match initial_advice {
        Some(value) => value,
        None => {
            let error = initial_advice_error
                .unwrap_or_else(|| anyhow::anyhow!("supervisor produced no initial advice"));
            tracing::warn!(
                "Asgard initial supervisor exhausted validation retries; \
                 continuing with one short candidate window: {error:#}"
            );
            crate::trace_logging::append_trace_record(serde_json::json!({
                "type": "asgard_supervisor_fallback",
                "call": "initial_advice",
                "error": format!("{error:#}"),
                "next_window_steps": ASGARD_MIN_WINDOW_STEPS,
            }));
            AsgardSupervisorInitialAdvice {
                advices: vec![Some(
                    "The supervisor could not produce a valid initial routing decision. \
                     Inspect the task and repository, identify the most consequential \
                     next action, and make concrete progress before the next review."
                        .to_string(),
                )],
                next_window_steps: ASGARD_MIN_WINDOW_STEPS,
                state_summary: "Initial supervisor routing failed validation; using a traced one-lane recovery window."
                    .to_string(),
            }
        }
    };
    let mut current_candidate_count = initial_advice.advices.len();
    let mut current_window_steps = initial_advice.next_window_steps;
    let mut consecutive_winner_failures = 0usize;
    let mut next_advices: Option<Vec<Option<String>>> = Some(initial_advice.advices.clone());
    send_thought(
        cx,
        session_id,
        &format!(
            "[Asgard initial strategies: next window = {} candidates × {} steps]\n{}\n{}\n",
            current_candidate_count,
            current_window_steps,
            initial_advice.state_summary,
            render_asgard_lane_advices(&initial_advice.advices)
        ),
    );
    let mut canonical_plan = initial_plan;
    for window in 1usize.. {
        let verified_at_window_start = canonical_ledger
            .last()
            .and_then(|(_, ledger)| render_asgard_verified_at_window_start(ledger));
        send_thought(
            cx,
            session_id,
            &format!(
                "\n[Asgard window {window}: {} candidates × {} steps]\n",
                current_candidate_count, current_window_steps
            ),
        );
        let mut futures = Vec::with_capacity(repositories.len());
        for (index, ((model, repository), registry)) in config
            .candidate_models
            .iter()
            .zip(&repositories)
            .zip(&registries)
            .take(current_candidate_count)
            .enumerate()
        {
            let mut messages = common_messages.clone();
            rewrite_asgard_cwd(
                &mut messages,
                parent_registry.cwd(),
                &repository.session_cwd,
            );
            let assigned_advice = next_advices
                .as_ref()
                .and_then(|advices| advices.get(index))
                .cloned()
                .flatten();
            let trajectory_message_start = messages.len();
            if let Some(advice) = &assigned_advice {
                messages.push(asgard_advice_message(
                    index,
                    advice,
                    verified_at_window_start.as_deref(),
                ));
            }
            let model = model.clone();
            let registry = registry.clone();
            let repository_root = repository.root.clone();
            let base_commit = repository.base_commit.clone();
            let selected_patch = common_patch.clone();
            let cancel = cancel.clone();
            let inherited_plan = canonical_plan.clone();
            let assessment_original_task = original_task.clone();
            let window_steps = current_window_steps;
            let candidate_output = live_output.clone();
            futures.push(async move {
                let sinks =
                    AsgardStreamSinks::new(&candidate_output, &format!("Candidate {}", index + 1));
                let spawned = crate::tool_loop::SpawnedCx::new(cx);
                let outcome = crate::tool_loop::run(
                    llm,
                    &registry,
                    &model,
                    reasoning_effort,
                    service_tier,
                    structured_output,
                    messages,
                    window_steps,
                    idle_timeout,
                    cancel.clone(),
                    sinks.text,
                    sinks.thought,
                    spawned,
                    session_id.to_string(),
                    sessions.clone(),
                    assessment_original_task.clone(),
                    crate::tool_loop::NotificationMode::Silent,
                    0,
                    None,
                    None,
                    true,
                    context_length,
                    context_prefix_len,
                    inherited_plan,
                )
                .await;
                let patches = crate::asgard::capture_patch(&repository_root, &base_commit)
                    .and_then(|patch| {
                        crate::asgard::capture_patch_since(
                            &repository_root,
                            &base_commit,
                            &selected_patch,
                        )
                        .map(|delta_patch| (patch, delta_patch))
                    });
                let window_messages = asgard_take_window_messages(
                    &outcome.continuation_messages,
                    trajectory_message_start,
                );
                let window_ledger =
                    asgard_extract_execution_ledger(window, index, &window_messages);
                let (supervisor_window_messages, raw_bytes, packed_bytes) =
                    asgard_deterministic_candidate_handoff(
                        window,
                        index,
                        &window_messages,
                        outcome.current_plan.as_ref(),
                    );
                let supervisor_bytes =
                    render_asgard_dossier_messages(&supervisor_window_messages).len();
                if std::env::var_os("ASGARD_CAPTURE_WINDOW_SUMMARIES").is_some() {
                    tracing::info!(
                        lane = index + 1,
                        raw_window = %render_asgard_dossier_messages(&window_messages),
                        candidate_handoff = %render_asgard_dossier_messages(&supervisor_window_messages),
                        "captured Asgard candidate window handoff for review"
                    );
                }
                tracing::info!(
                    lane = index + 1,
                    model,
                    raw_bytes,
                    packed_bytes,
                    supervisor_bytes,
                    "rendered compact deterministic Asgard candidate handoff"
                );
                crate::trace_logging::append_trace_record(serde_json::json!({
                    "type": "asgard_candidate_handoff",
                    "window": window,
                    "lane": index,
                    "mode": "compact_deterministic",
                    "raw_bytes": raw_bytes,
                    "packed_bytes": packed_bytes,
                    "supervisor_bytes": supervisor_bytes,
                }));
                (
                    index,
                    model,
                    outcome,
                    patches,
                    window_messages,
                    supervisor_window_messages,
                    window_ledger,
                )
            });
        }
        let mut candidates = Vec::with_capacity(current_candidate_count);
        for (
            index,
            model,
            outcome,
            patches,
            window_messages,
            supervisor_window_messages,
            window_ledger,
        ) in futures::future::join_all(futures).await
        {
            aggregate_usage.add(outcome.usage);
            usage_by_model
                .entry(model.clone())
                .or_default()
                .add(outcome.usage);
            match patches {
                Ok((patch, delta_patch)) => candidates.push(AsgardCandidate {
                    index,
                    model,
                    outcome,
                    patch,
                    delta_patch,
                    window_messages,
                    supervisor_window_messages,
                    window_ledger,
                }),
                Err(error) => {
                    cleanup_asgard_repositories(&repositories);
                    return (asgard_failure(error), usage_by_model);
                }
            }
        }
        let supervisor_audit_definitions = match registries.first() {
            Some(registry) => {
                asgard_audit_tool_definitions(registry.tool_definitions().await, candidates.len())
            }
            None => Vec::new(),
        };
        let supervisor = run_asgard_supervisor(
            llm,
            supervisor_model,
            idle_timeout,
            cancel.clone(),
            window,
            &candidates,
            &repositories[..current_candidate_count],
            &registries[..current_candidate_count],
            &supervisor_audit_definitions,
            config.candidate_models.len(),
            &task_contract_checklist,
            &original_task,
            &selected_trajectory_initial,
            &selected_trajectory_windows,
            &supervisor_history,
            &retained_windows,
            &live_output,
        )
        .await;
        let supervisor_usage = supervisor.1;
        trace_asgard_phase_usage(
            "routing_supervisor",
            supervisor_model,
            Some(window),
            None,
            supervisor_usage,
        );
        aggregate_usage.add(supervisor_usage);
        usage_by_model
            .entry(supervisor_model.to_string())
            .or_default()
            .add(supervisor_usage);
        let mut decision = match supervisor.0 {
            Ok(decision) => decision,
            Err(error) => {
                tracing::warn!(
                    window,
                    model = supervisor_model,
                    "Asgard supervisor exhausted validation retries; continuing with a traced one-lane fallback: {error:#}"
                );
                let winner = candidates
                    .iter()
                    .find(|candidate| {
                        !matches!(
                            candidate.outcome.stop,
                            crate::tool_loop::LoopStop::Failed(_)
                        )
                    })
                    .map_or(0, |candidate| candidate.index);
                crate::trace_logging::append_trace_record(serde_json::json!({
                    "type": "asgard_supervisor_fallback",
                    "call": "supervisor",
                    "window": window,
                    "error": format!("{error:#}"),
                    "winner": winner,
                    "next_window_steps": ASGARD_MIN_WINDOW_STEPS,
                }));
                AsgardSupervisorDecision {
                    winner,
                    complete: false,
                    advices: vec![Some(
                        "The supervisor could not produce a valid routing decision. \
                         Reassess the current implementation against the task, fix the \
                         clearest remaining defect, and report focused verification."
                            .to_string(),
                    )],
                    next_window_steps: Some(ASGARD_MIN_WINDOW_STEPS),
                    state_summary: format!(
                        "Window {window} supervisor routing failed validation; selected the first non-failed lane and scheduled a one-lane recovery window."
                    ),
                    contracts: None,
                }
            }
        };
        let mut completed_rejection_review: Option<AsgardSupervisorDecision> = None;
        if decision.complete
            && let Some(prior_review) = prior_completion_review.as_ref()
            && let Some(candidate) = candidates
                .iter()
                .find(|candidate| candidate.index == decision.winner)
            && let Some(repository) = repositories.get(decision.winner)
        {
            match asgard_completion_review_delta(
                prior_review,
                candidate,
                repository,
                &canonical_ledger,
            ) {
                Ok(delta) => {
                    let material = delta.is_material();
                    crate::trace_logging::append_trace_record(serde_json::json!({
                        "type": "asgard_completion_review_gate",
                        "window": window,
                        "prior_review_window": prior_review.window,
                        "selected_lane": decision.winner,
                        "material": material,
                        "net_patch_bytes": delta.patch.len(),
                        "production_bytes": delta.production_bytes,
                        "test_bytes": delta.test_bytes,
                        "new_evidence_count": delta.new_evidence_count,
                        "has_read_only_observation": delta.has_read_only_observation,
                    }));
                    if !material {
                        let mut carried = prior_review.decision.clone();
                        carried.winner = decision.winner;
                        carried.complete = false;
                        carried.contracts = Some(prior_review.rows.clone());
                        carried.state_summary = format!(
                            "Completion nomination in window {window} was not re-reviewed: no net \
                             production or test change, new execution evidence, or new read-only \
                             observation was present since the rejected completion review in window {}. {}",
                            prior_review.window, prior_review.decision.state_summary,
                        );
                        send_thought(
                            cx,
                            session_id,
                            &format!(
                                "[Asgard deferred repeated completion review: no material delta since window {}]\n",
                                prior_review.window
                            ),
                        );
                        decision = carried;
                    }
                }
                Err(error) => {
                    tracing::warn!(
                        window,
                        prior_review_window = prior_review.window,
                        "failed to evaluate Asgard completion-review delta; reviewing conservatively: {error:#}"
                    );
                }
            }
        }
        if decision.complete {
            send_thought(
                cx,
                session_id,
                &format!(
                    "[Asgard reviewing proposed completion of lane {} in isolation]\n",
                    decision.winner + 1
                ),
            );
            let completion_review = run_asgard_completion_review(
                llm,
                cancel.clone(),
                AsgardCompletionReviewContext {
                    model: supervisor_model,
                    idle_timeout,
                    window,
                    selected_lane: decision.winner,
                    candidates: &candidates,
                    registries: &registries[..current_candidate_count],
                    audit_definitions: &supervisor_audit_definitions,
                    max_candidate_count: config.candidate_models.len(),
                    task_contract_checklist: &task_contract_checklist,
                    canonical_ledger: &canonical_ledger,
                    retained_windows: &retained_windows,
                    original_task: &original_task,
                    selected_trajectory_initial: &selected_trajectory_initial,
                    selected_trajectory_windows: &selected_trajectory_windows,
                    supervisor_history: &supervisor_history,
                    prior_review: prior_completion_review.as_ref(),
                    window_deltas_since_prior_review: &window_deltas_since_prior_review,
                    live_output: &live_output,
                },
            )
            .await;
            trace_asgard_phase_usage(
                "completion_review",
                supervisor_model,
                Some(window),
                Some(decision.winner),
                completion_review.1,
            );
            aggregate_usage.add(completion_review.1);
            usage_by_model
                .entry(supervisor_model.to_string())
                .or_default()
                .add(completion_review.1);
            decision = match completion_review.0 {
                Ok(review) => {
                    if !review.complete {
                        completed_rejection_review = Some(review.clone());
                    }
                    review
                }
                Err(error) => {
                    // Failing closed used to abort the whole Asgard run, which
                    // wastes the remaining task budget and grades the very
                    // endpoint the reviewer refused to endorse. Degrade to an
                    // incomplete decision instead: keep the provisional winner
                    // and spend a short verification window producing the
                    // evidence the review could not obtain.
                    tracing::warn!(
                        window,
                        model = supervisor_model,
                        "Asgard isolated completion review produced no valid decision; continuing incomplete: {error:#}"
                    );
                    let advice = format!(
                        "The completion review could not validate the proposed endpoint \
                         ({error:#}). Produce concrete execution evidence for every task \
                         contract that lacks it: run the narrowest checks that exercise \
                         each contract's stated scenario, report each command and its \
                         output verbatim, and fix any contract the evidence contradicts."
                    );
                    AsgardSupervisorDecision {
                        winner: decision.winner,
                        complete: false,
                        advices: vec![Some(advice)],
                        next_window_steps: Some(ASGARD_MIN_WINDOW_STEPS),
                        state_summary: format!(
                            "Completion review failed to produce a valid decision in \
                             window {window}; continuing with a verification window."
                        ),
                        contracts: None,
                    }
                }
            };
        }
        // Cumulative per-model usage snapshot every window: the last record in
        // the trace prices the run even when the process is killed at a
        // deadline before any final summary could be written.
        crate::trace_logging::append_trace_record(serde_json::json!({
            "type": "asgard_usage_by_model",
            "window": window,
            "usage_by_model": usage_by_model
                .iter()
                .map(|(model, usage)| {
                    (
                        model.clone(),
                        serde_json::json!({
                            "input": usage.input_tokens,
                            "output": usage.output_tokens,
                            "thought": usage.thought_tokens,
                            "cachedRead": usage.cached_read_tokens,
                            "cachedWrite": usage.cached_write_tokens,
                        }),
                    )
                })
                .collect::<serde_json::Map<_, _>>(),
        }));
        crate::trace_logging::append_trace_record(serde_json::json!({
            "type": "asgard_window",
            "window": window,
            "candidate_count": candidates.len(),
            "window_steps": current_window_steps,
        }));
        let winner_index = decision.winner;
        let supervisor_complete = decision.complete;
        let supervisor_completion_summary = decision.state_summary.clone();
        next_advices = Some(decision.advices.clone());
        if !supervisor_complete && let Some(advices) = &next_advices {
            current_candidate_count = advices.len();
            current_window_steps = decision
                .next_window_steps
                .expect("incomplete Asgard decision has next_window_steps");
            send_thought(
                cx,
                session_id,
                &format!(
                    "[Asgard next-window strategies: next window = {} candidates × {} steps]\n{}\n",
                    current_candidate_count,
                    current_window_steps,
                    render_asgard_lane_advices(advices)
                ),
            );
        }
        let Some(mut winner) = candidates.into_iter().find(|c| c.index == winner_index) else {
            cleanup_asgard_repositories(&repositories);
            return (
                asgard_failure(anyhow::anyhow!(
                    "Asgard supervisor selected an unknown lane"
                )),
                usage_by_model,
            );
        };
        send_thought(
            cx,
            session_id,
            &format!(
                "[Asgard selected lane {}: {}]\n",
                winner.index + 1,
                winner.model
            ),
        );
        let selected_plan = winner.outcome.current_plan.clone();
        if let Some(plan) = selected_plan.as_ref() {
            let notification = SessionNotification::new(
                session_id.to_string(),
                SessionUpdate::Plan(plan.to_acp()),
            );
            if let Err(error) = cx.send_notification(notification) {
                tracing::warn!("failed to publish selected Asgard plan: {error}");
            }
        }
        if selected_plan.is_some() {
            canonical_plan = selected_plan.clone();
        }
        winner.outcome.current_plan = canonical_plan.clone();
        common_patch = winner.patch.clone();
        common_messages = winner.outcome.continuation_messages.clone();
        // Store the canonical history in terms of the live checkout. Each
        // candidate window rewrites that canonical path to its own clone.
        // Without this normalization, window N+1 inherited window N's
        // now-stale winning clone path.
        rewrite_asgard_cwd(
            &mut common_messages,
            &repositories[winner.index].session_cwd,
            parent_registry.cwd(),
        );
        // Candidate model history remains exact in `common_messages`, but the
        // supervisor should keep seeing the assessed representation after this
        // window becomes selected. Otherwise an omitted noisy result disappears
        // for one decision and is reintroduced in full at the next window.
        let mut selected_window = winner.supervisor_window_messages.clone();
        rewrite_asgard_cwd(
            &mut selected_window,
            &repositories[winner.index].session_cwd,
            parent_registry.cwd(),
        );
        selected_trajectory_windows.push(selected_window);
        canonical_ledger.push((window, winner.window_ledger.clone()));
        retained_windows.push(AsgardRetainedWindow {
            window,
            lane: winner.index,
            messages: winner.window_messages.clone(),
        });
        supervisor_history.push(window, &decision);
        if let Some(review) = completed_rejection_review {
            let rows = review.contracts.clone().unwrap_or_default();
            let evidence_fingerprints =
                asgard_review_evidence_fingerprints(&canonical_ledger, &winner.window_ledger);
            prior_completion_review = Some(AsgardPriorCompletionReview {
                window,
                rows,
                decision: review,
                reviewed_patch: winner.patch.clone(),
                evidence_fingerprints,
            });
            window_deltas_since_prior_review.clear();
        } else if prior_completion_review.is_some() {
            window_deltas_since_prior_review.push((window, winner.delta_patch.clone()));
        }
        if crate::tokens::approximate_tokens_messages(&common_messages)
            > crate::context_manager::context_budget(context_length)
        {
            let dynamic = common_messages[context_prefix_len.min(common_messages.len())..].to_vec();
            match crate::context_manager::compact_history(
                llm.as_ref(),
                &winner.model,
                &dynamic,
                canonical_plan.as_ref(),
                context_length,
                idle_timeout,
                cancel.clone(),
            )
            .await
            {
                Ok(compaction) => {
                    trace_asgard_phase_usage(
                        "selected_compaction",
                        &winner.model,
                        Some(window),
                        Some(winner.index),
                        compaction.usage,
                    );
                    aggregate_usage.add(compaction.usage);
                    usage_by_model
                        .entry(winner.model.clone())
                        .or_default()
                        .add(compaction.usage);
                    common_messages.truncate(context_prefix_len.min(common_messages.len()));
                    common_messages.extend(compaction.checkpoint_messages);
                    winner.outcome.continuation_messages = common_messages.clone();
                    winner.outcome.compaction_checkpoint =
                        Some(crate::session::CompactionCheckpoint {
                            messages: common_messages
                                [context_prefix_len.min(common_messages.len())..]
                                .to_vec(),
                            current_plan: canonical_plan.clone(),
                        });
                    selected_trajectory_initial = common_messages.clone();
                    selected_trajectory_windows.clear();
                    supervisor_history.checkpoint_selected_windows();
                    tracing::info!(
                        window,
                        before_tokens = compaction.before_tokens,
                        after_tokens = compaction.after_tokens,
                        "compacted selected Asgard trajectory"
                    );
                }
                Err(error) => {
                    tracing::warn!(window, "selected Asgard compaction failed: {error:#}");
                }
            }
        }
        winner.outcome.continuation_messages = common_messages.clone();
        if supervisor_complete {
            send_thought(
                cx,
                session_id,
                "[Asgard supervisor judged the selected endpoint complete]\n",
            );
            winner.outcome.response = supervisor_completion_summary;
            winner.outcome.stop = crate::tool_loop::LoopStop::Completed { had_text: true };
        }
        if matches!(winner.outcome.stop, crate::tool_loop::LoopStop::Failed(_)) {
            consecutive_winner_failures += 1;
            tracing::warn!(
                window,
                consecutive_winner_failures,
                "Asgard winner lane failed; resurrecting from the accepted state next window"
            );
            send_thought(
                cx,
                session_id,
                "[Asgard: selected lane failed this window; resurrecting from the accepted state]\n",
            );
        } else {
            consecutive_winner_failures = 0;
        }
        let finished = asgard_should_finish(supervisor_complete, &winner.outcome.stop)
            || consecutive_winner_failures >= 3;
        selected_outcome = Some(winner.outcome);
        if finished {
            break;
        }
        send_thought(
            cx,
            session_id,
            "[Asgard synchronizing selected repository state]\n",
        );
        let sync_started = Instant::now();
        match crate::asgard::synchronize_candidate_repositories(&repositories, winner_index) {
            Ok(stats) => crate::trace_logging::append_trace_record(serde_json::json!({
                "type": "asgard_repository_sync",
                "window": window,
                "elapsed_millis": sync_started.elapsed().as_millis(),
                "destinations": stats.destinations,
                "files_copied": stats.files_copied,
                "bytes_copied": stats.bytes_copied,
                "entries_removed": stats.entries_removed,
                "files_unchanged": stats.files_unchanged,
                "metadata_updated": stats.metadata_updated,
            })),
            Err(error) => {
                cleanup_asgard_repositories(&repositories);
                return (asgard_failure(error), usage_by_model);
            }
        }
    }
    let apply_result = crate::asgard::apply_selected_patch(parent_registry.cwd(), &common_patch);
    cleanup_asgard_repositories(&repositories);
    if let Err(error) = apply_result {
        return (asgard_failure(error), usage_by_model);
    }
    let mut outcome = selected_outcome
        .unwrap_or_else(|| asgard_failure(anyhow::anyhow!("Asgard produced no candidate windows")));
    outcome.usage = aggregate_usage;
    (outcome, usage_by_model)
}

pub(crate) fn asgard_should_finish(
    supervisor_complete: bool,
    stop: &crate::tool_loop::LoopStop,
) -> bool {
    // A failed winner lane does not end the run: every lane restarts from the
    // winner snapshot next window, which resurrects it. Only supervisor
    // completion or an explicit cancellation finishes Asgard; persistent
    // failures are bounded by the consecutive-failure circuit breaker at the
    // call site.
    supervisor_complete || matches!(stop, crate::tool_loop::LoopStop::Cancelled)
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_asgard_initial_advice(
    llm: &Arc<dyn crate::llm_client::LlmBackend>,
    model: &str,
    idle_timeout: IdleTimeouts,
    cancel: tokio_util::sync::CancellationToken,
    candidate_count: usize,
    original_task: &str,
    live_output: &AsgardLiveOutput,
) -> (
    anyhow::Result<AsgardSupervisorInitialAdvice>,
    crate::llm_client::TokenUsage,
) {
    let messages = asgard_initial_advice_messages(original_task, candidate_count);
    run_asgard_initial_advice_tool_steps(
        llm.as_ref(),
        messages,
        AsgardSupervisorToolContext {
            model,
            candidate_count,
            max_candidate_count: candidate_count,
            idle_timeout,
            audit: None,
            required_winner: None,
            checklist_ids: &[],
            carry_forward_allowed: false,
        },
        cancel,
        Some(AsgardStreamSinks::new(live_output, "Supervisor")),
    )
    .await
}

pub(crate) fn default_asgard_diff_presentation(
    candidates: &[AsgardCandidate],
) -> AsgardDiffPresentation {
    AsgardDiffPresentation {
        views: candidates
            .iter()
            .map(|candidate| AsgardCandidateDiffView {
                candidate_index: candidate.index,
                base: AsgardCandidateDiffBase::LastSelectedDecision,
                patch: candidate.delta_patch.clone(),
            })
            .collect(),
        anchor_lane: None,
        baseline_sum_bytes: candidates.iter().fold(0usize, |sum, candidate| {
            sum.saturating_add(candidate.delta_patch.len())
        }),
        best_candidate_sum_bytes: None,
    }
}

pub(crate) fn select_asgard_diff_anchor(
    baseline_sum_bytes: usize,
    candidate_sums: &[(usize, usize)],
) -> Option<(usize, usize)> {
    if baseline_sum_bytes == 0 || candidate_sums.len() < 2 {
        return None;
    }
    let &(lane, sum) = candidate_sums
        .iter()
        .min_by_key(|(lane, sum)| (*sum, *lane))?;
    ((sum as u128) * 10 < (baseline_sum_bytes as u128) * 4).then_some((lane, sum))
}

pub(crate) fn build_asgard_diff_presentation(
    mut inputs: Vec<AsgardDiffCandidateInput>,
) -> anyhow::Result<AsgardDiffPresentation> {
    inputs.sort_by_key(|candidate| candidate.index);
    let baseline_sum_bytes = inputs.iter().fold(0usize, |sum, candidate| {
        sum.saturating_add(candidate.delta_patch.len())
    });
    if inputs.len() < 2 || baseline_sum_bytes == 0 {
        return Ok(AsgardDiffPresentation {
            views: inputs
                .into_iter()
                .map(|candidate| AsgardCandidateDiffView {
                    candidate_index: candidate.index,
                    base: AsgardCandidateDiffBase::LastSelectedDecision,
                    patch: candidate.delta_patch,
                })
                .collect(),
            anchor_lane: None,
            baseline_sum_bytes,
            best_candidate_sum_bytes: None,
        });
    }

    let mut candidate_sums = Vec::with_capacity(inputs.len());
    let mut best_row: Option<AsgardCandidateDiffRow> = None;
    for anchor in &inputs {
        let mut sum = 0usize;
        let mut row = Vec::with_capacity(inputs.len());
        for candidate in &inputs {
            let patch = if candidate.index == anchor.index {
                Vec::new()
            } else {
                crate::asgard::capture_patch_since(
                    &candidate.repository_root,
                    &candidate.base_commit,
                    &anchor.patch,
                )
                .with_context(|| {
                    format!(
                        "capture Asgard diff from current-window lane {} to lane {}",
                        anchor.index, candidate.index
                    )
                })?
            };
            sum = sum.saturating_add(patch.len());
            row.push((candidate.index, patch));
        }
        candidate_sums.push((anchor.index, sum));
        if best_row
            .as_ref()
            .is_none_or(|best| (sum, anchor.index) < (best.sum_bytes, best.anchor_lane))
        {
            best_row = Some(AsgardCandidateDiffRow {
                anchor_lane: anchor.index,
                sum_bytes: sum,
                patches: row,
            });
        }
    }

    let Some((anchor_lane, best_candidate_sum_bytes)) =
        select_asgard_diff_anchor(baseline_sum_bytes, &candidate_sums)
    else {
        return Ok(AsgardDiffPresentation {
            views: inputs
                .into_iter()
                .map(|candidate| AsgardCandidateDiffView {
                    candidate_index: candidate.index,
                    base: AsgardCandidateDiffBase::LastSelectedDecision,
                    patch: candidate.delta_patch,
                })
                .collect(),
            anchor_lane: None,
            baseline_sum_bytes,
            best_candidate_sum_bytes: candidate_sums
                .iter()
                .min_by_key(|(lane, sum)| (*sum, *lane))
                .map(|(_, sum)| *sum),
        });
    };
    let best_row = best_row.expect("candidate sums have a best diff row");
    let anchor = inputs
        .iter()
        .find(|candidate| candidate.index == anchor_lane)
        .expect("selected Asgard diff anchor exists");
    let mut views = Vec::with_capacity(inputs.len());
    views.push(AsgardCandidateDiffView {
        candidate_index: anchor_lane,
        base: AsgardCandidateDiffBase::LastSelectedDecision,
        patch: anchor.delta_patch.clone(),
    });
    views.extend(
        best_row
            .patches
            .into_iter()
            .filter(|(candidate_index, _)| *candidate_index != anchor_lane)
            .map(|(candidate_index, patch)| AsgardCandidateDiffView {
                candidate_index,
                base: AsgardCandidateDiffBase::CurrentWindowLane(anchor_lane),
                patch,
            }),
    );
    Ok(AsgardDiffPresentation {
        views,
        anchor_lane: Some(anchor_lane),
        baseline_sum_bytes,
        best_candidate_sum_bytes: Some(best_candidate_sum_bytes),
    })
}

pub(crate) fn asgard_diff_baseline_trace_record(
    window: usize,
    presentation: &AsgardDiffPresentation,
    fallback: Option<(&str, &str)>,
) -> serde_json::Value {
    serde_json::json!({
        "type": "asgard_diff_baseline",
        "window": window,
        "mode": if presentation.anchor_lane.is_some() {
            "current_window_candidate"
        } else {
            "last_selected_decision"
        },
        "anchor_lane": presentation.anchor_lane,
        "baseline_sum_bytes": presentation.baseline_sum_bytes,
        "best_candidate_sum_bytes": presentation.best_candidate_sum_bytes,
        "threshold_numerator": 2,
        "threshold_denominator": 5,
        "fallback_reason": fallback.map(|(reason, _)| reason),
        "fallback_error": fallback.map(|(_, error)| error),
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_asgard_supervisor(
    llm: &Arc<dyn crate::llm_client::LlmBackend>,
    model: &str,
    idle_timeout: IdleTimeouts,
    cancel: tokio_util::sync::CancellationToken,
    window: usize,
    candidates: &[AsgardCandidate],
    repositories: &[crate::asgard::CandidateRepository],
    registries: &[Arc<crate::tools::ToolRegistry>],
    audit_definitions: &[ToolDefinition],
    max_candidate_count: usize,
    task_contract_checklist: &AsgardTaskContractChecklist,
    original_task: &str,
    selected_trajectory_initial: &[ChatMessage],
    selected_trajectory_windows: &[Vec<ChatMessage>],
    supervisor_history: &AsgardSupervisorHistory,
    retained_windows: &[AsgardRetainedWindow],
    live_output: &AsgardLiveOutput,
) -> (
    anyhow::Result<AsgardSupervisorDecision>,
    crate::llm_client::TokenUsage,
) {
    debug_assert_eq!(candidates.len(), registries.len());
    debug_assert_eq!(candidates.len(), repositories.len());
    let audit = AsgardAuditContext {
        registries,
        candidates,
        definitions: audit_definitions.to_vec(),
        allowed_lane: None,
        retained_windows,
        window,
    };
    let diff_inputs = candidates
        .iter()
        .map(|candidate| {
            let repository = repositories.get(candidate.index).ok_or_else(|| {
                anyhow::anyhow!(
                    "Asgard lane {} has no candidate repository",
                    candidate.index
                )
            })?;
            Ok(AsgardDiffCandidateInput {
                index: candidate.index,
                patch: candidate.patch.clone(),
                delta_patch: candidate.delta_patch.clone(),
                repository_root: repository.root.clone(),
                base_commit: repository.base_commit.clone(),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>();
    let mut diff_fallback: Option<(&'static str, String)> = None;
    let diff_presentation = match diff_inputs {
        Ok(inputs) => {
            match tokio::task::spawn_blocking(move || build_asgard_diff_presentation(inputs)).await
            {
                Ok(Ok(presentation)) => presentation,
                Ok(Err(error)) => {
                    tracing::warn!(
                        window,
                        "failed to build candidate-relative Asgard diffs; using the last selected decision as every lane's baseline: {error:#}"
                    );
                    diff_fallback = Some(("pairwise_diff_error", format!("{error:#}")));
                    default_asgard_diff_presentation(candidates)
                }
                Err(error) => {
                    tracing::warn!(
                        window,
                        "candidate-relative Asgard diff task failed; using the last selected decision as every lane's baseline: {error}"
                    );
                    diff_fallback = Some(("pairwise_diff_task_error", error.to_string()));
                    default_asgard_diff_presentation(candidates)
                }
            }
        }
        Err(error) => {
            tracing::warn!(
                window,
                "candidate-relative Asgard diff inputs were incomplete; using the last selected decision as every lane's baseline: {error:#}"
            );
            diff_fallback = Some(("pairwise_diff_input_error", format!("{error:#}")));
            default_asgard_diff_presentation(candidates)
        }
    };
    crate::trace_logging::append_trace_record(asgard_diff_baseline_trace_record(
        window,
        &diff_presentation,
        diff_fallback
            .as_ref()
            .map(|(reason, error)| (*reason, error.as_str())),
    ));
    let mut candidate_trajectories = format!("<candidate_trajectories window=\"{window}\">\n");
    if let Some(anchor_lane) = diff_presentation.anchor_lane {
        candidate_trajectories.push_str(&format!(
            "<candidate_diff_baseline mode=\"current_window_candidate\" anchor_lane=\"{anchor_lane}\">\n\
             Lane {anchor_lane} is shown first as a diff-compression anchor only. Its diff is against the last selected decision; every other lane's diff is against lane {anchor_lane}. All lanes still started independently from the last selected decision.\n\
             </candidate_diff_baseline>\n"
        ));
    }
    for diff_view in &diff_presentation.views {
        let Some(candidate) = candidates
            .iter()
            .find(|candidate| candidate.index == diff_view.candidate_index)
        else {
            return (
                Err(anyhow::anyhow!(
                    "Asgard diff presentation references unknown lane {}",
                    diff_view.candidate_index
                )),
                crate::llm_client::TokenUsage::default(),
            );
        };
        let Some(registry) = registries.get(candidate.index) else {
            return (
                Err(anyhow::anyhow!(
                    "Asgard lane {} has no checkout registry",
                    candidate.index
                )),
                crate::llm_client::TokenUsage::default(),
            );
        };
        match render_asgard_candidate_trajectory(
            candidate,
            registry.cwd(),
            &diff_view.base.label(),
            &diff_view.patch,
        ) {
            Ok(trajectory) => candidate_trajectories.push_str(&trajectory),
            Err(error) => {
                return (Err(error), crate::llm_client::TokenUsage::default());
            }
        }
    }
    candidate_trajectories.push_str("</candidate_trajectories>");
    let messages = asgard_supervisor_messages(
        original_task,
        selected_trajectory_initial,
        selected_trajectory_windows,
        supervisor_history,
        AsgardCandidateCounts {
            current: candidates.len(),
            max: max_candidate_count,
        },
        task_contract_checklist,
        candidate_trajectories,
    );
    tracing::info!(
        window,
        selected_initial_bytes = render_asgard_dossier_messages(selected_trajectory_initial).len(),
        selected_windows_bytes = selected_trajectory_windows
            .iter()
            .map(|messages| render_asgard_dossier_messages(messages).len())
            .sum::<usize>(),
        candidate_trajectories_bytes = messages
            .last()
            .map(asgard_message_text)
            .map_or(0, |text| text.len()),
        "assembled Asgard supervisor dossier"
    );
    let (decision, usage) = run_asgard_supervisor_tool_steps(
        llm.as_ref(),
        messages,
        AsgardSupervisorToolContext {
            model,
            candidate_count: candidates.len(),
            max_candidate_count,
            idle_timeout,
            audit: Some(audit),
            required_winner: None,
            checklist_ids: &[],
            carry_forward_allowed: false,
        },
        cancel,
        Some(AsgardStreamSinks::new(live_output, "Supervisor")),
    )
    .await;
    (decision.map(|(decision, _stats)| decision), usage)
}

pub(crate) struct AsgardCompletionReviewContext<'a> {
    pub(crate) model: &'a str,
    pub(crate) idle_timeout: IdleTimeouts,
    pub(crate) window: usize,
    pub(crate) selected_lane: usize,
    pub(crate) candidates: &'a [AsgardCandidate],
    pub(crate) registries: &'a [Arc<crate::tools::ToolRegistry>],
    pub(crate) audit_definitions: &'a [ToolDefinition],
    pub(crate) max_candidate_count: usize,
    pub(crate) task_contract_checklist: &'a AsgardTaskContractChecklist,
    pub(crate) canonical_ledger: &'a [(usize, AsgardExecutionLedger)],
    pub(crate) retained_windows: &'a [AsgardRetainedWindow],
    pub(crate) original_task: &'a str,
    pub(crate) selected_trajectory_initial: &'a [ChatMessage],
    pub(crate) selected_trajectory_windows: &'a [Vec<ChatMessage>],
    pub(crate) supervisor_history: &'a AsgardSupervisorHistory,
    pub(crate) prior_review: Option<&'a AsgardPriorCompletionReview>,
    pub(crate) window_deltas_since_prior_review: &'a [(usize, Vec<u8>)],
    pub(crate) live_output: &'a AsgardLiveOutput,
}

pub(crate) async fn run_asgard_completion_review(
    llm: &Arc<dyn crate::llm_client::LlmBackend>,
    cancel: tokio_util::sync::CancellationToken,
    context: AsgardCompletionReviewContext<'_>,
) -> (
    anyhow::Result<AsgardSupervisorDecision>,
    crate::llm_client::TokenUsage,
) {
    let AsgardCompletionReviewContext {
        model,
        idle_timeout,
        window,
        selected_lane,
        candidates,
        registries,
        audit_definitions,
        max_candidate_count,
        task_contract_checklist,
        canonical_ledger,
        retained_windows,
        original_task,
        selected_trajectory_initial,
        selected_trajectory_windows,
        supervisor_history,
        prior_review,
        window_deltas_since_prior_review,
        live_output,
    } = context;
    debug_assert_eq!(candidates.len(), registries.len());
    let Some(candidate) = candidates
        .iter()
        .find(|candidate| candidate.index == selected_lane)
    else {
        return (
            Err(anyhow::anyhow!(
                "terminal completion review selected unknown lane {selected_lane}"
            )),
            crate::llm_client::TokenUsage::default(),
        );
    };
    let Some(registry) = registries.get(selected_lane) else {
        return (
            Err(anyhow::anyhow!(
                "terminal completion review lane {selected_lane} has no checkout registry"
            )),
            crate::llm_client::TokenUsage::default(),
        );
    };
    let audit = AsgardAuditContext {
        registries,
        candidates,
        definitions: audit_definitions.to_vec(),
        allowed_lane: Some(selected_lane),
        retained_windows,
        window,
    };
    let (terminal_non_test_patch, terminal_test_patch) = asgard_patch_surfaces(&candidate.patch);
    let terminal_test_patch = if terminal_test_patch.len() > 200_000 {
        format!(
            "{}\n... test patch truncated (mechanical cap)",
            crate::text::truncate_utf8(&terminal_test_patch, 200_000)
        )
    } else {
        terminal_test_patch
    };
    let candidate_trajectory = match render_asgard_candidate_trajectory(
        candidate,
        registry.cwd(),
        "last_selected_decision",
        &candidate.delta_patch,
    ) {
        Ok(trajectory) => format!(
            "<candidate_trajectories window=\"{window}\">\n{trajectory}</candidate_trajectories>"
        ),
        Err(error) => {
            return (Err(error), crate::llm_client::TokenUsage::default());
        }
    };
    let prior_review_delta = prior_review
        .map(|_| {
            window_deltas_since_prior_review
                .iter()
                .map(|(window, patch)| {
                    format!(
                        "<window_delta window=\"{window}\">\n{}\n</window_delta>",
                        String::from_utf8_lossy(patch)
                    )
                })
                .chain(std::iter::once(format!(
                    "<window_delta window=\"{window}\">\n{}\n</window_delta>",
                    String::from_utf8_lossy(&candidate.delta_patch)
                )))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    let messages = asgard_completion_review_messages(
        original_task,
        selected_trajectory_initial,
        selected_trajectory_windows,
        supervisor_history,
        candidates.len(),
        max_candidate_count,
        task_contract_checklist,
        canonical_ledger,
        terminal_non_test_patch,
        terminal_test_patch,
        selected_lane,
        candidate_trajectory,
        prior_review,
        prior_review_delta,
    );
    tracing::info!(
        window,
        selected_lane = selected_lane + 1,
        selected_candidate_bytes = messages
            .last()
            .map(asgard_message_text)
            .map_or(0, |text| text.len()),
        "assembled isolated Asgard completion-review dossier"
    );
    let (decision, usage) = run_asgard_supervisor_tool_steps(
        llm.as_ref(),
        messages,
        AsgardSupervisorToolContext {
            model,
            candidate_count: candidates.len(),
            max_candidate_count,
            idle_timeout,
            audit: Some(audit),
            required_winner: Some(selected_lane),
            checklist_ids: &task_contract_checklist.contracts,
            carry_forward_allowed: prior_review.is_some(),
        },
        cancel,
        Some(AsgardStreamSinks::new(live_output, "Completion reviewer")),
    )
    .await;
    (decision.map(|(decision, _stats)| decision), usage)
}

pub(crate) fn render_asgard_candidate_trajectory(
    candidate: &AsgardCandidate,
    worktree: &Path,
    diff_base: &str,
    diff_patch: &[u8],
) -> anyhow::Result<String> {
    let trajectory = render_asgard_dossier_messages(&candidate.supervisor_window_messages);
    let candidate_changed_production_files = asgard_patch_production_inventory(&candidate.patch);
    let (candidate_created_test_files, candidate_modified_test_files) =
        asgard_patch_test_inventory(&candidate.patch);
    let candidate_patch_manifest = serde_json::to_string_pretty(&AsgardCandidatePatchManifest {
        candidate_changed_production_files,
        candidate_created_test_files,
        candidate_modified_test_files,
        patch_bytes: candidate.patch.len(),
    })
    .map_err(|error| {
        anyhow::anyhow!(
            "failed to serialize Asgard lane {} patch manifest: {error}",
            candidate.index
        )
    })?;
    let execution_ledger =
        serde_json::to_string_pretty(&candidate.window_ledger).map_err(|error| {
            anyhow::anyhow!(
                "failed to serialize Asgard lane {} execution ledger: {error}",
                candidate.index
            )
        })?;
    let last_shell_step = candidate
        .window_ledger
        .entries
        .last()
        .map(|entry| entry.step);
    let mut seen_files = HashSet::new();
    let files_edited_after_last_command = candidate
        .window_ledger
        .edit_steps
        .iter()
        .filter(|edit| last_shell_step.is_none_or(|step| edit.step > step))
        .filter_map(|edit| {
            if seen_files.insert(edit.file.clone()) {
                Some(edit.file.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let files_edited_after_last_command = serde_json::to_string(&files_edited_after_last_command)
        .expect("files edited after last command serialize");
    Ok(format!(
        "\n<lane_trajectory index=\"{}\" model=\"{}\" worktree=\"{}\" stop=\"{:?}\">\n\
         <candidate_patch_manifest derived_from_full_patch=\"true\">\n{}\n\
         </candidate_patch_manifest>\n\
         <candidate_window_diff base=\"{}\" bytes=\"{}\">\n{}\n\
         </candidate_window_diff>\n\
         <execution_ledger mechanically_derived=\"true\" source=\"lane tool calls, not candidate claims\">\n{}\n\
         </execution_ledger>\n\
         files_edited_after_last_command: {}\n\
         <window_trajectory>\n{}\n</window_trajectory>\n</lane_trajectory>\n",
        candidate.index,
        candidate.model,
        worktree.display(),
        candidate.outcome.stop,
        candidate_patch_manifest,
        diff_base,
        diff_patch.len(),
        String::from_utf8_lossy(diff_patch),
        execution_ledger,
        files_edited_after_last_command,
        trajectory,
    ))
}

pub(crate) struct AsgardSupervisorToolContext<'a> {
    pub(crate) model: &'a str,
    pub(crate) candidate_count: usize,
    pub(crate) max_candidate_count: usize,
    pub(crate) idle_timeout: IdleTimeouts,
    pub(crate) audit: Option<AsgardAuditContext<'a>>,
    pub(crate) required_winner: Option<usize>,
    pub(crate) checklist_ids: &'a [AsgardTaskContract],
    pub(crate) carry_forward_allowed: bool,
}

pub(crate) struct AsgardAuditContext<'a> {
    pub(crate) registries: &'a [Arc<crate::tools::ToolRegistry>],
    pub(crate) candidates: &'a [AsgardCandidate],
    pub(crate) definitions: Vec<ToolDefinition>,
    pub(crate) allowed_lane: Option<usize>,
    /// Window the live candidate trajectories were rendered for.
    pub(crate) window: usize,
    /// Winning lanes from earlier windows, whose handles the dossier still
    /// cites through the canonical ledger and carried-forward evidence.
    pub(crate) retained_windows: &'a [AsgardRetainedWindow],
}

/// One earlier window's winning lane, retained so its handles stay expandable.
#[derive(Debug, Clone)]
pub(crate) struct AsgardRetainedWindow {
    pub(crate) window: usize,
    pub(crate) lane: usize,
    pub(crate) messages: Vec<ChatMessage>,
}

pub(crate) fn asgard_is_audit_tool(name: &str) -> bool {
    ASGARD_AUDIT_BUILTIN_TOOLS.contains(&name) || ASGARD_AUDIT_BIFROST_TOOLS.contains(&name)
}

pub(crate) fn asgard_audit_tool_definitions(
    definitions: Vec<ToolDefinition>,
    candidate_count: usize,
) -> Vec<ToolDefinition> {
    definitions
        .into_iter()
        .filter(|definition| asgard_is_audit_tool(&definition.function.name))
        .filter_map(|mut definition| {
            let parameters = definition.function.parameters.as_object_mut()?;
            let properties = parameters.get_mut("properties")?.as_object_mut()?;
            properties.insert(
                "lane".to_string(),
                serde_json::json!({
                    "type": "integer",
                    "minimum": 0,
                    "maximum": candidate_count.saturating_sub(1),
                    "description": "Zero-based candidate lane whose checkout should be inspected."
                }),
            );
            let required = parameters
                .entry("required")
                .or_insert_with(|| serde_json::Value::Array(Vec::new()))
                .as_array_mut()?;
            if !required.iter().any(|value| value == "lane") {
                required.push(serde_json::Value::String("lane".to_string()));
            }
            definition.function.description = format!(
                "Read-only audit of a candidate checkout. {}",
                definition.function.description
            );
            Some(definition)
        })
        .collect()
}

pub(crate) fn asgard_view_tool_call_handles(
    arguments: &serde_json::Value,
) -> anyhow::Result<Vec<String>> {
    let handles = arguments
        .get("handles")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("`view_tool_call` requires a `handles` array"))?;
    anyhow::ensure!(!handles.is_empty(), "`handles` must not be empty");
    anyhow::ensure!(
        handles.len() <= ASGARD_VIEW_TOOL_CALL_MAX_HANDLES,
        "`handles` accepts at most {ASGARD_VIEW_TOOL_CALL_MAX_HANDLES} ids per call"
    );
    handles
        .iter()
        .map(|handle| {
            handle
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| anyhow::anyhow!("every entry in `handles` must be a string"))
        })
        .collect()
}

pub(crate) fn asgard_view_tool_call_tool() -> ToolDefinition {
    ToolDefinition {
        r#type: "function".to_string(),
        function: FunctionDef {
            name: ASGARD_VIEW_TOOL_CALL_NAME.to_string(),
            description:
                "Retrieve the complete arguments and untruncated result behind one or more compact \
                 trajectory tool lines. Pass the `id` values shown on <tool> lines. This reads \
                 recorded window data, executes nothing, and does not consume the \
                 information-gathering budget."
                    .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["handles"],
                "properties": {
                    "handles": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": ASGARD_VIEW_TOOL_CALL_MAX_HANDLES,
                        "items": { "type": "string" },
                        "description": "Tool-line ids such as \"w3l1m7\", taken verbatim from the compact trajectories."
                    }
                },
            }),
        },
    }
}

/// Resolves compact-trajectory handles back to the durable tool calls and their
/// complete results.
///
/// The full window is already retained on each candidate, so this is an
/// in-memory lookup rather than a re-read of the checkout — which is why it is
/// exempt from the audit-round budget.
pub(crate) fn resolve_asgard_tool_call_handles(
    audit: &AsgardAuditContext<'_>,
    handles: &[String],
) -> String {
    let window = audit.window;
    let mut rendered = String::new();
    for handle in handles {
        let Some((handle_window, lane, index)) = crate::asgard::parse_asgard_tool_handle(handle)
        else {
            rendered.push_str(&format!(
                "<tool_call id=\"{handle}\" error=\"unrecognized handle format\" />\n"
            ));
            continue;
        };
        // The terminal completion review sees exactly one lane; a handle naming
        // another lane must be refused here just as the audit tools refuse it.
        // Earlier windows are exempt: only their winner was retained, so their
        // lane is already the selected one by construction.
        if let Some(allowed_lane) = audit.allowed_lane
            && handle_window == window
            && lane != allowed_lane
        {
            rendered.push_str(&format!(
                "<tool_call id=\"{handle}\" error=\"only selected candidate lane {allowed_lane} is \
                 available during terminal completion review\" />\n"
            ));
            continue;
        }
        let messages = if handle_window == window {
            let Some(candidate) = audit
                .candidates
                .iter()
                .find(|candidate| candidate.index == lane)
            else {
                rendered.push_str(&format!(
                    "<tool_call id=\"{handle}\" error=\"candidate lane {lane} does not exist\" />\n"
                ));
                continue;
            };
            &candidate.window_messages
        } else {
            let retained = audit
                .retained_windows
                .iter()
                .find(|retained| retained.window == handle_window && retained.lane == lane);
            let Some(retained) = retained else {
                rendered.push_str(&format!(
                    "<tool_call id=\"{handle}\" error=\"window {handle_window} lane {lane} was not \
                     retained; only the winning lane of each earlier window is expandable\" />\n"
                ));
                continue;
            };
            &retained.messages
        };
        let Some(result) = messages.get(index).filter(|message| message.role == "tool") else {
            rendered.push_str(&format!(
                "<tool_call id=\"{handle}\" error=\"no tool result at this position\" />\n"
            ));
            continue;
        };
        let call = crate::asgard::originating_tool_call(messages, index);
        let name = call
            .map(|call| call.function.name.as_str())
            .or(result.name.as_deref())
            .unwrap_or("tool");
        let arguments = call
            .map(|call| call.function.arguments.as_str())
            .unwrap_or("(arguments unavailable)");
        rendered.push_str(&format!(
            "<tool_call id=\"{handle}\" lane=\"{lane}\" name=\"{name}\">\n\
             <arguments>\n{arguments}\n</arguments>\n\
             <result>\n{}\n</result>\n\
             </tool_call>\n",
            asgard_message_text(result),
        ));
    }
    rendered
}

pub(crate) fn asgard_audit_arguments(
    mut arguments: serde_json::Value,
    candidate_count: usize,
) -> anyhow::Result<(usize, serde_json::Value)> {
    let object = arguments
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("audit tool arguments must be an object"))?;
    let lane = object
        .remove("lane")
        .and_then(|value| value.as_u64())
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| anyhow::anyhow!("audit tool requires an integer `lane`"))?;
    anyhow::ensure!(
        lane < candidate_count,
        "candidate lane {lane} does not exist"
    );
    Ok((lane, arguments))
}

async fn execute_asgard_audit_tool(
    audit: &AsgardAuditContext<'_>,
    name: &str,
    arguments: serde_json::Value,
    cancel: &tokio_util::sync::CancellationToken,
) -> String {
    let (lane, arguments) = match asgard_audit_arguments(arguments, audit.candidates.len()) {
        Ok(parsed) => parsed,
        Err(error) => return format!("Error: {error}"),
    };
    if let Some(allowed_lane) = audit.allowed_lane
        && lane != allowed_lane
    {
        return format!(
            "Error: only selected candidate lane {allowed_lane} is available during terminal completion review"
        );
    }
    let Some(registry) = audit.registries.get(lane) else {
        return format!("Error: candidate lane {lane} has no checkout registry");
    };
    tracing::info!(
        lane = lane + 1,
        tool = name,
        worktree = %registry.cwd().display(),
        arguments = %arguments,
        "running Asgard supervisor audit tool"
    );
    let output = registry
        .execute_with_sandbox_mode_cancellable(
            name,
            arguments,
            crate::tools::sandbox::SandboxPolicy::ReadOnly,
            false,
            None,
            Some(cancel),
        )
        .await
        .output;
    if std::env::var_os("ASGARD_CAPTURE_WINDOW_SUMMARIES").is_some() {
        tracing::info!(
            lane = lane + 1,
            tool = name,
            output = %output,
            "captured Asgard supervisor audit result for review"
        );
    }
    output
}

pub(crate) async fn run_asgard_initial_advice_tool_steps(
    llm: &dyn crate::llm_client::LlmBackend,
    mut messages: Vec<ChatMessage>,
    context: AsgardSupervisorToolContext<'_>,
    cancel: tokio_util::sync::CancellationToken,
    stream_sinks: Option<AsgardStreamSinks>,
) -> (
    anyhow::Result<AsgardSupervisorInitialAdvice>,
    crate::llm_client::TokenUsage,
) {
    const MAX_STEPS: usize = 2;
    let tools = vec![asgard_advise_trajectories_tool(context.max_candidate_count)];
    let mut usage = crate::llm_client::TokenUsage::default();
    let mut last_invalid_response = None;

    for step in 1..=MAX_STEPS {
        let request_bytes = render_asgard_dossier_messages(&messages).len();
        let text_sink = stream_sinks.as_ref().map(|sinks| sinks.text.clone());
        let thought_sink = stream_sinks.as_ref().map(|sinks| sinks.thought.clone());
        let response = stream_chat_no_visible_output_with_retry(
            llm,
            "planning initial Asgard trajectories",
            &cancel,
            || {
                let text_sink = text_sink.clone();
                let thought_sink = thought_sink.clone();
                StreamChatRequest {
                    model: context.model.to_string(),
                    messages: messages.clone(),
                    tools: Some(tools.clone()),
                    reasoning_effort: None,
                    service_tier: None,
                    temperature: None,
                    structured_output: None,
                    on_token: Box::new(move |token| {
                        if let Some(sink) = &text_sink {
                            (sink
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner))(
                                token
                            );
                        }
                    }),
                    on_thought: Box::new(move |token| {
                        if let Some(sink) = &thought_sink {
                            (sink
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner))(
                                token
                            );
                        }
                    }),
                    cancel: cancel.clone(),
                    idle_timeouts: context.idle_timeout,
                }
            },
        )
        .await;
        let response = match response {
            Ok(response) => response,
            Err(error) => return (Err(error), usage),
        };
        let turn_usage = response.usage();
        trace_asgard_phase_turn_usage(
            "initial_advice",
            context.model,
            None,
            None,
            step,
            "selection",
            request_bytes,
            turn_usage,
        );
        usage.add(turn_usage);

        match response {
            LlmResponse::Text {
                text,
                reasoning_content,
                ..
            } => {
                messages.push(ChatMessage::assistant_with_reasoning(
                    text,
                    reasoning_content,
                ));
            }
            LlmResponse::ToolCalls {
                text,
                reasoning_content,
                calls,
                ..
            } => {
                let mut advice = None;
                for call in &calls {
                    let normalized = match crate::tool_arguments::normalize_tool_arguments(
                        &call.function.arguments,
                    ) {
                        Ok(arguments) => arguments.value,
                        Err(error) => {
                            last_invalid_response = Some(anyhow::anyhow!(
                                "Asgard supervisor emitted invalid `{}` arguments: {error}",
                                call.function.name
                            ));
                            continue;
                        }
                    };
                    match call.function.name.as_str() {
                        "advise_trajectories" => {
                            if advice.is_some() {
                                advice = None;
                                last_invalid_response = Some(anyhow::anyhow!(
                                    "Asgard supervisor called advise_trajectories more than once"
                                ));
                                break;
                            }
                            let serialized = serde_json::to_string(&normalized)
                                .expect("normalized tool arguments serialize");
                            match parse_asgard_initial_advice(
                                &serialized,
                                context.max_candidate_count,
                            ) {
                                Ok(value) => advice = Some(value),
                                Err(error) => last_invalid_response = Some(error),
                            }
                        }
                        other => {
                            last_invalid_response = Some(anyhow::anyhow!(
                                "Asgard supervisor called unexpected tool `{other}`"
                            ));
                        }
                    }
                }

                if let Some(advice) = advice {
                    return (Ok(advice), usage);
                }
                if !text.is_empty() || reasoning_content.as_deref().is_some_and(|s| !s.is_empty()) {
                    messages.push(ChatMessage::assistant_with_reasoning(
                        text,
                        reasoning_content,
                    ));
                }
            }
        }

        if step < MAX_STEPS {
            let detail = last_invalid_response
                .as_ref()
                .map(|error| format!(" Your previous response was invalid: {error}."))
                .unwrap_or_default();
            messages.push(ChatMessage::user(format!(
                "You have not advised the trajectories.{detail} Only advise_trajectories is available. Call advise_trajectories now. Do not answer in prose."
            )));
        }
    }

    (
        Err(last_invalid_response.unwrap_or_else(|| {
            anyhow::anyhow!(
                "Asgard supervisor did not call advise_trajectories after {MAX_STEPS} steps"
            )
        })),
        usage,
    )
}

pub(crate) async fn run_asgard_supervisor_tool_steps(
    llm: &dyn crate::llm_client::LlmBackend,
    mut messages: Vec<ChatMessage>,
    context: AsgardSupervisorToolContext<'_>,
    cancel: tokio_util::sync::CancellationToken,
    stream_sinks: Option<AsgardStreamSinks>,
) -> (
    anyhow::Result<(AsgardSupervisorDecision, AsgardChallengeStats)>,
    crate::llm_client::TokenUsage,
) {
    let audit_round_limit = if context.audit.is_some() {
        ASGARD_AUDIT_MAX_ROUNDS
    } else {
        0
    };
    let tools = asgard_supervisor_tool_definitions(&context);
    // The completion review's structural loop (violation retry, challenge,
    // re-verdict) legitimately needs more selector rounds than a comparative
    // decision does.
    let selection_attempt_limit = if context.required_winner.is_some() {
        ASGARD_REVIEW_SELECTION_MAX_ATTEMPTS
    } else {
        ASGARD_SELECTION_MAX_ATTEMPTS
    };
    let mut usage = crate::llm_client::TokenUsage::default();
    let mut last_invalid_response = None;
    let mut audit_rounds_used = 0usize;
    let mut selection_attempts = 0usize;
    let mut validation_rounds = 0usize;
    let mut retrieval_rounds_used = 0usize;

    loop {
        // A round that only expands compact trajectory lines is refunded below,
        // so retrieval never competes with inspection or selection.
        let mut retrieval_round = false;
        let mut retrieved_handles: Vec<String> = Vec::new();
        let mut unresolved_handles: Vec<String> = Vec::new();
        let audit_phase = audit_rounds_used < audit_round_limit;
        if audit_phase {
            audit_rounds_used += 1;
        } else {
            selection_attempts += 1;
        }
        let turn = audit_rounds_used + selection_attempts;
        let request_bytes = render_asgard_dossier_messages(&messages).len();
        let text_sink = stream_sinks.as_ref().map(|sinks| sinks.text.clone());
        let thought_sink = stream_sinks.as_ref().map(|sinks| sinks.thought.clone());
        let response = stream_chat_no_visible_output_with_retry(
            llm,
            "selecting an Asgard trajectory",
            &cancel,
            || {
                let text_sink = text_sink.clone();
                let thought_sink = thought_sink.clone();
                StreamChatRequest {
                    model: context.model.to_string(),
                    messages: messages.clone(),
                    tools: Some(tools.clone()),
                    // DeepSeek's reasoning mode cannot be combined with forced
                    // tool choice. Leave reasoning enabled and remind it once if
                    // the first step does not call the terminal selector.
                    reasoning_effort: None,
                    service_tier: None,
                    temperature: None,
                    structured_output: None,
                    on_token: Box::new(move |token| {
                        if let Some(sink) = &text_sink {
                            (sink
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner))(
                                token
                            );
                        }
                    }),
                    on_thought: Box::new(move |token| {
                        if let Some(sink) = &thought_sink {
                            (sink
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner))(
                                token
                            );
                        }
                    }),
                    cancel: cancel.clone(),
                    idle_timeouts: context.idle_timeout,
                }
            },
        )
        .await;
        let response = match response {
            Ok(response) => response,
            Err(error) => return (Err(error), usage),
        };
        let turn_usage = response.usage();
        trace_asgard_phase_turn_usage(
            if context.required_winner.is_some() {
                "completion_review"
            } else {
                "routing_supervisor"
            },
            context.model,
            None,
            context.required_winner,
            turn,
            if audit_phase { "audit" } else { "selection" },
            request_bytes,
            turn_usage,
        );
        usage.add(turn_usage);

        match response {
            LlmResponse::Text {
                text,
                reasoning_content,
                ..
            } => {
                messages.push(ChatMessage::assistant_with_reasoning(
                    text,
                    reasoning_content,
                ));
            }
            LlmResponse::ToolCalls {
                text,
                reasoning_content,
                calls,
                ..
            } => {
                let call_names = calls
                    .iter()
                    .map(|call| call.function.name.as_str())
                    .collect::<Vec<_>>();
                let selector_calls = calls
                    .iter()
                    .filter(|call| call.function.name == "select_trajectory")
                    .collect::<Vec<_>>();
                tracing::info!(
                    model = context.model,
                    phase = if audit_phase { "audit" } else { "selection" },
                    audit_round = if audit_phase {
                        Some(audit_rounds_used)
                    } else {
                        None
                    },
                    selection_attempt = if audit_phase {
                        None
                    } else {
                        Some(selection_attempts)
                    },
                    calls = ?call_names,
                    selector_count = selector_calls.len(),
                    "received Asgard supervisor tool-call batch"
                );

                let may_accept_selector =
                    selector_calls.len() == 1 && (calls.len() == 1 || !audit_phase);
                let mut selector_error = None;
                if may_accept_selector {
                    let selection_call = selector_calls[0];
                    match crate::tool_arguments::normalize_tool_arguments(
                        &selection_call.function.arguments,
                    ) {
                        Ok(arguments) => {
                            let serialized = serde_json::to_string(&arguments.value)
                                .expect("normalized tool arguments serialize");
                            match parse_asgard_supervisor_decision_with_max(
                                &serialized,
                                context.candidate_count,
                                context.max_candidate_count,
                            ) {
                                Ok(decision) => {
                                    if let Some(required_winner) = context.required_winner
                                        && decision.winner != required_winner
                                    {
                                        selector_error = Some(format!(
                                            "terminal completion review must keep selected lane {required_winner}, not lane {}",
                                            decision.winner
                                        ));
                                    } else if context.required_winner.is_some() && decision.complete
                                    {
                                        let violations = asgard_validate_contract_rows(
                                            &decision,
                                            context.checklist_ids,
                                            context.carry_forward_allowed,
                                        );
                                        if violations.is_empty() {
                                            {
                                                if calls.len() > 1 {
                                                    tracing::warn!(
                                                        model = context.model,
                                                        ignored_calls = ?call_names
                                                            .iter()
                                                            .copied()
                                                            .filter(|name| *name != "select_trajectory")
                                                            .collect::<Vec<_>>(),
                                                        "accepted final Asgard selection and ignored other tool calls"
                                                    );
                                                }
                                                let decision_call =
                                                    if context.required_winner.is_some() {
                                                        "completion_review"
                                                    } else {
                                                        "supervisor"
                                                    };
                                                let stats = AsgardChallengeStats {
                                                    issued: false,
                                                    flipped: false,
                                                    validation_rounds,
                                                };
                                                crate::trace_logging::append_trace_record(
                                                    asgard_decision_trace_record(
                                                        decision_call,
                                                        &decision,
                                                        stats,
                                                    ),
                                                );
                                                return (Ok((decision, stats)), usage);
                                            }
                                        } else {
                                            validation_rounds += 1;
                                            selector_error = Some(
                                                asgard_contract_violation_message(&violations),
                                            );
                                        }
                                    } else {
                                        if calls.len() > 1 {
                                            tracing::warn!(
                                                model = context.model,
                                                ignored_calls = ?call_names
                                                    .iter()
                                                    .copied()
                                                    .filter(|name| *name != "select_trajectory")
                                                    .collect::<Vec<_>>(),
                                                "accepted final Asgard selection and ignored other tool calls"
                                            );
                                        }
                                        let decision_call = if context.required_winner.is_some() {
                                            "completion_review"
                                        } else {
                                            "supervisor"
                                        };
                                        let stats = AsgardChallengeStats {
                                            issued: false,
                                            flipped: false,
                                            validation_rounds,
                                        };
                                        crate::trace_logging::append_trace_record(
                                            asgard_decision_trace_record(
                                                decision_call,
                                                &decision,
                                                stats,
                                            ),
                                        );
                                        return (Ok((decision, stats)), usage);
                                    }
                                }
                                Err(error) => selector_error = Some(error.to_string()),
                            }
                        }
                        Err(error) => {
                            selector_error =
                                Some(format!("invalid `select_trajectory` arguments: {error}"));
                        }
                    }
                } else if selector_calls.len() > 1 {
                    selector_error = Some(
                        "the supervisor called select_trajectory more than once in one batch"
                            .to_string(),
                    );
                }

                messages.push(
                    ChatMessage::assistant_tool_calls_with_content_and_reasoning(
                        text,
                        calls.clone(),
                        reasoning_content,
                    ),
                );
                let has_audit_calls = calls
                    .iter()
                    .any(|call| asgard_is_audit_tool(&call.function.name));
                let has_view_calls = calls
                    .iter()
                    .any(|call| call.function.name == ASGARD_VIEW_TOOL_CALL_NAME);
                // Only refund when retrieval could actually have returned
                // something; otherwise a hallucinated call would loop for free
                // until the runaway guard fires.
                retrieval_round = has_view_calls
                    && context.audit.is_some()
                    && !has_audit_calls
                    && selector_calls.is_empty();
                let unexpected_calls = calls
                    .iter()
                    .filter(|call| {
                        call.function.name != "select_trajectory"
                            && call.function.name != ASGARD_VIEW_TOOL_CALL_NAME
                            && !asgard_is_audit_tool(&call.function.name)
                    })
                    .map(|call| call.function.name.as_str())
                    .collect::<Vec<_>>();
                for call in &calls {
                    let output = if call.function.name == ASGARD_VIEW_TOOL_CALL_NAME {
                        // Available in every phase: retrieving evidence the
                        // supervisor was already shown costs nothing to execute.
                        match crate::tool_arguments::normalize_tool_arguments(
                            &call.function.arguments,
                        ) {
                            Ok(arguments) => match asgard_view_tool_call_handles(&arguments.value) {
                                Ok(handles) => match &context.audit {
                                    Some(audit) => {
                                        let resolved =
                                            resolve_asgard_tool_call_handles(audit, &handles);
                                        for handle in &handles {
                                            if resolved.contains(&format!(
                                                "<tool_call id=\"{handle}\" lane="
                                            )) {
                                                retrieved_handles.push(handle.clone());
                                            } else {
                                                unresolved_handles.push(handle.clone());
                                            }
                                        }
                                        resolved
                                    }
                                    None => "Error: no candidate trajectories are retrievable for this decision".to_string(),
                                },
                                Err(error) => format!("Error: {error}"),
                            },
                            Err(error) => {
                                format!("Error: invalid `view_tool_call` arguments: {error}")
                            }
                        }
                    } else if call.function.name == "select_trajectory" {
                        if let Some(error) = &selector_error {
                            format!("Error: {error}")
                        } else if audit_phase && calls.len() > 1 {
                            "Selection deferred because inspection tools were called in the same audit round. Review their results, then call select_trajectory by itself."
                                .to_string()
                        } else {
                            "Error: select_trajectory was not accepted; call it once with valid arguments."
                                .to_string()
                        }
                    } else if !asgard_is_audit_tool(&call.function.name) {
                        format!(
                            "Error: tool `{}` is not available for Asgard audit",
                            call.function.name
                        )
                    } else if !audit_phase {
                        format!(
                            "Audit budget exhausted; `{}` was not executed. Call select_trajectory now using the evidence already gathered.",
                            call.function.name
                        )
                    } else if let Some(audit) = &context.audit {
                        match crate::tool_arguments::normalize_tool_arguments(
                            &call.function.arguments,
                        ) {
                            Ok(arguments) => {
                                execute_asgard_audit_tool(
                                    audit,
                                    &call.function.name,
                                    arguments.value,
                                    &cancel,
                                )
                                .await
                            }
                            Err(error) => format!(
                                "Error: invalid `{}` arguments: {error}",
                                call.function.name
                            ),
                        }
                    } else {
                        format!(
                            "Error: tool `{}` is not available for this decision",
                            call.function.name
                        )
                    };
                    messages.push(ChatMessage::tool_result(
                        &call.id,
                        &call.function.name,
                        output,
                    ));
                }

                last_invalid_response = if let Some(error) = selector_error {
                    Some(anyhow::anyhow!(error))
                } else if !unexpected_calls.is_empty() {
                    Some(anyhow::anyhow!(
                        "Asgard supervisor called unexpected tool(s) `{}`",
                        unexpected_calls.join(", ")
                    ))
                } else if retrieval_round || (audit_phase && has_audit_calls) {
                    None
                } else {
                    Some(anyhow::anyhow!(
                        "Asgard supervisor did not call select_trajectory by itself"
                    ))
                };
            }
        }

        if retrieval_round {
            // Refund the round this iteration charged: expanding compact lines
            // is retrieval, not inspection or selection.
            if audit_phase {
                audit_rounds_used -= 1;
            } else {
                selection_attempts -= 1;
            }
            retrieval_rounds_used += 1;
            crate::trace_logging::append_trace_record(serde_json::json!({
                "type": "asgard_supervisor_retrieval",
                "phase": if context.required_winner.is_some() {
                    "completion_review"
                } else {
                    "routing_supervisor"
                },
                "model": context.model,
                "window": context.audit.as_ref().map(|audit| audit.window),
                "round": retrieval_rounds_used,
                "handles": retrieved_handles,
                "unresolved": unresolved_handles,
            }));
            if retrieval_rounds_used >= ASGARD_RETRIEVAL_MAX_ROUNDS {
                return (
                    Err(anyhow::anyhow!(
                        "Asgard supervisor made {retrieval_rounds_used} consecutive view_tool_call rounds without deciding"
                    )),
                    usage,
                );
            }
            messages.push(ChatMessage::user(
                "Those are the full tool calls you requested. Retrieving them did not consume your information-gathering budget. Continue: retrieve more ids, audit a checkout, or call select_trajectory.",
            ));
            continue;
        }

        let detail = last_invalid_response
            .as_ref()
            .map(|error| format!(" Your previous response was invalid: {error}."))
            .unwrap_or_default();
        if audit_phase {
            let remaining = audit_round_limit.saturating_sub(audit_rounds_used);
            if remaining == 0 {
                messages.push(ChatMessage::user(format!(
                    "The information-gathering budget is exhausted.{detail} You must now call select_trajectory with your best engineering judgment. Do not call audit tools. If verification is still needed, select the best incomplete lane and direct candidates to perform it in the next window. Do not answer in prose."
                )));
            } else if remaining == 1 {
                messages.push(ChatMessage::user(format!(
                    "This is your final information-gathering turn.{detail} Inspect only evidence necessary to distinguish the lanes or decide completion. After these results you must select the best lane under any remaining uncertainty. You may instead call select_trajectory now."
                )));
            } else {
                messages.push(ChatMessage::user(format!(
                    "You have {remaining} information-gathering turns remaining.{detail} Use audit tools only for a consequential unresolved question, or call select_trajectory now."
                )));
            }
        } else if selection_attempts < selection_attempt_limit {
            let remaining = selection_attempt_limit - selection_attempts;
            messages.push(ChatMessage::user(format!(
                "You have not made the required trajectory decision.{detail} Call select_trajectory now with your best judgment; {remaining} selection attempt(s) remain. Audit calls will be ignored. Do not answer in prose."
            )));
        } else {
            break;
        }
    }

    (
        Err(last_invalid_response.unwrap_or_else(|| {
            anyhow::anyhow!(
                "Asgard supervisor did not call select_trajectory after {audit_rounds_used} audit round(s) and {selection_attempts} selection attempt(s)"
            )
        })),
        usage,
    )
}

pub(crate) fn asgard_supervisor_tool_definitions(
    context: &AsgardSupervisorToolContext<'_>,
) -> Vec<ToolDefinition> {
    let mut tools = vec![asgard_select_trajectory_tool(
        context.candidate_count,
        context.max_candidate_count,
    )];
    if let Some(audit) = &context.audit {
        tools.push(asgard_view_tool_call_tool());
        tools.extend(audit.definitions.clone());
    }
    tools
}

pub(crate) fn asgard_decision_trace_record(
    decision_call: &str,
    decision: &AsgardSupervisorDecision,
    challenge: AsgardChallengeStats,
) -> serde_json::Value {
    serde_json::json!({
        "type": "asgard_decision",
        "call": decision_call,
        "decision": decision,
        "challenge": {
            "issued": challenge.issued,
            "flipped": challenge.flipped,
            "validation_rounds": challenge.validation_rounds,
        },
    })
}

pub(crate) fn asgard_original_task(initial_messages: &[ChatMessage]) -> String {
    // Session bootstrap can contribute user-role instruction messages (for
    // example AGENTS.md) before the actual prompt. The last user message is
    // the prompt that started this trajectory; treating the first one as the
    // task gives the supervisor an authoritative-looking but incorrect task
    // and also poisons advice scope validation.
    initial_messages
        .iter()
        .rfind(|message| message.role == "user")
        .map(asgard_message_text)
        .unwrap_or_default()
}

pub(crate) fn asgard_window_policy() -> &'static str {
    "Return one distinct advice for each lane to launch, from 1 through the configured maximum, and choose next_window_steps from 1 through 10. The number of advices is the next window's candidate count. Candidate count buys independent breadth: use more lanes when the diagnosis, architecture, contract reading, or evidence is uncertain enough that genuinely different approaches can teach you something. Do not spend lanes on cosmetic variations. If a concrete bug must be fixed before any new direction can be useful, return one advice to fix and verify that serial dependency first. The shared step horizon controls when comparison resumes, not when candidates should declare the task finished. Use short horizons when feedback is valuable soon and longer horizons only for clear, mechanically involved work where interruption adds little value."
}

pub(crate) fn asgard_initial_advice_messages(
    original_task: &str,
    max_candidate_count: usize,
) -> Vec<ChatMessage> {
    vec![
        ChatMessage::system(format!(
            r#"You are directing the first Asgard coding window. Up to {max_candidate_count} candidate lanes are available. Your job is to choose how many to launch, choose their shared window length, and give each launched lane one concise, actionable strategy for beginning the original task.

Asgard preserves one canonical trajectory. At each window, every launched lane starts independently from the same canonical history and repository state, runs for the shared step horizon, and is then compared by a supervisor. Exactly one winner becomes the next canonical state; all losing lane work is discarded. A candidate count therefore trades cost for useful breadth, while the step horizon controls how long lanes diverge before the next comparison.

The original task is authoritative. Every strategy must independently remain compliant with the task. Do not prescribe exact syntax, APIs, grammar shapes, or implementation facts unless the supplied history establishes them; when uncertain, direct a candidate to inspect the relevant source or behavior and resolve the uncertainty. If useful, advise a candidate to update_plan. The candidates continue normal rollouts and must not stop at Asgard window boundaries.

{}

Call advise_trajectories exactly once. Do not call another tool or answer in prose. If you omit advise_trajectories, you will receive one reminder."#,
            asgard_window_policy()
        )),
        ChatMessage::user(format!(
            r#"ORIGINAL TASK (complete):
{original_task}

<initial_advice_procedure>
Before calling advise_trajectories:
1. Re-read the original task. Identify the main required behaviors, explicit implementation constraints, and prohibitions.
2. Choose how many lane advices to return based on how many genuinely useful independent approaches the current uncertainty supports. The advice count is the number of candidates launched.
3. Choose next_window_steps based on when the next comparison should be valuable, not on a fixed rollout habit.
4. Produce that many task-compliant strategies. Make them genuinely different. When using multiple lanes, include one strategy that quickly falsifies the most consequential assumption.
Then call advise_trajectories exactly once.
</initial_advice_procedure>"#
        )),
    ]
}

#[derive(Clone, Copy)]
pub(crate) struct AsgardCandidateCounts {
    pub(crate) current: usize,
    pub(crate) max: usize,
}

pub(crate) fn asgard_supervisor_messages(
    original_task: &str,
    selected_trajectory_initial: &[ChatMessage],
    selected_trajectory_windows: &[Vec<ChatMessage>],
    supervisor_history: &AsgardSupervisorHistory,
    candidate_counts: AsgardCandidateCounts,
    task_contract_checklist: &AsgardTaskContractChecklist,
    candidate_trajectories: String,
) -> Vec<ChatMessage> {
    let AsgardCandidateCounts {
        current: candidate_count,
        max: max_candidate_count,
    } = candidate_counts;
    let mut messages = vec![
        ChatMessage::system(format!(
            r#"<mission>
You are the correctness supervisor for an Asgard coding trajectory. You are comparing {candidate_count} candidate lanes, and may launch 1-{max_candidate_count} lanes next. Every decision must choose the lane with the best long-term chance of solving the original task. Separately decide whether that selected endpoint is complete.

Asgard preserves one canonical trajectory. The lanes shown here all started independently from the same prior winner and repository state, ran for one shared step horizon, and now compete to become the sole next canonical state; losing work is discarded. If the endpoint is incomplete, the number of advices you return controls the cost-versus-breadth tradeoff for fresh continuations from this winner, and next_window_steps controls when those continuations are compared again.

Selection does not require certainty. Choose the best continuation under the available evidence even when every lane is flawed or an important question remains unanswered. complete=true is a nomination that the selected endpoint plausibly satisfies the original task; a separate isolated completion review owns the terminal adjudication.
</mission>

<task_and_evidence>
The original task is authoritative. Preserve its exact externally observable contracts, including argument order, return values, error behavior, atomicity, compatibility requirements, implementation constraints, and prohibitions. Merely defining the requested symbol or compiling the code does not establish that contract. A lane that contradicts the task is not rescued by confident prose, a large patch, or aggregate green checks. Judge architectural direction, correctness, recoverability, known defects, and evidence. Investigation that establishes an important constraint can be more valuable than immediate edits.

Each lane's work window is a candidate_window_handoff with format="compact_deterministic": a mechanically rendered record of what that lane actually did, produced by pure functions over its trajectory. No model summarized, characterized, or judged it, so it never reports work that did not happen — but it is compact, and a compact line states what a call was, not what its output showed. Tool results appear as one-line summaries stamped with an id, and a len attribute gives the full character count of any shortened text. Call view_tool_call with those ids to read the complete arguments and untruncated result. Retrieval is unbudgeted and executes nothing, so expand every line whose actual content bears on the decision instead of inferring from the summary; a claim resting on what a command printed requires reading what it printed. A tool line carrying exact_duplicate_of repeats a byte-identical earlier result and is not independent confirmation of anything. Treat these records as the original trajectory, while still judging what each command or inspection actually proves. A candidate_patch_manifest describes its cumulative changed production and test files. Normally each candidate_window_diff shows that lane's edits since the last selected decision. When candidate_diff_baseline names a current-window diff-compression anchor, that anchor is displayed first with its diff against the last selected decision, while every subsequent candidate_window_diff shows the transformation from the anchor lane to that lane. This is only diff compression: every lane still started independently from the same last selected decision, the anchor is not canonical or preferred, and lane indices remain authoritative. Use the compact trajectories, the tool calls you retrieve from them, the manifests, and the correctly based diffs together: challenge contradictions and consequential unsupported claims, but do not reread the repository merely to reproduce information the dossier already establishes or view_tool_call can return.

Interpret verification precisely. A successful command establishes only the behaviors its selected tests and assertions actually exercised; filters, wrappers, timeouts, zero-test selections, and missing combinations can mislead. Candidate-written tests can be valuable, but green tests changed alongside the implementation are not independent proof merely because they pass. Check whether they genuinely express the task, especially the highest-risk requirement and combinations of behaviors changed by the patch. Do not mechanically reject legitimate test or mock updates.

Prior supervisor decisions and lane advice provide continuity, not authority. Reconsider them when the original task or newer evidence disagrees.
</task_and_evidence>

<decision_depth>
First decide whether the leading lane plausibly appears terminal.

complete=true is a nomination, not a terminal verdict. A separate isolated completion review owns all terminal adjudication and re-judges the nominated endpoint against every task contract with the full evidence bundle. Nominate complete=true when the selected endpoint plausibly satisfies the task contracts; the compact trajectories and diffs are sufficient basis. Do not perform a terminal-grade audit, search for boundary counterexamples, or re-verify evidence before nominating — that depth is the review's job and paying it here duplicates the review. Do not nominate past a defect you can already see: a concrete task-relevant defect, contradiction, or plainly missing required work visible in the dossier means complete=false with advice targeting it.

If the best lanes are plainly incomplete, do not perform a terminal-grade exhaustive audit. Inspect only enough consequential evidence to rank their directions and formulate the next work. Unknowns can remain: select the best foundation, set complete=false, record the uncertainty, and delegate the needed implementation or verification to candidates in the next window.
</decision_depth>

<audit_protocol>
view_tool_call expands the compact trajectories you were given and is unbudgeted — use it freely and prefer it over auditing the checkout whenever the answer is in a tool result you were already shown. Every id is a handle of the form w<window>l<lane>m<index>, and execution_ledger entries are keyed by the same handles, so a ledger row you are about to cite as evidence can be expanded to the command's full output — the ledger carries only the last 400 bytes. Handles from earlier windows resolve too, for the lane that won that window. Lane-aware read_file, grep_search, list_directory, and Bifrost symbol tools additionally inspect the candidate checkouts for evidence the trajectories do not contain. They cannot run builds or tests. You have at most {audit_rounds} information-gathering responses using those checkout tools, including this one; the last opportunity will be announced. Batch related questions and stop once the answer cannot change the winner or completion judgment. After the budget is exhausted, audit calls are ignored and you must select.

Unavailable executable verification is not a reason to withhold a decision. If it is necessary to establish completeness, set complete=false and tell one or more next-window candidates exactly what behavior or command to verify. Repository auditing is evidence gathering, not another implementation rollout.
</audit_protocol>

<scope_and_completion>
Stay within the original scope. Do not repair dependencies, lockfiles, toolchains, generated machinery, warning policy, test selection, or expected outputs merely to hide a failure unless the task requires that surface or the candidate broke it. Treat a failure as environmental, pre-existing, flaky, or unrelated only with concrete evidence. Do not weaken required behavior to preserve obsolete callers or mocks.

Completion is a property of the endpoint, not of who introduced a defect. Set complete=true when the selected endpoint plausibly satisfies the exact task contracts on the basis of the compact trajectories and diffs; the isolated completion review — not this decision — adjudicates terminal completeness and will return the work to candidates if evidence falls short. Set complete=false when required work plainly remains or the dossier shows an unresolved concrete task-relevant defect. When executable verification is unavailable to you, select the best lane and delegate the specific check to the next candidate window rather than either assuming success or refusing to decide.
</scope_and_completion>

<continuation>
When incomplete, choose next_window_steps and return one concise, actionable, mutually distinct advice object per lane to launch, in zero-based lane order. The number of advices is the next window's candidate count. Use more candidates only when uncertainty supports genuinely different investigations or implementations. If an obvious concrete bug must be repaired before new directions become useful, return one advice to fix and verify it first. Each strategy must independently comply with the task. With multiple lanes, include one strategy that tests the selected direction's most consequential unverified assumption. When the checklist records materially ambiguous readings, use separate lanes only when implementing or testing those readings will resolve the ambiguity. Advice may tell candidates to inspect source, run a focused build or test, or update_plan. Do not assert exact APIs or implementation facts that the evidence does not establish. Candidates continue normal rollouts and do not stop at Asgard window boundaries.

{}
</continuation>

<output_contract>
Call select_trajectory exactly once and by itself as soon as your judgment is ready. Return advices=[] when complete. state_summary must concisely record why the endpoint won, the decisive evidence, and any unresolved risk. Do not answer in prose.
</output_contract>"#,
            asgard_window_policy(),
            audit_rounds = ASGARD_AUDIT_MAX_ROUNDS,
        )),
        ChatMessage::user(format!("ORIGINAL TASK (complete):\n{original_task}")),
    ];
    messages.push(ChatMessage::assistant(
        task_contract_checklist.block.clone(),
    ));
    messages.push(ChatMessage::assistant(format!(
        "<selected_trajectory_initial>\n{}\n</selected_trajectory_initial>",
        render_asgard_dossier_messages(selected_trajectory_initial),
    )));
    if !supervisor_history.checkpointed.is_empty() {
        messages.push(ChatMessage::assistant(format!(
            "<supervisor_decision_history checkpointed=\"true\">\n{}\n\
             </supervisor_decision_history>",
            render_asgard_supervisor_history(&supervisor_history.checkpointed),
        )));
    }
    debug_assert_eq!(
        selected_trajectory_windows.len(),
        supervisor_history.selected_windows.len()
    );
    for (index, window) in selected_trajectory_windows.iter().enumerate() {
        messages.push(ChatMessage::user(format!(
            "<selected_trajectory_window_boundary index=\"{index}\" />"
        )));
        let decision = supervisor_history
            .selected_windows
            .get(index)
            .map(render_asgard_supervisor_history_entry)
            .unwrap_or_default();
        messages.push(ChatMessage::assistant(format!(
            "<selected_trajectory_window index=\"{index}\">\n{}\n</selected_trajectory_window>\n{}",
            render_asgard_dossier_messages(window),
            decision,
        )));
    }
    messages.push(ChatMessage::user(format!(
        r#"{candidate_trajectories}

<decision_procedure>
1. Re-read the original task and preserve its exact observable contracts: signatures and argument order, returns, errors, atomicity, compatibility, constraints, and prohibitions.
2. Compare each lane's actual direction, defects, evidence, and recoverability. Choose a provisional winner even if all are incomplete.
3. Decide whether that winner plausibly appears terminal. If so, nominate: complete=true requires only that the endpoint plausibly satisfies the task contracts on the compact trajectories and diffs — the isolated completion review owns terminal adjudication, so do not audit for boundary cases first. If not, investigate only questions that could change the ranking or next-window direction.
4. Treat the completion judgment as a nomination. A concrete defect, contradiction, or plainly missing required work already visible in the dossier means complete=false and becomes targeted next-window advice; do not search for more before nominating.
5. If incomplete, choose the shared horizon and return one distinct compliant strategy per lane to launch. The strategy count determines candidate count. Scale breadth with uncertainty; use one lane for a concrete serial bug fix.
6. Call select_trajectory. Do not continue investigating once you can make the best available judgment.
</decision_procedure>"#,
    )));
    messages
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn asgard_completion_review_messages(
    original_task: &str,
    selected_trajectory_initial: &[ChatMessage],
    selected_trajectory_windows: &[Vec<ChatMessage>],
    supervisor_history: &AsgardSupervisorHistory,
    candidate_count: usize,
    max_candidate_count: usize,
    task_contract_checklist: &AsgardTaskContractChecklist,
    canonical_ledger: &[(usize, AsgardExecutionLedger)],
    terminal_non_test_patch: String,
    terminal_test_patch: String,
    selected_lane: usize,
    selected_candidate_trajectory: String,
    prior_review: Option<&AsgardPriorCompletionReview>,
    prior_review_delta: String,
) -> Vec<ChatMessage> {
    let mut messages = asgard_supervisor_messages(
        original_task,
        selected_trajectory_initial,
        selected_trajectory_windows,
        supervisor_history,
        AsgardCandidateCounts {
            current: candidate_count,
            max: max_candidate_count,
        },
        task_contract_checklist,
        selected_candidate_trajectory.clone(),
    );
    if let Some(prior_review) = prior_review {
        let rows = prior_review
            .rows
            .iter()
            .map(|row| {
                serde_json::json!({
                    "id": row.id,
                    "status": row.status,
                    "evidence": row.evidence,
                })
            })
            .collect::<Vec<_>>();
        let prior_review_message = ChatMessage::assistant(format!(
            "<prior_review_rows window=\"{}\">\n{}\n</prior_review_rows>\n<prior_review_delta>\n{}\n</prior_review_delta>",
            prior_review.window,
            serde_json::to_string_pretty(&rows).expect("prior review rows serialize"),
            prior_review_delta,
        ));
        let insert_at = messages
            .iter()
            .position(|message| {
                asgard_message_text(message).starts_with("<task_contract_checklist")
            })
            .map_or(messages.len(), |position| position + 1);
        messages.insert(insert_at, prior_review_message);
    }
    let canonical_ledger = render_asgard_canonical_ledger(canonical_ledger);
    let terminal_non_test_patch = if terminal_non_test_patch.is_empty() {
        "(no non-test changes)".to_string()
    } else {
        terminal_non_test_patch
    };
    let terminal_test_patch = if terminal_test_patch.is_empty() {
        "(no test changes)".to_string()
    } else {
        terminal_test_patch
    };
    let terminal_review_text = format!(
        r#"{canonical_ledger}
<terminal_non_test_patch cumulative_from_task_baseline="true">
{terminal_non_test_patch}
</terminal_non_test_patch>
<terminal_test_patch cumulative_from_task_baseline="true">
{terminal_test_patch}
</terminal_test_patch>
{selected_candidate_trajectory}

<terminal_completion_review selected_lane="{selected_lane}">
This is an independent completion review of the single would-be-final endpoint shown above. The comparative selection decision and all discarded candidate lanes are intentionally absent. Do not reconstruct, compare, or speculate about them. Keep selected_lane={selected_lane}; your only decision is whether this endpoint actually completes the original task.

The lane trajectory is a mechanically rendered compact record: its tool lines state what each call was, stamped with an id, not what the call output showed. Call view_tool_call with those ids to read the complete arguments and untruncated result of any line. Retrieval is unbudgeted, executes nothing, and is restricted to this lane. Because your evidence rules turn on what a command actually printed and what an assertion actually checks, read the full result before citing it — a compact summary reports that a command ran and its exit code, which is not evidence about the behavior it exercised.

The task_contract_checklist was derived from the task text alone before any candidate work existed. Your select_trajectory call must return one contracts row per checklist id. Keep verified rows terse — one clause citing the ledger entry or quoted line (aim under 150 characters); reserve detailed prose for violated rows, unverified rows, and adverse_condition_evidence. When prior_review_rows are present, re-adjudicate only contracts plausibly affected by the prior_review_delta, plus every contract that was not verified before. For a previously verified contract untouched by the delta, carry it forward: status verified, evidence 'carried-forward: unchanged since window N review'. A carried-forward row must still be listed. Evidence rules, in strength order: (1) Execution evidence — an execution_ledger entry (cite its id) whose command demonstrably exercised the contract and exited 0 at or after the last edit of the implementing files; a green command counts only for behaviors its selected tests actually assert — confirm in the terminal_test_patch that some assertion would fail if the contract were violated; a broad suite pass does not verify a contract no test asserts; a build proves compilation; a race detector proves absence of data races, not liveness. Golden, snapshot, or fixture expectation files created or regenerated by the candidate are candidate-authored assertions: a passing comparison against them proves only that the output matches itself; for exact-output contracts, quote the actual emitted output — from a ledger entry or from the expectation-file content in the patches — and show it satisfies the contract's required shape at its boundaries, including the very first and very last elements of the stream. Evidence must be discriminating: state the most plausible wrong implementation of this contract — the wrong label or classification, the missing boundary element, the partially-correct result — and confirm the cited assertion would catch it; an assertion that only counts results, checks non-emptiness, or matches a substring usually passes under mislabeled or partially-wrong behavior. For a contract naming enumerated values, classifications, or exact labels, quote — for each scenario the contract names — the assertion line or observed ledger output showing that exact value in that exact scenario; an assertion checking a different scenario's value does not transfer. For a contract that combines, maps, or constructs from multiple positional inputs, a green test counts only if its values are pairwise distinguishable and the checked operation non-commutative — a test combining equal values or reducing with numeric addition passes under any positional swap and proves nothing about position. Absent such a test, verify positionally by inspection: quote the complete argument flow from construction to application and argue each input reaches its position; mark the row violated if the quoted flow misroutes any input, and unverified only when neither a discriminating test nor a conclusive flow reading exists. For contracts requiring a behavior across a matrix of backends, dialects, or variants, quoted code showing the new path routes through the same shared machinery as an existing verified sibling is acceptable inspection evidence for the uncovered cells; demand per-cell execution only when the path diverges. For a contract specifying a typed public signature whose shape admits divergent readings — callbacks, generics, overloads, container/element relationships — evidence counts only when a ledger entry type-checks standalone usage authored from the contract's text; the implementation's own tests compiling proves self-consistency, not the contract. A simple signature quoted verbatim from the patch that textually matches the contract is ordinary inspection evidence and needs no type-check run. (2) Inspection evidence — exact quoted lines from the terminal patches showing the contract satisfied; a file name alone is not evidence; use only for contracts fully verifiable by reading. (3) Candidate claims are never evidence.

Unblocking contracts get the strictest treatment. For any contract that an event X (close, shutdown, abort, cancellation, deadline) must unblock, interrupt, or fail a pending operation P, execution evidence counts only if the verifying test performs no action after X that could itself wake P — releasing, closing, erroring, or enqueuing on the awaited stream or channel, sending or receiving data, advancing timers, or completing the awaited resource. A test that wakes P by such means verifies only that P notices a flag after being woken; the row is unverified. Inspection evidence must quote the affirmative wake path X triggers on the already-blocked waiter and show X can reach it while P is blocked: a flag tested inside a read or write loop wakes nothing, and a mutex held across a blocking wait prevents any unblocking path needing that mutex from running. For exact-output contracts, reconstruct the emitted stream from the writer code for the first and last element, not only the middle.

Contracts that carry an adverse_condition are verified only under that condition. Their rows must also fill adverse_condition_evidence: quote the test or code lines showing the stated condition is actually constructed — the operation genuinely pending or blocked when the event fires, the resource genuinely exhausted, the dependency genuinely stalled, the boundary value genuinely used — and list every action the verifying test performs after the triggering event. Conclusive inspection satisfies this too: when the code that handles the stated condition is locally readable — a deep copy at the capture site, a normalization that covers the input family, a guard whose semantics include the edge — quote that handling and state why it covers the condition; demand a constructed test only when the handling is distributed across components, depends on runtime interleaving, or the quoted code's behavior under the condition is not decidable by reading. This cuts both ways: quoted handling that mishandles the condition makes the row violated, exactly as a failing test would. If the cited test never constructs the stated condition, or performs any post-event action that could itself wake or complete the pending operation, the contract is unverified regardless of the test's name or its green result.

Delivery-mechanics contracts (kind "delivery": branch, commit, repository cleanliness) rank below functional contracts: mark such a row unverified rather than violated when evidence is merely absent, record the residual risk in state_summary, and do not block completion on absence alone — but a delivery contract affirmatively contradicted by evidence is violated and blocks completion like any other — except when the contradiction is an environmental failure of the delivery action itself (missing git identity, authentication, network): that is unverified with the residual risk noted, not violated, because no amount of task work can resolve it.

complete=true requires every functional (inspection or execution) row verified. Any violated or unverified functional row means complete=false: keep winner={selected_lane}, choose next_window_steps from 1 to 10, and provide 1 to {max_candidate_count} distinct advices telling the next candidate windows precisely what evidence to produce. The advice count is the next window's candidate count. Use more candidates when materially different fixes or verification strategies are useful under uncertainty; when one concrete defect is a serial prerequisite, return one advice to fix and verify it before exploring new directions. Structure each advice as an ordered work list over the violated and unverified contracts: for each, name the contract id, state what is broken or unproven, give the fix obligation first and the proving command second, and require the candidate to run that command and report its output verbatim after the fix — evidence produced before the fix proves nothing. Additionally, for an unverified execution contract, spell out the concrete scenario, the exact assertion, and instruct the candidate to report the command and output verbatim. When the contract is an unblocking contract, the advised scenario must keep the awaited resource permanently silent after the triggering event: assert that the pending operation rejects or returns within a timeout while nothing else wakes it, and perform any release or cleanup only after that assertion. For an unverified type-shape contract, instruct the candidate to write a standalone usage file authored from the contract text verbatim, run the project's type-checker against it, and report the command and output verbatim. Do not rationalize an unresolved row as rare, cosmetic, pre-existing, timing-dependent, or out of scope; the checklist defines scope. When the ledger and patches genuinely cover every contract, return complete=true and do not invent optional work. Call select_trajectory exactly once and by itself. Do not answer in prose.
</terminal_completion_review>"#,
    );
    let terminal_review = ChatMessage::user(terminal_review_text);
    if let Some(last) = messages.last_mut() {
        *last = terminal_review;
    } else {
        messages.push(terminal_review);
    }
    messages
}

pub(crate) fn render_asgard_canonical_ledger(
    canonical_ledger: &[(usize, AsgardExecutionLedger)],
) -> String {
    canonical_ledger
        .iter()
        .map(|(window, ledger)| {
            format!(
                "<canonical_execution_ledger window=\"{window}\">\n{}\n</canonical_execution_ledger>",
                serde_json::to_string_pretty(ledger).unwrap_or_default(),
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn render_asgard_supervisor_history(entries: &[AsgardSupervisorHistoryEntry]) -> String {
    entries
        .iter()
        .map(render_asgard_supervisor_history_entry)
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn render_asgard_supervisor_history_entry(
    entry: &AsgardSupervisorHistoryEntry,
) -> String {
    format!(
        "<supervisor_decision window=\"{}\" selected_lane=\"{}\">\n{}\n\
         </supervisor_decision>",
        entry.window, entry.winner, entry.state_summary,
    )
}

pub(crate) fn render_asgard_dossier_messages(messages: &[ChatMessage]) -> String {
    let mut rendered = String::new();
    for (index, message) in messages.iter().enumerate() {
        rendered.push_str(&format!(
            "<message index={index} role={:?} name={:?} tool_call_id={:?}>\n",
            message.role, message.name, message.tool_call_id,
        ));
        for part in &message.content {
            match part {
                ChatContentPart::Text { text } => {
                    rendered.push_str("<content>\n");
                    rendered.push_str(text);
                    rendered.push_str("\n</content>\n");
                }
                ChatContentPart::Image { image_url } => {
                    rendered.push_str("<image_url>\n");
                    rendered.push_str(image_url);
                    rendered.push_str("\n</image_url>\n");
                }
            }
        }
        if let Some(reasoning) = &message.reasoning_content {
            rendered.push_str("<reasoning>\n");
            rendered.push_str(reasoning);
            rendered.push_str("\n</reasoning>\n");
        }
        for call in message.tool_calls.iter().flatten() {
            rendered.push_str(&format!(
                "<tool_call id={:?} name={:?}>\n",
                call.id, call.function.name,
            ));
            rendered.push_str(&call.function.arguments);
            rendered.push_str("\n</tool_call>\n");
        }
        rendered.push_str("</message>\n");
    }
    rendered
}

/// Renders a candidate window into the compact, mechanically derived record the
/// supervisor reviews.
///
/// Every entry becomes a minimal line and every tool result carries a handle
/// that `view_tool_call` resolves back to the complete arguments and result, so
/// compression is lossless on demand rather than lossy by inference. This runs
/// unconditionally at any window size: an LLM asked to summarize a window
/// reports work that did not happen often enough to be unusable as supervisor
/// evidence (see research/asgard_summarizer_bench/RESULTS.md).
///
/// Returns the handoff messages plus the raw and compacted byte counts.
pub(crate) fn asgard_deterministic_candidate_handoff(
    window: usize,
    lane: usize,
    window_messages: &[ChatMessage],
    current_plan: Option<&crate::plan::UpdatePlanArgs>,
) -> (Vec<ChatMessage>, usize, usize) {
    let raw = render_asgard_dossier_messages(window_messages);
    let packed = crate::asgard::render_window_compact(window, lane, window_messages);
    let plan = current_plan
        .and_then(|plan| serde_json::to_string_pretty(plan).ok())
        .unwrap_or_else(|| "(no active plan)".to_string());
    let handoff = ChatMessage::assistant(format!(
        "<candidate_window_handoff format=\"compact_deterministic\" lane=\"{lane}\" duplicate_encoding=\"back_reference\">\n\
         This is a mechanically rendered trajectory, not a candidate-authored summary: every line \
         is a pure function of what this lane actually did. Tool results appear as deterministic \
         one-line summaries carrying an id; call view_tool_call with those ids to read the complete \
         arguments and untruncated result. The same ids key this lane's execution_ledger entries, \
         whose output_tail holds only the last 400 bytes of a command's output. A len attribute gives the full character count of text \
         that was shortened. A tool line with exact_duplicate_of repeats a byte-identical earlier \
         result and is not independent confirmation of anything.\n\
         <current_plan>\n{plan}\n</current_plan>\n\
         <candidate_window>\n{packed}</candidate_window>\n\
         </candidate_window_handoff>"
    ));
    (vec![handoff], raw.len(), packed.len())
}

pub(crate) fn asgard_evidence_fingerprint(entry: &AsgardLedgerEntry) -> String {
    format!(
        "{}\u{0}{}\u{0}{}\u{0}{}",
        entry.command,
        entry
            .exit_code
            .map_or_else(|| "none".to_string(), |code| code.to_string()),
        entry.output_bytes,
        entry.output_sha256,
    )
}

pub(crate) fn asgard_sha256(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

pub(crate) fn asgard_review_evidence_fingerprints(
    canonical_ledger: &[(usize, AsgardExecutionLedger)],
    current_ledger: &AsgardExecutionLedger,
) -> BTreeSet<String> {
    canonical_ledger
        .iter()
        .flat_map(|(_, ledger)| &ledger.entries)
        .chain(&current_ledger.entries)
        .map(asgard_evidence_fingerprint)
        .collect()
}

pub(crate) fn asgard_has_new_read_only_observation(messages: &[ChatMessage]) -> bool {
    messages.iter().any(|message| {
        message.tool_calls.iter().flatten().any(|call| {
            asgard_is_audit_tool(&call.function.name)
                || matches!(
                    call.function.name.as_str(),
                    "read_file" | "grep_search" | "list_directory"
                )
        })
    })
}

#[derive(Debug)]
pub(crate) struct AsgardReviewDelta {
    pub(crate) patch: Vec<u8>,
    pub(crate) production_bytes: usize,
    pub(crate) test_bytes: usize,
    pub(crate) new_evidence_count: usize,
    pub(crate) has_read_only_observation: bool,
}

impl AsgardReviewDelta {
    pub(crate) fn is_material(&self) -> bool {
        self.production_bytes > 0
            || self.test_bytes > 0
            || self.new_evidence_count > 0
            || self.has_read_only_observation
    }
}

pub(crate) fn asgard_completion_review_delta(
    prior: &AsgardPriorCompletionReview,
    candidate: &AsgardCandidate,
    repository: &crate::asgard::CandidateRepository,
    canonical_ledger: &[(usize, AsgardExecutionLedger)],
) -> anyhow::Result<AsgardReviewDelta> {
    let patch = crate::asgard::capture_patch_since(
        &repository.root,
        &repository.base_commit,
        &prior.reviewed_patch,
    )?;
    let (production, tests) = asgard_patch_surfaces(&patch);
    let current_evidence =
        asgard_review_evidence_fingerprints(canonical_ledger, &candidate.window_ledger);
    Ok(AsgardReviewDelta {
        patch,
        production_bytes: production.len(),
        test_bytes: tests.len(),
        new_evidence_count: current_evidence
            .difference(&prior.evidence_fingerprints)
            .count(),
        has_read_only_observation: asgard_has_new_read_only_observation(&candidate.window_messages),
    })
}

pub(crate) fn trace_asgard_phase_usage(
    phase: &str,
    model: &str,
    window: Option<usize>,
    lane: Option<usize>,
    usage: crate::llm_client::TokenUsage,
) {
    crate::trace_logging::append_trace_record(serde_json::json!({
        "type": "asgard_phase_usage",
        "phase": phase,
        "model": model,
        "window": window,
        "lane": lane,
        "usage": {
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "thought": usage.thought_tokens,
            "cachedRead": usage.cached_read_tokens,
            "cachedWrite": usage.cached_write_tokens,
        },
    }));
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn trace_asgard_phase_turn_usage(
    phase: &str,
    model: &str,
    window: Option<usize>,
    lane: Option<usize>,
    turn: usize,
    turn_kind: &str,
    message_bytes: usize,
    usage: crate::llm_client::TokenUsage,
) {
    crate::trace_logging::append_trace_record(serde_json::json!({
        "type": "asgard_phase_turn_usage",
        "phase": phase,
        "model": model,
        "window": window,
        "lane": lane,
        "turn": turn,
        "turn_kind": turn_kind,
        "message_bytes": message_bytes,
        "usage": {
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "thought": usage.thought_tokens,
            "cachedRead": usage.cached_read_tokens,
            "cachedWrite": usage.cached_write_tokens,
        },
    }));
}

pub(crate) fn asgard_extract_task_contracts_tool() -> ToolDefinition {
    ToolDefinition {
        r#type: "function".to_string(),
        function: FunctionDef {
            name: "extract_task_contracts".to_string(),
            description: "Extract the explicit, externally checkable contract checklist from a software task description, before any implementation exists.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["contracts"],
                "properties": {
                    "contracts": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 75,
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "required": ["id", "kind", "text", "adverse_condition"],
                            "properties": {
                                "id": { "type": "string", "pattern": "^C[0-9]+$" },
                                "kind": { "type": "string", "enum": ["inspection", "execution", "delivery"] },
                                "text": { "type": "string", "minLength": 1, "maxLength": 600 },
                                "adverse_condition": { "type": ["string", "null"], "maxLength": 400 }
                            }
                        }
                    }
                }
            }),
        },
    }
}

pub(crate) fn parse_asgard_task_contracts(
    value: &serde_json::Value,
) -> anyhow::Result<Vec<AsgardTaskContract>> {
    let contracts = value
        .get("contracts")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("contract extraction is missing `contracts` array"))?;
    let mut parsed = Vec::with_capacity(contracts.len());
    for (index, contract) in contracts.iter().enumerate() {
        let id = contract
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("contract {index} is missing string `id`"))?;
        let kind = contract
            .get("kind")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("contract {index} is missing string `kind`"))?;
        let text = contract
            .get("text")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("contract {index} is missing string `text`"))?;
        let adverse_condition = contract
            .get("adverse_condition")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|condition| !condition.is_empty())
            .map(str::to_string);
        parsed.push(AsgardTaskContract {
            id: id.to_string(),
            kind: kind.to_string(),
            text: text.to_string(),
            adverse_condition,
        });
    }
    anyhow::ensure!(!parsed.is_empty(), "contract checklist is empty");
    Ok(parsed)
}

pub(crate) fn asgard_render_task_contract_checklist(
    contracts: Vec<AsgardTaskContract>,
) -> AsgardTaskContractChecklist {
    let block = format!(
        "<task_contract_checklist derived_from_task_text_only=\"true\" independent_of_candidates=\"true\">\n{}\n</task_contract_checklist>",
        serde_json::to_string_pretty(&serde_json::json!({ "contracts": contracts }))
            .expect("contracts serialize")
    );
    AsgardTaskContractChecklist { contracts, block }
}

pub(crate) fn asgard_empty_task_contract_checklist() -> AsgardTaskContractChecklist {
    asgard_render_task_contract_checklist(Vec::new())
}

pub(crate) async fn run_asgard_task_contract_extraction(
    llm: &Arc<dyn crate::llm_client::LlmBackend>,
    model: &str,
    idle_timeout: IdleTimeouts,
    cancel: tokio_util::sync::CancellationToken,
    original_task: &str,
) -> (AsgardTaskContractChecklist, crate::llm_client::TokenUsage) {
    let mut messages = vec![
        ChatMessage::system(ASGARD_CONTRACT_EXTRACTION_PROMPT),
        ChatMessage::user(format!("ORIGINAL TASK (complete):\n{original_task}")),
        ChatMessage::user("Call extract_task_contracts exactly once now. Do not answer in prose."),
    ];
    let tools = vec![asgard_extract_task_contracts_tool()];
    let mut usage = crate::llm_client::TokenUsage::default();
    let mut last_invalid_response = None;

    for step in 1..=ASGARD_CONTRACT_EXTRACTION_MAX_ATTEMPTS {
        let request_bytes = render_asgard_dossier_messages(&messages).len();
        let response = stream_chat_no_visible_output_with_retry(
            llm.as_ref(),
            "extracting Asgard task contracts",
            &cancel,
            || StreamChatRequest {
                model: model.to_string(),
                messages: messages.clone(),
                tools: Some(tools.clone()),
                reasoning_effort: None,
                service_tier: None,
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: cancel.clone(),
                idle_timeouts: idle_timeout,
            },
        )
        .await;
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                last_invalid_response = Some(error);
                if step < ASGARD_CONTRACT_EXTRACTION_MAX_ATTEMPTS {
                    let detail = last_invalid_response
                        .as_ref()
                        .map(|error| format!(" Your previous response was invalid: {error}."))
                        .unwrap_or_default();
                    messages.push(ChatMessage::user(format!(
                        "You must call extract_task_contracts now.{detail} Do not answer in prose."
                    )));
                }
                continue;
            }
        };
        let turn_usage = response.usage();
        trace_asgard_phase_turn_usage(
            "task_contract",
            model,
            None,
            None,
            step,
            "extraction",
            request_bytes,
            turn_usage,
        );
        usage.add(turn_usage);

        match response {
            LlmResponse::Text {
                text,
                reasoning_content,
                ..
            } => messages.push(ChatMessage::assistant_with_reasoning(
                text,
                reasoning_content,
            )),
            LlmResponse::ToolCalls {
                text,
                reasoning_content,
                calls,
                ..
            } => {
                if calls.len() == 1 && calls[0].function.name == "extract_task_contracts" {
                    match crate::tool_arguments::normalize_tool_arguments(
                        &calls[0].function.arguments,
                    ) {
                        Ok(arguments) => match parse_asgard_task_contracts(&arguments.value) {
                            Ok(contracts) => {
                                return (asgard_render_task_contract_checklist(contracts), usage);
                            }
                            Err(error) => last_invalid_response = Some(error),
                        },
                        Err(error) => {
                            last_invalid_response = Some(anyhow::anyhow!(
                                "contract extractor emitted invalid extract_task_contracts arguments: {error}"
                            ));
                        }
                    }
                } else {
                    last_invalid_response = Some(anyhow::anyhow!(
                        "contract extractor must call extract_task_contracts exactly once and no other tool"
                    ));
                }
                if !text.is_empty() || reasoning_content.as_deref().is_some_and(|s| !s.is_empty()) {
                    messages.push(ChatMessage::assistant_with_reasoning(
                        text,
                        reasoning_content,
                    ));
                }
            }
        }

        if step < ASGARD_CONTRACT_EXTRACTION_MAX_ATTEMPTS {
            let detail = last_invalid_response
                .as_ref()
                .map(|error| format!(" Your previous response was invalid: {error}."))
                .unwrap_or_default();
            messages.push(ChatMessage::user(format!(
                "You must call extract_task_contracts now.{detail} Do not answer in prose."
            )));
        }
    }

    tracing::warn!(
        model,
        "Asgard task contract extraction failed; continuing with empty checklist: {:#}",
        last_invalid_response.unwrap_or_else(|| {
            anyhow::anyhow!(
                "contract extractor did not call extract_task_contracts after {ASGARD_CONTRACT_EXTRACTION_MAX_ATTEMPTS} steps"
            )
        })
    );
    (asgard_empty_task_contract_checklist(), usage)
}

pub(crate) fn asgard_advise_trajectories_tool(max_candidate_count: usize) -> ToolDefinition {
    ToolDefinition {
        r#type: "function".to_string(),
        function: FunctionDef {
            name: "advise_trajectories".to_string(),
            description: "Choose the first shared Asgard window length and provide distinct advisory strategies for all candidate trajectories. This is the terminal initial supervisor action and must be called exactly once.".to_string(),
            parameters: serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["next_window_steps", "state_summary", "advices"],
            "properties": {
                "next_window_steps": {
                    "type": "integer",
                    "minimum": ASGARD_MIN_WINDOW_STEPS,
                    "maximum": ASGARD_MAX_WINDOW_STEPS,
                },
                "state_summary": {
                    "type": "string",
                    "minLength": 1,
                },
                "advices": {
                    "type": "array",
                    "minItems": ASGARD_MIN_CANDIDATES,
                    "maxItems": max_candidate_count,
                    "uniqueItems": true,
                    "items": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["strategy", "scope_basis"],
                        "properties": {
                            "strategy": { "type": "string", "minLength": 1 },
                            "scope_basis": { "type": "string", "minLength": 1 },
                        },
                    },
                },
            },
            }),
        },
    }
}

pub(crate) fn asgard_select_trajectory_tool(
    candidate_count: usize,
    max_candidate_count: usize,
) -> ToolDefinition {
    ToolDefinition {
        r#type: "function".to_string(),
        function: FunctionDef {
            name: "select_trajectory".to_string(),
            description: "Select the canonical Asgard trajectory and decide whether the original task is complete. This is the terminal supervisor decision and must be called exactly once.".to_string(),
            parameters: serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["winner", "complete", "state_summary", "advices"],
            "properties": {
                "winner": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": candidate_count.saturating_sub(1),
                },
                "complete": { "type": "boolean" },
                "next_window_steps": {
                    "type": "integer",
                    "minimum": ASGARD_MIN_WINDOW_STEPS,
                    "maximum": ASGARD_MAX_WINDOW_STEPS,
                    "description": "Required when complete=false; omitted or ignored when complete=true.",
                },
                "state_summary": {
                    "type": "string",
                    "minLength": 1,
                },
                "advices": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": max_candidate_count,
                    "uniqueItems": true,
                    "items": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["strategy", "scope_basis"],
                        "properties": {
                            "strategy": { "type": "string", "minLength": 1 },
                            "scope_basis": { "type": "string", "minLength": 1 },
                        },
                    },
                },
                "contracts": {
                    "type": "array",
                    "maxItems": 90,
                    "items": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["id", "status", "evidence"],
                        "properties": {
                            "id": { "type": "string" },
                            "status": { "type": "string", "enum": ["verified", "violated", "unverified"] },
                            "evidence": { "type": "string", "minLength": 1, "maxLength": 1500 },
                            "adverse_condition_evidence": {
                                "type": "string",
                                "maxLength": 1200,
                                "description": "Required when the checklist entry carries an adverse_condition: quote the test or code lines showing that exact condition is constructed, and list every action the verifying test performs after the triggering event."
                            }
                        }
                    }
                },
            },
            }),
        },
    }
}

pub(crate) fn render_asgard_verified_at_window_start(
    ledger: &AsgardExecutionLedger,
) -> Option<String> {
    let mut seen = HashSet::new();
    let mut commands = ledger
        .entries
        .iter()
        .rev()
        .filter(|entry| entry.exit_code == Some(0))
        .filter_map(|entry| {
            if seen.insert(entry.command.as_str()) {
                Some(entry.command.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    commands.reverse();
    if commands.is_empty() {
        return None;
    }
    let start = commands.len().saturating_sub(15);
    let mut rendered = String::from(
        "<verified_at_window_start>\n\
         The following commands ran with exit code 0 on exactly this starting tree at the end of the previous window. Do not re-run them unless your edits invalidate what they exercised; run the narrowest check that covers your actual change instead.",
    );
    for command in &commands[start..] {
        rendered.push_str("\n- ");
        rendered.push_str(command);
    }
    rendered.push_str("\n</verified_at_window_start>");
    Some(rendered)
}

pub(crate) fn asgard_advice_message(
    lane: usize,
    advice: &str,
    verified_block: Option<&str>,
) -> ChatMessage {
    let advice = format!(
        "<asgard_next_window_advice lane=\"{lane}\">\n{advice}\n\
         </asgard_next_window_advice>\n\
         Treat this as advisory strategy for continuing the original task normally. \
         Do not stop because of an Asgard window boundary; Anvil controls those boundaries. \
         You may call update_plan if this strategy changes or clarifies the best plan for the task. \
         Unless the original task requires it, do not change unrelated dependencies, build \
         configuration, warning policy, tests, or test selection to hide a pre-existing, \
         environmental, dependency-audit, or harness failure, and do not chase it by changing SDKs, \
         MSBuild or Gradle properties, toolchains, classpaths, build daemons, wrappers, generated build \
         tooling, or checkout paths. Do not assume verification is blocked merely because it has not \
         been attempted. Run an available focused verifier; if an attempted verifier demonstrates an \
         environmental blocker, try an already-available equivalent or perform a bounded audit and \
         report the exact evidence without changing infrastructure. Treat a verifier timeout as an \
         inconclusive result and try a narrower task-relevant check. A successful test command that \
         discovered or selected zero tests is also inconclusive, not passing verification. Tests added \
         by this trajectory are useful diagnostics but do not replace applicable pre-existing or \
         boundary-level verification. Do not dismiss a failing existing \
         test as pre-existing, flaky, or unrelated without concrete baseline or subsequent passing \
         evidence, especially when it exercises code changed by this trajectory."
    );
    let text = match verified_block {
        Some(block) => format!("{block}\n{advice}"),
        None => advice,
    };
    ChatMessage::user(text)
}

pub(crate) fn render_asgard_lane_advices(advices: &[Option<String>]) -> String {
    advices
        .iter()
        .enumerate()
        .map(|(lane, advice)| {
            format!(
                "  lane {}: {}",
                lane + 1,
                advice.as_deref().unwrap_or("<missing advice>")
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn asgard_take_window_messages(
    messages: &[ChatMessage],
    trajectory_message_start: usize,
) -> Vec<ChatMessage> {
    messages
        .get(trajectory_message_start..)
        .unwrap_or_default()
        .to_vec()
}

/// Builds the execution ledger for one lane's window.
///
/// Entry ids are `view_tool_call` handles, not a private `L{n}` sequence: the
/// supervisor cites ledger ids as execution evidence and separately retrieves
/// full tool results by handle, and two vocabularies for the same shell call
/// only invited it to pass one where the other belonged.
pub(crate) fn asgard_extract_execution_ledger(
    window: usize,
    lane: usize,
    window_messages: &[ChatMessage],
) -> AsgardExecutionLedger {
    let exit_code_regex =
        regex::Regex::new(r"Exit code: (-?\d+)|Command completed with exit code (-?\d+)")
            .expect("valid exit-code regex");
    let mut entries = Vec::new();
    let mut edit_steps = Vec::new();
    let mut total_shell_commands = 0usize;

    for (step, message) in window_messages.iter().enumerate() {
        if message.role != "assistant" {
            continue;
        }
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for call in tool_calls {
            let name = call.function.name.as_str();
            if !matches!(name, "run_shell_command" | "edit" | "write_file") {
                continue;
            }
            let Ok(arguments) = serde_json::from_str::<serde_json::Value>(&call.function.arguments)
            else {
                continue;
            };
            let Some(arguments) = arguments.as_object() else {
                continue;
            };
            if name == "run_shell_command" {
                total_shell_commands += 1;
                let mut command = arguments
                    .get("command")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if command.chars().count() > 500 {
                    command = format!("{}…", command.chars().take(500).collect::<String>());
                }
                // The absolute index of the *result* message, not the assistant
                // message holding the call: that is what a handle addresses.
                let matching_result = window_messages
                    .get(step + 1..)
                    .unwrap_or_default()
                    .iter()
                    .enumerate()
                    .find(|(_, later)| {
                        later.role == "tool" && later.tool_call_id.as_deref() == Some(&call.id)
                    })
                    .map(|(offset, result)| (step + 1 + offset, result));
                let result_index = matching_result.map(|(index, _)| index);
                let (exit_code, output_bytes, output_sha256, output_tail) = match matching_result {
                    Some((_, result)) => {
                        let result_text = asgard_message_text(result);
                        let exit_code = exit_code_regex
                            .captures_iter(&result_text)
                            .last()
                            .and_then(|captures| captures.get(1).or_else(|| captures.get(2)))
                            .and_then(|capture| capture.as_str().parse::<i32>().ok())
                            .or(Some(0));
                        let filtered = result_text
                            .lines()
                            .filter(|line| !line.contains("[WARNING] OS sandbox unavailable"))
                            .collect::<Vec<_>>()
                            .join("\n");
                        (
                            exit_code,
                            filtered.len(),
                            asgard_sha256(filtered.as_bytes()),
                            asgard_utf8_suffix(&filtered, 400).to_string(),
                        )
                    }
                    None => (None, 0, asgard_sha256(&[]), String::new()),
                };
                entries.push(AsgardLedgerEntry {
                    // A command whose result never arrived has nothing to
                    // retrieve; the handle points at the call site and
                    // resolution reports the absence rather than inventing one.
                    id: crate::asgard::asgard_tool_handle(
                        window,
                        lane,
                        result_index.unwrap_or(step),
                    ),
                    step,
                    command,
                    exit_code,
                    output_bytes,
                    output_sha256,
                    output_tail,
                });
            } else {
                edit_steps.push(AsgardLedgerEdit {
                    step,
                    file: arguments
                        .get("file_path")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                });
            }
        }
    }

    let entries_truncated = entries.len() > 120;
    if entries_truncated {
        let mut capped = Vec::with_capacity(120);
        capped.extend_from_slice(&entries[..20]);
        capped.extend_from_slice(&entries[entries.len() - 100..]);
        entries = capped;
    }
    if edit_steps.len() > 150 {
        edit_steps = edit_steps[edit_steps.len() - 150..].to_vec();
    }

    AsgardExecutionLedger {
        entries,
        edit_steps,
        total_shell_commands,
        entries_truncated,
    }
}

pub(crate) fn asgard_utf8_suffix(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut start = text.len() - max_bytes;
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

pub(crate) fn asgard_message_text(message: &ChatMessage) -> String {
    message
        .content
        .iter()
        .filter_map(|part| match part {
            ChatContentPart::Text { text } => Some(text.as_str()),
            ChatContentPart::Image { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn asgard_patch_test_inventory(patch: &[u8]) -> (Vec<String>, Vec<String>) {
    let patch = String::from_utf8_lossy(patch);
    let mut created = Vec::new();
    let mut modified = Vec::new();
    let mut current_path: Option<String> = None;
    let mut current_is_new = false;

    for line in patch.lines() {
        if let Some(rest) = line.strip_prefix("diff --git a/") {
            asgard_flush_patch_test_path(
                &mut current_path,
                &mut current_is_new,
                &mut created,
                &mut modified,
            );
            current_path = rest.split_once(" b/").map(|(path, _)| path.to_string());
        } else if line.starts_with("new file mode ") {
            current_is_new = true;
        }
    }
    asgard_flush_patch_test_path(
        &mut current_path,
        &mut current_is_new,
        &mut created,
        &mut modified,
    );
    (created, modified)
}

pub(crate) fn asgard_patch_production_inventory(patch: &[u8]) -> Vec<String> {
    let patch = String::from_utf8_lossy(patch);
    let mut paths = BTreeSet::new();
    for line in patch.lines() {
        if let Some(rest) = line.strip_prefix("diff --git a/") {
            let Some((old_path, new_path)) = rest.split_once(" b/") else {
                continue;
            };
            for path in [old_path, new_path.trim_end()] {
                if !asgard_is_test_path(path) {
                    paths.insert(path.to_string());
                }
            }
        }
    }
    paths.into_iter().collect()
}

pub(crate) fn asgard_patch_surfaces(patch: &[u8]) -> (String, String) {
    let patch = String::from_utf8_lossy(patch);
    let patch = patch.as_ref();
    let section_regex = regex::Regex::new(r"(?m)^diff --git a/").expect("valid diff regex");
    let starts = section_regex
        .find_iter(patch)
        .map(|found| found.start())
        .collect::<Vec<_>>();
    let mut production = String::new();
    let mut test = String::new();

    for (index, start) in starts.iter().copied().enumerate() {
        let end = starts.get(index + 1).copied().unwrap_or(patch.len());
        let section = &patch[start..end];
        let Some(rest) = section.strip_prefix("diff --git a/") else {
            continue;
        };
        let Some((old_path, new_path)) = rest.split_once(" b/") else {
            continue;
        };
        let new_path = new_path.lines().next().unwrap_or_default();
        if asgard_is_test_path(old_path) || asgard_is_test_path(new_path) {
            test.push_str(section);
        } else {
            production.push_str(section);
        }
    }

    (production, test)
}

pub(crate) fn asgard_flush_patch_test_path(
    path: &mut Option<String>,
    is_new: &mut bool,
    created: &mut Vec<String>,
    modified: &mut Vec<String>,
) {
    if let Some(path) = path.take().filter(|path| asgard_is_test_path(path)) {
        if *is_new {
            created.push(path);
        } else {
            modified.push(path);
        }
    }
    *is_new = false;
}

pub(crate) fn asgard_is_test_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    let file = lower.rsplit('/').next().unwrap_or(lower.as_str());
    lower
        .split('/')
        .any(|segment| matches!(segment, "test" | "tests" | "__tests__" | "integrationtest"))
        || file.ends_with("_test.go")
        || file.ends_with("test.java")
        || file.ends_with("tests.cs")
        || file.ends_with("test.cs")
        || file.contains(".test.")
        || file.contains(".spec.")
}

pub(crate) fn parse_asgard_supervisor_decision_with_max(
    text: &str,
    candidate_count: usize,
    max_candidate_count: usize,
) -> anyhow::Result<AsgardSupervisorDecision> {
    for (start, _) in text.match_indices('{') {
        let Some(Ok(value)) = serde_json::Deserializer::from_str(&text[start..])
            .into_iter::<serde_json::Value>()
            .next()
        else {
            continue;
        };
        let Some(winner) = value.get("winner").and_then(serde_json::Value::as_u64) else {
            continue;
        };
        let winner = winner as usize;
        if winner >= candidate_count {
            continue;
        }
        let Some(complete) = value.get("complete").and_then(serde_json::Value::as_bool) else {
            continue;
        };
        let Some(state_summary) = value
            .get("state_summary")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|summary| !summary.is_empty())
        else {
            continue;
        };
        let Some(raw_advices) = value.get("advices").and_then(serde_json::Value::as_array) else {
            continue;
        };
        if complete {
            if !raw_advices.is_empty() {
                continue;
            }
            let contracts = parse_asgard_contract_rows(&value);
            return Ok(AsgardSupervisorDecision {
                winner,
                complete,
                advices: vec![None; candidate_count],
                next_window_steps: None,
                state_summary: state_summary.to_string(),
                contracts,
            });
        }
        let Some(next_window_steps) = parse_asgard_next_window_steps(&value) else {
            continue;
        };
        let Some(advices) = parse_asgard_incomplete_advices(raw_advices, max_candidate_count)
        else {
            continue;
        };
        let contracts = parse_asgard_contract_rows(&value);
        return Ok(AsgardSupervisorDecision {
            winner,
            complete,
            advices,
            next_window_steps: Some(next_window_steps),
            state_summary: state_summary.to_string(),
            contracts,
        });
    }
    anyhow::bail!(
        "Asgard supervisor returned neither a valid completed winner nor a winner plus 1-{max_candidate_count} distinct advices and a {ASGARD_MIN_WINDOW_STEPS}-{ASGARD_MAX_WINDOW_STEPS} next_window_steps"
    )
}

pub(crate) fn parse_asgard_initial_advice(
    text: &str,
    max_candidate_count: usize,
) -> anyhow::Result<AsgardSupervisorInitialAdvice> {
    for (start, _) in text.match_indices('{') {
        let Some(Ok(value)) = serde_json::Deserializer::from_str(&text[start..])
            .into_iter::<serde_json::Value>()
            .next()
        else {
            continue;
        };
        let Some(next_window_steps) = parse_asgard_next_window_steps(&value) else {
            continue;
        };
        let Some(state_summary) = value
            .get("state_summary")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|summary| !summary.is_empty())
        else {
            continue;
        };
        let Some(raw_advices) = value.get("advices").and_then(serde_json::Value::as_array) else {
            continue;
        };
        let Some(advices) = parse_asgard_incomplete_advices(raw_advices, max_candidate_count)
        else {
            continue;
        };
        return Ok(AsgardSupervisorInitialAdvice {
            advices,
            next_window_steps,
            state_summary: state_summary.to_string(),
        });
    }
    anyhow::bail!(
        "Asgard supervisor returned no valid initial advice with 1-{max_candidate_count} distinct advices and a {ASGARD_MIN_WINDOW_STEPS}-{ASGARD_MAX_WINDOW_STEPS} next_window_steps"
    )
}

pub(crate) fn parse_asgard_contract_rows(
    value: &serde_json::Value,
) -> Option<Vec<AsgardContractRow>> {
    value.get("contracts").map(|contracts| {
        contracts
            .as_array()
            .map(|rows| {
                rows.iter()
                    .filter_map(|row| {
                        Some(AsgardContractRow {
                            id: row.get("id")?.as_str()?.to_string(),
                            status: row.get("status")?.as_str()?.to_string(),
                            evidence: row.get("evidence")?.as_str()?.to_string(),
                            adverse_condition_evidence: row
                                .get("adverse_condition_evidence")
                                .and_then(serde_json::Value::as_str)
                                .map(str::to_string),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    })
}

pub(crate) fn asgard_validate_contract_rows(
    decision: &AsgardSupervisorDecision,
    checklist: &[AsgardTaskContract],
    carry_forward_allowed: bool,
) -> Vec<String> {
    let mut rows_by_id: HashMap<&str, Vec<&AsgardContractRow>> = HashMap::new();
    for row in decision.contracts.as_deref().unwrap_or_default() {
        rows_by_id.entry(row.id.as_str()).or_default().push(row);
    }

    let mut violations = Vec::new();
    for entry in checklist {
        let id = &entry.id;
        match rows_by_id.get(id.as_str()).map(Vec::as_slice) {
            None | Some([]) => violations.push(format!("contract {id} is missing")),
            Some([row]) => {
                if entry.kind == "delivery" {
                    // Delivery-mechanics obligations rank below functional
                    // contracts: a filled, non-violated row is sufficient.
                    if row.status == "violated" {
                        violations.push(format!("contract {id} is violated"));
                    }
                    continue;
                }
                if row.status != "verified" {
                    violations.push(format!("contract {id} is {}", row.status));
                } else if row.evidence.trim().is_empty() {
                    violations.push(format!("contract {id} has empty evidence"));
                } else if entry.kind == "execution"
                    && !asgard_cites_ledger_entry(row)
                    && !(carry_forward_allowed
                        && row.evidence.trim_start().starts_with("carried-forward:"))
                {
                    violations.push(format!(
                        "contract {id} is kind=execution but its row cites no \
                         execution_ledger entry id (w<window>l<lane>m<index>); \
                         execution contracts cannot be \
                         verified by inspection alone — cite the ledger entry that \
                         exercised this behavior, or mark the row unverified"
                    ));
                } else if entry.adverse_condition.is_some()
                    && row
                        .adverse_condition_evidence
                        .as_deref()
                        .is_none_or(|evidence| evidence.trim().is_empty())
                {
                    let condition: String = entry
                        .adverse_condition
                        .as_deref()
                        .unwrap_or_default()
                        .chars()
                        .take(120)
                        .collect();
                    violations.push(format!(
                        "contract {id} carries adverse_condition ({condition:?}) but its \
                         row has no adverse_condition_evidence showing that condition \
                         constructed"
                    ));
                }
            }
            Some(_) => violations.push(format!("contract {id} appears more than once")),
        }
    }
    violations
}

/// Whether a contract row cites an execution_ledger entry by id.
///
/// Ledger ids are `view_tool_call` handles (`w<window>l<lane>m<index>`). This
/// gates every execution-kind contract, so it must track the id format exactly:
/// a detector still looking for the old `L<n>` shape would reject every
/// execution row ever written and fail the completion gate outright.
pub(crate) fn asgard_cites_ledger_entry(row: &AsgardContractRow) -> bool {
    let cited = |text: &str| {
        text.match_indices('w').any(|(index, _)| {
            text[..index]
                .chars()
                .next_back()
                .is_none_or(|previous| !previous.is_alphanumeric())
                && crate::asgard::parse_asgard_tool_handle(
                    text[index..]
                        .split(|character: char| !character.is_ascii_alphanumeric())
                        .next()
                        .unwrap_or_default(),
                )
                .is_some()
        })
    };
    cited(&row.evidence) || row.adverse_condition_evidence.as_deref().is_some_and(cited)
}

pub(crate) fn asgard_contract_violation_message(violations: &[String]) -> String {
    format!(
        "complete=true was not accepted: {}. Every checklist contract must have a verified row citing concrete evidence (execution_ledger entry ids, or quoted code). If you cannot cite such evidence, return complete=false with advices that direct the next candidate windows to produce exactly that evidence.",
        violations.join("; ")
    )
}

pub(crate) fn parse_asgard_next_window_steps(value: &serde_json::Value) -> Option<usize> {
    let steps = value.get("next_window_steps")?.as_u64()? as usize;
    (ASGARD_MIN_WINDOW_STEPS..=ASGARD_MAX_WINDOW_STEPS)
        .contains(&steps)
        .then_some(steps)
}

pub(crate) fn parse_asgard_incomplete_advices(
    raw_advices: &[serde_json::Value],
    max_candidate_count: usize,
) -> Option<Vec<Option<String>>> {
    if !(ASGARD_MIN_CANDIDATES..=max_candidate_count).contains(&raw_advices.len()) {
        return None;
    }
    let count = raw_advices.len();
    let advices: Vec<_> = raw_advices
        .iter()
        .filter_map(serde_json::Value::as_object)
        .filter(|advice| {
            advice
                .get("scope_basis")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|basis| !basis.trim().is_empty())
        })
        .filter_map(|advice| advice.get("strategy"))
        .filter_map(serde_json::Value::as_str)
        .map(str::trim)
        .map(str::to_string)
        .collect();
    if advices.len() != count || advices.iter().any(String::is_empty) {
        return None;
    }
    let distinct: HashSet<_> = advices.iter().map(|advice| advice.to_lowercase()).collect();
    (distinct.len() == count).then(|| advices.into_iter().map(Some).collect())
}

pub(crate) fn rewrite_asgard_cwd(messages: &mut [ChatMessage], from: &Path, to: &Path) {
    // `PathBuf::join("")` preserves a trailing separator. That is how an
    // Asgard session rooted at the repository itself is represented, while
    // tool results and model-authored commands normally omit the separator.
    // Rewrite the path stem so both spellings are canonicalized.
    let from_display = from.display().to_string();
    let from = from_display.trim_end_matches(['/', '\\']);
    let from = if from.is_empty() {
        from_display.as_str()
    } else {
        from
    };
    let to_display = to.display().to_string();
    let to = to_display.trim_end_matches(['/', '\\']);
    let to = if to.is_empty() {
        to_display.as_str()
    } else {
        to
    };
    for message in messages {
        for part in &mut message.content {
            if let ChatContentPart::Text { text } = part {
                *text = text.replace(from, to);
            }
        }
        if let Some(tool_calls) = &mut message.tool_calls {
            for call in tool_calls {
                call.function.arguments = call.function.arguments.replace(from, to);
            }
        }
        if let Some(reasoning) = &mut message.reasoning_content {
            *reasoning = reasoning.replace(from, to);
        }
    }
}

pub(crate) fn cleanup_asgard_repositories(repositories: &[crate::asgard::CandidateRepository]) {
    for repository in repositories {
        crate::asgard::remove_candidate_repository(repository);
    }
}

pub(crate) fn asgard_failure(error: anyhow::Error) -> crate::tool_loop::LoopOutcome {
    crate::tool_loop::LoopOutcome {
        response: format!("\n**Error:** Asgard failed: {error:#}\n"),
        tool_exchanges: Vec::new(),
        replay_events: Vec::new(),
        usage: crate::llm_client::TokenUsage::default(),
        stop: crate::tool_loop::LoopStop::Failed(crate::tool_loop::TurnFailure {
            retryable: false,
            message: format!("Asgard failed: {error:#}"),
        }),
        continuation_messages: Vec::new(),
        current_plan: None,
        compaction_checkpoint: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::future::BoxFuture;
    use std::collections::VecDeque;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    fn asgard_ledger_entry_for_test(command: &str, exit_code: Option<i32>) -> AsgardLedgerEntry {
        AsgardLedgerEntry {
            id: format!("id-{command}"),
            step: 0,
            command: command.to_string(),
            exit_code,
            output_bytes: 0,
            output_sha256: asgard_sha256(&[]),
            output_tail: String::new(),
        }
    }

    #[derive(Debug)]
    struct SupervisorRequestRecord {
        model: String,
        messages: Vec<ChatMessage>,
        tool_names: Vec<String>,
        reasoning_effort: Option<String>,
    }

    struct ScriptedSupervisorBackend {
        responses: Mutex<VecDeque<LlmResponse>>,
        requests: Mutex<Vec<SupervisorRequestRecord>>,
    }

    impl ScriptedSupervisorBackend {
        fn new(responses: Vec<LlmResponse>) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                requests: Mutex::new(Vec::new()),
            }
        }
    }

    impl crate::llm_client::LlmBackend for ScriptedSupervisorBackend {
        fn list_models(&self) -> BoxFuture<'_, anyhow::Result<Vec<String>>> {
            Box::pin(async { Ok(Vec::new()) })
        }

        fn stream_chat(
            &self,
            request: StreamChatRequest,
        ) -> BoxFuture<'_, anyhow::Result<LlmResponse>> {
            self.requests
                .lock()
                .expect("request lock")
                .push(SupervisorRequestRecord {
                    model: request.model,
                    messages: request.messages,
                    tool_names: request
                        .tools
                        .unwrap_or_default()
                        .into_iter()
                        .map(|tool| tool.function.name)
                        .collect(),
                    reasoning_effort: request.reasoning_effort,
                });
            let response = self
                .responses
                .lock()
                .expect("response lock")
                .pop_front()
                .expect("scripted supervisor response");
            Box::pin(async move { Ok(response) })
        }
    }

    fn supervisor_tool_call(
        id: &str,
        name: &str,
        arguments: serde_json::Value,
    ) -> crate::llm_client::ToolCall {
        crate::llm_client::ToolCall {
            id: id.to_string(),
            r#type: "function".to_string(),
            function: crate::llm_client::FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    fn run_git(cwd: &Path, args: &[&str]) {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(cwd)
            .args(args)
            .output()
            .expect("run git");
        assert!(
            output.status.success(),
            "git {:?} failed\nstdout:\n{}\nstderr:\n{}",
            args,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn init_git_repo(cwd: &Path) {
        run_git(cwd, &["init"]);
        run_git(cwd, &["config", "user.email", "test@example.com"]);
        run_git(cwd, &["config", "user.name", "Test User"]);
    }

    #[test]
    fn asgard_supervisor_parser_infers_dynamic_count_from_advices() {
        let parsed = parse_asgard_supervisor_decision_with_max(
        r#"{"winner":2,"complete":false,"next_window_steps":5,"state_summary":"work remains","advices":[
            {"strategy":"test the parser","scope_basis":"parser behavior"},
            {"strategy":"challenge the API","scope_basis":"API behavior"}
        ]}"#,
        3,
        5,
        )
        .unwrap();
        assert_eq!(parsed.winner, 2);
        assert_eq!(parsed.next_window_steps, Some(5));
        assert_eq!(parsed.advices.len(), 2);

        let completed = parse_asgard_supervisor_decision_with_max(
            r#"{"winner":1,"complete":true,"state_summary":"done","advices":[]}"#,
            3,
            5,
        )
        .unwrap();
        assert!(completed.complete);
        assert_eq!(completed.advices, vec![None, None, None]);

        for invalid in [
            r#"{"winner":1,"complete":false,"next_window_steps":5,"state_summary":"work","advices":[]}"#,
            r#"{"winner":1,"complete":false,"next_window_steps":5,"state_summary":"work","advices":[{"strategy":"a","scope_basis":"task"},{"strategy":"b","scope_basis":"task"},{"strategy":"c","scope_basis":"task"},{"strategy":"d","scope_basis":"task"},{"strategy":"e","scope_basis":"task"},{"strategy":"f","scope_basis":"task"}]}"#,
            r#"{"winner":1,"complete":false,"next_window_steps":0,"state_summary":"work","advices":[{"strategy":"a","scope_basis":"task"},{"strategy":"b","scope_basis":"task"}]}"#,
            r#"{"winner":1,"complete":false,"next_window_steps":11,"state_summary":"work","advices":[{"strategy":"a","scope_basis":"task"},{"strategy":"b","scope_basis":"task"}]}"#,
            r#"{"winner":1,"complete":false,"next_window_steps":5,"state_summary":"work","advices":[{"strategy":"same","scope_basis":"task"},{"strategy":"same","scope_basis":"task"}]}"#,
            r#"{"winner":3,"complete":false,"next_window_steps":5,"state_summary":"work","advices":[{"strategy":"a","scope_basis":"task"}]}"#,
        ] {
            assert!(parse_asgard_supervisor_decision_with_max(invalid, 3, 5).is_err());
        }
    }

    #[test]
    fn asgard_advice_continues_the_normal_rollout() {
        let message = asgard_advice_message(2, "Focus on the concurrency invariant.", None);
        let text = asgard_message_text(&message);
        assert!(text.contains("lane=\"2\""));
        assert!(text.contains("Focus on the concurrency invariant."));
        assert!(text.contains("continuing the original task normally"));
        assert!(text.contains("Do not stop"));
        assert!(text.contains("do not change unrelated dependencies"));
    }

    #[test]
    fn asgard_verified_at_window_start_renders_green_last_unique_commands() {
        let ledger = AsgardExecutionLedger {
            entries: vec![
                asgard_ledger_entry_for_test("cargo test old", Some(0)),
                asgard_ledger_entry_for_test("cargo test failing", Some(1)),
                asgard_ledger_entry_for_test("cargo check", None),
                asgard_ledger_entry_for_test("cargo test kept", Some(0)),
                asgard_ledger_entry_for_test("cargo test old", Some(0)),
            ],
            ..Default::default()
        };

        let rendered = render_asgard_verified_at_window_start(&ledger).expect("verified block");

        assert!(rendered.starts_with("<verified_at_window_start>\nThe following commands ran"));
        assert!(rendered.contains("- cargo test kept\n- cargo test old"));
        assert!(!rendered.contains("cargo test failing"));
        assert!(!rendered.contains("cargo check"));
        assert!(rendered.ends_with("</verified_at_window_start>"));
    }

    #[test]
    fn asgard_verified_at_window_start_caps_at_last_fifteen() {
        let ledger = AsgardExecutionLedger {
            entries: (0..18)
                .map(|index| asgard_ledger_entry_for_test(&format!("cmd {index}"), Some(0)))
                .collect(),
            ..Default::default()
        };

        let rendered = render_asgard_verified_at_window_start(&ledger).expect("verified block");
        let commands = rendered
            .lines()
            .filter_map(|line| line.strip_prefix("- "))
            .collect::<Vec<_>>();

        assert_eq!(commands.len(), 15);
        assert_eq!(commands[0], "cmd 3");
        assert_eq!(commands[14], "cmd 17");
        assert!(!rendered.contains("- cmd 2\n"));
    }

    #[test]
    fn asgard_verified_at_window_start_is_omitted_when_empty() {
        let ledger = AsgardExecutionLedger {
            entries: vec![
                asgard_ledger_entry_for_test("cargo test failing", Some(1)),
                asgard_ledger_entry_for_test("cargo check", None),
            ],
            ..Default::default()
        };

        assert_eq!(render_asgard_verified_at_window_start(&ledger), None);
        let message = asgard_advice_message(0, "Try the parser path.", None);
        assert!(!asgard_message_text(&message).contains("verified_at_window_start"));
    }

    #[test]
    fn asgard_advice_prepends_verified_at_window_start_block() {
        let message = asgard_advice_message(
            0,
            "Try the parser path.",
            Some("<verified_at_window_start>\n- cargo test parser\n</verified_at_window_start>"),
        );
        let text = asgard_message_text(&message);

        assert!(text.starts_with("<verified_at_window_start>"));
        assert!(
            text.contains("</verified_at_window_start>\n<asgard_next_window_advice lane=\"0\">")
        );
    }

    #[test]
    fn asgard_advice_remains_in_canonical_history_after_window() {
        let messages = vec![
            ChatMessage::system("stable prefix"),
            ChatMessage::user("original task"),
            asgard_advice_message(0, "Try the parser path.", None),
            ChatMessage::assistant("I will inspect the parser."),
        ];

        let window = asgard_take_window_messages(&messages, 2);

        assert_eq!(
            window,
            vec![
                asgard_advice_message(0, "Try the parser path.", None),
                ChatMessage::assistant("I will inspect the parser."),
            ]
        );
        assert_eq!(messages.len(), 4);
        assert!(asgard_message_text(&messages[2]).contains("asgard_next_window_advice"));
        assert_eq!(
            messages[3],
            ChatMessage::assistant("I will inspect the parser.")
        );
    }

    #[test]
    fn asgard_supervisor_summary_is_kept_in_supervisor_history_only() {
        let decision = AsgardSupervisorDecision {
            winner: 2,
            complete: false,
            advices: vec![Some("independent strategy".to_string())],
            next_window_steps: Some(4),
            state_summary: "The selected parser still has an unresolved wildcard assumption."
                .to_string(),
            contracts: None,
        };
        let mut history = AsgardSupervisorHistory::default();
        history.push(4, &decision);

        assert_eq!(history.selected_windows.len(), 1);
        assert!(
            render_asgard_supervisor_history(&history.selected_windows)
                .contains("unresolved wildcard assumption")
        );

        let candidate_advice = asgard_advice_message(0, "independent strategy", None);
        assert!(!asgard_message_text(&candidate_advice).contains("wildcard assumption"));

        history.checkpoint_selected_windows();
        assert!(history.selected_windows.is_empty());
        assert_eq!(history.checkpointed.len(), 1);
    }

    #[test]
    fn asgard_diff_anchor_uses_strict_forty_percent_threshold_and_stable_ties() {
        assert_eq!(
            select_asgard_diff_anchor(1_000, &[(2, 399), (1, 500)]),
            Some((2, 399))
        );
        assert_eq!(
            select_asgard_diff_anchor(1_000, &[(2, 300), (1, 300)]),
            Some((1, 300))
        );
        assert_eq!(
            select_asgard_diff_anchor(1_000, &[(0, 400), (1, 500)]),
            None
        );
        assert_eq!(select_asgard_diff_anchor(0, &[(0, 0), (1, 0)]), None);
        assert_eq!(select_asgard_diff_anchor(1_000, &[(0, 1)]), None);
    }

    #[test]
    fn asgard_diff_presentation_uses_candidate_medoid_when_diffs_cluster() {
        let source = tempfile::tempdir().expect("source tempdir");
        init_git_repo(source.path());
        std::fs::write(source.path().join("README.md"), "base\n").expect("seed source");
        run_git(source.path(), &["add", "README.md"]);
        run_git(source.path(), &["commit", "-m", "seed"]);

        let lane_zero =
            crate::asgard::create_candidate_repository(source.path(), "diff-medoid-zero")
                .expect("lane zero repository");
        let lane_one = crate::asgard::create_candidate_repository(source.path(), "diff-medoid-one")
            .expect("lane one repository");
        let shared = (0..1_000)
            .map(|line| format!("shared-{line:04}\n"))
            .collect::<String>();
        std::fs::write(
            lane_zero.root.join("clustered.txt"),
            format!("{shared}anchor-only\n"),
        )
        .expect("write lane zero");
        std::fs::write(
            lane_one.root.join("clustered.txt"),
            format!("{shared}other-only\n"),
        )
        .expect("write lane one");

        let zero_patch = crate::asgard::capture_patch(&lane_zero.root, &lane_zero.base_commit)
            .expect("capture lane zero patch");
        let one_patch = crate::asgard::capture_patch(&lane_one.root, &lane_one.base_commit)
            .expect("capture lane one patch");
        let zero_delta =
            crate::asgard::capture_patch_since(&lane_zero.root, &lane_zero.base_commit, &[])
                .expect("capture lane zero delta");
        let one_delta =
            crate::asgard::capture_patch_since(&lane_one.root, &lane_one.base_commit, &[])
                .expect("capture lane one delta");
        let presentation = build_asgard_diff_presentation(vec![
            AsgardDiffCandidateInput {
                index: 1,
                patch: one_patch,
                delta_patch: one_delta,
                repository_root: lane_one.root.clone(),
                base_commit: lane_one.base_commit.clone(),
            },
            AsgardDiffCandidateInput {
                index: 0,
                patch: zero_patch.clone(),
                delta_patch: zero_delta.clone(),
                repository_root: lane_zero.root.clone(),
                base_commit: lane_zero.base_commit.clone(),
            },
        ])
        .expect("build clustered diff presentation");

        assert_eq!(presentation.anchor_lane, Some(0));
        assert_eq!(
            presentation.views[0],
            AsgardCandidateDiffView {
                candidate_index: 0,
                base: AsgardCandidateDiffBase::LastSelectedDecision,
                patch: zero_delta,
            }
        );
        assert_eq!(presentation.views[1].candidate_index, 1);
        assert_eq!(
            presentation.views[1].base,
            AsgardCandidateDiffBase::CurrentWindowLane(0)
        );
        let cross_diff = String::from_utf8_lossy(&presentation.views[1].patch);
        assert!(cross_diff.contains("-anchor-only"));
        assert!(cross_diff.contains("+other-only"));
        assert!(
            presentation.best_candidate_sum_bytes.unwrap() * 10
                < presentation.baseline_sum_bytes * 4
        );

        crate::asgard::remove_candidate_repository(&lane_zero);
        crate::asgard::remove_candidate_repository(&lane_one);
    }

    #[test]
    fn asgard_diff_presentation_errors_are_safe_to_fallback_from() {
        let missing = PathBuf::from("/definitely/missing/asgard-candidate");
        let input = |index| AsgardDiffCandidateInput {
            index,
            patch: b"full patch".to_vec(),
            delta_patch: b"delta patch".to_vec(),
            repository_root: missing.clone(),
            base_commit: "missing-base".to_string(),
        };

        let error = build_asgard_diff_presentation(vec![input(0), input(1)])
            .expect_err("pairwise capture should fail");
        assert!(
            format!("{error:#}")
                .contains("capture Asgard diff from current-window lane 0 to lane 1")
        );
    }

    fn asgard_test_candidate(index: usize, window_messages: Vec<ChatMessage>) -> AsgardCandidate {
        AsgardCandidate {
            index,
            model: "test-model".to_string(),
            outcome: asgard_failure(anyhow::anyhow!("test outcome")),
            patch: Vec::new(),
            delta_patch: Vec::new(),
            supervisor_window_messages: Vec::new(),
            window_messages,
            window_ledger: AsgardExecutionLedger::default(),
        }
    }

    #[test]
    fn asgard_candidate_renderer_labels_the_actual_diff_baseline() {
        let candidate = AsgardCandidate {
            index: 3,
            model: "test-model".to_string(),
            outcome: asgard_failure(anyhow::anyhow!("test outcome")),
            patch: b"diff --git a/src/lib.rs b/src/lib.rs\n+full\n".to_vec(),
            delta_patch: b"unused delta".to_vec(),
            window_messages: vec![ChatMessage::assistant("worked")],
            supervisor_window_messages: vec![ChatMessage::assistant("worked")],
            window_ledger: AsgardExecutionLedger::default(),
        };
        let rendered = render_asgard_candidate_trajectory(
            &candidate,
            Path::new("/tmp/lane-three"),
            "current_window_lane_1",
            b"cross-candidate diff",
        )
        .expect("render candidate");

        assert!(rendered.contains("<lane_trajectory index=\"3\""));
        assert!(
            rendered
                .contains("<candidate_window_diff base=\"current_window_lane_1\" bytes=\"20\">")
        );
        assert!(rendered.contains("cross-candidate diff"));
        assert!(!rendered.contains("unused delta"));
    }

    #[test]
    fn asgard_compact_handoff_summarizes_results_and_references_exact_duplicates() {
        let messages = vec![
            ChatMessage::assistant_tool_calls(vec![supervisor_tool_call(
                "read-1",
                "read_file",
                serde_json::json!({"file_path": "src/lib.rs"}),
            )]),
            ChatMessage::tool_result("read-1", "read_file", "important source observation"),
            ChatMessage::assistant_tool_calls(vec![supervisor_tool_call(
                "read-2",
                "read_file",
                serde_json::json!({"file_path": "src/lib.rs"}),
            )]),
            ChatMessage::tool_result("read-2", "read_file", "important source observation"),
            ChatMessage::assistant("The observation determines the next edit."),
        ];

        let (handoff, raw_bytes, packed_bytes) =
            asgard_deterministic_candidate_handoff(4, 2, &messages, None);
        let rendered = render_asgard_dossier_messages(&handoff);

        assert!(rendered.contains("format=\"compact_deterministic\""));
        // The result body is summarized, not carried, and the summary names the
        // call so the supervisor knows what to retrieve.
        assert!(!rendered.contains("important source observation"));
        assert!(rendered.contains("read src/lib.rs"));
        assert!(rendered.contains("id=\"w4l2m1\""));
        assert!(rendered.contains("exact_duplicate_of=\"w4l2m1\""));
        assert!(rendered.contains("The observation determines the next edit."));
        assert!(packed_bytes < raw_bytes);
    }

    #[test]
    fn asgard_compact_handoff_has_no_size_gate() {
        // The old deterministic path bailed above 16 KB and paid an LLM to
        // summarize instead. Compact rendering has no such ceiling.
        let messages = vec![
            ChatMessage::assistant_tool_calls(vec![supervisor_tool_call(
                "large",
                "run_shell_command",
                serde_json::json!({"command": "cargo test"}),
            )]),
            ChatMessage::tool_result(
                "large",
                "run_shell_command",
                "unique evidence".repeat(40_000),
            ),
        ];
        let (handoff, raw_bytes, packed_bytes) =
            asgard_deterministic_candidate_handoff(0, 0, &messages, None);
        assert!(raw_bytes > 500_000);
        assert!(packed_bytes < 500);
        let rendered = render_asgard_dossier_messages(&handoff);
        assert!(rendered.contains("$ \"cargo test\""));
        assert!(rendered.contains("600000 chars omitted"));
    }

    #[test]
    fn asgard_view_tool_call_resolves_handles_within_the_current_window() {
        let candidates = vec![
            asgard_test_candidate(
                0,
                vec![
                    ChatMessage::assistant_tool_calls(vec![supervisor_tool_call(
                        "c0",
                        "run_shell_command",
                        serde_json::json!({"command": "cargo test -p lane0"}),
                    )]),
                    ChatMessage::tool_result("c0", "run_shell_command", "lane 0 full output"),
                ],
            ),
            asgard_test_candidate(
                1,
                vec![
                    ChatMessage::assistant_tool_calls(vec![supervisor_tool_call(
                        "c1",
                        "run_shell_command",
                        serde_json::json!({"command": "cargo test -p lane1"}),
                    )]),
                    ChatMessage::tool_result("c1", "run_shell_command", "lane 1 full output"),
                ],
            ),
        ];
        let audit = AsgardAuditContext {
            registries: &[],
            candidates: &candidates,
            definitions: Vec::new(),
            allowed_lane: None,
            retained_windows: &[],
            window: 3,
        };

        // Concurrent lanes at the same message index resolve to their own data.
        let resolved =
            resolve_asgard_tool_call_handles(&audit, &["w3l0m1".to_string(), "w3l1m1".to_string()]);
        assert!(resolved.contains("lane 0 full output"));
        assert!(resolved.contains("lane 1 full output"));
        assert!(resolved.contains("cargo test -p lane0"));
        assert!(resolved.contains("cargo test -p lane1"));

        // An unretained earlier window is refused rather than silently resolved
        // against the current window's messages, as are malformed handles.
        let refused =
            resolve_asgard_tool_call_handles(&audit, &["w2l0m1".to_string(), "junk".to_string()]);
        assert!(refused.contains("was not retained"));
        assert!(refused.contains("unrecognized handle format"));
        assert!(!refused.contains("lane 0 full output"));
    }

    #[test]
    fn asgard_view_tool_call_expands_retained_earlier_windows() {
        let candidates = vec![asgard_test_candidate(
            0,
            vec![
                ChatMessage::assistant_tool_calls(vec![supervisor_tool_call(
                    "c0",
                    "run_shell_command",
                    serde_json::json!({"command": "cargo test current"}),
                )]),
                ChatMessage::tool_result("c0", "run_shell_command", "current window output"),
            ],
        )];
        // Window 2 was won by lane 1; the dossier still cites its ledger.
        let retained = vec![AsgardRetainedWindow {
            window: 2,
            lane: 1,
            messages: vec![
                ChatMessage::assistant_tool_calls(vec![supervisor_tool_call(
                    "p0",
                    "run_shell_command",
                    serde_json::json!({"command": "cargo test --release earlier"}),
                )]),
                ChatMessage::tool_result("p0", "run_shell_command", "earlier window full output"),
            ],
        }];
        let audit = AsgardAuditContext {
            registries: &[],
            candidates: &candidates,
            definitions: Vec::new(),
            allowed_lane: None,
            retained_windows: &retained,
            window: 3,
        };

        let resolved = resolve_asgard_tool_call_handles(&audit, &["w2l1m1".to_string()]);
        assert!(resolved.contains("earlier window full output"));
        assert!(resolved.contains("cargo test --release earlier"));

        // Only the lane that actually won that window is expandable.
        let refused = resolve_asgard_tool_call_handles(&audit, &["w2l0m1".to_string()]);
        assert!(refused.contains("was not retained"));
        assert!(!refused.contains("earlier window full output"));
    }

    #[test]
    fn asgard_ledger_ids_are_view_tool_call_handles_for_the_same_result() {
        let messages = vec![
            ChatMessage::assistant_tool_calls(vec![supervisor_tool_call(
                "s0",
                "run_shell_command",
                serde_json::json!({"command": "cargo test -p thing"}),
            )]),
            ChatMessage::tool_result("s0", "run_shell_command", "Exit code: 0\nall green"),
        ];
        let ledger = asgard_extract_execution_ledger(4, 2, &messages);
        let entry = &ledger.entries[0];

        // The id addresses the result message, not the assistant message that
        // issued the call — that is what view_tool_call resolves.
        assert_eq!(entry.id, "w4l2m1");
        assert_eq!(entry.step, 0);

        let candidates = vec![AsgardCandidate {
            index: 2,
            ..asgard_test_candidate(2, messages.clone())
        }];
        let audit = AsgardAuditContext {
            registries: &[],
            candidates: &candidates,
            definitions: Vec::new(),
            allowed_lane: None,
            retained_windows: &[],
            window: 4,
        };
        let resolved = resolve_asgard_tool_call_handles(&audit, std::slice::from_ref(&entry.id));
        assert!(resolved.contains("all green"));
        assert!(resolved.contains("cargo test -p thing"));
    }

    #[test]
    fn asgard_execution_contracts_accept_handle_shaped_ledger_citations() {
        let row = |evidence: &str| AsgardContractRow {
            id: "C1".to_string(),
            status: "verified".to_string(),
            evidence: evidence.to_string(),
            adverse_condition_evidence: None,
        };
        // The gate must recognize the handle form, or every execution contract
        // fails validation and completion can never be accepted.
        assert!(asgard_cites_ledger_entry(&row(
            "w3l1m12 ran the suite and exited 0"
        )));
        assert!(asgard_cites_ledger_entry(&row("see ledger entry w0l0m4.")));
        assert!(!asgard_cites_ledger_entry(&row(
            "the tests obviously pass here"
        )));
        // The retired L<n> vocabulary is no longer a citation.
        assert!(!asgard_cites_ledger_entry(&row("L3 exited 0")));
    }

    #[test]
    fn asgard_view_tool_call_refuses_other_lanes_during_completion_review() {
        let candidates = vec![
            asgard_test_candidate(
                0,
                vec![
                    ChatMessage::assistant_tool_calls(vec![supervisor_tool_call(
                        "c0",
                        "read_file",
                        serde_json::json!({"file_path": "a.rs"}),
                    )]),
                    ChatMessage::tool_result("c0", "read_file", "discarded lane content"),
                ],
            ),
            asgard_test_candidate(
                1,
                vec![
                    ChatMessage::assistant_tool_calls(vec![supervisor_tool_call(
                        "c1",
                        "read_file",
                        serde_json::json!({"file_path": "b.rs"}),
                    )]),
                    ChatMessage::tool_result("c1", "read_file", "selected lane content"),
                ],
            ),
        ];
        let audit = AsgardAuditContext {
            registries: &[],
            candidates: &candidates,
            definitions: Vec::new(),
            allowed_lane: Some(1),
            retained_windows: &[],
            window: 0,
        };
        let resolved =
            resolve_asgard_tool_call_handles(&audit, &["w0l0m1".to_string(), "w0l1m1".to_string()]);
        assert!(resolved.contains("selected lane content"));
        assert!(!resolved.contains("discarded lane content"));
        assert!(resolved.contains("only selected candidate lane 1"));
    }

    #[test]
    fn asgard_view_tool_call_arguments_are_validated() {
        assert!(asgard_view_tool_call_handles(&serde_json::json!({})).is_err());
        assert!(asgard_view_tool_call_handles(&serde_json::json!({"handles": []})).is_err());
        assert!(asgard_view_tool_call_handles(&serde_json::json!({"handles": [7]})).is_err());
        assert_eq!(
            asgard_view_tool_call_handles(&serde_json::json!({"handles": ["w0l0m1"]}))
                .expect("valid handles"),
            vec!["w0l0m1".to_string()]
        );
    }

    #[test]
    fn asgard_completion_review_gate_requires_net_change_or_new_evidence() {
        let temp = tempfile::tempdir().expect("candidate repository");
        init_git_repo(temp.path());
        std::fs::create_dir_all(temp.path().join("src")).expect("create src");
        std::fs::write(temp.path().join("src/lib.rs"), "base\n").expect("seed source");
        run_git(temp.path(), &["add", "src/lib.rs"]);
        run_git(temp.path(), &["commit", "-m", "seed"]);
        let base_commit = String::from_utf8(
            std::process::Command::new("git")
                .args(["-C"])
                .arg(temp.path())
                .args(["rev-parse", "HEAD"])
                .output()
                .expect("read base commit")
                .stdout,
        )
        .expect("utf8 commit")
        .trim()
        .to_string();
        std::fs::write(temp.path().join("src/lib.rs"), "reviewed\n").expect("write reviewed state");
        let reviewed_patch = crate::asgard::capture_patch(temp.path(), &base_commit)
            .expect("capture reviewed patch");
        let repeated_entry = AsgardLedgerEntry {
            id: "L1".to_string(),
            step: 0,
            command: "cargo test focused".to_string(),
            exit_code: Some(0),
            output_bytes: "1 passed".len(),
            output_sha256: asgard_sha256(b"1 passed"),
            output_tail: "1 passed".to_string(),
        };
        let prior = AsgardPriorCompletionReview {
            window: 2,
            rows: Vec::new(),
            decision: AsgardSupervisorDecision {
                winner: 0,
                complete: false,
                advices: vec![Some("produce new evidence".to_string())],
                next_window_steps: Some(1),
                state_summary: "prior rejection".to_string(),
                contracts: None,
            },
            reviewed_patch: reviewed_patch.clone(),
            evidence_fingerprints: [asgard_evidence_fingerprint(&repeated_entry)]
                .into_iter()
                .collect(),
        };
        let repository = crate::asgard::CandidateRepository {
            root: temp.path().to_path_buf(),
            session_cwd: temp.path().to_path_buf(),
            base_commit: base_commit.clone(),
        };
        let mut candidate = AsgardCandidate {
            index: 0,
            model: "test-model".to_string(),
            outcome: asgard_failure(anyhow::anyhow!("unused test outcome")),
            patch: reviewed_patch,
            delta_patch: Vec::new(),
            window_messages: Vec::new(),
            supervisor_window_messages: Vec::new(),
            window_ledger: AsgardExecutionLedger {
                entries: vec![repeated_entry],
                ..AsgardExecutionLedger::default()
            },
        };

        // An intermediate edit that is reverted to the reviewed state is not
        // material, nor is rerunning the same command with identical output.
        std::fs::write(temp.path().join("src/lib.rs"), "temporary\n").expect("temporary edit");
        std::fs::write(temp.path().join("src/lib.rs"), "reviewed\n").expect("revert edit");
        let stale = asgard_completion_review_delta(&prior, &candidate, &repository, &[])
            .expect("evaluate stale renomination");
        assert!(!stale.is_material());
        assert!(stale.patch.is_empty());

        std::fs::write(temp.path().join("src/lib.rs"), "material change\n").expect("material edit");
        let changed = asgard_completion_review_delta(&prior, &candidate, &repository, &[])
            .expect("evaluate production change");
        assert!(changed.is_material());
        assert!(changed.production_bytes > 0);

        std::fs::write(temp.path().join("src/lib.rs"), "reviewed\n")
            .expect("restore reviewed state");
        candidate.window_ledger.entries.push(AsgardLedgerEntry {
            id: "L2".to_string(),
            step: 1,
            command: "cargo test adverse_case".to_string(),
            exit_code: Some(0),
            output_bytes: "adverse case passed".len(),
            output_sha256: asgard_sha256(b"adverse case passed"),
            output_tail: "adverse case passed".to_string(),
        });
        let evidence = asgard_completion_review_delta(&prior, &candidate, &repository, &[])
            .expect("evaluate new execution evidence");
        assert!(evidence.is_material());
        assert_eq!(evidence.new_evidence_count, 1);

        candidate.window_ledger.entries.truncate(1);
        candidate.window_messages = vec![ChatMessage::assistant_tool_calls(vec![
            supervisor_tool_call(
                "read-new",
                "read_file",
                serde_json::json!({"file_path": "src/lib.rs"}),
            ),
        ])];
        let observation = asgard_completion_review_delta(&prior, &candidate, &repository, &[])
            .expect("evaluate new read-only observation");
        assert!(observation.is_material());
        assert!(observation.has_read_only_observation);
    }

    #[test]
    fn asgard_diff_trace_records_anchor_or_fallback() {
        let anchored = AsgardDiffPresentation {
            views: Vec::new(),
            anchor_lane: Some(2),
            baseline_sum_bytes: 1_000,
            best_candidate_sum_bytes: Some(250),
        };
        let anchored_trace = asgard_diff_baseline_trace_record(4, &anchored, None);
        assert_eq!(anchored_trace["mode"], "current_window_candidate");
        assert_eq!(anchored_trace["anchor_lane"], 2);
        assert_eq!(anchored_trace["threshold_numerator"], 2);
        assert_eq!(anchored_trace["threshold_denominator"], 5);

        let fallback = AsgardDiffPresentation {
            views: Vec::new(),
            anchor_lane: None,
            baseline_sum_bytes: 1_000,
            best_candidate_sum_bytes: None,
        };
        let fallback_trace = asgard_diff_baseline_trace_record(
            4,
            &fallback,
            Some(("pairwise_diff_error", "git failed")),
        );
        assert_eq!(fallback_trace["mode"], "last_selected_decision");
        assert_eq!(fallback_trace["fallback_reason"], "pairwise_diff_error");
        assert_eq!(fallback_trace["fallback_error"], "git failed");
    }

    #[test]
    fn asgard_candidate_completion_does_not_override_supervisor() {
        let natural_stop = crate::tool_loop::LoopStop::Completed { had_text: true };
        assert!(!asgard_should_finish(false, &natural_stop));
        assert!(asgard_should_finish(true, &natural_stop));
        assert!(asgard_should_finish(
            false,
            &crate::tool_loop::LoopStop::Cancelled
        ));
        assert!(!asgard_should_finish(
            false,
            &crate::tool_loop::LoopStop::Failed(crate::tool_loop::TurnFailure {
                retryable: true,
                message: "lane llm error".to_string(),
            })
        ));
    }

    #[test]
    fn asgard_supervisor_keeps_stable_task_prefix_ahead_of_window_dossier() {
        let selected_first = vec![
            ChatMessage::system("stable selected system"),
            ChatMessage::user("selected step one"),
        ];
        let selected_windows = vec![vec![
            asgard_advice_message(1, "Verify the selected implementation.", None),
            ChatMessage::assistant("selected step two"),
        ]];
        let first_history = AsgardSupervisorHistory::default();
        let second_history = AsgardSupervisorHistory {
            checkpointed: Vec::new(),
            selected_windows: vec![AsgardSupervisorHistoryEntry {
                window: 1,
                winner: 1,
                state_summary:
                    "Selected the parser implementation; boundary behavior remains uncertain."
                        .to_string(),
            }],
        };
        let task_contract_checklist =
            asgard_render_task_contract_checklist(vec![AsgardTaskContract {
                id: "C1".to_string(),
                kind: "inspection".to_string(),
                text: "The parser preserves the requested output format.".to_string(),
                adverse_condition: None,
            }]);
        let first = asgard_supervisor_messages(
            "fix the parser",
            &selected_first,
            &[],
            &first_history,
            AsgardCandidateCounts { current: 3, max: 5 },
            &task_contract_checklist,
            "candidate window one".to_string(),
        );
        let repeated_first = asgard_supervisor_messages(
            "fix the parser",
            &selected_first,
            &[],
            &first_history,
            AsgardCandidateCounts { current: 3, max: 5 },
            &task_contract_checklist,
            "candidate window one".to_string(),
        );
        let second = asgard_supervisor_messages(
            "fix the parser",
            &selected_first,
            &selected_windows,
            &second_history,
            AsgardCandidateCounts { current: 3, max: 5 },
            &task_contract_checklist,
            "candidate window two".to_string(),
        );
        let terminal = asgard_supervisor_messages(
            "fix the parser",
            &selected_first,
            &selected_windows,
            &second_history,
            AsgardCandidateCounts { current: 3, max: 5 },
            &task_contract_checklist,
            "terminal candidate window".to_string(),
        );
        let isolated_review = asgard_completion_review_messages(
            "fix the parser",
            &selected_first,
            &selected_windows,
            &second_history,
            3,
            5,
            &task_contract_checklist,
            &[(1, AsgardExecutionLedger::default())],
            "diff --git a/src/parser.rs b/src/parser.rs\n+impl\n".to_string(),
            "diff --git a/tests/parser_test.rs b/tests/parser_test.rs\n+assert\n".to_string(),
            1,
            "selected lane two only".to_string(),
            None,
            String::new(),
        );

        assert_eq!(&first[..first.len() - 1], &second[..first.len() - 1]);
        assert_ne!(first.last(), second.last());
        assert!(asgard_message_text(&first[1]).contains("fix the parser"));
        assert!(!asgard_message_text(&first[0]).contains("window one"));
        assert!(asgard_message_text(&first[0]).contains("correctness supervisor"));
        assert!(asgard_message_text(&first[0]).contains("1-5 lanes next"));
        assert!(asgard_message_text(&first[0]).contains("format=\"compact_deterministic\""));
        assert!(asgard_message_text(&first[0]).contains("diff-compression anchor"));
        assert!(asgard_message_text(&first[0]).contains("anchor is not canonical or preferred"));
        assert!(asgard_message_text(&first[0]).contains("lane indices remain authoritative"));
        assert!(asgard_message_text(&first[0]).contains("green tests changed alongside"));
        assert!(asgard_message_text(&first[0]).contains("argument order"));
        assert!(asgard_message_text(&first[0]).contains("missing combinations"));
        assert!(asgard_message_text(&first[0]).contains("nomination, not a terminal verdict"));
        assert!(
            asgard_message_text(&first[0])
                .contains("isolated completion review owns all terminal adjudication")
        );
        assert!(
            asgard_message_text(&first[0])
                .contains("plainly incomplete, do not perform a terminal-grade exhaustive audit")
        );
        assert!(
            asgard_message_text(&first[0]).contains("most consequential unverified assumption")
        );
        assert!(
            asgard_message_text(&first[0])
                .contains("use separate lanes only when implementing or testing those readings")
        );
        assert!(asgard_message_text(&first[0]).contains("Do not assert exact APIs"));
        assert!(!asgard_message_text(&first[0]).contains("Tool choice is not forced"));
        assert!(!asgard_message_text(&first[0]).contains("reasoning must remain enabled"));
        assert!(
            asgard_message_text(&first[0]).contains("inspect the candidate checkouts for evidence")
        );
        // Retrieval must be advertised as free, or the supervisor rations it
        // against the checkout-audit budget and reasons from summaries instead.
        assert!(asgard_message_text(&first[0]).contains(
            "view_tool_call expands the compact trajectories you were given and is unbudgeted"
        ));
        assert!(asgard_message_text(&first[0]).contains("edits since the last selected decision"));
        assert!(
            asgard_message_text(&first[0])
                .contains("complete=true is a nomination that the selected endpoint plausibly")
        );
        assert!(asgard_message_text(&first[0]).contains("at most 3 information-gathering"));
        assert!(asgard_message_text(&first[0]).contains("tell one or more next-window candidates"));
        assert_eq!(first[0], terminal[0]);
        assert_eq!(first[2].role, "assistant");
        assert!(asgard_message_text(&first[2]).contains("task_contract_checklist"));
        assert!(asgard_message_text(&first[2]).contains("C1"));
        assert_eq!(first[2], repeated_first[2]);
        assert!(asgard_message_text(&first[3]).contains("stable selected system"));
        assert_eq!(first[3].role, "assistant");
        assert_eq!(second[4].role, "user");
        assert!(asgard_message_text(&second[4]).contains("window_boundary"));
        assert_eq!(second[5].role, "assistant");
        assert!(asgard_message_text(&second[5]).contains("Verify the selected implementation."));
        assert!(asgard_message_text(&second[5]).contains("selected step two"));
        assert!(asgard_message_text(&second[5]).contains("boundary behavior remains uncertain"));
        assert_eq!(second[6].role, "user");
        assert!(asgard_message_text(&second[6]).contains("candidate window two"));
        assert!(asgard_message_text(&second[6]).contains("<decision_procedure>"));
        assert!(asgard_message_text(&second[6]).contains("Compare each lane's actual direction"));
        assert!(asgard_message_text(&second[6]).contains("If so, nominate: complete=true"));
        assert!(
            asgard_message_text(&second[6])
                .contains("Treat the completion judgment as a nomination")
        );
        assert_eq!(
            &second[..second.len() - 1],
            &isolated_review[..isolated_review.len() - 1]
        );
        let isolated_suffix = asgard_message_text(isolated_review.last().unwrap());
        assert!(isolated_suffix.contains("selected lane two only"));
        assert!(isolated_suffix.contains("single would-be-final endpoint"));
        assert!(isolated_suffix.contains("discarded candidate lanes are intentionally absent"));
        assert!(isolated_suffix.contains("Keep selected_lane=1"));
        assert!(isolated_suffix.contains("<canonical_execution_ledger"));
        assert!(isolated_suffix.contains("<terminal_non_test_patch"));
        assert!(isolated_suffix.contains("<terminal_test_patch"));
        assert!(isolated_suffix.contains("Keep verified rows terse"));
        assert!(isolated_suffix.contains("carried-forward: unchanged since window N review"));
        assert!(isolated_suffix.contains("Evidence rules, in strength order"));
        assert!(
            isolated_suffix
                .contains("type-checks standalone usage authored from the contract's text")
        );
        assert!(isolated_suffix.contains(
            "complete=true requires every functional (inspection or execution) row verified"
        ));
        assert!(
            isolated_suffix
                .contains("write a standalone usage file authored from the contract text verbatim")
        );
        assert!(!isolated_suffix.contains("candidate window two"));
        assert!(ASGARD_CONTRACT_EXTRACTION_PROMPT.contains(
            "add one contract per such signature stating its exact parameter and return types"
        ));
        assert!(
            ASGARD_CONTRACT_EXTRACTION_PROMPT
                .contains("record the contract once per reading and set each adverse_condition")
        );
    }

    #[test]
    fn asgard_execution_ledger_recovers_shell_results_and_edit_steps() {
        let messages = vec![
            ChatMessage::assistant_tool_calls(vec![supervisor_tool_call(
                "shell-fail",
                "run_shell_command",
                serde_json::json!({"command":"cargo test failing_case"}),
            )]),
            ChatMessage::tool_result(
                "shell-fail",
                "run_shell_command",
                "stdout\n[WARNING] OS sandbox unavailable\n\nExit code: 2",
            ),
            ChatMessage::assistant_tool_calls(vec![supervisor_tool_call(
                "shell-empty-success",
                "run_shell_command",
                serde_json::json!({"command":"true"}),
            )]),
            ChatMessage::tool_result(
                "shell-empty-success",
                "run_shell_command",
                "Command completed with exit code 0",
            ),
            ChatMessage::assistant_tool_calls(vec![supervisor_tool_call(
                "shell-output-success",
                "run_shell_command",
                serde_json::json!({"command":"printf ok"}),
            )]),
            ChatMessage::tool_result("shell-output-success", "run_shell_command", "ok\nall good"),
            ChatMessage::assistant_tool_calls(vec![
                supervisor_tool_call(
                    "edit-call",
                    "edit",
                    serde_json::json!({"file_path":"src/lib.rs"}),
                ),
                supervisor_tool_call(
                    "write-call",
                    "write_file",
                    serde_json::json!({"file_path":"tests/lib_test.rs"}),
                ),
            ]),
            ChatMessage::assistant_tool_calls(vec![supervisor_tool_call(
                "shell-missing",
                "run_shell_command",
                serde_json::json!({"command":"cargo test missing"}),
            )]),
        ];

        let ledger = asgard_extract_execution_ledger(0, 0, &messages);

        assert_eq!(ledger.total_shell_commands, 4);
        assert_eq!(ledger.entries.len(), 4);
        assert_eq!(ledger.entries[0].exit_code, Some(2));
        assert_eq!(ledger.entries[1].exit_code, Some(0));
        assert_eq!(ledger.entries[2].exit_code, Some(0));
        assert_eq!(ledger.entries[3].exit_code, None);
        assert!(
            !ledger.entries[0]
                .output_tail
                .contains("[WARNING] OS sandbox unavailable")
        );
        assert!(ledger.entries[0].output_tail.contains("stdout"));
        assert_eq!(ledger.entries[2].output_bytes, "ok\nall good".len());
        assert_eq!(
            ledger.entries[2].output_sha256,
            asgard_sha256(b"ok\nall good")
        );
        assert_eq!(ledger.entries[3].output_bytes, 0);
        assert_eq!(ledger.entries[3].output_sha256, asgard_sha256(&[]));
        assert_eq!(
            ledger.edit_steps,
            vec![
                AsgardLedgerEdit {
                    step: 6,
                    file: "src/lib.rs".to_string(),
                },
                AsgardLedgerEdit {
                    step: 6,
                    file: "tests/lib_test.rs".to_string(),
                },
            ]
        );
    }

    #[test]
    fn asgard_execution_ledger_matches_shell_result_documented_formats() {
        let shell = |id: &str, command: &str, result: &str| {
            vec![
                ChatMessage::assistant_tool_calls(vec![supervisor_tool_call(
                    id,
                    "run_shell_command",
                    serde_json::json!({"command": command}),
                )]),
                ChatMessage::tool_result(id, "run_shell_command", result),
            ]
        };
        // These strings mirror src/tools/shell.rs format_shell_tool_result:
        // successful output has no marker, empty success has a completion marker,
        // and failure appends "\n\nExit code: {code}".
        let mut messages = Vec::new();
        messages.extend(shell("success-output", "printf ok", "ok\n"));
        messages.extend(shell(
            "success-empty",
            "true",
            "Command completed with exit code 0",
        ));
        messages.extend(shell("failure-output", "false", "failed\n\nExit code: 17"));

        let ledger = asgard_extract_execution_ledger(0, 0, &messages);

        assert_eq!(
            ledger
                .entries
                .iter()
                .map(|entry| entry.exit_code)
                .collect::<Vec<_>>(),
            vec![Some(0), Some(0), Some(17)]
        );
    }

    #[tokio::test]
    async fn asgard_completion_review_enforces_contract_rows_only_for_terminal_complete() {
        let checklist_ids = vec![AsgardTaskContract {
            id: "C1".to_string(),
            kind: "inspection".to_string(),
            text: "The parser preserves the requested output format.".to_string(),
            adverse_condition: None,
        }];
        let missing_contracts = supervisor_tool_call(
            "missing-contracts",
            "select_trajectory",
            serde_json::json!({
                "winner": 0,
                "complete": true,
                "state_summary": "The endpoint is complete but lacks evidence rows.",
                "advices": []
            }),
        );
        let valid_contracts_json = serde_json::json!({
            "winner": 0,
            "complete": true,
            "state_summary": "The endpoint is complete with contract evidence.",
            "advices": [],
            "contracts": [{
                "id": "C1",
                "status": "verified",
                "evidence": "execution_ledger w1l0m3 exited 0 and terminal_test_patch asserts it"
            }]
        });
        let valid_contracts = supervisor_tool_call(
            "valid-contracts",
            "select_trajectory",
            valid_contracts_json.clone(),
        );
        let reconfirmed_contracts = supervisor_tool_call(
            "reconfirmed-contracts",
            "select_trajectory",
            valid_contracts_json,
        );
        let backend = ScriptedSupervisorBackend::new(vec![
            LlmResponse::ToolCalls {
                text: String::new(),
                reasoning_content: None,
                calls: vec![missing_contracts],
                usage: crate::llm_client::TokenUsage::default(),
            },
            LlmResponse::ToolCalls {
                text: String::new(),
                reasoning_content: None,
                calls: vec![valid_contracts],
                usage: crate::llm_client::TokenUsage::default(),
            },
            LlmResponse::ToolCalls {
                text: String::new(),
                reasoning_content: None,
                calls: vec![reconfirmed_contracts],
                usage: crate::llm_client::TokenUsage::default(),
            },
        ]);

        let (decision, _) = run_asgard_supervisor_tool_steps(
            &backend,
            vec![ChatMessage::user("completion review")],
            AsgardSupervisorToolContext {
                model: "deepseek::deepseek-v4-pro",
                candidate_count: 1,
                max_candidate_count: 1,
                idle_timeout: IdleTimeouts::uniform(std::time::Duration::from_secs(1)),
                audit: None,
                required_winner: Some(0),
                checklist_ids: &checklist_ids,
                carry_forward_allowed: false,
            },
            tokio_util::sync::CancellationToken::new(),
            None,
        )
        .await;

        let (decision, stats) = decision.expect("corrected contract rows");
        assert!(decision.complete);
        assert_eq!(
            stats,
            AsgardChallengeStats {
                issued: false,
                flipped: false,
                validation_rounds: 1,
            }
        );
        {
            let requests = backend.requests.lock().expect("request lock");
            // Round 1: missing rows rejected; round 2: valid rows accepted
            // immediately (the self-adjudication challenge round is retired:
            // measured live, its flips never changed a final outcome).
            assert_eq!(requests.len(), 2);
            assert!(requests[1].messages.iter().any(|message| {
                message.role == "tool"
                    && asgard_message_text(message).contains("contract C1 is missing")
            }));
        }

        let incomplete_backend = ScriptedSupervisorBackend::new(vec![LlmResponse::ToolCalls {
            text: String::new(),
            reasoning_content: None,
            calls: vec![supervisor_tool_call(
                "incomplete",
                "select_trajectory",
                serde_json::json!({
                    "winner": 0,
                    "complete": false,
                    "next_window_steps": 3,
                    "state_summary": "More evidence is needed.",
                    "advices": [{
                        "strategy": "Produce the missing evidence.",
                        "scope_basis": "The task contract remains unverified."
                    }]
                }),
            )],
            usage: crate::llm_client::TokenUsage::default(),
        }]);
        let (incomplete, _) = run_asgard_supervisor_tool_steps(
            &incomplete_backend,
            vec![ChatMessage::user("completion review")],
            AsgardSupervisorToolContext {
                model: "deepseek::deepseek-v4-pro",
                candidate_count: 1,
                max_candidate_count: 1,
                idle_timeout: IdleTimeouts::uniform(std::time::Duration::from_secs(1)),
                audit: None,
                required_winner: Some(0),
                checklist_ids: &checklist_ids,
                carry_forward_allowed: false,
            },
            tokio_util::sync::CancellationToken::new(),
            None,
        )
        .await;
        let (incomplete, stats) = incomplete.expect("incomplete accepted");
        assert!(!incomplete.complete);
        assert_eq!(
            stats,
            AsgardChallengeStats {
                issued: false,
                flipped: false,
                validation_rounds: 0,
            }
        );
        assert_eq!(
            incomplete_backend
                .requests
                .lock()
                .expect("request lock")
                .len(),
            1
        );

        let empty_checklist_backend =
            ScriptedSupervisorBackend::new(vec![LlmResponse::ToolCalls {
                text: String::new(),
                reasoning_content: None,
                calls: vec![supervisor_tool_call(
                    "complete-empty-checklist",
                    "select_trajectory",
                    serde_json::json!({
                        "winner": 0,
                        "complete": true,
                        "state_summary": "No checklist ids means no contract-row enforcement.",
                        "advices": []
                    }),
                )],
                usage: crate::llm_client::TokenUsage::default(),
            }]);
        let (empty_checklist, _) = run_asgard_supervisor_tool_steps(
            &empty_checklist_backend,
            vec![ChatMessage::user("completion review")],
            AsgardSupervisorToolContext {
                model: "deepseek::deepseek-v4-pro",
                candidate_count: 1,
                max_candidate_count: 1,
                idle_timeout: IdleTimeouts::uniform(std::time::Duration::from_secs(1)),
                audit: None,
                required_winner: Some(0),
                checklist_ids: &[],
                carry_forward_allowed: false,
            },
            tokio_util::sync::CancellationToken::new(),
            None,
        )
        .await;
        let (empty_checklist, stats) = empty_checklist.expect("empty checklist accepted");
        assert!(empty_checklist.complete);
        assert_eq!(
            stats,
            AsgardChallengeStats {
                issued: false,
                flipped: false,
                validation_rounds: 0,
            }
        );
        assert_eq!(
            empty_checklist_backend
                .requests
                .lock()
                .expect("request lock")
                .len(),
            1
        );
    }

    #[test]
    fn asgard_decision_trace_record_includes_challenge_stats() {
        let decision = AsgardSupervisorDecision {
            winner: 0,
            complete: false,
            advices: vec![Some("Verify the exact behavior.".to_string())],
            next_window_steps: Some(3),
            state_summary: "Lane 1 is best but incomplete.".to_string(),
            contracts: None,
        };

        let record = asgard_decision_trace_record(
            "completion_review",
            &decision,
            AsgardChallengeStats {
                issued: true,
                flipped: true,
                validation_rounds: 2,
            },
        );

        assert_eq!(record["type"], "asgard_decision");
        assert_eq!(record["challenge"]["issued"], true);
        assert_eq!(record["challenge"]["flipped"], true);
        assert_eq!(record["challenge"]["validation_rounds"], 2);
        assert!(record["decision"]["winner"].is_number());
    }

    #[test]
    fn asgard_validate_contract_rows_allows_carried_execution_evidence_only_when_enabled() {
        let checklist = vec![AsgardTaskContract {
            id: "C1".to_string(),
            kind: "execution".to_string(),
            text: "The command must produce the requested output.".to_string(),
            adverse_condition: None,
        }];
        let decision = AsgardSupervisorDecision {
            winner: 0,
            complete: true,
            advices: Vec::new(),
            next_window_steps: None,
            state_summary: "Complete.".to_string(),
            contracts: Some(vec![AsgardContractRow {
                id: "C1".to_string(),
                status: "verified".to_string(),
                evidence: "carried-forward: unchanged since window 2 review".to_string(),
                adverse_condition_evidence: None,
            }]),
        };

        assert!(asgard_validate_contract_rows(&decision, &checklist, true).is_empty());
        let violations = asgard_validate_contract_rows(&decision, &checklist, false);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("cites no execution_ledger entry"));
    }

    #[test]
    fn asgard_validate_contract_rows_still_requires_adverse_evidence_for_carried_rows() {
        let checklist = vec![AsgardTaskContract {
            id: "C1".to_string(),
            kind: "execution".to_string(),
            text: "Shutdown must unblock a pending read.".to_string(),
            adverse_condition: Some("read is pending when shutdown fires".to_string()),
        }];
        let decision = AsgardSupervisorDecision {
            winner: 0,
            complete: true,
            advices: Vec::new(),
            next_window_steps: None,
            state_summary: "Complete.".to_string(),
            contracts: Some(vec![AsgardContractRow {
                id: "C1".to_string(),
                status: "verified".to_string(),
                evidence: "carried-forward: unchanged since window 2 review".to_string(),
                adverse_condition_evidence: None,
            }]),
        };

        let violations = asgard_validate_contract_rows(&decision, &checklist, true);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("has no adverse_condition_evidence"));
    }

    #[test]
    fn asgard_completion_review_dossier_includes_terminal_evidence_bundle() {
        let checklist = asgard_render_task_contract_checklist(vec![AsgardTaskContract {
            id: "C1".to_string(),
            kind: "execution".to_string(),
            text: "The command must produce the requested output.".to_string(),
            adverse_condition: None,
        }]);
        let ledger = AsgardExecutionLedger {
            entries: vec![AsgardLedgerEntry {
                id: "L1".to_string(),
                step: 2,
                command: "cargo test output_contract".to_string(),
                exit_code: Some(0),
                output_bytes: 2,
                output_sha256: asgard_sha256(b"ok"),
                output_tail: "ok".to_string(),
            }],
            ..Default::default()
        };
        let messages = asgard_completion_review_messages(
            "fix output",
            &[],
            &[],
            &AsgardSupervisorHistory::default(),
            1,
            5,
            &checklist,
            &[(1, ledger)],
            "diff --git a/src/lib.rs b/src/lib.rs\n+pub fn output() {}\n".to_string(),
            "diff --git a/tests/output_test.rs b/tests/output_test.rs\n+assert_eq!()\n".to_string(),
            0,
            "<candidate_trajectories>selected lane</candidate_trajectories>".to_string(),
            None,
            String::new(),
        );

        let suffix = asgard_message_text(messages.last().expect("terminal message"));
        assert!(suffix.contains("<canonical_execution_ledger"));
        assert!(
            suffix.contains("<terminal_non_test_patch cumulative_from_task_baseline=\"true\">")
        );
        assert!(suffix.contains("<terminal_test_patch cumulative_from_task_baseline=\"true\">"));
        assert!(suffix.contains("<candidate_trajectories>selected lane</candidate_trajectories>"));
        assert!(suffix.contains("Your select_trajectory call must return one contracts row"));
    }

    #[test]
    fn asgard_completion_review_dossier_includes_prior_review_rows_after_checklist() {
        let checklist = asgard_render_task_contract_checklist(vec![AsgardTaskContract {
            id: "C1".to_string(),
            kind: "execution".to_string(),
            text: "The command must produce the requested output.".to_string(),
            adverse_condition: None,
        }]);
        let prior_review = AsgardPriorCompletionReview {
            window: 2,
            rows: vec![AsgardContractRow {
                id: "C1".to_string(),
                status: "verified".to_string(),
                evidence: "w1l0m3 exercised output contract".to_string(),
                adverse_condition_evidence: Some("omitted from prior review message".to_string()),
            }],
            decision: AsgardSupervisorDecision {
                winner: 0,
                complete: false,
                advices: vec![Some("produce missing evidence".to_string())],
                next_window_steps: Some(1),
                state_summary: "prior rejection".to_string(),
                contracts: None,
            },
            reviewed_patch: Vec::new(),
            evidence_fingerprints: BTreeSet::new(),
        };

        let messages = asgard_completion_review_messages(
            "fix output",
            &[],
            &[],
            &AsgardSupervisorHistory::default(),
            1,
            5,
            &checklist,
            &[],
            "diff --git a/src/lib.rs b/src/lib.rs\n+pub fn output() {}\n".to_string(),
            "diff --git a/tests/output_test.rs b/tests/output_test.rs\n+assert_eq!()\n".to_string(),
            0,
            "<candidate_trajectories>selected lane</candidate_trajectories>".to_string(),
            Some(&prior_review),
            "<window_delta window=\"3\">\n+delta text\n</window_delta>".to_string(),
        );

        let checklist_position = messages
            .iter()
            .position(|message| asgard_message_text(message).contains("<task_contract_checklist"))
            .expect("checklist message");
        let prior_message = asgard_message_text(&messages[checklist_position + 1]);
        assert!(prior_message.contains("<prior_review_rows window=\"2\">"));
        assert!(prior_message.contains("\"id\": \"C1\""));
        assert!(prior_message.contains("\"status\": \"verified\""));
        assert!(prior_message.contains("\"evidence\": \"w1l0m3 exercised output contract\""));
        assert!(!prior_message.contains("adverse_condition_evidence"));
        assert!(prior_message.contains("<prior_review_delta>"));
        assert!(prior_message.contains("+delta text"));

        let suffix = asgard_message_text(messages.last().expect("terminal message"));
        assert!(suffix.contains("Keep verified rows terse"));
        assert!(suffix.contains("carried-forward: unchanged since window N review"));

        let without_prior = asgard_completion_review_messages(
            "fix output",
            &[],
            &[],
            &AsgardSupervisorHistory::default(),
            1,
            5,
            &checklist,
            &[],
            String::new(),
            String::new(),
            0,
            "<candidate_trajectories>selected lane</candidate_trajectories>".to_string(),
            None,
            String::new(),
        );
        let rendered_without_prior = without_prior
            .iter()
            .map(asgard_message_text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!rendered_without_prior.contains("<prior_review_rows"));
        assert!(!rendered_without_prior.contains("<prior_review_delta"));
    }

    #[test]
    fn asgard_dossier_keeps_each_trajectory_item_exactly_once() {
        let mut assistant = ChatMessage::assistant("candidate conclusion");
        assistant.reasoning_content = Some("unique deliberation".to_string());
        assistant.tool_calls = Some(vec![crate::llm_client::ToolCall {
            id: "call-1".to_string(),
            r#type: "function".to_string(),
            function: crate::llm_client::FunctionCall {
                name: "read_file".to_string(),
                arguments: r#"{"file_path":"src/main.rs"}"#.to_string(),
            },
        }]);
        let large_result = format!("BEGIN{}END", "x".repeat(9_000));
        let messages = vec![
            assistant,
            ChatMessage::tool_result("call-1", "read_file", large_result),
            ChatMessage::tool_result("call-2", "edit", "small exact result"),
        ];

        let dossier = render_asgard_dossier_messages(&messages);

        assert_eq!(dossier.matches("unique deliberation").count(), 1);
        assert_eq!(dossier.matches(r#"{"file_path":"src/main.rs"}"#).count(), 1);
        assert!(dossier.contains("candidate conclusion"));
        assert!(dossier.contains("BEGIN"));
        assert!(dossier.contains("END"));
        assert_eq!(dossier.matches(&"x".repeat(9_000)).count(), 1);
        assert!(dossier.contains("small exact result"));
    }

    #[test]
    fn asgard_supervisor_uses_prompt_after_bootstrap_instructions_as_task() {
        let messages = vec![
            ChatMessage::system("agent system prompt"),
            ChatMessage::user("# AGENTS.md instructions\nFollow repository policy."),
            ChatMessage::user("Implement resolved Go imports and preserve nested declarations."),
        ];

        assert_eq!(
            asgard_original_task(&messages),
            "Implement resolved Go imports and preserve nested declarations."
        );
    }

    #[test]
    fn asgard_canonicalizes_candidate_paths_across_full_history() {
        let old = Path::new("/tmp/asgard-old/");
        let live = Path::new("/work/repo");
        let mut message = ChatMessage::assistant("worked in /tmp/asgard-old/src");
        message.tool_calls = Some(vec![crate::llm_client::ToolCall {
            id: "call-1".to_string(),
            r#type: "function".to_string(),
            function: crate::llm_client::FunctionCall {
                name: "read_file".to_string(),
                arguments: r#"{\"path\":\"/tmp/asgard-old/src/main.rs\"}"#.to_string(),
            },
        }]);
        message.reasoning_content = Some("check /tmp/asgard-old".to_string());
        let tool_result = ChatMessage::tool_result(
            "call-2",
            "get_active_workspace",
            r#"{"workspace_path":"/tmp/asgard-old"}"#,
        );
        let mut messages = vec![message, tool_result];

        rewrite_asgard_cwd(&mut messages, old, live);

        let message = &messages[0];
        assert!(asgard_message_text(message).contains("/work/repo/src"));
        let call = &message.tool_calls.as_ref().unwrap()[0];
        assert!(call.function.arguments.contains("/work/repo/src/main.rs"));
        assert_eq!(
            message.reasoning_content.as_deref(),
            Some("check /work/repo")
        );
        assert!(asgard_message_text(&messages[1]).contains("/work/repo"));
        assert!(!asgard_message_text(&messages[1]).contains("asgard-old"));
    }

    #[test]
    fn asgard_select_tool_is_fail_closed_and_candidate_bounded() {
        let tool = asgard_select_trajectory_tool(3, 5);
        assert_eq!(tool.function.name, "select_trajectory");
        let schema = tool.function.parameters;
        assert_eq!(schema["properties"]["winner"]["maximum"], 2);
        assert_eq!(
            schema["required"],
            serde_json::json!(["winner", "complete", "state_summary", "advices"])
        );
        assert!(schema["properties"].get("next_candidate_count").is_none());
        assert_eq!(schema["properties"]["next_window_steps"]["minimum"], 1);
        assert_eq!(schema["properties"]["next_window_steps"]["maximum"], 10);
        assert_eq!(schema["properties"]["advices"]["minItems"], 0);
        assert_eq!(schema["properties"]["advices"]["maxItems"], 5);
        assert_eq!(schema["properties"]["advices"]["uniqueItems"], true);
        assert_eq!(
            schema["properties"]["advices"]["items"]["required"],
            serde_json::json!(["strategy", "scope_basis"])
        );
        assert_eq!(schema["properties"]["contracts"]["maxItems"], 90);
        assert_eq!(
            schema["properties"]["contracts"]["items"]["properties"]["status"]["enum"],
            serde_json::json!(["verified", "violated", "unverified"])
        );
    }

    #[test]
    fn asgard_initial_advice_tool_requires_shared_horizon_and_exact_advices() {
        let tool = asgard_advise_trajectories_tool(2);
        assert_eq!(tool.function.name, "advise_trajectories");
        let schema = tool.function.parameters;
        assert_eq!(
            schema["required"],
            serde_json::json!(["next_window_steps", "state_summary", "advices"])
        );
        assert!(schema["properties"].get("next_candidate_count").is_none());
        assert_eq!(schema["properties"]["next_window_steps"]["minimum"], 1);
        assert_eq!(schema["properties"]["next_window_steps"]["maximum"], 10);
        assert_eq!(schema["properties"]["advices"]["minItems"], 1);
        assert_eq!(schema["properties"]["advices"]["maxItems"], 2);

        let parsed = parse_asgard_initial_advice(
        r#"{"next_window_steps":4,"state_summary":"Start with one implementation lane and one falsification lane.","advices":[
            {"strategy":"Inspect the relevant parser and implement the narrow fix.","scope_basis":"The task asks for parser behavior."},
            {"strategy":"Write or run the boundary check first, then implement from the observed contract.","scope_basis":"The task asks for parser behavior."}
        ]}"#,
        2,
    )
    .unwrap();
        assert_eq!(parsed.next_window_steps, 4);
        assert_eq!(parsed.advices.len(), 2);
        assert!(
            parse_asgard_initial_advice(
                r#"{"next_window_steps":0,"state_summary":"too short","advices":[
                {"strategy":"a","scope_basis":"task"},
                {"strategy":"b","scope_basis":"task"}
            ]}"#,
                2,
            )
            .is_err()
        );
    }

    #[test]
    fn asgard_initial_advice_prompt_contains_no_candidate_bootstrap() {
        let messages = asgard_initial_advice_messages("Implement the requested API.", 3);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        let prompt = asgard_message_text(&messages[1]);
        assert!(prompt.contains("Implement the requested API."));
        assert!(prompt.contains("<initial_advice_procedure>"));
        assert!(!prompt.contains("selected_trajectory_initial"));
    }

    #[tokio::test]
    async fn asgard_initial_advice_reminds_after_an_unadvertised_tool() {
        let valid_call = supervisor_tool_call(
            "advice-call",
            "advise_trajectories",
            serde_json::json!({
                "next_window_steps": 4,
                "state_summary": "Begin with implementation, contract discovery, and falsification lanes.",
                "advices": [
                    {"strategy":"Inspect and implement the narrow path.","scope_basis":"The task requests the behavior."},
                    {"strategy":"Locate existing contracts and tests before editing.","scope_basis":"The task requests compatibility."},
                    {"strategy":"Probe the riskiest assumption, then implement from evidence.","scope_basis":"The task requires correct boundary behavior."}
                ]
            }),
        );
        let backend = ScriptedSupervisorBackend::new(vec![
            LlmResponse::ToolCalls {
                text: String::new(),
                reasoning_content: Some("I should inspect the repository.".to_string()),
                calls: vec![supervisor_tool_call(
                    "wrong-call",
                    "bash",
                    serde_json::json!({"command":"ls"}),
                )],
                usage: crate::llm_client::TokenUsage::default(),
            },
            LlmResponse::ToolCalls {
                text: String::new(),
                reasoning_content: None,
                calls: vec![valid_call],
                usage: crate::llm_client::TokenUsage::default(),
            },
        ]);

        let (advice, _) = run_asgard_initial_advice_tool_steps(
            &backend,
            asgard_initial_advice_messages("Implement the requested API.", 3),
            AsgardSupervisorToolContext {
                model: "deepseek::deepseek-v4-pro",
                candidate_count: 3,
                max_candidate_count: 3,
                idle_timeout: IdleTimeouts::uniform(std::time::Duration::from_secs(1)),
                audit: None,
                required_winner: None,
                checklist_ids: &[],
                carry_forward_allowed: false,
            },
            tokio_util::sync::CancellationToken::new(),
            None,
        )
        .await;

        assert_eq!(
            advice.expect("valid reminder recovery").next_window_steps,
            4
        );
        let requests = backend.requests.lock().expect("request lock");
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].tool_names, vec!["advise_trajectories"]);
        assert!(
            asgard_message_text(requests[1].messages.last().expect("reminder"))
                .contains("Only advise_trajectories is available")
        );
    }

    #[tokio::test]
    async fn asgard_supervisor_retries_empty_advice_set() {
        let response = |id: &str, advices: serde_json::Value| LlmResponse::ToolCalls {
            text: String::new(),
            reasoning_content: None,
            calls: vec![supervisor_tool_call(
                id,
                "select_trajectory",
                serde_json::json!({
                    "winner": 0,
                    "complete": false,
                    "next_window_steps": 2,
                    "state_summary": "Work remains.",
                    "advices": advices,
                }),
            )],
            usage: crate::llm_client::TokenUsage::default(),
        };
        let backend = ScriptedSupervisorBackend::new(vec![
            response("invalid-empty", serde_json::json!([])),
            response(
                "valid-advice",
                serde_json::json!([
                    {"strategy":"fix the serial bug","scope_basis":"task"}
                ]),
            ),
        ]);

        let (decision, _) = run_asgard_supervisor_tool_steps(
            &backend,
            vec![ChatMessage::user("dossier")],
            AsgardSupervisorToolContext {
                model: "deepseek::deepseek-v4-pro",
                candidate_count: 2,
                max_candidate_count: 5,
                idle_timeout: IdleTimeouts::uniform(std::time::Duration::from_secs(1)),
                audit: None,
                required_winner: None,
                checklist_ids: &[],
                carry_forward_allowed: false,
            },
            tokio_util::sync::CancellationToken::new(),
            None,
        )
        .await;

        let (decision, _) = decision.expect("corrected decision");
        assert_eq!(decision.advices.len(), 1);
        let requests = backend.requests.lock().expect("request lock");
        assert_eq!(requests.len(), 2);
        assert!(requests[1].messages.iter().any(|message| {
            message.role == "tool" && asgard_message_text(message).contains("1-5 distinct advices")
        }));
    }

    #[tokio::test]
    async fn asgard_supervisor_reminds_once_and_only_advertises_selector() {
        let selection_call = supervisor_tool_call(
            "selection-call",
            "select_trajectory",
            serde_json::json!({
                "winner": 1,
                "complete": false,
                "next_window_steps": 4,
                "state_summary": "Lane 2 has the strongest implementation but lacks verification.",
                "advices": [
                    {
                        "strategy": "Run the focused parser tests and update_plan with the remaining work.",
                        "scope_basis": "Verify the requested parser behavior."
                    },
                    {
                        "strategy": "Audit malformed input handling, then run its tests.",
                        "scope_basis": "Verify the requested parser behavior."
                    }
                ]
            }),
        );
        let backend = ScriptedSupervisorBackend::new(vec![
            LlmResponse::Text {
                text: "I should choose carefully.".to_string(),
                reasoning_content: Some("Compare the evidence first.".to_string()),
                usage: crate::llm_client::TokenUsage {
                    input_tokens: 10,
                    ..Default::default()
                },
            },
            LlmResponse::ToolCalls {
                text: String::new(),
                reasoning_content: Some("Lane 2 is best.".to_string()),
                calls: vec![selection_call],
                usage: crate::llm_client::TokenUsage {
                    output_tokens: 3,
                    ..Default::default()
                },
            },
        ]);

        let (decision, usage) = run_asgard_supervisor_tool_steps(
            &backend,
            vec![
                ChatMessage::system("supervise"),
                ChatMessage::user("dossier"),
            ],
            AsgardSupervisorToolContext {
                model: "deepseek::deepseek-v4-pro",
                candidate_count: 2,
                max_candidate_count: 2,
                idle_timeout: IdleTimeouts::uniform(std::time::Duration::from_secs(1)),
                audit: None,
                required_winner: None,
                checklist_ids: &[],
                carry_forward_allowed: false,
            },
            tokio_util::sync::CancellationToken::new(),
            None,
        )
        .await;

        let (decision, _stats) = decision.expect("supervisor decision");
        assert_eq!(decision.winner, 1);
        assert!(!decision.complete);
        assert_eq!(decision.next_window_steps, Some(4));
        assert!(
            decision.advices[0]
                .as_deref()
                .is_some_and(|advice| advice.contains("update_plan"))
        );
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 3);

        let requests = backend.requests.lock().expect("request lock");
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].model, "deepseek::deepseek-v4-pro");
        assert_eq!(requests[0].tool_names, vec!["select_trajectory"]);
        assert_eq!(requests[0].reasoning_effort, None);
        let replay = &requests[1].messages;
        assert_eq!(replay[2].role, "assistant");
        assert_eq!(
            replay[2].reasoning_content.as_deref(),
            Some("Compare the evidence first.")
        );
        assert_eq!(replay[3].role, "user");
        assert!(asgard_message_text(&replay[3]).contains("select_trajectory now"));
    }

    #[tokio::test]
    async fn asgard_supervisor_accepts_selection_directly() {
        let selection_call = supervisor_tool_call(
            "selection-call",
            "select_trajectory",
            serde_json::json!({
                "winner": 0,
                "complete": true,
                "state_summary": "The selected lane satisfies the task and passed verification.",
                "advices": []
            }),
        );
        let backend = ScriptedSupervisorBackend::new(vec![LlmResponse::ToolCalls {
            text: String::new(),
            reasoning_content: Some("The evidence is sufficient.".to_string()),
            calls: vec![selection_call],
            usage: crate::llm_client::TokenUsage::default(),
        }]);

        let (decision, _) = run_asgard_supervisor_tool_steps(
            &backend,
            vec![ChatMessage::user("dossier")],
            AsgardSupervisorToolContext {
                model: "deepseek::deepseek-v4-pro",
                candidate_count: 1,
                max_candidate_count: 1,
                idle_timeout: IdleTimeouts::uniform(std::time::Duration::from_secs(1)),
                audit: None,
                required_winner: None,
                checklist_ids: &[],
                carry_forward_allowed: false,
            },
            tokio_util::sync::CancellationToken::new(),
            None,
        )
        .await;

        let (decision, _stats) = decision.expect("supervisor decision");
        assert_eq!(decision.winner, 0);
        assert!(decision.complete);
        let requests = backend.requests.lock().expect("request lock");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].tool_names, vec!["select_trajectory"]);
    }

    #[tokio::test]
    async fn asgard_completion_review_keeps_the_provisionally_selected_lane() {
        let wrong_lane = supervisor_tool_call(
            "wrong-lane",
            "select_trajectory",
            serde_json::json!({
                "winner": 0,
                "complete": true,
                "state_summary": "A discarded lane looked preferable.",
                "advices": []
            }),
        );
        let reviewed_lane = supervisor_tool_call(
            "reviewed-lane",
            "select_trajectory",
            serde_json::json!({
                "winner": 1,
                "complete": false,
                "next_window_steps": 3,
                "state_summary": "The isolated endpoint still needs focused verification.",
                "advices": [
                    {
                        "strategy": "Trace the boundary contract and repair it.",
                        "scope_basis": "The task explicitly requires the boundary behavior."
                    },
                    {
                        "strategy": "Add and run a focused regression for the boundary.",
                        "scope_basis": "The current verification omits that behavior."
                    },
                    {
                        "strategy": "Independently audit the implementation path.",
                        "scope_basis": "Confirm the selected endpoint before completion."
                    }
                ]
            }),
        );
        let backend = ScriptedSupervisorBackend::new(vec![
            LlmResponse::ToolCalls {
                text: String::new(),
                reasoning_content: None,
                calls: vec![wrong_lane],
                usage: crate::llm_client::TokenUsage::default(),
            },
            LlmResponse::ToolCalls {
                text: String::new(),
                reasoning_content: None,
                calls: vec![reviewed_lane],
                usage: crate::llm_client::TokenUsage::default(),
            },
        ]);

        let (decision, _) = run_asgard_supervisor_tool_steps(
            &backend,
            vec![ChatMessage::user("isolated endpoint")],
            AsgardSupervisorToolContext {
                model: "deepseek::deepseek-v4-pro",
                candidate_count: 3,
                max_candidate_count: 3,
                idle_timeout: IdleTimeouts::uniform(std::time::Duration::from_secs(1)),
                audit: None,
                required_winner: Some(1),
                checklist_ids: &[],
                carry_forward_allowed: false,
            },
            tokio_util::sync::CancellationToken::new(),
            None,
        )
        .await;

        let (decision, _stats) = decision.expect("completion review decision");
        assert_eq!(decision.winner, 1);
        assert!(!decision.complete);
        let requests = backend.requests.lock().expect("request lock");
        assert_eq!(requests.len(), 2);
        assert!(requests[1].messages.iter().any(|message| {
            message.role == "tool"
                && asgard_message_text(message).contains("must keep selected lane 1")
        }));
    }

    #[test]
    fn asgard_audit_catalog_is_lane_aware_stable_and_read_only() {
        let definition = |name: &str| ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: name.to_string(),
                description: format!("{name} description"),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }),
            },
        };
        let tools = asgard_audit_tool_definitions(
            vec![
                definition("read_file"),
                definition("search_symbols"),
                definition("run_shell_command"),
                definition("write_file"),
            ],
            3,
        );

        assert_eq!(
            tools
                .iter()
                .map(|tool| tool.function.name.as_str())
                .collect::<Vec<_>>(),
            vec!["read_file", "search_symbols"]
        );
        for tool in tools {
            assert_eq!(tool.function.parameters["properties"]["lane"]["minimum"], 0);
            assert_eq!(tool.function.parameters["properties"]["lane"]["maximum"], 2);
            assert!(
                tool.function.parameters["properties"]["lane"]
                    .get("enum")
                    .is_none()
            );
            assert!(
                tool.function.parameters["required"]
                    .as_array()
                    .is_some_and(|required| required.iter().any(|value| value == "lane"))
            );
            assert!(tool.function.description.contains("Read-only audit"));
        }
    }

    #[test]
    fn asgard_completion_review_preserves_the_exact_supervisor_tool_catalog() {
        let audit_definition = ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: "read_file".to_string(),
                description: "Read-only audit".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "lane": {"type": "integer", "minimum": 0, "maximum": 2}
                    },
                    "required": ["file_path", "lane"]
                }),
            },
        };
        let registries = Vec::new();
        let candidates = Vec::new();
        let selection_context = AsgardSupervisorToolContext {
            model: "deepseek::deepseek-v4-pro",
            candidate_count: 3,
            max_candidate_count: 3,
            idle_timeout: IdleTimeouts::uniform(std::time::Duration::from_secs(1)),
            audit: Some(AsgardAuditContext {
                registries: &registries,
                candidates: &candidates,
                definitions: vec![audit_definition.clone()],
                allowed_lane: None,
                retained_windows: &[],
                window: 0,
            }),
            required_winner: None,
            checklist_ids: &[],
            carry_forward_allowed: false,
        };
        let completion_context = AsgardSupervisorToolContext {
            model: "deepseek::deepseek-v4-pro",
            candidate_count: 3,
            max_candidate_count: 3,
            idle_timeout: IdleTimeouts::uniform(std::time::Duration::from_secs(1)),
            audit: Some(AsgardAuditContext {
                registries: &registries,
                candidates: &candidates,
                definitions: vec![audit_definition],
                allowed_lane: Some(1),
                retained_windows: &[],
                window: 0,
            }),
            required_winner: Some(1),
            checklist_ids: &[],
            carry_forward_allowed: false,
        };

        assert_eq!(
            serde_json::to_value(asgard_supervisor_tool_definitions(&selection_context)).unwrap(),
            serde_json::to_value(asgard_supervisor_tool_definitions(&completion_context)).unwrap(),
        );
    }

    #[tokio::test]
    async fn asgard_supervisor_can_read_before_selecting() {
        let cwd = tempfile::tempdir().expect("candidate cwd");
        init_git_repo(cwd.path());
        std::fs::write(
            cwd.path().join("evidence.txt"),
            "decisive repository evidence\n",
        )
        .expect("write evidence fixture");
        run_git(cwd.path(), &["add", "evidence.txt"]);
        run_git(cwd.path(), &["commit", "-m", "seed"]);
        std::fs::write(
            cwd.path().join("evidence.txt"),
            "changed repository evidence\n",
        )
        .expect("change evidence fixture");
        let registry = Arc::new(
            crate::tools::ToolRegistry::new(
                cwd.path().to_path_buf(),
                Vec::new(),
                Vec::new(),
                Arc::new(crate::skills::SkillRegistry::default()),
                Arc::new(crate::agents::AgentRegistry::default()),
                Vec::new(),
            )
            .await,
        );
        let definitions = asgard_audit_tool_definitions(registry.tool_definitions().await, 1);
        let registries = vec![registry];
        let candidates = vec![AsgardCandidate {
            index: 0,
            model: "deepseek::deepseek-v4-flash".to_string(),
            outcome: asgard_failure(anyhow::anyhow!("unused test outcome")),
            patch: Vec::new(),
            delta_patch: Vec::new(),
            window_messages: Vec::new(),
            supervisor_window_messages: Vec::new(),
            window_ledger: AsgardExecutionLedger::default(),
        }];
        let backend = ScriptedSupervisorBackend::new(vec![
            LlmResponse::ToolCalls {
                text: String::new(),
                reasoning_content: Some("I need to inspect the endpoint.".to_string()),
                calls: vec![supervisor_tool_call(
                    "read-call",
                    "read_file",
                    serde_json::json!({"lane": 0, "file_path": "evidence.txt"}),
                )],
                usage: crate::llm_client::TokenUsage::default(),
            },
            LlmResponse::ToolCalls {
                text: String::new(),
                reasoning_content: Some("The repository evidence resolves the risk.".to_string()),
                calls: vec![supervisor_tool_call(
                    "selection-call",
                    "select_trajectory",
                    serde_json::json!({
                        "winner": 0,
                        "complete": true,
                        "state_summary": "The selected lane satisfies the task after repository inspection.",
                        "advices": []
                    }),
                )],
                usage: crate::llm_client::TokenUsage::default(),
            },
        ]);

        let (decision, _) = run_asgard_supervisor_tool_steps(
            &backend,
            vec![ChatMessage::user("terminal dossier")],
            AsgardSupervisorToolContext {
                model: "deepseek::deepseek-v4-pro",
                candidate_count: 1,
                max_candidate_count: 1,
                idle_timeout: IdleTimeouts::uniform(std::time::Duration::from_secs(1)),
                audit: Some(AsgardAuditContext {
                    registries: &registries,
                    candidates: &candidates,
                    definitions,
                    allowed_lane: None,
                    retained_windows: &[],
                    window: 0,
                }),
                required_winner: None,
                checklist_ids: &[],
                carry_forward_allowed: false,
            },
            tokio_util::sync::CancellationToken::new(),
            None,
        )
        .await;

        assert!(decision.expect("terminal decision").0.complete);
        let requests = backend.requests.lock().expect("request lock");
        assert_eq!(requests.len(), 2);
        assert!(requests[0].tool_names.contains(&"read_file".to_string()));
        assert!(requests[0].tool_names.contains(&"grep_search".to_string()));
        assert!(
            requests[0]
                .tool_names
                .contains(&"list_directory".to_string())
        );
        assert!(
            !requests[0]
                .tool_names
                .contains(&"run_shell_command".to_string())
        );
        let replay = &requests[1].messages;
        assert!(replay.iter().any(|message| {
            message.role == "tool"
                && asgard_message_text(message).contains("changed repository evidence")
        }));
    }

    #[tokio::test]
    async fn asgard_supervisor_retrieval_rounds_do_not_consume_the_audit_budget() {
        // Three view_tool_call rounds interleaved with three audit rounds must
        // still leave all three audit rounds spendable: retrieval is refunded.
        // If it were charged, the fourth response here would be rejected as
        // out-of-budget and the selection would never be reached.
        let cwd = tempfile::tempdir().expect("candidate cwd");
        init_git_repo(cwd.path());
        std::fs::write(cwd.path().join("evidence.txt"), "repository evidence\n")
            .expect("write evidence fixture");
        run_git(cwd.path(), &["add", "evidence.txt"]);
        run_git(cwd.path(), &["commit", "-m", "seed"]);
        let registry = Arc::new(
            crate::tools::ToolRegistry::new(
                cwd.path().to_path_buf(),
                Vec::new(),
                Vec::new(),
                Arc::new(crate::skills::SkillRegistry::default()),
                Arc::new(crate::agents::AgentRegistry::default()),
                Vec::new(),
            )
            .await,
        );
        let definitions = asgard_audit_tool_definitions(registry.tool_definitions().await, 1);
        let registries = vec![registry];
        let candidates = vec![asgard_test_candidate(
            0,
            vec![
                ChatMessage::assistant_tool_calls(vec![supervisor_tool_call(
                    "shell-1",
                    "run_shell_command",
                    serde_json::json!({"command": "cargo test"}),
                )]),
                ChatMessage::tool_result("shell-1", "run_shell_command", "the full test output"),
            ],
        )];
        let audit_call = |id: &str| {
            supervisor_tool_call(
                id,
                "read_file",
                serde_json::json!({"lane": 0, "file_path": "evidence.txt"}),
            )
        };
        let view_call = |id: &str| {
            supervisor_tool_call(
                id,
                ASGARD_VIEW_TOOL_CALL_NAME,
                serde_json::json!({"handles": ["w0l0m1"]}),
            )
        };
        let scripted = |calls: Vec<crate::llm_client::ToolCall>| LlmResponse::ToolCalls {
            text: String::new(),
            reasoning_content: None,
            calls,
            usage: crate::llm_client::TokenUsage::default(),
        };
        let backend = ScriptedSupervisorBackend::new(vec![
            scripted(vec![view_call("view-1")]),
            scripted(vec![audit_call("audit-1")]),
            scripted(vec![view_call("view-2")]),
            scripted(vec![audit_call("audit-2")]),
            scripted(vec![view_call("view-3")]),
            scripted(vec![audit_call("audit-3")]),
            scripted(vec![supervisor_tool_call(
                "selection-call",
                "select_trajectory",
                serde_json::json!({
                    "winner": 0,
                    "complete": false,
                    "next_window_steps": 3,
                    "state_summary": "Retrieved the full outputs before selecting.",
                    "advices": [{
                        "strategy": "Finish the remaining verification.",
                        "scope_basis": "Establish the remaining task-required behavior."
                    }]
                }),
            )]),
        ]);

        let (decision, _) = run_asgard_supervisor_tool_steps(
            &backend,
            vec![ChatMessage::user("dossier")],
            AsgardSupervisorToolContext {
                model: "deepseek::deepseek-v4-pro",
                candidate_count: 1,
                max_candidate_count: 1,
                idle_timeout: IdleTimeouts::uniform(std::time::Duration::from_secs(1)),
                audit: Some(AsgardAuditContext {
                    registries: &registries,
                    candidates: &candidates,
                    definitions,
                    allowed_lane: None,
                    retained_windows: &[],
                    window: 0,
                }),
                required_winner: None,
                checklist_ids: &[],
                carry_forward_allowed: false,
            },
            tokio_util::sync::CancellationToken::new(),
            None,
        )
        .await;

        let (decision, _stats) = decision.expect("selection after interleaved retrieval");
        assert_eq!(decision.winner, 0);

        let requests = backend.requests.lock().expect("request lock");
        assert_eq!(requests.len(), 7);
        // view_tool_call is offered alongside the checkout audit tools.
        assert!(
            requests[0]
                .tool_names
                .contains(&ASGARD_VIEW_TOOL_CALL_NAME.to_string())
        );
        let conversation = requests
            .last()
            .expect("final request")
            .messages
            .iter()
            .map(asgard_message_text)
            .collect::<Vec<_>>()
            .join("\n");
        // Retrieval returned the untruncated result the compact line omitted.
        assert!(conversation.contains("the full test output"));
        assert!(conversation.contains("did not consume your information-gathering budget"));
        // All three audit rounds survived the interleaved retrieval.
        assert!(!conversation.contains("Audit budget exhausted"));
    }

    #[tokio::test]
    async fn asgard_supervisor_exhausts_audit_then_retries_selection_in_same_conversation() {
        let cwd = tempfile::tempdir().expect("candidate cwd");
        init_git_repo(cwd.path());
        std::fs::write(cwd.path().join("evidence.txt"), "repository evidence\n")
            .expect("write evidence fixture");
        run_git(cwd.path(), &["add", "evidence.txt"]);
        run_git(cwd.path(), &["commit", "-m", "seed"]);
        let registry = Arc::new(
            crate::tools::ToolRegistry::new(
                cwd.path().to_path_buf(),
                Vec::new(),
                Vec::new(),
                Arc::new(crate::skills::SkillRegistry::default()),
                Arc::new(crate::agents::AgentRegistry::default()),
                Vec::new(),
            )
            .await,
        );
        let definitions = asgard_audit_tool_definitions(registry.tool_definitions().await, 1);
        let registries = vec![registry];
        let candidates = vec![AsgardCandidate {
            index: 0,
            model: "deepseek::deepseek-v4-flash".to_string(),
            outcome: asgard_failure(anyhow::anyhow!("unused test outcome")),
            patch: Vec::new(),
            delta_patch: Vec::new(),
            window_messages: Vec::new(),
            supervisor_window_messages: Vec::new(),
            window_ledger: AsgardExecutionLedger::default(),
        }];
        let audit_call = |id: &str| {
            supervisor_tool_call(
                id,
                "read_file",
                serde_json::json!({"lane": 0, "file_path": "evidence.txt"}),
            )
        };
        let selection_call = supervisor_tool_call(
            "selection-call",
            "select_trajectory",
            serde_json::json!({
                "winner": 0,
                "complete": false,
                "next_window_steps": 3,
                "state_summary": "Lane 1 is the best incomplete foundation; focused verification remains.",
                "advices": [{
                    "strategy": "Run the focused verification and repair any resulting defect.",
                    "scope_basis": "Establish the remaining task-required behavior."
                }]
            }),
        );
        let backend = ScriptedSupervisorBackend::new(vec![
            LlmResponse::ToolCalls {
                text: String::new(),
                reasoning_content: Some("First consequential question.".to_string()),
                calls: vec![audit_call("audit-1")],
                usage: crate::llm_client::TokenUsage::default(),
            },
            LlmResponse::ToolCalls {
                text: String::new(),
                reasoning_content: Some("Second consequential question.".to_string()),
                calls: vec![audit_call("audit-2")],
                usage: crate::llm_client::TokenUsage::default(),
            },
            LlmResponse::ToolCalls {
                text: String::new(),
                reasoning_content: Some("Final audit question.".to_string()),
                calls: vec![audit_call("audit-3")],
                usage: crate::llm_client::TokenUsage::default(),
            },
            LlmResponse::Text {
                text: "Lane 1 is best, but I forgot the tool call.".to_string(),
                reasoning_content: None,
                usage: crate::llm_client::TokenUsage::default(),
            },
            LlmResponse::ToolCalls {
                text: String::new(),
                reasoning_content: Some("I should inspect once more.".to_string()),
                calls: vec![audit_call("ignored-audit")],
                usage: crate::llm_client::TokenUsage::default(),
            },
            LlmResponse::ToolCalls {
                text: String::new(),
                reasoning_content: Some("Selecting under the remaining uncertainty.".to_string()),
                calls: vec![selection_call, audit_call("mixed-ignored-audit")],
                usage: crate::llm_client::TokenUsage::default(),
            },
        ]);

        let (decision, _) = run_asgard_supervisor_tool_steps(
            &backend,
            vec![ChatMessage::user("terminal dossier")],
            AsgardSupervisorToolContext {
                model: "deepseek::deepseek-v4-pro",
                candidate_count: 1,
                max_candidate_count: 1,
                idle_timeout: IdleTimeouts::uniform(std::time::Duration::from_secs(1)),
                audit: Some(AsgardAuditContext {
                    registries: &registries,
                    candidates: &candidates,
                    definitions,
                    allowed_lane: None,
                    retained_windows: &[],
                    window: 0,
                }),
                required_winner: None,
                checklist_ids: &[],
                carry_forward_allowed: false,
            },
            tokio_util::sync::CancellationToken::new(),
            None,
        )
        .await;

        let (decision, _stats) = decision.expect("selection after retained audit conversation");
        assert_eq!(decision.winner, 0);
        assert!(!decision.complete);
        let requests = backend.requests.lock().expect("request lock");
        assert_eq!(requests.len(), 6);
        let tool_names = &requests[0].tool_names;
        assert!(
            requests
                .iter()
                .all(|request| &request.tool_names == tool_names)
        );
        assert!(
            asgard_message_text(requests[1].messages.last().expect("audit reminder"))
                .contains("2 information-gathering turns remaining")
        );
        assert!(
            asgard_message_text(requests[2].messages.last().expect("last audit reminder"))
                .contains("final information-gathering turn")
        );
        assert!(
            asgard_message_text(requests[3].messages.last().expect("selection instruction"))
                .contains("information-gathering budget is exhausted")
        );
        assert!(requests[5].messages.iter().any(|message| {
            message.role == "tool"
                && asgard_message_text(message).contains("Audit budget exhausted")
        }));
        assert!(requests[5].messages.iter().any(|message| {
            message.role == "tool" && asgard_message_text(message).contains("repository evidence")
        }));
    }

    #[tokio::test]
    async fn asgard_supervisor_rejects_unadvertised_plan_call() {
        let wrong_response = || LlmResponse::ToolCalls {
            text: String::new(),
            reasoning_content: None,
            calls: vec![supervisor_tool_call(
                "plan-call",
                "update_plan",
                serde_json::json!({
                    "plan": [{"step": "Verify the implementation", "status": "in_progress"}]
                }),
            )],
            usage: crate::llm_client::TokenUsage::default(),
        };
        let backend = ScriptedSupervisorBackend::new(vec![
            wrong_response(),
            wrong_response(),
            wrong_response(),
        ]);

        let (decision, _) = run_asgard_supervisor_tool_steps(
            &backend,
            vec![ChatMessage::user("dossier")],
            AsgardSupervisorToolContext {
                model: "deepseek::deepseek-v4-pro",
                candidate_count: 1,
                max_candidate_count: 1,
                idle_timeout: IdleTimeouts::uniform(std::time::Duration::from_secs(1)),
                audit: None,
                required_winner: None,
                checklist_ids: &[],
                carry_forward_allowed: false,
            },
            tokio_util::sync::CancellationToken::new(),
            None,
        )
        .await;

        let error = decision.expect_err("update_plan is not a supervisor tool");
        assert!(
            error
                .to_string()
                .contains("called unexpected tool(s) `update_plan`")
        );
        let requests = backend.requests.lock().expect("request lock");
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[0].tool_names, vec!["select_trajectory"]);
        assert!(
            asgard_message_text(requests[1].messages.last().expect("reminder"))
                .contains("Call select_trajectory now")
        );
    }
    #[test]
    fn asgard_test_file_inventory_surfaces_candidate_authorship() {
        let patch = b"diff --git a/test/Core/ExistingTests.cs b/test/Core/ExistingTests.cs\n\
                  --- a/test/Core/ExistingTests.cs\n\
                  +++ b/test/Core/ExistingTests.cs\n\
                  diff --git a/test/Core/NewValidatorTests.cs b/test/Core/NewValidatorTests.cs\n\
                  new file mode 100644\n\
                  --- /dev/null\n\
                  +++ b/test/Core/NewValidatorTests.cs\n";

        let (created, modified) = asgard_patch_test_inventory(patch);

        assert_eq!(created, vec!["test/Core/NewValidatorTests.cs"]);
        assert_eq!(modified, vec!["test/Core/ExistingTests.cs"]);
    }

    #[test]
    fn asgard_production_inventory_excludes_test_files() {
        let patch = b"diff --git a/src/Parser.java b/src/Parser.java\n\
                  --- a/src/Parser.java\n\
                  +++ b/src/Parser.java\n\
                  @@ -1 +1 @@\n\
                  -old\n\
                  +new\n\
                  diff --git a/src/ParserTest.java b/src/ParserTest.java\n\
                  --- a/src/ParserTest.java\n\
                  +++ b/src/ParserTest.java\n\
                  @@ -1 +1 @@\n\
                  -old test\n\
                  +new test\n";

        assert_eq!(
            asgard_patch_production_inventory(patch),
            vec!["src/Parser.java"]
        );
    }

    #[test]
    fn asgard_production_inventory_surfaces_renames_across_test_boundary() {
        let patch = b"diff --git a/src/Parser.java b/tests/ParserTest.java\n\
                  similarity index 90%\n\
                  rename from src/Parser.java\n\
                  rename to tests/ParserTest.java\n";

        assert_eq!(
            asgard_patch_production_inventory(patch),
            vec!["src/Parser.java"]
        );
    }
}
