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

pub(crate) const ASGARD_SUMMARIZE_WINDOW_TOOL_NAME: &str = "summarize_candidate_window";
pub(crate) const ASGARD_DETERMINISTIC_HANDOFF_MAX_BYTES: usize = 16_000;
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
    pub(crate) next_candidate_count: Option<usize>,
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
    pub(crate) next_candidate_count: usize,
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
                "next_candidate_count": ASGARD_MIN_CANDIDATES,
                "next_window_steps": ASGARD_MIN_WINDOW_STEPS,
            }));
            AsgardSupervisorInitialAdvice {
                advices: vec![Some(
                    "The supervisor could not produce a valid initial routing decision. \
                     Inspect the task and repository, identify the most consequential \
                     next action, and make concrete progress before the next review."
                        .to_string(),
                )],
                next_candidate_count: ASGARD_MIN_CANDIDATES,
                next_window_steps: ASGARD_MIN_WINDOW_STEPS,
                state_summary: "Initial supervisor routing failed validation; using a traced one-lane recovery window."
                    .to_string(),
            }
        }
    };
    let mut current_candidate_count = initial_advice.next_candidate_count;
    let mut current_window_steps = initial_advice.next_window_steps;
    let mut consecutive_winner_failures = 0usize;
    let mut next_advices: Option<Vec<Option<String>>> = Some(initial_advice.advices.clone());
    let mut current_supervisor_state_summary = initial_advice.state_summary.clone();
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
            let assessment_state_summary = current_supervisor_state_summary.clone();
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
                let window_ledger = asgard_extract_execution_ledger(&window_messages);
                let deterministic_handoff = asgard_deterministic_candidate_handoff(
                    &window_messages,
                    outcome.current_plan.as_ref(),
                );
                let (brief, grading_usage, deterministic_stats) =
                    if let Some((messages, raw_bytes, packed_bytes)) = deterministic_handoff {
                        (
                            Ok(messages),
                            crate::llm_client::TokenUsage::default(),
                            Some((raw_bytes, packed_bytes)),
                        )
                    } else {
                        let (brief, usage) = run_asgard_candidate_window_summary(
                            llm.as_ref(),
                            AsgardCandidateAssessmentContext {
                                model: &model,
                                window,
                                lane: index,
                                reasoning_effort,
                                service_tier,
                                idle_timeout,
                                original_task: &assessment_original_task,
                                canonical_state_summary: &assessment_state_summary,
                                current_plan: outcome.current_plan.as_ref(),
                            },
                            cancel,
                            &window_messages,
                        )
                        .await;
                        (brief.map(|brief| {
                            vec![ChatMessage::assistant(format!(
                                "<candidate_window_brief>\n{brief}\n</candidate_window_brief>"
                            ))]
                        }), usage, None)
                    };
                let supervisor_window_messages = match brief {
                    Ok(messages) => {
                        let brief = render_asgard_dossier_messages(&messages);
                        if std::env::var_os("ASGARD_CAPTURE_WINDOW_SUMMARIES").is_some() {
                            tracing::info!(
                                lane = index + 1,
                                raw_window = %render_asgard_dossier_messages(&window_messages),
                                candidate_brief = %brief,
                                "captured Asgard candidate window summary for review"
                            );
                        }
                        let original_bytes = render_asgard_dossier_messages(&window_messages).len();
                        let slim_bytes = brief.len();
                        if let Some((raw_bytes, packed_bytes)) = deterministic_stats {
                            tracing::info!(
                                lane = index + 1,
                                model,
                                original_bytes,
                                slim_bytes,
                                raw_bytes,
                                packed_bytes,
                                "used deterministic exact Asgard candidate handoff"
                            );
                            crate::trace_logging::append_trace_record(serde_json::json!({
                                "type": "asgard_candidate_handoff",
                                "window": window,
                                "lane": index,
                                "mode": "deterministic_exact",
                                "raw_bytes": raw_bytes,
                                "packed_bytes": packed_bytes,
                                "supervisor_bytes": slim_bytes,
                            }));
                        } else {
                            tracing::info!(
                                lane = index + 1,
                                model,
                                original_bytes,
                                slim_bytes,
                                "summarized Asgard candidate trajectory"
                            );
                            crate::trace_logging::append_trace_record(serde_json::json!({
                                "type": "asgard_candidate_handoff",
                                "window": window,
                                "lane": index,
                                "mode": "llm_summary",
                                "raw_bytes": original_bytes,
                                "supervisor_bytes": slim_bytes,
                            }));
                        }
                        messages
                    }
                    Err(error) => {
                        tracing::warn!(
                            lane = index + 1,
                            model,
                            "keeping full Asgard trajectory after window summarization failed: {error:#}"
                        );
                        window_messages.clone()
                    }
                };
                (
                    index,
                    model,
                    outcome,
                    patches,
                    window_messages,
                    supervisor_window_messages,
                    window_ledger,
                    grading_usage,
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
            grading_usage,
        ) in futures::future::join_all(futures).await
        {
            trace_asgard_phase_usage(
                "candidate_window_summary",
                &model,
                Some(window),
                Some(index),
                grading_usage,
            );
            aggregate_usage.add(outcome.usage);
            aggregate_usage.add(grading_usage);
            usage_by_model
                .entry(model.clone())
                .or_default()
                .add(outcome.usage);
            usage_by_model
                .entry(model.clone())
                .or_default()
                .add(grading_usage);
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
                    "next_candidate_count": ASGARD_MIN_CANDIDATES,
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
                    next_candidate_count: Some(ASGARD_MIN_CANDIDATES),
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
                        next_candidate_count: Some(ASGARD_MIN_CANDIDATES),
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
        current_supervisor_state_summary = decision.state_summary.clone();
        next_advices = Some(decision.advices.clone());
        if !supervisor_complete && let Some(advices) = &next_advices {
            current_candidate_count = decision
                .next_candidate_count
                .expect("incomplete Asgard decision has next_candidate_count");
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

    loop {
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
                let unexpected_calls = calls
                    .iter()
                    .filter(|call| {
                        call.function.name != "select_trajectory"
                            && !asgard_is_audit_tool(&call.function.name)
                    })
                    .map(|call| call.function.name.as_str())
                    .collect::<Vec<_>>();
                for call in &calls {
                    let output = if call.function.name == "select_trajectory" {
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
                } else if audit_phase && has_audit_calls {
                    None
                } else {
                    Some(anyhow::anyhow!(
                        "Asgard supervisor did not call select_trajectory by itself"
                    ))
                };
            }
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
    "Choose next_candidate_count from 1 through the configured maximum and next_window_steps from 1 through 10. Candidate count buys independent breadth: use more lanes when the diagnosis, architecture, contract reading, or evidence is uncertain enough that genuinely different approaches can teach you something. Do not spend lanes on cosmetic variations. If a concrete bug must be fixed before any new direction can be useful, choose one candidate to fix and verify that serial dependency first. The shared step horizon controls when comparison resumes, not when candidates should declare the task finished. Use short horizons when feedback is valuable soon and longer horizons only for clear, mechanically involved work where interruption adds little value."
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
2. Choose next_candidate_count based on how many genuinely useful independent approaches the current uncertainty supports.
3. Choose next_window_steps based on when the next comparison should be valuable, not on a fixed rollout habit.
4. Produce exactly next_candidate_count task-compliant strategies. Make them genuinely different. When using multiple lanes, include one strategy that quickly falsifies the most consequential assumption.
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

Asgard preserves one canonical trajectory. The lanes shown here all started independently from the same prior winner and repository state, ran for one shared step horizon, and now compete to become the sole next canonical state; losing work is discarded. If the endpoint is incomplete, your next_candidate_count controls the cost-versus-breadth tradeoff for fresh continuations from this winner, and next_window_steps controls when those continuations are compared again.

Selection does not require certainty. Choose the best continuation under the available evidence even when every lane is flawed or an important question remains unanswered. complete=true is a nomination that the selected endpoint plausibly satisfies the original task; a separate isolated completion review owns the terminal adjudication.
</mission>

<task_and_evidence>
The original task is authoritative. Preserve its exact externally observable contracts, including argument order, return values, error behavior, atomicity, compatibility requirements, implementation constraints, and prohibitions. Merely defining the requested symbol or compiling the code does not establish that contract. A lane that contradicts the task is not rescued by confident prose, a large patch, or aggregate green checks. Judge architectural direction, correctness, recoverability, known defects, and evidence. Investigation that establishes an important constraint can be more valuable than immediate edits.

Candidate briefs are loss-aware but candidate-authored summaries, not authoritative evidence. A candidate_window_handoff with format="deterministic_exact" is different: it mechanically preserves every unique message from a small window and replaces only byte-identical repeated tool results with explicit back-references. Treat its observations as the original trajectory record, while still judging what each command or inspection actually proves. A candidate_patch_manifest describes its cumulative changed production and test files. Normally each candidate_window_diff shows that lane's edits since the last selected decision. When candidate_diff_baseline names a current-window diff-compression anchor, that anchor is displayed first with its diff against the last selected decision, while every subsequent candidate_window_diff shows the transformation from the anchor lane to that lane. This is only diff compression: every lane still started independently from the same last selected decision, the anchor is not canonical or preferred, and lane indices remain authoritative. Use the handoffs or briefs, manifests, and correctly based diffs together: challenge contradictions and consequential unsupported claims, but do not reread the repository merely to reproduce information the dossier already establishes.

Interpret verification precisely. A successful command establishes only the behaviors its selected tests and assertions actually exercised; filters, wrappers, timeouts, zero-test selections, and missing combinations can mislead. Candidate-written tests can be valuable, but green tests changed alongside the implementation are not independent proof merely because they pass. Check whether they genuinely express the task, especially the highest-risk requirement and combinations of behaviors changed by the patch. Do not mechanically reject legitimate test or mock updates.

Prior supervisor decisions and lane advice provide continuity, not authority. Reconsider them when the original task or newer evidence disagrees.
</task_and_evidence>

<decision_depth>
First decide whether the leading lane plausibly appears terminal.

complete=true is a nomination, not a terminal verdict. A separate isolated completion review owns all terminal adjudication and re-judges the nominated endpoint against every task contract with the full evidence bundle. Nominate complete=true when the selected endpoint plausibly satisfies the task contracts; the candidate briefs and diffs are sufficient basis. Do not perform a terminal-grade audit, search for boundary counterexamples, or re-verify evidence before nominating — that depth is the review's job and paying it here duplicates the review. Do not nominate past a defect you can already see: a concrete task-relevant defect, contradiction, or plainly missing required work visible in the dossier means complete=false with advice targeting it.

If the best lanes are plainly incomplete, do not perform a terminal-grade exhaustive audit. Inspect only enough consequential evidence to rank their directions and formulate the next work. Unknowns can remain: select the best foundation, set complete=false, record the uncertainty, and delegate the needed implementation or verification to candidates in the next window.
</decision_depth>

<audit_protocol>
Lane-aware read_file, grep_search, list_directory, and Bifrost symbol tools are available for read-only evidence gathering. They cannot run builds or tests. You have at most {audit_rounds} information-gathering responses, including this one; the last opportunity will be announced. Batch related questions and stop once the answer cannot change the winner or completion judgment. After the budget is exhausted, audit calls are ignored and you must select.

Unavailable executable verification is not a reason to withhold a decision. If it is necessary to establish completeness, set complete=false and tell one or more next-window candidates exactly what behavior or command to verify. Repository auditing is evidence gathering, not another implementation rollout.
</audit_protocol>

<scope_and_completion>
Stay within the original scope. Do not repair dependencies, lockfiles, toolchains, generated machinery, warning policy, test selection, or expected outputs merely to hide a failure unless the task requires that surface or the candidate broke it. Treat a failure as environmental, pre-existing, flaky, or unrelated only with concrete evidence. Do not weaken required behavior to preserve obsolete callers or mocks.

Completion is a property of the endpoint, not of who introduced a defect. Set complete=true when the selected endpoint plausibly satisfies the exact task contracts on the basis of the candidate briefs and diffs; the isolated completion review — not this decision — adjudicates terminal completeness and will return the work to candidates if evidence falls short. Set complete=false when required work plainly remains or the dossier shows an unresolved concrete task-relevant defect. When executable verification is unavailable to you, select the best lane and delegate the specific check to the next candidate window rather than either assuming success or refusing to decide.
</scope_and_completion>

<continuation>
When incomplete, choose next_candidate_count, choose next_window_steps, and return exactly next_candidate_count concise, actionable, mutually distinct advice objects in zero-based lane order. Use more candidates only when uncertainty supports genuinely different investigations or implementations. If an obvious concrete bug must be repaired before new directions become useful, choose one candidate to fix and verify it first. Each strategy must independently comply with the task. With multiple lanes, include one strategy that tests the selected direction's most consequential unverified assumption. When the checklist records materially ambiguous readings, use separate lanes only when implementing or testing those readings will resolve the ambiguity. Advice may tell candidates to inspect source, run a focused build or test, or update_plan. Do not assert exact APIs or implementation facts that the evidence does not establish. Candidates continue normal rollouts and do not stop at Asgard window boundaries.

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
3. Decide whether that winner plausibly appears terminal. If so, nominate: complete=true requires only that the endpoint plausibly satisfies the task contracts on the candidate briefs and diffs — the isolated completion review owns terminal adjudication, so do not audit for boundary cases first. If not, investigate only questions that could change the ranking or next-window direction.
4. Treat the completion judgment as a nomination. A concrete defect, contradiction, or plainly missing required work already visible in the dossier means complete=false and becomes targeted next-window advice; do not search for more before nominating.
5. If incomplete, choose next_candidate_count, the shared horizon, and exactly next_candidate_count distinct compliant strategies. Scale breadth with uncertainty; use one lane for a concrete serial bug fix.
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

The task_contract_checklist was derived from the task text alone before any candidate work existed. Your select_trajectory call must return one contracts row per checklist id. Keep verified rows terse — one clause citing the ledger entry or quoted line (aim under 150 characters); reserve detailed prose for violated rows, unverified rows, and adverse_condition_evidence. When prior_review_rows are present, re-adjudicate only contracts plausibly affected by the prior_review_delta, plus every contract that was not verified before. For a previously verified contract untouched by the delta, carry it forward: status verified, evidence 'carried-forward: unchanged since window N review'. A carried-forward row must still be listed. Evidence rules, in strength order: (1) Execution evidence — an execution_ledger entry (cite its id) whose command demonstrably exercised the contract and exited 0 at or after the last edit of the implementing files; a green command counts only for behaviors its selected tests actually assert — confirm in the terminal_test_patch that some assertion would fail if the contract were violated; a broad suite pass does not verify a contract no test asserts; a build proves compilation; a race detector proves absence of data races, not liveness. Golden, snapshot, or fixture expectation files created or regenerated by the candidate are candidate-authored assertions: a passing comparison against them proves only that the output matches itself; for exact-output contracts, quote the actual emitted output — from a ledger entry or from the expectation-file content in the patches — and show it satisfies the contract's required shape at its boundaries, including the very first and very last elements of the stream. Evidence must be discriminating: state the most plausible wrong implementation of this contract — the wrong label or classification, the missing boundary element, the partially-correct result — and confirm the cited assertion would catch it; an assertion that only counts results, checks non-emptiness, or matches a substring usually passes under mislabeled or partially-wrong behavior. For a contract naming enumerated values, classifications, or exact labels, quote — for each scenario the contract names — the assertion line or observed ledger output showing that exact value in that exact scenario; an assertion checking a different scenario's value does not transfer. For a contract that combines, maps, or constructs from multiple positional inputs, a green test counts only if its values are pairwise distinguishable and the checked operation non-commutative — a test combining equal values or reducing with numeric addition passes under any positional swap and proves nothing about position. Absent such a test, verify positionally by inspection: quote the complete argument flow from construction to application and argue each input reaches its position; mark the row violated if the quoted flow misroutes any input, and unverified only when neither a discriminating test nor a conclusive flow reading exists. For contracts requiring a behavior across a matrix of backends, dialects, or variants, quoted code showing the new path routes through the same shared machinery as an existing verified sibling is acceptable inspection evidence for the uncovered cells; demand per-cell execution only when the path diverges. For a contract specifying a typed public signature whose shape admits divergent readings — callbacks, generics, overloads, container/element relationships — evidence counts only when a ledger entry type-checks standalone usage authored from the contract's text; the implementation's own tests compiling proves self-consistency, not the contract. A simple signature quoted verbatim from the patch that textually matches the contract is ordinary inspection evidence and needs no type-check run. (2) Inspection evidence — exact quoted lines from the terminal patches showing the contract satisfied; a file name alone is not evidence; use only for contracts fully verifiable by reading. (3) Candidate claims are never evidence.

Unblocking contracts get the strictest treatment. For any contract that an event X (close, shutdown, abort, cancellation, deadline) must unblock, interrupt, or fail a pending operation P, execution evidence counts only if the verifying test performs no action after X that could itself wake P — releasing, closing, erroring, or enqueuing on the awaited stream or channel, sending or receiving data, advancing timers, or completing the awaited resource. A test that wakes P by such means verifies only that P notices a flag after being woken; the row is unverified. Inspection evidence must quote the affirmative wake path X triggers on the already-blocked waiter and show X can reach it while P is blocked: a flag tested inside a read or write loop wakes nothing, and a mutex held across a blocking wait prevents any unblocking path needing that mutex from running. For exact-output contracts, reconstruct the emitted stream from the writer code for the first and last element, not only the middle.

Contracts that carry an adverse_condition are verified only under that condition. Their rows must also fill adverse_condition_evidence: quote the test or code lines showing the stated condition is actually constructed — the operation genuinely pending or blocked when the event fires, the resource genuinely exhausted, the dependency genuinely stalled, the boundary value genuinely used — and list every action the verifying test performs after the triggering event. Conclusive inspection satisfies this too: when the code that handles the stated condition is locally readable — a deep copy at the capture site, a normalization that covers the input family, a guard whose semantics include the edge — quote that handling and state why it covers the condition; demand a constructed test only when the handling is distributed across components, depends on runtime interleaving, or the quoted code's behavior under the condition is not decidable by reading. This cuts both ways: quoted handling that mishandles the condition makes the row violated, exactly as a failing test would. If the cited test never constructs the stated condition, or performs any post-event action that could itself wake or complete the pending operation, the contract is unverified regardless of the test's name or its green result.

Delivery-mechanics contracts (kind "delivery": branch, commit, repository cleanliness) rank below functional contracts: mark such a row unverified rather than violated when evidence is merely absent, record the residual risk in state_summary, and do not block completion on absence alone — but a delivery contract affirmatively contradicted by evidence is violated and blocks completion like any other — except when the contradiction is an environmental failure of the delivery action itself (missing git identity, authentication, network): that is unverified with the residual risk noted, not violated, because no amount of task work can resolve it.

complete=true requires every functional (inspection or execution) row verified. Any violated or unverified functional row means complete=false: keep winner={selected_lane}, choose next_candidate_count from 1 to {max_candidate_count}, choose next_window_steps from 1 to 10, and provide exactly next_candidate_count distinct advices telling the next candidate windows precisely what evidence to produce. Use more candidates when materially different fixes or verification strategies are useful under uncertainty; when one concrete defect is a serial prerequisite, choose one candidate to fix and verify it before exploring new directions. Structure each advice as an ordered work list over the violated and unverified contracts: for each, name the contract id, state what is broken or unproven, give the fix obligation first and the proving command second, and require the candidate to run that command and report its output verbatim after the fix — evidence produced before the fix proves nothing. Additionally, for an unverified execution contract, spell out the concrete scenario, the exact assertion, and instruct the candidate to report the command and output verbatim. When the contract is an unblocking contract, the advised scenario must keep the awaited resource permanently silent after the triggering event: assert that the pending operation rejects or returns within a timeout while nothing else wakes it, and perform any release or cleanup only after that assertion. For an unverified type-shape contract, instruct the candidate to write a standalone usage file authored from the contract text verbatim, run the project's type-checker against it, and report the command and its output. Do not rationalize an unresolved row as rare, cosmetic, pre-existing, timing-dependent, or out of scope; the checklist defines scope. When the ledger and patches genuinely cover every contract, return complete=true and do not invent optional work. Call select_trajectory exactly once and by itself. Do not answer in prose.
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

/// Produces an exact, mechanically structured handoff when the window is
/// already small enough that paying an LLM to summarize it would cost more
/// than carrying it forward. Exact duplicate tool results are replaced by an
/// explicit back-reference; no unique observation or reasoning is discarded.
pub(crate) fn asgard_deterministic_candidate_handoff(
    window_messages: &[ChatMessage],
    current_plan: Option<&crate::plan::UpdatePlanArgs>,
) -> Option<(Vec<ChatMessage>, usize, usize)> {
    let raw = render_asgard_dossier_messages(window_messages);
    let mut packed_messages = window_messages.to_vec();
    let mut prior_tool_results = HashMap::<String, usize>::new();
    for (index, message) in packed_messages.iter_mut().enumerate() {
        if message.role != "tool" {
            continue;
        }
        let text = asgard_message_text(message);
        if let Some(first_index) = prior_tool_results.get(&text).copied() {
            message.content = vec![ChatContentPart::text(format!(
                "<exact_duplicate_of_message index=\"{first_index}\" />"
            ))];
        } else {
            prior_tool_results.insert(text, index);
        }
    }
    let packed = render_asgard_dossier_messages(&packed_messages);
    if packed.len() > ASGARD_DETERMINISTIC_HANDOFF_MAX_BYTES {
        return None;
    }
    let plan = current_plan
        .and_then(|plan| serde_json::to_string_pretty(plan).ok())
        .unwrap_or_else(|| "(no active plan)".to_string());
    let handoff = ChatMessage::assistant(format!(
        "<candidate_window_handoff format=\"deterministic_exact\" duplicate_encoding=\"back_reference\">\n\
         This is a mechanically rendered trajectory, not a candidate-authored summary. Every unique \
         message is preserved exactly; exact duplicate tool results point to their first message.\n\
         <current_plan>\n{plan}\n</current_plan>\n\
         <candidate_window>\n{packed}</candidate_window>\n\
         </candidate_window_handoff>"
    ));
    Some((vec![handoff], raw.len(), packed.len()))
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

pub(crate) fn asgard_candidate_window_summary_messages(
    context: &AsgardCandidateAssessmentContext<'_>,
    window_messages: &[ChatMessage],
) -> Vec<ChatMessage> {
    let plan = context
        .current_plan
        .and_then(|plan| serde_json::to_string_pretty(plan).ok())
        .unwrap_or_else(|| "(no active plan)".to_string());
    vec![
        ChatMessage::system(
            "You are the candidate model writing a compact, loss-aware handoff of your latest \
             Asgard work window to a trajectory supervisor. Report what the window actually \
             established, not what you hoped to accomplish. Preserve adverse evidence, failed \
             checks, incomplete work, and consequential uncertainty. Summarize source edits \
             semantically by file or symbol; do not reproduce patches or routine narration.",
        ),
        ChatMessage::user(format!(
            "ORIGINAL TASK (complete):\n{}",
            context.original_task
        )),
        ChatMessage::assistant(format!(
            "<canonical_state_summary>\n{}\n</canonical_state_summary>\n\
             <current_plan>\n{}\n</current_plan>",
            context.canonical_state_summary, plan,
        )),
        ChatMessage::user(format!(
            "<candidate_window>\n{}\n</candidate_window>",
            render_asgard_dossier_messages(window_messages),
        )),
    ]
}

pub(crate) fn asgard_candidate_window_summary_prompt() -> ChatMessage {
    ChatMessage::user(
        r#"<candidate_window_handoff>
This work window has ended. Produce one faithful supervisor brief. Distinguish observed evidence
from your own claims. A successful command is evidence only for behavior it actually exercised.
If you edited tests, say so; green candidate-written tests are not independent confirmation.
Preserve exact failing check names, key error text, counts, paths, and concrete observations when
they matter. Explicitly record any original-task requirement that remains unimplemented or
unverified. Keep the brief much shorter than the raw window and do not continue the task.

Call summarize_candidate_window exactly once. Do not call another tool or answer in prose.
</candidate_window_handoff>"#,
    )
}

pub(crate) fn asgard_summarize_candidate_window_tool() -> ToolDefinition {
    ToolDefinition {
        r#type: "function".to_string(),
        function: FunctionDef {
            name: ASGARD_SUMMARIZE_WINDOW_TOOL_NAME.to_string(),
            description: "Produce one compact, evidence-focused handoff of the completed candidate work window.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["direction", "progress", "edits", "evidence", "unresolved_risks", "next_step"],
                "properties": {
                    "direction": { "type": "string", "minLength": 1, "maxLength": 1200 },
                    "progress": { "type": "string", "minLength": 1, "maxLength": 1200 },
                    "edits": {
                        "type": "array",
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "required": ["location", "change"],
                            "properties": {
                                "location": { "type": "string", "minLength": 1, "maxLength": 300 },
                                "change": { "type": "string", "minLength": 1, "maxLength": 700 }
                            },
                        },
                    },
                    "evidence": {
                        "type": "array",
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "required": ["check", "status", "details"],
                            "properties": {
                                "check": { "type": "string", "minLength": 1, "maxLength": 500 },
                                "status": { "type": "string", "enum": ["passed", "failed", "inconclusive"] },
                                "details": { "type": "string", "minLength": 1, "maxLength": 1200 }
                            }
                        }
                    },
                    "unresolved_risks": {
                        "type": "array",
                        "maxItems": 16,
                        "items": { "type": "string", "minLength": 1, "maxLength": 800 }
                    },
                    "next_step": { "type": "string", "minLength": 1, "maxLength": 1000 }
                },
            }),
        },
    }
}

pub(crate) fn parse_asgard_candidate_window_summary(
    arguments: &serde_json::Value,
) -> anyhow::Result<String> {
    for key in [
        "direction",
        "progress",
        "edits",
        "evidence",
        "unresolved_risks",
        "next_step",
    ] {
        anyhow::ensure!(
            arguments.get(key).is_some(),
            "candidate brief is missing `{key}`"
        );
    }
    serde_json::to_string_pretty(arguments)
        .map_err(|error| anyhow::anyhow!("failed to serialize candidate brief: {error}"))
}

pub(crate) async fn run_asgard_candidate_window_summary(
    llm: &dyn crate::llm_client::LlmBackend,
    context: AsgardCandidateAssessmentContext<'_>,
    cancel: tokio_util::sync::CancellationToken,
    window_messages: &[ChatMessage],
) -> (anyhow::Result<String>, crate::llm_client::TokenUsage) {
    const MAX_STEPS: usize = 2;
    let mut messages = asgard_candidate_window_summary_messages(&context, window_messages);
    messages.push(asgard_candidate_window_summary_prompt());
    let tools = vec![asgard_summarize_candidate_window_tool()];
    let mut usage = crate::llm_client::TokenUsage::default();
    let mut last_invalid_response = None;

    for step in 1..=MAX_STEPS {
        let request_bytes = render_asgard_dossier_messages(&messages).len();
        let response = stream_chat_no_visible_output_with_retry(
            llm,
            "summarizing Asgard candidate window",
            &cancel,
            || StreamChatRequest {
                model: context.model.to_string(),
                messages: messages.clone(),
                tools: Some(tools.clone()),
                reasoning_effort: context.reasoning_effort.map(str::to_string),
                service_tier: context.service_tier.map(str::to_string),
                temperature: None,
                structured_output: None,
                on_token: Box::new(|_| {}),
                on_thought: Box::new(|_| {}),
                cancel: cancel.clone(),
                idle_timeouts: context.idle_timeout,
            },
        )
        .await;
        let response = match response {
            Ok(response) => response,
            Err(error) => return (Err(error), usage),
        };
        let turn_usage = response.usage();
        trace_asgard_phase_turn_usage(
            "candidate_window_summary",
            context.model,
            Some(context.window),
            Some(context.lane),
            step,
            "summary",
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
                if calls.len() == 1 && calls[0].function.name == ASGARD_SUMMARIZE_WINDOW_TOOL_NAME {
                    match crate::tool_arguments::normalize_tool_arguments(
                        &calls[0].function.arguments,
                    ) {
                        Ok(arguments) => {
                            match parse_asgard_candidate_window_summary(&arguments.value) {
                                Ok(brief) => return (Ok(brief), usage),
                                Err(error) => last_invalid_response = Some(error),
                            }
                        }
                        Err(error) => {
                            last_invalid_response = Some(anyhow::anyhow!(
                                "candidate emitted invalid summarize_candidate_window arguments: {error}"
                            ));
                        }
                    }
                } else {
                    last_invalid_response = Some(anyhow::anyhow!(
                        "candidate must call summarize_candidate_window exactly once and no other tool"
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

        if step < MAX_STEPS {
            let detail = last_invalid_response
                .as_ref()
                .map(|error| format!(" Your previous response was invalid: {error}."))
                .unwrap_or_default();
            messages.push(ChatMessage::user(format!(
                "You have not summarized the candidate window.{detail} Only \
                 summarize_candidate_window is available. Call it exactly once now; do not answer in prose."
            )));
        }
    }

    (
        Err(last_invalid_response.unwrap_or_else(|| {
            anyhow::anyhow!(
                "candidate did not call summarize_candidate_window after {MAX_STEPS} steps"
            )
        })),
        usage,
    )
}

#[derive(Clone, Copy)]
pub(crate) struct AsgardCandidateAssessmentContext<'a> {
    pub(crate) model: &'a str,
    pub(crate) window: usize,
    pub(crate) lane: usize,
    pub(crate) reasoning_effort: Option<&'a str>,
    pub(crate) service_tier: Option<&'a str>,
    pub(crate) idle_timeout: IdleTimeouts,
    pub(crate) original_task: &'a str,
    pub(crate) canonical_state_summary: &'a str,
    pub(crate) current_plan: Option<&'a crate::plan::UpdatePlanArgs>,
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
            "required": ["next_candidate_count", "next_window_steps", "state_summary", "advices"],
            "properties": {
                "next_candidate_count": {
                    "type": "integer",
                    "minimum": ASGARD_MIN_CANDIDATES,
                    "maximum": max_candidate_count,
                },
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
                "next_candidate_count": {
                    "type": "integer",
                    "minimum": ASGARD_MIN_CANDIDATES,
                    "maximum": max_candidate_count,
                    "description": "Required when complete=false; omitted or ignored when complete=true.",
                },
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

pub(crate) fn asgard_extract_execution_ledger(
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
                let matching_result = window_messages
                    .get(step + 1..)
                    .unwrap_or_default()
                    .iter()
                    .find(|later| {
                        later.role == "tool" && later.tool_call_id.as_deref() == Some(&call.id)
                    });
                let (exit_code, output_bytes, output_sha256, output_tail) = match matching_result {
                    Some(result) => {
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
                    id: format!("L{total_shell_commands}"),
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
                next_candidate_count: None,
                next_window_steps: None,
                state_summary: state_summary.to_string(),
                contracts,
            });
        }
        let Some(next_candidate_count) =
            parse_asgard_next_candidate_count(&value, max_candidate_count)
        else {
            continue;
        };
        let Some(next_window_steps) = parse_asgard_next_window_steps(&value) else {
            continue;
        };
        let Some(advices) = parse_asgard_incomplete_advices(raw_advices, next_candidate_count)
        else {
            continue;
        };
        let contracts = parse_asgard_contract_rows(&value);
        return Ok(AsgardSupervisorDecision {
            winner,
            complete,
            advices,
            next_candidate_count: Some(next_candidate_count),
            next_window_steps: Some(next_window_steps),
            state_summary: state_summary.to_string(),
            contracts,
        });
    }
    anyhow::bail!(
        "Asgard supervisor returned neither a valid completed winner nor a winner plus a valid next_candidate_count, matching distinct advices, and a {ASGARD_MIN_WINDOW_STEPS}-{ASGARD_MAX_WINDOW_STEPS} next_window_steps"
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
        let Some(next_candidate_count) =
            parse_asgard_next_candidate_count(&value, max_candidate_count)
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
        let Some(advices) = parse_asgard_incomplete_advices(raw_advices, next_candidate_count)
        else {
            continue;
        };
        return Ok(AsgardSupervisorInitialAdvice {
            advices,
            next_candidate_count,
            next_window_steps,
            state_summary: state_summary.to_string(),
        });
    }
    anyhow::bail!(
        "Asgard supervisor returned no valid initial advice with a 1-{max_candidate_count} next_candidate_count, matching distinct advices, and a {ASGARD_MIN_WINDOW_STEPS}-{ASGARD_MAX_WINDOW_STEPS} next_window_steps"
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
                         execution_ledger entry (L<n>); execution contracts cannot be \
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

pub(crate) fn asgard_cites_ledger_entry(row: &AsgardContractRow) -> bool {
    let cited = |text: &str| {
        text.match_indices('L').any(|(index, _)| {
            text[index + 1..]
                .chars()
                .next()
                .is_some_and(|next| next.is_ascii_digit())
                && text[..index]
                    .chars()
                    .next_back()
                    .is_none_or(|previous| !previous.is_alphanumeric())
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

pub(crate) fn parse_asgard_next_candidate_count(
    value: &serde_json::Value,
    max_candidate_count: usize,
) -> Option<usize> {
    let count = value.get("next_candidate_count")?.as_u64()? as usize;
    (ASGARD_MIN_CANDIDATES..=max_candidate_count)
        .contains(&count)
        .then_some(count)
}

pub(crate) fn parse_asgard_incomplete_advices(
    raw_advices: &[serde_json::Value],
    count: usize,
) -> Option<Vec<Option<String>>> {
    if raw_advices.len() != count {
        return None;
    }
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
