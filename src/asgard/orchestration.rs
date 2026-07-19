use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use agent_client_protocol::schema::v1::{
    ContentBlock, ContentChunk, SessionNotification, SessionUpdate, TextContent,
};
use agent_client_protocol::{Client, ConnectionTo};
use anyhow::{Result, anyhow};
use futures::future::{BoxFuture, FutureExt};
use futures::stream::{FuturesUnordered, StreamExt};

use crate::asgard::{
    ASGARD_MAX_IN_FLIGHT, ASGARD_SUPERVISOR_MAX_STEPS, ASGARD_WORKER_MAX_STEPS, AsgardIntakeRun,
    CandidateRepository, CheckpointId, DISCARD_TOOL, DagLiveEntry, FINALIZE_TOOL,
    MERGE_CHECKPOINT_TOOL, PREFINALIZE_TOOL, SAVE_CHECKPOINT_TOOL, SPAWN_WORKERS_TOOL,
    SnapshotStage, SpawnRequest, SupervisorStreamCall, SupervisorTurnContext, TrajectoryDag,
    TrajectoryNode, TrajectoryWindow, UPDATE_PLAN_TOOL, VIEW_TOOL_CALL_TOOL, WAIT_TOOL,
    WorkerStopReason, elide_view_tool_results_for_permanent_record, parse_finalize,
    parse_merge_checkpoint, parse_spawn_workers, parse_update_plan, parse_view_tool_call,
    render_dag_overview, render_fragment, render_resolved_views, render_window_compact_for_worker,
    run_asgard_intake, stream_supervisor_response, summarize_resolved_views, supervisor_supplement,
    supervisor_tool_definitions,
};
use crate::llm_client::{
    ChatMessage, IdleTimeouts, LlmResponse, TokenUsage, ToolCall, ToolDefinition,
};
use crate::plan::UpdatePlanArgs;
use crate::session::SessionStore;
use crate::structured_output::StructuredOutputRequest;
use crate::tool_loop::{LoopOutcome, LoopStop, NotificationMode, SpawnedCx};

fn send_thought(cx: &ConnectionTo<Client>, session_id: &str, text: &str) {
    let chunk = ContentChunk::new(ContentBlock::Text(TextContent::new(text)));
    let update = SessionUpdate::AgentThoughtChunk(chunk);
    let notification = SessionNotification::new(session_id.to_string(), update);
    if let Err(e) = cx.send_notification(notification) {
        tracing::warn!("failed to send thought session update: {e}");
    }
}

fn send_plan_update(cx: &ConnectionTo<Client>, session_id: &str, plan: &UpdatePlanArgs) {
    let update = SessionUpdate::Plan(plan.to_acp());
    let notification = SessionNotification::new(session_id.to_string(), update);
    if let Err(e) = cx.send_notification(notification) {
        tracing::warn!("failed to send plan session update: {e}");
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

pub(crate) const ASGARD_MIN_CANDIDATES: usize = 1;
pub(crate) const ASGARD_MAX_CANDIDATES: usize = 5;

struct FinishedWorker {
    worker: usize,
    parent: CheckpointId,
    model: String,
    instructions: String,
    instruction_message: ChatMessage,
    window_messages: Vec<ChatMessage>,
    compact: String,
    final_response: String,
    stop: WorkerStopReason,
    steps: usize,
    diffstat: String,
    usage: TokenUsage,
    elapsed_millis: u64,
    repository: CandidateRepository,
}

impl FinishedWorker {
    fn window(&self) -> TrajectoryWindow {
        TrajectoryWindow {
            worker: self.worker,
            parent: self.parent.clone(),
            instructions: self.instructions.clone(),
            model: self.model.clone(),
            instruction_message: self.instruction_message.clone(),
            window_messages: self.window_messages.clone(),
            compact: self.compact.clone(),
            final_response: self.final_response.clone(),
            stop: self.stop.clone(),
            steps: self.steps,
            diffstat: self.diffstat.clone(),
            usage: self.usage,
            elapsed_millis: self.elapsed_millis,
        }
    }
}

#[derive(Clone)]
struct RunningWorkerMeta {
    parent: CheckpointId,
    model: String,
    instructions: String,
    turn_progress: Arc<AtomicUsize>,
}

#[derive(Clone, Copy)]
enum PendingResolution {
    Save,
    Discard,
}

enum AsgardExit {
    Finalize {
        checkpoint: CheckpointId,
        response: Option<String>,
        evidence: Vec<String>,
        abandoned: Vec<String>,
    },
    Failure(anyhow::Error),
    Cancelled,
}

struct SupervisorLoopContext<'ctx, 'fut> {
    llm: &'ctx Arc<dyn crate::llm_client::LlmBackend>,
    supervisor_model: &'ctx str,
    supervisor_turn: usize,
    permanent: &'ctx [ChatMessage],
    ephemeral: Vec<ChatMessage>,
    dag: &'ctx mut TrajectoryDag,
    pending: &'ctx mut Option<FinishedWorker>,
    pending_window: Option<TrajectoryWindow>,
    stage: &'ctx SnapshotStage,
    running: &'ctx mut FuturesUnordered<BoxFuture<'fut, FinishedWorker>>,
    running_meta: &'ctx mut HashMap<usize, RunningWorkerMeta>,
    review_queue: &'ctx mut VecDeque<FinishedWorker>,
    idle_pool: &'ctx mut Vec<CandidateRepository>,
    usage_by_model: &'ctx mut BTreeMap<String, TokenUsage>,
    aggregate_usage: &'ctx mut TokenUsage,
    worker_counter: &'ctx mut usize,
    clone_counter: &'ctx mut usize,
    launch: WorkerLaunch<'fut>,
    allowed_models: &'ctx [String],
    finalize_evidence_bounced: &'ctx mut bool,
    finalize_lineage_bounced: &'ctx mut bool,
    prefinalize_issued: &'ctx mut bool,
    prefinalize_workers: &'ctx mut Vec<usize>,
    latest_prefinalize_source_commits: &'ctx mut HashSet<String>,
    turn_capacity: usize,
    idle_timeout: IdleTimeouts,
    cancel: tokio_util::sync::CancellationToken,
    sinks: AsgardStreamSinks,
}

struct SupervisorLoopResult {
    finalizing: Option<(CheckpointId, Option<String>)>,
    finalizing_evidence: Vec<String>,
    finalizing_abandoned: Vec<String>,
    plan: Option<UpdatePlanArgs>,
    usage: TokenUsage,
    steps: usize,
    spawned: Vec<usize>,
    prefinalized: Vec<usize>,
    saved: bool,
    discarded: bool,
    merged: Vec<usize>,
    permanent_append: Vec<ChatMessage>,
}

struct SupervisorTurnState {
    latest_plan: Option<UpdatePlanArgs>,
    spawned: Vec<usize>,
    prefinalized: Vec<usize>,
    spawned_this_turn: usize,
    saved_pending: bool,
    discarded: bool,
    merged: Vec<usize>,
    finalizing: Option<(CheckpointId, Option<String>)>,
    finalizing_evidence: Vec<String>,
    finalizing_abandoned: Vec<String>,
    turn_ended: bool,
    finalize_bounced_this_step: bool,
    /// view_tool_call id -> per-handle summary kept in the permanent record.
    view_summaries: std::collections::HashMap<String, String>,
}

struct FinalizeContext<'a> {
    parent_cwd: &'a Path,
    stage: &'a SnapshotStage,
    dag: &'a TrajectoryDag,
    base_commit: &'a str,
    live_output: &'a AsgardLiveOutput,
    current_plan: Option<&'a UpdatePlanArgs>,
    evidence: &'a [String],
    abandoned: &'a [String],
    prefinalize_workers: &'a [usize],
}

const PREFINALIZE_VERIFICATION_ONLY_ERROR: &str =
    "prefinalize trajectories are verification-only; review the report, then discard";

struct WorkerLaunch<'a> {
    cx: &'a ConnectionTo<Client>,
    sessions: &'a SessionStore,
    session_id: &'a str,
    llm: &'a Arc<dyn crate::llm_client::LlmBackend>,
    parent_cwd: &'a Path,
    config: &'a crate::asgard::Config,
    reasoning_effort: Option<&'a str>,
    service_tier: Option<&'a str>,
    structured_output: Option<&'a StructuredOutputRequest>,
    idle_timeout: IdleTimeouts,
    worker_cancel: tokio_util::sync::CancellationToken,
    original_task: &'a str,
    live_output: &'a AsgardLiveOutput,
    context_length: Option<u32>,
    context_prefix_len: usize,
}

async fn launch_worker<'a>(
    launch: WorkerLaunch<'a>,
    spawn: &SpawnRequest,
    worker_id: usize,
    clone_label: String,
    parent_commit: String,
    mut messages: Vec<ChatMessage>,
    idle_pool: &mut Vec<CandidateRepository>,
) -> Result<(BoxFuture<'a, FinishedWorker>, RunningWorkerMeta)> {
    let repository = if let Some(repository) = idle_pool.pop() {
        if let Err(error) = crate::asgard::recycle_repository(&repository, &parent_commit) {
            crate::asgard::remove_candidate_repository(&repository);
            return Err(error);
        }
        repository
    } else {
        crate::asgard::create_candidate_repository_at(
            launch.parent_cwd,
            &clone_label,
            &parent_commit,
        )?
    };

    let Some(registry) = launch
        .sessions
        .create_trajectory_registry(launch.session_id, repository.session_cwd.clone())
        .await
    else {
        crate::asgard::remove_candidate_repository(&repository);
        return Err(anyhow!("unknown Asgard parent session"));
    };
    let tool_allowlist = Arc::new(worker_tool_allowlist(&registry).await);

    let instructions = spawn.instructions.clone();
    let instruction_message = ChatMessage::user(format!(
        "<supervisor_instructions>\n{instructions}\n</supervisor_instructions>\n\
         You are a worker agent directed by a supervisor. You have up to \
         {ASGARD_WORKER_MAX_STEPS} steps (each step = one batch of tool calls) before \
         you are paused for review; the supervisor may resume you or branch another \
         worker from your state. When you stop making tool calls, your turn ends and \
         your final message is delivered to the supervisor - make it a precise report \
         of what you did, what you verified, and what remains. Leave version control \
         to the harness: do not commit, reset, or switch branches; if the task itself \
         requires commits or branch delivery, state that in your final message instead \
         of doing it. Commits named 'asgard checkpoint wN' in the history are harness \
         bookkeeping - never amend, reword, rebase, or build on their messages."
    ));
    messages.push(instruction_message.clone());
    crate::asgard::rewrite_asgard_cwd(
        messages.as_mut_slice(),
        launch.parent_cwd,
        &repository.session_cwd,
    );
    let window_start = messages.len();
    let model = spawn
        .model
        .clone()
        .unwrap_or_else(|| launch.config.candidate_models[0].clone());
    let turn_progress = Arc::new(AtomicUsize::new(0));
    let meta = RunningWorkerMeta {
        parent: spawn.from.clone(),
        model: model.clone(),
        instructions: instructions.clone(),
        turn_progress: turn_progress.clone(),
    };
    let session_id = launch.session_id.to_string();
    let sessions = launch.sessions.clone();
    let original_task = launch.original_task.to_string();
    let live_output = launch.live_output.clone();
    let parent_cwd = launch.parent_cwd.to_path_buf();
    let reasoning_effort = launch.reasoning_effort;
    let service_tier = launch.service_tier;
    let structured_output = launch.structured_output;
    let idle_timeout = launch.idle_timeout;
    let worker_cancel = launch.worker_cancel.clone();
    let context_length = launch.context_length;
    let context_prefix_len = launch.context_prefix_len;
    let cx = launch.cx;
    let llm = launch.llm;
    let parent = spawn.from.clone();

    let future = async move {
        send_thought(
            &live_output.cx,
            &live_output.session_id,
            &format!("Asgard spawned worker w{worker_id} from {parent}.\n"),
        );
        let sinks = AsgardStreamSinks::new(&live_output, &format!("Worker w{worker_id}"));
        let spawned = SpawnedCx::new(cx);
        let started = Instant::now();
        let outcome = crate::tool_loop::run(
            llm,
            &registry,
            &model,
            reasoning_effort,
            service_tier,
            structured_output,
            messages,
            ASGARD_WORKER_MAX_STEPS,
            idle_timeout,
            worker_cancel,
            sinks.text,
            sinks.thought,
            spawned,
            session_id,
            sessions,
            original_task,
            NotificationMode::Silent,
            0,
            Some(tool_allowlist),
            None,
            true,
            Some(turn_progress),
            context_length,
            context_prefix_len,
            None,
        )
        .await;
        let elapsed_millis = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        let mut window_messages =
            asgard_take_window_messages(&outcome.continuation_messages, window_start);
        crate::asgard::rewrite_asgard_cwd(
            window_messages.as_mut_slice(),
            &repository.session_cwd,
            &parent_cwd,
        );
        let stop = worker_stop_reason(&outcome.stop);
        let steps = count_worker_steps(&window_messages);
        let final_response = extract_worker_final_response(&window_messages);
        let head_moved = worker_head_moved(&repository.root, &parent_commit, worker_id);
        let diffstat = match crate::asgard::capture_diffstat(&repository.root, &parent_commit) {
            Ok(diffstat) => diffstat,
            Err(error) => {
                tracing::warn!(
                    worker = worker_id,
                    "failed to capture Asgard worker diffstat: {error:#}"
                );
                String::new()
            }
        };
        let compact = render_window_compact_for_worker(worker_id, &window_messages);
        let raw_bytes = serialized_messages_len(&window_messages);
        let compact_bytes = compact.len();
        crate::trace_logging::append_trace_record(serde_json::json!({
            "type": "asgard_worker_window",
            "worker": worker_id,
            "parent": parent.to_string(),
            "model": model,
            "steps": steps,
            "stop": stop.label(),
            "elapsed_millis": elapsed_millis,
            "usage": usage_json(outcome.usage),
            "raw_bytes": raw_bytes,
            "compact_bytes": compact_bytes,
            "head_moved": head_moved,
        }));
        send_thought(
            &live_output.cx,
            &live_output.session_id,
            &format!("Asgard worker w{worker_id} finished: {}.\n", stop.label()),
        );
        FinishedWorker {
            worker: worker_id,
            parent,
            model,
            instructions,
            instruction_message,
            window_messages,
            compact,
            final_response,
            stop,
            steps,
            diffstat,
            usage: outcome.usage,
            elapsed_millis,
            repository,
        }
    }
    .boxed();

    Ok((future, meta))
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
) -> (LoopOutcome, BTreeMap<String, TokenUsage>) {
    let mut usage_by_model = BTreeMap::new();
    if let Err(error) = crate::asgard::ensure_compatible_checkout(parent_registry.cwd()) {
        return (asgard_failure(error), usage_by_model);
    }
    if !(ASGARD_MIN_CANDIDATES..=ASGARD_MAX_CANDIDATES).contains(&config.candidate_models.len()) {
        return (
            asgard_failure(anyhow!(
                "Asgard requires between {ASGARD_MIN_CANDIDATES} and \
                 {ASGARD_MAX_CANDIDATES} configured candidate models (got {})",
                config.candidate_models.len()
            )),
            usage_by_model,
        );
    }

    let base_commit = match crate::asgard::parent_head_commit(parent_registry.cwd()) {
        Ok(commit) => commit,
        Err(error) => return (asgard_failure(error), usage_by_model),
    };
    let run_id = uuid::Uuid::new_v4().to_string();
    let stage = match SnapshotStage::new(parent_registry.cwd(), &run_id) {
        Ok(stage) => stage,
        Err(error) => return (asgard_failure(error), usage_by_model),
    };
    let live_output = AsgardLiveOutput::new(cx, session_id);
    let supervisor_model = config.supervisor_model.as_deref().unwrap_or(selected_model);
    let original_task = asgard_original_task(&initial_messages);
    let (intake_contracts, intake_usages) = run_asgard_intake(AsgardIntakeRun {
        cx,
        sessions,
        session_id,
        llm,
        parent_cwd: parent_registry.cwd(),
        config,
        selected_model,
        reasoning_effort,
        service_tier,
        structured_output,
        idle_timeout,
        cancel: cancel.clone(),
        live_output: &live_output,
        original_task: &original_task,
        context_length,
        context_prefix_len,
    })
    .await;
    let mut aggregate_usage = TokenUsage::default();
    for (model, usage) in intake_usages {
        add_usage(&mut aggregate_usage, &mut usage_by_model, &model, usage);
    }
    let has_intake = intake_contracts.literal.is_some() || intake_contracts.grounded.is_some();
    let supervisor_system = supervisor_system_message(
        initial_messages
            .first()
            .filter(|message| message.role == "system")
            .map(ChatMessage::content_text)
            .as_deref(),
    );
    let mut dag = TrajectoryDag::new(initial_messages.clone(), base_commit.clone());
    let initial_permanent_user = initial_permanent_user_message(&original_task, &intake_contracts);
    let mut permanent = vec![
        ChatMessage::system(supervisor_system),
        ChatMessage::user(initial_permanent_user),
    ];
    let worker_cancel = cancel.child_token();
    let mut running: FuturesUnordered<BoxFuture<'_, FinishedWorker>> = FuturesUnordered::new();
    let mut review_queue = VecDeque::new();
    let mut idle_pool = Vec::new();
    let mut running_meta = HashMap::new();
    let mut supervisor_turn = 0usize;
    let mut worker_counter = 1usize;
    let mut clone_counter = 1usize;
    let mut fallback_idle_cycles = 0usize;
    let mut canonical_plan = initial_plan;
    let mut finalize_evidence_bounced = false;
    let mut finalize_lineage_bounced = false;
    let mut prefinalize_issued = false;
    let mut prefinalize_workers = Vec::new();
    let mut latest_prefinalize_source_commits = HashSet::new();

    let exit = loop {
        if cancel.is_cancelled() {
            break AsgardExit::Cancelled;
        }

        let mut pending = if !review_queue.is_empty() {
            review_queue.pop_front()
        } else {
            None
        };
        if pending.is_none() && !running.is_empty() {
            if let Some(finished) = running.next().await {
                running_meta.remove(&finished.worker);
                add_usage(
                    &mut aggregate_usage,
                    &mut usage_by_model,
                    &finished.model,
                    finished.usage,
                );
                review_queue.push_back(finished);
            }
            continue;
        }

        supervisor_turn += 1;
        let pending_window = pending.as_ref().map(FinishedWorker::window);
        if let Some(window) = &pending_window {
            permanent.push(ChatMessage::user(render_fragment(window)));
        }
        let mut idle_note = match (
            pending_window.is_some(),
            running.is_empty(),
            dag.checkpoint_labels().is_empty(),
        ) {
            (false, true, true) => Some(if has_intake {
                "No workers exist yet. First resolve the spec intake into a numbered obligations ledger via update_plan, then spawn 1 to 5 workers from \"root\"."
            } else {
                "No workers exist yet. Spawn 1 to 5 workers from \"root\" to begin. Consider dedicating the first worker to pinning the specification: tests written from the task text alone that lock in every detail that admits more than one reading."
            }),
            (false, true, false) => {
                Some("No worker is awaiting review. Spawn workers or finalize.")
            }
            _ => None,
        };
        if pending_window.is_none() && fallback_idle_cycles > 0 {
            idle_note = Some(
                "Previous supervisor turn ended without spawning or finalizing. Spawn workers or finalize now; after 3 idle turns Asgard will finalize the latest checkpoint or fail.",
            );
        }
        let status = render_asgard_status_block(
            &dag,
            &running_meta,
            &review_queue,
            pending.as_ref(),
            ASGARD_MAX_IN_FLIGHT.saturating_sub(running.len()),
            idle_note,
        );
        let ephemeral = vec![ChatMessage::user(status)];
        let sinks = AsgardStreamSinks::new(&live_output, "Supervisor");
        let turn_started = Instant::now();
        let turn_capacity = ASGARD_MAX_IN_FLIGHT.saturating_sub(running.len());
        let turn_result = run_supervisor_agentic_turn(SupervisorLoopContext {
            llm,
            supervisor_model,
            supervisor_turn,
            permanent: &permanent,
            ephemeral,
            dag: &mut dag,
            pending: &mut pending,
            pending_window: pending_window.clone(),
            stage: &stage,
            running: &mut running,
            running_meta: &mut running_meta,
            review_queue: &mut review_queue,
            idle_pool: &mut idle_pool,
            usage_by_model: &mut usage_by_model,
            aggregate_usage: &mut aggregate_usage,
            worker_counter: &mut worker_counter,
            clone_counter: &mut clone_counter,
            launch: WorkerLaunch {
                cx,
                sessions,
                session_id,
                llm,
                parent_cwd: parent_registry.cwd(),
                config,
                reasoning_effort,
                service_tier,
                structured_output,
                idle_timeout,
                worker_cancel: worker_cancel.clone(),
                original_task: &original_task,
                live_output: &live_output,
                context_length,
                context_prefix_len,
            },
            allowed_models: &config.candidate_models,
            finalize_evidence_bounced: &mut finalize_evidence_bounced,
            finalize_lineage_bounced: &mut finalize_lineage_bounced,
            prefinalize_issued: &mut prefinalize_issued,
            prefinalize_workers: &mut prefinalize_workers,
            latest_prefinalize_source_commits: &mut latest_prefinalize_source_commits,
            turn_capacity,
            idle_timeout,
            cancel: cancel.clone(),
            sinks,
        })
        .await;
        let elapsed_millis = turn_started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        let permanent_bytes = match &turn_result {
            Ok(outcome) => {
                let mut traced_permanent = permanent.clone();
                traced_permanent.extend(outcome.permanent_append.clone());
                serialized_messages_len(&traced_permanent)
            }
            Err(_) => serialized_messages_len(&permanent),
        };
        trace_supervisor_turn(
            supervisor_turn,
            pending_window.as_ref().map(|window| window.worker),
            &turn_result,
            elapsed_millis,
            permanent_bytes,
        );
        trace_usage_by_model(supervisor_turn, &usage_by_model);

        match turn_result {
            Ok(outcome) => {
                if let Some(plan) = outcome.plan {
                    send_plan_update(&live_output.cx, &live_output.session_id, &plan);
                    canonical_plan = Some(plan);
                }
                permanent.extend(outcome.permanent_append);
                if let Some((checkpoint, response)) = outcome.finalizing {
                    break AsgardExit::Finalize {
                        checkpoint,
                        response,
                        evidence: outcome.finalizing_evidence,
                        abandoned: outcome.finalizing_abandoned,
                    };
                }
                if outcome.spawned.is_empty()
                    && outcome.merged.is_empty()
                    && !outcome.saved
                    && pending_window.is_none()
                {
                    fallback_idle_cycles += 1;
                    if fallback_idle_cycles >= 3 {
                        if let Some(checkpoint) = latest_saved_checkpoint(&dag) {
                            break AsgardExit::Finalize {
                                checkpoint,
                                response: None,
                                evidence: Vec::new(),
                                abandoned: Vec::new(),
                            };
                        }
                        if running.is_empty() {
                            break AsgardExit::Failure(anyhow!(
                                "Asgard supervisor ended {fallback_idle_cycles} idle turns without spawning or finalizing"
                            ));
                        }
                    }
                } else {
                    fallback_idle_cycles = 0;
                }
                if let Some(finished) = pending.take() {
                    crate::asgard::remove_candidate_repository(&finished.repository);
                }
            }
            Err(error) => {
                let mode = if pending.is_some() {
                    "auto_save"
                } else {
                    "finalize_latest"
                };
                crate::trace_logging::append_trace_record(serde_json::json!({
                    "type": "asgard_supervisor_fallback",
                    "mode": mode,
                    "error": format!("{error:#}"),
                }));
                if let Some(finished) = pending.take() {
                    let window = pending_window.expect("pending window exists");
                    let resolution = if prefinalize_workers.contains(&finished.worker) {
                        PendingResolution::Discard
                    } else {
                        PendingResolution::Save
                    };
                    match resolve_pending(&stage, &mut dag, &finished, window, resolution) {
                        Ok(()) => {
                            idle_pool.push(finished.repository);
                            continue;
                        }
                        Err(snapshot_error) => {
                            crate::asgard::remove_candidate_repository(&finished.repository);
                            break AsgardExit::Failure(snapshot_error);
                        }
                    }
                }
                fallback_idle_cycles += 1;
                if let Some(checkpoint) = latest_saved_checkpoint(&dag) {
                    break AsgardExit::Finalize {
                        checkpoint,
                        response: None,
                        evidence: Vec::new(),
                        abandoned: Vec::new(),
                    };
                }
                if fallback_idle_cycles >= 3 {
                    break AsgardExit::Failure(error);
                }
                if running.is_empty() {
                    break AsgardExit::Failure(error);
                }
            }
        }
    };

    let mut outcome = match exit {
        AsgardExit::Finalize {
            checkpoint,
            response,
            evidence,
            abandoned,
        } => finalize_asgard(
            FinalizeContext {
                parent_cwd: parent_registry.cwd(),
                stage: &stage,
                dag: &dag,
                base_commit: &base_commit,
                live_output: &live_output,
                current_plan: canonical_plan.as_ref(),
                evidence: &evidence,
                abandoned: &abandoned,
                prefinalize_workers: &prefinalize_workers,
            },
            checkpoint,
            response,
            aggregate_usage,
        ),
        AsgardExit::Failure(error) => {
            worker_cancel.cancel();
            asgard_failure(error)
        }
        AsgardExit::Cancelled => {
            worker_cancel.cancel();
            LoopOutcome {
                response: String::new(),
                tool_exchanges: Vec::new(),
                replay_events: Vec::new(),
                usage: aggregate_usage,
                stop: LoopStop::Cancelled,
                continuation_messages: Vec::new(),
                current_plan: canonical_plan.clone(),
                compaction_checkpoint: None,
            }
        }
    };

    worker_cancel.cancel();
    while let Some(finished) = running.next().await {
        running_meta.remove(&finished.worker);
        add_usage(
            &mut aggregate_usage,
            &mut usage_by_model,
            &finished.model,
            finished.usage,
        );
        crate::asgard::remove_candidate_repository(&finished.repository);
    }
    for finished in review_queue {
        crate::asgard::remove_candidate_repository(&finished.repository);
    }
    cleanup_asgard_repositories(&idle_pool);
    let checkpoint_map = checkpoint_commit_map(&dag);
    crate::trace_logging::append_trace_record(serde_json::json!({
        "type": "asgard_checkpoints",
        "checkpoints": checkpoint_map,
    }));
    stage.cleanup();
    outcome.usage = aggregate_usage;

    (outcome, usage_by_model)
}

async fn run_supervisor_agentic_turn<'ctx, 'fut>(
    mut cx: SupervisorLoopContext<'ctx, 'fut>,
) -> Result<SupervisorLoopResult> {
    let tools = supervisor_tool_definitions(cx.allowed_models);
    let mut usage = TokenUsage::default();
    let mut tail = std::mem::take(&mut cx.ephemeral);
    let transcript_start = tail.len();
    let mut state = SupervisorTurnState {
        latest_plan: None,
        spawned: Vec::new(),
        prefinalized: Vec::new(),
        spawned_this_turn: 0,
        saved_pending: false,
        discarded: false,
        finalizing: None,
        finalizing_evidence: Vec::new(),
        finalizing_abandoned: Vec::new(),
        merged: Vec::new(),
        turn_ended: false,
        finalize_bounced_this_step: false,
        view_summaries: std::collections::HashMap::new(),
    };
    let mut steps = 0usize;
    let mut unresolved_reminder_sent = false;
    let mut step_warning_sent = false;
    let mut ephemeral_tail_indexes = Vec::new();

    while steps < ASGARD_SUPERVISOR_MAX_STEPS && state.finalizing.is_none() {
        if !step_warning_sent && ASGARD_SUPERVISOR_MAX_STEPS.saturating_sub(steps) == 2 {
            ephemeral_tail_indexes.push(tail.len());
            tail.push(ChatMessage::user(
                "2 steps remain this turn — resolve the reviewed trajectory and wrap up.",
            ));
            step_warning_sent = true;
        }
        let response = {
            let text_sink = Some(cx.sinks.text.clone());
            let thought_sink = Some(cx.sinks.thought.clone());
            let mut request_messages = cx.permanent.to_vec();
            request_messages.extend(tail.clone());
            trace_supervisor_llm_request(
                cx.supervisor_turn,
                cx.supervisor_model,
                &request_messages,
                &tools,
            );
            let supervisor_future = stream_supervisor_response(SupervisorStreamCall {
                llm: cx.llm.as_ref(),
                model: cx.supervisor_model,
                request_prefix: cx.permanent,
                tail: &tail,
                tools: &tools,
                cancel: &cx.cancel,
                idle_timeout: cx.idle_timeout,
                text_sink,
                thought_sink,
            });
            tokio::pin!(supervisor_future);
            loop {
                tokio::select! {
                    completed = cx.running.next(), if !cx.running.is_empty() => {
                        if let Some(finished) = completed {
                            cx.running_meta.remove(&finished.worker);
                            add_usage(cx.aggregate_usage, cx.usage_by_model, &finished.model, finished.usage);
                            cx.review_queue.push_back(finished);
                        }
                    }
                    result = &mut supervisor_future => {
                        break result?;
                    }
                }
            }
        };
        trace_supervisor_llm_response(cx.supervisor_turn, &response);
        usage.add(response.usage());
        steps += 1;
        state.finalize_bounced_this_step = false;

        match response {
            LlmResponse::Text {
                text,
                reasoning_content,
                ..
            } => {
                tail.push(ChatMessage::assistant_with_reasoning(
                    text,
                    reasoning_content,
                ));
                if pending_unresolved(cx.pending, state.saved_pending)
                    && !unresolved_reminder_sent
                    && steps < ASGARD_SUPERVISOR_MAX_STEPS
                {
                    let worker = cx
                        .pending
                        .as_ref()
                        .map(|finished| finished.worker)
                        .ok_or_else(|| anyhow!("pending worker disappeared"))?;
                    ephemeral_tail_indexes.push(tail.len());
                    tail.push(ChatMessage::user(format!(
                        "w{worker} must be saved, spawned from, or discarded — it is currently none of these"
                    )));
                    unresolved_reminder_sent = true;
                    continue;
                }
                break;
            }
            LlmResponse::ToolCalls {
                text,
                reasoning_content,
                calls,
                ..
            } => {
                tail.push(
                    ChatMessage::assistant_tool_calls_with_content_and_reasoning(
                        text,
                        calls.clone(),
                        reasoning_content,
                    ),
                );
                for call in &calls {
                    if state.turn_ended {
                        tail.push(ChatMessage::tool_result(
                            &call.id,
                            &call.function.name,
                            "error: turn already ended",
                        ));
                        continue;
                    }
                    if state.finalizing.is_some() {
                        tail.push(ChatMessage::tool_result(
                            &call.id,
                            &call.function.name,
                            "error: run already finalizing",
                        ));
                        continue;
                    }

                    let result = execute_supervisor_call(&mut cx, call, &mut state).await?;
                    tail.push(ChatMessage::tool_result(
                        &call.id,
                        &call.function.name,
                        result,
                    ));
                }
                if state.finalize_bounced_this_step {
                    steps = steps.saturating_sub(1);
                }
                if state.turn_ended {
                    if pending_unresolved(cx.pending, state.saved_pending)
                        && !unresolved_reminder_sent
                        && steps < ASGARD_SUPERVISOR_MAX_STEPS
                    {
                        let worker = cx
                            .pending
                            .as_ref()
                            .map(|finished| finished.worker)
                            .ok_or_else(|| anyhow!("pending worker disappeared"))?;
                        ephemeral_tail_indexes.push(tail.len());
                        tail.push(ChatMessage::user(format!(
                            "w{worker} must be saved, spawned from, or discarded — it is currently none of these"
                        )));
                        unresolved_reminder_sent = true;
                        state.turn_ended = false;
                        continue;
                    }
                    break;
                }
            }
        }
    }

    if let Some(finished) = cx.pending.take() {
        if state.saved_pending {
            cx.idle_pool.push(finished.repository);
        } else if state
            .finalizing
            .as_ref()
            .is_some_and(|(checkpoint, _)| *checkpoint != CheckpointId::Worker(finished.worker))
        {
            let window = cx
                .pending_window
                .clone()
                .ok_or_else(|| anyhow!("pending worker missing trajectory window"))?;
            resolve_pending(
                cx.stage,
                cx.dag,
                &finished,
                window,
                PendingResolution::Discard,
            )?;
            state.discarded = true;
            tail.push(ChatMessage::user(format!(
                "w{} was discarded (run finalized elsewhere)",
                finished.worker
            )));
            cx.idle_pool.push(finished.repository);
        } else if cx.prefinalize_workers.contains(&finished.worker) {
            let window = cx
                .pending_window
                .clone()
                .ok_or_else(|| anyhow!("pending worker missing trajectory window"))?;
            resolve_pending(
                cx.stage,
                cx.dag,
                &finished,
                window,
                PendingResolution::Discard,
            )?;
            state.discarded = true;
            tail.push(ChatMessage::user(format!(
                "w{} was discarded ({PREFINALIZE_VERIFICATION_ONLY_ERROR})",
                finished.worker
            )));
            cx.idle_pool.push(finished.repository);
        } else {
            let window = cx
                .pending_window
                .clone()
                .ok_or_else(|| anyhow!("pending worker missing trajectory window"))?;
            resolve_pending(cx.stage, cx.dag, &finished, window, PendingResolution::Save)?;
            crate::trace_logging::append_trace_record(serde_json::json!({
                "type": "asgard_supervisor_fallback",
                "mode": "auto_save_unresolved",
            }));
            tail.push(ChatMessage::user(format!(
                "w{} was auto-saved: the turn ended without resolving it",
                finished.worker
            )));
            cx.idle_pool.push(finished.repository);
        }
    }

    let permanent_tail = tail[transcript_start..]
        .iter()
        .enumerate()
        .filter_map(|(index, message)| {
            let tail_index = transcript_start + index;
            (!ephemeral_tail_indexes.contains(&tail_index)).then(|| message.clone())
        })
        .collect::<Vec<_>>();
    let permanent_append =
        elide_view_tool_results_for_permanent_record(&permanent_tail, &state.view_summaries);
    add_usage(
        cx.aggregate_usage,
        cx.usage_by_model,
        cx.supervisor_model,
        usage,
    );

    Ok(SupervisorLoopResult {
        finalizing: state.finalizing,
        plan: state.latest_plan,
        usage,
        steps,
        spawned: state.spawned,
        prefinalized: state.prefinalized,
        saved: state.saved_pending,
        discarded: state.discarded,
        finalizing_evidence: state.finalizing_evidence,
        finalizing_abandoned: state.finalizing_abandoned,
        merged: state.merged,
        permanent_append,
    })
}

async fn execute_supervisor_call<'ctx, 'fut>(
    cx: &mut SupervisorLoopContext<'ctx, 'fut>,
    call: &ToolCall,
    state: &mut SupervisorTurnState,
) -> Result<String> {
    match call.function.name.as_str() {
        WAIT_TOOL => {
            state.turn_ended = true;
            Ok("waiting for the next worker".to_string())
        }
        VIEW_TOOL_CALL_TOOL => match parse_view_tool_call(call) {
            Ok(handles) if handles.is_empty() => Ok(
                "error: no handles given. There is nothing to poll: to wait for running \
                 workers, call wait (or simply reply without tool calls) - you are re-engaged \
                 automatically when the next worker finishes."
                    .to_string(),
            ),
            Ok(handles) => {
                let in_flight: Vec<usize> = cx
                    .running_meta
                    .keys()
                    .copied()
                    .chain(cx.review_queue.iter().map(|finished| finished.worker))
                    .collect();
                let views = cx.dag.resolve_handle_views(
                    &handles,
                    cx.pending
                        .as_ref()
                        .map(|finished| (finished.worker, finished.window_messages.as_slice())),
                    &in_flight,
                );
                state
                    .view_summaries
                    .insert(call.id.clone(), summarize_resolved_views(&views));
                Ok(render_resolved_views(&views))
            }
            Err(error) => Ok(format!("error: {error}")),
        },
        UPDATE_PLAN_TOOL => match parse_update_plan(call) {
            Ok(plan) => {
                let steps = plan.plan.len();
                state.latest_plan = Some(plan);
                Ok(format!("plan updated ({steps} steps)"))
            }
            Err(error) => Ok(format!("error: {error}")),
        },
        SAVE_CHECKPOINT_TOOL => {
            if state.saved_pending {
                return Ok(
                    "error: save_checkpoint: pending trajectory is already saved".to_string(),
                );
            }
            let Some(finished) = cx.pending.as_ref() else {
                return Ok(
                    "error: save_checkpoint: requires a just-reviewed pending trajectory"
                        .to_string(),
                );
            };
            if cx.prefinalize_workers.contains(&finished.worker) {
                return Ok(format!("error: {PREFINALIZE_VERIFICATION_ONLY_ERROR}"));
            }
            save_pending_if_needed(cx, &mut state.saved_pending)?;
            Ok(format!(
                "saved checkpoint w{}",
                cx.pending
                    .as_ref()
                    .map(|finished| finished.worker)
                    .ok_or_else(|| anyhow!("pending worker disappeared during save"))?
            ))
        }
        MERGE_CHECKPOINT_TOOL => {
            let context = SupervisorTurnContext {
                dag: &*cx.dag,
                pending: cx.pending.as_ref().map(|finished| finished.worker),
                allowed_models: cx.allowed_models,
            };
            let parsed = match parse_merge_checkpoint(call, &context) {
                Ok(parsed) => parsed,
                Err(error) => return Ok(format!("error: merge_checkpoint: {error}")),
            };
            let new_worker = *cx.worker_counter;
            match merge_checkpoint(cx.stage, cx.dag, parsed.from, parsed.onto, new_worker) {
                Ok(result) => {
                    *cx.worker_counter += 1;
                    state.merged.push(new_worker);
                    Ok(result)
                }
                Err(error) => Ok(format!("error: merge_checkpoint: {error:#}")),
            }
        }
        DISCARD_TOOL => {
            if state.saved_pending {
                return Ok("error: discard: pending trajectory is already saved".to_string());
            }
            let Some(finished) = cx.pending.take() else {
                return Ok(
                    "error: discard: requires a just-reviewed pending trajectory".to_string(),
                );
            };
            let worker = finished.worker;
            let window = cx
                .pending_window
                .clone()
                .ok_or_else(|| anyhow!("pending worker missing trajectory window"))?;
            resolve_pending(
                cx.stage,
                cx.dag,
                &finished,
                window,
                PendingResolution::Discard,
            )?;
            state.discarded = true;
            cx.idle_pool.push(finished.repository);
            Ok(format!("trajectory w{worker} discarded"))
        }
        SPAWN_WORKERS_TOOL => {
            let context = SupervisorTurnContext {
                dag: &*cx.dag,
                pending: cx.pending.as_ref().map(|finished| finished.worker),
                allowed_models: cx.allowed_models,
            };
            let spawns = match parse_spawn_workers(call, &context) {
                Ok(spawns) => spawns,
                Err(error) => return Ok(format!("error: {error}")),
            };
            if spawns_from_pending_prefinalize(cx, &spawns) {
                return Ok(format!("error: {PREFINALIZE_VERIFICATION_ONLY_ERROR}"));
            }
            execute_spawn_requests(cx, state, spawns, SpawnKind::Regular).await
        }
        PREFINALIZE_TOOL => {
            let context = SupervisorTurnContext {
                dag: &*cx.dag,
                pending: cx.pending.as_ref().map(|finished| finished.worker),
                allowed_models: cx.allowed_models,
            };
            let spawns = match parse_spawn_workers(call, &context) {
                Ok(spawns) => spawns,
                Err(error) => return Ok(format!("error: {error}")),
            };
            execute_spawn_requests(cx, state, spawns, SpawnKind::Prefinalize).await
        }
        FINALIZE_TOOL => {
            let context = SupervisorTurnContext {
                dag: &*cx.dag,
                pending: cx.pending.as_ref().map(|finished| finished.worker),
                allowed_models: cx.allowed_models,
            };
            let parsed = match parse_finalize(call, &context) {
                Ok(parsed) => parsed,
                Err(error) => return Ok(format!("error: finalize: {error}")),
            };
            let checkpoint = parsed.checkpoint;
            let response = parsed.response;
            let evidence = parsed.evidence;
            let abandoned = parsed.abandoned;
            if !*cx.prefinalize_issued {
                state.turn_ended = true;
                return Ok("error: finalize requires a prefinalize verification pass first: spawn verification workers via prefinalize, review their reports, then finalize.".to_string());
            }
            let unresolved_prefinalize = unresolved_prefinalize_workers(cx, state);
            if !unresolved_prefinalize.is_empty() {
                state.turn_ended = true;
                return Ok(format!(
                    "error: prefinalize workers [{}] have not been reviewed yet; wait for them and review their reports before finalizing.",
                    unresolved_prefinalize
                        .iter()
                        .map(|worker| format!("w{worker}"))
                        .collect::<Vec<_>>()
                        .join(",")
                ));
            }
            let Some(checkpoint_commit) = cx.dag.commit_for(&checkpoint) else {
                return Ok(format!("error: finalize: unknown checkpoint {checkpoint}"));
            };
            if !cx
                .latest_prefinalize_source_commits
                .contains(checkpoint_commit)
            {
                state.turn_ended = true;
                return Ok(format!(
                    "error: the delivered checkpoint ({checkpoint}) is not the state your latest prefinalize verified; run prefinalize from {checkpoint} (or the checkpoint you intend to deliver), review it, then finalize."
                ));
            }
            let pending_messages = cx
                .pending
                .as_ref()
                .map(|finished| (finished.worker, finished.window_messages.as_slice()));
            let has_sufficient_evidence = evidence.iter().any(|handle| {
                cx.dag
                    .handle_is_run_shell_command_result(handle, pending_messages)
            });
            if !has_sufficient_evidence && !*cx.finalize_evidence_bounced {
                *cx.finalize_evidence_bounced = true;
                state.finalize_bounced_this_step = true;
                return Ok("error: before finalizing, name the evidence: the handles of the test runs you inspected (e.g. [\"w9m4\"]). If you have not seen test output for this checkpoint, spawn a verification worker on it first.".to_string());
            }
            let off_lineage = cx.dag.off_lineage_checkpoints_with_diffstat(&checkpoint);
            let unacknowledged = off_lineage
                .iter()
                .filter(|entry| {
                    !abandoned
                        .iter()
                        .any(|id| id == &entry.checkpoint.to_string())
                })
                .collect::<Vec<_>>();
            if !unacknowledged.is_empty() && !*cx.finalize_lineage_bounced {
                *cx.finalize_lineage_bounced = true;
                state.finalize_bounced_this_step = true;
                let mut message =
                    "error: finalize would leave diff-bearing saved checkpoints off-lineage:\n"
                        .to_string();
                for entry in unacknowledged {
                    message.push_str(&format!("{}:\n", entry.checkpoint));
                    message.push_str(entry.diffstat.trim_end());
                    message.push('\n');
                }
                message.push_str("Their work is absent from the delivered lineage. Merge them (merge_checkpoint), or list them in `abandoned` to confirm intentional abandonment.");
                return Ok(message);
            }
            if cx
                .pending
                .as_ref()
                .is_some_and(|finished| checkpoint == CheckpointId::Worker(finished.worker))
                && !state.saved_pending
            {
                save_pending_if_needed(cx, &mut state.saved_pending)?;
            }
            state.finalizing = Some((checkpoint.clone(), response));
            state.finalizing_evidence = evidence.clone();
            state.finalizing_abandoned = abandoned;
            let evidence_text = if evidence.is_empty() {
                "no evidence named".to_string()
            } else {
                format!("evidence: {}", evidence.join(", "))
            };
            Ok(format!("finalizing {checkpoint}; {evidence_text}"))
        }
        other => Ok(format!("error: unknown supervisor tool {other}")),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SpawnDedupKey {
    from: CheckpointId,
    instructions: String,
    model: String,
}

#[derive(Clone, Debug)]
enum SpawnDuplicateNote {
    InCall { kept_index: usize, count: usize },
    Running { worker: usize, count: usize },
}

#[derive(Clone, Copy)]
enum SpawnKind {
    Regular,
    Prefinalize,
}

async fn execute_spawn_requests<'ctx, 'fut>(
    cx: &mut SupervisorLoopContext<'ctx, 'fut>,
    state: &mut SupervisorTurnState,
    spawns: Vec<SpawnRequest>,
    kind: SpawnKind,
) -> Result<String> {
    let (spawns, duplicate_notes) = dedup_spawn_requests(cx, spawns);
    let remaining_capacity = cx.turn_capacity.saturating_sub(state.spawned_this_turn);
    if spawns.len() > remaining_capacity {
        return Ok(format!(
            "error: requested {} workers but only {remaining_capacity} capacity slots remain this turn",
            spawns.len()
        ));
    }
    let mut lines = Vec::new();
    if spawns.iter().any(|spawn| {
        cx.pending
            .as_ref()
            .is_some_and(|finished| spawn.from == CheckpointId::Worker(finished.worker))
    }) && !state.saved_pending
    {
        save_pending_if_needed(cx, &mut state.saved_pending)?;
        lines.push(format!(
            "saved checkpoint w{}",
            cx.pending
                .as_ref()
                .map(|finished| finished.worker)
                .ok_or_else(|| anyhow!("pending worker disappeared during spawn save"))?
        ));
    }
    if matches!(kind, SpawnKind::Prefinalize) {
        *cx.latest_prefinalize_source_commits = spawns
            .iter()
            .filter_map(|spawn| cx.dag.commit_for(&spawn.from).map(str::to_string))
            .collect();
    }
    let mut spawned_by_index = HashMap::new();
    for spawn in spawns {
        let from = spawn.from.clone();
        let spawn_index = spawned_by_index.len();
        let worker = launch_spawn(cx, spawn).await?;
        state.spawned_this_turn += 1;
        state.spawned.push(worker);
        if matches!(kind, SpawnKind::Prefinalize) {
            *cx.prefinalize_issued = true;
            cx.prefinalize_workers.push(worker);
            state.prefinalized.push(worker);
        }
        spawned_by_index.insert(spawn_index, worker);
        let verb = match kind {
            SpawnKind::Regular => "spawned",
            SpawnKind::Prefinalize => "prefinalize spawned",
        };
        lines.push(format!("{verb} w{worker} from {from}"));
    }
    lines.extend(render_spawn_duplicate_notes(
        &duplicate_notes,
        &spawned_by_index,
    ));
    Ok(lines.join("\n"))
}

fn unresolved_prefinalize_workers(
    cx: &SupervisorLoopContext<'_, '_>,
    state: &SupervisorTurnState,
) -> Vec<usize> {
    cx.prefinalize_workers
        .iter()
        .copied()
        .filter(|worker| {
            cx.running_meta.contains_key(worker)
                || cx
                    .review_queue
                    .iter()
                    .any(|finished| finished.worker == *worker)
                || cx
                    .pending
                    .as_ref()
                    .is_some_and(|finished| finished.worker == *worker && !state.saved_pending)
        })
        .collect()
}

fn spawns_from_pending_prefinalize(
    cx: &SupervisorLoopContext<'_, '_>,
    spawns: &[SpawnRequest],
) -> bool {
    cx.pending.as_ref().is_some_and(|finished| {
        cx.prefinalize_workers.contains(&finished.worker)
            && spawns
                .iter()
                .any(|spawn| spawn.from == CheckpointId::Worker(finished.worker))
    })
}

fn dedup_spawn_requests(
    cx: &SupervisorLoopContext<'_, '_>,
    spawns: Vec<SpawnRequest>,
) -> (Vec<SpawnRequest>, Vec<SpawnDuplicateNote>) {
    let default_model = &cx.launch.config.candidate_models[0];
    let mut running_by_key = HashMap::new();
    for (worker, meta) in cx.running_meta.iter() {
        running_by_key.insert(
            SpawnDedupKey {
                from: meta.parent.clone(),
                instructions: meta.instructions.clone(),
                model: meta.model.clone(),
            },
            *worker,
        );
    }

    let mut kept = Vec::new();
    let mut kept_by_key = HashMap::new();
    let mut in_call_duplicates: HashMap<usize, usize> = HashMap::new();
    let mut running_duplicates: HashMap<usize, usize> = HashMap::new();
    for spawn in spawns {
        let key = SpawnDedupKey {
            from: spawn.from.clone(),
            instructions: spawn.instructions.clone(),
            model: spawn
                .model
                .clone()
                .unwrap_or_else(|| default_model.to_string()),
        };
        if let Some(worker) = running_by_key.get(&key) {
            *running_duplicates.entry(*worker).or_default() += 1;
            continue;
        }
        if let Some(kept_index) = kept_by_key.get(&key) {
            *in_call_duplicates.entry(*kept_index).or_default() += 1;
            continue;
        }
        let kept_index = kept.len();
        kept_by_key.insert(key, kept_index);
        kept.push(spawn);
    }

    let mut notes = in_call_duplicates
        .into_iter()
        .map(|(kept_index, count)| SpawnDuplicateNote::InCall { kept_index, count })
        .chain(
            running_duplicates
                .into_iter()
                .map(|(worker, count)| SpawnDuplicateNote::Running { worker, count }),
        )
        .collect::<Vec<_>>();
    notes.sort_by_key(|note| match note {
        SpawnDuplicateNote::InCall { kept_index, .. } => (0usize, *kept_index),
        SpawnDuplicateNote::Running { worker, .. } => (1usize, *worker),
    });
    (kept, notes)
}

fn render_spawn_duplicate_notes(
    notes: &[SpawnDuplicateNote],
    spawned_by_index: &HashMap<usize, usize>,
) -> Vec<String> {
    notes
        .iter()
        .filter_map(|note| match note {
            SpawnDuplicateNote::InCall { kept_index, count } => {
                let worker = spawned_by_index.get(kept_index)?;
                Some(format!(
                    "skipped {count} duplicate specs (identical to w{worker})"
                ))
            }
            SpawnDuplicateNote::Running { worker, count } => {
                if *count == 1 {
                    Some(format!("skipped spec identical to running w{worker}"))
                } else {
                    Some(format!(
                        "skipped {count} specs identical to running w{worker}"
                    ))
                }
            }
        })
        .collect()
}

fn save_pending_if_needed(
    cx: &mut SupervisorLoopContext<'_, '_>,
    saved_pending: &mut bool,
) -> Result<()> {
    if *saved_pending {
        anyhow::bail!("save_checkpoint: pending trajectory is already saved");
    }
    let Some(finished) = cx.pending.as_ref() else {
        anyhow::bail!("save_checkpoint: requires a just-reviewed pending trajectory");
    };
    let window = cx
        .pending_window
        .clone()
        .ok_or_else(|| anyhow!("pending worker missing trajectory window"))?;
    resolve_pending(cx.stage, cx.dag, finished, window, PendingResolution::Save)?;
    *saved_pending = true;
    send_thought(
        &cx.launch.live_output.cx,
        &cx.launch.live_output.session_id,
        &format!("Asgard saved checkpoint w{}.\n", finished.worker),
    );
    Ok(())
}

fn pending_unresolved(pending: &Option<FinishedWorker>, saved_pending: bool) -> bool {
    pending.is_some() && !saved_pending
}

async fn launch_spawn<'a>(
    cx: &mut SupervisorLoopContext<'_, 'a>,
    spawn: SpawnRequest,
) -> Result<usize> {
    let worker_id = *cx.worker_counter;
    *cx.worker_counter += 1;
    let clone_label = format!("c{}", *cx.clone_counter);
    *cx.clone_counter += 1;
    let parent_commit = cx
        .dag
        .commit_for(&spawn.from)
        .ok_or_else(|| anyhow!("unknown checkpoint {}", spawn.from))?
        .to_string();
    let messages = cx.dag.ancestor_messages(&spawn.from)?;
    let (future, meta) = launch_worker(
        WorkerLaunch {
            cx: cx.launch.cx,
            sessions: cx.launch.sessions,
            session_id: cx.launch.session_id,
            llm: cx.launch.llm,
            parent_cwd: cx.launch.parent_cwd,
            config: cx.launch.config,
            reasoning_effort: cx.launch.reasoning_effort,
            service_tier: cx.launch.service_tier,
            structured_output: cx.launch.structured_output,
            idle_timeout: cx.launch.idle_timeout,
            worker_cancel: cx.launch.worker_cancel.clone(),
            original_task: cx.launch.original_task,
            live_output: cx.launch.live_output,
            context_length: cx.launch.context_length,
            context_prefix_len: cx.launch.context_prefix_len,
        },
        &spawn,
        worker_id,
        clone_label,
        parent_commit,
        messages,
        cx.idle_pool,
    )
    .await?;
    cx.running.push(future);
    cx.running_meta.insert(worker_id, meta);
    Ok(worker_id)
}

fn merge_checkpoint(
    stage: &SnapshotStage,
    dag: &mut TrajectoryDag,
    from: CheckpointId,
    onto: CheckpointId,
    worker: usize,
) -> Result<String> {
    match from {
        CheckpointId::Root => return Err(anyhow!("from root is not a saved checkpoint")),
        CheckpointId::Worker(worker) => dag
            .node(worker)
            .ok_or_else(|| anyhow!("from w{worker} is not a saved checkpoint"))?,
    };
    let from_commit = dag
        .commit_for(&from)
        .ok_or_else(|| anyhow!("unknown checkpoint {from}"))?
        .to_string();
    let onto_commit = dag
        .commit_for(&onto)
        .ok_or_else(|| anyhow!("unknown checkpoint {onto}"))?
        .to_string();
    let name = format!("w{worker}");
    let (commit, diffstat) = match stage.merge_checkpoint(&from_commit, &onto_commit, &name)? {
        crate::asgard::MergeCheckpointOutcome::Merged { commit, diffstat } => (commit, diffstat),
        crate::asgard::MergeCheckpointOutcome::NoChanges { diffstat } => {
            let mut result = format!("merged {from} onto {onto}");
            if !diffstat.trim().is_empty() {
                result.push_str(&format!(" (diffstat: {})", diffstat.trim()));
            }
            result.push_str(": merge produced no changes; onto already contains this content");
            return Ok(result);
        }
    };
    let instruction_text = format!(
        "The harness merged checkpoint {from}'s changes into this branch:\n{}",
        diffstat.trim_end()
    );
    dag.insert(TrajectoryNode {
        window: TrajectoryWindow {
            worker,
            parent: onto.clone(),
            instructions: format!("merged {from}'s changes onto {onto}"),
            model: "asgard".to_string(),
            instruction_message: ChatMessage::user(instruction_text),
            window_messages: Vec::new(),
            compact: String::new(),
            final_response: String::new(),
            stop: WorkerStopReason::Finished,
            steps: 0,
            diffstat: diffstat.clone(),
            usage: TokenUsage::default(),
            elapsed_millis: 0,
        },
        commit: commit.clone(),
        merged_from: vec![from.clone()],
    })?;
    crate::trace_logging::append_trace_record(serde_json::json!({
        "type": "asgard_checkpoint",
        "worker": worker,
        "parent": onto.to_string(),
        "commit": commit,
        "synthetic": "merge_checkpoint",
        "from": from.to_string(),
    }));
    Ok(format!(
        "merged {from} onto {onto} as w{worker} (diffstat: {})",
        diffstat.trim()
    ))
}

fn resolve_pending(
    stage: &SnapshotStage,
    dag: &mut TrajectoryDag,
    finished: &FinishedWorker,
    window: TrajectoryWindow,
    resolution: PendingResolution,
) -> Result<()> {
    match resolution {
        PendingResolution::Save => {
            let parent_commit = dag
                .commit_for(&window.parent)
                .ok_or_else(|| anyhow!("unknown checkpoint {}", window.parent))?
                .to_string();
            let commit = stage.snapshot(
                &finished.repository.root,
                &parent_commit,
                &format!("w{}", finished.worker),
            )?;
            dag.insert(TrajectoryNode {
                window,
                commit: commit.clone(),
                merged_from: Vec::new(),
            })?;
            crate::trace_logging::append_trace_record(serde_json::json!({
                "type": "asgard_checkpoint",
                "worker": finished.worker,
                "parent": finished.parent.to_string(),
                "commit": commit,
            }));
        }
        PendingResolution::Discard => {
            dag.discard(
                finished.worker,
                finished.parent.clone(),
                dag_instruction_stub(&finished.instructions),
            )?;
            crate::trace_logging::append_trace_record(serde_json::json!({
                "type": "asgard_discard",
                "worker": finished.worker,
            }));
        }
    }
    Ok(())
}

fn dag_instruction_stub(value: &str) -> String {
    let text = value.replace('\n', " ");
    if text.chars().count() <= 60 {
        text
    } else {
        text.chars().take(60).collect()
    }
}

fn worker_head_moved(root: &Path, parent_commit: &str, worker: usize) -> bool {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output();
    let Ok(output) = output else {
        tracing::warn!(worker, "failed to inspect Asgard worker HEAD");
        return false;
    };
    if !output.status.success() {
        tracing::warn!(worker, "git rev-parse HEAD failed for Asgard worker");
        return false;
    }
    let head = String::from_utf8_lossy(&output.stdout);
    let moved = head.trim() != parent_commit;
    if moved {
        tracing::warn!(
            worker,
            expected = parent_commit,
            actual = head.trim(),
            "Asgard worker moved HEAD; snapshot will ignore worker history"
        );
    }
    moved
}

fn finalize_asgard(
    context: FinalizeContext<'_>,
    checkpoint: CheckpointId,
    response: Option<String>,
    usage: TokenUsage,
) -> LoopOutcome {
    let Some(checkpoint_commit) = context.dag.commit_for(&checkpoint) else {
        return asgard_failure(anyhow!("unknown checkpoint {checkpoint}"));
    };
    let patch = match context
        .stage
        .finalize_patch(context.base_commit, checkpoint_commit)
    {
        Ok(patch) => patch,
        Err(error) => return asgard_failure(error),
    };
    if let Err(error) = crate::asgard::apply_selected_patch(context.parent_cwd, &patch) {
        return asgard_failure(error);
    }
    let final_text = response.unwrap_or_else(|| match checkpoint {
        CheckpointId::Root => String::new(),
        CheckpointId::Worker(worker) => context
            .dag
            .node(worker)
            .map(|node| node.window.final_response.clone())
            .unwrap_or_default(),
    });
    let off_lineage_unmerged = context
        .dag
        .off_lineage_checkpoints_with_diffstat(&checkpoint)
        .into_iter()
        .map(|entry| {
            serde_json::json!({
                "checkpoint": entry.checkpoint.to_string(),
                "diffstat": entry.diffstat,
            })
        })
        .collect::<Vec<_>>();
    crate::trace_logging::append_trace_record(serde_json::json!({
        "type": "asgard_finalize",
        "checkpoint": checkpoint.to_string(),
        "commit": checkpoint_commit,
        "response_bytes": final_text.len(),
        "evidence": context.evidence,
        "off_lineage_unmerged": off_lineage_unmerged,
        "abandoned": context.abandoned,
        "prefinalize_workers": context.prefinalize_workers,
    }));
    let evidence_text = if context.evidence.is_empty() {
        "no evidence named".to_string()
    } else {
        format!("evidence: {}", context.evidence.join(", "))
    };
    send_thought(
        &context.live_output.cx,
        &context.live_output.session_id,
        &format!("Asgard finalized checkpoint {checkpoint}; {evidence_text}.\n"),
    );
    let continuation_messages = match context.dag.ancestor_messages(&checkpoint) {
        Ok(messages) => messages,
        Err(error) => return asgard_failure(error),
    };
    LoopOutcome {
        response: final_text.clone(),
        tool_exchanges: Vec::new(),
        replay_events: Vec::new(),
        usage,
        stop: LoopStop::Completed {
            had_text: !final_text.is_empty(),
        },
        continuation_messages,
        current_plan: context.current_plan.cloned(),
        compaction_checkpoint: None,
    }
}

async fn worker_tool_allowlist(registry: &Arc<crate::tools::ToolRegistry>) -> HashSet<String> {
    registry
        .tool_definitions()
        .await
        .into_iter()
        .map(|tool| tool.function.name)
        .filter(|name| name != "update_plan")
        .collect()
}

fn add_usage(
    aggregate: &mut TokenUsage,
    usage_by_model: &mut BTreeMap<String, TokenUsage>,
    model: &str,
    usage: TokenUsage,
) {
    aggregate.add(usage);
    usage_by_model
        .entry(model.to_string())
        .or_default()
        .add(usage);
}

fn trace_supervisor_turn(
    turn: usize,
    reviewed: Option<usize>,
    result: &Result<SupervisorLoopResult>,
    elapsed_millis: u64,
    permanent_bytes: usize,
) {
    let (decision, steps, usage) = match result {
        Ok(outcome) => (
            supervisor_decision_summary(outcome),
            outcome.steps,
            outcome.usage,
        ),
        Err(error) => (format!("error: {error:#}"), 0, TokenUsage::default()),
    };
    crate::trace_logging::append_trace_record(serde_json::json!({
        "type": "asgard_supervisor_turn",
        "turn": turn,
        "reviewed": reviewed,
        "decision": decision,
        "elapsed_millis": elapsed_millis,
        "steps": steps,
        "usage": usage_json(usage),
        "permanent_bytes": permanent_bytes,
    }));
}

fn trace_supervisor_llm_request(
    turn: usize,
    model: &str,
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
) {
    crate::trace_logging::append_trace_record(serde_json::json!({
        "type": "llm_request",
        "actor": "asgard_supervisor",
        "turn": turn,
        "model": model,
        "reasoning_effort": null,
        "service_tier": null,
        "messages": messages,
        "tools": tools,
    }));
}

fn trace_supervisor_llm_response(turn: usize, response: &LlmResponse) {
    match response {
        LlmResponse::Text { text, usage, .. } => {
            crate::trace_logging::append_trace_record(serde_json::json!({
                "type": "llm_response",
                "actor": "asgard_supervisor",
                "turn": turn,
                "response": {
                    "kind": "text",
                    "text": text,
                },
                "usage": usage_json(*usage),
            }));
        }
        LlmResponse::ToolCalls {
            text, calls, usage, ..
        } => {
            crate::trace_logging::append_trace_record(serde_json::json!({
                "type": "llm_response",
                "actor": "asgard_supervisor",
                "turn": turn,
                "response": {
                    "kind": "tool_calls",
                    "text": text,
                    "tool_calls": calls,
                },
                "usage": usage_json(*usage),
            }));
        }
    }
}

fn supervisor_decision_summary(outcome: &SupervisorLoopResult) -> String {
    if let Some((checkpoint, response)) = &outcome.finalizing {
        return format!(
            "spawned={:?} prefinalized={:?} saved={} finalized=true checkpoint={checkpoint} response={} discarded={}",
            outcome.spawned,
            outcome.prefinalized,
            outcome.saved,
            response.as_ref().is_some_and(|value| !value.is_empty()),
            outcome.discarded
        );
    }
    format!(
        "spawned={:?} prefinalized={:?} merged={:?} saved={} finalized=false discarded={}",
        outcome.spawned, outcome.prefinalized, outcome.merged, outcome.saved, outcome.discarded
    )
}

fn trace_usage_by_model(turn: usize, usage_by_model: &BTreeMap<String, TokenUsage>) {
    crate::trace_logging::append_trace_record(serde_json::json!({
        "type": "asgard_usage_by_model",
        "window": turn,
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
}

fn usage_json(usage: TokenUsage) -> serde_json::Value {
    serde_json::json!({
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "thought": usage.thought_tokens,
        "cachedRead": usage.cached_read_tokens,
        "cachedWrite": usage.cached_write_tokens,
    })
}

fn latest_saved_checkpoint(dag: &TrajectoryDag) -> Option<CheckpointId> {
    dag.checkpoint_labels()
        .into_iter()
        .filter_map(|label| CheckpointId::parse(&label))
        .max_by_key(|checkpoint| match checkpoint {
            CheckpointId::Root => 0,
            CheckpointId::Worker(worker) => *worker,
        })
}

fn checkpoint_commit_map(dag: &TrajectoryDag) -> BTreeMap<String, String> {
    let mut checkpoints = BTreeMap::from([(
        CheckpointId::Root.to_string(),
        dag.commit_for(&CheckpointId::Root)
            .unwrap_or_default()
            .to_string(),
    )]);
    for label in dag.checkpoint_labels() {
        if let Some(checkpoint) = CheckpointId::parse(&label)
            && let Some(commit) = dag.commit_for(&checkpoint)
        {
            checkpoints.insert(label, commit.to_string());
        }
    }
    checkpoints
}

fn serialized_messages_len(messages: &[ChatMessage]) -> usize {
    serde_json::to_vec(messages)
        .map(|bytes| bytes.len())
        .unwrap_or_else(|_| {
            messages
                .iter()
                .map(|message| message.content_text().len())
                .sum()
        })
}

pub(crate) fn asgard_original_task(initial_messages: &[ChatMessage]) -> String {
    initial_messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .map(ChatMessage::content_text)
        .unwrap_or_default()
}

fn initial_permanent_user_message(
    original_task: &str,
    contracts: &crate::asgard::IntakeContracts,
) -> String {
    let base = format!(
        "<task>\n{original_task}\n</task>\n\
         The repository's starting state is checkpoint \"root\"."
    );
    if contracts.literal.is_none() && contracts.grounded.is_none() {
        return base;
    }

    let mut message = format!(
        "{base}\n\n\
         <spec_intake>\n\
         Before this run began, two independent readers examined the task. Reader L read \
         the task text alone with no repository access; Reader G examined the repository \
         for evidence that constrains interpretation. Neither saw the other's report.\n"
    );
    if let Some(literal) = &contracts.literal {
        message.push_str("<literal_contract>\n");
        message.push_str(literal);
        message.push_str("\n</literal_contract>\n");
    }
    if let Some(grounded) = &contracts.grounded {
        message.push_str("<grounded_contract>\n");
        message.push_str(grounded);
        message.push_str("\n</grounded_contract>\n");
    }
    message.push_str("</spec_intake>");
    message
}

pub(crate) fn asgard_take_window_messages(
    continuation_messages: &[ChatMessage],
    start: usize,
) -> Vec<ChatMessage> {
    continuation_messages
        .get(start..)
        .map_or_else(Vec::new, <[ChatMessage]>::to_vec)
}

/// Composes the supervisor's system message from the session's system prompt.
///
/// The supervisor keeps the agent doctrine sections ("How you work",
/// "Verification") but never holds file, shell, or skill tools, so everything
/// from the "# Tools" heading onward — Tools, Output, Safety, and the
/// appended skills/MCP catalogs — is replaced by the Asgard supervision
/// supplement. A session prompt without the marker is kept whole.
fn supervisor_system_message(session_system: Option<&str>) -> String {
    match session_system {
        Some(text) => {
            let head = text.split("\n# Tools").next().unwrap_or(text).trim_end();
            format!("{head}\n\n{}", supervisor_supplement())
        }
        None => supervisor_supplement(),
    }
}

fn worker_stop_reason(stop: &LoopStop) -> WorkerStopReason {
    match stop {
        LoopStop::Completed { .. } => WorkerStopReason::Finished,
        LoopStop::MaxTurns { .. } => WorkerStopReason::StepLimit,
        LoopStop::Cancelled => WorkerStopReason::Cancelled,
        LoopStop::Failed(failure) => WorkerStopReason::Failed(failure.message.clone()),
    }
}

pub(crate) fn extract_worker_final_response(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .rev()
        .find(|message| {
            message.role == "assistant"
                && message
                    .tool_calls
                    .as_ref()
                    .is_none_or(|calls| calls.is_empty())
        })
        .map(ChatMessage::content_text)
        .unwrap_or_default()
}

fn count_worker_steps(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .filter(|message| {
            message.role == "assistant"
                && message
                    .tool_calls
                    .as_ref()
                    .is_some_and(|calls| !calls.is_empty())
        })
        .count()
}

fn render_asgard_status_block(
    dag: &TrajectoryDag,
    running: &HashMap<usize, RunningWorkerMeta>,
    review_queue: &VecDeque<FinishedWorker>,
    pending: Option<&FinishedWorker>,
    capacity_available: usize,
    idle_note: Option<&str>,
) -> String {
    let mut live = Vec::new();
    live.extend(running.iter().map(|(worker, meta)| DagLiveEntry {
        worker: *worker,
        parent: meta.parent.clone(),
        status: format!(
            "in flight, step {}/{}",
            meta.turn_progress.load(Ordering::Relaxed),
            ASGARD_WORKER_MAX_STEPS
        ),
        instructions: meta.instructions.clone(),
    }));
    live.extend(review_queue.iter().map(|worker| DagLiveEntry {
        worker: worker.worker,
        parent: worker.parent.clone(),
        status: "queued for review".to_string(),
        instructions: worker.instructions.clone(),
    }));
    if let Some(worker) = pending {
        live.push(DagLiveEntry {
            worker: worker.worker,
            parent: worker.parent.clone(),
            status: "under review".to_string(),
            instructions: worker.instructions.clone(),
        });
    }

    let mut rendered = String::new();
    rendered.push_str("<asgard_status>\n");
    rendered.push_str("<dag>\n");
    rendered.push_str(&render_dag_overview(dag, &live));
    rendered.push_str("</dag>\n");
    rendered.push_str(&format!("capacity_available: {capacity_available}\n"));
    rendered.push_str("</asgard_status>\n");
    if let Some(note) = idle_note {
        rendered.push_str(note);
        rendered.push('\n');
    }
    rendered
}

pub(crate) fn rewrite_asgard_cwd(messages: &mut [ChatMessage], from: &Path, to: &Path) {
    // `PathBuf::join("")` preserves a trailing separator. That is how an
    // Asgard session rooted at the repository itself is represented, while
    // tool results and model-authored commands normally omit the separator.
    // Rewrite the path stem so both spellings are canonicalized.
    let from_display = from.display().to_string();
    let from_trimmed = from_display.trim_end_matches(['/', '\\']);
    let from = if from_trimmed.is_empty() {
        from_display.as_str()
    } else {
        from_trimmed
    };
    let to_display = to.display().to_string();
    let to_trimmed = to_display.trim_end_matches(['/', '\\']);
    let to = if to_trimmed.is_empty() {
        to_display.as_str()
    } else {
        to_trimmed
    };
    if from.is_empty() || from == to {
        return;
    }
    for message in messages {
        for part in &mut message.content {
            if let crate::llm_client::ChatContentPart::Text { text } = part {
                *text = text.replace(from, to);
            }
        }
        if let Some(reasoning) = &mut message.reasoning_content {
            *reasoning = reasoning.replace(from, to);
        }
        if let Some(calls) = &mut message.tool_calls {
            for call in calls {
                call.function.arguments = call.function.arguments.replace(from, to);
            }
        }
    }
}

pub(crate) fn cleanup_asgard_repositories(repositories: &[CandidateRepository]) {
    for repository in repositories {
        crate::asgard::remove_candidate_repository(repository);
    }
}

pub(crate) fn asgard_failure(error: anyhow::Error) -> LoopOutcome {
    let message = format!("Asgard failed: {error:#}");
    tracing::warn!("{message}");
    LoopOutcome {
        response: format!("\n**Error:** {message}\n"),
        tool_exchanges: Vec::new(),
        replay_events: Vec::new(),
        usage: TokenUsage::default(),
        stop: LoopStop::Failed(crate::tool_loop::TurnFailure {
            retryable: false,
            message,
        }),
        continuation_messages: Vec::new(),
        current_plan: None,
        compaction_checkpoint: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::fs;
    use std::process::Command;
    use std::sync::Mutex;
    use std::time::Duration;

    use agent_client_protocol::{Agent, ByteStreams, Dispatch, Handled, on_receive_dispatch};
    use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

    use crate::asgard::Config;
    use crate::llm_client::{FunctionCall, LlmResponse, StreamChatRequest, ToolCall};
    use crate::session::PermissionMode;

    fn tool_call(id: &str) -> ToolCall {
        named_tool_call(
            id,
            "run_shell_command",
            serde_json::json!({"command": "pwd"}),
        )
    }

    fn named_tool_call(id: &str, name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            r#type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    fn tool_response(calls: Vec<ToolCall>) -> LlmResponse {
        LlmResponse::ToolCalls {
            text: String::new(),
            reasoning_content: None,
            calls,
            usage: TokenUsage::default(),
        }
    }

    fn text_response(text: &str) -> LlmResponse {
        LlmResponse::Text {
            text: text.to_string(),
            reasoning_content: None,
            usage: TokenUsage::default(),
        }
    }

    fn empty_intake_response() -> LlmResponse {
        LlmResponse::ToolCalls {
            text: String::new(),
            reasoning_content: None,
            calls: vec![tool_call("empty-intake-tool-call")],
            usage: TokenUsage::default(),
        }
    }

    async fn run_scripted_asgard(
        repo: std::path::PathBuf,
        backend: Arc<ScriptedAsgardBackend>,
        initial_messages: Vec<ChatMessage>,
    ) -> (LoopOutcome, BTreeMap<String, TokenUsage>) {
        let llm: Arc<dyn crate::llm_client::LlmBackend> = backend;
        let sessions = SessionStore::new("worker-model".to_string());
        let session = sessions.create_session(repo.clone()).await;
        assert!(
            sessions
                .set_permission_mode(&session.id, PermissionMode::BypassPermissions)
                .await
        );
        let parent_registry = sessions
            .get_or_create_registry(&session.id, repo)
            .await
            .expect("parent registry");
        let config = Config {
            candidate_models: vec!["worker-model".to_string()],
            supervisor_model: Some("sv-model".to_string()),
        };
        let (agent_io, mut client_io) = tokio::io::duplex(1 << 20);
        let (agent_read, agent_write) = tokio::io::split(agent_io);
        let drain = tokio::spawn(async move {
            let _ = tokio::io::copy(&mut client_io, &mut tokio::io::sink()).await;
        });

        let result = Agent
            .builder()
            .on_receive_dispatch(
                async move |message: Dispatch, _cx| {
                    Ok(Handled::No {
                        message,
                        retry: false,
                    })
                },
                on_receive_dispatch!(),
            )
            .connect_with(
                ByteStreams::new(agent_write.compat_write(), agent_read.compat()),
                async |cx| {
                    Ok(run_asgard_trajectory_loop(
                        &cx,
                        &sessions,
                        &session.id,
                        &llm,
                        &parent_registry,
                        "worker-model",
                        None,
                        None,
                        None,
                        initial_messages,
                        IdleTimeouts::uniform(Duration::from_secs(30)),
                        tokio_util::sync::CancellationToken::new(),
                        &config,
                        None,
                        0,
                        None,
                    )
                    .await)
                },
            )
            .await
            .expect("in-memory ACP connect_with");
        drain.abort();
        result
    }

    fn spawn_call(id: &str, from: &str, instructions: &str) -> ToolCall {
        named_tool_call(
            id,
            "spawn_workers",
            serde_json::json!({
                "workers": [{
                    "from": from,
                    "instructions": instructions,
                }],
            }),
        )
    }

    fn spawn_workers_call(id: &str, workers: serde_json::Value) -> ToolCall {
        named_tool_call(
            id,
            "spawn_workers",
            serde_json::json!({ "workers": workers }),
        )
    }

    fn prefinalize_call(id: &str, from: &str, instructions: &str) -> ToolCall {
        named_tool_call(
            id,
            "prefinalize",
            serde_json::json!({
                "workers": [{
                    "from": from,
                    "instructions": instructions,
                }],
            }),
        )
    }

    fn prefinalize_workers_call(id: &str, workers: serde_json::Value) -> ToolCall {
        named_tool_call(id, "prefinalize", serde_json::json!({ "workers": workers }))
    }

    fn save_call(id: &str) -> ToolCall {
        named_tool_call(id, "save_checkpoint", serde_json::json!({}))
    }

    fn discard_call(id: &str) -> ToolCall {
        named_tool_call(id, "discard", serde_json::json!({}))
    }

    fn finalize_call(id: &str, checkpoint: &str) -> ToolCall {
        named_tool_call(
            id,
            "finalize",
            serde_json::json!({ "checkpoint": checkpoint }),
        )
    }

    fn finalize_call_with_evidence(id: &str, checkpoint: &str, evidence: &[&str]) -> ToolCall {
        named_tool_call(
            id,
            "finalize",
            serde_json::json!({ "checkpoint": checkpoint, "evidence": evidence }),
        )
    }

    fn finalize_call_with_evidence_and_abandoned(
        id: &str,
        checkpoint: &str,
        evidence: &[&str],
        abandoned: &[&str],
    ) -> ToolCall {
        named_tool_call(
            id,
            "finalize",
            serde_json::json!({
                "checkpoint": checkpoint,
                "evidence": evidence,
                "abandoned": abandoned
            }),
        )
    }

    fn merge_call(id: &str, from: &str, onto: &str) -> ToolCall {
        named_tool_call(
            id,
            "merge_checkpoint",
            serde_json::json!({ "from": from, "onto": onto }),
        )
    }

    fn wait_call(id: &str) -> ToolCall {
        named_tool_call(id, "wait", serde_json::json!({}))
    }

    fn update_plan_call(id: &str, step: &str, status: &str) -> ToolCall {
        named_tool_call(
            id,
            "update_plan",
            serde_json::json!({
                "plan": [{
                    "step": step,
                    "status": status,
                }],
            }),
        )
    }

    fn view_call(id: &str, handles: &[&str]) -> ToolCall {
        named_tool_call(
            id,
            "view_tool_call",
            serde_json::json!({ "handles": handles }),
        )
    }

    fn write_file_call(id: &str, file_path: &str, content: &str) -> ToolCall {
        named_tool_call(
            id,
            "write_file",
            serde_json::json!({
                "file_path": file_path,
                "content": content,
            }),
        )
    }

    #[derive(Debug)]
    struct RecordedRequest {
        model: String,
        messages: Vec<ChatMessage>,
        tool_names: Vec<String>,
    }

    struct ScriptedAsgardBackend {
        responses: HashMap<String, Mutex<VecDeque<LlmResponse>>>,
        requests: Mutex<Vec<RecordedRequest>>,
    }

    impl ScriptedAsgardBackend {
        fn new(responses: Vec<(&str, Vec<LlmResponse>)>) -> Self {
            Self {
                responses: responses
                    .into_iter()
                    .map(|(model, responses)| (model.to_string(), Mutex::new(responses.into())))
                    .collect(),
                requests: Mutex::new(Vec::new()),
            }
        }
    }

    impl crate::llm_client::LlmBackend for ScriptedAsgardBackend {
        fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>>> {
            Box::pin(async { Ok(Vec::new()) })
        }

        fn stream_chat(&self, request: StreamChatRequest) -> BoxFuture<'_, Result<LlmResponse>> {
            let StreamChatRequest {
                model,
                messages,
                tools,
                ..
            } = request;
            let roles = messages
                .iter()
                .map(|message| message.role.clone())
                .collect::<Vec<_>>();
            let tool_names = tools
                .unwrap_or_default()
                .into_iter()
                .map(|tool| tool.function.name)
                .collect::<Vec<_>>();
            self.requests
                .lock()
                .expect("requests lock")
                .push(RecordedRequest {
                    model: model.clone(),
                    messages,
                    tool_names,
                });
            let response = self
                .responses
                .get(&model)
                .unwrap_or_else(|| panic!("no scripted responses registered for model {model}"))
                .lock()
                .expect("response lock")
                .pop_front()
                .unwrap_or_else(|| {
                    panic!(
                        "empty scripted response queue for model {model}; last request roles: {roles:?}"
                    )
                });
            Box::pin(async move { Ok(response) })
        }
    }

    fn run_git(root: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .unwrap_or_else(|error| panic!("git {} failed to start: {error}", args.join(" ")));
        assert!(
            output.status.success(),
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).expect("git stdout was not UTF-8")
    }

    fn saved_node(
        worker: usize,
        parent: CheckpointId,
        instructions: &str,
        commit: String,
    ) -> TrajectoryNode {
        TrajectoryNode {
            window: TrajectoryWindow {
                worker,
                parent,
                instructions: instructions.to_string(),
                model: "model-a".to_string(),
                instruction_message: ChatMessage::user(instructions),
                window_messages: Vec::new(),
                compact: String::new(),
                final_response: String::new(),
                stop: WorkerStopReason::Finished,
                steps: 1,
                diffstat: String::new(),
                usage: TokenUsage::default(),
                elapsed_millis: 0,
            },
            commit,
            merged_from: Vec::new(),
        }
    }

    fn all_message_text(messages: &[ChatMessage]) -> String {
        messages
            .iter()
            .map(ChatMessage::content_text)
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[tokio::test]
    async fn asgard_intake_injects_contracts_and_uses_read_only_tools() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).expect("create repo dir");
        run_git(&repo, &["init", "--quiet"]);
        run_git(&repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(&repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("README.md"), "hello\n").expect("write README");
        run_git(&repo, &["add", "README.md"]);
        run_git(&repo, &["commit", "--quiet", "-m", "initial"]);

        let sessions = SessionStore::new("worker-model".to_string());
        let session = sessions.create_session(repo.clone()).await;
        let parent_registry = sessions
            .get_or_create_registry(&session.id, repo.clone())
            .await
            .expect("parent registry");
        let allowlist = crate::asgard::intake_read_only_allowlist(&parent_registry).await;
        assert!(allowlist.contains("read_file"));
        assert!(!allowlist.contains("run_shell_command"));
        assert!(!allowlist.contains("write_file"));
        assert!(!allowlist.contains("edit"));
        assert!(!allowlist.contains("update_plan"));

        let backend = Arc::new(ScriptedAsgardBackend::new(vec![
            (
                "sv-model",
                vec![
                    text_response("literal contract text"),
                    tool_response(vec![spawn_call("sv-spawn-w1", "root", "report")]),
                    text_response("w1 launched"),
                    tool_response(vec![finalize_call("sv-finalize", "w1")]),
                    tool_response(vec![finalize_call("sv-finalize-again", "w1")]),
                    text_response("idle two"),
                    text_response("idle three"),
                    text_response("idle four"),
                    text_response("idle five"),
                    text_response("idle six"),
                    text_response("idle seven"),
                ],
            ),
            (
                "worker-model",
                vec![
                    text_response("grounded contract text"),
                    text_response("worker final report"),
                ],
            ),
        ]));
        let _ = run_scripted_asgard(
            repo.clone(),
            backend.clone(),
            vec![ChatMessage::user("implement intake test")],
        )
        .await;

        {
            let requests = backend.requests.lock().expect("requests");
            let supervisor_request = requests
                .iter()
                .find(|request| {
                    request.model == "sv-model"
                        && request
                            .tool_names
                            .iter()
                            .any(|name| name == crate::asgard::SPAWN_WORKERS_TOOL)
                })
                .expect("supervisor request");
            let initial_user = supervisor_request.messages[1].content_text();
            assert!(initial_user.contains("<spec_intake>"));
            assert!(
                initial_user
                    .contains("<literal_contract>\nliteral contract text\n</literal_contract>")
            );
            assert!(
                initial_user
                    .contains("<grounded_contract>\ngrounded contract text\n</grounded_contract>")
            );
        }

        let temp_failure = tempfile::tempdir().expect("tempdir");
        let repo_failure = temp_failure.path().join("repo");
        fs::create_dir(&repo_failure).expect("create repo dir");
        run_git(&repo_failure, &["init", "--quiet"]);
        run_git(
            &repo_failure,
            &["config", "user.email", "asgard@example.invalid"],
        );
        run_git(&repo_failure, &["config", "user.name", "Asgard Test"]);
        fs::write(repo_failure.join("README.md"), "hello\n").expect("write README");
        run_git(&repo_failure, &["add", "README.md"]);
        run_git(&repo_failure, &["commit", "--quiet", "-m", "initial"]);

        let failure_backend = Arc::new(ScriptedAsgardBackend::new(vec![
            (
                "sv-model",
                vec![
                    empty_intake_response(),
                    tool_response(vec![spawn_call("sv-spawn-w1", "root", "report")]),
                    text_response("w1 launched"),
                    tool_response(vec![finalize_call("sv-finalize", "w1")]),
                    tool_response(vec![finalize_call("sv-finalize-again", "w1")]),
                    text_response("idle two"),
                    text_response("idle three"),
                    text_response("idle four"),
                    text_response("idle five"),
                    text_response("idle six"),
                    text_response("idle seven"),
                ],
            ),
            (
                "worker-model",
                vec![
                    text_response("grounded only contract"),
                    text_response("worker final report"),
                ],
            ),
        ]));
        let _ = run_scripted_asgard(
            repo_failure,
            failure_backend.clone(),
            vec![ChatMessage::user("implement intake failure test")],
        )
        .await;
        let failure_requests = failure_backend.requests.lock().expect("failure requests");
        let failure_supervisor_request = failure_requests
            .iter()
            .find(|request| {
                request.model == "sv-model"
                    && request
                        .tool_names
                        .iter()
                        .any(|name| name == crate::asgard::SPAWN_WORKERS_TOOL)
            })
            .expect("failure supervisor request");
        let failure_initial_user = failure_supervisor_request.messages[1].content_text();
        assert!(failure_initial_user.contains("<spec_intake>"));
        assert!(!failure_initial_user.contains("<literal_contract>"));
        assert!(
            failure_initial_user
                .contains("<grounded_contract>\ngrounded only contract\n</grounded_contract>")
        );
    }

    #[tokio::test]
    async fn asgard_grounded_intake_forces_report_after_step_cap() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).expect("create repo dir");
        run_git(&repo, &["init", "--quiet"]);
        run_git(&repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(&repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("README.md"), "hello\n").expect("write README");
        run_git(&repo, &["add", "README.md"]);
        run_git(&repo, &["commit", "--quiet", "-m", "initial"]);

        let mut worker_responses = Vec::new();
        for index in 0..crate::asgard::ASGARD_INTAKE_MAX_STEPS {
            worker_responses.push(tool_response(vec![named_tool_call(
                &format!("intake-read-{index}"),
                "read_file",
                serde_json::json!({ "file_path": "README.md" }),
            )]));
        }
        worker_responses.push(text_response(
            "1. forced grounded report from observed README",
        ));
        worker_responses.push(text_response("worker final report"));

        let backend = Arc::new(ScriptedAsgardBackend::new(vec![
            (
                "sv-model",
                vec![
                    text_response("literal contract text"),
                    tool_response(vec![spawn_call("sv-spawn-w1", "root", "report")]),
                    text_response("w1 launched"),
                    tool_response(vec![finalize_call("sv-finalize", "w1")]),
                    tool_response(vec![finalize_call("sv-finalize-again", "w1")]),
                    text_response("idle two"),
                    text_response("idle three"),
                    text_response("idle four"),
                ],
            ),
            ("worker-model", worker_responses),
        ]));
        let _ = run_scripted_asgard(
            repo,
            backend.clone(),
            vec![ChatMessage::user("implement intake forced report test")],
        )
        .await;

        let requests = backend.requests.lock().expect("requests");
        let fallback_request = requests
            .iter()
            .find(|request| {
                request.model == "worker-model"
                    && request.tool_names.is_empty()
                    && request.messages.last().is_some_and(|message| {
                        message
                            .content_text()
                            .contains("Write your numbered report now")
                    })
            })
            .expect("forced grounded intake fallback request");
        assert!(
            fallback_request
                .messages
                .last()
                .expect("fallback user message")
                .content_text()
                .contains("Do not call any tools.")
        );

        let supervisor_request = requests
            .iter()
            .find(|request| {
                request.model == "sv-model"
                    && request
                        .tool_names
                        .iter()
                        .any(|name| name == crate::asgard::SPAWN_WORKERS_TOOL)
            })
            .expect("supervisor request");
        let initial_user = supervisor_request.messages[1].content_text();
        assert!(initial_user.contains(
            "<grounded_contract>\n1. forced grounded report from observed README\n</grounded_contract>"
        ));
    }

    #[tokio::test]
    async fn asgard_spawn_workers_dedups_in_call_and_running_specs() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).expect("create repo dir");
        run_git(&repo, &["init", "--quiet"]);
        run_git(&repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(&repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("README.md"), "hello\n").expect("write README");
        run_git(&repo, &["add", "README.md"]);
        run_git(&repo, &["commit", "--quiet", "-m", "initial"]);

        let duplicate = "inspect parser";
        let distinct = "inspect planner";
        let backend = Arc::new(ScriptedAsgardBackend::new(vec![
            (
                "sv-model",
                vec![
                    text_response("literal contract text"),
                    tool_response(vec![spawn_workers_call(
                        "sv-spawn-batch",
                        serde_json::json!([
                            { "from": "root", "instructions": duplicate },
                            { "from": "root", "instructions": duplicate },
                            { "from": "root", "instructions": distinct }
                        ]),
                    )]),
                    tool_response(vec![spawn_call(
                        "sv-spawn-running-duplicate",
                        "root",
                        duplicate,
                    )]),
                    text_response("waiting"),
                    tool_response(vec![discard_call("sv-discard-w1")]),
                    tool_response(vec![discard_call("sv-discard-w2")]),
                    text_response("done"),
                    text_response("done"),
                    text_response("done"),
                    text_response("done"),
                    text_response("done"),
                    text_response("done"),
                    text_response("done"),
                ],
            ),
            (
                "worker-model",
                vec![
                    text_response("grounded contract text"),
                    tool_response(vec![named_tool_call(
                        "w1-read",
                        "read_file",
                        serde_json::json!({ "file_path": "README.md" }),
                    )]),
                    tool_response(vec![named_tool_call(
                        "w2-read",
                        "read_file",
                        serde_json::json!({ "file_path": "README.md" }),
                    )]),
                    text_response("w1 report"),
                    text_response("w2 report"),
                ],
            ),
        ]));
        let _ = run_scripted_asgard(
            repo,
            backend.clone(),
            vec![ChatMessage::user("implement spawn dedup test")],
        )
        .await;

        let requests = backend.requests.lock().expect("requests");
        let supervisor_text = requests
            .iter()
            .filter(|request| request.model == "sv-model")
            .map(|request| all_message_text(&request.messages))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(supervisor_text.contains("spawned w1 from root"));
        assert!(supervisor_text.contains("spawned w2 from root"));
        assert!(supervisor_text.contains("skipped 1 duplicate specs (identical to w1)"));
        assert!(supervisor_text.contains("skipped spec identical to running w1"));

        let spawned_lines = supervisor_text
            .lines()
            .filter(|line| line.starts_with("spawned w"))
            .collect::<HashSet<_>>();
        assert_eq!(spawned_lines.len(), 2);
        assert!(
            !spawned_lines
                .iter()
                .any(|line| line.starts_with("spawned w3"))
        );
    }

    #[tokio::test]
    async fn prefinalize_gate_rejects_then_accepts_after_review() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).expect("create repo dir");
        run_git(&repo, &["init", "--quiet"]);
        run_git(&repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(&repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("README.md"), "hello\n").expect("write README");
        run_git(&repo, &["add", "README.md"]);
        run_git(&repo, &["commit", "--quiet", "-m", "initial"]);

        let backend = Arc::new(ScriptedAsgardBackend::new(vec![
            (
                "sv-model",
                vec![
                    text_response("intake literal"),
                    tool_response(vec![spawn_call("sv-spawn-w1", "root", "implement")]),
                    text_response("w1 launched"),
                    tool_response(vec![finalize_call_with_evidence(
                        "sv-finalize-before-prefinalize",
                        "w1",
                        &["w1m1"],
                    )]),
                    tool_response(vec![prefinalize_call(
                        "sv-prefinalize-w2",
                        "w1",
                        "verify w1",
                    )]),
                    text_response("prefinalize launched"),
                    tool_response(vec![
                        discard_call("sv-discard-w2"),
                        finalize_call_with_evidence("sv-finalize-w1", "w1", &["w1m1"]),
                    ]),
                ],
            ),
            (
                "worker-model",
                vec![
                    text_response("intake grounded"),
                    tool_response(vec![named_tool_call(
                        "w1-test",
                        "run_shell_command",
                        serde_json::json!({ "command": "true" }),
                    )]),
                    text_response("w1 report"),
                    text_response("w2 verification report"),
                ],
            ),
        ]));

        let (outcome, _) = run_scripted_asgard(
            repo,
            backend.clone(),
            vec![ChatMessage::user("exercise prefinalize gate")],
        )
        .await;

        assert_eq!(outcome.response, "w1 report");
        let supervisor_text = backend
            .requests
            .lock()
            .expect("requests")
            .iter()
            .filter(|request| request.model == "sv-model")
            .map(|request| all_message_text(&request.messages))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(supervisor_text.contains("error: finalize requires a prefinalize verification pass first: spawn verification workers via prefinalize, review their reports, then finalize."));
        assert!(supervisor_text.contains("prefinalize spawned w2 from w1"));
    }

    #[tokio::test]
    async fn prefinalize_gate_names_unreviewed_workers() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).expect("create repo dir");
        run_git(&repo, &["init", "--quiet"]);
        run_git(&repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(&repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("README.md"), "hello\n").expect("write README");
        run_git(&repo, &["add", "README.md"]);
        run_git(&repo, &["commit", "--quiet", "-m", "initial"]);

        let backend = Arc::new(ScriptedAsgardBackend::new(vec![
            (
                "sv-model",
                vec![
                    text_response("intake literal"),
                    tool_response(vec![spawn_call("sv-spawn-w1", "root", "implement")]),
                    text_response("w1 launched"),
                    tool_response(vec![
                        prefinalize_call("sv-prefinalize-w2", "w1", "verify w1"),
                        finalize_call_with_evidence("sv-finalize-unreviewed", "w1", &["w1m1"]),
                    ]),
                    text_response("saw gate"),
                    tool_response(vec![discard_call("sv-discard-w2")]),
                    text_response("idle one"),
                    text_response("idle two"),
                    text_response("idle three"),
                    text_response("idle four"),
                ],
            ),
            (
                "worker-model",
                vec![
                    text_response("intake grounded"),
                    tool_response(vec![named_tool_call(
                        "w1-test",
                        "run_shell_command",
                        serde_json::json!({ "command": "true" }),
                    )]),
                    text_response("w1 report"),
                    text_response("w2 verification report"),
                ],
            ),
        ]));

        let _ = run_scripted_asgard(
            repo,
            backend.clone(),
            vec![ChatMessage::user("exercise unreviewed prefinalize gate")],
        )
        .await;

        let supervisor_text = backend
            .requests
            .lock()
            .expect("requests")
            .iter()
            .filter(|request| request.model == "sv-model")
            .map(|request| all_message_text(&request.messages))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(supervisor_text.contains("error: prefinalize workers [w2] have not been reviewed yet; wait for them and review their reports before finalizing."));
    }

    #[tokio::test]
    async fn prefinalize_trajectories_are_viewable_but_verification_only() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).expect("create repo dir");
        run_git(&repo, &["init", "--quiet"]);
        run_git(&repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(&repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("README.md"), "hello\n").expect("write README");
        run_git(&repo, &["add", "README.md"]);
        run_git(&repo, &["commit", "--quiet", "-m", "initial"]);

        let backend = Arc::new(ScriptedAsgardBackend::new(vec![
            (
                "sv-model",
                vec![
                    text_response("intake literal"),
                    tool_response(vec![spawn_call("sv-spawn-w1", "root", "implement")]),
                    text_response("w1 launched"),
                    tool_response(vec![prefinalize_call(
                        "sv-prefinalize-w2",
                        "w1",
                        "verify w1",
                    )]),
                    text_response("prefinalize launched"),
                    tool_response(vec![
                        view_call("sv-view-w2", &["w2m1"]),
                        save_call("sv-save-w2"),
                        spawn_call("sv-spawn-from-w2", "w2", "continue from verification"),
                    ]),
                    text_response("leave w2 unresolved"),
                    tool_response(vec![spawn_call("sv-spawn-w3", "w1", "regular followup")]),
                    text_response("w3 launched"),
                    text_response("leave w3 unresolved"),
                    text_response("idle one"),
                    text_response("idle two"),
                    text_response("idle three"),
                    text_response("idle four"),
                    text_response("idle five"),
                    text_response("idle six"),
                ],
            ),
            (
                "worker-model",
                vec![
                    text_response("intake grounded"),
                    tool_response(vec![named_tool_call(
                        "w1-test",
                        "run_shell_command",
                        serde_json::json!({ "command": "true" }),
                    )]),
                    text_response("w1 report"),
                    tool_response(vec![named_tool_call(
                        "w2-read",
                        "read_file",
                        serde_json::json!({ "file_path": "README.md" }),
                    )]),
                    text_response("w2 verification report"),
                    text_response("w3 report"),
                ],
            ),
        ]));

        let (outcome, _) = run_scripted_asgard(
            repo,
            backend.clone(),
            vec![ChatMessage::user("exercise prefinalize verification-only")],
        )
        .await;

        assert_eq!(outcome.response, "w3 report");
        let supervisor_text = backend
            .requests
            .lock()
            .expect("requests")
            .iter()
            .filter(|request| request.model == "sv-model")
            .map(|request| all_message_text(&request.messages))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(supervisor_text.contains("[viewed w2m1: read_file"));
        assert!(supervisor_text.contains(&format!("error: {PREFINALIZE_VERIFICATION_ONLY_ERROR}")));
        assert!(supervisor_text.contains(&format!(
            "w2 was discarded ({PREFINALIZE_VERIFICATION_ONLY_ERROR})"
        )));
        assert!(supervisor_text.contains("w3 was auto-saved: the turn ended without resolving it"));
    }

    #[tokio::test]
    async fn finalize_requires_latest_prefinalize_source_commit() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).expect("create repo dir");
        run_git(&repo, &["init", "--quiet"]);
        run_git(&repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(&repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("README.md"), "hello\n").expect("write README");
        run_git(&repo, &["add", "README.md"]);
        run_git(&repo, &["commit", "--quiet", "-m", "initial"]);

        let backend = Arc::new(ScriptedAsgardBackend::new(vec![
            (
                "sv-model",
                vec![
                    text_response("intake literal"),
                    tool_response(vec![spawn_call("sv-spawn-w1", "root", "create a")]),
                    text_response("w1 launched"),
                    tool_response(vec![
                        save_call("sv-save-w1"),
                        spawn_call("sv-spawn-w2", "root", "create b"),
                    ]),
                    text_response("w2 launched"),
                    tool_response(vec![save_call("sv-save-w2")]),
                    tool_response(vec![prefinalize_call(
                        "sv-prefinalize-w3",
                        "w1",
                        "verify w1",
                    )]),
                    text_response("w3 launched"),
                    tool_response(vec![discard_call("sv-discard-w3")]),
                    tool_response(vec![merge_call("sv-merge", "w2", "w1")]),
                    tool_response(vec![finalize_call_with_evidence(
                        "sv-finalize-w4-before-verify",
                        "w4",
                        &["w1m2"],
                    )]),
                    tool_response(vec![prefinalize_call(
                        "sv-prefinalize-w5",
                        "w4",
                        "verify w4",
                    )]),
                    text_response("w5 launched"),
                    tool_response(vec![discard_call("sv-discard-w5")]),
                    tool_response(vec![finalize_call_with_evidence(
                        "sv-finalize-w4",
                        "w4",
                        &["w1m2"],
                    )]),
                ],
            ),
            (
                "worker-model",
                vec![
                    text_response("intake grounded"),
                    tool_response(vec![
                        write_file_call("w1-write", "a.txt", "a\n"),
                        named_tool_call(
                            "w1-test",
                            "run_shell_command",
                            serde_json::json!({ "command": "test -f a.txt" }),
                        ),
                    ]),
                    text_response("w1 done"),
                    tool_response(vec![write_file_call("w2-write", "b.txt", "b\n")]),
                    text_response("w2 done"),
                    text_response("w3 verification report"),
                    text_response("w5 verification report"),
                ],
            ),
        ]));

        let (outcome, _) = run_scripted_asgard(
            repo.clone(),
            backend.clone(),
            vec![ChatMessage::user("exercise prefinalize commit equality")],
        )
        .await;

        assert_eq!(outcome.response, "");
        assert_eq!(fs::read_to_string(repo.join("a.txt")).unwrap(), "a\n");
        assert_eq!(fs::read_to_string(repo.join("b.txt")).unwrap(), "b\n");

        let supervisor_text = backend
            .requests
            .lock()
            .expect("requests")
            .iter()
            .filter(|request| request.model == "sv-model")
            .map(|request| all_message_text(&request.messages))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(supervisor_text.contains("merged w2 onto w1 as w4"));
        assert!(supervisor_text.contains("error: the delivered checkpoint (w4) is not the state your latest prefinalize verified; run prefinalize from w4 (or the checkpoint you intend to deliver), review it, then finalize."));
    }

    #[tokio::test]
    async fn idle_fallback_finalize_bypasses_prefinalize_gate() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).expect("create repo dir");
        run_git(&repo, &["init", "--quiet"]);
        run_git(&repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(&repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("README.md"), "hello\n").expect("write README");
        run_git(&repo, &["add", "README.md"]);
        run_git(&repo, &["commit", "--quiet", "-m", "initial"]);

        let backend = Arc::new(ScriptedAsgardBackend::new(vec![
            (
                "sv-model",
                vec![
                    text_response("intake literal"),
                    tool_response(vec![spawn_call("sv-spawn-w1", "root", "finish")]),
                    text_response("w1 launched"),
                    tool_response(vec![save_call("sv-save-w1")]),
                    text_response("idle one"),
                    text_response("idle two"),
                    text_response("idle three"),
                    text_response("idle four"),
                ],
            ),
            (
                "worker-model",
                vec![text_response("intake grounded"), text_response("w1 report")],
            ),
        ]));

        let (outcome, _) = run_scripted_asgard(
            repo,
            backend,
            vec![ChatMessage::user("exercise idle fallback")],
        )
        .await;

        assert!(matches!(outcome.stop, LoopStop::Completed { .. }));
        assert_eq!(outcome.response, "w1 report");
    }

    #[tokio::test]
    async fn prefinalize_uses_spawn_dedup_path() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).expect("create repo dir");
        run_git(&repo, &["init", "--quiet"]);
        run_git(&repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(&repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("README.md"), "hello\n").expect("write README");
        run_git(&repo, &["add", "README.md"]);
        run_git(&repo, &["commit", "--quiet", "-m", "initial"]);

        let backend = Arc::new(ScriptedAsgardBackend::new(vec![
            (
                "sv-model",
                vec![
                    text_response("intake literal"),
                    tool_response(vec![prefinalize_workers_call(
                        "sv-prefinalize-batch",
                        serde_json::json!([
                            { "from": "root", "instructions": "verify" },
                            { "from": "root", "instructions": "verify" }
                        ]),
                    )]),
                    text_response("prefinalize launched"),
                    tool_response(vec![discard_call("sv-discard-w1")]),
                    text_response("idle one"),
                    text_response("idle two"),
                    text_response("idle three"),
                    text_response("idle four"),
                ],
            ),
            (
                "worker-model",
                vec![
                    text_response("intake grounded"),
                    text_response("w1 verification"),
                ],
            ),
        ]));

        let _ = run_scripted_asgard(
            repo,
            backend.clone(),
            vec![ChatMessage::user("exercise prefinalize dedup")],
        )
        .await;

        let supervisor_text = backend
            .requests
            .lock()
            .expect("requests")
            .iter()
            .filter(|request| request.model == "sv-model")
            .map(|request| all_message_text(&request.messages))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(supervisor_text.contains("prefinalize spawned w1 from root"));
        assert!(supervisor_text.contains("skipped 1 duplicate specs (identical to w1)"));
    }

    #[tokio::test]
    async fn asgard_v2_scripted_e2e_runs_real_loop_and_checkpoints() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).expect("create repo dir");
        run_git(&repo, &["init", "--quiet"]);
        run_git(&repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(&repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("README.md"), "hello\n").expect("write README");
        run_git(&repo, &["add", "README.md"]);
        run_git(&repo, &["commit", "--quiet", "-m", "initial"]);
        let base_head = run_git(&repo, &["rev-parse", "HEAD"]).trim().to_string();

        let backend = Arc::new(ScriptedAsgardBackend::new(vec![
            (
                "sv-model",
                vec![
                    text_response("intake literal"),
                    tool_response(vec![
                        spawn_call("sv-spawn-w1", "root", "Create foo.txt containing alpha"),
                        update_plan_call("sv-plan", "Create foo.txt", "in_progress"),
                    ]),
                    text_response("bootstrap workers launched"),
                    tool_response(vec![spawn_call(
                        "sv-spawn-w2",
                        "w1",
                        "Rewrite foo.txt to alpha then beta",
                    )]),
                    text_response("w2 launched from w1"),
                    tool_response(vec![
                        save_call("sv-save-w2"),
                        spawn_call(
                            "sv-spawn-w3",
                            "w2",
                            "Check whether README.md exists and report",
                        ),
                    ]),
                    text_response("w2 saved and w3 launched"),
                    text_response("w3 confirms README exists; nothing worth keeping"),
                    tool_response(vec![prefinalize_call(
                        "sv-prefinalize-w4",
                        "w2",
                        "verify w2",
                    )]),
                    text_response("w4 launched"),
                    tool_response(vec![discard_call("sv-discard-w4")]),
                    tool_response(vec![finalize_call("sv-finalize", "w2")]),
                    tool_response(vec![finalize_call("sv-finalize-again", "w2")]),
                ],
            ),
            (
                "worker-model",
                vec![
                    text_response("intake grounded"),
                    tool_response(vec![write_file_call("w1-write", "foo.txt", "alpha\n")]),
                    text_response("wrote foo.txt with alpha"),
                    tool_response(vec![write_file_call(
                        "w2-write",
                        "foo.txt",
                        "alpha\nbeta\n",
                    )]),
                    text_response("foo.txt now has alpha and beta"),
                    text_response("README.md exists; nothing to change"),
                    text_response("w4 verification report"),
                ],
            ),
        ]));
        let llm: Arc<dyn crate::llm_client::LlmBackend> = backend.clone();
        let sessions = SessionStore::new("worker-model".to_string());
        let session = sessions.create_session(repo.clone()).await;
        assert!(
            sessions
                .set_permission_mode(&session.id, PermissionMode::BypassPermissions)
                .await
        );
        let parent_registry = sessions
            .get_or_create_registry(&session.id, repo.clone())
            .await
            .expect("parent registry");
        let config = Config {
            candidate_models: vec!["worker-model".to_string()],
            supervisor_model: Some("sv-model".to_string()),
        };
        let initial_messages = vec![
            ChatMessage::system("You are a helpful engineer."),
            ChatMessage::user("Make foo.txt say alpha then beta."),
        ];

        let (agent_io, mut client_io) = tokio::io::duplex(1 << 20);
        let (agent_read, agent_write) = tokio::io::split(agent_io);
        let drain = tokio::spawn(async move {
            let _ = tokio::io::copy(&mut client_io, &mut tokio::io::sink()).await;
        });

        let (outcome, usage_by_model) = Agent
            .builder()
            .on_receive_dispatch(
                async move |message: Dispatch, _cx| {
                    Ok(Handled::No {
                        message,
                        retry: false,
                    })
                },
                on_receive_dispatch!(),
            )
            .connect_with(
                ByteStreams::new(agent_write.compat_write(), agent_read.compat()),
                async |cx| {
                    Ok(run_asgard_trajectory_loop(
                        &cx,
                        &sessions,
                        &session.id,
                        &llm,
                        &parent_registry,
                        "worker-model",
                        None,
                        None,
                        None,
                        initial_messages,
                        IdleTimeouts::uniform(Duration::from_secs(30)),
                        tokio_util::sync::CancellationToken::new(),
                        &config,
                        None,
                        0,
                        None,
                    )
                    .await)
                },
            )
            .await
            .expect("in-memory ACP connect_with");
        drain.abort();

        assert_eq!(outcome.response, "foo.txt now has alpha and beta");
        assert!(matches!(
            outcome.stop,
            LoopStop::Completed { had_text: true }
        ));
        assert_eq!(
            fs::read_to_string(repo.join("foo.txt")).expect("foo.txt"),
            "alpha\nbeta\n"
        );
        assert_eq!(
            fs::read_to_string(repo.join("README.md")).expect("README.md"),
            "hello\n"
        );
        assert_eq!(run_git(&repo, &["rev-parse", "HEAD"]).trim(), base_head);
        assert_eq!(run_git(&repo, &["for-each-ref", "refs/asgard"]), "");
        assert!(usage_by_model.contains_key("worker-model"));
        assert!(usage_by_model.contains_key("sv-model"));
        assert_eq!(
            outcome
                .current_plan
                .as_ref()
                .expect("Asgard supervisor plan")
                .plan[0]
                .step,
            "Create foo.txt"
        );

        let requests = backend.requests.lock().expect("requests");
        let supervisor_requests = requests
            .iter()
            .filter(|request| {
                request.model == "sv-model"
                    && request
                        .tool_names
                        .iter()
                        .any(|name| name == crate::asgard::SPAWN_WORKERS_TOOL)
            })
            .collect::<Vec<_>>();
        let worker_requests = requests
            .iter()
            .filter(|request| request.model == "worker-model")
            .collect::<Vec<_>>();
        assert_eq!(supervisor_requests.len(), 12);
        assert_eq!(worker_requests.len(), 7);
        let first_supervisor_system = supervisor_requests[0].messages[0].content_text();
        let session_prompt_pos = first_supervisor_system
            .find("You are a helpful engineer.")
            .expect("supervisor request includes session system prompt");
        let supplement_pos = first_supervisor_system
            .find("# Asgard supervision")
            .expect("supervisor request includes Asgard supplement");
        assert!(
            session_prompt_pos < supplement_pos,
            "session system prompt should precede Asgard supplement:\n{first_supervisor_system}"
        );
        let worker_agent_requests = worker_requests
            .iter()
            .copied()
            .filter(|request| request.tool_names.iter().any(|name| name == "write_file"))
            .collect::<Vec<_>>();
        assert_eq!(worker_agent_requests.len(), 6);
        assert!(
            worker_agent_requests
                .iter()
                .all(|request| request.tool_names.iter().any(|name| name == "write_file"))
        );
        assert!(
            worker_agent_requests
                .iter()
                .all(|request| request.tool_names.iter().all(|name| name != "update_plan"))
        );
        assert!(
            supervisor_requests[0]
                .tool_names
                .iter()
                .any(|name| name == "spawn_workers")
        );
        assert!(
            supervisor_requests[0]
                .tool_names
                .iter()
                .any(|name| name == "discard")
        );
        assert!(
            supervisor_requests[0]
                .tool_names
                .iter()
                .any(|name| name == "update_plan")
        );

        let review_w1_text = all_message_text(&supervisor_requests[2].messages);
        assert!(review_w1_text.contains(r#"<worker_trajectory id="w1" continues_from="root""#));
        assert!(review_w1_text.contains("<asgard_status>"));
        assert!(review_w1_text.contains("<dag>"));

        let w2_first_text = all_message_text(&worker_requests[2].messages);
        assert!(w2_first_text.contains("Create foo.txt containing alpha"));
        assert!(w2_first_text.contains("foo.txt"));
        assert!(
            !w2_first_text.contains(&repo.display().to_string()),
            "parent repo path leaked into w2 request:\n{w2_first_text}"
        );

        let review_w3_text = all_message_text(&supervisor_requests[6].messages);
        assert!(review_w3_text.contains("w2 \"Rewrite foo.txt to alpha then beta\" saved"));
        let final_idle_text = all_message_text(&supervisor_requests[7].messages);
        assert!(final_idle_text.contains(
            "w3 must be saved, spawned from, or discarded — it is currently none of these"
        ));
        let finalize_bounce_text = all_message_text(&supervisor_requests[11].messages);
        assert!(finalize_bounce_text.contains("error: before finalizing, name the evidence: the handles of the test runs you inspected (e.g. [\"w9m4\"]). If you have not seen test output for this checkpoint, spawn a verification worker on it first."));
    }

    #[tokio::test]
    async fn finalize_bounces_once_for_unabandoned_off_lineage_diffstat() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).expect("create repo dir");
        run_git(&repo, &["init", "--quiet"]);
        run_git(&repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(&repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("README.md"), "hello\n").expect("write README");
        run_git(&repo, &["add", "README.md"]);
        run_git(&repo, &["commit", "--quiet", "-m", "initial"]);
        let trace_path = temp.path().join("trace.jsonl");
        let _env_guard = crate::openrouter_auth::test_support::ENV_GUARD.lock().await;
        let _trace_env =
            crate::openrouter_auth::test_support::EnvScope::set("ANVIL_TRACE_JSONL", &trace_path);

        let backend = Arc::new(ScriptedAsgardBackend::new(vec![
            (
                "sv-model",
                vec![
                    text_response("intake literal"),
                    tool_response(vec![spawn_call("sv-spawn-w1", "root", "create a")]),
                    text_response("w1 launched"),
                    tool_response(vec![
                        save_call("sv-save-w1"),
                        spawn_call("sv-spawn-w2", "root", "create b"),
                    ]),
                    text_response("w2 launched"),
                    tool_response(vec![save_call("sv-save-w2")]),
                    tool_response(vec![prefinalize_call(
                        "sv-prefinalize-w3",
                        "w1",
                        "verify w1",
                    )]),
                    text_response("w3 launched"),
                    tool_response(vec![discard_call("sv-discard-w3")]),
                    tool_response(vec![finalize_call_with_evidence(
                        "sv-finalize-w1",
                        "w1",
                        &["w1m2"],
                    )]),
                    tool_response(vec![finalize_call_with_evidence_and_abandoned(
                        "sv-finalize-w1-abandoned",
                        "w1",
                        &["w1m2"],
                        &["w2"],
                    )]),
                ],
            ),
            (
                "worker-model",
                vec![
                    text_response("intake grounded"),
                    tool_response(vec![
                        write_file_call("w1-write", "a.txt", "a\n"),
                        named_tool_call(
                            "w1-test",
                            "run_shell_command",
                            serde_json::json!({ "command": "test -f a.txt" }),
                        ),
                    ]),
                    text_response("w1 done"),
                    tool_response(vec![write_file_call("w2-write", "b.txt", "b\n")]),
                    text_response("w2 done"),
                    text_response("w3 verification report"),
                ],
            ),
        ]));

        let (outcome, _) = run_scripted_asgard(
            repo.clone(),
            backend.clone(),
            vec![ChatMessage::user("exercise finalize abandoned")],
        )
        .await;

        assert_eq!(outcome.response, "w1 done");
        assert_eq!(fs::read_to_string(repo.join("a.txt")).unwrap(), "a\n");
        assert!(!repo.join("b.txt").exists());

        let requests = backend.requests.lock().expect("requests");
        let supervisor_text = requests
            .iter()
            .filter(|request| request.model == "sv-model")
            .map(|request| all_message_text(&request.messages))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(supervisor_text.contains("w2:"));
        assert!(supervisor_text.contains("b.txt"));
        assert!(supervisor_text.contains("Their work is absent from the delivered lineage. Merge them (merge_checkpoint), or list them in `abandoned` to confirm intentional abandonment."));

        let trace_records = fs::read_to_string(trace_path).expect("trace");
        let finalize = trace_records
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .find(|record| {
                record["type"] == "asgard_finalize"
                    && record["checkpoint"] == "w1"
                    && record["abandoned"] == serde_json::json!(["w2"])
                    && record["off_lineage_unmerged"]
                        .as_array()
                        .is_some_and(|entries| {
                            entries.iter().any(|entry| {
                                entry["checkpoint"] == "w2"
                                    && entry["diffstat"]
                                        .as_str()
                                        .is_some_and(|diffstat| diffstat.contains("b.txt"))
                            })
                        })
            })
            .expect("finalize trace");
        assert_eq!(finalize["abandoned"], serde_json::json!(["w2"]));
        let w2_entry = finalize["off_lineage_unmerged"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["checkpoint"] == "w2")
            .expect("w2 off-lineage entry");
        assert!(w2_entry["diffstat"].as_str().unwrap().contains("b.txt"));
    }

    #[tokio::test]
    async fn merge_checkpoint_then_finalize_delivers_both_sibling_changes() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).expect("create repo dir");
        run_git(&repo, &["init", "--quiet"]);
        run_git(&repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(&repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("README.md"), "hello\n").expect("write README");
        run_git(&repo, &["add", "README.md"]);
        run_git(&repo, &["commit", "--quiet", "-m", "initial"]);

        let backend = Arc::new(ScriptedAsgardBackend::new(vec![
            (
                "sv-model",
                vec![
                    text_response("intake literal"),
                    tool_response(vec![spawn_call("sv-spawn-w1", "root", "create a")]),
                    text_response("w1 launched"),
                    tool_response(vec![
                        save_call("sv-save-w1"),
                        spawn_call("sv-spawn-w2", "root", "create b"),
                    ]),
                    text_response("w2 launched"),
                    tool_response(vec![save_call("sv-save-w2")]),
                    tool_response(vec![merge_call("sv-merge", "w2", "w1")]),
                    tool_response(vec![prefinalize_call(
                        "sv-prefinalize-w4",
                        "w3",
                        "verify w3",
                    )]),
                    text_response("w4 launched"),
                    tool_response(vec![discard_call("sv-discard-w4")]),
                    tool_response(vec![finalize_call_with_evidence(
                        "sv-finalize-merged",
                        "w3",
                        &["w1m2"],
                    )]),
                ],
            ),
            (
                "worker-model",
                vec![
                    text_response("intake grounded"),
                    tool_response(vec![
                        write_file_call("w1-write", "a.txt", "a\n"),
                        named_tool_call(
                            "w1-test",
                            "run_shell_command",
                            serde_json::json!({ "command": "test -f a.txt" }),
                        ),
                    ]),
                    text_response("w1 done"),
                    tool_response(vec![write_file_call("w2-write", "b.txt", "b\n")]),
                    text_response("w2 done"),
                    text_response("w4 verification report"),
                ],
            ),
        ]));

        let (outcome, _) = run_scripted_asgard(
            repo.clone(),
            backend.clone(),
            vec![ChatMessage::user("exercise merge checkpoint")],
        )
        .await;

        assert!(matches!(outcome.stop, LoopStop::Completed { .. }));
        assert_eq!(fs::read_to_string(repo.join("a.txt")).unwrap(), "a\n");
        assert_eq!(fs::read_to_string(repo.join("b.txt")).unwrap(), "b\n");
        let status = run_git(&repo, &["status", "--porcelain"]);
        assert!(status.contains("a.txt"));
        assert!(status.contains("b.txt"));

        let requests = backend.requests.lock().expect("requests");
        let supervisor_text = requests
            .iter()
            .filter(|request| request.model == "sv-model")
            .map(|request| all_message_text(&request.messages))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(supervisor_text.contains("merged w2 onto w1 as w3"));
        assert!(supervisor_text.contains("b.txt"));
        assert!(supervisor_text.contains("diffstat: a.txt"));
    }

    #[test]
    fn merge_checkpoint_tool_noop_returns_warning_without_synthetic_node() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).expect("create repo dir");
        run_git(&repo, &["init", "--quiet"]);
        run_git(&repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(&repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("README.md"), "hello\n").expect("write README");
        run_git(&repo, &["add", "README.md"]);
        run_git(&repo, &["commit", "--quiet", "-m", "initial"]);
        let base_commit = run_git(&repo, &["rev-parse", "HEAD"]).trim().to_string();
        let stage = SnapshotStage::new(&repo, &format!("test-{}", uuid::Uuid::new_v4()))
            .expect("snapshot stage");

        let from_worker =
            crate::asgard::create_candidate_repository_at(&repo, "noop-from", &base_commit)
                .expect("from worker");
        fs::write(from_worker.root.join("from.txt"), "from\n").expect("write from");
        let from_commit = stage
            .snapshot(&from_worker.root, &base_commit, "from")
            .expect("from snapshot");

        let onto_worker =
            crate::asgard::create_candidate_repository_at(&repo, "noop-onto", &from_commit)
                .expect("onto worker");
        fs::write(onto_worker.root.join("onto.txt"), "onto\n").expect("write onto");
        let onto_commit = stage
            .snapshot(&onto_worker.root, &from_commit, "onto")
            .expect("onto snapshot");

        let mut dag = TrajectoryDag::new(Vec::new(), base_commit);
        dag.insert(saved_node(
            1,
            CheckpointId::Root,
            "from",
            from_commit.clone(),
        ))
        .expect("insert from");
        dag.insert(saved_node(2, CheckpointId::Worker(1), "onto", onto_commit))
            .expect("insert onto");

        let result = merge_checkpoint(
            &stage,
            &mut dag,
            CheckpointId::Worker(1),
            CheckpointId::Worker(2),
            3,
        )
        .expect("merge no-op");

        assert_eq!(
            result,
            "merged w1 onto w2: merge produced no changes; onto already contains this content"
        );
        assert!(!dag.contains(&CheckpointId::Worker(3)));
        let refs = run_git(
            &repo,
            &["for-each-ref", "--format=%(refname)", "refs/asgard"],
        );
        assert!(!refs.lines().any(|line| line.ends_with("/w3")));

        crate::asgard::remove_candidate_repository(&from_worker);
        crate::asgard::remove_candidate_repository(&onto_worker);
        stage.cleanup();
    }

    #[tokio::test]
    async fn failed_merge_checkpoint_then_finalize_delivers_target_tree() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).expect("create repo dir");
        run_git(&repo, &["init", "--quiet"]);
        run_git(&repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(&repo, &["config", "user.name", "Asgard Test"]);
        fs::write(
            repo.join("same.txt"),
            "line 1\nshared line\nline 3\nline 4\n",
        )
        .expect("write same");
        run_git(&repo, &["add", "same.txt"]);
        run_git(&repo, &["commit", "--quiet", "-m", "initial"]);

        let backend = Arc::new(ScriptedAsgardBackend::new(vec![
            (
                "sv-model",
                vec![
                    text_response("intake literal"),
                    tool_response(vec![spawn_call("sv-spawn-w1", "root", "create A")]),
                    text_response("w1 launched"),
                    tool_response(vec![
                        save_call("sv-save-w1"),
                        spawn_call("sv-spawn-w2", "root", "create B"),
                    ]),
                    text_response("w2 launched"),
                    tool_response(vec![
                        save_call("sv-save-w2"),
                        spawn_call("sv-spawn-w3", "w1", "create target"),
                    ]),
                    text_response("w3 launched"),
                    tool_response(vec![save_call("sv-save-w3")]),
                    tool_response(vec![merge_call("sv-merge-conflict", "w2", "w3")]),
                    tool_response(vec![prefinalize_call(
                        "sv-prefinalize-w4",
                        "w3",
                        "verify w3",
                    )]),
                    text_response("w4 launched"),
                    tool_response(vec![discard_call("sv-discard-w4")]),
                    tool_response(vec![finalize_call_with_evidence_and_abandoned(
                        "sv-finalize-w3",
                        "w3",
                        &["w3m3"],
                        &["w2"],
                    )]),
                ],
            ),
            (
                "worker-model",
                vec![
                    text_response("intake grounded"),
                    tool_response(vec![write_file_call(
                        "w1-write",
                        "same.txt",
                        "line 1\nA edits the shared line\nline 3\nline 4\n",
                    )]),
                    text_response("w1 done"),
                    tool_response(vec![write_file_call(
                        "w2-write",
                        "same.txt",
                        "line 1\nB edits the shared line\nline 3\nline 4\n",
                    )]),
                    text_response("w2 done"),
                    tool_response(vec![
                        write_file_call(
                            "w3-write-same",
                            "same.txt",
                            "line 1\nT keeps A's shared-line choice\nline 3\nline 4\n",
                        ),
                        write_file_call(
                            "w3-write-target",
                            "target.txt",
                            "target content line 1\n\
                             target content line 2\n\
                             target content line 3\n\
                             target content line 4\n\
                             target content line 5\n",
                        ),
                        named_tool_call(
                            "w3-test",
                            "run_shell_command",
                            serde_json::json!({ "command": "test -f target.txt" }),
                        ),
                    ]),
                    text_response("w3 done"),
                    tool_response(vec![named_tool_call(
                        "w4-test",
                        "run_shell_command",
                        serde_json::json!({ "command": "test -f target.txt" }),
                    )]),
                    text_response("w4 verification report"),
                ],
            ),
        ]));

        let (outcome, _) = run_scripted_asgard(
            repo.clone(),
            backend.clone(),
            vec![ChatMessage::user("exercise failed merge then finalize")],
        )
        .await;

        assert!(matches!(outcome.stop, LoopStop::Completed { .. }));
        assert_eq!(
            fs::read_to_string(repo.join("same.txt")).unwrap(),
            "line 1\nT keeps A's shared-line choice\nline 3\nline 4\n"
        );
        assert_eq!(
            fs::read_to_string(repo.join("target.txt")).unwrap(),
            "target content line 1\n\
             target content line 2\n\
             target content line 3\n\
             target content line 4\n\
             target content line 5\n"
        );

        let requests = backend.requests.lock().expect("requests");
        let supervisor_text = requests
            .iter()
            .filter(|request| request.model == "sv-model")
            .map(|request| all_message_text(&request.messages))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(supervisor_text.contains("error: merge_checkpoint"));
        assert!(supervisor_text.contains("same.txt"));
        assert!(supervisor_text.contains("conflicting hunks"));
        assert!(supervisor_text.contains("spawn a worker from the onto checkpoint"));
    }

    #[tokio::test]
    async fn mixed_view_and_spawn_response_executes_and_elides_view_in_permanent() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).expect("create repo dir");
        run_git(&repo, &["init", "--quiet"]);
        run_git(&repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(&repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("README.md"), "hello\n").expect("write README");
        fs::write(
            repo.join("payload.txt"),
            format!("MIXED_VIEW_PAYLOAD_123 {}\n", "x".repeat(600)),
        )
        .expect("write payload");
        run_git(&repo, &["add", "README.md", "payload.txt"]);
        run_git(&repo, &["commit", "--quiet", "-m", "initial"]);

        let backend = Arc::new(ScriptedAsgardBackend::new(vec![
            (
                "sv-model",
                vec![
                    text_response("intake literal"),
                    tool_response(vec![spawn_call("sv-spawn-w1", "root", "produce output")]),
                    text_response("w1 launched"),
                    tool_response(vec![
                        view_call("sv-view-w1", &["w1m1"]),
                        spawn_call("sv-spawn-w2", "w1", "continue after viewing"),
                    ]),
                    tool_response(vec![discard_call("sv-discard-w2")]),
                    text_response("mixed call accepted"),
                    tool_response(vec![prefinalize_call(
                        "sv-prefinalize-w3",
                        "w1",
                        "verify w1",
                    )]),
                    text_response("w3 launched"),
                    tool_response(vec![discard_call("sv-discard-w3")]),
                    text_response("idle one"),
                    text_response("idle two"),
                    text_response("idle three"),
                    text_response("idle four"),
                    text_response("idle five"),
                    text_response("idle six"),
                    text_response("idle seven"),
                ],
            ),
            (
                "worker-model",
                vec![
                    text_response("intake grounded"),
                    // read_file keeps this fork-free (shell forks fail under
                    // per-uid NPROC pressure on busy hosts) while producing a
                    // result large enough to cross the elision threshold.
                    tool_response(vec![
                        named_tool_call(
                            "w1-read",
                            "read_file",
                            serde_json::json!({ "file_path": "payload.txt" }),
                        ),
                        named_tool_call(
                            "w1-test",
                            "run_shell_command",
                            serde_json::json!({ "command": "true" }),
                        ),
                    ]),
                    text_response("w1 done"),
                    text_response("w2 done"),
                    text_response("w3 verification report"),
                ],
            ),
        ]));
        let llm: Arc<dyn crate::llm_client::LlmBackend> = backend.clone();
        let sessions = SessionStore::new("worker-model".to_string());
        let session = sessions.create_session(repo.clone()).await;
        assert!(
            sessions
                .set_permission_mode(&session.id, PermissionMode::BypassPermissions)
                .await
        );
        let parent_registry = sessions
            .get_or_create_registry(&session.id, repo.clone())
            .await
            .expect("parent registry");
        let config = Config {
            candidate_models: vec!["worker-model".to_string()],
            supervisor_model: Some("sv-model".to_string()),
        };
        let initial_messages = vec![ChatMessage::user("exercise mixed supervisor calls")];
        let (agent_io, mut client_io) = tokio::io::duplex(1 << 20);
        let (agent_read, agent_write) = tokio::io::split(agent_io);
        let drain = tokio::spawn(async move {
            let _ = tokio::io::copy(&mut client_io, &mut tokio::io::sink()).await;
        });

        let (outcome, _) = Agent
            .builder()
            .on_receive_dispatch(
                async move |message: Dispatch, _cx| {
                    Ok(Handled::No {
                        message,
                        retry: false,
                    })
                },
                on_receive_dispatch!(),
            )
            .connect_with(
                ByteStreams::new(agent_write.compat_write(), agent_read.compat()),
                async |cx| {
                    Ok(run_asgard_trajectory_loop(
                        &cx,
                        &sessions,
                        &session.id,
                        &llm,
                        &parent_registry,
                        "worker-model",
                        None,
                        None,
                        None,
                        initial_messages,
                        IdleTimeouts::uniform(Duration::from_secs(30)),
                        tokio_util::sync::CancellationToken::new(),
                        &config,
                        None,
                        0,
                        None,
                    )
                    .await)
                },
            )
            .await
            .expect("in-memory ACP connect_with");
        drain.abort();

        assert_eq!(outcome.response, "w1 done");
        let requests = backend.requests.lock().expect("requests");
        let supervisor_requests = requests
            .iter()
            .filter(|request| {
                request.model == "sv-model"
                    && request
                        .tool_names
                        .iter()
                        .any(|name| name == crate::asgard::SPAWN_WORKERS_TOOL)
            })
            .collect::<Vec<_>>();
        let supervisor_text = supervisor_requests
            .iter()
            .map(|request| all_message_text(&request.messages))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!supervisor_text.contains("MIXED_VIEW_PAYLOAD_123"));
        assert!(supervisor_text.contains("[viewed w1"));
        assert!(supervisor_text.contains("spawned w2 from w1"));
    }

    #[tokio::test]
    async fn wait_turn_records_results_reminds_pending_and_allows_evidence_finalize() {
        let temp = tempfile::tempdir().expect("tempdir");
        let _env_guard = crate::openrouter_auth::test_support::ENV_GUARD.lock().await;
        let trace_path = temp.path().join("trace.jsonl");
        let _trace_env =
            crate::openrouter_auth::test_support::EnvScope::set("ANVIL_TRACE_JSONL", &trace_path);

        let repo = temp.path().join("repo");
        fs::create_dir(&repo).expect("create repo dir");
        run_git(&repo, &["init", "--quiet"]);
        run_git(&repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(&repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("README.md"), "hello\n").expect("write README");
        run_git(&repo, &["add", "README.md"]);
        run_git(&repo, &["commit", "--quiet", "-m", "initial"]);

        let backend = Arc::new(ScriptedAsgardBackend::new(vec![
            (
                "sv-model",
                vec![
                    text_response("intake literal"),
                    tool_response(vec![
                        wait_call("sv-wait-clean"),
                        spawn_call("sv-spawn-after-wait", "root", "must not run"),
                    ]),
                    tool_response(vec![spawn_call(
                        "sv-spawn-w1",
                        "root",
                        "Run pwd and report",
                    )]),
                    text_response("w1 launched"),
                    tool_response(vec![wait_call("sv-wait-pending")]),
                    tool_response(vec![wait_call("sv-wait-pending-again")]),
                    tool_response(vec![prefinalize_call(
                        "sv-prefinalize-w2",
                        "w1",
                        "verify w1",
                    )]),
                    text_response("w2 launched"),
                    tool_response(vec![
                        discard_call("sv-discard-w2"),
                        finalize_call_with_evidence("sv-finalize-evidence", "w1", &["w1m1"]),
                    ]),
                ],
            ),
            (
                "worker-model",
                vec![
                    text_response("intake grounded"),
                    tool_response(vec![named_tool_call(
                        "w1-pwd",
                        "run_shell_command",
                        serde_json::json!({ "command": "pwd" }),
                    )]),
                    text_response("w1 final report"),
                    text_response("w2 verification report"),
                ],
            ),
        ]));
        let llm: Arc<dyn crate::llm_client::LlmBackend> = backend.clone();
        let sessions = SessionStore::new("worker-model".to_string());
        let session = sessions.create_session(repo.clone()).await;
        assert!(
            sessions
                .set_permission_mode(&session.id, PermissionMode::BypassPermissions)
                .await
        );
        let parent_registry = sessions
            .get_or_create_registry(&session.id, repo.clone())
            .await
            .expect("parent registry");
        let config = Config {
            candidate_models: vec!["worker-model".to_string()],
            supervisor_model: Some("sv-model".to_string()),
        };
        let (agent_io, mut client_io) = tokio::io::duplex(1 << 20);
        let (agent_read, agent_write) = tokio::io::split(agent_io);
        let drain = tokio::spawn(async move {
            let _ = tokio::io::copy(&mut client_io, &mut tokio::io::sink()).await;
        });

        let (outcome, _) = Agent
            .builder()
            .on_receive_dispatch(
                async move |message: Dispatch, _cx| {
                    Ok(Handled::No {
                        message,
                        retry: false,
                    })
                },
                on_receive_dispatch!(),
            )
            .connect_with(
                ByteStreams::new(agent_write.compat_write(), agent_read.compat()),
                async |cx| {
                    Ok(run_asgard_trajectory_loop(
                        &cx,
                        &sessions,
                        &session.id,
                        &llm,
                        &parent_registry,
                        "worker-model",
                        None,
                        None,
                        None,
                        vec![ChatMessage::user("exercise wait and evidence finalize")],
                        IdleTimeouts::uniform(Duration::from_secs(30)),
                        tokio_util::sync::CancellationToken::new(),
                        &config,
                        None,
                        0,
                        None,
                    )
                    .await)
                },
            )
            .await
            .expect("in-memory ACP connect_with");
        drain.abort();

        assert_eq!(outcome.response, "w1 final report");
        let requests = backend.requests.lock().expect("requests");
        let supervisor_requests = requests
            .iter()
            .filter(|request| {
                request.model == "sv-model"
                    && request
                        .tool_names
                        .iter()
                        .any(|name| name == crate::asgard::SPAWN_WORKERS_TOOL)
            })
            .collect::<Vec<_>>();
        assert_eq!(supervisor_requests.len(), 8);

        let after_clean_wait = all_message_text(&supervisor_requests[1].messages);
        assert!(after_clean_wait.contains("waiting for the next worker"));
        assert!(after_clean_wait.contains("error: turn already ended"));
        assert!(
            !after_clean_wait.contains("sv-spawn-after-wait"),
            "the blocked call id should not leak into content text"
        );

        let after_pending_wait = all_message_text(&supervisor_requests[4].messages);
        assert!(after_pending_wait.contains("waiting for the next worker"));
        assert!(after_pending_wait.contains(
            "w1 must be saved, spawned from, or discarded — it is currently none of these"
        ));

        let final_request_text = all_message_text(&supervisor_requests[5].messages);
        assert!(
            final_request_text.contains("w1 was auto-saved: the turn ended without resolving it")
        );
        assert!(!final_request_text.contains("error: before finalizing"));

        let trace = fs::read_to_string(&trace_path).expect("trace jsonl");
        let finalize_record = trace
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("trace json"))
            .find(|record| {
                record["type"] == "asgard_finalize"
                    && record["evidence"] == serde_json::json!(["w1m1"])
            })
            .expect("asgard finalize trace");
        assert_eq!(finalize_record["evidence"], serde_json::json!(["w1m1"]));
        assert_eq!(
            finalize_record["prefinalize_workers"],
            serde_json::json!([2])
        );
    }

    #[tokio::test]
    async fn supervisor_step_cap_auto_saves_unresolved_pending() {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).expect("create repo dir");
        run_git(&repo, &["init", "--quiet"]);
        run_git(&repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(&repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("README.md"), "hello\n").expect("write README");
        run_git(&repo, &["add", "README.md"]);
        run_git(&repo, &["commit", "--quiet", "-m", "initial"]);

        let mut supervisor_responses = vec![
            text_response("intake literal"),
            tool_response(vec![spawn_call(
                "sv-spawn-w1",
                "root",
                "finish without changes",
            )]),
            text_response("w1 launched"),
        ];
        for index in 0..ASGARD_SUPERVISOR_MAX_STEPS {
            supervisor_responses.push(tool_response(vec![update_plan_call(
                &format!("sv-plan-{index}"),
                "Spin",
                "in_progress",
            )]));
        }
        supervisor_responses.extend([
            text_response("idle one"),
            text_response("idle two"),
            text_response("idle three"),
        ]);

        let backend = Arc::new(ScriptedAsgardBackend::new(vec![
            ("sv-model", supervisor_responses),
            (
                "worker-model",
                vec![
                    text_response("intake grounded"),
                    text_response("w1 final report"),
                ],
            ),
        ]));
        let llm: Arc<dyn crate::llm_client::LlmBackend> = backend.clone();
        let sessions = SessionStore::new("worker-model".to_string());
        let session = sessions.create_session(repo.clone()).await;
        assert!(
            sessions
                .set_permission_mode(&session.id, PermissionMode::BypassPermissions)
                .await
        );
        let parent_registry = sessions
            .get_or_create_registry(&session.id, repo.clone())
            .await
            .expect("parent registry");
        let config = Config {
            candidate_models: vec!["worker-model".to_string()],
            supervisor_model: Some("sv-model".to_string()),
        };
        let (agent_io, mut client_io) = tokio::io::duplex(1 << 20);
        let (agent_read, agent_write) = tokio::io::split(agent_io);
        let drain = tokio::spawn(async move {
            let _ = tokio::io::copy(&mut client_io, &mut tokio::io::sink()).await;
        });

        let (outcome, _) = Agent
            .builder()
            .on_receive_dispatch(
                async move |message: Dispatch, _cx| {
                    Ok(Handled::No {
                        message,
                        retry: false,
                    })
                },
                on_receive_dispatch!(),
            )
            .connect_with(
                ByteStreams::new(agent_write.compat_write(), agent_read.compat()),
                async |cx| {
                    Ok(run_asgard_trajectory_loop(
                        &cx,
                        &sessions,
                        &session.id,
                        &llm,
                        &parent_registry,
                        "worker-model",
                        None,
                        None,
                        None,
                        vec![ChatMessage::user("exercise supervisor step cap")],
                        IdleTimeouts::uniform(Duration::from_secs(30)),
                        tokio_util::sync::CancellationToken::new(),
                        &config,
                        None,
                        0,
                        None,
                    )
                    .await)
                },
            )
            .await
            .expect("in-memory ACP connect_with");
        drain.abort();

        assert!(matches!(
            outcome.stop,
            LoopStop::Completed { had_text: true }
        ));
        assert_eq!(outcome.response, "w1 final report");
        let requests = backend.requests.lock().expect("requests");
        let supervisor_requests = requests
            .iter()
            .filter(|request| {
                request.model == "sv-model"
                    && request
                        .tool_names
                        .iter()
                        .any(|name| name == crate::asgard::SPAWN_WORKERS_TOOL)
            })
            .collect::<Vec<_>>();
        assert_eq!(supervisor_requests.len(), ASGARD_SUPERVISOR_MAX_STEPS + 5);
        let first_idle_after_cap =
            all_message_text(&supervisor_requests[ASGARD_SUPERVISOR_MAX_STEPS + 2].messages);
        assert!(
            first_idle_after_cap.contains("w1 was auto-saved: the turn ended without resolving it")
        );
        assert!(first_idle_after_cap.contains("w1 \"finish without changes\" saved"));
    }

    #[test]
    fn supervisor_system_message_splices_before_tools_and_appends_supplement() {
        let session_system = "Intro\n# How you work\nDo work\n# Verification\nRun tests\n# Tools\nshell docs\n# Output\nskills catalog text";

        let result = supervisor_system_message(Some(session_system));

        assert!(result.contains("# Verification"));
        assert!(result.contains("# Asgard supervision"));
        assert!(!result.contains("shell docs"));
        assert!(!result.contains("# Output"));
        let verification_pos = result.find("# Verification").expect("verification");
        let supplement_pos = result.find("# Asgard supervision").expect("supplement");
        assert!(
            verification_pos < supplement_pos,
            "supplement should follow truncated session head:\n{result}"
        );
    }

    #[test]
    fn supervisor_system_message_retains_full_prompt_without_marker_and_handles_none() {
        let session_system = "Intro\n# How you work\nDo work\n# Verification\nRun tests";

        let result = supervisor_system_message(Some(session_system));

        assert!(result.starts_with(session_system));
        assert!(result.contains("# Asgard supervision"));
        assert_eq!(supervisor_system_message(None), supervisor_supplement());
    }

    #[test]
    fn asgard_original_task_uses_last_user_message() {
        let messages = vec![
            ChatMessage::system("system"),
            ChatMessage::user("first"),
            ChatMessage::assistant("middle"),
            ChatMessage::user("last"),
        ];
        assert_eq!(asgard_original_task(&messages), "last");
    }

    #[test]
    fn asgard_original_task_is_empty_without_user_message() {
        assert_eq!(asgard_original_task(&[ChatMessage::system("system")]), "");
    }

    #[test]
    fn asgard_take_window_messages_returns_tail_from_start() {
        let messages = vec![
            ChatMessage::user("before"),
            ChatMessage::assistant("one"),
            ChatMessage::assistant("two"),
        ];
        let window = asgard_take_window_messages(&messages, 1);
        assert_eq!(
            window
                .iter()
                .map(ChatMessage::content_text)
                .collect::<Vec<_>>(),
            vec!["one".to_string(), "two".to_string()]
        );
    }

    #[test]
    fn asgard_take_window_messages_handles_out_of_range_start() {
        assert!(asgard_take_window_messages(&[ChatMessage::user("x")], 9).is_empty());
    }

    #[test]
    fn rewrite_asgard_cwd_rewrites_text_reasoning_and_tool_arguments() {
        let from = Path::new("/parent/repo");
        let to = Path::new("/tmp/asgard/repo");
        let mut assistant = ChatMessage::assistant("read /parent/repo/src/lib.rs");
        assistant.reasoning_content = Some("thinking in /parent/repo".to_string());
        let mut messages = vec![
            ChatMessage::user("work in /parent/repo"),
            assistant,
            ChatMessage::assistant_tool_calls(vec![ToolCall {
                id: "call-1".to_string(),
                r#type: "function".to_string(),
                function: FunctionCall {
                    name: "run_shell_command".to_string(),
                    arguments: r#"{"command":"cd /parent/repo && cargo test"}"#.to_string(),
                },
            }]),
        ];

        rewrite_asgard_cwd(&mut messages, from, to);

        let rendered = serde_json::to_string(&messages).unwrap();
        assert!(rendered.contains("/tmp/asgard/repo"));
        assert!(!rendered.contains("/parent/repo"));
    }

    #[test]
    fn rewrite_asgard_cwd_canonicalizes_trailing_separator_spellings() {
        // A session rooted at the repository toplevel carries a trailing
        // separator in its cwd; tool results and model-authored commands
        // spell the same path without it. Both must rewrite.
        let mut messages = vec![ChatMessage::user("ls /parent/repo/src")];
        rewrite_asgard_cwd(
            &mut messages,
            Path::new("/parent/repo/"),
            Path::new("/tmp/asgard/repo/"),
        );
        let rendered = serde_json::to_string(&messages).unwrap();
        assert!(rendered.contains("/tmp/asgard/repo/src"));
        assert!(!rendered.contains("/parent/repo"));
    }

    #[test]
    fn stop_reason_mapping_is_exhaustive() {
        assert_eq!(
            worker_stop_reason(&LoopStop::Completed { had_text: true }),
            WorkerStopReason::Finished
        );
        assert_eq!(
            worker_stop_reason(&LoopStop::MaxTurns { max_turns: 10 }),
            WorkerStopReason::StepLimit
        );
        assert_eq!(
            worker_stop_reason(&LoopStop::Cancelled),
            WorkerStopReason::Cancelled
        );
        assert_eq!(
            worker_stop_reason(&LoopStop::Failed(crate::tool_loop::TurnFailure {
                retryable: false,
                message: "boom".to_string(),
            })),
            WorkerStopReason::Failed("boom".to_string())
        );
    }

    #[test]
    fn final_response_extraction_uses_last_assistant_without_tool_calls() {
        let messages = vec![
            ChatMessage::assistant("earlier"),
            ChatMessage::assistant_tool_calls(vec![tool_call("call-1")]),
            ChatMessage::tool_result("call-1", "run_shell_command", "ok"),
            ChatMessage::assistant("final report"),
        ];
        assert_eq!(extract_worker_final_response(&messages), "final report");
    }

    #[test]
    fn final_response_extraction_is_empty_without_plain_assistant_message() {
        let messages = vec![ChatMessage::assistant_tool_calls(vec![tool_call("call-1")])];
        assert_eq!(extract_worker_final_response(&messages), "");
    }

    #[test]
    fn worker_step_count_counts_assistant_tool_batches() {
        let messages = vec![
            ChatMessage::assistant_tool_calls(vec![tool_call("call-1")]),
            ChatMessage::tool_result("call-1", "run_shell_command", "ok"),
            ChatMessage::assistant("plain"),
            ChatMessage::assistant_tool_calls(vec![tool_call("call-2"), tool_call("call-3")]),
        ];
        assert_eq!(count_worker_steps(&messages), 2);
    }

    #[test]
    fn status_block_renders_running_queue_checkpoints_and_capacity() {
        let mut dag = TrajectoryDag::new(Vec::new(), "base".to_string());
        dag.insert(TrajectoryNode {
            window: TrajectoryWindow {
                worker: 2,
                parent: CheckpointId::Root,
                instructions: "saved checkpoint".to_string(),
                model: "model-a".to_string(),
                instruction_message: ChatMessage::user("saved worker instructions"),
                window_messages: Vec::new(),
                compact: String::new(),
                final_response: String::new(),
                stop: WorkerStopReason::Finished,
                steps: 1,
                diffstat: String::new(),
                usage: TokenUsage::default(),
                elapsed_millis: 0,
            },
            commit: "c2".to_string(),
            merged_from: Vec::new(),
        })
        .unwrap();
        let progress = Arc::new(AtomicUsize::new(3));
        let mut running = HashMap::new();
        running.insert(
            5,
            RunningWorkerMeta {
                parent: CheckpointId::Worker(2),
                model: "worker-model".to_string(),
                instructions: "inspect the parser\nthen test it".to_string(),
                turn_progress: progress,
            },
        );
        let review_queue = VecDeque::new();

        let rendered = render_asgard_status_block(
            &dag,
            &running,
            &review_queue,
            None,
            4,
            Some("No worker is awaiting review. Spawn workers or finalize."),
        );

        assert!(rendered.contains("<asgard_status>"));
        assert!(rendered.contains("<dag>\nroot\n"));
        assert!(rendered.contains("w2 \"saved checkpoint\" saved, finished/1 steps"));
        assert!(
            rendered.contains("└─ w5 \"inspect the parser then test it\" in flight, step 3/10")
        );
        assert!(rendered.contains("capacity_available: 4"));
        assert!(rendered.contains("Spawn workers or finalize."));
    }
}
