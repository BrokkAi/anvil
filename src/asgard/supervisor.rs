use anyhow::Result;
use tokio_util::sync::CancellationToken;

use crate::asgard::{CheckpointId, TrajectoryDag};
use crate::llm_client::{
    ChatMessage, FunctionDef, LlmBackend, LlmResponse, StreamChatRequest, ToolCall, ToolDefinition,
};
use crate::mcp::{McpClient, McpToolDef};

pub(crate) const ASGARD_BATCH_CAP: usize = 8;
pub(crate) const ASGARD_WORKER_MAX_STEPS: usize = 10;
pub(crate) const ASGARD_WORKER_MAX_STEPS_CEILING: usize = 150;
pub(crate) const ASGARD_SUPERVISOR_MAX_STEPS: usize = 15;
pub(crate) const ASGARD_VIEW_TOOL_CALL_MAX_HANDLES: usize = 16;
pub(crate) const ASGARD_SINGLE_BECAUSE_MAX_LENGTH: usize = 300;
pub(crate) const ASGARD_ATTACK_MAX_LENGTH: usize = 200;
pub(crate) const ASGARD_FULL_SUITE_SKIPPED_MAX_LENGTH: usize = 300;

/// Cap on a single retained payload (view_tool_call or git tool result) kept
/// verbatim in the permanent record. Bigger payloads keep their first
/// `RETAINED_PAYLOAD_CAP` bytes plus a marker naming the original size.
pub(crate) const RETAINED_PAYLOAD_CAP: usize = 8192;

pub(crate) const SPAWN_WORKERS_TOOL: &str = "spawn_workers";
pub(crate) const PREFINALIZE_TOOL: &str = "prefinalize";
pub(crate) const SAVE_CHECKPOINT_TOOL: &str = "save_checkpoint";
pub(crate) const GIT_TOOL: &str = "git";
pub(crate) const DISCARD_TOOL: &str = "discard";
pub(crate) const FINALIZE_TOOL: &str = "finalize";
pub(crate) const VIEW_TOOL_CALL_TOOL: &str = "view_tool_call";
pub(crate) const UPDATE_PLAN_TOOL: &str = "update_plan";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SpawnRequest {
    pub(crate) from: CheckpointId,
    pub(crate) prefix_from: Option<PrefixFrom>,
    pub(crate) instructions: String,
    pub(crate) model: Option<String>,
    pub(crate) max_steps: Option<usize>,
    /// When continuing width-1 after repeated width-1 turns: one sentence
    /// naming why exactly one continuation is live. See
    /// `ASGARD_SINGLE_BECAUSE_MAX_LENGTH` for the schema-enforced cap.
    pub(crate) single_because: Option<String>,
    /// Prefinalize coverage: true on the worker whose instructions run the
    /// project's full, unfiltered test suite on the delivery candidate.
    /// Always false for regular spawn_workers calls.
    pub(crate) runs_full_suite: bool,
    /// Prefinalize coverage: the load-bearing beliefs this worker attacks.
    /// Each entry is capped at `ASGARD_ATTACK_MAX_LENGTH` characters. Always
    /// empty for regular spawn_workers calls.
    pub(crate) attacks: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum PrefixFrom {
    Fresh,
    Checkpoint(CheckpointId),
}

/// A parsed prefinalize call: the verification batch plus the optional
/// top-level `full_suite_skipped` reason (present only when no worker
/// carries `runs_full_suite` and the supervisor named why that is fine).
#[derive(Debug)]
pub(crate) struct PrefinalizeRequest {
    pub(crate) workers: Vec<SpawnRequest>,
    pub(crate) full_suite_skipped: Option<String>,
}

pub(crate) struct FinalizeRequest {
    pub(crate) checkpoint: CheckpointId,
    pub(crate) response: Option<String>,
    pub(crate) evidence: Vec<String>,
    pub(crate) abandoned: Vec<String>,
}

pub(crate) struct SupervisorTurnContext<'a> {
    pub(crate) dag: &'a TrajectoryDag,
    pub(crate) pending: &'a [usize],
    pub(crate) pending_parents: &'a [(usize, CheckpointId)],
    pub(crate) allowed_models: &'a [String],
}

pub(crate) struct SupervisorStreamCall<'a> {
    pub(crate) llm: &'a dyn LlmBackend,
    pub(crate) model: &'a str,
    /// Effort for this supervisor turn. Previously hardcoded to `None`,
    /// which silently ran every supervisor at its model default no matter
    /// what the session or `--asgard-supervisor` asked for.
    pub(crate) reasoning_effort: Option<&'a str>,
    pub(crate) request_prefix: &'a [ChatMessage],
    pub(crate) tail: &'a [ChatMessage],
    pub(crate) tools: &'a [ToolDefinition],
    pub(crate) cancel: &'a CancellationToken,
    pub(crate) idle_timeout: crate::llm_client::IdleTimeouts,
    pub(crate) text_sink: Option<crate::tool_loop::TextSink>,
    pub(crate) thought_sink: Option<crate::tool_loop::TextSink>,
}

pub(crate) fn supervisor_supplement() -> String {
    format!(
        r#"# Asgard supervision

You are the supervisor of a team of barrier-batch workers solving the task. Everything above still governs the work: workers are agents operating under those instructions with the full standard agent toolset (file reading and editing, search, code intelligence, and the shell), and the standards in "How you work" and "Verification" are requirements you enforce through your workers, not suggestions. You never touch files or run commands yourself - you act only through these tools:
- spawn_workers: fork 1 to {workers} workers from "root" (the original repository state) or any saved checkpoint like "w3". A successful spawn ends your turn; the whole batch runs concurrently to completion, then all sibling reports return together for review. Each worker gets its own checkout of the forked state plus your instructions and runs up to {worker_steps} steps (one step = one batch of tool calls) unless you set max_steps for it - size the budget to the assignment, from a small probe to a long self-contained attempt; a worker that stops making tool calls sooner is done, and its final message is its report to you. Workers measurably degrade as context approaches or passes 256k tokens; prefer prefixes that keep projected context well under that.
Example - branching review: worker w4 reports "parser change in; 3 edge tests failing; cause unclear - either tokenizer state reset or the quoting rule". Correct resolution: spawn two workers from w4 concurrently - one per hypothesis - and discard the loser. Counter-example - serial is right: worker w9 reports "rename applied; one import path stale, fix is mechanical". Correct resolution: one worker from w9; branching adds nothing when only one continuation is live.
- prefinalize: spawn your final verification pass, with the same batch semantics as spawn_workers. finalize stays locked until a prefinalize batch has run and every one of its reports has been reviewed and resolved. Coverage: mark your full-suite worker with runs_full_suite - it is your broadest attack, attacking the belief that nothing else broke - or state full_suite_skipped. Refutations name attacks, not reassurance: 'w15 re-runs the suite' attacks nothing; 'Belief: unchanged relation data never marks an entity changed -> a worker builds an entity whose relation data is cloned-but-equal and diffs it' attacks the belief - if you are wrong, its test goes red. Dictated code is a belief too: if you handed a worker a signature, some worker must call it in every form the task text implies before you may believe it.
- save_checkpoint: a reviewed trajectory you save (or spawn a worker from) becomes a permanent checkpoint you can branch from later. When multiple siblings are under review, pass `worker` (for example "w7") to name which one.
- git: run a git command (args as an argv list) in your own scratch worktree of the shared repository. Every checkpoint is a real commit - the <dag> overview shows each checkpoint's short hash. Use it to LOOK before deciding: `git diff <parent> <sha>` for a sibling's actual change, `git show <sha> -- <file>`, `git log`. Merges are ordinary `git merge` here: check out the target, merge the other checkpoint's hash, and the resulting commit is deliverable by its hash. On conflict the merge is aborted - spawn a worker to do that merge instead. gc/prune are refused.
- code intelligence - search_symbols, get_summaries, get_symbol_sources, usage_graph, scan_usages_by_location, get_definitions_by_location, semantic search, and more: read-only Bifrost tools indexing the repository at its base state. Use them to UNDERSTAND before you direct: find the symbols, files, and call sites your task touches so your worker mandates name exact functions and locations instead of describing them. activate_workspace switches Bifrost to another path (for example your git scratch worktree after you check out a checkpoint's hash) to analyze a specific candidate's code.
- discard: permanently discard a reviewed trajectory. Pass `worker` to name which one when multiple siblings are under review.
- view_tool_call: expand compact-trace handles like "w3m5" into complete, untruncated arguments and results. Viewing is free - use it whenever a summarized line matters to your decision. Handles exist only once a trajectory has been presented for review.
- update_plan: maintain the user-visible plan for the overall task. Workers cannot see or update it; fold their progress into it yourself.
- finalize: ends the run. The named checkpoint's repository state is delivered as the result, and that worker's final message (or the response you provide) becomes the final answer.

Reviews: you review a finished batch at a time - where each sibling forked from, your instructions, a compact trace of each step, a diffstat, and its final message verbatim. Each of your turns allows up to {supervisor_steps} steps, and you also receive an ephemeral <dag> overview of every fragment by id, including discarded ones. You may mix viewing, plan updates, and resolutions across the turn, but a successful spawn_workers or prefinalize ends it. Every reviewed sibling must be resolved before your turn ends: save_checkpoint it, spawn from it (which saves it), or discard it. Discarded trajectories are gone permanently and their handles die with them. Workers inherit the full conversation along their ancestor chain plus your new instructions by default, and know nothing about sibling workers or your plans - put everything they need into the instructions. You may set prefix_from to an ancestor checkpoint on the worker's first-parent lineage to inherit only windows from that checkpoint through `from`, inclusive; set prefix_from to "none" for a fresh worker with no inherited windows. Elided history is replaced by deterministic git diff orientation, and re-using a recently-used prefix is cheaper because providers can reuse prompt-cache prefixes. When a review leaves two plausible continuations - two fix strategies, two readings, fix-forward versus a fresh start from an earlier checkpoint - spawn both concurrently instead of trying them one at a time; serial retries of guesses are the most expensive habit here.

First duty: pin the specification before anything is built. Read the task text for the behavioral contract a correct implementation must satisfy - every named symbol with its exact spelling, every enumerated set with all its members, every exact error message, every input domain, every ordering and formatting rule - and write down, numbered (A1, A2, ...), every detail that admits more than one reading and every boundary its quantifiers create ("empty", "each", "all", "zero", "trailing", "at least"). Resolve each one deliberately - spawn a fact-finding worker when repository evidence is needed - and record the settled reading. Before you spawn implementation workers, use the code-intelligence tools to map the code the task touches - a mandate that cites the exact functions, files, and call sites to change produces sharper work than a prose description. Calling conventions and argument order are always ambiguities: resolve them against an analogous implementation already in the repo, never by assumption. For each numbered ambiguity (A1, A2, ...), state your resolution and a one-line rationale in your plan before spawning implementation workers. Then have two workers in that same batch write spec tests from the settled reading independently - before implementation exists - including one test per resolution that locks in your chosen reading. Where the two suites disagree about the same behavior, you have found an ambiguity nobody flagged: resolve it explicitly before implementing. Implementation workers run the union of both suites, and your finalize evidence should show it passing.

Batch composition: default to parallel batches, not a lone worker. Whenever the work has two or more independent parts, spawn one worker per part in the same batch and merge the results with the git tool; a single-worker turn is justified only when the next step genuinely depends on the last result. Enumerate the separable groups of files an implementation would touch before you spawn: when the work splits into such groups, spawn one worker per group in the same batch, land any file two groups would both edit in a precursor first, and aim the redundant pair (below) at the riskiest contract area - the densest edge surface or the requirement with the least repository evidence. Plan the split so the same files are not edited twice and merges stay clean. If independent parts share a file, land the shared file first in a quick precursor worker, then split from that checkpoint. Never spawn multiple workers with identical instructions - duplicates are collapsed, and a stalled approach needs diagnosis, not repetition. Deliberate redundancy is different: for the riskiest core of the contract - the densest edge surface, the most contested ambiguities - spawn two workers in the same batch to implement that same contract independently from different vantages you name (one from the spec's data model outward, one from the test cases inward), sharing your settled ambiguity resolutions with both. Their tests and behavior will diverge precisely on the edges nobody knew were ambiguous and on slips a single worker's self-tests cannot see. When siblings have implemented the same contract, harvest that divergence before resolving the batch: spawn a differential worker from one sibling's checkpoint that checks out the other sibling's test files directly from its commit (git checkout <sha> -- <paths>) and runs them against the implementation it is standing in. Every cross-sibling failure is an ambiguity two readings resolved differently - adjudicate it against the task text, never by majority or by trusting whichever implementation looks cleaner. Fact-finding stays cheap throughout: short workers that answer questions ("find how X is implemented; report file and line"), variants branched from a good checkpoint, losses cut early, instructions explicit and testable.

Finalize is where "Verification" binds you: a worker's report is a claim, not evidence. Do not finalize until the Verification requirements above have actually been discharged on the finalized checkpoint's chain - real test runs whose commands and output you have inspected via view_tool_call. A pre-existing suite that passed before the change proves nothing about the change: the evidence must include tests that exercise the new behavior the task demands - ideally the spec tests written at the start. If that evidence does not exist yet, your prefinalize batch must produce it. A plant is only evidence after re-checking the expected behavior against the spec text or an existing reference implementation - a self-authored test that encodes your own misreading will catch the correct code as the bug. A filtered or single-file test run is progress evidence, not completion evidence: completion evidence is the project's full suite, or a stated reason it cannot be run. Delivery, branches, and commit ceremony are handled outside the run - never spend a worker on commit messages or branch bookkeeping."#,
        supervisor_steps = ASGARD_SUPERVISOR_MAX_STEPS,
        workers = ASGARD_BATCH_CAP,
        worker_steps = ASGARD_WORKER_MAX_STEPS
    )
}

pub(crate) fn supervisor_tool_definitions(allowed_models: &[String]) -> Vec<ToolDefinition> {
    let mut model_enum: Vec<&String> = Vec::new();
    for model in allowed_models {
        if !model_enum.contains(&model) {
            model_enum.push(model);
        }
    }
    let allowed_models = model_enum;
    vec![
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: SPAWN_WORKERS_TOOL.to_string(),
                description: "Fork a barrier batch of new workers from \"root\" or a saved checkpoint id like \"w3\" (or a pending reviewed worker's id, which saves it); each runs up to 10 steps in its own checkout, then the whole batch reports back together. A successful spawn ends your turn immediately.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["workers"],
                    "properties": {
                        "workers": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": ASGARD_BATCH_CAP,
                            "items": {
                                "type": "object",
                                "additionalProperties": false,
                                "required": ["from", "instructions"],
                                "properties": {
                                    "from": { "type": "string" },
                                    "prefix_from": {
                                        "type": "string",
                                        "description": "Optional context prefix control. Omit for full first-parent lineage inheritance. Set to an ancestor checkpoint id on the from lineage to inherit only windows from that checkpoint through from; set to \"none\" for no inherited windows. Elided history is replaced by deterministic git diff orientation. Reusing a recently used prefix_from is cheaper because provider prompt caching can hit.",
                                    },
                                    "instructions": { "type": "string", "minLength": 1 },
                                    "model": {
                                        "type": "string",
                                        "enum": allowed_models,
                                    },
                                    "single_because": {
                                        "type": "string",
                                        "maxLength": ASGARD_SINGLE_BECAUSE_MAX_LENGTH,
                                        "description": "When continuing width-1 after repeated width-1 turns: one sentence naming why exactly one continuation is live.",
                                    },
                                },
                            },
                        },
                    },
                }),
            },
        },
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: PREFINALIZE_TOOL.to_string(),
                description: "Spawn your final verification pass; finalize is locked until this has run and been reviewed. Verification workers should: run the project's full test suite on the checkpoint you intend to deliver; spawn one worker per plant - multi-plant protocols exhaust step budgets; plant one classic bug in the diff's critical paths - swap an argument order, invert a boundary, drop a term - and confirm the suite goes red. Each plant must be a real mutation of the SHIPPED implementation confirmed by inspecting the diff, never a synthetic stand-in function or an edit to test files/mocks. Revert the plant and report any plant the suite failed to catch (a surviving plant means a test must be strengthened before finalizing); re-check the task's obligations against the delivered state. Alongside the plant workers, include one verification worker dedicated to adversarial inputs derived from the spec text alone - empty, zero, single vs many, negative, trailing, an attribute present but valueless - exercised through the public surface; near-misses live in the edges the implementation's own tests never generated. A hung or timed-out verification run is a blocker: identify the exact hanging test before attributing it to anything pre-existing. Never substitute a weaker ad-hoc check. If verification reveals needed work stranded in another checkpoint, merge it in with the git tool.".to_string(),
                parameters: prefinalize_parameters(&allowed_models),
            },
        },
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: SAVE_CHECKPOINT_TOOL.to_string(),
                description: "Save a reviewed trajectory as a permanent checkpoint without spawning from it yet. Pass worker like \"w7\" when multiple siblings are pending.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "worker": { "type": "string" },
                    },
                }),
            },
        },
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: GIT_TOOL.to_string(),
                description: "Run a git command directly (args as an argv list, e.g. [\"diff\", \"w2\", \"w5\"]) in your own scratch worktree of the shared repository, created lazily on first use and reused for the whole run. Every checkpoint is a real commit, so this is how you look before deciding (`git diff <parent> <sha>`, `git show <sha> -- <file>`, `git log`) and how you merge: check out the target and merge the other checkpoint's hash - the resulting commit is deliverable by its hash. On a conflicted merge the merge is aborted automatically; spawn a worker to do that merge instead. gc, prune, and reflog expire are refused - they would endanger the shared object store every worktree depends on.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["args"],
                    "properties": {
                        "args": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "argv passed to the git CLI directly, never through a shell",
                        },
                    },
                }),
            },
        },
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: DISCARD_TOOL.to_string(),
                description: "Discard a reviewed trajectory permanently. Every reviewed trajectory must be saved, spawned from (which saves it implicitly), or discarded. Pass worker like \"w7\" when multiple siblings are pending.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "worker": { "type": "string" },
                    },
                }),
            },
        },
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: FINALIZE_TOOL.to_string(),
                description: "End the run delivering the named checkpoint's state; response overrides the checkpoint worker's final message as the user-facing answer. Evidence handles for inspected test runs remain welcome context.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["checkpoint"],
                    "properties": {
                        "checkpoint": { "type": "string" },
                        "response": { "type": "string" },
                        "evidence": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "the handles of the test runs you inspected that verify the finalized state",
                        },
                        "abandoned": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "checkpoint ids you are deliberately leaving out of the delivered lineage",
                        },
                    },
                }),
            },
        },
        ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: VIEW_TOOL_CALL_TOOL.to_string(),
                description: "Expand compact-trace handles like \"w3m5\" into complete untruncated arguments and results; free of charge; works across every saved trajectory and all siblings in the batch under review.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["handles"],
                    "properties": {
                        "handles": {
                            "type": "array",
                            "maxItems": ASGARD_VIEW_TOOL_CALL_MAX_HANDLES,
                            "items": { "type": "string" },
                        },
                    },
                }),
            },
        },
        crate::tools::update_plan_tool_definition(),
    ]
}

/// Maps a live Bifrost (`core` toolset) MCP client's advertised tools into
/// supervisor `ToolDefinition`s, preserving the client's tool order verbatim so
/// the supervisor's tool set stays byte-identical across turns (prefix-cache
/// stability). Names, descriptions, and input schemas pass through unchanged.
pub(crate) fn bifrost_tool_definitions(client: &McpClient) -> Vec<ToolDefinition> {
    mcp_tools_to_definitions(client.tools())
}

fn mcp_tools_to_definitions(tools: &[McpToolDef]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .map(|tool| ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDef {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.input_schema.clone(),
            },
        })
        .collect()
}

/// Prefinalize's own parameters: the spawn_workers worker shape plus the
/// coverage contract - per-worker `runs_full_suite` and `attacks`, and a
/// top-level `full_suite_skipped` reason. spawn_workers keeps its own
/// schema; these fields never appear there.
fn prefinalize_parameters(allowed_models: &[&String]) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["workers"],
        "properties": {
            "workers": {
                "type": "array",
                "minItems": 1,
                "maxItems": ASGARD_BATCH_CAP,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["from", "instructions"],
                    "properties": {
                        "from": { "type": "string" },
                        "prefix_from": {
                            "type": "string",
                            "description": "Optional context prefix control. Prefinalize workers default to \"none\" when omitted. Otherwise use an ancestor checkpoint id on the from lineage to inherit only windows from that checkpoint through from. Elided history is replaced by deterministic git diff orientation.",
                        },
                        "instructions": { "type": "string", "minLength": 1 },
                        "model": {
                            "type": "string",
                            "enum": allowed_models,
                        },
                        "max_steps": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": ASGARD_WORKER_MAX_STEPS_CEILING,
                            "description": "Step budget for this worker (default 10). Give a worker pursuing a complete solution a large budget; keep quick probes small.",
                        },
                        "single_because": {
                            "type": "string",
                            "maxLength": ASGARD_SINGLE_BECAUSE_MAX_LENGTH,
                            "description": "When continuing width-1 after repeated width-1 turns: one sentence naming why exactly one continuation is live.",
                        },
                        "runs_full_suite": {
                            "type": "boolean",
                            "description": "True on the worker whose instructions run the project's FULL, unfiltered test suite on the delivery candidate. Scoped or filtered runs do not count. Every prefinalize batch needs one of these or a top-level full_suite_skipped reason.",
                        },
                        "attacks": {
                            "type": "array",
                            "items": { "type": "string", "maxLength": ASGARD_ATTACK_MAX_LENGTH },
                            "description": "Load-bearing beliefs this worker attacks. A belief is load-bearing when the delivery is wrong if the belief is: an ambiguity you resolved, a 'this cannot affect X' assumption, a compliance claim you have never seen fail-then-pass, any code or signature you dictated. An attack constructs the situation where a wrong belief breaks - it does not re-run tests that already pass. The belief you do not name is the one that fails the hidden suite.",
                        },
                    },
                },
            },
            "full_suite_skipped": {
                "type": "string",
                "maxLength": ASGARD_FULL_SUITE_SKIPPED_MAX_LENGTH,
                "description": "Only when no worker carries runs_full_suite: one sentence naming why a full-suite run is not warranted for this delivery.",
            },
        },
    })
}

pub(crate) async fn stream_supervisor_response(
    call: SupervisorStreamCall<'_>,
) -> Result<LlmResponse> {
    crate::llm_client::stream_chat_no_visible_output_with_retry(
        call.llm,
        "running Asgard supervisor turn",
        call.cancel,
        || {
            let mut messages = call.request_prefix.to_vec();
            messages.extend(call.tail.to_vec());
            let text_sink = call.text_sink.clone();
            let thought_sink = call.thought_sink.clone();
            StreamChatRequest {
                model: call.model.to_string(),
                messages,
                tools: Some(call.tools.to_vec()),
                reasoning_effort: call.reasoning_effort.map(str::to_string),
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
                cancel: call.cancel.clone(),
                idle_timeouts: call.idle_timeout,
            }
        },
    )
    .await
}

pub(crate) fn parse_view_tool_call(call: &ToolCall) -> std::result::Result<Vec<String>, String> {
    let arguments = normalize_arguments(&call.function.arguments)?;
    let handles = string_array_property(&arguments, "handles")?;
    if handles.len() > ASGARD_VIEW_TOOL_CALL_MAX_HANDLES {
        return Err(format!(
            "view_tool_call accepts at most {ASGARD_VIEW_TOOL_CALL_MAX_HANDLES} handles, got {}",
            handles.len()
        ));
    }
    Ok(handles)
}

pub(crate) fn parse_spawn_workers(
    call: &ToolCall,
    context: &SupervisorTurnContext<'_>,
) -> std::result::Result<Vec<SpawnRequest>, String> {
    // Regular spawn_workers never carries the coverage fields; if a model
    // passes them anyway they are ignored (absent from the tool's schema).
    parse_spawn_workers_impl(call, context, false)
}

/// Parses a prefinalize call: the same worker batch as spawn_workers plus the
/// coverage contract (`runs_full_suite`/`attacks` per worker, top-level
/// `full_suite_skipped`).
pub(crate) fn parse_prefinalize(
    call: &ToolCall,
    context: &SupervisorTurnContext<'_>,
) -> std::result::Result<PrefinalizeRequest, String> {
    let workers = parse_spawn_workers_impl(call, context, true)?;
    let arguments = normalize_arguments(&call.function.arguments)?;
    let full_suite_skipped = match arguments.get("full_suite_skipped") {
        None | Some(serde_json::Value::Null) => None,
        Some(value) => {
            let text = value
                .as_str()
                .ok_or_else(|| "full_suite_skipped must be a string".to_string())?
                .trim()
                .to_string();
            if text.len() > ASGARD_FULL_SUITE_SKIPPED_MAX_LENGTH {
                return Err(format!(
                    "full_suite_skipped must be at most {ASGARD_FULL_SUITE_SKIPPED_MAX_LENGTH} characters"
                ));
            }
            if text.is_empty() { None } else { Some(text) }
        }
    };
    Ok(PrefinalizeRequest {
        workers,
        full_suite_skipped,
    })
}

fn parse_spawn_workers_impl(
    call: &ToolCall,
    context: &SupervisorTurnContext<'_>,
    parse_coverage: bool,
) -> std::result::Result<Vec<SpawnRequest>, String> {
    let arguments = normalize_arguments(&call.function.arguments)?;
    let workers = arguments
        .get("workers")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "workers must be an array".to_string())?;
    if workers.is_empty() {
        return Err("workers must contain at least one worker.".to_string());
    }
    if workers.len() > ASGARD_BATCH_CAP {
        return Err(format!(
            "workers must contain at most {ASGARD_BATCH_CAP} workers"
        ));
    }

    workers
        .iter()
        .enumerate()
        .map(|(index, worker)| parse_spawn_worker(index, worker, context, parse_coverage))
        .collect()
}

fn parse_spawn_worker(
    index: usize,
    worker: &serde_json::Value,
    context: &SupervisorTurnContext<'_>,
    parse_coverage: bool,
) -> std::result::Result<SpawnRequest, String> {
    let from_raw = worker
        .get("from")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("workers[{index}].from must be a string"))?;
    let from = parse_available_checkpoint(from_raw, context)
        .map_err(|error| format!("workers[{index}].from {error}"))?;
    let prefix_from = parse_prefix_from(index, worker, &from, context, parse_coverage)?;
    let instructions = worker
        .get("instructions")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("workers[{index}].instructions must be a string"))?
        .trim()
        .to_string();
    if instructions.is_empty() {
        return Err(format!("workers[{index}].instructions must not be empty"));
    }
    let model = worker
        .get("model")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    if let Some(model) = &model
        && !context
            .allowed_models
            .iter()
            .any(|allowed| allowed == model)
    {
        return Err(format!("workers[{index}].model {model:?} is not allowed"));
    }
    let max_steps = match worker.get("max_steps") {
        None | Some(serde_json::Value::Null) => None,
        Some(value) => {
            let steps = value
                .as_u64()
                .ok_or_else(|| format!("workers[{index}].max_steps must be an integer"))?;
            if steps < 1 || steps > ASGARD_WORKER_MAX_STEPS_CEILING as u64 {
                return Err(format!(
                    "workers[{index}].max_steps must be between 1 and {ASGARD_WORKER_MAX_STEPS_CEILING}"
                ));
            }
            Some(steps as usize)
        }
    };
    let single_because = match worker.get("single_because") {
        None | Some(serde_json::Value::Null) => None,
        Some(value) => {
            let text = value
                .as_str()
                .ok_or_else(|| format!("workers[{index}].single_because must be a string"))?
                .trim()
                .to_string();
            if text.len() > ASGARD_SINGLE_BECAUSE_MAX_LENGTH {
                return Err(format!(
                    "workers[{index}].single_because must be at most {ASGARD_SINGLE_BECAUSE_MAX_LENGTH} characters"
                ));
            }
            if text.is_empty() { None } else { Some(text) }
        }
    };
    let runs_full_suite = if parse_coverage {
        match worker.get("runs_full_suite") {
            None | Some(serde_json::Value::Null) => false,
            Some(value) => value
                .as_bool()
                .ok_or_else(|| format!("workers[{index}].runs_full_suite must be a boolean"))?,
        }
    } else {
        false
    };
    let attacks = if parse_coverage {
        match worker.get("attacks") {
            None | Some(serde_json::Value::Null) => Vec::new(),
            Some(_) => {
                let attacks = string_array_property(worker, "attacks")
                    .map_err(|error| format!("workers[{index}].{error}"))?;
                let mut trimmed = Vec::new();
                for (attack_index, attack) in attacks.iter().enumerate() {
                    let attack = attack.trim();
                    if attack.len() > ASGARD_ATTACK_MAX_LENGTH {
                        return Err(format!(
                            "workers[{index}].attacks[{attack_index}] must be at most {ASGARD_ATTACK_MAX_LENGTH} characters"
                        ));
                    }
                    if !attack.is_empty() {
                        trimmed.push(attack.to_string());
                    }
                }
                trimmed
            }
        }
    } else {
        Vec::new()
    };
    Ok(SpawnRequest {
        from,
        prefix_from,
        instructions,
        model,
        max_steps,
        single_because,
        runs_full_suite,
        attacks,
    })
}

fn parse_prefix_from(
    index: usize,
    worker: &serde_json::Value,
    from: &CheckpointId,
    context: &SupervisorTurnContext<'_>,
    prefinalize_default_fresh: bool,
) -> std::result::Result<Option<PrefixFrom>, String> {
    let Some(value) = worker.get("prefix_from") else {
        return Ok(prefinalize_default_fresh.then_some(PrefixFrom::Fresh));
    };
    let text = value
        .as_str()
        .ok_or_else(|| format!("workers[{index}].prefix_from must be a string"))?
        .trim();
    if text == "none" {
        return Ok(Some(PrefixFrom::Fresh));
    }
    if text.is_empty() {
        return Err(format!(
            "workers[{index}].prefix_from must be \"none\" or a checkpoint id"
        ));
    }
    let checkpoint = parse_available_checkpoint(text, context)
        .map_err(|error| format!("workers[{index}].prefix_from {error}"))?;
    if !is_lineage_prefix(&checkpoint, from, context) {
        return Err(format!(
            "workers[{index}].prefix_from {checkpoint} is not an ancestor of {from} on its first-parent lineage"
        ));
    }
    Ok(Some(PrefixFrom::Checkpoint(checkpoint)))
}

fn is_lineage_prefix(
    prefix: &CheckpointId,
    from: &CheckpointId,
    context: &SupervisorTurnContext<'_>,
) -> bool {
    if prefix == from {
        return true;
    }
    if let CheckpointId::Worker(worker) = from
        && let Some((_, parent)) = context
            .pending_parents
            .iter()
            .find(|(pending_worker, _)| pending_worker == worker)
    {
        return context.dag.is_first_parent_ancestor_of(prefix, parent);
    }
    context.dag.is_first_parent_ancestor_of(prefix, from)
}

pub(crate) fn parse_finalize(
    call: &ToolCall,
    context: &SupervisorTurnContext<'_>,
) -> std::result::Result<FinalizeRequest, String> {
    let arguments = normalize_arguments(&call.function.arguments)?;
    let checkpoint_raw = arguments
        .get("checkpoint")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "checkpoint must be a string".to_string())?;
    let checkpoint = parse_finalize_checkpoint(checkpoint_raw, context)?;
    let response = arguments
        .get("response")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string);
    let evidence = match arguments.get("evidence") {
        Some(_) => string_array_property(&arguments, "evidence")?,
        None => Vec::new(),
    };
    let abandoned = match arguments.get("abandoned") {
        Some(_) => string_array_property(&arguments, "abandoned")?,
        None => Vec::new(),
    };
    Ok(FinalizeRequest {
        checkpoint,
        response,
        evidence,
        abandoned,
    })
}

/// `checkpoint` is normally "root"/"wN", but also accepts a commit hash
/// (short or full): rev-parsed in the parent repo and required to descend
/// from the run's base commit. When the hash matches an already-known
/// checkpoint's commit that checkpoint is returned directly; a novel
/// descendant (e.g. a merge made by the `git` tool) is still accepted, so it
/// can be finalized once its own verification evidence exists.
fn parse_finalize_checkpoint(
    value: &str,
    context: &SupervisorTurnContext<'_>,
) -> std::result::Result<CheckpointId, String> {
    if let Some(checkpoint) = CheckpointId::parse(value) {
        return if context.dag.contains(&checkpoint) || is_pending_checkpoint(&checkpoint, context) {
            Ok(checkpoint)
        } else {
            Err(format!(
                "checkpoint {checkpoint} is not root, saved, or under review"
            ))
        };
    }
    context
        .dag
        .resolve_checkpoint_by_commit(value)
        .map_err(|error| format!("checkpoint {error}"))
}

pub(crate) fn parse_update_plan(
    call: &ToolCall,
) -> std::result::Result<crate::plan::UpdatePlanArgs, String> {
    let arguments = normalize_arguments(&call.function.arguments)?;
    serde_json::from_value::<crate::plan::UpdatePlanArgs>(arguments)
        .map_err(|error| format!("Invalid update_plan arguments: {error}"))
}

/// Parses the `git` tool's `{args: string[]}` payload into the argv that
/// will be passed to the git CLI directly - never through a shell, so a
/// value containing shell metacharacters stays exactly one argv token.
pub(crate) fn parse_git_args(call: &ToolCall) -> std::result::Result<Vec<String>, String> {
    let arguments = normalize_arguments(&call.function.arguments)?;
    string_array_property(&arguments, "args")
}

/// Returns the first argument that isn't a flag (doesn't start with `-`).
/// This is a deliberately static, naive scan (it does not special-case
/// value-taking global flags like `-c <val>`) matching the refusal check's
/// own "first non-flag argument" rule.
pub(crate) fn first_non_flag_arg(args: &[String]) -> Option<&str> {
    args.iter()
        .find(|arg| !arg.starts_with('-'))
        .map(String::as_str)
}

/// Refuses git subcommands that would endanger the object store every
/// worktree (worker and supervisor scratch alike) shares: `gc`, `prune`, and
/// `reflog expire`. The check is static - on the argv alone, before any git
/// process runs.
pub(crate) fn git_refusal(args: &[String]) -> Option<String> {
    let mut non_flags = args.iter().filter(|arg| !arg.starts_with('-'));
    let first = non_flags.next()?.as_str();
    let refused_cmd = match first {
        "gc" => Some("gc"),
        "prune" => Some("prune"),
        "reflog" if non_flags.next().map(String::as_str) == Some("expire") => Some("reflog expire"),
        _ => None,
    };
    refused_cmd.map(|cmd| {
        format!("refused: {cmd} endangers the shared object store all worktrees depend on.")
    })
}

/// Retains read tool payloads - `view_tool_call`, `git`, and the read-only
/// Bifrost code-intelligence tools - in the permanent record instead of
/// collapsing them to a byte-count stub, capping each at `RETAINED_PAYLOAD_CAP`
/// bytes (verbatim prefix + a marker naming the original size). The predicate is
/// inverted: a tool result is retained unless it comes from an ACTION tool whose
/// result is a short status string (spawn/prefinalize/save/discard/finalize/
/// update_plan). This keeps code-intelligence output - evidence later turns need
/// to plan against - and automatically covers any future read tool without
/// re-listing names. Action-tool results are already under the cap, so capping
/// them is a no-op; they pass through effectively unchanged.
pub(crate) fn retain_payloads_for_permanent_record(transcript: &[ChatMessage]) -> Vec<ChatMessage> {
    transcript
        .iter()
        .map(|message| {
            if message.role == "tool"
                && message
                    .name
                    .as_deref()
                    .is_some_and(|name| !is_supervisor_action_tool(name))
                && let Some(call_id) = &message.tool_call_id
            {
                let name = message.name.clone().unwrap_or_default();
                let retained = retain_payload(&message.content_text());
                return ChatMessage::tool_result(call_id, name, retained);
            }
            message.clone()
        })
        .collect()
}

/// Supervisor tools whose result is a short status string, not evidence: these
/// pass through the permanent record unchanged. Everything else (read tools:
/// view_tool_call, git, and Bifrost code-intelligence) is retained under the
/// cap. See [`retain_payloads_for_permanent_record`].
fn is_supervisor_action_tool(name: &str) -> bool {
    matches!(
        name,
        SPAWN_WORKERS_TOOL
            | PREFINALIZE_TOOL
            | SAVE_CHECKPOINT_TOOL
            | DISCARD_TOOL
            | FINALIZE_TOOL
            | UPDATE_PLAN_TOOL
    )
}

fn retain_payload(content: &str) -> String {
    if content.len() <= RETAINED_PAYLOAD_CAP {
        return content.to_string();
    }
    let total = content.len();
    let prefix = crate::text::truncate_utf8(content, RETAINED_PAYLOAD_CAP);
    format!("{prefix}\n[retained first {RETAINED_PAYLOAD_CAP} bytes of {total}]")
}

fn parse_available_checkpoint(
    value: &str,
    context: &SupervisorTurnContext<'_>,
) -> std::result::Result<CheckpointId, String> {
    let checkpoint =
        CheckpointId::parse(value).ok_or_else(|| format!("{value:?} is not a checkpoint id"))?;
    if context.dag.contains(&checkpoint) || is_pending_checkpoint(&checkpoint, context) {
        Ok(checkpoint)
    } else {
        Err(format!("{checkpoint} is not root, saved, or under review"))
    }
}

pub(crate) fn is_pending_checkpoint(
    checkpoint: &CheckpointId,
    context: &SupervisorTurnContext<'_>,
) -> bool {
    matches!(checkpoint, CheckpointId::Worker(worker) if context.pending.contains(worker))
}

fn normalize_arguments(raw: &str) -> std::result::Result<serde_json::Value, String> {
    crate::tool_arguments::normalize_tool_arguments(raw)
        .map(|arguments| arguments.value)
        .map_err(|error| format!("unparseable arguments: {error:#}"))
}

fn string_array_property(
    arguments: &serde_json::Value,
    property: &str,
) -> std::result::Result<Vec<String>, String> {
    arguments
        .get(property)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{property} must be an array"))?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("{property}[{index}] must be a string"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::asgard::{TrajectoryNode, TrajectoryWindow, WorkerStopReason};
    use crate::llm_client::{FunctionCall, TokenUsage};

    fn supervisor_tool_call(id: &str, name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            r#type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    fn saved_dag() -> TrajectoryDag {
        let mut dag = TrajectoryDag::new(vec![ChatMessage::user("task")], "base".to_string());
        dag.insert(TrajectoryNode {
            window: TrajectoryWindow {
                worker: 1,
                parent: CheckpointId::Root,
                instructions: "saved".to_string(),
                model: "model-a".to_string(),
                instruction_message: ChatMessage::user("saved worker instructions"),
                window_messages: Vec::new(),
                rendered_tokens: 0,
                compact: String::new(),
                final_response: "saved result".to_string(),
                stop: WorkerStopReason::Finished,
                steps: 1,
                diffstat: String::new(),
                usage: TokenUsage::default(),
                elapsed_millis: 0,
            },
            commit: "commit-1".to_string(),
            merged_from: Vec::new(),
        })
        .unwrap();
        dag
    }

    fn context<'a>(dag: &'a TrajectoryDag, pending: &'a [usize]) -> SupervisorTurnContext<'a> {
        static ALLOWED: &[String] = &[];
        SupervisorTurnContext {
            dag,
            pending,
            pending_parents: &[],
            allowed_models: ALLOWED,
        }
    }

    #[test]
    fn spawn_parser_accepts_root_saved_and_pending_checkpoints() {
        let dag = saved_dag();
        let allowed = vec!["model-a".to_string()];
        let context = SupervisorTurnContext {
            dag: &dag,
            pending: &[4],
            pending_parents: &[(4, CheckpointId::Worker(1))],
            allowed_models: &allowed,
        };
        let call = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({
                "workers": [
                    { "from": "root", "instructions": "bootstrap" },
                    { "from": "w1", "instructions": "branch saved", "model": "model-a" },
                    { "from": "w4", "instructions": "branch pending" }
                ]
            }),
        );

        let spawns = parse_spawn_workers(&call, &context).expect("valid spawns");

        assert_eq!(spawns.len(), 3);
        assert_eq!(spawns[2].from, CheckpointId::Worker(4));
    }

    #[test]
    fn spawn_parser_accepts_and_round_trips_single_because() {
        let dag = saved_dag();
        let context = SupervisorTurnContext {
            dag: &dag,
            pending: &[],
            pending_parents: &[],
            allowed_models: &[],
        };

        let with_reason = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({
                "workers": [
                    {
                        "from": "root",
                        "instructions": "continue the fix",
                        "single_because": "Only one plausible fix location remains after ruling out the other hypothesis.",
                    }
                ]
            }),
        );
        let spawns = parse_spawn_workers(&with_reason, &context).expect("valid spawn");
        assert_eq!(
            spawns[0].single_because.as_deref(),
            Some("Only one plausible fix location remains after ruling out the other hypothesis.")
        );

        // Absent and blank single_because both parse to None.
        let without_reason = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({ "workers": [{ "from": "root", "instructions": "x" }] }),
        );
        let spawns = parse_spawn_workers(&without_reason, &context).expect("valid spawn");
        assert_eq!(spawns[0].single_because, None);

        let blank_reason = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({ "workers": [{ "from": "root", "instructions": "x", "single_because": "   " }] }),
        );
        let spawns = parse_spawn_workers(&blank_reason, &context).expect("valid spawn");
        assert_eq!(spawns[0].single_because, None);

        // Over the schema's maxLength is rejected.
        let too_long = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({
                "workers": [{
                    "from": "root",
                    "instructions": "x",
                    "single_because": "x".repeat(ASGARD_SINGLE_BECAUSE_MAX_LENGTH + 1),
                }]
            }),
        );
        assert!(
            parse_spawn_workers(&too_long, &context)
                .expect_err("too long single_because")
                .contains("must be at most")
        );
    }

    #[test]
    fn spawn_parser_validates_prefix_from_on_first_parent_lineage() {
        let mut dag = saved_dag();
        dag.insert(TrajectoryNode {
            window: TrajectoryWindow {
                worker: 2,
                parent: CheckpointId::Worker(1),
                instructions: "child".to_string(),
                model: "model-a".to_string(),
                instruction_message: ChatMessage::user("child worker instructions"),
                window_messages: Vec::new(),
                rendered_tokens: 0,
                compact: String::new(),
                final_response: "child result".to_string(),
                stop: WorkerStopReason::Finished,
                steps: 1,
                diffstat: String::new(),
                usage: TokenUsage::default(),
                elapsed_millis: 0,
            },
            commit: "commit-2".to_string(),
            merged_from: Vec::new(),
        })
        .unwrap();
        dag.insert(TrajectoryNode {
            window: TrajectoryWindow {
                worker: 3,
                parent: CheckpointId::Root,
                instructions: "sibling".to_string(),
                model: "model-a".to_string(),
                instruction_message: ChatMessage::user("sibling worker instructions"),
                window_messages: Vec::new(),
                rendered_tokens: 0,
                compact: String::new(),
                final_response: "sibling result".to_string(),
                stop: WorkerStopReason::Finished,
                steps: 1,
                diffstat: String::new(),
                usage: TokenUsage::default(),
                elapsed_millis: 0,
            },
            commit: "commit-3".to_string(),
            merged_from: Vec::new(),
        })
        .unwrap();
        let context = context(&dag, &[]);

        let valid = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({
                "workers": [{ "from": "w2", "prefix_from": "w1", "instructions": "x" }]
            }),
        );
        let spawns = parse_spawn_workers(&valid, &context).expect("valid prefix");
        assert_eq!(
            spawns[0].prefix_from,
            Some(PrefixFrom::Checkpoint(CheckpointId::Worker(1)))
        );

        let fresh = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({
                "workers": [{ "from": "w2", "prefix_from": "none", "instructions": "x" }]
            }),
        );
        let spawns = parse_spawn_workers(&fresh, &context).expect("fresh prefix");
        assert_eq!(spawns[0].prefix_from, Some(PrefixFrom::Fresh));

        let sibling = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({
                "workers": [{ "from": "w2", "prefix_from": "w3", "instructions": "x" }]
            }),
        );
        assert!(
            parse_spawn_workers(&sibling, &context)
                .expect_err("sibling prefix")
                .contains("is not an ancestor of w2 on its first-parent lineage")
        );
    }

    #[test]
    fn prefinalize_defaults_prefix_from_to_fresh() {
        let dag = saved_dag();
        let context = context(&dag, &[]);
        let call = supervisor_tool_call(
            "prefinalize",
            PREFINALIZE_TOOL,
            serde_json::json!({
                "workers": [{ "from": "w1", "instructions": "verify", "runs_full_suite": true }]
            }),
        );
        let parsed = parse_prefinalize(&call, &context).expect("prefinalize parses");
        assert_eq!(parsed.workers[0].prefix_from, Some(PrefixFrom::Fresh));
    }

    #[test]
    fn prefinalize_parser_parses_coverage_fields_and_threads_full_suite_skipped() {
        let dag = saved_dag();
        let context = SupervisorTurnContext {
            dag: &dag,
            pending: &[],
            pending_parents: &[],
            allowed_models: &[],
        };

        let call = supervisor_tool_call(
            "pf",
            PREFINALIZE_TOOL,
            serde_json::json!({
                "workers": [
                    {
                        "from": "root",
                        "instructions": "run the full suite",
                        "runs_full_suite": true,
                        "attacks": ["  Belief: sorting is stable -> equal keys diffed  "],
                    },
                    { "from": "w1", "instructions": "adversarial edges" }
                ],
                "full_suite_skipped": "  not warranted: docs-only delivery  ",
            }),
        );
        let parsed = parse_prefinalize(&call, &context).expect("valid prefinalize");
        assert!(parsed.workers[0].runs_full_suite);
        assert_eq!(
            parsed.workers[0].attacks,
            vec!["Belief: sorting is stable -> equal keys diffed".to_string()]
        );
        assert!(!parsed.workers[1].runs_full_suite);
        assert!(parsed.workers[1].attacks.is_empty());
        assert_eq!(
            parsed.full_suite_skipped.as_deref(),
            Some("not warranted: docs-only delivery")
        );

        // Absent and blank full_suite_skipped both thread through as None.
        let without_reason = supervisor_tool_call(
            "pf",
            PREFINALIZE_TOOL,
            serde_json::json!({ "workers": [{ "from": "root", "instructions": "x" }] }),
        );
        let parsed = parse_prefinalize(&without_reason, &context).expect("valid prefinalize");
        assert_eq!(parsed.full_suite_skipped, None);
        assert!(!parsed.workers[0].runs_full_suite);
        assert!(parsed.workers[0].attacks.is_empty());

        let blank_reason = supervisor_tool_call(
            "pf",
            PREFINALIZE_TOOL,
            serde_json::json!({
                "workers": [{ "from": "root", "instructions": "x" }],
                "full_suite_skipped": "   ",
            }),
        );
        let parsed = parse_prefinalize(&blank_reason, &context).expect("valid prefinalize");
        assert_eq!(parsed.full_suite_skipped, None);
    }

    #[test]
    fn prefinalize_parser_enforces_attack_and_skip_reason_caps() {
        let dag = saved_dag();
        let context = SupervisorTurnContext {
            dag: &dag,
            pending: &[],
            pending_parents: &[],
            allowed_models: &[],
        };

        let long_attack = supervisor_tool_call(
            "pf",
            PREFINALIZE_TOOL,
            serde_json::json!({
                "workers": [{
                    "from": "root",
                    "instructions": "x",
                    "attacks": ["a".repeat(ASGARD_ATTACK_MAX_LENGTH + 1)],
                }]
            }),
        );
        let error = parse_prefinalize(&long_attack, &context).expect_err("201-char attack");
        assert!(
            error.contains("workers[0].attacks[0] must be at most 200 characters"),
            "unexpected error: {error}"
        );

        let long_reason = supervisor_tool_call(
            "pf",
            PREFINALIZE_TOOL,
            serde_json::json!({
                "workers": [{ "from": "root", "instructions": "x" }],
                "full_suite_skipped": "b".repeat(ASGARD_FULL_SUITE_SKIPPED_MAX_LENGTH + 1),
            }),
        );
        let error = parse_prefinalize(&long_reason, &context).expect_err("301-char reason");
        assert!(
            error.contains("full_suite_skipped must be at most 300 characters"),
            "unexpected error: {error}"
        );

        let bad_flag = supervisor_tool_call(
            "pf",
            PREFINALIZE_TOOL,
            serde_json::json!({
                "workers": [{ "from": "root", "instructions": "x", "runs_full_suite": "yes" }]
            }),
        );
        assert!(
            parse_prefinalize(&bad_flag, &context)
                .expect_err("non-boolean runs_full_suite")
                .contains("workers[0].runs_full_suite must be a boolean")
        );
    }

    #[test]
    fn spawn_workers_parser_ignores_coverage_fields() {
        let dag = saved_dag();
        let context = SupervisorTurnContext {
            dag: &dag,
            pending: &[],
            pending_parents: &[],
            allowed_models: &[],
        };

        // The coverage fields are absent from spawn_workers' schema; a model
        // passing them anyway must not affect the spawn - not even the
        // per-item cap applies.
        let call = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({
                "workers": [{
                    "from": "root",
                    "instructions": "x",
                    "runs_full_suite": true,
                    "attacks": ["a".repeat(ASGARD_ATTACK_MAX_LENGTH + 100)],
                }]
            }),
        );
        let spawns = parse_spawn_workers(&call, &context).expect("coverage fields ignored");
        assert!(!spawns[0].runs_full_suite);
        assert!(spawns[0].attacks.is_empty());
    }

    #[test]
    fn prefinalize_schema_carries_coverage_contract_and_spawn_workers_does_not() {
        let tools = supervisor_tool_definitions(&["model-a".to_string()]);
        let prefinalize = tools
            .iter()
            .find(|tool| tool.function.name == PREFINALIZE_TOOL)
            .expect("prefinalize tool");
        let parameters = &prefinalize.function.parameters;
        let worker_properties = &parameters["properties"]["workers"]["items"]["properties"];
        assert_eq!(worker_properties["runs_full_suite"]["type"], "boolean");
        assert_eq!(
            worker_properties["attacks"]["items"]["maxLength"],
            ASGARD_ATTACK_MAX_LENGTH
        );
        assert_eq!(
            parameters["properties"]["full_suite_skipped"]["maxLength"],
            ASGARD_FULL_SUITE_SKIPPED_MAX_LENGTH
        );
        // Prefinalize keeps the shared worker fields after the schema split.
        assert!(worker_properties.get("prefix_from").is_some());
        assert!(worker_properties.get("max_steps").is_some());
        assert!(worker_properties.get("single_because").is_some());

        let spawn_workers = tools
            .iter()
            .find(|tool| tool.function.name == SPAWN_WORKERS_TOOL)
            .expect("spawn_workers tool");
        let spawn_parameters = &spawn_workers.function.parameters;
        let spawn_worker_properties =
            &spawn_parameters["properties"]["workers"]["items"]["properties"];
        assert!(spawn_worker_properties.get("prefix_from").is_some());
        assert!(spawn_worker_properties.get("runs_full_suite").is_none());
        assert!(spawn_worker_properties.get("attacks").is_none());
        assert!(
            spawn_parameters["properties"]
                .get("full_suite_skipped")
                .is_none()
        );
    }

    #[test]
    fn spawn_parser_rejects_bad_checkpoint_model_and_empty_instructions() {
        let dag = saved_dag();
        let allowed = vec!["model-a".to_string()];
        let context = SupervisorTurnContext {
            dag: &dag,
            pending: &[],
            pending_parents: &[],
            allowed_models: &allowed,
        };

        let bad_from = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({ "workers": [{ "from": "w99", "instructions": "x" }] }),
        );
        assert!(
            parse_spawn_workers(&bad_from, &context)
                .expect_err("bad checkpoint")
                .contains("w99 is not root, saved, or under review")
        );

        let bad_model = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({ "workers": [{ "from": "root", "instructions": "x", "model": "model-b" }] }),
        );
        assert!(
            parse_spawn_workers(&bad_model, &context)
                .expect_err("bad model")
                .contains("model-b")
        );

        let empty = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({ "workers": [{ "from": "root", "instructions": "  " }] }),
        );
        assert!(
            parse_spawn_workers(&empty, &context)
                .expect_err("empty instructions")
                .contains("must not be empty")
        );

        let nine_workers = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({
                "workers": (0..9)
                    .map(|index| serde_json::json!({
                        "from": "root",
                        "instructions": format!("worker {index}")
                    }))
                    .collect::<Vec<_>>()
            }),
        );
        assert!(
            parse_spawn_workers(&nine_workers, &context)
                .expect_err("too many workers")
                .contains("workers must contain at most 8 workers")
        );
    }

    #[test]
    fn finalize_parser_accepts_pending_and_response() {
        let dag = saved_dag();
        let call = supervisor_tool_call(
            "finalize",
            FINALIZE_TOOL,
            serde_json::json!({
                "checkpoint": "w4",
                "response": "done",
                "abandoned": ["w1"]
            }),
        );

        let parsed = parse_finalize(&call, &context(&dag, &[4])).expect("valid finalize");

        assert_eq!(parsed.checkpoint, CheckpointId::Worker(4));
        assert_eq!(parsed.response.as_deref(), Some("done"));
        assert!(parsed.evidence.is_empty());
        assert_eq!(parsed.abandoned, vec!["w1".to_string()]);
    }

    fn init_git_repo() -> (tempfile::TempDir, std::path::PathBuf) {
        let temp = tempfile::tempdir().expect("tempdir");
        let repo = temp.path().to_path_buf();
        let run = |args: &[&str]| {
            let status = std::process::Command::new("git")
                .args(args)
                .current_dir(&repo)
                .status()
                .expect("run git");
            assert!(status.success(), "git {args:?} failed");
        };
        run(&["init", "--quiet"]);
        run(&["config", "user.email", "asgard@example.invalid"]);
        run(&["config", "user.name", "Asgard Test"]);
        (temp, repo)
    }

    fn commit_file(repo: &std::path::Path, name: &str, content: &str) -> String {
        std::fs::write(repo.join(name), content).expect("write file");
        for args in [vec!["add", name], vec!["commit", "--quiet", "-m", name]] {
            let status = std::process::Command::new("git")
                .args(&args)
                .current_dir(repo)
                .status()
                .expect("run git");
            assert!(status.success(), "git {args:?} failed");
        }
        String::from_utf8(
            std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(repo)
                .output()
                .expect("rev-parse")
                .stdout,
        )
        .expect("utf8")
        .trim()
        .to_string()
    }

    #[test]
    fn finalize_parser_accepts_a_commit_hash_matching_a_saved_checkpoint() {
        let (_temp, repo) = init_git_repo();
        let base = commit_file(&repo, "base.txt", "base\n");
        let c1 = commit_file(&repo, "one.txt", "one\n");
        let mut dag = TrajectoryDag::new_with_git_root(
            vec![ChatMessage::user("task")],
            base.clone(),
            repo.clone(),
        );
        dag.insert(TrajectoryNode {
            window: TrajectoryWindow {
                worker: 1,
                parent: CheckpointId::Root,
                instructions: "one".to_string(),
                model: "model-a".to_string(),
                instruction_message: ChatMessage::user("one"),
                window_messages: Vec::new(),
                rendered_tokens: 0,
                compact: String::new(),
                final_response: "done".to_string(),
                stop: WorkerStopReason::Finished,
                steps: 1,
                diffstat: String::new(),
                usage: TokenUsage::default(),
                elapsed_millis: 0,
            },
            commit: c1.clone(),
            merged_from: Vec::new(),
        })
        .unwrap();
        let ctx = context(&dag, &[]);

        let by_full_sha =
            supervisor_tool_call("f1", FINALIZE_TOOL, serde_json::json!({ "checkpoint": c1 }));
        let parsed = parse_finalize(&by_full_sha, &ctx).expect("full sha resolves");
        assert_eq!(parsed.checkpoint, CheckpointId::Worker(1));

        let by_short_sha = supervisor_tool_call(
            "f2",
            FINALIZE_TOOL,
            serde_json::json!({ "checkpoint": &c1[..10] }),
        );
        let parsed = parse_finalize(&by_short_sha, &ctx).expect("short sha resolves");
        assert_eq!(
            parsed.checkpoint,
            CheckpointId::Worker(1),
            "finalize by w1's own commit hash must equal finalize by \"w1\""
        );

        let unknown = supervisor_tool_call(
            "f3",
            FINALIZE_TOOL,
            serde_json::json!({ "checkpoint": "not-a-known-ref" }),
        );
        assert!(parse_finalize(&unknown, &ctx).is_err());
    }

    #[test]
    fn git_refusal_blocks_gc_prune_and_reflog_expire_but_allows_reflog_show() {
        let refuses = |args: &[&str]| {
            git_refusal(&args.iter().map(|arg| arg.to_string()).collect::<Vec<_>>())
        };

        assert!(
            refuses(&["gc"])
                .expect("gc refused")
                .contains("refused: gc")
        );
        assert!(
            refuses(&["--no-pager", "prune"])
                .expect("prune refused")
                .contains("refused: prune")
        );
        assert!(
            refuses(&["reflog", "expire", "--all"])
                .expect("reflog expire refused")
                .contains("refused: reflog expire")
        );
        assert!(refuses(&["reflog", "show"]).is_none());
        assert!(refuses(&["log", "--oneline"]).is_none());
        assert!(refuses(&["merge", "--no-ff", "deadbeef"]).is_none());
        assert!(refuses(&[]).is_none());
    }

    #[test]
    fn parse_git_args_preserves_argv_tokens_without_shell_join() {
        let call = supervisor_tool_call(
            "git",
            GIT_TOOL,
            serde_json::json!({ "args": ["commit", "-m", "hello world; rm -rf /"] }),
        );

        let args = parse_git_args(&call).expect("valid args");

        // A single argv token survives untouched: no shell ever gets to
        // reinterpret the `;` as a command separator.
        assert_eq!(
            args,
            vec![
                "commit".to_string(),
                "-m".to_string(),
                "hello world; rm -rf /".to_string()
            ]
        );

        let bad = supervisor_tool_call("git", GIT_TOOL, serde_json::json!({ "args": [1, 2] }));
        assert!(parse_git_args(&bad).is_err());
    }

    #[test]
    fn view_parser_enforces_handle_array_and_cap() {
        let call = supervisor_tool_call(
            "view",
            VIEW_TOOL_CALL_TOOL,
            serde_json::json!({ "handles": ["w1m1", "w2m3"] }),
        );
        assert_eq!(
            parse_view_tool_call(&call).expect("valid handles"),
            vec!["w1m1".to_string(), "w2m3".to_string()]
        );

        let too_many = supervisor_tool_call(
            "view",
            VIEW_TOOL_CALL_TOOL,
            serde_json::json!({ "handles": vec!["x"; ASGARD_VIEW_TOOL_CALL_MAX_HANDLES + 1] }),
        );
        assert!(
            parse_view_tool_call(&too_many)
                .expect_err("too many")
                .contains("at most")
        );
    }

    #[test]
    fn update_plan_parser_reports_schema_errors() {
        let call = supervisor_tool_call(
            "plan",
            UPDATE_PLAN_TOOL,
            serde_json::json!({ "plan": [{ "step": "missing status" }] }),
        );

        assert!(
            parse_update_plan(&call)
                .expect_err("invalid plan")
                .contains("Invalid update_plan arguments")
        );
    }

    #[test]
    fn permanent_record_retains_small_view_and_git_payloads_whole() {
        let view_call = supervisor_tool_call(
            "view",
            VIEW_TOOL_CALL_TOOL,
            serde_json::json!({ "handles": ["w1m1", "w2m3"] }),
        );
        let spawn_call = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({ "workers": [{ "from": "root", "instructions": "x" }] }),
        );
        let git_call =
            supervisor_tool_call("git", GIT_TOOL, serde_json::json!({ "args": ["log"] }));
        let view_payload = "PASS (12) FAIL (0)".to_string();
        let git_payload = "commit abc123\n    initial\n".to_string();
        let transcript = vec![
            ChatMessage::assistant_tool_calls(vec![view_call, spawn_call, git_call]),
            ChatMessage::tool_result("view", VIEW_TOOL_CALL_TOOL, view_payload.clone()),
            ChatMessage::tool_result("spawn", SPAWN_WORKERS_TOOL, "spawned w3 from root"),
            ChatMessage::tool_result("git", GIT_TOOL, git_payload.clone()),
        ];

        let permanent = retain_payloads_for_permanent_record(&transcript);
        let text = permanent
            .iter()
            .map(ChatMessage::content_text)
            .collect::<Vec<_>>()
            .join("\n");

        // Payloads at or under the cap are retained verbatim - not collapsed
        // to a "[viewed ...: N bytes]" stub.
        assert!(text.contains(&view_payload));
        assert!(!text.contains("[viewed"));
        assert!(text.contains(&git_payload));
        assert!(!text.contains("[retained"));
        // Non-view/git tool results are untouched.
        assert!(text.contains("spawned w3 from root"));
    }

    #[test]
    fn permanent_record_truncates_oversized_payloads_at_the_retention_cap() {
        let view_call = supervisor_tool_call(
            "view",
            VIEW_TOOL_CALL_TOOL,
            serde_json::json!({ "handles": ["w1m1"] }),
        );
        let oversized = "x".repeat(RETAINED_PAYLOAD_CAP + 500);
        let transcript = vec![
            ChatMessage::assistant_tool_calls(vec![view_call]),
            ChatMessage::tool_result("view", VIEW_TOOL_CALL_TOOL, oversized.clone()),
        ];

        let permanent = retain_payloads_for_permanent_record(&transcript);
        let text = permanent
            .iter()
            .map(ChatMessage::content_text)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(!text.contains(&oversized));
        assert!(text.contains(&"x".repeat(RETAINED_PAYLOAD_CAP)));
        assert!(text.contains(&format!(
            "[retained first {RETAINED_PAYLOAD_CAP} bytes of {}]",
            oversized.len()
        )));
    }

    #[test]
    fn supervisor_supplement_requires_numbered_ambiguity_resolutions() {
        let supplement = supervisor_supplement();

        assert!(!supplement.contains("numbered obligations ledger"));
        assert!(supplement.contains("For each numbered ambiguity (A1, A2, ...)"));
        assert!(supplement.contains("state your resolution and a one-line rationale"));
        assert!(supplement.contains("one test per resolution"));
        assert!(supplement.contains("If independent parts share a file"));
        assert!(supplement.contains("Enumerate the separable groups of files"));
        assert!(supplement.contains("two workers in that same batch write spec tests"));
        // The supervisor now pins the spec itself: no upstream intake pass
        // hands it contracts, so the prompt must never promise one.
        assert!(!supplement.contains("intake"));
        assert!(
            supplement.contains("Calling conventions and argument order are always ambiguities")
        );
        assert!(supplement.contains("A plant is only evidence after re-checking"));
        assert!(supplement.contains("code intelligence"));
        assert!(supplement.contains("cites the exact functions"));
    }

    #[test]
    fn supervisor_supplement_teaches_branching_via_few_shot_and_reviews_law() {
        let supplement = supervisor_supplement();

        assert!(supplement.contains(
            "Example - branching review: worker w4 reports \"parser change in; 3 edge tests \
             failing; cause unclear - either tokenizer state reset or the quoting rule\". \
             Correct resolution: spawn two workers from w4 concurrently - one per hypothesis \
             - and discard the loser."
        ));
        assert!(supplement.contains(
            "Counter-example - serial is right: worker w9 reports \"rename applied; one import \
             path stale, fix is mechanical\". Correct resolution: one worker from w9; branching \
             adds nothing when only one continuation is live."
        ));
        assert!(supplement.contains(
            "When a review leaves two plausible continuations - two fix strategies, two \
             readings, fix-forward versus a fresh start from an earlier checkpoint - spawn \
             both concurrently instead of trying them one at a time; serial retries of \
             guesses are the most expensive habit here."
        ));
    }

    #[test]
    fn supervisor_supplement_teaches_prefinalize_coverage() {
        let supplement = supervisor_supplement();

        assert!(supplement.contains("mark your full-suite worker with runs_full_suite"));
        assert!(supplement.contains("broadest attack"));
        assert!(supplement.contains("or state full_suite_skipped"));
        assert!(supplement.contains("Refutations name attacks, not reassurance"));
        assert!(supplement.contains("Dictated code is a belief too"));
    }

    #[test]
    fn prefinalize_tool_docs_require_real_single_plants_and_hang_diagnosis() {
        let tools = supervisor_tool_definitions(&["model-a".to_string()]);
        let prefinalize = tools
            .iter()
            .find(|tool| tool.function.name == PREFINALIZE_TOOL)
            .expect("prefinalize tool");
        let description = &prefinalize.function.description;

        assert!(description.contains("spawn one worker per plant"));
        assert!(description.contains("real mutation of the SHIPPED implementation"));
        assert!(description.contains("confirmed by inspecting the diff"));
        assert!(description.contains("hung or timed-out verification run is a blocker"));
        assert!(description.contains("Never substitute a weaker ad-hoc check"));
    }

    #[test]
    fn bifrost_tool_defs_preserve_names_descriptions_schemas_and_order() {
        let tools = vec![
            McpToolDef {
                name: "search_symbols".to_string(),
                description: "find symbols by pattern".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": { "patterns": { "type": "array" } },
                    "required": ["patterns"],
                }),
                annotations: crate::mcp::McpToolAnnotations {
                    read_only_hint: Some(true),
                },
            },
            McpToolDef {
                name: "activate_workspace".to_string(),
                description: "point bifrost at another path".to_string(),
                input_schema: serde_json::json!({ "type": "object" }),
                annotations: crate::mcp::McpToolAnnotations::default(),
            },
        ];

        let defs = mcp_tools_to_definitions(&tools);

        assert_eq!(defs.len(), 2);
        // Client order is preserved verbatim (cache stability).
        assert_eq!(defs[0].function.name, "search_symbols");
        assert_eq!(defs[1].function.name, "activate_workspace");
        assert_eq!(defs[0].r#type, "function");
        assert_eq!(defs[0].function.description, "find symbols by pattern");
        // input_schema passes through unchanged into `parameters`.
        assert_eq!(defs[0].function.parameters, tools[0].input_schema);
        assert_eq!(defs[1].function.parameters, tools[1].input_schema);
    }

    #[test]
    fn permanent_record_retains_bifrost_results_and_still_keeps_view_and_git() {
        let bifrost_call = supervisor_tool_call(
            "b1",
            "search_symbols",
            serde_json::json!({ "patterns": ["X"] }),
        );
        let view_call = supervisor_tool_call(
            "view",
            VIEW_TOOL_CALL_TOOL,
            serde_json::json!({ "handles": ["w1m1"] }),
        );
        let git_call =
            supervisor_tool_call("git", GIT_TOOL, serde_json::json!({ "args": ["log"] }));
        let spawn_call = supervisor_tool_call(
            "spawn",
            SPAWN_WORKERS_TOOL,
            serde_json::json!({ "workers": [{ "from": "root", "instructions": "x" }] }),
        );

        let bifrost_payload = "fn foo() at src/lib.rs:12\nfn foo_bar() at src/x.rs:3".to_string();
        let view_payload = "PASS (12) FAIL (0)".to_string();
        let git_payload = "commit abc123\n    initial\n".to_string();
        let transcript = vec![
            ChatMessage::assistant_tool_calls(vec![bifrost_call, view_call, git_call, spawn_call]),
            ChatMessage::tool_result("b1", "search_symbols", bifrost_payload.clone()),
            ChatMessage::tool_result("view", VIEW_TOOL_CALL_TOOL, view_payload.clone()),
            ChatMessage::tool_result("git", GIT_TOOL, git_payload.clone()),
            ChatMessage::tool_result("spawn", SPAWN_WORKERS_TOOL, "spawned w3 from root"),
        ];

        let permanent = retain_payloads_for_permanent_record(&transcript);
        let text = permanent
            .iter()
            .map(ChatMessage::content_text)
            .collect::<Vec<_>>()
            .join("\n");

        // Bifrost code-intelligence output is retained as evidence.
        assert!(text.contains(&bifrost_payload));
        // view_tool_call and git are still retained.
        assert!(text.contains(&view_payload));
        assert!(text.contains(&git_payload));
        // Action-tool status strings still pass through.
        assert!(text.contains("spawned w3 from root"));
        // Nothing under the cap gets a truncation marker.
        assert!(!text.contains("[retained"));
    }

    #[test]
    fn permanent_record_caps_oversized_bifrost_results() {
        let bifrost_call =
            supervisor_tool_call("b1", "usage_graph", serde_json::json!({ "symbol": "X" }));
        let oversized = "y".repeat(RETAINED_PAYLOAD_CAP + 500);
        let transcript = vec![
            ChatMessage::assistant_tool_calls(vec![bifrost_call]),
            ChatMessage::tool_result("b1", "usage_graph", oversized.clone()),
        ];

        let permanent = retain_payloads_for_permanent_record(&transcript);
        let text = permanent
            .iter()
            .map(ChatMessage::content_text)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(!text.contains(&oversized));
        assert!(text.contains(&"y".repeat(RETAINED_PAYLOAD_CAP)));
        assert!(text.contains(&format!(
            "[retained first {RETAINED_PAYLOAD_CAP} bytes of {}]",
            oversized.len()
        )));
    }
}
