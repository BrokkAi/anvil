use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;

use anyhow::{Context, Result, anyhow, bail, ensure};

use crate::llm_client::{ChatMessage, TokenUsage};

/// A successfully expanded tool call, kept structured so the same resolution
/// can be rendered in full for the model and summarized for the permanent record.
#[derive(Clone, Debug)]
pub(crate) struct ResolvedView {
    pub(crate) worker: usize,
    pub(crate) name: String,
    pub(crate) arguments: String,
    pub(crate) result: String,
}

fn in_flight_error(worker: usize) -> String {
    format!(
        "w{worker} is running; its results become viewable when its batch is presented for review."
    )
}

pub(crate) fn render_resolved_views(
    views: &[(String, std::result::Result<ResolvedView, String>)],
) -> String {
    let mut rendered = String::new();
    for (handle, view) in views {
        match view {
            Ok(view) => rendered.push_str(&format!(
                "<tool_call id=\"{handle}\" worker=\"w{}\" name=\"{}\">\n\
                 <arguments>{}</arguments>\n\
                 <result>{}</result>\n\
                 </tool_call>\n",
                view.worker, view.name, view.arguments, view.result,
            )),
            Err(error) => rendered.push_str(&format!(
                "<tool_call handle=\"{handle}\" error=\"{error}\" />\n"
            )),
        }
    }
    rendered
}

#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub(crate) enum CheckpointId {
    Root,
    Worker(usize),
    /// A commit that descends from the run's base commit but was never
    /// wrapped in a `TrajectoryNode` (e.g. a merge produced directly by the
    /// supervisor's `git` tool). Only ever constructed by
    /// [`TrajectoryDag::resolve_checkpoint_by_commit`], which has already
    /// verified the commit exists and descends from the base commit.
    Commit(String),
}

impl CheckpointId {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        if value == "root" {
            return Some(Self::Root);
        }
        let worker = value.strip_prefix('w')?;
        if worker.is_empty() || !worker.chars().all(|character| character.is_ascii_digit()) {
            return None;
        }
        Some(Self::Worker(worker.parse().ok()?))
    }
}

impl fmt::Display for CheckpointId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Root => formatter.write_str("root"),
            Self::Worker(worker) => write!(formatter, "w{worker}"),
            Self::Commit(commit) => formatter.write_str(short_sha(commit)),
        }
    }
}

impl FromStr for CheckpointId {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value).ok_or_else(|| anyhow!("invalid checkpoint id {value:?}"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum WorkerStopReason {
    Finished,
    StepLimit,
    Failed(String),
    Cancelled,
}

impl WorkerStopReason {
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::Finished => "finished",
            Self::StepLimit => "step_limit",
            Self::Failed(_) => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TrajectoryWindow {
    pub(crate) worker: usize,
    pub(crate) parent: CheckpointId,
    pub(crate) instructions: String,
    pub(crate) model: String,
    pub(crate) instruction_message: ChatMessage,
    pub(crate) window_messages: Vec<ChatMessage>,
    pub(crate) compact: String,
    pub(crate) final_response: String,
    pub(crate) stop: WorkerStopReason,
    pub(crate) steps: usize,
    pub(crate) diffstat: String,
    pub(crate) usage: TokenUsage,
    pub(crate) elapsed_millis: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct TrajectoryNode {
    pub(crate) window: TrajectoryWindow,
    pub(crate) commit: String,
    pub(crate) merged_from: Vec<CheckpointId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OffLineageCheckpoint {
    pub(crate) checkpoint: CheckpointId,
    pub(crate) diffstat: String,
}

#[derive(Clone, Debug)]
pub(crate) struct DiscardedTombstone {
    pub(crate) parent: CheckpointId,
    pub(crate) instructions: String,
}

pub(crate) struct DagLiveEntry {
    pub(crate) worker: usize,
    pub(crate) parent: CheckpointId,
    pub(crate) status: String,
    pub(crate) instructions: String,
}

#[derive(Clone, Debug)]
pub(crate) struct TrajectoryDag {
    initial_messages: Vec<ChatMessage>,
    base_commit: String,
    git_root: Option<PathBuf>,
    nodes: BTreeMap<usize, TrajectoryNode>,
    discarded: BTreeMap<usize, DiscardedTombstone>,
}

impl TrajectoryDag {
    #[cfg(test)]
    pub(crate) fn new(initial_messages: Vec<ChatMessage>, base_commit: String) -> Self {
        Self {
            initial_messages,
            base_commit,
            git_root: None,
            nodes: BTreeMap::new(),
            discarded: BTreeMap::new(),
        }
    }

    pub(crate) fn new_with_git_root(
        initial_messages: Vec<ChatMessage>,
        base_commit: String,
        git_root: PathBuf,
    ) -> Self {
        Self {
            initial_messages,
            base_commit,
            git_root: Some(git_root),
            nodes: BTreeMap::new(),
            discarded: BTreeMap::new(),
        }
    }

    pub(crate) fn insert(&mut self, node: TrajectoryNode) -> Result<()> {
        let worker = node.window.worker;
        ensure!(
            !self.nodes.contains_key(&worker),
            "worker {worker} is already saved"
        );
        ensure!(
            !self.discarded.contains_key(&worker),
            "worker {worker} was already discarded"
        );
        ensure!(
            self.contains(&node.window.parent),
            "parent checkpoint {} does not exist",
            node.window.parent
        );
        self.nodes.insert(worker, node);
        Ok(())
    }

    pub(crate) fn discard(
        &mut self,
        worker: usize,
        parent: CheckpointId,
        instructions: String,
    ) -> Result<()> {
        ensure!(
            !self.nodes.contains_key(&worker),
            "worker {worker} is already saved"
        );
        ensure!(
            !self.discarded.contains_key(&worker),
            "worker {worker} was already discarded"
        );
        ensure!(
            self.contains(&parent),
            "parent checkpoint {parent} does not exist"
        );
        self.discarded.insert(
            worker,
            DiscardedTombstone {
                parent,
                instructions,
            },
        );
        Ok(())
    }

    pub(crate) fn contains(&self, ckpt: &CheckpointId) -> bool {
        match ckpt {
            CheckpointId::Root => true,
            CheckpointId::Worker(worker) => self.nodes.contains_key(worker),
            // Only ever constructed after resolve_checkpoint_by_commit has
            // already verified the commit exists and descends from base.
            CheckpointId::Commit(_) => true,
        }
    }

    pub(crate) fn commit_for<'a>(&'a self, ckpt: &'a CheckpointId) -> Option<&'a str> {
        match ckpt {
            CheckpointId::Root => Some(&self.base_commit),
            CheckpointId::Worker(worker) => self.nodes.get(worker).map(|node| node.commit.as_str()),
            CheckpointId::Commit(commit) => Some(commit.as_str()),
        }
    }

    /// Resolves a supervisor-supplied string that is not "root"/"wN" to a
    /// checkpoint by treating it as a git commit reference (short or full
    /// sha, or any other rev-parse-able ref). The commit must exist in the
    /// parent repo and must descend from this run's base commit. If it
    /// happens to equal an already-known checkpoint's commit, that
    /// checkpoint is returned directly (no wrapper needed); otherwise it is
    /// wrapped as `CheckpointId::Commit` so it can still be finalized.
    pub(crate) fn resolve_checkpoint_by_commit(
        &self,
        value: &str,
    ) -> std::result::Result<CheckpointId, String> {
        let root = self
            .git_root
            .as_deref()
            .ok_or_else(|| "commit-hash checkpoints require a git-backed Asgard run".to_string())?;
        let full = git_rev_parse_commit(root, value).map_err(|error| {
            format!("{value:?} is not a known checkpoint or a valid commit ({error:#})")
        })?;
        if full == self.base_commit {
            return Ok(CheckpointId::Root);
        }
        if let Some(worker) = self
            .nodes
            .iter()
            .find(|(_, node)| node.commit == full)
            .map(|(worker, _)| *worker)
        {
            return Ok(CheckpointId::Worker(worker));
        }
        match git_is_ancestor(root, &self.base_commit, &full) {
            Ok(true) => Ok(CheckpointId::Commit(full)),
            Ok(false) => Err(format!(
                "commit {value} is not a descendant of this run's base commit"
            )),
            Err(error) => Err(format!(
                "failed to verify commit {value} ancestry: {error:#}"
            )),
        }
    }

    /// Walks first-parent ancestry from `sha` until it reaches a commit that
    /// is either this run's base commit or an already-saved checkpoint's
    /// commit, returning that checkpoint. Used only for
    /// `CheckpointId::Commit` fragments (a commit made directly by the
    /// supervisor's `git` tool, never wrapped in a `TrajectoryNode`).
    fn nearest_known_checkpoint(&self, root: &Path, sha: &str) -> Result<CheckpointId> {
        let known: HashMap<&str, CheckpointId> =
            std::iter::once((self.base_commit.as_str(), CheckpointId::Root))
                .chain(self.nodes.values().map(|node| {
                    (
                        node.commit.as_str(),
                        CheckpointId::Worker(node.window.worker),
                    )
                }))
                .collect();
        let output = Command::new("git")
            .args(["log", "--first-parent", "--format=%H", sha])
            .current_dir(root)
            .output()?;
        ensure!(
            output.status.success(),
            "git log --first-parent {sha} failed in {}",
            root.display()
        );
        let history = String::from_utf8(output.stdout).context("git log output was not UTF-8")?;
        for commit in history.lines().skip(1) {
            if let Some(checkpoint) = known.get(commit) {
                return Ok(checkpoint.clone());
            }
        }
        bail!(
            "commit {sha} does not descend from any known Asgard checkpoint via first-parent ancestry"
        );
    }

    pub(crate) fn node(&self, worker: usize) -> Option<&TrajectoryNode> {
        self.nodes.get(&worker)
    }

    pub(crate) fn is_discarded(&self, worker: usize) -> bool {
        self.discarded.contains_key(&worker)
    }

    pub(crate) fn checkpoint_labels(&self) -> Vec<String> {
        self.nodes
            .keys()
            .map(|worker| CheckpointId::Worker(*worker).to_string())
            .collect()
    }

    pub(crate) fn is_ancestor_of(&self, ancestor: &CheckpointId, target: &CheckpointId) -> bool {
        if ancestor == target {
            return true;
        }
        let (Some(root), Some(ancestor_commit), Some(target_commit)) = (
            self.git_root.as_deref(),
            self.commit_for(ancestor),
            self.commit_for(target),
        ) else {
            return false;
        };
        match git_is_ancestor(root, ancestor_commit, target_commit) {
            Ok(is_ancestor) => is_ancestor,
            Err(error) => {
                tracing::warn!(
                    ancestor = %ancestor,
                    target = %target,
                    "failed to check Asgard checkpoint ancestry via git: {error:#}"
                );
                false
            }
        }
    }

    pub(crate) fn off_lineage_checkpoints_with_diffstat(
        &self,
        target: &CheckpointId,
    ) -> Vec<OffLineageCheckpoint> {
        self.nodes
            .iter()
            .filter_map(|(worker, node)| {
                let checkpoint = CheckpointId::Worker(*worker);
                (!self.is_ancestor_of(&checkpoint, target)
                    && !node.window.diffstat.trim().is_empty())
                .then(|| OffLineageCheckpoint {
                    checkpoint,
                    diffstat: node.window.diffstat.clone(),
                })
            })
            .collect()
    }

    pub(crate) fn ancestor_messages(&self, ckpt: &CheckpointId) -> Result<Vec<ChatMessage>> {
        if let CheckpointId::Commit(sha) = ckpt {
            let root = self
                .git_root
                .as_deref()
                .ok_or_else(|| anyhow!("no git root available to resolve commit {sha}"))?;
            let nearest = self.nearest_known_checkpoint(root, sha)?;
            let mut messages = self.ancestor_messages(&nearest)?;
            messages.push(ChatMessage::user(format!(
                "You start from commit {}, created by the supervisor with git (e.g. a merge); \
                 it descends from {nearest}.",
                short_sha(sha)
            )));
            return Ok(messages);
        }
        if !self.contains(ckpt) {
            bail!("unknown checkpoint {ckpt}");
        }

        let mut worker_ids = Vec::new();
        let mut current = ckpt.clone();
        let mut visited = HashSet::new();
        for _ in 0..=self.nodes.len() {
            match current {
                CheckpointId::Root => break,
                CheckpointId::Worker(worker) => {
                    ensure!(visited.insert(worker), "cycle detected at worker {worker}");
                    let node = self
                        .nodes
                        .get(&worker)
                        .ok_or_else(|| anyhow!("unknown checkpoint w{worker}"))?;
                    worker_ids.push(worker);
                    current = node.window.parent.clone();
                }
                CheckpointId::Commit(sha) => {
                    bail!("checkpoint chain reached an unresolved commit fragment {sha}");
                }
            }
        }
        ensure!(
            matches!(current, CheckpointId::Root),
            "ancestor walk exceeded saved node count"
        );

        let mut messages = self.initial_messages.clone();
        for worker in worker_ids.into_iter().rev() {
            let node = self
                .nodes
                .get(&worker)
                .ok_or_else(|| anyhow!("unknown checkpoint w{worker}"))?;
            messages.push(node.window.instruction_message.clone());
            messages.extend(node.window.window_messages.clone());
        }
        Ok(messages)
    }

    /// Resolve each handle independently, so callers can render the full
    /// expansion for the model and a per-handle summary for the permanent
    /// record from the same resolution.
    pub(crate) fn resolve_handle_views(
        &self,
        handles: &[String],
        pending: &[(usize, &[ChatMessage])],
        in_flight: &[usize],
    ) -> Vec<(String, std::result::Result<ResolvedView, String>)> {
        handles
            .iter()
            .map(|handle| {
                (
                    handle.clone(),
                    self.resolve_handle(handle, pending, in_flight),
                )
            })
            .collect()
    }

    fn resolve_handle(
        &self,
        handle: &str,
        pending: &[(usize, &[ChatMessage])],
        in_flight: &[usize],
    ) -> std::result::Result<ResolvedView, String> {
        let Some((worker, index)) = crate::asgard::parse_worker_tool_handle(handle) else {
            // A bare checkpoint id ("w11") is a natural but wrong way to
            // ask for a whole trajectory; answer the intent instead of
            // calling it malformed.
            return Err(match CheckpointId::parse(handle) {
                Some(CheckpointId::Worker(worker)) if self.nodes.contains_key(&worker) => format!(
                    "w{worker} is a saved checkpoint, not a tool-result handle; its compact trace was shown at its review. Expand a specific result with a handle like w{worker}m4."
                ),
                Some(CheckpointId::Worker(worker)) if in_flight.contains(&worker) => {
                    in_flight_error(worker)
                }
                Some(CheckpointId::Worker(worker)) if self.discarded.contains_key(&worker) => {
                    format!("trajectory w{worker} was discarded; its full results are gone")
                }
                _ => "malformed handle".to_string(),
            });
        };

        let messages = if let Some((_, pending_messages)) = pending
            .iter()
            .find(|(pending_worker, _)| *pending_worker == worker)
        {
            Some(*pending_messages)
        } else {
            self.nodes
                .get(&worker)
                .map(|node| node.window.window_messages.as_slice())
        };
        let Some(messages) = messages else {
            return Err(if self.discarded.contains_key(&worker) {
                "trajectory was discarded; its full results are gone".to_string()
            } else if in_flight.contains(&worker) {
                in_flight_error(worker)
            } else {
                "unknown worker".to_string()
            });
        };
        let result = messages
            .get(index)
            .filter(|message| message.role == "tool")
            .ok_or_else(|| "handle does not name a tool result".to_string())?;

        let call = crate::asgard::originating_tool_call(messages, index);
        Ok(ResolvedView {
            worker,
            name: call
                .map_or("tool", |call| call.function.name.as_str())
                .to_string(),
            arguments: call
                .map_or("", |call| call.function.arguments.as_str())
                .to_string(),
            result: result.content_text().to_string(),
        })
    }
}

fn git_is_ancestor(root: &Path, ancestor_commit: &str, target_commit: &str) -> Result<bool> {
    let output = Command::new("git")
        .args([
            "merge-base",
            "--is-ancestor",
            ancestor_commit,
            target_commit,
        ])
        .current_dir(root)
        .output()?;
    if output.status.success() {
        return Ok(true);
    }
    if output.status.code() == Some(1) {
        return Ok(false);
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    bail!(
        "git merge-base --is-ancestor {ancestor_commit} {target_commit} failed in {} with status {}; stderr:\n{}",
        root.display(),
        output.status,
        stderr.trim()
    );
}

/// Renders a trajectory fragment for supervisor review. `commit` is this
/// trajectory's own checkpoint commit, if it has one; in practice a
/// trajectory under review has not been snapshotted yet (that happens at
/// save/spawn decision time), so callers pass `None` here today. `parent_commit`
/// is the checkpoint it forked from, which is already known whenever the
/// parent is root or an already-saved checkpoint.
pub(crate) fn render_fragment(
    window: &TrajectoryWindow,
    commit: Option<&str>,
    parent_commit: Option<&str>,
) -> String {
    let mut rendered = String::new();
    rendered.push_str(&format!("<worker_trajectory id=\"w{}\"", window.worker));
    if let Some(commit) = commit {
        rendered.push_str(&format!(" commit=\"{}\"", short_sha(commit)));
    }
    rendered.push_str(&format!(" continues_from=\"{}\"", window.parent));
    if let Some(parent_commit) = parent_commit {
        rendered.push_str(&format!(" parent_commit=\"{}\"", short_sha(parent_commit)));
    }
    rendered.push_str(&format!(
        " model=\"{}\" stop=\"{}\" steps=\"{}\">\n",
        escape_attribute(&window.model),
        window.stop.label(),
        window.steps
    ));
    rendered.push_str(&format!(
        "<instructions>{}</instructions>\n",
        escape_content(&window.instructions)
    ));
    if !window.diffstat.is_empty() {
        rendered.push_str(&format!(
            "<diffstat>{}</diffstat>\n",
            escape_content(&window.diffstat)
        ));
    }
    rendered.push_str(&format!(
        "<runtime elapsed_millis=\"{}\" input_tokens=\"{}\" output_tokens=\"{}\" thought_tokens=\"{}\" cached_read_tokens=\"{}\" cached_write_tokens=\"{}\" />\n",
        window.elapsed_millis,
        window.usage.input_tokens,
        window.usage.output_tokens,
        window.usage.thought_tokens,
        window.usage.cached_read_tokens,
        window.usage.cached_write_tokens
    ));
    if let WorkerStopReason::Failed(message) = &window.stop {
        rendered.push_str(&format!("<failure>{}</failure>\n", escape_content(message)));
    }
    rendered.push_str("<window_trajectory>\n");
    rendered.push_str(&window.compact);
    rendered.push_str("</window_trajectory>\n");
    if window.final_response.is_empty() {
        rendered.push_str("<final_response none=\"true\" />\n");
    } else {
        rendered.push_str("<final_response>");
        rendered.push_str(&window.final_response);
        rendered.push_str("</final_response>\n");
    }
    rendered.push_str(
        "This trajectory must be saved, spawned from, or discarded before you end your turn.\n",
    );
    rendered.push_str("</worker_trajectory>\n");
    rendered
}

pub(crate) fn render_dag_overview(dag: &TrajectoryDag, live: &[DagLiveEntry]) -> String {
    let mut rendered = String::from("root\n");
    render_dag_children(dag, live, &CheckpointId::Root, "", &mut rendered);
    rendered
}

enum DagOverviewChild<'a> {
    Saved(&'a TrajectoryNode),
    Discarded(&'a DiscardedTombstone),
    Live(&'a DagLiveEntry),
}

fn render_dag_children(
    dag: &TrajectoryDag,
    live: &[DagLiveEntry],
    parent: &CheckpointId,
    prefix: &str,
    rendered: &mut String,
) {
    let mut children = Vec::new();
    for node in dag.nodes.values() {
        if &node.window.parent == parent {
            children.push((node.window.worker, DagOverviewChild::Saved(node)));
        }
    }
    for (worker, tombstone) in &dag.discarded {
        if &tombstone.parent == parent {
            children.push((*worker, DagOverviewChild::Discarded(tombstone)));
        }
    }
    for entry in live {
        if &entry.parent == parent {
            children.push((entry.worker, DagOverviewChild::Live(entry)));
        }
    }
    children.sort_by_key(|(worker, _)| *worker);

    let child_count = children.len();
    for (index, (worker, child)) in children.into_iter().enumerate() {
        let is_last = index + 1 == child_count;
        rendered.push_str(prefix);
        rendered.push_str(if is_last { "└─ " } else { "├─ " });
        match child {
            DagOverviewChild::Saved(node) => {
                rendered.push_str(&format!(
                    "w{worker} ({}) \"{}\" saved, {}/{} steps{}{}\n",
                    short_sha(&node.commit),
                    instruction_stub(&node.window.instructions),
                    node.window.stop.label(),
                    node.window.steps,
                    compact_diffstat_suffix(&node.window.diffstat),
                    compact_merged_from_suffix(&node.merged_from)
                ));
                let mut next_prefix = prefix.to_string();
                next_prefix.push_str(if is_last { "   " } else { "│  " });
                render_dag_children(
                    dag,
                    live,
                    &CheckpointId::Worker(worker),
                    &next_prefix,
                    rendered,
                );
            }
            DagOverviewChild::Discarded(tombstone) => {
                rendered.push_str(&format!(
                    "w{worker} \"{}\" discarded\n",
                    instruction_stub(&tombstone.instructions)
                ));
            }
            DagOverviewChild::Live(entry) => {
                rendered.push_str(&format!(
                    "w{worker} \"{}\" {}\n",
                    instruction_stub(&entry.instructions),
                    entry.status
                ));
            }
        }
    }
}

pub(crate) fn short_sha(commit: &str) -> &str {
    commit.get(..7).unwrap_or(commit)
}

/// Resolves `value` (short or full sha, or any other git ref) to a full
/// commit sha, failing if it doesn't name a commit object.
fn git_rev_parse_commit(root: &Path, value: &str) -> Result<String> {
    let output = Command::new("git")
        .args([
            "rev-parse",
            "--verify",
            "-q",
            &format!("{value}^{{commit}}"),
        ])
        .current_dir(root)
        .output()?;
    ensure!(output.status.success(), "not a valid commit");
    Ok(String::from_utf8(output.stdout)
        .context("git rev-parse output was not UTF-8")?
        .trim()
        .to_string())
}

fn compact_diffstat_suffix(diffstat: &str) -> String {
    let text = diffstat
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("; ");
    if text.is_empty() {
        String::new()
    } else {
        format!("; diffstat: {text}")
    }
}

fn compact_merged_from_suffix(merged_from: &[CheckpointId]) -> String {
    if merged_from.is_empty() {
        String::new()
    } else {
        format!(
            "; merged from: {}",
            merged_from
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

fn instruction_stub(value: &str) -> String {
    let text = value.replace('\n', " ");
    if text.chars().count() <= 60 {
        text
    } else {
        text.chars().take(60).collect()
    }
}

fn escape_content(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn escape_attribute(value: &str) -> String {
    escape_content(value).replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_client::{FunctionCall, ToolCall};
    use std::fs;

    fn call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            r#type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    fn assistant_call(call: ToolCall) -> ChatMessage {
        ChatMessage::assistant_tool_calls_with_content_and_reasoning(
            String::new(),
            vec![call],
            None,
        )
    }

    fn window(worker: usize, parent: CheckpointId, label: &str) -> TrajectoryWindow {
        let call_id = format!("c{worker}");
        let messages = vec![
            assistant_call(call(
                &call_id,
                "run_shell_command",
                &format!(r#"{{"command":"echo {label}"}}"#),
            )),
            ChatMessage::tool_result(&call_id, "run_shell_command", format!("result {label}")),
            ChatMessage::assistant(format!("final {label}")),
        ];
        TrajectoryWindow {
            worker,
            parent,
            instructions: format!("instructions {label}"),
            model: "model-a".to_string(),
            instruction_message: ChatMessage::user(format!("instruction message {label}")),
            compact: crate::asgard::render_window_compact_for_worker(worker, &messages),
            window_messages: messages,
            final_response: format!("final {label}"),
            stop: WorkerStopReason::Finished,
            steps: 2,
            diffstat: String::new(),
            usage: TokenUsage::default(),
            elapsed_millis: 0,
        }
    }

    fn node(worker: usize, parent: CheckpointId, label: &str, commit: &str) -> TrajectoryNode {
        TrajectoryNode {
            window: window(worker, parent, label),
            commit: commit.to_string(),
            merged_from: Vec::new(),
        }
    }

    fn node_with_diffstat(
        worker: usize,
        parent: CheckpointId,
        label: &str,
        commit: &str,
        diffstat: &str,
    ) -> TrajectoryNode {
        let mut node = node(worker, parent, label, commit);
        node.window.diffstat = diffstat.to_string();
        node
    }

    fn text(message: &ChatMessage) -> String {
        message.content_text()
    }

    fn run_git(root: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .expect("run git");
        assert!(
            output.status.success(),
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).expect("git stdout utf8")
    }

    fn commit_file(repo: &Path, name: &str, content: &str, message: &str) -> String {
        fs::write(repo.join(name), content).expect("write commit file");
        run_git(repo, &["add", name]);
        run_git(repo, &["commit", "--quiet", "-m", message]);
        run_git(repo, &["rev-parse", "HEAD"]).trim().to_string()
    }

    #[test]
    fn checkpoint_ids_parse_and_display_exactly() {
        assert_eq!(CheckpointId::parse("root"), Some(CheckpointId::Root));
        assert_eq!(CheckpointId::parse("w12"), Some(CheckpointId::Worker(12)));
        assert_eq!(CheckpointId::Worker(3).to_string(), "w3");
        assert_eq!(CheckpointId::parse("w"), None);
        assert_eq!(CheckpointId::parse("w3x"), None);
        assert_eq!(CheckpointId::parse("Root"), None);
    }

    #[test]
    fn ancestor_messages_follow_requested_branch() {
        let mut dag = TrajectoryDag::new(vec![ChatMessage::system("base")], "base".to_string());
        dag.insert(node(1, CheckpointId::Root, "one", "c1"))
            .unwrap();
        dag.insert(node(2, CheckpointId::Worker(1), "two", "c2"))
            .unwrap();
        dag.insert(node(3, CheckpointId::Worker(1), "three", "c3"))
            .unwrap();

        let w2 = dag.ancestor_messages(&CheckpointId::Worker(2)).unwrap();
        assert_eq!(
            w2.iter().map(text).collect::<Vec<_>>(),
            vec![
                "base",
                "instruction message one",
                "",
                "result one",
                "final one",
                "instruction message two",
                "",
                "result two",
                "final two",
            ]
        );

        let w3 = dag.ancestor_messages(&CheckpointId::Worker(3)).unwrap();
        assert_eq!(
            w3.iter().map(text).collect::<Vec<_>>(),
            vec![
                "base",
                "instruction message one",
                "",
                "result one",
                "final one",
                "instruction message three",
                "",
                "result three",
                "final three",
            ]
        );
        assert!(!w3.iter().any(|message| text(message) == "result two"));
    }

    #[test]
    fn insert_rejects_duplicate_discarded_and_unknown_parent() {
        let mut dag = TrajectoryDag::new(Vec::new(), "base".to_string());
        dag.insert(node(1, CheckpointId::Root, "one", "c1"))
            .unwrap();
        assert!(
            dag.insert(node(1, CheckpointId::Root, "dup", "c1b"))
                .is_err()
        );
        dag.discard(2, CheckpointId::Root, "discard two".to_string())
            .unwrap();
        assert!(
            dag.insert(node(2, CheckpointId::Root, "two", "c2"))
                .is_err()
        );
        assert!(
            dag.insert(node(3, CheckpointId::Worker(99), "three", "c3"))
                .is_err()
        );
        assert!(
            dag.discard(4, CheckpointId::Worker(99), "unknown parent".to_string())
                .is_err()
        );
    }

    #[test]
    fn resolve_handles_covers_saved_pending_and_errors() {
        let mut dag = TrajectoryDag::new(Vec::new(), "base".to_string());
        let mut saved = window(1, CheckpointId::Root, "saved");
        let long_result = "x".repeat(10_000);
        saved.window_messages = vec![
            assistant_call(call("long", "read_file", r#"{"file_path":"big.txt"}"#)),
            ChatMessage::tool_result("long", "read_file", long_result.clone()),
            ChatMessage::assistant("done"),
        ];
        saved.compact = crate::asgard::render_window_compact_for_worker(1, &saved.window_messages);
        dag.insert(TrajectoryNode {
            window: saved,
            commit: "c1".to_string(),
            merged_from: Vec::new(),
        })
        .unwrap();
        dag.discard(3, CheckpointId::Root, "discarded".to_string())
            .unwrap();

        let pending = vec![
            assistant_call(call("pending", "run_shell_command", r#"{"command":"pwd"}"#)),
            ChatMessage::tool_result("pending", "run_shell_command", "pending result"),
            ChatMessage::assistant("pending final"),
        ];
        let views = dag.resolve_handle_views(
            &[
                "w1m1".to_string(),
                "w2m1".to_string(),
                "w3m1".to_string(),
                "w4m1".to_string(),
                "w1l0m1".to_string(),
                "w1m2".to_string(),
                "w9m0".to_string(),
            ],
            &[(2, &pending)],
            &[9],
        );
        let rendered = render_resolved_views(&views);

        assert!(rendered.contains(&long_result));
        assert!(rendered.contains(r#"<tool_call id="w1m1" worker="w1" name="read_file">"#));
        assert!(rendered.contains(r#"<arguments>{"file_path":"big.txt"}</arguments>"#));
        assert!(rendered.contains(r#"<tool_call id="w2m1" worker="w2" name="run_shell_command">"#));
        assert!(rendered.contains("<result>pending result</result>"));
        assert!(rendered.contains(
            r#"<tool_call handle="w3m1" error="trajectory was discarded; its full results are gone" />"#
        ));
        assert!(rendered.contains(r#"<tool_call handle="w4m1" error="unknown worker" />"#));
        assert!(rendered.contains(
            r#"<tool_call handle="w9m0" error="w9 is running; its results become viewable when its batch is presented for review." />"#
        ));
        assert!(rendered.contains(r#"<tool_call handle="w1l0m1" error="malformed handle" />"#));
        assert!(
            rendered.contains(
                r#"<tool_call handle="w1m2" error="handle does not name a tool result" />"#
            )
        );
    }

    #[test]
    fn render_fragment_includes_verbatim_response_and_failure_details() {
        let mut trajectory = window(3, CheckpointId::Root, "fragment");
        trajectory.final_response = format!("{}<raw>&unescaped", "a".repeat(200));
        trajectory.diffstat = " src/lib.rs | 2 +".to_string();
        let rendered = render_fragment(&trajectory, None, None);
        assert!(rendered.contains(
            r#"<worker_trajectory id="w3" continues_from="root" model="model-a" stop="finished" steps="2">"#
        ));
        assert!(rendered.contains("<diffstat> src/lib.rs | 2 +</diffstat>"));
        assert!(rendered.contains(&trajectory.final_response));

        trajectory.final_response.clear();
        let empty = render_fragment(&trajectory, None, None);
        assert!(empty.contains(r#"<final_response none="true" />"#));

        trajectory.stop = WorkerStopReason::Failed("boom <bad>".to_string());
        let failed = render_fragment(&trajectory, None, None);
        assert!(failed.contains(r#"stop="failed""#));
        assert!(failed.contains("<failure>boom &lt;bad&gt;</failure>"));
    }

    #[test]
    fn render_fragment_header_gains_commit_and_parent_shas_when_available() {
        let trajectory = window(7, CheckpointId::Worker(3), "shorthash");

        // At review time the trajectory itself has not been snapshotted yet
        // (that happens at save/spawn decision time) - so its own commit is
        // never available here, but the parent's commit already is.
        let with_parent = render_fragment(&trajectory, None, Some("91b02dexxxxxxxxxxxxxxxxxxxx"));
        assert!(with_parent.contains(r#"id="w7""#));
        assert!(!with_parent.contains(r#" commit="91b02de""#));
        assert!(with_parent.contains(r#"continues_from="w3""#));
        assert!(with_parent.contains(r#"parent_commit="91b02de""#));

        let without_either = render_fragment(&trajectory, None, None);
        assert!(!without_either.contains("commit="));
        assert!(!without_either.contains("parent_commit="));

        // The trajectory's own commit renders too, when a caller has one.
        let with_both = render_fragment(&trajectory, Some("a3f81c2xxxxxxxxxxxxxxxxxxxx"), None);
        assert!(with_both.contains(r#"commit="a3f81c2""#));
    }

    #[test]
    fn checkpoint_labels_commit_for_and_contains_basics() {
        let mut dag = TrajectoryDag::new(Vec::new(), "base".to_string());
        dag.insert(node(3, CheckpointId::Root, "three", "c3"))
            .unwrap();
        dag.insert(node(1, CheckpointId::Root, "one", "c1"))
            .unwrap();

        assert!(dag.contains(&CheckpointId::Root));
        assert!(dag.contains(&CheckpointId::Worker(1)));
        assert!(!dag.contains(&CheckpointId::Worker(2)));
        assert_eq!(dag.commit_for(&CheckpointId::Root), Some("base"));
        assert_eq!(dag.commit_for(&CheckpointId::Worker(3)), Some("c3"));
        assert_eq!(dag.commit_for(&CheckpointId::Worker(2)), None);
        assert_eq!(dag.checkpoint_labels(), vec!["w1", "w3"]);
        assert!(dag.node(1).is_some());
    }

    #[test]
    fn off_lineage_diffstat_checkpoints_excludes_ancestors_and_empty_diffstats() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init", "--quiet"]);
        run_git(repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(repo, &["config", "user.name", "Asgard Test"]);
        let base = commit_file(repo, "base.txt", "base\n", "base");
        let c1 = commit_file(repo, "ancestor.txt", "ancestor\n", "ancestor");
        let c2 = commit_file(repo, "target.txt", "target\n", "target");
        run_git(repo, &["checkout", "--quiet", "--detach", &base]);
        let c3 = commit_file(repo, "empty-orphan.txt", "empty\n", "empty orphan");
        run_git(repo, &["checkout", "--quiet", "--detach", &base]);
        let c4 = commit_file(repo, "orphan.txt", "orphan\n", "diff orphan");
        let mut dag = TrajectoryDag::new_with_git_root(Vec::new(), base, repo.to_path_buf());
        dag.insert(node_with_diffstat(
            1,
            CheckpointId::Root,
            "ancestor",
            &c1,
            " ancestor.txt | 1 +\n",
        ))
        .unwrap();
        dag.insert(node_with_diffstat(
            2,
            CheckpointId::Worker(1),
            "target",
            &c2,
            " target.txt | 1 +\n",
        ))
        .unwrap();
        dag.insert(node_with_diffstat(
            3,
            CheckpointId::Root,
            "empty orphan",
            &c3,
            "   \n",
        ))
        .unwrap();
        dag.insert(node_with_diffstat(
            4,
            CheckpointId::Root,
            "diff orphan",
            &c4,
            " orphan.txt | 2 ++\n",
        ))
        .unwrap();

        let off_lineage = dag.off_lineage_checkpoints_with_diffstat(&CheckpointId::Worker(2));

        assert_eq!(
            off_lineage,
            vec![OffLineageCheckpoint {
                checkpoint: CheckpointId::Worker(4),
                diffstat: " orphan.txt | 2 ++\n".to_string(),
            }]
        );
    }

    #[test]
    fn is_ancestor_of_explores_plain_parent_chain_after_merge_path_hits_root() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init", "--quiet"]);
        run_git(repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(repo, &["config", "user.name", "Asgard Test"]);
        let base = commit_file(repo, "base.txt", "base\n", "base");
        let c2 = commit_file(repo, "two.txt", "two\n", "two");
        let c3 = commit_file(repo, "three.txt", "three\n", "three");
        run_git(repo, &["checkout", "--quiet", "--detach", &base]);
        let c1 = commit_file(repo, "one.txt", "one\n", "one");
        run_git(repo, &["checkout", "--quiet", "--detach", &c3]);
        run_git(repo, &["merge", "--quiet", "--no-ff", &c1, "-m", "merge"]);
        let c4 = run_git(repo, &["rev-parse", "HEAD"]).trim().to_string();
        let c5 = commit_file(repo, "five.txt", "five\n", "five");
        let mut dag = TrajectoryDag::new_with_git_root(Vec::new(), base, repo.to_path_buf());
        dag.insert(node(2, CheckpointId::Root, "two", &c2)).unwrap();
        dag.insert(node(3, CheckpointId::Worker(2), "three", &c3))
            .unwrap();
        dag.insert(node(1, CheckpointId::Root, "one", &c1)).unwrap();
        let mut merge = node(4, CheckpointId::Worker(3), "merge", &c4);
        merge.merged_from = vec![CheckpointId::Worker(1)];
        dag.insert(merge).unwrap();
        dag.insert(node(5, CheckpointId::Worker(4), "five", &c5))
            .unwrap();

        assert!(dag.is_ancestor_of(&CheckpointId::Worker(2), &CheckpointId::Worker(5)));
        assert!(dag.is_ancestor_of(&CheckpointId::Worker(1), &CheckpointId::Worker(5)));
        assert!(!dag.is_ancestor_of(&CheckpointId::Worker(5), &CheckpointId::Worker(2)));
    }

    #[test]
    fn resolve_checkpoint_by_commit_matches_known_checkpoints_by_full_or_short_sha() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init", "--quiet"]);
        run_git(repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(repo, &["config", "user.name", "Asgard Test"]);
        let base = commit_file(repo, "base.txt", "base\n", "base");
        let c1 = commit_file(repo, "one.txt", "one\n", "one");
        let mut dag =
            TrajectoryDag::new_with_git_root(Vec::new(), base.clone(), repo.to_path_buf());
        dag.insert(node(1, CheckpointId::Root, "one", &c1)).unwrap();

        assert_eq!(
            dag.resolve_checkpoint_by_commit(&base),
            Ok(CheckpointId::Root)
        );
        assert_eq!(
            dag.resolve_checkpoint_by_commit(&c1),
            Ok(CheckpointId::Worker(1))
        );
        // Short shas resolve too - "resolve short->full once at parse time".
        assert_eq!(
            dag.resolve_checkpoint_by_commit(&c1[..10]),
            Ok(CheckpointId::Worker(1))
        );
    }

    #[test]
    fn resolve_checkpoint_by_commit_rejects_invalid_and_non_descendant_refs() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init", "--quiet"]);
        run_git(repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(repo, &["config", "user.name", "Asgard Test"]);
        let base = commit_file(repo, "base.txt", "base\n", "base");
        run_git(repo, &["checkout", "--quiet", "--orphan", "unrelated"]);
        run_git(repo, &["rm", "-rf", "--quiet", "."]);
        let unrelated = commit_file(repo, "other.txt", "other\n", "unrelated");
        let dag = TrajectoryDag::new_with_git_root(Vec::new(), base, repo.to_path_buf());

        assert!(dag.resolve_checkpoint_by_commit("not-a-commit").is_err());
        assert!(
            dag.resolve_checkpoint_by_commit(&unrelated)
                .unwrap_err()
                .contains("not a descendant")
        );
    }

    #[test]
    fn resolve_checkpoint_by_commit_wraps_novel_descendant_and_ancestor_messages_notes_it() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init", "--quiet"]);
        run_git(repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(repo, &["config", "user.name", "Asgard Test"]);
        let base = commit_file(repo, "base.txt", "base\n", "base");
        let c1 = commit_file(repo, "one.txt", "one\n", "one");
        let novel = commit_file(repo, "novel.txt", "novel\n", "made outside a worker");
        let mut dag = TrajectoryDag::new_with_git_root(Vec::new(), base, repo.to_path_buf());
        dag.insert(node(1, CheckpointId::Root, "one", &c1)).unwrap();

        let resolved = dag
            .resolve_checkpoint_by_commit(&novel)
            .expect("descendant resolves");
        assert_eq!(resolved, CheckpointId::Commit(novel.clone()));
        assert_eq!(dag.commit_for(&resolved), Some(novel.as_str()));

        let messages = dag.ancestor_messages(&resolved).expect("ancestor messages");
        let last = messages.last().unwrap().content_text();
        assert!(last.contains("You start from commit"));
        assert!(last.contains(short_sha(&novel)));
        assert!(last.contains("descends from w1"));
    }

    #[test]
    fn render_dag_overview_merges_saved_discarded_and_live_tree() {
        let mut dag = TrajectoryDag::new(Vec::new(), "base".to_string());
        dag.insert(node_with_diffstat(
            3,
            CheckpointId::Root,
            "saved root",
            "c3",
            " saved.txt | 1 +\n",
        ))
        .unwrap();
        dag.insert(node(
            7,
            CheckpointId::Worker(3),
            "saved child instructions that are intentionally longer than sixty characters",
            "c7",
        ))
        .unwrap();
        dag.discard(4, CheckpointId::Root, "discarded\nroot child".to_string())
            .unwrap();
        let live = vec![
            DagLiveEntry {
                worker: 5,
                parent: CheckpointId::Root,
                status: "in flight, step 2/10".to_string(),
                instructions: "live root child".to_string(),
            },
            DagLiveEntry {
                worker: 8,
                parent: CheckpointId::Worker(3),
                status: "under review".to_string(),
                instructions: "live child\nunder review".to_string(),
            },
        ];

        let rendered = render_dag_overview(&dag, &live);

        assert_eq!(
            rendered,
            "root\n\
├─ w3 (c3) \"instructions saved root\" saved, finished/2 steps; diffstat: saved.txt | 1 +\n\
│  ├─ w7 (c7) \"instructions saved child instructions that are intentionally\" saved, finished/2 steps\n\
│  └─ w8 \"live child under review\" under review\n\
├─ w4 \"discarded root child\" discarded\n\
└─ w5 \"live root child\" in flight, step 2/10\n"
        );
    }
}
