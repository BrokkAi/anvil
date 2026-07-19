use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::str::FromStr;

use anyhow::{Result, anyhow, bail, ensure};

use crate::llm_client::{ChatMessage, TokenUsage};

#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub(crate) enum CheckpointId {
    Root,
    Worker(usize),
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
    nodes: BTreeMap<usize, TrajectoryNode>,
    discarded: BTreeMap<usize, DiscardedTombstone>,
}

impl TrajectoryDag {
    pub(crate) fn new(initial_messages: Vec<ChatMessage>, base_commit: String) -> Self {
        Self {
            initial_messages,
            base_commit,
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
        }
    }

    pub(crate) fn commit_for(&self, ckpt: &CheckpointId) -> Option<&str> {
        match ckpt {
            CheckpointId::Root => Some(&self.base_commit),
            CheckpointId::Worker(worker) => self.nodes.get(worker).map(|node| node.commit.as_str()),
        }
    }

    pub(crate) fn node(&self, worker: usize) -> Option<&TrajectoryNode> {
        self.nodes.get(&worker)
    }

    pub(crate) fn checkpoint_labels(&self) -> Vec<String> {
        self.nodes
            .keys()
            .map(|worker| CheckpointId::Worker(*worker).to_string())
            .collect()
    }

    pub(crate) fn ancestor_messages(&self, ckpt: &CheckpointId) -> Result<Vec<ChatMessage>> {
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

    pub(crate) fn resolve_handles(
        &self,
        handles: &[String],
        pending: Option<(usize, &[ChatMessage])>,
        in_flight: &[usize],
    ) -> String {
        let mut rendered = String::new();
        for handle in handles {
            let Some((worker, index)) = crate::asgard::parse_worker_tool_handle(handle) else {
                // A bare checkpoint id ("w11") is a natural but wrong way to
                // ask for a whole trajectory; answer the intent instead of
                // calling it malformed.
                match CheckpointId::parse(handle) {
                    Some(CheckpointId::Worker(worker)) if self.nodes.contains_key(&worker) => {
                        rendered.push_str(&format!(
                            "<tool_call handle=\"{handle}\" error=\"w{worker} is a saved checkpoint, not a tool-result handle; its compact trace was shown at its review. Expand a specific result with a handle like w{worker}m4.\" />\n"
                        ));
                    }
                    Some(CheckpointId::Worker(worker)) if in_flight.contains(&worker) => {
                        rendered.push_str(&format!(
                            "<tool_call handle=\"{handle}\" error=\"w{worker} is in flight or awaiting review; its results become viewable when its trajectory is presented for review. Call wait (or simply reply without tool calls) to let that happen.\" />\n"
                        ));
                    }
                    Some(CheckpointId::Worker(worker)) if self.discarded.contains_key(&worker) => {
                        rendered.push_str(&format!(
                            "<tool_call handle=\"{handle}\" error=\"trajectory w{worker} was discarded; its full results are gone\" />\n"
                        ));
                    }
                    _ => {
                        rendered.push_str(&format!(
                            "<tool_call handle=\"{handle}\" error=\"malformed handle\" />\n"
                        ));
                    }
                }
                continue;
            };

            let messages = if let Some((pending_worker, pending_messages)) = pending
                && pending_worker == worker
            {
                Some(pending_messages)
            } else {
                self.nodes
                    .get(&worker)
                    .map(|node| node.window.window_messages.as_slice())
            };
            let Some(messages) = messages else {
                if self.discarded.contains_key(&worker) {
                    rendered.push_str(&format!(
                        "<tool_call handle=\"{handle}\" error=\"trajectory was discarded; its full results are gone\" />\n"
                    ));
                } else if in_flight.contains(&worker) {
                    rendered.push_str(&format!(
                        "<tool_call handle=\"{handle}\" error=\"w{worker} is in flight or awaiting review; its results become viewable when its trajectory is presented for review. Call wait (or simply reply without tool calls) to let that happen.\" />\n"
                    ));
                } else {
                    rendered.push_str(&format!(
                        "<tool_call handle=\"{handle}\" error=\"unknown worker\" />\n"
                    ));
                }
                continue;
            };
            let Some(result) = messages.get(index).filter(|message| message.role == "tool") else {
                rendered.push_str(&format!(
                    "<tool_call handle=\"{handle}\" error=\"handle does not name a tool result\" />\n"
                ));
                continue;
            };

            let call = crate::asgard::originating_tool_call(messages, index);
            let name = call.map_or("tool", |call| call.function.name.as_str());
            let arguments = call.map_or("", |call| call.function.arguments.as_str());
            rendered.push_str(&format!(
                "<tool_call id=\"{handle}\" worker=\"w{worker}\" name=\"{name}\">\n\
                 <arguments>{arguments}</arguments>\n\
                 <result>{}</result>\n\
                 </tool_call>\n",
                result.content_text(),
            ));
        }
        rendered
    }

    pub(crate) fn handle_is_run_shell_command_result(
        &self,
        handle: &str,
        pending: Option<(usize, &[ChatMessage])>,
    ) -> bool {
        let Some((worker, index)) = crate::asgard::parse_worker_tool_handle(handle) else {
            return false;
        };
        let messages = if let Some((pending_worker, pending_messages)) = pending
            && pending_worker == worker
        {
            Some(pending_messages)
        } else {
            self.nodes
                .get(&worker)
                .map(|node| node.window.window_messages.as_slice())
        };
        let Some(messages) = messages else {
            return false;
        };
        if messages
            .get(index)
            .is_none_or(|message| message.role != "tool")
        {
            return false;
        }
        crate::asgard::originating_tool_call(messages, index)
            .is_some_and(|call| call.function.name == "run_shell_command")
    }
}

pub(crate) fn render_fragment(window: &TrajectoryWindow) -> String {
    let mut rendered = String::new();
    rendered.push_str(&format!(
        "<worker_trajectory id=\"w{}\" continues_from=\"{}\" model=\"{}\" stop=\"{}\" steps=\"{}\">\n",
        window.worker,
        window.parent,
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
                    "w{worker} \"{}\" saved, {}/{} steps\n",
                    instruction_stub(&node.window.instructions),
                    node.window.stop.label(),
                    node.window.steps
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
        }
    }

    fn text(message: &ChatMessage) -> String {
        message.content_text()
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
        })
        .unwrap();
        dag.discard(3, CheckpointId::Root, "discarded".to_string())
            .unwrap();

        let pending = vec![
            assistant_call(call("pending", "run_shell_command", r#"{"command":"pwd"}"#)),
            ChatMessage::tool_result("pending", "run_shell_command", "pending result"),
            ChatMessage::assistant("pending final"),
        ];
        let rendered = dag.resolve_handles(
            &[
                "w1m1".to_string(),
                "w2m1".to_string(),
                "w3m1".to_string(),
                "w4m1".to_string(),
                "w1l0m1".to_string(),
                "w1m2".to_string(),
                "w9m0".to_string(),
            ],
            Some((2, &pending)),
            &[9],
        );

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
            r#"<tool_call handle="w9m0" error="w9 is in flight or awaiting review; its results become viewable when its trajectory is presented for review. Call wait (or simply reply without tool calls) to let that happen." />"#
        ));
        assert!(rendered.contains(r#"<tool_call handle="w1l0m1" error="malformed handle" />"#));
        assert!(
            rendered.contains(
                r#"<tool_call handle="w1m2" error="handle does not name a tool result" />"#
            )
        );
    }

    #[test]
    fn handle_is_run_shell_command_result_checks_saved_and_pending_results() {
        let mut dag = TrajectoryDag::new(Vec::new(), "base".to_string());
        let mut saved = window(1, CheckpointId::Root, "saved");
        saved.window_messages = vec![
            assistant_call(call("read", "read_file", r#"{"file_path":"x"}"#)),
            ChatMessage::tool_result("read", "read_file", "read result"),
            assistant_call(call(
                "shell",
                "run_shell_command",
                r#"{"command":"cargo test"}"#,
            )),
            ChatMessage::tool_result("shell", "run_shell_command", "test result"),
        ];
        dag.insert(TrajectoryNode {
            window: saved,
            commit: "c1".to_string(),
        })
        .unwrap();
        let pending = vec![
            assistant_call(call(
                "pending-shell",
                "run_shell_command",
                r#"{"command":"pwd"}"#,
            )),
            ChatMessage::tool_result("pending-shell", "run_shell_command", "pending result"),
        ];

        assert!(dag.handle_is_run_shell_command_result("w1m3", None));
        assert!(dag.handle_is_run_shell_command_result("w2m1", Some((2, &pending))));
        assert!(!dag.handle_is_run_shell_command_result("w1m1", None));
        assert!(!dag.handle_is_run_shell_command_result("w1m2", None));
        assert!(!dag.handle_is_run_shell_command_result("bad", Some((2, &pending))));
    }

    #[test]
    fn render_fragment_includes_verbatim_response_and_failure_details() {
        let mut trajectory = window(3, CheckpointId::Root, "fragment");
        trajectory.final_response = format!("{}<raw>&unescaped", "a".repeat(200));
        trajectory.diffstat = " src/lib.rs | 2 +".to_string();
        let rendered = render_fragment(&trajectory);
        assert!(rendered.contains(
            r#"<worker_trajectory id="w3" continues_from="root" model="model-a" stop="finished" steps="2">"#
        ));
        assert!(rendered.contains("<diffstat> src/lib.rs | 2 +</diffstat>"));
        assert!(rendered.contains(&trajectory.final_response));

        trajectory.final_response.clear();
        let empty = render_fragment(&trajectory);
        assert!(empty.contains(r#"<final_response none="true" />"#));

        trajectory.stop = WorkerStopReason::Failed("boom <bad>".to_string());
        let failed = render_fragment(&trajectory);
        assert!(failed.contains(r#"stop="failed""#));
        assert!(failed.contains("<failure>boom &lt;bad&gt;</failure>"));
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
    fn render_dag_overview_merges_saved_discarded_and_live_tree() {
        let mut dag = TrajectoryDag::new(Vec::new(), "base".to_string());
        dag.insert(node(3, CheckpointId::Root, "saved root", "c3"))
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
├─ w3 \"instructions saved root\" saved, finished/2 steps\n\
│  ├─ w7 \"instructions saved child instructions that are intentionally\" saved, finished/2 steps\n\
│  └─ w8 \"live child under review\" under review\n\
├─ w4 \"discarded root child\" discarded\n\
└─ w5 \"live root child\" in flight, step 2/10\n"
        );
    }
}
