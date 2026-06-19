mod filesystem;
pub mod sandbox;
mod shell;

use crate::agents::AgentRegistry;
use crate::llm_client::{FunctionDef, ToolDefinition};
use crate::mcp::{McpClient, McpServerConfig};
use crate::skills::SkillRegistry;
use agent_client_protocol::schema::ToolKind;
use sandbox::SandboxPolicy;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

/// Result of executing a tool.
pub struct ToolResult {
    pub status: ToolStatus,
    pub output: String,
}

pub enum ToolStatus {
    Success,
    RequestError,
    InternalError,
}

fn cancelled_tool_result(name: &str) -> ToolResult {
    ToolResult {
        status: ToolStatus::RequestError,
        output: format!("Tool '{name}' was cancelled before it completed."),
    }
}

/// Single source of truth for per-tool metadata (`ToolKind` for the
/// permission gate, `display_name` for the UI fallback). Adding a new
/// tool means adding one row here; the dispatcher in `execute` derives
/// builtin routing from the inline `match`, and bifrost-loaded tools we
/// don't recognize fall through to `ToolKind::Other` / "Executing tool".
struct ToolMeta {
    name: &'static str,
    kind: ToolKind,
    display_name: &'static str,
}

const TOOLS: &[ToolMeta] = &[
    // --- Built-in tools (executed inline in `ToolRegistry::execute`) -------
    ToolMeta {
        name: "read_file",
        kind: ToolKind::Read,
        display_name: "Reading file",
    },
    ToolMeta {
        name: "write_file",
        kind: ToolKind::Edit,
        display_name: "Writing file",
    },
    ToolMeta {
        name: "edit",
        kind: ToolKind::Edit,
        display_name: "Editing file",
    },
    ToolMeta {
        name: "list_directory",
        kind: ToolKind::Read,
        display_name: "Listing directory",
    },
    ToolMeta {
        name: "grep_search",
        kind: ToolKind::Search,
        display_name: "Searching file contents",
    },
    ToolMeta {
        name: "run_shell_command",
        kind: ToolKind::Execute,
        display_name: "Running shell command",
    },
    ToolMeta {
        name: crate::goal::GET_GOAL_TOOL_NAME,
        kind: ToolKind::Read,
        display_name: "Getting goal",
    },
    ToolMeta {
        name: crate::goal::CREATE_GOAL_TOOL_NAME,
        kind: ToolKind::Read,
        display_name: "Creating goal",
    },
    ToolMeta {
        name: crate::goal::UPDATE_GOAL_TOOL_NAME,
        kind: ToolKind::Read,
        display_name: "Updating goal",
    },
    // --- MCP-loaded Bifrost tools (dispatched via `execute_mcp`) -----------
    // Listed here so the permission gate can classify them; their actual
    // execution is delegated to the configured MCP server. The cross-check
    // in `mcp::tests::handshake_and_call_search_tools` keeps this list in
    // sync with what the running Bifrost MCP server exposes.
    ToolMeta {
        name: "get_summaries",
        kind: ToolKind::Read,
        display_name: "Getting code summaries",
    },
    ToolMeta {
        name: "get_active_workspace",
        kind: ToolKind::Read,
        display_name: "Getting active workspace",
    },
    ToolMeta {
        name: "search_symbols",
        kind: ToolKind::Search,
        display_name: "Searching for symbols",
    },
    ToolMeta {
        name: "get_symbol_locations",
        kind: ToolKind::Search,
        display_name: "Finding symbol locations",
    },
    ToolMeta {
        name: "get_symbol_ancestors",
        kind: ToolKind::Search,
        display_name: "Finding symbol ancestors",
    },
    ToolMeta {
        name: "get_symbol_summaries",
        kind: ToolKind::Search,
        display_name: "Getting symbol summaries",
    },
    ToolMeta {
        name: "get_symbol_sources",
        kind: ToolKind::Search,
        display_name: "Fetching symbol source",
    },
    ToolMeta {
        name: "most_relevant_files",
        kind: ToolKind::Search,
        display_name: "Finding related files",
    },
    ToolMeta {
        name: "scan_usages",
        kind: ToolKind::Search,
        display_name: "Scanning symbol usages",
    },
    ToolMeta {
        name: "usage_graph",
        kind: ToolKind::Search,
        display_name: "Building usage graph",
    },
    ToolMeta {
        name: "semantic_search",
        kind: ToolKind::Search,
        display_name: "Searching semantically",
    },
    ToolMeta {
        name: "get_file_contents",
        kind: ToolKind::Read,
        display_name: "Reading file contents",
    },
    ToolMeta {
        name: "find_filenames",
        kind: ToolKind::Search,
        display_name: "Finding filenames",
    },
    ToolMeta {
        name: "find_files_containing",
        kind: ToolKind::Search,
        display_name: "Finding files containing text",
    },
    ToolMeta {
        name: "search_file_contents",
        kind: ToolKind::Search,
        display_name: "Searching file contents",
    },
    ToolMeta {
        name: "list_files",
        kind: ToolKind::Read,
        display_name: "Listing files",
    },
    ToolMeta {
        name: "skim_files",
        kind: ToolKind::Read,
        display_name: "Skimming files",
    },
    ToolMeta {
        name: "search_git_commit_messages",
        kind: ToolKind::Search,
        display_name: "Searching git commit messages",
    },
    ToolMeta {
        name: "get_git_log",
        kind: ToolKind::Read,
        display_name: "Reading git log",
    },
    ToolMeta {
        name: "get_commit_diff",
        kind: ToolKind::Read,
        display_name: "Reading commit diff",
    },
    ToolMeta {
        name: "jq",
        kind: ToolKind::Search,
        display_name: "Querying JSON",
    },
    ToolMeta {
        name: "xml_skim",
        kind: ToolKind::Read,
        display_name: "Skimming XML",
    },
    ToolMeta {
        name: "xml_select",
        kind: ToolKind::Search,
        display_name: "Selecting XML",
    },
    ToolMeta {
        name: "compute_cyclomatic_complexity",
        kind: ToolKind::Read,
        display_name: "Computing cyclomatic complexity",
    },
    ToolMeta {
        name: "compute_cognitive_complexity",
        kind: ToolKind::Read,
        display_name: "Computing cognitive complexity",
    },
    ToolMeta {
        name: "report_comment_density_for_code_unit",
        kind: ToolKind::Read,
        display_name: "Reporting comment density",
    },
    ToolMeta {
        name: "report_comment_density_for_files",
        kind: ToolKind::Read,
        display_name: "Reporting file comment density",
    },
    ToolMeta {
        name: "report_exception_handling_smells",
        kind: ToolKind::Read,
        display_name: "Reporting exception handling smells",
    },
    ToolMeta {
        name: "report_test_assertion_smells",
        kind: ToolKind::Read,
        display_name: "Reporting test assertion smells",
    },
    ToolMeta {
        name: "report_structural_clone_smells",
        kind: ToolKind::Read,
        display_name: "Reporting structural clone smells",
    },
    ToolMeta {
        name: "report_long_method_and_god_object_smells",
        kind: ToolKind::Read,
        display_name: "Reporting long method and god object smells",
    },
    ToolMeta {
        name: "report_dead_code_and_unused_abstraction_smells",
        kind: ToolKind::Read,
        display_name: "Reporting dead code smells",
    },
    ToolMeta {
        name: "report_secret_like_code",
        kind: ToolKind::Read,
        display_name: "Reporting secret-like code",
    },
    ToolMeta {
        name: "analyze_git_hotspots",
        kind: ToolKind::Read,
        display_name: "Analyzing git hotspots",
    },
    // `activate_workspace` and `refresh` mutate analyzer state, so they
    // stay `Other` rather than `Read`: prompted in `default`, refused in
    // `readOnly`.
    ToolMeta {
        name: "activate_workspace",
        kind: ToolKind::Other,
        display_name: "Activating workspace",
    },
    ToolMeta {
        name: "refresh",
        kind: ToolKind::Other,
        display_name: "Refreshing analyzer index",
    },
    // --- Agent Skills activation -------------------------------------------
    // The tool itself is registered dynamically in `tool_definitions()`
    // only when the session has at least one discovered skill; this row
    // is what the permission gate looks up by name. Classified `Read`
    // because activating a skill only reads `SKILL.md` and produces
    // text -- the skill's body can then drive other (gated) tool calls.
    ToolMeta {
        name: "activate_skill",
        kind: ToolKind::Read,
        display_name: "Activating skill",
    },
    // --- Subagent dispatch -------------------------------------------------
    // Like `activate_skill`, registered dynamically in `tool_definitions()`
    // only when at least one subagent is discovered. Classified `Other`:
    // its transitive effects are whatever tools the subagent invokes, so
    // we want the gate to refuse it in `readOnly` mode and prompt in
    // `default`. Actual dispatch happens in `tool_loop::run` (not
    // `ToolRegistry::execute`) because the subagent needs `llm`,
    // `spawned_cx`, and `sessions` -- none of which the registry sees.
    ToolMeta {
        name: "task",
        kind: ToolKind::Other,
        display_name: "Running subagent",
    },
];

#[cfg(test)]
pub(crate) const SLOPCOP_BIFROST_READ_ONLY_TOOLS: &[&str] = &[
    "compute_cyclomatic_complexity",
    "compute_cognitive_complexity",
    "report_comment_density_for_code_unit",
    "report_comment_density_for_files",
    "report_exception_handling_smells",
    "report_test_assertion_smells",
    "report_structural_clone_smells",
    "report_long_method_and_god_object_smells",
    "report_dead_code_and_unused_abstraction_smells",
    "report_secret_like_code",
    "analyze_git_hotspots",
];

fn tool_meta(name: &str) -> Option<&'static ToolMeta> {
    TOOLS.iter().find(|t| t.name == name)
}

/// `true` iff `name` has a row in the `TOOLS` metadata table. Used by
/// the bifrost handshake test to flag drift when bifrost adds or
/// renames a tool without a matching `TOOLS` entry (which would
/// otherwise silently fall back to `ToolKind::Other` / "Executing
/// tool" in the permission gate and UI).
#[cfg(test)]
pub(crate) fn is_known_tool(name: &str) -> bool {
    tool_meta(name).is_some()
}

/// Built-in tool names handled by the inline `match` in
/// `ToolRegistry::execute`. Used by tests to keep the metadata table
/// in sync with the actual builtin dispatch.
const BUILTIN_TOOL_NAMES: &[&str] = &[
    "read_file",
    "write_file",
    "edit",
    "list_directory",
    "grep_search",
    "run_shell_command",
];

fn is_builtin_tool(name: &str) -> bool {
    BUILTIN_TOOL_NAMES.contains(&name)
}

fn is_harness_only_mcp_tool(name: &str) -> bool {
    name == "refresh"
}

/// Unified tool registry: filesystem tools + shell + configured
/// MCP tools + Agent Skills activation.
///
/// `skills` is wrapped in `RwLock` so the session can swap in a fresh
/// `SkillRegistry` after `update_cwd` without rebuilding the registry
/// (which would re-spawn MCP subprocesses).
pub struct ToolRegistry {
    cwd: PathBuf,
    mcp_clients: Vec<Arc<McpClient>>,
    mcp_tool_servers: HashMap<String, Arc<McpClient>>,
    advertised_builtin_tools: RwLock<HashSet<String>>,
    skills: RwLock<Arc<SkillRegistry>>,
    agents: RwLock<Arc<AgentRegistry>>,
}

impl ToolRegistry {
    /// Working directory this registry is rooted in.
    pub(crate) fn cwd(&self) -> &Path {
        &self.cwd
    }

    /// Replace the cached SkillRegistry. Called by `update_cwd` so the
    /// next prompt's tool catalog reflects the fresh on-disk skills.
    pub async fn set_skills(&self, skills: Arc<SkillRegistry>) {
        *self.skills.write().await = skills;
    }

    /// Replace the cached AgentRegistry. Same pattern as `set_skills`:
    /// `update_cwd` re-discovers from the new working directory and the
    /// next prompt's tool catalog reflects it.
    pub async fn set_agents(&self, agents: Arc<AgentRegistry>) {
        *self.agents.write().await = agents;
    }

    /// Replace the set of built-in tools advertised to the model. This does
    /// not affect the underlying builtin dispatch; it only changes which
    /// builtin schemas appear in subsequent `tool_definitions()` snapshots.
    pub async fn set_builtin_tools(&self, tools: HashSet<String>) {
        *self.advertised_builtin_tools.write().await = tools;
    }

    /// Snapshot the current built-in tool names advertised to the model.
    pub async fn active_builtin_tools(&self) -> HashSet<String> {
        self.advertised_builtin_tools.read().await.clone()
    }

    /// Whether the built-in `name` is currently advertised to the model.
    pub async fn is_builtin_tool_advertised(&self, name: &str) -> bool {
        self.advertised_builtin_tools.read().await.contains(name)
    }

    /// Snapshot the current AgentRegistry. Used by `tool_loop::run` to
    /// look up `subagent_type` without holding the read lock across the
    /// nested LLM call.
    pub(crate) async fn agents_snapshot(&self) -> Arc<AgentRegistry> {
        self.agents.read().await.clone()
    }

    pub async fn new(
        cwd: PathBuf,
        mcp_servers: Vec<McpServerConfig>,
        skills: Arc<SkillRegistry>,
        agents: Arc<AgentRegistry>,
    ) -> Self {
        // Best-effort sweep of any stale seatbelt policy files left by a
        // previous SIGKILL/panic. Bounded by file age so we don't yank a
        // profile from a concurrent in-flight shell call.
        sandbox::cleanup_stale_policy_files();

        let mut mcp_clients = Vec::new();
        let mut mcp_tool_servers = HashMap::new();
        for config in mcp_servers.iter().filter(|server| server.enabled) {
            match McpClient::spawn(config, &cwd).await {
                Ok(client) => {
                    let client = Arc::new(client);
                    for tool in client.tools() {
                        if is_builtin_tool(&tool.name) {
                            tracing::warn!(
                                server = %client.name(),
                                tool = %tool.name,
                                "mcp tool name collides with a built-in tool; ignoring server tool"
                            );
                            continue;
                        }
                        if is_harness_only_mcp_tool(&tool.name) {
                            tracing::info!(
                                server = %client.name(),
                                tool = %tool.name,
                                "mcp tool is reserved for harness use; hiding from model dispatch"
                            );
                            continue;
                        }
                        if mcp_tool_servers
                            .insert(tool.name.clone(), client.clone())
                            .is_some()
                        {
                            tracing::warn!(
                                server = %client.name(),
                                tool = %tool.name,
                                "mcp tool name collision; later server wins"
                            );
                        }
                    }
                    mcp_clients.push(client);
                }
                Err(err) => {
                    tracing::warn!(
                        cwd = %cwd.display(),
                        server = %config.name,
                        command = %config.command,
                        %err,
                        "mcp server failed to start; its tools are disabled for this session"
                    );
                }
            }
        }
        Self {
            cwd,
            mcp_clients,
            mcp_tool_servers,
            advertised_builtin_tools: RwLock::new(
                BUILTIN_TOOL_NAMES
                    .iter()
                    .map(|name| (*name).to_string())
                    .collect(),
            ),
            skills: RwLock::new(skills),
            agents: RwLock::new(agents),
        }
    }

    /// All tool definitions for the OpenAI tools parameter.
    pub async fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tool_definitions_with_shell_escalation(false).await
    }

    /// Tool definitions for the OpenAI tools parameter.
    ///
    /// The outside-sandbox retry field is intentionally hidden until a
    /// sandboxed shell command fails with a sandbox-looking error. If the model
    /// sees that field in the first schema, some providers eagerly choose it
    /// even though the description says it is only for retries.
    pub async fn tool_definitions_with_shell_escalation(
        &self,
        allow_shell_sandbox_escalation: bool,
    ) -> Vec<ToolDefinition> {
        let builtin_tools = self.active_builtin_tools().await;
        let mut defs = Vec::new();
        if builtin_tools.contains("read_file") {
            defs.push(tool_def(
                "read_file",
                "Reads and returns the content of a specified text file. Use after you have selected an exact file/range; for code definitions prefer get_symbol_sources, and for broad code orientation prefer get_summaries.",
                json!({
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to read. Relative paths are resolved against the working directory; absolute paths must remain inside it."
                        },
                        "offset": {
                            "type": "number",
                            "description": "Optional 0-based line number to start reading from."
                        },
                        "limit": {
                            "type": "number",
                            "description": "Optional maximum number of lines to read."
                        }
                    },
                    "required": ["file_path"]
                }),
            ));
        }
        if builtin_tools.contains("write_file") {
            defs.push(tool_def(
                "write_file",
                "Writes content to a specified file in the local filesystem. Paths may be relative to the working directory or absolute paths inside it.",
                json!({
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to write. Relative paths are resolved against the working directory; absolute paths must remain inside it."
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file."
                        }
                    },
                    "required": ["file_path", "content"]
                }),
            ));
        }
        if builtin_tools.contains("edit") {
            defs.push(tool_def(
                "edit",
                "Replaces exact literal text within a file. By default, replaces a single occurrence. Set `replace_all` to true to replace every matching occurrence.",
                json!({
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to modify. Relative paths are resolved against the working directory; absolute paths must remain inside it."
                        },
                        "old_string": {
                            "type": "string",
                            "description": "The exact literal text to replace, including whitespace and indentation."
                        },
                        "new_string": {
                            "type": "string",
                            "description": "The exact literal text to replace `old_string` with."
                        },
                        "replace_all": {
                            "type": "boolean",
                            "description": "Replace all occurrences of old_string. Defaults to false."
                        }
                    },
                    "required": ["file_path", "old_string", "new_string"]
                }),
            ));
        }
        if builtin_tools.contains("list_directory") {
            defs.push(tool_def(
                "list_directory",
                "Lists the names of files and subdirectories directly within a specified directory path. Paths may be relative to the working directory or absolute paths inside it.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the directory to list. Use '.' for the working directory root. Absolute paths must remain inside the working directory."
                        }
                    },
                    "required": ["path"]
                }),
            ));
        }
        if builtin_tools.contains("grep_search") {
            defs.push(tool_def(
                "grep_search",
                "Searches file contents with a regex. Use for text/config/docs or when symbol tools do not fit; for code declarations prefer search_symbols, and for references/callers prefer scan_usages.",
                json!({
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "The regular expression pattern to search for in file contents."
                        },
                        "glob": {
                            "type": "string",
                            "description": "Optional glob pattern to filter files (e.g. '*.rs', '**/*.java')."
                        },
                        "path": {
                            "type": "string",
                            "description": "Optional file or directory to search in. Relative paths are resolved against the working directory; absolute paths must remain inside it. Defaults to the working directory."
                        },
                        "limit": {
                            "type": "number",
                            "description": "Optional limit on matching lines. Defaults to 50."
                        }
                    },
                    "required": ["pattern"]
                }),
            ));
        }
        if builtin_tools.contains("run_shell_command") {
            let mut shell_properties = json!({
                "command": {
                    "type": "string",
                    "description": "The shell command to execute (passed to sh -c)."
                },
                "timeout": {
                    "type": "number",
                    "description": "Optional timeout in milliseconds."
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of the command for the user. Accepted for compatibility and not used for execution."
                },
                "directory": {
                    "type": "string",
                    "description": "Optional directory to run the command in. Relative paths are resolved against the working directory; absolute paths must remain inside it."
                }
            });
            if allow_shell_sandbox_escalation {
                shell_properties["sandbox_permissions"] = json!({
                    "type": "string",
                    "enum": ["require_escalated"],
                    "description": "Set to `require_escalated` only when retrying a command that already failed under the sandbox and your investigation indicates the sandbox boundary is the likely cause (for example EPERM from namespace/process isolation, writes outside the workspace, or sandbox-blocked network/DNS access). This asks the user for permission to run the command outside the sandbox once."
                });
            }
            defs.push(tool_def(
                "run_shell_command",
                "Execute a shell command in the working directory. Returns stdout and stderr. Prefer built-in tools for ordinary file reads/search/list/edit/write operations and Bifrost tools for code symbols, definitions, usages, and source orientation. Use shell when CLI semantics matter, such as build, test, git, package-manager, project-specific commands, pipelines, or raw-byte/format inspection. When the session uses sandboxing, commands run in that sandbox by default. If a sandboxed attempt fails because of the sandbox boundary, including blocked network or DNS access, the retry field for requesting one-time outside-sandbox permission will be exposed in the next tool schema.",
                json!({
                    "type": "object",
                    "properties": shell_properties,
                    "required": ["command"]
                }),
            ));
        }
        defs.push(tool_def(
            crate::goal::GET_GOAL_TOOL_NAME,
            "Get the current goal for this session, including status, token budget, used tokens, and remaining token budget.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        ));
        defs.push(tool_def(
            crate::goal::CREATE_GOAL_TOOL_NAME,
            "Create a goal only when explicitly requested by the user or system/developer instructions; do not infer goals from ordinary tasks. Set token_budget only when an explicit token budget is requested. Fails if an unfinished goal exists; use update_goal only for status.",
            json!({
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "Required. The concrete objective to start pursuing."
                    },
                    "token_budget": {
                        "type": "integer",
                        "description": "Positive token budget for the new goal. Omit unless explicitly requested."
                    }
                },
                "required": ["objective"]
            }),
        ));
        defs.push(tool_def(
            crate::goal::UPDATE_GOAL_TOOL_NAME,
            "Update the existing goal. Use this tool only to mark the goal achieved or genuinely blocked. Set status to complete only when the objective has actually been achieved and no required work remains. Set status to blocked only when the same blocking condition has repeated for at least three consecutive goal turns and the agent is at an impasse. You cannot use this tool to pause, resume, budget-limit, or usage-limit a goal; those status changes are controlled by the user or system.",
            json!({
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["complete", "blocked"],
                        "description": "Required. Set to complete only when done; set to blocked only after a genuine repeated blocking condition."
                    }
                },
                "required": ["status"]
            }),
        ));
        let mut advertised_names: HashSet<String> =
            defs.iter().map(|def| def.function.name.clone()).collect();
        for client in &self.mcp_clients {
            for tool in client.tools() {
                if is_harness_only_mcp_tool(&tool.name) {
                    continue;
                }
                if !advertised_names.insert(tool.name.clone()) {
                    tracing::warn!(
                        server = %client.name(),
                        tool = %tool.name,
                        "mcp tool name collision; skipping duplicate tool definition"
                    );
                    continue;
                }
                defs.push(tool_def(
                    &tool.name,
                    &tool.description,
                    tool.input_schema.clone(),
                ));
            }
        }

        // Append `activate_skill` only when at least one skill exists, and
        // constrain `name` to the discovered set via JSON-schema enum.
        // The spec's "Filtering" note: don't expose the tool with an
        // empty enum -- the model would waste turns guessing.
        let skills = self.skills.read().await;
        if !skills.is_empty() {
            let names: Vec<String> = skills.iter_sorted().map(|m| m.name.clone()).collect();
            defs.push(tool_def(
                "activate_skill",
                "Load the full instructions for a previously listed skill from `<available_skills>`. \
                 Call this BEFORE attempting the task when the user's request matches a skill's description. \
                 Returns the skill's body and a list of its bundled resource files; use your file-read tool \
                 to load those resources only when the skill instructions tell you to.",
                json!({
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": names,
                            "description": "Exact skill name from the catalog."
                        }
                    },
                    "required": ["name"]
                }),
            ));
        }
        drop(skills);

        // Append `task` only when at least one subagent is discovered.
        // The enum constraint keeps the model from guessing names not in
        // the catalog (mirrors `activate_skill`).
        let agents = self.agents.read().await;
        if !agents.is_empty() {
            let names: Vec<String> = agents.iter_sorted().map(|m| m.name.clone()).collect();
            let catalog: String = agents
                .iter_sorted()
                .map(|m| format!("- {}: {}", m.name, m.description))
                .collect::<Vec<_>>()
                .join("\n");
            defs.push(tool_def(
                "task",
                &format!(
                    "Delegate a focused task to a specialized subagent. The subagent runs in an \
                     isolated context with the same tools as you; only its final text answer comes \
                     back. Use when the work is well-defined and self-contained, or when you want \
                     to keep its tool-call noise out of the main conversation. The subagent does \
                     NOT see this conversation -- give it a self-contained prompt.\n\n\
                     Available subagents:\n{catalog}"
                ),
                json!({
                    "type": "object",
                    "properties": {
                        "subagent_type": {
                            "type": "string",
                            "enum": names,
                            "description": "Name of the subagent from the catalog."
                        },
                        "description": {
                            "type": "string",
                            "description": "Short (3-5 word) label describing the delegation."
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Self-contained task prompt for the subagent."
                        }
                    },
                    "required": ["subagent_type", "description", "prompt"]
                }),
            ));
        }

        defs
    }

    pub(crate) fn is_bifrost_tool(&self, name: &str) -> bool {
        self.mcp_tool_servers
            .get(name)
            .is_some_and(|client| client.name() == "bifrost")
    }

    /// Execute a tool by name with JSON arguments.
    ///
    /// SECURITY: LLM-initiated callers MUST consult `tool_loop::consult_gate`
    /// first. User-initiated callers (slash command handlers like
    /// `handle_pr_create`) are exempt because the slash command itself is
    /// the user's explicit consent for the action. `pub(crate)` is
    /// intentional -- external crates must not be able to dispatch tools
    /// at all.
    ///
    /// `policy` controls the OS-level sandbox applied to `run_shell_command`.
    /// Other tools ignore it (their own seams, e.g. `safe_resolve_for_write`,
    /// enforce path containment).
    pub(crate) async fn execute(
        &self,
        name: &str,
        args: serde_json::Value,
        policy: SandboxPolicy,
    ) -> ToolResult {
        self.execute_with_shell_notice(name, args, policy, false)
            .await
    }

    /// Same as `execute`, but lets the caller attach a one-time shell audit
    /// marker for `run_shell_command`. Other tools ignore the extra flag.
    pub(crate) async fn execute_with_shell_notice(
        &self,
        name: &str,
        args: serde_json::Value,
        policy: SandboxPolicy,
        outside_sandbox_once: bool,
    ) -> ToolResult {
        self.execute_with_sandbox_mode(name, args, policy, outside_sandbox_once, None)
            .await
    }

    /// Same as `execute_with_shell_notice`, with the session sandbox mode
    /// threaded through tools that parse untrusted input (`grep_search`).
    pub(crate) async fn execute_with_sandbox_mode(
        &self,
        name: &str,
        args: serde_json::Value,
        policy: SandboxPolicy,
        outside_sandbox_once: bool,
        sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
    ) -> ToolResult {
        self.execute_with_sandbox_mode_cancellable(
            name,
            args,
            policy,
            outside_sandbox_once,
            sandbox_mode,
            None,
        )
        .await
    }

    /// Same as `execute_with_sandbox_mode`, but returns promptly when the
    /// session cancellation token fires. Shell calls receive the token so
    /// they can terminate their child process tree instead of waiting for
    /// the wall-clock timeout.
    pub(crate) async fn execute_with_sandbox_mode_cancellable(
        &self,
        name: &str,
        args: serde_json::Value,
        policy: SandboxPolicy,
        outside_sandbox_once: bool,
        sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
        cancel: Option<&CancellationToken>,
    ) -> ToolResult {
        if let Some(cancel) = cancel
            && cancel.is_cancelled()
        {
            return cancelled_tool_result(name);
        }

        let execute = self.execute_with_sandbox_mode_inner(
            name,
            args,
            policy,
            outside_sandbox_once,
            sandbox_mode,
            cancel,
        );
        if let Some(cancel) = cancel
            && name != "run_shell_command"
        {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => cancelled_tool_result(name),
                result = execute => result,
            }
        } else {
            execute.await
        }
    }

    async fn execute_with_sandbox_mode_inner(
        &self,
        name: &str,
        args: serde_json::Value,
        policy: SandboxPolicy,
        outside_sandbox_once: bool,
        sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
        cancel: Option<&CancellationToken>,
    ) -> ToolResult {
        match name {
            "read_file" => {
                let path = args.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
                let offset = args
                    .get("offset")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
                let limit = args
                    .get("limit")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize);
                filesystem::read_file(&self.cwd, path, offset, limit)
            }
            "write_file" => {
                let path = args.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
                let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
                filesystem::write_file(&self.cwd, path, content)
            }
            "edit" => {
                let path = args.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
                let old = args
                    .get("old_string")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let new = args
                    .get("new_string")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let replace_all = args
                    .get("replace_all")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                filesystem::edit_file(&self.cwd, path, old, new, replace_all)
            }
            "list_directory" => {
                let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                filesystem::list_directory(&self.cwd, path)
            }
            "grep_search" => {
                let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
                let glob = args.get("glob").and_then(|v| v.as_str());
                let max_results = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
                let path = args.get("path").and_then(|v| v.as_str());
                filesystem::search_file_contents_with_sandbox_mode(
                    &self.cwd,
                    pattern,
                    glob,
                    path,
                    max_results,
                    sandbox_mode,
                )
            }
            "run_shell_command" => {
                let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
                let timeout_seconds = args
                    .get("timeout")
                    .and_then(|v| v.as_u64())
                    .map(|ms| ms.saturating_add(999) / 1000)
                    .unwrap_or(60)
                    .max(1);
                let command_cwd = match args.get("directory").and_then(|v| v.as_str()) {
                    Some(directory) if !directory.trim().is_empty() => {
                        match safe_resolve(&self.cwd, directory) {
                            Ok(path) if path.is_dir() => path,
                            Ok(_) => {
                                return ToolResult {
                                    status: ToolStatus::RequestError,
                                    output: format!("Directory is not a directory: {directory}"),
                                };
                            }
                            Err(e) => {
                                return ToolResult {
                                    status: ToolStatus::RequestError,
                                    output: e,
                                };
                            }
                        }
                    }
                    _ => self.cwd.clone(),
                };
                shell::run_shell_command_cancellable(
                    &command_cwd,
                    command,
                    timeout_seconds,
                    policy,
                    outside_sandbox_once,
                    cancel,
                )
                .await
            }
            "activate_skill" => self.execute_activate_skill(args).await,
            // Any name not handled above is delegated to a configured MCP
            // server. This avoids a hardcoded list of server tool names
            // drifting out of sync with what each server actually exposes.
            _ => self.execute_mcp(name, args).await,
        }
    }

    /// Dispatch `activate_skill`. Looks up the requested name against
    /// the cached `SkillRegistry`; the schema's `enum` constraint should
    /// keep this from being called with an unknown name, but treat that
    /// case as a request error rather than an internal error so the
    /// model gets a clear correction.
    async fn execute_activate_skill(&self, args: serde_json::Value) -> ToolResult {
        let name = match args.get("name").and_then(|v| v.as_str()) {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => {
                return ToolResult {
                    status: ToolStatus::RequestError,
                    output: "activate_skill requires a non-empty `name` argument.".to_string(),
                };
            }
        };
        let skills = self.skills.read().await.clone();
        let Some(meta) = skills.get(&name) else {
            let available: Vec<&str> = skills.iter_sorted().map(|m| m.name.as_str()).collect();
            return ToolResult {
                status: ToolStatus::RequestError,
                output: format!(
                    "Unknown skill '{name}'. Available skills: {}",
                    available.join(", ")
                ),
            };
        };
        ToolResult {
            status: ToolStatus::Success,
            output: crate::agent::build_skill_payload(meta),
        }
    }

    async fn execute_mcp(&self, name: &str, args: serde_json::Value) -> ToolResult {
        if is_harness_only_mcp_tool(name) {
            return ToolResult {
                status: ToolStatus::RequestError,
                output: format!("MCP tool '{name}' is reserved for harness use."),
            };
        }
        let Some(client) = self.mcp_tool_servers.get(name).cloned() else {
            return ToolResult {
                status: ToolStatus::RequestError,
                output: format!(
                    "MCP tool '{name}' is unavailable: no configured server exposed it."
                ),
            };
        };
        // Reshape a few honest model mistakes (e.g. a bare string where the
        // tool's schema asks for an array) before dispatch, so they don't burn
        // a turn on a server-side -32602.
        let args = match client.tools().iter().find(|tool| tool.name == name) {
            Some(tool) => coerce_scalar_args_to_array(args, &tool.input_schema),
            None => args,
        };
        match client.call_tool(name, args).await {
            Ok(value) => {
                let output = if let Some(s) = value.as_str() {
                    s.to_string()
                } else {
                    serde_json::to_string_pretty(&value)
                        .unwrap_or_else(|e| format!("<failed to serialize MCP result: {e}>"))
                };
                ToolResult {
                    status: ToolStatus::Success,
                    output,
                }
            }
            Err(err) => ToolResult {
                status: ToolStatus::InternalError,
                output: format!("MCP tool '{name}' on '{}' failed: {err}", client.name()),
            },
        }
    }

    /// ACP `ToolKind` for a tool, used by the permission gate to classify calls.
    /// Looked up from the `TOOLS` table; tools we don't recognize fall
    /// through to `Other`. Bifrost-loaded tools added without an entry in
    /// `TOOLS` will hit this fallback (and a debug log).
    pub fn tool_kind(tool_name: &str) -> ToolKind {
        match tool_meta(tool_name) {
            Some(t) => t.kind,
            None => {
                tracing::debug!(
                    tool_name,
                    "tool_kind: unrecognized tool, classifying as Other"
                );
                ToolKind::Other
            }
        }
    }

    /// Static display name for a tool. Used as a fallback when a richer
    /// title can't be derived from the call's input args (notably for
    /// Bifrost-loaded tools we don't introspect by name in `announce`).
    pub fn display_name(tool_name: &str) -> &'static str {
        tool_meta(tool_name)
            .map(|t| t.display_name)
            .unwrap_or("Executing tool")
    }
}

/// Wrap scalar arguments in a single-element array wherever an MCP tool's
/// input schema declares an array-typed property.
///
/// Models routinely emit `{"file_patterns": "src/main.rs"}` instead of
/// `{"file_patterns": ["src/main.rs"]}`. The schema we advertise (forwarded
/// verbatim from the MCP server) correctly asks for an array, but a single
/// path is the most common case and the model elides the brackets. Without
/// this the call reaches bifrost as a bare string and serde rejects it with
/// `-32602 invalid type: string "src/main.rs", expected a sequence`, costing
/// the model a turn to self-correct.
///
/// We coerce only at the host boundary, right before dispatch; the advertised
/// schema is left untouched, so well-behaved callers still send arrays and the
/// model is still nudged toward the correct shape. Values already shaped as
/// arrays (or absent/null) are left alone, as are scalars whose own type the
/// schema already accepts -- so a field declared `"type": ["string", "array"]`
/// keeps a bare string intact rather than silently changing its serde variant.
pub(crate) fn coerce_scalar_args_to_array(
    args: serde_json::Value,
    input_schema: &serde_json::Value,
) -> serde_json::Value {
    use serde_json::Value;
    let Value::Object(mut map) = args else {
        return args;
    };
    let Some(properties) = input_schema.get("properties").and_then(Value::as_object) else {
        return Value::Object(map);
    };
    for (key, value) in map.iter_mut() {
        if value.is_array() || value.is_null() {
            continue;
        }
        let Some(property_schema) = properties.get(key) else {
            continue;
        };
        if scalar_needs_array_wrapping(property_schema, value) {
            let scalar = value.take();
            *value = Value::Array(vec![scalar]);
        }
    }
    Value::Object(map)
}

/// Whether a scalar `value` must be wrapped in a single-element array to satisfy
/// `property_schema`. True only when the schema declares an array type AND does
/// not already accept the scalar's own type: a strict array field (`"array"` or
/// `["array", "null"]`) gets a bare scalar wrapped, while a field that genuinely
/// accepts either form (`["string", "array"]`) leaves the scalar untouched.
fn scalar_needs_array_wrapping(
    property_schema: &serde_json::Value,
    value: &serde_json::Value,
) -> bool {
    let types = schema_declared_types(property_schema);
    types.contains(&"array") && !scalar_type_accepted(&types, value)
}

/// The JSON-schema `type` values a property node declares, accepting both the
/// scalar form (`"type": "array"`) and the union form (`"type": ["array", "null"]`).
fn schema_declared_types(property_schema: &serde_json::Value) -> Vec<&str> {
    match property_schema.get("type") {
        Some(serde_json::Value::String(t)) => vec![t.as_str()],
        Some(serde_json::Value::Array(types)) => {
            types.iter().filter_map(serde_json::Value::as_str).collect()
        }
        _ => Vec::new(),
    }
}

/// Whether `value`'s own JSON type is among the schema's declared types, meaning
/// it is already valid as-is and must not be wrapped. A JSON integer satisfies
/// both `integer` and `number`. Arrays and null never reach here.
fn scalar_type_accepted(types: &[&str], value: &serde_json::Value) -> bool {
    use serde_json::Value;
    types.iter().any(|t| match value {
        Value::String(_) => *t == "string",
        Value::Bool(_) => *t == "boolean",
        Value::Number(n) if n.is_i64() || n.is_u64() => *t == "integer" || *t == "number",
        Value::Number(_) => *t == "number",
        Value::Object(_) => *t == "object",
        Value::Array(_) | Value::Null => false,
    })
}

fn tool_def(name: &str, description: &str, parameters: serde_json::Value) -> ToolDefinition {
    ToolDefinition {
        r#type: "function".to_string(),
        function: FunctionDef {
            name: name.to_string(),
            description: description.to_string(),
            parameters,
        },
    }
}

/// Resolve a relative path against cwd and ensure it stays within cwd.
pub fn safe_resolve(cwd: &Path, requested: &str) -> Result<PathBuf, String> {
    let joined = cwd.join(requested);
    let resolved = joined
        .canonicalize()
        .map_err(|e| format!("Cannot resolve path '{}': {}", requested, e))?;
    let cwd_canonical = cwd
        .canonicalize()
        .map_err(|e| format!("Cannot resolve cwd: {}", e))?;
    if !resolved.starts_with(&cwd_canonical) {
        return Err(format!(
            "Path '{}' escapes the working directory",
            requested
        ));
    }
    Ok(resolved)
}

/// Like safe_resolve but allows the target (and intermediate ancestors) not to exist yet.
/// We walk up until we find an existing ancestor, canonicalize it, and verify it lies
/// under the canonical cwd. Returns the canonical cwd joined with the remaining tail,
/// which guarantees the final path resolves under cwd without relying on canonicalize
/// of the still-missing target.
pub fn safe_resolve_for_write(cwd: &Path, requested: &str) -> Result<PathBuf, String> {
    let cwd_canonical = cwd
        .canonicalize()
        .map_err(|e| format!("Cannot resolve cwd: {}", e))?;

    let joined = cwd.join(requested);

    // Walk up to the first existing ancestor (including the target itself if it exists).
    // Use symlink_metadata rather than exists(): exists() follows symlinks, so a dangling
    // symlink at the leaf would be reported as non-existent and we'd skip past it,
    // letting fs::write follow the link and write outside cwd. symlink_metadata reports
    // the symlink itself as "existing" so the canonicalize step below either resolves it
    // (rejecting if the target lies outside cwd) or errors on a dangling target.
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    let mut cursor: &Path = &joined;
    let existing = loop {
        if cursor.symlink_metadata().is_ok() {
            break cursor.to_path_buf();
        }
        match (cursor.file_name(), cursor.parent()) {
            (Some(name), Some(parent)) => {
                tail.push(name.to_os_string());
                cursor = parent;
            }
            _ => {
                return Err(format!(
                    "Cannot resolve path '{}': no existing ancestor",
                    requested
                ));
            }
        }
    };

    let existing_canonical = existing
        .canonicalize()
        .map_err(|e| format!("Cannot resolve ancestor of '{}': {}", requested, e))?;
    if !existing_canonical.starts_with(&cwd_canonical) {
        return Err(format!(
            "Path '{}' escapes the working directory",
            requested
        ));
    }

    // Reject any `..` components in the still-missing tail so an attacker
    // can't re-escape via unwritten path components.
    let mut resolved = existing_canonical;
    for component in tail.into_iter().rev() {
        if component == std::ffi::OsStr::new("..") || component == std::ffi::OsStr::new(".") {
            return Err(format!(
                "Path '{}' contains unsupported '..' or '.' components",
                requested
            ));
        }
        resolved.push(component);
    }

    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Allocate a fresh empty directory under the system temp dir for one test
    /// to scribble in. Caller is responsible for cleaning it up.
    fn fresh_tmp_dir(label: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("brokk-acp-rust-{}-{}", label, uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("create tmp dir");
        dir
    }

    /// Regression: a dangling symlink at the leaf must be rejected, not silently
    /// followed by the eventual fs::write at the call site. See issue #3408.
    #[cfg(unix)]
    #[test]
    fn safe_resolve_for_write_rejects_dangling_symlink_to_outside_cwd() {
        let cwd = fresh_tmp_dir("dangling-symlink");
        let outside = fresh_tmp_dir("dangling-target").join("does-not-exist-yet");
        std::os::unix::fs::symlink(&outside, cwd.join("evil")).expect("create symlink");

        let result = safe_resolve_for_write(&cwd, "evil");
        assert!(result.is_err(), "expected rejection, got Ok({:?})", result);

        std::fs::remove_dir_all(&cwd).ok();
        std::fs::remove_dir_all(outside.parent().unwrap()).ok();
    }

    /// A symlink whose target *exists* but lies outside cwd must also be rejected.
    /// This case worked before the fix; the test pins it down so a future change
    /// doesn't regress it.
    #[cfg(unix)]
    #[test]
    fn safe_resolve_for_write_rejects_live_symlink_to_outside_cwd() {
        let cwd = fresh_tmp_dir("live-symlink");
        let outside_dir = fresh_tmp_dir("live-target");
        let outside_file = outside_dir.join("real");
        std::fs::write(&outside_file, "hello").expect("seed outside file");
        std::os::unix::fs::symlink(&outside_file, cwd.join("evil")).expect("create symlink");

        let result = safe_resolve_for_write(&cwd, "evil");
        assert!(result.is_err(), "expected rejection, got Ok({:?})", result);

        std::fs::remove_dir_all(&cwd).ok();
        std::fs::remove_dir_all(&outside_dir).ok();
    }

    /// A symlink that points back inside cwd should still be allowed: the
    /// fix must not over-restrict legitimate intra-sandbox links.
    #[cfg(unix)]
    #[test]
    fn safe_resolve_for_write_allows_symlink_pointing_inside_cwd() {
        let cwd = fresh_tmp_dir("inside-symlink");
        let real = cwd.join("real.txt");
        std::fs::write(&real, "ok").expect("seed real file");
        std::os::unix::fs::symlink(&real, cwd.join("link")).expect("create symlink");

        let resolved = safe_resolve_for_write(&cwd, "link").expect("resolve must succeed");
        let cwd_canonical = cwd.canonicalize().unwrap();
        assert!(
            resolved.starts_with(&cwd_canonical),
            "resolved {:?} must stay under cwd {:?}",
            resolved,
            cwd_canonical
        );

        std::fs::remove_dir_all(&cwd).ok();
    }

    /// An intermediate directory that is a symlink to outside cwd must be
    /// rejected even if the leaf is a not-yet-existing file.
    #[cfg(unix)]
    #[test]
    fn safe_resolve_for_write_rejects_intermediate_symlink_escape() {
        let cwd = fresh_tmp_dir("intermediate-symlink");
        let outside = fresh_tmp_dir("intermediate-target");
        std::os::unix::fs::symlink(&outside, cwd.join("escape")).expect("create symlink");

        let result = safe_resolve_for_write(&cwd, "escape/newfile.txt");
        assert!(result.is_err(), "expected rejection, got Ok({:?})", result);

        std::fs::remove_dir_all(&cwd).ok();
        std::fs::remove_dir_all(&outside).ok();
    }

    /// Happy path: writing to a not-yet-existing file in an existing,
    /// symlink-free directory still resolves under cwd.
    #[test]
    fn safe_resolve_for_write_allows_new_file_in_existing_dir() {
        let cwd = fresh_tmp_dir("new-file");

        let resolved =
            safe_resolve_for_write(&cwd, "subdir/new.txt").expect("resolve must succeed");
        let cwd_canonical = cwd.canonicalize().unwrap();
        assert!(
            resolved.starts_with(&cwd_canonical),
            "resolved {:?} must stay under cwd {:?}",
            resolved,
            cwd_canonical
        );
        assert!(resolved.ends_with("subdir/new.txt"));

        std::fs::remove_dir_all(&cwd).ok();
    }

    /// Anti-drift: every built-in tool name must (1) have a `ToolMeta` row in
    /// the `TOOLS` table (otherwise the permission gate falls through to
    /// `Other` and the UI to a generic label), and (2) be advertised by
    /// `tool_definitions()` (otherwise the LLM never sees it). If you add a
    /// new built-in dispatch arm in `execute`, also add the name to
    /// `BUILTIN_TOOL_NAMES`, the `TOOLS` table, and `tool_definitions()`.
    use crate::skills::{SkillMeta, SkillScope};

    fn registry_with_skills(skills: Vec<SkillMeta>) -> ToolRegistry {
        let mut reg = SkillRegistry::default();
        for meta in skills {
            reg.insert_for_test(meta);
        }
        let cwd = std::env::temp_dir();
        ToolRegistry {
            cwd,
            mcp_clients: Vec::new(),
            mcp_tool_servers: HashMap::new(),
            advertised_builtin_tools: RwLock::new(
                BUILTIN_TOOL_NAMES
                    .iter()
                    .map(|name| (*name).to_string())
                    .collect(),
            ),
            skills: RwLock::new(Arc::new(reg)),
            agents: RwLock::new(Arc::new(AgentRegistry::default())),
        }
    }

    fn registry_with_agents(agents: Vec<crate::agents::AgentMeta>) -> ToolRegistry {
        let mut reg = AgentRegistry::default();
        for meta in agents {
            reg.insert_for_test(meta);
        }
        let cwd = std::env::temp_dir();
        ToolRegistry {
            cwd,
            mcp_clients: Vec::new(),
            mcp_tool_servers: HashMap::new(),
            advertised_builtin_tools: RwLock::new(
                BUILTIN_TOOL_NAMES
                    .iter()
                    .map(|name| (*name).to_string())
                    .collect(),
            ),
            skills: RwLock::new(Arc::new(SkillRegistry::default())),
            agents: RwLock::new(Arc::new(reg)),
        }
    }

    fn write_skill_fixture(name: &str, body: &str) -> (tempfile::TempDir, SkillMeta) {
        let tmp = tempfile::TempDir::new().unwrap();
        let skill_dir = tmp.path().join(name);
        std::fs::create_dir_all(&skill_dir).unwrap();
        let location = skill_dir.join("SKILL.md");
        std::fs::write(
            &location,
            format!("---\nname: {name}\ndescription: ds\n---\n{body}"),
        )
        .unwrap();
        let meta = SkillMeta {
            name: name.to_string(),
            description: "ds".to_string(),
            location,
            skill_dir,
            scope: SkillScope::Project,
        };
        (tmp, meta)
    }

    #[tokio::test]
    async fn activate_skill_tool_enum_restricted_to_discovered_names() {
        let (_a, meta_a) = write_skill_fixture("foo", "fb");
        let (_b, meta_b) = write_skill_fixture("bar", "bb");
        let registry = registry_with_skills(vec![meta_a, meta_b]);
        let defs = registry.tool_definitions().await;
        let activate = defs
            .iter()
            .find(|d| d.function.name == "activate_skill")
            .expect("activate_skill must be advertised");
        let enum_field = activate
            .function
            .parameters
            .pointer("/properties/name/enum")
            .expect("name property has an enum constraint")
            .as_array()
            .unwrap();
        let names: Vec<&str> = enum_field.iter().filter_map(|v| v.as_str()).collect();
        // Alphabetically sorted by SkillRegistry::iter_sorted.
        assert_eq!(names, vec!["bar", "foo"]);
    }

    #[tokio::test]
    async fn activate_skill_tool_absent_when_registry_empty() {
        let registry = registry_with_skills(vec![]);
        let defs = registry.tool_definitions().await;
        assert!(
            !defs.iter().any(|d| d.function.name == "activate_skill"),
            "activate_skill must be hidden when no skills are discovered"
        );
    }

    #[tokio::test]
    async fn activate_skill_returns_wrapped_body() {
        let (_t, meta) = write_skill_fixture("hello", "Greet the user briefly.\n");
        let registry = registry_with_skills(vec![meta]);
        let result = registry
            .execute(
                "activate_skill",
                json!({ "name": "hello" }),
                SandboxPolicy::WorkspaceWrite,
            )
            .await;
        assert!(matches!(result.status, ToolStatus::Success));
        assert!(result.output.starts_with("<skill_content name=\"hello\">"));
        assert!(result.output.contains("Greet the user briefly."));
        assert!(result.output.ends_with("</skill_content>"));
    }

    #[tokio::test]
    async fn activate_skill_rejects_unknown_name() {
        let (_t, meta) = write_skill_fixture("real-skill", "body");
        let registry = registry_with_skills(vec![meta]);
        let result = registry
            .execute(
                "activate_skill",
                json!({ "name": "nonexistent" }),
                SandboxPolicy::WorkspaceWrite,
            )
            .await;
        assert!(matches!(result.status, ToolStatus::RequestError));
        assert!(result.output.contains("Unknown skill 'nonexistent'"));
        assert!(result.output.contains("real-skill"));
    }

    #[tokio::test]
    async fn refresh_is_reserved_for_harness_dispatch() {
        let registry = registry_with_skills(vec![]);
        let result = registry
            .execute("refresh", json!({}), SandboxPolicy::WorkspaceWrite)
            .await;

        assert!(matches!(result.status, ToolStatus::RequestError));
        assert!(result.output.contains("reserved for harness use"));
        assert!(is_harness_only_mcp_tool("refresh"));
        assert!(!is_harness_only_mcp_tool("search_symbols"));
    }

    #[tokio::test]
    async fn builtin_tools_have_metadata_and_are_advertised() {
        let registry = ToolRegistry {
            cwd: std::env::temp_dir(),
            mcp_clients: Vec::new(),
            mcp_tool_servers: HashMap::new(),
            advertised_builtin_tools: RwLock::new(
                BUILTIN_TOOL_NAMES
                    .iter()
                    .map(|name| (*name).to_string())
                    .collect(),
            ),
            skills: RwLock::new(Arc::new(SkillRegistry::default())),
            agents: RwLock::new(Arc::new(AgentRegistry::default())),
        };
        let advertised: Vec<String> = registry
            .tool_definitions()
            .await
            .into_iter()
            .map(|d| d.function.name)
            .collect();

        for name in BUILTIN_TOOL_NAMES {
            assert!(
                TOOLS.iter().any(|t| t.name == *name),
                "built-in tool '{name}' is missing from the TOOLS metadata table"
            );
            assert!(
                advertised.iter().any(|a| a == name),
                "built-in tool '{name}' is missing from tool_definitions(); LLM will not see it"
            );
        }

        // The inverse: with bifrost disabled, advertised tools should be a
        // subset of the metadata table (no UI fallback for built-ins).
        for advertised_name in &advertised {
            assert!(
                TOOLS.iter().any(|t| t.name == advertised_name.as_str()),
                "tool_definitions() advertises '{advertised_name}' but it is missing from the TOOLS metadata table"
            );
        }
    }

    #[test]
    fn slopcop_bifrost_reporters_are_read_safe() {
        for name in SLOPCOP_BIFROST_READ_ONLY_TOOLS {
            assert_eq!(
                ToolRegistry::tool_kind(name),
                ToolKind::Read,
                "{name} must remain callable in read-only ACP sessions"
            );
        }
    }

    #[tokio::test]
    async fn shell_tool_schema_hides_sandbox_escalation_by_default() {
        let registry = registry_with_skills(vec![]);
        let defs = registry.tool_definitions().await;
        let shell = defs
            .iter()
            .find(|def| def.function.name == "run_shell_command")
            .expect("run_shell_command should be advertised");

        assert!(
            shell
                .function
                .parameters
                .pointer("/properties/sandbox_permissions")
                .is_none(),
            "first-attempt shell schema must not expose sandbox escalation"
        );
    }

    #[tokio::test]
    async fn shell_tool_schema_exposes_sandbox_escalation_when_enabled() {
        let registry = registry_with_skills(vec![]);
        let defs = registry.tool_definitions_with_shell_escalation(true).await;
        let shell = defs
            .iter()
            .find(|def| def.function.name == "run_shell_command")
            .expect("run_shell_command should be advertised");

        assert!(
            shell
                .function
                .parameters
                .pointer("/properties/sandbox_permissions")
                .is_some(),
            "retry shell schema must expose sandbox escalation"
        );
    }

    #[tokio::test]
    async fn tool_definitions_respect_filtered_builtin_set() {
        let registry = registry_with_skills(vec![]);
        registry
            .set_builtin_tools(
                ["edit", "write_file", "list_directory"]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
            )
            .await;
        let advertised: Vec<String> = registry
            .tool_definitions()
            .await
            .into_iter()
            .map(|d| d.function.name)
            .collect();

        assert!(advertised.iter().any(|name| name == "edit"));
        assert!(advertised.iter().any(|name| name == "write_file"));
        assert!(advertised.iter().any(|name| name == "list_directory"));
        assert!(!advertised.iter().any(|name| name == "read_file"));
        assert!(!advertised.iter().any(|name| name == "grep_search"));
        assert!(!advertised.iter().any(|name| name == "run_shell_command"));
    }

    #[tokio::test]
    async fn hidden_builtins_still_execute_for_non_llm_callers() {
        let registry = registry_with_skills(vec![]);
        registry.set_builtin_tools(HashSet::new()).await;

        #[cfg(target_os = "windows")]
        let command = "echo ok";
        #[cfg(not(target_os = "windows"))]
        let command = "printf ok";

        let result = registry
            .execute(
                "run_shell_command",
                json!({ "command": command }),
                SandboxPolicy::None,
            )
            .await;

        assert!(
            matches!(result.status, ToolStatus::Success),
            "hidden builtins should still execute for non-LLM callers; output={}",
            result.output
        );
        assert_eq!(result.output.trim(), "ok");
    }

    #[tokio::test]
    async fn shell_tool_returns_promptly_on_cancellation() {
        let registry = registry_with_skills(vec![]);
        let cancel = CancellationToken::new();
        let cancel_from_task = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            cancel_from_task.cancel();
        });

        #[cfg(target_os = "windows")]
        let command = "ping 127.0.0.1 -n 30 > nul";
        #[cfg(not(target_os = "windows"))]
        let command = "sleep 30";

        let started = std::time::Instant::now();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            registry.execute_with_sandbox_mode_cancellable(
                "run_shell_command",
                json!({ "command": command, "timeout": 30_000 }),
                SandboxPolicy::None,
                false,
                None,
                Some(&cancel),
            ),
        )
        .await
        .expect("cancelled shell command should return before the test timeout");

        assert!(
            matches!(result.status, ToolStatus::RequestError),
            "cancelled shell command should report a request error"
        );
        assert!(
            result.output.contains("cancelled"),
            "cancelled shell command should explain cancellation; output={}",
            result.output
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "cancelled shell command waited too long"
        );
    }

    /// `task` is gated on having at least one discovered subagent.
    /// With an empty `AgentRegistry`, the LLM shouldn't see the tool at
    /// all -- exposing it with an empty `enum` would just teach the
    /// model to guess names that don't exist.
    #[tokio::test]
    async fn task_tool_hidden_when_no_subagents() {
        let registry = registry_with_skills(vec![]);
        let advertised: Vec<String> = registry
            .tool_definitions()
            .await
            .into_iter()
            .map(|d| d.function.name)
            .collect();
        assert!(
            !advertised.iter().any(|n| n == "task"),
            "task should not be advertised without subagents; got {advertised:?}"
        );
    }

    /// Once at least one subagent is in the registry, `task` is
    /// advertised with `subagent_type` constrained to the discovered
    /// names via JSON-schema `enum` (mirrors `activate_skill`).
    #[tokio::test]
    async fn task_tool_exposed_with_subagent_enum() {
        use crate::agents::{AgentMeta, AgentScope};
        let registry = registry_with_agents(vec![
            AgentMeta {
                name: "doc-writer".into(),
                description: "Drafts docs from code".into(),
                location: PathBuf::from("/tmp/doc-writer.md"),
                scope: AgentScope::Project,
            },
            AgentMeta {
                name: "bug-hunter".into(),
                description: "Hunts for regressions".into(),
                location: PathBuf::from("/tmp/bug-hunter.md"),
                scope: AgentScope::User,
            },
        ]);
        let defs = registry.tool_definitions().await;
        let task_def = defs
            .iter()
            .find(|d| d.function.name == "task")
            .expect("task tool should be advertised");

        // Enum must contain the discovered names.
        let enum_vals = task_def.function.parameters["properties"]["subagent_type"]["enum"]
            .as_array()
            .expect("subagent_type should constrain via enum");
        let mut got: Vec<String> = enum_vals
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        got.sort();
        assert_eq!(got, vec!["bug-hunter", "doc-writer"]);

        // Description should surface the catalog so the model can pick.
        assert!(
            task_def.function.description.contains("doc-writer"),
            "catalog should mention each subagent; got: {}",
            task_def.function.description
        );
        assert!(task_def.function.description.contains("bug-hunter"));
    }

    /// MCP schemas often ask for arrays. The host must wrap scalar strings into
    /// single-element arrays so servers do not reject them with invalid params.
    #[test]
    fn coerce_wraps_scalar_string_for_array_property() {
        let schema = json!({
            "type": "object",
            "properties": {
                "file_patterns": { "type": "array", "items": { "type": "string" } }
            }
        });
        let coerced =
            coerce_scalar_args_to_array(json!({ "file_patterns": "src/main.rs" }), &schema);
        assert_eq!(coerced, json!({ "file_patterns": ["src/main.rs"] }));
    }

    /// A correctly-shaped array argument must pass through untouched -- the
    /// coercion only rescues scalars, it never re-wraps an existing array.
    #[test]
    fn coerce_leaves_array_argument_untouched() {
        let schema = json!({
            "type": "object",
            "properties": {
                "file_patterns": { "type": "array", "items": { "type": "string" } }
            }
        });
        let args = json!({ "file_patterns": ["src/main.rs", "src/lib.rs"] });
        assert_eq!(coerce_scalar_args_to_array(args.clone(), &schema), args);
    }

    /// Non-array properties (and properties absent from the schema) must keep
    /// their scalar value; we only reshape where the schema declares an array.
    #[test]
    fn coerce_leaves_non_array_properties_untouched() {
        let schema = json!({
            "type": "object",
            "properties": {
                "patterns": { "type": "array", "items": { "type": "string" } },
                "max_results": { "type": "number" }
            }
        });
        let args = json!({
            "patterns": "McpClient",
            "max_results": 10,
            "unschema'd": "left as-is"
        });
        let coerced = coerce_scalar_args_to_array(args, &schema);
        assert_eq!(
            coerced,
            json!({
                "patterns": ["McpClient"],
                "max_results": 10,
                "unschema'd": "left as-is"
            })
        );
    }

    /// Nullable array fields advertise `"type": ["array", "null"]`; a scalar
    /// supplied for one must still be wrapped, but an explicit null stays null
    /// (absent/optional), never `[null]`.
    #[test]
    fn coerce_handles_nullable_array_union_type() {
        let schema = json!({
            "type": "object",
            "properties": {
                "globs": { "type": ["array", "null"], "items": { "type": "string" } }
            }
        });
        assert_eq!(
            coerce_scalar_args_to_array(json!({ "globs": "*.rs" }), &schema),
            json!({ "globs": ["*.rs"] })
        );
        assert_eq!(
            coerce_scalar_args_to_array(json!({ "globs": null }), &schema),
            json!({ "globs": null })
        );
    }

    /// A field that accepts either form (`"type": ["string", "array"]`) must
    /// leave a bare string intact: the string is already valid, and wrapping it
    /// could flip which serde variant the server deserializes. An explicit array
    /// for the same field also passes through unchanged.
    #[test]
    fn coerce_leaves_scalar_for_string_or_array_union() {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": { "type": ["string", "array"], "items": { "type": "string" } }
            }
        });
        assert_eq!(
            coerce_scalar_args_to_array(json!({ "query": "McpClient" }), &schema),
            json!({ "query": "McpClient" })
        );
        assert_eq!(
            coerce_scalar_args_to_array(json!({ "query": ["a", "b"] }), &schema),
            json!({ "query": ["a", "b"] })
        );
    }

    /// A numeric value for a `["number", "array"]` union is already valid and
    /// must not be wrapped; a JSON integer satisfies the `number` type.
    #[test]
    fn coerce_leaves_number_for_number_or_array_union() {
        let schema = json!({
            "type": "object",
            "properties": {
                "weights": { "type": ["number", "array"] }
            }
        });
        assert_eq!(
            coerce_scalar_args_to_array(json!({ "weights": 5 }), &schema),
            json!({ "weights": 5 })
        );
    }
}
