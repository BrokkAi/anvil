mod filesystem;
pub mod sandbox;
mod shell;
mod web_search;

use crate::agents::AgentRegistry;
use crate::llm_client::{FunctionDef, ToolDefinition};
use crate::mcp::{McpClient, McpServerConfig};
use crate::skills::{SkillKind, SkillRegistry};
use agent_client_protocol::schema::v1::ToolKind;
use sandbox::SandboxPolicy;
use serde::de;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer};
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

#[derive(PartialEq, Eq)]
pub enum ToolStatus {
    Success,
    RequestError,
    InternalError,
}

/// Whether a rendered tool result text represents a failure. Matches the
/// `"Error: "` / `"Internal error: "` status prefixes `tool_result_to_execution`
/// applies in `tool_loop.rs` for `ToolStatus::RequestError` / `InternalError`,
/// plus the `"Tool use denied"` prefix every permission-gate rejection uses
/// (user rejections, auto-permission denials, read-only mode, oversized
/// permission cards -- see `tool_loop.rs` / `tool_loop/announce.rs`).
pub(crate) fn tool_result_failed(result: &str) -> bool {
    result.starts_with("Error:")
        || result.starts_with("Internal error:")
        || result.starts_with("Tool use denied")
}

#[derive(Debug, Deserialize)]
struct ReadFileArgs {
    file_path: String,
    #[serde(default, deserialize_with = "deserialize_optional_non_null")]
    offset: Option<usize>,
    #[serde(default, deserialize_with = "deserialize_optional_non_null")]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct WriteFileArgs {
    file_path: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct DeleteFileArgs {
    file_path: String,
}

#[derive(Debug, Deserialize)]
struct MoveFileArgs {
    source_path: String,
    destination_path: String,
}

#[derive(Debug, Clone, Deserialize)]
struct EditEntryArgs {
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

#[derive(Debug)]
struct EditFileArgs {
    file_path: String,
    edits: Vec<EditEntryArgs>,
}

#[derive(Debug, Deserialize)]
struct BatchEditFileArgs {
    file_path: String,
    edits: Vec<EditEntryArgs>,
}

#[derive(Debug, Deserialize)]
struct FlatEditFileArgs {
    file_path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

impl<'de> Deserialize<'de> for EditFileArgs {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        if value.get("edits").is_some() {
            let args: BatchEditFileArgs =
                deserialize_json_value_with_path(value).map_err(de::Error::custom)?;
            return Ok(Self {
                file_path: args.file_path,
                edits: args.edits,
            });
        }

        if value.get("old_string").is_some()
            || value.get("new_string").is_some()
            || value.get("replace_all").is_some()
        {
            let args: FlatEditFileArgs =
                deserialize_json_value_with_path(value).map_err(de::Error::custom)?;
            return Ok(Self {
                file_path: args.file_path,
                edits: vec![EditEntryArgs {
                    old_string: args.old_string,
                    new_string: args.new_string,
                    replace_all: args.replace_all,
                }],
            });
        }

        let args: BatchEditFileArgs =
            deserialize_json_value_with_path(value).map_err(de::Error::custom)?;
        Ok(Self {
            file_path: args.file_path,
            edits: args.edits,
        })
    }
}

#[derive(Debug, Deserialize)]
struct ListDirectoryArgs {
    path: String,
}

fn default_grep_limit() -> usize {
    50
}

#[derive(Debug, Deserialize)]
struct GrepSearchArgs {
    pattern: String,
    #[serde(default, deserialize_with = "deserialize_optional_non_null")]
    glob: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_non_null")]
    path: Option<String>,
    #[serde(default = "default_grep_limit")]
    limit: usize,
}

fn default_web_search_limit() -> usize {
    5
}

#[derive(Debug, Deserialize)]
struct WebSearchArgs {
    query: String,
    #[serde(default = "default_web_search_limit")]
    max_results: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ShellSandboxPermissionArg {
    /// Run inside the active sandbox (the implicit behavior when the field is
    /// omitted). Accepted so a model can state the default explicitly without a
    /// deserialization error.
    UseDefault,
    /// Ask the user for one-time approval to run this command outside the
    /// sandbox. The tool loop reads this intent off the raw arguments
    /// (`tool_loop::shell_sandbox_escalation_requested`); the parsed value here
    /// only validates the schema enum.
    RequireEscalated,
}

impl ShellSandboxPermissionArg {
    /// Argument key carrying the per-command sandbox override. Single source of
    /// truth shared by the advertised schema (below) and the gate's raw-JSON
    /// matcher (`tool_loop::shell_sandbox_escalation_requested`), so the name the
    /// model is told and the name the gate looks for cannot drift. Kept equal to
    /// the serde `rename` on `RunShellCommandArgs::_sandbox_permissions`.
    pub(crate) const FIELD: &'static str = "sandbox_permissions";
    /// The only behavior-changing value: requests one-time outside-sandbox
    /// approval. Equal to the serde name of [`Self::RequireEscalated`].
    pub(crate) const REQUIRE_ESCALATED: &'static str = "require_escalated";
    /// Advertised JSON-schema enum, in order. Each entry must deserialize into a
    /// variant; `run_shell_command_args_accept_every_schema_value` enforces that
    /// these stay in lockstep with the `snake_case` variant names.
    pub(crate) const SCHEMA_VALUES: [&'static str; 2] = ["use_default", Self::REQUIRE_ESCALATED];
}

#[derive(Debug, Deserialize)]
struct DiagnosticsArgs {
    #[serde(default, deserialize_with = "deserialize_optional_non_null")]
    file_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RunShellCommandArgs {
    command: String,
    /// The advertised timeout field, in seconds. Clamped to
    /// `[shell::MIN_TIMEOUT_SECONDS, shell::MAX_TIMEOUT_SECONDS]`.
    #[serde(default, deserialize_with = "deserialize_optional_non_null")]
    timeout_seconds: Option<u64>,
    /// Legacy millisecond timeout. Deserialized but deliberately absent from
    /// the advertised schema: models kept reading "timeout" as seconds and
    /// having their commands killed at the 1s rounding floor. Kept so
    /// in-process callers and replayed traces that still pass milliseconds
    /// behave exactly as they did, with only the ceiling moved.
    /// `timeout_seconds` wins when both are present.
    #[serde(default, deserialize_with = "deserialize_optional_non_null")]
    timeout: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_optional_non_null")]
    directory: Option<String>,
    #[serde(
        default,
        deserialize_with = "deserialize_optional_non_null",
        rename = "description"
    )]
    _description: Option<String>,
    #[serde(
        default,
        deserialize_with = "deserialize_optional_non_null",
        rename = "sandbox_permissions"
    )]
    _sandbox_permissions: Option<ShellSandboxPermissionArg>,
}

#[derive(Debug, Deserialize)]
struct ActivateSkillArgs {
    name: String,
}

#[cfg(test)]
trait BuiltinArgsContract {
    const REQUIRED_FIELDS: &'static [&'static str];
    const PROPERTY_TYPES: &'static [(&'static str, &'static str)];
    /// `(property, advertised enum values)` pairs the schema must expose. Empty
    /// for tools with no enum-constrained field; overridden where a field's
    /// allowed values are pinned to a Rust enum, so the hand-written schema
    /// can't silently drift from the deserializer.
    const ENUM_VALUES: &'static [(&'static str, &'static [&'static str])] = &[];
}

#[cfg(test)]
impl BuiltinArgsContract for ReadFileArgs {
    const REQUIRED_FIELDS: &'static [&'static str] = &["file_path"];
    const PROPERTY_TYPES: &'static [(&'static str, &'static str)] = &[
        ("file_path", "string"),
        ("offset", "integer"),
        ("limit", "integer"),
    ];
}

#[cfg(test)]
impl BuiltinArgsContract for WriteFileArgs {
    const REQUIRED_FIELDS: &'static [&'static str] = &["file_path", "content"];
    const PROPERTY_TYPES: &'static [(&'static str, &'static str)] =
        &[("file_path", "string"), ("content", "string")];
}

#[cfg(test)]
impl BuiltinArgsContract for DeleteFileArgs {
    const REQUIRED_FIELDS: &'static [&'static str] = &["file_path"];
    const PROPERTY_TYPES: &'static [(&'static str, &'static str)] = &[("file_path", "string")];
}

#[cfg(test)]
impl BuiltinArgsContract for MoveFileArgs {
    const REQUIRED_FIELDS: &'static [&'static str] = &["source_path", "destination_path"];
    const PROPERTY_TYPES: &'static [(&'static str, &'static str)] =
        &[("source_path", "string"), ("destination_path", "string")];
}

#[cfg(test)]
impl BuiltinArgsContract for EditFileArgs {
    const REQUIRED_FIELDS: &'static [&'static str] = &["file_path", "edits"];
    const PROPERTY_TYPES: &'static [(&'static str, &'static str)] =
        &[("file_path", "string"), ("edits", "array")];
}

#[cfg(test)]
impl BuiltinArgsContract for ListDirectoryArgs {
    const REQUIRED_FIELDS: &'static [&'static str] = &["path"];
    const PROPERTY_TYPES: &'static [(&'static str, &'static str)] = &[("path", "string")];
}

#[cfg(test)]
impl BuiltinArgsContract for GrepSearchArgs {
    const REQUIRED_FIELDS: &'static [&'static str] = &["pattern"];
    const PROPERTY_TYPES: &'static [(&'static str, &'static str)] = &[
        ("pattern", "string"),
        ("glob", "string"),
        ("path", "string"),
        ("limit", "integer"),
    ];
}

#[cfg(test)]
impl BuiltinArgsContract for WebSearchArgs {
    const REQUIRED_FIELDS: &'static [&'static str] = &["query"];
    const PROPERTY_TYPES: &'static [(&'static str, &'static str)] =
        &[("query", "string"), ("max_results", "integer")];
}

#[cfg(test)]
impl BuiltinArgsContract for DiagnosticsArgs {
    const REQUIRED_FIELDS: &'static [&'static str] = &[];
    const PROPERTY_TYPES: &'static [(&'static str, &'static str)] = &[("file_path", "string")];
}

#[cfg(test)]
impl BuiltinArgsContract for RunShellCommandArgs {
    const REQUIRED_FIELDS: &'static [&'static str] = &["command"];
    const PROPERTY_TYPES: &'static [(&'static str, &'static str)] = &[
        ("command", "string"),
        ("timeout_seconds", "integer"),
        ("description", "string"),
        ("directory", "string"),
        (ShellSandboxPermissionArg::FIELD, "string"),
    ];
    const ENUM_VALUES: &'static [(&'static str, &'static [&'static str])] = &[(
        ShellSandboxPermissionArg::FIELD,
        &ShellSandboxPermissionArg::SCHEMA_VALUES,
    )];
}

#[cfg(test)]
impl BuiltinArgsContract for ActivateSkillArgs {
    const REQUIRED_FIELDS: &'static [&'static str] = &["name"];
    const PROPERTY_TYPES: &'static [(&'static str, &'static str)] = &[("name", "string")];
}

fn deserialize_optional_non_null<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

fn deserialize_json_value_with_path<T: DeserializeOwned>(
    value: serde_json::Value,
) -> Result<T, String> {
    let json = value.to_string();
    let mut deserializer = serde_json::Deserializer::from_str(&json);
    serde_path_to_error::deserialize(&mut deserializer).map_err(|err| err.to_string())
}

fn parse_builtin_args<T: DeserializeOwned>(
    tool_name: &str,
    args: serde_json::Value,
) -> Result<T, ToolResult> {
    let json = args.to_string();
    let mut deserializer = serde_json::Deserializer::from_str(&json);
    serde_path_to_error::deserialize(&mut deserializer).map_err(|err| ToolResult {
        status: ToolStatus::RequestError,
        output: format!("Invalid arguments for `{tool_name}`: {err}"),
    })
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
    concurrency_safe: bool,
}

const TOOLS: &[ToolMeta] = &[
    // --- Built-in tools (executed inline in `ToolRegistry::execute`) -------
    ToolMeta {
        name: "read_file",
        kind: ToolKind::Read,
        display_name: "Reading file",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "write_file",
        kind: ToolKind::Edit,
        display_name: "Writing file",
        concurrency_safe: false,
    },
    ToolMeta {
        name: "delete_file",
        kind: ToolKind::Delete,
        display_name: "Deleting file",
        concurrency_safe: false,
    },
    ToolMeta {
        name: "move_file",
        kind: ToolKind::Move,
        display_name: "Moving file",
        concurrency_safe: false,
    },
    ToolMeta {
        name: "edit",
        kind: ToolKind::Edit,
        display_name: "Editing file",
        concurrency_safe: false,
    },
    ToolMeta {
        name: "list_directory",
        kind: ToolKind::Read,
        display_name: "Listing directory",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "grep_search",
        kind: ToolKind::Search,
        display_name: "Searching file contents",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "web_search",
        kind: ToolKind::Search,
        display_name: "Searching the web",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "diagnostics",
        kind: ToolKind::Read,
        display_name: "Getting diagnostics",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "run_shell_command",
        kind: ToolKind::Execute,
        display_name: "Running shell command",
        concurrency_safe: false,
    },
    ToolMeta {
        name: "update_plan",
        kind: ToolKind::Read,
        display_name: "Updating plan",
        concurrency_safe: false,
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
        concurrency_safe: true,
    },
    ToolMeta {
        name: "get_active_workspace",
        kind: ToolKind::Read,
        display_name: "Getting active workspace",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "search_symbols",
        kind: ToolKind::Search,
        display_name: "Searching for symbols",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "search_ast",
        kind: ToolKind::Search,
        display_name: "Searching AST",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "query_code",
        kind: ToolKind::Search,
        display_name: "Querying code structure",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "get_symbol_locations",
        kind: ToolKind::Search,
        display_name: "Finding symbol locations",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "get_symbol_ancestors",
        kind: ToolKind::Search,
        display_name: "Finding symbol ancestors",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "get_symbol_summaries",
        kind: ToolKind::Search,
        display_name: "Getting symbol summaries",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "get_symbol_sources",
        kind: ToolKind::Search,
        display_name: "Fetching symbol source",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "most_relevant_files",
        kind: ToolKind::Search,
        display_name: "Finding related files",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "scan_usages_by_reference",
        kind: ToolKind::Search,
        display_name: "Scanning symbol usages",
        concurrency_safe: true,
    },
    // Compatibility classification for Bifrost versions that advertise the
    // legacy unsplit reference name.
    ToolMeta {
        name: "scan_usages",
        kind: ToolKind::Search,
        display_name: "Scanning symbol usages",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "usage_graph",
        kind: ToolKind::Search,
        display_name: "Building usage graph",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "get_definitions_by_reference",
        kind: ToolKind::Search,
        display_name: "Finding definition",
        concurrency_safe: true,
    },
    // Compatibility classification for the legacy singular reference name.
    ToolMeta {
        name: "get_definition_by_reference",
        kind: ToolKind::Search,
        display_name: "Finding definition",
        concurrency_safe: true,
    },
    ToolMeta {
        // bifrost returns the non-mutating rename edit set (it never writes),
        // with readOnlyHint=true, so it is classified as a read tool here.
        name: "rename_symbol",
        kind: ToolKind::Read,
        display_name: "Computing symbol rename",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "semantic_search",
        kind: ToolKind::Search,
        display_name: "Searching semantically",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "get_file_contents",
        kind: ToolKind::Read,
        display_name: "Reading file contents",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "classify_test_files",
        kind: ToolKind::Read,
        display_name: "Classifying test files",
        concurrency_safe: true,
    },
    // Compatibility classification for the legacy test-classification name.
    ToolMeta {
        name: "contains_tests",
        kind: ToolKind::Read,
        display_name: "Checking for test files",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "find_filenames",
        kind: ToolKind::Search,
        display_name: "Finding filenames",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "find_files_containing",
        kind: ToolKind::Search,
        display_name: "Finding files containing text",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "search_file_contents",
        kind: ToolKind::Search,
        display_name: "Searching file contents",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "list_files",
        kind: ToolKind::Read,
        display_name: "Listing files",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "skim_files",
        kind: ToolKind::Read,
        display_name: "Skimming files",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "search_git_commit_messages",
        kind: ToolKind::Search,
        display_name: "Searching git commit messages",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "get_git_log",
        kind: ToolKind::Read,
        display_name: "Reading git log",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "get_commit_diff",
        kind: ToolKind::Read,
        display_name: "Reading commit diff",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "jq",
        kind: ToolKind::Search,
        display_name: "Querying JSON",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "xml_skim",
        kind: ToolKind::Read,
        display_name: "Skimming XML",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "xml_select",
        kind: ToolKind::Search,
        display_name: "Selecting XML",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "compute_cyclomatic_complexity",
        kind: ToolKind::Read,
        display_name: "Computing cyclomatic complexity",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "compute_cognitive_complexity",
        kind: ToolKind::Read,
        display_name: "Computing cognitive complexity",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "report_comment_density_for_code_unit",
        kind: ToolKind::Read,
        display_name: "Reporting comment density",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "report_comment_density_for_files",
        kind: ToolKind::Read,
        display_name: "Reporting file comment density",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "report_exception_handling_smells",
        kind: ToolKind::Read,
        display_name: "Reporting exception handling smells",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "report_test_assertion_smells",
        kind: ToolKind::Read,
        display_name: "Reporting test assertion smells",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "report_structural_clone_smells",
        kind: ToolKind::Read,
        display_name: "Reporting structural clone smells",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "report_long_method_and_god_object_smells",
        kind: ToolKind::Read,
        display_name: "Reporting long method and god object smells",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "report_dead_code_and_unused_abstraction_smells",
        kind: ToolKind::Read,
        display_name: "Reporting dead code smells",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "report_secret_like_code",
        kind: ToolKind::Read,
        display_name: "Reporting secret-like code",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "analyze_git_hotspots",
        kind: ToolKind::Read,
        display_name: "Analyzing git hotspots",
        concurrency_safe: true,
    },
    ToolMeta {
        name: "analyze_commit",
        kind: ToolKind::Read,
        display_name: "Analyzing commit",
        concurrency_safe: true,
    },
    // `activate_workspace` and `refresh` mutate analyzer state, so they
    // stay `Other` rather than `Read`: prompted in `default`, refused in
    // `readOnly`.
    ToolMeta {
        name: "activate_workspace",
        kind: ToolKind::Other,
        display_name: "Activating workspace",
        concurrency_safe: false,
    },
    ToolMeta {
        name: "refresh",
        kind: ToolKind::Other,
        display_name: "Refreshing analyzer index",
        concurrency_safe: false,
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
        concurrency_safe: true,
    },
    // --- Subagent dispatch -------------------------------------------------
    // Like `activate_skill`, registered dynamically in `tool_definitions()`
    // only when at least one subagent is discovered. Classified `Other` by
    // default because inherited lanes have transitive effects from whatever
    // tools the subagent invokes; the tool loop special-cases lanes whose
    // effective permission mode is read-only. Actual dispatch happens in
    // `tool_loop::run` (not `ToolRegistry::execute`) because the subagent
    // needs `llm`, `spawned_cx`, and `sessions` -- none of which the registry
    // sees.
    ToolMeta {
        name: "task",
        kind: ToolKind::Other,
        display_name: "Running subagent",
        concurrency_safe: false,
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
    "delete_file",
    "move_file",
    "edit",
    "list_directory",
    "grep_search",
    "web_search",
    "diagnostics",
    "run_shell_command",
    "update_plan",
];

fn is_builtin_tool(name: &str) -> bool {
    BUILTIN_TOOL_NAMES.contains(&name)
}

/// One row of the static tool catalog exposed over the HTTP API
/// (`GET /v1/tools`). Derived from the `TOOLS` metadata table so the HTTP
/// surface cannot drift from the permission gate's view of the harness.
/// Rows for MCP-loaded tools describe Anvil's default Bifrost toolset; a
/// session's live registry may expose fewer (server disabled) or more
/// (extra MCP servers) at prompt time.
#[cfg(feature = "http-api")]
pub(crate) struct ToolCatalogEntry {
    pub(crate) name: &'static str,
    pub(crate) kind: ToolKind,
    pub(crate) display_name: &'static str,
    pub(crate) concurrency_safe: bool,
    pub(crate) builtin: bool,
}

#[cfg(feature = "http-api")]
pub(crate) fn tool_catalog() -> Vec<ToolCatalogEntry> {
    TOOLS
        .iter()
        .map(|meta| ToolCatalogEntry {
            name: meta.name,
            kind: meta.kind,
            display_name: meta.display_name,
            concurrency_safe: meta.concurrency_safe,
            builtin: is_builtin_tool(meta.name),
        })
        .collect()
}

fn is_harness_only_mcp_tool(name: &str) -> bool {
    name == "refresh"
}

/// Description advertised to the model for an MCP tool. Overrides bifrost's own
/// description where the harness changes a tool's observable behaviour.
///
/// `semantic_search` results are transparently reranked by the harness (see
/// `crate::semantic_rerank`), so the model receives a single relevance-ordered
/// list of hits with source/summaries -- not bifrost's raw ranked lists.
///
/// The description says "semantic", not "semantic + lexical": bifrost deleted
/// its BM25 arm in c353c862, so promising the model a lexical leg would
/// misdescribe the tool and invite keyword-shaped queries that dense retrieval
/// handles worst.
fn mcp_tool_description<'a>(name: &str, original: &'a str) -> &'a str {
    const SEMANTIC_SEARCH: &str = "Semantic code search. Given a natural-language \
        query, returns a single relevance-ordered list of the most relevant symbols and files, \
        each with its source or a summary. Results are reranked for relevance to your query and \
        the current task, so prefer the order given and start from the top.";
    match name {
        "semantic_search" => SEMANTIC_SEARCH,
        _ => original,
    }
}

/// Unified tool registry: filesystem tools + shell + configured
/// MCP tools + Agent Skills activation.
///
/// `skills` is wrapped in `RwLock` so the session can swap in a fresh
/// `SkillRegistry` after `update_cwd` without rebuilding the registry
/// (which would re-spawn MCP subprocesses).
pub struct ToolRegistry {
    cwd: PathBuf,
    additional_roots: Vec<PathBuf>,
    mcp_clients: Vec<Arc<McpClient>>,
    mcp_tool_servers: HashMap<String, Arc<McpClient>>,
    advertised_builtin_tools: RwLock<HashSet<String>>,
    skills: RwLock<Arc<SkillRegistry>>,
    agents: RwLock<Arc<AgentRegistry>>,
    /// Hooks contributed by enabled plugins, captured at registry build
    /// time. Ordered; executed by `tool_loop::execute_tool` around each
    /// tool call.
    plugin_hooks: Vec<crate::plugins::HookCommand>,
    /// Post-capture shell-output minimizer; `None` when disabled via
    /// `--no-shell-minimizer`.
    shell_minimizer: Option<shell::ShellMinimizer>,
    /// Guards the one-time semantic index readiness wait; see
    /// `semantic_readiness`.
    semantic_readiness: tokio::sync::OnceCell<()>,
    lsp: Option<Arc<crate::lsp::LspManager>>,
}

fn render_mcp_instructions(mut entries: Vec<(String, String)>) -> Option<String> {
    entries.retain(|(_, instructions)| !instructions.trim().is_empty());
    entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    if entries.is_empty() {
        return None;
    }

    let mut block = String::from("<mcp_instructions>\n");
    for (name, instructions) in entries {
        block.push_str(&format!(
            "  <server name=\"{name}\">\n{}\n  </server>\n",
            instructions.trim()
        ));
    }
    block.push_str("</mcp_instructions>");
    Some(block)
}

/// Feature knobs for `ToolRegistry::new`. Bundled so the constructor stays
/// under clippy's argument-count limit while each knob remains explicit at
/// call sites.
pub(crate) struct ToolRegistryOptions {
    pub lsp_settings: crate::lsp::LspSettings,
    pub shell_minimizer_enabled: bool,
}

impl ToolRegistry {
    /// Working directory this registry is rooted in.
    pub(crate) fn cwd(&self) -> &Path {
        &self.cwd
    }

    pub(crate) fn additional_roots(&self) -> &[PathBuf] {
        &self.additional_roots
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

    /// Render instructions from currently connected MCP servers for inclusion
    /// in the model's system prompt. Clients are sorted by server name so the
    /// prefix remains deterministic even when configuration order changes.
    pub(crate) async fn mcp_instructions(&self) -> Option<String> {
        let mut entries = Vec::new();
        for client in &self.mcp_clients {
            if let Some(instructions) = client.instructions().await {
                entries.push((client.name().to_string(), instructions));
            }
        }
        render_mcp_instructions(entries)
    }

    pub async fn new(
        cwd: PathBuf,
        additional_roots: Vec<PathBuf>,
        mcp_servers: Vec<McpServerConfig>,
        skills: Arc<SkillRegistry>,
        agents: Arc<AgentRegistry>,
        plugin_hooks: Vec<crate::plugins::HookCommand>,
        options: ToolRegistryOptions,
    ) -> Self {
        // Best-effort sweep of any stale seatbelt policy files left by a
        // previous SIGKILL/panic. Bounded by file age so we don't yank a
        // profile from a concurrent in-flight shell call.
        sandbox::cleanup_stale_policy_files();
        shell::cleanup_stale_shell_outputs(&cwd);

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
        let lsp = if options
            .lsp_settings
            .servers
            .iter()
            .any(|server| server.enabled)
        {
            Some(crate::lsp::LspManager::start(cwd.clone(), options.lsp_settings).await)
        } else {
            None
        };
        let shell_minimizer = options
            .shell_minimizer_enabled
            .then(|| shell::ShellMinimizer::new(&cwd));
        Self {
            cwd,
            additional_roots,
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
            plugin_hooks,
            lsp,
            shell_minimizer,
            semantic_readiness: tokio::sync::OnceCell::new(),
        }
    }

    /// Whether `name` is served by a connected MCP server (in practice: a
    /// Bifrost tool). Bifrost bounds its own responses and marks elisions
    /// with `----- OMITTED` delimiters the model can act on, so its results
    /// are exempt from the harness's tool-result truncation.
    pub(crate) fn is_mcp_tool(&self, name: &str) -> bool {
        self.mcp_tool_servers.contains_key(name)
    }

    /// Hooks contributed by enabled plugins at registry build time.
    pub(crate) fn plugin_hooks(&self) -> &[crate::plugins::HookCommand] {
        &self.plugin_hooks
    }

    /// All tool definitions for the OpenAI tools parameter.
    ///
    /// `run_shell_command` always advertises the `sandbox_permissions` field
    /// (Codex-style explicit escalation): the model may request a one-time
    /// outside-sandbox run up front, and the permission gate
    /// (`tool_loop::evaluate_pure_gate`) decides whether that request is valid
    /// and prompts the user. The field is no longer hidden behind a prior
    /// sandbox-looking failure.
    pub async fn tool_definitions(&self) -> Vec<ToolDefinition> {
        let builtin_tools = self.active_builtin_tools().await;
        let mut defs = Vec::new();
        if builtin_tools.contains("read_file") {
            let read_description = format!(
                "Reads and returns the content of a specified text file, up to {} bytes. Use after you have selected an exact file/range; for code definitions prefer get_symbol_sources, and for broad code orientation prefer get_summaries.",
                filesystem::READ_MAX_BYTES
            );
            defs.push(tool_def(
                "read_file",
                &read_description,
                json!({
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to read. Relative paths are resolved against the working directory; absolute paths must remain inside it."
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Optional 0-based line number to start reading from."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Optional maximum number of lines to read."
                        }
                    },
                    "required": ["file_path"]
                }),
            ));
        }
        if builtin_tools.contains("write_file") {
            let write_description = format!(
                "Writes content to a specified file in the local filesystem, capped at {} bytes. Use it to create files and to REPLACE a file's entire contents when editing would take more than a few hunks -- one write_file call is better than a chain of edits. Paths may be relative to the working directory or absolute paths inside it.",
                filesystem::WRITE_MAX_BYTES
            );
            defs.push(tool_def(
                "write_file",
                &write_description,
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
        if builtin_tools.contains("delete_file") {
            defs.push(tool_def(
                "delete_file",
                "Deletes one regular file from the local filesystem. Directory and symlink deletion is refused. Relative paths are resolved against the working directory; absolute paths must remain inside a configured workspace root.",
                json!({
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the regular file to delete. Relative paths are resolved against the working directory; absolute paths must remain inside a configured workspace root."
                        }
                    },
                    "required": ["file_path"]
                }),
            ));
        }
        if builtin_tools.contains("move_file") {
            defs.push(tool_def(
                "move_file",
                "Moves or renames one regular file without overwriting an existing destination. Directory and symlink moves are refused. Both paths must remain inside configured workspace roots; missing destination parent directories are created.",
                json!({
                    "type": "object",
                    "properties": {
                        "source_path": {
                            "type": "string",
                            "description": "Path to the existing regular file. Relative paths are resolved against the working directory; absolute paths must remain inside a configured workspace root."
                        },
                        "destination_path": {
                            "type": "string",
                            "description": "New path for the file. The destination must not already exist. Relative paths are resolved against the working directory; absolute paths must remain inside a configured workspace root."
                        }
                    },
                    "required": ["source_path", "destination_path"]
                }),
            ));
        }
        if builtin_tools.contains("edit") {
            defs.push(tool_def(
                "edit",
                "Replaces exact literal text within a file using one or more sequential edit entries. Each entry matches against the file content produced by previous entries, and each `old_string` must be the smallest text that uniquely identifies the change. If `old_string` is ambiguous, expand it with more context or set `replace_all` to true. When no exact match exists, matching falls back to whole-line comparison ignoring leading/trailing whitespace, re-adjusting replacement indentation. Batch related changes to the same file into one call with multiple `edits` entries. If an entry fails, earlier entries remain applied and later entries are not attempted. For heavy rewrites -- many hunks or most of a file changing -- prefer `write_file` with the full new contents instead of a long edit chain.",
                json!({
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to modify. Relative paths are resolved against the working directory; absolute paths must remain inside it."
                        },
                        "edits": {
                            "type": "array",
                            "description": "Sequential exact-string replacements to apply to this file. Later entries see the content produced by earlier entries.",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "old_string": {
                                        "type": "string",
                                        "description": "The exact literal text to replace, including whitespace and indentation. Use the smallest text that uniquely identifies the change."
                                    },
                                    "new_string": {
                                        "type": "string",
                                        "description": "The exact literal text to replace `old_string` with."
                                    },
                                    "replace_all": {
                                        "type": "boolean",
                                        "description": "Replace all occurrences of old_string for this edit entry. Defaults to false."
                                    }
                                },
                                "required": ["old_string", "new_string"]
                            }
                        }
                    },
                    "required": ["file_path", "edits"]
                }),
            ));
        }
        if builtin_tools.contains("list_directory") {
            let list_description = format!(
                "Lists up to {} files and subdirectories directly within a specified directory path. Paths may be relative to the working directory or absolute paths inside it.",
                filesystem::LIST_MAX_ENTRIES
            );
            defs.push(tool_def(
                "list_directory",
                &list_description,
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
                "Searches file contents with a regex. Use for text/config/docs or when symbol tools do not fit; for code declarations prefer search_symbols, and for references/callers prefer scan_usages_by_reference.",
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
                            "type": "integer",
                            "description": "Optional limit on matching lines. Defaults to 50."
                        }
                    },
                    "required": ["pattern"]
                }),
            ));
        }
        if builtin_tools.contains("web_search") {
            defs.push(tool_def(
                "web_search",
                "Searches the public web using DuckDuckGo's no-key HTML search endpoint. No API key, account, or user setup is required. Use for current or external information that is not in the local workspace; returns titles, URLs, and snippets.",
                json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The web search query."
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return. Defaults to 5 and is clamped to 1-10."
                        }
                    },
                    "required": ["query"]
                }),
            ));
        }
        if builtin_tools.contains("diagnostics") {
            defs.push(tool_def(
                "diagnostics",
                "Get cached language-server diagnostics for a file or the project. Opens the requested file in configured LSP servers before reading diagnostics. Requires LSP servers configured in `/setup lsp`.",
                json!({
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Optional file path to check. If omitted, returns cached project diagnostics."
                        }
                    }
                }),
            ));
        }
        if builtin_tools.contains("run_shell_command") {
            // Seconds, not milliseconds: models routinely passed `120`
            // meaning two minutes, and the millisecond reading rounded that
            // to a 1-second budget that killed the command. The unit is
            // stated twice on purpose -- in the field name and in the text.
            let timeout_description = format!(
                "Optional timeout in seconds (not milliseconds) for this command. Clamped to a minimum of {} seconds and a maximum of {} seconds. Defaults to {} seconds when omitted.",
                shell::MIN_TIMEOUT_SECONDS,
                shell::MAX_TIMEOUT_SECONDS,
                shell::DEFAULT_TIMEOUT_SECONDS
            );
            let mut shell_properties = json!({
                "command": {
                    "type": "string",
                    "description": "The shell command to execute (passed to sh -c)."
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": timeout_description
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
            // Keyed off the shared `FIELD`/`SCHEMA_VALUES` constants so the
            // advertised name + enum cannot drift from the deserializer or the
            // gate's matcher (`json!` can't take a path expression as an object
            // key, hence the index assignment).
            shell_properties[ShellSandboxPermissionArg::FIELD] = json!({
                "type": "string",
                "enum": ShellSandboxPermissionArg::SCHEMA_VALUES,
                "description": "Per-command sandbox override. Defaults to `use_default` (run inside the active sandbox). Use `require_escalated` only when the command needs access the sandbox blocks -- network/DNS, package downloads, `git push`, attaching to or debugging host processes, or writing outside the working directory when explicitly requested. In auto permission mode this is decided by the classifier without a user prompt; in other modes it asks the user for one-time approval. Do not escalate for ordinary reads, searches, builds, tests, or workspace writes that should already work inside the sandbox."
            });
            defs.push(tool_def(
                "run_shell_command",
                "Execute a shell command in the working directory. Returns stdout and stderr. Prefer built-in tools for ordinary file reads/search/list/edit/write operations and Bifrost tools for code symbols, definitions, usages, and source orientation. Use shell when CLI semantics matter, such as build, test, git, package-manager, project-specific commands, pipelines, or raw-byte/format inspection. When the session uses sandboxing, commands run in that sandbox by default. Set `sandbox_permissions` to `require_escalated` only when the command genuinely needs access the sandbox blocks -- such as network or DNS access, package downloads, `git push`, attaching to or debugging host processes, or writing outside the working directory when explicitly requested. In auto permission mode this is decided by the classifier without a user prompt; in other modes it asks the user for one-time approval to run outside the sandbox. Do not escalate for ordinary reads, searches, builds, tests, or workspace writes that should already work inside the sandbox.",
                json!({
                    "type": "object",
                    "properties": shell_properties,
                    "required": ["command"]
                }),
            ));
        }
        if builtin_tools.contains("update_plan") {
            defs.push(update_plan_tool_definition());
        }
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
                    mcp_tool_description(&tool.name, &tool.description),
                    tool.input_schema.clone(),
                ));
            }
        }

        // Append `activate_skill` only when at least one skill exists, and
        // constrain `name` to the discovered set via JSON-schema enum.
        // The spec's "Filtering" note: don't expose the tool with an
        // empty enum -- the model would waste turns guessing.
        let skills = self.skills.read().await;
        let names: Vec<String> = skills
            .iter_sorted()
            .filter(|m| m.kind == SkillKind::Skill)
            .map(|m| m.name.clone())
            .collect();
        if !names.is_empty() {
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
                     NOT see this conversation -- give it a self-contained prompt. Read-only task \
                     lanes can run in parallel; use `permission_mode: \"readOnly\"` for review, \
                     exploration, triage, tests/log analysis, and summarization. Use \
                     `permission_mode: \"inherit\"` only when the subagent needs the parent \
                     permission behavior for implementation or fixes.\n\n\
                     Available subagents:\n{catalog}"
                ),
                json!({
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "A short description of the delegated task."
                        },
                        "prompt": {
                            "type": "string",
                            "description": "A complete, self-contained prompt for the subagent."
                        },
                        "subagent_type": {
                            "type": "string",
                            "enum": names,
                            "description": "Exact subagent name from the catalog."
                        },
                        "permission_mode": {
                            "type": "string",
                            "enum": ["readOnly", "inherit"],
                            "description": "Permission behavior for this lane. Defaults to `readOnly`, which is safe for parallel review/exploration lanes and prevents edits or shell execution inside the subagent. Use `inherit` only for implementation/fix lanes that should use the parent session's permission behavior; inherited lanes are not parallelized."
                        }
                    },
                    "required": ["description", "prompt", "subagent_type"]
                }),
            ));
        }

        // Apply the install-wide allowlist only after every source has
        // contributed definitions. In particular, `activate_skill` and `task`
        // are dynamic tools assembled after built-ins and MCP tools.
        if let Some(allowed_tools) = crate::setup_state::read_allowed_tools() {
            let allowed_tools: HashSet<String> = allowed_tools.into_iter().collect();
            defs.retain(|tool| allowed_tools.contains(&tool.function.name));
        }
        defs
    }

    pub(crate) fn is_bifrost_tool(&self, name: &str) -> bool {
        self.mcp_tool_servers
            .get(name)
            .is_some_and(|client| client.name() == "bifrost")
    }

    /// A connected MCP server by name, for harness-internal calls that cannot
    /// go through the tool map: a server may expose a tool it deliberately
    /// keeps out of `tools/list` (bifrost's `semantic_search_status` readiness
    /// probe), and an unadvertised tool has no entry in `mcp_tool_servers`.
    pub(crate) fn mcp_client(&self, name: &str) -> Option<&Arc<McpClient>> {
        self.mcp_clients.iter().find(|client| client.name() == name)
    }

    /// Set once the semantic readiness wait has run for this registry. The
    /// registry lives exactly as long as the session's MCP connections, so this
    /// makes the wait a once-per-session event rather than a per-turn one.
    pub(crate) fn semantic_readiness_once(&self) -> &tokio::sync::OnceCell<()> {
        &self.semantic_readiness
    }

    /// Whether a tool's calls may run concurrently with adjacent safe calls.
    ///
    /// Unknown tools default to `false`; newly-added MCP tools must be
    /// classified explicitly in `TOOLS` before they can use the parallel path.
    pub(crate) fn is_concurrency_safe(&self, name: &str) -> bool {
        tool_meta(name).is_some_and(|meta| meta.concurrency_safe)
    }

    /// Invoke a bifrost MCP tool and return its raw structured `Value`,
    /// bypassing the `ToolResult` string formatting used by the model-facing
    /// dispatch path. This is for harness-internal orchestration (e.g. the
    /// `semantic_search` reranker) that needs to read the structured payload
    /// (`vector_ranked`, `sources`, `summaries`, ...) rather than a
    /// pretty-printed blob. The tool runs without any permission gate, so only
    /// call it for read-only bifrost tools on harness initiative.
    pub(crate) async fn call_bifrost_tool_raw(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {
        let client = self
            .mcp_tool_servers
            .get(name)
            .filter(|client| client.name() == "bifrost")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("bifrost tool '{name}' is not available"))?;
        let args = match client.tools().iter().find(|tool| tool.name == name) {
            Some(tool) => coerce_scalar_args_to_array(args, &tool.input_schema),
            None => args,
        };
        client
            .call_tool(name, args)
            .await
            .map_err(|err| anyhow::anyhow!("bifrost tool '{name}' failed: {err}"))
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
            && !self.mcp_tool_servers.contains_key(name)
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
                let args: ReadFileArgs = match parse_builtin_args(name, args) {
                    Ok(args) => args,
                    Err(result) => return result,
                };
                let cwd = self.cwd.clone();
                let additional_roots = self.additional_roots.clone();
                let path = args.file_path;
                let diagnostics_path = path.clone();
                let offset = args.offset;
                let limit = args.limit;
                let result = run_blocking_filesystem_tool(move || {
                    filesystem::read_file_in_roots(&cwd, &additional_roots, &path, offset, limit)
                })
                .await;
                self.with_read_diagnostics(result, &diagnostics_path).await
            }
            "write_file" => {
                let args: WriteFileArgs = match parse_builtin_args(name, args) {
                    Ok(args) => args,
                    Err(result) => return result,
                };
                let path = args.file_path;
                let diagnostics_path = path.clone();
                let content = args.content;
                if content.len() > filesystem::WRITE_MAX_BYTES {
                    return filesystem::oversized_write_payload_result(&path, content.len());
                }
                let cwd = self.cwd.clone();
                let additional_roots = self.additional_roots.clone();
                let result = run_blocking_filesystem_tool(move || {
                    filesystem::write_file_in_roots(&cwd, &additional_roots, &path, &content)
                })
                .await;
                self.with_write_diagnostics(result, &diagnostics_path).await
            }
            "delete_file" => {
                let args: DeleteFileArgs = match parse_builtin_args(name, args) {
                    Ok(args) => args,
                    Err(result) => return result,
                };
                let cwd = self.cwd.clone();
                let additional_roots = self.additional_roots.clone();
                let path = args.file_path;
                run_blocking_filesystem_tool(move || {
                    filesystem::delete_file_in_roots(&cwd, &additional_roots, &path)
                })
                .await
            }
            "move_file" => {
                let args: MoveFileArgs = match parse_builtin_args(name, args) {
                    Ok(args) => args,
                    Err(result) => return result,
                };
                let cwd = self.cwd.clone();
                let additional_roots = self.additional_roots.clone();
                let source_path = args.source_path;
                let destination_path = args.destination_path;
                run_blocking_filesystem_tool(move || {
                    filesystem::move_file_in_roots(
                        &cwd,
                        &additional_roots,
                        &source_path,
                        &destination_path,
                    )
                })
                .await
            }
            "edit" => {
                let args: EditFileArgs = match parse_builtin_args(name, args) {
                    Ok(args) => args,
                    Err(result) => return result,
                };
                let cwd = self.cwd.clone();
                let additional_roots = self.additional_roots.clone();
                let path = args.file_path;
                let diagnostics_path = path.clone();
                let edits: Vec<filesystem::EditFileEntry> = args
                    .edits
                    .into_iter()
                    .map(|edit| filesystem::EditFileEntry {
                        old_string: edit.old_string,
                        new_string: edit.new_string,
                        replace_all: edit.replace_all,
                    })
                    .collect();
                let result = run_blocking_filesystem_tool(move || {
                    filesystem::edit_file_entries_in_roots(&cwd, &additional_roots, &path, &edits)
                })
                .await;
                self.with_write_diagnostics(result, &diagnostics_path).await
            }
            "list_directory" => {
                let args: ListDirectoryArgs = match parse_builtin_args(name, args) {
                    Ok(args) => args,
                    Err(result) => return result,
                };
                let cwd = self.cwd.clone();
                let additional_roots = self.additional_roots.clone();
                let path = args.path;
                run_blocking_filesystem_tool(move || {
                    filesystem::list_directory_in_roots(&cwd, &additional_roots, &path)
                })
                .await
            }
            "grep_search" => {
                let args: GrepSearchArgs = match parse_builtin_args(name, args) {
                    Ok(args) => args,
                    Err(result) => return result,
                };
                filesystem::search_file_contents_with_sandbox_mode(
                    &self.cwd,
                    &self.additional_roots,
                    &args.pattern,
                    args.glob.as_deref(),
                    args.path.as_deref(),
                    args.limit,
                    sandbox_mode,
                )
            }
            "web_search" => {
                let args: WebSearchArgs = match parse_builtin_args(name, args) {
                    Ok(args) => args,
                    Err(result) => return result,
                };
                web_search::web_search(&args.query, args.max_results).await
            }
            "diagnostics" => {
                let args: DiagnosticsArgs = match parse_builtin_args(name, args) {
                    Ok(args) => args,
                    Err(result) => return result,
                };
                self.execute_diagnostics(args).await
            }
            "run_shell_command" => {
                let args: RunShellCommandArgs = match parse_builtin_args(name, args) {
                    Ok(args) => args,
                    Err(result) => return result,
                };
                let timeout = shell::ShellTimeout::resolve(args.timeout_seconds, args.timeout);
                let command_cwd = match args.directory.as_deref() {
                    Some(directory) if !directory.trim().is_empty() => {
                        match safe_resolve_in_roots(&self.cwd, &self.additional_roots, directory) {
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
                shell::run_shell_command_with_timeout(
                    &command_cwd,
                    &args.command,
                    timeout,
                    policy,
                    outside_sandbox_once,
                    cancel,
                    self.shell_minimizer.as_ref(),
                )
                .await
            }
            "update_plan" => {
                let _: crate::plan::UpdatePlanArgs = match parse_builtin_args(name, args) {
                    Ok(args) => args,
                    Err(result) => return result,
                };
                ToolResult {
                    status: ToolStatus::Success,
                    output: "Plan updated".to_string(),
                }
            }
            "activate_skill" => self.execute_activate_skill(args).await,
            // Any name not handled above is delegated to a configured MCP
            // server. This avoids a hardcoded list of server tool names
            // drifting out of sync with what each server actually exposes.
            _ => self.execute_mcp(name, args, cancel).await,
        }
    }

    async fn with_read_diagnostics(&self, mut result: ToolResult, path: &str) -> ToolResult {
        let Some(lsp) = &self.lsp else {
            return result;
        };
        if result.status != ToolStatus::Success || !lsp.diagnostics_on_read() {
            return result;
        }
        let Ok(resolved) = safe_resolve_in_roots(&self.cwd, &self.additional_roots, path) else {
            return result;
        };
        lsp.open_file(&resolved).await;
        let diagnostics = lsp.diagnostics_for_file(&resolved).await;
        result.output.push_str(&crate::lsp::format_diagnostics(
            Some(&resolved),
            &diagnostics,
        ));
        result
    }

    async fn with_write_diagnostics(&self, mut result: ToolResult, path: &str) -> ToolResult {
        let Some(lsp) = &self.lsp else {
            return result;
        };
        if result.status != ToolStatus::Success || !lsp.diagnostics_on_write() {
            return result;
        }
        let Ok(resolved) = safe_resolve_in_roots(&self.cwd, &self.additional_roots, path) else {
            return result;
        };
        let diagnostics = lsp
            .change_file_and_wait(&resolved, std::time::Duration::from_secs(5))
            .await;
        result.output.push_str(&crate::lsp::format_diagnostics(
            Some(&resolved),
            &diagnostics,
        ));
        result
    }

    async fn execute_diagnostics(&self, args: DiagnosticsArgs) -> ToolResult {
        let Some(lsp) = &self.lsp else {
            return ToolResult {
                status: ToolStatus::RequestError,
                output: "No LSP servers are configured. Use `/setup lsp add <name> <command> [args...]`.".to_string(),
            };
        };
        if !lsp.has_clients() {
            return ToolResult {
                status: ToolStatus::RequestError,
                output: "No LSP clients are running. Check `/setup lsp`.".to_string(),
            };
        }
        let (file_path, diagnostics) = if let Some(path) =
            args.file_path.as_deref().filter(|p| !p.trim().is_empty())
        {
            let resolved = match safe_resolve_in_roots(&self.cwd, &self.additional_roots, path) {
                Ok(path) => path,
                Err(e) => {
                    return ToolResult {
                        status: ToolStatus::RequestError,
                        output: e,
                    };
                }
            };
            let diagnostics = lsp
                .open_file_and_wait(&resolved, std::time::Duration::from_secs(2))
                .await;
            (Some(resolved), diagnostics)
        } else {
            (None, lsp.all_diagnostics().await)
        };
        let output = crate::lsp::format_diagnostics(file_path.as_deref(), &diagnostics);
        ToolResult {
            status: ToolStatus::Success,
            output: if output.is_empty() {
                "No LSP diagnostics.".to_string()
            } else {
                output
            },
        }
    }

    /// Dispatch `activate_skill`. Looks up the requested name against
    /// the cached `SkillRegistry`; the schema's `enum` constraint should
    /// keep this from being called with an unknown name, but treat that
    /// case as a request error rather than an internal error so the
    /// model gets a clear correction.
    async fn execute_activate_skill(&self, args: serde_json::Value) -> ToolResult {
        let args: ActivateSkillArgs = match parse_builtin_args("activate_skill", args) {
            Ok(args) => args,
            Err(result) => return result,
        };
        let name = args.name;
        let skills = self.skills.read().await.clone();
        let Some(meta) = skills.get(&name).filter(|m| m.kind == SkillKind::Skill) else {
            let available: Vec<&str> = skills
                .iter_sorted()
                .filter(|m| m.kind == SkillKind::Skill)
                .map(|m| m.name.as_str())
                .collect();
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
            output: crate::acp::build_skill_payload(meta),
        }
    }

    async fn execute_mcp(
        &self,
        name: &str,
        args: serde_json::Value,
        cancel: Option<&CancellationToken>,
    ) -> ToolResult {
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
        match client.call_tool_cancellable(name, args, cancel).await {
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

async fn run_blocking_filesystem_tool(
    f: impl FnOnce() -> ToolResult + Send + 'static,
) -> ToolResult {
    match tokio::task::spawn_blocking(f).await {
        Ok(result) => result,
        Err(error) => ToolResult {
            status: ToolStatus::InternalError,
            output: format!("Filesystem tool task failed: {error}"),
        },
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

pub(crate) fn update_plan_tool_definition() -> ToolDefinition {
    tool_def(
        "update_plan",
        "Updates the task plan. Provide an optional explanation and a list of plan items, each with a step and status. At most one step can be in_progress at a time. Plan updates are bookkeeping: include this call in the same response as your next tool call rather than spending a response on it alone.",
        json!({
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "Optional explanation for this plan update."
                },
                "plan": {
                    "type": "array",
                    "description": "The list of steps.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": { "type": "string", "description": "Task step text." },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "Step status."
                            }
                        },
                        "required": ["step", "status"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["plan"],
            "additionalProperties": false
        }),
    )
}

/// Resolve a relative path against cwd and ensure it stays within cwd.
#[cfg(test)]
pub fn safe_resolve(cwd: &Path, requested: &str) -> Result<PathBuf, String> {
    safe_resolve_in_roots(cwd, &[], requested)
}

/// Resolve a path against cwd and ensure it stays within cwd or one of the
/// ordered ACP additional workspace roots. Relative paths are intentionally
/// resolved against cwd only; callers can address additional roots with
/// absolute paths.
pub fn safe_resolve_in_roots(
    cwd: &Path,
    additional_roots: &[PathBuf],
    requested: &str,
) -> Result<PathBuf, String> {
    let requested_path = Path::new(requested);
    let joined = if requested_path.is_absolute() {
        requested_path.to_path_buf()
    } else {
        cwd.join(requested_path)
    };
    let resolved = joined
        .canonicalize()
        .map_err(|e| format!("Cannot resolve path '{}': {}", requested, e))?;
    let roots = canonical_workspace_roots(cwd, additional_roots)?;
    if !roots.iter().any(|root| resolved.starts_with(root)) {
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
#[cfg(test)]
pub fn safe_resolve_for_write(cwd: &Path, requested: &str) -> Result<PathBuf, String> {
    safe_resolve_for_write_in_roots(cwd, &[], requested)
}

pub fn safe_resolve_for_write_in_roots(
    cwd: &Path,
    additional_roots: &[PathBuf],
    requested: &str,
) -> Result<PathBuf, String> {
    let roots = canonical_workspace_roots(cwd, additional_roots)?;
    let requested_path = Path::new(requested);
    let joined = if requested_path.is_absolute() {
        requested_path.to_path_buf()
    } else {
        cwd.join(requested_path)
    };

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
    if !roots
        .iter()
        .any(|root| existing_canonical.starts_with(root))
    {
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

fn canonical_workspace_roots(
    cwd: &Path,
    additional_roots: &[PathBuf],
) -> Result<Vec<PathBuf>, String> {
    let mut roots = Vec::with_capacity(1 + additional_roots.len());
    roots.push(
        cwd.canonicalize()
            .map_err(|e| format!("Cannot resolve cwd: {}", e))?,
    );
    for root in additional_roots {
        roots.push(root.canonicalize().map_err(|e| {
            format!(
                "Cannot resolve additional workspace root '{}': {}",
                root.display(),
                e
            )
        })?);
    }
    Ok(roots)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcp_instruction_block_is_deterministic_and_omits_empty_entries() {
        let rendered = render_mcp_instructions(vec![
            ("zeta".to_string(), "Last policy".to_string()),
            ("empty".to_string(), " \n".to_string()),
            ("alpha".to_string(), "First policy".to_string()),
        ]);
        assert_eq!(
            rendered.as_deref(),
            Some(
                "<mcp_instructions>\n  <server name=\"alpha\">\nFirst policy\n  </server>\n  <server name=\"zeta\">\nLast policy\n  </server>\n</mcp_instructions>"
            )
        );
        assert_eq!(render_mcp_instructions(Vec::new()), None);
    }

    /// Allocate a fresh empty directory under the system temp dir for one test
    /// to scribble in. Caller is responsible for cleaning it up.
    fn fresh_tmp_dir(label: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("brokk-acp-rust-{}-{}", label, uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("create tmp dir");
        dir
    }

    /// Existing files inside cwd should resolve through the compatibility
    /// wrapper used by unit tests for the single-root path.
    #[test]
    fn safe_resolve_allows_existing_file_inside_cwd() {
        let cwd = fresh_tmp_dir("resolve-existing");
        std::fs::write(cwd.join("note.txt"), "ok").expect("seed file");

        let resolved = safe_resolve(&cwd, "note.txt").expect("resolve must succeed");
        assert_eq!(resolved, cwd.join("note.txt").canonicalize().unwrap());

        std::fs::remove_dir_all(&cwd).ok();
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
    use crate::skills::{SkillKind, SkillMeta, SkillScope};

    fn registry_with_skills(skills: Vec<SkillMeta>) -> ToolRegistry {
        let mut reg = SkillRegistry::default();
        for meta in skills {
            reg.insert_for_test(meta);
        }
        let cwd = std::env::temp_dir();
        ToolRegistry {
            cwd,
            additional_roots: Vec::new(),
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
            plugin_hooks: Vec::new(),
            lsp: None,
            shell_minimizer: None,
            semantic_readiness: tokio::sync::OnceCell::new(),
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
            additional_roots: Vec::new(),
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
            plugin_hooks: Vec::new(),
            lsp: None,
            shell_minimizer: None,
            semantic_readiness: tokio::sync::OnceCell::new(),
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
            kind: SkillKind::Skill,
        };
        (tmp, meta)
    }

    fn write_command_fixture(name: &str, body: &str) -> (tempfile::TempDir, SkillMeta) {
        let tmp = tempfile::TempDir::new().unwrap();
        let skill_dir = tmp.path().join(name);
        std::fs::create_dir_all(&skill_dir).unwrap();
        let location = skill_dir.join(format!("{name}.md"));
        std::fs::write(&location, body).unwrap();
        let meta = SkillMeta {
            name: name.to_string(),
            description: "cmd".to_string(),
            location,
            skill_dir,
            scope: SkillScope::Plugin,
            kind: SkillKind::Command,
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
    async fn activate_skill_ignores_plugin_commands() {
        let (_skill_tmp, skill) = write_skill_fixture("real-skill", "body");
        let (_command_tmp, command) = write_command_fixture("deploy", "run deploy");
        let registry = registry_with_skills(vec![skill, command]);
        let defs = registry.tool_definitions().await;
        let activate = defs
            .iter()
            .find(|d| d.function.name == "activate_skill")
            .expect("activate_skill must be advertised for the real skill");
        let names: Vec<&str> = activate
            .function
            .parameters
            .pointer("/properties/name/enum")
            .expect("name property has an enum constraint")
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert_eq!(names, vec!["real-skill"]);

        let result = registry
            .execute(
                "activate_skill",
                json!({ "name": "deploy" }),
                SandboxPolicy::WorkspaceWrite,
            )
            .await;
        assert!(matches!(result.status, ToolStatus::RequestError));
        assert!(result.output.contains("Unknown skill 'deploy'"));
        assert!(result.output.contains("real-skill"));
        assert!(!result.output.contains("deploy,"));
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

    #[test]
    fn semantic_search_description_is_overridden() {
        let overridden = mcp_tool_description("semantic_search", "bifrost's raw description");
        assert!(overridden.contains("relevance-ordered"));
        assert_ne!(overridden, "bifrost's raw description");
        // Bifrost has no lexical arm since c353c862. Do not tell the model it
        // does.
        assert!(
            !overridden.to_lowercase().contains("lexical"),
            "the description must not advertise a retrieval leg bifrost removed: {overridden}"
        );
        // Other tools keep bifrost's description unchanged.
        assert_eq!(
            mcp_tool_description("search_symbols", "bifrost's raw description"),
            "bifrost's raw description"
        );
    }

    #[test]
    fn split_and_legacy_scan_usage_names_are_permission_classified() {
        assert_eq!(
            ToolRegistry::tool_kind("scan_usages_by_reference"),
            ToolKind::Search
        );
        assert_eq!(ToolRegistry::tool_kind("scan_usages"), ToolKind::Search);
    }

    #[test]
    fn first_class_file_mutations_have_acp_delete_and_move_kinds() {
        assert_eq!(ToolRegistry::tool_kind("delete_file"), ToolKind::Delete);
        assert_eq!(ToolRegistry::tool_kind("move_file"), ToolKind::Move);
    }

    #[tokio::test]
    async fn builtin_tools_have_metadata_and_are_advertised() {
        let registry = ToolRegistry {
            cwd: std::env::temp_dir(),
            additional_roots: Vec::new(),
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
            plugin_hooks: Vec::new(),
            lsp: None,
            shell_minimizer: None,
            semantic_readiness: tokio::sync::OnceCell::new(),
        };
        let advertised: Vec<String> = registry
            .tool_definitions()
            .await
            .into_iter()
            .map(|d| d.function.name)
            .collect();

        assert_eq!(ToolRegistry::tool_kind("diagnostics"), ToolKind::Read);
        let diagnostics = registry
            .tool_definitions()
            .await
            .into_iter()
            .find(|def| def.function.name == "diagnostics")
            .expect("diagnostics tool advertised");
        assert_eq!(
            diagnostics
                .function
                .parameters
                .pointer("/properties/file_path/type")
                .and_then(serde_json::Value::as_str),
            Some("string")
        );

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

    fn schema_required_fields(defs: &[ToolDefinition], name: &str) -> Vec<String> {
        defs.iter()
            .find(|def| def.function.name == name)
            .unwrap_or_else(|| panic!("{name} should be advertised"))
            .function
            .parameters
            .get("required")
            .and_then(serde_json::Value::as_array)
            .unwrap_or_else(|| panic!("{name} should declare required fields"))
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .expect("required field names are strings")
                    .to_string()
            })
            .collect()
    }

    fn assert_schema_required_matches<T: BuiltinArgsContract>(defs: &[ToolDefinition], name: &str) {
        assert_eq!(
            schema_required_fields(defs, name),
            T::REQUIRED_FIELDS,
            "{name} schema required fields drifted from typed args contract"
        );
    }

    fn assert_schema_property_types_match<T: BuiltinArgsContract>(
        defs: &[ToolDefinition],
        name: &str,
    ) {
        let def = defs
            .iter()
            .find(|def| def.function.name == name)
            .unwrap_or_else(|| panic!("{name} should be advertised"));
        for (property, expected_type) in T::PROPERTY_TYPES {
            let actual_type = def.function.parameters["properties"][*property]["type"]
                .as_str()
                .unwrap_or_else(|| panic!("{name}.{property} should declare a JSON schema type"));
            assert_eq!(
                actual_type, *expected_type,
                "{name}.{property} schema type drifted from typed args contract"
            );
        }
    }

    fn assert_schema_enum_values_match<T: BuiltinArgsContract>(
        defs: &[ToolDefinition],
        name: &str,
    ) {
        let def = defs
            .iter()
            .find(|def| def.function.name == name)
            .unwrap_or_else(|| panic!("{name} should be advertised"));
        for (property, expected_values) in T::ENUM_VALUES {
            let actual = &def.function.parameters["properties"][*property]["enum"];
            assert_eq!(
                actual,
                &json!(expected_values),
                "{name}.{property} schema enum drifted from typed args contract"
            );
        }
    }

    fn assert_builtin_schema_matches<T: BuiltinArgsContract>(defs: &[ToolDefinition], name: &str) {
        assert_schema_required_matches::<T>(defs, name);
        assert_schema_property_types_match::<T>(defs, name);
        assert_schema_enum_values_match::<T>(defs, name);
    }

    #[tokio::test]
    async fn builtin_tool_schemas_match_typed_arg_contracts() {
        let (_t, meta) = write_skill_fixture("hello", "body");
        let registry = registry_with_skills(vec![meta]);
        let defs = registry.tool_definitions().await;

        assert_builtin_schema_matches::<ReadFileArgs>(&defs, "read_file");
        assert_builtin_schema_matches::<WriteFileArgs>(&defs, "write_file");
        assert_builtin_schema_matches::<DeleteFileArgs>(&defs, "delete_file");
        assert_builtin_schema_matches::<MoveFileArgs>(&defs, "move_file");
        assert_builtin_schema_matches::<EditFileArgs>(&defs, "edit");
        assert_builtin_schema_matches::<ListDirectoryArgs>(&defs, "list_directory");
        assert_builtin_schema_matches::<GrepSearchArgs>(&defs, "grep_search");
        assert_builtin_schema_matches::<WebSearchArgs>(&defs, "web_search");
        assert_builtin_schema_matches::<RunShellCommandArgs>(&defs, "run_shell_command");
        assert_builtin_schema_matches::<ActivateSkillArgs>(&defs, "activate_skill");
    }

    /// The model is offered exactly one timeout field, in seconds. The legacy
    /// millisecond `timeout` stays deserializable for replay and in-process
    /// callers but must never be advertised again: seeing both would just
    /// reintroduce the unit confusion this switch exists to remove.
    #[tokio::test]
    async fn shell_schema_advertises_timeout_seconds_only() {
        let registry = registry_with_skills(vec![]);
        let defs = registry.tool_definitions().await;
        let shell_def = defs
            .iter()
            .find(|def| def.function.name == "run_shell_command")
            .expect("run_shell_command should be advertised");
        let properties = &shell_def.function.parameters["properties"];

        assert!(
            properties["timeout"].is_null(),
            "the millisecond timeout field must not be advertised: {properties}"
        );
        assert_eq!(properties["timeout_seconds"]["type"], "integer");
        let description = properties["timeout_seconds"]["description"]
            .as_str()
            .expect("timeout_seconds should carry a description");
        assert_eq!(
            description,
            "Optional timeout in seconds (not milliseconds) for this command. \
             Clamped to a minimum of 10 seconds and a maximum of 3600 seconds. \
             Defaults to 120 seconds when omitted."
        );
        // The advertised numbers are the constants the resolver enforces.
        assert!(description.contains(&format!(
            "minimum of {} seconds",
            shell::MIN_TIMEOUT_SECONDS
        )));
        assert!(description.contains(&format!(
            "maximum of {} seconds",
            shell::MAX_TIMEOUT_SECONDS
        )));
        assert!(description.contains(&format!(
            "Defaults to {} seconds",
            shell::DEFAULT_TIMEOUT_SECONDS
        )));
    }

    #[tokio::test]
    async fn edit_tool_schema_surfaces_batch_entries() {
        let registry = registry_with_skills(vec![]);
        let defs = registry.tool_definitions().await;
        let edit = defs
            .iter()
            .find(|def| def.function.name == "edit")
            .expect("edit should be advertised");

        assert!(edit.function.description.contains(
            "When no exact match exists, matching falls back to whole-line comparison ignoring leading/trailing whitespace, re-adjusting replacement indentation."
        ));
        assert_eq!(
            edit.function.parameters["required"],
            json!(["file_path", "edits"])
        );
        assert!(
            edit.function.parameters["properties"]["old_string"].is_null(),
            "flat old_string must not be advertised at top level"
        );
        let edits = &edit.function.parameters["properties"]["edits"];
        assert_eq!(edits["type"], "array");
        assert_eq!(edits["minItems"], 1);
        assert_eq!(
            edits["items"]["required"],
            json!(["old_string", "new_string"])
        );
        assert_eq!(
            edits["items"]["properties"]["replace_all"]["type"],
            "boolean"
        );
    }

    async fn assert_invalid_builtin_args(
        registry: &ToolRegistry,
        name: &str,
        args: serde_json::Value,
        expected: &str,
    ) {
        let result = registry
            .execute(name, args, SandboxPolicy::WorkspaceWrite)
            .await;
        assert!(
            matches!(result.status, ToolStatus::RequestError),
            "{name} should reject invalid args, got: {}",
            result.output
        );
        assert!(
            result.output.contains("Invalid arguments"),
            "{name} should identify argument validation, got: {}",
            result.output
        );
        assert!(
            result.output.contains(expected),
            "{name} should mention {expected:?}, got: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn builtin_tools_reject_missing_or_wrong_typed_args_before_execution() {
        let (_t, meta) = write_skill_fixture("hello", "body");
        let registry = registry_with_skills(vec![meta]);

        assert_invalid_builtin_args(&registry, "read_file", json!({}), "file_path").await;
        assert_invalid_builtin_args(
            &registry,
            "read_file",
            json!({ "file_path": "x", "offset": null }),
            "offset",
        )
        .await;
        assert_invalid_builtin_args(
            &registry,
            "read_file",
            json!({ "file_path": "x", "limit": 1.5 }),
            "limit",
        )
        .await;
        assert_invalid_builtin_args(
            &registry,
            "write_file",
            json!({ "file_path": "x", "content": 123 }),
            "content",
        )
        .await;
        assert_invalid_builtin_args(&registry, "delete_file", json!({}), "file_path").await;
        assert_invalid_builtin_args(
            &registry,
            "move_file",
            json!({ "source_path": "x" }),
            "destination_path",
        )
        .await;
        assert_invalid_builtin_args(
            &registry,
            "edit",
            json!({
                "file_path": "x",
                "edits": [{
                    "old_string": "a",
                    "new_string": "b",
                    "replace_all": "yes"
                }]
            }),
            "replace_all",
        )
        .await;
        assert_invalid_builtin_args(&registry, "list_directory", json!({ "path": 7 }), "path")
            .await;
        assert_invalid_builtin_args(&registry, "grep_search", json!({}), "pattern").await;
        assert_invalid_builtin_args(
            &registry,
            "grep_search",
            json!({ "pattern": "x", "path": null }),
            "path",
        )
        .await;
        assert_invalid_builtin_args(&registry, "web_search", json!({}), "query").await;
        assert_invalid_builtin_args(
            &registry,
            "web_search",
            json!({ "query": "rust", "max_results": "five" }),
            "max_results",
        )
        .await;
        assert_invalid_builtin_args(
            &registry,
            "run_shell_command",
            json!({ "timeout": 1000 }),
            "command",
        )
        .await;
        assert_invalid_builtin_args(
            &registry,
            "run_shell_command",
            json!({ "command": "echo ok", "timeout_seconds": 30.5 }),
            "timeout_seconds",
        )
        .await;
        // Legacy millisecond field: unadvertised, still validated.
        assert_invalid_builtin_args(
            &registry,
            "run_shell_command",
            json!({ "command": "echo ok", "timeout": 1000.5 }),
            "timeout",
        )
        .await;
        assert_invalid_builtin_args(
            &registry,
            "run_shell_command",
            json!({ "command": "echo ok", "directory": null }),
            "directory",
        )
        .await;
        assert_invalid_builtin_args(
            &registry,
            "activate_skill",
            json!({ "name": ["hello"] }),
            "name",
        )
        .await;
    }

    #[tokio::test]
    async fn edit_tool_accepts_legacy_flat_args_as_one_entry_batch() {
        let cwd = fresh_tmp_dir("edit-flat-compat");
        std::fs::write(cwd.join("a.txt"), "alpha\nbeta\n").unwrap();
        let mut registry = registry_with_skills(vec![]);
        registry.cwd = cwd.clone();

        let result = registry
            .execute(
                "edit",
                json!({
                    "file_path": "a.txt",
                    "old_string": "beta",
                    "new_string": "BETA"
                }),
                SandboxPolicy::WorkspaceWrite,
            )
            .await;

        assert!(
            matches!(result.status, ToolStatus::Success),
            "{}",
            result.output
        );
        assert_eq!(result.output, "Edited 'a.txt' (1 replacement)");
        assert_eq!(
            std::fs::read_to_string(cwd.join("a.txt")).unwrap(),
            "alpha\nBETA\n"
        );
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[tokio::test]
    async fn first_class_move_and_delete_dispatch_through_registry() {
        let cwd = fresh_tmp_dir("move-delete-dispatch");
        std::fs::write(cwd.join("source.txt"), "content").unwrap();
        let mut registry = registry_with_skills(vec![]);
        registry.cwd = cwd.clone();

        let moved = registry
            .execute(
                "move_file",
                json!({
                    "source_path": "source.txt",
                    "destination_path": "nested/destination.txt"
                }),
                SandboxPolicy::WorkspaceWrite,
            )
            .await;
        assert!(
            matches!(moved.status, ToolStatus::Success),
            "{}",
            moved.output
        );
        assert!(cwd.join("nested/destination.txt").exists());

        let deleted = registry
            .execute(
                "delete_file",
                json!({ "file_path": "nested/destination.txt" }),
                SandboxPolicy::WorkspaceWrite,
            )
            .await;
        assert!(
            matches!(deleted.status, ToolStatus::Success),
            "{}",
            deleted.output
        );
        assert!(!cwd.join("nested/destination.txt").exists());
        std::fs::remove_dir_all(&cwd).ok();
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
    async fn shell_tool_schema_always_exposes_sandbox_escalation() {
        let registry = registry_with_skills(vec![]);
        let defs = registry.tool_definitions().await;
        let shell = defs
            .iter()
            .find(|def| def.function.name == "run_shell_command")
            .expect("run_shell_command should be advertised");

        let field = ShellSandboxPermissionArg::FIELD;
        assert!(
            shell
                .function
                .parameters
                .pointer(&format!("/properties/{field}"))
                .is_some(),
            "shell schema must expose sandbox escalation up front (Codex-style)"
        );
        assert_eq!(
            shell.function.parameters["properties"][field]["type"],
            "string"
        );
        assert_eq!(
            shell.function.parameters["properties"][field]["enum"],
            json!(ShellSandboxPermissionArg::SCHEMA_VALUES),
            "shell schema enum must match ShellSandboxPermissionArg"
        );
    }

    #[test]
    fn run_shell_command_args_accept_every_schema_value() {
        // Every advertised enum value must deserialize into a variant; iterating
        // the same `SCHEMA_VALUES` the schema advertises keeps the enum, the
        // deserializer, and the gate's matcher in lockstep (the gate keys off
        // the raw `require_escalated` string, but `use_default` must also parse).
        for value in ShellSandboxPermissionArg::SCHEMA_VALUES {
            parse_builtin_args::<RunShellCommandArgs>(
                "run_shell_command",
                json!({ "command": "echo ok", "sandbox_permissions": value }),
            )
            .unwrap_or_else(|err| {
                panic!("'{value}' should deserialize: {:?}", err.output);
            });
        }
    }

    #[test]
    fn run_shell_command_args_reject_unknown_sandbox_permission_value() {
        let err = parse_builtin_args::<RunShellCommandArgs>(
            "run_shell_command",
            json!({ "command": "echo ok", "sandbox_permissions": "yolo" }),
        )
        .expect_err("an unknown sandbox_permissions value must be rejected");
        assert!(matches!(err.status, ToolStatus::RequestError));
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
        assert!(!advertised.iter().any(|name| name == "web_search"));
        assert!(!advertised.iter().any(|name| name == "run_shell_command"));
    }

    #[tokio::test]
    async fn setup_allowed_tools_filters_builtins_and_dynamic_tools() {
        use crate::agents::{AgentMeta, AgentScope};

        let config_dir = tempfile::tempdir().expect("config dir");
        let _scope = crate::setup_state::TestConfigHomeScope::set(config_dir.path().to_path_buf());
        std::fs::write(
            config_dir.path().join("setup.json"),
            serde_json::json!({ "allowed_tools": ["read_file", "task"] }).to_string(),
        )
        .expect("write setup state");

        let registry = registry_with_agents(vec![AgentMeta {
            name: "reviewer".into(),
            description: "Reviews code".into(),
            max_turns: None,
            allowed_tools: None,
            location: PathBuf::from("/tmp/reviewer.md"),
            scope: AgentScope::Project,
            bundled_body: None,
        }]);
        let advertised: Vec<String> = registry
            .tool_definitions()
            .await
            .into_iter()
            .map(|definition| definition.function.name)
            .collect();

        assert_eq!(advertised, vec!["read_file", "task"]);

        std::fs::write(
            config_dir.path().join("setup.json"),
            serde_json::json!({ "allowed_tools": [] }).to_string(),
        )
        .expect("write empty allowed_tools");
        assert!(registry.tool_definitions().await.is_empty());
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
            result.output.contains("terminated the child process tree"),
            "cancelled shell command should mention child-tree termination; output={}",
            result.output
        );
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "cancelled shell command waited too long"
        );
    }

    /// End-to-end wiring of the ceiling. The exact clamped value is pinned by
    /// `shell::timeout_tests`, which serializes against the deployment-cap
    /// override; asserting the number here would race that test.
    #[tokio::test]
    async fn shell_timeout_seconds_above_the_cap_is_clamped_and_reported() {
        let registry = registry_with_skills(vec![]);
        let result = registry
            .execute_with_sandbox_mode_cancellable(
                "run_shell_command",
                json!({ "command": "echo ok", "timeout_seconds": 4000 }),
                SandboxPolicy::None,
                false,
                None,
                None,
            )
            .await;

        assert!(
            matches!(result.status, ToolStatus::Success),
            "clamped timeout command should still run; output={}",
            result.output
        );
        assert!(
            result
                .output
                .contains("exceeded the server maximum; clamped to"),
            "clamped timeout should be reported; output={}",
            result.output
        );
        assert!(
            result.output.contains("ok"),
            "command output should be preserved; output={}",
            result.output
        );
    }

    /// The trace that motivated the seconds switch: a model passed a small
    /// number meaning seconds, got a sub-second budget, and gave up on
    /// verifying its work. Now the floor rescues it and says so.
    #[tokio::test]
    async fn shell_timeout_seconds_below_the_floor_is_raised_and_reported() {
        let registry = registry_with_skills(vec![]);
        let result = registry
            .execute_with_sandbox_mode_cancellable(
                "run_shell_command",
                json!({ "command": "echo ok", "timeout_seconds": 3 }),
                SandboxPolicy::None,
                false,
                None,
                None,
            )
            .await;

        assert!(
            matches!(result.status, ToolStatus::Success),
            "floored timeout command should still run; output={}",
            result.output
        );
        assert!(
            result.output.contains(&format!(
                "Notice: requested timeout 3s was below the {}s minimum; raised to {}s.",
                shell::MIN_TIMEOUT_SECONDS,
                shell::MIN_TIMEOUT_SECONDS
            )),
            "raised timeout should be reported; output={}",
            result.output
        );
        assert!(
            result.output.contains("ok"),
            "command output should be preserved; output={}",
            result.output
        );
    }

    /// The millisecond field is no longer advertised, but replayed traces and
    /// in-process callers still pass it and must keep working unchanged: 2000ms
    /// is a 2 second budget, not floored to the new 10 second minimum.
    #[tokio::test]
    async fn legacy_millisecond_timeout_is_still_accepted() {
        let registry = registry_with_skills(vec![]);
        let result = registry
            .execute_with_sandbox_mode_cancellable(
                "run_shell_command",
                json!({ "command": "echo ok", "timeout": 2_000 }),
                SandboxPolicy::None,
                false,
                None,
                None,
            )
            .await;

        assert!(
            matches!(result.status, ToolStatus::Success),
            "legacy millisecond timeout should still run; output={}",
            result.output
        );
        assert!(
            !result.output.contains("Notice: requested timeout"),
            "a legacy millisecond value inside the range should not be clamped; output={}",
            result.output
        );
        assert!(
            result.output.contains("ok"),
            "command output should be preserved; output={}",
            result.output
        );
    }

    /// Both fields present: the advertised seconds field wins. A legacy
    /// 600000ms would be a silent 600s budget, so the floor notice is the
    /// tell that `timeout_seconds` was the one that took effect.
    #[tokio::test]
    async fn timeout_seconds_takes_precedence_over_legacy_milliseconds() {
        let registry = registry_with_skills(vec![]);
        let result = registry
            .execute_with_sandbox_mode_cancellable(
                "run_shell_command",
                json!({ "command": "echo ok", "timeout_seconds": 3, "timeout": 600_000 }),
                SandboxPolicy::None,
                false,
                None,
                None,
            )
            .await;

        assert!(
            matches!(result.status, ToolStatus::Success),
            "command should still run; output={}",
            result.output
        );
        assert!(
            result
                .output
                .contains("Notice: requested timeout 3s was below"),
            "timeout_seconds should win over the legacy millisecond field; output={}",
            result.output
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
                max_turns: Some(7),
                allowed_tools: Some(vec!["grep_search".into(), "read_file".into()]),
                location: PathBuf::from("/tmp/doc-writer.md"),
                scope: AgentScope::Project,
                bundled_body: None,
            },
            AgentMeta {
                name: "bug-hunter".into(),
                description: "Hunts for regressions".into(),
                max_turns: None,
                allowed_tools: None,
                location: PathBuf::from("/tmp/bug-hunter.md"),
                scope: AgentScope::User,
                bundled_body: None,
            },
        ]);
        let defs = registry.tool_definitions().await;
        let task_def = defs
            .iter()
            .find(|d| d.function.name == "task")
            .expect("task tool should be advertised");

        assert_eq!(
            schema_required_fields(&defs, "task"),
            vec![
                "description".to_string(),
                "prompt".to_string(),
                "subagent_type".to_string()
            ],
            "task schema required fields must match TaskArgs"
        );
        for property in ["description", "prompt", "subagent_type", "permission_mode"] {
            assert_eq!(
                task_def.function.parameters["properties"][property]["type"], "string",
                "task.{property} schema type must match TaskArgs"
            );
        }

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

        let permission_enum = task_def.function.parameters["properties"]["permission_mode"]["enum"]
            .as_array()
            .expect("permission_mode should constrain via enum");
        let permission_values: Vec<&str> = permission_enum
            .iter()
            .filter_map(|value| value.as_str())
            .collect();
        assert_eq!(permission_values, vec!["readOnly", "inherit"]);

        // The model-facing catalog contains only selection-relevant metadata.
        // Execution constraints remain enforced from AgentMeta at dispatch,
        // but repeating long tool allowlists here wastes tokens every turn.
        assert!(
            task_def
                .function
                .description
                .contains("- doc-writer: Drafts docs from code"),
            "catalog should mention each subagent; got: {}",
            task_def.function.description
        );
        assert!(
            task_def
                .function
                .description
                .contains("- bug-hunter: Hunts for regressions")
        );
        assert!(!task_def.function.description.contains("max_turns:"));
        assert!(!task_def.function.description.contains("tools:"));
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
