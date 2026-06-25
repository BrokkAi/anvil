use super::{ToolResult, ToolStatus, safe_resolve, safe_resolve_for_write};
use std::path::Path;

/// Hard cap for `read_file` and the existing file read by `edit_file`.
pub(super) const READ_MAX_BYTES: u64 = 1_048_576; // 1 MiB
/// Hard cap for model-provided write payloads and post-edit file contents.
pub(super) const WRITE_MAX_BYTES: usize = 1_048_576; // 1 MiB
/// Hard cap on directory entries returned by `list_directory`.
pub(super) const LIST_MAX_ENTRIES: usize = 1_000;
/// Hard cap on individual file size scanned by `search_file_contents`.
/// Files larger than this are skipped to keep memory bounded on big repos.
const SEARCH_MAX_FILE_BYTES: u64 = 1_048_576; // 1 MiB
/// Total-bytes-scanned budget across the whole walk. Together with
/// the per-file cap this bounds the worst case the sandbox has to
/// chew through for a single `grep_search` call.
const SEARCH_MAX_TOTAL_BYTES: u64 = 256 * 1024 * 1024;

pub fn read_file(
    cwd: &Path,
    path: &str,
    offset: Option<usize>,
    limit: Option<usize>,
) -> ToolResult {
    read_file_with_backend(cwd, path, offset, limit, crate::sandbox_backend::global())
}

fn read_file_with_backend(
    cwd: &Path,
    path: &str,
    offset: Option<usize>,
    limit: Option<usize>,
    backend: &crate::sandbox_backend::SandboxBackend,
) -> ToolResult {
    let resolved = match safe_resolve(cwd, path) {
        Ok(p) => p,
        Err(e) => {
            return ToolResult {
                status: ToolStatus::RequestError,
                output: e,
            };
        }
    };
    match read_bounded_text(backend, &resolved) {
        Ok(Some(content)) => {
            let output = match (offset, limit) {
                (None, None) => content,
                _ => {
                    let start = offset.unwrap_or(0);
                    let iter = content.lines().skip(start);
                    let lines: Vec<&str> = match limit {
                        Some(limit) => iter.take(limit).collect(),
                        None => iter.collect(),
                    };
                    lines.join("\n")
                }
            };
            ToolResult {
                status: ToolStatus::Success,
                output,
            }
        }
        Ok(None) => ToolResult {
            status: ToolStatus::RequestError,
            output: format!("Failed to read '{path}': not a regular file"),
        },
        Err(e) => ToolResult {
            status: ToolStatus::RequestError,
            output: format!("Failed to read '{}': {}", path, e),
        },
    }
}

fn read_bounded_text(
    backend: &crate::sandbox_backend::SandboxBackend,
    resolved: &Path,
) -> std::io::Result<Option<String>> {
    backend.read_file_bounded(resolved, READ_MAX_BYTES)
}

pub fn edit_file(
    cwd: &Path,
    path: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
) -> ToolResult {
    if old_string.is_empty() {
        return ToolResult {
            status: ToolStatus::RequestError,
            output: "`old_string` must not be empty".to_string(),
        };
    }

    let resolved = match safe_resolve_for_write(cwd, path) {
        Ok(p) => p,
        Err(e) => {
            return ToolResult {
                status: ToolStatus::RequestError,
                output: e,
            };
        }
    };
    let content = match read_bounded_text(crate::sandbox_backend::global(), &resolved) {
        Ok(Some(content)) => content,
        Ok(None) => {
            return ToolResult {
                status: ToolStatus::RequestError,
                output: format!("Failed to read '{path}': not a regular file"),
            };
        }
        Err(e) => {
            return ToolResult {
                status: ToolStatus::RequestError,
                output: format!("Failed to read '{}': {}", path, e),
            };
        }
    };

    let matches = content.matches(old_string).count();
    if matches == 0 {
        return ToolResult {
            status: ToolStatus::RequestError,
            output: format!("No occurrences of `old_string` found in '{path}'"),
        };
    }
    if !replace_all && matches > 1 {
        return ToolResult {
            status: ToolStatus::RequestError,
            output: format!(
                "`old_string` occurs {matches} times in '{path}'. Provide more context or set `replace_all` to true."
            ),
        };
    }

    let updated = if replace_all {
        content.replace(old_string, new_string)
    } else {
        content.replacen(old_string, new_string, 1)
    };
    if updated.len() > WRITE_MAX_BYTES {
        return ToolResult {
            status: ToolStatus::RequestError,
            output: format!(
                "Edited content for '{path}' is {} bytes, exceeds cap of {WRITE_MAX_BYTES}",
                updated.len()
            ),
        };
    }
    match atomic_write(&resolved, updated.as_bytes()) {
        Ok(()) => ToolResult {
            status: ToolStatus::Success,
            output: format!(
                "Edited '{path}' ({matches} replacement{})",
                if matches == 1 { "" } else { "s" }
            ),
        },
        Err(e) => ToolResult {
            status: ToolStatus::RequestError,
            output: format!("Failed to edit '{}': {}", path, e),
        },
    }
}

pub fn write_file(cwd: &Path, path: &str, content: &str) -> ToolResult {
    if content.len() > WRITE_MAX_BYTES {
        return oversized_write_payload_result(path, content.len());
    }
    let resolved = match safe_resolve_for_write(cwd, path) {
        Ok(p) => p,
        Err(e) => {
            return ToolResult {
                status: ToolStatus::RequestError,
                output: e,
            };
        }
    };
    // Create parent directories if needed
    if let Some(parent) = resolved.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return ToolResult {
            status: ToolStatus::InternalError,
            output: format!("Failed to create directories for '{}': {}", path, e),
        };
    }
    match atomic_write(&resolved, content.as_bytes()) {
        Ok(()) => ToolResult {
            status: ToolStatus::Success,
            output: format!("Written {} bytes to '{}'", content.len(), path),
        },
        Err(e) => ToolResult {
            status: ToolStatus::RequestError,
            output: format!("Failed to write '{}': {}", path, e),
        },
    }
}

pub(super) fn oversized_write_payload_result(path: &str, len: usize) -> ToolResult {
    ToolResult {
        status: ToolStatus::RequestError,
        output: format!(
            "Write payload for '{path}' is {len} bytes, exceeds cap of {WRITE_MAX_BYTES}"
        ),
    }
}

/// Write `content` to `target` such that the destination either holds the
/// previous contents or the new contents in full -- never a partial mix.
///
/// We write to a sibling tempfile, fsync it, then `rename(2)` over the
/// destination. `rename(2)` on POSIX (and `MoveFileExW` with REPLACE on
/// Windows) is the standard primitive for this. If anything fails before
/// the rename, the temp file is removed via a drop guard and the existing
/// destination is untouched.
///
/// `safe_resolve_for_write` is the caller's responsibility -- this helper
/// trusts `target` to already be inside the cwd.
fn atomic_write(target: &Path, content: &[u8]) -> std::io::Result<()> {
    use std::io::Write;

    let parent = target.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "target path has no parent directory",
        )
    })?;
    let file_name = target.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "target path has no file name",
        )
    })?;

    // Preserve mode of the existing destination so atomic replace doesn't
    // silently drop e.g. an executable bit. `metadata` follows symlinks,
    // which matches what plain `fs::write` would have done before this
    // change; symlink targets already had to land inside cwd to pass
    // `safe_resolve_for_write`.
    let existing_perms = std::fs::metadata(target).ok().map(|m| m.permissions());

    // Hidden + uuid suffix keeps the temp invisible to most tools and
    // collision-free against concurrent writers. Sibling of the target so
    // `rename` stays on the same filesystem (atomic).
    let mut tmp_name = std::ffi::OsString::from(".");
    tmp_name.push(file_name);
    tmp_name.push(format!(".tmp.{}", uuid::Uuid::new_v4()));
    let tmp_path = parent.join(&tmp_name);

    struct TmpGuard<'a> {
        path: &'a Path,
        armed: bool,
    }
    impl<'a> Drop for TmpGuard<'a> {
        fn drop(&mut self) {
            if self.armed {
                let _ = std::fs::remove_file(self.path);
            }
        }
    }
    let mut guard = TmpGuard {
        path: &tmp_path,
        armed: true,
    };

    {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)?;
        f.write_all(content)?;
        // sync_all so a power loss between rename and the next fsync still
        // leaves the file's data on disk, not just its directory entry.
        f.sync_all()?;
    }

    if let Some(perms) = existing_perms {
        std::fs::set_permissions(&tmp_path, perms)?;
    }

    std::fs::rename(&tmp_path, target)?;
    guard.armed = false;
    Ok(())
}

pub fn list_directory(cwd: &Path, path: &str) -> ToolResult {
    let resolved = match safe_resolve(cwd, path) {
        Ok(p) => p,
        Err(e) => {
            return ToolResult {
                status: ToolStatus::RequestError,
                output: e,
            };
        }
    };
    let entries = match std::fs::read_dir(&resolved) {
        Ok(e) => e,
        Err(e) => {
            return ToolResult {
                status: ToolStatus::RequestError,
                output: format!("Failed to list '{}': {}", path, e),
            };
        }
    };

    let mut lines: Vec<String> = Vec::new();
    let mut truncated = false;
    for entry in entries.flatten() {
        if lines.len() >= LIST_MAX_ENTRIES {
            truncated = true;
            break;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        if is_dir {
            lines.push(format!("{}/", name));
        } else {
            lines.push(name);
        }
    }
    lines.sort();
    if truncated {
        lines.push(format!("... truncated at {LIST_MAX_ENTRIES} entries"));
    }
    ToolResult {
        status: ToolStatus::Success,
        output: lines.join("\n"),
    }
}

/// `grep_search` tool, routed through `SandboxBackend` so the
/// user-controlled regex runs inside the wasm sandbox by default.
/// The `regex` crate is engineered to be linear-time, but a future
/// engine bug or accidental enabling of a backtracking feature
/// shouldn't be able to hang the agent -- the wasm fuel cap is the
/// definitive backstop.
pub fn search_file_contents(
    cwd: &Path,
    pattern: &str,
    glob_filter: Option<&str>,
    search_path: Option<&str>,
    max_results: usize,
) -> ToolResult {
    search_file_contents_with_backend(
        cwd,
        pattern,
        glob_filter,
        search_path,
        max_results,
        crate::sandbox_backend::global(),
    )
}

pub fn search_file_contents_with_sandbox_mode(
    cwd: &Path,
    pattern: &str,
    glob_filter: Option<&str>,
    search_path: Option<&str>,
    max_results: usize,
    sandbox_mode: Option<crate::sandbox_backend::SandboxMode>,
) -> ToolResult {
    if sandbox_mode.is_none() {
        return search_file_contents(cwd, pattern, glob_filter, search_path, max_results);
    }
    let backend = match crate::sandbox_backend::backend_for_mode(sandbox_mode) {
        Ok(backend) => backend,
        Err(e) => {
            return ToolResult {
                status: ToolStatus::InternalError,
                output: format!("Failed to initialize sandbox backend: {e}"),
            };
        }
    };
    search_file_contents_with_backend(
        cwd,
        pattern,
        glob_filter,
        search_path,
        max_results,
        &backend,
    )
}

fn search_file_contents_with_backend(
    cwd: &Path,
    pattern: &str,
    glob_filter: Option<&str>,
    search_path: Option<&str>,
    max_results: usize,
    backend: &crate::sandbox_backend::SandboxBackend,
) -> ToolResult {
    let root = match search_path {
        Some(path) if !path.trim().is_empty() => match safe_resolve(cwd, path) {
            Ok(p) => p,
            Err(e) => {
                return ToolResult {
                    status: ToolStatus::RequestError,
                    output: e,
                };
            }
        },
        _ => cwd.to_path_buf(),
    };
    if root.is_file() {
        return search_single_file_with_backend(cwd, &root, pattern, max_results, backend);
    }

    let outcome = match backend.search_file_contents(
        &root,
        pattern,
        glob_filter,
        max_results as u64,
        SEARCH_MAX_FILE_BYTES,
        SEARCH_MAX_TOTAL_BYTES,
    ) {
        Ok(o) => o,
        Err(brokk_acp_sandbox::SearchError::InvalidRegex(msg)) => {
            return ToolResult {
                status: ToolStatus::RequestError,
                output: format!("Invalid regex '{}': {}", pattern, msg),
            };
        }
        Err(brokk_acp_sandbox::SearchError::InvalidGlob(msg)) => {
            return ToolResult {
                status: ToolStatus::RequestError,
                output: format!("Invalid glob: {msg}"),
            };
        }
        Err(brokk_acp_sandbox::SearchError::Walk(msg)) => {
            return ToolResult {
                status: ToolStatus::InternalError,
                output: msg,
            };
        }
    };

    format_search_outcome(pattern, max_results, outcome)
}

fn format_search_outcome(
    pattern: &str,
    max_results: usize,
    outcome: brokk_acp_sandbox::SearchOutcome,
) -> ToolResult {
    if outcome.matches.is_empty() {
        return ToolResult {
            status: ToolStatus::Success,
            output: format!("No matches found for '{}'", pattern),
        };
    }

    let mut lines: Vec<String> = outcome
        .matches
        .iter()
        .map(|m| format!("{}:{}: {}", m.path, m.line_num, m.line))
        .collect();
    if outcome.truncated {
        lines.push(format!("... truncated at {} results", max_results));
    }
    ToolResult {
        status: ToolStatus::Success,
        output: lines.join("\n"),
    }
}

fn search_single_file_with_backend(
    cwd: &Path,
    path: &Path,
    pattern: &str,
    max_results: usize,
    backend: &crate::sandbox_backend::SandboxBackend,
) -> ToolResult {
    match backend {
        crate::sandbox_backend::SandboxBackend::OsNative => {
            search_single_file(cwd, path, pattern, max_results)
        }
        crate::sandbox_backend::SandboxBackend::WasmFallback(_) => {
            search_single_file_in_wasm(cwd, path, pattern, max_results, backend)
        }
    }
}

fn search_single_file_in_wasm(
    cwd: &Path,
    path: &Path,
    pattern: &str,
    max_results: usize,
    backend: &crate::sandbox_backend::SandboxBackend,
) -> ToolResult {
    let Some(parent) = path.parent() else {
        return ToolResult {
            status: ToolStatus::RequestError,
            output: format!(
                "Cannot search '{}': file has no parent directory",
                path.display()
            ),
        };
    };
    let Some(file_name) = path.file_name().and_then(|s| s.to_str()) else {
        return ToolResult {
            status: ToolStatus::RequestError,
            output: format!(
                "Cannot search '{}': file name is not valid UTF-8",
                path.display()
            ),
        };
    };
    let Some(glob) = exact_simple_glob_for_file_name(file_name) else {
        return ToolResult {
            status: ToolStatus::RequestError,
            output: format!(
                "Cannot search '{}' in wasm sandbox: file names containing '*' are unsupported",
                path.display()
            ),
        };
    };

    let mut outcome = match backend.search_file_contents(
        parent,
        pattern,
        Some(&glob),
        max_results as u64,
        SEARCH_MAX_FILE_BYTES,
        SEARCH_MAX_TOTAL_BYTES,
    ) {
        Ok(o) => o,
        Err(brokk_acp_sandbox::SearchError::InvalidRegex(msg)) => {
            return ToolResult {
                status: ToolStatus::RequestError,
                output: format!("Invalid regex '{}': {}", pattern, msg),
            };
        }
        Err(brokk_acp_sandbox::SearchError::InvalidGlob(msg)) => {
            return ToolResult {
                status: ToolStatus::RequestError,
                output: format!("Invalid glob: {msg}"),
            };
        }
        Err(brokk_acp_sandbox::SearchError::Walk(msg)) => {
            return ToolResult {
                status: ToolStatus::InternalError,
                output: msg,
            };
        }
    };

    let display_path = path.strip_prefix(cwd).unwrap_or(path).display().to_string();
    for m in &mut outcome.matches {
        if m.path == file_name {
            m.path = display_path.clone();
        }
    }
    format_search_outcome(pattern, max_results, outcome)
}

fn exact_simple_glob_for_file_name(name: &str) -> Option<String> {
    if name.contains('*') {
        return None;
    }
    let mut out = String::from("^");
    for ch in name.chars() {
        match ch {
            // `compile_glob` in brokk_acp_sandbox escapes `.` and expands
            // `*`, then treats the result as a regex. Escape all other regex
            // metacharacters here so the generated glob remains exact.
            '?' | '+' | '(' | ')' | '|' | '^' | '$' | '[' | ']' | '{' | '}' | '\\' => {
                out.push('\\');
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    Some(out)
}

fn search_single_file(cwd: &Path, path: &Path, pattern: &str, max_results: usize) -> ToolResult {
    let regex = match regex::Regex::new(pattern) {
        Ok(regex) => regex,
        Err(e) => {
            return ToolResult {
                status: ToolStatus::RequestError,
                output: format!("Invalid regex '{}': {}", pattern, e),
            };
        }
    };
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => {
            return ToolResult {
                status: ToolStatus::RequestError,
                output: format!("Failed to read '{}': {}", path.display(), e),
            };
        }
    };
    let display_path = path.strip_prefix(cwd).unwrap_or(path).display().to_string();
    let mut lines = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        if regex.is_match(line) {
            lines.push(format!("{}:{}: {}", display_path, idx + 1, line));
            if lines.len() >= max_results {
                lines.push(format!("... truncated at {} results", max_results));
                break;
            }
        }
    }
    if lines.is_empty() {
        ToolResult {
            status: ToolStatus::Success,
            output: format!("No matches found for '{}'", pattern),
        }
    } else {
        ToolResult {
            status: ToolStatus::Success,
            output: lines.join("\n"),
        }
    }
}

// `is_binary_file` and `BINARY_SNIFF_BYTES` moved to the shared
// `brokk_acp_sandbox::search` module so the native and wasm-sandboxed
// backends classify binary files identically.

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Allocate a fresh empty directory under the system temp dir for one test
    /// to scribble in. Caller is responsible for cleaning it up.
    fn fresh_tmp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "brokk-acp-rust-fs-{}-{}",
            label,
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create tmp dir");
        dir
    }

    /// Round-trip: write then read returns the same content. Verifies both
    /// dispatch paths land on the same on-disk file via the cwd-relative
    /// resolver.
    #[test]
    fn write_then_read_round_trips() {
        let cwd = fresh_tmp_dir("rw");
        let w = write_file(&cwd, "hello.txt", "world");
        assert!(matches!(w.status, ToolStatus::Success));
        assert!(w.output.contains("Written 5 bytes"));

        let r = read_file(&cwd, "hello.txt", None, None);
        assert!(matches!(r.status, ToolStatus::Success));
        assert_eq!(r.output, "world");

        std::fs::remove_dir_all(&cwd).ok();
    }

    /// `write_file` creates intermediate directories that don't exist yet
    /// (mkdir -p semantics) -- otherwise the LLM would have to chain a
    /// `run_shell_command mkdir` for every nested write.
    #[test]
    fn write_file_creates_missing_parent_directories() {
        let cwd = fresh_tmp_dir("mkdir-p");
        let w = write_file(&cwd, "a/b/c/note.md", "ok");
        assert!(matches!(w.status, ToolStatus::Success), "{}", w.output);
        assert!(cwd.join("a/b/c/note.md").exists());
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn read_and_write_accept_absolute_paths_inside_cwd() {
        let cwd = fresh_tmp_dir("abs-rw");
        let path = cwd.join("note.txt");
        let w = write_file(&cwd, path.to_str().unwrap(), "zero\none\ntwo\n");
        assert!(matches!(w.status, ToolStatus::Success), "{}", w.output);

        let r = read_file(&cwd, path.to_str().unwrap(), Some(1), Some(1));
        assert!(matches!(r.status, ToolStatus::Success), "{}", r.output);
        assert_eq!(r.output, "one");
        std::fs::remove_dir_all(&cwd).ok();
    }

    /// Helper: count sibling temp files left behind by `atomic_write`. They
    /// are named `.{filename}.tmp.{uuid}` so we look for the prefix.
    fn count_tmp_siblings(dir: &Path, target_name: &str) -> usize {
        let prefix = format!(".{}.tmp.", target_name);
        std::fs::read_dir(dir)
            .map(|it| {
                it.flatten()
                    .filter(|e| e.file_name().to_string_lossy().starts_with(prefix.as_str()))
                    .count()
            })
            .unwrap_or(0)
    }

    /// Atomic replace: writing over an existing file leaves the new content
    /// and no `.tmp.*` sibling. The on-disk inode rotates via rename(2);
    /// existing readers holding the old fd keep their consistent view.
    #[test]
    fn write_file_atomically_replaces_existing_file() {
        let cwd = fresh_tmp_dir("atomic-replace");
        let first = write_file(&cwd, "doc.txt", "original content");
        assert!(matches!(first.status, ToolStatus::Success));

        let second = write_file(&cwd, "doc.txt", "replaced content");
        assert!(matches!(second.status, ToolStatus::Success));

        assert_eq!(
            std::fs::read_to_string(cwd.join("doc.txt")).unwrap(),
            "replaced content"
        );
        assert_eq!(
            count_tmp_siblings(&cwd, "doc.txt"),
            0,
            "no tempfile should be left behind after a successful write"
        );
        std::fs::remove_dir_all(&cwd).ok();
    }

    /// Atomic create: a fresh write produces the target file and no stray
    /// `.tmp.*` sibling. Guards against the temp file leaking when the
    /// destination did not previously exist.
    #[test]
    fn write_file_atomically_creates_new_file() {
        let cwd = fresh_tmp_dir("atomic-new");
        let w = write_file(&cwd, "new.txt", "hello");
        assert!(matches!(w.status, ToolStatus::Success));
        assert_eq!(
            std::fs::read_to_string(cwd.join("new.txt")).unwrap(),
            "hello"
        );
        assert_eq!(count_tmp_siblings(&cwd, "new.txt"), 0);
        std::fs::remove_dir_all(&cwd).ok();
    }

    /// Failure simulation: pointing `atomic_write` at a path that is an
    /// existing directory makes `rename(2)` fail (a regular file cannot
    /// replace a directory on any supported platform). The contract is:
    /// the original entry is untouched and the temp file is cleaned up by
    /// the drop guard.
    #[test]
    fn atomic_write_failure_before_rename_preserves_destination_and_cleans_up() {
        let cwd = fresh_tmp_dir("atomic-fail");
        let target = cwd.join("entry");
        std::fs::create_dir(&target).unwrap();
        let canary = target.join("inside.txt");
        std::fs::write(&canary, "untouched").unwrap();

        let err = atomic_write(&target, b"new bytes").expect_err("rename over a dir must fail");
        // Be permissive about the exact io::ErrorKind; behavior varies
        // across libc/Windows. The structural guarantee is what we test.
        assert!(
            !err.to_string().is_empty(),
            "error should carry a message: {:?}",
            err
        );

        assert!(target.is_dir(), "destination directory must survive");
        assert_eq!(
            std::fs::read_to_string(&canary).unwrap(),
            "untouched",
            "directory contents must survive"
        );
        assert_eq!(
            count_tmp_siblings(&cwd, "entry"),
            0,
            "tempfile must be removed by the drop guard on failure"
        );
        std::fs::remove_dir_all(&cwd).ok();
    }

    /// On Unix the destination's file mode survives an atomic replace so a
    /// previously-chmod'd file (e.g. a script with the exec bit set)
    /// doesn't silently drop permissions when the agent rewrites it.
    #[cfg(unix)]
    #[test]
    fn write_file_preserves_existing_unix_mode_on_replace() {
        use std::os::unix::fs::PermissionsExt;
        let cwd = fresh_tmp_dir("preserve-mode");
        let path = cwd.join("script.sh");
        std::fs::write(&path, "#!/bin/sh\necho old\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o750)).unwrap();

        let w = write_file(&cwd, "script.sh", "#!/bin/sh\necho new\n");
        assert!(matches!(w.status, ToolStatus::Success));

        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o750, "exec/group bits must survive the replace");
        std::fs::remove_dir_all(&cwd).ok();
    }

    /// `read_file` for a missing path is a `RequestError` (the LLM can
    /// recover by listing the directory), not an `InternalError`. The
    /// rejection comes from `safe_resolve`'s canonicalize step, which
    /// requires the target to exist.
    #[test]
    fn read_file_missing_path_is_request_error() {
        let cwd = fresh_tmp_dir("missing");
        let r = read_file(&cwd, "nope.txt", None, None);
        assert!(matches!(r.status, ToolStatus::RequestError));
        assert!(
            r.output.contains("nope.txt"),
            "expected error to mention path, got: {}",
            r.output
        );
        std::fs::remove_dir_all(&cwd).ok();
    }

    /// Path-traversal: any attempt to escape cwd via `..` must be rejected
    /// before we touch the filesystem. Coverage for the safe_resolve gate.
    #[test]
    fn read_file_rejects_escape_via_dotdot() {
        let cwd = fresh_tmp_dir("escape-read");
        let r = read_file(&cwd, "../../../etc/passwd", None, None);
        assert!(matches!(r.status, ToolStatus::RequestError));
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn read_file_rejects_absolute_path_outside_cwd() {
        let cwd = fresh_tmp_dir("abs-escape-read");
        let outside = std::env::temp_dir().join(format!("outside-{}", uuid::Uuid::new_v4()));
        std::fs::write(&outside, "secret").unwrap();

        let r = read_file(&cwd, outside.to_str().unwrap(), None, None);
        assert!(matches!(r.status, ToolStatus::RequestError));
        assert!(r.output.contains("escapes"));

        std::fs::remove_file(outside).ok();
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn read_file_rejects_oversized_file() {
        let cwd = fresh_tmp_dir("oversize-read");
        std::fs::write(
            cwd.join("huge.txt"),
            vec![b'x'; READ_MAX_BYTES as usize + 1],
        )
        .unwrap();

        let r = read_file(&cwd, "huge.txt", None, None);

        assert!(matches!(r.status, ToolStatus::RequestError));
        assert!(r.output.contains("exceeds cap"), "{}", r.output);
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn write_file_rejects_escape_via_dotdot() {
        let cwd = fresh_tmp_dir("escape-write");
        let w = write_file(&cwd, "../escaped.txt", "x");
        assert!(matches!(w.status, ToolStatus::RequestError));
        // The error message either mentions escape or unsupported `..`.
        assert!(
            w.output.contains("escapes") || w.output.contains(".."),
            "expected traversal error, got: {}",
            w.output
        );
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn write_file_rejects_oversized_payload() {
        let cwd = fresh_tmp_dir("oversize-write");
        let content = "x".repeat(WRITE_MAX_BYTES + 1);

        let w = write_file(&cwd, "huge.txt", &content);

        assert!(matches!(w.status, ToolStatus::RequestError));
        assert!(w.output.contains("exceeds cap"), "{}", w.output);
        assert!(!cwd.join("huge.txt").exists());
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn write_file_rejects_absolute_path_outside_cwd() {
        let cwd = fresh_tmp_dir("abs-escape-write");
        let outside = std::env::temp_dir().join(format!("outside-{}", uuid::Uuid::new_v4()));

        let w = write_file(&cwd, outside.to_str().unwrap(), "secret");
        assert!(matches!(w.status, ToolStatus::RequestError));
        assert!(w.output.contains("escapes"));
        assert!(!outside.exists());

        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn edit_file_replaces_single_occurrence() {
        let cwd = fresh_tmp_dir("edit-one");
        std::fs::write(cwd.join("a.txt"), "alpha\nbeta\ngamma\n").unwrap();

        let r = edit_file(&cwd, "a.txt", "beta", "BETA", false);
        assert!(matches!(r.status, ToolStatus::Success), "{}", r.output);
        assert_eq!(
            std::fs::read_to_string(cwd.join("a.txt")).unwrap(),
            "alpha\nBETA\ngamma\n"
        );
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn edit_file_replace_all_replaces_multiple_occurrences() {
        let cwd = fresh_tmp_dir("edit-all");
        std::fs::write(cwd.join("a.txt"), "x y x\n").unwrap();

        let r = edit_file(&cwd, "a.txt", "x", "z", true);
        assert!(matches!(r.status, ToolStatus::Success), "{}", r.output);
        assert_eq!(
            std::fs::read_to_string(cwd.join("a.txt")).unwrap(),
            "z y z\n"
        );
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn edit_file_rejects_ambiguous_single_replacement() {
        let cwd = fresh_tmp_dir("edit-ambiguous");
        std::fs::write(cwd.join("a.txt"), "x y x\n").unwrap();

        let r = edit_file(&cwd, "a.txt", "x", "z", false);
        assert!(matches!(r.status, ToolStatus::RequestError));
        assert!(r.output.contains("occurs 2 times"));
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn edit_file_rejects_missing_match_and_outside_absolute_path() {
        let cwd = fresh_tmp_dir("edit-reject");
        std::fs::write(cwd.join("a.txt"), "abc\n").unwrap();
        let missing = edit_file(&cwd, "a.txt", "zzz", "x", false);
        assert!(matches!(missing.status, ToolStatus::RequestError));
        assert!(missing.output.contains("No occurrences"));

        let outside = std::env::temp_dir().join(format!("outside-{}", uuid::Uuid::new_v4()));
        std::fs::write(&outside, "abc\n").unwrap();
        let escaped = edit_file(&cwd, outside.to_str().unwrap(), "abc", "x", false);
        assert!(matches!(escaped.status, ToolStatus::RequestError));
        assert!(escaped.output.contains("escapes"));

        std::fs::remove_file(outside).ok();
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn edit_file_rejects_oversized_result() {
        let cwd = fresh_tmp_dir("edit-oversize");
        std::fs::write(cwd.join("a.txt"), "a".repeat(WRITE_MAX_BYTES)).unwrap();

        let r = edit_file(&cwd, "a.txt", "a", "aa", true);

        assert!(matches!(r.status, ToolStatus::RequestError));
        assert!(r.output.contains("exceeds cap"), "{}", r.output);
        assert_eq!(
            std::fs::read_to_string(cwd.join("a.txt")).unwrap().len(),
            WRITE_MAX_BYTES
        );
        std::fs::remove_dir_all(&cwd).ok();
    }

    /// `list_directory` sorts entries alphabetically and suffixes
    /// directories with `/` so the LLM can distinguish them without an
    /// extra round-trip.
    #[test]
    fn list_directory_sorts_and_marks_subdirs() {
        let cwd = fresh_tmp_dir("ls");
        std::fs::create_dir_all(cwd.join("zdir")).unwrap();
        std::fs::write(cwd.join("a.txt"), "").unwrap();
        std::fs::write(cwd.join("m.txt"), "").unwrap();

        let r = list_directory(&cwd, ".");
        assert!(matches!(r.status, ToolStatus::Success));
        let lines: Vec<&str> = r.output.lines().collect();
        assert_eq!(lines, vec!["a.txt", "m.txt", "zdir/"]);
        std::fs::remove_dir_all(&cwd).ok();
    }

    /// `list_directory` on a missing path is a `RequestError`.
    #[test]
    fn list_directory_missing_is_request_error() {
        let cwd = fresh_tmp_dir("ls-missing");
        let r = list_directory(&cwd, "no-such-dir");
        assert!(matches!(r.status, ToolStatus::RequestError));
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn list_directory_truncates_large_directories() {
        let cwd = fresh_tmp_dir("ls-truncate");
        for idx in 0..(LIST_MAX_ENTRIES + 5) {
            std::fs::write(cwd.join(format!("file-{idx:04}.txt")), "").unwrap();
        }

        let r = list_directory(&cwd, ".");

        assert!(matches!(r.status, ToolStatus::Success), "{}", r.output);
        assert_eq!(r.output.lines().count(), LIST_MAX_ENTRIES + 1);
        assert!(
            r.output
                .contains(&format!("... truncated at {LIST_MAX_ENTRIES} entries"))
        );
        std::fs::remove_dir_all(&cwd).ok();
    }

    /// Invalid regex must surface as a `RequestError` so the LLM can fix
    /// the pattern, not an `InternalError`.
    #[test]
    fn search_file_contents_invalid_regex_is_request_error() {
        let cwd = fresh_tmp_dir("bad-regex");
        let r = search_file_contents(&cwd, "(unclosed", None, None, 100);
        assert!(matches!(r.status, ToolStatus::RequestError));
        assert!(r.output.contains("Invalid regex"));
        std::fs::remove_dir_all(&cwd).ok();
    }

    /// Search returns matches in `path:line: snippet` format, with a
    /// "No matches" message when nothing hits.
    #[test]
    fn search_file_contents_finds_matching_lines_and_reports_empty() {
        let cwd = fresh_tmp_dir("search-hit");
        std::fs::write(cwd.join("a.txt"), "alpha\nbeta\ngamma\n").unwrap();
        std::fs::write(cwd.join("b.txt"), "delta\n").unwrap();

        let hit = search_file_contents(&cwd, "beta", None, None, 100);
        assert!(matches!(hit.status, ToolStatus::Success));
        assert!(
            hit.output.contains("a.txt:2: beta"),
            "expected match line, got: {}",
            hit.output
        );

        let miss = search_file_contents(&cwd, "no-such-token", None, None, 100);
        assert!(matches!(miss.status, ToolStatus::Success));
        assert!(miss.output.contains("No matches found"));
        std::fs::remove_dir_all(&cwd).ok();
    }

    /// Glob filter must restrict the walk to matching files; non-matching
    /// files are not searched even if they contain the pattern.
    #[test]
    fn search_file_contents_glob_filter_limits_search() {
        let cwd = fresh_tmp_dir("search-glob");
        std::fs::write(cwd.join("keep.rs"), "needle\n").unwrap();
        std::fs::write(cwd.join("skip.txt"), "needle\n").unwrap();

        let r = search_file_contents(&cwd, "needle", Some("*.rs"), None, 100);
        assert!(matches!(r.status, ToolStatus::Success));
        assert!(r.output.contains("keep.rs"));
        assert!(
            !r.output.contains("skip.txt"),
            "glob *.rs must exclude skip.txt, got: {}",
            r.output
        );
        std::fs::remove_dir_all(&cwd).ok();
    }

    /// `max_results` must cap output and append a `... truncated` marker so
    /// the LLM knows there are more matches it didn't see.
    #[test]
    fn search_file_contents_truncates_at_max_results() {
        let cwd = fresh_tmp_dir("search-cap");
        let body: String = (0..20).map(|_| "needle\n").collect();
        std::fs::write(cwd.join("a.txt"), body).unwrap();

        let r = search_file_contents(&cwd, "needle", None, None, 3);
        assert!(matches!(r.status, ToolStatus::Success));
        // 3 matches + 1 truncation marker = 4 lines.
        assert_eq!(r.output.lines().count(), 4);
        assert!(r.output.contains("... truncated at 3 results"));
        std::fs::remove_dir_all(&cwd).ok();
    }

    /// Files containing a NUL byte in the first sniff window must be
    /// classified as binary and skipped, even if they would otherwise match
    /// the regex.
    #[test]
    fn search_file_contents_skips_binary_files() {
        let cwd = fresh_tmp_dir("binary");
        // NUL early so the sniff catches it; "needle" appears literally
        // after the NUL but the file should never be opened.
        let mut bytes = vec![b'h', b'i', 0u8, b'\n'];
        bytes.extend_from_slice(b"needle\n");
        std::fs::write(cwd.join("data.bin"), bytes).unwrap();
        std::fs::write(cwd.join("real.txt"), "needle\n").unwrap();

        let r = search_file_contents(&cwd, "needle", None, None, 100);
        assert!(matches!(r.status, ToolStatus::Success));
        assert!(r.output.contains("real.txt"));
        assert!(
            !r.output.contains("data.bin"),
            "binary file must be skipped, got: {}",
            r.output
        );
        std::fs::remove_dir_all(&cwd).ok();
    }

    /// Hidden directories (`.git`), `node_modules`, `target`, and
    /// `__pycache__` are pruned by `filter_entry` so the walk doesn't
    /// drown in transient build output.
    #[test]
    fn search_file_contents_skips_well_known_noise_directories() {
        let cwd = fresh_tmp_dir("noise");
        for noisy in [".git", "node_modules", "target", "__pycache__"] {
            std::fs::create_dir_all(cwd.join(noisy)).unwrap();
            std::fs::write(cwd.join(noisy).join("hit.txt"), "needle\n").unwrap();
        }
        std::fs::write(cwd.join("real.txt"), "needle\n").unwrap();

        let r = search_file_contents(&cwd, "needle", None, None, 100);
        assert!(matches!(r.status, ToolStatus::Success));
        assert!(r.output.contains("real.txt"));
        for noisy in [".git", "node_modules", "target", "__pycache__"] {
            assert!(
                !r.output.contains(noisy),
                "noise dir '{}' must be pruned, got: {}",
                noisy,
                r.output
            );
        }
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn search_file_contents_accepts_scoped_absolute_path_inside_cwd() {
        let cwd = fresh_tmp_dir("search-path");
        std::fs::create_dir_all(cwd.join("src")).unwrap();
        std::fs::write(cwd.join("src").join("hit.txt"), "needle\n").unwrap();
        std::fs::write(cwd.join("miss.txt"), "needle\n").unwrap();
        let scope = cwd.join("src");

        let r = search_file_contents(&cwd, "needle", None, Some(scope.to_str().unwrap()), 100);
        assert!(matches!(r.status, ToolStatus::Success), "{}", r.output);
        assert!(r.output.contains("hit.txt"));
        assert!(!r.output.contains("miss.txt"));
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn search_file_contents_accepts_scoped_file_path() {
        let cwd = fresh_tmp_dir("search-file");
        std::fs::write(cwd.join("hit.txt"), "needle\n").unwrap();
        std::fs::write(cwd.join("miss.txt"), "needle\n").unwrap();

        let r = search_file_contents(&cwd, "needle", None, Some("hit.txt"), 100);
        assert!(matches!(r.status, ToolStatus::Success), "{}", r.output);
        assert!(r.output.contains("hit.txt:1: needle"));
        assert!(!r.output.contains("miss.txt"));
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn search_file_contents_scoped_file_uses_requested_wasm_backend() {
        let cwd = fresh_tmp_dir("search-file-wasm");
        std::fs::write(cwd.join("hit+one.txt"), "needle\n").unwrap();
        std::fs::write(cwd.join("miss.txt"), "needle\n").unwrap();

        let r = search_file_contents_with_sandbox_mode(
            &cwd,
            "needle",
            None,
            Some("hit+one.txt"),
            100,
            Some(crate::sandbox_backend::SandboxMode::Wasm),
        );
        if crate::sandbox_backend::wasm_sandbox_compiled() {
            assert!(matches!(r.status, ToolStatus::Success), "{}", r.output);
            assert!(r.output.contains("hit+one.txt:1: needle"));
            assert!(
                !r.output.contains("miss.txt"),
                "wasm single-file search must stay scoped to the requested file, got: {}",
                r.output
            );
        } else {
            assert!(
                matches!(r.status, ToolStatus::InternalError),
                "{}",
                r.output
            );
            assert!(
                r.output.contains("not compiled into this build"),
                "{}",
                r.output
            );
        }
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn list_directory_accepts_absolute_path_inside_cwd() {
        let cwd = fresh_tmp_dir("list-abs");
        std::fs::create_dir_all(cwd.join("src")).unwrap();
        std::fs::write(cwd.join("src").join("lib.rs"), "").unwrap();

        let r = list_directory(&cwd, cwd.join("src").to_str().unwrap());
        assert!(matches!(r.status, ToolStatus::Success), "{}", r.output);
        assert_eq!(r.output, "lib.rs");
        std::fs::remove_dir_all(&cwd).ok();
    }

    #[test]
    fn search_file_contents_rejects_absolute_path_outside_cwd() {
        let cwd = fresh_tmp_dir("search-escape");
        let outside = std::env::temp_dir().join(format!("outside-{}", uuid::Uuid::new_v4()));
        std::fs::write(&outside, "needle\n").unwrap();

        let r = search_file_contents(&cwd, "needle", None, Some(outside.to_str().unwrap()), 100);
        assert!(matches!(r.status, ToolStatus::RequestError));
        assert!(r.output.contains("escapes"));

        std::fs::remove_file(outside).ok();
        std::fs::remove_dir_all(&cwd).ok();
    }

    // The binary-sniff test moved to `brokk-acp-sandbox::search` along
    // with `is_binary_file` itself; the host fn is gone and the test
    // would now be redundant with the sandbox unit test.
}
