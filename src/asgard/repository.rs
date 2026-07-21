#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::Duration;
use std::{fs, io::Write};

use anyhow::{Context, Result, bail};

#[derive(Debug, Clone)]
pub(crate) struct Config {
    pub candidate_models: Vec<String>,
    pub supervisor_model: Option<String>,
}

static CONFIG: OnceLock<Option<Config>> = OnceLock::new();

pub(crate) fn configure(config: Option<Config>) {
    let _ = CONFIG.set(config);
}

pub(crate) fn config() -> Option<&'static Config> {
    CONFIG.get().and_then(Option::as_ref)
}

#[derive(Debug)]
pub(crate) struct CandidateRepository {
    pub root: PathBuf,
    pub session_cwd: PathBuf,
    parent_root: PathBuf,
}

pub(crate) fn ensure_compatible_checkout(cwd: &Path) -> Result<()> {
    let output = git(cwd, &["status", "--porcelain"])?;
    let status = String::from_utf8(output.stdout).context("git status was not UTF-8")?;
    let unexpected: Vec<_> = status
        .lines()
        .filter(|line| {
            let path = line.get(3..).unwrap_or("");
            !path.starts_with(".brokk/") && !path.starts_with(".bifrost/")
        })
        .collect();
    if !unexpected.is_empty() {
        bail!(
            "Asgard requires code files to be clean (dirty: {})",
            unexpected.join(", ")
        );
    }
    Ok(())
}

pub(crate) fn parent_head_commit(cwd: &Path) -> Result<String> {
    Ok(git_text(cwd, &["rev-parse", "HEAD"])?.trim().to_string())
}

/// Applies the selected candidate delta to the live checkout without resetting
/// harness-owned state such as `.brokk/` and `.bifrost/`.
pub(crate) fn apply_selected_patch(root: &Path, patch: &[u8]) -> Result<()> {
    if patch.is_empty() {
        return Ok(());
    }
    let mut child = Command::new("git")
        .args(["apply", "--binary", "-"])
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .context("git apply stdin")?
        .write_all(patch)?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        bail!(
            "{}",
            git_error_message(root, &["apply", "--binary", "-"], &output)
        );
    }
    Ok(())
}

#[cfg(test)]
fn create_candidate_repository(cwd: &Path, label: &str) -> Result<CandidateRepository> {
    let checkout_commit = git_text(cwd, &["rev-parse", "HEAD"])?;
    create_candidate_repository_at(cwd, label, checkout_commit.trim())
}

pub(crate) fn create_candidate_repository_at(
    cwd: &Path,
    label: &str,
    checkout_commit: &str,
) -> Result<CandidateRepository> {
    let repo = git_text(cwd, &["rev-parse", "--show-toplevel"])?;
    let repo = PathBuf::from(repo.trim());
    let relative = cwd.strip_prefix(&repo).unwrap_or(Path::new(""));
    let parent = std::env::temp_dir().join("anvil-asgard-worktrees");
    fs::create_dir_all(&parent)?;
    let root = parent.join(format!(
        "asgard-{}-{}",
        safe_repository_label(label),
        uuid::Uuid::new_v4()
    ));
    if let Err(error) = git_worktree_add(&repo, &root, checkout_commit) {
        remove_directory(&root, "incomplete Asgard worktree");
        return Err(error);
    }
    let repository = CandidateRepository {
        session_cwd: root.join(relative),
        parent_root: repo.clone(),
        root,
    };
    if let Err(error) = seed_build_bootstrap_files(&repo, &repository.root) {
        remove_candidate_repository(&repository);
        return Err(error);
    }
    Ok(repository)
}

pub(crate) fn recycle_repository(
    repository: &CandidateRepository,
    checkout_commit: &str,
) -> Result<()> {
    git(
        &repository.root,
        &[
            "checkout",
            "--detach",
            "--quiet",
            "--force",
            checkout_commit,
        ],
    )?;
    git(&repository.root, &["clean", "-fd"])?;
    Ok(())
}

fn safe_repository_label(label: &str) -> String {
    let sanitized: String = label
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '-'
            }
        })
        .collect();
    sanitized.trim_matches('-').to_owned()
}

/// Fresh Git worktrees omit ignored files that a harness may have provisioned
/// after checkout. Keep small build-launcher payloads available to every lane so
/// candidates have the same validation entry point as the parent checkout.
fn seed_build_bootstrap_files(repo: &Path, worktree: &Path) -> Result<()> {
    for relative in [Path::new("gradle/wrapper"), Path::new(".mvn/wrapper")] {
        let source = repo.join(relative);
        if source.is_dir() {
            copy_missing_tree(&source, &worktree.join(relative))?;
        }
    }
    Ok(())
}

fn copy_missing_tree(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination).with_context(|| {
        format!(
            "create Asgard bootstrap directory {}",
            destination.display()
        )
    })?;
    for entry in fs::read_dir(source)
        .with_context(|| format!("read Asgard bootstrap directory {}", source.display()))?
    {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            copy_missing_tree(&source_path, &destination_path)?;
        } else if file_type.is_file() && !destination_path.exists() {
            fs::copy(&source_path, &destination_path).with_context(|| {
                format!(
                    "copy Asgard bootstrap file {} to {}",
                    source_path.display(),
                    destination_path.display()
                )
            })?;
        }
    }
    Ok(())
}

#[cfg(test)]
fn capture_patch(root: &Path, base_commit: &str) -> Result<Vec<u8>> {
    let index_path =
        std::env::temp_dir().join(format!("anvil-asgard-index-{}", uuid::Uuid::new_v4()));
    let _index_guard = TemporaryIndex::new(index_path.clone());
    git_with_index(root, &index_path, &["read-tree", base_commit])?;
    add_intent_to_add_untracked(root, &index_path)?;
    Ok(git_with_index(
        root,
        &index_path,
        &[
            "diff",
            "--binary",
            "--no-ext-diff",
            base_commit,
            "--",
            ".",
            ":(exclude).brokk/**",
            ":(exclude).bifrost/**",
        ],
    )?
    .stdout)
}

pub(crate) fn capture_diffstat(root: &Path, base_commit: &str) -> Result<String> {
    capture_worktree_diffstat(root, base_commit)
}

fn capture_worktree_diffstat(root: &Path, base_commit: &str) -> Result<String> {
    let index_path =
        std::env::temp_dir().join(format!("anvil-asgard-index-{}", uuid::Uuid::new_v4()));
    let _index_guard = TemporaryIndex::new(index_path.clone());
    git_with_index(root, &index_path, &["read-tree", base_commit])?;
    add_intent_to_add_untracked(root, &index_path)?;
    String::from_utf8(
        git_with_index(
            root,
            &index_path,
            &[
                "diff",
                "--stat",
                "--no-ext-diff",
                base_commit,
                "--",
                ".",
                ":(exclude).brokk/**",
                ":(exclude).bifrost/**",
            ],
        )?
        .stdout,
    )
    .context("git diff --stat output was not UTF-8")
}

fn add_intent_to_add_untracked(root: &Path, index: &Path) -> Result<()> {
    let listed = git_with_index(
        root,
        index,
        &[
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            ".",
            ":(exclude).brokk/**",
            ":(exclude).bifrost/**",
        ],
    )?
    .stdout;
    // Candidates run concurrently with this snapshot, so a listed file can
    // vanish before `git add -N` resolves it (observed: SQLite -shm files),
    // and git treats an unmatched pathspec as fatal. Re-check existence and
    // feed git only paths that are still present.
    let untracked: Vec<u8> = listed
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .filter(|path| path_exists(root, path))
        .flat_map(|path| path.iter().copied().chain(std::iter::once(0)))
        .collect();
    if untracked.is_empty() {
        return Ok(());
    }
    let mut child = Command::new("git")
        .args(["add", "-N", "--pathspec-from-file=-", "--pathspec-file-nul"])
        .current_dir(root)
        .env("GIT_INDEX_FILE", index)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .context("write untracked paths to temporary Asgard index")?
        .write_all(&untracked)?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        bail!(
            "{}",
            git_error_message(
                root,
                &["add", "-N", "--pathspec-from-file=-", "--pathspec-file-nul"],
                &output
            )
        );
    }
    Ok(())
}

#[cfg(unix)]
fn path_exists(root: &Path, path: &[u8]) -> bool {
    root.join(std::ffi::OsStr::from_bytes(path))
        .symlink_metadata()
        .is_ok()
}

#[cfg(not(unix))]
fn path_exists(root: &Path, path: &[u8]) -> bool {
    std::str::from_utf8(path)
        .map(|path| root.join(path).symlink_metadata().is_ok())
        .unwrap_or(false)
}

struct TemporaryIndex {
    path: PathBuf,
}

impl TemporaryIndex {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Drop for TemporaryIndex {
    fn drop(&mut self) {
        for path in [&self.path, &self.path.with_extension("lock")] {
            if let Err(error) = fs::remove_file(path)
                && error.kind() != std::io::ErrorKind::NotFound
            {
                tracing::warn!(path = %path.display(), "failed to remove temporary Asgard index: {error}");
            }
        }
    }
}

pub(crate) struct SnapshotStage {
    parent_root: PathBuf,
    run_id: String,
    ref_lock: Mutex<()>,
}

impl SnapshotStage {
    pub(crate) fn new(parent_root: &Path, run_id: &str) -> Result<Self> {
        let parent_root =
            PathBuf::from(git_text(parent_root, &["rev-parse", "--show-toplevel"])?.trim());
        Ok(Self {
            parent_root,
            run_id: run_id.to_string(),
            ref_lock: Mutex::new(()),
        })
    }

    pub(crate) fn snapshot(&self, worker_root: &Path, name: &str) -> Result<String> {
        add_all_for_checkpoint(worker_root)?;
        let message = format!("asgard checkpoint {name}");
        // Bookkeeping commits must not run repo hooks: porcelain `git commit`
        // triggers husky/commitlint (observed: happy-dom rejecting the message
        // format and eslint --fix mutating files mid-checkpoint), which the old
        // commit-tree plumbing never did.
        git(
            worker_root,
            &[
                "-c",
                "user.name=asgard",
                "-c",
                "user.email=asgard@anvil.invalid",
                "-c",
                "core.hooksPath=/dev/null",
                "commit",
                "--no-verify",
                "--allow-empty",
                "-m",
                &message,
            ],
        )?;
        let commit = git_text(worker_root, &["rev-parse", "HEAD"])?
            .trim()
            .to_string();
        let reference = format!("refs/asgard/{}/{name}", self.run_id);
        self.update_ref(&reference, &commit)?;
        Ok(commit)
    }

    pub(crate) fn finalize_patch(
        &self,
        base_commit: &str,
        checkpoint_commit: &str,
    ) -> Result<Vec<u8>> {
        Ok(git(
            &self.parent_root,
            &[
                "diff",
                "--binary",
                "--no-ext-diff",
                base_commit,
                checkpoint_commit,
                "--",
                ".",
                ":(exclude).brokk/**",
                ":(exclude).bifrost/**",
            ],
        )?
        .stdout)
    }

    fn update_ref(&self, reference: &str, commit: &str) -> Result<()> {
        let _guard = self
            .ref_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        git(&self.parent_root, &["update-ref", reference, commit])?;
        Ok(())
    }

    pub(crate) fn cleanup(&self) {
        let prefix = format!("refs/asgard/{}", self.run_id);
        match git(
            &self.parent_root,
            &["for-each-ref", "--format=%(refname)", &prefix],
        ) {
            Ok(output) => match String::from_utf8(output.stdout) {
                Ok(refs) => {
                    for reference in refs.lines().filter(|line| !line.is_empty()) {
                        if let Err(error) = git(&self.parent_root, &["update-ref", "-d", reference])
                        {
                            tracing::warn!(
                                refname = reference,
                                "failed to delete Asgard snapshot ref: {error}"
                            );
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!(
                        "git for-each-ref output was not UTF-8 during Asgard snapshot cleanup: {error}"
                    );
                }
            },
            Err(error) => {
                tracing::warn!("failed to list Asgard snapshot refs for cleanup: {error}");
            }
        }
        if let Err(error) = git(&self.parent_root, &["worktree", "prune"]) {
            tracing::warn!("failed to prune Asgard worktrees during cleanup: {error}");
        }
    }
}

pub(crate) fn remove_candidate_repository(repository: &CandidateRepository) {
    if let Err(error) = git(
        &repository.parent_root,
        &[
            "worktree",
            "remove",
            "--force",
            path_to_str(&repository.root).unwrap_or(""),
        ],
    ) {
        tracing::warn!(
            path = %repository.root.display(),
            "failed to remove Asgard worktree via git: {error}"
        );
        remove_directory(&repository.root, "Asgard candidate worktree");
    }
    if let Err(error) = git(&repository.parent_root, &["worktree", "prune"]) {
        tracing::warn!("failed to prune Asgard worktrees after removal: {error}");
    }
}

fn remove_directory(path: &Path, description: &str) {
    if let Err(error) = fs::remove_dir_all(path)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!(path = %path.display(), "failed to remove {description}: {error}");
    }
}

fn git(cwd: &Path, args: &[&str]) -> Result<std::process::Output> {
    let output = Command::new("git").args(args).current_dir(cwd).output()?;
    if !output.status.success() {
        bail!("{}", git_error_message(cwd, args, &output));
    }
    Ok(output)
}

fn git_with_index(cwd: &Path, index: &Path, args: &[&str]) -> Result<std::process::Output> {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .env("GIT_INDEX_FILE", index)
        .output()?;
    if !output.status.success() {
        bail!("{}", git_error_message(cwd, args, &output));
    }
    Ok(output)
}

fn git_worktree_add(parent_root: &Path, root: &Path, checkout_commit: &str) -> Result<()> {
    let root = path_to_str(root)?;
    git(
        parent_root,
        &[
            "worktree",
            "add",
            "--detach",
            "--quiet",
            root,
            checkout_commit,
        ],
    )?;
    Ok(())
}

fn add_all_for_checkpoint(root: &Path) -> Result<()> {
    let args = [
        "add",
        "-A",
        "--",
        ".",
        ":(exclude).brokk/**",
        ":(exclude).bifrost/**",
    ];
    let mut last_error = None;
    for attempt in 0..3 {
        match git(root, &args) {
            Ok(_) => return Ok(()),
            Err(error) => {
                last_error = Some(error);
                if attempt < 2 {
                    thread::sleep(Duration::from_millis(50));
                }
            }
        }
    }
    add_all_for_checkpoint_from_filtered_pathspecs(root).with_context(|| {
        format!(
            "git add -A failed after retries in {}; final retry error: {}",
            root.display(),
            last_error
                .as_ref()
                .map(|error| format!("{error:#}"))
                .unwrap_or_else(|| "unknown".to_string())
        )
    })
}

fn add_all_for_checkpoint_from_filtered_pathspecs(root: &Path) -> Result<()> {
    git(
        root,
        &[
            "add",
            "-u",
            "--",
            ".",
            ":(exclude).brokk/**",
            ":(exclude).bifrost/**",
        ],
    )?;
    let listed = git(
        root,
        &[
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            ".",
            ":(exclude).brokk/**",
            ":(exclude).bifrost/**",
        ],
    )?
    .stdout;
    let untracked: Vec<u8> = listed
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .filter(|path| path_exists(root, path))
        .flat_map(|path| path.iter().copied().chain(std::iter::once(0)))
        .collect();
    if untracked.is_empty() {
        return Ok(());
    }
    let mut child = Command::new("git")
        .args(["add", "--pathspec-from-file=-", "--pathspec-file-nul"])
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .context("write untracked paths to Asgard checkpoint add")?
        .write_all(&untracked)?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        bail!(
            "{}",
            git_error_message(root, &["add", "--pathspec-from-file=-"], &output)
        );
    }
    Ok(())
}

fn path_to_str(path: &Path) -> Result<&str> {
    path.to_str()
        .ok_or_else(|| anyhow::anyhow!("path is not valid UTF-8: {}", path.display()))
}

fn git_error_message(cwd: &Path, args: &[&str], output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stderr = stderr.trim();
    if stderr.is_empty() {
        format!(
            "git {} failed in {} with status {}",
            args.join(" "),
            cwd.display(),
            output.status
        )
    } else {
        format!(
            "git {} failed in {} with status {}; stderr:\n{}",
            args.join(" "),
            cwd.display(),
            output.status,
            stderr
        )
    }
}

fn git_text(cwd: &Path, args: &[&str]) -> Result<String> {
    String::from_utf8(git(cwd, args)?.stdout).context("git output was not UTF-8")
}

/// Execution cap for the supervisor's `git` tool: combined stdout+stderr is
/// truncated to this many bytes before being handed back as the tool
/// result. Independent of, and applied before, the 8 KiB permanent-record
/// retention cap (`RETAINED_PAYLOAD_CAP` in supervisor.rs).
pub(crate) const SUPERVISOR_GIT_OUTPUT_CAP: usize = 32 * 1024;

pub(crate) struct SupervisorGitOutcome {
    /// The rendered tool result: combined stdout+stderr (truncated at
    /// `SUPERVISOR_GIT_OUTPUT_CAP`, with a conflict-abort note and exit code
    /// appended when relevant).
    pub(crate) text: String,
    pub(crate) exit_code: i32,
    /// Untruncated combined stdout+stderr size, for tracing.
    pub(crate) bytes: usize,
}

/// Runs `git <args>` for the supervisor's `git` tool in `root` (the
/// supervisor's scratch worktree). Never goes through a shell: `args` is
/// passed to the git CLI as argv, exactly as received. Unlike the other git
/// helpers in this module, a nonzero git exit code is not a Rust-level
/// error - the caller (and ultimately the supervisor) sees the command's
/// real output and exit code either way.
pub(crate) fn run_supervisor_git(root: &Path, args: &[String]) -> Result<SupervisorGitOutcome> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .env("GIT_PAGER", "cat")
        .env("GIT_EDITOR", "true")
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(Stdio::null())
        .output()
        .context("failed to spawn git for the supervisor git tool")?;

    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    let bytes = combined.len();
    let exit_code = output.status.code().unwrap_or(-1);

    let mut text = if combined.len() > SUPERVISOR_GIT_OUTPUT_CAP {
        let prefix = crate::text::truncate_utf8(&combined, SUPERVISOR_GIT_OUTPUT_CAP);
        format!("{prefix}\n[truncated, {bytes} bytes total]")
    } else {
        combined
    };

    if exit_code != 0
        && crate::asgard::first_non_flag_arg(args) == Some("merge")
        && merge_head_exists(root)
    {
        let _ = git(root, &["merge", "--abort"]);
        text.push_str(
            "\nnote: conflicted merge aborted; to resolve conflicts spawn a worker instead.",
        );
    }

    if exit_code != 0 {
        text.push_str(&format!("\nexit code: {exit_code}"));
    }

    Ok(SupervisorGitOutcome {
        text,
        exit_code,
        bytes,
    })
}

fn merge_head_exists(root: &Path) -> bool {
    Command::new("git")
        .args(["rev-parse", "--verify", "-q", "MERGE_HEAD"])
        .current_dir(root)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_git(root: &Path, args: &[&str]) {
        assert!(
            Command::new("git")
                .args(args)
                .current_dir(root)
                .status()
                .unwrap()
                .success(),
            "git {} failed",
            args.join(" ")
        );
    }

    fn assert_text_file_eq(path: &Path, expected: &str) {
        let actual = fs::read_to_string(path).unwrap();
        assert_eq!(actual.replace("\r\n", "\n"), expected);
    }

    fn configure_test_user(repo: &Path) {
        run_git(repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(repo, &["config", "user.name", "Asgard Test"]);
    }

    #[test]
    fn bootstrap_seed_copies_missing_wrappers_without_copying_build_outputs() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path().join("repo");
        let worktree = temp.path().join("worktree");
        fs::create_dir_all(repo.join("gradle/wrapper")).unwrap();
        fs::create_dir_all(repo.join(".mvn/wrapper")).unwrap();
        fs::create_dir_all(repo.join("build/cache")).unwrap();
        fs::create_dir_all(worktree.join("gradle/wrapper")).unwrap();
        fs::write(repo.join("gradle/wrapper/gradle-wrapper.jar"), b"gradle").unwrap();
        fs::write(
            repo.join("gradle/wrapper/gradle-wrapper.properties"),
            b"parent",
        )
        .unwrap();
        fs::write(
            worktree.join("gradle/wrapper/gradle-wrapper.properties"),
            b"tracked worktree copy",
        )
        .unwrap();
        fs::write(repo.join(".mvn/wrapper/maven-wrapper.jar"), b"maven").unwrap();
        fs::write(repo.join("build/cache/large.bin"), b"cache").unwrap();

        seed_build_bootstrap_files(&repo, &worktree).unwrap();

        assert_eq!(
            fs::read(worktree.join("gradle/wrapper/gradle-wrapper.jar")).unwrap(),
            b"gradle"
        );
        assert_eq!(
            fs::read(worktree.join(".mvn/wrapper/maven-wrapper.jar")).unwrap(),
            b"maven"
        );
        assert_eq!(
            fs::read(worktree.join("gradle/wrapper/gradle-wrapper.properties")).unwrap(),
            b"tracked worktree copy"
        );
        assert!(!worktree.join("build/cache/large.bin").exists());
    }

    #[test]
    fn repository_labels_are_safe_for_build_tool_paths() {
        assert_eq!(
            safe_repository_label("0-deepseek::deepseek-v4-flash"),
            "0-deepseek--deepseek-v4-flash"
        );
        assert_eq!(safe_repository_label("lane/model name"), "lane-model-name");
    }

    #[test]
    fn captured_patch_round_trips_tracked_deletions() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        run_git(repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("deleted.txt"), "remove me\n").unwrap();
        run_git(repo, &["add", "deleted.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);

        fs::remove_file(repo.join("deleted.txt")).unwrap();
        fs::write(repo.join("added.txt"), "keep me\n").unwrap();
        let base_commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();
        let patch = capture_patch(repo, base_commit.trim()).unwrap();
        let patch_text = String::from_utf8_lossy(&patch);
        assert!(patch_text.contains("deleted file mode"));
        assert!(patch_text.contains("added.txt"));

        run_git(repo, &["reset", "--hard", "HEAD"]);
        run_git(repo, &["clean", "-fd"]);
        apply_selected_patch(repo, &patch).unwrap();
        assert!(!repo.join("deleted.txt").exists());
        assert_text_file_eq(&repo.join("added.txt"), "keep me\n");
    }

    #[test]
    fn candidate_worktree_patch_applies_tracked_deletions_to_parent() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        run_git(repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("deleted.txt"), "remove me\n").unwrap();
        run_git(repo, &["add", "deleted.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let base_commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();

        let candidate = create_candidate_repository(repo, "deletion-test").unwrap();
        fs::remove_file(candidate.root.join("deleted.txt")).unwrap();
        fs::write(candidate.root.join("added.txt"), "keep me\n").unwrap();
        let patch = capture_patch(&candidate.root, base_commit.trim()).unwrap();
        assert!(String::from_utf8_lossy(&patch).contains("deleted file mode"));

        apply_selected_patch(repo, &patch).unwrap();
        remove_candidate_repository(&candidate);
        assert!(!repo.join("deleted.txt").exists());
        assert_text_file_eq(&repo.join("added.txt"), "keep me\n");
    }

    #[test]
    fn candidate_patch_includes_changes_committed_after_the_task_base() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        run_git(repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("tracked.txt"), "base\n").unwrap();
        run_git(repo, &["add", "tracked.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let base_commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();

        let candidate = create_candidate_repository(repo, "committed-test").unwrap();
        run_git(
            &candidate.root,
            &["config", "user.email", "candidate@example.invalid"],
        );
        run_git(&candidate.root, &["config", "user.name", "Candidate"]);
        run_git(&candidate.root, &["checkout", "-b", "solution"]);
        fs::write(candidate.root.join("tracked.txt"), "committed solution\n").unwrap();
        fs::write(candidate.root.join("added.txt"), "committed addition\n").unwrap();
        run_git(&candidate.root, &["add", "tracked.txt", "added.txt"]);
        run_git(&candidate.root, &["commit", "-m", "solution"]);

        let patch = capture_patch(&candidate.root, base_commit.trim()).unwrap();
        apply_selected_patch(repo, &patch).unwrap();
        remove_candidate_repository(&candidate);

        assert_text_file_eq(&repo.join("tracked.txt"), "committed solution\n");
        assert_text_file_eq(&repo.join("added.txt"), "committed addition\n");
    }

    #[test]
    fn snapshot_succeeds_despite_failing_repo_commit_hooks() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        fs::write(repo.join("tracked.txt"), "base\n").unwrap();
        run_git(repo, &["add", "tracked.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);

        // Simulate husky/commitlint: hooks shared via the parent .git reject
        // every commit message. Bookkeeping commits must bypass them.
        let hooks = repo.join(".git/hooks");
        fs::create_dir_all(&hooks).unwrap();
        for hook in ["commit-msg", "pre-commit"] {
            let path = hooks.join(hook);
            fs::write(&path, "#!/bin/sh\nexit 1\n").unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
            }
        }

        let worker = create_candidate_repository(repo, "hooked-worker").unwrap();
        fs::write(worker.root.join("tracked.txt"), "changed\n").unwrap();
        let stage = SnapshotStage::new(repo, &format!("test-{}", uuid::Uuid::new_v4())).unwrap();
        let checkpoint = stage.snapshot(&worker.root, "hooked").unwrap();
        assert_eq!(
            git_text(repo, &["show", &format!("{checkpoint}:tracked.txt")])
                .unwrap()
                .trim(),
            "changed"
        );
    }

    #[test]
    fn snapshot_writes_parent_ref_and_worktree_can_checkout_commit() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        fs::write(repo.join("tracked.txt"), "base\n").unwrap();
        run_git(repo, &["add", "tracked.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let base_commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();

        let worker = create_candidate_repository(repo, "snapshot-worker").unwrap();
        fs::write(worker.root.join("tracked.txt"), "snapshot\n").unwrap();
        fs::write(worker.root.join("untracked.txt"), "included\n").unwrap();
        fs::create_dir_all(worker.root.join(".brokk")).unwrap();
        fs::write(worker.root.join(".brokk/state.txt"), "excluded\n").unwrap();

        let stage = SnapshotStage::new(repo, &format!("test-{}", uuid::Uuid::new_v4())).unwrap();
        let checkpoint = stage.snapshot(&worker.root, "first").unwrap();

        assert_eq!(
            git_text(
                repo,
                &["rev-parse", &format!("refs/asgard/{}/first", stage.run_id)]
            )
            .unwrap()
            .trim(),
            checkpoint
        );
        assert_eq!(
            git_text(repo, &["show", &format!("{checkpoint}:tracked.txt")])
                .unwrap()
                .replace("\r\n", "\n"),
            "snapshot\n"
        );
        assert_eq!(
            git_text(repo, &["show", &format!("{checkpoint}:untracked.txt")])
                .unwrap()
                .replace("\r\n", "\n"),
            "included\n"
        );
        let tree_paths = git_text(repo, &["ls-tree", "-r", "--name-only", &checkpoint]).unwrap();
        assert!(!tree_paths.contains(".brokk/state.txt"));

        let materialized =
            create_candidate_repository_at(repo, "snapshot-materialized", &checkpoint).unwrap();
        assert_text_file_eq(&materialized.root.join("tracked.txt"), "snapshot\n");
        assert_text_file_eq(&materialized.root.join("untracked.txt"), "included\n");
        assert!(!materialized.root.join(".brokk/state.txt").exists());

        let second =
            create_candidate_repository_at(repo, "snapshot-reset", base_commit.trim()).unwrap();
        run_git(&second.root, &["reset", "--hard", &checkpoint]);
        assert_text_file_eq(&second.root.join("tracked.txt"), "snapshot\n");
        assert_text_file_eq(&second.root.join("untracked.txt"), "included\n");

        remove_candidate_repository(&worker);
        remove_candidate_repository(&materialized);
        remove_candidate_repository(&second);
        stage.cleanup();
    }

    #[test]
    fn snapshot_chains_to_previous_snapshot_commit() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        fs::write(repo.join("tracked.txt"), "base\n").unwrap();
        run_git(repo, &["add", "tracked.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let worker = create_candidate_repository(repo, "snapshot-chain").unwrap();
        let stage = SnapshotStage::new(repo, &format!("test-{}", uuid::Uuid::new_v4())).unwrap();
        fs::write(worker.root.join("tracked.txt"), "snapshot a\n").unwrap();
        let first = stage.snapshot(&worker.root, "a").unwrap();
        fs::write(worker.root.join("tracked.txt"), "snapshot b\n").unwrap();
        let second = stage.snapshot(&worker.root, "b").unwrap();

        assert_eq!(
            git_text(repo, &["rev-parse", &format!("{second}^")])
                .unwrap()
                .trim(),
            first
        );

        remove_candidate_repository(&worker);
        stage.cleanup();
    }

    #[test]
    fn snapshot_preserves_worker_commit_history() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        fs::write(repo.join("tracked.txt"), "base\n").unwrap();
        run_git(repo, &["add", "tracked.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let _base_commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();

        let worker = create_candidate_repository(repo, "snapshot-committed-worker").unwrap();
        configure_test_user(&worker.root);
        fs::write(worker.root.join("tracked.txt"), "worker commit\n").unwrap();
        fs::write(worker.root.join("committed.txt"), "committed addition\n").unwrap();
        run_git(&worker.root, &["add", "tracked.txt", "committed.txt"]);
        run_git(&worker.root, &["commit", "-m", "worker commit"]);
        let worker_commit = git_text(&worker.root, &["rev-parse", "HEAD"]).unwrap();
        fs::write(worker.root.join("tracked.txt"), "worktree snapshot\n").unwrap();
        fs::write(worker.root.join("untracked.txt"), "worktree addition\n").unwrap();

        let stage = SnapshotStage::new(repo, &format!("test-{}", uuid::Uuid::new_v4())).unwrap();
        let checkpoint = stage.snapshot(&worker.root, "after-worker-commit").unwrap();
        let checkpoint_parent = git_text(repo, &["rev-parse", &format!("{checkpoint}^")]).unwrap();

        assert_eq!(checkpoint_parent.trim(), worker_commit.trim());
        assert_eq!(
            git_text(repo, &["show", &format!("{checkpoint}:tracked.txt")])
                .unwrap()
                .replace("\r\n", "\n"),
            "worktree snapshot\n"
        );
        assert_eq!(
            git_text(repo, &["show", &format!("{checkpoint}:untracked.txt")])
                .unwrap()
                .replace("\r\n", "\n"),
            "worktree addition\n"
        );

        remove_candidate_repository(&worker);
        stage.cleanup();
    }

    #[test]
    fn recycle_repository_preserves_ignored_files_and_cleans_untracked() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        fs::write(repo.join(".gitignore"), "build-cache.txt\n").unwrap();
        fs::write(repo.join("tracked.txt"), "base\n").unwrap();
        run_git(repo, &["add", ".gitignore", "tracked.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let base_commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();

        fs::write(repo.join("tracked.txt"), "target\n").unwrap();
        fs::write(repo.join("target-only.txt"), "target file\n").unwrap();
        run_git(repo, &["add", "tracked.txt", "target-only.txt"]);
        run_git(repo, &["commit", "-m", "target"]);
        let target_commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();

        let candidate =
            create_candidate_repository_at(repo, "recycle", base_commit.trim()).unwrap();
        run_git(&candidate.root, &["checkout", "-b", "leaked-worker-branch"]);
        fs::write(candidate.root.join("branch-only.txt"), "branch file\n").unwrap();
        run_git(&candidate.root, &["add", "branch-only.txt"]);
        run_git(&candidate.root, &["commit", "-m", "worker branch"]);
        fs::write(candidate.root.join("build-cache.txt"), "keep cache\n").unwrap();
        fs::write(candidate.root.join("untracked.txt"), "remove me\n").unwrap();

        recycle_repository(&candidate, target_commit.trim()).unwrap();

        assert_text_file_eq(&candidate.root.join("build-cache.txt"), "keep cache\n");
        assert!(!candidate.root.join("untracked.txt").exists());
        assert_text_file_eq(&candidate.root.join("tracked.txt"), "target\n");
        assert_text_file_eq(&candidate.root.join("target-only.txt"), "target file\n");
        assert_eq!(
            git_text(&candidate.root, &["rev-parse", "HEAD"])
                .unwrap()
                .trim(),
            target_commit.trim()
        );
        assert_eq!(
            git_text(&candidate.root, &["branch", "--show-current"])
                .unwrap()
                .trim(),
            ""
        );

        remove_candidate_repository(&candidate);
    }

    #[test]
    fn finalized_snapshot_patch_applies_to_fresh_base_checkout() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        fs::write(repo.join("tracked.txt"), "base\n").unwrap();
        fs::write(repo.join("deleted.txt"), "delete\n").unwrap();
        run_git(repo, &["add", "tracked.txt", "deleted.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let base_commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();

        let worker = create_candidate_repository(repo, "patch-worker").unwrap();
        fs::write(worker.root.join("tracked.txt"), "changed\n").unwrap();
        fs::remove_file(worker.root.join("deleted.txt")).unwrap();
        fs::write(worker.root.join("added.txt"), "added\n").unwrap();
        let stage = SnapshotStage::new(repo, &format!("test-{}", uuid::Uuid::new_v4())).unwrap();
        let checkpoint = stage.snapshot(&worker.root, "patch").unwrap();
        let patch = stage
            .finalize_patch(base_commit.trim(), &checkpoint)
            .unwrap();

        let fresh =
            create_candidate_repository_at(repo, "patch-fresh", base_commit.trim()).unwrap();
        apply_selected_patch(&fresh.root, &patch).unwrap();
        assert_text_file_eq(&fresh.root.join("tracked.txt"), "changed\n");
        assert_text_file_eq(&fresh.root.join("added.txt"), "added\n");
        assert!(!fresh.root.join("deleted.txt").exists());

        remove_candidate_repository(&worker);
        remove_candidate_repository(&fresh);
        stage.cleanup();
    }

    #[test]
    fn run_supervisor_git_reports_output_exit_code_and_truncates_oversized_combined_output() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        fs::write(repo.join("base.txt"), "base\n").unwrap();
        run_git(repo, &["add", "base.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let worktree = create_candidate_repository(repo, "sv-git").unwrap();

        let ok = run_supervisor_git(
            &worktree.root,
            &["log".to_string(), "--oneline".to_string()],
        )
        .unwrap();
        assert_eq!(ok.exit_code, 0);
        assert!(ok.text.contains("initial"));
        assert!(!ok.text.contains("exit code"));

        let failed = run_supervisor_git(
            &worktree.root,
            &["show".to_string(), "not-a-real-ref".to_string()],
        )
        .unwrap();
        assert_ne!(failed.exit_code, 0);
        assert!(failed.text.contains("exit code:"));

        let big = "z".repeat(SUPERVISOR_GIT_OUTPUT_CAP * 2);
        fs::write(worktree.root.join("big.txt"), &big).unwrap();
        run_git(&worktree.root, &["add", "big.txt"]);
        run_git(&worktree.root, &["commit", "-m", "big file"]);
        let oversized = run_supervisor_git(
            &worktree.root,
            &["show".to_string(), "HEAD:big.txt".to_string()],
        )
        .unwrap();
        assert!(oversized.bytes > SUPERVISOR_GIT_OUTPUT_CAP);
        assert!(oversized.text.len() < oversized.bytes);
        assert!(
            oversized
                .text
                .contains(&format!("[truncated, {} bytes total]", oversized.bytes))
        );

        remove_candidate_repository(&worktree);
    }

    #[test]
    fn run_supervisor_git_aborts_conflicted_merge_and_leaves_worktree_clean() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        fs::write(repo.join("same.txt"), "base\n").unwrap();
        run_git(repo, &["add", "same.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let base_commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();
        let base_commit = base_commit.trim();

        let from_worker = create_candidate_repository(repo, "conflict-from").unwrap();
        fs::write(from_worker.root.join("same.txt"), "from\n").unwrap();
        run_git(&from_worker.root, &["commit", "-am", "from edits"]);
        let from_commit = git_text(&from_worker.root, &["rev-parse", "HEAD"])
            .unwrap()
            .trim()
            .to_string();

        let onto_worker =
            create_candidate_repository_at(repo, "conflict-onto", base_commit).unwrap();
        fs::write(onto_worker.root.join("same.txt"), "onto\n").unwrap();
        run_git(&onto_worker.root, &["commit", "-am", "onto edits"]);

        // The supervisor's scratch worktree, checked out at the same
        // divergent "onto" state, attempts to merge "from" and conflicts.
        let scratch = create_candidate_repository_at(
            repo,
            "sv-git-conflict",
            git_text(&onto_worker.root, &["rev-parse", "HEAD"])
                .unwrap()
                .trim(),
        )
        .unwrap();

        let result = run_supervisor_git(
            &scratch.root,
            &[
                "merge".to_string(),
                "--no-ff".to_string(),
                "--no-edit".to_string(),
                from_commit,
            ],
        )
        .unwrap();

        assert_ne!(result.exit_code, 0);
        assert!(result.text.contains("note: conflicted merge aborted"));
        assert!(result.text.contains("spawn a worker instead"));
        // The abort must actually run: no dangling MERGE_HEAD, clean status,
        // and a later, unrelated command in the same scratch worktree still
        // works (a stray merge state does not corrupt it for next use).
        assert!(!merge_head_exists(&scratch.root));
        assert_eq!(
            git_text(&scratch.root, &["status", "--porcelain"]).unwrap(),
            ""
        );
        let log_after =
            run_supervisor_git(&scratch.root, &["log".to_string(), "-1".to_string()]).unwrap();
        assert_eq!(log_after.exit_code, 0);

        remove_candidate_repository(&from_worker);
        remove_candidate_repository(&onto_worker);
        remove_candidate_repository(&scratch);
    }
}
