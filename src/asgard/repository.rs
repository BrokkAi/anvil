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

pub(crate) fn is_test_path(path: &str) -> bool {
    let basename = path.rsplit('/').next().unwrap_or(path);
    path.split('/')
        .any(|component| matches!(component, "test" | "tests" | "__tests__" | "testdata"))
        || basename.ends_with("_test.go")
        || basename.contains(".test.")
        || basename.contains(".spec.")
        || (basename.starts_with("test_") && basename.ends_with(".py"))
        || basename == "conftest.py"
}

pub(crate) fn sibling_test_files(
    repo_root: &Path,
    author_commit: &str,
    author_parent_commit: &str,
    candidate_commit: &str,
    max_files: usize,
) -> Result<Vec<(String, Vec<u8>)>> {
    if max_files == 0 {
        return Ok(Vec::new());
    }

    let candidate_verify = format!("{candidate_commit}^{{commit}}");
    git(repo_root, &["cat-file", "-e", &candidate_verify])?;

    let diff = git(
        repo_root,
        &[
            "diff",
            "--name-status",
            "--no-renames",
            "-z",
            author_parent_commit,
            author_commit,
        ],
    )?;
    let mut paths = Vec::new();
    let mut fields = diff.stdout.split(|byte| *byte == 0);
    while let Some(status) = fields.next() {
        if status.is_empty() {
            continue;
        }
        let Some(path) = fields.next() else {
            break;
        };
        let status = String::from_utf8(status.to_vec()).context("git diff status was not UTF-8")?;
        if !matches!(status.as_str(), "A" | "M") {
            continue;
        }
        let path = String::from_utf8(path.to_vec()).context("git diff path was not UTF-8")?;
        if is_test_path(&path) {
            paths.push(path);
        }
    }
    paths.sort();
    paths.dedup();

    let mut files = Vec::new();
    for path in paths {
        if files.len() >= max_files {
            break;
        }
        let author_blob = git_blob_id(repo_root, author_commit, &path)?
            .with_context(|| format!("missing author blob {author_commit}:{path}"))?;
        if git_blob_id(repo_root, candidate_commit, &path)?.as_deref() == Some(author_blob.as_str())
        {
            continue;
        }
        let revision = format!("{author_commit}:{path}");
        let content = git(repo_root, &["show", &revision])?.stdout;
        files.push((path, content));
    }
    Ok(files)
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

fn capture_commit_diffstat(root: &Path, base_commit: &str, target_commit: &str) -> Result<String> {
    String::from_utf8(
        git(
            root,
            &[
                "diff",
                "--stat",
                "--no-ext-diff",
                base_commit,
                target_commit,
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

#[derive(Debug)]
pub(crate) enum MergeCheckpointOutcome {
    Merged { commit: String, diffstat: String },
    NoChanges { diffstat: String },
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

    pub(crate) fn merge_checkpoint(
        &self,
        from_commit: &str,
        onto_commit: &str,
        name: &str,
    ) -> Result<MergeCheckpointOutcome> {
        let merge_base = merge_base(&self.parent_root, from_commit, onto_commit)?;
        let diffstat = capture_commit_diffstat(&self.parent_root, &merge_base, from_commit)?;
        if merge_base == from_commit {
            return Ok(MergeCheckpointOutcome::NoChanges { diffstat });
        }
        let scratch = create_candidate_repository_at(
            &self.parent_root,
            &format!("merge-{name}"),
            onto_commit,
        )?;
        let merge_result = git(
            &scratch.root,
            &[
                "-c",
                "user.name=asgard",
                "-c",
                "user.email=asgard@anvil.invalid",
                "-c",
                "core.hooksPath=/dev/null",
                "merge",
                "--no-verify",
                "--no-ff",
                "--no-edit",
                from_commit,
            ],
        );
        let commit = match merge_result {
            Ok(_) => git_text(&scratch.root, &["rev-parse", "HEAD"])?
                .trim()
                .to_string(),
            Err(error) => {
                let conflicts = conflicted_files_from_worktree(&scratch.root)?;
                let abort_result = git(&scratch.root, &["merge", "--abort"]);
                let abort_error = abort_result.err().map(|error| format!("{error:#}"));
                remove_candidate_repository(&scratch);
                return Err(merge_conflict_error(
                    conflicts,
                    &error,
                    abort_error.as_deref(),
                ));
            }
        };
        let reference = format!("refs/asgard/{}/{name}", self.run_id);
        if let Err(error) = self.update_ref(&reference, &commit) {
            remove_candidate_repository(&scratch);
            return Err(error);
        }
        remove_candidate_repository(&scratch);
        Ok(MergeCheckpointOutcome::Merged { commit, diffstat })
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

fn merge_base(root: &Path, from_commit: &str, onto_commit: &str) -> Result<String> {
    let output = Command::new("git")
        .args(["merge-base", from_commit, onto_commit])
        .current_dir(root)
        .output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "merge_checkpoint could not find a merge-base for {from_commit} and {onto_commit}; \
             these checkpoints appear to come from disjoint histories. Asgard checkpoints should \
             descend from the run root. git merge-base said: {}",
            stderr.trim()
        );
    }
    Ok(String::from_utf8(output.stdout)
        .context("git merge-base output was not UTF-8")?
        .trim()
        .to_string())
}

fn conflicted_files_from_worktree(root: &Path) -> Result<Vec<String>> {
    let output = git(root, &["diff", "--name-only", "--diff-filter=U", "-z"])?;
    Ok(output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| String::from_utf8_lossy(path).to_string())
        .collect())
}

fn merge_conflict_error(
    conflicts: Vec<String>,
    merge_error: &anyhow::Error,
    abort_error: Option<&str>,
) -> anyhow::Error {
    let mut message = String::from("merge_checkpoint failed to merge");
    if conflicts.is_empty() {
        message.push_str(&format!(":\n{merge_error:#}"));
    } else {
        message.push_str("; conflicted files:\n");
        for file in conflicts {
            message.push_str(&format!("- {file}\n"));
        }
        message.push_str(
            "The supervisor can spawn a worker from the onto checkpoint with instructions to \
             resolve these conflicts, then save that resolved checkpoint.",
        );
    }
    if let Some(abort_error) = abort_error {
        message.push_str(&format!("\ngit merge --abort failed: {abort_error}"));
    }
    anyhow::anyhow!("{message}")
}

fn git_text(cwd: &Path, args: &[&str]) -> Result<String> {
    String::from_utf8(git(cwd, args)?.stdout).context("git output was not UTF-8")
}

fn git_blob_id(cwd: &Path, commit: &str, path: &str) -> Result<Option<String>> {
    let revision = format!("{commit}:{path}");
    let output = Command::new("git")
        .args(["rev-parse", &revision])
        .current_dir(cwd)
        .output()?;
    if !output.status.success() {
        return Ok(None);
    }
    Ok(Some(
        String::from_utf8(output.stdout)
            .context("git rev-parse output was not UTF-8")?
            .trim()
            .to_string(),
    ))
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
    fn test_path_classifier_matches_only_test_conventions() {
        for path in [
            "tests/foo.py",
            "a/b/__tests__/x.ts",
            "pkg/x_test.go",
            "src/util.spec.tsx",
            "testdata/cfg.yml",
            "test_edge.py",
            "pkg/test/example.rs",
            "conftest.py",
        ] {
            assert!(is_test_path(path), "{path} should be classified as a test");
        }

        for path in [
            "src/lib.rs",
            "contest.py",
            "latest_go.go",
            "attestation/x.go",
        ] {
            assert!(
                !is_test_path(path),
                "{path} should not be classified as a test"
            );
        }
    }

    #[test]
    fn sibling_test_files_extracts_novel_sibling_tests() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        fs::write(repo.join("README.md"), "base\n").unwrap();
        run_git(repo, &["add", "README.md"]);
        run_git(repo, &["commit", "-m", "base"]);
        let base = git_text(repo, &["rev-parse", "HEAD"]).unwrap();
        let base = base.trim().to_string();

        run_git(repo, &["checkout", "-b", "author"]);
        fs::create_dir_all(repo.join("tests")).unwrap();
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(
            repo.join("tests/a_test.py"),
            "def test_a():\n    assert True\n",
        )
        .unwrap();
        fs::write(repo.join("src/x.py"), "VALUE = 1\n").unwrap();
        run_git(repo, &["add", "tests/a_test.py", "src/x.py"]);
        run_git(repo, &["commit", "-m", "author"]);
        let author = git_text(repo, &["rev-parse", "HEAD"]).unwrap();
        let author = author.trim().to_string();

        run_git(repo, &["checkout", "-b", "candidate", &base]);
        fs::write(repo.join("README.md"), "candidate\n").unwrap();
        run_git(repo, &["add", "README.md"]);
        run_git(repo, &["commit", "-m", "candidate"]);
        let candidate = git_text(repo, &["rev-parse", "HEAD"]).unwrap();
        let candidate = candidate.trim().to_string();

        let files = sibling_test_files(repo, &author, &base, &candidate, 24).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].0, "tests/a_test.py");
        assert_eq!(
            String::from_utf8(files[0].1.clone()).unwrap(),
            "def test_a():\n    assert True\n"
        );

        fs::create_dir_all(repo.join("tests")).unwrap();
        fs::write(
            repo.join("tests/a_test.py"),
            "def test_a():\n    assert True\n",
        )
        .unwrap();
        run_git(repo, &["add", "tests/a_test.py"]);
        run_git(repo, &["commit", "-m", "candidate has test"]);
        let candidate_with_test = git_text(repo, &["rev-parse", "HEAD"]).unwrap();
        let candidate_with_test = candidate_with_test.trim().to_string();

        let files = sibling_test_files(repo, &author, &base, &candidate_with_test, 24).unwrap();
        assert!(files.is_empty());
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
    fn merge_checkpoint_combines_sibling_changes_and_updates_ref() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        fs::write(repo.join("base.txt"), "base\n").unwrap();
        run_git(repo, &["add", "base.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let stage = SnapshotStage::new(repo, &format!("test-{}", uuid::Uuid::new_v4())).unwrap();

        let from_worker = create_candidate_repository(repo, "merge-from").unwrap();
        fs::write(from_worker.root.join("from.txt"), "from\n").unwrap();
        let from = stage.snapshot(&from_worker.root, "from").unwrap();

        let onto_worker = create_candidate_repository(repo, "merge-onto").unwrap();
        fs::write(onto_worker.root.join("onto.txt"), "onto\n").unwrap();
        let onto = stage.snapshot(&onto_worker.root, "onto").unwrap();

        let MergeCheckpointOutcome::Merged { commit, diffstat } =
            stage.merge_checkpoint(&from, &onto, "merged").unwrap()
        else {
            panic!("merge should create a checkpoint");
        };

        assert_eq!(
            git_text(
                repo,
                &["rev-parse", &format!("refs/asgard/{}/merged", stage.run_id)]
            )
            .unwrap()
            .trim(),
            commit
        );
        assert_eq!(
            git_text(repo, &["show", &format!("{commit}:from.txt")])
                .unwrap()
                .replace("\r\n", "\n"),
            "from\n"
        );
        assert_eq!(
            git_text(repo, &["show", &format!("{commit}:onto.txt")])
                .unwrap()
                .replace("\r\n", "\n"),
            "onto\n"
        );
        assert!(diffstat.contains("from.txt"));
        let parents = git_text(repo, &["rev-list", "--parents", "-n", "1", &commit]).unwrap();
        let parents = parents.split_whitespace().collect::<Vec<_>>();
        assert_eq!(parents, vec![commit.as_str(), onto.as_str(), from.as_str()]);

        remove_candidate_repository(&from_worker);
        remove_candidate_repository(&onto_worker);
        stage.cleanup();
    }

    #[test]
    fn merge_checkpoint_transplants_full_divergence_since_merge_base() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        fs::write(repo.join("base.txt"), "base\n").unwrap();
        run_git(repo, &["add", "base.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let stage = SnapshotStage::new(repo, &format!("test-{}", uuid::Uuid::new_v4())).unwrap();

        let from_one_worker = create_candidate_repository(repo, "merge-from-one").unwrap();
        fs::write(from_one_worker.root.join("first.txt"), "first\n").unwrap();
        let from_one = stage.snapshot(&from_one_worker.root, "from-one").unwrap();

        let from_two_worker =
            create_candidate_repository_at(repo, "merge-from-two", &from_one).unwrap();
        fs::write(from_two_worker.root.join("second.txt"), "second\n").unwrap();
        let from_two = stage.snapshot(&from_two_worker.root, "from-two").unwrap();

        let onto_worker = create_candidate_repository(repo, "merge-onto-multihop").unwrap();
        fs::write(onto_worker.root.join("onto.txt"), "onto\n").unwrap();
        let onto = stage.snapshot(&onto_worker.root, "onto").unwrap();

        let MergeCheckpointOutcome::Merged { commit, diffstat } =
            stage.merge_checkpoint(&from_two, &onto, "merged").unwrap()
        else {
            panic!("merge should create a checkpoint");
        };

        assert_eq!(
            git_text(repo, &["show", &format!("{commit}:first.txt")])
                .unwrap()
                .replace("\r\n", "\n"),
            "first\n"
        );
        assert_eq!(
            git_text(repo, &["show", &format!("{commit}:second.txt")])
                .unwrap()
                .replace("\r\n", "\n"),
            "second\n"
        );
        assert!(diffstat.contains("first.txt"));
        assert!(diffstat.contains("second.txt"));

        remove_candidate_repository(&from_one_worker);
        remove_candidate_repository(&from_two_worker);
        remove_candidate_repository(&onto_worker);
        stage.cleanup();
    }

    #[test]
    fn merge_checkpoint_noop_when_onto_already_contains_from() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        fs::write(repo.join("base.txt"), "base\n").unwrap();
        run_git(repo, &["add", "base.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let stage = SnapshotStage::new(repo, &format!("test-{}", uuid::Uuid::new_v4())).unwrap();

        let from_worker = create_candidate_repository(repo, "merge-noop-from").unwrap();
        fs::write(from_worker.root.join("from.txt"), "from\n").unwrap();
        let from = stage.snapshot(&from_worker.root, "from").unwrap();

        let onto_worker = create_candidate_repository_at(repo, "merge-noop-onto", &from).unwrap();
        fs::write(onto_worker.root.join("onto.txt"), "onto\n").unwrap();
        let onto = stage.snapshot(&onto_worker.root, "onto").unwrap();

        let MergeCheckpointOutcome::NoChanges { diffstat } =
            stage.merge_checkpoint(&from, &onto, "noop").unwrap()
        else {
            panic!("merge should be a no-op");
        };

        assert!(diffstat.is_empty());
        assert!(
            Command::new("git")
                .args([
                    "rev-parse",
                    "--verify",
                    &format!("refs/asgard/{}/noop", stage.run_id)
                ])
                .current_dir(repo)
                .status()
                .unwrap()
                .code()
                .is_some_and(|code| code != 0)
        );

        remove_candidate_repository(&from_worker);
        remove_candidate_repository(&onto_worker);
        stage.cleanup();
    }

    #[test]
    fn merge_checkpoint_conflict_does_not_create_checkpoint_ref() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        fs::write(repo.join("same.txt"), "base\n").unwrap();
        run_git(repo, &["add", "same.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let stage = SnapshotStage::new(repo, &format!("test-{}", uuid::Uuid::new_v4())).unwrap();

        let from_worker = create_candidate_repository(repo, "merge-conflict-from").unwrap();
        fs::write(from_worker.root.join("same.txt"), "from\n").unwrap();
        let from = stage.snapshot(&from_worker.root, "from").unwrap();

        let onto_worker = create_candidate_repository(repo, "merge-conflict-onto").unwrap();
        fs::write(onto_worker.root.join("same.txt"), "onto\n").unwrap();
        let onto = stage.snapshot(&onto_worker.root, "onto").unwrap();

        let error = stage
            .merge_checkpoint(&from, &onto, "scratch-conflict")
            .expect_err("conflict");
        let error = format!("{error:#}");

        assert!(error.contains("same.txt"));
        assert!(!error.contains("conflicting hunks"));
        assert!(error.contains("spawn a worker from the onto checkpoint"));
        assert_eq!(git_text(repo, &["status", "--porcelain"]).unwrap(), "");
        assert!(
            !git_text(repo, &["worktree", "list", "--porcelain"])
                .unwrap()
                .contains("asgard-merge-scratch-conflict-")
        );
        assert!(
            Command::new("git")
                .args([
                    "rev-parse",
                    "--verify",
                    &format!("refs/asgard/{}/scratch-conflict", stage.run_id)
                ])
                .current_dir(repo)
                .status()
                .unwrap()
                .code()
                .is_some_and(|code| code != 0)
        );

        remove_candidate_repository(&from_worker);
        remove_candidate_repository(&onto_worker);
        stage.cleanup();
    }

    #[test]
    fn failed_merge_checkpoint_does_not_corrupt_later_finalize_patch() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        fs::write(
            repo.join("same.txt"),
            "line 1\nshared line\nline 3\nline 4\n",
        )
        .unwrap();
        fs::write(repo.join("base.txt"), "base\n").unwrap();
        run_git(repo, &["add", "same.txt", "base.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let base_commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();
        let base_commit = base_commit.trim();
        let stage = SnapshotStage::new(repo, &format!("test-{}", uuid::Uuid::new_v4())).unwrap();

        let a_worker = create_candidate_repository(repo, "merge-corruption-a").unwrap();
        fs::write(
            a_worker.root.join("same.txt"),
            "line 1\nA edits the shared line\nline 3\nline 4\n",
        )
        .unwrap();
        let a = stage.snapshot(&a_worker.root, "a").unwrap();

        let b_worker = create_candidate_repository(repo, "merge-corruption-b").unwrap();
        fs::write(
            b_worker.root.join("same.txt"),
            "line 1\nB edits the shared line\nline 3\nline 4\n",
        )
        .unwrap();
        let b = stage.snapshot(&b_worker.root, "b").unwrap();

        let target_worker =
            create_candidate_repository_at(repo, "merge-corruption-target", &a).unwrap();
        fs::write(
            target_worker.root.join("same.txt"),
            "line 1\nT keeps A's shared-line choice\nline 3\nline 4\n",
        )
        .unwrap();
        fs::write(
            target_worker.root.join("target.txt"),
            "target content line 1\n\
             target content line 2\n\
             target content line 3\n\
             target content line 4\n\
             target content line 5\n",
        )
        .unwrap();
        let target = stage.snapshot(&target_worker.root, "target").unwrap();

        let error = stage
            .merge_checkpoint(&b, &target, "conflicted-merge")
            .expect_err("conflicting merge");
        let error = format!("{error:#}");
        assert!(error.contains("same.txt"));
        assert!(!error.contains("conflicting hunks"));
        assert_eq!(git_text(repo, &["status", "--porcelain"]).unwrap(), "");

        let expected_patch = git(
            repo,
            &[
                "diff",
                "--binary",
                "--no-ext-diff",
                base_commit,
                &target,
                "--",
                ".",
                ":(exclude).brokk/**",
                ":(exclude).bifrost/**",
            ],
        )
        .unwrap()
        .stdout;
        let finalized_patch = stage.finalize_patch(base_commit, &target).unwrap();
        assert_eq!(finalized_patch, expected_patch);

        let fresh =
            create_candidate_repository_at(repo, "merge-corruption-fresh", base_commit).unwrap();
        apply_selected_patch(&fresh.root, &finalized_patch).unwrap();
        run_git(&fresh.root, &["add", "-A"]);
        let fresh_tree = git_text(&fresh.root, &["write-tree"]).unwrap();
        let target_tree = git_text(repo, &["rev-parse", &format!("{target}^{{tree}}")]).unwrap();
        assert_eq!(fresh_tree.trim(), target_tree.trim());
        assert_text_file_eq(
            &fresh.root.join("same.txt"),
            "line 1\nT keeps A's shared-line choice\nline 3\nline 4\n",
        );
        assert_text_file_eq(
            &fresh.root.join("target.txt"),
            "target content line 1\n\
             target content line 2\n\
             target content line 3\n\
             target content line 4\n\
             target content line 5\n",
        );

        assert!(
            Command::new("git")
                .args([
                    "rev-parse",
                    "--verify",
                    &format!("refs/asgard/{}/conflicted-merge", stage.run_id)
                ])
                .current_dir(repo)
                .status()
                .unwrap()
                .code()
                .is_some_and(|code| code != 0)
        );

        remove_candidate_repository(&a_worker);
        remove_candidate_repository(&b_worker);
        remove_candidate_repository(&target_worker);
        remove_candidate_repository(&fresh);
        stage.cleanup();
    }
}
