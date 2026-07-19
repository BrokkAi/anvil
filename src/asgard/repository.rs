#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
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
            "Asgard clone prototype requires code files to be clean (dirty: {})",
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
        .spawn()?;
    child
        .stdin
        .as_mut()
        .context("git apply stdin")?
        .write_all(patch)?;
    if !child.wait()?.success() {
        bail!(
            "failed to apply selected Asgard patch in {}",
            root.display()
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
    let parent = std::env::temp_dir().join("anvil-asgard-clones");
    fs::create_dir_all(&parent)?;
    let root = parent.join(format!(
        "asgard-{}-{}",
        safe_repository_label(label),
        uuid::Uuid::new_v4()
    ));
    let status = Command::new("git")
        .args(["clone", "--shared", "--no-checkout", "--quiet", "--"])
        .arg(&repo)
        .arg(&root)
        .stdout(Stdio::null())
        .status()?;
    if !status.success() {
        remove_directory(&root, "incomplete Asgard clone");
        bail!("failed to create Asgard clone {}", root.display());
    }
    let checkout = Command::new("git")
        .args(["checkout", "--detach", "--quiet", "--force"])
        .arg(checkout_commit)
        .current_dir(&root)
        .stdout(Stdio::null())
        .status()?;
    if !checkout.success() {
        remove_directory(&root, "incomplete Asgard clone");
        bail!("failed to check out Asgard clone {}", root.display());
    }
    let repository = CandidateRepository {
        session_cwd: root.join(relative),
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
        &["reset", "--hard", "--quiet", checkout_commit],
    )?;
    git(&repository.root, &["clean", "-fdq"])?;
    git(
        &repository.root,
        &["checkout", "--detach", "--quiet", "HEAD"],
    )?;
    let branches = git_text(
        &repository.root,
        &["for-each-ref", "refs/heads", "--format=%(refname:short)"],
    )?;
    for branch in branches.lines().filter(|branch| !branch.is_empty()) {
        git(&repository.root, &["branch", "-D", branch])?;
    }
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

/// Fresh Git clones omit ignored files that a harness may have provisioned
/// after checkout. Keep small build-launcher payloads available to every lane so
/// candidates have the same validation entry point as the parent checkout.
fn seed_build_bootstrap_files(repo: &Path, clone: &Path) -> Result<()> {
    for relative in [Path::new("gradle/wrapper"), Path::new(".mvn/wrapper")] {
        let source = repo.join(relative);
        if source.is_dir() {
            copy_missing_tree(&source, &clone.join(relative))?;
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
        .spawn()?;
    child
        .stdin
        .as_mut()
        .context("write untracked paths to temporary Asgard index")?
        .write_all(&untracked)?;
    if !child.wait()?.success() {
        bail!(
            "git add -N for untracked files failed in {}",
            root.display()
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
}

#[derive(Debug)]
pub(crate) struct MergeCheckpointOutcome {
    pub(crate) commit: String,
    pub(crate) diffstat: String,
}

impl SnapshotStage {
    pub(crate) fn new(parent_root: &Path, run_id: &str) -> Result<Self> {
        let parent_root =
            PathBuf::from(git_text(parent_root, &["rev-parse", "--show-toplevel"])?.trim());
        Ok(Self {
            parent_root,
            run_id: run_id.to_string(),
        })
    }

    pub(crate) fn snapshot(
        &self,
        worker_root: &Path,
        parent_commit: &str,
        name: &str,
    ) -> Result<String> {
        let git_dir = git_text(&self.parent_root, &["rev-parse", "--absolute-git-dir"])?;
        let git_dir = PathBuf::from(git_dir.trim());
        let object_dir = git_dir.join("objects");
        let index_path =
            std::env::temp_dir().join(format!("anvil-asgard-index-{}", uuid::Uuid::new_v4()));
        let _index_guard = TemporaryIndex::new(index_path.clone());

        git_with_index_object_directory(
            worker_root,
            &index_path,
            &object_dir,
            &[
                "add",
                "-A",
                "--",
                ".",
                ":(exclude).brokk/**",
                ":(exclude).bifrost/**",
            ],
        )?;
        let tree = String::from_utf8(
            git_with_index_object_directory(
                worker_root,
                &index_path,
                &object_dir,
                &["write-tree"],
            )?
            .stdout,
        )
        .context("git write-tree output was not UTF-8")?;
        let tree = tree.trim();
        let message = format!("asgard checkpoint {name}");
        let commit = String::from_utf8(
            git_with_index_object_directory(
                worker_root,
                &index_path,
                &object_dir,
                &[
                    "-c",
                    "user.name=asgard",
                    "-c",
                    "user.email=asgard@anvil.invalid",
                    "commit-tree",
                    tree,
                    "-p",
                    parent_commit,
                    "-m",
                    &message,
                ],
            )?
            .stdout,
        )
        .context("git commit-tree output was not UTF-8")?;
        let commit = commit.trim().to_string();
        let reference = format!("refs/asgard/{}/{name}", self.run_id);
        git(&self.parent_root, &["update-ref", &reference, &commit])?;
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
        from_parent_commit: &str,
        from_commit: &str,
        onto_commit: &str,
        name: &str,
    ) -> Result<MergeCheckpointOutcome> {
        let diff = git(
            &self.parent_root,
            &[
                "diff",
                "--binary",
                "--no-ext-diff",
                from_parent_commit,
                from_commit,
                "--",
                ".",
                ":(exclude).brokk/**",
                ":(exclude).bifrost/**",
            ],
        )?
        .stdout;
        let index_path =
            std::env::temp_dir().join(format!("anvil-asgard-index-{}", uuid::Uuid::new_v4()));
        let _index_guard = TemporaryIndex::new(index_path.clone());

        git_with_index(&self.parent_root, &index_path, &["read-tree", onto_commit])?;
        git_with_index_stdin(
            &self.parent_root,
            &index_path,
            &["apply", "--cached", "--3way", "--binary", "-"],
            &diff,
        )
        .map_err(|error| anyhow::anyhow!("merge_checkpoint failed to apply patch:\n{error}"))?;
        let tree = String::from_utf8(
            git_with_index(&self.parent_root, &index_path, &["write-tree"])?.stdout,
        )
        .context("git write-tree output was not UTF-8")?;
        let tree = tree.trim();
        let message = format!("asgard checkpoint {name}");
        let commit = String::from_utf8(
            git_with_index(
                &self.parent_root,
                &index_path,
                &[
                    "-c",
                    "user.name=asgard",
                    "-c",
                    "user.email=asgard@anvil.invalid",
                    "commit-tree",
                    tree,
                    "-p",
                    onto_commit,
                    "-m",
                    &message,
                ],
            )?
            .stdout,
        )
        .context("git commit-tree output was not UTF-8")?;
        let commit = commit.trim().to_string();
        let reference = format!("refs/asgard/{}/{name}", self.run_id);
        git(&self.parent_root, &["update-ref", &reference, &commit])?;
        let diffstat = String::from_utf8(
            git(
                &self.parent_root,
                &[
                    "diff",
                    "--stat",
                    "--no-ext-diff",
                    onto_commit,
                    &commit,
                    "--",
                    ".",
                    ":(exclude).brokk/**",
                    ":(exclude).bifrost/**",
                ],
            )?
            .stdout,
        )
        .context("git diff --stat output was not UTF-8")?;
        Ok(MergeCheckpointOutcome { commit, diffstat })
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
    }
}

pub(crate) fn remove_candidate_repository(repository: &CandidateRepository) {
    remove_directory(&repository.root, "Asgard candidate repository");
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
        bail!("git {} failed in {}", args.join(" "), cwd.display());
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
        bail!("git {} failed in {}", args.join(" "), cwd.display());
    }
    Ok(output)
}

fn git_with_index_stdin(
    cwd: &Path,
    index: &Path,
    args: &[&str],
    stdin: &[u8],
) -> Result<std::process::Output, String> {
    let mut child = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .env("GIT_INDEX_FILE", index)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("git {} failed to start: {error}", args.join(" ")))?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| format!("git {} stdin unavailable", args.join(" ")))?
        .write_all(stdin)
        .map_err(|error| format!("git {} stdin failed: {error}", args.join(" ")))?;
    let output = child
        .wait_with_output()
        .map_err(|error| format!("git {} failed: {error}", args.join(" ")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(stderr.trim().to_string());
    }
    Ok(output)
}

fn git_with_index_object_directory(
    cwd: &Path,
    index: &Path,
    object_directory: &Path,
    args: &[&str],
) -> Result<std::process::Output> {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .env("GIT_INDEX_FILE", index)
        .env("GIT_OBJECT_DIRECTORY", object_directory)
        .output()?;
    if !output.status.success() {
        bail!("git {} failed in {}", args.join(" "), cwd.display());
    }
    Ok(output)
}

fn git_text(cwd: &Path, args: &[&str]) -> Result<String> {
    String::from_utf8(git(cwd, args)?.stdout).context("git output was not UTF-8")
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
    fn candidate_clone_patch_applies_tracked_deletions_to_parent() {
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
    fn snapshot_writes_parent_ref_and_shared_clone_can_checkout_commit() {
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
        let checkpoint = stage
            .snapshot(&worker.root, base_commit.trim(), "first")
            .unwrap();

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
        let base_commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();

        let worker = create_candidate_repository(repo, "snapshot-chain").unwrap();
        let stage = SnapshotStage::new(repo, &format!("test-{}", uuid::Uuid::new_v4())).unwrap();
        fs::write(worker.root.join("tracked.txt"), "snapshot a\n").unwrap();
        let first = stage
            .snapshot(&worker.root, base_commit.trim(), "a")
            .unwrap();
        fs::write(worker.root.join("tracked.txt"), "snapshot b\n").unwrap();
        let second = stage.snapshot(&worker.root, &first, "b").unwrap();

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
    fn snapshot_ignores_worker_commit_history_and_uses_harness_parent() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        fs::write(repo.join("tracked.txt"), "base\n").unwrap();
        run_git(repo, &["add", "tracked.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let base_commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();

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
        let checkpoint = stage
            .snapshot(&worker.root, base_commit.trim(), "after-worker-commit")
            .unwrap();
        let checkpoint_parent = git_text(repo, &["rev-parse", &format!("{checkpoint}^")]).unwrap();

        assert_eq!(checkpoint_parent.trim(), base_commit.trim());
        assert_ne!(checkpoint_parent.trim(), worker_commit.trim());
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
    fn recycle_repository_preserves_ignored_files_cleans_untracked_and_deletes_branches() {
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
            git_text(
                &candidate.root,
                &["for-each-ref", "refs/heads", "--format=%(refname:short)"]
            )
            .unwrap(),
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
        let checkpoint = stage
            .snapshot(&worker.root, base_commit.trim(), "patch")
            .unwrap();
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
        let base_commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();
        let stage = SnapshotStage::new(repo, &format!("test-{}", uuid::Uuid::new_v4())).unwrap();

        let from_worker = create_candidate_repository(repo, "merge-from").unwrap();
        fs::write(from_worker.root.join("from.txt"), "from\n").unwrap();
        let from = stage
            .snapshot(&from_worker.root, base_commit.trim(), "from")
            .unwrap();

        let onto_worker = create_candidate_repository(repo, "merge-onto").unwrap();
        fs::write(onto_worker.root.join("onto.txt"), "onto\n").unwrap();
        let onto = stage
            .snapshot(&onto_worker.root, base_commit.trim(), "onto")
            .unwrap();

        let merged = stage
            .merge_checkpoint(base_commit.trim(), &from, &onto, "merged")
            .unwrap();

        assert_eq!(
            git_text(
                repo,
                &["rev-parse", &format!("refs/asgard/{}/merged", stage.run_id)]
            )
            .unwrap()
            .trim(),
            merged.commit
        );
        assert_eq!(
            git_text(repo, &["show", &format!("{}:from.txt", merged.commit)])
                .unwrap()
                .replace("\r\n", "\n"),
            "from\n"
        );
        assert_eq!(
            git_text(repo, &["show", &format!("{}:onto.txt", merged.commit)])
                .unwrap()
                .replace("\r\n", "\n"),
            "onto\n"
        );
        assert!(merged.diffstat.contains("from.txt"));
        assert_eq!(
            git_text(repo, &["rev-parse", &format!("{}^", merged.commit)])
                .unwrap()
                .trim(),
            onto
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
        let base_commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();
        let stage = SnapshotStage::new(repo, &format!("test-{}", uuid::Uuid::new_v4())).unwrap();

        let from_worker = create_candidate_repository(repo, "merge-conflict-from").unwrap();
        fs::write(from_worker.root.join("same.txt"), "from\n").unwrap();
        let from = stage
            .snapshot(&from_worker.root, base_commit.trim(), "from")
            .unwrap();

        let onto_worker = create_candidate_repository(repo, "merge-conflict-onto").unwrap();
        fs::write(onto_worker.root.join("same.txt"), "onto\n").unwrap();
        let onto = stage
            .snapshot(&onto_worker.root, base_commit.trim(), "onto")
            .unwrap();

        let error = stage
            .merge_checkpoint(base_commit.trim(), &from, &onto, "conflict")
            .expect_err("conflict");

        assert!(format!("{error:#}").contains("same.txt"));
        assert!(
            Command::new("git")
                .args([
                    "rev-parse",
                    "--verify",
                    &format!("refs/asgard/{}/conflict", stage.run_id)
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
        let a = stage.snapshot(&a_worker.root, base_commit, "a").unwrap();

        let b_worker = create_candidate_repository(repo, "merge-corruption-b").unwrap();
        fs::write(
            b_worker.root.join("same.txt"),
            "line 1\nB edits the shared line\nline 3\nline 4\n",
        )
        .unwrap();
        let b = stage.snapshot(&b_worker.root, base_commit, "b").unwrap();

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
        let target = stage.snapshot(&target_worker.root, &a, "target").unwrap();

        let error = stage
            .merge_checkpoint(base_commit, &b, &target, "conflicted-merge")
            .expect_err("conflicting merge");
        assert!(format!("{error:#}").contains("same.txt"));

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
