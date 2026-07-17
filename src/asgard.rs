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
    pub base_commit: String,
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

pub(crate) fn create_candidate_repository(cwd: &Path, label: &str) -> Result<CandidateRepository> {
    let repo = git_text(cwd, &["rev-parse", "--show-toplevel"])?;
    let repo = PathBuf::from(repo.trim());
    let base_commit = git_text(&repo, &["rev-parse", "HEAD"])?;
    let base_commit = base_commit.trim().to_string();
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
        .arg(&base_commit)
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
        base_commit,
    };
    if let Err(error) = seed_build_bootstrap_files(&repo, &repository.root) {
        remove_candidate_repository(&repository);
        return Err(error);
    }
    Ok(repository)
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

pub(crate) fn capture_patch(root: &Path, base_commit: &str) -> Result<Vec<u8>> {
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

/// Captures only the candidate changes made since the selected state at the
/// start of the current window. The selected state is a cumulative patch from
/// `base_commit`; a temporary index materializes it without changing the
/// candidate checkout or its real index.
pub(crate) fn capture_patch_since(
    root: &Path,
    base_commit: &str,
    selected_patch: &[u8],
) -> Result<Vec<u8>> {
    let index_path =
        std::env::temp_dir().join(format!("anvil-asgard-index-{}", uuid::Uuid::new_v4()));
    let _index_guard = TemporaryIndex::new(index_path.clone());

    git_with_index(root, &index_path, &["read-tree", base_commit])?;
    if !selected_patch.is_empty() {
        let mut child = Command::new("git")
            .args(["apply", "--cached", "--binary", "-"])
            .current_dir(root)
            .env("GIT_INDEX_FILE", &index_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()?;
        child
            .stdin
            .as_mut()
            .context("git apply selected Asgard state to temporary index")?
            .write_all(selected_patch)?;
        if !child.wait()?.success() {
            bail!(
                "failed to materialize selected Asgard state in {}",
                root.display()
            );
        }
    }
    add_intent_to_add_untracked(root, &index_path)?;
    Ok(git_with_index(
        root,
        &index_path,
        &[
            "diff",
            "--no-ext-diff",
            "--",
            ".",
            ":(exclude).brokk/**",
            ":(exclude).bifrost/**",
        ],
    )?
    .stdout)
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
        .filter(|path| {
            root.join(std::ffi::OsStr::from_bytes(path))
                .symlink_metadata()
                .is_ok()
        })
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
        if let Err(error) = fs::remove_file(&self.path)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(path = %self.path.display(), "failed to remove temporary Asgard index: {error}");
        }
    }
}

pub(crate) fn synchronize_candidate_repositories(
    repositories: &[CandidateRepository],
    selected_index: usize,
) -> Result<()> {
    let selected = repositories
        .get(selected_index)
        .context("selected Asgard repository index is out of range")?;
    for (index, repository) in repositories.iter().enumerate() {
        if index != selected_index {
            replace_repository_contents(&selected.root, &repository.root)?;
        }
    }
    Ok(())
}

fn replace_repository_contents(source: &Path, destination: &Path) -> Result<()> {
    let parent = destination
        .parent()
        .context("Asgard candidate repository has no parent directory")?;
    let staging = parent.join(format!(".asgard-sync-{}", uuid::Uuid::new_v4()));
    let _staging_guard = TemporaryDirectory::new(staging.clone());
    copy_repository(source, &staging)?;

    clear_directory(destination)?;
    for entry in fs::read_dir(&staging)
        .with_context(|| format!("read staged Asgard snapshot {}", staging.display()))?
    {
        let entry = entry?;
        fs::rename(entry.path(), destination.join(entry.file_name())).with_context(|| {
            format!(
                "install staged Asgard snapshot from {} into {}",
                staging.display(),
                destination.display()
            )
        })?;
    }
    Ok(())
}

fn copy_repository(source: &Path, destination: &Path) -> Result<()> {
    if try_reflink_copy_repository(source, destination)? {
        return Ok(());
    }
    remove_directory(destination, "failed reflink staging directory");
    fs::create_dir_all(destination).with_context(|| {
        format!(
            "create Asgard repository copy destination {}",
            destination.display()
        )
    })?;
    copy_directory_contents(source, destination)
}

#[cfg(unix)]
fn try_reflink_copy_repository(source: &Path, destination: &Path) -> Result<bool> {
    let source_contents = source.join(".");
    let status = Command::new("cp")
        .args(["-a", "--reflink=auto", "--"])
        .arg(&source_contents)
        .arg(destination)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    Ok(status.success())
}

#[cfg(not(unix))]
fn try_reflink_copy_repository(_source: &Path, _destination: &Path) -> Result<bool> {
    Ok(false)
}

fn copy_directory_contents(source: &Path, destination: &Path) -> Result<()> {
    for entry in fs::read_dir(source)
        .with_context(|| format!("read Asgard repository {}", source.display()))?
    {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            fs::create_dir_all(&destination_path).with_context(|| {
                format!(
                    "create Asgard repository directory {}",
                    destination_path.display()
                )
            })?;
            copy_directory_contents(&source_path, &destination_path)?;
        } else if file_type.is_symlink() {
            copy_symlink(&source_path, &destination_path)?;
        } else if file_type.is_file() {
            fs::copy(&source_path, &destination_path).with_context(|| {
                format!(
                    "copy Asgard repository file {} to {}",
                    source_path.display(),
                    destination_path.display()
                )
            })?;
        }
    }
    Ok(())
}

#[cfg(unix)]
fn copy_symlink(source: &Path, destination: &Path) -> Result<()> {
    std::os::unix::fs::symlink(fs::read_link(source)?, destination).with_context(|| {
        format!(
            "copy Asgard repository symlink {} to {}",
            source.display(),
            destination.display()
        )
    })
}

#[cfg(windows)]
fn copy_symlink(source: &Path, destination: &Path) -> Result<()> {
    let target = fs::read_link(source)?;
    let result = if source.metadata().is_ok_and(|metadata| metadata.is_dir()) {
        std::os::windows::fs::symlink_dir(&target, destination)
    } else {
        std::os::windows::fs::symlink_file(&target, destination)
    };
    result.with_context(|| {
        format!(
            "copy Asgard repository symlink {} to {}",
            source.display(),
            destination.display()
        )
    })
}

fn clear_directory(path: &Path) -> Result<()> {
    for entry in
        fs::read_dir(path).with_context(|| format!("read Asgard repository {}", path.display()))?
    {
        let entry = entry?;
        let entry_path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() && !file_type.is_symlink() {
            fs::remove_dir_all(&entry_path)?;
        } else {
            fs::remove_file(&entry_path)?;
        }
    }
    Ok(())
}

struct TemporaryDirectory {
    path: PathBuf,
}

impl TemporaryDirectory {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        remove_directory(&self.path, "temporary Asgard directory");
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
    fn captured_window_patch_excludes_the_selected_baseline() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        run_git(repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("tracked.txt"), "base\n").unwrap();
        fs::write(repo.join("removed.txt"), "remove later\n").unwrap();
        run_git(repo, &["add", "tracked.txt", "removed.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let base_commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();

        fs::write(repo.join("tracked.txt"), "selected\n").unwrap();
        fs::write(repo.join("selected-only.txt"), "selected file\n").unwrap();
        let selected = capture_patch(repo, base_commit.trim()).unwrap();

        fs::write(repo.join("tracked.txt"), "candidate\n").unwrap();
        fs::remove_file(repo.join("removed.txt")).unwrap();
        fs::write(repo.join("candidate-only.txt"), "candidate file\n").unwrap();
        let delta = capture_patch_since(repo, base_commit.trim(), &selected).unwrap();
        let delta_text = String::from_utf8_lossy(&delta);

        assert!(delta_text.contains("-selected"));
        assert!(delta_text.contains("+candidate"));
        assert!(delta_text.contains("candidate-only.txt"));
        assert!(delta_text.contains("deleted file mode"));
        assert!(delta_text.contains("removed.txt"));
        assert!(!delta_text.contains("selected-only.txt"));
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

        let candidate = create_candidate_repository(repo, "deletion-test").unwrap();
        fs::remove_file(candidate.root.join("deleted.txt")).unwrap();
        fs::write(candidate.root.join("added.txt"), "keep me\n").unwrap();
        let patch = capture_patch(&candidate.root, &candidate.base_commit).unwrap();
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

        let patch = capture_patch(&candidate.root, &candidate.base_commit).unwrap();
        apply_selected_patch(repo, &patch).unwrap();
        remove_candidate_repository(&candidate);

        assert_text_file_eq(&repo.join("tracked.txt"), "committed solution\n");
        assert_text_file_eq(&repo.join("added.txt"), "committed addition\n");
    }

    #[test]
    fn repository_sync_copies_branch_index_worktree_and_untracked_state() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        run_git(repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join(".gitignore"), "ignored.log\n").unwrap();
        fs::write(repo.join("tracked.txt"), "base\n").unwrap();
        run_git(repo, &["add", ".gitignore", "tracked.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);

        let selected = create_candidate_repository(repo, "selected").unwrap();
        let losing = create_candidate_repository(repo, "losing").unwrap();
        for candidate in [&selected, &losing] {
            run_git(
                &candidate.root,
                &["config", "user.email", "candidate@example.invalid"],
            );
            run_git(&candidate.root, &["config", "user.name", "Candidate"]);
            // This succeeds in both independent clones. Linked worktrees cannot
            // safely give every lane the same checked-out branch.
            run_git(&candidate.root, &["checkout", "-b", "solution"]);
        }

        fs::write(selected.root.join("tracked.txt"), "committed\n").unwrap();
        run_git(&selected.root, &["add", "tracked.txt"]);
        run_git(&selected.root, &["commit", "-m", "selected commit"]);
        fs::write(selected.root.join("staged.txt"), "index version\n").unwrap();
        run_git(&selected.root, &["add", "staged.txt"]);
        fs::write(
            selected.root.join("staged.txt"),
            "index version\nworktree version\n",
        )
        .unwrap();
        fs::write(selected.root.join("untracked.txt"), "untracked\n").unwrap();
        fs::write(selected.root.join("ignored.log"), "ignored state\n").unwrap();
        run_git(&selected.root, &["config", "asgard.selected", "winner"]);

        fs::write(losing.root.join("tracked.txt"), "losing state\n").unwrap();
        fs::write(losing.root.join("loser-only.txt"), "remove me\n").unwrap();

        let selected_root = selected.root.clone();
        let losing_root = losing.root.clone();
        let repositories = vec![selected, losing];
        synchronize_candidate_repositories(&repositories, 0).unwrap();

        for args in [
            &["symbolic-ref", "--short", "HEAD"][..],
            &["rev-parse", "HEAD"],
            &["status", "--porcelain", "--ignored"],
            &["diff", "--cached", "--binary"],
            &["diff", "--binary"],
        ] {
            assert_eq!(
                git(&selected_root, args).unwrap().stdout,
                git(&losing_root, args).unwrap().stdout
            );
        }
        assert_eq!(
            git_text(&losing_root, &["config", "--get", "asgard.selected"])
                .unwrap()
                .trim(),
            "winner"
        );
        assert_eq!(
            fs::read_to_string(losing_root.join("ignored.log")).unwrap(),
            "ignored state\n"
        );
        assert!(!losing_root.join("loser-only.txt").exists());

        for repository in &repositories {
            remove_candidate_repository(repository);
        }
    }
}
