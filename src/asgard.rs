use std::collections::HashMap;
use std::ffi::OsString;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::Instant;
use std::{fs, io::Write, thread};

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
) -> Result<RepositorySyncStats> {
    let selected = repositories
        .get(selected_index)
        .context("selected Asgard repository index is out of range")?;
    let started = Instant::now();
    let stats = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(repositories.len().saturating_sub(1));
        for (index, repository) in repositories.iter().enumerate() {
            if index != selected_index {
                let source = &selected.root;
                let destination = &repository.root;
                handles.push(scope.spawn(move || {
                    synchronize_directory_contents(source, destination).with_context(|| {
                        format!(
                            "synchronize selected Asgard repository {} to {}",
                            source.display(),
                            destination.display()
                        )
                    })
                }));
            }
        }

        let mut total = RepositorySyncStats::default();
        for handle in handles {
            let candidate = handle
                .join()
                .map_err(|_| anyhow::anyhow!("Asgard repository sync thread panicked"))??;
            total.add(candidate);
            total.destinations += 1;
        }
        Ok::<_, anyhow::Error>(total)
    })?;
    tracing::info!(
        elapsed_ms = started.elapsed().as_millis(),
        destinations = stats.destinations,
        files_copied = stats.files_copied,
        bytes_copied = stats.bytes_copied,
        entries_removed = stats.entries_removed,
        files_unchanged = stats.files_unchanged,
        metadata_updated = stats.metadata_updated,
        "synchronized Asgard candidate repositories"
    );
    Ok(stats)
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RepositorySyncStats {
    pub destinations: usize,
    pub files_copied: u64,
    pub bytes_copied: u64,
    pub entries_removed: u64,
    pub files_unchanged: u64,
    pub metadata_updated: u64,
}

impl RepositorySyncStats {
    fn add(&mut self, other: Self) {
        self.files_copied += other.files_copied;
        self.bytes_copied += other.bytes_copied;
        self.entries_removed += other.entries_removed;
        self.files_unchanged += other.files_unchanged;
        self.metadata_updated += other.metadata_updated;
    }
}

fn synchronize_directory_contents(
    source: &Path,
    destination: &Path,
) -> Result<RepositorySyncStats> {
    let mut stats = RepositorySyncStats::default();
    let mut destination_entries: HashMap<OsString, fs::DirEntry> = fs::read_dir(destination)
        .with_context(|| format!("read Asgard repository {}", destination.display()))?
        .map(|entry| entry.map(|entry| (entry.file_name(), entry)))
        .collect::<std::io::Result<_>>()?;

    for entry in fs::read_dir(source)
        .with_context(|| format!("read Asgard repository {}", source.display()))?
    {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let source_metadata = fs::symlink_metadata(&source_path)?;
        let destination_entry = destination_entries.remove(&entry.file_name());
        let destination_metadata = destination_entry
            .as_ref()
            .map(|entry| fs::symlink_metadata(entry.path()))
            .transpose()?;

        if destination_metadata
            .as_ref()
            .is_some_and(|metadata| !same_entry_type(&source_metadata, metadata))
        {
            remove_entry(&destination_path, destination_metadata.as_ref().unwrap())?;
            stats.entries_removed += 1;
        }

        if source_metadata.is_dir() {
            if destination_metadata
                .as_ref()
                .is_none_or(|metadata| !metadata.is_dir())
            {
                fs::create_dir(&destination_path).with_context(|| {
                    format!(
                        "create Asgard repository directory {}",
                        destination_path.display()
                    )
                })?;
            }
            stats.add(synchronize_directory_contents(
                &source_path,
                &destination_path,
            )?);
            if !same_permissions(&source_metadata, &fs::metadata(&destination_path)?) {
                fs::set_permissions(&destination_path, source_metadata.permissions())?;
                stats.metadata_updated += 1;
            }
        } else if source_metadata.file_type().is_symlink() {
            let same_target = destination_metadata
                .as_ref()
                .is_some_and(|metadata| metadata.file_type().is_symlink())
                && fs::read_link(&source_path)? == fs::read_link(&destination_path)?;
            if !same_target {
                if let Some(metadata) = destination_metadata.as_ref()
                    && same_entry_type(&source_metadata, metadata)
                {
                    remove_entry(&destination_path, metadata)?;
                    stats.entries_removed += 1;
                }
                copy_symlink(&source_path, &destination_path)?;
                stats.files_copied += 1;
            } else {
                stats.files_unchanged += 1;
            }
        } else if source_metadata.is_file() {
            let destination_metadata = destination_metadata
                .as_ref()
                .filter(|metadata| metadata.is_file());
            if destination_metadata
                .is_some_and(|metadata| same_file_contents(&source_metadata, metadata))
            {
                stats.files_unchanged += 1;
                let destination_metadata = destination_metadata.unwrap();
                if !same_permissions(&source_metadata, destination_metadata) {
                    fs::set_permissions(&destination_path, source_metadata.permissions())?;
                    stats.metadata_updated += 1;
                }
            } else {
                if let Some(metadata) = destination_metadata {
                    remove_entry(&destination_path, metadata)?;
                    stats.entries_removed += 1;
                }
                copy_file(&source_path, &destination_path, &source_metadata)?;
                stats.files_copied += 1;
                stats.bytes_copied += source_metadata.len();
            }
        } else {
            bail!(
                "unsupported special file in Asgard repository: {}",
                source_path.display()
            );
        }
    }

    for (_, entry) in destination_entries {
        let metadata = fs::symlink_metadata(entry.path())?;
        remove_entry(&entry.path(), &metadata)?;
        stats.entries_removed += 1;
    }
    Ok(stats)
}

fn same_entry_type(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    left.is_dir() == right.is_dir()
        && left.is_file() == right.is_file()
        && left.file_type().is_symlink() == right.file_type().is_symlink()
}

fn same_file_contents(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    left.len() == right.len() && left.modified().ok() == right.modified().ok()
}

#[cfg(unix)]
fn same_permissions(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;

    left.mode() == right.mode()
}

#[cfg(not(unix))]
fn same_permissions(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    left.permissions().readonly() == right.permissions().readonly()
}

fn copy_file(source: &Path, destination: &Path, source_metadata: &fs::Metadata) -> Result<()> {
    fs::copy(source, destination).with_context(|| {
        format!(
            "copy Asgard repository file {} to {}",
            source.display(),
            destination.display()
        )
    })?;
    fs::set_permissions(destination, source_metadata.permissions())?;
    fs::File::open(destination)?
        .set_times(fs::FileTimes::new().set_modified(source_metadata.modified()?))?;
    Ok(())
}

fn remove_entry(path: &Path, metadata: &fs::Metadata) -> Result<()> {
    if metadata.is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    }
    .with_context(|| format!("remove stale Asgard repository entry {}", path.display()))
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

    #[test]
    fn shadow_survivor_repositories_stay_isolated_until_final_sync() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        run_git(repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("tracked.txt"), "base\n").unwrap();
        run_git(repo, &["add", "tracked.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);

        let repositories = (0..3)
            .map(|lane| create_candidate_repository(repo, &format!("shadow-{lane}")))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        for (lane, repository) in repositories.iter().enumerate() {
            fs::write(
                repository.root.join("tracked.txt"),
                format!("isolated-{lane}{}\n", "x".repeat(lane)),
            )
            .unwrap();
        }
        for (lane, repository) in repositories.iter().enumerate() {
            assert_eq!(
                fs::read_to_string(repository.root.join("tracked.txt")).unwrap(),
                format!("isolated-{lane}{}\n", "x".repeat(lane))
            );
        }

        synchronize_candidate_repositories(&repositories, 2).unwrap();
        for repository in &repositories {
            assert_eq!(
                fs::read_to_string(repository.root.join("tracked.txt")).unwrap(),
                "isolated-2xx\n"
            );
            remove_candidate_repository(repository);
        }
    }

    #[cfg(unix)]
    #[test]
    fn repository_sync_skips_matching_files_and_applies_metadata_deltas() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        use std::time::{Duration, SystemTime};

        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        let destination = temp.path().join("destination");
        fs::create_dir_all(source.join("nested")).unwrap();
        fs::create_dir_all(destination.join("nested")).unwrap();

        let set_mtime = |path: &Path, seconds| {
            fs::File::open(path)
                .unwrap()
                .set_times(
                    fs::FileTimes::new()
                        .set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(seconds)),
                )
                .unwrap();
        };

        fs::write(source.join("unchanged.txt"), "same\n").unwrap();
        fs::write(destination.join("unchanged.txt"), "same\n").unwrap();
        set_mtime(&source.join("unchanged.txt"), 10);
        set_mtime(&destination.join("unchanged.txt"), 10);
        let unchanged_inode = fs::metadata(destination.join("unchanged.txt"))
            .unwrap()
            .ino();

        fs::write(source.join("changed.txt"), "new!\n").unwrap();
        fs::write(destination.join("changed.txt"), "old!\n").unwrap();
        set_mtime(&source.join("changed.txt"), 20);
        set_mtime(&destination.join("changed.txt"), 10);

        fs::write(source.join("executable.sh"), "echo ok\n").unwrap();
        fs::write(destination.join("executable.sh"), "echo ok\n").unwrap();
        set_mtime(&source.join("executable.sh"), 10);
        set_mtime(&destination.join("executable.sh"), 10);
        fs::set_permissions(
            source.join("executable.sh"),
            fs::Permissions::from_mode(0o755),
        )
        .unwrap();
        fs::set_permissions(
            destination.join("executable.sh"),
            fs::Permissions::from_mode(0o644),
        )
        .unwrap();

        fs::write(source.join("was-directory"), "now a file\n").unwrap();
        fs::create_dir(destination.join("was-directory")).unwrap();
        fs::write(destination.join("was-directory/stale.txt"), "stale\n").unwrap();
        fs::write(destination.join("nested/remove-me.txt"), "stale\n").unwrap();

        std::os::unix::fs::symlink("new-target", source.join("link")).unwrap();
        std::os::unix::fs::symlink("old-target", destination.join("link")).unwrap();

        let stats = synchronize_directory_contents(&source, &destination).unwrap();

        assert_eq!(
            fs::metadata(destination.join("unchanged.txt"))
                .unwrap()
                .ino(),
            unchanged_inode,
            "matching metadata should avoid replacing the destination file"
        );
        assert_text_file_eq(&destination.join("changed.txt"), "new!\n");
        assert_eq!(
            fs::metadata(destination.join("changed.txt"))
                .unwrap()
                .modified()
                .unwrap(),
            fs::metadata(source.join("changed.txt"))
                .unwrap()
                .modified()
                .unwrap()
        );
        assert_eq!(
            fs::metadata(destination.join("executable.sh"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o755
        );
        assert_text_file_eq(&destination.join("was-directory"), "now a file\n");
        assert_eq!(
            fs::read_link(destination.join("link")).unwrap(),
            Path::new("new-target")
        );
        assert!(!destination.join("nested/remove-me.txt").exists());
        assert!(stats.files_unchanged >= 2);
        assert!(stats.files_copied >= 3);
        assert!(stats.entries_removed >= 3);
        assert_eq!(stats.metadata_updated, 1);
    }
}
