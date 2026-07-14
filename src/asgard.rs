use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::{fs, io::Write};

use anyhow::{Context, Result, bail};

#[derive(Debug, Clone)]
pub(crate) struct Config {
    pub candidate_models: Vec<String>,
    pub supervisor_model: Option<String>,
    pub window_steps: usize,
}

static CONFIG: OnceLock<Option<Config>> = OnceLock::new();

pub(crate) fn configure(config: Option<Config>) {
    let _ = CONFIG.set(config);
}

pub(crate) fn config() -> Option<&'static Config> {
    CONFIG.get().and_then(Option::as_ref)
}

#[derive(Debug)]
pub(crate) struct Worktree {
    pub root: PathBuf,
    pub session_cwd: PathBuf,
    repo: PathBuf,
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
            "Asgard worktree prototype requires code files to be clean (dirty: {})",
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

pub(crate) fn create_worktree(cwd: &Path, label: &str) -> Result<Worktree> {
    let repo = git_text(cwd, &["rev-parse", "--show-toplevel"])?;
    let repo = PathBuf::from(repo.trim());
    let relative = cwd.strip_prefix(&repo).unwrap_or(Path::new(""));
    let parent = std::env::temp_dir().join("anvil-asgard-worktrees");
    fs::create_dir_all(&parent)?;
    let root = parent.join(format!(
        "asgard-{}-{}",
        safe_worktree_label(label),
        uuid::Uuid::new_v4()
    ));
    let status = Command::new("git")
        .args(["worktree", "add", "--detach"])
        .arg(&root)
        .arg("HEAD")
        .current_dir(&repo)
        .stdout(Stdio::null())
        .status()?;
    if !status.success() {
        bail!("failed to create Asgard worktree {}", root.display());
    }
    let worktree = Worktree {
        session_cwd: root.join(relative),
        root,
        repo,
    };
    if let Err(error) = seed_build_bootstrap_files(&worktree.repo, &worktree.root) {
        remove_worktree(&worktree);
        return Err(error);
    }
    Ok(worktree)
}

fn safe_worktree_label(label: &str) -> String {
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

/// Detached Git worktrees omit ignored files that a harness may have provisioned
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

pub(crate) fn capture_patch(root: &Path) -> Result<Vec<u8>> {
    let index_path =
        std::env::temp_dir().join(format!("anvil-asgard-index-{}", uuid::Uuid::new_v4()));
    let _index_guard = TemporaryIndex::new(index_path.clone());
    git_with_index(root, &index_path, &["read-tree", "HEAD"])?;
    add_intent_to_add_untracked(root, &index_path)?;
    Ok(git_with_index(
        root,
        &index_path,
        &[
            "diff",
            "--binary",
            "--no-ext-diff",
            "HEAD",
            "--",
            ".",
            ":(exclude).brokk/**",
            ":(exclude).bifrost/**",
        ],
    )?
    .stdout)
}

/// Captures only the candidate's changes relative to the selected state at the
/// start of the window. The selected state is represented by a patch from
/// `HEAD`; a temporary index materializes that state without changing either
/// the candidate worktree or the live checkout.
pub(crate) fn capture_patch_since(root: &Path, selected_patch: &[u8]) -> Result<Vec<u8>> {
    let index_path =
        std::env::temp_dir().join(format!("anvil-asgard-index-{}", uuid::Uuid::new_v4()));
    let _index_guard = TemporaryIndex::new(index_path.clone());

    git_with_index(root, &index_path, &["read-tree", "HEAD"])?;
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
            .context("git apply selected state to temporary index")?
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
            "--binary",
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
    let untracked = git_with_index(
        root,
        index,
        &[
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            ".",
        ],
    )?
    .stdout;
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

pub(crate) fn install_patch(root: &Path, patch: &[u8]) -> Result<()> {
    for args in [["reset", "--hard", "HEAD"], ["clean", "-fd", "--"]] {
        let status = Command::new("git")
            .args(args)
            .current_dir(root)
            .stdout(Stdio::null())
            .status()?;
        if !status.success() {
            bail!("failed to reset Asgard workspace {}", root.display());
        }
    }
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

pub(crate) fn remove_worktree(worktree: &Worktree) {
    let result = Command::new("git")
        .args(["worktree", "remove", "--force"])
        .arg(&worktree.root)
        .current_dir(&worktree.repo)
        .stdout(Stdio::null())
        .status();
    if let Err(error) = result {
        tracing::warn!(path = %worktree.root.display(), "failed to remove Asgard worktree: {error}");
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
    fn worktree_labels_are_safe_for_build_tool_paths() {
        assert_eq!(
            safe_worktree_label("0-deepseek::deepseek-v4-flash"),
            "0-deepseek--deepseek-v4-flash"
        );
        assert_eq!(safe_worktree_label("lane/model name"), "lane-model-name");
    }

    #[test]
    fn candidate_delta_is_relative_to_selected_state() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        run_git(repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("tracked.txt"), "head\n").unwrap();
        fs::write(repo.join("removed.txt"), "remove in candidate\n").unwrap();
        run_git(repo, &["add", "tracked.txt", "removed.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);

        fs::write(repo.join("tracked.txt"), "selected\n").unwrap();
        fs::write(repo.join("selected-only.txt"), "selected file\n").unwrap();
        let selected = capture_patch(repo).unwrap();

        fs::write(repo.join("tracked.txt"), "candidate\n").unwrap();
        fs::remove_file(repo.join("removed.txt")).unwrap();
        fs::write(repo.join("candidate-only.txt"), "candidate file\n").unwrap();
        let delta = capture_patch_since(repo, &selected).unwrap();
        let delta_text = String::from_utf8_lossy(&delta);
        assert!(delta_text.contains("-selected"));
        assert!(delta_text.contains("+candidate"));
        assert!(delta_text.contains("candidate-only.txt"));
        assert!(delta_text.contains("deleted file mode"));
        assert!(delta_text.contains("removed.txt"));
        assert!(!delta_text.contains("selected-only.txt"));

        run_git(repo, &["reset", "--hard", "HEAD"]);
        run_git(repo, &["clean", "-fd"]);
        install_patch(repo, &selected).unwrap();
        apply_selected_patch(repo, &delta).unwrap();
        assert_eq!(
            fs::read_to_string(repo.join("tracked.txt")).unwrap(),
            "candidate\n"
        );
        assert_eq!(
            fs::read_to_string(repo.join("selected-only.txt")).unwrap(),
            "selected file\n"
        );
        assert_eq!(
            fs::read_to_string(repo.join("candidate-only.txt")).unwrap(),
            "candidate file\n"
        );
        assert!(!repo.join("removed.txt").exists());
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
        let patch = capture_patch(repo).unwrap();
        let patch_text = String::from_utf8_lossy(&patch);
        assert!(patch_text.contains("deleted file mode"));
        assert!(patch_text.contains("added.txt"));

        run_git(repo, &["reset", "--hard", "HEAD"]);
        run_git(repo, &["clean", "-fd"]);
        apply_selected_patch(repo, &patch).unwrap();
        assert!(!repo.join("deleted.txt").exists());
        assert_eq!(
            fs::read_to_string(repo.join("added.txt")).unwrap(),
            "keep me\n"
        );
    }

    #[test]
    fn linked_worktree_patch_applies_tracked_deletions_to_parent() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        run_git(repo, &["config", "user.email", "asgard@example.invalid"]);
        run_git(repo, &["config", "user.name", "Asgard Test"]);
        fs::write(repo.join("deleted.txt"), "remove me\n").unwrap();
        run_git(repo, &["add", "deleted.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);

        let worktree = create_worktree(repo, "deletion-test").unwrap();
        fs::remove_file(worktree.root.join("deleted.txt")).unwrap();
        fs::write(worktree.root.join("added.txt"), "keep me\n").unwrap();
        let patch = capture_patch(&worktree.root).unwrap();
        assert!(String::from_utf8_lossy(&patch).contains("deleted file mode"));

        apply_selected_patch(repo, &patch).unwrap();
        remove_worktree(&worktree);
        assert!(!repo.join("deleted.txt").exists());
        assert_eq!(
            fs::read_to_string(repo.join("added.txt")).unwrap(),
            "keep me\n"
        );
    }
}
