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
    pub max_windows: usize,
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
    let root = parent.join(format!("asgard-{label}-{}", uuid::Uuid::new_v4()));
    let status = Command::new("git")
        .args(["worktree", "add", "--detach"])
        .arg(&root)
        .arg("HEAD")
        .current_dir(&repo)
        .status()?;
    if !status.success() {
        bail!("failed to create Asgard worktree {}", root.display());
    }
    Ok(Worktree {
        session_cwd: root.join(relative),
        root,
        repo,
    })
}

pub(crate) fn capture_patch(root: &Path) -> Result<Vec<u8>> {
    let status = Command::new("git")
        .args(["add", "-N", "--all"])
        .current_dir(root)
        .status()?;
    if !status.success() {
        bail!("git add -N failed in {}", root.display());
    }
    Ok(git(
        root,
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

pub(crate) fn install_patch(root: &Path, patch: &[u8]) -> Result<()> {
    for args in [["reset", "--hard", "HEAD"], ["clean", "-fd", "--"]] {
        let status = Command::new("git").args(args).current_dir(root).status()?;
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

fn git_text(cwd: &Path, args: &[&str]) -> Result<String> {
    String::from_utf8(git(cwd, args)?.stdout).context("git output was not UTF-8")
}
