use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

const WORKSPACE_DELTA_MAX_TEXT_BYTES: u64 = 1_048_576;

#[derive(Debug, Clone, PartialEq, Eq)]
enum WorkspaceTextState {
    Present(String),
    Absent,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct WorkspaceDeltaTracker {
    root: PathBuf,
    pre_turn: HashMap<PathBuf, WorkspaceTextState>,
}

impl WorkspaceDeltaTracker {
    pub(crate) async fn snapshot(root: &Path) -> Self {
        let root = root.to_path_buf();
        let mut pre_turn = HashMap::new();
        for rel_path in git_status_paths(&root).await.unwrap_or_default() {
            let path = root.join(&rel_path);
            if let Some(state) = read_workspace_text_state(&path).await {
                pre_turn.insert(rel_path, state);
            }
        }
        Self { root, pre_turn }
    }

    async fn changed_files(&self) -> Vec<PathBuf> {
        let post_paths = git_status_paths(&self.root).await.unwrap_or_default();
        let mut candidates = BTreeSet::new();
        candidates.extend(self.pre_turn.keys().cloned());
        candidates.extend(post_paths);

        let mut changed = Vec::new();
        for rel_path in candidates {
            let path = self.root.join(&rel_path);
            let Some(new_state) = read_workspace_text_state(&path).await else {
                continue;
            };
            let old_state = match self.pre_turn.get(&rel_path) {
                Some(state) => state.clone(),
                None => read_head_text_state(&self.root, &rel_path)
                    .await
                    .unwrap_or(WorkspaceTextState::Absent),
            };
            if old_state != new_state {
                changed.push(rel_path);
            }
        }
        changed
    }
}

pub(crate) async fn workspace_delta_for_turn(
    root: &Path,
    tracker: WorkspaceDeltaTracker,
) -> crate::host_notice::WorkspaceDelta {
    let paths = tracker
        .changed_files()
        .await
        .into_iter()
        .map(|path| path_relative_to(root, &path));
    crate::host_notice::WorkspaceDelta::from_paths(paths)
}

fn path_relative_to(_root: &Path, rel_path: &Path) -> PathBuf {
    rel_path.to_path_buf()
}

async fn git_status_paths(root: &Path) -> Option<BTreeSet<PathBuf>> {
    let output = tokio::process::Command::new("git")
        .arg("-c")
        .arg("core.fsmonitor=false")
        .arg("-C")
        .arg(root)
        .args(["status", "--porcelain=v1", "-z", "--untracked-files=all"])
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(parse_git_status_paths(&output.stdout))
}

fn parse_git_status_paths(output: &[u8]) -> BTreeSet<PathBuf> {
    let mut paths = BTreeSet::new();
    let mut entries = output
        .split(|byte| *byte == 0)
        .filter(|entry| !entry.is_empty());
    while let Some(entry) = entries.next() {
        if entry.len() < 4 {
            continue;
        }
        let status = &entry[..2];
        let path = &entry[3..];
        if !path.is_empty() {
            paths.insert(PathBuf::from(String::from_utf8_lossy(path).into_owned()));
        }
        if matches!(status.first(), Some(b'R' | b'C')) || matches!(status.get(1), Some(b'R' | b'C'))
        {
            let _ = entries.next();
        }
    }
    paths
}

async fn read_workspace_text_state(path: &Path) -> Option<WorkspaceTextState> {
    let metadata = match tokio::fs::metadata(path).await {
        Ok(metadata) => metadata,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Some(WorkspaceTextState::Absent);
        }
        Err(_) => return None,
    };
    if !metadata.is_file() || metadata.len() > WORKSPACE_DELTA_MAX_TEXT_BYTES {
        return None;
    }
    tokio::fs::read_to_string(path)
        .await
        .ok()
        .map(WorkspaceTextState::Present)
}

async fn read_head_text_state(root: &Path, rel_path: &Path) -> Option<WorkspaceTextState> {
    let output = tokio::process::Command::new("git")
        .arg("-c")
        .arg("core.fsmonitor=false")
        .arg("-C")
        .arg(root)
        .arg("show")
        .arg(git_head_object_spec(rel_path)?)
        .output()
        .await
        .ok()?;
    if output.status.success() {
        String::from_utf8(output.stdout)
            .ok()
            .map(WorkspaceTextState::Present)
    } else {
        Some(WorkspaceTextState::Absent)
    }
}

fn git_head_object_spec(path: &Path) -> Option<String> {
    path.to_str().map(|path| format!("HEAD:{path}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_git(cwd: &Path, args: &[&str]) {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(cwd)
            .args(args)
            .output()
            .expect("run git");
        assert!(
            output.status.success(),
            "git {:?} failed\nstdout:\n{}\nstderr:\n{}",
            args,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn init_git_repo(cwd: &Path) {
        run_git(cwd, &["init"]);
        run_git(cwd, &["config", "user.email", "test@example.com"]);
        run_git(cwd, &["config", "user.name", "Test User"]);
    }

    #[tokio::test]
    async fn workspace_delta_tracker_reports_shell_written_tracked_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        init_git_repo(temp.path());
        let path = temp.path().join("notes.txt");
        tokio::fs::write(&path, "before\n")
            .await
            .expect("seed file");
        run_git(temp.path(), &["add", "notes.txt"]);
        run_git(temp.path(), &["commit", "-m", "seed"]);

        let tracker = WorkspaceDeltaTracker::snapshot(temp.path()).await;
        tokio::fs::write(&path, "after\n").await.expect("edit file");

        assert_eq!(
            tracker.changed_files().await,
            vec![PathBuf::from("notes.txt")]
        );
    }

    #[tokio::test]
    async fn workspace_delta_tracker_uses_dirty_pre_turn_baseline() {
        let temp = tempfile::tempdir().expect("tempdir");
        init_git_repo(temp.path());
        let path = temp.path().join("notes.txt");
        tokio::fs::write(&path, "committed\n")
            .await
            .expect("seed file");
        run_git(temp.path(), &["add", "notes.txt"]);
        run_git(temp.path(), &["commit", "-m", "seed"]);

        tokio::fs::write(&path, "dirty before\n")
            .await
            .expect("dirty file");
        let tracker = WorkspaceDeltaTracker::snapshot(temp.path()).await;
        tokio::fs::write(&path, "after\n").await.expect("edit file");

        assert_eq!(
            tracker.changed_files().await,
            vec![PathBuf::from("notes.txt")]
        );
    }
}
