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
    /// Reasoning effort for supervisor turns, parsed off a `model+effort`
    /// suffix on `--asgard-supervisor`. `None` means inherit the session's
    /// effort rather than sending nothing at all.
    pub supervisor_reasoning_effort: Option<String>,
}

/// Split a `model+effort` selector into its parts. The suffix is only
/// treated as an effort when it is one anvil recognizes, so a model id that
/// legitimately contains `+` is left intact.
pub(crate) fn split_model_effort(selector: &str) -> (String, Option<String>) {
    const EFFORTS: [&str; 7] = ["none", "off", "low", "medium", "high", "xhigh", "max"];
    match selector.rsplit_once('+') {
        Some((model, suffix))
            if !model.is_empty() && EFFORTS.contains(&suffix.to_ascii_lowercase().as_str()) =>
        {
            let effort = suffix.to_ascii_lowercase();
            let effort = if effort == "none" {
                "off".to_string()
            } else {
                effort
            };
            (model.to_string(), Some(effort))
        }
        _ => (selector.to_string(), None),
    }
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

/// Env gate for the delivered-patch test-file guard (Change B). Off unless
/// explicitly switched on, so ordinary development use of Asgard delivers
/// test files like any other file; benchmark harnesses set it.
const ASGARD_TEST_FILE_GUARD_ENV: &str = "ASGARD_TEST_FILE_GUARD";

/// The one place a path is judged to be a test file. Language-generic and
/// deliberately conservative -- it names the conventions a test runner
/// itself keys on, so it stays right without knowing the project.
///
/// Two shapes only:
/// - `<dir>/**` matches when any directory component of the path equals
///   `<dir>`, so a file anywhere under a `tests/` or `test/` tree counts;
/// - everything else matches the file's basename with a single `*`
///   wildcard.
const TEST_FILE_PATH_PATTERNS: &[&str] = &[
    "*_test.go",
    "test_*.py",
    "*_test.py",
    "tests/**",
    "test/**",
    "*.test.ts",
    "*.test.tsx",
    "*.test.js",
    "*.spec.ts",
    "*.spec.tsx",
    "*.spec.js",
    "*_spec.rb",
    "*Test.java",
    "*Tests.cs",
];

/// Whether the delivered-patch test-file guard runs, given the raw
/// `ASGARD_TEST_FILE_GUARD` value. Anything other than an explicit on
/// ("1", "true", "on", "yes", case-insensitive) leaves it off, so a typo in
/// a harness config fails safe toward normal delivery.
fn test_file_guard_enabled_from(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        )
    })
}

fn test_file_guard_enabled() -> bool {
    test_file_guard_enabled_from(std::env::var(ASGARD_TEST_FILE_GUARD_ENV).ok().as_deref())
}

/// Whether `path` (repo-root-relative, `/`-separated) names a test file per
/// [`TEST_FILE_PATH_PATTERNS`].
fn is_test_file_path(path: &str) -> bool {
    TEST_FILE_PATH_PATTERNS
        .iter()
        .any(|pattern| matches_test_file_pattern(path, pattern))
}

fn matches_test_file_pattern(path: &str, pattern: &str) -> bool {
    if let Some(directory) = pattern.strip_suffix("/**") {
        // Every component but the last one, which is the file's own name.
        let mut components = path.split('/').collect::<Vec<_>>();
        components.pop();
        return components.contains(&directory);
    }
    let basename = path.rsplit('/').next().unwrap_or(path);
    let Some((prefix, suffix)) = pattern.split_once('*') else {
        return basename == pattern;
    };
    basename.len() >= prefix.len() + suffix.len()
        && basename.starts_with(prefix)
        && basename.ends_with(suffix)
}

/// A test file the run touched, as reported by `git diff --name-status`.
#[derive(Debug, PartialEq, Eq)]
struct TouchedTestFile {
    path: String,
    /// True when the file already existed at the base commit (the worker
    /// edited or deleted it) rather than being created by the run.
    pre_existing: bool,
}

/// Parses `git diff --name-status -z --no-renames` output into the test
/// files it names. Every record is a status field followed by one path,
/// each NUL-terminated.
fn parse_touched_test_files(stdout: &[u8]) -> Vec<TouchedTestFile> {
    let mut fields = stdout
        .split(|byte| *byte == 0)
        .filter(|field| !field.is_empty());
    let mut touched = Vec::new();
    while let (Some(status), Some(path)) = (fields.next(), fields.next()) {
        let Ok(path) = std::str::from_utf8(path) else {
            tracing::warn!("skipping non-UTF-8 path in Asgard test-file guard");
            continue;
        };
        if !is_test_file_path(path) {
            continue;
        }
        // "A" is the only status that means the run created the file; "M",
        // "D", "T" and the rest all imply it was there at the base commit.
        let pre_existing = status.first() != Some(&b'A');
        touched.push(TouchedTestFile {
            path: path.to_string(),
            pre_existing,
        });
    }
    touched
}

/// Reports non-empty, trimmed `git status --porcelain` output for `cwd`,
/// excluding harness-owned paths (`.brokk/`, `.bifrost/`) via pathspec magic
/// so Asgard's own bookkeeping never counts as user dirt. Untracked files
/// that are gitignored are already omitted by porcelain's default (no
/// `--ignored`), so they never count either. An empty string means the
/// working tree is clean from Asgard's point of view.
pub(crate) fn working_tree_dirt(cwd: &Path) -> Result<String> {
    Ok(git_text(
        cwd,
        &[
            "status",
            "--porcelain",
            "--",
            ".",
            ":(exclude).brokk/**",
            ":(exclude).bifrost/**",
        ],
    )?
    .trim()
    .to_string())
}

pub(crate) fn parent_head_commit(cwd: &Path) -> Result<String> {
    Ok(git_text(cwd, &["rev-parse", "HEAD"])?.trim().to_string())
}

/// Applies the selected candidate delta to the live checkout without resetting
/// harness-owned state such as `.brokk/` and `.bifrost/`.
///
/// A plain `git apply` refuses when the patch creates a file that already
/// exists, untracked, in the target worktree (observed live: a task image
/// ships a generated, gitignored file such as `src/parser/grammar.ts`; the
/// delivered patch also adds it; the apply bounces and the delivery is
/// silently empty). When the straightforward apply would fail, clear any
/// path the patch touches that is untracked *and* gitignored in the target
/// worktree -- never anything git already knows about -- and retry once.
pub(crate) fn apply_selected_patch(root: &Path, patch: &[u8]) -> Result<()> {
    if patch.is_empty() {
        return Ok(());
    }
    if git_apply_check(root, patch)? {
        return run_git_apply(root, patch);
    }
    for path in git_apply_numstat_paths(root, patch)? {
        let full_path = root.join(&path);
        if full_path.symlink_metadata().is_ok() && is_untracked_and_ignored(root, &path) {
            fs::remove_file(&full_path).with_context(|| {
                format!(
                    "removing untracked, ignored file blocking patch apply: {}",
                    full_path.display()
                )
            })?;
        }
    }
    run_git_apply(root, patch)
}

fn run_git_apply(root: &Path, patch: &[u8]) -> Result<()> {
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

/// Runs `git apply --check --binary -`, reporting whether the patch would
/// apply cleanly without touching the worktree.
fn git_apply_check(root: &Path, patch: &[u8]) -> Result<bool> {
    let mut child = Command::new("git")
        .args(["apply", "--check", "--binary", "-"])
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .context("git apply --check stdin")?
        .write_all(patch)?;
    Ok(child.wait_with_output()?.status.success())
}

/// Paths the patch touches, per `git apply --numstat -z --binary -`: NUL-
/// separated records of `<added>\t<deleted>\t<path>` (a rename's record
/// carries only the destination path, never both names).
fn git_apply_numstat_paths(root: &Path, patch: &[u8]) -> Result<Vec<PathBuf>> {
    let mut child = Command::new("git")
        .args(["apply", "--numstat", "-z", "--binary", "-"])
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .context("git apply --numstat stdin")?
        .write_all(patch)?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        bail!(
            "{}",
            git_error_message(
                root,
                &["apply", "--numstat", "-z", "--binary", "-"],
                &output
            )
        );
    }
    Ok(output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|record| !record.is_empty())
        .filter_map(|record| {
            let mut fields = record.splitn(3, |byte| *byte == b'\t');
            fields.next()?;
            fields.next()?;
            fields.next().map(bytes_to_path)
        })
        .collect())
}

#[cfg(unix)]
fn bytes_to_path(path: &[u8]) -> PathBuf {
    PathBuf::from(std::ffi::OsStr::from_bytes(path))
}

#[cfg(not(unix))]
fn bytes_to_path(path: &[u8]) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(path).into_owned())
}

/// True when `path` (relative to `root`) is both untracked and covered by a
/// gitignore rule -- the only case where deleting a file the patch wants to
/// create is safe.
fn is_untracked_and_ignored(root: &Path, path: &Path) -> bool {
    let tracked = Command::new("git")
        .args(["ls-files", "--error-unmatch", "--"])
        .arg(path)
        .current_dir(root)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(true); // spawn failure: assume tracked, stay conservative
    if tracked {
        return false;
    }
    Command::new("git")
        .args(["check-ignore", "-q", "--"])
        .arg(path)
        .current_dir(root)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Resolves a commit's tree object id via `git rev-parse <commit>^{tree}`,
/// for identity checks that must compare worktree content rather than commit
/// shas (a no-op snapshot mints a new sha over an identical tree).
pub(crate) fn tree_of(repo_root: &Path, commit: &str) -> Result<String> {
    Ok(
        git_text(repo_root, &["rev-parse", &format!("{commit}^{{tree}}")])?
            .trim()
            .to_string(),
    )
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

    /// The delivered patch: `base_commit..checkpoint_commit`, minus
    /// harness-owned paths, and -- when [`test_file_guard_enabled`] -- minus
    /// every test file the run touched (see [`TEST_FILE_PATH_PATTERNS`]).
    pub(crate) fn finalize_patch(
        &self,
        base_commit: &str,
        checkpoint_commit: &str,
    ) -> Result<Vec<u8>> {
        self.finalize_patch_guarded(base_commit, checkpoint_commit, test_file_guard_enabled())
    }

    /// [`finalize_patch`](Self::finalize_patch) with the env gate already
    /// resolved, so tests can exercise both sides without mutating the
    /// process environment.
    fn finalize_patch_guarded(
        &self,
        base_commit: &str,
        checkpoint_commit: &str,
        guard_test_files: bool,
    ) -> Result<Vec<u8>> {
        let mut args: Vec<String> = [
            "diff",
            "--binary",
            "--no-ext-diff",
            base_commit,
            checkpoint_commit,
            "--",
            ".",
            ":(exclude).brokk/**",
            ":(exclude).bifrost/**",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        if guard_test_files {
            let excluded = self.touched_test_files(base_commit, checkpoint_commit)?;
            // `literal` magic so a path that happens to contain glob
            // metacharacters still excludes exactly itself.
            args.extend(
                excluded
                    .iter()
                    .map(|file| format!(":(exclude,literal){}", file.path)),
            );
            crate::trace_logging::append_trace_record(serde_json::json!({
                "type": "asgard_test_guard",
                "base_commit": base_commit,
                "checkpoint_commit": checkpoint_commit,
                "excluded_count": excluded.len(),
                "excluded": excluded
                    .iter()
                    .map(|file| serde_json::json!({
                        "path": file.path,
                        "pre_existing": file.pre_existing,
                    }))
                    .collect::<Vec<_>>(),
            }));
            if !excluded.is_empty() {
                tracing::info!(
                    count = excluded.len(),
                    "Asgard test-file guard excluded test files from the delivered patch"
                );
            }
        }
        let args = args.iter().map(String::as_str).collect::<Vec<_>>();
        Ok(git(&self.parent_root, &args)?.stdout)
    }

    /// Every test file changed between the two commits, whether the worker
    /// created it or edited one that already existed at `base_commit`.
    fn touched_test_files(
        &self,
        base_commit: &str,
        checkpoint_commit: &str,
    ) -> Result<Vec<TouchedTestFile>> {
        // `--no-renames` keeps every record a single status plus a single
        // path, so the NUL-separated stream parses without special cases.
        let output = git(
            &self.parent_root,
            &[
                "diff",
                "--name-status",
                "-z",
                "--no-renames",
                base_commit,
                checkpoint_commit,
                "--",
                ".",
                ":(exclude).brokk/**",
                ":(exclude).bifrost/**",
            ],
        )?;
        Ok(parse_touched_test_files(&output.stdout))
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

    // Same one-way normalization as an Asgard worker's tool results (see
    // `tool_loop::asgard_relativize_output`): strip `<root>/` occurrences so
    // file references are repository-relative, while a bare `root` line
    // (e.g. from `git rev-parse --show-toplevel`) still shows physical
    // reality.
    let text = crate::text::relativize_root_prefix(&text, root);

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
    fn test_file_patterns_cover_the_conventions_graders_reset() {
        for path in [
            "pkg/service_test.go",
            "tests/test_parser.py",
            "src/parser_test.py",
            "tests/fixtures/data.json",
            "test/helpers.rb",
            "deep/nested/tests/unit/thing.js",
            "web/src/App.test.ts",
            "web/src/App.test.tsx",
            "web/src/App.test.js",
            "web/src/App.spec.ts",
            "web/src/App.spec.tsx",
            "web/src/App.spec.js",
            "spec/models/user_spec.rb",
            "src/main/java/com/x/ParserTest.java",
            "src/ParserTests.cs",
        ] {
            assert!(is_test_file_path(path), "{path} should be a test file");
        }

        for path in [
            // Production code that merely mentions a test-ish word.
            "src/parser.go",
            "src/testing.go",
            "src/latest.py",
            "src/contest.rb",
            // A directory named `tests` only counts as a directory
            // component, never as the file itself or a longer name.
            "tests",
            "src/attests/thing.js",
            // `*Test.java` is a suffix on the basename, not anywhere.
            "src/main/java/com/x/TestParser.java",
            // Rust's in-file `#[cfg(test)]` convention has no path
            // signature, and `src/lib.rs` must never be treated as a test.
            "src/lib.rs",
        ] {
            assert!(!is_test_file_path(path), "{path} should not be a test file");
        }
    }

    #[test]
    fn test_file_guard_is_off_unless_explicitly_switched_on() {
        assert!(!test_file_guard_enabled_from(None), "absent means off");
        assert!(!test_file_guard_enabled_from(Some("")));
        assert!(!test_file_guard_enabled_from(Some("0")));
        assert!(!test_file_guard_enabled_from(Some("no")));
        assert!(!test_file_guard_enabled_from(Some("maybe")));
        for on in ["1", "true", "TRUE", "on", "yes", " 1 "] {
            assert!(test_file_guard_enabled_from(Some(on)), "{on} should enable");
        }
    }

    #[test]
    fn touched_test_files_parses_name_status_and_marks_pre_existing() {
        let stdout =
            b"M\0pkg/existing_test.go\0A\0pkg/new_test.go\0M\0pkg/service.go\0D\0tests/old.py\0";
        assert_eq!(
            parse_touched_test_files(stdout),
            vec![
                TouchedTestFile {
                    path: "pkg/existing_test.go".to_string(),
                    pre_existing: true,
                },
                TouchedTestFile {
                    path: "pkg/new_test.go".to_string(),
                    pre_existing: false,
                },
                TouchedTestFile {
                    path: "tests/old.py".to_string(),
                    pre_existing: true,
                },
            ]
        );
    }

    #[test]
    fn finalize_patch_guard_drops_test_files_but_keeps_production_changes() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        fs::write(repo.join("service.go"), "package main\n").unwrap();
        fs::write(repo.join("service_test.go"), "// original expectation\n").unwrap();
        run_git(repo, &["add", "service.go", "service_test.go"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let base_commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();
        let base_commit = base_commit.trim();

        let worker = create_candidate_repository(repo, "guard-worker").unwrap();
        fs::write(worker.root.join("service.go"), "package main\n// fixed\n").unwrap();
        // The carnage case: the worker rewrites a pre-existing test AND
        // authors a new one whose name a hidden test patch may also claim.
        fs::write(
            worker.root.join("service_test.go"),
            "// rewritten expectation\n",
        )
        .unwrap();
        fs::create_dir(worker.root.join("tests")).unwrap();
        fs::write(worker.root.join("tests/extra_test.go"), "// new\n").unwrap();
        fs::write(worker.root.join("helper.go"), "package main\n").unwrap();

        let stage = SnapshotStage::new(repo, &format!("test-{}", uuid::Uuid::new_v4())).unwrap();
        let checkpoint = stage.snapshot(&worker.root, "guard").unwrap();

        let unguarded = String::from_utf8(
            stage
                .finalize_patch_guarded(base_commit, &checkpoint, false)
                .unwrap(),
        )
        .unwrap();
        assert!(unguarded.contains("service_test.go"));
        assert!(unguarded.contains("tests/extra_test.go"));

        let guarded = String::from_utf8(
            stage
                .finalize_patch_guarded(base_commit, &checkpoint, true)
                .unwrap(),
        )
        .unwrap();
        assert!(
            !guarded.contains("service_test.go"),
            "an edited pre-existing test file must not be delivered:\n{guarded}"
        );
        assert!(
            !guarded.contains("tests/extra_test.go"),
            "a worker-authored test file must not be delivered:\n{guarded}"
        );
        assert!(
            guarded.contains("service.go") && guarded.contains("// fixed"),
            "production changes must survive the guard:\n{guarded}"
        );
        assert!(
            guarded.contains("helper.go"),
            "new production files must survive the guard:\n{guarded}"
        );

        // Snapshots keep the test files regardless -- only delivery filters.
        assert_eq!(
            git_text(
                repo,
                &["show", &format!("{checkpoint}:tests/extra_test.go")]
            )
            .unwrap()
            .replace("\r\n", "\n"),
            "// new\n"
        );

        remove_candidate_repository(&worker);
        stage.cleanup();
    }

    #[test]
    fn apply_selected_patch_replaces_an_untracked_ignored_file_the_patch_creates() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        // The task image ships a generated file that is gitignored (e.g. a
        // build artifact like `src/parser/grammar.ts`); a delivered patch
        // that also adds it must not bounce off a plain `git apply`.
        fs::write(repo.join(".gitignore"), "generated.txt\n").unwrap();
        run_git(repo, &["add", ".gitignore"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let base_commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();

        let worker = create_candidate_repository(repo, "collision-worker").unwrap();
        fs::write(worker.root.join("generated.txt"), "worker generated\n").unwrap();
        // The worker force-adds its own copy of the otherwise-ignored file so
        // its checkpoint (and the resulting finalize patch) carries it.
        run_git(&worker.root, &["add", "-f", "generated.txt"]);
        let stage = SnapshotStage::new(repo, &format!("test-{}", uuid::Uuid::new_v4())).unwrap();
        let checkpoint = stage.snapshot(&worker.root, "collision").unwrap();
        let patch = stage
            .finalize_patch(base_commit.trim(), &checkpoint)
            .unwrap();
        assert!(String::from_utf8_lossy(&patch).contains("generated.txt"));

        // The target checkout already has its own, different, untracked
        // copy of the ignored file -- exactly the collision `git apply`
        // refuses on its own.
        fs::write(
            repo.join("generated.txt"),
            "stale local generated content\n",
        )
        .unwrap();
        // Sanity check that this really is the collision a plain `git apply`
        // refuses, so the fallback path below is actually exercised.
        assert!(!git_apply_check(repo, &patch).unwrap());

        apply_selected_patch(repo, &patch).unwrap();
        assert_text_file_eq(&repo.join("generated.txt"), "worker generated\n");

        remove_candidate_repository(&worker);
        stage.cleanup();
    }

    #[test]
    fn apply_selected_patch_still_fails_on_a_genuine_conflict() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        fs::write(repo.join("tracked.txt"), "line1\nline2\nline3\n").unwrap();
        run_git(repo, &["add", "tracked.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let base_commit = git_text(repo, &["rev-parse", "HEAD"]).unwrap();

        let worker = create_candidate_repository(repo, "conflict-worker").unwrap();
        fs::write(worker.root.join("tracked.txt"), "line1\nCHANGED\nline3\n").unwrap();
        let stage = SnapshotStage::new(repo, &format!("test-{}", uuid::Uuid::new_v4())).unwrap();
        let checkpoint = stage.snapshot(&worker.root, "conflict").unwrap();
        let patch = stage
            .finalize_patch(base_commit.trim(), &checkpoint)
            .unwrap();

        // The target's tracked file no longer matches the patch's context at
        // all (unlike the untracked+ignored collision above, this file is
        // tracked, so the collision fallback must not touch it).
        fs::write(
            repo.join("tracked.txt"),
            "unrelated content\nno match here\n",
        )
        .unwrap();

        let result = apply_selected_patch(repo, &patch);
        assert!(
            result.is_err(),
            "a genuine content conflict must still fail"
        );
        assert_text_file_eq(
            &repo.join("tracked.txt"),
            "unrelated content\nno match here\n",
        );

        remove_candidate_repository(&worker);
        stage.cleanup();
    }

    #[test]
    fn run_supervisor_git_relativizes_its_own_scratch_root_but_preserves_bare_root() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        fs::write(repo.join("base.txt"), "base\n").unwrap();
        run_git(repo, &["add", "base.txt"]);
        run_git(repo, &["commit", "-m", "initial"]);
        let worktree = create_candidate_repository(repo, "sv-git-relativize").unwrap();

        // git's own error message names the absolute path we pass verbatim,
        // preceded by a quote rather than a path character -- a clean
        // trailing-slash occurrence of the scratch root.
        let absolute_arg = format!("HEAD:{}/base.txt", worktree.root.display());
        let missing = run_supervisor_git(&worktree.root, &["show".to_string(), absolute_arg])
            .expect("run_supervisor_git");
        assert_ne!(missing.exit_code, 0);
        assert!(
            missing.text.contains("'base.txt'"),
            "expected the scratch root stripped down to a relative reference:\n{}",
            missing.text
        );
        assert!(
            !missing.text.contains(&worktree.root.display().to_string()),
            "the supervisor's own scratch worktree root leaked into git tool output:\n{}",
            missing.text
        );

        // A bare-root spelling (no trailing separator), as `git
        // rev-parse --show-toplevel` prints, is left untouched.
        let toplevel = run_supervisor_git(
            &worktree.root,
            &["rev-parse".to_string(), "--show-toplevel".to_string()],
        )
        .expect("run_supervisor_git");
        assert_eq!(toplevel.exit_code, 0);
        assert_eq!(
            Path::new(toplevel.text.trim()),
            worktree.root.canonicalize().unwrap()
        );

        remove_candidate_repository(&worktree);
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

    #[test]
    fn working_tree_dirt_ignores_harness_owned_and_gitignored_paths() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path();
        run_git(repo, &["init"]);
        configure_test_user(repo);
        fs::write(repo.join("tracked.txt"), "base\n").unwrap();
        fs::write(repo.join(".gitignore"), "ignored.txt\n").unwrap();
        run_git(repo, &["add", "tracked.txt", ".gitignore"]);
        run_git(repo, &["commit", "-m", "initial"]);

        assert_eq!(working_tree_dirt(repo).unwrap(), "");

        // A gitignored, untracked file must not count as dirt.
        fs::write(repo.join("ignored.txt"), "generated\n").unwrap();
        assert_eq!(working_tree_dirt(repo).unwrap(), "");

        // Harness-owned paths (.brokk/, .bifrost/) must not count as dirt,
        // whether tracked or untracked.
        fs::create_dir_all(repo.join(".brokk")).unwrap();
        fs::write(repo.join(".brokk/state.json"), "{}\n").unwrap();
        fs::create_dir_all(repo.join(".bifrost")).unwrap();
        fs::write(repo.join(".bifrost/index.bin"), "x\n").unwrap();
        assert_eq!(working_tree_dirt(repo).unwrap(), "");

        // A genuine uncommitted change to a tracked file must count as dirt.
        // (The leading status-column space is stripped by the outer `trim()`
        // when it is also the first character of the whole string.)
        fs::write(repo.join("tracked.txt"), "base\nedited\n").unwrap();
        let dirt = working_tree_dirt(repo).unwrap();
        assert_eq!(dirt, "M tracked.txt");
    }

    #[test]
    fn split_model_effort_separates_a_recognized_suffix() {
        assert_eq!(
            split_model_effort("bedrock::openai.gpt-5.6-sol+high"),
            (
                "bedrock::openai.gpt-5.6-sol".to_string(),
                Some("high".to_string())
            )
        );
        assert_eq!(
            split_model_effort("bedrock::us.anthropic.claude-fable-5+XHigh"),
            (
                "bedrock::us.anthropic.claude-fable-5".to_string(),
                Some("xhigh".to_string())
            )
        );
        // `none` is spelled `off` internally.
        assert_eq!(
            split_model_effort("m+none"),
            ("m".to_string(), Some("off".to_string()))
        );
    }

    #[test]
    fn split_model_effort_leaves_unsuffixed_and_plus_bearing_ids_alone() {
        assert_eq!(
            split_model_effort("deepseek::deepseek-v4-pro"),
            ("deepseek::deepseek-v4-pro".to_string(), None)
        );
        // A `+` that is not a known effort must not be eaten -- otherwise a
        // legitimate model id would be silently truncated.
        assert_eq!(
            split_model_effort("vendor::model+turbo"),
            ("vendor::model+turbo".to_string(), None)
        );
        assert_eq!(split_model_effort("+high"), ("+high".to_string(), None));
    }
}
