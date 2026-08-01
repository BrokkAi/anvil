use super::sandbox::{self, ENV_WHITELIST, SandboxPolicy};
use super::{ToolResult, ToolStatus};
use std::borrow::Cow;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

const MAX_OUTPUT_BYTES: usize = 100_000; // 100KB
/// Wall-clock budget for one `run_shell_command` call when the model names
/// neither `timeout_seconds` nor the legacy millisecond `timeout`.
pub(super) const DEFAULT_TIMEOUT_SECONDS: u64 = 120;
/// Floor for a model-supplied `timeout_seconds`. A model that asks for a
/// couple of seconds is almost always confusing units or guessing; killing
/// its command at the requested value teaches it that verification is
/// impossible here, so we raise the budget and say so in the output.
/// Deliberately NOT applied to the legacy millisecond field, whose exact
/// semantics replayed traces depend on.
pub(super) const MIN_TIMEOUT_SECONDS: u64 = 10;
/// Ceiling for every timeout, whichever field requested it. Deployments can
/// lower it via `ANVIL_SHELL_TIMEOUT_CAP_SECONDS`; the model is never told
/// about that override, it just sees the clamp notice if one fires.
pub(super) const MAX_TIMEOUT_SECONDS: u64 = 3600;
/// Deployment-level override for [`MAX_TIMEOUT_SECONDS`]. Read from the
/// agent's own environment (not the sandboxed child's), so a command cannot
/// widen its own budget by exporting this.
const TIMEOUT_CAP_ENV: &str = "ANVIL_SHELL_TIMEOUT_CAP_SECONDS";

/// Where raw captures land when the minimizer rewrites a command's output,
/// relative to the session cwd. Must stay under the session cwd so the
/// reference spliced into tool results resolves through
/// `safe_resolve_in_roots` when the model passes it back to `read_file`.
const SPILL_DIR_RELATIVE: &str = ".brokk/shell-output";
const STALE_SPILL_FILE_AGE: Duration = Duration::from_secs(24 * 60 * 60);
const EXPLICIT_OUTSIDE_SANDBOX_NOTICE: &str =
    "Notice: this command was explicitly approved to run outside the OS sandbox once.";

const SANDBOX_BYPASS_WARNING: &str = "[WARNING] OS sandbox unavailable on this platform; the command above ran without one. \
     Install bubblewrap (`apt install bubblewrap`) on Linux to enable kernel-enforced isolation.\n";

/// Hint appended when the shell exits 127 (POSIX "command not found"). The
/// sandbox uses a curated PATH that excludes anything outside the parent's
/// safe PATH entries + the well-known toolchain dirs (see
/// `sandbox::discover_sandbox_path`), so an exit-127 here often means the
/// tool is installed somewhere we didn't auto-discover. Helps the LLM and
/// the user disambiguate "tool missing" from "PATH layout problem" without
/// either side having to grep the source.
///
/// Unix-only: 127 is POSIX-specific ("command not found" from `sh`), the
/// sandbox PATH discovery is Unix-only, and `BROKK_ACP_PATH` only has an
/// effect on Unix. On Windows the same exit code can mean unrelated things
/// from `cmd.exe`/PowerShell-launched children, so we don't surface this.
#[cfg(unix)]
const EXIT_127_HINT: &str = "\n\nHint: exit 127 typically means \"command not found\". \
The brokk-acp sandbox builds PATH from your parent environment plus well-known toolchain \
dirs (~/.cargo/bin, ~/.local/bin, ~/.local/share/mise/shims, ~/.asdf/shims, ~/.pyenv/shims, \
~/.bun/bin, ~/.deno/bin, /opt/homebrew/bin, /opt/homebrew/sbin, /usr/local/bin, /usr/bin, /bin). \
If your tool's install dir is none of these, export BROKK_ACP_PATH=\"<dirs>\" in the parent \
shell before launching brokk-acp to replace the discovered PATH.";

/// Per-process rlimit caps applied to every `run_shell_command` child on Unix.
///
/// These are a parallel safety net to the OS sandbox: the sandbox bounds
/// *what* the command can touch (filesystem, namespaces); these bound *how
/// much* it can consume (memory, processes, fds, file size, CPU, core
/// dumps). They apply even with `SandboxPolicy::None` and on platforms
/// where the sandbox is unavailable, so a fork bomb or `yes > /tmp/x`
/// from an unsandboxed shell can't take the host down.
///
/// Each can be overridden per-process via the matching env var; the value
/// `unlimited` (or empty) lifts that specific cap (`RLIM_INFINITY`). An
/// invalid env value falls back to the default and emits a warning at
/// parse time (before pre_exec, so `tracing` is safe).
///
/// The `BROKK_ACP_RLIMIT_*` env vars are read from the **parent agent's
/// environment** at spawn time, not the child's: the child's env is
/// scrubbed via `env_clear()` plus an explicit whitelist (see
/// `ENV_WHITELIST` in `sandbox.rs`), so even if these names were leaked
/// into a malicious child, the values inside the sandbox cannot widen
/// the parent's caps.
///
/// Defaults are deliberately generous to accommodate JVM tooling
/// (`mvn`, `gradle`), Go binaries (which reserve large virtual ranges),
/// `rustc` LTO builds, and shared dev/CI hosts where `RLIMIT_NPROC` is
/// counted per-UID across all sessions. Values are clamped to the
/// parent's hard limit at spawn time, so requests above what the host
/// permits land at the host limit (with a warning) instead of failing
/// silently with EPERM inside `pre_exec`.
#[cfg(unix)]
const DEFAULT_RLIMIT_AS_BYTES: u64 = 32 * 1024 * 1024 * 1024; // 32 GiB virtual address space per child
#[cfg(unix)]
const DEFAULT_RLIMIT_NPROC: u64 = 8192; // user-wide process count (Linux: per real UID across all sessions)
#[cfg(unix)]
const DEFAULT_RLIMIT_NOFILE: u64 = 4096; // open file descriptors per child
#[cfg(unix)]
const DEFAULT_RLIMIT_FSIZE_BYTES: u64 = 1024 * 1024 * 1024; // 1 GiB max single-file write per child
#[cfg(unix)]
const DEFAULT_RLIMIT_CPU_SECONDS: u64 = 1800; // 30 minutes wall-equivalent CPU time per child
#[cfg(unix)]
const DEFAULT_RLIMIT_CORE_BYTES: u64 = 0; // disable core dumps (info-leak vector, can fill disk)

#[cfg(unix)]
const RLIMIT_AS_ENV: &str = "BROKK_ACP_RLIMIT_AS_BYTES";
#[cfg(unix)]
const RLIMIT_NPROC_ENV: &str = "BROKK_ACP_RLIMIT_NPROC";
#[cfg(unix)]
const RLIMIT_NOFILE_ENV: &str = "BROKK_ACP_RLIMIT_NOFILE";
#[cfg(unix)]
const RLIMIT_FSIZE_ENV: &str = "BROKK_ACP_RLIMIT_FSIZE_BYTES";
#[cfg(unix)]
const RLIMIT_CPU_ENV: &str = "BROKK_ACP_RLIMIT_CPU_SECONDS";
#[cfg(unix)]
const RLIMIT_CORE_ENV: &str = "BROKK_ACP_RLIMIT_CORE_BYTES";

#[cfg(unix)]
#[derive(Debug, Clone, Copy)]
struct RlimitConfig {
    as_bytes: u64,
    nproc: u64,
    nofile: u64,
    fsize_bytes: u64,
    cpu_seconds: u64,
    core_bytes: u64,
}

#[cfg(unix)]
impl RlimitConfig {
    fn from_env() -> Self {
        Self {
            as_bytes: parse_rlimit_env(RLIMIT_AS_ENV, DEFAULT_RLIMIT_AS_BYTES),
            nproc: parse_rlimit_env(RLIMIT_NPROC_ENV, DEFAULT_RLIMIT_NPROC),
            nofile: parse_rlimit_env(RLIMIT_NOFILE_ENV, DEFAULT_RLIMIT_NOFILE),
            fsize_bytes: parse_rlimit_env(RLIMIT_FSIZE_ENV, DEFAULT_RLIMIT_FSIZE_BYTES),
            cpu_seconds: parse_rlimit_env(RLIMIT_CPU_ENV, DEFAULT_RLIMIT_CPU_SECONDS),
            core_bytes: parse_rlimit_env(RLIMIT_CORE_ENV, DEFAULT_RLIMIT_CORE_BYTES),
        }
    }

    /// Clamp each requested value to the parent's current *hard* limit and
    /// warn the operator when a clamp occurs. Run once per spawn in the
    /// parent, before `pre_exec`, so `tracing` is safe. The child inherits
    /// the parent's hard limits across `fork()` unchanged, so by the time
    /// `setrlimit` runs in the child the requested value will be `<= hard`
    /// and won't EPERM. This makes the closure body's EPERM swallow a
    /// belt-and-suspenders check for racy limit changes rather than the
    /// primary path: configured caps either land or surface as warnings.
    fn clamp_to_parent_hard_limits(self) -> Self {
        Self {
            as_bytes: clamp_to_hard_limit(RLIMIT_AS_ENV, libc::RLIMIT_AS, self.as_bytes),
            nproc: clamp_to_hard_limit(RLIMIT_NPROC_ENV, libc::RLIMIT_NPROC, self.nproc),
            nofile: clamp_to_hard_limit(RLIMIT_NOFILE_ENV, libc::RLIMIT_NOFILE, self.nofile),
            fsize_bytes: clamp_to_hard_limit(
                RLIMIT_FSIZE_ENV,
                libc::RLIMIT_FSIZE,
                self.fsize_bytes,
            ),
            cpu_seconds: clamp_to_hard_limit(RLIMIT_CPU_ENV, libc::RLIMIT_CPU, self.cpu_seconds),
            core_bytes: clamp_to_hard_limit(RLIMIT_CORE_ENV, libc::RLIMIT_CORE, self.core_bytes),
        }
    }
}

#[cfg(unix)]
fn clamp_to_hard_limit<R>(name: &str, resource: R, requested: u64) -> u64
where
    R: TryInto<libc::c_int>,
{
    let res: libc::c_int = match resource.try_into() {
        Ok(r) => r,
        Err(_) => return requested,
    };
    let mut rlim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: getrlimit is async-signal-safe and only reads.
    let ret = unsafe { libc::getrlimit(res as _, &mut rlim) };
    if ret != 0 {
        return requested;
    }
    // rlim_t is u64 on tier-1 64-bit Unixes and u32 on 32-bit Linux;
    // cast to u64 to match the rest of the rlimit plumbing without
    // truncating on the small-int side.
    #[allow(clippy::unnecessary_cast)]
    let hard = rlim.rlim_max as u64;
    if hard == libc::RLIM_INFINITY {
        return requested;
    }
    if requested == libc::RLIM_INFINITY || requested > hard {
        tracing::warn!(
            var = name,
            requested,
            hard,
            "rlimit request exceeds parent's hard limit; clamping to hard limit"
        );
        hard
    } else {
        requested
    }
}

/// Parse a single rlimit-override env var. Empty string and "unlimited"
/// (case-insensitive) map to `RLIM_INFINITY` (no cap). Other non-numeric
/// values log a warning and fall back to `default`. Pure -- exposed for
/// unit tests so the parsing matrix can be exercised without spawning.
#[cfg(unix)]
fn parse_rlimit_env(var: &str, default: u64) -> u64 {
    match std::env::var(var) {
        Ok(s) => parse_rlimit_value(var, &s, default),
        Err(_) => default,
    }
}

#[cfg(unix)]
fn parse_rlimit_value(var: &str, raw: &str, default: u64) -> u64 {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("unlimited") {
        return libc::RLIM_INFINITY;
    }
    match trimmed.parse::<u64>() {
        Ok(v) => v,
        Err(_) => {
            tracing::warn!(
                var,
                value = trimmed,
                "invalid rlimit env var value; falling back to default"
            );
            default
        }
    }
}

/// Apply `setrlimit` for AS, NPROC, NOFILE, FSIZE, CPU, CORE on the
/// calling process.
///
/// Intended to run inside `Command::pre_exec`, i.e. between `fork()` and
/// `exec()`. Must remain async-signal-safe -- no allocation, no `tracing`,
/// no locking. Errors round-trip back to the parent via the `io::Result`
/// pre_exec contract, which aborts the spawn.
///
/// Each `set_rlimit` call only lowers the soft cap and leaves the
/// inherited hard cap unchanged (see comment in `set_rlimit`); EPERM
/// and EINVAL are swallowed so a single problematic resource doesn't
/// abort the whole spawn, while operator-config-vs-host mismatches
/// still surface as `tracing::warn!` from the parent-side
/// `clamp_to_parent_hard_limits`.
#[cfg(unix)]
fn apply_rlimits(config: &RlimitConfig) -> std::io::Result<()> {
    set_rlimit(libc::RLIMIT_AS, config.as_bytes)?;
    set_rlimit(libc::RLIMIT_NPROC, config.nproc)?;
    set_rlimit(libc::RLIMIT_NOFILE, config.nofile)?;
    set_rlimit(libc::RLIMIT_FSIZE, config.fsize_bytes)?;
    set_rlimit(libc::RLIMIT_CPU, config.cpu_seconds)?;
    set_rlimit(libc::RLIMIT_CORE, config.core_bytes)?;
    Ok(())
}

#[cfg(unix)]
fn set_rlimit<R>(resource: R, value: u64) -> std::io::Result<()>
where
    R: TryInto<libc::c_int>,
{
    // `libc::RLIMIT_*` are `__rlimit_resource_t` (u32) on Linux and
    // `c_int` on macOS; coerce to whatever `setrlimit` takes on this
    // platform. Failure here is a programming/libc-binding error: every
    // `RLIMIT_*` constant in use is a small non-negative integer that
    // fits in `c_int` on every Unix we ship for. Fail closed with an
    // explicit io::Error so the spawn aborts loudly rather than
    // silently dropping a cap if a future libc bump changes the type.
    let res: libc::c_int = resource.try_into().map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "rlimit resource constant did not fit in c_int (libc-binding error)",
        )
    })?;
    // Read the inherited limits and only lower the *soft* cap; never
    // touch the hard cap. Two consequences:
    //   1. We can't EPERM on "raising hard limit" (Linux without
    //      CAP_SYS_RESOURCE).
    //   2. We can't EINVAL on "rlim_cur > rlim_max" (macOS, where
    //      setrlimit's accepted-value envelope is narrower than what
    //      getrlimit's report suggests -- e.g. RLIMIT_RSS/RLIMIT_AS is
    //      deprecated, RLIMIT_NPROC is bounded by `kern.maxprocperuid`,
    //      and RLIM_INFINITY operands round-trip differently).
    // Tradeoff: a child can `setrlimit` its own rlim_cur back up to the
    // inherited rlim_max. That is acceptable for this code's role: it
    // is a safety net against accidental runaways (fork bombs, `dd
    // if=/dev/zero`), not the primary security boundary -- the OS
    // sandbox (bwrap/Seatbelt) is. The parent-side
    // `clamp_to_parent_hard_limits` already warns the operator when a
    // requested cap can't be enforced past the host's own ceiling.
    let mut current = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: getrlimit is async-signal-safe and only reads.
    if unsafe { libc::getrlimit(res as _, &mut current) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let want = value as libc::rlim_t;
    let new_cur = if current.rlim_max == libc::RLIM_INFINITY {
        want
    } else if want == libc::RLIM_INFINITY {
        // "Unlimited" can't go above what we already inherited.
        current.rlim_max
    } else {
        std::cmp::min(want, current.rlim_max)
    };
    // No-op if the soft cap is already where we want it. macOS in
    // particular can EINVAL on identity calls where rlim_cur and
    // rlim_max are at their reported values, so we'd rather not poke
    // setrlimit unless we're actually changing something.
    if new_cur == current.rlim_cur {
        return Ok(());
    }
    let rlim = libc::rlimit {
        rlim_cur: new_cur,
        rlim_max: current.rlim_max,
    };
    // SAFETY: `setrlimit` is async-signal-safe and is called only from
    // `pre_exec`, which is the canonical place to apply per-child caps
    // on Unix. The pointer is to a stack-local that outlives the call.
    let ret = unsafe { libc::setrlimit(res as _, &rlim) };
    if ret == 0 {
        Ok(())
    } else {
        let err = std::io::Error::last_os_error();
        // We never raise rlim_max and never set rlim_cur > rlim_max, so
        // EPERM/EINVAL here means a libc/kernel divergence we have no
        // recourse for from inside pre_exec (no allocation, no
        // tracing). Swallow rather than aborting the spawn -- the
        // parent has already emitted any clamp warnings via
        // `tracing::warn!`, and the other rlimits in this batch may
        // still apply.
        match err.raw_os_error() {
            Some(libc::EPERM) | Some(libc::EINVAL) => Ok(()),
            _ => Err(err),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn format_shell_tool_result(
    stdout: &str,
    stderr: &str,
    minimized: Option<String>,
    exit_code: i32,
    success: bool,
    outside_sandbox_once: bool,
    bypass_warning: bool,
    timeout_clamp_notice: Option<&str>,
) -> ToolResult {
    // The body is truncated middle-out (head + tail kept) BEFORE the exit-code
    // line and notices are attached: build/test failures cluster at the tail,
    // and the exit code must survive truncation unconditionally.
    let mut combined = minimized.unwrap_or_else(|| {
        let mut body = String::new();
        if !stdout.is_empty() {
            body.push_str(stdout);
        }
        if !stderr.is_empty() {
            if !body.is_empty() {
                body.push_str("\n--- stderr ---\n");
            }
            body.push_str(stderr);
        }
        body
    });

    if combined.len() > MAX_OUTPUT_BYTES {
        combined = crate::text::truncate_middle_utf8(&combined, MAX_OUTPUT_BYTES, |elided| {
            format!("\n[... {elided} bytes elided ...]\n")
        });
    }

    if !success {
        combined.push_str(&format!("\n\nExit code: {exit_code}"));
        #[cfg(unix)]
        if exit_code == 127 {
            combined.push_str(EXIT_127_HINT);
        }
    }

    if combined.is_empty() {
        combined = format!("Command completed with exit code {exit_code}");
    }

    if let Some(notice) = timeout_clamp_notice {
        combined = format!("{notice}\n\n{combined}");
    }

    if outside_sandbox_once {
        combined = format!("{EXPLICIT_OUTSIDE_SANDBOX_NOTICE}\n\n{combined}");
    }

    if bypass_warning {
        combined.push('\n');
        combined.push_str(SANDBOX_BYPASS_WARNING);
    }

    ToolResult {
        status: if success {
            ToolStatus::Success
        } else {
            ToolStatus::RequestError
        },
        output: combined,
    }
}

/// Post-capture output minimizer for one session's `run_shell_command`.
///
/// Wraps the vendored oh-my-pi minimizer (`anvil_minimizer`): after the child
/// exits, output of well-known commands (git, cargo, pytest, npm, ...) is
/// condensed with the exit code as an input, and the raw capture is preserved
/// under [`SPILL_DIR_RELATIVE`] so minimization never loses information. The
/// engine refuses pipes, compound commands, and captures over its size cap,
/// and converts filter panics to passthrough, so it can only ever shrink
/// well-understood output.
pub(crate) struct ShellMinimizer {
    config: anvil_minimizer::MinimizerConfig,
    /// `<session cwd>/.brokk/shell-output`. Rooted at the session cwd, not a
    /// per-command `directory` override -- see [`SPILL_DIR_RELATIVE`].
    spill_dir: PathBuf,
}

impl ShellMinimizer {
    pub(crate) fn new(session_cwd: &Path) -> Self {
        Self {
            config: anvil_minimizer::MinimizerConfig {
                enabled: true,
                ..Default::default()
            },
            spill_dir: session_cwd.join(SPILL_DIR_RELATIVE),
        }
    }

    /// Condense the capture when a filter recognizes the command. Returns the
    /// minimized body -- with a trailing `[raw output: ...]` reference when
    /// the original was spilled -- or `None` to keep the verbatim output.
    fn minimize(
        &self,
        command: &str,
        stdout: &str,
        stderr: &str,
        exit_code: i32,
    ) -> Option<String> {
        // Feed one merged buffer, mirroring upstream's capture: filters for
        // compilers/test runners expect diagnostics and results interleaved,
        // and a divider line would confuse their line-oriented parsers.
        let captured: Cow<'_, str> = if stderr.is_empty() {
            Cow::Borrowed(stdout)
        } else if stdout.is_empty() {
            Cow::Borrowed(stderr)
        } else {
            Cow::Owned(format!("{stdout}\n{stderr}"))
        };
        let out = anvil_minimizer::apply(command, &captured, exit_code, &self.config);
        if !out.changed {
            return None;
        }
        // `original_text` is Some exactly when the filter rewrote the output
        // (upstream contract); without it we cannot offer recovery, so keep
        // the verbatim capture.
        let original = out.original_text?;
        let mut text = out.text;
        if let Some(reference) = spill_original(&self.spill_dir, &original) {
            if !text.ends_with('\n') {
                text.push('\n');
            }
            text.push_str(&format!("[raw output: {reference}]"));
        }
        Some(text)
    }
}

/// Write `original` under the spill dir, creating it (with a self-ignoring
/// `.gitignore`) on first use. Returns the session-relative reference path,
/// or `None` on any error: spilling must never fail the tool call, the
/// result is merely shown without a recovery reference.
fn spill_original(spill_dir: &Path, original: &str) -> Option<String> {
    if let Err(e) = std::fs::create_dir_all(spill_dir) {
        tracing::debug!("could not create shell-output spill dir: {e}");
        return None;
    }
    let gitignore = spill_dir.join(".gitignore");
    if !gitignore.exists()
        && let Err(e) = std::fs::write(&gitignore, "*\n")
    {
        tracing::debug!("could not write shell-output .gitignore: {e}");
    }
    let name = next_spill_file_name();
    let path = spill_dir.join(&name);
    match std::fs::write(&path, original) {
        Ok(()) => Some(format!("{SPILL_DIR_RELATIVE}/{name}")),
        Err(e) => {
            tracing::debug!("could not write shell-output spill file: {e}");
            None
        }
    }
}

/// `{unix_secs}-{pid}-{seq}`: `seq` disambiguates within a process, `pid`
/// across concurrent anvil processes sharing a workspace, and the timestamp
/// guards pid reuse across restarts while keeping age-based GC legible.
fn next_spill_file_name() -> String {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}-{}-{seq}.txt", std::process::id())
}

/// Age-based GC for spill files, mirroring
/// `sandbox::cleanup_stale_policy_files`: runs at registry construction so
/// captures from crashed or abandoned sessions don't accumulate.
pub(crate) fn cleanup_stale_shell_outputs(session_cwd: &Path) {
    let dir = session_cwd.join(SPILL_DIR_RELATIVE);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    let now = std::time::SystemTime::now();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("txt") {
            continue;
        }
        let stale = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age > STALE_SPILL_FILE_AGE);
        if stale {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Effective ceiling on a shell timeout, in seconds.
///
/// [`MAX_TIMEOUT_SECONDS`] unless `ANVIL_SHELL_TIMEOUT_CAP_SECONDS` names a
/// positive integer. An absent, empty, zero, or unparseable value falls back
/// to the default rather than failing the call: a mistyped deployment knob
/// must not break every shell command.
fn timeout_cap_seconds() -> u64 {
    parse_timeout_cap(std::env::var(TIMEOUT_CAP_ENV).ok().as_deref())
}

fn parse_timeout_cap(raw: Option<&str>) -> u64 {
    raw.and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .unwrap_or(MAX_TIMEOUT_SECONDS)
}

/// A resolved wall-clock budget for one shell call, plus the user-visible
/// notice to emit when the request had to be clamped.
///
/// Every timeout policy decision lives here so the advertised
/// `timeout_seconds` field, the retained legacy millisecond `timeout` field,
/// and internal callers cannot drift apart:
///
/// - [`Self::from_request_seconds`] -- what the model asks for. Clamped to
///   `[MIN_TIMEOUT_SECONDS, cap]`, with a notice in either direction.
/// - [`Self::from_legacy_millis`] -- the unadvertised millisecond field kept
///   for replay/internal compatibility. Rounds up to whole seconds with a 1s
///   floor exactly as before; only the ceiling moved. It is deliberately NOT
///   raised to `MIN_TIMEOUT_SECONDS`, so a replayed trace that asked for
///   1500ms still gets 2s.
/// - [`Self::from_exact_seconds`] -- in-process callers that already speak
///   seconds; same permissive floor as the legacy path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellTimeout {
    seconds: u64,
    clamp_notice: Option<String>,
}

impl ShellTimeout {
    /// Pick the budget for one `run_shell_command` call from its arguments.
    /// `timeout_seconds` wins when both fields are present.
    pub(super) fn resolve(timeout_seconds: Option<u64>, legacy_millis: Option<u64>) -> Self {
        match (timeout_seconds, legacy_millis) {
            (Some(seconds), _) => Self::from_request_seconds(seconds),
            (None, Some(millis)) => Self::from_legacy_millis(millis),
            (None, None) => Self::default_budget(),
        }
    }

    /// Neither timeout field was supplied. The model asked for nothing, so
    /// there is nothing to report even if a deployment cap lowers the
    /// default.
    fn default_budget() -> Self {
        Self {
            seconds: DEFAULT_TIMEOUT_SECONDS.min(timeout_cap_seconds()),
            clamp_notice: None,
        }
    }

    fn from_request_seconds(requested: u64) -> Self {
        let cap = timeout_cap_seconds();
        let seconds = requested.max(MIN_TIMEOUT_SECONDS).min(cap);
        let clamp_notice = if seconds < requested {
            Some(format!(
                "Notice: requested timeout {requested}s exceeded the server maximum; clamped to {seconds}s."
            ))
        } else if seconds > requested {
            Some(format!(
                "Notice: requested timeout {requested}s was below the {MIN_TIMEOUT_SECONDS}s minimum; raised to {seconds}s."
            ))
        } else {
            None
        };
        Self {
            seconds,
            clamp_notice,
        }
    }

    fn from_legacy_millis(millis: u64) -> Self {
        Self::from_exact_seconds(millis.saturating_add(999) / 1000)
    }

    fn from_exact_seconds(requested: u64) -> Self {
        let requested = requested.max(1);
        let seconds = requested.min(timeout_cap_seconds());
        let clamp_notice = (seconds != requested).then(|| {
            format!(
                "Notice: requested timeout {requested}s exceeded the server maximum; clamped to {seconds}s."
            )
        });
        Self {
            seconds,
            clamp_notice,
        }
    }

    #[cfg(test)]
    fn seconds(&self) -> u64 {
        self.seconds
    }

    #[cfg(test)]
    fn clamp_notice(&self) -> Option<&str> {
        self.clamp_notice.as_deref()
    }
}

/// Seconds-taking entry point kept for the tests that predate
/// [`ShellTimeout`]; production dispatch resolves the budget from the tool
/// arguments and calls [`run_shell_command_with_timeout`] directly.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub async fn run_shell_command_cancellable(
    cwd: &Path,
    command: &str,
    timeout_seconds: u64,
    policy: SandboxPolicy,
    outside_sandbox_once: bool,
    cancel: Option<&CancellationToken>,
    minimizer: Option<&ShellMinimizer>,
) -> ToolResult {
    run_shell_command_with_timeout(
        cwd,
        command,
        ShellTimeout::from_exact_seconds(timeout_seconds),
        policy,
        outside_sandbox_once,
        cancel,
        minimizer,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn run_shell_command_with_timeout(
    cwd: &Path,
    command: &str,
    timeout: ShellTimeout,
    policy: SandboxPolicy,
    outside_sandbox_once: bool,
    cancel: Option<&CancellationToken>,
    minimizer: Option<&ShellMinimizer>,
) -> ToolResult {
    if command.trim().is_empty() {
        return ToolResult {
            status: ToolStatus::RequestError,
            output: "Command must not be empty".to_string(),
        };
    }
    let ShellTimeout {
        seconds: timeout_seconds,
        clamp_notice: timeout_clamp_notice,
    } = timeout;

    // Wrap once. `wrapped` owns the temp policy file (Seatbelt) and must
    // outlive the spawned child.
    let wrapped = match sandbox::wrap_command(policy, cwd, command) {
        Ok(w) => w,
        Err(e) => {
            // Sandbox-layer errors are tagged with `[sandbox]` by sandbox.rs
            // so the user can tell wrap-side failures from command-side ones.
            return ToolResult {
                status: ToolStatus::InternalError,
                output: format!("Failed to prepare sandbox: {e}"),
            };
        }
    };
    tracing::debug!(
        target: "brokk_acp_rust::tools::shell",
        policy = ?policy,
        cwd = %cwd.display(),
        sandboxed = wrapped.sandboxed,
        argv0 = %wrapped.argv[0],
        "run_shell_command: wrapped command ready",
    );

    // The user requested a sandbox tier (ReadOnly / WorkspaceWrite) but the
    // platform tooling was missing. We still execute -- matching Java's
    // log-and-skip posture -- but prepend a visible warning to the output so
    // the LLM and the ACP client both know the call wasn't actually bounded.
    let bypass_warning = !matches!(policy, SandboxPolicy::None) && !wrapped.sandboxed;

    // Mirrors `Environment.createProcessBuilder` (Environment.java:647-661),
    // and stricter: we env_clear() and explicitly add only a small whitelist
    // so secrets in the parent process (OPENAI_API_KEY, AWS_*, GH tokens,
    // LD_PRELOAD/DYLD_*) cannot leak into LLM-driven shell calls. On Linux
    // the same scrubbing is applied via bwrap's `--clearenv`/`--setenv`.
    // On Unix the sandbox PATH is discovered (not hardcoded) so toolchains
    // under ~/.cargo/bin, /opt/homebrew/bin, mise/asdf shims, etc. are
    // reachable from LLM-driven shell calls. On Linux this PATH only
    // governs how bwrap itself resolves `sh`; the inner child's PATH is
    // set via bwrap's --setenv (sandbox.rs). On macOS Seatbelt does not
    // alter env so this is the PATH the child actually sees.
    //
    // On Windows there is no sandbox (`wrap_platform` is a passthrough),
    // and the home-tool-dir / Homebrew layout makes no sense, so we
    // fall back to the historic hardcoded Unix-style PATH the platform
    // already had. The point is to keep Windows on its prior path of
    // behavior; deciding what `run_shell_command` *should* do on Windows
    // is a separate concern from this issue.
    #[cfg(unix)]
    let sandbox_path = sandbox::discover_sandbox_path();
    #[cfg(not(unix))]
    let sandbox_path = std::env::var_os("PATH").unwrap_or_default();

    let mut cmd = Command::new(&wrapped.argv[0]);
    cmd.args(&wrapped.argv[1..])
        .current_dir(cwd)
        .env_clear()
        .env("PATH", &sandbox_path)
        .env("TERM", "dumb")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        // If the wall-clock timeout fires below, dropping `cmd.output()`'s
        // future tears down the Child; without `kill_on_drop`, tokio leaves
        // a runaway/CPU-spinning child alive past timeout, holding the
        // rlimit budget and consuming CPU until reaped some other way.
        .kill_on_drop(true);
    for key in ENV_WHITELIST {
        if let Some(value) = std::env::var_os(key) {
            cmd.env(key, value);
        }
    }

    // Apply per-process rlimits via `pre_exec` so AS/NPROC/NOFILE/FSIZE/
    // CPU/CORE caps land on the child (and any sandbox wrapper, which
    // inherits them) without affecting the parent agent. Read env once
    // here and clamp to the parent's hard limits so the closure body
    // stays async-signal-safe and the child's setrlimit calls can't
    // EPERM on "asked for more than hard limit" -- a clamp-with-warning
    // in the parent (where `tracing` is safe) is strictly more visible
    // than a silent EPERM swallow inside pre_exec.
    #[cfg(unix)]
    {
        let rlimits = RlimitConfig::from_env().clamp_to_parent_hard_limits();
        // SAFETY: `apply_rlimits` only calls `libc::setrlimit`, which is
        // async-signal-safe. No allocation, no locking, no `tracing` --
        // safe to invoke between fork() and exec().
        unsafe {
            cmd.pre_exec(move || apply_rlimits(&rlimits));
        }
    }

    #[cfg(unix)]
    {
        // Give each shell command its own process group so cancellation and
        // timeout can kill descendants that outlive the shell wrapper.
        unsafe {
            cmd.pre_exec(|| {
                if libc::setpgid(0, 0) == 0 {
                    Ok(())
                } else {
                    Err(std::io::Error::last_os_error())
                }
            });
        }
    }

    let result = run_child_to_completion(cmd, timeout_seconds, cancel).await;

    // `wrapped` MUST stay in scope until output() resolves so the
    // TempPolicyFile Drop guard doesn't yank `sandbox-exec`'s `-f` profile
    // mid-call. The explicit drop is a tripwire for refactors that might
    // otherwise reuse `wrapped`'s name and shrink its lifetime.
    drop(wrapped);

    match result {
        ShellRunResult::Completed {
            status,
            stdout,
            stderr,
        } => {
            let stdout = String::from_utf8_lossy(&stdout);
            let stderr = String::from_utf8_lossy(&stderr);
            let exit_code = status.code().unwrap_or(-1);
            // Minimization sees the untruncated capture (the engine's own
            // 4 MiB cap turns monsters into passthrough); truncation happens
            // afterwards in the formatter.
            let minimized =
                minimizer.and_then(|m| m.minimize(command, &stdout, &stderr, exit_code));
            format_shell_tool_result(
                &stdout,
                &stderr,
                minimized,
                exit_code,
                status.success(),
                outside_sandbox_once,
                bypass_warning,
                timeout_clamp_notice.as_deref(),
            )
        }
        ShellRunResult::FailedToExecute(e) => {
            let mut output = format!("Failed to execute command: {e}");
            prepend_notice(&mut output, timeout_clamp_notice.as_deref());
            ToolResult {
                status: ToolStatus::InternalError,
                output,
            }
        }
        ShellRunResult::TimedOut => {
            let mut msg = format!(
                "Command timed out after {timeout_seconds}s; terminated the child process tree."
            );
            prepend_notice(&mut msg, timeout_clamp_notice.as_deref());
            if outside_sandbox_once {
                msg = format!("{EXPLICIT_OUTSIDE_SANDBOX_NOTICE}\n\n{msg}");
            }
            if bypass_warning {
                msg.push('\n');
                msg.push_str(SANDBOX_BYPASS_WARNING);
            }
            ToolResult {
                status: ToolStatus::RequestError,
                output: msg,
            }
        }
        ShellRunResult::Cancelled => {
            let mut msg =
                "Command was cancelled before it completed; terminated the child process tree."
                    .to_string();
            prepend_notice(&mut msg, timeout_clamp_notice.as_deref());
            if outside_sandbox_once {
                msg = format!("{EXPLICIT_OUTSIDE_SANDBOX_NOTICE}\n\n{msg}");
            }
            if bypass_warning {
                msg.push('\n');
                msg.push_str(SANDBOX_BYPASS_WARNING);
            }
            ToolResult {
                status: ToolStatus::RequestError,
                output: msg,
            }
        }
    }
}

fn prepend_notice(output: &mut String, notice: Option<&str>) {
    if let Some(notice) = notice {
        *output = format!("{notice}\n\n{output}");
    }
}

enum ShellRunResult {
    Completed {
        status: ExitStatus,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    },
    FailedToExecute(io::Error),
    TimedOut,
    Cancelled,
}

async fn run_child_to_completion(
    mut cmd: Command,
    timeout_seconds: u64,
    cancel: Option<&CancellationToken>,
) -> ShellRunResult {
    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(error) => return ShellRunResult::FailedToExecute(error),
    };

    let stdout_task = tokio::spawn(read_pipe_to_end(child.stdout.take()));
    let stderr_task = tokio::spawn(read_pipe_to_end(child.stderr.take()));
    let timeout = tokio::time::sleep(Duration::from_secs(timeout_seconds));
    tokio::pin!(timeout);

    // When there is no cancel token, a `pending()` future keeps the select
    // arm shape identical without ever firing -- avoiding two near-duplicate
    // `select!` blocks that drift apart over time.
    let cancelled = async {
        match cancel {
            Some(cancel) => cancel.cancelled().await,
            None => std::future::pending::<()>().await,
        }
    };
    tokio::pin!(cancelled);

    let termination = tokio::select! {
        biased;
        _ = &mut cancelled => {
            terminate_child_tree(&mut child).await;
            ShellTermination::Cancelled
        }
        _ = &mut timeout => {
            terminate_child_tree(&mut child).await;
            ShellTermination::TimedOut
        }
        wait_result = child.wait() => match wait_result {
            Ok(status) => ShellTermination::Completed(status),
            Err(error) => ShellTermination::FailedToWait(error),
        },
    };

    match termination {
        ShellTermination::Completed(status) => ShellRunResult::Completed {
            status,
            stdout: join_pipe_output(stdout_task).await,
            stderr: join_pipe_output(stderr_task).await,
        },
        ShellTermination::FailedToWait(error) => ShellRunResult::FailedToExecute(error),
        // The child tree has been SIGKILLed/taskkilled, but a grandchild that
        // escaped the process group (e.g. `setsid`) can keep the stdout/stderr
        // pipe open, so `read_to_end` would never see EOF. We discard the
        // output on these paths anyway, so abort the readers instead of
        // awaiting them -- otherwise the join would hang indefinitely (there
        // is no outer timeout around shell cancellation).
        ShellTermination::TimedOut => {
            stdout_task.abort();
            stderr_task.abort();
            ShellRunResult::TimedOut
        }
        ShellTermination::Cancelled => {
            stdout_task.abort();
            stderr_task.abort();
            ShellRunResult::Cancelled
        }
    }
}

async fn terminate_child_tree(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        let pgid = -(pid as libc::pid_t);
        // SAFETY: `kill` is called with a negative process-group id that we
        // created in `pre_exec`; errors just fall through to the child kill
        // fallback below.
        let _ = unsafe { libc::kill(pgid, libc::SIGKILL) };
    }

    #[cfg(windows)]
    if let Some(pid) = child.id() {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await;
    }

    let _ = child.kill().await;
}

enum ShellTermination {
    Completed(ExitStatus),
    FailedToWait(io::Error),
    TimedOut,
    Cancelled,
}

async fn read_pipe_to_end<R>(pipe: Option<R>) -> Vec<u8>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let Some(mut pipe) = pipe else {
        return Vec::new();
    };
    let mut bytes = Vec::new();
    let _ = pipe.read_to_end(&mut bytes).await;
    bytes
}

async fn join_pipe_output(task: tokio::task::JoinHandle<Vec<u8>>) -> Vec<u8> {
    task.await.unwrap_or_default()
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tokio::sync::Mutex;

    /// Serializes tests that mutate process-wide env vars. `cargo test`
    /// runs `#[test]`/`#[tokio::test]` cases on parallel threads inside a
    /// single test binary, so env reads/writes against
    /// `BROKK_ACP_RLIMIT_*` must be funneled through this lock or one
    /// test's setup races against another's `from_env()`.
    ///
    /// `tokio::sync::Mutex` because the guard is held across `await`
    /// points (the test invokes `run_shell_command_cancellable(...).await`); a
    /// `std::sync::Mutex` here would trigger `clippy::await_holding_lock`.
    static ENV_LOCK: Mutex<()> = Mutex::const_new(());

    /// Restores a single env var on Drop. Pair with `ENV_LOCK` so the
    /// drop order is well-defined.
    struct EnvGuard {
        var: &'static str,
    }

    impl EnvGuard {
        fn set(var: &'static str, value: &str) -> Self {
            // SAFETY: the caller holds ENV_LOCK, which serializes env
            // mutation across this crate's tests. Outside test code,
            // `std::env::set_var` is unsafe in Rust 2024 because it
            // races with concurrent reads from other threads.
            unsafe {
                std::env::set_var(var, value);
            }
            Self { var }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: same as set() -- guarded by ENV_LOCK.
            unsafe {
                std::env::remove_var(self.var);
            }
        }
    }

    #[test]
    fn parse_rlimit_value_accepts_decimal_byte_count() {
        assert_eq!(parse_rlimit_value("X", "1024", 999), 1024);
        assert_eq!(parse_rlimit_value("X", "  4096  ", 999), 4096);
    }

    #[test]
    fn parse_rlimit_value_treats_unlimited_and_empty_as_infinity() {
        assert_eq!(
            parse_rlimit_value("X", "unlimited", 999),
            libc::RLIM_INFINITY
        );
        assert_eq!(
            parse_rlimit_value("X", "UNLIMITED", 999),
            libc::RLIM_INFINITY
        );
        assert_eq!(parse_rlimit_value("X", "", 999), libc::RLIM_INFINITY);
        assert_eq!(parse_rlimit_value("X", "   ", 999), libc::RLIM_INFINITY);
    }

    #[test]
    fn parse_rlimit_value_falls_back_on_garbage() {
        // Non-numeric input should not crash and should return the default
        // -- protects against operator typos in env config.
        assert_eq!(parse_rlimit_value("X", "lots", 999), 999);
        assert_eq!(parse_rlimit_value("X", "-1", 999), 999);
        assert_eq!(parse_rlimit_value("X", "1.5", 999), 999);
    }

    #[test]
    fn clamp_to_hard_limit_passes_through_when_request_below_hard() {
        // RLIMIT_NOFILE hard limit is at least 1024 on every realistic
        // host; 64 is far below and should pass through unchanged.
        assert_eq!(
            clamp_to_hard_limit(RLIMIT_NOFILE_ENV, libc::RLIMIT_NOFILE, 64),
            64
        );
    }

    #[test]
    fn clamp_to_hard_limit_caps_request_above_hard() {
        // Read the current hard NOFILE; ask for hard+1 and expect hard.
        // If the host has no hard cap (RLIM_INFINITY), there's nothing
        // to clamp against and the test is a no-op -- which is the
        // documented behavior of clamp_to_hard_limit.
        let mut current = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: getrlimit is async-signal-safe and only reads.
        let ret = unsafe { libc::getrlimit(libc::RLIMIT_NOFILE as _, &mut current) };
        assert_eq!(ret, 0);
        // Same `as u64` rationale as in `clamp_to_hard_limit`: rlim_t is
        // u32 on 32-bit Linux and u64 elsewhere.
        #[allow(clippy::unnecessary_cast)]
        let hard = current.rlim_max as u64;
        if hard == libc::RLIM_INFINITY {
            return;
        }
        let clamped = clamp_to_hard_limit(
            RLIMIT_NOFILE_ENV,
            libc::RLIMIT_NOFILE,
            hard.saturating_add(1),
        );
        assert_eq!(clamped, hard);
        let clamped_inf =
            clamp_to_hard_limit(RLIMIT_NOFILE_ENV, libc::RLIMIT_NOFILE, libc::RLIM_INFINITY);
        assert_eq!(clamped_inf, hard);
    }

    #[test]
    fn shell_output_truncation_preserves_utf8() {
        let mut stdout = "a".repeat(MAX_OUTPUT_BYTES - 1);
        stdout.push('\u{25cf}');
        stdout.push_str("tail");

        let result = format_shell_tool_result(&stdout, "", None, 0, true, false, false, None);

        assert!(result.output.starts_with('a'), "head must be preserved");
        assert!(
            result.output.contains("bytes elided"),
            "middle-out marker expected; got tail: {}",
            &result.output[result.output.len().saturating_sub(80)..]
        );
        assert!(
            result.output.ends_with("tail"),
            "tail must be preserved; got tail: {}",
            &result.output[result.output.len().saturating_sub(80)..]
        );
    }

    /// The exit-code line is appended after truncation, so it must survive
    /// even when the body is far over budget. Regression for the head-only
    /// truncation that used to eat it.
    #[test]
    fn exit_code_survives_truncation_of_huge_output() {
        let stdout = "x".repeat(MAX_OUTPUT_BYTES * 3);
        let result = format_shell_tool_result(&stdout, "", None, 1, false, false, false, None);
        assert!(
            result.output.ends_with("Exit code: 1"),
            "exit code must be the suffix; got tail: {}",
            &result.output[result.output.len().saturating_sub(80)..]
        );
    }

    /// End-to-end: a tight `BROKK_ACP_RLIMIT_FSIZE_BYTES` actually kills
    /// `dd` when it tries to write past the cap. This is the regression
    /// test the reviewer flagged as missing -- a future refactor that
    /// drops `cmd.pre_exec`, breaks the env-var read, or wires the cap
    /// to the wrong syscall would silently disable the sandbox feature
    /// without this assertion failing.
    #[tokio::test]
    async fn rlimit_fsize_actually_kills_oversized_writes() {
        let _guard = ENV_LOCK.lock().await;
        // 1 KiB cap; the dd block size below is 8 KiB so the very first
        // write exceeds the cap -> SIGXFSZ -> non-zero exit.
        let _env = EnvGuard::set(RLIMIT_FSIZE_ENV, "1024");

        let dir = std::env::temp_dir();
        let target: PathBuf = dir.join(format!("brokk-rlimit-fsize-{}", std::process::id()));
        let _ = std::fs::remove_file(&target);
        let cmd = format!(
            "dd if=/dev/zero of='{}' bs=8192 count=1 2>&1",
            target.display()
        );

        let result =
            run_shell_command_cancellable(&dir, &cmd, 30, SandboxPolicy::None, false, None, None)
                .await;
        let written = std::fs::metadata(&target).map(|m| m.len()).unwrap_or(0);
        let _ = std::fs::remove_file(&target);

        assert!(
            !matches!(result.status, ToolStatus::Success),
            "RLIMIT_FSIZE=1024 should have killed dd; got success with output: {}",
            result.output
        );
        assert!(
            written <= 1024,
            "RLIMIT_FSIZE=1024 should have stopped writes at 1 KiB but file grew to {} bytes",
            written
        );
    }

    /// End-to-end: lifting a cap via `unlimited` makes `dd` succeed at a
    /// size that the default (1 GiB) also permits. Pairs with the test
    /// above: together they show the env-var path actually reaches the
    /// child and the limit is what governs success/failure.
    #[tokio::test]
    async fn rlimit_fsize_unlimited_allows_writes() {
        let _guard = ENV_LOCK.lock().await;
        let _env = EnvGuard::set(RLIMIT_FSIZE_ENV, "unlimited");

        let dir = std::env::temp_dir();
        let target: PathBuf = dir.join(format!("brokk-rlimit-fsize-ok-{}", std::process::id()));
        let _ = std::fs::remove_file(&target);
        let cmd = format!(
            "dd if=/dev/zero of='{}' bs=8192 count=1 2>&1",
            target.display()
        );

        let result =
            run_shell_command_cancellable(&dir, &cmd, 30, SandboxPolicy::None, false, None, None)
                .await;
        let _ = std::fs::remove_file(&target);

        assert!(
            matches!(result.status, ToolStatus::Success),
            "unlimited FSIZE should allow an 8 KiB write; got non-success with output: {}",
            result.output
        );
    }

    /// `exit 127` ("command not found") triggers the BROKK_ACP_PATH hint so
    /// the LLM and the user can tell sandbox-PATH problems from genuine
    /// missing tools. Runs with `SandboxPolicy::None` so we don't depend on
    /// bwrap/sandbox-exec being available in CI.
    #[tokio::test]
    async fn shell_exit_127_appends_brokk_acp_path_hint() {
        let _guard = ENV_LOCK.lock().await;
        let dir = std::env::temp_dir();
        let result = run_shell_command_cancellable(
            &dir,
            "nonexistent_brokk_acp_command_xyz_qqq_42",
            30,
            SandboxPolicy::None,
            false,
            None,
            None,
        )
        .await;
        assert!(
            result.output.contains("Hint: exit 127"),
            "exit-127 output must contain the diagnostic hint; got: {}",
            result.output
        );
        assert!(
            result.output.contains("BROKK_ACP_PATH"),
            "hint must mention the BROKK_ACP_PATH escape hatch; got: {}",
            result.output
        );
    }

    /// Successful commands must NOT carry the exit-127 hint -- it would
    /// just be noise. Pairs with `shell_exit_127_appends_brokk_acp_path_hint`
    /// to lock in the gate condition.
    #[tokio::test]
    async fn shell_exit_zero_omits_hint() {
        let _guard = ENV_LOCK.lock().await;
        let dir = std::env::temp_dir();
        let result = run_shell_command_cancellable(
            &dir,
            "echo hello-brokk",
            30,
            SandboxPolicy::None,
            false,
            None,
            None,
        )
        .await;
        assert!(
            matches!(result.status, ToolStatus::Success),
            "echo must succeed; got: {}",
            result.output
        );
        assert!(
            !result.output.contains("Hint: exit 127"),
            "successful command must not contain exit-127 hint; got: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn run_shell_command_minimizes_git_status_and_writes_spill() {
        let _guard = ENV_LOCK.lock().await;
        let dir = tempfile::tempdir().expect("create temp git repo");
        let minimizer = ShellMinimizer::new(dir.path());
        let init = run_shell_command_cancellable(
            dir.path(),
            "git init",
            30,
            SandboxPolicy::None,
            false,
            None,
            None,
        )
        .await;
        assert!(
            matches!(init.status, ToolStatus::Success),
            "git init must succeed; got: {}",
            init.output
        );
        std::fs::write(dir.path().join("changed.txt"), "hello\n").expect("write file");

        let result = run_shell_command_cancellable(
            dir.path(),
            "git status",
            30,
            SandboxPolicy::None,
            false,
            None,
            Some(&minimizer),
        )
        .await;

        assert!(
            matches!(result.status, ToolStatus::Success),
            "git status must succeed; got: {}",
            result.output
        );
        assert!(
            result.output.contains("changed.txt"),
            "condensed git status must still mention the untracked file; got: {}",
            result.output
        );
        let reference_start = result
            .output
            .find("[raw output: ")
            .unwrap_or_else(|| panic!("expected raw-output reference; got: {}", result.output));
        let reference = &result.output[reference_start + "[raw output: ".len()..];
        let reference = &reference[..reference.find(']').expect("closing bracket")];
        let spill_path = dir.path().join(reference);
        let spilled = std::fs::read_to_string(&spill_path).expect("spill file readable");
        assert!(
            spilled.contains("changed.txt"),
            "spill file must hold the raw capture; got: {spilled}"
        );
        let gitignore = dir.path().join(SPILL_DIR_RELATIVE).join(".gitignore");
        assert_eq!(
            std::fs::read_to_string(gitignore).expect("spill .gitignore"),
            "*\n",
            "spill dir must self-ignore"
        );
    }

    #[tokio::test]
    async fn run_shell_command_without_minimizer_is_verbatim() {
        let _guard = ENV_LOCK.lock().await;
        let dir = tempfile::tempdir().expect("create temp git repo");
        let init = run_shell_command_cancellable(
            dir.path(),
            "git init",
            30,
            SandboxPolicy::None,
            false,
            None,
            None,
        )
        .await;
        assert!(matches!(init.status, ToolStatus::Success));

        let result = run_shell_command_cancellable(
            dir.path(),
            "git status",
            30,
            SandboxPolicy::None,
            false,
            None,
            None,
        )
        .await;

        assert!(
            !result.output.contains("[raw output: "),
            "no minimizer means no spill reference; got: {}",
            result.output
        );
        assert!(
            !dir.path().join(SPILL_DIR_RELATIVE).exists(),
            "no minimizer means no spill dir"
        );
    }

    #[tokio::test]
    async fn explicit_outside_sandbox_once_adds_audit_notice() {
        let _guard = ENV_LOCK.lock().await;
        let dir = std::env::temp_dir();
        let result = run_shell_command_cancellable(
            &dir,
            "echo hello-brokk",
            30,
            SandboxPolicy::None,
            true,
            None,
            None,
        )
        .await;
        assert!(
            result.output.starts_with(EXPLICIT_OUTSIDE_SANDBOX_NOTICE),
            "explicit outside-sandbox runs must prefix an audit notice; got: {}",
            result.output
        );
    }

    /// True if `name` resolves on PATH. Used to skip tests that rely on an
    /// optional tool (e.g. `setsid`, which stock macOS does not ship).
    async fn command_on_path(name: &str) -> bool {
        Command::new("sh")
            .arg("-c")
            .arg(format!("command -v {name}"))
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await
            .map(|status| status.success())
            .unwrap_or(false)
    }

    /// Regression test for the unbounded-drain hang: a grandchild that
    /// detaches into its own session via `setsid` escapes the shell's
    /// process group, so the SIGKILL we send to that group never reaches it
    /// and it keeps the inherited stdout pipe open. If the cancel path
    /// `await`ed the pipe readers, `read_to_end` would never see EOF and the
    /// call would hang forever (there is no outer timeout around shell
    /// cancellation). `abort()`ing the readers keeps cancellation prompt.
    #[tokio::test]
    async fn cancellation_returns_even_when_grandchild_escapes_process_group() {
        let _guard = ENV_LOCK.lock().await;

        if !command_on_path("setsid").await {
            eprintln!("skipping: `setsid` not available on this host");
            return;
        }

        let dir = std::env::temp_dir();
        // The outer shell backgrounds a setsid grandchild (new session, so it
        // outlives the process-group kill while still holding the stdout pipe)
        // and then blocks itself so cancellation -- not natural exit -- ends
        // the call.
        let command = "setsid sh -c 'sleep 10' & sleep 10";

        let cancel = CancellationToken::new();
        let cancel_from_task = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            cancel_from_task.cancel();
        });

        let started = std::time::Instant::now();
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            run_shell_command_cancellable(
                &dir,
                command,
                10,
                SandboxPolicy::None,
                false,
                Some(&cancel),
                None,
            ),
        )
        .await
        .expect("cancellation must return even when a grandchild keeps the pipe open");

        assert!(
            matches!(result.status, ToolStatus::RequestError),
            "cancelled command should report a request error; got: {}",
            result.output
        );
        assert!(
            result.output.contains("cancelled"),
            "cancelled command should explain cancellation; got: {}",
            result.output
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "cancellation waited too long: {:?}",
            started.elapsed()
        );
    }
}

/// Timeout-resolution tests. Separate from the `unix`-gated module above
/// because this arithmetic is platform-independent, and because every case
/// that depends on the deployment cap must be serialized against the cases
/// that override it.
#[cfg(test)]
mod timeout_tests {
    use super::*;
    use std::sync::Mutex;

    /// Serializes tests that mutate `ANVIL_SHELL_TIMEOUT_CAP_SECONDS`. Every
    /// test here sets the var explicitly (empty string == "unset, use the
    /// default cap") so an ambient value in the developer's environment
    /// cannot change the expected numbers either.
    static CAP_ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Sets `ANVIL_SHELL_TIMEOUT_CAP_SECONDS` and removes it on drop. Pair
    /// with `CAP_ENV_LOCK`; the guard must be dropped before the lock.
    struct CapEnvGuard;

    impl CapEnvGuard {
        fn set(value: &str) -> Self {
            // SAFETY: the caller holds CAP_ENV_LOCK, which serializes
            // mutation of this var across the crate's tests.
            unsafe {
                std::env::set_var(TIMEOUT_CAP_ENV, value);
            }
            Self
        }
    }

    impl Drop for CapEnvGuard {
        fn drop(&mut self) {
            // SAFETY: same as set() -- guarded by CAP_ENV_LOCK.
            unsafe {
                std::env::remove_var(TIMEOUT_CAP_ENV);
            }
        }
    }

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        CAP_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn omitted_timeout_uses_the_two_minute_default() {
        let _lock = lock();
        let _env = CapEnvGuard::set("");

        let timeout = ShellTimeout::resolve(None, None);
        assert_eq!(timeout.seconds(), DEFAULT_TIMEOUT_SECONDS);
        assert_eq!(timeout.seconds(), 120);
        assert_eq!(timeout.clamp_notice(), None);
    }

    #[test]
    fn timeout_seconds_below_the_floor_is_raised_with_a_notice() {
        let _lock = lock();
        let _env = CapEnvGuard::set("");

        let timeout = ShellTimeout::resolve(Some(3), None);
        assert_eq!(timeout.seconds(), MIN_TIMEOUT_SECONDS);
        assert_eq!(
            timeout.clamp_notice(),
            Some("Notice: requested timeout 3s was below the 10s minimum; raised to 10s.")
        );
    }

    #[test]
    fn timeout_seconds_above_the_cap_is_clamped_with_a_notice() {
        let _lock = lock();
        let _env = CapEnvGuard::set("");

        let timeout = ShellTimeout::resolve(Some(4000), None);
        assert_eq!(timeout.seconds(), MAX_TIMEOUT_SECONDS);
        assert_eq!(timeout.seconds(), 3600);
        assert_eq!(
            timeout.clamp_notice(),
            Some("Notice: requested timeout 4000s exceeded the server maximum; clamped to 3600s.")
        );
    }

    #[test]
    fn timeout_seconds_inside_the_range_passes_through_unannounced() {
        let _lock = lock();
        let _env = CapEnvGuard::set("");

        for requested in [MIN_TIMEOUT_SECONDS, 120, MAX_TIMEOUT_SECONDS] {
            let timeout = ShellTimeout::resolve(Some(requested), None);
            assert_eq!(timeout.seconds(), requested);
            assert_eq!(timeout.clamp_notice(), None, "requested={requested}");
        }
    }

    /// The unadvertised millisecond field keeps its exact pre-`timeout_seconds`
    /// behavior -- round up to whole seconds, 1s floor -- so replayed traces
    /// and in-process callers that still pass milliseconds are unaffected.
    /// Only the ceiling moved (600s -> 3600s).
    #[test]
    fn legacy_millis_round_up_and_keep_the_one_second_floor() {
        let _lock = lock();
        let _env = CapEnvGuard::set("");

        for (millis, expected) in [
            (0_u64, 1_u64),
            (1, 1),
            (999, 1),
            (1_000, 1),
            (1_500, 2),
            (60_000, 60),
            (120_000, 120),
            (601_000, 601),
        ] {
            let timeout = ShellTimeout::resolve(None, Some(millis));
            assert_eq!(timeout.seconds(), expected, "millis={millis}");
            assert_eq!(timeout.clamp_notice(), None, "millis={millis}");
        }
    }

    #[test]
    fn legacy_millis_are_capped_but_never_floored_to_the_new_minimum() {
        let _lock = lock();
        let _env = CapEnvGuard::set("");

        let capped = ShellTimeout::resolve(None, Some(3_601_000));
        assert_eq!(capped.seconds(), MAX_TIMEOUT_SECONDS);
        assert_eq!(
            capped.clamp_notice(),
            Some("Notice: requested timeout 3601s exceeded the server maximum; clamped to 3600s.")
        );
    }

    #[test]
    fn timeout_seconds_wins_when_both_fields_are_present() {
        let _lock = lock();
        let _env = CapEnvGuard::set("");

        let timeout = ShellTimeout::resolve(Some(300), Some(1_000));
        assert_eq!(timeout.seconds(), 300);
        assert_eq!(timeout.clamp_notice(), None);

        // Even when the seconds value is the one that needs clamping.
        let floored = ShellTimeout::resolve(Some(3), Some(600_000));
        assert_eq!(floored.seconds(), MIN_TIMEOUT_SECONDS);
    }

    #[test]
    fn deployment_cap_env_lowers_the_ceiling_for_every_path() {
        let _lock = lock();
        let _env = CapEnvGuard::set("30");

        assert_eq!(timeout_cap_seconds(), 30);

        let requested = ShellTimeout::resolve(Some(600), None);
        assert_eq!(requested.seconds(), 30);
        assert_eq!(
            requested.clamp_notice(),
            Some("Notice: requested timeout 600s exceeded the server maximum; clamped to 30s.")
        );

        let legacy = ShellTimeout::resolve(None, Some(600_000));
        assert_eq!(legacy.seconds(), 30);

        // The default is lowered silently: the model asked for nothing, so
        // there is no request to report a clamp against.
        let defaulted = ShellTimeout::resolve(None, None);
        assert_eq!(defaulted.seconds(), 30);
        assert_eq!(defaulted.clamp_notice(), None);
    }

    #[test]
    fn invalid_or_absent_cap_env_falls_back_to_the_default_ceiling() {
        assert_eq!(parse_timeout_cap(None), MAX_TIMEOUT_SECONDS);
        assert_eq!(parse_timeout_cap(Some("")), MAX_TIMEOUT_SECONDS);
        assert_eq!(parse_timeout_cap(Some("   ")), MAX_TIMEOUT_SECONDS);
        assert_eq!(parse_timeout_cap(Some("abc")), MAX_TIMEOUT_SECONDS);
        assert_eq!(parse_timeout_cap(Some("-5")), MAX_TIMEOUT_SECONDS);
        assert_eq!(parse_timeout_cap(Some("0")), MAX_TIMEOUT_SECONDS);
        assert_eq!(parse_timeout_cap(Some(" 45 ")), 45);
        // A cap above the compiled-in default is honored too: the env var is
        // a deployment decision, not a second maximum.
        assert_eq!(parse_timeout_cap(Some("7200")), 7200);
    }
}

/// Pure minimizer/spill tests -- no process spawning, so they run on every
/// platform (the spawn-based coverage above is unix-gated).
#[cfg(test)]
mod minimizer_tests {
    use super::*;

    const GIT_STATUS_LONG: &str = "On branch main\n\n\
        Untracked files:\n  (use \"git add <file>...\" to include in what will be committed)\n\
        \tchanged.txt\n\n\
        nothing added to commit but untracked files present (use \"git add\" to track)\n";

    #[test]
    fn minimize_condenses_git_status_and_spills() {
        let dir = tempfile::tempdir().expect("tempdir");
        let minimizer = ShellMinimizer::new(dir.path());

        let text = minimizer
            .minimize("git status", GIT_STATUS_LONG, "", 0)
            .expect("git status long format should be condensed");

        let reference_start = text
            .find("[raw output: ")
            .unwrap_or_else(|| panic!("expected raw-output reference; got: {text}"));
        let reference = &text[reference_start + "[raw output: ".len()..];
        let reference = &reference[..reference.find(']').expect("closing bracket")];
        assert!(
            reference.starts_with(SPILL_DIR_RELATIVE),
            "reference must be session-relative; got: {reference}"
        );
        let spilled =
            std::fs::read_to_string(dir.path().join(reference)).expect("spill file readable");
        assert_eq!(spilled, GIT_STATUS_LONG, "spill must be the raw capture");
        assert_eq!(
            std::fs::read_to_string(dir.path().join(SPILL_DIR_RELATIVE).join(".gitignore"))
                .expect("spill .gitignore"),
            "*\n"
        );
    }

    #[test]
    fn minimize_passes_through_unknown_commands() {
        let dir = tempfile::tempdir().expect("tempdir");
        let minimizer = ShellMinimizer::new(dir.path());
        assert!(
            minimizer
                .minimize("frobnicate --xyz", "some output\n", "", 0)
                .is_none(),
            "unknown command must pass through verbatim"
        );
        assert!(
            !dir.path().join(SPILL_DIR_RELATIVE).exists(),
            "passthrough must not create a spill dir"
        );
    }

    #[test]
    fn spill_failure_falls_back_to_minimized_without_reference() {
        let dir = tempfile::tempdir().expect("tempdir");
        // `.brokk` as a *file* makes create_dir_all fail deterministically.
        std::fs::write(dir.path().join(".brokk"), "not a dir").expect("write blocker");
        let minimizer = ShellMinimizer::new(dir.path());

        let text = minimizer
            .minimize("git status", GIT_STATUS_LONG, "", 0)
            .expect("minimization itself must still happen");
        assert!(
            !text.contains("[raw output: "),
            "failed spill must omit the reference; got: {text}"
        );
    }

    #[test]
    fn cleanup_removes_only_stale_spill_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let spill_dir = dir.path().join(SPILL_DIR_RELATIVE);
        std::fs::create_dir_all(&spill_dir).expect("create spill dir");
        std::fs::write(spill_dir.join(".gitignore"), "*\n").expect("gitignore");
        std::fs::write(spill_dir.join("fresh.txt"), "fresh").expect("fresh");
        let stale_path = spill_dir.join("stale.txt");
        std::fs::write(&stale_path, "stale").expect("stale");
        let stale_mtime = std::time::SystemTime::now() - (STALE_SPILL_FILE_AGE * 2);
        std::fs::File::options()
            .write(true)
            .open(&stale_path)
            .expect("open stale")
            .set_modified(stale_mtime)
            .expect("backdate stale");

        cleanup_stale_shell_outputs(dir.path());

        assert!(!stale_path.exists(), "stale spill file must be removed");
        assert!(spill_dir.join("fresh.txt").exists(), "fresh file must stay");
        assert!(
            spill_dir.join(".gitignore").exists(),
            ".gitignore must survive GC"
        );
    }
}
