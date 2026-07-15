use super::sandbox::{self, ENV_WHITELIST, SandboxPolicy};
use super::{ToolResult, ToolStatus};
use std::io;
use std::path::Path;
use std::process::ExitStatus;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

const MAX_OUTPUT_BYTES: usize = 100_000; // 100KB
pub(super) const MAX_TIMEOUT_SECONDS: u64 = 600;
#[cfg(not(target_os = "windows"))]
const ANVIL_RTK_DISABLED_ENV: &str = "ANVIL_RTK_DISABLED";
#[cfg(not(target_os = "windows"))]
const ANVIL_RTK_PROXY_PREFIX_ENV: &str = "ANVIL_RTK_PROXY_PREFIX";
#[cfg(not(target_os = "windows"))]
const RTK_HOSTED_ENV: &str = "RTK_HOSTED";
#[cfg(not(target_os = "windows"))]
const RTK_TELEMETRY_DISABLED_ENV: &str = "RTK_TELEMETRY_DISABLED";
#[cfg(not(target_os = "windows"))]
const RTK_TRACKING_DISABLED_ENV: &str = "RTK_TRACKING_DISABLED";
#[cfg(not(target_os = "windows"))]
const ANVIL_RTK_SUBCOMMAND: &str = "__rtk";
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

fn format_shell_tool_result(
    stdout: &str,
    stderr: &str,
    exit_code: i32,
    success: bool,
    outside_sandbox_once: bool,
    bypass_warning: bool,
    timeout_clamp_notice: Option<&str>,
) -> ToolResult {
    let mut combined = String::new();
    if let Some(notice) = timeout_clamp_notice {
        combined.push_str(notice);
        combined.push_str("\n\n");
    }
    if !stdout.is_empty() {
        combined.push_str(stdout);
    }

    if !stderr.is_empty() {
        if !combined.is_empty() {
            combined.push_str("\n--- stderr ---\n");
        }
        combined.push_str(stderr);
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

    if outside_sandbox_once {
        combined = format!("{EXPLICIT_OUTSIDE_SANDBOX_NOTICE}\n\n{combined}");
    }

    if combined.len() > MAX_OUTPUT_BYTES {
        let end = crate::text::truncate_utf8(&combined, MAX_OUTPUT_BYTES).len();
        combined.truncate(end);
        combined.push_str("\n... output truncated");
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

#[cfg(not(target_os = "windows"))]
fn env_var_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| env_value_truthy(&value))
}

#[cfg(not(target_os = "windows"))]
fn rtk_disabled_in_command_prefix(command: &str) -> bool {
    for token in command.split_whitespace() {
        if token.contains('=') && !token.starts_with('-') {
            if let Some(value) = token.strip_prefix("RTK_DISABLED=") {
                return env_value_truthy(value);
            }
            continue;
        }
        return false;
    }
    false
}

#[cfg(not(target_os = "windows"))]
fn env_value_truthy(value: &str) -> bool {
    matches!(
        value
            .trim_matches(|c| matches!(c, '\'' | '"'))
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(target_os = "windows")]
fn rtk_rewritten_command(command: &str) -> String {
    command.to_string()
}

#[cfg(not(target_os = "windows"))]
fn rtk_rewritten_command(command: &str) -> String {
    // RTK's native Windows support is CLI/filtering-oriented; its automatic
    // rewrite hook path requires Unix shell semantics. Anvil's hosted rewrite
    // emits POSIX env-prefix syntax, so leave native Windows shell commands
    // untouched. WSL is a Linux target and still takes the rewrite path.
    if env_var_truthy(ANVIL_RTK_DISABLED_ENV)
        || rtk_disabled_in_command_prefix(command)
        || command.contains(ANVIL_RTK_SUBCOMMAND)
    {
        return command.to_string();
    }

    let proxy = if let Some(value) = std::env::var(ANVIL_RTK_PROXY_PREFIX_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        value
    } else {
        match std::env::current_exe() {
            Ok(path) => format!("{} {}", shell_quote_path(&path), ANVIL_RTK_SUBCOMMAND),
            Err(e) => {
                tracing::warn!("could not resolve current executable for RTK rewrite: {e}");
                return command.to_string();
            }
        }
    };
    let proxy = format!(
        "{RTK_HOSTED_ENV}=1 {RTK_TELEMETRY_DISABLED_ENV}=1 {RTK_TRACKING_DISABLED_ENV}=1 {proxy}"
    );

    match rtk_core::rewrite_command_with_proxy(command, &[], &[], &proxy) {
        Some(rewritten) => {
            tracing::debug!(
                target: "brokk_acp_rust::tools::shell",
                original = %command,
                rewritten = %rewritten,
                "rewrote shell command through hosted RTK",
            );
            rewritten
        }
        None => command.to_string(),
    }
}

#[cfg(not(target_os = "windows"))]
fn shell_quote_path(path: &Path) -> String {
    let raw = path.to_string_lossy();
    format!("'{}'", raw.replace('\'', "'\\''"))
}

pub async fn run_shell_command_cancellable(
    cwd: &Path,
    command: &str,
    timeout_seconds: u64,
    policy: SandboxPolicy,
    outside_sandbox_once: bool,
    cancel: Option<&CancellationToken>,
) -> ToolResult {
    if command.trim().is_empty() {
        return ToolResult {
            status: ToolStatus::RequestError,
            output: "Command must not be empty".to_string(),
        };
    }
    let requested_timeout_seconds = timeout_seconds.max(1);
    let timeout_seconds = requested_timeout_seconds.min(MAX_TIMEOUT_SECONDS);
    let timeout_clamp_notice = (requested_timeout_seconds != timeout_seconds).then(|| {
        format!(
            "Notice: requested timeout {requested_timeout_seconds}s exceeded the server maximum; clamped to {timeout_seconds}s."
        )
    });

    let command_to_run = rtk_rewritten_command(command);

    // Wrap once. `wrapped` owns the temp policy file (Seatbelt) and must
    // outlive the spawned child.
    let wrapped = match sandbox::wrap_command(policy, cwd, &command_to_run) {
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
            format_shell_tool_result(
                &stdout,
                &stderr,
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

    fn fake_rtk_proxy(dir: &Path) -> PathBuf {
        let path = dir.join("fake-rtk-proxy.sh");
        std::fs::write(
            &path,
            "#!/bin/sh\n\
             if [ \"$1\" = cargo ] && [ \"$2\" = test ]; then\n\
               echo hosted-rtk-cargo-test\n\
               exit 0\n\
             fi\n\
             if [ \"$1\" = git ] && [ \"$2\" = status ]; then\n\
               echo '## main'\n\
               echo '?? changed.txt'\n\
               exit 0\n\
             fi\n\
             echo \"unexpected hosted rtk args: $*\" >&2\n\
             exit 42\n",
        )
        .expect("write fake RTK proxy");
        let mut perms = std::fs::metadata(&path)
            .expect("stat fake RTK proxy")
            .permissions();
        use std::os::unix::fs::PermissionsExt;
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).expect("chmod fake RTK proxy");
        path
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

        let result = format_shell_tool_result(&stdout, "", 0, true, false, false, None);

        assert_eq!(
            result.output,
            format!("{}\n... output truncated", "a".repeat(MAX_OUTPUT_BYTES - 1))
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
            run_shell_command_cancellable(&dir, &cmd, 30, SandboxPolicy::None, false, None).await;
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
            run_shell_command_cancellable(&dir, &cmd, 30, SandboxPolicy::None, false, None).await;
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

    #[test]
    fn rtk_disabled_prefix_requires_truthy_value() {
        assert!(rtk_disabled_in_command_prefix("RTK_DISABLED=1 cargo test"));
        assert!(rtk_disabled_in_command_prefix(
            "FOO=bar RTK_DISABLED=true git status"
        ));
        assert!(rtk_disabled_in_command_prefix(
            "RTK_DISABLED='yes' git status"
        ));
        assert!(!rtk_disabled_in_command_prefix("RTK_DISABLED=0 cargo test"));
        assert!(!rtk_disabled_in_command_prefix(
            "RTK_DISABLED=false cargo test"
        ));
        assert!(!rtk_disabled_in_command_prefix("echo RTK_DISABLED=1"));
    }

    #[tokio::test]
    async fn anvil_rtk_disabled_env_requires_truthy_value() {
        let _guard = ENV_LOCK.lock().await;
        let dir = tempfile::tempdir().expect("create temp project");
        let proxy = fake_rtk_proxy(dir.path());
        let _proxy_env = EnvGuard::set(ANVIL_RTK_PROXY_PREFIX_ENV, &shell_quote_path(&proxy));
        let _disabled_env = EnvGuard::set(ANVIL_RTK_DISABLED_ENV, "0");

        let result = run_shell_command_cancellable(
            dir.path(),
            "git status",
            30,
            SandboxPolicy::None,
            false,
            None,
        )
        .await;

        assert!(
            result.output.contains("## main"),
            "ANVIL_RTK_DISABLED=0 should not disable hosted RTK rewrite; got: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn anvil_rtk_disabled_env_truthy_skips_rewrite() {
        let _guard = ENV_LOCK.lock().await;
        let dir = tempfile::tempdir().expect("create temp project");
        let proxy = fake_rtk_proxy(dir.path());
        let _proxy_env = EnvGuard::set(ANVIL_RTK_PROXY_PREFIX_ENV, &shell_quote_path(&proxy));
        let _disabled_env = EnvGuard::set(ANVIL_RTK_DISABLED_ENV, "1");

        let result = run_shell_command_cancellable(
            dir.path(),
            "git status",
            30,
            SandboxPolicy::None,
            false,
            None,
        )
        .await;

        assert!(
            !result.output.contains("## main"),
            "ANVIL_RTK_DISABLED=1 should skip hosted RTK rewrite; got: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn run_shell_command_rewrites_to_hosted_rtk_for_cargo_test() {
        let _guard = ENV_LOCK.lock().await;
        let dir = tempfile::tempdir().expect("create temp cargo project");
        let proxy = fake_rtk_proxy(dir.path());
        let _proxy_env = EnvGuard::set(ANVIL_RTK_PROXY_PREFIX_ENV, &shell_quote_path(&proxy));
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"rtk-filter-demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .expect("write Cargo.toml");
        std::fs::create_dir(dir.path().join("src")).expect("create src");
        std::fs::write(
            dir.path().join("src/lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\
             #[cfg(test)]\n\
             mod tests {\n\
                 #[test]\n\
                 fn adds() { assert_eq!(super::add(1, 2), 3); }\n\
             }\n",
        )
        .expect("write lib.rs");

        let result = run_shell_command_cancellable(
            dir.path(),
            "cargo test --quiet",
            120,
            SandboxPolicy::None,
            false,
            None,
        )
        .await;

        assert!(
            matches!(result.status, ToolStatus::Success),
            "cargo test must succeed; got: {}",
            result.output
        );
        assert!(
            result.output.contains("hosted-rtk-cargo-test"),
            "cargo test should be routed through the hosted RTK proxy; got: {}",
            result.output
        );
        assert!(
            !result.output.contains("running 1 test"),
            "native cargo test output means the rewrite did not run; got: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn run_shell_command_rewrites_to_hosted_rtk_for_git_status() {
        let _guard = ENV_LOCK.lock().await;
        let dir = tempfile::tempdir().expect("create temp git repo");
        let proxy = fake_rtk_proxy(dir.path());
        let _proxy_env = EnvGuard::set(ANVIL_RTK_PROXY_PREFIX_ENV, &shell_quote_path(&proxy));
        let init = run_shell_command_cancellable(
            dir.path(),
            "git init",
            30,
            SandboxPolicy::None,
            false,
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
        )
        .await;

        assert!(
            matches!(result.status, ToolStatus::Success),
            "git status must succeed; got: {}",
            result.output
        );
        assert!(
            result.output.contains("changed.txt") || result.output.contains("??"),
            "hosted RTK git status output should mention the untracked file; got: {}",
            result.output
        );
        assert!(
            !result.output.contains("On branch"),
            "native git status output means the rewrite did not run; got: {}",
            result.output
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
