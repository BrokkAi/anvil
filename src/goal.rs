use std::time::Duration;

use crate::slash::{is_slash_command, slash_command_args};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoopSpec {
    pub(crate) interval_secs: u64,
    pub(crate) target: String,
}

pub(crate) fn loop_target_runs_without_model(target: &str) -> bool {
    is_slash_command(target, "context")
        || is_slash_command(target, "setup")
        || is_slash_command(target, "permissions")
        || is_slash_command(target, "mcp")
        || is_slash_command(target, "pr-create")
        || is_slash_command(target, "usage")
        || is_slash_command(target, "fast")
        || is_slash_command(target, "rewind")
}

pub(crate) fn parse_loop_command(prompt_text: &str) -> Result<LoopSpec, String> {
    let trimmed = slash_command_args(prompt_text);
    if trimmed.is_empty() {
        return Err("Usage: `/loop <seconds> <slash-command-or-prompt>`\n\
             Example: `/loop 30 /context`\n\
             Example: `/loop 300 check CI status`\n\n\
             The loop runs until you cancel the session."
            .to_string());
    }

    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let raw_secs = parts.next().unwrap_or("");
    let target = parts.next().unwrap_or("").trim();
    if target.is_empty() {
        return Err("Usage: `/loop <seconds> <slash-command-or-prompt>`\n\
             Missing command or prompt after the interval."
            .to_string());
    }
    if is_slash_command(target, "loop") {
        return Err("Nested `/loop` is not supported.".to_string());
    }

    let interval_secs = match raw_secs.parse::<u64>() {
        Ok(secs) if (1..=86_400).contains(&secs) => secs,
        Ok(other) => {
            return Err(format!(
                "Interval `{other}` is out of range. Pick a value between 1 and 86400 seconds."
            ));
        }
        Err(_) => {
            return Err(format!(
                "Invalid interval `{raw_secs}`. Usage: `/loop <seconds> <slash-command-or-prompt>`"
            ));
        }
    };

    Ok(LoopSpec {
        interval_secs,
        target: target.to_string(),
    })
}

/// Bounds for the *optional* `--max-turns` guardrail. A goal is unbounded by
/// default: the stopping condition is the model's verified completion or a
/// genuine block, not an arbitrary turn count -- a turn cap that fired on its
/// own would stop the agent before the goal is met, defeating the purpose.
/// (This matches Codex, whose token budget is `Option` and defaults to none.)
/// `--max-turns` only applies when the user explicitly opts into a ceiling,
/// and then must fall in this range.
const GOAL_MIN_MAX_TURNS: u32 = 1;
const GOAL_MAX_MAX_TURNS: u32 = 10_000;

/// Sentinel the model emits, alone on the final line, once it has verified
/// the objective is complete. Detected by [`detect_goal_signal`].
const GOAL_COMPLETE_SENTINEL: &str = "GOAL_COMPLETE";
/// Sentinel prefix the model emits when genuinely at an impasse.
const GOAL_BLOCKED_SENTINEL: &str = "GOAL_BLOCKED";
/// How many consecutive blocked reports are required before the loop stops
/// and hands back to the user. Mirrors Codex's three-turn blocked rule so
/// the agent doesn't surrender on a transient blocker.
const GOAL_BLOCKED_THRESHOLD: u32 = 3;

/// A parsed `/goal` invocation. `max_turns` is `None` for an unbounded goal
/// (the default) and `Some(n)` only when the user opts into a ceiling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GoalSpec {
    pub(crate) objective: String,
    pub(crate) max_turns: Option<u32>,
}

/// Which framing the continuation prompt uses for a given turn.
#[derive(Clone, Copy)]
pub(crate) enum GoalPhase {
    /// A normal continuation turn: make verifiable progress.
    Continue,
    /// The last turn of an opt-in `--max-turns` ceiling: wrap up cleanly and
    /// summarize. Never used for an unbounded goal.
    FinalWrapUp,
}

/// The stop signal (if any) parsed from a goal turn's assistant text.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum GoalSignal {
    Complete,
    Blocked(String),
    Continue,
}

fn goal_usage() -> String {
    "Usage: `/goal [--max-turns N] <objective>`\n\
     Example: `/goal make `cargo test` pass`\n\
     Example: `/goal --max-turns 40 migrate the config loader to serde`\n\n\
     Anvil works autonomously across turns until the objective is verifiably met or \
     it is blocked -- there is no turn limit by default. Cancel the session to stop \
     early, or pass `--max-turns N` to set an optional ceiling."
        .to_string()
}

/// Parse `/goal [--max-turns N] <objective>`.
///
/// `--max-turns` (also `--max-turns=N`) is optional and, when present, must
/// lead; without it the goal is unbounded (`max_turns: None`). The remainder
/// is the free-text objective. An empty objective or an out-of-range ceiling
/// is a user error returned as a usage string.
pub(crate) fn parse_goal_command(prompt_text: &str) -> Result<GoalSpec, String> {
    let args = slash_command_args(prompt_text);
    let trimmed = args.trim();
    if trimmed.is_empty() {
        return Err(goal_usage());
    }

    let mut max_turns: Option<u32> = None;
    let mut rest = trimmed;
    loop {
        let head = rest.trim_start();
        let Some(after) = head.strip_prefix("--max-turns") else {
            break;
        };
        // Only treat this as the flag when `--max-turns` is a whole token
        // (followed by `=`, whitespace, or end) -- otherwise it's the start
        // of the objective and we leave it alone.
        if !(after.is_empty() || after.starts_with('=') || after.starts_with(char::is_whitespace)) {
            break;
        }
        let after = after.trim_start_matches('=').trim_start();
        let mut parts = after.splitn(2, char::is_whitespace);
        let raw = parts.next().unwrap_or("");
        let remainder = parts.next().unwrap_or("");
        let n = raw.parse::<u32>().map_err(|_| {
            format!(
                "Invalid `--max-turns` value `{raw}`. Pick an integer between \
                 {GOAL_MIN_MAX_TURNS} and {GOAL_MAX_MAX_TURNS}."
            )
        })?;
        if !(GOAL_MIN_MAX_TURNS..=GOAL_MAX_MAX_TURNS).contains(&n) {
            return Err(format!(
                "`--max-turns` {n} is out of range. Pick a value between \
                 {GOAL_MIN_MAX_TURNS} and {GOAL_MAX_MAX_TURNS}."
            ));
        }
        max_turns = Some(n);
        rest = remainder;
    }

    let objective = rest.trim().to_string();
    if objective.is_empty() {
        return Err(goal_usage());
    }

    Ok(GoalSpec {
        objective,
        max_turns,
    })
}

/// Strip surrounding markdown emphasis, quoting, and trailing punctuation
/// from a single line so a sentinel the model lightly decorated (e.g.
/// `` `GOAL_COMPLETE` `` or `**GOAL_COMPLETE.**`) still matches.
fn normalize_sentinel_line(raw: &str) -> String {
    raw.trim()
        .trim_matches(|c: char| {
            matches!(
                c,
                '`' | '*' | '_' | '#' | '"' | '\'' | '.' | '!' | ':' | ' '
            )
        })
        .trim()
        .to_string()
}

/// Inspect a goal turn's assistant text for a stop signal.
///
/// A standalone line equal to [`GOAL_COMPLETE_SENTINEL`] wins outright. A
/// standalone line starting with [`GOAL_BLOCKED_SENTINEL`] reports a
/// blocker (the trailing text, if any, is the reason). Anything else means
/// keep going. Only whole-line matches count, so the agent can discuss the
/// sentinels in prose without accidentally tripping a stop.
pub(crate) fn detect_goal_signal(response: &str) -> GoalSignal {
    let mut blocked: Option<String> = None;
    for raw in response.lines() {
        let line = normalize_sentinel_line(raw);
        if line == GOAL_COMPLETE_SENTINEL {
            return GoalSignal::Complete;
        }
        if let Some(reason) = line.strip_prefix(GOAL_BLOCKED_SENTINEL) {
            let reason = reason.trim_start_matches([':', '-', ' ']).trim();
            blocked = Some(reason.to_string());
        }
    }
    match blocked {
        Some(reason) => GoalSignal::Blocked(reason),
        None => GoalSignal::Continue,
    }
}

/// Why a goal loop stopped. Carried out of [`decide_after_goal_turn`] so the
/// caller can render the right user-facing message.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum GoalStop {
    /// The model verified the objective and emitted the completion sentinel.
    Completed,
    /// The model reported a blocker on [`GOAL_BLOCKED_THRESHOLD`] consecutive
    /// turns (reasons need not match); carries the latest reason.
    Blocked(String),
    /// The user's opt-in `--max-turns` ceiling was reached without a
    /// completion signal. Never produced for an unbounded goal.
    CeilingReached,
}

/// What the goal loop should do after a turn completes.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum GoalStep {
    /// Run another turn. `consecutive_blocked` is the updated counter to
    /// carry forward (non-zero means the just-finished turn reported a
    /// blocker that has not yet hit the threshold).
    Continue { consecutive_blocked: u32 },
    /// Stop the loop with this disposition.
    Stop(GoalStop),
}

/// Pure control logic for the goal loop: given the signal parsed from a
/// turn and the loop's counters, decide whether to continue or stop.
///
/// Cancellation and terminal errors are handled by the caller; they
/// pre-empt this decision. Kept side-effect-free (no `cx`, no LLM) so the
/// branching -- completion-wins-over-ceiling, the consecutive-blocked
/// threshold and its reset, and the optional ceiling -- is unit-testable
/// and the runtime loop and tests share one source of truth.
pub(crate) fn decide_after_goal_turn(
    signal: GoalSignal,
    turn: u32,
    max_turns: Option<u32>,
    consecutive_blocked: u32,
) -> GoalStep {
    let ceiling_reached = max_turns.is_some_and(|max| turn >= max);
    match signal {
        // A verified completion wins even on the final allowed turn.
        GoalSignal::Complete => GoalStep::Stop(GoalStop::Completed),
        GoalSignal::Blocked(reason) => {
            let blocked = consecutive_blocked + 1;
            if blocked >= GOAL_BLOCKED_THRESHOLD {
                GoalStep::Stop(GoalStop::Blocked(reason))
            } else if ceiling_reached {
                GoalStep::Stop(GoalStop::CeilingReached)
            } else {
                GoalStep::Continue {
                    consecutive_blocked: blocked,
                }
            }
        }
        GoalSignal::Continue => {
            if ceiling_reached {
                GoalStep::Stop(GoalStop::CeilingReached)
            } else {
                GoalStep::Continue {
                    consecutive_blocked: 0,
                }
            }
        }
    }
}

fn render_goal_stop(stop: &GoalStop, turn: u32, max_turns: Option<u32>) -> String {
    match stop {
        GoalStop::Completed => format!(
            "\n✅ Goal achieved in {turn} turn(s): the agent reported the \
             objective verifiably complete.\n"
        ),
        GoalStop::Blocked(reason) => format!(
            "\n⛔ Goal blocked after {turn} turn(s) \
             ({GOAL_BLOCKED_THRESHOLD} consecutive blocked reports). \
             Stopping for user input.\nReason: {reason}\n"
        ),
        GoalStop::CeilingReached => format!(
            "\n🛑 Goal stopped: reached the opt-in {}-turn ceiling without a \
             completion signal. Review the progress above and re-run `/goal` \
             (raise or drop `--max-turns`) to keep going.\n",
            max_turns.unwrap_or(turn)
        ),
    }
}

/// Why a goal run ended. Every `run_goal_loop` break site carries one of
/// these plus the number of goal turns that ran, so a single pure function
/// ([`render_goal_exit`]) owns all exit wording -- the live stop message
/// and the aggregate recap's Stop line + detail -- instead of six break
/// sites assembling strings by hand.
pub(crate) enum GoalExit {
    /// Sentinel-driven stop: completed, blocked, or turn ceiling.
    Stop(GoalStop),
    Cancelled,
    /// The model request failed and cannot be retried.
    FatalFailure(crate::tool_loop::TurnFailure),
    /// The turn pipeline failed before a model turn could run.
    Terminal(String),
}

/// Everything user-visible about one goal exit, from one match.
pub(crate) struct GoalExitText {
    /// Streamed to the transcript when the loop ends.
    pub(crate) user_message: String,
    /// The aggregate recap's `Stop:` line (host-authored, single line).
    pub(crate) recap_stop_line: String,
    /// Optional recap detail paragraph: blocked reason, failure message,
    /// or remaining-work note.
    pub(crate) recap_detail: Option<String>,
}

/// Render every user-visible string for a goal exit. Pure so the wording
/// is unit-testable without driving the loop; the sentinel-driven arm
/// reuses [`render_goal_stop`] and [`goal_recap_parts`] so the historical
/// message wording is unchanged.
pub(crate) fn render_goal_exit(
    exit: &GoalExit,
    turns_ran: u32,
    max_turns: Option<u32>,
) -> GoalExitText {
    match exit {
        GoalExit::Stop(stop) => {
            let (recap_stop_line, recap_detail) = goal_recap_parts(stop, turns_ran, max_turns);
            GoalExitText {
                user_message: render_goal_stop(stop, turns_ran, max_turns),
                recap_stop_line,
                recap_detail,
            }
        }
        GoalExit::Cancelled => GoalExitText {
            user_message: "Goal cancelled.\n".to_string(),
            recap_stop_line: format!("goal cancelled after {turns_ran} goal turn(s)"),
            recap_detail: None,
        },
        GoalExit::FatalFailure(failure) => GoalExitText {
            user_message: format!(
                "\n⛔ Goal stopped after {turns_ran} turn(s): the model request \
                 failed and cannot be retried.\nReason: {}\n",
                failure.message
            ),
            recap_stop_line: format!(
                "goal stopped after {turns_ran} goal turn(s) on a fatal model failure"
            ),
            recap_detail: Some(failure.message.clone()),
        },
        GoalExit::Terminal(err) => GoalExitText {
            user_message: format!("\nGoal stopped: {err}\n"),
            recap_stop_line: format!(
                "goal stopped after {turns_ran} goal turn(s) on a fatal error"
            ),
            recap_detail: Some(err.clone()),
        },
    }
}

/// Host-authored Stop line + optional detail paragraph for the aggregate
/// goal recap, for the sentinel-driven stops. Pure so the wording can be
/// unit-tested like [`render_goal_stop`].
fn goal_recap_parts(
    stop: &GoalStop,
    turn: u32,
    max_turns: Option<u32>,
) -> (String, Option<String>) {
    match stop {
        GoalStop::Completed => (format!("goal achieved after {turn} goal turn(s)"), None),
        GoalStop::Blocked(reason) => (
            format!("goal blocked after {turn} goal turn(s)"),
            Some(format!("Blocked: {reason}")),
        ),
        GoalStop::CeilingReached => (
            format!(
                "goal stopped at the opt-in {}-turn ceiling",
                max_turns.unwrap_or(turn)
            ),
            Some(
                "The objective did not report completion before the turn ceiling; \
                 see the final wrap-up turn above for remaining work."
                    .to_string(),
            ),
        ),
    }
}

pub(crate) fn render_blocked_progress(consecutive_blocked: u32) -> Option<String> {
    (consecutive_blocked > 0).then(|| {
        format!(
            "[goal: blocked report {consecutive_blocked}/{GOAL_BLOCKED_THRESHOLD}; \
             retrying]\n"
        )
    })
}

/// What the goal loop should do about a turn that ended in an LLM failure
/// (vs. a real model response). Kept side-effect-free so the
/// transient-vs-fatal branch is unit-testable, like [`decide_after_goal_turn`].
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum GoalFailureAction {
    /// Transient outage (server overload, rate limit, stream/connection drop):
    /// wait a backoff scaled by `consecutive_failures` and retry the turn.
    /// Unbounded by design so the goal survives a long outage and resumes when
    /// it clears -- the delay is capped instead of the retry count.
    Backoff { consecutive_failures: u32 },
    /// Fatal (auth, invalid request, panic): retrying would not help, so stop
    /// the goal and hand back to the user.
    Stop,
}

/// Classify a failed goal turn. Transient failures back off and retry; fatal
/// ones stop. Mirrors Codex's retryable/fatal split (the underlying predicate
/// is [`crate::llm_client::is_retryable_llm_error`], applied in
/// [`crate::tool_loop::run`]); the divergence is deliberate -- Codex bounds
/// transient retries by a fixed count then aborts the turn, whereas an
/// unbounded goal keeps retrying (with a capped delay) to survive the outage.
pub(crate) fn decide_after_goal_failure(
    failure: &crate::tool_loop::TurnFailure,
    consecutive_failures: u32,
) -> GoalFailureAction {
    if failure.retryable {
        GoalFailureAction::Backoff {
            consecutive_failures: consecutive_failures.saturating_add(1),
        }
    } else {
        GoalFailureAction::Stop
    }
}

/// Upper bound on the inter-turn backoff for a goal surviving an outage. The
/// base schedule is the codex-compatible [`crate::http_retry::retry_backoff`]
/// (200ms * 2^(n-1) + jitter); because a goal retries an unbounded number of
/// times, the delay is capped here so a long outage settles into a steady
/// ~1-minute poll rather than growing without limit.
const GOAL_FAILURE_BACKOFF_CAP: Duration = Duration::from_secs(60);

/// Capped exponential backoff for the `consecutive_failures`-th transient
/// failure in a row (1-based).
pub(crate) fn goal_failure_backoff(consecutive_failures: u32) -> Duration {
    crate::http_retry::retry_backoff(u64::from(consecutive_failures)).min(GOAL_FAILURE_BACKOFF_CAP)
}

/// Build the continuation prompt injected as the user message for one goal
/// turn. Adapts Codex's `continuation.md` (objective framing + completion
/// audit + blocked discipline) to Anvil's sentinel-based stop signal.
pub(crate) fn build_goal_prompt(
    objective: &str,
    turn: u32,
    max_turns: Option<u32>,
    phase: GoalPhase,
) -> String {
    let header = match max_turns {
        Some(max) => {
            format!("You are operating in autonomous goal mode (turn {turn} of at most {max}).")
        }
        None => format!("You are operating in autonomous goal mode (turn {turn})."),
    };
    let objective_block = format!("<objective>\n{}\n</objective>", objective.trim());

    let completion_protocol = format!(
        "Completion protocol:\n\
         - Treat completion as unproven. Before claiming success, derive concrete \
         requirements from the objective and verify each against the ACTUAL current state \
         of the worktree and any commands/tests it implies -- inspect file contents, command \
         output, and test results rather than relying on intent, memory, or a plausible answer.\n\
         - Keep the full objective intact; do not redefine success around a smaller or easier \
         task just to finish.\n\
         - Only when every requirement is satisfied and verified, end your message with a line \
         containing exactly:\n\
         {GOAL_COMPLETE_SENTINEL}\n\
         Put it alone on the final line, with no surrounding text, quotes, or formatting. \
         Emitting it is a claim that the full objective is done and can withstand \
         requirement-by-requirement scrutiny. If any requirement is missing, weak, indirect, \
         or unverified, do NOT emit it -- keep working."
    );

    let blocked_protocol = format!(
        "If you are genuinely at an impasse and cannot make progress without user input or an \
         external change, end your message with a line:\n\
         {GOAL_BLOCKED_SENTINEL}: <one-line reason>\n\
         Use this only when truly stuck -- never because the work is merely hard, slow, or \
         incomplete. If the same blocker persists for {GOAL_BLOCKED_THRESHOLD} consecutive \
         turns, the goal stops and hands back to the user."
    );

    match phase {
        GoalPhase::Continue => format!(
            "{header}\n\n\
             Continue working toward the objective below. This goal persists across turns, so \
             you do not need to shrink it to what fits in one turn -- make concrete, verifiable \
             progress toward the real end state.\n\n\
             {objective_block}\n\n\
             Work from evidence: treat the current worktree and command output as authoritative \
             before relying on earlier conversation. Use your tools to actually make the changes \
             -- do not just describe them. If the next work is meaningfully multi-step, keep a \
             short task list.\n\n\
             {completion_protocol}\n\n\
             {blocked_protocol}"
        ),
        GoalPhase::FinalWrapUp => format!(
            "{header}\n\n\
             This is the FINAL turn of the goal's opt-in turn ceiling. Do not start new large work. \
             Bring the current work to a safe, coherent stopping point, then summarize what was \
             accomplished, what remains, and the clear next step for the user.\n\n\
             {objective_block}\n\n\
             If -- and only if -- the objective is actually complete and verified, end with a \
             line containing exactly {GOAL_COMPLETE_SENTINEL}. Otherwise do not emit it; just \
             summarize."
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_loop_command_parses_interval_and_target() {
        assert_eq!(
            parse_loop_command("/loop 30 /context"),
            Ok(LoopSpec {
                interval_secs: 30,
                target: "/context".to_string(),
            })
        );
    }

    #[test]
    fn parse_loop_command_rejects_missing_target() {
        let err = parse_loop_command("/loop 30").expect_err("missing target must reject");
        assert!(err.contains("Missing command or prompt"), "got: {err}");
    }

    #[test]
    fn parse_loop_command_rejects_invalid_interval() {
        let err = parse_loop_command("/loop soon /context").expect_err("junk interval must reject");
        assert!(err.contains("Invalid interval"), "got: {err}");
    }

    #[test]
    fn parse_loop_command_rejects_out_of_range() {
        let err = parse_loop_command("/loop 0 /context").expect_err("zero must reject");
        assert!(err.contains("out of range"), "got: {err}");

        let err = parse_loop_command("/loop 86401 /context").expect_err("too large must reject");
        assert!(err.contains("out of range"), "got: {err}");
    }

    #[test]
    fn parse_loop_command_rejects_nested_loop() {
        let err = parse_loop_command("/loop 30 /loop 60 hi").expect_err("nested loop must reject");
        assert!(err.contains("Nested `/loop`"), "got: {err}");
    }

    #[test]
    fn parse_goal_command_is_unbounded_by_default() {
        // No `--max-turns` means no ceiling: the goal runs until it is
        // verifiably complete, blocked, or cancelled.
        assert_eq!(
            parse_goal_command("/goal make cargo test pass"),
            Ok(GoalSpec {
                objective: "make cargo test pass".to_string(),
                max_turns: None,
            })
        );
    }

    #[test]
    fn parse_goal_command_parses_max_turns_flag() {
        assert_eq!(
            parse_goal_command("/goal --max-turns 40 migrate the loader"),
            Ok(GoalSpec {
                objective: "migrate the loader".to_string(),
                max_turns: Some(40),
            })
        );
        // `=` form is equivalent.
        assert_eq!(
            parse_goal_command("/goal --max-turns=7 do the thing"),
            Ok(GoalSpec {
                objective: "do the thing".to_string(),
                max_turns: Some(7),
            })
        );
    }

    #[test]
    fn parse_goal_command_requires_objective() {
        let err = parse_goal_command("/goal").expect_err("bare /goal must reject");
        assert!(err.contains("Usage:"), "got: {err}");
        // A flag with no objective after it is still a usage error.
        let err =
            parse_goal_command("/goal --max-turns 5").expect_err("flag-only /goal must reject");
        assert!(err.contains("Usage:"), "got: {err}");
    }

    #[test]
    fn parse_goal_command_rejects_bad_max_turns() {
        let err = parse_goal_command("/goal --max-turns soon do it")
            .expect_err("junk budget must reject");
        assert!(err.contains("Invalid `--max-turns`"), "got: {err}");

        let err = parse_goal_command("/goal --max-turns 0 do it").expect_err("zero must reject");
        assert!(err.contains("out of range"), "got: {err}");

        let err =
            parse_goal_command("/goal --max-turns 99999 do it").expect_err("too large must reject");
        assert!(err.contains("out of range"), "got: {err}");
    }

    #[test]
    fn parse_goal_command_treats_lookalike_flag_as_objective() {
        // `--max-turnsy` is not the flag, so it stays part of the objective
        // and the goal stays unbounded.
        assert_eq!(
            parse_goal_command("/goal --max-turnsy is a weird objective"),
            Ok(GoalSpec {
                objective: "--max-turnsy is a weird objective".to_string(),
                max_turns: None,
            })
        );
    }

    #[test]
    fn detect_goal_signal_recognizes_complete() {
        assert_eq!(
            detect_goal_signal("All tests pass now.\n\nGOAL_COMPLETE"),
            GoalSignal::Complete
        );
        // Lightly decorated / trailing punctuation still matches.
        assert_eq!(
            detect_goal_signal("done\n`GOAL_COMPLETE`"),
            GoalSignal::Complete
        );
        assert_eq!(
            detect_goal_signal("**GOAL_COMPLETE.**"),
            GoalSignal::Complete
        );
    }

    #[test]
    fn detect_goal_signal_recognizes_blocked_with_reason() {
        assert_eq!(
            detect_goal_signal("I cannot proceed.\nGOAL_BLOCKED: missing API credentials"),
            GoalSignal::Blocked("missing API credentials".to_string())
        );
    }

    #[test]
    fn detect_goal_signal_continue_when_no_sentinel() {
        assert_eq!(
            detect_goal_signal("Made progress: refactored the parser, two tests still red."),
            GoalSignal::Continue
        );
    }

    #[test]
    fn detect_goal_signal_ignores_sentinel_discussed_in_prose() {
        // The model mentioning the sentinel mid-sentence must NOT trip a
        // stop -- only a standalone line counts.
        assert_eq!(
            detect_goal_signal("I will emit GOAL_COMPLETE once the suite is green."),
            GoalSignal::Continue
        );
    }

    #[test]
    fn detect_goal_signal_complete_wins_over_blocked() {
        assert_eq!(
            detect_goal_signal("GOAL_BLOCKED: earlier note\nGOAL_COMPLETE"),
            GoalSignal::Complete
        );
    }

    #[test]
    fn decide_continue_runs_forever_when_unbounded() {
        // No ceiling + no signal => keep going, counter stays reset.
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Continue, 1_000, None, 0),
            GoalStep::Continue {
                consecutive_blocked: 0
            }
        );
    }

    #[test]
    fn decide_complete_stops_immediately() {
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Complete, 1, None, 0),
            GoalStep::Stop(GoalStop::Completed)
        );
    }

    #[test]
    fn decide_complete_wins_on_the_final_turn() {
        // Even when the ceiling is reached, a verified completion reports
        // success rather than a budget stop.
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Complete, 5, Some(5), 0),
            GoalStep::Stop(GoalStop::Completed)
        );
    }

    #[test]
    fn decide_blocked_needs_three_consecutive_turns() {
        // First two blocked reports keep going with an incrementing counter.
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Blocked("x".into()), 1, None, 0),
            GoalStep::Continue {
                consecutive_blocked: 1
            }
        );
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Blocked("x".into()), 2, None, 1),
            GoalStep::Continue {
                consecutive_blocked: 2
            }
        );
        // The third consecutive blocked report stops the loop.
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Blocked("stuck".into()), 3, None, 2),
            GoalStep::Stop(GoalStop::Blocked("stuck".into()))
        );
    }

    #[test]
    fn decide_continue_resets_the_blocked_counter() {
        // A productive turn after some blocked reports clears the counter,
        // so a later transient blocker starts counting from scratch.
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Continue, 4, None, 2),
            GoalStep::Continue {
                consecutive_blocked: 0
            }
        );
    }

    #[test]
    fn decide_ceiling_stops_only_when_opted_in() {
        // Unbounded: never a ceiling stop.
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Continue, 9_999, None, 0),
            GoalStep::Continue {
                consecutive_blocked: 0
            }
        );
        // Opt-in ceiling reached with no completion => budget stop.
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Continue, 25, Some(25), 0),
            GoalStep::Stop(GoalStop::CeilingReached)
        );
        // A sub-threshold blocker on the final allowed turn also yields a
        // ceiling stop (it can't keep retrying past the budget).
        assert_eq!(
            decide_after_goal_turn(GoalSignal::Blocked("y".into()), 25, Some(25), 0),
            GoalStep::Stop(GoalStop::CeilingReached)
        );
    }

    #[test]
    fn render_goal_stop_messages_are_stable() {
        assert_eq!(
            render_goal_stop(&GoalStop::Completed, 4, None),
            "\n✅ Goal achieved in 4 turn(s): the agent reported the objective verifiably complete.\n"
        );
        assert_eq!(
            render_goal_stop(&GoalStop::Blocked("needs credentials".into()), 7, None),
            "\n⛔ Goal blocked after 7 turn(s) (3 consecutive blocked reports). Stopping for user input.\nReason: needs credentials\n"
        );
        assert_eq!(
            render_goal_stop(&GoalStop::CeilingReached, 10, Some(10)),
            "\n🛑 Goal stopped: reached the opt-in 10-turn ceiling without a completion signal. Review the progress above and re-run `/goal` (raise or drop `--max-turns`) to keep going.\n"
        );
    }

    #[test]
    fn render_goal_exit_wording_is_stable() {
        // Cancelled: fixed live message, turn count in the recap line only.
        let text = render_goal_exit(&GoalExit::Cancelled, 3, None);
        assert_eq!(text.user_message, "Goal cancelled.\n");
        assert_eq!(text.recap_stop_line, "goal cancelled after 3 goal turn(s)");
        assert_eq!(text.recap_detail, None);

        // Fatal model failure: message matches the historical ⛔ wording and
        // the failure reason rides into the recap detail.
        let failure = crate::tool_loop::TurnFailure {
            retryable: false,
            message: "invalid_api_key".to_string(),
        };
        let text = render_goal_exit(&GoalExit::FatalFailure(failure), 2, None);
        assert_eq!(
            text.user_message,
            "\n⛔ Goal stopped after 2 turn(s): the model request failed and \
             cannot be retried.\nReason: invalid_api_key\n"
        );
        assert_eq!(
            text.recap_stop_line,
            "goal stopped after 2 goal turn(s) on a fatal model failure"
        );
        assert_eq!(text.recap_detail.as_deref(), Some("invalid_api_key"));

        // Terminal pipeline error.
        let text = render_goal_exit(&GoalExit::Terminal("unknown session".to_string()), 0, None);
        assert_eq!(text.user_message, "\nGoal stopped: unknown session\n");
        assert_eq!(
            text.recap_stop_line,
            "goal stopped after 0 goal turn(s) on a fatal error"
        );
        assert_eq!(text.recap_detail.as_deref(), Some("unknown session"));

        // Sentinel stops reuse render_goal_stop + goal_recap_parts verbatim.
        let text = render_goal_exit(&GoalExit::Stop(GoalStop::Completed), 4, None);
        assert_eq!(
            text.user_message,
            render_goal_stop(&GoalStop::Completed, 4, None)
        );
        assert_eq!(
            (text.recap_stop_line, text.recap_detail),
            goal_recap_parts(&GoalStop::Completed, 4, None)
        );
    }

    #[test]
    fn goal_recap_parts_are_stable() {
        assert_eq!(
            goal_recap_parts(&GoalStop::Completed, 4, None),
            ("goal achieved after 4 goal turn(s)".to_string(), None)
        );
        assert_eq!(
            goal_recap_parts(&GoalStop::Blocked("needs credentials".into()), 7, None),
            (
                "goal blocked after 7 goal turn(s)".to_string(),
                Some("Blocked: needs credentials".to_string())
            )
        );
        let (stop_line, detail) = goal_recap_parts(&GoalStop::CeilingReached, 10, Some(10));
        assert_eq!(stop_line, "goal stopped at the opt-in 10-turn ceiling");
        assert!(
            detail
                .expect("ceiling stop carries a remaining-work note")
                .contains("did not report completion"),
        );
    }

    #[test]
    fn render_blocked_progress_only_reports_nonzero_counts() {
        assert_eq!(render_blocked_progress(0), None);
        assert_eq!(
            render_blocked_progress(2),
            Some("[goal: blocked report 2/3; retrying]\n".to_string())
        );
    }

    #[test]
    fn decide_after_goal_failure_backs_off_on_transient() {
        // A retryable (transient) failure backs off and retries, incrementing
        // the outage streak -- the goal survives the outage rather than stopping.
        let transient = crate::tool_loop::TurnFailure {
            retryable: true,
            message: "server_is_overloaded".to_string(),
        };
        assert_eq!(
            decide_after_goal_failure(&transient, 0),
            GoalFailureAction::Backoff {
                consecutive_failures: 1
            }
        );
        assert_eq!(
            decide_after_goal_failure(&transient, 4),
            GoalFailureAction::Backoff {
                consecutive_failures: 5
            }
        );
    }

    #[test]
    fn decide_after_goal_failure_stops_on_fatal() {
        // A non-retryable failure (auth, invalid request, panic) stops the goal:
        // retrying would not help.
        let fatal = crate::tool_loop::TurnFailure {
            retryable: false,
            message: "agent loop panicked".to_string(),
        };
        assert_eq!(
            decide_after_goal_failure(&fatal, 0),
            GoalFailureAction::Stop
        );
        assert_eq!(
            decide_after_goal_failure(&fatal, 9),
            GoalFailureAction::Stop
        );
    }

    #[test]
    fn goal_failure_backoff_grows_then_caps() {
        // First failure ~200ms (codex base, jittered); the delay grows
        // exponentially but never exceeds the cap, so a long outage settles
        // into a steady poll instead of growing without bound.
        let first = goal_failure_backoff(1);
        assert!(
            (180..=220).contains(&first.as_millis()),
            "first backoff should jitter around 200ms, got {first:?}"
        );
        assert!(first <= GOAL_FAILURE_BACKOFF_CAP);
        // A large streak is clamped to the cap (and must not overflow/panic).
        assert_eq!(goal_failure_backoff(1_000), GOAL_FAILURE_BACKOFF_CAP);
    }

    #[test]
    fn build_goal_prompt_embeds_objective_and_sentinels() {
        // Unbounded goal: header carries the turn number but no ceiling.
        let p = build_goal_prompt("ship the feature", 1, None, GoalPhase::Continue);
        assert!(p.contains("ship the feature"), "objective missing");
        assert!(
            p.contains(GOAL_COMPLETE_SENTINEL),
            "complete sentinel missing"
        );
        assert!(
            p.contains(GOAL_BLOCKED_SENTINEL),
            "blocked sentinel missing"
        );
        assert!(p.contains("turn 1)"), "unbounded turn header missing");
        assert!(
            !p.contains("of at most"),
            "unbounded goal must not advertise a ceiling"
        );

        // Capped goal: header advertises the ceiling.
        let capped = build_goal_prompt("ship it", 3, Some(25), GoalPhase::Continue);
        assert!(
            capped.contains("turn 3 of at most 25"),
            "capped turn header missing"
        );

        let wrap = build_goal_prompt("ship it", 25, Some(25), GoalPhase::FinalWrapUp);
        assert!(wrap.contains("FINAL turn"), "wrap-up framing missing");
    }
}
