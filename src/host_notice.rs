//! Host-generated assistant transcript notices.
//!
//! These strings are visible to users and persisted in transcripts, but they are
//! not model output. Model-history builders must strip only validated trailing
//! notices from persisted assistant text.

use std::collections::{BTreeMap, BTreeSet};

use crate::session::{ToolExchange, ToolExchangeStatus};
use crate::tool_loop::LoopStop;

pub(crate) const STOP_NOTICE_SENTINEL: &str = "\n⏹ ";
pub(crate) const TURN_RECAP_NOTICE_SENTINEL: &str = "\n\n**Anvil Recap**\n";

/// Upper bound on the rendered work-summary, as a safety valve against a
/// model that ignores the "a few bullets" instruction. Truncated on a char
/// boundary with an ellipsis; the deterministic stat lines are unaffected.
const MAX_RECAP_SUMMARY_CHARS: usize = 800;

pub(crate) fn render_loop_stop(stop: &LoopStop) -> Option<String> {
    match stop {
        LoopStop::MaxTurns { max_turns } => Some(format!(
            "{STOP_NOTICE_SENTINEL}Stopped: reached the {max_turns}-turn limit before the model \
             finished. Send another message to continue, or restart with a higher `--max-turns`.\n"
        )),
        LoopStop::Completed { had_text: false } => Some(format!(
            "{STOP_NOTICE_SENTINEL}Stopped: the model ended the turn without a final message.\n"
        )),
        LoopStop::Completed { had_text: true } | LoopStop::Cancelled | LoopStop::Failed(_) => None,
    }
}

fn plural(count: usize, singular: &str, plural: &str) -> String {
    if count == 1 {
        format!("{count} {singular}")
    } else {
        format!("{count} {plural}")
    }
}

fn recap_field_text(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_control() {
            escaped.extend(ch.escape_default());
        } else {
            escaped.push(ch);
        }
    }
    escaped
}

fn describe_loop_stop_for_recap(stop: &LoopStop) -> String {
    match stop {
        LoopStop::Completed { had_text: true } => "completed".to_string(),
        LoopStop::Completed { had_text: false } => "completed without a final message".to_string(),
        LoopStop::MaxTurns { max_turns } => {
            format!("stopped at the {max_turns}-turn limit")
        }
        LoopStop::Cancelled => "cancelled".to_string(),
        LoopStop::Failed(failure) if failure.retryable => "retryable model failure".to_string(),
        LoopStop::Failed(_) => "model failure".to_string(),
    }
}

/// Aggregate tool-call statistics for one or more turns. The per-turn recap
/// builds one from a single turn's exchanges; `/goal` merges one per goal
/// turn so the final aggregate recap can report the whole run without
/// keeping every exchange (diff bodies included) resident across turns.
#[derive(Debug, Clone, Default)]
pub(crate) struct ToolCallStats {
    total: usize,
    failed: usize,
    by_name: BTreeMap<String, usize>,
    changed_files: BTreeSet<String>,
}

impl ToolCallStats {
    pub(crate) fn from_exchanges(tool_exchanges: &[ToolExchange]) -> Self {
        let mut stats = Self::default();
        for exchange in tool_exchanges {
            stats.total += 1;
            if matches!(exchange.status, ToolExchangeStatus::Failed) {
                stats.failed += 1;
            }
            *stats.by_name.entry(exchange.tool_name.clone()).or_default() += 1;
            if matches!(exchange.status, ToolExchangeStatus::Completed)
                && let Some(diff) = &exchange.diff
            {
                stats
                    .changed_files
                    .insert(recap_field_text(&diff.path.display().to_string()));
            }
        }
        stats
    }

    pub(crate) fn merge(&mut self, other: &ToolCallStats) {
        self.total += other.total;
        self.failed += other.failed;
        for (name, count) in &other.by_name {
            *self.by_name.entry(name.clone()).or_default() += count;
        }
        self.changed_files
            .extend(other.changed_files.iter().cloned());
    }

    pub(crate) fn has_changed_files(&self) -> bool {
        !self.changed_files.is_empty()
    }
}

fn render_tool_counts(stats: &ToolCallStats) -> String {
    if stats.total == 0 {
        return "none".to_string();
    }

    let succeeded = stats.total.saturating_sub(stats.failed);
    let mut names: Vec<String> = stats
        .by_name
        .iter()
        .map(|(name, count)| {
            let name = recap_field_text(name);
            if *count == 1 {
                name
            } else {
                format!("{name} x{count}")
            }
        })
        .collect();
    const MAX_TOOL_NAMES_IN_RECAP: usize = 6;
    let extra = names.len().saturating_sub(MAX_TOOL_NAMES_IN_RECAP);
    names.truncate(MAX_TOOL_NAMES_IN_RECAP);
    if extra > 0 {
        names.push(format!("+{extra} more"));
    }

    format!(
        "{} ({} succeeded, {} failed): {}",
        plural(stats.total, "call", "calls"),
        succeeded,
        stats.failed,
        names.join(", ")
    )
}

fn render_changed_files(stats: &ToolCallStats) -> String {
    if stats.changed_files.is_empty() {
        return "none".to_string();
    }

    const MAX_CHANGED_FILES_IN_RECAP: usize = 8;
    let total = stats.changed_files.len();
    let mut listed: Vec<String> = stats
        .changed_files
        .iter()
        .take(MAX_CHANGED_FILES_IN_RECAP)
        .cloned()
        .collect();
    if total > MAX_CHANGED_FILES_IN_RECAP {
        listed.push(format!("+{} more", total - MAX_CHANGED_FILES_IN_RECAP));
    }
    listed.join(", ")
}

/// Neutralize the recap header in the model-written summary, then bound its
/// length. Stripping anchors on the `**Anvil Recap**` sentinel via `rfind`,
/// so a summary that echoed that header could otherwise create a second
/// match and confuse the stripper.
fn sanitize_recap_summary(summary: &str) -> String {
    let mut cleaned = summary.trim().replace("**Anvil Recap**", "Anvil Recap");
    if cleaned.chars().count() > MAX_RECAP_SUMMARY_CHARS {
        let truncated: String = cleaned.chars().take(MAX_RECAP_SUMMARY_CHARS).collect();
        cleaned = format!("{}…", truncated.trim_end());
    }
    cleaned
}

/// Render the turn recap: an optional work summary (what the assistant did
/// this turn) above the deterministic Stop / Tools / Files-changed stats.
/// The three stat lines stay last so the model-history stripper can detach
/// the whole block (summary included) by validating them as the tail.
pub(crate) fn render_turn_recap(
    summary: Option<&str>,
    tool_exchanges: &[ToolExchange],
    stop: &LoopStop,
) -> String {
    render_recap_block(
        &describe_loop_stop_for_recap(stop),
        summary,
        &ToolCallStats::from_exchanges(tool_exchanges),
    )
}

/// Render the aggregate `/goal` recap: an optional detail paragraph (blocked
/// reason, failure message, remaining-work note) above goal-level Stop /
/// Tools / Files-changed stats accumulated across every goal turn. Same
/// block shape as the per-turn recap, so `strip_trailing_turn_recap`
/// detaches it from model history unchanged.
pub(crate) fn render_goal_recap(
    stop_line: &str,
    detail: Option<&str>,
    stats: &ToolCallStats,
) -> String {
    render_recap_block(stop_line, detail, stats)
}

fn render_recap_block(stop_line: &str, detail: Option<&str>, stats: &ToolCallStats) -> String {
    let mut out = String::from(TURN_RECAP_NOTICE_SENTINEL);
    if let Some(detail) = detail {
        let detail = sanitize_recap_summary(detail);
        if !detail.is_empty() {
            out.push_str(&detail);
            out.push_str("\n\n");
        }
    }
    // Wrap each stat line's content in `*…*` so markdown clients render the
    // recap as an italic aside. The `- ` bullet and trailing `.` stay outside
    // the emphasis so `strip_trailing_turn_recap` still matches on its prefix
    // and `.` suffix anchors (and so the bullet renders as a list marker).
    // The stop line is escaped to stay single-line: a stray newline would
    // break the three-line tail validation and leak the recap into model
    // history.
    out.push_str(&format!(
        "- *Stop: {}*.\n- *Tools: {}*.\n- *Files changed: {}*.\n",
        recap_field_text(stop_line),
        render_tool_counts(stats),
        render_changed_files(stats)
    ));
    out
}

fn strip_trailing_turn_recap(text: &str) -> Option<&str> {
    let index = text.rfind(TURN_RECAP_NOTICE_SENTINEL)?;
    let body = &text[index + TURN_RECAP_NOTICE_SENTINEL.len()..];

    // The recap may carry a variable-length work summary above its three fixed
    // Stop / Tools / Files lines, which are always last. Validating the final
    // three lines lets the whole block strip while leaving model-authored text
    // that merely contains "**Anvil Recap**" untouched. If a later notice were
    // appended after the recap, these would not be the tail and the strip loop
    // in `model_visible_assistant_text` peels that off first.
    let lines: Vec<&str> = body.lines().collect();
    let [.., stop, tools, files] = lines.as_slice() else {
        return None;
    };
    if stop.starts_with("- *Stop: ")
        && stop.ends_with('.')
        && tools.starts_with("- *Tools: ")
        && tools.ends_with('.')
        && files.starts_with("- *Files changed: ")
        && files.ends_with('.')
    {
        Some(&text[..index])
    } else {
        None
    }
}

fn strip_trailing_loop_stop(text: &str) -> Option<&str> {
    let index = text.rfind(STOP_NOTICE_SENTINEL)?;
    let suffix = &text[index..];
    let body = &suffix[STOP_NOTICE_SENTINEL.len()..];
    if body == "Stopped: the model ended the turn without a final message.\n"
        || (body.starts_with("Stopped: reached the ")
            && body.ends_with(
                "-turn limit before the model finished. Send another message to continue, or restart with a higher `--max-turns`.\n",
            )
            && body["Stopped: reached the ".len()..]
                .split_once("-turn limit")
                .is_some_and(|(n, _)| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit())))
    {
        Some(&text[..index])
    } else {
        None
    }
}

pub(crate) fn model_visible_assistant_text(agent_response: &str) -> &str {
    let mut text = agent_response;
    loop {
        if let Some(stripped) = strip_trailing_turn_recap(text) {
            text = stripped;
            continue;
        }
        if let Some(stripped) = strip_trailing_loop_stop(text) {
            text = stripped;
            continue;
        }
        return text;
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::session::{ToolExchangeDiff, ToolExchangeStatus};
    use crate::tool_loop::TurnFailure;

    #[test]
    fn render_loop_stop_only_narrates_silent_terminations() {
        let max =
            render_loop_stop(&LoopStop::MaxTurns { max_turns: 7 }).expect("MaxTurns is narrated");
        assert!(max.starts_with(STOP_NOTICE_SENTINEL));
        assert!(max.contains("reached the 7-turn limit"));

        let empty = render_loop_stop(&LoopStop::Completed { had_text: false })
            .expect("empty completion is narrated");
        assert!(empty.starts_with(STOP_NOTICE_SENTINEL));

        assert!(render_loop_stop(&LoopStop::Completed { had_text: true }).is_none());
        assert!(render_loop_stop(&LoopStop::Cancelled).is_none());
        assert!(
            render_loop_stop(&LoopStop::Failed(TurnFailure {
                retryable: true,
                message: "x".into(),
            }))
            .is_none()
        );
    }

    #[test]
    fn render_turn_recap_reports_stop_tools_and_changed_files() {
        let recap = render_turn_recap(
            None,
            &[
                ToolExchange {
                    call_id: "c1".into(),
                    tool_name: "edit".into(),
                    status: ToolExchangeStatus::Completed,
                    diff: Some(ToolExchangeDiff {
                        path: PathBuf::from("src/lib.rs"),
                        old_text: Some("old".into()),
                        new_text: "new".into(),
                    }),
                    ..ToolExchange::default()
                },
                ToolExchange {
                    call_id: "c2".into(),
                    tool_name: "run_shell_command".into(),
                    status: ToolExchangeStatus::Failed,
                    ..ToolExchange::default()
                },
            ],
            &LoopStop::Completed { had_text: true },
        );

        assert!(recap.starts_with(TURN_RECAP_NOTICE_SENTINEL));
        assert!(recap.contains("- *Stop: completed*."));
        assert!(
            recap.contains("- *Tools: 2 calls (1 succeeded, 1 failed): edit, run_shell_command*.")
        );
        assert!(recap.contains("- *Files changed: src/lib.rs*."));
        // The three stat lines are the tail of the block (the strip anchor).
        assert!(recap.trim_end().ends_with("- *Files changed: src/lib.rs*."));
    }

    #[test]
    fn render_turn_recap_includes_work_summary_above_stats() {
        let recap = render_turn_recap(
            Some("- Edited `src/lib.rs` to add a guard.\n- Ran the tests; all passed."),
            &[ToolExchange {
                call_id: "c1".into(),
                tool_name: "edit".into(),
                status: ToolExchangeStatus::Completed,
                ..ToolExchange::default()
            }],
            &LoopStop::Completed { had_text: true },
        );

        let summary_at = recap
            .find("Edited `src/lib.rs`")
            .expect("summary present in recap");
        let stop_at = recap.find("- *Stop: completed*.").expect("stats present");
        assert!(summary_at < stop_at, "summary renders above the stat lines");
        // The work summary is model prose passed through verbatim, so its own
        // bullet stays plain (only the deterministic stat lines are italicized).
        assert!(recap.contains("- Ran the tests; all passed."));

        // The whole block -- summary included -- strips back out of model history.
        let persisted = format!("the answer{recap}");
        assert_eq!(model_visible_assistant_text(&persisted), "the answer");
    }

    #[test]
    fn render_turn_recap_neutralizes_sentinel_lookalikes_in_summary() {
        // A summary that echoes the recap's own start sentinel must not create a
        // second match that `rfind` latches onto, which would strip mid-summary.
        let evil_summary = format!("- Quoting a marker: {TURN_RECAP_NOTICE_SENTINEL} oops.");
        let recap = render_turn_recap(
            Some(&evil_summary),
            &[],
            &LoopStop::Completed { had_text: true },
        );
        let persisted = format!("answer{recap}");
        assert_eq!(model_visible_assistant_text(&persisted), "answer");
    }

    #[test]
    fn render_turn_recap_keeps_control_char_fields_single_line() {
        let recap = render_turn_recap(
            None,
            &[ToolExchange {
                call_id: "c1".into(),
                tool_name: "edit\nname".into(),
                status: ToolExchangeStatus::Completed,
                diff: Some(ToolExchangeDiff {
                    path: PathBuf::from("dir/a\nb.rs"),
                    old_text: Some("old".into()),
                    new_text: "new".into(),
                }),
                ..ToolExchange::default()
            }],
            &LoopStop::Completed { had_text: true },
        );

        assert!(recap.contains("- *Tools: 1 call (1 succeeded, 0 failed): edit\\nname*."));
        assert!(recap.contains("- *Files changed: dir/a\\nb.rs*."));
        let body = recap
            .strip_prefix(TURN_RECAP_NOTICE_SENTINEL)
            .expect("recap starts with sentinel");
        // Exactly the three escaped stat lines; the control chars in the fields
        // must not spill onto extra lines.
        assert_eq!(
            body.lines().count(),
            3,
            "recap body must remain line-parseable"
        );

        let persisted = format!("answer{recap}");
        assert_eq!(model_visible_assistant_text(&persisted), "answer");
    }

    #[test]
    fn tool_call_stats_tracks_whether_files_changed() {
        let read_only = ToolCallStats::from_exchanges(&[ToolExchange {
            call_id: "c1".into(),
            tool_name: "read_file".into(),
            status: ToolExchangeStatus::Completed,
            ..ToolExchange::default()
        }]);
        assert!(!read_only.has_changed_files());

        let write = ToolCallStats::from_exchanges(&[ToolExchange {
            call_id: "c1".into(),
            tool_name: "edit".into(),
            status: ToolExchangeStatus::Completed,
            diff: Some(ToolExchangeDiff {
                path: PathBuf::from("src/lib.rs"),
                old_text: Some("old".into()),
                new_text: "new".into(),
            }),
            ..ToolExchange::default()
        }]);
        assert!(write.has_changed_files());
    }

    #[test]
    fn tool_call_stats_merge_accumulates_across_turns() {
        let mut aggregate = ToolCallStats::from_exchanges(&[
            ToolExchange {
                call_id: "c1".into(),
                tool_name: "edit".into(),
                status: ToolExchangeStatus::Completed,
                diff: Some(ToolExchangeDiff {
                    path: PathBuf::from("src/a.rs"),
                    old_text: Some("old".into()),
                    new_text: "new".into(),
                }),
                ..ToolExchange::default()
            },
            ToolExchange {
                call_id: "c2".into(),
                tool_name: "run_shell_command".into(),
                status: ToolExchangeStatus::Failed,
                ..ToolExchange::default()
            },
        ]);
        aggregate.merge(&ToolCallStats::from_exchanges(&[
            ToolExchange {
                call_id: "c3".into(),
                tool_name: "edit".into(),
                status: ToolExchangeStatus::Completed,
                diff: Some(ToolExchangeDiff {
                    path: PathBuf::from("src/b.rs"),
                    old_text: None,
                    new_text: "new".into(),
                }),
                ..ToolExchange::default()
            },
            // A second write to an already-changed file must not double-count.
            ToolExchange {
                call_id: "c4".into(),
                tool_name: "edit".into(),
                status: ToolExchangeStatus::Completed,
                diff: Some(ToolExchangeDiff {
                    path: PathBuf::from("src/a.rs"),
                    old_text: Some("new".into()),
                    new_text: "newer".into(),
                }),
                ..ToolExchange::default()
            },
        ]));

        let recap = render_goal_recap("goal achieved after 2 goal turn(s)", None, &aggregate);
        assert!(recap.contains("- *Stop: goal achieved after 2 goal turn(s)*."));
        assert!(
            recap.contains(
                "- *Tools: 4 calls (3 succeeded, 1 failed): edit x3, run_shell_command*."
            )
        );
        assert!(recap.contains("- *Files changed: src/a.rs, src/b.rs*."));
    }

    #[test]
    fn render_goal_recap_strips_from_model_history_and_stays_line_parseable() {
        let stats = ToolCallStats::from_exchanges(&[ToolExchange {
            call_id: "c1".into(),
            tool_name: "read_file".into(),
            status: ToolExchangeStatus::Completed,
            ..ToolExchange::default()
        }]);

        // Detail paragraph (e.g. a blocked reason) renders above the stats
        // and the whole block strips back out of model history.
        let recap = render_goal_recap(
            "goal blocked after 5 goal turn(s)",
            Some("Blocked: waiting on credentials for the staging registry."),
            &stats,
        );
        assert!(recap.starts_with(TURN_RECAP_NOTICE_SENTINEL));
        let detail_at = recap
            .find("waiting on credentials")
            .expect("detail present");
        let stop_at = recap.find("- *Stop: ").expect("stats present");
        assert!(detail_at < stop_at, "detail renders above the stat lines");
        let persisted = format!("final goal turn text{recap}");
        assert_eq!(
            model_visible_assistant_text(&persisted),
            "final goal turn text"
        );

        // A stop line carrying control characters must stay single-line, or
        // the stripper's three-line tail validation would leak the recap
        // into model history.
        let evil = render_goal_recap("stopped\nafter 1 goal turn(s)", None, &stats);
        assert!(evil.contains("- *Stop: stopped\\nafter 1 goal turn(s)*."));
        let persisted = format!("answer{evil}");
        assert_eq!(model_visible_assistant_text(&persisted), "answer");
    }

    #[test]
    fn model_visible_assistant_text_strips_trailing_host_notices_only() {
        let notice = render_loop_stop(&LoopStop::MaxTurns { max_turns: 3 }).unwrap();
        let persisted = format!("the model's real answer{notice}");
        assert_eq!(
            model_visible_assistant_text(&persisted),
            "the model's real answer"
        );

        let recap = render_turn_recap(None, &[], &LoopStop::Completed { had_text: true });
        let persisted = format!("answer{recap}");
        assert_eq!(model_visible_assistant_text(&persisted), "answer");

        let persisted = format!("answer{notice}{recap}");
        assert_eq!(model_visible_assistant_text(&persisted), "answer");

        // A summary-bearing recap preceded by a stop notice strips both, in
        // either peel order the loop encounters.
        let recap_with_summary = render_turn_recap(
            Some("- Looked at `a.rs`.\n- Edited `b.rs`."),
            &[],
            &LoopStop::MaxTurns { max_turns: 3 },
        );
        let persisted = format!("answer{notice}{recap_with_summary}");
        assert_eq!(model_visible_assistant_text(&persisted), "answer");

        let only_notice = render_loop_stop(&LoopStop::Completed { had_text: false }).unwrap();
        assert_eq!(model_visible_assistant_text(&only_notice), "");

        assert_eq!(
            model_visible_assistant_text("just a normal answer"),
            "just a normal answer"
        );

        let model_authored = "answer\n\n**Anvil Recap**\nthis is model text";
        assert_eq!(model_visible_assistant_text(model_authored), model_authored);

        let embedded_marker =
            format!("answer{TURN_RECAP_NOTICE_SENTINEL}- Stop: model-authored paragraph");
        assert_eq!(
            model_visible_assistant_text(&embedded_marker),
            embedded_marker
        );
    }
}
