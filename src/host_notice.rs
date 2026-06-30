//! Host-generated assistant transcript notices.
//!
//! These strings are visible to users and persisted in transcripts, but they are
//! not model output. Model-history builders must strip only validated trailing
//! notices from persisted assistant text.

use std::collections::{BTreeMap, BTreeSet};

use crate::session::{ToolExchange, ToolExchangeStatus};
use crate::tool_loop::LoopStop;

pub(crate) const STOP_NOTICE_SENTINEL: &str = "\n⏹ ";
pub(crate) const TURN_RECAP_NOTICE_SENTINEL: &str =
    "\n\n<!-- anvil:host-notice:turn-recap:v1 -->\n**Anvil Recap**\n";

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

fn render_tool_counts(tool_exchanges: &[ToolExchange]) -> String {
    if tool_exchanges.is_empty() {
        return "none".to_string();
    }

    let total = tool_exchanges.len();
    let failed = tool_exchanges
        .iter()
        .filter(|exchange| matches!(exchange.status, ToolExchangeStatus::Failed))
        .count();
    let succeeded = total.saturating_sub(failed);
    let mut by_name: BTreeMap<&str, usize> = BTreeMap::new();
    for exchange in tool_exchanges {
        *by_name.entry(exchange.tool_name.as_str()).or_default() += 1;
    }
    let mut names: Vec<String> = by_name
        .into_iter()
        .map(|(name, count)| {
            if count == 1 {
                name.to_string()
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
        plural(total, "call", "calls"),
        succeeded,
        failed,
        names.join(", ")
    )
}

fn render_changed_files(tool_exchanges: &[ToolExchange]) -> String {
    let mut paths = BTreeSet::new();
    for exchange in tool_exchanges {
        if matches!(exchange.status, ToolExchangeStatus::Completed)
            && let Some(diff) = &exchange.diff
        {
            paths.insert(diff.path.display().to_string());
        }
    }
    if paths.is_empty() {
        return "none".to_string();
    }

    const MAX_CHANGED_FILES_IN_RECAP: usize = 8;
    let total = paths.len();
    let mut listed: Vec<String> = paths.into_iter().take(MAX_CHANGED_FILES_IN_RECAP).collect();
    if total > MAX_CHANGED_FILES_IN_RECAP {
        listed.push(format!("+{} more", total - MAX_CHANGED_FILES_IN_RECAP));
    }
    listed.join(", ")
}

pub(crate) fn render_turn_recap(tool_exchanges: &[ToolExchange], stop: &LoopStop) -> String {
    format!(
        "{}- Stop: {}.\n- Tools: {}.\n- Files changed: {}.\n",
        TURN_RECAP_NOTICE_SENTINEL,
        describe_loop_stop_for_recap(stop),
        render_tool_counts(tool_exchanges),
        render_changed_files(tool_exchanges)
    )
}

fn strip_trailing_turn_recap(text: &str) -> Option<&str> {
    let index = text.rfind(TURN_RECAP_NOTICE_SENTINEL)?;
    let suffix = &text[index..];
    let mut lines = suffix[TURN_RECAP_NOTICE_SENTINEL.len()..].lines();
    let stop = lines.next()?;
    let tools = lines.next()?;
    let files = lines.next()?;
    if lines.next().is_some() {
        return None;
    }
    if stop.starts_with("- Stop: ")
        && stop.ends_with('.')
        && tools.starts_with("- Tools: ")
        && tools.ends_with('.')
        && files.starts_with("- Files changed: ")
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
        assert!(recap.contains("- Stop: completed."));
        assert!(
            recap.contains("- Tools: 2 calls (1 succeeded, 1 failed): edit, run_shell_command.")
        );
        assert!(recap.contains("- Files changed: src/lib.rs."));
    }

    #[test]
    fn model_visible_assistant_text_strips_trailing_host_notices_only() {
        let notice = render_loop_stop(&LoopStop::MaxTurns { max_turns: 3 }).unwrap();
        let persisted = format!("the model's real answer{notice}");
        assert_eq!(
            model_visible_assistant_text(&persisted),
            "the model's real answer"
        );

        let recap = render_turn_recap(&[], &LoopStop::Completed { had_text: true });
        let persisted = format!("answer{recap}");
        assert_eq!(model_visible_assistant_text(&persisted), "answer");

        let persisted = format!("answer{notice}{recap}");
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
