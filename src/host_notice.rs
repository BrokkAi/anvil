//! Host-generated assistant transcript notices.
//!
//! These strings are visible to users and persisted in transcripts, but they are
//! not model output. Model-history builders must strip only validated trailing
//! notices from persisted assistant text.

use crate::tool_loop::LoopStop;

pub(crate) const STOP_NOTICE_SENTINEL: &str = "\n⏹ ";

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

/// The portion of a persisted `agent_response` the model actually produced, with
/// any trailing host-injected loop-stop notice removed.
///
/// The notice is appended to `agent_response` so it survives `session/load` in
/// the transcript, but it must not be fed back to the model as its own prior
/// words. History reconstruction calls this; transcript replay does not. Only
/// validated trailing notices are stripped, so model text that merely resembles
/// a notice is returned unchanged.
pub(crate) fn model_visible_assistant_text(agent_response: &str) -> &str {
    let mut text = agent_response;
    while let Some(stripped) = strip_trailing_loop_stop(text) {
        text = stripped;
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn model_visible_assistant_text_strips_trailing_loop_stop_notices_only() {
        let notice = render_loop_stop(&LoopStop::MaxTurns { max_turns: 3 }).unwrap();
        let persisted = format!("the model's real answer{notice}");
        assert_eq!(
            model_visible_assistant_text(&persisted),
            "the model's real answer"
        );

        let only_notice = render_loop_stop(&LoopStop::Completed { had_text: false }).unwrap();
        assert_eq!(model_visible_assistant_text(&only_notice), "");

        assert_eq!(
            model_visible_assistant_text("just a normal answer"),
            "just a normal answer"
        );

        // A model-authored line that merely resembles a notice is preserved.
        let model_authored = "answer\n⏹ Stopped: my own sentence about stopping.";
        assert_eq!(model_visible_assistant_text(model_authored), model_authored);
    }
}
