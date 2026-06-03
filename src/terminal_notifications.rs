use std::io::{IsTerminal, Write};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalNotificationEvent {
    Prompt,
    TurnEnded,
}

impl TerminalNotificationEvent {
    fn matches_token(self, token: &str) -> bool {
        match self {
            Self::Prompt => matches!(token, "prompt"),
            Self::TurnEnded => matches!(token, "turn-ended" | "turn_ended" | "end-of-turn"),
        }
    }
}

pub fn emit(event: TerminalNotificationEvent) {
    let config = std::env::var("BROKK_TERMINAL_NOTIFICATIONS").ok();
    if !is_enabled(std::io::stderr().is_terminal(), config.as_deref(), event) {
        return;
    }

    let mut stderr = std::io::stderr().lock();
    if let Err(err) = stderr.write_all(b"\x07").and_then(|_| stderr.flush()) {
        tracing::debug!(?event, "terminal notification failed: {err}");
    }
}

fn is_enabled(
    stderr_is_terminal: bool,
    config: Option<&str>,
    event: TerminalNotificationEvent,
) -> bool {
    if !stderr_is_terminal {
        return false;
    }

    let Some(config) = config.map(str::trim) else {
        return true;
    };
    if config.is_empty() {
        return true;
    }

    let lowered = config.to_ascii_lowercase();
    match lowered.as_str() {
        "0" | "false" | "off" | "none" => return false,
        "1" | "true" | "on" | "all" => return true,
        _ => {}
    }

    lowered
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .any(|token| event.matches_token(token))
}

#[cfg(test)]
mod tests {
    use super::{TerminalNotificationEvent, is_enabled};

    #[test]
    fn defaults_to_all_events_on_a_real_terminal() {
        assert!(is_enabled(true, None, TerminalNotificationEvent::Prompt));
        assert!(is_enabled(true, None, TerminalNotificationEvent::TurnEnded));
    }

    #[test]
    fn can_disable_notifications_entirely() {
        assert!(!is_enabled(
            true,
            Some("off"),
            TerminalNotificationEvent::Prompt
        ));
        assert!(!is_enabled(
            true,
            Some("false"),
            TerminalNotificationEvent::TurnEnded
        ));
    }

    #[test]
    fn can_select_individual_events() {
        assert!(is_enabled(
            true,
            Some("prompt"),
            TerminalNotificationEvent::Prompt
        ));
        assert!(!is_enabled(
            true,
            Some("prompt"),
            TerminalNotificationEvent::TurnEnded
        ));
        assert!(is_enabled(
            true,
            Some("turn-ended"),
            TerminalNotificationEvent::TurnEnded
        ));
        assert!(is_enabled(
            true,
            Some("prompt,turn-ended"),
            TerminalNotificationEvent::TurnEnded
        ));
    }

    #[test]
    fn never_emits_without_a_terminal() {
        assert!(!is_enabled(
            false,
            Some("all"),
            TerminalNotificationEvent::Prompt
        ));
    }
}
