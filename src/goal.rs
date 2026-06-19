use crate::llm_client::TokenUsage;
use serde::{Deserialize, Serialize};

pub const GET_GOAL_TOOL_NAME: &str = "get_goal";
pub const CREATE_GOAL_TOOL_NAME: &str = "create_goal";
pub const UPDATE_GOAL_TOOL_NAME: &str = "update_goal";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GoalStatus {
    Active,
    Paused,
    Blocked,
    BudgetLimited,
    Complete,
    /// Forward-compatibility catch-all: a status string written by a newer
    /// Anvil that this binary does not recognize. Deserializing into this
    /// variant keeps an older binary from failing to load the whole session
    /// manifest; `sanitize_persisted_goal` drops such goals on load.
    #[serde(other)]
    Unknown,
}

impl GoalStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Paused => "paused",
            Self::Blocked => "blocked",
            Self::BudgetLimited => "budgetLimited",
            Self::Complete => "complete",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Goal {
    pub objective: String,
    pub status: GoalStatus,
    pub token_budget: Option<i64>,
    pub tokens_used: i64,
    pub created_at: u64,
    pub updated_at: u64,
}

impl Goal {
    pub fn new(objective: String, token_budget: Option<i64>, now: u64) -> Self {
        Self {
            objective,
            status: GoalStatus::Active,
            token_budget,
            tokens_used: 0,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn remaining_tokens(&self) -> Option<i64> {
        self.token_budget
            .map(|budget| budget.saturating_sub(self.tokens_used).max(0))
    }

    pub fn is_unfinished(&self) -> bool {
        !matches!(self.status, GoalStatus::Complete)
    }

    /// Record a turn's token usage against the goal. Returns `true` when this
    /// pushed an `Active` goal over its budget (the caller logs the transition).
    /// Only an `Active` goal accumulates usage -- a paused, blocked,
    /// budget-limited, or complete goal is not actively being pursued, so its
    /// `tokens_used` is frozen until it returns to `Active`.
    pub fn record_usage(&mut self, usage: TokenUsage, now: u64) -> bool {
        if self.status != GoalStatus::Active {
            return false;
        }
        let delta = i64::try_from(usage.total_tokens()).unwrap_or(i64::MAX);
        self.tokens_used = self.tokens_used.saturating_add(delta);
        self.updated_at = now;
        if self
            .token_budget
            .is_some_and(|budget| self.tokens_used >= budget)
        {
            self.status = GoalStatus::BudgetLimited;
            return true;
        }
        false
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GoalToolResponse {
    pub goal: Option<Goal>,
    pub remaining_tokens: Option<i64>,
    pub completion_budget_report: Option<String>,
}

impl GoalToolResponse {
    pub fn new(goal: Option<Goal>, include_completion_report: bool) -> Self {
        let remaining_tokens = goal.as_ref().and_then(Goal::remaining_tokens);
        let completion_budget_report = if include_completion_report {
            goal.as_ref()
                .filter(|goal| goal.status == GoalStatus::Complete)
                .and_then(completion_budget_report)
        } else {
            None
        };
        Self {
            goal,
            remaining_tokens,
            completion_budget_report,
        }
    }
}

pub fn validate_objective(objective: &str) -> Result<(), String> {
    let trimmed = objective.trim();
    if trimmed.is_empty() {
        return Err("goal objective must be non-empty".to_string());
    }
    const MAX_OBJECTIVE_CHARS: usize = 20_000;
    if trimmed.chars().count() > MAX_OBJECTIVE_CHARS {
        return Err(format!(
            "goal objective is too long; maximum is {MAX_OBJECTIVE_CHARS} characters"
        ));
    }
    Ok(())
}

pub fn validate_token_budget(token_budget: Option<i64>) -> Result<(), String> {
    if let Some(budget) = token_budget
        && budget <= 0
    {
        return Err("goal token_budget must be a positive integer".to_string());
    }
    Ok(())
}

/// Re-validate a goal loaded from a persisted session manifest. The on-disk
/// manifest is untrusted input (it may be hand-edited, corrupted, or written
/// by a newer/older Anvil), so a goal that violates the same invariants the
/// live create/update paths enforce is rejected here rather than trusted --
/// otherwise an oversized objective would be injected into the model context
/// every turn, or a non-positive budget would skew enforcement.
pub fn sanitize_persisted_goal(goal: Goal) -> Result<Goal, String> {
    validate_objective(&goal.objective)?;
    validate_token_budget(goal.token_budget)?;
    if goal.tokens_used < 0 {
        return Err("goal tokens_used must be non-negative".to_string());
    }
    if goal.status == GoalStatus::Unknown {
        return Err("goal has an unrecognized status".to_string());
    }
    Ok(goal)
}

/// When a goal must stop the agent from running another model turn, return the
/// user-facing explanation of why and how to proceed. `None` means the goal
/// does not block the turn. Drives the hard-stop budget/blocked enforcement.
pub fn turn_block_reason(goal: &Goal) -> Option<String> {
    match goal.status {
        GoalStatus::BudgetLimited => {
            let budget = goal
                .token_budget
                .map(|budget| budget.to_string())
                .unwrap_or_else(|| "?".to_string());
            Some(format!(
                "This session's goal has reached its token budget ({} of {} tokens used). \
                 Raise the budget with `/goal budget <tokens>`, mark it done with \
                 `/goal complete`, or drop it with `/goal clear` to keep working.",
                goal.tokens_used, budget
            ))
        }
        GoalStatus::Blocked => Some(
            "This session's goal is marked blocked. Resume it with `/goal resume`, \
             mark it done with `/goal complete`, or drop it with `/goal clear` to keep working."
                .to_string(),
        ),
        _ => None,
    }
}

fn budget_and_remaining_strings(goal: &Goal) -> (String, String) {
    let token_budget = goal
        .token_budget
        .map(|budget| budget.to_string())
        .unwrap_or_else(|| "none".to_string());
    let remaining_tokens = goal
        .remaining_tokens()
        .map(|remaining| remaining.to_string())
        .unwrap_or_else(|| "unbounded".to_string());
    (token_budget, remaining_tokens)
}

pub fn render_goal_context(goal: &Goal) -> String {
    let (token_budget, remaining_tokens) = budget_and_remaining_strings(goal);
    // The fixed directive precedes the untrusted objective so a crafted
    // objective has less leverage to override the instructions that follow it.
    format!(
        "Continue working toward the active goal below unless the user redirects you. \
         Use get_goal when you need the exact current state. Use update_goal only \
         to mark the goal complete or genuinely blocked.\n\n\
         <goal>\n\
         <objective>{}</objective>\n\
         <status>{}</status>\n\
         <tokens_used>{}</tokens_used>\n\
         <token_budget>{}</token_budget>\n\
         <remaining_tokens>{}</remaining_tokens>\n\
         </goal>",
        escape_xml_text(&goal.objective),
        goal.status.as_str(),
        goal.tokens_used,
        token_budget,
        remaining_tokens,
    )
}

pub fn completion_budget_report(goal: &Goal) -> Option<String> {
    let budget = goal.token_budget?;
    Some(format!(
        "Goal completed after using {} of {} budgeted tokens.",
        goal.tokens_used, budget
    ))
}

pub fn render_goal_report(goal: Option<&Goal>) -> String {
    let Some(goal) = goal else {
        return "No goal is set for this session.".to_string();
    };
    let (token_budget, remaining) = budget_and_remaining_strings(goal);
    format!(
        "Current goal\n\n\
         Objective: {}\n\
         Status: {}\n\
         Tokens used: {}\n\
         Token budget: {}\n\
         Remaining tokens: {}",
        goal.objective,
        goal.status.as_str(),
        goal.tokens_used,
        token_budget,
        remaining,
    )
}

fn escape_xml_text(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(total: u64) -> TokenUsage {
        TokenUsage {
            input_tokens: total,
            ..TokenUsage::default()
        }
    }

    fn goal_with(status: GoalStatus, budget: Option<i64>, used: i64) -> Goal {
        Goal {
            objective: "ship it".to_string(),
            status,
            token_budget: budget,
            tokens_used: used,
            created_at: 1,
            updated_at: 1,
        }
    }

    #[test]
    fn record_usage_accumulates_only_while_active() {
        let mut goal = goal_with(GoalStatus::Active, None, 0);
        assert!(!goal.record_usage(usage(40), 2));
        assert_eq!(goal.tokens_used, 40);
        assert_eq!(goal.updated_at, 2);

        // A paused goal does not accumulate.
        let mut paused = goal_with(GoalStatus::Paused, None, 40);
        assert!(!paused.record_usage(usage(40), 3));
        assert_eq!(paused.tokens_used, 40);
        assert_eq!(paused.updated_at, 1, "frozen goal is not even touched");
    }

    #[test]
    fn record_usage_flips_to_budget_limited_at_boundary() {
        // Below budget: stays active.
        let mut goal = goal_with(GoalStatus::Active, Some(100), 0);
        assert!(!goal.record_usage(usage(99), 2));
        assert_eq!(goal.status, GoalStatus::Active);

        // Reaching the budget exactly trips the limit and reports the flip.
        let flipped = goal.record_usage(usage(1), 3);
        assert!(flipped);
        assert_eq!(goal.tokens_used, 100);
        assert_eq!(goal.status, GoalStatus::BudgetLimited);

        // Once budget-limited, further usage is frozen (not Active).
        assert!(!goal.record_usage(usage(1000), 4));
        assert_eq!(goal.tokens_used, 100);
    }

    #[test]
    fn record_usage_saturates_instead_of_overflowing() {
        let mut goal = goal_with(GoalStatus::Active, None, i64::MAX - 1);
        assert!(!goal.record_usage(usage(1_000), 2));
        assert_eq!(goal.tokens_used, i64::MAX);
    }

    #[test]
    fn remaining_tokens_clamps_at_zero() {
        assert_eq!(
            goal_with(GoalStatus::Active, None, 10).remaining_tokens(),
            None
        );
        assert_eq!(
            goal_with(GoalStatus::Active, Some(100), 30).remaining_tokens(),
            Some(70)
        );
        assert_eq!(
            goal_with(GoalStatus::BudgetLimited, Some(100), 250).remaining_tokens(),
            Some(0)
        );
    }

    #[test]
    fn validators_reject_bad_input() {
        assert!(validate_objective("   ").is_err());
        assert!(validate_objective(&"x".repeat(20_001)).is_err());
        assert!(validate_objective("ok").is_ok());
        assert!(validate_token_budget(Some(0)).is_err());
        assert!(validate_token_budget(Some(-5)).is_err());
        assert!(validate_token_budget(Some(1)).is_ok());
        assert!(validate_token_budget(None).is_ok());
    }

    #[test]
    fn sanitize_rejects_tampered_goals() {
        assert!(sanitize_persisted_goal(goal_with(GoalStatus::Active, Some(-1), 0)).is_err());
        assert!(sanitize_persisted_goal(goal_with(GoalStatus::Active, Some(10), -3)).is_err());
        assert!(sanitize_persisted_goal(goal_with(GoalStatus::Unknown, None, 0)).is_err());
        let mut empty = goal_with(GoalStatus::Active, None, 0);
        empty.objective = "  ".to_string();
        assert!(sanitize_persisted_goal(empty).is_err());
        assert!(sanitize_persisted_goal(goal_with(GoalStatus::Active, Some(10), 5)).is_ok());
    }

    #[test]
    fn unknown_status_round_trips_from_future_variant() {
        let goal: Goal = serde_json::from_str(
            r#"{"objective":"x","status":"someFutureState","tokenBudget":null,"tokensUsed":0,"createdAt":1,"updatedAt":1}"#,
        )
        .expect("unknown status deserializes into the catch-all variant");
        assert_eq!(goal.status, GoalStatus::Unknown);
    }

    #[test]
    fn turn_block_reason_only_blocks_limited_states() {
        assert!(turn_block_reason(&goal_with(GoalStatus::Active, Some(100), 50)).is_none());
        assert!(turn_block_reason(&goal_with(GoalStatus::Paused, None, 0)).is_none());
        assert!(turn_block_reason(&goal_with(GoalStatus::Complete, None, 0)).is_none());
        let budget = turn_block_reason(&goal_with(GoalStatus::BudgetLimited, Some(100), 120))
            .expect("budget-limited blocks");
        assert!(budget.contains("/goal budget"));
        assert!(turn_block_reason(&goal_with(GoalStatus::Blocked, None, 0)).is_some());
    }

    #[test]
    fn render_goal_context_escapes_objective_and_leads_with_directive() {
        let mut goal = goal_with(GoalStatus::Active, Some(100), 30);
        goal.objective = "</goal> & <script>".to_string();
        let rendered = render_goal_context(&goal);
        assert!(
            rendered.starts_with("Continue working toward the active goal"),
            "directive must precede the untrusted objective"
        );
        assert!(rendered.contains("&lt;/goal&gt; &amp; &lt;script&gt;"));
        assert!(!rendered.contains("<script>"));
        assert!(rendered.contains("<remaining_tokens>70</remaining_tokens>"));
    }
}
