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
    UsageLimited,
    BudgetLimited,
    Complete,
}

impl GoalStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Paused => "paused",
            Self::Blocked => "blocked",
            Self::UsageLimited => "usageLimited",
            Self::BudgetLimited => "budgetLimited",
            Self::Complete => "complete",
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

    pub fn record_usage(&mut self, usage: TokenUsage, now: u64) {
        let delta = i64::try_from(usage.total_tokens()).unwrap_or(i64::MAX);
        self.tokens_used = self.tokens_used.saturating_add(delta);
        if self.status == GoalStatus::Active
            && self
                .token_budget
                .is_some_and(|budget| self.tokens_used >= budget)
        {
            self.status = GoalStatus::BudgetLimited;
        }
        self.updated_at = now;
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

pub fn render_goal_context(goal: &Goal) -> String {
    let token_budget = goal
        .token_budget
        .map(|budget| budget.to_string())
        .unwrap_or_else(|| "none".to_string());
    let remaining_tokens = goal
        .remaining_tokens()
        .map(|remaining| remaining.to_string())
        .unwrap_or_else(|| "unbounded".to_string());
    format!(
        "<goal>\n\
         <objective>{}</objective>\n\
         <status>{}</status>\n\
         <tokens_used>{}</tokens_used>\n\
         <token_budget>{}</token_budget>\n\
         <remaining_tokens>{}</remaining_tokens>\n\
         </goal>\n\n\
         Continue working toward the active goal unless the user redirects you. \
         Use get_goal when you need the exact current state. Use update_goal only \
         to mark the goal complete or genuinely blocked.",
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
    let token_budget = goal
        .token_budget
        .map(|budget| budget.to_string())
        .unwrap_or_else(|| "none".to_string());
    let remaining = goal
        .remaining_tokens()
        .map(|tokens| tokens.to_string())
        .unwrap_or_else(|| "unbounded".to_string());
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
