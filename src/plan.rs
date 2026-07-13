use agent_client_protocol::schema::v1::{Plan, PlanEntry, PlanEntryPriority, PlanEntryStatus};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StepStatus {
    Pending,
    InProgress,
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PlanItem {
    pub step: String,
    pub status: StepStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct UpdatePlanArgs {
    #[serde(default)]
    pub explanation: Option<String>,
    pub plan: Vec<PlanItem>,
}

impl UpdatePlanArgs {
    pub(crate) fn to_acp(&self) -> Plan {
        Plan::new(
            self.plan
                .iter()
                .map(|item| {
                    PlanEntry::new(
                        item.step.clone(),
                        PlanEntryPriority::Medium,
                        match item.status {
                            StepStatus::Pending => PlanEntryStatus::Pending,
                            StepStatus::InProgress => PlanEntryStatus::InProgress,
                            StepStatus::Completed => PlanEntryStatus::Completed,
                        },
                    )
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_plan_maps_statuses_and_uses_medium_priority() {
        let plan = UpdatePlanArgs {
            explanation: Some("Revised after inspection".to_string()),
            plan: vec![
                PlanItem {
                    step: "Inspect".to_string(),
                    status: StepStatus::Completed,
                },
                PlanItem {
                    step: "Implement".to_string(),
                    status: StepStatus::InProgress,
                },
                PlanItem {
                    step: "Verify".to_string(),
                    status: StepStatus::Pending,
                },
            ],
        };

        let value = serde_json::to_value(plan.to_acp()).expect("ACP plan should serialize");
        assert_eq!(value["entries"][0]["status"], "completed");
        assert_eq!(value["entries"][1]["status"], "in_progress");
        assert_eq!(value["entries"][2]["status"], "pending");
        assert!(
            value["entries"]
                .as_array()
                .expect("entries array")
                .iter()
                .all(|entry| entry["priority"] == "medium")
        );
    }
}
