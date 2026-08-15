//! Routing policy for short-lived internal LLM calls.

use std::sync::OnceLock;

static CONFIG: OnceLock<UtilityModelConfig> = OnceLock::new();

#[derive(Debug, Clone, Default)]
pub(crate) struct UtilityModelConfig {
    configured_model: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UtilityModelSelection {
    pub model: String,
    pub reasoning_effort: Option<String>,
    pub source: &'static str,
}

impl UtilityModelConfig {
    pub(crate) fn new(configured_model: Option<String>) -> Self {
        Self { configured_model }
    }

    pub(crate) fn configured_model(&self) -> Option<&str> {
        self.configured_model.as_deref()
    }

    pub(crate) fn select(&self, session_model: &str) -> UtilityModelSelection {
        match self.configured_model.as_deref() {
            Some(model) => UtilityModelSelection {
                model: model.to_string(),
                reasoning_effort: None,
                source: "configured",
            },
            None => UtilityModelSelection {
                model: session_model.to_string(),
                reasoning_effort: Some("low".to_string()),
                source: "session_fallback",
            },
        }
    }
}

pub(crate) fn configure(config: UtilityModelConfig) {
    CONFIG
        .set(config)
        .expect("utility model configuration may only be installed once");
}

pub(crate) fn select(session_model: &str) -> UtilityModelSelection {
    CONFIG
        .get()
        .cloned()
        .unwrap_or_default()
        .select(session_model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_model_uses_provider_default_effort() {
        let selected =
            UtilityModelConfig::new(Some("deepseek::flash".to_string())).select("bedrock::luna");
        assert_eq!(selected.model, "deepseek::flash");
        assert_eq!(selected.reasoning_effort, None);
        assert_eq!(selected.source, "configured");
    }

    #[test]
    fn unset_model_uses_session_at_low_effort() {
        let selected = UtilityModelConfig::default().select("bedrock::luna");
        assert_eq!(selected.model, "bedrock::luna");
        assert_eq!(selected.reasoning_effort.as_deref(), Some("low"));
        assert_eq!(selected.source, "session_fallback");
    }
}
