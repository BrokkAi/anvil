use std::collections::BTreeMap;
use std::time::Duration;

use crate::discovery::{ModelSource, split_wire_id};

pub(crate) fn usage_by_model_meta(
    usage_by_model: &BTreeMap<String, crate::llm_client::TokenUsage>,
) -> serde_json::Map<String, serde_json::Value> {
    serde_json::Map::from_iter([(
        crate::structured_output::ACP_META_NAMESPACE.to_string(),
        serde_json::json!({
            "usageByModel": usage_by_model
                .iter()
                .map(|(model, usage)| {
                    (model.clone(), serde_json::json!({
                        "inputTokens": usage.input_tokens,
                        "outputTokens": usage.output_tokens,
                        "thoughtTokens": usage.thought_tokens,
                        "cachedReadTokens": usage.cached_read_tokens,
                        "cachedWriteTokens": usage.cached_write_tokens,
                        "totalTokens": usage.total_tokens(),
                    }))
                })
                .collect::<serde_json::Map<_, _>>()
        }),
    )])
}

pub(crate) fn insert_turn_failure_meta(
    meta: &mut serde_json::Map<String, serde_json::Value>,
    failure: &crate::tool_loop::TurnFailure,
) {
    let mut namespace = meta
        .remove(crate::structured_output::ACP_META_NAMESPACE)
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    namespace.insert(
        "turnFailure".to_string(),
        serde_json::json!({
            "retryable": failure.retryable,
            "message": failure.message,
        }),
    );
    meta.insert(
        crate::structured_output::ACP_META_NAMESPACE.to_string(),
        serde_json::Value::Object(namespace),
    );
}

pub(crate) fn insert_bedrock_credits_meta(
    meta: &mut serde_json::Map<String, serde_json::Value>,
    model_wire_id: &str,
    status: crate::bedrock_credits::Status,
) {
    if !is_bedrock_model(model_wire_id) {
        return;
    }
    let mut namespace = meta
        .remove(crate::structured_output::ACP_META_NAMESPACE)
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    namespace.insert(
        "bedrockCredits".to_string(),
        serde_json::to_value(status).expect("Bedrock credit status is serializable"),
    );
    meta.insert(
        crate::structured_output::ACP_META_NAMESPACE.to_string(),
        serde_json::Value::Object(namespace),
    );
}

pub(crate) fn attach_bedrock_credits_meta<F>(
    meta: &mut serde_json::Map<String, serde_json::Value>,
    model_wire_id: &str,
    status: F,
) where
    F: FnOnce() -> crate::bedrock_credits::Status,
{
    if is_bedrock_model(model_wire_id) {
        insert_bedrock_credits_meta(meta, model_wire_id, status());
    }
}

pub(crate) fn is_bedrock_model(model_wire_id: &str) -> bool {
    split_wire_id(model_wire_id).map(|(source, _)| source) == Some(ModelSource::BEDROCK)
}

pub(crate) fn insert_openrouter_balance_meta(
    meta: &mut serde_json::Map<String, serde_json::Value>,
    model_wire_id: &str,
    status: crate::openrouter_credits::Status,
) {
    if !is_openrouter_model(model_wire_id) {
        return;
    }
    let mut namespace = meta
        .remove(crate::structured_output::ACP_META_NAMESPACE)
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    namespace.insert(
        "openrouterBalance".to_string(),
        serde_json::to_value(status).expect("OpenRouter balance status is serializable"),
    );
    meta.insert(
        crate::structured_output::ACP_META_NAMESPACE.to_string(),
        serde_json::Value::Object(namespace),
    );
}

pub(crate) fn attach_openrouter_balance_meta<F>(
    meta: &mut serde_json::Map<String, serde_json::Value>,
    model_wire_id: &str,
    status: F,
) where
    F: FnOnce() -> crate::openrouter_credits::Status,
{
    if is_openrouter_model(model_wire_id) {
        insert_openrouter_balance_meta(meta, model_wire_id, status());
    }
}

pub(crate) fn is_openrouter_model(model_wire_id: &str) -> bool {
    matches!(
        split_wire_id(model_wire_id),
        Some((ModelSource::OPENROUTER, model_id)) if !model_id.is_empty()
    )
}

pub(crate) fn insert_deepseek_balance_meta(
    meta: &mut serde_json::Map<String, serde_json::Value>,
    model_wire_id: &str,
    status: crate::deepseek_balance::Status,
) {
    if !is_deepseek_model(model_wire_id) {
        return;
    }
    let mut namespace = meta
        .remove(crate::structured_output::ACP_META_NAMESPACE)
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    namespace.insert(
        "deepseekBalance".to_string(),
        serde_json::to_value(status).expect("DeepSeek balance status is serializable"),
    );
    meta.insert(
        crate::structured_output::ACP_META_NAMESPACE.to_string(),
        serde_json::Value::Object(namespace),
    );
}

pub(crate) fn attach_deepseek_balance_meta<F>(
    meta: &mut serde_json::Map<String, serde_json::Value>,
    model_wire_id: &str,
    status: F,
) where
    F: FnOnce() -> crate::deepseek_balance::Status,
{
    if is_deepseek_model(model_wire_id) {
        insert_deepseek_balance_meta(meta, model_wire_id, status());
    }
}

pub(crate) fn is_deepseek_model(model_wire_id: &str) -> bool {
    matches!(
        split_wire_id(model_wire_id),
        Some((ModelSource::DEEPSEEK, model_id)) if !model_id.is_empty()
    )
}

/// Cost of a turn whose tokens are split across models, which happens when
/// the harness routes its own internal work (permission classification, the
/// semantic reranker, subagents) to a cheaper utility model. Returns `None`
/// when any model that actually spent tokens has no price, so a partial
/// figure is never reported as the whole turn's cost.
pub(crate) fn estimate_usage_by_model_cost(
    metadata: &[crate::llm_client::ModelMetadata],
    usage_by_model: &BTreeMap<String, crate::llm_client::TokenUsage>,
) -> Option<f64> {
    usage_by_model
        .iter()
        .try_fold(0.0, |total, (model, usage)| {
            if usage.is_zero() {
                return Some(total);
            }
            let model = metadata.iter().find(|metadata| metadata.id == *model)?;
            Some(total + model.estimate_cost_usd(*usage)?)
        })
}

/// `render_usage_report` can render a distinct line for "no credential
/// found" vs. "credential found but the upstream call failed" without
/// the call site re-shaping a nested type.
#[derive(Debug, Clone)]
pub(crate) enum OpenRouterCreditsOutcome {
    /// No credential resolved (no env var, no on-disk file). Either the
    /// user hasn't logged in, or the active model isn't an OpenRouter
    /// one and we deliberately skipped the lookup.
    Skipped,
    Fetched(crate::openrouter_credits::Credits),
    Failed(String),
}

/// Outcome of the ChatGPT `wham/usage` lookup performed by `/usage`.
/// Carries more skip reasons than the OpenRouter sibling because Codex
/// has two distinct "logged in but the endpoint doesn't apply to you"
/// states: api-key mode (billing through OPENAI_API_KEY) vs no auth at
/// all. Conflating them would tell an api-key user to "run /setup
/// codex" when their setup is actually fine.
#[derive(Debug, Clone)]
pub(crate) enum CodexCreditsOutcome {
    /// Active model isn't a Codex one; no Codex line in the report.
    NotApplicable,
    /// On a Codex model but `~/.codex/auth.json` is missing.
    NoAuth,
    /// On a Codex model and authenticated, but with an api-key billing
    /// path that the subscription `wham/usage` endpoint doesn't cover.
    ApiKeyMode,
    Fetched(crate::codex_credits::CodexUsage),
    Failed(String),
}

/// Hard wall-clock ceiling for each `/usage` credits fetch. The inner
/// reqwest client already has a 5s request budget, but `codex_credits`
/// runs `refresh_if_stale` first against a *different* client that has
/// no timeout configured -- without an outer cancel the slash command
/// can hang on a stuck token refresh for as long as the OS waits on a
/// dead TCP connection. Sized to match the inner request budget so a
/// clean reqwest error normally surfaces first; the outer timeout only
/// fires when something upstream of the credits request itself stalls.
const USAGE_CREDITS_FETCH_TIMEOUT: Duration = Duration::from_secs(5);

/// Run the `/credits` lookup only when it makes sense:
/// - The active model is an OpenRouter one (`openrouter::<id>`), so the
///   balance is relevant to the bill the user is racking up right now;
///   AND
/// - A credential is resolvable via env or on-disk file (same
///   precedence as `build_openrouter_backend`).
///
/// Returns `Skipped` for any other configuration so the report stays
/// quiet on non-OpenRouter sessions instead of dragging in an
/// irrelevant network call.
pub(crate) async fn fetch_openrouter_credits_for_usage(
    model_wire_id: &str,
) -> OpenRouterCreditsOutcome {
    let active_source = split_wire_id(model_wire_id).map(|(src, _)| src);
    if active_source != Some(ModelSource::OPENROUTER) {
        return OpenRouterCreditsOutcome::Skipped;
    }
    let Some(key) = crate::openrouter_credits::active_api_key() else {
        return OpenRouterCreditsOutcome::Skipped;
    };
    match tokio::time::timeout(
        USAGE_CREDITS_FETCH_TIMEOUT,
        crate::openrouter_credits::fetch(&key),
    )
    .await
    {
        Ok(Ok(credits)) => OpenRouterCreditsOutcome::Fetched(credits),
        Ok(Err(e)) => OpenRouterCreditsOutcome::Failed(format!("{e:#}")),
        Err(_) => OpenRouterCreditsOutcome::Failed(format!(
            "timed out after {}s",
            USAGE_CREDITS_FETCH_TIMEOUT.as_secs()
        )),
    }
}

/// Run the Codex `wham/usage` lookup only when:
/// - The active model is a Codex one (`codex::<id>`), so the credits
///   are relevant to the bill the user is racking up right now; AND
/// - The on-disk `auth.json` is in `chatgpt` mode (api-key mode means
///   spend goes through the OpenAI billing dashboard, not the ChatGPT
///   subscription credits this endpoint reports).
///
/// Inspects `auth_status` before the network call so the report can
/// distinguish "no auth at all" from "api-key billing" -- the previous
/// implementation conflated them and told api-key users to re-run
/// `/setup codex` when their setup was already correct.
pub(crate) async fn fetch_codex_credits_for_usage(model_wire_id: &str) -> CodexCreditsOutcome {
    let active_source = split_wire_id(model_wire_id).map(|(src, _)| src);
    if active_source != Some(ModelSource::CODEX) {
        return CodexCreditsOutcome::NotApplicable;
    }
    match crate::codex_credits::auth_status() {
        Ok(crate::codex_credits::AuthStatus::Missing) => return CodexCreditsOutcome::NoAuth,
        Ok(crate::codex_credits::AuthStatus::ApiKeyMode) => {
            return CodexCreditsOutcome::ApiKeyMode;
        }
        Ok(crate::codex_credits::AuthStatus::ChatGptMode) => {}
        Err(e) => return CodexCreditsOutcome::Failed(format!("{e:#}")),
    }
    match tokio::time::timeout(USAGE_CREDITS_FETCH_TIMEOUT, crate::codex_credits::fetch()).await {
        Ok(Ok(Some(usage))) => CodexCreditsOutcome::Fetched(usage),
        // `fetch` only returns Ok(None) for the same conditions
        // `auth_status` already classified, but auth.json could have
        // been deleted between the two calls. Treat as no-auth in
        // that race.
        Ok(Ok(None)) => CodexCreditsOutcome::NoAuth,
        Ok(Err(e)) => CodexCreditsOutcome::Failed(format!("{e:#}")),
        Err(_) => CodexCreditsOutcome::Failed(format!(
            "timed out after {}s",
            USAGE_CREDITS_FETCH_TIMEOUT.as_secs()
        )),
    }
}

/// Capitalize a plan slug for human display. The wire form is
/// snake_case (`"free_workspace"`, `"self_serve_business_usage_based"`)
/// which reads poorly inline; we replace underscores with spaces and
/// title-case each word. Unknown slugs pass through unchanged because
/// the server may add new plan tokens we haven't seen.
pub(crate) fn humanize_plan_type(plan: &str) -> String {
    plan.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().chain(chars).collect::<String>(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Render `reset_after_seconds` as a short human-friendly string.
/// Returns `"soon"` for sub-minute values to avoid a noisy "0m" line.
pub(crate) fn format_reset_after(seconds: i32) -> String {
    if seconds <= 60 {
        return "soon".to_string();
    }
    let total = seconds as u64;
    let days = total / 86_400;
    let hours = (total % 86_400) / 3_600;
    let minutes = (total % 3_600) / 60;
    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {minutes}m")
    } else {
        format!("{minutes}m")
    }
}

/// `/usage` body: session token totals + USD cost + (for OpenRouter
/// sessions) the live credit balance. Kept pure so it's unit-testable
/// without standing up a session store or a mock HTTP server -- the
/// network call sits in `fetch_openrouter_credits_for_usage` above and
/// is invoked by the slash dispatch sites.
pub(crate) fn render_usage_report(
    snap: &crate::session::SessionSnapshot,
    usage: crate::llm_client::TokenUsage,
    cost_usd: Option<f64>,
    openrouter_credits: OpenRouterCreditsOutcome,
    codex_credits: CodexCreditsOutcome,
) -> String {
    let (model_display, source_display) = match split_wire_id(&snap.model) {
        Some((source, bare)) => (bare.to_string(), source.to_string()),
        None if snap.model.is_empty() => ("(none)".to_string(), "(unset)".to_string()),
        None => (snap.model.clone(), "(unknown)".to_string()),
    };

    let mut out = String::new();
    out.push_str("**Session usage**\n\n");
    out.push_str(&format!("- Model: `{model_display}` ({source_display})\n"));
    if usage.is_zero() {
        out.push_str("- Tokens: no LLM turns recorded yet this session\n");
    } else {
        out.push_str(&format!(
            "- Tokens: {} total (input {}, output {}, reasoning {}, cached read {}, cached write {})\n",
            usage.total_tokens(),
            usage.input_tokens,
            usage.output_tokens,
            usage.thought_tokens,
            usage.cached_read_tokens,
            usage.cached_write_tokens,
        ));
    }
    match cost_usd {
        Some(amount) => out.push_str(&format!("- Session cost: ${amount:.4} USD\n")),
        None if usage.is_zero() => {
            out.push_str("- Session cost: $0.0000 USD (no billable turns yet)\n")
        }
        None => out
            .push_str("- Session cost: unavailable (at least one turn lacked pricing metadata)\n"),
    }

    match openrouter_credits {
        OpenRouterCreditsOutcome::Fetched(credits) => {
            out.push_str(&format!(
                "- OpenRouter balance: ${:.4} remaining (${:.4} purchased − ${:.4} used)\n",
                credits.balance(),
                credits.total_credits,
                credits.total_usage,
            ));
        }
        OpenRouterCreditsOutcome::Failed(msg) => {
            out.push_str(&format!("- OpenRouter balance: lookup failed ({msg})\n"));
        }
        OpenRouterCreditsOutcome::Skipped => {
            // Two distinct skip reasons, distinguished by inspecting
            // the active model id. We keep the message short and
            // actionable rather than dumping internal state.
            if split_wire_id(&snap.model).map(|(src, _)| src) == Some(ModelSource::OPENROUTER) {
                out.push_str(
                    "- OpenRouter balance: no credential configured. Set \
                     OPENROUTER_API_KEY or run `/setup openrouter key <key>`.\n",
                );
            } else {
                out.push_str(
                    "- OpenRouter balance: not applicable (active model is not OpenRouter)\n",
                );
            }
        }
    }

    match codex_credits {
        CodexCreditsOutcome::Fetched(usage) => {
            let plan = usage
                .plan_type
                .as_deref()
                .filter(|s| !s.is_empty())
                .map(humanize_plan_type)
                .unwrap_or_else(|| "Unknown".to_string());
            out.push_str(&format!("- ChatGPT plan: {plan}\n"));
            match usage.credits {
                Some(c) if c.unlimited => {
                    out.push_str("- Codex credits: unlimited on this plan\n");
                }
                Some(c) if c.has_credits => match c.balance.as_deref() {
                    Some(balance) if !balance.is_empty() => {
                        out.push_str(&format!("- Codex credits: ${balance} remaining\n"));
                    }
                    _ => out.push_str("- Codex credits: available\n"),
                },
                Some(_) => {
                    out.push_str(
                        "- Codex credits: depleted (subscription rate-limit still applies)\n",
                    );
                }
                None => {
                    // No credits block -- typical for Pro/Team plans
                    // where access is gated by rate-limits rather than
                    // metered credits. Silent here; the rate-limit
                    // line below carries the actionable info.
                }
            }
            if let Some(window) = usage.rate_limit.and_then(|r| r.primary_window) {
                let reset = format_reset_after(window.reset_after_seconds);
                out.push_str(&format!(
                    "- Codex rate limit: {}% of primary window used (resets in {reset})\n",
                    window.used_percent
                ));
            }
        }
        CodexCreditsOutcome::Failed(msg) => {
            out.push_str(&format!(
                "- Codex (ChatGPT) status: lookup failed ({msg})\n"
            ));
        }
        CodexCreditsOutcome::NoAuth => {
            out.push_str(
                "- Codex (ChatGPT) status: no ChatGPT credentials. Run `/setup codex` \
                 to authenticate, or set `OPENAI_API_KEY` to bill against the API.\n",
            );
        }
        CodexCreditsOutcome::ApiKeyMode => {
            // The wham/usage endpoint only reports ChatGPT-subscription
            // state. Calls in api-key mode bill the OpenAI account
            // dashboard instead, so point the user there rather than
            // surfacing an irrelevant "balance: unknown" line.
            out.push_str(
                "- Codex billing: OPENAI_API_KEY (api-key mode); subscription credit \
                 balance not applicable. See platform.openai.com/usage for spend.\n",
            );
        }
        CodexCreditsOutcome::NotApplicable => {
            // Stay quiet on non-Codex sessions so the `/usage` report
            // doesn't grow a line per unconfigured provider.
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SessionMode;

    fn usage_snapshot(model: &str) -> crate::session::SessionSnapshot {
        crate::session::SessionSnapshot {
            cwd: std::path::PathBuf::from("/tmp/cwd"),
            additional_directories: Vec::new(),
            analysis_workspaces: None,
            mode: SessionMode::Lutz,
            model: model.into(),
            history: vec![],
            reasoning_effort: None,
            service_tier: None,
            idle_timeout_secs: None,
            project_instructions: String::new(),
            skills: std::sync::Arc::new(crate::skills::SkillRegistry::default()),
        }
    }

    #[test]
    fn render_usage_report_with_openrouter_balance_shows_all_lines() {
        let snap = usage_snapshot("openrouter::anthropic/claude-sonnet-4.5");
        let usage = crate::llm_client::TokenUsage {
            input_tokens: 1_000,
            output_tokens: 500,
            thought_tokens: 200,
            cached_read_tokens: 50,
            cached_write_tokens: 25,
        };
        let credits = crate::openrouter_credits::Credits {
            total_credits: 50.0,
            total_usage: 7.5,
        };
        let report = render_usage_report(
            &snap,
            usage,
            Some(0.1234),
            OpenRouterCreditsOutcome::Fetched(credits),
            CodexCreditsOutcome::NotApplicable,
        );

        assert!(report.contains("Model: `anthropic/claude-sonnet-4.5` (openrouter)"));
        // 1000+500+200+50+25 = 1775 across the five buckets.
        assert!(report.contains("1775 total"));
        assert!(report.contains("input 1000"));
        assert!(report.contains("reasoning 200"));
        assert!(report.contains("Session cost: $0.1234 USD"));
        assert!(
            report.contains("$42.5000 remaining"),
            "expected balance line in: {report}"
        );
        assert!(report.contains("$50.0000 purchased"));
        assert!(report.contains("$7.5000 used"));
    }

    #[test]
    fn render_usage_report_skips_openrouter_when_model_is_codex() {
        let snap = usage_snapshot("codex::gpt-5-codex");
        let report = render_usage_report(
            &snap,
            crate::llm_client::TokenUsage::default(),
            None,
            OpenRouterCreditsOutcome::Skipped,
            CodexCreditsOutcome::NoAuth,
        );
        assert!(report.contains("Model: `gpt-5-codex` (codex)"));
        assert!(report.contains("no LLM turns recorded yet this session"));
        // No billable turns yet should show $0 rather than the
        // "pricing unavailable" wording.
        assert!(report.contains("$0.0000 USD (no billable turns yet)"));
        assert!(report.contains("not applicable (active model is not OpenRouter)"));
        // NoAuth on a Codex session must surface the actionable hint --
        // the user is clearly trying to use ChatGPT-subscription routing.
        assert!(report.contains("Codex (ChatGPT) status: no ChatGPT credentials"));
        assert!(report.contains("/setup codex"));
    }

    #[test]
    fn render_usage_report_api_key_mode_on_codex_session_points_at_dashboard() {
        // api-key-mode Codex users (auth.json with OPENAI_API_KEY but
        // no chatgpt tokens) must not be told to run `/setup codex` --
        // their setup is correct, the subscription endpoint just
        // doesn't apply. Verify the renderer points them at the
        // OpenAI dashboard instead.
        let snap = usage_snapshot("codex::gpt-5-codex");
        let report = render_usage_report(
            &snap,
            crate::llm_client::TokenUsage::default(),
            None,
            OpenRouterCreditsOutcome::Skipped,
            CodexCreditsOutcome::ApiKeyMode,
        );
        assert!(report.contains("Codex billing: OPENAI_API_KEY"));
        assert!(report.contains("platform.openai.com/usage"));
        // Critically: must NOT tell them to re-authenticate.
        assert!(
            !report.contains("/setup codex"),
            "api-key-mode users should not be told to re-auth: {report}"
        );
    }

    #[test]
    fn render_usage_report_distinguishes_no_credential_for_openrouter_model() {
        let snap = usage_snapshot("openrouter::openai/gpt-4o");
        let report = render_usage_report(
            &snap,
            crate::llm_client::TokenUsage::default(),
            None,
            OpenRouterCreditsOutcome::Skipped,
            CodexCreditsOutcome::NotApplicable,
        );
        // The "no credential" branch must fire for OpenRouter models,
        // distinct from "not applicable" for non-OpenRouter ones.
        assert!(report.contains("no credential configured"));
        assert!(report.contains("/setup openrouter key"));
        // Skipped Codex on a non-Codex session stays silent so the
        // report doesn't grow a line per unconfigured provider.
        assert!(
            !report.contains("Codex"),
            "Codex line must be silent on non-Codex session: {report}"
        );
    }

    #[test]
    fn render_usage_report_surfaces_upstream_failure() {
        let snap = usage_snapshot("openrouter::openai/gpt-4o");
        let report = render_usage_report(
            &snap,
            crate::llm_client::TokenUsage::default(),
            None,
            OpenRouterCreditsOutcome::Failed("HTTP 401: invalid api key".into()),
            CodexCreditsOutcome::NotApplicable,
        );
        assert!(report.contains("OpenRouter balance: lookup failed"));
        assert!(report.contains("HTTP 401: invalid api key"));
    }

    #[test]
    fn render_usage_report_marks_partial_pricing_as_unavailable() {
        let snap = usage_snapshot("openrouter::openai/gpt-4o");
        let usage = crate::llm_client::TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        };
        // cost_usd = None *with* non-zero usage means at least one turn
        // ran without pricing metadata. Distinct from the no-LLM-yet
        // wording (covered above).
        let report = render_usage_report(
            &snap,
            usage,
            None,
            OpenRouterCreditsOutcome::Skipped,
            CodexCreditsOutcome::NotApplicable,
        );
        assert!(report.contains("Session cost: unavailable"));
        assert!(report.contains("at least one turn lacked pricing metadata"));
    }

    #[test]
    fn render_usage_report_handles_bare_model_id_without_panic() {
        // A bare id (no `source::` prefix) is legal on the wire when
        // the user typed a default model that routes to a backend.
        let snap = usage_snapshot("llama3:latest");
        let report = render_usage_report(
            &snap,
            crate::llm_client::TokenUsage::default(),
            None,
            OpenRouterCreditsOutcome::Skipped,
            CodexCreditsOutcome::NotApplicable,
        );
        assert!(report.contains("Model: `llama3:latest` ((unknown))"));
        assert!(report.contains("not applicable"));
    }

    #[test]
    fn render_usage_report_shows_codex_metered_credits() {
        let snap = usage_snapshot("codex::gpt-5-codex");
        let usage = crate::codex_credits::CodexUsage {
            plan_type: Some("plus".to_string()),
            credits: Some(crate::codex_credits::CreditStatus {
                has_credits: true,
                unlimited: false,
                balance: Some("12.50".to_string()),
            }),
            rate_limit: Some(crate::codex_credits::RateLimitStatus {
                primary_window: Some(crate::codex_credits::RateLimitWindow {
                    used_percent: 42,
                    reset_after_seconds: 7_200,
                }),
            }),
        };
        let report = render_usage_report(
            &snap,
            crate::llm_client::TokenUsage::default(),
            None,
            OpenRouterCreditsOutcome::Skipped,
            CodexCreditsOutcome::Fetched(usage),
        );
        assert!(report.contains("ChatGPT plan: Plus"));
        assert!(report.contains("Codex credits: $12.50 remaining"));
        assert!(report.contains("Codex rate limit: 42% of primary window used"));
        assert!(report.contains("resets in 2h 0m"));
    }

    #[test]
    fn render_usage_report_shows_codex_unlimited_plan() {
        let snap = usage_snapshot("codex::gpt-5-codex");
        let usage = crate::codex_credits::CodexUsage {
            plan_type: Some("pro".to_string()),
            credits: Some(crate::codex_credits::CreditStatus {
                has_credits: true,
                unlimited: true,
                balance: None,
            }),
            rate_limit: None,
        };
        let report = render_usage_report(
            &snap,
            crate::llm_client::TokenUsage::default(),
            None,
            OpenRouterCreditsOutcome::Skipped,
            CodexCreditsOutcome::Fetched(usage),
        );
        assert!(report.contains("ChatGPT plan: Pro"));
        assert!(report.contains("Codex credits: unlimited on this plan"));
        // No rate-limit info on the wire -> renderer omits the line
        // rather than guessing.
        assert!(!report.contains("Codex rate limit"));
    }

    #[test]
    fn render_usage_report_codex_failure_includes_diagnostic() {
        let snap = usage_snapshot("codex::gpt-5-codex");
        let report = render_usage_report(
            &snap,
            crate::llm_client::TokenUsage::default(),
            None,
            OpenRouterCreditsOutcome::Skipped,
            CodexCreditsOutcome::Failed(
                "chatgpt /wham/usage returned HTTP 401: invalid_token".into(),
            ),
        );
        assert!(report.contains("Codex (ChatGPT) status: lookup failed"));
        assert!(report.contains("HTTP 401"));
        // The actual bearer token must never appear in the failure
        // line -- we only echo upstream body excerpts, not local state.
        assert!(!report.contains("Bearer"));
    }

    #[test]
    fn humanize_plan_type_title_cases_known_snakes() {
        assert_eq!(humanize_plan_type("plus"), "Plus");
        assert_eq!(humanize_plan_type("free_workspace"), "Free Workspace");
        assert_eq!(
            humanize_plan_type("self_serve_business_usage_based"),
            "Self Serve Business Usage Based"
        );
        // Unknown slugs pass through with whatever casing the server sent.
        assert_eq!(humanize_plan_type("hyperspace"), "Hyperspace");
    }

    #[test]
    fn format_reset_after_renders_sane_durations() {
        assert_eq!(format_reset_after(0), "soon");
        assert_eq!(format_reset_after(30), "soon");
        assert_eq!(format_reset_after(60), "soon");
        assert_eq!(format_reset_after(125), "2m");
        assert_eq!(format_reset_after(3_600), "1h 0m");
        assert_eq!(format_reset_after(7_320), "2h 2m");
        assert_eq!(format_reset_after(90_000), "1d 1h");
    }

    #[test]
    fn usage_by_model_keeps_model_attribution_for_cost_and_acp_metadata() {
        let flash_usage = crate::llm_client::TokenUsage {
            input_tokens: 100,
            output_tokens: 10,
            thought_tokens: 2,
            cached_read_tokens: 20,
            cached_write_tokens: 0,
        };
        let pro_usage = crate::llm_client::TokenUsage {
            input_tokens: 200,
            output_tokens: 30,
            thought_tokens: 4,
            cached_read_tokens: 40,
            cached_write_tokens: 0,
        };
        let usage_by_model = BTreeMap::from([
            ("deepseek::deepseek-v4-flash".to_string(), flash_usage),
            ("deepseek::deepseek-v4-pro".to_string(), pro_usage),
        ]);
        let meta = usage_by_model_meta(&usage_by_model);
        assert_eq!(
            meta["anvil"]["usageByModel"]["deepseek::deepseek-v4-pro"]["totalTokens"],
            pro_usage.total_tokens()
        );

        let mut failure_meta = meta;
        insert_turn_failure_meta(
            &mut failure_meta,
            &crate::tool_loop::TurnFailure {
                retryable: false,
                message: "supervisor produced no valid decision".to_string(),
            },
        );
        assert_eq!(
            failure_meta["anvil"]["turnFailure"]["message"],
            "supervisor produced no valid decision"
        );
        assert_eq!(
            failure_meta["anvil"]["usageByModel"]["deepseek::deepseek-v4-pro"]["totalTokens"],
            pro_usage.total_tokens()
        );
    }

    #[test]
    fn bedrock_credit_metadata_attaches_and_preserves_existing_anvil_metadata() {
        let mut meta = usage_by_model_meta(&BTreeMap::from([(
            "bedrock::model".to_string(),
            crate::llm_client::TokenUsage::default(),
        )]));
        insert_turn_failure_meta(
            &mut meta,
            &crate::tool_loop::TurnFailure {
                retryable: true,
                message: "retry".into(),
            },
        );
        insert_bedrock_credits_meta(
            &mut meta,
            "bedrock::model",
            crate::bedrock_credits::Status::Unavailable {
                reason: "billing credentials are unavailable".into(),
                as_of: "2026-07-15T18:42:00Z".into(),
            },
        );
        assert!(meta["anvil"]["usageByModel"].is_object());
        assert_eq!(meta["anvil"]["turnFailure"]["message"], "retry");
        assert_eq!(meta["anvil"]["bedrockCredits"]["status"], "unavailable");
    }

    #[test]
    fn bedrock_credit_metadata_does_not_attach_for_openrouter() {
        let mut meta = serde_json::Map::new();
        insert_bedrock_credits_meta(
            &mut meta,
            "openrouter::model",
            crate::bedrock_credits::Status::Unavailable {
                reason: "billing credentials are unavailable".into(),
                as_of: "2026-07-15T18:42:00Z".into(),
            },
        );
        assert!(meta.is_empty());
    }

    #[test]
    fn non_bedrock_usage_metadata_does_not_lookup_bedrock_credits() {
        let mut meta = serde_json::Map::new();
        let mut looked_up = false;

        attach_bedrock_credits_meta(&mut meta, "openrouter::model", || {
            looked_up = true;
            crate::bedrock_credits::Status::Unavailable {
                reason: "refresh pending".into(),
                as_of: "2026-07-15T18:42:00Z".into(),
            }
        });

        assert!(!looked_up);
        assert!(meta.is_empty());
    }

    #[test]
    fn openrouter_balance_metadata_attaches_and_preserves_existing_anvil_metadata() {
        let mut meta = usage_by_model_meta(&BTreeMap::from([(
            "openrouter::vendor/model".to_string(),
            crate::llm_client::TokenUsage::default(),
        )]));
        insert_turn_failure_meta(
            &mut meta,
            &crate::tool_loop::TurnFailure {
                retryable: true,
                message: "retry".into(),
            },
        );
        insert_openrouter_balance_meta(
            &mut meta,
            "openrouter::vendor/model",
            crate::openrouter_credits::Status::Available {
                remaining_usd: 0.0,
                total_credits_usd: 10.0,
                total_usage_usd: 10.0,
                as_of: "2026-07-15T18:42:00Z".into(),
            },
        );
        assert!(meta["anvil"]["usageByModel"].is_object());
        assert_eq!(meta["anvil"]["turnFailure"]["message"], "retry");
        assert_eq!(meta["anvil"]["openrouterBalance"]["status"], "available");
        assert_eq!(meta["anvil"]["openrouterBalance"]["remainingUsd"], 0.0);
    }

    #[test]
    fn openrouter_balance_metadata_is_lazy_for_other_providers() {
        let mut meta = serde_json::Map::new();
        let mut looked_up = false;

        attach_openrouter_balance_meta(&mut meta, "bedrock::model", || {
            looked_up = true;
            crate::openrouter_credits::Status::Unavailable {
                reason: "refresh pending".into(),
                as_of: "2026-07-15T18:42:00Z".into(),
            }
        });

        assert!(!looked_up);
        assert!(meta.is_empty());
        assert!(!is_openrouter_model("openrouter::"));
    }

    #[test]
    fn deepseek_balance_metadata_attaches_and_preserves_existing_anvil_metadata() {
        let mut meta = usage_by_model_meta(&BTreeMap::from([(
            "deepseek::deepseek-chat".to_string(),
            crate::llm_client::TokenUsage::default(),
        )]));
        insert_turn_failure_meta(
            &mut meta,
            &crate::tool_loop::TurnFailure {
                retryable: true,
                message: "retry".into(),
            },
        );
        insert_deepseek_balance_meta(
            &mut meta,
            "deepseek::deepseek-chat",
            crate::deepseek_balance::Status::Available {
                balances: vec![crate::deepseek_balance::Balance {
                    currency: "CNY".into(),
                    total_balance: "12.3400".into(),
                    granted_balance: "2.0000".into(),
                    topped_up_balance: "10.3400".into(),
                }],
                as_of: "2026-07-15T18:42:00Z".into(),
            },
        );
        assert!(meta["anvil"]["usageByModel"].is_object());
        assert_eq!(meta["anvil"]["turnFailure"]["message"], "retry");
        assert_eq!(meta["anvil"]["deepseekBalance"]["status"], "available");
        assert_eq!(
            meta["anvil"]["deepseekBalance"]["balances"][0]["totalBalance"],
            "12.3400"
        );
    }

    #[test]
    fn deepseek_balance_metadata_is_lazy_for_other_providers() {
        let mut meta = serde_json::Map::new();
        let mut looked_up = false;
        attach_deepseek_balance_meta(&mut meta, "openrouter::model", || {
            looked_up = true;
            crate::deepseek_balance::Status::Unavailable {
                reason: "refresh pending".into(),
                as_of: "2026-07-15T18:42:00Z".into(),
            }
        });
        assert!(!looked_up);
        assert!(meta.is_empty());
        assert!(!is_deepseek_model("deepseek::"));
    }
}
