//! Anvil's LLM client, extracted as a standalone crate.
//!
//! Everything needed to discover, authenticate against, and stream chat
//! completions from the supported providers lives here, with no dependency on
//! the rest of Anvil:
//!
//! - [`llm_client`]: core request/response types (`ChatMessage`, `ToolDefinition`,
//!   `StreamChatRequest`, `LlmResponse`, …), the [`llm_client::LlmBackend`]
//!   trait every provider implements, and the generic OpenAI-compatible
//!   [`llm_client::OpenAiClient`].
//! - [`multi_backend`]: [`multi_backend::MultiBackend`], a router that fans a
//!   `<source>::<id>` wire id out to the right provider backend.
//! - [`discovery`]: model-catalog discovery and the `split_wire_id` wire-id
//!   convention shared with Anvil.
//! - [`infer`]: one-shot, tool-free, schema-constrained inference for callers
//!   that do not need an ACP session or Anvil's agent loop.
//! - Provider backends: [`bedrock_client`], [`codex_client`], [`grok_client`],
//!   and user-configured OpenAI-compatible endpoints via [`openai_providers`].
//! - Auth and credential storage: [`secrets`], [`codex_auth`], [`bedrock_auth`],
//!   [`grok_auth`], [`kimi_auth`], [`openrouter_auth`], [`deepseek_auth`].
//! - Credit/balance probes: [`bedrock_credits`], [`codex_credits`],
//!   [`openrouter_credits`], [`deepseek_balance`].
//! - Streaming/transport plumbing: [`http_retry`], [`responses_api`],
//!   [`responses_chain`], [`structured_output`], [`tool_arguments`],
//!   [`trace_logging`].
//!
//! The crate's default feature set does not pull in the AWS SDK. Consumers
//! that use the Bedrock credit probe must opt into `bedrock-credits`.

pub mod bedrock_auth;
pub mod bedrock_client;
#[cfg(feature = "bedrock-credits")]
pub mod bedrock_credits;
pub mod codex_auth;
pub mod codex_client;
pub mod codex_credits;
pub mod deepseek_auth;
pub mod deepseek_balance;
pub mod discovery;
pub mod grok_auth;
pub mod grok_client;
pub mod http_retry;
pub mod infer;
pub mod kimi_auth;
pub mod llm_client;
pub mod multi_backend;
pub mod openai_providers;
pub mod openrouter_auth;
pub mod openrouter_credits;
pub mod responses_api;
pub mod responses_chain;
pub mod secrets;
pub mod structured_output;
pub mod tool_arguments;
pub mod trace_logging;
