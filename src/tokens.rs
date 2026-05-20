//! Token counting for context-window accounting.
//!
//! Mirrors Brokk's Java implementation in
//! `brokk-core/src/main/java/ai/brokk/util/Messages.java`, which uses
//! jtokkit's `O200K_BASE` encoding (the GPT-4o / o-series tokenizer)
//! uniformly for every model. We follow the same convention so a
//! session zip opened in either Anvil or Brokk reports the same
//! number. The count is explicitly an approximation -- for non-OpenAI
//! models (Claude, Gemini, Llama) it will diverge from the provider's
//! true tokenizer by single-digit percentages, which is fine because
//! we use it to trigger compression at a configurable threshold, not
//! to predict billing.
//!
//! The encoder is initialized lazily on first use and reused for the
//! lifetime of the process. `tiktoken-rs::o200k_base` returns a
//! `CoreBPE` which is `Send + Sync`, so a single global instance is
//! safe across threads.

use std::sync::OnceLock;

use tiktoken_rs::CoreBPE;
use tiktoken_rs::o200k_base;

use crate::llm_client::ChatMessage;

fn encoder() -> &'static CoreBPE {
    static ENCODER: OnceLock<CoreBPE> = OnceLock::new();
    ENCODER.get_or_init(|| o200k_base().expect("o200k_base tokenizer initializes"))
}

/// Approximate token count for a single string. Matches Brokk's
/// `Messages.getApproximateTokens(String)`.
pub fn approximate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    encoder().encode_with_special_tokens(text).len()
}

/// Approximate token count for a full chat message list. Sums the
/// content of each message plus its tool-call payloads. Does not add
/// per-message structural overhead (the ~4 tokens per message in
/// OpenAI's published formula) because that overhead is provider-
/// specific and would mislead users on non-OpenAI backends; the
/// threshold should be set with a safety margin instead.
///
/// Currently unused outside tests -- Phase 2 of the context-compression
/// feature (PLANS.md) calls this from the threshold check before
/// `build_prompt_messages` to decide whether to summarize. Kept public
/// here so the API surface is reviewable in isolation.
#[allow(dead_code)]
pub fn approximate_tokens_messages(messages: &[ChatMessage]) -> usize {
    let mut total = 0usize;
    for msg in messages {
        if let Some(content) = msg.content.as_deref() {
            total += approximate_tokens(content);
        }
        if let Some(calls) = msg.tool_calls.as_ref() {
            for call in calls {
                total += approximate_tokens(&call.function.name);
                total += approximate_tokens(&call.function.arguments);
            }
        }
        if let Some(name) = msg.name.as_deref() {
            total += approximate_tokens(name);
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_client::{FunctionCall, ToolCall};

    #[test]
    fn empty_string_is_zero_tokens() {
        assert_eq!(approximate_tokens(""), 0);
    }

    #[test]
    fn short_ascii_string_counts_reasonably() {
        // "Hello, world!" tokenizes to a small handful under o200k_base;
        // pin a loose range rather than the exact count so a future
        // tiktoken-rs bump that re-tunes the vocab doesn't break this.
        let n = approximate_tokens("Hello, world!");
        assert!(n > 0 && n < 10, "got {n} tokens for 'Hello, world!'");
    }

    #[test]
    fn longer_text_counts_more_than_short_text() {
        let short = approximate_tokens("hi");
        let long = approximate_tokens(
            "The quick brown fox jumps over the lazy dog. \
             The quick brown fox jumps over the lazy dog.",
        );
        assert!(long > short);
    }

    #[test]
    fn approximate_tokens_messages_sums_content() {
        let msgs = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
        ];
        let total = approximate_tokens_messages(&msgs);
        let pieces = approximate_tokens("You are a helpful assistant.")
            + approximate_tokens("Hello")
            + approximate_tokens("Hi there!");
        assert_eq!(total, pieces);
    }

    #[test]
    fn approximate_tokens_messages_counts_tool_call_payloads() {
        let msg = ChatMessage::assistant_tool_calls(vec![ToolCall {
            id: "call_1".to_string(),
            r#type: "function".to_string(),
            function: FunctionCall {
                name: "search".to_string(),
                arguments: r#"{"query":"context compression"}"#.to_string(),
            },
        }]);
        let total = approximate_tokens_messages(std::slice::from_ref(&msg));
        // Name + arguments both contribute; total must exceed each piece.
        assert!(total >= approximate_tokens("search"));
        assert!(total >= approximate_tokens(r#"{"query":"context compression"}"#));
    }

    #[test]
    fn approximate_tokens_messages_ignores_none_content() {
        // assistant_tool_calls() leaves `content: None` -- must not panic
        // or double-count.
        let msg = ChatMessage::assistant_tool_calls(vec![]);
        assert_eq!(approximate_tokens_messages(std::slice::from_ref(&msg)), 0);
    }
}
