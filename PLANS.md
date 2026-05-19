# Feature: Context Compression (Context Window Management)

Currently, Anvil sends the entire session history to the LLM for every prompt. As sessions grow, this will eventually exceed the model's context window or increase latency and cost. This feature introduces a mechanism to manage and compress the context window.

## Goals
- Prevent context window overflow.
- Reduce token usage for long sessions while maintaining critical state.
- Provide a configurable strategy for how context is pruned or compressed.

## Proposed Design

### 1. Context Management Strategy
We will implement a `ContextStrategy` trait to allow for different compression methods:
- **Sliding Window**: Keep the last `N` turns.
- **Summarization**: Summarize early parts of the conversation into a "system memory" block.
- **Hybrid**: Keep a system prompt, a summary of old turns, and a sliding window of recent turns.

### 2. Implementation Steps

#### Phase 1: Token Counting & Monitoring
- [ ] Integrate a lightweight token counting library (or use model-specific estimates) to track the current context size.
- [ ] Add logic in `src/agent.rs` to determine when the context exceeds a predefined threshold.

#### Phase 2: The Compression Engine
- [ ] Create `src/context_manager.rs` to handle the logic of selecting which messages to keep/compress.
- [ ] Implement the `Summarization` strategy:
    - When the threshold is hit, take the oldest `X` turns.
    - Send them to the LLM with a specific "summarize this conversation" prompt.
    - Store the resulting summary in the `Session` state.
- [ ] Implement the `Sliding Window` strategy as a fallback/alternative.

#### Phase 3: Session Store Integration
- [ ] Update `src/session.rs` to store the "Conversation Summary" and the "Compression Pivot Point" (the index before which everything is summarized).
- [ ] Ensure that when the session is saved/loaded, the summary is persisted.

#### Phase 4: LLM Request Modification
- [ ] Modify the prompt construction in `src/agent.rs` to:
    1. Include the System Prompt.
    2. Include the Conversation Summary (if it exists).
    3. Include the turns from the Pivot Point onwards.

#### Phase 5: Configuration & Testing
- [ ] Add a configuration option in `/setup advanced` to choose the compression strategy (e.g., `none`, `sliding_window`, `summarize`).
- [ ] Add unit tests for `context_manager.rs` to ensure turns are pruned/summarized correctly.
- [ ] Add an integration test with a mocked LLM to verify that summaries are inserted into the prompt.

## Technical Considerations
- **Token Overhead**: The summarization process itself requires an LLM call, which adds latency. This should be done asynchronously or as part of the prompt turn if possible.
- **Information Loss**: Summarization is lossy. We must ensure that "critical" information (like specific file paths mentioned early on) is preserved in the summary.
- **Model Compatibility**: Different models have wildly different context windows. The threshold should be model-dependent (fetched from `ModelMetadata`).
