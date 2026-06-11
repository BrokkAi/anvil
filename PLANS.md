# Context Window Management — shipped

A history of conversation, especially with tool-use, will eventually
exceed the model's context window. Anvil's defense is a **per-turn
summary** layer, persisted in the existing `summaryContentId` slot of
the session zip (the same slot Brokk's Java side already reads).

This document describes what's in the tree today. Open issues and
follow-up work are listed at the bottom.

## What lands in the prompt

`build_prompt_messages` in `src/agent.rs` walks the session's
`ConversationTurn` history and, for each turn:

- If the turn carries a `summary`, emit a single user message wrapping
  it in `<conversation_summary>...</conversation_summary>`. The
  verbatim user prompt / tool exchanges / assistant response for that
  turn are **not** re-emitted.
- Otherwise replay the turn verbatim (user prompt, optional
  `assistant_tool_calls` + `tool_result` pairs, optional assistant text).

The full log stays on disk regardless — `set_turn_summary` only
flips the `summaryContentId` reference. A session reload from disk
reproduces the same prompt the live session would build.

## Two trigger paths

### Automatic, per prompt

`build_prompt_messages_with_compression` runs before each turn's
prompt is dispatched. It:

1. Builds the messages.
2. Estimates tokens via `tokens::approximate_tokens_messages` (the
   same `o200k_base` tokenizer used by Brokk's
   `Messages.getApproximateTokens`).
3. If projected tokens exceed `context_budget(model_context_length)`
   — i.e. 75% of the model's declared window, falling back to 128k —
   walks the history oldest-first and compresses each uncompressed
   turn via `summarize_turn`, mutating the in-memory snapshot and
   persisting via `set_turn_summary` until the prompt fits or every
   turn is compressed.

### User-initiated: `/compress`

Runs the same compression loop, but unconditionally for every
uncompressed turn, with progress notifications streamed to the
client. Dispatched after `start_prompt` so a `session/cancel` aborts
the in-flight summarization and stops the loop between turns.

Idempotent: re-running reports "Nothing to compress: N turn(s), all
already summarized." Failed turns stay verbatim and the rest of the
session still gets compressed.

## How a single turn gets summarized

`context_manager::summarize_turn` is the single entry point.

**Fast path.** If the full turn (user prompt + tool exchanges +
assistant response, wrapped with `SYSTEM_PROMPT_TURN`) fits inside
`summarizer_input_budget = 65% * context_length` (fallback 128k →
~83k tokens), it goes out as one LLM call. The model returns a
bulleted summary inside `<conversation_summary>` tags; tags are
stripped before persistence.

**Hierarchical path.** When the turn is too big for one call:

1. **Atomize.** Split the turn into ordered atoms: `User: <prompt>`,
   one `Tool <name> args=... -> <result>` per exchange, `Assistant:
   <response>`.
2. **Pack.** Greedily pack atoms into chunks at
   `per_chunk_budget = budget - CHUNK_OVERHEAD_TOKENS`, with a hard
   floor of `MIN_CHUNK_TOKENS`. Atoms larger than the per-chunk
   budget are split — by line first, then by character as a
   last-resort fallback.
3. **Chunk summarize.** Each chunk goes through the LLM with
   `SYSTEM_PROMPT_CHUNK`, which tells the model it's summarizing
   *one part of a larger turn* and that a later step will combine
   its output. Output is plain bullets, no tags.
4. **Meta summarize.** Combined chunk summaries are fed back to
   the LLM with `SYSTEM_PROMPT_META`, which produces one coherent
   summary wrapped in `<conversation_summary>` tags. If the
   combined input still overruns, recurses — chunk the combined
   summaries, meta-pass over those, etc.

The recursion is bounded: each level shrinks the input
materially (chunk summaries are bullet-only output), and the
`MIN_CHUNK_TOKENS` floor combined with a "no further split possible"
check in `combine_chunk_summaries` guarantees termination.

This guarantees a session never reaches the "too big to send AND too
big to compress" wedge state: every turn either gets a summary (lossy
if the original was monstrous, but the verbatim log stays on disk) or
returns a clean `Err` that the caller surfaces to the user.

## On-disk schema

No new files. The summary text lives under `content/<uuid>.txt` and
is referenced from the `summaryContentId` field of the matching task
entry in `contexts.jsonl` — the exact slot Brokk's Java side already
reads. Sessions compressed by Anvil open cleanly in Brokk and render
the "compressed" indicator; sessions compressed by Brokk get their
summaries picked up by Anvil's load path.

`append_turn_to_zip` returns the assigned fragment id so the
in-memory `ConversationTurn` knows where it lives on disk;
`set_turn_summary` uses that id to surgically rewrite just the
matching task's `summaryContentId` plus add one new `content/*.txt`
blob, atomically via `with_temp_zip_writer`.

## Counterpart in Brokk

This layer mirrors Brokk's `ContextManager.compressHistory(TaskEntry)`
and `ContextManager.compressHistoryAsync(Context)`. Differences:

| Concern | Brokk | Anvil |
| --- | --- | --- |
| Wedge prevention | Upstream via `ContextSizeGuard` at add-time | Downstream via hierarchical summarization |
| Concurrency | Parallel via `compressHistoryAsync` | Sequential (one prompt per session at a time, gated by `start_prompt`) |
| Trigger | UI button + automatic in `ArchitectAgent` | `/compress` slash command + automatic per-prompt threshold |
| Oversized turn | `compressHistory` returns the original on failure | `summarize_turn` chunks and recurses; returns `Err` only on actual LLM/network failure |

## Concurrency

Chunk summarization within `summarize_turn_hierarchical` and the
recursive meta path in `combine_chunk_summaries` both run through
`summarize_chunks_parallel`, which uses `futures::buffered(N)` with
`MAX_CONCURRENT_CHUNK_REQUESTS = 2` to keep up to two chunk requests
in flight at once. The cap is intentionally low: raising it without
a per-backend rate-limit story will trigger `429`s on long compress
runs.

`buffered` preserves submission order, so the combine step sees
chunk summaries in chronological turn order. `try_collect`
short-circuits on the first `Err` so a single chunk failure aborts
the rest of the run rather than burning credits on doomed work.

## Open follow-ups

- **No input-time gate.** Anvil could add one (ACP's `session/prompt`
  is a real server-side intake; tool output is already bounded
  per-call). The downstream summarizer makes this optional, not
  required.
- **Cost surfacing.** A `/compress` run on a long session can fire
  many LLM calls (one per chunk + meta). We report verbatim-vs-summary
  token tallies but don't expose dollar/credit cost.
- **Summarizer model selection.** Today summarization uses the
  session's active model with `reasoning_effort: "low"`. A
  `/setup advanced` knob would let users route summarization to a
  cheaper model.
- **Provider-aware concurrency.** Once we have provider-level rate
  awareness, `MAX_CONCURRENT_CHUNK_REQUESTS` can be lifted (or made
  per-backend) to speed up long compress runs.
