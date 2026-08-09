# Qwen 3.5 4B vs Gemma 4 E4B: candidate-window summarization

## Recommendation

Use **Gemma 4 E4B** as the first production candidate, with the explicit read-only guard in the prompt. Qwen 3.5 4B is competitive on literal retention and tends to preserve more adverse evidence, but it is materially more prone to treating advice or previously established state as work completed in the captured read-only window. Gemma is more conservative and grounds edit locations better.

This is a recommendation for a follow-up integration probe, not a claim that Gemma is globally the better model. The archived v9 traces lack attributable terminal candidate responses, and the weak reference is a cumulative supervisor summary rather than a window-local gold summary.

## Cohorts and budget

The extractor found 1,443 attributable v9 candidate-lane windows:

- 300 positively read-only: only known non-writing tools
- 730 positively edit-producing: an explicit edit/write tool or unmistakable in-place editor
- 413 ambiguous shell-only windows, excluded from both cohorts

Two deterministic 100-row cohorts each cover all 20 tasks and all 40 runs. Read-only windows range from 1,835 to 169,024 rendered bytes (median 9,667); edit-producing windows range from 2,910 to 93,922 bytes (median 14,349).

Each request is tokenized by the serving model. The completion ceiling is 1× the complete tokenized summarizer input, capped by remaining context. Compactness is therefore measured without a common small ceiling, while a model that cannot finish a summary before consuming input-sized output is still counted as failing.

## Results

### Primary: read-only

The prototype prompt by itself is unsafe for read-only windows:

| Metric | Qwen 3.5 4B | Gemma 4 E4B |
|---|---:|---:|
| Valid summaries | 94/100 | 99/100 |
| Request errors | 2 | 0 |
| Schema-invalid / degenerate | 4 | 1 |
| Summaries claiming edits | 86/94 (91.5%) | 80/99 (80.8%) |
| Claimed edit-location grounding | 63.5% | 66.0% |

The read-only guard says that the window is positively classified read-only, requires `edits: []`, and warns that reads/verification may expose changes made before the window. With that guard:

| Metric | Qwen 3.5 4B | Gemma 4 E4B |
|---|---:|---:|
| Valid summaries | 100/100 | 100/100 |
| Summaries claiming edits | 0 | 0 |
| Median completion tokens | 315.5 | 343 |
| P90 completion tokens | 701 | 562 |
| Weak-reference literal recall, mean | 0.236 | 0.222 |
| Paired weak-reference wins | 11 | 14 |
| Paired ties | 32 | 32 |
| Completion-like `direction` values | 44 | 2 |

The completion-direction count is a risk proxy, not a gold-label metric. Its practical significance was confirmed in a focused audit of the largest reference-recall disagreements. Qwen frequently converted advisory instructions into completed facts—for example claiming all 12 mux tests passed and a race was fixed when the window only read `handleCloseStreamLocked`. Gemma described the actual inspection and retained the unresolved race. Similar overclaiming appeared in the action-pinning and Request.formData windows.

The supervisor pseudo-reference sometimes describes cumulative candidate state that is absent from the captured window. Literal recall can therefore reward exactly this Qwen behavior. The paired recall difference should not override the transcript-faithfulness audit.

### Secondary: edit-producing

| Metric | Qwen 3.5 4B | Gemma 4 E4B |
|---|---:|---:|
| Valid summaries | 100/100 | 100/100 |
| Summaries detecting edits | 100/100 | 100/100 |
| Claimed edit-location grounding | 91.9% | 95.5% |
| Weak-reference literal recall, mean | 0.239 | 0.199 |
| Median completion tokens | 402 | 477 |
| Failed evidence items retained | 81 | 30 |

Both models are viable on explicit-edit windows. Qwen is terser at the median and reports more adverse evidence, but manual inspection found some invented checks and overconfident completion claims. Gemma's edit locations are more consistently traceable to the source transcript and its language is generally more conservative. The evidence supports Gemma, but less decisively than on read-only windows.

## Degenerate tails

- Qwen timed out after 900 seconds on two unguarded read-only cases while consuming source-sized budgets, and produced four other invalid unguarded read-only summaries.
- Gemma produced one unguarded read-only failure by repeating escaped tab characters inside an alleged code edit until it exhausted a 6,134-token input-sized budget.
- The explicit read-only guard eliminated all request/schema failures for both models in the 100-row primary cohort.

These are model/prompt outcomes, not artifacts of the original 1,400-token prototype ceiling.

## Performance caveat

Latency is not an apples-to-apples model benchmark. Gemma ran with vLLM compilation/CUDA graphs; Qwen required eager, language-model-only serving to avoid disproportionate multimodal/compilation startup on this WSL host. Both used continuous batching with concurrency 16 and stable GPU UUID selection. Gemma's observed latency was much lower, but the quality recommendation does not rely on that difference.

## Limitations and next probe

Facts:

- v9 traces contain candidate requests and window boundaries but not lane-attributable terminal responses.
- Each row is the final attributable request for a lane, restricted to the current-window suffix; `terminal_response_included` is false.
- Supervisor summaries exist for 59 read-only and 53 edit-producing selected-lane rows, but only 57/52 contain literals usable by the weak metric.

Inference:

- Gemma plus the guard is the safer summarizer for a supervisor whose first requirement is not inventing work.
- Qwen may be preferable only if maximizing adverse-detail recall matters more than conservative attribution.

Speculation to test next:

- A window-local verifier prompt that requires every progress/evidence claim to cite a message or tool-call index may reduce the remaining cumulative-state hallucinations.
- Exact replay traces with response IDs could change the ordering; collect those before making the model choice permanent.
