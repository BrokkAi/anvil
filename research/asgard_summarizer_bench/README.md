# Qwen 3.5 4B vs Gemma 4 E4B summarizer probe

This probe samples two disjoint analytical cohorts from the 40-run Asgard v9 corpus: 100 read-only candidate-lane windows (the primary question) and 100 edit-producing windows (secondary). It sends identical handoff prompts to `Qwen/Qwen3.5-4B` and `google/gemma-4-E4B-it` through vLLM. Concurrent HTTP requests exercise vLLM continuous batching (`--concurrency 16`), rather than a Transformers loop whose nominal batch is defeated by highly variable prompt lengths.

`read-only` contains only windows whose captured tool calls are from known non-writing tools. `edit-producing` requires positive mutation evidence: an explicit `edit`/`write_file` call, or an unmistakable in-place shell editor. General-purpose shell-only windows are assigned `ambiguous-shell` and excluded from both benchmark cohorts—even when the command looks like a build or test—because incidental and opaque writes cannot be ruled out. Classification evidence is retained in each row's `write_evidence`.

The archived v9 traces predate the dedicated `summarize_candidate_window` call. They trace each candidate request and every completed Asgard window, but responses have no request/lane ID. The extractor therefore takes the final attributable request for each lane and retains the current-window suffix beginning at its latest `<asgard_next_window_advice>`. It intentionally does not guess which concurrently completed terminal response belongs to which lane; each record says `terminal_response_included: false`. Thus this is a representative-ish summarizer probe, not an exact replay corpus.

Create the deterministic sample:

```bash
python3 research/asgard_summarizer_bench/extract_v9_windows.py \
  --analysis /tmp/asgard-v9-analysis.json \
  --output research/asgard_summarizer_bench/v9_sample_100.jsonl \
  --read-only-output research/asgard_summarizer_bench/v9_read_only_100.jsonl \
  --edit-producing-output research/asgard_summarizer_bench/v9_edit_producing_100.jsonl \
  --all-output /tmp/asgard-v9-candidate-windows.jsonl
```

Use Python 3.13 because the newest vLLM/model support is ahead of older stable combinations:

```bash
uv venv --python 3.13 .venv-vllm
uv pip install --python .venv-vllm/bin/python vllm --torch-backend=auto
```

On WSL, select cards by stable UUID. GPU1 is `GPU-d2383b55-ffa4-3529-3288-18f447a66ec8`; GPU3 is `GPU-27368065-5d8b-6e5e-4e81-2e924bd9ce73`.

```bash
CUDA_VISIBLE_DEVICES=GPU-d2383b55-ffa4-3529-3288-18f447a66ec8 \
  VLLM_USE_V2_MODEL_RUNNER=0 \
  .venv-vllm/bin/vllm serve Qwen/Qwen3.5-4B \
  --host 127.0.0.1 --port 8101 --served-model-name qwen35-4b \
  --max-model-len 65536 --max-num-seqs 32 --gpu-memory-utilization 0.90 \
  --enforce-eager --language-model-only

python3 research/asgard_summarizer_bench/run_openai_batch.py \
  --dataset research/asgard_summarizer_bench/v9_read_only_100.jsonl \
  --dataset research/asgard_summarizer_bench/v9_edit_producing_100.jsonl \
  --output research/asgard_summarizer_bench/qwen35-4b-stratified-results.jsonl \
  --url http://127.0.0.1:8101 --model qwen35-4b --concurrency 16
```

```bash
CUDA_VISIBLE_DEVICES=GPU-27368065-5d8b-6e5e-4e81-2e924bd9ce73 \
  VLLM_USE_V2_MODEL_RUNNER=0 \
  .venv-vllm/bin/vllm serve google/gemma-4-E4B-it \
  --host 127.0.0.1 --port 8103 --served-model-name gemma4-e4b \
  --max-model-len 65536 --max-num-seqs 32 --gpu-memory-utilization 0.90

python3 research/asgard_summarizer_bench/run_openai_batch.py \
  --dataset research/asgard_summarizer_bench/v9_read_only_100.jsonl \
  --dataset research/asgard_summarizer_bench/v9_edit_producing_100.jsonl \
  --output research/asgard_summarizer_bench/gemma4-e4b-stratified-results.jsonl \
  --url http://127.0.0.1:8103 --model gemma4-e4b --concurrency 16
```

The runner calls vLLM's `/tokenize` endpoint for every row. Its default completion ceiling is 1× the exact tokenized summarizer input (candidate window plus task, instructions, and chat-template overhead), capped by the remaining 65,536-token context. This avoids a one-size-fits-all ceiling while still treating failure to compress before the source-sized budget as a real failure. Use `--max-output-input-ratio` to change the ratio or `--max-tokens` for a deliberately fixed-ceiling experiment.

The baseline prompt does not reliably distinguish earlier/cumulative edits from actions in a positively read-only window. Run the primary guarded probe with `--read-only-guard`:

```bash
python3 research/asgard_summarizer_bench/run_openai_batch.py \
  --dataset research/asgard_summarizer_bench/v9_read_only_100.jsonl \
  --output research/asgard_summarizer_bench/qwen35-4b-read-only-guard-results.jsonl \
  --url http://127.0.0.1:8101 --model qwen35-4b --concurrency 16 \
  --read-only-guard
```

Use the analogous Gemma URL/model/output on port 8103, then produce the paired reports:

```bash
python3 research/asgard_summarizer_bench/analyze_results.py \
  --dataset research/asgard_summarizer_bench/v9_read_only_100.jsonl \
  --result research/asgard_summarizer_bench/qwen35-4b-read-only-guard-results.jsonl \
  --result research/asgard_summarizer_bench/gemma4-e4b-read-only-guard-results.jsonl \
  --output research/asgard_summarizer_bench/read-only-guard-paired-metrics.json
```

The evidence-backed interpretation is in [RESULTS.md](RESULTS.md). `v9_sample_100.jsonl` and the non-stratified result files are retained as the initial exploratory run; the recommendation uses only the two positive-evidence cohorts and guarded primary results.

The WSL host lacks CUDA unified virtual addressing, so vLLM's internal V2 model runner fails for Gemma. `VLLM_USE_V2_MODEL_RUNNER=0` selects the internal V1 runner while retaining vLLM's current V1 engine and continuous batching. Qwen also required eager, language-model-only serving to avoid disproportionate multimodal/compilation startup on this host; consequently latency is diagnostic, not a controlled apples-to-apples performance comparison.
