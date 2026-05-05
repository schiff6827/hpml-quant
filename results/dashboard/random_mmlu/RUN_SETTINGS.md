# random_mmlu_grid — exact run settings

Companion grid to `sharegpt_gsm8k_grid/`. Same 15-cell model × quantization
matrix, but swaps the workload (sharegpt → random) and the eval (gsm8k → MMLU).
All runs in this folder must use these settings to be comparable. Do NOT mix
in runs that deviate.

## Models (Qwen2.5-Instruct family) — same as sharegpt_gsm8k_grid
- Sizes: 7B, 14B, 32B, 72B
- Quantizations: Normal (BF16), GPTQ-Int8, AWQ, BNB-NF4-DQ
- Skipped intentionally: 72B Normal (OOM)
- Total: 15 cells

## vLLM launch (Servers tab → Add to Queue) — identical to sharegpt_gsm8k_grid
- Cache mode: **KV Cache (GB) = 10** (UI default — do NOT switch to GPU Mem %)
- dtype: auto
- Quantization: leave UI default (auto-populated based on model: awq / gptq / bitsandbytes / blank)
- Trust remote code: ✅
- Note: vLLM serve gets `--kv-cache-memory-bytes 10G`. The `gpu_memory_utilization=0.9`
  that appears in lm_eval mmlu JSONs is the hidden slider's default value being
  plumbed through to lm_eval's in-process vLLM (lm_eval has no kv-cache option).
  It does NOT mean the GPU Mem % radio was selected.

## Performance run (random)
- Backend: vllm bench serve (openai endpoint)
- Dataset: **random**
- num_prompts: 500
- max_concurrency: 64
- request_rate: inf
- burstiness: 1.0
- **Input length: 512** (UI default)
- **Output length: 256** (UI default)
- Expected total_input_tokens ≈ 256,000 (500 × 512). vLLM's random sampler
  jitters length slightly, so the exact value will vary per run by ~1–2%.
  Sanity-check this is in the right ballpark; don't expect the bit-exact
  determinism that sharegpt has.

## Quality run (mmlu)
- lm_eval backend: --model vllm (in-process; queue path, NOT "Run Benchmark" button)
- Task: **mmlu** (expands to 57 mmlu_* subtasks automatically)
- num_fewshot: **5** (canonical MMLU)
- limit: 5 (since it's 5 per subtask, 285 total)
- batch_size: auto
- Resulting model_args (auto-built; sanity-check in JSON `raw.config.model_args`):
  pretrained=<model>, dtype=auto, gpu_memory_utilization=0.9, trust_remote_code=True
- Seeds (lm_eval defaults): random_seed=0, numpy_seed=1234, torch_seed=1234, fewshot_seed=1234

### MMLU result location in the JSON
The headline metric is the aggregate `mmlu` row (averaged over 57 subtasks):
```
raw.results.mmlu["acc,none"]          # accuracy
raw.results.mmlu["acc_stderr,none"]   # stderr
```
Per-subtask scores live at `raw.results.mmlu_<subtask>` and are useful for
reasoning-vs-recall breakdowns.

### Why we do NOT use limit=250 here (unlike gsm8k)
gsm8k has 1,319 questions → limit=250 picks a fast 250-sample subset. MMLU's
57 subtasks have 100–500 questions each, and `--limit N` is *per subtask*, so
limit=250 would still run essentially the whole benchmark. Cleaner to set
limit=0 and report full MMLU. Wall-clock is short anyway because each MMLU
question generates exactly one logit, not a chain-of-thought.

## How to add a new run to the grid
1. Servers tab → set the launch config above → Add to Queue.
2. Benchmark tab:
   - Check Perf, Dataset=random, num_prompts=500, conc=64, input=512, output=256, request_rate=inf
   - Check Quality, only the `mmlu` checkbox, num_fewshot=5, limit=0
   - Add to Queue.
3. Start the queue.
4. After it finishes, sanity-check the resulting JSON:
   - perf:    raw.num_prompts == 500, raw.max_concurrency == 64,
              raw.total_input_tokens ≈ 256,000 (±2%)
   - quality: raw["n-shot"] has 57 mmlu_* keys all == 5,
              raw.results.mmlu["acc,none"] is populated
5. Copy the JSON(s) into this folder (`benchmarks/random_mmlu_grid/`).

## Note on profile CSVs
Profile CSVs land in `app/metrics/` keyed by the queue's job-name token
(e.g. `__perf_quality_HHMMSS`). Don't move them — `analyze.py` will glob them
by token like the sharegpt_gsm8k_grid version does.
