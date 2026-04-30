# sharegpt_gsm8k_grid — exact run settings

All runs in this folder must use these settings to be comparable.
Reference run: Qwen2.5-7B-Instruct-AWQ, queued by donbool on 2026-04-26
(perf_quality_192337). Do NOT mix in runs that deviate.

## Models (Qwen2.5-Instruct family)
- Sizes: 7B, 14B, 32B, 72B
- Quantizations: Normal (BF16), GPTQ-Int8, AWQ, BNB-NF4-DQ
- Skipped intentionally: 72B Normal (OOM)

## vLLM launch (Servers tab → Add to Queue)
- Cache mode: **KV Cache (GB) = 10** (UI default — do NOT switch to GPU Mem %)
- dtype: auto
- Quantization: leave UI default (auto-populated based on model: awq / gptq / bitsandbytes / blank)
- Trust remote code: ✅
- Note: vLLM serve gets `--kv-cache-memory-bytes 10G`. The `gpu_memory_utilization=0.9`
  that appears in lm_eval gsm8k JSONs is the hidden slider's default value being
  plumbed through to lm_eval's in-process vLLM (lm_eval has no kv-cache option).
  It does NOT mean the GPU Mem % radio was selected.

## Performance run (sharegpt)
- Backend: vllm bench serve (openai endpoint)
- Dataset: sharegpt
- num_prompts: 500
- max_concurrency: 64
- request_rate: inf
- burstiness: 1.0
- Expected total_input_tokens: 103437 (deterministic — sanity check after the run)

## Quality run (gsm8k)
- lm_eval backend: --model vllm (in-process; queue path, NOT "Run Benchmark" button)
- Task: gsm8k
- num_fewshot: 8
- limit: 250
- batch_size: auto
- Resulting model_args (auto-built; sanity-check in JSON `raw.config.model_args`):
  pretrained=<model>, dtype=auto, gpu_memory_utilization=0.9, trust_remote_code=True
- Seeds (lm_eval defaults): random_seed=0, numpy_seed=1234, torch_seed=1234, fewshot_seed=1234

## How to add a new run to the grid
1. Servers tab → set the launch config above → Add to Queue.
2. Benchmark tab → uncheck Perf, check Quality (only gsm8k), set n-shot=8, limit=250 → Add to Queue.
   (For perf, check Perf with the settings above and uncheck Quality.)
3. Start the queue.
4. After it finishes, sanity-check the resulting JSON:
   - perf:    raw.total_input_tokens == 103437, raw.num_prompts == 500, raw.max_concurrency == 64
   - quality: raw["n-shot"]["gsm8k"] == 8, raw["n-samples"]["gsm8k"]["effective"] == 250
5. Copy the JSON(s) into this folder.
