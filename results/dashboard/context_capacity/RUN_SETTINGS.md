# context_sweep_grid

Context-length sweep for the same Qwen2.5-Instruct grid as `sharegpt_gsm8k_grid/`.

## ⚠️ Current data does NOT measure VRAM-bounded max context

Every cell in this folder hit vLLM's default `max_model_len = 32,768` (Qwen2.5's
native training context) and the binary search settled at **28,672 tokens for all
15 cells, regardless of model size or quantization**. None of the probes reached a
real VRAM limit — they all returned `HTTP 400: model's context length is only
32768 tokens`.

For the project's stated metric ("maximum achievable context length under VRAM
budget"), these need to be re-run with `--max-model-len 131072` (or whatever
upper bound) so the bottleneck becomes memory rather than the config preset.

The queue path needs a small extension to plumb extra vLLM args through. Until
that lands, treat this folder as documentation that **at 32K context, every
quantization × size we tested fits comfortably in 96 GB VRAM**.

## Sweep settings used (per UI Max Context preset)

- `--upper-bound 131072`
- `--step 4096`
- All other launch settings identical to `sharegpt_gsm8k_grid/RUN_SETTINGS.md`

## To re-run with VRAM as the binding constraint

1. Patch the queue to allow `extra_args` per launch (e.g. `--max-model-len 131072`).
2. For Qwen2.5 7B/14B which support YaRN, optionally add
   `--rope-scaling '{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'`
   so quality is preserved past 32K. For pure VRAM-ceiling measurement this is
   optional — vLLM will allocate KV for whatever max_model_len you set regardless.
3. Re-queue the Max Context preset for all 15 models.
