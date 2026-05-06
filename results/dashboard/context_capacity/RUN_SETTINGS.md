# context_sweep_grid

KV-cache capacity sweep for the same Qwen2.5-Instruct grid as `sharegpt_gsm8k_grid/`.

## What This Measures

The current chart data uses the new-format `context_sweep_*.json` files with a
`launch` field. These runs launch vLLM with `gpu_mem_util=0.95` and read vLLM's
reported GPU KV-cache token capacity from startup logs.

These values should be described as **KV-cache capacity**, not maximum
single-request context length. A served request is still capped by the
configured `max_model_len` unless vLLM is launched with a higher cap.

Older 2026-04-30 probe-style files are retained in the source benchmark folder
for provenance, but `analyze.py` skips them because they measured the
model/config context cap rather than KV-cache capacity.

## Sweep settings used

- `--upper-bound 131072`
- `--gpu-mem-util 0.95`
- All other launch settings identical to `sharegpt_gsm8k_grid/RUN_SETTINGS.md`
