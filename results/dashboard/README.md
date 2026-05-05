# HPML Quantization Dashboard Export

This directory is a static export of the experiment results for the Qwen2.5 scale vs. precision study. It is intended for submission in environments where a public dashboard link is not available.

## Experiment Scope

The main experiment evaluates Qwen2.5-Instruct across:

- Model sizes: 7B, 14B, 32B, 72B
- Weight formats: unquantized 16-bit baseline, GPTQ-Int8, AWQ-Int4, BitsAndBytes NF4

The unquantized 72B configuration is omitted because it does not fit within the 96 GB VRAM budget, leaving 15 runnable cells per primary sweep.

## Primary Result Sets

### `sharegpt_gsm8k/`

Performance uses ShareGPT prompts with 500 requests, infinite request rate, and max concurrency 64. Quality uses GSM8K with 8-shot prompting and a 250-example limit. The headline chart is `pareto_gsm8k_vs_throughput.png`.

### `random_mmlu/`

Performance uses a fixed random workload with 500 requests, 512 input tokens, 256 output tokens, infinite request rate, and max concurrency 64. Quality uses MMLU with 5-shot prompting across all 57 MMLU subtasks and a limit of 5 examples per subtask. The headline chart is `pareto_mmlu_vs_throughput.png`.

### `context_capacity/`

Context capacity is measured separately from the fixed-KV main sweeps. These runs launch vLLM with GPU memory utilization set to 0.95 and read the reported GPU KV-cache token capacity from startup logs.

### `nsys/`

Nsight Systems results are a selected kernel-level profiling slice used to explain hardware behavior behind the benchmark results. These charts should be interpreted as representative profiling, not as a full Nsight capture for every grid cell.

## Source Artifacts

The underlying JSON result files, CSV metric traces, and analysis scripts remain in:

- `benchmarks/sharegpt_gsm8k_grid/`
- `benchmarks/random_mmlu_grid/`
- `benchmarks/context_sweep_grid/`
- `benchmarks/nsys/`
- `metrics/`

Open `index.html` in a browser to view the exported figures as a static dashboard.

Open `runs.html` to browse the individual experiment run records. Each row corresponds to one runnable model/precision cell in one primary sweep, with expandable metric summaries and links to copied perf/quality JSON artifacts.
