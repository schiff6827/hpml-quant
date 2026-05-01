#!/usr/bin/env bash
# Run scripts/run_context_sweep.py against all 15 grid models with
# --gpu-mem-util 0.95 — i.e. let vLLM use ~91 GB of the 96 GB GPU for
# weights + KV pool. This answers the project's "max context under VRAM
# budget" question, not the artificial "max context with a fixed 10 GB
# KV pool" version.
#
# Output JSONs go to benchmarks/context_sweep_grid/. Each run kills its
# vLLM as soon as the answer is observed, so total wall time is roughly
# (model load time × 15) — maybe 5–15 min depending on cache state.
#
# Run from app/:
#   bash scripts/sweep_all_models_context.sh
#
# To resume after an interruption, just re-run — successful prior outputs
# will sit alongside new ones (each has a unique timestamp filename).

set -u
cd "$(dirname "$0")/.."   # cd to app/
PYTHON=/opt/hpml_project/hpml_env/bin/python
OUT_DIR=benchmarks/context_sweep_grid
mkdir -p "$OUT_DIR"

# Sanity: bail early if a vLLM is already holding the GPU.
if pgrep -f "vllm.entrypoints" >/dev/null; then
  echo "ERROR: a vLLM process is already running on this host. Stop it first."
  pgrep -af "vllm.entrypoints"
  exit 1
fi

# (model_id, short_label) pairs. Same 15 cells as sharegpt_gsm8k_grid,
# minus 72B-Normal which doesn't fit in 96 GB.
MODELS=(
  "Qwen/Qwen2.5-7B-Instruct                                                  7B-Normal"
  "Qwen/Qwen2.5-14B-Instruct                                                 14B-Normal"
  "Qwen/Qwen2.5-32B-Instruct                                                 32B-Normal"
  "Qwen/Qwen2.5-7B-Instruct-AWQ                                              7B-AWQ"
  "Qwen/Qwen2.5-14B-Instruct-AWQ                                             14B-AWQ"
  "Qwen/Qwen2.5-32B-Instruct-AWQ                                             32B-AWQ"
  "Qwen/Qwen2.5-72B-Instruct-AWQ                                             72B-AWQ"
  "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8                                        7B-GPTQ"
  "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8                                       14B-GPTQ"
  "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8                                       32B-GPTQ"
  "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8                                       72B-GPTQ"
  "/opt/hpml_project/models/local/Qwen2.5-7B-Instruct-BNB-NF4-DQ             7B-BnB"
  "/opt/hpml_project/models/local/Qwen2.5-14B-Instruct-BNB-NF4-DQ            14B-BnB"
  "/opt/hpml_project/models/local/Qwen2.5-32B-Instruct-BNB-NF4-DQ            32B-BnB"
  "/opt/hpml_project/models/local/Qwen2.5-72B-Instruct-BNB-NF4-DQ            72B-BnB"
)

results_summary=()   # collected for the end-of-run table
total=${#MODELS[@]}
i=0

for spec in "${MODELS[@]}"; do
  i=$((i + 1))
  read -r model label <<< "$spec"
  ts=$(date +%H%M%S)
  echo
  echo "================================================================"
  echo "[$i/$total] $label"
  echo "  model: $model"
  echo "================================================================"

  $PYTHON scripts/run_context_sweep.py \
    --model "$model" \
    --port 8001 \
    --upper-bound 131072 \
    --gpu-mem-util 0.95 \
    --result-dir "$OUT_DIR" \
    --run-name "${label}_ctx_${ts}" \
    --launch-timeout 900

  rc=$?

  # Find the JSON we just wrote (most recent context_sweep_*.json) and pull
  # max_context_tokens + source for the summary.
  latest=$(ls -t "$OUT_DIR"/context_sweep_*.json 2>/dev/null | head -1)
  if [[ -n "$latest" ]]; then
    summary=$($PYTHON -c "
import json, sys
d = json.load(open('$latest'))
mc = d.get('max_context_tokens')
src = d.get('source') or 'failed'
err = (d.get('launch_error') or '')[:60]
print(f'{mc if mc else \"FAIL\"}|{src}|{err}')
")
    results_summary+=("$label|$summary")
  else
    results_summary+=("$label|FAIL|no_output|rc=$rc")
  fi

  # Belt-and-suspenders: kill any leftover vLLM children before the next iter.
  # The script's own cleanup should handle this, but run-on cleanups protect
  # against vLLM workers that didn't exit cleanly.
  pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
  sleep 3
done

echo
echo "================================================================"
echo "SUMMARY — max VRAM-bounded context per model"
echo "(--gpu-mem-util 0.95, --max-model-len 131072 ceiling)"
echo "================================================================"
printf '%-12s %-10s %-22s %s\n' "Model" "Max ctx" "Source" "Error (if any)"
printf '%-12s %-10s %-22s %s\n' "-----" "-------" "------" "--------------"
for line in "${results_summary[@]}"; do
  IFS='|' read -r label maxctx source err <<< "$line"
  printf '%-12s %-10s %-22s %s\n' "$label" "$maxctx" "$source" "$err"
done

echo
echo "JSONs written to $OUT_DIR/"
