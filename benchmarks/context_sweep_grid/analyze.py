"""Visualize max VRAM-bounded context per model.

Reads the new-format context_sweep JSONs (those with a `launch` field, written
by the rewritten scripts/run_context_sweep.py). Skips the older bottlenecked
runs from 4/30 that hit vLLM's max_position_embeddings ceiling at 28,672 — they
all measured the same artifact, not what we actually want.

Run from app/:
  /opt/hpml_project/hpml_env/bin/python benchmarks/context_sweep_grid/analyze.py
"""
import os, re, json, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GRID_DIR = os.path.dirname(os.path.abspath(__file__))

# Same legend order as the other analyze scripts (highest precision -> lowest).
QUANT_ORDER = {'Normal': 0, 'GPTQ': 1, 'AWQ': 2, 'BnB': 3}
COLORS = {'Normal': '#1f77b4', 'GPTQ': '#ff7f0e', 'AWQ': '#2ca02c', 'BnB': '#d62728'}
SIZES = ['7B', '14B', '32B', '72B']
QUANTS = ['Normal', 'GPTQ', 'AWQ', 'BnB']
QWEN25_TRAINED_MAX = 32768   # the model's native max_position_embeddings


def classify(model_id):
    m = re.search(r'(\d+)B', model_id or '')
    size = f'{m.group(1)}B' if m else '?'
    if 'BNB-NF4' in model_id: quant = 'BnB'
    elif 'GPTQ' in model_id:  quant = 'GPTQ'
    elif 'AWQ' in model_id:   quant = 'AWQ'
    else:                     quant = 'Normal'
    return size, quant


# Load only new-format runs (have a `launch` dict). Older bottlenecked files
# from the queue path don't have it.
runs = {}   # (size, quant) -> result dict
for f in sorted(glob.glob(os.path.join(GRID_DIR, 'context_sweep_*.json'))):
    d = json.load(open(f))
    if 'launch' not in d:
        continue
    model = (d.get('launch') or {}).get('model', '')
    size, quant = classify(model)
    if size == '?':
        continue
    # If multiple runs exist for the same cell, keep the most recent (file glob is sorted).
    runs[(size, quant)] = d

print(f'Loaded {len(runs)} cells of new-format context-sweep data\n')

print(f'{"Cell":<12}{"Max ctx":>14}{"Source":<22}{"Notes"}')
print('-' * 70)
for size in SIZES:
    for quant in QUANTS:
        d = runs.get((size, quant))
        if d is None:
            note = '(intentionally skipped: 72B Normal OOM)' if (size, quant) == ('72B', 'Normal') else 'NO DATA'
            print(f'{size:<3}-{quant:<8}{"-":>14}{"-":<22}{note}')
            continue
        mc = d.get('max_context_tokens')
        src = d.get('source') or 'failed'
        err = (d.get('launch_error') or '')[:40]
        if mc is None:
            print(f'{size:<3}-{quant:<8}{"-":>14}{src:<22}FAIL: {err}')
        else:
            print(f'{size:<3}-{quant:<8}{mc:>14,}{src:<22}')


# ---- chart: log-scale bar grouped by size, colored by quant ----
fig, ax = plt.subplots(figsize=(11, 6.5))

# Bar layout: 4 size groups on x-axis, 4 quant bars per group
bar_w = 0.18
x_centers = list(range(len(SIZES)))

for j, quant in enumerate(QUANTS):
    offset = (j - len(QUANTS) / 2 + 0.5) * bar_w
    xs = [x + offset for x in x_centers]
    ys = []
    labels = []
    for size in SIZES:
        d = runs.get((size, quant))
        if d and d.get('max_context_tokens'):
            ys.append(d['max_context_tokens'])
            labels.append(f'{d["max_context_tokens"]:,}')
        else:
            ys.append(0)
            if (size, quant) == ('72B', 'Normal'):
                labels.append('OOM')
            elif d is None or not d.get('max_context_tokens'):
                labels.append('—')
            else:
                labels.append('')
    bars = ax.bar(xs, ys, bar_w, label=quant, color=COLORS[quant],
                  edgecolor='black', linewidth=0.4)
    # annotate (above-bar for visible, near y-min for missing — placing the
    # missing annotation near a positive value rather than 0 keeps it inside
    # the log-scale visible range and prevents the figure from stretching).
    for x, y, lab in zip(xs, ys, labels):
        if y > 0:
            ax.text(x, y * 1.08, lab, ha='center', fontsize=8, rotation=90, va='bottom')
        else:
            ax.text(x, 1.3e4, lab, ha='center', fontsize=8, color='#888', va='bottom')

# Reference line: Qwen2.5's trained 32K context
ax.axhline(QWEN25_TRAINED_MAX, color='red', linestyle='--', linewidth=1.2,
           label=f"Qwen2.5 trained max ({QWEN25_TRAINED_MAX:,})", alpha=0.7)

ax.set_xticks(x_centers)
ax.set_xticklabels(SIZES)
ax.set_yscale('log')
ax.set_ylim(1e4, 5e6)
ax.set_ylabel('Max VRAM-bounded context (tokens, log scale)')
ax.set_xlabel('Model size')
ax.set_title('Max single-request context per model — gpu_mem_util=0.95, max_model_len ≤ 131,072')
ax.legend(loc='upper right', framealpha=0.95, fontsize=9)
ax.grid(True, alpha=0.3, axis='y', which='both')

plt.tight_layout()
out = os.path.join(GRID_DIR, 'max_context_per_model.png')
fig.savefig(out, dpi=140, bbox_inches='tight')
print(f'\nSaved: {out}')


# ---- secondary view: KV pool tokens vs weight memory tradeoff ----
# This makes the "where the VRAM goes" tradeoff explicit by comparing model
# weight size (from the same JSONs we saved earlier) to the resulting KV pool.
# Only useful for the cells whose source = kv_pool_tokens (not vllm_estimated).
weight_by_cell = {}   # (size, quant) -> approx weight GB
# Use known-good numbers from the earlier sharegpt_gsm8k_grid analysis. These
# came from vLLM's "Model loading took X GiB" log line.
weight_by_cell.update({
    ('7B', 'Normal'): 14.2,  ('14B', 'Normal'): 27.6, ('32B', 'Normal'): 61.0,
    ('7B', 'GPTQ'):   8.3,   ('14B', 'GPTQ'):   15.7, ('32B', 'GPTQ'):   32.8, ('72B', 'GPTQ'): 71.8,
    ('7B', 'AWQ'):    5.2,   ('14B', 'AWQ'):    9.4,  ('32B', 'AWQ'):    18.1, ('72B', 'AWQ'):  38.8,
    ('7B', 'BnB'):    5.5,   ('14B', 'BnB'):    9.9,  ('32B', 'BnB'):    19.4, ('72B', 'BnB'):  41.5,
})

fig2, ax2 = plt.subplots(figsize=(11, 6))
for quant in QUANTS:
    xs, ys, labels = [], [], []
    for size in SIZES:
        cell = (size, quant)
        d = runs.get(cell)
        w = weight_by_cell.get(cell)
        if not (d and d.get('max_context_tokens') and w):
            continue
        xs.append(w)
        ys.append(d['max_context_tokens'])
        labels.append(size)
    if xs:
        ax2.scatter(xs, ys, s=120, label=quant, color=COLORS[quant],
                    edgecolors='black', linewidths=0.5)
        for x, y, lab in zip(xs, ys, labels):
            ax2.annotate(lab, (x, y), fontsize=9, xytext=(6, 6), textcoords='offset points')

ax2.axhline(QWEN25_TRAINED_MAX, color='red', linestyle='--', linewidth=1.2,
            label=f"Qwen2.5 trained max ({QWEN25_TRAINED_MAX:,})", alpha=0.7)
ax2.set_xlabel('Model weight memory (GiB)')
ax2.set_ylabel('Max VRAM-bounded context (tokens)')
ax2.set_yscale('log')
ax2.set_ylim(1e4, 5e6)
ax2.set_title('Context capacity vs weight footprint — every GB of weights is GB less for KV')
ax2.legend(loc='upper right', framealpha=0.95, fontsize=9)
ax2.grid(True, alpha=0.3, which='both')
plt.tight_layout()
out2 = os.path.join(GRID_DIR, 'context_vs_weight.png')
fig2.savefig(out2, dpi=140, bbox_inches='tight')
print(f'Saved: {out2}')
