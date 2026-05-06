"""Analyze the random_mmlu grid: pareto chart + profile (GPU/VRAM/power) table.

Run from the app/ directory:
    /opt/hpml_project/hpml_env/bin/python benchmarks/random_mmlu_grid/analyze.py

Companion to sharegpt_gsm8k_grid/analyze.py — same plots, swapped workload (random)
and quality task (mmlu).

Note on the mmlu metric:
    Unlike gsm8k (which has a strict/flex name collision in parse_quality_result),
    MMLU emits a single `acc,none` aggregate. We read it from raw.results.mmlu
    directly so we can also surface stderr (which the saved tasks[] strips).
"""
import os, sys, glob, re, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
GRID_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(APP_DIR, 'metrics')
sys.path.insert(0, APP_DIR)

from services import benchmark_service
benchmark_service.BENCHMARKS_DIR = GRID_DIR  # scope build_pareto_dataset to this folder

# ---------- match runs to their profile CSVs ----------
# run_name embeds a `__<jobname>_HHMMSS` token (e.g. `__perf_quality_020120`),
# which is shared between the result JSON and the profile CSV in metrics/.
_token_re = re.compile(r'(perf_quality_\d{6}|perf_\d{6}|quality_\d{6})')

def _job_token(name):
    m = _token_re.search(name or '')
    return m.group(1) if m else None

# ---------- pull mmlu acc directly from raw lm_eval results ----------
def _mmlu_score(qual_json_path):
    d = json.load(open(qual_json_path))
    res = d.get('raw', {}).get('results', {}).get('mmlu', {})
    return res.get('acc,none'), res.get('acc_stderr,none')


# ---------- assemble dataset ----------
rows = benchmark_service.build_pareto_dataset('mmlu')

# Filename globs aren't reliable here because job names contain `perf_quality_HHMMSS`,
# so a perf JSON's filename matches `*quality*` too. Filter by data['type'] instead.
qual_files_by_run = {}
perf_meta_by_run = {}
for f in glob.glob(os.path.join(GRID_DIR, '*.json')):
    d = json.load(open(f))
    canonical = benchmark_service._canonical_run_name(d.get('run_name'))
    if not canonical:
        continue
    if d.get('type') == 'quality':
        qual_files_by_run[canonical] = f
    elif d.get('type') == 'perf':
        perf_meta_by_run[canonical] = {
            'model_id': (d.get('model_meta') or {}).get('model_id') or '',
            'token': _job_token(d.get('run_name')),
        }

def _find_csv_strict(token, model_id):
    """Match a CSV by token AND by the *basename* of the model_id.
    BnB JSONs use a local path (`/opt/.../models/local/<basename>`) while CSVs
    use just the basename, so compare on basename only."""
    if not token or not model_id: return None
    model_basename = re.sub(r'^.*[/\\]', '', model_id)        # strip dirs
    model_basename = model_basename.replace('/', '_')          # in case org left in
    # Anchor with '___' (the separator the queue uses between model name and job
    # name in CSV filenames). Without the anchor, e.g. 'Qwen2.5-7B-Instruct' would
    # also match 'Qwen2.5-7B-Instruct-AWQ___...csv'.
    needle = f'{model_basename}___'
    candidates = sorted(glob.glob(os.path.join(METRICS_DIR, f'*{token}*.csv')))
    for c in candidates:
        if needle in os.path.basename(c):
            return c
    return None

# Older grid runs (before the metrics_service KV-regex fix) have empty
# `kv_cache_capacity_gib` columns in their CSVs. Fall back to this constant in
# that case — every queued grid run used KV Cache (GB) = 10. After the fix,
# the column populates correctly and we use it.
KV_CAPACITY_FALLBACK_GIB = 10.0

def _extract_mem_and_kv(csv_path):
    """Memory-breakdown summary from a profile CSV. Uses kv_cache_pct + capacity
    (real column when populated, fallback constant otherwise) to estimate KV usage."""
    if not csv_path or not os.path.exists(csv_path):
        return {}
    import csv as _csv
    weight = peak_kv_pct = peak_total = peak_kv_capacity = 0.0
    with open(csv_path, newline='') as f:
        for row in _csv.DictReader(f):
            def _f(k):
                try: return float(row.get(k) or 0)
                except (TypeError, ValueError): return 0.0
            weight = max(weight, _f('weight_mem_gib'))
            peak_kv_pct = max(peak_kv_pct, _f('kv_cache_pct'))
            peak_kv_capacity = max(peak_kv_capacity, _f('kv_cache_capacity_gib'))
            peak_total = max(peak_total, _f('gpu_mem_used_mb') / 1024.0)
    kv_capacity_gib = peak_kv_capacity if peak_kv_capacity > 0 else KV_CAPACITY_FALLBACK_GIB
    peak_kv_used_gib = peak_kv_pct / 100.0 * kv_capacity_gib
    return {
        'weight_gib': weight,
        'kv_capacity_gib': kv_capacity_gib,
        'peak_kv_pct': peak_kv_pct,
        'peak_kv_used_gib': peak_kv_used_gib,
        'peak_kv_to_weight': (peak_kv_used_gib / weight) if weight else 0.0,
        'peak_total_vram_gib': peak_total,
    }


def _perf_raw(perf_file):
    """Load the perf JSON's raw dict (for TTFT percentiles)."""
    if not perf_file or not os.path.exists(perf_file):
        return {}
    return json.load(open(perf_file)).get('raw', {})


# Index of perf JSON paths by canonical run_name (we already have model_id+token by canonical above).
perf_files_by_run = {}
for f in glob.glob(os.path.join(GRID_DIR, '*.json')):
    d = json.load(open(f))
    if d.get('type') == 'perf':
        perf_files_by_run[benchmark_service._canonical_run_name(d.get('run_name'))] = f

for r in rows:
    qf = qual_files_by_run.get(r['run_name'])
    if qf:
        acc, stderr = _mmlu_score(qf)
        r['mmlu_acc'] = acc
        r['mmlu_stderr'] = stderr
    else:
        r['mmlu_acc'] = r['mmlu_stderr'] = None
    pm = perf_meta_by_run.get(r['run_name'], {})
    csv_path = _find_csv_strict(pm.get('token'), pm.get('model_id') or r.get('model_id'))
    r['profile_csv'] = csv_path
    r['profile'] = benchmark_service.summarize_profile_csv(csv_path) if csv_path else {}
    r['mem_kv'] = _extract_mem_and_kv(csv_path)
    r['perf_raw'] = _perf_raw(perf_files_by_run.get(r['run_name']))

rows.sort(key=lambda r: ((r.get('quantization') or 'ZZZ'), r.get('parameters') or 0))

# ---------- table ----------
hdr = f'{"Run":<55}{"Quant":<8}{"P(B)":<7}{"Tput":<9}{"mmlu":<9}{"±se":<9}{"VRAM_GB":<10}{"avgUtil%":<10}{"avgPow_W":<10}{"peakPow_W"}'
print('\n' + hdr); print('-' * len(hdr))
for r in rows:
    p = r.get('profile') or {}
    print(f'{(r["run_name"] or "")[:54]:<55}'
          f'{(r["quantization"] or "")[:7]:<8}'
          f'{(r.get("parameters_b") or 0):<7.1f}'
          f'{(r.get("throughput_tps") or float("nan")):<9.0f}'
          f'{(r.get("mmlu_acc")    if r.get("mmlu_acc")    is not None else float("nan")):<9.3f}'
          f'{(r.get("mmlu_stderr") if r.get("mmlu_stderr") is not None else float("nan")):<9.3f}'
          f'{(p.get("peak_gpu_mem_mb",0)/1024):<10.2f}'
          f'{p.get("avg_gpu_util_pct",0):<10.1f}'
          f'{p.get("avg_gpu_power_w",0):<10.1f}'
          f'{p.get("peak_gpu_power_w",0):<8.1f}')

missing_csv = [r['run_name'] for r in rows if not r.get('profile_csv')]
if missing_csv:
    print(f'\nNo profile CSV matched for: {missing_csv}')


# ---------- pareto frontier ----------
def pareto(points, x_higher_is_better=True, y_higher_is_better=True):
    front = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if j == i:
                continue
            x_ge = q[0] >= p[0] if x_higher_is_better else q[0] <= p[0]
            y_ge = q[1] >= p[1] if y_higher_is_better else q[1] <= p[1]
            x_gt = q[0] > p[0] if x_higher_is_better else q[0] < p[0]
            y_gt = q[1] > p[1] if y_higher_is_better else q[1] < p[1]
            if x_ge and y_ge and (x_gt or y_gt):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p[0])


COLORS = {'BFLOAT16': '#1f77b4', 'GPTQ': '#ff7f0e', 'AWQ': '#2ca02c', 'BITSANDBYTES': '#d62728'}
# Legend order: highest precision -> lowest (FP16 > INT8 > INT4-AWQ > INT4-BnB)
QUANT_ORDER = {'BFLOAT16': 0, 'GPTQ': 1, 'AWQ': 2, 'BITSANDBYTES': 3}

_SIZE_RE = re.compile(r'(\d+)B(?:-|$)')

def _nominal_size_label(r):
    """Use the nominal model size (7B / 14B / 32B / 72B) from model_id, NOT the
    reported parameter count — for quantized variants vLLM reports a slightly
    different value (e.g. 5.4B for 7B-AWQ) that mis-labels points on the chart."""
    mid = r.get('model_id') or r.get('run_name') or ''
    m = _SIZE_RE.search(mid)
    return f'{m.group(1)}B' if m else ''


def _scatter(ax, plottable, xkey, ykey, xlabel, ylabel, title, ylim=None,
             x_higher_is_better=True, y_higher_is_better=True):
    by_q = {}
    for r in plottable:
        by_q.setdefault(r['quantization'] or 'UNKNOWN', []).append(r)
    for q in sorted(by_q.keys(), key=lambda k: QUANT_ORDER.get(k, 99)):
        group = by_q[q]
        xs = [r[xkey] for r in group]; ys = [r[ykey] for r in group]
        ax.scatter(xs, ys, s=120, label=q, color=COLORS.get(q, '#888'), edgecolors='black', linewidths=0.5)
        for r in group:
            ax.annotate(_nominal_size_label(r), (r[xkey], r[ykey]), fontsize=9,
                        xytext=(5, 5), textcoords='offset points')
    pts = [[r[xkey], r[ykey], r['run_name']] for r in plottable]
    front = pareto(pts, x_higher_is_better, y_higher_is_better)
    if len(front) >= 2:
        ax.plot([p[0] for p in front], [p[1] for p in front], '--', color='#888', linewidth=1.5, label='Pareto frontier', zorder=1)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3); ax.legend(loc='best', framealpha=0.95)
    return front


# MMLU axis range — matches the gsm8k pareto charts (0.5–1.0) for visual parity
# across the two grids. Scores cluster ~0.75–0.90 for this Qwen family.
MMLU_YLIM = (0.5, 1.0)
WORKLOAD_LABEL = 'random in=512 out=256 n=500, conc=64'
EVAL_LABEL = 'mmlu (5-shot, full 57 subtasks)'


# Chart 1: mmlu acc vs throughput
plot1 = [r for r in rows if r.get('throughput_tps') is not None and r.get('mmlu_acc') is not None]
fig, ax = plt.subplots(figsize=(10, 7))
front1 = _scatter(ax, [{**r, 'tput': r['throughput_tps'], 'q': r['mmlu_acc']} for r in plot1],
                  'tput', 'q',
                  f'Throughput (output tok/s, {WORKLOAD_LABEL})',
                  f'MMLU acc ({EVAL_LABEL})',
                  'Pareto: MMLU vs Throughput - Qwen2.5-Instruct grid', ylim=MMLU_YLIM)
out1 = os.path.join(GRID_DIR, 'pareto_mmlu_vs_throughput.png')
fig.savefig(out1, dpi=140, bbox_inches='tight')
print(f'\nPareto frontier (mmlu vs throughput):')
for p in front1: print(f'  tput={p[0]:7.0f}  mmlu={p[1]:.3f}  {p[2]}')
print(f'Saved: {out1}')

# Chart 2: mmlu acc vs peak VRAM (efficiency)
plot2 = [r for r in rows if (r.get('profile') or {}).get('peak_gpu_mem_mb') and r.get('mmlu_acc') is not None]
for r in plot2:
    r['_vram'] = r['profile']['peak_gpu_mem_mb'] / 1024.0
fig2, ax2 = plt.subplots(figsize=(10, 7))
front2 = _scatter(ax2, [{**r, 'vram': r['_vram'], 'q': r['mmlu_acc']} for r in plot2],
                  'vram', 'q',
                  'Peak GPU VRAM (GB)',
                  f'MMLU acc ({EVAL_LABEL})',
                  'Quality vs VRAM cost - lower-left dominated, upper-left ideal', ylim=MMLU_YLIM,
                  x_higher_is_better=False)
ax2.invert_xaxis()  # left = less VRAM = better; pareto front is upper-left
out2 = os.path.join(GRID_DIR, 'pareto_mmlu_vs_vram.png')
fig2.savefig(out2, dpi=140, bbox_inches='tight')
print(f'Saved: {out2}')

# Chart 3: throughput vs avg GPU power (efficiency: tok per watt)
plot3 = [r for r in rows if (r.get('profile') or {}).get('avg_gpu_power_w') and r.get('throughput_tps')]
for r in plot3:
    r['_pow'] = r['profile']['avg_gpu_power_w']
fig3, ax3 = plt.subplots(figsize=(10, 7))
_scatter(ax3, [{**r, 'pow': r['_pow'], 'tput': r['throughput_tps']} for r in plot3],
         'pow', 'tput',
         'Avg GPU power (W)',
         'Throughput (output tok/s)',
         'Throughput vs Power draw',
         x_higher_is_better=False)
out3 = os.path.join(GRID_DIR, 'tput_vs_power.png')
fig3.savefig(out3, dpi=140, bbox_inches='tight')
print(f'Saved: {out3}')


# ------------------------------------------------------------------
# Memory / latency charts (use the profile CSV columns directly)
# ------------------------------------------------------------------

def _model_label(r):
    """Compact 'size-quant' label, e.g. '14B-AWQ' or '7B-BF16'."""
    short = {'BFLOAT16': 'BF16', 'GPTQ': 'GPTQ', 'AWQ': 'AWQ', 'BITSANDBYTES': 'BnB'}.get(
        r.get('quantization'), r.get('quantization') or '?')
    return f'{_nominal_size_label(r)}-{short}'


def _size_int(r):
    s = _nominal_size_label(r)
    return int(s[:-1]) if s.endswith('B') and s[:-1].isdigit() else 0


def _grid_sort_key(r):
    """Group by quant (legend order), then by model size — for consistent x-ordering."""
    return (QUANT_ORDER.get(r.get('quantization'), 99), _size_int(r))


# Chart 4: peak KV-to-weight ratio per model (the project's explicit metric)
plot4 = [r for r in rows if (r.get('mem_kv') or {}).get('peak_kv_to_weight')]
plot4.sort(key=_grid_sort_key)

fig4, ax4 = plt.subplots(figsize=(11, 6))
labels = [_model_label(r) for r in plot4]
vals = [r['mem_kv']['peak_kv_to_weight'] for r in plot4]
bar_colors = [COLORS.get(r.get('quantization'), '#888') for r in plot4]
bars = ax4.bar(labels, vals, color=bar_colors, edgecolor='black', linewidth=0.5)
for b, v in zip(bars, vals):
    ax4.text(b.get_x() + b.get_width()/2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
# Manual legend (one entry per quant, in QUANT_ORDER)
seen = []
for q in sorted(set(r.get('quantization') for r in plot4), key=lambda k: QUANT_ORDER.get(k, 99)):
    seen.append(plt.Rectangle((0, 0), 1, 1, color=COLORS.get(q, '#888'), label=q))
ax4.legend(handles=seen, loc='upper right')
ax4.axhline(1.0, color='#888', linestyle='--', linewidth=1, alpha=0.6)
ax4.text(len(labels) - 0.5, 1.02, 'KV = weights', fontsize=8, color='#666', ha='right')
ax4.set_ylabel('Peak KV cache size / model weight size (ratio)')
ax4.set_title(f'KV cache vs weight memory at peak — {WORKLOAD_LABEL}')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
out4 = os.path.join(GRID_DIR, 'kv_to_weight_ratio.png')
fig4.savefig(out4, dpi=140, bbox_inches='tight')
print(f'Saved: {out4}')


# Chart 5: VRAM breakdown stacked bar (weight / KV-used / KV-headroom / other)
plot5 = [r for r in rows if (r.get('mem_kv') or {}).get('weight_gib')]
plot5.sort(key=_grid_sort_key)

fig5, ax5 = plt.subplots(figsize=(11, 6.5))
labels = [_model_label(r) for r in plot5]
weight = [r['mem_kv']['weight_gib'] for r in plot5]
kv_used = [r['mem_kv']['peak_kv_used_gib'] for r in plot5]
kv_head = [max(0, r['mem_kv']['kv_capacity_gib'] - r['mem_kv']['peak_kv_used_gib']) for r in plot5]
# 'Other' = (peak total VRAM) - (weight + kv capacity).  Could be negative if the
# accounting doesn't perfectly add up — clamp at 0 for plotting so the bar still
# represents observed peak VRAM.
other = [max(0, r['mem_kv']['peak_total_vram_gib']
             - r['mem_kv']['weight_gib']
             - r['mem_kv']['kv_capacity_gib']) for r in plot5]

x = list(range(len(labels)))
ax5.bar(x, weight, label='Model weights', color='#4c78a8', edgecolor='black', linewidth=0.5)
ax5.bar(x, kv_used, bottom=weight, label='KV cache (used at peak)', color='#f58518', edgecolor='black', linewidth=0.5)
ax5.bar(x, kv_head, bottom=[w+u for w,u in zip(weight, kv_used)],
        label='KV cache (headroom)', color='#ffbf79', edgecolor='black', linewidth=0.5)
ax5.bar(x, other, bottom=[w+u+h for w,u,h in zip(weight, kv_used, kv_head)],
        label='Activations / framework', color='#b0b0b0', edgecolor='black', linewidth=0.5)
# annotate total peak VRAM on top
for xi, r in zip(x, plot5):
    ax5.text(xi, r['mem_kv']['peak_total_vram_gib'] + 1.5,
             f'{r["mem_kv"]["peak_total_vram_gib"]:.0f}', ha='center', fontsize=8, color='#333')
ax5.axhline(96, color='red', linestyle='--', linewidth=1, alpha=0.7, label='96 GB GPU budget')
ax5.set_xticks(x); ax5.set_xticklabels(labels, rotation=45, ha='right')
ax5.set_ylabel('Peak VRAM (GiB)')
ax5.set_title('VRAM breakdown at peak — where the 96 GB budget goes')
ax5.set_ylim(0, 105)
ax5.legend(loc='upper left', framealpha=0.95, fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
out5 = os.path.join(GRID_DIR, 'vram_breakdown.png')
fig5.savefig(out5, dpi=140, bbox_inches='tight')
print(f'Saved: {out5}')


# Chart 6: TTFT p50 / p90 / p99 grouped bars per model
plot6 = [r for r in rows if (r.get('perf_raw') or {}).get('p50_ttft_ms') is not None]
plot6.sort(key=_grid_sort_key)

fig6, ax6 = plt.subplots(figsize=(12, 6.5))
labels = [_model_label(r) for r in plot6]
p50 = [r['perf_raw']['p50_ttft_ms'] for r in plot6]
p90 = [r['perf_raw']['p90_ttft_ms'] for r in plot6]
p99 = [r['perf_raw']['p99_ttft_ms'] for r in plot6]

x = list(range(len(labels)))
w = 0.27
ax6.bar([xi - w for xi in x], p50, w, label='p50', color='#4c78a8', edgecolor='black', linewidth=0.5)
ax6.bar(x,                   p90, w, label='p90', color='#f58518', edgecolor='black', linewidth=0.5)
ax6.bar([xi + w for xi in x], p99, w, label='p99', color='#e45756', edgecolor='black', linewidth=0.5)
ax6.set_xticks(x); ax6.set_xticklabels(labels, rotation=45, ha='right')
ax6.set_ylabel('Time to first token (ms)')
ax6.set_title(f'TTFT latency distribution per model — {WORKLOAD_LABEL}')
ax6.set_yscale('log')   # TTFT spans 100ms - 10s+, log makes the cluster legible
ax6.grid(True, alpha=0.3, axis='y', which='both')
ax6.legend(loc='upper left', framealpha=0.95)
plt.tight_layout()
out6 = os.path.join(GRID_DIR, 'ttft_percentiles.png')
fig6.savefig(out6, dpi=140, bbox_inches='tight')
print(f'Saved: {out6}')


# ------------------------------------------------------------------
# Headline single-number bars (mmlu accuracy, throughput, energy/token)
# ------------------------------------------------------------------

def _bar_chart(rows_in, value_fn, ylabel, title, fmt, fname, ylim=None,
               legend_loc='upper right'):
    """Shared bar-chart skeleton: one bar per run, colored by quant, sorted by
    (quant, size). Skip rows where value_fn returns None / 0."""
    plot = []
    for r in rows_in:
        v = value_fn(r)
        if v is None:
            continue
        plot.append((r, v))
    plot.sort(key=lambda rv: _grid_sort_key(rv[0]))
    if not plot:
        print(f'[skip] {fname}: no rows')
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [_model_label(r) for r, _ in plot]
    vals = [v for _, v in plot]
    bar_colors = [COLORS.get(r.get('quantization'), '#888') for r, _ in plot]
    bars = ax.bar(labels, vals, color=bar_colors, edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, fmt.format(v),
                ha='center', va='bottom', fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS.get(q, '#888'), label=q)
               for q in sorted({r.get('quantization') for r, _ in plot},
                               key=lambda k: QUANT_ORDER.get(k, 99))]
    ax.legend(handles=handles, loc=legend_loc)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out = os.path.join(GRID_DIR, fname)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'Saved: {out}')


# Chart 7: mmlu accuracy
_bar_chart(rows, lambda r: r.get('mmlu_acc'),
           ylabel='MMLU acc (5-shot, full)',
           title='MMLU accuracy per model — 5-shot, all 57 subtasks',
           fmt='{:.3f}', fname='mmlu_accuracy.png', ylim=(0, 1.0),
           legend_loc='lower right')


# Chart 8: decode throughput
_bar_chart(rows, lambda r: r.get('throughput_tps'),
           ylabel='Output throughput (decode tok/s)',
           title=f'Decode throughput per model — {WORKLOAD_LABEL}',
           fmt='{:.0f}', fname='throughput_bar.png',
           legend_loc='upper right')


# Chart 9: energy per output token (J/tok) — proxy for deployment cost-per-token
def _j_per_tok(r):
    """avg GPU power × wall duration / total output tokens.
    Includes any idle gap before first request, but those are small relative to run length."""
    p = (r.get('profile') or {}).get('avg_gpu_power_w') or 0
    raw = r.get('perf_raw') or {}
    dur = raw.get('duration')
    out_tok = raw.get('total_output_tokens')
    if not (p and dur and out_tok):
        return None
    return p * dur / out_tok


_bar_chart(rows, _j_per_tok,
           ylabel='Energy per output token (J / tok)',
           title='Energy efficiency: avg GPU power × wall time / output tokens',
           fmt='{:.2f}', fname='energy_per_token.png',
           legend_loc='upper left')


# ------------------------------------------------------------------
# Time-series charts: KV growth, throughput-over-time (faceted by run)
# ------------------------------------------------------------------

import csv as _csv
from datetime import datetime as _dt

SIZES_FACET = ['7B', '14B', '32B', '72B']
QUANTS_FACET = ['BFLOAT16', 'GPTQ', 'AWQ', 'BITSANDBYTES']


def _read_timeseries(csv_path):
    """Return (elapsed_s, kv_pct, gen_tokens_cum) for the perf-benchmark phase only.

    A queue run captures both perf + quality in one CSV against the same vLLM
    server, but vLLM's gen_tokens_total counter resets to 0 between phases. We
    truncate at the first large drop in gen_tokens_total so charts show only
    the perf phase (which is the workload these visualizations are about)."""
    if not csv_path or not os.path.exists(csv_path):
        return [], [], []
    rows_ = []
    with open(csv_path, newline='') as f:
        for row in _csv.DictReader(f):
            try:
                ts = _dt.fromisoformat(row['timestamp'])
                kv = float(row.get('kv_cache_pct') or 0)
                gen = float(row.get('gen_tokens_total') or 0)
            except (TypeError, ValueError, KeyError):
                continue
            rows_.append((ts, kv, gen))
    start = next((i for i, r in enumerate(rows_) if r[1] > 0), None)
    if start is None:
        return [], [], []
    rows_ = rows_[start:]
    # Truncate at first phase boundary (gen_tokens_total resets to ~0 between perf + quality)
    end = len(rows_)
    for i in range(1, len(rows_)):
        if rows_[i][2] < rows_[i-1][2] - 100:   # >100-token backwards jump = counter reset
            end = i
            break
    rows_ = rows_[:end]
    t0 = rows_[0][0]
    return ([(r[0] - t0).total_seconds() for r in rows_],
            [r[1] for r in rows_],
            [r[2] for r in rows_])


def _facet_idx(r):
    size = _nominal_size_label(r)
    if size not in SIZES_FACET or r.get('quantization') not in QUANTS_FACET:
        return None, None
    return QUANTS_FACET.index(r.get('quantization')), SIZES_FACET.index(size)


# Pre-load every run's time series (used by both faceted charts below)
_ts_cache = {r['run_name']: _read_timeseries(r.get('profile_csv')) for r in rows}


# Chart 10: KV cache occupancy over time, faceted (rows = quants, cols = sizes)
fig10, axes10 = plt.subplots(len(QUANTS_FACET), len(SIZES_FACET),
                             figsize=(14, 10), sharey=True)
for r in rows:
    qi, si = _facet_idx(r)
    if qi is None:
        continue
    elapsed, kv, _ = _ts_cache.get(r['run_name'], ([], [], []))
    if not elapsed:
        continue
    ax = axes10[qi][si]
    color = COLORS.get(r.get('quantization'), '#888')
    ax.plot(elapsed, kv, color=color, linewidth=1.2)
    ax.fill_between(elapsed, 0, kv, color=color, alpha=0.2)
    ax.set_title(_model_label(r), fontsize=9)
    ax.grid(True, alpha=0.3)
for qi in range(len(QUANTS_FACET)):
    for si in range(len(SIZES_FACET)):
        if not axes10[qi][si].lines:
            axes10[qi][si].set_visible(False)
fig10.suptitle(f'KV cache occupancy over time per run — {WORKLOAD_LABEL}', y=0.995)
fig10.supxlabel('Elapsed time within active window (s)')
fig10.supylabel('KV cache used (% of capacity)')
plt.figure(fig10.number); plt.tight_layout()
out10 = os.path.join(GRID_DIR, 'kv_growth_over_time.png')
fig10.savefig(out10, dpi=140, bbox_inches='tight')
print(f'Saved: {out10}')


# Chart 11: instantaneous decode throughput over time (Δgen_tokens / Δt), faceted
def _smoothed_tokrate(elapsed, gen, window=5):
    """First-difference rate with a left-aligned `window`-sample moving average."""
    raw_t, raw_v = [], []
    for i in range(1, len(elapsed)):
        dt = elapsed[i] - elapsed[i-1]
        if dt > 0:
            raw_t.append(elapsed[i])
            raw_v.append((gen[i] - gen[i-1]) / dt)
    smooth = [sum(raw_v[max(0, i-window+1):i+1]) / (i + 1 - max(0, i-window+1))
              for i in range(len(raw_v))]
    return raw_t, smooth


fig11, axes11 = plt.subplots(len(QUANTS_FACET), len(SIZES_FACET),
                             figsize=(14, 10), sharey=False)
for r in rows:
    qi, si = _facet_idx(r)
    if qi is None:
        continue
    elapsed, _, gen = _ts_cache.get(r['run_name'], ([], [], []))
    t, rate = _smoothed_tokrate(elapsed, gen)
    if not t:
        continue
    ax = axes11[qi][si]
    color = COLORS.get(r.get('quantization'), '#888')
    ax.plot(t, rate, color=color, linewidth=1.2)
    ax.set_title(_model_label(r), fontsize=9)
    ax.grid(True, alpha=0.3)
for qi in range(len(QUANTS_FACET)):
    for si in range(len(SIZES_FACET)):
        if not axes11[qi][si].lines:
            axes11[qi][si].set_visible(False)
fig11.suptitle('Instantaneous output throughput over time (Δgen_tokens / Δt, 5-sample MA)', y=0.995)
fig11.supxlabel('Elapsed time within active window (s)')
fig11.supylabel('Decode tok/s')
plt.figure(fig11.number); plt.tight_layout()
out11 = os.path.join(GRID_DIR, 'throughput_over_time.png')
fig11.savefig(out11, dpi=140, bbox_inches='tight')
print(f'Saved: {out11}')


# Chart 12: decode vs prefill throughput scatter
plot12 = [r for r in rows if r.get('throughput_tps') and r.get('prefill_tps')]
fig12, ax12 = plt.subplots(figsize=(10, 7))
by_q = {}
for r in plot12:
    by_q.setdefault(r.get('quantization') or 'UNKNOWN', []).append(r)
for q in sorted(by_q.keys(), key=lambda k: QUANT_ORDER.get(k, 99)):
    group = by_q[q]
    xs = [r['prefill_tps'] for r in group]
    ys = [r['throughput_tps'] for r in group]
    ax12.scatter(xs, ys, s=120, label=q, color=COLORS.get(q, '#888'),
                 edgecolors='black', linewidths=0.5)
    for r in group:
        ax12.annotate(_nominal_size_label(r), (r['prefill_tps'], r['throughput_tps']),
                      fontsize=9, xytext=(5, 5), textcoords='offset points')
if plot12:
    all_max = max([r['throughput_tps'] for r in plot12] + [r['prefill_tps'] for r in plot12])
    ax12.plot([0, all_max], [0, all_max], '--', color='#888', alpha=0.4, label='y = x')
ax12.set_xlabel('Prefill throughput (input tok/s; derived = total_input / (mean_TTFT × completed))')
ax12.set_ylabel('Decode throughput (output tok/s)')
ax12.set_title(f'Decode vs Prefill throughput per run — {WORKLOAD_LABEL}')
ax12.legend(loc='upper left', framealpha=0.95)
ax12.grid(True, alpha=0.3)
plt.tight_layout()
out12 = os.path.join(GRID_DIR, 'decode_vs_prefill.png')
fig12.savefig(out12, dpi=140, bbox_inches='tight')
print(f'Saved: {out12}')
