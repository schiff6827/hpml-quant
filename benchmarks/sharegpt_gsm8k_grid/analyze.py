"""Analyze the sharegpt_gsm8k grid: pareto chart + profile (GPU/VRAM/power) table.

Run from the app/ directory:
    /opt/hpml_project/hpml_env/bin/python benchmarks/sharegpt_gsm8k_grid/analyze.py

Note on gsm8k metric:
    parse_quality_result() in benchmark_service.py has a name-collision bug —
    both `exact_match,strict-match` and `exact_match,flexible-extract` get
    written to the same `exact_match` key, so the second (flexible) overwrites
    the first. flexible-extract grabs the last number from the model output,
    which is sensitive to post-answer formatting and dramatically under-scores
    runs whose models continue generating after the `#### N` marker.
    This script reads `exact_match,strict-match` directly from raw.results.gsm8k
    instead, which is the standard gsm8k metric and gives correct numbers.
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
# run_name embeds a `__<jobname>_HHMMSS` token (e.g. `__perf_quality_192337`),
# which is shared between the result JSON and the profile CSV in metrics/.
_token_re = re.compile(r'(perf_quality_\d{6}|perf_\d{6}|quality_\d{6})')

def _job_token(name):
    m = _token_re.search(name or '')
    return m.group(1) if m else None

# ---------- pull strict-match scores directly from raw lm_eval results ----------
def _strict_score(qual_json_path):
    d = json.load(open(qual_json_path))
    res = d.get('raw', {}).get('results', {}).get('gsm8k', {})
    return res.get('exact_match,strict-match'), res.get('exact_match,flexible-extract')


# ---------- assemble dataset ----------
rows = benchmark_service.build_pareto_dataset('gsm8k')

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

for r in rows:
    qf = qual_files_by_run.get(r['run_name'])
    if qf:
        s, fl = _strict_score(qf)
        r['gsm8k_strict'] = s
        r['gsm8k_flex'] = fl
    else:
        r['gsm8k_strict'] = r['gsm8k_flex'] = None
    pm = perf_meta_by_run.get(r['run_name'], {})
    csv_path = _find_csv_strict(pm.get('token'), pm.get('model_id') or r.get('model_id'))
    r['profile_csv'] = csv_path
    r['profile'] = benchmark_service.summarize_profile_csv(csv_path) if csv_path else {}

rows.sort(key=lambda r: ((r.get('quantization') or 'ZZZ'), r.get('parameters') or 0))

# ---------- table ----------
hdr = f'{"Run":<55}{"Quant":<8}{"P(B)":<7}{"Tput":<9}{"strict":<9}{"flex":<9}{"VRAM_GB":<10}{"avgUtil%":<10}{"avgPow_W":<10}{"peakPow_W"}'
print('\n' + hdr); print('-' * len(hdr))
for r in rows:
    p = r.get('profile') or {}
    print(f'{(r["run_name"] or "")[:54]:<55}'
          f'{(r["quantization"] or "")[:7]:<8}'
          f'{(r.get("parameters_b") or 0):<7.1f}'
          f'{(r.get("throughput_tps") or float("nan")):<9.0f}'
          f'{(r.get("gsm8k_strict") if r.get("gsm8k_strict") is not None else float("nan")):<9.3f}'
          f'{(r.get("gsm8k_flex")   if r.get("gsm8k_flex")   is not None else float("nan")):<9.3f}'
          f'{(p.get("peak_gpu_mem_mb",0)/1024):<10.2f}'
          f'{p.get("avg_gpu_util_pct",0):<10.1f}'
          f'{p.get("avg_gpu_power_w",0):<10.1f}'
          f'{p.get("peak_gpu_power_w",0):<8.1f}')

missing_csv = [r['run_name'] for r in rows if not r.get('profile_csv')]
if missing_csv:
    print(f'\nNo profile CSV matched for: {missing_csv}')


# ---------- pareto frontier (corrected to strict-match) ----------
def pareto(points):
    front = []
    for i, p in enumerate(points):
        if not any((q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]))
                   for j, q in enumerate(points) if j != i):
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


def _scatter(ax, plottable, xkey, ykey, xlabel, ylabel, title, ylim=None):
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
    front = pareto(pts)
    if len(front) >= 2:
        ax.plot([p[0] for p in front], [p[1] for p in front], '--', color='#888', linewidth=1.5, label='Pareto frontier', zorder=1)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3); ax.legend(loc='best', framealpha=0.95)
    return front


# Chart 1: gsm8k (strict) vs throughput
plot1 = [r for r in rows if r.get('throughput_tps') is not None and r.get('gsm8k_strict') is not None]
for r in plot1:
    r['_q'] = r['gsm8k_strict']
fig, ax = plt.subplots(figsize=(10, 7))
front1 = _scatter(ax, [{**r, 'tput': r['throughput_tps'], 'q': r['gsm8k_strict']} for r in plot1],
                  'tput', 'q',
                  'Throughput (output tok/s, sharegpt n=500, conc=64)',
                  'gsm8k exact_match (strict-match, n-shot=8, limit=250)',
                  'Pareto: gsm8k vs Throughput - Qwen2.5-Instruct grid', ylim=(0.5, 1))
out1 = os.path.join(GRID_DIR, 'pareto_gsm8k_vs_throughput.png')
fig.savefig(out1, dpi=140, bbox_inches='tight')
print(f'\nPareto frontier (gsm8k strict vs throughput):')
for p in front1: print(f'  tput={p[0]:7.0f}  gsm8k={p[1]:.3f}  {p[2]}')
print(f'Saved: {out1}')

# Chart 2: gsm8k (strict) vs peak VRAM (efficiency)
plot2 = [r for r in rows if (r.get('profile') or {}).get('peak_gpu_mem_mb') and r.get('gsm8k_strict') is not None]
for r in plot2:
    r['_vram'] = r['profile']['peak_gpu_mem_mb'] / 1024.0
fig2, ax2 = plt.subplots(figsize=(10, 7))
front2 = _scatter(ax2, [{**r, 'vram': r['_vram'], 'q': r['gsm8k_strict']} for r in plot2],
                  'vram', 'q',
                  'Peak GPU VRAM (GB)',
                  'gsm8k exact_match (strict-match)',
                  'Quality vs VRAM cost - lower-left dominated, upper-left ideal', ylim=(0.5, 1))
ax2.invert_xaxis()  # left = less VRAM = better; pareto front is upper-left
out2 = os.path.join(GRID_DIR, 'pareto_gsm8k_vs_vram.png')
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
         'Throughput vs Power draw')
out3 = os.path.join(GRID_DIR, 'tput_vs_power.png')
fig3.savefig(out3, dpi=140, bbox_inches='tight')
print(f'Saved: {out3}')
