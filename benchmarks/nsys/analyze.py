"""Cross-model nsys profile analysis.

For each *.nsys-rep in this folder, runs `nsys stats --report cuda_gpu_sum`,
categorizes each kernel/memop into a small set of buckets, and produces:
  - <model>_top_kernels.txt — top-10 kernels by time, raw nsys output
  - kernel_breakdown.png    — stacked bar across models, time by category
  - compute_vs_memory.png   — bar chart of compute time vs memory-op time
  - kernel_breakdown.csv    — same as the chart but as a flat table

Run from the app/ directory:
    /opt/hpml_project/hpml_env/bin/python benchmarks/nsys/analyze.py

Limitations:
  - This is Nsight Systems data only. Real roofline/arithmetic-intensity needs
    Nsight Compute (`ncu`) to capture per-kernel FLOP counts and DRAM bytes.
  - Categorization is heuristic, regex-based on kernel names.
  - Each model was profiled at sharegpt n=100, conc=64 (Nsight overhead made
    n=500 prohibitively slow). Numbers are not directly comparable to the
    sharegpt_gsm8k_grid perf JSONs which used n=500.
"""
import os, re, csv, sys, glob, subprocess, io, json
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NSYS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.abspath(os.path.join(NSYS_DIR, '..', '..'))
PERF_JSON_DIR = os.path.join(APP_DIR, 'benchmarks')

# RTX PRO 6000 Blackwell — dense FP16/BF16 throughput.
# Verify against the official NVIDIA datasheet for your specific SKU; this is
# the "no structured sparsity" number. AWQ runs FP16 GEMMs after dequant, so
# FP16 peak is the right denominator. For INT8 (GPTQ-Int8) you'd want ~580 TOPS.
PEAK_FP16_TFLOPS = 290.0

# Categorize a kernel/memop into a bucket. Order matters — first match wins.
# Operation strings come from `nsys stats --report cuda_gpu_sum`.
CATEGORIES = [
    ('AWQ dequant',     re.compile(r'awq', re.I)),
    ('GPTQ dequant',    re.compile(r'gptq|marlin', re.I)),
    ('Flash attention', re.compile(r'flash[_ ]?(fwd|attn)|paged_attn', re.I)),
    ('KV cache ops',    re.compile(r'reshape_and_cache|kv[_ ]cache|gather_kv', re.I)),
    ('GEMM (cutlass)',  re.compile(r'cutlass.*gemm|cublas.*gemm', re.I)),
    ('Sampling (topk/topp)', re.compile(r'topk|topp|sample', re.I)),
    ('Softmax / norm',  re.compile(r'softmax|rmsnorm|layernorm|rsqrt|_norm_', re.I)),
    ('Triton/vLLM custom', re.compile(r'^triton_|vllm::|silu', re.I)),
    ('Memory ops',      None),  # category column == MEMORY_OPER
    ('Other compute',   None),  # fallback for CUDA_KERNEL not matching above
]
COLORS = {
    'AWQ dequant':       '#2ca02c',
    'GPTQ dequant':      '#ff7f0e',
    'GEMM (cutlass)':    '#1f77b4',
    'Flash attention':   '#9467bd',
    'KV cache ops':      '#bcbd22',
    'Softmax / norm':    '#17becf',
    'Sampling (topk/topp)': '#7f7f7f',
    'Triton/vLLM custom': '#8c564b',
    'Memory ops':        '#e377c2',
    'Other compute':     '#aec7e8',
}


def categorize(category, op):
    if category == 'MEMORY_OPER':
        return 'Memory ops'
    for label, rx in CATEGORIES:
        if rx is not None and rx.search(op):
            return label
    return 'Other compute'


def model_label(nsys_path):
    """Extract a compact 'NN B-Quant' label from the nsys filename, e.g. '7B-AWQ'."""
    base = os.path.basename(nsys_path)
    m = re.search(r'Qwen2?\.?5-(\d+)B-Instruct(?:-([A-Za-z0-9-]+?))?(?:_\d+)?_\d{8}', base)
    if not m:
        return base[:40]
    size = m.group(1) + 'B'
    quant_raw = (m.group(2) or '').strip('_-')
    quant = {'AWQ': 'AWQ', 'GPTQ-Int8': 'GPTQ', 'BNB-NF4-DQ': 'BnB'}.get(quant_raw, quant_raw or 'BF16')
    return f'{size}-{quant}'


def run_nsys_stats(nsys_path):
    """Returns parsed rows of (time_pct, total_ns, instances, category, op).
    Caches the SQLite alongside the .nsys-rep so subsequent calls are fast."""
    proc = subprocess.run(
        ['nsys', 'stats', '--report', 'cuda_gpu_sum', '--format', 'csv', nsys_path],
        capture_output=True, text=True, check=False
    )
    out = proc.stdout
    # nsys stats output has stderr-y header lines mixed in; find the real CSV by
    # locating the line that starts with "Time (%)".
    lines = out.split('\n')
    start = next((i for i, ln in enumerate(lines) if ln.startswith('Time (%)')), None)
    if start is None:
        print(f'[warn] no CSV header in nsys output for {nsys_path}', file=sys.stderr)
        return []
    csv_text = '\n'.join(lines[start:])
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = []
    for r in reader:
        try:
            rows.append((
                float(r['Time (%)']),
                int(r['Total Time (ns)']),
                int(r['Instances']),
                r['Category'],
                r['Operation'],
            ))
        except (KeyError, ValueError):
            continue
    return rows


def aggregate_by_category(rows):
    by_cat = defaultdict(lambda: {'total_ns': 0, 'instances': 0, 'top_op': ('', 0)})
    for time_pct, total_ns, instances, category, op in rows:
        cat = categorize(category, op)
        agg = by_cat[cat]
        agg['total_ns'] += total_ns
        agg['instances'] += instances
        if total_ns > agg['top_op'][1]:
            agg['top_op'] = (op[:80], total_ns)
    return by_cat


def run_nsys_stats_full(nsys_path):
    """Like run_nsys_stats but returns the full per-kernel dict including latency
    percentiles (Avg/Med/Min/Max/StdDev) for latency-distribution analysis."""
    proc = subprocess.run(
        ['nsys', 'stats', '--report', 'cuda_gpu_sum', '--format', 'csv', nsys_path],
        capture_output=True, text=True, check=False
    )
    lines = proc.stdout.split('\n')
    start = next((i for i, ln in enumerate(lines) if ln.startswith('Time (%)')), None)
    if start is None:
        return []
    reader = csv.DictReader(io.StringIO('\n'.join(lines[start:])))
    rows = []
    for r in reader:
        try:
            rows.append({
                'pct': float(r['Time (%)']),
                'total_ns': int(r['Total Time (ns)']),
                'instances': int(r['Instances']),
                'avg_ns': float(r['Avg (ns)']),
                'med_ns': float(r['Med (ns)']),
                'min_ns': float(r['Min (ns)']),
                'max_ns': float(r['Max (ns)']),
                'std_ns': float(r['StdDev (ns)']),
                'category': r['Category'],
                'op': r['Operation'],
            })
        except (KeyError, ValueError):
            continue
    return rows


def main():
    nsys_files = sorted(glob.glob(os.path.join(NSYS_DIR, '*.nsys-rep')))
    if not nsys_files:
        print('No .nsys-rep files in', NSYS_DIR)
        return
    print(f'Analyzing {len(nsys_files)} reports...')

    per_model = {}        # label -> by_cat dict (time aggregates)
    per_model_full = {}   # label -> full per-kernel rows
    for f in nsys_files:
        label = model_label(f)
        print(f'  {label}: parsing {os.path.basename(f)}')
        rows_full = run_nsys_stats_full(f)
        if not rows_full:
            continue
        per_model_full[label] = rows_full
        # legacy tuple form for the existing aggregator
        rows = [(r['pct'], r['total_ns'], r['instances'], r['category'], r['op'])
                for r in rows_full]
        per_model[label] = aggregate_by_category(rows)

        # Top-15 kernels for this model
        top_path = os.path.join(NSYS_DIR, f'{label}_top_kernels.txt')
        with open(top_path, 'w') as out:
            out.write(f'Top kernels — {label} (sharegpt n=100, conc=64)\n')
            out.write(f'Source: {os.path.basename(f)}\n\n')
            out.write(f'{"Time%":>7}  {"Total (ms)":>11}  {"Calls":>8}  {"Avg (us)":>10}  {"p50 (us)":>10}  {"p99~ (us)":>10}  {"CV":>6}  {"Category":<14}  Operation\n')
            out.write('-' * 150 + '\n')
            for r in rows_full[:15]:
                bucket = categorize(r['category'], r['op'])
                avg_us = r['avg_ns'] / 1000
                med_us = r['med_ns'] / 1000
                # No true p99 in the data; max is a worst-case upper bound. Use it as a tail proxy.
                p99_us = r['max_ns'] / 1000
                cv = (r['std_ns'] / r['avg_ns']) if r['avg_ns'] > 0 else 0
                out.write(f'{r["pct"]:>7.1f}  {r["total_ns"]/1e6:>11.1f}  {r["instances"]:>8d}  {avg_us:>10.1f}  {med_us:>10.1f}  {p99_us:>10.1f}  {cv:>6.2f}  {bucket:<14}  {r["op"][:80]}\n')

    # ---- chart 1: stacked bar of time-by-category per model ----
    sort_order = ['7B-AWQ', '14B-AWQ', '32B-AWQ', '72B-AWQ', '7B-GPTQ', '14B-GPTQ', '32B-GPTQ', '72B-GPTQ']
    labels = [l for l in sort_order if l in per_model] + [l for l in per_model if l not in sort_order]
    all_cats = list(dict.fromkeys(c for m in per_model.values() for c in m.keys()))
    # consistent category ordering for the legend (compute first, mem ops last)
    cat_order = [c for c, _ in CATEGORIES if c in all_cats] + [c for c in all_cats if c not in [x for x, _ in CATEGORIES]]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bottoms = [0.0] * len(labels)
    for cat in cat_order:
        vals = [per_model[lab].get(cat, {}).get('total_ns', 0) / 1e9 for lab in labels]
        if not any(vals):
            continue
        ax.bar(labels, vals, bottom=bottoms, label=cat,
               color=COLORS.get(cat, '#888'), edgecolor='black', linewidth=0.4)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    # annotate total seconds on top
    for i, total in enumerate(bottoms):
        ax.text(i, total + 0.3, f'{total:.1f}s', ha='center', fontsize=9)
    ax.set_ylabel('GPU time (seconds)')
    ax.set_title('GPU time by kernel category — sharegpt n=100, conc=64')
    ax.legend(loc='upper left', framealpha=0.95, fontsize=9, bbox_to_anchor=(1.0, 1.0))
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out_path = os.path.join(NSYS_DIR, 'kernel_breakdown.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'Saved: {out_path}')

    # ---- chart 2: compute time vs memory-op time per model ----
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    compute_secs = []
    mem_secs = []
    for lab in labels:
        m = per_model[lab]
        memv = m.get('Memory ops', {}).get('total_ns', 0) / 1e9
        compv = sum(v.get('total_ns', 0) for cat, v in m.items() if cat != 'Memory ops') / 1e9
        mem_secs.append(memv); compute_secs.append(compv)
    x = list(range(len(labels)))
    w = 0.4
    ax2.bar([xi - w/2 for xi in x], compute_secs, w, label='Compute (kernels)',
            color='#1f77b4', edgecolor='black', linewidth=0.4)
    ax2.bar([xi + w/2 for xi in x], mem_secs, w, label='Memory ops (memcpy/memset)',
            color='#e377c2', edgecolor='black', linewidth=0.4)
    for i, (c, m) in enumerate(zip(compute_secs, mem_secs)):
        ratio = c / m if m else float('inf')
        ax2.text(i, max(c, m) + 0.3,
                 f'{ratio:.1f}×' if ratio != float('inf') else '∞',
                 ha='center', fontsize=9, color='#444')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=30, ha='right')
    ax2.set_ylabel('GPU time (seconds)')
    ax2.set_title('Compute vs Memory-op time (label = compute/memory ratio)')
    ax2.legend(loc='upper left', framealpha=0.95)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out2 = os.path.join(NSYS_DIR, 'compute_vs_memory.png')
    fig2.savefig(out2, dpi=140, bbox_inches='tight')
    print(f'Saved: {out2}')

    # ---- flat CSV of the breakdown for the writeup ----
    csv_path = os.path.join(NSYS_DIR, 'kernel_breakdown.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'category', 'total_ms', 'instances', 'pct_of_model', 'top_op'])
        for lab in labels:
            total = sum(v['total_ns'] for v in per_model[lab].values()) or 1
            for cat in cat_order:
                v = per_model[lab].get(cat)
                if not v:
                    continue
                w.writerow([lab, cat, f'{v["total_ns"]/1e6:.1f}',
                            v['instances'], f'{v["total_ns"]/total*100:.1f}',
                            v['top_op'][0]])
    print(f'Saved: {csv_path}')

    # ---- chart 3: latency-variance hotspots (cache-thrashing proxy) ----
    # Coefficient of variation (stddev/mean) for each kernel that ran enough
    # times to be statistically interesting and contributes >=0.5% of GPU time.
    # High CV on a workload that should be deterministic (e.g. fixed-shape GEMM)
    # suggests cache effects, scheduler contention, or unpredictable memory
    # traffic. Real cache hit/miss requires Nsight Compute, but this is a
    # useful directional signal from the existing nsys data.
    fig3, ax3 = plt.subplots(figsize=(11, 6.5))
    width = 0.16
    x_positions = list(range(5))   # top-5 high-CV kernels per model
    for i, lab in enumerate(labels):
        rows = [r for r in per_model_full[lab]
                if r['instances'] >= 50 and r['pct'] >= 0.5 and r['avg_ns'] > 0]
        rows.sort(key=lambda r: -r['std_ns'] / r['avg_ns'])
        top = rows[:5]
        cvs = [r['std_ns'] / r['avg_ns'] for r in top]
        offset = (i - len(labels) / 2 + 0.5) * width
        ax3.bar([xp + offset for xp in x_positions], cvs, width, label=lab,
                edgecolor='black', linewidth=0.4)
    ax3.set_xticks(x_positions); ax3.set_xticklabels([f'#{i+1}' for i in x_positions])
    ax3.set_xlabel('Top-N kernels by execution-time variance (per model)')
    ax3.set_ylabel('Coefficient of variation (stddev / mean)')
    ax3.set_title('Latency variance hotspots — high CV ≈ cache/scheduling unpredictability\n'
                  '(use Nsight Compute for true cache hit/miss; this is a directional proxy)')
    ax3.legend(loc='upper right', framealpha=0.95)
    ax3.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out3 = os.path.join(NSYS_DIR, 'latency_variance.png')
    fig3.savefig(out3, dpi=140, bbox_inches='tight')
    print(f'Saved: {out3}')

    # ---- chart 4: latency distribution (avg / median / max) for the dominant kernel ----
    fig4, ax4 = plt.subplots(figsize=(11, 6))
    kernel_names = []
    avgs, meds, maxes = [], [], []
    for lab in labels:
        # Pick the model's #1 compute kernel (excluding memory ops)
        compute = [r for r in per_model_full[lab] if r['category'] != 'MEMORY_OPER']
        if not compute:
            continue
        top = compute[0]
        kernel_names.append(f'{lab}\n{top["op"][:30]}...')
        avgs.append(top['avg_ns'] / 1000)
        meds.append(top['med_ns'] / 1000)
        maxes.append(top['max_ns'] / 1000)
    x = list(range(len(kernel_names)))
    w = 0.27
    ax4.bar([xi - w for xi in x], meds,  w, label='median', color='#4c78a8', edgecolor='black', linewidth=0.4)
    ax4.bar(x,                   avgs,  w, label='mean',   color='#f58518', edgecolor='black', linewidth=0.4)
    ax4.bar([xi + w for xi in x], maxes, w, label='max (tail proxy)', color='#e45756', edgecolor='black', linewidth=0.4)
    ax4.set_xticks(x); ax4.set_xticklabels(kernel_names, fontsize=8)
    ax4.set_ylabel('Per-call kernel latency (μs)')
    ax4.set_title('Top compute kernel: per-call latency distribution per model')
    ax4.set_yscale('log')   # max can be 100x median
    ax4.legend(loc='upper left', framealpha=0.95)
    ax4.grid(True, alpha=0.3, axis='y', which='both')
    plt.tight_layout()
    out4 = os.path.join(NSYS_DIR, 'top_kernel_latency.png')
    fig4.savefig(out4, dpi=140, bbox_inches='tight')
    print(f'Saved: {out4}')

    # ---- chart 5: Tier-1 model-level achieved utilization (vs FP16 peak) ----
    # Pairs each nsys-rep with its matching perf JSON (same model, paired by
    # model name in the filename) to get total tokens + wall duration. Estimates
    # FLOPs as 2 * params * tokens (standard transformer-inference approximation,
    # ignores attention's quadratic term — fine for short prompts), divides by
    # wall time, compares to peak FP16. Gives a single "achieved utilization"
    # number per model, which is the project's stated roofline-style headline.
    def _find_perf_json_for(label):
        # label is e.g. '7B-AWQ'. Find the most recent matching perf JSON.
        size, quant = label.split('-', 1)
        if quant == 'BF16':
            pat = f'Qwen_Qwen2.5-{size}-Instruct___*perf*.json'
        elif quant == 'AWQ':
            pat = f'Qwen_Qwen2.5-{size}-Instruct-AWQ___*perf*.json'
        elif quant == 'GPTQ':
            pat = f'Qwen_Qwen2.5-{size}-Instruct-GPTQ-Int8___*perf*.json'
        elif quant == 'BnB':
            pat = f'Qwen2.5-{size}-Instruct-BNB-NF4-DQ___*perf*.json'
        else:
            return None
        # Prefer perf-only runs (perf_NNNNNN_perf) over perf_quality runs, since
        # those are the ones paired with these nsys captures.
        matches = sorted(glob.glob(os.path.join(PERF_JSON_DIR, pat)))
        perf_only = [m for m in matches if '__perf_' in os.path.basename(m) and '__perf_quality_' not in os.path.basename(m)]
        return (perf_only or matches)[-1] if (perf_only or matches) else None

    util_data = []   # (label, params, tokens, duration_s, achieved_TFLOPS, util_pct)
    for lab in labels:
        pj = _find_perf_json_for(lab)
        if not pj:
            print(f'  [Tier 1] no perf JSON paired with {lab}')
            continue
        d = json.load(open(pj))
        params = (d.get('model_meta') or {}).get('parameters') or 0
        raw = d.get('raw') or {}
        in_tok = raw.get('total_input_tokens') or 0
        out_tok = raw.get('total_output_tokens') or 0
        duration = raw.get('duration') or 0
        if not (params and (in_tok or out_tok) and duration):
            continue
        flops = 2.0 * params * (in_tok + out_tok)
        achieved_tflops = flops / duration / 1e12
        util_pct = achieved_tflops / PEAK_FP16_TFLOPS * 100
        util_data.append((lab, params, in_tok + out_tok, duration, achieved_tflops, util_pct))

    if util_data:
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        labs = [u[0] for u in util_data]
        achieved = [u[4] for u in util_data]
        bars = ax5.bar(labs, achieved, color='#4c78a8', edgecolor='black', linewidth=0.5)
        ax5.axhline(PEAK_FP16_TFLOPS, color='red', linestyle='--', linewidth=1.2,
                    label=f'Peak FP16 ({PEAK_FP16_TFLOPS:.0f} TFLOPS, dense)')
        # annotate each bar with achieved + util%
        for b, (lab, _, _, _, t, u) in zip(bars, util_data):
            ax5.text(b.get_x() + b.get_width() / 2, t + PEAK_FP16_TFLOPS * 0.015,
                     f'{t:.1f} TFLOPS\n({u:.1f}% peak)', ha='center', fontsize=9)
        ax5.set_ylabel('Achieved compute (TFLOPS, FP16)')
        ax5.set_ylim(0, PEAK_FP16_TFLOPS * 1.1)
        ax5.set_title('Tier-1 roofline: achieved compute vs RTX PRO 6000 peak FP16\n'
                      '(estimated from 2·params·tokens / wall time — standard inference approximation)')
        ax5.legend(loc='upper right', framealpha=0.95)
        ax5.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        out5 = os.path.join(NSYS_DIR, 'achieved_vs_peak_compute.png')
        fig5.savefig(out5, dpi=140, bbox_inches='tight')
        print(f'Saved: {out5}')

        print('\n=== Tier-1 model-level utilization ===')
        print(f'{"Model":<10}{"Params(B)":>11}{"Tokens":>10}{"Wall(s)":>10}'
              f'{"TFLOPS":>10}{"Peak%":>8}')
        print('-' * 60)
        for lab, params, tokens, dur, t, u in util_data:
            print(f'{lab:<10}{params/1e9:>11.2f}{tokens:>10,}{dur:>10.1f}{t:>10.1f}{u:>8.1f}%')

    # ---- print headline numbers ----
    print('\n=== Summary ===')
    print(f'{"Model":<10}{"Total GPU s":>12}{"Compute":>10}{"Mem ops":>10}{"C/M ratio":>11}'
          f'{"Top compute cat":<25}{"% of total"}')
    print('-' * 100)
    for lab in labels:
        m = per_model[lab]
        total = sum(v['total_ns'] for v in m.values()) / 1e9
        memv = m.get('Memory ops', {}).get('total_ns', 0) / 1e9
        compv = total - memv
        # find top non-memory category
        compute_cats = [(c, v['total_ns']) for c, v in m.items() if c != 'Memory ops']
        compute_cats.sort(key=lambda x: -x[1])
        top_cat, top_ns = compute_cats[0] if compute_cats else ('?', 0)
        ratio = compv / memv if memv else float('inf')
        print(f'{lab:<10}{total:>12.1f}{compv:>10.1f}{memv:>10.1f}{ratio:>10.1f}x  '
              f'{top_cat:<25}{top_ns/total/1e9*100:>5.1f}%')


if __name__ == '__main__':
    main()
