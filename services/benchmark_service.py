import subprocess
import signal
import os
import sys
import re
import csv
import json
import glob
from datetime import datetime

BENCHMARKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'benchmarks')
SCRIPTS_DIR = os.path.join(BENCHMARKS_DIR, 'scripts')

_QUANT_PATTERNS = [
    ('awq', 'AWQ'), ('gptq', 'GPTQ'), ('int4', 'INT4'), ('int8', 'INT8'),
    ('w4a16', 'W4A16'), ('w8a8', 'W8A8'), ('w8a16', 'W8A16'),
    ('fp8', 'FP8'), ('fp4', 'FP4'), ('bnb', 'BNB'), ('bitsandbytes', 'BNB'),
    ('nf4', 'NF4'), ('marlin', 'MARLIN'), ('smoothquant', 'SMOOTHQUANT'),
]

_PARAM_NAME_RE = re.compile(r'[\-_/](\d+(?:\.\d+)?)\s*[xX]?\s*([BMTbmt])\b')


def _infer_quant_from_name(name):
    lower = (name or '').lower()
    for needle, label in _QUANT_PATTERNS:
        if needle in lower:
            return label
    return None


def _params_from_name(name):
    m = _PARAM_NAME_RE.search(name or '')
    if not m:
        return 0
    num = float(m.group(1))
    unit = m.group(2).upper()
    if unit == 'B':
        return int(num * 1e9)
    if unit == 'M':
        return int(num * 1e6)
    if unit == 'T':
        return int(num * 1e12)
    return 0


def _find_snapshot_dir(model_id):
    """Return a local snapshot dir for a cached HF model, or None."""
    if not model_id:
        return None
    if os.path.isdir(model_id):
        return model_id
    try:
        import config
        root = config.MODEL_CACHE_DIR
    except Exception:
        return None
    safe = model_id.replace('/', '--')
    pattern = os.path.join(root, f'models--{safe}', 'snapshots', '*')
    snaps = sorted(glob.glob(pattern))
    return snaps[-1] if snaps else None


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _params_from_snapshot(snap_dir):
    """Sum parameter counts from safetensors index, or estimate from config."""
    if not snap_dir:
        return 0
    idx = _read_json(os.path.join(snap_dir, 'model.safetensors.index.json'))
    if idx and isinstance(idx.get('metadata'), dict):
        tp = idx['metadata'].get('total_parameters')
        if isinstance(tp, (int, float)) and tp > 0:
            return int(tp)
    cfg = _read_json(os.path.join(snap_dir, 'config.json'))
    if cfg:
        h = cfg.get('hidden_size') or cfg.get('n_embd')
        layers = cfg.get('num_hidden_layers') or cfg.get('n_layer')
        vocab = cfg.get('vocab_size')
        if h and layers and vocab:
            return int(12 * h * h * layers + 2 * h * vocab)
    return 0


def _size_bytes_from_snapshot(snap_dir):
    if not snap_dir:
        return 0
    total = 0
    for root, _, files in os.walk(snap_dir):
        for fn in files:
            if fn.endswith(('.safetensors', '.bin', '.pt')):
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                except OSError:
                    pass
    return total


def get_model_metadata(model_id, port=None):
    """Best-effort quantization / parameters / size / dtype for a model.

    Priority: running-server info (authoritative for quant/dtype) > HF config
    on disk > inference from model id.
    """
    meta = {
        'model_id': model_id,
        'quantization': None,
        'parameters': 0,
        'size_bytes': 0,
        'dtype': None,
        'weight_mem_gib': None,
    }

    if port is not None:
        try:
            from services import vllm_service
            info = vllm_service.get_server_info(port) or {}
            if info.get('quantization'):
                meta['quantization'] = str(info['quantization']).upper()
            if info.get('dtype') and info['dtype'] != 'auto':
                meta['dtype'] = info['dtype']
            if info.get('weight_mem_gib'):
                meta['weight_mem_gib'] = info['weight_mem_gib']
        except Exception:
            pass

    snap = _find_snapshot_dir(model_id)
    cfg = _read_json(os.path.join(snap, 'config.json')) if snap else None
    if cfg:
        qc = cfg.get('quantization_config')
        if qc and not meta['quantization']:
            qm = qc.get('quant_method') or qc.get('quant_type')
            if qm:
                meta['quantization'] = str(qm).upper()
        if not meta['dtype']:
            td = cfg.get('torch_dtype')
            if td:
                meta['dtype'] = str(td)

    if not meta['quantization']:
        inferred = _infer_quant_from_name(model_id)
        meta['quantization'] = inferred or (meta['dtype'] and meta['dtype'].upper()) or 'NONE'

    meta['parameters'] = _params_from_snapshot(snap) or _params_from_name(model_id)
    meta['size_bytes'] = _size_bytes_from_snapshot(snap)
    return meta


def summarize_profile_csv(csv_path):
    """Reduce a profile CSV to scalar peaks/averages for storage on a result."""
    if not csv_path or not os.path.exists(csv_path):
        return {}
    gpu_mems, gpu_utils, gpu_temps, gpu_powers = [], [], [], []
    rss_list, kv_list = [], []
    gpu_mem_total = 0
    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                def _f(k):
                    try:
                        return float(row.get(k) or 0)
                    except (TypeError, ValueError):
                        return 0.0
                gpu_mems.append(_f('gpu_mem_used_mb'))
                gpu_utils.append(_f('gpu_util_pct'))
                gpu_temps.append(_f('gpu_temp_c'))
                gpu_powers.append(_f('gpu_power_w'))
                rss_list.append(_f('cpu_mem_rss_mb'))
                kv_list.append(_f('kv_cache_pct'))
                gmt = _f('gpu_mem_total_mb')
                if gmt:
                    gpu_mem_total = gmt
    except Exception:
        return {}

    def _avg(xs):
        xs = [x for x in xs if x]
        return sum(xs) / len(xs) if xs else 0

    return {
        'samples': len(gpu_mems),
        'peak_gpu_mem_mb': max(gpu_mems, default=0),
        'avg_gpu_mem_mb': _avg(gpu_mems),
        'gpu_mem_total_mb': gpu_mem_total,
        'peak_gpu_util_pct': max(gpu_utils, default=0),
        'avg_gpu_util_pct': _avg(gpu_utils),
        'peak_gpu_temp_c': max(gpu_temps, default=0),
        'peak_gpu_power_w': max(gpu_powers, default=0),
        'avg_gpu_power_w': _avg(gpu_powers),
        'peak_cpu_rss_mb': max(rss_list, default=0),
        'peak_kv_cache_pct': max(kv_list, default=0),
    }


def read_profile_csv(csv_path, max_points=600):
    """Load a profile CSV as parallel lists for charting. Downsamples to max_points."""
    if not csv_path or not os.path.exists(csv_path):
        return {}
    rows = []
    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception:
        return {}
    if not rows:
        return {}
    if len(rows) > max_points:
        step = len(rows) // max_points
        rows = rows[::step]

    def _col(name, fn=float):
        out = []
        for r in rows:
            try:
                out.append(fn(r.get(name) or 0))
            except (TypeError, ValueError):
                out.append(0)
        return out

    return {
        'timestamps': [r.get('timestamp', '')[-8:] for r in rows],
        'gpu_mem_mb': _col('gpu_mem_used_mb'),
        'gpu_util_pct': _col('gpu_util_pct'),
        'gpu_temp_c': _col('gpu_temp_c'),
        'gpu_power_w': _col('gpu_power_w'),
        'kv_cache_pct': _col('kv_cache_pct'),
        'tokens_per_sec': _col('gen_tokens_per_sec') if any(r.get('gen_tokens_per_sec') for r in rows) else [],
        'requests_running': _col('requests_running'),
        'requests_waiting': _col('requests_waiting'),
        'cpu_rss_mb': _col('cpu_mem_rss_mb'),
    }


def _safe_script_name(name):
    return ''.join(c for c in name if c.isalnum() or c in '-_.')


def _safe_filename_stem(name):
    """Safe stem for result filenames: keep letters/digits/.-_, collapse everything
    else (spaces, slashes, etc.) to underscores. Prevents path-separator surprises
    like a run_name containing '/'."""
    cleaned = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in (name or '').strip())
    return cleaned or 'unnamed'


def list_scripts():
    if not os.path.isdir(SCRIPTS_DIR):
        return []
    out = []
    for fpath in sorted(glob.glob(os.path.join(SCRIPTS_DIR, '*.json'))):
        try:
            data = json.loads(open(fpath).read())
            out.append({'name': data.get('name', os.path.splitext(os.path.basename(fpath))[0]),
                        'path': fpath})
        except Exception:
            pass
    return out


def save_script(name, config):
    os.makedirs(SCRIPTS_DIR, exist_ok=True)
    safe = _safe_script_name(name) or 'script'
    path = os.path.join(SCRIPTS_DIR, f'{safe}.json')
    data = dict(config)
    data['name'] = name
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    return path


def load_script(name_or_path):
    if os.path.isabs(name_or_path) and os.path.exists(name_or_path):
        path = name_or_path
    else:
        path = os.path.join(SCRIPTS_DIR, f'{_safe_script_name(name_or_path)}.json')
    if not os.path.exists(path):
        return None
    return json.loads(open(path).read())


def delete_script(name_or_path):
    if os.path.isabs(name_or_path) and os.path.exists(name_or_path):
        path = name_or_path
    else:
        path = os.path.join(SCRIPTS_DIR, f'{_safe_script_name(name_or_path)}.json')
    if os.path.exists(path):
        os.remove(path)
        return True
    return False

_active_proc = None

_SHAREGPT_PATH = None


def _get_sharegpt_path():
    global _SHAREGPT_PATH
    if _SHAREGPT_PATH and os.path.exists(_SHAREGPT_PATH):
        return _SHAREGPT_PATH
    from huggingface_hub import hf_hub_download
    _SHAREGPT_PATH = hf_hub_download(
        repo_id='anon8231489123/ShareGPT_Vicuna_unfiltered',
        filename='ShareGPT_V3_unfiltered_cleaned_split.json',
        repo_type='dataset',
    )
    return _SHAREGPT_PATH


def run_perf_benchmark(port, model, dataset, num_prompts, request_rate,
                       max_concurrency, random_input_len, random_output_len,
                       result_dir, run_name):
    global _active_proc
    os.makedirs(result_dir, exist_ok=True)
    cmd = [
        'vllm', 'bench', 'serve',
        '--base-url', f'http://localhost:{port}',
        '--model', model,
        '--dataset-name', dataset,
        '--num-prompts', str(num_prompts),
        '--request-rate', str(request_rate),
        '--max-concurrency', str(max_concurrency),
        '--save-result', '--result-dir', result_dir,
        '--metric-percentiles', '50,75,90,95,99',
        '--percentile-metrics', 'ttft,tpot,itl,e2el',
    ]
    if dataset == 'random':
        cmd += ['--random-input-len', str(random_input_len),
                '--random-output-len', str(random_output_len)]
    elif dataset == 'sharegpt':
        path = _get_sharegpt_path()
        if path:
            cmd += ['--dataset-path', path]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    _active_proc = proc
    return proc


_CTX_SWEEP_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts', 'run_context_sweep.py')


def run_context_sweep(port, model, result_dir, run_name, upper_bound, step):
    """Launch a context-length sweep as a subprocess that streams probe results."""
    global _active_proc
    os.makedirs(result_dir, exist_ok=True)
    cmd = [
        sys.executable, _CTX_SWEEP_SCRIPT,
        '--port', str(port),
        '--model', model,
        '--upper-bound', str(int(upper_bound)),
        '--step', str(int(step)),
        '--result-dir', result_dir,
        '--run-name', run_name,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    _active_proc = proc
    return proc


def _wait_for_glob(pattern, timeout=5.0, interval=0.2):
    """Retry a glob for up to `timeout` seconds. Handles fs flush races
    (e.g. WSL2) where the file appears a moment after the subprocess exits."""
    import time
    deadline = time.time() + timeout
    while True:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches
        if time.time() >= deadline:
            return []
        time.sleep(interval)


def parse_context_sweep_result(result_dir, run_name, extras=None):
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    jsons = _wait_for_glob(os.path.join(result_dir, 'context_sweep_*.json'))
    if not jsons:
        return None
    raw = json.loads(open(jsons[-1]).read())
    parsed = {
        'type': 'context_sweep',
        'run_name': run_name,
        'timestamp': datetime.now().isoformat(),
        'max_context_tokens': raw.get('max_context_tokens'),
        'probes': raw.get('probes', []),
        'raw': raw,
    }
    if extras:
        parsed.update(extras)
    save_path = os.path.join(BENCHMARKS_DIR,
                             f'{_safe_filename_stem(run_name)}_context_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(save_path, 'w') as f:
        json.dump(parsed, f, indent=2)
    parsed['save_path'] = save_path
    return parsed


def run_quality_benchmark(port, model, tasks, num_fewshot, num_concurrent,
                          limit, result_dir, run_name):
    global _active_proc
    os.makedirs(result_dir, exist_ok=True)
    cmd = [
        'lm_eval', 'run',
        '--model', 'local-completions',
        '--model_args', f'model={model},base_url=http://localhost:{port}/v1/completions,num_concurrent={num_concurrent}',
        '--tasks', ','.join(tasks),
        '--num_fewshot', str(num_fewshot),
        '--output_path', result_dir,
    ]
    if limit and int(limit) > 0:
        cmd += ['--limit', str(int(limit))]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    _active_proc = proc
    return proc


def stop_benchmark():
    global _active_proc
    if _active_proc and _active_proc.poll() is None:
        _active_proc.send_signal(signal.SIGTERM)
        try:
            _active_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _active_proc.kill()
    _active_proc = None


def is_running():
    return _active_proc is not None and _active_proc.poll() is None


def parse_perf_result(result_dir, run_name, extras=None):
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    jsons = _wait_for_glob(os.path.join(result_dir, '*.json'))
    if not jsons:
        return None
    raw = json.loads(open(jsons[-1]).read())
    # Derive prefill throughput: total prompt tokens / total prefill time.
    # Prefill time per request ≈ TTFT, so sum_prefill ≈ mean_ttft_ms/1000 * completed.
    prefill_tps = None
    total_input = raw.get('total_input_tokens')
    mean_ttft_ms = raw.get('mean_ttft_ms')
    completed = raw.get('completed') or raw.get('num_prompts')
    if total_input and mean_ttft_ms and completed:
        total_ttft_sec = (mean_ttft_ms / 1000.0) * completed
        if total_ttft_sec > 0:
            prefill_tps = total_input / total_ttft_sec
    parsed = {
        'type': 'perf',
        'run_name': run_name,
        'timestamp': datetime.now().isoformat(),
        'request_throughput': raw.get('request_throughput'),
        'output_throughput': raw.get('output_throughput'),
        'prefill_throughput': prefill_tps,
        'metrics': {},
        'raw': raw,
    }
    for metric in ['ttft', 'tpot', 'itl', 'e2el']:
        parsed['metrics'][metric] = {}
        for p in ['50', '75', '90', '95', '99']:
            key = f'{metric}_percentiles_p{p}'
            if key in raw:
                parsed['metrics'][metric][f'p{p}'] = raw[key]
        mean_key = f'mean_{metric}_ms'
        if mean_key in raw:
            parsed['metrics'][metric]['mean'] = raw[mean_key]
    if extras:
        parsed.update(extras)
    save_path = os.path.join(BENCHMARKS_DIR,
                             f'{_safe_filename_stem(run_name)}_perf_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(save_path, 'w') as f:
        json.dump(parsed, f, indent=2)
    parsed['save_path'] = save_path
    return parsed


def parse_quality_result(result_dir, run_name, extras=None):
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    # lm-eval saves results as results_<timestamp>.json in a model subdirectory.
    # Retry briefly to absorb fs flush races.
    import time
    deadline = time.time() + 5.0
    results_file = None
    while True:
        for root, dirs, files in os.walk(result_dir):
            for fname in sorted(files, reverse=True):
                if fname.startswith('results') and fname.endswith('.json'):
                    results_file = os.path.join(root, fname)
                    break
            if results_file:
                break
        if results_file or time.time() >= deadline:
            break
        time.sleep(0.2)
    if not results_file:
        return None
    raw = json.loads(open(results_file).read())
    parsed = {
        'type': 'quality',
        'run_name': run_name,
        'timestamp': datetime.now().isoformat(),
        'tasks': [],
        'raw': raw,
    }
    results = raw.get('results', {})
    for task_name, task_data in results.items():
        entry = {'task': task_name}
        for key in ['acc,none', 'acc_norm,none', 'exact_match,strict-match',
                     'exact_match,flexible-extract']:
            if key in task_data:
                metric_name = key.split(',')[0]
                entry[metric_name] = task_data[key]
                stderr_key = f'{key}_stderr'
                if stderr_key in task_data:
                    entry[f'{metric_name}_stderr'] = task_data[stderr_key]
        if len(entry) > 1:
            parsed['tasks'].append(entry)
    if extras:
        parsed.update(extras)
    save_path = os.path.join(BENCHMARKS_DIR,
                             f'{_safe_filename_stem(run_name)}_quality_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(save_path, 'w') as f:
        json.dump(parsed, f, indent=2)
    parsed['save_path'] = save_path
    return parsed


def build_pareto_dataset(quality_metric_preference=None):
    """Aggregate all saved perf and/or quality results into Pareto rows.

    Each row represents one run_name. Rows include a quality score if there is
    a matching quality result, and throughput if there is a matching perf result.
    Rows without either are still returned so the UI can plot alternate axes.

    quality_metric_preference: optional task key (e.g. 'mmlu'). If None, pick the
    first task present on each quality result.
    """
    if not os.path.isdir(BENCHMARKS_DIR):
        return []
    perf_by_run = {}
    qual_by_run = {}
    for fpath in glob.glob(os.path.join(BENCHMARKS_DIR, '*.json')):
        try:
            data = json.loads(open(fpath).read())
        except Exception:
            continue
        run = data.get('run_name')
        if not run:
            continue
        if data.get('type') == 'perf':
            if run not in perf_by_run or data.get('timestamp', '') > perf_by_run[run].get('timestamp', ''):
                perf_by_run[run] = data
        elif data.get('type') == 'quality':
            if run not in qual_by_run or data.get('timestamp', '') > qual_by_run[run].get('timestamp', ''):
                qual_by_run[run] = data

    rows = []
    all_runs = sorted(set(perf_by_run) | set(qual_by_run))
    for run in all_runs:
        perf = perf_by_run.get(run) or {}
        qual = qual_by_run.get(run) or {}

        q_val = None
        q_task = None
        tasks = qual.get('tasks', []) if qual else []
        if tasks:
            task_entry = None
            if quality_metric_preference:
                for t in tasks:
                    if t.get('task') == quality_metric_preference:
                        task_entry = t
                        break
            if task_entry is None:
                task_entry = tasks[0]
            q_val = task_entry.get('acc_norm', task_entry.get('acc', task_entry.get('exact_match')))
            q_task = task_entry.get('task')

        meta = perf.get('model_meta') or qual.get('model_meta') or {}
        prof = perf.get('profile_summary') or qual.get('profile_summary') or {}
        params = meta.get('parameters') or 0
        size_bytes = meta.get('size_bytes') or 0
        peak_vram_mb = prof.get('peak_gpu_mem_mb') or 0
        avg_power_w = prof.get('avg_gpu_power_w') or 0
        rows.append({
            'run_name': run,
            'has_perf': bool(perf),
            'has_quality': bool(qual),
            'throughput_tps': perf.get('output_throughput') if perf else None,
            'prefill_tps': perf.get('prefill_throughput') if perf else None,
            'quality_score': q_val,
            'quality_task': q_task,
            'model_id': meta.get('model_id') or '',
            'quantization': (meta.get('quantization') or 'UNKNOWN').upper(),
            'parameters': params,
            'parameters_b': (params / 1e9) if params else 0,
            'size_bytes': size_bytes,
            'size_gb': (size_bytes / 1e9) if size_bytes else 0,
            'peak_vram_gb': (peak_vram_mb / 1024) if peak_vram_mb else 0,
            'avg_gpu_power_w': avg_power_w,
            'dtype': meta.get('dtype') or '',
        })
    return rows


def list_run_names_seen():
    """Distinct run_names across all saved results — for the Pareto run picker."""
    if not os.path.isdir(BENCHMARKS_DIR):
        return []
    names = set()
    for fpath in glob.glob(os.path.join(BENCHMARKS_DIR, '*.json')):
        try:
            data = json.loads(open(fpath).read())
        except Exception:
            continue
        if data.get('type') not in ('perf', 'quality', 'context_sweep'):
            continue
        run = data.get('run_name')
        if run:
            names.add(run)
    return sorted(names)


def list_quantizations_seen():
    """Distinct quantization labels across all perf/quality results."""
    if not os.path.isdir(BENCHMARKS_DIR):
        return []
    out = set()
    for fpath in glob.glob(os.path.join(BENCHMARKS_DIR, '*.json')):
        try:
            data = json.loads(open(fpath).read())
        except Exception:
            continue
        meta = data.get('model_meta') or {}
        q = meta.get('quantization')
        if q:
            out.add(str(q).upper())
    return sorted(out) or ['UNKNOWN']


def backfill_metadata(port_for_running_model=None):
    """Re-derive model_meta and profile_summary for existing benchmark JSONs
    that lack them. Returns how many files were updated.
    """
    if not os.path.isdir(BENCHMARKS_DIR):
        return 0
    updated = 0
    for fpath in glob.glob(os.path.join(BENCHMARKS_DIR, '*.json')):
        try:
            data = json.loads(open(fpath).read())
        except Exception:
            continue
        if data.get('type') not in ('perf', 'quality', 'context_sweep'):
            continue
        changed = False
        model_id = (data.get('raw') or {}).get('model_id') or data.get('model_id') or ''
        if 'model_meta' not in data and model_id:
            data['model_meta'] = get_model_metadata(model_id, port=port_for_running_model)
            changed = True
        csv_path = data.get('profile_csv')
        if csv_path and 'profile_summary' not in data:
            summary = summarize_profile_csv(csv_path)
            if summary:
                data['profile_summary'] = summary
                changed = True
        if changed:
            try:
                with open(fpath, 'w') as f:
                    json.dump(data, f, indent=2)
                updated += 1
            except Exception:
                pass
    return updated


def list_quality_tasks_seen():
    """Return all unique task names across saved quality results (for y-axis chooser)."""
    if not os.path.isdir(BENCHMARKS_DIR):
        return []
    tasks = set()
    for fpath in glob.glob(os.path.join(BENCHMARKS_DIR, '*.json')):
        try:
            data = json.loads(open(fpath).read())
        except Exception:
            continue
        if data.get('type') != 'quality':
            continue
        for t in data.get('tasks', []):
            if t.get('task'):
                tasks.add(t['task'])
    return sorted(tasks)


def list_saved_results():
    if not os.path.isdir(BENCHMARKS_DIR):
        return []
    results = []
    for fpath in sorted(glob.glob(os.path.join(BENCHMARKS_DIR, '*.json')), reverse=True):
        fname = os.path.basename(fpath)
        try:
            data = json.loads(open(fpath).read())
            results.append({
                'name': data.get('run_name', fname),
                'type': data.get('type', 'unknown'),
                'timestamp': data.get('timestamp', ''),
                'path': fpath,
                'filename': fname,
            })
        except Exception:
            pass
    return results


def load_result(path):
    with open(path) as f:
        return json.load(f)


def compare_results(path1, path2):
    r1 = load_result(path1)
    r2 = load_result(path2)
    comparison = {
        'run_a': r1.get('run_name', 'A'),
        'run_b': r2.get('run_name', 'B'),
        'type': r1.get('type', 'unknown'),
        'rows': [],
    }
    if r1.get('type') == 'perf' and r2.get('type') == 'perf':
        comparison['rows'].append({
            'metric': 'Request Throughput (req/s)',
            'a': r1.get('request_throughput'),
            'b': r2.get('request_throughput'),
        })
        comparison['rows'].append({
            'metric': 'Output Throughput (tok/s)',
            'a': r1.get('output_throughput'),
            'b': r2.get('output_throughput'),
        })
        comparison['rows'].append({
            'metric': 'Prefill Throughput (tok/s)',
            'a': r1.get('prefill_throughput'),
            'b': r2.get('prefill_throughput'),
        })
        for metric in ['ttft', 'tpot', 'itl', 'e2el']:
            for p in ['mean', 'p50', 'p99']:
                a_val = r1.get('metrics', {}).get(metric, {}).get(p)
                b_val = r2.get('metrics', {}).get(metric, {}).get(p)
                if a_val is not None or b_val is not None:
                    comparison['rows'].append({
                        'metric': f'{metric.upper()} {p} (ms)',
                        'a': a_val,
                        'b': b_val,
                    })
    elif r1.get('type') == 'quality' and r2.get('type') == 'quality':
        tasks_a = {t['task']: t for t in r1.get('tasks', [])}
        tasks_b = {t['task']: t for t in r2.get('tasks', [])}
        all_tasks = sorted(set(list(tasks_a.keys()) + list(tasks_b.keys())))
        for task in all_tasks:
            a_data = tasks_a.get(task, {})
            b_data = tasks_b.get(task, {})
            metric_key = 'acc_norm' if 'acc_norm' in a_data or 'acc_norm' in b_data else 'acc'
            comparison['rows'].append({
                'metric': task,
                'a': a_data.get(metric_key),
                'b': b_data.get(metric_key),
            })
    return comparison
