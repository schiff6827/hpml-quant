import subprocess
import signal
import os
import sys
import json
import glob
from datetime import datetime

BENCHMARKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'benchmarks')
SCRIPTS_DIR = os.path.join(BENCHMARKS_DIR, 'scripts')


def _safe_script_name(name):
    return ''.join(c for c in name if c.isalnum() or c in '-_.')


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


def parse_context_sweep_result(result_dir, run_name):
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    jsons = sorted(glob.glob(os.path.join(result_dir, 'context_sweep_*.json')))
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
    save_path = os.path.join(BENCHMARKS_DIR,
                             f'{run_name}_context_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
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


def parse_perf_result(result_dir, run_name):
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    jsons = sorted(glob.glob(os.path.join(result_dir, '*.json')))
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
    save_path = os.path.join(BENCHMARKS_DIR,
                             f'{run_name}_perf_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(save_path, 'w') as f:
        json.dump(parsed, f, indent=2)
    parsed['save_path'] = save_path
    return parsed


def parse_quality_result(result_dir, run_name):
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    # lm-eval saves results as results_<timestamp>.json in a model subdirectory
    results_file = None
    for root, dirs, files in os.walk(result_dir):
        for fname in sorted(files, reverse=True):
            if fname.startswith('results') and fname.endswith('.json'):
                results_file = os.path.join(root, fname)
                break
        if results_file:
            break
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
    save_path = os.path.join(BENCHMARKS_DIR,
                             f'{run_name}_quality_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(save_path, 'w') as f:
        json.dump(parsed, f, indent=2)
    parsed['save_path'] = save_path
    return parsed


def build_pareto_dataset(quality_metric_preference=None):
    """Pair perf and quality results by run_name; return rows for the Pareto chart.

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
    for run, perf in perf_by_run.items():
        qual = qual_by_run.get(run)
        if not qual:
            continue
        tasks = qual.get('tasks', [])
        if not tasks:
            continue
        task_entry = None
        if quality_metric_preference:
            for t in tasks:
                if t.get('task') == quality_metric_preference:
                    task_entry = t
                    break
        if task_entry is None:
            task_entry = tasks[0]
        q_val = task_entry.get('acc_norm', task_entry.get('acc', task_entry.get('exact_match')))
        rows.append({
            'run_name': run,
            'throughput_tps': perf.get('output_throughput'),
            'prefill_tps': perf.get('prefill_throughput'),
            'quality_score': q_val,
            'quality_task': task_entry.get('task'),
        })
    return rows


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
