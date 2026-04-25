"""Queue service for serial benchmark runs.

Queue schema (v2 — models × benchmarks):
    queue = {
        name, created, status, current_job_index, started_at, finished_at,
        models:       [{id, name, model, launch, timeouts}, ...],
        benchmarks:   [{id, name, config}, ...],  # config is a saved-script dict
        jobs:         [... expanded at start time from models × benchmarks],
    }

- Worker = single asyncio task; one job at a time; each job = (one model, one benchmark)
- Soft cap + idle window + hard cap per phase; GPU + process CPU used as activity signals
- Heartbeats every 30 min; immediate notify on failure/done
- Restart recovery: active.json is the source of truth; running-at-crash = failed
- Retry queue = new queue whose `jobs` list has the exact failed combinations
"""
import asyncio
import glob
import json
import os
import subprocess
import time
import uuid
from datetime import datetime

import config
from services import benchmark_service, metrics_service, notify_service, vllm_service

QUEUE_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'benchmarks', 'queue')
ACTIVE_PATH = os.path.join(QUEUE_ROOT, 'active.json')
PRESETS_DIR = os.path.join(QUEUE_ROOT, 'presets')
LOGS_DIR = os.path.join(QUEUE_ROOT, 'logs')

for d in (QUEUE_ROOT, PRESETS_DIR, LOGS_DIR):
    os.makedirs(d, exist_ok=True)

HEARTBEAT_INTERVAL_S = 30 * 60
GPU_ACTIVITY_THRESHOLD = 5.0
CPU_ACTIVITY_THRESHOLD = 1.0
HEALTH_POLL_INTERVAL_S = 1.0

DEFAULT_TIMEOUTS = {
    'launch_soft_s': 300,
    'launch_hard_s': 1200,
    'bench_soft_s': 1800,
    'bench_hard_s': 14400,
    'launch_idle_window_s': 300,     # 5 min (per spec)
    'bench_idle_window_s': 600,      # 10 min (per spec)
}


class SoftTimeout(Exception):
    pass


class HardTimeout(Exception):
    pass


class QueueCancelled(Exception):
    pass


# ── Module state ─────────────────────────────────────────────────────────────
_state = {
    'queue': None,
    'running': False,
    'task': None,
    'cancel_requested': False,
    'log_tail': [],
    'last_heartbeat_at': 0,
}
_LOG_TAIL_MAX = 300


def _now_iso():
    return datetime.now().isoformat(timespec='seconds')


def _safe(name):
    return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in (name or '').strip()) or 'queue'


def new_queue(name='queue'):
    return {
        'name': name,
        'created': _now_iso(),
        'models': [],
        'benchmarks': [],
        'jobs': [],
        'status': 'idle',
        'current_job_index': -1,
        'started_at': None,
        'finished_at': None,
    }


def _new_model_entry(model, launch, timeouts=None, name=None):
    return {
        'id': str(uuid.uuid4())[:8],
        'name': name or model,
        'model': model,
        'launch': dict(launch),
        'timeouts': {**DEFAULT_TIMEOUTS, **(timeouts or {})},
    }


def _new_benchmark_entry(name, config_dict):
    return {
        'id': str(uuid.uuid4())[:8],
        'name': name,
        'config': dict(config_dict),
    }


def _new_job(model_entry, bench_entry):
    return {
        'id': str(uuid.uuid4())[:8],
        'name': f'{model_entry["name"]} · {bench_entry["name"]}',
        'model': model_entry['model'],
        'launch': dict(model_entry['launch']),
        'timeouts': dict(model_entry['timeouts']),
        'bench_name': bench_entry['name'],
        'bench_config': dict(bench_entry['config']),
        'status': 'pending',
        'error': None,
        'started_at': None,
        'finished_at': None,
        'result_paths': [],
        'port': None,
        'current_step': None,
        'phase_started_at': None,
        'phase_soft_s': None,
        'phase_hard_s': None,
        'phase_idle_window_s': None,
        'phase_last_activity_at': None,
        'phase_soft_crossed': False,
    }


def _ensure_queue():
    if _state['queue'] is None:
        _state['queue'] = new_queue()
    return _state['queue']


# ── Persistence ──────────────────────────────────────────────────────────────
def _persist():
    q = _state['queue']
    if not q:
        if os.path.exists(ACTIVE_PATH):
            os.remove(ACTIVE_PATH)
        return
    serial = {
        **q,
        'jobs': [
            {k: v for k, v in j.items() if not k.startswith('phase_')}
            for j in q.get('jobs', [])
        ],
    }
    tmp = ACTIVE_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(serial, f, indent=2)
    os.replace(tmp, ACTIVE_PATH)


def _log(line):
    tag = datetime.now().strftime('%H:%M:%S')
    formatted = f'[{tag}] {line}'
    _state['log_tail'].append(formatted)
    if len(_state['log_tail']) > _LOG_TAIL_MAX:
        del _state['log_tail'][:-_LOG_TAIL_MAX]
    q = _state.get('queue')
    if q:
        path = os.path.join(LOGS_DIR, f'{_safe(q["name"])}.log')
        try:
            with open(path, 'a') as f:
                f.write(formatted + '\n')
        except Exception:
            pass


# ── Activity detection ───────────────────────────────────────────────────────
def _gpu_util_pct():
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0:
            return float(r.stdout.strip().split('\n')[0])
    except Exception:
        pass
    return 0.0


class _CpuMonitor:
    _HZ = os.sysconf('SC_CLK_TCK') if hasattr(os, 'sysconf') else 100

    def __init__(self, pid):
        self.pid = pid
        self.last_ticks = None
        self.last_wall = None

    def sample(self):
        try:
            with open(f'/proc/{self.pid}/stat') as f:
                parts = f.read().split()
            total = int(parts[13]) + int(parts[14])
            now = time.monotonic()
            if self.last_ticks is None:
                self.last_ticks = total
                self.last_wall = now
                return 0.0
            dt = now - self.last_wall
            dticks = total - self.last_ticks
            self.last_ticks = total
            self.last_wall = now
            if dt <= 0:
                return 0.0
            return (dticks / self._HZ / dt) * 100
        except Exception:
            return 0.0


# ── Phase runner ─────────────────────────────────────────────────────────────
async def _monitor_phase(job, phase_label, condition_fn, is_active_fn, soft_s, hard_s, idle_window_s):
    start = time.monotonic()
    last_active = start
    job['current_step'] = phase_label
    job['phase_started_at'] = start
    job['phase_soft_s'] = soft_s
    job['phase_hard_s'] = hard_s
    job['phase_idle_window_s'] = idle_window_s
    job['phase_last_activity_at'] = start
    job['phase_soft_crossed'] = False

    while True:
        now = time.monotonic()
        elapsed = now - start

        if elapsed >= hard_s:
            raise HardTimeout(f'{phase_label}: hard cap {hard_s}s reached')

        if condition_fn():
            return

        if _state.get('cancel_requested'):
            raise QueueCancelled()

        if is_active_fn():
            last_active = now
            job['phase_last_activity_at'] = now

        if elapsed >= soft_s:
            job['phase_soft_crossed'] = True
            if (now - last_active) >= idle_window_s:
                raise SoftTimeout(
                    f'{phase_label}: soft cap {soft_s}s + idle {idle_window_s}s reached'
                )

        await _maybe_heartbeat()
        await asyncio.sleep(HEALTH_POLL_INTERVAL_S)


# ── Launch phase ─────────────────────────────────────────────────────────────
def _resolve_model(model_id):
    return model_id  # local paths already resolved by the caller


async def _launch_phase(job):
    launch = job['launch']
    use_kv_gb = launch.get('use_kv_gb', True)
    extra = ['--trust-remote-code'] if launch.get('trust_remote_code') else None
    cpu_offload = launch.get('cpu_offload_gb') or None

    _log(f"Launching vLLM for '{job['model']}'")
    kwargs = dict(
        model=_resolve_model(job['model']),
        port=None,
        gpu_mem_util=launch.get('gpu_mem_util', config.DEFAULT_GPU_MEM_UTIL),
        dtype=launch.get('dtype', config.DEFAULT_DTYPE),
        quantization=launch.get('quantization') or None,
        extra_args=extra,
        kv_cache_gb=int(launch.get('kv_cache_gb', 10)) if use_kv_gb else None,
    )
    try:
        port = vllm_service.launch_server(cpu_offload_gb=cpu_offload, **kwargs)
    except TypeError:
        port = vllm_service.launch_server(**kwargs)
    job['port'] = port
    _log(f'  → port {port}')

    proc = None
    try:
        proc = vllm_service._running.get(port, {}).get('proc')
    except Exception:
        pass
    pid = proc.pid if proc and hasattr(proc, 'pid') else None
    cpu_mon = _CpuMonitor(pid) if pid else None

    def is_active():
        gpu = _gpu_util_pct()
        cpu = cpu_mon.sample() if cpu_mon else 0
        return gpu >= GPU_ACTIVITY_THRESHOLD or cpu >= CPU_ACTIVITY_THRESHOLD

    def healthy():
        if not vllm_service.is_alive(port):
            raise RuntimeError('vLLM process exited during launch')
        return vllm_service.check_health(port)

    await _monitor_phase(
        job=job, phase_label='launch',
        condition_fn=healthy, is_active_fn=is_active,
        soft_s=job['timeouts']['launch_soft_s'],
        hard_s=job['timeouts']['launch_hard_s'],
        idle_window_s=job['timeouts']['launch_idle_window_s'],
    )
    _log(f'  → healthy on {port}')
    return port


# ── Benchmark phase ──────────────────────────────────────────────────────────
def _run_one_subprocess(bench_type, script_cfg, port, model, run_name, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    if bench_type == 'perf':
        perf = script_cfg.get('perf', {})
        proc = benchmark_service.run_perf_benchmark(
            port=port, model=model,
            dataset=perf.get('dataset', 'random'),
            num_prompts=int(perf.get('num_prompts', 100)),
            request_rate=float(perf.get('request_rate', 0)),
            max_concurrency=int(perf.get('max_concurrency', 1)),
            random_input_len=int(perf.get('random_input_len', 1024)),
            random_output_len=int(perf.get('random_output_len', 128)),
            result_dir=result_dir, run_name=run_name,
        )
        return proc, lambda extras: benchmark_service.parse_perf_result(result_dir, run_name, extras=extras)
    if bench_type == 'quality':
        qual = script_cfg.get('quality', {})
        proc = benchmark_service.run_quality_benchmark(
            port=port, model=model,
            tasks=qual.get('tasks', []),
            num_fewshot=int(qual.get('num_fewshot', 0)),
            num_concurrent=int(qual.get('num_concurrent', 1)),
            limit=int(qual.get('limit', 0) or 0),
            result_dir=result_dir, run_name=run_name,
        )
        return proc, lambda extras: benchmark_service.parse_quality_result(result_dir, run_name, extras=extras)
    if bench_type == 'context_sweep':
        ctx = script_cfg.get('context_sweep', {})
        proc = benchmark_service.run_context_sweep(
            port=port, model=model, result_dir=result_dir, run_name=run_name,
            upper_bound=int(ctx.get('upper_bound', 32768)),
            step=int(ctx.get('step', 1024)),
        )
        return proc, lambda extras: benchmark_service.parse_context_sweep_result(result_dir, run_name, extras=extras)
    raise ValueError(f'unknown bench type: {bench_type}')


def _enabled_bench_types(script_cfg):
    out = []
    for t in ('perf', 'quality', 'context_sweep'):
        if script_cfg.get(t, {}).get('enabled'):
            out.append(t)
    return out


async def _bench_phase(job, port):
    script_cfg = job.get('bench_config') or {}
    bench_types = _enabled_bench_types(script_cfg)
    if not bench_types:
        raise RuntimeError(f"benchmark '{job['bench_name']}' has no enabled types (perf/quality/context_sweep)")

    model_meta = benchmark_service.get_model_metadata(job['model'], port=port)

    # Start GPU/CPU metrics recording for the whole job (shared across bench types).
    metrics_run_name = _safe(job['name'])
    profile_csv = None
    try:
        profile_csv = metrics_service.start_run_recording(port, metrics_run_name)
        if profile_csv:
            _log(f'  profile recording → {os.path.basename(profile_csv)}')
    except Exception as e:
        _log(f'  profile recording failed to start: {e}')

    try:
        for bench_type in bench_types:
            run_name = f'{_safe(job["name"])}_{bench_type}'
            result_dir = os.path.join('/tmp', f'queue_{job["id"]}_{bench_type}')
            _log(f"Starting {bench_type} benchmark")

            proc, parser = _run_one_subprocess(bench_type, script_cfg, port, job['model'], run_name, result_dir)
            cpu_mon = _CpuMonitor(proc.pid)

            def alive_check(p=proc):
                return p.poll() is not None

            def is_active(cm=cpu_mon):
                gpu = _gpu_util_pct()
                cpu = cm.sample()
                return gpu >= GPU_ACTIVITY_THRESHOLD or cpu >= CPU_ACTIVITY_THRESHOLD

            try:
                await _monitor_phase(
                    job=job,
                    phase_label=f'bench:{bench_type}',
                    condition_fn=alive_check,
                    is_active_fn=is_active,
                    soft_s=job['timeouts']['bench_soft_s'],
                    hard_s=job['timeouts']['bench_hard_s'],
                    idle_window_s=job['timeouts']['bench_idle_window_s'],
                )
            except (SoftTimeout, HardTimeout, QueueCancelled):
                _kill_proc(proc)
                raise

            if proc.returncode != 0:
                raise RuntimeError(f'{bench_type} benchmark exited with code {proc.returncode}')

            try:
                profile_summary = benchmark_service.summarize_profile_csv(profile_csv) if profile_csv else {}
            except Exception:
                profile_summary = {}
            extras = {
                'model_meta': model_meta,
                'profile_csv': profile_csv,
                'profile_summary': profile_summary,
            }
            parsed = parser(extras)
            if parsed:
                save_path = parsed.get('save_path')
                if save_path:
                    job['result_paths'].append(save_path)
                _log(f'  → {bench_type} done: {os.path.basename(save_path or "?")}')
            else:
                _log(f'  → {bench_type} finished but no result file parsed')
    finally:
        if profile_csv:
            try:
                metrics_service.stop_run_recording(metrics_run_name)
                _log('  profile recording stopped')
            except Exception as e:
                _log(f'  profile stop failed: {e}')


def _kill_proc(proc):
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:
        pass


# ── Heartbeat / notifications ────────────────────────────────────────────────
def _fmt_elapsed(seconds):
    try:
        seconds = int(seconds)
    except Exception:
        return '?'
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f'{h}h{m:02d}m{s:02d}s'
    return f'{m}m{s:02d}s'


def _summarize_result(path):
    try:
        with open(path) as f:
            d = json.load(f)
        name = d.get('run_name', os.path.basename(path))
        t = d.get('type', '')
        if t == 'perf':
            tps = d.get('output_throughput')
            raw = d.get('raw') or {}
            ttft = (d.get('metrics', {}).get('ttft') or {}).get('mean') or raw.get('mean_ttft_ms')
            bits = []
            if tps is not None:
                bits.append(f'{tps:.0f} tok/s')
            if ttft is not None:
                bits.append(f'ttft={ttft:.0f}ms')
            return f'{name} perf: {", ".join(bits) or "(no metrics)"}'
        if t == 'quality':
            tasks = d.get('tasks', [])
            if tasks:
                parts = []
                for t0 in tasks[:3]:
                    met = t0.get('acc_norm', t0.get('acc', t0.get('exact_match')))
                    if met is not None:
                        parts.append(f'{t0.get("task")}={met:.3f}')
                return f'{name} quality: ' + (', '.join(parts) or '(no metrics)')
            return f'{name} quality: (no tasks)'
        if t == 'context_sweep':
            return f'{name} ctx sweep: max={d.get("max_context_tokens")}'
        return f'{name} ({t or "unknown"})'
    except Exception:
        return os.path.basename(path or '')


def _build_heartbeat_text(queue, job):
    i = (queue.get('current_job_index') or 0) + 1
    n = len(queue['jobs'])
    started = job.get('phase_started_at')
    phase_elapsed = (time.monotonic() - started) if started else 0
    lines = [
        f"[queue '{queue['name']}'] job {i}/{n}: {job['name']}",
        f'status: {job["status"]} | step: {job.get("current_step")} | phase elapsed: {_fmt_elapsed(phase_elapsed)}',
    ]
    recent = job.get('result_paths', [])[-3:]
    if recent:
        lines.append('recent results:')
        for p in recent:
            lines.append('  - ' + _summarize_result(p))
    done = sum(1 for j in queue['jobs'] if j['status'] == 'completed')
    failed = sum(1 for j in queue['jobs'] if j['status'] == 'failed')
    pending = sum(1 for j in queue['jobs'] if j['status'] == 'pending')
    lines.append(f'queue totals so far: {done} done, {failed} failed, {pending} pending')
    return '\n'.join(lines)


def _notify(subject, body):
    backends = notify_service.enabled_backends()
    if not backends:
        _log(f'(no notify backends enabled; skipping: {subject})')
        return
    results = notify_service.send(subject, body)
    for backend, (ok, err) in results.items():
        if backend not in backends:
            continue
        if ok:
            _log(f'notify {backend}: sent')
        else:
            _log(f'notify {backend}: FAILED ({err})')


async def _maybe_heartbeat():
    now = time.monotonic()
    if now - _state['last_heartbeat_at'] < HEARTBEAT_INTERVAL_S:
        return
    q = _state.get('queue')
    if not q or q.get('status') != 'running':
        return
    idx = q.get('current_job_index', -1)
    if idx < 0 or idx >= len(q['jobs']):
        return
    job = q['jobs'][idx]
    _state['last_heartbeat_at'] = now
    text = _build_heartbeat_text(q, job)
    asyncio.get_event_loop().run_in_executor(
        None, _notify, f'HPML Queue heartbeat: {q["name"]}', text,
    )


# ── Failure preset ───────────────────────────────────────────────────────────
import re
_FAILURE_SUFFIX_RE = re.compile(r'(?:__failures_\d{8}_\d{6})+$|(?:_retry_\d{8}_\d{6})+$')


def _strip_failure_suffixes(name):
    """Strip trailing failure/retry suffixes so failure-preset names don't nest."""
    if not name:
        return name
    prev = None
    while name != prev:
        prev = name
        name = _FAILURE_SUFFIX_RE.sub('', name)
    return name


def _generate_failure_preset(queue):
    """When a queue finishes with any failed jobs, drop a new preset containing
    only the failed (model, benchmark) combinations. Loads and runs through the
    normal preset UI."""
    failed = [j for j in queue.get('jobs', []) if j['status'] == 'failed']
    if not failed:
        return None
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    orig = _strip_failure_suffixes(queue.get('name') or 'queue')
    preset_name = f'{orig}__failures_{ts}'
    preset = new_queue(preset_name)
    preset['parent_queue'] = orig
    # Failed jobs are placed directly in `jobs` — the worker sees them as
    # pending and skips cartesian expansion (no models/benchmarks arrays needed).
    for j in failed:
        rj = dict(j)
        rj['id'] = str(uuid.uuid4())[:8]
        rj['status'] = 'pending'
        rj['error'] = None
        rj['started_at'] = None
        rj['finished_at'] = None
        rj['result_paths'] = []
        rj['port'] = None
        for k in ('phase_started_at', 'phase_soft_s', 'phase_hard_s',
                  'phase_idle_window_s', 'phase_last_activity_at'):
            rj[k] = None
        rj['phase_soft_crossed'] = False
        rj['current_step'] = None
        preset['jobs'].append(rj)
    path = os.path.join(PRESETS_DIR, f'{_safe(preset_name)}.json')
    with open(path, 'w') as f:
        json.dump(preset, f, indent=2)
    return path


# ── Main worker ──────────────────────────────────────────────────────────────
def expand_jobs(queue):
    """Return cartesian product of models × benchmarks as a list of jobs."""
    models = queue.get('models', []) or []
    benches = queue.get('benchmarks', []) or []
    jobs = []
    for m in models:
        for b in benches:
            jobs.append(_new_job(m, b))
    return jobs


async def _worker():
    q = _state['queue']

    # If the queue has inputs and no expanded jobs (typical first-run), expand now.
    if q.get('models') and q.get('benchmarks') and not any(
        j['status'] == 'pending' for j in q.get('jobs', [])
    ):
        q['jobs'] = expand_jobs(q)

    q['status'] = 'running'
    q['started_at'] = _now_iso()
    _persist()
    n_jobs = len(q['jobs'])
    _log(f"Queue '{q['name']}' started with {n_jobs} job(s)")

    asyncio.get_event_loop().run_in_executor(
        None, _notify,
        f"HPML Queue started: {q['name']}",
        f"Queue '{q['name']}' started with {n_jobs} jobs at {q['started_at']}.",
    )

    try:
        for i, job in enumerate(q['jobs']):
            if job['status'] not in ('pending',):
                continue
            if _state['cancel_requested']:
                raise QueueCancelled()
            q['current_job_index'] = i
            job['status'] = 'launching'
            job['started_at'] = _now_iso()
            _persist()
            _log(f"── Job {i+1}/{n_jobs}: {job['name']}")

            port = None
            try:
                port = await _launch_phase(job)
                job['status'] = 'running'
                _persist()
                await _bench_phase(job, port)
                job['status'] = 'completed'
                _log(f"✓ Job {i+1} completed")
            except QueueCancelled:
                job['status'] = 'cancelled'
                job['error'] = 'cancelled by user'
                raise
            except SoftTimeout as e:
                job['status'] = 'failed'
                job['error'] = f'soft timeout: {e}'
                _log(f'✗ Job {i+1} failed (soft timeout)')
                _notify_failure(q, job)
            except HardTimeout as e:
                job['status'] = 'failed'
                job['error'] = f'hard timeout: {e}'
                _log(f'✗ Job {i+1} failed (hard timeout)')
                _notify_failure(q, job)
            except Exception as e:
                job['status'] = 'failed'
                job['error'] = str(e)[:500]
                _log(f'✗ Job {i+1} failed: {e}')
                _notify_failure(q, job)
            finally:
                job['finished_at'] = _now_iso()
                job['current_step'] = None
                job['phase_started_at'] = None
                job['phase_soft_s'] = None
                job['phase_hard_s'] = None
                job['phase_idle_window_s'] = None
                job['phase_last_activity_at'] = None
                if port:
                    _log(f'  stopping server on port {port}')
                    try:
                        vllm_service.stop_server(port)
                    except Exception as ee:
                        _log(f'  (stop_server failed: {ee})')
                _persist()
    except QueueCancelled:
        q['status'] = 'cancelled'
        _log('Queue cancelled')
    except Exception as e:
        q['status'] = 'failed'
        _log(f'Queue worker crashed: {e}')
    else:
        has_failures = any(j['status'] == 'failed' for j in q['jobs'])
        q['status'] = 'completed_with_failures' if has_failures else 'completed'
    finally:
        q['finished_at'] = _now_iso()
        _persist()

        failure_preset_path = _generate_failure_preset(q)
        if failure_preset_path:
            _log(f'Failure preset saved: {os.path.basename(failure_preset_path)} (in Presets)')

        done = sum(1 for j in q['jobs'] if j['status'] == 'completed')
        failed = sum(1 for j in q['jobs'] if j['status'] == 'failed')
        cancelled = sum(1 for j in q['jobs'] if j['status'] == 'cancelled')
        body_lines = [
            f"Queue '{q['name']}' finished: {q['status']}.",
            f'Completed: {done} | Failed: {failed} | Cancelled: {cancelled}',
        ]
        if failure_preset_path:
            body_lines.append(f'Failure preset saved: {os.path.basename(failure_preset_path)} — load it from the Queue tab to re-run failed jobs.')
        for j in q['jobs']:
            if j['status'] == 'failed':
                body_lines.append(f'  ✗ {j["name"]}: {j["error"]}')
        asyncio.get_event_loop().run_in_executor(
            None, _notify,
            f"HPML Queue done: {q['name']} ({q['status']})",
            '\n'.join(body_lines),
        )

        _state['running'] = False
        _state['task'] = None
        _state['cancel_requested'] = False


def _notify_failure(queue, job):
    lines = [
        f"[queue '{queue['name']}'] FAILED: {job['name']}",
        f'step: {job.get("current_step")}',
        f'error: {job.get("error")}',
    ]
    tail = '\n'.join(_state['log_tail'][-20:])
    if tail:
        lines += ['', 'recent log:', tail]
    asyncio.get_event_loop().run_in_executor(
        None, _notify,
        f"HPML Queue FAIL: {queue['name']} - {job['name']}",
        '\n'.join(lines),
    )


# ── Public API ───────────────────────────────────────────────────────────────
def load_active():
    if not os.path.exists(ACTIVE_PATH):
        return None
    try:
        with open(ACTIVE_PATH) as f:
            q = json.load(f)
    except Exception:
        return None
    # Backfill schema fields for older files
    q.setdefault('models', [])
    q.setdefault('benchmarks', [])
    q.setdefault('jobs', [])
    for j in q['jobs']:
        for k in ('port', 'current_step', 'phase_started_at', 'phase_soft_s',
                  'phase_hard_s', 'phase_idle_window_s', 'phase_last_activity_at',
                  'error'):
            j.setdefault(k, None)
        j.setdefault('phase_soft_crossed', False)
        j.setdefault('result_paths', [])
        if j.get('status') in ('launching', 'running'):
            j['status'] = 'failed'
            j['error'] = 'app restart during execution'
            j['finished_at'] = _now_iso()
    return q


def set_queue(queue):
    if _state['running']:
        raise RuntimeError('cannot replace queue while worker is running')
    _state['queue'] = queue
    _persist()


def get_queue():
    return _state['queue']


def get_log_tail(n=100):
    return list(_state['log_tail'])[-n:]


def is_running():
    return _state['running']


def is_cancelling():
    return _state['cancel_requested']


def add_model(model, launch, timeouts=None, name=None):
    """Append a model+launch config to the current queue's models list.
    Auto-creates a default queue if none is loaded. Cannot be called while running."""
    if _state['running']:
        raise RuntimeError('cannot modify queue while running')
    q = _ensure_queue()
    q['models'].append(_new_model_entry(model, launch, timeouts=timeouts, name=name))
    _persist()


def add_benchmark(name, config_dict):
    """Append a benchmark config to the current queue's benchmarks list.
    Auto-creates a default queue if none is loaded. Cannot be called while running."""
    if _state['running']:
        raise RuntimeError('cannot modify queue while running')
    q = _ensure_queue()
    q['benchmarks'].append(_new_benchmark_entry(name, config_dict))
    _persist()


def remove_model(entry_id):
    if _state['running']:
        raise RuntimeError('cannot modify queue while running')
    q = _state.get('queue')
    if not q:
        return
    q['models'] = [m for m in q.get('models', []) if m['id'] != entry_id]
    _persist()


def remove_benchmark(entry_id):
    if _state['running']:
        raise RuntimeError('cannot modify queue while running')
    q = _state.get('queue')
    if not q:
        return
    q['benchmarks'] = [b for b in q.get('benchmarks', []) if b['id'] != entry_id]
    _persist()


def clear_models():
    if _state['running']:
        raise RuntimeError('cannot modify queue while running')
    q = _state.get('queue')
    if q:
        q['models'] = []
        _persist()


def clear_benchmarks():
    if _state['running']:
        raise RuntimeError('cannot modify queue while running')
    q = _state.get('queue')
    if q:
        q['benchmarks'] = []
        _persist()


def rename_queue(new_name):
    q = _state.get('queue')
    if q:
        q['name'] = new_name
        _persist()


def start():
    if _state['running']:
        return False
    q = _state.get('queue')
    if not q:
        raise RuntimeError('no queue loaded')
    # Two valid cases: (a) we have inputs to expand, or (b) we have pre-populated jobs (retry queue)
    has_inputs = bool(q.get('models') and q.get('benchmarks'))
    has_pending_jobs = any(j.get('status') == 'pending' for j in q.get('jobs', []))
    if not has_inputs and not has_pending_jobs:
        raise RuntimeError('queue has no models/benchmarks and no pending jobs')
    _state['running'] = True
    _state['cancel_requested'] = False
    _state['last_heartbeat_at'] = time.monotonic()
    loop = asyncio.get_event_loop()
    _state['task'] = loop.create_task(_worker())
    return True


def cancel():
    if not _state['running']:
        return False
    _state['cancel_requested'] = True
    _log('Cancel requested')
    return True


def force_reset_state():
    """Clear the running flag without running a worker. Use only when the
    in-memory state is wedged (worker died without resetting the flag, leaving
    start()/set_queue() permanently blocked)."""
    was_running = _state['running']
    _state['running'] = False
    _state['cancel_requested'] = False
    _state['task'] = None
    _log(f'State force-reset by user (was_running={was_running})')
    return was_running


# ── Presets ──────────────────────────────────────────────────────────────────
def _reset_queue_for_save(queue):
    data = dict(queue)
    data['status'] = 'idle'
    data['current_job_index'] = -1
    data['started_at'] = None
    data['finished_at'] = None
    # Presets capture models + benchmarks inputs; drop any expanded jobs.
    # (Retry queues re-use save_preset internally — they keep their jobs list.)
    data.setdefault('models', [])
    data.setdefault('benchmarks', [])
    data['jobs'] = []
    return data


def save_preset(name, queue):
    safe = _safe(name)
    path = os.path.join(PRESETS_DIR, f'{safe}.json')
    data = _reset_queue_for_save(queue)
    data['name'] = name
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    return path


def list_presets():
    out = []
    for p in sorted(glob.glob(os.path.join(PRESETS_DIR, '*.json'))):
        try:
            with open(p) as f:
                d = json.load(f)
            out.append({
                'name': d.get('name', os.path.splitext(os.path.basename(p))[0]),
                'path': p,
                'num_models': len(d.get('models', [])),
                'num_benchmarks': len(d.get('benchmarks', [])),
                'num_jobs': len(d.get('jobs', [])),
                'parent': d.get('parent_queue'),  # set only for failure presets
            })
        except Exception:
            continue
    return out


def load_from_path(path):
    with open(path) as f:
        data = json.load(f)
    data.setdefault('models', [])
    data.setdefault('benchmarks', [])
    data.setdefault('jobs', [])
    return data


def delete_preset(path):
    try:
        if os.path.isabs(path) and path.startswith(PRESETS_DIR):
            os.remove(path)
            return True
    except Exception:
        pass
    return False


# ── Bootstrap ────────────────────────────────────────────────────────────────
_TERMINAL_STATUSES = ('completed', 'completed_with_failures', 'failed', 'cancelled')


def bootstrap_load_state():
    """On app start: if the last queue was completed/failed/cancelled, drop it
    so the UI starts clean. Otherwise load it (with any accumulated failure
    suffixes stripped from the name). Worker is NOT auto-started."""
    q = load_active()
    if not q:
        return
    n_pending = sum(1 for j in q.get('jobs', []) if j['status'] == 'pending')
    if q.get('status') in _TERMINAL_STATUSES and not n_pending:
        try:
            if os.path.exists(ACTIVE_PATH):
                os.remove(ACTIVE_PATH)
        except Exception:
            pass
        return
    # Sanitize any nested retry/failure suffixes that accumulated historically
    q['name'] = _strip_failure_suffixes(q.get('name') or 'queue')
    _state['queue'] = q
    n_models = len(q.get('models', []))
    n_benches = len(q.get('benchmarks', []))
    _log(f'Loaded last queue on startup: {q.get("name")} ({n_models} models, {n_benches} benchmarks, {n_pending} pending jobs)')


__all__ = [
    'new_queue', 'set_queue', 'get_queue', 'get_log_tail', 'is_running',
    'is_cancelling', 'start', 'cancel', 'force_reset_state',
    'save_preset', 'list_presets', 'load_from_path', 'delete_preset',
    'add_model', 'add_benchmark', 'remove_model', 'remove_benchmark',
    'clear_models', 'clear_benchmarks', 'rename_queue', 'expand_jobs',
    'bootstrap_load_state', 'load_active',
    'DEFAULT_TIMEOUTS',
]
