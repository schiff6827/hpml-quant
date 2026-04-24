"""Queue service for serial benchmark runs.

- Job = (model, launch config, list of benchmark scripts, timeouts)
- Queue = ordered list of jobs with persistence
- Worker = single asyncio task; one job at a time; one benchmark inside a job at a time
- Soft cap + idle window + hard cap per phase; GPU + process CPU used as activity signals
- Heartbeats every 30 min; immediate notify on failure/done
- Restart recovery: active.json is the source of truth; running-at-crash = failed
- Retry queue = new queue containing only failed jobs, saved for user review
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
from services import benchmark_service, notify_service, vllm_service

QUEUE_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'benchmarks', 'queue')
ACTIVE_PATH = os.path.join(QUEUE_ROOT, 'active.json')
PRESETS_DIR = os.path.join(QUEUE_ROOT, 'presets')
RETRIES_DIR = os.path.join(QUEUE_ROOT, 'retries')
LOGS_DIR = os.path.join(QUEUE_ROOT, 'logs')

for d in (QUEUE_ROOT, PRESETS_DIR, RETRIES_DIR, LOGS_DIR):
    os.makedirs(d, exist_ok=True)

HEARTBEAT_INTERVAL_S = 30 * 60
GPU_ACTIVITY_THRESHOLD = 5.0   # percent util
CPU_ACTIVITY_THRESHOLD = 1.0   # percent
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
    'queue': None,       # dict: queue object currently loaded
    'running': False,    # bool: worker task alive
    'task': None,        # asyncio.Task or None
    'cancel_requested': False,
    'log_tail': [],      # recent log lines for UI display (capped)
    'last_heartbeat_at': 0,
    'last_retry_path': None,  # path to last generated retry queue, for UI banner
}
_LOG_TAIL_MAX = 300


# ── Helpers ──────────────────────────────────────────────────────────────────
def _now_iso():
    return datetime.now().isoformat(timespec='seconds')


def _new_job(model, launch, benchmarks, timeouts=None, name=None):
    return {
        'id': str(uuid.uuid4())[:8],
        'name': name or model,
        'model': model,
        'launch': dict(launch),
        'benchmarks': list(benchmarks),   # list of saved script names
        'timeouts': {**DEFAULT_TIMEOUTS, **(timeouts or {})},
        'status': 'pending',              # pending|launching|running|completed|failed|cancelled
        'error': None,
        'started_at': None,
        'finished_at': None,
        'result_paths': [],
        'port': None,
        'current_step': None,             # 'launch' or 'bench:<script>' or None
        'phase_started_at': None,         # monotonic (reset every phase)
        'phase_soft_s': None,
        'phase_hard_s': None,
        'phase_idle_window_s': None,
        'phase_last_activity_at': None,
        'phase_soft_crossed': False,
    }


def new_queue(name, jobs=None):
    return {
        'name': name,
        'created': _now_iso(),
        'jobs': list(jobs or []),
        'status': 'idle',                 # idle|running|completed|failed|cancelled
        'current_job_index': -1,
        'started_at': None,
        'finished_at': None,
    }


def _persist():
    q = _state['queue']
    if not q:
        if os.path.exists(ACTIVE_PATH):
            os.remove(ACTIVE_PATH)
        return
    # Don't serialize runtime-only monotonic timestamps
    serial = {
        **q,
        'jobs': [
            {k: v for k, v in j.items() if not k.startswith('phase_')}
            for j in q['jobs']
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
    # Append to per-queue file if we have a queue
    q = _state.get('queue')
    if q:
        path = os.path.join(LOGS_DIR, f'{_safe(q["name"])}.log')
        try:
            with open(path, 'a') as f:
                f.write(formatted + '\n')
        except Exception:
            pass


def _safe(name):
    return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in (name or '').strip()) or 'queue'


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
    """Windowed %CPU for a PID. Returns percent-of-one-core since last sample.
    sum_over_cores so 100+ is possible on multi-threaded processes."""
    _HZ = os.sysconf('SC_CLK_TCK') if hasattr(os, 'sysconf') else 100

    def __init__(self, pid):
        self.pid = pid
        self.last_ticks = None
        self.last_wall = None

    def sample(self):
        try:
            with open(f'/proc/{self.pid}/stat') as f:
                parts = f.read().split()
            utime = int(parts[13])
            stime = int(parts[14])
            total = utime + stime
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


# ── Phase runner (soft/hard cap + idle detection) ────────────────────────────
async def _monitor_phase(job, phase_label, condition_fn, is_active_fn, soft_s, hard_s, idle_window_s):
    """Loop until condition_fn() returns True or a timeout fires.
    Updates job phase counters once per second for the UI.
    Raises SoftTimeout or HardTimeout on cap.
    """
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

        active = is_active_fn()
        if active:
            last_active = now
            job['phase_last_activity_at'] = now

        if elapsed >= soft_s:
            job['phase_soft_crossed'] = True
            idle_dur = now - last_active
            if idle_dur >= idle_window_s:
                raise SoftTimeout(
                    f'{phase_label}: soft cap {soft_s}s + idle {idle_window_s}s reached'
                )

        await _maybe_heartbeat()
        await asyncio.sleep(HEALTH_POLL_INTERVAL_S)


# ── Launch phase ─────────────────────────────────────────────────────────────
def _resolve_model_path(model_id):
    """Mirror servers.py: local-model entries store the real filesystem path."""
    # We don't import hf_service here to keep this lean — launch_server handles
    # the path/id distinction internally via HF_HUB_CACHE.
    if os.path.isdir(model_id):
        return model_id
    return model_id


async def _launch_phase(job):
    launch = job['launch']
    use_kv_gb = launch.get('use_kv_gb', True)
    extra = ['--trust-remote-code'] if launch.get('trust_remote_code') else None
    cpu_offload = launch.get('cpu_offload_gb') or None

    _log(f"Launching vLLM for '{job['model']}'")
    kwargs = dict(
        model=_resolve_model_path(job['model']),
        port=None,
        gpu_mem_util=launch.get('gpu_mem_util', config.DEFAULT_GPU_MEM_UTIL),
        dtype=launch.get('dtype', config.DEFAULT_DTYPE),
        quantization=launch.get('quantization') or None,
        extra_args=extra,
        kv_cache_gb=int(launch.get('kv_cache_gb', 10)) if use_kv_gb else None,
    )
    # vllm_service.launch_server may or may not accept cpu_offload_gb on this branch.
    try:
        port = vllm_service.launch_server(cpu_offload_gb=cpu_offload, **kwargs)
    except TypeError:
        port = vllm_service.launch_server(**kwargs)
    job['port'] = port
    _log(f'  → port {port}')

    info = vllm_service.get_server_info(port) or {}
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
        job=job,
        phase_label='launch',
        condition_fn=healthy,
        is_active_fn=is_active,
        soft_s=job['timeouts']['launch_soft_s'],
        hard_s=job['timeouts']['launch_hard_s'],
        idle_window_s=job['timeouts']['launch_idle_window_s'],
    )
    _log(f'  → healthy on {port}')
    return port


# ── Benchmark phase ──────────────────────────────────────────────────────────
def _run_one_subprocess(bench_type, script_cfg, port, model, run_name, result_dir):
    """Start the appropriate benchmark subprocess. Returns (proc, parser_fn)."""
    os.makedirs(result_dir, exist_ok=True)
    if bench_type == 'perf':
        perf = script_cfg.get('perf', {})
        proc = benchmark_service.run_perf_benchmark(
            port=port,
            model=model,
            dataset=perf.get('dataset', 'random'),
            num_prompts=int(perf.get('num_prompts', 100)),
            request_rate=float(perf.get('request_rate', 0)),
            max_concurrency=int(perf.get('max_concurrency', 1)),
            random_input_len=int(perf.get('random_input_len', 1024)),
            random_output_len=int(perf.get('random_output_len', 128)),
            result_dir=result_dir,
            run_name=run_name,
        )
        return proc, lambda: benchmark_service.parse_perf_result(result_dir, run_name)
    if bench_type == 'quality':
        qual = script_cfg.get('quality', {})
        proc = benchmark_service.run_quality_benchmark(
            port=port,
            model=model,
            tasks=qual.get('tasks', []),
            num_fewshot=int(qual.get('num_fewshot', 0)),
            num_concurrent=int(qual.get('num_concurrent', 1)),
            limit=int(qual.get('limit', 0) or 0),
            result_dir=result_dir,
            run_name=run_name,
        )
        return proc, lambda: benchmark_service.parse_quality_result(result_dir, run_name)
    if bench_type == 'context_sweep':
        ctx = script_cfg.get('context_sweep', {})
        proc = benchmark_service.run_context_sweep(
            port=port,
            model=model,
            result_dir=result_dir,
            run_name=run_name,
            upper_bound=int(ctx.get('upper_bound', 32768)),
            step=int(ctx.get('step', 1024)),
        )
        return proc, lambda: benchmark_service.parse_context_sweep_result(result_dir, run_name)
    raise ValueError(f'unknown bench type: {bench_type}')


def _enabled_bench_types(script_cfg):
    out = []
    for t in ('perf', 'quality', 'context_sweep'):
        if script_cfg.get(t, {}).get('enabled'):
            out.append(t)
    return out


async def _bench_phase(job, port, script_name):
    script_cfg = benchmark_service.load_script(script_name)
    if not script_cfg:
        raise RuntimeError(f"benchmark script '{script_name}' not found")

    bench_types = _enabled_bench_types(script_cfg)
    if not bench_types:
        raise RuntimeError(f"script '{script_name}' has no enabled benchmark types")

    model_meta = benchmark_service.get_model_metadata(job['model'], port=port)

    for bench_type in bench_types:
        run_name = f'{_safe(job["name"])}_{_safe(script_name)}'
        result_dir = os.path.join('/tmp', f'queue_{job["id"]}_{_safe(script_name)}_{bench_type}')
        _log(f"Starting {bench_type} benchmark (script='{script_name}')")

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
                phase_label=f'bench:{bench_type}:{script_name}',
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

        parsed = parser()
        if parsed:
            parsed['model_meta'] = model_meta
            # Rewrite the parsed file to include model_meta
            save_path = parsed.get('save_path')
            if save_path and os.path.exists(save_path):
                try:
                    with open(save_path, 'w') as f:
                        json.dump(parsed, f, indent=2)
                except Exception:
                    pass
                job['result_paths'].append(save_path)
            _log(f'  → {bench_type} done: {os.path.basename(save_path or "?")}')
        else:
            _log(f'  → {bench_type} finished but no result file parsed')


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
        f"[queue '{queue['name']}'] job {i}/{n}: {job['model']}",
        f'status: {job["status"]} | step: {job.get("current_step")} | phase elapsed: {_fmt_elapsed(phase_elapsed)}',
    ]
    recent = job.get('result_paths', [])[-3:]
    if recent:
        lines.append('recent results:')
        for p in recent:
            lines.append('  - ' + _summarize_result(p))
    # Any prior completed jobs this run
    done = [j for j in queue['jobs'] if j['status'] == 'completed']
    failed = [j for j in queue['jobs'] if j['status'] == 'failed']
    lines.append(f'queue totals so far: {len(done)} done, {len(failed)} failed, {sum(1 for j in queue["jobs"] if j["status"] == "pending")} pending')
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
    # Fire-and-forget in a thread to avoid blocking the event loop on SMTP/HTTP
    asyncio.get_event_loop().run_in_executor(None, _notify, f'HPML Queue heartbeat: {q["name"]}', text)


# ── Retry queue generation ───────────────────────────────────────────────────
def _generate_retry_queue(queue):
    failed = [j for j in queue['jobs'] if j['status'] == 'failed']
    if not failed:
        return None
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    orig = queue.get('name') or 'queue'
    retry = new_queue(f'{orig}_retry_{ts}')
    retry['parent_queue'] = orig
    for j in failed:
        rj = dict(j)
        rj['id'] = str(uuid.uuid4())[:8]
        rj['status'] = 'pending'
        rj['error'] = None
        rj['started_at'] = None
        rj['finished_at'] = None
        rj['result_paths'] = []
        rj['current_step'] = None
        rj['port'] = None
        for k in ('phase_started_at', 'phase_soft_s', 'phase_hard_s',
                  'phase_idle_window_s', 'phase_last_activity_at', 'phase_soft_crossed'):
            rj[k] = None if k != 'phase_soft_crossed' else False
        retry['jobs'].append(rj)
    path = os.path.join(RETRIES_DIR, f'{_safe(orig)}_retry_{ts}.json')
    with open(path, 'w') as f:
        json.dump(retry, f, indent=2)
    return path


# ── Main worker ──────────────────────────────────────────────────────────────
async def _worker():
    q = _state['queue']
    q['status'] = 'running'
    q['started_at'] = _now_iso()
    _persist()
    _log(f"Queue '{q['name']}' started with {len(q['jobs'])} job(s)")

    # Initial start notification
    asyncio.get_event_loop().run_in_executor(
        None, _notify,
        f"HPML Queue started: {q['name']}",
        f"Queue '{q['name']}' started with {len(q['jobs'])} jobs at {q['started_at']}.",
    )

    try:
        for i, job in enumerate(q['jobs']):
            if job['status'] not in ('pending',):
                continue  # skip already-done / already-failed on resume
            if _state['cancel_requested']:
                raise QueueCancelled()
            q['current_job_index'] = i
            job['status'] = 'launching'
            job['started_at'] = _now_iso()
            _persist()
            _log(f"── Job {i+1}/{len(q['jobs'])}: {job['model']}")

            port = None
            try:
                port = await _launch_phase(job)
                job['status'] = 'running'
                _persist()
                for script_name in job['benchmarks']:
                    if _state['cancel_requested']:
                        raise QueueCancelled()
                    await _bench_phase(job, port, script_name)
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

        # Retry queue generation
        retry_path = _generate_retry_queue(q)
        if retry_path:
            _state['last_retry_path'] = retry_path
            _log(f'Retry queue saved: {retry_path}')

        # Completion notification
        done = sum(1 for j in q['jobs'] if j['status'] == 'completed')
        failed = sum(1 for j in q['jobs'] if j['status'] == 'failed')
        cancelled = sum(1 for j in q['jobs'] if j['status'] == 'cancelled')
        body_lines = [
            f"Queue '{q['name']}' finished: {q['status']}.",
            f'Completed: {done} | Failed: {failed} | Cancelled: {cancelled}',
        ]
        if retry_path:
            body_lines.append(f'Retry queue saved: {os.path.basename(retry_path)}')
        for j in q['jobs']:
            if j['status'] == 'failed':
                body_lines.append(f'  ✗ {j["model"]}: {j["error"]}')
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
        f"[queue '{queue['name']}'] FAILED: {job['model']}",
        f'step: {job.get("current_step")}',
        f'error: {job.get("error")}',
    ]
    tail = '\n'.join(_state['log_tail'][-20:])
    if tail:
        lines += ['', 'recent log:', tail]
    asyncio.get_event_loop().run_in_executor(
        None, _notify,
        f"HPML Queue FAIL: {queue['name']} - {job['model']}",
        '\n'.join(lines),
    )


# ── Public API ───────────────────────────────────────────────────────────────
def load_active():
    """Load active.json if it exists. Jobs that were running at crash marked failed."""
    if not os.path.exists(ACTIVE_PATH):
        return None
    try:
        with open(ACTIVE_PATH) as f:
            q = json.load(f)
    except Exception:
        return None
    for j in q.get('jobs', []):
        # Backfill any missing phase fields
        for k in ('port', 'current_step', 'phase_started_at', 'phase_soft_s',
                  'phase_hard_s', 'phase_idle_window_s', 'phase_last_activity_at',
                  'phase_soft_crossed', 'result_paths', 'error'):
            if k not in j:
                j[k] = None if k != 'phase_soft_crossed' and k != 'result_paths' else (False if k == 'phase_soft_crossed' else [])
        if j.get('status') in ('launching', 'running'):
            j['status'] = 'failed'
            j['error'] = 'app restart during execution'
            j['finished_at'] = _now_iso()
    return q


def set_queue(queue):
    """Replace the active queue. Cannot be called while running."""
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


def start():
    """Spawn the worker task on the current event loop. Returns True if started."""
    if _state['running']:
        return False
    if not _state.get('queue'):
        raise RuntimeError('no queue loaded')
    if not any(j['status'] == 'pending' for j in _state['queue']['jobs']):
        raise RuntimeError('no pending jobs in queue')
    _state['running'] = True
    _state['cancel_requested'] = False
    _state['last_heartbeat_at'] = time.monotonic()  # don't fire immediately on start
    loop = asyncio.get_event_loop()
    _state['task'] = loop.create_task(_worker())
    return True


def cancel():
    """Request cancellation; worker will stop after current subprocess step."""
    if not _state['running']:
        return False
    _state['cancel_requested'] = True
    _log('Cancel requested')
    return True


# ── Presets ──────────────────────────────────────────────────────────────────
def save_preset(name, queue):
    safe = _safe(name)
    path = os.path.join(PRESETS_DIR, f'{safe}.json')
    data = dict(queue)
    data['name'] = name
    # Reset runtime-y fields; presets shouldn't carry status
    data['status'] = 'idle'
    data['current_job_index'] = -1
    data['started_at'] = None
    data['finished_at'] = None
    for j in data.get('jobs', []):
        j['status'] = 'pending'
        j['error'] = None
        j['started_at'] = None
        j['finished_at'] = None
        j['result_paths'] = []
        j['port'] = None
        for k in list(j.keys()):
            if k.startswith('phase_'):
                j[k] = None if k != 'phase_soft_crossed' else False
        j['current_step'] = None
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    return path


def list_presets():
    out = []
    for p in sorted(glob.glob(os.path.join(PRESETS_DIR, '*.json'))):
        try:
            with open(p) as f:
                d = json.load(f)
            out.append({'name': d.get('name', os.path.splitext(os.path.basename(p))[0]),
                        'path': p, 'num_jobs': len(d.get('jobs', []))})
        except Exception:
            continue
    return out


def list_retries():
    out = []
    for p in sorted(glob.glob(os.path.join(RETRIES_DIR, '*.json')), reverse=True):
        try:
            with open(p) as f:
                d = json.load(f)
            out.append({'name': d.get('name', os.path.basename(p)),
                        'path': p, 'num_jobs': len(d.get('jobs', [])),
                        'parent': d.get('parent_queue')})
        except Exception:
            continue
    return out


def load_from_path(path):
    with open(path) as f:
        return json.load(f)


def delete_preset(path):
    try:
        if os.path.isabs(path) and path.startswith(PRESETS_DIR):
            os.remove(path)
            return True
    except Exception:
        pass
    return False


# ── Bootstrap (called from app startup) ──────────────────────────────────────
def bootstrap_autoresume():
    """Sync bootstrap: load active.json into memory so the UI shows it.
    Does NOT start the worker — call maybe_resume_worker() from an async context
    (e.g. app.on_startup) to actually kick things off."""
    q = load_active()
    if not q:
        return
    _state['queue'] = q
    pending = [j for j in q['jobs'] if j['status'] == 'pending']
    _log(f'Loaded active queue on startup: {q.get("name")} ({len(pending)} pending)')


def maybe_resume_worker():
    """Async-safe: start the worker if the active queue has pending jobs.
    Intended to be called from app.on_startup() after the event loop is up."""
    if _state['running']:
        return False
    q = _state.get('queue')
    if not q:
        return False
    if not any(j['status'] == 'pending' for j in q['jobs']):
        return False
    was_running = q.get('status') == 'running'
    prefix = 'Auto-resuming' if was_running else 'Starting loaded queue'
    _log(f'{prefix}: {q["name"]}')
    return start()


__all__ = [
    'new_queue', 'set_queue', 'get_queue', 'get_log_tail', 'is_running',
    'is_cancelling', 'start', 'cancel', 'save_preset', 'list_presets',
    'list_retries', 'load_from_path', 'delete_preset', 'bootstrap_autoresume',
    'maybe_resume_worker', 'load_active', 'DEFAULT_TIMEOUTS', '_new_job',
]
