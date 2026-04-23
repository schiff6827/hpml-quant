import subprocess
import requests
import csv
import time
import threading
import os
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

METRICS_DIR = Path("metrics")

_recorders = {}
_run_recorders = {}


def fetch_gpu_metrics():
    """Get GPU metrics from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "gpu_mem_used_mb": float(parts[0]),
            "gpu_mem_total_mb": float(parts[1]),
            "gpu_mem_free_mb": float(parts[2]),
            "gpu_util_pct": float(parts[3]),
            "gpu_temp_c": float(parts[4]),
            "gpu_power_w": float(parts[5]),
        }
    except Exception:
        return {}


def fetch_cpu_metrics(port=None):
    """Get CPU memory metrics for a vLLM process tree plus host-level memory.

    If psutil is unavailable or the port is unknown, returns host metrics only
    (or empty dict on total failure).
    """
    metrics = {}
    if psutil is None:
        return metrics
    try:
        vm = psutil.virtual_memory()
        metrics["host_mem_used_mb"] = (vm.total - vm.available) / (1024 * 1024)
        metrics["host_mem_total_mb"] = vm.total / (1024 * 1024)
    except Exception:
        pass

    if port is None:
        return metrics
    try:
        from services import vllm_service
        info = vllm_service._running.get(port)
        if not info:
            return metrics
        proc = info.get("proc")
        pid = getattr(proc, "pid", None)
        if not pid:
            return metrics
        root = psutil.Process(pid)
        procs = [root] + root.children(recursive=True)
        rss = 0
        vms = 0
        for p in procs:
            try:
                mi = p.memory_info()
                rss += mi.rss
                vms += mi.vms
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        metrics["cpu_mem_rss_mb"] = rss / (1024 * 1024)
        metrics["cpu_mem_vms_mb"] = vms / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
        pass
    return metrics


def fetch_vllm_metrics(port):
    """Fetch and parse Prometheus metrics from a vLLM server."""
    try:
        resp = requests.get(f"http://localhost:{port}/metrics", timeout=3)
        if resp.status_code != 200:
            return {}
    except Exception:
        return {}

    metrics = {}
    prev_gen_tokens = getattr(fetch_vllm_metrics, '_prev_gen_tokens', {})
    prev_time = getattr(fetch_vllm_metrics, '_prev_time', {})

    for line in resp.text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        try:
            name_labels, value = line.rsplit(" ", 1)
            val = float(value)
        except (ValueError, IndexError):
            continue

        name = name_labels.split("{")[0]

        if name == "vllm:kv_cache_usage_perc" or name == "vllm:gpu_cache_usage_perc":
            metrics["kv_cache_pct"] = val * 100
        elif name == "vllm:num_tokens_running":
            metrics["tokens_in_flight"] = val
        elif name == "vllm:num_requests_running":
            metrics["requests_running"] = val
        elif name == "vllm:num_requests_waiting":
            metrics["requests_waiting"] = val
        elif name == "vllm:generation_tokens_total":
            now = time.time()
            prev = prev_gen_tokens.get(port, 0)
            prev_t = prev_time.get(port, now)
            dt = now - prev_t
            if dt > 0 and prev > 0:
                metrics["gen_tokens_per_sec"] = (val - prev) / dt
            else:
                metrics["gen_tokens_per_sec"] = 0
            prev_gen_tokens[port] = val
            prev_time[port] = now
            metrics["gen_tokens_total"] = val
        elif name == "vllm:prompt_tokens_total":
            metrics["prompt_tokens_total"] = val
        elif name == "vllm:num_preemptions_total":
            metrics["preemptions"] = val
        elif name == "vllm:prefix_cache_hits_total":
            metrics["prefix_cache_hits"] = val
        elif name == "vllm:prefix_cache_queries_total":
            metrics["prefix_cache_queries"] = val
        elif name == "vllm:time_to_first_token_seconds_sum":
            metrics["ttft_sum"] = val
        elif name == "vllm:time_to_first_token_seconds_count":
            metrics["ttft_count"] = val
        elif name == "vllm:inter_token_latency_seconds_sum":
            metrics["itl_sum"] = val
        elif name == "vllm:inter_token_latency_seconds_count":
            metrics["itl_count"] = val
        elif name == "vllm:e2e_request_latency_seconds_sum":
            metrics["e2e_sum"] = val
        elif name == "vllm:e2e_request_latency_seconds_count":
            metrics["e2e_count"] = val

    fetch_vllm_metrics._prev_gen_tokens = prev_gen_tokens
    fetch_vllm_metrics._prev_time = prev_time

    hits = metrics.get("prefix_cache_hits", 0)
    queries = metrics.get("prefix_cache_queries", 0)
    metrics["prefix_cache_hit_rate"] = (hits / queries * 100) if queries > 0 else 0

    ttft_count = metrics.get("ttft_count", 0)
    metrics["ttft_avg_ms"] = (metrics.get("ttft_sum", 0) / ttft_count * 1000) if ttft_count > 0 else 0
    itl_count = metrics.get("itl_count", 0)
    metrics["itl_avg_ms"] = (metrics.get("itl_sum", 0) / itl_count * 1000) if itl_count > 0 else 0
    e2e_count = metrics.get("e2e_count", 0)
    metrics["e2e_avg_ms"] = (metrics.get("e2e_sum", 0) / e2e_count * 1000) if e2e_count > 0 else 0

    # Memory derivations: KV-used in absolute GiB and ratio vs weights.
    try:
        from services import vllm_service
        info = vllm_service._running.get(port) or {}
    except Exception:
        info = {}
    weight_gib = info.get("weight_mem_gib", 0) or 0
    kv_capacity_gib = info.get("kv_cache_capacity_gib", 0) or 0
    metrics["weight_mem_gib"] = weight_gib
    metrics["kv_cache_capacity_gib"] = kv_capacity_gib
    kv_pct = metrics.get("kv_cache_pct", 0)
    kv_used_gib = kv_pct / 100.0 * kv_capacity_gib if kv_capacity_gib else 0
    metrics["kv_mem_used_gib"] = kv_used_gib
    metrics["kv_to_weight_ratio"] = (kv_used_gib / weight_gib) if weight_gib else 0
    tokens_live = metrics.get("tokens_in_flight", 0) or 0
    if tokens_live > 0 and kv_used_gib > 0:
        metrics["kv_bytes_per_token"] = kv_used_gib * (1024 ** 3) / tokens_live
    else:
        metrics["kv_bytes_per_token"] = 0

    return metrics


CSV_COLUMNS = [
    "timestamp", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_util_pct",
    "gpu_temp_c", "gpu_power_w", "kv_cache_pct", "requests_running",
    "requests_waiting", "gen_tokens_total", "prompt_tokens_total",
    "preemptions", "prefix_cache_hit_rate", "ttft_avg_ms", "itl_avg_ms",
    "e2e_avg_ms",
    "cpu_mem_rss_mb", "cpu_mem_rss_peak_mb", "cpu_mem_vms_mb",
    "host_mem_used_mb", "host_mem_total_mb",
    "weight_mem_gib", "kv_cache_capacity_gib", "kv_mem_used_gib",
    "kv_to_weight_ratio", "kv_bytes_per_token", "tokens_in_flight",
]


def start_recording(port, model_name):
    """Start recording metrics to CSV in a background thread."""
    if port in _recorders:
        return
    METRICS_DIR.mkdir(exist_ok=True)
    safe_name = model_name.replace("/", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = METRICS_DIR / f"{safe_name}_{ts}.csv"

    stop_event = threading.Event()
    peak_state = {"rss_peak_mb": 0.0}

    def record_loop():
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            while not stop_event.is_set():
                gpu = fetch_gpu_metrics()
                vllm = fetch_vllm_metrics(port)
                cpu = fetch_cpu_metrics(port)
                rss = cpu.get("cpu_mem_rss_mb", 0)
                if rss > peak_state["rss_peak_mb"]:
                    peak_state["rss_peak_mb"] = rss
                cpu["cpu_mem_rss_peak_mb"] = peak_state["rss_peak_mb"]
                row = {"timestamp": datetime.now().isoformat()}
                for col in CSV_COLUMNS[1:]:
                    row[col] = gpu.get(col, vllm.get(col, cpu.get(col, 0)))
                writer.writerow(row)
                f.flush()
                stop_event.wait(2)

    thread = threading.Thread(target=record_loop, daemon=True)
    thread.start()
    _recorders[port] = {
        "thread": thread, "stop": stop_event, "path": str(csv_path),
        "peak_state": peak_state,
    }


def get_peak_rss_mb(port):
    """Return current peak CPU RSS in MB for a recording, or 0 if not recording."""
    rec = _recorders.get(port)
    if not rec:
        return 0.0
    return rec.get("peak_state", {}).get("rss_peak_mb", 0.0)


def stop_recording(port):
    """Stop recording metrics for a port."""
    rec = _recorders.pop(port, None)
    if rec:
        rec["stop"].set()


def is_recording(port):
    return port in _recorders


def _safe_filename(name):
    return ''.join(c for c in name if c.isalnum() or c in '-_.') or 'run'


def start_run_recording(port, run_name, interval=1.0):
    """Record metrics for a single benchmark run. Keyed by run_name so it can
    coexist with the Monitor tab's port-keyed recorder.

    Returns the CSV path, or the existing path if already recording under
    this run_name.
    """
    if run_name in _run_recorders:
        return _run_recorders[run_name]["path"]
    METRICS_DIR.mkdir(exist_ok=True)
    safe = _safe_filename(run_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = METRICS_DIR / f"run_{safe}_{ts}.csv"

    stop_event = threading.Event()
    peak_state = {
        "rss_peak_mb": 0.0,
        "gpu_mem_peak_mb": 0.0,
        "gpu_power_peak_w": 0.0,
    }
    latest = {}

    def record_loop():
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            while not stop_event.is_set():
                gpu = fetch_gpu_metrics()
                vllm = fetch_vllm_metrics(port)
                cpu = fetch_cpu_metrics(port)
                rss = cpu.get("cpu_mem_rss_mb", 0)
                if rss > peak_state["rss_peak_mb"]:
                    peak_state["rss_peak_mb"] = rss
                cpu["cpu_mem_rss_peak_mb"] = peak_state["rss_peak_mb"]
                gmu = gpu.get("gpu_mem_used_mb", 0) or 0
                if gmu > peak_state["gpu_mem_peak_mb"]:
                    peak_state["gpu_mem_peak_mb"] = gmu
                gpw = gpu.get("gpu_power_w", 0) or 0
                if gpw > peak_state["gpu_power_peak_w"]:
                    peak_state["gpu_power_peak_w"] = gpw
                row = {"timestamp": datetime.now().isoformat()}
                for col in CSV_COLUMNS[1:]:
                    row[col] = gpu.get(col, vllm.get(col, cpu.get(col, 0)))
                writer.writerow(row)
                f.flush()
                latest.clear()
                latest.update(row)
                latest["gpu_mem_total_mb"] = gpu.get("gpu_mem_total_mb", 0)
                stop_event.wait(interval)

    thread = threading.Thread(target=record_loop, daemon=True)
    thread.start()
    _run_recorders[run_name] = {
        "thread": thread, "stop": stop_event, "path": str(csv_path),
        "peak_state": peak_state, "latest": latest, "port": port,
        "started": time.time(),
    }
    return str(csv_path)


def stop_run_recording(run_name):
    """Stop a run-scoped recorder. Returns the CSV path, or None."""
    rec = _run_recorders.pop(run_name, None)
    if not rec:
        return None
    rec["stop"].set()
    rec["thread"].join(timeout=5)
    return rec["path"]


def get_run_latest(run_name):
    """Return the most recent sampled row for a running recorder, or None."""
    rec = _run_recorders.get(run_name)
    if not rec:
        return None
    return dict(rec.get("latest", {}))


def get_run_peaks(run_name):
    rec = _run_recorders.get(run_name)
    if not rec:
        return {}
    return dict(rec.get("peak_state", {}))


def is_run_recording(run_name):
    return run_name in _run_recorders
