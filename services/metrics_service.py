import subprocess
import requests
import csv
import time
import threading
import os
from datetime import datetime
from pathlib import Path

METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)

_recorders = {}


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

        if name == "vllm:kv_cache_usage_perc":
            metrics["kv_cache_pct"] = val * 100
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

    return metrics


CSV_COLUMNS = [
    "timestamp", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_util_pct",
    "gpu_temp_c", "gpu_power_w", "kv_cache_pct", "requests_running",
    "requests_waiting", "gen_tokens_total", "prompt_tokens_total",
    "preemptions", "prefix_cache_hit_rate", "ttft_avg_ms", "itl_avg_ms",
    "e2e_avg_ms",
]


def start_recording(port, model_name):
    """Start recording metrics to CSV in a background thread."""
    if port in _recorders:
        return
    safe_name = model_name.replace("/", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = METRICS_DIR / f"{safe_name}_{ts}.csv"

    stop_event = threading.Event()

    def record_loop():
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            while not stop_event.is_set():
                gpu = fetch_gpu_metrics()
                vllm = fetch_vllm_metrics(port)
                row = {"timestamp": datetime.now().isoformat()}
                for col in CSV_COLUMNS[1:]:
                    row[col] = gpu.get(col, vllm.get(col, 0))
                writer.writerow(row)
                f.flush()
                stop_event.wait(2)

    thread = threading.Thread(target=record_loop, daemon=True)
    thread.start()
    _recorders[port] = {"thread": thread, "stop": stop_event, "path": str(csv_path)}


def stop_recording(port):
    """Stop recording metrics for a port."""
    rec = _recorders.pop(port, None)
    if rec:
        rec["stop"].set()


def is_recording(port):
    return port in _recorders
