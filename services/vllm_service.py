import subprocess
import signal
import sys
import os
import re
import getpass
import requests
import config

# port -> {proc, model, port, gpu_mem_util, dtype, quantization, log_path}
_running = {}


# Project-relative so reports survive reboots and the 30-day systemd-tmpfiles
# sweep on /tmp. Large binary artifacts — keep this path in .gitignore.
NSYS_DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'benchmarks', 'nsys')


def launch_server(model, port=None, gpu_mem_util=None, dtype=None, quantization=None, extra_args=None, token=None, kv_cache_gb=None, nsys_profile=False, nsys_output_dir=None):
    if port is None:
        port = _next_free_port()
    if port in _running:
        raise RuntimeError(f"Port {port} already in use by {_running[port]['model']}")

    gpu_mem_util = gpu_mem_util or config.DEFAULT_GPU_MEM_UTIL
    dtype = dtype or config.DEFAULT_DTYPE

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", dtype,
        "--download-dir", config.MODEL_CACHE_DIR,
        "--no-enable-log-requests",
        "--uvicorn-log-level", "warning",
    ]
    if kv_cache_gb is not None:
        cmd += ["--kv-cache-memory-bytes", f"{kv_cache_gb}G"]
    else:
        cmd += ["--gpu-memory-utilization", str(gpu_mem_util)]
    if quantization:
        cmd += ["--quantization", quantization]
    if extra_args:
        cmd += extra_args

    nsys_report = None
    if nsys_profile:
        out_dir = nsys_output_dir or NSYS_DEFAULT_DIR
        os.makedirs(out_dir, exist_ok=True)
        safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
        ts = subprocess.check_output(["date", "+%Y%m%d_%H%M%S"]).decode().strip()
        nsys_report = os.path.join(out_dir, f"{safe_model}_{port}_{ts}.nsys-rep")
        cmd = [
            "nsys", "profile",
            "-t", "cuda,nvtx,cudnn,cublas",
            "-o", nsys_report,
            "--force-overwrite=true",
            "--stop-on-exit=true",
        ] + cmd

    env = {**os.environ, "HF_HUB_CACHE": config.MODEL_CACHE_DIR}
    if token:
        env["HF_TOKEN"] = token
    # If --max-model-len was passed and exceeds the model's trained position
    # embeddings, vLLM refuses to start unless this env var is set. We use the
    # override deliberately for VRAM-bounded context-sweep experiments — quality
    # past the trained max is undefined (RoPE extrapolation), but for memory
    # ceiling measurement this is intentional.
    if extra_args and any(a == "--max-model-len" for a in extra_args):
        env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    log_path = f"/tmp/vllm_{getpass.getuser()}_{port}.log"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)

    _running[port] = {
        "proc": proc,
        "model": model,
        "port": port,
        "gpu_mem_util": gpu_mem_util,
        "dtype": dtype,
        "quantization": quantization,
        "log_path": log_path,
        "log_file": log_file,
        "nsys_report": nsys_report,
    }
    return port


def stop_server(port):
    info = _running.get(port)
    if not info:
        return False
    proc = info["proc"]
    if isinstance(proc, subprocess.Popen) and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    elif isinstance(proc, _StubProcess):
        _kill_port(port)
    if info.get("log_file"):
        info["log_file"].close()
    del _running[port]
    return True


def _kill_port(port):
    """Kill whatever process is listening on a port."""
    try:
        result = subprocess.run(["fuser", f"{port}/tcp"], capture_output=True, text=True)
        pids = result.stdout.strip().split()
        for pid in pids:
            os.kill(int(pid.strip()), signal.SIGTERM)
    except Exception:
        pass


def is_alive(port):
    """Check if the server process is still running."""
    info = _running.get(port)
    if not info:
        return False
    return info["proc"].poll() is None


def check_health(port):
    try:
        r = requests.get(f"http://localhost:{port}/health", timeout=3)
        if r.status_code != 200:
            return False
    except requests.exceptions.RequestException:
        return False
    _refresh_memory_stats(port)
    return True


_WEIGHT_RE = re.compile(r"Model loading took\s+(?P<gib>[\d.]+)\s*Gi?B", re.IGNORECASE)
# vLLM emits the KV capacity differently across versions and launch modes:
#   - vLLM 0.17 with --kv-cache-memory-bytes: "reserved 10.0 GiB memory for KV Cache"
#   - older vLLM:                              "Available KV cache memory: 10.0 GiB" / "KV cache size: 10.0 GiB"
# Try the new format first, then fall back. Both put the number into the `gib` group.
_KV_RES = [
    re.compile(r"reserved\s+(?P<gib>[\d.]+)\s*Gi?B\s+memory\s+for\s+KV\s+Cache", re.IGNORECASE),
    re.compile(r"(?:Available KV cache memory|KV cache size)[^\d]*(?P<gib>[\d.]+)\s*Gi?B", re.IGNORECASE),
]
# vLLM also reports KV pool capacity in tokens — useful for derived kv_bytes_per_token.
_KV_TOKENS_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s+tokens", re.IGNORECASE)
_INIT_RE = re.compile(r"init engine.*took\s+([\d.]+)\s*seconds", re.IGNORECASE)


def _refresh_memory_stats(port):
    """Parse vLLM startup log for weight-mem, KV-capacity, and engine-init time
    once per server. The strings vLLM emits change between versions, so we try
    multiple regexes and skip silently if none match."""
    info = _running.get(port)
    if not info:
        return
    if (info.get("weight_mem_gib") and info.get("kv_cache_capacity_gib")
            and info.get("kv_cache_capacity_tokens") and info.get("engine_init_seconds")):
        return
    log_path = info.get("log_path")
    if not log_path or not os.path.exists(log_path):
        return
    try:
        with open(log_path, "r") as f:
            text = f.read()
    except Exception:
        return
    if "weight_mem_gib" not in info:
        m = _WEIGHT_RE.search(text)
        if m:
            info["weight_mem_gib"] = float(m.group("gib"))
    if "kv_cache_capacity_gib" not in info:
        for rx in _KV_RES:
            m = rx.search(text)
            if m:
                info["kv_cache_capacity_gib"] = float(m.group("gib"))
                break
    if "kv_cache_capacity_tokens" not in info:
        m = _KV_TOKENS_RE.search(text)
        if m:
            info["kv_cache_capacity_tokens"] = int(m.group(1).replace(",", ""))
    if "engine_init_seconds" not in info:
        m = _INIT_RE.search(text)
        if m:
            info["engine_init_seconds"] = float(m.group(1))


def get_log_by_path(log_path, lines=10):
    """Read the last N lines from a log file by path. Works even after process cleanup."""
    if not log_path or not os.path.exists(log_path):
        return []
    try:
        with open(log_path, "r") as f:
            all_lines = f.readlines()
        return [l.rstrip() for l in all_lines[-lines:]]
    except Exception:
        return []


def get_server_log(port, lines=10):
    """Read the last N lines from a server's log file."""
    info = _running.get(port)
    if not info:
        return []
    return get_log_by_path(info.get("log_path"), lines)


def get_latest_status(port, process_dead=False):
    """Get the most recent meaningful status line from the vLLM log."""
    lines = get_server_log(port, lines=50)
    if not lines:
        return "Starting..."

    if process_dead:
        for line in reversed(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("("):
                if "] " in stripped:
                    stripped = stripped.split("] ", 1)[-1]
                return stripped[:150]
        return "Process exited unexpectedly"

    keywords = [
        "Loading model", "Loading weights", "Model loading took",
        "Available KV cache", "init engine", "Starting to load",
        "Resolved architecture", "Using max model len",
        "Initializing", "world_size", "Starting", "ready",
        "Error", "error", "not supported", "failed", "ValidationError",
    ]
    for line in reversed(lines):
        for kw in keywords:
            if kw.lower() in line.lower():
                if "] " in line:
                    line = line.split("] ", 1)[-1]
                return line.strip()[:150]
    return "Starting..."


def list_running():
    dead = []
    for port, info in _running.items():
        proc = info["proc"]
        if isinstance(proc, subprocess.Popen) and proc.poll() is not None:
            if info.get("log_file"):
                info["log_file"].close()
            dead.append(port)
        elif isinstance(proc, _StubProcess) and not check_health(port):
            dead.append(port)
    for port in dead:
        del _running[port]

    return {
        port: {k: v for k, v in info.items() if k not in ("proc", "log_file")}
        for port, info in _running.items()
    }


def get_server_info(port):
    info = _running.get(port)
    if not info:
        return None
    return {k: v for k, v in info.items() if k not in ("proc", "log_file")}


def _find_log_for_port(port):
    import glob
    matches = glob.glob(f"/tmp/vllm_*_{port}.log")
    if matches:
        return max(matches, key=os.path.getmtime)
    old = f"/tmp/vllm_{port}.log"
    if os.path.exists(old):
        return old
    return None


def reconnect_orphans():
    """Detect vLLM servers started by previous GUI sessions and reconnect."""
    for port in range(config.VLLM_PORT_START, config.VLLM_PORT_START + 20):
        if port in _running:
            continue
        if not check_health(port):
            continue
        try:
            r = requests.get(f"http://localhost:{port}/v1/models", timeout=3)
            if r.status_code == 200:
                data = r.json()
                model = data.get("data", [{}])[0].get("id", "unknown")
                _running[port] = {
                    "proc": _StubProcess(),
                    "model": model,
                    "port": port,
                    "gpu_mem_util": "?",
                    "dtype": "?",
                    "quantization": None,
                    "log_path": _find_log_for_port(port),
                    "log_file": None,
                }
        except Exception:
            pass


class _StubProcess:
    """Stub for reconnected processes where we don't have the real Popen object."""
    def poll(self):
        return None


def _next_free_port():
    used = set(_running.keys())
    port = config.VLLM_PORT_START
    while port in used:
        port += 1
    return port
