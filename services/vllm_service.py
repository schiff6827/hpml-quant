import subprocess
import signal
import sys
import os
import requests
import config

# port -> {proc, model, port, gpu_mem_util, dtype, quantization, log_path}
_running = {}


def launch_server(model, port=None, gpu_mem_util=None, dtype=None, quantization=None, extra_args=None, token=None, kv_cache_gb=None):
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

    env = {**os.environ, "HF_HUB_CACHE": config.MODEL_CACHE_DIR}
    if token:
        env["HF_TOKEN"] = token

    log_path = f"/tmp/vllm_{port}.log"
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
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


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
        port: {k: v for k, v in info.items() if k not in ("proc", "log_file", "log_path")}
        for port, info in _running.items()
    }


def get_server_info(port):
    info = _running.get(port)
    if not info:
        return None
    return {k: v for k, v in info.items() if k not in ("proc", "log_file", "log_path")}


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
                    "log_path": f"/tmp/vllm_{port}.log",
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
