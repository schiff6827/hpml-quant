"""Determine VRAM-bounded max context length for a model under a given launch
config by reading what vLLM reports at startup, instead of probing.

The old version of this script probed an already-running vLLM server with
gradually-larger prompts and binary-searched for the failure boundary. That
approach is bounded by whatever max_model_len vLLM was launched with — so it
can only ever discover the configured cap, never the true VRAM ceiling.

vLLM, however, computes the VRAM-bounded ceiling itself at startup:
  - On success it logs `GPU KV cache size: X tokens` (the KV pool capacity).
  - On failure (max_model_len too large for the KV pool) it logs
    `estimated maximum model length is X` and exits.
Either line is the answer we want — we just need to read it.

This script launches vLLM directly with `--max-model-len` set to a deliberately
high upper bound, watches the startup log, kills vLLM as soon as the answer
is observed, and writes the same JSON shape the old script produced.

NOTE on queue integration: the existing queue path calls this script against
an already-launched server (passes --port of a running vLLM). That path is now
incompatible with this rewrite — context-sweep jobs from the queue need their
launch step skipped (handled separately in queue_service). For ad-hoc runs,
invoke directly:

    python scripts/run_context_sweep.py \\
      --model Qwen/Qwen2.5-72B-Instruct-AWQ \\
      --result-dir benchmarks/ \\
      --run-name 72B-AWQ_ctx_$(date +%H%M%S) \\
      --kv-cache-gb 10
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime


# vLLM emits these strings on startup. Both formats are needed because we
# might catch either the success path or the "max_model_len too large" failure.
# Patterns kept loose enough to survive minor vLLM version changes.
_KV_TOKENS_RE     = re.compile(r'GPU KV cache size:\s*([\d,]+)\s+tokens')
_ESTIMATED_MAX_RE = re.compile(r'estimated maximum model length is\s+(\d+)')
_STARTUP_DONE_RE  = re.compile(r'Application startup complete|Uvicorn running')
_FATAL_HINTS      = ('CUDA out of memory', 'OutOfMemoryError', 'No CUDA GPUs are available')


def _build_vllm_cmd(args):
    cmd = [
        sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', args.model,
        '--host', '127.0.0.1',
        '--port', str(args.port),
        '--dtype', args.dtype,
        '--max-model-len', str(args.upper_bound),
        '--no-enable-log-requests',
        '--uvicorn-log-level', 'warning',
    ]
    if args.kv_cache_gb is not None:
        cmd += ['--kv-cache-memory-bytes', f'{args.kv_cache_gb}G']
    elif args.gpu_mem_util is not None:
        cmd += ['--gpu-memory-utilization', str(args.gpu_mem_util)]
    if args.quantization:
        cmd += ['--quantization', args.quantization]
    if args.trust_remote_code:
        cmd += ['--trust-remote-code']
    return cmd


def launch_and_extract(args):
    """Launch vLLM, watch its log, return what we learn about max context.
    Always kills the subprocess (and its process group) before returning."""
    log_path = f'/tmp/run_context_sweep_{args.port}_{int(time.time())}.log'
    log_file = open(log_path, 'w+')
    cmd = _build_vllm_cmd(args)
    print(f'launching| {" ".join(cmd)}', flush=True)

    env = {**os.environ}
    # Required when --max-model-len > the model's trained max_position_embeddings.
    # Without this, vLLM refuses to start before even attempting KV-pool sizing.
    env['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
    if args.hf_cache:
        env['HF_HUB_CACHE'] = args.hf_cache

    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)

    extracted = {
        'launch_succeeded':         False,
        'max_context_tokens':       None,
        'source':                   None,   # 'configured' | 'vllm_estimated'
        'kv_pool_tokens':           None,
        'configured_max_model_len': args.upper_bound,
        'launch_error':             None,
        'returncode':               None,
        'launch_log':               log_path,
    }

    deadline = time.time() + args.launch_timeout
    next_progress = time.time() + 15
    try:
        while time.time() < deadline:
            time.sleep(2)
            log_file.flush()
            log_file.seek(0)
            text = log_file.read()

            # Failure-path: this line is logged before vLLM crashes when the
            # KV pool can't serve a single request at the requested max_model_len.
            if extracted['max_context_tokens'] is None:
                m = _ESTIMATED_MAX_RE.search(text)
                if m:
                    extracted['max_context_tokens'] = int(m.group(1))
                    extracted['source'] = 'vllm_estimated'
                    print(f'detected| vLLM estimated maximum model length: {m.group(1)}', flush=True)

            # Success path — vLLM logs KV pool capacity AFTER it has validated
            # that max_model_len fits. Seeing this line means the launch will
            # succeed; the remaining startup work (cudagraph capture, uvicorn)
            # is irrelevant to our measurement and can take many minutes more.
            # We have everything we need; break immediately.
            if extracted['kv_pool_tokens'] is None:
                m = _KV_TOKENS_RE.search(text)
                if m:
                    extracted['kv_pool_tokens'] = int(m.group(1).replace(',', ''))
                    extracted['launch_succeeded'] = True
                    # The TRUE VRAM-bounded ceiling is the KV pool size in tokens
                    # (a single request can use the whole pool). The configured
                    # max_model_len is just a cap we set; if pool > cap, the
                    # cap is what limits a single request in practice.
                    extracted['max_context_tokens'] = extracted['kv_pool_tokens']
                    extracted['source'] = 'kv_pool_tokens'
                    print(f'detected| GPU KV cache size: {m.group(1)} tokens (== VRAM-bounded max single-request context)', flush=True)
                    break

            # Backstop: if Uvicorn comes up before we see the KV pool line for
            # any reason (vLLM version differences, log buffering), still treat
            # that as success and use the configured max_model_len as the answer.
            if _STARTUP_DONE_RE.search(text):
                extracted['launch_succeeded'] = True
                if extracted['max_context_tokens'] is None:
                    extracted['max_context_tokens'] = args.upper_bound
                    extracted['source'] = 'configured'
                print(f'detected| vLLM startup complete; configured max_model_len = {args.upper_bound} fits', flush=True)
                break

            # Process died — extract a useful error line for the JSON.
            if proc.poll() is not None:
                extracted['returncode'] = proc.returncode
                # One last parse in case the message landed in the final flush.
                if extracted['max_context_tokens'] is None:
                    m = _ESTIMATED_MAX_RE.search(text)
                    if m:
                        extracted['max_context_tokens'] = int(m.group(1))
                        extracted['source'] = 'vllm_estimated'
                if extracted['max_context_tokens'] is None:
                    # surface the last meaningful error line
                    candidates = [ln.strip() for ln in text.split('\n')
                                  if any(t in ln for t in ('ValueError', 'RuntimeError', 'Error', 'Traceback'))
                                  or any(t in ln for t in _FATAL_HINTS)]
                    extracted['launch_error'] = candidates[-1] if candidates else f'vllm exited rc={proc.returncode} without a parseable error (see {log_path})'
                break

            # Periodic progress so the queue log panel doesn't look hung.
            if time.time() >= next_progress:
                elapsed = int(args.launch_timeout - (deadline - time.time()))
                print(f'progress| {elapsed}s elapsed, watching vLLM startup', flush=True)
                next_progress = time.time() + 15
        else:
            extracted['launch_error'] = f'timed out after {args.launch_timeout}s waiting for vLLM startup'
    finally:
        # Kill the process group (vLLM forks an EngineCore child that holds VRAM).
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=5)
        except (ProcessLookupError, PermissionError):
            pass
        finally:
            log_file.close()

    return extracted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='HF id or local path')
    ap.add_argument('--port', type=int, default=8001)
    ap.add_argument('--upper-bound', type=int, default=131072,
                    help='vLLM is launched with --max-model-len = this value (default 128K). '
                         'Set higher if you want to discover ceilings above 128K.')
    ap.add_argument('--step', type=int, default=4096,
                    help='Unused in this rewrite — accepted for CLI back-compat with the old script.')
    ap.add_argument('--result-dir', required=True)
    ap.add_argument('--run-name', default='context_sweep')
    ap.add_argument('--kv-cache-gb', type=float, default=None,
                    help='Pass --kv-cache-memory-bytes <N>G to vLLM. If unset and --gpu-mem-util '
                         'is also unset, defaults to 10 GB (the queue default).')
    ap.add_argument('--gpu-mem-util', type=float, default=None,
                    help='Pass --gpu-memory-utilization to vLLM (0..1). Mutually exclusive with --kv-cache-gb.')
    ap.add_argument('--dtype', default='auto')
    ap.add_argument('--quantization', default=None,
                    help='Optional vLLM quantization (awq, gptq, bitsandbytes). Usually auto-detected from model name.')
    ap.add_argument('--trust-remote-code', action='store_true', default=True)
    ap.add_argument('--hf-cache', default=None, help='HF_HUB_CACHE override')
    ap.add_argument('--launch-timeout', type=int, default=600,
                    help='Seconds to wait for vLLM startup or failure (default 600 = 10 min). '
                         'Larger models take longer to load weights — bump for 70B+.')
    args = ap.parse_args()

    if args.kv_cache_gb is not None and args.gpu_mem_util is not None:
        ap.error('--kv-cache-gb and --gpu-mem-util are mutually exclusive')
    if args.kv_cache_gb is None and args.gpu_mem_util is None:
        args.kv_cache_gb = 10.0   # match the queue's default

    os.makedirs(args.result_dir, exist_ok=True)
    print(f'context-sweep| model = {args.model}', flush=True)
    print(f'context-sweep| upper_bound (max_model_len) = {args.upper_bound}', flush=True)

    extracted = launch_and_extract(args)

    # Match the JSON shape the old script wrote so downstream consumers
    # (parse_context_sweep_result, the analyze scripts) still work.
    result = {
        'type':                'context_sweep',
        'run_name':            args.run_name,
        'timestamp':           datetime.now().isoformat(),
        'max_context_tokens':  extracted['max_context_tokens'],
        'source':              extracted['source'],
        'kv_pool_tokens':      extracted['kv_pool_tokens'],
        'launch_succeeded':    extracted['launch_succeeded'],
        'launch_error':        extracted['launch_error'],
        'launch_log':          extracted['launch_log'],
        'returncode':          extracted['returncode'],
        'launch': {
            'configured_max_model_len': extracted['configured_max_model_len'],
            'kv_cache_gb':              args.kv_cache_gb,
            'gpu_mem_util':             args.gpu_mem_util,
            'dtype':                    args.dtype,
            'quantization':             args.quantization,
            'model':                    args.model,
        },
        'probes': [],   # kept empty for back-compat; this script doesn't probe
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.result_dir, f'context_sweep_{ts}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result['max_context_tokens']:
        print(f'\nMax VRAM-bounded context: {result["max_context_tokens"]:,} tokens '
              f'(source: {result["source"]})', flush=True)
    else:
        print(f'\nFAILED to determine max context: {result["launch_error"]}', flush=True)
    print(f'Wrote {out_path}', flush=True)


if __name__ == '__main__':
    main()
