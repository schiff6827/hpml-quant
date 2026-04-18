"""Context-length sweep probe runner.

Submits single completions at increasing prompt lengths to a running vLLM
server until a failure is observed, then binary-searches the boundary.
Prints tqdm-style progress lines so the Benchmark tab log panel renders
them, and writes context_sweep_<ts>.json into the result dir when done.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import requests


def fetch_kv_pct(port, timeout=2):
    try:
        r = requests.get(f"http://localhost:{port}/metrics", timeout=timeout)
        if r.status_code != 200:
            return 0.0
        for line in r.text.split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            name_labels, _, value = line.rpartition(" ")
            if not value:
                continue
            name = name_labels.split("{")[0]
            if name in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
                try:
                    return float(value) * 100
                except ValueError:
                    return 0.0
    except Exception:
        pass
    return 0.0


def probe(port, model, n_tokens, max_output=16, timeout=None):
    """Send a request with exactly n_tokens prompt tokens. Returns probe dict."""
    prompt_ids = [1] * int(n_tokens)
    start = time.monotonic()
    err = None
    ttft_ms = None
    success = False
    try:
        r = requests.post(
            f"http://localhost:{port}/v1/completions",
            json={
                "model": model,
                "prompt": prompt_ids,
                "max_tokens": int(max_output),
                "temperature": 0.0,
                "stream": False,
            },
            timeout=timeout or max(60, n_tokens / 100),
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        if r.status_code == 200:
            success = True
            ttft_ms = elapsed_ms
        else:
            err = f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    kv_pct = fetch_kv_pct(port)
    return {
        "n": int(n_tokens),
        "success": success,
        "ttft_ms": ttft_ms,
        "kv_cache_pct_at_end": kv_pct,
        "error": err,
    }


def sweep(port, model, upper_bound, step):
    probes = []
    # Doubling phase: find first failure.
    n = step
    last_success = 0
    first_fail = None
    while n <= upper_bound:
        pct = int(100 * min(n / upper_bound, 0.5))
        print(f"{pct}%| probing {n} tokens", flush=True)
        result = probe(port, model, n)
        probes.append(result)
        if result["success"]:
            last_success = n
            n = min(n * 2, upper_bound + 1)
            if last_success == upper_bound:
                break
        else:
            first_fail = n
            break
    # If we ran past upper_bound without failing, cap there.
    if first_fail is None:
        max_context = last_success
    else:
        # Binary search between last_success and first_fail.
        lo, hi = last_success, first_fail
        while hi - lo > step:
            mid = (lo + hi) // 2
            mid = (mid // step) * step
            if mid <= lo:
                mid = lo + step
            pct = int(100 * (0.5 + 0.5 * (mid - last_success) / max(first_fail - last_success, 1)))
            print(f"{pct}%| binary-search {mid} tokens", flush=True)
            result = probe(port, model, mid)
            probes.append(result)
            if result["success"]:
                lo = mid
            else:
                hi = mid
        max_context = lo
    print("100%| done", flush=True)
    return {
        "max_context_tokens": max_context,
        "probes": probes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--upper-bound", type=int, default=131072)
    ap.add_argument("--step", type=int, default=4096)
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--run-name", default="context_sweep")
    args = ap.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    result = sweep(args.port, args.model, args.upper_bound, args.step)
    result["run_name"] = args.run_name
    result["timestamp"] = datetime.now().isoformat()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.result_dir, f"context_sweep_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Max context: {result['max_context_tokens']} tokens", flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
