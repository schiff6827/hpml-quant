from huggingface_hub import HfApi, snapshot_download, scan_cache_dir
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import re
import os
import shutil
import time
import threading
import config

api = HfApi()

TOP_PROVIDERS = [
    "All", "Qwen", "meta-llama", "deepseek-ai", "microsoft", "openai",
    "nvidia", "mistralai", "google", "EleutherAI", "facebook",
    "RedHatAI", "h2oai", "lmstudio-community", "mlx-community",
    "zai-org", "tiiuae", "bigscience", "stabilityai", "HuggingFaceH4",
]

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "BF16": 2, "F16": 2,
    "I32": 4, "I16": 2, "I8": 1, "U8": 1,
    "F8_E5M2": 1, "F8_E4M3": 1,
}

_PARAM_RE = re.compile(r'[\-_](\d+(?:\.\d+)?)\s*[xX]?\s*([BMTbmt])\b')


def _params_from_safetensors(safetensors):
    if not safetensors or not safetensors.parameters:
        return 0
    return next(iter(safetensors.parameters.values()))


def _params_from_name(name):
    """Fallback: extract param count from model name like '8B', '70B', '1.5B'."""
    match = _PARAM_RE.search(name)
    if not match:
        return 0
    num = float(match.group(1))
    unit = match.group(2).upper()
    if unit == "B":
        return int(num * 1e9)
    if unit == "M":
        return int(num * 1e6)
    if unit == "T":
        return int(num * 1e12)
    return 0


def _format_params(raw):
    if not raw:
        return "N/A"
    if raw >= 1e12:
        return f"{raw / 1e12:.1f}T"
    if raw >= 1e9:
        return f"{raw / 1e9:.1f}B"
    if raw >= 1e6:
        return f"{raw / 1e6:.0f}M"
    return str(raw)


def _size_from_safetensors(safetensors):
    if not safetensors or not safetensors.parameters:
        return 0
    total_bytes = 0
    for dtype, count in safetensors.parameters.items():
        total_bytes += count * DTYPE_BYTES.get(dtype, 2)
    return total_bytes


def _size_from_siblings(siblings):
    if not siblings:
        return 0
    return sum(s.size for s in siblings if s.size)


def _format_size(raw_bytes):
    if not raw_bytes:
        return "N/A"
    gb = raw_bytes / 1e9
    if gb >= 1:
        return f"{gb:.1f} GB"
    mb = raw_bytes / 1e6
    if mb >= 1:
        return f"{mb:.0f} MB"
    return f"{raw_bytes / 1e3:.0f} KB"


def get_cached_model_ids():
    try:
        cache_info = scan_cache_dir(config.MODEL_CACHE_DIR)
        return {repo.repo_id for repo in cache_info.repos if repo.repo_type == "model"}
    except Exception:
        return set()


def get_disk_space():
    """Return free/total disk space for the model cache directory in GB."""
    import shutil
    u = shutil.disk_usage(config.MODEL_CACHE_DIR)
    return {"free_gb": u.free / 1e9, "total_gb": u.total / 1e9, "used_gb": u.used / 1e9}


def search_models(query="", provider="", limit=None, sort_by="downloads"):
    limit = limit or config.DEFAULT_SEARCH_LIMIT
    if provider == "All":
        provider = ""
    cached_ids = get_cached_model_ids()

    models = api.list_models(
        search=query or None,
        author=provider or None,
        sort=sort_by,
        direction=-1,
        limit=limit,
        expand=["downloadsAllTime", "trendingScore", "safetensors", "gated",
                "likes", "createdAt", "pipeline_tag", "siblings"],
    )
    results = []
    for m in models:
        parts = m.id.split("/", 1)
        provider_name = parts[0] if len(parts) > 1 else ""
        model_name = parts[1] if len(parts) > 1 else parts[0]

        params_raw = _params_from_safetensors(m.safetensors) or _params_from_name(m.id)
        size_raw = _size_from_safetensors(m.safetensors) or _size_from_siblings(m.siblings)
        size_estimated = False
        if not size_raw and params_raw:
            size_raw = params_raw * 2
            size_estimated = True

        size_str = _format_size(size_raw)
        if size_estimated and size_str != "N/A":
            size_str = f"~{size_str}"

        results.append({
            "id": m.id,
            "provider": provider_name,
            "model_name": model_name,
            "type": m.pipeline_tag or "",
            "params_raw": params_raw,
            "params": _format_params(params_raw),
            "size_raw": size_raw,
            "size_gb": size_str,
            "downloads_30d": m.downloads or 0,
            "downloads_all": m.downloads_all_time or 0,
            "trending": m.trending_score or 0,
            "likes": m.likes or 0,
            "gated": str(m.gated) if m.gated else "No",
            "date": m.created_at.strftime("%Y-%m-%d") if m.created_at else "",
            "downloaded": m.id in cached_ids,
        })
    return results


# -- Download with byte-level progress monitoring --

def _get_model_cache_dir(repo_id):
    """Get the HF cache directory for a model repo."""
    safe_name = repo_id.replace("/", "--")
    return Path(config.MODEL_CACHE_DIR) / f"models--{safe_name}"


def _measure_download_bytes(repo_id):
    """Measure total bytes currently on disk for a model's blobs."""
    model_dir = _get_model_cache_dir(repo_id)
    blobs_dir = model_dir / "blobs"
    if not blobs_dir.exists():
        return 0
    total = 0
    for f in blobs_dir.iterdir():
        if f.is_file():
            total += f.stat().st_size
    return total


def download_model(repo_id, token=None, progress=None):
    """Blocking download — call via run.io_bound from the UI.
    progress: optional dict that gets updated with download state.
    """
    if progress is not None:
        progress["value"] = 0.0
        progress["total_files"] = 0
        progress["completed_files"] = 0
        progress["bytes_downloaded"] = 0
        progress["bytes_total"] = progress.get("expected_bytes", 0)
        progress["start_time"] = time.time()
        progress["status"] = "Starting..."

        # Start a monitoring thread that polls disk usage
        progress["_stop"] = False

        def monitor():
            while not progress["_stop"]:
                try:
                    on_disk = _measure_download_bytes(repo_id)
                    progress["bytes_downloaded"] = on_disk
                    total = progress["bytes_total"]
                    if total > 0:
                        progress["value"] = min(on_disk / total, 0.99)
                except Exception:
                    pass
                time.sleep(1)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    try:
        result = snapshot_download(
            repo_id=repo_id,
            cache_dir=config.MODEL_CACHE_DIR,
            token=token or None,
            ignore_patterns=["*.bin", "original/*"],
        )
        if progress is not None:
            progress["value"] = 1.0
            progress["status"] = "Complete"
        return result
    finally:
        if progress is not None:
            progress["_stop"] = True


# -- Cache operations --

def list_cached_models():
    try:
        cache_info = scan_cache_dir(config.MODEL_CACHE_DIR)
    except Exception:
        return []
    results = []
    _hidden = {'sentence-transformers/all-MiniLM-L6-v2'}
    for repo in cache_info.repos:
        if repo.repo_type != "model":
            continue
        if repo.repo_id in _hidden:
            continue
        last_mod = ""
        if repo.revisions:
            ts = max(r.last_modified for r in repo.revisions)
            last_mod = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        results.append({
            "id": repo.repo_id,
            "size_gb": f"{repo.size_on_disk / 1e9:.1f} GB",
            "size_bytes": repo.size_on_disk,
            "revisions": len(repo.revisions),
            "last_modified": last_mod,
        })
    return results


def delete_cached_model(repo_id):
    """Delete a cached model, including partial downloads and lock files."""
    cache_info = scan_cache_dir(config.MODEL_CACHE_DIR)
    hashes_to_delete = []
    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            for rev in repo.revisions:
                hashes_to_delete.append(rev.commit_hash)

    if hashes_to_delete:
        delete_strategy = cache_info.delete_revisions(*hashes_to_delete)
        delete_strategy.execute()

    # Clean up .incomplete blobs and stale lock files
    model_dir = _get_model_cache_dir(repo_id)
    if model_dir.exists():
        blobs_dir = model_dir / "blobs"
        if blobs_dir.exists():
            for f in blobs_dir.glob("*.incomplete"):
                f.unlink(missing_ok=True)
        # Remove the model dir entirely if no refs remain
        refs_dir = model_dir / "refs"
        has_refs = refs_dir.exists() and any(refs_dir.iterdir())
        if not has_refs:
            shutil.rmtree(model_dir, ignore_errors=True)

    # Clean up lock files
    safe_name = repo_id.replace("/", "--")
    locks_dir = Path(config.MODEL_CACHE_DIR) / ".locks" / f"models--{safe_name}"
    if locks_dir.exists():
        shutil.rmtree(locks_dir, ignore_errors=True)

    return True
