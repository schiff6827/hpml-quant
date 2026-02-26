#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml
from huggingface_hub import snapshot_download


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text())
    except Exception as ex:
        raise SystemExit(f"Failed to read YAML: {path}\n{ex}")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_already_downloaded(dst: Path) -> bool:
    if not dst.exists():
        return False
    # Common HF snapshot artifacts
    if (dst / "config.json").exists():
        return True
    if any(dst.glob("*.safetensors")):
        return True
    if any(dst.glob("*.bin")):
        return True
    # Sometimes nested under model/ etc.
    if any(dst.rglob("*.safetensors")):
        return True
    if any(dst.rglob("*.bin")):
        return True
    return False


def within_root(dst: Path, root: Optional[Path]) -> bool:
    if root is None:
        return True
    try:
        dst.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def disk_free_gb(path: Path) -> float:
    st = os.statvfs(str(path))
    free_bytes = st.f_bavail * st.f_frsize
    return free_bytes / (1024**3)


def normalize_model_entry(m: Dict[str, Any]) -> Dict[str, Any]:
    # Expect schema like your YAML.
    key = m.get("key")
    src = m.get("source") or {}
    local = m.get("local") or {}

    if not key:
        raise ValueError(f"Model entry missing 'key': {m}")
    if src.get("type") != "hf":
        # We only handle HF models here.
        return {"skip": True, "key": key}

    repo_id = src.get("id")
    if not repo_id:
        raise ValueError(f"Model '{key}' missing source.id")

    revision = src.get("revision")  # optional

    snapshot_dir = local.get("snapshot_dir")
    if not snapshot_dir:
        raise ValueError(f"Model '{key}' missing local.snapshot_dir")

    tags = m.get("tags") or []
    if not isinstance(tags, list):
        raise ValueError(f"Model '{key}' tags must be a list")

    return {
        "skip": False,
        "key": key,
        "repo_id": repo_id,
        "revision": revision,
        "snapshot_dir": snapshot_dir,
        "tags": set(tags),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Download HF model snapshots specified in models YAML.")
    ap.add_argument("yaml_path", type=str, help="Path to models.yaml")
    ap.add_argument("--tag", action="append", default=[], help="Only include models with this tag (repeatable)")
    ap.add_argument("--key", action="append", default=[], help="Only include these model keys (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="Print what would happen, but do nothing")
    ap.add_argument("--max-models", type=int, default=0, help="Hard cap number of downloads this run (0 = no cap)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between downloads")
    ap.add_argument("--root", type=str, default="", help="Require snapshot_dir be under this root (safety)")
    ap.add_argument("--min-free-gb", type=float, default=0.0, help="Abort if free space under root is below this")
    ap.add_argument("--no-symlinks", action="store_true", help="Store real files (portable). Default: real files.")
    args = ap.parse_args()

    yaml_path = Path(args.yaml_path).expanduser()
    cfg = load_yaml(yaml_path)

    # If your YAML defines paths.hf_home, set HF_HOME accordingly unless already set.
    hf_home = (cfg.get("paths") or {}).get("hf_home")
    if hf_home and not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = hf_home
    # Keep hub cache explicit if HF_HOME set
    if os.environ.get("HF_HOME") and not os.environ.get("HF_HUB_CACHE"):
        os.environ["HF_HUB_CACHE"] = str(Path(os.environ["HF_HOME"]) / "hub")

    models = cfg.get("models")
    if not isinstance(models, list):
        raise SystemExit("YAML must contain a top-level 'models:' list")

    want_tags: Set[str] = set(args.tag or [])
    want_keys: Set[str] = set(args.key or [])

    root = Path(args.root).expanduser().resolve() if args.root else None
    if root:
        ensure_dir(root)

    # Disk guardrail
    if root and args.min_free_gb > 0:
        free = disk_free_gb(root)
        if free < args.min_free_gb:
            raise SystemExit(f"Refusing to run: free space under {root} is {free:.1f} GB (< {args.min_free_gb} GB)")

    # Build download plan with dedupe by (repo_id, revision)
    planned: List[Dict[str, Any]] = []
    seen_sources: Set[Tuple[str, Optional[str]]] = set()

    for raw in models:
        if not isinstance(raw, dict):
            continue
        m = normalize_model_entry(raw)
        if m["skip"]:
            continue

        key = m["key"]
        tags = m["tags"]

        if want_keys and key not in want_keys:
            continue
        if want_tags and not (want_tags & tags):
            continue

        repo_id = m["repo_id"]
        revision = m["revision"]
        dst = Path(m["snapshot_dir"]).expanduser()

        if root and not within_root(dst, root):
            raise SystemExit(f"Safety check failed: {dst} is not under root {root}")

        source_key = (repo_id, revision)
        if source_key in seen_sources:
            # Deduplicate (e.g., bnb_nf4 shares base model)
            continue
        seen_sources.add(source_key)

        planned.append({**m, "dst": dst})

    if args.max_models and len(planned) > args.max_models:
        planned = planned[: args.max_models]

    if not planned:
        print("No models matched filters; nothing to do.")
        return

    print(f"HF_HOME={os.environ.get('HF_HOME','')}")
    print(f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE','')}")
    print(f"Planned unique downloads: {len(planned)}")

    for i, m in enumerate(planned, 1):
        dst: Path = m["dst"]
        repo_id: str = m["repo_id"]
        revision: Optional[str] = m["revision"]
        key: str = m["key"]

        ensure_dir(dst)

        if is_already_downloaded(dst):
            print(f"[{i}/{len(planned)}] SKIP exists  key={key}  repo={repo_id}  -> {dst}")
            continue

        print(f"[{i}/{len(planned)}] DOWNLOAD   key={key}  repo={repo_id}  rev={revision or 'default'}  -> {dst}")

        if args.dry_run:
            continue

        # Re-check disk before each download if guard enabled
        if root and args.min_free_gb > 0:
            free = disk_free_gb(root)
            if free < args.min_free_gb:
                raise SystemExit(f"Abort: free space under {root} is {free:.1f} GB (< {args.min_free_gb} GB)")

        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(dst),
            local_dir_use_symlinks=not args.no_symlinks,  # default False below by arg
        )

        if args.sleep > 0:
            import time
            time.sleep(args.sleep)

    print("Done.")


if __name__ == "__main__":
    main()