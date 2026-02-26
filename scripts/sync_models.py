#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
import time
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
    key = m.get("key")
    src = m.get("source") or {}
    local = m.get("local") or {}

    if not key:
        raise ValueError(f"Model entry missing 'key': {m}")

    if src.get("type") != "hf":
        return {"skip": True, "key": key}

    repo_id = src.get("id")
    if not repo_id:
        raise ValueError(f"Model '{key}' missing source.id")

    revision = None  # strongly recommended

    # Back-compat: accept snapshot_dir as the "view dir"
    view_dir = local.get("view_dir") or local.get("snapshot_dir")

    tags = m.get("tags") or []
    if not isinstance(tags, list):
        raise ValueError(f"Model '{key}' tags must be a list")

    return {
        "skip": False,
        "key": key,
        "repo_id": repo_id,
        "view_dir": view_dir,
        "tags": set(tags),
    }


def atomic_symlink(target: Path, link_path: Path) -> None:
    """
    Create/replace link_path -> target atomically.
    """
    link_path_parent = link_path.parent
    ensure_dir(link_path_parent)

    tmp_link = link_path_parent / (link_path.name + ".tmp-link")
    try:
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()
        tmp_link.symlink_to(target)
        # Replace existing link/file/dir
        if link_path.is_symlink() or link_path.exists():
            if link_path.is_dir() and not link_path.is_symlink():
                # If it's a real directory, remove it (dangerous if not intended)
                shutil.rmtree(link_path)
            else:
                link_path.unlink()
        tmp_link.replace(link_path)
    finally:
        if tmp_link.exists() or tmp_link.is_symlink():
            try:
                tmp_link.unlink()
            except Exception:
                pass


def safe_rmtree(p: Path) -> None:
    if p.is_symlink() or p.is_file():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)


def copy_snapshot(snapshot_path: Path, dst: Path) -> None:
    """
    Materialize a snapshot directory into dst as real files.
    """
    if dst.exists() or dst.is_symlink():
        safe_rmtree(dst)
    ensure_dir(dst)

    # Copy directory tree
    # Note: this duplicates disk usage; only do if you truly need portability.
    shutil.copytree(snapshot_path, dst, dirs_exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sync HF model snapshots from a models YAML into the HF cache, with optional local view dirs."
    )
    ap.add_argument("yaml_path", type=str, help="Path to models.yaml")

    ap.add_argument("--tag", action="append", default=[], help="Only include models with this tag (repeatable)")
    ap.add_argument("--key", action="append", default=[], help="Only include these model keys (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="Print what would happen, but do nothing")
    ap.add_argument("--max-models", type=int, default=0, help="Hard cap number of models processed this run (0=no cap)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between downloads")

    ap.add_argument("--root", type=str, default="", help="Require view_dir (if set) be under this root (safety)")
    ap.add_argument("--min-free-gb", type=float, default=0.0, help="Abort if free space under --root is below this")

    ap.add_argument(
        "--materialize",
        action="store_true",
        help="Instead of symlink view dirs, copy real files into view_dir (duplicates disk). Default: symlink views.",
    )

    ap.add_argument(
        "--allow",
        action="append",
        default=[],
        help="Allow patterns passed to snapshot_download (repeatable). Example: --allow '*.safetensors'",
    )
    ap.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Ignore patterns passed to snapshot_download (repeatable). Example: --ignore '*.msgpack'",
    )

    args = ap.parse_args()

    yaml_path = Path(args.yaml_path).expanduser().resolve()
    cfg = load_yaml(yaml_path)

    # Configure HF cache location from YAML unless user already set env vars.
    hf_home = (cfg.get("paths") or {}).get("hf_home")
    if hf_home and not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = hf_home
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

    if root and args.min_free_gb > 0:
        free = disk_free_gb(root)
        if free < args.min_free_gb:
            raise SystemExit(f"Refusing to run: free space under {root} is {free:.1f} GB (< {args.min_free_gb} GB)")

    # Filter models (do NOT dedupe models; we may want multiple views)
    selected: List[Dict[str, Any]] = []
    for raw in models:
        if not isinstance(raw, dict):
            continue

        raw_key = raw.get("key")
        raw_tags = set(raw.get("tags", []) or [])

        if want_keys and raw_key not in want_keys:
            continue
        if want_tags and not (want_tags & raw_tags):
            continue

        m = normalize_model_entry(raw)
        if m["skip"]:
            continue

        # Safety: if a view_dir is set, enforce root constraint
        if m["view_dir"]:
            view = Path(m["view_dir"]).expanduser()
            if root and not within_root(view, root):
                raise SystemExit(f"Safety check failed: view_dir {view} is not under root {root}")

        selected.append(m)

    if args.max_models and len(selected) > args.max_models:
        selected = selected[: args.max_models]

    if not selected:
        print("No models matched filters; nothing to do.")
        return

    allow_patterns = args.allow if args.allow else None
    ignore_patterns = args.ignore if args.ignore else None

    print(f"HF_HOME={os.environ.get('HF_HOME','')}")
    print(f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE','')}")
    print(f"Models to process: {len(selected)}")
    if allow_patterns:
        print(f"allow_patterns={allow_patterns}")
    if ignore_patterns:
        print(f"ignore_patterns={ignore_patterns}")
    print(f"view_mode={'copy' if args.materialize else 'symlink'}")

    # Manifest alongside YAML
    manifest_path = yaml_path.with_suffix(".manifest.jsonl")

    for i, m in enumerate(selected, 1):
        key: str = m["key"]
        repo_id: str = m["repo_id"]
        # revision: Optional[str] = m["revision"]
        view_dir_raw: Optional[str] = m["view_dir"]

        if root and args.min_free_gb > 0:
            free = disk_free_gb(root)
            if free < args.min_free_gb:
                raise SystemExit(f"Abort: free space under {root} is {free:.1f} GB (< {args.min_free_gb} GB)")

        print(f"[{i}/{len(selected)}] CACHE   key={key}  repo={repo_id}")

        if args.dry_run:
            snapshot_path = Path("/HF/CACHE/SNAPSHOT/PATH/WOULD/BE/HERE")
        else:
            snapshot_path_str = snapshot_download(
                repo_id=repo_id,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            snapshot_path = Path(snapshot_path_str)

        # Optional: create view dir
        if view_dir_raw:
            view_dir = Path(view_dir_raw).expanduser()
            print(f"              VIEW    {view_dir} -> {snapshot_path}")

            if not args.dry_run:
                ensure_dir(view_dir.parent)
                if args.materialize:
                    copy_snapshot(snapshot_path, view_dir)
                else:
                    atomic_symlink(snapshot_path, view_dir)

        # Append manifest record
        rec = {
            "key": key,
            "repo_id": repo_id,
            "snapshot_path": str(snapshot_path),
            "view_dir": view_dir_raw,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        if not args.dry_run:
            with manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        if args.sleep > 0 and not args.dry_run:
            time.sleep(args.sleep)

    print(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()