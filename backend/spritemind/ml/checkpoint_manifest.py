"""Checkpoint manifest: what a weights file actually contains.

A checkpoint is only useful together with the taxonomy and tensor shapes it was trained
against. Recording them next to the weights turns a silent shape-mismatch fallback into a
readable diagnostic.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

MANIFEST_NAME = "manifest.json"


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def manifest_path(checkpoints_dir: Path) -> Path:
    return Path(checkpoints_dir) / MANIFEST_NAME


def read_manifest(checkpoints_dir: Path) -> Dict[str, Any]:
    path = manifest_path(checkpoints_dir)
    if not path.exists():
        return {"version": 1, "checkpoints": {}}
    data = json.loads(path.read_text())
    data.setdefault("version", 1)
    data.setdefault("checkpoints", {})
    return data


def write_manifest(checkpoints_dir: Path, entries: Dict[str, Dict[str, Any]]) -> Path:
    path = manifest_path(checkpoints_dir)
    path.write_text(json.dumps({"version": 1, "checkpoints": entries}, indent=2, sort_keys=True) + "\n")
    return path


def describe_file(path: Path, **extra: Any) -> Dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    return {"file": path.name, "bytes": stat.st_size, "sha256": _digest(path), **extra}


def record(checkpoints_dir: Path, entry: Dict[str, Any]) -> Path:
    """Adds or replaces one checkpoint entry, keeping the rest of the manifest intact."""
    data = read_manifest(checkpoints_dir)
    data["checkpoints"][entry["file"]] = entry
    return write_manifest(checkpoints_dir, data["checkpoints"])


def describe(checkpoints_dir: Path, checkpoint_name: str) -> Optional[Dict[str, Any]]:
    return read_manifest(checkpoints_dir)["checkpoints"].get(checkpoint_name)


def verify(checkpoints_dir: Path, checkpoint_name: str) -> Dict[str, Any]:
    """Checks presence and, when recorded, integrity of a checkpoint."""
    checkpoints_dir = Path(checkpoints_dir)
    path = checkpoints_dir / checkpoint_name
    entry = describe(checkpoints_dir, checkpoint_name)
    if not path.exists():
        return {"exists": False, "recorded": entry is not None, "ok": False, "reason": "missing file"}
    if entry is None:
        return {"exists": True, "recorded": False, "ok": True, "reason": "not in manifest"}
    if entry.get("bytes") != path.stat().st_size:
        return {"exists": True, "recorded": True, "ok": False, "reason": "size mismatch"}
    if entry.get("sha256") and entry["sha256"] != _digest(path):
        return {"exists": True, "recorded": True, "ok": False, "reason": "sha256 mismatch"}
    return {"exists": True, "recorded": True, "ok": True, "reason": "verified"}


def prune_missing(checkpoints_dir: Path) -> List[str]:
    """Removes manifest entries whose file is gone; returns the pruned names.

    Entries are dropped, never files: a checkpoint on disk that is not in the
    manifest is always kept.
    """
    checkpoints_dir = Path(checkpoints_dir)
    entries = read_manifest(checkpoints_dir)["checkpoints"]
    missing = [name for name in sorted(entries) if not (checkpoints_dir / name).exists()]
    if missing:
        for name in missing:
            del entries[name]
        write_manifest(checkpoints_dir, entries)
    return missing


def list_entries(checkpoints_dir: Path) -> List[Dict[str, Any]]:
    entries = read_manifest(checkpoints_dir)["checkpoints"]
    return [entries[name] for name in sorted(entries)]
