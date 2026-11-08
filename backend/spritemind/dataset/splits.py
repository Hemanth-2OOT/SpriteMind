"""Frozen held-out split definitions (R10).

A random record split is not a generalization test: if the same
(category, style, pose) combination appears in train and test, the model can
memorise the composition. This module reads a held-out spec that names the
combinations excluded from *every* training and validation split, so a
held-out score measures unseen combinations instead of unseen seeds.

The spec is a JSON file written once by `scripts/build_heldout_split.py` and
never regenerated implicitly; datasets record its digest in their metadata.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SPEC_VERSION = 1


def canonical_json(spec: Dict[str, Any]) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def spec_digest(spec: Dict[str, Any]) -> str:
    """SHA-256 of the spec body, recorded in dataset metadata for provenance."""
    body = {k: v for k, v in spec.items() if k != "digest"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def triple_keys(spec: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    return [tuple(t) for t in spec.get("heldout_triples", [])]  # type: ignore[misc]


def category_element_keys(spec: Dict[str, Any]) -> List[Tuple[str, str]]:
    return [tuple(p) for p in spec.get("heldout_category_element", [])]  # type: ignore[misc]


def record_key(record: Dict[str, Any]) -> Tuple[str, str, str]:
    return (record.get("category"), record.get("style"), record.get("pose"))


def heldout_reason(record: Dict[str, Any], spec: Dict[str, Any]) -> Optional[str]:
    """Why this record is held out, or None when it may be used for training."""
    if record_key(record) in set(triple_keys(spec)):
        return "triple"
    pair = (record.get("category"), record.get("element"))
    if pair[1] and pair in set(category_element_keys(spec)):
        return "category_element"
    return None


def audit(records: Iterable[Dict[str, Any]], spec: Dict[str, Any]) -> Dict[str, Any]:
    """Counts of held-out records per reason, and any held-out combo in train/val."""
    per_reason: Dict[str, int] = {"triple": 0, "category_element": 0}
    leaks: List[Dict[str, str]] = []
    for record in records:
        reason = heldout_reason(record, spec)
        if reason:
            per_reason[reason] += 1
            if record.get("split") in {"train", "val"}:
                leaks.append({
                    "id": str(record.get("id")),
                    "split": str(record.get("split")),
                    "reason": reason,
                })
    return {
        "spec_digest": spec_digest(spec),
        "heldout_records": per_reason,
        "leaks": leaks[:20],
        "leak_count": len(leaks),
    }


def load_spec(path: Path | str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != SPEC_VERSION:
        raise ValueError(f"unsupported held-out spec version: {data.get('version')!r}")
    if data.get("digest") != spec_digest(data):
        raise ValueError("held-out spec digest does not match its contents")
    return data
