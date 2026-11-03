"""Storage Cleanup and Optimization Utility for SpriteMind Stage 5.5.

Identifies disposable cache, intermediate renders, and temporary files.
Provides safe Dry-Run mode and strictly guarded Execution mode.
"""
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from spritemind.storage.classification import (
    DataClassification,
    StorageClassifier,
    ProtectedResourceException,
    DELETABLE_CLASSIFICATIONS,
)
from spritemind.storage.database import StorageDatabase


class StorageCleaner:
    """Auditable storage cleanup engine with mandatory Dry-Run support."""

    def __init__(self, db: Optional[StorageDatabase] = None):
        self.db = db or StorageDatabase()

    def scan(self) -> Dict[str, Any]:
        """Performs a read-only Dry-Run scan for disposable temporary files."""
        candidates: List[Dict[str, Any]] = []
        total_reclaimable_bytes = 0

        # 1. Bytecode caches
        for p in Path(".").rglob("__pycache__"):
            if p.is_dir():
                for pyc in p.glob("*.pyc"):
                    sz = pyc.stat().st_size
                    total_reclaimable_bytes += sz
                    candidates.append({
                        "path": pyc.as_posix(),
                        "size_bytes": sz,
                        "category": "bytecode_cache",
                        "reason": "Compiled Python bytecode (.pyc) can be safely regenerated at runtime.",
                    })

        # 2. Pytest caches
        for p in Path(".pytest_cache").rglob("*"):
            if p.is_file():
                sz = p.stat().st_size
                total_reclaimable_bytes += sz
                candidates.append({
                    "path": p.as_posix(),
                    "size_bytes": sz,
                    "category": "test_cache",
                    "reason": "Temporary pytest cache metadata.",
                })

        # 3. Temporary text or export files
        for p in Path("data").rglob("*.tmp"):
            if p.is_file():
                sz = p.stat().st_size
                total_reclaimable_bytes += sz
                candidates.append({
                    "path": p.as_posix(),
                    "size_bytes": sz,
                    "category": "temp_file",
                    "reason": "Intermediate temporary file.",
                })

        return {
            "mode": "dry_run",
            "candidate_count": len(candidates),
            "total_candidates": len(candidates),
            "reclaimable_bytes": total_reclaimable_bytes,
            "reclaimable_kb": round(total_reclaimable_bytes / 1024, 2),
            "reclaimable_mb": round(total_reclaimable_bytes / (1024 * 1024), 2),
            "candidates": candidates,
        }

    def execute(self, candidate_paths: Optional[List[str]] = None, confirm: bool = False) -> Dict[str, Any]:
        """Removes approved disposable files. Strictly blocks any protected datasets or models."""
        if not confirm:
            raise ValueError("Cleanup execution requires explicit confirm=True.")

        dry_run_results = self.scan()
        allowed_paths = {c["path"] for c in dry_run_results["candidates"]}

        targets = candidate_paths if candidate_paths is not None else list(allowed_paths)
        deleted_count = 0
        freed_bytes = 0

        for target_str in targets:
            p = Path(target_str)
            rel_path = p.as_posix()

            # Absolute protection check
            StorageClassifier.assert_deletable(rel_path)

            classification = StorageClassifier.classify_path(rel_path)
            if classification != DataClassification.TEMPORARY_REGENERABLE:
                raise ProtectedResourceException(
                    f"Cleanup rejected: '{rel_path}' is classified as '{classification.value}', not temporary!"
                )

            if p.exists() and p.is_file():
                sz = p.stat().st_size
                p.unlink()
                deleted_count += 1
                freed_bytes += sz
                self.db.log_audit("CLEANUP_FILE", rel_path, "SUCCESS", f"Removed {sz} bytes.")

        return {
            "mode": "executed",
            "deleted_count": deleted_count,
            "freed_bytes": freed_bytes,
            "freed_kb": round(freed_bytes / 1024, 2),
        }
