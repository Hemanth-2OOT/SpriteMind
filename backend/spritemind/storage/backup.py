"""Verifiable Backup and Restore Engine for SpriteMind Stage 5.5.

Safely archives SQLite database catalog, asset metadata, metrics, and benchmarks.
Validates SHA-256 checksums and prevents accidental overwriting of active checkpoints.
"""
import io
import os
import json
import time
import zipfile
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

from spritemind.storage.database import StorageDatabase, DB_PATH
from spritemind.storage.classification import StorageClassifier, ProtectedResourceException


class BackupManager:
    """Creates, verifies, and restores auditable backups of SpriteMind state."""

    def __init__(self, db: Optional[StorageDatabase] = None, backup_dir: str = "data/backups"):
        self.db = db or StorageDatabase()
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _compute_sha256(data_bytes: bytes) -> str:
        return hashlib.sha256(data_bytes).hexdigest()

    def create_backup(
        self,
        backup_name: Optional[str] = None,
        include_db: bool = True,
        include_benchmarks: bool = True,
        include_assets_meta: bool = True,
    ) -> Dict[str, Any]:
        """Creates a verified ZIP backup containing database catalog and metadata."""
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        bname = backup_name or f"spritemind_backup_{timestamp_str}"
        zip_filename = f"{bname}.zip"
        zip_path = self.backup_dir / zip_filename

        bio = io.BytesIO()
        file_hashes: Dict[str, str] = {}
        files_archived = 0

        with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # 1. SQLite Database
            db_file = Path(DB_PATH)
            if include_db and db_file.exists():
                db_bytes = db_file.read_bytes()
                zf.writestr("database/spritemind.db", db_bytes)
                file_hashes["database/spritemind.db"] = self._compute_sha256(db_bytes)
                files_archived += 1

            # 2. Benchmarks and Metrics
            if include_benchmarks:
                benchmarks_dir = Path("data/benchmarks")
                if benchmarks_dir.exists():
                    for p in benchmarks_dir.glob("*.json"):
                        b_bytes = p.read_bytes()
                        arch_path = f"benchmarks/{p.name}"
                        zf.writestr(arch_path, b_bytes)
                        file_hashes[arch_path] = self._compute_sha256(b_bytes)
                        files_archived += 1

                # Root metrics files in data/
                for p in Path("data").glob("*metrics.json"):
                    m_bytes = p.read_bytes()
                    arch_path = f"metrics/{p.name}"
                    zf.writestr(arch_path, m_bytes)
                    file_hashes[arch_path] = self._compute_sha256(m_bytes)
                    files_archived += 1

            # 3. User Asset Metadata
            if include_assets_meta:
                assets_dir = Path("data/assets")
                if assets_dir.exists():
                    for p in assets_dir.glob("*.json"):
                        a_bytes = p.read_bytes()
                        arch_path = f"assets/{p.name}"
                        zf.writestr(arch_path, a_bytes)
                        file_hashes[arch_path] = self._compute_sha256(a_bytes)
                        files_archived += 1

            # 4. Write Backup Manifest
            manifest = {
                "backup_name": bname,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_files": files_archived,
                "file_hashes": file_hashes,
                "version": "0.5.5",
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        zip_bytes = bio.getvalue()
        zip_sha = self._compute_sha256(zip_bytes)

        # Write to disk
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)

        self.db.log_audit("CREATE_BACKUP", zip_path.as_posix(), "SUCCESS", f"Archived {files_archived} files ({len(zip_bytes)} bytes)")

        return {
            "status": "success",
            "backup_name": bname,
            "filename": zip_filename,
            "path": zip_path.as_posix(),
            "size_bytes": len(zip_bytes),
            "size_kb": round(len(zip_bytes) / 1024, 2),
            "sha256": zip_sha,
            "files_count": files_archived,
        }

    def verify_backup(self, zip_path_str: str) -> Dict[str, Any]:
        """Validates that a backup ZIP is intact and all archived file hashes match."""
        p = Path(zip_path_str)
        if not p.exists():
            return {"valid": False, "error": f"File '{zip_path_str}' does not exist."}

        try:
            with zipfile.ZipFile(p, "r") as zf:
                # Test CRC integrity
                bad_file = zf.testzip()
                if bad_file:
                    return {"valid": False, "error": f"Corrupted file in archive: {bad_file}"}

                if "manifest.json" not in zf.namelist():
                    return {"valid": False, "error": "Archive missing manifest.json."}

                manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
                file_hashes = manifest.get("file_hashes", {})

                for arch_path, expected_hash in file_hashes.items():
                    if arch_path not in zf.namelist():
                        return {"valid": False, "error": f"Expected file '{arch_path}' missing from archive."}
                    file_bytes = zf.read(arch_path)
                    actual_hash = self._compute_sha256(file_bytes)
                    if actual_hash != expected_hash:
                        return {"valid": False, "error": f"Hash mismatch for '{arch_path}'."}

                return {
                    "valid": True,
                    "backup_name": manifest.get("backup_name"),
                    "created_at": manifest.get("created_at"),
                    "verified_files_count": len(file_hashes),
                }
        except Exception as e:
            return {"valid": False, "error": f"Archive verification failed: {str(e)}"}

    def restore_backup(
        self,
        zip_path_str: str,
        target_dir: str = "data/restored_backup",
        confirm: bool = False,
    ) -> Dict[str, Any]:
        """Safely restores files from an archive into a specified destination directory."""
        if not confirm:
            raise ValueError("Restoration requires explicit confirm=True.")

        # Safety verification
        ver = self.verify_backup(zip_path_str)
        if not ver.get("valid"):
            raise ValueError(f"Cannot restore invalid backup: {ver.get('error')}")

        p = Path(zip_path_str)
        dest = Path(target_dir)
        dest.mkdir(parents=True, exist_ok=True)

        # Protection assertion: Never allow restoring directly over active checkpoints
        if "models/checkpoints" in dest.as_posix():
            raise ProtectedResourceException("Restoration cannot directly target active models/checkpoints.")

        restored_count = 0
        with zipfile.ZipFile(p, "r") as zf:
            for name in zf.namelist():
                if name != "manifest.json":
                    zf.extract(name, path=dest)
                    restored_count += 1

        self.db.log_audit("RESTORE_BACKUP", str(dest), "SUCCESS", f"Restored {restored_count} files from {p.name}")

        return {
            "status": "success",
            "restored_count": restored_count,
            "target_dir": str(dest),
            "source_backup": p.name,
        }

    def list_backups(self) -> List[Dict[str, Any]]:
        """Lists all existing backup archives with sizes and timestamps."""
        backups = []
        for p in sorted(self.backup_dir.glob("*.zip"), reverse=True):
            backups.append({
                "filename": p.name,
                "path": p.as_posix(),
                "size_bytes": p.stat().st_size,
                "size_kb": round(p.stat().st_size / 1024, 2),
                "modified_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
            })
        return backups
