"""Asset Lifecycle, Trash, and Protected Deletion Engine for SpriteMind Stage 5.5.

Guarantees that user-generated assets can be safely trashed and restored
WITHOUT EVER deleting, corrupting, or modifying models, checkpoints, or datasets.
"""
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

from spritemind.storage.classification import (
    DataClassification,
    StorageClassifier,
    ProtectedResourceException,
    IMMUTABLE_CLASSIFICATIONS,
)
from spritemind.storage.database import StorageDatabase, DB_PATH


class AssetLifecycleManager:
    """Manages soft-deletion (trash), restoration, and permanent removal of user-generated assets."""

    def __init__(self, db: Optional[StorageDatabase] = None, trash_dir: str = "data/trash"):
        self.db = db or StorageDatabase()
        self.trash_dir = Path(trash_dir)
        self.trash_dir.mkdir(parents=True, exist_ok=True)

    def trash_asset(self, asset_id: str, reason: str = "User deletion request") -> Dict[str, Any]:
        """Moves a user-generated asset to recoverable trash. Strictly blocks protected files."""
        # Non-negotiable immutability gate: check input ID/path first
        StorageClassifier.assert_deletable(asset_id)

        # Find asset file path
        asset_file = Path(f"data/assets/{asset_id}.json")
        if not asset_file.exists():
            # Check if asset_id is already a direct path
            p_check = Path(asset_id)
            if p_check.exists():
                asset_file = p_check
            else:
                raise FileNotFoundError(f"Asset '{asset_id}' not found on disk.")

        rel_path = asset_file.as_posix()

        # Non-negotiable immutability gate on resolved path
        StorageClassifier.assert_deletable(rel_path)

        # Move to trash directory
        trash_id = f"trash_{uuid.uuid4().hex[:8]}"
        dest_filename = f"{trash_id}_{asset_file.name}"
        dest_path = self.trash_dir / dest_filename
        size_bytes = asset_file.stat().st_size
        classification = StorageClassifier.classify_path(rel_path)

        shutil.move(str(asset_file), str(dest_path))

        # Record in SQLite database
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.db._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
            INSERT INTO trash (trash_id, asset_id, original_path, trash_path, size_bytes, classification, deleted_at, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (trash_id, asset_id, rel_path, dest_path.as_posix(), size_bytes, classification.value, now_str, reason))

            # Mark in registry and assets
            cur.execute("UPDATE storage_registry SET in_trash = 1 WHERE path = ?;", (rel_path,))
            cur.execute("UPDATE assets SET in_trash = 1 WHERE asset_id = ?;", (asset_id,))
            conn.commit()

        self.db.log_audit("TRASH_ASSET", rel_path, "SUCCESS", f"Moved to {dest_path.as_posix()} (ID: {trash_id})")

        return {
            "status": "trashed",
            "trash_id": trash_id,
            "asset_id": asset_id,
            "original_path": rel_path,
            "trash_path": dest_path.as_posix(),
            "size_bytes": size_bytes,
        }

    def restore_asset(self, trash_id: str) -> Dict[str, Any]:
        """Restores a trashed asset back to its original location."""
        with self.db._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM trash WHERE trash_id = ? OR asset_id = ?;", (trash_id, trash_id))
            row = cur.fetchone()
            if not row:
                raise FileNotFoundError(f"Trash record '{trash_id}' not found.")

            t_record = dict(row)
            trash_path = Path(t_record["trash_path"])
            orig_path = Path(t_record["original_path"])

            if not trash_path.exists():
                raise FileNotFoundError(f"Physical trash file '{trash_path}' is missing.")

            # Ensure parent exists and restore
            orig_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(trash_path), str(orig_path))

            # Remove from trash table and unmark in_trash
            cur.execute("DELETE FROM trash WHERE trash_id = ?;", (t_record["trash_id"],))
            cur.execute("UPDATE storage_registry SET in_trash = 0 WHERE path = ?;", (t_record["original_path"],))
            cur.execute("UPDATE assets SET in_trash = 0 WHERE asset_id = ?;", (t_record["asset_id"],))
            conn.commit()

        self.db.log_audit("RESTORE_ASSET", t_record["original_path"], "SUCCESS", f"Restored from {trash_path.as_posix()}")

        return {
            "status": "restored",
            "trash_id": t_record["trash_id"],
            "asset_id": t_record["asset_id"],
            "restored_path": t_record["original_path"],
        }

    def permanent_delete(self, trash_id: str, confirm: bool = False) -> Dict[str, Any]:
        """Permanently deletes a file that is ALREADY in trash. Requires confirmation."""
        if not confirm:
            raise ValueError("Permanent deletion requires explicit confirm=True.")

        with self.db._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM trash WHERE trash_id = ? OR asset_id = ?;", (trash_id, trash_id))
            row = cur.fetchone()
            if not row:
                raise FileNotFoundError(f"Trash record '{trash_id}' not found.")

            t_record = dict(row)
            trash_path = Path(t_record["trash_path"])

            # Safety assertion: Must be in trash directory and classified as deletable
            if not trash_path.as_posix().startswith("data/trash"):
                raise ProtectedResourceException(f"Security breach: '{trash_path}' is not inside data/trash!")

            StorageClassifier.assert_deletable(t_record["original_path"])

            # Delete physical file
            if trash_path.exists():
                trash_path.unlink()

            # Remove from registry and trash table
            cur.execute("DELETE FROM trash WHERE trash_id = ?;", (t_record["trash_id"],))
            cur.execute("DELETE FROM storage_registry WHERE path = ?;", (t_record["original_path"],))
            cur.execute("DELETE FROM assets WHERE asset_id = ?;", (t_record["asset_id"],))
            conn.commit()

        self.db.log_audit("PERMANENT_DELETE", t_record["original_path"], "SUCCESS", f"Permanently deleted trash item {trash_id}")

        return {
            "status": "permanently_deleted",
            "trash_id": t_record["trash_id"],
            "asset_id": t_record["asset_id"],
            "freed_bytes": t_record["size_bytes"],
        }

    def empty_trash(self, confirm: bool = False) -> Dict[str, Any]:
        """Empties all files currently in trash. Requires confirmation."""
        if not confirm:
            raise ValueError("Emptying trash requires explicit confirm=True.")

        freed_bytes = 0
        deleted_count = 0

        with self.db._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM trash;")
            rows = [dict(r) for r in cur.fetchall()]

            for r in rows:
                t_path = Path(r["trash_path"])
                if t_path.exists():
                    freed_bytes += t_path.stat().st_size
                    t_path.unlink()
                deleted_count += 1

                cur.execute("DELETE FROM trash WHERE trash_id = ?;", (r["trash_id"],))
                cur.execute("DELETE FROM storage_registry WHERE path = ?;", (r["original_path"],))
                cur.execute("DELETE FROM assets WHERE asset_id = ?;", (r["asset_id"],))

            conn.commit()

        self.db.log_audit("EMPTY_TRASH", "data/trash", "SUCCESS", f"Emptied {deleted_count} items ({freed_bytes} bytes freed)")

        return {
            "status": "trash_emptied",
            "deleted_count": deleted_count,
            "freed_bytes": freed_bytes,
            "freed_kb": round(freed_bytes / 1024, 2),
        }

    def list_trash(self) -> List[Dict[str, Any]]:
        """Returns all items currently in trash."""
        with self.db._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM trash ORDER BY deleted_at DESC;")
            return [dict(r) for r in cur.fetchall()]

    def estimate_recoverable_space(self) -> Dict[str, Any]:
        """Estimates disk space that would be recovered if trash is emptied."""
        items = self.list_trash()
        total_bytes = sum(it.get("size_bytes", 0) for it in items)
        return {
            "items_count": len(items),
            "recoverable_bytes": total_bytes,
            "recoverable_kb": round(total_bytes / 1024, 2),
            "recoverable_mb": round(total_bytes / (1024 * 1024), 2),
        }
