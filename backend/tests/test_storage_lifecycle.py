"""Comprehensive Test Suite for SpriteMind Stage 5.5 Storage, Lifecycle, and Immutability Protection.

Guarantees:
- Strict classification into protected vs deletable tiers.
- Non-negotiable deletion barrier for model weights and training datasets.
- Recoverable trash, restore, and permanent deletion lifecycle for user assets.
- Dry-run cleanup and verified backup with SHA-256 checksums.
- Absolute byte-for-byte immutability of checkpoints and datasets during asset deletion.
"""
import os
import json
import shutil
import hashlib
from pathlib import Path
import pytest

from spritemind.storage.classification import (
    DataClassification,
    StorageClassifier,
    ProtectedResourceException,
    IMMUTABLE_CLASSIFICATIONS,
    DELETABLE_CLASSIFICATIONS,
)
from spritemind.storage.database import StorageDatabase
from spritemind.storage.lifecycle import AssetLifecycleManager
from spritemind.storage.cleanup import StorageCleaner
from spritemind.storage.backup import BackupManager


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestStorageClassification:
    """Tests classification logic across all directories and file types."""

    def test_checkpoint_classification(self):
        assert StorageClassifier.classify_path("models/checkpoints/svan_checkpoint_best.pt") == DataClassification.PROTECTED_MODEL
        assert StorageClassifier.classify_path("models/checkpoints/temporal_svan_best.pt") == DataClassification.PROTECTED_MODEL

    def test_training_dataset_classification(self):
        assert StorageClassifier.classify_path("data/datasets/generalization_10k/records.jsonl") == DataClassification.PROTECTED_TRAINING
        assert StorageClassifier.classify_path("data/temporal_dataset_1k/train_sequences.jsonl") == DataClassification.PROTECTED_TRAINING

    def test_research_classification(self):
        assert StorageClassifier.classify_path("docs/research/STAGE_4_TEMPORAL_ANIMATION_REPORT.md") == DataClassification.PROTECTED_RESEARCH
        assert StorageClassifier.classify_path("data/benchmarks/composition_benchmark_results.json") == DataClassification.PROTECTED_RESEARCH

    def test_user_generated_classification(self):
        assert StorageClassifier.classify_path("data/assets/asset_123.json") == DataClassification.USER_GENERATED
        assert StorageClassifier.classify_path("data/renders/sprite_idle.png") == DataClassification.USER_GENERATED

    def test_temporary_regenerable_classification(self):
        assert StorageClassifier.classify_path("backend/spritemind/__pycache__/app.cpython-311.pyc") == DataClassification.TEMPORARY_REGENERABLE
        assert StorageClassifier.classify_path(".pytest_cache/v/cache/lastfailed") == DataClassification.TEMPORARY_REGENERABLE

    def test_protection_barrier_raises_exception_on_checkpoints(self):
        with pytest.raises(ProtectedResourceException):
            StorageClassifier.assert_deletable("models/checkpoints/svan_checkpoint_best.pt")

    def test_protection_barrier_raises_exception_on_datasets(self):
        with pytest.raises(ProtectedResourceException):
            StorageClassifier.assert_deletable("data/datasets/generalization_10k/records.jsonl")

    def test_protection_barrier_raises_exception_on_research(self):
        with pytest.raises(ProtectedResourceException):
            StorageClassifier.assert_deletable("docs/research/BENCHMARK_INDEX.md")

    def test_protection_barrier_allows_user_assets_and_temp(self):
        assert StorageClassifier.assert_deletable("data/assets/my_custom_knight.json") is True
        assert StorageClassifier.assert_deletable("data/trash/temp_sample.png") is True


class TestStorageDatabase:
    """Tests SQLite catalog indexing, querying, and dashboard reporting."""

    def test_database_initialization_and_sync(self):
        db = StorageDatabase()
        sync_res = db.sync_from_disk()
        assert sync_res["files_scanned"] > 50

        dashboard = db.get_storage_dashboard()
        assert dashboard["status"] == "healthy"
        assert dashboard["total_files_indexed"] >= sync_res["files_scanned"]
        assert dashboard["total_disk_mb"] > 0
        assert "PROTECTED_MODEL" in dashboard["breakdown"]
        assert "PROTECTED_TRAINING" in dashboard["breakdown"]

    def test_list_records_and_search(self):
        db = StorageDatabase()
        records = db.list_table_records("storage_registry", limit=10)
        assert len(records) > 0
        assert "sha256" in records[0]

        # Search specifically for checkpoint
        cp_records = db.list_table_records("storage_registry", search_query="svan_checkpoint_best")
        assert len(cp_records) >= 1
        assert cp_records[0]["classification"] in ("protected_model", "PROTECTED_MODEL")


class TestAssetLifecycle:
    """Tests recoverable trash, restore, and permanent deletion workflows."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.db = StorageDatabase()
        self.lifecycle = AssetLifecycleManager(db=self.db)
        self.test_asset_dir = Path("data/assets")
        self.test_asset_dir.mkdir(parents=True, exist_ok=True)
        self.dummy_asset_id = "test_asset_lifecycle_dummy"
        self.dummy_asset_path = self.test_asset_dir / f"{self.dummy_asset_id}.json"

        # Create dummy user asset
        with open(self.dummy_asset_path, "w", encoding="utf-8") as f:
            json.dump({"id": self.dummy_asset_id, "name": "Dummy Goblin", "type": "character"}, f)

        # Index into DB
        self.db.sync_from_disk()

        yield

        # Cleanup if left
        if self.dummy_asset_path.exists():
            self.dummy_asset_path.unlink()
        for p in Path("data/trash").glob(f"*{self.dummy_asset_id}*"):
            if p.exists():
                p.unlink()

    def test_trash_and_restore_cycle(self):
        # 1. Trash asset
        trash_res = self.lifecycle.trash_asset(self.dummy_asset_id, reason="Testing lifecycle")
        assert trash_res["status"] == "trashed"
        trash_id = trash_res["trash_id"]
        trash_path = Path(trash_res["trash_path"])

        assert not self.dummy_asset_path.exists(), "Original file should have been moved"
        assert trash_path.exists(), "File should now reside in data/trash"

        # Verify database record
        trash_list = self.lifecycle.list_trash()
        assert any(t["trash_id"] == trash_id for t in trash_list)

        # 2. Restore asset
        restore_res = self.lifecycle.restore_asset(trash_id)
        assert restore_res["status"] == "restored"
        assert self.dummy_asset_path.exists(), "Original file should be restored"
        assert not trash_path.exists(), "File should be gone from data/trash"

    def test_permanent_delete_requires_confirmation(self):
        trash_res = self.lifecycle.trash_asset(self.dummy_asset_id)
        trash_id = trash_res["trash_id"]

        with pytest.raises(ValueError, match="confirm=True"):
            self.lifecycle.permanent_delete(trash_id, confirm=False)

        # Successfully permanently delete with confirm=True
        del_res = self.lifecycle.permanent_delete(trash_id, confirm=True)
        assert del_res["status"] == "permanently_deleted"
        assert not Path(trash_res["trash_path"]).exists()

    def test_protected_files_cannot_be_trashed(self):
        with pytest.raises(ProtectedResourceException):
            self.lifecycle.trash_asset("models/checkpoints/svan_checkpoint_best.pt")

        with pytest.raises(ProtectedResourceException):
            self.lifecycle.trash_asset("data/datasets/generalization_10k/records.jsonl")


class TestStorageCleanerAndBackup:
    """Tests dry-run scanner, cleanup execution, and backup checksum verification."""

    def test_cleaner_dry_run_and_execution(self):
        cleaner = StorageCleaner()
        scan_res = cleaner.scan()
        assert "candidate_count" in scan_res
        assert "reclaimable_bytes" in scan_res

        with pytest.raises(ValueError, match="confirm=True"):
            cleaner.execute(confirm=False)

    def test_backup_create_verify_and_restore(self, tmp_path):
        backup_mgr = BackupManager()
        backup_res = backup_mgr.create_backup("test_automated_backup")
        assert backup_res["status"] == "success"
        assert Path(backup_res["path"]).exists()

        # Verify integrity
        ver_res = backup_mgr.verify_backup(backup_res["path"])
        assert ver_res["valid"] is True
        assert ver_res["verified_files_count"] > 0

        # Test restore
        restore_dir = tmp_path / "restored_backup"
        restore_res = backup_mgr.restore_backup(backup_res["path"], target_dir=str(restore_dir), confirm=True)
        assert restore_res["status"] == "success"
        assert (restore_dir / "database" / "spritemind.db").exists()

        # Cleanup backup archive
        Path(backup_res["path"]).unlink()


class TestCriticalImmutabilityRegression:
    """CRITICAL TEST: Ensures deleting generated assets NEVER modifies models or training datasets."""

    def test_checkpoints_and_datasets_hashes_remain_identical_after_asset_deletion(self):
        # 1. Identify and hash all checkpoints
        checkpoint_dir = Path("models/checkpoints")
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoint_files) >= 5, f"Expected at least 5 checkpoints, found {len(checkpoint_files)}"

        initial_checkpoint_hashes = {p.as_posix(): compute_file_sha256(p) for p in checkpoint_files}

        # 2. Identify and hash all training datasets
        dataset_files = list(Path("data").glob("**/*.jsonl"))
        assert len(dataset_files) >= 1, "Expected at least 1 dataset file"

        initial_dataset_hashes = {p.as_posix(): compute_file_sha256(p) for p in dataset_files}

        # 3. Create, trash, and permanently delete a dummy generated user asset
        lifecycle = AssetLifecycleManager()
        dummy_path = Path("data/assets/immutability_test_asset.json")
        with open(dummy_path, "w", encoding="utf-8") as f:
            json.dump({"id": "immutability_test_asset", "frames": [{"pixels": [[1, 2], [3, 4]]}]}, f)

        trash_res = lifecycle.trash_asset("immutability_test_asset", reason="Immutability regression test")
        lifecycle.permanent_delete(trash_res["trash_id"], confirm=True)

        assert not dummy_path.exists()
        assert not Path(trash_res["trash_path"]).exists()

        # 4. Recompute all checkpoint hashes and assert bit-for-bit parity
        for cp_path, initial_hash in initial_checkpoint_hashes.items():
            current_hash = compute_file_sha256(Path(cp_path))
            assert current_hash == initial_hash, (
                f"REGRESSION VIOLATION: Checkpoint '{cp_path}' was modified during asset deletion! "
                f"Initial SHA-256: {initial_hash}, Current SHA-256: {current_hash}"
            )

        # 5. Recompute all dataset hashes and assert bit-for-bit parity
        for ds_path, initial_hash in initial_dataset_hashes.items():
            current_hash = compute_file_sha256(Path(ds_path))
            assert current_hash == initial_hash, (
                f"REGRESSION VIOLATION: Dataset '{ds_path}' was modified during asset deletion! "
                f"Initial SHA-256: {initial_hash}, Current SHA-256: {current_hash}"
            )
