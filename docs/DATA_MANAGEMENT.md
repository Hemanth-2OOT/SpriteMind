# SpriteMind Data Management & Lifecycle Guide

**Platform Version:** SpriteMind v0.5.5  
**Data Storage Implementation:** Hybrid Relational SQLite Catalog (`data/spritemind.db`) + Structured Filesystem Store  

---

## 1. Data Classification Hierarchy

SpriteMind organizes all data into five strict classification tiers:

```
+----------------------------------------------------------------------------------------+
|                               SpriteMind Data Hierarchy                                |
+----------------------------------------------------------------------------------------+
  ├── [Tier A] PROTECTED_TRAINING   --> data/datasets/, data/temporal_dataset_1k/
  ├── [Tier B] PROTECTED_MODEL      --> models/checkpoints/*.pt
  ├── [Tier C] PROTECTED_RESEARCH   --> docs/research/, data/benchmarks/, *metrics.json
  ├── [Tier D] USER_GENERATED       --> data/assets/, data/asset_packs/, data/atlases/
  └── [Tier E] TEMPORARY_REGENERABLE--> data/cache/, data/trash/, __pycache__/
```

### Protection Guarantees:
- **Tiers A, B, and C are IMMUTABLE.** They can **never** be deleted, overwritten, or modified through any generated-asset deletion endpoint or user interface.
- **Tiers D and E are DELETABLE.** User-generated assets, compositions, and temporary files can be trashed, restored, or permanently deleted.

---

## 2. Asset Lifecycle & Recoverable Trash

SpriteMind implements a soft-deletion pattern:

```
[User Asset: data/assets/xyz.json]
             │
             │ (Trash Request)
             ▼
[Protected Boundary Check] ──(Pass)──> [Move to data/trash/trash_xxx_xyz.json]
             │                                       │
     (Failed if Protected)                           ├── [Restore Request] ──> [Restored to data/assets/]
             │                                       │
[ProtectedResourceException Raised]                  └── [Permanent Delete (confirm=True)] ──> [Unlinked]
```

### Key API & CLI Operations:
- **Trash Asset:** `AssetLifecycleManager().trash_asset(asset_id)` moves asset to `data/trash/` and marks `in_trash = 1` in SQLite.
- **Restore Asset:** `AssetLifecycleManager().restore_asset(trash_id)` restores asset to `data/assets/` and marks `in_trash = 0`.
- **Permanent Delete:** `AssetLifecycleManager().permanent_delete(trash_id, confirm=True)` unlinks from `data/trash/` permanently.
- **Empty Trash:** `AssetLifecycleManager().empty_trash(confirm=True)` purges only trashed files, leaving active assets and all models intact.

---

## 3. Storage Database Catalog (`spritemind.db`)

SpriteMind uses SQLite (`data/spritemind.db`) as an auditable relational registry.

### Core Tables:
1. `storage_registry`: Tracks all filesystem paths, SHA-256 hashes, sizes, and protection flags.
2. `assets`: Indexes user-generated sprite programs, frame counts, pixel counts, and previews.
3. `checkpoints`: Records model weights, parameter counts, architectures, and verification hashes.
4. `datasets`: Records training dataset names, sample counts, split ratios, and record paths.
5. `benchmarks`: Indexes historical benchmark results, evaluation dates, and summary scores.
6. `trash`: Tracks soft-deleted files, original locations, and deletion reasons.
7. `audit_log`: Immutable ledger recording every storage operation (`SYNC`, `TRASH`, `RESTORE`, `BACKUP`, `CLEANUP`).

---

## 4. Backup & Disaster Recovery

Backups are created as self-contained ZIP archives containing the SQLite database, benchmark results, metrics, and user metadata:

- **Create Backup:**
  ```python
  from spritemind.storage.backup import BackupManager
  bm = BackupManager()
  res = bm.create_backup()
  # Created: data/backups/spritemind_backup_YYYYMMDD_HHMMSS.zip
  ```
- **Verify Backup:**
  ```python
  status = bm.verify_backup("data/backups/spritemind_backup_xxx.zip")
  assert status["valid"] is True
  ```
- **Restore Backup:**
  ```python
  # Restores into isolated directory with hash validation
  bm.restore_backup("data/backups/spritemind_backup_xxx.zip", target_dir="data/restored_backup", confirm=True)
  ```
- **Safety Rule:** Restoration can never directly target `models/checkpoints/` without explicit manual intervention.
