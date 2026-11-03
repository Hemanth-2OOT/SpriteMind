# SpriteMind Storage Architecture & Schema Reference

**Document Version:** 1.0.0  
**Storage Engine:** Hybrid SQLite (`data/spritemind.db`) + Flat-File Document Storage  
**Primary Database Path:** `data/spritemind.db`  

---

## 1. Storage Layout Map

```
MindSprit/
├── data/
│   ├── spritemind.db           # SQLite relational storage catalog & audit ledger
│   ├── assets/                 # User-generated assets (.json with PixelScript & previews)
│   ├── asset_packs/            # Exported game-ready ZIP asset packages
│   ├── benchmarks/             # Raw JSON benchmark result files (Protected Research)
│   ├── datasets/               # 10,000-sample supervised action datasets (Protected Training)
│   ├── temporal_dataset_1k/    # 600-sequence kinematic temporal dataset (Protected Training)
│   ├── trash/                  # Soft-deleted assets pending permanent deletion or restore
│   ├── backups/                # Verified ZIP archives of SQLite catalog & metadata
│   └── *metrics.json           # Real-time training loss and convergence metrics
├── models/
│   └── checkpoints/            # PyTorch .pt model weights (Protected Models)
│       ├── svan_checkpoint_best.pt      (1.74M parameters)
│       ├── visual_critic_best.pt        (368K parameters)
│       ├── temporal_svan_best.pt        (426K parameters)
│       └── socket_predictor_best.pt     (619K parameters)
└── docs/
    └── research/               # Technical research reports & benchmark indexes
```

---

## 2. Relational Database Schema (`data/spritemind.db`)

### 2.1 Table: `storage_registry`
Tracks every registered file on disk and its immutability status.

```sql
CREATE TABLE storage_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,             -- Relative POSIX path (e.g., 'data/assets/xyz.json')
    filename TEXT NOT NULL,                -- Base filename
    classification TEXT NOT NULL,          -- protected_model | protected_training | ...
    category TEXT NOT NULL,                -- models | data | docs | etc.
    size_bytes INTEGER NOT NULL,           -- File size on disk
    sha256 TEXT,                           -- SHA-256 integrity hash
    is_protected INTEGER NOT NULL DEFAULT 0,-- 1 = Protected (Deletion Forbidden), 0 = Deletable
    in_trash INTEGER NOT NULL DEFAULT 0,    -- 1 = In Trash, 0 = Active
    created_at TEXT,
    modified_at TEXT
);
```

### 2.2 Table: `assets`
Indexes user-created and saved pixel-art game assets.

```sql
CREATE TABLE assets (
    asset_id TEXT PRIMARY KEY,             -- Unique identifier (e.g., 'asset_271fb459')
    name TEXT NOT NULL,                    -- Display name
    category TEXT NOT NULL,                -- knight | zombie | weapon | etc.
    tags TEXT,                             -- JSON array of tag strings
    frame_count INTEGER NOT NULL DEFAULT 1,-- Number of animation keyframes
    pixel_count INTEGER NOT NULL DEFAULT 0,-- Rendered opaque pixel count
    script_text TEXT,                      -- PixelScript program
    preview_base64 TEXT,                   -- Base64 PNG data URI
    metadata_json TEXT,                    -- JSON dictionary of custom attributes
    in_trash INTEGER NOT NULL DEFAULT 0,
    created_at TEXT,
    modified_at TEXT
);
```

### 2.3 Table: `checkpoints`
Indexes model checkpoints and verification signatures.

```sql
CREATE TABLE checkpoints (
    checkpoint_name TEXT PRIMARY KEY,      -- File name (e.g., 'temporal_svan_best.pt')
    architecture TEXT NOT NULL,            -- Architecture type
    parameter_count INTEGER NOT NULL,      -- Total trainable parameters
    file_path TEXT NOT NULL,               -- Relative path
    size_bytes INTEGER NOT NULL,           -- Byte size on disk
    sha256 TEXT NOT NULL,                  -- SHA-256 cryptographic checksum
    is_protected INTEGER NOT NULL DEFAULT 1,-- Always 1
    metadata_json TEXT,                    -- Epoch, loss, training metrics
    created_at TEXT
);
```

### 2.4 Table: `trash`
Maintains recoverable soft-deleted files.

```sql
CREATE TABLE trash (
    trash_id TEXT PRIMARY KEY,             -- Unique trash ticket (e.g., 'trash_a1b2c3d4')
    asset_id TEXT,                         -- Original asset ID
    original_path TEXT NOT NULL,           -- Path before trashing
    trash_path TEXT NOT NULL,              -- Path inside data/trash/
    size_bytes INTEGER NOT NULL,           -- Space occupied
    classification TEXT NOT NULL,          -- user_generated | temporary_regenerable
    deleted_at TEXT NOT NULL,              -- Timestamp of trashing
    reason TEXT                            -- User justification or system event
);
```

### 2.5 Table: `audit_log`
Immutable record of all filesystem and database mutations.

```sql
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    action TEXT NOT NULL,                  -- TRASH_ASSET | RESTORE_ASSET | CLEANUP | etc.
    target TEXT NOT NULL,                  -- Target path or resource
    status TEXT NOT NULL,                  -- SUCCESS | FAILED
    details TEXT
);
```

---

## 3. Immutability Enforcement Mechanics

Every destructive operation passes through `StorageClassifier.assert_deletable(path)`:
1. If the path matches any `PROTECTED_*` classification, the operation halts immediately and raises `ProtectedResourceException`.
2. Even raw SQL execution via API endpoints is disabled; operations use parameterized queries strictly mapped to whitelisted tables.
3. Model checkpoints and dataset directories cannot be passed to the trash or permanent deletion endpoints under any circumstances.
