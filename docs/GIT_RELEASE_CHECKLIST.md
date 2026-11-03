# SpriteMind GitHub Release & Repository Preparation Checklist

**Project:** SpriteMind — Local AI Pixel Art Generator & Research Platform  
**Target Repository:** `github.com/<owner>/MindSprit`  
**Current Branch:** `main` (clean, uncommitted)  

---

## 1. File Size Audit & Git LFS Strategy

```
+------------------------------------------------+-----------+----------------------+--------------------+
| File / Directory Path                          | Size      | Git Status           | Recommendation     |
+------------------------------------------------+-----------+----------------------+--------------------+
| data/datasets/generalization_10k/records.jsonl | 47.90 MB  | Standard Git / LFS   | Track via Git LFS  |
| models/checkpoints/svan_checkpoint_best.pt     |  6.61 MB  | Standard Git (<10MB) | Commit directly    |
| models/checkpoints/svan_trained_baseline.pt    |  6.60 MB  | Standard Git (<10MB) | Commit directly    |
| models/checkpoints/socket_predictor_best.pt    |  2.38 MB  | Standard Git (<10MB) | Commit directly    |
| models/checkpoints/temporal_svan_best.pt       |  1.65 MB  | Standard Git (<10MB) | Commit directly    |
| models/checkpoints/visual_critic_best.pt       |  1.43 MB  | Standard Git (<10MB) | Commit directly    |
| data/temporal_dataset_1k/train_sequences.jsonl |  4.14 MB  | Standard Git (<10MB) | Commit directly    |
| data/benchmarks/*.json                         |  0.46 MB  | Standard Git         | Commit directly    |
| frontend/dist/app.js                           |  0.11 MB  | Standard Git         | Commit directly    |
+------------------------------------------------+-----------+----------------------+--------------------+
Total Repository Size: ~74.5 MB
```

### GitHub Limits & Rules:
- **Maximum individual file size:** 100 MB (Hard limit on GitHub).
- **File size warning threshold:** 50 MB.
- `records.jsonl` is **$47.90\text{ MB}$**, which is within the 100 MB limit but close to the 50 MB warning.
- **Git LFS Recommendation:**
  ```bash
  # Enable Git LFS for large datasets
  git lfs install
  git lfs track "*.jsonl"
  git lfs track "*.pt"
  git add .gitattributes
  ```

---

## 2. Dataset Reproduction Strategy (Zero Download Dependency)

If users clone without downloading large `.jsonl` datasets, SpriteMind provides deterministic, one-click dataset generators:

1. **Reproduce Generalization 10k Dataset:**
   ```bash
   python -c "from spritemind.dataset.storage import DatasetManager; DatasetManager().generate_dataset(name='generalization_10k', total_samples=10000, seed=42)"
   ```
2. **Reproduce Temporal 1k Dataset:**
   ```bash
   python -c "from spritemind.dataset.temporal_dataset import TemporalDatasetGenerator; TemporalDatasetGenerator.generate_dataset(total_sequences=600, seed=42)"
   ```
3. **Reproduce Composition Dataset:**
   ```bash
   python -c "from spritemind.dataset.composition_dataset import CompositionDatasetBuilder; CompositionDatasetBuilder.generate_samples(total_samples=300, seed=777)"
   ```

---

## 3. Security & Cleanliness Verification

- [x] **No Private Credentials:** Verified zero API keys, secrets, or cloud tokens. SpriteMind operates 100% locally.
- [x] **No Operating System Clutter:** `Thumbs.db`, `.DS_Store`, `Desktop.ini` excluded via `.gitignore`.
- [x] **No Cache Clutter:** `__pycache__`, `.pytest_cache`, `vite_test.json`, `*.pyc` excluded via `.gitignore`.
- [x] **No Disposable Trash Committed:** `data/trash/`, `data/temp/`, `data/backups/` excluded via `.gitignore`.
- [x] **Verified Checkpoint Integrity:** All 5 model checkpoints verified functional and sha256-hashed.
- [x] **Verified Frontend Build:** `frontend/dist/app.js` bundled and syntax-verified (node exit code 0).
- [x] **Passing Backend Tests:** All 33 pytest suites passing cleanly (100% pass rate).

---

## 4. Manual Git Release Instructions

When ready to publish to GitHub:

```bash
# 1. Initialize and review status
git status

# 2. Stage verified files
git add .gitignore README.md backend/ docs/ examples/ frontend/ models/ scripts/ start.ps1 data/benchmarks/ data/asset_packs/ data/*.json

# 3. If using Git LFS for datasets:
git lfs install
git lfs track "*.jsonl"
git add .gitattributes data/datasets/ data/temporal_dataset_1k/

# 4. Commit
git commit -m "feat(spritemind): Initial release through Stage 5.5 (Game-Ready Pixel Art Platform)"

# 5. Add remote and push
git remote add origin https://github.com/<your-username>/MindSprit.git
git branch -M main
git push -u origin main
```
