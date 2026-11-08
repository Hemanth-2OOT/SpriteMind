# Final release readiness

Date: 2026-11-05. Scope: repository audit, cleanup, git verification and end-to-end
test readiness. This was a maintenance pass — **no research feature, model, dataset,
benchmark or production path was changed**. Evidence for the experiments behind the
current numbers stays in `docs/research/` and `data/benchmarks/`.

## 1. Repository audit

- 455 tracked files, 1.8 GB working tree (most of it untracked, regenerable corpora).
- Layout: `backend/` (130 files, 3.0 MB), `frontend/` (27 files + `node_modules`),
  `data/` (208 tracked research/benchmark files), `docs/` (36 files), `models/` (9 checkpoints, 51 MB),
  `scripts/` (36 files), `examples/` (1 file).
- Largest tracked blobs: `svan_v5_colour.ep1.pt`, `svan_v4_palette_ep2.pt`,
  `svan_checkpoint_best.pt` (9.1 MB each), `svan_trained_baseline.pt` (6.7 MB),
  `data/temporal_dataset_1k/train_sequences.jsonl` (4.2 MB). Nothing near GitHub's 100 MB limit,
  so no Git LFS is needed.
- Lint: `ruff check backend scripts` → **All checks passed** (CI runs the same command).
  `ruff format --check` reports 144 files it would reformat; the project has never been
  formatter-normalised and `check.sh`/CI do not enforce it, so reformatting was deliberately
  **not** done here (it would bury the real diff and touch unrelated modules).
- Secrets: `git grep` for GitHub/OpenAI/AWS/private-key patterns over the whole tree → no hits.
  Runtime state (`data/spritemind.db`, `data/history.sqlite`) is untracked and was not committed.

## 2. Cleanup performed

Removed (regenerable caches only, all gitignored — nothing tracked was deleted):

| Path | Reason |
| --- | --- |
| `__pycache__/` (all trees, incl. `examples/deadzone_integration/`) | Python bytecode, regenerated on import |
| `*.pyc` | same |
| `.pytest_cache/`, `backend/.pytest_cache/` | pytest scratch |
| `.ruff_cache/`, `backend/.ruff_cache/` | ruff scratch |
| `.coverage`, `backend/.coverage` | coverage artefacts |
| `frontend/.vite` | build cache |

Rewritten (was inaccurate, now matches the repository):

- `docs/GIT_RELEASE_CHECKLIST.md` — the old version named a non-existent target repository
  (`MindSprit`), referenced a 47.9 MB `generalization_10k/records.jsonl` that is not in the tree,
  claimed the working branch was `main` with uncommitted changes, listed 5 checkpoints (9 now),
  and described a Git-LFS plan that is not needed. Replaced with an accurate checklist that
  links back to this document.

No tests were removed, no source file was deleted, and no dataset, checkpoint, benchmark result,
report, golden image or user asset was touched. `data/trash/` and `data/temp/` were already empty.

## 3. Preserved research artefacts (explicitly kept)

- `models/checkpoints/` — all 9 `.pt` files. sha256 of every file compared before and after this
  pass: **all unchanged**, including production `svan_checkpoint_best.pt`
  (`dbe663ed…327b45ec`).
- `data/benchmarks/` (47 files incl. `benchmark_results.json`, R12/R13 runs), `data/reports/`,
  `data/splits/heldout_v1.json`, `data/datasets/*/metadata.json`, `docs/research/`,
  `docs/CANDIDATE_DIVERSITY.md`, `docs/COVERAGE_RERANK.md`, `docs/COLOUR_RECOVERY.md`,
  `docs/RETRAIN_V4.md`, `docs/IMPLEMENTATION_AUDIT.md`.
- Unreferenced-but-working utilities kept rather than deleted: `scripts/compare_checkpoints.py`,
  `scripts/make_animation_preview.py`, `scripts/audit_storage.py`.

## 4. Git verification

Every local branch head was compared with `git ls-remote --heads origin`; all match.

| Branch | HEAD | Remote | PR | Base | State |
| --- | --- | --- | --- | --- | --- |
| `main` | `e64daeb` | same | #26 (merged) | — | merged |
| `lemon/audit-r4r5r6` | `675079b` | same | #27 | `main` | open |
| `lemon/r10-retrain-v4` | `e7bc2b1` | same | #28 | `lemon/audit-r4r5r6` | open |
| `lemon/r11-colour-recovery` | `f794dd4` | same | #29 | `lemon/r10-retrain-v4` | open |
| `lemon/r12-coverage-rerank` | `3668a81` | same | #30 | `lemon/r11-colour-recovery` | open |
| `lemon/r13-candidate-diversity` | `a35c0b1` (+ this commit, `2d4c4fa`) | same | #31 | `lemon/r12-coverage-rerank` | open |
| `lemon/category-report-card` | `be509a0` | same | — | — | no PR |
| `lemon/element-aura` | `6b89075` | same | — | — | no PR, not an ancestor of `main` |
| `lemon/onion-skin-export` | `6f827c0` | same | — | — | no PR (merged as #16 on `main`) |
| `lemon/prompt-history` | `9c50a03` | same | — | — | no PR |
| `lemon/sampler-controls` | `2c0eb2e` | same | — | — | no PR |

No PR was merged, closed or opened during this pass. Nothing was force-pushed and no history
was rewritten. Remote-tracking was repaired locally: `remote.origin.fetch` had been narrowed to
`main` only, which made every other branch look un-pushed; it was widened to
`+refs/heads/*:refs/remotes/origin/*` (local git config only, no repository content change).

**Branch destination for this commit:** the only branch that is both the tip of the open PR stack
and free of other in-flight work is `lemon/r13-candidate-diversity` (PR #31). The maintenance
commit was therefore appended there rather than to `main`, so the release paperwork travels with
the branch it describes and `main` is untouched. Say the word if it should instead go to `main`.

## 5. Test results

- `pytest backend/tests` — **314 tests, all passing**, coverage **73.4 %** (floor 71 %;
  73.37 % and 73.44 % on two consecutive runs, 7548 statements, 2214 branches). Suite unchanged from the R13 commit: no test was removed,
  skipped or weakened.
- `ruff check backend scripts` — clean.
- `cd frontend && npm run build` (`tsc --noEmit` + `vite build`) — clean; 1485 modules,
  `index.html` 1.18 kB, CSS 36.26 kB, JS 340.84 kB (gzip 85.74 kB).
- `python scripts/update_readme_metrics.py --check` — up to date.
- Runner environment: `/tmp/sm-venv` (Python 3.11.16, torch 2.15.0+cpu, CPU-only container)
  because this workspace has no GPU and no system torch.

## 6. End-to-end smoke tests (live server, port 7860)

Backend started with `SPRITEMIND_THREADS=2 python backend/run_server.py`, which serves the built
frontend from `frontend/dist`.

| Workflow | Result |
| --- | --- |
| `GET /api/health` | 200 healthy, torch 2.15.0+cpu, CPU device |
| `GET /api/models` | 9 checkpoints listed; lazy `active_checkpoint: null` before first neural call |
| `POST /api/generate` default engine | 200, `candidate_source: procedural_baseline`, 7.3 ms |
| `POST /api/generate` `engine=trained_model` | 200, `candidate_source: neural`, 440.9 ms, valid PixelScript (0 invalid actions, 224 px) |
| Same call after load | `/api/models` reports `active_checkpoint: svan_checkpoint_best.pt`, 2,302,283 params |
| Prompt with element ("ice dragon with a helmet") | 200, element parsed, `src: procedural_guard` (critic rejected the neural candidate) |
| Prompt with style + attack pose | 200, `src: neural` |
| `POST /api/animation/generate` | 200, 6-frame walk |
| `POST /api/action_set/plan`, `POST /api/action_set` | 200, 8 actions with per-action frame counts |
| `POST /api/export/sheet` | 422 without `actions` (correct validation), 200 with a sheet body |
| `POST /api/assets` → `GET /api/assets/{id}` | saved, then readable by id |
| `POST /api/storage/trash` → `GET /api/storage/trash` → `POST /api/storage/restore` | 200 each; asset leaves the list, appears in trash, comes back |
| `POST /api/storage/permanent_delete` | 400 without `confirm: true` (safety gate), 200 with it; only the smoke asset was deleted |
| Restart the server | health OK, remaining asset (user's `ZOMBIE_4480`) still present |
| `GET /api/metrics`, `/api/storage/dashboard`, `/api/storage/sync`, `/api/history`, `/api/dataset/stats`, `/api/benchmarks/gallery`, `/api/evaluation/results` | all 200 |
| Browser UI | Generate (procedural and trained engines), render canvas, critic panel, frame strip, Asset Library listing the saved asset — all rendered without console-blocking errors |

Error paths behave: unknown asset → 404, invalid script → 400 with the offending line,
missing checkpoint → 404, unsupported palette name → falls back to a valid 16-colour palette and
still renders (see limitation 1).

## 7. Known limitations (honest list)

1. **Unknown palette names are accepted.** `{"palette": "nope"}` returns 200, renders with the
   PICO-8 fallback, and echoes `"palette": "nope"` back in the response. Cosmetically misleading;
   not changed here because it is behaviour, not cleanup.
2. **Neural candidacy is still gated.** With the procedural guard active, some prompts return
   `candidate_source: procedural_guard` even when the engine selector says trained model — this is
   the 90 % critic-clean gate working, and it is the open research gap
   ("neural quality ≥ procedural quality").
3. **`is_clean` is `null`** in `/api/generate` responses for some poses; the critic panel computes
   it separately via `/api/critic/inspect`.
4. **Parameter count discrepancy (unresolved).** `GET /api/models` reports 2,302,283 parameters for
   `svan_checkpoint_best.pt`; the research overview states 2,361,304. One of the two numbers is
   wrong; not investigated in this pass and not "fixed" by editing the doc.
5. **5 stale rows in the storage catalogue** (`/api/storage/orphans` "missing files") reference
   files that no longer exist, including one `.pyc` removed in this cleanup. `/api/storage/sync`
   adds but does not prune. Left untouched because it would mean deleting DB records.
6. **Untested here:** GPU/CUDA paths (no GPU in this container), model training/fine-tuning UI
   flows (explicitly out of scope: no retraining), and the GitHub Actions workflow (no `workflow`
   scope on the CLI token). The workflow file itself is unchanged.

## 8. Launch steps

Backend + built frontend (one process, serves the UI at the same port):

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
cd frontend && npm install && npm run build && cd ..   # only needed after frontend changes
SPRITEMIND_THREADS=2 python backend/run_server.py
# open http://127.0.0.1:7860
```

Frontend hot-reload development (two processes):

```bash
python backend/run_server.py                    # terminal 1, API on :7860
cd frontend && npm run dev                      # terminal 2, Vite on :5173, proxies /api
```

Full verification before a release: `./scripts/check.sh`.

Notes for a first run: the engine selector defaults to **Procedural Baseline**; pick
**Trained SVAN Model** to exercise the neural path (first call loads the 9 MB checkpoint, ~0.45 s
on CPU, later calls reuse it). The database and history files are created on first boot.
