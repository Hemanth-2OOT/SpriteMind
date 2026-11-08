# SpriteMind release checklist

What to verify before tagging a release or opening the repository to outside users.
Repository: `github.com/Hemanth-2OOT/SpriteMind`. The current verified state is recorded in
[`FINAL_RELEASE_READINESS.md`](FINAL_RELEASE_READINESS.md).

## 1. Local checks (one command)

```bash
./scripts/check.sh        # ruff, pytest + coverage floor, README metrics, frontend build, quality gate
```

It fails on a lint error, a failing test, a coverage drop below the 71 % floor, stale README
metrics, a broken frontend build, or a quality-gate regression. `SKIP_QUALITY_GATE=1` skips the
last step (it samples held-out prompts and takes a few seconds per sample).

## 2. What must not be committed

`.gitignore` already excludes these; check `git status` before staging anyway.

- Python caches: `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ruff_cache/`, `.coverage*`
- Frontend build and dependencies: `frontend/dist/`, `frontend/node_modules/`
- Runtime state, regenerated on first boot: `data/spritemind.db`, `data/history.sqlite`,
  `data/cache/`, `data/trash/`, `data/temp/`, `data/backups/`
- Regenerable training corpora (`data/datasets/*/records.jsonl` and the `*_synthetic_dataset.json`
  files); the `metadata.json` provenance files are tracked on purpose
- Local-only checkpoints (`models/checkpoints/svan_v4_palette.pt`)
- Secrets: `.env`, `.env.local`, OS/editor noise, `*.log`

Protected and intentionally tracked: `models/checkpoints/*.pt` (the shipped five plus the recorded
experimental ones), `data/benchmarks/`, `data/reports/`, `data/splits/`, `docs/research/`.

## 3. Repository size

No single file is near GitHub's 100 MB limit. Largest tracked blobs: three SVAN checkpoints at
9.1 MB, one at 6.7 MB, `data/temporal_dataset_1k/train_sequences.jsonl` at 4.2 MB. Git LFS is not
required; if a future checkpoint or corpus crosses ~50 MB, decide between LFS and a generated-on-demand
corpus before committing it.

## 4. Security and cleanliness

- No credentials or tokens in the tree; `git grep -nE "(ghp_|sk-|AKIA|-----BEGIN)"` returns nothing.
- `data/spritemind.db` and `data/history.sqlite` stay untracked (runtime state, possibly personal).
- Checkpoints load from `models/checkpoints/` and are verified against `checkpoint_manifest.json`
  (`integrity: verified` in `GET /api/models`).

## 5. Branch and PR state

Work lands through one review PR per milestone, stacked on the previous branch and never merged by
the agent. Before publishing, confirm with `gh pr list` that every open PR is intentional, that each
base branch is the stack's parent, and that releases are tagged only from `main`.

## 6. Publish

Only after the owner explicitly asks for it:

```bash
git status                      # expect a clean tree
./scripts/check.sh              # expect ALL CHECKS PASSED
git switch main && git pull     # merge the approved PRs through review, not locally
git tag -a v0.1.0 -m "SpriteMind v0.1.0"
git push origin main --follow-tags
```

No force-push, no history rewrite, no tag delete.
