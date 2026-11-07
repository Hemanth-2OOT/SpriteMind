# SpriteMind — Roadmap & TODO

Working backlog for SpriteMind. Everything here is drawn from measured gaps in this repo, not
wishlists. Each item lists **why**, **how**, **done when**, and a **size**:

- `S` — under an hour
- `M` — half a day to two days
- `L` — a week or more / needs a retrain

Tick items as they land and move them to the "Shipped" section at the bottom with the PR number.

Current baseline (as of `386c060`):

| Metric | Value |
| --- | --- |
| SVAN checkpoint | 2,351,639 params, val loss 0.3482, token acc 98.68%, vocab 403 |
| Benchmark (100 held-out prompts) | IoU 0.495, valid 100%, palette 100%, 465 ms |
| Quality gate clean rate | 87.5% (8 prompts), 72.5% (per-prompt seeds) |
| Visual critic | F1 0.777, recall 88.8%, calibrated p=0.72 → 100% precision / 0% FP |
| Tests | 179 passing, CI green on every PR |

---

## P0 — Correctness we cannot ship without

- [x] **Seed determinism test** — `S` ✓ #8
  *Why:* a `seed` field that does not reproduce the same PNG is a bug users will file as "the model is random".
  *How:* assert two calls with the same `(prompt, seed, engine)` are byte-identical after PNG encode; run over 20 seeds in `test_pipeline_determinism.py`.
  *Done when:* the test is in CI and passes on `trained_model` and `procedural`.

- [x] **Cache correctness guard** — `S` ✓ #8
  *Why:* the LRU cache keys on the request; a param added later (e.g. `best_of`) silently returns stale sprites if it is missing from the key.
  *How:* assert every request field that changes output is part of the cache key, with a test that two different `best_of` values do not share an entry.
  *Done when:* key is derived from the validated request model, not a hand-written tuple.

- [x] **Golden-image regression set** — `M` ✓ #10
  *Why:* renderer and grammar changes today are validated only by aggregate metrics; a visual regression can hide inside a stable clean rate.
  *How:* commit 10 reference sprites (one per category) as PNGs with a stored perceptual hash; fail CI when Hamming distance exceeds tolerance. Regenerate deliberately with `--update-goldens`.
  *Done when:* a deliberate `SHADE` bug makes the suite fail.

## P1 — Prompt faithfulness (the real capability gap)

Today the text prompt only picks a category, style and palette; "blue wizard" and "red wizard"
give the same distribution. This is the largest quality jump available.

- [x] **Colour-faithfulness metric** — `S` ✓ #9 (`docs/PROMPT_FAITHFULNESS.md`, baseline 6.1%)
  *Why:* no measurement exists for whether the sprite honours colours in the prompt; without it the fix below cannot be validated.
  *How:* parse colour words from the prompt, map to the nearest palette ramp, measure share of silhouette pixels within that ramp; add to `scripts/quality_gate.py` and the benchmark JSON.
  *Done when:* the number is reported and the current baseline (expected: poor) is recorded.

- [ ] **Prompt text encoder in SVAN** — `L`
  *Why:* gives the model a conditioning signal it currently lacks entirely.
  *How:* character-level embedding over the prompt → cross-attention (or a 4th memory token) into the decoder; keep a null-embedding path so existing checkpoints still load.
  *Done when:* colour-faithfulness rises materially at the same IoU, and the old checkpoint still runs through the fallback path.

- [ ] **Conditioned corpus generator** — `M`
  *Why:* the encoder is useless if the training data ignores the prompt.
  *How:* extend `ml/generator.py` so prompts carry colour, element (fire/ice/poison), weapon and armour tags, and the generated PixelScript actually uses them.
  *Done when:* a held-out probe shows the corpus itself is >90% faithful to its own prompt tags.

- [ ] **Element/weapon vocabulary in the DSL** — `M`
  *Why:* fire vs ice currently differ only by palette, so the model has nothing to learn.
  *How:* new primitives or a parameterised `AURA`/`WEAPON` action in the grammar, plus tokenizer entries (renumber tokens again — bump `vocab` and retrain; see the grammar-v2 note on incompatibility).
  *Done when:* `docs/PIXELSCRIPT_GRAMMAR.md` documents them and the validator accepts them.

## P2 — Speed and control

- [ ] **Batched `best_of` sampling** — `M`
  *Why:* 4 candidates cost 4× serial forward passes; batching should cut the 465 ms mean substantially.
  *How:* route candidates through `SVAN._sample_batch` in one pass, then rerank as today.
  *Done when:* benchmark mean drops with identical seed-for-seed output.

- [ ] **Expose `best_of`, `temperature`, `top_k` in the API and Generate form** — `S`
  *Why:* users cannot trade speed for quality today; the knobs exist but are hard-coded.
  *How:* add to the request schema with bounds, wire a compact slider row in the Generate panel, include them in the cache key.
  *Done when:* the UI changes output and the latency difference is visible.

- [ ] **`GET /api/metrics` (p50/p95 ring buffer)** — `S`
  *Why:* latency is currently a one-off JSON file in `data/`, so regressions are invisible at runtime.
  *How:* in-process deque per endpoint, exposed as JSON; the Model Lab renders it.
  *Done when:* the numbers survive a restart cycle in the UI (not just in a file).

- [ ] **ONNX or quantised export path** — `L`
  *Why:* CPU inference is the ceiling on adoption for anyone without a GPU box.
  *How:* export SVAN to ONNX, verify token-for-token parity against torch on 50 prompts, ship behind `engine="onnx"`.
  *Done when:* parity test passes and latency is measured.

## P3 — Animation and action sets

- [ ] **Retrain TemporalSVAN on grammar-v2 sequences with clean-rate gating** — `M`
  *Why:* best val loss 3.1834 means motion is effectively procedural today; attacks and dashes are the weakest visual output.
  *How:* regenerate the temporal dataset from gv2 scripts, train, and accept a checkpoint only if coherence islands stay at zero and the critic clean rate does not fall.
  *Done when:* a side-by-side action-set render shows improved limb continuity versus the current checkpoint at the same seed.

- [ ] **Sprite-sheet / atlas export in Action Sets** — `M`
  *Why:* game engines want one atlas plus JSON, not four separate downloads.
  *How:* pack frames into a horizontal sheet with uniform padding, emit a JSON frame map (`name`, `x`, `y`, `w`, `h`, `duration`), offer PNG + Aseprite JSON.
  *Done when:* the exported pair loads in Aseprite and in a sample pygame loader script in `docs/`.

- [ ] **Inter-frame continuity metric** — `S`
  *Why:* the user-visible complaint was "missing pixels between transitions"; we measure islands but not per-pixel frame deltas.
  *How:* per-frame-pair delta of opaque masks, report mean/median and worst pair; add to the coherence report.
  *Done when:* a deliberately broken frame ranks worst.

- [ ] **Onion-skin export** — `S`
  *Why:* animators want to check motion without opening the app.
  *How:* composite Ghost PNG from the existing onion-skin view.

## P4 — Evaluation and trust

- [ ] **Human-rated comparison set** — `M`
  *Why:* IoU and clean rate are proxies; nobody has checked they correlate with "looks good".
  *How:* 200 prompts × 3 seeds, rated 1–5 by you (or a helper), stored in `data/eval/human_ratings.json`; compute Spearman against IoU and clean rate.
  *Done when:* the correlation is documented (even if weak — that finding matters).

- [ ] **Per-category report card in Model Lab** — `M`
  *Why:* zombie scored ~52 vs wizard ~92 IoU in our probe; that gap is invisible in the UI.
  *How:* group the benchmark JSON by category, render a sorted table with bars and a "worst category" callout.
  *Done when:* the weakest category is obvious without reading a file.

- [ ] **Expanded benchmark suite (300 prompts)** — `M`
  *Why:* 100 prompts make a 1-point IoU change statistically meaningless.
  *How:* extend `data/benchmarks/generalization_prompts.json`, report mean ± standard error, keep the runtime under 5 minutes.
  *Done when:* confidence intervals are reported in the README metrics block.

## P5 — Product surface

- [ ] **Batch generation / sprite pack export** — `M`
  *Why:* generating a whole cast (8 characters × 4 poses) is the actual game-dev use case.
  *How:* queue prompts, run server-side with a concurrency cap, zip results with a manifest.
  *Done when:* a 32-sprite pack downloads in one click and every file is individually valid.

- [ ] **Prompt history with re-seed** — `S`
  *Why:* finding the sprite you liked last week is currently impossible (cache is in-memory only).
  *How:* persist (prompt, params, seed, thumbnail) rows in SQLite and show them in the sidebar; click restores params and regenerates from the seed.
  *Done when:* a restart preserves history.

- [ ] **Style transfer between generated sprites** — `L`
  *Why:* users want the same character in 8-bit and 16-bit without re-prompting.
  *How:* re-quantise an existing sprite's PixelScript into another palette/style pass rather than a new generation.

- [ ] **Docs site pass** — `S`
  *Why:* 7 docs in `docs/` with no index; new users do not know where to start.
  *How:* `docs/README.md` linking each file with a one-line purpose, plus a quickstart in the root README.

## P6 — Hygiene (opportunistic)

- [ ] CI cache for pip and npm — `S` (backend job re-installs torch every run, ~40 s of the 48 s)
- [ ] Nightly benchmark job with a regression comment on the PR — `M`
- [ ] `SECURITY-REPORT.md` sweep before any public launch — `M`
- [ ] Revoke the development GitHub token once development pauses — `S` *(user-owned action)*

---

## Shipped

| Item | PR |
| --- | --- |
| Grammar v2 (OUTLINE/SHADE), guarded rerank, gv2 retrain, latency report | #4 |
| Visual critic retrained on grammar sprites, calibrated at p=0.72 | #5 |
| Neural critic verdict surfaced in the Inspector | #6 |
| Seed determinism + cache-key coverage (`test_pipeline_determinism.py`) | #8 |
| Colour-faithfulness metric in gate + benchmark (`faithfulness.py`) | #9 |
| Golden-image regression set (10 categories, exact pixel diff) | #10 |

---

## Ordering advice

1. **P0 first** — determinism and cache keys are cheap and prevent embarrassing bugs.
2. **Colour metric (P1) then batched `best_of` (P2)** — visible progress within a day.
3. **Prompt encoder (P1) is the real leap**, but budget a corpus change and a retrain; do it when
   you can leave the machine training for a few hours.
4. Never accept a checkpoint that improves loss while dropping the gate clean rate.
