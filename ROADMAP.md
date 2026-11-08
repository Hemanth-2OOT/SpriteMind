# SpriteMind — Roadmap & TODO

Working backlog for SpriteMind. Everything here is drawn from measured gaps in this repo, not
wishlists. Each item lists **why**, **how**, **done when**, and a **size**:

- `S` — under an hour
- `M` — half a day to two days
- `L` — a week or more / needs a retrain

Tick items as they land and move them to the "Shipped" section at the bottom with the PR number.

Current baseline (as of `6d0d58a`):

| Metric | Value |
| --- | --- |
| SVAN checkpoint | 2,351,639 params, val loss 0.3482, token acc 98.68%, vocab 403 |
| Benchmark (100 held-out prompts) | IoU 0.495, valid 100%, palette 100%, 291 ms mean |
| Per-category report card | mean 65.38; weakest wizard 64.2 (`/api/benchmarks/report_card`) |
| Silhouette coherence | 24 prompts, outline coverage 1.0, 0 islands |
| Inter-frame continuity | mean IoU 0.8206, worst pair 0.7688, 311 px lost over 12 transitions |
| Colour faithfulness | trained 6.1%, procedural 10.1%, untrained 0.9% |
| Latency | `best_of=1` ~160 ms, `best_of=4` 291 ms mean over 100 prompts (was 233 / 351 ms); repeated request < 1 ms |
| Quality gate clean rate | 87.5% (8 prompts), 72.5% (per-prompt seeds) |
| Visual critic | F1 0.7799, recall 88.8%, calibrated p=0.72 → 100% precision / 0% FP |
| Tests | 233 passing, CI green on every PR |
| API | generate (+ sampler controls), refine, action sets, animation, export (PNG / spritesheet / GIF / onion skin / JSON), critic, benchmarks (+ per-category report card), metrics, history |

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

- [x] **Prompt text encoder in SVAN** — `L` ✓ #24 (`docs/PROMPT_FAITHFULNESS.md`)
  *Why:* gives the model a conditioning signal it currently lacks entirely.
  *How:* character-level embedding over the prompt → cross-attention (or a 4th memory token) into the decoder; keep a null-embedding path so existing checkpoints still load.
  *Done:* `ml/prompt_encoder.py`, opt-in via `--prompt-text`, legacy checkpoints unchanged. **No measurable win yet** (clean rate 93.8% both, IoU 0.8206 both): the reconstruction text repeats the categorical labels and the corpus has no colour tags, so the token adds no information. Prerequisite is the colour/element corpus below.

- [x] **Conditioned corpus generator** — `M` ✓ #11 (`docs/PROMPT_CONDITIONING.md`, colour present 100%, element visible 100%)
  *Done when:* a held-out probe shows the corpus itself is >90% faithful to its own prompt tags. **Met.**
  *Why:* the encoder is useless if the training data ignores the prompt.
  *How:* extend `ml/generator.py` so prompts carry colour, element (fire/ice/poison), weapon and armour tags, and the generated PixelScript actually uses them.

- [x] **Element vocabulary in the DSL** — `M` ✓ #25 (`docs/PIXELSCRIPT_GRAMMAR.md`, `AURA`)
  *Done:* the `AURA` primitive (dithered halo, `density` 1–4) is emitted for elemental prompts and
  tinted from `ELEMENT_RGB`; tokenizer vocab 403 → 404; colour/element ids ride along as SVAN
  embeddings. Weapon nouns are parsed (`WEAPON_WORDS`) but not yet rendered — see P5.
- [ ] **Weapon vocabulary in the DSL** — `M`
  *Why:* fire vs ice currently differ only by palette, so the model has nothing to learn.
  *How:* new primitives or a parameterised `AURA`/`WEAPON` action in the grammar, plus tokenizer entries (renumber tokens again — bump `vocab` and retrain; see the grammar-v2 note on incompatibility).
  *Done when:* `docs/PIXELSCRIPT_GRAMMAR.md` documents them and the validator accepts them.

## P2 — Speed and control

- [x] **Batched `best_of` sampling** — `M` ✓ #12 (`_sample_batch`, 4 candidates: 478 ms serial → 182 ms)
  *Why:* 4 candidates cost 4× serial forward passes; batching should cut the 465 ms mean substantially.
  *How:* route candidates through `SVAN._sample_batch` in one pass, then rerank as today.
  *Done when:* benchmark mean drops with identical seed-for-seed output. **Met:** 2.6× faster, candidates still distinct and seed-deterministic.

- [x] **Expose `best_of`, `temperature`, `top_k` in the API and Generate form** — `S` ✓ #13
  *Why:* users cannot trade speed for quality today; the knobs exist but are hard-coded.
  *How:* add to the request schema with bounds, wire a compact slider row in the Generate panel, include them in the cache key.
  *Done when:* the UI changes output and the latency difference is visible. **Met:** all three reach the sampler, cache key covers them, sliders live in the Sampler block.

- [x] **`GET /api/metrics` (p50/p95 ring buffer)** — `S` ✓ #14 (`spritemind/server/metrics.py`, Model Lab card)
  *Done when:* the Model Lab shows percentiles that move as you generate. **Met.**
  *Why:* latency is currently a one-off JSON file in `data/`, so regressions are invisible at runtime.
  *How:* in-process deque per endpoint, exposed as JSON; the Model Lab renders it.

- [ ] **ONNX or quantised export path** — `L`
  *Why:* CPU inference is the ceiling on adoption for anyone without a GPU box.
  *How:* export SVAN to ONNX, verify token-for-token parity against torch on 50 prompts, ship behind `engine="onnx"`.
  *Done when:* parity test passes and latency is measured.

## P3 — Animation and action sets

- [x] **Retrain TemporalSVAN on grammar-v2 sequences with clean-rate gating** — `M` ✓ #25 (`docs/MODEL_TUNING.md`)
  *Done:* `data/temporal_dataset_gv2` (2,400 sequences, 1,931/469) regenerated from the current
  generator, 80 epochs, best val 3.2831; the previous checkpoint could not load at all (18 vs 20
  animation classes), so this restores animation quality rather than only improving it.
  *Why:* best val loss 3.1834 means motion is effectively procedural today; attacks and dashes are the weakest visual output.
  *How:* regenerate the temporal dataset from gv2 scripts, train, and accept a checkpoint only if coherence islands stay at zero and the critic clean rate does not fall.
  *Done when:* a side-by-side action-set render shows improved limb continuity versus the current checkpoint at the same seed.

- [x] **Sprite-sheet / atlas export in Action Sets** — `M` ✓ #25 (`docs/ACTION_SETS.md`, `POST /api/export/sheet`)
  *Done:* one animation per row with per-action `row`/`frames` metadata, uniform frame size enforced
  (HTTP 400 otherwise), PNG + atlas JSON downloaded from the Action Sets page.
  *Why:* game engines want one atlas plus JSON, not four separate downloads.
  *How:* pack frames into a horizontal sheet with uniform padding, emit a JSON frame map (`name`, `x`, `y`, `w`, `h`, `duration`), offer PNG + Aseprite JSON.
  *Done when:* the exported pair loads in Aseprite and in a sample pygame loader script in `docs/`.

- [x] **Inter-frame continuity metric** — `S` ✓ #15 (`renderer/continuity.py`, 4 walk cycles in the coherence report)
  *Why:* the user-visible complaint was "missing pixels between transitions"; we measure islands but not per-pixel frame deltas.
  *How:* per-frame-pair delta of opaque masks, report mean/median and worst pair; add to the coherence report.
  **Met:** mean IoU 0.8206, worst pair 0.7688, 311 pixels lost over 12 transitions; the broken-frame test ranks worst.

- [x] **Onion-skin export** — `S` ✓ #16 (`compose_onion_skin`, `format: "onion"` in `/api/export`)
  *Why:* animators want to check motion without opening the app.
  *How:* composite Ghost PNG from the existing onion-skin view.
  **Met:** Export ▸ *Onion Skin PNG* writes the current frame over the previous (30%) and next (18%) ghosts.

## P4 — Evaluation and trust

- [ ] **Human-rated comparison set** — `M`
  *Why:* IoU and clean rate are proxies; nobody has checked they correlate with "looks good".
  *How:* 200 prompts × 3 seeds, rated 1–5 by you (or a helper), stored in `data/eval/human_ratings.json`; compute Spearman against IoU and clean rate.
  *Done when:* the correlation is documented (even if weak — that finding matters).

- [x] **Per-category report card in Model Lab** — `M` ✓ #17 (`GET /api/benchmarks/report_card`, `ml/report_card.py`)
  *Why:* zombie scored ~52 vs wizard ~92 IoU in our probe; that gap is invisible in the UI.
  *How:* group the benchmark JSON by category, render a sorted table with bars and a "worst category" callout.
  *Done when:* the weakest category is obvious without reading a file.
  **Met:** the Model Lab shows every category weakest first with score bars; on the current 100-prompt run wizard is
  weakest at 64.2 and the mean is 65.38.

- [x] **Expanded benchmark suite (300 prompts)** — `M` ✓ #25 (`data/benchmarks/benchmark_results.json`)
  *Done:* 100 curated + 200 generated prompts at a fixed seed, mean ± standard error for every
  metric; the README block reports 0.503 ± 0.002 IoU and colour faithfulness 15.0%.
  *Why:* 100 prompts make a 1-point IoU change statistically meaningless.
  *How:* extend `data/benchmarks/generalization_prompts.json`, report mean ± standard error, keep the runtime under 5 minutes.
  *Done when:* confidence intervals are reported in the README metrics block.

## P4b — Audit findings (2026-11-08)

Full evidence in `docs/IMPLEMENTATION_AUDIT.md`. Ordered by cost; R1–R3 landed with the audit.

- [x] **R1 manifest drift** — stale entries pruned; `checkpoint_manifest.prune_missing()` covers it from now on.
- [x] **R2 parameter counts** — every published count is now `sum(p.numel())`; the state-dict tensor count is labelled as such.
- [x] **R3 publishing** — README reports the legacy metrics as proxies and adds a pixel-grounded benchmark line.
- [ ] **R4 critic-driven rerank** — the critic verdict now participates in `_rerank_score` and is tested, but switching it off changes colour faithfulness by −0.06 points (29.60 vs 29.66) and guard wins by 7/300, inside one standard error (`M`)
  *Done when:* `trained_best4` beats `trained_best1` on IoU and colour faithfulness in `scripts/benchmark_neural_vs_procedural.py`. Currently IoU 0.745 vs 0.751 and colour 29.60 vs 29.31 — not met. The critic's `quality_score` does not yet separate sprites well enough to steer selection; see `docs/RERANK_EVALUATION.md`.
- [x] **R5 colour-aware rerank** — production colour faithfulness 13.9% → 29.6% (2.1×) at n=300 with the same prompts, seeds and checkpoint. Caveat: neural-only colour moved 7.0% → 9.9%, so the gain comes from selection, not generation.
  *Done when:* neural colour faithfulness ≥ procedural at n=300 — 29.60 vs 29.18, met. Neural-only remains below procedural (9.9% vs 29.2%), so "neural ≥ procedural" stays the primary open objective.
- [x] **R6 palette embedding actually used** — `_palette_index()` feeds the real row, gated on `palette_conditioning_trained`; `PixelScriptDataset` now reads each record's `palette`, and `finetune_svan.py --palette-conditioning` trains the embedding. At a fixed seed the shipped checkpoint produces 1.00 distinct programs across 5 palettes (the plumbing was inert, as predicted), the retrained one 2.50.
  *Evidence:* `docs/RETRAIN_V4.md`, `data/benchmarks/palette_conditioning.json`.
- [x] **R7 socket labels** — `spritemind.dataset.anatomical_sockets` labels sockets from the drawn parts (head centre, torso centre, distal arm pixels); `scripts/train_socket_anatomical.py` trains `socket_predictor_anatomical_v2.pt` to **0.349 px** MAE against anatomical targets, versus 2.469 px for the shipped weights and 1.803 px for the proportional prior.
  *Not wired into production:* promotion changes accessory placement and needs its own review.
- [ ] **R8 temporal attribution** — 70/30 neural/prior blend; neural magnitude 1.284 px vs prior 1.281 px (`M`)
  *Done when:* the API reports both magnitudes and trained motion differs from the prior by > 2 SE.
- [ ] **R9 element corpus** — `data/datasets/cond_v1_15k` labels 7,533/15,000 records (50%, up from 25%) with per-element counts in its metadata; element *faithfulness* has not been re-measured yet, so the 33.4% figure still stands as the last measurement.
  *Done when:* element faithfulness ≥ 60%.
- [x] **R10 compositional split** — `data/splits/heldout_v1.json` freezes 30 (category, style, pose) triples and 10 (category, element) pairs behind a SHA-256 digest that `load_spec()` verifies; conditioned corpora keep those combinations out of train/val and the digest is recorded in their metadata.
  *Evidence:* `data/benchmarks/heldout_v1_baseline.json` (v3) and `heldout_v1_v4_ep2.json` (retrained), both on 80 held-out + 80 in-distribution prompts at identical seeds. Caveat recorded in the doc: the IoU reference is the canonical idle template, so cross-regime deltas compare prompt mixes, and only same-regime old-vs-new rows are checkpoint improvements.

### R6/R7/R9/R10 results (2026-11-05)

Full evidence in `docs/RETRAIN_V4.md`; production checkpoint and corpus untouched.

- Neural-only IoU on held-out prompts: 0.7682 (v3) → **0.7904** (v4 ep2); in-distribution 0.7599 → **0.7767**. Both now exceed the procedural baseline's 0.754 without the guard.
- Neural-only colour faithfulness **regressed**: 19.81 → 17.43 (held-out), 21.00 → 15.36 (in-distribution). Guarded output still improved slightly (48.63 → 49.08 held-out), i.e. selection masked it. This is the open item.
- The finetune adoption gate (critic clean rate ≥ 90%) rejected both epochs at 87.5%, so no checkpoint was promoted automatically.

## P5 — Product surface

- [ ] **Batch generation / sprite pack export** — `M`
  *Why:* generating a whole cast (8 characters × 4 poses) is the actual game-dev use case.
  *How:* queue prompts, run server-side with a concurrency cap, zip results with a manifest.
  *Done when:* a 32-sprite pack downloads in one click and every file is individually valid.

- [x] **Prompt history with re-seed** — `S` ✓ #18 (`server/history.py`, `GET/DELETE /api/history`, sidebar panel)
  *Why:* finding the sprite you liked last week is currently impossible (cache is in-memory only).
  *How:* persist (prompt, params, seed, thumbnail) rows in SQLite and show them in the sidebar; click restores params and regenerates from the seed.
  *Done when:* a restart preserves history.
  **Met:** rows survive restarts in `data/history.sqlite` (500-row cap, git-ignored). A click restores the prompt, category,
  style, pose, palette, seed and sampler settings; the dice button re-runs the same prompt with a fresh seed. Thumbnails
  are deliberately not stored — the file stays small and the seed reproduces the sprite instead.

- [ ] **Style transfer between generated sprites** — `L`
  *Why:* users want the same character in 8-bit and 16-bit without re-prompting.
  *How:* re-quantise an existing sprite's PixelScript into another palette/style pass rather than a new generation.

- [x] **Docs site pass** — `S` ✓ #19 (`docs/README.md`)
  *Why:* 7 docs in `docs/` with no index; new users do not know where to start.
  *How:* `docs/README.md` linking each file with a one-line purpose, plus a quickstart in the root README.
  **Met:** `docs/README.md` groups the docs by generating, training and operating, and points at the scripts that produce
  every published number.

## P6 — Hygiene (opportunistic)

- [x] CI cache for pip and npm — `S` ✓ (`.github/workflows/ci.yml`: `cache: pip` and `cache: npm` with `frontend/package-lock.json`)
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
| Prompt-conditioned corpus (colour/element tags, palette-aware mapping) | #11 |
| Batched `best_of` sampling (`test_best_of.py`, 2.6× faster) | #12 |
| `best_of` / `temperature` / `top_k` exposed in API, cache key and Generate panel | #13 |
| `GET /api/metrics` p50/p95 ring buffer + Model Lab card | #14 |
| Inter-frame continuity metric in the coherence report | #15 |
| Onion-skin PNG export from the animation timeline | #16 |
| Per-category report card in the Model Lab | #17 |
| Prompt history with re-seed (SQLite) | #18 |
| Docs index (`docs/README.md`) | #19 |

---

## Where this stands

Everything cheap and self-contained is done: determinism and cache guards, a golden-image set,
colour faithfulness, prompt-conditioned corpus, batched `best_of`, sampler controls in the API
and UI, latency percentiles, inter-frame continuity, onion-skin export, the per-category report
card, prompt history and the docs index. That work is all `S`/`M` items with objective pass
criteria, and each one shipped with tests.

What is left is deliberately not in that category. Ranked by what would change the product most.

*Status after #25:* items 1, 2 and the temporal half of item 4 are shipped. What remains is the
prompt-text-encoder retrain (now unblocked — the corpus finally carries colour and element ids),
the human-rated set, ONNX export and the batch/pack product surface.

1. **Prompt text encoder in SVAN** (`L`, encoder shipped #24, retrain blocked) — the only remaining
   real capability gap. The encoder exists and is measurable now, but a one-epoch retrain changed
   nothing (93.8% clean rate and 0.8206 IoU on both checkpoints) because the corpus carries no
   colour or element tags to learn from. Order of work: colour/element vocabulary in the DSL, a
   corpus that uses it, then the retrain and the benchmark comparison.
2. **Expanded 300-prompt benchmark** (`M`) — 100 prompts make a 1-point IoU change statistically
   meaningless, and the report card explicitly shows category gaps. Bigger suite first means every
   later model change can be judged. Cheap in compute, but it needs the prompts written.
3. **Human-rated comparison set** (`M`) — IoU and clean rate are proxies nobody has validated.
   Needs a person to rate 200 prompts × 3 seeds; the code around it is small.
4. **TemporalSVAN gv2 retrain + sprite-sheet/atlas export** (`M` + `M`) — temporal quality is
   measured (continuity 0.82) but the temporal model is still trained on grammar v1. Export is
   mechanical once the frames are right.
5. **ONNX / quantised export** (`L`) — a pure latency and portability win; no quality change.
   Only worth it if the browser needs to run the model.
6. **Batch generation, style transfer, element vocabulary in the DSL** (`M`/`L`) — product surface.
   Useful, but each adds rendering paths that must be kept coherent with the rest.
7. **Nightly benchmark + security sweep** (`M`) — hygiene before a public launch; the sweep is
   most valuable once the product stops changing daily.

Two things are not code: **revoke the development GitHub token** when development pauses (yours to
do), and only run the `SECURITY-REPORT.md` sweep once the surface stops moving.

## Ordering advice

1. **P0 first** — determinism and cache keys are cheap and prevent embarrassing bugs.
2. **Colour metric (P1) then batched `best_of` (P2)** — visible progress within a day.
3. **Prompt encoder (P1) is the real leap**, but budget a corpus change and a retrain; do it when
   you can leave the machine training for a few hours.
4. Never accept a checkpoint that improves loss while dropping the gate clean rate.
