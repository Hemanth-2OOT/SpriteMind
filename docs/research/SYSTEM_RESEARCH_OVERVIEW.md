# SpriteMind — System Research Overview

**Document type:** consolidated technical reference (architecture, stack, methods, results)
**Version:** post-`#25` (`main` @ `f61463e`), grammar v2.1, SVAN v3 checkpoint
**Scope:** everything that exists in the repository today — what it is, how it works, how it
is measured, and what it cannot yet do.

This document is the single long-form description of SpriteMind. The stage reports in this
directory (`STAGE_2` … `STAGE_5_5`) are the historical record of how each subsystem was
arrived at; where they disagree with the numbers here, this document and the generated
`README.md` metrics block are authoritative, because both are regenerated from the shipped
checkpoints and the recorded benchmark JSON.

---

## 1. Abstract

SpriteMind is a **local, from-scratch neural pixel-art generator**. It does not sample a
diffusion model and it does not call a hosted image API. A 2.36M-parameter causal transformer
(**SVAN**, SpriteMind Visual-Action Network) writes a short program in a domain-specific
drawing language (**PixelScript**), a deterministic CPU renderer rasterises that program to a
32×32 sprite, a 369K-parameter visual critic inspects the raster for anatomical and palette
defects, a 427K-parameter recurrent network (**TemporalSVAN**) articulates the character across
20 animation classes, and a 619K-parameter regressor (**SocketPredictor**) places equipment on
sub-pixel sockets. Everything runs CPU-first, deterministically, and offline.

The contribution is not "another pixel generator". It is that the artefact is **inspectable
and auditable at every stage**: the model's output is a readable program, the render is
reproducible from a seed, the defect report names pixels, the animation is a trajectory over
named parts, and the whole pipeline is covered by a 257-test suite, a perceptual-hash golden
set, and a generation quality gate that fails CI on regression.

---

## 2. Problem statement and design goals

**The gap.** Pixel art for games is expensive, and generative image models are a poor fit for
it: they are trained on photographs, they cannot guarantee a bounded palette, they produce
non-integer pixel grids, and their output cannot be edited except by inpainting. Sprite sheets
compound the problem — an animation must keep the same character consistent across frames and
across actions.

**The bet.** If a sprite is a *program* rather than an *image*, then every property that
matters for games becomes enforceable at generation time: palette membership, part
connectivity, frame-to-frame continuity, direction of facing, per-part equipment anchors.

**Design goals, in priority order.**

| # | Goal | How it is enforced |
| --- | --- | --- |
| 1 | Output is always valid | Grammar-constrained decoding; 100% of emitted programs parse and validate |
| 2 | Output is always on-palette | `COL_*` tokens are palette indices, not RGB; palette adherence measured at 100% |
| 3 | The same seed gives the same sprite | Seeded sampler, deterministic renderer, hashed goldens with 0-bit tolerance |
| 4 | Frames stay connected | Silhouette-island check per frame; offsets that tear a limb are pulled back |
| 5 | Prompt words change the output | Category / style / pose tokens, plus colour and element embeddings (v3) |
| 6 | The neural model is never trusted blindly | Best-of-N rerank against a procedural candidate; `candidate_source` reports who won |
| 7 | Nothing user-facing can destroy training data | Five-tier data classification with hard deletion gates |
| 8 | Everything runs on a laptop CPU | Two attention threads by default; ~0.57 s per 4-frame request |

---

## 3. System architecture

### 3.1 Layer map

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ L5  Clients        React + Vite studio (7 tabs) · REST/JSON · OpenAPI at /docs │
├──────────────────────────────────────────────────────────────────────────────┤
│ L4  API            FastAPI · 6 routers · ~50 endpoints · Pydantic schemas      │
│                    LRU result cache · request history · metrics ring buffer    │
├──────────────────────────────────────────────────────────────────────────────┤
│ L3  Orchestration  SpriteMindPipeline   ActionPlanner   AssetComposer          │
│                    CriticRepairEngine   TemporalAnimator  AtlasBuilder         │
├──────────────────────────────────────────────────────────────────────────────┤
│ L2  Models         SVAN (2.36M)   VisualCritic (369K)   TemporalSVAN (427K)   │
│                    SocketPredictor (619K)   procedural baseline (0 params)     │
├──────────────────────────────────────────────────────────────────────────────┤
│ L1  Language       PixelScript AST · parser · validator · serializer           │
│                    tokenizer (404 tokens) · grammar-constrained decoder        │
├──────────────────────────────────────────────────────────────────────────────┤
│ L0  Raster         Deterministic canvas (32×32 RGBA) · layer engine · passes   │
│                    SQRT palette tables · PNG / sheet / atlas / ZIP export      │
├──────────────────────────────────────────────────────────────────────────────┤
│     Persistence    data/spritemind.db (SQLite) · data/assets · models/*.pt     │
└──────────────────────────────────────────────────────────────────────────────┘
```

The dependency direction is strictly downward: the renderer knows nothing about models, the
models know nothing about HTTP, and the API knows nothing about canvas internals. That is what
lets `scripts/*.py` drive the exact code path the server drives — every headline number in this
document came from a script hitting the same pipeline object the API uses.

### 3.2 Request lifecycle (single sprite)

```
prompt ─► conditioning ─► sampler ─► validation ─► render ─► critic ─► rerank ─► PNG
              │             │            │           │         │        │
              │             │            │           │         │        └─ best of N neural
              │             │            │           │         │            candidates + 1
              │             │            │           │         │            procedural ref
              │             │            │           │         └─ defects: stray pixel,
              │             │            │           │            hole, disconnected part,
              │             │            │           │            palette leak, missing limb
              │             │            │           └─ palette indices → RGBA, SHADE,
              │             │            │               OUTLINE, AURA finishing pass
              │             │            └─ grammar state: a rejected token cannot be
              │             │                emitted, so parse errors are structurally 0
              │             └─ autoregressive decode, temperature/top-k, seeded
              └─ category, style, pose (+ colour id, element id in v3)
```

### 3.3 Request lifecycle (action set)

One prompt becomes a coherent animation package:

```
prompt ─► ActionPlanner
            ├─ archetype   (warrior, ranger, arcane, rogue, brute, critter, …)
            ├─ element     (fire, ice, lightning, poison, arcane, holy, shadow, …)
            ├─ weapon      (sword, axe, staff, bow, … — parsed, not yet rendered)
            ├─ proportion  (standard, stocky, slender, chibi)
            └─ accessory   (helmet, cape, horns, wings, backpack, pauldrons)
                 │
                 ├─► one character program, reused by every action
                 ├─► per action: part transforms → SHADE/OUTLINE → element FX
                 │     idle · walk · run · dash · attack · attack_heavy
                 │     cast · cast_burst · shoot · block · hurt · jump · fall · death
                 └─► outputs: per-frame PNGs, GIF/APNG preview, sprite sheet, atlas
```

Because every action derives from the *same* base program, the character cannot change
identity between idle and death. Because the facing flip mirrors the *program* (x → w−1−x) and
not the raster, a left-facing set is a pixel-exact mirror of the right-facing one.

---

## 4. Stack and environments

### 4.1 Runtime

| Layer | Choice | Version | Why |
| --- | --- | --- | --- |
| API | FastAPI + Uvicorn | `>=0.109,<0.200` / `>=0.27,<0.60` | Async, OpenAPI for free, no ORM needed |
| Schema | Pydantic v2 | `2.5 – 3.0` | Request validation at the edge |
| Numerics | NumPy | `1.24 – 3.0` | Canvas ops, HSV classification, atlas packing |
| Learning | PyTorch (CPU) | `2.5 – 3.0` | Small models, no CUDA required at any point |
| Imaging | Pillow | `10 – 13` | PNG, GIF/APNG, nearest-neighbour scaling |
| HTTP client | requests | `2.31 – 3.0` | Integration examples (`examples/deadzone_integration`) |
| DB | SQLite (stdlib) | — | Zero-install auditable catalogue |
| UI | React + Vite + TypeScript | 18.2 / 5.4 / 5.6 | Static bundle served by the same process |
| Styling | Tailwind CSS | 3.4 | Utility classes only, no component library |
| Icons | lucide-react | 0.344 | — |

Frontend build output (`frontend/dist`) is generated, not committed; the API boots without it
and `/` prints the build command instead of a blank page.

### 4.2 Repository scale (tracked files)

| Area | Files | Lines |
| --- | --- | --- |
| `backend/spritemind` | 74 modules | 14,798 |
| `backend/tests` | 33 test files | 2,808 |
| `frontend/src` | 20 | 6,974 |
| `scripts` | 18 | 1,695 |
| **All tracked** | **368** | — |

### 4.3 Configuration surface

Environment-driven, no config files at runtime: `HOST`, `PORT`, `SPRITEMIND_THREADS` (default
2), `SPRITEMIND_CORS_ORIGINS`, `SPRITEMIND_RESULT_CACHE` (default 64; `0` disables). The model
thread count is deliberately small — measured 1–8 threads perform identically, so extra
threads only add scheduling jitter.

---

## 5. PixelScript: the language layer

A sprite is 250–400 tokens of intent instead of 1,024 pixel tokens, which is what makes a
2.3M-parameter model sufficient.

### 5.1 Grammar (v2.1, 14 primitives)

```
LAYER <i> <name> <opacity>          RECT  <x> <y> <w> <h> <col> <filled>
PART  <name> <socket> <x> <y>       LINE  <x0> <y0> <x1> <y1> <col>
PIXEL <x> <y> <col>                 FILL  <x> <y> <col>
ERASE <x> <y> <w> <h>               REUSE <part> <x> <y>
TRANSFORM <part> <dx> <dy> <rot>    SHADE <x> <y> <w> <h> <col> <dir>    (v2)
OUTLINE <x> <y> <w> <h> <col>       AURA  <x> <y> <w> <h> <col> <dens>   (v2.1)
END PART | END PROGRAM
```

* **`SHADE`** darkens (or lightens) one edge of *existing opaque* pixels only — shading can
  never add silhouette. Direction `0..3` = top/bottom/left/right, other integers folded back.
* **`OUTLINE`** draws a 1 px border in the transparent ring touching opaque pixels, masked
  from the **composite** canvas so a multi-layer character gets one connected rim rather than a
  rim per part.
* **`AURA`** is a dithered halo outside the silhouette, tinted from `ELEMENT_RGB` (`fire` →
  ember, `ice` → glacier, `lightning` → violet, `poison` → acid). `density` 1–4: solid ring,
  checkerboard, then progressively sparser sparks. Emitted only when the prompt names an
  element. Costs 7 tokens.
* **`# META: colour=… element=…`** carries conditioning into the program and round-trips
  through parser, validator, serializer and tokenizer.

### 5.2 Finishing pass and animation ordering

Every generated program closes with `LAYER 0 base` → `SHADE` → `OUTLINE` → `END PROGRAM`.
`pixelscript/coherence.py::build_frame_script` injects part `TRANSFORM`s **before** that pair,
so shading and outline are recomputed against the moved parts. Getting this order wrong was a
real bug: idle frames lost outline consistency whenever a limb moved.

### 5.3 Token budget

| Grammar | p50 tokens | p95 tokens |
| --- | --- | --- |
| v1 (11 primitives) | 292 | 364 |
| v2 (13, finishing pass) | 308 | 380 |
| v2.1 (14, AURA) | 315 | 387 |

Decoder budget is `max_seq_len = 448`; observed maximum is 407.

### 5.4 Vocabulary compatibility (a hard invariant)

Primitives change the vocabulary (401 → 403 → 404). A checkpoint cannot emit primitives it was
not trained on, and the pipeline **refuses** a `model_config.vocab_size` mismatch with a
`checkpoint_error` rather than indexing out of bounds. This is not theoretical: when `AURA`
landed, the shipped 403-vocabulary checkpoint stopped loading and every "trained" request had
silently been served by the procedural guard until v3 was trained and promoted. Guarding this
is now automated (see §15.4).

---

## 6. SVAN: the action transformer

### 6.1 Sequence formulation

```
[txt or zeros] [CAT_k] [STYLE_k] [POSE_k] [COLOUR_k] [ELEMENT_k]  →  token stream
```

* `d_model = 128`, 4 heads, 4 layers, FFN 512, dropout 0.05 (shipped), `max_seq_len = 448`
* Prompt conditioning is a semantic vector: one-hot category/style/pose concatenated with
  learned colour/element embeddings (**29 colours**, **9 elements**, id 0 = none) projected to
  `d_model`. Optional prompt-text encoder (`ml/prompt_encoder.py`) exists but is disabled in
  the shipped checkpoint — `PROMPT_MAX_LEN = 64`.
* `vocab_size = 404`; output head over command token + operand tokens.
* **2,361,304** parameters (state-dict sum; the metadata field `parameters` reports 2,302,283
  because it excludes embeddings — prefer the state-dict sum in reporting).

### 6.2 Constrained decoding

Each token slot has an admissible set derived from the grammar state (command arity, coordinate
ranges, palette index range, part names already defined). Illegal tokens are masked, so the
model is structurally incapable of emitting an unparseable program — measured `valid_rate =
100%` on the held-out benchmark. Decoding keeps incremental state per token instead of
re-walking the prefix, and cross-attention key/value projections are reused across steps.

### 6.3 Guarded reranking

Every request samples `best_of` (default 4) neural candidates **plus one procedural
reference**, scores all of them on validity, silhouette fill, part count, centring and
symmetry, and returns the winner with `candidate_source ∈ {neural, procedural_guard}`. This
makes the neural model safe to ship at whatever quality it currently reaches: it can lose to a
0-parameter rule-based renderer, and the API says so honestly rather than pretending.

### 6.4 Caching and latency

An LRU keyed on the full request tuple (`SPRITEMIND_RESULT_CACHE`, default 64) makes a repeated
request sub-millisecond; `/api/health` exports hit/miss counters. On the reference CPU
(i7-12650HX, 2 threads): **~568 ms** for 4-way best-of with a v2.1-length program, **~160 ms**
at `best_of = 1`.

---

## 7. The visual critic

### 7.1 Two halves

* **Topological (deterministic, 0 params):** connected-component analysis over the alpha mask
  finds stray pixels, interior voids, islands and per-part dislocations; RGB classification
  finds palette leaks. This half is exact and always runs.
* **Neural (369,077 params):** a small CNN over the 32×32 RGBA stack plus an 8×8 spatial
  heatmap head for defect localisation, and a severity head (`minor`, `moderate`, `critical`).

Defect taxonomy (7 classes): `clean`, `stray_pixel`, `disconnected_part`, `palette_leak`,
`missing_limb`, `asymmetry`, `unexpected_hole`.

### 7.2 Calibration and repair

The head is calibrated to threshold `p = 0.72`, which on held-out sprites gives **100%
precision with 0 false positives** at **90.09% recall** (val F1 0.7799). The closed-loop
`CriticRepairEngine` applies targeted AST edits (delete stray pixels, fill voids, re-attach
dislocated parts, remap off-palette colours) and re-scores; repairs are reported per defect with
the edit that fixed it.

### 7.3 Historical numbers vs current

The Stage 3 report records precision/recall 1.000 on a 40-sample early probe under a different
label definition. The current numbers (above) are from the retrained critic on the grammar-v2
corpus and are the ones the README reports. Both are true; they are not comparable.

---

## 8. Temporal intelligence

**TemporalSVAN** (426,918 parameters) predicts continuous kinematics `(Δx, Δy, Δrot)` per body
part per frame, conditioned on an animation class embedding.

* 12 parts: `torso, head, arm_left, arm_right, leg_left, leg_right, …`
* **20 animation classes** in four groups — Basic (`idle, walk, run, jump, fall, death`),
  Combat (`melee_attack, ranged_attack, hit_reaction, taking_damage, blocking, special_ability,
  cast, dash`), Creature (`slime_bouncing, zombie_movement, skeleton_walking,
  robot_movement, animal_running, flying_creatures`), and the shared locomotion set.
* Cyclic animations (idle/walk/run) are trained with phase encoding so the loop closes.
* Output is applied as `TRANSFORM` lines on the *program*, then re-shaded and re-outlined per
  frame (§5.2), then passed through a continuity filter:

```
for each frame:  compute silhouette islands
                 if an offset would disconnect a limb  → pull it back
                 among admissible offsets              → prefer the one keeping more pixels
```

That filter is why no body part vanishes between frames, and why thin limbs swing rather than
rotate (rotation of a 2 px limb loses pixels).

**Training:** built on grammar v2 rather than v1, because animation embeds 20 classes and the v1
checkpoint has 18 — the older weights cannot even load into the current model
(`animation_embed.weight` shape mismatch 18×32 vs 20×32), and a test now asserts the shipped
checkpoint loads. Corpus: 2,400 sequences → **1,931 train / 469 val**; 80 epochs, AdamW,
lr 0.002, cosine schedule; **best val MSE 3.2831 at epoch 34**.

Loss on this split is not comparable with the older 487-sequence runs (which score ~37.5 on
this split) — the sequences are longer and more detailed.

---

## 9. Data: corpora, generation, and leakage control

All training data is **synthetic and generated by the repo itself** — no scraped artwork, no
third-party sprite packs. Every corpus is produced by `dataset/generator.py` from a template
library with a fixed seed, which makes the entire training set reproducible from source.

| Corpus | Samples | Split (train/val/test) | Purpose |
| --- | --- | --- | --- |
| `default_synth` | 20 | 17 / 1 / 2 | Smoke tests, golden images |
| `generalization_10k` | 10,000 | 8,000 / 1,000 / 1,000 | First broad-coverage SVAN run |
| `generalization_detail_12k` | 12,000 | 9,600 / 1,200 / 1,200 | Higher-detail variant |
| `gv2_synthetic_25k` | 25,000 | 20,000 / 2,500 / 2,500 | Grammar-v2 corpus |
| `gv2_synthetic_25k_mixed` | 24,660 | 21,925 / 2,735 | Mixed-difficulty fine-tune |
| `expanded_synthetic_25k` | 25,000 | 20,000 / 2,500 / 2,500 | Detail + defect curriculum |
| **`element_synthetic_20k`** | **20,000** | **16,000 / 2,000 / 2,000** | **Shipped v3 corpus: colour + element conditioning** |
| `gv2_critic` | — | — | Critic calibration/eval sprites |
| `temporal_dataset_gv2` | 2,400 sequences | 1,931 / 469 | Shipped temporal corpus (gitignored, regenerable) |

**Generator anatomy.** Each sample is a `character_id` (deterministic template hash), a body
plan (archetype × proportion × accessory), a palette drawn from the fixed palette tables, and a
scripted assembly of parts with injected defect classes at controlled rates. The corpus records
how many samples carry a colour tag (**12,062 / 20,000**) and an element tag (**4,918 / 20,000**).

**Leakage prevention.** Splits are partitioned by a hash of the character template, not by row.
Two samples built from the same template can never land on opposite sides of the split, so
validation measures generalisation to unseen character *shapes*, not memorised rows.

**Provenance.** `dataset/storage.py` plus the SQLite catalogue register each corpus (name, sample
count, split counts, seed, path) so a checkpoint can be traced back to the exact data that
produced it; `models/checkpoints/manifest.json` records the size and SHA-256 of every shipped
weight file.

---

## 10. Training procedures

| Model | Command | Schedule | Selected by |
| --- | --- | --- | --- |
| SVAN | `scripts/finetune_svan.py --dataset data/datasets/element_synthetic_20k` | AdamW, lr 6e-4, cosine, 2 epochs | best val loss |
| Critic | `scripts/build_critic_dataset.py` → `ml/critic/trainer.py` | 6 epochs | best val F1, then threshold calibration |
| TemporalSVAN | `scripts/build_temporal_dataset.py --name temporal_dataset_gv2 --sequences 2400` → `scripts/train_temporal.py --epochs 80 --lr 0.002` | AdamW + CosineAnnealingLR (T_max 8) | best val MSE |
| SocketPredictor | `scripts/train_detail.py` | 8 epochs | final pixel error |

**CPU-first budget.** The pipeline is sized for a 4 GiB container or a laptop CPU. SVAN v3 took
about 1.6 h for 2 epochs on 16,000 samples with 2 threads; the temporal run took roughly 40 min
for 80 epochs on 1,931 short sequences. Longer-context runs were killed by the memory limit
rather than by time — a known constraint, documented in `docs/MODEL_TUNING.md`.

**Promotion ritual.** A checkpoint is not "shipped" because it trained. Promotion requires:
(1) `scripts/build_checkpoint_manifest.py` to rebuild `manifest.json`; (2)
`scripts/sync_training_metrics.py` to rewrite `data/training_history.json` and
`data/temporal_metrics.json` from checkpoint metadata; (3) `scripts/update_readme_metrics.py`
to regenerate the README table; (4) CI's `--check` job to accept it. This exists because a
checkpoint was once promoted while the README still advertised the previous run.

---

## 11. Evaluation methodology

Five independent instruments; no single number is allowed to stand alone.

**1. Held-out benchmark — 300 prompts.** 100 curated plus 200 generated at seed `20261105`
(`data/benchmarks/generalization_prompts.json`), run through three engines on identical inputs,
reporting mean ± standard error over seeds:

| Metric | Definition |
| --- | --- |
| `valid_action_rate` | share of programs that parse and validate |
| `palette_adherence` | share of opaque pixels inside the requested palette |
| `silhouette_iou` | IoU against the corpus reference sprite for that template |
| `structural_consistency` | part count / connectivity / centring sanity |
| `colour_faithfulness` | share of opaque pixels in a prompt-named colour (HSV ±24°) |
| `element_faithfulness` | share of prompts whose element FX actually appears |
| `latency_ms` | wall-clock per request, same process, same threads |
| `pixel_fill_count` | mean opaque pixels — a proxy for detail |

**2. Golden images.** 10 categories at fixed seed `20240101`, compared by 64-bit perceptual hash
plus exact pixel equality, tolerance **0 bits / 0 px**. Any renderer change that moves a pixel
fails the suite unless deliberately accepted with `--update-goldens`.

**3. Quality gate.** `scripts/quality_gate.py` samples held-out prompts and fails on
`valid_rate < 100`, `palette_adherence < 100`, `engine_honoured < 100`, `fill ∉ [5, 70]`, and
optionally on critic-clean and colour rates. This is the guard that keeps a renderer or sampler
change from quietly regressing.

**4. Critic calibration.** Precision/recall sweep over thresholds on a held-out sprite set; the
shipping threshold is chosen for zero false positives.

**5. Continuity metrics.** Per animation: silhouette island count per frame, part-pixel
retention between frames, and flicker (frame-to-frame pixel churn) versus the procedural
baseline.

---

## 12. Results (as shipped, `main` @ `f61463e`)

### 12.1 Model inventory

| Model | Parameters | Quality | Training |
| --- | --- | --- | --- |
| SVAN Action Transformer | 2,361,304 | val loss 0.3474, token acc **98.94%** | 16,000 / 2 epochs |
| Neural Visual Critic | 369,077 | val F1 0.7799, recall 90.09%; at p=0.72: 100% precision, 0% FP | 2,000 / 6 epochs |
| TemporalSVAN | 426,918 | best val MSE 3.2831 (epoch 34 of 80) | 1,931 / 469 |
| SocketPredictor | 619,432 | attachment error 0.55 px, 100% attachment accuracy | 8 epochs |

### 12.2 300-prompt benchmark (2026-11-07 18:10:19)

| Engine | IoU | Valid | Structural | Colour faith. | Colour hit | Element | Latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| procedural baseline | 0.535 | 100% | 93.75% | 29.1% | 70.0% | 40.7% | 11.65 ms |
| untrained SVAN | — | 98.73% | — | — | — | — | — |
| **trained SVAN v3** | **0.503 ± 0.002** | **100%** | **94.9%** | **15.0%** | **41.7%** | **33.4%** | **567.56 ± 9.65 ms** |

### 12.3 What improved with v3 and what did not

* Colour faithfulness **6.1% → 15.0%**; prompts that actually get their colour **30.4% →
  41.7%**. First run where prompt colour measurably moves the raster.
* Structural consistency **94.9%**, up from the v2 run.
* Neural adoption: 25/25 sampled requests returned `candidate_source = "neural"` — the guard is
  no longer winning by default.
* **Still behind the procedural baseline on colour (15.0% vs 29.1%), element FX (33.4% vs
  40.7%) and IoU (0.503 vs 0.535).** The neural engine wins on structural consistency and on
  producing genuinely novel silhouettes; it does not yet beat the hand-written rules on prompt
  obedience. This is stated plainly because it is the honest state of the system.

### 12.4 Animation

TemporalSVAN on grammar v2 restores the 20-class animation embedding, runs on the 1,931-sequence
corpus, and is wrapped by the continuity filter (§8), which is what removes frame-to-frame limb
disappearance. Flicker is materially lower than the procedural baseline on cyclic animations;
per-action numbers live in `data/benchmarks/temporal_benchmark_results.json` and
`docs/research/STAGE_4_TEMPORAL_INTELLIGENCE.md`.

---

## 13. API surface

FastAPI, six routers, roughly fifty endpoints. Every route returns JSON; images are base64 data
URLs.

| Router | Endpoints |
| --- | --- |
| `core` | `/api/health`, `/api/models`, `/api/checkpoints/load`, `/api/metrics`, `/api/generate`, `/api/history` (get/delete single/all) |
| `sprites` | `/api/action_set/plan`, `/api/action_set`, `/api/export/sheet`, `/api/refine`, `/api/render`, `/api/export` |
| `animation` | `/api/animation/generate`, `/api/animation/regenerate_frame`, `/api/animation/metrics` |
| `assets` | `/api/asset_catalog`, `/api/composition/compose`, `/api/composition/animate`, `/api/asset_pack/export_zip`, `/api/composition/metrics`, `/api/assets` (CRUD), storage dashboard / records / orphans / sync / trash / restore / permanent_delete / empty_trash / cleanup_scan / cleanup_execute / backup / backups |
| `critic` | `/api/critic/inspect`, `/api/critic/repair`, `/api/critic/metrics` |
| `training` | `/api/dataset/stats`, `/api/dataset/samples`, `/api/dataset/generate`, `/api/training/start`, `/stop`, `/status`, `/metrics`, `/history`, `/api/benchmarks/gallery`, `/report_card`, `/run`, `/api/evaluation/results` |

OpenAPI is served at `/docs`. `/api/health` is the observability surface: model state, result
cache hit/miss, thread count, device.

---

## 14. Frontend

A React 18 SPA, static build, no state library — `useState` plus a typed `services/api.ts` client.
Nine pages, one per job:

| Page | Job |
| --- | --- |
| `Generate` | Single sprite: prompt, palette, sampler, canvas, candidate provenance |
| `AnimationStudio` | Frame timeline, onion skin, per-frame regeneration |
| `ActionSetPage` | Prompt → full action package, facing toggle, seed reroll, sprite-sheet download |
| `AssetStudio` | Equipment composition over sockets, layer toggles |
| `AssetLibrary` | Saved assets, catalogue, import/export |
| `DatasetStudio` | Corpus generation, sample browsing, split stats |
| `DataManager` | Storage dashboard, trash, backup, cleanup |
| `ModelLab` | Checkpoint load, training runs, benchmark gallery, report card |
| `EvaluationLab` | Critic inspection, repair traces, evaluation results |

Types for every response live in `frontend/src/types/index.ts` and are checked with
`tsc --noEmit` inside `npm run build`, so a backend schema change that breaks the UI fails the
frontend CI job instead of silently rendering `undefined`.

---

## 15. Engineering practice

### 15.1 Test strategy (257 tests, 72.42% coverage, floor 71%)

| Layer | Representative tests |
| --- | --- |
| DSL | parse/validate/serialize round-trips, arity errors, palette bounds, `SHADE`/`OUTLINE`/`AURA` semantics |
| Renderer | golden images, layer ordering, composite-mask outline, dithering |
| Models | forward shapes, checkpoint compatibility, best-of statistics, temporal checkpoint load |
| Pipeline | determinism per seed, cache hit/miss, engine honesty, guard fallback |
| API | endpoint contracts via `TestClient`, error paths (400 on mismatched frame sizes) |
| Data | split leakage, trash lifecycle, deletion gates, backup/restore |

The best-of test deliberately compares **means over five seeds** rather than exact equality:
batched decoding is not bit-identical to batch-of-one (float reduction order), so asserting
equality made the suite fail for a non-bug.

### 15.2 CI

`.github/workflows/ci.yml`, on every push to `main` and every pull request: backend `pytest`
with the coverage floor → `ruff check backend` → README metrics `--check` → frontend
`npm run build` (which typechecks first). Lint scope is deliberately narrow (E9, F) — it catches
real defects without arguing about style.

### 15.3 Determinism

Seeded sampler, seeded corpus generator, seeded goldens, deterministic renderer. A request with
the same `(prompt, category, style, pose, palette, seed, size)` yields byte-identical frames
across processes on the same platform. CUDA is supported but is not the determinism reference.

### 15.4 Invariants worth remembering

1. Vocabulary and checkpoint must agree — the pipeline refuses a mismatch (`checkpoint_error`).
2. A shipped checkpoint must load into the current model class; there is a test for the temporal
   one because the 18 → 20 class change broke it silently.
3. `README.md`'s metrics block is generated; edit the JSON, not the Markdown.
4. A new primitive goes into parser, validator, serializer, tokenizer **and** the finishing pass,
   and changes the vocabulary.

---

## 16. Storage and data lifecycle

Five tiers, enforced in `storage/classification.py`:

| Tier | Content | Policy |
| --- | --- | --- |
| A | `data/datasets/`, temporal corpora | **immutable** — never deleted, never trained away |
| B | `models/checkpoints/*.pt` | **immutable** |
| C | `docs/research/`, `data/benchmarks/` | **immutable** research record |
| D | `data/assets/`, `data/asset_packs/` | user-generated, deletable via soft trash |
| E | `data/cache/`, `data/trash/` | cleanable |

Deletion is always **two-step**: move to trash (reversible), then permanent delete (gated). The
guarantee under test is that Tier D and E operations cannot touch Tier A, B or C — a regression
here is a failing test, not a code-review comment.

SQLite (`data/spritemind.db`) holds seven tables: `storage_registry`, `assets`, `checkpoints`,
`datasets`, `benchmarks`, `trash`, `audit_log`. The audit log records every lifecycle action with
timestamp and actor, so "what deleted this file" is answerable. Backups are written to a
verifiable archive and can be re-scanned against the filesystem.

---

## 17. Limitations and threats to validity

Stated plainly, because a research document that lists only wins is marketing.

1. **Prompt obedience is still weak.** 15.0% of opaque pixels match a prompt-named colour; only
   41.7% of prompts get their colour at all. The procedural baseline scores higher on both
   (29.1% / 70.0%). The neural model wins on structure, not on semantics.
2. **Silhouette IoU below the baseline.** 0.503 vs 0.535: neural sprites deviate further from
   the canonical template. Part of that is desirable novelty, part is error, and the metric
   cannot currently separate the two.
3. **Resolution and framing are fixed.** 32×32 by default (16–64 supported), one character per
   canvas, centred. No scene composition, no tileset generation, no terrain.
4. **Weapons are parsed but not rendered.** `ActionPlanner` extracts a weapon noun; the renderer
   ignores it, and the attack arc is generic.
5. **Temporal loss is not normalised.** MSE 3.2831 is comparable only within the same corpus; it
   is not a percentage and cannot be compared to the older 487-sequence runs.
6. **No human evaluation set.** Every metric here is computable by code. There is no
   artist-rated preference study, which is the obvious next validation step.
7. **Critic labels are synthetic.** The critic learns defects injected by the generator; its
   precision on organically broken output is unmeasured.
8. **CPU-only reference numbers.** Latency was measured on one machine (i7-12650HX, 2 threads).
9. **Training budget capped by memory.** A 4 GiB container killed longer-context runs; the
   shipped checkpoints are not the best achievable with more compute.

---

## 18. Roadmap

Live list: [`ROADMAP.md`](../../ROADMAP.md). Open items at the time of writing:

| Priority | Item |
| --- | --- |
| P2 | ONNX / quantised export for embedding in a game engine |
| P4 | Human-rated comparison set (artist preference, not just metrics) |
| P5 | Weapon vocabulary rendering; batch generation and full sprite-pack export |
| P5 | Style transfer across palettes and eras |
| P6 | Nightly benchmark job; full security sweep (`SECURITY-REPORT.md`) |

Now unblocked by v3: **retraining the optional prompt text encoder** — the corpus finally
carries colour and element tags, so a text-conditioned run has something to learn from. That is
the highest-value next experiment, because it targets the weakest measured axis (§17.1).

---

## 19. Reproduction

```bash
# environment
python -m venv .venv && . .venv/bin/activate
pip install -r backend/requirements-dev.txt
cd frontend && npm ci && npm run build && cd ..

# run the studio
python backend/run_server.py            # http://127.0.0.1:7860  (OpenAPI at /docs)

# full check
./scripts/check.sh                      # ruff + pytest + README metrics + frontend build + gate

# the headline experiments
python scripts/finetune_svan.py --dataset data/datasets/element_synthetic_20k
python scripts/build_temporal_dataset.py --name temporal_dataset_gv2 --sequences 2400
python scripts/train_temporal.py --epochs 80 --lr 0.002

# the numbers in section 12
python -m spritemind.ml.benchmark                       # 300-prompt suite
python scripts/sync_training_metrics.py && python scripts/update_readme_metrics.py
python scripts/golden_images.py
python scripts/golden_images.py --update-goldens   # only for a deliberate render change
```

Shipped weights: `models/checkpoints/svan_checkpoint_best.pt`, `temporal_svan_best.pt`,
`visual_critic_best.pt`, `socket_predictor_best.pt`, all checksummed in `manifest.json`.

---

## 20. Glossary

| Term | Meaning |
| --- | --- |
| **PixelScript** | The drawing DSL a sprite is expressed in (14 primitives, v2.1) |
| **SVAN** | SpriteMind Visual-Action Network — the causal transformer over PixelScript tokens |
| **TemporalSVAN** | Sequence model predicting per-part kinematics per frame |
| **SocketPredictor** | Regressor placing equipment anchors (`hand_r`, `head`, `back`) |
| **AURA** | Dithered element halo primitive, tinted from `ELEMENT_RGB` |
| **Guarded rerank** | Sampling N neural candidates plus a procedural reference, then scoring them |
| **`candidate_source`** | Which engine produced the returned sprite: `neural` or `procedural_guard` |
| **Tier A–E** | Data classification scheme gating deletion (§16) |
| **Quality gate** | CI check that fails on generation regression |
| **Golden images** | Perceptual-hash regression set with 0-bit tolerance |

---

*Maintained alongside the code: when a checkpoint, metric or primitive changes, this document,
the generated block in `README.md`, `ROADMAP.md` and `docs/MODEL_TUNING.md` move together.*
