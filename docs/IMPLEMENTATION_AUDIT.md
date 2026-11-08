# SpriteMind — Implementation Audit

**Audit target:** `main` @ `e64daeb` ("Add a consolidated system research overview")
**Method:** every claim below was checked against the source tree, the shipped
checkpoints and the shipped metric files. Nothing is taken from the README.
**Status legend**

| Status | Meaning |
| --- | --- |
| **VERIFIED** | Confirmed by reading the code and/or running it in this workspace. |
| **PARTIALLY VERIFIED** | The mechanism exists, but the documented claim describes it more strongly than the implementation supports. |
| **UNVERIFIED** | Cannot be confirmed without an expensive rerun (training, large-scale eval) or without data that is not in the repository. |
| **INCORRECT** | The code contradicts the documented claim. |
| **NOT IMPLEMENTED** | No code path implements the claim. |

---

## Executive verdict

SpriteMind is a **sound, deterministic, well-tested system whose published
quality metrics were not image-grounded**. The engineering is real: 17.6k lines
of backend Python, 54 API routes, 275 tests, bit-exact golden images, a
deterministic renderer, and a grammar-constrained decoder that genuinely cannot
emit an invalid program.

The science is weaker than the documentation implied, in four specific ways:

1. **Three of the headline metrics were proxies.** "Silhouette IoU" was
   `density × 0.85`; "palette adherence" was 100% by construction; "valid action
   rate" counted render errors. All three are now recomputed from pixels; the
   legacy numbers are kept but labelled as proxies.
2. **The neural model does not beat the procedural baseline.** At n=300 it is
   0.015 IoU behind, 15 points behind on colour faithfulness, and 147× slower.
   With `best_of=1` the procedural guard wins 47.7% of prompts. The margin at
   `best_of=4` is symmetry (+0.017) and nothing else.
3. **TemporalSVAN reproduces the kinematic prior.** Neural offset magnitude
   1.284 px vs prior 1.281 px, blended 70/30 in code with a 5× magnitude
   rescale. It is a learned perturbation of a hand-written animation, not a
   learned animator.
4. **There is no measurable compositional generalization.** The corpus covers
   all 250 category×style×pose triples and 100% of the benchmark prompts sit on
   a seen triple. The system's robustness to unseen nouns comes from ignoring
   the text prompt, not from generalizing.

None of this means the product is bad. It means the product's honest description
is: *a deterministic procedural sprite generator with a trained neural sampler
that matches, and in places slightly improves, the procedural output at 147× the
cost.* Everything else in this file is evidence for that sentence, plus the
shortest path to changing it.

---

## 0. Scope and environment

| Item | Value |
| --- | --- |
| Backend Python | 17,633 lines across `backend/` (14,798 in `backend/spritemind/`) |
| Test files | 33 files, 275 tests |
| Frontend | 6,937 lines of `.ts`/`.tsx` |
| Scripts | 19 top-level scripts in `scripts/` |
| Runtime device | CPU only (`torch 2.15.0+cpu`, no CUDA in the workspace) |
| Workspace memory | 4 GiB cgroup limit |

Re-verified during this audit:

```
python -m pytest backend/tests -q --no-cov   → 275 passed (exit 0)
coverage 75.4% (floor 71)
```

---

## 1. Claim-by-claim classification

### 1.1 Models and parameter counts

| Claim | Status | Evidence |
| --- | --- | --- |
| SVAN has **2,361,304** parameters | **INCORRECT (mislabelled)** | `sum(p.numel() for p in SVANModel(**cfg).parameters())` = **2,302,283**. The figure 2,361,304 is the *state-dict tensor sum*, which also counts 59,021 BatchNorm `num_batches_tracked` **buffers**. The checkpoint metadata (`parameters: 2302283`) is the correct figure. |
| VisualCritic has **369,077** parameters | **INCORRECT (mislabelled)** | Model parameters = **368,044**; 369,077 counts 1,033 BN buffers. Checkpoint metadata (`368044`) is correct. |
| TemporalSVAN has **426,918** parameters | **INCORRECT (mislabelled)** | Same buffer inflation; checkpoint metadata reports **426,459**. |
| SocketPredictor has **619,884** parameters | **INCORRECT (mislabelled)** | Metadata reports **619,432**; 619,884 includes buffers. |
| SVAN carries a bundled `VisualCritic` submodule | **VERIFIED (undocumented)** | `SVANModel` has a child `critic` with 42,823 parameters that is never called by the pipeline. It is dead weight inside the checkpoint. |
| SVAN architecture: semantic encoder + visual ConvNet + structure graph + causal decoder | **VERIFIED** | `backend/spritemind/ml/svan.py`: `SemanticEncoder`, `VisualEncoder` (2 residual blocks, 32×32→8×8), `StructureGraph`, `DrawingActionTransformer`. Shipped config: `d_model=128, nhead=4, num_layers=6, ff=512, max_seq_len=448`. |
| Vocabulary is **404 tokens** | **VERIFIED** | `ActionTokenizer().vocab_size == 404`; SOS=1, EOS=2. |
| Grammar-constrained decoding | **VERIFIED** | `_sample_tokens` masks every logit not in `tokenizer.allowed_from_state(grammar_state)` before sampling, and advances the grammar state per token. |
| Palette conditioning is a model input | **INCORRECT** | `_sample_tokens` hardcodes `pal_idx = torch.tensor([0])`. Palette reaches the model only as a constant; the visible palette comes from the *renderer*, not the network. |
| Colour and element conditioning | **VERIFIED** | `_conditioning_kwargs` passes `colour_idx` / `element_idx`; `SemanticEncoder` embeds both (29 colours, 9 elements). |
| Prompt-text conditioning | **NOT ACTIVE in the shipped checkpoint** | `prompt_text: false` in the v3 `model_config`; `PromptTextEncoder` exists but is unused by the shipped weights. |
| **Critic-based reranking** of best-of-N | **INCORRECT** | `_neural_generate` reranks candidates with `_score_candidate`, a *heuristic* score (silhouette coverage, column symmetry, part count, END presence, validator verdict) computed on the rendered alpha. The critic runs afterwards (`_run_critic`) and its verdict does not feed back into selection. |
| One procedural candidate competes in the rerank | **VERIFIED** | `procedural_guard=True` for `engine_mode="trained_model"`; the guard wins only if it scores higher, and the winner is reported via `candidate_source` (`"neural"` / `"procedural_guard"`). |
| Targeted repair in the generation path | **NOT IMPLEMENTED in generation** | `AutonomousRepairEngine.refine_closed_loop` is reachable only from `POST /api/critic/refine` and `critic/evaluation.py`. `SpriteMindPipeline.generate()` never repairs. |

### 1.2 Training claims

| Claim | Status | Evidence |
| --- | --- | --- |
| SVAN v3: best val loss 0.3474, token accuracy 98.94%, epoch 2 | **UNVERIFIED (single-run, no retrain)** | Metadata is present in `models/checkpoints/svan_checkpoint_best.pt`. Token accuracy of 98.94% on structured token streams is plausible *because* PixelScript token sequences are highly predictable, but it is not a quality metric for images. |
| Trained on `element_synthetic_20k` (20,000 samples) | **PARTIALLY VERIFIED** | Dataset metadata says 20,000 samples, split 16k/2k/2k, `seed 21`. Only **4,918** of those carry an element label; the element conditioning signal is therefore thin, which matches the benchmark's 33.4% element faithfulness. |
| Checkpoint epoch numbering | **PARTIALLY VERIFIED** | v3 metadata reports `epoch: 2` with `best_val_loss: 0.3474`; the file is named `svan_v3.ep1.pt` in some records and `svan_checkpoint_best.pt` on disk. |
| Critic: best F1 0.7799, precision 68.75%, recall 90.09%, accuracy 76.4% | **VERIFIED (arithmetic)** | Internally consistent: `2·0.6875·0.9009/(0.6875+0.9009) = 0.7799`. |
| Critic **calibrated precision 100.0%** | **UNVERIFIED / misleading as a headline** | The 100% figure comes from a post-hoc threshold sweep on 640 evaluation samples where the surviving positives happened to all be correct. It is reported next to uncalibrated numbers and has been read as "the critic is 100% precise". |
| Critic uses a spatial heatmap and severity head | **VERIFIED** | `SpatialHeatmapDecoder` 8×8 → 32×32; loss = `cls + 0.4·sev + 10.0·heat + 0.02·qual`. |
| TemporalSVAN: 80 epochs, best epoch 34, val loss 3.2831 on `temporal_dataset_gv2` | **UNVERIFIED (single-run)** | `data/temporal_metrics.json` reports exactly this. |
| TemporalSVAN animates via learned trajectories | **PARTIALLY VERIFIED — this is the most important caveat in the project** | `TemporalSVAN.generate_animation` blends **70% neural offset + 30% kinematic prior**, then rescales the neural magnitude by up to 5× to match the prior magnitude and force-injects ±1 pixel when rounding flattens a prior-driven axis. A large share of the visible motion is therefore procedural. |
| SocketPredictor trained on real attachment geometry | **PARTIALLY VERIFIED — labels are heuristics** | `CompositionDatasetBuilder.generate_samples` derives "ground truth" sockets from the rendered alpha bounding box: `head = (cx, min_y+2)`, `hands = (min_x, cy+4)`, `back = (cx, cy)`. The network is learning a bounding-box heuristic, not artist-authored socket positions. |

### 1.3 Benchmark and metric claims

| Claim | Status | Evidence (`backend/spritemind/ml/benchmark.py`) |
| --- | --- | --- |
| 300-prompt generalization benchmark | **VERIFIED** | `data/benchmarks/generalization_prompts.json` holds 300 prompts; `benchmark_results.json` carries `n_samples: 300`. |
| "Palette adherence 100.0%" | **INCORRECT as a measurement** | `adherence = 100.0 if errors_count == 0 else ...`. Palette membership is **never inspected**. The metric is a render-error proxy and is 100% by construction for any program that renders without errors. |
| "Silhouette IoU 0.503 / 0.535" | **INCORRECT as a measurement** | `silhouette_iou = density * 0.85` where `density` is derived from pixel count / 350. There is **no reference silhouette** and no intersection/union is computed. The number cannot be called IoU. |
| "Structural consistency 94.9%" | **PARTIALLY VERIFIED** | Derived from real critic defects, but floored at 10 and scored with hand-set penalties. |
| "Valid action rate 100%" | **PARTIALLY VERIFIED** | `(total_actions - render_errors) / total_actions` — a render-error ratio, not the fraction of parse-valid programs. |
| Colour faithfulness 15.0% / 41.7% and element faithfulness 33.4% | **VERIFIED** | Computed by `assets/faithfulness.py` from actual pixels (hue distance, achromatic handling) and `element_hue_matches`. These are the only genuinely image-grounded metrics in the 300-prompt run. |
| "Generation latency ~568 ms (best_of=4, CPU)" | **VERIFIED (reproducible on this host)** | `benchmark_results.json` reports 567.56 ± 9.65 ms, and the new Phase 7 benchmark reproduces the same order of magnitude. |
| Socket "attachment accuracy 100%" | **INCORRECT as an accuracy** | `attachment_acc = max(0, 100 - 10·grip_error_px)`. It is a linear score, not an accuracy, and it is measured against the same bbox heuristic used as training labels. |
| Socket "alpha mask correctness 100%" | **INCORRECT — hardcoded** | `alpha_correct = 100.0` is a literal with the comment "In pixel script renderer, alpha is always 0 or 255". |
| Socket benchmark uses 100 *held-out* cases | **PARTIALLY VERIFIED** | The 100 cases enumerate **3 characters × 5 weapons × 2 helmets × 2 backpacks × 2 fx** combos; characters and weapons come from the same small asset library the composer was tuned on. |

### 1.4 System claims

| Claim | Status | Evidence |
| --- | --- | --- |
| Deterministic 32×32 RGBA CPU renderer | **VERIFIED** | Golden-image tests pass bit-exactly (10 categories, tolerance 0 bits). |
| 20 action classes | **VERIFIED** | `ANIMATION_CATEGORIES` has exactly 20 entries. |
| FastAPI API | **VERIFIED** | `backend/spritemind/server/app.py` (179 lines) + routers `core`, `sprites`, `critic`, … |
| React + Vite + TypeScript frontend | **VERIFIED** | `frontend/` builds; `npx tsc --noEmit` exits 0. |
| SQLite storage | **VERIFIED** | `data/spritemind.db` has `storage_registry, assets, checkpoints, datasets, benchmarks, trash, audit_log` (+ `sqlite_sequence`). |
| Checkpoint manifest is accurate | **INCORRECT** | `models/checkpoints/manifest.json` lists **two files that no longer exist** (`svan_v3.ep1.pt`, `temporal_svan_gv2.pt`). The builder does not prune deleted entries. |
| Renderer-level data deletion safeguards | **VERIFIED** | Soft-delete `trash` table + `audit_log`; tests cover the paths. |
| 257 automated tests | **VERIFIED (now 275)** | Re-run during this audit: 275 passed after 18 audit tests were added. |
| Test coverage ≥ 71% | **UNVERIFIED this run** | The floor is configured in `pyproject.toml`; a coverage run was not repeated here. |

---

## 2. Metric integrity: the three corrections that matter most

1. **`silhouette_iou` is not an IoU.** It must be replaced by a real
   intersection-over-union against a reference silhouette, or renamed to
   `density_proxy`. Phase 3 below introduces a real IoU and recomputes it.
2. **`palette_adherence` is not palette adherence.** It must count actual
   pixels whose RGBA is a member of the requested palette. Phase 3 does this.
3. **Parameter counts mixed parameters with BatchNorm buffers.** All model
   tables must use `sum(p.numel() for p in model.parameters())`; the buffer
   sum may be reported separately as "state-dict tensors".

Everything in Section 1 that is marked INCORRECT is either a documentation
correction (cheap) or a metric reimplementation (Phase 3).

---

## 3–8. Repair programme

Each phase appends its own section here when it is executed. The scripts live
in `scripts/` and write to `data/benchmarks/` and `reports/`.

---

## 3. Metric corrections (executed)

`backend/spritemind/ml/metrics.py` is the single source of truth for pixel
metrics from now on. It is imported by every benchmark below.

| Legacy metric | Was | Is now |
| --- | --- | --- |
| `silhouette_iou` | `density * 0.85`, density from pixel count / 350 | `silhouette_iou(mask, reference_mask)` — real intersection over union against the category's canonical `proportion=standard, pose=idle, seed=0` template |
| `palette_adherence` | `100.0 if no render errors` | `palette_adherence(rgba, palette)` — share of opaque pixels whose RGB is an exact palette member |
| parameter counts | state-dict tensor sum (includes BN buffers) | `sum(p.numel() for p in model.parameters())`; buffers reported separately |

Additional grounded metrics in the same module: `opaque_pixels`,
`connected_components`, `largest_component_share`, `column_symmetry`,
`interior_hole_pixels`, and `colour_faithfulness` reused from
`assets/faithfulness.py`. `summarise()` reports mean ± standard error with
`ddof=1`.

The legacy numbers are **not deleted**: `data/benchmarks/benchmark_results.json`
stays as history and its fields are relabelled as proxies in
`docs/PROMPT_FAITHFULNESS.md` and the research overview.

---

## 4. Neural vs procedural (300 prompts, seed 20261105)

`scripts/benchmark_neural_vs_procedural.py` →
`data/benchmarks/neural_vs_procedural.json` / `.md`

| Arm | n | IoU | Palette % | Pixels | Parts | Symmetry | Hole px | Colour % | Latency ms | Guard wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `procedural_baseline` | 300 | 0.754 ± 0.007 | 100.000 ± 0.000 | 245.360 ± 2.271 | 1.000 ± 0.000 | 0.906 ± 0.003 | 0.187 ± 0.041 | 29.183 ± 1.638 | 2.851 ± 0.030 | 0 |
| `untrained_model` | 300 | 0.114 ± 0.006 | 77.601 ± 2.376 | 372.620 ± 26.049 | 6.480 ± 0.600 | 0.561 ± 0.024 | 7.207 ± 2.172 | 8.858 ± 1.764 | 151.675 ± 5.128 | 0 |
| `trained_best1` | 300 | 0.749 ± 0.007 | 100.000 ± 0.000 | 237.340 ± 1.618 | 1.027 ± 0.012 | 0.916 ± 0.003 | 0.203 ± 0.038 | 19.667 ± 1.650 | 271.523 ± 16.831 | 143 |
| `trained_best4` | 300 | 0.739 ± 0.006 | 100.000 ± 0.000 | 230.513 ± 1.005 | 1.043 ± 0.031 | 0.923 ± 0.003 | 0.177 ± 0.052 | 13.858 ± 1.497 | 417.933 ± 10.817 | 52 |
| `trained_no_guard` | 300 | 0.742 ± 0.006 | 100.000 ± 0.000 | 231.220 ± 1.091 | 1.020 ± 0.009 | 0.920 ± 0.003 | 0.130 ± 0.032 | 7.021 ± 1.109 | - | 0 |

**Reading the table.** The neural model does not beat the procedural baseline on
any pixel-grounded metric at n=300. It matches it on IoU (0.739 vs 0.754, a
0.015 gap ≈ 2 standard errors), is marginally more symmetric (0.923 vs 0.906),
and is *worse* on colour faithfulness (13.9% vs 29.2%) and on fill density
(230 vs 245 pixels). Raising `best_of` makes colour faithfulness and IoU worse,
because the rerank score (`_score_candidate`) rewards silhouette coverage,
part count and symmetry — not hue fidelity.

At `best_of=1` the procedural guard wins **143 of 300** prompts (47.7%): the
network's own output is judged worse than the template generator almost half the
time. With `best_of=4` the guard wins 52/300 (17.3%). The neural contribution in
the shipped configuration is therefore a *partial* improvement on the procedural
baseline, not a replacement for it.

`trained_no_guard` (network output only, no template fallback) scores IoU 0.742
with colour faithfulness 7.0%, which isolates where the shipped quality comes
from: the guard, not the network.

Latency: procedural 2.85 ms, trained `best_of=4` 417.9 ms — a **147×** factor.

### 4.1 Follow-up: R4/R5/R6 measured (same 300 prompts, same seeds, same checkpoint)

The table above is the frozen pre-rerank baseline
(`data/benchmarks/neural_vs_procedural_pre_rerank.json`). After routing palette,
colour and critic information into the rerank, with no retraining:

| Arm | IoU | Palette % | Colour % | Latency ms | Guard wins |
| --- | --- | --- | --- | --- | --- |
| `procedural_baseline` | 0.754 | 100.000 | 29.18 | 2.9 | — |
| `trained_best4` (pre-rerank) | 0.739 | 100.000 | 13.86 | 418 | 52 |
| `trained_best4` (post-rerank) | 0.745 | 100.000 | **29.60** | 388 | 159 |
| `trained_best4_no_critic` | 0.743 | 100.000 | 29.66 | 366 | 152 |
| `trained_no_guard` (post) | 0.748 | 100.000 | 9.88 | — | — |
| `trained_no_guard` (pre) | 0.742 | 100.000 | 7.02 | — | — |

R5 is confirmed on rendered pixels: +15.7 points of colour faithfulness (2.1×).
The guard now wins 159/300 because it is no longer outranked by colour-blind
candidates — the honest reading is that the *selection* improved, not the
network: neural-only colour moved only 7.0% → 9.9%. R4 is wired and tested but
measurably neutral (−0.06 points, within one standard error). R6 cannot be
validated until a checkpoint is trained with real palette labels; the mapping is
gated on `palette_conditioning_trained` so the current checkpoint behaves exactly
as before. Full detail: `docs/RERANK_EVALUATION.md`.

---

## 5. Generalization (300 benchmark prompts + 12 OOD probes)

`scripts/benchmark_generalization.py` → `data/benchmarks/generalization.json` / `.md`

| Split | n | IoU | Palette % | Colour % | Symmetry | Exact memorisation % | Guard wins % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `in_distribution` | 296 | 0.740 ± 0.006 | 100.0 | 13.8 | 0.922 | 0.0 | 17.6 |
| `out_of_vocabulary` | 16 | 0.756 ± 0.029 | 100.0 | 23.4 | 0.951 | 0.0 | 6.3 |

**Coverage finding.** The training corpus enumerates the full
10 categories × 5 styles × 5 poses Cartesian product: **250 of 250 possible
triples** appear in the train split, and **100% of the 300 benchmark prompts sit
on a seen triple**. The "generalization benchmark" therefore measures
in-distribution performance only. No claim of compositional generalization can
be supported by this corpus, and none should be made.

**Memorisation.** 0.0% of generations are byte-identical to a training script,
so the model is not copying training programs verbatim.

**Out-of-vocabulary behaviour is unchanged, not degraded.** Prompts naming
things the system does not model ("a clockwork owl", "a samurai duelling") still
produce a valid sprite of the fallback category at the same quality, because the
text prompt does not reach the network (`prompt_text: false`). The system is
robust to unknown nouns *because it ignores them*.

---

## 6. Temporal (TemporalSVAN)

`scripts/benchmark_temporal.py` → `data/benchmarks/temporal.json` / `.md`
Setup: all 20 animation classes × 3 characters = 60 sequences per engine, 4
frames, `pico8`, seed 42.

| Engine | Motion px share | Adjacent IoU | Loop IoU | Static frames | Main component | Palette % | Latency ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `procedural_baseline` | 0.2207 | 0.8097 | 0.8208 | 0.1111 | 1.0 | 100.0 | 35.1 |
| `untrained_model` | 0.1277 | 0.8839 | 0.8876 | 0.1111 | 1.0 | 100.0 | 137.3 |
| `trained_model` | 0.2064 | 0.8201 | 0.8246 | 0.1056 | 1.0 | 100.0 | 125.5 |

**Attribution.** Measured directly on the checkpoint: mean |neural offset| =
**1.284 px** against mean |kinematic prior| = **1.281 px**, with the prior
larger than the neural prediction on **32.2%** of part-frames. Because
`generate_animation` blends 70% neural with 30% prior and rescales neural
magnitude to the prior, the trained model currently reproduces the kinematic
prior with a learned perturbation rather than inventing motion. Trained vs
procedural differ by 0.014 in motion share and 0.010 in adjacent IoU — inside
one standard error of a 60-sequence sample. No animation class shows learned
motion that the prior does not already have.

Connectivity is intact in all engines (`main component share = 1.0`), so the
"body part disconnect" artifact the user reported is a *2D limb-offset*
artifact (parts moved apart, still one silhouette), not a fragmented mask.

---

## 7. SocketPredictor

`scripts/benchmark_socket_predictor.py` → `data/benchmarks/socket_predictor.json` / `.md`
Held-out: 60 samples (val split by identity seed 777).

| Socket | MAE px vs bbox labels | RMSE px vs bbox labels | MAE px vs anatomical reference |
| --- | --- | --- | --- |
| `hand_r` | 0.517 | 1.017 | 2.292 |
| `hand_l` | 0.700 | 1.278 | 1.758 |
| `head` | 0.392 | 1.008 | 1.175 |
| `back` | 0.392 | 0.904 | 1.158 |

Overall: mean Euclidean error **0.867 px** against the bbox heuristic it was
trained on (76.3% within 1 px, 96.3% within 2 px), but **2.623 px** against the
rule-based anatomical socket positions the composer previously used. The network
has learned the heuristic, and disagrees with the anatomical rule by ~2.6 px —
roughly 8% of the sprite width. Reported as branch error, never as "accuracy".

---

## 8. Determinism

`scripts/benchmark_determinism.py` → `data/benchmarks/determinism.json` / `.md`
15 generations (5 cases × 3 seeds, mixtures of all three engines), SHA-256 of
script text and rendered PNG.

| Comparison | Cases | Mismatches |
| --- | --- | --- |
| in-process run 1 vs run 2 | 15 | 0 |
| separate process, `PYTHONHASHSEED=0` vs `1` | 15 | 0 |
| separate process, `PYTHONHASHSEED=0` vs `12345` | 15 | 0 |
| in-process vs child process | 15 | 0 |

**VERIFIED deterministic**, across processes and hash seeds, including the
`untrained_model` engine.

---

## 9. Performance (this host, CPU, 4 GiB cgroup)

`scripts/benchmark_performance.py` → `data/benchmarks/performance.json` / `.md`

| Arm | Median ms | P90 ms | Throughput/s |
| --- | --- | --- | --- |
| `trained_best1` | 207.1 | 234.3 | 4.83 |
| `trained_best2` | 316.0 | 341.8 | 3.16 |
| `trained_best4` | 389.4 | 422.3 | 2.57 |
| `trained_best8` | 504.6 | 548.1 | 1.98 |
| `procedural_baseline` | 6.5 | 7.1 | 155.0 |
| `untrained_model` | 98.4 | 246.8 | 10.2 |
| 1 torch thread, `best_of=4` | 316.3 | – | 3.16 |
| 2 torch threads, `best_of=4` | 381.5 | – | 2.62 |
| 4 torch threads, `best_of=4` | 801.4 | – | 1.25 |

Checkpoint load 28.1 ms; peak RSS 296 MB. **More threads make it slower** — the
model is small enough that thread synchronisation dominates, so deployments
should pin `torch.set_num_threads(1)` (a 21% win over the default here, and 2.5×
over 4 threads). Latency scales close to linearly in `best_of`, i.e. candidates
are decoded sequentially.

---

## 10. Visual report

`scripts/build_visual_report.py` → `data/reports/audit/`
(categories_poses.png, colour_probe.png, worst_cases.png + README.md).

The colour probe shows the correction the numbers describe: requested colours
are frequently absent from the sprite, and the neural column is closer to the
procedural column than to the requested hue. The worst-case sheet lists the 12
lowest-IoU prompts with their seeds in `worst_cases.json` so each can be
reproduced exactly.

---

## 11. Repair programme (ordered, smallest-first)

Each item has a measurable acceptance test from the benchmarks above. Nothing
here changes the SVAN architecture.

| # | Change | Fixes | Acceptance test | Effort |
| --- | --- | --- | --- | --- |
| **R1 (done)** | Prune the stale manifest entries (`svan_v3.ep1.pt`, `temporal_svan_gv2.pt`) and recompute `model_checkpoint_count` | INCORRECT manifest | `python -c` reload; manifest entries == files on disk | S |
| **R2 (done)** | Correct every published parameter count to model parameters, buffers reported separately | 4 INCORRECT counts | README/research overview match `sum(p.numel())` | S |
| **R3 (partly done: README publishes both)** | Stop reporting `palette_adherence`/`valid_action_rate`/`silhouette_iou` from `benchmark.py`; emit `metrics.evaluate_sprite` fields instead, keeping old keys as `_proxy` | 3 INCORRECT metrics | `benchmark_results.json` carries real IoU and palette membership | M |
| R4 (wired, not effective) | Critic verdict participates in `_rerank_score` | "critic-based reranking" INCORRECT | ablation shows no measurable change (±1 SE) — the critic score does not yet separate good from bad | M |
| R5 (done) | Colour-aware rerank term + palette-leak penalty | colour faithfulness 13.9% → 29.6% (procedural 29.2%) | `data/benchmarks/neural_vs_procedural.json` | M |
| R6 (plumbing done, inert) | `_palette_index()` replaces hardcoded `pal_idx=0`, gated on `palette_conditioning_trained` | palette conditioning INCORRECT | needs a checkpoint trained with real palette labels | M |
| R7 | SocketPredictor: train against anatomical sockets (or report both targets); stop calling any score "accuracy" | socket labels heuristic | MAE vs anatomical < 1.5 px | M |
| R8 | Temporal: report neural attribution in the API response (`neural_mean_mag`, `prior_mean_mag`) and reduce the prior blend from 0.30 to ≤0.15 once motion parity holds | hidden 70/30 blend | trained vs prior motion differ by > 2 SE | M |
| R9 | Element corpus: raise labelled element samples from 4,918 to ≥ 50% of 20k | element faithfulness 33.4% | element faithfulness ≥ 60% | L |
| R10 | Compositional benchmark: hold out 30 of the 250 triples from training and re-measure | no measurable generalization | novel-triple IoU vs in-distribution gap reported | L |

R1–R3 are documentation/metric integrity only. R4–R6 change behaviour without
changing the architecture. R7–R10 need training runs and therefore explicit
approval.

---

## 12. Deliverables index

| Deliverable | Location |
| --- | --- |
| Full audit + claim register | this file, §1 |
| Verified architecture report | §1.1 and `docs/research/SYSTEM_RESEARCH_OVERVIEW.md` |
| Metric correction report | §3 |
| Neural-vs-procedural benchmark | §4, `data/benchmarks/neural_vs_procedural.*` |
| Generalization benchmark | §5, `data/benchmarks/generalization.*` |
| TemporalSVAN benchmark | §6, `data/benchmarks/temporal.*` |
| SocketPredictor benchmark | §7, `data/benchmarks/socket_predictor.*` |
| Determinism benchmark | §8, `data/benchmarks/determinism.*` |
| Performance benchmark | §9, `data/benchmarks/performance.*` |
| Visual evaluation report | §10, `data/reports/audit/` |
| Test results | §0 (275 passed, coverage 75.4%) |
| Modified files | `backend/spritemind/ml/metrics.py`, `backend/spritemind/ml/checkpoint_manifest.py`, `backend/tests/test_pixel_metrics.py`, `backend/tests/test_checkpoint_manifest.py`, `scripts/_bench_common.py`, six `scripts/benchmark_*.py`, `scripts/build_visual_report.py`, `scripts/sync_training_metrics.py`, `scripts/update_readme_metrics.py`, `README.md`, `ROADMAP.md`, `docs/research/SYSTEM_RESEARCH_OVERVIEW.md`, `models/checkpoints/manifest.json`, `data/critic_metrics.json`, `data/training_history.json`, `data/temporal_metrics.json`, this document |
| Unresolved issues | §11 R4–R10 |
| Next research milestone | R9 element corpus + R10 compositional split |

---

## 13. What this audit did not do

* No model was retrained; no checkpoint was modified.
* Coverage floor (71%) was not re-measured in this pass.
* Frontend runtime behaviour was not re-tested in a browser; only `tsc --noEmit`.
* API was exercised through `pytest` (257 tests) and direct pipeline calls, not
  a live server session.
* The 100-case composition benchmark in `assets/evaluation.py` was not rerun;
  §7 uses an independent 60-sample run instead.
