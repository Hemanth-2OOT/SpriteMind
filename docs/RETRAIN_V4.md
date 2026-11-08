# R6/R7/R9/R10 — held-out evaluation and the v4 retrain

Scope of this milestone: make generalization measurable (R10), give the palette
embedding real labels (R6), label more elements (R9) and train socket targets
from anatomy instead of the silhouette (R7) — **without touching the production
checkpoint, the production corpus, or any existing artefact**.

Everything below is reproducible from the repository; commands are given with
each artefact.

## 1. Frozen held-out split (R10)

`scripts/build_heldout_split.py` wrote `data/splits/heldout_v1.json` once, from
seed `20261105`:

| item | value |
| --- | --- |
| spec version / digest | `1` / `581bd3fa4c1253723cd8e7169d45847cb209d0a09ea99b9b65cecd830086a06b` |
| held-out (category, style, pose) triples | 30 (3 per category) |
| held-out (category, element) pairs | 10 (1 per category) |
| selection rule | per category: 1 element pair + 3 triples, ranked by a seeded SHA-256 of the combination, so it is reproducible but not hand-picked |

Why combinations and not records: every earlier corpus covers 250/250
category×style×pose triples, so a random record split leaves the same
combination in train and test and measures memorisation, not generalization.
Anything whose combination is in the spec is written to `split = "heldout"` and
never enters train/val.

Guards:

* the spec digest is SHA-256 over the sorted body; `load_spec()` refuses a file
  whose digest does not match, so the evaluation set cannot drift silently;
* `spritemind.dataset.splits.audit()` lists held-out combinations found in
  train/val; `backend/tests/test_heldout_split.py` asserts the corpus metadata
  records the same digest.

## 2. Conditioned corpus (R6 + R9)

Built by `scripts/build_conditioned_dataset.py` into
`data/datasets/cond_v1_15k/` (new directory; nothing existing was rewritten):

```
python scripts/build_conditioned_dataset.py --samples 15000 --name cond_v1_15k
```

| property | value |
| --- | --- |
| records | 15,000 (train 10,903 / val 1,483 / heldout 2,614) |
| palette | each program is generated **in** its recorded palette; 5 palettes in rotation |
| requested colour | drawn from `renderable_colours(palette)`, recorded in `colour` |
| element | 7,533/15,000 labelled (50% target rate, previously 25%) |
| leakage prevention | held-out combinations excluded from train/val + identity hash partition |
| provenance | `metadata.json`: label rules, per-palette and per-element counts, split counts, spec digest, builder commit |

This fixes the two data gaps behind R6 and R9: the old corpora had no `palette`
field at all (so `pal_idx` was always 0) and only a quarter of records carried an
element.

## 3. Palette labels reach the model (R6 plumbing)

* `spritemind.dataset.conditioning.palette_id()` maps a palette name to the same
  index order the pipeline's `_palette_index()` uses.
* `PixelScriptDataset` now reads the record's `palette` field instead of
  hardcoding `0`; records without one still fall back to 0, so older corpora train
  exactly as before (`backend/tests/test_palette_labels.py`).
* `finetune_svan.py --palette-conditioning` builds `SVANModel` with
  `palette_conditioning=True`, which is what sets `palette_conditioning_trained`
  on the saved checkpoint — the flag the pipeline already gates on.

## 4. Anatomical socket targets (R7)

The shipped `SocketPredictor` was trained on labels taken from the rendered alpha
bounding box. Measured against those labels it looks accurate (0.87 px); measured
against the body it is 2.62 px off, because a bbox edge is not a hand.

`spritemind.dataset.anatomical_sockets.anatomical_sockets()` derives targets from
the part pixels the renderer already tracks:

* `head` — centre of the drawn `head` part;
* `back` — centre of the drawn `torso` part;
* `hand_r` / `hand_l` — distal pixel of the drawn `arm_right` / `arm_left`
  (`max(x+y)` / `max(y−x)`), so a raised arm moves the socket with it.

Fallbacks (part absent) use the part anchor, then a proportional guess, so
species without arms (slime, animal) still yield valid sockets.
`scripts/train_socket_anatomical.py` trains a new checkpoint and evaluates three
arms on the same held-out samples.

Measured on 225 held-out renders (`data/benchmarks/socket_anatomical.json`):

| arm | mean error | hand_r | hand_l | head | back |
| --- | --- | --- | --- | --- | --- |
| procedural proportional prior | 1.803 px | 1.902 | 2.080 | 2.706 | 0.524 |
| shipped `socket_predictor_best.pt` | 2.469 px | 3.133 | 2.473 | 2.690 | 1.578 |
| `socket_predictor_anatomical_v2.pt` | **0.349 px** | 0.415 | 0.390 | 0.386 | 0.203 |

The new checkpoint meets the R7 target (< 1.5 px) by a wide margin. It is **not**
wired into the production path: switching the socket model changes placed
accessories, which is a separate review decision.

## 5. Controlled retrain (new checkpoint, production untouched)

```
python scripts/finetune_svan.py --dataset cond_v1_15k --palette-conditioning \
    --epochs 1 --lr 0.0003 --save-name svan_v4_palette.pt --resume-from svan_checkpoint_best.pt
python scripts/finetune_svan.py --dataset cond_v1_15k --palette-conditioning \
    --epochs 1 --lr 0.00015 --save-name svan_v4_palette_ep2.pt --resume-from svan_v4_palette.pt
```

| checkpoint | val loss | val token acc | critic clean rate (16 samples) |
| --- | --- | --- | --- |
| `svan_checkpoint_best.pt` (v3, production) | — | — | baseline reference |
| `svan_v4_palette.pt` (1 epoch) | 0.3516 | 98.78% | 87.5% |
| `svan_v4_palette_ep2.pt` (2 epochs) | 0.3423 | 99.09% | 87.5% |

The finetune script's adoption gate (clean rate ≥ 90%) rejected both epochs, so
nothing was promoted automatically and `svan_checkpoint_best.pt` is byte-identical
to before (sha256 `16c2cd7310a167dd6ac3ecc64dfbf1d03d1c11f68dc8ba3b1e6f87d23ca2d71e`
in the manifest, unchanged). Only the second-epoch candidate is stored in the repo;
re-run the first command to regenerate the one-epoch candidate. That gate is why the production model cannot drift by accident.

## 6. Held-out evaluation (R10) — identical prompts, seeds and settings

`scripts/benchmark_heldout.py` builds 80 held-out prompts (30 triples × 2 + 20
category/element prompts, 80 matched in-distribution prompts), runs `best_of=4`,
palette `pico8`, seed `20261105`, and reports the network alone and the shipped
guarded path separately.

| checkpoint | regime | arm | IoU | colour % | palette % | guard wins |
| --- | --- | --- | --- | --- | --- | --- |
| v3 (production) | held-out | neural only | 0.7682 | 19.81 | 100 | — |
| v3 (production) | held-out | production | 0.7789 | 48.63 | 100 | 19/80 |
| v3 (production) | in-distribution | neural only | 0.7599 | 21.00 | 100 | — |
| v3 (production) | in-distribution | production | 0.7512 | 46.93 | 100 | 12/80 |
| v4 ep1 | held-out | neural only | 0.7840 | 17.14 | 100 | — |
| v4 ep1 | held-out | production | 0.7829 | 49.29 | 100 | 27/80 |
| v4 ep1 | in-distribution | neural only | 0.7623 | 20.00 | 100 | — |
| v4 ep1 | in-distribution | production | 0.7622 | 47.10 | 100 | 16/80 |
| v4 ep2 | held-out | neural only | **0.7904** | 17.43 | 100 | — |
| v4 ep2 | held-out | production | **0.7901** | 49.08 | 100 | 20/80 |
| v4 ep2 | in-distribution | neural only | **0.7767** | 15.36 | 100 | — |
| v4 ep2 | in-distribution | production | 0.7698 | 47.29 | 100 | 10/80 |
| procedural baseline (n=300, other benchmark) | — | — | 0.754 | 29.18 | 100 | — |

Read this honestly:

* **Neural geometry improved.** Neural-only IoU rose +2.2 points on held-out
  prompts (0.7682 → 0.7904) and +1.7 on in-distribution (0.7599 → 0.7767), now
  above the procedural baseline's 0.754 without the guard. The guarded path rose
  on both grids too.
* **Neural colour faithfulness appeared to fall** (19.81 → 17.43 held-out,
  21.00 → 15.36 in-distribution). **This measurement was wrong in the harness,
  not only in the checkpoints**: the neural-only arm of `benchmark_heldout.py`
  never passed the requested colour into `_neural_generate`, so both checkpoints
  were scored on a colour they were never told. Re-measured with the request
  actually passed (same spec, seeds and arms), the numbers are 48.52 (v3) and
  47.46 (v4 ep2) held-out, and 25.82 / 46.45 in-distribution — see
  [`COLOUR_RECOVERY.md`](COLOUR_RECOVERY.md) (R11). The guarded path is unchanged
  (48.63 → 49.08 held-out), because it always passed the request.
* **No held-out gap was measured**; both checkpoints score *higher* on held-out
  prompts than on the in-distribution grid. That is not evidence of generalization
  being solved: the IoU reference is each category's canonical *idle* template, so
  the in-distribution grid (which contains more attack/walk poses) is scored more
  harshly, and the held-out grid is dominated by idle-ish prompts. The grids are
  not metric-matched, so cross-regime deltas are comparisons of prompt mix, and
  only *same-regime, old-vs-new* rows should be read as checkpoint improvements.
* Palette conditioning (R6) now demonstrably affects the drawing: with actions
  only (metadata stripped) the network produces 1.00 distinct programs across 5
  palettes for v3 vs 2.25 (ep1) and 2.50 (ep2) for the conditioned checkpoints
  (`data/benchmarks/palette_conditioning.json`).

## 7. What was not done

* `svan_v4_palette_ep2.pt` is **not** the production model. Promotion needs a
  separate review, and the neural colour regression above is a reason to hold.
* `socket_predictor_anatomical_v2.pt` is trained and benchmarked but not wired
  into `SocketPredictorManager`.
* R8 (temporal blend) was deferred by the user and is untouched.
* The 300-prompt `neural_vs_procedural` benchmark was not re-run against v4; the
  held-out harness is the comparison used here, on identical prompts and seeds for
  both checkpoints.

## 8. Reproduce

```
python scripts/build_heldout_split.py --seed 20261105          # frozen spec, refuses to overwrite
python scripts/build_conditioned_dataset.py --samples 15000 --name cond_v1_15k
python scripts/finetune_svan.py --dataset cond_v1_15k --palette-conditioning \
    --epochs 1 --lr 0.0003 --save-name svan_v4_palette.pt
python scripts/train_socket_anatomical.py --samples 1500 --epochs 15
python scripts/benchmark_heldout.py --checkpoint svan_checkpoint_best.pt \
    --label svan_v3_production --out data/benchmarks/heldout_v1_baseline.json
python scripts/benchmark_heldout.py --checkpoint svan_v4_palette_ep2.pt \
    --label svan_v4_palette_ep2 --out data/benchmarks/heldout_v1_v4_ep2.json
python scripts/benchmark_palette_conditioning.py
```

Artefacts: `data/splits/heldout_v1.json`, `data/datasets/cond_v1_15k/`,
`data/benchmarks/heldout_v1_baseline.json(.md)`, `heldout_v1_v4.json(.md)`,
`heldout_v1_v4_ep2.json(.md)`, `socket_anatomical.json`,
`palette_conditioning.json`, `data/reports/v4_comparison.png`,
`data/finetune_v4_history.json`, `data/finetune_v4_ep2_history.json`.
