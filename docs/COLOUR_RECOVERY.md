# R11 — Neural colour recovery

Scope: find out why neural colour faithfulness appeared to fall after the v4
palette-conditioned retrain, and whether the neural generator can be made to
produce correctly coloured sprites on its own — without copying the procedural
reference. Production is untouched: no checkpoint promoted, no evaluation split
changed, nothing merged.

## 1. The regression was mostly a measurement bug

`scripts/benchmark_heldout.py` built prompts that carried `colour`, but its
neural-only arm called `_neural_generate(...)` without passing `colour` (the
guarded arm always did). The neural model was therefore scored on a request it
had never been given, and both checkpoints were punished for "wrong colour".

Fix: the benchmark now forwards `colour` and `element` to the neural arm, so the
comparison measures colour *generation*, not colour *blindness*. Baseline files
from before the fix stay in the repo as history
(`heldout_v1_baseline.*`, `heldout_v1_v4.*`); everything below was produced with
the corrected harness, same frozen spec `581bd3fa…`, same seeds and best_of.

| checkpoint | regime | neural IoU | neural colour % | guarded IoU | guarded colour % |
| --- | --- | --- | --- | --- | --- |
| v3 `svan_checkpoint_best.pt` | held-out | 0.7707 | 48.52 | 0.7789 | 48.63 |
| v4 `svan_v4_palette_ep2.pt` | held-out | 0.7814 | 47.46 | 0.7901 | 49.08 |
| v4 + steering 3.0 | held-out | **0.7856** | **52.36** | 0.7901 | 49.08 |
| v5 `svan_v5_colour.ep1.pt` | held-out | 0.7782 | 47.48 | 0.7798 | 48.77 |
| v3 | in-distribution | 0.7514 | 25.82 | 0.7512 | 46.93 |
| v4 | in-distribution | 0.7587 | 46.45 | 0.7698 | 47.29 |
| v4 + steering 3.0 | in-distribution | 0.7696 | **55.63** | 0.7698 | 47.29 |
| v5 | in-distribution | 0.7453 | 46.07 | 0.7595 | 47.73 |

Read this as: the v3 → v4 colour difference is ~1 point, not the 2.4 points the
blind harness reported, and v3's low in-distribution 25.82 shows the old
checkpoint is the one that actually collapses on coloured prompts that also name
an element.

## 2. Decomposition: selection vs generation

`scripts/diagnose_colour.py` scores 40 coloured prompts (best_of=4, pico8) and
separates two failure modes: did the program pick a palette entry matching the
requested hue (`sel_hit`), and how much of the sprite ended up in that colour.

| checkpoint | arm | faithfulness | sel_hit | faith\|sel | requested share | colours used |
| --- | --- | --- | --- | --- | --- | --- |
| v3 | neural, no colour given | 12.20 | 0.775 | 15.75 | 0.069 | 6.72 |
| v3 | neural, colour given | 48.38 | 1.000 | 48.38 | 0.221 | 5.85 |
| v4 ep2 | neural, no colour given | 11.95 | 0.675 | 17.70 | 0.061 | 6.08 |
| v4 ep2 | neural, colour given | 46.27 | 1.000 | 46.27 | 0.201 | 5.83 |
| v3 | guarded | 45.27 | 1.000 | 45.27 | 0.211 | 6.20 |
| v4 ep2 | guarded | 45.27 | 1.000 | 45.27 | 0.211 | 6.20 |

Conclusion: when the request reaches the network, palette *selection* is perfect
(1.000) and the remaining loss is *coverage* — the model paints with the right
hue but on too few pixels. So neither retraining the label corpus nor adding
palette labels would fix the number that mattered.

## 3. Controlled experiments

Each experiment was measured on the frozen split, neural-only and guarded
separately, at the same seed and best_of=4.

**E1 — colour supervision (corpus).** `data/datasets/cond_v2_colour`, 15 000
records, every record carries a colour label (`--colour-rate 1.0`), elements at
50%, disjoint from the held-out spec. Purpose: remove the 40% of records that
taught "no colour requested".

**E2 — inference-time colour steering.** New `colour_steer` bias added to the
matching `COL_*` logits inside `_sample_tokens` / `_sample_batch` /
`_neural_generate`, threaded through `SpriteMindPipeline.generate`. Default 0.0
is byte-identical to the old sampler (`test_steering_off_matches_the_plain_sampler`).

**E3 — colour-weighted objective.** `--colour-loss-weight` multiplies the
cross-entropy of `COL_*` targets (`colour_token_weights()` in
`backend/spritemind/ml/training.py`); weight 1.0 reproduces the original loss
exactly. One epoch from v4 on the E1 corpus, lr 1.5e-4, label smoothing kept
identical to the unweighted path.

| experiment | held-out IoU | held-out colour % | in-dist IoU | in-dist colour % | verdict |
| --- | --- | --- | --- | --- | --- |
| v4 reference | 0.7814 | 47.46 | 0.7587 | 46.45 | — |
| E1+E3 (v5 ep1) | 0.7782 | 47.48 | 0.7453 | 46.07 | **negative** |
| E2 steering 1.5 | 48.56 (diag) | — | — | — | small gain |
| E2 steering 3.0 | 0.7856 | 52.36 | 0.7696 | 55.63 | **positive** |

E1 and E3 were trained as one stage, so their individual contributions are not
separable — stated plainly: the combination bought nothing (+0.02 colour,
−0.003 IoU held-out), and repeating an epoch to isolate them was not justified
by that result. v5 was not adopted (critic clean rate 87.5%).

E2 is the only change that moved the metric, and it moves *generation*: the
program already chose a matching palette entry, steering raises the probability
of painting with it. Nothing is copied from the procedural baseline.

## 4. Acceptance gate (R11)

| gate item | requirement | result |
| --- | --- | --- |
| Neural colour faithfulness | improve over v4 | **pass** with E2 steering: 47.46 → 52.36 held-out, 46.45 → 55.63 in-distribution. E1+E3: no change (fail) |
| Neural geometry | retain v4 improvement | **pass**: 0.7856 / 0.7696, above v4's 0.7814 / 0.7587 |
| Palette adherence | 100% | **pass**: 100.0 in every arm and regime |
| Determinism | no regression | **pass**: steering 3.0, 15 cases × 4 process comparisons, 0 mismatches (`data/benchmarks/determinism_steer3.*`) |
| Critic gate | ≥ 90% clean | **fail**: 87.5% with and without steering on the 16 coloured held-out prompts — steering is neutral on the critic, and the gate remains unmet |
| Procedural comparison | reported separately | **pass**: guarded path unchanged (0.7901 / 49.08 held-out); neural-only is reported beside it everywhere above |

Because the critic gate fails, **nothing is promoted**: `svan_checkpoint_best.pt`
stays production, `colour_steer` defaults to 0.0 and is not exposed through the
API or the UI.

## 5. Reproduce

```bash
python scripts/diagnose_colour.py --out data/benchmarks/colour_diagnosis.json
python scripts/benchmark_heldout.py --checkpoint svan_v4_palette_ep2.pt \
    --label v4_conditioned --out data/benchmarks/heldout_v1_v4_conditioned.json
python scripts/benchmark_heldout.py --checkpoint svan_v4_palette_ep2.pt \
    --label v4_steer3 --colour-steer 3.0 --out data/benchmarks/heldout_v1_v4_steer3.json
python scripts/build_conditioned_dataset.py --name cond_v2_colour --samples 15000 \
    --seed 20261108 --colour-rate 1.0 --element-rate 0.5
python scripts/finetune_svan.py --dataset cond_v2_colour \
    --resume-from svan_v4_palette_ep2.pt --lr 0.00015 --palette-conditioning \
    --colour-loss-weight 3.0 --save-name svan_v5_colour --history data/finetune_v5_history.json
SPRITEMIND_COLOUR_STEER=3 SPRITEMIND_BENCH_SUFFIX=_steer3 python scripts/benchmark_determinism.py
```

## 6. Where the remaining colour loss is

Coverage, not selection. The requested colour occupies ~0.20–0.24 of the sprite
against ~0.21 for the procedural baseline, so the gap is now mostly in *how much*
of the character is filled with the asked-for hue. Next candidates, in order of
expected value per hour:

1. **Structured colour plans** — emit a colour choice per body part instead of
   one global palette index with free coordinates (grammar change, needs its own
   version bump).
2. **Coverage-aware reranking** — score `matched_share` in the reranker, not only
   presence, then re-measure. Cheap, no retraining.
3. **Curricula on coverage** — train only on records where the procedural
   reference gives the requested colour a large share, to teach the mapping
   without teaching copying.
