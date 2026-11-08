# R12 — Coverage-aware reranking (isolated experiment)

**Status: measured, not promoted.** `COVERAGE_RERANK_WEIGHT` stays at `0.0`, so the
shipped ranking is unchanged. Nothing here is exposed in the API or UI, and the
critic gate was not met.

## 1. The question

R11 established two separate facts about colour:

* colour **selection** — whether the sprite contains the requested hue at all — is
  solved by conditioning (100% of conditioned candidates hit);
* colour **coverage** — how much of the sprite the hue fills — is not, and
  inference-time `colour_steer` was the only lever that moved it (+2.4 pp).

R12 asks a narrower question: *if the generator is left exactly as it is, can a
better candidate-ranking signal raise coverage instead?* That separates
"generation is wrong" from "we pick the wrong candidate".

## 2. Protocol

| Rule | How it was honoured |
| --- | --- |
| Tune only on validation data | `data/splits/validation_v1.json` — 24 coloured `pico8` prompts built by `scripts/build_validation_prompts.py`, digest `7c6216ee8a694ef5…`, asserted disjoint from the held-out triples |
| Evaluate once on the frozen split | `data/splits/heldout_v1.json` (digest `581bd3fa4c1253723cd8e7169d45847cb209d0a09ea99b9b65cecd830086a06b`), 80 prompts, seed `20261105`, `best_of=4` |
| Same candidates across policies | One candidate pool per prompt (4 neural + 1 procedural guard) is measured once, then every ranking policy selects from that same pool |
| Colour selection and coverage kept apart | Reported as `selection_hit` (share of prompts with any matched pixels) and `coverage_pct` (mean share of opaque pixels in the requested hue) |
| Neural-only and guarded reported separately | `*/neural_only` ranks only the neural candidates; `*/production` also allows the procedural guard |
| No retraining, no checkpoint changes | `svan_v4_palette_ep2.pt` used for both phases |

Tuning rule, fixed before the held-out run: over a grid of weights × targets,
keep configs whose silhouette IoU is within 0.01 of that arm's own baseline, and
among those pick the highest neural-only coverage (ties → smaller weight).

## 3. Implementation

`SpriteMindPipeline` now splits scoring in two: `_rerank_components()` renders a
candidate once and measures the heuristic, palette leak, colour score, coverage
and critic verdict; `_score_components()` applies the weights. That way a
ranking policy can be varied without re-rendering, and the experiment measures
the same code path that production uses.

The coverage term is a **saturating shortfall**:

```
score -= COVERAGE_RERANK_WEIGHT * max(0.0, target - matched_share)
```

It credits coverage up to `target` and stops, so a candidate cannot win by
flooding the sprite with the requested hue. `COVERAGE_RERANK_WEIGHT = 0.0`
short-circuits the term, so the shipped ranking is byte-identical.

## 4. Tuning on the validation split

Baseline (shipped ranking, `steer 0`): neural coverage 42.88%, IoU 0.8013;
production 42.70%, IoU 0.7725. With `steer 3`: neural 44.49%, IoU 0.7816;
production 44.31%, IoU 0.7586.

Grid (`data/benchmarks/coverage_rerank_tuning.json`, `…_steer3.json`):

| weight | target | neural coverage % | neural IoU |
| --- | --- | --- | --- |
| 0 (baseline) | — | 42.88 | 0.8013 |
| 25 | 0.50 | 43.07 | 0.7978 |
| 100 | 0.50 | 43.37 | 0.7967 |
| 400 | 0.50 | 43.53 | 0.7815 |

Selected: weight `100`, target `0.50` for `steer 0`; weight `400`, target `0.50`
for `steer 3` (its IoU *rose* to 0.7854).

**Read the tuned value carefully.** Achievable coverage sits around 0.25–0.45, so
any target above that makes the hinge algebraically identical to a plain linear
bonus on `matched_share`. The tuned setting is therefore best described as "the
same signal, with more weight", not as a cover/no-cover decision. Targets below
the achievable range invert the term into a *penalty* on coverage (masking as
over-paint protection) and made results worse — that variant was discarded.

## 5. Held-out result (frozen split, 80 prompts)

`data/benchmarks/coverage_rerank_heldout.json` / `.md`. Coverage is reported with
its standard error (σ ≈ 3.0 pp).

| arm | coverage % | selection hit | IoU | opaque px | palette % | defects | clean share | guard share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| steer0 / existing / neural_only | 23.73 | 0.500 | 0.7814 | 237.9 | 100.0 | 0.263 | 0.763 | — |
| steer0 / coverage / neural_only | 24.41 | 0.500 | 0.7794 | 239.4 | 100.0 | 0.275 | 0.750 | — |
| steer0 / existing / production | 24.54 | 0.500 | 0.7784 | 236.9 | 100.0 | 0.238 | 0.763 | 0.275 |
| steer0 / coverage / production | 25.19 | 0.500 | 0.7767 | 238.0 | 100.0 | 0.238 | 0.763 | 0.275 |
| steer3 / existing / neural_only | 26.18 | 0.500 | 0.7856 | 238.4 | 100.0 | 0.250 | 0.775 | — |
| steer3 / coverage / neural_only | 26.62 | 0.500 | 0.7858 | 243.2 | 100.0 | 0.275 | 0.750 | — |
| steer3 / existing / production | 26.30 | 0.500 | 0.7810 | 236.8 | 100.0 | 0.225 | 0.775 | 0.238 |
| steer3 / coverage / production | 26.66 | 0.500 | 0.7838 | 241.1 | 100.0 | 0.238 | 0.763 | 0.213 |

## 6. What this means

1. **Coverage-aware reranking helps, but only marginally and not demonstrably
   beyond noise.** +0.68 pp neural and +0.66 pp production at `steer 0`; +0.44 /
   +0.36 pp at `steer 3` — all inside one standard error. Colour steering alone
   moves coverage by +2.45 pp, roughly four times as much, at no IoU cost.
2. **Selection is not the remaining bottleneck.** `selection_hit` is 0.50 in
   every arm including the baseline: half of the held-out coloured prompts have
   no matching pixels in *any* ranked candidate. No ranking of equally colourless
   candidates can fix that; it needs the generator or the corpus.
3. **The "over-painting" hypothesis was not supported.** Coverage hovers at a
   quarter of the sprite; nothing needed capping. The saturating form still
   matters for safety, but it bounds a behaviour that does not occur.
4. **Geometry and palette are safe.** IoU moves by −0.002 (neural, `steer 0`) and
   −0.0017 (production) at most, well inside the 0.01 tolerance; palette
   adherence is 100% in all eight arms; connected components, interior holes and
   column symmetry are unchanged within rounding.
5. **The critic gate still fails.** Clean share 0.75–0.775 against a 90% gate, and
   the coverage arms are not cleaner than the existing ranking. Nothing is
   promotable.

## 7. Decision

* Coverage-aware reranking is **not** enabled in production: weight stays `0.0`,
  no API parameter, no UI control.
* The evidence points back to R11 §6: the gap is in generation (the model does
  not paint the colour at all in half the coloured prompts), not in ranking.
  Further work on reranking would need a *candidate pool that contains the
  answer*, i.e. more diverse sampling, not a better score.
* Determinism is unaffected: with the default weight the ranking is identical,
  and the determinism benchmark shows 0 mismatches.

## 8. Reproducing

```bash
python scripts/build_validation_prompts.py
python scripts/benchmark_coverage_rerank.py --mode tune
python scripts/benchmark_coverage_rerank.py --mode tune --steer 3 \
    --out data/benchmarks/coverage_rerank_tuning_steer3.json
python scripts/benchmark_coverage_rerank.py --mode eval
```
