# R13 — Neural candidate diversity

**Status: measured, not promoted.** No retraining, no checkpoint change, no API/UI
exposure. The critic gate (≥ 90% clean) still fails, so nothing is promoted.

## 0. Correction to R12, found while building this experiment

R12 reported `selection_hit = 0.500` and concluded that "half of the coloured
held-out prompts have no matching pixels in any candidate". **That was wrong.**
Half of the 80 held-out prompts name no colour at all (`colour = None`).
`colour_faithfulness()` correctly returns `None` for them, and both benchmark
scripts collapsed that `None` to `0.0` before thresholding, so 40 uncoloured
prompts were scored as colour misses.

Measured properly, **selection is 100 %**: all 40 coloured prompts contain the
requested hue at `best_of=4`, and at every larger pool. R12's other numbers keep
their direction but were diluted by the same 50 %: its "coverage" figure was a
mean that included 40 structural zeros. Both scripts now exclude uncoloured
prompts from colour metrics (`coverage` and `selection_hit` are reported over
coloured prompts only, with `coverage_all_prompts` kept for continuity) and R12's
held-out table has been re-run.

This is the second measurement bug of this family (the first was R11's missing
colour conditioning in the harness). The lesson repeats: a metric that can be
"not applicable" must never be silently coerced to zero.

## 1. The question

R12 showed ranking cannot manufacture colour that is absent from the pool. R13
asks whether the pool is the limiting factor:

* does a bigger pool (`4 → 8 → 16`) contain more *correct* sprites?
* do different sampling temperatures produce genuinely different candidates?
* if the pool already contains the answer, is the shipped ranking the reason we
  don't see it? (answered by comparing selection against an oracle over the pool)

Protocol: frozen `heldout_v1` (80 prompts, 40 coloured), `svan_v4_palette_ep2.pt`,
seed `20261105 + prompt index`, `max_actions=64`, `pico8`, shipped ranking weights
(no coverage term). Each candidate is measured once (render, heuristic, colour
selection/coverage, palette leak, critic) and every arm selects from an
already-measured pool, so arms differ only in sampling and selection.

Arms:

| arm | samples | samples per decision | note |
| --- | --- | --- | --- |
| `best_of_4` | 16 | 4 | nested prefix of the uniform 16-pool |
| `best_of_8` | 16 | 8 | nested prefix |
| `best_of_16` | 16 | 16 | equal-compute partner of `four_pools_of_4` |
| `temp_mixed_8` | 16 | 8 | 2 each at T = 0.7 / 0.9 / 1.1 / 1.3 |
| `temp_mixed_16` | 16 | 16 | 4 each at T = 0.7 / 0.9 / 1.1 / 1.3 |
| `four_pools_of_4` | 16 | 4 × 4 | four independent 4-pools, one decision each |

`best_of_4/8/16` are prefixes of a single 16-candidate batch, which is what
"best_of=k from the same RNG stream" means; a production `best_of=4` call draws a
different stream, so its number is expected to differ slightly, not systematically.

## 2. Selection: does the pool contain the colour?

`selection_hit` over the 40 coloured prompts — **1.000 in every arm**, and
`hit_any` over the union of all 48 candidates per prompt is also 1.000.

| pool | candidates | hit_any | max coverage | unique sprites | pixel disagreement | program distance |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | 4 | 1.000 | 0.248 | 0.963 | 0.154 | 0.014 |
| uniform | 8 | 1.000 | 0.259 | 0.917 | 0.156 | 0.015 |
| uniform | 16 | 1.000 | 0.268 | 0.873 | 0.154 | 0.013 |
| mixed T | 8 | 1.000 | 0.255 | 0.897 | 0.150 | 0.008 |
| mixed T | 16 | 1.000 | 0.355 | 0.925 | 0.255 | 0.119 |
| union | 48 | 1.000 | 0.734 | — | — | — |

**Answer to the research question R13 was built around: the model already knows
how to generate the requested colour and does so within the first four candidates
every time.** Pool growth cannot raise reachability because reachability is
already saturated. The constraint is *how much* of the sprite carries the colour,
not whether the colour appears.

The diversity numbers are the interesting part: at uniform temperature the 16
candidates are nearly the same *program* (action-set Jaccard distance 0.013)
despite differing on 15 % of pixels — the sampler emits one structure with
stochastic detail, not 16 independent ideas. Mixing temperatures breaks that:
program distance 0.119 and pixel disagreement 0.255 at the same pool size.

## 3. Coverage, geometry and quality

| arm | coverage % | ±SE | IoU | ±SE | palette % | defects | clean | sampling s* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `best_of_4` | 48.78 | 2.46 | 0.7659 | 0.0133 | 100.0 | 0.238 | 0.762 | 0.80 |
| `best_of_8` | 49.05 | 2.40 | 0.7607 | 0.0124 | 100.0 | 0.225 | 0.775 | 0.80 |
| `best_of_16` | 49.69 | 2.34 | 0.7445 | 0.0125 | 100.0 | 0.225 | 0.775 | 0.80 |
| `temp_mixed_8` | 48.85 | 2.46 | 0.7516 | 0.0131 | 100.0 | 0.263 | 0.750 | 1.49 |
| `temp_mixed_16` | **56.13** | 3.04 | 0.7411 | 0.0141 | 100.0 | 0.338 | 0.675 | 1.49 |
| `four_pools_of_4` (320 decisions) | 47.40 | 1.21 | 0.7588 | 0.0065 | 100.0 | 0.241 | 0.766 | 1.49 |

\* single 16-candidate batch, idle machine; the two mixed/4×4 arms pay ~1.9×.

Paired, per prompt (coloured prompts with non-zero baseline coverage, n = 40):

* `best_of_16 − best_of_4`: **+0.91 pp** (SE 0.63), 19 prompts improved, 4 worse.
* `temp_mixed_16 − best_of_4`: **+7.35 pp** (SE 2.27), 30 improved, 7 worse.
* `temp_mixed_16 − best_of_16`: **+6.44 pp** (SE 2.18), 26 improved, 10 worse.

Reading:

1. **Pool size is a weak lever.** Four times the candidates buys +0.9 pp coverage
   (1.4 SE) and costs −0.02 IoU. Diminishing returns start immediately.
2. **Temperature mixing is a real lever.** +7.4 pp at 3.2 SE over `best_of=4`,
   and +6.4 pp at ~3 SE over the equally sized uniform pool. It is the sampling
   distribution, not the pool size, that matters.
3. **It is not free.** `temp_mixed_16` loses 0.025 IoU and drops clean share from
   0.762 to 0.675 (defects 0.238 → 0.338): hot samples sometimes fracture the
   silhouette. Palette adherence stays 100 % — nothing leaks off-palette.
4. **Equal compute favours concentration.** Sixteen samples spent on one decision
   (49.69 %) beat sixteen samples spread over four decisions (47.40 %), and
   batching them in one decode pass is 1.7–1.9× faster (0.80 s vs 1.49 s in an
   idle run; absolute times scale with machine load, the ratio holds).
5. **Memory is not a constraint.** Peak RSS 342 MB across all arms; a 16-candidate
   batch costs no measurable extra memory over 4.

## 4. Where the headroom actually is

Oracle max coverage over the pool is 0.268 at `k=16` uniform and 0.355 at
`k=16` mixed, but 0.734 over the 48-candidate union. The headroom is real, but it
is only reachable by sampling *different distributions*, not more of the same
one — and the shipped ranking (a heuristic/critic blend with no coverage term)
captures almost none of it beyond the first four candidates.

Summary: **selection was never the bottleneck for reachability; coverage is a
generation property; temperature diversity moves it, pool size barely does, and
neither passes the 90 % critic gate.**

## 5. What this means for the roadmap

* Do **not** raise the production `best_of` to 8 or 16: 4× compute for +0.9 pp
  coverage and slightly worse geometry.
* A temperature-mixed sampler is the only sampling change with a statistically
  solid gain; it needs a guard against the clean-rate loss (0.675) before any
  production consideration. That is a separate milestone (R14), not part of this
  one.
* The critic gate remains the binding constraint: 0.675–0.775 clean against 0.90.
  Improving critic/repair coverage is worth more than further sampling work.

## 6. Reproducing

```bash
python scripts/benchmark_candidate_diversity.py --label r13        # ~6 min, CPU
python scripts/benchmark_coverage_rerank.py --mode eval            # corrected R12 table
```
