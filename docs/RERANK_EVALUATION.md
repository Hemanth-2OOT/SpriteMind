# Rerank evaluation — R4, R5, R6

**Question:** how much of SpriteMind's quality gap is caused by the selection step
rather than by the network, using the existing `svan_checkpoint_best.pt` unchanged?

**Protocol:** the same 300 prompts, the same seeds, the same checkpoint and the same
renderer as `data/benchmarks/neural_vs_procedural_pre_rerank.json`. Only the ranking
function changed. Every number is computed from rendered RGBA by
`spritemind.ml.metrics`; nothing here is a proxy.

## What changed

| Fix | Before | After |
| --- | --- | --- |
| **R6** palette conditioning | `pal_idx = 0` hardcoded in both samplers | `_palette_index()` maps the requested palette name to its embedding row, clamped to the table, and is gated on `model.palette_conditioning_trained` because every shipped checkpoint was trained with `palette_idx == 0` for every sample |
| **R5** colour-aware rerank | rank = structural heuristic only | rank += `0.5 × (rendered colour faithfulness − 15)` |
| **R4** critic-aware rerank | the critic ran after selection and its verdict was discarded | rank += `0.3 × (critic quality − 60)`, and the score is computed once per candidate and reused for the procedural guard so both arms are judged identically |
| palette safety | — | rank −= `40 × share of opaque pixels outside the palette` (added after the first R5 run let 0.18% of sprites through with off-palette pixels) |

`use_critic` and `use_colour` are explicit arguments and are part of the result cache
key, so an ablation cannot be served from a stale cache entry. The pipeline result
now reports `rerank: {colour_weight, critic_weight, critic_used}`.

## Results (n = 300, seed 20261105, mean ± standard error)

| Arm | IoU | Palette % | Colour % | Latency ms | Guard wins |
| --- | --- | --- | --- | --- | --- |
| procedural baseline | 0.754 ± 0.007 | 100.000 | 29.18 ± 1.64 | 2.9 | — |
| trained, best_of=4, **before R4/R5/R6** | 0.739 ± 0.006 | 100.000 | 13.86 ± 1.5 | 418 | 52 / 300 |
| trained, best_of=4, **after R4/R5/R6** | 0.745 ± 0.007 | 100.000 | **29.60 ± 1.65** | 388 | 159 / 300 |
| trained, best_of=4, critic term off | 0.743 ± 0.006 | 100.000 | 29.66 ± 1.65 | 366 | 152 / 300 |
| trained, best_of=1, after | 0.751 ± 0.007 | 100.000 | 29.31 ± 1.65 | 194 | 211 / 300 |
| neural only, best_of=4, before | 0.742 ± 0.006 | 100.000 | 7.02 ± 1.11 | — | — |
| neural only, best_of=4, after | 0.748 ± 0.006 | 100.000 | 9.88 ± 1.37 | — | — |

## What the numbers actually mean

1. **R5 works, measured on pixels.** Production colour faithfulness went from 13.9%
   to 29.6% (+15.7 points, 2.1×) with unchanged prompts, seeds and checkpoint. The
   criterion was not "the score changed"; it is the same metric recomputed from the
   rendered sprite.
2. **Most of that gain is selection, not generation.** With the guard disabled the
   network itself only moved from 7.0% to 9.9%. The network still under-colours; the
   rerank now stops the pipeline from *choosing* the most colour-blind of its own
   candidates. Neural-only quality is therefore still below the procedural baseline
   (9.9% vs 29.2%), which is exactly why "neural ≥ procedural" is the primary open
   objective.
3. **R4 is correct but not yet useful.** Turning the critic term off changes colour
   by −0.06 points and guard wins by 7 of 300, both inside one standard error; it
   costs ~22 ms. The wiring is tested (`test_rerank_decision_follows_the_critic`,
   `test_critic_term_equals_the_reported_quality_score_delta`) but the critic's
   `quality_score` does not separate good from bad sprites well enough to steer
   selection. Reporting this as an improvement would be dishonest.
4. **R6 cannot be validated on the current checkpoint.** The palette embedding rows
   1–4 were never trained (the dataset wrote `palette_idx = 0` for every sample), so
   feeding a real index would inject a random vector. The plumbing is in place, gated
   on `palette_conditioning_trained`, and is covered by four tests; it activates only
   when a checkpoint is trained on real palette labels. `test_palette_index_*` proves
   the mapping, including the clamp and the unknown-palette fallback.
5. **No regressions in the guards.** Palette adherence is back at 100.000% on every
   trained arm once the leak penalty is in. Determinism was re-run after the change:
   0 mismatches over 15 cases across in-process repeats and fresh processes with
   `PYTHONHASHSEED` 0/1/12345. Grammar validity is unchanged because candidates still
   come from the same grammar-masked decoder. Production latency fell slightly
   (418 → 388 ms) because the winning candidate is now often the procedural guard.

## Reproducing

```bash
python scripts/benchmark_neural_vs_procedural.py --limit 300
python scripts/benchmark_determinism.py
```

The pre-fix table is preserved verbatim in
`data/benchmarks/neural_vs_procedural_pre_rerank.json`; that file is the baseline for
every future rerank claim.
