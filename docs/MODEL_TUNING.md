# Model Tuning and Generation Performance

## Fine-tuning SVAN

`scripts/finetune_svan.py` continues training from the shipped checkpoint
(`models/checkpoints/svan_checkpoint_best.pt`) and writes a separate checkpoint, so the shipped
weights are never overwritten by an experiment:

```bash
# build a dataset first if needed (see below)
PYTHONPATH=backend python scripts/finetune_svan.py \
    --dataset expanded_synthetic_25k --epochs 2 --lr 0.0002 \
    --save-name svan_expanded_ft.pt --history data/finetune_expanded_history.json
```

Flags: `--dataset`, `--epochs`, `--lr`, `--batch-size`, `--threads`, `--save-name`,
`--resume-from`, `--history`. The run records per-epoch history to the `--history` path. A
checkpoint is adopted only when it beats the shipped one on the held-out benchmark below;
otherwise it is discarded and deleted.

### Building a larger dataset

`scripts/build_expanded_dataset.py` wraps `DatasetManager.generate_dataset` and writes both the
trainable layout and a flat JSON export:

```bash
PYTHONPATH=backend python scripts/build_expanded_dataset.py --samples 25000 --seed 7
```

- `data/datasets/expanded_synthetic_25k/` — `records.jsonl`, `metadata.json`, `samples/`
  (20,000 train / 2,500 val / 2,500 test, split strictly by character identity hash).
- `data/expanded_synthetic_dataset.json` — every record in one JSON array (189 MB for 25k).

Both are gitignored; the script takes ~13 s and renders 24 preview samples.

### Measured results

All rows come from `GeneralizationBenchmark` on the same machine and pipeline. The 12k runs were
scored on 40 held-out prompts; the 25k runs were re-scored on the full 100 prompts, so the
`original shipped` row is listed twice to give each comparison the right baseline. `Clean rate`
is the project quality gate over 40 prompts (share of sprites the visual critic finds free of
defects).

| Weights | Train data | Prompts | Val loss | IoU | Structural | Clean rate |
| --- | --- | --- | --- | --- | --- | --- |
| original shipped | 12k | 40 | 0.5456 | 0.407 | 89.62 | 100% |
| 3 epochs, lr 5e-4 | 12k | 40 | 0.3459 | 0.397 | 86.75 | — |
| 1 epoch, lr 1.5e-4 | 12k | 40 | 0.542 | 0.407 | 84.25 | — |
| original shipped | 12k | 100 | 0.5456 | 0.406 | 90.15 | 100% |
| 2 epochs, lr 2e-4 | 25k | 100 | 0.3431 | **0.410** | **94.45** | **67.5%** |
| +1 epoch, lr 1e-4 | 25k | 100 | 0.3423 | 0.409 | 94.70 | — |

The 25k run is the first fine-tune to move the benchmark: silhouette IoU 0.406 → 0.410 and
structural consistency 90.15 → 94.45 on the full 100 prompts. It was still **rejected**, because
the quality gate tells the opposite story: the share of sprites with critic-detected defects fell
from 100% to 67.5% over 40 prompts. Pixel-level artefacts (floating islands, detached limbs)
that the silhouette and inter-frame metrics do not see are what the critic catches.

Three findings drove the outcome:

1. **More data beats more passes over the same data.** On the original 12k, token loss fell but
   rendering degraded. On the fresh 25k (new seed, 20k training characters) both benchmark
   metrics improved at a similar validation loss, which suggests the 12k program distribution
   was the bottleneck, not optimisation.
2. **Judge checkpoints on the whole gate, not one metric.** IoU and structural consistency are
   silhouette-level; the critic's clean rate is pixel-level. A checkpoint has to satisfy both,
   and the gate is the cheaper signal to run before adopting.
3. **Evaluate on 100 prompts.** On a 40-prompt subset the 25k checkpoint looked like a tie
   (IoU 0.405 vs 0.407); the same weights win clearly over 100 prompts. Small subsets are too
   noisy to accept or reject a checkpoint on.

What would plausibly keep the 25k gains without the pixel regressions: mixed training (25k plus
a replay slice of the 12k programs), a critic-defect penalty in the training loss, or early
stopping on this clean rate directly instead of validation loss.

The shipped checkpoint is unchanged, and `models/checkpoints/manifest.json` lists exactly the
committed checkpoints (`scripts/build_checkpoint_manifest.py --prune` removes stale entries).

## Generation performance

The neural path decodes one token at a time, so latency is dominated by the number of decode
steps and the per-step transformer cost.

| Path | Before | After |
| --- | --- | --- |
| Cold generation, `best_of=4` (benchmark mean over 100 prompts) | 351 ms | 291 ms |
| Cold generation, `best_of=1` | 233 ms | ~160 ms |
| Repeated identical request | full regeneration | < 1 ms (LRU cache) |
| Quality gate `avg_latency_ms` (8 samples) | 384 ms | 277 ms |

The changes:

1. **Incremental grammar state** (`ml/tokenizer.py`): `allowed_token_ids` used to walk the whole
   prefix at every step; it now folds one token through a small state machine
   (`grammar_start_state`, `advance_grammar`, `allowed_from_state`) and caches allowed sets.
2. **Cross-attention cache reuse** (`ml/svan.py`): the memory key/value projections are computed
   once per request instead of once per decode step. Verified numerically equivalent to the full
   forward pass (max abs diff 2.4e-7).
3. **LRU result cache** (`ml/pipeline.py`): keyed by the full request, capacity from
   `SPRITEMIND_RESULT_CACHE` (default 64, `0` disables). Hit/miss counters are exposed on
   `/api/health` and each result reports `cache_hit`.

`SPRITEMIND_THREADS` defaults to 2. The model is small enough that 1–8 torch threads are within
noise of each other (430–480 ms on the same box); oversubscribing with 40 threads made
generation about 20x slower.

## Grammar v2 retrain (SHADE / OUTLINE)

Grammar v2 adds two finishing primitives (see [`PIXELSCRIPT_GRAMMAR.md`](PIXELSCRIPT_GRAMMAR.md)),
which grows the tokenizer vocabulary from 401 to 403. A v1 checkpoint cannot emit the new tokens,
so `SpriteMindPipeline.load_checkpoint` refuses a vocabulary mismatch, records the reason in
`checkpoint_error` and callers fall back to the procedural engine instead of sampling garbage.

Training run (`data/finetune_gv2_history.json`):

| Epoch | Data | lr | batch | val loss | token acc | critic clean rate | time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `gv2_synthetic_25k` + 20% `generalization_detail_12k` replay (21,925 train) | 6e-4 | 16 | 0.3639 | 98.48% | 93.8% | 55 min |
| 2 | same, resumed from epoch 1 | 4e-4 | 8 | 0.3482 | 98.68% | 95.8% | 44 min |

Both epochs cleared the gate (`--min-clean 90`); epoch 2 was adopted as
`models/checkpoints/svan_checkpoint_best.pt` (2,351,639 params, vocab 403, `max_seq_len` 448).
Epoch 2 was resumed with a smaller batch because each epoch allocates close to the workspace's
4 GB cgroup limit; running a benchmark alongside training trips the OOM killer.

Result on the 100 held-out benchmark prompts:

| Metric | v1 baseline | grammar v2 |
| --- | --- | --- |
| Silhouette IoU vs. procedural target | 0.406 | 0.495 |
| Valid action rate | 100% | 100% |
| Palette adherence | 100% | 100% |
| Structural consistency | 99.1% | 99.25% |
| Mean latency | 351 ms | 465 ms |

Latency rose with the longer programs (v2 scripts average 308 tokens, p95 380). Silhouette
coherence (`scripts/coherence_metrics.py`, 24 generalization prompts): zero prompts with detached
alpha islands, mean island pixels 0.0, full dark-rim coverage on the silhouette border.

Quality gate note: the critic clean rate on the 40-prompt gate is seed sensitive — 100% with a
fixed seed, 72.5% with per-prompt seeds. Always compare checkpoints at the same seeds and read the
clean rate alongside the defect breakdown rather than a single number.

## Endpoint latency

`scripts/latency_report.py` drives the real FastAPI app through `TestClient` and writes
`data/benchmarks/latency_report.json`:

| Endpoint | p50 | p95 |
| --- | --- | --- |
| `POST /api/generate` | 12.6 ms | 44.5 ms |
| `POST /api/generate` (cache hit) | 2.6 ms | 3.2 ms |
| `POST /api/action_set` | 138.8 ms | 152.4 ms |
| `POST /api/action_set/plan` | 3.3 ms | 4.2 ms |
| `GET /api/models` | 23.3 ms | 24.5 ms |
| `GET /api/health` | 1.7 ms | 2.7 ms |

`/api/generate` defaults to the procedural engine, which is why it is two orders of magnitude
below the 465 ms neural benchmark mean. Pass `engine="trained_model"` to pay the neural cost;
the LRU cache still collapses repeat requests.

Regenerate with `python scripts/latency_report.py --requests 50`.
