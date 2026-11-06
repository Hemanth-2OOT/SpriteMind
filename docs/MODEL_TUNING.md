# Model Tuning and Generation Performance

## Fine-tuning SVAN

`scripts/finetune_svan.py` fine-tunes the shipped action transformer
(`models/checkpoints/svan_checkpoint_best.pt`) on `generalization_detail_12k` and writes a
separate checkpoint, so the shipped weights are never overwritten by an experiment:

```bash
PYTHONPATH=backend python scripts/finetune_svan.py
```

The run records per-epoch history to `data/finetune_history.json`. To adopt a checkpoint, it
has to beat the shipped one on the held-out benchmark below; otherwise it is discarded.

### Measured results

Both runs were evaluated on the same 40 held-out benchmark prompts with
`GeneralizationBenchmark`, on the same machine and pipeline.

| Weights | Val loss | Val token acc | Silhouette IoU | Structural consistency | Pixels filled |
| --- | --- | --- | --- | --- | --- |
| shipped `svan_checkpoint_best.pt` | 0.5456 | 98.53% | **0.407** | **89.62** | 186.2 |
| 3 epochs, lr 5e-4, dropout 0.05, label smoothing 0.03 | **0.3459** | 98.55% | 0.397 | 86.75 | 181.6 |
| 1 epoch, lr 1.5e-4, dropout 0.10, label smoothing 0.05 | 0.542 | 98.55% | 0.407 | 84.25 | 186.2 |

Both fine-tunes lower (or match) teacher-forced validation loss but do not improve held-out
rendering: silhouette IoU stalls and structural consistency drops. Token-level loss on this
dataset saturates, so more epochs over the same 12k programs mostly model the training
distribution rather than the prompt-to-program mapping. The shipped checkpoint was kept.

What would plausibly move held-out quality: new synthetic programs (more categories, props and
poses) rather than more passes over the current ones, and program-level metrics in the training
loop instead of token accuracy alone.

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
