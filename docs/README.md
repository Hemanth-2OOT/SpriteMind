# SpriteMind documentation

Start with the root [`README.md`](../README.md) for install, quickstart and the current
benchmark numbers. Everything below is the detail behind those numbers.

## Generating sprites

- [`PIXELSCRIPT_GRAMMAR.md`](PIXELSCRIPT_GRAMMAR.md) — the text language every sprite is written in: primitives, `SHADE` / `OUTLINE` finishing passes, and the validation rules.
- [`PROMPT_FAITHFULNESS.md`](PROMPT_FAITHFULNESS.md) — how well the generated colours match the prompt, how the metric is computed, and the current baselines.
- [`ACTION_SETS.md`](ACTION_SETS.md) — turning one prompt into a playable animation package: idle, walk, attack, cast and dash, and how frames stay connected.
- [`PROMPT_CONDITIONING.md`](PROMPT_CONDITIONING.md) — how colour and element tags from the prompt reach the training corpus and the model.

## Training and tuning

- [`MODEL_TUNING.md`](MODEL_TUNING.md) — fine-tuning SVAN, latency and sampler measurements, coherence and inter-frame continuity, and the flags on the benchmark scripts.
- [`research/README.md`](research/README.md) — stage-by-stage research write-ups, from learning validation to game-ready asset generation.
- [`research/BENCHMARK_INDEX.md`](research/BENCHMARK_INDEX.md) — every recorded benchmark run in order, with the numbers that moved.

## Running and operating

- [`STORAGE_ARCHITECTURE.md`](STORAGE_ARCHITECTURE.md) — database schemas, where datasets, checkpoints and exports live, and the invariants between them.
- [`DATA_MANAGEMENT.md`](DATA_MANAGEMENT.md) — the data lifecycle: retention, trash and restore, and what is safe to delete.
- [`GIT_RELEASE_CHECKLIST.md`](GIT_RELEASE_CHECKLIST.md) — what to check before tagging a release or making the repository public.

## Where the numbers come from

- `scripts/coherence_metrics.py` — silhouette coherence plus inter-frame continuity over a prompt set.
- `POST /api/benchmarks/run` — regenerates the 100-prompt generalisation gallery behind the report card in the Model Lab.
- `scripts/golden_images.py` — the perceptual-hash regression set (`--update-goldens` to accept a deliberate change).
- `GET /api/metrics` — p50/p95 latency ring buffer, also shown in the Model Lab.
