# SpriteMind documentation

Start with the root [`README.md`](../README.md) for install, quickstart and the current
benchmark numbers. Everything below is the detail behind those numbers.

## Start here

- [`research/SYSTEM_RESEARCH_OVERVIEW.md`](research/SYSTEM_RESEARCH_OVERVIEW.md) — the consolidated system reference: architecture, stack, every model and hyper-parameter, the data corpora, the evaluation methodology, the shipped numbers, the engineering invariants, and the limitations.

## Generating sprites

- [`PIXELSCRIPT_GRAMMAR.md`](PIXELSCRIPT_GRAMMAR.md) — the text language every sprite is written in: primitives, `SHADE` / `OUTLINE` finishing passes, and the validation rules.
- [`PROMPT_FAITHFULNESS.md`](PROMPT_FAITHFULNESS.md) — how well the generated colours match the prompt, how the metric is computed, and the current baselines.
- [`ACTION_SETS.md`](ACTION_SETS.md) — turning one prompt into a playable animation package: idle, walk, attack, cast and dash, and how frames stay connected.
- [`PROMPT_CONDITIONING.md`](PROMPT_CONDITIONING.md) — how colour and element tags from the prompt reach the training corpus and the model.
- [`COLOUR_RECOVERY.md`](COLOUR_RECOVERY.md) — why neural colour faithfulness looked like it regressed, the selection-vs-coverage decomposition, and the steering experiment that recovered it (R11).
- [`COVERAGE_RERANK.md`](COVERAGE_RERANK.md) — the coverage-aware reranking experiment: validation-tuned, held-out once, measured as a small and not-clearly-significant gain over the shipped ranking (R12).
- [`CANDIDATE_DIVERSITY.md`](CANDIDATE_DIVERSITY.md) — the candidate-pool diversity experiment: selection is already saturated at `best_of=4`, pool size barely moves coverage, temperature mixing does but costs critic cleanliness; includes the correction to R12's `selection_hit` (R13).

## Training and tuning

- [`MODEL_TUNING.md`](MODEL_TUNING.md) — fine-tuning SVAN, latency and sampler measurements, coherence and inter-frame continuity, and the flags on the benchmark scripts.
- [`research/README.md`](research/README.md) — stage-by-stage research write-ups, from learning validation to game-ready asset generation.
- [`research/BENCHMARK_INDEX.md`](research/BENCHMARK_INDEX.md) — every recorded benchmark run in order, with the numbers that moved.

## Running and operating

- [`STORAGE_ARCHITECTURE.md`](STORAGE_ARCHITECTURE.md) — database schemas, where datasets, checkpoints and exports live, and the invariants between them.
- [`DATA_MANAGEMENT.md`](DATA_MANAGEMENT.md) — the data lifecycle: retention, trash and restore, and what is safe to delete.
- [`GIT_RELEASE_CHECKLIST.md`](GIT_RELEASE_CHECKLIST.md) — what to check before tagging a release or making the repository public.
- [`FINAL_RELEASE_READINESS.md`](FINAL_RELEASE_READINESS.md) — the recorded state of the last maintenance pass: audit, cleanup, git verification, test and smoke-test results, known limitations, launch steps.

## Where the numbers come from

- `scripts/coherence_metrics.py` — silhouette coherence plus inter-frame continuity over a prompt set.
- `POST /api/benchmarks/run` — regenerates the 100-prompt generalisation gallery behind the report card in the Model Lab.
- `scripts/finetune_svan.py --prompt-text` — trains the optional prompt text encoder (`ml/prompt_encoder.py`) as a fourth memory token; `docs/PROMPT_FAITHFULNESS.md` explains why the shipped checkpoint leaves it off.
- `scripts/golden_images.py` — the perceptual-hash regression set (`--update-goldens` to accept a deliberate change).
- `GET /api/metrics` — p50/p95 latency ring buffer, also shown in the Model Lab.
- `pytest` — the 314-test suite also measures backend coverage (`pytest-cov`, statement + branch) and fails below the 71% floor set in `pyproject.toml`. Use `pytest --no-cov` when you only want to run one file.
- `npm run typecheck` — TypeScript over `frontend/src`; `npm run build` runs it first, so the frontend CI job typechecks too.
