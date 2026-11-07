# Prompt colour faithfulness

The prompt currently reaches the model only as `category`, `style` and `pose`, so
"A blue wizard" and "A red wizard" condition the decoder identically. Colour words
are honest casualties of that design. This metric makes the gap measurable before
we change the architecture.

## Definition

`backend/spritemind/assets/faithfulness.py` scans the prompt for colour names,
then classifies every opaque sprite pixel in HSV:

- chromatic names (`blue`, `crimson`, `emerald`, ...) match when the pixel is
  saturated, bright enough, and within ±24° of the name's hue;
- `brown` additionally requires a darker value, `tan` a lighter one;
- achromatic names (`black`, `gray`, `silver`, `white`) match by saturation and
  value instead of hue.

`score` is the percentage of opaque pixels drawn in **any** requested colour.
`missing` lists the requested colours covering less than 3% of the silhouette,
which is the failure a human notices immediately. Prompts without a colour name
return `None` so they are excluded from averages rather than scored.

## Current numbers (300-prompt benchmark, 2026-11-07 18:10:19)

| Engine | Colour faithfulness | Prompts that get their colour | Silhouette IoU |
| --- | --- | --- | --- |
| Procedural baseline | 29.1% | 70.0% | 0.535 |
| Untrained SVAN | — | — | — |
| Trained SVAN v3 | 15.0% | 41.7% | 0.503 ± 0.002 |

Earlier on `main` the same measurement read 10.1% / 30.4% for the procedural
baseline, 0.9% / 4.3% untrained and 6.1% / 30.4% trained. The procedural row moved
because the baseline now emits element and colour-tinted materials; the trained row
moved 6.1% → 15.0% with the grammar-v2.1 colour/element conditioning corpus
(`element_synthetic_20k`) and checkpoint v3.

The trained model is still *below* the rule-based baseline on colour, because the
DSL's per-category default materials occasionally land on a requested colour by
accident, while the neural decoder samples whatever its category prior prefers. Any
further prompt-driven colour work — the prompt text encoder, a colour-token
curriculum, or a post-hoc palette remap — is judged by moving the trained row above
the procedural row.

## How to run

```bash
# Gate (per-sample detail + gateable rate)
PATH=/tmp/sm-venv/bin:$PATH python scripts/quality_gate.py --limit 40 --min-colour 0

# Full 100-prompt benchmark, writes data/benchmarks/benchmark_results.json
PATH=/tmp/sm-venv/bin:$PATH python -c \
  "import sys; sys.path.insert(0,'backend'); \
   from spritemind.ml.benchmark import GeneralizationBenchmark; \
   GeneralizationBenchmark().run_benchmark()"
```

`--min-colour` is the share of colour prompts that must contain every requested
colour. It defaults to `0` so the gate records the number without failing the
build until the encoder work lands; raise it once the trained model clears the
procedural baseline.

## Prompt text encoder (experiment, #24)

`ml/prompt_encoder.py` adds a character-level encoder over the request text and
feeds its pooled embedding into the decoder as a fourth memory token. Training
turns it on with `scripts/finetune_svan.py --prompt-text`; at inference the
checkpoint's own `model_config.prompt_text` decides, so older checkpoints keep
the three-token memory and behave exactly as before.

Measured on 2026-11-07, fine-tuning the shipped checkpoint for one epoch on the
first 6,000 records of `expanded_synthetic_25k` (643 s) against the current
`svan_checkpoint_best.pt`:

| Metric | Shipped | With text encoder |
| --- | --- | --- |
| Critic clean rate (16 prompts) | 93.8% | 93.8% |
| Outline coverage / islands (16 prompts) | 1.0 / 0 | 1.0 / 0 |
| Animation continuity (4 anims, IoU) | 0.8206 | 0.8206 |
| Lost pixels across transitions | 311 | 311 |

The encoder is *live* (two different prompts shift the next-token logits by up
to 0.54) but the grammar-guided decoder masks nearly every choice, and the
training text is rebuilt from the same attributes the categorical encoders
already receive, so the extra token carries no new information. Nothing is
turned on by default.

**What unblocks it:** a corpus whose records carry `colour` and `element` tags,
and a DSL that renders them (the "Element/weapon vocabulary" item). Once
`describe_record()` sees those fields it already puts them in the text; the
faithfulness number then has something to move against.

## Known limits

- Only ~30% of the benchmark prompts name a colour, so per-run averages rest on
  ~30 samples. Use the full benchmark before quoting a delta.
- Hue matching is deliberately generous; a "blue" sprite painted navy and cyan
  both count. It measures *did the colour arrive*, not art direction.
- Compound colours (`blue-green`, `rose gold`) are not understood; the metric
  sees only the first matching name.
