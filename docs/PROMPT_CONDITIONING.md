# Prompt conditioning in the synthetic corpus

The colour-faithfulness metric ([`PROMPT_FAITHFULNESS.md`](PROMPT_FAITHFULNESS.md)) showed the
trained model at **6.1%** while the procedural corpus managed only **10.1%**. Both numbers are low
for the same reason: prompts carried no colour words, so no conditioning signal ever existed for the
model to learn. This change builds that signal in the *data* before touching the model.

## What a sample now carries

`DatasetManager.generate_dataset` tags roughly 60% of samples with a colour and 25% with an element
(mutually exclusive), and the prompt string is generated from the tags:

```
a red knight with a sword
an ice-touched slime
```

Records gain `colour` and `element` fields alongside the existing `category`/`style`/`pose`/… keys,
so a future text encoder can be trained and *evaluated* against prompt words that actually exist in
the data.

## Palette-aware colour mapping

`backend/spritemind/dataset/conditioning.py` owns the mapping. Two details matter:

1. **Hue-first nearest entry.** Plain RGB distance maps `blue` onto cyan, `grey` onto a lavender
   entry and `black` onto navy — pico8 is only 16 colours. `_colour_cost` weights hue delta at 4×
   saturation and 3× value for chromatic targets, and compares achromatic targets by luminance only.
2. **Only tag colours the palette can express.** `renderable_colours("pico8")` paints an 8×8 block of
   each candidate entry and keeps the name only if the faithfulness metric scores it ≥99%. pico8 can
   show 22 of the 28 names; `lime`, `indigo`, `purple`, `violet` and `magenta` are excluded because no
   entry lands within 24° of their hue. The corpus never promises a colour the palette cannot render.

`ramp()` returns `(base, highlight, shadow)` and prefers hue siblings of the base entry, so a red
knight gets three steps of red rather than grey shading. Elements tint the body (element-only
prompts) or the eyes (colour + element prompts).

## Held-out probe

`python scripts/probe_conditioning.py --samples 200` renders fresh prompts and scores them:

| measurement | before | after |
| --- | --- | --- |
| colour present (metric: not "missing") | 80.8% | **100.0%** |
| colour silhouette share | — | 41.6% |
| element visible on element prompts | 0.3% | **100.0%** |
| element hue share | — | 46.1% |

Colour *presence* is the number the roadmap requires (>90% faithful to its own tags). The ~40%
silhouette share is expected: the outline, eyes and metal details are legitimately not the body
colour. Element share is measured against the palette entry the element resolves to, not against a
nominal RGB target, because pico8 cannot show a true poison green.

## Numeric conditioning in SVAN

A text encoder on these short prompts did not move the metrics (see
[`MODEL_TUNING.md`](MODEL_TUNING.md)), so the signal reaches the model the
cheap way: as embedding ids. `conditioning.py` exposes `COLOUR_IDS`/`ELEMENT_IDS` and
`colour_id()`/`element_id()` (0 = none, otherwise stable alphabetical ids), and
`SemanticEncoder` concatenates `colour`/`element` embeddings onto its conditioning vector next to
category, style, pose and palette. `SVANModel` stores `num_colours=29` and `num_elements=9` in its
config, so a checkpoint carries the embedding table it was trained with, and
`PixelScriptDataset`/`training.py` feed both ids into the train and validation loops while
`pipeline.py` supplies them at sampling time from the same prompt parser.

The corpus adds the matching supervision: `element_synthetic_20k` (20k records) tags 60% colour
(12,062) and 25% element (4,918), and the generator's finishing pass emits an `AURA` halo tinted
from `ELEMENT_RGB` whenever an element is set, so the network can learn the halo as well as the
body tint. Regenerate with the same builder to change those rates.

## Still open

Element coverage is deliberately the smaller slice (25% of records), which keeps the body-tint
branch dominant; if the element/AURA branch needs more data, raise the element rate in the corpus
builder and retrain rather than oversampling the existing records.

The prompt text encoder still exists behind `prompt_text=False` with a null-embedding fallback;
with numeric ids carrying colour and element, it is only worth enabling if a future prompt carries
structure the ids cannot express.
