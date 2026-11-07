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

## Still open (P1)

The corpus now contains the signal; SVAN does not read it yet. The remaining P1 work is the prompt
text encoder (character embedding → cross-attention, with a null-embedding fallback so the current
checkpoint keeps loading), then a retrain and a re-run of the benchmark to see whether the 6.1%
colour-faithfulness figure moves.
