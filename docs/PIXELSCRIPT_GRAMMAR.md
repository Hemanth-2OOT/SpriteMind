# PixelScript Grammar (v2)

PixelScript is the text language SpriteMind models emit. Every sprite is a short program of
one-line drawing commands, so a 32×32 character is usually 250–400 tokens instead of 1024
pixel tokens. Grammar-constrained decoding keeps the transformer inside this language, which
is why 100% of generated programs parse and validate.

## Commands

```
LAYER   <index> <name> <opacity>
PART    <name> <socket> <x> <y>
RECT    <x> <y> <w> <h> <color> <filled>
LINE    <x0> <y0> <x1> <y1> <color>
PIXEL   <x> <y> <color>
FILL    <x> <y> <color>
ERASE   <x> <y> <w> <h>
REUSE   <part_name> <x> <y>
TRANSFORM <part_name> <dx> <dy> <rot_deg>
SHADE   <x> <y> <w> <h> <color> <direction>   # v2
OUTLINE <x> <y> <w> <h> <color>               # v2
END PART | END PROGRAM
```

### SHADE

Darkens (or with a light color, lightens) one edge of an existing opaque region. `direction`
is `0` = top, `1` = bottom, `2` = left, `3` = right, and the parser folds any other integer
back into `0..3`. Shading only touches pixels that are already opaque, so a shade never adds
silhouette.

### OUTLINE

Adds a 1px border in the transparent ring that touches opaque pixels inside the region. The
mask comes from the **composite** canvas, not the active layer, so a multi-layer character
(torso + arms + weapon on separate layers) gets a single connected silhouette border rather
than a border around each part.

Both primitives are region-scoped: nothing outside `<x> <y> <w> <h>` changes.

## Finishing pass

`CharacterGenerator` closes every generated program with a fixed finishing pass:

```
LAYER 0 base 1.00        # back to the base layer
SHADE 0 0 <w> <h> <rim> 1
OUTLINE 0 0 <w> <h> <outline>
END PROGRAM
```

That ordering matters. Animation code (`pixelscript/coherence.py::build_frame_script`) injects
part `TRANSFORM` lines **before** the `SHADE`/`OUTLINE` pair, so every frame's shading and
outline are recomputed against the moved parts instead of being carried over from a static
sprite. Before that change, idle frames lost their outline consistency whenever a limb moved.

## Cost

| Program | Tokens (p50) | Tokens (p95) |
| --- | --- | --- |
| v1 (11 primitives) | 292 | 364 |
| v2 (13 primitives, with finishing pass) | 308 | 380 |

The decoder sequence budget is therefore 448 (see `SVAN.max_seq_len`), leaving headroom above
the observed max of 407.

## Compatibility

Adding primitives changes the vocabulary (401 → 403 tokens), so **a checkpoint trained on v1
cannot emit SHADE/OUTLINE**. `GenerationPipeline.load_checkpoint` compares the checkpoint's
`model_config.vocab_size` with the live tokenizer and refuses a mismatch with a clear
`checkpoint_error` instead of indexing out of bounds; callers fall back to the procedural
engine until a v2 checkpoint is trained.
