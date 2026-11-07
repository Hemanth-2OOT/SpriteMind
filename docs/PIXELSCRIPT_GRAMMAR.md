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
AURA    <x> <y> <w> <h> <color> [density]     # v2.1
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

### AURA

Paints a dithered halo in the transparent ring just outside the silhouette, tinted with the
prompt's element colour (`fire` -> ember orange, `ice` -> glacier blue, `lightning` -> violet,
`poison` -> acid green). The mask uses the same composite-based 4-neighbour `OUTLINE` ring, so
the halo hugs the whole character rather than each part, and it never writes inside the body.

`density` is `1`..`4`: `1` is a solid ring, `2` a checkerboard, `3` and `4` progressively
sparser sparks. It defaults to `2`. Like `SHADE` and `OUTLINE`, the region is
region-scoped: nothing outside `<x> <y> <w> <h>` changes.

### Metadata and conditioning

The leading `# META: colour=blue element=ice` comment line carries the prompt's colour and
element into the program. `CharacterGenerator` emits `AURA` from its finishing pass only when
an element was requested, using the element's palette tint for the halo; the parser, validator,
serializer and tokenizer all round-trip it. `AURA` costs 7 tokens
(`CMD_AURA NUM_x NUM_y NUM_w NUM_h COL_c NUM_density`).

Both colour and element also reach SVAN numerically: the semantic encoder takes extra
`colour`/`element` embedding slots (`0` = none, then alphabetical ids from
`dataset/conditioning.py::COLOUR_IDS`/`ELEMENT_IDS`), so the network can learn the tint from
the prompt instead of relying on the procedural fallback.

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
| v2.1 (14 primitives, AURA on elemental prompts) | 315 | 387 |

The decoder sequence budget is therefore 448 (see `SVAN.max_seq_len`), leaving headroom above
the observed max of 407.

## Compatibility

Adding primitives changes the vocabulary (401 → 403 for v2, 403 → 404 with `AURA`), so **an
older checkpoint cannot emit newer primitives**. `GenerationPipeline.load_checkpoint` compares the checkpoint's
`model_config.vocab_size` with the live tokenizer and refuses a mismatch with a clear
`checkpoint_error` instead of indexing out of bounds; callers fall back to the procedural
engine until a v2 checkpoint is trained.
