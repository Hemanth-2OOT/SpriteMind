# SpriteMind — Technology Comparison

**Status:** research only, nothing implemented.
Companion to `docs/OPEN_SOURCE_RESEARCH.md`.

Each section compares **what SpriteMind does today** against the strongest alternatives found.
The recommendation column is the only column that matters for a decision.

---

## 1. Animation transform model ← the decisive comparison

| | **A. Current: absolute per-part offsets** | **B. DragonBones-style bone hierarchy** (MIT) | **C. LPC layered slots** (GPL, reference only) | **D. Fully skeletal runtime (Spine)** (proprietary) |
|---|---|---|---|---|
| Where it lives | `renderer/engine.py:151`, `canvas.py:321` | would be new renderer state | `Universal-LPC-Spritesheet-Character-Generator` | `spine-runtimes` |
| How a weapon follows the hand | hand-tuned independent offset (`temporal_dataset.py:117`) | weapon is a child of `arm_right` with a socket offset; inherits parent delta | drawn last, positioned per frame from a fixed table | bone attachment |
| Weapon drift possible? | **yes, by construction** | no | no | no |
| Limb detachment possible? | yes, mitigated by `coherence.py` | no | no (pre-drawn frames) | no |
| Rotation composition | each part rotates around its own anchor | child angle = parent angle + local angle | n/a (no runtime rotation) | full |
| Implementation cost | 0 (existing) | **~200–300 lines** in our own renderer + motion table migration | 0 usable (GPL) | large, and licence-blocked |
| Licence risk | none | none (we write it) | GPL-3.0 | proprietary |
| Risks | the current bugs D2/D3 | migration of every `TRANSFORM` emission site; determinism must be re-proven | none usable | unacceptable |
| Verdict | keep as the wire format, **add hierarchy beneath it** | **RECOMMENDED** | reference only | reject |

**Key insight:** we do not have to change PixelScript. `TRANSFORM` can keep its syntax while
gaining optional parent semantics (`TRANSFORM weapon ... parent=arm_right`), or the renderer can
resolve a declared socket map before executing transforms. Either way the on-disk format and the
grammar version stay compatible — which matters because `docs/PIXELSCRIPT_GRAMMAR.md` is a frozen
spec and 30+ tests depend on it.

---

## 2. Motion generation: tables vs IK vs learned

| | **Current** | **Two-bone analytic IK + C1 interpolation** | **TemporalSVAN priors** (existing) | **Learned 3D mocap retarget** |
|---|---|---|---|---|
| Mechanism | keyframed offset tables per animation (`temporal_dataset.py:87-380`) | solve hip/knee so the foot lands on a fixed ground line; interpolate before quantising | neural prior blended with the table | CMU/AMASS retarget |
| Foot sliding | present (no ground constraint) | removed analytically | inherited from tables | depends on retarget |
| Frame popping | measured: 7–8 of 8 parts change between adjacent `walk`/`dash` frames | reduced by interpolation + quantise-after | no effect | n/a |
| Determinism | perfect | perfect (closed form) | depends on sampler | n/a |
| Cost | 0 | ~80–120 lines, no dependency | 0 | months |
| Verdict | keep tables as the *style* source | **RECOMMENDED as a post-process** | keep | reject (P3 at best) |

---

## 3. Generation approach

| | **Current: SVAN + grammar-constrained decoding** | **Diffusion LoRA** (`pixel-art-xl`, OpenRAIL-M) | **Sprite-sheet diffusion** (arXiv:2412.03685) | **VQ-VAE pixel tokenizer** (arXiv:2203.12130) |
|---|---|---|---|---|
| Output guarantee | valid PixelScript, renderable deterministically | none | none | valid tokens |
| 32×32 grid alignment | exact | not guaranteed | by construction of the sheet | exact |
| Palette conformance | enforced | not enforced | not enforced | improves |
| CPU feasibility | ~0.5 s best-of-4 (documented) | GPU required (~7 GB weights) | GPU required | CPU-trainable at our scale |
| Determinism | strong | seed-dependent, not byte-stable across hardware | same | strong |
| Licence | ours | OpenRAIL-M use restrictions | paper only | paper only |
| Practical value to us | ships today | low (breaks product promises) | low | **medium, research-grade** |
| Verdict | keep | reject | reject for now | **P3 experiment**: tokenizer fidelity is a legitimate way to raise SVAN quality without changing architecture |

Honest note: the user's core objective "neural quality ≥ procedural quality" is not solved by any
external project. It is solved by tokenizer/conditioning work we already have a roadmap for.

---

## 4. Quality evaluation and reranking

| | **Current rule critic + coverage rerank** | **LPIPS** (BSD-2) | **IQA-PyTorch** (PolyForm Noncommercial) | **scikit-image / OpenCV** (BSD-3 / Apache-2.0) |
|---|---|---|---|---|
| What it measures | holes, islands, proportions, palette leaks, coverage | learned perceptual distance | 30+ full-reference metrics | morphology, components, contours |
| Pixel-art appropriate | yes (domain-specific) | **doubtful** — trained on natural images | doubtful | n/a (primitives) |
| CPU cost on 32×32 | ~ms | ~10–40 ms/candidate + model weights | higher | <1 ms |
| Licence | ours | BSD-2 | **non-commercial → unusable** | permissive |
| What we already have | `ml/metrics.py` has IoU, components, holes, symmetry | — | — | mostly duplicative |
| Verdict | keep, extend | **conditional P3** (must be validated against human ranking first) | **reject** | do not add |

---

## 5. Export and engine compatibility

| Format | Who consumes it | Effort to emit | Value |
|---|---|---|---|
| Current `atlas/texture_atlas.json` + `manifest.json` | only SpriteMind | 0 | low outside our app |
| **Aseprite JSON** (documented schema) | Godot, Unity, GameMaker, Phaser importers | ~80–120 lines | **high** |
| **Godot `SpriteFrames` `.tres`** | Godot 4 directly, no importer needed | ~60–90 lines | **high** for Godot users |
| **TexturePacker / LibGDX atlas JSON** | libGDX, Phaser, many engines | ~60 lines (`free-tex-packer-core` schema, MIT) | medium |
| Unity `.meta` sprite slicing | Unity | ~80 lines, Unity-specific Y-flip pitfalls | medium |
| `.aseprite` binary | — | high (complex format) | low — nobody needs us to *write* it |

Verdict: emit **Aseprite JSON + Godot `.tres`** first. Both are documented formats; implementing a
writer is not a licensing act and needs no dependency.

---

## 6. Editor / workflow inspiration

| | **Current SpriteMind UI** | **Pixelorama** (MIT) | **LibreSprite/Aseprite** (GPL/proprietary) |
|---|---|---|---|
| Onion skin | exists (`compose_onion_skin`) | full timeline | full timeline |
| Frame tags | no | yes | yes |
| Palette management | palettes are fixed sets | full palette editor + remap | full |
| Usable code | n/a | MIT → could mirror behaviours | no |
| Verdict | add **frame tags** only if users ask | reference | reject |

---

## 7. Datasets

| | **Current** | **LPC art** (CC-BY-SA/GPL) | **Kenney** (per-pack, unverified) | **Synthetic (ours)** |
|---|---|---|---|---|
| Size | 15k + 10k conditioned rows | thousands of frames, 64×64, 4 directions | varies | unlimited |
| Resolution match | 32×32 | 64×64 | mixed | 32×32 |
| Licence clarity | clear (ours) | share-alike, mixed per file | per-pack | clear |
| Retraining required to use | — | yes, and risks the frozen splits | yes | no |
| Verdict | keep | reference for timing only | verify before any use | keep |

---

## 8. Scorecard of the five strongest candidates

| Candidate | Practical value | Licence safety | Integration cost | CPU cost | Recommendation |
|---|---|---|---|---|---|
| DragonBones bone/socket semantics | **very high** | high (MIT, and we implement ourselves) | medium (~300 lines + migration) | none | **Adopt the idea** |
| Aseprite JSON export | high | high (documented format) | low | none | **Adopt** |
| Godot `SpriteFrames` export | high (for Godot users) | high | low | none | **Adopt** |
| Two-bone IK + C1 interpolation | high | high (own code) | low–medium | negligible | **Adopt** |
| Pixel VQ-VAE tokenizer research | medium | high (paper) | high (retrain) | low | **Experiment later** |

---

## 9. Explicit rejections with reasons

| Rejected | Reason |
|---|---|
| Diffusion LoRAs (A4, A5) | GPU-bound, non-deterministic, no grid/palette guarantee |
| `spritebrew` (A6) | AGPL-3.0 network copyleft |
| Spine runtimes (B2) | proprietary licence, requires editor licence |
| EbSynth (B6) | no licence granted at all |
| Aseprite source (C4) | proprietary EULA |
| IQA-PyTorch (D2) | PolyForm **Noncommercial** — incompatible with a product |
| scikit-image / OpenCV as new deps | duplicate `ml/metrics.py`; heavy for 32×32 |
| LPC generator code (A1) | GPL-3.0 would propagate to our whole product |
| Mocap retargeting | no measurable benefit at 32×32, huge cost |
