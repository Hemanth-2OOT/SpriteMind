# SpriteMind — Open-Source Ecosystem Research

**Status:** research only. No source code, checkpoint, dataset, test or dependency was changed.
**Date:** 2026-11-05
**Branch inspected:** `lemon/r13-candidate-diversity` @ `d837422`

---

## 0. Method and what was actually verified

Everything below marked **VERIFIED** was checked by running code or reading files in this
repository. Everything marked **UNVERIFIED** is a claim I did not personally confirm and must
not be treated as fact.

Verification performed:

| Action | Result |
|---|---|
| Read `README.md`, `docs/`, `docs/research/`, `docs/IMPLEMENTATION_AUDIT.md`, `docs/FINAL_RELEASE_READINESS.md` | done |
| Read `ml/temporal.py`, `ml/pipeline.py`, `dataset/temporal_dataset.py`, `pixelscript/coherence.py`, `renderer/engine.py`, `renderer/canvas.py`, `renderer/export.py`, `assets/pack.py`, `ml/metrics.py`, `ml/composition_model.py` | done |
| Ran a 6-frame animation audit across 7 animation names on a fixed wizard seed | done (results below) |
| Queried GitHub REST (`repos`, `license`, `search`) and HuggingFace Hub API for every licence claim | done |
| Queried arXiv API for paper existence and identifiers | done |
| Installed or executed any external project | **not done** — no dependency was added |
| Trained, retrained or modified any model | **not done** |

### 0.1 Verified defects found during this audit

These are measured, not assumed. They matter because they are precisely the
"disconnected / inconsistent / weapon drifting" complaints, and no library will fix them for us.

**D1 — Unknown animation names are silently coerced to `walk`.**
`backend/spritemind/ml/temporal.py:302-303`

```python
if anim_category not in ANIMATION_CATEGORIES:
    anim_category = "walk"
```

The canonical taxonomy (`dataset/temporal_dataset.py:27-47`) is `idle, walk, run, jump, fall,
death, melee_attack, ranged_attack, hit_reaction, taking_damage, blocking, dash, cast,
special_ability, ...`. The frontend dropdown and `actions/catalog.py` use the canonical names, so
the UI is correct — but the HTTP API accepts any string, and natural words (`attack`, `hurt`,
`block`) fall through. Measured: `attack` and `hurt` produced **4/4 byte-identical frames to
`walk`**, with no warning in the response.

**D2 — Part transforms are absolute, so equipment cannot be rigidly attached to a hand.**
`backend/spritemind/renderer/engine.py:151-157` → `backend/spritemind/renderer/canvas.py:321`

`TRANSFORM part dx dy rot` translates the part's own pixel set by an absolute delta. There is no
parent, no child, and no socket. `compute_motion_offsets` therefore has to *guess* the weapon
offset independently (`temporal_dataset.py:117`, `:130`, `:151`, `:187`, `:202`...), and the guess
is a hand-tuned approximation of where the hand ended up. This is the mechanical cause of
weapon drift.

**D3 — The connection guard tolerates 1px of separation, which is visible at 32×32.**
`backend/spritemind/pixelscript/coherence.py` (`MAX_ATTACH_DISTANCE = 1`). Islands per frame
measured at 1 for every animation tested, so the silhouette is technically connected — but a gap
of one pixel on a 32px sprite is ~3% of the body width and reads as a detached shoulder or a
floating staff.

Measured baseline (`wizard`, staff + cape, seed 3, 6 frames, `procedural_baseline`):

| animation | islands/frame | pixels range | identical to `walk` |
|---|---|---|---|
| walk | 1 | 314–330 | — |
| attack | 1 | 314–330 | **4/4** |
| hurt | 1 | 314–330 | **4/4** |
| run | 1 | 302–343 | 0/4 |
| cast | 1 | 311–337 | 0/4 |
| dash | 1 | 309–325 | 0/4 |
| death | 1 | — | 0/4 |

---

## 1. Existing SpriteMind capabilities (baseline for comparison)

Do not re-buy what already exists. As implemented today:

| Capability | Where | Honest assessment |
|---|---|---|
| Grammar-constrained script language | `pixelscript/parser.py`, `validator.py`, `engine.py`, `docs/PIXELSCRIPT_GRAMMAR.md` | Genuinely good; a real differentiator. Keep. |
| Deterministic 32×32 CPU renderer | `renderer/canvas.py` | Fast, dependency-free, deterministic. Keep. |
| Neural generator SVAN (~2.3M params) | `ml/svan.py`, `ml/tokenizer.py` | Works, but currently loses to the procedural baseline on quality. |
| Visual critic | `critic/visual_understanding.py`, `repair_engine.py` | Useful defect taxonomy; rule-based. |
| Temporal model | `ml/temporal.py`, `dataset/temporal_dataset.py` | Motion tables are decent; hierarchy (D2) is missing. |
| Socket predictor | `ml/composition_model.py` | Exists and is trained; not used to drive animation attachment. |
| Silhouette/topology metrics | `ml/metrics.py` (`silhouette_iou`, `connected_component_count`, `largest_component_share`, `interior_holes`, `column_symmetry`) | Already covers most of what `scikit-image` would be added for. |
| Sheet export + atlas | `renderer/export.py`, `assets/pack.py` (`manifest.json`, `atlas/texture_atlas.json`) | Custom format; no engine-native formats. |
| Palette conditioning, coverage rerank | `ml/pipeline.py`, R5/R6/R12/R13 | Real measured gains; leave alone. |

**Conclusion up front:** SpriteMind's gaps are *not* missing libraries. They are a missing
transform hierarchy, a silent enum coercion, and a small attachment tolerance. The ecosystem
research below is therefore weighted towards (a) rigging semantics we can implement in a few
hundred lines and (b) formats that buy engine compatibility. It is deliberately *not* a
recommendation to bolt on a diffusion stack.

---

## 2. Candidates by category

Verdict legend: **GREEN** = can be incorporated subject to its licence · **YELLOW** = reference
or further verification needed · **RED** = do not incorporate.

### A. Pixel-art generation

**A1. `LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator`** — YELLOW (code GPL-3.0)
https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator
1.9k stars, TypeScript, actively maintained (pushed 2026-10-28). Licence verified via GitHub API:
**GPL-3.0** for code. The artwork is separately licensed per-asset (CC-BY-SA 3.0 / GPL / some
OGA-BY), which is a *different* licence from the code — do not conflate them.
*What is valuable:* it is the reference implementation of layered, slot-based character
composition with a canonical body anchor and per-animation part offsets. Its part taxonomy
(hair, torso, legs, feet, weapon, shield...) and its rule that a part is drawn in a fixed z-order
relative to the body is exactly the discipline SpriteMind's rig lacks.
*What is not:* their code is GDScript/TypeScript under GPL; copying it would force SpriteMind's
derivative to be GPL. **Read as specification only.** Do not vendor, do not import, do not copy
source lines.

**A2. `ilomon10/csg`** — GREEN (MIT)
https://github.com/ilomon10/csg — 0 stars, TypeScript/Three.js, last push 2026-10-15.
Modular 3D parts → pixel sprite sheets, in-browser. Small and unproven, but MIT. Value is the
*data model*: named part slots with attachment transforms. Low practical value beyond reference;
do not depend on it (single author, no tests visible, 0 stars ≠ recommendation).

**A3. `MozeeB/pixel-asset-gen`** — GREEN (MIT)
https://github.com/MozeeB/pixel-asset-gen — 5 stars, Python, last push 2026-04-02.
136 procedurally generated sprites with animations, JSON metadata, atlases, GIF export, "all
generated from math". Closest in spirit to SpriteMind's procedural path. MIT means its motion
profiles and atlas JSON schema can be read and re-implemented freely. Value: comparison of
animation profiles and atlas metadata shape.

**A4. `nerijs/pixel-art-xl`** (HuggingFace) — YELLOW (CreativeML OpenRAIL-M)
https://huggingface.co/nerijs/pixel-art-xl — 37k downloads, 668 likes, SDXL LoRA.
Licence verified via Hub API: `creativeml-openrail-m`. OpenRAIL-M permits commercial use but
carries **use-based restrictions** that must be passed on to downstream users, and it is a
*weights* licence rather than a code licence. Technically: requires an SDXL base (~6.9 GB) and a
GPU; on this CPU-only workspace it is not runnable at usable latency. It also cannot guarantee
32×32 grid alignment or palette conformance, which are SpriteMind's actual product promises.

**A5. `Onodofthenorth/SD_PixelArt_SpriteSheet_Generator`** (HuggingFace) — YELLOW (Apache-2.0)
https://huggingface.co/Onodofthenorth/SD_PixelArt_SpriteSheet_Generator — 7.4k downloads, 466
likes. Licence `apache-2.0` per Hub metadata (the *weights*; the underlying base model's terms
still apply). Outputs a *sheet* rather than an aligned, per-frame, palette-safe sprite; frames
are not guaranteed consistent. Interesting as prior art for "generate the whole sheet at once",
not as a component.

**A6. `GAlbanese09/spritebrew`** — RED (AGPL-3.0)
https://github.com/GAlbanese09/spritebrew — 74 stars, TypeScript/Next.js. Closest commercial
analogue to SpriteMind (upload → animate → Unity/Godot export, uses the Retro Diffusion API).
AGPL-3.0 plus a paid third-party API dependency. **Do not copy or link.** Reading its UI/UX
decisions for product ideas is fine; that is not copyright infringement, but no code may enter
our tree.

**A7. Papers (no code to license):**

| Paper | ID | Relevance |
|---|---|---|
| *Pixel VQ-VAEs for Improved Pixel Art Representation* | arXiv:2203.12130 (2022-03-23) | Directly relevant to `ml/tokenizer.py`: argues pixel-art-specific codebooks beat generic VQ-VAE for reconstruction fidelity. A research direction for SVAN's tokenizer, not a drop-in. |
| *Generating Pixel Art Character Sprites using GANs* | arXiv:2208.06413 (2022-08-15) | Prior art for the GAN route we are not taking. Useful for benchmark comparison claims. |
| *Sprite Sheet Diffusion* | arXiv:2412.03685 (2024-12-04) | Frame consistency via sheet-level generation. Reference for the "generate all frames jointly" idea. |
| *APES: Articulated Part Extraction from Sprite Sheets* | arXiv:2206.02015 (2022-06-04) | **Most relevant paper here**: recovering articulated part structure from an existing sheet. If we ever want to auto-derive rigs from reference art, this is the method. |

### B. Character rigging and animation

**B1. `DragonBones/DragonBonesJS`** — GREEN (MIT)
https://github.com/DragonBones/DragonBonesJS — 860 stars, MIT verified via API, last push
2026-01-23. Runtime for 2D skeletal animation with a bone hierarchy, slots, and **attachment
sockets**. This is the single most useful find: the semantics we need (child transform inherits
parent delta; an attachment is bound to a bone with an offset) are small, well-understood, and
MIT-licensed. We should implement the *semantics* (~200 lines in our own renderer), not vendor a
JavaScript runtime.
*Disadvantages:* it is JS for Flash-era formats; irrelevant as a library to us. Value is purely
conceptual + licence-safe.

**B2. `EsotericSoftware/spine-runtimes`** — RED (proprietary "Spine Runtimes License")
https://github.com/EsotericSoftware/spine-runtimes — 5.3k stars, GitHub reports `NOASSERTION`.
The Spine Runtimes License is not an open-source licence: it restricts use to developers who own
a Spine editor licence, and redistribution terms are conditional. Even reading it as a design
reference is fine, but **no code may be copied and no dependency introduced**. Marked RED.

**B3. `LibreSprite/LibreSprite`** — YELLOW (GPL-2.0)
https://github.com/LibreSprite/LibreSprite — 8.75k stars, C++, last push 2026-10-30. GPL-2.0 fork
of Aseprite. Value: reference for frame/tag/onion-skin data models and for how a paint program
stores per-frame layers. GPL → no code reuse in a closed product. Reference only.

**B4. `Orama-Interactive/Pixelorama`** — GREEN (MIT)
https://github.com/Orama-Interactive/Pixelorama — 11k stars, GDScript, MIT verified, last push
2026-11-07. Aseprite-compatible import/export, frame tags, onion skinning, palette management.
MIT, so its *formats and behaviours* can be mirrored without obligation. Highest practical value
of the editor projects, because it documents what a game dev expects from an exported sheet.

**B5. `godotengine/godot`** — GREEN (MIT), as a format reference only
https://github.com/godotengine/godot — 119.8k stars, MIT. Relevant only for the `SpriteFrames`
`.tres` resource schema and Godot's `AnimatedSprite2D` expectations. Writing a `.tres` exporter
ourselves is trivial and licence-clean; we would not embed any Godot code.

**B6. `jamriska/ebsynth`** — RED (no licence declared)
https://github.com/jamriska/ebsynth — 1.7k stars, GitHub reports **no licence**. Absent an
explicit licence, default copyright applies: source is viewable, not reusable. EbSynth's
keyframe-propagation idea (stylise one frame, propagate to the rest with optical-flow-guided
synthesis) is the classic answer to frame-to-frame consistency, but for us it would require
optical flow on a 32×32 sprite — almost certainly worse than solving the rig hierarchy properly.

### C. Editing, export and production tools

**C1. `odrick/free-tex-packer-core`** — GREEN (MIT)
https://github.com/odrick/free-tex-packer-core — 151 stars, MIT, last push 2026-09-11. Emits
TexturePacker-compatible JSON (hash + array) and LibGDX atlases. Value: a known atlas schema so
our `assets/pack.py` output can be consumed by existing engine importers. We already emit our own
`atlas/texture_atlas.json`; matching a de-facto schema is the cheap win.

**C2. `libgdx/libgdx`** — GREEN (Apache-2.0), format reference
https://github.com/libgdx/libgdx — 25.5k stars, Apache-2.0. Its `TexturePacker` atlas format is
widely consumed. Reference for schema only.

**C3. Aseprite JSON export schema** — no licence needed (documented file format)
The de-facto sprite-sheet metadata format for Godot, Unity and GameMaker importers: `frames`
array with `frame`, `spriteSourceSize`, `sourceSize`, plus `meta.frameTags`. Implementing a
writer for a documented format is not a licensing act. High practical value for "game-ready".

**C4. `aseprite/aseprite`** — RED (EULA, no open-source licence)
https://github.com/aseprite/aseprite — 40.8k stars, `license: none`. Source-available under a
proprietary EULA; compiling it for distribution requires a paid licence. Never a dependency.

### D. Visual quality and critic systems

**D1. `richzhang/PerceptualSimilarity` (LPIPS)** — GREEN (BSD-2-Clause)
https://github.com/richzhang/PerceptualSimilarity — 4.3k stars, BSD-2 verified. Learns perceptual
distance; a defensible way to say "candidate B looks more like the prompt than candidate A".
Cost: torch (already a dependency) + ~few MB of AlexNet/VGG weights; CPU-usable at 32×32 but adds
latency to best-of-N. **Caveat:** LPIPS was trained on natural images and is known to be a poor
judge of pixel art; its use must be validated against human ranking before it enters the rerank
score.

**D2. `chaofengc/IQA-PyTorch`** — RED (PolyForm Noncommercial 1.0.0)
https://github.com/chaofengc/IQA-PyTorch — 3.4k stars. Licence verified by reading `LICENSE`:
**PolyForm Noncommercial 1.0.0**. This forbids commercial use. A popular library with a
non-commercial licence is exactly the trap this audit exists to catch. Do not use.

**D3. `scikit-image/scikit-image`** — GREEN (BSD-3-Clause)
https://github.com/scikit-image/scikit-image — 6.6k stars. Licence verified by reading
`LICENSE.txt` (GitHub metadata says NOASSERTION; the file says BSD-3-Clause). Offers
`measure.label`, `morphology.skeletonize`, `measure.regionprops`, `filters`. **But** `ml/metrics.py`
already implements connected components, holes, bbox and symmetry by hand. Value today is
marginal; the honest recommendation is "do not add it for what we already have; consider it only
if we need skeletonisation or medial-axis work".

**D4. `opencv/opencv`** — GREEN (Apache-2.0)
https://github.com/opencv/opencv — 91.5k stars. `connectedComponentsWithStats`, `findContours`,
`matchShapes`. Same verdict as D3: we already have the primitives, and OpenCV is a heavyweight
dependency (~60 MB) for a 32×32 raster.

**D5. SpriteMind's own critic** — already good enough for defect classification
`critic/visual_understanding.py` + `repair_engine.py` already detect holes, islands and
proportion faults and propose `RECT` repairs. Nothing in the ecosystem offers a better
*pixel-art-specific* defect taxonomy at this size. Keep it; extend it (see roadmap).

### E. Datasets and training resources

**E1. Universal LPC spritesheet art** — YELLOW (per-asset: CC-BY-SA 3.0 / GPL / OGA-BY)
The art repository behind A1. Attractively large and animation-complete (walkcycles, attacks,
casting in 4 directions), but **share-alike**: training on it is one question, redistributing
derived weights is another and is legally murky. It is also 64×64, not 32×32. Recommend: use as a
*reference* for pose timing, not as training data without a legal review.

**E2. Kenney asset packs** — licence must be re-verified per pack
Widely described as CC0, but packs differ and I did **not** verify each one here. Marked
**UNVERIFIED**; if we want CC0 sprite data, verify pack-by-pack at download time and record the
licence file alongside the data.

**E3. Existing synthetic corpus** — already ours
`data/datasets/cond_v1_15k`, `generalization_10k`, `data/splits/heldout_v1.json` (digest
`581bd3fa...`). Any new external data must not disturb these frozen splits.

**E4. Motion capture** — not recommended
CMU/AMASS-style 3D mocap would need retargeting to a 32×32 four-way sprite, with no clear gain
over 8 hand-authored keyframe profiles per animation.

### F. Other relevant technology

**F1. Two-bone analytic IK (no library)** — GREEN (own implementation)
For foot planting during walk/run: solve the knee/hip angle analytically so the foot stays on the
ground line while the hip bobs. ~40 lines of trigonometry. Fixes the "feet slide / legs detach"
class of artefact far better than tuning motion tables by hand.

**F2. Velocity-continuous interpolation (Catmull-Rom / critically damped spring)** — GREEN (own)
Motion tables are sampled at fixed phases; sharp `int(round(...))` quantisation makes offsets pop
between frames (`parts_moved_vs_prev` measured as 7–8 of 8 parts changing at once in `walk` and
`dash`). Interpolating positions across frames and only then quantising reduces that popping.
No dependency.

**F3. Aseprite/Godot/Unity importers (third-party, MIT)** — YELLOW
Various community importers exist for each engine; they consume the Aseprite JSON schema (C3).
We do not need them as dependencies — we need to emit the schema they read.

---

## 3. Licence compliance matrix

| ID | Project | Code licence | Assets | Weights | Commercial | Redistribute | Verdict |
|---|---|---|---|---|---|---|---|
| A1 | Universal LPC Generator | GPL-3.0 | CC-BY-SA 3.0 / GPL / OGA-BY per file | n/a | GPL obligations | copyleft propagates | YELLOW |
| A2 | `ilomon10/csg` | MIT | in-repo | n/a | yes | yes + notice | GREEN |
| A3 | `pixel-asset-gen` | MIT | in-repo | n/a | yes | yes + notice | GREEN |
| A4 | `pixel-art-xl` | — | — | CreativeML OpenRAIL-M | yes, with use restrictions | restrictions pass downstream | YELLOW |
| A5 | `SD_PixelArt_SpriteSheet_Generator` | — | — | Apache-2.0 (base model terms also apply) | verify | verify | YELLOW |
| A6 | `spritebrew` | AGPL-3.0 | in-repo | n/a | AGPL + network clause | copyleft + network | RED |
| B1 | DragonBonesJS | MIT | examples mixed | n/a | yes | yes + notice | GREEN |
| B2 | spine-runtimes | Spine Runtimes Licence (proprietary) | proprietary | n/a | requires editor licence | conditional | RED |
| B3 | LibreSprite | GPL-2.0 | GPL-2.0 | n/a | GPL | copyleft | YELLOW |
| B4 | Pixelorama | MIT | MIT | n/a | yes | yes + notice | GREEN |
| B5 | Godot | MIT | MIT | n/a | yes | yes + notice | GREEN |
| B6 | EbSynth | none declared | none | n/a | **no rights granted** | no | RED |
| C1 | free-tex-packer-core | MIT | MIT | n/a | yes | yes + notice | GREEN |
| C2 | libgdx | Apache-2.0 | Apache-2.0 | n/a | yes | yes + notice + patent grant | GREEN |
| C3 | Aseprite JSON schema | n/a (documented format) | n/a | n/a | yes | yes | GREEN |
| C4 | Aseprite | proprietary EULA | proprietary | n/a | paid licence required | restricted | RED |
| D1 | LPIPS (PerceptualSimilarity) | BSD-2-Clause | BSD-2 | model weights same repo | yes | yes + notice | GREEN |
| D2 | IQA-PyTorch | **PolyForm Noncommercial 1.0.0** | — | — | **no** | no | RED |
| D3 | scikit-image | BSD-3-Clause | BSD-3 | n/a | yes | yes + notice | GREEN |
| D4 | OpenCV | Apache-2.0 | Apache-2.0 | n/a | yes | yes + notice | GREEN |
| E1 | LPC art | — | CC-BY-SA 3.0 / GPL / OGA-BY | — | attribution + share-alike | share-alike | YELLOW |
| E2 | Kenney packs | — | described CC0, **per-pack unverified** | — | verify per pack | verify per pack | YELLOW |

**Never** assume a repository's code licence covers its assets, datasets or weights. A1 and E1
are the clearest examples of three different licences in one project.

---

## 4. What this research says we should *not* do

1. **Do not adopt a diffusion pipeline.** It cannot satisfy determinism, 32×32 grid alignment,
   palette conformance or CPU latency — four of SpriteMind's stated product promises.
2. **Do not switch the critic to a general-purpose IQA model.** LPIPS is a natural-image metric
   and IQA-PyTorch is non-commercial; our rule critic is more appropriate at this scale.
3. **Do not add scikit-image or OpenCV** to replace metric code we already have.
4. **Do not copy GPL/AGPL/proprietary code** (A1, A6, B2, B3, C4, B6) under any framing.
5. **Do not treat star count as a proxy for fitness**; A2 has 0 stars and is more relevant than
   `public-apis` with 497k.

---

## 5. Summary of the strongest opportunities

| Rank | Opportunity | Source of the idea | Why it matters |
|---|---|---|---|
| 1 | Bone hierarchy + attachment sockets (implement ourselves) | DragonBones (MIT), LPC anchors | Fixes weapon drift and limb detachment at the *cause*, not by tuning numbers. |
| 2 | Reject or alias unknown animation names | LPC / Pixelorama taxonomies | Removes silent `attack → walk` substitution. |
| 3 | Aseprite-JSON + Godot `SpriteFrames` export | C3, B5, Pixelorama (MIT) | Makes output directly consumable by the engines our users ship in. |
| 4 | Two-bone IK + C1-continuous interpolation | own implementation | Removes foot sliding and frame popping. |
| 5 | Rigid-body silhouette/attachment metrics in CI | own `ml/metrics.py` | Turns "looks disconnected" into a number that can fail a build. |

See `docs/TECHNOLOGY_COMPARISON.md` for the side-by-side evaluation, and
`docs/SPRITEMIND_IMPROVEMENT_ROADMAP.md` for priorities.
