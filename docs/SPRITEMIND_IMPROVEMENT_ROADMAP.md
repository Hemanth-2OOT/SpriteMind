# SpriteMind — Improvement Roadmap (research-derived)

**Status:** proposal only. Nothing here has been implemented. Requires explicit approval.

Derived from `docs/OPEN_SOURCE_RESEARCH.md` and `docs/TECHNOLOGY_COMPARISON.md`.
Every item states how success can be *measured*, because the standing rule is
"no claim without measurement".

---

## 0. Guiding constraint

The roadmap adds **no third-party runtime dependency**. Every item marked GREEN below is either
(a) a documented file format we write ourselves, or (b) an algorithm small enough to own
(≤ 300 lines). This is deliberate: the user's stated goal is "a focused, high-quality platform",
not a dependency graph.

---

## 1. Immediate correctness fixes (P0)

| # | Issue | Evidence | Fix | Files that would change | Success measure |
|---|---|---|---|---|---|
| P0-1 | Unknown animation names silently become `walk` | `temporal.py:302-303`; measured `attack`/`hurt` = `walk` 4/4 frames | Alias map (`attack→melee_attack`, `hurt→hit_reaction`, `block→blocking`, `idle/stand→idle`, `slash→melee_attack`, `shoot→ranged_attack`) + return `requested_animation` and `animation_aliased: true` in the response; unknown-but-unmappable names return a 400 rather than a walk | `ml/temporal.py`, `ml/temporal_evaluation.py`, `server/routers/animation.py`, `server/schemas.py` | 0 silent coercions across a 40-name table test; `attack` produces ≥1 distinct transformation signature from `walk` |
| P0-2 | Weapon/equipment drifts from the hand | `engine.py:151` absolute `TRANSFORM`; weapon offset hand-tuned at `temporal_dataset.py:117,130,151,187,202` | Declare a socket map (`weapon→arm_right`, `accessory→torso`) and resolve child transforms relative to the parent's post-motion pixel set; keep `TRANSFORM` syntax unchanged | `renderer/canvas.py`, `renderer/engine.py`, `ml/temporal.py`, new `pixelscript/rig.py` | Weapon-to-hand min-distance ≤ 1 px in **every** frame of 12 animations × 5 categories, measured by a new test |
| P0-3 | 1-px attachment tolerance is visible at 32×32 | `coherence.py` `MAX_ATTACH_DISTANCE = 1`; all frames measured 1 island but visually torn | Either tighten to 0 for connected parts or make the tolerance per-part (weapon 0, cape 1) | `pixelscript/coherence.py` | Island count still 1 **and** zero part-pairs separated by > 1 px; golden images unchanged for static sprites |
| P0-4 | Hardcoded `ident_conf = 0.90` in the procedural path | `ml/temporal.py` procedural branch | Report `null` unless a real confidence exists (matches the earlier frontend honesty fix) | `ml/temporal.py` | API returns `null` or a measured value; no constant 0.90 |

P0 items are small, reversible, and each has a test that fails before and passes after.

---

## 2. Low-risk improvements (P1)

| # | Improvement | Approach | Cost | Measure |
|---|---|---|---|---|
| P1-1 | C1-continuous motion | Interpolate offsets across frames in float, quantise once at render: removes the measured 7–8-part simultaneous jump | ~60 lines in `ml/temporal.py` | Mean per-part inter-frame delta ↓ ≥ 30% on walk/run/dash, with identical frame count and determinism preserved |
| P1-2 | Foot planting via two-bone IK | Analytic solve so the planted foot stays on the ground row while the hip bobs | ~100 lines, new `ml/ik.py` | Vertical foot drift ≤ 1 px across a walk cycle; silhouette IoU with the base pose stays ≥ current |
| P1-3 | Aseprite JSON export | New writer following the documented schema (`frames`, `spriteSourceSize`, `meta.frameTags`) | ~120 lines in `renderer/export.py` | File imports into Godot and Aseprite without error (manual check, recorded) |
| P1-4 | Godot `SpriteFrames` `.tres` export | Emit the resource text format directly | ~90 lines | Opens as `AnimatedSprite2D` frames in Godot 4 (manual check) |
| P1-5 | Attachment/rigidity metrics in `ml/metrics.py` | `weapon_hand_distance`, `part_parent_gap`, `max_interframe_part_delta` | ~80 lines + tests | New metrics appear in `benchmark.py` output; thresholds enforced in CI |

---

## 3. Medium-sized enhancements (P2)

| # | Improvement | Notes | Risk |
|---|---|---|---|
| P2-1 | Frame tags / animation events in the app | Tag frames (`attack_hit`, `cast_release`) so exports carry timing metadata engines can consume | Low; UI + schema |
| P2-2 | TexturePacker/LibGDX atlas schema option | Schema-only adoption (`free-tex-packer-core`, MIT) alongside our existing atlas | Low |
| P2-3 | Rig editor surface (read-only viewer) | Show part tree + sockets in the UI so users can *see* why a sprite is assembled as it is | Medium; UI work |
| P2-4 | Critic extension for "attachment" defects | New defect type alongside `UNEXPECTED HOLE`, reusing `repair_engine.py` patch machinery | Medium; must not regress the 90% critic gate |
| P2-5 | Palette remap on export | Let a user export the same sprite in another palette without regenerating | Low–medium |

---

## 4. Long-term research (P3)

| # | Direction | Why it is only P3 |
|---|---|---|
| P3-1 | Pixel VQ-VAE tokenizer (arXiv:2203.12130) | Promising for SVAN fidelity but requires retraining and a new checkpoint; must not disturb frozen splits. Gate: only if neural quality is provably below procedural after P0/P1 fixes. |
| P3-2 | LPIPS as a reranking term | Natural-image metric on pixel art; needs a human-ranking validation study before it may influence selection. |
| P3-3 | Joint frame generation (arXiv:2412.03685) | Architectural change; the current temporal model is adequate once the rig is hierarchical. |
| P3-4 | Auto-rigging from reference sheets (APES, arXiv:2206.02015) | Only relevant if we ingest external art, which raises licence questions (LPC is share-alike). |
| P3-5 | Neural quality ≥ procedural quality | The standing primary objective. Not solved by any external project; needs tokenizer/conditioning work on our own checkpoint. |

---

## 5. Deliberately postponed / not recommended (P4)

| Item | Why not now |
|---|---|
| Diffusion backend (SDXL LoRA) | Breaks determinism, grid alignment, palette conformance and CPU latency promises |
| Spine / DragonBones runtime as a dependency | We only need the semantics; the runtimes are JS/proprietary-adjacent and add nothing |
| GPL/AGPL code (LPC generator, spritebrew, LibreSprite) | Copyleft propagation to the whole product |
| IQA-PyTorch | Non-commercial licence |
| EbSynth-style propagation | No licence granted; wrong tool at 32×32 |
| scikit-image / OpenCV | Duplicates `ml/metrics.py` |
| Mocap retargeting | Cost/benefit is poor at this resolution |
| 3D part assembly pipeline (`csg`) | Would replace, not improve, a working renderer |

---

## 6. Suggested sequencing

```
Milestone A (correctness)      P0-1, P0-2, P0-4        + tests → no visual promise yet
Milestone B (motion quality)   P0-3, P1-1, P1-2, P1-5  + benchmark before/after
Milestone C (interop)          P1-3, P1-4, P2-2
Milestone D (product surface)  P2-1, P2-3, P2-5
Milestone E (research)         P3-1 only if D's benchmarks justify it
```

Each milestone: new branch, one commit per item, before/after benchmark evidence preserved,
no checkpoint or dataset changes, no PR merges.

---

## 7. What we should leave alone

Protect these deliberately — they are already the parts that work:

- `docs/PIXELSCRIPT_GRAMMAR.md` (frozen v2.1) and `pixelscript/parser.py` / `validator.py`.
- The deterministic CPU renderer's core (`canvas.py` pixel/layer primitives).
- `ml/metrics.py` successor metrics (silhouette IoU, components, holes, symmetry).
- The 90% critic gate and the rerank ablation switches (`use_colour`, `use_critic`).
- Frozen evaluation splits (`data/splits/heldout_v1.json`, digest `581bd3fa...`) and all existing
  benchmark records.
- The procedural baseline — it is currently the honest quality floor and must remain available as
  a comparison, not be removed.

---

## 8. Executive summary

**Strongest five opportunities**

1. Bone hierarchy + attachment sockets (DragonBones semantics, MIT, implemented by us) — fixes
   weapon drift and limb detachment structurally.
2. Alias-or-reject animation names — removes the silent `attack → walk` substitution.
3. Aseprite JSON export — one writer unlocks Godot, Unity, GameMaker and Phaser importers.
4. Two-bone IK + C1-continuous interpolation — removes foot sliding and frame popping.
5. Rigidity metrics in CI — converts "looks disconnected" into a number that can fail a build.

**Three highest practical value**

- P0-2 (socketed transforms) — the direct cause of the user's complaint.
- P0-1 (animation alias/reject) — an afternoon's work that removes an entire class of "why does
  attack look like walking".
- P1-3 (Aseprite JSON) — smallest change that materially improves the product for game devs.

**Best projects to study further**

DragonBones (semantics only), Pixelorama (MIT, formats/workflow), Universal-LPC generator
(specification only, GPL — read, never copy), `pixel-asset-gen` (MIT, procedural animation
profiles), and the APES paper if auto-rigging ever becomes relevant.

**Not worth adopting**

Diffusion LoRAs, Spine runtimes, EbSynth, LibreSprite/Aseprite source, `spritebrew`, IQA-PyTorch,
scikit-image/OpenCV, mocap retargeting.

**Recommended next milestone**

Milestone A: **P0-1 + P0-2 + P0-4**, each with a failing test first, plus the new
`weapon_hand_distance` metric so the effect is measurable. Expected: weapon stays in hand in every
frame of every animation, and no animation silently degrades to `walk`. No retraining, no
checkpoint change, no production promotion, no PR merge without approval.
