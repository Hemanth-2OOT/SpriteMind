# Open-source motion and articulation options for a 32×32 sprite rig

**Scope: research plus one measured experiment. No retraining, no checkpoint,
dataset or benchmark change, no new runtime dependency.**

Question: SpriteMind's limbs are attached to the body but do not articulate —
the knee and the elbow are rigid. What existing open-source technique or project
can fix that, and what does it cost to adopt?

## 1. What the renderer can and cannot express

`PixelScriptEngine` renders each `PART` by translating and rotating **its own**
pixels about **its own** anchor. A part's transform is independent of its
parent's. The rig (`backend/spritemind/pixelscript/rig.py`) therefore keeps parts
in contact by *translating* a child back onto its parent's socket; it cannot
rotate a child **about the parent's socket**.

A real two-bone solve needs exactly that composition:

```
child_world = parent_transform ∘ rotate(joint_angle) ∘ child_rest
```

So the limit is the transform model, not the solver. Any IK library dropped into
this renderer would compute correct joint angles and then be unable to apply
them, because applying them means rotating a child around a point that is not
its own anchor.

## 2. Candidate projects (licence-audited)

2D IK / skeletal solvers found by search, with the licence GitHub reports:

| Project | Stars | Licence | What it gives | Verdict for SpriteMind |
| --- | --- | --- | --- | --- |
| `FedUni/caliko` | 182 | MIT | FABRIK, 2D and 3D, bone chains | Correct maths, Java; would need a port, and still needs the transform model above |
| `yamahigashi/fabric-fabrik-fullbody-ik` | 128 | MIT | Full-body FABRIK rig demo | Same limit: assumes a hierarchical scene graph |
| `kattkieru/grim_IK` | 66 | MIT | FABRIK/CCD for game rigs | Same limit |
| `ekorudiawan/CCD-Inverse-Kinematics-2D` | 24 | MIT | Small CCD solver | Small enough to read, same limit |
| `gedeschaines/IK_Solver` | 22 | LGPL-3.0 | CCD/FABRIK | LGPL conflicts with the project's no-copyleft rule for runtime code |
| `NuclearVenom/Robot-Arm-Simulator-2D` | 3 | MIT | Two-bone demo, no library | Reference only |
| `Kryolyz/Damped_Least_Squares_IK`, `vionabb/linkage-simulator` | 3 each | none declared | — | Cannot be adopted without a licence |
| `Jasminestrone/RASW` | 3 | GPL-3.0 | — | Copyleft, rejected |

Adjacent sprite tooling, for completeness: `DragonBones/DragonBonesJS` (MIT) and
`EsotericSoftware/spine-runtimes` (licence not machine-readable) both define a
*skeletal animation format* with bone hierarchies — the thing whose absence is
the actual blocker. `Orama-Interactive/Pixelorama` (MIT) and
`LibreSprite/LibreSprite` (GPL-2.0) are editors, not solvers.

None of these is a drop-in for a 32×32, CPU-only, deterministic renderer, and
the MIT ones all assume a bone hierarchy the engine does not have.

## 3. What was adopted instead, and why

A dependency-free approximation, kept inside the existing rig:

1. **Two real segments per limb** (`arm_*` + `forearm_*`, `leg_*` + `shin_*`) in
   `backend/spritemind/dataset/generator.py`, so an elbow and a knee exist as
   parts with their own socket and their own pixels.
2. **Rigid follow plus a stepped lag** in `rig.apply_limb_flex`: the lower
   segment inherits the upper segment's whole transform, and on a slow swing
   (≤ `MAX_FLEX_SWING = 18°`) also steps one or two pixels sideways. At 2–3 px
   wide, a sideways step is what reads as a bent joint; rotating the segment
   instead produces diagonal holes and shatters the silhouette.
3. **Weapon parented to the hand** (`forearm_left`/`forearm_right`) so equipment
   follows the whole arm chain.
4. **Guarded application** in `coherence.build_frame_script`: a bent pose is kept
   only if it adds no island and changes at most `BEND_PIXEL_BUDGET = 6` pixels.

Measured (4 characters × 7 animations × 6 frames, seed 3, `pico8`):

- single-island frames: 168 / 168, island regressions: 0;
- weapon-to-hand gap: median 1 px, max 2 px;
- pixels changed versus the previous commit: 0 on `idle`, ≤ 20 on
  `melee_attack` / `dash` / `walk`, up to ~72 on the fastest `run` frames.

The run figures are the honest ceiling of the stepped approach: on a fast swing
the lag is skipped and the limb is rigid again.

## 4. What a real fix would take

Ranked by cost, with the benefit each one actually buys:

1. **Parent-composed rotation in the rig** (small, ~1 file). Make the resolved
   transform of a chain child compose through the parent's rotation, so a child
   can rotate *about the parent's socket*. This is what turns "stepped lag" into
   a real knee angle and also removes the run-cycle artefact. No new dependency.
2. **Two-bone analytic solve** (small, no dependency). With the composition from
   (1) in place, a closed-form two-bone solution for hip→knee→foot is a dozen
   lines and needs no FABRIK.
3. **Hierarchical transform in the renderer** (medium). Generalise
   `PART ... TRANSFORM` to accumulate parent transforms, which makes any imported
   IK library usable and the whole rig position-based rather than translation
   patched. Largest quality ceiling, largest blast radius (renderer, grammar,
   goldens).
4. **Adopt a skeletal format** (large). DragonBones/Spine-style data would let
   third-party animations in, but replaces the PixelScript pipeline — contrary to
   "make the existing system better, not rebuild it".

Recommendation: do (1) and then (2). They are dependency-free, testable against
the existing golden images, and they address the actual defect (transform
composition) rather than adding a solver the renderer cannot feed.
