# SpriteMind — Integration Recommendations

**Status:** research only. No file below has been modified. This document describes *where* a
future change would go and *what could break*, so the work can be scoped before approval.

---

## 1. Principle

Adopt **semantics and formats**, not codebases.

| Adopt | Do not adopt |
|---|---|
| DragonBones transform semantics (MIT) | DragonBonesJS as a dependency |
| Aseprite JSON schema (documented format) | Aseprite source (EULA) |
| Godot `SpriteFrames` text format (documented) | Godot as a dependency |
| TexturePacker/LibGDX atlas schema (documented) | libGDX as a dependency |
| Two-bone IK math (textbook) | any IK library |
| LPC part taxonomy (specification) | LPC generator code (GPL-3.0) |
| APES / Pixel VQ-VAE papers (research) | their authors' unreleased code |

---

## 2. Recommendation R1 — Socketed, parent-relative part transforms

**Maps to:** `backend/spritemind/renderer/canvas.py`, `backend/spritemind/renderer/engine.py`,
`backend/spritemind/ml/temporal.py`, plus a new `backend/spritemind/pixelscript/rig.py`.

### What would be added

```python
# pixelscript/rig.py — illustrative only, not implemented
PART_PARENT = {
    "head": "torso", "arm_left": "torso", "arm_right": "torso",
    "leg_left": "torso", "leg_right": "torso",
    "weapon": "arm_right",            # socket
    "accessory": "torso",
}
SOCKET_OFFSET = {"weapon": (1, 1)}     # px, in the parent's local frame
```

Resolution order per frame: `torso` first, then children with
`child_delta = parent_delta + rotate(local_offset, parent_rot)`.

### Compatibility

- `TRANSFORM part dx dy [rot]` **syntax unchanged**; the renderer resolves the parent chain
  internally, so `docs/PIXELSCRIPT_GRAMMAR.md` v2.1 and the 404-token vocabulary stay valid.
- Existing scripts that already encode absolute offsets must still render sensibly: introduce the
  hierarchy behind a flag (e.g. `RenderOptions.rig = "flat" | "hierarchical"`, default `flat`) and
  promote the default only after golden-image comparison.

### Regression risks

| Risk | Mitigation |
|---|---|
| Golden images change for every sprite | Keep `flat` as default; add golden tests for both modes |
| Determinism breaks | Hierarchy is pure integer arithmetic; assert byte-identical output across two runs |
| Coherence pass fights the hierarchy | Run `stabilize_offsets` **after** hierarchical resolution, not before |
| Weapon rotates oddly if parent rotation is large | Clamp inherited rotation, or attach with rotation ignored (positional socket only) |

### Tests that must be added

1. `test_weapon_stays_in_hand_all_frames` — 12 animations × 5 categories, assert
   `weapon_hand_distance <= 1`.
2. `test_hierarchical_matches_flat_for_torso_only_scripts`.
3. `test_hierarchical_is_deterministic` (two runs, byte equality).
4. Golden-image snapshot per category, `flat` mode unchanged (proves no visual regression).

### Benchmark needed

Before/after on the frozen prompt set: `silhouette_iou`, `connected_component_count`,
`opaque_pixel_count`, plus the new `weapon_hand_distance`. Add the numbers to a new
`docs/RIG_HIERARCHY.md`; do not overwrite existing benchmark records.

---

## 3. Recommendation R2 — Animation name aliasing / rejection

**Maps to:** `backend/spritemind/ml/temporal.py:302`, `server/schemas.py`,
`server/routers/animation.py`.

```python
ANIMATION_ALIASES = {
    "attack": "melee_attack", "slash": "melee_attack", "swing": "melee_attack",
    "shoot": "ranged_attack", "fire": "ranged_attack", "throw": "ranged_attack",
    "hurt": "hit_reaction", "damage": "taking_damage", "block": "blocking",
    "idle": "idle", "stand": "idle", "jump": "jump", "fall": "fall",
}
```

Unknown *and* unmappable names → HTTP 400 with the canonical list, instead of a silent walk.

Response gains `requested_animation` and (when remapped) `animation_aliased: true` so the client can
display it honestly.

**Regression risks:** any frontend or test that relies on the silent fallback. Search for
`"walk"` defaults in `server/routers/animation.py` and `frontend/src/pages/AnimationStudioPage.tsx`
before changing. Keep `walk` as the *default when the field is absent*, never as the *fallback for
a wrong value*.

---

## 4. Recommendation R3 — Aseprite JSON + Godot `SpriteFrames` export

**Maps to:** `backend/spritemind/renderer/export.py` (new functions next to `export_spritesheet`),
`backend/spritemind/assets/pack.py` (extra file inside `export_zip`), and one API route in
`server/routers/assets.py`.

New functions (illustrative):

```python
def export_aseprite_json(frames, frame_tags=None, scale=1) -> dict
def export_godot_spriteframes(frames, anim_name, fps=8, scale=4) -> str
```

Output added to the ZIP as `atlas/spritesheet.json` and `atlas/sprite_frames.tres`, referenced from
`manifest.json` **in addition to** the existing `texture_atlas.json` (never replacing it).

**Regression risks:** Y-axis convention differs between Aseprite (`frame.y` from top) and Unity;
Godot's `.tres` needs a stable `ExtResource` id. Keep the existing manifest keys untouched so old
consumers keep working.

**Verification:** open the generated `.json` in Aseprite and the `.tres` in Godot 4, then record
the outcome in `docs/EXPORT_FORMATS.md` — manual, but explicit.

---

## 5. Recommendation R4 — Motion continuity and IK

**Maps to:** `backend/spritemind/ml/temporal.py`, new `backend/spritemind/ml/ik.py`.

Order of application per frame:

```
base pose → motion table offsets (float) → interpolate across frames → IK foot plant
→ socket resolution → quantise to int → coherence pass → render
```

Quantise last is the important part: today offsets are integers inside
`compute_motion_offsets`, which is why neighbouring frames jump by several pixels.

**Regression risks:** IK must not move the hip so far that the torso detaches from the legs;
guard with `max(hip_dy) <= 2`. Interpolation must not change frame count or loop closure —
assert `frame[0]` and `frame[-1]` continuity for cyclic animations.

---

## 6. Recommendation R5 — Rigidity metrics in CI

**Maps to:** `backend/spritemind/ml/metrics.py`, `backend/spritemind/ml/benchmark.py`,
`backend/tests/`.

```python
def weapon_hand_distance(rgba, parts) -> int
def part_parent_gap(rgba, parts, parent_map) -> int
def max_interframe_part_delta(offsets_a, offsets_b) -> int
```

These use only numpy + existing component logic — no new dependency, and they deliberately avoid
duplicating what `scikit-image` would add.

**Regression risks:** thresholds must be calibrated on the *current* output first (record the
baseline) and only then enforced, otherwise CI fails on day one for reasons unrelated to a change.

---

## 7. What must not be touched while doing any of the above

- `docs/PIXELSCRIPT_GRAMMAR.md` and the tokenizer vocabulary (404 tokens).
- Any checkpoint (`svan_checkpoint_best.pt`, `svan_v4_palette_ep2.pt`, `socket_predictor_*.pt`).
- `data/splits/heldout_v1.json` and every existing benchmark/visual-comparison record.
- The 90% critic gate.
- The PR chain #27 → #28 → #29 → #30 → #31 and the production default engine.
- The data-deletion safeguards in `docs/DATA_MANAGEMENT.md`.

---

## 8. Ordering recommendation

1. **R2** (aliasing) — smallest, no visual change, removes a whole failure class.
2. **R5** (metrics) — measure before fixing, so R1/R4 can be proven.
3. **R1** (hierarchy/sockets) behind a flag, `flat` default preserved.
4. **R4** (interpolation + IK) once R1 is stable.
5. **R3** (export formats) — independent of the rig work, can run in parallel.

Rejected integrations are listed in `docs/TECHNOLOGY_COMPARISON.md` §9 with reasons.
