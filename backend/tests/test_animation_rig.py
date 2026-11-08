"""Milestone A: animation name resolution and parent-relative attachment.

Covers the three defects the animation audit found:
1. unsupported animation names were silently rendered as ``walk``;
2. ``attack`` and ``hurt`` produced byte-identical walk frames;
3. equipment drifted away from the hand when the holding limb moved.
"""
import numpy as np
import pytest

from spritemind.animations import (
    SUPPORTED_ANIMATIONS,
    UnknownAnimationError,
    resolve_animation_name,
)
from spritemind.ml.metrics import mask_min_distance, weapon_hand_distance
from spritemind.ml.temporal import TemporalAnimator
from spritemind.pixelscript.rig import ATTACHMENTS, RigGeometry, part_parent_map
from spritemind.renderer.engine import PixelScriptEngine


def _animate(anim, frames=4, mode="socketed"):
    return TemporalAnimator.generate_animation(
        base_script_or_prompt="",
        category="wizard",
        style="classic_16bit",
        anim_category=anim,
        palette_name="pico8",
        width=32,
        height=32,
        num_frames=frames,
        fps=8,
        engine_mode="procedural_baseline",
        seed=3,
        weapon="staff",
        accessory="cape",
        rig_mode=mode,
    )


def _frame_images(anim, frames=4, mode="socketed"):
    out = _animate(anim, frames=frames, mode=mode)
    return [
        PixelScriptEngine.execute(f["script_text"], width=32, height=32, palette_name="pico8")
        for f in out["frames"]
    ]


def _mask(result, part):
    mask = np.zeros((32, 32), dtype=bool)
    for x, y in result.canvas.parts[part].pixels:
        mask[y, x] = True
    return mask


# --- animation name resolution -------------------------------------------------


@pytest.mark.parametrize("name", SUPPORTED_ANIMATIONS)
def test_canonical_names_are_not_aliased(name):
    assert resolve_animation_name(name) == (name, False)


@pytest.mark.parametrize(
    "alias,canonical",
    [("attack", "melee_attack"), ("hurt", "hit_reaction"), ("spell", "cast"), ("sprint", "run")],
)
def test_documented_aliases_are_reported(alias, canonical):
    assert resolve_animation_name(alias) == (canonical, True)


def test_alias_matching_is_case_and_separator_insensitive():
    assert resolve_animation_name(" Melee-Attack ") == ("melee_attack", False)


def test_empty_name_keeps_the_historical_default():
    assert resolve_animation_name(None) == ("walk", False)
    assert resolve_animation_name("") == ("walk", False)


def test_unknown_name_is_rejected_instead_of_becoming_walk():
    with pytest.raises(UnknownAnimationError):
        resolve_animation_name("backflip")


def test_generation_surfaces_aliasing_instead_of_hiding_it():
    out = _animate("attack")
    assert out["requested_animation"] == "attack"
    assert out["animation_category"] == "melee_attack"
    assert out["animation_aliased"] is True
    assert out["rig_mode"] == "socketed"


def test_generation_rejects_an_unsupported_animation():
    with pytest.raises(UnknownAnimationError):
        _animate("backflip")


# --- attack and hurt are distinct from walk ------------------------------------


@pytest.mark.parametrize("anim", ["attack", "hurt"])
def test_combat_animations_are_not_byte_identical_to_walk(anim):
    walk = [f["script_text"] for f in _animate("walk")["frames"]]
    other = [f["script_text"] for f in _animate(anim)["frames"]]
    assert walk != other
    assert sum(1 for a, b in zip(walk, other) if a == b) == 0


def test_melee_attack_moves_the_weapon_arm_differently_from_walk():
    walk = _frame_images("walk")
    melee = _frame_images("melee_attack")
    walk_arm = [_mask(r, "arm_right") for r in walk]
    melee_arm = [_mask(r, "arm_right") for r in melee]
    assert any(not np.array_equal(a, b) for a, b in zip(walk_arm, melee_arm))


# --- weapon-to-hand distance ---------------------------------------------------


def test_mask_min_distance_measures_the_gap():
    a = np.zeros((8, 8), dtype=bool)
    b = np.zeros((8, 8), dtype=bool)
    a[2, 2] = True
    b[2, 5] = True
    assert mask_min_distance(a, b) == pytest.approx(3.0)
    assert weapon_hand_distance(a, b) == pytest.approx(3.0)


def test_mask_min_distance_of_empty_mask_is_none():
    a = np.zeros((8, 8), dtype=bool)
    b = np.zeros((8, 8), dtype=bool)
    b[1, 1] = True
    assert mask_min_distance(a, b) is None


@pytest.mark.parametrize("anim", ["idle", "walk", "run", "attack", "hurt", "cast", "dash"])
def test_weapon_stays_in_hand_with_sockets(anim):
    for result in _frame_images(anim):
        dist = weapon_hand_distance(_mask(result, "weapon"), _mask(result, "arm_right"))
        assert dist is not None
        assert dist <= 1.0, f"{anim}: weapon drifted {dist}px from the hand"


@pytest.mark.parametrize("anim", ["dash", "hit_reaction"])
def test_flat_mode_still_shows_the_historical_drift(anim):
    """Guards the baseline: the bug is reproducible with rig_mode='flat'."""
    drifted = []
    for result in _frame_images(anim, mode="flat"):
        dist = weapon_hand_distance(_mask(result, "weapon"), _mask(result, "arm_right"))
        drifted.append(dist)
    assert max(d for d in drifted if d is not None) > 1.0


def test_sockets_do_not_fragment_the_sprite():
    for anim in ("walk", "melee_attack"):
        for result in _frame_images(anim):
            from spritemind.ml.metrics import alpha_mask, connected_component_count

            assert connected_component_count(alpha_mask(result.composite_array)) == 1


# --- rig composition -----------------------------------------------------------


def test_rig_is_deterministic():
    offsets = {"arm_right": (3, -2, 25), "weapon": (1, 1, 5), "torso": (1, 0, 3)}
    geometry = RigGeometry.from_script(_sample_script(), 32, 32, "pico8")
    first = geometry.resolve(offsets)
    second = geometry.resolve(offsets)
    assert first == second


def test_rig_pulls_a_drifting_limb_back_to_its_parent():
    geometry = RigGeometry.from_script(_sample_script(), 32, 32, "pico8")
    resolved = geometry.resolve({"arm_right": (4, 0, 0), "weapon": (1, 1, 5)})
    assert abs(resolved["arm_right"][0]) <= 1, "a 4 px detach must be mostly removed"
    assert resolved["weapon"] != (1, 1, 5), "the weapon still follows the hand"


def test_rig_keeps_the_authored_rotation_of_body_parts():
    geometry = RigGeometry.from_script(_sample_script(), 32, 32, "pico8")
    resolved = geometry.resolve({"arm_right": (0, 0, 25)})
    assert resolved["arm_right"][2] == 25


def test_a_rigid_body_translation_is_not_altered():
    geometry = RigGeometry.from_script(_sample_script(), 32, 32, "pico8")
    body = {"torso": (4, 0, 0), "arm_right": (4, 0, 0), "weapon": (0, 0, 0)}
    resolved = geometry.resolve(body)
    for part in ("torso", "arm_right", "weapon", "head", "leg_left"):
        assert resolved[part][:2] == (4, 0), f"{part} must move with the body"


def test_weapon_follows_the_parent_translation():
    geometry = RigGeometry.from_script(_sample_script(), 32, 32, "pico8")
    base = geometry.resolve({"torso": (0, 0, 0), "arm_right": (0, 0, 0), "weapon": (0, 0, 0)})
    moved = geometry.resolve({"torso": (4, 0, 0), "arm_right": (4, 0, 0), "weapon": (0, 0, 0)})
    assert moved["weapon"][0] - base["weapon"][0] == 4
    assert moved["weapon"][1] == base["weapon"][1]


def test_accessory_follows_the_torso_without_spinning():
    geometry = RigGeometry.from_script(_cape_script(), 32, 32, "pico8")
    assert "accessory" in geometry.sockets
    assert ATTACHMENTS["accessory"]["inherit_rotation"] is False
    still = geometry.resolve({"torso": (0, 0, 0), "accessory": (0, 0, 0)})["accessory"]
    translated = geometry.resolve({"torso": (2, 1, 0), "accessory": (0, 0, 0)})["accessory"]
    assert (translated[0] - still[0], translated[1] - still[1]) == (2, 1)
    assert translated[2] == still[2] == 0

    rotated = geometry.resolve({"torso": (0, 0, 30), "accessory": (0, 0, 0)})["accessory"]
    assert rotated != still, "the cape must follow a rotating torso"
    assert rotated[2] == 0, "but the cape itself must not spin"


def test_declared_part_parent_overrides_the_default():
    geometry = RigGeometry.from_script(_cape_script(), 32, 32, "pico8")
    assert part_parent_map(geometry)["accessory"] == "torso"


def test_geometry_without_weapon_is_a_no_op():
    geometry = RigGeometry.from_script(_bare_script(), 32, 32, "pico8")
    offsets = {"torso": (1, 1, 5)}
    assert geometry.sockets == {}
    assert geometry.resolve(offsets) == offsets


@pytest.mark.parametrize(
    "anim", ["idle", "walk", "run", "cast", "dash", "melee_attack", "special_ability", "slime_bouncing"]
)
def test_rendered_limbs_stay_in_contact_with_their_parent(anim):
    # A bent knee or elbow meets its parent corner-to-corner, i.e. diagonally at
    # 1.41 px. Anything beyond that is a visible hole, not a joint.
    diagonal = 2 ** 0.5 + 1e-6
    geometry = RigGeometry.from_script(_sample_script(), 32, 32, "pico8")
    for result in _frame_images(anim, frames=6):
        for child, (_, _, parent) in geometry.joints.items():
            gap = mask_min_distance(_mask(result, child), _mask(result, parent))
            assert gap is not None
            assert gap <= diagonal, f"{anim}: {child} detached from {parent} by {gap} px"


# --- API surface ---------------------------------------------------------------


def test_api_returns_the_resolved_animation():
    from spritemind.server.routers.animation import generate_animation
    from spritemind.server.schemas import AnimationGenerateRequest

    out = generate_animation(
        AnimationGenerateRequest(
            prompt="a wizard with a staff",
            category="wizard",
            animation_category="attack",
            num_frames=4,
        )
    )
    assert out["animation_category"] == "melee_attack"
    assert out["requested_animation"] == "attack"


def test_api_rejects_an_unsupported_animation_with_400():
    from fastapi import HTTPException

    from spritemind.server.routers.animation import generate_animation
    from spritemind.server.schemas import AnimationGenerateRequest

    with pytest.raises(HTTPException) as exc:
        generate_animation(
            AnimationGenerateRequest(
                prompt="a wizard", category="wizard", animation_category="backflip", num_frames=4
            )
        )
    assert exc.value.status_code == 400
    assert "backflip" in str(exc.value.detail)


def _bare_script() -> str:
    return "PART torso none 16 16\nRECT 14 16 4 6 3 1\nEND PART\n"


def _cape_script() -> str:
    return (
        "PART torso none 16 14\nRECT 12 14 8 8 3 1\nEND PART\n"
        "PART accessory torso 16 14\nRECT 15 15 2 6 5 1\nEND PART\n"
    )


def _sample_script() -> str:
    out = TemporalAnimator.generate_animation(
        base_script_or_prompt="",
        category="wizard",
        style="classic_16bit",
        anim_category="idle",
        palette_name="pico8",
        width=32,
        height=32,
        num_frames=2,
        fps=8,
        engine_mode="procedural_baseline",
        seed=3,
        weapon="staff",
        accessory="cape",
    )
    return out["frames"][0]["script_text"]
