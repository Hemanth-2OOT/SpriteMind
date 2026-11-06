import numpy as np
import pytest

from spritemind.actions.builder import ActionSetBuilder
from spritemind.dataset.generator import CharacterGenerator
from spritemind.pixelscript.serializer import PixelScriptSerializer
from spritemind.dataset.temporal_dataset import TemporalDatasetGenerator
from spritemind.pixelscript import coherence
from spritemind.ml.temporal import TemporalAnimator
from spritemind.renderer.engine import PixelScriptEngine


def _script(category="knight", seed=1):
    program = CharacterGenerator.generate_character(category=category, seed=seed)
    return PixelScriptSerializer.serialize_program(program)


def _stats(script):
    return coherence._frame_stats(script, {}, 32, 32, "pico8")


@pytest.mark.parametrize("category", ["knight", "zombie", "wizard", "rogue", "animal", "fantasy_creature"])
@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8, 13])
def test_rest_pose_is_one_island(category, seed):
    islands, pixels = _stats(_script(category, seed))
    assert islands == 1, f"{category} seed {seed} has {islands} islands"
    assert pixels > 100


def test_stabilizer_keeps_limb_attached_to_its_parent():
    base = _script()
    offsets = {"arm_right": (-4, -5, 40), "head": (0, -6, 0)}
    stable = coherence.stabilize_offsets(base, offsets, 32, 32, "pico8")
    pixels, parents = coherence._rasterize(base, stable, 32, 32, "pico8")
    for name, parent in parents.items():
        if parent is None or parent not in pixels:
            continue
        assert coherence._distance(pixels[name], pixels[parent]) <= coherence.MAX_ATTACH_DISTANCE


def test_stabilizer_never_adds_islands():
    base = _script()
    base_islands, _ = _stats(base)
    for phase in (0.0, 0.25, 0.5, 0.75):
        offsets = TemporalDatasetGenerator.compute_motion_offsets("run", phase)
        stable = coherence.stabilize_offsets(base, offsets, 32, 32, "pico8")
        assert coherence._frame_stats(base, stable, 32, 32, "pico8")[0] == base_islands


def test_zero_offsets_are_returned_untouched():
    offsets = {"torso": (0, 0, 0), "head": (0, 0, 0)}
    assert coherence.stabilize_offsets(_script(), offsets, 32, 32, "pico8") == offsets


@pytest.mark.parametrize("anim", ["idle", "walk", "run", "melee_attack", "cast", "dash", "jump"])
def test_animated_frames_have_no_floating_parts(anim):
    base = _script("knight", 2)
    animation = TemporalAnimator.generate_animation(
        base_script_or_prompt=base,
        anim_category=anim,
        num_frames=5,
        category="knight",
        engine_mode="procedural_baseline",
        seed=3,
    )
    for index, frame in enumerate(animation["frames"]):
        islands, _ = coherence._frame_stats(frame["script_text"], {}, 32, 32, "pico8")
        assert islands == 1, f"{anim} frame {index} has {islands} islands"


def test_idle_frames_keep_a_stable_silhouette():
    result = ActionSetBuilder.build(
        prompt="a knight standing", category="knight", actions=["idle"], include_gif=False
    )
    counts = []
    for frame in result["actions"][0]["frames"]:
        arr = PixelScriptEngine.execute(
            frame["script_text"], width=32, height=32, palette_name="pico8"
        ).composite_array
        counts.append(int((arr[..., 3] > 0).sum()))
    # A 1px limb bob can tuck an arm inside the torso, which drops the outline
    # pixels it owned; allow 2% of the sprite rather than a fixed pixel count.
    assert max(counts) - min(counts) <= max(2, round(max(counts) * 0.02))
