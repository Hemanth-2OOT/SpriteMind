"""Prompt conditioning must reach the action-set body, not only the effects."""
import base64
import io

import numpy as np
from PIL import Image

from spritemind.actions import ActionSetBuilder


def _build(prompt: str, seed: int = 7):
    return ActionSetBuilder.build(
        prompt=prompt, actions=["idle"], seed=seed, include_gif=False
    )


def _pixels(action_set):
    frame = action_set["actions"][0]["frames"][0]
    data = base64.b64decode(frame["image_base64"].split(",")[-1])
    return np.array(Image.open(io.BytesIO(data)).convert("RGBA"))


def test_prompt_colour_changes_the_body():
    green = _build("green zombie mage")
    crimson = _build("crimson zombie mage")
    assert green["colour"] == "green"
    assert crimson["colour"] == "crimson"
    assert green["base_script"] != crimson["base_script"]
    assert np.any(_pixels(green) != _pixels(crimson))


def test_prompt_element_tints_the_body():
    icy = _build("ice paladin with shield")
    plain = _build("paladin with shield")
    assert icy["element"] == "ice"
    assert plain["element"] != "ice"
    assert icy["base_script"] != plain["base_script"]


def test_seed_changes_the_body():
    first = _build("zombie mage", seed=1)
    second = _build("zombie mage", seed=2024)
    assert first["base_script"] != second["base_script"]


def test_same_prompt_and_seed_are_reproducible():
    assert _build("zombie mage", seed=99)["base_script"] == _build(
        "zombie mage", seed=99
    )["base_script"]


def _meta(action_set) -> str:
    return action_set["base_script"].splitlines()[0]


def test_prompt_weapon_reaches_the_body():
    with_staff = _build("a mage with a staff", seed=5)
    with_sword = _build("a mage with a sword", seed=5)
    assert "weapon=staff" in _meta(with_staff)
    assert "weapon=sword" in _meta(with_sword)
    assert with_staff["base_script"] != with_sword["base_script"]


def test_weapon_synonyms_map_to_rig_weapons():
    from spritemind.dataset.conditioning import parse_conditioning

    assert parse_conditioning("wizard with a wand")["weapon"] == "staff"
    assert parse_conditioning("rogue with a katana")["weapon"] == "sword"
    assert parse_conditioning("knight in a cloak")["accessory"] == "cape"
    assert parse_conditioning("plain knight")["weapon"] is None


def test_pipeline_prompt_parsing_keeps_gear_words(tmp_path):
    from spritemind.ml.pipeline import SpriteMindPipeline

    parsed = SpriteMindPipeline(checkpoints_dir=str(tmp_path)).parse_prompt(
        "crimson fire mage with a wand and a cloak"
    )
    assert parsed["colour"] == "crimson"
    assert parsed["element"] == "fire"
    assert parsed["weapon"] == "staff"
    assert parsed["accessory"] == "cape"
