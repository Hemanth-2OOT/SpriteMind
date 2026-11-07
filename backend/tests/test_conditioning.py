"""Prompt conditioning: palette-aware colour mapping and its effect on the corpus."""
from __future__ import annotations

import json

import pytest

from spritemind.dataset.conditioning import (
    COLOUR_RGB,
    HUE_TOLERANCE,
    _hue_delta,
    describe,
    element_hue_matches,
    palette_index,
    ramp,
    renderable_colours,
)
from spritemind.assets.faithfulness import colour_faithfulness
from spritemind.dataset.generator import CharacterGenerator
from spritemind.dataset.storage import DatasetManager
from spritemind.renderer.engine import PixelScriptEngine
from spritemind.utils.palettes import PALETTES


def _render(prog):
    return PixelScriptEngine.execute(prog, width=32, height=32, palette_name="pico8").composite_array


def _opaque(rgba):
    return sorted({tuple(int(v) for v in px[:3]) for px in rgba.reshape(-1, 4) if px[3] > 0})


def test_palette_index_matches_hue_not_rgb_distance():
    blue = PALETTES["pico8"][palette_index("pico8", COLOUR_RGB["blue"])][:3]
    grey = PALETTES["pico8"][palette_index("pico8", COLOUR_RGB["grey"])][:3]
    assert _hue_delta(blue, COLOUR_RGB["blue"]) <= HUE_TOLERANCE
    assert max(grey) - min(grey) <= 24, "grey must land on an achromatic entry"


def test_renderable_colours_excludes_what_the_palette_cannot_show():
    names = renderable_colours("pico8")
    assert "red" in names and "blue" in names and "white" in names
    assert "lime" not in names


def test_ramp_returns_a_hue_consistent_base_highlight_and_shadow():
    base, highlight, shadow = ramp("pico8", COLOUR_RGB["red"])
    entry = PALETTES["pico8"][base][:3]
    assert _hue_delta(entry, COLOUR_RGB["red"]) <= HUE_TOLERANCE
    assert len({base, highlight, shadow}) == 3


def test_ramp_falls_back_for_achromatic_colours():
    base, highlight, shadow = ramp("pico8", COLOUR_RGB["black"])
    assert len({base, highlight, shadow}) >= 2


def test_describe_puts_the_prompt_words_in_the_text():
    prompt = describe("knight", colour="red", element="ice", weapon="sword")
    assert "red" in prompt and "ice" in prompt and "sword" in prompt


def test_colour_conditioning_changes_the_pixels_and_keeps_the_hue():
    plain = _render(CharacterGenerator.generate_character(category="knight", seed=7))
    red = _render(CharacterGenerator.generate_character(category="knight", seed=7, colour="red"))
    assert _opaque(plain) != _opaque(red)
    result = colour_faithfulness("a red knight", red)
    assert result and not result["missing"]
    assert result["score"] > 30.0


def test_element_conditioning_is_visible_on_its_own():
    prog = CharacterGenerator.generate_character(category="animal", seed=11, element="ice")
    rgba = _render(prog)
    assert element_hue_matches("ice", rgba.reshape(-1, 4).tolist()) >= 0.3


def test_dataset_records_carry_prompt_conditioning(tmp_path):
    storage = DatasetManager(data_root=str(tmp_path))
    storage.generate_dataset(name="conditioning_probe", total_samples=12, defect_rate=0.0, seed=5, render_sample_limit=1)
    records = [json.loads(line) for line in (tmp_path / "conditioning_probe" / "records.jsonl").read_text().splitlines()]
    assert records
    for record in records:
        assert "colour" in record and "element" in record
        assert record["prompt"]
    tagged = [r for r in records if r["colour"] or r["element"]]
    assert tagged, "at least some samples must be colour- or element-tagged"
    for record in tagged:
        word = record["colour"] or record["element"]
        assert word in record["prompt"]
