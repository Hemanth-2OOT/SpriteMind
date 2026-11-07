"""Held-out probe: does the corpus honour the prompt words it stores?

Two arms, because a sample carries either a colour or an element tag:

* colour arm  -- generate from a colour prompt, score with the faithfulness metric
* element arm -- generate from an element prompt, measure the share of pixels in
                 that element's hue

    python scripts/probe_conditioning.py --samples 200
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from spritemind.assets.faithfulness import colour_faithfulness  # noqa: E402
from spritemind.dataset.conditioning import (  # noqa: E402
    ELEMENT_RGB,
    describe,
    element_hue_matches,
    renderable_colours,
)
from spritemind.dataset.generator import CATEGORIES, POSES, STYLES, WEAPONS, CharacterGenerator  # noqa: E402
from spritemind.pixelscript.validator import PixelScriptValidator  # noqa: E402
from spritemind.renderer.engine import PixelScriptEngine  # noqa: E402

PALETTE = "pico8"


def render(prog):
    return PixelScriptEngine.execute(prog, width=32, height=32, palette_name=PALETTE).composite_array


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    colours = renderable_colours(PALETTE)
    elements = sorted(ELEMENT_RGB)

    scores, present, element_scores, invalid = [], [], [], 0
    for i in range(args.samples):
        category = rng.choice(CATEGORIES)
        weapon = rng.choice(WEAPONS)
        is_element = i % 2 == 1
        colour = None if is_element else rng.choice(colours)
        element = rng.choice(elements) if is_element else None
        prompt = describe(
            category,
            colour=colour,
            element=element,
            weapon=None if weapon == "none" else weapon,
        )
        prog = CharacterGenerator.generate_character(
            category=category,
            style=rng.choice(STYLES),
            pose=rng.choice(POSES),
            palette_name=PALETTE,
            seed=args.seed + i * 31,
            weapon=weapon,
            colour=colour,
            element=element,
        )
        invalid += 0 if PixelScriptValidator.validate(prog, canvas_width=32, canvas_height=32).is_valid else 1
        rgba = render(prog)
        if colour:
            result = colour_faithfulness(prompt, rgba)
            if result:
                scores.append(result["score"])
                present.append(not result["missing"])
        else:
            element_scores.append(element_hue_matches(element, rgba.reshape(-1, 4).tolist()))

    mean_colour = sum(scores) / len(scores) if scores else 0.0
    present_rate = 100 * sum(present) / len(present) if present else 0.0
    mean_element = 100 * sum(element_scores) / len(element_scores) if element_scores else 0.0
    element_visible = 100 * sum(1 for s in element_scores if s >= 0.05) / len(element_scores) if element_scores else 0.0

    print(f"samples={args.samples} invalid={invalid}")
    print(f"colour present:      {present_rate:.1f}% of {len(present)} colour prompts")
    print(f"colour silhouette:   {mean_colour:.1f}% share")
    print(f"element visible:     {element_visible:.1f}% of {len(element_scores)} element prompts")
    print(f"element hue share:   {mean_element:.1f}% share")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
