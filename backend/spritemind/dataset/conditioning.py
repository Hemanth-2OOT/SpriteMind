"""Prompt conditioning for the synthetic corpus.

Turns free-form description words (colour, element, weapon) into palette indices the
generator can act on, so a record's prompt and its pixels agree. The point is to give
SVAN a conditioning signal worth learning from, not to be a general NLP parser.
"""
from __future__ import annotations

import colorsys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spritemind.dataset.generator import ACCESSORIES, WEAPONS
from spritemind.utils.palettes import PALETTES

RGB = Tuple[int, int, int]

COLOUR_RGB: Dict[str, RGB] = {
    "red": (220, 40, 40),
    "crimson": (180, 20, 50),
    "scarlet": (255, 60, 20),
    "orange": (245, 130, 30),
    "amber": (240, 170, 40),
    "gold": (235, 195, 45),
    "golden": (235, 195, 45),
    "yellow": (240, 230, 80),
    "lime": (150, 220, 50),
    "green": (70, 175, 70),
    "emerald": (35, 185, 120),
    "teal": (30, 150, 145),
    "turquoise": (60, 200, 190),
    "cyan": (60, 215, 235),
    "azure": (50, 140, 235),
    "blue": (55, 80, 220),
    "indigo": (85, 60, 190),
    "purple": (140, 60, 200),
    "violet": (165, 85, 215),
    "magenta": (215, 50, 190),
    "pink": (240, 130, 180),
    "brown": (130, 85, 50),
    "tan": (200, 165, 120),
    "white": (245, 245, 245),
    "silver": (185, 195, 205),
    "grey": (140, 140, 145),
    "gray": (140, 140, 145),
    "black": (30, 30, 35),
}

ELEMENT_RGB: Dict[str, RGB] = {
    "fire": (255, 120, 30),
    "flame": (255, 120, 30),
    "ice": (120, 220, 255),
    "frost": (120, 220, 255),
    "poison": (120, 220, 60),
    "toxic": (120, 220, 60),
    "lightning": (255, 240, 90),
    "thunder": (255, 240, 90),
}

#: Model-facing ids: 0 means "no colour/element requested", ids follow a stable alphabetical order.
COLOUR_IDS: Dict[str, int] = {name: i + 1 for i, name in enumerate(sorted(COLOUR_RGB))}
ELEMENT_IDS: Dict[str, int] = {name: i + 1 for i, name in enumerate(sorted(ELEMENT_RGB))}
NUM_COLOUR_IDS = len(COLOUR_IDS) + 1
NUM_ELEMENT_IDS = len(ELEMENT_IDS) + 1

#: Palette ids follow `PALETTES` order, which is also what the pipeline's
#: `_palette_index` uses when it feeds a checkpoint's palette embedding.
PALETTE_IDS: Dict[str, int] = {name: i for i, name in enumerate(PALETTES)}
NUM_PALETTE_IDS = len(PALETTE_IDS)


def palette_id(name: Optional[str]) -> int:
    return PALETTE_IDS.get(str(name), 0) if name else 0


def colour_id(name: Optional[str]) -> int:
    return COLOUR_IDS.get(str(name).lower(), 0) if name else 0


def element_id(name: Optional[str]) -> int:
    return ELEMENT_IDS.get(str(name).lower(), 0) if name else 0


# Everyday synonyms so a prompt like "mage with a wand" picks the staff rig
# instead of falling back to a random weapon.
WEAPON_SYNONYMS: Dict[str, str] = {
    "wand": "staff",
    "stave": "staff",
    "scepter": "staff",
    "katana": "sword",
    "blade": "sword",
    "sabre": "sword",
    "knife": "dagger",
    "crossbow": "bow",
    "longbow": "bow",
    "buckler": "shield",
}

ACCESSORY_SYNONYMS: Dict[str, str] = {
    "cloak": "cape",
    "helm": "helmet",
    "helmet": "helmet",
    "horn": "horns",
    "wing": "wings",
    "wings": "wings",
    "satchel": "backpack",
    "armor": "pauldrons",
    "armour": "pauldrons",
}

WEAPON_WORDS = tuple(WEAPONS) + tuple(WEAPON_SYNONYMS)
ACCESSORY_WORDS = tuple(ACCESSORIES) + tuple(ACCESSORY_SYNONYMS)


def _normalise(word: str, synonyms: Dict[str, str]) -> str:
    return synonyms.get(word, word)


def _luma(rgb: RGB) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def palette_entries(palette_name: str) -> List[Tuple[int, RGB]]:
    colours = PALETTES.get(palette_name) or PALETTES["pico8"]
    return [(i, tuple(c[:3])) for i, c in enumerate(colours) if i > 0 and c[3] > 0]


HUE_TOLERANCE = 24.0


def _hsv(rgb: RGB) -> Tuple[float, float, float]:
    r, g, b = (channel / 255 for channel in rgb)
    return colorsys.rgb_to_hsv(r, g, b)


def _hue_delta(a: RGB, b: RGB) -> float:
    delta = abs(_hsv(a)[0] - _hsv(b)[0]) * 360
    return min(delta, 360 - delta)


def _is_achromatic(rgb: RGB) -> bool:
    _, saturation, value = _hsv(rgb)
    return saturation < 0.18 or value < 0.12


def _colour_cost(target: RGB, entry: RGB) -> float:
    """Hue-first distance. Plain RGB distance maps "blue" onto cyan and "grey" onto purple."""
    if _is_achromatic(target) or _is_achromatic(entry):
        # Achromatic targets only match achromatic entries, by lightness.
        penalty = 0.0 if _is_achromatic(target) == _is_achromatic(entry) else 120.0
        return penalty + abs(_luma(target) - _luma(entry))
    _, ts, tv = _hsv(target)
    _, es, ev = _hsv(entry)
    return 4 * _hue_delta(target, entry) + 90 * abs(ts - es) + 60 * abs(tv - ev)


def palette_index(palette_name: str, rgb: RGB) -> int:
    best_i, best_cost = 1, float("inf")
    for index, entry in palette_entries(palette_name):
        cost = _colour_cost(rgb, entry)
        if cost < best_cost:
            best_i, best_cost = index, cost
    return best_i


def renderable_colours(palette_name: str) -> List[str]:
    """Colour names this palette can show *as judged by the faithfulness metric*.

    Checked by painting a solid block and scoring it, so the corpus never tags a
    prompt with a colour the palette cannot express under our own definition.
    """
    from spritemind.assets.faithfulness import colour_faithfulness

    names: List[str] = []
    for name, rgb in COLOUR_RGB.items():
        if name in names:
            continue
        entry = PALETTES.get(palette_name, PALETTES["pico8"])[palette_index(palette_name, rgb)]
        block = np.zeros((8, 8, 4), dtype=np.uint8)
        block[:, :] = entry[:3] + (255,)
        result = colour_faithfulness(f"a {name} sprite", block)
        if result and not result["missing"] and result["score"] >= 99.0:
            names.append(name)
    return names


def ramp(palette_name: str, rgb: RGB) -> Tuple[int, int, int]:
    """(base, highlight, shadow) indices nearest to the colour, ordered by luminance."""
    base = palette_index(palette_name, rgb)
    entries = palette_entries(palette_name)
    hue_siblings = [pair for pair in entries if _hue_delta(pair[1], rgb) <= HUE_TOLERANCE]
    if len(hue_siblings) > 1 and base in [pair[0] for pair in hue_siblings]:
        ordered = sorted(hue_siblings, key=lambda pair: _luma(pair[1]))
        position = [pair[0] for pair in ordered].index(base)
    else:
        ordered = sorted(entries, key=lambda pair: _luma(pair[1]))
        position = [pair[0] for pair in ordered].index(base)
    shadow = ordered[max(0, position - 1)][0]
    highlight = ordered[min(len(ordered) - 1, position + 1)][0]
    return base, highlight, shadow


def _hue_of(rgb: RGB) -> float:
    r, g, b = (channel / 255 for channel in rgb)
    h, _, _ = colorsys.rgb_to_hsv(r, g, b)
    return h * 360


def parse_conditioning(prompt: str) -> Dict[str, Any]:
    """Colour, element, weapon and accessory words in a prompt, in that order."""
    words = prompt.lower().replace("_", " ").split()
    colours = [w for w in words if w in COLOUR_RGB]
    elements = [w for w in words if w in ELEMENT_RGB]
    weapons = [_normalise(w, WEAPON_SYNONYMS) for w in words if w in WEAPON_WORDS]
    weapons = [w for w in weapons if w not in ("none",)]
    accessories = [_normalise(w, ACCESSORY_SYNONYMS) for w in words if w in ACCESSORY_WORDS]
    accessories = [w for w in accessories if w not in ("none",)]
    return {
        "colour": colours[0] if colours else None,
        "element": elements[0] if elements else None,
        "weapon": weapons[0] if weapons else None,
        "accessory": accessories[0] if accessories else None,
    }


def describe(category: str, colour: Optional[str] = None, element: Optional[str] = None,
             weapon: Optional[str] = None) -> str:
    """A prompt whose words the generator actually honours."""
    parts = ["a"]
    if colour:
        parts.append(colour)
    if element:
        parts.append(f"{element}-touched")
    if weapon:
        parts.append(weapon)
    parts.append(category.replace("_", " "))
    return " ".join(parts)


def element_hue_matches(element: str, rgba_pixels, palette_name: str = "pico8") -> float:
    """Share of opaque pixels that carry the element's palette tint.

    Accepts either a flat sequence of (r, g, b, a) tuples or an (h, w, 4) array.
    """
    if element not in ELEMENT_RGB:
        return 0.0
    rgba_pixels = np.asarray(rgba_pixels).reshape(-1, 4)
    entry = PALETTES.get(palette_name, PALETTES["pico8"])[ramp(palette_name, ELEMENT_RGB[element])[0]][:3]
    total = matched = 0
    for r, g, b, a in rgba_pixels:
        if a == 0:
            continue
        total += 1
        if _hue_delta((r, g, b), entry) <= HUE_TOLERANCE:
            matched += 1
    return (matched / total) if total else 0.0
