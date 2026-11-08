"""Prompt colour faithfulness.

The text prompt only reaches the model as category/style/pose/palette today, so
"a blue wizard" and "a red wizard" share one distribution. This module measures
that gap in pixels: which colours the prompt asked for, and how much of the
silhouette actually uses them.
"""
import colorsys
import re

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Hue targets in degrees for chromatic names, plus achromatic special cases.
COLOUR_TARGETS: Dict[str, Dict[str, float]] = {
    "red": {"hue": 0.0},
    "crimson": {"hue": 348.0},
    "scarlet": {"hue": 8.0},
    "orange": {"hue": 30.0},
    "amber": {"hue": 40.0},
    "gold": {"hue": 45.0},
    "golden": {"hue": 45.0},
    "yellow": {"hue": 58.0},
    "lime": {"hue": 88.0},
    "green": {"hue": 120.0},
    "emerald": {"hue": 140.0},
    "teal": {"hue": 172.0},
    "turquoise": {"hue": 176.0},
    "cyan": {"hue": 186.0},
    "azure": {"hue": 202.0},
    "blue": {"hue": 220.0},
    "indigo": {"hue": 258.0},
    "purple": {"hue": 282.0},
    "violet": {"hue": 292.0},
    "magenta": {"hue": 302.0},
    "pink": {"hue": 330.0},
    "brown": {"hue": 26.0, "max_value": 0.62, "min_saturation": 0.30},
    "tan": {"hue": 32.0, "min_saturation": 0.18, "min_value": 0.55},
    "white": {"achromatic": 1.0},
    "silver": {"achromatic": 0.85},
    "gray": {"achromatic": 0.55},
    "grey": {"achromatic": 0.55},
    "black": {"achromatic": 0.0},
}

HUE_TOLERANCE = 24.0
MIN_SATURATION = 0.25
MIN_VALUE = 0.16
MATCH_SHARE_FOR_PRESENCE = 0.03

_WORD_RE = re.compile(r"[a-z]+")


def requested_colours(prompt: str) -> List[str]:
    """Colour names mentioned in the prompt, in order, without duplicates."""
    found: List[str] = []
    for word in _WORD_RE.findall(prompt.lower()):
        if word in COLOUR_TARGETS and word not in found:
            found.append(word)
    return found


def _hue_delta(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def _matches(r: int, g: int, b: int, target: Dict[str, float]) -> bool:
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    if "achromatic" in target:
        grey = target["achromatic"]
        return s < 0.18 and abs(v - grey) < 0.45
    if s < target.get("min_saturation", MIN_SATURATION) or v < target.get("min_value", MIN_VALUE):
        return False
    if "max_value" in target and v > target["max_value"]:
        return False
    return _hue_delta(h * 360.0, target["hue"]) <= HUE_TOLERANCE


def colour_faithfulness(prompt: str, rgba: np.ndarray) -> Optional[Dict[str, Any]]:
    """Share of opaque pixels drawn in a colour the prompt asked for.

    Returns ``None`` when the prompt names no colour, so callers can exclude the
    sample from an average instead of scoring it as a pass or a fail.
    """
    requested = requested_colours(prompt)
    if not requested:
        return None

    opaque = rgba[:, :, 3] > 0
    total = int(opaque.sum())
    if total == 0:
        return {"requested": requested, "score": 0.0, "matched_share": 0.0, "per_colour": {}, "missing": requested}

    pixels = rgba[opaque][:, :3]
    colours, counts = np.unique(pixels, axis=0, return_counts=True)
    per_colour = {name: 0.0 for name in requested}
    matched = 0
    for colour, count in zip(colours, counts):
        r, g, b = int(colour[0]), int(colour[1]), int(colour[2])
        hit = False
        for name in requested:
            if _matches(r, g, b, COLOUR_TARGETS[name]):
                per_colour[name] += float(count)
                hit = True
        if hit:
            matched += int(count)

    per_share = {name: round(value / total, 3) for name, value in per_colour.items()}
    return {
        "requested": requested,
        "score": round(matched / total * 100.0, 1),
        "matched_share": round(matched / total, 3),
        "per_colour": per_share,
        "missing": [name for name, share in per_share.items() if share < MATCH_SHARE_FOR_PRESENCE],
    }


def selection_hit(faith: Optional[Dict[str, Any]]) -> Optional[int]:
    """1/0 when a colour was requested and present, ``None`` when none was.

    Callers must drop ``None`` from hit-rate averages: scoring a prompt that
    asked for no colour as a miss silently halves every rate on mixed suites.
    """
    if not faith:
        return None
    return int(float(faith.get("matched_share") or 0.0) >= MATCH_SHARE_FOR_PRESENCE)
