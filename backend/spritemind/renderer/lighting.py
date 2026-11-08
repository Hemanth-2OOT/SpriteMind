"""Deterministic lighting pass.

The rig decides where the pixels are; this only re-colours a thin band of them so
a flat silhouette reads as a lit object. Nothing outside the existing alpha mask
is touched, no pixel is added or removed, and every new colour is snapped to the
sprite's own palette.

A pixel is pushed to another palette entry only when that step is visible
(``MIN_LUMA_STEP``); the entry closest to the ``strength``-scaled target wins, so
a low strength gives one subtle band and a high one a stronger one.
"""
import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

RGBA = Tuple[int, int, int, int]

# Fraction of the way to the palette's lightest / darkest colour used as the
# target when picking which entry to step to.
RIM_MIX = 0.30
AO_MIX = 0.38
MIN_LUMA_STEP = 8.0

# Global dial so the strength can be compared without editing code.
DEFAULT_STRENGTH = float(os.getenv("SPRITEMIND_SHADE_STRENGTH", "1.0"))


def _luma(colour: Sequence[int]) -> float:
    r, g, b = int(colour[0]), int(colour[1]), int(colour[2])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _pick(
    current: Sequence[int], entries: Sequence[RGBA], target: Sequence[float], direction: int
) -> Optional[RGBA]:
    """Palette entry nearest the target luma, never closer than a visible step."""
    cur_luma = _luma(current)
    target_luma = _luma(target)
    best: Optional[RGBA] = None
    best_score = float("inf")
    for colour in entries:
        if (_luma(colour) - cur_luma) * direction < MIN_LUMA_STEP:
            continue
        score = abs(_luma(colour) - target_luma) * 4.0
        score += ((colour[0] - target[0]) ** 2 + (colour[1] - target[1]) ** 2 + (colour[2] - target[2]) ** 2) ** 0.5
        if score < best_score:
            best_score = score
            best = colour
    return best


def palette_entries(comp: np.ndarray, palette: Optional[Iterable[Sequence[int]]] = None) -> list:
    """The sprite's palette plus any colour it actually uses, deduplicated."""
    seen = {}
    for colour in palette or ():
        key = (int(colour[0]), int(colour[1]), int(colour[2]), int(colour[3]) if len(colour) > 3 else 255)
        seen[key] = key
    alpha = comp[:, :, 3]
    for y, x in zip(*np.nonzero(alpha)):
        seen[tuple(int(v) for v in comp[y, x])] = tuple(int(v) for v in comp[y, x])
    return list(seen.values())


def apply_lighting(
    comp: np.ndarray,
    palette: Optional[Iterable[Sequence[int]]] = None,
    light_dir: Tuple[int, int] = (-1, -1),
    rim: float = RIM_MIX,
    ao: float = AO_MIX,
    strength: float = DEFAULT_STRENGTH,
) -> np.ndarray:
    """Adds a rim light and contact shadow inside the existing silhouette.

    ``light_dir`` is the direction the light comes from, so the lit band sits on
    the side the light hits and the shadow band opposite it. Transparent pixels
    stay transparent and alpha is never modified.
    """
    out = comp.copy()
    alpha = comp[:, :, 3]
    opaque = alpha > 0
    if not opaque.any():
        return out

    rim = max(0.0, min(1.0, rim * strength))
    ao = max(0.0, min(1.0, ao * strength))
    if rim == 0.0 and ao == 0.0:
        return out

    entries = palette_entries(comp, palette)
    if len(entries) < 2:
        return out

    lightest = max(entries, key=_luma)
    darkest = min(entries, key=_luma)
    lx, ly = light_dir
    h, w = opaque.shape

    for y in range(h):
        for x in range(w):
            if not opaque[y, x]:
                continue

            current = comp[y, x]

            # Lit edge: nothing in front of this pixel on the lit side.
            ox, oy = x + lx, y + ly
            lit_edge = not (0 <= ox < w and 0 <= oy < h) or not opaque[oy, ox]
            if lit_edge:
                target = [current[c] + (_luma(lightest) - current[c]) * rim for c in range(3)]
                picked = _pick(current, entries, target, 1)
                if picked is not None:
                    out[y, x] = [picked[0], picked[1], picked[2], current[3]]
                    continue

            # Contact shadow: transparent on the side the light leaves.
            sx, sy = x - lx, y - ly
            shadow_edge = not (0 <= sx < w and 0 <= sy < h) or not opaque[sy, sx]
            if shadow_edge:
                target = [current[c] + (_luma(darkest) - current[c]) * ao for c in range(3)]
                picked = _pick(current, entries, target, -1)
                if picked is not None:
                    out[y, x] = [picked[0], picked[1], picked[2], current[3]]

    return out
