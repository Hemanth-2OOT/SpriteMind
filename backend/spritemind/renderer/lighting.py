"""Deterministic lighting pass.

The rig decides where the pixels are; this only re-colours a thin band of them so
a flat silhouette reads as a lit object. Nothing outside the existing alpha mask
is touched, no pixel is added or removed, and every new colour is snapped to the
sprite's own palette.
"""
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

RGBA = Tuple[int, int, int, int]

# Fraction of the way to the palette's lightest / darkest colour.
RIM_MIX = 0.34
AO_MIX = 0.42
# A step has to be visible to a player looking at a 64 px sprite.
MIN_LUMA_STEP = 10.0


def _luma(colour: Sequence[int]) -> float:
    r, g, b = int(colour[0]), int(colour[1]), int(colour[2])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _snap(target: Sequence[float], palette: Sequence[RGBA]) -> Optional[RGBA]:
    """Nearest palette colour to an RGB target, ignoring entries equal to it."""
    best: Optional[RGBA] = None
    best_d = float("inf")
    tr, tg, tb = float(target[0]), float(target[1]), float(target[2])
    for colour in palette:
        d = (colour[0] - tr) ** 2 + (colour[1] - tg) ** 2 + (colour[2] - tb) ** 2
        if d < best_d:
            best_d = d
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
        key = tuple(int(v) for v in comp[y, x])
        seen[key] = key
    return list(seen.values())


def apply_lighting(
    comp: np.ndarray,
    palette: Optional[Iterable[Sequence[int]]] = None,
    light_dir: Tuple[int, int] = (-1, -1),
    rim: float = RIM_MIX,
    ao: float = AO_MIX,
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
            cur_luma = _luma(current)

            # Lit edge: nothing in front of this pixel on the lit side.
            ox, oy = x + lx, y + ly
            lit_edge = not (0 <= ox < w and 0 <= oy < h) or not opaque[oy, ox]
            if lit_edge:
                target = [
                    current[c] + (_luma(lightest) - current[c]) * rim
                    for c in range(3)
                ]
                snapped = _snap(target, entries)
                if snapped is not None and _luma(snapped) - cur_luma >= MIN_LUMA_STEP:
                    out[y, x] = [snapped[0], snapped[1], snapped[2], current[3]]
                    continue

            # Contact shadow: transparent on the side the light leaves.
            sx, sy = x - lx, y - ly
            shadow_edge = not (0 <= sx < w and 0 <= sy < h) or not opaque[sy, sx]
            if shadow_edge:
                target = [
                    current[c] + (_luma(darkest) - current[c]) * ao
                    for c in range(3)
                ]
                snapped = _snap(target, entries)
                if snapped is not None and cur_luma - _luma(snapped) >= MIN_LUMA_STEP:
                    out[y, x] = [snapped[0], snapped[1], snapped[2], current[3]]

    return out
