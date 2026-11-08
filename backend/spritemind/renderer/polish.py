"""Deterministic pixel-art clean-up applied after shading.

Three defects survive the rig and the shading pass and are corrected here:

* floating stray pixels -- opaque islands of one or two pixels that touch
  nothing else, which read as noise next to the character;
* notches -- single transparent pixels surrounded on three sides, which is how
  a staircase diagonal leaves a visible bite in the silhouette;
* mixed outline weight -- edges that are body colour in one place and the
  outline colour in another, which is the most obvious "unfinished" tell.

Every change is made in place on the alpha mask and uses a colour that is
already in the palette, so the pass cannot introduce anti-aliasing, gradients
or new colours.  It is idempotent: a second run changes nothing.
"""
from typing import Optional, Sequence, Tuple

import numpy as np

Colour = Tuple[int, int, int]


def _alpha_mask(comp: np.ndarray) -> np.ndarray:
    return comp[:, :, 3] > 0


def _luma(colour: Sequence[int]) -> float:
    return 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2]


def _palette_entries(palette: Optional[Sequence[Sequence[int]]]) -> list:
    if not palette:
        return []
    return [tuple(int(v) for v in entry[:3]) for entry in palette]


def _outline_colour(comp: np.ndarray, palette: Optional[Sequence[Sequence[int]]]) -> Colour:
    """The darkest colour the sprite actually uses, or the darkest palette entry."""
    mask = _alpha_mask(comp)
    used = [tuple(int(v) for v in comp[y, x, :3]) for y, x in zip(*np.nonzero(mask))]
    entries = sorted(set(used) | set(_palette_entries(palette)), key=_luma)
    return entries[0] if entries else (0, 0, 0)


def drop_stray_pixels(comp: np.ndarray, min_size: int = 3) -> np.ndarray:
    """Erases opaque islands smaller than `min_size` pixels."""
    out = comp.copy()
    mask = _alpha_mask(out)
    seen = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        island = []
        while stack:
            y, x = stack.pop()
            island.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(island) < min_size:
            for y, x in island:
                out[y, x] = 0
    return out


def fill_notches(comp: np.ndarray, palette: Optional[Sequence[Sequence[int]]] = None) -> np.ndarray:
    """Fills a transparent pixel that is walled in on three or more sides."""
    out = comp.copy()
    mask = _alpha_mask(out)
    pad = np.pad(mask, 1, constant_values=False)
    up, down = pad[:-2, 1:-1], pad[2:, 1:-1]
    left, right = pad[1:-1, :-2], pad[1:-1, 2:]
    count = up.astype(np.uint8) + down + left + right
    holes = mask == False  # noqa: E712 - explicit for readability
    dark = _outline_colour(out, palette)
    body = mask & (np.abs(out[:, :, :3].astype(int) - np.array(dark)).sum(2) > 0)
    pad = np.pad(body, 1, constant_values=False)
    body_count = (
        pad[:-2, 1:-1].astype(np.uint8)
        + pad[2:, 1:-1]
        + pad[1:-1, :-2]
        + pad[1:-1, 2:]
    )
    # A gap that is walled in only by the outline is a hole between limbs, not
    # a notch in the body, so it is left open.
    targets = holes & (count >= 3) & (body_count >= 2)
    if not targets.any():
        return out
    for y, x in zip(*np.nonzero(targets)):
        neighbours = [
            out[y + dy, x + dx, :3]
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= y + dy < out.shape[0] and 0 <= x + dx < out.shape[1] and mask[y + dy, x + dx]
        ]
        inside = np.mean(np.array(neighbours, dtype=float), axis=0).round().astype(int)
        pick = min(
            _palette_entries(palette) or [tuple(int(v) for v in inside)],
            key=lambda c: abs(_luma(c) - _luma(inside)),
        )
        out[y, x, :3] = pick
        out[y, x, 3] = 255
    return out


def close_outline(comp: np.ndarray, palette: Optional[Sequence[Sequence[int]]] = None) -> np.ndarray:
    """Draws one pixel of outline around the silhouette, into empty space.

    Painting outwards rather than over the body keeps the rim light that the
    shading pass puts on the outermost body pixel, and it gives every edge the
    same one-pixel weight.  Only transparent pixels with an opaque orthogonal
    neighbour are painted, so a second run finds nothing left to do.
    """
    out = comp.copy()
    mask = _alpha_mask(out)
    dark = _outline_colour(out, palette)
    # Only the body counts as a seed: an existing outline must not be outlined
    # again, or repeated runs would grow the sprite a pixel at a time.
    body = mask & (np.abs(out[:, :, :3].astype(int) - np.array(dark)).sum(2) > 0)
    ring = np.zeros_like(mask)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = np.roll(body, (dy, dx), axis=(0, 1))
        if dy == 1:
            shifted[0, :] = False
        elif dy == -1:
            shifted[-1, :] = False
        if dx == 1:
            shifted[:, 0] = False
        elif dx == -1:
            shifted[:, -1] = False
        ring |= shifted
    ring &= ~mask
    for y, x in zip(*np.nonzero(ring)):
        out[y, x, :3] = dark
        out[y, x, 3] = 255
    return out


def polish(comp: np.ndarray, palette: Optional[Sequence[Sequence[int]]] = None) -> np.ndarray:
    """Cleans up stray pixels, notches and outline weight in one pass."""
    if comp.ndim != 3 or comp.shape[2] != 4:
        raise ValueError("polish expects an RGBA array")
    out = drop_stray_pixels(comp)
    out = fill_notches(out, palette)
    out = close_outline(out, palette)
    return out
