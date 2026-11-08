"""Image-grounded metrics for SpriteMind evaluation.

Every function here measures pixels or parse results directly. Nothing is
derived from renderer error counts, and no score is a linear squashing of a
distance. This module exists because the legacy benchmark metrics in
`spritemind.ml.benchmark.compute_metrics` reported a "silhouette IoU" that was
a density proxy and a "palette adherence" that was 100% by construction.
"""
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from spritemind.assets.faithfulness import colour_faithfulness
from spritemind.utils.palettes import PALETTES


def alpha_mask(rgba: np.ndarray) -> np.ndarray:
    return rgba[..., 3] > 0


def silhouette_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """True intersection-over-union of two boolean masks."""
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shape mismatch: {a.shape} vs {b.shape}")
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def palette_adherence(rgba: np.ndarray, palette_name: str) -> float:
    """Percentage of opaque pixels whose RGB is an exact member of the palette."""
    palette = PALETTES.get(palette_name)
    if not palette:
        return float("nan")
    allowed = {(r, g, b) for r, g, b, *_ in (tuple(entry) for entry in palette)}
    opaque = rgba[..., 3] > 0
    total = int(opaque.sum())
    if total == 0:
        return 0.0
    rgb = rgba[..., :3][opaque]
    matches = sum(1 for pixel in map(tuple, rgb) if pixel in allowed)
    return 100.0 * matches / total


def opaque_pixel_count(rgba: np.ndarray) -> int:
    return int((rgba[..., 3] > 0).sum())


def bounding_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def connected_component_count(mask: np.ndarray) -> int:
    """4-connected components of an opaque mask."""
    seen = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    count = 0
    for sy, sx in zip(*np.where(mask)):
        if seen[sy, sx]:
            continue
        count += 1
        stack = [(int(sy), int(sx))]
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
    return count


def largest_component_share(mask: np.ndarray) -> float:
    """Fraction of opaque pixels belonging to the largest connected component."""
    total = int(mask.sum())
    if total == 0:
        return 0.0
    seen = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    best = 0
    for sy, sx in zip(*np.where(mask)):
        if seen[sy, sx]:
            continue
        size = 0
        stack = [(int(sy), int(sx))]
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            size += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        best = max(best, size)
    return best / total


def column_symmetry(rgba: np.ndarray) -> float:
    """1.0 for a perfectly left-right symmetric silhouette, 0.0 for none."""
    mask = alpha_mask(rgba)
    col = mask.sum(axis=0).astype(float)
    total = col.sum()
    if total == 0:
        return 0.0
    return float(1.0 - np.abs(col - col[::-1]).sum() / (2.0 * total))


def interior_holes(mask: np.ndarray) -> int:
    """Background pixels that are not reachable from the border (4-connectivity)."""
    height, width = mask.shape
    background = ~mask
    outside = np.zeros_like(background)
    stack = [(y, x) for y in (0, height - 1) for x in range(width) if background[y, x]]
    stack += [(y, x) for x in (0, width - 1) for y in range(height) if background[y, x]]
    for y, x in stack:
        outside[y, x] = True
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and background[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                stack.append((ny, nx))
    return int((background & ~outside).sum())


def evaluate_sprite(
    rgba: np.ndarray,
    palette_name: str,
    reference_mask: Optional[np.ndarray] = None,
    prompt: str = "",
    element: Optional[str] = None,
) -> Dict[str, Any]:
    """Full pixel-level report for one rendered sprite.

    `reference_mask` is the silhouette IoU is measured against; callers must
    state which reference they used, because IoU is only meaningful relative to
    a named target.
    """
    mask = alpha_mask(rgba)
    rows: Dict[str, Any] = {
        "opaque_pixels": opaque_pixel_count(rgba),
        "palette_adherence_pct": round(palette_adherence(rgba, palette_name), 2),
        "connected_components": connected_component_count(mask),
        "largest_component_share": round(largest_component_share(mask), 4),
        "column_symmetry": round(column_symmetry(rgba), 4),
        "interior_hole_pixels": interior_holes(mask),
        "bbox": bounding_box(mask),
    }
    if reference_mask is not None:
        rows["silhouette_iou"] = round(silhouette_iou(mask, reference_mask), 4)
    if prompt:
        faithfulness = colour_faithfulness(prompt, rgba)
        if faithfulness is not None:
            rows["colour_faithfulness"] = faithfulness["score"]
            rows["colour_requested"] = faithfulness["requested"]
    return rows


def summarise(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean / stderr over a list of per-sample metric dicts."""
    numeric: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric.setdefault(key, []).append(float(value))

    out: Dict[str, Any] = {"n": len(rows)}
    for key, values in numeric.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            out[key] = None
            continue
        stderr = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
        out[key] = {"mean": round(float(arr.mean()), 4), "stderr": round(stderr, 4)}
    return out
