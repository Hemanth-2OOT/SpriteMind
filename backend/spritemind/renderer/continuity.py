"""Per-frame-pair continuity for animation frames.

The reported defect was "missing pixels from one transition to the next": a limb
that exists on frame N vanishes on frame N+1 because a part offset landed one grid
step too far. Island counting cannot see that, because both frames can be a single
connected blob. This module compares opaque masks of consecutive frames and reports
how much of the silhouette is lost, gained, and moved.
"""
from typing import Any, Dict, List, Sequence

import numpy as np


def opaque_mask(rgba: np.ndarray) -> np.ndarray:
    return rgba[..., 3] > 0


def pair_delta(previous: np.ndarray, current: np.ndarray) -> Dict[str, float]:
    """Statistics for one frame pair; `lost` is what the user perceives as a hole."""
    prev, cur = previous.astype(bool), current.astype(bool)
    union = int((prev | cur).sum())
    lost = int((prev & ~cur).sum())
    gained = int((cur & ~prev).sum())
    return {
        "iou": round(float((prev & cur).sum() / union), 4) if union else 1.0,
        "lost_pixels": lost,
        "gained_pixels": gained,
        "lost_share": round(lost / max(1, int(prev.sum())), 4),
        "changed_share": round((lost + gained) / max(1, union), 4),
    }


def continuity_report(masks: Sequence[np.ndarray]) -> Dict[str, Any]:
    """Aggregates consecutive-frame deltas, including the worst pair.

    A one-frame animation has nothing to compare, so it reports `pairs: 0` rather
    than pretending to be perfectly continuous.
    """
    pairs = [
        {"from_frame": index, "to_frame": index + 1, **pair_delta(masks[index], masks[index + 1])}
        for index in range(len(masks) - 1)
    ]
    if not pairs:
        return {"pairs": 0, "mean_iou": None, "worst_iou": None, "worst_pair": None, "per_pair": []}

    ious = sorted(p["iou"] for p in pairs)
    worst = min(pairs, key=lambda p: p["iou"])
    return {
        "pairs": len(pairs),
        "mean_iou": round(float(np.mean(ious)), 4),
        "median_iou": round(float(np.median(ious)), 4),
        "worst_iou": worst["iou"],
        "worst_pair": {"from_frame": worst["from_frame"], "to_frame": worst["to_frame"]},
        "mean_lost_share": round(float(np.mean([p["lost_share"] for p in pairs])), 4),
        "total_lost_pixels": int(sum(p["lost_pixels"] for p in pairs)),
        "per_pair": pairs,
    }


def masks_from_frames(frames: Sequence[Dict[str, Any]], decode) -> List[np.ndarray]:
    """Decodes base64 frames from a pipeline/animation result into opaque masks."""
    return [opaque_mask(decode(frame)) for frame in frames]
