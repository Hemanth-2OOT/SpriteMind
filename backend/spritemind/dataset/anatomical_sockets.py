"""Anatomical socket targets derived from drawn body-part geometry (R7).

The shipped SocketPredictor was trained on labels taken from the *rendered alpha
bounding box* (`head = (cx, min_y + 2)`, `hands = (±bbox_x, cy + 4)`), which is a
property of the silhouette's extent, not of the body. The audit measured the
result: 0.87 px mean error against those same bbox labels but 2.62 px against an
independent anatomical reference.

These labels instead use the part pixels the renderer already tracks, so a hand
socket sits on the distal end of the drawn arm and a head socket on the drawn
head, whatever the pose or proportion:

* ``head``   — centre of the ``head`` part's pixel bounding box (falls back to the
  part anchor, then to a fixed proportional guess).
* ``back``   — centre of the ``torso`` part's pixel bounding box.
* ``hand_r`` — distal end of ``arm_right``: the part pixel furthest along +x+y.
* ``hand_l`` — distal end of ``arm_left``: the part pixel furthest along +y-x.

Part pixels come from the renderer (`PixelCanvas.parts[name].pixels`), i.e. from
what was actually drawn, so the labels cannot drift from the sprite.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

SocketPoint = Tuple[int, int]
SocketTargets = Dict[str, SocketPoint]

SOCKET_RULE = (
    "head=torso-anchored drawn head centre; back=torso centre; "
    "hand_r/hand_l=distal pixel of the drawn arm parts; silhouette-bbox fallback only when a part is absent"
)


def _bounds(pixels: Iterable[Tuple[int, int]]) -> Optional[Tuple[int, int, int, int]]:
    points = list(pixels)
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _clamp(point: SocketPoint, width: int, height: int) -> SocketPoint:
    x = max(0, min(width - 1, int(round(point[0]))))
    y = max(0, min(height - 1, int(round(point[1]))))
    return x, y


def anatomical_sockets(canvas, width: int = 32, height: int = 32) -> SocketTargets:
    """Socket positions from the parts this canvas actually painted."""
    parts = getattr(canvas, "parts", {}) or {}

    def pixels(name: str):
        part = parts.get(name)
        return getattr(part, "pixels", set()) or set()

    head_px = _bounds(pixels("head"))
    torso_px = _bounds(pixels("torso"))
    arm_r = pixels("arm_right")
    arm_l = pixels("arm_left")

    if head_px:
        head = ((head_px[0] + head_px[2]) / 2.0, (head_px[1] + head_px[3]) / 2.0)
    elif "head" in parts:
        head = (parts["head"].anchor_x, parts["head"].anchor_y)
    else:
        head = (width * 0.5, height * 0.22)

    if torso_px:
        back = ((torso_px[0] + torso_px[2]) / 2.0, (torso_px[1] + torso_px[3]) / 2.0)
    elif "torso" in parts:
        back = (parts["torso"].anchor_x, parts["torso"].anchor_y)
    else:
        back = (width * 0.5, height * 0.5)

    if arm_r:
        hand_r = max(arm_r, key=lambda p: p[0] + p[1])
    elif "arm_right" in parts:
        hand_r = (parts["arm_right"].anchor_x, parts["arm_right"].anchor_y)
    else:
        hand_r = (width * 0.68, height * 0.65)

    if arm_l:
        hand_l = max(arm_l, key=lambda p: p[1] - p[0])
    elif "arm_left" in parts:
        hand_l = (parts["arm_left"].anchor_x, parts["arm_left"].anchor_y)
    else:
        hand_l = (width * 0.32, height * 0.65)

    return {
        "head": _clamp(head, width, height),
        "back": _clamp(back, width, height),
        "hand_r": _clamp(hand_r, width, height),
        "hand_l": _clamp(hand_l, width, height),
    }


def bbox_legacy_sockets(rgba, width: int = 32, height: int = 32) -> SocketTargets:
    """The old bbox-extent labels, kept so a benchmark can report both targets."""
    import numpy as np

    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return {"head": (16, 7), "back": (16, 16), "hand_r": (22, 20), "hand_l": (10, 20)}
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    cx = (min_x + max_x) // 2
    cy = (min_y + max_y) // 2
    return {
        "head": (cx, max(2, min_y + 2)),
        "back": (cx, cy),
        "hand_r": (min(width - 2, max_x), min(height - 2, cy + 4)),
        "hand_l": (max(1, min_x), min(height - 2, cy + 4)),
    }
