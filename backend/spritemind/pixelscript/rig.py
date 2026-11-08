"""Parent-relative attachment ("sockets") for PixelScript part transforms.

PixelScript's ``TRANSFORM part dx dy rot`` moves a part by an absolute delta.
For a limb that is fine, but an equipped weapon needs to stay in the hand: its
position is a consequence of where the arm ended up, not an independent number.

This module composes attachment offsets *before* the frame script is emitted, so
the grammar, the renderer and every existing absolute-transform script stay
untouched. Only parts listed in ``ATTACHMENTS`` are recomposed; body parts keep
their authored motion.

Every other part that declares a parent (``PART arm_left torso ...``) is kept
touching that parent: if the authored motion would pull the two apart by more
than ``JOINT_MAX_GAP`` pixels, the child is translated back just enough to stay
attached. This is the limb joint constraint — pixel art has no visible joints, so
a limb that drifts off the torso reads as a detached sprite.

Resolution per attachment::

    hand'   = R(rot_parent) * (hand - anchor_parent) + anchor_parent + delta_parent
    rot_out = rot_parent + rot_authored       (or just rot_authored)
    delta   = hand' - anchor_part - R(rot_out) * (grip - anchor_part)

where ``grip``/``hand`` is the closest pixel pair between the part and its parent
in the rest pose, i.e. the actual point of contact the character was drawn with.
"""
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from spritemind.renderer.engine import PixelScriptEngine

Offset = Tuple[int, int, int]

# Which parts are attached to a parent, and whether they spin with it.
# A part always follows its parent's translation. Rotation inheritance is opt-in:
# the motion tables already author a weapon swing (e.g. 55 deg at the melee peak),
# so adding the arm's rotation on top would double it (95 deg). Cloth must not
# rotate with the torso at all.
ATTACHMENTS: Dict[str, Dict[str, object]] = {
    "weapon": {"parent": "arm_right", "inherit_rotation": False},
    "accessory": {"parent": "torso", "inherit_rotation": False},
}

# A body part may drift this many pixels beyond the gap it was drawn with before
# the joint constraint pulls it back. Restoring the drawn gap (usually 1 px, i.e.
# adjacent pixels) is what keeps a limb looking attached.
JOINT_STRAY_SLACK = 0.25


def part_parent_map(geometry: Optional["RigGeometry"] = None) -> Dict[str, str]:
    """Part -> parent. Uses the sprite's own PART declarations when available."""
    if geometry is not None:
        return {child: info[2] for child, info in geometry.sockets.items()}
    return {name: str(cfg["parent"]) for name, cfg in ATTACHMENTS.items()}


def _anchor(pixels, anchor_x: int, anchor_y: int) -> Tuple[int, int]:
    if anchor_x != 0 or anchor_y != 0:
        return anchor_x, anchor_y
    if not pixels:
        return 0, 0
    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    return min(xs), min(ys)


def _rotate(px: float, py: float, deg: float) -> Tuple[float, float]:
    if not deg:
        return px, py
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return px * c - py * s, px * s + py * c


def _closest_pair(a, b) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    best = None
    best_d = 10 ** 9
    for ax, ay in a:
        for bx, by in b:
            d = (ax - bx) ** 2 + (ay - by) ** 2
            if d < best_d:
                best_d = d
                best = ((ax, ay), (bx, by))
    return best


@dataclass
class RigGeometry:
    """Rest-pose geometry needed to resolve sockets for one base sprite."""

    anchors: Dict[str, Tuple[int, int]]
    # child -> (grip point, parent contact point, parent name)
    sockets: Dict[str, Tuple[Tuple[int, int], Tuple[int, int], str]]
    # body-part joints: child -> (contact on child, contact on parent, parent name)
    joints: Dict[str, Tuple[Tuple[int, int], Tuple[int, int], str]]
    # child -> parent, for hierarchy-ordered joint resolution
    hierarchy: Dict[str, str]

    @classmethod
    def from_script(
        cls, script: str, width: int = 32, height: int = 32, palette_name: str = "pico8"
    ) -> "RigGeometry":
        res = PixelScriptEngine.execute(script, width=width, height=height, palette_name=palette_name)
        parts = res.canvas.parts
        anchors: Dict[str, Tuple[int, int]] = {}
        for name, part in parts.items():
            anchors[name] = _anchor(part.pixels, part.anchor_x, part.anchor_y)

        joints: Dict[str, Tuple[Tuple[int, int], Tuple[int, int], str]] = {}
        hierarchy: Dict[str, str] = {}
        for name, part in parts.items():
            parent = part.parent
            if not parent or parent == "none" or parent not in parts:
                continue
            hierarchy[name] = parent

        sockets: Dict[str, Tuple[Tuple[int, int], Tuple[int, int], str]] = {}
        for child, cfg in ATTACHMENTS.items():
            if child not in parts:
                continue
            declared = parts[child].parent
            if declared and declared != "none" and declared in parts:
                parent = declared  # the sprite itself declares the hierarchy
            else:
                parent = str(cfg["parent"])
            if parent not in parts:
                continue
            pair = _closest_pair(parts[child].pixels, parts[parent].pixels)
            if pair is not None:
                sockets[child] = (pair[0], pair[1], parent)

        for child, parent in hierarchy.items():
            if child in ATTACHMENTS:
                continue
            pair = _closest_pair(parts[child].pixels, parts[parent].pixels)
            if pair is not None:
                joints[child] = (pair[0], pair[1], parent)
        return cls(anchors=anchors, sockets=sockets, joints=joints, hierarchy=hierarchy)

    def _depth(self, part: str) -> int:
        depth = 0
        seen = set()
        while part in self.hierarchy and part not in seen:
            seen.add(part)
            part = self.hierarchy[part]
            depth += 1
        return depth

    def _snap_joints(self, out: Dict[str, Offset]) -> None:
        """Pull a body part back when it drifts off its parent.

        The child keeps the offset vector it was drawn with; only drift beyond
        ``JOINT_STRAY_SLACK`` is removed, so a rigid parent translation still
        moves the limb by exactly the same amount.
        """
        for child in sorted(self.joints, key=self._depth):
            grip, socket, parent = self.joints[child]
            anchor_c = self.anchors.get(child, (0, 0))
            anchor_p = self.anchors.get(parent, (0, 0))
            pdx, pdy, prot = out.get(parent, (0, 0, 0))
            cdx, cdy, crot = out.get(child, (0, 0, 0))

            sx, sy = _rotate(socket[0] - anchor_p[0], socket[1] - anchor_p[1], prot)
            parent_x = sx + anchor_p[0] + pdx
            parent_y = sy + anchor_p[1] + pdy

            rx, ry = _rotate(grip[0] - socket[0], grip[1] - socket[1], prot)
            target_x = parent_x + rx
            target_y = parent_y + ry

            gx, gy = _rotate(grip[0] - anchor_c[0], grip[1] - anchor_c[1], crot)
            child_x = gx + anchor_c[0] + cdx
            child_y = gy + anchor_c[1] + cdy

            ex = target_x - child_x
            ey = target_y - child_y
            err = math.hypot(ex, ey)
            if err <= JOINT_STRAY_SLACK or err == 0:
                continue
            keep = JOINT_STRAY_SLACK / err
            out[child] = (
                int(round(cdx + ex * (1.0 - keep))),
                int(round(cdy + ey * (1.0 - keep))),
                crot,
            )

    def resolve(self, offsets: Dict[str, Offset]) -> Dict[str, Offset]:
        """Returns offsets with joints kept in contact, then attachments socketed.

        Joints are resolved first: a weapon is socketed to where the arm ended up,
        so the body has to be settled before the equipment follows it.
        """
        out = dict(offsets)
        self._snap_joints(out)
        for child, cfg in ATTACHMENTS.items():
            if child not in self.sockets:
                continue
            grip, hand, parent = self.sockets[child]
            anchor_p = self.anchors.get(parent, (0, 0))
            anchor_c = self.anchors.get(child, (0, 0))

            pdx, pdy, prot = out.get(parent, (0, 0, 0))
            cdx, cdy, crot = out.get(child, (0, 0, 0))

            rx, ry = _rotate(hand[0] - anchor_p[0], hand[1] - anchor_p[1], prot)
            hand_x = rx + anchor_p[0] + pdx
            hand_y = ry + anchor_p[1] + pdy

            rot_out = int(round(prot + crot)) if cfg["inherit_rotation"] else crot
            gx, gy = _rotate(grip[0] - anchor_c[0], grip[1] - anchor_c[1], rot_out)

            dx = int(round(hand_x - anchor_c[0] - gx))
            dy = int(round(hand_y - anchor_c[1] - gy))
            out[child] = (dx, dy, rot_out)
        return out


def compose_socketed_offsets(
    base_script: str,
    offsets: Dict[str, Offset],
    width: int = 32,
    height: int = 32,
    palette_name: str = "pico8",
    geometry: Optional[RigGeometry] = None,
) -> Dict[str, Offset]:
    """Convenience wrapper: resolve sockets for one frame without reusing geometry."""
    geo = geometry or RigGeometry.from_script(base_script, width, height, palette_name)
    return geo.resolve(offsets)


__all__ = [
    "ATTACHMENTS",
    "JOINT_STRAY_SLACK",
    "RigGeometry",
    "compose_socketed_offsets",
    "part_parent_map",
]
