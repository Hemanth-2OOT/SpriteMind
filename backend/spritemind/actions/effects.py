"""Elemental effect overlays layered on top of an animated character program.

Effects are emitted as PixelScript lines appended to a frame script, so they go
through the same deterministic renderer as the character and stay editable.
"""
import math
from typing import List, Tuple

from spritemind.utils.palettes import PALETTES

ELEMENT_TARGET_RGB = {
    "fire": (255, 110, 30),
    "ice": (150, 220, 255),
    "lightning": (255, 240, 120),
    "holy": (255, 250, 205),
    "shadow": (120, 60, 160),
    "poison": (150, 230, 90),
    "arcane": (200, 140, 255),
    "none": (255, 255, 255),
}

FX_LAYER_ID = 3


def _color_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    return (a[0] - b[0]) ** 2 * 0.30 + (a[1] - b[1]) ** 2 * 0.59 + (a[2] - b[2]) ** 2 * 0.11


def element_colors(palette_name: str, element: str) -> Tuple[int, int, int]:
    """Returns (glow, core, spark) palette indices for an element."""
    palette = PALETTES.get(palette_name, PALETTES["pico8"])
    target = ELEMENT_TARGET_RGB.get(element, ELEMENT_TARGET_RGB["arcane"])

    opaque = [(i, c) for i, c in enumerate(palette) if i > 0 and c[3] > 0]
    if not opaque:
        return 1, 1, 1

    glow = min(opaque, key=lambda ic: _color_distance(ic[1][:3], target))[0]
    core = max(opaque, key=lambda ic: sum(ic[1][:3]))[0]
    spark = min(opaque, key=lambda ic: _color_distance(ic[1][:3], (255, 255, 200)))[0]
    return glow, core, spark


def _scaled(value: float, width: int) -> int:
    return int(round(value * width / 32.0))


def _orb(cx: int, cy: int, radius: int, glow: int, core: int) -> List[str]:
    lines = [f"PART fx_orb None {cx} {cy}"]
    lines.append(f"RECT {cx - radius} {cy - radius} {radius * 2 + 1} {radius * 2 + 1} {glow} 1")
    if radius >= 1:
        lines.append(f"RECT {cx - radius + 1} {cy - radius + 1} {radius * 2 - 1} {radius * 2 - 1} {core} 1")
    lines.append("END PART")
    return lines


def build_effects(
    effect: str,
    phase: float,
    palette_name: str = "pico8",
    width: int = 32,
    height: int = 32,
    element: str = "arcane",
) -> str:
    """Renders one frame of an effect as PixelScript lines (empty when inactive)."""
    if not effect or effect == "none":
        return ""

    glow, core, spark = element_colors(palette_name, element)
    lines: List[str] = [f"LAYER {FX_LAYER_ID} fx 1.00"]

    hand_x = _scaled(23.0, width)
    hand_y = _scaled(14.0, height)
    body_mid_y = _scaled(17.0, height)

    if effect == "orb_charge":
        if phase >= 0.25:
            grow = min(1.0, (phase - 0.25) / 0.45)
            cx, cy = hand_x, hand_y - _scaled(2.0, height)
            radius = 1 if grow < 0.7 else 2
            lines += _orb(cx, cy, radius, glow, core)
            if math.sin(phase * math.pi * 6) > 0:
                lines.append(f"PIXEL {cx - 3} {cy} {glow}")
                lines.append(f"PIXEL {cx + 3} {cy} {glow}")

    elif effect == "projectile":
        if phase >= 0.45:
            travel = min(1.0, (phase - 0.45) / 0.55)
            px = int(round(hand_x + travel * (width - 3 - hand_x)))
            py = int(round(hand_y - _scaled(2.0, height) + math.sin(travel * math.pi) * _scaled(1.5, height)))
            lines += _orb(px, py, 1, glow, core)
            if travel > 0.15:
                lines.append(f"PIXEL {px - 2} {py + 1} {glow}")
                lines.append(f"PIXEL {px - 3} {py} {glow}")

    elif effect == "slash_arc":
        if 0.45 <= phase <= 0.9:
            prog = (phase - 0.45) / 0.45
            sweep = int(round(prog * _scaled(6.0, width)))
            for k in range(4):
                ax = hand_x + k
                ay = hand_y - _scaled(4.0, height) + sweep + k
                if 0 <= ax < width and 0 <= ay < height:
                    lines.append(f"PIXEL {ax} {ay} {glow if k % 2 else spark}")

    elif effect == "impact":
        if phase >= 0.6:
            prog = min(1.0, (phase - 0.6) / 0.25)
            cx = min(width - 2, hand_x + _scaled(5.0, width))
            cy = hand_y
            r = 1 + int(round(prog * 2))
            for angle in range(0, 360, 45):
                px = cx + int(round(math.cos(math.radians(angle)) * r))
                py = cy + int(round(math.sin(math.radians(angle)) * r * 0.7))
                if 0 <= px < width and 0 <= py < height:
                    lines.append(f"PIXEL {px} {py} {spark}")

    elif effect == "dash_trail":
        if 0.15 <= phase <= 0.75:
            prog = (phase - 0.15) / 0.6
            length = int(round(_scaled(10.0, width) * (1.0 - abs(prog - 0.5) * 1.2)))
            for k, dy in enumerate((0, _scaled(3.0, height), _scaled(6.0, height))):
                x0 = max(1, hand_x - _scaled(6.0, width) - length)
                x1 = max(x0 + 1, hand_x - _scaled(6.0, width) - k * 2)
                y = body_mid_y - _scaled(3.0, height) + dy
                lines.append(f"LINE {x0} {y} {x1} {y} {glow if k == 0 else spark}")

    elif effect == "aura":
        pulse = 1 if math.sin(phase * math.pi * 2) > 0 else 0
        head_x = _scaled(16.0, width)
        head_y = _scaled(8.0, height) - pulse
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (-1, -1), (1, -1)):
            px, py = head_x + dx, head_y + dy
            if 0 <= px < width and 0 <= py < height:
                lines.append(f"PIXEL {px} {py} {glow}")

    if len(lines) <= 1:
        return ""
    return "\n".join(lines)
