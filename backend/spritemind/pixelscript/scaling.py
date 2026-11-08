"""Integer scaling of a finished PixelScript between canvas sizes.

The rig, the socket predictor and the frozen motion priors are all authored in
the 32 px design space. Frames are posed there and then redrawn at the requested
canvas size, so a 64 px animation is the same motion as the 32 px one at twice
the scale rather than a differently resolved approximation of it.
"""
import re
from typing import Iterable

DESIGN_SIZE = 32

_XY = (1, 2)
_BOX = (1, 2, 3, 4)
_SINGLE = (3, 4)


def _scale_numbers(parts, indices: Iterable[int], scale: int) -> None:
    for index in indices:
        try:
            parts[index] = str(int(round(float(parts[index]) * scale)))
        except (ValueError, IndexError):
            pass


def _scale_meta(line: str, scale: int) -> str:
    def bump(match: "re.Match") -> str:
        return f"{match.group(1)}={int(match.group(2)) * scale}"

    return re.sub(r"\b(width|height)=(\d+)", bump, line)


def scale_script(script_text: str, scale: int) -> str:
    """Scales every coordinate in a PixelScript by an integer factor."""
    if scale <= 1:
        return script_text
    out = []
    for line in script_text.splitlines():
        parts = line.split()
        if not parts:
            out.append(line)
            continue
        if parts[0] == "#" and len(parts) > 1 and parts[1].upper() == "META:":
            # The renderer takes the canvas size from the META header, so a
            # scaled copy has to declare the scaled canvas or it renders small.
            out.append(_scale_meta(line, scale))
            continue
        op = parts[0].upper()
        if op == "PIXEL":
            _scale_numbers(parts, _XY, scale)
        elif op in ("RECT", "ERASE", "LINE", "SHADE", "OUTLINE", "AURA"):
            _scale_numbers(parts, _BOX, scale)
        elif op == "PART":
            _scale_numbers(parts, _SINGLE, scale)
        elif op == "TRANSFORM":
            _scale_numbers(parts, _XY, scale)
        out.append(" ".join(parts))
    return "\n".join(out)
