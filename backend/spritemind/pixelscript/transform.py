"""Horizontal mirroring for PixelScript programs.

Sprites are authored facing right (or front). A left-facing variant is the same
program with every x coordinate reflected across the canvas, which keeps the
result editable and lets REUSE/TRANSFORM pairs stay consistent.
"""
from typing import List

from spritemind.pixelscript.ast import ActionType, PixelAction, PixelScriptProgram
from spritemind.pixelscript.parser import PixelScriptParser
from spritemind.pixelscript.serializer import PixelScriptSerializer


def _flip_x(x: int, width: int) -> int:
    return width - 1 - x


def _flip_span(x: int, length: int, width: int) -> int:
    return width - (x + length)


def mirror_action(action: PixelAction, width: int) -> PixelAction:
    params = dict(action.params)
    kind = action.action_type

    if kind == ActionType.PIXEL:
        params["x"] = _flip_x(params["x"], width)
    elif kind == ActionType.RECT:
        params["x"] = _flip_span(params["x"], params["w"], width)
    elif kind == ActionType.LINE:
        params["x0"] = _flip_x(params["x0"], width)
        params["x1"] = _flip_x(params["x1"], width)
    elif kind == ActionType.FILL:
        params["x"] = _flip_x(params["x"], width)
    elif kind == ActionType.ERASE:
        params["x"] = _flip_span(params["x"], params["w"], width)
    elif kind == ActionType.REUSE:
        params["dest_x"] = _flip_x(params["dest_x"], width)
        params["flip_x"] = not params.get("flip_x", False)
    elif kind == ActionType.PART:
        params["anchor_x"] = _flip_x(params["anchor_x"], width)
    elif kind == ActionType.TRANSFORM:
        params["dx"] = -params.get("dx", 0)
        if params.get("rot_deg"):
            params["rot_deg"] = -params["rot_deg"]

    return PixelAction(kind, params)


def mirror_script(script_text: str, width: int) -> str:
    """Reflect a PixelScript program horizontally across a canvas of `width`."""
    program = PixelScriptParser.parse_text(script_text)
    mirrored = [mirror_action(action, width) for action in program.actions]
    return PixelScriptSerializer.serialize_program(
        PixelScriptProgram(actions=mirrored, metadata=dict(program.metadata))
    )


def mirror_actions(actions: List[PixelAction], width: int) -> List[PixelAction]:
    return [mirror_action(action, width) for action in actions]
