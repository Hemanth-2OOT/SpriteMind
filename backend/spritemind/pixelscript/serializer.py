"""PixelScript serializer."""
from typing import List
from spritemind.pixelscript.ast import PixelAction, ActionType, PixelScriptProgram


class PixelScriptSerializer:
    """Serializes PixelScriptProgram or PixelAction objects into DSL text."""

    @classmethod
    def serialize_action(cls, action: PixelAction) -> str:
        t = action.action_type
        p = action.params

        if t == ActionType.PALETTE:
            return f"PALETTE {p.get('index', 0)} {p.get('color', '#000000')}"
        elif t == ActionType.LAYER:
            name = p.get("name", f"layer_{p.get('id', 0)}")
            opacity = p.get("opacity", 1.0)
            return f"LAYER {p.get('id', 0)} {name} {opacity:.2f}"
        elif t == ActionType.PART:
            name = p.get("name", "part")
            parent = p.get("parent") or "none"
            ax = p.get("anchor_x", 0)
            ay = p.get("anchor_y", 0)
            return f"PART {name} {parent} {ax} {ay}"
        elif t == ActionType.PIXEL:
            return f"PIXEL {p.get('x', 0)} {p.get('y', 0)} {p.get('color', 1)}"
        elif t == ActionType.RECT:
            filled = 1 if p.get("filled", True) else 0
            return f"RECT {p.get('x', 0)} {p.get('y', 0)} {p.get('w', 1)} {p.get('h', 1)} {p.get('color', 1)} {filled}"
        elif t == ActionType.LINE:
            return f"LINE {p.get('x0', 0)} {p.get('y0', 0)} {p.get('x1', 0)} {p.get('y1', 0)} {p.get('color', 1)}"
        elif t == ActionType.FILL:
            return f"FILL {p.get('x', 0)} {p.get('y', 0)} {p.get('color', 1)}"
        elif t == ActionType.ERASE:
            return f"ERASE {p.get('x', 0)} {p.get('y', 0)} {p.get('w', 1)} {p.get('h', 1)}"
        elif t == ActionType.REUSE:
            fx = 1 if p.get("flip_x", False) else 0
            fy = 1 if p.get("flip_y", False) else 0
            return f"REUSE {p.get('part_name', '')} {p.get('dest_x', 0)} {p.get('dest_y', 0)} {fx} {fy}"
        elif t == ActionType.TRANSFORM:
            return f"TRANSFORM {p.get('part_name', '')} {p.get('dx', 0)} {p.get('dy', 0)} {p.get('rot_deg', 0)}"
        elif t == ActionType.END:
            return f"END {p.get('target', 'PROGRAM')}"
        return f"# UNKNOWN {t}"

    @classmethod
    def serialize_program(cls, program: PixelScriptProgram) -> str:
        lines: List[str] = []
        if program.metadata:
            meta_str = " ".join(f"{k}={v}" for k, v in program.metadata.items())
            lines.append(f"# META: {meta_str}")
        for action in program.actions:
            lines.append(cls.serialize_action(action))
        return "\n".join(lines)
