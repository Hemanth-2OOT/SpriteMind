"""PixelScript text and JSON parser."""
import re
from typing import List, Union, Dict, Any
from spritemind.pixelscript.ast import ActionType, PixelAction, PixelScriptProgram


class PixelScriptParseError(Exception):
    def __init__(self, message: str, line_number: int = None, line_text: str = None):
        super().__init__(f"Line {line_number or '?'}: {message} (text: '{line_text or ''}')")
        self.line_number = line_number
        self.line_text = line_text


class PixelScriptParser:
    """Parses text DSL or JSON representations of PixelScript."""

    @classmethod
    def parse_text(cls, script_text: str) -> PixelScriptProgram:
        lines = script_text.splitlines()
        actions: List[PixelAction] = []
        metadata: Dict[str, Any] = {}

        for line_no, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                if line.startswith("# META:"):
                    # e.g. # META: width=32 height=32 name=knight
                    parts = line[7:].strip().split()
                    for p in parts:
                        if "=" in p:
                            k, v = p.split("=", 1)
                            metadata[k] = v
                continue

            tokens = line.split()
            cmd = tokens[0].upper()

            try:
                action = cls._parse_tokens(cmd, tokens[1:], line_no, line)
                actions.append(action)
            except Exception as e:
                raise PixelScriptParseError(str(e), line_number=line_no, line_text=line)

        return PixelScriptProgram(actions=actions, metadata=metadata)

    @classmethod
    def _parse_tokens(cls, cmd: str, args: List[str], line_no: int, raw_text: str) -> PixelAction:
        if cmd == ActionType.PALETTE.value:
            # PALETTE <idx> <color_hex>
            if len(args) < 2:
                raise ValueError("PALETTE requires <index> <hex_or_rgba>")
            idx = int(args[0])
            color = args[1]
            return PixelAction(ActionType.PALETTE, {"index": idx, "color": color}, raw_text, line_no)

        elif cmd == ActionType.LAYER.value:
            # LAYER <id> [name] [opacity]
            if len(args) < 1:
                raise ValueError("LAYER requires <id> [name] [opacity]")
            layer_id = int(args[0])
            name = args[1] if len(args) > 1 else f"layer_{layer_id}"
            opacity = float(args[2]) if len(args) > 2 else 1.0
            return PixelAction(ActionType.LAYER, {"id": layer_id, "name": name, "opacity": opacity}, raw_text, line_no)

        elif cmd == ActionType.PART.value:
            # PART <name> [parent] [anchor_x] [anchor_y]
            if len(args) < 1:
                raise ValueError("PART requires <name> [parent] [anchor_x] [anchor_y]")
            name = args[0]
            parent = args[1] if len(args) > 1 and args[1].lower() != "none" else None
            ax = int(args[2]) if len(args) > 2 else 0
            ay = int(args[3]) if len(args) > 3 else 0
            return PixelAction(ActionType.PART, {"name": name, "parent": parent, "anchor_x": ax, "anchor_y": ay}, raw_text, line_no)

        elif cmd == ActionType.PIXEL.value:
            # PIXEL <x> <y> <color>
            if len(args) < 3:
                raise ValueError("PIXEL requires <x> <y> <color>")
            x, y = int(args[0]), int(args[1])
            # color can be int index or hex string
            color = int(args[2]) if args[2].isdigit() or (args[2].startswith("-") and args[2][1:].isdigit()) else args[2]
            return PixelAction(ActionType.PIXEL, {"x": x, "y": y, "color": color}, raw_text, line_no)

        elif cmd == ActionType.RECT.value:
            # RECT <x> <y> <w> <h> <color> [filled: 1/0]
            if len(args) < 5:
                raise ValueError("RECT requires <x> <y> <w> <h> <color> [filled]")
            x, y = int(args[0]), int(args[1])
            w, h = int(args[2]), int(args[3])
            color = int(args[4]) if args[4].isdigit() else args[4]
            filled = bool(int(args[5])) if len(args) > 5 else True
            return PixelAction(ActionType.RECT, {"x": x, "y": y, "w": w, "h": h, "color": color, "filled": filled}, raw_text, line_no)

        elif cmd == ActionType.LINE.value:
            # LINE <x0> <y0> <x1> <y1> <color>
            if len(args) < 5:
                raise ValueError("LINE requires <x0> <y0> <x1> <y1> <color>")
            x0, y0 = int(args[0]), int(args[1])
            x1, y1 = int(args[2]), int(args[3])
            color = int(args[4]) if args[4].isdigit() else args[4]
            return PixelAction(ActionType.LINE, {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "color": color}, raw_text, line_no)

        elif cmd == ActionType.FILL.value:
            # FILL <x> <y> <color>
            if len(args) < 3:
                raise ValueError("FILL requires <x> <y> <color>")
            x, y = int(args[0]), int(args[1])
            color = int(args[2]) if args[2].isdigit() else args[2]
            return PixelAction(ActionType.FILL, {"x": x, "y": y, "color": color}, raw_text, line_no)

        elif cmd == ActionType.ERASE.value:
            # ERASE <x> <y> [w] [h]
            if len(args) < 2:
                raise ValueError("ERASE requires <x> <y> [w] [h]")
            x, y = int(args[0]), int(args[1])
            w = int(args[2]) if len(args) > 2 else 1
            h = int(args[3]) if len(args) > 3 else 1
            return PixelAction(ActionType.ERASE, {"x": x, "y": y, "w": w, "h": h}, raw_text, line_no)

        elif cmd == ActionType.REUSE.value:
            # REUSE <part_name> <dest_x> <dest_y> [flip_x: 0/1] [flip_y: 0/1]
            if len(args) < 3:
                raise ValueError("REUSE requires <part_name> <dest_x> <dest_y> [flip_x] [flip_y]")
            part_name = args[0]
            dest_x, dest_y = int(args[1]), int(args[2])
            flip_x = bool(int(args[3])) if len(args) > 3 else False
            flip_y = bool(int(args[4])) if len(args) > 4 else False
            return PixelAction(ActionType.REUSE, {"part_name": part_name, "dest_x": dest_x, "dest_y": dest_y, "flip_x": flip_x, "flip_y": flip_y}, raw_text, line_no)

        elif cmd == ActionType.TRANSFORM.value:
            # TRANSFORM <part_name> <dx> <dy> [rot_deg]
            if len(args) < 3:
                raise ValueError("TRANSFORM requires <part_name> <dx> <dy> [rot_deg]")
            part_name = args[0]
            dx, dy = int(args[1]), int(args[2])
            rot_deg = int(args[3]) if len(args) > 3 else 0
            return PixelAction(ActionType.TRANSFORM, {"part_name": part_name, "dx": dx, "dy": dy, "rot_deg": rot_deg}, raw_text, line_no)

        elif cmd == ActionType.OUTLINE.value:
            # OUTLINE <x> <y> <w> <h> <color>
            if len(args) < 5:
                raise ValueError("OUTLINE requires <x> <y> <w> <h> <color>")
            x, y = int(args[0]), int(args[1])
            w, h = int(args[2]), int(args[3])
            color = int(args[4]) if args[4].isdigit() else args[4]
            return PixelAction(ActionType.OUTLINE, {"x": x, "y": y, "w": w, "h": h, "color": color}, raw_text, line_no)

        elif cmd == ActionType.SHADE.value:
            # SHADE <x> <y> <w> <h> <color> <direction 0..3>
            if len(args) < 6:
                raise ValueError("SHADE requires <x> <y> <w> <h> <color> <direction>")
            x, y = int(args[0]), int(args[1])
            w, h = int(args[2]), int(args[3])
            color = int(args[4]) if args[4].isdigit() else args[4]
            direction = int(args[5]) % 4
            return PixelAction(ActionType.SHADE, {"x": x, "y": y, "w": w, "h": h, "color": color, "direction": direction}, raw_text, line_no)

        elif cmd == ActionType.END.value:
            # END [target]
            target = args[0] if len(args) > 0 else "PROGRAM"
            return PixelAction(ActionType.END, {"target": target}, raw_text, line_no)

        else:
            raise ValueError(f"Unknown action command '{cmd}'")

    @classmethod
    def parse_dict(cls, data: Dict[str, Any]) -> PixelScriptProgram:
        return PixelScriptProgram.from_dict(data)
