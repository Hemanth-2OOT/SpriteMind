"""PixelScript Execution Engine."""
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from spritemind.pixelscript.ast import PixelAction, ActionType, PixelScriptProgram
from spritemind.pixelscript.parser import PixelScriptParser
from spritemind.pixelscript.validator import PixelScriptValidator, ValidationReport
from spritemind.renderer.canvas import PixelCanvas
from spritemind.renderer.passes import RenderPasses
from spritemind.renderer.export import array_to_base64


class RenderResult:
    def __init__(
        self,
        canvas: PixelCanvas,
        program: PixelScriptProgram,
        validation: ValidationReport,
        execution_time_ms: float = 0.0,
    ):
        self.canvas = canvas
        self.program = program
        self.validation = validation
        self.execution_time_ms = execution_time_ms

    @property
    def composite_array(self) -> np.ndarray:
        return self.canvas.composite()

    def to_dict(self, include_passes: bool = True, scale: int = 1) -> Dict[str, Any]:
        comp = self.composite_array
        out = {
            "width": self.canvas.width,
            "height": self.canvas.height,
            "image_base64": array_to_base64(comp, scale=scale),
            "layers": [l.to_dict() for l in self.canvas.layers.values()],
            "parts": {name: p.to_dict() for name, p in self.canvas.parts.items()},
            "palette": [list(c) for c in self.canvas.palette],
            "validation": self.validation.to_dict(),
            "execution_time_ms": round(self.execution_time_ms, 2),
            "pixel_count": int(np.count_nonzero(comp[:, :, 3] > 0)),
        }

        if include_passes:
            out["passes"] = {
                "silhouette": array_to_base64(RenderPasses.silhouette(self.canvas), scale=scale),
                "edge_map": array_to_base64(RenderPasses.edge_map(self.canvas), scale=scale),
                "segmentation": array_to_base64(RenderPasses.segmentation(self.canvas), scale=scale),
            }

        return out


class PixelScriptEngine:
    """Deterministic executor for PixelScript."""

    @classmethod
    def execute(
        cls,
        program_or_text: Any,
        width: int = 32,
        height: int = 32,
        palette_name: str = "pico8",
    ) -> RenderResult:
        import time
        start_time = time.perf_counter()

        if isinstance(program_or_text, str):
            program = PixelScriptParser.parse_text(program_or_text)
        elif isinstance(program_or_text, dict):
            program = PixelScriptParser.parse_dict(program_or_text)
        elif isinstance(program_or_text, PixelScriptProgram):
            program = program_or_text
        else:
            raise TypeError(f"Unsupported program type: {type(program_or_text)}")

        # Override width/height if specified in metadata
        if "width" in program.metadata:
            width = int(program.metadata["width"])
        if "height" in program.metadata:
            height = int(program.metadata["height"])
        if "palette" in program.metadata:
            palette_name = str(program.metadata["palette"])

        validation = PixelScriptValidator.validate(program, width, height)
        canvas = PixelCanvas(width=width, height=height, palette_name=palette_name)

        for action in program.actions:
            cls._execute_action(canvas, action)

        elapsed = (time.perf_counter() - start_time) * 1000.0
        return RenderResult(canvas, program, validation, execution_time_ms=elapsed)

    @classmethod
    def _execute_action(cls, canvas: PixelCanvas, action: PixelAction):
        t = action.action_type
        p = action.params

        if t == ActionType.PALETTE:
            color = p.get("color")
            if color is not None:
                canvas.set_palette_color(int(p["index"]), color)

        elif t == ActionType.LAYER:
            lid = int(p["id"])
            canvas.set_active_layer(lid)
            if "name" in p:
                canvas.active_layer.name = str(p["name"])
            if "opacity" in p:
                canvas.active_layer.opacity = float(p["opacity"])

        elif t == ActionType.PART:
            name = str(p["name"])
            parent = p.get("parent")
            ax = int(p.get("anchor_x", 0))
            ay = int(p.get("anchor_y", 0))
            canvas.define_part(name, parent, ax, ay)

        elif t == ActionType.PIXEL:
            canvas.set_pixel(int(p["x"]), int(p["y"]), p["color"])

        elif t == ActionType.RECT:
            canvas.draw_rect(
                int(p["x"]),
                int(p["y"]),
                int(p["w"]),
                int(p["h"]),
                p["color"],
                filled=bool(p.get("filled", True)),
            )

        elif t == ActionType.LINE:
            canvas.draw_line(int(p["x0"]), int(p["y0"]), int(p["x1"]), int(p["y1"]), p["color"])

        elif t == ActionType.FILL:
            canvas.flood_fill(int(p["x"]), int(p["y"]), p["color"])

        elif t == ActionType.ERASE:
            w = int(p.get("w", 1))
            h = int(p.get("h", 1))
            canvas.erase_rect(int(p["x"]), int(p["y"]), w, h)

        elif t == ActionType.REUSE:
            canvas.reuse_part(
                str(p["part_name"]),
                int(p["dest_x"]),
                int(p["dest_y"]),
                flip_x=bool(p.get("flip_x", False)),
                flip_y=bool(p.get("flip_y", False)),
            )

        elif t == ActionType.TRANSFORM:
            canvas.transform_part(
                str(p["part_name"]),
                int(p["dx"]),
                int(p["dy"]),
                rot_deg=int(p.get("rot_deg", 0)),
            )

        elif t == ActionType.OUTLINE:
            canvas.outline_rect(int(p["x"]), int(p["y"]), int(p["w"]), int(p["h"]), p["color"])

        elif t == ActionType.SHADE:
            canvas.shade_rect(
                int(p["x"]),
                int(p["y"]),
                int(p["w"]),
                int(p["h"]),
                p["color"],
                direction=int(p.get("direction", 1)),
            )

        elif t == ActionType.AURA:
            canvas.aura_rect(
                int(p["x"]),
                int(p["y"]),
                int(p["w"]),
                int(p["h"]),
                p["color"],
                density=int(p.get("density", 2)),
            )

        elif t == ActionType.END:
            if p.get("target") == "PART":
                canvas.end_part()
