"""Single-sprite actions: action-set planning, refine, render, export."""
from typing import Any, Dict
from fastapi import APIRouter, HTTPException

from spritemind.actions import ActionSetBuilder
from spritemind.actions import ActionSetPlanner
from spritemind.pixelscript.parser import PixelScriptParser
from spritemind.renderer.engine import PixelScriptEngine
from spritemind.renderer.export import array_to_png_bytes
from spritemind.renderer.export import compose_onion_skin
from spritemind.renderer.export import export_gif
from spritemind.renderer.export import export_spritesheet
from spritemind.server.schemas import ActionSetRequest
from spritemind.server.schemas import ExportRequest
from spritemind.server.schemas import RefineRequest
from spritemind.server.schemas import RenderRequest
import base64
import io
import json

from spritemind.server.runtime import repair_engine
router = APIRouter()

@router.post("/api/action_set/plan")
def action_set_plan(req: ActionSetRequest) -> Dict[str, Any]:
    """Cheap prompt interpretation: which animations the prompt implies."""
    plan = ActionSetPlanner.plan(
        prompt=req.prompt,
        category=req.category,
        style=req.style,
        palette=req.palette,
        width=req.width,
        height=req.height,
        requested_actions=req.actions,
        max_actions=req.max_actions,
        seed=req.seed,
    )
    return {"status": "success", "plan": plan.to_dict()}

@router.post("/api/action_set")
def action_set(req: ActionSetRequest) -> Dict[str, Any]:
    """Builds a coherent, game-ready animation set from a single prompt."""
    try:
        return ActionSetBuilder.build(
            prompt=req.prompt,
            category=req.category,
            style=req.style,
            palette=req.palette,
            width=req.width,
            height=req.height,
            engine=req.engine,
            seed=req.seed,
            actions=req.actions,
            max_actions=req.max_actions,
            include_gif=req.include_gif,
            facing=req.facing,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/api/refine")
def refine(req: RefineRequest) -> Dict[str, Any]:
    """Applies suggested fix or appends action, then re-renders."""
    script = req.script_text
    if req.suggested_fix:
        script = repair_engine.apply_patch_to_script(script, req.suggested_fix)

    res = PixelScriptEngine.execute(script, width=req.width, height=req.height, palette_name=req.palette)
    frame_dict = res.to_dict(include_passes=True)
    frame_dict["script_text"] = script
    prog = PixelScriptParser.parse_text(script)
    critic = repair_engine.inspect(res, category="knight", palette_name=req.palette, prog=prog)

    return {
        "status": "refined",
        "frame": frame_dict,
        "critic": critic,
    }

@router.post("/api/render")
def render(req: RenderRequest) -> Dict[str, Any]:
    try:
        res = PixelScriptEngine.execute(
            req.script_text,
            width=req.width,
            height=req.height,
            palette_name=req.palette,
        )
        return res.to_dict(include_passes=True, scale=req.scale)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/api/export")
def export_assets(req: ExportRequest):
    frames_arr = []
    for s in req.frames_scripts:
        res = PixelScriptEngine.execute(s, width=req.width, height=req.height, palette_name=req.palette)
        frames_arr.append(res.composite_array)

    if not frames_arr:
        raise HTTPException(status_code=400, detail="No valid frames provided for export")

    if req.format == "spritesheet":
        png_bytes, meta = export_spritesheet(frames_arr, cols=req.cols, scale=req.scale)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        return {
            "format": "spritesheet",
            "image_base64": f"data:image/png;base64,{b64}",
            "metadata": meta,
        }
    elif req.format == "gif":
        gif_bytes = export_gif(frames_arr, fps=req.fps, scale=max(1, req.scale))
        b64 = base64.b64encode(gif_bytes).decode("utf-8")
        return {
            "format": "gif",
            "image_base64": f"data:image/gif;base64,{b64}",
            "fps": req.fps,
        }
    elif req.format == "onion":
        ghost = compose_onion_skin(frames_arr, req.frame_index, scale=max(1, req.scale))
        buf = io.BytesIO()
        ghost.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {
            "format": "onion",
            "image_base64": f"data:image/png;base64,{b64}",
            "frame_index": max(0, min(req.frame_index, len(frames_arr) - 1)),
            "total_frames": len(frames_arr),
        }
    elif req.format == "png":
        png_bytes = array_to_png_bytes(frames_arr[0], scale=req.scale)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        return {
            "format": "png",
            "image_base64": f"data:image/png;base64,{b64}",
        }
    else:
        # JSON
        return {
            "format": "json",
            "frames_scripts": req.frames_scripts,
            "width": req.width,
            "height": req.height,
        }
