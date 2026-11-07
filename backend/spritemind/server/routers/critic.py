"""Visual critic inspection and autonomous repair."""
from typing import Any, Dict
from fastapi import APIRouter, HTTPException

from pathlib import Path
from spritemind.critic.model import DEFECT_CLASSES
from spritemind.pixelscript.parser import PixelScriptParser
from spritemind.renderer.engine import PixelScriptEngine
from spritemind.server import history
from spritemind.server.schemas import CriticRepairRequest
from spritemind.server.schemas import InspectRequest
import json

from spritemind.server.runtime import repair_engine
router = APIRouter()

@router.post("/api/critic/inspect")
def inspect_sprite(req: InspectRequest) -> Dict[str, Any]:
    try:
        pal = req.palette_name or req.palette
        res = PixelScriptEngine.execute(
            req.script_text,
            width=req.width,
            height=req.height,
            palette_name=pal,
        )
        prog = PixelScriptParser.parse_text(req.script_text)
        report = repair_engine.inspect(res, category=req.category, palette_name=pal, prog=prog)
        report["image_base64"] = res.to_dict()["image_base64"]
        return report
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/api/critic/repair")
def repair_sprite(req: CriticRepairRequest) -> Dict[str, Any]:
    try:
        pal = req.palette_name or req.palette
        if req.auto_refine:
            res = repair_engine.refine_closed_loop(
                req.script_text,
                category=req.category,
                palette_name=pal,
                width=req.width,
                height=req.height,
                max_iterations=req.max_iterations,
            )
            return {
                "initial_quality": res["initial_quality"],
                "final_quality": res["final_quality"],
                "quality_gain": res["quality_gain"],
                "initial_defects": res["initial_defects"],
                "final_defects": res["final_defects"],
                "resolved_defects": res["resolved_defects"],
                "iterations_completed": res["iterations_completed"],
                "applied_repairs": res["applied_repairs"],
                "repaired_script": res["final_frame"]["script_text"],
                "image_base64": res["final_frame"]["image_base64"],
                "history": res["history"],
                "final_critic": res["final_critic"],
                "elapsed_ms": res["elapsed_ms"],
            }
        elif req.action_patch:
            # Single patch application
            candidate_script = repair_engine.apply_patch_to_script(req.script_text, req.action_patch)
            cand_res = PixelScriptEngine.execute(candidate_script, width=req.width, height=req.height, palette_name=pal)
            cand_prog = PixelScriptParser.parse_text(candidate_script)
            cand_report = repair_engine.inspect(cand_res, category=req.category, palette_name=pal, prog=cand_prog)
            return {
                "repaired_script": candidate_script,
                "image_base64": cand_res.to_dict()["image_base64"],
                "final_critic": cand_report,
                "quality_score": cand_report["quality_score"],
                "defects_count": cand_report["defects_count"],
            }
        else:
            raise HTTPException(status_code=400, detail="Either auto_refine=True or action_patch must be provided.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/api/critic/metrics")
def get_critic_metrics() -> Dict[str, Any]:
    res = {
        "status": "ready",
        "model_architecture": "NeuralVisualCritic (Residual2D + Heatmap Decoder)",
        "parameter_count": 368044,
        "classes": DEFECT_CLASSES,
    }
    metrics_path = Path("data/critic_metrics.json")
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                res.update(json.load(f))
        except Exception:
            pass

    eval_path = Path("data/benchmarks/critic_eval_results.json")
    if eval_path.exists():
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                res["evaluation_summary"] = json.load(f).get("summary", {})
        except Exception:
            pass

    return res
