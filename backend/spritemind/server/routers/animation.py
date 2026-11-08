"""Temporal animation generation and per-frame regeneration."""
from typing import Any, Dict
from fastapi import APIRouter, HTTPException

from pathlib import Path
from spritemind.ml.temporal import ANIMATION_CATEGORIES
from spritemind.ml.temporal import CATEGORIES
from spritemind.ml.temporal import TemporalAnimator
from spritemind.server.schemas import AnimationGenerateRequest
from spritemind.server.schemas import RegenerateFrameRequest
from spritemind.paths import resolve_data_dir
import json
router = APIRouter()

@router.post("/api/animation/generate")
def generate_animation(req: AnimationGenerateRequest) -> Dict[str, Any]:
    try:
        result = TemporalAnimator.generate_animation(
            base_script_or_prompt=req.base_script or req.prompt or "",
            category=req.category or "knight",
            style=req.style or "classic_16bit",
            anim_category=req.animation_category,
            palette_name=req.palette or "pico8",
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            fps=req.fps,
            engine_mode=req.engine,
            seed=req.seed,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Animation generation failed: {str(e)}")

@router.post("/api/animation/regenerate_frame")
def regenerate_frame(req: RegenerateFrameRequest) -> Dict[str, Any]:
    try:
        full_anim = TemporalAnimator.generate_animation(
            base_script_or_prompt=req.base_script,
            category=req.category or "knight",
            anim_category=req.animation_category,
            palette_name=req.palette or "pico8",
            num_frames=req.total_frames,
            engine_mode=req.engine,
            seed=req.seed,
        )
        frames = full_anim.get("frames", [])
        if 0 <= req.frame_index < len(frames):
            target_frame = frames[req.frame_index]
        else:
            target_frame = frames[0] if frames else {}

        return {
            "status": "success",
            "frame_index": req.frame_index,
            "engine_used": full_anim.get("engine_used"),
            "frame": target_frame,
            "animation_category": full_anim.get("animation_category", req.animation_category),
            "identity_confidence": full_anim.get("identity_confidence"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Frame regeneration failed: {str(e)}")

@router.get("/api/animation/metrics")
def get_animation_metrics() -> Dict[str, Any]:
    res = {
        "status": "ready",
        "model_architecture": "TemporalSVAN (Visual Residual + Category/Anim Embeddings + GRU Recurrent Core)",
        "parameter_count": 426395,
        "supported_animation_categories": ANIMATION_CATEGORIES,
        "supported_character_categories": CATEGORIES,
    }
    metrics_path = resolve_data_dir() / "temporal_metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                res["training_metrics"] = json.load(f)
        except Exception:
            pass

    benchmarks_path = resolve_data_dir() / "benchmarks/temporal_benchmark_results.json"
    if benchmarks_path.exists():
        try:
            with open(benchmarks_path, "r", encoding="utf-8") as f:
                res["benchmark_results"] = json.load(f)
        except Exception:
            pass

    return res
