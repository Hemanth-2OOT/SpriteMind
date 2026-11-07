"""Health, model selection, generation and prompt history."""
from typing import Any, Dict
from fastapi import APIRouter, HTTPException

from spritemind.ml.checkpoint_manifest import list_entries as list_checkpoint_entries
from spritemind.ml.checkpoint_manifest import verify as verify_checkpoint
from spritemind.ml.temporal import TemporalAnimator
from spritemind.server import history
from spritemind.server.metrics import record_generation
from spritemind.server.schemas import CheckpointLoadRequest
from spritemind.server.schemas import GenerateRequest
import torch

from spritemind.server.runtime import LATENCY, pipeline, trainer
router = APIRouter()

@router.get("/api/health")
def get_health() -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    device_memory = {}
    if cuda_available:
        device_memory = {
            "allocated_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024, 2),
            "reserved_mb": round(torch.cuda.memory_reserved(0) / 1024 / 1024, 2),
            "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 2),
        }

    return {
        "status": "healthy",
        "version": "0.1.0",
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device": device_name,
        "device_memory": device_memory,
        "training_active": trainer.is_training,
        "result_cache": pipeline.cache_stats(),
    }

@router.get("/api/models")
def get_models() -> Dict[str, Any]:
    checkpoints = [p.name for p in pipeline.checkpoints_dir.glob("*.pt")]
    model_info = pipeline.get_loaded_model_info()
    recorded = {entry["file"]: entry for entry in list_checkpoint_entries(pipeline.checkpoints_dir)}
    integrity = {
        name: verify_checkpoint(pipeline.checkpoints_dir, name) for name in checkpoints
    }
    return {
        "available_checkpoints": checkpoints,
        "loaded_model": model_info,
        "supported_engines": ["procedural_baseline", "untrained_model", "trained_model"],
        "checkpoints": [
            {
                "file": name,
                "recorded": name in recorded,
                "parameter_count": recorded.get(name, {}).get("parameter_count"),
                "trained_animations": recorded.get(name, {}).get("trained_animations"),
                "integrity": integrity[name]["reason"],
                "ok": integrity[name]["ok"],
            }
            for name in checkpoints
        ],
        "temporal_checkpoint_status": TemporalAnimator.last_load_status,
    }

@router.post("/api/checkpoints/load")
def load_checkpoint(req: CheckpointLoadRequest) -> Dict[str, Any]:
    success = pipeline.load_checkpoint(req.checkpoint_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Checkpoint '{req.checkpoint_name}' not found")
    return {
        "status": "success",
        "loaded_checkpoint": req.checkpoint_name,
        "model_info": pipeline.get_loaded_model_info(),
    }

@router.get("/api/metrics")
def get_metrics() -> Dict[str, Any]:
    """In-process latency percentiles plus pipeline cache counters."""
    return {
        "latency": LATENCY.to_dict(),
        "cache": {
            "hits": pipeline.cache_hits,
            "misses": pipeline.cache_misses,
            "size": pipeline._result_cache_size,
        },
    }

@router.post("/api/generate")
def generate(req: GenerateRequest) -> Dict[str, Any]:
    result = pipeline.generate(
        prompt=req.prompt or "",
        category=req.category,
        style=req.style,
        pose=req.pose,
        palette_name=req.palette,
        width=req.width,
        height=req.height,
        num_frames=req.num_frames,
        engine_mode=req.engine,
        seed=req.seed,
        max_actions=req.max_actions,
        best_of=req.best_of,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    record_generation(result, req.engine)
    try:
        history.record_run(req.model_dump(), result)
    except Exception:  # history must never fail a generation
        pass
    return result

@router.get("/api/history")
def get_history(limit: int = 50) -> Dict[str, Any]:
    runs = history.list_runs(limit=limit)
    return {"runs": runs, "total": len(runs)}

@router.delete("/api/history/{run_id}")
def delete_history_run(run_id: int) -> Dict[str, Any]:
    return {"deleted": history.delete_run(run_id), "id": run_id}

@router.delete("/api/history")
def clear_history() -> Dict[str, Any]:
    return {"deleted": history.clear_runs()}
