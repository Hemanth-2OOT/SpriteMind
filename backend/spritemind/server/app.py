"""FastAPI server application for SpriteMind."""
import os
import io
import json
import base64
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from spritemind.server.schemas import (
    GenerateRequest,
    RefineRequest,
    RenderRequest,
    ExportRequest,
    DatasetGenerateRequest,
    TrainingStartRequest,
    CheckpointLoadRequest,
    AssetSaveRequest,
    InspectRequest,
    CriticRepairRequest,
    AnimationGenerateRequest,
    RegenerateFrameRequest,
    CompositionRequest,
    CompositionAnimationRequest,
    AssetPackExportRequest,
    TrashAssetRequest,
    RestoreAssetRequest,
    PermanentDeleteRequest,
    EmptyTrashRequest,
    CleanupExecuteRequest,
    BackupCreateRequest,
)
from spritemind.critic.repair_engine import AutonomousRepairEngine
from spritemind.critic.model import DEFECT_CLASSES
from spritemind.ml.pipeline import SpriteMindPipeline
from spritemind.ml.temporal import TemporalAnimator, ANIMATION_CATEGORIES, CATEGORIES
from spritemind.assets.library import AssetLibrary, ASSET_CATEGORIES
from spritemind.assets.composer import AssetComposer
from spritemind.assets.pack import AssetPack
from spritemind.storage.database import StorageDatabase
from spritemind.storage.lifecycle import AssetLifecycleManager
from spritemind.storage.cleanup import StorageCleaner
from spritemind.storage.backup import BackupManager
from spritemind.storage.classification import ProtectedResourceException
from spritemind.ml.training import ModelTrainer
from spritemind.ml.evaluation import ModelEvaluator
from spritemind.dataset.storage import DatasetManager
from spritemind.dataset.assets import AssetManager
from spritemind.renderer.engine import PixelScriptEngine
from spritemind.renderer.export import (
    array_to_png_bytes,
    array_to_base64,
    export_gif,
    export_spritesheet,
)
from spritemind.pixelscript.serializer import PixelScriptSerializer
from spritemind.pixelscript.parser import PixelScriptParser

# Initialize FastAPI
app = FastAPI(
    title="SpriteMind API",
    description="Local AI Pixel Art Generator & Research Platform",
    version="0.1.0",
)

DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:7860,http://127.0.0.1:7860,"
    "http://localhost:7861,http://127.0.0.1:7861"
)
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("SPRITEMIND_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core singletons
pipeline = SpriteMindPipeline()
trainer = ModelTrainer()
dataset_mgr = DatasetManager()
asset_mgr = AssetManager()
repair_engine = AutonomousRepairEngine()
storage_db = StorageDatabase()
lifecycle_mgr = AssetLifecycleManager(db=storage_db)
cleaner = StorageCleaner(db=storage_db)
backup_mgr = BackupManager(db=storage_db)


@app.get("/api/health")
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
    }


@app.get("/api/models")
def get_models() -> Dict[str, Any]:
    checkpoints = [p.name for p in pipeline.checkpoints_dir.glob("*.pt")]
    model_info = pipeline.get_loaded_model_info()
    return {
        "available_checkpoints": checkpoints,
        "loaded_model": model_info,
        "supported_engines": ["procedural_baseline", "untrained_model", "trained_model"],
    }


@app.post("/api/checkpoints/load")
def load_checkpoint(req: CheckpointLoadRequest) -> Dict[str, Any]:
    success = pipeline.load_checkpoint(req.checkpoint_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Checkpoint '{req.checkpoint_name}' not found")
    return {
        "status": "success",
        "loaded_checkpoint": req.checkpoint_name,
        "model_info": pipeline.get_loaded_model_info(),
    }


@app.post("/api/generate")
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
    )
    return result

@app.post("/api/refine")
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


@app.post("/api/render")
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


@app.post("/api/export")
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


# --- Dataset Endpoints ---
@app.get("/api/dataset/stats")
def get_dataset_stats(name: str = "default_synth") -> Dict[str, Any]:
    stats = dataset_mgr.get_dataset_stats(name)
    if stats is None:
        return {
            "exists": False,
            "name": name,
            "total_samples": 0,
            "splits": {"train": 0, "val": 0, "test": 0},
            "category_distribution": {},
            "defect_distribution": {},
        }
    return {**stats, "exists": True}


@app.get("/api/dataset/samples")
def get_dataset_samples(name: str = "default_synth", limit: int = 24, offset: int = 0) -> List[Dict[str, Any]]:
    return dataset_mgr.list_samples(name=name, limit=limit, offset=offset)


@app.post("/api/dataset/generate")
def generate_dataset(req: DatasetGenerateRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    summary = dataset_mgr.generate_dataset(
        name=req.name,
        total_samples=req.total_samples,
        defect_rate=req.defect_rate,
        seed=req.seed,
    )
    return {"status": "success", "summary": summary}


# --- Training Endpoints ---
@app.post("/api/training/start")
def start_training(req: TrainingStartRequest) -> Dict[str, Any]:
    res = trainer.start_training(
        dataset_name=req.dataset_name,
        epochs=req.epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
        optimizer_name=req.optimizer,
        save_checkpoint_name=req.checkpoint_name,
    )
    return res


@app.post("/api/training/stop")
def stop_training() -> Dict[str, Any]:
    return trainer.stop_training()


@app.get("/api/training/status")
def get_training_status() -> Dict[str, Any]:
    status = trainer.get_status()
    cuda_available = torch.cuda.is_available()
    status["device"] = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    return status


@app.get("/api/training/metrics")
def get_training_metrics() -> Dict[str, Any]:
    status = trainer.get_status()
    return {
        "loss_history": status.get("loss_history", []),
        "val_loss_history": status.get("val_loss_history", []),
        "current_loss": status.get("loss", 0.0),
        "current_val_loss": status.get("val_loss", 0.0),
        "epoch": status.get("epoch", 0),
        "total_epochs": status.get("total_epochs", 0),
    }


# --- Evaluation Endpoints ---
@app.get("/api/training/history")
def get_training_history() -> Dict[str, Any]:
    history_file = Path("data/training_history.json")
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    status = trainer.get_status()
    return {
        "dataset_name": status.get("dataset_name", "generalization_10k"),
        "total_epochs": status.get("total_epochs", 4),
        "completed_epochs": status.get("epoch", 0),
        "best_epoch": status.get("best_epoch", 0),
        "best_val_loss": status.get("best_val_loss", 0.0),
        "loss_history": status.get("loss_history", []),
        "val_loss_history": status.get("val_loss_history", []),
        "token_accuracy_history": status.get("token_accuracy_history", []),
    }


@app.get("/api/benchmarks/gallery")
def get_benchmark_gallery(limit: int = 100) -> Dict[str, Any]:
    results_path = Path("data/benchmarks/benchmark_results.json")
    if results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                gallery = data.get("gallery", [])[:limit]
                return {
                    "summary": data.get("summary", {}),
                    "gallery": gallery,
                    "total": len(gallery),
                }
        except Exception:
            pass
    from spritemind.ml.benchmark import BENCHMARK_PROMPTS
    return {
        "summary": {},
        "gallery": BENCHMARK_PROMPTS[:limit],
        "total": len(BENCHMARK_PROMPTS[:limit]),
    }


@app.post("/api/benchmarks/run")
def run_benchmark_endpoint(limit: int = 20) -> Dict[str, Any]:
    from spritemind.ml.benchmark import GeneralizationBenchmark
    bm = GeneralizationBenchmark()
    results = bm.run_benchmark(limit=limit)
    return {"status": "success", "summary": results.get("summary", {})}


@app.get("/api/evaluation/results")
def get_evaluation_results(num_samples: int = 5) -> Dict[str, Any]:
    return ModelEvaluator.compare_all(pipeline, num_samples=num_samples)


# --- Visual Critic & Autonomous Repair Endpoints ---
@app.post("/api/critic/inspect")
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


@app.post("/api/critic/repair")
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


@app.get("/api/critic/metrics")
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


# --- Animation & Temporal Intelligence Endpoints ---
@app.post("/api/animation/generate")
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


@app.post("/api/animation/regenerate_frame")
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
            "animation_category": req.animation_category,
            "identity_confidence": full_anim.get("identity_confidence"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Frame regeneration failed: {str(e)}")


@app.get("/api/animation/metrics")
def get_animation_metrics() -> Dict[str, Any]:
    res = {
        "status": "ready",
        "model_architecture": "TemporalSVAN (Visual Residual + Category/Anim Embeddings + GRU Recurrent Core)",
        "parameter_count": 426395,
        "supported_animation_categories": ANIMATION_CATEGORIES,
        "supported_character_categories": CATEGORIES,
    }
    metrics_path = Path("data/temporal_metrics.json")
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                res["training_metrics"] = json.load(f)
        except Exception:
            pass

    benchmarks_path = Path("data/benchmarks/temporal_benchmark_results.json")
    if benchmarks_path.exists():
        try:
            with open(benchmarks_path, "r", encoding="utf-8") as f:
                res["benchmark_results"] = json.load(f)
        except Exception:
            pass

    return res


# --- Stage 5: Asset Catalog, Composition & Packaging Endpoints ---
@app.get("/api/asset_catalog")
def get_asset_catalog(category: Optional[str] = None) -> List[Dict[str, Any]]:
    assets = AssetLibrary.list_all(category=category)
    return [a.to_dict() for a in assets]


@app.post("/api/composition/compose")
def compose_asset(req: CompositionRequest) -> Dict[str, Any]:
    try:
        res = AssetComposer.compose(
            character_asset_or_script=req.character_script or req.character_id or "character_player_soldier",
            weapon_id=req.weapon_id,
            helmet_id=req.helmet_id,
            backpack_id=req.backpack_id,
            fx_id=req.fx_id,
            engine_mode=req.engine,
            palette_name=req.palette,
            width=req.width,
            height=req.height,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Composition failed: {str(e)}")


@app.post("/api/composition/animate")
def animate_composition(req: CompositionAnimationRequest) -> Dict[str, Any]:
    try:
        res = AssetComposer.compose_animation(
            character_asset_or_script=req.character_script or req.character_id or "character_zombie_walker",
            animation_category=req.animation_category,
            num_frames=req.num_frames,
            fps=req.fps,
            weapon_id=req.weapon_id,
            helmet_id=req.helmet_id,
            backpack_id=req.backpack_id,
            fx_id=req.fx_id,
            engine_mode=req.engine,
            palette_name=req.palette,
            width=req.width,
            height=req.height,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Animated composition failed: {str(e)}")


@app.post("/api/asset_pack/export_zip")
def export_asset_pack(req: AssetPackExportRequest) -> Dict[str, Any]:
    try:
        if req.pack_type == "deadzone":
            pack = AssetPack.create_deadzone_starter_pack()
        else:
            pack = AssetPack(
                pack_id="pack_custom",
                name=req.pack_name or "Custom Asset Pack",
            )
            for aid in req.asset_ids:
                a = AssetLibrary.get(aid)
                if a:
                    pack.add_asset(a)

        zip_bytes = pack.export_zip()
        b64_zip = base64.b64encode(zip_bytes).decode("utf-8")
        filename = f"{pack.pack_id}.zip"

        return {
            "status": "success",
            "filename": filename,
            "pack_name": pack.name,
            "total_assets": len(pack.assets),
            "zip_base64": f"data:application/zip;base64,{b64_zip}",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Asset pack export failed: {str(e)}")


@app.get("/api/composition/metrics")
def get_composition_metrics() -> Dict[str, Any]:
    res = {
        "status": "ready",
        "model_architecture": "SocketPredictor (Residual2D + Sigmoid Regressor)",
        "parameter_count": 619432,
        "supported_categories": ASSET_CATEGORIES,
    }
    metrics_path = Path("data/composition_metrics.json")
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                res["training_metrics"] = json.load(f)
        except Exception:
            pass

    benchmarks_path = Path("data/benchmarks/composition_benchmark_results.json")
    if benchmarks_path.exists():
        try:
            with open(benchmarks_path, "r", encoding="utf-8") as f:
                res["benchmark_results"] = json.load(f)
        except Exception:
            pass

    return res


# --- Asset Library Endpoints ---
@app.get("/api/assets")
def list_assets(query: Optional[str] = None) -> List[Dict[str, Any]]:
    return asset_mgr.list_assets(query=query)


@app.get("/api/assets/{asset_id}")
def get_asset(asset_id: str) -> Dict[str, Any]:
    asset = asset_mgr.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset


@app.post("/api/assets")
def save_asset(req: AssetSaveRequest) -> Dict[str, Any]:
    return asset_mgr.save_asset(req.model_dump())


@app.delete("/api/assets/{asset_id}")
def delete_asset(asset_id: str) -> Dict[str, Any]:
    try:
        res = lifecycle_mgr.trash_asset(asset_id)
        return {"status": "trashed", "id": asset_id, "trash_id": res["trash_id"]}
    except FileNotFoundError:
        success = asset_mgr.delete_asset(asset_id)
        if not success:
            raise HTTPException(status_code=404, detail="Asset not found")
        return {"status": "deleted", "id": asset_id}
    except ProtectedResourceException as e:
        raise HTTPException(status_code=403, detail=str(e))


# --- Storage Management & Lifecycle Endpoints ---
@app.get("/api/storage/dashboard")
def get_storage_dashboard() -> Dict[str, Any]:
    """Returns storage dashboard statistics, classification breakdown, and health."""
    return storage_db.get_storage_dashboard()


@app.get("/api/storage/records")
def get_storage_records(table: str = "storage_registry", query: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
    """Inspects records in the database with optional search query."""
    allowed_tables = ["storage_registry", "assets", "checkpoints", "datasets", "benchmarks", "trash", "audit_log"]
    if table not in allowed_tables:
        raise HTTPException(status_code=400, detail=f"Table must be one of {allowed_tables}")
    records = storage_db.list_table_records(table_name=table, search_query=query, limit=limit)
    return {
        "table": table,
        "count": len(records),
        "records": records,
    }


@app.get("/api/storage/orphans")
def get_storage_orphans() -> Dict[str, Any]:
    """Identifies missing files and unindexed files on disk."""
    return storage_db.find_orphans_and_missing()


@app.post("/api/storage/sync")
def sync_storage_registry() -> Dict[str, Any]:
    """Rescans filesystem and synchronizes SQLite storage catalog."""
    count = storage_db.sync_from_disk()
    return {"status": "synced", "files_indexed": count}


@app.get("/api/storage/trash")
def list_trash() -> List[Dict[str, Any]]:
    """Lists all items currently in the trash bin."""
    return lifecycle_mgr.list_trash()


@app.post("/api/storage/trash")
def trash_asset_endpoint(req: TrashAssetRequest) -> Dict[str, Any]:
    """Safely moves a user-generated asset to trash. Rejects protected resources."""
    try:
        return lifecycle_mgr.trash_asset(asset_id=req.asset_id, reason=req.reason)
    except ProtectedResourceException as e:
        raise HTTPException(status_code=403, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/storage/restore")
def restore_asset_endpoint(req: RestoreAssetRequest) -> Dict[str, Any]:
    """Restores an item from trash back to active storage."""
    try:
        return lifecycle_mgr.restore_asset(trash_id=req.trash_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/storage/permanent_delete")
def permanent_delete_endpoint(req: PermanentDeleteRequest) -> Dict[str, Any]:
    """Permanently deletes an item from trash. Strictly requires confirm=True."""
    try:
        return lifecycle_mgr.permanent_delete(trash_id=req.trash_id, confirm=req.confirm)
    except ProtectedResourceException as e:
        raise HTTPException(status_code=403, detail=str(e))
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/storage/empty_trash")
def empty_trash_endpoint(req: EmptyTrashRequest) -> Dict[str, Any]:
    """Empties all eligible items from trash bin. Requires confirm=True."""
    try:
        return lifecycle_mgr.empty_trash(confirm=req.confirm)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/storage/cleanup_scan")
def storage_cleanup_scan() -> Dict[str, Any]:
    """Performs a non-destructive dry-run scan for disposable temporary files."""
    return cleaner.scan()


@app.post("/api/storage/cleanup_execute")
def storage_cleanup_execute(req: CleanupExecuteRequest) -> Dict[str, Any]:
    """Executes safe cleanup of verified temporary files. Requires confirm=True."""
    try:
        return cleaner.execute(confirm=req.confirm)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/storage/backup")
def create_backup_endpoint(req: BackupCreateRequest) -> Dict[str, Any]:
    """Creates a verified ZIP backup containing database catalog and metadata."""
    return backup_mgr.create_backup(
        backup_name=req.backup_name,
        include_db=req.include_db,
        include_benchmarks=req.include_benchmarks,
        include_assets_meta=req.include_assets_meta,
    )


@app.get("/api/storage/backups")
def list_backups_endpoint() -> List[Dict[str, Any]]:
    """Lists all available ZIP backups."""
    return backup_mgr.list_backups()


# Mount frontend build if directory exists
possible_dist_paths = [
    Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "dist",
    Path(__file__).resolve().parent.parent.parent / "frontend" / "dist",
    Path.cwd() / "frontend" / "dist",
]

for dist_path in possible_dist_paths:
    if dist_path.exists():
        app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="static")
        break
else:
    @app.get("/", include_in_schema=False)
    def frontend_not_built() -> HTMLResponse:
        return HTMLResponse(
            "<h1>SpriteMind API is running</h1>"
            "<p>The web studio has not been built yet. Run:</p>"
            "<pre>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</pre>"
            "<p>API docs: <a href='/docs'>/docs</a></p>",
            status_code=200,
        )
