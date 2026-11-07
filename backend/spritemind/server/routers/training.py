"""Datasets, training runs, evaluation and benchmarks."""
from typing import Any, Dict, List
from fastapi import APIRouter, BackgroundTasks, HTTPException

from pathlib import Path
from spritemind.ml.evaluation import ModelEvaluator
from spritemind.server import history
from spritemind.server.schemas import DatasetGenerateRequest
from spritemind.server.schemas import TrainingStartRequest
import json
import torch

from spritemind.server.runtime import dataset_mgr, pipeline, trainer
router = APIRouter()

@router.get("/api/dataset/stats")
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

@router.get("/api/dataset/samples")
def get_dataset_samples(name: str = "default_synth", limit: int = 24, offset: int = 0) -> List[Dict[str, Any]]:
    return dataset_mgr.list_samples(name=name, limit=limit, offset=offset)

@router.post("/api/dataset/generate")
def generate_dataset(req: DatasetGenerateRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    summary = dataset_mgr.generate_dataset(
        name=req.name,
        total_samples=req.total_samples,
        defect_rate=req.defect_rate,
        seed=req.seed,
    )
    return {"status": "success", "summary": summary}

@router.post("/api/training/start")
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

@router.post("/api/training/stop")
def stop_training() -> Dict[str, Any]:
    return trainer.stop_training()

@router.get("/api/training/status")
def get_training_status() -> Dict[str, Any]:
    status = trainer.get_status()
    cuda_available = torch.cuda.is_available()
    status["device"] = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    return status

@router.get("/api/training/metrics")
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

@router.get("/api/training/history")
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

@router.get("/api/benchmarks/gallery")
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

@router.get("/api/benchmarks/report_card")
def get_benchmark_report_card(engine: str = "trained_model") -> Dict[str, Any]:
    """Per-category aggregation of the stored benchmark gallery."""
    from spritemind.ml.report_card import summarise_gallery

    results_path = Path("data/benchmarks/benchmark_results.json")
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="No benchmark results yet; run a benchmark first")
    data = json.loads(results_path.read_text(encoding="utf-8"))
    card = summarise_gallery(data.get("gallery", []), engine=engine)
    card["timestamp"] = data.get("summary", {}).get("timestamp")
    return card

@router.post("/api/benchmarks/run")
def run_benchmark_endpoint(limit: int = 20) -> Dict[str, Any]:
    from spritemind.ml.benchmark import GeneralizationBenchmark
    bm = GeneralizationBenchmark()
    results = bm.run_benchmark(limit=limit)
    return {"status": "success", "summary": results.get("summary", {})}

@router.get("/api/evaluation/results")
def get_evaluation_results(num_samples: int = 5) -> Dict[str, Any]:
    return ModelEvaluator.compare_all(pipeline, num_samples=num_samples)
