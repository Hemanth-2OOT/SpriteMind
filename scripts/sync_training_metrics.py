"""Rewrites data/training_history.json and data/temporal_metrics.json from checkpoint metadata.

Run after promoting a checkpoint, then `python scripts/update_readme_metrics.py`:

    python scripts/sync_training_metrics.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import torch  # noqa: E402

CHECKPOINTS = ROOT / "models" / "checkpoints"


def _metadata(name: str) -> dict:
    """SVAN nests its metadata under "metadata"; the temporal trainer stores it at top level."""
    payload = torch.load(CHECKPOINTS / name, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {}
    meta = dict(payload.get("metadata") or {})
    for key in ("epoch", "val_loss", "best_val_loss", "parameter_count"):
        if key in payload and key not in meta:
            meta[key] = payload[key]
    state = payload.get("model_state_dict", payload)
    if isinstance(state, dict):
        meta["state_dict_parameters"] = int(
            sum(v.numel() for v in state.values() if hasattr(v, "numel"))
        )
    return meta


def _dataset_meta(name: str) -> dict:
    path = ROOT / "data" / "datasets" / name / "metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _write(rel: str, data: dict) -> None:
    (ROOT / rel).write_text(json.dumps(data, indent=2) + "\n")


def sync_svan() -> None:
    meta = _metadata("svan_checkpoint_best.pt")
    dataset = meta.get("dataset_name", "")
    splits = _dataset_meta(dataset).get("splits", {})
    epochs = int(meta.get("epoch", 0))
    loss = meta.get("train_loss")
    acc = meta.get("token_accuracy")
    history = {
        "dataset_name": dataset,
        "train_samples": splits.get("train"),
        "val_samples": splits.get("val"),
        "total_epochs": epochs,
        "completed_epochs": epochs,
        "best_epoch": epochs,
        "best_val_loss": meta.get("val_loss", meta.get("best_val_loss")),
        "device": meta.get("device"),
        "parameters": meta.get("state_dict_parameters") or meta.get("parameters"),
        "vocab_size": meta.get("vocab_size"),
        "loss_history": [{"epoch": epochs, "loss": loss}],
        "val_loss_history": [{"epoch": epochs, "val_loss": meta.get("val_loss", meta.get("best_val_loss"))}],
        "token_accuracy_history": [{"epoch": epochs, "train_accuracy": loss, "val_accuracy": acc}],
        "source": "checkpoint metadata; per-epoch history not retained for this run",
    }
    _write("data/training_history.json", history)
    print(f"training_history.json <- svan_checkpoint_best.pt (val {history['best_val_loss']}, {epochs} epochs)")


def sync_temporal() -> None:
    meta = _metadata("temporal_svan_best.pt")
    current = json.loads((ROOT / "data" / "temporal_metrics.json").read_text())
    current.update(
        {
            "best_epoch": int(meta.get("epoch", current.get("best_epoch", 0))),
            "best_val_loss": meta.get("val_loss") or meta.get("best_val_loss") or current.get("best_val_loss"),
            "parameter_count": int(meta.get("state_dict_parameters") or current.get("parameter_count", 0)),
            "checkpoint": "models/checkpoints/temporal_svan_best.pt",
        }
    )
    _write("data/temporal_metrics.json", current)
    print(f"temporal_metrics.json <- temporal_svan_best.pt (val {current['best_val_loss']})")


if __name__ == "__main__":
    sync_svan()
    sync_temporal()
