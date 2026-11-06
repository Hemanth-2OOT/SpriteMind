"""Trains TemporalSVAN on the expanded animation dataset (4k sequences)."""
import sys
import torch
from pathlib import Path

backend = Path(__file__).resolve().parent.parent / "backend"
if str(backend) not in sys.path:
    sys.path.insert(0, str(backend))

from spritemind.ml.temporal_trainer import TemporalModelTrainer

if __name__ == "__main__":
    torch.set_num_threads(2)
    trainer = TemporalModelTrainer(learning_rate=0.001)
    hist = trainer.train_model(
        train_path="data/temporal_dataset_4k/train_sequences.jsonl",
        val_path="data/temporal_dataset_4k/val_sequences.jsonl",
        epochs=12,
        batch_size=32,
        save_ckpt_path="models/checkpoints/temporal_svan_best.pt",
    )
    print("best_val_loss", hist["best_val_loss"], "params", hist["parameter_count"])
