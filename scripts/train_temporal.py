"""Trains TemporalSVAN on an animation dataset (default: the 4k grammar-v2 sequences).

    python scripts/train_temporal.py --train data/temporal_dataset_4k/train_sequences.jsonl
"""
import argparse
import sys
from pathlib import Path

import torch

backend = Path(__file__).resolve().parent.parent / "backend"
if str(backend) not in sys.path:
    sys.path.insert(0, str(backend))

from spritemind.ml.temporal_trainer import TemporalModelTrainer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/temporal_dataset_4k/train_sequences.jsonl")
    parser.add_argument("--val", default="data/temporal_dataset_4k/val_sequences.jsonl")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--save-path", default="models/checkpoints/temporal_svan_best.pt")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    trainer = TemporalModelTrainer(learning_rate=args.lr)
    hist = trainer.train_model(
        train_path=args.train,
        val_path=args.val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_ckpt_path=args.save_path,
    )
    print("best_val_loss", hist["best_val_loss"], "params", hist["parameter_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
