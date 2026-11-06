"""Continue SVAN training from the shipped checkpoint at a lower learning rate.

Kept separate from scripts/train_detail.py so the fine-tune writes its own
history file and checkpoint name: the shipped weights are only replaced once
the fine-tuned ones win on the held-out benchmark.

    python scripts/finetune_svan.py --dataset expanded_synthetic_25k --epochs 2 --lr 0.0002
"""
import argparse
import json
import sys
import time

from pathlib import Path

backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from spritemind.ml.training import ModelTrainer


CONFIG = {
    "dataset_name": "generalization_detail_12k",
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 0.0005,
    "max_seq_len": 320,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 6,
    "dim_feedforward": 512,
    "dropout": 0.05,
    "label_smoothing": 0.03,
    "threads": 8,
    "patience": 2,
    "save_checkpoint_name": "svan_finetuned.pt",
    "resume_from": "svan_checkpoint_best.pt",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=CONFIG["dataset_name"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--threads", type=int, default=CONFIG["threads"])
    parser.add_argument("--save-name", default=CONFIG["save_checkpoint_name"])
    parser.add_argument("--resume-from", default=CONFIG["resume_from"])
    parser.add_argument("--history", default="data/finetune_history.json")
    return parser.parse_args()


def main():
    args = parse_args()
    config = dict(CONFIG)
    config.update(
        dataset_name=args.dataset,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        threads=args.threads,
        save_checkpoint_name=args.save_name,
        resume_from=args.resume_from,
    )
    trainer = ModelTrainer(checkpoints_dir="models/checkpoints", history_path=args.history)
    print("device:", trainer.device, "config:", json.dumps(config), flush=True)
    trainer.start_training(
        seed=42,
        resume=True,
        **{k: v for k, v in config.items() if k != "save_checkpoint_name"},
        save_checkpoint_name=config["save_checkpoint_name"],
    )
    last_epoch = 0
    while trainer.is_training:
        status = trainer.get_status()
        epoch = status.get("epoch", 0)
        if epoch != last_epoch and epoch > 0:
            last_epoch = epoch
            print(
                f"[Epoch {epoch}/{config['epochs']}] loss {status.get('loss')} val {status.get('val_loss')} "
                f"acc {status.get('token_accuracy')}% val_acc {status.get('val_token_accuracy')}% "
                f"best {status.get('best_val_loss')} elapsed {status.get('elapsed_seconds')}s",
                flush=True,
            )
        time.sleep(5)
    final = trainer.get_status()
    print("final:", json.dumps(final.get("state")), final.get("error_message", ""), flush=True)


if __name__ == "__main__":
    main()
