"""Supervised SVAN training on detailed-anatomy datasets (full-length programs)."""
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
    "learning_rate": 0.0012,
    "max_seq_len": 320,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 6,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "label_smoothing": 0.05,
    "threads": 6,
    "patience": 2,
    "save_checkpoint_name": "svan_checkpoint_best.pt",
}


def main():
    trainer = ModelTrainer(checkpoints_dir="models/checkpoints", history_path="data/training_history.json")
    print("device:", trainer.device, "config:", json.dumps(CONFIG), flush=True)
    trainer.start_training(seed=42, **{k: v for k, v in CONFIG.items() if k != "save_checkpoint_name"},
                           save_checkpoint_name=CONFIG["save_checkpoint_name"])
    last_epoch = 0
    while trainer.is_training:
        st = trainer.get_status()
        ep = st.get("epoch", 0)
        if ep != last_epoch and ep > 0:
            last_epoch = ep
            print(f"[Epoch {ep}/{CONFIG['epochs']}] loss {st.get('loss')} val {st.get('val_loss')} "
                  f"acc {st.get('token_accuracy')}% val_acc {st.get('val_token_accuracy')}% "
                  f"best {st.get('best_val_loss')} elapsed {st.get('elapsed_seconds')}s", flush=True)
        time.sleep(5)
    print("final:", json.dumps(trainer.get_status().get("state")), trainer.get_status().get("error_message", ""), flush=True)


if __name__ == "__main__":
    main()
