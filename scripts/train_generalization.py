"""Standalone script to execute supervised SVAN training on generalization_10k dataset."""
import sys
import time
from pathlib import Path

# Add backend to sys.path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from spritemind.ml.training import ModelTrainer


def main():
    print("=" * 60)
    print("SpriteMind: SVAN Supervised Generalization Training")
    print("=" * 60)

    trainer = ModelTrainer(checkpoints_dir="models/checkpoints", history_path="data/training_history.json")
    print(f"Target device: {trainer.device}")

    epochs = 4
    batch_size = 128
    lr = 0.0015
    dataset_name = "generalization_10k"
    save_ckpt = "svan_checkpoint_best.pt"

    print(f"Dataset: {dataset_name} | Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")
    print(f"Saving best checkpoint to: models/checkpoints/{save_ckpt}")

    res = trainer.start_training(
        dataset_name=dataset_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        optimizer_name="adamw",
        save_checkpoint_name=save_ckpt,
        patience=4,
        seed=42,
    )
    print("Training started:", res)

    last_epoch = 0
    while trainer.is_training:
        status = trainer.get_status()
        current_epoch = status.get("epoch", 0)
        if current_epoch != last_epoch and current_epoch > 0:
            last_epoch = current_epoch
            loss = status.get("loss", 0.0)
            val_loss = status.get("val_loss", 0.0)
            train_acc = status.get("token_accuracy", 0.0)
            val_acc = status.get("val_token_accuracy", 0.0)
            elapsed = status.get("elapsed_seconds", 0.0)
            lr_curr = status.get("learning_rate", 0.0)
            print(
                f"[Epoch {current_epoch}/{epochs}] "
                f"Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
                f"LR: {lr_curr:.6f} | Elapsed: {elapsed:.1f}s"
            )
        time.sleep(3)

    final_status = trainer.get_status()
    print("=" * 60)
    print(f"Training status: {final_status.get('state')}")
    print(f"Best Epoch: {final_status.get('best_epoch')}")
    print(f"Best Val Loss: {final_status.get('best_val_loss')}")
    print(f"Saved Checkpoint: {final_status.get('saved_checkpoint')}")
    print(f"Total Time: {final_status.get('elapsed_seconds')}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
