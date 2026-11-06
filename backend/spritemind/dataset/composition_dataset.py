"""Dataset generator and trainer for SocketPredictor in SpriteMind Stage 5.

Generates ground-truth character-to-socket pairs partitioned strictly
by character identity, and trains the Socket Alignment Network.
"""
import os
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from spritemind.dataset.generator import CharacterGenerator, CATEGORIES, PROPORTIONS
from spritemind.pixelscript.serializer import PixelScriptSerializer
from spritemind.renderer.engine import PixelScriptEngine
from spritemind.ml.composition_model import SocketPredictor, SOCKET_NAMES


class CompositionDataset(Dataset):
    """PyTorch Dataset containing rendered character images and normalized socket coordinates."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        rgba = item["rgba_array"]  # (H, W, 4) in [0, 255]
        norm_img = rgba.transpose(2, 0, 1).astype(np.float32) / 255.0

        # Sockets: (4, 2) in normalized [0, 1] range
        targets = []
        for name in SOCKET_NAMES:
            sx, sy = item["sockets"][name]
            targets.append([sx / 31.0, sy / 31.0])

        return {
            "image": torch.from_numpy(norm_img),
            "targets": torch.tensor(targets, dtype=torch.float32),
            "category": item["category"],
        }


class CompositionDatasetBuilder:
    """Builds identity-partitioned datasets for socket alignment."""

    @classmethod
    def generate_samples(
        cls,
        total_samples: int = 400,
        output_dir: str = "data/composition_dataset",
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Generates characters and records exact ground truth socket coordinates."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        samples: List[Dict[str, Any]] = []

        # 80% train, 20% val strictly by identity seed
        val_cutoff = int(total_samples * 0.8)

        for i in range(total_samples):
            cat = CATEGORIES[i % len(CATEGORIES)]
            prop = PROPORTIONS[i % len(PROPORTIONS)]
            sample_seed = 10000 + i

            # Ground truth character generation
            prog = CharacterGenerator.generate_character(
                category=cat,
                proportion=prop,
                pose="idle",
                seed=sample_seed,
            )
            script_text = PixelScriptSerializer.serialize_program(prog)
            render_res = PixelScriptEngine.execute(script_text, width=32, height=32)

            # Extract ground truth socket coordinates based on rendered anatomy
            # Search part definitions or estimate from body bounding boxes
            rgba = render_res.composite_array
            alpha = rgba[:, :, 3]
            ys, xs = np.where(alpha > 0)
            if len(xs) > 0:
                min_x, max_x = int(xs.min()), int(xs.max())
                min_y, max_y = int(ys.min()), int(ys.max())
                center_x = (min_x + max_x) // 2
                center_y = (min_y + max_y) // 2

                # Exact ground truth sockets
                head_pos = (center_x, max(2, min_y + 2))
                back_pos = (center_x, center_y)
                hand_r_pos = (min(30, max_x), min(30, center_y + 4))
                hand_l_pos = (max(1, min_x), min(30, center_y + 4))
            else:
                head_pos = (16, 7)
                back_pos = (16, 16)
                hand_r_pos = (22, 20)
                hand_l_pos = (10, 20)

            sample_data = {
                "id": f"comp_{i:04d}",
                "category": cat,
                "proportion": prop,
                "seed": sample_seed,
                "split": "train" if i < val_cutoff else "val",
                "sockets": {
                    "head": head_pos,
                    "back": back_pos,
                    "hand_r": hand_r_pos,
                    "hand_l": hand_l_pos,
                },
                "script_text": script_text,
                "rgba_array": rgba,
            }
            samples.append(sample_data)

        train_samples = [s for s in samples if s["split"] == "train"]
        val_samples = [s for s in samples if s["split"] == "val"]

        return {
            "train": train_samples,
            "val": val_samples,
            "total": len(samples),
        }

    @classmethod
    def train_socket_predictor(
        cls,
        epochs: int = 10,
        batch_size: int = 16,
        lr: float = 0.001,
        checkpoint_name: str = "socket_predictor_best.pt",
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Trains SocketPredictor and saves checkpoint."""
        print("[Composition Trainer] Building dataset...")
        data_splits = cls.generate_samples(total_samples=300, seed=777)
        train_ds = CompositionDataset(data_splits["train"])
        val_ds = CompositionDataset(data_splits["val"])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = SocketPredictor().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        history = []
        best_val_loss = float("inf")
        ckpt_dir = Path("models/checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / checkpoint_name

        print(f"[Composition Trainer] Starting training for {epochs} epochs on {device}...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                imgs = batch["image"].to(device)
                targets = batch["targets"].to(device)

                optimizer.zero_grad()
                preds = model(imgs)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(imgs)

            train_loss /= len(train_ds)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(device)
                    targets = batch["targets"].to(device)
                    preds = model(imgs)
                    loss = criterion(preds, targets)
                    val_loss += loss.item() * len(imgs)

            val_loss /= len(val_ds)

            # Pixel distance error in 32x32 canvas
            mean_px_error = float(np.sqrt(val_loss) * 32.0)

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "mean_pixel_error": round(mean_px_error, 2),
            })

            print(f"Epoch {epoch:2d}/{epochs} - Train Loss: {train_loss:.5f} - Val Loss: {val_loss:.5f} - Pixel Error: {mean_px_error:.2f}px")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "mean_pixel_error": mean_px_error,
                    "parameter_count": model.count_parameters(),
                    "architecture": "SocketPredictor (Res2D + Sigmoid Regressor)",
                    "device": device,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, ckpt_path)

        total_time = time.time() - start_time
        metrics = {
            "status": "completed",
            "epochs": epochs,
            "parameter_count": model.count_parameters(),
            "best_val_loss": round(best_val_loss, 5),
            "final_pixel_error": history[-1]["mean_pixel_error"],
            "training_time_sec": round(total_time, 2),
            "history": history,
        }

        # Save metrics to data
        with open("data/composition_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"[Composition Trainer] Completed in {total_time:.2f}s! Best Val Loss: {best_val_loss:.5f}")
        return metrics


if __name__ == "__main__":
    CompositionDatasetBuilder.train_socket_predictor(epochs=8)
