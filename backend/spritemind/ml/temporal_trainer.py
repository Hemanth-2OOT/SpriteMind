"""Supervised Temporal Training Pipeline for TemporalSVAN."""
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from spritemind.dataset.generator import CATEGORIES
from spritemind.dataset.temporal_dataset import ANIMATION_CATEGORIES, CYCLIC_ANIMATIONS
from spritemind.ml.temporal import TemporalSVAN, PARTS, PART_TO_IDX, CAT_TO_IDX, ANIM_TO_IDX
from spritemind.renderer.engine import PixelScriptEngine


class AnimationSequenceDataset(Dataset):
    """Loads temporal animation sequences from JSONL dataset files."""

    def __init__(self, jsonl_path: str, max_samples: Optional[int] = None):
        self.records: List[Dict[str, Any]] = []
        path = Path(jsonl_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.records.append(json.loads(line.strip()))
                        if max_samples and len(self.records) >= max_samples:
                            break

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        cat = rec.get("category", "knight")
        anim = rec.get("animation_category", "walk")
        frames = rec.get("frames", [])
        num_frames = len(frames)

        cat_idx = CAT_TO_IDX.get(cat, 0)
        anim_idx = ANIM_TO_IDX.get(anim, 0)
        is_loop = anim in CYCLIC_ANIMATIONS

        # Extract phases & target trajectories
        phases = np.zeros((num_frames, 1), dtype=np.float32)
        target_trajectories = np.zeros((num_frames, len(PARTS), 3), dtype=np.float32)

        for f_idx, fr in enumerate(frames):
            phases[f_idx, 0] = fr.get("phase", f_idx / float(max(1, num_frames - 1)))
            offs = fr.get("offsets", {})
            for p_idx, part_name in enumerate(PARTS):
                if part_name in offs:
                    dx, dy, rot = offs[part_name]
                    target_trajectories[f_idx, p_idx, 0] = float(dx)
                    target_trajectories[f_idx, p_idx, 1] = float(dy)
                    target_trajectories[f_idx, p_idx, 2] = float(rot)

        # Generate reference Frame 0 RGBA tensor from base frame script
        base_script = frames[0].get("script_text", "")
        pal = rec.get("palette", "pico8")
        try:
            res = PixelScriptEngine.execute(base_script, width=32, height=32, palette_name=pal)
            rgba = res.composite_array.transpose(2, 0, 1).astype(np.float32) / 255.0
        except Exception:
            rgba = np.zeros((4, 32, 32), dtype=np.float32)

        return {
            "cat_idx": torch.tensor(cat_idx, dtype=torch.long),
            "anim_idx": torch.tensor(anim_idx, dtype=torch.long),
            "phases": torch.from_numpy(phases),
            "targets": torch.from_numpy(target_trajectories),
            "ref_frame": torch.from_numpy(rgba),
            "is_loop": torch.tensor(1.0 if is_loop else 0.0, dtype=torch.float32),
        }


def pad_sequence_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collates sequences of varying frame counts by padding to max length in batch."""
    max_len = max(b["phases"].shape[0] for b in batch)
    B = len(batch)
    num_parts = len(PARTS)

    padded_phases = torch.zeros((B, max_len, 1), dtype=torch.float32)
    padded_targets = torch.zeros((B, max_len, num_parts, 3), dtype=torch.float32)
    mask = torch.zeros((B, max_len), dtype=torch.bool)

    cat_indices = torch.zeros(B, dtype=torch.long)
    anim_indices = torch.zeros(B, dtype=torch.long)
    ref_frames = torch.zeros((B, 4, 32, 32), dtype=torch.float32)
    is_loops = torch.zeros(B, dtype=torch.float32)

    for i, b in enumerate(batch):
        L = b["phases"].shape[0]
        padded_phases[i, :L] = b["phases"]
        padded_targets[i, :L] = b["targets"]
        mask[i, :L] = True

        cat_indices[i] = b["cat_idx"]
        anim_indices[i] = b["anim_idx"]
        ref_frames[i] = b["ref_frame"]
        is_loops[i] = b["is_loop"]

    return {
        "cat_idx": cat_indices,
        "anim_idx": anim_indices,
        "phases": padded_phases,
        "targets": padded_targets,
        "mask": mask,
        "ref_frame": ref_frames,
        "is_loop": is_loops,
    }


class TemporalModelTrainer:
    """Trains TemporalSVAN using multi-task kinematic trajectory, smoothness, and loop closure losses."""

    def __init__(
        self,
        d_model: int = 128,
        learning_rate: float = 0.002,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TemporalSVAN(d_model=d_model).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=8)

        self.mse_loss = nn.MSELoss(reduction="none")
        self.l1_loss = nn.L1Loss(reduction="none")

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_traj_loss = 0.0
        total_smooth_loss = 0.0
        total_loop_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            cat_idx = batch["cat_idx"].to(self.device)
            anim_idx = batch["anim_idx"].to(self.device)
            phases = batch["phases"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)  # (B, SeqLen)
            ref_frame = batch["ref_frame"].to(self.device)
            is_loop = batch["is_loop"].to(self.device)

            self.optimizer.zero_grad()
            out = self.model(cat_idx, anim_idx, phases, ref_frame)
            pred_traj = out["trajectories"]  # (B, SeqLen, num_parts, 3)

            # 1. Trajectory MSE loss with length masking
            mask_exp = mask.unsqueeze(-1).unsqueeze(-1)  # (B, SeqLen, 1, 1)
            traj_diff = (pred_traj - targets) * mask_exp
            loss_traj = (traj_diff ** 2).sum() / (mask_exp.sum() * pred_traj.shape[2] * 3).clamp(min=1.0)

            # 2. Temporal Smoothness loss (penalty on 2nd derivative acceleration)
            if pred_traj.shape[1] >= 3:
                accel = pred_traj[:, 2:] - 2 * pred_traj[:, 1:-1] + pred_traj[:, :-2]
                mask_accel = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).unsqueeze(-1).unsqueeze(-1)
                loss_smooth = ((accel * mask_accel) ** 2).sum() / (mask_accel.sum() * pred_traj.shape[2] * 3).clamp(min=1.0)
            else:
                loss_smooth = torch.tensor(0.0, device=self.device)

            # 3. Loop Closure loss for cyclic animations: |T[last] - T[0]|
            loop_diff = (pred_traj[:, -1] - pred_traj[:, 0]) * is_loop.unsqueeze(-1).unsqueeze(-1)
            loss_loop = (loop_diff ** 2).mean()

            # 4. Identity confidence loss (encourage identity retention)
            loss_ident = (1.0 - out["identity_confidence"]).mean()

            loss = loss_traj + 0.25 * loss_smooth + 0.15 * loss_loop + 0.05 * loss_ident
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            bs = cat_idx.size(0)
            total_loss += loss.item() * bs
            total_traj_loss += loss_traj.item() * bs
            total_smooth_loss += loss_smooth.item() * bs
            total_loop_loss += loss_loop.item() * bs
            total_samples += bs

        self.scheduler.step()
        n = max(1, total_samples)
        return {
            "loss": round(total_loss / n, 4),
            "traj_loss": round(total_traj_loss / n, 4),
            "smooth_loss": round(total_smooth_loss / n, 4),
            "loop_loss": round(total_loop_loss / n, 4),
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_traj_mse = 0.0
        total_smooth = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                cat_idx = batch["cat_idx"].to(self.device)
                anim_idx = batch["anim_idx"].to(self.device)
                phases = batch["phases"].to(self.device)
                targets = batch["targets"].to(self.device)
                mask = batch["mask"].to(self.device)
                ref_frame = batch["ref_frame"].to(self.device)

                out = self.model(cat_idx, anim_idx, phases, ref_frame)
                pred_traj = out["trajectories"]

                mask_exp = mask.unsqueeze(-1).unsqueeze(-1)
                traj_diff = (pred_traj - targets) * mask_exp
                loss_traj = (traj_diff ** 2).sum() / (mask_exp.sum() * pred_traj.shape[2] * 3).clamp(min=1.0)

                bs = cat_idx.size(0)
                total_traj_mse += loss_traj.item() * bs
                total_loss += loss_traj.item() * bs
                total_samples += bs

        n = max(1, total_samples)
        return {
            "val_loss": round(total_loss / n, 4),
            "val_traj_mse": round(total_traj_mse / n, 4),
        }

    def train_model(
        self,
        train_path: str = "data/temporal_dataset_1k/train_sequences.jsonl",
        val_path: str = "data/temporal_dataset_1k/val_sequences.jsonl",
        epochs: int = 6,
        batch_size: int = 32,
        save_ckpt_path: str = "models/checkpoints/temporal_svan_best.pt",
    ) -> Dict[str, Any]:
        """Runs multi-epoch supervised temporal training loop."""
        train_ds = AnimationSequenceDataset(train_path)
        val_ds = AnimationSequenceDataset(val_path)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_sequence_collate)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_sequence_collate)

        best_val_loss = float("inf")
        history = {
            "epochs": epochs,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "train_loss": [],
            "val_loss": [],
            "val_traj_mse": [],
        }

        print(f"Starting TemporalSVAN training: {len(train_ds)} train sequences, {len(val_ds)} val sequences on {self.device}")
        for ep in range(1, epochs + 1):
            t0 = time.perf_counter()
            t_metrics = self.train_epoch(train_loader)
            v_metrics = self.evaluate(val_loader)
            elapsed = time.perf_counter() - t0

            history["train_loss"].append(t_metrics["loss"])
            history["val_loss"].append(v_metrics["val_loss"])
            history["val_traj_mse"].append(v_metrics["val_traj_mse"])

            print(f"Epoch {ep}/{epochs} [{elapsed:.1f}s] - Train Loss: {t_metrics['loss']} (Traj: {t_metrics['traj_loss']}) - Val Loss: {v_metrics['val_loss']}")

            if v_metrics["val_loss"] < best_val_loss:
                best_val_loss = v_metrics["val_loss"]
                # Save best checkpoint
                ckpt_p = Path(save_ckpt_path)
                ckpt_p.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "epoch": ep,
                    "val_loss": best_val_loss,
                    "val_traj_mse": v_metrics["val_traj_mse"],
                    "parameter_count": self.model.count_parameters(),
                    "architecture": "TemporalSVAN",
                    "device": self.device,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, ckpt_p)

        history["best_val_loss"] = best_val_loss
        history["parameter_count"] = self.model.count_parameters()
        history["checkpoint"] = save_ckpt_path

        # Save metrics JSON
        metrics_p = Path("data/temporal_metrics.json")
        metrics_p.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_p, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        return history


if __name__ == "__main__":
    trainer = TemporalModelTrainer()
    trainer.train_model(epochs=6)
