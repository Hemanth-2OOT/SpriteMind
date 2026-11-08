"""Training pipeline for Neural Visual Critic on clean and synthetic defect sprites."""
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from spritemind.critic.model import NeuralVisualCritic, DEFECT_CLASSES, SEVERITY_CLASSES
from spritemind.renderer.engine import PixelScriptEngine
from spritemind.paths import DEFAULT_CHECKPOINTS, resolve_checkpoints_dir


DEFECT_TYPE_MAP = {
    "clean": "clean",
    "stray_pixel": "stray_pixel",
    "disconnected_part": "disconnected_part",
    "palette_clash": "palette_leak",
    "palette_leak": "palette_leak",
    "missing_feature": "missing_limb",
    "missing_limb": "missing_limb",
    "silhouette_hole": "unexpected_hole",
    "unexpected_hole": "unexpected_hole",
    "asymmetry": "asymmetry",
}


def generate_gaussian_heatmap(h: int = 32, w: int = 32, cx: int = 16, cy: int = 16, sigma: float = 2.5) -> np.ndarray:
    """Generates a 2D Gaussian heatmap centered at (cx, cy)."""
    y, x = np.ogrid[:h, :w]
    heatmap = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


class CriticDataset(Dataset):
    """PyTorch Dataset loading pre-rendered sprite RGBA tensors with localized defect labels."""

    def __init__(self, jsonl_path: Path, split: str = "train", max_samples: int = 2500):
        self.images = []
        self.defect_labels = []
        self.severity_labels = []
        self.quality_scores = []
        self.heatmaps = []

        raw_records = []
        clean_count = 0
        defect_count = 0

        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if split != "all" and item.get("split") != split:
                        continue

                    has_defect = item.get("has_defect", False)
                    # Balance classes in training
                    if split == "train":
                        if has_defect and defect_count < max_samples // 2:
                            raw_records.append(item)
                            defect_count += 1
                        elif not has_defect and clean_count < max_samples // 2:
                            raw_records.append(item)
                            clean_count += 1
                    else:
                        raw_records.append(item)

                    if len(raw_records) >= max_samples:
                        break

        # Pre-render all into tensors
        img_arrays = []
        for item in raw_records:
            script = item.get("script_text", "")
            has_defect = item.get("has_defect", False)
            d_info = item.get("defect_info") or {}
            raw_dtype = d_info.get("defect_type") or d_info.get("type") or "clean"
            mapped_dtype = DEFECT_TYPE_MAP.get(raw_dtype, "clean" if not has_defect else "stray_pixel")

            if not has_defect:
                mapped_dtype = "clean"
            elif mapped_dtype == "clean":
                mapped_dtype = "stray_pixel"

            label_idx = DEFECT_CLASSES.index(mapped_dtype) if mapped_dtype in DEFECT_CLASSES else 0
            self.defect_labels.append(label_idx)

            # Severity
            sev_str = d_info.get("severity", "minor")
            sev_idx = SEVERITY_CLASSES.index(sev_str) if sev_str in SEVERITY_CLASSES else 0
            self.severity_labels.append(sev_idx)

            # Quality
            quality = 95.0 if not has_defect else (55.0 if sev_str == "critical" else 72.0)
            self.quality_scores.append(quality)

            # Heatmap
            hm = np.zeros((1, 32, 32), dtype=np.float32)
            if has_defect and "bbox" in d_info:
                bb = d_info["bbox"]
                cx = (bb.get("min_x", 16) + bb.get("max_x", 16)) // 2
                cy = (bb.get("min_y", 16) + bb.get("max_y", 16)) // 2
                hm[0] = generate_gaussian_heatmap(32, 32, cx, cy)
            self.heatmaps.append(hm)

            # Render RGBA
            try:
                res = PixelScriptEngine.execute(script, width=32, height=32)
                arr = res.composite_array.transpose(2, 0, 1).astype(np.float32) / 255.0
            except Exception:
                arr = np.zeros((4, 32, 32), dtype=np.float32)
            img_arrays.append(arr)

        if len(img_arrays) > 0:
            self.images = torch.from_numpy(np.array(img_arrays, dtype=np.float32))
            self.defect_labels = torch.tensor(self.defect_labels, dtype=torch.long)
            self.severity_labels = torch.tensor(self.severity_labels, dtype=torch.long)
            self.quality_scores = torch.tensor(self.quality_scores, dtype=torch.float32).unsqueeze(1)
            self.heatmaps = torch.from_numpy(np.array(self.heatmaps, dtype=np.float32))
        else:
            self.images = torch.empty((0, 4, 32, 32), dtype=torch.float32)
            self.defect_labels = torch.empty(0, dtype=torch.long)
            self.severity_labels = torch.empty(0, dtype=torch.long)
            self.quality_scores = torch.empty((0, 1), dtype=torch.float32)
            self.heatmaps = torch.empty((0, 1, 32, 32), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "image": self.images[idx],
            "defect_label": self.defect_labels[idx],
            "severity_label": self.severity_labels[idx],
            "quality_score": self.quality_scores[idx],
            "heatmap": self.heatmaps[idx],
        }


class VisualCriticTrainer:
    """Trainer and evaluator for NeuralVisualCritic."""

    def __init__(self, checkpoints_dir: str = DEFAULT_CHECKPOINTS):
        self.checkpoints_dir = resolve_checkpoints_dir(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralVisualCritic().to(self.device)

    def train(
        self,
        dataset_path: str = "data/datasets/generalization_10k/records.jsonl",
        epochs: int = 5,
        batch_size: int = 64,
        lr: float = 0.001,
        save_name: str = "visual_critic_best.pt",
    ) -> Dict[str, Any]:
        """Trains visual critic and computes true precision, recall, and F1 metrics."""
        t0 = time.time()
        jsonl = Path(dataset_path)

        train_ds = CriticDataset(jsonl, split="train", max_samples=2000)
        val_ds = CriticDataset(jsonl, split="val", max_samples=500)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        # Inverse-frequency class weights: defect classes are rare, and an unweighted
        # loss collapses to "everything is clean" (recall below 25%).
        counts = torch.bincount(train_ds.defect_labels, minlength=len(DEFECT_CLASSES)).float().clamp(min=1.0)
        cls_weights = (counts.sum() / (counts * len(DEFECT_CLASSES))).to(self.device) if len(train_ds) else None
        cls_criterion = nn.CrossEntropyLoss(weight=cls_weights)
        sev_criterion = nn.CrossEntropyLoss()
        heat_criterion = nn.MSELoss()
        qual_criterion = nn.L1Loss()

        best_f1 = 0.0
        best_metrics: Dict[str, Any] = {}

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0

            for batch in train_loader:
                img = batch["image"].to(self.device)
                d_tgt = batch["defect_label"].to(self.device)
                s_tgt = batch["severity_label"].to(self.device)
                q_tgt = batch["quality_score"].to(self.device)
                h_tgt = batch["heatmap"].to(self.device)

                optimizer.zero_grad()
                out = self.model(img)

                loss_cls = cls_criterion(out["defect_logits"], d_tgt)
                loss_sev = sev_criterion(out["severity_logits"], s_tgt)
                loss_heat = heat_criterion(out["heatmap"], h_tgt)
                loss_qual = qual_criterion(out["quality_score"], q_tgt)

                total_loss = loss_cls + 0.4 * loss_sev + 10.0 * loss_heat + 0.02 * loss_qual
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += total_loss.item()

            # Validation Loop
            self.model.eval()
            val_loss = 0.0
            tp, fp, fn, tn = 0, 0, 0, 0
            correct_cat = 0
            total_val = 0

            with torch.no_grad():
                for v_batch in val_loader:
                    v_img = v_batch["image"].to(self.device)
                    v_d_tgt = v_batch["defect_label"].to(self.device)
                    v_s_tgt = v_batch["severity_label"].to(self.device)
                    v_q_tgt = v_batch["quality_score"].to(self.device)
                    v_h_tgt = v_batch["heatmap"].to(self.device)

                    v_out = self.model(v_img)
                    v_l_cls = cls_criterion(v_out["defect_logits"], v_d_tgt)
                    v_l_sev = sev_criterion(v_out["severity_logits"], v_s_tgt)
                    v_l_heat = heat_criterion(v_out["heatmap"], v_h_tgt)
                    v_l_qual = qual_criterion(v_out["quality_score"], v_q_tgt)

                    v_loss = v_l_cls + 0.4 * v_l_sev + 10.0 * v_l_heat + 0.02 * v_l_qual
                    val_loss += v_loss.item()

                    preds = torch.argmax(v_out["defect_logits"], dim=-1)
                    correct_cat += (preds == v_d_tgt).sum().item()
                    total_val += len(v_d_tgt)

                    # Binary defect classification metrics: 0 is clean, >0 is defective
                    for p, t in zip(preds.cpu().numpy(), v_d_tgt.cpu().numpy()):
                        pred_defect = (p > 0)
                        true_defect = (t > 0)
                        if pred_defect and true_defect:
                            tp += 1
                        elif pred_defect and not true_defect:
                            fp += 1
                        elif not pred_defect and true_defect:
                            fn += 1
                        else:
                            tn += 1

            precision = tp / max(1, (tp + fp))
            recall = tp / max(1, (tp + fn))
            f1 = (2 * precision * recall) / max(1e-6, (precision + recall))
            cat_accuracy = (correct_cat / max(1, total_val)) * 100.0

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": round(running_loss / max(1, len(train_loader)), 4),
                "val_loss": round(val_loss / max(1, len(val_loader)), 4),
                "precision": round(precision * 100.0, 2),
                "recall": round(recall * 100.0, 2),
                "f1": round(f1, 4),
                "category_accuracy": round(cat_accuracy, 2),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }

            if f1 > best_f1:
                best_f1 = f1
                best_metrics = epoch_metrics

                # Save best checkpoint
                ckpt_path = self.checkpoints_dir / save_name
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "metadata": {
                        "epoch": epoch,
                        "parameters": self.model.count_parameters(),
                        "precision": round(precision * 100.0, 2),
                        "recall": round(recall * 100.0, 2),
                        "f1": round(f1, 4),
                        "accuracy": round(cat_accuracy, 2),
                        "defect_classes": DEFECT_CLASSES,
                        "device": str(self.device),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                }, ckpt_path)

        # Save training summary
        metrics_file = Path("data/critic_metrics.json")
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "total_epochs": epochs,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "best_f1": best_metrics.get("f1", 0.0),
            "precision": best_metrics.get("precision", 0.0),
            "recall": best_metrics.get("recall", 0.0),
            "accuracy": best_metrics.get("category_accuracy", 0.0),
            "elapsed_seconds": round(time.time() - t0, 1),
            "defect_classes": DEFECT_CLASSES,
        }
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
