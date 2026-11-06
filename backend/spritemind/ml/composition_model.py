"""Trainable Socket Alignment Network (SocketPredictor) for SpriteMind Stage 5.

Predicts exact pixel coordinates for equipment attachment sockets
(hand_r, hand_l, head, back) by visually inspecting character sprites.
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np


SOCKET_NAMES = ["hand_r", "hand_l", "head", "back"]
NUM_SOCKETS = len(SOCKET_NAMES)


class SocketPredictor(nn.Module):
    """Convolutional Neural Network predicting socket attachment points from visual silhouettes."""

    def __init__(self, num_sockets: int = NUM_SOCKETS):
        super().__init__()
        self.num_sockets = num_sockets

        # Visual Stem: (B, 4, 32, 32) -> (B, 64, 8, 8)
        self.stem = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
        )

        # Residual Block: (B, 64, 8, 8) -> (B, 64, 8, 8)
        self.res_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
        self.relu = nn.ReLU(inplace=True)

        # Coordinate Regression Head: (B, 64*8*8 = 4096) -> (B, num_sockets * 2)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_sockets * 2),
            nn.Sigmoid(),  # Normalized to [0, 1] canvas coordinates
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x: (B, 4, 32, 32) RGBA image tensor normalized to [0, 1].
        Returns:
            (B, num_sockets, 2) in normalized [0, 1] range.
        """
        feat = self.stem(x)
        res = self.res_block(feat)
        feat = self.relu(feat + res)
        coords = self.regressor(feat)
        return coords.view(-1, self.num_sockets, 2)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SocketPredictorManager:
    """Manages loading, inference, and fallback for SocketPredictor."""

    _cached_model: Optional[SocketPredictor] = None
    _cached_checkpoint: Optional[str] = None

    @classmethod
    def get_or_load_model(
        cls,
        checkpoint_name: str = "socket_predictor_best.pt",
        device: str = "cpu",
    ) -> Tuple[SocketPredictor, bool]:
        """Loads trained checkpoint or returns initialized model with status flag."""
        ckpt_path = Path("models/checkpoints") / checkpoint_name
        model = SocketPredictor().to(device)
        has_trained = False

        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=device)
                state = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state)
                has_trained = True
                cls._cached_model = model
                cls._cached_checkpoint = checkpoint_name
            except Exception:
                has_trained = False
        else:
            has_trained = False

        model.eval()
        return model, has_trained

    @classmethod
    def predict_sockets(
        cls,
        rgba_image: np.ndarray,
        engine_mode: str = "trained_model",
        device: str = "cpu",
    ) -> Dict[str, Tuple[int, int]]:
        """Predicts socket attachment coordinates for a 32x32 RGBA numpy array."""
        h, w = rgba_image.shape[:2]

        if engine_mode == "procedural_baseline":
            # Deterministic anatomical heuristic
            return {
                "hand_r": (int(w * 0.68), int(h * 0.65)),
                "hand_l": (int(w * 0.32), int(h * 0.65)),
                "head": (int(w * 0.50), int(h * 0.22)),
                "back": (int(w * 0.50), int(h * 0.50)),
            }

        # Neural inference
        model, has_trained = cls.get_or_load_model(device=device)
        if engine_mode == "untrained_model" or not has_trained:
            if engine_mode == "untrained_model":
                model = SocketPredictor().to(device)
                model.eval()

        # Prepare tensor (1, 4, 32, 32)
        norm_img = rgba_image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(tensor)[0].cpu().numpy()  # (NUM_SOCKETS, 2)

        sockets = {}
        for idx, name in enumerate(SOCKET_NAMES):
            sx = int(round(preds[idx, 0] * (w - 1)))
            sy = int(round(preds[idx, 1] * (h - 1)))
            sockets[name] = (max(0, min(w - 1, sx)), max(0, min(h - 1, sy)))

        return sockets
