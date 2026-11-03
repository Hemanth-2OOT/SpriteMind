"""Neural Visual Critic architecture for pixel art defect localization and classification."""
import math
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from spritemind.ml.svan import ResidualBlock2D


DEFECT_CLASSES = [
    "clean",
    "stray_pixel",
    "disconnected_part",
    "palette_leak",
    "missing_limb",
    "asymmetry",
    "unexpected_hole",
]

SEVERITY_CLASSES = ["minor", "moderate", "critical"]


class SpatialHeatmapDecoder(nn.Module):
    """Upsamples 8x8 spatial features to 32x32 pixel defect localization heatmap."""

    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 64, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)  # (B, 1, 32, 32)


class NeuralVisualCritic(nn.Module):
    """Deep visual critic that predicts defect presence, category, spatial heatmap, and quality."""

    def __init__(self, in_channels: int = 4, num_classes: int = len(DEFECT_CLASSES)):
        super().__init__()
        self.num_classes = num_classes

        # ConvNet Backbone (32x32 -> 8x8)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock2D(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock2D(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Defect Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

        # Severity Head
        self.severity_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        # Overall Quality Head (0 - 100)
        self.quality_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Spatial Defect Localization Heatmap
        self.heatmap_decoder = SpatialHeatmapDecoder(in_channels=128)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, 4, 32, 32) normalized in [0, 1]
        feats = self.stem(x)  # (B, 128, 8, 8)
        pooled = self.pool(feats).flatten(1)  # (B, 128)

        defect_logits = self.classifier(pooled)       # (B, num_classes)
        severity_logits = self.severity_head(pooled)  # (B, 3)
        quality = self.quality_head(pooled) * 100.0   # (B, 1) in [0, 100]
        heatmap = self.heatmap_decoder(feats)         # (B, 1, 32, 32) in [0, 1]

        return {
            "defect_logits": defect_logits,
            "severity_logits": severity_logits,
            "quality_score": quality,
            "heatmap": heatmap,
        }
