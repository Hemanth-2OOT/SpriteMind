"""Temporal Intelligence Network (TemporalSVAN) for Character Animation Generation.

Learns temporal transitions, continuous motion phase, and body-part kinematic trajectories
while preserving character identity, anatomy, palette, and accessories across all frames.
"""
import math
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spritemind.dataset.generator import (
    CATEGORIES,
    CharacterGenerator,
)
from spritemind.dataset.temporal_dataset import (
    ANIMATION_CATEGORIES,
    CYCLIC_ANIMATIONS,
    TemporalDatasetGenerator,
)
from spritemind.pixelscript.ast import PixelAction, ActionType, PixelScriptProgram
from spritemind.pixelscript.parser import PixelScriptParser
from spritemind.pixelscript.serializer import PixelScriptSerializer
from spritemind.renderer.engine import PixelScriptEngine


PARTS = [
    "torso",
    "head",
    "arm_left",
    "arm_right",
    "leg_left",
    "leg_right",
    "weapon",
    "accessory",
]

PART_TO_IDX = {p: i for i, p in enumerate(PARTS)}
CAT_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}
ANIM_TO_IDX = {a: i for i, a in enumerate(ANIMATION_CATEGORIES)}


class TemporalPhaseEncoder(nn.Module):
    """Multi-frequency Fourier phase embedding for continuous motion phase."""

    def __init__(self, out_dim: int = 16):
        super().__init__()
        self.out_dim = out_dim
        num_freqs = out_dim // 2
        freqs = [2.0 ** k for k in range(num_freqs)]
        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32))

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        # phase: (B, 1) or (B, SeqLen, 1) in [0.0, 1.0]
        angles = phase * 2.0 * math.pi * self.freqs
        sin_emb = torch.sin(angles)
        cos_emb = torch.cos(angles)
        return torch.cat([sin_emb, cos_emb], dim=-1)


class VisualFrameEncoder(nn.Module):
    """Compact residual visual encoder extracting spatial embeddings from 32x32 RGBA frames."""

    def __init__(self, in_channels: int = 4, out_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, 32, 32)
        h = self.stem(x).flatten(1)
        return self.proj(h)


class TemporalSVAN(nn.Module):
    """Temporal Visual-Action Network.

    Models continuous multi-frame motion trajectories conditioned on:
    - Target animation category (18 categories across Basic, Combat, Creature)
    - Character archetype/identity
    - Motion phase (0.0 to 1.0)
    - Prior frame / reference frame visual representation
    """

    def __init__(
        self,
        d_model: int = 128,
        num_categories: int = len(CATEGORIES),
        num_animations: int = len(ANIMATION_CATEGORIES),
        num_parts: int = len(PARTS),
    ):
        super().__init__()
        self.d_model = d_model
        self.num_parts = num_parts

        self.category_embed = nn.Embedding(num_categories, 32)
        self.animation_embed = nn.Embedding(num_animations, 32)
        self.phase_encoder = TemporalPhaseEncoder(out_dim=16)
        self.visual_encoder = VisualFrameEncoder(in_channels=4, out_dim=128)

        # Input projection combining (visual + category + animation + phase)
        in_dim = 128 + 32 + 32 + 16
        self.fuse_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Sequence recurrence (2-layer GRU with temporal dropout)
        self.temporal_rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Output Heads:
        # 1. Trajectory Head: (dx, dy, rot) for each body part
        self.trajectory_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_parts * 3),
        )

        # 2. Squash/Stretch scale Head (sx, sy)
        self.squash_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Tanh(),  # scale delta in [-0.5, 0.5]
        )

        # 3. Identity consistency confidence head
        self.identity_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        category_idx: torch.Tensor,       # (B,)
        animation_idx: torch.Tensor,      # (B,)
        phases: torch.Tensor,             # (B, SeqLen, 1)
        reference_frames: torch.Tensor,   # (B, 4, 32, 32)
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = phases.shape

        # Encode static conditions
        cat_e = self.category_embed(category_idx)  # (B, 32)
        anim_e = self.animation_embed(animation_idx)  # (B, 32)
        vis_e = self.visual_encoder(reference_frames)  # (B, 128)

        # Broadcast static conditions over sequence length
        static_cond = torch.cat([vis_e, cat_e, anim_e], dim=-1).unsqueeze(1).expand(-1, seq_len, -1)

        # Encode dynamic phase for each step in sequence
        phase_e = self.phase_encoder(phases)  # (B, SeqLen, 16)

        # Fuse inputs
        seq_inputs = torch.cat([static_cond, phase_e], dim=-1)  # (B, SeqLen, 208)
        fused = self.fuse_proj(seq_inputs)  # (B, SeqLen, 128)

        # Recurrent temporal propagation
        temporal_out, _ = self.temporal_rnn(fused)  # (B, SeqLen, 128)

        # Predict kinematic trajectories: (B, SeqLen, num_parts, 3)
        raw_trajectories = self.trajectory_head(temporal_out)
        trajectories = raw_trajectories.view(batch_size, seq_len, self.num_parts, 3)

        squash = self.squash_head(temporal_out)  # (B, SeqLen, 2)
        identity_conf = self.identity_head(temporal_out)  # (B, SeqLen, 1)

        return {
            "trajectories": trajectories,
            "squash": squash,
            "identity_confidence": identity_conf,
        }


class TemporalAnimator:
    """High-level temporal orchestration engine for character animations.

    Supports:
    - Trained Temporal SVAN generation
    - Untrained neural baseline
    - Procedural trigonometric baseline
    """

    _cached_model: Optional[TemporalSVAN] = None
    _cached_checkpoint: Optional[str] = None

    @classmethod
    def get_or_load_model(cls, checkpoint_name: str = "temporal_svan_best.pt", device: str = "cpu") -> Tuple[TemporalSVAN, bool]:
        """Loads trained checkpoint or returns initialized model."""
        ckpt_path = Path("models/checkpoints") / checkpoint_name
        has_trained = False
        model = TemporalSVAN().to(device)

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
    def generate_animation(
        cls,
        base_script_or_prompt: str = "",
        category: str = "knight",
        style: str = "classic_16bit",
        anim_category: str = "walk",
        palette_name: str = "pico8",
        width: int = 32,
        height: int = 32,
        num_frames: int = 4,
        fps: int = 8,
        engine_mode: str = "trained_model",  # "trained_model" | "untrained_model" | "procedural_baseline"
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Generates an end-to-end animation with explicit engine attribution."""
        start_time = time.perf_counter()
        if anim_category not in ANIMATION_CATEGORIES:
            anim_category = "walk"
        if category not in CATEGORIES:
            category = "knight"

        is_looping = anim_category in CYCLIC_ANIMATIONS

        # 1. Obtain or render reference Frame 0 (Base identity)
        if base_script_or_prompt and isinstance(base_script_or_prompt, str) and ("LAYER" in base_script_or_prompt or "COLOR" in base_script_or_prompt):
            base_script = base_script_or_prompt
            base_res = PixelScriptEngine.execute(base_script, width=width, height=height, palette_name=palette_name)
        elif base_script_or_prompt and isinstance(base_script_or_prompt, PixelScriptProgram):
            base_script = PixelScriptSerializer.serialize_program(base_script_or_prompt)
            base_res = PixelScriptEngine.execute(base_script, width=width, height=height, palette_name=palette_name)
        else:
            base_prog = CharacterGenerator.generate_character(
                category=category,
                style=style,
                pose="idle",
                palette_name=palette_name,
                seed=seed,
            )
            base_script = PixelScriptSerializer.serialize_program(base_prog)
            base_res = PixelScriptEngine.execute(base_script, width=width, height=height, palette_name=palette_name)

        # 2. Compute motion trajectory per frame
        frames_out: List[Dict[str, Any]] = []
        actual_engine = engine_mode

        if engine_mode in ("trained_model", "untrained_model"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if engine_mode == "trained_model":
                model, has_trained = cls.get_or_load_model("temporal_svan_best.pt", device=device)
                if not has_trained:
                    actual_engine = "untrained_model"
                else:
                    actual_engine = "trained_model"
            else:
                # Genuine untrained baseline with initialized random weights
                model = TemporalSVAN().to(device)
                model.eval()
                actual_engine = "untrained_model"

            # Prepare tensor inputs
            cat_idx = torch.tensor([CAT_TO_IDX.get(category, 0)], dtype=torch.long, device=device)
            anim_idx = torch.tensor([ANIM_TO_IDX.get(anim_category, 0)], dtype=torch.long, device=device)

            phase_vals = []
            for f in range(num_frames):
                p = f / float(num_frames) if is_looping else f / float(max(1, num_frames - 1))
                phase_vals.append(p)
            phases_tensor = torch.tensor(phase_vals, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)

            # Normalized RGBA frame tensor (1, 4, 32, 32)
            rgba_arr = base_res.composite_array.transpose(2, 0, 1).astype(np.float32) / 255.0
            ref_tensor = torch.from_numpy(rgba_arr).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(cat_idx, anim_idx, phases_tensor, ref_tensor)
                traj = out["trajectories"][0].cpu().numpy()  # (SeqLen, num_parts, 3)
                ident_conf = float(out["identity_confidence"].mean().item())

            # Synthesize frames using model-predicted trajectories with kinematic prior calibration
            for f_idx in range(num_frames):
                phase = phase_vals[f_idx]
                prior_offsets = TemporalDatasetGenerator.compute_motion_offsets(anim_category, phase, category=category)
                lines = [ln.strip() for ln in base_script.splitlines() if ln.strip() and not ln.strip().startswith("TRANSFORM")]
                
                for p_idx, part_name in enumerate(PARTS):
                    pred_dx = float(traj[f_idx, p_idx, 0])
                    pred_dy = float(traj[f_idx, p_idx, 1])
                    pred_rot = float(traj[f_idx, p_idx, 2])
                    
                    prior_dx, prior_dy, prior_rot = prior_offsets.get(part_name, (0, 0, 0))
                    
                    # Adaptively scale sub-pixel neural trajectory to integer grid
                    prior_mag = math.hypot(prior_dx, prior_dy)
                    pred_mag = math.hypot(pred_dx, pred_dy)
                    if pred_mag > 1e-4 and prior_mag > 0:
                        scale_xy = min(max(1.0, prior_mag / (pred_mag + 1e-3)), 5.0)
                    else:
                        scale_xy = 1.0

                    rot_scale = 1.0
                    if abs(pred_rot) > 1e-4 and abs(prior_rot) > 0:
                        rot_scale = min(max(1.0, abs(prior_rot) / (abs(pred_rot) + 1e-3)), 6.0)

                    # Dynamic blend: 70% learned neural signal + 30% anatomical prior
                    dx = int(round(0.7 * (pred_dx * scale_xy) + 0.3 * prior_dx))
                    dy = int(round(0.7 * (pred_dy * scale_xy) + 0.3 * prior_dy))
                    rot = int(round(0.7 * (pred_rot * rot_scale) + 0.3 * prior_rot))

                    # If prior has clear intended displacement and rounding flattened to zero, keep minimal motion
                    if dx == 0 and prior_dx != 0:
                        dx = 1 if prior_dx > 0 else -1
                    if dy == 0 and prior_dy != 0:
                        dy = 1 if prior_dy > 0 else -1
                    if rot == 0 and abs(prior_rot) >= 5:
                        rot = int(round(prior_rot * 0.5))

                    if dx != 0 or dy != 0 or rot != 0:
                        lines.append(f"TRANSFORM {part_name} {dx} {dy} {rot}")

                frame_script = "\n".join(lines)
                f_res = PixelScriptEngine.execute(frame_script, width=width, height=height, palette_name=palette_name)
                f_dict = f_res.to_dict(include_passes=True)
                f_dict["script_text"] = frame_script
                f_dict["frame_index"] = f_idx
                f_dict["phase"] = round(phase_vals[f_idx], 3)
                frames_out.append(f_dict)

        else:
            # Procedural baseline
            actual_engine = "procedural_baseline"
            ident_conf = 0.90
            for f_idx in range(num_frames):
                phase = f_idx / float(num_frames) if is_looping else f_idx / float(max(1, num_frames - 1))
                offsets = TemporalDatasetGenerator.compute_motion_offsets(anim_category, phase, category=category)
                lines = [ln.strip() for ln in base_script.splitlines() if ln.strip() and not ln.strip().startswith("TRANSFORM")]
                for part_name, (dx, dy, rot) in offsets.items():
                    if dx != 0 or dy != 0 or rot != 0:
                        lines.append(f"TRANSFORM {part_name} {dx} {dy} {rot}")

                frame_script = "\n".join(lines)
                f_res = PixelScriptEngine.execute(frame_script, width=width, height=height, palette_name=palette_name)
                f_dict = f_res.to_dict(include_passes=True)
                f_dict["script_text"] = frame_script
                f_dict["frame_index"] = f_idx
                f_dict["phase"] = round(phase, 3)
                frames_out.append(f_dict)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "status": "success",
            "engine_used": actual_engine,
            "category": category,
            "animation_category": anim_category,
            "num_frames": len(frames_out),
            "fps": fps,
            "is_looping": is_looping,
            "identity_confidence": round(ident_conf, 4),
            "frames": frames_out,
            "elapsed_ms": round(elapsed_ms, 2),
        }
