"""Learned rig parameters, conditioned on the prompt (experiment).

Today ``TemporalDatasetGenerator.compute_motion_offsets`` returns the same joint
offsets for every prompt: a knight and a lanky zombie walk with identical
motion. This module trains a small head that predicts those offsets from
(action, phase, category, prompt style tags) so the same action can read
differently per character.

Honest scope: the training targets are produced by the analytic generator plus
explicit per-tag modulation rules. The head therefore learns to *interpolate and
smooth* an authored mapping, not a new motion prior. It is not wired into the
production animation path; see ``scripts/eval_rig_params.py`` for the measured
comparison.
"""
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from spritemind.dataset.generator import CATEGORIES
from spritemind.dataset.temporal_dataset import TemporalDatasetGenerator

PARTS: List[str] = [
    "torso",
    "head",
    "arm_left",
    "arm_right",
    "leg_left",
    "leg_right",
    "weapon",
]

ACTIONS: List[str] = [
    "idle",
    "walk",
    "run",
    "jump",
    "fall",
    "dash",
    "melee_attack",
    "ranged_attack",
    "cast",
    "blocking",
    "special_ability",
    "death",
    "zombie_movement",
    "skeleton_walking",
    "slime_bouncing",
    "robot_movement",
    "animal_running",
    "flying_creatures",
]

STYLE_TAGS: Tuple[str, ...] = ("heavy", "nimble", "floating", "undead", "armored", "casting")

_KEYWORDS = {
    "heavy": ("brute", "heavy", "giant", "ogre", "golem", "bulky", "armoured tank", "hulk"),
    "nimble": ("agile", "nimble", "rogue", "assassin", "quick", "slim", "ninja", "dancer"),
    "floating": ("ghost", "wraith", "flying", "hover", "float", "spirit", "drone", "spectre"),
    "undead": ("zombie", "skeleton", "undead", "ghoul", "lich", "corpse"),
    "armored": ("armor", "armour", "plate", "shield", "knight", "paladin", "mech"),
    "casting": ("mage", "wizard", "sorcer", "witch", "staff", "caster", "shaman", "druid"),
}

MAX_DX = 3
MAX_DY = 3
MAX_ROT = 25


def tags_from_prompt(prompt: str) -> Dict[str, float]:
    """Soft style tags: 1.0 when a keyword is present, else 0.0."""
    text = (prompt or "").lower()
    return {tag: (1.0 if any(word in text for word in words) else 0.0) for tag, words in _KEYWORDS.items()}


def base_offsets(action: str, phase: float, category: str) -> Dict[str, Tuple[int, int, int]]:
    return TemporalDatasetGenerator.compute_motion_offsets(action, phase, category=category)


def modulate(
    offsets: Dict[str, Tuple[int, int, int]], tags: Dict[str, float]
) -> Dict[str, Tuple[int, int, int]]:
    """Applies the authored prompt-style rules to analytic offsets."""
    heavy = tags.get("heavy", 0.0)
    nimble = tags.get("nimble", 0.0)
    floating = tags.get("floating", 0.0)
    undead = tags.get("undead", 0.0)
    armored = tags.get("armored", 0.0)
    casting = tags.get("casting", 0.0)

    amp = 1.0 + 0.30 * heavy + 0.15 * nimble
    rot_amp = 1.0 + 0.25 * heavy - 0.35 * armored
    out: Dict[str, Tuple[int, int, int]] = {}
    for part, (dx, dy, rot) in offsets.items():
        ndx = dx * amp
        ndy = dy * amp
        nrot = rot * rot_amp
        if floating:
            if part.startswith("leg"):
                ndy -= 1.0
            ndy -= 0.5
        if undead:
            ndx += 1.0 if part in ("head", "arm_left") else 0.0
            nrot *= 0.6
        if casting and part in ("arm_right", "weapon"):
            ndy -= 1.0
            nrot += 6.0
        out[part] = (
            int(max(-MAX_DX, min(MAX_DX, round(ndx)))),
            int(max(-MAX_DY, min(MAX_DY, round(ndy)))),
            int(max(-MAX_ROT, min(MAX_ROT, round(nrot)))),
        )
    return out


def feature_vector(
    action: str, phase: float, category: str, tags: Dict[str, float]
) -> List[float]:
    vec: List[float] = []
    vec.extend(1.0 if action == a else 0.0 for a in ACTIONS)
    vec.append(math.sin(2.0 * math.pi * phase))
    vec.append(math.cos(2.0 * math.pi * phase))
    vec.append(2.0 * phase - 1.0)
    vec.extend(1.0 if category == c else 0.0 for c in CATEGORIES)
    vec.extend(float(tags.get(t, 0.0)) for t in STYLE_TAGS)
    return vec


FEATURE_DIM = len(ACTIONS) + 3 + len(CATEGORIES) + len(STYLE_TAGS)
OUTPUT_DIM = len(PARTS) * 3


def target_vector(action: str, phase: float, category: str, tags: Dict[str, float]) -> List[float]:
    shaped = modulate(base_offsets(action, phase, category), tags)
    flat: List[float] = []
    for part in PARTS:
        flat.extend(float(v) for v in shaped.get(part, (0, 0, 0)))
    return flat


class RigParameterHead(nn.Module):
    """Maps prompt/style conditioning to per-part (dx, dy, rot)."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, OUTPUT_DIM),
        )
        self.register_buffer("scale", torch.tensor([MAX_DX, MAX_DY, MAX_ROT] * len(PARTS), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x)) * self.scale

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _build_dataset(samples: int, seed: int, phases: Sequence[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    import random

    rng = random.Random(seed)
    tag_names = list(STYLE_TAGS)
    xs: List[List[float]] = []
    ys: List[List[float]] = []
    for _ in range(samples):
        action = rng.choice(ACTIONS)
        category = rng.choice(CATEGORIES)
        phase = rng.choice(phases)
        tags = {t: (1.0 if rng.random() < 0.25 else 0.0) for t in tag_names}
        xs.append(feature_vector(action, phase, category, tags))
        ys.append(target_vector(action, phase, category, tags))
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)


def train(
    steps: int = 600,
    samples: int = 4096,
    seed: int = 0,
    hidden: int = 64,
    lr: float = 3e-3,
    phases: Optional[Sequence[float]] = None,
) -> Tuple[RigParameterHead, Dict[str, float]]:
    torch.manual_seed(seed)
    phases = list(phases or [i / 7.0 for i in range(8)])
    x, y = _build_dataset(samples, seed, phases)
    split = int(len(x) * 0.85)
    model = RigParameterHead(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for step in range(steps):
        opt.zero_grad()
        pred = model(x[:split])
        loss = loss_fn(pred, y[:split])
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_loss = float(loss_fn(model(x[:split]), y[:split]))
        eval_loss = float(loss_fn(model(x[split:]), y[split:]))
        mae = float((model(x[split:]) - y[split:]).abs().mean())
    metrics = {"train_mse": train_loss, "held_out_mse": eval_loss, "held_out_mae": mae}
    return model, metrics


def predict(
    model: RigParameterHead,
    action: str,
    phase: float,
    category: str,
    prompt: str = "",
) -> Dict[str, Tuple[int, int, int]]:
    tags = tags_from_prompt(prompt)
    feats = torch.tensor([feature_vector(action, phase, category, tags)], dtype=torch.float32)
    with torch.no_grad():
        out = model(feats)[0].tolist()
    result: Dict[str, Tuple[int, int, int]] = {}
    for i, part in enumerate(PARTS):
        dx, dy, rot = out[i * 3 : i * 3 + 3]
        result[part] = (
            int(max(-MAX_DX, min(MAX_DX, round(dx)))),
            int(max(-MAX_DY, min(MAX_DY, round(dy)))),
            int(max(-MAX_ROT, min(MAX_ROT, round(rot)))),
        )
    return result


_HEAD: Optional[RigParameterHead] = None


def get_head(steps: int = 800, samples: int = 4096, seed: int = 0) -> RigParameterHead:
    """Trains the head once per process; deterministic because the seed is fixed."""
    global _HEAD
    if _HEAD is None:
        model, _ = train(steps=steps, samples=samples, seed=seed)
        model.eval()
        _HEAD = model
    return _HEAD


def offsets_for(action: str, phase: float, category: str, prompt: str = "") -> Dict[str, Tuple[int, int, int]]:
    """Offsets for one frame: learned per-part motion when available, else the analytic prior."""
    try:
        return predict(get_head(), action, phase, category, prompt)
    except Exception:
        return base_offsets(action, phase, category)


def offset_diversity(offsets: Iterable[Dict[str, Tuple[int, int, int]]]) -> float:
    """Mean pairwise L2 distance between offsets; 0 means the prompt changes nothing."""
    items = list(offsets)
    if len(items) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            total += sum(
                abs(a - b)
                for part in PARTS
                for a, b in zip(items[i].get(part, (0, 0, 0)), items[j].get(part, (0, 0, 0)))
            )
            pairs += 1
    return total / pairs
