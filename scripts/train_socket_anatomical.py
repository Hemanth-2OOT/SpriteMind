"""Trains SocketPredictor on anatomical labels (R7) and compares it to the shipped weights.

    python scripts/train_socket_anatomical.py --samples 1500 --epochs 15

Writes `models/checkpoints/socket_predictor_anatomical_v2.pt`; the production
`socket_predictor_best.pt` is never touched. Evaluation reports mean pixel error
against anatomical labels for three arms on the same held-out samples: the
procedural baseline, the shipped checkpoint and the new checkpoint.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from spritemind.dataset.anatomical_sockets import SOCKET_RULE, anatomical_sockets  # noqa: E402
from spritemind.dataset.composition_dataset import CompositionDataset  # noqa: E402
from spritemind.dataset.generator import CATEGORIES, POSES, PROPORTIONS, CharacterGenerator  # noqa: E402
from spritemind.ml.composition_model import SOCKET_NAMES, SocketPredictor  # noqa: E402
from spritemind.pixelscript.serializer import PixelScriptSerializer  # noqa: E402
from spritemind.renderer.engine import PixelScriptEngine  # noqa: E402
from spritemind.utils.palettes import PALETTES  # noqa: E402

CHECKPOINTS = ROOT / "models" / "checkpoints"


def build_samples(total: int, seed: int, val_fraction: float = 0.15):
    rng = random.Random(seed)
    palettes = list(PALETTES)
    samples = []
    for i in range(total):
        category = CATEGORIES[i % len(CATEGORIES)]
        pose = rng.choice(POSES)
        proportion = rng.choice(PROPORTIONS)
        sample_seed = seed + i * 13
        prog = CharacterGenerator.generate_character(
            category=category,
            pose=pose,
            proportion=proportion,
            palette_name=palettes[i % len(palettes)],
            seed=sample_seed,
        )
        result = PixelScriptEngine.execute(PixelScriptSerializer.serialize_program(prog), width=32, height=32)
        samples.append({
            "id": f"anat_{i:05d}",
            "category": category,
            "pose": pose,
            "proportion": proportion,
            "sockets": anatomical_sockets(result.canvas),
            "rgba_array": result.composite_array,
        })
    cut = int(total * (1.0 - val_fraction))
    return samples[:cut], samples[cut:]


def predict(model: SocketPredictor, rgba: np.ndarray) -> dict:
    tensor = torch.from_numpy(rgba.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)
    with torch.inference_mode():
        preds = model(tensor)[0].numpy()
    out = {}
    for idx, name in enumerate(SOCKET_NAMES):
        out[name] = (float(preds[idx, 0] * 31.0), float(preds[idx, 1] * 31.0))
    return out


def proportional_baseline() -> dict:
    return {
        "hand_r": (32 * 0.68, 32 * 0.65),
        "hand_l": (32 * 0.32, 32 * 0.65),
        "head": (32 * 0.50, 32 * 0.22),
        "back": (32 * 0.50, 32 * 0.50),
    }


def evaluate(predictor, samples) -> dict:
    errors = {name: [] for name in SOCKET_NAMES}
    for sample in samples:
        preds = predictor(sample["rgba_array"])
        for name in SOCKET_NAMES:
            truth = np.asarray(sample["sockets"][name], dtype=float)
            errors[name].append(float(np.linalg.norm(np.asarray(preds[name]) - truth)))
    flat = [value for values in errors.values() for value in values]
    return {
        "mean_px_error": round(float(np.mean(flat)), 3),
        "max_px_error": round(float(np.max(flat)), 3),
        "per_socket_px": {name: round(float(np.mean(values)), 3) for name, values in errors.items()},
    }


def load_checkpoint(path: Path) -> SocketPredictor:
    model = SocketPredictor()
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--save-name", default="socket_predictor_anatomical_v2.pt")
    parser.add_argument("--baseline-name", default="socket_predictor_best.pt")
    parser.add_argument("--out", default="data/benchmarks/socket_anatomical.json")
    args = parser.parse_args()

    started = time.perf_counter()
    train_samples, val_samples = build_samples(args.samples, args.seed)
    print(f"train {len(train_samples)} / val {len(val_samples)}", flush=True)

    train_ds = CompositionDataset(train_samples)
    val_ds = CompositionDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    torch.manual_seed(args.seed)
    model = SocketPredictor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch["image"]), batch["targets"])
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch["image"])
        model.eval()
        val = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                val += float(criterion(model(batch["image"]), batch["targets"]).item()) * len(batch["image"])
        train_loss = total / len(train_ds)
        val_loss = val / len(val_ds)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_mean_px": round(float(np.sqrt(val_loss) * 32.0), 3),
        })
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:2d} train {train_loss:.5f} val {val_loss:.5f}", flush=True)

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    save_path = CHECKPOINTS / args.save_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "metadata": {
            "labels": "anatomical part geometry",
            "label_rule": SOCKET_RULE,
            "samples": args.samples,
            "epochs": args.epochs,
            "seed": args.seed,
            "best_val_loss": round(best_val, 6),
        },
    }, save_path)

    baseline_path = CHECKPOINTS / args.baseline_name
    results = {"samples": len(val_samples), "label_rule": SOCKET_RULE, "history": history}
    results["procedural_baseline"] = evaluate(lambda rgba: proportional_baseline(), val_samples)
    if baseline_path.exists():
        baseline_model = load_checkpoint(baseline_path)
        results["shipped_checkpoint"] = evaluate(lambda rgba: predict(baseline_model, rgba), val_samples)
    results["anatomical_checkpoint"] = evaluate(lambda rgba: predict(model, rgba), val_samples)
    results["checkpoint"] = args.save_name
    results["elapsed_seconds"] = round(time.perf_counter() - started, 1)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for arm, metrics in results.items():
        if isinstance(metrics, dict) and "mean_px_error" in metrics:
            print(f"{arm:24s} mean {metrics['mean_px_error']:.3f} px  per-socket {metrics['per_socket_px']}")
    print(f"saved: {save_path}")
    print(f"results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
