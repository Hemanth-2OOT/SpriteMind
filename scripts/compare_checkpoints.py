"""Renders a side-by-side sprite sheet for old vs new SVAN checkpoints plus quality metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import sys
backend = Path(__file__).resolve().parent.parent / "backend"
if str(backend) not in sys.path:
    sys.path.insert(0, str(backend))

from spritemind.dataset.generator import CATEGORIES
from spritemind.ml.pipeline import SpriteMindPipeline
from spritemind.ml.svan import SVANModel
from spritemind.pixelscript.validator import PixelScriptValidator
from spritemind.renderer.engine import PixelScriptEngine

DEFAULT_CONFIG = {"vocab_size": 401, "d_model": 128, "nhead": 4, "num_layers": 4,
                  "dim_feedforward": 512, "max_seq_len": 448, "dropout": 0.1}


def load_legacy(path: Path, device="cpu"):
    """Load any SVAN checkpoint by reading the architecture it was trained with."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    config = dict(DEFAULT_CONFIG)
    config.update((ckpt.get("metadata") or {}).get("model_config") or {})
    pe = state.get("action_transformer.pos_encoder.pe")
    if pe is not None:
        config["max_seq_len"] = int(pe.shape[1])
    model = SVANModel.from_config(config)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def summarize(pipe, model, palettes, seeds, best_of):
    rows, valid, coverage, colors, parts = [], 0, [], [], []
    for cat in CATEGORIES:
        for s in seeds:
            prog, _source = pipe._neural_generate(
                model=model, category=cat, style="classic_16bit", pose="idle",
                palette="pico8", max_actions=64, seed=s, width=32, height=32, best_of=best_of,
            )
            report = PixelScriptValidator.validate(prog)
            valid += int(report.is_valid)
            res = PixelScriptEngine.execute(prog, width=32, height=32, palette_name="pico8")
            rgba = res.composite_array
            alpha = rgba[..., 3] > 0
            coverage.append(float(alpha.mean()))
            uniq = len({tuple(c) for c in rgba[alpha][:, :3]}) if alpha.any() else 0
            colors.append(uniq)
            parts.append(len({a.params.get("name") for a in prog.actions if a.action_type.value == "PART"}))
            rows.append((cat, s, rgba))
    n = len(rows)
    return rows, {
        "samples": n,
        "valid_rate": round(100.0 * valid / n, 1),
        "mean_coverage": round(float(np.mean(coverage)), 3),
        "mean_unique_colors": round(float(np.mean(colors)), 2),
        "mean_parts": round(float(np.mean(parts)), 2),
    }


def sheet(rows, cell=64, pad=4):
    cols = len(rows)
    img = Image.new("RGBA", (cols * (cell + pad) + pad, cell + 2 * pad + 12), (24, 24, 32, 255))
    d = ImageDraw.Draw(img)
    for i, (cat, s, rgba) in enumerate(rows):
        tile = Image.fromarray(rgba).resize((cell, cell), Image.NEAREST)
        x = pad + i * (cell + pad)
        img.alpha_composite(tile, (x, pad + 12))
        d.text((x, 2), f"{cat[:6]}/{s}", fill=(220, 220, 230, 255))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="/tmp/svan_checkpoint_prev.pt")
    ap.add_argument("--new", default="models/checkpoints/svan_checkpoint_best.pt")
    ap.add_argument("--out", default="data/reports/checkpoint_comparison.png")
    args = ap.parse_args()

    pipe = SpriteMindPipeline()
    results = {}
    imgs = []
    for label, best_of in (("old", 1), ("new", 1), ("new_bestof4", 4)):
        if label.startswith("old"):
            model = load_legacy(Path(args.old), pipe.device)
        else:
            assert pipe.load_checkpoint(Path(args.new).name)
            model = pipe.trained_model
        rows, metrics = summarize(pipe, model, ["pico8"], [11, 23], best_of)
        results[label] = metrics
        imgs.append((label, sheet(rows)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    height = sum(im.height + 18 for _, im in imgs)
    width = max(im.width for _, im in imgs)
    canvas = Image.new("RGBA", (width, height), (16, 16, 20, 255))
    d = ImageDraw.Draw(canvas)
    y = 0
    for label, im in imgs:
        d.text((4, y + 2), label, fill=(255, 210, 120, 255))
        canvas.alpha_composite(im, (0, y + 16))
        y += im.height + 18
    canvas.save(out)
    print(json.dumps(results, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
