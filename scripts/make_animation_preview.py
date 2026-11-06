"""Renders animation GIFs (walk, run, melee_attack, jump) for trained vs procedural engines."""
import argparse
import json
from pathlib import Path

import base64
import io

import numpy as np
from PIL import Image

import sys
backend = Path(__file__).resolve().parent.parent / "backend"
if str(backend) not in sys.path:
    sys.path.insert(0, str(backend))

from spritemind.ml.temporal import TemporalAnimator

ANIMS = ["walk", "run", "melee_attack", "jump"]


def _frame_image(f) -> Image.Image:
    payload = f["image_base64"].split(",", 1)[-1]
    raw = base64.b64decode(payload)
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def frames_to_gif(frames, out_path: Path, scale: int = 4, fps: int = 8):
    imgs = []
    for f in frames:
        img = _frame_image(f)
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        canvas = Image.new("RGBA", img.size, (18, 18, 26, 255))
        canvas.alpha_composite(img)
        imgs.append(canvas.convert("P", palette=Image.ADAPTIVE))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
    return out_path


def strip(frames, scale: int = 3, gap: int = 2):
    tiles = []
    for f in frames:
        img = _frame_image(f)
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        canvas = Image.new("RGBA", img.size, (18, 18, 26, 255))
        canvas.alpha_composite(img)
        tiles.append(canvas)
    w = sum(t.width + gap for t in tiles) + gap
    out = Image.new("RGBA", (w, tiles[0].height + 2 * gap), (18, 18, 26, 255))
    x = gap
    for t in tiles:
        out.alpha_composite(t, (x, gap))
        x += t.width + gap
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="knight")
    ap.add_argument("--out-dir", default="data/reports/animations")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    summary = {}
    strips = []
    for anim in ANIMS:
        for engine in ("procedural_baseline", "trained_model"):
            res = TemporalAnimator.generate_animation(
                category=args.category, anim_category=anim, num_frames=8,
                engine_mode=engine, seed=7, palette_name="pico8",
            )
            name = f"{args.category}_{anim}_{engine}.gif"
            frames_to_gif(res["frames"], out_dir / name, fps=res["fps"])
            summary.setdefault(anim, {})[engine] = {
                "engine_used": res["engine_used"],
                "frames": res["num_frames"],
                "identity_confidence": res["identity_confidence"],
            }
            strips.append((f"{anim} / {engine}", strip(res["frames"])))
    canvas_h = sum(s.height + 16 for _, s in strips)
    canvas_w = max(s.width for _, s in strips)
    from PIL import ImageDraw
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (16, 16, 20, 255))
    d = ImageDraw.Draw(canvas)
    y = 0
    for label, s in strips:
        d.text((4, y + 2), label, fill=(240, 220, 160, 255))
        canvas.alpha_composite(s, (0, y + 14))
        y += s.height + 16
    canvas.save(out_dir / f"{args.category}_strips.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
