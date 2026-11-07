"""Measures silhouette coherence of generated sprites.

Two defect classes the user reported are covered here:

* detached pixels -- alpha components that do not touch the main body
* missing outline  -- silhouette pixels with no darker pixel beside them

Run:  python scripts/coherence_metrics.py --limit 24
"""
import argparse
import base64
import io
import json
import sys

from collections import deque
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from spritemind.ml.pipeline import SpriteMindPipeline  # noqa: E402
from spritemind.ml.temporal import TemporalAnimator  # noqa: E402
from spritemind.renderer.continuity import continuity_report, masks_from_frames  # noqa: E402

PROMPTS_PATH = Path("data/benchmarks/generalization_prompts.json")


def decode(frame: Dict[str, Any]) -> np.ndarray:
    raw = frame["image_base64"].split(",", 1)[-1]
    return np.array(Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGBA"))


def components(alpha: np.ndarray) -> List[int]:
    seen = np.zeros_like(alpha, dtype=bool)
    sizes: List[int] = []
    h, w = alpha.shape
    for y in range(h):
        for x in range(w):
            if not alpha[y, x] or seen[y, x]:
                continue
            size = 0
            queue = deque([(y, x)])
            seen[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and alpha[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            sizes.append(size)
    return sorted(sizes, reverse=True)


def outline_coverage(rgba: np.ndarray) -> float:
    """Share of silhouette pixels that are a dark rim around a brighter body."""
    alpha = rgba[..., 3] > 0
    if not alpha.any():
        return 0.0
    luma = rgba[..., :3].astype(np.int16).sum(axis=2)
    interior = _erode(alpha)
    body = float(luma[alpha].mean())
    rim = alpha & luma <= 0.55 * body
    touched = np.zeros_like(alpha)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        touched |= np.roll(rim, (dy, dx), axis=(0, 1))
    border = alpha & ~interior
    return float((border & (rim | touched)).sum() / max(1, border.sum()))


def _erode(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        out &= np.roll(mask, (dy, dx), axis=(0, 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--engine", default="trained_model")
    ap.add_argument("--out", default="data/benchmarks/coherence_metrics.json")
    ap.add_argument("--anim-frames", type=int, default=4, help="frames per animation used for continuity")
    ap.add_argument("--skip-continuity", action="store_true")
    ap.add_argument("--checkpoint", default="svan_checkpoint_best.pt")
    args = ap.parse_args()

    prompts = json.loads(PROMPTS_PATH.read_text()) if PROMPTS_PATH.exists() else []
    prompts = prompts[: args.limit]
    pipeline = SpriteMindPipeline()
    if not pipeline.load_checkpoint(args.checkpoint):
        raise SystemExit(pipeline.checkpoint_error or "no usable checkpoint")

    continuity_rows = []
    rows = []
    for index, item in enumerate(prompts):
        prompt = item["prompt"] if isinstance(item, dict) else str(item)
        result = pipeline.generate(prompt=prompt, engine_mode=args.engine, seed=index + 1)
        rgba = decode(result["frames"][0])
        alpha = rgba[..., 3] > 0
        sizes = components(alpha)
        main = sizes[0] if sizes else 0
        detached = int(sum(1 for s in sizes[1:] if s >= 2))
        stray = int(sum(s for s in sizes[1:] if s < main))
        rows.append(
            {
                "prompt": prompt,
                "category": result["category"],
                "body_pixels": main,
                "islands": detached,
                "island_pixels": stray,
                "outline_coverage": round(outline_coverage(rgba), 3),
            }
        )

    if not args.skip_continuity:
        for index, item in enumerate(prompts[: max(1, min(4, len(prompts)))]):
            prompt = item["prompt"] if isinstance(item, dict) else str(item)
            anim = TemporalAnimator.generate_animation(
                base_script_or_prompt=prompt,
                category=next((r["category"] for r in rows if r["prompt"] == prompt), "knight"),
                anim_category="walk",
                num_frames=args.anim_frames,
                engine_mode=args.engine,
                seed=index + 1,
            )
            report = continuity_report(masks_from_frames(anim["frames"], decode))
            continuity_rows.append({"prompt": prompt, **report})

    summary = {
        "samples": len(rows),
        "mean_outline_coverage": round(float(np.mean([r["outline_coverage"] for r in rows])), 3),
        "prompts_with_islands": int(sum(1 for r in rows if r["islands"] > 1)),
        "mean_island_pixels": round(float(np.mean([r["island_pixels"] for r in rows])), 2),
    }
    if continuity_rows:
        summary["continuity"] = {
            "animations": len(continuity_rows),
            "mean_iou": round(float(np.mean([c["mean_iou"] for c in continuity_rows])), 4),
            "worst_iou": round(min(c["worst_iou"] for c in continuity_rows), 4),
            "total_lost_pixels": int(sum(c["total_lost_pixels"] for c in continuity_rows)),
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows, "continuity": continuity_rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
