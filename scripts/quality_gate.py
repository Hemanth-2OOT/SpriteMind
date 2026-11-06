"""Regression gate for generation quality.

Generates a fixed sample of prompts with one engine and fails (exit 1) when
validity, palette adherence, silhouette fill or engine honesty drops below the
thresholds. Run before a release, or from CI:

    python scripts/quality_gate.py --limit 12 --engine trained_model
"""
import argparse
import base64
import io
import json
import sys
import time

from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from spritemind.ml.pipeline import SpriteMindPipeline  # noqa: E402

PROMPTS_PATH = Path("data/benchmarks/generalization_prompts.json")


def decode_frame(frame: Dict[str, Any]) -> np.ndarray:
    payload = frame["image_base64"].split(",", 1)[-1]
    raw = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGBA")
    return np.array(raw)


def evaluate(pipeline: SpriteMindPipeline, prompts: List[Dict[str, Any]], engine: str, seed: int) -> List[Dict[str, Any]]:
    rows = []
    for item in prompts:
        prompt = item["prompt"] if isinstance(item, dict) else str(item)
        started = time.perf_counter()
        result = pipeline.generate(prompt=prompt, engine_mode=engine, seed=seed)
        elapsed = (time.perf_counter() - started) * 1000.0

        frame = result["frames"][0]
        rgba = decode_frame(frame)
        allowed = {tuple(color) for color in frame["palette"]}
        opaque = rgba[:, :, 3] > 0
        used = {tuple(int(v) for v in pixel) for pixel in rgba[opaque][:, :4]} if opaque.any() else set()
        stray = len(used - allowed)
        fill = float(opaque.sum())
        validation = frame.get("validation", {})

        rows.append(
            {
                "prompt": prompt,
                "engine_used": result.get("engine_used"),
                "valid": bool(validation.get("is_valid")),
                "invalid_actions": int(validation.get("invalid_action_count", 0)),
                "off_palette_colors": stray,
                "pixels": fill,
                "fill_ratio": round(fill / (rgba.shape[0] * rgba.shape[1]) * 100.0, 2),
                "defects": int(result.get("critic", {}).get("defects_count", 0)),
                "latency_ms": round(elapsed, 1),
            }
        )
    return rows


def summarize(rows: List[Dict[str, Any]], engine: str) -> Dict[str, Any]:
    n = len(rows) or 1
    return {
        "samples": len(rows),
        "engine_requested": engine,
        "engine_honoured": round(sum(1 for r in rows if r["engine_used"] == engine) / n * 100.0, 1),
        "valid_rate": round(sum(1 for r in rows if r["valid"]) / n * 100.0, 1),
        "palette_adherence": round(sum(1 for r in rows if r["off_palette_colors"] == 0) / n * 100.0, 1),
        "avg_fill_ratio": round(sum(r["fill_ratio"] for r in rows) / n, 2),
        "clean_rate": round(sum(1 for r in rows if r["defects"] == 0) / n * 100.0, 1),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / n, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--engine", default="trained_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-valid", type=float, default=100.0)
    parser.add_argument("--min-palette", type=float, default=100.0)
    parser.add_argument("--min-fill", type=float, default=5.0)
    parser.add_argument("--max-fill", type=float, default=70.0)
    parser.add_argument("--min-engine-honoured", type=float, default=100.0)
    parser.add_argument("--min-clean", type=float, default=0.0, help="share of samples with zero critic defects")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    prompts = json.loads(PROMPTS_PATH.read_text())[: args.limit]

    pipeline = SpriteMindPipeline()
    best = Path("models/checkpoints/svan_checkpoint_best.pt")
    if args.engine != "procedural_baseline" and best.exists():
        pipeline.load_checkpoint(best.name)

    rows = evaluate(pipeline, prompts, args.engine, args.seed)
    summary = summarize(rows, args.engine)

    failures = []
    if summary["valid_rate"] < args.min_valid:
        failures.append(f"valid_rate {summary['valid_rate']}% < {args.min_valid}%")
    if summary["palette_adherence"] < args.min_palette:
        failures.append(f"palette_adherence {summary['palette_adherence']}% < {args.min_palette}%")
    if summary["engine_honoured"] < args.min_engine_honoured:
        failures.append(f"engine_honoured {summary['engine_honoured']}% < {args.min_engine_honoured}%")
    if summary["clean_rate"] < args.min_clean:
        failures.append(f"clean_rate {summary['clean_rate']}% < {args.min_clean}%")
    if not args.min_fill <= summary["avg_fill_ratio"] <= args.max_fill:
        failures.append(f"avg_fill_ratio {summary['avg_fill_ratio']}% outside {args.min_fill}-{args.max_fill}%")

    print(json.dumps(summary, indent=2))
    for row in rows:
        if not row["valid"] or row["off_palette_colors"]:
            print(f"  FAIL {row['prompt'][:60]}: valid={row['valid']} off_palette={row['off_palette_colors']}")

    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "samples": rows}, indent=2) + "\n")

    if failures:
        print("\nQUALITY GATE FAILED")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print("\nQUALITY GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
