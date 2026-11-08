"""R10: generalization measured on combinations excluded from train and val.

Evaluates a checkpoint on two prompt grids built from the frozen spec:

* `heldout` — prompts whose (category, style, pose) triple, or
  (category, element) pair, is named in `data/splits/heldout_v1.json`;
* `in_distribution` — the same number of prompts drawn from combinations the
  training corpus does contain, so a held-out score can be read as a gap.

Both grids contain a plain prompt and a colour+element prompt per key, so the
comparison separates "unseen composition" from "conditioned request".

Arms per checkpoint: `neural_only` (no procedural guard — the network alone) and
`production` (shipped path with the guard). Metrics come from rendered pixels.

    python scripts/benchmark_heldout.py --checkpoint svan_checkpoint_best.pt \
        --out data/benchmarks/heldout_v1_baseline.json
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_common import (  # noqa: E402
    BENCH_DIR,
    canonical_reference,
    decode_rgba,
    markdown_table,
    write_json,
)
from benchmark_neural_vs_procedural import _neural_only  # noqa: E402

from spritemind.dataset.conditioning import COLOUR_RGB, ELEMENT_RGB, renderable_colours  # noqa: E402
from spritemind.dataset.generator import CATEGORIES, POSES, STYLES  # noqa: E402
from spritemind.dataset.splits import heldout_reason, load_spec  # noqa: E402
from spritemind.ml.metrics import evaluate_sprite, summarise  # noqa: E402
from spritemind.ml.pipeline import SpriteMindPipeline  # noqa: E402

PALETTE = "pico8"


def _prompt_item(category, style, pose, colour=None, element=None):
    words = ["a"]
    if colour:
        words.append(colour)
    if element:
        words.append(f"{element}-touched")
    words.append(category.replace("_", " "))
    return {
        "prompt": " ".join(words),
        "category": category,
        "style": style,
        "pose": pose,
        "palette": PALETTE,
        "colour": colour,
        "element": element,
    }


def build_grids(spec, per_key_prompts=2):
    colours = renderable_colours(PALETTE)
    pairs = spec["heldout_category_element"]

    heldout = []
    for triple in spec["heldout_triples"]:
        category, style, pose = triple
        heldout.append(_prompt_item(category, style, pose))
        colour = colours[len(heldout) % len(colours)]
        heldout.append(_prompt_item(category, style, pose, colour=colour))
    for category, element in pairs:
        heldout.append(_prompt_item(category, "classic_16bit", "idle", element=element))
        heldout.append(_prompt_item(category, "flat", "attack", colour=colours[len(heldout) % len(colours)]))

    spec_keys = {tuple(t) for t in spec["heldout_triples"]}
    in_dist = []
    for i in range(len(heldout)):
        for offset in range(250):
            category = CATEGORIES[(i + offset) % len(CATEGORIES)]
            style = STYLES[(i * 3 + offset) % len(STYLES)]
            pose = POSES[(i * 2 + offset) % len(POSES)]
            element = sorted(ELEMENT_RGB)[(i + offset) % len(ELEMENT_RGB)] if i % 2 else None
            if heldout_reason({"category": category, "style": style, "pose": pose, "element": element}, spec):
                continue
            if (category, style, pose) in spec_keys:
                continue
            if i % 2:
                colour = colours[(i + offset) % len(colours)]
                in_dist.append(_prompt_item(category, style, pose, colour=colour, element=element))
            else:
                in_dist.append(_prompt_item(category, style, pose))
            break
    return {"heldout": heldout, "in_distribution": in_dist}


def run_arm(pipeline, items, arm, seed, best_of=4, colour_steer=0.0):
    rows, failures, sources = [], 0, Counter()
    for index, item in enumerate(items):
        palette = item.get("palette", PALETTE)
        try:
            if arm == "neural_only":
                prog, source = _neural_only(
                    pipeline,
                    item,
                    seed + index,
                    best_of,
                    use_critic=True,
                    use_colour=True,
                    colour_steer=colour_steer,
                )
                from spritemind.renderer.engine import PixelScriptEngine

                rgba = PixelScriptEngine.execute(
                    prog, width=32, height=32, palette_name=palette
                ).composite_array
                latency = None
            else:
                out = pipeline.generate(
                    prompt=item["prompt"],
                    category=item["category"],
                    style=item.get("style"),
                    pose=item.get("pose"),
                    palette_name=palette,
                    engine_mode="trained_model",
                    seed=seed + index,
                    best_of=best_of,
                )
                rgba = decode_rgba(out["frames"][0]["image_base64"])
                source = out.get("candidate_source", "trained_model")
                latency = out.get("inference_time_ms")
            if rgba is None:
                failures += 1
                continue
            sources[source] += 1
            row = evaluate_sprite(
                rgba,
                palette,
                reference_mask=canonical_reference(item["category"], palette),
                prompt=item["prompt"],
            )
            if latency is not None:
                row["latency_ms"] = latency
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - a crashed sample is a data point
            failures += 1
            print(f"  sample error: {type(exc).__name__}: {exc}", flush=True)
    return {
        "summary": summarise(rows),
        "n": len(rows),
        "failures": failures,
        "candidate_sources": dict(sources),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="svan_checkpoint_best.pt")
    parser.add_argument("--label", default=None, help="arm name in the JSON (defaults to the checkpoint name)")
    parser.add_argument("--spec", default="data/splits/heldout_v1.json")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--best-of", type=int, default=4)
    parser.add_argument("--colour-steer", type=float, default=0.0)
    parser.add_argument("--out", default="data/benchmarks/heldout_v1_baseline.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    spec = load_spec(root / args.spec)
    grids = build_grids(spec)
    print(f"heldout prompts: {len(grids['heldout'])}, in-distribution: {len(grids['in_distribution'])}", flush=True)

    pipeline = SpriteMindPipeline()
    if not pipeline.load_checkpoint(args.checkpoint):
        raise SystemExit(f"could not load checkpoint {args.checkpoint}")
    palette_trained = bool(getattr(pipeline.trained_model, "palette_conditioning_trained", False))

    label = args.label or args.checkpoint
    started = time.perf_counter()
    results = {}
    for regime, items in grids.items():
        for arm in ("neural_only", "production"):
            results[f"{regime}/{arm}"] = run_arm(
                pipeline, items, arm, args.seed, args.best_of, colour_steer=args.colour_steer
            )
            print(f"[{regime}/{arm}] {results[f'{regime}/{arm}']['summary']}", flush=True)

    payload = {
        "colour_steer": args.colour_steer,
        "benchmark": "heldout_generalization",
        "checkpoint": args.checkpoint,
        "label": label,
        "spec": args.spec,
        "spec_digest": spec["digest"],
        "heldout_triples": len(spec["heldout_triples"]),
        "heldout_category_element": len(spec["heldout_category_element"]),
        "palette_conditioning_trained": palette_trained,
        "prompts_per_grid": {k: len(v) for k, v in grids.items()},
        "best_of": args.best_of,
        "seed": args.seed,
        "palette": PALETTE,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "results": results,
    }
    out_path = root / args.out
    write_json(out_path, payload)

    columns = ["regime/arm", "IoU", "palette %", "colour %", "pixels", "latency ms", "sources"]
    rows = []
    for key, value in results.items():
        summary = value["summary"]
        def mean(name, digits=4):
            block = summary.get(name)
            return "-" if not block else round(float(block["mean"]), digits)

        rows.append({
            "regime/arm": key,
            "IoU": mean("silhouette_iou"),
            "palette %": mean("palette_adherence_pct", 1),
            "colour %": mean("colour_faithfulness", 2),
            "pixels": mean("opaque_pixels", 1),
            "latency ms": mean("latency_ms", 1),
            "sources": json.dumps(value["candidate_sources"]),
        })
    table = markdown_table(rows, columns)

    md_path = out_path.with_suffix(".md")
    md_path.write_text(
        f"# Held-out generalization — {label}\n\nSpec `{payload['spec_digest'][:12]}` "
        f"({payload['heldout_triples']} triples, {payload['heldout_category_element']} category/element pairs), "
        f"best_of={args.best_of}, seed={args.seed}.\n\n{table}\n",
        encoding="utf-8",
    )
    print(table)
    print(f"written: {out_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
