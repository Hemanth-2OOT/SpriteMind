"""R11 stage 1: why did neural colour faithfulness fall after the R6/R9 retrain?

For each coloured prompt the script renders three ways and reports pixels, not
scores:

* `neural_uncond` — the network sampled with no colour input, which is exactly
  what `benchmark_heldout.py`'s neural arm does today;
* `neural_cond` — the same network with the requested colour fed to its semantic
  encoder (`colour_idx`), which is what training conditioned it on;
* `guard` — the procedural baseline, which reads the colour from the prompt name.

Per render it records colour faithfulness, whether the program *chose* a palette
entry that hue-matches the request (a selection error if not), and how much
coverage that colour actually got (a generation error if low). That split is the
difference between "the model never picks the colour" and "the model picks it but
cannot paint with it".

    python scripts/diagnose_colour.py --checkpoints svan_checkpoint_best.pt svan_v4_palette_ep2.pt
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_common import ROOT, write_json  # noqa: E402
from benchmark_heldout import PALETTE, build_grids  # noqa: E402

from spritemind.assets.faithfulness import (  # noqa: E402
    COLOUR_TARGETS,
    _matches,
    colour_faithfulness,
)
from spritemind.dataset.splits import load_spec  # noqa: E402
from spritemind.ml.pipeline import SpriteMindPipeline  # noqa: E402
from spritemind.renderer.engine import PixelScriptEngine  # noqa: E402
from spritemind.utils.palettes import PALETTES  # noqa: E402

SPEC_PATH = ROOT / "data/splits/heldout_v1.json"
SEED = 20261105


def nearest_palette_entry(palette, target):
    for index, entry in enumerate(PALETTES[palette]):
        if index and _matches(*entry[:3], target):
            return index
    return None


def coverage_share(rgba, rgb):
    opaque = rgba[:, :, 3] > 0
    if not opaque.any():
        return 0.0
    hit = np.all(rgba[:, :, :3] == np.array(rgb[:3], dtype=np.uint8), axis=2) & opaque
    return float(hit.sum() / opaque.sum())


def chosen_colours(program):
    used = []
    for action in program.actions:
        color = action.params.get("color")
        if isinstance(color, int) and color not in used:
            used.append(color)
    return used


def analyse(program, rgba, palette, request):
    target = COLOUR_TARGETS[request]
    used = chosen_colours(program)
    matching = [
        index
        for index in used
        if index < len(PALETTES[palette]) and _matches(*PALETTES[palette][index][:3], target)
    ]
    opaque = int((rgba[:, :, 3] > 0).sum())
    counts = Counter(tuple(int(v) for v in pixel[:3]) for pixel in rgba[rgba[:, :, 3] > 0])
    modal_rgb, modal_count = counts.most_common(1)[0] if counts else ((0, 0, 0), 0)
    entry = nearest_palette_entry(palette, target)
    return {
        "request": request,
        "faith": (colour_faithfulness(request, rgba) or {}).get("score", 0.0),
        "selection_hit": bool(matching),
        "colours_used": len(used),
        "distinct_colours": len(counts),
        "opaque_pixels": opaque,
        "modal_share": round(float(modal_count / opaque), 4) if opaque else 0.0,
        "modal_is_requested": bool(opaque) and _matches(*modal_rgb, target),
        "requested_coverage": round(
            coverage_share(rgba, PALETTES[palette][entry][:3]) if entry is not None else 0.0, 4
        ),
    }


def run_arm(pipeline, items, mode, steer=0.0):
    from spritemind.ml.procedural import ProceduralBaseline

    rows = []
    for index, item in enumerate(items):
        request = item["colour"]
        seed = SEED + index
        if mode == "guard":
            program = ProceduralBaseline.generate(
                category=item["category"],
                style=item["style"],
                pose=item["pose"],
                palette_name=PALETTE,
                width=32,
                height=32,
                seed=seed,
                colour=request,
                element=item.get("element"),
            )
        else:
            program, _ = pipeline._neural_generate(
                pipeline.trained_model,
                item["category"],
                item["style"],
                item["pose"],
                PALETTE,
                max_actions=64,
                seed=seed,
                best_of=4,
                temperature=0.9,
                top_k=0,
                procedural_guard=False,
                colour=request if mode in ("neural_cond", "neural_steer") else None,
                colour_steer=steer,
                element=None,
                use_critic=True,
                use_colour=True,
            )
        rgba = PixelScriptEngine.execute(
            program, width=32, height=32, palette_name=PALETTE
        ).composite_array
        rows.append(analyse(program, rgba, PALETTE, request))
    return rows


def summarise(rows):
    faith = [row["faith"] for row in rows]
    selected = [row for row in rows if row["selection_hit"]]
    return {
        "n": len(rows),
        "faith_mean": round(float(np.mean(faith)), 2),
        "selection_hit_rate": round(len(selected) / len(rows), 3),
        "faith_when_selected": round(float(np.mean([r["faith"] for r in selected])), 2) if selected else None,
        "modal_is_requested_rate": round(float(np.mean([r["modal_is_requested"] for r in rows])), 3),
        "modal_share_mean": round(float(np.mean([r["modal_share"] for r in rows])), 3),
        "requested_coverage_mean": round(float(np.mean([r["requested_coverage"] for r in rows])), 3),
        "distinct_colours_mean": round(float(np.mean([r["distinct_colours"] for r in rows])), 2),
        "colours_used_mean": round(float(np.mean([r["colours_used"] for r in rows])), 2),
    }


def grid_colours(grids):
    names = []
    for entry in grids["heldout"] + grids["in_distribution"]:
        for word in entry["prompt"].replace(",", " ").split():
            if word in COLOUR_TARGETS and word not in names:
                names.append(word)
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints", nargs="+", default=["svan_checkpoint_best.pt", "svan_v4_palette_ep2.pt"]
    )
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--steer", type=float, nargs="+", default=[1.5, 3.0])
    parser.add_argument("--out", default="data/benchmarks/colour_diagnosis.json")
    args = parser.parse_args()

    spec = load_spec(SPEC_PATH)
    grids = build_grids(spec)
    colours = grid_colours(grids)
    items = []
    for entry in grids["heldout"]:
        name = next((c for c in colours if c in entry["prompt"].split()), None)
        if name:
            items.append(dict(entry, colour=name))
    items = items[: args.limit]

    results = []
    for checkpoint in args.checkpoints:
        pipeline = SpriteMindPipeline()
        if not pipeline.load_checkpoint(checkpoint):
            raise SystemExit(f"could not load {checkpoint}")
        arms = {"neural_uncond": summarise(run_arm(pipeline, items, "neural_uncond"))}
        arms["neural_cond"] = summarise(run_arm(pipeline, items, "neural_cond"))
        for steer in args.steer:
            arms[f"neural_steer_{steer}"] = summarise(
                run_arm(pipeline, items, "neural_steer", steer=steer)
            )
        arms["guard"] = summarise(run_arm(pipeline, items, "guard"))
        results.append({"checkpoint": checkpoint, "arms": arms})

    payload = {"benchmark": "colour_diagnosis", "seed": SEED, "prompts": len(items), "results": results}
    out_path = Path(args.out)
    write_json(out_path if out_path.is_absolute() else ROOT / out_path, payload)

    for entry in results:
        print(entry["checkpoint"])
        for arm, stats in entry["arms"].items():
            print(
                f"  {arm:18s} faith={stats['faith_mean']:5.2f} sel_hit={stats['selection_hit_rate']:.3f} "
                f"faith|sel={stats['faith_when_selected']} modal_req={stats['modal_is_requested_rate']:.3f} "
                f"req_cov={stats['requested_coverage_mean']:.3f} colours={stats['distinct_colours_mean']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
