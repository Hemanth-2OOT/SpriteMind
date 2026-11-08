"""R6 check: does the checkpoint actually react to the palette argument?

For one prompt and one seed, the network is sampled once per palette with the
palette embedding fed from that palette's id, then rendered in that palette. A
checkpoint trained with `pal_idx == 0` produces identical geometry for every
palette (only the rendered colours differ); a palette-conditioned checkpoint
should change the drawing itself.

    python scripts/benchmark_palette_conditioning.py --checkpoints \
        svan_checkpoint_best.pt svan_v4_palette.pt
"""
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_common import ROOT, write_json  # noqa: E402

from spritemind.ml.pipeline import SpriteMindPipeline  # noqa: E402
from spritemind.pixelscript.ast import PixelScriptProgram  # noqa: E402
from spritemind.pixelscript.serializer import PixelScriptSerializer  # noqa: E402
from spritemind.renderer.engine import PixelScriptEngine  # noqa: E402
from spritemind.utils.palettes import PALETTES  # noqa: E402

PROMPTS = [
    ("a knight with a sword", "knight", "classic_16bit"),
    ("a wizard casting a spell", "wizard", "dark_fantasy"),
    ("a slime creature", "slime", "flat"),
    ("a robot with a blaster", "robot", "cyberpunk"),
]
SEED = 20261105


def geometry_for(pipeline, category, style, palette, prompt):
    prog, source = pipeline._neural_generate(
        pipeline.trained_model,
        category,
        style,
        "idle",
        palette,
        max_actions=64,
        seed=SEED,
        best_of=1,
        temperature=0.9,
        top_k=0,
        procedural_guard=False,
        prompt_text=prompt,
        use_critic=True,
        use_colour=True,
    )
    # Serialise only the actions: the program's metadata carries the palette
    # name, so a full serialisation would differ for every palette regardless of
    # whether the network reacted at all.
    script = PixelScriptSerializer.serialize_program(PixelScriptProgram(actions=prog.actions))
    return script, source


def run(checkpoint):
    pipeline = SpriteMindPipeline()
    if not pipeline.load_checkpoint(checkpoint):
        raise SystemExit(f"could not load {checkpoint}")
    trained = bool(getattr(pipeline.trained_model, "palette_conditioning_trained", False))

    per_prompt = []
    for prompt, category, style in PROMPTS:
        scripts = {}
        for palette in PALETTES:
            script, _ = geometry_for(pipeline, category, style, palette, prompt)
            scripts[palette] = script
        digests = {p: hashlib.sha256(s.encode("utf-8")).hexdigest()[:12] for p, s in scripts.items()}
        rendered = {}
        for palette in PALETTES:
            rgba = PixelScriptEngine.execute(
                scripts[palette], width=32, height=32, palette_name=palette
            ).composite_array
            rendered[palette] = rgba.astype(np.int16)
        diffs = []
        for a, b in itertools.combinations(PALETTES, 2):
            diffs.append(float(np.abs(rendered[a] - rendered[b]).sum(axis=2).mean()))
        per_prompt.append({
            "prompt": prompt,
            "geometry_digest": digests,
            "distinct_geometries": len(set(digests.values())),
            "mean_pairwise_rgba_diff": round(float(np.mean(diffs)), 3),
        })

    return {
        "checkpoint": checkpoint,
        "palette_conditioning_trained": trained,
        "seed": SEED,
        "prompts": per_prompt,
        "mean_distinct_geometries": round(
            float(np.mean([p["distinct_geometries"] for p in per_prompt])), 2
        ),
        "mean_pairwise_rgba_diff": round(float(np.mean([p["mean_pairwise_rgba_diff"] for p in per_prompt])), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=["svan_checkpoint_best.pt", "svan_v4_palette.pt"])
    parser.add_argument("--out", default="data/benchmarks/palette_conditioning.json")
    args = parser.parse_args()

    results = [run(name) for name in args.checkpoints]
    payload = {"benchmark": "palette_conditioning", "palettes": list(PALETTES), "results": results}
    out_path = Path(args.out)
    write_json(out_path if out_path.is_absolute() else ROOT / out_path, payload)
    for row in results:
        print(
            f"{row['checkpoint']:28s} trained={row['palette_conditioning_trained']!s:5s} "
            f"distinct_geometries={row['mean_distinct_geometries']:4.2f}/{len(PALETTES)} "
            f"mean_rgba_diff={row['mean_pairwise_rgba_diff']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
