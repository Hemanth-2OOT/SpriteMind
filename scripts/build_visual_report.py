"""Phase 9: visual evaluation report (contact sheets, colour probes, failures).

Writes PNG contact sheets under `data/reports/audit/` and an index markdown so
a human can check the numbers by eye:

* `categories_poses.png` — 10 categories x 5 poses, trained vs procedural, same prompt.
* `colour_probe.png`     — requested colour vs dominant sprite hue, one row per colour.
* `worst_cases.png`      — the 12 lowest-IoU neural generations from the
  neural-vs-procedural benchmark, re-rendered with the same seed.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_common import BENCH_DIR, ROOT, decode_rgba, write_json  # noqa: E402

from spritemind.dataset.generator import CATEGORIES, POSES  # noqa: E402
from spritemind.ml.pipeline import SpriteMindPipeline  # noqa: E402

OUT = ROOT / "data" / "reports" / "audit"
SCALE = 4
CELL = 32 * SCALE
LABEL_W = 96
HEADER_H = 16


def _sheet(rows, cols, cell_size=CELL):
    width = LABEL_W + cols * cell_size
    height = HEADER_H + rows * cell_size
    canvas = Image.new("RGBA", (width, height), (24, 24, 28, 255))
    return canvas, ImageDraw.Draw(canvas)


def _paste(canvas, rgba, row, col, cell_size=CELL):
    tile = Image.fromarray(rgba.astype(np.uint8), mode="RGBA").resize((cell_size, cell_size), Image.NEAREST)
    canvas.alpha_composite(tile, (LABEL_W + col * cell_size, HEADER_H + row * cell_size))


def _draw(canvas, draw, text, xy, fill=(200, 200, 210, 255)):
    draw.text(xy, text, fill=fill)


def categories_poses(pipeline):
    pairs = [("trained_model", "neural"), ("procedural_baseline", "procedural")]
    cols = len(POSES) * len(pairs)
    canvas, draw = _sheet(len(CATEGORIES), cols)
    for c_index, category in enumerate(CATEGORIES):
        _draw(canvas, draw, category, (4, 2 + HEADER_H + c_index * CELL + CELL // 2 - 4))
        col = 0
        for pose in POSES:
            for engine, _ in pairs:
                out = pipeline.generate(
                    prompt=f"a {category} {pose}",
                    category=category,
                    style="classic_16bit",
                    pose=pose,
                    palette_name="pico8",
                    engine_mode=engine,
                    seed=5,
                    best_of=4,
                )
                rgba = decode_rgba(out["frames"][0]["image_base64"])
                if rgba is not None:
                    _paste(canvas, rgba, c_index, col)
                col += 1
    path = OUT / "categories_poses.png"
    canvas.convert("RGB").save(path)
    return path.name, f"{len(CATEGORIES)} categories x {len(POSES)} poses, left of each pair = neural, right = procedural"


def colour_probe(pipeline):
    colours = ["red", "blue", "green", "gold", "purple", "white", "black", "orange", "pink", "cyan"]
    canvas, draw = _sheet(len(colours), 3)
    for index, colour in enumerate(colours):
        _draw(canvas, draw, colour, (4, 2 + HEADER_H + index * CELL + CELL // 2 - 4))
        for col, engine in enumerate(("trained_model", "procedural_baseline", "untrained_model")):
            out = pipeline.generate(
                prompt=f"a {colour} knight with a sword",
                category="knight",
                style="classic_16bit",
                pose="idle",
                palette_name="pico8",
                engine_mode=engine,
                seed=11,
                best_of=4,
            )
            rgba = decode_rgba(out["frames"][0]["image_base64"])
            if rgba is not None:
                _paste(canvas, rgba, index, col)
    path = OUT / "colour_probe.png"
    canvas.convert("RGB").save(path)
    return path.name, "rows = requested colour; columns = neural / procedural / untrained"


def worst_cases(pipeline):
    import json

    benchmark = json.loads((BENCH_DIR / "neural_vs_procedural.json").read_text())
    del benchmark
    from _bench_common import canonical_reference, load_prompts
    from spritemind.ml.metrics import evaluate_sprite

    prompts = load_prompts(300)
    scored = []
    for index, item in enumerate(prompts):
        out = pipeline.generate(
            prompt=item["prompt"],
            category=item["category"],
            style=item.get("style"),
            pose=item.get("pose"),
            palette_name=item["palette"],
            engine_mode="trained_model",
            seed=20261105 + index,
            best_of=4,
        )
        rgba = decode_rgba(out["frames"][0]["image_base64"])
        if rgba is None:
            continue
        row = evaluate_sprite(rgba, item["palette"], reference_mask=canonical_reference(item["category"], item["palette"]))
        scored.append((row["silhouette_iou"], index, item, rgba))
    scored.sort(key=lambda entry: entry[0])
    worst = scored[:12]

    cols = 4
    rows = (len(worst) + cols - 1) // cols
    cell = CELL * 2
    canvas, draw = _sheet(rows, cols, cell_size=cell)
    for position, (iou, index, item, rgba) in enumerate(worst):
        row, col = divmod(position, cols)
        _paste(canvas, rgba, row, col, cell_size=cell)
        _draw(canvas, draw, f"IoU {iou:.2f}", (LABEL_W + col * cell + 2, HEADER_H + row * cell + 2))
    path = OUT / "worst_cases.png"
    canvas.convert("RGB").save(path)
    write_json(
        OUT / "worst_cases.json",
        [
            {"iou": round(iou, 3), "prompt": item["prompt"], "category": item["category"], "style": item.get("style"), "pose": item.get("pose"), "seed": 20261105 + index}
            for iou, index, item, _ in worst
        ],
    )
    return path.name, "12 lowest-IoU neural generations with their prompts (see worst_cases.json)"


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline = SpriteMindPipeline()
    pipeline.load_checkpoint("svan_checkpoint_best.pt")
    entries = []
    for func in (categories_poses, colour_probe, worst_cases):
        name, description = func(pipeline)
        entries.append((name, description))
        print(f"wrote {name}", flush=True)

    lines = ["# Visual evaluation report", "", "Generated from `svan_checkpoint_best.pt`, palette `pico8`, 32x32, nearest 4x.", ""]
    for name, description in entries:
        lines += [f"## {name}", "", f"{description}", "", f"![{name}]({name})", ""]
    (OUT / "README.md").write_text("\n".join(lines))
    print(OUT)


if __name__ == "__main__":
    run()
