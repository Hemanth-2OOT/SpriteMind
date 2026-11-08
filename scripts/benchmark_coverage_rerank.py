"""R12: coverage-aware reranking measured against the shipped ranking.

One candidate pool per prompt is sampled once, then every ranking policy is
applied to that same pool, so a difference between policies is a difference in
*selection* and nothing else. Colour selection (did the sprite contain the
requested hue at all) and colour coverage (what share of the sprite it fills) are
reported separately, as are the neural-only and guarded arms.

Weights are tuned on `data/splits/validation_v1.json` only; the frozen held-out
split is evaluated once at the tuned setting.

    python scripts/benchmark_coverage_rerank.py --mode tune
    python scripts/benchmark_coverage_rerank.py --mode eval --weights data/benchmarks/coverage_rerank_tuning.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_common import BENCH_DIR, canonical_reference, markdown_table, write_json  # noqa: E402
from benchmark_heldout import build_grids, load_spec  # noqa: E402

from spritemind.assets.faithfulness import MATCH_SHARE_FOR_PRESENCE, colour_faithfulness  # noqa: E402
from spritemind.ml.metrics import evaluate_sprite  # noqa: E402
from spritemind.ml.pipeline import SpriteMindPipeline  # noqa: E402
from spritemind.pixelscript.ast import PixelScriptProgram  # noqa: E402
from spritemind.ml.procedural import ProceduralBaseline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GRID = {"weights": [10.0, 25.0, 50.0, 100.0, 200.0, 400.0], "targets": [0.30, 0.36, 0.42, 0.50]}
IOU_TOLERANCE = 0.01


def candidate_pool(pipeline, item, seed, best_of, steer):
    """Neural candidates plus the procedural guard, all measured once."""
    palette = item.get("palette", "pico8")
    model = pipeline.trained_model
    tokens = pipeline._sample_batch(
        model, item["category"], item.get("style"), item.get("pose"), 64, seed, 0.9, best_of, 0,
        item["prompt"], item.get("colour"), item.get("element"), palette=palette, colour_steer=steer,
    )
    progs = [PixelScriptProgram(actions=pipeline.tokenizer.decode_tokens(t)) for t in tokens]
    progs.append(ProceduralBaseline.generate(
        category=item["category"], style=item.get("style"), pose=item.get("pose"),
        palette_name=palette, width=32, height=32, seed=seed,
        colour=item.get("colour"), element=item.get("element"),
    ))
    out = []
    for index, prog in enumerate(progs):
        comp = pipeline._rerank_components(
            prog, palette, 32, 32, category=item["category"], colour=item.get("colour"),
            prompt_text=item["prompt"], use_critic=True, use_colour=True,
        )
        if comp is None:
            continue
        comp["source"] = "procedural_guard" if index == len(progs) - 1 else "neural"
        out.append(comp)
    return out


def measure(pipeline, comp, item):
    rgba = comp["render"].composite_array
    palette = item.get("palette", "pico8")
    row = evaluate_sprite(rgba, palette, reference_mask=canonical_reference(item["category"], palette), prompt=item["prompt"])
    faith = colour_faithfulness(item["prompt"], rgba) or {}
    share = float(faith.get("matched_share") or 0.0)
    row.update({
        "coverage_pct": round(share * 100.0, 2),
        "selection_hit": int(share >= MATCH_SHARE_FOR_PRESENCE),
        "critic_quality": comp["critic"],
        "critic_defects": int(comp.get("critic_defects") or 0),
        "source": comp["source"],
    })
    return row


def rank(pipeline, comps, weight, target, allow_guard):
    pool = [c for c in comps if c["source"] == "neural" or allow_guard]
    pipeline.coverage_rerank_weight = weight
    pipeline.coverage_rerank_target = target
    if not pool:
        return None
    return max(pool, key=pipeline._score_components)


def run_policy(pipeline, items, seed, best_of, steer, weight, target, arm, pools=None):
    rows = []
    for index, item in enumerate(items):
        comps = pools[index] if pools is not None else candidate_pool(pipeline, item, seed + index, best_of, steer)
        chosen = rank(pipeline, comps, weight, target, allow_guard=(arm == "production"))
        if chosen is not None:
            rows.append(measure(pipeline, chosen, item))
    keys = ["silhouette_iou", "coverage_pct", "selection_hit", "colour_faithfulness", "palette_adherence_pct",
            "opaque_pixels", "connected_components", "largest_component_share", "column_symmetry",
            "interior_hole_pixels", "critic_quality", "critic_defects"]
    summary = {"n": len(rows)}
    for key in keys:
        values = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        if values:
            arr = np.asarray(values, dtype=float)
            summary[key] = round(float(arr.mean()), 4)
            summary[f"{key}_se"] = round(float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0, 4)
    summary["clean_share"] = round(sum(1 for r in rows if (r.get("critic_defects") or 0) == 0) / max(1, len(rows)), 4)
    summary["guard_share"] = round(sum(1 for r in rows if r.get("source") == "procedural_guard") / max(1, len(rows)), 4)
    return summary


def tune(pipeline, items, seed, best_of, steer):
    base = {arm: run_policy(pipeline, items, seed, best_of, steer, 0.0, 0.0, arm)
            for arm in ("neural_only", "production")}
    grid_rows = []
    for weight in GRID["weights"]:
        for target in GRID["targets"]:
            row = {"weight": weight, "target": target}
            for arm in ("neural_only", "production"):
                row[arm] = run_policy(pipeline, items, seed, best_of, steer, weight, target, arm)
            grid_rows.append(row)
    floors = {arm: base[arm]["silhouette_iou"] - IOU_TOLERANCE for arm in ("neural_only", "production")}
    eligible = [r for r in grid_rows
                if all(r[arm]["silhouette_iou"] >= floors[arm] for arm in floors)]
    pool = eligible or grid_rows
    best = max(pool, key=lambda r: (r["neural_only"]["coverage_pct"], -r["weight"]))
    return {"weight": best["weight"], "target": best["target"], "baseline": base,
            "grid": grid_rows, "floors_iou": {k: round(v, 4) for k, v in floors.items()},
            "eligible": len(eligible)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tune", "eval"], default="tune")
    parser.add_argument("--checkpoint", default="svan_v4_palette_ep2.pt")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--best-of", type=int, default=4)
    parser.add_argument("--steer", type=float, default=0.0)
    parser.add_argument("--weights", default="data/benchmarks/coverage_rerank_tuning.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    pipeline = SpriteMindPipeline()
    if not pipeline.load_checkpoint(args.checkpoint):
        raise SystemExit(f"could not load checkpoint {args.checkpoint}")

    if args.mode == "tune":
        data = json.loads((ROOT / "data/splits/validation_v1.json").read_text(encoding="utf-8"))
        items = data["prompts"]
        result = tune(pipeline, items, args.seed, args.best_of, args.steer)
        result.update({"benchmark": "coverage_rerank_tuning", "checkpoint": args.checkpoint,
                       "split": "data/splits/validation_v1.json", "split_digest": data["digest"],
                       "steer": args.steer, "n_prompts": len(items), "grid_spec": GRID})
        out = ROOT / (args.out or "data/benchmarks/coverage_rerank_tuning.json")
        write_json(out, result)
        rows = [{"weight": r["weight"], "target": r["target"],
                 "neural coverage %": r["neural_only"]["coverage_pct"],
                 "neural IoU": r["neural_only"]["silhouette_iou"],
                 "prod coverage %": r["production"]["coverage_pct"],
                 "prod IoU": r["production"]["silhouette_iou"]} for r in result["grid"]]
        print(markdown_table(rows, list(rows[0])))
        print(f"\ntuned: weight={result['weight']} target={result['target']} → {out}")
        return 0

    tuned = json.loads((ROOT / args.weights).read_text(encoding="utf-8"))
    weight, target = tuned["weight"], tuned["target"]
    spec = load_spec(ROOT / "data/splits/heldout_v1.json")
    items = build_grids(spec)["heldout"]
    arms = {}
    tuned_steer3 = json.loads((ROOT / "data/benchmarks/coverage_rerank_tuning_steer3.json").read_text(encoding="utf-8")) \
        if (ROOT / "data/benchmarks/coverage_rerank_tuning_steer3.json").exists() else tuned
    for steer in (0.0, 3.0):
        pools = [candidate_pool(pipeline, item, args.seed + i, args.best_of, steer) for i, item in enumerate(items)]
        settings = tuned if steer == 0.0 else tuned_steer3
        for name, w, t in (("existing", 0.0, 0.0), ("coverage", settings["weight"], settings["target"])):
            for arm in ("neural_only", "production"):
                key = f"steer{steer:g}/{name}/{arm}"
                arms[key] = run_policy(pipeline, items, args.seed, args.best_of, steer, w, t, arm, pools=pools)
                print(f"[{key}] {arms[key]}", flush=True)
    payload = {"benchmark": "coverage_rerank_eval", "checkpoint": args.checkpoint,
               "spec": "data/splits/heldout_v1.json", "spec_digest": spec["digest"],
               "seed": args.seed, "best_of": args.best_of,
               "tuned": {"weight": weight, "target": target},
               "tuned_steer3": {"weight": tuned_steer3["weight"], "target": tuned_steer3["target"]},
               "tuning_source": args.weights, "n_prompts": len(items), "results": arms}
    out = ROOT / (args.out or "data/benchmarks/coverage_rerank_heldout.json")
    write_json(out, payload)
    metrics = ("silhouette_iou", "coverage_pct", "selection_hit", "palette_adherence_pct",
               "opaque_pixels", "connected_components", "interior_hole_pixels", "critic_defects",
               "clean_share", "guard_share")
    rows = [{"arm": k, **{m: v.get(m) for m in metrics}} for k, v in arms.items()]
    (out.with_suffix(".md")).write_text(markdown_table(rows, list(rows[0])) + "\n")
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
