"""R13: does the candidate pool contain a correctly coloured sprite, and at what cost?

Every candidate is measured once (render, heuristic, palette leak, colour
selection/coverage, critic), then each arm selects from an already-measured pool,
so arm differences are selection + sampling effects only.

Arms answer two separate questions:

* equal-pool-size growth  — best_of 4 / 8 / 16 from one sampling batch;
* equal-compute growth    — four independent 4-pools (16 samples, four
  decisions) against one 16-pool (16 samples, one decision);
* temperature variation   — mixed 0.7/0.9/1.1/1.3 pools at 8 and 16.

    python scripts/benchmark_candidate_diversity.py --label r13
"""
import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_common import BENCH_DIR, canonical_reference, markdown_table, write_json  # noqa: E402
from benchmark_heldout import build_grids, load_spec  # noqa: E402

from spritemind.assets.faithfulness import MATCH_SHARE_FOR_PRESENCE, colour_faithfulness, selection_hit  # noqa: E402
from spritemind.ml.metrics import evaluate_sprite  # noqa: E402
from spritemind.ml.pipeline import SpriteMindPipeline  # noqa: E402
from spritemind.pixelscript.ast import PixelScriptProgram  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPS = (0.7, 0.9, 1.1, 1.3)
PER_TEMP = 4
ARMS = {
    "best_of_4": {"pool": "uniform", "k": 4},
    "best_of_8": {"pool": "uniform", "k": 8},
    "best_of_16": {"pool": "uniform", "k": 16},
    "temp_mixed_8": {"pool": "mixed", "k": 8},
    "temp_mixed_16": {"pool": "mixed", "k": 16},
    "four_pools_of_4": {"pool": "repeat", "k": 4},
}


def measure_candidate(pipeline, tokens, item, palette, index):
    prog = PixelScriptProgram(actions=pipeline.tokenizer.decode_tokens(tokens))
    comp = pipeline._rerank_components(
        prog, palette, 32, 32, category=item["category"], colour=item.get("colour"),
        prompt_text=item["prompt"], use_critic=True, use_colour=True,
    )
    if comp is None:
        return None
    rgba = comp["render"].composite_array
    row = evaluate_sprite(rgba, palette, reference_mask=canonical_reference(item["category"], palette),
                          prompt=item["prompt"])
    faith = colour_faithfulness(item["prompt"], rgba)
    # A prompt that names no colour cannot fail a colour request; keeping it in
    # the hit rate silently turns "not asked" into "missed".
    share = float(faith["matched_share"]) if faith else None
    row.update({
        "index": index,
        "src": rgba,
        "coverage": share,
        "selection_hit": selection_hit(faith),
        "critic_defects": int(comp.get("critic_defects") or 0),
        "sprite_hash": hashlib.sha1(np.ascontiguousarray(rgba).tobytes()).hexdigest(),
        "program_key": tuple(a.action_type.value for a in prog.actions),
        "score": pipeline._score_components(comp),
    })
    row.pop("src")
    return {"row": row, "rgba": rgba}


def pixel_disagreement(rows):
    if len(rows) < 2:
        return 0.0
    masks = [r["rgba"] for r in rows]
    pairs, total = 0.0, 0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            diff = (masks[i] != masks[j]).any(axis=-1).mean()
            pairs += float(diff)
            total += 1
    return pairs / total


def program_distance(rows):
    if len(rows) < 2:
        return 0.0
    sets = [set(r["row"]["program_key"]) for r in rows]
    values, total = 0.0, 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            values += 0.0 if not union else 1.0 - len(sets[i] & sets[j]) / len(union)
            total += 1
    return values / total


def summarise(pool_rows):
    keys = ("silhouette_iou", "colour_faithfulness", "palette_adherence_pct",
            "opaque_pixels", "connected_components", "interior_hole_pixels")
    out = {"n": len(pool_rows)}
    for key in keys:
        values = np.asarray([float(r[key]) for r in pool_rows if isinstance(r.get(key), (int, float))], dtype=float)
        if values.size:
            out[key] = round(float(values.mean()), 4)
            out[f"{key}_se"] = round(float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0, 4)
    out["clean_share"] = round(sum(1 for r in pool_rows if r["critic_defects"] == 0) / max(1, len(pool_rows)), 4)
    out["defects"] = round(float(np.mean([r["critic_defects"] for r in pool_rows])), 4)
    hits = [r["selection_hit"] for r in pool_rows if r["selection_hit"] is not None]
    out["n_coloured"] = len(hits)
    out["selection_hit"] = round(float(np.mean(hits)), 4) if hits else None
    coloured = [r["coverage"] for r in pool_rows if r["coverage"] is not None]
    # Prompts that name no colour have no coverage to report; averaging their 0
    # into the headline hides the real value behind a fixed 50% dilution.
    out["coverage"] = round(float(np.mean(coloured)) * 100.0, 4) if coloured else None
    out["coverage_all_prompts"] = round(float(np.mean([r["coverage"] or 0.0 for r in pool_rows])) * 100.0, 4)
    if len(coloured) > 1:
        arr = np.asarray(coloured, dtype=float)
        out["coverage_se"] = round(float(arr.std(ddof=1) / np.sqrt(arr.size)) * 100.0, 4)
    return out


def sample_pool(pipeline, model, item, palette, seed, temperature, n):
    return pipeline._sample_batch(
        model, item["category"], item.get("style"), item.get("pose"), 64, seed, temperature, n, 0,
        item["prompt"], item.get("colour"), item.get("element"), palette=palette,
    )


def run_arm(pipeline, arm, pools, items):
    spec = ARMS[arm]
    results, timings, per_prompt = [], [], []
    for index, item in enumerate(items):
        start = time.perf_counter()
        measured = pools[spec["pool"]][index]
        if spec["pool"] == "repeat":
            selected = [max(pool, key=lambda c: c["row"]["score"]) for pool in measured]
        else:
            selected = [max(measured[: spec["k"]], key=lambda c: c["row"]["score"])]
        timings.append(time.perf_counter() - start)
        shares = [c["row"]["coverage"] for c in selected if c["row"]["coverage"] is not None]
        per_prompt.append({"hit": int(any(c["row"]["selection_hit"] for c in selected if c["row"]["selection_hit"] is not None)),
                           "coverage": round(float(np.mean(shares)), 4) if shares else 0.0})
        for cand in selected:
            results.append(cand["row"])
    summary = summarise(results)
    summary["select_ms"] = round(1000.0 * float(np.mean(timings)), 2)
    summary["per_prompt"] = per_prompt
    return summary


def pool_diagnostics(pools, items, key, sizes):
    diag = {}
    for size in sizes:
        hits, coverage, uniq, pix, prog = [], [], [], [], []
        for index, _ in enumerate(items):
            rows = pools[key][index][:size]
            flags = [r["row"]["selection_hit"] for r in rows if r["row"]["selection_hit"] is not None]
            if flags:
                hits.append(int(any(flags)))
                coverage.append(max(r["row"]["coverage"] for r in rows))
            uniq.append(len({r["row"]["sprite_hash"] for r in rows}) / len(rows))
            pix.append(pixel_disagreement(rows))
            prog.append(program_distance(rows))
        diag[f"k={size}"] = {"hit_any": round(float(np.mean(hits)), 4),
                             "max_coverage": round(float(np.mean(coverage)), 4),
                             "unique_sprite_share": round(float(np.mean(uniq)), 4),
                             "pixel_disagreement": round(float(np.mean(pix)), 4),
                             "program_distance": round(float(np.mean(prog)), 4)}
    return diag


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="svan_v4_palette_ep2.pt")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--label", default="r13")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    pipeline = SpriteMindPipeline()
    if not pipeline.load_checkpoint(args.checkpoint):
        raise SystemExit(f"could not load checkpoint {args.checkpoint}")
    spec = load_spec(ROOT / "data/splits/heldout_v1.json")
    items = build_grids(spec)["heldout"]
    if args.limit:
        items = items[: args.limit]

    pools = {"uniform": [], "mixed": [], "repeat": []}
    timings = {"uniform16": [], "mixed16": [], "repeat4x4": []}
    for index, item in enumerate(items):
        palette = item.get("palette", "pico8")
        seed = args.seed + index
        start = time.perf_counter()
        tokens = sample_pool(pipeline, pipeline.trained_model, item, palette, seed, 0.9, 16)
        uniform = [measure_candidate(pipeline, t, item, palette, i) for i, t in enumerate(tokens)]
        timings["uniform16"].append(time.perf_counter() - start)
        pools["uniform"].append([c for c in uniform if c is not None])

        start = time.perf_counter()
        mixed = []
        for offset, temperature in enumerate(TEMPS):
            for t in sample_pool(pipeline, pipeline.trained_model, item, palette, seed + 7919 * (offset + 1),
                                 temperature, PER_TEMP):
                cand = measure_candidate(pipeline, t, item, palette, len(mixed))
                if cand is not None:
                    mixed.append(cand)
        timings["mixed16"].append(time.perf_counter() - start)
        pools["mixed"].append(mixed)

        start = time.perf_counter()
        repeat = []
        for offset in range(4):
            batch = sample_pool(pipeline, pipeline.trained_model, item, palette, seed + 104729 * (offset + 1), 0.9, 4)
            rows = [measure_candidate(pipeline, t, item, palette, i) for i, t in enumerate(batch)]
            repeat.append([c for c in rows if c is not None])
        timings["repeat4x4"].append(time.perf_counter() - start)
        pools["repeat"].append(repeat)
        print(f"[{index + 1}/{len(items)}] sampled", flush=True)

    arms = {name: run_arm(pipeline, name, pools, items) for name in ARMS}
    diagnostics = {
        "uniform": pool_diagnostics(pools, items, "uniform", (4, 8, 16)),
        "mixed": pool_diagnostics(pools, items, "mixed", (8, 16)),
    }
    union_hits, union_cov = [], []
    for index, _ in enumerate(items):
        every = pools["uniform"][index] + pools["mixed"][index] + [c for g in pools["repeat"][index] for c in g]
        flags = [c["row"]["selection_hit"] for c in every if c["row"]["selection_hit"] is not None]
        if flags:
            union_hits.append(int(any(flags)))
            union_cov.append(max(c["row"]["coverage"] for c in every))
    diagnostics["union_all_48"] = {"hit_any": round(float(np.mean(union_hits)), 4),
                                   "max_coverage": round(float(np.mean(union_cov)), 4),
                                   "n_candidates": 48}
    per_prompt = {"hits": {k: [p["hit"] for p in v["per_prompt"]] for k, v in arms.items()},
                  "coverage": {k: [p["coverage"] for p in v["per_prompt"]] for k, v in arms.items()}}
    heap_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    for name in arms:
        arms[name].pop("per_prompt", None)
    payload = {
        "benchmark": "candidate_diversity",
        "label": args.label,
        "checkpoint": args.checkpoint,
        "spec": "data/splits/heldout_v1.json",
        "spec_digest": spec["digest"],
        "seed": args.seed,
        "n_prompts": len(items),
        "max_actions": 64,
        "temperatures": list(TEMPS),
        "per_temperature": PER_TEMP,
        "arms": ARMS,
        "results": arms,
        "pool_diagnostics": diagnostics,
        "per_prompt": per_prompt,
        "sample_seconds": {k: round(float(np.mean(v)), 3) for k, v in timings.items()},
        "peak_rss_mb": heap_mb,
    }
    out = ROOT / (args.out or f"data/benchmarks/candidate_diversity_{args.label}.json")
    write_json(out, payload)
    rows = [{"arm": k, "n": v["n"], "selection_hit": v["selection_hit"],
             "coverage %": v["coverage"], "cov incl. blank %": v["coverage_all_prompts"], "faith": v["colour_faithfulness"],
             "IoU": v["silhouette_iou"], "palette %": v["palette_adherence_pct"],
             "defects": v["defects"], "clean": v["clean_share"], "n colour": v["n_coloured"]} for k, v in arms.items()]
    table = markdown_table(rows, list(rows[0]))
    payload["selection_hit_definition"] = "coloured prompts only (prompts naming no colour are excluded, not counted as misses)"
    diag_rows = []
    for pool, group in diagnostics.items():
        if "hit_any" in group:
            diag_rows.append({"pool": pool, "k": group.get("n_candidates", "-"), **group})
            continue
        for label, values in group.items():
            diag_rows.append({"pool": pool, "k": label, **values})
    (out.with_suffix(".md")).write_text(table + "\n\n" + markdown_table(
        diag_rows,
        ["pool", "k", "hit_any", "max_coverage", "unique_sprite_share", "pixel_disagreement", "program_distance"],
    ) + "\n")
    print(table)
    print(f"\npeak RSS {heap_mb} MB · sample s {payload['sample_seconds']} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
