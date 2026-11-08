"""Phase 3: neural-vs-procedural comparison on pixel-grounded metrics.

Arms
----
* `procedural_baseline`  — CharacterGenerator only.
* `untrained_model`      — SVAN with random weights, best_of=4, procedural guard on.
* `trained_best1`        — SVAN v3, best_of=1, procedural guard on.
* `trained_best4`        — SVAN v3, best_of=4, procedural guard on (shipped default).
* `trained_no_guard`     — SVAN v3, best_of=4, guard disabled; isolates the network.
* `trained_best4_no_critic` — production arm with the critic term switched off (R4 ablation).
* `trained_no_guard_heuristic_only` — pre-R4/R5/R6 ranking (no colour, no critic), guard off.

Metrics are computed by `spritemind.ml.metrics` from the rendered RGBA and the
canonical category silhouette. No legacy proxy metric is used.
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_common import (  # noqa: E402
    BENCH_DIR,
    canonical_reference,
    decode_rgba,
    load_prompts,
    markdown_table,
    write_json,
)

from spritemind.ml.metrics import evaluate_sprite, summarise  # noqa: E402
from spritemind.ml.pipeline import (  # noqa: E402
    COLOUR_RERANK_REFERENCE,
    COLOUR_RERANK_WEIGHT,
    CRITIC_RERANK_REFERENCE,
    CRITIC_RERANK_WEIGHT,
    SpriteMindPipeline,
)

CHECKPOINT = "svan_checkpoint_best.pt"


def _neural_only(
    pipeline: SpriteMindPipeline,
    item: dict,
    seed: int,
    best_of: int,
    use_critic: bool = True,
    use_colour: bool = True,
    colour_steer: float = 0.0,
):
    """SVAN candidates with no procedural guard, so the network is measured alone."""
    model = pipeline.trained_model
    prog, source = pipeline._neural_generate(
        model,
        item["category"],
        item.get("style", "classic_16bit"),
        item.get("pose", "idle"),
        item.get("palette", "pico8"),
        max_actions=64,
        seed=seed,
        best_of=best_of,
        temperature=0.9,
        top_k=0,
        procedural_guard=False,
        prompt_text=item.get("prompt", ""),
        colour=item.get("colour"),
        element=item.get("element"),
        colour_steer=colour_steer,
        use_critic=use_critic,
        use_colour=use_colour,
    )
    return prog, source


def run(limit, best_of, seed):
    prompts = load_prompts(limit)
    pipeline = SpriteMindPipeline()
    if not pipeline.load_checkpoint(CHECKPOINT):
        raise SystemExit("could not load SVAN checkpoint")

    arms = [
        "procedural_baseline", "untrained_model", "trained_best1", "trained_best4",
        "trained_best4_no_critic", "trained_no_guard", "trained_no_guard_heuristic_only",
    ]
    results = {}

    for arm in arms:
        per_sample, failures, sources = [], 0, Counter()
        t0 = time.perf_counter()
        for index, item in enumerate(prompts):
            palette = item.get("palette", "pico8")
            try:
                if arm.startswith("trained_no_guard"):
                    prog, source = _neural_only(
                        pipeline, item, seed + index, best_of,
                        use_critic=not arm.endswith("heuristic_only"),
                        use_colour=not arm.endswith("heuristic_only"),
                    )
                    from spritemind.renderer.engine import PixelScriptEngine

                    rendered = PixelScriptEngine.execute(prog, width=32, height=32, palette_name=palette)
                    rgba = rendered.composite_array
                    elapsed_ms = None
                else:
                    engine = {
                        "procedural_baseline": "procedural_baseline",
                        "untrained_model": "untrained_model",
                    }.get(arm, "trained_model")
                    n = best_of if arm.startswith("trained_best4") or arm == "untrained_model" else 1
                    out = pipeline.generate(
                        prompt=item["prompt"],
                        category=item["category"],
                        style=item.get("style"),
                        pose=item.get("pose"),
                        palette_name=palette,
                        engine_mode=engine,
                        seed=seed + index,
                        best_of=n,
                        use_critic=arm != "trained_best4_no_critic",
                    )
                    rgba = decode_rgba(out["frames"][0]["image_base64"])
                    source = out.get("candidate_source", engine)
                    elapsed_ms = out.get("inference_time_ms")
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
                row["category"] = item["category"]
                row["style"] = item.get("style")
                row["pose"] = item.get("pose")
                row["candidate_source"] = source
                if elapsed_ms is not None:
                    row["latency_ms"] = elapsed_ms
                per_sample.append(row)
            except Exception as exc:  # noqa: BLE001 - a crashed sample is a data point
                failures += 1
                per_sample.append({"error": f"{type(exc).__name__}: {exc}"})

        cleaned = [r for r in per_sample if "error" not in r]
        results[arm] = {
            "summary": summarise(cleaned),
            "failures": failures,
            "candidate_sources": dict(sources),
            "wall_seconds": round(time.perf_counter() - t0, 2),
            "by_category": {
                cat: summarise([r for r in cleaned if r["category"] == cat])
                for cat in sorted({r["category"] for r in cleaned})
            },
        }
        print(f"[{arm}] n={len(cleaned)} failures={failures} sources={dict(sources)}", flush=True)

    payload = {
        "benchmark": "neural_vs_procedural",
        "checkpoint": CHECKPOINT,
        "prompts": len(prompts),
        "best_of": best_of,
        "seed": seed,
        "iou_reference": "canonical category idle template (proportion=standard, seed=0)",
        "rerank": {
            "colour_weight": COLOUR_RERANK_WEIGHT,
            "colour_reference": COLOUR_RERANK_REFERENCE,
            "critic_weight": CRITIC_RERANK_WEIGHT,
            "critic_reference": CRITIC_RERANK_REFERENCE,
        },
        "results": results,
    }
    out_path = BENCH_DIR / "neural_vs_procedural.json"
    write_json(out_path, payload)

    metrics = [
        ("silhouette_iou", "IoU"),
        ("palette_adherence_pct", "Palette %"),
        ("opaque_pixels", "Pixels"),
        ("connected_components", "Parts"),
        ("largest_component_share", "Main share"),
        ("column_symmetry", "Symmetry"),
        ("interior_hole_pixels", "Hole px"),
        ("colour_faithfulness", "Colour %"),
        ("latency_ms", "Latency ms"),
    ]
    rows = []
    for arm in arms:
        summary = results[arm]["summary"]
        row = {"Arm": arm, "n": summary["n"]}
        for key, label in metrics:
            value = summary.get(key)
            row[label] = "-" if not value else f"{value['mean']:.3f} ± {value['stderr']:.3f}"
        row["Guard wins"] = results[arm]["candidate_sources"].get("procedural_guard", 0)
        rows.append(row)
    table = markdown_table(rows, ["Arm", "n"] + [label for _, label in metrics] + ["Guard wins"])
    (BENCH_DIR / "neural_vs_procedural.md").write_text(table + "\n")
    print(table)
    print(f"\nwrote {out_path}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--best-of", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    run(args.limit, args.best_of, args.seed)
