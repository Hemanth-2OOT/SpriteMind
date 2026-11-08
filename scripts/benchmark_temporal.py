"""Phase 5: temporal (animation) benchmark for TemporalSVAN.

Evaluated on all 20 animation classes for three characters and three engines.
Per sequence it reports frame-to-frame silhouette motion, flicker, loop
closure, per-frame connectivity and palette adherence, and how much motion is
visible at all.

The `neural_share` column is the attribution the shipped generator hides: the
production path blends 70% neural offset with a 30% kinematic prior, so this
benchmark also reports the mean magnitude of the raw neural trajectory against
the prior it is blended with.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_common import BENCH_DIR, decode_rgba, markdown_table, write_json  # noqa: E402

from spritemind.dataset.temporal_dataset import ANIMATION_CATEGORIES, CYCLIC_ANIMATIONS, TemporalDatasetGenerator  # noqa: E402
from spritemind.ml.metrics import (  # noqa: E402
    alpha_mask,
    largest_component_share,
    palette_adherence,
    silhouette_iou,
)
from spritemind.ml.temporal import PARTS, TemporalAnimator  # noqa: E402

CHARACTERS = ["knight", "robot", "animal"]


def _sequence_metrics(frames, palette):
    masks = [alpha_mask(decode_rgba(f["image_base64"])) for f in frames]
    adjacent = [silhouette_iou(masks[i], masks[i + 1]) for i in range(len(masks) - 1)]
    union_change = [
        float(np.logical_xor(masks[i], masks[i + 1]).sum() / max(1, masks[i].sum()))
        for i in range(len(masks) - 1)
    ]
    return {
        "adjacent_iou": float(np.mean(adjacent)),
        "motion_px_share": float(np.mean(union_change)),
        "loop_iou": silhouette_iou(masks[0], masks[-1]),
        "static_frames": sum(1 for value in union_change if value < 1e-6) / max(1, len(union_change)),
        "main_component_share": float(np.mean([largest_component_share(m) for m in masks])),
        "palette_adherence_pct": float(np.mean([palette_adherence(decode_rgba(f["image_base64"]), palette) for f in frames])),
        "pixels_per_frame": float(np.mean([m.sum() for m in masks])),
    }


def _reference_frames(category, palette="pico8"):
    import torch

    from _bench_common import canonical_reference

    mask = canonical_reference(category, palette)
    rgba = np.zeros((32, 32, 4), dtype=np.float32)
    rgba[..., 0] = 200.0
    rgba[..., 1] = 120.0
    rgba[..., 2] = 60.0
    rgba[mask, 3] = 255.0
    return torch.from_numpy(rgba.transpose(2, 0, 1).copy()).unsqueeze(0)


def _neural_vs_prior(model, category, anim, num_frames):
    """Raw neural offset magnitude against the kinematic prior it is blended with."""
    import torch

    from spritemind.ml.temporal import CAT_TO_IDX

    anim_idx = ANIMATION_CATEGORIES.index(anim) if anim in ANIMATION_CATEGORIES else 0
    phases = [f / float(num_frames) for f in range(num_frames)]
    with torch.no_grad():
        traj = model(
            torch.tensor([CAT_TO_IDX.get(category, 0)]),
            torch.tensor([anim_idx]),
            torch.tensor([[[p] for p in phases]]),
            _reference_frames(category),
        )["trajectories"][0].numpy()
    neural, prior = [], []
    for f_idx, phase in enumerate(phases):
        prior_offsets = TemporalDatasetGenerator.compute_motion_offsets(anim, phase, category=category)
        for p_idx, part in enumerate(PARTS):
            neural.append(float(np.hypot(traj[f_idx, p_idx, 0], traj[f_idx, p_idx, 1])))
            dx, dy, _ = prior_offsets.get(part, (0, 0, 0))
            prior.append(float(np.hypot(dx, dy)))
    neural_arr, prior_arr = np.asarray(neural), np.asarray(prior)
    nonzero = prior_arr > 0
    return {
        "neural_mean_mag": float(neural_arr.mean()),
        "prior_mean_mag": float(prior_arr.mean()),
        "neural_zero_share": float((neural_arr < 1e-4).mean()),
        "prior_dominant_share": float((prior_arr > neural_arr).mean()) if prior_arr.size else 0.0,
        "motion_on_prior_parts": float(neural_arr[nonzero].mean()) if nonzero.any() else 0.0,
    }


def run(num_frames, seed):
    model, loaded = TemporalAnimator.get_or_load_model("temporal_svan_best.pt")
    print(f"temporal checkpoint loaded={loaded}", flush=True)

    engines = ["procedural_baseline", "untrained_model", "trained_model"]
    rows, attribution = [], {}
    for engine in engines:
        buckets = []
        for anim in ANIMATION_CATEGORIES:
            for character in CHARACTERS:
                out = TemporalAnimator.generate_animation(
                    category=character,
                    anim_category=anim,
                    palette_name="pico8",
                    num_frames=num_frames,
                    engine_mode=engine,
                    seed=seed,
                )
                row = _sequence_metrics(out["frames"], "pico8")
                row["animation"] = anim
                row["character"] = character
                row["is_looping"] = anim in CYCLIC_ANIMATIONS
                row["latency_ms"] = out["elapsed_ms"]
                buckets.append(row)
        rows.append((engine, buckets))
        print(f"[{engine}] {len(buckets)} sequences", flush=True)

    results = {}
    for engine, buckets in rows:
        numeric_keys = [k for k, v in buckets[0].items() if isinstance(v, float)]
        summary = {k: round(float(np.mean([b[k] for b in buckets])), 4) for k in numeric_keys}
        summary["n_sequences"] = len(buckets)
        per_anim = {}
        for anim in ANIMATION_CATEGORIES:
            subset = [b for b in buckets if b["animation"] == anim]
            per_anim[anim] = {
                "motion_px_share": round(float(np.mean([b["motion_px_share"] for b in subset])), 4),
                "adjacent_iou": round(float(np.mean([b["adjacent_iou"] for b in subset])), 4),
                "static_frames": round(float(np.mean([b["static_frames"] for b in subset])), 4),
            }
        results[engine] = {"summary": summary, "per_animation": per_anim}

    for anim in ANIMATION_CATEGORIES:
        attribution[anim] = _neural_vs_prior(model, "knight", anim, num_frames)

    payload = {
        "benchmark": "temporal",
        "checkpoint": "temporal_svan_best.pt",
        "num_frames": num_frames,
        "seed": seed,
        "characters": CHARACTERS,
        "results": results,
        "neural_vs_prior": attribution,
    }
    write_json(BENCH_DIR / "temporal.json", payload)

    table_rows = []
    for engine in engines:
        s = results[engine]["summary"]
        table_rows.append(
            {
                "Engine": engine,
                "Motion px share": s["motion_px_share"],
                "Adjacent IoU": s["adjacent_iou"],
                "Loop IoU": s["loop_iou"],
                "Static frames": s["static_frames"],
                "Main component": s["main_component_share"],
                "Palette %": s["palette_adherence_pct"],
                "Latency ms": s["latency_ms"],
            }
        )
    table = markdown_table(
        table_rows,
        ["Engine", "Motion px share", "Adjacent IoU", "Loop IoU", "Static frames", "Main component", "Palette %", "Latency ms"],
    )
    (BENCH_DIR / "temporal.md").write_text(table + "\n")
    print(table)

    neural = np.mean([v["neural_mean_mag"] for v in attribution.values()])
    prior = np.mean([v["prior_mean_mag"] for v in attribution.values()])
    dominant = np.mean([v["prior_dominant_share"] for v in attribution.values()])
    print(
        f"\nATTRIBUTION: mean |neural| offset = {neural:.3f}px, mean |prior| = {prior:.3f}px, "
        f"prior magnitude exceeds neural on {dominant*100:.1f}% of part-frames"
    )
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.frames, args.seed)
