"""Phase 6: SocketPredictor benchmark with honest ground truth.

The shipped trainer derives its "ground truth" sockets from a rendered alpha
bounding box (`head = (cx, min_y+2)`, `hands = (±bbox, cy+4)`, `back = centre`).
This benchmark therefore reports two things separately:

* error against the **bbox heuristic labels** the model was trained on — this is
  the number the old "attachment accuracy 100%" was hiding;
* agreement with an **anatomical reference** (the socket positions the
  procedural baseline returns), which is an independent rule-based target.

No linear score is presented as an accuracy.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_common import BENCH_DIR, markdown_table, write_json  # noqa: E402

from spritemind.dataset.composition_dataset import CompositionDatasetBuilder  # noqa: E402
from spritemind.ml.composition_model import SOCKET_NAMES, SocketPredictorManager  # noqa: E402


def bbox_labels(rgba):
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return {"head": (16, 7), "back": (16, 16), "hand_r": (22, 20), "hand_l": (10, 20)}
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y = int(ys.min())
    cx = (min_x + max_x) // 2
    cy = (min_y + int(ys.max())) // 2
    return {
        "head": (cx, max(2, min_y + 2)),
        "back": (cx, cy),
        "hand_r": (min(30, max_x), min(30, cy + 4)),
        "hand_l": (max(1, min_x), min(30, cy + 4)),
    }


def run(num_samples, seed):
    data = CompositionDatasetBuilder.generate_samples(total_samples=num_samples, output_dir="data/composition_dataset", seed=seed)
    samples = data["val"]
    print(f"held-out samples: {len(samples)} (split by identity seed)", flush=True)

    per_socket = {name: {"abs_err": [], "l2": []} for name in SOCKET_NAMES}
    per_socket_anat = {name: {"abs_err": [], "l2": []} for name in SOCKET_NAMES}
    arm_rows = {}

    for item in samples:
        rgba = item["rgba_array"]
        labels = bbox_labels(rgba)
        preds = SocketPredictorManager.predict_sockets(rgba, engine_mode="trained_model")
        heuristic = SocketPredictorManager.predict_sockets(rgba, engine_mode="procedural_baseline")
        for name in SOCKET_NAMES:
            truth = np.array(labels[name], dtype=float)
            pred = np.array(preds[name], dtype=float)
            anat = np.array(heuristic[name], dtype=float)
            per_socket[name]["abs_err"].append(float(np.abs(pred - truth).mean()))
            per_socket[name]["l2"].append(float(np.linalg.norm(pred - truth)))
            per_socket_anat[name]["abs_err"].append(float(np.abs(pred - anat).mean()))
            per_socket_anat[name]["l2"].append(float(np.linalg.norm(pred - anat)))

    summary = {}
    for name in SOCKET_NAMES:
        summary[name] = {
            "mae_px_vs_bbox_labels": round(float(np.mean(per_socket[name]["abs_err"])), 3),
            "rmse_px_vs_bbox_labels": round(float(np.sqrt(np.mean(np.square(per_socket[name]["l2"])))), 3),
            "mae_px_vs_anatomical_reference": round(float(np.mean(per_socket_anat[name]["abs_err"])), 3),
        }
        arm_rows[name] = summary[name]

    all_l2 = np.concatenate([np.asarray(per_socket[n]["l2"]) for n in SOCKET_NAMES])
    all_anat = np.concatenate([np.asarray(per_socket_anat[n]["l2"]) for n in SOCKET_NAMES])
    payload = {
        "benchmark": "socket_predictor",
        "checkpoint": "socket_predictor_best.pt",
        "val_samples": len(samples),
        "label_source": "alpha-bbox heuristic used by the shipped trainer",
        "results": summary,
        "overall": {
            "mean_euclidean_error_px_vs_bbox_labels": round(float(all_l2.mean()), 3),
            "mean_euclidean_error_px_vs_anatomical_reference": round(float(all_anat.mean()), 3),
            "within_1px_vs_bbox_pct": round(float((all_l2 <= 1.0).mean() * 100), 2),
            "within_2px_vs_bbox_pct": round(float((all_l2 <= 2.0).mean() * 100), 2),
        },
    }
    write_json(BENCH_DIR / "socket_predictor.json", payload)

    rows = [
        {
            "Socket": name,
            "MAE px (bbox labels)": summary[name]["mae_px_vs_bbox_labels"],
            "RMSE px (bbox labels)": summary[name]["rmse_px_vs_bbox_labels"],
            "MAE px (anatomical)": summary[name]["mae_px_vs_anatomical_reference"],
        }
        for name in SOCKET_NAMES
    ]
    table = markdown_table(rows, ["Socket", "MAE px (bbox labels)", "RMSE px (bbox labels)", "MAE px (anatomical)"])
    (BENCH_DIR / "socket_predictor.md").write_text(table + "\n")
    print(table)
    print(json.dumps(payload["overall"], indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()
    run(args.samples, args.seed)
