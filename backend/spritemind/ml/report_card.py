"""Per-category report card built from the stored benchmark gallery.

The benchmark summary is one number per engine across every category, which hides
the fact that some categories are far weaker than others (animals and fantasy
creatures consistently trail knights). This module aggregates the same gallery
records per category so the weak spots are visible without re-running anything.
"""
from typing import Any, Dict, List, Optional

ENGINES = ("trained_model", "procedural_baseline", "untrained_model")

# Weighted into a single 0-100 score. Structural consistency and palette adherence
# are pass/fail in practice, so the discriminating metrics carry most of the weight.
WEIGHTS = {
    "structural_consistency": 0.35,
    "silhouette_iou": 0.30,
    "colour_faithfulness": 0.20,
    "valid_action_rate": 0.15,
}


def _mean(values: List[Any]) -> Optional[float]:
    numbers = [float(v) for v in values if isinstance(v, (int, float))]
    return round(sum(numbers) / len(numbers), 3) if numbers else None


def score_category(metrics: Dict[str, Optional[float]]) -> Optional[float]:
    """Weighted 0-100 score; unmeasured components are skipped, not zeroed."""
    total_weight = 0.0
    total = 0.0
    for key, weight in WEIGHTS.items():
        value = metrics.get(key)
        if value is None:
            continue
        normalised = value / 100.0 if key != "silhouette_iou" else value
        total += weight * normalised
        total_weight += weight
    if total_weight == 0:
        return None
    return round(100.0 * total / total_weight, 1)


def summarise_gallery(
    gallery: List[Dict[str, Any]], engine: str = "trained_model"
) -> Dict[str, Any]:
    """Per-category means plus a score, ordered weakest first."""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for record in gallery:
        output = (record.get("outputs") or {}).get(engine)
        if not output:
            continue
        buckets.setdefault(record.get("category") or "unknown", []).append(output.get("metrics") or {})

    rows = []
    for category, metrics_list in buckets.items():
        metrics = {
            "structural_consistency": _mean([m.get("structural_consistency") for m in metrics_list]),
            "silhouette_iou": _mean([m.get("silhouette_iou") for m in metrics_list]),
            "colour_faithfulness": _mean([m.get("colour_faithfulness") for m in metrics_list]),
            "valid_action_rate": _mean([m.get("valid_action_rate") for m in metrics_list]),
            "palette_adherence": _mean([m.get("palette_adherence") for m in metrics_list]),
            "latency_ms": _mean([m.get("latency_ms") for m in metrics_list]),
        }
        rows.append({"category": category, "prompts": len(metrics_list), "score": score_category(metrics), **metrics})

    rows.sort(key=lambda row: (row["score"] is None, row["score"] if row["score"] is not None else 0.0))
    scores = [row["score"] for row in rows if row["score"] is not None]
    return {
        "engine": engine,
        "categories": rows,
        "worst_category": rows[0]["category"] if rows else None,
        "mean_score": _mean(scores),
    }
