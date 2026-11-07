"""Per-category report card aggregated from stored benchmark records."""
from spritemind.ml.report_card import score_category, summarise_gallery

METRICS = {
    "structural_consistency": 100.0,
    "silhouette_iou": 0.5,
    "colour_faithfulness": 50.0,
    "valid_action_rate": 100.0,
    "palette_adherence": 100.0,
    "latency_ms": 300.0,
}


def _record(category, **overrides):
    metrics = {**METRICS, **overrides}
    return {"category": category, "outputs": {"trained_model": {"metrics": metrics}}}


def test_score_is_weighted_and_normalised_to_100():
    assert score_category(METRICS) == 75.0


def test_missing_metrics_are_skipped_not_zeroed():
    partial = {"structural_consistency": 100.0, "valid_action_rate": 100.0}
    assert score_category(partial) == 100.0


def test_categories_are_averaged_and_sorted_weakest_first():
    gallery = [
        _record("knight", silhouette_iou=0.6),
        _record("knight", silhouette_iou=0.4),
        _record("slime", silhouette_iou=0.2, colour_faithfulness=0.0),
    ]
    card = summarise_gallery(gallery)
    assert [row["category"] for row in card["categories"]] == ["slime", "knight"]
    assert card["worst_category"] == "slime"
    assert card["categories"][1]["prompts"] == 2
    assert card["categories"][1]["silhouette_iou"] == 0.5


def test_none_metric_values_are_ignored():
    gallery = [_record("robot", colour_faithfulness=None), _record("robot", colour_faithfulness=10.0)]
    card = summarise_gallery(gallery)
    assert card["categories"][0]["colour_faithfulness"] == 10.0


def test_records_without_the_requested_engine_are_skipped():
    gallery = [{"category": "knight", "outputs": {"procedural_baseline": {"metrics": METRICS}}}]
    card = summarise_gallery(gallery, engine="trained_model")
    assert card["categories"] == []
    assert card["worst_category"] is None
    assert card["mean_score"] is None
