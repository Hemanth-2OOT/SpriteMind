"""Runtime latency telemetry behind GET /api/metrics."""
from spritemind.server.app import get_metrics
from spritemind.server.metrics import LATENCY, LatencyMeter, record_generation


def test_percentiles_interpolate_like_numpy():
    meter = LatencyMeter()
    for value in (100, 200, 300, 400, 500):
        meter.record("trained_model", value)
    data = meter.to_dict()
    assert data["p50_ms"] == 300.0
    assert data["p95_ms"] == 480.0
    assert data["mean_ms"] == 300.0


def test_window_keeps_only_the_most_recent_samples():
    meter = LatencyMeter(window=3)
    for value in (10, 20, 30, 40):
        meter.record("procedural_baseline", value)
    assert meter.to_dict()["samples"] == 3
    assert meter.to_dict()["p50_ms"] == 30.0


def test_engines_are_reported_separately():
    meter = LatencyMeter()
    meter.record("trained_model", 400)
    meter.record("procedural_baseline", 20)
    engines = meter.to_dict()["engines"]
    assert engines["trained_model"]["p50_ms"] == 400.0
    assert engines["procedural_baseline"]["p50_ms"] == 20.0


def test_cache_hits_are_counted_but_kept_out_of_the_percentiles():
    LATENCY.reset()
    record_generation({"inference_time_ms": 500.0, "engine_used": "trained_model", "cache_hit": False}, "trained_model")
    record_generation({"inference_time_ms": 0.4, "engine_used": "trained_model", "cache_hit": True}, "trained_model")
    data = LATENCY.to_dict()
    assert data["samples"] == 1
    assert data["cache_hits"] == 1
    assert data["p50_ms"] == 500.0


def test_metrics_endpoint_shape():
    data = get_metrics()
    assert set(data) == {"latency", "cache"}
    assert "p50_ms" in data["latency"]
    assert "p95_ms" in data["latency"]
    assert {"hits", "misses", "size"} <= set(data["cache"])
