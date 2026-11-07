"""Runtime latency telemetry: a bounded ring buffer with p50/p95 per engine."""
from collections import deque
from typing import Any, Deque, Dict, List, Optional

WINDOW = 200


def _percentile(sorted_values: List[float], fraction: float) -> Optional[float]:
    """Linear-interpolated percentile, matching numpy's default."""
    if not sorted_values:
        return None
    position = fraction * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return round(sorted_values[low] * (1 - weight) + sorted_values[high] * weight, 2)


class LatencyMeter:
    """Keeps the last `window` generation latencies and reports percentiles."""

    def __init__(self, window: int = WINDOW) -> None:
        self.window = window
        self._samples: Deque[Dict[str, Any]] = deque(maxlen=window)
        self._cache_hits = 0

    def record(self, engine: str, milliseconds: float) -> None:
        self._samples.append({"engine": engine or "unknown", "ms": float(milliseconds)})

    def note_cache_hit(self) -> None:
        self._cache_hits += 1

    def to_dict(self) -> Dict[str, Any]:
        overall = sorted(s["ms"] for s in self._samples)
        by_engine: Dict[str, List[float]] = {}
        for sample in self._samples:
            by_engine.setdefault(sample["engine"], []).append(sample["ms"])
        return {
            "window": self.window,
            "samples": len(overall),
            "cache_hits": self._cache_hits,
            "p50_ms": _percentile(overall, 0.50),
            "p95_ms": _percentile(overall, 0.95),
            "mean_ms": round(sum(overall) / len(overall), 2) if overall else None,
            "engines": {
                engine: {
                    "samples": len(values),
                    "p50_ms": _percentile(sorted(values), 0.50),
                    "p95_ms": _percentile(sorted(values), 0.95),
                }
                for engine, values in sorted(by_engine.items())
            },
        }

    def reset(self) -> None:
        self._samples.clear()
        self._cache_hits = 0


LATENCY = LatencyMeter()


def record_generation(result: Dict[str, Any], engine: str) -> None:
    """Records one real generation; replay hits are counted but kept out of the percentiles."""
    elapsed = result.get("inference_time_ms")
    if elapsed is None:
        return
    cache_hit = bool(result.get("cache_hit"))
    if cache_hit:
        LATENCY.note_cache_hit()
        return
    LATENCY.record(result.get("engine_used") or engine, float(elapsed))
