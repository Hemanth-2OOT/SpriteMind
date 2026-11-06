"""Per-endpoint latency profile (p50/p95/p99) for the SpriteMind HTTP API.

Runs the FastAPI app in-process so numbers exclude network noise, then times
the endpoints a client actually waits on. Results are written to
data/benchmarks/latency_report.json and printed as a table.

    python scripts/latency_report.py --requests 20 --engine trained_model
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from spritemind.server.app import app  # noqa: E402

PROMPTS = [
    "a knight in plate armor with a sword",
    "an ice wizard casting a frost bolt",
    "a slime that hops and splits",
    "a robot with a laser cannon",
    "a forest archer with a bow",
    "a fire demon wielding an axe",
]


def percentile(values: List[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def summarise(samples: List[float]) -> Dict[str, float]:
    return {
        "count": len(samples),
        "p50_ms": round(percentile(samples, 50), 1),
        "p95_ms": round(percentile(samples, 95), 1),
        "p99_ms": round(percentile(samples, 99), 1),
        "mean_ms": round(statistics.fmean(samples), 1),
        "max_ms": round(max(samples), 1),
    }


def timed(fn: Callable[[], Any]) -> float:
    started = time.perf_counter()
    fn()
    return (time.perf_counter() - started) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=12, help="samples per endpoint")
    parser.add_argument("--engine", default=None, help="engine override for /api/generate")
    parser.add_argument("--out", default="data/benchmarks/latency_report.json")
    args = parser.parse_args()

    client = TestClient(app)
    results: Dict[str, Dict[str, float]] = {}

    def generate_call(index: int) -> Callable[[], Any]:
        payload = {"prompt": PROMPTS[index % len(PROMPTS)], "seed": 1000 + index}
        if args.engine:
            payload["engine"] = args.engine
        return lambda: client.post("/api/generate", json=payload)

    generate_paths = {
        "POST /api/generate": lambda i: generate_call(i * 17 + 3),
        "POST /api/action_set/plan": lambda i: (
            lambda: client.post("/api/action_set/plan", json={"prompt": PROMPTS[i % len(PROMPTS)]})
        ),
        "POST /api/action_set": lambda i: (
            lambda: client.post(
                "/api/action_set",
                json={"prompt": PROMPTS[i % len(PROMPTS)], "seed": 2000 + i, "actions": ["idle", "walk"]},
            )
        ),
        "GET /api/health": lambda i: (lambda: client.get("/api/health")),
        "GET /api/models": lambda i: (lambda: client.get("/api/models")),
    }

    for name, factory in generate_paths.items():
        samples: List[float] = []
        for i in range(args.requests):
            samples.append(timed(factory(i)))
        results[name] = summarise(samples)
        stats = results[name]
        print(f"{name:<38} p50 {stats['p50_ms']:>8} ms  p95 {stats['p95_ms']:>8} ms  n {stats['count']}")

    warm_payload = {"prompt": PROMPTS[0], "seed": 4242}
    if args.engine:
        warm_payload["engine"] = args.engine
    client.post("/api/generate", json=warm_payload)
    cached: List[float] = []
    for _ in range(args.requests):
        cached.append(timed(lambda: client.post("/api/generate", json=warm_payload)))
    results["POST /api/generate (result cache hit)"] = summarise(cached)
    stats = results["POST /api/generate (result cache hit)"]
    print(f"{'POST /api/generate (result cache hit)':<38} p50 {stats['p50_ms']:>8} ms  p95 {stats['p95_ms']:>8} ms  n {stats['count']}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "engine": args.engine or "default",
                "requests_per_endpoint": args.requests,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "endpoints": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
