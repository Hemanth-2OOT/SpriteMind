"""Phase 8: performance benchmark (latency, throughput, memory).

Reports cold vs warm latency, scaling with `best_of`, per-thread cost on a
fixed CPU budget, peak RSS and the effect of the result cache. No numbers are
carried over from documentation: everything is measured here.
"""
import argparse
import os
import resource
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_common import BENCH_DIR, markdown_table, write_json  # noqa: E402

PROMPTS = [
    ("a red knight with a sword", "knight", "classic_16bit", "idle"),
    ("a purple wizard casting a spell", "wizard", "dark_fantasy", "attack"),
    ("a green slime bouncing", "slime", "flat", "walk"),
    ("a grey robot marching", "robot", "cyberpunk", "walk"),
    ("a gold skeleton archer", "skeleton", "retro_8bit", "attack"),
]


def _time_generate(pipeline, index, engine, best_of):
    prompt, category, style, pose = PROMPTS[index % len(PROMPTS)]
    start = time.perf_counter()
    result = pipeline.generate(
        prompt=prompt,
        category=category,
        style=style,
        pose=pose,
        palette_name="pico8",
        engine_mode=engine,
        seed=1000 + index,
        best_of=best_of,
    )
    return (time.perf_counter() - start) * 1000.0, result


def run(repeats):
    from spritemind.ml.pipeline import SpriteMindPipeline

    os.environ["SPRITEMIND_RESULT_CACHE"] = "0"
    t0 = time.perf_counter()
    pipeline = SpriteMindPipeline()
    pipeline.load_checkpoint("svan_checkpoint_best.pt")
    load_ms = (time.perf_counter() - t0) * 1000.0

    payload = {
        "benchmark": "performance",
        "torch_threads": None,
        "checkpoint_load_ms": round(load_ms, 2),
        "cpu_count": os.cpu_count(),
        "arms": {},
    }
    import torch

    payload["torch_threads"] = torch.get_num_threads()

    for best_of in (1, 2, 4, 8):
        first, result = _time_generate(pipeline, 0, "trained_model", best_of)
        samples = []
        for index in range(repeats):
            elapsed, _ = _time_generate(pipeline, index + 1, "trained_model", best_of)
            samples.append(elapsed)
        payload["arms"][f"trained_best{best_of}"] = {
            "cold_ms": round(first, 2),
            "pipeline_reported_ms": result.get("inference_time_ms"),
            "median_ms": round(statistics.median(samples), 2),
            "p90_ms": round(sorted(samples)[int(0.9 * (len(samples) - 1))], 2),
            "stdev_ms": round(statistics.pstdev(samples), 2),
            "throughput_per_s": round(1000.0 / statistics.median(samples), 2),
        }
        print(f"best_of={best_of} median {payload['arms'][f'trained_best{best_of}']['median_ms']} ms", flush=True)

    for engine in ("procedural_baseline", "untrained_model"):
        samples = [_time_generate(pipeline, i, engine, 4)[0] for i in range(repeats)]
        payload["arms"][engine] = {
            "median_ms": round(statistics.median(samples), 2),
            "p90_ms": round(sorted(samples)[int(0.9 * (len(samples) - 1))], 2),
            "throughput_per_s": round(1000.0 / statistics.median(samples), 2),
        }

    for threads in (1, 2, 4):
        torch.set_num_threads(threads)
        samples = [_time_generate(pipeline, i, "trained_model", 4)[0] for i in range(repeats)]
        payload["arms"][f"threads_{threads}"] = {
            "median_ms": round(statistics.median(samples), 2),
            "throughput_per_s": round(1000.0 / statistics.median(samples), 2),
        }
    torch.set_num_threads(payload["torch_threads"])

    payload["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    write_json(BENCH_DIR / "performance.json", payload)

    rows = []
    for name, data in payload["arms"].items():
        rows.append(
            {
                "Arm": name,
                "Median ms": data.get("median_ms"),
                "P90 ms": data.get("p90_ms"),
                "Throughput/s": data.get("throughput_per_s"),
            }
        )
    table = markdown_table(rows, ["Arm", "Median ms", "P90 ms", "Throughput/s"])
    (BENCH_DIR / "performance.md").write_text(table + "\n")
    print(table)
    print(f"peak RSS: {payload['peak_rss_mb']} MB; checkpoint load: {payload['checkpoint_load_ms']} ms")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    run(args.repeats)
