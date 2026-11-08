"""Phase 7: determinism benchmark across fresh processes.

Runs the same generation set in three separate Python processes (different
`PYTHONHASHSEED`) and in-process twice, then compares the SHA-256 of every
rendered frame. A generation is deterministic only if every hash matches.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_common import BENCH_DIR, markdown_table, write_json  # noqa: E402

CASES = [
    {"category": "knight", "style": "classic_16bit", "pose": "idle", "palette": "pico8", "engine": "trained_model"},
    {"category": "wizard", "style": "dark_fantasy", "pose": "attack", "palette": "dawnbringer16", "engine": "trained_model"},
    {"category": "robot", "style": "cyberpunk", "pose": "run", "palette": "cyberpunk", "engine": "trained_model"},
    {"category": "slime", "style": "flat", "pose": "idle", "palette": "pico8", "engine": "untrained_model"},
    {"category": "zombie", "style": "retro_8bit", "pose": "walk", "palette": "gameboy", "engine": "procedural_baseline"},
]
SEEDS = [0, 42, 1337]


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _run_cases(seed_offset: int = 0) -> dict:
    from spritemind.ml.pipeline import SpriteMindPipeline

    pipeline = SpriteMindPipeline()
    pipeline.load_checkpoint("svan_checkpoint_best.pt")
    os.environ.setdefault("SPRITEMIND_RESULT_CACHE", "0")
    out = {}
    for case in CASES:
        for seed in SEEDS:
            result = pipeline.generate(
                prompt=f"{case['category']} {case['pose']}",
                category=case["category"],
                style=case["style"],
                pose=case["pose"],
                palette_name=case["palette"],
                engine_mode=case["engine"],
                seed=seed + seed_offset,
                best_of=2,
            )
            script = result["frames"][0].get("script_text", "")
            image = result["frames"][0].get("image_base64", "")
            out[f"{case['category']}/{case['pose']}/{seed}"] = (_digest(script), _digest(image))
    return out


CHILD = """
import json, sys
sys.path.insert(0, {scripts!r})
sys.path.insert(0, {backend!r})
from benchmark_determinism import _run_cases
print(json.dumps(_run_cases()))
"""


def run():
    in_process_a = _run_cases()
    in_process_b = _run_cases()

    children = []
    for hash_seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, SPRITEMIND_RESULT_CACHE="0")
        proc = subprocess.run(
            [sys.executable, "-c", CHILD.format(scripts=str(Path(__file__).resolve().parent), backend=str(ROOT / "backend"))],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(ROOT),
        )
        if proc.returncode != 0:
            raise SystemExit(proc.stderr[-2000:])
        children.append(json.loads(proc.stdout))

    def compare(a, b):
        mismatched = [key for key in a if list(a[key]) != list(b.get(key, []))]
        return {"cases": len(a), "mismatches": len(mismatched), "examples": mismatched[:5]}

    rows = [
        {"Comparison": "in-process run 1 vs run 2", **compare(in_process_a, in_process_b)},
        {"Comparison": "process PYTHONHASHSEED=0 vs 1", **compare(children[0], children[1])},
        {"Comparison": "process PYTHONHASHSEED=0 vs 12345", **compare(children[0], children[2])},
        {"Comparison": "in-process vs child process", **compare(in_process_a, children[0])},
    ]
    payload = {
        "benchmark": "determinism",
        "cases": len(in_process_a),
        "seeds": SEEDS,
        "comparisons": rows,
        "deterministic": all(row["mismatches"] == 0 for row in rows),
        "hashes": {"in_process": in_process_a, "children": children},
    }
    write_json(BENCH_DIR / "determinism.json", payload)
    table = markdown_table(rows, ["Comparison", "cases", "mismatches", "examples"])
    (BENCH_DIR / "determinism.md").write_text(table + "\n")
    print(table)
    print(f"\ndeterministic: {payload['deterministic']}")
    return payload


if __name__ == "__main__":
    run()
