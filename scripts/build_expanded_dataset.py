"""Builds a large synthetic PixelScript dataset and a flat JSON export of it.

    python scripts/build_expanded_dataset.py --samples 25000 --seed 7

Writes the trainable layout under `data/datasets/<name>/` (records.jsonl, metadata.json,
samples/) and, unless `--json-out ""` is passed, a single JSON array of every record at
`--json-out` for inspection or external use.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from spritemind.dataset.storage import DatasetManager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="expanded_synthetic_25k")
    parser.add_argument("--samples", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--defect-rate", type=float, default=0.0)
    parser.add_argument("--json-out", default="data/expanded_synthetic_dataset.json")
    parser.add_argument("--render-samples", type=int, default=24)
    args = parser.parse_args()

    started = time.perf_counter()
    manager = DatasetManager(str(ROOT / "data" / "datasets"))
    stats = manager.generate_dataset(
        name=args.name,
        total_samples=args.samples,
        defect_rate=args.defect_rate,
        seed=args.seed,
        render_sample_limit=args.render_samples,
    )
    dataset_dir = manager.get_dataset_dir(args.name)
    records_path = dataset_dir / "records.jsonl"
    print(f"dataset: {dataset_dir} ({stats['total_samples']} samples, splits {stats['splits']})")

    if args.json_out:
        out_path = ROOT / args.json_out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        with open(records_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f)
        print(f"json:    {out_path} ({len(records)} records, {out_path.stat().st_size / 1e6:.1f} MB)")

    print(f"records: {records_path} ({records_path.stat().st_size / 1e6:.1f} MB)")
    print(f"elapsed: {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
