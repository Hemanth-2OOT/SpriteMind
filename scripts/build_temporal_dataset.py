"""Builds a temporal (animation) dataset with the current generator, so frames carry
the latest grammar primitives (SHADE, OUTLINE, AURA).

    python scripts/build_temporal_dataset.py --name temporal_dataset_gv2 --sequences 4000
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from spritemind.dataset.temporal_dataset import TemporalDatasetGenerator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="temporal_dataset_gv2")
    parser.add_argument("--sequences", type=int, default=4000)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = ROOT / "data" / args.name
    meta = TemporalDatasetGenerator.build_dataset(
        output_dir=str(out_dir), total_sequences=args.sequences, val_ratio=args.val_ratio
    )
    meta["dataset_name"] = args.name
    (out_dir / "dataset_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built {out_dir.name}: {meta['train_sequences']} train / {meta['val_sequences']} val")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
