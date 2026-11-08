"""Freezes the R10 held-out split (combinations that never appear in train/val).

    python scripts/build_heldout_split.py --seed 20261105

Writes `data/splits/heldout_v1.json` once. Re-running with `--force` writes a new
file under `--out`, so a spec that a checkpoint was evaluated against can never
silently change.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from spritemind.dataset.conditioning import ELEMENT_RGB  # noqa: E402
from spritemind.dataset.generator import CATEGORIES, POSES, STYLES  # noqa: E402
from spritemind.dataset.splits import SPEC_VERSION, spec_digest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--triples-per-category", type=int, default=3)
    parser.add_argument("--out", default="data/splits/heldout_v1.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_path = ROOT / args.out
    if out_path.exists() and not args.force:
        print(f"refusing to overwrite frozen spec: {out_path} (pass --force with a new --out)")
        return 1

    rng = random.Random(args.seed)
    triples = []
    for category in CATEGORIES:
        combos = [(category, style, pose) for style in STYLES for pose in POSES]
        triples.extend(rng.sample(combos, args.triples_per_category))

    elements = sorted(ELEMENT_RGB)
    category_element = [(category, rng.choice(elements)) for category in CATEGORIES]

    spec = {
        "version": SPEC_VERSION,
        "name": "heldout_v1",
        "created_with_seed": args.seed,
        "rationale": (
            "Combinations excluded from train and val so held-out scores measure unseen "
            "category/style/pose compositions and unseen category/element pairings rather "
            "than unseen seeds. Selected once, deterministically, before any training run."
        ),
        "heldout_triples": [list(t) for t in triples],
        "heldout_category_element": [list(p) for p in category_element],
    }
    spec["digest"] = spec_digest(spec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"spec: {out_path} ({len(triples)} triples, {len(category_element)} category/element pairs)")
    print(f"digest: {spec['digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
