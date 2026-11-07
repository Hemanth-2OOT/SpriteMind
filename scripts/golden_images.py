"""Golden-image regression CLI.

    python scripts/golden_images.py            # verify (exit 1 on drift)
    python scripts/golden_images.py --update   # accept the current renders
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from spritemind.ml.goldens import (  # noqa: E402
    CATEGORIES,
    GOLDEN_DIR,
    HAMMING_TOLERANCE,
    generate_all,
    verify,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="accept the current renders as the new goldens")
    args = parser.parse_args()

    if args.update:
        hashes = generate_all()
        print(f"wrote {len(hashes)} goldens to {GOLDEN_DIR.relative_to(ROOT)}")
        return 0

    failures = verify()
    if failures:
        print("GOLDEN IMAGE REGRESSION")
        for failure in failures:
            print(f" - {failure}")
        print("If the change is intended, run python scripts/golden_images.py --update and review the PNG diff.")
        return 1
    print(f"golden images ok ({len(CATEGORIES)} categories, tolerance {HAMMING_TOLERANCE} bits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
