"""Builds a Visual Critic training corpus from a PixelScript dataset.

Half the records stay clean, half get one synthetic defect injected by
`DefectGenerator`, so the critic learns what a *finished* grammar-v2 sprite looks
like (outlines and shading included) instead of the flat v1 fills.

Run:  python scripts/build_critic_dataset.py --samples 3200
"""
import argparse
import json
import random
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from spritemind.dataset.defects import DefectGenerator  # noqa: E402
from spritemind.pixelscript.parser import PixelScriptParser  # noqa: E402
from spritemind.pixelscript.serializer import PixelScriptSerializer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/datasets/gv2_synthetic_25k/records.jsonl")
    ap.add_argument("--out", default="data/datasets/gv2_critic/records.jsonl")
    ap.add_argument("--samples", type=int, default=3200)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    parser = PixelScriptParser
    rows = []

    with open(args.source, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= args.samples:
                break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean = defects = 0
    val_cut = int(len(rows) * (1.0 - args.val_fraction))

    with out.open("w", encoding="utf-8") as fh:
        for index, item in enumerate(rows):
            text = item.get("script_text")
            if not text:
                continue
            split = "train" if index < val_cut else "val"
            has_defect = rng.random() < 0.5
            info = None
            if has_defect:
                try:
                    program = parser.parse_text(text)
                    corrupted, info = DefectGenerator.inject_defect(program, seed=rng.randrange(1 << 30))
                    text = PixelScriptSerializer.serialize_program(corrupted)
                except Exception:
                    continue
                defects += 1
            else:
                clean += 1
            fh.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "split": split,
                        "script_text": text,
                        "has_defect": has_defect,
                        "defect_info": info,
                    }
                )
                + "\n"
            )

    print(json.dumps({"records": clean + defects, "clean": clean, "defect": defects, "out": str(out)}))


if __name__ == "__main__":
    main()
