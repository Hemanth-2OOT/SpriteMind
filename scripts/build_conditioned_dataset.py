"""Builds the palette-conditioned, element-labelled corpus (R6 + R9) under a frozen R10 split.

    python scripts/build_conditioned_dataset.py --name cond_v1_12k --samples 15000 --seed 20261105

Differences from `build_expanded_dataset.py`, all deliberate:

* every sample records the palette it was drawn in, and the program is generated
  *in that palette*, so palette index N in a record means the same colour the
  prompt asked for (previously every record was pico8 and every palette label 0);
* requested colours are drawn from `renderable_colours(palette)`, verified by the
  colour-faithfulness metric, so no record asks for a colour its palette cannot show;
* element labels are assigned at `--element-rate` (default 0.5) instead of a 25%
  tail, because the old corpus left 75% of records with `element: null`;
* combinations named in the held-out spec go to split `heldout` and are never
  written into train/val;
* `metadata.json` records provenance: rule text, per-palette and per-element
  counts, split counts, spec digest and the git commit of the builder.
"""
import argparse
import json
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from spritemind.dataset.conditioning import (  # noqa: E402
    ELEMENT_RGB,
    describe,
    renderable_colours,
)
from spritemind.dataset.generator import (  # noqa: E402
    ACCESSORIES,
    CATEGORIES,
    POSES,
    PROPORTIONS,
    STYLES,
    WEAPONS,
    CharacterGenerator,
)
from spritemind.dataset.splits import heldout_reason, load_spec, spec_digest  # noqa: E402
from spritemind.pixelscript.serializer import PixelScriptSerializer  # noqa: E402
from spritemind.utils.palettes import PALETTES  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="cond_v1_12k")
    parser.add_argument("--samples", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--colour-rate", type=float, default=0.6)
    parser.add_argument("--element-rate", type=float, default=0.5)
    parser.add_argument("--defect-rate", type=float, default=0.0)
    parser.add_argument("--train-ratio", type=float, default=0.88)
    parser.add_argument("--spec", default="data/splits/heldout_v1.json")
    parser.add_argument("--render-samples", type=int, default=8)
    args = parser.parse_args()

    started = time.perf_counter()
    spec = load_spec(ROOT / args.spec)
    out_dir = ROOT / "data" / "datasets" / args.name
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"refusing to overwrite existing dataset: {out_dir}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)

    palettes = list(PALETTES)
    colour_pool: Dict[str, List[str]] = {}
    for palette in palettes:
        names = renderable_colours(palette)
        colour_pool[palette] = names
        print(f"{palette:14s} renderable colours: {len(names):2d}", flush=True)
    usable = [p for p in palettes if colour_pool[p]]
    elements = sorted(ELEMENT_RGB)

    rng = random.Random(args.seed)
    split_counts: Counter = Counter()
    palette_counts: Counter = Counter()
    colour_counts: Counter = Counter()
    element_counts: Counter = Counter()
    category_counts: Counter = Counter()
    heldout_counts: Counter = Counter()

    records_path = out_dir / "records.jsonl"
    with open(records_path, "w", encoding="utf-8") as handle:
        for i in range(args.samples):
            category = CATEGORIES[i % len(CATEGORIES)]
            style = rng.choice(STYLES)
            pose = rng.choice(POSES)
            proportion = rng.choice(PROPORTIONS)
            weapon = rng.choice(WEAPONS)
            accessory = rng.choice(ACCESSORIES)

            palette = usable[i % len(usable)] if i < len(usable) else rng.choice(usable)
            colour = rng.choice(colour_pool[palette]) if rng.random() < args.colour_rate else None
            element = rng.choice(elements) if rng.random() < args.element_rate else None
            prompt = describe(category, colour=colour, element=element, weapon=weapon if weapon != "none" else None)

            probe = {"category": category, "style": style, "pose": pose, "element": element}
            reason = heldout_reason(probe, spec)
            if reason:
                split = "heldout"
                heldout_counts[reason] += 1
            else:
                character_id = i // 5
                identity = (character_id * 2654435761 + args.seed) % 100
                split = "train" if identity < int(args.train_ratio * 100) else "val"

            sample_seed = args.seed + i * 17
            prog = CharacterGenerator.generate_character(
                category=category,
                style=style,
                pose=pose,
                palette_name=palette,
                width=32,
                height=32,
                seed=sample_seed,
                proportion=proportion,
                weapon=weapon,
                accessory=accessory,
                colour=colour,
                element=element,
            )

            record = {
                "id": f"cond_{i:06d}",
                "character_id": i // 5,
                "split": split,
                "heldout_reason": reason,
                "category": category,
                "style": style,
                "pose": pose,
                "proportion": proportion,
                "prompt": prompt,
                "palette": palette,
                "colour": colour,
                "element": element,
                "weapon": weapon,
                "accessory": accessory,
                "has_defect": False,
                "defect_info": None,
                "actions": [a.to_dict() for a in prog.actions],
                "script_text": PixelScriptSerializer.serialize_program(prog),
            }
            handle.write(json.dumps(record) + "\n")

            split_counts[split] += 1
            palette_counts[palette] += 1
            category_counts[category] += 1
            colour_counts[colour or "none"] += 1
            element_counts[element or "none"] += 1

    metadata = {
        "name": args.name,
        "builder": "scripts/build_conditioned_dataset.py",
        "builder_commit": git_commit(),
        "seed": args.seed,
        "total_samples": args.samples,
        "splits": dict(split_counts),
        "palette_distribution": dict(palette_counts),
        "colour_distribution": dict(colour_counts),
        "element_distribution": dict(element_counts),
        "category_distribution": dict(category_counts),
        "labelling_rules": {
            "palette": "cycled uniformly over the palettes with at least one renderable colour, then sampled randomly",
            "colour": f"uniform over renderable_colours(palette), assigned to {args.colour_rate:.0%} of records",
            "element": f"uniform over {elements}, assigned to {args.element_rate:.0%} of records",
            "program": "CharacterGenerator.generate_character(..., palette_name=palette) so colour indices are palette-correct",
            "split": "held-out spec combinations -> 'heldout'; otherwise identity-hash 88/12 train/val",
        },
        "heldout": {
            "spec": args.spec,
            "spec_digest": spec_digest(spec),
            "records": dict(heldout_counts),
            "triples": len(spec["heldout_triples"]),
            "category_element_pairs": len(spec["heldout_category_element"]),
        },
        "leakage_prevention": "held-out combinations excluded from train/val + identity hash partition",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"dataset: {out_dir}")
    print(f"splits: {dict(split_counts)}")
    print(f"elements labelled: {args.samples - element_counts['none']}/{args.samples}")
    print(f"elapsed: {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
