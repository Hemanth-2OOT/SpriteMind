"""Freeze a validation prompt set for reranker tuning (R12).

Reranking weights must be tuned away from the frozen held-out test set. This
writes `data/splits/validation_v1.json`: coloured prompts drawn from the
combination space the corpus does contain, none of which is a held-out triple or
a held-out (category, element) pair. Every held-out evaluation prompt is also
checked for collision, so tuning and testing cannot overlap.

    python scripts/build_validation_prompts.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from spritemind.dataset.conditioning import renderable_colours  # noqa: E402
from spritemind.dataset.generator import CATEGORIES, POSES, STYLES  # noqa: E402
from spritemind.dataset.splits import heldout_reason, load_spec, spec_digest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "data/splits/heldout_v1.json"
OUT_PATH = ROOT / "data/splits/validation_v1.json"
PALETTE = "pico8"
COUNT = 24


def build(spec):
    heldout_triples = {tuple(t) for t in spec["heldout_triples"]}
    colours = renderable_colours(PALETTE)
    allowed = []
    for category in CATEGORIES:
        for style in STYLES:
            for pose in POSES:
                if (category, style, pose) in heldout_triples:
                    continue
                if heldout_reason({"category": category, "style": style, "pose": pose}, spec):
                    continue
                allowed.append((category, style, pose))
    if len(allowed) < COUNT:
        raise SystemExit(f"only {len(allowed)} usable combinations, need {COUNT}")
    step = max(1, len(allowed) // COUNT)
    prompts = []
    for i in range(COUNT):
        category, style, pose = allowed[(i * step) % len(allowed)]
        colour = colours[(i * 11 + 4) % len(colours)]
        prompts.append({"prompt": f"a {colour} {category.replace('_', ' ')}", "category": category,
                        "style": style, "pose": pose, "palette": PALETTE, "colour": colour})
    return {"version": 1, "name": "validation_v1", "palette": PALETTE,
            "rationale": "Coloured prompts disjoint from data/splits/heldout_v1.json; used only to tune reranker weights.",
            "prompts": prompts}


def main() -> int:
    spec = load_spec(SPEC_PATH)
    data = build(spec)
    data["digest"] = spec_digest(data)
    keys = {(p["category"], p["style"], p["pose"]) for p in data["prompts"]}
    heldout = {tuple(t) for t in spec["heldout_triples"]}
    assert not keys & heldout, "validation prompts overlap the frozen held-out triples"
    OUT_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH} · {len(data['prompts'])} prompts · digest {data['digest'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
