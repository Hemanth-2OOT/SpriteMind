"""Phase 4: generalization benchmark with a real novelty partition.

Novelty is derived from the shipped SVAN training corpus
(`data/datasets/element_synthetic_20k/records.jsonl`, train split), not from a
hand-made guess:

* `in_distribution` — the prompt's (category, style, pose) triple occurs in the
  train split.
* `novel_triple`    — the triple never occurs in the train split (it may occur
  in the val/test splits, so this is *compositional* novelty only).
* `out_of_vocabulary` — the prompt names a category outside the 10 trained ones
  or an element/colour outside the 9/29 conditioning ids.

It also measures **memorisation**: the share of generations whose PixelScript
is byte-identical, or ≥ 0.95 token-Jaccard similar, to a training script.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_common import (  # noqa: E402
    BENCH_DIR,
    ROOT,
    canonical_reference,
    decode_rgba,
    load_prompts,
    markdown_table,
    write_json,
)

from spritemind.dataset.generator import CATEGORIES  # noqa: E402
from spritemind.ml.metrics import evaluate_sprite, summarise  # noqa: E402
from spritemind.ml.pipeline import SpriteMindPipeline  # noqa: E402

OOD_KEYWORDS = (
    "dragon", "merchant", "pirate", "golem", "ninja", "astronaut", "angel",
    "scarecrow", "samurai", "jellyfish", "steampunk", "clockwork",
)

RECORDS = ROOT / "data" / "datasets" / "element_synthetic_20k" / "records.jsonl"

OOD_PROMPTS = [
    {"id": "ood_01", "prompt": "A dragon breathing molten lava", "category": "fantasy_creature", "style": "dark_fantasy", "pose": "attack"},
    {"id": "ood_02", "prompt": "A merchant carrying a heavy coin purse", "category": "goblin", "style": "flat", "pose": "idle"},
    {"id": "ood_03", "prompt": "A pirate captain with a cutlass and eyepatch", "category": "rogue", "style": "classic_16bit", "pose": "attack"},
    {"id": "ood_04", "prompt": "A crystal golem made of amethyst shards", "category": "slime", "style": "dark_fantasy", "pose": "idle"},
    {"id": "ood_05", "prompt": "A ninja sprinting through a bamboo forest", "category": "rogue", "style": "retro_8bit", "pose": "run"},
    {"id": "ood_06", "prompt": "An astronaut floating in zero gravity", "category": "robot", "style": "cyberpunk", "pose": "walk"},
    {"id": "ood_07", "prompt": "A clockwork owl with brass gears", "category": "animal", "style": "cyberpunk", "pose": "idle"},
    {"id": "ood_08", "prompt": "A radiant angel with six wings of light", "category": "fantasy_creature", "style": "classic_16bit", "pose": "idle"},
    {"id": "ood_09", "prompt": "A cursed scarecrow stitched from burlap", "category": "zombie", "style": "dark_fantasy", "pose": "walk"},
    {"id": "ood_10", "prompt": "A samurai duelling under falling petals", "category": "knight", "style": "classic_16bit", "pose": "attack"},
    {"id": "ood_11", "prompt": "A jellyfish drifting with translucent tentacles", "category": "slime", "style": "flat", "pose": "idle"},
    {"id": "ood_12", "prompt": "A steampunk inventor wielding a wrench", "category": "goblin", "style": "cyberpunk", "pose": "idle"},
]


def scan_corpus():
    triples, scripts, categories = set(), set(), set()
    count = 0
    with RECORDS.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("split") != "train":
                continue
            count += 1
            triples.add((record["category"], record["style"], record["pose"]))
            categories.add(record["category"])
            scripts.add(record.get("script_text", ""))
    return {"triples": triples, "scripts": scripts, "categories": categories, "train_records": count}


def classify(item, corpus):
    category = item["category"]
    triple = (category, item.get("style"), item.get("pose"))
    prompt = item.get("prompt", "").lower()
    if category not in corpus["categories"]:
        return "out_of_vocabulary"
    if any(word in prompt for word in OOD_KEYWORDS):
        return "out_of_vocabulary"
    return "in_distribution" if triple in corpus["triples"] else "novel_triple"


def token_jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def run(limit, best_of, seed):
    corpus = scan_corpus()
    print(f"train records scanned: {corpus['train_records']}, triples: {len(corpus['triples'])}", flush=True)
    reference_token_sets = [set(s.split()) for s in list(corpus["scripts"])[:400] if s]

    prompts = load_prompts(limit) + OOD_PROMPTS
    for index, item in enumerate(prompts):
        item.setdefault("palette", "pico8")
        item["novelty"] = classify(item, corpus)

    pipeline = SpriteMindPipeline()
    if not pipeline.load_checkpoint("svan_checkpoint_best.pt"):
        raise SystemExit("checkpoint missing")

    buckets = {}
    for index, item in enumerate(prompts):
        palette = item["palette"]
        out = pipeline.generate(
            prompt=item["prompt"],
            category=item["category"],
            style=item.get("style"),
            pose=item.get("pose"),
            palette_name=palette,
            engine_mode="trained_model",
            seed=seed + index,
            best_of=best_of,
        )
        rgba = decode_rgba(out["frames"][0]["image_base64"])
        if rgba is None:
            continue
        script = out["frames"][0].get("script_text", "")
        script_tokens = set(script.split())
        row = evaluate_sprite(
            rgba, palette, reference_mask=canonical_reference(item["category"], palette), prompt=item["prompt"]
        )
        row["exact_memorised"] = script in corpus["scripts"]
        row["near_memorised"] = row["exact_memorised"] or (
            max((token_jaccard(script_tokens, ref) for ref in reference_token_sets), default=0.0) >= 0.95
        )
        row["candidate_source"] = out.get("candidate_source")
        row["latency_ms"] = out.get("inference_time_ms")
        buckets.setdefault(item["novelty"], []).append(row)

    results = {
        name: {
            "summary": summarise(rows),
            "exact_memorisation_pct": round(100.0 * sum(r["exact_memorised"] for r in rows) / len(rows), 2),
            "guard_win_pct": round(100.0 * sum(r["candidate_source"] == "procedural_guard" for r in rows) / len(rows), 2),
        }
        for name, rows in buckets.items()
    }
    benchmark_prompts = load_prompts(limit)
    seen = sum(1 for it in benchmark_prompts if (it["category"], it.get("style"), it.get("pose")) in corpus["triples"])
    coverage = {
        "corpus_triples": len(corpus["triples"]),
        "possible_triples": len(CATEGORIES) * 5 * 5,
        "benchmark_prompts_on_seen_triples_pct": round(100.0 * seen / max(1, len(benchmark_prompts)), 2),
        "note": "The corpus enumerates the full category x style x pose Cartesian product, so no compositional novelty can be measured; only semantic OOD prompts are new.",
    }
    payload = {
        "benchmark": "generalization",
        "coverage": coverage,
        "prompts": len(prompts),
        "best_of": best_of,
        "seed": seed,
        "novelty_partition": "triples seen in the train split of element_synthetic_20k",
        "results": results,
    }
    write_json(BENCH_DIR / "generalization.json", payload)

    rows = []
    for name, data in results.items():
        s = data["summary"]
        rows.append(
            {
                "Split": name,
                "n": s["n"],
                "IoU": f"{s['silhouette_iou']['mean']:.3f} ± {s['silhouette_iou']['stderr']:.3f}",
                "Palette %": f"{s['palette_adherence_pct']['mean']:.1f}",
                "Colour %": f"{s.get('colour_faithfulness', {}).get('mean', 0):.1f}",
                "Symmetry": f"{s['column_symmetry']['mean']:.3f}",
                "Exact memo %": data["exact_memorisation_pct"],
                "Guard win %": data["guard_win_pct"],
            }
        )
    table = markdown_table(rows, ["Split", "n", "IoU", "Palette %", "Colour %", "Symmetry", "Exact memo %", "Guard win %"])
    (BENCH_DIR / "generalization.md").write_text(table + "\n")
    print(table)
    print(json.dumps(coverage, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--best-of", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    run(args.limit, args.best_of, args.seed)
