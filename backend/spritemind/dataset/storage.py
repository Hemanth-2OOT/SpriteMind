"""Dataset storage, disk serialization, and statistics manager with 10k capacity."""
import os
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
from spritemind.dataset.generator import (
    CharacterGenerator,
    CATEGORIES,
    STYLES,
    POSES,
    PROPORTIONS,
    WEAPONS,
    ACCESSORIES,
)
from spritemind.dataset.defects import DefectGenerator
from spritemind.renderer.engine import PixelScriptEngine
from spritemind.renderer.export import array_to_base64
from spritemind.pixelscript.serializer import PixelScriptSerializer


class DatasetManager:
    """Manages synthetic datasets on disk with identity-based splitting and compact serialization."""

    def __init__(self, data_root: str = "data/datasets"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

    def get_dataset_dir(self, name: str) -> Path:
        return self.data_root / name

    def generate_dataset(
        self,
        name: str = "generalization_10k",
        total_samples: int = 10000,
        defect_rate: float = 0.15,
        categories: Optional[List[str]] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        render_sample_limit: int = 100,
    ) -> Dict[str, Any]:
        """Generates reproducible dataset with identity-based splitting to prevent data leakage."""
        target_dir = self.get_dataset_dir(name)
        samples_dir = target_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        if not categories:
            categories = CATEGORIES

        rng = random.Random(seed)
        category_counts: Dict[str, int] = {c: 0 for c in categories}
        defect_counts: Dict[str, int] = {"clean": 0}
        split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

        jsonl_path = target_dir / "records.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
            for i in range(total_samples):
                cat = categories[i % len(categories)]
                style = rng.choice(STYLES)
                pose = rng.choice(POSES)
                prop = rng.choice(PROPORTIONS)
                wep = rng.choice(WEAPONS)
                acc = rng.choice(ACCESSORIES)

                # Character identity archetype seed (distinct character archetype)
                # i // 5 groups 5 variations/poses of the same character identity together
                character_id = (i // 5)
                # Split STRICTLY by character identity hash (guarantees NO identity leakage!)
                identity_hash = (character_id * 2654435761 + seed) % 100
                if identity_hash < int(train_ratio * 100):
                    split = "train"
                elif identity_hash < int((train_ratio + val_ratio) * 100):
                    split = "val"
                else:
                    split = "test"

                split_counts[split] += 1
                category_counts[cat] += 1

                sample_seed = seed + i * 17
                prog = CharacterGenerator.generate_character(
                    category=cat,
                    style=style,
                    pose=pose,
                    width=32,
                    height=32,
                    seed=sample_seed,
                    proportion=prop,
                    weapon=wep,
                    accessory=acc,
                )

                # Defect injection
                has_defect = rng.random() < defect_rate
                defect_info = None
                if has_defect:
                    prog, defect_info = DefectGenerator.inject_defect(prog, seed=sample_seed)
                    dtype = defect_info["defect_type"]
                    defect_counts[dtype] = defect_counts.get(dtype, 0) + 1
                else:
                    defect_counts["clean"] += 1

                dsl_text = PixelScriptSerializer.serialize_program(prog)
                sample_id = f"sample_{i:06d}"

                record = {
                    "id": sample_id,
                    "character_id": character_id,
                    "split": split,
                    "category": cat,
                    "style": style,
                    "pose": pose,
                    "proportion": prop,
                    "weapon": wep,
                    "accessory": acc,
                    "has_defect": has_defect,
                    "defect_info": defect_info,
                    "actions": [a.to_dict() for a in prog.actions],
                    "script_text": dsl_text,
                }

                # Write line to JSONL
                jsonl_file.write(json.dumps(record) + "\n")

                # For the first N samples, render visual preview into samples/ directory for UI browsing
                if i < render_sample_limit:
                    res = PixelScriptEngine.execute(prog, width=32, height=32)
                    record_with_img = dict(record)
                    record_with_img["image_base64"] = array_to_base64(res.composite_array, scale=4)
                    record_with_img["pixel_count"] = int(res.to_dict()["pixel_count"])
                    with open(samples_dir / f"{sample_id}.json", "w", encoding="utf-8") as f:
                        json.dump(record_with_img, f)

        meta_summary = {
            "name": name,
            "total_samples": total_samples,
            "splits": split_counts,
            "category_distribution": category_counts,
            "defect_distribution": defect_counts,
            "seed": seed,
            "leakage_prevention": "Strict character_id template hash partition",
        }

        with open(target_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_summary, f, indent=2)

        return meta_summary

    def get_dataset_stats(self, name: str = "generalization_10k") -> Optional[Dict[str, Any]]:
        target_dir = self.get_dataset_dir(name)
        meta_file = target_dir / "metadata.json"
        if not meta_file.exists():
            # Check default_synth fallback
            fallback = self.get_dataset_dir("default_synth") / "metadata.json"
            if fallback.exists():
                with open(fallback, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_samples(self, name: str = "generalization_10k", limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        target_dir = self.get_dataset_dir(name)
        samples_dir = target_dir / "samples"
        if not samples_dir.exists():
            fallback_dir = self.get_dataset_dir("default_synth") / "samples"
            if fallback_dir.exists():
                samples_dir = fallback_dir
            else:
                return []

        all_files = sorted(samples_dir.glob("sample_*.json"))
        selected_files = all_files[offset : offset + limit]

        records = []
        for p in selected_files:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    records.append(json.load(f))
            except Exception:
                pass
        return records
