"""Palette labels must reach the model (R6 plumbing)."""
import json
from pathlib import Path

from spritemind.dataset.conditioning import PALETTE_IDS, palette_id
from spritemind.dataset.generator import CharacterGenerator
from spritemind.ml.tokenizer import ActionTokenizer
from spritemind.ml.training import PixelScriptDataset
from spritemind.pixelscript.serializer import PixelScriptSerializer


def test_palette_id_follows_palettes_order():
    assert PALETTE_IDS["pico8"] == 0
    assert PALETTE_IDS["gameboy"] == 1
    assert palette_id("endesga32") == PALETTE_IDS["endesga32"]
    assert palette_id(None) == 0
    assert palette_id("not_a_palette") == 0


def _record(index, palette, split="train"):
    prog = CharacterGenerator.generate_character(category="knight", seed=100 + index, palette_name=palette)
    return {
        "id": f"t{index}",
        "split": split,
        "category": "knight",
        "style": "flat",
        "pose": "idle",
        "palette": palette,
        "prompt": "a knight",
        "actions": [a.to_dict() for a in prog.actions],
        "script_text": PixelScriptSerializer.serialize_program(prog),
    }


def test_dataset_reads_the_palette_field(tmp_path):
    dataset_dir = tmp_path / "corpus"
    dataset_dir.mkdir()
    with open(dataset_dir / "records.jsonl", "w", encoding="utf-8") as handle:
        handle.write(json.dumps(_record(0, "pico8")) + "\n")
        handle.write(json.dumps(_record(1, "gameboy")) + "\n")

    dataset = PixelScriptDataset(dataset_dir, ActionTokenizer(), split="train")
    assert len(dataset) == 2
    assert dataset.palette_idx.tolist() == [PALETTE_IDS["pico8"], PALETTE_IDS["gameboy"]]


def test_legacy_records_without_a_palette_default_to_zero(tmp_path):
    dataset_dir = tmp_path / "legacy"
    dataset_dir.mkdir()
    record = _record(0, "pico8")
    record.pop("palette")
    with open(dataset_dir / "records.jsonl", "w", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    dataset = PixelScriptDataset(dataset_dir, ActionTokenizer(), split="train")
    assert dataset.palette_idx.tolist() == [0]
