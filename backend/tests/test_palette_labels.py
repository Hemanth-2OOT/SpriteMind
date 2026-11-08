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


# --- Regression: the legacy sample_*.json path (no records.jsonl) --------------
# CI has no records.jsonl because the corpus is gitignored, so this folder path is
# the only loading route there. It used to append input_ids before raising on a
# stale variable, leaving palette_idx shorter than input_ids and crashing __getitem__.


def _sample_dir(tmp_path, records, with_jsonl=False):
    dataset_dir = tmp_path / "corpus"
    samples = dataset_dir / "samples"
    samples.mkdir(parents=True)
    for i, record in enumerate(records):
        with open(samples / f"sample_{i:06d}.json", "w", encoding="utf-8") as handle:
            json.dump(record, handle)
    if with_jsonl:
        with open(dataset_dir / "records.jsonl", "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    return dataset_dir


def test_legacy_sample_folder_without_palette_defaults_to_zero(tmp_path):
    record = _record(0, "pico8")
    record.pop("palette")
    dataset = PixelScriptDataset(_sample_dir(tmp_path, [record]), ActionTokenizer(), split="train")

    assert len(dataset) == 1
    assert dataset.palette_idx.tolist() == [0]
    assert dataset[0]["palette_idx"].item() == 0


def test_sample_folder_keeps_real_palette_labels(tmp_path):
    dataset = PixelScriptDataset(
        _sample_dir(tmp_path, [_record(0, "pico8"), _record(1, "gameboy")]),
        ActionTokenizer(),
        split="train",
    )

    assert len(dataset) == 2
    assert dataset.palette_idx.tolist() == [PALETTE_IDS["pico8"], PALETTE_IDS["gameboy"]]


def test_empty_dataset_has_zero_length_and_consistent_tensors(tmp_path):
    dataset = PixelScriptDataset(_sample_dir(tmp_path, []), ActionTokenizer(), split="train")

    assert len(dataset) == 0
    for name in (
        "category_idx",
        "style_idx",
        "pose_idx",
        "palette_idx",
        "colour_idx",
        "element_idx",
    ):
        assert getattr(dataset, name).shape == (0,)
    assert dataset.input_ids.shape[0] == 0


def test_every_sample_has_one_conditioning_entry_per_input(tmp_path):
    with_palette = _record(0, "endesga32")
    without_palette = _record(1, "pico8")
    without_palette.pop("palette")
    dataset = PixelScriptDataset(
        _sample_dir(tmp_path, [with_palette, without_palette]), ActionTokenizer(), split="train"
    )

    assert len(dataset) == 2
    n = dataset.input_ids.shape[0]
    names = (
        "target_ids",
        "category_idx",
        "style_idx",
        "pose_idx",
        "palette_idx",
        "colour_idx",
        "element_idx",
        "prompt_tokens",
    )
    for name in names:
        assert getattr(dataset, name).shape[0] == n, name
    assert dataset.palette_idx.tolist() == [PALETTE_IDS["endesga32"], 0]
    for i in range(n):
        assert dataset[i]["palette_idx"].item() == dataset.palette_idx[i].item()
