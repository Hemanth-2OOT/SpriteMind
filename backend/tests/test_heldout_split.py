"""Tests for the frozen held-out split spec (R10)."""
import json
from pathlib import Path

import pytest

from spritemind.dataset import splits

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def spec():
    path = ROOT / "data" / "splits" / "heldout_v1.json"
    if not path.exists():
        pytest.skip("held-out spec not generated in this checkout")
    return splits.load_spec(path)


def test_spec_digest_matches_body(spec):
    assert spec["digest"] == splits.spec_digest(spec)
    assert len(spec["heldout_triples"]) == 30
    assert len(spec["heldout_category_element"]) == 10


def test_spec_digest_changes_when_a_triple_changes(spec):
    tampered = json.loads(json.dumps(spec))
    tampered["heldout_triples"][0] = ["slime", "flat", "idle"]
    assert splits.spec_digest(tampered) != spec["digest"]


def test_load_spec_rejects_a_tampered_digest(tmp_path, spec):
    tampered = json.loads(json.dumps(spec))
    tampered["heldout_triples"].pop()
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError):
        splits.load_spec(path)


def test_heldout_reason_flags_triples_and_category_element_pairs(spec):
    triple = spec["heldout_triples"][0]
    record = {"category": triple[0], "style": triple[1], "pose": triple[2], "element": None}
    assert splits.heldout_reason(record, spec) == "triple"

    category, element = spec["heldout_category_element"][0]
    record = {"category": category, "style": "flat", "pose": "idle", "element": element}
    assert splits.heldout_reason(record, spec) == "category_element"


def test_unlisted_combination_is_trainable(spec):
    triples = {tuple(t) for t in spec["heldout_triples"]}
    pairs = {tuple(p) for p in spec["heldout_category_element"]}
    checked = 0
    for category in ("knight", "robot"):
        for style in ("flat", "classic_16bit"):
            for pose in ("idle", "walk"):
                for element in (None, "ice", "fire"):
                    if (category, style, pose) in triples or (category, element) in pairs:
                        continue
                    record = {"category": category, "style": style, "pose": pose, "element": element}
                    assert splits.heldout_reason(record, spec) is None
                    checked += 1
    assert checked > 0


def test_audit_reports_leaked_heldout_records(spec):
    triple = spec["heldout_triples"][0]
    leaked = {"id": "x1", "category": triple[0], "style": triple[1], "pose": triple[2], "split": "train"}
    clean = {"id": "x2", "category": "slime", "style": "flat", "pose": "walk", "split": "train"}
    assert splits.heldout_reason(clean, spec) is None
    result = splits.audit([leaked, clean], spec)
    assert result["leak_count"] == 1
    assert result["leaks"][0]["id"] == "x1"
    assert result["heldout_records"]["triple"] == 1


def test_dataset_metadata_records_the_spec_digest(spec):
    meta_path = ROOT / "data" / "datasets" / "cond_v1_15k" / "metadata.json"
    if not meta_path.exists():
        pytest.skip("conditioned corpus not built in this checkout")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["heldout"]["spec_digest"] == spec["digest"]
    assert meta["splits"].get("heldout", 0) > 0
