import json

import pytest
import torch

from spritemind.ml.pipeline import SpriteMindPipeline


def test_procedural_engine_reports_its_source():
    pipeline = SpriteMindPipeline()
    result = pipeline.generate(category="knight", engine_mode="procedural_baseline", seed=3)
    assert result["engine_used"] == "procedural_baseline"
    assert result["candidate_source"] == "procedural_baseline"


def test_incompatible_checkpoint_is_refused_instead_of_crashing(tmp_path):
    pipeline = SpriteMindPipeline(checkpoints_dir=str(tmp_path))
    checkpoint = {"metadata": {"model_config": {"vocab_size": 17, "max_seq_len": 448}}, "model_state_dict": {}}
    torch.save(checkpoint, tmp_path / "stale.pt")
    assert pipeline.load_checkpoint("stale.pt") is False
    assert "vocab 17" in (pipeline.checkpoint_error or "")


def test_trained_generation_uses_a_tracked_candidate_source():
    pipeline = SpriteMindPipeline()
    if not pipeline.load_checkpoint("svan_checkpoint_best.pt"):
        pytest.skip(f"no usable checkpoint: {pipeline.checkpoint_error}")

    result = pipeline.generate(category="wizard", engine_mode="trained_model", seed=11)
    assert result["engine_used"] == "trained_model"
    assert result["candidate_source"] in {"neural", "procedural_guard"}
    json.dumps(result)  # the API response must stay JSON-serialisable
