"""Batched best-of-N sampling: one decode pass, same rerank, no extra latency per candidate."""
import os

import pytest

from spritemind.ml.pipeline import SpriteMindPipeline


@pytest.fixture(scope="module")
def pipeline():
    os.environ["SPRITEMIND_RESULT_CACHE"] = "0"
    try:
        p = SpriteMindPipeline()
        if not p.load_checkpoint("svan_checkpoint_best.pt"):
            pytest.skip(f"no usable checkpoint: {p.checkpoint_error}")
        yield p
    finally:
        os.environ.pop("SPRITEMIND_RESULT_CACHE", None)


def _candidates(pipeline, best_of, seed=17):
    return pipeline._sample_batch(pipeline.trained_model, "knight", "classic_16bit", "idle", 24, seed, 0.9, best_of)


def test_batch_sampling_returns_the_requested_number_of_distinct_candidates(pipeline):
    candidates = _candidates(pipeline, 4)
    assert len(candidates) == 4
    assert len({tuple(seq) for seq in candidates}) > 1, "batched rows collapsed onto one sample"


def test_batch_sampling_is_deterministic_for_a_seed(pipeline):
    assert _candidates(pipeline, 4, seed=23) == _candidates(pipeline, 4, seed=23)


def test_best_of_never_scores_worse_than_a_single_candidate(pipeline):
    def best_score(best_of):
        scores = []
        for tokens in _candidates(pipeline, best_of, seed=31):
            from spritemind.pixelscript.ast import PixelScriptProgram

            prog = PixelScriptProgram(actions=pipeline.tokenizer.decode_tokens(tokens))
            scores.append(pipeline._score_candidate(prog, "pico8", 32, 32))
        return max(scores)

    assert best_score(4) >= best_score(1)


def test_trained_generation_accepts_best_of(pipeline):
    one = pipeline.generate(category="wizard", engine_mode="trained_model", seed=5, best_of=1)
    four = pipeline.generate(category="wizard", engine_mode="trained_model", seed=5, best_of=4)
    assert one["engine_used"] == four["engine_used"] == "trained_model"
    assert four["frames"] and one["frames"]
