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


def _best_score(pipeline, best_of, seed):
    from spritemind.pixelscript.ast import PixelScriptProgram

    scores = [
        pipeline._score_candidate(PixelScriptProgram(actions=pipeline.tokenizer.decode_tokens(tokens)), "pico8", 32, 32)
        for tokens in _candidates(pipeline, best_of, seed=seed)
    ]
    return max(scores)


def test_best_of_raises_the_score_a_single_candidate_reaches(pipeline):
    # Batched decoding is not bit-identical to a batch of one, so compare means over seeds.
    seeds = (31, 32, 33, 34, 35)
    mean_four = sum(_best_score(pipeline, 4, s) for s in seeds) / len(seeds)
    mean_one = sum(_best_score(pipeline, 1, s) for s in seeds) / len(seeds)
    assert mean_four > mean_one


def test_trained_generation_accepts_best_of(pipeline):
    one = pipeline.generate(category="wizard", engine_mode="trained_model", seed=5, best_of=1)
    four = pipeline.generate(category="wizard", engine_mode="trained_model", seed=5, best_of=4)
    assert one["engine_used"] == four["engine_used"] == "trained_model"
    assert four["frames"] and one["frames"]
