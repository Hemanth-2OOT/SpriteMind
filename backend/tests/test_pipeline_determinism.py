"""Seed determinism: the same request must replay to the same pixels."""
import os

import pytest

from spritemind.ml.pipeline import SpriteMindPipeline

PROMPT = "a blue wizard with a tall hat"
SEEDS = [1, 7, 42, 99, 2024]


def frames_of(result):
    return [frame["image_base64"] for frame in result["frames"]]


@pytest.fixture(scope="module")
def pipeline():
    # Cache off: this fixture is about the sampler, not the replay path.
    os.environ["SPRITEMIND_RESULT_CACHE"] = "0"
    try:
        yield SpriteMindPipeline()
    finally:
        os.environ.pop("SPRITEMIND_RESULT_CACHE", None)


def test_procedural_generation_is_seed_deterministic(pipeline):
    for seed in SEEDS:
        first = pipeline.generate(prompt=PROMPT, engine_mode="procedural_baseline", seed=seed)
        second = pipeline.generate(prompt=PROMPT, engine_mode="procedural_baseline", seed=seed)
        assert frames_of(first) == frames_of(second), f"procedural drifted at seed {seed}"


def test_trained_generation_is_seed_deterministic(pipeline):
    if not pipeline.load_checkpoint("svan_checkpoint_best.pt"):
        pytest.skip(f"no usable checkpoint: {pipeline.checkpoint_error}")

    for seed in SEEDS:
        first = pipeline.generate(prompt=PROMPT, engine_mode="trained_model", seed=seed)
        second = pipeline.generate(prompt=PROMPT, engine_mode="trained_model", seed=seed)
        assert first["engine_used"] == "trained_model"
        assert frames_of(first) == frames_of(second), f"trained model drifted at seed {seed}"


def test_cache_replay_matches_a_cold_generation():
    os.environ.pop("SPRITEMIND_RESULT_CACHE", None)
    cold = SpriteMindPipeline()
    warm = SpriteMindPipeline()
    first = cold.generate(prompt=PROMPT, engine_mode="procedural_baseline", seed=5)
    assert warm.generate(prompt=PROMPT, engine_mode="procedural_baseline", seed=5)["cache_hit"] is False
    replay = warm.generate(prompt=PROMPT, engine_mode="procedural_baseline", seed=5)
    assert replay["cache_hit"] is True
    assert frames_of(replay) == frames_of(first)


def test_different_seeds_produce_different_sprites(pipeline):
    renders = {
        tuple(frames_of(pipeline.generate(prompt=PROMPT, engine_mode="procedural_baseline", seed=seed)))
        for seed in SEEDS
    }
    assert len(renders) > 1, "seeds are not reaching the sampler"


def test_cache_key_covers_every_output_affecting_field():
    os.environ.pop("SPRITEMIND_RESULT_CACHE", None)
    pipeline = SpriteMindPipeline()
    base = pipeline.generate(prompt=PROMPT, engine_mode="trained_model", seed=5, best_of=1)
    variant = pipeline.generate(prompt=PROMPT, engine_mode="trained_model", seed=5, best_of=3)
    assert variant["cache_hit"] is False, "best_of must be part of the cache key"

    hot = pipeline.generate(prompt=PROMPT, engine_mode="trained_model", seed=5, best_of=1, temperature=0.3)
    assert hot["cache_hit"] is False, "temperature must be part of the cache key"

    narrow = pipeline.generate(prompt=PROMPT, engine_mode="trained_model", seed=5, best_of=1, top_k=10)
    assert narrow["cache_hit"] is False, "top_k must be part of the cache key"

    pose = pipeline.generate(prompt=PROMPT, engine_mode="trained_model", seed=5, pose="walk")
    assert pose["cache_hit"] is False, "pose must be part of the cache key"
    assert base["pose"] != pose["pose"]


def test_seed_is_echoed_for_persistence(pipeline):
    result = pipeline.generate(prompt=PROMPT, engine_mode="procedural_baseline", seed=31337)
    assert result["seed"] == 31337
