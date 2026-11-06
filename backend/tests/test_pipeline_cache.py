"""Tests for incremental grammar decoding and the generation result cache."""
import random

import pytest

from spritemind.ml.pipeline import SpriteMindPipeline
from spritemind.ml.tokenizer import ActionTokenizer


def _random_token_sequence(tokenizer: ActionTokenizer, length: int, seed: int):
    rng = random.Random(seed)
    ids = [tokenizer.sos_id]
    for _ in range(length):
        choices = tokenizer.allowed_token_ids(ids) or [tokenizer.eos_id]
        ids.append(rng.choice(choices))
    return ids


@pytest.mark.parametrize("seed", [1, 7, 13])
def test_incremental_grammar_matches_full_recompute(seed):
    tokenizer = ActionTokenizer()
    tokens = _random_token_sequence(tokenizer, 60, seed)

    state = tokenizer.grammar_start_state()
    for index, token_id in enumerate(tokens):
        prefix = tokens[:index]
        assert tokenizer.allowed_token_ids(prefix) == tokenizer.allowed_from_state(state)
        state = tokenizer.advance_grammar(state, token_id)

    assert tokenizer._grammar_state(tokens) == state[:2]


def test_allowed_state_cache_returns_same_list():
    tokenizer = ActionTokenizer()
    state = tokenizer.grammar_start_state()
    first = tokenizer.allowed_from_state(state)
    assert tokenizer.allowed_from_state(state) is first


@pytest.fixture()
def pipeline():
    return SpriteMindPipeline()


def test_repeat_request_is_served_from_cache(pipeline):
    kwargs = dict(prompt="a knight with a sword", engine_mode="procedural_baseline", seed=5)

    first = pipeline.generate(**kwargs)
    second = pipeline.generate(**kwargs)

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["frames"][0]["script_text"] == first["frames"][0]["script_text"]
    assert pipeline.cache_stats()["hits"] == 1


def test_different_seed_is_not_a_cache_hit(pipeline):
    first = pipeline.generate(prompt="a mage", engine_mode="procedural_baseline", seed=1)
    second = pipeline.generate(prompt="a mage", engine_mode="procedural_baseline", seed=2)

    assert first["cache_hit"] is False
    assert second["cache_hit"] is False
    assert pipeline.cache_stats()["misses"] == 2


def test_cached_payload_is_a_copy(pipeline):
    kwargs = dict(prompt="a rogue", engine_mode="procedural_baseline", seed=3)

    first = pipeline.generate(**kwargs)
    first["frames"][0]["script_text"] = "mutated"
    second = pipeline.generate(**kwargs)

    assert second["frames"][0]["script_text"] != "mutated"


def test_cache_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SPRITEMIND_RESULT_CACHE", "0")
    pipeline = SpriteMindPipeline()
    kwargs = dict(prompt="a goblin", engine_mode="procedural_baseline", seed=9)

    first = pipeline.generate(**kwargs)
    second = pipeline.generate(**kwargs)

    assert first["cache_hit"] is False
    assert second["cache_hit"] is False
    assert pipeline.cache_stats()["size"] == 0


def test_cache_evicts_oldest_entry(monkeypatch):
    monkeypatch.setenv("SPRITEMIND_RESULT_CACHE", "2")
    pipeline = SpriteMindPipeline()
    for seed in range(3):
        pipeline.generate(prompt="a paladin", engine_mode="procedural_baseline", seed=seed)

    assert pipeline.cache_stats()["size"] == 2
    assert pipeline.generate(prompt="a paladin", engine_mode="procedural_baseline", seed=0)["cache_hit"] is False
