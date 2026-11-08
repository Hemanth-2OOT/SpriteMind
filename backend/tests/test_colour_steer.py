"""R11 colour steering: the sampler can favour palette entries the prompt asked for.

Steering is an inference-time bias on the `COL_*` logits, so it must never break
grammar validity, determinism, or the palette: it may only change which legal
colour the model prefers.
"""
import os

import pytest

from spritemind.assets.faithfulness import COLOUR_TARGETS, _matches
from spritemind.ml.pipeline import SpriteMindPipeline
from spritemind.ml.tokenizer import ActionTokenizer
from spritemind.utils.palettes import PALETTES


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


def _steer_ids(pipeline, palette, colour):
    return pipeline._colour_steer_ids(palette, colour)


def test_steer_ids_point_at_palette_entries_matching_the_request(pipeline):
    ids = _steer_ids(pipeline, "pico8", "blue")
    assert ids, "pico8 has blue entries; steering must find them"
    tokenizer = pipeline.tokenizer
    for token_id in ids:
        index = int(tokenizer.id_to_token[token_id].split("_")[1])
        assert index > 0
        assert _matches(*PALETTES["pico8"][index][:3], COLOUR_TARGETS["blue"])


def test_steer_ids_are_empty_when_the_palette_cannot_show_the_colour(pipeline):
    assert _steer_ids(pipeline, "gameboy", "blue") == []


def test_steer_ids_are_empty_without_a_request(pipeline):
    assert _steer_ids(pipeline, "pico8", None) == []
    assert _steer_ids(pipeline, "pico8", "not_a_colour") == []


def test_steering_is_deterministic_for_a_seed(pipeline):
    def sample():
        return pipeline._sample_tokens(
            pipeline.trained_model, "knight", "classic_16bit", "idle", 24, 31, 0.9, 0,
            colour="blue", palette="pico8", colour_steer=3.0,
        )

    assert sample() == sample()


def test_strong_steering_only_draws_with_the_requested_colour(pipeline):
    tokens = pipeline._sample_tokens(
        pipeline.trained_model, "knight", "classic_16bit", "idle", 24, 5, 0.9, 0,
        colour="blue", palette="pico8", colour_steer=50.0,
    )
    actions = pipeline.tokenizer.decode_tokens(tokens)
    used = {a.params.get("color") for a in actions if isinstance(a.params.get("color"), int)}
    assert used, "the program should still paint"
    matching = {i for i in used if _matches(*PALETTES["pico8"][i][:3], COLOUR_TARGETS["blue"])}
    assert used <= matching, f"unexpected colours under strong steering: {sorted(used - matching)}"


def test_steering_off_matches_the_plain_sampler(pipeline):
    plain = pipeline._sample_tokens(
        pipeline.trained_model, "knight", "classic_16bit", "idle", 24, 11, 0.9, 0,
        colour="blue", palette="pico8",
    )
    zero = pipeline._sample_tokens(
        pipeline.trained_model, "knight", "classic_16bit", "idle", 24, 11, 0.9, 0,
        colour="blue", palette="pico8", colour_steer=0.0,
    )
    assert plain == zero


def test_colour_token_weights_only_touch_colour_slots():
    from spritemind.ml.training import colour_token_weights

    tokenizer = ActionTokenizer()
    weights = colour_token_weights(tokenizer, 3.0)
    assert float(weights[tokenizer.token_to_id["COL_5"]]) == 3.0
    assert float(weights[tokenizer.token_to_id["NUM_5"]]) == 1.0
    assert float(weights.sum()) == pytest.approx(tokenizer.vocab_size + 2.0 * 32)
