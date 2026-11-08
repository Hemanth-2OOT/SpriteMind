"""R4/R5/R6: palette conditioning, colour-aware reranking and critic-aware reranking."""
import os

import pytest

from spritemind.ml.pipeline import (
    CRITIC_RERANK_REFERENCE,
    CRITIC_RERANK_WEIGHT,
    SpriteMindPipeline,
)


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


# --- R6: palette conditioning -------------------------------------------------

def test_palette_index_stays_zero_when_checkpoint_has_no_palette_labels(pipeline):
    assert pipeline.trained_model.palette_conditioning_trained is False
    assert pipeline._palette_index(pipeline.trained_model, "gameboy") == 0


def test_palette_index_uses_the_real_palette_when_trained_with_labels(pipeline):
    model = pipeline.trained_model
    model.palette_conditioning_trained = True
    try:
        assert pipeline._palette_index(model, "pico8") == 0
        assert pipeline._palette_index(model, "gameboy") == 1
        assert pipeline._palette_index(model, "not_a_palette") == 0
    finally:
        model.palette_conditioning_trained = False


def test_palette_index_is_clamped_to_the_embedding(pipeline):
    model = pipeline.trained_model
    model.palette_conditioning_trained = True
    try:
        limit = model.semantic_encoder.palette_embed.num_embeddings
        assert pipeline._palette_index(model, "endesga32") == min(4, limit - 1)
        assert 0 <= pipeline._palette_index(model, "endesga32") < limit
    finally:
        model.palette_conditioning_trained = False


def test_palette_index_reaches_the_conditioning(monkeypatch, pipeline):
    model = pipeline.trained_model
    model.palette_conditioning_trained = True
    seen = {}

    original = pipeline._palette_index
    monkeypatch.setattr(pipeline, "_palette_index", lambda m, p: seen.setdefault("palette", p) and original(m, p))
    try:
        pipeline._sample_tokens(model, "knight", "classic_16bit", "idle", 8, 5, 0.9, 0, palette="gameboy")
        assert seen["palette"] == "gameboy"
    finally:
        model.palette_conditioning_trained = False


# --- R5: colour-aware reranking ------------------------------------------------

def _program(pipeline, colour):
    from spritemind.ml.procedural import ProceduralBaseline

    return ProceduralBaseline.generate(
        category="knight", style="classic_16bit", pose="idle",
        palette_name="pico8", colour=colour, seed=3,
    )


def _colour_share(program, palette="pico8"):
    from spritemind.assets.faithfulness import colour_faithfulness
    from spritemind.renderer.engine import PixelScriptEngine

    res = PixelScriptEngine.execute(program, width=32, height=32, palette_name=palette)
    return res, colour_faithfulness


def test_rerank_prefers_a_candidate_that_draws_the_requested_colour(pipeline):
    from spritemind.assets.faithfulness import colour_faithfulness
    from spritemind.renderer.engine import PixelScriptEngine

    red = _program(pipeline, "red")
    blue = _program(pipeline, "blue")
    res_red = PixelScriptEngine.execute(red, width=32, height=32, palette_name="pico8")
    res_blue = PixelScriptEngine.execute(blue, width=32, height=32, palette_name="pico8")

    red_has_red = colour_faithfulness("red", res_red.composite_array)["score"]
    red_has_blue = colour_faithfulness("blue", res_red.composite_array)["score"]
    assert red_has_red != red_has_blue, "test needs two visibly different colours"

    score_for_red = pipeline._rerank_score(red, "pico8", colour="red", use_critic=False)
    score_for_blue = pipeline._rerank_score(red, "pico8", colour="blue", use_critic=False)
    assert score_for_red > score_for_blue


def test_colour_term_can_be_switched_off(pipeline):
    red = _program(pipeline, "red")
    with_colour = pipeline._rerank_score(red, "pico8", colour="red", use_critic=False)
    without = pipeline._rerank_score(red, "pico8", colour="red", use_critic=False, use_colour=False)
    assert without == pipeline._score_candidate(red, "pico8")


def test_colour_term_does_not_fire_without_a_requested_colour(pipeline):
    red = _program(pipeline, "red")
    assert pipeline._rerank_score(red, "pico8", colour=None, prompt_text="", use_critic=False) == pytest.approx(
        pipeline._score_candidate(red, "pico8")
    )


# --- R4: critic-aware reranking ------------------------------------------------

def test_critic_term_equals_the_reported_quality_score_delta(pipeline):
    prog = _program(pipeline, "red")
    from spritemind.renderer.engine import PixelScriptEngine

    res = PixelScriptEngine.execute(prog, width=32, height=32, palette_name="pico8")
    report = pipeline._run_critic(res, prog, category="knight", palette_name="pico8")
    expected = CRITIC_RERANK_WEIGHT * (float(report["quality_score"]) - CRITIC_RERANK_REFERENCE)

    without = pipeline._rerank_score(prog, "pico8", use_critic=False)
    with_critic = pipeline._rerank_score(prog, "pico8", use_critic=True)
    assert with_critic - without == pytest.approx(expected, abs=1e-6)


def test_generate_reports_which_rerank_terms_were_used(pipeline):
    out = pipeline.generate(
        prompt="a blue knight standing ready", category="knight", pose="idle",
        engine_mode="trained_model", seed=11, best_of=2,
    )
    assert out["rerank"]["critic_used"] is True
    assert out["rerank"]["critic_weight"] == CRITIC_RERANK_WEIGHT

    plain = pipeline.generate(
        prompt="a blue knight standing ready", category="knight", pose="idle",
        engine_mode="trained_model", seed=11, best_of=2, use_critic=False,
    )
    assert plain["rerank"]["critic_used"] is False
    assert plain["cache_hit"] is False, "use_critic must be part of the cache key"


def test_rerank_decision_follows_the_critic(monkeypatch, pipeline):
    """The winning candidate must be the one the critic likes best, not a fixed seed."""
    preferences = {}

    def fake_rank(prog, palette, width=32, height=32, **kwargs):
        return float(preferences.get(tuple(a.action_type.value for a in prog.actions[:1]), 0.0))

    monkeypatch.setattr(pipeline, "_rerank_score", fake_rank)
    candidates = pipeline._sample_batch(
        pipeline.trained_model, "knight", "classic_16bit", "idle", 16, 29, 0.9, 3,
    )
    decoded = [pipeline.tokenizer.decode_tokens(tokens) for tokens in candidates]
    for index, actions in enumerate(decoded):
        key = tuple(a.action_type.value for a in actions[:1])
        preferences[key] = float(index) + 1.0

    prog, source = pipeline._neural_generate(
        pipeline.trained_model, "knight", "classic_16bit", "idle", "pico8",
        max_actions=16, seed=29, best_of=3, procedural_guard=False,
    )
    assert source == "neural"
    best_key = max(preferences, key=preferences.get)
    assert tuple(a.action_type.value for a in prog.actions[:1]) == best_key


# --- Palette safety guard ------------------------------------------------------

def test_palette_leak_penalises_a_candidate_that_draws_off_palette(pipeline):
    import numpy as np

    leaky = np.zeros((4, 4, 4), dtype=np.uint8)
    leaky[0, 0] = [255, 0, 255, 255]
    clean_rgba = np.zeros((4, 4, 4), dtype=np.uint8)

    assert pipeline._palette_leak_share(clean_rgba, "pico8") == 0.0
    assert pipeline._palette_leak_share(leaky, "pico8") == 1.0
    assert pipeline._palette_leak_share(leaky, "unknown_palette") is None
