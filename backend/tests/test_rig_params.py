import pytest

from spritemind.ml import rig_params


def test_style_tags_read_the_prompt():
    assert rig_params.tags_from_prompt("a heavy armored knight")["heavy"] == 1.0
    assert rig_params.tags_from_prompt("an agile rogue")["nimble"] == 1.0
    assert rig_params.tags_from_prompt("a wizard casting fireballs")["casting"] == 1.0
    assert rig_params.tags_from_prompt("a small slime")["heavy"] == 0.0


def test_modulate_stays_inside_the_rig_limits():
    tags = {t: 1.0 for t in rig_params.STYLE_TAGS}
    offsets = rig_params.modulate(rig_params.base_offsets("run", 0.3, "knight"), tags)
    for dx, dy, rot in offsets.values():
        assert abs(dx) <= rig_params.MAX_DX
        assert abs(dy) <= rig_params.MAX_DY
        assert abs(rot) <= rig_params.MAX_ROT


def test_modulate_actually_reacts_to_the_prompt():
    base = rig_params.base_offsets("walk", 0.25, "knight")
    heavy = rig_params.modulate(base, rig_params.tags_from_prompt("a heavy brute"))
    plain = rig_params.modulate(base, rig_params.tags_from_prompt("a soldier"))
    assert heavy != plain


def test_feature_and_target_vectors_line_up():
    tags = rig_params.tags_from_prompt("a floating ghost")
    assert len(rig_params.feature_vector("walk", 0.5, "knight", tags)) == rig_params.FEATURE_DIM
    assert len(rig_params.target_vector("walk", 0.5, "knight", tags)) == rig_params.OUTPUT_DIM


@pytest.fixture(scope="module")
def head():
    model, metrics = rig_params.train(steps=300, samples=2048, seed=1)
    return model, metrics


def test_the_head_learns_the_authored_targets(head):
    _, metrics = head
    assert metrics["held_out_mse"] < 3.0


def test_predictions_are_bounded_and_complete(head):
    model, _ = head
    offsets = rig_params.predict(model, "walk", 0.4, "knight", "a heavy knight")
    assert set(offsets) == set(rig_params.PARTS)
    for dx, dy, rot in offsets.values():
        assert abs(dx) <= rig_params.MAX_DX
        assert abs(dy) <= rig_params.MAX_DY
        assert abs(rot) <= rig_params.MAX_ROT


def test_predict_is_deterministic(head):
    model, _ = head
    first = rig_params.predict(model, "run", 0.6, "rogue", "an agile rogue")
    second = rig_params.predict(model, "run", 0.6, "rogue", "an agile rogue")
    assert first == second


def test_the_prompt_changes_the_predicted_motion(head):
    model, _ = head
    a = rig_params.predict(model, "walk", 0.25, "knight", "a heavy armored knight")
    b = rig_params.predict(model, "walk", 0.25, "knight", "a floating ghost")
    assert rig_params.offset_diversity([a, b]) > 0.0


def test_the_fixed_template_has_no_prompt_diversity():
    same = [rig_params.base_offsets("walk", 0.25, "knight") for _ in range(4)]
    assert rig_params.offset_diversity(same) == 0.0


def test_offsets_for_matches_predict_and_keeps_every_part():
    from spritemind.ml.rig_params import get_head, predict

    expected = predict(get_head(), "walk", 0.3, "knight", "a heavy knight")
    assert rig_params.offsets_for("walk", 0.3, "knight", "a heavy knight") == expected


def test_the_learned_flag_changes_the_animation():
    from spritemind.ml.temporal import TemporalAnimator

    common = dict(
        base_script_or_prompt="a lanky skeleton warrior",
        category="skeleton",
        anim_category="walk",
        width=64,
        height=64,
        num_frames=4,
        engine_mode="procedural_baseline",
        seed=5,
    )
    plain = TemporalAnimator.generate_animation(**common)
    learned = TemporalAnimator.generate_animation(learned_rig=True, prompt="a lanky skeleton warrior", **common)
    assert [f["script_text"] for f in plain["frames"]] != [f["script_text"] for f in learned["frames"]]
    assert len(learned["frames"]) == 4
