import numpy as np

from spritemind.assets.faithfulness import colour_faithfulness, requested_colours


def solid(rgba_colour):
    image = np.zeros((8, 8, 4), dtype=np.uint8)
    image[..., 3] = 255
    image[..., :3] = rgba_colour
    return image


def test_requested_colours_keeps_order_without_duplicates():
    assert requested_colours("A red and gold knight with a red cape") == ["red", "gold"]
    assert requested_colours("A knight with a sword") == []


def test_matching_colour_scores_full_marks():
    report = colour_faithfulness("a blue knight", solid((41, 173, 255)))
    assert report["score"] == 100.0
    assert report["missing"] == []


def test_wrong_colour_scores_zero_and_reports_the_gap():
    report = colour_faithfulness("a blue knight", solid((255, 0, 77)))
    assert report["score"] == 0.0
    assert report["missing"] == ["blue"]


def test_prompt_without_colour_returns_none():
    assert colour_faithfulness("a knight with a sword", solid((255, 0, 77))) is None


def test_partial_coverage_is_weighted_by_pixels():
    image = np.zeros((8, 8, 4), dtype=np.uint8)
    image[..., 3] = 255
    image[:, :4, :3] = (41, 173, 255)
    image[:, 4:, :3] = (95, 87, 79)
    report = colour_faithfulness("a blue knight", image)
    assert 45.0 <= report["score"] <= 55.0
    assert report["per_colour"]["blue"] == 0.5


def test_transparent_pixels_are_excluded():
    image = np.zeros((8, 8, 4), dtype=np.uint8)
    image[:4, :, :3] = (41, 173, 255)
    image[:4, :, 3] = 255
    report = colour_faithfulness("a blue knight", image)
    assert report["score"] == 100.0


def test_selection_hit_is_none_when_no_colour_was_requested():
    from spritemind.assets.faithfulness import selection_hit

    assert selection_hit(None) is None
    assert selection_hit({}) is None
    assert selection_hit({"matched_share": 0.0}) == 0
    assert selection_hit({"matched_share": 0.5}) == 1
