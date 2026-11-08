"""Tests for the pixel-grounded metric module (`spritemind.ml.metrics`)."""
import numpy as np
import pytest

from spritemind.ml import metrics
from spritemind.utils.palettes import PALETTES


def make_sprite(fill=0.5, colour=(255, 0, 0, 255)):
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    size = int(32 * fill)
    rgba[8:8 + size, 10:10 + size] = colour
    return rgba


def test_alpha_mask_matches_alpha_channel():
    rgba = make_sprite()
    assert metrics.alpha_mask(rgba).sum() == (rgba[:, :, 3] > 0).sum()


def test_silhouette_iou_identical_masks_is_one():
    mask = metrics.alpha_mask(make_sprite())
    assert metrics.silhouette_iou(mask, mask) == 1.0


def test_silhouette_iou_disjoint_masks_is_zero():
    a = np.zeros((32, 32), dtype=bool)
    b = np.zeros((32, 32), dtype=bool)
    a[0:4, 0:4] = True
    b[10:14, 10:14] = True
    assert metrics.silhouette_iou(a, b) == 0.0


def test_silhouette_iou_half_overlap():
    a = np.zeros((32, 32), dtype=bool)
    b = np.zeros((32, 32), dtype=bool)
    a[0:4, 0:4] = True
    a[4:8, 0:4] = True
    b[0:4, 0:4] = True
    assert metrics.silhouette_iou(a, b) == pytest.approx(0.5)


def test_silhouette_iou_empty_union_is_one():
    empty = np.zeros((32, 32), dtype=bool)
    assert metrics.silhouette_iou(empty, empty) == 1.0


def test_silhouette_iou_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        metrics.silhouette_iou(np.zeros((32, 32), bool), np.zeros((16, 16), bool))


def test_palette_adherence_counts_only_palette_members():
    member = PALETTES["pico8"][1][:3]
    assert metrics.palette_adherence(make_sprite(colour=(*member, 255)), "pico8") == 100.0
    assert metrics.palette_adherence(make_sprite(colour=(1, 2, 3, 255)), "pico8") == 0.0


def test_palette_adherence_is_nan_for_unknown_palette():
    assert np.isnan(metrics.palette_adherence(make_sprite(), "no_such_palette"))


def test_palette_adherence_ignores_transparent_pixels():
    rgba = make_sprite(colour=(1, 2, 3, 255))
    rgba[0:10, 0:10] = (9, 9, 9, 0)
    assert metrics.palette_adherence(rgba, "pico8") == 0.0


def test_connected_components_counts_separate_blobs():
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[2:6, 2:6] = (255, 0, 0, 255)
    rgba[20:24, 20:24] = (255, 0, 0, 255)
    assert metrics.connected_component_count(metrics.alpha_mask(rgba)) == 2


def test_largest_component_share_of_single_blob_is_one():
    assert metrics.largest_component_share(metrics.alpha_mask(make_sprite())) == 1.0


def test_column_symmetry_of_symmetric_sprite_is_one():
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[10:20, 8:16] = (255, 0, 0, 255)
    rgba[10:20, 16:24] = (255, 0, 0, 255)
    assert metrics.column_symmetry(rgba) == pytest.approx(1.0, abs=0.02)


def test_interior_hole_pixels_detects_enclosed_gap():
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[6:20, 6:20] = (255, 0, 0, 255)
    rgba[10:14, 10:14] = (0, 0, 0, 0)
    assert metrics.interior_holes(metrics.alpha_mask(rgba)) > 0


def test_evaluate_sprite_returns_grounded_fields():
    rgba = make_sprite(colour=(255, 0, 0, 255))
    row = metrics.evaluate_sprite(rgba, "pico8", reference_mask=metrics.alpha_mask(rgba), prompt="a red knight")
    assert row["palette_adherence_pct"] == 0.0
    assert row["silhouette_iou"] == 1.0
    assert row["colour_faithfulness"] > 50.0
    assert row["colour_requested"] == ["red"]


def test_evaluate_sprite_without_reference_or_prompt_omits_them():
    row = metrics.evaluate_sprite(make_sprite(), "pico8")
    assert "silhouette_iou" not in row
    assert "colour_faithfulness" not in row


def test_summarise_reports_mean_and_stderr():
    summary = metrics.summarise([{"x": 1.0}, {"x": 3.0}])
    assert summary["x"]["mean"] == 2.0
    assert summary["x"]["stderr"] == pytest.approx(1.0)
    assert summary["n"] == 2


def test_summarise_skips_non_numeric_values():
    summary = metrics.summarise([{"x": 1.0, "label": "a"}, {"x": 2.0, "label": "b"}])
    assert "label" not in summary
