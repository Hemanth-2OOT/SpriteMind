"""The pixel clean-up pass: stray pixels, notches and outline weight."""
import numpy as np
import pytest

from spritemind.renderer.polish import close_outline, drop_stray_pixels, fill_notches, polish

PALETTE = [(0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81), (171, 82, 54)]


def _blank(size: int = 16) -> np.ndarray:
    return np.zeros((size, size, 4), dtype=np.uint8)


def _body(size: int = 16, colour=PALETTE[3]) -> np.ndarray:
    comp = _blank(size)
    comp[4:12, 4:12, :3] = colour
    comp[4:12, 4:12, 3] = 255
    return comp


def _opaque(comp: np.ndarray) -> int:
    return int((comp[:, :, 3] > 0).sum())


def test_stray_pixels_are_erased():
    comp = _body()
    comp[1, 1, :3] = PALETTE[2]
    comp[1, 1, 3] = 255
    out = drop_stray_pixels(comp)
    assert out[1, 1, 3] == 0
    assert _opaque(out) == _opaque(comp) - 1


def test_a_notch_in_the_body_is_filled():
    comp = _body()
    comp[7, 7, 3] = 0
    out = fill_notches(comp, PALETTE)
    assert out[7, 7, 3] == 255
    assert tuple(out[7, 7, :3]) in PALETTE


def test_a_gap_between_limbs_stays_open():
    comp = _body()
    comp[:, 7, 3] = 0
    comp[:, 8, 3] = 0
    comp[0:4, 7:9, 3] = 255
    comp[12:16, 7:9, 3] = 255
    comp[:, 7:9, :3] = PALETTE[0]
    height, width = comp.shape[:2]
    for y in range(height):
        for x in (7, 8):
            if y < 4 or y >= 12:
                comp[y, x, 3] = 255
    out = fill_notches(comp, PALETTE)
    assert out[8, 7, 3] == 0 and out[8, 8, 3] == 0


def test_the_outline_is_one_pixel_and_uses_the_darkest_colour():
    comp = _body()
    before = _opaque(comp)
    out = close_outline(comp, PALETTE)
    edge = out[3, 7]
    assert edge[3] == 255 and tuple(edge[:3]) == PALETTE[0]
    assert _opaque(out) > before


def test_polish_keeps_the_body_and_adds_no_new_colours():
    comp = _body()
    out = polish(comp, PALETTE)
    original = {tuple(c) for c in comp[comp[:, :, 3] > 0][:, :3].tolist()}
    result = {tuple(c) for c in out[out[:, :, 3] > 0][:, :3].tolist()}
    assert result <= set(PALETTE) | original
    assert out[6, 6, 3] == 255


def test_polish_is_idempotent():
    comp = _body()
    comp[1, 1, :3] = PALETTE[1]
    comp[1, 1, 3] = 255
    once = polish(comp, PALETTE)
    twice = polish(once, PALETTE)
    assert np.array_equal(once, twice)


def test_polish_rejects_a_non_rgba_array():
    with pytest.raises(ValueError):
        polish(np.zeros((8, 8), dtype=np.uint8))
