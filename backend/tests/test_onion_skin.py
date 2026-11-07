"""Onion-skin export: current frame over faint ghosts of its neighbours."""
import numpy as np
from PIL import Image

from spritemind.renderer.export import compose_onion_skin


def _frame(colour, x):
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[2:6, x : x + 2] = (*colour, 255)
    return rgba


def _alpha_sum(image: Image.Image) -> int:
    return int(np.array(image)[..., 3].sum())


def test_current_frame_is_fully_opaque_and_ghosts_are_faint():
    frames = [_frame((255, 0, 0), 1), _frame((0, 255, 0), 3), _frame((0, 0, 255), 5)]
    image = compose_onion_skin(frames, 1, previous_alpha=0.3, next_alpha=0.2, scale=1)
    pixels = np.array(image)
    assert pixels.shape == (8, 8, 4)
    assert pixels[2:6, 3:5, 3].min() == 255
    previous_ghost_alpha = pixels[2:6, 1:3, 3]
    assert previous_ghost_alpha.max() == int(255 * 0.3)
    next_ghost_alpha = pixels[2:6, 5:7, 3]
    assert next_ghost_alpha.max() == round(255 * 0.2)


def test_edges_have_only_one_ghost():
    frames = [_frame((255, 0, 0), 1), _frame((0, 255, 0), 3)]
    first = np.array(compose_onion_skin(frames, 0, scale=1))
    assert first[2:6, 1:3, 3].min() == 255
    assert int(first[2:6, 3:5, 3].max()) == int(255 * 0.18)
    assert int(first[2:6, 0:1, 3].sum()) == 0


def test_scale_upsamples_nearest_neighbour():
    frames = [_frame((255, 0, 0), 1), _frame((0, 255, 0), 3)]
    image = compose_onion_skin(frames, 1, scale=4)
    assert image.size == (32, 32)


def test_out_of_range_index_is_clamped():
    frames = [_frame((255, 0, 0), 1)]
    image = compose_onion_skin(frames, 9, scale=1)
    assert _alpha_sum(image) == 2 * 4 * 255


def test_empty_frame_list_raises():
    import pytest

    with pytest.raises(ValueError):
        compose_onion_skin([], 0)

