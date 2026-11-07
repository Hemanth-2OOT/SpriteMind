import json

import numpy as np

from spritemind.ml.goldens import CATEGORIES, HASH_FILE, PIXEL_TOLERANCE, differing_pixels, verify


def test_golden_hashes_are_committed_for_every_category():
    assert HASH_FILE.exists(), "run python scripts/golden_images.py --update"
    recorded = json.loads(HASH_FILE.read_text())
    assert sorted(recorded) == sorted(CATEGORIES)


def test_renderer_matches_the_golden_images():
    failures = verify()
    assert not failures, "\n".join(failures)


def test_pixel_diff_catches_a_single_missing_pixel():
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[8:24, 8:24] = (120, 120, 255, 255)
    same = rgba.copy()
    assert differing_pixels(rgba, same) <= PIXEL_TOLERANCE

    hole = rgba.copy()
    hole[16, 16] = (0, 0, 0, 0)
    assert differing_pixels(rgba, hole) == 1
