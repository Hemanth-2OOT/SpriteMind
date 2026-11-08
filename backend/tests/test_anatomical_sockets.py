"""Tests for anatomical socket labels (R7)."""
import numpy as np

from spritemind.dataset.anatomical_sockets import (
    SOCKET_RULE,
    anatomical_sockets,
    bbox_legacy_sockets,
)
from spritemind.dataset.generator import CharacterGenerator
from spritemind.pixelscript.serializer import PixelScriptSerializer
from spritemind.renderer.engine import PixelScriptEngine


def render(category="knight", pose="idle", seed=7, palette="pico8"):
    prog = CharacterGenerator.generate_character(category=category, pose=pose, palette_name=palette, seed=seed)
    return PixelScriptEngine.execute(PixelScriptSerializer.serialize_program(prog), width=32, height=32)


def test_sockets_are_inside_the_canvas():
    canvas = render().canvas
    for point in anatomical_sockets(canvas).values():
        assert 0 <= point[0] <= 31 and 0 <= point[1] <= 31


def test_head_socket_tracks_the_drawn_head_part():
    canvas = render().canvas
    head_pixels = canvas.parts["head"].pixels
    xs = [p[0] for p in head_pixels]
    ys = [p[1] for p in head_pixels]
    expected = (round((min(xs) + max(xs)) / 2), round((min(ys) + max(ys)) / 2))
    assert anatomical_sockets(canvas)["head"] == expected


def test_hand_sockets_sit_on_the_drawn_arm_extremes():
    canvas = render(pose="attack").canvas
    sockets = anatomical_sockets(canvas)
    assert sockets["hand_r"] in canvas.parts["arm_right"].pixels
    assert sockets["hand_l"] in canvas.parts["arm_left"].pixels


def test_hand_sockets_are_laterally_separated():
    sockets = anatomical_sockets(render(category="wizard").canvas)
    assert sockets["hand_l"][0] < sockets["back"][0] < sockets["hand_r"][0]


def test_head_socket_differs_from_the_bbox_label():
    # The bbox label rides the silhouette top edge plus two pixels; the anatomical
    # label sits on the drawn head, so the two must not coincide in general.
    result = render(category="wizard")
    anatomical = anatomical_sockets(result.canvas)
    legacy = bbox_legacy_sockets(result.composite_array)
    assert anatomical["head"] != legacy["head"]


def test_missing_parts_fall_back_to_proportional_guesses():
    class Canvas:
        parts = {}

    sockets = anatomical_sockets(Canvas(), width=32, height=32)
    assert sockets["head"] == (16, 7)
    assert sockets["back"] == (16, 16)
    assert sockets["hand_r"] == (22, 21)
    assert sockets["hand_l"] == (10, 21)


def test_rule_is_documented():
    assert "distal" in SOCKET_RULE and "torso" in SOCKET_RULE
