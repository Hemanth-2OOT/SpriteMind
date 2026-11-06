import base64
import io

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from spritemind.pixelscript.transform import mirror_script
from spritemind.renderer.engine import PixelScriptEngine
from spritemind.server.app import app

client = TestClient(app)

SCRIPT = """LAYER 0 base 1.00
PART torso none 16 14
RECT 13 14 6 5 8 1
LINE 10 10 14 12 9
PIXEL 4 3 2"""


def _decode(image_base64: str) -> np.ndarray:
    payload = image_base64.split(",", 1)[-1]
    return np.array(Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGBA"))


def _render(script: str) -> np.ndarray:
    return PixelScriptEngine.execute(
        script, width=32, height=32, palette_name="pico8"
    ).composite_array


def test_mirror_script_reflects_every_coordinate():
    mirrored = mirror_script(SCRIPT, 32)

    assert "RECT 13 14 6 5 8 1" in mirrored  # 32 - (13 + 6) = 13, so it lands unchanged
    assert "PIXEL 27 3 2" in mirrored
    assert "LINE 21 10 17 12 9" in mirrored
    assert np.array_equal(_render(SCRIPT), _render(mirrored)[:, ::-1])


def test_action_set_left_facing_is_exact_mirror():
    body = {"prompt": "green knight with sword", "actions": ["idle"], "include_gif": False}

    right = client.post("/api/action_set", json=body).json()
    left = client.post("/api/action_set", json={**body, "facing": "left"}).json()

    assert right["facing"] == "right"
    assert left["facing"] == "left"
    right_frame = _decode(right["actions"][0]["frames"][0]["image_base64"])
    left_frame = _decode(left["actions"][0]["frames"][0]["image_base64"])
    assert np.array_equal(right_frame, left_frame[:, ::-1])
