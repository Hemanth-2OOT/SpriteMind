"""Sprite-sheet export across a whole animation set."""
import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from spritemind.renderer.export import export_action_sheet
from spritemind.server.app import app


def _frame(value: int, height: int = 32, width: int = 32) -> np.ndarray:
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[8:24, 8:24] = (value, value, value, 255)
    return arr


def test_action_sheet_rows_and_metadata():
    frames = {"walk": [_frame(10), _frame(20), _frame(30)], "attack": [_frame(40), _frame(50)]}
    png, meta = export_action_sheet(frames, scale=2, padding=1)

    assert png[:4] == b"\x89PNG"
    assert meta["frame_width"] == 64 and meta["frame_height"] == 64
    assert meta["columns"] == 3
    assert meta["width"] == 3 * 64 + 2 * 1
    assert meta["height"] == 2 * 64 + 1
    rows = {row["name"]: row for row in meta["actions"]}
    assert rows["walk"]["row"] == 0 and rows["walk"]["frames"] == 3
    assert rows["attack"]["row"] == 1 and rows["attack"]["frames"] == 2
    assert rows["attack"]["y"] == 65

    from PIL import Image
    import io

    image = Image.open(io.BytesIO(png))
    assert image.size == (meta["width"], meta["height"])
    assert image.getpixel((32, 32))[3] == 255
    assert image.getpixel((32, 32 + 65))[0] == 40


def test_action_sheet_rejects_mixed_frame_sizes():
    with pytest.raises(ValueError):
        export_action_sheet({"a": [_frame(1)], "b": [_frame(2, height=16, width=16)]})


def test_action_sheet_requires_frames():
    with pytest.raises(ValueError):
        export_action_sheet({"idle": []})


def test_export_sheet_endpoint():
    client = TestClient(app)
    script_a = "LAYER 0 base 1.00\nRECT 12 12 8 8 5 1\nEND PROGRAM"
    script_b = "LAYER 0 base 1.00\nRECT 13 12 8 8 6 1\nEND PROGRAM"
    res = client.post(
        "/api/export/sheet",
        json={"actions": {"walk": [script_a, script_b], "cast": [script_b]}, "scale": 2},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["format"] == "sheet"
    payload = body["image_base64"].split(",", 1)[1]
    raw = base64.b64decode(payload + "=" * (-len(payload) % 4))
    assert raw[:4] == b"\x89PNG"
    assert [row["name"] for row in body["metadata"]["actions"]] == ["walk", "cast"]


def test_export_sheet_endpoint_rejects_empty_actions():
    client = TestClient(app)
    res = client.post("/api/export/sheet", json={"actions": {}})
    assert res.status_code == 400
