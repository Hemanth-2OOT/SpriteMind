"""Critic auto-repair: patch placement, and the render/repair round-trip the UI uses."""

from fastapi.testclient import TestClient

import numpy as np

from spritemind.critic.repair_engine import AutonomousRepairEngine
from spritemind.ml.pipeline import SpriteMindPipeline
from spritemind.pixelscript.parser import PixelScriptParser
from spritemind.renderer.engine import PixelScriptEngine
from spritemind.server.app import app

client = TestClient(app)
engine = AutonomousRepairEngine()

SCRIPT = """LAYER 0 base 1.00
PART torso none 16 14
RECT 13 14 6 8 2 1
END PART
SHADE 0 0 32 32 1 1
OUTLINE 0 0 32 32 1
END PROGRAM"""


def _generated_script() -> str:
    result = SpriteMindPipeline().generate(
        prompt="blue knight", engine_mode="procedural_baseline", seed=7, palette_name="pico8"
    )
    return result["frames"][0]["script_text"]


def test_patch_lands_before_end_program():
    patched = engine.apply_patch_to_script(SCRIPT, "RECT 20 9 1 4 0 1")
    lines = [ln for ln in patched.splitlines() if ln.strip()]
    assert lines[-1] == "END PROGRAM"
    assert lines[-2] == "RECT 20 9 1 4 0 1"


def test_patch_is_inside_the_program_not_appended_after_it():
    patched = engine.apply_patch_to_script(SCRIPT, "PIXEL 4 4 7")
    assert patched.splitlines().index("PIXEL 4 4 7") < patched.splitlines().index("END PROGRAM")
    assert len(PixelScriptParser.parse_text(patched).actions) == len(
        PixelScriptParser.parse_text(SCRIPT).actions
    ) + 1


def test_patch_changes_rendered_pixels():
    script = _generated_script()
    before = PixelScriptEngine.execute(script, palette_name="pico8").composite_array
    rows, cols = np.nonzero(before[..., 3])
    patch = f"RECT {int(cols.mean())} {int(rows.mean())} 2 2 0 1"
    patched = engine.apply_patch_to_script(script, patch)
    after = PixelScriptEngine.execute(patched, palette_name="pico8").composite_array
    assert np.any(before != after)


def test_multiline_patch_applies_every_command():
    patched = engine.apply_patch_to_script(SCRIPT, "PIXEL 3 3 5\nRECT 20 9 1 4 0 1")
    assert "PIXEL 3 3 5" in patched
    assert "RECT 20 9 1 4 0 1" in patched
    assert patched.splitlines()[-1] == "END PROGRAM"


def test_transform_patch_replaces_existing_transform():
    script = "PART arm torso 10 10\nTRANSFORM arm 1 0\nEND PART\nEND PROGRAM"
    patched = engine.apply_patch_to_script(script, "TRANSFORM arm 0 1")
    assert "TRANSFORM arm 0 1" in patched
    assert "TRANSFORM arm 1 0" not in patched


def test_render_response_carries_the_script_back():
    res = client.post("/api/render", json={"script_text": SCRIPT})
    assert res.status_code == 200
    assert res.json()["script_text"] == SCRIPT


def test_render_then_repair_round_trip():
    render = client.post("/api/render", json={"script_text": SCRIPT})
    frame = render.json()

    repair = client.post(
        "/api/critic/repair",
        json={"script_text": frame["script_text"], "action_patch": "RECT 20 9 1 4 0 1"},
    )
    assert repair.status_code == 200
    assert "RECT 20 9 1 4 0 1" in repair.json()["repaired_script"]
