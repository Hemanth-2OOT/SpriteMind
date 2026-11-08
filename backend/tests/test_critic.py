import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient

from spritemind.critic.model import NeuralVisualCritic
from spritemind.critic.visual_understanding import VisualUnderstandingPipeline
from spritemind.critic.repair_engine import AutonomousRepairEngine
from spritemind.renderer.engine import PixelScriptEngine
from spritemind.server.app import inspect_sprite, repair_sprite, get_critic_metrics
from spritemind.server.schemas import InspectRequest, CriticRepairRequest


def test_neural_visual_critic_forward():
    model = NeuralVisualCritic(in_channels=4, num_classes=7)
    model.eval()
    x = torch.randn(2, 4, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out["defect_logits"].shape == (2, 7)
    assert out["severity_logits"].shape == (2, 3)
    assert out["quality_score"].shape == (2, 1)
    assert out["heatmap"].shape == (2, 1, 32, 32)


def test_visual_understanding_pipeline_defect_detection():
    # Construct a script with an intentional isolated stray pixel at (2, 2)
    script = """LAYER 0 base 1.00
PART torso none 16 14
RECT 13 14 6 5 8 1
PIXEL 2 2 8"""
    render_res = PixelScriptEngine.execute(script, width=32, height=32, palette_name="pico8")
    pipeline = VisualUnderstandingPipeline()
    report = pipeline.analyze(render_res, category="knight", palette_name="pico8")

    assert report["has_defects"] is True
    defect_types = [d["type"] for d in report["detected_defects"]]
    assert "stray_pixel" in defect_types


def test_visual_understanding_interior_hole_detection():
    # Construct solid block with 1px hole inside
    script = """LAYER 0 base 1.00
PART torso none 16 16
RECT 12 12 8 8 9 1
ERASE 15 15 1 1"""
    render_res = PixelScriptEngine.execute(script, width=32, height=32, palette_name="pico8")
    pipeline = VisualUnderstandingPipeline()
    report = pipeline.analyze(render_res, category="robot", palette_name="pico8")

    assert report["has_defects"] is True
    defect_types = [d["type"] for d in report["detected_defects"]]
    assert "unexpected_hole" in defect_types


def test_autonomous_repair_engine_closed_loop():
    engine = AutonomousRepairEngine()
    corrupted_script = """LAYER 0 base 1.00
PART torso none 16 14
RECT 13 14 6 5 8 1
PIXEL 2 2 8"""
    res = engine.refine_closed_loop(
        corrupted_script,
        category="knight",
        palette_name="pico8",
        max_iterations=3,
    )
    assert res["final_quality"] >= res["initial_quality"]
    assert res["iterations_completed"] >= 1
    assert "ERASE 2 2 1 1" in [a.get("action") for a in res["applied_repairs"]]


def test_api_critic_endpoints():
    # 1. Inspect
    inspect_data = inspect_sprite(InspectRequest(
        script_text="LAYER 0 base 1.00\nPART torso none 16 14\nRECT 13 14 6 5 8 1\nPIXEL 2 2 8",
        category="knight",
        palette_name="pico8"
    ))
    assert inspect_data["has_defects"] is True
    assert len(inspect_data["detected_defects"]) > 0

    # 2. Repair
    repair_data = repair_sprite(CriticRepairRequest(
        script_text="LAYER 0 base 1.00\nPART torso none 16 14\nRECT 13 14 6 5 8 1\nPIXEL 2 2 8",
        category="knight",
        palette_name="pico8",
        auto_refine=True,
        max_iterations=2
    ))
    assert repair_data["final_quality"] >= repair_data["initial_quality"]
    assert "repaired_script" in repair_data

    # 3. Metrics
    metrics_data = get_critic_metrics()
    assert "status" in metrics_data
    assert "model_architecture" in metrics_data


CLEAN_HUMANOID = """LAYER 0 base 1.00
PART head none 16 10
RECT 13 8 6 5 8 1
PART eyes none 16 10
PIXEL 14 10 1
PIXEL 17 10 1
PART torso none 16 16
RECT 13 14 6 4 8 1
PART leg_left none 14 22
RECT 13 18 2 6 8 1
PART leg_right none 18 22
RECT 17 18 2 6 8 1"""


def _analyze(script: str, category: str = "knight"):
    res = PixelScriptEngine.execute(script, width=32, height=32, palette_name="pico8")
    return VisualUnderstandingPipeline().analyze(res, category=category, palette_name="pico8")


def test_facial_features_mounted_on_head_are_not_disconnected():
    report = _analyze(CLEAN_HUMANOID)

    assert report["defects_count"] == 0


def test_between_legs_gap_is_not_reported_as_a_hole():
    report = _analyze(CLEAN_HUMANOID)

    assert [d["type"] for d in report["detected_defects"]] == []


def test_between_legs_gap_helper_rejects_off_centre_holes():
    from spritemind.critic.visual_understanding import _is_between_legs_gap

    class Torso:
        bounding_box = {"min_x": 12, "min_y": 13, "max_x": 20, "max_y": 18}

    centred = {"bbox": {"min_x": 15, "min_y": 20, "max_x": 16, "max_y": 22}}
    off_centre = {"bbox": {"min_x": 21, "min_y": 20, "max_x": 22, "max_y": 22}}
    above_hips = {"bbox": {"min_x": 15, "min_y": 4, "max_x": 16, "max_y": 5}}

    assert _is_between_legs_gap(centred, Torso(), 32) is True
    assert _is_between_legs_gap(off_centre, Torso(), 32) is False
    assert _is_between_legs_gap(above_hips, Torso(), 32) is False


def test_eyes_far_from_head_are_still_reported():
    script = CLEAN_HUMANOID.replace("PIXEL 14 10 1", "PIXEL 14 24 1").replace(
        "PIXEL 17 10 1", "PIXEL 17 24 1"
    )

    report = _analyze(script)

    assert {"stray_pixel", "disconnected_part"} & {
        d["type"] for d in report["detected_defects"]
    }


def test_two_segment_limb_is_measured_against_its_own_parent():
    # A shin sits four rows below the torso by design; it hangs off the leg.
    script = """LAYER 0 base 1.00
PART torso none 16 14
RECT 13 14 6 8 10 1
PART leg_left torso 13 22
RECT 13 22 2 3 2 1
PART shin_left leg_left 13 25
RECT 13 25 2 4 2 1
PART leg_right torso 17 22
RECT 17 22 2 3 2 1
PART shin_right leg_right 17 25
RECT 17 25 2 4 2 1"""
    report = _analyze(script)

    assert [d for d in report["detected_defects"] if d["type"] == "disconnected_part"] == []


def test_a_shin_gap_to_its_own_leg_is_still_reported():
    script = """LAYER 0 base 1.00
PART torso none 16 14
RECT 13 14 6 8 10 1
PART leg_left torso 13 22
RECT 13 22 2 3 2 1
PART shin_left leg_left 13 25
RECT 13 29 2 2 2 1"""
    report = _analyze(script)

    descriptions = [d.get("description", "") for d in report["detected_defects"]]
    assert any("shin_left" in d and "leg_left" in d for d in descriptions)
