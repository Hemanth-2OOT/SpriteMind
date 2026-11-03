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
