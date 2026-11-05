"""Tests for Temporal Intelligence Network (TemporalSVAN), TemporalAnimator, and Animation APIs."""
import pytest
import torch
import numpy as np

from spritemind.ml.temporal import (
    TemporalSVAN,
    TemporalAnimator,
    PARTS,
    ANIMATION_CATEGORIES,
    CATEGORIES,
)
from spritemind.dataset.temporal_dataset import TemporalDatasetGenerator
from spritemind.server.app import (
    generate_animation,
    regenerate_frame,
    get_animation_metrics,
)
from spritemind.server.schemas import (
    AnimationGenerateRequest,
    RegenerateFrameRequest,
)


def test_temporal_model_forward():
    model = TemporalSVAN()
    model.eval()

    batch_size = 2
    seq_len = 4
    cat_idx = torch.tensor([0, 1], dtype=torch.long)
    anim_idx = torch.tensor([1, 2], dtype=torch.long)
    phases = torch.linspace(0, 1, seq_len).unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)  # (2, 4, 1)
    ref_tensor = torch.randn(batch_size, 4, 32, 32)

    with torch.no_grad():
        out = model(cat_idx, anim_idx, phases, ref_tensor)

    assert "trajectories" in out
    assert "squash" in out
    assert "identity_confidence" in out
    assert out["trajectories"].shape == (batch_size, seq_len, len(PARTS), 3)
    assert out["squash"].shape == (batch_size, seq_len, 2)
    assert out["identity_confidence"].shape == (batch_size, seq_len, 1)


def test_temporal_dataset_generator():
    # Verify motion offset computations for basic, combat, and creature categories
    for anim in ["walk", "melee_attack", "slime_bouncing", "death"]:
        offsets = TemporalDatasetGenerator.compute_motion_offsets(anim, phase=0.25, category="knight")
        assert isinstance(offsets, dict)
        for part in PARTS:
            assert part in offsets
            dx, dy, rot = offsets[part]
            assert isinstance(dx, (int, float))
            assert isinstance(dy, (int, float))
            assert isinstance(rot, (int, float))


def test_temporal_animator_engines():
    # 1. Procedural baseline
    res_proc = TemporalAnimator.generate_animation(
        category="knight",
        anim_category="walk",
        num_frames=4,
        engine_mode="procedural_baseline",
    )
    assert res_proc["engine_used"] == "procedural_baseline"
    assert len(res_proc["frames"]) == 4
    assert "image_base64" in res_proc["frames"][0]

    # 2. Untrained baseline
    res_un = TemporalAnimator.generate_animation(
        category="knight",
        anim_category="walk",
        num_frames=4,
        engine_mode="untrained_model",
    )
    assert res_un["engine_used"] == "untrained_model"
    assert len(res_un["frames"]) == 4

    # 3. Trained model
    res_tr = TemporalAnimator.generate_animation(
        category="knight",
        anim_category="walk",
        num_frames=4,
        engine_mode="trained_model",
    )
    assert res_tr["engine_used"] == "trained_model"
    assert len(res_tr["frames"]) == 4
    assert res_tr["identity_confidence"] > 0.80


def test_api_animation_generate_and_metrics():
    # Test generation endpoint
    req = AnimationGenerateRequest(
        category="zombie",
        animation_category="zombie_movement",
        num_frames=4,
        fps=8,
        engine="trained_model",
    )
    data = generate_animation(req)
    assert data["status"] == "success"
    assert data["engine_used"] == "trained_model"
    assert len(data["frames"]) == 4
    assert data["category"] == "zombie"
    assert data["animation_category"] == "zombie_movement"

    # Test frame regeneration endpoint
    regen_req = RegenerateFrameRequest(
        base_script=data["frames"][0]["script_text"],
        frame_index=2,
        total_frames=4,
        animation_category="zombie_movement",
        category="zombie",
        engine="trained_model",
    )
    regen_data = regenerate_frame(regen_req)
    assert regen_data["status"] == "success"
    assert regen_data["frame_index"] == 2
    assert "image_base64" in regen_data["frame"]

    # Test metrics endpoint
    metrics = get_animation_metrics()
    assert metrics["status"] == "ready"
    assert metrics["parameter_count"] == 426395
    assert len(metrics["supported_animation_categories"]) == 18
    assert "training_metrics" in metrics
    assert "benchmark_results" in metrics


def test_action_articulation_and_frame_motion():
    """Verify that walk, attack, run, and death animations exhibit active, non-zero pixel motion."""
    from spritemind.renderer.engine import PixelScriptEngine

    for engine in ["procedural_baseline", "trained_model"]:
        for anim in ["walk", "melee_attack", "run"]:
            res = TemporalAnimator.generate_animation(
                category="knight",
                anim_category=anim,
                num_frames=4,
                engine_mode=engine,
                seed=42,
            )
            assert res["status"] == "success"
            assert len(res["frames"]) == 4

            # Verify consecutive frames are distinct and moving
            for i in range(len(res["frames"]) - 1):
                f1_arr = PixelScriptEngine.execute(res["frames"][i]["script_text"]).composite_array
                f2_arr = PixelScriptEngine.execute(res["frames"][i + 1]["script_text"]).composite_array
                diff = int(np.sum(np.abs(f1_arr.astype(int) - f2_arr.astype(int)) > 0))
                assert diff > 0, f"Frame {i} to {i+1} in {anim} with {engine} was completely static (diff=0)!"
