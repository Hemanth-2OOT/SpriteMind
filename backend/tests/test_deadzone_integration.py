"""Automated test for Dead Zone integration example."""
import sys
from pathlib import Path
import pytest

# Ensure repo root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from examples.deadzone_integration.deadzone_loader import run_deadzone_integration_demo


def test_deadzone_loader_execution():
    res = run_deadzone_integration_demo("data/asset_packs/deadzone_starter_pack.zip")
    assert res["status"] == "success"
    assert res["sprites_count"] >= 10
    assert res["armed_zombie_pixel_count"] > 100
    assert res["has_atlas"] is True
