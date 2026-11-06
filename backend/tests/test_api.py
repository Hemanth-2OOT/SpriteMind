"""Tests for FastAPI backend routes."""
import pytest
from spritemind.server.app import (
    get_health,
    get_models,
    generate,
    render,
    get_dataset_stats,
    list_assets,
    get_storage_dashboard,
    get_storage_records,
    list_trash,
    storage_cleanup_scan,
    list_backups_endpoint,
    sync_storage_registry,
)
from spritemind.server.schemas import GenerateRequest, RenderRequest


def test_api_health_endpoint():
    data = get_health()
    assert data["status"] == "healthy"
    assert "torch_version" in data
    assert "device" in data


def test_api_models_endpoint():
    data = get_models()
    assert "available_checkpoints" in data
    assert "supported_engines" in data
    assert "procedural_baseline" in data["supported_engines"]


def test_api_generate_endpoint():
    req = GenerateRequest(
        category="knight",
        style="classic_16bit",
        pose="idle",
        palette="pico8",
        width=32,
        height=32,
        num_frames=2,
        engine="procedural_baseline",
    )
    data = generate(req)
    assert data["engine_used"] == "procedural_baseline"
    assert len(data["frames"]) == 2
    assert "image_base64" in data["frames"][0]
    assert "critic" in data
    assert data["critic"]["quality_score"] > 0


def test_api_render_endpoint():
    req = RenderRequest(
        script_text="RECT 4 4 8 8 1 1\nPIXEL 6 6 2",
        width=16,
        height=16,
        palette="pico8",
    )
    data = render(req)
    assert "image_base64" in data
    assert data["pixel_count"] > 0


def test_api_dataset_stats_endpoint():
    data = get_dataset_stats("default_synth")
    assert "total_samples" in data
    assert "splits" in data
    assert data["total_samples"] > 0


def test_api_assets_endpoint():
    assets = list_assets()
    assert isinstance(assets, list)


def test_api_storage_endpoints():
    # The registry is built by scanning the data directory, so index it first
    # instead of depending on whatever a previous local run left behind.
    sync_storage_registry()
    dash = get_storage_dashboard()
    assert dash["status"] == "healthy"
    assert dash["total_files_indexed"] > 0

    recs = get_storage_records(table="storage_registry", limit=10)
    assert recs["table"] == "storage_registry"
    assert len(recs["records"]) > 0

    trash = list_trash()
    assert isinstance(trash, list)

    scan = storage_cleanup_scan()
    assert "candidate_count" in scan

    bkups = list_backups_endpoint()
    assert isinstance(bkups, list)
