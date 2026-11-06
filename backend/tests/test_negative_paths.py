import numpy as np
import pytest
from fastapi.testclient import TestClient

from spritemind.assets.atlas import TextureAtlasBuilder as AtlasBuilder
from spritemind.critic.visual_understanding import VisualUnderstandingPipeline
from spritemind.storage.backup import BackupManager
from spritemind.storage.database import StorageDatabase
from spritemind.server.app import app

client = TestClient(app)


def _sprite(w: int, h: int) -> np.ndarray:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., 0] = 255
    arr[..., 3] = 255
    return arr


def test_empty_atlas_is_a_blank_canvas_not_a_crash():
    result = AtlasBuilder.build_atlas([])

    assert result["regions"] == {}
    assert result["atlas_array"].shape == (32, 32, 4)
    assert not result["atlas_array"].any()


def test_atlas_rejects_oversized_sprites_with_a_readable_error():
    with pytest.raises(ValueError, match="cannot fit an atlas"):
        AtlasBuilder.build_atlas([{"name": "huge", "rgba": _sprite(600, 600)}], max_size=64)


def test_atlas_extract_outside_bounds_returns_empty_crop():
    atlas = AtlasBuilder.build_atlas([{"name": "a", "rgba": _sprite(8, 8)}])
    atlas_array = atlas["atlas_array"]

    crop = AtlasBuilder.extract_sprite(atlas_array, {"rect": [9999, 9999, 8, 8]})

    assert crop.size == 0


def test_critic_on_empty_canvas_reports_defects_without_raising():
    from spritemind.renderer.engine import PixelScriptEngine

    empty = PixelScriptEngine.execute(
        "LAYER 0 base 1.00", width=32, height=32, palette_name="pico8"
    )
    report = VisualUnderstandingPipeline().analyze(empty, category="knight", palette_name="pico8")

    assert report["pixel_fill_count"] == 0
    assert isinstance(report["detected_defects"], list)


def test_database_initialises_missing_parent_directory(tmp_path):
    db_path = tmp_path / "nested" / "deeper" / "spritemind.db"

    db = StorageDatabase(db_path=str(db_path))

    assert db.db_path.exists()
    assert db.get_storage_dashboard() is not None


def test_restore_of_unknown_backup_is_rejected(tmp_path):
    manager = BackupManager(backup_dir=str(tmp_path / "backups"))

    with pytest.raises(ValueError):
        manager.restore_backup(str(tmp_path / "backups" / "missing.zip"), confirm=True)


def test_restore_requires_explicit_confirmation(tmp_path):
    manager = BackupManager(backup_dir=str(tmp_path / "backups"))

    with pytest.raises(ValueError):
        manager.restore_backup(str(tmp_path / "backups" / "missing.zip"))


def test_export_pack_with_no_assets_returns_an_error_not_a_500():
    response = client.post("/api/asset_pack/export_zip", json={"pack_type": "custom", "asset_ids": []})

    assert response.status_code < 500


def test_generate_rejects_out_of_range_size():
    response = client.post("/api/generate", json={"prompt": "knight", "width": 4, "height": 4})

    assert response.status_code == 422
