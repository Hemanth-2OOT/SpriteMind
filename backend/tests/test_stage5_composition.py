"""Unit and Integration tests for SpriteMind Stage 5: Asset Catalog, Composition, Atlas, and Packs."""
import pytest
import io
import zipfile
import numpy as np
from PIL import Image

from spritemind.assets.library import AssetLibrary, AssetDefinition
from spritemind.assets.composer import AssetComposer
from spritemind.assets.atlas import TextureAtlasBuilder
from spritemind.assets.pack import AssetPack
from spritemind.ml.composition_model import SocketPredictorManager
from spritemind.server.app import (
    get_asset_catalog,
    compose_asset,
    animate_composition,
    export_asset_pack,
    get_composition_metrics,
)
from spritemind.server.schemas import (
    CompositionRequest,
    CompositionAnimationRequest,
    AssetPackExportRequest,
)


def test_asset_catalog_loading():
    assets = AssetLibrary.list_all()
    assert len(assets) >= 15

    categories = {a.category for a in assets}
    assert "characters" in categories
    assert "weapons" in categories
    assert "equipment" in categories
    assert "world" in categories
    assert "effects" in categories

    shotgun = AssetLibrary.get("weapon_pump_shotgun")
    assert shotgun is not None
    assert "grip" in shotgun.sockets
    assert "muzzle" in shotgun.sockets


def test_socket_predictor_inference():
    # Test on synthetic 32x32 RGBA canvas
    dummy_rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    dummy_rgba[6:28, 10:22, :3] = 180
    dummy_rgba[6:28, 10:22, 3] = 255

    # Procedural baseline
    proc_sockets = SocketPredictorManager.predict_sockets(dummy_rgba, engine_mode="procedural_baseline")
    assert "hand_r" in proc_sockets
    assert "head" in proc_sockets

    # Trained neural model
    neural_sockets = SocketPredictorManager.predict_sockets(dummy_rgba, engine_mode="trained_model")
    for name in ["hand_r", "hand_l", "head", "back"]:
        assert name in neural_sockets
        x, y = neural_sockets[name]
        assert 0 <= x <= 31
        assert 0 <= y <= 31


def test_static_asset_composition():
    # 1. Zombie + Shotgun
    comp1 = AssetComposer.compose(
        character_asset_or_script="character_zombie_walker",
        weapon_id="weapon_pump_shotgun",
        engine_mode="trained_model",
    )
    assert comp1["status"] == "success"
    assert comp1["pixel_count"] > 200
    assert any("weapon" in layer for layer in comp1["layers_used"])
    assert any("body" in layer for layer in comp1["layers_used"])

    # 2. Soldier + Rifle + Helmet + Backpack + FX
    comp2 = AssetComposer.compose(
        character_asset_or_script="character_player_soldier",
        weapon_id="weapon_assault_rifle",
        helmet_id="armor_combat_helmet",
        backpack_id="equipment_military_backpack",
        fx_id="fx_muzzle_flash_shotgun",
        engine_mode="hybrid",
    )
    assert comp2["status"] == "success"
    assert len(comp2["layers_used"]) == 5
    assert comp2["pixel_count"] > 250


def test_animated_composition_synchronization():
    anim = AssetComposer.compose_animation(
        character_asset_or_script="character_zombie_walker",
        animation_category="walk",
        num_frames=4,
        weapon_id="weapon_pump_shotgun",
        engine_mode="trained_model",
    )
    assert anim["status"] == "success"
    assert anim["num_frames"] == 4
    assert len(anim["frames"]) == 4

    for f in anim["frames"]:
        assert f["pixel_count"] > 200
        assert "image_base64" in f
        assert "script_text" in f


def test_texture_atlas_packing_and_extraction():
    # Prepare 4 test sprites
    s1 = np.full((16, 16, 4), 255, dtype=np.uint8)
    s2 = np.full((24, 12, 4), 180, dtype=np.uint8)
    s3 = np.full((8, 32, 4), 100, dtype=np.uint8)

    sprites = [
        {"name": "sprite_1", "rgba": s1},
        {"name": "sprite_2", "rgba": s2},
        {"name": "sprite_3", "rgba": s3},
    ]

    atlas = TextureAtlasBuilder.build_atlas(sprites, padding=2)
    assert atlas["width"] >= 32
    assert atlas["height"] >= 32
    assert "sprite_1" in atlas["regions"]
    assert "sprite_2" in atlas["regions"]
    assert "sprite_3" in atlas["regions"]

    # Reversible extraction test: extracted sprite matches original
    ext1 = TextureAtlasBuilder.extract_sprite(atlas["atlas_array"], atlas["regions"]["sprite_1"])
    assert np.array_equal(ext1, s1)


def test_asset_pack_zip_export():
    pack = AssetPack.create_deadzone_starter_pack()
    zip_bytes = pack.export_zip()
    assert len(zip_bytes) > 2000

    # Read back and inspect archive integrity
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        names = z.namelist()
        assert "manifest.json" in names
        assert "atlas/texture_atlas.png" in names
        assert "atlas/texture_atlas.json" in names
        assert any(n.startswith("sprites/") for n in names)
        assert any(n.startswith("pixelscript/") for n in names)


def test_api_stage5_endpoints():
    # 1. Catalog
    cat = get_asset_catalog()
    assert len(cat) >= 15

    # 2. Compose
    req = CompositionRequest(
        character_id="character_player_soldier",
        weapon_id="weapon_tactical_pistol",
        engine="trained_model",
    )
    res = compose_asset(req)
    assert res["status"] == "success"
    assert "image_base64" in res

    # 3. Animate Composition
    anim_req = CompositionAnimationRequest(
        character_id="character_zombie_walker",
        animation_category="walk",
        num_frames=3,
        weapon_id="weapon_pump_shotgun",
        engine="trained_model",
    )
    anim_res = animate_composition(anim_req)
    assert anim_res["status"] == "success"
    assert len(anim_res["frames"]) == 3

    # 4. Export Pack
    pack_req = AssetPackExportRequest(pack_type="deadzone")
    pack_res = export_asset_pack(pack_req)
    assert pack_res["status"] == "success"
    assert "zip_base64" in pack_res

    # 5. Metrics
    metrics = get_composition_metrics()
    assert metrics["status"] == "ready"
    assert metrics["parameter_count"] == 619432
    assert "benchmark_results" in metrics
