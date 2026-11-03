"""Game-Ready Asset Pack Packaging, Serialization, and ZIP Exporter for SpriteMind Stage 5.

Packages collections of characters, equipment, animations, and world items
into standardized game engine asset bundles (Godot, Unity, Defold, DeadZone).
"""
import io
import os
import json
import zipfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

from spritemind.assets.library import AssetDefinition, AssetLibrary
from spritemind.assets.atlas import TextureAtlasBuilder
from spritemind.renderer.engine import PixelScriptEngine
from spritemind.renderer.export import export_spritesheet, export_gif


class AssetPack:
    """Standardized container for exportable 2D pixel art game asset packages."""

    def __init__(
        self,
        pack_id: str,
        name: str,
        version: str = "1.0.0",
        author: str = "SpriteMind AI Engine",
        license_str: str = "CC-BY-4.0 / MIT",
        description: str = "",
    ):
        self.pack_id = pack_id
        self.name = name
        self.version = version
        self.author = author
        self.license = license_str
        self.description = description
        self.assets: List[Dict[str, Any]] = []
        self.animations: List[Dict[str, Any]] = []

    def add_asset(self, asset: AssetDefinition) -> None:
        self.assets.append(asset.to_dict())

    def add_composed_asset(self, composed_dict: Dict[str, Any]) -> None:
        self.assets.append(composed_dict)

    def add_animation(self, anim_dict: Dict[str, Any]) -> None:
        self.animations.append(anim_dict)

    def build_atlas(self, padding: int = 2) -> Dict[str, Any]:
        """Builds a unified texture atlas for all static assets and animation keyframes."""
        sprites_to_pack = []

        # Static assets
        for a in self.assets:
            script = a.get("script_text", "")
            if script:
                render_res = PixelScriptEngine.execute(script, width=a.get("width", 32), height=a.get("height", 32))
                sprites_to_pack.append({
                    "name": a.get("asset_id") or a.get("name", "asset"),
                    "rgba": render_res.composite_array,
                })

        # Animation frames
        for anim in self.animations:
            anim_name = anim.get("name", "anim")
            for f in anim.get("frames", []):
                idx = f.get("frame_index", 0)
                script = f.get("script_text", "")
                if script:
                    r = PixelScriptEngine.execute(script, width=32, height=32)
                    sprites_to_pack.append({
                        "name": f"{anim_name}_frame_{idx:02d}",
                        "rgba": r.composite_array,
                    })

        return TextureAtlasBuilder.build_atlas(sprites_to_pack, padding=padding)

    def export_zip(self, output_path: Optional[str] = None) -> bytes:
        """Exports the complete asset bundle into a deterministic ZIP archive."""
        bio = io.BytesIO()

        with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # 1. Manifest
            manifest = {
                "pack_id": self.pack_id,
                "name": self.name,
                "version": self.version,
                "author": self.author,
                "license": self.license,
                "description": self.description,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_assets": len(self.assets),
                "total_animations": len(self.animations),
                "model_attribution": {
                    "svan_action_transformer": "svan_checkpoint_best.pt",
                    "visual_critic": "visual_critic_best.pt",
                    "temporal_svan": "temporal_svan_best.pt",
                    "socket_predictor": "socket_predictor_best.pt",
                },
                "provenance": "Generated locally by SpriteMind Stage 5 Game-Ready Pipeline",
                "assets": [],
                "animations": [],
            }

            # 2. Export individual static sprites & scripts
            for a in self.assets:
                aid = a.get("asset_id") or a.get("name", "asset").lower().replace(" ", "_")
                script = a.get("script_text", "")
                w = a.get("width", 32)
                h = a.get("height", 32)

                if script:
                    # Write PixelScript
                    zf.writestr(f"pixelscript/{aid}.pxs", script)

                    # Render and write PNG
                    render_res = PixelScriptEngine.execute(script, width=w, height=h)
                    pil_img = Image.fromarray(render_res.composite_array, mode="RGBA")
                    img_bio = io.BytesIO()
                    pil_img.save(img_bio, format="PNG")
                    zf.writestr(f"sprites/{aid}.png", img_bio.getvalue())

                # Metadata record
                manifest["assets"].append({
                    "id": aid,
                    "name": a.get("name"),
                    "category": a.get("category"),
                    "sockets": a.get("sockets", {}),
                    "hitboxes": a.get("hitboxes", {}),
                    "pivot": a.get("pivot", [16, 28]),
                })

            # 3. Export animation spritesheets & frames
            for anim in self.animations:
                aname = anim.get("name", "anim").lower().replace(" ", "_")
                frames = anim.get("frames", [])
                scripts = [f.get("script_text", "") for f in frames if f.get("script_text")]

                if scripts:
                    # Spritesheet PNG
                    ss_bytes = export_spritesheet(scripts, width=32, height=32, palette_name="pico8", scale=1)
                    zf.writestr(f"animations/{aname}_spritesheet.png", ss_bytes)

                    # GIF animation
                    gif_bytes = export_gif(scripts, width=32, height=32, fps=anim.get("fps", 8), palette_name="pico8", scale=2)
                    zf.writestr(f"animations/{aname}.gif", gif_bytes)

                manifest["animations"].append({
                    "name": anim.get("name"),
                    "animation_category": anim.get("animation_category"),
                    "num_frames": len(frames),
                    "fps": anim.get("fps", 8),
                    "spritesheet_path": f"animations/{aname}_spritesheet.png",
                })

            # 4. Export Texture Atlas & UV Map
            atlas_data = self.build_atlas(padding=2)
            if "png_bytes" in atlas_data:
                zf.writestr("atlas/texture_atlas.png", atlas_data["png_bytes"])
                atlas_json = {
                    "width": atlas_data["width"],
                    "height": atlas_data["height"],
                    "padding": atlas_data["padding"],
                    "regions": atlas_data["regions"],
                }
                zf.writestr("atlas/texture_atlas.json", json.dumps(atlas_json, indent=2))
                manifest["atlas"] = "atlas/texture_atlas.json"

            # 5. Write manifest.json
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        zip_bytes = bio.getvalue()

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as f:
                f.write(zip_bytes)

        return zip_bytes

    @classmethod
    def create_deadzone_starter_pack(cls) -> "AssetPack":
        """Constructs an asset pack tailored for top-down shooter 'Dead Zone'."""
        pack = cls(
            pack_id="pack_deadzone_alpha",
            name="Dead Zone: Quarantine Survivor Pack",
            version="1.0.0",
            description="Complete game-ready 2D pixel asset pack including survivor soldier, infected walker, 12G shotgun, tactical pistol, medkit, and muzzle flashes.",
        )

        # Characters
        for cid in ["character_player_soldier", "character_zombie_walker"]:
            asset = AssetLibrary.get(cid)
            if asset:
                pack.add_asset(asset)

        # Weapons & Gear
        for wid in ["weapon_pump_shotgun", "weapon_tactical_pistol", "armor_combat_helmet", "equipment_military_backpack"]:
            asset = AssetLibrary.get(wid)
            if asset:
                pack.add_asset(asset)

        # World
        for wid in ["world_medkit_large", "world_ammo_box_heavy", "world_explosive_barrel", "world_blast_door"]:
            asset = AssetLibrary.get(wid)
            if asset:
                pack.add_asset(asset)

        # FX & UI
        for fid in ["fx_muzzle_flash_shotgun", "fx_bullet_impact_sparks", "fx_blood_splatter", "ui_icon_skull_kill"]:
            asset = AssetLibrary.get(fid)
            if asset:
                pack.add_asset(asset)

        return pack
