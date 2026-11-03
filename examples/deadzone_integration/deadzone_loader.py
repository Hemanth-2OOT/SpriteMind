"""Dead Zone Game Engine Asset Pack Integration Example.

Demonstrates how game engines (e.g., Dead Zone) import, unpack, load,
and instantiate SpriteMind Stage 5 asset packs:
1. Importing and reading manifest.json from ZIP asset pack.
2. Loading character and weapon sprites.
3. Attaching weapon to character socket dynamically.
4. Loading animation spritesheet and parsing frames.
5. Simulating game runtime playback.
6. Spawning pickups and muzzle flash effects.
"""
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np


class DeadZoneAssetImporter:
    """Simulated Dead Zone engine asset manager."""

    def __init__(self, zip_path_or_bytes: Any):
        self.assets: Dict[str, Dict[str, Any]] = {}
        self.sprites: Dict[str, Image.Image] = {}
        self.spritesheets: Dict[str, Image.Image] = {}
        self.manifest: Dict[str, Any] = {}
        self.atlas_image: Optional[Image.Image] = None
        self.atlas_metadata: Dict[str, Any] = {}

        self._load_from_zip(zip_path_or_bytes)

    def _load_from_zip(self, zip_source: Any) -> None:
        if isinstance(zip_source, (str, Path)):
            zf = zipfile.ZipFile(zip_source, "r")
        else:
            zf = zipfile.ZipFile(io.BytesIO(zip_source), "r")

        with zf:
            # 1. Read Manifest
            if "manifest.json" in zf.namelist():
                self.manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

            # 2. Load Sprites
            for name in zf.namelist():
                if name.startswith("sprites/") and name.endswith(".png"):
                    aid = Path(name).stem
                    img_data = zf.read(name)
                    img = Image.open(io.BytesIO(img_data)).convert("RGBA")
                    self.sprites[aid] = img
                elif name.startswith("animations/") and name.endswith(".png"):
                    anim_id = Path(name).stem
                    img_data = zf.read(name)
                    img = Image.open(io.BytesIO(img_data)).convert("RGBA")
                    self.spritesheets[anim_id] = img
                elif name == "atlas/texture_atlas.png":
                    self.atlas_image = Image.open(io.BytesIO(zf.read(name))).convert("RGBA")
                elif name == "atlas/texture_atlas.json":
                    self.atlas_metadata = json.loads(zf.read(name).decode("utf-8"))

            # Index asset metadata from manifest
            for a in self.manifest.get("assets", []):
                self.assets[a["id"]] = a

    def get_character(self, character_id: str) -> Optional[Image.Image]:
        """Returns PIL RGBA sprite for character."""
        return self.sprites.get(character_id)

    def get_weapon(self, weapon_id: str) -> Optional[Image.Image]:
        """Returns PIL RGBA sprite for weapon."""
        return self.sprites.get(weapon_id)

    def attach_weapon_to_character(
        self,
        character_id: str,
        weapon_id: str,
    ) -> Image.Image:
        """Attaches weapon to character's hand_r socket and returns composite image."""
        char_img = self.sprites.get(character_id)
        wep_img = self.sprites.get(weapon_id)
        if not char_img or not wep_img:
            raise ValueError(f"Asset not found: {character_id} or {weapon_id}")

        char_meta = self.assets.get(character_id, {})
        wep_meta = self.assets.get(weapon_id, {})

        char_hand = char_meta.get("sockets", {}).get("hand_r", (21, 20))
        wep_grip = wep_meta.get("sockets", {}).get("grip", (7, 10))

        dx = char_hand[0] - wep_grip[0]
        dy = char_hand[1] - wep_grip[1]

        # Composite weapon over character with transparency
        composite = char_img.copy()
        composite.alpha_composite(wep_img, dest=(dx, dy))
        return composite

    def extract_animation_frames(
        self,
        spritesheet_id: str,
        frame_width: int = 32,
        frame_height: int = 32,
    ) -> List[Image.Image]:
        """Splits an animation spritesheet strip into individual frames."""
        sheet = self.spritesheets.get(spritesheet_id)
        if not sheet:
            return []

        w, h = sheet.size
        cols = w // frame_width
        rows = h // frame_height
        frames = []

        for r in range(rows):
            for c in range(cols):
                box = (c * frame_width, r * frame_height, (c + 1) * frame_width, (r + 1) * frame_height)
                frame = sheet.crop(box)
                frames.append(frame)

        return frames


def run_deadzone_integration_demo(zip_path: str = "data/asset_packs/deadzone_starter_pack.zip") -> Dict[str, Any]:
    """Runs a complete test verification of the Dead Zone engine asset import."""
    print("[Dead Zone Integration] Loading asset pack from:", zip_path)
    importer = DeadZoneAssetImporter(zip_path)

    manifest = importer.manifest
    print(f"[Dead Zone Integration] Loaded pack: {manifest.get('name')} v{manifest.get('version')}")
    print(f"[Dead Zone Integration] Total sprites loaded: {len(importer.sprites)}")
    print(f"[Dead Zone Integration] Atlas loaded: {importer.atlas_image is not None}")

    # 1. Load Character (Zombie Walker)
    zombie = importer.get_character("character_zombie_walker")
    assert zombie is not None, "Zombie walker sprite missing"

    # 2. Load Weapon (Shotgun)
    shotgun = importer.get_weapon("weapon_pump_shotgun")
    assert shotgun is not None, "Shotgun sprite missing"

    # 3. Attach Weapon to Zombie
    armed_zombie = importer.attach_weapon_to_character("character_zombie_walker", "weapon_pump_shotgun")
    assert armed_zombie.size == (32, 32), "Incorrect composite size"

    # Count opaque pixels
    arr = np.array(armed_zombie)
    opaque_px = (arr[:, :, 3] > 0).sum()
    print(f"[Dead Zone Integration] Armed zombie composited successfully ({opaque_px} opaque pixels)")

    # 4. Load Pickups & FX
    medkit = importer.sprites.get("world_medkit_large")
    assert medkit is not None, "Medkit pickup missing"
    muzzle_fx = importer.sprites.get("fx_muzzle_flash_shotgun")
    assert muzzle_fx is not None, "Muzzle flash missing"

    return {
        "status": "success",
        "pack_name": manifest.get("name"),
        "sprites_count": len(importer.sprites),
        "armed_zombie_pixel_count": int(opaque_px),
        "has_atlas": importer.atlas_image is not None,
    }


if __name__ == "__main__":
    run_deadzone_integration_demo()
