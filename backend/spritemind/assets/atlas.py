"""2D Texture Atlas Generator and UV Bin-Packer for SpriteMind Stage 5.

Packs multiple sprites into a single power-of-two texture atlas, avoiding
overlap, preserving transparent pixels, and generating game-engine UV coordinates.
"""
from typing import Dict, Any, List, Tuple, Optional
import io
import math
import numpy as np
from PIL import Image


class AtlasRegion:
    """Represents an individual sprite's packed placement in the atlas."""

    def __init__(self, name: str, x: int, y: int, width: int, height: int, atlas_w: int, atlas_h: int):
        self.name = name
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.u0 = round(x / float(atlas_w), 6)
        self.v0 = round(y / float(atlas_h), 6)
        self.u1 = round((x + width) / float(atlas_w), 6)
        self.v1 = round((y + height) / float(atlas_h), 6)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rect": [self.x, self.y, self.width, self.height],
            "uv": [self.u0, self.v0, self.u1, self.v1],
        }


class TextureAtlasBuilder:
    """Packs multiple sprites into a clean 2D texture atlas with metadata."""

    @classmethod
    def build_atlas(
        cls,
        sprites: List[Dict[str, Any]],  # [{"name": str, "rgba": np.ndarray}]
        padding: int = 2,
        max_size: int = 512,
    ) -> Dict[str, Any]:
        """Packs sprites using a 2D shelf bin-packing algorithm.
        Args:
            sprites: list of items with 'name' and 'rgba' (H, W, 4) numpy arrays.
            padding: transparent boundary padding in pixels.
            max_size: maximum allowed atlas dimension.
        Returns:
            Dict containing atlas_array, image_bytes, regions metadata, and stats.
        """
        if not sprites:
            empty = np.zeros((32, 32, 4), dtype=np.uint8)
            return {"atlas_array": empty, "regions": {}, "width": 32, "height": 32}

        # Sort sprites by height descending for optimal shelf packing
        items = []
        for s in sprites:
            arr = s["rgba"]
            items.append({
                "name": s["name"],
                "rgba": arr,
                "w": arr.shape[1],
                "h": arr.shape[0],
            })
        items.sort(key=lambda x: (x["h"], x["w"]), reverse=True)

        # Estimate required atlas size (power of two)
        total_area = sum((it["w"] + padding) * (it["h"] + padding) for it in items)
        atlas_dim = 64
        while atlas_dim * atlas_dim < total_area * 1.3 and atlas_dim < max_size:
            atlas_dim *= 2

        # Shelf packing loop
        packed_placements = []
        cur_x = padding
        cur_y = padding
        shelf_height = 0

        for it in items:
            w, h = it["w"], it["h"]
            # If item overflows current row, advance to next shelf
            if cur_x + w + padding > atlas_dim:
                cur_x = padding
                cur_y += shelf_height + padding
                shelf_height = 0

            # If item overflows atlas height, double atlas dimension
            while cur_y + h + padding > atlas_dim:
                if atlas_dim >= max_size:
                    break
                atlas_dim *= 2

            packed_placements.append({
                "name": it["name"],
                "rgba": it["rgba"],
                "x": cur_x,
                "y": cur_y,
                "w": w,
                "h": h,
            })

            cur_x += w + padding
            shelf_height = max(shelf_height, h)

        # Create RGBA canvas
        atlas_array = np.zeros((atlas_dim, atlas_dim, 4), dtype=np.uint8)
        regions_map: Dict[str, Dict[str, Any]] = {}

        for p in packed_placements:
            x, y, w, h = p["x"], p["y"], p["w"], p["h"]
            atlas_array[y : y + h, x : x + w] = p["rgba"]
            reg = AtlasRegion(p["name"], x, y, w, h, atlas_dim, atlas_dim)
            regions_map[p["name"]] = reg.to_dict()

        # Convert to PNG PIL Image
        pil_img = Image.fromarray(atlas_array, mode="RGBA")
        bio = io.BytesIO()
        pil_img.save(bio, format="PNG")
        png_bytes = bio.getvalue()

        return {
            "width": atlas_dim,
            "height": atlas_dim,
            "atlas_array": atlas_array,
            "png_bytes": png_bytes,
            "regions": regions_map,
            "sprite_count": len(items),
            "padding": padding,
        }

    @classmethod
    def extract_sprite(cls, atlas_array: np.ndarray, region: Dict[str, Any]) -> np.ndarray:
        """Extracts an individual sprite from an atlas using region metadata."""
        rect = region["rect"]
        x, y, w, h = rect[0], rect[1], rect[2], rect[3]
        return atlas_array[y : y + h, x : x + w].copy()
