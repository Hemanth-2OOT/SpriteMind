"""Auxiliary render passes: Silhouette, Edge map, Segmentation, Grid preview."""
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, List, Tuple
from spritemind.renderer.canvas import PixelCanvas

# Distinct vibrant colors for body parts in segmentation pass
PART_COLORS: List[Tuple[int, int, int, int]] = [
    (255, 60, 60, 255),    # Head - Red
    (60, 120, 255, 255),   # Torso - Blue
    (60, 230, 80, 255),    # Arm left - Green
    (230, 200, 40, 255),   # Arm right - Yellow
    (180, 70, 230, 255),   # Leg left - Purple
    (255, 140, 40, 255),   # Leg right - Orange
    (40, 230, 220, 255),   # Eyes - Cyan
    (230, 60, 160, 255),   # Weapon/Item - Magenta
    (150, 150, 150, 255),  # Accessory/Other - Grey
]


class RenderPasses:
    @staticmethod
    def silhouette(canvas: PixelCanvas) -> np.ndarray:
        """Returns RGBA array where opaque pixels are solid white and empty pixels are transparent."""
        comp = canvas.composite()
        alpha = comp[:, :, 3]
        sil = np.zeros_like(comp)
        mask = alpha > 0
        sil[mask] = [255, 255, 255, 255]
        return sil

    @staticmethod
    def edge_map(canvas: PixelCanvas) -> np.ndarray:
        """Detects sharp pixel boundaries/outlines."""
        comp = canvas.composite()
        alpha = (comp[:, :, 3] > 0).astype(np.int32)
        h, w = alpha.shape
        edges = np.zeros((h, w, 4), dtype=np.uint8)

        # Pixel edge: where pixel is opaque and any of 4 neighbors is transparent, or internal color jump
        for y in range(h):
            for x in range(w):
                if alpha[y, x] == 0:
                    continue
                is_edge = False
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if alpha[ny, nx] == 0:
                            is_edge = True
                            break
                        # Color difference threshold
                        if np.sum(np.abs(comp[y, x, :3].astype(np.int32) - comp[ny, nx, :3].astype(np.int32))) > 100:
                            is_edge = True
                            break
                    else:
                        is_edge = True
                        break
                if is_edge:
                    edges[y, x] = [0, 255, 255, 255]  # Cyan outline

        return edges

    @staticmethod
    def segmentation(canvas: PixelCanvas) -> np.ndarray:
        """Assigns each identified body part a distinct color ID for anatomical visualization."""
        h, w = canvas.height, canvas.width
        seg = np.zeros((h, w, 4), dtype=np.uint8)

        for idx, (pname, part) in enumerate(canvas.parts.items()):
            color = PART_COLORS[idx % len(PART_COLORS)]
            for px, py in part.pixels:
                if 0 <= px < w and 0 <= py < h:
                    seg[py, px] = color

        return seg

    @staticmethod
    def grid_overlay(comp: np.ndarray, scale: int = 8, grid_color=(80, 80, 80, 160)) -> np.ndarray:
        """Upscales nearest-neighbor and draws 1px grid lines between pixels."""
        h, w, _ = comp.shape
        img = Image.fromarray(comp, mode="RGBA").resize((w * scale, h * scale), Image.NEAREST)
        draw = ImageDraw.Draw(img)

        for x in range(0, w * scale + 1, scale):
            draw.line([(x, 0), (x, h * scale)], fill=grid_color)
        for y in range(0, h * scale + 1, scale):
            draw.line([(0, y), (w * scale, y)], fill=grid_color)

        return np.array(img)
