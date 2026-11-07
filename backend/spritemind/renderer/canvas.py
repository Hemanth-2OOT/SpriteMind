"""Deterministic Pixel Art Canvas."""
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import deque
from spritemind.renderer.layers import Layer
from spritemind.utils.palettes import PALETTES, hex_to_rgba


class BodyPart:
    def __init__(self, name: str, parent: Optional[str] = None, anchor_x: int = 0, anchor_y: int = 0):
        self.name = name
        self.parent = parent
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y
        # Set of (x, y) coordinates
        self.pixels: Set[Tuple[int, int]] = set()

    @property
    def bounding_box(self) -> Optional[Dict[str, int]]:
        if not self.pixels:
            return None
        xs = [p[0] for p in self.pixels]
        ys = [p[1] for p in self.pixels]
        return {
            "min_x": min(xs),
            "min_y": min(ys),
            "max_x": max(xs),
            "max_y": max(ys),
            "width": max(xs) - min(xs) + 1,
            "height": max(ys) - min(ys) + 1,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parent": self.parent,
            "anchor_x": self.anchor_x,
            "anchor_y": self.anchor_y,
            "pixel_count": len(self.pixels),
            "bounding_box": self.bounding_box,
        }


class PixelCanvas:
    """Core deterministic 2D RGBA pixel canvas."""

    def __init__(self, width: int = 32, height: int = 32, palette_name: str = "pico8"):
        self.width = width
        self.height = height
        self.palette: List[Tuple[int, int, int, int]] = list(PALETTES.get(palette_name, PALETTES["pico8"]))
        self.layers: Dict[int, Layer] = {0: Layer(0, "base", width, height)}
        self.active_layer_id: int = 0
        self.active_part_name: Optional[str] = None
        self.parts: Dict[str, BodyPart] = {}
        # History stacks for undo/redo
        self._undo_stack: List[Dict[str, Any]] = []
        self._redo_stack: List[Dict[str, Any]] = []

    @property
    def active_layer(self) -> Layer:
        if self.active_layer_id not in self.layers:
            self.layers[self.active_layer_id] = Layer(self.active_layer_id, f"layer_{self.active_layer_id}", self.width, self.height)
        return self.layers[self.active_layer_id]

    def set_active_layer(self, layer_id: int):
        if layer_id not in self.layers:
            self.layers[layer_id] = Layer(layer_id, f"layer_{layer_id}", self.width, self.height)
        self.active_layer_id = layer_id

    def set_palette_color(self, index: int, color: Any):
        if isinstance(color, str):
            rgba = hex_to_rgba(color)
        elif isinstance(color, (tuple, list)):
            rgba = tuple(color) if len(color) == 4 else (color[0], color[1], color[2], 255)
        else:
            raise ValueError(f"Invalid color specification: {color}")

        # Expand palette if index is beyond current size
        while len(self.palette) <= index:
            self.palette.append((0, 0, 0, 255))
        self.palette[index] = rgba

    def resolve_color(self, color_spec: Any) -> Tuple[int, int, int, int]:
        """Resolves color from int palette index or hex/tuple."""
        if isinstance(color_spec, int):
            if 0 <= color_spec < len(self.palette):
                return self.palette[color_spec]
            return (0, 0, 0, 255)  # fallback
        elif isinstance(color_spec, str):
            return hex_to_rgba(color_spec)
        elif isinstance(color_spec, (tuple, list)):
            return tuple(color_spec) if len(color_spec) == 4 else (color_spec[0], color_spec[1], color_spec[2], 255)
        return (0, 0, 0, 255)

    def define_part(self, name: str, parent: Optional[str] = None, anchor_x: int = 0, anchor_y: int = 0):
        if name not in self.parts:
            self.parts[name] = BodyPart(name, parent, anchor_x, anchor_y)
        else:
            self.parts[name].parent = parent
            self.parts[name].anchor_x = anchor_x
            self.parts[name].anchor_y = anchor_y
        self.active_part_name = name

    def end_part(self):
        self.active_part_name = None

    def _track_pixel(self, x: int, y: int):
        if self.active_part_name and self.active_part_name in self.parts:
            self.parts[self.active_part_name].pixels.add((x, y))

    def set_pixel(self, x: int, y: int, color: Any):
        if 0 <= x < self.width and 0 <= y < self.height:
            rgba = self.resolve_color(color)
            self.active_layer.set_pixel(x, y, rgba)
            self._track_pixel(x, y)

    def erase_pixel(self, x: int, y: int):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.active_layer.set_pixel(x, y, (0, 0, 0, 0))
            if self.active_part_name and self.active_part_name in self.parts:
                self.parts[self.active_part_name].pixels.discard((x, y))

    def erase_rect(self, x: int, y: int, w: int, h: int):
        for cy in range(y, y + h):
            for cx in range(x, x + w):
                self.erase_pixel(cx, cy)

    def draw_rect(self, x: int, y: int, w: int, h: int, color: Any, filled: bool = True):
        rgba = self.resolve_color(color)
        if filled:
            for cy in range(y, y + h):
                for cx in range(x, x + w):
                    if 0 <= cx < self.width and 0 <= cy < self.height:
                        self.active_layer.set_pixel(cx, cy, rgba)
                        self._track_pixel(cx, cy)
        else:
            # Top and bottom
            for cx in range(x, x + w):
                if 0 <= cx < self.width and 0 <= y < self.height:
                    self.active_layer.set_pixel(cx, y, rgba)
                    self._track_pixel(cx, y)
                if 0 <= cx < self.width and 0 <= (y + h - 1) < self.height:
                    self.active_layer.set_pixel(cx, y + h - 1, rgba)
                    self._track_pixel(cx, y + h - 1)
            # Left and right
            for cy in range(y + 1, y + h - 1):
                if 0 <= x < self.width and 0 <= cy < self.height:
                    self.active_layer.set_pixel(x, cy, rgba)
                    self._track_pixel(x, cy)
                if 0 <= (x + w - 1) < self.width and 0 <= cy < self.height:
                    self.active_layer.set_pixel(x + w - 1, cy, rgba)
                    self._track_pixel(x + w - 1, cy)

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: Any):
        """Bresenham's line algorithm."""
        rgba = self.resolve_color(color)
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        cx, cy = x0, y0
        while True:
            if 0 <= cx < self.width and 0 <= cy < self.height:
                self.active_layer.set_pixel(cx, cy, rgba)
                self._track_pixel(cx, cy)

            if cx == x1 and cy == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy

    def _opaque_mask(self) -> np.ndarray:
        return self.composite()[:, :, 3] > 0

    def _region_window(self, x: int, y: int, w: int, h: int):
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + w), min(self.height, y + h)
        if x1 <= x0 or y1 <= y0:
            return None
        mask = self._opaque_mask()
        padded = np.pad(mask, 1)
        window = padded[y0 : y1 + 2, x0 : x1 + 2]
        rw, rh = x1 - x0, y1 - y0
        return x0, y0, rw, rh, mask[y0:y1, x0:x1], window

    def outline_rect(self, x: int, y: int, w: int, h: int, color: Any):
        """Draws a 1px border in the transparent ring that touches opaque pixels inside the region."""
        region = self._region_window(x, y, w, h)
        if region is None:
            return
        x0, y0, rw, rh, inside, window = region
        neighbours = (
            window[0:rh, 1 : rw + 1]
            | window[2 : rh + 2, 1 : rw + 1]
            | window[1 : rh + 1, 0:rw]
            | window[1 : rh + 1, 2 : rw + 2]
        )
        target = (~inside) & neighbours
        rgba = self.resolve_color(color)
        for yy, xx in zip(*np.nonzero(target)):
            self.active_layer.set_pixel(x0 + int(xx), y0 + int(yy), rgba)

    def shade_rect(self, x: int, y: int, w: int, h: int, color: Any, direction: int = 1):
        """Recolors opaque edge pixels that face `direction` (0 up, 1 down, 2 left, 3 right)."""
        region = self._region_window(x, y, w, h)
        if region is None:
            return
        x0, y0, rw, rh, inside, window = region
        facing = {
            0: window[0:rh, 1 : rw + 1],
            1: window[2 : rh + 2, 1 : rw + 1],
            2: window[1 : rh + 1, 0:rw],
            3: window[1 : rh + 1, 2 : rw + 2],
        }[int(direction) % 4]
        target = inside & (~facing)
        rgba = self.resolve_color(color)
        for yy, xx in zip(*np.nonzero(target)):
            self.active_layer.set_pixel(x0 + int(xx), y0 + int(yy), rgba)

    def aura_rect(self, x: int, y: int, w: int, h: int, color: Any, density: int = 2):
        """Dithers a halo of `color` just outside the silhouette inside the region.

        `density` 1 paints a solid ring, 2 every other pixel (checkerboard), 3-4 sparser,
        so an elemental aura reads as sparks instead of a second outline.
        """
        region = self._region_window(x, y, w, h)
        if region is None:
            return
        x0, y0, rw, rh, inside, window = region
        neighbours = (
            window[0:rh, 1 : rw + 1]
            | window[2 : rh + 2, 1 : rw + 1]
            | window[1 : rh + 1, 0:rw]
            | window[1 : rh + 1, 2 : rw + 2]
        )
        target = (~inside) & neighbours
        step = max(1, int(density))
        rgba = self.resolve_color(color)
        for yy, xx in zip(*np.nonzero(target)):
            gx, gy = x0 + int(xx), y0 + int(yy)
            if step > 1 and (gx + gy) % step != 0:
                continue
            self.active_layer.set_pixel(gx, gy, rgba)

    def flood_fill(self, start_x: int, start_y: int, color: Any):
        """Deterministic 4-way flood fill."""
        if not (0 <= start_x < self.width and 0 <= start_y < self.height):
            return
        target_rgba = self.active_layer.get_pixel(start_x, start_y)
        fill_rgba = self.resolve_color(color)
        if target_rgba == fill_rgba:
            return

        queue = deque([(start_x, start_y)])
        visited: Set[Tuple[int, int]] = set([(start_x, start_y)])

        while queue:
            cx, cy = queue.popleft()
            self.active_layer.set_pixel(cx, cy, fill_rgba)
            self._track_pixel(cx, cy)

            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) not in visited:
                        if self.active_layer.get_pixel(nx, ny) == target_rgba:
                            visited.add((nx, ny))
                            queue.append((nx, ny))

    def reuse_part(self, part_name: str, dest_x: int, dest_y: int, flip_x: bool = False, flip_y: bool = False):
        """Copies an existing body part's pixels relative to its anchor to dest_x, dest_y."""
        if part_name not in self.parts or not self.parts[part_name].pixels:
            return
        src_part = self.parts[part_name]
        bbox = src_part.bounding_box
        if not bbox:
            return

        # Read pixels from layers where they actually live
        pixel_data = []
        for px, py in src_part.pixels:
            rgba = (0, 0, 0, 0)
            if self.active_layer.get_pixel(px, py)[3] > 0:
                rgba = self.active_layer.get_pixel(px, py)
            else:
                for layer in self.layers.values():
                    c = layer.get_pixel(px, py)
                    if c[3] > 0:
                        rgba = c
                        break
            pixel_data.append((px, py, rgba))

        ref_x = src_part.anchor_x if src_part.anchor_x != 0 else bbox["min_x"]
        ref_y = src_part.anchor_y if src_part.anchor_y != 0 else bbox["min_y"]
        w = bbox["width"]
        h = bbox["height"]

        for px, py, rgba in pixel_data:
            rel_x = px - ref_x
            rel_y = py - ref_y

            if flip_x:
                rel_x = (w - 1) - rel_x
            if flip_y:
                rel_y = (h - 1) - rel_y

            nx = dest_x + rel_x
            ny = dest_y + rel_y

            if 0 <= nx < self.width and 0 <= ny < self.height:
                self.active_layer.set_pixel(nx, ny, rgba)
                self._track_pixel(nx, ny)

    def transform_part(self, part_name: str, dx: int, dy: int, rot_deg: int = 0):
        """Translates and rotates existing part pixels around its anchor across whichever layer they reside."""
        if part_name not in self.parts or not self.parts[part_name].pixels:
            return
        part = self.parts[part_name]
        bbox = part.bounding_box
        anchor_x = part.anchor_x if part.anchor_x != 0 else (bbox["min_x"] if bbox else 0)
        anchor_y = part.anchor_y if part.anchor_y != 0 else (bbox["min_y"] if bbox else 0)

        # Gather pixels, locating their actual layer
        pixels_to_move = []
        for px, py in list(part.pixels):
            found_layer = None
            rgba = (0, 0, 0, 0)
            if self.active_layer.get_pixel(px, py)[3] > 0:
                found_layer = self.active_layer
                rgba = self.active_layer.get_pixel(px, py)
            else:
                for lid, layer in sorted(self.layers.items(), key=lambda kv: kv[0], reverse=True):
                    candidate_rgba = layer.get_pixel(px, py)
                    if candidate_rgba[3] > 0:
                        found_layer = layer
                        rgba = candidate_rgba
                        break

            if found_layer is not None:
                pixels_to_move.append((found_layer, px, py, rgba))
                found_layer.set_pixel(px, py, (0, 0, 0, 0))

        # Rotate around anchor point
        rad = math.radians(rot_deg) if rot_deg != 0 else 0.0
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        part.pixels.clear()
        for layer, px, py, rgba in pixels_to_move:
            if rot_deg != 0:
                rel_x = px - anchor_x
                rel_y = py - anchor_y
                rot_x = rel_x * cos_a - rel_y * sin_a
                rot_y = rel_x * sin_a + rel_y * cos_a
                nx = int(round(anchor_x + rot_x + dx))
                ny = int(round(anchor_y + rot_y + dy))
            else:
                nx = px + dx
                ny = py + dy

            if 0 <= nx < self.width and 0 <= ny < self.height:
                layer.set_pixel(nx, ny, rgba)
                part.pixels.add((nx, ny))

    def composite(self) -> np.ndarray:
        """Returns the final combined (height, width, 4) uint8 RGBA array."""
        out = np.zeros((self.height, self.width, 4), dtype=np.float32)

        # Sort layers by id ascending
        sorted_layer_ids = sorted(self.layers.keys())
        for lid in sorted_layer_ids:
            layer = self.layers[lid]
            if not layer.visible or layer.opacity <= 0.0:
                continue

            src = layer.buffer.astype(np.float32)
            src_alpha = (src[:, :, 3] / 255.0) * layer.opacity

            # Alpha compositing: out = src_rgb * src_a + out_rgb * out_a * (1 - src_a)
            out_alpha = out[:, :, 3] / 255.0
            new_alpha = src_alpha + out_alpha * (1.0 - src_alpha)

            # Avoid division by zero
            mask = new_alpha > 0.001
            for c in range(3):
                out[:, :, c][mask] = (
                    src[:, :, c][mask] * src_alpha[mask] +
                    out[:, :, c][mask] * out_alpha[mask] * (1.0 - src_alpha[mask])
                ) / new_alpha[mask]

            out[:, :, 3] = new_alpha * 255.0

        return np.clip(out, 0, 255).astype(np.uint8)

    def snapshot(self) -> Dict[str, Any]:
        """Creates a deep copy of layers and parts for undo/history."""
        return {
            "layers": {lid: l.copy() for lid, l in self.layers.items()},
            "parts": {name: BodyPart(name, p.parent, p.anchor_x, p.anchor_y) for name, p in self.parts.items()},
            "palette": list(self.palette),
            "active_layer_id": self.active_layer_id,
            "active_part_name": self.active_part_name,
        }

    def restore_snapshot(self, snap: Dict[str, Any]):
        self.layers = {lid: l.copy() for lid, l in snap["layers"].items()}
        self.parts = snap["parts"]
        self.palette = list(snap["palette"])
        self.active_layer_id = snap["active_layer_id"]
        self.active_part_name = snap["active_part_name"]
