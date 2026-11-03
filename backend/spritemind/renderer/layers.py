"""Layer management for pixel art canvas."""
import numpy as np
from typing import Dict, Any, Optional


class Layer:
    def __init__(self, layer_id: int, name: str = "", width: int = 32, height: int = 32, opacity: float = 1.0, visible: bool = True):
        self.id = layer_id
        self.name = name or f"layer_{layer_id}"
        self.width = width
        self.height = height
        self.opacity = max(0.0, min(1.0, float(opacity)))
        self.visible = visible
        self.locked = False
        # RGBA buffer: shape (height, width, 4), uint8
        self.buffer = np.zeros((height, width, 4), dtype=np.uint8)

    def set_pixel(self, x: int, y: int, rgba: tuple):
        if 0 <= x < self.width and 0 <= y < self.height and not self.locked:
            self.buffer[y, x] = rgba

    def get_pixel(self, x: int, y: int) -> tuple:
        if 0 <= x < self.width and 0 <= y < self.height:
            return tuple(self.buffer[y, x])
        return (0, 0, 0, 0)

    def clear(self):
        if not self.locked:
            self.buffer.fill(0)

    def copy(self) -> "Layer":
        new_layer = Layer(self.id, self.name, self.width, self.height, self.opacity, self.visible)
        new_layer.locked = self.locked
        new_layer.buffer = np.copy(self.buffer)
        return new_layer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "opacity": self.opacity,
            "visible": self.visible,
            "locked": self.locked,
            "pixel_count": int(np.count_nonzero(self.buffer[:, :, 3] > 0)),
        }
