"""Tests for deterministic RGBA PixelCanvas and rendering engine."""
import numpy as np
import pytest
from spritemind.renderer.canvas import PixelCanvas
from spritemind.renderer.engine import PixelScriptEngine
from spritemind.renderer.passes import RenderPasses
from spritemind.renderer.export import export_spritesheet, export_gif, array_to_base64


def test_canvas_primitives():
    canvas = PixelCanvas(width=16, height=16)

    # Pixel
    canvas.set_pixel(2, 2, (255, 0, 0, 255))
    comp = canvas.composite()
    assert tuple(comp[2, 2]) == (255, 0, 0, 255)

    # Erase
    canvas.erase_pixel(2, 2)
    comp = canvas.composite()
    assert comp[2, 2, 3] == 0

    # Rect filled
    canvas.draw_rect(4, 4, 4, 4, (0, 255, 0, 255), filled=True)
    comp = canvas.composite()
    assert np.all(comp[4:8, 4:8, 1] == 255)

    # Bresenham line
    canvas.draw_line(0, 0, 3, 3, (0, 0, 255, 255))
    comp = canvas.composite()
    assert comp[0, 0, 2] == 255
    assert comp[3, 3, 2] == 255


def test_part_reuse_and_transform():
    canvas = PixelCanvas(width=32, height=32)
    canvas.define_part("arm", anchor_x=5, anchor_y=5)
    canvas.draw_rect(5, 5, 2, 4, (255, 255, 0, 255), filled=True)
    canvas.end_part()

    assert "arm" in canvas.parts
    assert canvas.parts["arm"].bounding_box is not None

    # Reuse part (mirrored)
    canvas.reuse_part("arm", dest_x=20, dest_y=5, flip_x=True)
    comp = canvas.composite()
    assert comp[5, 20, 3] > 0


def test_render_passes_and_export():
    canvas = PixelCanvas(width=16, height=16)
    canvas.draw_rect(4, 4, 8, 8, (200, 100, 50, 255), filled=True)

    sil = RenderPasses.silhouette(canvas)
    assert sil[4, 4, 0] == 255 and sil[4, 4, 3] == 255

    edge = RenderPasses.edge_map(canvas)
    assert np.count_nonzero(edge[:, :, 3] > 0) > 0

    # Spritesheet export
    sheet_bytes, meta = export_spritesheet([canvas.composite(), canvas.composite()], cols=2)
    assert len(sheet_bytes) > 0
    assert meta["total_frames"] == 2

    # GIF export
    gif_bytes = export_gif([canvas.composite(), canvas.composite()], fps=8)
    assert len(gif_bytes) > 0
