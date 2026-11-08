"""The sprite canvas is 64x64 everywhere a person or client asks for one."""
import base64
import io

import numpy as np
from PIL import Image

from spritemind.dataset.generator import CharacterGenerator
from spritemind.dataset.temporal_dataset import TemporalDatasetGenerator
from spritemind.pixelscript.coherence import build_frame_script
from spritemind.pixelscript.rig import RigGeometry, resolution_scale
from spritemind.ml.temporal import TemporalAnimator
from spritemind.pixelscript.serializer import PixelScriptSerializer
from spritemind.renderer.engine import PixelScriptEngine
from spritemind.server.schemas import (
    ActionSetRequest,
    AnimationGenerateRequest,
    CompositionAnimationRequest,
    CompositionRequest,
    GenerateRequest,
    RenderRequest,
)


def _render(program, width: int):
    script = PixelScriptSerializer.serialize_program(program)
    return PixelScriptEngine.execute(script, width=width, height=width, palette_name="pico8"), script


def _islands(alpha: np.ndarray) -> int:
    seen = np.zeros_like(alpha, dtype=bool)
    rows, cols = alpha.nonzero()
    count = 0
    for start in zip(rows.tolist(), cols.tolist()):
        if seen[start]:
            continue
        count += 1
        stack = [start]
        seen[start] = True
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < alpha.shape[0] and 0 <= nx < alpha.shape[1]:
                        if alpha[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
    return count


def test_request_schemas_default_to_64():
    for schema in (
        GenerateRequest, ActionSetRequest, AnimationGenerateRequest, RenderRequest,
        CompositionRequest, CompositionAnimationRequest,
    ):
        model = schema.model_fields
        assert model["width"].default == 64, schema.__name__
        assert model["height"].default == 64, schema.__name__


def test_the_character_fills_a_64_canvas():
    for category in ("knight", "zombie", "wizard", "skeleton", "slime", "animal"):
        program = CharacterGenerator.generate_character(category=category, seed=3, width=64, height=64)
        result, _ = _render(program, 64)
        rows, cols = (result.composite_array[..., 3] > 0).nonzero()
        height = rows.max() - rows.min() + 1
        width = cols.max() - cols.min() + 1
        assert height >= 30 and width >= 20, (category, height, width)
        assert rows.max() < 64 and cols.max() < 64


def test_a_64_sprite_is_the_same_character_at_twice_the_size():
    small, _ = _render(CharacterGenerator.generate_character(category="knight", seed=5, width=32, height=32), 32)
    large, _ = _render(CharacterGenerator.generate_character(category="knight", seed=5, width=64, height=64), 64)
    small_px = int((small.composite_array[..., 3] > 0).sum())
    large_px = int((large.composite_array[..., 3] > 0).sum())
    # Scaling doubles each axis, so a filled 64 sprite covers about four times
    # the pixels of the 32 one. The outline pass keeps a few pixels thin.
    assert 2.5 <= large_px / small_px <= 4.2, (small_px, large_px)
    assert _islands(large.composite_array[..., 3] > 0) == 1


def test_the_rig_scales_with_the_canvas():
    program = CharacterGenerator.generate_character(category="knight", seed=7, width=32, height=32)
    base_script = PixelScriptSerializer.serialize_program(program)
    assert resolution_scale(32, 32) == 1
    assert resolution_scale(64, 64) == 2
    assert RigGeometry.from_script(base_script, 32, 32, "pico8").scale == 1


def test_a_64_animation_keeps_its_limbs_together():
    program = CharacterGenerator.generate_character(category="knight", seed=7, weapon="sword", width=32, height=32)
    base_script = PixelScriptSerializer.serialize_program(program)

    for animation in ("walk", "run", "attack", "dash", "special_ability"):
        for frame in range(6):
            offsets = TemporalDatasetGenerator.compute_motion_offsets(animation, frame / 6)
            posed = build_frame_script(base_script, offsets, 32, 32, "pico8")
            result = PixelScriptEngine.execute(posed, width=32, height=32, palette_name="pico8")
            alpha = result.composite_array[..., 3] > 0
            assert _islands(alpha) == 1, (animation, frame)


def test_the_animation_api_delivers_a_64_canvas():
    out = TemporalAnimator.generate_animation(
        category="knight", anim_category="run", num_frames=4, fps=8, seed=7, width=64, height=64
    )
    assert out["status"] == "success"
    for frame in out["frames"]:
        assert (frame["width"], frame["height"]) == (64, 64)
        data = frame["image_base64"].split(",", 1)[-1]
        image = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGBA")
        alpha = np.array(image)[..., 3] > 0
        assert _islands(alpha) == 1
