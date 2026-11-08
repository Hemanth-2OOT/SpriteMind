import numpy as np

from spritemind.dataset.generator import CharacterGenerator
from spritemind.renderer.engine import PixelScriptEngine
from spritemind.renderer.lighting import apply_lighting

PALETTE = [(0, 0, 0, 255), (80, 80, 80, 255), (200, 200, 200, 255), (240, 240, 240, 255)]


def _knight():
    program = CharacterGenerator.generate_character(category="knight", seed=7, width=32, height=32)
    result = PixelScriptEngine.execute(program, width=32, height=32)
    return result.composite_array, result.canvas.palette


def test_lighting_never_changes_the_silhouette():
    comp, palette = _knight()
    lit = apply_lighting(comp, palette)
    assert np.array_equal(comp[:, :, 3], lit[:, :, 3])


def test_lighting_only_uses_palette_colours():
    comp, palette = _knight()
    lit = apply_lighting(comp, palette)
    allowed = {(c[0], c[1], c[2]) for c in palette}
    used = {tuple(int(v) for v in lit[y, x, :3]) for y, x in zip(*np.nonzero(lit[:, :, 3]))}
    assert used <= allowed


def test_lighting_recolours_some_but_not_all_pixels():
    comp, palette = _knight()
    lit = apply_lighting(comp, palette)
    changed = int((comp[:, :, :3] != lit[:, :, :3]).any(axis=-1).sum())
    opaque = int((comp[:, :, 3] > 0).sum())
    assert 0 < changed < opaque


def test_lighting_is_deterministic():
    comp, palette = _knight()
    assert np.array_equal(apply_lighting(comp, palette), apply_lighting(comp.copy(), palette))


def test_lighting_is_a_no_op_on_an_empty_canvas():
    empty = np.zeros((8, 8, 4), dtype=np.uint8)
    assert np.array_equal(apply_lighting(empty, PALETTE), empty)


def test_lighting_is_a_no_op_with_a_single_colour():
    comp = np.zeros((8, 8, 4), dtype=np.uint8)
    comp[2:6, 2:6] = (120, 120, 200, 255)
    assert np.array_equal(apply_lighting(comp, [(120, 120, 200, 255)]), comp)


def test_render_result_shades_only_when_asked():
    program = CharacterGenerator.generate_character(category="rogue", seed=3, width=32, height=32)
    result = PixelScriptEngine.execute(program, width=32, height=32)
    plain = result.to_dict()
    shaded = result.to_dict(shade=True)
    assert plain["image_base64"] != shaded["image_base64"]
    assert plain["pixel_count"] == shaded["pixel_count"]


def test_zero_strength_is_the_flat_sprite():
    comp, palette = _knight()
    assert np.array_equal(apply_lighting(comp, palette, strength=0.0), comp)


def test_strength_changes_the_result():
    comp, palette = _knight()
    assert not np.array_equal(
        apply_lighting(comp, palette, strength=0.5), apply_lighting(comp, palette, strength=1.5)
    )
