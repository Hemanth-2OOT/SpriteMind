import numpy as np

from spritemind.dataset.generator import CharacterGenerator
from spritemind.ml.tokenizer import ActionTokenizer
from spritemind.pixelscript.ast import ActionType, PixelAction, PixelScriptProgram
from spritemind.pixelscript.coherence import build_frame_script
from spritemind.pixelscript.parser import PixelScriptParser
from spritemind.pixelscript.serializer import PixelScriptSerializer
from spritemind.pixelscript.validator import PixelScriptValidator
from spritemind.renderer.engine import PixelScriptEngine


SCRIPT = "\n".join(
    [
        "LAYER 0 base 1.00",
        "PART torso none 10 10",
        "RECT 10 10 8 8 5 1",
        "END PART",
        "SHADE 10 10 8 8 1 1",
        "OUTLINE 0 0 32 32 1",
        "END PROGRAM",
    ]
)


def test_outline_and_shade_round_trip_through_parser_and_serializer():
    program = PixelScriptParser.parse_text(SCRIPT)
    types = [action.action_type for action in program.actions]
    assert ActionType.SHADE in types and ActionType.OUTLINE in types

    serialized = PixelScriptSerializer.serialize_program(program)
    assert "SHADE 10 10 8 8 1 1" in serialized
    assert "OUTLINE 0 0 32 32 1" in serialized
    assert PixelScriptParser.parse_text(serialized).actions == program.actions


def test_outline_adds_a_border_that_shade_never_touches():
    plain = PixelScriptEngine.execute("LAYER 0 base 1.00\nPART torso none 10 10\nRECT 10 10 8 8 5 1\nEND PART\nEND PROGRAM")
    outlined = PixelScriptEngine.execute(SCRIPT)

    plain_opaque = (plain.composite_array[:, :, 3] > 0)
    outlined_opaque = (outlined.composite_array[:, :, 3] > 0)
    assert outlined_opaque.sum() > plain_opaque.sum()
    assert int(outlined_opaque[9, 10]) == 1  # one pixel above the torso is now filled
    assert np.all(plain_opaque <= outlined_opaque)


def test_shade_darkens_only_the_requested_edge():
    down = PixelScriptEngine.execute(SCRIPT).composite_array
    up = PixelScriptEngine.execute(SCRIPT.replace("SHADE 10 10 8 8 1 1", "SHADE 10 10 8 8 12 0")).composite_array

    assert tuple(down[17, 12]) != tuple(down[10, 12])
    assert tuple(up[10, 12]) != tuple(up[17, 12])
    assert tuple(down[17, 12]) != tuple(up[17, 12])


def test_tokenizer_encodes_and_decodes_the_new_primitives():
    tokenizer = ActionTokenizer()
    program = PixelScriptParser.parse_text(SCRIPT)

    tokens = tokenizer.encode_program(program)
    decoded = tokenizer.decode_tokens(tokens)
    primitives = [action.action_type for action in decoded]

    assert primitives.count(ActionType.SHADE) == 1
    assert primitives.count(ActionType.OUTLINE) == 1
    shade = next(action for action in decoded if action.action_type == ActionType.SHADE)
    assert shade.params["direction"] == 1


def test_validator_rejects_bad_shade_directions_and_out_of_bounds_regions():
    # The parser folds directions into 0..3, so a wild value has to be checked on the AST.
    wild = PixelScriptProgram(actions=[PixelAction(ActionType.SHADE, {"x": 1, "y": 1, "w": 4, "h": 4, "color": 1, "direction": 9})])
    report = PixelScriptValidator.validate(wild)
    assert not report.is_valid
    assert any("direction" in error for error in report.errors)

    off_canvas = PixelScriptParser.parse_text(SCRIPT.replace("OUTLINE 0 0 32 32 1", "OUTLINE 40 40 8 8 1"))
    messages = " ".join(PixelScriptValidator.validate(off_canvas).errors)
    messages += " ".join(PixelScriptValidator.validate(off_canvas).warnings)
    assert "bound" in messages.lower()


def test_generated_characters_carry_a_finishing_pass():
    program = CharacterGenerator.generate_character(category="knight", seed=5)
    types = [action.action_type for action in program.actions]
    assert types[-4:] == [ActionType.LAYER, ActionType.SHADE, ActionType.OUTLINE, ActionType.END]

    result = PixelScriptEngine.execute(program)
    assert result.validation.is_valid
    assert (result.composite_array[:, :, 3] > 0).sum() > 200


def test_frame_transforms_are_applied_before_the_finishing_pass():
    line_count = len(SCRIPT.splitlines())
    frame = build_frame_script(SCRIPT, {"torso": (1, -1, 0)}, 32, 32, "pico8")
    lines = frame.splitlines()

    assert len(lines) == line_count + 1
    transform_index = next(i for i, line in enumerate(lines) if line.startswith("TRANSFORM"))
    assert transform_index < next(i for i, line in enumerate(lines) if line.startswith("SHADE"))
