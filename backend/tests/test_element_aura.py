import numpy as np

from spritemind.dataset.conditioning import (
    COLOUR_IDS,
    ELEMENT_IDS,
    colour_id,
    element_id,
    element_hue_matches,
)
from spritemind.dataset.generator import CharacterGenerator
from spritemind.ml.tokenizer import ActionTokenizer
from spritemind.pixelscript.ast import ActionType, PixelAction, PixelScriptProgram
from spritemind.pixelscript.parser import PixelScriptParser
from spritemind.pixelscript.serializer import PixelScriptSerializer
from spritemind.pixelscript.validator import PixelScriptValidator
from spritemind.renderer.engine import PixelScriptEngine


BODY = "LAYER 0 base 1.00\nPART torso none 12 12\nRECT 12 12 8 8 5 1\nEND PART\n"
AURA = "AURA 0 0 32 32 12 2\n"


def _execute(script: str):
    return PixelScriptEngine.execute(script)


def test_aura_parses_serialises_and_round_trips():
    script = "# META: colour=blue element=ice\n" + BODY + AURA + "END PROGRAM"
    program = PixelScriptParser.parse_text(script)
    aura = next(action for action in program.actions if action.action_type == ActionType.AURA)

    assert aura.params == {"x": 0, "y": 0, "w": 32, "h": 32, "color": 12, "density": 2}
    assert program.metadata == {"colour": "blue", "element": "ice"}

    serialized = PixelScriptSerializer.serialize_program(program)
    assert AURA.strip() in serialized
    assert PixelScriptParser.parse_text(serialized).actions == program.actions


def test_aura_only_paints_outside_the_silhouette():
    plain = _execute(BODY + "END PROGRAM").composite_array
    haloed = _execute(BODY + AURA + "END PROGRAM").composite_array

    plain_opaque = plain[:, :, 3] > 0
    haloed_opaque = haloed[:, :, 3] > 0
    assert haloed_opaque.sum() > plain_opaque.sum()
    assert np.all(plain_opaque <= haloed_opaque)
    # Inside the torso rect nothing moves: the aura works on the transparent ring.
    assert np.array_equal(plain[12:20, 12:20], haloed[12:20, 12:20])


def test_aura_density_controls_how_many_sparks_are_painted():
    sparse = _execute(BODY + "AURA 0 0 32 32 12 4\nEND PROGRAM").composite_array
    solid = _execute(BODY + "AURA 0 0 32 32 12 1\nEND PROGRAM").composite_array

    sparse_count = int((sparse[:, :, 3] > 0).sum())
    solid_count = int((solid[:, :, 3] > 0).sum())
    body_count = int((_execute(BODY + "END PROGRAM").composite_array[:, :, 3] > 0).sum())
    assert sparse_count < solid_count
    # A solid ring hugs the silhouette, so it adds far fewer pixels than a filled box would.
    assert 0 < solid_count - body_count < 64


def test_aura_is_deterministic():
    first = _execute(BODY + AURA + "END PROGRAM").composite_array
    second = _execute(BODY + AURA + "END PROGRAM").composite_array
    assert np.array_equal(first, second)


def test_tokenizer_encodes_aura_and_its_grammar():
    tokenizer = ActionTokenizer()
    action = PixelAction(ActionType.AURA, {"x": 0, "y": 0, "w": 32, "h": 32, "color": 12, "density": 3})
    ids = tokenizer.encode_action(action)

    assert tokenizer.id_to_token[ids[0]] == "CMD_AURA"
    assert len(ids) == 7
    program = PixelScriptProgram(actions=[action, PixelAction(ActionType.END, {})])
    decoded = tokenizer.decode_tokens(tokenizer.encode_program(program))
    assert decoded[0].action_type == ActionType.AURA
    assert decoded[0].params["density"] == 3

    state = tokenizer.grammar_start_state()
    for token_id in ids[:5]:
        state = tokenizer.advance_grammar(state, token_id)
    allowed = set(tokenizer.allowed_from_state(state) or [])
    colour_slot = {tid for tok, tid in tokenizer.token_to_id.items() if tok.startswith("COL_")}
    assert colour_slot and colour_slot <= allowed  # the 5th slot is the tint colour


def test_validator_rejects_an_out_of_range_aura_density():
    wild = PixelScriptProgram(actions=[
        PixelAction(ActionType.AURA, {"x": 0, "y": 0, "w": 8, "h": 8, "color": 1, "density": 9})
    ])
    report = PixelScriptValidator.validate(wild)
    assert not report.is_valid
    assert any("density" in error for error in report.errors)


def test_element_characters_get_a_hue_matching_aura():
    program = CharacterGenerator.generate_character(category="wizard", seed=13, element="fire")
    aura = next(action for action in program.actions if action.action_type == ActionType.AURA)

    result = PixelScriptEngine.execute(program)
    assert result.validation.is_valid
    assert element_hue_matches("fire", result.composite_array) > 0.02


def test_plain_characters_ship_no_aura():
    types = [action.action_type for action in CharacterGenerator.generate_character(category="knight", seed=5).actions]
    assert ActionType.AURA not in types


def test_conditioning_ids_are_stable_and_zero_is_none():
    assert colour_id(None) == 0 and colour_id("none") == 0
    assert element_id(None) == 0
    assert colour_id("Blue") == COLOUR_IDS["blue"]
    assert element_id("ICE") == ELEMENT_IDS["ice"]
    assert len(COLOUR_IDS) + 1 == 29 and len(ELEMENT_IDS) + 1 == 9
