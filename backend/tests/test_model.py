"""Tests for ActionTokenizer and SVAN Neural Network."""
import torch
import pytest
from spritemind.ml.tokenizer import ActionTokenizer
from spritemind.pixelscript.ast import ActionType, PixelAction, PixelScriptProgram
from spritemind.ml.svan import SVANModel
from spritemind.dataset.generator import CharacterGenerator
from spritemind.dataset.defects import DefectGenerator
from spritemind.pixelscript.parser import PixelScriptParser


def test_tokenizer_encode_decode():
    tokenizer = ActionTokenizer()
    assert tokenizer.vocab_size > 50

    prog = CharacterGenerator.generate_character(category="knight", seed=42)
    token_ids = tokenizer.encode_program(prog)

    assert token_ids[0] == tokenizer.sos_id
    assert token_ids[-1] == tokenizer.eos_id

    decoded_actions = tokenizer.decode_tokens(token_ids)
    assert len(decoded_actions) > 0


ROUND_TRIP_ACTIONS = [
    PixelAction(ActionType.PALETTE, {"index": 3, "color": "#ff0000"}),
    PixelAction(ActionType.PALETTE, {"index": 4, "color": [10, 20, 30, 255]}),
    PixelAction(ActionType.LAYER, {"id": 1, "name": "equipment", "opacity": 1.0}),
    PixelAction(ActionType.PART, {"name": "weapon", "parent": "arm_right", "anchor_x": 20, "anchor_y": 12}),
    PixelAction(ActionType.PIXEL, {"x": 9, "y": 4, "color": 7}),
    PixelAction(ActionType.RECT, {"x": 12, "y": 14, "w": 8, "h": 7, "color": 10, "filled": True}),
    PixelAction(ActionType.LINE, {"x0": 2, "y0": 3, "x1": 18, "y1": 20, "color": 5}),
    PixelAction(ActionType.FILL, {"x": 5, "y": 5, "color": 9}),
    PixelAction(ActionType.ERASE, {"x": 2, "y": 2, "w": 3, "h": 4}),
    PixelAction(ActionType.REUSE, {"part_name": "head", "dest_x": 10, "dest_y": 4, "flip_x": True, "flip_y": False}),
    PixelAction(ActionType.TRANSFORM, {"part_name": "arm_left", "dx": 2, "dy": -1, "rot_deg": 15}),
    PixelAction(ActionType.END, {"target": "PART"}),
    PixelAction(ActionType.END, {"target": "PROGRAM"}),
]


@pytest.mark.parametrize("action", ROUND_TRIP_ACTIONS, ids=lambda a: a.action_type.value)
def test_tokenizer_round_trips_every_action(action):
    tokenizer = ActionTokenizer()
    decoded = tokenizer.decode_tokens(tokenizer.encode_action(action))

    assert len(decoded) == 1
    assert decoded[0].action_type == action.action_type
    for key, value in action.params.items():
        decoded_value = decoded[0].params.get(key)
        if key == "color" and isinstance(value, (list, tuple)):
            value = "#" + "".join(f"{c:02x}" for c in value)
        assert decoded_value == value, f"{key} lost in round-trip"


def test_tokenizer_round_trips_mixed_program():
    tokenizer = ActionTokenizer()
    program = PixelScriptProgram(actions=ROUND_TRIP_ACTIONS)
    decoded = tokenizer.decode_tokens(tokenizer.encode_program(program))

    assert [a.action_type for a in decoded] == [a.action_type for a in program.actions]
    assert decoded[-1].params["target"] == "PROGRAM"


def test_palette_without_color_renders_safely():
    tokenizer = ActionTokenizer()
    program = PixelScriptProgram(actions=[PixelAction(ActionType.PALETTE, {"index": 3})])
    decoded = tokenizer.decode_tokens(tokenizer.encode_program(program))
    assert decoded[0].params == {"index": 3}

    from spritemind.renderer.engine import PixelScriptEngine
    result = PixelScriptEngine.execute(PixelScriptProgram(actions=decoded), width=32, height=32)
    assert result.to_dict()["image_base64"]


def test_incremental_decoding_matches_full_forward():
    torch.manual_seed(0)
    model = SVANModel(vocab_size=64)
    model.eval()
    tokens = torch.tensor([[0, 1, 2, 3]])
    cond = torch.tensor([0])
    memory = torch.stack([model.semantic_encoder(cond, cond, cond, cond)] * 3, dim=1)

    with torch.no_grad():
        full = model.action_transformer(tokens, memory)
        cache = model.action_transformer.new_cache()
        steps = [
            model.action_transformer.forward_step(torch.tensor([[t]]), memory, cache, i)
            for i, t in enumerate(tokens[0].tolist())
        ]
    incremental = torch.cat(steps, dim=1)
    assert torch.allclose(full, incremental, atol=1e-5)


def test_defect_generator():
    prog = CharacterGenerator.generate_character(category="zombie", seed=42)
    corrupted, info = DefectGenerator.inject_defect(prog, defect_type="stray_pixel", seed=42)

    assert info["defect_type"] == "stray_pixel"
    assert info["bbox"] is not None
    assert info["suggested_fix"] is not None
    assert corrupted.metadata.get("has_defect") is True


def test_svan_model_forward():
    tokenizer = ActionTokenizer()
    model = SVANModel(vocab_size=tokenizer.vocab_size, d_model=64)

    batch_size = 2
    seq_len = 16
    tgt = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
    c_idx = torch.tensor([0, 1])
    s_idx = torch.tensor([0, 0])
    p_idx = torch.tensor([0, 1])
    pal_idx = torch.tensor([0, 0])

    logits = model(tgt, c_idx, s_idx, p_idx, pal_idx)
    assert logits.shape == (batch_size, seq_len, tokenizer.vocab_size)

    # Visual critic test
    dummy_img = torch.rand(batch_size, 4, 32, 32)
    q_score, def_logits = model.critic(dummy_img)
    assert q_score.shape == (batch_size, 1)
    assert def_logits.shape == (batch_size, 6)


def test_grammar_mask_restricts_token_classes():
    tokenizer = ActionTokenizer()
    cmd = tokenizer.token_to_id["CMD_PIXEL"]
    allowed = tokenizer.allowed_token_ids([tokenizer.sos_id, cmd])
    assert allowed
    assert all(tokenizer.id_to_token[t].startswith("NUM_") for t in allowed)

    boundary = tokenizer.allowed_token_ids([tokenizer.sos_id])
    assert tokenizer.eos_id in boundary
    assert all(tokenizer.id_to_token[t].startswith("CMD_") for t in boundary if t != tokenizer.eos_id)


def test_trained_generation_respects_grammar():
    from spritemind.ml.pipeline import SpriteMindPipeline
    from spritemind.pixelscript.validator import PixelScriptValidator

    pipeline = SpriteMindPipeline()
    pipeline.load_checkpoint("svan_checkpoint_best.pt")
    for seed in range(3):
        result = pipeline.generate(category="knight", engine_mode="trained_model", seed=seed)
        program = PixelScriptParser.parse_text(result["frames"][0]["script_text"])
        report = PixelScriptValidator.validate(program)
        assert report.is_valid, report
