"""Tests for ActionTokenizer and SVAN Neural Network."""
import torch
import pytest
from spritemind.ml.tokenizer import ActionTokenizer
from spritemind.ml.svan import SVANModel
from spritemind.dataset.generator import CharacterGenerator
from spritemind.dataset.defects import DefectGenerator


def test_tokenizer_encode_decode():
    tokenizer = ActionTokenizer()
    assert tokenizer.vocab_size > 50

    prog = CharacterGenerator.generate_character(category="knight", seed=42)
    token_ids = tokenizer.encode_program(prog)

    assert token_ids[0] == tokenizer.sos_id
    assert token_ids[-1] == tokenizer.eos_id

    decoded_actions = tokenizer.decode_tokens(token_ids)
    assert len(decoded_actions) > 0


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
