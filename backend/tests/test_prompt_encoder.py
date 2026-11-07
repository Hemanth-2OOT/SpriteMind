from pathlib import Path

import torch

from spritemind.ml.prompt_encoder import (
    PROMPT_MAX_LEN,
    PromptTextEncoder,
    describe_record,
    encode_prompt,
)
from spritemind.ml.svan import SVANModel
from spritemind.ml.tokenizer import ActionTokenizer
from spritemind.ml.training import PixelScriptDataset


def test_encode_prompt_pads_and_truncates():
    ids = encode_prompt("a mage")
    assert len(ids) == PROMPT_MAX_LEN
    assert ids[:6] == [ord(c) - 31 for c in "a mage"]
    assert all(i == 0 for i in ids[6:])

    long_ids = encode_prompt("x" * (PROMPT_MAX_LEN + 20))
    assert len(long_ids) == PROMPT_MAX_LEN
    assert 0 not in long_ids


def test_encode_prompt_is_lowercase_and_maps_non_ascii():
    assert encode_prompt("Knight")[:6] == encode_prompt("knight")[:6]
    assert encode_prompt("käse")[:4][1] == 95


def test_describe_record_reads_attributes():
    text = describe_record(
        {"category": "mage", "style": "flat", "pose": "walk", "proportion": "chibi", "weapon": "staff"}
    )
    assert "chibi mage" in text
    assert "staff" in text
    assert "flat style" in text
    assert "walking" in text


def test_encoder_ignores_padding():
    encoder = PromptTextEncoder(d_model=32)
    encoder.eval()
    short = torch.tensor([encode_prompt("a knight")])
    padded = torch.tensor([[ord(c) - 31 for c in "a knight"] + [0] * 40])
    with torch.inference_mode():
        assert torch.allclose(encoder(short), encoder(padded), atol=1e-6)


def test_memory_has_prompt_token_only_when_enabled():
    with_prompt = SVANModel(vocab_size=401, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, prompt_text=True)
    without = SVANModel(vocab_size=401, d_model=32, nhead=2, num_layers=1, dim_feedforward=64)
    idx = torch.zeros(2, dtype=torch.long)
    tokens = torch.tensor([encode_prompt("a knight")] * 2)

    assert with_prompt.memory_tokens(idx, idx, idx, idx, tokens).shape == (2, 4, 32)
    assert with_prompt.memory_tokens(idx, idx, idx, idx).shape == (2, 3, 32)
    assert without.memory_tokens(idx, idx, idx, idx, tokens).shape == (2, 3, 32)


def test_prompt_encoder_survives_config_round_trip():
    model = SVANModel(vocab_size=401, d_model=32, nhead=2, num_layers=1, dim_feedforward=64, prompt_text=True)
    clone = SVANModel.from_config(model.model_config)
    assert clone.prompt_encoder is not None
    assert clone.model_config["prompt_max_len"] == PROMPT_MAX_LEN

    legacy = SVANModel.from_config({"vocab_size": 401, "d_model": 32, "nhead": 2, "num_layers": 1, "dim_feedforward": 64})
    assert legacy.prompt_encoder is None


def test_dataset_carries_prompt_tokens():
    ds_path = Path("data/datasets/generalization_10k")
    if not ds_path.exists():
        return
    ds = PixelScriptDataset(ds_path, ActionTokenizer(), split="train", max_samples=5)
    if len(ds) == 0:
        return
    sample = ds[0]
    assert sample["prompt_tokens"].shape == (PROMPT_MAX_LEN,)
    assert int(sample["prompt_tokens"][0]) != 0
