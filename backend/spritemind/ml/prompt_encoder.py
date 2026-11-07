"""Character-level encoder that turns the request text into a conditioning vector.

The categorical encoders can only see the six labels the pipeline parsed out of a
prompt, so two requests that differ in wording reach the model as the same input.
This encoder keeps the words: it pools a small character embedding over the prompt
and projects it to ``d_model`` as one extra memory token.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

PROMPT_MAX_LEN = 64
# id 0 is padding, printable ASCII (32-126) maps to 1-95, anything else to 95.
_PAD_ID = 0
_UNKNOWN_ID = 95
CHAR_VOCAB = 96


def encode_prompt(text: Optional[str], max_len: int = PROMPT_MAX_LEN) -> List[int]:
    """Encodes a prompt into padded character ids."""
    ids: List[int] = []
    for ch in (text or "").strip().lower():
        code = ord(ch)
        ids.append(code - 31 if 32 <= code <= 126 else _UNKNOWN_ID)
        if len(ids) >= max_len:
            break
    ids.extend([_PAD_ID] * (max_len - len(ids)))
    return ids


def describe_record(record: Dict[str, Any]) -> str:
    """Builds the request text a dataset record would have come from.

    Training pairs are stored as structured attributes only, so the prompt text is
    reconstructed from them in the same shape a user writes: subject, gear, style.
    """
    category = str(record.get("category") or "knight")
    pose = str(record.get("pose") or "idle")
    style = str(record.get("style") or "classic_16bit")
    head = f"a {record.get('proportion') or 'normal'} {category}".strip()
    gear = [str(record[key]) for key in ("weapon", "accessory") if record.get(key) and record[key] != "none"]
    if gear:
        head += " with a " + " and ".join(gear)
    # Colour and element are not in older corpora yet; include them the moment a
    # record carries them so the encoder can learn the words the labels lose.
    extras = [str(record[key]) for key in ("colour", "element") if record.get(key) and str(record[key]).lower() != "none"]
    if extras:
        head += " in " + " and ".join(extras)
    motion = {"idle": "standing idle", "walk": "walking", "run": "running", "attack": "attacking"}.get(pose, pose)
    return f"{head}, {style} style, {motion}"


class PromptTextEncoder(nn.Module):
    """Pools a character embedding over the prompt into a single memory token."""

    def __init__(self, d_model: int = 128, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(CHAR_VOCAB, embed_dim, padding_idx=_PAD_ID)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        mask = (prompt_tokens != _PAD_ID).unsqueeze(-1).to(torch.float32)
        embedded = self.embedding(prompt_tokens)
        summed = (embedded * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        return self.norm(self.projection(pooled))
