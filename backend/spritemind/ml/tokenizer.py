"""Action Tokenizer for SVAN Neural Network."""
from typing import List, Dict, Any, Optional, Tuple
from spritemind.pixelscript.ast import PixelAction, ActionType, PixelScriptProgram


class ActionTokenizer:
    """Encodes PixelScript actions into discrete token IDs and decodes them back."""

    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._build_vocab()

    def _add_token(self, token: str):
        if token not in self.token_to_id:
            tid = len(self.token_to_id)
            self.token_to_id[token] = tid
            self.id_to_token[tid] = token

    def _build_vocab(self):
        # 1. Special tokens
        for tok in (self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN):
            self._add_token(tok)

        # 2. Action command tokens
        for act in ActionType:
            self._add_token(f"CMD_{act.value}")

        # 3. Numeric tokens: coordinates (0..64)
        for i in range(65):
            self._add_token(f"NUM_{i}")
        # Negative offsets (-16..-1)
        for i in range(-16, 0):
            self._add_token(f"NUM_{i}")

        # 4. Color index tokens (0..31)
        for i in range(32):
            self._add_token(f"COL_{i}")
        # Raw RGB(A) channels, used to preserve PALETTE colours losslessly
        for i in range(256):
            self._add_token(f"CH_{i}")

        # 5. Part and parent identifiers
        common_parts = [
            "none", "torso", "head", "eyes", "arm_left", "arm_right",
            "leg_left", "leg_right", "weapon", "wings", "tail", "horns",
            "base", "equipment", "details", "PROGRAM", "PART"
        ]
        for p in common_parts:
            self._add_token(f"NAME_{p}")

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def sos_id(self) -> int:
        return self.token_to_id[self.SOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode_action(self, action: PixelAction) -> List[int]:
        """Converts a single PixelAction into a slice of token IDs."""
        tokens = [f"CMD_{action.action_type.value}"]
        p = action.params

        if action.action_type == ActionType.PALETTE:
            tokens.append(f"COL_{int(p.get('index', 0)) % 32}")
            channels = self._color_channels(p.get("color"))
            if channels is None:
                tokens.append("NUM_0")
            else:
                tokens.extend(f"CH_{c}" for c in channels)

        elif action.action_type == ActionType.LAYER:
            tokens.append(f"NUM_{p.get('id', 0)}")
            tokens.append(f"NAME_{p.get('name', 'base')}")

        elif action.action_type == ActionType.PART:
            tokens.append(f"NAME_{p.get('name', 'torso')}")
            tokens.append(f"NAME_{p.get('parent') or 'none'}")
            tokens.append(f"NUM_{p.get('anchor_x', 0)}")
            tokens.append(f"NUM_{p.get('anchor_y', 0)}")

        elif action.action_type == ActionType.PIXEL:
            tokens.append(f"NUM_{p.get('x', 0)}")
            tokens.append(f"NUM_{p.get('y', 0)}")
            c = p.get("color", 1)
            c_val = c if isinstance(c, int) else 1
            tokens.append(f"COL_{c_val % 32}")

        elif action.action_type == ActionType.RECT:
            tokens.append(f"NUM_{p.get('x', 0)}")
            tokens.append(f"NUM_{p.get('y', 0)}")
            tokens.append(f"NUM_{p.get('w', 1)}")
            tokens.append(f"NUM_{p.get('h', 1)}")
            c = p.get("color", 1)
            c_val = c if isinstance(c, int) else 1
            tokens.append(f"COL_{c_val % 32}")
            tokens.append("NUM_1" if p.get("filled", True) else "NUM_0")

        elif action.action_type == ActionType.LINE:
            tokens.append(f"NUM_{p.get('x0', 0)}")
            tokens.append(f"NUM_{p.get('y0', 0)}")
            tokens.append(f"NUM_{p.get('x1', 0)}")
            tokens.append(f"NUM_{p.get('y1', 0)}")
            c = p.get("color", 1)
            c_val = c if isinstance(c, int) else 1
            tokens.append(f"COL_{c_val % 32}")

        elif action.action_type == ActionType.FILL:
            tokens.append(f"NUM_{p.get('x', 0)}")
            tokens.append(f"NUM_{p.get('y', 0)}")
            c = p.get("color", 1)
            c_val = c if isinstance(c, int) else 1
            tokens.append(f"COL_{c_val % 32}")

        elif action.action_type == ActionType.ERASE:
            tokens.append(f"NUM_{p.get('x', 0)}")
            tokens.append(f"NUM_{p.get('y', 0)}")
            tokens.append(f"NUM_{p.get('w', 1)}")
            tokens.append(f"NUM_{p.get('h', 1)}")

        elif action.action_type == ActionType.REUSE:
            tokens.append(f"NAME_{p.get('part_name', 'torso')}")
            tokens.append(f"NUM_{p.get('dest_x', 0)}")
            tokens.append(f"NUM_{p.get('dest_y', 0)}")
            tokens.append("NUM_1" if p.get("flip_x", False) else "NUM_0")
            tokens.append("NUM_1" if p.get("flip_y", False) else "NUM_0")

        elif action.action_type == ActionType.TRANSFORM:
            tokens.append(f"NAME_{p.get('part_name', 'torso')}")
            tokens.append(f"NUM_{p.get('dx', 0)}")
            tokens.append(f"NUM_{p.get('dy', 0)}")
            tokens.append(f"NUM_{p.get('rot_deg', 0)}")

        elif action.action_type == ActionType.END:
            tokens.append(f"NAME_{p.get('target', 'PROGRAM')}")

        return [self.token_to_id.get(t, self.unk_id) for t in tokens]

    def encode_program(self, program: PixelScriptProgram, max_length: Optional[int] = None) -> List[int]:
        ids = [self.sos_id]
        for a in program.actions:
            ids.extend(self.encode_action(a))
        ids.append(self.eos_id)

        if max_length is not None:
            if len(ids) < max_length:
                ids.extend([self.pad_id] * (max_length - len(ids)))
            else:
                ids = ids[:max_length]
        return ids

    def decode_tokens(self, token_ids: List[int]) -> List[PixelAction]:
        """Decodes token sequence into high-level PixelAction objects."""
        # Clean special tokens
        tokens = [self.id_to_token.get(tid, self.UNK_TOKEN) for tid in token_ids]
        actions: List[PixelAction] = []
        i = 0
        n = len(tokens)

        while i < n:
            tok = tokens[i]
            if tok in (self.PAD_TOKEN, self.SOS_TOKEN, self.UNK_TOKEN):
                i += 1
                continue
            if tok == self.EOS_TOKEN:
                break

            if tok.startswith("CMD_"):
                cmd = tok[4:]
                try:
                    act_type = ActionType(cmd)
                except ValueError:
                    i += 1
                    continue

                if act_type == ActionType.PALETTE and i + 2 < n:
                    idx = self._parse_col(tokens[i + 1], 0)
                    j = i + 2
                    channels: List[int] = []
                    while j < n and tokens[j].startswith("CH_") and len(channels) < 4:
                        channels.append(int(tokens[j][3:]))
                        j += 1
                    params: Dict[str, Any] = {"index": idx}
                    if len(channels) in (3, 4):
                        params["color"] = "#" + "".join(f"{c:02x}" for c in channels)
                    actions.append(PixelAction(act_type, params))
                    i = max(j, i + 2)
                elif act_type == ActionType.PIXEL and i + 3 < n:
                    x = self._parse_num(tokens[i + 1], 0)
                    y = self._parse_num(tokens[i + 2], 0)
                    col = self._parse_col(tokens[i + 3], 1)
                    actions.append(PixelAction(act_type, {"x": x, "y": y, "color": col}))
                    i += 4
                elif act_type == ActionType.RECT and i + 6 < n:
                    x = self._parse_num(tokens[i + 1], 0)
                    y = self._parse_num(tokens[i + 2], 0)
                    w = max(1, self._parse_num(tokens[i + 3], 1))
                    h = max(1, self._parse_num(tokens[i + 4], 1))
                    col = self._parse_col(tokens[i + 5], 1)
                    filled = self._parse_num(tokens[i + 6], 1) == 1
                    actions.append(PixelAction(act_type, {"x": x, "y": y, "w": w, "h": h, "color": col, "filled": filled}))
                    i += 7
                elif act_type == ActionType.LINE and i + 5 < n:
                    x0 = self._parse_num(tokens[i + 1], 0)
                    y0 = self._parse_num(tokens[i + 2], 0)
                    x1 = self._parse_num(tokens[i + 3], 0)
                    y1 = self._parse_num(tokens[i + 4], 0)
                    col = self._parse_col(tokens[i + 5], 1)
                    actions.append(PixelAction(act_type, {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "color": col}))
                    i += 6
                elif act_type == ActionType.FILL and i + 3 < n:
                    x = self._parse_num(tokens[i + 1], 0)
                    y = self._parse_num(tokens[i + 2], 0)
                    col = self._parse_col(tokens[i + 3], 1)
                    actions.append(PixelAction(act_type, {"x": x, "y": y, "color": col}))
                    i += 4
                elif act_type == ActionType.ERASE and i + 4 < n:
                    x = self._parse_num(tokens[i + 1], 0)
                    y = self._parse_num(tokens[i + 2], 0)
                    w = max(1, self._parse_num(tokens[i + 3], 1))
                    h = max(1, self._parse_num(tokens[i + 4], 1))
                    actions.append(PixelAction(act_type, {"x": x, "y": y, "w": w, "h": h}))
                    i += 5
                elif act_type == ActionType.REUSE and i + 5 < n:
                    name = self._parse_name(tokens[i + 1])
                    dx = self._parse_num(tokens[i + 2], 0)
                    dy = self._parse_num(tokens[i + 3], 0)
                    flip_x = self._parse_num(tokens[i + 4], 0) == 1
                    flip_y = self._parse_num(tokens[i + 5], 0) == 1
                    actions.append(PixelAction(act_type, {
                        "part_name": name, "dest_x": dx, "dest_y": dy,
                        "flip_x": flip_x, "flip_y": flip_y,
                    }))
                    i += 6
                elif act_type == ActionType.TRANSFORM and i + 4 < n:
                    name = self._parse_name(tokens[i + 1])
                    dx = self._parse_num(tokens[i + 2], 0)
                    dy = self._parse_num(tokens[i + 3], 0)
                    rot = self._parse_num(tokens[i + 4], 0)
                    actions.append(PixelAction(act_type, {
                        "part_name": name, "dx": dx, "dy": dy, "rot_deg": rot,
                    }))
                    i += 5
                elif act_type == ActionType.PART and i + 4 < n:
                    name = tokens[i + 1].replace("NAME_", "")
                    parent_raw = tokens[i + 2].replace("NAME_", "")
                    parent = None if parent_raw == "none" else parent_raw
                    ax = self._parse_num(tokens[i + 3], 0)
                    ay = self._parse_num(tokens[i + 4], 0)
                    actions.append(PixelAction(act_type, {"name": name, "parent": parent, "anchor_x": ax, "anchor_y": ay}))
                    i += 5
                elif act_type == ActionType.LAYER and i + 2 < n:
                    lid = self._parse_num(tokens[i + 1], 0)
                    name = tokens[i + 2].replace("NAME_", "")
                    actions.append(PixelAction(act_type, {"id": lid, "name": name, "opacity": 1.0}))
                    i += 3
                elif act_type == ActionType.END and i + 1 < n:
                    target = tokens[i + 1].replace("NAME_", "")
                    actions.append(PixelAction(act_type, {"target": target}))
                    i += 2
                else:
                    # Generic advance
                    i += 1
            else:
                i += 1

        return actions

    @staticmethod
    def _parse_name(tok: str) -> str:
        return tok[5:] if tok.startswith("NAME_") else tok.replace("NAME_", "")

    @staticmethod
    def _color_channels(color: Any) -> Optional[List[int]]:
        if isinstance(color, str):
            s = color.strip().lstrip("#")
            if len(s) in (6, 8):
                try:
                    return [int(s[i:i + 2], 16) for i in range(0, len(s), 2)]
                except ValueError:
                    return None
            return None
        if isinstance(color, (list, tuple)) and len(color) in (3, 4):
            try:
                return [max(0, min(255, int(c))) for c in color]
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _parse_num(tok: str, default: int = 0) -> int:
        if tok.startswith("NUM_"):
            try:
                return int(tok[4:])
            except ValueError:
                pass
        return default

    @staticmethod
    def _parse_col(tok: str, default: int = 1) -> int:
        if tok.startswith("COL_"):
            try:
                return int(tok[4:])
            except ValueError:
                pass
        return default
