"""SpriteMind Visual-Action Network (SVAN) Architecture.

Implements:
1. SemanticEncoder: Embeds category, style, pose, palette, resolution.
2. VisualEncoder: ConvNet + ResBlocks on 32x32 RGBA canvas + edge map.
3. StructureGraph: Hierarchical body-part conditioning.
4. DrawingActionTransformer: Decoder-only causal transformer (d_model=128, 4 heads, 4 layers).
5. VisualCritic: Defect detection & quality scoring network.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List


class SemanticEncoder(nn.Module):
    """Encodes character metadata (category, style, pose, palette) into conditioning vectors."""

    def __init__(
        self,
        d_model: int = 128,
        num_categories: int = 16,
        num_styles: int = 8,
        num_poses: int = 8,
        num_palettes: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.cat_embed = nn.Embedding(num_categories, 32)
        self.style_embed = nn.Embedding(num_styles, 32)
        self.pose_embed = nn.Embedding(num_poses, 32)
        self.palette_embed = nn.Embedding(num_palettes, 32)

        self.projection = nn.Sequential(
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        category_idx: torch.Tensor,
        style_idx: torch.Tensor,
        pose_idx: torch.Tensor,
        palette_idx: torch.Tensor,
    ) -> torch.Tensor:
        # Each input tensor: (batch_size,)
        c = self.cat_embed(category_idx)
        s = self.style_embed(style_idx)
        p = self.pose_embed(pose_idx)
        pal = self.palette_embed(palette_idx)
        cat_all = torch.cat([c, s, p, pal], dim=-1)
        return self.projection(cat_all)  # (batch_size, d_model)


class ResidualBlock2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + res)


class VisualEncoder(nn.Module):
    """Encodes 32x32 RGBA canvas (plus optional edge channel) into spatial/latent features."""

    def __init__(self, in_channels: int = 4, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock2D(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock2D(64),
            nn.Conv2d(64, d_model, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            ResidualBlock2D(d_model),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, in_channels, 32, 32)
        spatial_features = self.stem(x)  # (batch_size, d_model, 8, 8)
        pooled = self.pool(spatial_features).flatten(1)  # (batch_size, d_model)
        global_embed = self.proj(pooled)
        return global_embed, spatial_features


class StructureGraph(nn.Module):
    """Encodes anatomy relationships between parts (torso -> head, arms, legs, etc.)."""

    def __init__(self, d_model: int = 128, num_parts: int = 12):
        super().__init__()
        self.part_embed = nn.Embedding(num_parts, 64)
        self.relation_matrix = nn.Parameter(torch.randn(num_parts, num_parts) * 0.05)
        self.mlp = nn.Sequential(
            nn.Linear(64, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, part_indices: torch.Tensor) -> torch.Tensor:
        # part_indices: (batch_size, num_active_parts)
        embeds = self.part_embed(part_indices)
        # Message passing over relation matrix
        adj = torch.softmax(self.relation_matrix, dim=-1)
        graph_features = torch.matmul(adj[part_indices], embeds)
        pooled = graph_features.mean(dim=1)
        return self.mlp(pooled)  # (batch_size, d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class DrawingActionTransformer(nn.Module):
    """Decoder-only causal transformer that predicts structured PixelScript tokens."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # tgt_tokens: (batch_size, seq_len)
        # memory: (batch_size, mem_len, d_model)
        seq_len = tgt_tokens.size(1)
        tgt_emb = self.token_embed(tgt_tokens) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        tgt_mask = self.generate_causal_mask(seq_len, tgt_tokens.device)
        out = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.head(out)  # (batch_size, seq_len, vocab_size)
        return logits

    # ------------------------------------------------------------------
    # Incremental (cached) decoding: same weights, one token at a time.
    # ------------------------------------------------------------------
    def new_cache(self, max_len: int = 0, batch_size: int = 1) -> List[Dict[str, torch.Tensor]]:
        """Per-layer caches. With max_len > 0 the self-attention KV is a preallocated
        ring-free buffer written in place, which avoids re-copying a growing tensor
        at every decode step."""
        heads = self.transformer_decoder.layers[0].self_attn.num_heads
        head_dim = self.d_model // heads
        caches: List[Dict[str, torch.Tensor]] = []
        for layer in self.transformer_decoder.layers:
            cross = layer.multihead_attn
            # Cross-attention only ever needs the query projection of the step
            # token; keeping its slice avoids projecting keys and values again.
            entry: Dict[str, torch.Tensor] = {"w_q": cross.in_proj_weight[: self.d_model]}
            if cross.in_proj_bias is not None:
                entry["b_q"] = cross.in_proj_bias[: self.d_model]
            if max_len > 0:
                entry["k"] = torch.zeros(batch_size, heads, max_len, head_dim)
                entry["v"] = torch.zeros(batch_size, heads, max_len, head_dim)
                entry["len"] = 0
            caches.append(entry)
        return caches

    @staticmethod
    def _split(t: torch.Tensor, nhead: int) -> torch.Tensor:
        b, length, d = t.shape
        return t.view(b, length, nhead, d // nhead).transpose(1, 2)

    @staticmethod
    def _merge(t: torch.Tensor) -> torch.Tensor:
        b, _, length, _ = t.shape
        return t.transpose(1, 2).reshape(b, length, -1)

    def _project(self, x: torch.Tensor, attn: nn.MultiheadAttention):
        qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        return self._split(q, attn.num_heads), self._split(k, attn.num_heads), self._split(v, attn.num_heads)

    @staticmethod
    def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        return torch.matmul(torch.softmax(scores, dim=-1), v)

    def forward_step(
        self,
        token: torch.Tensor,
        memory: torch.Tensor,
        cache: List[Dict[str, torch.Tensor]],
        position: int,
    ) -> torch.Tensor:
        """Single-token forward pass with self/cross attention KV caching.

        Mathematically equivalent to `forward` restricted to the last position,
        but O(1) in sequence length per call.
        """
        x = self.token_embed(token) * math.sqrt(self.d_model)
        x = x + self.pos_encoder.pe[:, position : position + 1]

        split = self._split
        attend = self._attend
        merge = self._merge
        for layer, layer_cache in zip(self.transformer_decoder.layers, cache):
            attn = layer.self_attn
            q, k, v = self._project(x, attn)
            if "len" in layer_cache:
                n = layer_cache["len"]
                layer_cache["k"][:, :, n : n + 1, :] = k
                layer_cache["v"][:, :, n : n + 1, :] = v
                layer_cache["len"] = n + 1
                keys, values = layer_cache["k"][:, :, : n + 1, :], layer_cache["v"][:, :, : n + 1, :]
            else:
                layer_cache["k"] = k if "k" not in layer_cache else torch.cat([layer_cache["k"], k], dim=2)
                layer_cache["v"] = v if "v" not in layer_cache else torch.cat([layer_cache["v"], v], dim=2)
                keys, values = layer_cache["k"], layer_cache["v"]
            sa = attend(q, keys, values)
            sa = attn.out_proj(merge(sa))
            x = layer.norm1(x + sa)

            cross = layer.multihead_attn
            q_m = split(F.linear(x, layer_cache["w_q"], layer_cache.get("b_q")), cross.num_heads)
            if "k_mem" not in layer_cache:
                _, k_m, v_m = self._project(memory, cross)
                layer_cache["k_mem"], layer_cache["v_mem"] = k_m, v_m
            ca = attend(q_m, layer_cache["k_mem"], layer_cache["v_mem"])
            ca = cross.out_proj(merge(ca))
            x = layer.norm2(x + ca)

            ff = layer.linear2(layer.activation(layer.linear1(x)))
            x = layer.norm3(x + ff)

        return self.head(x)


class VisualCritic(nn.Module):
    """Predicts defect presence, defect classification, and overall visual quality."""

    def __init__(self, in_channels: int = 4, num_defect_classes: int = 6):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock2D(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.defect_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_defect_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, 4, 32, 32)
        feats = self.backbone(x)
        quality_score = self.quality_head(feats)  # (batch_size, 1) in [0, 1]
        defect_logits = self.defect_head(feats)   # (batch_size, num_defect_classes)
        return quality_score, defect_logits


class SVANModel(nn.Module):
    """Integrated SpriteMind Visual-Action Network."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        max_seq_len: int = 448,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.model_config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "max_seq_len": max_seq_len,
            "dropout": dropout,
        }
        self.semantic_encoder = SemanticEncoder(d_model=d_model)
        self.visual_encoder = VisualEncoder(in_channels=4, d_model=d_model)
        self.structure_graph = StructureGraph(d_model=d_model)
        self.action_transformer = DrawingActionTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.critic = VisualCritic(in_channels=4)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SVANModel":
        keys = ("vocab_size", "d_model", "nhead", "num_layers", "dim_feedforward", "max_seq_len", "dropout")
        return cls(**{k: config[k] for k in keys if k in config})

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        category_idx: torch.Tensor,
        style_idx: torch.Tensor,
        pose_idx: torch.Tensor,
        palette_idx: torch.Tensor,
        part_indices: Optional[torch.Tensor] = None,
        canvas_images: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Semantic conditioning
        sem_cond = self.semantic_encoder(category_idx, style_idx, pose_idx, palette_idx) # (B, d_model)

        # 2. Structure conditioning (if available)
        if part_indices is not None:
            struct_cond = self.structure_graph(part_indices)
        else:
            struct_cond = torch.zeros_like(sem_cond)

        # 3. Visual conditioning (if available)
        if canvas_images is not None:
            vis_cond, _ = self.visual_encoder(canvas_images)
        else:
            vis_cond = torch.zeros_like(sem_cond)

        # Combine memory tokens: (B, 3, d_model)
        memory = torch.stack([sem_cond, struct_cond, vis_cond], dim=1)

        # 4. Action generation
        logits = self.action_transformer(tgt_tokens, memory, tgt_key_padding_mask=tgt_key_padding_mask)
        return logits
