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

    def __init__(self, vocab_size: int = 256, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.semantic_encoder = SemanticEncoder(d_model=d_model)
        self.visual_encoder = VisualEncoder(in_channels=4, d_model=d_model)
        self.structure_graph = StructureGraph(d_model=d_model)
        self.action_transformer = DrawingActionTransformer(vocab_size=vocab_size, d_model=d_model)
        self.critic = VisualCritic(in_channels=4)

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
