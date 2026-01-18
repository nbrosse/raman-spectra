"""
Transformer model for Raman spectra (SpecBERT-style encoder-only).

Implements:
- Patch embedding for 1D spectra
- Sinusoidal or learned positional embeddings
- Optional CLS token
- Encoder stack using PyTorch (optionally xformers attention if available)
- Heads for masked spectral modeling (reconstruction) and classification/regression
"""

from __future__ import annotations

from typing import Literal

import math
import torch
import torch.nn as nn

try:
    from xformers.ops import memory_efficient_attention
    XFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    XFORMERS_AVAILABLE = False


def build_sinusoidal_position_embedding(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class PatchEmbedding1D(nn.Module):
    def __init__(self, patch_size: int, in_dim: int = 1, embed_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * in_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N) intensities
        b, n = x.shape
        assert n % self.patch_size == 0, "sequence length must be multiple of patch_size"
        num_patches = n // self.patch_size
        x = x.view(b, num_patches, self.patch_size)
        x = self.proj(x)
        return x  # (B, num_patches, embed_dim)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if XFORMERS_AVAILABLE and x.is_cuda:  # Only use xformers on GPU
            out = memory_efficient_attention(q, k, v)
        else:
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = torch.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, n, c)
        out = self.out(out)
        out = self.proj_drop(out)
        return out


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadSelfAttention(dim, num_heads, attn_dropout=attn_drop, proj_dropout=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SpecBERT(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        pos_encoding: Literal["sin", "learned"] = "sin",
        use_cls_token: bool = True,
        recon_head: bool = True,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_cls = use_cls_token
        self.patch_embed = PatchEmbedding1D(patch_size=patch_size, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        self.pos_encoding_type = pos_encoding
        self.pos_embed: nn.Parameter | None = None
        self.encoder = nn.ModuleList(
            [TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.recon_head = nn.Linear(embed_dim, patch_size) if recon_head else None
        self.classifier = (
            nn.Linear(embed_dim, num_classes) if num_classes is not None else None
        )

    def _get_positional(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.pos_encoding_type == "sin":
            pe = build_sinusoidal_position_embedding(seq_len, self.embed_dim).to(device)
            return pe
        # learned
        if self.pos_embed is None or self.pos_embed.shape[1] != seq_len:
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, self.embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        return self.pos_embed

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # x: (B, N)
        device = x.device
        tokens = self.patch_embed(x)  # (B, P, D)
        if self.use_cls:
            cls = self.cls_token.expand(x.shape[0], 1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        seq_len = tokens.shape[1]
        pos = self._get_positional(seq_len, device)
        tokens = tokens + pos
        for block in self.encoder:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        # heads
        recon = None
        logits = None
        if self.recon_head is not None:
            body = tokens[:, 1:, :] if self.use_cls else tokens
            recon = self.recon_head(body)  # (B, P, patch)
        if self.classifier is not None:
            pooled = tokens[:, 0, :] if self.use_cls else tokens.mean(dim=1)
            logits = self.classifier(pooled)
        return tokens, recon, logits


