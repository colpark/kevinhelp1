# src/models/layers/perceiver.py
"""
Perceiver IO implementation for sparse conditioning.
Used by PixNerDiT for encoder_type="perceiver".
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    """
    Direct Fourier feature embedding for coordinates.

    Simpler than LearnableFourierMLP - just applies sin/cos and a linear projection.
    Used for query coordinate embedding in the baseline (non-unified) configuration.
    """
    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 256,
        n_bands: int = 16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bands = n_bands

        # Frequency bands: 2^0, 2^1, ..., 2^(n_bands-1) scaled by pi
        freqs = (2.0 ** torch.arange(n_bands, dtype=torch.float32)) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)

        # Output projection: sin + cos for each input dim and band
        fourier_dim = in_features * n_bands * 2  # (sin, cos) * in_features * n_bands
        self.proj = nn.Linear(fourier_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features) coordinates
        returns: (..., out_features) embeddings
        """
        orig_shape = x.shape[:-1]
        orig_dtype = x.dtype
        x_flat = x.reshape(-1, self.in_features).float()  # (N, in_features)

        # (N, in_features, n_bands)
        proj = x_flat.unsqueeze(-1) * self.freqs.view(1, 1, -1)

        # (N, in_features * n_bands * 2)
        fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        fourier = fourier.reshape(x_flat.shape[0], -1)

        # Project to output dimension
        out = self.proj(fourier)
        out = out.reshape(*orig_shape, self.out_features)
        return out.to(dtype=orig_dtype)


class Attention(nn.Module):
    """Multi-head attention for Perceiver."""
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        q: (B, N_q, D)
        kv: (B, N_kv, D)
        mask: (B, N_kv) bool, True = valid
        """
        B, N_q, _ = q.shape
        _, N_kv, _ = kv.shape
        h = self.heads

        queries = self.to_q(q)
        k, v = self.to_kv(kv).chunk(2, dim=-1)

        # Reshape to multi-head
        queries = queries.view(B, N_q, h, -1).transpose(1, 2)   # (B, h, N_q, dim_head)
        k = k.view(B, N_kv, h, -1).transpose(1, 2)              # (B, h, N_kv, dim_head)
        v = v.view(B, N_kv, h, -1).transpose(1, 2)              # (B, h, N_kv, dim_head)

        # Attention
        attn = torch.matmul(queries, k.transpose(-2, -1)) * self.scale  # (B, h, N_q, N_kv)

        if mask is not None:
            # mask: (B, N_kv) -> (B, 1, 1, N_kv)
            mask = mask.unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, h, N_q, dim_head)
        out = out.transpose(1, 2).reshape(B, N_q, -1)  # (B, N_q, inner_dim)
        return self.to_out(out)


class FeedForward(nn.Module):
    """Feed-forward network for Perceiver."""
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverBlock(nn.Module):
    """Single Perceiver block: cross-attention + self-attention + FFN."""
    def __init__(
        self,
        latent_dim: int,
        cross_heads: int,
        cross_dim_head: int,
        latent_heads: int,
        latent_dim_head: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Cross-attention (latents attend to data)
        self.cross_norm_latent = nn.LayerNorm(latent_dim)
        self.cross_norm_data = nn.LayerNorm(latent_dim)
        self.cross_attn = Attention(
            dim=latent_dim,
            heads=cross_heads,
            dim_head=cross_dim_head,
            dropout=dropout,
        )

        # Self-attention on latents
        self.self_norm = nn.LayerNorm(latent_dim)
        self.self_attn = Attention(
            dim=latent_dim,
            heads=latent_heads,
            dim_head=latent_dim_head,
            dropout=dropout,
        )

        # FFN
        self.ff_norm = nn.LayerNorm(latent_dim)
        self.ff = FeedForward(latent_dim, dropout=dropout)

    def forward(
        self,
        latents: torch.Tensor,
        data: torch.Tensor,
        data_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        latents: (B, N_latent, D)
        data: (B, N_data, D)
        data_mask: (B, N_data) bool
        """
        # Cross-attention: latents attend to data
        latents = latents + self.cross_attn(
            self.cross_norm_latent(latents),
            self.cross_norm_data(data),
            mask=data_mask,
        )

        # Self-attention on latents
        latents = latents + self.self_attn(
            self.self_norm(latents),
            self.self_norm(latents),
        )

        # FFN
        latents = latents + self.ff(self.ff_norm(latents))

        return latents


class PerceiverIO(nn.Module):
    """
    Perceiver IO architecture.

    Takes variable-length data, processes through latent bottleneck,
    and outputs query-specific results.

    Args:
        depth: Number of Perceiver blocks
        dim: Data dimension (input projection target)
        queries_dim: Query input dimension
        logits_dim: Output dimension (None = same as latent_dim)
        num_latents: Number of latent vectors
        latent_dim: Latent vector dimension
        cross_heads: Number of heads for cross-attention
        latent_heads: Number of heads for self-attention
        cross_dim_head: Dimension per head in cross-attention
        latent_dim_head: Dimension per head in self-attention
        weight_tie_layers: Whether to share weights across layers
        decoder_ff: Whether to include FFN in decoder
    """
    def __init__(
        self,
        depth: int,
        dim: int,
        queries_dim: int,
        logits_dim: Optional[int] = None,
        num_latents: int = 256,
        latent_dim: int = 512,
        cross_heads: int = 1,
        latent_heads: int = 8,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        weight_tie_layers: bool = False,
        decoder_ff: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents

        # Learnable latent vectors
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)

        # Data projection (if needed)
        self.data_proj = nn.Linear(dim, latent_dim) if dim != latent_dim else nn.Identity()

        # Encoder blocks
        if weight_tie_layers:
            encoder_block = PerceiverBlock(
                latent_dim=latent_dim,
                cross_heads=cross_heads,
                cross_dim_head=cross_dim_head,
                latent_heads=latent_heads,
                latent_dim_head=latent_dim_head,
            )
            self.encoder_blocks = nn.ModuleList([encoder_block] * depth)
        else:
            self.encoder_blocks = nn.ModuleList([
                PerceiverBlock(
                    latent_dim=latent_dim,
                    cross_heads=cross_heads,
                    cross_dim_head=cross_dim_head,
                    latent_heads=latent_heads,
                    latent_dim_head=latent_dim_head,
                )
                for _ in range(depth)
            ])

        # Decoder cross-attention (queries attend to latents)
        self.decoder_cross_norm_q = nn.LayerNorm(latent_dim)
        self.decoder_cross_norm_kv = nn.LayerNorm(latent_dim)
        self.decoder_cross_attn = Attention(
            dim=latent_dim,
            heads=cross_heads,
            dim_head=cross_dim_head,
        )

        # Query projection
        self.query_proj = nn.Linear(queries_dim, latent_dim) if queries_dim != latent_dim else nn.Identity()

        # Decoder FFN (optional)
        self.decoder_ff = None
        if decoder_ff:
            self.decoder_ff_norm = nn.LayerNorm(latent_dim)
            self.decoder_ff = FeedForward(latent_dim)

        # Output projection
        output_dim = logits_dim if logits_dim is not None else latent_dim
        self.output_proj = nn.Linear(latent_dim, output_dim) if output_dim != latent_dim else nn.Identity()

    def forward(
        self,
        data: torch.Tensor,
        queries: torch.Tensor,
        data_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        data: (B, N_data, dim) - input data tokens
        queries: (B, N_queries, queries_dim) - query vectors
        data_mask: (B, N_data) bool - True = valid data token

        returns: (B, N_queries, logits_dim or latent_dim)
        """
        B = data.shape[0]

        # Project data
        data = self.data_proj(data)

        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, num_latents, latent_dim)

        # Encoder: process data through latent bottleneck
        for block in self.encoder_blocks:
            latents = block(latents, data, data_mask)

        # Project queries
        queries = self.query_proj(queries)  # (B, N_queries, latent_dim)

        # Decoder: queries attend to latents
        out = queries + self.decoder_cross_attn(
            self.decoder_cross_norm_q(queries),
            self.decoder_cross_norm_kv(latents),
        )

        # Optional decoder FFN
        if self.decoder_ff is not None:
            out = out + self.decoder_ff(self.decoder_ff_norm(out))

        # Output projection
        out = self.output_proj(out)

        return out
