# src/models/layers/perceiver_io.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else d


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module, context_dim: Optional[int] = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context) and "context" in kwargs and kwargs["context"] is not None:
            kwargs["context"] = self.norm_context(kwargs["context"])
        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner * 2),
            GEGLU(),
            nn.Linear(inner, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head attention with optional context and optional key padding mask.
    mask: (B, Nk) boolean, True = keep, False = mask out
    """
    def __init__(self, query_dim: int, context_dim: Optional[int] = None, heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_kv = nn.Linear(context_dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, query_dim, bias=True)

    def forward(self, x, context=None, mask=None):
        """
        x: (B, Nq, Dq)
        context: (B, Nk, Dc) or None (defaults to x)
        mask: (B, Nk) boolean (True = keep)
        """
        b, nq, _ = x.shape
        h = self.heads

        context = default(context, x)
        _, nk, _ = context.shape

        q = self.to_q(x)  # (B, Nq, h*dh)
        kv = self.to_kv(context)  # (B, Nk, 2*h*dh)
        k, v = kv.chunk(2, dim=-1)

        # reshape to heads
        q = q.view(b, nq, h, self.dim_head).transpose(1, 2)  # (B, h, Nq, dh)
        k = k.view(b, nk, h, self.dim_head).transpose(1, 2)  # (B, h, Nk, dh)
        v = v.view(b, nk, h, self.dim_head).transpose(1, 2)  # (B, h, Nk, dh)

        # attention
        sim = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, h, Nq, Nk)

        if exists(mask):
            # mask: (B, Nk) -> (B, 1, 1, Nk)
            mask_ = mask[:, None, None, :].to(torch.bool)
            sim = sim.masked_fill(~mask_, torch.finfo(sim.dtype).min)

        attn = sim.softmax(dim=-1)
        out = torch.matmul(attn, v)  # (B, h, Nq, dh)

        out = out.transpose(1, 2).contiguous().view(b, nq, h * self.dim_head)  # (B, Nq, h*dh)
        return self.to_out(out)


class PerceiverIO(nn.Module):
    """
    PerceiverIO:
      - Learnable latents cross-attend to data tokens
      - Latents self-attend for depth steps
      - Queries cross-attend to latents to produce per-query outputs
    """
    def __init__(
        self,
        *,
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
        self.latent_dim = latent_dim

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=dim),
            PreNorm(latent_dim, FeedForward(latent_dim)),
        ])

        def make_latent_attn():
            return PreNorm(latent_dim, Attention(latent_dim, latent_dim, heads=latent_heads, dim_head=latent_dim_head))

        def make_latent_ff():
            return PreNorm(latent_dim, FeedForward(latent_dim))

        if weight_tie_layers:
            attn_block = make_latent_attn()
            ff_block = make_latent_ff()
            self.layers = nn.ModuleList([nn.ModuleList([attn_block, ff_block]) for _ in range(depth)])
        else:
            self.layers = nn.ModuleList([nn.ModuleList([make_latent_attn(), make_latent_ff()]) for _ in range(depth)])

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(queries_dim, latent_dim, heads=cross_heads, dim_head=cross_dim_head),
            context_dim=latent_dim,
        )
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    def forward(self, data, mask=None, queries=None):
        """
        data: (B, Nd, dim)
        mask: (B, Nd) boolean (True keep)
        queries: (B, Nq, queries_dim) or (Nq, queries_dim)
        """
        b = data.shape[0]

        x = self.latents.unsqueeze(0).expand(b, -1, -1)  # (B, Nl, latent_dim)

        cross_attn, cross_ff = self.cross_attend_blocks
        x = cross_attn(x, context=data, mask=mask) + x
        x = cross_ff(x) + x

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        if queries is None:
            return x

        if queries.ndim == 2:
            queries = queries.unsqueeze(0).expand(b, -1, -1)

        out = self.decoder_cross_attn(queries, context=x)  # (B, Nq, queries_dim)
        if exists(self.decoder_ff):
            out = out + self.decoder_ff(out)
        return self.to_logits(out)


class FourierFeatures(nn.Module):
    """
    coords: (B, N, D) -> (B, N, out_features)
    """
    def __init__(self, in_features: int, out_features: int, n_bands: int = 16):
        super().__init__()
        self.in_features = in_features
        self.n_bands = n_bands

        freqs = (2 ** torch.arange(n_bands, dtype=torch.float32)) * torch.pi
        self.register_buffer("freqs", freqs, persistent=False)

        fourier_dim = in_features * n_bands * 2
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B,N,D)
        b, n, d = coords.shape
        assert d == self.in_features, f"Expected coords last dim={self.in_features}, got {d}"

        # (B,N,D,nb)
        proj = coords.unsqueeze(-1) * self.freqs
        feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (B,N,D,2nb)
        feats = feats.reshape(b, n, d * self.n_bands * 2)
        return self.mlp(feats)
