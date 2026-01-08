# src/models/layers/sfc_encoder.py
from __future__ import annotations

import math
from functools import lru_cache
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


# ============================================================================
# SPACE-FILLING CURVE IMPLEMENTATIONS
# ============================================================================

def xy_to_zorder(x: int, y: int, bits: int = 5) -> int:
    """
    Convert (x, y) to Z-order (Morton code) index.

    Z-order interleaves the bits of x and y:
      z = y_{bits-1} x_{bits-1} ... y1 x1 y0 x0
    """
    z = 0
    for i in range(bits):
        z |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1))
    return z


def zorder_to_xy(z: int, bits: int = 5) -> Tuple[int, int]:
    """Convert Z-order index back to (x, y)."""
    x = 0
    y = 0
    for i in range(bits):
        x |= ((z >> (2 * i)) & 1) << i
        y |= ((z >> (2 * i + 1)) & 1) << i
    return x, y


def xy_to_hilbert(x: int, y: int, order: int = 5) -> int:
    """
    Convert (x, y) to Hilbert curve index.

    Hilbert has better locality than Z-order:
    consecutive indices are always adjacent in 2D space.
    """
    n = 1 << order
    d = 0
    s = n >> 1
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)

        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x

        s >>= 1
    return d


def hilbert_to_xy(d: int, order: int = 5) -> Tuple[int, int]:
    """Convert Hilbert curve index back to (x, y)."""
    n = 1 << order
    x = y = 0
    s = 1
    while s < n:
        rx = 1 & (d // 2)
        ry = 1 & (d ^ rx)

        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x

        x += s * rx
        y += s * ry
        d //= 4
        s *= 2
    return x, y


# ============================================================================
# Helpers: coords + SFC order precompute
# ============================================================================

def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@lru_cache(maxsize=64)
def _precompute_sfc_order_and_coords(
    size: int,
    curve: Literal["hilbert", "zorder"],
    coord_mode: Literal["neg1to1", "zero1"] = "neg1to1",
) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Returns:
      order_flat:   (HW,) long, flat indices (y*W+x) sorted by SFC index.
      coords_flat:  (HW,2) float, coords aligned to FLAT indexing (not ordered).
      sfc_pos_flat: (HW,1) float, normalized SFC position aligned to FLAT indexing.
    """
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")
    if not _is_power_of_two(size):
        raise ValueError(
            f"SFC tokenizer assumes square power-of-two size. Got size={size}."
        )

    bits = int(math.log2(size))

    # (sfc_idx, flat_idx)
    pairs = []
    for y in range(size):
        for x in range(size):
            if curve == "hilbert":
                sfc = xy_to_hilbert(x, y, order=bits)
            elif curve == "zorder":
                sfc = xy_to_zorder(x, y, bits=bits)
            else:
                raise ValueError(f"Unknown curve={curve}")
            flat = y * size + x
            pairs.append((sfc, flat))

    pairs.sort(key=lambda t: t[0])
    order_flat = torch.tensor([flat for _, flat in pairs], dtype=torch.long)  # (HW,)

    # flat-aligned coords
    xs = torch.arange(size, dtype=torch.float32)
    ys = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)

    if coord_mode == "neg1to1":
        x_norm = (xx / (size - 1)) * 2.0 - 1.0
        y_norm = (yy / (size - 1)) * 2.0 - 1.0
    elif coord_mode == "zero1":
        x_norm = xx / (size - 1)
        y_norm = yy / (size - 1)
    else:
        raise ValueError(f"Unknown coord_mode={coord_mode}")

    coords_flat = torch.stack([x_norm, y_norm], dim=-1).reshape(-1, 2)  # (HW,2)

    # flat-aligned normalized SFC position in [0,1]
    inv = torch.empty_like(order_flat)
    inv[order_flat] = torch.arange(order_flat.numel(), dtype=torch.long)
    sfc_pos_flat = inv.to(torch.float32).unsqueeze(-1) / max(order_flat.numel() - 1, 1)  # (HW,1)

    return order_flat, coords_flat, sfc_pos_flat


def _fourier_features_fixed(coords: torch.Tensor, n_bands: int) -> torch.Tensor:
    """
    Fixed Fourier features (non-learnable).
    coords: (N,2) in [-1,1] or [0,1]
    returns: (N, 4*n_bands) = [sin/cos for x bands, sin/cos for y bands]
    Computed in float32 for stability, then cast back.
    """
    if n_bands <= 0:
        raise ValueError(f"n_bands must be > 0, got {n_bands}")

    orig_dtype = coords.dtype
    coords_f = coords.to(dtype=torch.float32)

    freqs = (2.0 ** torch.arange(n_bands, device=coords.device, dtype=torch.float32)) * math.pi  # (B,)
    x = coords_f[:, 0:1]  # (N,1)
    y = coords_f[:, 1:2]  # (N,1)

    xw = x * freqs[None, :]  # (N,B)
    yw = y * freqs[None, :]  # (N,B)

    feat = torch.cat([torch.sin(xw), torch.cos(xw), torch.sin(yw), torch.cos(yw)], dim=-1)  # (N,4B)
    return feat.to(dtype=orig_dtype)


class LearnableFourierMLP(nn.Module):
    """
    coords -> fixed Fourier (sin/cos) -> learnable MLP projection.

    Output dim is typically set to hidden_size so coords live in the same
    representational space as the rest of the model (Perceiver-like).
    """
    def __init__(
        self,
        out_features: int,
        n_bands: int = 16,
        hidden_dim: Optional[int] = None,
        mlp_layers: int = 2,
    ):
        super().__init__()
        if n_bands <= 0:
            raise ValueError(f"n_bands must be > 0, got {n_bands}")
        if out_features <= 0:
            raise ValueError(f"out_features must be > 0, got {out_features}")
        if mlp_layers <= 0:
            raise ValueError(f"mlp_layers must be > 0, got {mlp_layers}")

        self.n_bands = int(n_bands)
        self.out_features = int(out_features)

        freqs = (2.0 ** torch.arange(self.n_bands, dtype=torch.float32)) * math.pi  # (n_bands,)
        self.register_buffer("freqs", freqs, persistent=False)

        fourier_dim = 4 * self.n_bands  # for 2D coords: (sin,cos)x + (sin,cos)y
        if hidden_dim is None:
            hidden_dim = max(self.out_features, fourier_dim)

        layers: list[nn.Module] = []
        if mlp_layers == 1:
            layers.append(nn.Linear(fourier_dim, self.out_features, bias=True))
        else:
            layers.append(nn.Linear(fourier_dim, hidden_dim, bias=True))
            layers.append(nn.GELU())
            for _ in range(mlp_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, self.out_features, bias=True))

        self.mlp = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (N,2)
        returns: (N,out_features)
        """
        orig_dtype = coords.dtype
        coords_f = coords.to(dtype=torch.float32)

        # (N,2) -> (N,2,n_bands)
        proj = coords_f.unsqueeze(-1) * self.freqs.view(1, 1, -1)
        # (N,2,n_bands) -> (N,2,2*n_bands) -> (N,4*n_bands)
        fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1).reshape(coords.shape[0], -1)

        out = self.mlp(fourier)
        return out.to(dtype=orig_dtype)


# ============================================================================
# SFCTokenizer: "ViT-style" grouping via CONCATENATION (no mean pool)
# ============================================================================

class SFCTokenizer(nn.Module):
    """
    Turn sparse pixels into tokens by:
      1) ordering ALL pixels by a space-filling curve (Hilbert/Z-order),
      2) selecting only cond_mask==1 pixels in that order,
      3) grouping consecutive points in groups of g,
      4) flattening/concatenating the group's per-point features into one vector,
      5) projecting that vector to hidden_size.

    Inputs:
      x:         (B,C,H,W)
      cond_mask: (B,1,H,W) or (B,H,W) or None
                - if None, we treat everything as "observed" (dense; mainly for debugging)

    Outputs:
      tokens:     (B,T,D)
      token_mask: (B,T) bool, True=real token, False=pad
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        curve: Literal["hilbert", "zorder"] = "hilbert",
        group_size: int = 8,
        coord_mode: Literal["neg1to1", "zero1"] = "neg1to1",
        # positional-ish extras
        add_coords: bool = True,
        coord_bands: int = 16,
        coord_embed: Literal["mlp", "fixed"] = "mlp",
        coord_mlp_hidden: Optional[int] = None,
        coord_mlp_layers: int = 2,
        add_sfc_position: bool = False,      # per-point scalar in [0,1]
        # small MLP after token projection (optional)
        token_mlp: bool = False,
    ):
        super().__init__()
        if group_size <= 0:
            raise ValueError(f"group_size must be > 0, got {group_size}")

        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.curve = curve
        self.group_size = int(group_size)
        self.coord_mode = coord_mode

        self.add_coords = bool(add_coords)
        self.coord_bands = int(coord_bands)
        self.coord_embed_mode = coord_embed
        self.add_sfc_position = bool(add_sfc_position)

        # ---- coord embed (THIS is the upgrade) ----
        # If coord_embed="mlp", coord features are projected to hidden_size.
        self.coord_embed = None
        coord_feat_dim = 0
        if self.add_coords:
            if self.coord_embed_mode == "mlp":
                self.coord_embed = LearnableFourierMLP(
                    out_features=self.hidden_size,
                    n_bands=self.coord_bands,
                    hidden_dim=coord_mlp_hidden,
                    mlp_layers=coord_mlp_layers,
                )
                coord_feat_dim = self.hidden_size
            elif self.coord_embed_mode == "fixed":
                coord_feat_dim = 4 * self.coord_bands
            else:
                raise ValueError(f"Unknown coord_embed={self.coord_embed_mode!r}")

        # token projection input dim
        per_point_dim = self.in_channels + coord_feat_dim + (1 if self.add_sfc_position else 0)
        token_in_dim = self.group_size * per_point_dim
        self.token_proj = nn.Linear(token_in_dim, self.hidden_size)

        self.token_mlp = (
            nn.Sequential(
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
            )
            if token_mlp
            else nn.Identity()
        )
        self.norm = nn.LayerNorm(self.hidden_size)

    @staticmethod
    def _ensure_mask_shape(
        cond_mask: torch.Tensor, B: int, H: int, W: int
    ) -> torch.Tensor:
        if cond_mask.dim() == 4:
            if cond_mask.shape[1] != 1:
                raise ValueError(f"cond_mask must have 1 channel. Got {tuple(cond_mask.shape)}")
            m = cond_mask[:, 0]
        elif cond_mask.dim() == 3:
            m = cond_mask
        else:
            raise ValueError(f"cond_mask must be (B,1,H,W) or (B,H,W). Got {tuple(cond_mask.shape)}")

        if m.shape != (B, H, W):
            raise ValueError(f"cond_mask shape mismatch. Expected {(B,H,W)}, got {tuple(m.shape)}")
        return m

    def _get_precomputed(self, H: int, W: int, device: torch.device):
        if H != W:
            raise ValueError(f"SFCTokenizer assumes square inputs. Got H={H}, W={W}.")
        order_flat, coords_flat, sfc_pos_flat = _precompute_sfc_order_and_coords(
            H, self.curve, self.coord_mode
        )
        return order_flat.to(device), coords_flat.to(device), sfc_pos_flat.to(device)

    def forward(
        self,
        x: torch.Tensor,
        cond_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B,C,H,W)
        cond_mask: (B,1,H,W) or (B,H,W) or None

        returns:
          tokens: (B,T,D)
          token_mask: (B,T) bool
        """
        if x.dim() != 4:
            raise ValueError(f"x must be (B,C,H,W). Got {tuple(x.shape)}")
        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"in_channels mismatch: expected {self.in_channels}, got {C}")

        device = x.device
        dtype = x.dtype

        order_flat, coords_flat, sfc_pos_flat = self._get_precomputed(H, W, device)

        HW = H * W
        x_flat = x.permute(0, 2, 3, 1).reshape(B, HW, C)  # (B,HW,C)

        if cond_mask is None:
            m_flat = torch.ones(B, HW, device=device, dtype=torch.bool)
        else:
            m = self._ensure_mask_shape(cond_mask, B, H, W).to(device)
            m_flat = (m > 0.5).reshape(B, HW)  # (B,HW)

        # order everything by SFC once
        x_ord = x_flat[:, order_flat, :]  # (B,HW,C)
        m_ord = m_flat[:, order_flat]     # (B,HW)

        # ordered coords/sfcpos for optional extras
        coords_ord = coords_flat[order_flat, :].to(device=device, dtype=dtype)   # (HW,2)
        sfcpos_ord = sfc_pos_flat[order_flat, :].to(device=device, dtype=dtype) # (HW,1)

        tokens_list = []
        masks_list = []

        for b in range(B):
            idx = torch.nonzero(m_ord[b], as_tuple=False).squeeze(1)  # (K,)
            if idx.numel() == 0:
                # produce 1 padded token so shapes are valid
                T = 1
                tok = x.new_zeros((T, self.hidden_size))
                tmask = torch.zeros((T,), dtype=torch.bool, device=device)
                tokens_list.append(tok)
                masks_list.append(tmask)
                continue

            vals = x_ord[b, idx, :].to(dtype=dtype)  # (K,C)

            feats = [vals]

            if self.add_coords:
                xy = coords_ord[idx, :]  # (K,2)
                if self.coord_embed is None:
                    feats.append(_fourier_features_fixed(xy, self.coord_bands))  # (K,4*bands)
                else:
                    feats.append(self.coord_embed(xy))  # (K,hidden_size)

            if self.add_sfc_position:
                feats.append(sfcpos_ord[idx, :])  # (K,1)

            p = torch.cat(feats, dim=-1)  # (K, per_point_dim)
            K = p.shape[0]

            # pad K up to multiple of group_size
            total = int(math.ceil(K / self.group_size) * self.group_size)
            pad = total - K
            if pad > 0:
                p = torch.cat([p, p.new_zeros((pad, p.shape[-1]))], dim=0)  # (total, per_point_dim)

            # group and FLATTEN (concat)
            T = total // self.group_size
            p = p.view(T, self.group_size, -1)  # (T,g,per_point_dim)
            p = p.reshape(T, -1)                # (T, g*per_point_dim)

            tok = self.token_proj(p)            # (T,D)
            tok = self.token_mlp(tok)
            tok = self.norm(tok)

            real_T = int(math.ceil(K / self.group_size))
            tmask = torch.zeros((T,), dtype=torch.bool, device=device)
            tmask[:real_T] = True

            tok = tok * tmask.unsqueeze(-1).to(tok.dtype)

            tokens_list.append(tok)
            masks_list.append(tmask)

        # pad across batch
        T_max = max(t.shape[0] for t in tokens_list)
        tokens = x.new_zeros((B, T_max, self.hidden_size))
        token_mask = torch.zeros((B, T_max), dtype=torch.bool, device=device)

        for b in range(B):
            t = tokens_list[b]
            m = masks_list[b]
            tokens[b, : t.shape[0]] = t
            token_mask[b, : m.shape[0]] = m

        return tokens, token_mask


# ============================================================================
# Cross-attend: patch queries <- SFC tokens
# ============================================================================

class _CrossBlock(nn.Module):
    """
    One block of:
      queries = queries + CrossAttn(LN(queries), LN(tokens))
      queries = queries + MLP(LN(queries))
    """
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(proj_drop)

        self.mlp_norm = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )
        self.drop2 = nn.Dropout(proj_drop)

    def forward(
        self,
        queries: torch.Tensor,               # (B,L,D)
        tokens: torch.Tensor,                # (B,T,D)
        token_mask: Optional[torch.Tensor],  # (B,T) True=real, False=pad
    ) -> torch.Tensor:
        if token_mask is not None:
            key_padding_mask = ~token_mask  # True = ignore
        else:
            key_padding_mask = None

        q = self.q_norm(queries)
        kv = self.kv_norm(tokens)

        attn_out, _ = self.attn(
            q, kv, kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        queries = queries + self.drop1(attn_out)

        h = self.mlp_norm(queries)
        queries = queries + self.drop2(self.mlp(h))
        return queries


class SFCQueryCrossEncoder(nn.Module):
    """
    Stack of cross-attention blocks:
      queries (patch-grid tokens) attend to SFC tokens.

    Inputs:
      queries: (B,L,D)
      tokens: (B,T,D)
      token_mask: (B,T) bool
    Output:
      (B,L,D)
    """
    def __init__(
        self,
        dim: int,
        heads: int,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            _CrossBlock(
                dim=dim,
                heads=heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            for _ in range(int(depth))
        ])

    def forward(
        self,
        queries: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for blk in self.blocks:
            queries = blk(queries, tokens, token_mask)
        return queries
