"""
PixNerDiT Class-Conditional Heavy Decoder for Super-Resolution

This model combines class-conditional generation with the heavy decoder architecture
that enables arbitrary resolution output via decoder_patch_scaling.

Key differences from T2I heavy decoder:
- Uses LabelEmbedder instead of text embeddings
- No TextRefineBlocks (class labels don't need refinement)
- Supports num_classes for CIFAR-10 (10) or ImageNet (1000)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
from typing import Optional

from src.models.layers.attention_op import attention
from src.models.layers.rope import apply_rotary_emb, precompute_freqs_cis_ex2d as precompute_freqs_cis_2d
from src.models.layers.time_embed import TimestepEmbedder
from src.models.layers.patch_embed import Embed
from src.models.layers.swiglu import SwiGLU as FeedForward
from src.models.layers.rmsnorm import RMSNorm as Norm
from src.models.layers.perceiver import PerceiverIO, FourierFeatures

from src.models.layers.sfc_encoder import SFCTokenizer, SFCQueryCrossEncoder


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations."""
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        # +1 for null/unconditional class
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        return self.embedding_table(labels)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = Norm(self.head_dim)
        self.k_norm = Norm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, pos) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q.contiguous())
        k = self.k_norm(k.contiguous())
        q, k = apply_rotary_emb(q, k, freqs_cis=pos)

        q = q.view(B, self.num_heads, -1, C // self.num_heads)
        k = k.view(B, self.num_heads, -1, C // self.num_heads).contiguous()
        v = v.view(B, self.num_heads, -1, C // self.num_heads).contiguous()

        x = attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlattenDiTBlock(nn.Module):
    """Encoder block with self-attention and adaptive layer norm."""
    def __init__(self, hidden_size, groups, mlp_ratio=4):
        super().__init__()
        self.norm1 = Norm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = Norm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c, pos):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), pos)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class NerfEmbedder(nn.Module):
    """Embeds pixel positions using continuous coordinates (NeRF-style)."""
    def __init__(self, in_channels, hidden_size_input, max_freqs):
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input
        self.embedder = nn.Sequential(
            nn.Linear(in_channels + max_freqs ** 2, hidden_size_input, bias=True),
        )

    @lru_cache
    def fetch_pos(self, patch_size_h, patch_size_w, device, dtype):
        pos = precompute_freqs_cis_2d(
            self.max_freqs ** 2 * 2,
            patch_size_h,
            patch_size_w,
            scale=(16 / patch_size_h, 16 / patch_size_w),
        )
        pos = pos[None, :, :].to(device=device, dtype=dtype)
        return pos

    def forward(self, inputs, patch_size_h, patch_size_w):
        B, _, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype
        dct = self.fetch_pos(patch_size_h, patch_size_w, device, dtype)
        dct = dct.repeat(B, 1, 1)
        inputs = torch.cat([inputs, dct], dim=-1)
        inputs = self.embedder(inputs)
        return inputs


class NerfBlock(nn.Module):
    """Decoder block with hypernetwork-generated MLP weights."""
    def __init__(self, hidden_size_s, hidden_size_x, mlp_ratio=4):
        super().__init__()
        self.param_generator1 = nn.Sequential(
            nn.Linear(hidden_size_s, 2 * hidden_size_x ** 2 * mlp_ratio, bias=True),
        )
        self.norm = Norm(hidden_size_x, eps=1e-6)
        self.mlp_ratio = mlp_ratio

    def forward(self, x, s):
        batch_size, _, hidden_size_x = x.shape
        mlp_params1 = self.param_generator1(s)
        fc1_param1, fc2_param1 = mlp_params1.chunk(2, dim=-1)

        fc1_param1 = fc1_param1.view(batch_size, hidden_size_x, hidden_size_x * self.mlp_ratio)
        fc2_param1 = fc2_param1.view(batch_size, hidden_size_x * self.mlp_ratio, hidden_size_x)

        normalized_fc1_param1 = torch.nn.functional.normalize(fc1_param1, dim=-2)

        res_x = x
        x = self.norm(x)
        x = torch.bmm(x, normalized_fc1_param1)
        x = torch.nn.functional.silu(x)
        x = torch.bmm(x, fc2_param1)
        x = x + res_x
        return x


class NerfFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        return self.linear(x)


# ============================================================
# NEW: Self-attention over sparse SFC tokens (B,T,D)
# ============================================================
class SFCTransformerBlock(nn.Module):
    """
    Self-attention block for sparse SFC tokens (B,T,D) with padding mask.
    Uses torch MultiheadAttention (batch_first).
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = Norm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(proj_drop)

        self.norm2 = Norm(dim, eps=1e-6)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim, mlp_hidden)
        self.drop2 = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, token_mask: Optional[torch.Tensor] = None):
        """
        x: (B,T,D)
        token_mask: (B,T) bool, True=real token, False=pad
        """
        if token_mask is not None:
            # MultiheadAttention expects True for positions to IGNORE
            key_padding_mask = ~token_mask
        else:
            key_padding_mask = None

        h = self.norm1(x)
        attn_out, _ = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop1(attn_out)

        h = self.norm2(x)
        x = x + self.drop2(self.mlp(h))

        # keep padded tokens clean/zeroed
        if token_mask is not None:
            x = x * token_mask.unsqueeze(-1).to(x.dtype)

        return x


class PixNerDiT(nn.Module):
    """
    Class-Conditional PixNerDiT with Heavy Decoder for Super-Resolution.

    Encoder options:
      - encoder_type="grid": original patchified DiT encoder
      - encoder_type="perceiver": point-token PerceiverIO + (optional) grid-patch tokens,
        then run the SAME DiT blocks (AdaLN+RoPE) as the grid encoder on the Perceiver outputs.
      - encoder_type="sfc": space-filling-curve tokenizer (Hilbert/Z-order) grouped tokens,
        (OPTIONAL) self-attn over tokens, then cross-attn into patch queries (no latent bottleneck),
        then SAME DiT blocks.
    """
    def __init__(
        self,
        in_channels=3,
        num_groups=12,
        hidden_size=768,
        decoder_hidden_size=64,
        num_encoder_blocks=12,
        num_decoder_blocks=2,
        patch_size=2,
        num_classes=10,
        weight_path=None,
        load_ema=False,
        # encoder switch
        encoder_type: str = "grid",          # "grid" | "perceiver" | "sfc"
        perceiver_num_latents: int = 256,
        perceiver_cross_heads: int = 1,
        perceiver_depth: int = 2,
        include_patch_tokens_in_perceiver: bool = False,
        # SFC knobs
        sfc_curve: str = "hilbert",          # "hilbert" | "zorder"
        sfc_group_size: int = 8,             # g
        sfc_cross_depth: int = 2,            # cross-attn blocks: queries <- tokens
        # NEW: token self-attn depth before cross-attn
        sfc_self_depth: int = 2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.decoder_hidden_size = decoder_hidden_size
        self.num_encoder_blocks = num_encoder_blocks
        self.num_decoder_blocks = num_decoder_blocks
        self.num_blocks = self.num_encoder_blocks + self.num_decoder_blocks
        self.num_classes = num_classes

        self.encoder_type = encoder_type
        self.include_patch_tokens_in_perceiver = bool(include_patch_tokens_in_perceiver)

        # Decoder patch scaling for super-resolution
        self.decoder_patch_scaling_h = 1.0
        self.decoder_patch_scaling_w = 1.0
        self.patch_size = patch_size

        # Embedders
        self.s_embedder = Embed(in_channels * patch_size ** 2, hidden_size, bias=True)

        # For mask-aware sparse conditioning (patch-level hint density)
        self.patch_hint_proj = torch.nn.Linear(1, hidden_size, bias=True)

        self.x_embedder = NerfEmbedder(in_channels, decoder_hidden_size, max_freqs=8)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # Perceiver pieces (for encoder_type="perceiver")
        self.point_hint_embed = nn.Embedding(2, hidden_size)  # 0/1 flag
        self.point_value_proj = nn.Linear(in_channels, hidden_size, bias=True)
        self.point_coord_embed = FourierFeatures(in_features=2, out_features=hidden_size, n_bands=16)
        self.query_coord_embed = FourierFeatures(in_features=2, out_features=hidden_size, n_bands=16)

        head_dim = max(16, hidden_size // max(1, num_groups))
        self.perceiver = PerceiverIO(
            depth=int(perceiver_depth),
            dim=hidden_size,
            queries_dim=hidden_size,
            logits_dim=None,
            num_latents=perceiver_num_latents,
            latent_dim=hidden_size,
            cross_heads=perceiver_cross_heads,
            latent_heads=num_groups,
            cross_dim_head=head_dim,
            latent_dim_head=head_dim,
            weight_tie_layers=False,
            decoder_ff=True,
        )

        # SFC encoder (for encoder_type="sfc")
        self.sfc_tokenizer = SFCTokenizer(
            in_channels=in_channels,
            hidden_size=hidden_size,
            curve=sfc_curve,
            group_size=sfc_group_size,
            coord_bands=16,
            add_sfc_position=True,
        )
        self.sfc_cross = SFCQueryCrossEncoder(
            dim=hidden_size,
            heads=num_groups,
            depth=int(sfc_cross_depth),
            mlp_ratio=4.0,
        )

        # NEW: self-attn on sparse SFC tokens (cheap since T is small)
        self.sfc_self = nn.ModuleList([
            SFCTransformerBlock(
                dim=hidden_size,
                num_heads=num_groups,
                mlp_ratio=4.0,
                attn_drop=0.0,
                proj_drop=0.0,
            )
            for _ in range(int(sfc_self_depth))
        ])

        self.final_layer = NerfFinalLayer(decoder_hidden_size, in_channels)

        # Encoder DiT blocks
        encoder_blocks = nn.ModuleList([
            FlattenDiTBlock(self.hidden_size, self.num_groups) for _ in range(self.num_encoder_blocks)
        ])

        # Decoder NerfBlocks
        decoder_blocks = nn.ModuleList([
            NerfBlock(self.hidden_size, self.decoder_hidden_size, mlp_ratio=2) for _ in range(self.num_decoder_blocks)
        ])

        self.blocks = nn.ModuleList(list(encoder_blocks) + list(decoder_blocks))

        self.initialize_weights()
        self.precompute_pos = dict()
        self.weight_path = weight_path
        self.load_ema = load_ema

        # coord caches
        self._coord_cache = {}
        self._patch_query_cache = {}

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
        self.precompute_pos[(height, width)] = pos
        return pos

    def _fetch_coord_grid(self, h: int, w: int, device, dtype):
        key = (h, w, str(device), str(dtype))
        if key in self._coord_cache:
            return self._coord_cache[key]

        ys = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (h,w)
        coords = torch.stack([xx, yy], dim=-1).view(1, h * w, 2)  # (1,N,2)
        self._coord_cache[key] = coords
        return coords

    def _fetch_patch_query_coords(self, encoder_h: int, encoder_w: int, device, dtype):
        ph = encoder_h // self.patch_size
        pw = encoder_w // self.patch_size
        key = (ph, pw, encoder_h, encoder_w, self.patch_size, str(device), str(dtype))
        if key in self._patch_query_cache:
            return self._patch_query_cache[key]

        ys = (torch.arange(ph, device=device, dtype=dtype) + 0.5) * self.patch_size
        xs = (torch.arange(pw, device=device, dtype=dtype) + 0.5) * self.patch_size

        ys = (ys / max(1.0, float(encoder_h))) * 2.0 - 1.0
        xs = (xs / max(1.0, float(encoder_w))) * 2.0 - 1.0

        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (ph,pw)
        coords = torch.stack([xx, yy], dim=-1).view(1, ph * pw, 2)  # (1,L,2)
        self._patch_query_cache[key] = coords
        return coords

    def initialize_weights(self):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.xavier_uniform_(self.point_value_proj.weight)
        nn.init.constant_(self.point_value_proj.bias, 0.0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        nn.init.zeros_(self.patch_hint_proj.weight)
        nn.init.zeros_(self.patch_hint_proj.bias)

    def _build_point_tokens(self, x_enc, cond_mask_enc):
        """
        Build Perceiver "data tokens" from ALL conditioning pixels.

        - If cond_mask_enc is provided: tokens correspond exactly to cond pixels (mask==1).
          (No random subsampling. No max_points.)
        - If cond_mask_enc is None: fall back to dense tokens (all pixels).
        - Pads per-batch to max number of cond pixels and returns a boolean mask.
        """
        B, C, He, We = x_enc.shape
        device, dtype = x_enc.device, x_enc.dtype
        N = He * We

        vals = x_enc.permute(0, 2, 3, 1).reshape(B, N, C)  # (B,N,C)
        coords = self._fetch_coord_grid(He, We, device, dtype).expand(B, -1, -1)  # (B,N,2)

        if cond_mask_enc is None:
            # dense fallback: everything is a "hint"
            out_vals = vals
            out_coords = coords
            hint = torch.ones(B, N, device=device, dtype=torch.long)
            data_mask = torch.ones(B, N, device=device, dtype=torch.bool)
        else:
            flat = cond_mask_enc[:, 0].reshape(B, N)  # (B,N) in {0,1} (float)
            idx_list = [
                torch.nonzero(flat[b] > 0.5, as_tuple=False).squeeze(1)
                for b in range(B)
            ]
            maxM = max([int(i.numel()) for i in idx_list] + [1])  # at least 1 token

            out_vals = torch.zeros(B, maxM, C, device=device, dtype=dtype)
            out_coords = torch.zeros(B, maxM, 2, device=device, dtype=dtype)
            hint = torch.zeros(B, maxM, device=device, dtype=torch.long)
            data_mask = torch.zeros(B, maxM, device=device, dtype=torch.bool)

            for b, idx in enumerate(idx_list):
                m = int(idx.numel())
                if m == 0:
                    # edge-case: no cond pixels → provide one dummy token
                    data_mask[b, 0] = True
                    # hint stays 0, vals/coords stay 0
                else:
                    out_vals[b, :m] = vals[b, idx]
                    out_coords[b, :m] = coords[b, idx]
                    hint[b, :m] = 1
                    data_mask[b, :m] = True

        # kill value signal where hint==0 (dummy/pad tokens contribute nothing)
        out_vals = out_vals * hint.float().unsqueeze(-1)

        data = (
            self.point_value_proj(out_vals) +
            self.point_coord_embed(out_coords) +
            self.point_hint_embed(hint)
        )  # (B,M,D)

        return data, data_mask

    def forward(self, x, t, y, cond_mask=None, superres_scale: float = 1.0, **kwargs):
        """
        x: [B,C,H,W]
        t: [B]
        y: [B]
        cond_mask: [B,1,H,W] or None
        superres_scale: optional upscaling factor applied only in the decoder
                        (encoder and diffusion stay at the input resolution)
        """
        B, _, H, W = x.shape
        device, dtype = x.device, x.dtype

        encoder_h = int(H / self.decoder_patch_scaling_h)
        encoder_w = int(W / self.decoder_patch_scaling_w)
        decoder_patch_size_h = int(self.patch_size * self.decoder_patch_scaling_h * superres_scale)
        decoder_patch_size_w = int(self.patch_size * self.decoder_patch_scaling_w * superres_scale)

        # ------------------------------------------------------------
        # stride downsample for non-grid SR encoders
        # ------------------------------------------------------------
        def _int_if_close(v: float):
            vi = int(round(float(v)))
            return vi if abs(float(v) - vi) < 1e-6 else None

        sh = _int_if_close(self.decoder_patch_scaling_h)
        sw = _int_if_close(self.decoder_patch_scaling_w)

        use_stride = (
            (self.encoder_type in ("perceiver", "sfc")) and
            (sh is not None) and (sw is not None) and
            (sh >= 2) and (sw >= 2) and
            (H % sh == 0) and (W % sw == 0)
        )

        if use_stride:
            x_for_encoder = x[:, :, ::sh, ::sw]
        else:
            x_for_encoder = F.interpolate(
                x, (encoder_h, encoder_w),
                mode="bilinear",
                align_corners=False
            )

        # mask downsample to encoder grid
        cond_mask_enc = None
        if cond_mask is not None:
            if use_stride:
                cond_mask_enc = cond_mask[:, :, ::sh, ::sw].to(dtype=dtype)
            else:
                cond_mask_enc = F.interpolate(
                    cond_mask.to(dtype=dtype),
                    size=(encoder_h, encoder_w),
                    mode="nearest",
                )

        # global conditioning
        t_emb = self.t_embedder(t.view(-1)).view(B, 1, self.hidden_size)  # (B,1,D)
        y_emb = self.y_embedder(y).view(B, 1, self.hidden_size)           # (B,1,D)
        condition = torch.nn.functional.silu(t_emb + y_emb)               # (B,1,D)

        # number of patches at encoder resolution
        ph = encoder_h // self.patch_size
        pw = encoder_w // self.patch_size
        L = ph * pw

        # ------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------
        if self.encoder_type == "perceiver":
            # 1) point tokens (ALL hints)
            point_tokens, point_mask = self._build_point_tokens(
                x_for_encoder, cond_mask_enc
            )

            # 2) OPTIONAL: add grid patch tokens
            if self.include_patch_tokens_in_perceiver:
                x_for_encoder_patches = F.unfold(
                    x_for_encoder, kernel_size=self.patch_size, stride=self.patch_size
                ).transpose(1, 2)  # (B,L,C*ps^2)

                patch_tokens = self.s_embedder(x_for_encoder_patches)  # (B,L,D)

                # apply patch hint density
                if cond_mask_enc is not None:
                    m = F.unfold(
                        cond_mask_enc, kernel_size=self.patch_size, stride=self.patch_size
                    ).transpose(1, 2)  # (B,L,ps^2)
                    m = m.mean(dim=-1, keepdim=True)  # (B,L,1)
                    patch_tokens = patch_tokens + self.patch_hint_proj(m)

                patch_mask = torch.ones(B, patch_tokens.shape[1], device=device, dtype=torch.bool)

                data_tokens = torch.cat([point_tokens, patch_tokens], dim=1)  # (B,M+L,D)
                data_mask = torch.cat([point_mask, patch_mask], dim=1)        # (B,M+L)
            else:
                data_tokens = point_tokens
                data_mask = point_mask

            # add global condition to all data tokens
            data_tokens = data_tokens + condition

            # queries: one per patch
            query_coords = self._fetch_patch_query_coords(encoder_h, encoder_w, device, dtype).expand(B, -1, -1)
            queries = self.query_coord_embed(query_coords) + condition  # (B,L,D)

            # Perceiver produces per-patch embeddings
            s = self.perceiver(data_tokens, mask=data_mask, queries=queries)  # (B,L,D)

            # optional parity with grid encoder: inject patch hint density on s
            if cond_mask_enc is not None:
                m = F.unfold(
                    cond_mask_enc, kernel_size=self.patch_size, stride=self.patch_size
                ).transpose(1, 2)
                m = m.mean(dim=-1, keepdim=True)  # (B,L,1)
                s = s + self.patch_hint_proj(m)

            # run the same AdaLN+RoPE DiT blocks as the grid encoder
            xpos = self.fetch_pos(ph, pw, device)
            for i in range(self.num_encoder_blocks):
                s = self.blocks[i](s, condition, xpos)

        elif self.encoder_type == "sfc":
            # 1) SFC tokens from sparse hints (or dense if cond_mask_enc is None)
            tokens, token_mask = self.sfc_tokenizer(x_for_encoder, cond_mask_enc)  # (B,T,D), (B,T) bool
            tokens = tokens + condition  # (B,T,D)

            # NEW: self-attn over sparse tokens before cross-attn
            for blk in self.sfc_self:
                tokens = blk(tokens, token_mask)

            # 2) queries: one per patch (same coordinate embedding as Perceiver path)
            query_coords = self._fetch_patch_query_coords(encoder_h, encoder_w, device, dtype).expand(B, -1, -1)
            queries = self.query_coord_embed(query_coords) + condition  # (B,L,D)

            # 3) cross-attend tokens -> queries (no latent bottleneck)
            s = self.sfc_cross(queries, tokens, token_mask)  # (B,L,D)

            # 4) parity with grid: inject patch hint density on s
            if cond_mask_enc is not None:
                m = F.unfold(
                    cond_mask_enc, kernel_size=self.patch_size, stride=self.patch_size
                ).transpose(1, 2)
                m = m.mean(dim=-1, keepdim=True)  # (B,L,1)
                s = s + self.patch_hint_proj(m)

            # 5) SAME AdaLN+RoPE DiT blocks
            xpos = self.fetch_pos(ph, pw, device)
            for i in range(self.num_encoder_blocks):
                s = self.blocks[i](s, condition, xpos)

        else:
            # Grid patch encoder (mask-aware)
            x_for_encoder_patches = F.unfold(
                x_for_encoder, kernel_size=self.patch_size, stride=self.patch_size
            ).transpose(1, 2)  # (B,L,C*ps^2)

            xpos = self.fetch_pos(ph, pw, device)
            s = self.s_embedder(x_for_encoder_patches)  # (B,L,D)

            if cond_mask_enc is not None:
                m = F.unfold(
                    cond_mask_enc, kernel_size=self.patch_size, stride=self.patch_size
                ).transpose(1, 2)
                m = m.mean(dim=-1, keepdim=True)  # (B,L,1)
                s = s + self.patch_hint_proj(m)

            for i in range(self.num_encoder_blocks):
                s = self.blocks[i](s, condition, xpos)

        # Prepare for decoder
        s = torch.nn.functional.silu(t_emb + s)  # (B,L,D)

        # ------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------
        H_out = int(H * superres_scale)
        W_out = int(W * superres_scale)
        if superres_scale != 1.0:
            x_for_decoder_input = F.interpolate(x, (H_out, W_out), mode="bilinear", align_corners=False)
        else:
            x_for_decoder_input = x

        x_for_decoder = F.unfold(
            x_for_decoder_input,
            kernel_size=(decoder_patch_size_h, decoder_patch_size_w),
            stride=(decoder_patch_size_h, decoder_patch_size_w),
        ).transpose(1, 2)  # (B,L,C*dp^2)

        batch_size, length, _ = s.shape  # length should be L
        x_dec = x_for_decoder.reshape(
            batch_size * length, self.in_channels, decoder_patch_size_h * decoder_patch_size_w
        ).transpose(1, 2)  # (B*L, dp^2, C)

        s_dec = s.view(batch_size * length, self.hidden_size)  # (B*L, D)

        x_dec = self.x_embedder(x_dec, decoder_patch_size_h, decoder_patch_size_w)

        for i in range(self.num_decoder_blocks):
            x_dec = self.blocks[i + self.num_encoder_blocks](x_dec, s_dec)

        x_dec = self.final_layer(x_dec)  # (B*L, dp^2, C)
        x_dec = x_dec.transpose(1, 2).reshape(batch_size, length, -1)  # (B,L,C*dp^2)

        x_out = F.fold(
            x_dec.transpose(1, 2).contiguous(),
            (H_out, W_out),
            kernel_size=(decoder_patch_size_h, decoder_patch_size_w),
            stride=(decoder_patch_size_h, decoder_patch_size_w),
        )
        return x_out
