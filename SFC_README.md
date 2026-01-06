# PixNerDiT with Space-Filling Curve Encoder

A class-conditional diffusion model for image generation and super-resolution, featuring a novel **Space-Filling Curve (SFC) encoder** for efficient sparse pixel conditioning.

## Overview

This repository implements PixNerDiT (Pixel-level NeRF Diffusion Transformer) with three encoder options:
- **Grid**: Traditional patch-based ViT encoder
- **Perceiver**: Point-token Perceiver with latent bottleneck
- **SFC**: Space-filling curve tokenizer (Hilbert/Z-order) - **NEW**

The SFC encoder is designed for **sparse conditioning** scenarios where only a subset of pixels are observed.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           INPUTS                                     │
│  x: (B,C,H,W)        cond_mask: (B,1,H,W)      y: (B,)    t: (B,)   │
│  noisy image         sparse pixel mask         class      timestep  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SFC TOKENIZER                                   │
│                                                                      │
│  1. Order pixels by Hilbert/Z-order curve                           │
│  2. Select only observed pixels (cond_mask=1)                       │
│  3. Group consecutive g pixels → concat → project                   │
│                                                                      │
│  Output: tokens (B,T,D), token_mask (B,T), token_coords (B,T,2)     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SELF-ATTENTION (sfc_self)                        │
│                                                                      │
│  Transformer blocks on sparse tokens (T << H*W)                     │
│  Enables token-to-token communication before cross-attention        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  CROSS-ATTENTION (sfc_cross)                        │
│                                                                      │
│  Queries: patch grid positions (B,L,D) where L = (H/p)*(W/p)       │
│  Keys/Values: sparse SFC tokens (B,T,D)                             │
│                                                                      │
│  Each patch query attends to ALL sparse tokens                      │
│  → No fixed bottleneck (unlike Perceiver)                           │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DiT ENCODER BLOCKS                                │
│                                                                      │
│  AdaLN (Adaptive LayerNorm) conditioned on time + class             │
│  RoPE (Rotary Position Embedding) for patch positions               │
│  Self-attention among patch embeddings                              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    HEAVY DECODER (NerfBlocks)                        │
│                                                                      │
│  Hypernetwork: encoder output → generates MLP weights               │
│  Enables arbitrary output resolution (super-resolution)             │
│  NeRF-style continuous coordinate encoding                          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
                    Output: noise/velocity prediction
```

## Space-Filling Curves

### Why SFC for Sparse Tokenization?

Traditional ViT tokenization creates fixed patches regardless of which pixels are observed. For sparse inputs, this is wasteful. SFC tokenization:

1. **Preserves spatial locality**: Nearby pixels remain nearby in the 1D sequence
2. **Variable-length tokens**: Only observed pixels are tokenized
3. **Efficient for sparse data**: Token count scales with observed pixels, not image size

### Hilbert vs Z-order Curves

```
Hilbert Curve (better locality)     Z-order/Morton Curve (simpler)
┌───┬───┬───┬───┐                   ┌───┬───┬───┬───┐
│ 0 │ 1 │14 │15 │                   │ 0 │ 1 │ 4 │ 5 │
├───┼───┼───┼───┤                   ├───┼───┼───┼───┤
│ 3 │ 2 │13 │12 │                   │ 2 │ 3 │ 6 │ 7 │
├───┼───┼───┼───┤                   ├───┼───┼───┼───┤
│ 4 │ 7 │ 8 │11 │                   │ 8 │ 9 │12 │13 │
├───┼───┼───┼───┤                   ├───┼───┼───┼───┤
│ 5 │ 6 │ 9 │10 │                   │10 │11 │14 │15 │
└───┴───┴───┴───┘                   └───┴───┴───┴───┘

Hilbert: consecutive indices are    Z-order: interleaves bits of (x,y)
always spatially adjacent           May have jumps (e.g., 7→8)
```

## Ablation Options

We provide two orthogonal improvements to the SFC encoder:

### Option A: Unified Coordinate Embeddings (`--sfc_unified_coords`)

**Problem**: Tokens and queries use different coordinate embeddings:
- Tokens: `LearnableFourierMLP` (2-layer MLP on Fourier features)
- Queries: `FourierFeatures` (direct Fourier projection)

The cross-attention must learn to align these different representations.

**Solution**: Share a single `LearnableFourierMLP` for both:

```
                 shared_coord_embed
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   Token Coords (B,T,2)        Query Coords (B,L,2)
          │                           │
          ▼                           ▼
   Token Embeddings            Query Embeddings
   (same representation space)
```

### Option B: Spatial Attention Bias (`--sfc_spatial_bias`)

**Problem**: Cross-attention must learn from scratch that a patch query should attend more to spatially nearby tokens.

**Solution**: Add explicit distance-based bias to attention scores:

```python
# Compute pairwise distances
dist = cdist(query_coords, token_coords)  # (B, L, T)

# Learnable per-head bias
bias = -scale[head] * dist + offset[head]

# Add to attention
attn_scores = (Q @ K.T) / sqrt(d) + bias
```

Closer tokens automatically get higher attention weights, with learnable scale per head.

### Ablation Matrix

| Configuration | Unified Coords | Spatial Bias | Flag Combination |
|--------------|----------------|--------------|------------------|
| Baseline     | No             | No           | (none)           |
| Option A     | Yes            | No           | `--sfc_unified_coords` |
| Option B     | No             | Yes          | `--sfc_spatial_bias` |
| Option A+B   | Yes            | Yes          | `--sfc_unified_coords --sfc_spatial_bias` |

## Installation

```bash
# Clone the repository
git clone https://github.com/colpark/kevinhelp1.git
cd kevinhelp1

# Install dependencies (assuming PixNerd base is available)
pip install torch torchvision lightning wandb
```

## Usage

### Training

```bash
# Baseline SFC encoder
python train_cifar10.py --encoder_type sfc --exp_name sfc_baseline

# Option A: Unified coordinates
python train_cifar10.py --encoder_type sfc --sfc_unified_coords --exp_name sfc_unified

# Option B: Spatial bias
python train_cifar10.py --encoder_type sfc --sfc_spatial_bias --exp_name sfc_spatial

# Option A+B: Both
python train_cifar10.py --encoder_type sfc --sfc_unified_coords --sfc_spatial_bias --exp_name sfc_both
```

### Running Ablations

```bash
# Run all 4 configurations
./train_cifar10_ablations.sh all

# Run specific configuration
./train_cifar10_ablations.sh baseline
./train_cifar10_ablations.sh optionA
./train_cifar10_ablations.sh optionB
./train_cifar10_ablations.sh optionAB
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--encoder_type` | `grid` | Encoder type: `grid`, `perceiver`, `sfc` |
| `--sfc_curve` | `hilbert` | SFC type: `hilbert`, `zorder` |
| `--sfc_group_size` | `8` | Pixels per token group |
| `--sfc_cross_depth` | `2` | Cross-attention layers |
| `--sfc_unified_coords` | `False` | Option A: shared coord embedder |
| `--sfc_spatial_bias` | `False` | Option B: spatial attention bias |
| `--sparsity` | `0.4` | Fraction of pixels to keep |
| `--cond_fraction` | `0.5` | Fraction of kept pixels as conditioning |

## File Structure

```
kevinhelp1/
├── README.md                      # This file
├── sfc_encoder.py                 # SFC tokenizer and cross-attention
├── pixnerd_c2i_heavydecoder.py   # Main PixNerDiT model
├── train_cifar10.py              # Training script
├── train_cifar10_ablations.sh    # Ablation experiment runner
└── sfc_architecture_walkthrough.ipynb  # Step-by-step notebook
```

## Detailed Component Breakdown

### 1. SFCTokenizer (`sfc_encoder.py`)

Converts sparse pixels to tokens:

```python
# Input
x: (B, C, H, W)           # Image (e.g., 8, 3, 32, 32)
cond_mask: (B, 1, H, W)   # Binary mask of observed pixels

# Output
tokens: (B, T, D)         # T varies per batch based on sparsity
token_mask: (B, T)        # True = real token, False = padding
token_coords: (B, T, 2)   # Mean coordinate per token group
```

Per-token features:
- Pixel values (C channels)
- Coordinate embedding (hidden_size via LearnableFourierMLP)
- SFC position scalar (where along the curve, in [0,1])

### 2. SFCQueryCrossEncoder (`sfc_encoder.py`)

Cross-attention from patch queries to sparse tokens:

```python
# Queries: one per output patch
queries: (B, L, D)        # L = (H/patch_size) * (W/patch_size)
query_coords: (B, L, 2)   # Patch center coordinates

# Keys/Values: sparse tokens
tokens: (B, T, D)
token_coords: (B, T, 2)

# Output: enriched patch representations
output: (B, L, D)
```

### 3. Heavy Decoder (`pixnerd_c2i_heavydecoder.py`)

Hypernetwork-based decoder for arbitrary resolution:

```python
# Encoder output conditions weight generation
s: (B, L, D)  # Patch embeddings from encoder

# Hypernetwork generates MLP weights
mlp_weights = param_generator(s)  # Dynamic per-patch MLPs

# Apply to decoder input for super-resolution
output = dynamic_mlp(decoder_input, mlp_weights)
```

## Citation

If you use this code, please cite:

```bibtex
@misc{pixnerdit_sfc,
  title={PixNerDiT with Space-Filling Curve Encoder},
  author={...},
  year={2024},
  url={https://github.com/colpark/kevinhelp1}
}
```

## License

[Add license information]
