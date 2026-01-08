# PixNerd vs SFC-Based Architecture: Technical Comparison

## Executive Summary

This document compares the original PixNerd architecture with our Space-Filling Curve (SFC) based approach for sparse-conditioned image generation. We introduce **Options A and B** to address fundamental challenges in handling sparse pixel conditioning while maintaining spatial coherence.

---

## 1. The Challenge: Sparse Conditioning with Spatial Coherence

### Problem Statement

Traditional diffusion models condition on global signals (class labels, text embeddings). We want to condition on **sparse pixel observations**—a small fraction of pixels (5-40%) that provide local color/structure hints for image completion or super-resolution.

### Key Challenges

```
Challenge 1: VARIABLE SPARSITY
├── Training: ~20% random pixels observed
├── Inference: Could be 5%, 10%, 40%, or irregular patterns
└── Model must generalize across sparsity levels

Challenge 2: SPATIAL LOCALITY
├── Nearby pixels should influence each other more
├── Object boundaries must remain coherent
└── "Semantic bleeding" must be prevented

Challenge 3: RESOLUTION FLEXIBILITY
├── Train at 32x32, inference at 128x128 (4x SR)
├── Conditioning pattern scales with resolution
└── Decoder must handle variable output sizes

Challenge 4: EFFICIENT SPARSE REPRESENTATION
├── Dense attention over all pixels is O(N²) expensive
├── Most pixels are unobserved (no information)
└── Need compact representation of sparse observations
```

### The Core Insight

**Information should flow from observed pixels to queries based on:**
1. **Spatial proximity** (nearby pixels matter more)
2. **Semantic consistency** (same object → similar treatment)
3. **Coordinate awareness** (explicit position encoding)

---

## 2. Architecture Comparison

### 2.1 Original PixNerd: Grid-Based Encoding

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ORIGINAL PIXNERD ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUT IMAGE (32x32x3)          CONDITIONING                        │
│       │                              │                               │
│       ▼                              ▼                               │
│  ┌─────────────┐              ┌─────────────┐                       │
│  │  Patchify   │              │ Class Label │                       │
│  │  (8x8 grid) │              │  Embedding  │                       │
│  └─────────────┘              └─────────────┘                       │
│       │                              │                               │
│       ▼                              │                               │
│  ┌─────────────┐                     │                               │
│  │  16 Patch   │◄────────────────────┘                              │
│  │   Tokens    │     (AdaLN conditioning)                           │
│  │  (16, 512)  │                                                    │
│  └─────────────┘                                                    │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────┐                        │
│  │         ENCODER: DiT Blocks (x8)        │                        │
│  │  ┌─────────────────────────────────┐    │                        │
│  │  │  Self-Attention (all patches)   │    │                        │
│  │  │  + RoPE positional encoding     │    │                        │
│  │  │  + AdaLN (time + class)         │    │                        │
│  │  └─────────────────────────────────┘    │                        │
│  └─────────────────────────────────────────┘                        │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────┐                        │
│  │      DECODER: NerfBlocks (x2)           │                        │
│  │  ┌─────────────────────────────────┐    │                        │
│  │  │  Per-patch hypernetwork MLP     │    │                        │
│  │  │  Generates pixels within patch  │    │                        │
│  │  └─────────────────────────────────┘    │                        │
│  └─────────────────────────────────────────┘                        │
│       │                                                              │
│       ▼                                                              │
│  OUTPUT IMAGE (32x32x3 or upscaled)                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

LIMITATIONS:
• Fixed grid structure - cannot handle arbitrary sparse patterns
• Patch-level granularity - loses fine pixel-level detail
• All patches treated equally - no sparsity awareness
• Self-attention among patches - indirect spatial relationships
```

### 2.2 Our SFC-Based Architecture with Options A+B

```
┌─────────────────────────────────────────────────────────────────────┐
│              SFC-BASED ARCHITECTURE (Options A+B)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUT IMAGE        COND MASK (20%)       CLASS LABEL               │
│   (32x32x3)          (32x32x1)               │                      │
│       │                  │                    │                      │
│       ▼                  ▼                    ▼                      │
│  ┌────────────────────────────┐        ┌───────────┐                │
│  │    SFC TOKENIZER           │        │  Label    │                │
│  │  ┌──────────────────────┐  │        │ Embedding │                │
│  │  │ 1. Hilbert curve     │  │        └───────────┘                │
│  │  │    ordering          │  │              │                      │
│  │  │ 2. Select observed   │  │              │                      │
│  │  │    pixels only       │  │              │                      │
│  │  │ 3. Group by 8        │  │              │                      │
│  │  │ 4. Concat + project  │  │              │                      │
│  │  └──────────────────────┘  │              │                      │
│  └────────────────────────────┘              │                      │
│       │                                      │                      │
│       ▼                                      │                      │
│  ┌─────────────┐                             │                      │
│  │  T Sparse   │  (T = ceil(K/8) where       │                      │
│  │   Tokens    │   K = observed pixels)      │                      │
│  │  (T, 512)   │                             │                      │
│  └─────────────┘                             │                      │
│       │                                      │                      │
│       ▼                                      │                      │
│  ┌─────────────────────────────────────────┐ │                      │
│  │    TOKEN SELF-ATTENTION (x2)            │ │                      │
│  │    Sparse tokens attend to each other   │ │                      │
│  └─────────────────────────────────────────┘ │                      │
│       │                                      │                      │
│       │    ┌──────────────────────────────┐  │                      │
│       │    │   PATCH QUERIES (16, 512)    │◄─┘                      │
│       │    │   + Coordinate Embedding     │                         │
│       │    │   ══════════════════════════ │                         │
│       │    │   OPTION A: Shared coord     │                         │
│       │    │   embedder with tokens       │                         │
│       │    └──────────────────────────────┘                         │
│       │                  │                                          │
│       ▼                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │           CROSS-ATTENTION (x2 blocks)                    │        │
│  │  ┌─────────────────────────────────────────────────┐    │        │
│  │  │  Queries: 16 patch positions                    │    │        │
│  │  │  Keys/Values: T sparse tokens                   │    │        │
│  │  │  ════════════════════════════════════════════   │    │        │
│  │  │  OPTION B: Spatial attention bias               │    │        │
│  │  │  bias = -scale * distance(query, token) + off   │    │        │
│  │  │  → Nearby tokens get more attention             │    │        │
│  │  └─────────────────────────────────────────────────┘    │        │
│  └─────────────────────────────────────────────────────────┘        │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────┐                        │
│  │      ENCODER: DiT Blocks (x8)           │                        │
│  │      (Same as original PixNerd)         │                        │
│  └─────────────────────────────────────────┘                        │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────┐                        │
│  │      DECODER: NerfBlocks (x2)           │                        │
│  │      (Same as original PixNerd)         │                        │
│  └─────────────────────────────────────────┘                        │
│       │                                                              │
│       ▼                                                              │
│  OUTPUT IMAGE (32x32x3 or upscaled via decoder_patch_scaling)       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Four Largest Architectural Differences

### Difference 1: Sparse-Aware Tokenization via Space-Filling Curves

```
PIXNERD (Grid):                    SFC (Ours):
┌───┬───┬───┬───┐                 Hilbert curve ordering:
│ 1 │ 2 │ 3 │ 4 │                 ┌───┬───┬───┬───┐
├───┼───┼───┼───┤                 │ 0 │ 1 │14 │15 │
│ 5 │ 6 │ 7 │ 8 │  Fixed grid     ├───┼───┼───┼───┤
├───┼───┼───┼───┤  16 tokens      │ 3 │ 2 │13 │12 │  Variable T tokens
│ 9 │10 │11 │12 │  always         ├───┼───┼───┼───┤  based on observed
├───┼───┼───┼───┤                 │ 4 │ 7 │ 8 │11 │  pixels
│13 │14 │15 │16 │                 ├───┼───┼───┼───┤
└───┴───┴───┴───┘                 │ 5 │ 6 │ 9 │10 │
                                  └───┴───┴───┴───┘

                                  Only observed pixels → tokens
                                  Consecutive SFC pixels grouped
```

**What it enables:**
- **Adaptive token count**: More observed pixels → more tokens (more information)
- **Locality preservation**: Hilbert curve keeps spatially close pixels close in 1D
- **Efficient representation**: No tokens wasted on unobserved regions
- **Variable sparsity**: Works with any observation pattern

### Difference 2: Cross-Attention Instead of Self-Attention for Sparse→Dense

```
PIXNERD:                           SFC (Ours):
┌─────────────────┐               ┌─────────────────┐
│  Self-Attention │               │ Cross-Attention │
│                 │               │                 │
│  P₁ ←→ P₂       │               │  Q₁ ← T₁,T₂,T₃  │
│   ↕     ↕       │               │  Q₂ ← T₁,T₂,T₃  │
│  P₃ ←→ P₄       │               │  ...            │
│                 │               │                 │
│  All patches    │               │ Queries: patches│
│  attend to all  │               │ Keys: sparse    │
│                 │               │       tokens    │
└─────────────────┘               └─────────────────┘

P = patch tokens                   Q = query (patch center)
                                   T = sparse SFC tokens
```

**What it enables:**
- **Direct information flow**: Observed pixels directly inform output queries
- **No dilution**: Information doesn't pass through unobserved patches
- **Asymmetric roles**: Queries (where to generate) vs Keys (what we observed)
- **Scalability**: T tokens << N pixels, efficient for sparse conditioning

### Difference 3: Option A - Unified Coordinate Space

```
WITHOUT Option A:                  WITH Option A:

Token coords:                      SHARED EMBEDDER:
┌────────────────┐                ┌────────────────┐
│ LearnableMLP_1 │                │                │
│ (for tokens)   │                │ LearnableMLP   │──┐
└────────────────┘                │   (shared)     │  │
                                  │                │  │
Query coords:                     └────────────────┘  │
┌────────────────┐                      │             │
│ FourierFeatures│                      ▼             ▼
│ (for queries)  │                   Tokens        Queries
└────────────────┘                   (B,T,2)       (B,L,2)
                                        │             │
Different embedding                     ▼             ▼
spaces → harder to                  Same embedding space
compute spatial                     → spatial relationships
relationships                         directly comparable
```

**What it enables:**
- **Consistent coordinate representation**: Tokens and queries speak same "spatial language"
- **Meaningful distance**: Option B's spatial bias is well-defined
- **Transfer learning**: Coordinates learned from tokens help queries
- **Simpler architecture**: One embedder instead of two

### Difference 4: Option B - Spatial Attention Bias

```
WITHOUT Option B:                  WITH Option B:

Attention weights:                 Attention weights:
┌─────────────────────┐           ┌─────────────────────┐
│ softmax(Q·K / √d)   │           │ softmax(Q·K/√d      │
│                     │           │   - scale*dist      │
│ Purely content-     │           │   + offset)         │
│ based matching      │           │                     │
└─────────────────────┘           │ Distance-aware      │
                                  │ attention           │
                                  └─────────────────────┘

Example: Query at (0.5, 0.5)

Token at (0.4, 0.5): dist=0.1 → bias = -0.1*scale + offset (HIGH)
Token at (0.9, 0.1): dist=0.5 → bias = -0.5*scale + offset (LOW)

→ Nearby tokens contribute more to query output
```

**What it enables:**
- **Spatial locality**: Nearby observations matter more (physically correct)
- **Reduced bleeding**: Object boundaries respected (red bird stays red bird)
- **Learnable tradeoff**: scale/offset learned per-head for flexibility
- **Explicit spatial prior**: Model doesn't need to learn locality from scratch

---

## 4. Information Flow Comparison

### PixNerd: Dense → Dense

```
Input (all pixels)
       │
       ▼
┌──────────────┐
│   Patchify   │  → Loses pixel-level detail
└──────────────┘
       │
       ▼
┌──────────────┐
│ Self-Attn    │  → All patches see all patches
│ (symmetric)  │     No notion of "observed" vs "unobserved"
└──────────────┘
       │
       ▼
┌──────────────┐
│   Decoder    │  → Generate all pixels
└──────────────┘
```

### Ours: Sparse → Dense

```
Input (sparse pixels)         Queries (dense grid)
       │                             │
       ▼                             ▼
┌──────────────┐            ┌──────────────┐
│ SFC Tokenize │            │ Coord Embed  │
│ (K pixels →  │            │ (L positions)│
│  T tokens)   │            └──────────────┘
└──────────────┘                   │
       │                           │
       ▼                           │
┌──────────────┐                   │
│ Self-Attn    │  Tokens refine    │
│ (T tokens)   │  each other       │
└──────────────┘                   │
       │                           │
       └──────────┬────────────────┘
                  ▼
         ┌──────────────┐
         │ Cross-Attn   │  Queries ← Tokens
         │ + Spatial    │  (with distance bias)
         │   Bias       │
         └──────────────┘
                  │
                  ▼
         ┌──────────────┐
         │   Encoder    │  Same DiT blocks
         └──────────────┘
                  │
                  ▼
         ┌──────────────┐
         │   Decoder    │  Same NerfBlocks
         └──────────────┘
```

---

## 5. What These Changes Enable

### 5.1 Variable Sparsity Handling

| Sparsity | PixNerd | SFC (Ours) |
|----------|---------|------------|
| 5% | Same 16 tokens (wasteful) | ~13 tokens (compact) |
| 20% | Same 16 tokens | ~26 tokens |
| 80% | Same 16 tokens | ~103 tokens |

**Benefit**: Information scales with observation density.

### 5.2 Super-Resolution (SR)

```
Training: 32x32 with 20% pixels observed
Inference: 128x128 with same 32x32 conditioning lifted to HR grid

PixNerd Challenge:
- Fixed 4x4 patch grid at 32x32
- Must interpolate patches for 128x128

SFC Advantage:
- Tokens represent POINTS, not patches
- Same tokens work at any resolution
- Decoder scales via decoder_patch_scaling
```

### 5.3 Spatial Coherence

```
Without Options A+B:
┌─────────────────┐
│ Red bird with   │  Cross-attention is "diffuse"
│ red dots in     │  → Color bleeds to nearby regions
│ background      │
└─────────────────┘

With Options A+B:
┌─────────────────┐
│ Red bird with   │  Spatial bias focuses attention
│ clean edges     │  → Each query attends to nearest tokens
│                 │  → Object boundaries preserved
└─────────────────┘
```

### 5.4 Computational Efficiency

| Operation | PixNerd | SFC (Ours) |
|-----------|---------|------------|
| Encoder attention | O(L² · D) | O(T² + L·T) |
| With 20% sparsity | O(256 · D) | O(~26² + 16·26) |
| Memory | Fixed | Adaptive |

Where L=16 patches, T≈sparsity×1024/8 tokens, D=512.

---

## 6. Ablation Options Summary

| Option | Name | What it does | Expected benefit |
|--------|------|--------------|------------------|
| A | sfc_unified_coords | Shared coord embedder for tokens & queries | Consistent spatial representation |
| B | sfc_spatial_bias | Distance-based attention bias | Enforce spatial locality |
| C | sfc_attn_temperature | Sharpen/soften attention | Reduce information smearing |
| D | decoder_pixel_coords | Per-pixel coords in decoder | Sharper output boundaries |

### Recommended Configurations

```bash
# Baseline (original SFC without enhancements)
--no_option_a --no_option_b

# Standard (A+B, our default)
--option_a --option_b

# Sharp attention (if smearing observed)
--option_a --option_b --option_c --attn_temperature 0.3

# Full enhancement
--option_a --option_b --option_c --option_d
```

---

## 7. Conclusion

The SFC-based architecture with Options A+B fundamentally changes how sparse conditioning is handled:

1. **From grid to curve**: Variable, locality-preserving tokenization
2. **From self to cross**: Direct sparse→dense information flow
3. **From separate to unified**: Consistent coordinate representation
4. **From content to spatial**: Explicit distance-aware attention

These changes enable efficient sparse conditioning, flexible super-resolution, and better spatial coherence—addressing the core challenges of pixel-level conditional generation.

---

*Document version: 1.0*
*Last updated: January 2026*
