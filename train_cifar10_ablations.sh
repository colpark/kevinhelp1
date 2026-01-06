#!/bin/bash
# =============================================================================
# SFC Encoder Ablation Experiments
# =============================================================================
#
# This script runs 4 ablation configurations for the SFC encoder:
#   1. Baseline:     No options (separate coords, no spatial bias)
#   2. Option A:     Unified coordinate embedder (shared between tokens & queries)
#   3. Option B:     Spatial attention bias in cross-attention
#   4. Option A+B:   Both unified coords and spatial bias
#
# Ablation Matrix:
#   |                    | No Spatial Bias | Spatial Bias |
#   |--------------------|-----------------|--------------|
#   | Separate Coords    | Baseline        | +B only      |
#   | Unified Coords     | +A only         | +A+B         |
#
# Usage:
#   ./train_cifar10_ablations.sh [baseline|optionA|optionB|optionAB|all]
#
# =============================================================================

set -e  # Exit on error

# Common training arguments
COMMON_ARGS="
    --encoder_type sfc
    --sfc_curve hilbert
    --sfc_group_size 8
    --sfc_cross_depth 2
    --max_steps 100000
    --batch_size 128
    --lr 1e-4
    --hidden_size 512
    --num_encoder_blocks 8
    --patch_size 8
    --save_every_n_steps 10000
    --sample_every_n_steps 5000
"

# Output directory base
OUTPUT_BASE="./workdirs/ablations"

run_baseline() {
    echo "=============================================="
    echo "Running BASELINE (separate coords, no spatial bias)"
    echo "=============================================="
    python train_cifar10.py \
        $COMMON_ARGS \
        --exp_name "sfc_baseline" \
        --output_dir "${OUTPUT_BASE}"
}

run_option_a() {
    echo "=============================================="
    echo "Running OPTION A (unified coords)"
    echo "=============================================="
    python train_cifar10.py \
        $COMMON_ARGS \
        --sfc_unified_coords \
        --exp_name "sfc_unified_coords" \
        --output_dir "${OUTPUT_BASE}"
}

run_option_b() {
    echo "=============================================="
    echo "Running OPTION B (spatial bias)"
    echo "=============================================="
    python train_cifar10.py \
        $COMMON_ARGS \
        --sfc_spatial_bias \
        --exp_name "sfc_spatial_bias" \
        --output_dir "${OUTPUT_BASE}"
}

run_option_ab() {
    echo "=============================================="
    echo "Running OPTION A+B (unified coords + spatial bias)"
    echo "=============================================="
    python train_cifar10.py \
        $COMMON_ARGS \
        --sfc_unified_coords \
        --sfc_spatial_bias \
        --exp_name "sfc_unified_coords_spatial_bias" \
        --output_dir "${OUTPUT_BASE}"
}

run_all() {
    echo "Running all 4 ablation configurations sequentially..."
    run_baseline
    run_option_a
    run_option_b
    run_option_ab
    echo ""
    echo "=============================================="
    echo "All ablations complete!"
    echo "Results saved to: ${OUTPUT_BASE}"
    echo "=============================================="
}

# Parse command line argument
case "${1:-all}" in
    baseline)
        run_baseline
        ;;
    optionA|option_a|a)
        run_option_a
        ;;
    optionB|option_b|b)
        run_option_b
        ;;
    optionAB|option_ab|ab|both)
        run_option_ab
        ;;
    all)
        run_all
        ;;
    *)
        echo "Usage: $0 [baseline|optionA|optionB|optionAB|all]"
        echo ""
        echo "Options:"
        echo "  baseline  - Run baseline (separate coords, no spatial bias)"
        echo "  optionA   - Run Option A only (unified coords)"
        echo "  optionB   - Run Option B only (spatial bias)"
        echo "  optionAB  - Run Option A+B (unified coords + spatial bias)"
        echo "  all       - Run all 4 configurations sequentially (default)"
        exit 1
        ;;
esac
