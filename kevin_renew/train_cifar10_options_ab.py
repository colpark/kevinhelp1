#!/usr/bin/env python
"""
CIFAR-10 Training with Options A/B (Ablation Study)

Option A (sfc_unified_coords): Shared coordinate embedder for tokens & queries
Option B (sfc_spatial_bias): Spatial attention bias in cross-attention

Usage:
    # Both options enabled (default)
    python train_cifar10_options_ab.py --exp_name cifar10_sfc_ab

    # Only Option A
    python train_cifar10_options_ab.py --exp_name cifar10_sfc_a_only --option_a --no_option_b

    # Only Option B
    python train_cifar10_options_ab.py --exp_name cifar10_sfc_b_only --no_option_a --option_b

    # Neither (baseline, same as kevin_pnbase_01072026)
    python train_cifar10_options_ab.py --exp_name cifar10_sfc_baseline --no_option_a --no_option_b
"""
import os
import sys
import argparse
import math
from pathlib import Path

# Add PixNerd to path
sys.path.insert(0, str(Path(__file__).parent / "PixNerd"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from src.models.transformer.pixnerd_c2i_heavydecoder import PixNerDiT
from src.diffusion.flow_matching.training import FlowMatchingTrainer
from src.diffusion.flow_matching.sampling import EulerSampler
from src.diffusion.base.scheduling import CosineScheduler
from src.diffusion.base.guidance import cfg_guidance


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=128, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, transform=self.transform
        )
        self.val_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, transform=self.test_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class SparseConditioningModule(pl.LightningModule):
    """
    Lightning module for sparse conditioning with Options A/B ablation.
    """
    def __init__(
        self,
        hidden_size=768,
        num_groups=12,
        num_encoder_blocks=12,
        num_decoder_blocks=2,
        decoder_hidden_size=64,
        num_classes=10,
        patch_size=2,
        learning_rate=1e-4,
        weight_decay=0.0,
        warmup_steps=5000,
        max_steps=200000,
        sparsity=0.4,
        cond_fraction=0.5,
        # Options A/B
        sfc_unified_coords=True,   # Option A
        sfc_spatial_bias=True,     # Option B
        # SFC settings
        sfc_curve="hilbert",
        sfc_group_size=8,
        sfc_cross_depth=2,
        sfc_self_depth=2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.sparsity = sparsity
        self.cond_fraction = cond_fraction

        # Build model with Options A/B
        self.model = PixNerDiT(
            in_channels=3,
            hidden_size=hidden_size,
            num_groups=num_groups,
            num_encoder_blocks=num_encoder_blocks,
            num_decoder_blocks=num_decoder_blocks,
            decoder_hidden_size=decoder_hidden_size,
            num_classes=num_classes,
            patch_size=patch_size,
            encoder_type="sfc",
            sfc_curve=sfc_curve,
            sfc_group_size=sfc_group_size,
            sfc_cross_depth=sfc_cross_depth,
            sfc_self_depth=sfc_self_depth,
            # Options A/B
            sfc_unified_coords=sfc_unified_coords,
            sfc_spatial_bias=sfc_spatial_bias,
        )

        # Flow matching trainer
        scheduler = CosineScheduler()
        self.diffusion_trainer = FlowMatchingTrainer(scheduler=scheduler)

        # Sampler for validation
        self.sampler = EulerSampler(
            scheduler=scheduler,
            w_scheduler=None,
            num_steps=50,
            guidance=4.0,
            guidance_fn=cfg_guidance,
            timeshift=1.0,
        )

    def generate_sparsity_masks(self, batch_size, height, width, device, dtype=torch.float32):
        """
        Generate disjoint cond_mask and target_mask.

        With sparsity=0.4 and cond_fraction=0.5:
          - 40% of pixels are "observed"
          - 20% go to cond_mask (hints given to model)
          - 20% go to target_mask (where loss is computed)
          - 60% are unobserved (not in either mask)

        Returns:
            cond_mask: (B,1,H,W) float, 1=hint pixel
            target_mask: (B,1,H,W) float, 1=compute loss here
        """
        B, H, W = batch_size, height, width
        total = H * W

        # Total observed pixels
        k_obs = int(round(self.sparsity * total))
        k_obs = max(0, min(total, k_obs))

        if k_obs == 0:
            cond_mask = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
            target_mask = torch.zeros_like(cond_mask)
            return cond_mask, target_mask

        # Split observed into cond and target
        k_cond = int(round(self.cond_fraction * k_obs))
        k_cond = max(0, min(k_obs, k_cond))
        k_target = k_obs - k_cond

        cond_mask = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
        target_mask = torch.zeros_like(cond_mask)

        flat_idx = torch.arange(total, device=device)

        for b in range(B):
            perm = flat_idx[torch.randperm(total, device=device)]
            obs_idx = perm[:k_obs]

            if k_cond > 0:
                cond_idx = obs_idx[:k_cond]
                cond_mask[b].view(-1)[cond_idx] = 1.0

            if k_target > 0:
                target_idx = obs_idx[k_cond:k_cond + k_target]
                target_mask[b].view(-1)[target_idx] = 1.0

        return cond_mask, target_mask

    def forward(self, x, t, y, cond_mask=None, **kwargs):
        return self.model(x, t, y, cond_mask=cond_mask, **kwargs)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        B, C, H, W = images.shape
        device = images.device

        # Generate disjoint cond and target masks
        # With sparsity=0.4, cond_fraction=0.5:
        #   - cond_mask: 20% of pixels (hints)
        #   - target_mask: 20% of pixels (loss computed here)
        #   - remaining 60%: unobserved
        cond_mask, target_mask = self.generate_sparsity_masks(B, H, W, device, images.dtype)

        # Flow matching training step
        # - cond_mask pixels are clamped to clean GT in x_t
        # - loss is computed ONLY on target_mask pixels
        loss = self.diffusion_trainer.training_step(
            self.model, images, labels,
            cond_mask=cond_mask,
            target_mask=target_mask,
        )

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        B, C, H, W = images.shape
        device = images.device

        # Same mask generation for validation
        cond_mask, target_mask = self.generate_sparsity_masks(B, H, W, device, images.dtype)
        loss = self.diffusion_trainer.training_step(
            self.model, images, labels,
            cond_mask=cond_mask,
            target_mask=target_mask,
        )

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

    @torch.no_grad()
    def sample(self, labels, cond_mask=None, x_cond=None, num_steps=50, guidance=4.0, disable_spatial_bias=False):
        """Generate samples with optional sparse conditioning."""
        device = next(self.parameters()).device
        B = labels.shape[0]

        # Start from noise
        noise = torch.randn(B, 3, 32, 32, device=device)

        # Null labels for CFG
        null_labels = torch.full_like(labels, self.model.num_classes)

        # Create sampler with desired settings
        sampler = EulerSampler(
            scheduler=CosineScheduler(),
            w_scheduler=None,
            num_steps=num_steps,
            guidance=guidance,
            guidance_fn=cfg_guidance,
            timeshift=1.0,
        )

        # Wrap model for CFG
        def denoiser(x, t, cond, cond_mask=None, **kwargs):
            return self.model(x, t, cond, cond_mask=cond_mask, disable_spatial_bias=disable_spatial_bias, **kwargs)

        # Sample
        x_trajs, _ = sampler.sampling(
            denoiser, noise, labels, null_labels,
            cond_mask=cond_mask,
            x_cond=x_cond,
        )

        return x_trajs[-1]


class SparseReconProgressCallback(Callback):
    """Callback to generate and save sample reconstructions during training."""

    def __init__(self, sample_every=10000, output_dir="outputs/samples", num_samples=8):
        super().__init__()
        self.sample_every = sample_every
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step > 0 and trainer.global_step % self.sample_every == 0:
            self._generate_samples(trainer, pl_module, batch)

    @torch.no_grad()
    def _generate_samples(self, trainer, pl_module, batch):
        pl_module.eval()
        device = pl_module.device

        images, labels = batch
        images = images[:self.num_samples].to(device)
        labels = labels[:self.num_samples].to(device)

        B, C, H, W = images.shape

        # For inference/visualization, we use cond_mask as hints
        # (we don't need target_mask for sampling, only for training)
        cond_mask, _ = pl_module.generate_sparsity_masks(B, H, W, device, images.dtype)
        x_cond = images  # Use original images as conditioning values

        # ============================================================
        # 1. 32x32 Reconstruction
        # ============================================================
        recon_32 = pl_module.sample(
            labels, cond_mask=cond_mask, x_cond=x_cond,
            num_steps=50, guidance=4.0
        )

        # ============================================================
        # 2. 128x128 Super-Resolution (with disable_spatial_bias=True)
        # ============================================================
        # For SR, use full 32x32 as conditioning
        full_mask = torch.ones(B, 1, 32, 32, device=device)

        # Set decoder patch scaling for 4x SR
        pl_module.model.decoder_patch_scaling_h = 4.0
        pl_module.model.decoder_patch_scaling_w = 4.0

        noise_sr = torch.randn(B, 3, 128, 128, device=device)
        null_labels = torch.full_like(labels, pl_module.model.num_classes)

        sampler = EulerSampler(
            scheduler=CosineScheduler(),
            w_scheduler=None,
            num_steps=50,
            guidance=4.0,
            guidance_fn=cfg_guidance,
            timeshift=1.0,
        )

        def denoiser_sr(x, t, cond, cond_mask=None, **kwargs):
            # IMPORTANT: disable_spatial_bias=True for SR to avoid checkerboard
            return pl_module.model(x, t, cond, cond_mask=cond_mask, disable_spatial_bias=True, **kwargs)

        x_trajs_sr, _ = sampler.sampling(
            denoiser_sr, noise_sr, labels, null_labels,
            cond_mask=full_mask,
            x_cond=images,
        )
        recon_128 = x_trajs_sr[-1]

        # Reset scaling
        pl_module.model.decoder_patch_scaling_h = 1.0
        pl_module.model.decoder_patch_scaling_w = 1.0

        # ============================================================
        # Save images
        # ============================================================
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Denormalize
        def denorm(x):
            return (x * 0.5 + 0.5).clamp(0, 1)

        # Create visualization grid
        # Row 1: Original | Sparse Hints | Recon 32x32 | SR 128x128 (resized to 32x32 for comparison)
        originals = denorm(images)
        sparse_vis = originals * cond_mask + (1 - cond_mask) * 0.5
        recons = denorm(recon_32)
        sr_resized = F.interpolate(denorm(recon_128), size=(32, 32), mode='bilinear', align_corners=False)

        # Concat horizontally
        grid_rows = []
        for i in range(min(4, B)):
            row = torch.cat([originals[i], sparse_vis[i], recons[i], sr_resized[i]], dim=2)  # (C, H, 4*W)
            grid_rows.append(row)
        grid = torch.cat(grid_rows, dim=1)  # (C, 4*H, 4*W)

        # Save
        grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(grid_np).save(self.output_dir / f"step_{trainer.global_step:06d}.png")

        # Also save full-res SR
        sr_grid = torchvision.utils.make_grid(denorm(recon_128), nrow=4)
        sr_np = (sr_grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(sr_np).save(self.output_dir / f"step_{trainer.global_step:06d}_sr128.png")

        pl_module.train()
        print(f"[Step {trainer.global_step}] Saved samples to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 with Options A/B")

    # Experiment
    parser.add_argument("--exp_name", type=str, default="cifar10_sfc_ab")
    parser.add_argument("--output_dir", type=str, default="outputs")

    # Data
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    # Model
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_groups", type=int, default=12)
    parser.add_argument("--num_encoder_blocks", type=int, default=12)
    parser.add_argument("--num_decoder_blocks", type=int, default=2)
    parser.add_argument("--decoder_hidden_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=2)

    # Options A/B (the key ablation flags)
    parser.add_argument("--option_a", action="store_true", default=True,
                        help="Enable Option A: unified coord embedding")
    parser.add_argument("--no_option_a", action="store_false", dest="option_a",
                        help="Disable Option A")
    parser.add_argument("--option_b", action="store_true", default=True,
                        help="Enable Option B: spatial attention bias")
    parser.add_argument("--no_option_b", action="store_false", dest="option_b",
                        help="Disable Option B")

    # SFC settings
    parser.add_argument("--sfc_curve", type=str, default="hilbert", choices=["hilbert", "zorder"])
    parser.add_argument("--sfc_group_size", type=int, default=8)
    parser.add_argument("--sfc_cross_depth", type=int, default=2)
    parser.add_argument("--sfc_self_depth", type=int, default=2)

    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--sparsity", type=float, default=0.4)
    parser.add_argument("--cond_fraction", type=float, default=0.5)

    # Callbacks
    parser.add_argument("--sample_every", type=int, default=10000)

    # Hardware
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="16-mixed")

    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print(f"Training CIFAR-10 with Options A/B Ablation")
    print("=" * 60)
    print(f"  Experiment: {args.exp_name}")
    print(f"  Option A (unified coords): {args.option_a}")
    print(f"  Option B (spatial bias):   {args.option_b}")
    print(f"  Sparsity: {args.sparsity}")
    print(f"  Cond fraction: {args.cond_fraction}")
    print(f"  Max steps: {args.max_steps}")
    print("=" * 60)

    # Data
    datamodule = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model = SparseConditioningModule(
        hidden_size=args.hidden_size,
        num_groups=args.num_groups,
        num_encoder_blocks=args.num_encoder_blocks,
        num_decoder_blocks=args.num_decoder_blocks,
        decoder_hidden_size=args.decoder_hidden_size,
        num_classes=10,
        patch_size=args.patch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        sparsity=args.sparsity,
        cond_fraction=args.cond_fraction,
        # Options A/B
        sfc_unified_coords=args.option_a,
        sfc_spatial_bias=args.option_b,
        # SFC
        sfc_curve=args.sfc_curve,
        sfc_group_size=args.sfc_group_size,
        sfc_cross_depth=args.sfc_cross_depth,
        sfc_self_depth=args.sfc_self_depth,
    )

    # Callbacks
    output_dir = Path(args.output_dir) / args.exp_name
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="step_{step:06d}",
            save_top_k=-1,
            every_n_train_steps=args.sample_every,
        ),
        LearningRateMonitor(logging_interval="step"),
        SparseReconProgressCallback(
            sample_every=args.sample_every,
            output_dir=output_dir / "samples",
        ),
    ]

    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="logs",
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_steps=args.max_steps,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=args.sample_every,
        gradient_clip_val=1.0,
    )

    # Train
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
