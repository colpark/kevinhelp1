#!/usr/bin/env python3
"""
CIFAR-10 Training with Options A/B/C/D (Ablation Study)

Based on kevin_pnbase_01072026/train_cifar10.py with ablation options added.

Option A (sfc_unified_coords): Shared coordinate embedder for tokens & queries
Option B (sfc_spatial_bias): Spatial attention bias in cross-attention
Option C (sfc_attn_temperature): Temperature scaling to sharpen attention (< 1.0 sharpens)
Option D (decoder_pixel_coords): Per-pixel coordinate injection in decoder

Usage:
    # Options A+B enabled (default)
    python train_cifar10_options_ab.py --exp_name cifar10_sfc_ab

    # Options A+B+C+D (all options)
    python train_cifar10_options_ab.py --exp_name cifar10_sfc_abcd --option_c --option_d

    # Option C only (sharper attention)
    python train_cifar10_options_ab.py --exp_name cifar10_sfc_c --no_option_a --no_option_b --option_c

    # Option D only (per-pixel coords)
    python train_cifar10_options_ab.py --exp_name cifar10_sfc_d --no_option_a --no_option_b --option_d

    # Custom temperature (0.3 = very sharp)
    python train_cifar10_options_ab.py --exp_name cifar10_sfc_sharp --option_c --attn_temperature 0.3

    # Baseline (no options)
    python train_cifar10_options_ab.py --exp_name cifar10_sfc_baseline --no_option_a --no_option_b
"""
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image

import os
import sys
import argparse
from pathlib import Path
from functools import partial

# Add PixNerd to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PIXNERD_DIR = SCRIPT_DIR / "PixNerd"
if PIXNERD_DIR.exists():
    os.chdir(PIXNERD_DIR)
    sys.path.insert(0, str(PIXNERD_DIR))

import torch
# Completely disable torch.compile/dynamo to avoid inductor errors
torch._dynamo.config.disable = True

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import CSVLogger

from src.models.autoencoder.pixel import PixelAE
from src.models.conditioner.class_label import LabelConditioner
from src.models.transformer.pixnerd_c2i_heavydecoder import PixNerDiT
from src.diffusion.flow_matching.scheduling import LinearScheduler
from src.diffusion.flow_matching.sampling import EulerSampler, ode_step_fn
from src.diffusion.base.guidance import simple_guidance_fn
from src.diffusion.flow_matching.training import FlowMatchingTrainer
from src.callbacks.simple_ema import SimpleEMA
from src.callbacks.save_images import SaveImagesHook
from src.lightning_model import LightningModel
from src.lightning_data import DataModule
from src.data.dataset.cifar10 import PixCIFAR10, CIFAR10RandomNDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 with Options A/B")

    # Training config
    parser.add_argument("--max_steps", type=int, default=300000, help="Max training steps")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=32)

    # Model config
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--decoder_hidden_size", type=int, default=64)
    parser.add_argument("--num_encoder_blocks", type=int, default=8)
    parser.add_argument("--num_decoder_blocks", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--num_groups", type=int, default=8)

    # Sampler config
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--num_sample_steps", type=int, default=200)

    # Logging
    parser.add_argument("--exp_name", type=str, default="cifar10_options_ab")
    parser.add_argument("--output_dir", type=str, default="./workdirs")

    # Checkpointing
    parser.add_argument("--save_every_n_steps", type=int, default=5000)
    parser.add_argument("--val_every_n_epochs", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)

    # Hardware
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--devices", type=int, default=1)

    # Sparsity-conditioning (training uses cond_mask + target_mask)
    parser.add_argument("--sparsity", type=float, default=0.4)
    parser.add_argument("--cond_fraction", type=float, default=0.5)

    # Progress sampling
    parser.add_argument("--sample_every_n_steps", type=int, default=10000)
    parser.add_argument("--sample_batch_size", type=int, default=8)

    # SFC encoder settings
    parser.add_argument("--sfc_curve", type=str, default="hilbert", choices=["hilbert", "zorder"])
    parser.add_argument("--sfc_group_size", type=int, default=8)
    parser.add_argument("--sfc_cross_depth", type=int, default=2)
    parser.add_argument("--sfc_self_depth", type=int, default=2)

    # Options A/B (the main ablation controls)
    parser.add_argument("--option_a", dest="option_a", action="store_true",
                        help="Enable Option A (sfc_unified_coords)")
    parser.add_argument("--no_option_a", dest="option_a", action="store_false",
                        help="Disable Option A")
    parser.set_defaults(option_a=True)

    parser.add_argument("--option_b", dest="option_b", action="store_true",
                        help="Enable Option B (sfc_spatial_bias)")
    parser.add_argument("--no_option_b", dest="option_b", action="store_false",
                        help="Disable Option B")
    parser.set_defaults(option_b=True)

    # Option C: Attention temperature (< 1.0 sharpens attention)
    parser.add_argument("--option_c", dest="option_c", action="store_true",
                        help="Enable Option C (sfc_attn_temperature=0.5)")
    parser.add_argument("--no_option_c", dest="option_c", action="store_false",
                        help="Disable Option C (temperature=1.0)")
    parser.set_defaults(option_c=False)
    parser.add_argument("--attn_temperature", type=float, default=0.5,
                        help="Attention temperature when Option C is enabled (default: 0.5)")

    # Option D: Per-pixel coordinate injection in decoder
    parser.add_argument("--option_d", dest="option_d", action="store_true",
                        help="Enable Option D (decoder_pixel_coords)")
    parser.add_argument("--no_option_d", dest="option_d", action="store_false",
                        help="Disable Option D")
    parser.set_defaults(option_d=False)

    return parser.parse_args()


class SparseReconProgressCallback(Callback):
    """
    Periodically runs sparse-conditioned reconstruction & super-res on a fixed
    set of images and saves image grids to disk.
    """
    def __init__(
        self,
        data_root: str,
        out_dir: Path,
        sample_every_n_steps: int = 10000,
        sample_batch_size: int = 8,
        base_res: int = 32,
        superres_factor: int = 4,
    ):
        super().__init__()
        self.data_root = data_root
        self.out_dir = Path(out_dir)
        self.sample_every_n_steps = sample_every_n_steps
        self.sample_batch_size = sample_batch_size
        self.base_res = base_res
        self.superres_factor = superres_factor

        self.progress_dir = self.out_dir / "progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)

        self._fixed_images = None
        self._fixed_labels = None
        self._build_fixed_batch()

    def _build_fixed_batch(self):
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        ds = CIFAR10(root=self.data_root, train=False, transform=tfm, download=True)
        idx = torch.arange(self.sample_batch_size)
        imgs = [ds[i][0] for i in idx]
        labels = [ds[i][1] for i in idx]

        self._fixed_images = torch.stack(imgs, dim=0)
        self._fixed_labels = torch.tensor(labels, dtype=torch.long)

    def _save_grid(self, tensor: torch.Tensor, path: Path, nrow: int = None):
        x = tensor.detach().cpu()
        x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
        if nrow is None:
            nrow = x.size(0)
        save_image(x, str(path), nrow=nrow)

    @torch.no_grad()
    def _run_sampling(self, trainer, pl_module):
        if getattr(trainer, "global_rank", 0) != 0:
            return

        device = pl_module.device
        B = self.sample_batch_size

        images = self._fixed_images.to(device)
        labels = self._fixed_labels.to(device)

        was_training = pl_module.training
        pl_module.eval()

        # 32x32 sparse recon
        x_latent_32 = pl_module.vae.encode(images)
        cond_mask_32, target_mask_32 = pl_module._make_sparsity_masks(x_latent_32)

        condition, uncondition = pl_module.conditioner(labels)

        noise_32 = torch.randn_like(x_latent_32)
        samples_latent_32 = pl_module.diffusion_sampler(
            pl_module.ema_denoiser,
            noise_32,
            condition,
            uncondition,
            cond_mask=cond_mask_32,
            x_cond=x_latent_32,
        )
        recon_32 = pl_module.vae.decode(samples_latent_32)

        # 128x128 sparse super-res (1-pixel lift from 32x32 cond)
        scale = self.superres_factor
        H_hr = self.base_res * scale
        W_hr = self.base_res * scale

        cond_mask_128 = torch.zeros((B, 1, H_hr, W_hr), device=device, dtype=x_latent_32.dtype)
        cond_mask_128[:, :, ::scale, ::scale] = cond_mask_32

        x_cond_128 = torch.zeros((B, x_latent_32.shape[1], H_hr, W_hr), device=device, dtype=x_latent_32.dtype)
        x_cond_128[:, :, ::scale, ::scale] = x_latent_32

        old_scales = {}
        for name, net in [("denoiser", pl_module.denoiser), ("ema_denoiser", pl_module.ema_denoiser)]:
            if hasattr(net, "decoder_patch_scaling_h"):
                old_scales[name] = (net.decoder_patch_scaling_h, net.decoder_patch_scaling_w)
                net.decoder_patch_scaling_h = scale
                net.decoder_patch_scaling_w = scale

        noise_128 = torch.randn_like(x_cond_128)
        samples_latent_128 = pl_module.diffusion_sampler(
            pl_module.ema_denoiser,
            noise_128,
            condition,
            uncondition,
            cond_mask=cond_mask_128,
            x_cond=x_cond_128,
        )
        recon_128 = pl_module.vae.decode(samples_latent_128)

        for name, net in [("denoiser", pl_module.denoiser), ("ema_denoiser", pl_module.ema_denoiser)]:
            if name in old_scales:
                h, w = old_scales[name]
                net.decoder_patch_scaling_h = h
                net.decoder_patch_scaling_w = w

        step = trainer.global_step
        tag = f"step_{step:06d}"

        self._save_grid(images, self.progress_dir / f"{tag}_gt32.png")
        self._save_grid(recon_32, self.progress_dir / f"{tag}_recon32.png")
        self._save_grid(recon_128, self.progress_dir / f"{tag}_recon128.png")

        if was_training:
            pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step > 0 and step % self.sample_every_n_steps == 0:
            self._run_sampling(trainer, pl_module)


def build_model(args):
    main_scheduler = LinearScheduler()

    vae = PixelAE(scale=1.0)
    conditioner = LabelConditioner(num_classes=args.num_classes)

    # Build denoiser with Options A/B/C/D
    denoiser = PixNerDiT(
        in_channels=3,
        patch_size=args.patch_size,
        num_groups=args.num_groups,
        hidden_size=args.hidden_size,
        decoder_hidden_size=args.decoder_hidden_size,
        num_encoder_blocks=args.num_encoder_blocks,
        num_decoder_blocks=args.num_decoder_blocks,
        num_classes=args.num_classes,
        encoder_type="sfc",
        sfc_curve=args.sfc_curve,
        sfc_group_size=args.sfc_group_size,
        sfc_cross_depth=args.sfc_cross_depth,
        sfc_self_depth=args.sfc_self_depth,
        # Options A/B
        sfc_unified_coords=args.option_a,
        sfc_spatial_bias=args.option_b,
        # Options C/D
        sfc_attn_temperature=args.attn_temperature if args.option_c else 1.0,
        decoder_pixel_coords=args.option_d,
    )

    sampler = EulerSampler(
        num_steps=args.num_sample_steps,
        guidance=args.guidance,
        guidance_interval_min=0.0,
        guidance_interval_max=1.0,
        scheduler=main_scheduler,
        w_scheduler=LinearScheduler(),
        guidance_fn=simple_guidance_fn,
        step_fn=ode_step_fn,
    )

    trainer = FlowMatchingTrainer(
        scheduler=main_scheduler,
        lognorm_t=True,
        timeshift=1.0,
    )

    ema_tracker = SimpleEMA(decay=0.9999)
    optimizer = partial(torch.optim.AdamW, lr=args.lr, weight_decay=0.0)

    model = LightningModel(
        vae=vae,
        conditioner=conditioner,
        denoiser=denoiser,
        diffusion_trainer=trainer,
        diffusion_sampler=sampler,
        ema_tracker=ema_tracker,
        optimizer=optimizer,
        lr_scheduler=None,
        eval_original_model=False,
        sparsity=args.sparsity,
        cond_fraction=args.cond_fraction,
    )
    return model


def build_datamodule(args):
    train_dataset = PixCIFAR10(root="./data", train=True, random_flip=True, download=True)

    eval_dataset = CIFAR10RandomNDataset(
        num_classes=args.num_classes,
        latent_shape=(3, 32, 32),
        max_num_instances=1000,
    )
    pred_dataset = CIFAR10RandomNDataset(
        num_classes=args.num_classes,
        latent_shape=(3, 32, 32),
        max_num_instances=1000,
    )

    datamodule = DataModule(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        pred_dataset=pred_dataset,
        train_batch_size=args.batch_size,
        train_num_workers=args.num_workers,
        pred_batch_size=64,
        pred_num_workers=2,
    )
    return datamodule


def main():
    args = parse_args()

    print("=" * 60)
    print("CIFAR-10 Training with Options A/B/C/D")
    print("=" * 60)
    print(f"Option A (sfc_unified_coords): {args.option_a}")
    print(f"Option B (sfc_spatial_bias): {args.option_b}")
    print(f"Option C (sfc_attn_temperature): {args.option_c} (temp={args.attn_temperature if args.option_c else 1.0})")
    print(f"Option D (decoder_pixel_coords): {args.option_d}")
    print(f"Sparsity: {args.sparsity}, Cond Fraction: {args.cond_fraction}")
    print(f"Config: {vars(args)}")
    print()

    output_dir = Path(args.output_dir) / f"exp_{args.exp_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print("\nBuilding model...")
    model = build_model(args)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nBuilding datamodule...")
    datamodule = build_datamodule(args)

    # Use CSVLogger (no tensorboard)
    logger = CSVLogger(save_dir=str(output_dir), name="logs")

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            every_n_train_steps=args.save_every_n_steps,
            save_top_k=-1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        SaveImagesHook(save_dir="val", save_compressed=True),
        SparseReconProgressCallback(
            data_root="./data",
            out_dir=output_dir,
            sample_every_n_steps=args.sample_every_n_steps,
            sample_batch_size=args.sample_batch_size,
            base_res=args.image_size,
            superres_factor=4,
        ),
    ]

    trainer = Trainer(
        default_root_dir=str(output_dir),
        accelerator="auto",
        devices=args.devices,
        precision=args.precision,
        max_steps=args.max_steps,
        check_val_every_n_epoch=args.val_every_n_epochs,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
        logger=logger,
        callbacks=callbacks,
    )

    print("\nStarting training...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {output_dir / 'checkpoints'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
