#!/usr/bin/env python3
"""
CIFAR-10 Class-Conditional Training Script

Train a PixNerd model on CIFAR-10 with heavy decoder for super-resolution.
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
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, CSVLogger

# Check if tensorboard is actually available (not just the logger class)
try:
    import tensorboard
    TENSORBOARD_AVAILABLE = True
    from lightning.pytorch.loggers import TensorBoardLogger
except (ImportError, ModuleNotFoundError):
    try:
        import tensorboardX
        TENSORBOARD_AVAILABLE = True
        from lightning.pytorch.loggers import TensorBoardLogger
    except (ImportError, ModuleNotFoundError):
        TENSORBOARD_AVAILABLE = False

from src.data.dataset.imagenet import PixImageNet
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


class SparseLightningModel(LightningModel):
    """
    Custom LightningModel that generates sparse conditioning masks during training.

    For SFC encoder training, we need to:
    1. Generate random sparse masks for each batch
    2. Pass the masks to the denoiser via metadata
    """
    def __init__(self, *args, sparsity: float = 0.4, cond_fraction: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparsity = sparsity
        self.cond_fraction = cond_fraction

    def _generate_sparse_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate sparse conditioning mask.

        Args:
            x: (B, C, H, W) input tensor

        Returns:
            cond_mask: (B, 1, H, W) binary mask where 1 = observed pixel
        """
        B, C, H, W = x.shape
        device = x.device

        # Total pixels to keep (sparse observation)
        total_keep = int(self.sparsity * H * W)

        # Generate random mask for each sample in batch
        cond_mask = torch.zeros(B, 1, H, W, device=device)

        for b in range(B):
            # Random pixel indices to keep
            indices = torch.randperm(H * W, device=device)[:total_keep]
            mask_flat = torch.zeros(H * W, device=device)
            mask_flat[indices] = 1.0
            cond_mask[b, 0] = mask_flat.view(H, W)

        return cond_mask

    def _make_sparsity_masks(self, x: torch.Tensor):
        """
        Generate sparse mask for visualization callback.
        Returns (cond_mask, target_mask) tuple.
        """
        cond_mask = self._generate_sparse_mask(x)
        # For visualization, target_mask is the complement (pixels to predict)
        target_mask = 1.0 - cond_mask
        return cond_mask, target_mask

    def training_step(self, batch, batch_idx):
        x, y, metadata = batch

        with torch.no_grad():
            x = self.vae.encode(x)
            condition, uncondition = self.conditioner(y, metadata)

            # Generate sparse mask for SFC encoder
            cond_mask = self._generate_sparse_mask(x)
            metadata['cond_mask'] = cond_mask
            # Store clean latent values for repaint-style conditioning
            # The trainer will use these at hint locations during training
            metadata['x_cond'] = x.clone()

        loss = self.diffusion_trainer(
            self.denoiser, self.ema_denoiser, self.diffusion_sampler,
            x, condition, uncondition, metadata
        )
        self.log_dict(loss, prog_bar=True, on_step=True, sync_dist=False)
        return loss["loss"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 class-conditional model")

    # Training config
    parser.add_argument("--max_steps", type=int, default=300000, help="Max training steps")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "imagenet"],
        help="Which dataset to use",
    )
    parser.add_argument(
        "--imagenet_root",
        type=str,
        default="/pscratch/sd/k/kevinval/datasets/imagenet256",
        help="Root folder for ImageNet-style data (class folders with jpgs)",
    )
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=32)

    # Model config (smaller defaults for CIFAR-10 32x32)
    parser.add_argument("--hidden_size", type=int, default=256, help="Transformer hidden size")
    parser.add_argument("--decoder_hidden_size", type=int, default=64)
    parser.add_argument("--num_encoder_blocks", type=int, default=4, help="Number of DiT encoder blocks")
    parser.add_argument("--num_decoder_blocks", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size (4 for 32x32 gives 64 patches)")
    parser.add_argument("--num_groups", type=int, default=4, help="Number of attention heads")

    # Sampler config
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--num_sample_steps", type=int, default=200)

    # Logging
    parser.add_argument("--exp_name", type=str, default="cifar10_c2i_superres_grid")
    parser.add_argument("--output_dir", type=str, default="./workdirs")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="pixnerd_cifar10")

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

    # Encoder selection
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="grid",
        choices=["grid", "perceiver", "sfc"],   # <-- CHANGED
    )
    parser.add_argument("--perceiver_num_latents", type=int, default=256)

    # ---- NEW: SFC knobs (only used when --encoder_type sfc) ----
    parser.add_argument("--sfc_curve", type=str, default="hilbert", choices=["hilbert", "zorder"])
    parser.add_argument("--sfc_group_size", type=int, default=8)
    parser.add_argument("--sfc_cross_depth", type=int, default=2)

    # ---- Ablation flags (Option A and B) ----
    parser.add_argument(
        "--sfc_unified_coords",
        action="store_true",
        help="Option A: Use shared coordinate embedder for both tokens and queries",
    )
    parser.add_argument(
        "--sfc_spatial_bias",
        action="store_true",
        help="Option B: Add spatial attention bias in cross-attention based on coordinate distance",
    )

    return parser.parse_args()


from lightning.pytorch.callbacks import Callback
import torch.nn.functional as F


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
        dataset: str = "cifar10",
    ):
        super().__init__()
        self.data_root = data_root
        self.out_dir = Path(out_dir)
        self.sample_every_n_steps = sample_every_n_steps
        self.sample_batch_size = sample_batch_size
        self.base_res = base_res
        self.superres_factor = superres_factor
        self.dataset = dataset.lower()

        self.progress_dir = self.out_dir / "progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)

        self._fixed_images = None
        self._fixed_labels = None
        self._build_fixed_batch()

    def _build_fixed_batch(self):
        if self.dataset == "cifar10":
            tfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            ds = CIFAR10(root=self.data_root, train=False, transform=tfm, download=True)
            idx = torch.arange(self.sample_batch_size)
            imgs = [ds[i][0] for i in idx]
            labels = [ds[i][1] for i in idx]

        elif self.dataset == "imagenet":
            ds = PixImageNet(root=self.data_root)
            idx = torch.arange(self.sample_batch_size)
            imgs, labels = [], []
            for i in idx:
                img, label, metadata = ds[int(i)]
                imgs.append(img)
                labels.append(label)
        else:
            raise ValueError(f"Unsupported dataset for progress callback: {self.dataset}")

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

        # Get latent representation and generate sparse mask for visualization
        x_latent = pl_module.vae.encode(images)
        cond_mask, _ = pl_module._make_sparsity_masks(x_latent)

        condition, uncondition = pl_module.conditioner(labels)

        # 1. Class-conditional generation (NO sparse conditioning - baseline)
        noise = torch.randn_like(x_latent)
        samples_class_only = pl_module.diffusion_sampler(
            pl_module.ema_denoiser,
            noise,
            condition,
            uncondition,
        )
        if isinstance(samples_class_only, tuple):
            samples_class_only = samples_class_only[0][-1]
        samples_class_only = pl_module.vae.decode(samples_class_only)

        # 2. Sparse-conditioned reconstruction (WITH sparse conditioning)
        # This uses the clean latent values at hint locations to guide generation
        noise = torch.randn_like(x_latent)
        samples_sparse_cond = pl_module.diffusion_sampler(
            pl_module.ema_denoiser,
            noise,
            condition,
            uncondition,
            cond_mask=cond_mask,
            x_cond=x_latent,  # Clean latent values at hint locations
        )
        if isinstance(samples_sparse_cond, tuple):
            samples_sparse_cond = samples_sparse_cond[0][-1]
        samples_sparse_cond = pl_module.vae.decode(samples_sparse_cond)

        step = trainer.global_step
        tag = f"step_{step:06d}"

        # Create sparse input visualization (ground truth with unobserved pixels blacked out)
        sparse_input = images.clone()
        # Upsample mask to image resolution if needed (for latent-space masks)
        if cond_mask.shape[-2:] != images.shape[-2:]:
            mask_vis = F.interpolate(cond_mask, size=images.shape[-2:], mode='nearest')
        else:
            mask_vis = cond_mask
        mask_expanded = mask_vis.expand_as(images)
        sparse_input = sparse_input * mask_expanded

        # 3. Super-resolution to 128x128
        scale = self.superres_factor
        H_hr = self.base_res * scale
        W_hr = self.base_res * scale

        # Create HR tensors with 32x32 hints at stride positions
        cond_mask_hr = torch.zeros((B, 1, H_hr, W_hr), device=device, dtype=x_latent.dtype)
        cond_mask_hr[:, :, ::scale, ::scale] = cond_mask

        x_cond_hr = torch.zeros((B, x_latent.shape[1], H_hr, W_hr), device=device, dtype=x_latent.dtype)
        x_cond_hr[:, :, ::scale, ::scale] = x_latent

        # Temporarily modify decoder_patch_scaling
        old_scales = {}
        for name, net in [("denoiser", pl_module.denoiser), ("ema_denoiser", pl_module.ema_denoiser)]:
            if hasattr(net, "decoder_patch_scaling_h"):
                old_scales[name] = (net.decoder_patch_scaling_h, net.decoder_patch_scaling_w)
                net.decoder_patch_scaling_h = scale
                net.decoder_patch_scaling_w = scale

        # Run diffusion at high resolution
        # Note: disable_spatial_bias=True reduces checkerboard artifacts at SR
        # by preventing overly-localized attention that causes patch boundary discontinuities
        noise_hr = torch.randn_like(x_cond_hr)
        samples_hr = pl_module.diffusion_sampler(
            pl_module.ema_denoiser,
            noise_hr,
            condition,
            uncondition,
            cond_mask=cond_mask_hr,
            x_cond=x_cond_hr,
            disable_spatial_bias=True,
        )
        if isinstance(samples_hr, tuple):
            samples_hr = samples_hr[0][-1]
        samples_hr = pl_module.vae.decode(samples_hr)

        # Restore original decoder_patch_scaling
        for name, net in [("denoiser", pl_module.denoiser), ("ema_denoiser", pl_module.ema_denoiser)]:
            if name in old_scales:
                h, w = old_scales[name]
                net.decoder_patch_scaling_h = h
                net.decoder_patch_scaling_w = w

        # Save all visualizations
        self._save_grid(images, self.progress_dir / f"{tag}_1_gt.png")
        self._save_grid(sparse_input, self.progress_dir / f"{tag}_2_sparse_input.png")
        self._save_grid(samples_class_only, self.progress_dir / f"{tag}_3_class_only.png")
        self._save_grid(samples_sparse_cond, self.progress_dir / f"{tag}_4_sparse_conditioned.png")
        self._save_grid(samples_hr, self.progress_dir / f"{tag}_5_superres_{H_hr}x{W_hr}.png")

        print(f"\n[Step {step}] Saved visualizations to {self.progress_dir}")
        print(f"  - 1_gt.png: Ground truth images ({self.base_res}x{self.base_res})")
        print(f"  - 2_sparse_input.png: Sparse input (observed pixels only)")
        print(f"  - 3_class_only.png: Class-conditional (no sparse hints)")
        print(f"  - 4_sparse_conditioned.png: Sparse-conditioned reconstruction")
        print(f"  - 5_superres_{H_hr}x{W_hr}.png: Super-resolution ({scale}x upscale)")

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

    denoiser = PixNerDiT(
        in_channels=3,
        patch_size=args.patch_size,
        num_groups=args.num_groups,
        hidden_size=args.hidden_size,
        decoder_hidden_size=args.decoder_hidden_size,
        num_encoder_blocks=args.num_encoder_blocks,
        num_decoder_blocks=args.num_decoder_blocks,
        num_classes=args.num_classes,
        encoder_type=args.encoder_type,
        perceiver_num_latents=args.perceiver_num_latents,
        # ---- SFC args ----
        sfc_curve=args.sfc_curve,
        sfc_group_size=args.sfc_group_size,
        sfc_cross_depth=args.sfc_cross_depth,
        # ---- Ablation flags (Option A and B) ----
        sfc_unified_coords=args.sfc_unified_coords,
        sfc_spatial_bias=args.sfc_spatial_bias,
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

    model = SparseLightningModel(
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
    if args.dataset == "cifar10":
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

    elif args.dataset == "imagenet":
        train_dataset = PixImageNet(root=args.imagenet_root)
        eval_dataset = train_dataset
        pred_dataset = train_dataset

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

    # Enable tensor core optimization for H100/A100
    torch.set_float32_matmul_precision('high')

    print("=" * 60)
    print("CIFAR-10 Class-Conditional Training with Heavy Decoder")
    print("=" * 60)
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

    if args.use_wandb:
        logger = WandbLogger(project=args.wandb_project, name=args.exp_name, save_dir=str(output_dir))
    elif TENSORBOARD_AVAILABLE:
        logger = TensorBoardLogger(save_dir=str(output_dir), name="logs")
    else:
        logger = CSVLogger(save_dir=str(output_dir), name="logs")
        print("Using CSVLogger (tensorboard not available)")

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            every_n_train_steps=args.save_every_n_steps,
            save_top_k=-1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        SaveImagesHook(save_dir="val", save_compressed=True),
    ]

    if args.dataset == "cifar10":
        progress_cb = SparseReconProgressCallback(
            data_root="./data",
            out_dir=output_dir,
            sample_every_n_steps=args.sample_every_n_steps,
            sample_batch_size=args.sample_batch_size,
            base_res=args.image_size,
            superres_factor=4,
            dataset="cifar10",
        )
    elif args.dataset == "imagenet":
        progress_cb = SparseReconProgressCallback(
            data_root=args.imagenet_root,
            out_dir=output_dir,
            sample_every_n_steps=args.sample_every_n_steps,
            sample_batch_size=args.sample_batch_size,
            base_res=args.image_size,
            superres_factor=2,
            dataset="imagenet",
        )
    else:
        progress_cb = None

    callbacks.append(progress_cb)

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
