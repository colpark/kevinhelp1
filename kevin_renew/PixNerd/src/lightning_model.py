from typing import Callable, Iterable, Any, Optional, Union, Sequence, Mapping, Dict
import os.path
import copy
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from lightning.pytorch.callbacks import Callback

from src.models.autoencoder.base import BaseAE, fp2uint8
from src.models.conditioner.base import BaseConditioner
from src.utils.model_loader import ModelLoader
from src.callbacks.simple_ema import SimpleEMA
from src.diffusion.base.sampling import BaseSampler
from src.diffusion.base.training import BaseTrainer
from src.utils.no_grad import no_grad, filter_nograd_tensors
from src.utils.copy import copy_params

EMACallable = Callable[[nn.Module, nn.Module], SimpleEMA]
OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], LRScheduler]


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        vae: BaseAE,
        conditioner: BaseConditioner,
        denoiser: nn.Module,
        diffusion_trainer: BaseTrainer,
        diffusion_sampler: BaseSampler,
        ema_tracker: SimpleEMA = None,
        optimizer: OptimizerCallable = None,
        lr_scheduler: LRSchedulerCallable = None,
        eval_original_model: bool = False,
        # --- NEW: sparsity-conditioning hyperparameters ---
        sparsity: float = 0.0,       # total observed density (cond + target), e.g. 0.4 = 40%
        cond_fraction: float = 0.5,  # fraction of observed points used as conditioning
    ):
        super().__init__()
        self.vae = vae
        self.conditioner = conditioner
        self.denoiser = denoiser
        self.ema_denoiser = copy.deepcopy(self.denoiser)
        self.diffusion_sampler = diffusion_sampler
        self.diffusion_trainer = diffusion_trainer
        self.ema_tracker = ema_tracker
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.eval_original_model = eval_original_model

        # NEW: store sparsity hyperparameters
        self.sparsity = float(sparsity)
        self.cond_fraction = float(cond_fraction)

        self._strict_loading = False

    def configure_model(self) -> None:
        self.trainer.strategy.barrier()
        copy_params(src_model=self.denoiser, dst_model=self.ema_denoiser)

        # disable grad for conditioner and vae
        no_grad(self.conditioner)
        no_grad(self.vae)
        # no_grad(self.diffusion_sampler)
        no_grad(self.ema_denoiser)

        # torch.compile
        self.denoiser.compile()
        self.ema_denoiser.compile()

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return [self.ema_tracker]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params_denoiser = filter_nograd_tensors(self.denoiser.parameters())
        params_trainer = filter_nograd_tensors(self.diffusion_trainer.parameters())
        params_sampler = filter_nograd_tensors(self.diffusion_sampler.parameters())
        param_groups = [
            {"params": params_denoiser},
            {"params": params_trainer},
            {"params": params_sampler, "lr": 1e-3},
        ]
        optimizer: torch.optim.Optimizer = self.optimizer(param_groups)
        if self.lr_scheduler is None:
            return dict(optimizer=optimizer)
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return dict(
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )

    def on_validation_start(self) -> None:
        self.ema_denoiser.to(torch.float32)

    def on_predict_start(self) -> None:
        self.ema_denoiser.to(torch.float32)

    # sanity check before training start
    def on_train_start(self) -> None:
        self.ema_denoiser.to(torch.float32)
        self.ema_tracker.setup_models(net=self.denoiser, ema_net=self.ema_denoiser)

    def training_step(self, batch, batch_idx):
        x, y, metadata = batch

        # Encode and build class conditioning without grad
        with torch.no_grad():
            x = self.vae.encode(x)  # latent (or identity) space, shape (B,C,H,W)
            condition, uncondition = self.conditioner(y, metadata)

        # Build sparsity masks in latent space if enabled
        if self.sparsity > 0.0:
            cond_mask, target_mask = self._make_sparsity_masks(x)
        else:
            cond_mask, target_mask = None, None

        # Pass masks into diffusion_trainer (will be used to:
        # - freeze cond pixels along the flow
        # - restrict loss to target pixels)
        loss = self.diffusion_trainer(
            self.denoiser,
            self.ema_denoiser,
            self.diffusion_sampler,
            x,
            condition,
            uncondition,
            metadata,
            cond_mask=cond_mask,
            target_mask=target_mask,
        )

        self.log_dict(loss, prog_bar=True, on_step=True, sync_dist=False)
        return loss["loss"]

    def predict_step(self, batch, batch_idx):
        xT, y, metadata = batch
        with torch.no_grad():
            condition, uncondition = self.conditioner(y)

        # sample images
        if self.eval_original_model:
            samples = self.diffusion_sampler(self.denoiser, xT, condition, uncondition)
        else:
            samples = self.diffusion_sampler(self.ema_denoiser, xT, condition, uncondition)

        samples = self.vae.decode(samples)
        # fp32 -1,1 -> uint8 0,255
        samples = fp2uint8(samples)
        return samples

    def validation_step(self, batch, batch_idx):
        samples = self.predict_step(batch, batch_idx)
        return samples

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)
        self.denoiser.state_dict(
            destination=destination,
            prefix=prefix + "denoiser.",
            keep_vars=keep_vars,
        )
        self.ema_denoiser.state_dict(
            destination=destination,
            prefix=prefix + "ema_denoiser.",
            keep_vars=keep_vars,
        )
        self.diffusion_trainer.state_dict(
            destination=destination,
            prefix=prefix + "diffusion_trainer.",
            keep_vars=keep_vars,
        )
        return destination

    # --- NEW: helper to build cond/target masks in latent space ---
    def _make_sparsity_masks(self, x: torch.Tensor):
        """
        Build per-sample conditioning and target masks in latent space.

        x: (B, C, H, W)
        returns:
            cond_mask:   (B, 1, H, W) float {0,1}
            target_mask: (B, 1, H, W) float {0,1}
        """
        B, C, H, W = x.shape
        device = x.device
        total = H * W

        # Total observed pixels
        k_obs = int(round(self.sparsity * total))
        k_obs = max(0, min(total, k_obs))

        if k_obs == 0:
            # Degenerate case: no observed pixels
            cond_mask = torch.zeros(B, 1, H, W, device=device)
            target_mask = torch.zeros_like(cond_mask)
            return cond_mask, target_mask

        # Cond fraction split
        k_cond = int(round(self.cond_fraction * k_obs))
        k_cond = max(0, min(k_obs, k_cond))
        k_target = k_obs - k_cond

        cond_mask = torch.zeros(B, 1, H, W, device=device,dtype=x.dtype)
        target_mask = torch.zeros_like(cond_mask)

        flat_idx = torch.arange(total, device=device)

        for b in range(B):
            perm = flat_idx[torch.randperm(total, device=device)]
            obs_idx = perm[:k_obs]

            if k_cond > 0:
                cond_idx = obs_idx[:k_cond]
            else:
                cond_idx = obs_idx[:0]

            if k_target > 0:
                target_idx = obs_idx[k_cond : k_cond + k_target]
            else:
                target_idx = obs_idx[:0]

            cond_mask[b].view(-1)[cond_idx] = 1.0
            target_mask[b].view(-1)[target_idx] = 1.0

        return cond_mask, target_mask
