import torch
from typing import Callable
from src.diffusion.base.training import BaseTrainer
from src.diffusion.base.scheduling import BaseScheduler

def inverse_sigma(alpha, sigma):
    return 1 / sigma**2

def snr(alpha, sigma):
    return alpha / sigma

def minsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha / sigma, min=threshold)

def maxsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha / sigma, max=threshold)

def constant(alpha, sigma):
    return 1


def time_shift_fn(t, timeshift=1.0):
    return t / (t + (1 - t) * timeshift)


class FlowMatchingTrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn: Callable = constant,
            lognorm_t: bool = False,
            timeshift: float = 1.0,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.timeshift = timeshift
        self.loss_weight_fn = loss_weight_fn

    def _impl_trainstep(
        self,
        net,
        ema_net,
        solver,
        x,
        y,
        metadata=None,
        cond_mask=None,
        target_mask=None,
    ):
        """
        Flow-matching training step with optional sparse conditioning.

        x:           full GT image/latent (B,C,H,W) – BUT
                     the net will only see GT on cond+target pixels.
        y:           conditioning (e.g., class embedding)
        cond_mask:   (B,1,H,W) – hint pixels (no loss)
        target_mask: (B,1,H,W) – supervision pixels (loss region)
        """
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        # sample times t
        if self.lognorm_t:
            t = torch.randn(B, device=device, dtype=dtype).sigmoid()
        else:
            t = torch.rand(B, device=device, dtype=dtype)
        t = time_shift_fn(t, self.timeshift)

        noise = torch.randn_like(x)

        # scheduler values (broadcastable to x)
        alpha = self.scheduler.alpha(t)
        dalpha = self.scheduler.dalpha(t)
        sigma = self.scheduler.sigma(t)
        dsigma = self.scheduler.dsigma(t)
        w = self.scheduler.w(t)

        # -------------------------
        # OBSERVED MASK: where GT is allowed to appear
        # -------------------------
        obs_mask = None
        if (cond_mask is not None) or (target_mask is not None):
            if cond_mask is None:
                obs_mask = target_mask
            elif target_mask is None:
                obs_mask = cond_mask
            else:
                obs_mask = torch.clamp(cond_mask + target_mask, 0.0, 1.0)
            obs_mask = obs_mask.to(dtype)

        # base components of the flow (full data)
        base_data = alpha * x          # GT term
        base_noise = sigma * noise     # noise term
        v_full = dalpha * x + dsigma * noise

        if obs_mask is not None:
            # Only on obs pixels do we inject GT (via alpha * x).
            # Everywhere else is pure noise.
            x_t = base_noise + obs_mask * base_data

            # Optionally: make cond region perfectly clean hints
            if cond_mask is not None:
                x_t = cond_mask * x + (1.0 - cond_mask) * x_t

            # We only ever care about v_t on observed pixels.
            v_t = obs_mask * v_full
        else:
            # No sparsity: standard FM
            x_t = base_data + base_noise
            v_t = v_full

        # model prediction
        out = net(x_t, t, y)

        # base weighted squared error
        weight = self.loss_weight_fn(alpha, sigma)
        loss_sq = (out - v_t) ** 2      # (B,C,H,W)
        loss_sq = weight * loss_sq      # broadcast weight

        # -------------------------
        # Restrict loss to target_mask
        # -------------------------
        if target_mask is not None:
            if target_mask.dim() == 4 and target_mask.shape[1] == 1 and loss_sq.dim() == 4:
                mask_expanded = target_mask.expand(-1, loss_sq.shape[1], -1, -1)
            else:
                mask_expanded = target_mask

            loss_sq = loss_sq * mask_expanded

            denom = mask_expanded.sum() * loss_sq.shape[1]
            loss = loss_sq.sum() / (denom + 1e-8)
        else:
            # dense training fallback
            loss = loss_sq.mean()

        return dict(loss=loss)
