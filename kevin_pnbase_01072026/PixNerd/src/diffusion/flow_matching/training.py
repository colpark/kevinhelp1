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

def _bcast(v: torch.Tensor, B: int, device, dtype):
    """
    Ensure (B,) scheduler outputs broadcast over (B,C,H,W) as (B,1,1,1).
    If scheduler already returns (B,1,1,1), this is a no-op.
    """
    if v is None:
        return None
    v = v.to(device=device, dtype=dtype)
    if v.dim() == 1 and v.shape[0] == B:
        return v.view(B, 1, 1, 1)
    return v

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
        Correct FM training with sparse conditioning:

        - Build the *standard* FM state: x_t = alpha*x + sigma*noise
        - Optionally clamp only cond pixels to be clean hints (x)
        - Predict v_t = dalpha*x + dsigma*noise
        - Compute loss ONLY on target_mask pixels
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

        # scheduler values
        alpha  = _bcast(self.scheduler.alpha(t),  B, device, dtype)
        dalpha = _bcast(self.scheduler.dalpha(t), B, device, dtype)
        sigma  = _bcast(self.scheduler.sigma(t),  B, device, dtype)
        dsigma = _bcast(self.scheduler.dsigma(t), B, device, dtype)

        # standard FM state + target
        x_t = alpha * x + sigma * noise
        v_t = dalpha * x + dsigma * noise

        # clamp ONLY conditioning pixels to clean GT
        if cond_mask is not None:
            cond_mask = cond_mask.to(device=device, dtype=dtype)  # (B,1,H,W)
            x_t = cond_mask * x + (1.0 - cond_mask) * x_t

        # model prediction (keep passing cond_mask if your net uses it)
        out = net(x_t, t, y, cond_mask=cond_mask)

        # loss weight (and broadcast safely)
        w_raw = self.loss_weight_fn(
            alpha.view(B),  # (B,)
            sigma.view(B),  # (B,)
        )
        weight = _bcast(w_raw if torch.is_tensor(w_raw) else torch.tensor(w_raw, device=device, dtype=dtype), B, device, dtype)

        loss_sq = (out - v_t) ** 2
        loss_sq = loss_sq * weight  # (B,C,H,W)

        # restrict loss to target pixels only
        if target_mask is not None:
            target_mask = target_mask.to(device=device, dtype=dtype)  # (B,1,H,W)
            mask_expanded = target_mask.expand(-1, loss_sq.shape[1], -1, -1)  # (B,C,H,W)

            loss_sq = loss_sq * mask_expanded
            denom = mask_expanded.sum()
            loss = loss_sq.sum() / (denom + 1e-8)
        else:
            loss = loss_sq.mean()

        return dict(loss=loss)
