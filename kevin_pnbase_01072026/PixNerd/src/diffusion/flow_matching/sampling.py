# src/diffusion/flow_matching/sampling.py

import torch
import logging
from typing import Callable

from src.diffusion.base.guidance import *
from src.diffusion.base.scheduling import *
from src.diffusion.base.sampling import *

logger = logging.getLogger(__name__)


def shift_respace_fn(t, shift=3.0):
    return t / (t + (1 - t) * shift)

def ode_step_fn(x, v, dt, s, w):
    return x + v * dt

def sde_mean_step_fn(x, v, dt, s, w):
    return x + v * dt + s * w * dt

def sde_step_fn(x, v, dt, s, w):
    return x + v*dt + s * w* dt + torch.sqrt(2*w*dt)*torch.randn_like(x)

def sde_preserve_step_fn(x, v, dt, s, w):
    return x + v*dt + 0.5*s*w* dt + torch.sqrt(w*dt)*torch.randn_like(x)


class EulerSampler(BaseSampler):
    def __init__(
            self,
            w_scheduler: BaseScheduler = None,
            timeshift=1.0,
            guidance_interval_min: float = 0.0,
            guidance_interval_max: float = 1.0,
            step_fn: Callable = ode_step_fn,
            last_step=None,
            last_step_fn: Callable = ode_step_fn,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.step_fn = step_fn
        self.last_step = last_step
        self.last_step_fn = last_step_fn
        self.w_scheduler = w_scheduler
        self.timeshift = timeshift
        self.guidance_interval_min = guidance_interval_min
        self.guidance_interval_max = guidance_interval_max

        if self.last_step is None or self.num_steps == 1:
            self.last_step = 1.0 / self.num_steps

        timesteps = torch.linspace(0.0, 1 - self.last_step, self.num_steps)
        timesteps = torch.cat([timesteps, torch.tensor([1.0])], dim=0)
        self.timesteps = shift_respace_fn(timesteps, self.timeshift)

        assert self.last_step > 0.0
        assert self.scheduler is not None
        assert self.w_scheduler is not None or self.step_fn in [ode_step_fn, ]
        if self.w_scheduler is not None and self.step_fn == ode_step_fn:
            logger.warning("current sampler is ODE sampler, but w_scheduler is enabled")

    # NEW: accept cond_mask and x_cond
    def _impl_sampling(self,
                       net,
                       noise,
                       condition,
                       uncondition,
                       cond_mask=None,
                       x_cond=None):
        """
        sampling process of Euler sampler.

        If cond_mask & x_cond are provided, we clamp x_t on those pixels at every
        step so x_t(cond) == x_cond(cond) for all t.
        """
        batch_size = noise.shape[0]
        steps = self.timesteps.to(noise.device, noise.dtype)
        cfg_condition = torch.cat([uncondition, condition], dim=0)

        x = noise

        # prepare mask and cond values if provided
        mask_expanded = None
        if cond_mask is not None and x_cond is not None:
            cond_mask = cond_mask.to(noise.device, noise.dtype)     # (B,1,H,W)
            x_cond = x_cond.to(noise.device, noise.dtype)           # (B,C,H,W)
            if cond_mask.dim() == 4 and cond_mask.shape[1] == 1 and x.dim() == 4:
                mask_expanded = cond_mask.expand(-1, x.shape[1], -1, -1)  # (B,C,H,W)
            else:
                mask_expanded = cond_mask

            # clamp initial state: GT on cond, noise elsewhere
            x = mask_expanded * x_cond + (1.0 - mask_expanded) * x

        x_trajs = [x]
        v_trajs = []

        for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            dt = t_next - t_cur
            t_cur = t_cur.repeat(batch_size)

            sigma = self.scheduler.sigma(t_cur)
            dalpha_over_alpha = self.scheduler.dalpha_over_alpha(t_cur)
            dsigma_mul_sigma = self.scheduler.dsigma_mul_sigma(t_cur)

            if self.w_scheduler:
                w = self.w_scheduler.w(t_cur)
            else:
                w = 0.0

            cfg_x = torch.cat([x, x], dim=0)
            cfg_t = t_cur.repeat(2)

            cfg_cond_mask = None
            if cond_mask is not None:
                cfg_cond_mask = torch.cat([cond_mask, cond_mask], dim=0)
    
            out = net(cfg_x, cfg_t, cfg_condition, cond_mask=cfg_cond_mask)

            if self.guidance_interval_min < t_cur[0] < self.guidance_interval_max:
                out = self.guidance_fn(out, self.guidance)
            else:
                out = self.guidance_fn(out, 1.0)

            v = out
            s = ((1 / dalpha_over_alpha) * v - x) / (
                sigma**2 - (1 / dalpha_over_alpha) * dsigma_mul_sigma
            )

            if i < self.num_steps - 1:
                x = self.step_fn(x, v, dt, s=s, w=w)
            else:
                x = self.last_step_fn(x, v, dt, s=s, w=w)

            # enforce conditioning after each step
            if mask_expanded is not None:
                x = mask_expanded * x_cond + (1.0 - mask_expanded) * x

            x_trajs.append(x)
            v_trajs.append(v)

        v_trajs.append(torch.zeros_like(x))
        return x_trajs, v_trajs
