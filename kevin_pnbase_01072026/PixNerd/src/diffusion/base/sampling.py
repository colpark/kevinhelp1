# src/diffusion/base/sampling.py

from typing import Union, List
import torch
import torch.nn as nn
from typing import Callable
from src.diffusion.base.scheduling import BaseScheduler

class BaseSampler(nn.Module):
    def __init__(self,
                 scheduler: BaseScheduler = None,
                 guidance_fn: Callable = None,
                 num_steps: int = 250,
                 guidance: Union[float, List[float]] = 1.0,
                 *args,
                 **kwargs
        ):
        super(BaseSampler, self).__init__()
        self.num_steps = num_steps
        self.guidance = guidance
        self.guidance_fn = guidance_fn
        self.scheduler = scheduler

    # NEW signature: allow cond_mask/x_cond but keep them optional
    def _impl_sampling(self, net, noise, condition, uncondition,
                       cond_mask=None, x_cond=None):
        raise NotImplementedError

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(self,
                net,
                noise,
                condition,
                uncondition,
                return_x_trajs: bool = False,
                return_v_trajs: bool = False,
                cond_mask=None,
                x_cond=None):
        """
        Wrapper around the concrete sampler implementation.

        return_x_trajs / return_v_trajs are kept in the same positions as before
        to avoid breaking existing positional calls.

        cond_mask / x_cond are for sparse-conditioned sampling and default to None.
        """
        x_trajs, v_trajs = self._impl_sampling(
            net,
            noise,
            condition,
            uncondition,
            cond_mask=cond_mask,
            x_cond=x_cond,
        )

        if return_x_trajs and return_v_trajs:
            return x_trajs[-1], x_trajs, v_trajs
        elif return_x_trajs:
            return x_trajs[-1], x_trajs
        elif return_v_trajs:
            return x_trajs[-1], v_trajs
        else:
            return x_trajs[-1]
