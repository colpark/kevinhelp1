from typing import Union, List, Optional

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


    def _impl_sampling(self, net, noise, condition, uncondition,
                       cond_mask: Optional[torch.Tensor] = None,
                       x_cond: Optional[torch.Tensor] = None):
        raise NotImplementedError

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(self, net, noise, condition, uncondition,
                return_x_trajs=False, return_v_trajs=False,
                cond_mask: Optional[torch.Tensor] = None,
                x_cond: Optional[torch.Tensor] = None):
        """
        Args:
            net: The denoising network
            noise: Starting noise tensor [B,C,H,W]
            condition: Conditioning embeddings
            uncondition: Unconditional embeddings
            return_x_trajs: Whether to return trajectory of x values
            return_v_trajs: Whether to return trajectory of v values
            cond_mask: [B,1,H,W] binary mask where 1=conditioned pixel
            x_cond: [B,C,H,W] clean pixel values at conditioned locations
        """
        x_trajs, v_trajs = self._impl_sampling(
            net, noise, condition, uncondition,
            cond_mask=cond_mask, x_cond=x_cond
        )
        if return_x_trajs and return_v_trajs:
            return x_trajs[-1], x_trajs, v_trajs
        elif return_x_trajs:
            return x_trajs[-1], x_trajs
        elif return_v_trajs:
            return x_trajs[-1], v_trajs
        else:
            return x_trajs[-1]


