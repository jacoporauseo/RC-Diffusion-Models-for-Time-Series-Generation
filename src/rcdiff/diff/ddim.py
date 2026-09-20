import torch 
import numpy as np 
from typing import Tuple, List, Set, 
from rcdiff.diff.diffusion import Diffusion 
from rcdiff.diff.scheduler import BaseScheduler

class DDIM(Diffusion):
    """
    A high-level wrapper of DDIM.
    """
    def __init__(self, scheduler : BaseScheduler): 
        super().__init__(scheduler=scheduler)

    @torch.no_grad()
    def q_sample(self, x_0 : torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from `q(x_t^{k}|x_t)`. The DDIM (Song et al. (2021)) forward process is the same as the DDPM 
        forward process: 
                    x_t^k = sqrt(ᾱ_k) * x_t + sqrt(1 - ᾱ_k) * ε    ε ~ N(0, I_N)

        Parameters
        ---------
            x_0 (torch.Tensor) : time series observations `x_t` of shape `[T, N]` or a batch of 
                                 shape `(B, N)
            k (torch.Tensor) : torch tensor of `long` type that gives the diffusion step 
                                    
        Returns
        --------
            x_k (torch.Tensor) : noisy version `x_t^k` in DDPM.
            epsilon (torch.Tensor) : noise added to the original sample
        """
            
        epsilon = torch.randn_like(x_0)
        alpha_bar_k = self.scheduler.get_alpha_bar(k, x_shape=x_0.shape)
        # alpha_bar_k = self.scheduler.alpha_bar[k].view(-1, 1) #type: ignore
        x_t = torch.sqrt(alpha_bar_k) * x_0 + torch.sqrt(1 - alpha_bar_k) * epsilon
        return x_t, epsilon

    @torch.no_grad()
    def p_sample(self, 
                 model, 
                 x_k: torch.Tensor, 
                 k : int, 
                 k_prev : int, 
                 s : torch.Tensor,
                 eta : float = 1.0
                 ) -> torch.Tensor:
        r"""
        DDIM reverse process step. Get x^k_t given x^{k-1}_t and s. 
        NOTE: alpha_k in DDIM is alpha_bar in DDPM!!!

        Parameters
        ---------
            x_k (torch.Tensor)  : the vector x^k_t of the noised version at step k
            s (torch.Tensor)    : the econded vector s_{t-1} \in \mathcal{R}^S of shape [S,1] or [S,]
            k (int)             : diffusion step k associated with x_k
            k_prev (int)        : diffusion step associated with the next step of the DDIM chain
            eta (float)         : deterministic coefficient of DDIM in [0,1]
            model (nn.Module)   : denoisers model (U-Net, MLP, RWDN, etc.)

        Returns 
        ------
            x_prev (torch.Tensor) : the vector x^{k_l}_t of the noised version at step k_l < k
        """
        k_tensor = torch.tensor([k], device=x_k.device).expand(x_k.shape[0])
        eps = model(x_k, k_tensor, s)

        alpha_bar_k = self.scheduler.alpha_bar[k] # type: ignore
        if k_prev >= 0:
            alpha_bar_k_prev = self.scheduler.alpha_bar[k_prev] # type: ignore
        else: 
            torch.tensor(1.0, device=x_k.device, dtype=x_k.dtype)

        # NOTE: cannot use precomputed since when skipping steps it is not the same 
        sigma_k = eta * torch.sqrt((1 - alpha_bar_k_prev) / (1 - alpha_bar_k) * (1 - alpha_bar_k / alpha_bar_k_prev))

        predicted_x_0 = (x_k - torch.sqrt(1 - alpha_bar_k) * eps) / torch.sqrt(alpha_bar_k)
        direction_to_x_t = torch.sqrt(1 - alpha_bar_k_prev - sigma_k**2) * eps

        if eta > 0.0 and k_prev >= 0:
            z = torch.randn_like(x_k)
            random_noise = sigma_k * z
        else:
            random_noise = 0.0

        x_prev = torch.sqrt(alpha_bar_k_prev) * predicted_x_0 + direction_to_x_t + random_noise
        return x_prev