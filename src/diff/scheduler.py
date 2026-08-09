import torch
import torch.nn as nn
import numpy as np 

class BaseScheduler(nn.Module):
    """
    Variance scheduler of DDPM.

    Parameters 
    ---------
        K (int) : number of diffusion steps
        beta_1 (float) : starting value for the variance schedule *Default* 1e-4 in TimeGrad)
        beta_K (float) : final value for the variance schedule. *Default* 0.1 in TimeGrad)
        s (float) : adj in the cosine schedule
        mode (str) : available `['linear', 'cosine', 'quad']`. *Default* linear as in TimeGrad
    """
    def __init__(
        self,
        K: int,
        beta_1: float = 1e-4,
        beta_K : float = 0.1, # Rasul et al. from 1*10^-4 to 0.1 (different from DDPM)
        s : float = 0.008,
        mode: str = "linear",
        ):
        super().__init__()
        self.K = K
        self.timesteps = torch.from_numpy(
            np.arange(0, self.K)[::-1].copy().astype(np.int64)
            )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_K, steps=K)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_K**0.5, K) ** 2
            )
        elif mode == "cosine":
            """Cosine schedule from Imporved DDPM"""
            k = torch.arange(0, self.K + 1).float()
            f_k = torch.cos((k / self.K + s) / (1 + s) * torch.pi / 2) ** 2
            alpha_bar = f_k / f_k[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = torch.clip(betas, 0.0001, 0.999)

        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = torch.ones(betas.shape) - betas
        alphas_cumprod = torch.cumprod(alphas, dim = 0)

        # use register_buffer for cuda 
        self.register_buffer("betas", betas)
        self.register_buffer("alpha", alphas)
        self.register_buffer("alpha_bar", alphas_cumprod)