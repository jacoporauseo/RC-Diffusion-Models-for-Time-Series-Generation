import torch
import torch.nn as nn
import numpy as np 

# TODO: to add beta_tilde? 


class BaseScheduler(nn.Module):
    """
    Variance scheduler of DDPM. It defines the variance schedule 

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
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        beta_tilde = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # use register_buffer for cuda 
        self.register_buffer("beta", betas)
        self.register_buffer("alpha", alphas)
        self.register_buffer("alpha_bar", alphas_cumprod)
        self.register_buffer("alpha_bar_prev", alphas_cumprod_prev)
        self.register_buffer("beta_tilde", beta_tilde)

    def _extract(self, buffer_name: str, k: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Gather values from a named buffer at timesteps k, and reshape for
        broadcasting against a tensor of shape x_shape (e.g. (B, N) or (B, C, N)).
        """
        buf = getattr(self, buffer_name)          # (K,)
        out = buf.gather(0, k)                     # (B,)
        return out.reshape(k.shape[0], *((1,) * (len(x_shape) - 1)))

    def get_alpha(self, k, x_shape):
        return self._extract("alpha", k, x_shape)

    def get_alpha_bar(self, k, x_shape):
        return self._extract("alpha_bar", k, x_shape)

    def get_beta(self, k, x_shape):
        return self._extract("beta", k, x_shape)

    def get_beta_tilde(self, k, x_shape):
        return self._extract("beta_tilde", k, x_shape)