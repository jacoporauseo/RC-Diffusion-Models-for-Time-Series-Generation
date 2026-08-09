from abc import ABC, abstractmethod 
import torch 
from typing import Tuple 


class NoiseProcess(ABC):

    @abstractmethod
    def q_sample(self, x: torch.Tensor, k : int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from `q(x_t^{k}|x_t)`. From DDPM of `Ho et al. (2020)`: \n
                                x_t^k ~ q(x_t^k | x_t) 
        By using the reparameterization trick this distribution is: \n
                        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε    ε ~ N(0, I_N)

        Parameters
        ---------
            x (torch.Tensor) : time series observations `x_t` of shape `[T, N]`
            k (torch.Tensor) : torch tensor of `long` type that gives the diffusion step 
                                    
        Returns
        -------
            x_k (torch.Tensor) : noisy version `x_t^k` in DDPM.
            epsilon (torch.Tensor) : noise added to the original sample

        """
        pass 

    @abstractmethod
    def marginal_prob(self, x_0: torch.Tensor, t):
        """Returns (mean, std) of p(x_t | x_0) — used by both q_sample and loss weighting."""
        ...

    @abstractmethod
    def prior_sampling(self, shape) -> torch.Tensor:
        """Sample x_T ~ N(0, I) or whatever the terminal distribution is."""
        ...

    @abstractmethod
    def sample_timesteps(self, batch_size: int):
        """Discrete: randint(0,K). Continuous: uniform(eps, 1)."""
        ...