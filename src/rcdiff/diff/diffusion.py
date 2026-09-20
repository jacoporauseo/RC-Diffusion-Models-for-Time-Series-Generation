from abc import ABC, abstractmethod 
import torch 
from typing import Tuple 
from rcdiff.diff.scheduler import BaseScheduler

class Diffusion(ABC):

    def __init__(self, scheduler : BaseScheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    @abstractmethod
    def q_sample(self, x_0 : torch.Tensor, k : torch.Tensor | int) -> Tuple[torch.Tensor, torch.Tensor]:
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

    # @abstractmethod
    def p_sample(self,): 
        """ 
        Reverse posterior
        """
        pass 


def extract(input, t: torch.Tensor, x: torch.Tensor):
    if t.ndim == 0:
        t = t.unsqueeze(0)
    shape = x.shape
    t = t.long().to(input.device)
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)