import torch 
import torch
import torch.nn as nn 
import torch.nn.functional as F
import time 
from tqdm import tqdm 
import copy 
import math 
from dataclasses import asdict, replace
from pathlib import Path

from rcdiff.models.encoders import ESN
from rcdiff.diff import Diffusion
from rcdiff.models.config import TrainConfig
from rcdiff.models.tsdiff.base_diff import TSDiffusionModel
from rcdiff.models.denoisers.rwdn import RWDenoisingNetwork



class RCDiff(TSDiffusionModel): 

    def __init__(self, 
                 diff : Diffusion, 
                 esn : ESN,
                 denoiser : RWDenoisingNetwork, 
                 spec : TrainConfig
                 ): 
        super().__init__(diff=diff, esn=esn, denoiser=denoiser) 
        self.spec = spec 
        self.transform = {}
        self.history = None

    def fit(self, 
            x : torch.Tensor, 
            c : torch.Tensor | None,
            **kwargs):
        
        T, N = x.shape
        x, c = self._build_conditioning_set(x, c)
    
        assert self.diff.scheduler.K == self.denoiser.K, f"Inconsistent number of diffusion steps! Got {self.diff.scheduler.K} and {self.denoiser.K}"
    
        
        s = self.esn.get_states(X=x, include_initial=True) 
        x = x[self.spec.t0:T] # [x_{t_0}, ..., x_{T}]
        s = s[self.spec.t0:T] # [s_{t_0-1}, ..., x_{T-1}]
    
        losses_before = []
        losses_after  = []

        start = time.time()

        for k in range(self.diff.scheduler.K):

            k_tensor = torch.full((T-self.spec.t0,), fill_value = k, dtype=torch.long)
            x_k, epsilon = self.diff.q_sample(x_0 = x, k = k_tensor) 
    
            eps_hat = self.denoiser(x_k = x_k, s = s, k = k_tensor)                    
            loss_bt = F.mse_loss(eps_hat, epsilon)
            losses_before.append(loss_bt.item())
    
            self.denoiser.fit(x_k = x, s = s, epsilon_k = epsilon, k = k, method = "Ridge")
    
            #Check Loss after fitting
            eps_hat2 = Z @ model.W[k]   # use the forward... here just for debug was easy to reycle Z since it is the same.
            loss_at  = F.mse_loss(eps_hat2, epsilon)
            losses_after.append(loss_at.item())
    
            if k % 25 == 0:
                print(f"k={k:3d} | Loss before: {loss_bt.item():.4f} | Loss after: {loss_at.item():.6f}")
    
        elapsed = time.time() - start
        print(f"\nTraining complete :=) ! Time: {elapsed:.2f}s")
        return losses_before, losses_after, elapsed

    def conditional_distribution(self, 
                                 x : torch.Tensor,
                                 c : torch.Tensor | None = None,
                                 n_samples : int = 1
                                 ): 
        r""" 
        Sample from:
        
        ``p(x_{T+1} | s_{T}) \approx p(x_t | \mathcal{f}(x_{T}))``

        where `s_{T} = Φ(x_{T}, s_{T-1})`. 

        Parameters
        ---------
            x (torch.Tensor) : target series of shape (T, N)
            c (torch.Tensor) : exogenous covariates of shape (T, M)

        Returns
        -------
            x_hat (torch.Tensor) : samples from the reverse process of shape `(n_samples, N)`
        """
        pass



