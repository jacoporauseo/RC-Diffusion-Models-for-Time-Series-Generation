import torch 
import torch.nn as nn 
import torch.nn.functional as F
import time 
from tqdm import tqdm 
import pandas as pd 
import numpy as np 

from rcdiff.models.encoders import ESN
from rcdiff.diff import Diffusion
from rcdiff.models.config import TrainConfig
from rcdiff.models.tsdiff.base_diff import TSDiffusionModel
from rcdiff.models.denoisers.rwdn import RWDenoisingNetwork



class RCDiff(TSDiffusionModel): 

    denoiser: RWDenoisingNetwork

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
        
        T, _ = x.shape
        x, c = self._build_conditioning_set(x, c)
    
        assert self.diff.scheduler.K == self.denoiser.K, f"Inconsistent number of diffusion steps! Got {self.diff.scheduler.K} and {self.denoiser.K}"
    
        s = self.esn.get_states(X=x, include_initial=True) 
        x = x[self.spec.t0:T] # [x_{t_0}, ..., x_{T}]
        s = s[self.spec.t0:T] # [s_{t_0-1}, ..., x_{T-1}]
    
        losses_before = []
        losses_after  = []

        start = time.time() 

        ks = list(range(self.diff.scheduler.K))[::-1]
        
        for k in tqdm(ks):

            k_tensor = torch.full((T-self.spec.t0,), fill_value = k, dtype=torch.long, device=x.device)
            x_k, epsilon = self.diff.q_sample(x_0 = x, k = k_tensor) 
    
            eps_hat = self.denoiser(x = x_k, s = s, k = k)                    
            loss_bt = F.mse_loss(eps_hat, epsilon)
            losses_before.append(loss_bt.item())

            self.denoiser.fit(x_k = x_k, s = s, epsilon_k = epsilon, k = k, method = "Ridge")
            eps_hat = self.denoiser(x = x_k, s = s, k = k)  
            loss_at  = F.mse_loss(eps_hat, epsilon)
            losses_after.append(loss_at.item())
    
        elapsed = time.time() - start
        print(f"\nTraining complete Time: {elapsed:.2f}s")

        print(len(losses_before))
        df_losses = pd.DataFrame({'LossBefore' : losses_before, 'LossAfter' : losses_after,
                                  'k' : ks},)

        return df_losses, elapsed

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
        x, c = self._build_conditioning_set(x = x, c = c)
        s = self.esn.get_states(X=c)[-1]            
        s = s.unsqueeze(0).expand(n_samples, -1)   
        x_hat, full_reverse = self.diff.reverse_process(x=x, s=s, denoiser=self.denoiser, n_samples=n_samples)
        return x_hat, full_reverse

    
    @torch.no_grad()
    def one_step_ahead_forecast(self, x_train, x_test, c_train = None, c_test = None, n_samples : int = 100):
        """ 
        Perform a recursive one-step ahead probabilistic forecast. 

        Returns
        -------
            forecast (torch.Tensor) : (T_test, N, n_samples)
        """

        # NOTE: change this just use a vectorized form...
        _, T_test = self._check_shapes(x_train, x_test, c_train, c_test)
        x_train, cond_train = self._build_conditioning_set(x_train, c_train) 
        _, cond_test = self._build_conditioning_set(x_test, c_test)

        S = self.esn.get_states(X=cond_train.unsqueeze(0))  # (1, T_train, R)

        forecast = []
        for t in tqdm(range(T_test)):
            s = S[:, -1].expand(n_samples, -1) # take last t (n_samples, R)
            x_hat, _ = self.diff.reverse_process(x = x_train, 
                                                 s = s, 
                                                 denoiser = self.denoiser, 
                                                 n_samples = n_samples)
            forecast.append(x_hat)                                             
            S = self.esn.update_states(S[:, -1:], cond_test[t].unsqueeze(0))   

        return torch.stack(forecast)

    