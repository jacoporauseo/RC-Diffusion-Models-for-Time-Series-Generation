import torch 
from typing import Tuple

from rcdiff.diff.scheduler import BaseScheduler 
from rcdiff.diff.noise_process import NoiseProcess 



class DDPM(NoiseProcess):
    """
    DDPM Diffusion Process. From original paper of  `Ho et al. (2020)`.
    """

    def __init__(self, scheduler : BaseScheduler):
        self.scheduler = scheduler 

    def q_sample(self, x_0 : torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from `q(x_t^{k}|x_t)`. From DDPM of `Ho et al. (2020)`: \n
                                x_t^k ~ q(x_t^k | x_t)
        By using the reparameterization trick this distribution is: \n
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
    def p_sample(self, model, x_t: torch.Tensor, k : int, s : torch.Tensor) -> torch.Tensor:
        """
        Sample from the reverse  `p(x_t^{k-1} | x_t^{k}) ~ N(μ,σ^2)`. From DDPM of 
        `Ho et al. (2020)`: \n
                x_t^{k-1} | x_t^{k-1}, k, s_{t-1} ~ p_{θ}(x_t^{k-1} | x_t, s_{t-1}, k) 
        where `s_{t-1}` is a compact representation of the history of time series `x_t` and 
        `k` is the diffusion step. The reverse distribution is modeled as Gaussian under the
        assumption that the variance schedule `β_k` is small. The mean of the reverse distribution is: \n
                μ = 1/sqrt(α_k) * [x_t^{k} - (1-α_k)/sqrt(1 - ᾱ_k) * ε(x_t^{k}, s_{t-1}, k)]
        The variance is: \n
                            σ^2 = (1-ᾱ_{k-1})/(1 - ᾱ_k) * β_k
        This for all `k` except the `k=1` case (in our code `k=0`) where the variance is zero. Another
        option in this case is to model the reverse process in `k=1` with an ad-hoc decoder. 
        """
        k_tensor = torch.tensor([k], device=x_t.device).expand(x_t.shape[0])
        eps = model(x_t, k_tensor, y=s)
        
        alpha_k     = self.scheduler.alpha[k] # type: ignore
        alpha_bar_k = self.scheduler.alpha_bar[k] # type: ignore
        beta_k      = self.scheduler.beta[k] # type: ignore
        beta_tilde_k = self.scheduler.beta_tilde[k] # type: ignore
        # alpha_bar_k_prev = self.scheduler.betas[k-1] # was wrong

        mu = (1 / torch.sqrt(alpha_k)) * (x_t - (beta_k / torch.sqrt(1 - alpha_bar_k)) * eps)

        if k > 0:
            z = torch.randn_like(x_t)
            sigma_k = torch.sqrt(beta_tilde_k)
            x_prev = mu + sigma_k * z
        else:
            x_prev = mu 

        return x_prev

