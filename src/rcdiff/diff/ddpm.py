import torch 
from typing import Tuple

from rcdiff.diff.scheduler import BaseScheduler 
from rcdiff.diff.diffusion import Diffusion 



class DDPM(Diffusion):
    """
    DDPM Diffusion Process. From original paper of  `Ho et al. (2020)`.
    """

    def __init__(self, scheduler : BaseScheduler):
        super().__init__(scheduler=scheduler)

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
    def p_sample(self, model, x_k: torch.Tensor, k : int, s : torch.Tensor) -> torch.Tensor:
        r"""
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

        Parameters
        ---------
            x_k (torch.Tensor)  : the vector x^k_t of the noised version at step k
            s (torch.Tensor)    : the econded vector s_{t-1} \in \mathcal{R}^S of shape [S,1] or [S,]
            k (int)             : diffusion step k
            model (nn.Module)   : denoisers model (U-Net, MLP, RWDN, etc.)

        Returns 
        ------
            x_prev (torch.Tensor) : the vector x^{k-1}_t of the noised version at step k-1
        """
        k_tensor = torch.tensor([k], device=x_k.device).expand(x_k.shape[0])
        eps = model(x_k, k_tensor, y=s)

        # can use the getters instead but nothing changes since here just need scalars
        alpha_k     = self.scheduler.alpha[k] # type: ignore
        alpha_bar_k = self.scheduler.alpha_bar[k] # type: ignore
        beta_k      = self.scheduler.beta[k] # type: ignore
        beta_tilde_k = self.scheduler.beta_tilde[k] # type: ignore

        if k == 0: 
            alpha_prev = 1

        mu = (1 / torch.sqrt(alpha_k)) * (x_k - (beta_k / torch.sqrt(1 - alpha_bar_k)) * eps)

        if k > 0:
            z = torch.randn_like(x_k)
            sigma_k = torch.sqrt(beta_tilde_k)
            x_prev = mu + sigma_k * z
        else:
            x_prev = mu 

        return x_prev

    @torch.no_grad() 
    def reverse_process(self,
                        x : torch.Tensor,
                        s : torch.Tensor,
                        denoiser : torch.nn.Module, 
                        n_samples : int,
                        ) -> Tuple[torch.Tensor, list]:
        r""" 
        Full Reverse Process of DDPM. Sample from p(x_t|s_{t-1}). 

        Parameters
        ---------
            x (torch.Tensor)     : the vector x_t \in \mathcal{R}^N of shape (N,) or (N,1)
            s (torch.Tensor)     : the econded vector s_{t-1} \in \mathcal{R}^S of shape [S,1] or [S,]
            n_samples (int)      : the number of obs to sample
            denoiser (nn.Module) : denoisers model (U-Net, MLP, RWDN, etc.)

        Returns 
        ------
            x_k (torch.Tensor)  : the clean sample \hat{x}_t 
            full_reverse (list) : list of the history of the reverse process for plotting 
                                  and debugging 
        """
        N, _ = x.shape # (N,) of the target
        final_shape = (n_samples, N)
        x_k = torch.randn(size = final_shape)
        full_reverse = [x_k]
        for k in reversed(range(self.scheduler.K)):
            x_k = self.p_sample(model = denoiser, x_k = x_k, k=k, s=s)
            full_reverse.append(x_k)
        return x_k, full_reverse

