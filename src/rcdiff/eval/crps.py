import torch 
from torch.distributions import Normal
import math 


class CRPS: 

    _N = Normal(0.0, 1.0)

    @staticmethod
    def crps_estimator(forecast: torch.Tensor, actuals: torch.Tensor):
        """ 
        Cpntinuoulsy Ranked Probability Score (CRPS). Estimator of CRPS for a one-step ahead conditional 
        distribution. 

        Parameters
        ----------
            forecast (torch.Tensor): distributional forecast of shape (T, S, N)
            actuals (torch.Tensor): distributional forecast of shape (T, S, N)
        
        Returns
        ----------
            crps (float): CRPS 
            crps_n (torch.Tensor): CRPS per variable (N,)
            crps_tn (torch.Tensor): CRPS per timestep and variable, shape (T, N)

        """

        assert forecast.shape[-1] == actuals.shape[-1]
        assert forecast.shape[0] == actuals.shape[0]

        T, S, N = forecast.shape 

        term1 = torch.abs(forecast - actuals.unsqueeze(1)).mean(dim=1)   # [T, N] # [T, N]

        # Compute sum_{i<j} |x_i - x_j| efficiently
        sorted_samples = torch.sort(forecast, dim=1).values  # [T, S, N]
        i = torch.arange(S, device=forecast.device, dtype=forecast.dtype).view(1, S, 1)
        weights = 2 * i - S + 1
        pairwise_sum = (weights * sorted_samples).sum(dim=1)  # [T, N]

        if S < 2:
            raise ValueError("CRPS requires at least 2 samples.")
        
        term2 = pairwise_sum / (S * (S - 1))

        crps_tn = term1 - term2 # [T, N]
        crps_n = crps_tn.mean(axis = 0) # [N]
        crps = crps_n.mean()

        return (crps.item(), crps_n, crps_tn) 

    @staticmethod
    def crps_gaussian(actuals, mu, sigma):
        """
        Close form of the CRPS for a normal conditional distribution. 
        It works only for univariate series for now. 
        For a conditional forecast of N(mu, sigma^2) 

        Parameters
        ---------
            actuals (torch.Tensor): actual observations of the ts of shape (T, N)
            mu (torch.Tensor): conditional mean of the forecast (T,N)
            sigma (torch.Tensor): conditional variance of the forecast (T,N)
        

        """
        z = (actuals - mu) / sigma
        crps_t = sigma * (z * (2 * CRPS._N.cdf(z) - 1) + 2 * torch.exp(CRPS._N.log_prob(z)) - 1 / math.sqrt(math.pi))
        crps = crps_t.mean()
        return crps, crps_t

    @staticmethod
    def crps_lognormal(actuals, mu, sigma):
        """
        
        Close form of the CRPS for a LogNormal conditional distribution. 
        It works only for univariate series for now. 
        For a conditional forecast of LogNormal(mu, sigma^2) and actuals > 0, i.e. ln actuals ~ N(mu, sigma^2)
        Taken from Baran & Lerch (2015)

        Parameters
        ---------
            actuals (torch.Tensor): actual observations of the ts of shape (T, N)
            mu (torch.Tensor): conditional mean of the forecast (T,N)
            sigma (torch.Tensor): conditional variance of the forecast (T,N)
        
        """

        sigma = torch.as_tensor(sigma, dtype=actuals.dtype)
        w = (torch.log(actuals) - mu) / sigma
        crps_t = actuals * (2 * CRPS._N.cdf(w) - 1) - 2 * torch.exp(mu + sigma**2 / 2) * (
            CRPS._N.cdf(w - sigma) + CRPS._N.cdf(sigma / math.sqrt(2)) - 1)
        crps = crps_t.mean()
        return crps, crps_t







