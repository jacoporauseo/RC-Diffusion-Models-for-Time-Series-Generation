import torch
import torch.nn as nn
from typing import Optional
from rcdiff.models.random_params import ParametersRWDN


class RWDenoisingNetwork(nn.Module):
    """
    Random-weight network used as a closed-form denoiser for a diffusion model.
    One random layer per diffusion step: eps_hat_k = sigma(A_k u + b_k)^T W_k,
    where only the readout W_k is fitted (least squares on the DDPM loss).

    Parameters
    ----------
        in_dim (int) : number of features N in `x_t^k`.
        hidden_size (int) : number of random features H.
        reservoir_size (int) : size S of the ESN state `s_t`.
        K (int) : total number of diffusion steps.
        params (ParametersRWDN) : object used to sample the random weights.
        activ (str) : activation function, one of `activations`. Default: 'SiLU'.
        lam (float) : ridge penalty. Default: 1e-3.
        device (torch.device, optional) : device of the buffers. Default: None (CPU).

    Attributes
    ----------
        methods (list) : available readout solvers.
        activations (list) : available activation names.
        nn_activ (dict) : name -> torch activation class.
        A (torch.Tensor) : buffer, shape (K, H, N+S).
        b (torch.Tensor) : buffer, shape (K, H).
        W (torch.Tensor) : buffer, shape (K, H, N), fitted readout.
    """

    A: torch.Tensor
    b: torch.Tensor
    W: torch.Tensor
    methods = ['OLS', 'Inverse', 'Ridge']
    activations = ['SiLU', 'ReLU', 'tanh', 'sigmoid']
    nn_activ = {'SiLU': nn.SiLU, 'ReLU': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}

    def __init__(self, in_dim: int, hidden_size: int, reservoir_size: int, K: int,
                 params: ParametersRWDN, activ: str = 'SiLU', lam: float = 1e-3,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.in_dim, self.hidden_size = in_dim, hidden_size
        self.reservoir_size, self.K, self.lam = reservoir_size, K, lam

        A, b = params.sample(K=K, hidden_size=hidden_size,
                             in_size=in_dim + reservoir_size, device=device)
        self.register_buffer("A", A)                       # (K, H, N+S)
        self.register_buffer("b", b.squeeze(-1))           # (K, H)
        self.register_buffer("W", torch.zeros(K, hidden_size, in_dim,
                                              device=A.device, dtype=A.dtype))
        self.set_activation(activ)

    def set_activation(self, activ: str) -> None:
        if activ not in self.activations:
            raise ValueError(f"Available activations: {self.activations}, got {activ}")
        self.sigma = self.nn_activ[activ]()

    def _features(self, x_k, s, k):
        u = torch.cat([x_k, s], dim=1) # (T, N+S)
        return self.sigma(u @ self.A[k].T + self.b[k]) # (T, H)

    @torch.no_grad()
    def fit(self, 
            x_k: torch.Tensor, 
            s: torch.Tensor, 
            epsilon_k: torch.Tensor,
            k: int, 
            method: str = "Ridge"
            ) -> None:
        """
        Fit the readout W_k in closed form (least squares on the DDPM noise-prediction loss).

        Parameters
        ----------
            x_k (torch.Tensor) : noisy samples at step k, shape (T, N).
            s (torch.Tensor) : ESN states `[s_0, ..., s_{T-1}]`, shape (T, S).
            epsilon_k (torch.Tensor) : noise used to produce `x_k`, shape (T, N).
            k (int) : diffusion step.
            method (str) : 'OLS' (needs T >= H), 'Inverse' (min-norm, needs T <= H) or 'Ridge'.
        """
        if method not in self.methods:
            raise ValueError(f"Possible {self.methods}, got {method}")
        Z = self._features(x_k, s, k)                      # (T, H)
        if method == 'OLS':
            self.W[k] = torch.linalg.lstsq(Z, epsilon_k).solution
        elif method == 'Inverse':
            self.W[k] = Z.T @ torch.linalg.solve(Z @ Z.T, epsilon_k)
        else:
            I = torch.eye(self.hidden_size, device=Z.device, dtype=Z.dtype)
            self.W[k] = torch.linalg.solve(Z.T @ Z + self.lam * I, Z.T @ epsilon_k)

    @torch.no_grad()
    def forward(self, x_k: torch.Tensor, s: torch.Tensor, k: int) -> torch.Tensor:
        r"""
        Predict `\epsilon` from `x_t^k`.

        Parameters
        ----------
            x_k (torch.Tensor) : noisy samples, shape (T, N).
            s (torch.Tensor) : states, shape (T, S) or (1, S) (broadcast over T).
            k (int) : diffusion step.

        Returns
        -------
            epsilon_hat (torch.Tensor) : predicted noise, shape (T, N).
        """
        if s.shape[0] == 1 and x_k.shape[0] > 1:
            s = s.expand(x_k.shape[0], -1)
        return self._features(x_k, s, k) @ self.W[k]