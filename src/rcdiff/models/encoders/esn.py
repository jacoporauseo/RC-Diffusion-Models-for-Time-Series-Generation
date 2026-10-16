import torch
import torch.nn as nn
from typing import Callable
from rcdiff.models.random_params import ParametersESN
from rcdiff.models.encoders.ols import Ridge, OLS

class ESN(nn.Module):
    """
    Leaky ESN used as a fixed (non-trainable) state generator, following Ballarin & Grigoryeva (2024):
        s_t = alpha * s_{t-1} + (1 - alpha) * sigma(A s_{t-1} + C x_t + zeta),   s_0 = 0
    alpha = leak_rate multiplies the PREVIOUS state (not Jaeger's convention).
    """
    A: torch.Tensor
    C: torch.Tensor
    zeta: torch.Tensor
    activ: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, 
                 in_dim: int, 
                 reservoir_size: int, 
                 params: ParametersESN, 
                 activ: str = 'tanh'
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.reservoir_size = reservoir_size
        self.leak_rate = params.leak_rate
        self.ridge = Ridge(in_dim=reservoir_size,alpha=1, intercept=True)

        if activ == 'tanh':
            self.activ = torch.tanh
        elif activ == 'sigmoid':
            self.activ = torch.sigmoid
        else:
            raise ValueError(f"activ must be 'tanh' or 'sigmoid', got {activ}")

        A, C, zeta = params.sample(reservoir_size, in_dim)
        self.register_buffer('A', A)                    # (R, R)
        self.register_buffer('C', C)                    # (R, N)
        self.register_buffer('zeta', zeta.squeeze(-1))  # (R,)

    def step(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """One update. s: (B, R), x: (B, N) -> s_next: (B, R)."""

        s_t_plus_1 = (self.leak_rate * s +
                      (1.0 - self.leak_rate) * self.activ(s @ self.A.T + x @ self.C.T + self.zeta))

        return s_t_plus_1

    @torch.no_grad()
    def get_states(self, X, s0=None, include_initial=False):
        """
        Compute the ESN states for a given input series.

        The states are deterministic given the ESN parameters and the initial
        state ``s0``.

        Parameters
        ----------
        X : torch.Tensor
            Inputs ``[x_1, ..., x_T]`` of shape ``(T, N)`` or ``(B, T, N)``.
        s0 : torch.Tensor or None, default None
            Initial state ``s_0`` of shape ``(B, R)``. If None, zeros.
        include_initial : bool, default False
            If True, ``s_0`` is prepended to the output.

        Returns
        -------
        S (torch.Tensor) : ``[s_1, ..., s_T]`` of shape ``(B, T, R)`` (``(T, R)`` if ``X`` is 2D).
            If ``include_initial=True``, ``[s_0, s_1, ..., s_T]`` of shape
            ``(B, T+1, R)``.

        Notes
        -----
        The conditioning vector for ``x_t`` is ``s_{t-1}``. With
        ``include_initial=True``::

            S = esn.get_states(X, include_initial=True)
            cond = S[:, :-1]   # s_0, ..., s_{T-1}, aligned with x_1, ..., x_T
        """
        unbatched = X.ndim == 2
        if unbatched:
            X = X.unsqueeze(0)
        X = X.to(device=self.A.device, dtype=self.A.dtype)
        B, T, _ = X.shape

        if s0 is None:
            s = torch.zeros(B, self.reservoir_size, device=X.device, dtype=X.dtype)  
        else:
            s = s0.to(device=X.device, dtype=X.dtype)

        states = [s] if include_initial else []
        for t in range(T):
            s = self.step(s, X[:, t])
            states.append(s)
        S = torch.stack(states, dim=1)
        return S.squeeze(0) if unbatched else S

    @torch.no_grad()
    def update_states(self, 
                      S: torch.Tensor, 
                      x_next: torch.Tensor
                      ) -> torch.Tensor:
        """
        Update an arbitrary long sequence of states. Useful in inference time 

        Parameters
        ---------
            S (torch.Tensor) : a tensor with ordered states of the ESN `[s_0, ..., s_t]` or `[s_{t-L}, ..., s_t]` for a int L 
            x_next (torch.Tensor) : the tensor with `x_{t+1}` of shape (B, N) for a batch B or (1,N)

        Returns 
        -------
            s_next (torch.Tensor) :  The (B, t+2, R) = [s_0, ..., s_{t+1}]
        """
        x_next = x_next.to(device=self.A.device, dtype=self.A.dtype)
        s_next = self.step(S[:, -1], x_next)
        return torch.cat([S, s_next.unsqueeze(1)], dim=1)


    def fit(self, x : torch.Tensor, y : torch.Tensor): 
        s = self.get_states(X = x, include_initial=False) 
        self.ridge.fit(X = s, y = y)

    def predict(self, x : torch.Tensor): 
        s = self.get_states(X = x, include_initial=False)
        return self.ridge.predict(X = s)

    def rolling_window_prediction(self,):
        pass