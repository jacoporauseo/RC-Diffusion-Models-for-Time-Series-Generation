import torch
from typing import Tuple

from rcdiff.models.random_params.random_params import RandomParams
from rcdiff.models.random_params.random_weight import RandomWeight

class ParametersRWDN(RandomParams):
    """
    Parameters of an RWDN with K random layers:
        A_k   : law of each random weight matrix  A_k in R^{hidden_size x in_size}
        b_k   : law of each random bias           b_k in R^{hidden_size x 1}
        gamma : input scaling   (A = gamma * A_bar)
        omega : bias scaling    (b = omega * b_bar)
    """
    def __init__(self,
                 A_k   : RandomWeight,
                 b_k   : RandomWeight,
                 gamma : float,
                 omega : float,
                 seed  : int = 12345):
        super().__init__(seed=seed)
        self.A_k, self.b_k = A_k, b_k
        self.gamma, self.omega = gamma, omega

        assert self.gamma >= 0.0, "Input scaling gamma must be >= 0"
        assert self.omega >= 0.0, "Bias scaling omega must be >= 0"
        # eigvals needs a square matrix; A_k and b_k are rectangular in general
        assert A_k.norm_type != 'spectral_radius', "spectral_radius norm requires square matrices"
        assert b_k.norm_type != 'spectral_radius', "spectral_radius norm requires square matrices"

    @torch.no_grad()
    def sample(self, K: int, hidden_size: int, in_size: int, device=None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return A of shape (K, hidden_size, in_size) and b of shape (K, hidden_size, 1)."""
        gA, gb = self._generators(2)   # independent streams for A and b

        # Each layer is drawn as a 2D matrix so the normalisation in RandomWeight.draw
        # acts per layer. Layer k does not depend on K, so increasing K keeps the first layers unchanged.
        A_bar = torch.stack([self.A_k.draw((hidden_size, in_size), generator=gA) for _ in range(K)])
        b_bar = torch.stack([self.b_k.draw((hidden_size, 1),       generator=gb) for _ in range(K)])

        A = self.gamma * A_bar
        b = self.omega * b_bar
        return A.to(device), b.to(device)