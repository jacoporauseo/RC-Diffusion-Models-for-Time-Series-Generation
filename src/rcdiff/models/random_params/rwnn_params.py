import torch
from typing import Tuple, Optional

from rcdiff.models.random_params.random_params import RandomParams
from rcdiff.models.random_params.random_weight import RandomWeight


class ParametersRWDN(RandomParams):
    """
    Random Parameters of an RWDN with K random layers

    Parameters
    ----------


        A_k (RandomWeight) : law of each random weight matrix `A_k in R^{hidden_size x in_size}`
        b_k  (RandomWeight) : law of each random bias `b_k in R^{hidden_size x 1}`
        gamma : input scaling 
        omega : bias scaling  
    """
    def __init__(self,
                 A_k   : RandomWeight,
                 b_k   : RandomWeight,
                 gamma : float,
                 omega : float,
                 seed  : int = 12345
                 ):
        
        super().__init__(seed=seed)
        self.A_k, self.b_k = A_k, b_k
        self.gamma, self.omega = gamma, omega

        assert self.gamma >= 0.0, "Input scaling gamma must be >= 0"
        assert self.omega >= 0.0, "Bias scaling omega must be >= 0"
        # eigvals needs a square matrix; A_k and b_k are rectangular in general
        assert A_k.norm_type != 'spectral_radius', "spectral_radius norm requires square matrices"
        assert b_k.norm_type != 'spectral_radius', "spectral_radius norm requires square matrices"


    @torch.no_grad()
    def sample(self,
            K: int,
            hidden_size: int,
            in_size: int,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw the fixed random parameters of the K random layers.

        Parameters
        ----------
            K (int) : number of diffusion steps (one random layer per step).
            hidden_size (int) : number of random features per layer.
            in_size (int) : input dimension of each layer.
            device (torch.device, optional) : device of the returned tensors. Default: None (CPU).
            dtype (torch.dtype) : dtype of the returned tensors. Default: torch.float32.

        Returns
        -------
            A (torch.Tensor) : input weights, shape (K, hidden_size, in_size), equal to gamma * A_bar.
            b (torch.Tensor) : biases, shape (K, hidden_size, 1), equal to omega * b_bar.
        """
        if K < 1:
            raise ValueError("K must be >= 1")
        gA, gb = self._generators(2)  # independent streams for A and b

        A_bar = torch.stack([self.A_k.draw((hidden_size, in_size), generator=gA) for _ in range(K)])
        b_bar = torch.stack([self.b_k.draw((hidden_size, 1), generator=gb) for _ in range(K)])

        A = (self.gamma * A_bar).to(dtype=dtype)
        b = (self.omega * b_bar).to(dtype=dtype)
        if device is not None:
            A, b = A.to(device), b.to(device)
        
        return A, b