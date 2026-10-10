
import torch 
import numpy as np
from rcdiff.models.random_params.random_weight import RandomWeight
from rcdiff.models.random_params.random_params import RandomParams


class ParametersESN(RandomParams):
    def __init__(self, 
                 A_RW       : RandomWeight,
                 C_RW       : RandomWeight,
                 zeta_RW    : RandomWeight,
                 leak_rate  : float,
                 rho        : float,
                 gamma      : float,
                 omega      : float, 
                 seed       : int = 12345
                ):
        super().__init__(seed=seed)
        self.A_RW, self.C_RW, self.zeta_RW = A_RW, C_RW, zeta_RW
        self.leak_rate, self.rho, self.gamma, self.omega = leak_rate, rho, gamma, omega
        
        assert 0.0 < self.rho < 1.0,    "Spectral radius rho must be in (0, 1)"
        assert self.omega >= 0.0,        "Shift scaling omega must be >= 0"
        assert self.gamma >= 0.0,        "Input scaling gamma must be >= 0"
        assert 0.0 <= self.leak_rate <= 1.0, "leak_rate must be in [0, 1]"

    @torch.no_grad()
    def sample(self, reservoir_size : int, in_size : int, device=None):
        gA, gC, gZ = self._generators()
        A_bar    = self.A_RW.draw((reservoir_size, reservoir_size), generator=gA)
        C_bar    = self.C_RW.draw((reservoir_size, in_size), generator=gC)
        zeta_bar = self.zeta_RW.draw((reservoir_size, 1), generator=gZ)
        A, C, zeta = self.rho * A_bar, self.gamma * C_bar, self.omega * zeta_bar
        return A.to(device), C.to(device), zeta.to(device)


