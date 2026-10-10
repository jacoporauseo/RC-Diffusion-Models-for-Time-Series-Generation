
import torch 
from src.rcdiff.models.random_params.random_params import RandomWeight, draw


class ParametersESN:
    """
    Parameters of ESN:
        leak_rate   : leak rate of the ESN
        rho         : spectral radious of the reservois matrix
        gamma       : input shifting
        omega       : bias shifting
    """
    def __init__(self,
                 A_RW    : RandomWeight,
                 C_RW    : RandomWeight,
                 zeta_RW : RandomWeight,
                 leak_rate : float,
                 rho       : float,
                 gamma     : float,
                 omega     : float, 
                 seed      : int = 12345):
        super().__init__()
        self.A_RW     = A_RW
        self.C_RW     = C_RW
        self.zeta_RW  = zeta_RW
        self.leak_rate = leak_rate
        self.rho      = rho
        self.gamma    = gamma
        self.omega    = omega
        self.seed     = seed
        self.check_params()

    def check_params(self):
        assert 0.0 < self.rho < 1.0,    "Spectral radius rho must be in (0, 1)"
        assert self.omega >= 0.0,        "Shift scaling omega must be >= 0"
        assert self.gamma >= 0.0,        "Input scaling gamma must be >= 0"
        assert 0.0 <= self.leak_rate <= 1.0, "leak_rate must be in [0, 1]"

    def __repr__(self):
        return (f"ParametersESN(rho={self.rho}, gamma={self.gamma}, "
                f"omega={self.omega}, leak_rate={self.leak_rate})")

    @torch.no_grad()
    def sample(self, reservoir_size, in_size):
        """Sample bar{A},bar{C},bar{zeta} and return A,C,zeta from appropriate laws"""
        torch.manual_seed(self.seed)
        A_bar = draw(shape=(reservoir_size,reservoir_size), dist = self.A_RW.dist, sparsity = self.A_RW.sparsity, norm_type = self.A_RW.norm_type)
        C_bar = draw(shape=(reservoir_size,in_size), dist = self.C_RW.dist, sparsity = self.C_RW.sparsity, norm_type = self.C_RW.norm_type)
        zeta_bar = draw(shape=(reservoir_size,1), dist = self.zeta_RW.dist, sparsity = self.zeta_RW.sparsity, norm_type = self.zeta_RW.norm_type)

        A = self.rho*A_bar
        C = self.gamma*C_bar 
        zeta = self.omega*zeta_bar 

        return A,C,zeta