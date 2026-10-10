import torch 
from src.rcdiff.models.random_params.random_params import RandomWeight, draw



class ParametersRWDN():
    """
    Parameters of RWDN:
        gamma       : input shifting
        omega       : bias shifting
    """
    def __init__(self,
                 A_k    : RandomWeight,
                 b_k    : RandomWeight,
                 gamma     : float,
                 omega     : float, 
                 seed      : int = 12345):
        super().__init__()
        self.A_k     = A_k
        self.b_k     = b_k
        self.gamma    = gamma
        self.omega    = omega
        self.seed     = seed
    
    def sample(self, K, hidden_size, in_size) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample bar{A},bar{C},bar{zeta} and return A,C,zeta from appropriate laws"""
        torch.manual_seed(self.seed)
        A = draw(shape=(K, hidden_size,in_size), dist = self.A_k.dist, sparsity = self.A_k.sparsity, norm_type = self.A_k.norm_type)
        b = draw(shape=(K, hidden_size,1), dist = self.b_k.dist, sparsity = self.b_k.sparsity, norm_type = self.b_k.norm_type)

        A = self.gamma*A 
        b = self.omega*b 

        return A,b