import torch 
import torch.nn as nn
from typing import Tuple

class RandomWeight:
    """
    Class for defining the distribution of the parameters, sparsity and normalization
    """
    VALID_DISTS = ('normal', 'uniform', 'zeros')
    VALID_NORMS = (None, 'spectral_radius', 'l1', 'l2', 'linf', 'euclidean')

    def __init__(self, dist: str, sparsity: float, norm_type: str):
        assert dist in self.VALID_DISTS, f"Invalid dist '{dist}'. Choose from {self.VALID_DISTS}"
        assert norm_type in self.VALID_NORMS, f"Invalid norm '{norm_type}'. Choose from {self.VALID_NORMS}"
        assert 0.0 <= sparsity < 1.0, f"sparsity must be in [0, 1), got {sparsity}"
        self.dist = dist
        self.sparsity = sparsity
        self.norm_type = norm_type

    def __repr__(self):
        return f"RandomWeight(dist='{self.dist}', sparsity={self.sparsity}, norm_type='{self.norm_type}')"


def draw(shape: Tuple[int, ...], 
         dist: str, 
         sparsity: float = 0.0, 
         norm_type: str = 'spectral_radius'
         ) -> torch.Tensor:
    """
    Draw a random tensor form choosen law, with selected degree of sparsity and normalization
    """

    if dist == 'normal':
        M = torch.randn(size=shape)
    elif dist == 'uniform':
        M = torch.empty(size=shape).uniform_(-1.0, 1.0)
    elif dist == 'zeros':
        M = torch.zeros(size=shape)
    else:
        raise ValueError(f"Invalid dist '{dist}'. Choose from ('normal', 'uniform', 'zeros')")

    # Sparsity
    if sparsity > 0:
        mask = torch.bernoulli(torch.full_like(M, 1.0 - sparsity))
        M = M * mask

    # Normalization
    if norm_type is None:
        return M
    elif norm_type == 'spectral_radius':
        rho = torch.linalg.eigvals(M).abs().max()
        if rho == 0.0:
            raise RuntimeError("Spectral radius is 0 after sparsification. Reduce sparsity or change seed.")
        return M / rho
    elif norm_type in ('l1', 'l2', 'linf'):
        ord_ = {'l1': 1, 'l2': 2, 'linf': float('inf')}[norm_type]
        col_norms = torch.linalg.norm(M, ord=ord_, dim=0, keepdim=True).clamp(min=1e-8)
        return M / col_norms
    elif norm_type == 'euclidean':
        return M / torch.linalg.norm(M).clamp(min=1e-8)
    else:
        raise NotImplementedError("Normalization is not implemented")






