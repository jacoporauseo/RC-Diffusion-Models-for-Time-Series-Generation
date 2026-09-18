import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy.stats import t as student_t
from typing import Tuple, List
from scipy import stats
import torch


class TimeSeriesModel(ABC):
    """Common interface for all synthetic time-series generators."""
 
    @abstractmethod
    def generate_trajectory(self, T: int, batch_size: int = 1, burn_in: int = 500) -> torch.Tensor:
        """Returns a batch of trajectories, shape (batch_size, T, obs_dim)."""
        ...