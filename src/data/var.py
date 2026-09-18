"""
VAR(p) trajectory generator, built on statsmodels.tsa.vector_ar.var_model.VARProcess.

VARProcess.is_stable() checks the standard condition: all eigenvalues of the
(pk x pk) companion matrix built from A_1..A_p must have modulus < 1.

Note: unlike ArmaProcess.generate_sample, VARProcess.simulate_var has no
built-in burn-in (it starts the recursion at the zero vector by default), so
we simulate `T + burn_in` steps and discard the initial `burn_in` steps
ourselves to remove the transient and reach the stationary distribution.
"""

import numpy as np
import torch
from statsmodels.tsa.vector_ar.var_model import VARProcess
from src.data.ts_sim import TimeSeriesModel


class VAR(TimeSeriesModel):
    def __init__(self, A_coefs: list[np.ndarray], sigma_u: np.ndarray):
        """
        Parameters
        ----------
        A_coefs : list of p coefficient matrices [A_1, ..., A_p], each (k, k),
                  for y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + eps_t.
        sigma_u : (k, k) innovation covariance matrix.
        """
        self.k = A_coefs[0].shape[0]
        coefs = np.stack(A_coefs)  # (p, k, k), the shape VARProcess expects
        intercept = np.zeros(self.k)

        self.process = VARProcess(coefs, intercept, sigma_u)

        if not self.process.is_stable():
            raise ValueError(
                "VAR coefficients are not stable (an eigenvalue of the companion "
                "matrix has modulus >= 1)."
            )

    def generate_trajectory(self, T: int, batch_size: int = 1, burn_in: int = 500) -> torch.Tensor:
        samples = np.stack([
            self.process.simulate_var(steps=T + burn_in)[burn_in:]
            for _ in range(batch_size)])  #(batch_size, T, k)
        return torch.from_numpy(samples).float()

    

