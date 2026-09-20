"""
ARMA(p, q) trajectory generator, built on statsmodels.tsa.arima_process.ArmaProcess.

We deliberately do NOT hand-roll the AR/MA recursion or the stationarity math:
ArmaProcess already (a) checks stationarity/invertibility via the polynomial
roots, and (b) generates samples with proper burn-in via scipy's lfilter
internally. Re-implementing this would just be a worse, less-tested version
of the same thing.

Coefficient convention: this class uses the "textbook" convention
    y_t = phi_1 y_{t-1} + ... + phi_p y_{t-p} + eps_t + theta_1 eps_{t-1} + ... + theta_q eps_{t-q}
statsmodels' ArmaProcess instead takes the *polynomial* coefficients, which
requires negating the AR side:
    ar_poly = [1, -phi_1, -phi_2, ..., -phi_p]
    ma_poly = [1,  theta_1, theta_2, ..., theta_q]
This conversion is handled internally -- you pass phi/theta in the natural
convention.
"""

import numpy as np
import torch
from statsmodels.tsa.arima_process import ArmaProcess
from rcdiff.data.ts_sim import TimeSeriesModel


class ARMA(TimeSeriesModel):
    def __init__(self, phi_coefs: list[float], ma_coefs: list[float], sigma: float = 1.0):
        """
        Parameters
        ----------
        phi_coefs : AR coefficients [phi_1, ..., phi_p] in the textbook convention
                    y_t = phi_1 y_{t-1} + ... (NOT the negated polynomial form).
        ma_coefs  : MA coefficients [theta_1, ..., theta_q].
        sigma     : std of the innovations eps_t.
        """
        self.phi_coefs = phi_coefs
        self.ma_coefs = ma_coefs
        self.sigma = sigma

        ar_poly = np.r_[1, [-p for p in phi_coefs]]
        ma_poly = np.r_[1, ma_coefs]
        self.process = ArmaProcess(ar_poly, ma_poly)

        if not self.process.isstationary:
            raise ValueError(
                f"AR coefficients {phi_coefs} do not give a stationary process "
                "(a root of the AR polynomial lies inside/on the unit circle)."
            )
        if not self.process.isinvertible:
            raise ValueError(
                f"MA coefficients {ma_coefs} do not give an invertible process "
                "(a root of the MA polynomial lies inside/on the unit circle)."
            )

    def generate_trajectory(self, T: int, batch_size: int = 1, burn_in: int = 500) -> torch.Tensor:
        samples = np.stack([
            self.process.generate_sample(nsample=T, burnin=burn_in, scale=self.sigma)
            for _ in range(batch_size)
        ])  # (batch_size, T)
        return torch.from_numpy(samples).float().unsqueeze(-1)  # (batch_size, T, 1)