import numpy as np
import matplotlib.pyplot as plt
 
 
def companion_matrix(A_list : list[np.ndarray]):
    """
    Build the companion form matrix for a list of lag matrices A_1..A_p.
    """
    n = A_list[0].shape[0]
    p = len(A_list)
    top = np.hstack(A_list)  # shape (n, n*p)
    if p > 1:
        bottom = np.hstack([np.eye(n * (p - 1)), np.zeros((n * (p - 1), n))])
        return np.vstack([top, bottom])
    return top
 
def spectral_radius(A_list):
    return np.max(np.abs(np.linalg.eigvals(companion_matrix(A_list))))
 
def generate_stable_var(N : int, 
                        p : int, 
                        scale=0.5, 
                        target_radius=0.9,
                        max_tries=10_000, 
                        seed=None
                        ):
    """
    Draw random VAR coefficient matrices and rescale them so the
    companion matrix has spectral radius `target_radius` (< 1),
    which guarantees stability/stationarity.
    """
    rng = np.random.default_rng(seed)
 
    for _ in range(max_tries):
        A_list = [rng.normal(scale=scale, size=(N, N)) for _ in range(p)]
        radius = spectral_radius(A_list)
        if radius > 0:
            factor = target_radius / radius
            A_list = [A * factor for A in A_list]
            return A_list
 
    raise RuntimeError("Failed to generate a stable VAR (try different scale).")
 
 
def simulate_var(A_list, mu : np.ndarray, cov, n_obs, burn_in=500, seed=None):
    """
    Simulate a VAR(p) process.
 
    A_list : list of (n x n) coefficient matrices [A_1, ..., A_p]
    mu      : (n,) intercept vector
    cov    : (n x n) innovation covariance matrix
    n_obs  : number of observations to keep (after burn-in)
    """
    rng = np.random.default_rng(seed)
    n = mu.shape[0]
    p = len(A_list)
    total = n_obs + burn_in 
 
    y = np.zeros((total + p, n))  
    chol = np.linalg.cholesky(cov)
 
    for t in range(p, total + p):
        eps = chol @ rng.standard_normal(n)
        y_t = mu.copy()
        for lag in range(1, p + 1):
            y_t = y_t + A_list[lag - 1] @ y[t - lag]
        y[t] = y_t + eps
 
    return y[p + burn_in:]  
 
