import numpy as np 
import torch 
import matplotlib.pyplot as plt
from src.diff.scheduler import BaseScheduler 
from src.diff.ddim import DDIM 
from src.diff.ddpm import DDPM 
from src.diff.noise_process import NoiseProcess
from typing import Final
from src.data.ar import ARMA 


# plot forward process and chec

K : Final[int] = 100 
beta_1 : Final[float] = 1e-4
beta_K : Final[float] = 0.1 
s : Final[float] = 8e-3

B : Final[int] = 32
N : Final[int] = 3

scheduler = BaseScheduler(K = K, 
                          beta_1=beta_1, 
                          beta_K=beta_K,
                          mode = 'linear')
ddpm = DDPM(scheduler)

phi_coefs = [0.8]
ma_coefs = []

arma = ARMA(phi_coefs=phi_coefs, 
            ma_coefs=ma_coefs)

def plot_series(y : torch.Tensor): 
    T, N = y.shape
    time = np.arange(0, T)
    for i in range(N): 
        y_i = y[:,i]
        plt.plot(time, y_i.detach().numpy(), label = f'Series y_{i}')
    plt.ylabel('Value')
    plt.xlabel('Time')
    plt.legend() 
    plt.show()

def plot_forward_process(
    y: torch.Tensor,
    noise: NoiseProcess,
    steps: list[int] = [0, 11, 49, 99],
):
    """
    Visualize the forward diffusion process at several timesteps k.

    Parameters
    ----------
    y : (T, N) tensor -- a single trajectory of length T with N covariates/dimensions.
    noise : object implementing q_sample(x_0, k) -> (x_k, added_noise).
    steps : which diffusion steps k to visualize, alongside the original (clean) series.
    """
    T, N = y.shape

    process_forw = {"Original": y.detach().numpy()}
    for k in steps:
        k_tensor = torch.full((T,), k, dtype=torch.long)
        x_k, _ = noise.q_sample(x_0=y, k=k_tensor)
        process_forw[f"k = {k + 1}"] = x_k.detach().numpy()

    n_cols = len(process_forw)
    fig, axes = plt.subplots(
        nrows=N, ncols=n_cols, figsize=(4 * n_cols, 2.5 * N),
        sharex=True, sharey="row", squeeze=False,
    )

    for j, (label, series) in enumerate(process_forw.items()):
        for d in range(N):
            ax = axes[d, j]
            ax.plot(series[:, d], alpha=0.8)
            if d == 0:
                ax.set_title(label)
            if j == 0:
                ax.set_ylabel(f"dim {d}" if N > 1 else "value")
            if d == N - 1:
                ax.set_xlabel("t")

    fig.suptitle("Forward diffusion process")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    T = 1000
    x_t = arma.generate_trajectory(T = T,)[0]
    print(x_t.shape)
    plot_forward_process(x_t, noise = ddpm, steps=[9, 19, 49, 99])
    




