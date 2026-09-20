from src.rcdiff.diff.scheduler import BaseScheduler
import torch 
import matplotlib.pyplot as plt 
from typing import Final
import numpy as np 

K : Final[int] = 100 
beta_1 : Final[float] = 1e-4
beta_K : Final[float] = 0.1 
s : Final[float] = 8e-3


def plot_schedule(schedule : dict,
                  title : str, 
                  ylabel : str, 
                  xlabel : str,
                  **kwargs): 

    k = np.arange(1, K + 1)/K # Shift by one to have as in DDPM notation
    for legend, series in schedule.items():
        plt.plot(k, series.detach().numpy(), label = legend)
    plt.title(title, **kwargs)
    plt.ylabel(ylabel, **kwargs)
    plt.xlabel(xlabel, **kwargs)
    plt.legend(**kwargs)
    plt.show()
    


if __name__ == '__main__':

    print('*************************************')
    print(f'Number of diffusion step: {K}')
    print(f'The schedule uses beta_0 = {beta_1} and \beta_K = {beta_K}')
    print('*************************************')

    linear_scheduler = BaseScheduler(K = K, 
                                    beta_1 = beta_1, 
                                    beta_K = beta_K, 
                                    s = s,
                                    mode = 'linear')
    
    cosine_scheduler = BaseScheduler(K = K, 
                                    beta_1 = beta_1, 
                                    beta_K = beta_K, 
                                    s = s,
                                    mode = 'cosine')

    forward_variance_ddpm = {'linear schedule' : linear_scheduler.alpha_bar, 
                             'cosine_schedule' : cosine_scheduler.alpha_bar
                             }

    reverse_variance_ddpm = {'linear schedule' : linear_scheduler.beta_tilde, 
                                 'cosine_schedule' : cosine_scheduler.beta_tilde
                                 }

    plot_schedule(forward_variance_ddpm, 
                  title = r'Variance Schedule of the Reverse process, $\alpha_{k}$', 
                  ylabel = 'Alpha bar', 
                  xlabel = 'k/K',
                  fontsize = 16)

    plot_schedule(reverse_variance_ddpm, 
                      title = r'Variance Schedule of the Reverse process, $\tilde{\beta}_{k}$', 
                      ylabel = 'Beta tilde', 
                      xlabel = 'k/K',
                      fontsize = 16)
