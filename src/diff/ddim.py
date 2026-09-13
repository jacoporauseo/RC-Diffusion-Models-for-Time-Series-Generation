import torch 
from src.diff.ddpm import DDPM 
import numpy as np 


class DDIM(DDPM):
    """
    A high-level wrapper of DDIM.
    """

    @torch.no_grad()
    def ddim_p_sample(self, xt, t, t_prev, eta=0.0):
        """
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        # Compute x_t_prev based on ddim reverse process, following Equation 12 of the DDIM paper.
        alpha_prod_t = extract(self.scheduler.alphas_cumprod, t, xt)
        if t_prev >= 0:
            alpha_prod_t_prev = extract(self.scheduler.alphas_cumprod, t_prev, xt)

    @torch.no_grad()
    def ddim_p_sample_loop(self, shape, num_inference_timesteps=50, eta=0.0):
        r"""
        The loop of the reverse process of DDIM.

        Input:
            shape (`Tuple`): The shape of output. e.g., (num particles, 2)
            num_inference_timesteps (`int`): the number of timesteps in the reverse process.
            eta (`float`): correspond to η in DDIM which controls the stochasticity of a reverse process.
        Output:
            x0_pred (`torch.Tensor`): The final denoised output through the DDPM reverse process.
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        step_ratio = self.var_scheduler.num_train_timesteps // num_inference_timesteps
        timesteps = (
            (np.arange(0, num_inference_timesteps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        timesteps = torch.from_numpy(timesteps)


        prev_timesteps = timesteps - step_ratio

        xt = torch.zeros(shape).to(self.device)
        for t, t_prev in zip(timesteps, prev_timesteps):
            pass

        x0_pred = xt

        ######################

        return x0_pred