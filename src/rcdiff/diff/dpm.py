import torch
from tqdm import tqdm
from typing import Tuple

from rcdiff.diff.scheduler import BaseScheduler
from rcdiff.diff.diffusion import Diffusion 

class DPMSolver(Diffusion):
    """
    DPM-Solver (Lu et al., 2022), orders 1 and 2, for an ε-prediction denoiser.
    Reverse process is a deterministic ODE solve from a given x_K.
    """

    def __init__(self, scheduler: BaseScheduler, seed: int = 1234,
                 num_inference_steps: int = 20, order: int = 2) -> None:
        super().__init__(scheduler, seed)
        assert order in (1, 2), "Only DPM-Solver-1 and -2 are implemented."
        self.order = order
        self.num_inference_steps = num_inference_steps  # NFE (network evaluations)

        abar = scheduler.alphas_cumprod.float()
        self.K = abar.shape[0]
        self.alpha = abar.sqrt()
        self.sigma = (1.0 - abar).sqrt()
        self.lam = self.alpha.log() - self.sigma.log()  # decreasing in k


    def _to(self, ref: torch.Tensor):
        for n in ("alpha", "sigma", "lam"):
            setattr(self, n, getattr(self, n).to(ref.device, ref.dtype))

    @staticmethod
    def _bc(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """[B] or scalar -> [B,1,...,1] (broadcastable against x)."""
        return v.reshape(-1, *([1] * (x.ndim - 1)))

    def _inverse_lambda(self, lam_target: torch.Tensor) -> torch.Tensor:
        """Continuous lambda -> nearest discrete index (piecewise-linear in log alpha vs k)."""
        log_alpha = -0.5 * torch.logaddexp(torch.zeros_like(lam_target), -2.0 * lam_target)
        la = self.alpha.log()
        idx = torch.searchsorted(-la, -log_alpha.contiguous()).clamp(1, self.K - 1)
        x0, x1 = la[idx - 1], la[idx]
        k_cont = (idx - 1).to(log_alpha.dtype) + (log_alpha - x0) / (x1 - x0)
        return k_cont.round().clamp(0, self.K - 1).long()


    def q_sample(self, 
                 x_0: torch.Tensor, 
                 k: torch.Tensor | int,
                 generator: torch.Generator | None = None, 
                 **kwargs
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_k = alpha_k x_0 + sigma_k eps  (Ho et al. Eq. 4 / Lu et al. Eq. 2.1).
        x_0: [T,N] with k scalar, or [B,T,N] with k scalar or [B].
        """
        self._to(x_0)
        k = torch.as_tensor(k, device=x_0.device).long()
        eps = torch.randn(x_0.shape, generator=generator, device=x_0.device, dtype=x_0.dtype)
        x_k = self._bc(self.alpha[k], x_0) * x_0 + self._bc(self.sigma[k], x_0) * eps
        return x_k, eps


    def _step1(self, x, s, t, eps_s):
        """DPM-Solver-1 (= DDIM), Lu et al. Eq. 3.7."""
        h = self._bc(self.lam[t] - self.lam[s], x)
        return (self._bc(self.alpha[t] / self.alpha[s], x) * x
                - self._bc(self.sigma[t], x) * torch.expm1(h) * eps_s)

    def _step2(self, x, t_prev, t_next, eps_prev, denoise):
        """DPM-Solver-2 (Lu et al., Alg. 4), midpoint in lambda rounded to the grid."""
        lam_p, lam_n = self.lam[t_prev], self.lam[t_next]
        s_i = self._inverse_lambda(0.5 * (lam_p + lam_n))
        h = lam_n - lam_p
        r1 = (self.lam[s_i] - lam_p) / h

        u = (self._bc(self.alpha[s_i] / self.alpha[t_prev], x) * x
             - self._bc(self.sigma[s_i], x) * self._bc(torch.expm1(r1 * h), x) * eps_prev)
        D = denoise(u, s_i) - eps_prev
        e = self._bc(torch.expm1(h), x)
        return (self._bc(self.alpha[t_next] / self.alpha[t_prev], x) * x
                - self._bc(self.sigma[t_next], x) * e * eps_prev
                - self._bc(self.sigma[t_next] / (2.0 * r1), x) * e * D)



    @torch.no_grad()
    def reverse_process(self, x: torch.Tensor, s: torch.Tensor,
                        denoiser: torch.nn.Module, n_samples: int = 1,
                        generator: torch.Generator | None = None,
                        start_step: int | None = None, **kwargs
                        ) -> Tuple[torch.Tensor, list]:
        """
        x : noisy input x_K, shape [T,N] or [B,T,N]
        s : conditioning for the denoiser
        Deterministic: `generator`/`seed` unused. If x is [T,N], it is repeated n_samples times.
        """
        self._to(x)
        start = self.K - 1 if start_step is None else int(start_step)

        x_t = x
        if x.ndim == 2:  # single series -> batch of n_samples identical copies
            x_t = x.unsqueeze(0).expand(n_samples, *x.shape).clone()
            if torch.is_tensor(s):
                s = s.unsqueeze(0).expand(n_samples, *s.shape)
        B = x_t.shape[0]

        denoise = lambda z, k: denoiser(z, k, s)   # [ASSUMPTION] signature

        n_steps = self.num_inference_steps // self.order
        grid = torch.linspace(start, 0, n_steps + 1, device=x.device).round().long()

        traj = [x_t.cpu()]
        for i in tqdm(range(n_steps)):
            t_prev, t_next = grid[i].expand(B), grid[i + 1].expand(B)
            eps = denoise(x_t, t_prev)
            x_t = (self._step1(x_t, t_prev, t_next, eps) if self.order == 1
                   else self._step2(x_t, t_prev, t_next, eps, denoise))
            traj.append(x_t.cpu())
        return x_t, traj