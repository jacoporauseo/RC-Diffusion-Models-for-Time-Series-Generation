import torch
import torch.nn as nn 
import torch.nn.functional as F
from abc import ABC, abstractmethod
import time 
from tqdm import tqdm 
import copy 
import math 

from rcdiff.models.denoisers import SimpleMLP 
from rcdiff.models.encoders import ESN
from rcdiff.diff import Diffusion, DDPM, BaseScheduler
from rcdiff.models.config import TrainConfig
# from rcdiff.models.random_params import RandomWeight, ParametersESN


class RCDiff(ABC): 

    def __init__(self, 
                 diff : Diffusion, 
                 esn : ESN,
                 denoiser : nn.Module
                 ):
        self.diff = diff 
        self.esn = esn 
        self.denoiser = denoiser 

    def _build_conditioning_set(self, 
                                x : torch.Tensor, 
                                c : torch.Tensor | None ,
                                ): 
        x = x.float() 
        if c is not None: 
            c = c.float()
            T_c, M = c.shape
            T_x, N = x.shape
            assert T_c == T_x, "c and x should have the same time length"    
            c = torch.cat([x,c], dim=1) # (T, N + M)
            return x, c
        elif c is None: 
            c = x.clone()
            return x, c

    @abstractmethod
    def fit(self, 
            x : torch.Tensor, 
            c : torch.Tensor | None,
            **kwargs):
        pass

    @abstractmethod
    def conditional_distribution(self, 
                                 x : torch.Tensor,
                                 c : torch.Tensor | None = None,
                                 n_samples : int = 1
                                 ): 
        r""" 
        Sample from:
        
        ``p(x_{T+1} | s_{T}) \approx p(x_t | \mathcal{f}(x_{T}))``

        where `s_{T} = Φ(x_{T}, s_{T-1})`. 

        Parameters
        ---------
            x (torch.Tensor) : target series of shape (T, N)
            c (torch.Tensor) : exogenous covariates of shape (T, M)

        Returns
        -------
            x_hat (torch.Tensor) : samples from the reverse process of shape `(n_samples, N)`
        """
        pass


class ESNGrad(RCDiff): 

    def __init__(self, 
                 diff : Diffusion, 
                 esn : ESN,
                 denoiser : nn.Module, 
                 spec : TrainConfig
                 ): 
        super().__init__(diff=diff, esn=esn, denoiser=denoiser) 
        self.spec = spec 


    def fit(self, 
            x : torch.Tensor, 
            c : torch.Tensor | None, 
            **kwargs
            ):
        x, c = self._build_conditioning_set(x, c)
        T, N = x.shape
        sp = self.spec
        assert 0 <= sp.t0 < T

        # Note here it works since we include the first in the s
        s = self.esn.get_states(X=c, s0=None, include_initial=True) 
        x, s = x[sp.t0:], s[sp.t0:T]                                 


        n_val = int(sp.val_fraction * len(x))
        assert n_val > 0, "val_fraction too small"
        n_tr = len(x) - n_val
        x_tr, s_tr = x[:n_tr], s[:n_tr]
        x_va, s_va = x[n_tr:], s[n_tr:]


        with torch.random.fork_rng():
            torch.manual_seed(sp.seed)
            k_va = torch.randint(0, self.diff.scheduler.K, (len(x_va),))
            x_va_noisy, eps_va = self.diff.q_sample(x_va, k=k_va)

        optimizer = torch.optim.Adam(self.denoiser.parameters(), lr=sp.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=max(1, sp.patience // 2), factor=0.5, min_lr=1e-6) 

        gen = torch.Generator().manual_seed(sp.seed)   
        history = {"iter": [], "train": [], "val": []}
        best_val, best_state, bad_evals = float("inf"), None, 0
        run_loss, n_run = torch.zeros(()), 0

        start = time.time()
        pbar = tqdm(range(1, sp.max_iters + 1), leave=False, desc="Training")

        for it in pbar:
            self.denoiser.train()
            idx = torch.randint(0, n_tr, (sp.batch_size,), generator=gen)
            x_batch, s_batch = x_tr[idx], s_tr[idx]                
            k = torch.randint(0, self.diff.scheduler.K, (sp.batch_size,), generator=gen)

            x_noisy, noise = self.diff.q_sample(x_batch, k=k)
            eps_theta = self.denoiser(x_noisy, k=k, s=s_batch)
            loss = F.mse_loss(eps_theta, noise)

            optimizer.zero_grad()
            loss.backward()
            if sp.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), sp.grad_clip)
            optimizer.step()

            run_loss += loss.detach()
            n_run += 1                    

            if it % sp.eval_every == 0:
                val = self._evaluate(x_va_noisy, k_va, s_va, eps_va)
                train = (run_loss / n_run).item(); run_loss.zero_(); n_run = 0
                if not math.isfinite(val):
                    raise RuntimeError(f"Non-finite validation loss at iteration {it}")
                scheduler.step(val)

                if val < best_val:
                    best_val, bad_evals = val, 0
                    best_state = copy.deepcopy(self.denoiser.state_dict())
                else:
                    bad_evals += 1

                history["iter"].append(it); history["train"].append(train); history["val"].append(val)
                pbar.set_postfix(train=f"{train:.4f}", val=f"{val:.4f}", best=f"{best_val:.4f}",
                                bad=f"{bad_evals}/{sp.patience}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

                if bad_evals >= sp.patience:
                    tqdm.write(f"Early stopping at iteration {it} | best val loss {best_val:.4f}")
                    break

        self.denoiser.load_state_dict(best_state)
        self.denoiser.eval()
        return history, time.time() - start

    @torch.no_grad()
    def _evaluate(self, x_noisy, k, s, eps) -> float:
        self.denoiser.eval()
        return F.mse_loss(self.denoiser(x_noisy, k=k, s=s), eps).item()

    @torch.no_grad()
    def conditional_distribution(self, 
                                 x : torch.Tensor,
                                 c : torch.Tensor | None = None,
                                 n_samples : int = 1
                                 ): 
        r""" 
        Sample from:
        
        ``p(x_{T+1} | s_{T}) \approx p(x_t | \mathcal{f}(x_{T}))``

        where `s_{T} = Φ(x_{T}, s_{T-1})`. 

        Parameters
        ---------
            x (torch.Tensor) : target series of shape (T, N)
            c (torch.Tensor) : exogenous covariates of shape (T, M)

        Returns
        -------
            x_hat (torch.Tensor) : samples from the reverse process of shape `(n_samples, N)`
        """

        x, c = self._build_conditioning_set(x = x, c = c)
        # Take s_T and expand to (n_samples, R)
        s = self.esn.get_states(X=c)[-1]            
        s = s.unsqueeze(0).expand(n_samples, -1)   
        
        x_hat, full_reverse = self.diff.reverse_process(x=x, s=s, denoiser=self.denoiser, n_samples=n_samples)

        return x_hat, full_reverse

    @torch.no_grad()
    def one_step_ahead_forecast(self, x_train, x_test, c_train = None, c_test = None, n_samples : int = 100):
        """ 
        Perform a recursive one-step ahead probabilistic forecast. 

        Returns
        -------
            forecast (torch.Tensor) : (T_test, N, n_samples)

        """

        _, T_test = self._check_shapes(x_train, x_test, c_train, c_test)

        x_train, cond_train = self._build_conditioning_set(x_train, c_train) 
        _, cond_test = self._build_conditioning_set(x_test, c_test)

        S = self.esn.get_states(X=cond_train.unsqueeze(0))  # (1, T_train, R)

        forecast = []
        for t in tqdm(range(T_test)):
            s = S[:, -1].expand(n_samples, -1) # take last t (n_samples, R)
            x_hat, _ = self.diff.reverse_process(x = x_train, 
                                                 s = s, 
                                                 denoiser = self.denoiser, 
                                                 n_samples = n_samples)
            forecast.append(x_hat)                                             # (n_samples, N)
            S = self.esn.update_states(S[:, -1:], cond_test[t].unsqueeze(0))   # absorb realized x_{T+t+1}

        return torch.stack(forecast)




    def _check_shapes(self, x_train : torch.Tensor, x_test : torch.Tensor, 
                      c_train : torch.Tensor | None, c_test : torch.Tensor | None) -> tuple[int, int]:

        T_train, N_1 = x_train.shape
        T_test, N_2 = x_test.shape

        assert N_1 == N_2, f'The number of features must be the same in x_test and x_train! Got {N_1} and {N_2}'


        if (c_test is None and c_train is not None) or (c_test is not None and c_train is None): 
            raise ValueError('If the covariates are None in the test/train set they should be None also in the other set')
        
        elif c_test is not None and c_train is not None:
            T_train_c, M_1 = c_train.shape
            T_test_c, M_2 = c_test.shape

            assert M_1 == M_2, f'The covariates must have same number of features in both sets! Got {M_1} and {M_2}'
            assert T_train_c == T_train, f'The lenght of the training set must be the same in x_train and c_train! Got {T_train} and {T_train_c}'
            assert T_test == T_test_c, f'The lenght of the test set must be the same in x_test and c_test! Got {T_test} and {T_test_c}'

            return T_train, T_test

        else: 
            return T_train, T_test




