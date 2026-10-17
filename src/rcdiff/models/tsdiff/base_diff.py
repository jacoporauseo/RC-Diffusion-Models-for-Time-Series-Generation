import torch
import torch.nn as nn 
from abc import ABC, abstractmethod
from typing import Any 

from rcdiff.models.encoders import ESN
from rcdiff.diff import Diffusion


class TSDiffusionModel(ABC): 

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
