import pandas as pd
import torch 
from pathlib import Path


class RVSP500:

    RV_PATH = Path('data/SP500_RV_5min.xlsx')

    def __init__(self,): 
        self._rv_data()
        
    def _rv_data(self,):
        raw = pd.read_excel(self.RV_PATH)
        raw['Date'] = pd.to_datetime(raw['Date'], format='%Y%m%d')
        self.dates = raw['Date'].to_numpy()
        RV = raw['RV']
        self.RV = torch.tensor(raw['RV'].to_numpy(), dtype=torch.float32).reshape(-1, 1)
        self.T, self.N = self.RV.shape

    def _check_len_training(self,len_train): 
        assert self.T >= len_train, f'Number of train period is larger than the total number of obs! {self.T} vs {self.len_train}'

    def get_realized_variance_data(self, len_train : int, standardize = False): 
        self._check_len_training(len_train) 
        
        RV_train = self.RV[:len_train].clone()
        RV_test = self.RV[len_train:].clone()
        dates_train = self.dates[:len_train]
        dates_test = self.dates[len_train:]

        if standardize: 
            RV_train, RV_test = self.standardize(RV_train, RV_test)

        return (dates_train, RV_train), (dates_test, RV_test)

    def get_taining_set_mean_and_std(self, RV_train): 
        mu =  RV_train.mean(dim = 0)
        sigma = RV_train.std(dim = 0)
        return mu, sigma

    def standardize(self, x_train : torch.Tensor, x_test : torch.Tensor): 

        mu, sigma = self.get_taining_set_mean_and_std(x_train)
        x_train_std = (x_train - mu)/sigma 
        x_test_std = (x_test - mu)/sigma 

        return x_train_std, x_test_std


