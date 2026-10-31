# Reservoir Computing Diffusion Models for Time Series Generation 


Jacopo Rauseo \
University of St Gallen 

# Objective 

Generative time series models aims to learn the conditional distribution of a (discrete) stochastic process $\bm{x}_t \in \mathbb{R}^N$: 

$$ p(\bm{x}_t | \bm{x}_{t-1}, ..., \bm{x}_{t-L}, \bm{c}_{t-1}, ..., \bm{c}_{t-L})  $$

for some exogenous covariates $\bm{c}_{t}$. This is the framework of many time-series models. Check out GlutonTs (https://ts.gluon.ai/stable/). TimeGrad is a diffusion probabilistic model (https://github.com/zalandoresearch/pytorch-ts) that leverages Recurrent Neural Networks to generate a compact representation of the time dimention while it applies the DDPM framework of Ho et al. (2020) to sample from the conditional distribution. 


# RC Time Series Models

A typical time series diffusion model (as TimeGrad or Scoregrad) has three main components: 

* A **diffusion process** (either VP or VE). We adopt DDPM forward and reverse process (Ho et al., 2020). However many representation of the forward and the reverse can be implemented, for example DDIM, SDEs or accelerated sampling versions. 

* A **time encoding network**. Standard literature uses RNNs or Transformers. Here we present a version with a Echo State Network (ESN) 

* A **denoising network**. Much work has been done and there are thousants of possible architectures. A U-Net, CNNs or Vision Transformers are the most common ones. Here we suggest to use a Random Weight Neural Network. 


# Notebooks schema (updated at 31.10)


* The `Forward.ipynb` contains some code for checking the forward and reverse process of the diffusion 
* The `esngrad.ipynb` tests the ESN + trained denoising MLP 
* The `full_rc.ipynb` tests the ESN + RWDN
* The `forecast_comparison.ipynb` comes from the paper of Jordi & Mika
* The `simple_ddpm.ipynb` tests a simple DDPM with conditioning $X_{t-1} = [x_{t-1}, ..., x_{t-5}]$ to sample from $p_{\theta}(x_t|X_{t-1} )$ 
