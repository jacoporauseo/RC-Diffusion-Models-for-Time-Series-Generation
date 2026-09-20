# Reservoir Computing Diffusion Models for Time Series Generation 


Jacopo Rauseo \
University of St Gallen 

# Objective 

Generative time series models aims to learn the conditional distribution of a (discrete) stochastic process $\bm{x}_t \in \mathbb{R}^N$: 

$$ p(\bm{x}_t | \bm{x}_{t-1}, ..., \bm{x}_{t-L}, \bm{c}_{t-1}, ..., \bm{c}_{t-L})  $$

for some exogenous covariates $\bm{c}_{t}$. This is the framework of many time-series models. Check out GlutonTs (https://ts.gluon.ai/stable/). TimeGrad is a diffusion probabilistic model (https://github.com/zalandoresearch/pytorch-ts) that leverages Recurrent Neural Networks to generate a compact representation of the time dimention while it applies the DDPM framework of Ho et al. (2020) to sample from the conditional distribution. 


# Reservoir Computing 

... 

