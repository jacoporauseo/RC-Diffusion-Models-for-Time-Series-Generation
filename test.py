from src.diff.ddpm import DDPM 
from src.diff.scheduler import BaseScheduler 
import torch 
from src.data.var import * 


K = 100
T = 1e4 
N = 4 


if __name__ == '__main__':
    N = 3      # number of variables
    p = 2      # VAR order p
    n_obs = 500      # observations to keep
    seed = 42
    
    A_list = generate_stable_var(N, p, scale=0.5,
                                    target_radius=0.9, seed=seed)
    
    mu = np.zeros(N)                 # zero intercept -> series fluctuate around 0
    cov = np.eye(N) * 0.5           # innovation covariance
    
    y = simulate_var(A_list, mu, cov, n_obs=n_obs, burn_in=500, seed=seed)
    print(type(y))
    print(y)
    print(y.shape)
    scheduler = BaseScheduler(K = 100)
    ddpm = DDPM(scheduler)





# N = 2
# T = 100
# x = torch.randn(size = (T,1,N), dtype=torch.float32)
# rnn = TimeGradRNN_FH(input_size=N, hidden_size=16, cell = 'LSTM')
# s = rnn(x)

# print("Shape: \n")
# print(s.shape)
# print("Tensor: \n")
# print(s)

# k = torch.randint(low = 0, high=100,size = (T,))

# epsilon = EpsilonTheta(target_dim=N,cond_length=16) 

# e = epsilon.forward(inputs = x,time = k, cond=s)

# print(e.shape) # (batch, 1, N)
