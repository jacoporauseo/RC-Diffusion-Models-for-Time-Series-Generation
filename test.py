from rcdiff.diff.ddpm import DDPM 
from rcdiff.diff.scheduler import BaseScheduler 
import torch 
from rcdiff.data.var import * 


K = 100
T = 1e4 
N = 4 

scheduler = BaseScheduler(K = 1000)
ddpm = DDPM(scheduler)


if __name__ == '__main__':
    k = 23
    x_k = torch.randn(size = (100,1))
    k_tensor = torch.tensor([k]).expand(x_k.shape[0])
    alpha_bar = scheduler.get_alpha_bar(k_tensor, x_shape = x_k.shape)
    print(alpha_bar.shape)
    print(alpha_bar)



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
