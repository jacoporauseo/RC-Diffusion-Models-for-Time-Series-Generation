#######################################################################################3
# TEST ESN output       
#######################################################################################3

from rcdiff.data.var import * 
from rcdiff import ESN 
from rcdiff import RandomWeight, ParametersESN
from rcdiff import DDPM, BaseScheduler
from rcdiff.models.denoisers.simple_mlp import SimpleMLP


Phi_1 = np.array([[0.5,0.1,0], [0,.1,.2],[0,.3,.3]]).T
Phi_2 = np.array([[0.25,0.06,0.02], [0,.07,8e-2],[0,.12,.15]]).T
Phi_3 = np.array([[0.125,0.037,0.018], [0,.031,.038],[0,.057,.069]]).T
sigma_u = np.array([[1.0, 0.3, 0.1],
                    [0.3, 1.0, 0.2],
                    [0.1, 0.2, 1.0]])
A = [Phi_1, Phi_2, Phi_3]

R = 50 
K = 100
T = 1000
N = 3 
t0 = 100
B = 20

if __name__ == '__main__':
    var = VAR(A_coefs=A, sigma_u=sigma_u)
    x = var.generate_trajectory(T = T, batch_size=1, burn_in=50)

    # x of shape (T, N): [x_1, ..., x_T]
    x = x.squeeze(0) # squeeze since we need (B, N) not (B, T, N)



    A = RandomWeight(dist="normal", sparsity=0.2,norm_type="spectral_radius")
    C = RandomWeight(dist="uniform", sparsity=0.2,norm_type="l2")
    zeta = RandomWeight(dist="uniform", sparsity=0.2,norm_type="euclidean")

    esnParams = ParametersESN(A_RW = A, 
                    C_RW = C,
                    zeta_RW = zeta,
                    leak_rate = 0.2, 
                    rho = 0.1, 
                    gamma = 0.4,
                    omega = 0, 
                    seed=1234) 

    esn = ESN(in_dim = N, 
            reservoir_size= R, 
            params=esnParams
            )

    # s of shape (T+1): [s_0, s_1, ..., s_T]
    s = esn.get_states(X = x, s0 = None, include_initial=True)


    x, s = x[t0:], s[t0:T] 
    idx     = torch.randint(0, T - t0, (B,))
    x_batch = x[idx]        # (B, N)
    print('Shape of the x batch')
    print(x_batch.shape)
    s_batch = s[idx] 
    print('Shape of the states batch')
    print(s_batch.shape)

    scheduler = BaseScheduler(K = K)
    ddpm = DDPM(scheduler=scheduler)
    k = torch.randint(0, K, (B,))
    x_k_batch, epsilon = ddpm.q_sample(x_0 = x_batch, k = k)

    print('Shape of the noisy x batch')
    print(x_k_batch.shape)

    print('Shape of the noise batch')
    print(epsilon.shape)

    mlp = SimpleMLP(in_dim=N, cond_dim= R,dim_hids=[32, 64, 32])
    epsilon_theta = mlp(x = x_k_batch, s = s_batch, k = k)
    print('Shape of the noise batch')
    print(epsilon.shape)
    print(epsilon)





