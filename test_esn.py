#######################################################################################3
# TEST ESN output       
#######################################################################################3

from rcdiff.data.var import * 
from rcdiff import ESN 
from rcdiff import RandomWeight, ParametersESN


Phi_1 = np.array([[0.5,0.1,0], [0,.1,.2],[0,.3,.3]]).T
Phi_2 = np.array([[0.25,0.06,0.02], [0,.07,8e-2],[0,.12,.15]]).T
Phi_3 = np.array([[0.125,0.037,0.018], [0,.031,.038],[0,.057,.069]]).T
sigma_u = np.array([[1.0, 0.3, 0.1],
                    [0.3, 1.0, 0.2],
                    [0.1, 0.2, 1.0]])
A = [Phi_1, Phi_2, Phi_3]

R = 50 


if __name__ == '__main__':
    var = VAR(A_coefs=A, sigma_u=sigma_u)
    y = var.generate_trajectory(T = 1000, batch_size=1, burn_in=50)
    y = y.squeeze(0)
    print('VAR(3) process: ')
    print(y.shape) 
    print(y)


    T, N = y.shape
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

    S = esn.get_states(X = y, s0 = None, include_initial=True)
    print('States: ')
    print(S)
    print(S.shape)









