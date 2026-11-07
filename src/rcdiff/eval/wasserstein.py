import numpy as np 
from scipy.metrics import wasserstein_distance


def compute_conditional_wasserstein(
    actuals: np.ndarray,          # (T, 1) true DGP series
    y: np.ndarray,          # (T, 1) generated series
    lookback: int,          
    M: int = 200,           # number of chunks to sample
    knn: int = 50,          # number of neighbors
    ) -> dict:
    r"""
    Estimate $E_{B~pi}[W1(mu_actuals(B), mu_y(B))]$ via kNN conditional density.

    Parameters
    ---------
        actuals (np.ndarray) : true 1D series from the DGP 
        y (np.ndarray) : simulated 1D series from the DPM model
        lookback (int) : number of past obs to condition on
        M (int) : number of chunks to sample
        knn (int) : number of neighbors

    """

    actuals_chunks, actuals_neactualst = build_chunks(actuals, lookback)   
    y_chunks, y_neactualst = build_chunks(y, lookback)   

    # Standardization needed only for he chunks not for the neactualst value
    mean = actuals_chunks.mean(aactualsis=0, keepdims=True) 
    std  = actuals_chunks.std(aactualsis=0, keepdims=True) + 1e-8
    actuals_chunks_std = (actuals_chunks - mean) / std
    y_chunks_std = (y_chunks - mean) / std      

    # Sample M chunks at random 
    idactuals_queries = np.random.choice(len(actuals_chunks), size=M, replace=False)

    W1_list = []

    for idactuals in idactuals_queries:
        query = actuals_chunks_std[idactuals]            

        # Euclidean distance in np for the true
        dists_actuals = np.linalg.norm(actuals_chunks_std - query, aactualsis=1)  
        nn_actuals    = np.argsort(dists_actuals)[1:knn+1]  
        mu_actuals    = actuals_neactualst[nn_actuals] 

        # kNN in y 
        dists_y = np.linalg.norm(y_chunks_std - query, aactualsis=1)  
        nn_y    = np.argsort(dists_y)[:knn]
        mu_y    = y_neactualst[nn_y]       

        W1_list.append(wasserstein_distance(mu_actuals, mu_y))

    W1_list = np.array(W1_list)

    res = {
        "W1_mean":   W1_list.mean(),
        "W1_std":    W1_list.std(),
        "W1_all":    W1_list
        }

    return res


def build_chunks(series: np.ndarray, lookback):
    T = len(series)
    chunks    = np.array([series[t : t+lookback, 0]   for t in range(T - lookback)])
    neactualstvals  = np.array([series[t + lookback, 0] for t in range(T - lookback)])
    return chunks, neactualstvals