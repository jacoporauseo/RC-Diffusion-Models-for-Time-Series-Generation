from abc import ABC, abstractmethod 
import torch 




class ObservationMap(ABC):

    def __init__(self, in_dim : int, intercept: bool = False):
        self.intercept = intercept
        self.in_dim = in_dim    

    def _intercept(self, X: torch.Tensor) -> torch.Tensor:
        """Prepend a column of ones to X (same device & dtype as X)."""
        bias = torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)
        return torch.cat([bias, X], dim=1)
    
    @abstractmethod
    def fit(self, X : torch.Tensor, y : torch.Tensor):
        pass 

    @abstractmethod
    def predict(self, X : torch.Tensor):
        pass

class OLS(ObservationMap):
    """ 
    OLS estimation of the output layer: 
        W = (X'X)^-1 X'y
    """
    def __init__(self, in_dim : int, intercept: bool = False):
        super().__init__(in_dim=in_dim, intercept=intercept)
        self.W = None

    @torch.no_grad()
    def fit(self, X : torch.Tensor, y : torch.Tensor):
        """
        X : (T,N)
        y : (T, M)
        """
        assert X.shape[0] == y.shape[0], "The number of time period must be equal for X and y"

        if self.intercept:
            X = self._intercept(X=X)

        inv = torch.linalg.inv(X.T @ X)
        self.W = inv @ X.T @ y # (N,M)
        return self.W 

    @torch.no_grad()
    def predict(self, X : torch.Tensor):
        if self.W is None:
            raise RuntimeError("Call fit() before forward()")
        if self.intercept:
            X = self._intercept(X=X)
        return torch.matmul(X,self.W)


class Ridge(ObservationMap):
    """
    Ridge regression: y = X @ W + e: min_W  ||y - X @ W||^2_2 + alpha * ||W||^2_2
    Implementation according to Ballarin et al. (2025)
    Idea for the code from: https://gist.github.com/myazdani/3d8a00cf7c9793e9fead1c89c1398f12
    """
    def __init__(self, in_dim : int, alpha: float = 1.0, intercept: bool = False):
        super().__init__(in_dim=in_dim, intercept=intercept)
        assert alpha >= 0, "alpha must be non-negative"
        self.alpha = alpha
        self.W_ridge = None

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Fit the ridge estimator.
        X : (T, N)
        y : (T, M)
        """
        assert X.shape[-1] == self.in_dim, f"Mispecified number of X features: in_dim = {self.in_dim}, got {X.shape[-1]}"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of rows"

        if self.intercept:
            X = self._intercept(X)      

        N = X.shape[1]
        T = X.shape[0]

        I = torch.eye(N, dtype=X.dtype, device=X.device)
        if self.intercept:
            I[0, 0] = 0.0  # don't regularize intercept
           

        # W = (X'X + alpha*(T-1)*I)^{-1} X'y
        S = torch.linalg.inv((X.T @ X) + self.alpha*(T-1)*I)
        self.W_ridge = S @ X.T @ y # Should be (N,M)

        return self.W_ridge 

    @torch.no_grad()
    def predict(self, X: torch.Tensor):
        """
        Predict:  ŷ = X @ W_ridge
        X : (T,N)
        """
        if self.W_ridge is None:
            raise RuntimeError("Call fit() before forward()")

        if self.intercept:
            X = self._intercept(X)

        return torch.matmul(X,self.W_ridge)