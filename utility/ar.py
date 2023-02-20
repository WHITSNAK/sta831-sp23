from __future__ import annotations

import numpy as np

from scipy.linalg import hankel
from scipy.stats import gamma, norm
from numpy.linalg import inv, cholesky

class AR:
    """
    AR(p) model with Reference Bayesian Sampling
    using fully conjugate normal model
    """
    def __init__(self, lags:int=1):
        self.lags = lags
    
    def fit(self, data) -> AR:
        """
        Train the model and calculates all statistics
        It uses the conditional likelihood p(x[1:n]|θ, x[0])
        """
        lags = self.lags
        data = data - data.mean()

        # construct design matrix
        y = data[lags:]
        X = self._get_design_matrix(data.values)

        # sufficient statistics
        B = X.T @ X
        b = np.linalg.inv(B) @ X.T @ y

        self.data_ = data
        self.design_matrix_ = X
        self.B_ = B
        self.coeff_ = b
        self.resid_ = y - X@b
        self.var_ = self.resid_.var(ddof=lags)
        self.ddof_ = y.shape[0] - lags

        return self
    
    def sample_innovation_var(self, size:int=1) -> np.array:
        """Sample the posterior innovation variance"""
        alpha = self.ddof_ / 2
        beta = (self.var_ * self.ddof_) / 2
        rv = gamma(a=alpha, scale=1/beta)

        return 1 / rv.rvs(size=size)
    
    def sample_coeff(self, var_post:np.array) -> np.array:
        """
        Sample the conditional posterior beta given innovation variance

        parameters
        ----------
        var_post: posterior samples for innovation variance
        """
        n = var_post.shape[0]
        corr = cholesky(inv(self.B_))
        scales = np.sqrt(var_post)
        mean = np.tile(self.coeff_, (n, 1))

        # cholesky decomp of multivate normal sampling from standard univariate normal
        coeffs = mean + (corr @ norm().rvs(size=(self.lags, n)) * scales).T
        return coeffs
    
    def predict(self, h:int, coeff_post:np.array, var_post:np.array) -> np.array:
        """
        Sample from posterior predictive distribution with h-step ahead

        parameters
        ----------
        h: prediction horizon
        coeff_post: posterior samples for the coeffs
        var_post: posterior samples for innovation variance
        """
        yhats = []
        std_post = np.sqrt(var_post)
        x = np.tile(self.data_.values[-1], var_post.shape)
        for _ in range(h):
            rv = norm(loc=coeff_post * x, scale=std_post)
            x = rv.rvs()
            yhats.append(x)
            
        return np.array(yhats)

    def _get_design_matrix(self, data: np.array) -> np.array:
        """
        Create the design matrix out of the time series based on the given lags

        parameters:
        -----------
        data: the original time series with n observations row vector with shape (n, )

        return:
        -------
        Deisgn Matrix X with shape (n-p, p)
        """
        lags = self.lags
        n = data.shape[0]
        X = hankel(data[:n-lags], data[-(lags+1):-1])
        return X
    
    def logp(self, b_samples: np.array, v_samples: np.array) -> np.array:
        """
        The unnormalized log probability of all posterior samples

        parameters:
        -----------
        b_samples: posterior samples of ϕ[p] with shape [m, p]. m for MC sample size.
        v_samples: posterior samples of v with shape [m, 1]. m for MC sample size.

        return:
        -------
        log probability array with shape (m, )
        """
        X = self.design_matrix_
        x0 = self.data_[0]
        n = X.shape[0]

        logps = []
        for phi, v in zip(b_samples, v_samples):
            e = X - phi * X
            Q = (1-phi**2)*x0**2 + e.T@e
            logp = np.log(1-phi**2)/2 - (n/2)*np.log(v) - Q.flatten()[0]/(2*v)
            logps.append(logp.flatten()[0])

        return np.array(logps)
