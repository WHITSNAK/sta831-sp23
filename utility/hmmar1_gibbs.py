from __future__ import annotations

import numpy as np
from .hmmar1 import HMMAR1
from typing import NamedTuple, List, Tuple


class HMMAR1GibbsSampler:
    class Priors(NamedTuple):
        # observation bias μ ~ N(g, G)
        mu0: float
        mu_var: float

        # observation variance 1/w ~ Ga(b/2, bw0/2)
        w0: float
        w_ddof: float

        # latent coef 𝛷 ~ N(c, C)I(|𝛷|<1)
        phi0: float
        phi_var: float
        phi_bound: Tuple[float, float]

        # latent innovation 1/v ~ Ga(a/2, av0/2)
        v0: float
        v_ddof: float
    
    class Sample(NamedTuple):
        para: HMMAR1.Parameter
        trace: np.array

    def __init__(self, ys: np.array, priors: Priors, seed=None):
        self.ys = ys
        self.priors = priors
        self.samples: List[HMMAR1GibbsSampler.Sample] = []
        self.rng = np.random.default_rng(seed)
    
    def sample_prior(self, size=None) -> HMMAR1.Parameter:
        priors = self.priors
        μ = priors.mu0 + self.rng.standard_normal(size=size) * np.sqrt(priors.mu_var)
        𝛷 = self._draw_truncnorm(loc=priors.phi0, scale=np.sqrt(priors.phi_var), bound=priors.phi_bound, size=size)
        inv_w = self.rng.gamma(shape=priors.w_ddof/2, scale=2/(priors.w_ddof*priors.w0), size=size)
        inv_v = self.rng.gamma(shape=priors.v_ddof/2, scale=2/(priors.v_ddof*priors.v0), size=size)

        para = HMMAR1.Parameter(phi=𝛷, v=1/inv_v, wt=1/inv_w, bt=0, mu=μ)
        return para
    
    def sample_posterior(
            self, x0: HMMAR1.LatentState,
            n_samples: int = 1000, burnin: int = 0,
            theta: HMMAR1.Parameter = None
        ) -> List[Sample]:
        
        ys = self.ys
        para = self.sample_prior() if theta is None else theta
        𝛷, v, w, _, μ = para
        inv_v, inv_w = 1/v, 1/w
        xs = HMMAR1(x0, self.rng).filter_all(ys, para).sample_trace(1, para).ravel()

        # gibbs sampler
        for i in range(n_samples + burnin):
            μ = self.step_mu(xs, inv_w)
            inv_w = self.step_inv_w(xs, μ)
            𝛷 = self.step_phi(xs, inv_v)
            inv_v = self.step_inv_v(xs, 𝛷)

            para = HMMAR1.Parameter(phi=𝛷, v=1/inv_v, wt=1/inv_w, bt=0, mu=μ)
            xs = self.step_xs(x0, para)

            # save sample
            if i >= burnin:
                self.samples.append(self.Sample(para, xs))
        
        return self.samples

    def step_mu(self, xs, inv_w):
        n = xs.shape[0] - 1
        mu0, mu_var = self.priors.mu0, self.priors.mu_var

        precision = n*inv_w + 1/mu_var
        residuals = (self.ys - xs[1:]).sum()
        mean = (mu0/mu_var + inv_w*residuals) / precision

        return mean + np.sqrt(1/precision) * self.rng.standard_normal()

    def step_inv_w(self, xs, mu):
        n = xs.shape[0] - 1
        w_ddof, w0 = self.priors.w_ddof, self.priors.w0

        ddof = (w_ddof + n) / 2
        error = self.ys - (mu + xs[1:])
        wn = 2 / (w_ddof*w0 + error @ error)

        return self.rng.gamma(shape=ddof, scale=wn)

    def step_phi(self, xs, inv_v):
        phi0, phi_var = self.priors.phi0, self.priors.phi_var
        phi_bound = self.priors.phi_bound

        sq1 = xs[1:] @ xs[:-1]
        sq0 = xs[:-1] @ xs[:-1]
        precision = 1/phi_var + inv_v * sq0
        mean = (phi0/phi_var + inv_v*sq1) / precision

        return self._draw_truncnorm(loc=mean, scale=np.sqrt(1/precision), bound=phi_bound)

    def step_inv_v(self, xs, phi):
        n = xs.shape[0] - 1
        v_ddof, v0 = self.priors.v_ddof, self.priors.v0

        ddof = (v_ddof + n) / 2
        error = xs[1:] - phi*xs[:-1]
        vn = 2 / (v_ddof*v0 + error @ error)

        return self.rng.gamma(shape=ddof, scale=vn)

    def step_xs(self, x0, para: HMMAR1.Parameter):
        return HMMAR1(x0, self.rng).filter_all(self.ys, para).sample_trace(1, para).ravel()

    def _draw_truncnorm(self, loc, scale, bound, size=None):
        """
        parameters:
        -----------
        loc: The mean of the normal
        scale: The standard deviation of the normal
        bound: the left and right bound of the actual values, (a, b]
        size: the size of array
        """
        n = 1 if size is None else size
        a, b = bound
        zs = []

        while True:
            z = loc + scale * self.rng.standard_normal()

            if z >= a and z < b: zs.append(z)
            if len(zs) == n: break

        return zs[0] if n == 1 else np.array(zs)
