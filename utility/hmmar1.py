from __future__ import annotations

import numpy as np
from collections import deque
from typing import NamedTuple, List, Tuple


class HMMAR1:
    """HMM AR(1) model with Gaussian innovation and measurement error"""
    class LatentState(NamedTuple):
        mean: float
        var: float
    
    class Parameter(NamedTuple):
        """
        parameters:
        -----------
        phi: The first-order AR(1) coefficient
        v: The latent process 1-step conditional variances
        wt: The observation conditional variances at time t
        bt: the observation conditional bias at time t
        mu: the observation conditional constant shift at time t
        """
        phi: float
        v: float
        wt: float
        bt: float = 0
        mu: float = 0

        def s(self):
            return self.v / (1 - self.phi**2)

    def __init__(self, init_state: LatentState):
        self.x0 = init_state
        self.xt = init_state
        self.states: List[HMMAR1.LatentState] = [init_state]

    def filter(self, new_y: float, theta: Parameter) -> LatentState:
        """Gaussian Forward Filter"""
        phi, v, wt, bt, mu = theta

        a = phi * self.xt.mean
        ht = phi**2 * self.xt.var + v
        At = ht / (ht + wt)
        
        new_mean = a + At * (new_y - (a+bt+mu))
        new_var = wt * At

        new_state = HMMAR1.LatentState(mean=new_mean, var=new_var)
        self.xt = new_state
        self.states.append(new_state)
        return new_state

    def filter_all(self, ys: np.array, theta: Parameter) -> HMMAR1:
        """Do the forward filter of the entier sequence"""
        for y in ys:
            self.filter(y, theta)

        return self

    def sample_trace(self, n_traces: int, theta: Parameter) -> np.array:
        """Sample the entire latent trace based on the current state using FFBS"""
        size = len(self.states)
        normal_samples = deque(np.random.standard_normal(size=(size, n_traces)))

        # process the last one first for a starting point
        x = self.states[-1].mean + np.sqrt(self.states[-1].var) * normal_samples.pop()

        trace = deque((x, ))
        for i in range(len(self.states)-1, 0, -1): # working backward starting at the second last
            state = self.states[i-1]
            state_post = self._get_backward_posterior(state, x, theta)
            z = normal_samples.pop()
            x = state_post.mean + np.sqrt(state_post.var) * z
            trace.appendleft(x)

        return np.array(trace)
    
    def _get_backward_posterior(self, xt: LatentState, last_x, theta: Parameter) -> LatentState:
        """Calculate the backward posterior p(x[t]|x[t+1], y[1:t]) based off the markov property"""
        phi, v = theta.phi, theta.v

        ht = phi**2 * xt.var + v
        At = xt.var / ht

        new_mean = xt.mean + phi*At * (last_x - phi*xt.mean)
        new_var = v * At

        return HMMAR1.LatentState(mean=new_mean, var=new_var)


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
        𝛷 = self._draw_truncnorm(loc=priors.phi0, scale=np.sqrt(priors.phi_var), a=-1, b=1)
        inv_w = self.rng.gamma(shape=priors.w_ddof/2, scale=2/(priors.w_ddof*priors.w0), size=size)
        inv_v = self.rng.gamma(shape=priors.v_ddof/2, scale=2/(priors.v_ddof*priors.v0), size=size)

        para = HMMAR1.Parameter(phi=𝛷, v=1/inv_v, wt=1/inv_w, bt=0, mu=μ)
        return para
    
    def sample_posterior(
            self, x0: HMMAR1.LatentState,
            n_samples: int = 1000, burnin: int = 0,
            theta: HMMAR1.Parameter = None
        ):
        ys = self.ys
        para = self.sample_prior() if theta is None else theta
        𝛷, v, w, _, μ = para
        inv_v, inv_w = 1/v, 1/w
        xs = HMMAR1(x0).filter_all(ys, para).sample_trace(1, para).ravel()

        # gibbs sampler
        for i in range(n_samples + burnin):
            μ = self.step_mu(xs, inv_w)
            inv_w = self.step_inv_w(xs, μ)
            𝛷 = self.step_phi(xs, inv_v)
            inv_v = self.step_phi(xs, inv_v)

            para = HMMAR1.Parameter(phi=𝛷, v=1/inv_v, wt=1/inv_w, bt=0, mu=μ)
            xs = self.step_xs(x0, para)

            # save sample
            if i >= burnin:
                self.samples.append(HMMAR1GibbsSampler.Sample(para, xs))
        
        return self.samples

    def step_mu(self, xs, inv_w):
        n = xs.shape[0] - 1
        priors = self.priors
        obs_error = (self.ys - xs[1:]).sum()
        μ = (
            (priors.mu0*1/priors.mu_var + inv_w*obs_error)/(n*inv_w+1/priors.mu_var)
            + np.sqrt(1/(n*inv_w+1/priors.mu_var)) * self.rng.standard_normal()
        )
        return μ

    def step_inv_w(self, xs, mu):
        n = xs.shape[0] - 1
        ys = self.ys
        priors = self.priors
        obs_square_error = ys - (mu + xs[1:])
        obs_square_error = obs_square_error @ obs_square_error
        inv_w = self.rng.gamma(
            shape=(priors.w_ddof + n)/2,
            scale=2 / (priors.w_ddof*priors.w0 + obs_square_error)
        )
        return inv_w

    def step_phi(self, xs, inv_v):
        priors = self.priors
        latent_square = xs[1:] @ xs[:-1]
        latent_prevsquare = xs[:-1] @ xs[:-1]
        𝛷 = self._draw_truncnorm(
            loc=(1/priors.phi_var*priors.phi0 + inv_v*latent_square)/(1/priors.phi_var + inv_v*latent_prevsquare),
            scale=np.sqrt(1/(1/priors.phi_var + inv_v*latent_prevsquare)),
            a=priors.phi_bound[0], b=priors.phi_bound[1]
        )
        return 𝛷

    def step_inv_v(self, xs, phi):
        n = xs.shape[0] - 1
        priors = self.priors
        latent_s = xs[1:] - phi*xs[:-1]
        latent_s = latent_s @ latent_s
        inv_v = self.rng.gamma(
            shape=(priors.v_ddof + n)/2,
            scale=2 / (priors.v_ddof*priors.v0 + latent_s)
        )
        return inv_v

    def step_xs(self, x0, para: HMMAR1.Parameter):
        ys = self.ys
        xs = HMMAR1(x0).filter_all(ys, para).sample_trace(1, para).ravel()
        return xs

    def _draw_truncnorm(self, loc, scale, a, b):
        """
        parameters:
        -----------
        loc: The mean of the normal
        scale: The standard deviation of the normal
        a, b: the left and right bound of the actual values
        """
        while True:
            z = loc + scale * self.rng.standard_normal()
            if z >= a and z < b: break
        return z
