from __future__ import annotations

import numpy as np
from collections import deque
from typing import NamedTuple, List


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

    def __init__(self, init_state: LatentState, seed=None):
        self.x0 = init_state
        self.xt = init_state
        self.states: List[HMMAR1.LatentState] = [init_state]
        self.rng = np.random.default_rng(seed)

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
        normal_samples = deque(self.rng.standard_normal(size=(size, n_traces)))

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
