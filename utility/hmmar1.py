from __future__ import annotations

import pandas as pd
import numpy as np
from numpy.random import normal
from collections import deque
from typing import NamedTuple, List


class HMMAR1:
    """HMM AR(1) model with Gaussian innovation and measurement error"""
    class GaussianLatentState(NamedTuple):
        mean: float
        var: float

    class Parameter(NamedTuple):
        phi: float
        v: float
        w: float

        def s(self):
            return self.v / (1 - self.phi**2)

    def __init__(self, x0_mean, x0_var):
        self.x0_mean = x0_mean
        self.x0_var = x0_var

        self.xt = HMMAR1.GaussianLatentState(x0_mean, x0_var)
        self.states: List[HMMAR1.GaussianLatentState] = []
        self.states.append(self.xt)

    def filter(self, new_y: float, theta: Parameter) -> GaussianLatentState:
        """Gaussian Forward Filter"""
        phi, v, w = theta.phi, theta.v, theta.w

        at = phi * self.xt.mean
        ht = phi**2 * self.xt.var + v
        At = ht / (ht + w)
        
        new_mean = at + At * (new_y - at)
        new_var = w * At

        new_state = HMMAR1.GaussianLatentState(mean=new_mean, var=new_var)
        self.xt = new_state
        self.states.append(new_state)
        return new_state

    def sample_trace(self, theta: Parameter) -> pd.Series:
        """Sample the entire latent trace based on the current state"""
        size = len(self.states)
        normal_samples = deque(normal(loc=0, scale=1, size=size))

        # process the last one first for a starting point
        x = self.states[-1].mean + np.sqrt(self.states[-1].var) * normal_samples.pop()

        trace = deque()
        for i in range(len(self.states)-2, 0, -1): # working backward starting at the second last
            state = self.states[i]
            state_post = self.get_backward_posterior(state, x, theta)
            standard_normal = normal_samples.pop()
            x = state_post.mean + np.sqrt(state_post.var) * standard_normal
            trace.appendleft(x)

        return pd.Series(trace)
    
    def get_backward_posterior(self, xt: GaussianLatentState, last_x, theta: Parameter) -> GaussianLatentState:
        """Calculate the backward posterior p(x[t]|x[t+1], y[1:t]) based off the markov property"""
        phi, v = theta.phi, theta.v

        ht = phi**2 * xt.var + v
        At = xt.var / ht

        new_mean = xt.mean + phi*At * (last_x - phi*xt.mean)
        new_var = v * At

        return HMMAR1.GaussianLatentState(mean=new_mean, var=new_var)
