from typing import Tuple

import numpy as np

class MixtureNormal:
    """
    Mixture of Normal with k-class and respective means and variances

    paramater
    ---------
    qs: tuple, respective weight or concentration for each mixture, qᵢ > 0
    ms: tuple, mean for each normal mixture
    ws: tuple, variances for each normal mixture
    """
    def __init__(self, qs: Tuple[float], ms: Tuple[float], ws: Tuple[float], seed: int=None):
        self.qs = np.array(qs) / np.array(qs).sum() # ensure normalized up to 1
        self.ms = np.array(ms)
        self.ws = np.array(ws)
        self.rng = np.random.default_rng(seed)
    
    def sample(self, size=None):
        ks = self.rng.multinomial(1, self.qs, size=size)
        mean_trace = ks @ self.ms
        var_trace = ks @ self.ws
        standard_normals = self.rng.normal(loc=0, scale=1, size=size)
        ys = mean_trace + np.sqrt(var_trace) * standard_normals

        return ys
