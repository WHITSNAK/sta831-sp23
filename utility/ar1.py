import numpy as np
import pandas as pd
from typing import Optional

class AR1:
    """Everything about a AR(1) process"""
    def __init__(self, phi: float, var: float):
        """
        parameters
        ----------
        phi: The first-order correlation
        var: The innovation variance
        """
        self.phi = phi
        self.var = var
    
    @property
    def s(self) -> Optional[float]:
        """The marginal variance"""
        if self.is_stationary:
            return self.var / (1 - self.phi**2)
        else:
            return None
    
    def multistep_var(self, r) -> float:
        """The multi-step innovation variance"""
        scale = (1 - self.phi**(2*r)) / (1 - self.phi**2)
        return self.var * scale
    
    @property
    def is_stationary(self) -> bool:
        return np.abs(self.phi) < 1
    
    def simulate(self, x0, length: int) -> pd.Series:
        """
        Simulate a AR(1) trace of a given length.
        It conditions on the first value, x0

        parameters
        ----------
        x0: The starting value of the trace
        length: The length of the trace
        """
        phi = self.phi
        v = self.var

        # list of normal random for the future
        zs = np.random.normal(size=length-1)
        x = x0
        trace = [x]
        for z in zs:
            x = phi * x + np.sqrt(v) * z
            trace.append(x)

        return pd.Series(trace)
