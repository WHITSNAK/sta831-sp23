import numpy as np
import pandas as pd
from typing import Optional
from .ar1 import AR1

class HMM1(AR1):
    def __init__(self, phi:float, v_var:float, w_var:float):
        """
        parameters
        ----------
        phi: The first-order correlation
        v_var: The innovation variance
        w_var: The measurement noise variance
        """
        super().__init__(phi=phi, var=v_var)
        self.w_var = w_var
    
    @property
    def signal2noise_ratio(self) -> Optional[float]:
        s = self.s
        if s:
            return s / (s + self.w_var)
        else:
            return None
    
    def simulate(self, x0, length:int) -> pd.Series:
        """Simulate a HMM(1) with the latent process as AR(1)"""
        xs = super().simulate(x0=x0, length=length)
        errors = np.random.normal(loc=0, scale=np.sqrt(self.w_var), size=length)
        return xs + errors
