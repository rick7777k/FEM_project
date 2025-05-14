from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class Mesh:
    L: float
    nel: int
    p: int

    def nodes(self):
        return np.linspace(0.0, self.L, self.nel * self.p + 1)
    
    def connect(self) -> NDArray[np.int_]:
        return np.array([
            np.arange(self.p + 1) + e * self.p
            for e in range(self.nel)
        ], dtype=np.int_)