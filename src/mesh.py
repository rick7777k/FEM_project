from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class Mesh:
    length: float
    num_elements: int
    p: int

    def nodes(self) -> NDArray[np.float64]:
        return np.linspace(0.0, self.length, self.num_elements * self.p + 1)
    
    def connect(self) -> NDArray[np.int_]:
        return np.array([
            np.arange(self.p + 1) + e * self.p
            for e in range(self.num_elements)
        ], dtype=np.int_)