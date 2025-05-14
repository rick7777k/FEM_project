import numpy as np
from numpy.typing import NDArray
from typing import Tuple

def gauss_rule(order: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    return np.polynomial.legendre.leggauss(order)