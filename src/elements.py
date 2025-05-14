import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple

class ShapeFunctions:
    @staticmethod
    def linear() -> Tuple[
        Callable[[NDArray[np.float64]], NDArray[np.float64]],
        Callable[[NDArray[np.float64]], NDArray[np.float64]]
    ]:
        def phi(xi: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.stack([0.5 * (1 - xi), 0.5 * (1 + xi)], axis=-1)
        
        def dphi(xi: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.stack([-0.5 * np.ones_like(xi), 0.5 * np.ones_like(xi)], axis=-1)

        return phi, dphi

    @staticmethod
    def quadratic() -> Tuple[
        Callable[[NDArray[np.float64]], NDArray[np.float64]],
        Callable[[NDArray[np.float64]], NDArray[np.float64]]
    ]:
        def phi(xi: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.array([
                0.5 * xi * (xi - 1),
                1 - xi**2,
                0.5 * xi * (xi + 1)
            ])
        
        def dphi(xi: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.array([
                xi - 0.5,
                -2 * xi,
                xi + 0.5
            ])

        return phi, dphi
    
    @staticmethod
    def hierarchical(p: int) -> Tuple[
        Callable[[NDArray[np.float64]], NDArray[np.float64]],
        Callable[[NDArray[np.float64]], NDArray[np.float64]]
    ]:
        if p < 1:
            raise ValueError("Polynomial degree p must be at least 1.")
        
        def legendre(xi: np.float64, p: int):
            if p == 0:
                return 1
            elif p == 1:
                return xi
            else:
                return 1/p * ((2 * p - 1) * xi * legendre(xi, p - 1) + 
                              (1 - p) * legendre(xi, p - 2))
        
        def phi(xi: NDArray[np.float64]) -> NDArray[np.float64]:
            if p >= 2:
                phi_list = [0.5 * (1 - xi), 0.5 * (1 + xi)]
                for j in range(2, p + 1):
                    coeff = 1 / np.sqrt(4 * j - 2)
                    phi_list.append(coeff * (legendre(xi, j) - legendre(xi, j - 2)))
            return np.array(phi_list)
            
        def dphi(xi: NDArray[np.float64]) -> NDArray[np.float64]:
            if p >= 2:
                dphi_list = [0.5 * np.ones_like(xi), 0.5 * np.ones_like(xi)]
                for j in range(2, p + 1):
                    coeff = np.sqrt((2 * j - 1) / 2)
                    dphi_list.append(coeff * legendre(xi, j - 1))
            return np.array(dphi_list)
            
        return phi, dphi