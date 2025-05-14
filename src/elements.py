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
            return 0.5 * np.stack([1 - xi, 1 + xi], axis=-1)
        
        def dphi(xi: NDArray[np.float64]) -> NDArray[np.float64]:
            return 0.5 * np.stack([-np.ones_like(xi), np.ones_like(xi)], axis=-1)
        
        return phi, dphi
    
    @staticmethod
    def quadratic() -> Tuple[
        Callable[[NDArray[np.float64]], NDArray[np.float64]],
        Callable[[NDArray[np.float64]], NDArray[np.float64]]
    ]:
        def phi(xi: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.vstack([
                0.5 * xi * (xi - 1),
                1 - xi**2,
                0.5 * xi * (xi + 1)
            ]).T
        
        def dphi(xi: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.vstack([
                xi - 0.5,
                -2 * xi,
                xi + 0.5
            ]).T
        
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
            phi_list = [0.5 * (1 - xi), 0.5 * (1 + xi)]
            if p > 2:
                for j in range(2, p + 1):
                    phi_list.append(1/np.sqrt(4*j-2) * (legendre(xi, j) - legendre(xi, j - 2)))
            return np.vstack(phi_list).T
        
        def dphi(xi: NDArray[np.float64]) -> NDArray[np.float64]:
            dphi_list = [
                -0.5 * np.ones_like(xi),
                0.5 * np.ones_like(xi)
            ]
            if p >= 2:
                for j in range(2, p + 1):
                    pj_minus1 = legendre(xi, j - 1)
                    dphi_j = np.sqrt((2 * j - 1) / 2) * pj_minus1
                    dphi_list.append(dphi_j)
            return np.vstack(dphi_list).T
        
        return phi, dphi