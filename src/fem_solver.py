import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple

from .mesh import Mesh
from .assembly import assemble

def apply_bc(
    K: NDArray[np.float64],
    F: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    K[0, :] = K[-1, :] = 0.0
    K[:, 0] = K[:, -1] = 0.0
    K[0, 0] = K[-1, -1] = 1.0
    F[0] = F[-1] = 0.0
    return K, F

def solve_bvp(
    L: float,
    nel: int,
    p: int,
    rhs: Callable[[float], float]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    mesh = Mesh(L, nel, p)
    K, F = assemble(mesh, rhs, p)
    K, F = apply_bc(K, F)
    U = np.linalg.solve(K, F)
    return mesh.nodes(), U
