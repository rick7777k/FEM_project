import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from typing import Callable, Tuple

from .mesh import Mesh
from .assembly import assemble

def apply_boundary_conditions(
    K: NDArray[np.float64],
    F: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Apply Dirichlet boundary conditions
    K[0, :] = K[-1, :] = 0.0    # Set rows to zero
    K[:, 0] = K[:, -1] = 0.0    # Set columns to zero
    K[0, 0] = K[-1, -1] = 1.0   # Set diagonal to 1
    F[0] = F[-1] = 0.0          # Set force to zero
    return K, F

def solve_boundary_value_problem(
    length: float,
    num_elements: int,
    p: int,
    rhs_function: Callable[[float], float]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    mesh = Mesh(length, num_elements, p)
    K, F = assemble(mesh, rhs_function, p)
    K, F = apply_boundary_conditions(K, F)
    U = sp.linalg.spsolve(K.tocsr(), F)
    return mesh.nodes(), U