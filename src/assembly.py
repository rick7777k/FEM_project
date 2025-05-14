import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
from typing import Callable, Tuple

from .mesh import Mesh
from .elements import ShapeFunctions
from .quadrature import gauss_rule

def assemble(
        mesh: Mesh,
        f: Callable[[float], float],
        p: int
) -> Tuple[sp.csr_matrix, NDArray[np.float64]]:
    xi_g, gamma_g = gauss_rule(2*p + 1)
    phi, dphi = (
        ShapeFunctions.hierarchical(p) if p > 2 else
        ShapeFunctions.quadratic() if p == 2 else
        ShapeFunctions.linear()
    )

    num_nodes = mesh.num_elements * p + 1
    K = sp.lil_matrix((num_nodes, num_nodes)) # Using LIL format for efficient assembly
    F = np.zeros(num_nodes)
    nodes = mesh.nodes()
    h = nodes[1] - nodes[0]

    for element, connection in enumerate(mesh.connect()):
        # Get the coordinates of the element
        x_e = nodes[connection]

        for xi, gamma in zip(xi_g, gamma_g):
            # Evaluate shape functions and their derivatives
            N = phi(xi)
            dN_dxi = dphi(xi)

            # Compute the Jacobian and its inverse
            J = dN_dxi @ x_e
            dN_dx = dN_dxi / J

            # Compute the physical coordinates
            x = N @ x_e

            # Assemble the stiffness matrix and load vector
            K[np.ix_(connection, connection)] += gamma * J * np.outer(dN_dx, dN_dx)
            F[connection] += gamma * J * N * f(x)

    # Convert K to CSR format for efficient matrix-vector operations
    return K, F