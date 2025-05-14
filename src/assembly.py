import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple

from .mesh import Mesh
from .elements import ShapeFunctions
from .quadrature import gauss_rule

def assemble(
        mesh: Mesh,
        f: Callable[[float], float],
        p: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    xi_g, gamma_g = gauss_rule(2*p + 1)
    phi, dphi = (
        ShapeFunctions.hierarchical(p) if p > 2 else
        ShapeFunctions.quadratic() if p == 2 else
        ShapeFunctions.linear()
    )

    nn = mesh.nel * p + 1
    K = np.zeros((nn, nn))
    F = np.zeros(nn)
    nodes = mesh.nodes()
    h = nodes[1] - nodes[0]

    for e, conn in enumerate(mesh.connect()):
        x_e = nodes[conn]
        for xi, gamma in zip(xi_g, gamma_g):
            N = phi(np.array([xi]))[0]
            dN_dxi = dphi(np.array([xi]))[0]
            J = dN_dxi @ x_e
            dN_dx = dN_dxi / J
            x = N @ x_e

            K[np.ix_(conn, conn)] += gamma * J * np.outer(dN_dx, dN_dx)
            F[conn] += gamma * J * N * f(x)
            
    return K, F