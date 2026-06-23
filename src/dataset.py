"""Build (mass, condensate) datasets by exact diagonalization.
Migrated from test.ipynb (get_chiral_based_on_m / build_dataset).
"""
import numpy as np
from .hamiltonian import build_hamiltonian, chiral_condensate_op


def get_chiral_based_on_m(N, m, a=1, g=1):
    """Exact chiral condensate of the N-site ground state at mass m."""
    H = build_hamiltonian(N, a=a, g=g, m=m, theta=0.0, boundary="open")
    eigenvalues, P = np.linalg.eig(H)
    ground_state_idx = np.argmin(eigenvalues)
    ground_state_vector = P[:, ground_state_idx]
    op = chiral_condensate_op(N)
    expectation_value = np.vdot(ground_state_vector, op @ ground_state_vector)
    return np.real(expectation_value)


def build_dataset(N, m_max=2.0, num_points=31, m_min=0.1, a=1, g=1):
    """Return a list of (m, condensate) pairs over the mass grid."""
    data = []
    for m in np.linspace(m_min, m_max, num_points):
        chiral_value = get_chiral_based_on_m(N, m, a=a, g=g)
        data.append((m, chiral_value))
    return data
