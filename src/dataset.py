"""Build (mass, condensate) datasets by exact diagonalization.
Migrated from test.ipynb (get_chiral_based_on_m / build_dataset).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    from .hamiltonian import build_hamiltonian, condensate_op
except ImportError:
    # Support direct script execution: add project root to sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hamiltonian import build_hamiltonian, condensate_op

def get_condensate_based_on_m(N, m, w=1.0, J=1/8):
    H = build_hamiltonian(N, m=m, w=w, J=J)
    evals, evecs = np.linalg.eigh(H)
    gs = evecs[:, 0]                      
    op = condensate_op(N, w=w)
    return np.real(gs.conj() @ op @ gs)


def build_dataset(N, m_max=2.0, num_points=31, m_min=0.1, w=1.0, J=1/8):
    """Return a list of (m, condensate) pairs over the mass grid."""
    data = []
    for m in np.linspace(m_min, m_max, num_points):
        condensate_value = get_condensate_based_on_m(N, m, w=w, J=J)
        data.append((m, condensate_value))
    return data

