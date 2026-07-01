"""Higher-dimensional datasets: condensate as a function of SEVERAL Hamiltonian
parameters at once. Tests whether the QCL-vs-classical comparison changes as
input dimension grows (the regime where classical curse-of-dimensionality might
bite and a quantum feature map could, in principle, have an edge).

Reuses the validated SU(2) build_hamiltonian / condensate_op.
"""
import numpy as np
import itertools


def get_condensate(build_hamiltonian, condensate_op, N, m, w, J):
    H = build_hamiltonian(N, m=m, w=w, J=J)
    evals, evecs = np.linalg.eigh(H)
    gs = evecs[:, 0]
    O = condensate_op(N, w=w)
    return float(np.real(gs.conj() @ O @ gs))


def build_dataset_2d(build_hamiltonian, condensate_op, N,
                     m_range=(0.1, 2.0), w_range=(0.5, 2.0),
                     n_m=20, n_w=20, J=1/8):
    """condensate as f(m, w). Returns (X, y) with X shape (n_m*n_w, 2)."""
    ms = np.linspace(*m_range, n_m)
    ws = np.linspace(*w_range, n_w)
    X, y = [], []
    for m, w in itertools.product(ms, ws):
        X.append([m, w])
        y.append(get_condensate(build_hamiltonian, condensate_op, N, m, w, J))
    return np.array(X), np.array(y)


def build_dataset_3d(build_hamiltonian, condensate_op, N,
                     m_range=(0.1, 2.0), w_range=(0.5, 2.0), J_range=(0.05, 0.3),
                     n_m=12, n_w=12, n_J=12):
    """condensate as f(m, w, J). Returns (X, y) with X shape (n_m*n_w*n_J, 3)."""
    ms = np.linspace(*m_range, n_m)
    ws = np.linspace(*w_range, n_w)
    Js = np.linspace(*J_range, n_J)
    X, y = [], []
    for m, w, J in itertools.product(ms, ws, Js):
        X.append([m, w, J])
        y.append(get_condensate(build_hamiltonian, condensate_op, N, m, w, J))
    return np.array(X), np.array(y)