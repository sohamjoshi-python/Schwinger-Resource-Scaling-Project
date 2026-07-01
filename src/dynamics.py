"""Multi-parameter real-time dynamics: condensate <O>(t; m) as a 2-D target.

Input is (t, m): time AND mass. The target is a FAMILY of post-quench
oscillating curves, one per mass. This is the genuinely hard regime --
oscillatory (polynomials fail) AND with m-dependent frequencies (a single fixed
Fourier basis fails too) -- so it is the fair test of whether QCL has any edge
over structure-matched classical baselines.

Exact time evolution via diagonalization. Reuses the validated SU(2) builders.
"""
import numpy as np


def site_staggered_state(N):
    nq = 2 * N; idx = 0
    for site in range(1, N + 1):
        occ = (site % 2 == 1)
        for _c in range(2):
            idx = (idx << 1) | (1 if occ else 0)
    psi = np.zeros(2**nq, dtype=complex); psi[idx] = 1.0
    return psi


def evolve_observable(H, O, psi0, times):
    evals, V = np.linalg.eigh(H)
    psi0_eig = V.conj().T @ psi0
    O_eig = V.conj().T @ O @ V
    return np.array([np.real(np.vdot(np.exp(-1j*evals*t)*psi0_eig,
                                     O_eig @ (np.exp(-1j*evals*t)*psi0_eig)))
                     for t in times])


def build_dynamics_2d(build_hamiltonian, condensate_op, N,
                      m_values, t_max=12.0, n_t=60, w=1.0, J=1/8):
    """Return (X, y): X rows are (t, m), y is condensate(t; m).
    Diagonalizes once per m (the expensive step), evolves over the time grid."""
    psi0 = site_staggered_state(N)
    times = np.linspace(0.0, t_max, n_t)
    X, y = [], []
    for m in m_values:
        H = build_hamiltonian(N, m=m, w=w, J=J)
        O = condensate_op(N, w=w)
        cond = evolve_observable(H, O, psi0, times)
        for t, c in zip(times, cond):
            X.append([t, m]); y.append(c)
    return np.array(X), np.array(y), times