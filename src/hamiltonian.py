"""Lattice Schwinger model: Pauli operators, Hamiltonian (Ikeda eq. 3),
and the chiral condensate operator.

Migrated verbatim from the validated test.ipynb. These functions are the
known-good baseline; do not change them without re-checking the N=4 result.
"""
import numpy as np
from scipy.linalg import ishermitian


def PauliX(pos, num_qubits):
    m = pos - 1
    X = np.array([[0, 1],
                  [1, 0]])
    I = np.identity(2)
    operator = 1
    for i in range(num_qubits):
        current_gate = X if i == m else I
        operator = np.kron(operator, current_gate) if i > 0 else current_gate
    return operator


def PauliY(pos, num_qubits):
    m = pos - 1
    Y = np.array([[0, -1j],
                  [1j, 0]])
    I = np.identity(2)
    operator = 1
    for i in range(num_qubits):
        current_gate = Y if i == m else I
        operator = np.kron(operator, current_gate) if i > 0 else current_gate
    return operator


def PauliZ(pos, num_qubits):
    m = pos - 1
    Z = np.array([[1, 0],
                  [0, -1]])
    I = np.identity(2)
    operator = 1
    for i in range(num_qubits):
        current_gate = Z if i == m else I
        operator = np.kron(operator, current_gate) if i > 0 else current_gate
    return operator


def L_n(n, N):
    """Cumulative electric-field operator L_n (Ikeda eq. 4), 1-indexed."""
    temp = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(1, n + 1):
        temp += (np.identity(2**N) * (-1)**i + PauliZ(i, N)) / 2
    return temp


def build_hamiltonian(N, a=1, g=1, m=0.25, theta=0.0, boundary="open"):
    """Lattice Schwinger Hamiltonian (Ikeda eq. 3) as a dense 2**N x 2**N matrix."""
    H = np.zeros((2**N, 2**N), dtype=complex)
    # hopping term
    for i in range(1, N):
        H += (PauliX(i, N) @ PauliX(i + 1, N) + PauliY(i, N) @ PauliY(i + 1, N)) / (4 * a)
    # staggered mass term
    for i in range(1, N + 1):
        H += m * (-1)**i * PauliZ(i, N) / 2
    # electric-field term
    for i in range(1, N):
        H += a * g**2 * L_n(i, N) @ L_n(i, N) / 2

    if ishermitian(H):
        return H
    raise ValueError("Constructed Hamiltonian is not Hermitian.")


def chiral_condensate_op(N):
    """Chiral condensate operator: staggered sum of Z_n, normalized by N."""
    op = np.zeros((2**N, 2**N), dtype=complex)
    for n in range(1, N + 1):
        op += (-1)**n * PauliZ(n, N)
    return op / N
