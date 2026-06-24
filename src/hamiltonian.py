import numpy as np
from scipy.linalg import ishermitian

def PauliX(pos, num_qubits):
    m = pos - 1
    # Define single-qubit Pauli-X and Identity
    X = np.array([[0, 1], 
                  [1, 0]])
    I = np.identity(2)
    
    operator = 1
    for i in range(num_qubits):
        # Apply X at the nth position (0-indexed), and I everywhere else
        current_gate = X if i == m else I
        operator = np.kron(operator, current_gate) if i > 0 else current_gate
        
    return operator

def PauliY(pos, num_qubits):
    m = pos - 1
    # Define single-qubit Pauli-Y and Identity
    Y = np.array([[0, -1j], 
                  [1j, 0]])
    I = np.identity(2)
    
    operator = 1
    for i in range(num_qubits):
        # Apply Y at the nth position (0-indexed), and I everywhere else
        current_gate = Y if i == m else I
        operator = np.kron(operator, current_gate) if i > 0 else current_gate
        
    return operator

def PauliZ(pos, num_qubits):
    m = pos - 1
    # Define single-qubit Pauli-Z and Identity
    Z = np.array([[1, 0], 
                  [0, -1]])
    I = np.identity(2)
    
    operator = 1
    for i in range(num_qubits):
        # Apply Z at the nth position (0-indexed), and I everywhere else
        current_gate = Z if i == m else I
        operator = np.kron(operator, current_gate) if i > 0 else current_gate
        
    return operator

def sigma_plus(pos, num_qubits):
    return 0.5 * (PauliX(pos, num_qubits) + 1j * PauliY(pos, num_qubits))

def sigma_minus(pos, num_qubits):
    return 0.5 * (PauliX(pos, num_qubits) - 1j * PauliY(pos, num_qubits))


def Q(i, n, N):
    a = 2 * (n-1) + 1
    b = 2 * (n-1) + 2
    if i == 1:
        return (sigma_plus(a, 2 * N) @ sigma_minus(b, 2 * N) + sigma_minus(a, 2 * N) @ sigma_plus(b, 2 * N)) / 2
    if i == 2:
        return (sigma_plus(a, 2 * N) @ sigma_minus(b, 2 * N) - sigma_minus(a, 2 * N) @ sigma_plus(b, 2 * N) ) / (2j)
    if i == 3:
        return (PauliZ(a, 2 * N) - PauliZ(b, 2 * N)) / 4





def QQ(n, m, N):
    """Sum over color index i of Q_i(n) @ Q_i(m), for the SU(2) electric term."""
    dim = 2**(2*N)
    out = np.zeros((dim, dim), dtype=complex)
    for i in range(1, 4):
        out += Q(i, n, N) @ Q(i, m, N)
    return out


    
def electric_hamiltonian(N, J=1/8):
    dim = 2**(2*N)
    H_E = np.zeros((dim, dim), dtype=complex)

    for n in range(1, N):
        H_E += (N - n) * QQ(n, n, N)

    for n in range(1, N-1):
        for m in range(n+1, N):
            H_E += 2 * (N - m) * QQ(n, m, N)
    return J * H_E

def mass_hamiltonian(N, m=0.5):
    dim = 2**(2*N)
    H_m = np.zeros((dim, dim), dtype=complex)
    I = np.identity(dim)
    for n in range(1, N+1):
        for c in range(1, 3):                    
            qubit = 2*(n-1) + c                   
            H_m += m * (-1)**n * (PauliZ(qubit, 2*N) + I) / 2

    return H_m

def hopping_hamiltonian(N, w=1.0):
    dim = 2**(2*N)
    H_hop = np.zeros((dim, dim), dtype=complex)
    nq = 2*N
    for n in range(1, N):              # sites n = 1 .. N-1
        for c in range(1, 3):          # colors c = 1, 2
            # 0-indexed qubit positions:
            l_n   = 2*(n-1) + (c-1)     # site n,   color c
            l_np1 = 2*n     + (c-1)     # site n+1, color c
            # JW string: product of (-PauliZ) over qubits strictly between
            # k from l_n to l_np1 - 1  (0-indexed)  -> convert to 1-indexed (+1)
            string = np.identity(dim, dtype=complex)
            for k in range(l_n, l_np1):          # k = l_n .. l_np1 - 1
                string = string @ (-PauliZ(k+1, nq))   # +1 for 1-indexed pos
            # the two terms (note: sigma indices use 1-indexed positions too)
            sp_n,   sm_n   = sigma_plus(l_n+1, nq),   sigma_minus(l_n+1, nq)
            sp_np1, sm_np1 = sigma_plus(l_np1+1, nq), sigma_minus(l_np1+1, nq)
            H_hop += sp_np1 @ string @ sm_n + sp_n @ string @ sm_np1
    return w * H_hop


def build_hamiltonian(N, m=0.5, w=1.0, J=1/8):
    H = mass_hamiltonian(N, m) + hopping_hamiltonian(N, w) + electric_hamiltonian(N, J)
    return H

def condensate_op(N, w=1.0):
    """Averaged chiral condensate, eq. 2.71 structure. Sign and shape validated
    against J=0 free theory; absolute normalization not yet matched to the
    paper's eq. 4.6 convention (irrelevant for QCL, which is scale-invariant)."""
    L = N // 2
    dim = 2**(2*N)
    op = np.zeros((dim, dim), dtype=complex)
    for n in range(1, L+1):
        for c in range(1, 3):
            q_even = 2*(2*n - 1) + c
            q_odd  = 2*(2*n - 2) + c
            op += w * 0.5 * (PauliZ(q_even, 2*N) - PauliZ(q_odd, 2*N))
    return op / L