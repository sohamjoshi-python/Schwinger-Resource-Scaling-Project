"""Faster 2-D QCL with BATCHED evaluation + progress logging.

The slow version called the circuit once per training point in a Python loop
(480 calls per loss, x400 steps x seeds x configs = millions of single calls
through autograd). Here the circuit is evaluated on the WHOLE batch of inputs at
once, which PennyLane vectorizes -- the dominant speedup.

Drop these into src/model.py (replacing make_circuit_2d / train_qcl_2d).
"""
import time
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def make_circuit_2d(n_qubits, n_layers, t_max, m_max):
    """Batched QNode: accepts t, m as ARRAYS (shape (batch,)) and returns
    <Z_0> for each, in one vectorized call. Re-uploads both inputs per layer."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(t, m, theta):
        a_t = pnp.pi * t / t_max          # shape (batch,)
        a_m = pnp.pi * m / m_max
        for L in range(n_layers):
            for q in range(n_qubits):
                qml.RY(a_t if (q % 2 == 0) else a_m, wires=q)  # broadcasts over batch
            qml.BasicEntanglerLayers(weights=theta[L:L+1],
                                     wires=range(n_qubits), rotation=qml.RY)
        return qml.expval(qml.PauliZ(0))
    return circuit


def qcl_param_count(n_qubits, n_layers):
    return n_layers * n_qubits + 2


def train_qcl_2d(circuit, n_qubits, n_layers, X_tr, y_tr,
                 steps=400, lr=0.05, seed=0, log_every=50, tag=""):
    """Train with BATCHED loss. X_tr shape (n,2). Logs loss every `log_every`."""
    rng = np.random.default_rng(seed)
    theta = pnp.array(rng.random((n_layers, n_qubits)), requires_grad=True)
    w = pnp.array(1.0, requires_grad=True)
    b = pnp.array(float(np.mean(y_tr)), requires_grad=True)
    opt = qml.AdamOptimizer(lr)

    t_arr = pnp.array(X_tr[:, 0]); m_arr = pnp.array(X_tr[:, 1])
    yt = pnp.array(y_tr)

    def loss(th, ww, bb):
        preds = ww * circuit(t_arr, m_arr, th) + bb   # ONE batched call
        return pnp.mean((preds - yt) ** 2)

    t0 = time.time()
    for s in range(steps):
        (theta, w, b), L = opt.step_and_cost(loss, theta, w, b)
        if log_every and (s % log_every == 0 or s == steps - 1):
            print(f"      [{tag}] step {s:4d}/{steps}  loss={float(L):.4e}  "
                  f"({time.time()-t0:.1f}s elapsed)", flush=True)

    def predict(X):
        t = pnp.array(X[:, 0]); m = pnp.array(X[:, 1])
        return np.array(np.real(w * circuit(t, m, theta) + b))
    return predict