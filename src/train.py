"""Training and evaluation.

Parameters (theta, w, b) are passed POSITIONALLY to the optimizer so autograd
sees them as trainable -- do NOT bundle them in a dict (that silently disables
training). Refactored from test.ipynb to take the circuit and config as
arguments and to RETURN the final loss (needed for the sweep tables).
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .model import predict


def mse_loss(circuit, theta, w, b, m_train, y_train):
    y = pnp.array(y_train)
    preds = pnp.stack([predict(circuit, m, theta, w, b) for m in m_train])
    return pnp.mean((preds - y) ** 2)


def train(circuit, n_qubits, n_layers, m_train, y_train,
          steps=200, lr=0.05, seed=0, w_init=0.2, b_init=-0.78, verbose=False):
    """Train theta, w, b. Returns (theta, w, b, final_loss)."""
    rng = np.random.default_rng(seed)
    theta_shape = qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_qubits)

    theta = pnp.array(rng.random(theta_shape), requires_grad=True)
    w = pnp.array(w_init, requires_grad=True)
    b = pnp.array(b_init, requires_grad=True)

    opt = qml.AdamOptimizer(stepsize=lr)
    loss = None
    for step in range(steps):
        (theta, w, b), loss = opt.step_and_cost(
            lambda t, ww, bb: mse_loss(circuit, t, ww, bb, m_train, y_train),
            theta, w, b,
        )
        if verbose and step % 20 == 0:
            print(f"step {step:4d}   loss {loss:.6f}")

    return theta, w, b, float(loss)


def test_mse(circuit, theta, w, b, testing_data):
    """Mean squared error on held-out (m, condensate) pairs."""
    m_test = [m for m, _ in testing_data]
    y_test = np.array([np.real(y) for _, y in testing_data])
    preds = np.array([float(np.real(predict(circuit, m, theta, w, b))) for m in m_test])
    return float(np.mean((preds - y_test) ** 2))


def count_trainable_params(n_qubits, n_layers):
    """Trainable parameter count: ansatz angles + affine (w, b)."""
    return n_layers * n_qubits + 2
