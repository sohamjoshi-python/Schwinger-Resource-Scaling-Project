"""The QCL model: encodings, ansatz, circuit, predict.

Refactored from test.ipynb so that n_qubits / n_layers / encoding are
ARGUMENTS rather than module-level globals. This is the one change needed
to sweep configurations: make_circuit(n_qubits, n_layers, encoding) returns
a fresh QNode for any configuration. Encodings are named functions so the
encoding study is a one-line swap.
"""
import pennylane as qml
from pennylane import numpy as pnp


# ----- encodings (each loads the scalar m onto the qubits) -----
def encode_linear(m, n_qubits, m_max=2.0):
    """Soham's validated encoding: linear map of m into [0, pi]."""
    angle = pnp.pi * m / m_max
    features = [angle] * n_qubits
    qml.AngleEmbedding(features=features, wires=range(n_qubits), rotation="Y")


def encode_arctan(m, n_qubits):
    """Alternative encoding (arctan); compresses the large-m end."""
    angle = pnp.arctan(m)
    features = [angle] * n_qubits
    qml.AngleEmbedding(features=features, wires=range(n_qubits), rotation="Y")


# TODO (Phase 3): encode_reupload -- encode m, ansatz layer, encode m again, ...
# to enrich the accessible Fourier spectrum.


def make_circuit(n_qubits, encoding=encode_linear):
    """Return a QNode circuit(m, theta) -> <Z_0> for the given configuration."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(m, theta):
        encoding(m, n_qubits)
        qml.BasicEntanglerLayers(weights=theta, wires=range(n_qubits), rotation=qml.RY)
        return qml.expval(qml.PauliZ(0))

    return circuit


def predict(circuit, m, theta, w, b):
    """Affine readout on top of the single-qubit measurement."""
    v = circuit(m, theta)
    return w * v + b
