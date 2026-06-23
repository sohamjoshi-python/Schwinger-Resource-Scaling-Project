"""Phase 1 check: confirm the refactored src/ reproduces the validated N=4 baseline
(test MSE ~6e-5 with 3 qubits, 4 layers, linear encoding). Run this BEFORE any sweep.

Usage:  python -m experiments.baseline_check
"""
from sklearn.model_selection import train_test_split

from src.dataset import build_dataset
from src.model import make_circuit, encode_linear
from src.train import train, test_mse

N = 4
data = build_dataset(N, m_max=2.0, num_points=31)
training_data, testing_data = train_test_split(data, test_size=0.2, random_state=42)

circuit = make_circuit(n_qubits=3, encoding=encode_linear)
theta, w, b, final_loss = train(
    circuit, n_qubits=3, n_layers=4,
    m_train=[m for m, _ in training_data],
    y_train=[y for _, y in training_data],
    steps=200, lr=0.05, seed=0, verbose=True,
)
mse = test_mse(circuit, theta, w, b, testing_data)
print(f"\nFinal training loss: {final_loss:.6e}")
print(f"Test MSE:            {mse:.6e}")
print("Baseline reproduced." if mse < 1e-3 else "WARNING: baseline not reproduced.")
