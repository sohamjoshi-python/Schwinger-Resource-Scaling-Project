"""Phase 2: sweep model qubit count x circuit depth at fixed physical size N.
Records (n_qubits, n_layers, n_params, test_mse) to results/phase2.csv.

For trustworthy numbers each configuration is run over several seeds and the
mean / std of test MSE are reported -- QCL training is stochastic, so a single
run can mislead. The train/test split is held FIXED across all configs so
differences reflect the model, not the split.

Usage:  python -m experiments.phase2_depth_width_sweep
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import os
import numpy as np
from sklearn.model_selection import train_test_split

from src.dataset import build_dataset
from src.model import make_circuit, encode_linear
from src.train import train, test_mse, count_trainable_params

N = 4
QUBITS = [2, 3, 4]
LAYERS = [1, 2, 3, 4, 5]
SEEDS = [0, 1, 2]
STEPS = 200

data = build_dataset(N, m_max=2.0, num_points=31)
training_data, testing_data = train_test_split(data, test_size=0.2, random_state=42)
m_train = [m for m, _ in training_data]
y_train = [y for _, y in training_data]

os.makedirs("results", exist_ok=True)
rows = []
for nq in QUBITS:
    for nl in LAYERS:
        circuit = make_circuit(n_qubits=nq, encoding=encode_linear)
        mses = []
        for seed in SEEDS:
            theta, w, b, _ = train(circuit, nq, nl, m_train, y_train,
                                   steps=STEPS, lr=0.05, seed=seed)
            mses.append(test_mse(circuit, theta, w, b, testing_data))
        n_params = count_trainable_params(nq, nl)
        row = {
            "n_qubits": nq, "n_layers": nl, "n_params": n_params,
            "test_mse_mean": float(np.mean(mses)),
            "test_mse_std": float(np.std(mses)),
        }
        rows.append(row)
        print(f"q={nq} L={nl} params={n_params:2d} "
              f"MSE={row['test_mse_mean']:.2e} +/- {row['test_mse_std']:.1e}")

with open("results/phase2.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print("\nWrote results/phase2.csv")
