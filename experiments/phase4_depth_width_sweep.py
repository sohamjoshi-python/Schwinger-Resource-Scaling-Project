"""Phase 4: how does the accuracy-cost frontier shift as the physical system
size N grows? Repeats the qubit x depth sweep at several N and records results
tagged by N, so the frontiers can be compared.

KEY QUESTION: does 2-qubit sufficiency (seen at N=4) survive as N grows, or do
larger systems demand more model resources to hit the same accuracy?

Cost note: dense exact diagonalization is fine to ~N=12; the Hamiltonian is
2^N x 2^N, so build/diagonalization time grows fast. N=4,6,8 is a sensible
first sweep. Each (N, qubits, layers, seed) trains one model, so this is the
longest-running experiment -- expect several minutes.

Usage:  python -m experiments.phase4_system_size_scaling
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

#SYSTEM_SIZES = [4, 6, 8]      # physical lattice sites (= 2^N Hilbert space)
SYSTEM_SIZES = [4, 6] 
QUBITS = [2, 3, 4]            # MODEL qubit count (independent of N)
LAYERS = [1, 2, 3, 4, 5]
SEEDS = [0, 1, 2]
STEPS = 200
M_MAX = 2.0
NUM_POINTS = 31

os.makedirs("results", exist_ok=True)
rows = []

for N in SYSTEM_SIZES:
    print(f"\n=== building dataset for N={N} (Hilbert dim {2**(2*N)}) ===")
    data = build_dataset(N, m_max=M_MAX, num_points=NUM_POINTS)
    training_data, testing_data = train_test_split(
        data, test_size=0.2, random_state=42
    )
    m_train = [m for m, _ in training_data]
    y_train = [y for _, y in training_data]

    for nq in QUBITS:
        for nl in LAYERS:
            circuit = make_circuit(n_qubits=nq, encoding=encode_linear)
            mses = []
            for seed in SEEDS:
                theta, w, b, _ = train(circuit, nq, nl, m_train, y_train,
                                       steps=STEPS, lr=0.05, seed=seed)
                mses.append(test_mse(circuit, theta, w, b, testing_data))
            row = {
                "N": N,
                "n_qubits": nq,
                "n_layers": nl,
                "n_params": count_trainable_params(nq, nl),
                "test_mse_mean": float(np.mean(mses)),
                "test_mse_std": float(np.std(mses)),
            }
            rows.append(row)
            print(f"  N={N} q={nq} L={nl} "
                  f"MSE={row['test_mse_mean']:.2e} +/- {row['test_mse_std']:.1e}")

with open("results/phase4.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print("\nWrote results/phase4.csv")


# ---- min cost to reach a fixed accuracy target, as a function of N ----
# This is the scaling result: for each N, the cheapest config beating TARGET.
TARGET = 1e-4
print(f"\nMinimum-cost config reaching test MSE < {TARGET:.0e}, per N:")
for N in SYSTEM_SIZES:
    feasible = [r for r in rows if r["N"] == N and r["test_mse_mean"] < TARGET]
    if feasible:
        best = min(feasible, key=lambda r: r["n_params"])
        print(f"  N={N}: {best['n_qubits']}q/{best['n_layers']}L "
              f"({best['n_params']} params), MSE={best['test_mse_mean']:.2e}")
    else:
        print(f"  N={N}: no config reached the target")