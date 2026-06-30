"""Experiment: run the classical baselines and write results/classical_baseline.csv.

Mirrors phase2_depth_width_sweep.py: same dataset, same split convention, same
CSV schema (model, n_params, test_mse_mean, test_mse_std) so the output can be
overlaid directly on the QCL Pareto frontier in a separate plotting step.

Usage:  python -m experiments.classical_baseline_experiment
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import os

from src.dataset import build_dataset
from src.classical_model import classical_baselines

N = 4                      # match the QCL sweep's physical system size
data = build_dataset(N, m_max=2.0, num_points=31)

rows = classical_baselines(data)

os.makedirs("results", exist_ok=True)
with open("results/classical_baseline.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["model", "n_params", "test_mse_mean", "test_mse_std"])
    writer.writeheader()
    writer.writerows(rows)

for r in rows:
    print(f"{r['model']:>12}  params={r['n_params']:3d}  "
          f"MSE={r['test_mse_mean']:.2e} +/- {r['test_mse_std']:.1e}")
print("\nWrote results/classical_baseline.csv")