"""Multi-parameter dynamics: the full QCL-vs-classical comparison on f(t, m).

Runs three splits of increasing stringency and, for each, compares QCL against
the matched classical baselines, writing results/multiparam_<split>.csv and a
combined comparison plot.

  interpolation        - random split; can the model REPRESENT the surface?
  time_extrapolation   - train early t, forecast late t (all masses)
  parameter_extrapolation - hold out whole mass slices; predict UNSEEN masses
                            (Ikeda's 'unsupervised prediction' challenge)

Heavy: each mass is a 2^(2N) diagonalization and QCL trains on a 2-D grid.
Run on real hardware, not a short-timeout sandbox. Tune GRID / STEPS / SEEDS below.

Usage:  python -m experiments.multiparam_dynamics_experiment
"""
import sys, os, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.hamiltonian import build_hamiltonian, condensate_op
from src.dynamics import build_dynamics_2d
from src.splits import (interpolation_split, time_extrapolation_split,
                        parameter_extrapolation_split)
from src.classical_model import classical_baselines_2d
from src.model import make_circuit_2d, train_qcl_2d, qcl_param_count

# ----------------------- config (tune for real run) -----------------------
N = 4                       # SU(2): 2N = 8 qubits physical
M_VALUES = np.linspace(0.3, 1.5, 10)
T_MAX, N_T = 12.0, 60
HELD_OUT_MASSES = [M_VALUES[3], M_VALUES[6]]   # unseen masses for param-extrap
T_SPLIT = 8.0
QCL_CONFIGS = [(4, 4), (6, 6), (6, 8)]         # (n_qubits, n_layers)
QCL_STEPS, QCL_SEEDS, QCL_LR = 400, (0, 1, 2), 0.05
M_MAX = float(M_VALUES.max())


def run_split(split_name, X_tr, X_te, y_tr, y_te):
    rows = classical_baselines_2d(X_tr, y_tr, X_te, y_te)
    for nq, nl in QCL_CONFIGS:
        mses = []
        for seed in QCL_SEEDS:
            circ = make_circuit_2d(nq, nl, T_MAX, M_MAX)
            pred = train_qcl_2d(circ, nq, nl, X_tr, y_tr,
                                steps=QCL_STEPS, lr=QCL_LR, seed=seed)
            mses.append(np.mean((pred(X_te) - y_te) ** 2))
        rows.append({"model": f"QCL_{nq}q{nl}L", "n_params": qcl_param_count(nq, nl),
                     "test_mse_mean": float(np.mean(mses)),
                     "test_mse_std": float(np.std(mses))})
    # write CSV
    os.makedirs("results", exist_ok=True)
    path = f"results/multiparam_{split_name}.csv"
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["model", "n_params",
                                           "test_mse_mean", "test_mse_std"])
        wr.writeheader(); wr.writerows(rows)
    print(f"\n[{split_name}] -> {path}")
    for r in rows:
        print(f"  {r['model']:14s} params={r['n_params']:4d}  "
              f"MSE={r['test_mse_mean']:.3e} +/- {r['test_mse_std']:.1e}")
    return rows


def plot_all(results_by_split):
    fig, axes = plt.subplots(1, len(results_by_split),
                             figsize=(6 * len(results_by_split), 5), squeeze=False)
    for ax, (split, rows) in zip(axes[0], results_by_split.items()):
        for r in rows:
            is_qcl = r["model"].startswith("QCL")
            ax.errorbar(r["n_params"], r["test_mse_mean"], yerr=r["test_mse_std"],
                        marker="o" if is_qcl else "s",
                        color="tab:red" if is_qcl else "tab:blue",
                        markersize=8, capsize=3)
            ax.annotate(r["model"], (r["n_params"], r["test_mse_mean"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_yscale("log"); ax.set_xscale("log")
        ax.set_xlabel("trainable parameters"); ax.set_ylabel("test MSE")
        ax.set_title(split); ax.grid(True, which="both", alpha=0.3)
    # legend proxy
    from matplotlib.lines import Line2D
    axes[0][0].legend(handles=[
        Line2D([], [], color="tab:red", marker="o", ls="", label="QCL"),
        Line2D([], [], color="tab:blue", marker="s", ls="", label="classical")],
        fontsize=9)
    fig.suptitle("QCL vs classical: multi-parameter dynamics f(t, m)")
    fig.tight_layout()
    fig.savefig("results/multiparam_comparison.png", dpi=150)
    print("\nsaved results/multiparam_comparison.png")


def main():
    print("building 2-D dynamics dataset (slow: one diagonalization per mass)...")
    X, y, _ = build_dynamics_2d(build_hamiltonian, condensate_op, N,
                                m_values=M_VALUES, t_max=T_MAX, n_t=N_T)
    print(f"dataset: {X.shape[0]} points")

    splits = {
        "interpolation": interpolation_split(X, y),
        "time_extrapolation": time_extrapolation_split(X, y, T_SPLIT),
        "parameter_extrapolation":
            parameter_extrapolation_split(X, y, HELD_OUT_MASSES),
    }
    results = {name: run_split(name, *sp) for name, sp in splits.items()}
    plot_all(results)


if __name__ == "__main__":
    main()