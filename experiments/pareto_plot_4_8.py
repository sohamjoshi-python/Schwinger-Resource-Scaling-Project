"""Plot how the accuracy-cost frontier shifts with physical system size N,
from results/phase4.csv.

Two panels:
  (left)  Pareto frontier (test MSE vs trainable params) overlaid for each N.
          If the frontiers sit on top of each other, the learning difficulty is
          ~independent of N for this observable.
  (right) Minimum cost (params) needed to reach a fixed accuracy target, vs N.
          A flat line = N-independent; an upward slope = larger systems cost more.

Usage:  python -m experiments.plot_scaling
"""
import csv
import matplotlib.pyplot as plt

TARGET = 1e-4   # accuracy target for the min-cost analysis


def load(path="results/phase4.csv"):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "N": int(r["N"]),
                "n_qubits": int(r["n_qubits"]),
                "n_layers": int(r["n_layers"]),
                "n_params": int(r["n_params"]),
                "mse": float(r["test_mse_mean"]),
                "std": float(r["test_mse_std"]),
            })
    return rows


def pareto_front(rows):
    front = []
    for a in rows:
        if not any(
            b is not a
            and b["n_params"] <= a["n_params"] and b["mse"] <= a["mse"]
            and (b["n_params"] < a["n_params"] or b["mse"] < a["mse"])
            for b in rows
        ):
            front.append(a)
    return sorted(front, key=lambda d: d["n_params"])


def main():
    rows = load()
    sizes = sorted({r["N"] for r in rows})
    colors = {4: "tab:blue", 6: "tab:green", 8: "tab:red"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ----- left: overlaid Pareto frontiers per N -----
    for N in sizes:
        sub = [r for r in rows if r["N"] == N]
        front = pareto_front(sub)
        xs = [r["n_params"] for r in front]
        ys = [r["mse"] for r in front]
        ax1.plot(xs, ys, marker="o", color=colors.get(N, "gray"),
                 label=f"N = {N}")
    ax1.set_yscale("log")
    ax1.set_xlabel("trainable parameters (cost)")
    ax1.set_ylabel("test MSE  (log scale)")
    ax1.set_title("Pareto frontier by system size N")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    # ----- right: minimum cost to reach TARGET, vs N -----
    Ns, min_costs = [], []
    for N in sizes:
        feasible = [r for r in rows if r["N"] == N and r["mse"] < TARGET]
        if feasible:
            best = min(feasible, key=lambda r: r["n_params"])
            Ns.append(N)
            min_costs.append(best["n_params"])
    ax2.plot(Ns, min_costs, marker="s", color="black", lw=2)
    ax2.set_xlabel("physical system size N")
    ax2.set_ylabel(f"min params to reach MSE < {TARGET:.0e}")
    ax2.set_title("Resource cost vs system size")
    ax2.set_xticks(sizes)
    ax2.set_ylim(0, max(min_costs) + 4)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/scaling_vs_N.png", dpi=150)
    print("Saved results/scaling_vs_N.png")
    print(f"\nMinimum-cost config reaching MSE < {TARGET:.0e}:")
    for N in sizes:
        feasible = [r for r in rows if r["N"] == N and r["mse"] < TARGET]
        if feasible:
            best = min(feasible, key=lambda r: r["n_params"])
            print(f"  N={N}: {best['n_qubits']}q/{best['n_layers']}L "
                  f"({best['n_params']} params), MSE={best['mse']:.2e}")


if __name__ == "__main__":
    main()