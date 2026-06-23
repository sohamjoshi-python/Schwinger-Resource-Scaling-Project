"""Plot the accuracy-vs-cost Pareto frontier from results/phase2.csv.

Reads the sweep table, plots test MSE (log scale) vs trainable-parameter count
with each qubit-count as its own series, and marks the Pareto-optimal points
(the lower-left envelope: no other config is both cheaper AND more accurate).

Usage:  python -m experiments.plot_pareto
"""
import csv
import matplotlib.pyplot as plt


def load(path="results/phase2.csv"):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "n_qubits": int(r["n_qubits"]),
                "n_layers": int(r["n_layers"]),
                "n_params": int(r["n_params"]),
                "mse": float(r["test_mse_mean"]),
                "std": float(r["test_mse_std"]),
            })
    return rows


def pareto_front(rows):
    """A config is Pareto-optimal if no other config has both fewer (or equal)
    params and lower (or equal) MSE. Lower is better on both axes."""
    front = []
    for a in rows:
        dominated = False
        for b in rows:
            if b is a:
                continue
            if (b["n_params"] <= a["n_params"] and b["mse"] <= a["mse"]
                    and (b["n_params"] < a["n_params"] or b["mse"] < a["mse"])):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda d: d["n_params"])


def main():
    rows = load()
    colors = {2: "tab:blue", 3: "tab:green", 4: "tab:red"}

    plt.figure(figsize=(8, 5.5))

    # one series per qubit count
    for nq in sorted({r["n_qubits"] for r in rows}):
        series = sorted([r for r in rows if r["n_qubits"] == nq],
                        key=lambda d: d["n_params"])
        xs = [r["n_params"] for r in series]
        ys = [r["mse"] for r in series]
        errs = [r["std"] for r in series]
        plt.errorbar(xs, ys, yerr=errs, marker="o", capsize=3,
                     color=colors.get(nq, "gray"), label=f"{nq} qubits", alpha=0.8)

    # Pareto frontier
    front = pareto_front(rows)
    fx = [r["n_params"] for r in front]
    fy = [r["mse"] for r in front]
    plt.plot(fx, fy, "k--", lw=1.5, zorder=1, label="Pareto frontier")
    plt.scatter(fx, fy, s=160, facecolors="none", edgecolors="black",
                linewidths=1.8, zorder=6)

    # annotate frontier points with their (qubits, layers) config
    for r in front:
        plt.annotate(f"{r['n_qubits']}q/{r['n_layers']}L",
                     (r["n_params"], r["mse"]),
                     textcoords="offset points", xytext=(8, 8), fontsize=9)

    plt.yscale("log")
    plt.xlabel("trainable parameters (cost)")
    plt.ylabel("test MSE  (log scale)")
    plt.title("QCL accuracy vs cost for the Schwinger condensate (N=4)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/pareto_frontier_n4.png", dpi=150)
    print("Saved results/pareto_frontier_n4.png")
    print("\nPareto-optimal configurations:")
    for r in front:
        print(f"  {r['n_qubits']}q / {r['n_layers']}L  "
              f"({r['n_params']} params)  MSE={r['mse']:.2e}")


if __name__ == "__main__":
    main()