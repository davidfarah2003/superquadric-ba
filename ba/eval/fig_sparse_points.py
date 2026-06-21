#!/usr/bin/env python3
"""Pose AUC@5 vs. triangulated-point budget (report figure).

Source: ba/eval/analysis/sparse_sweep.json (ten-view setting, max_points
subsample). The prior gives its largest gain at a moderate point budget
(around 500 points) and is neutral when points are very sparse.

Outputs a vector PDF to report/figures/sparse_points.pdf.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = [100, 250, 500, 1000]
REPROJ = [19.96, 22.80, 29.78, 30.58]  # BA baseline (lambda = 0)
OURS = [19.82, 22.67, 31.24, 30.84]    # BA + prior (lambda = 15)

C_BA, C_OURS = "#E69F00", "#0072B2"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "report", "figures", "sparse_points.pdf"))


def main():
    plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 2.7))

    ax.plot(N, REPROJ, marker="s", color=C_BA, lw=1.8, ms=6, label="BA baseline")
    ax.plot(N, OURS, marker="^", color=C_OURS, lw=2.2, ms=7, label="Ours (BA + prior)")

    ax.set_xlabel("Number of triangulated points")
    ax.set_ylabel(r"Pose AUC@5 ($\uparrow$)")
    ax.set_xscale("log")
    ax.set_xticks(N)
    ax.set_xticklabels([str(n) for n in N])
    ax.grid(True, alpha=0.3, lw=0.6)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
