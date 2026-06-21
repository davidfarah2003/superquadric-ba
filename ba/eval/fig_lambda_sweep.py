#!/usr/bin/env python3
"""Surface-weight (lambda) sweep at 4 views (report figure).

Pose AUC@5 vs. lambda_surface for the 4-view setting. lambda=0 is the
reprojection-only baseline; lambda=15 is the value used elsewhere in the paper.
Values are read off the benchmark runs:
  lambda=0   -> 39.33   (reprojection-only baseline)
  lambda=15  -> 40.67   (our setting)
  lambda=30  -> 40.67   logs/lamsweep_v4_lam30
  lambda=60  -> 39.33   logs/lamsweep_v4_lam60
  lambda=100 -> 41.33   logs/lamsweep_v4_lam100
  lambda=200 -> 40.33   logs/lamsweep_v4_lam200

Outputs a vector PDF to report/figures/lambda_sweep.pdf.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAM = [0, 15, 30, 60, 100, 200]
AUC = [39.33, 40.67, 40.67, 39.33, 41.33, 40.33]
BASELINE = 39.33  # lambda = 0

C_LINE, C_BASE, C_OURS = "#0072B2", "#999999", "#D55E00"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "report", "figures", "lambda_sweep.pdf"))


def main():
    plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 2.7))

    ax.axhline(BASELINE, color=C_BASE, ls="--", lw=1.4, label=r"baseline ($\lambda=0$)")
    ax.plot(LAM, AUC, marker="o", color=C_LINE, lw=1.8, ms=6, label="BA + prior")

    # Mark the value used in the rest of the paper.
    i = LAM.index(15)
    ax.plot(15, AUC[i], marker="o", color=C_OURS, ms=9, zorder=5)
    ax.annotate(r"$\lambda=15$", (15, AUC[i]), textcoords="offset points",
                xytext=(6, -12), fontsize=9, color=C_OURS)

    ax.set_xlabel(r"Surface weight $\lambda$")
    ax.set_ylabel(r"Pose AUC@5 ($\uparrow$)")
    ax.set_xticks(LAM)
    ax.set_ylim(38.5, 42)
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
