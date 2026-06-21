#!/usr/bin/env python3
"""Pose AUC@5 vs. number of input views (report Figure 3).

The numbers are the verified benchmark results reported in Table 1 of the
report, averaged over the ten ASE scenes (see ba/eval/EXPERIMENTS.md and the
logs/sweep_v{4,6,8,10}_lam{0,15} runs). Kept inline so the figure regenerates
without re-reading the logs.

Outputs a vector PDF to report/figures/auc_views.pdf.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIEWS = [4, 6, 8, 10]
VGGT = [27.7, 20.4, 13.4, 9.3]   # raw feed-forward
BA = [39.3, 29.6, 27.9, 29.4]    # reprojection-only BA (lambda = 0)
OURS = [40.7, 31.1, 28.0, 29.6]  # BA + superquadric prior
# Gain over baseline, computed at full precision then rounded (matches Table 1;
# e.g. 4 views is 40.67 - 39.33 = +1.3, not the +1.4 the rounded points imply).
GAIN = [1.3, 1.5, 0.1, 0.2]

# Okabe-Ito colour-blind-safe palette; Ours highlighted in blue.
C_VGGT, C_BA, C_OURS = "#999999", "#E69F00", "#0072B2"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "report", "figures", "auc_views.pdf"))


def main():
    plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 2.7))

    ax.plot(VIEWS, VGGT, marker="o", ls="--", color=C_VGGT, lw=1.8, ms=6, label="VGGT-only")
    ax.plot(VIEWS, BA, marker="s", ls="-", color=C_BA, lw=1.8, ms=6, label="BA baseline")
    ax.plot(VIEWS, OURS, marker="^", ls="-", color=C_OURS, lw=2.2, ms=7, label="Ours (BA + prior)")

    # Annotate the prior's shrinking gain over the baseline (Table 1 values).
    for v, o, g in zip(VIEWS, OURS, GAIN):
        ax.annotate(f"+{g:.1f}", (v, o), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color=C_OURS)

    ax.set_xlabel("Number of input views")
    ax.set_ylabel(r"Pose AUC@5 ($\uparrow$)")
    ax.set_xticks(VIEWS)
    ax.set_ylim(5, 45)
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
