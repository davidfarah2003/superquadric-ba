#!/usr/bin/env python3
"""Pose AUC@5 by view count (report Figure 3), poster-style.

Two panels: grouped bars for the three configurations across view counts,
so the big bundle-adjustment jump and the small prior bump are both visible
(overlapping lines hid the latter), plus a dedicated panel isolating the
prior's gain. Numbers match Table 1 of the report, averaged over the ten ASE
scenes (logs/sweep_v{4,6,8,10}_lam{0,15}). Kept inline so the figure
regenerates without re-reading the logs.

Outputs a vector PDF to report/figures/auc_views.pdf.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIEWS = [4, 6, 8, 10]
VGGT = [27.7, 20.4, 13.4, 9.3]   # raw feed-forward
BA = [39.3, 29.6, 27.9, 29.4]    # reprojection-only BA (lambda = 0)
OURS = [40.7, 31.1, 28.0, 29.6]  # BA + superquadric prior (lambda = 15)
# Gain over baseline, computed at full precision then rounded (matches Table 1;
# e.g. 4 views is 40.67 - 39.33 = +1.3, not the +1.4 the rounded points imply).
GAIN = [1.3, 1.5, 0.1, 0.2]

# Okabe-Ito colour-blind-safe palette; Ours highlighted in blue.
C_VGGT, C_BA, C_OURS = "#9AA0A6", "#E69F00", "#0072B2"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "report", "figures", "auc_views.pdf"))


def main():
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.45),
                                   gridspec_kw={"width_ratios": [1.85, 1]})

    x = np.arange(len(VIEWS))
    w = 0.27
    ax1.bar(x - w, VGGT, w, color=C_VGGT, label="VGGT-only")
    ax1.bar(x,     BA,   w, color=C_BA,   label="BA baseline")
    ax1.bar(x + w, OURS, w, color=C_OURS, label="Ours (BA + prior)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(VIEWS)
    ax1.set_xlabel("Number of input views")
    ax1.set_ylabel(r"Pose AUC@5 ($\uparrow$)")
    ax1.set_title("AUC@5 by view count", fontsize=10)
    ax1.grid(axis="y", alpha=0.3, lw=0.6)
    ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    ax2.bar(x, GAIN, 0.62, color=C_OURS)
    for xi, g in zip(x, GAIN):
        ax2.text(xi, g + 0.04, f"+{g}", ha="center", va="bottom",
                 fontsize=8.5, color=C_OURS, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(VIEWS)
    ax2.set_xlabel("Input views")
    ax2.set_ylabel(r"$\Delta$ AUC@5")
    ax2.set_title("Gain from prior", fontsize=10)
    ax2.set_ylim(0, max(GAIN) * 1.4)
    ax2.grid(axis="y", alpha=0.3, lw=0.6)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=9, handlelength=1.4, columnspacing=1.6,
               bbox_to_anchor=(0.5, 1.04))

    fig.tight_layout(pad=0.3, w_pad=2.0, rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
