"""Poster explainer: 'What is a super-quadric?'

A compact horizontal strip morphing one super-quadric through its shape family by
sweeping the roundness exponent epsilon (eps1=eps2): cuboid -> rounded box ->
sphere -> bulged -> octahedron. Two exponents + three half-extents (a1,a2,a3) and
a pose describe each primitive, so a handful tile a whole room (cf. the scene
figure). Pure geometry, no data load.

Writes poster/figures/superquadric_family.png.
"""
import os, sys
import numpy as np
sys.path.insert(0, "/work/courses/3dv/team39/ba/eval")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

OUT = "/work/courses/3dv/team39/poster/figures/superquadric_family.png"
FACE = "#3F74B8"   # ETH-blue-ish surface
EDGE = "#16365C"
DARK = "#123362"


def sq_surface(a, eps, n=40):
    e1 = float(np.clip(eps[0], 0.06, 1.94)); e2 = float(np.clip(eps[1], 0.06, 1.94))
    eta = np.linspace(-np.pi / 2, np.pi / 2, n)
    om = np.linspace(-np.pi, np.pi, 2 * n)
    E, O = np.meshgrid(eta, om)
    c = lambda t, p: np.sign(np.cos(t)) * np.abs(np.cos(t)) ** p
    s = lambda t, p: np.sign(np.sin(t)) * np.abs(np.sin(t)) ** p
    x = a[0] * c(E, e1) * c(O, e2)
    y = a[1] * c(E, e1) * s(O, e2)
    z = a[2] * s(E, e1)
    return x, y, z


# (epsilon, label, sublabel)
PANELS = [
    (0.2, r"$\varepsilon=0.2$", "cuboid"),
    (0.6, r"$\varepsilon=0.6$", "rounded box"),
    (1.0, r"$\varepsilon=1.0$", "sphere"),
    (1.5, r"$\varepsilon=1.5$", "bulged"),
    (1.9, r"$\varepsilon=1.9$", "octahedron"),
]


def main():
    fig = plt.figure(figsize=(15.5, 3.7))
    ls = LightSource(azdeg=120, altdeg=55)
    for i, (eps, lab, sub) in enumerate(PANELS):
        ax = fig.add_subplot(1, len(PANELS), i + 1, projection="3d")
        x, y, z = sq_surface((1, 1, 1), (eps, eps))
        rgb = ls.shade(z, cmap=plt.get_cmap("Blues"), vmin=z.min() - 0.4, vmax=z.max() + 0.1,
                       blend_mode="soft")
        ax.plot_surface(x, y, z, facecolors=rgb, linewidth=0, antialiased=True,
                        rcount=40, ccount=40, shade=False)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=35)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_axis_off()
        ax.set_title(lab, fontsize=23, fontweight="bold", color=DARK, pad=-2)
        ax.text2D(0.5, -0.02, sub, transform=ax.transAxes, ha="center", va="top",
                  fontsize=17, color="#444444")
    # No suptitle: the LaTeX caption carries the explanation (avoids a redundant,
    # em-dashed second caption baked into the image).
    fig.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.06, wspace=0.0)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
