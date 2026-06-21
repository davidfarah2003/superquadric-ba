#!/usr/bin/env python3
"""
exp_significance.py — Statistical significance & variance of the superquadric
surface prior's gain on per-scene pose AUC@5.

Reads EXISTING per-scene benchmark results (no pipeline re-run) from
logs/sweep_v{V}_lam{0,15}/ *_per_scene_results.json, and for each available
view count V computes, paired by scene:

  - per-scene gain = AUC@5(Ours, lambda=15) - AUC@5(Baseline, lambda=0)
  - mean +/- std of the gain
  - bootstrap 95% CI on the mean gain (paired scene resampling)
  - paired significance test: Wilcoxon signed-rank (scipy if available),
    plus a numpy paired sign-flip permutation test as a fallback / cross-check.

Outputs:
  ba/eval/analysis/significance_table.csv
  ba/eval/analysis/significance_table.md
  ba/eval/analysis/significance_gain.png   (per-view-count gain bar+box figure)

CPU-only. Pure-numpy stats; scipy used only for Wilcoxon if present.
"""
import os
import glob
import json
import csv

import numpy as np

try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False

ROOT = "/work/courses/3dv/team39"
LOGS = os.path.join(ROOT, "logs")
OUT_DIR = os.path.join(ROOT, "ba/eval/analysis")

# View counts to look for; (baseline tag, ours tag)
VIEW_COUNTS = [4, 6, 8, 10]
BASELINE_TAG = "lam0"      # lambda = 0  (pure reprojection BA)
OURS_TAG = "lam15.0"       # lambda = 15 (surface prior)
METRIC = "pose_auc_5"
N_BOOT = 20000
SEED = 0


def _find_per_scene(view, tag):
    """Return path to per-scene results json for a given view count and lambda tag, or None."""
    d = os.path.join(LOGS, f"sweep_v{view}_{tag}")
    if not os.path.isdir(d):
        return None
    hits = glob.glob(os.path.join(d, "*per_scene_results.json"))
    return hits[0] if hits else None


def _load_auc(path):
    """Load {scene_id(int): pose_auc_5} from a per-scene results json."""
    with open(path) as f:
        j = json.load(f)
    out = {}
    for k, v in j.items():
        val = v[METRIC]
        if isinstance(val, list):
            val = val[0]
        out[int(k)] = float(val)
    return out


def bootstrap_ci_mean(diffs, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """Percentile bootstrap CI on the mean of paired differences."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi), boot_means


def permutation_test_paired(diffs, n_perm=N_BOOT, seed=SEED):
    """
    Two-sided paired permutation (sign-flip) test on mean of differences.
    H0: the gain distribution is symmetric about 0 (random sign).
    """
    rng = np.random.default_rng(seed + 1)
    n = len(diffs)
    obs = np.abs(diffs.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = np.abs((signs * np.abs(diffs)).mean(axis=1))
    # +1 for the observed statistic (standard practice)
    p = (np.sum(perm_means >= obs - 1e-12) + 1) / (n_perm + 1)
    return float(p)


def analyze_view(view):
    bpath = _find_per_scene(view, BASELINE_TAG)
    opath = _find_per_scene(view, OURS_TAG)
    if bpath is None or opath is None:
        missing = []
        if bpath is None:
            missing.append(f"sweep_v{view}_{BASELINE_TAG}/*per_scene_results.json")
        if opath is None:
            missing.append(f"sweep_v{view}_{OURS_TAG}/*per_scene_results.json")
        return {"view": view, "status": "missing", "missing": missing}

    base = _load_auc(bpath)
    ours = _load_auc(opath)
    scenes = sorted(set(base) & set(ours))
    if not scenes:
        return {"view": view, "status": "no_common_scenes",
                "missing": ["no overlapping scene ids"]}

    b = np.array([base[s] for s in scenes], dtype=float)
    o = np.array([ours[s] for s in scenes], dtype=float)
    diffs = o - b
    n = len(scenes)

    mean_gain = float(diffs.mean())
    std_gain = float(diffs.std(ddof=1)) if n > 1 else float("nan")
    sem = std_gain / np.sqrt(n) if n > 1 else float("nan")
    lo, hi, _ = bootstrap_ci_mean(diffs)

    # Wilcoxon signed-rank (paired). Needs at least one non-zero diff.
    nonzero = np.count_nonzero(diffs)
    wilcoxon_p = None
    wilcoxon_stat = None
    wilcoxon_note = ""
    if HAVE_SCIPY:
        if nonzero == 0:
            wilcoxon_note = "all paired diffs are zero; Wilcoxon undefined"
        else:
            try:
                res = _scipy_wilcoxon(o, b, zero_method="wilcox",
                                      alternative="two-sided")
                wilcoxon_stat = float(res.statistic)
                wilcoxon_p = float(res.pvalue)
            except Exception as e:  # pragma: no cover
                wilcoxon_note = f"wilcoxon failed: {e}"
    else:
        wilcoxon_note = "scipy unavailable"

    perm_p = permutation_test_paired(diffs)

    wins = int(np.sum(diffs > 0))
    losses = int(np.sum(diffs < 0))
    ties = int(np.sum(diffs == 0))

    return {
        "view": view,
        "status": "ok",
        "n_scenes": n,
        "scenes": scenes,
        "baseline_auc": b.tolist(),
        "ours_auc": o.tolist(),
        "gains": diffs.tolist(),
        "baseline_mean": float(b.mean()),
        "ours_mean": float(o.mean()),
        "mean_gain": mean_gain,
        "std_gain": std_gain,
        "sem_gain": sem,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
        "wilcoxon_note": wilcoxon_note,
        "perm_p": perm_p,
        "n_nonzero": int(nonzero),
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = [analyze_view(v) for v in VIEW_COUNTS]
    ok = [r for r in results if r["status"] == "ok"]
    missing = [r for r in results if r["status"] != "ok"]

    # ---- CSV ----
    csv_path = os.path.join(OUT_DIR, "significance_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "view_count", "n_scenes", "baseline_mean_auc5", "ours_mean_auc5",
            "mean_gain", "std_gain", "sem_gain", "ci95_lo", "ci95_hi",
            "wilcoxon_stat", "wilcoxon_p", "permutation_p",
            "wins", "losses", "ties",
        ])
        for r in ok:
            w.writerow([
                r["view"], r["n_scenes"], fmt(r["baseline_mean"]),
                fmt(r["ours_mean"]), fmt(r["mean_gain"]), fmt(r["std_gain"]),
                fmt(r["sem_gain"]), fmt(r["ci95_lo"]), fmt(r["ci95_hi"]),
                fmt(r["wilcoxon_stat"]), fmt(r["wilcoxon_p"], 4),
                fmt(r["perm_p"], 4), r["wins"], r["losses"], r["ties"],
            ])

    # ---- Markdown ----
    md_path = os.path.join(OUT_DIR, "significance_table.md")
    with open(md_path, "w") as f:
        f.write("# Significance & variance of the surface-prior gain "
                "(pose AUC@5)\n\n")
        f.write(f"Baseline = lambda=0 (reprojection-only BA); "
                f"Ours = lambda=15 (superquadric surface prior).\n")
        f.write(f"Gain = AUC@5(Ours) - AUC@5(Baseline), paired per scene. "
                f"Bootstrap = {N_BOOT} resamples; permutation = {N_BOOT} "
                f"sign-flips. scipy={'yes' if HAVE_SCIPY else 'no'}.\n\n")
        f.write("| Views | n | Base mean | Ours mean | Mean gain | Std | "
                "95% CI (boot) | Wilcoxon p | Perm p | W/L/T |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in ok:
            f.write(
                f"| {r['view']} | {r['n_scenes']} | "
                f"{fmt(r['baseline_mean'],2)} | {fmt(r['ours_mean'],2)} | "
                f"{fmt(r['mean_gain'],3)} | {fmt(r['std_gain'],3)} | "
                f"[{fmt(r['ci95_lo'],3)}, {fmt(r['ci95_hi'],3)}] | "
                f"{fmt(r['wilcoxon_p'],4)} | {fmt(r['perm_p'],4)} | "
                f"{r['wins']}/{r['losses']}/{r['ties']} |\n"
            )
        if missing:
            f.write("\n## Missing / unavailable view counts\n\n")
            for r in missing:
                f.write(f"- views={r['view']}: {r['status']} "
                        f"({'; '.join(r.get('missing', []))})\n")
        f.write("\n## Per-scene detail\n\n")
        for r in ok:
            f.write(f"\n### {r['view']} views (n={r['n_scenes']})\n\n")
            f.write("| Scene | Baseline | Ours | Gain |\n|---|---|---|---|\n")
            for s, bb, oo, gg in zip(r["scenes"], r["baseline_auc"],
                                     r["ours_auc"], r["gains"]):
                f.write(f"| {s} | {fmt(bb,2)} | {fmt(oo,2)} | "
                        f"{fmt(gg,2)} |\n")

    # ---- JSON dump (full detail) ----
    json_path = os.path.join(OUT_DIR, "significance.json")
    with open(json_path, "w") as f:
        json.dump({"results": results, "n_boot": N_BOOT,
                   "have_scipy": HAVE_SCIPY}, f, indent=2)

    # ---- Figure ----
    fig_path = os.path.join(OUT_DIR, "significance_gain.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_ok = len(ok)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        # Left: mean gain +/- bootstrap CI bar chart
        ax = axes[0]
        xs = np.arange(n_ok)
        means = [r["mean_gain"] for r in ok]
        los = [r["mean_gain"] - r["ci95_lo"] for r in ok]
        his = [r["ci95_hi"] - r["mean_gain"] for r in ok]
        bars = ax.bar(xs, means, yerr=[los, his], capsize=6,
                      color="#4C72B0", alpha=0.85, edgecolor="k")
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{r['view']} views\n(n={r['n_scenes']})"
                            for r in ok])
        ax.set_ylabel("Mean gain in pose AUC@5 (Ours - Baseline)")
        ax.set_title("Mean prior gain with bootstrap 95% CI")
        for i, r in enumerate(ok):
            pstr = (f"Wilcoxon p={fmt(r['wilcoxon_p'],3)}"
                    if r["wilcoxon_p"] is not None
                    else f"perm p={fmt(r['perm_p'],3)}")
            ax.annotate(pstr, (xs[i], r["ci95_hi"]),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8)

        # Right: per-scene gain boxplot + jittered points
        ax = axes[1]
        data = [np.array(r["gains"]) for r in ok]
        bp = ax.boxplot(data, positions=xs, widths=0.5,
                        showmeans=True, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set(facecolor="#DD8452", alpha=0.5)
        rng = np.random.default_rng(SEED)
        for i, g in enumerate(data):
            jit = rng.normal(0, 0.05, size=len(g))
            ax.scatter(xs[i] + jit, g, color="k", s=18, zorder=3, alpha=0.7)
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{r['view']} views" for r in ok])
        ax.set_ylabel("Per-scene gain in pose AUC@5")
        ax.set_title("Distribution of per-scene gains")

        fig.suptitle("Surface-prior (lambda=15) vs reprojection-only (lambda=0) "
                     "on ASE", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(fig_path, dpi=140)
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        fig_path = f"FIGURE FAILED: {e}"

    # ---- Console summary ----
    print("=" * 70)
    print("SIGNIFICANCE & VARIANCE OF SURFACE-PRIOR GAIN (pose AUC@5)")
    print(f"scipy={'available' if HAVE_SCIPY else 'MISSING'}  "
          f"bootstrap={N_BOOT}  permutation={N_BOOT}")
    print("=" * 70)
    for r in ok:
        print(f"\n[{r['view']} views, n={r['n_scenes']} scenes]")
        print(f"  baseline mean AUC@5 = {fmt(r['baseline_mean'])}  "
              f"ours mean AUC@5 = {fmt(r['ours_mean'])}")
        print(f"  mean gain = {fmt(r['mean_gain'])}  "
              f"std = {fmt(r['std_gain'])}  sem = {fmt(r['sem_gain'])}")
        print(f"  bootstrap 95% CI = [{fmt(r['ci95_lo'])}, {fmt(r['ci95_hi'])}]")
        print(f"  Wilcoxon: stat={fmt(r['wilcoxon_stat'])} "
              f"p={fmt(r['wilcoxon_p'],4)} {r['wilcoxon_note']}")
        print(f"  Permutation p = {fmt(r['perm_p'],4)}")
        print(f"  wins/losses/ties = {r['wins']}/{r['losses']}/{r['ties']} "
              f"(nonzero diffs={r['n_nonzero']})")
    for r in missing:
        print(f"\n[{r['view']} views] UNAVAILABLE ({r['status']}): "
              f"{'; '.join(r.get('missing', []))}")
    print("\nArtifacts:")
    print(" ", csv_path)
    print(" ", md_path)
    print(" ", json_path)
    print(" ", fig_path)


if __name__ == "__main__":
    main()
