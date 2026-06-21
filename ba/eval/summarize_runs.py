#!/usr/bin/env python3
# =============================================================================
# summarize_runs.py -- aggregate scaled benchmark runs into one comparison table
# =============================================================================
#
# WHAT IT DOES
#   Walks a directory of benchmark run folders (e.g. the views_<N>[_<mode>]/
#   dirs produced by scale_ase_views.sh) and collects the headline metrics from
#   each run's `per_dataset_results.json` (pose_auc_5, pose_ate_rmse, etc.). It
#   parses the view count and BA mode out of the folder name and prints a tidy
#   table sorted by (views, mode), plus optionally writes a CSV + JSON summary.
#
#   This is pure stdlib (json/os/argparse) -- NO torch, NO GPU. It only reads
#   already-written result files, so it is safe to run on the login node or any
#   CPU shell.
#
# HOW TO LAUNCH (login node or interactive shell is fine -- no GPU)
#   source /work/courses/3dv/team39/envs/3dv/bin/activate   # (or plain python3)
#   python ba/eval/summarize_runs.py \
#       --runs_root /work/courses/3dv/team39/logs/benchmark_sparse_ase_vggt
#   # write machine-readable outputs under ba/eval/analysis/:
#   python ba/eval/summarize_runs.py \
#       --runs_root /work/courses/3dv/team39/logs/benchmark_sparse_ase_vggt \
#       --out_csv ba/eval/analysis/scaled_runs.csv \
#       --out_json ba/eval/analysis/scaled_runs.json
#
#   Flags:
#     --runs_root  dir whose immediate subdirs are benchmark run folders (required)
#     --glob       subdir name pattern (default 'views_*'); use '*' for all
#     --metrics    comma list of metrics to show (default
#                  pose_auc_5,pose_ate_rmse,ray_dirs_err_deg,pointmaps_abs_rel)
#     --out_csv    optional CSV output path
#     --out_json   optional JSON output path
#
# MONITOR: runs in <1 s; just read its stdout.
# =============================================================================
import argparse
import csv
import fnmatch
import json
import os
import re

DEFAULT_METRICS = [
    "pose_auc_5", "pose_ate_rmse", "ray_dirs_err_deg", "pointmaps_abs_rel",
]


def parse_run_name(name):
    """views_8_superbundle_surface -> (8, 'superbundle_surface'); views_4 -> (4,'none')."""
    m = re.match(r"views_(\d+)(?:_(.+))?$", name)
    if m:
        return int(m.group(1)), (m.group(2) or "none")
    return None, name  # unknown layout: keep raw name, no view count


def find_dataset_block(per_dataset):
    """Prefer the 'Average' block; else the single non-Average dataset block."""
    if "Average" in per_dataset:
        return per_dataset["Average"]
    keys = [k for k in per_dataset if k != "Average"]
    return per_dataset[keys[0]] if keys else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--glob", default="views_*")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    rows = []
    for name in sorted(os.listdir(args.runs_root)):
        d = os.path.join(args.runs_root, name)
        if not os.path.isdir(d) or not fnmatch.fnmatch(name, args.glob):
            continue
        pj = os.path.join(d, "per_dataset_results.json")
        if not os.path.isfile(pj):
            continue
        try:
            block = find_dataset_block(json.load(open(pj)))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warn] skip {name}: {e}")
            continue
        nv, mode = parse_run_name(name)
        row = {"run": name, "views": nv, "mode": mode}
        for m in metrics:
            row[m] = block.get(m)
        rows.append(row)

    if not rows:
        print(f"No runs with per_dataset_results.json under {args.runs_root} "
              f"matching '{args.glob}'.")
        return

    # sort: numeric views first (None last), then mode
    rows.sort(key=lambda r: (r["views"] is None, r["views"] or 0, r["mode"]))

    # pretty table
    headers = ["views", "mode"] + metrics
    widths = {h: len(h) for h in headers}
    fmt = {}
    for r in rows:
        fmt[r["run"]] = {}
        for h in headers:
            v = r.get(h)
            if isinstance(v, float):
                s = f"{v:.4f}"
            elif v is None:
                s = "-"
            else:
                s = str(v)
            fmt[r["run"]][h] = s
            widths[h] = max(widths[h], len(s))
    line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("  ".join("-" * widths[h] for h in headers))
    for r in rows:
        print("  ".join(fmt[r["run"]][h].ljust(widths[h]) for h in headers))

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers + ["run"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in headers + ["run"]})
        print(f"\n[wrote] {args.out_csv}")
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        json.dump(rows, open(args.out_json, "w"), indent=1)
        print(f"[wrote] {args.out_json}")


if __name__ == "__main__":
    main()
