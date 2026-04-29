#!/usr/bin/env python3
"""
Generate the dataset metadata files needed to run the map-anything benchmark on ASE WAI data.

Creates:
  - <metadata_dir>/test/ase_scene_list_test.npy  (scene name list)

Usage:
    python scripts/prepare_benchmark_data.py --wai_root data/wai --metadata_dir data/dataset_metadata
"""

import argparse
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Prepare ASE benchmark metadata")
    parser.add_argument(
        "--wai_root",
        type=str,
        default="/work/courses/3dv/team39/compose/data/wai",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default="/work/courses/3dv/team39/compose/data/dataset_metadata",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
    )
    args = parser.parse_args()

    wai_root = Path(args.wai_root)
    metadata_dir = Path(args.metadata_dir)

    # Find all scenes with scene_meta.json
    scene_names = sorted(
        d.name
        for d in wai_root.iterdir()
        if d.is_dir() and (d / "scene_meta.json").exists()
    )

    print(f"Found {len(scene_names)} scenes: {scene_names}")

    # Save scene list
    out_dir = metadata_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ase_scene_list_{args.split}.npy"
    np.save(str(out_path), np.array(scene_names))
    print(f"Saved scene list to {out_path}")


if __name__ == "__main__":
    main()
