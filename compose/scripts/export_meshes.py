#!/usr/bin/env python3
"""Export SuperDec results to a .glb file for easy viewing."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "superdec"))

import numpy as np
import trimesh
from superdec.utils.predictions_handler import PredictionHandler
from superdec.utils.visualizations import generate_ncolors

def export(npz_path, output_path="scene.glb", resolution=30):
    print(f"Loading {npz_path}...")
    predictions = PredictionHandler.from_npz(npz_path)

    print("Generating meshes...")
    meshes = predictions.get_meshes(resolution=resolution)

    num_objects = len(meshes)
    colors = generate_ncolors(num_objects)

    scene = trimesh.Scene()
    added = 0
    for idx, mesh in enumerate(meshes):
        if mesh is None:
            continue
        mesh.visual.face_colors = np.tile(colors[idx], (len(mesh.faces), 1))
        scene.add_geometry(mesh, node_name=f"obj_{idx}")
        added += 1

    print(f"Exporting {added} objects to {output_path}")
    scene.export(output_path)
    print("Done!")

if __name__ == "__main__":
    npz_path = sys.argv[1] if len(sys.argv) > 1 else "data/output_npz/ase_scene_0.npz"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "scene_0.glb"
    export(npz_path, out_path)
