"""Test script for superdec installation and basic functionality.

Can be run from anywhere — automatically resolves project root.

On login node (CPU-only tests):
    python test_superdec.py

On compute node (all tests including GPU):
    srun --account=3dv --gpus=1 --mem=24G --time=00:10:00 --pty bash -c \
        'source /work/courses/3dv/team39/envs/3dv/bin/activate && python /work/courses/3dv/team39/compose/test_superdec.py'
"""

import os
import sys

# Resolve project root regardless of where the script is run from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

HAS_CUDA = False
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    pass


def requires_cuda(fn):
    """Decorator to skip tests that need CUDA."""
    def wrapper(*args, **kwargs):
        if not HAS_CUDA:
            print("   SKIP — requires CUDA (run on compute node)")
            return None
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


def test_imports():
    """Test that CPU-safe modules import correctly."""
    print("1. Testing imports...")
    import torch
    import numpy as np
    from omegaconf import OmegaConf

    from superdec.superdec import SuperDec
    from superdec.models.heads import SuperDecHead
    from superdec.models.decoder import TransformerDecoder
    from superdec.data.dataloader import normalize_points, denormalize_outdict, denormalize_points
    from superdec.data.transform import rotate_around_axis
    from superdec.utils.predictions_handler import PredictionHandler

    print("   All imports OK")


@requires_cuda
def test_model_creation():
    """Test model instantiation from the training config. Needs CUDA for PVCNN."""
    print("2. Testing model creation...")
    from omegaconf import OmegaConf
    from superdec.superdec import SuperDec

    config = OmegaConf.load("superdec/configs/train.yaml")
    model = SuperDec(config.superdec)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model created: {n_params:,} parameters")
    return model


@requires_cuda
def test_forward_pass():
    """Test a forward pass with dummy point clouds. Needs CUDA for PVCNN."""
    print("3. Testing forward pass...")
    import torch
    from omegaconf import OmegaConf
    from superdec.superdec import SuperDec

    config = OmegaConf.load("superdec/configs/train.yaml")
    model = SuperDec(config.superdec).to("cuda").eval()

    # SuperDec expects (B, N, 3) point clouds with N=4096
    dummy_input = torch.randn(2, 4096, 3, device="cuda")

    with torch.no_grad():
        outdict = model(dummy_input)

    expected_keys = {"scale", "shape", "rotate", "trans", "exist"}
    assert expected_keys.issubset(outdict.keys()), f"Missing keys: {expected_keys - outdict.keys()}"

    B, N_queries = 2, 16
    assert outdict["scale"].shape == (B, N_queries, 3), f"scale shape: {outdict['scale'].shape}"
    assert outdict["shape"].shape == (B, N_queries, 2), f"shape shape: {outdict['shape'].shape}"
    assert outdict["rotate"].shape == (B, N_queries, 3, 3), f"rotate shape: {outdict['rotate'].shape}"
    assert outdict["trans"].shape == (B, N_queries, 3), f"trans shape: {outdict['trans'].shape}"
    assert outdict["exist"].shape == (B, N_queries, 1), f"exist shape: {outdict['exist'].shape}"

    print(f"   Forward pass OK — output keys: {list(outdict.keys())}")
    print(f"   Predicted {N_queries} superquadrics per sample")


def test_normalize_roundtrip():
    """Test that normalize/denormalize is consistent."""
    print("4. Testing point normalization roundtrip...")
    import numpy as np
    import torch
    from superdec.data.dataloader import normalize_points, denormalize_points

    points = np.random.randn(4096, 3).astype(np.float32) * 5 + np.array([10, -3, 7])
    normed, translation, scale = normalize_points(points)

    assert abs(normed.mean()) < 0.1, f"Mean too large: {normed.mean()}"
    assert normed.std() < 2.0, f"Std too large: {normed.std()}"

    normed_t = torch.from_numpy(normed).unsqueeze(0).float()
    translation = np.array([translation])
    scale = np.array([scale])
    recovered = denormalize_points(normed_t, translation, scale, z_up=False)
    recovered = recovered.squeeze(0).numpy()

    error = np.abs(recovered - points).max()
    assert error < 1e-4, f"Roundtrip error too large: {error}"
    print(f"   Roundtrip error: {error:.2e} — OK")


def test_example_pointcloud():
    """Test loading the included example chair.ply."""
    print("5. Testing example point cloud loading...")
    ply_path = "superdec/examples/chair.ply"
    if not os.path.exists(ply_path):
        print(f"   SKIP — {ply_path} not found")
        return

    import open3d as o3d
    import numpy as np

    pc = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pc.points)
    print(f"   Loaded {points.shape[0]} points, bounds: [{points.min(axis=0)}] to [{points.max(axis=0)}]")


if __name__ == "__main__":
    passed = 0
    failed = 0
    skipped = 0

    tests = [test_imports, test_model_creation, test_forward_pass,
             test_normalize_roundtrip, test_example_pointcloud]

    for fn in tests:
        try:
            result = fn()
            if result is None and hasattr(fn, '__wrapped__'):
                skipped += 1
            else:
                passed += 1
        except Exception as e:
            print(f"   FAIL: {e}")
            failed += 1

    print(f"\n{'='*40}")
    if not HAS_CUDA:
        print("Note: no CUDA available — GPU tests skipped")
        print("Run on a compute node to test model creation & forward pass")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
