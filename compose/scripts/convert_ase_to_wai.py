#!/usr/bin/env python3
"""
Convert ASE (Aria Synthetic Environments) dataset to WAI (World-Aligned Images) format.

This script processes public ASE scenes and outputs data in the universal WAI format,
which includes:
- Undistorted RGB images (rotated to portrait)
- Depth maps converted from range to z-depth (rotated to portrait)
- Validity masks for depth
- Instance segmentation masks (rotated to portrait)
- scene_meta.json with camera parameters and frame metadata

Based on:
- preprocess_ase_scene.py (public ASE processing logic)
- map-anything/ase.py (WAI output format)
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Enable OpenEXR support in OpenCV (must be set before importing cv2)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

try:
    from projectaria_tools.projects import ase as ase_project
    from projectaria_tools.core import calibration as aria_calibration
    from projectaria_tools.core.image import InterpolationMethod
except ImportError:
    print("Error: projectaria_tools not found. Please install it:")
    print("  pip install projectaria-tools")
    sys.exit(1)

# Constants
MAX_UINT_16 = np.iinfo(np.uint16).max
PINHOLE_CAM_KEYS = ["w", "h", "fl_x", "fl_y", "cx", "cy"]

# 90 degree clockwise rotation matrix (for portrait mode)
ROT90_CW = np.array(
    [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
    dtype=np.float32,
)


class DistanceToDepthConverter:
    """
    Converts range images (distance along ray) to depth maps (z-axis depth).
    
    For a pinhole camera:
    - range = distance from camera center to 3D point along the ray
    - depth = z-coordinate of the 3D point in camera frame
    
    Conversion: depth = range / sqrt((x-cx)^2/fx^2 + (y-cy)^2/fy^2 + 1)
    """
    
    def __init__(self, width: int, height: int, fx: float, fy: float, cx: float, cy: float):
        # Create pixel coordinate grids
        u = np.arange(width).astype(np.float32)
        v = np.arange(height).astype(np.float32)
        u, v = np.meshgrid(u, v)
        
        # Compute normalized ray directions
        dx = (u - cx) / fx
        dy = (v - cy) / fy
        
        # Compute ray length factor: sqrt(dx^2 + dy^2 + 1)
        self.ray_length = np.sqrt(dx**2 + dy**2 + 1).astype(np.float32)
    
    def distance_to_depth(self, range_image: np.ndarray) -> np.ndarray:
        """Convert range image to depth map."""
        return range_image / self.ray_length


def rt_transformation_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Computes a 4x4 transformation matrix from rotation and translation.
    
    Args:
        rotation: 3x3 rotation matrix
        translation: 3D translation vector
    
    Returns:
        4x4 transformation matrix
    """
    if translation.shape[-1] == 1:
        translation = translation.squeeze(axis=-1)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def read_trajectory_csv(trajectory_path: Path) -> Dict[str, np.ndarray]:
    """
    Read public ASE trajectory.csv file.
    
    Returns dict with:
        - Ts_world_from_device: [N, 4, 4] transformation matrices
        - timestamps: [N] timestamps
    """
    transforms = []
    timestamps = []
    
    with open(trajectory_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract timestamp
            timestamp = int(row.get('tracking_timestamp_us', row.get('timestamp', 0)))
            timestamps.append(timestamp)
            
            # Extract translation
            translation = np.array([
                float(row['tx_world_device']),
                float(row['ty_world_device']),
                float(row['tz_world_device'])
            ])
            
            # Extract quaternion (xyzw format)
            quat_xyzw = np.array([
                float(row['qx_world_device']),
                float(row['qy_world_device']),
                float(row['qz_world_device']),
                float(row['qw_world_device'])
            ])
            
            # Convert to rotation matrix
            rotation = Rotation.from_quat(quat_xyzw).as_matrix()
            
            # Build transformation matrix
            T = rt_transformation_matrix(rotation, translation)
            transforms.append(T)
    
    return {
        "Ts_world_from_device": np.stack(transforms),
        "timestamps": np.array(timestamps),
    }


def rotate_pinhole_90deg_cw(
    W: int, H: int, fx: float, fy: float, cx: float, cy: float
) -> Tuple[int, int, float, float, float, float]:
    """
    Rotate pinhole camera intrinsics 90 degrees clockwise.
    
    Matches the map-anything implementation.
    
    After rotation:
    - Image size: (H, W) instead of (W, H)
    - Principal point transforms according to 90° CW rotation
    """
    # New dimensions (swapped)
    W_new = H
    H_new = W
    
    # Focal lengths swap (since axes swap)
    fx_new = fy
    fy_new = fx
    
    # Principal point transformation for 90° CW rotation
    # Following map-anything convention
    cy_new = cx
    cx_new = H - 1 - cy
    
    return W_new, H_new, fx_new, fy_new, cx_new, cy_new


def process_ase_scene(
    scene_path: Path,
    output_path: Path,
    rotate_to_portrait: bool = True,
    dataset_name: str = "ASE",
    version: str = "1.0",
) -> None:
    """
    Process a single ASE scene and output in WAI format.
    
    Args:
        scene_path: Path to ASE scene directory (e.g., /path/to/ase/0)
        output_path: Path to output WAI scene directory
        rotate_to_portrait: Whether to rotate images 90° CW to portrait mode
        dataset_name: Name for the dataset in scene_meta
        version: Version string for scene_meta
    """
    scene_name = scene_path.name
    print(f"\nProcessing scene: {scene_name}")
    print(f"  Input: {scene_path}")
    print(f"  Output: {output_path}")
    
    # Define input paths
    rgb_dir = scene_path / "rgb"
    depth_dir = scene_path / "depth"
    instance_dir = scene_path / "instances"
    trajectory_path = scene_path / "trajectory.csv"
    
    # Check required files
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    
    # Load trajectory
    trajectory = read_trajectory_csv(trajectory_path)
    n_poses = len(trajectory["timestamps"])
    print(f"  Loaded {n_poses} camera poses")
    
    # Determine calibration size from first image
    first_rgb = sorted(rgb_dir.glob("vignette*.jpg")) + sorted(rgb_dir.glob("vignette*.png"))
    if not first_rgb:
        raise FileNotFoundError(f"No RGB images found in {rgb_dir}")
    
    first_img = Image.open(first_rgb[0])
    orig_w, orig_h = first_img.size
    
    # Determine calibration size (ASE uses 704 or 1408)
    if orig_w == 704 or orig_h == 704:
        calibration_size = 704
    elif orig_w == 1408 or orig_h == 1408:
        calibration_size = 1408
    else:
        calibration_size = 704
        print(f"  Warning: Non-standard image size {orig_w}x{orig_h}, using 704 calibration")
    
    # Get ASE RGB calibration (fisheye) - source calibration
    device_calib = ase_project.get_ase_rgb_calibration(calibration_size)
    
    # Rescale source calibration if image size differs
    calib_size = device_calib.get_image_size()
    if (orig_w, orig_h) != (calib_size[0], calib_size[1]):
        scale = orig_w / calib_size[0]
        device_calib = device_calib.rescale((orig_w, orig_h), scale, (0, 0))
        print(f"  Rescaled source calibration from {calib_size} to ({orig_w}, {orig_h})")
    
    # Get device-to-camera transform
    T_device_from_camera = device_calib.get_transform_device_camera().to_matrix()
    if rotate_to_portrait:
        # Apply 90° CW rotation to the camera frame
        T_device_from_camera[:3, :3] = T_device_from_camera[:3, :3] @ ROT90_CW
    
    # Compute camera-to-world transforms for all frames
    cam2worlds = np.array([
        T_world_device @ T_device_from_camera 
        for T_world_device in trajectory["Ts_world_from_device"]
    ])
    
    # Create linear (pinhole) calibration for undistortion target
    # Use fixed 512x512 output with focal length 280
    linear_calib = aria_calibration.get_linear_camera_calibration(512, 512, 280)
    
    # Get intrinsics from linear calibration
    fx, fy = linear_calib.get_focal_lengths().tolist()
    cx, cy = linear_calib.get_principal_point().tolist()
    W, H = int(linear_calib.get_image_size()[0]), int(linear_calib.get_image_size()[1])
    
    print(f"  Output resolution: {W}x{H}, focal: {fx}")
    
    # Create distance-to-depth converter (before rotation)
    dist2depth = DistanceToDepthConverter(W, H, fx, fy, cx, cy)
    
    # Adjust intrinsics for portrait rotation
    if rotate_to_portrait:
        W, H, fx, fy, cx, cy = rotate_pinhole_90deg_cw(W, H, fx, fy, cx, cy)
    
    # Create output directories
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "depth").mkdir(parents=True, exist_ok=True)
    (output_path / "masks").mkdir(parents=True, exist_ok=True)
    if instance_dir.exists():
        (output_path / "instances").mkdir(parents=True, exist_ok=True)
    
    # Process frames
    wai_frames = []
    rgb_paths = sorted(rgb_dir.glob("vignette*.jpg")) + sorted(rgb_dir.glob("vignette*.png"))
    
    if len(rgb_paths) != n_poses:
        print(f"  Warning: Found {len(rgb_paths)} RGB images but {n_poses} poses")
    
    print(f"  Processing {min(len(rgb_paths), n_poses)} frames...")
    
    for idx in tqdm(range(min(len(rgb_paths), n_poses)), desc=f"  Scene {scene_name}"):
        frame_idx = f"{idx:07d}"
        frame_name = f"frame_{frame_idx}"
        
        # --- Process RGB ---
        rgb_path = rgb_dir / f"vignette{frame_idx}.jpg"
        if not rgb_path.exists():
            rgb_path = rgb_dir / f"vignette{frame_idx}.png"
        
        if rgb_path.exists():
            img = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
            
            # Undistort from fisheye to pinhole
            img_undist = aria_calibration.distort_by_calibration(
                img, linear_calib, device_calib, InterpolationMethod.BILINEAR
            )
            
            if rotate_to_portrait:
                # 90° clockwise rotation (matching map-anything convention)
                img_undist = np.rot90(img_undist, axes=(1, 0))
            
            # Save as JPG
            target_image_path = f"images/{frame_name}.jpg"
            Image.fromarray(img_undist).save(output_path / target_image_path, quality=95)
        else:
            target_image_path = None
        
        # --- Process Depth (range -> depth) ---
        depth_path = depth_dir / f"depth{frame_idx}.png"
        target_depth_path = None
        target_mask_path = None
        
        if depth_path.exists():
            # Load range image as uint16 (millimeters)
            range_img = np.array(Image.open(depth_path), dtype=np.uint16)
            
            # Create validity mask (0 and MAX are invalid)
            mask = np.ones_like(range_img, dtype=np.uint8)
            mask[np.logical_or(range_img == 0, range_img == MAX_UINT_16)] = 0
            
            # Convert to float32 meters
            range_float = range_img.astype(np.float32) / 1000.0
            range_float[mask == 0] = 0
            
            # Undistort range image
            range_undist = aria_calibration.distort_by_calibration(
                range_float, linear_calib, device_calib, InterpolationMethod.BILINEAR
            )
            mask_undist = aria_calibration.distort_by_calibration(
                mask, linear_calib, device_calib, InterpolationMethod.NEAREST_NEIGHBOR
            )
            
            # Convert range to depth (z-axis)
            depth_undist = dist2depth.distance_to_depth(range_undist)
            
            if rotate_to_portrait:
                # 90° clockwise rotation (matching map-anything convention)
                depth_undist = np.rot90(depth_undist, axes=(1, 0))
                mask_undist = np.rot90(mask_undist, axes=(1, 0))
            
            # Save depth as EXR (float32 meters) - standard WAI format
            target_depth_path = f"depth/{frame_name}.exr"
            cv2.imwrite(str(output_path / target_depth_path), depth_undist.astype(np.float32))
            
            # Save validity mask
            target_mask_path = f"masks/{frame_name}.png"
            mask_uint8 = (mask_undist * 255).astype(np.uint8)
            Image.fromarray(mask_uint8).save(output_path / target_mask_path)
        
        # --- Process Instance Mask ---
        instance_path = instance_dir / f"instance{frame_idx}.png"
        target_instance_path = None
        
        if instance_path.exists():
            instance_img = np.array(Image.open(instance_path), dtype=np.uint16)
            
            # Use distort_label_by_calibration for segmentation maps
            # This is specifically designed for discrete label values
            instance_undist = aria_calibration.distort_label_by_calibration(
                instance_img, linear_calib, device_calib
            )
            
            if rotate_to_portrait:
                # 90° clockwise rotation (matching map-anything convention)
                instance_undist = np.rot90(instance_undist, axes=(1, 0))
            
            # Save as uint16 PNG
            target_instance_path = f"instances/{frame_name}.png"
            Image.fromarray(instance_undist.astype(np.uint16), mode="I;16").save(
                output_path / target_instance_path
            )
        
        # Build WAI frame entry
        wai_frame = {
            "frame_name": frame_name,
            "transform_matrix": cam2worlds[idx].tolist(),
            # Per-frame intrinsics
            "w": W,
            "h": H,
            "fl_x": fx,
            "fl_y": fy,
            "cx": cx,
            "cy": cy,
        }
        
        if target_image_path:
            wai_frame["image"] = target_image_path
            wai_frame["file_path"] = target_image_path  # Alias for compatibility
        if target_depth_path:
            wai_frame["depth"] = target_depth_path
        if target_mask_path:
            wai_frame["mask_path"] = target_mask_path
        if target_instance_path:
            wai_frame["instance"] = target_instance_path
        
        wai_frames.append(wai_frame)
    
    if len(wai_frames) == 0:
        raise RuntimeError("Processed 0 frames")
    
    # Build scene_meta.json
    scene_meta = {
        "scene_name": scene_name,
        "dataset_name": dataset_name,
        "version": version,
        "shared_intrinsics": True,  # All frames have same intrinsics
        "camera_model": "PINHOLE",
        "camera_convention": "opencv",
        "scale_type": "metric",
        # Shared intrinsics at scene level
        "w": W,
        "h": H,
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        # Frame list
        "frames": wai_frames,
        # Modality descriptions
        "frame_modalities": {
            "image": {"frame_key": "image", "format": "image"},
            "depth": {"frame_key": "depth", "format": "depth"},
            "mask": {"frame_key": "mask_path", "format": "binary"},
        },
    }
    
    if instance_dir.exists():
        scene_meta["frame_modalities"]["instance"] = {
            "frame_key": "instance", 
            "format": "segmentation"
        }
    
    # Record applied transforms
    if rotate_to_portrait:
        scene_meta["_applied_transform"] = ROT90_CW.tolist()
        scene_meta["_applied_transforms"] = {"image_rotation": ROT90_CW.tolist()}
    else:
        scene_meta["_applied_transform"] = np.eye(3).tolist()
        scene_meta["_applied_transforms"] = {}
    
    # Save scene_meta.json
    with open(output_path / "scene_meta.json", "w") as f:
        json.dump(scene_meta, f, indent=2)
    
    # Copy metadata files from source (for bounding spheres, superquadric GT, etc.)
    import shutil
    metadata_files = [
        "object_instances_to_classes.json",
        "sq.npz",
    ]
    for fname in metadata_files:
        src_file = scene_path / fname
        if src_file.exists():
            shutil.copy2(src_file, output_path / fname)
            print(f"  Copied {fname}")
    
    print(f"  ✓ Scene {scene_name} converted successfully!")
    print(f"    Output: {output_path}")
    print(f"    Frames: {len(wai_frames)}")
    print(f"    Image size: {W}x{H}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ASE dataset to WAI format"
    )
    parser.add_argument(
        "--scene_path",
        type=str,
        required=True,
        help="Path to ASE scene directory (e.g., /path/to/ase/0)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to output WAI scene directory. Default: <scene_path>_wai",
    )
    parser.add_argument(
        "--no_rotate",
        action="store_true",
        help="Don't rotate images to portrait mode",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ASE",
        help="Dataset name for scene_meta.json",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0",
        help="Version string for scene_meta.json",
    )
    
    args = parser.parse_args()
    
    scene_path = Path(args.scene_path)
    if not scene_path.exists():
        print(f"Error: Scene path does not exist: {scene_path}")
        sys.exit(1)
    
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = scene_path.parent / f"{scene_path.name}_wai"
    
    process_ase_scene(
        scene_path=scene_path,
        output_path=output_path,
        rotate_to_portrait=not args.no_rotate,
        dataset_name=args.dataset_name,
        version=args.version,
    )


if __name__ == "__main__":
    main()
