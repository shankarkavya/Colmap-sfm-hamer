"""
COLMAP dense reconstruction pipeline (MVS).
Requires the sparse reconstruction at sparse/1 and undistorted images.
Steps:
  1. image_undistorter  — set up dense workspace
  2. patch_match_stereo — dense depth maps (GPU)
  3. stereo_fusion      — fuse depth maps into dense point cloud
"""

import subprocess
import sys
from pathlib import Path

BASE        = Path(__file__).parent
SPARSE_DIR  = BASE / "sparse" / "1"
IMAGE_DIR   = BASE / "rgb_undistorted"
DENSE_DIR   = BASE / "dense"
COLMAP_BIN  = "/home/user/shankark2/miniconda3/envs/slam_env/bin/colmap"

DENSE_DIR.mkdir(exist_ok=True)

def run(cmd, step):
    print(f"\n{'=' * 60}")
    print(f"  {step}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, text=True, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(f"[ERROR] {step} failed (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"  [✓] Done")

# ── 1. Undistorter ─────────────────────────────────────────────────────────────
# Sets up the proper dense workspace structure COLMAP expects.
# Since our images are already undistorted (PINHOLE), this is fast.
run([
    COLMAP_BIN, "image_undistorter",
    "--image_path",        str(IMAGE_DIR),
    "--input_path",        str(SPARSE_DIR),
    "--output_path",       str(DENSE_DIR),
    "--output_type",       "COLMAP",
    "--max_image_size",    "1280",
], "Step 1/3: image_undistorter")

# ── 2. PatchMatch Stereo (GPU) ─────────────────────────────────────────────────
run([
    COLMAP_BIN, "patch_match_stereo",
    "--workspace_path",    str(DENSE_DIR),
    "--workspace_format",  "COLMAP",
    "--PatchMatchStereo.gpu_index",    "0",
    "--PatchMatchStereo.depth_min",    "0.1",
    "--PatchMatchStereo.depth_max",    "20.0",
    "--PatchMatchStereo.window_radius","5",
    "--PatchMatchStereo.num_samples",  "15",
    "--PatchMatchStereo.num_iterations","5",
    "--PatchMatchStereo.geom_consistency", "true",
], "Step 2/3: patch_match_stereo (GPU)")

# ── 3. Stereo Fusion ───────────────────────────────────────────────────────────
FUSED_PLY = DENSE_DIR / "fused.ply"
run([
    COLMAP_BIN, "stereo_fusion",
    "--workspace_path",    str(DENSE_DIR),
    "--workspace_format",  "COLMAP",
    "--input_type",        "geometric",
    "--output_path",       str(FUSED_PLY),
    "--StereoFusion.min_num_pixels", "3",
], "Step 3/3: stereo_fusion")

print(f"\n{'=' * 60}")
print(f"  Dense reconstruction complete!")
print(f"  Output: {FUSED_PLY}")
print(f"{'=' * 60}")
