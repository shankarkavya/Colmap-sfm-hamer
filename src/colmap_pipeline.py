"""
COLMAP SfM pipeline for sequential RGB images with known intrinsics.
Camera: 1280x720, FULL_OPENCV (rational_polynomial distortion)
"""

import os
import shutil
import pycolmap
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
IMAGE_DIR  = BASE / "rgb_undistorted"   # undistorted images
DB_PATH    = BASE / "database.db"
SPARSE_DIR = BASE / "sparse"

# ── Camera intrinsics — PINHOLE on undistorted images ─────────────────────────
# fx, fy, cx, cy  (from undistort_images.py output)
CAMERA_MODEL  = "PINHOLE"
CAMERA_PARAMS = "638.7274186439614,638.5936105236774,642.252823408419,368.0877567316025"

# ── Setup ──────────────────────────────────────────────────────────────────────
SPARSE_DIR.mkdir(exist_ok=True)
if DB_PATH.exists():
    DB_PATH.unlink()

print("=" * 60)
print("  COLMAP SfM Pipeline")
print("=" * 60)
images = sorted([f for f in IMAGE_DIR.iterdir() if f.suffix == ".jpg"])
print(f"  Images : {len(images)}")
print(f"  Camera : {CAMERA_MODEL}")
print(f"  DB     : {DB_PATH}")
print(f"  Output : {SPARSE_DIR}")
print("=" * 60)

# ── Step 1: Feature Extraction ─────────────────────────────────────────────────
print("\n[1/3] Extracting features ...")

reader_opts = pycolmap.ImageReaderOptions()
reader_opts.camera_model  = CAMERA_MODEL
reader_opts.camera_params = CAMERA_PARAMS

pycolmap.extract_features(
    database_path    = DB_PATH,
    image_path       = IMAGE_DIR,
    camera_mode      = pycolmap.CameraMode.SINGLE,   # one shared camera
    reader_options   = reader_opts,
    device           = pycolmap.Device.auto,
)
print("  [✓] Feature extraction done")

# ── Step 2: Sequential Matching ────────────────────────────────────────────────
print("\n[2/3] Sequential feature matching ...")

pairing_opts = pycolmap.SequentialPairingOptions()
pairing_opts.overlap = 10          # match each frame to its 10 neighbours
pairing_opts.loop_detection = False  # no vocab tree available

pycolmap.match_sequential(
    database_path  = DB_PATH,
    pairing_options= pairing_opts,
    device         = pycolmap.Device.auto,
)
print("  [✓] Sequential matching done")

# ── Step 3: Incremental Mapping (SfM) ─────────────────────────────────────────
print("\n[3/3] Incremental mapping ...")

maps = pycolmap.incremental_mapping(
    database_path = DB_PATH,
    image_path    = IMAGE_DIR,
    output_path   = SPARSE_DIR,
)

print(f"\n{'=' * 60}")
print(f"  Reconstruction complete — {len(maps)} model(s)")
for idx, rec in maps.items():
    print(f"  Model {idx}: {rec.num_reg_images()} images | {rec.num_points3D()} 3D points")
print(f"{'=' * 60}")
print(f"\nResults written to: {SPARSE_DIR}")
print("Visualise with:  rerun colmap_reconstruction.rrd  (after running visualize_rerun.py)")
