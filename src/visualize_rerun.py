"""
Visualise COLMAP reconstruction with Rerun.
- Dense point cloud (dense/fused.ply) if available, else sparse
- Cameras appear one-by-one as the trajectory plays back
"""

import rerun as rr
import rerun.blueprint as rrb
import pycolmap
import numpy as np
from pathlib import Path

try:
    from plyfile import PlyData
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False

BASE       = Path(__file__).parent
MODEL_DIR  = BASE / "sparse" / "1"
DENSE_PLY  = BASE / "dense" / "fused.ply"

print("=" * 60)
print("  COLMAP → Rerun Visualization")
print("=" * 60)

rec = pycolmap.Reconstruction(MODEL_DIR)
print(f"Sparse model: {rec.num_cameras()} cameras | {rec.num_reg_images()} images | {rec.num_points3D()} 3D points")

# ── Init Rerun ─────────────────────────────────────────────────────────────────
rr.init(
    "colmap_reconstruction",
    spawn=True,
    default_blueprint=rrb.Blueprint(
        rrb.Spatial3DView(name="Scene", origin="/world"),
    ),
)

# ── Point cloud (dense if available, else sparse) ──────────────────────────────
if DENSE_PLY.exists() and HAS_PLYFILE:
    print(f"  Loading dense point cloud from {DENSE_PLY} ...")
    ply = PlyData.read(str(DENSE_PLY))
    v = ply["vertex"]
    pts    = np.stack([v["x"], v["y"], v["z"]], axis=1)
    colors = np.stack([v["red"], v["green"], v["blue"]], axis=1)

    # Filter outliers: keep within 2.5 std on each axis
    mask = np.ones(len(pts), dtype=bool)
    for i in range(3):
        m, s = pts[:, i].mean(), pts[:, i].std()
        mask &= np.abs(pts[:, i] - m) < 2.5 * s
    pts, colors = pts[mask], colors[mask]
    print(f"  After outlier filter: {len(pts):,} points")

    # Subsample to 10M for Rerun performance
    MAX_PTS = 10_000_000
    if len(pts) > MAX_PTS:
        idx = np.random.choice(len(pts), MAX_PTS, replace=False)
        pts, colors = pts[idx], colors[idx]
        print(f"  Subsampled to {len(pts):,} points")

    rr.log("world/points", rr.Points3D(positions=pts, colors=colors, radii=0.005), static=True)
    print(f"  [✓] Dense point cloud logged")
elif DENSE_PLY.exists():
    print("  plyfile not installed, falling back to sparse cloud.")
    _use_sparse = True
else:
    _use_sparse = True

if not DENSE_PLY.exists() or not HAS_PLYFILE:
    points, colors = [], []
    for p3d in rec.points3D.values():
        points.append(p3d.xyz)
        colors.append(p3d.color)
    rr.log("world/points", rr.Points3D(
        positions=np.array(points),
        colors=np.array(colors),
        radii=0.01,
    ), static=True)
    print(f"  [✓] Sparse point cloud: {len(points):,} points")

# ── Sort images by name (timestamp order) ─────────────────────────────────────
sorted_images = sorted(rec.images.values(), key=lambda img: img.name)

# ── Animate cameras one-by-one ─────────────────────────────────────────────────
print(f"\n  Logging {len(sorted_images)} camera poses (animated) ...")
trajectory = []

for frame_idx, img in enumerate(sorted_images):
    rr.set_time("frame", sequence=frame_idx)

    cfw    = img.cam_from_world()
    w2c    = cfw.inverse()
    tc2w   = w2c.translation
    Rc2w   = w2c.rotation.matrix()
    trajectory.append(tc2w)

    cam = rec.cameras[img.camera_id]

    # Camera frustum
    rr.log("world/camera", rr.Transform3D(
        translation=tc2w,
        mat3x3=Rc2w,
    ))
    rr.log("world/camera/image", rr.Pinhole(
        focal_length=(cam.params[0], cam.params[1]),
        principal_point=(cam.params[2], cam.params[3]),
        width=cam.width,
        height=cam.height,
    ))

    # Growing trajectory line
    if len(trajectory) > 1:
        rr.log("world/trajectory", rr.LineStrips3D(
            [np.array(trajectory)],
            radii=0.005,
            colors=[[0, 200, 255]],
        ))

print(f"  [✓] Animation logged: {len(sorted_images)} frames")
print(f"\nRerun viewer launched — use the timeline to scrub through the trajectory.")
