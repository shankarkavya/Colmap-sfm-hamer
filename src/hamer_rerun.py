"""
HaMeR 3D hand reconstruction on COLMAP images.
- Runs HaMeR on every Nth frame (stride)
- Keeps ONE hand per frame (right preferred, else left)
- Transforms hand mesh from camera space → world space using COLMAP poses
- Logs to Rerun: hand mesh, camera frustum, trajectory, RGB image
- Saves to hamer_hands.rrd
"""

import sys
import os
sys.path.insert(0, '/home/user/shankark2/Artipoint/hamer')
os.chdir('/home/user/shankark2/Artipoint/hamer')

from pathlib import Path
import torch
import cv2
import numpy as np
import pycolmap
import rerun as rr
import rerun.blueprint as rrb

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import cam_crop_to_full
from vitpose_model import ViTPoseModel

# ── Config ─────────────────────────────────────────────────────────────────────
COLMAP_DIR  = Path('/home/user/shankark2/Learning/SLAM/Colmap')
IMG_DIR     = COLMAP_DIR / 'rgb_undistorted'
MODEL_DIR   = COLMAP_DIR / 'sparse' / '1'
OUT_RRD     = COLMAP_DIR / 'hamer_hands_all.rrd'
STRIDE      = 1         # process every frame
LIGHT_BLUE  = [0.651, 0.741, 0.859]

# ── Load COLMAP reconstruction ─────────────────────────────────────────────────
print("Loading COLMAP reconstruction...")
rec = pycolmap.Reconstruction(MODEL_DIR)
# Build name → image lookup
colmap_images = {img.name: img for img in rec.images.values()}
colmap_cam    = rec.cameras[list(rec.cameras.keys())[0]]
fx, fy, cx, cy = colmap_cam.params
W, H = colmap_cam.width, colmap_cam.height
print(f"  {len(colmap_images)} registered images, camera {W}x{H}, fx={fx:.1f}")

# ── Load HaMeR ─────────────────────────────────────────────────────────────────
print("Loading HaMeR model...")
from hamer.models import download_models
download_models(CACHE_DIR_HAMER)
model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device).eval()
mano_faces = model.mano.faces

# ── Load detectors ─────────────────────────────────────────────────────────────
print("Loading detectors...")
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import hamer as hamer_pkg
cfg_path = Path(hamer_pkg.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
detectron2_cfg = LazyConfig.load(str(cfg_path))
detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
for i in range(3):
    detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
detector = DefaultPredictor_Lazy(detectron2_cfg)
cpm = ViTPoseModel(device)

# ── Load sparse point cloud ────────────────────────────────────────────────────
print("Loading sparse point cloud...")
sparse_pts, sparse_colors = [], []
for p3d in rec.points3D.values():
    sparse_pts.append(p3d.xyz)
    sparse_colors.append(p3d.color)
sparse_pts    = np.array(sparse_pts)
sparse_colors = np.array(sparse_colors)

# ── Init Rerun ─────────────────────────────────────────────────────────────────
print("Initialising Rerun...")
rr.init("hamer_colmap", default_blueprint=rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(name="World", origin="/world"),
        rrb.Spatial2DView(name="RGB + Hand", origin="/camera/image"),
    )
))
rr.save(str(OUT_RRD))

# Declare world coordinate system (COLMAP uses right-hand, Y-down)
rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

# Log sparse point cloud as static
rr.log('world/points', rr.Points3D(positions=sparse_pts, colors=sparse_colors, radii=0.01), static=True)
print(f"  [✓] Sparse cloud: {len(sparse_pts):,} points")

# ── Process images ─────────────────────────────────────────────────────────────
img_paths = sorted(IMG_DIR.glob('*.jpg'))
img_paths = img_paths[::STRIDE]
print(f"Processing {len(img_paths)} images (stride={STRIDE})...")

trajectory = []
hands_found = 0

for frame_idx, img_path in enumerate(img_paths):
    img_name = img_path.name
    if img_name not in colmap_images:
        continue

    colmap_img = colmap_images[img_name]

    # Camera pose: cam_from_world (R, t) s.t. X_cam = R @ X_world + t
    cfw  = colmap_img.cam_from_world()
    R    = cfw.rotation.matrix()        # (3,3)
    t    = cfw.translation              # (3,)
    # camera centre in world frame
    cam_centre = -R.T @ t

    trajectory.append(cam_centre)

    # ── Set Rerun time ──────────────────────────────────────────────────────────
    rr.set_time('frame', sequence=frame_idx)

    # ── Log camera pose ─────────────────────────────────────────────────────────
    Rc2w = R.T
    rr.log('world/camera', rr.Transform3D(translation=cam_centre, mat3x3=Rc2w))
    rr.log('world/camera/image', rr.Pinhole(
        focal_length=(fx, fy),
        principal_point=(cx, cy),
        width=W, height=H,
        camera_xyz=rr.ViewCoordinates.RDF,  # Right, Down, Forward (OpenCV = COLMAP)
    ))

    # ── Log RGB image ───────────────────────────────────────────────────────────
    img_cv2 = cv2.imread(str(img_path))
    img_rgb = img_cv2[:, :, ::-1]
    rr.log('camera/image', rr.Image(img_rgb))

    # ── Log trajectory ──────────────────────────────────────────────────────────
    if len(trajectory) > 1:
        rr.log('world/trajectory', rr.LineStrips3D(
            [np.array(trajectory)], radii=0.005, colors=[[0, 200, 255]]
        ))

    # ── Detect people ───────────────────────────────────────────────────────────
    det_out = detector(img_cv2)
    det_instances = det_out['instances']
    valid_idx  = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()
    if len(pred_bboxes) == 0:
        continue

    vitposes_out = cpm.predict_pose(
        img_rgb,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    bboxes, is_right_list = [], []
    for vitposes in vitposes_out:
        for hand_keyp, hand_side in [
            (vitposes['keypoints'][-21:], 1),   # right hand
            (vitposes['keypoints'][-42:-21], 0), # left hand
        ]:
            valid = hand_keyp[:, 2] > 0.5
            if valid.sum() > 3:
                bbox = [hand_keyp[valid,0].min(), hand_keyp[valid,1].min(),
                        hand_keyp[valid,0].max(), hand_keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right_list.append(hand_side)
                break  # ONE hand only — right preferred
        if bboxes:
            break  # ONE person only

    if not bboxes:
        continue

    boxes = np.stack(bboxes)
    right = np.array(is_right_list)

    # ── Run HaMeR ───────────────────────────────────────────────────────────────
    dataset    = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=2.0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        multiplier         = (2 * batch['right'] - 1)
        pred_cam           = out['pred_cam'].clone()
        pred_cam[:, 1]     = multiplier * pred_cam[:, 1]
        box_center         = batch['box_center'].float()
        box_size           = batch['box_size'].float()
        img_size           = batch['img_size'].float()
        # Use COLMAP's actual focal length so hand depth matches scene scale
        colmap_fl = torch.tensor(fx).float().to(device)
        pred_cam_t_full    = cam_crop_to_full(
            pred_cam, box_center, box_size, img_size, colmap_fl
        ).detach().cpu().numpy()

        # Take first (only) detection
        verts_cam_local = out['pred_vertices'][0].detach().cpu().numpy()   # (778,3) in MANO local cam
        is_r            = batch['right'][0].cpu().numpy()
        verts_cam_local[:, 0] = (2 * is_r - 1) * verts_cam_local[:, 0]
        cam_t           = pred_cam_t_full[0]                                # (3,) translation in cam space

        # Hand verts in camera space
        verts_cam = verts_cam_local + cam_t                                 # (778,3)

        # Transform to world space: X_world = R^T @ (X_cam - t)
        verts_world = (R.T @ (verts_cam - t).T).T                          # (778,3)

        # ── Log hand mesh in world frame (unique path per frame = persists) ───────
        hand_label = 'right' if is_r else 'left'
        rgba = np.array([int(LIGHT_BLUE[0]*255), int(LIGHT_BLUE[1]*255), int(LIGHT_BLUE[2]*255), 255], dtype=np.uint8)
        color = np.tile(rgba, (verts_world.shape[0], 1))
        rr.log(f'world/hands/frame_{frame_idx:04d}', rr.Mesh3D(
            vertex_positions=verts_world,
            triangle_indices=mano_faces,
            vertex_colors=color,
        ))  # appears at this frame, persists forward — future hands not shown yet

        hands_found += 1
        if frame_idx % 20 == 0:
            print(f"  [{frame_idx}/{len(img_paths)}] {hand_label} hand @ frame {frame_idx} — total {hands_found}")
        break  # one hand only

print(f"\nDone. {hands_found} hands reconstructed across {len(img_paths)} frames.")
print(f"Saved → {OUT_RRD}")
print(f"Open with: rerun {OUT_RRD}")
