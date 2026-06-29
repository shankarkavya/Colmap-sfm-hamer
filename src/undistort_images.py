"""
Undistort RGB images using known camera intrinsics (rational_polynomial / FULL_OPENCV).
Outputs undistorted images + new pinhole camera_info to rgb_undistorted/.
"""

import cv2
import numpy as np
from pathlib import Path

BASE      = Path(__file__).parent
IN_DIR    = BASE / "rgb"
OUT_DIR   = BASE / "rgb_undistorted"
OUT_DIR.mkdir(exist_ok=True)

# Intrinsics from camera_info.txt
K = np.array([
    [609.1403198242188, 0.0,              642.583251953125 ],
    [0.0,              609.1126708984375, 367.40484619140625],
    [0.0,              0.0,              1.0              ],
], dtype=np.float64)

# rational_polynomial: k1, k2, p1, p2, k3, k4, k5, k6
D = np.array([
    0.20317251980304718, -2.4349634647369385,
    0.0008450598106719553, -0.00022232322953641415,
    1.4997299909591675, 0.08393746614456177,
    -2.2444663047790527, 1.4160453081130981,
], dtype=np.float64)

W, H = 1280, 720

# Compute optimal new camera matrix (no black borders)
K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, (W, H), alpha=0.0)
print(f"Original K:\n{K}")
print(f"\nNew pinhole K:\n{K_new}")
print(f"ROI: {roi}")

# Save new camera info
fx_new = K_new[0, 0]
fy_new = K_new[1, 1]
cx_new = K_new[0, 2]
cy_new = K_new[1, 2]
with open(OUT_DIR / "camera_info_pinhole.txt", "w") as f:
    f.write(f"width: {W}\n")
    f.write(f"height: {H}\n")
    f.write(f"camera_model: PINHOLE\n")
    f.write(f"fx: {fx_new}\n")
    f.write(f"fy: {fy_new}\n")
    f.write(f"cx: {cx_new}\n")
    f.write(f"cy: {cy_new}\n")

images = sorted(IN_DIR.glob("*.jpg"))
print(f"\nUndistorting {len(images)} images ...")

for i, img_path in enumerate(images):
    img = cv2.imread(str(img_path))
    undist = cv2.undistort(img, K, D, None, K_new)
    out_path = OUT_DIR / img_path.name
    cv2.imwrite(str(out_path), undist)
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(images)}")

print(f"Done. Undistorted images → {OUT_DIR}")
print(f"\nNew camera params for COLMAP PINHOLE: {fx_new},{fy_new},{cx_new},{cy_new}")
