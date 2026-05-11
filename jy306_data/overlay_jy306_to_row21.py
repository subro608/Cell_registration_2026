"""
Warp JY306 in-vivo MIP into row2_1 nd2 space and create overlay + side-by-side.

Uses the inverted affine from affine_nd2_to_exvivo.npy.
Overlays landmarks from landmarks_row21_jy306.npz.

Output:
  png_exports/coarse_registration/jy306mip_warped_to_row21_side.png
  png_exports/coarse_registration/jy306mip_warped_to_row21_overlay.png
"""

import numpy as np
import cv2
import tifffile
import os

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_DIR = os.path.join(BASE, 'png_exports/coarse_registration')

# JY306 intensity boost (increase to make in-vivo brighter)
JY306_BOOST = 8.0

# ============================================================
# Load transforms
# ============================================================
M_nd2_to_jy = np.load(os.path.join(BASE, 'registration_video/affine_nd2_to_exvivo.npy'))
M_3x3 = np.vstack([M_nd2_to_jy, [0, 0, 1]])
M_jy_to_nd2 = np.linalg.inv(M_3x3)[:2, :]

# ============================================================
# Load images
# ============================================================
# JY306 MIP
jy306_vol = tifffile.imread(os.path.join(BASE, 'JY306_in_Vivo_stack_flipped_s80.tif'))
jy_mip = np.max(jy306_vol, axis=0)
jy_mip_u8 = np.clip(jy_mip / jy_mip.max() * 255 * JY306_BOOST, 0, 255).astype(np.uint8)

# Warp JY306 MIP to nd2 space (4200x4200)
jy_warped = cv2.warpAffine(jy_mip_u8, M_jy_to_nd2, (4200, 4200))

# Row2_1 GFP MIP
row21_mip = cv2.imread(os.path.join(BASE, 'png_exports/registration_video/row2_1/GFP_MIP.png'), cv2.IMREAD_GRAYSCALE)

# ============================================================
# Load landmarks
# ============================================================
d = np.load(os.path.join(BASE, 'registration_video/landmarks_row21_jy306.npz'), allow_pickle=True)
src_pts = d['src_points']  # nd2 space
tgt_pts = d['tgt_points']  # JY306 space

PANEL = 800
scale = PANEL / 4200.0

# ============================================================
# Side-by-side
# ============================================================
left = cv2.cvtColor(cv2.resize(row21_mip, (PANEL, PANEL)), cv2.COLOR_GRAY2BGR)
right = cv2.cvtColor(cv2.resize(jy_warped, (PANEL, PANEL)), cv2.COLOR_GRAY2BGR)
img_side = np.hstack([left, right])

for i in range(len(src_pts)):
    sx, sy = src_pts[i, 0], src_pts[i, 1]
    dx_l, dy_l = int(sx * scale), int(sy * scale)
    tx, ty = tgt_pts[i, 0], tgt_pts[i, 1]
    p_nd2 = M_jy_to_nd2 @ np.array([tx, ty, 1.0])
    dx_r, dy_r = int(p_nd2[0] * scale) + PANEL, int(p_nd2[1] * scale)
    cv2.circle(img_side, (dx_l, dy_l), 8, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(img_side, (dx_r, dy_r), 8, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(img_side, (dx_l, dy_l), (dx_r, dy_r), (0, 255, 0), 1, cv2.LINE_AA)

cv2.putText(img_side, "row2_1 GFP MIP", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
cv2.putText(img_side, f"JY306 MIP warped (boost={JY306_BOOST}x)", (PANEL + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

out1 = os.path.join(OUT_DIR, 'jy306mip_warped_to_row21_side.png')
cv2.imwrite(out1, img_side)
print(f"Saved {out1}")

# ============================================================
# Overlay: green=row2_1, magenta=JY306
# ============================================================
row21_color = cv2.merge([np.zeros_like(row21_mip), row21_mip, np.zeros_like(row21_mip)])
jy_color = cv2.merge([jy_warped, np.zeros_like(jy_warped), jy_warped])
overlay = cv2.addWeighted(row21_color, 1.0, jy_color, 1.0, 0)
overlay_resized = cv2.resize(overlay, (PANEL, PANEL))

for i in range(len(src_pts)):
    sx, sy = src_pts[i, 0], src_pts[i, 1]
    dx, dy = int(sx * scale), int(sy * scale)
    cv2.circle(overlay_resized, (dx, dy), 10, (255, 255, 255), 1, cv2.LINE_AA)

cv2.putText(overlay_resized, f"Green=row2_1  Magenta=JY306 (boost={JY306_BOOST}x)", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

out2 = os.path.join(OUT_DIR, 'jy306mip_warped_to_row21_overlay.png')
cv2.imwrite(out2, overlay_resized)
print(f"Saved {out2}")
