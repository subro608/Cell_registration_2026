"""
JY306 → nd2 row2_1 space using the 4-point affine.
Verified: M_yx applied directly (no swap) to nd2→JY306 matches reference with diff=3.7.
Inverse: cv2.invertAffineTransform(M_yx) applied to JY306 → nd2 space.
"""
import numpy as np
import cv2
import tifffile
import os

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT  = f'{BASE}/png_exports/coarse_registration'
os.makedirs(OUT, exist_ok=True)

# Stored affine: maps nd2 (4200px) → JY306 (658x629)
M_yx = np.load(f'{BASE}/registration_video/affine_nd2_to_exvivo.npy')  # (2,3)
M_inv = cv2.invertAffineTransform(M_yx)  # maps JY306 → nd2

lm = np.load(f'{BASE}/registration_video/landmarks.npz')
src_pts = lm['src_points'][:, :2]  # nd2 (y,x)
tgt_pts = lm['tgt_points'][:, :2]  # JY306 (y,x)

print("Landmark verification (nd2 → JY306 via M_yx):")
for i in range(len(src_pts)):
    r, c = src_pts[i]  # nd2 row, col
    pred = M_yx @ np.array([r, c, 1])
    gt   = tgt_pts[i]
    print(f"  #{i+1}  nd2=({r:.0f},{c:.0f}) → pred=({pred[0]:.1f},{pred[1]:.1f})  gt=({gt[0]:.1f},{gt[1]:.1f})  err={np.linalg.norm(pred-gt):.2f}px")

# Load data
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif')  # (16,658,629)
nd2_dir = f'{BASE}/png_exports/registration_video/row2_1'

def norm8(v, p_lo=1, p_hi=99.5):
    v = v.astype(np.float32)
    vals = v[v > 0]
    if not len(vals): return np.zeros_like(v, dtype=np.uint8)
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    return np.clip((v - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

H_nd2, W_nd2 = 4200, 4200

# Warp JY306 → nd2 space using M_inv
jy_warped = np.stack([
    cv2.warpAffine(jy306[i].astype(np.float32), M_inv, (W_nd2, H_nd2), flags=cv2.INTER_LINEAR)
    for i in range(jy306.shape[0])
])  # (16, 4200, 4200)

# Load nd2 slices
nd2_slices = np.stack([
    cv2.imread(f'{nd2_dir}/GFP_z{i:03d}.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    for i in range(12)
])  # (12, 4200, 4200)

# MIPs
mip_jy  = norm8(jy_warped.max(axis=0))
mip_nd2 = norm8(nd2_slices.max(axis=0))

DISP = 1000
# Overlay: green=nd2, magenta=JY306 warped
ov = np.zeros((H_nd2, W_nd2, 3), dtype=np.uint8)
ov[:,:,1] = mip_nd2
ov[:,:,0] = mip_jy
ov[:,:,2] = mip_jy
ov_d = cv2.resize(ov, (DISP, DISP))

# Add landmark dots (nd2 positions in cyan)
scale = DISP / H_nd2
for i in range(len(src_pts)):
    cy = int(round(src_pts[i,0] * scale))
    cx = int(round(src_pts[i,1] * scale))
    cv2.circle(ov_d, (cx, cy), 10, (0,255,255), 2)
    cv2.putText(ov_d, str(i+1), (cx+12,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

# Side by side
side = np.hstack([
    cv2.resize(mip_jy,  (DISP, DISP)),
    cv2.resize(mip_nd2, (DISP, DISP))
])

cv2.imwrite(f'{OUT}/row21_4pt_mip_overlay.png', ov_d)
cv2.imwrite(f'{OUT}/row21_4pt_side_by_side.png', side)

# Per-slice overlays
slices_out = f'{OUT}/row21_4pt_slices'
os.makedirs(slices_out, exist_ok=True)
for nd2_z, jy_z in {0:0,1:1,2:3,3:4,4:5,5:7,6:8,7:10,8:11,9:12,10:14,11:15}.items():
    ov_sl = np.zeros((H_nd2, W_nd2, 3), dtype=np.uint8)
    ov_sl[:,:,1] = norm8(nd2_slices[nd2_z])
    ov_sl[:,:,0] = norm8(jy_warped[jy_z])
    ov_sl[:,:,2] = norm8(jy_warped[jy_z])
    cv2.imwrite(f'{slices_out}/z{nd2_z:02d}_nd2_vs_jy{jy_z:02d}.png', cv2.resize(ov_sl,(DISP,DISP)))

print(f"\nSaved: {OUT}/row21_4pt_mip_overlay.png")
print("green=nd2 native, magenta=JY306 warped to nd2 space, cyan dots=4 landmarks")
