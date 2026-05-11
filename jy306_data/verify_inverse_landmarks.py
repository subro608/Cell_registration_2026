"""
Verify inverse landmark transform:
- Apply inverse to both pcd_invivo and pcd_exvivo
- Print coordinate differences at each step
- Plot BOTH on same ex-vivo MIP (red=pcd_invivo_inv, blue=pcd_exvivo_inv)
"""
import pickle
import numpy as np
import cv2
from scipy.ndimage import map_coordinates

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT  = f'{BASE}/png_exports/pkl_transform_test/inverse_transform'

with open(f'{BASE}/transformation/3_2_merscope15transformed_alt_20250425102844_type2.pkl','rb') as f:
    d = pickle.load(f)

transforms = d['transformations']
pcd_i = d['pcd_invivo'].astype(np.float64)   # (41,3) in JY306 space
pcd_e = d['pcd_exvivo'].astype(np.float64)   # (41,3) in JY306 space

print("=== Input: difference between pcd_invivo and pcd_exvivo (both in JY306 space) ===")
diff = pcd_i - pcd_e
print(f"  Mean |pcd_i - pcd_e|: {np.linalg.norm(diff, axis=1).mean():.3f} px")
print(f"  Max  |pcd_i - pcd_e|: {np.linalg.norm(diff, axis=1).max():.3f} px")
print(f"  pcd_i[0]: {pcd_i[0]}")
print(f"  pcd_e[0]: {pcd_e[0]}")
print()

def interp_vf(vf, pts):
    out = np.zeros_like(pts)
    for c in range(3):
        out[:, c] = map_coordinates(vf[..., c], pts.T, order=1, mode='nearest')
    return out

def point_inverse(pts):
    p = pts.copy()
    for t in reversed(transforms):
        key = list(t.keys())[0]; val = t[key]
        if key == 'scale':
            p = p / val
        elif key == 'bhat':
            R, tv = val[:3].astype(np.float64), val[3].astype(np.float64)
            p = (p - tv) @ np.linalg.inv(R)
        elif key == 'vec_field_total':
            p = p - interp_vf(val, p)
    return p

print("=== Applying inverse transform ===")
p_i_inv = point_inverse(pcd_i)
print("  pcd_invivo inverse done")
p_e_inv = point_inverse(pcd_e)
print("  pcd_exvivo inverse done")

print()
print("=== Output: difference between inverses (in pkl input space) ===")
diff_inv = p_i_inv - p_e_inv
norms_inv = np.linalg.norm(diff_inv, axis=1)
print(f"  Mean |inv(pcd_i) - inv(pcd_e)|: {norms_inv.mean():.3f} px")
print(f"  Max  |inv(pcd_i) - inv(pcd_e)|: {norms_inv.max():.3f} px")
print(f"  p_i_inv[0]: {p_i_inv[0]}")
print(f"  p_e_inv[0]: {p_e_inv[0]}")
print()

# Scale to native 4200px MIP
SCALE = 4200 / 1734
print(f"  After scaling to 4200px MIP (factor {SCALE:.4f}):")
print(f"  p_i_inv[0] → ({p_i_inv[0,1]*SCALE:.1f}, {p_i_inv[0,2]*SCALE:.1f})")
print(f"  p_e_inv[0] → ({p_e_inv[0,1]*SCALE:.1f}, {p_e_inv[0,2]*SCALE:.1f})")
print(f"  Pixel distance at 4200px: {norms_inv[0]*SCALE:.2f} px")
print()

# Load ex-vivo native MIP
mip = cv2.imread(f'{BASE}/png_exports/registration_video/row3_2/GFP_MIP.png', cv2.IMREAD_GRAYSCALE)
H_nat, W_nat = mip.shape

# Normalize
mip_norm = np.clip(mip.astype(np.float32), 0, None)
lo, hi = np.percentile(mip_norm[mip_norm > 0], [1, 99.5])
mip8 = np.clip((mip_norm - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

# RGB canvas
img = cv2.cvtColor(mip8, cv2.COLOR_GRAY2BGR)

print(f"Native MIP size: {H_nat}x{W_nat}")
print(f"Plotting {len(p_i_inv)} pcd_invivo_inv (RED) and {len(p_e_inv)} pcd_exvivo_inv (BLUE) landmarks")

# Plot pcd_invivo inverse in RED
for k in range(len(p_i_inv)):
    y_px = int(round(p_i_inv[k, 1] * W_nat / 1734))
    x_px = int(round(p_i_inv[k, 2] * H_nat / 1734))
    if 0 <= y_px < H_nat and 0 <= x_px < W_nat:
        cv2.circle(img, (x_px, y_px), 12, (0, 0, 255), 2)   # RED
        cv2.circle(img, (x_px, y_px), 2,  (0, 0, 255), -1)

# Plot pcd_exvivo inverse in BLUE (slightly offset so visible)
for k in range(len(p_e_inv)):
    y_px = int(round(p_e_inv[k, 1] * W_nat / 1734))
    x_px = int(round(p_e_inv[k, 2] * H_nat / 1734))
    if 0 <= y_px < H_nat and 0 <= x_px < W_nat:
        cv2.circle(img, (x_px, y_px), 7, (255, 100, 0), 2)  # BLUE-ORANGE
        cv2.circle(img, (x_px, y_px), 2, (255, 100, 0), -1)

cv2.imwrite(f'{OUT}/both_landmarks_inv_comparison.png', img)
print(f"\nSaved: {OUT}/both_landmarks_inv_comparison.png")
print("  RED circles = pcd_invivo inverse (~12px radius)")
print("  BLUE-ORANGE circles = pcd_exvivo inverse (~7px radius)")
print("  If code is correct, red and blue-orange should be nearly co-located")
print("  (they are ~4px apart in input space → ~4px apart in output space)")

# Also save a zoomed crop around the first few landmarks to show separation
# Pick centroid of all landmarks
cy = int(np.mean(p_i_inv[:, 1]) * W_nat / 1734)
cx = int(np.mean(p_i_inv[:, 2]) * H_nat / 1734)
PAD = 300
y0, y1 = max(0, cy-PAD), min(H_nat, cy+PAD)
x0, x1 = max(0, cx-PAD), min(W_nat, cx+PAD)
crop = img[y0:y1, x0:x1]
cv2.imwrite(f'{OUT}/both_landmarks_inv_zoom.png', crop)
print(f"Saved: {OUT}/both_landmarks_inv_zoom.png  (zoomed around landmark centroid)")
