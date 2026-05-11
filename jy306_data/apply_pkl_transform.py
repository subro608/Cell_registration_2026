"""
Apply pkl transformations to raw row3_2 GFP images and compare with stored result.

Pipeline (14 stages):
  scale → bhat (affine) → scale → bhat → scale → bhat
  → vec_field → bhat → vec_field → vec_field → vec_field → vec_field → bhat → vec_field

Assumptions:
  - Raw 4200px PNGs downsampled by 7 → ~600px input space (matches pcd_exvivo range)
  - 12 PNGs (z000-z011) padded with 5 zeros at end → 17 z-slices
  - bhat forward: p_out = p @ bhat[:3] + bhat[3]  (backward: p_in = (p_out - t) @ inv(R))
  - vec_field: displacement field, backward mapping subtracts displacement
"""
import pickle
import numpy as np
import cv2
import os
from scipy.ndimage import map_coordinates

BASE = '/Users/neurolab/neuroinformatics/margaret'
PNG_DIR = f'{BASE}/png_exports/registration_video/row3_2'
OUT_DIR = f'{BASE}/png_exports/pkl_transform_test/applied_transform'
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load pkl ----
pkl_path = f'{BASE}/transformation/3_2_merscope15transformed_alt_20250425102844_type2.pkl'
with open(pkl_path, 'rb') as f:
    d = pickle.load(f)
transforms = d['transformations']
ref_transformed = d['transformed']   # (3, 17, 1734, 1734) — the stored result

# ---- Load raw GFP PNGs and build input volume ----
print("Loading raw GFP z-slices...")
INPUT_SCALE = 7          # 4200 / 7 = 600
TARGET_XY   = 600        # input space XY size
TARGET_Z    = 17         # pkl z-slices
OUT_XY      = 1734       # pkl output canvas size

slices = []
for z in range(12):
    img = cv2.imread(f'{PNG_DIR}/GFP_z{z:03d}.png', cv2.IMREAD_GRAYSCALE)
    img_ds = cv2.resize(img.astype(np.float32), (TARGET_XY, TARGET_XY), interpolation=cv2.INTER_AREA)
    slices.append(img_ds)

# Pad to 17 z-slices (5 zeros at end)
for _ in range(TARGET_Z - len(slices)):
    slices.append(np.zeros((TARGET_XY, TARGET_XY), dtype=np.float32))

vol = np.stack(slices, axis=0)   # (17, 600, 600)
print(f"Input volume: {vol.shape}  range: [{vol.min():.1f}, {vol.max():.1f}]")

# ---- Transform application functions ----

def apply_scale(vol, scale):
    """Backward: sample input at coords / scale"""
    Z, Y, X = vol.shape
    z, y, x = np.mgrid[:Z, :Y, :X].astype(np.float32)
    coords = np.array([z / scale, y / scale, x / scale])
    return map_coordinates(vol, coords, order=1, mode='constant', cval=0.0)

def apply_affine(vol, bhat):
    """
    bhat (4x3): rows 0-2 = rotation R, row 3 = translation t
    Forward:  p_out = p @ R + t
    Backward: p_in  = (p_out - t) @ inv(R)
    """
    R = bhat[:3].astype(np.float64)   # (3x3)
    t = bhat[3].astype(np.float64)    # (3,)
    R_inv = np.linalg.inv(R)
    Z, Y, X = vol.shape
    z, y, x = np.mgrid[:Z, :Y, :X]
    pts = np.stack([z.ravel(), y.ravel(), x.ravel()], axis=1).astype(np.float64)   # (N,3)
    pts_in = (pts - t) @ R_inv   # (N,3)
    coords = pts_in.T.reshape(3, Z, Y, X)
    return map_coordinates(vol, coords, order=1, mode='constant', cval=0.0)

def apply_vecfield(vol, vf):
    """
    vf (Z, Y, X, 3): displacement (dz, dy, dx) per voxel
    Backward: sample input at (z - dz, y - dy, x - dx)
    """
    Z, Y, X = vol.shape
    z, y, x = np.mgrid[:Z, :Y, :X].astype(np.float32)
    # Resize vf to match current volume size if needed
    if vf.shape[:3] != (Z, Y, X):
        vf_r = np.stack([
            cv2.resize(vf[..., c].astype(np.float32).reshape(-1, vf.shape[2]),
                       (X, Z), interpolation=cv2.INTER_LINEAR).reshape(Z, Y, X)
            for c in range(3)
        ], axis=-1)
    else:
        vf_r = vf.astype(np.float32)
    coords = np.array([z - vf_r[..., 0], y - vf_r[..., 1], x - vf_r[..., 2]])
    return map_coordinates(vol, coords, order=1, mode='constant', cval=0.0)

# ---- Resize input to output canvas (1734x1734) before applying transforms ----
# The pkl canvas is 1734×1734 — upsample the 600px volume first
print("Upsampling to pkl canvas size (1734x1734)...")
vol_up = np.stack([
    cv2.resize(vol[z], (OUT_XY, OUT_XY), interpolation=cv2.INTER_LINEAR)
    for z in range(TARGET_Z)
], axis=0)   # (17, 1734, 1734)
print(f"Upsampled volume: {vol_up.shape}")

# ---- Apply all 14 transform stages ----
result = vol_up.copy()
for i, t in enumerate(transforms):
    key = list(t.keys())[0]
    val = t[key]
    print(f"  Stage [{i:2d}] {key}...", end=' ', flush=True)
    if key == 'scale':
        result = apply_scale(result, val)
    elif key == 'bhat':
        result = apply_affine(result, val)
    elif key == 'vec_field_total':
        result = apply_vecfield(result, val)
    print(f"done  range=[{result.min():.1f}, {result.max():.1f}]")

print(f"\nApplied transform result: {result.shape}")

# ---- Save result ----
def norm8(img):
    vals = img[img > 0]
    if len(vals) == 0: return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(vals, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

print("\nSaving applied transform z-slices...")
for z in range(TARGET_Z):
    cv2.imwrite(f'{OUT_DIR}/applied_z{z:02d}.png', norm8(result[z]))

# Save MIP
mip_applied = result.max(axis=0)
cv2.imwrite(f'{OUT_DIR}/applied_MIP.png', norm8(mip_applied))

# ---- Compare with stored pkl result ----
ref_ch0 = ref_transformed[0]   # (17, 1734, 1734)
mip_ref  = ref_ch0.max(axis=0)
cv2.imwrite(f'{OUT_DIR}/pkl_ref_MIP.png', norm8(mip_ref.astype(np.float32)))

# Side-by-side comparison
mip_a8 = norm8(mip_applied)
mip_r8 = norm8(mip_ref.astype(np.float32))
h, w   = mip_a8.shape
comp   = np.zeros((h, w*2 + 20, 3), dtype=np.uint8)
comp[:, :w, 1]      = mip_a8    # green = applied
comp[:, w+20:, 0]   = mip_r8   # magenta = pkl ref
comp[:, w+20:, 2]   = mip_r8
cv2.putText(comp, 'Applied (green)', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
cv2.putText(comp, 'PKL ref (magenta)', (w+30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,255), 2)
cv2.imwrite(f'{OUT_DIR}/comparison.png', comp)

print(f"\nAll saved to: {OUT_DIR}/")
print("Check comparison.png to see if applied transform matches pkl result.")
