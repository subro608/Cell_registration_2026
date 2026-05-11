"""
Overlay row2_1 (nd2 warped to JY306 space) vs JY306 in-vivo.
Uses existing warped_z*.png slices (already in JY306 658x629 space).
"""
import cv2
import numpy as np
import tifffile
import os

BASE = '/Users/neurolab/neuroinformatics/margaret'
WARP = f'{BASE}/png_exports/nd2_transformed_to_exvivo'
OUT  = f'{BASE}/png_exports/coarse_registration'
os.makedirs(OUT, exist_ok=True)

# Load JY306
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif')  # (16,658,629)
H, W = jy306.shape[1], jy306.shape[2]

# Load all warped slices
warped = []
for i in range(12):
    img = cv2.imread(f'{WARP}/warped_z{i:03d}.png', cv2.IMREAD_GRAYSCALE)
    warped.append(img)
warped = np.stack(warped, axis=0)  # (12, 658, 629)

def norm8(v, p_lo=1, p_hi=99.5):
    v = v.astype(np.float32)
    vals = v[v > 0]
    if not len(vals): return np.zeros_like(v, dtype=np.uint8)
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    return np.clip((v - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

# MIPs
mip_nd2 = norm8(warped.max(axis=0))   # row2_1 warped MIP
mip_jy  = norm8(jy306.max(axis=0))    # JY306 MIP

# Save side by side
side = np.hstack([mip_jy, mip_nd2])
cv2.imwrite(f'{OUT}/row21_jy306_side_by_side.png', side)

# Overlay: green=row2_1 nd2, magenta=JY306
ov = np.zeros((H, W, 3), dtype=np.uint8)
ov[:, :, 1] = mip_nd2   # green  = row2_1 nd2
ov[:, :, 0] = mip_jy    # R      } magenta = JY306
ov[:, :, 2] = mip_jy    # B      }
cv2.imwrite(f'{OUT}/row21_jy306_mip_overlay.png', ov)

# Per-slice overlays at key z-levels (use the slice mapping from overlay filenames)
# nd2 z000→ev z00, z001→z01, z002→z03, z003→z04, z004→z05, z005→z07...
nd2_to_jy_z = {0:0, 1:1, 2:3, 3:4, 4:5, 5:7, 6:8, 7:10, 8:11, 9:12, 10:14, 11:15}

slices_out = f'{OUT}/row21_jy306_slices'
os.makedirs(slices_out, exist_ok=True)

for nd2_z, jy_z in nd2_to_jy_z.items():
    nd2_sl = norm8(warped[nd2_z])
    jy_sl  = norm8(jy306[jy_z])
    ov_sl  = np.zeros((H, W, 3), dtype=np.uint8)
    ov_sl[:, :, 1] = nd2_sl   # green  = row2_1
    ov_sl[:, :, 0] = jy_sl    # magenta = JY306
    ov_sl[:, :, 2] = jy_sl
    cv2.imwrite(f'{slices_out}/overlay_nd2z{nd2_z:02d}_jyz{jy_z:02d}.png', ov_sl)

print(f"Saved:")
print(f"  {OUT}/row21_jy306_side_by_side.png")
print(f"  {OUT}/row21_jy306_mip_overlay.png")
print(f"  {slices_out}/  ({len(nd2_to_jy_z)} z-slice overlays)")
print(f"\ngreen=row2_1 nd2 warped, magenta=JY306 in-vivo")
print(f"nd2 warped shape: {warped.shape}, JY306 shape: {jy306.shape}")
