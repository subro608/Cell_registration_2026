"""
Coarse rigid registration: JY306 in-vivo → ex-vivo stitched (both 1µm isotropic)
Step 1: NCC template match on XY MIP at 4x downsampled to find (y,x) translation
Step 2: Repeat for XZ MIP to find z offset
Step 3: Show overlay at full resolution
"""
import tifffile
import numpy as np
import cv2
import os

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT  = f'{BASE}/png_exports/coarse_registration'
os.makedirs(OUT, exist_ok=True)

print("Loading volumes...")
inv = tifffile.imread(f'{BASE}/registration_video/stitched/invivo_1um_isotropic.tif')
ex  = tifffile.imread(f'{BASE}/registration_video/stitched/stitched_gfp_elastix_1um_isotropic.tif')
print(f"  invivo:  {inv.shape}  {inv.dtype}")
print(f"  exvivo:  {ex.shape}  {ex.dtype}")

def norm32(v, p_lo=1, p_hi=99.5):
    v = v.astype(np.float32)
    vals = v[v > 0]
    if not len(vals): return v
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    return np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)

def norm8(v, p_lo=1, p_hi=99.5):
    return (norm32(v, p_lo, p_hi) * 255).astype(np.uint8)

# ── XY MIP ──────────────────────────────────────────────────────────────────
print("\n--- XY MIP NCC search ---")
mip_inv_xy = norm32(inv.max(axis=0))  # (1229, 1177)
mip_ex_xy  = norm32(ex.max(axis=0))   # (2748, 2748)

DS = 4  # downsample factor for speed
tmpl = cv2.resize(mip_inv_xy, (mip_inv_xy.shape[1]//DS, mip_inv_xy.shape[0]//DS))
srch = cv2.resize(mip_ex_xy,  (mip_ex_xy.shape[1]//DS,  mip_ex_xy.shape[0]//DS))

print(f"  Template (in-vivo XY, DS={DS}): {tmpl.shape}")
print(f"  Search   (ex-vivo XY, DS={DS}): {srch.shape}")

res_xy = cv2.matchTemplate(srch.astype(np.float32), tmpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
_, max_val_xy, _, max_loc_xy = cv2.minMaxLoc(res_xy)
y0_ds, x0_ds = max_loc_xy[1], max_loc_xy[0]
y0_xy = y0_ds * DS
x0_xy = x0_ds * DS
print(f"  XY best NCC={max_val_xy:.4f}  → top-left in ex-vivo: y={y0_xy}, x={x0_xy}")

# Save NCC heatmap
heatmap = norm8(res_xy)
heatmap_big = cv2.resize(heatmap, (686, 686))  # quarter size for display
cv2.imwrite(f'{OUT}/ncc_heatmap_xy.png', heatmap_big)

# ── XZ MIP ──────────────────────────────────────────────────────────────────
print("\n--- XZ MIP NCC search (for z offset) ---")
mip_inv_xz = norm32(inv.max(axis=1))   # (189, 1177)
mip_ex_xz  = norm32(ex.max(axis=1))    # (516, 2748)

# Crop ex-vivo XZ search to expected x range from XY result
x_lo = max(0, x0_xy - 100)
x_hi = min(ex.shape[2], x0_xy + inv.shape[2] + 100)
mip_ex_xz_crop = mip_ex_xz[:, x_lo:x_hi]

DS2 = 2
tmpl_xz = cv2.resize(mip_inv_xz, (mip_inv_xz.shape[1]//DS2, mip_inv_xz.shape[0]//DS2))
srch_xz = cv2.resize(mip_ex_xz_crop, (mip_ex_xz_crop.shape[1]//DS2, mip_ex_xz_crop.shape[0]//DS2))

print(f"  Template (in-vivo XZ, DS={DS2}): {tmpl_xz.shape}")
print(f"  Search   (ex-vivo XZ cropped, DS={DS2}): {srch_xz.shape}")

if srch_xz.shape[0] >= tmpl_xz.shape[0] and srch_xz.shape[1] >= tmpl_xz.shape[1]:
    res_xz = cv2.matchTemplate(srch_xz.astype(np.float32), tmpl_xz.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, max_val_xz, _, max_loc_xz = cv2.minMaxLoc(res_xz)
    z0 = max_loc_xz[1] * DS2
    x0_xz = x_lo + max_loc_xz[0] * DS2
    print(f"  XZ best NCC={max_val_xz:.4f}  → z_offset={z0}, x_check={x0_xz}")
else:
    z0 = 0
    print("  XZ search region too small, setting z0=0")

print(f"\n=== Coarse translation: y={y0_xy}, x={x0_xy}, z={z0} ===")

# ── Overlay at full resolution XY MIP ────────────────────────────────────────
print("\nBuilding overlay...")
H_inv, W_inv = inv.shape[1], inv.shape[2]
H_ex,  W_ex  = ex.shape[1],  ex.shape[2]

ex_mip8  = norm8(ex.max(axis=0))
inv_mip8 = norm8(inv.max(axis=0))

# Build RGB: green=ex-vivo, magenta=in-vivo placed at (y0,x0)
ov = np.zeros((H_ex, W_ex, 3), dtype=np.uint8)
ov[:, :, 1] = ex_mip8  # green = ex-vivo

# Place in-vivo into ex-vivo canvas
y1 = min(y0_xy + H_inv, H_ex)
x1 = min(x0_xy + W_inv, W_ex)
iy1 = y1 - y0_xy
ix1 = x1 - x0_xy

ov[y0_xy:y1, x0_xy:x1, 0] = inv_mip8[:iy1, :ix1]  # R  } magenta
ov[y0_xy:y1, x0_xy:x1, 2] = inv_mip8[:iy1, :ix1]  # B  }

# Draw bounding box for in-vivo region
cv2.rectangle(ov, (x0_xy, y0_xy), (x0_xy+W_inv-1, y0_xy+H_inv-1), (0, 255, 255), 4)

cv2.imwrite(f'{OUT}/overlay_coarse_ncc.png', ov)
print(f"Saved: {OUT}/overlay_coarse_ncc.png")
print(f"  Green=ex-vivo stitched MIP, Magenta=in-vivo MIP, Cyan box=in-vivo extent")

# Also save a zoomed crop showing overlap region
PAD = 100
yA = max(0, y0_xy - PAD); yB = min(H_ex, y0_xy + H_inv + PAD)
xA = max(0, x0_xy - PAD); xB = min(W_ex, x0_xy + W_inv + PAD)
crop = ov[yA:yB, xA:xB]
crop_disp = cv2.resize(crop, (1000, int(1000 * crop.shape[0] / crop.shape[1])))
cv2.imwrite(f'{OUT}/overlay_coarse_ncc_zoom.png', crop_disp)
print(f"Saved: {OUT}/overlay_coarse_ncc_zoom.png  (zoomed to overlap region)")

print(f"\nTranslation to apply to in-vivo cells: add (z={z0}, y={y0_xy}, x={x0_xy}) to map into ex-vivo space")
