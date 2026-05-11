#!/usr/bin/env python3
"""
Row2_2 tile: 2D affine (XY only) + NCC z-matching.
1. Fit 2D affine from 40 landmarks (in-vivo XY → nd2 XY)
2. For each of 16 in-vivo z-slices: warp into nd2 XY, NCC against all 12 nd2 z
3. Contact sheet: 16 rows × 3 cols (nd2 best-z, iv warped, overlay)
"""
import numpy as np
import cv2
import os
from scipy.ndimage import median_filter

BASE = '/Users/neurolab/neuroinformatics/margaret'

IV_XY_UM = 0.6835
IV_Z_UM = 3.0
ND2_XY_UM = 0.645
ND2_Z_UM = 2.0

TILE = 'row2_2'
OUT_DIR = f'{BASE}/png_exports/registration_row2_2'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Load nd2 tile (12 z-slices, 4200x4200)
# ============================================================
print("Loading nd2 tile GFP slices...")
img_dir = f'{BASE}/png_exports/registration_video/{TILE}'
nd2_slices = []
for zi in range(12):
    img = cv2.imread(f'{img_dir}/GFP_z{zi:03d}.png', cv2.IMREAD_UNCHANGED)
    if img is None:
        nd2_slices.append(np.zeros((4200, 4200), dtype=np.float32))
    else:
        nd2_slices.append(img.astype(np.float32))
nd2_slices = np.array(nd2_slices)
print(f"  nd2: {nd2_slices.shape}")

# ============================================================
# Load in-vivo (16 z-slices)
# ============================================================
print("Loading JY306 in-vivo...")
import tifffile
iv_vol_raw = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv, ny_iv, nx_iv = iv_vol_raw.shape
print(f"  In-vivo: {iv_vol_raw.shape}")

print("Median filter bg subtraction...")
iv_vol = np.zeros_like(iv_vol_raw)
for z in range(nz_iv):
    bg = median_filter(iv_vol_raw[z], size=15)
    iv_vol[z] = np.clip(iv_vol_raw[z] - bg, 0, None)
del iv_vol_raw

# ============================================================
# Load landmarks
# ============================================================
print("Loading landmarks...")
d = np.load(f'{BASE}/registration_video/landmarks_nd2_native_{TILE}.npz')
ev_nd2 = d['ev_nd2']  # (N, 3) — col 0=x, col 1=y, col 2=~0 or z-hint
pcd_iv = d['pcd_invivo_jy306']  # (N, 3) — z, y, x in JY306 px
N_LM = ev_nd2.shape[0]
print(f"  {N_LM} landmarks")

# ============================================================
# 2D affine: in-vivo px (x,y) → nd2 px (x,y)
# ============================================================
# Convert both to µm first, then fit
iv_xy_um = np.column_stack([pcd_iv[:, 2] * IV_XY_UM, pcd_iv[:, 1] * IV_XY_UM])  # x, y in µm
nd2_xy_um = np.column_stack([ev_nd2[:, 0] * ND2_XY_UM, ev_nd2[:, 1] * ND2_XY_UM])  # x, y in µm

# Fit 2D affine: iv_um → nd2_um
ones = np.ones((N_LM, 1))
src_h = np.hstack([iv_xy_um, ones])  # (N, 3)
A_T, _, _, _ = np.linalg.lstsq(src_h, nd2_xy_um, rcond=None)  # (3, 2)
A_2d = A_T.T  # (2, 3)

predicted = src_h @ A_T
errors = np.sqrt(np.sum((predicted - nd2_xy_um) ** 2, axis=1))
print(f"  2D affine XY: mean={errors.mean():.1f}µm median={np.median(errors):.1f}µm max={errors.max():.1f}µm")

# Now convert to pixel-space transform: iv_px → nd2_px
# nd2_um = A_2d @ [iv_um; 1]
# nd2_px = nd2_um / ND2_XY_UM
# iv_um = iv_px * IV_XY_UM
# So: nd2_px = (1/ND2_XY_UM) * A_2d @ [[IV_XY_UM, 0, 0], [0, IV_XY_UM, 0], [0, 0, 1]] @ [iv_px; 1]
scale_src = np.array([[IV_XY_UM, 0, 0],
                       [0, IV_XY_UM, 0],
                       [0, 0, 1.0]])
M_px = (1.0 / ND2_XY_UM) * (A_2d @ scale_src)  # (2, 3)
print(f"  Pixel transform:\n{M_px}")

# ============================================================
# NCC z-matching: for each iv z, warp into nd2 XY, correlate with all 12 nd2 z
# ============================================================
print("\nNCC z-matching...")

def ncc(a, b):
    """Normalized cross-correlation between two images (masked to nonzero)."""
    mask = (a > 0) & (b > 0)
    if mask.sum() < 100:
        return -1.0
    a_m = a[mask].astype(np.float64)
    b_m = b[mask].astype(np.float64)
    a_m = a_m - a_m.mean()
    b_m = b_m - b_m.mean()
    denom = np.sqrt((a_m ** 2).sum() * (b_m ** 2).sum())
    if denom < 1e-10:
        return -1.0
    return float((a_m * b_m).sum() / denom)

nd2_h, nd2_w = nd2_slices.shape[1], nd2_slices.shape[2]
z_matches = []  # (iv_z, best_nd2_z, best_ncc, all_nccs)

for z_iv in range(nz_iv):
    # Warp in-vivo slice into nd2 pixel space
    iv_warped = cv2.warpAffine(iv_vol[z_iv], M_px, (nd2_w, nd2_h),
                                flags=cv2.INTER_LINEAR, borderValue=0)

    nccs = []
    for z_nd2 in range(12):
        score = ncc(iv_warped, nd2_slices[z_nd2])
        nccs.append(score)

    best_z = np.argmax(nccs)
    best_score = nccs[best_z]
    z_matches.append((z_iv, best_z, best_score, nccs))
    print(f"  iv_z={z_iv:2d} → nd2_z={best_z:2d}  NCC={best_score:.4f}  all={[f'{x:.3f}' for x in nccs]}")

# ============================================================
# Contact sheet: 16 rows × 3 cols
# ============================================================
print("\nBuilding contact sheet...")

def norm8(img):
    vals = img[img > 0]
    if len(vals) < 100:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(vals, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)

# Downsample for contact sheet
DS = max(1, nd2_w // 600)
out_w, out_h = nd2_w // DS, nd2_h // DS

LABEL_H, GAP, COLS, HEADER_H = 30, 4, 3, 60
row_ht = out_h + LABEL_H + GAP
sheet_w = COLS * out_w + (COLS + 1) * GAP
sheet_h = HEADER_H + nz_iv * row_ht + GAP
sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

# Predicted landmarks in nd2 pixel space
iv_lm_px = np.column_stack([pcd_iv[:, 2], pcd_iv[:, 1]])  # (N, 2) x, y in JY306 px
nd2_lm_pred = (M_px[:, :2] @ iv_lm_px.T).T + M_px[:, 2]  # (N, 2) x, y in nd2 px
nd2_lm_actual = ev_nd2[:, :2]  # (N, 2) x, y in nd2 px
iv_lm_z = pcd_iv[:, 0]  # z in JY306 slices

for si, (z_iv, best_nd2_z, best_ncc, nccs) in enumerate(z_matches):
    # Warp in-vivo
    iv_warped = cv2.warpAffine(iv_vol[z_iv], M_px, (nd2_w, nd2_h),
                                flags=cv2.INTER_LINEAR, borderValue=0)

    nd2_sl = nd2_slices[best_nd2_z]

    # Downsample
    ev_d = cv2.resize(nd2_sl, (out_w, out_h), interpolation=cv2.INTER_AREA)
    iv_d = cv2.resize(iv_warped, (out_w, out_h), interpolation=cv2.INTER_AREA)

    ev_rgb = cv2.cvtColor(norm8(ev_d), cv2.COLOR_GRAY2BGR)
    iv_rgb = cv2.cvtColor(norm8(iv_d), cv2.COLOR_GRAY2BGR)
    ov_rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    ov_rgb[:, :, 1] = norm8(ev_d)   # green = ex-vivo
    ov_rgb[:, :, 0] = norm8(iv_d)   # blue+red = magenta = in-vivo
    ov_rgb[:, :, 2] = norm8(iv_d)

    # Draw landmarks near this z
    n_lm = 0
    for i in range(N_LM):
        if abs(iv_lm_z[i] - z_iv) > 1.5:
            continue
        n_lm += 1
        # Actual nd2 position
        ex_x = int(round(nd2_lm_actual[i, 0] / DS))
        ex_y = int(round(nd2_lm_actual[i, 1] / DS))
        # Predicted nd2 position from in-vivo
        pr_x = int(round(nd2_lm_pred[i, 0] / DS))
        pr_y = int(round(nd2_lm_pred[i, 1] / DS))

        cv2.drawMarker(ev_rgb, (ex_x, ex_y), (255, 255, 0), cv2.MARKER_CROSS, 16, 2)
        cv2.drawMarker(iv_rgb, (pr_x, pr_y), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.drawMarker(ov_rgb, (ex_x, ex_y), (255, 255, 0), cv2.MARKER_CROSS, 16, 1)
        cv2.drawMarker(ov_rgb, (pr_x, pr_y), (0, 255, 255), cv2.MARKER_CROSS, 16, 1)
        cv2.line(ov_rgb, (ex_x, ex_y), (pr_x, pr_y), (255, 255, 255), 1)

    y0 = HEADER_H + si * row_ht + GAP
    for ci, p in enumerate([ev_rgb, iv_rgb, ov_rgb]):
        x0 = GAP + ci * (out_w + GAP)
        sheet[y0:y0 + out_h, x0:x0 + out_w] = p

    cv2.putText(sheet, f'iv_z={z_iv} -> nd2_z={best_nd2_z}  NCC={best_ncc:.3f}  ({n_lm} lm)',
                (GAP + 4, y0 + out_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# Headers
for ci, hdr in enumerate(['nd2 best-z (cyan=actual lm)', 'In-vivo warped (yellow=predicted)', 'Overlay']):
    cv2.putText(sheet, hdr, (GAP + ci * (out_w + GAP) + 4, HEADER_H - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

cv2.putText(sheet,
            f'{TILE}: 2D affine XY ({N_LM} lm, mean err={errors.mean():.1f}um) + NCC z-match',
            (GAP + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

# Z-match summary
z_map_str = '  '.join([f'{z_iv}->{bz}' for z_iv, bz, _, _ in z_matches])
cv2.putText(sheet, f'z-map: {z_map_str}',
            (GAP + 4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

out_path = f'{OUT_DIR}/registration_{TILE}_2d_affine.png'
cv2.imwrite(out_path, sheet)
print(f"\nSaved: {out_path}")
print(f"Sheet size: {sheet.shape[1]}x{sheet.shape[0]}")
