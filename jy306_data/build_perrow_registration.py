#!/usr/bin/env python3
"""
Per-row affine registration: pick one tile row, use only its landmarks,
compute 3D affine, warp the overlapping in-vivo z-slices, show contact sheet.
"""
import numpy as np
import tifffile
import cv2
import os
import json
import glob
from scipy.ndimage import median_filter

BASE = '/Users/neurolab/neuroinformatics/margaret'

IV_XY_UM = 0.6835
IV_Z_UM = 3.0
NATIVE_XY_UM = 0.645
NATIVE_Z_UM = 2.0

# Which row to process
TARGET_ROW = 'row2'

# ============================================================
# Load stitch params
# ============================================================
print("Loading stitch params...")
with open(f'{BASE}/registration_video/stitch_iou_only_params.json') as f:
    params = json.load(f)

TILE_ORDER = params['tile_order']
tile_z_offsets = {k: int(v) for k, v in params['tile_z_offsets'].items()}
cum_iou = {k: np.array(v) for k, v in params['cumulative_iou'].items()}

# Find tiles in target row
row_tiles = [t for t in TILE_ORDER if t.startswith(TARGET_ROW + '_')]
print(f"  {TARGET_ROW} tiles: {row_tiles}")

# Z-range for this row in stitched 1µm iso
z_native_lo = tile_z_offsets[row_tiles[0]]
z_native_hi = tile_z_offsets[row_tiles[-1]] + 12  # last tile's 12 slices
z_iso_lo = int(z_native_lo * NATIVE_Z_UM)
z_iso_hi = int(z_native_hi * NATIVE_Z_UM)
print(f"  Z range in stitched 1µm iso: [{z_iso_lo}, {z_iso_hi}]")

# ============================================================
# Load landmark files for this row only
# ============================================================
print("\nLoading landmarks for this row...")
lm_files = sorted(glob.glob(f'{BASE}/registration_video/landmarks_nd2_native_*.npz'))
legacy = f'{BASE}/registration_video/landmarks_27_nd2_native.npz'
if os.path.exists(legacy):
    lm_files.append(legacy)

all_ev_stitched = []
all_iv_um = []

for lm_file in lm_files:
    bn = os.path.basename(lm_file)
    if 'landmarks_27_nd2_native' in bn:
        tile = 'row2_1'
    else:
        tile = bn.replace('landmarks_nd2_native_', '').replace('.npz', '')

    if tile not in row_tiles:
        continue

    d = np.load(lm_file)
    ev_nd2 = d['ev_nd2']
    pcd_iv = d['pcd_invivo_jy306']
    N = ev_nd2.shape[0]

    # Find best z per cell
    img_dir = f'{BASE}/png_exports/registration_video/{tile}'
    nd2_slices = []
    for zi in range(12):
        img = cv2.imread(f'{img_dir}/GFP_z{zi:03d}.png', cv2.IMREAD_UNCHANGED)
        if img is None:
            nd2_slices.append(np.zeros((4200, 4200), dtype=np.float32))
        else:
            nd2_slices.append(img.astype(np.float32))
    nd2_slices = np.array(nd2_slices)
    H = nd2_slices.shape[1]

    for i in range(N):
        x_nd2, y_nd2 = ev_nd2[i, 0], ev_nd2[i, 1]
        c = int(round(np.clip(x_nd2, 10, H - 11)))
        r = int(round(np.clip(y_nd2, 10, H - 11)))
        intensities = [nd2_slices[z][r-10:r+10, c-10:c+10].mean() for z in range(12)]
        z_nd2 = np.argmax(intensities)

        # Ex-vivo: nd2 → stitched 1µm iso
        M = cum_iou[tile]
        p = M @ np.array([x_nd2, y_nd2, 1.0])
        x_iso = p[0] * NATIVE_XY_UM
        y_iso = p[1] * NATIVE_XY_UM
        z_iso = (tile_z_offsets[tile] + z_nd2) * NATIVE_Z_UM
        all_ev_stitched.append([x_iso, y_iso, z_iso])

        # In-vivo: JY306 px → µm
        iv_z, iv_y, iv_x = pcd_iv[i]
        all_iv_um.append([iv_x * IV_XY_UM, iv_y * IV_XY_UM, iv_z * IV_Z_UM])

    print(f"  {tile}: {N} cells")
    del nd2_slices

all_ev = np.array(all_ev_stitched)
all_iv = np.array(all_iv_um)
N_LM = len(all_ev)
print(f"\n{TARGET_ROW} total landmarks: {N_LM}")

# ============================================================
# Compute 3D affine for this row
# ============================================================
print("\nFitting 3D affine...")
ones = np.ones((N_LM, 1))
src_h = np.hstack([all_iv, ones])
A_T, _, _, _ = np.linalg.lstsq(src_h, all_ev, rcond=None)
A = A_T.T

predicted = src_h @ A_T
errors = np.sqrt(np.sum((predicted - all_ev) ** 2, axis=1))
print(f"  Mean error: {errors.mean():.2f} µm, median: {np.median(errors):.2f}, max: {errors.max():.2f}")

# SVD decomposition
R = A[:, :3]
t = A[:, 3]
U, S, Vt = np.linalg.svd(R)
print(f"  Scale factors: {S[0]:.3f}, {S[1]:.3f}, {S[2]:.3f}")
print(f"  Translation: ({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}) µm")

# ============================================================
# Build forward/inverse in (z,y,x) pixel convention
# ============================================================
sx, sy, sz = IV_XY_UM, IV_XY_UM, IV_Z_UM
M_fwd = np.array([
    [A[2,2]*sz, A[2,1]*sy, A[2,0]*sx],
    [A[1,2]*sz, A[1,1]*sy, A[1,0]*sx],
    [A[0,2]*sz, A[0,1]*sy, A[0,0]*sx],
])
t_fwd = np.array([A[2,3], A[1,3], A[0,3]])
M_inv = np.linalg.inv(M_fwd)
offset_inv = -M_inv @ t_fwd

# ============================================================
# Load volumes
# ============================================================
print("\nLoading stitched volume...")
with tifffile.TiffFile(f'{BASE}/registration_video/stitched/stitched_gfp_iou_only_1um_isotropic.tif') as tif:
    n_pages = len(tif.pages)
    h, w = tif.pages[0].shape
    ev_vol = np.zeros((n_pages, h, w), dtype=np.uint16)
    for i, page in enumerate(tif.pages):
        ev_vol[i] = page.asarray()
print(f"  Ex-vivo: {ev_vol.shape}")

print("Loading JY306...")
iv_vol_raw = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv, ny_iv, nx_iv = iv_vol_raw.shape
print(f"  In-vivo: {iv_vol_raw.shape}")

print("Median filter...")
iv_vol = np.zeros_like(iv_vol_raw)
for z in range(nz_iv):
    bg = median_filter(iv_vol_raw[z], size=15)
    iv_vol[z] = np.clip(iv_vol_raw[z] - bg, 0, None)
del iv_vol_raw

# ============================================================
# Find which in-vivo z-slices overlap with this row's z-range
# ============================================================
iv_z_to_stitch = []
for z_iv in range(nz_iv):
    center = np.array([z_iv, ny_iv/2, nx_iv/2])
    out = M_fwd @ center + t_fwd
    z_st = int(round(out[0]))
    iv_z_to_stitch.append(z_st)

# Filter to slices that overlap with this row's z-range (with some margin)
margin = 30
overlap_slices = [(z_iv, z_st) for z_iv, z_st in enumerate(iv_z_to_stitch)
                  if z_iso_lo - margin <= z_st <= z_iso_hi + margin]
print(f"\nIn-vivo slices overlapping {TARGET_ROW} (z={z_iso_lo}-{z_iso_hi}):")
for z_iv, z_st in overlap_slices:
    print(f"  iv_z={z_iv} → stitched_z={z_st}")

if not overlap_slices:
    print("No overlap! Showing all 16 slices.")
    overlap_slices = [(z_iv, iv_z_to_stitch[z_iv]) for z_iv in range(nz_iv)]

# ============================================================
# XY crop
# ============================================================
corners_iv = np.array([
    [0,0,0],[0,0,nx_iv-1],[0,ny_iv-1,0],[0,ny_iv-1,nx_iv-1],
    [nz_iv-1,0,0],[nz_iv-1,0,nx_iv-1],[nz_iv-1,ny_iv-1,0],[nz_iv-1,ny_iv-1,nx_iv-1]
], dtype=np.float64)
corners_out = (M_fwd @ corners_iv.T).T + t_fwd
y_lo = max(0, int(np.floor(corners_out[:,1].min())))
y_hi = min(ev_vol.shape[1], int(np.ceil(corners_out[:,1].max())) + 1)
x_lo = max(0, int(np.floor(corners_out[:,2].min())))
x_hi = min(ev_vol.shape[2], int(np.ceil(corners_out[:,2].max())) + 1)
crop_h = y_hi - y_lo
crop_w = x_hi - x_lo

DS = max(1, crop_w // 600)
out_w = crop_w // DS
out_h = crop_h // DS
print(f"  XY crop: {crop_h}x{crop_w}, DS{DS} → {out_h}x{out_w}")

# Predicted landmarks
iv_pred = src_h @ A_T

def norm8(img):
    vals = img[img > 0]
    if len(vals) < 100:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(vals, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)

# ============================================================
# Build contact sheet
# ============================================================
n_slices = len(overlap_slices)
LABEL_H = 30
GAP = 4
COLS = 3
row_h = out_h + LABEL_H + GAP
HEADER_H = 50

sheet_w = COLS * out_w + (COLS + 1) * GAP
sheet_h = HEADER_H + n_slices * row_h + GAP
sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

print(f"\nContact sheet: {sheet_w}x{sheet_h}, {n_slices} slices")

for si, (z_iv, z_st) in enumerate(overlap_slices):
    z_st_clip = np.clip(z_st, 0, ev_vol.shape[0] - 1)
    print(f"  [{si+1}/{n_slices}] iv_z={z_iv} → stitch_z={z_st}", end="")

    # 2D warp
    M2d = np.array([
        [M_inv[2,2], M_inv[2,1], M_inv[2,0]*z_st_clip + M_inv[2,1]*y_lo + M_inv[2,2]*x_lo + offset_inv[2]],
        [M_inv[1,2], M_inv[1,1], M_inv[1,0]*z_st_clip + M_inv[1,1]*y_lo + M_inv[1,2]*x_lo + offset_inv[1]],
    ], dtype=np.float64)

    iv_warped = cv2.warpAffine(iv_vol[z_iv].astype(np.float32), M2d, (crop_w, crop_h),
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    ev_slice = ev_vol[z_st_clip, y_lo:y_hi, x_lo:x_hi].astype(np.float32)

    ev_ds = cv2.resize(ev_slice, (out_w, out_h), interpolation=cv2.INTER_AREA)
    iv_ds = cv2.resize(iv_warped, (out_w, out_h), interpolation=cv2.INTER_AREA)

    ev_rgb = cv2.cvtColor(norm8(ev_ds), cv2.COLOR_GRAY2BGR)
    iv_rgb = cv2.cvtColor(norm8(iv_ds), cv2.COLOR_GRAY2BGR)

    ov_rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    ov_rgb[:,:,1] = norm8(ev_ds)
    ov_rgb[:,:,0] = norm8(iv_ds)
    ov_rgb[:,:,2] = norm8(iv_ds)

    # Draw landmarks
    n_lm = 0
    for i in range(N_LM):
        if abs(all_ev[i, 2] - z_st) > 20:
            continue
        n_lm += 1
        R = 8
        ex_x = int(round((all_ev[i, 0] - x_lo) / DS))
        ex_y = int(round((all_ev[i, 1] - y_lo) / DS))
        iv_x = int(round((iv_pred[i, 0] - x_lo) / DS))
        iv_y = int(round((iv_pred[i, 1] - y_lo) / DS))

        cv2.drawMarker(ev_rgb, (ex_x, ex_y), (255, 255, 0), cv2.MARKER_CROSS, R*2, 2)
        cv2.drawMarker(iv_rgb, (iv_x, iv_y), (0, 255, 255), cv2.MARKER_CROSS, R*2, 2)

        cv2.drawMarker(ov_rgb, (ex_x, ex_y), (255, 255, 0), cv2.MARKER_CROSS, R*2, 1)
        cv2.drawMarker(ov_rgb, (iv_x, iv_y), (0, 255, 255), cv2.MARKER_CROSS, R*2, 1)
        cv2.line(ov_rgb, (ex_x, ex_y), (iv_x, iv_y), (255, 255, 255), 1)

    print(f"  {n_lm} landmarks")

    y0 = HEADER_H + si * row_h + GAP
    for ci, p in enumerate([ev_rgb, iv_rgb, ov_rgb]):
        x0 = GAP + ci * (out_w + GAP)
        sheet[y0:y0+out_h, x0:x0+out_w] = p

    cv2.putText(sheet, f'iv_z={z_iv} -> stitch_z={z_st}  ({n_lm} lm)',
                (GAP + 4, y0 + out_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# Headers
for ci, h in enumerate(['Ex-vivo + landmarks (cyan)', 'In-vivo warped + predicted (yellow)', 'Overlay + error']):
    cv2.putText(sheet, h, (GAP + ci * (out_w + GAP) + 4, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

cv2.putText(sheet, f'{TARGET_ROW}: {N_LM} landmarks | per-row affine | mean err={errors.mean():.1f}um | median={np.median(errors):.1f}um',
            (GAP + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

out_path = f'{BASE}/registration_overlay_{TARGET_ROW}.png'
cv2.imwrite(out_path, sheet)
print(f"\nSaved: {out_path} ({sheet.shape[1]}x{sheet.shape[0]})")

# Also save the per-row affine
np.savez(f'{BASE}/registration_video/affine_3d_{TARGET_ROW}.npz',
         affine_3x4=A, ev_stitched_um=all_ev, iv_um=all_iv,
         errors=errors, predicted=predicted)
print(f"Affine saved: affine_3d_{TARGET_ROW}.npz")
