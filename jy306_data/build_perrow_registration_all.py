#!/usr/bin/env python3
"""
Per-row affine registration for ALL rows.
For each row: use its landmarks, compute 3D affine, warp overlapping
in-vivo z-slices, save contact sheet PNG.
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

ALL_ROWS = ['row1', 'row2', 'row3', 'row4', 'row5']
OUT_DIR = f'{BASE}/png_exports/registration_perrow'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Load shared data once
# ============================================================
print("Loading stitch params...")
with open(f'{BASE}/registration_video/stitch_iou_only_params.json') as f:
    params = json.load(f)
TILE_ORDER = params['tile_order']
tile_z_offsets = {k: int(v) for k, v in params['tile_z_offsets'].items()}
cum_iou = {k: np.array(v) for k, v in params['cumulative_iou'].items()}

print("Loading stitched volume...")
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

# Load all landmark files into per-tile dict
print("Loading landmark files...")
lm_files = sorted(glob.glob(f'{BASE}/registration_video/landmarks_nd2_native_*.npz'))
legacy = f'{BASE}/registration_video/landmarks_27_nd2_native.npz'
if os.path.exists(legacy):
    lm_files.append(legacy)

tile_landmarks = {}
for lm_file in lm_files:
    bn = os.path.basename(lm_file)
    if 'landmarks_27_nd2_native' in bn:
        tile = 'row2_1'
    else:
        tile = bn.replace('landmarks_nd2_native_', '').replace('.npz', '')
    if tile not in TILE_ORDER:
        continue
    d = np.load(lm_file)
    tile_landmarks[tile] = {'ev_nd2': d['ev_nd2'], 'pcd_iv': d['pcd_invivo_jy306']}
print(f"  {len(tile_landmarks)} tiles with landmarks")


def norm8(img):
    vals = img[img > 0]
    if len(vals) < 100:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(vals, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


# ============================================================
# Process each row
# ============================================================
for TARGET_ROW in ALL_ROWS:
    print(f"\n{'='*60}")
    print(f"  {TARGET_ROW}")
    print(f"{'='*60}")

    row_tiles = [t for t in TILE_ORDER if t.startswith(TARGET_ROW + '_')]
    print(f"  Tiles: {row_tiles}")

    z_native_lo = tile_z_offsets[row_tiles[0]]
    z_native_hi = tile_z_offsets[row_tiles[-1]] + 12
    z_iso_lo = int(z_native_lo * NATIVE_Z_UM)
    z_iso_hi = int(z_native_hi * NATIVE_Z_UM)
    print(f"  Stitched z: [{z_iso_lo}, {z_iso_hi}]")

    # Collect landmarks
    all_ev_stitched = []
    all_iv_um = []

    for tile in row_tiles:
        if tile not in tile_landmarks:
            continue
        ev_nd2 = tile_landmarks[tile]['ev_nd2']
        pcd_iv = tile_landmarks[tile]['pcd_iv']
        N = ev_nd2.shape[0]

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

            Mp = cum_iou[tile]
            p = Mp @ np.array([x_nd2, y_nd2, 1.0])
            all_ev_stitched.append([p[0] * NATIVE_XY_UM, p[1] * NATIVE_XY_UM,
                                    (tile_z_offsets[tile] + z_nd2) * NATIVE_Z_UM])

            iv_z, iv_y, iv_x = pcd_iv[i]
            all_iv_um.append([iv_x * IV_XY_UM, iv_y * IV_XY_UM, iv_z * IV_Z_UM])

        del nd2_slices
        print(f"    {tile}: {N} cells")

    all_ev = np.array(all_ev_stitched) if all_ev_stitched else np.zeros((0, 3))
    all_iv = np.array(all_iv_um) if all_iv_um else np.zeros((0, 3))
    N_LM = len(all_ev)

    if N_LM < 4:
        print(f"  SKIP: only {N_LM} landmarks")
        continue

    print(f"  Total: {N_LM} landmarks")

    # Fit affine
    ones = np.ones((N_LM, 1))
    src_h = np.hstack([all_iv, ones])
    A_T, _, _, _ = np.linalg.lstsq(src_h, all_ev, rcond=None)
    A = A_T.T
    predicted = src_h @ A_T
    errors = np.sqrt(np.sum((predicted - all_ev) ** 2, axis=1))
    Rm = A[:, :3]
    _, S, _ = np.linalg.svd(Rm)
    print(f"  Affine: mean={errors.mean():.1f}µm median={np.median(errors):.1f}µm max={errors.max():.1f}µm")
    print(f"  Scales: {S[0]:.2f}, {S[1]:.2f}, {S[2]:.2f}")

    # (z,y,x) pixel transforms
    sx, sy, sz = IV_XY_UM, IV_XY_UM, IV_Z_UM
    M_fwd = np.array([
        [A[2,2]*sz, A[2,1]*sy, A[2,0]*sx],
        [A[1,2]*sz, A[1,1]*sy, A[1,0]*sx],
        [A[0,2]*sz, A[0,1]*sy, A[0,0]*sx],
    ])
    t_fwd = np.array([A[2,3], A[1,3], A[0,3]])
    M_inv = np.linalg.inv(M_fwd)
    offset_inv = -M_inv @ t_fwd

    # Overlapping in-vivo slices
    margin = 30
    overlap = []
    for z_iv in range(nz_iv):
        center = np.array([z_iv, ny_iv / 2, nx_iv / 2])
        z_st = int(round((M_fwd @ center + t_fwd)[0]))
        if z_iso_lo - margin <= z_st <= z_iso_hi + margin:
            overlap.append((z_iv, z_st))

    if not overlap:
        print(f"  No overlap! Using all.")
        overlap = [(z, int(round((M_fwd @ np.array([z, ny_iv/2, nx_iv/2]) + t_fwd)[0]))) for z in range(nz_iv)]

    print(f"  {len(overlap)} overlapping slices")

    # XY crop
    corners_iv = np.array([
        [0,0,0],[0,0,nx_iv-1],[0,ny_iv-1,0],[0,ny_iv-1,nx_iv-1],
        [nz_iv-1,0,0],[nz_iv-1,0,nx_iv-1],[nz_iv-1,ny_iv-1,0],[nz_iv-1,ny_iv-1,nx_iv-1]
    ], dtype=np.float64)
    corners_out = (M_fwd @ corners_iv.T).T + t_fwd
    y_lo = max(0, int(np.floor(corners_out[:, 1].min())))
    y_hi = min(ev_vol.shape[1], int(np.ceil(corners_out[:, 1].max())) + 1)
    x_lo = max(0, int(np.floor(corners_out[:, 2].min())))
    x_hi = min(ev_vol.shape[2], int(np.ceil(corners_out[:, 2].max())) + 1)
    crop_h, crop_w = y_hi - y_lo, x_hi - x_lo
    DS = max(1, crop_w // 600)
    out_w, out_h = crop_w // DS, crop_h // DS

    iv_pred = src_h @ A_T

    # Contact sheet
    n_sl = len(overlap)
    LABEL_H, GAP, COLS, HEADER_H = 30, 4, 3, 50
    row_ht = out_h + LABEL_H + GAP
    sheet_w = COLS * out_w + (COLS + 1) * GAP
    sheet_h = HEADER_H + n_sl * row_ht + GAP
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for si, (z_iv, z_st) in enumerate(overlap):
        z_st_c = np.clip(z_st, 0, ev_vol.shape[0] - 1)

        M2d = np.array([
            [M_inv[2,2], M_inv[2,1], M_inv[2,0]*z_st_c + M_inv[2,1]*y_lo + M_inv[2,2]*x_lo + offset_inv[2]],
            [M_inv[1,2], M_inv[1,1], M_inv[1,0]*z_st_c + M_inv[1,1]*y_lo + M_inv[1,2]*x_lo + offset_inv[1]],
        ], dtype=np.float64)

        iv_w = cv2.warpAffine(iv_vol[z_iv].astype(np.float32), M2d, (crop_w, crop_h),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
        ev_sl = ev_vol[z_st_c, y_lo:y_hi, x_lo:x_hi].astype(np.float32)

        ev_d = cv2.resize(ev_sl, (out_w, out_h), interpolation=cv2.INTER_AREA)
        iv_d = cv2.resize(iv_w, (out_w, out_h), interpolation=cv2.INTER_AREA)

        ev_rgb = cv2.cvtColor(norm8(ev_d), cv2.COLOR_GRAY2BGR)
        iv_rgb = cv2.cvtColor(norm8(iv_d), cv2.COLOR_GRAY2BGR)
        ov_rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        ov_rgb[:, :, 1] = norm8(ev_d)
        ov_rgb[:, :, 0] = norm8(iv_d)
        ov_rgb[:, :, 2] = norm8(iv_d)

        n_lm = 0
        for i in range(N_LM):
            if abs(all_ev[i, 2] - z_st) > 20:
                continue
            n_lm += 1
            ex_x = int(round((all_ev[i, 0] - x_lo) / DS))
            ex_y = int(round((all_ev[i, 1] - y_lo) / DS))
            iv_x = int(round((iv_pred[i, 0] - x_lo) / DS))
            iv_y = int(round((iv_pred[i, 1] - y_lo) / DS))

            cv2.drawMarker(ev_rgb, (ex_x, ex_y), (255, 255, 0), cv2.MARKER_CROSS, 16, 2)
            cv2.drawMarker(iv_rgb, (iv_x, iv_y), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.drawMarker(ov_rgb, (ex_x, ex_y), (255, 255, 0), cv2.MARKER_CROSS, 16, 1)
            cv2.drawMarker(ov_rgb, (iv_x, iv_y), (0, 255, 255), cv2.MARKER_CROSS, 16, 1)
            cv2.line(ov_rgb, (ex_x, ex_y), (iv_x, iv_y), (255, 255, 255), 1)

        y0 = HEADER_H + si * row_ht + GAP
        for ci, p in enumerate([ev_rgb, iv_rgb, ov_rgb]):
            x0 = GAP + ci * (out_w + GAP)
            sheet[y0:y0 + out_h, x0:x0 + out_w] = p

        cv2.putText(sheet, f'iv_z={z_iv} -> stitch_z={z_st}  ({n_lm} lm)',
                    (GAP + 4, y0 + out_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    for ci, hdr in enumerate(['Ex-vivo + landmarks (cyan)', 'In-vivo warped (yellow)', 'Overlay + error']):
        cv2.putText(sheet, hdr, (GAP + ci * (out_w + GAP) + 4, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    cv2.putText(sheet,
                f'{TARGET_ROW}: {N_LM} lm | mean={errors.mean():.1f}um median={np.median(errors):.1f}um | scales={S[0]:.1f},{S[1]:.1f},{S[2]:.1f}',
                (GAP + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    out_path = f'{OUT_DIR}/registration_{TARGET_ROW}.png'
    cv2.imwrite(out_path, sheet)
    print(f"  Saved: {out_path}")

    np.savez(f'{BASE}/registration_video/affine_3d_{TARGET_ROW}.npz',
             affine_3x4=A, ev_stitched_um=all_ev, iv_um=all_iv,
             errors=errors, predicted=predicted)

print(f"\nDone! All saved to {OUT_DIR}/")
