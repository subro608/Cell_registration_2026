#!/usr/bin/env python3
"""
Per-row registration in NATIVE space (no stitching).
For each row:
  - Stack native nd2 GFP tiles into a row volume (N_tiles × 12 z-slices, 4200×4200)
  - Use landmarks in nd2 native coords (ev_nd2) and JY306 native coords (pcd_invivo)
  - Fit 3D affine: JY306 µm → nd2 row µm
  - Warp in-vivo into nd2 space per z-slice
  - Contact sheet: ex-vivo | in-vivo warped | overlay
"""
import numpy as np
import tifffile
import cv2
import os
import json
import glob
from scipy.ndimage import median_filter

BASE = '/Users/neurolab/neuroinformatics/margaret'
PNG_DIR = f'{BASE}/png_exports/registration_video'
OUT_DIR = f'{BASE}/png_exports/registration_perrow_native'
os.makedirs(OUT_DIR, exist_ok=True)

IV_XY_UM = 0.6835
IV_Z_UM = 3.0
ND2_XY_UM = 0.645
ND2_Z_UM = 2.0

ALL_ROWS = ['row1', 'row2', 'row3', 'row4', 'row5']

# ============================================================
# Load shared data
# ============================================================
print("Loading stitch params (for tile order only)...")
with open(f'{BASE}/registration_video/stitch_iou_only_params.json') as f:
    params = json.load(f)
TILE_ORDER = params['tile_order']

print("Loading JY306 in-vivo...")
iv_vol_raw = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv, ny_iv, nx_iv = iv_vol_raw.shape
print(f"  In-vivo: {iv_vol_raw.shape}")

print("Median filter...")
iv_vol = np.zeros_like(iv_vol_raw)
for z in range(nz_iv):
    bg = median_filter(iv_vol_raw[z], size=15)
    iv_vol[z] = np.clip(iv_vol_raw[z] - bg, 0, None)
del iv_vol_raw

# Load landmark files
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
    n_tiles = len(row_tiles)
    nz_row = n_tiles * 12
    print(f"  Tiles: {row_tiles} → {nz_row} z-slices")

    # --------------------------------------------------------
    # Collect landmarks in native coords
    # ev: (col, row) in nd2 4200px + z_row = tile_idx*12 + best_z
    # iv: (z, y, x) in JY306 pixels
    # --------------------------------------------------------
    ev_pts_um = []  # (x_um, y_um, z_um) in nd2 row space
    iv_pts_um = []  # (x_um, y_um, z_um) in JY306 space

    for ti, tile in enumerate(row_tiles):
        if tile not in tile_landmarks:
            continue
        ev_nd2 = tile_landmarks[tile]['ev_nd2']
        pcd_iv = tile_landmarks[tile]['pcd_iv']
        N = ev_nd2.shape[0]

        # Find best nd2 z per cell
        img_dir = f'{PNG_DIR}/{tile}'
        nd2_slices = []
        for zi in range(12):
            img = cv2.imread(f'{img_dir}/GFP_z{zi:03d}.png', cv2.IMREAD_UNCHANGED)
            if img is None:
                nd2_slices.append(np.zeros((4200, 4200), dtype=np.float32))
            else:
                nd2_slices.append(img.astype(np.float32))
        nd2_slices = np.array(nd2_slices)

        for i in range(N):
            col, row_px = ev_nd2[i, 0], ev_nd2[i, 1]
            c = int(round(np.clip(col, 10, 4189)))
            r = int(round(np.clip(row_px, 10, 4189)))
            intensities = [nd2_slices[z][r-10:r+10, c-10:c+10].mean() for z in range(12)]
            best_z = np.argmax(intensities)

            z_row = ti * 12 + best_z

            ev_pts_um.append([col * ND2_XY_UM, row_px * ND2_XY_UM, z_row * ND2_Z_UM])

            iv_z, iv_y, iv_x = pcd_iv[i]
            iv_pts_um.append([iv_x * IV_XY_UM, iv_y * IV_XY_UM, iv_z * IV_Z_UM])

        del nd2_slices
        print(f"    {tile}: {N} cells")

    ev_pts = np.array(ev_pts_um) if ev_pts_um else np.zeros((0, 3))
    iv_pts = np.array(iv_pts_um) if iv_pts_um else np.zeros((0, 3))
    N_LM = len(ev_pts)

    if N_LM < 4:
        print(f"  SKIP: only {N_LM} landmarks")
        continue

    print(f"  Total: {N_LM} landmarks")
    print(f"  EV range: x=[{ev_pts[:,0].min():.0f},{ev_pts[:,0].max():.0f}] "
          f"y=[{ev_pts[:,1].min():.0f},{ev_pts[:,1].max():.0f}] "
          f"z=[{ev_pts[:,2].min():.0f},{ev_pts[:,2].max():.0f}] µm")
    print(f"  IV range: x=[{iv_pts[:,0].min():.0f},{iv_pts[:,0].max():.0f}] "
          f"y=[{iv_pts[:,1].min():.0f},{iv_pts[:,1].max():.0f}] "
          f"z=[{iv_pts[:,2].min():.0f},{iv_pts[:,2].max():.0f}] µm")

    # --------------------------------------------------------
    # Fit 3D affine: JY306 µm → nd2 row µm
    # --------------------------------------------------------
    ones = np.ones((N_LM, 1))
    src_h = np.hstack([iv_pts, ones])
    A_T, _, _, _ = np.linalg.lstsq(src_h, ev_pts, rcond=None)
    A = A_T.T

    predicted = src_h @ A_T
    errors = np.sqrt(np.sum((predicted - ev_pts) ** 2, axis=1))
    Rm = A[:, :3]
    _, S, _ = np.linalg.svd(Rm)
    print(f"  Affine: mean={errors.mean():.1f}µm median={np.median(errors):.1f}µm max={errors.max():.1f}µm")
    print(f"  Scales: {S[0]:.2f}, {S[1]:.2f}, {S[2]:.2f}")

    # --------------------------------------------------------
    # Build (z,y,x) pixel-space transforms
    # Forward: JY306 pixel → nd2 row pixel
    # A maps JY306 µm → nd2 µm, convert pixel→µm→affine→µm→pixel
    # --------------------------------------------------------
    sx_iv, sy_iv, sz_iv = IV_XY_UM, IV_XY_UM, IV_Z_UM
    sx_ev, sy_ev, sz_ev = ND2_XY_UM, ND2_XY_UM, ND2_Z_UM

    # Forward in (z,y,x) pixels: nd2_px = M_fwd @ jy306_px + t_fwd
    M_fwd = np.array([
        [A[2,2]*sz_iv/sz_ev, A[2,1]*sy_iv/sy_ev, A[2,0]*sx_iv/sx_ev],
        [A[1,2]*sz_iv/sy_ev, A[1,1]*sy_iv/sy_ev, A[1,0]*sx_iv/sx_ev],
        [A[0,2]*sz_iv/sx_ev, A[0,1]*sy_iv/sx_ev, A[0,0]*sx_iv/sx_ev],
    ])
    t_fwd = np.array([A[2,3]/sz_ev, A[1,3]/sy_ev, A[0,3]/sx_ev])

    M_inv = np.linalg.inv(M_fwd)
    offset_inv = -M_inv @ t_fwd

    # --------------------------------------------------------
    # Find overlapping in-vivo slices
    # --------------------------------------------------------
    overlap = []
    for z_iv in range(nz_iv):
        center = np.array([z_iv, ny_iv / 2, nx_iv / 2])
        z_nd2 = (M_fwd @ center + t_fwd)[0]
        z_nd2_int = int(round(z_nd2))
        if -2 <= z_nd2_int <= nz_row + 2:
            overlap.append((z_iv, z_nd2_int))

    if not overlap:
        overlap = [(z, int(round((M_fwd @ np.array([z, ny_iv/2, nx_iv/2]) + t_fwd)[0])))
                    for z in range(nz_iv)]

    print(f"  {len(overlap)} overlapping in-vivo slices")
    for z_iv, z_nd2 in overlap:
        print(f"    iv_z={z_iv} → nd2_row_z={z_nd2}")

    # --------------------------------------------------------
    # Load row nd2 volume (lazy, per z-slice as needed)
    # --------------------------------------------------------
    DS = 4
    out_w = 4200 // DS
    out_h = 4200 // DS

    def load_nd2_z(z_row):
        """Load a single z-slice from the row volume."""
        if z_row < 0 or z_row >= nz_row:
            return np.zeros((4200, 4200), dtype=np.float32)
        ti = z_row // 12
        zi = z_row % 12
        tile = row_tiles[ti]
        path = f'{PNG_DIR}/{tile}/GFP_z{zi:03d}.png'
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return np.zeros((4200, 4200), dtype=np.float32)
        return img.astype(np.float32)

    iv_pred = src_h @ A_T  # predicted nd2 µm positions

    # --------------------------------------------------------
    # Build contact sheet
    # --------------------------------------------------------
    n_sl = len(overlap)
    LABEL_H, GAP, COLS, HEADER_H = 30, 4, 3, 50
    row_ht = out_h + LABEL_H + GAP
    sheet_w = COLS * out_w + (COLS + 1) * GAP
    sheet_h = HEADER_H + n_sl * row_ht + GAP
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for si, (z_iv, z_nd2) in enumerate(overlap):
        z_nd2_c = np.clip(z_nd2, 0, nz_row - 1)
        print(f"  [{si+1}/{n_sl}] iv_z={z_iv} → nd2_z={z_nd2}", end="")

        # 2D backward warp: for each (y_out, x_out) in nd2 4200 space,
        # find (y_in, x_in) in JY306 space
        M2d = np.array([
            [M_inv[2,2], M_inv[2,1], M_inv[2,0]*z_nd2_c + offset_inv[2]],
            [M_inv[1,2], M_inv[1,1], M_inv[1,0]*z_nd2_c + offset_inv[1]],
        ], dtype=np.float64)

        iv_warped = cv2.warpAffine(iv_vol[z_iv].astype(np.float32), M2d, (4200, 4200),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderValue=0)

        ev_slice = load_nd2_z(z_nd2_c)

        # DS
        ev_ds = cv2.resize(ev_slice, (out_w, out_h), interpolation=cv2.INTER_AREA)
        iv_ds = cv2.resize(iv_warped, (out_w, out_h), interpolation=cv2.INTER_AREA)

        ev_rgb = cv2.cvtColor(norm8(ev_ds), cv2.COLOR_GRAY2BGR)
        iv_rgb = cv2.cvtColor(norm8(iv_ds), cv2.COLOR_GRAY2BGR)
        ov_rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        ov_rgb[:, :, 1] = norm8(ev_ds)
        ov_rgb[:, :, 0] = norm8(iv_ds)
        ov_rgb[:, :, 2] = norm8(iv_ds)

        # Draw ALL landmarks on every slice (no z-filter)
        n_lm = 0
        for i in range(N_LM):
            n_lm += 1

            # Ex-vivo in DS coords
            ex_x = int(round(ev_pts[i, 0] / ND2_XY_UM / DS))
            ex_y = int(round(ev_pts[i, 1] / ND2_XY_UM / DS))
            # In-vivo predicted in nd2 DS coords
            iv_x = int(round(iv_pred[i, 0] / ND2_XY_UM / DS))
            iv_y = int(round(iv_pred[i, 1] / ND2_XY_UM / DS))

            cv2.drawMarker(ev_rgb, (ex_x, ex_y), (255, 255, 0), cv2.MARKER_CROSS, 16, 2)
            cv2.drawMarker(iv_rgb, (iv_x, iv_y), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.drawMarker(ov_rgb, (ex_x, ex_y), (255, 255, 0), cv2.MARKER_CROSS, 16, 1)
            cv2.drawMarker(ov_rgb, (iv_x, iv_y), (0, 255, 255), cv2.MARKER_CROSS, 16, 1)
            cv2.line(ov_rgb, (ex_x, ex_y), (iv_x, iv_y), (255, 255, 255), 1)

        print(f"  {n_lm} lm")

        y0 = HEADER_H + si * row_ht + GAP
        for ci, p in enumerate([ev_rgb, iv_rgb, ov_rgb]):
            x0 = GAP + ci * (out_w + GAP)
            sheet[y0:y0+out_h, x0:x0+out_w] = p

        ti_name = row_tiles[z_nd2_c // 12] if 0 <= z_nd2_c < nz_row else '?'
        zi_in_tile = z_nd2_c % 12
        cv2.putText(sheet, f'iv_z={z_iv} -> {ti_name} z{zi_in_tile} (row_z={z_nd2})  ({n_lm} lm)',
                    (GAP + 4, y0 + out_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    for ci, hdr in enumerate(['Ex-vivo native nd2 (cyan=lm)', 'In-vivo warped (yellow=pred)', 'Overlay + error']):
        cv2.putText(sheet, hdr, (GAP + ci * (out_w + GAP) + 4, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    cv2.putText(sheet,
                f'{TARGET_ROW} NATIVE: {N_LM} lm | mean={errors.mean():.1f}um median={np.median(errors):.1f}um | {n_tiles} tiles x 12z',
                (GAP + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    out_path = f'{OUT_DIR}/registration_{TARGET_ROW}_native.png'
    cv2.imwrite(out_path, sheet)
    print(f"  Saved: {out_path}")

    np.savez(f'{BASE}/registration_video/affine_3d_{TARGET_ROW}_native.npz',
             affine_3x4=A, ev_pts_um=ev_pts, iv_pts_um=iv_pts,
             errors=errors, predicted=predicted)

print(f"\nDone! All saved to {OUT_DIR}/")