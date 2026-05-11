#!/usr/bin/env python3
"""
All tiles: 3D affine + RBF warp of in-vivo JY306 → each nd2 tile.
For each tile with landmarks:
  1. Gaussian z-fit for nd2 z
  2. 3D affine (all landmarks, no outlier removal)
  3. RBF (thin-plate spline) on affine residuals
  4. Dense displacement field: affine + RBF → remap in-vivo image
  5. Save PNG contact sheet + HTML + affine .npz
"""
import numpy as np
import cv2
import os
import base64
import json
import glob
import tifffile
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.interpolate import RBFInterpolator

BASE = '/Users/neurolab/neuroinformatics/margaret'

IV_XY_UM = 0.6835
IV_Z_UM = 3.0
ND2_XY_UM = 0.645
ND2_Z_UM = 2.0

OUT_BASE = f'{BASE}/png_exports/registration_per_tile'
os.makedirs(OUT_BASE, exist_ok=True)

# ============================================================
# Load shared data once
# ============================================================
print("Loading stitch params...")
with open(f'{BASE}/registration_video/stitch_iou_only_params.json') as f:
    params = json.load(f)
TILE_ORDER = params['tile_order']

print("Loading JY306 in-vivo...")
iv_vol_raw = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv, ny_iv, nx_iv = iv_vol_raw.shape
print(f"  In-vivo: {iv_vol_raw.shape}")

print("Median filter bg subtraction...")
iv_vol = np.zeros_like(iv_vol_raw)
for z in range(nz_iv):
    bg = median_filter(iv_vol_raw[z], size=15)
    iv_vol[z] = np.clip(iv_vol_raw[z] - bg, 0, None)
del iv_vol_raw

# Discover all landmark files
print("Finding landmark files...")
lm_files = sorted(glob.glob(f'{BASE}/registration_video/landmarks_nd2_native_*.npz'))
legacy = f'{BASE}/registration_video/landmarks_27_nd2_native.npz'
if os.path.exists(legacy):
    lm_files.append(legacy)

tile_lm_files = {}
for lm_file in lm_files:
    bn = os.path.basename(lm_file)
    if 'landmarks_27_nd2_native' in bn:
        tile = 'row2_1'
    else:
        tile = bn.replace('landmarks_nd2_native_', '').replace('.npz', '')
    if tile in TILE_ORDER:
        tile_lm_files[tile] = lm_file

print(f"  {len(tile_lm_files)} tiles with landmarks")


# ============================================================
# Helpers
# ============================================================
def gauss(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def find_z_gaussian(intensities):
    zs = np.arange(len(intensities), dtype=np.float64)
    vals = np.array(intensities, dtype=np.float64)
    vals = vals - vals.min()
    total = vals.sum()
    if total < 1e-6:
        return float(np.argmax(intensities))
    centroid = float(np.sum(zs * vals) / total)
    peak_z = np.argmax(vals)
    try:
        p0 = [vals[peak_z], float(peak_z), 2.0]
        popt, _ = curve_fit(gauss, zs, vals, p0=p0,
                            bounds=([0, -1, 0.3], [vals.max() * 3, 12, 8]),
                            maxfev=1000)
        mu = popt[1]
        if 0 <= mu <= 11:
            return mu
    except (RuntimeError, ValueError):
        pass
    return centroid


def norm8(img):
    vals = img[img > 0]
    if len(vals) < 100:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(vals, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


def to_b64(rgb_img):
    _, buf = cv2.imencode('.jpg', rgb_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('ascii')


# ============================================================
# Process each tile
# ============================================================
summary = []

for tile in sorted(tile_lm_files.keys()):
    print(f"\n{'='*60}")
    print(f"  {tile}")
    print(f"{'='*60}")

    out_dir = f'{OUT_BASE}/{tile}'
    os.makedirs(out_dir, exist_ok=True)

    # Load nd2 tile slices
    img_dir = f'{BASE}/png_exports/registration_video/{tile}'
    nd2_slices = []
    for zi in range(12):
        img = cv2.imread(f'{img_dir}/GFP_z{zi:03d}.png', cv2.IMREAD_UNCHANGED)
        if img is None:
            nd2_slices.append(np.zeros((4200, 4200), dtype=np.float32))
        else:
            nd2_slices.append(img.astype(np.float32))
    nd2_slices = np.array(nd2_slices)
    nd2_h, nd2_w = nd2_slices.shape[1], nd2_slices.shape[2]

    # Load landmarks
    d = np.load(tile_lm_files[tile])
    ev_nd2 = d['ev_nd2']
    pcd_iv = d['pcd_invivo_jy306']
    N_LM = ev_nd2.shape[0]

    if N_LM < 4:
        print(f"  SKIP: only {N_LM} landmarks (need >=4)")
        continue

    # Gaussian z for nd2
    nd2_z_vals = []
    for i in range(N_LM):
        x, y = ev_nd2[i, 0], ev_nd2[i, 1]
        c = int(round(np.clip(x, 10, nd2_h - 11)))
        r = int(round(np.clip(y, 10, nd2_h - 11)))
        intensities = [nd2_slices[z][r-10:r+10, c-10:c+10].mean() for z in range(12)]
        nd2_z_vals.append(find_z_gaussian(intensities))

    # 3D affine (all landmarks)
    src = np.column_stack([pcd_iv[:, 2] * IV_XY_UM, pcd_iv[:, 1] * IV_XY_UM, pcd_iv[:, 0] * IV_Z_UM])
    dst = np.column_stack([ev_nd2[:, 0] * ND2_XY_UM, ev_nd2[:, 1] * ND2_XY_UM, np.array(nd2_z_vals) * ND2_Z_UM])

    src_h = np.hstack([src, np.ones((N_LM, 1))])
    A_T, _, _, _ = np.linalg.lstsq(src_h, dst, rcond=None)
    A = A_T.T
    predicted = src_h @ A_T
    errors = np.sqrt(np.sum((predicted - dst) ** 2, axis=1))
    Rm = A[:, :3]
    _, S, _ = np.linalg.svd(Rm)
    print(f"  {N_LM} lm | affine: mean={errors.mean():.1f}µm median={np.median(errors):.1f}µm max={errors.max():.1f}µm")
    print(f"  Scales: {S[0]:.3f}, {S[1]:.3f}, {S[2]:.3f}")

    # Pixel-space transforms
    sx, sy, sz = IV_XY_UM, IV_XY_UM, IV_Z_UM
    ex, ey, ez = ND2_XY_UM, ND2_XY_UM, ND2_Z_UM
    M_fwd = np.array([
        [A[2,2]*sz/ez, A[2,1]*sy/ez, A[2,0]*sx/ez],
        [A[1,2]*sz/ey, A[1,1]*sy/ey, A[1,0]*sx/ey],
        [A[0,2]*sz/ex, A[0,1]*sy/ex, A[0,0]*sx/ex],
    ])
    t_fwd = np.array([A[2,3]/ez, A[1,3]/ey, A[0,3]/ex])
    M_inv = np.linalg.inv(M_fwd)
    offset_inv = -M_inv @ t_fwd

    # RBF on residuals
    lm_nd2_px = np.column_stack([ev_nd2[:, 0], ev_nd2[:, 1]])
    lm_nd2_zyx = np.column_stack([np.array(nd2_z_vals), ev_nd2[:, 1], ev_nd2[:, 0]])
    lm_iv_zyx_from_affine = (M_inv @ lm_nd2_zyx.T).T + offset_inv
    lm_iv_zyx_true = np.column_stack([pcd_iv[:, 0], pcd_iv[:, 1], pcd_iv[:, 2]])
    corrections_iv = lm_iv_zyx_true - lm_iv_zyx_from_affine

    rbf_corr_x = RBFInterpolator(lm_nd2_px, corrections_iv[:, 2], kernel='thin_plate_spline', smoothing=1.0)
    rbf_corr_y = RBFInterpolator(lm_nd2_px, corrections_iv[:, 1], kernel='thin_plate_spline', smoothing=1.0)

    # Downsample
    DS = max(1, nd2_w // 600)
    out_w, out_h = nd2_w // DS, nd2_h // DS
    iv_lm_z = pcd_iv[:, 0]
    nd2_lm_actual = ev_nd2[:, :2]
    nd2_lm_pred = predicted[:, :2] / ND2_XY_UM

    # Contact sheet
    LABEL_H, GAP, COLS, HEADER_H = 30, 4, 3, 60
    row_ht = out_h + LABEL_H + GAP
    sheet_w = COLS * out_w + (COLS + 1) * GAP
    sheet_h = HEADER_H + nz_iv * row_ht + GAP
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    rows_html = []

    for z_iv in range(nz_iv):
        center_iv = np.array([z_iv, ny_iv / 2, nx_iv / 2])
        center_nd2 = M_fwd @ center_iv + t_fwd
        z_nd2_float = center_nd2[0]
        z_nd2 = int(round(np.clip(z_nd2_float, 0, 11)))

        # Dense remap: affine + RBF
        DS_MAP = 4
        map_h, map_w = nd2_h // DS_MAP, nd2_w // DS_MAP
        ys_nd2 = np.arange(map_h) * DS_MAP
        xs_nd2 = np.arange(map_w) * DS_MAP
        xx_nd2, yy_nd2 = np.meshgrid(xs_nd2, ys_nd2)

        iv_x_affine = M_inv[2, 0] * z_nd2 + M_inv[2, 1] * yy_nd2 + M_inv[2, 2] * xx_nd2 + offset_inv[2]
        iv_y_affine = M_inv[1, 0] * z_nd2 + M_inv[1, 1] * yy_nd2 + M_inv[1, 2] * xx_nd2 + offset_inv[1]

        query_pts = np.column_stack([xx_nd2.ravel(), yy_nd2.ravel()])
        corr_x = rbf_corr_x(query_pts).reshape(map_h, map_w)
        corr_y = rbf_corr_y(query_pts).reshape(map_h, map_w)

        map_x = cv2.resize((iv_x_affine + corr_x).astype(np.float32), (nd2_w, nd2_h), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize((iv_y_affine + corr_y).astype(np.float32), (nd2_w, nd2_h), interpolation=cv2.INTER_LINEAR)

        iv_warped = cv2.remap(iv_vol[z_iv], map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
        nd2_sl = nd2_slices[z_nd2]

        ev_d = cv2.resize(nd2_sl, (out_w, out_h), interpolation=cv2.INTER_AREA)
        iv_d = cv2.resize(iv_warped, (out_w, out_h), interpolation=cv2.INTER_AREA)

        ev_rgb = cv2.cvtColor(norm8(ev_d), cv2.COLOR_GRAY2BGR)
        iv_rgb = cv2.cvtColor(norm8(iv_d), cv2.COLOR_GRAY2BGR)
        ov_rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        ov_rgb[:, :, 1] = norm8(ev_d)
        ov_rgb[:, :, 0] = norm8(iv_d)
        ov_rgb[:, :, 2] = norm8(iv_d)

        n_lm = 0
        for i in range(N_LM):
            if abs(iv_lm_z[i] - z_iv) > 1.5:
                continue
            n_lm += 1
            ex_x = int(round(nd2_lm_actual[i, 0] / DS))
            ex_y = int(round(nd2_lm_actual[i, 1] / DS))
            pr_x = int(round(nd2_lm_pred[i, 0] / DS))
            pr_y = int(round(nd2_lm_pred[i, 1] / DS))

            color_ev = (180, 0, 0)    # dark blue
            color_iv = (0, 0, 255)    # red

            cv2.drawMarker(ev_rgb, (ex_x, ex_y), color_ev, cv2.MARKER_CROSS, 16, 2)
            cv2.drawMarker(iv_rgb, (pr_x, pr_y), color_iv, cv2.MARKER_CROSS, 16, 2)
            cv2.drawMarker(ov_rgb, (ex_x, ex_y), color_ev, cv2.MARKER_CROSS, 16, 1)
            cv2.drawMarker(ov_rgb, (pr_x, pr_y), color_iv, cv2.MARKER_CROSS, 16, 1)
            cv2.line(ov_rgb, (ex_x, ex_y), (pr_x, pr_y), (255, 255, 255), 1)

        y0 = HEADER_H + z_iv * row_ht + GAP
        for ci, p in enumerate([ev_rgb, iv_rgb, ov_rgb]):
            x0 = GAP + ci * (out_w + GAP)
            sheet[y0:y0 + out_h, x0:x0 + out_w] = p
        cv2.putText(sheet, f'iv_z={z_iv} -> nd2_z={z_nd2} ({z_nd2_float:.1f})  ({n_lm} lm)',
                    (GAP + 4, y0 + out_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        rows_html.append(f"""
        <tr>
          <td class="label">iv_z={z_iv} → nd2_z={z_nd2} ({z_nd2_float:.1f})<br>{n_lm} lm</td>
          <td><img src="data:image/jpeg;base64,{to_b64(ev_rgb)}"></td>
          <td><img src="data:image/jpeg;base64,{to_b64(iv_rgb)}"></td>
          <td><img src="data:image/jpeg;base64,{to_b64(ov_rgb)}"></td>
        </tr>""")

    # PNG headers
    for ci, hdr in enumerate(['nd2 (blue=actual lm)', 'In-vivo warped affine+RBF (red=predicted)', 'Overlay']):
        cv2.putText(sheet, hdr, (GAP + ci * (out_w + GAP) + 4, HEADER_H - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(sheet,
                f'{tile}: 3D affine+RBF ({N_LM} lm) | mean={errors.mean():.1f}um | scales={S[0]:.2f},{S[1]:.2f},{S[2]:.2f}',
                (GAP + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    z_map = []
    for z_iv in range(nz_iv):
        c = np.array([z_iv, ny_iv/2, nx_iv/2])
        z_nd2_f = (M_fwd @ c + t_fwd)[0]
        z_map.append(f'{z_iv}->{z_nd2_f:.1f}')
    cv2.putText(sheet, f'z-map: {" ".join(z_map)}',
                (GAP + 4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

    out_png = f'{out_dir}/registration_{tile}.png'
    cv2.imwrite(out_png, sheet)
    print(f"  Saved PNG: {out_png}")

    # HTML
    z_map_str = ', '.join(z_map)
    html = f"""<!DOCTYPE html>
<html>
<head>
<title>{tile} — 3D Affine + RBF Registration</title>
<style>
  body {{ background: #111; color: #eee; font-family: monospace; margin: 20px; }}
  h1 {{ font-size: 18px; }}
  .info {{ font-size: 13px; color: #aaa; margin-bottom: 16px; line-height: 1.6; }}
  table {{ border-collapse: collapse; }}
  td {{ padding: 2px 4px; vertical-align: top; }}
  td img {{ display: block; width: 100%; }}
  th {{ font-size: 13px; color: #ccc; padding: 4px 8px; text-align: center; }}
  .label {{ font-size: 12px; color: #aaa; width: 100px; vertical-align: middle; text-align: right; padding-right: 8px; }}
  tr:hover {{ outline: 1px solid #555; }}
</style>
</head>
<body>
<h1>{tile}: In-vivo → nd2 (3D affine + RBF warp)</h1>
<div class="info">
  <b>Landmarks:</b> {N_LM} | <b>Affine error:</b> mean={errors.mean():.1f}µm, median={np.median(errors):.1f}µm, max={errors.max():.1f}µm<br>
  SVD scales: {S[0]:.3f}, {S[1]:.3f}, {S[2]:.3f} | z-map: {z_map_str}
</div>
<table>
  <tr>
    <th></th>
    <th>nd2 ex-vivo (blue=actual landmark)</th>
    <th>In-vivo warped affine+RBF (red=predicted)</th>
    <th>Overlay (green=exvivo, magenta=invivo)</th>
  </tr>
  {''.join(rows_html)}
</table>
</body>
</html>
"""
    out_html = f'{out_dir}/registration_{tile}.html'
    with open(out_html, 'w') as f:
        f.write(html)
    print(f"  Saved HTML: {out_html}")

    # Save transform
    np.savez(f'{out_dir}/transform_{tile}.npz',
             affine_3x4=A, src_um=src, dst_um=dst,
             errors=errors, predicted=predicted,
             nd2_z_gauss=np.array(nd2_z_vals),
             M_fwd=M_fwd, t_fwd=t_fwd, M_inv=M_inv, offset_inv=offset_inv)
    print(f"  Saved transform: {out_dir}/transform_{tile}.npz")

    summary.append((tile, N_LM, errors.mean(), np.median(errors), errors.max(), S))
    del nd2_slices

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print(f"{'Tile':>10} {'N_lm':>5} {'Mean':>8} {'Median':>8} {'Max':>8} {'S1':>6} {'S2':>6} {'S3':>6}")
print(f"{'='*70}")
for tile, n, mean, med, mx, s in summary:
    print(f"{tile:>10} {n:5d} {mean:8.1f} {med:8.1f} {mx:8.1f} {s[0]:6.2f} {s[1]:6.2f} {s[2]:6.2f}")
print(f"\nAll saved to {OUT_BASE}/")
