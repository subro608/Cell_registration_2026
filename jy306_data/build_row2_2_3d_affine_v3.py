#!/usr/bin/env python3
"""
Row2_2 tile: 3D affine v3 — Gaussian z + outlier removal + TPS refinement.
1. Gaussian z-fit for nd2 z (same as v2)
2. Fit 3D affine, remove top 5 outliers, refit
3. Apply thin-plate spline (TPS) on residuals for non-rigid correction
4. Output PNG + HTML + affine .npz
"""
import numpy as np
import cv2
import os
import base64
import tifffile
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.interpolate import RBFInterpolator

BASE = '/Users/neurolab/neuroinformatics/margaret'

IV_XY_UM = 0.6835
IV_Z_UM = 3.0
ND2_XY_UM = 0.645
ND2_Z_UM = 2.0

TILE = 'row2_2'
OUT_DIR = f'{BASE}/png_exports/registration_row2_2'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Load nd2 tile
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
nd2_h, nd2_w = nd2_slices.shape[1], nd2_slices.shape[2]
print(f"  nd2: {nd2_slices.shape}")

# ============================================================
# Load in-vivo
# ============================================================
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

# ============================================================
# Landmarks + Gaussian z
# ============================================================
print("Loading landmarks...")
d = np.load(f'{BASE}/registration_video/landmarks_nd2_native_{TILE}.npz')
ev_nd2 = d['ev_nd2']
pcd_iv = d['pcd_invivo_jy306']
N_LM = ev_nd2.shape[0]


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


nd2_z_vals = []
for i in range(N_LM):
    x, y = ev_nd2[i, 0], ev_nd2[i, 1]
    c = int(round(np.clip(x, 10, nd2_h - 11)))
    r = int(round(np.clip(y, 10, nd2_h - 11)))
    intensities = [nd2_slices[z][r-10:r+10, c-10:c+10].mean() for z in range(12)]
    nd2_z_vals.append(find_z_gaussian(intensities))

print(f"  {N_LM} landmarks, Gaussian z: {min(nd2_z_vals):.1f}-{max(nd2_z_vals):.1f}")

# ============================================================
# 3D affine + outlier removal
# ============================================================
print("\nFitting 3D affine (all landmarks)...")
src_all = np.column_stack([pcd_iv[:, 2] * IV_XY_UM, pcd_iv[:, 1] * IV_XY_UM, pcd_iv[:, 0] * IV_Z_UM])
dst_all = np.column_stack([ev_nd2[:, 0] * ND2_XY_UM, ev_nd2[:, 1] * ND2_XY_UM, np.array(nd2_z_vals) * ND2_Z_UM])

ones_all = np.ones((N_LM, 1))
src_h_all = np.hstack([src_all, ones_all])
A_T_init, _, _, _ = np.linalg.lstsq(src_h_all, dst_all, rcond=None)
pred_init = src_h_all @ A_T_init
errors_init = np.sqrt(np.sum((pred_init - dst_all) ** 2, axis=1))
print(f"  Initial: mean={errors_init.mean():.1f}µm, max={errors_init.max():.1f}µm")

# Remove worst outliers (iterative: remove worst, refit, repeat)
keep = np.arange(N_LM)
N_REMOVE = 5
for iteration in range(N_REMOVE):
    src_k = src_all[keep]
    dst_k = dst_all[keep]
    ones_k = np.ones((len(keep), 1))
    src_h_k = np.hstack([src_k, ones_k])
    A_T_k, _, _, _ = np.linalg.lstsq(src_h_k, dst_k, rcond=None)
    pred_k = src_h_k @ A_T_k
    errs_k = np.sqrt(np.sum((pred_k - dst_k) ** 2, axis=1))
    worst = np.argmax(errs_k)
    removed_idx = keep[worst]
    print(f"  Iter {iteration+1}: remove lm {removed_idx} (err={errs_k[worst]:.1f}µm), {len(keep)-1} remain")
    keep = np.delete(keep, worst)

# Final affine on remaining landmarks
src_clean = src_all[keep]
dst_clean = dst_all[keep]
N_clean = len(keep)
ones_clean = np.ones((N_clean, 1))
src_h_clean = np.hstack([src_clean, ones_clean])
A_T, _, _, _ = np.linalg.lstsq(src_h_clean, dst_clean, rcond=None)
A = A_T.T

pred_clean = src_h_clean @ A_T
errors_clean = np.sqrt(np.sum((pred_clean - dst_clean) ** 2, axis=1))
Rm = A[:, :3]
_, S, _ = np.linalg.svd(Rm)
print(f"\n  After outlier removal ({N_clean} lm): mean={errors_clean.mean():.1f}µm, median={np.median(errors_clean):.1f}µm, max={errors_clean.max():.1f}µm")
print(f"  Scales: {S[0]:.3f}, {S[1]:.3f}, {S[2]:.3f}")

# Predict ALL landmarks with clean affine (for visualization)
src_h_all_pred = np.hstack([src_all, np.ones((N_LM, 1))])
pred_all = src_h_all_pred @ A_T
errors_all = np.sqrt(np.sum((pred_all - dst_all) ** 2, axis=1))
print(f"  All {N_LM} lm with clean affine: mean={errors_all.mean():.1f}µm")

# ============================================================
# TPS / RBF refinement on residuals
# ============================================================
print("\nRBF refinement on affine residuals...")
# Residuals at clean landmark positions (in nd2 µm space)
residuals = dst_clean - pred_clean  # (N_clean, 3)

# Build RBF interpolator: given a point in predicted nd2 µm, return correction
rbf_x = RBFInterpolator(pred_clean, residuals[:, 0], kernel='thin_plate_spline', smoothing=1.0)
rbf_y = RBFInterpolator(pred_clean, residuals[:, 1], kernel='thin_plate_spline', smoothing=1.0)
rbf_z = RBFInterpolator(pred_clean, residuals[:, 2], kernel='thin_plate_spline', smoothing=1.0)

# Test on clean landmarks
corr_x = rbf_x(pred_clean)
corr_y = rbf_y(pred_clean)
corr_z = rbf_z(pred_clean)
refined_clean = pred_clean + np.column_stack([corr_x, corr_y, corr_z])
errors_refined = np.sqrt(np.sum((refined_clean - dst_clean) ** 2, axis=1))
print(f"  After RBF: mean={errors_refined.mean():.1f}µm, median={np.median(errors_refined):.1f}µm, max={errors_refined.max():.1f}µm")

# Predict ALL landmarks with affine + RBF
corr_all_x = rbf_x(pred_all)
corr_all_y = rbf_y(pred_all)
corr_all_z = rbf_z(pred_all)
refined_all = pred_all + np.column_stack([corr_all_x, corr_all_y, corr_all_z])
errors_all_refined = np.sqrt(np.sum((refined_all - dst_all) ** 2, axis=1))
print(f"  All {N_LM} lm with affine+RBF: mean={errors_all_refined.mean():.1f}µm")

# ============================================================
# Pixel-space transforms (affine part only for warp)
# ============================================================
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

# ============================================================
# Helpers
# ============================================================
def norm8(img):
    vals = img[img > 0]
    if len(vals) < 100:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(vals, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)

def to_b64(rgb_img):
    _, buf = cv2.imencode('.jpg', rgb_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('ascii')

DS = max(1, nd2_w // 600)
out_w, out_h = nd2_w // DS, nd2_h // DS

# Landmark positions for visualization
nd2_lm_actual = ev_nd2[:, :2]  # actual nd2 px
nd2_lm_pred_affine = pred_all[:, :2] / ND2_XY_UM  # affine-predicted nd2 px
nd2_lm_pred_refined = refined_all[:, :2] / ND2_XY_UM  # affine+RBF predicted nd2 px
iv_lm_z = pcd_iv[:, 0]
removed_set = set(range(N_LM)) - set(keep)

# ============================================================
# Contact sheet
# ============================================================
print("\nGenerating contact sheet...")

LABEL_H, GAP, COLS, HEADER_H = 30, 4, 3, 80
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

    M2d = np.array([
        [M_inv[2, 2], M_inv[2, 1], M_inv[2, 0] * z_nd2 + offset_inv[2]],
        [M_inv[1, 2], M_inv[1, 1], M_inv[1, 0] * z_nd2 + offset_inv[1]],
    ], dtype=np.float64)

    iv_warped = cv2.warpAffine(iv_vol[z_iv], M2d, (nd2_w, nd2_h),
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
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
        # Use refined (affine+RBF) prediction
        pr_x = int(round(nd2_lm_pred_refined[i, 0] / DS))
        pr_y = int(round(nd2_lm_pred_refined[i, 1] / DS))

        is_outlier = i in removed_set
        color_ev = (0, 0, 255) if is_outlier else (255, 255, 0)  # red for outlier
        color_iv = (0, 0, 255) if is_outlier else (0, 255, 255)

        cv2.drawMarker(ev_rgb, (ex_x, ex_y), color_ev, cv2.MARKER_CROSS, 16, 2)
        cv2.drawMarker(iv_rgb, (pr_x, pr_y), color_iv, cv2.MARKER_CROSS, 16, 2)
        cv2.drawMarker(ov_rgb, (ex_x, ex_y), color_ev, cv2.MARKER_CROSS, 16, 1)
        cv2.drawMarker(ov_rgb, (pr_x, pr_y), color_iv, cv2.MARKER_CROSS, 16, 1)
        line_color = (0, 0, 255) if is_outlier else (255, 255, 255)
        cv2.line(ov_rgb, (ex_x, ex_y), (pr_x, pr_y), line_color, 1)

    print(f"  iv_z={z_iv:2d} → nd2_z={z_nd2:2d} ({z_nd2_float:.1f}) {n_lm} lm")

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
for ci, hdr in enumerate(['nd2 (cyan=lm, red=outlier)', 'In-vivo warped (yellow=pred)', 'Overlay']):
    cv2.putText(sheet, hdr, (GAP + ci * (out_w + GAP) + 4, HEADER_H - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
cv2.putText(sheet,
            f'{TILE} v3: affine+RBF ({N_clean} lm) | affine={errors_clean.mean():.1f}um | +RBF={errors_refined.mean():.1f}um',
            (GAP + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(sheet,
            f'scales={S[0]:.2f},{S[1]:.2f},{S[2]:.2f} | removed {N_REMOVE} outliers (red)',
            (GAP + 4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

z_map = []
for z_iv in range(nz_iv):
    c = np.array([z_iv, ny_iv/2, nx_iv/2])
    z_nd2_f = (M_fwd @ c + t_fwd)[0]
    z_map.append(f'{z_iv}->{z_nd2_f:.1f}')
cv2.putText(sheet, f'z-map: {" ".join(z_map)}',
            (GAP + 4, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

out_png = f'{OUT_DIR}/registration_{TILE}_3d_affine_v3.png'
cv2.imwrite(out_png, sheet)
print(f"\nSaved PNG: {out_png}")

# HTML
z_map_str = ', '.join(z_map)
html = f"""<!DOCTYPE html>
<html>
<head>
<title>{TILE} — 3D Affine v3 (Gaussian z + outlier removal + RBF)</title>
<style>
  body {{ background: #111; color: #eee; font-family: monospace; margin: 20px; }}
  h1 {{ font-size: 18px; }}
  .info {{ font-size: 13px; color: #aaa; margin-bottom: 16px; line-height: 1.6; }}
  .improvement {{ color: #4f4; }}
  table {{ border-collapse: collapse; }}
  td {{ padding: 2px 4px; vertical-align: top; }}
  td img {{ display: block; width: 100%; }}
  th {{ font-size: 13px; color: #ccc; padding: 4px 8px; text-align: center; }}
  .label {{ font-size: 12px; color: #aaa; width: 100px; vertical-align: middle; text-align: right; padding-right: 8px; }}
  tr:hover {{ outline: 1px solid #555; }}
</style>
</head>
<body>
<h1>{TILE}: 3D Affine v3 — Gaussian z + Outlier Removal + RBF Refinement</h1>
<div class="info">
  <b>Pipeline:</b> Gaussian z-fit → 3D affine → remove {N_REMOVE} worst outliers → refit → RBF (thin-plate spline) on residuals<br>
  <b>Landmarks:</b> {N_clean}/{N_LM} (red = removed outliers)<br>
  <b>Affine error:</b> mean={errors_clean.mean():.1f}µm, median={np.median(errors_clean):.1f}µm, max={errors_clean.max():.1f}µm<br>
  <b class="improvement">After RBF:</b> mean={errors_refined.mean():.1f}µm, median={np.median(errors_refined):.1f}µm, max={errors_refined.max():.1f}µm<br>
  <b>Improvement:</b> v1 argmax 11.2µm → v2 gaussian 7.8µm → <span class="improvement">v3 affine+RBF {errors_refined.mean():.1f}µm</span><br>
  SVD scales: {S[0]:.3f}, {S[1]:.3f}, {S[2]:.3f}<br>
  z-map: {z_map_str}
</div>
<table>
  <tr>
    <th></th>
    <th>nd2 ex-vivo (cyan=lm, red=outlier)</th>
    <th>In-vivo warped (yellow=predicted)</th>
    <th>Overlay (green=exvivo, magenta=invivo)</th>
  </tr>
  {''.join(rows_html)}
</table>
</body>
</html>
"""

out_html = f'{OUT_DIR}/registration_{TILE}_3d_affine_v3.html'
with open(out_html, 'w') as f:
    f.write(html)
print(f"Saved HTML: {out_html}")

# Save transforms
np.savez(f'{OUT_DIR}/affine_3d_{TILE}_v3.npz',
         affine_3x4=A, src_um=src_all, dst_um=dst_all,
         keep_indices=keep, removed_indices=np.array(sorted(removed_set)),
         errors_affine=errors_clean, errors_refined=errors_refined,
         errors_all_affine=errors_all, errors_all_refined=errors_all_refined,
         predicted_affine=pred_all, predicted_refined=refined_all,
         nd2_z_gauss=np.array(nd2_z_vals))
print(f"Saved: {OUT_DIR}/affine_3d_{TILE}_v3.npz")
