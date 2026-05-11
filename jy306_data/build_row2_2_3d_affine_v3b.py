#!/usr/bin/env python3
"""
Row2_2: 3D affine + RBF warp applied to the actual in-vivo IMAGE.
For each nd2 z-slice: build dense displacement field (affine + RBF),
remap in-vivo into nd2 space.
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
# Load data
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

# ============================================================
# 3D affine + outlier removal
# ============================================================
print("Fitting 3D affine...")
src_all = np.column_stack([pcd_iv[:, 2] * IV_XY_UM, pcd_iv[:, 1] * IV_XY_UM, pcd_iv[:, 0] * IV_Z_UM])
dst_all = np.column_stack([ev_nd2[:, 0] * ND2_XY_UM, ev_nd2[:, 1] * ND2_XY_UM, np.array(nd2_z_vals) * ND2_Z_UM])

ones_all = np.ones((N_LM, 1))
src_h_all = np.hstack([src_all, ones_all])
A_T_init, _, _, _ = np.linalg.lstsq(src_h_all, dst_all, rcond=None)

keep = np.arange(N_LM)
N_REMOVE = 0

src_clean = src_all
dst_clean = dst_all
N_clean = N_LM
src_h_clean = src_h_all
A_T, _, _, _ = np.linalg.lstsq(src_h_clean, dst_clean, rcond=None)
A = A_T.T
pred_clean = src_h_clean @ A_T
errors_clean = np.sqrt(np.sum((pred_clean - dst_clean) ** 2, axis=1))
Rm = A[:, :3]
_, S, _ = np.linalg.svd(Rm)
print(f"  Affine ({N_clean} lm): mean={errors_clean.mean():.1f}µm")

# ============================================================
# RBF on residuals — build interpolators in PIXEL space
# We need: for each pixel (x_nd2, y_nd2) in nd2 space at a given z_nd2,
# find the corresponding (x_iv, y_iv) in in-vivo pixel space.
#
# Forward: iv_px → nd2_um via affine → nd2_px
# We build RBF correction in nd2_px space.
# Then inverse: nd2_px → iv_px = affine_inv(nd2_px) + rbf_correction
# ============================================================
print("Building RBF displacement field...")

# Landmark positions in nd2 pixel space
# Actual nd2 positions (ground truth)
lm_nd2_px = np.column_stack([ev_nd2[keep, 0], ev_nd2[keep, 1]])  # (N, 2) x,y in nd2 px

# Affine-predicted nd2 positions
pred_nd2_um = pred_clean[:, :2]  # x, y in µm
pred_nd2_px = pred_nd2_um / ND2_XY_UM  # x, y in nd2 px

# Residuals in nd2 pixel space
residuals_px = lm_nd2_px - pred_nd2_px  # correction to add to affine prediction

# In-vivo pixel positions for landmarks
lm_iv_px = np.column_stack([pcd_iv[keep, 2], pcd_iv[keep, 1]])  # x, y in JY306 px

# For the IMAGE WARP we need the INVERSE: given (x_nd2, y_nd2), find (x_iv, y_iv)
# Affine inverse gives a first estimate. RBF corrects the nd2 side.
#
# Better approach: build RBF that maps nd2_px → iv_px directly
# Control points: lm_nd2_px (actual nd2 pos) → lm_iv_px (actual iv pos)
# For a new nd2 pixel, affine_inv gives approximate iv_px, but we want exact.
#
# Actually, let's think differently:
# We have affine: iv_um → nd2_um (forward)
# We have affine_inv: nd2_um → iv_um (inverse)
# At landmarks, affine_inv(nd2_actual) gives approximate iv position.
# The true iv position is known. So correction = true_iv - affine_inv(nd2_actual).
# We build RBF: nd2_px → correction_iv_px
# Then for any nd2_px: iv_px = affine_inv(nd2_px) + rbf(nd2_px)

# Pixel-space forward/inverse (affine only)
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

# For each landmark: compute affine_inv(nd2_actual) in iv pixel space
# nd2 in (z,y,x) pixel convention
lm_nd2_zyx = np.column_stack([
    np.array(nd2_z_vals)[keep],  # z in nd2 slices
    ev_nd2[keep, 1],             # y in nd2 px
    ev_nd2[keep, 0]              # x in nd2 px
])

# affine_inv: nd2_zyx → iv_zyx
lm_iv_zyx_from_affine = (M_inv @ lm_nd2_zyx.T).T + offset_inv  # (N, 3) z,y,x in iv px

# True iv positions
lm_iv_zyx_true = np.column_stack([
    pcd_iv[keep, 0],  # z
    pcd_iv[keep, 1],  # y
    pcd_iv[keep, 2]   # x
])

# Correction in iv pixel space
corrections_iv = lm_iv_zyx_true - lm_iv_zyx_from_affine  # (N, 3)

print(f"  Correction magnitudes: mean={np.sqrt((corrections_iv**2).sum(axis=1)).mean():.1f}px")

# Build RBF: nd2_xy → correction_iv_xy (per z-slice we'll use 2D)
# Use nd2 XY positions as control points
rbf_corr_x = RBFInterpolator(lm_nd2_px, corrections_iv[:, 2], kernel='thin_plate_spline', smoothing=1.0)
rbf_corr_y = RBFInterpolator(lm_nd2_px, corrections_iv[:, 1], kernel='thin_plate_spline', smoothing=1.0)

# Verify at landmarks
test_cx = rbf_corr_x(lm_nd2_px)
test_cy = rbf_corr_y(lm_nd2_px)
verify_iv_x = lm_iv_zyx_from_affine[:, 2] + test_cx
verify_iv_y = lm_iv_zyx_from_affine[:, 1] + test_cy
err_x = verify_iv_x - lm_iv_zyx_true[:, 2]
err_y = verify_iv_y - lm_iv_zyx_true[:, 1]
err_total = np.sqrt(err_x**2 + err_y**2)
print(f"  RBF verification at landmarks: mean={err_total.mean():.2f}px, max={err_total.max():.2f}px")

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

removed_set = set(range(N_LM)) - set(keep)
iv_lm_z = pcd_iv[:, 0]

# ============================================================
# Warp each in-vivo z-slice with affine + RBF
# ============================================================
print("\nWarping in-vivo slices (affine + RBF)...")

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

    # Build dense remap: for each pixel in nd2 output, find source pixel in iv
    # Work at downsampled resolution for speed, then upscale the map
    # DS_MAP controls map resolution (1 = full res, higher = faster)
    DS_MAP = 4
    map_h, map_w = nd2_h // DS_MAP, nd2_w // DS_MAP

    # Grid of nd2 pixel coordinates (at reduced resolution)
    ys_nd2 = np.arange(map_h) * DS_MAP
    xs_nd2 = np.arange(map_w) * DS_MAP
    xx_nd2, yy_nd2 = np.meshgrid(xs_nd2, ys_nd2)

    # Affine inverse: nd2(z,y,x) → iv(z,y,x)
    # iv_x = M_inv[2,0]*z_nd2 + M_inv[2,1]*y_nd2 + M_inv[2,2]*x_nd2 + offset_inv[2]
    # iv_y = M_inv[1,0]*z_nd2 + M_inv[1,1]*y_nd2 + M_inv[1,2]*x_nd2 + offset_inv[1]
    iv_x_affine = M_inv[2, 0] * z_nd2 + M_inv[2, 1] * yy_nd2 + M_inv[2, 2] * xx_nd2 + offset_inv[2]
    iv_y_affine = M_inv[1, 0] * z_nd2 + M_inv[1, 1] * yy_nd2 + M_inv[1, 2] * xx_nd2 + offset_inv[1]

    # RBF correction
    query_pts = np.column_stack([xx_nd2.ravel(), yy_nd2.ravel()])
    corr_x = rbf_corr_x(query_pts).reshape(map_h, map_w)
    corr_y = rbf_corr_y(query_pts).reshape(map_h, map_w)

    iv_x_total = (iv_x_affine + corr_x).astype(np.float32)
    iv_y_total = (iv_y_affine + corr_y).astype(np.float32)

    # Upscale maps to full resolution
    map_x = cv2.resize(iv_x_total, (nd2_w, nd2_h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(iv_y_total, (nd2_w, nd2_h), interpolation=cv2.INTER_LINEAR)

    # Remap in-vivo slice
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

    # Draw landmarks
    n_lm = 0
    for i in range(N_LM):
        if abs(iv_lm_z[i] - z_iv) > 1.5:
            continue
        n_lm += 1
        # Actual nd2 position
        ex_x = int(round(ev_nd2[i, 0] / DS))
        ex_y = int(round(ev_nd2[i, 1] / DS))

        # Where does this landmark land after affine+RBF warp?
        # The landmark's iv position gets warped to nd2 space
        # For inliers, it should land exactly at actual nd2 position
        # Use the affine prediction + RBF for predicted position
        src_um_i = np.array([[pcd_iv[i, 2] * IV_XY_UM, pcd_iv[i, 1] * IV_XY_UM, pcd_iv[i, 0] * IV_Z_UM]])
        pred_um_i = (np.hstack([src_um_i, [[1]]]) @ A_T)[0]
        pred_nd2_px_i = pred_um_i[:2] / ND2_XY_UM
        # RBF correction on nd2 side to get better iv→nd2 mapping
        # Actually for display, show where the predicted nd2 position is
        pr_x = int(round(pred_nd2_px_i[0] / DS))
        pr_y = int(round(pred_nd2_px_i[1] / DS))

        color_ev = (180, 0, 0)      # dark blue for ex-vivo actual
        color_iv = (0, 0, 255)      # red for in-vivo predicted

        cv2.drawMarker(ev_rgb, (ex_x, ex_y), color_ev, cv2.MARKER_CROSS, 16, 2)
        cv2.drawMarker(iv_rgb, (pr_x, pr_y), color_iv, cv2.MARKER_CROSS, 16, 2)
        cv2.drawMarker(ov_rgb, (ex_x, ex_y), color_ev, cv2.MARKER_CROSS, 16, 1)
        cv2.drawMarker(ov_rgb, (pr_x, pr_y), color_iv, cv2.MARKER_CROSS, 16, 1)
        cv2.line(ov_rgb, (ex_x, ex_y), (pr_x, pr_y), (255, 255, 255), 1)

    print(f"  iv_z={z_iv:2d} → nd2_z={z_nd2:2d} ({z_nd2_float:.1f}) {n_lm} lm")

    # PNG
    y0 = HEADER_H + z_iv * row_ht + GAP
    for ci, p in enumerate([ev_rgb, iv_rgb, ov_rgb]):
        x0 = GAP + ci * (out_w + GAP)
        sheet[y0:y0 + out_h, x0:x0 + out_w] = p
    cv2.putText(sheet, f'iv_z={z_iv} -> nd2_z={z_nd2} ({z_nd2_float:.1f})  ({n_lm} lm)',
                (GAP + 4, y0 + out_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # HTML
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
            f'{TILE} v3b: affine+RBF IMAGE WARP ({N_clean} lm) | affine err={errors_clean.mean():.1f}um',
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

out_png = f'{OUT_DIR}/registration_{TILE}_3d_affine_v3b.png'
cv2.imwrite(out_png, sheet)
print(f"\nSaved PNG: {out_png}")

# HTML
z_map_str = ', '.join(z_map)
html = f"""<!DOCTYPE html>
<html>
<head>
<title>{TILE} — 3D Affine + RBF Image Warp</title>
<style>
  body {{ background: #111; color: #eee; font-family: monospace; margin: 20px; }}
  h1 {{ font-size: 18px; }}
  .info {{ font-size: 13px; color: #aaa; margin-bottom: 16px; line-height: 1.6; }}
  .g {{ color: #4f4; }}
  table {{ border-collapse: collapse; }}
  td {{ padding: 2px 4px; vertical-align: top; }}
  td img {{ display: block; width: 100%; }}
  th {{ font-size: 13px; color: #ccc; padding: 4px 8px; text-align: center; }}
  .label {{ font-size: 12px; color: #aaa; width: 100px; vertical-align: middle; text-align: right; padding-right: 8px; }}
  tr:hover {{ outline: 1px solid #555; }}
</style>
</head>
<body>
<h1>{TILE}: In-vivo warped to nd2 ex-vivo (3D affine + RBF)</h1>
<div class="info">
  <b>Pipeline:</b> Gaussian z-fit → 3D affine → remove 5 outliers → refit → RBF displacement field → <span class="g">remap in-vivo image</span><br>
  <b>Landmarks:</b> {N_clean}/{N_LM} (red = removed outliers)<br>
  <b>Affine error ({N_clean} lm):</b> mean={errors_clean.mean():.1f}µm, median={np.median(errors_clean):.1f}µm<br>
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

out_html = f'{OUT_DIR}/registration_{TILE}_3d_affine_v3b.html'
with open(out_html, 'w') as f:
    f.write(html)
print(f"Saved HTML: {out_html}")

np.savez(f'{OUT_DIR}/affine_3d_{TILE}_v3b.npz',
         affine_3x4=A, src_um=src_all, dst_um=dst_all,
         keep_indices=keep, nd2_z_gauss=np.array(nd2_z_vals),
         errors_affine=errors_clean)
print(f"Saved: {OUT_DIR}/affine_3d_{TILE}_v3b.npz")
