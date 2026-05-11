#!/usr/bin/env python3
"""
Row2_2 tile: 3D affine registration v2.
Fix: use Gaussian fit / intensity-weighted centroid for nd2 z instead of argmax.
Outputs both PNG and HTML contact sheets.
"""
import numpy as np
import cv2
import os
import base64
import tifffile
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

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
# Load landmarks — Gaussian fit for nd2 z
# ============================================================
print("Loading landmarks...")
d = np.load(f'{BASE}/registration_video/landmarks_nd2_native_{TILE}.npz')
ev_nd2 = d['ev_nd2']
pcd_iv = d['pcd_invivo_jy306']
N_LM = ev_nd2.shape[0]


def gauss(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def find_z_gaussian(intensities):
    """Fit Gaussian to intensity profile, return sub-pixel z. Fall back to centroid."""
    zs = np.arange(len(intensities), dtype=np.float64)
    vals = np.array(intensities, dtype=np.float64)

    # Subtract minimum to get positive signal
    vals = vals - vals.min()
    total = vals.sum()
    if total < 1e-6:
        return float(np.argmax(intensities))

    # Intensity-weighted centroid
    centroid = float(np.sum(zs * vals) / total)

    # Try Gaussian fit
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


nd2_z_argmax = []
nd2_z_gauss = []
nd2_z_centroid = []

print(f"\n  {'i':>3} {'iv_z':>5} {'argmax':>7} {'gauss':>7} {'centroid':>9}  intensity_profile")
for i in range(N_LM):
    x, y = ev_nd2[i, 0], ev_nd2[i, 1]
    c = int(round(np.clip(x, 10, nd2_h - 11)))
    r = int(round(np.clip(y, 10, nd2_h - 11)))
    intensities = [nd2_slices[z][r-10:r+10, c-10:c+10].mean() for z in range(12)]

    z_argmax = int(np.argmax(intensities))
    z_gauss = find_z_gaussian(intensities)

    vals = np.array(intensities)
    vals_pos = vals - vals.min()
    total = vals_pos.sum()
    z_centroid = float(np.sum(np.arange(12) * vals_pos) / total) if total > 0 else z_argmax

    nd2_z_argmax.append(z_argmax)
    nd2_z_gauss.append(z_gauss)
    nd2_z_centroid.append(z_centroid)

    prof_str = ' '.join([f'{v:.0f}' for v in intensities])
    print(f"  {i:3d} {pcd_iv[i,0]:5.0f} {z_argmax:7d} {z_gauss:7.2f} {z_centroid:9.2f}  [{prof_str}]")

# Use Gaussian z
nd2_z_vals = nd2_z_gauss

print(f"\n  Argmax z:   {sorted(set(nd2_z_argmax))}")
print(f"  Gaussian z: min={min(nd2_z_gauss):.2f} max={max(nd2_z_gauss):.2f} mean={np.mean(nd2_z_gauss):.2f}")
print(f"  Centroid z: min={min(nd2_z_centroid):.2f} max={max(nd2_z_centroid):.2f} mean={np.mean(nd2_z_centroid):.2f}")

# ============================================================
# 3D affine
# ============================================================
print("\nFitting 3D affine...")
src = np.column_stack([pcd_iv[:, 2] * IV_XY_UM, pcd_iv[:, 1] * IV_XY_UM, pcd_iv[:, 0] * IV_Z_UM])
dst = np.column_stack([ev_nd2[:, 0] * ND2_XY_UM, ev_nd2[:, 1] * ND2_XY_UM, np.array(nd2_z_vals) * ND2_Z_UM])

ones = np.ones((N_LM, 1))
src_h = np.hstack([src, ones])
A_T, _, _, _ = np.linalg.lstsq(src_h, dst, rcond=None)
A = A_T.T
predicted = src_h @ A_T
errors = np.sqrt(np.sum((predicted - dst) ** 2, axis=1))
Rm = A[:, :3]
_, S, _ = np.linalg.svd(Rm)
print(f"  mean={errors.mean():.1f}µm median={np.median(errors):.1f}µm max={errors.max():.1f}µm")
print(f"  Scales: {S[0]:.3f}, {S[1]:.3f}, {S[2]:.3f}")

# Compare with argmax version
dst_argmax = np.column_stack([ev_nd2[:, 0] * ND2_XY_UM, ev_nd2[:, 1] * ND2_XY_UM, np.array(nd2_z_argmax) * ND2_Z_UM])
A_T_old, _, _, _ = np.linalg.lstsq(src_h, dst_argmax, rcond=None)
pred_old = src_h @ A_T_old
err_old = np.sqrt(np.sum((pred_old - dst_argmax) ** 2, axis=1))
print(f"  (argmax for comparison: mean={err_old.mean():.1f}µm)")

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

nd2_lm_actual = ev_nd2[:, :2]
nd2_lm_pred = predicted[:, :2] / ND2_XY_UM
iv_lm_z = pcd_iv[:, 0]

# ============================================================
# Contact sheet (PNG + HTML)
# ============================================================
print("\nGenerating contact sheet...")

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
        pr_x = int(round(nd2_lm_pred[i, 0] / DS))
        pr_y = int(round(nd2_lm_pred[i, 1] / DS))

        cv2.drawMarker(ev_rgb, (ex_x, ex_y), (255, 255, 0), cv2.MARKER_CROSS, 16, 2)
        cv2.drawMarker(iv_rgb, (pr_x, pr_y), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.drawMarker(ov_rgb, (ex_x, ex_y), (255, 255, 0), cv2.MARKER_CROSS, 16, 1)
        cv2.drawMarker(ov_rgb, (pr_x, pr_y), (0, 255, 255), cv2.MARKER_CROSS, 16, 1)
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
      <td class="label">iv_z={z_iv} → nd2_z={z_nd2} ({z_nd2_float:.1f})<br>{n_lm} landmarks</td>
      <td><img src="data:image/jpeg;base64,{to_b64(ev_rgb)}"></td>
      <td><img src="data:image/jpeg;base64,{to_b64(iv_rgb)}"></td>
      <td><img src="data:image/jpeg;base64,{to_b64(ov_rgb)}"></td>
    </tr>""")

# PNG headers
for ci, hdr in enumerate(['nd2 (cyan=actual lm)', 'In-vivo warped (yellow=predicted)', 'Overlay']):
    cv2.putText(sheet, hdr, (GAP + ci * (out_w + GAP) + 4, HEADER_H - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
cv2.putText(sheet,
            f'{TILE} v2 (gauss z): 3D affine ({N_LM} lm) | mean={errors.mean():.1f}um | scales={S[0]:.2f},{S[1]:.2f},{S[2]:.2f}',
            (GAP + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

z_map = []
for z_iv in range(nz_iv):
    c = np.array([z_iv, ny_iv/2, nx_iv/2])
    z_nd2_f = (M_fwd @ c + t_fwd)[0]
    z_map.append(f'{z_iv}->{z_nd2_f:.1f}')
cv2.putText(sheet, f'z-map: {" ".join(z_map)}',
            (GAP + 4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

out_png = f'{OUT_DIR}/registration_{TILE}_3d_affine_v2.png'
cv2.imwrite(out_png, sheet)
print(f"\nSaved PNG: {out_png}")

# HTML
z_map_str = ', '.join(z_map)
html = f"""<!DOCTYPE html>
<html>
<head>
<title>{TILE} — 3D Affine v2 (Gaussian z)</title>
<style>
  body {{ background: #111; color: #eee; font-family: monospace; margin: 20px; }}
  h1 {{ font-size: 18px; }}
  .info {{ font-size: 13px; color: #aaa; margin-bottom: 16px; }}
  table {{ border-collapse: collapse; }}
  td {{ padding: 2px 4px; vertical-align: top; }}
  td img {{ display: block; width: 100%; }}
  th {{ font-size: 13px; color: #ccc; padding: 4px 8px; text-align: center; }}
  .label {{ font-size: 12px; color: #aaa; width: 100px; vertical-align: middle; text-align: right; padding-right: 8px; }}
  tr:hover {{ outline: 1px solid #555; }}
</style>
</head>
<body>
<h1>{TILE}: 3D Affine v2 (Gaussian z-fit) — 16 in-vivo z → 12 nd2 z</h1>
<div class="info">
  {N_LM} landmarks | mean error = {errors.mean():.1f}µm | median = {np.median(errors):.1f}µm | max = {errors.max():.1f}µm<br>
  SVD scales: {S[0]:.3f}, {S[1]:.3f}, {S[2]:.3f}<br>
  z-map: {z_map_str}<br>
  (v1 argmax: mean={err_old.mean():.1f}µm → v2 gaussian: mean={errors.mean():.1f}µm)
</div>
<table>
  <tr>
    <th></th>
    <th>nd2 ex-vivo (cyan = actual landmark)</th>
    <th>In-vivo warped (yellow = predicted)</th>
    <th>Overlay (green=exvivo, magenta=invivo)</th>
  </tr>
  {''.join(rows_html)}
</table>
</body>
</html>
"""

out_html = f'{OUT_DIR}/registration_{TILE}_3d_affine_v2.html'
with open(out_html, 'w') as f:
    f.write(html)
print(f"Saved HTML: {out_html}")

np.savez(f'{BASE}/registration_video/affine_3d_{TILE}_v2.npz',
         affine_3x4=A, src_um=src, dst_um=dst,
         errors=errors, predicted=predicted,
         nd2_z_gauss=np.array(nd2_z_gauss),
         nd2_z_argmax=np.array(nd2_z_argmax))
print(f"Affine saved: affine_3d_{TILE}_v2.npz")
