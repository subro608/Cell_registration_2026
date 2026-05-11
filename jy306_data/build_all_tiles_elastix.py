#!/usr/bin/env python3
"""
All tiles: 3D affine + Elastix B-spline deformable warp of in-vivo JY306 → each nd2 tile.
For each tile:
  1. Gaussian z-fit, 3D affine (all landmarks)
  2. Per z-slice: affine warp → elastix B-spline refinement (MI)
  3. Save PNG + HTML + transform .npz
"""
import numpy as np
import cv2
import os
import base64
import json
import glob
import tifffile
import SimpleITK as sitk
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

BASE = '/Users/neurolab/neuroinformatics/margaret'

IV_XY_UM = 0.6835
IV_Z_UM = 3.0
ND2_XY_UM = 0.645
ND2_Z_UM = 2.0

OUT_BASE = f'{BASE}/png_exports/registration_per_tile_elastix'
os.makedirs(OUT_BASE, exist_ok=True)

# ============================================================
# Load shared data
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


def norm_f(img):
    vals = img[img > 0]
    if len(vals) < 100:
        return img.copy()
    lo, hi = np.percentile(vals, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.float32)


def to_b64(rgb_img):
    _, buf = cv2.imencode('.jpg', rgb_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('ascii')


def ncc(a, b):
    mask = (a > 0) & (b > 0)
    if mask.sum() < 100:
        return -1.0
    am = a[mask].astype(np.float64)
    bm = b[mask].astype(np.float64)
    am -= am.mean()
    bm -= bm.mean()
    d = np.sqrt((am**2).sum() * (bm**2).sum())
    return float((am * bm).sum() / d) if d > 1e-10 else -1.0


def run_elastix(fixed_np, moving_np):
    fixed_sitk = sitk.GetImageFromArray(norm_f(fixed_np))
    moving_sitk = sitk.GetImageFromArray(norm_f(moving_np))
    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(fixed_sitk)
    elastix.SetMovingImage(moving_sitk)
    elastix.SetLogToConsole(False)
    pm = sitk.GetDefaultParameterMap('bspline')
    pm['Metric'] = ['AdvancedMattesMutualInformation']
    pm['NumberOfResolutions'] = ['3']
    pm['MaximumNumberOfIterations'] = ['500']
    pm['FinalGridSpacingInPhysicalUnits'] = ['50']
    pm['NumberOfSpatialSamples'] = ['4000']
    pm['GridSpacingSchedule'] = ['4.0', '2.0', '1.0']
    pm['ImagePyramidSchedule'] = ['8', '8', '4', '4', '2', '2']
    elastix.SetParameterMap(pm)
    try:
        elastix.Execute()
        tp = elastix.GetTransformParameterMap()
        transformix = sitk.TransformixImageFilter()
        transformix.SetMovingImage(sitk.GetImageFromArray(moving_np))
        transformix.SetTransformParameterMap(tp)
        transformix.SetLogToConsole(False)
        transformix.Execute()
        return sitk.GetArrayFromImage(transformix.GetResultImage())
    except Exception as e:
        return moving_np


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

    # Load nd2 slices
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
        print(f"  SKIP: only {N_LM} landmarks")
        continue

    # Gaussian z
    nd2_z_vals = []
    for i in range(N_LM):
        x, y = ev_nd2[i, 0], ev_nd2[i, 1]
        c = int(round(np.clip(x, 10, nd2_h - 11)))
        r = int(round(np.clip(y, 10, nd2_h - 11)))
        intensities = [nd2_slices[z][r-10:r+10, c-10:c+10].mean() for z in range(12)]
        nd2_z_vals.append(find_z_gaussian(intensities))

    # 3D affine
    src = np.column_stack([pcd_iv[:, 2] * IV_XY_UM, pcd_iv[:, 1] * IV_XY_UM, pcd_iv[:, 0] * IV_Z_UM])
    dst = np.column_stack([ev_nd2[:, 0] * ND2_XY_UM, ev_nd2[:, 1] * ND2_XY_UM, np.array(nd2_z_vals) * ND2_Z_UM])
    src_h = np.hstack([src, np.ones((N_LM, 1))])
    A_T, _, _, _ = np.linalg.lstsq(src_h, dst, rcond=None)
    A = A_T.T
    predicted = src_h @ A_T
    errors = np.sqrt(np.sum((predicted - dst) ** 2, axis=1))
    Rm = A[:, :3]
    _, S, _ = np.linalg.svd(Rm)
    print(f"  {N_LM} lm | affine: mean={errors.mean():.1f}µm | scales={S[0]:.2f},{S[1]:.2f},{S[2]:.2f}")

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

    DS = max(1, nd2_w // 600)
    out_w, out_h = nd2_w // DS, nd2_h // DS
    iv_lm_z = pcd_iv[:, 0]
    nd2_lm_actual = ev_nd2[:, :2]
    nd2_lm_pred = predicted[:, :2] / ND2_XY_UM

    # Contact sheet: 3 cols (nd2, affine overlay, elastix overlay)
    LABEL_H, GAP, COLS, HEADER_H = 30, 4, 3, 60
    row_ht = out_h + LABEL_H + GAP
    sheet_w = COLS * out_w + (COLS + 1) * GAP
    sheet_h = HEADER_H + nz_iv * row_ht + GAP
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    rows_html = []
    ncc_results = []

    for z_iv in range(nz_iv):
        center_iv = np.array([z_iv, ny_iv / 2, nx_iv / 2])
        center_nd2 = M_fwd @ center_iv + t_fwd
        z_nd2_float = center_nd2[0]
        z_nd2 = int(round(np.clip(z_nd2_float, 0, 11)))

        nd2_sl = nd2_slices[z_nd2]

        # Affine warp
        M2d = np.array([
            [M_inv[2, 2], M_inv[2, 1], M_inv[2, 0] * z_nd2 + offset_inv[2]],
            [M_inv[1, 2], M_inv[1, 1], M_inv[1, 0] * z_nd2 + offset_inv[1]],
        ], dtype=np.float64)
        iv_affine = cv2.warpAffine(iv_vol[z_iv], M2d, (nd2_w, nd2_h),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)

        # Elastix
        print(f"    iv_z={z_iv:2d} → nd2_z={z_nd2:2d}", end="", flush=True)
        iv_elastix = run_elastix(nd2_sl, iv_affine)

        ncc_af = ncc(nd2_sl, iv_affine)
        ncc_el = ncc(nd2_sl, iv_elastix)
        ncc_results.append((z_iv, z_nd2, ncc_af, ncc_el))
        print(f"  NCC: {ncc_af:.3f} → {ncc_el:.3f}")

        ev_d = cv2.resize(nd2_sl, (out_w, out_h), interpolation=cv2.INTER_AREA)
        af_d = cv2.resize(iv_affine, (out_w, out_h), interpolation=cv2.INTER_AREA)
        el_d = cv2.resize(iv_elastix, (out_w, out_h), interpolation=cv2.INTER_AREA)

        ev_rgb = cv2.cvtColor(norm8(ev_d), cv2.COLOR_GRAY2BGR)

        def make_ov(ev, iv):
            ov = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            ov[:, :, 1] = norm8(ev)
            ov[:, :, 0] = norm8(iv)
            ov[:, :, 2] = norm8(iv)
            return ov

        ov_af = make_ov(ev_d, af_d)
        ov_el = make_ov(ev_d, el_d)

        n_lm = 0
        for i in range(N_LM):
            if abs(iv_lm_z[i] - z_iv) > 1.5:
                continue
            n_lm += 1
            ex_x = int(round(nd2_lm_actual[i, 0] / DS))
            ex_y = int(round(nd2_lm_actual[i, 1] / DS))
            pr_x = int(round(nd2_lm_pred[i, 0] / DS))
            pr_y = int(round(nd2_lm_pred[i, 1] / DS))
            cv2.drawMarker(ev_rgb, (ex_x, ex_y), (180, 0, 0), cv2.MARKER_CROSS, 16, 2)
            for ov in [ov_af, ov_el]:
                cv2.drawMarker(ov, (ex_x, ex_y), (180, 0, 0), cv2.MARKER_CROSS, 16, 1)
                cv2.drawMarker(ov, (pr_x, pr_y), (0, 0, 255), cv2.MARKER_CROSS, 16, 1)
                cv2.line(ov, (ex_x, ex_y), (pr_x, pr_y), (255, 255, 255), 1)

        y0 = HEADER_H + z_iv * row_ht + GAP
        for ci, p in enumerate([ev_rgb, ov_af, ov_el]):
            x0 = GAP + ci * (out_w + GAP)
            sheet[y0:y0 + out_h, x0:x0 + out_w] = p
        cv2.putText(sheet, f'iv_z={z_iv}->nd2_z={z_nd2} NCC:{ncc_af:.3f}->{ncc_el:.3f} ({n_lm}lm)',
                    (GAP + 4, y0 + out_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        rows_html.append(f"""
        <tr>
          <td class="label">iv_z={z_iv} → nd2_z={z_nd2}<br>{n_lm} lm<br>
            af={ncc_af:.3f}<br>el={ncc_el:.3f}</td>
          <td><img src="data:image/jpeg;base64,{to_b64(ev_rgb)}"></td>
          <td><img src="data:image/jpeg;base64,{to_b64(ov_af)}"></td>
          <td><img src="data:image/jpeg;base64,{to_b64(ov_el)}"></td>
        </tr>""")

    # Mean NCCs
    mean_ncc_af = np.mean([r[2] for r in ncc_results])
    mean_ncc_el = np.mean([r[3] for r in ncc_results])

    for ci, hdr in enumerate(['nd2 ex-vivo', 'Affine overlay', 'Affine + Elastix overlay']):
        cv2.putText(sheet, hdr, (GAP + ci * (out_w + GAP) + 4, HEADER_H - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(sheet,
                f'{tile}: {N_LM}lm | affine={errors.mean():.1f}um | NCC: af={mean_ncc_af:.3f} el={mean_ncc_el:.3f}',
                (GAP + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    out_png = f'{out_dir}/registration_{tile}.png'
    cv2.imwrite(out_png, sheet)
    print(f"  Saved PNG: {out_png}")

    # HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
<title>{tile} — Affine + Elastix Registration</title>
<style>
  body {{ background: #111; color: #eee; font-family: monospace; margin: 20px; }}
  h1 {{ font-size: 18px; }}
  .info {{ font-size: 13px; color: #aaa; margin-bottom: 16px; line-height: 1.6; }}
  table {{ border-collapse: collapse; }}
  td {{ padding: 2px 4px; vertical-align: top; }}
  td img {{ display: block; width: 100%; }}
  th {{ font-size: 13px; color: #ccc; padding: 4px 8px; text-align: center; }}
  .label {{ font-size: 11px; color: #aaa; width: 80px; vertical-align: middle; text-align: right; padding-right: 8px; }}
  tr:hover {{ outline: 1px solid #555; }}
</style>
</head>
<body>
<h1>{tile}: 3D Affine + Elastix B-spline</h1>
<div class="info">
  {N_LM} landmarks | Affine error: mean={errors.mean():.1f}µm | scales: {S[0]:.3f}, {S[1]:.3f}, {S[2]:.3f}<br>
  Mean NCC: affine={mean_ncc_af:.3f} → elastix={mean_ncc_el:.3f}<br>
  Green = nd2 ex-vivo, Magenta = in-vivo | Blue = actual lm, Red = predicted
</div>
<table>
  <tr><th></th><th>nd2 ex-vivo</th><th>Affine overlay</th><th>Affine + Elastix overlay</th></tr>
  {''.join(rows_html)}
</table>
</body>
</html>
"""
    out_html = f'{out_dir}/registration_{tile}.html'
    with open(out_html, 'w') as f:
        f.write(html)
    print(f"  Saved HTML: {out_html}")

    np.savez(f'{out_dir}/transform_{tile}.npz',
             affine_3x4=A, M_fwd=M_fwd, t_fwd=t_fwd, M_inv=M_inv, offset_inv=offset_inv,
             errors=errors, nd2_z_gauss=np.array(nd2_z_vals),
             ncc_results=np.array([(r[0], r[1], r[2], r[3]) for r in ncc_results]))
    print(f"  Saved transform: {out_dir}/transform_{tile}.npz")

    summary.append((tile, N_LM, errors.mean(), mean_ncc_af, mean_ncc_el))
    del nd2_slices

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print(f"{'Tile':>10} {'N_lm':>5} {'Affine_err':>10} {'NCC_af':>8} {'NCC_el':>8} {'Improve':>8}")
print(f"{'='*70}")
for tile, n, err, ncc_a, ncc_e in summary:
    imp = (ncc_e - ncc_a) / max(ncc_a, 0.001) * 100
    print(f"{tile:>10} {n:5d} {err:10.1f} {ncc_a:8.3f} {ncc_e:8.3f} {imp:7.0f}%")
print(f"\nAll saved to {OUT_BASE}/")
