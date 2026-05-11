#!/usr/bin/env python3
"""
All tiles: 3D affine + Elastix B-spline.
3 columns: ex-vivo nd2 | in-vivo warped (elastix) | overlay
Save PNG + HTML + transform per tile.
"""
import numpy as np
import cv2
import os
import base64
import json
import glob
import tempfile
import tifffile
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from collections import defaultdict

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

print("Median filter bg subtraction (for elastix only)...")
iv_vol_filt = np.zeros_like(iv_vol_raw)
for z in range(nz_iv):
    bg = median_filter(iv_vol_raw[z], size=15)
    iv_vol_filt[z] = np.clip(iv_vol_raw[z] - bg, 0, None)

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


def parse_elastix_log(log_dir):
    """Parse elastix iteration info files from output directory."""
    iterations = []
    for res in range(10):
        log_file = os.path.join(log_dir, f'IterationInfo.0.R{res}.txt')
        if not os.path.exists(log_file):
            continue
        with open(log_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        it = int(parts[0])
                        metric = float(parts[1])
                        iterations.append({'res': res, 'iter': it, 'metric': metric})
                    except ValueError:
                        pass
    return iterations


def run_elastix(fixed_np, moving_filt, moving_raw=None,
                 fixed_points=None, moving_points=None,
                 save_tp_path=None, save_log_dir=None):
    """Run elastix with dual metric (MI + landmark correspondences).
    Returns (warped_raw, tp, iter_log) tuple. Saves tp to disk if save_tp_path given."""
    if moving_raw is None:
        moving_raw = moving_filt
    fixed_sitk = sitk.GetImageFromArray(norm_f(fixed_np))
    moving_sitk = sitk.GetImageFromArray(norm_f(moving_filt))
    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(fixed_sitk)
    elastix.SetMovingImage(moving_sitk)
    elastix.SetLogToConsole(False)

    # Use temp dir for iteration logs
    log_dir = tempfile.mkdtemp(prefix='elastix_')
    elastix.SetOutputDirectory(log_dir)

    pm = sitk.GetDefaultParameterMap('bspline')
    pm['Metric'] = ['AdvancedMattesMutualInformation']
    pm['NumberOfResolutions'] = ['3']
    pm['MaximumNumberOfIterations'] = ['500']
    pm['FinalGridSpacingInPhysicalUnits'] = ['50']
    pm['NumberOfSpatialSamples'] = ['4000']
    pm['GridSpacingSchedule'] = ['4.0', '2.0', '1.0']
    pm['ImagePyramidSchedule'] = ['8', '8', '4', '4', '2', '2']
    pm['WriteIterationInfo'] = ['true']

    pts_files = []
    if fixed_points and moving_points and len(fixed_points) > 0:
        pm['Registration'] = ['MultiMetricMultiResolutionRegistration']
        pm['Metric'] = ['AdvancedMattesMutualInformation',
                         'CorrespondingPointsEuclideanDistanceMetric']
        pm['Metric0Weight'] = ['1.0']
        pm['Metric1Weight'] = ['10.0']
        fp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        mp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        fp.write('point\n'); fp.write(f'{len(fixed_points)}\n')
        for x, y in fixed_points:
            fp.write(f'{x:.2f} {y:.2f}\n')
        fp.close()
        mp.write('point\n'); mp.write(f'{len(moving_points)}\n')
        for x, y in moving_points:
            mp.write(f'{x:.2f} {y:.2f}\n')
        mp.close()
        elastix.SetFixedPointSetFileName(fp.name)
        elastix.SetMovingPointSetFileName(mp.name)
        pts_files = [fp.name, mp.name]

    elastix.SetParameterMap(pm)
    iter_log = []
    try:
        elastix.Execute()
        tp = elastix.GetTransformParameterMap()
        if save_tp_path:
            sitk.WriteParameterFile(tp[0], save_tp_path)
        # Parse iteration logs
        iter_log = parse_elastix_log(log_dir)
        # Copy logs if save dir given
        if save_log_dir:
            os.makedirs(save_log_dir, exist_ok=True)
            for fname in os.listdir(log_dir):
                if fname.startswith('IterationInfo'):
                    import shutil
                    shutil.copy2(os.path.join(log_dir, fname),
                                 os.path.join(save_log_dir, fname))
        transformix = sitk.TransformixImageFilter()
        transformix.SetMovingImage(sitk.GetImageFromArray(moving_raw))
        transformix.SetTransformParameterMap(tp)
        transformix.SetLogToConsole(False)
        transformix.Execute()
        return sitk.GetArrayFromImage(transformix.GetResultImage()), tp, iter_log
    except Exception:
        return moving_raw, None, iter_log
    finally:
        for f in pts_files:
            try: os.unlink(f)
            except OSError: pass
        # Clean up temp log dir
        import shutil
        shutil.rmtree(log_dir, ignore_errors=True)


# ============================================================
# Process each tile
# ============================================================
summary = []
ONLY_TILES = {'row3_6'}  # Set to None to process all tiles
POOL_LM_BY_ZIV = False  # Pool landmarks across z_nd2 for same z_iv

for tile in sorted(tile_lm_files.keys()):
    if ONLY_TILES and tile not in ONLY_TILES:
        continue
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

    # Save directory for transforms
    tile_save_dir = f'{out_dir}/transforms'
    os.makedirs(tile_save_dir, exist_ok=True)

    # Group landmarks by known (z_iv, z_nd2) pairs — no 3D affine for z mapping
    z_pair_to_lm = defaultdict(list)
    for i in range(N_LM):
        z_iv_i = int(round(pcd_iv[i, 0]))
        z_iv_i = max(0, min(nz_iv - 1, z_iv_i))
        z_nd2_i = int(round(np.clip(nd2_z_vals[i], 0, 11)))
        z_pair_to_lm[(z_iv_i, z_nd2_i)].append(i)

    # Build contact sheet rows keyed by z_iv for display
    # (show all z_iv slices, using the best z_nd2 for each)
    z_iv_best_nd2 = {}  # z_iv → most common z_nd2 among its landmarks
    for (z_iv_i, z_nd2_i), lm_list in z_pair_to_lm.items():
        if z_iv_i not in z_iv_best_nd2 or len(lm_list) > len(z_pair_to_lm.get((z_iv_i, z_iv_best_nd2[z_iv_i]), [])):
            z_iv_best_nd2[z_iv_i] = z_nd2_i

    # Contact sheet: 3 cols (ex-vivo, in-vivo warped elastix, overlay)
    n_rows = len(z_iv_best_nd2)
    LABEL_H, GAP, COLS, HEADER_H = 30, 4, 3, 60
    row_ht = out_h + LABEL_H + GAP
    sheet_w = COLS * out_w + (COLS + 1) * GAP
    sheet_h = HEADER_H + n_rows * row_ht + GAP
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    rows_html = []
    ncc_results = []

    # Run elastix for each unique (z_iv, z_nd2) pair and cache
    elastix_cache = {}
    all_iter_logs = {}
    for (z_iv, z_nd2), lm_indices in sorted(z_pair_to_lm.items()):
        if (z_iv, z_nd2) in elastix_cache:
            continue
        nd2_sl = nd2_slices[z_nd2]
        M2d = np.array([
            [M_inv[2, 2], M_inv[2, 1], M_inv[2, 0] * z_nd2 + offset_inv[2]],
            [M_inv[1, 2], M_inv[1, 1], M_inv[1, 0] * z_nd2 + offset_inv[1]],
        ], dtype=np.float64)
        iv_affine_filt = cv2.warpAffine(iv_vol_filt[z_iv], M2d, (nd2_w, nd2_h),
                                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
        iv_affine_raw = cv2.warpAffine(iv_vol_raw[z_iv], M2d, (nd2_w, nd2_h),
                                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)

        # Landmark correspondences — pool all landmarks with same z_iv if enabled
        if POOL_LM_BY_ZIV:
            pooled_indices = []
            for (ziv2, znd2_2), lms2 in z_pair_to_lm.items():
                if ziv2 == z_iv:
                    pooled_indices.extend(lms2)
            pooled_indices = sorted(set(pooled_indices))
        else:
            pooled_indices = lm_indices

        fixed_pts = [(float(ev_nd2[i, 0]), float(ev_nd2[i, 1])) for i in pooled_indices]
        moving_pts = [(float(predicted[i, 0] / ND2_XY_UM), float(predicted[i, 1] / ND2_XY_UM))
                      for i in pooled_indices]

        tp_path = f'{tile_save_dir}/elastix_tp_ziv{z_iv:02d}_znd2{z_nd2:02d}.txt'
        log_save_dir = f'{tile_save_dir}/logs_ziv{z_iv:02d}_znd2{z_nd2:02d}'
        n_own = len(lm_indices)
        n_pool = len(pooled_indices)
        pool_str = f" (+{n_pool - n_own} pooled)" if n_pool > n_own else ""
        print(f"    Elastix iv_z={z_iv:2d} → nd2_z={z_nd2:2d} ({n_own} lm{pool_str}) ...", end="", flush=True)
        iv_elastix, tp, iter_log = run_elastix(nd2_sl, iv_affine_filt, moving_raw=iv_affine_raw,
                                      fixed_points=fixed_pts, moving_points=moving_pts,
                                      save_tp_path=tp_path, save_log_dir=log_save_dir)

        ncc_el = ncc(nd2_sl, iv_elastix)
        ncc_results.append((z_iv, z_nd2, ncc_el))
        all_iter_logs[(z_iv, z_nd2)] = iter_log
        print(f"  NCC={ncc_el:.3f} ({len(iter_log)} iters logged)")

        # Save per-slice affine
        np.savez(f'{tile_save_dir}/slice_ziv{z_iv:02d}_znd2{z_nd2:02d}.npz',
                 M2d=M2d, z_iv=z_iv, z_nd2=z_nd2, lm_indices=np.array(lm_indices))

        elastix_cache[(z_iv, z_nd2)] = iv_elastix

    # Build contact sheet rows for each z_iv with landmarks
    for row_idx, z_iv in enumerate(sorted(z_iv_best_nd2.keys())):
        z_nd2 = z_iv_best_nd2[z_iv]
        nd2_sl = nd2_slices[z_nd2]
        iv_elastix = elastix_cache.get((z_iv, z_nd2))
        if iv_elastix is None:
            continue

        ev_d = cv2.resize(nd2_sl, (out_w, out_h), interpolation=cv2.INTER_AREA)
        el_d = cv2.resize(iv_elastix, (out_w, out_h), interpolation=cv2.INTER_AREA)

        ev_rgb = cv2.cvtColor(norm8(ev_d), cv2.COLOR_GRAY2BGR)
        iv_rgb = cv2.cvtColor(norm8(el_d), cv2.COLOR_GRAY2BGR)
        ov_rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        ov_rgb[:, :, 1] = norm8(ev_d)
        ov_rgb[:, :, 0] = norm8(el_d)
        ov_rgb[:, :, 2] = norm8(el_d)

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
            cv2.drawMarker(iv_rgb, (pr_x, pr_y), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.drawMarker(ov_rgb, (ex_x, ex_y), (180, 0, 0), cv2.MARKER_CROSS, 16, 1)
            cv2.drawMarker(ov_rgb, (pr_x, pr_y), (0, 0, 255), cv2.MARKER_CROSS, 16, 1)
            cv2.line(ov_rgb, (ex_x, ex_y), (pr_x, pr_y), (255, 255, 255), 1)

        ncc_el = ncc(nd2_sl, iv_elastix)
        y0 = HEADER_H + row_idx * row_ht + GAP
        for ci, p in enumerate([ev_rgb, iv_rgb, ov_rgb]):
            x0 = GAP + ci * (out_w + GAP)
            sheet[y0:y0 + out_h, x0:x0 + out_w] = p
        cv2.putText(sheet, f'iv_z={z_iv} -> nd2_z={z_nd2}  NCC={ncc_el:.3f}  ({n_lm} lm)',
                    (GAP + 4, y0 + out_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        rows_html.append(f"""
        <tr>
          <td class="label">iv_z={z_iv} → nd2_z={z_nd2}<br>{n_lm} lm<br>NCC={ncc_el:.3f}</td>
          <td><img src="data:image/jpeg;base64,{to_b64(ev_rgb)}"></td>
          <td><img src="data:image/jpeg;base64,{to_b64(iv_rgb)}"></td>
          <td><img src="data:image/jpeg;base64,{to_b64(ov_rgb)}"></td>
        </tr>""")

    mean_ncc = np.mean([r[2] for r in ncc_results])

    for ci, hdr in enumerate(['Ex-vivo nd2 (blue=actual lm)', 'In-vivo warped elastix (red=predicted)', 'Overlay (green=exvivo, magenta=invivo)']):
        cv2.putText(sheet, hdr, (GAP + ci * (out_w + GAP) + 4, HEADER_H - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    cv2.putText(sheet,
                f'{tile}: 3D affine + elastix | {N_LM} lm | affine={errors.mean():.1f}um | mean NCC={mean_ncc:.3f}',
                (GAP + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    out_png = f'{out_dir}/registration_{tile}.png'
    cv2.imwrite(out_png, sheet)
    print(f"  Saved PNG: {out_png}")

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>{tile} — 3D Affine + Elastix</title>
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
<h1>{tile}: In-vivo → Ex-vivo (3D Affine + Elastix B-spline)</h1>
<div class="info">
  {N_LM} landmarks | Affine error: mean={errors.mean():.1f}µm | scales: {S[0]:.3f}, {S[1]:.3f}, {S[2]:.3f}<br>
  Mean NCC (elastix): {mean_ncc:.3f} | Blue = actual ex-vivo lm, Red = predicted in-vivo lm
</div>
<table>
  <tr>
    <th></th>
    <th>Ex-vivo nd2 (blue=actual lm)</th>
    <th>In-vivo warped elastix (red=predicted)</th>
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

    np.savez(f'{out_dir}/transform_{tile}.npz',
             affine_3x4=A, A_T=A_T, M_fwd=M_fwd, t_fwd=t_fwd, M_inv=M_inv, offset_inv=offset_inv,
             errors=errors, nd2_z_gauss=np.array(nd2_z_vals),
             ev_nd2=ev_nd2, pcd_iv=pcd_iv, predicted=predicted,
             z_pairs=np.array(list(z_pair_to_lm.keys())),
             ncc_results=np.array([(r[0], r[1], r[2]) for r in ncc_results]))
    print(f"  Saved transform: {out_dir}/transform_{tile}.npz")
    print(f"  Saved elastix params: {tile_save_dir}/ ({len(elastix_cache)} z-pairs)")

    # Plot metric vs iterations per z-pair
    if all_iter_logs:
        fig, ax = plt.subplots(figsize=(10, 5))
        for (ziv, znd2), logs in sorted(all_iter_logs.items()):
            if not logs:
                continue
            # Build cumulative iteration count across resolutions
            iters = []
            metrics = []
            offset = 0
            prev_res = -1
            for entry in logs:
                if entry['res'] != prev_res:
                    if prev_res >= 0 and iters:
                        offset = iters[-1] + 1
                    prev_res = entry['res']
                iters.append(offset + entry['iter'])
                metrics.append(entry['metric'])
            ax.plot(iters, metrics, alpha=0.7, linewidth=0.8,
                    label=f'iv={ziv}→nd2={znd2}')
        ax.set_xlabel('Cumulative iteration')
        ax.set_ylabel('Metric value')
        ax.set_title(f'{tile}: Elastix metric vs iterations ({len(all_iter_logs)} z-pairs)')
        if len(all_iter_logs) <= 15:
            ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = f'{out_dir}/metric_iterations_{tile}.png'
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Saved plot: {plot_path}")

    summary.append((tile, N_LM, errors.mean(), mean_ncc))
    del nd2_slices, elastix_cache, all_iter_logs

print(f"\n{'='*60}")
print(f"{'Tile':>10} {'N_lm':>5} {'Affine_err':>10} {'NCC_elastix':>11}")
print(f"{'='*60}")
for tile, n, err, ncc_e in summary:
    print(f"{tile:>10} {n:5d} {err:10.1f} {ncc_e:11.3f}")
print(f"\nAll saved to {OUT_BASE}/")
