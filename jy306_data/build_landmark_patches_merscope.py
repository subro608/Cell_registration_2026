#!/usr/bin/env python3
"""
Rebuild landmark_patches_merscope.html — same as landmark_patches but with
an extra MERSCOPE patch column for each landmark.

MERSCOPE coords derived via:
  - Tiles with saved SIFT affine: M_sift @ [ev_nd2_x, ev_nd2_y, 1]
  - Tiles without SIFT: compute SIFT on the fly
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
from collections import defaultdict

BASE = '/Users/neurolab/neuroinformatics/margaret'

IV_XY_UM  = 0.6835
IV_Z_UM   = 3.0
ND2_XY_UM = 0.645
ND2_Z_UM  = 2.0
MERC_UM_PER_PX = 0.701   # measured from pairwise cell distances

PATCH_R      = 100        # nd2 half-size (100px × 0.645 = 64.5µm)
PHYS_R_UM    = PATCH_R * ND2_XY_UM   # 64.5µm
CROP_MERC    = int(round(PHYS_R_UM / MERC_UM_PER_PX))  # 92px

OUT_DIR  = f'{BASE}/png_exports/registration_per_tile_elastix'
OUT_HTML = f'{OUT_DIR}/landmark_patches_merscope.html'

# ============================================================
# Build tile → MERSCOPE tif mapping
# ============================================================
merscope_tif_map = {}
for f in glob.glob(f'{BASE}/exvivo_merscope_combined/*.tif'):
    bn = os.path.basename(f).replace('.tif', '')
    r, c = bn.split('_merscope')[0].split('_')
    merscope_tif_map[f'row{r}_{c}'] = f
print(f"MERSCOPE tifs: {len(merscope_tif_map)} tiles")

# ============================================================
# Load / compute SIFT affines
# ============================================================
def compute_sift_affine(nd2_mip, merc_mip):
    """Returns M (2x3) mapping nd2-4200 → MERSCOPE-1627, or None."""
    nd2_ds  = cv2.resize(nd2_mip,  (1627, 1627), interpolation=cv2.INTER_AREA)
    nd2_u8  = norm8(nd2_ds)
    merc_u8 = norm8(merc_mip)
    sift = cv2.SIFT_create(nfeatures=5000)
    kp1, des1 = sift.detectAndCompute(nd2_u8,  None)
    kp2, des2 = sift.detectAndCompute(merc_u8, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good) < 10:
        return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if M is None:
        return None
    # M maps 1627-downsampled nd2 → 1627 MERSCOPE
    # Scale translation to 4200-space input: multiply rotation cols by (1627/4200)
    scale_factor = 1627.0 / 4200.0
    M_4200 = M.copy()
    M_4200[:, :2] *= scale_factor
    return M_4200

def get_sift_affine(tile, nd2_mip=None, merc_tif_path=None):
    """Load saved SIFT or compute on the fly. Returns 2x3 matrix or None."""
    saved = f'{BASE}/registration_video/affine_nd2_to_merscope_sift_{tile}.npy'
    if os.path.exists(saved):
        return np.load(saved)
    if nd2_mip is None or merc_tif_path is None:
        return None
    merc_vol = tifffile.imread(merc_tif_path)  # (3, 1627, 1627, 3)
    merc_mip = merc_vol[1, :, :, 0].astype(np.float32)  # middle z, ch0
    M = compute_sift_affine(nd2_mip, merc_mip)
    if M is not None:
        np.save(saved, M)
        print(f"    SIFT computed and saved for {tile}")
    return M

# ============================================================
# Helpers
# ============================================================
def gauss(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def find_z_gaussian(intensities):
    zs   = np.arange(len(intensities), dtype=np.float64)
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
    _, buf = cv2.imencode('.jpg', rgb_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode('ascii')

def crop_patch(img, cx, cy, r):
    h, w = img.shape[:2]
    x0, y0 = cx - r, cy - r
    x1, y1 = cx + r, cy + r
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    dx0, dy0 = sx0 - x0, sy0 - y0
    dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)
    if len(img.shape) == 3:
        patch = np.zeros((2*r, 2*r, img.shape[2]), dtype=img.dtype)
    else:
        patch = np.zeros((2*r, 2*r), dtype=img.dtype)
    if sx1 > sx0 and sy1 > sy0:
        patch[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]
    return patch

def run_elastix_with_raw(fixed_np, moving_filt, moving_raw):
    fixed_sitk  = sitk.GetImageFromArray(norm_f(fixed_np))
    moving_sitk = sitk.GetImageFromArray(norm_f(moving_filt))
    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(fixed_sitk)
    elastix.SetMovingImage(moving_sitk)
    elastix.SetLogToConsole(False)
    pm = sitk.GetDefaultParameterMap('bspline')
    pm['Metric']                              = ['AdvancedMattesMutualInformation']
    pm['NumberOfResolutions']                 = ['3']
    pm['MaximumNumberOfIterations']           = ['500']
    pm['FinalGridSpacingInPhysicalUnits']     = ['50']
    pm['NumberOfSpatialSamples']              = ['4000']
    pm['GridSpacingSchedule']                 = ['4.0', '2.0', '1.0']
    pm['ImagePyramidSchedule']                = ['8', '8', '4', '4', '2', '2']
    elastix.SetParameterMap(pm)
    try:
        elastix.Execute()
        tp = elastix.GetTransformParameterMap()
        transformix = sitk.TransformixImageFilter()
        transformix.SetMovingImage(sitk.GetImageFromArray(moving_raw))
        transformix.SetTransformParameterMap(tp)
        transformix.SetLogToConsole(False)
        transformix.Execute()
        return sitk.GetArrayFromImage(transformix.GetResultImage())
    except Exception:
        return moving_raw

# ============================================================
# Load shared in-vivo data
# ============================================================
print("Loading stitch params...")
with open(f'{BASE}/registration_video/stitch_iou_only_params.json') as f:
    params = json.load(f)
TILE_ORDER = params['tile_order']

print("Loading JY306 in-vivo...")
iv_vol_raw  = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv, ny_iv, nx_iv = iv_vol_raw.shape
print(f"  In-vivo: {iv_vol_raw.shape}")

print("Median filter bg subtraction...")
iv_vol_filt = np.zeros_like(iv_vol_raw)
for z in range(nz_iv):
    bg = median_filter(iv_vol_raw[z], size=15)
    iv_vol_filt[z] = np.clip(iv_vol_raw[z] - bg, 0, None)

# ============================================================
# Find landmark files
# ============================================================
lm_files = sorted(glob.glob(f'{BASE}/registration_video/landmarks_nd2_native_*.npz'))
legacy = f'{BASE}/registration_video/landmarks_27_nd2_native.npz'
if os.path.exists(legacy):
    lm_files.append(legacy)

tile_lm_files = {}
for lm_file in lm_files:
    bn = os.path.basename(lm_file)
    tile = 'row2_1' if 'landmarks_27_nd2_native' in bn \
           else bn.replace('landmarks_nd2_native_', '').replace('.npz', '')
    if tile in TILE_ORDER:
        tile_lm_files[tile] = lm_file
print(f"  {len(tile_lm_files)} tiles with landmarks")

# ============================================================
# Process each tile
# ============================================================
all_tiles_data = {}

for tile in sorted(tile_lm_files.keys()):
    print(f"\n{'='*60}")
    print(f"  {tile}")
    print(f"{'='*60}")

    img_dir   = f'{BASE}/png_exports/registration_video/{tile}'
    nd2_slices = []
    for zi in range(12):
        img = cv2.imread(f'{img_dir}/GFP_z{zi:03d}.png', cv2.IMREAD_UNCHANGED)
        nd2_slices.append(img.astype(np.float32) if img is not None
                          else np.zeros((4200, 4200), dtype=np.float32))
    nd2_slices = np.array(nd2_slices)
    nd2_h, nd2_w = nd2_slices.shape[1], nd2_slices.shape[2]
    nd2_mip = nd2_slices.max(axis=0)

    d      = np.load(tile_lm_files[tile])
    ev_nd2 = d['ev_nd2']
    pcd_iv = d['pcd_invivo_jy306']
    N_LM   = ev_nd2.shape[0]
    if N_LM < 4:
        print(f"  SKIP: only {N_LM} landmarks")
        continue

    # --- MERSCOPE setup ---
    merc_vol = None
    M_sift   = None
    M_sift_fwd = None  # nd2-4200 → MERSCOPE-1627
    if tile in merscope_tif_map:
        merc_path = merscope_tif_map[tile]
        print(f"  Loading MERSCOPE tif: {os.path.basename(merc_path)}")
        merc_vol = tifffile.imread(merc_path)  # (3, 1627, 1627, 3)
        merc_mip = merc_vol[1, :, :, 0].astype(np.float32)  # middle z, ch0
        M_sift_fwd = get_sift_affine(tile, nd2_mip, merc_path)
        if M_sift_fwd is not None:
            print(f"  SIFT affine ready")
        else:
            print(f"  SIFT failed — MERSCOPE patches will be blank")
    else:
        print(f"  No MERSCOPE tif for {tile}")

    # --- Gaussian z-fit ---
    nd2_z_vals = []
    for i in range(N_LM):
        x = int(round(np.clip(ev_nd2[i, 0], 10, nd2_h - 11)))
        y = int(round(np.clip(ev_nd2[i, 1], 10, nd2_h - 11)))
        intensities = [nd2_slices[z][y-10:y+10, x-10:x+10].mean() for z in range(12)]
        nd2_z_vals.append(find_z_gaussian(intensities))

    # --- 3D affine (in-vivo → nd2) ---
    src   = np.column_stack([pcd_iv[:, 2] * IV_XY_UM,
                              pcd_iv[:, 1] * IV_XY_UM,
                              pcd_iv[:, 0] * IV_Z_UM])
    dst   = np.column_stack([ev_nd2[:, 0] * ND2_XY_UM,
                              ev_nd2[:, 1] * ND2_XY_UM,
                              np.array(nd2_z_vals) * ND2_Z_UM])
    src_h = np.hstack([src, np.ones((N_LM, 1))])
    A_T, _, _, _ = np.linalg.lstsq(src_h, dst, rcond=None)
    predicted = src_h @ A_T
    errors    = np.sqrt(np.sum((predicted - dst) ** 2, axis=1))
    print(f"  {N_LM} lm | affine: mean={errors.mean():.1f}µm")

    sx, sy, sz = IV_XY_UM, IV_XY_UM, IV_Z_UM
    ex, ey, ez = ND2_XY_UM, ND2_XY_UM, ND2_Z_UM
    A = A_T.T
    M_fwd = np.array([
        [A[2,2]*sz/ez, A[2,1]*sy/ez, A[2,0]*sx/ez],
        [A[1,2]*sz/ey, A[1,1]*sy/ey, A[1,0]*sx/ey],
        [A[0,2]*sz/ex, A[0,1]*sy/ex, A[0,0]*sx/ex],
    ])
    t_fwd  = np.array([A[2,3]/ez, A[1,3]/ey, A[0,3]/ex])
    M_inv  = np.linalg.inv(M_fwd)
    offset_inv = -M_inv @ t_fwd

    z_to_lm = defaultdict(list)
    for i in range(N_LM):
        z_iv = int(round(np.clip(pcd_iv[i, 0], 0, nz_iv - 1)))
        z_to_lm[z_iv].append(i)

    tile_patches  = []
    elastix_cache = {}

    for z_iv in sorted(z_to_lm.keys()):
        center_iv  = np.array([z_iv, ny_iv / 2, nx_iv / 2])
        center_nd2 = M_fwd @ center_iv + t_fwd
        z_nd2      = int(round(np.clip(center_nd2[0], 0, 11)))

        if z_iv not in elastix_cache:
            nd2_sl = nd2_slices[z_nd2]
            M2d = np.array([
                [M_inv[2,2], M_inv[2,1], M_inv[2,0]*z_nd2 + offset_inv[2]],
                [M_inv[1,2], M_inv[1,1], M_inv[1,0]*z_nd2 + offset_inv[1]],
            ], dtype=np.float64)
            iv_affine_filt = cv2.warpAffine(iv_vol_filt[z_iv], M2d, (nd2_w, nd2_h),
                                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            iv_affine_raw  = cv2.warpAffine(iv_vol_raw[z_iv],  M2d, (nd2_w, nd2_h),
                                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            print(f"    Elastix iv_z={z_iv} → nd2_z={z_nd2} ...", end="", flush=True)
            iv_elastix_raw = run_elastix_with_raw(nd2_sl, iv_affine_filt, iv_affine_raw)
            print(" done")
            elastix_cache[z_iv] = (nd2_sl, iv_elastix_raw, z_nd2)

        nd2_sl, iv_elastix_raw, z_nd2 = elastix_cache[z_iv]
        ev_norm = norm8(nd2_sl)
        iv_norm = norm8(iv_elastix_raw)

        for i in z_to_lm[z_iv]:
            ex_x = int(round(ev_nd2[i, 0]))
            ex_y = int(round(ev_nd2[i, 1]))
            pr_x = int(round(predicted[i, 0] / ND2_XY_UM))
            pr_y = int(round(predicted[i, 1] / ND2_XY_UM))
            cx, cy = ex_x, ex_y

            ev_patch = crop_patch(ev_norm, cx, cy, PATCH_R)
            iv_patch = crop_patch(iv_norm, cx, cy, PATCH_R)

            ev_rgb = cv2.cvtColor(ev_patch, cv2.COLOR_GRAY2BGR)
            iv_rgb = cv2.cvtColor(iv_patch, cv2.COLOR_GRAY2BGR)
            ov_rgb = np.zeros((2*PATCH_R, 2*PATCH_R, 3), dtype=np.uint8)
            ov_rgb[:, :, 1] = ev_patch
            ov_rgb[:, :, 0] = iv_patch
            ov_rgb[:, :, 2] = iv_patch

            # --- MERSCOPE patch ---
            merc_b64 = None
            if M_sift_fwd is not None and merc_vol is not None:
                merc_pt = M_sift_fwd @ np.array([ex_x, ex_y, 1.0])
                mx, my  = int(round(merc_pt[0])), int(round(merc_pt[1]))
                # Use middle z-slice (z=1), channel 0
                merc_slice = merc_vol[1, :, :, 0].astype(np.float32)
                merc_patch = crop_patch(merc_slice, mx, my, CROP_MERC)
                merc_u8    = norm8(merc_patch)
                merc_rgb   = cv2.cvtColor(merc_u8, cv2.COLOR_GRAY2BGR)
                # Green crosshair at centre
                cv2.drawMarker(merc_rgb, (CROP_MERC, CROP_MERC),
                               (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
                merc_b64   = to_b64(merc_rgb)

            tile_patches.append({
                'idx':          i,
                'z_iv':         int(pcd_iv[i, 0]),
                'z_nd2':        z_nd2,
                'z_nd2_gauss':  nd2_z_vals[i],
                'ex_x':         ex_x, 'ex_y': ex_y,
                'pr_x':         pr_x, 'pr_y': pr_y,
                'pred_dx':      pr_x - cx, 'pred_dy': pr_y - cy,
                'err_um':       float(errors[i]),
                'ev_b64':       to_b64(ev_rgb),
                'iv_b64':       to_b64(iv_rgb),
                'ov_b64':       to_b64(ov_rgb),
                'merc_b64':     merc_b64,
            })

    all_tiles_data[tile] = {
        'n_lm':     N_LM,
        'mean_err': float(errors.mean()),
        'patches':  tile_patches,
    }
    print(f"  {len(tile_patches)} patches extracted")
    del nd2_slices, elastix_cache, merc_vol

# ============================================================
# Build HTML
# ============================================================
print("\nBuilding HTML...")

SZ         = 2 * PATCH_R
SZ_MERC    = 2 * CROP_MERC
tile_options = ""
tile_divs    = ""
first        = True

for tile in sorted(all_tiles_data.keys()):
    td   = all_tiles_data[tile]
    sel  = " selected" if first else ""
    disp = "flex"      if first else "none"
    tile_options += f'<option value="{tile}"{sel}>{tile} ({td["n_lm"]} lm, err={td["mean_err"]:.1f}µm)</option>\n'

    cards = ""
    for p in td['patches']:
        ecx, ecy = PATCH_R, PATCH_R
        pcx, pcy = PATCH_R + p['pred_dx'], PATCH_R + p['pred_dy']

        merc_col = ""
        if p['merc_b64']:
            merc_col = f"""
            <div class="patch-col">
              <div class="patch-label" style="color:#fa0">MERSCOPE (ch0)</div>
              <img class="patch-img merc-img" src="data:image/jpeg;base64,{p['merc_b64']}"
                   style="width:200px;height:200px">
            </div>"""
        else:
            merc_col = f"""
            <div class="patch-col">
              <div class="patch-label" style="color:#fa0">MERSCOPE (n/a)</div>
              <div class="patch-img" style="width:200px;height:200px;background:#1a1a1a;display:inline-block"></div>
            </div>"""

        cards += f"""
        <div class="card">
          <div class="card-header">LM #{p['idx']}  |  iv_z={p['z_iv']} → nd2_z={p['z_nd2']} (gauss={p['z_nd2_gauss']:.1f})  |  err={p['err_um']:.1f}µm  |  nd2:({p['ex_x']},{p['ex_y']})  pred:({p['pr_x']},{p['pr_y']})</div>
          <div class="patches">
            <div class="patch-col">
              <div class="patch-label">Ex-vivo nd2</div>
              <div class="patch-wrap">
                <img class="patch-img" src="data:image/jpeg;base64,{p['ev_b64']}">
                <svg class="xhair" viewBox="0 0 {SZ} {SZ}">
                  <line x1="{ecx-12}" y1="{ecy}" x2="{ecx+12}" y2="{ecy}" stroke="#00a0ff" stroke-width="2"/>
                  <line x1="{ecx}" y1="{ecy-12}" x2="{ecx}" y2="{ecy+12}" stroke="#00a0ff" stroke-width="2"/>
                </svg>
              </div>
            </div>
            <div class="patch-col">
              <div class="patch-label">In-vivo warped</div>
              <div class="patch-wrap">
                <img class="patch-img" src="data:image/jpeg;base64,{p['iv_b64']}">
                <svg class="xhair" viewBox="0 0 {SZ} {SZ}">
                  <line x1="{pcx-12}" y1="{pcy}" x2="{pcx+12}" y2="{pcy}" stroke="#ff3030" stroke-width="2"/>
                  <line x1="{pcx}" y1="{pcy-12}" x2="{pcx}" y2="{pcy+12}" stroke="#ff3030" stroke-width="2"/>
                </svg>
              </div>
            </div>
            <div class="patch-col">
              <div class="patch-label">Overlay</div>
              <div class="patch-wrap">
                <img class="patch-img" src="data:image/jpeg;base64,{p['ov_b64']}">
                <svg class="xhair" viewBox="0 0 {SZ} {SZ}">
                  <line x1="{ecx-12}" y1="{ecy}" x2="{ecx+12}" y2="{ecy}" stroke="#00a0ff" stroke-width="2"/>
                  <line x1="{ecx}" y1="{ecy-12}" x2="{ecx}" y2="{ecy+12}" stroke="#00a0ff" stroke-width="2"/>
                  <line x1="{pcx-12}" y1="{pcy}" x2="{pcx+12}" y2="{pcy}" stroke="#ff3030" stroke-width="2"/>
                  <line x1="{pcx}" y1="{pcy-12}" x2="{pcx}" y2="{pcy+12}" stroke="#ff3030" stroke-width="2"/>
                  <line x1="{ecx}" y1="{ecy}" x2="{pcx}" y2="{pcy}" stroke="white" stroke-width="1" stroke-dasharray="3,2"/>
                </svg>
              </div>
            </div>
            {merc_col}
          </div>
        </div>"""

    tile_divs += f"""
    <div class="tile-section" id="tile-{tile}" style="display:{disp}">
      <div class="tile-info">{tile}: {td['n_lm']} landmarks | Mean affine error: {td['mean_err']:.1f}µm</div>
      <div class="cards-grid">{cards}
      </div>
    </div>"""
    first = False

html = f"""<!DOCTYPE html>
<html>
<head>
<title>Landmark Patches — Ex-vivo vs In-vivo + MERSCOPE</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #111; color: #eee; font-family: 'Menlo', monospace; padding: 20px; }}
  h1 {{ font-size: 18px; margin-bottom: 12px; }}
  .controls {{ margin-bottom: 16px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
  .controls label {{ font-size: 14px; color: #aaa; }}
  select {{ background: #222; color: #eee; border: 1px solid #555; padding: 6px 10px;
            font-size: 14px; font-family: monospace; border-radius: 4px; cursor: pointer; }}
  select:hover {{ border-color: #888; }}
  .btn {{ background: #333; color: #eee; border: 1px solid #555; padding: 5px 14px;
          font-size: 13px; font-family: monospace; border-radius: 4px; cursor: pointer; user-select: none; }}
  .btn:hover {{ background: #444; }}
  .btn.active {{ background: #264; border-color: #4a8; color: #8e8; }}
  .size-controls {{ display: flex; align-items: center; gap: 6px; }}
  .size-controls span {{ font-size: 13px; color: #aaa; min-width: 50px; text-align: center; }}
  .tile-info {{ font-size: 14px; color: #aaa; margin-bottom: 12px; padding: 8px 12px;
                background: #1a1a1a; border-radius: 4px; border-left: 3px solid #4a9; }}
  .cards-grid {{ display: flex; flex-wrap: wrap; gap: 12px; }}
  .card {{ background: #1a1a1a; border: 1px solid #333; border-radius: 6px; overflow: hidden; }}
  .card:hover {{ border-color: #666; }}
  .card-header {{ font-size: 11px; color: #999; padding: 6px 10px; background: #151515;
                  border-bottom: 1px solid #333; }}
  .patches {{ display: flex; gap: 2px; padding: 4px; }}
  .patch-col {{ text-align: center; }}
  .patch-label {{ font-size: 10px; color: #777; margin-bottom: 2px; }}
  .patch-wrap {{ position: relative; display: inline-block; }}
  .patch-img {{ display: block; image-rendering: pixelated; }}
  .merc-img {{ border: 1px solid #fa04; }}
  .xhair {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}
  .legend {{ font-size: 11px; color: #666; }}
</style>
</head>
<body>
<h1>Landmark Patches: Ex-vivo vs In-vivo Warped + MERSCOPE</h1>
<div class="controls">
  <label>Tile:</label>
  <select id="tileSelect" onchange="switchTile()">
    {tile_options}
  </select>
  <div class="size-controls">
    <button class="btn" onclick="changeSize(-1)">−</button>
    <span id="sizeLabel">200px</span>
    <button class="btn" onclick="changeSize(1)">+</button>
  </div>
  <button class="btn active" id="xhairBtn" onclick="toggleCrosshairs()">Crosshairs</button>
  <span class="legend">Blue=exvivo actual | Red=invivo predicted | <span style="color:#fa0">Orange=MERSCOPE</span></span>
</div>
{tile_divs}
<script>
let patchSize = 200;
let showCrosshairs = true;

function switchTile() {{
  document.querySelectorAll('.tile-section').forEach(el => el.style.display = 'none');
  const tile = document.getElementById('tileSelect').value;
  const el = document.getElementById('tile-' + tile);
  el.style.display = 'flex';
  el.style.flexDirection = 'column';
}}

function changeSize(dir) {{
  const sizes = [100, 150, 200, 250, 300, 400, 500];
  let idx = sizes.indexOf(patchSize);
  if (idx < 0) idx = 2;
  idx = Math.max(0, Math.min(sizes.length - 1, idx + dir));
  patchSize = sizes[idx];
  document.getElementById('sizeLabel').textContent = patchSize + 'px';
  document.querySelectorAll('.patch-img').forEach(img => {{
    img.style.width  = patchSize + 'px';
    img.style.height = patchSize + 'px';
  }});
}}

function toggleCrosshairs() {{
  showCrosshairs = !showCrosshairs;
  document.getElementById('xhairBtn').classList.toggle('active', showCrosshairs);
  document.querySelectorAll('.xhair').forEach(svg => {{
    svg.style.display = showCrosshairs ? '' : 'none';
  }});
}}

// Init
document.querySelectorAll('.patch-img').forEach(img => {{
  img.style.width  = '200px';
  img.style.height = '200px';
}});
</script>
</body>
</html>
"""

with open(OUT_HTML, 'w') as f:
    f.write(html)
print(f"\nSaved: {OUT_HTML}")
print(f"Total landmarks: {sum(len(td['patches']) for td in all_tiles_data.values())}")
