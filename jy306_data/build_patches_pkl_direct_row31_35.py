#!/usr/bin/env python3
"""
Landmark patch viewer for row3_1 and row3_5 using DIRECT pkl deformation field
instead of the 3D affine approach (which has insufficient z-overlap for these tiles).

Pipeline: JY306 in-vivo → pkl inverse → MERSCOPE space → SIFT affine inverse → nd2 native
Uses pkl-derived correspondences to fit a 2D affine per z-pair and warp full in-vivo slices.

Format matches landmark_patches_warp_err.html: 5 columns with crosshairs & SVG overlays.

Output: 3d_viewer/landmark_patches_pkl_row31_35.html
"""
import numpy as np
import cv2
import os
import base64
import json
import tifffile
from scipy.optimize import curve_fit
from collections import defaultdict

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_HTML = f'{BASE}/3d_viewer/landmark_patches_pkl_row31_35.html'

IV_XY_UM = 0.6835
IV_Z_UM = 3.0
ND2_XY_UM = 0.645
ND2_Z_UM = 2.0
PATCH_R = 100  # half-size in full-res pixels

TILES = {
    'row3_1': 'landmarks_nd2_native_row3_1.npz',
    'row3_5': 'landmarks_nd2_native_row3_5.npz',
}

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
    centroid = float(np.sum(zs * vals) / total)
    return centroid

def norm8(img):
    vals = img[img > 0]
    if len(vals) < 100:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(vals, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)

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
        patch = np.zeros((2 * r, 2 * r, img.shape[2]), dtype=img.dtype)
    else:
        patch = np.zeros((2 * r, 2 * r), dtype=img.dtype)
    if sx1 > sx0 and sy1 > sy0:
        patch[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]
    return patch


# ============================================================
# Load shared data
# ============================================================
print("Loading JY306 in-vivo...")
iv_vol_raw = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv, ny_iv, nx_iv = iv_vol_raw.shape
print(f"  In-vivo: {iv_vol_raw.shape}")


# ============================================================
# Process each tile
# ============================================================
all_tiles_data = {}

for tile, lm_file in sorted(TILES.items()):
    print(f"\n{'='*60}")
    print(f"  {tile} — pkl direct")
    print(f"{'='*60}")

    # Load landmarks (from make_contact_sheet.py)
    lm_path = f'{BASE}/registration_video/{lm_file}'
    d = np.load(lm_path)
    ev_nd2 = d['ev_nd2']       # (N,3) x,y,z in nd2 pixels
    iv_nd2 = d['iv_nd2']       # (N,3) x,y,z in nd2 pixels (from pkl inverse + SIFT)
    pcd_iv = d['pcd_invivo_jy306']  # (N,3) z,y,x in JY306
    N_LM = len(ev_nd2)
    print(f"  {N_LM} landmarks loaded")

    # Load nd2 z-slices
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

    # Find gaussian z for each landmark in nd2
    nd2_z_vals = []
    for i in range(N_LM):
        x, y = ev_nd2[i, 0], ev_nd2[i, 1]
        c = int(round(np.clip(x, 10, nd2_h - 11)))
        r = int(round(np.clip(y, 10, nd2_h - 11)))
        intensities = [nd2_slices[z][r-10:r+10, c-10:c+10].mean() for z in range(12)]
        nd2_z_vals.append(find_z_gaussian(intensities))

    # Compute pkl-based 2D errors
    pkl_dx = iv_nd2[:, 0] - ev_nd2[:, 0]
    pkl_dy = iv_nd2[:, 1] - ev_nd2[:, 1]
    pkl_dist_px = np.sqrt(pkl_dx**2 + pkl_dy**2)
    pkl_dist_um = pkl_dist_px * ND2_XY_UM
    print(f"  PKL 2D dist: mean={pkl_dist_um.mean():.1f}µm, max={pkl_dist_um.max():.1f}µm")

    # Group landmarks by z-pair (z_iv, z_nd2)
    z_pair_to_lm = defaultdict(list)
    for i in range(N_LM):
        z_iv = int(round(pcd_iv[i, 0]))
        z_iv = max(0, min(nz_iv - 1, z_iv))
        z_nd2 = int(round(np.clip(nd2_z_vals[i], 0, 11)))
        z_pair_to_lm[(z_iv, z_nd2)].append(i)

    # For each z-pair: compute 2D affine from PKL correspondences, warp in-vivo
    tile_patches = []
    WARP_ERR_MAX_UM = 5.0

    for (z_iv, z_nd2), lm_indices in sorted(z_pair_to_lm.items()):
        # All landmarks (even from other z-pairs) to fit a robust 2D affine
        # Using JY306 coords → nd2 coords via pkl chain
        src_pts_all = np.column_stack([pcd_iv[:, 2], pcd_iv[:, 1]]).astype(np.float64)  # x,y in JY306
        dst_pts_all = np.column_stack([iv_nd2[:, 0], iv_nd2[:, 1]]).astype(np.float64)  # x,y in nd2

        # Fit 2D affine: JY306 (x,y) → nd2 (x,y)
        if N_LM >= 3:
            src_h = np.hstack([src_pts_all, np.ones((N_LM, 1))])
            M_jy_to_nd2, _, _, _ = np.linalg.lstsq(src_h, dst_pts_all, rcond=None)
            # M_jy_to_nd2 is (3,2): [sx, 1]^T @ M = [dx, dy]
            M2d = np.zeros((2, 3), dtype=np.float64)
            M2d[0, :] = [M_jy_to_nd2[0, 0], M_jy_to_nd2[1, 0], M_jy_to_nd2[2, 0]]
            M2d[1, :] = [M_jy_to_nd2[0, 1], M_jy_to_nd2[1, 1], M_jy_to_nd2[2, 1]]
        else:
            print(f"    SKIP z_iv={z_iv}→z_nd2={z_nd2}: insufficient landmarks for affine")
            continue

        # Warp in-vivo z-slice to nd2 space using fitted affine
        iv_slice = iv_vol_raw[z_iv]
        iv_warped = cv2.warpAffine(iv_slice, M2d, (nd2_w, nd2_h),
                                    flags=cv2.INTER_LINEAR, borderValue=0)

        nd2_sl = nd2_slices[z_nd2]
        ev_norm = norm8(nd2_sl)
        iv_norm = norm8(iv_warped)
        iv_raw_norm = norm8(iv_vol_raw[z_iv])

        print(f"    z_iv={z_iv} → z_nd2={z_nd2} ({len(lm_indices)} lm)")

        for i in lm_indices:
            ex_x = int(round(ev_nd2[i, 0]))
            ex_y = int(round(ev_nd2[i, 1]))
            # Predicted position from pkl chain (iv_nd2)
            pr_x = int(round(iv_nd2[i, 0]))
            pr_y = int(round(iv_nd2[i, 1]))

            cx, cy = ex_x, ex_y

            ev_patch = crop_patch(ev_norm, cx, cy, PATCH_R)
            iv_patch = crop_patch(iv_norm, cx, cy, PATCH_R)

            # Raw in-vivo patch at JY306 position
            iv_raw_x = int(round(pcd_iv[i, 2]))
            iv_raw_y = int(round(pcd_iv[i, 1]))
            iv_raw_patch = crop_patch(iv_raw_norm, iv_raw_x, iv_raw_y, PATCH_R)

            ev_rgb = cv2.cvtColor(ev_patch, cv2.COLOR_GRAY2BGR)
            iv_rgb = cv2.cvtColor(iv_patch, cv2.COLOR_GRAY2BGR)
            iv_raw_rgb = cv2.cvtColor(iv_raw_patch, cv2.COLOR_GRAY2BGR)

            # Overlay: green=exvivo, magenta=invivo warped
            ov_rgb = np.zeros((2*PATCH_R, 2*PATCH_R, 3), dtype=np.uint8)
            ov_rgb[:, :, 1] = ev_patch
            ov_rgb[:, :, 0] = iv_patch
            ov_rgb[:, :, 2] = iv_patch

            # Crosshair offset
            pred_dx = pr_x - cx
            pred_dy = pr_y - cy
            err_px = float(np.sqrt(pred_dx**2 + pred_dy**2))
            err_um = err_px * ND2_XY_UM

            # Warp diagnostic: Gaussian blob at iv position, warped through same affine
            bimg = np.zeros((ny_iv, nx_iv), dtype=np.float32)
            ly = int(round(pcd_iv[i, 1]))
            lx = int(round(pcd_iv[i, 2]))
            for dy in range(-8, 9):
                for dx in range(-8, 9):
                    yy, xx = ly + dy, lx + dx
                    if 0 <= yy < ny_iv and 0 <= xx < nx_iv:
                        bimg[yy, xx] = 255.0 * np.exp(-0.5 * (dy*dy + dx*dx) / 9.0)
            blob_warped = cv2.warpAffine(bimg, M2d, (nd2_w, nd2_h),
                                          flags=cv2.INTER_LINEAR, borderValue=0)

            # Measure blob centroid distance from ex-vivo landmark
            bmax_full = blob_warped.max()
            blob_dist_um = 999.0
            if bmax_full > 1:
                yy_b, xx_b = np.where(blob_warped > bmax_full * 0.1)
                if len(xx_b) > 0:
                    ww = blob_warped[yy_b, xx_b]
                    blob_cx = float(np.average(xx_b, weights=ww))
                    blob_cy = float(np.average(yy_b, weights=ww))
                    blob_dist_um = np.sqrt((blob_cx - cx)**2 + (blob_cy - cy)**2) * ND2_XY_UM

            blob_patch = crop_patch(blob_warped, cx, cy, PATCH_R)
            bmax = blob_patch.max()
            if bmax > 1:
                blob_norm = np.clip(blob_patch / bmax * 255, 0, 255).astype(np.uint8)
            else:
                blob_norm = np.zeros((2*PATCH_R, 2*PATCH_R), dtype=np.uint8)
            bin_vis = np.zeros((2*PATCH_R, 2*PATCH_R, 3), dtype=np.uint8)
            bin_vis[:, :, 1] = blob_norm

            passed = blob_dist_um <= WARP_ERR_MAX_UM

            tile_patches.append({
                'idx': i,
                'z_iv': int(pcd_iv[i, 0]),
                'z_nd2': z_nd2,
                'z_nd2_gauss': nd2_z_vals[i],
                'ex_x': ex_x, 'ex_y': ex_y,
                'pr_x': pr_x, 'pr_y': pr_y,
                'iv_raw_x': iv_raw_x, 'iv_raw_y': iv_raw_y,
                'pred_dx': pred_dx, 'pred_dy': pred_dy,
                'err_um': err_um,
                'blob_dist_um': blob_dist_um,
                'passed': passed,
                'corr_method': f"pkl={err_um:.1f}µm",
                'ev_b64': to_b64(ev_rgb),
                'iv_b64': to_b64(iv_rgb),
                'iv_raw_b64': to_b64(iv_raw_rgb),
                'ov_b64': to_b64(ov_rgb),
                'bin_b64': to_b64(bin_vis),
            })

    n_passed = sum(1 for p in tile_patches if p['passed'])
    all_tiles_data[tile] = {
        'n_lm': N_LM,
        'mean_err': float(pkl_dist_um.mean()),
        'patches': tile_patches,
        'n_passed': n_passed,
    }
    print(f"  {len(tile_patches)} patches | {n_passed} passed blob filter (<{WARP_ERR_MAX_UM}µm)")
    del nd2_slices


# ============================================================
# Build HTML — same format as landmark_patches_warp_err.html
# ============================================================
print("\nBuilding HTML...")

tile_options = ""
tile_divs = ""
first = True
SZ = 2 * PATCH_R

for tile in sorted(all_tiles_data.keys()):
    td = all_tiles_data[tile]
    sel = " selected" if first else ""
    disp = "flex" if first else "none"
    tile_options += f'<option value="{tile}"{sel}>{tile} ({td["n_lm"]} lm, pkl err={td["mean_err"]:.1f}µm, {td["n_passed"]} passed)</option>\n'

    cards = ""
    for p in td['patches']:
        ecx, ecy = PATCH_R, PATCH_R
        pcx, pcy = PATCH_R + p['pred_dx'], PATCH_R + p['pred_dy']
        badge_color = "#4a8" if p['passed'] else "#a44"
        badge_text = f"blob={p['blob_dist_um']:.1f}µm"

        cards += f"""
        <div class="card" style="border-left: 3px solid {badge_color}">
          <div class="card-header">LM #{p['idx']} | iv_z={p['z_iv']}→nd2_z={p['z_nd2']} (g={p['z_nd2_gauss']:.1f}) | {p['corr_method']} | {badge_text}</div>
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
              <div class="patch-label">In-vivo warped (pkl)</div>
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
            <div class="patch-col">
              <div class="patch-label">In-vivo raw ({p['iv_raw_x']},{p['iv_raw_y']})</div>
              <div class="patch-wrap">
                <img class="patch-img" src="data:image/jpeg;base64,{p['iv_raw_b64']}">
                <svg class="xhair" viewBox="0 0 {SZ} {SZ}">
                  <line x1="{PATCH_R-12}" y1="{PATCH_R}" x2="{PATCH_R+12}" y2="{PATCH_R}" stroke="#ff3030" stroke-width="2"/>
                  <line x1="{PATCH_R}" y1="{PATCH_R-12}" x2="{PATCH_R}" y2="{PATCH_R+12}" stroke="#ff3030" stroke-width="2"/>
                </svg>
              </div>
            </div>
            <div class="patch-col">
              <div class="patch-label">Warp diagnostic</div>
              <div class="patch-wrap">
                <img class="patch-img" src="data:image/jpeg;base64,{p['bin_b64']}">
                <svg class="xhair" viewBox="0 0 {SZ} {SZ}">
                  <line x1="{ecx-12}" y1="{ecy}" x2="{ecx+12}" y2="{ecy}" stroke="#00a0ff" stroke-width="2"/>
                  <line x1="{ecx}" y1="{ecy-12}" x2="{ecx}" y2="{ecy+12}" stroke="#00a0ff" stroke-width="2"/>
                </svg>
              </div>
            </div>
          </div>
        </div>"""

    tile_divs += f"""
    <div class="tile-section" id="tile-{tile}" style="display:{disp}">
      <div class="tile-info">{tile}: {td['n_lm']} landmarks | PKL mean error: {td['mean_err']:.1f}µm | {td['n_passed']} passed blob filter</div>
      <div class="cards-grid">{cards}
      </div>
    </div>"""
    first = False

html = f"""<!DOCTYPE html>
<html>
<head>
<title>Landmark Patches — PKL Direct (row3_1 & row3_5)</title>
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
  .xhair {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}
  .legend {{ font-size: 11px; color: #666; }}
</style>
</head>
<body>
<h1>Landmark Patches: PKL Direct Pipeline (row3_1 & row3_5)</h1>
<div class="controls">
  <label>Tile:</label>
  <select id="tileSelect" onchange="switchTile()">
    {tile_options}
  </select>
  <div class="size-controls">
    <button class="btn" onclick="changeSize(-1)">&minus;</button>
    <span id="sizeLabel">200px</span>
    <button class="btn" onclick="changeSize(1)">+</button>
  </div>
  <button class="btn active" id="xhairBtn" onclick="toggleCrosshairs()">Crosshairs</button>
  <span class="legend">Blue=exvivo actual | Red=invivo predicted (pkl) | Green/Magenta=overlay | Green blob=warp diag</span>
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
    img.style.width = patchSize + 'px';
    img.style.height = patchSize + 'px';
  }});
}}

function toggleCrosshairs() {{
  showCrosshairs = !showCrosshairs;
  const btn = document.getElementById('xhairBtn');
  btn.classList.toggle('active', showCrosshairs);
  document.querySelectorAll('.xhair').forEach(svg => {{
    svg.style.display = showCrosshairs ? '' : 'none';
  }});
}}

// Init
document.querySelectorAll('.patch-img').forEach(img => {{
  img.style.width = '200px';
  img.style.height = '200px';
}});
</script>
</body>
</html>
"""

with open(OUT_HTML, 'w') as f:
    f.write(html)
sz_mb = os.path.getsize(OUT_HTML) / 1e6
print(f"\nSaved: {OUT_HTML} ({sz_mb:.1f} MB)")
total_patches = sum(len(td['patches']) for td in all_tiles_data.values())
total_passed = sum(td['n_passed'] for td in all_tiles_data.values())
print(f"Total: {total_patches} patches, {total_passed} passed blob filter")
