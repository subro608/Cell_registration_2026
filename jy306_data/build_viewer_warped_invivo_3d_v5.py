#!/usr/bin/env python3
"""
3D HTML viewer: 4 modalities registered to ex-vivo stitched space.
Ex-vivo + In-vivo warped + Calcium warped + MERSCOPE dots.
All in single merged 3D scene with per-modality controls.
"""
import numpy as np
import cv2
import os
import base64
import json
import glob
import io
import re
import pickle
import tifffile
import SimpleITK as sitk
import pandas as pd
from PIL import Image
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from collections import defaultdict, Counter

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT = f'{BASE}/3d_viewer/viewer_warped_invivo_3d_v5.html'
os.makedirs(f'{BASE}/3d_viewer', exist_ok=True)

VAROL = os.path.join(BASE, 'jy306_varol')
PKL_MERC_DIR = os.path.join(BASE, 'merscope_exvivo ')  # trailing space

TILE_TO_REGION = {
    'row1_3': 23,
    'row2_1': 17, 'row2_2': 18, 'row2_3': 19, 'row2_4': 20, 'row2_5': 21,
    'row3_1': 16, 'row3_2': 15, 'row3_3': 14, 'row3_4': 13, 'row3_5': 12, 'row3_6': 11,
    'row4_1': 5,  'row4_2': 6,  'row4_3': 7,  'row4_4': 8,  'row4_5': 9,  'row4_6': 10,
    'row5_1': 4,
}

def make_rainbow_palette(n):
    pal = []
    for i in range(n):
        h = int(180 * i / n)
        s = 200 + int(55 * (i % 3) / 2)
        v = 200 + int(55 * (i % 5) / 4)
        bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        pal.append([int(bgr[2]), int(bgr[1]), int(bgr[0])])  # RGB
    return pal

def build_pkl_affine(T_dict):
    B = np.eye(4)
    for step in T_dict:
        for k, v in step.items():
            if k == 'bhat':
                B = B @ np.c_[v, np.array((0, 0, 0, 1))]
            if k == 'scale':
                B[:, :3] *= v
    R_3 = np.linalg.inv(B[:3, :3]).T
    offset_3 = -B[-1, :-1] @ np.linalg.inv(B[:3, :3])
    R_3_inv = np.linalg.inv(R_3)
    return R_3_inv, offset_3

N_GENE_COLORS = 500
GENE_PALETTE = make_rainbow_palette(N_GENE_COLORS)
MAX_MERC = 150000

IV_XY_UM = 0.6835
IV_Z_UM = 3.0
ND2_XY_UM = 0.645
ND2_Z_UM = 2.0

DS = 4
DS_EX = 4
VOXEL_THRESH_EX = 12
VOXEL_THRESH_IV = 40
PATCH_SZ = 80
DZ_SLICES = 2
PHYS_RADIUS = 50  # 50µm half-width
CROP_ND2 = int(round(PHYS_RADIUS / ND2_XY_UM))  # ~78 px
CROP_JY = int(round(PHYS_RADIUS / IV_XY_UM))     # ~73 px
CROP_JY_ZOOM = int(round(PHYS_RADIUS / 2 / IV_XY_UM))  # ~37 px (2x zoom)
WARP_ERR_MAX_UM = 5.0

# ============================================================
# Helpers
# ============================================================
def normalize_u8(img):
    vals = img[img > 0]
    if len(vals) < 100:
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(vals, [2, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.float32)

def stitch_tile_v5(sl, tile, tile_idx_map, iou_transforms, cum_iou,
                    stitch_elastix_dir, tile_order, canvas_w, canvas_h):
    """Apply full v5 stitching: pair IOU rigid + pair elastix + cumulative placement."""
    tidx = tile_idx_map[tile]
    if tidx == 0:
        M_cum = np.array(cum_iou[tile])[:2, :]
        return cv2.warpAffine(sl, M_cum, (canvas_w, canvas_h),
                               flags=cv2.INTER_LINEAR, borderValue=0)
    prev_key = tile_order[tidx - 1]
    pair_key = f'{prev_key}_to_{tile}'
    if pair_key in iou_transforms:
        pair_warp = np.array(iou_transforms[pair_key]['warp_matrix'], dtype=np.float32)
        sl_rigid = cv2.warpAffine(sl, pair_warp, (4200, 4200),
                                   flags=cv2.INTER_LINEAR, borderValue=0)
    else:
        sl_rigid = sl
    tfm_file = f'{stitch_elastix_dir}/{pair_key}/TransformParameters.0.txt'
    if os.path.exists(tfm_file):
        sl_n = normalize_u8(sl_rigid)
        sl_itk = sitk.GetImageFromArray(sl_n)
        tfm = sitk.ReadParameterFile(tfm_file)
        transformix = sitk.TransformixImageFilter()
        transformix.SetTransformParameterMap(tfm)
        transformix.SetMovingImage(sl_itk)
        transformix.LogToConsoleOff()
        try:
            transformix.Execute()
            sl_elx = sitk.GetArrayFromImage(transformix.GetResultImage()).astype(np.float32)
            vals = sl_rigid[sl_rigid > 0]
            if len(vals) > 0:
                p2, p995 = np.percentile(vals, [2, 99.5])
                sl_deformed = sl_elx / 255.0 * (p995 - p2) + p2
            else:
                sl_deformed = sl_elx
        except Exception:
            sl_deformed = sl_rigid
    else:
        sl_deformed = sl_rigid
    M_prev = np.array(cum_iou[prev_key])[:2, :]
    return cv2.warpAffine(sl_deformed, M_prev, (canvas_w, canvas_h),
                           flags=cv2.INTER_LINEAR, borderValue=0)

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

def encode_f32(arr):
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()

def depth_color(t):
    r = min(2 * t, 1.0)
    g = min(2 * (1 - t), 1.0)
    return (r, g, 0.0)

def make_depth_mip(slices_list, patch_sz):
    if not slices_list:
        return np.zeros((patch_sz, patch_sz, 3), dtype=np.uint8)
    stack = np.array([s for s, _ in slices_list])
    depths = np.array([d for _, d in slices_list])
    p99 = np.percentile(stack[stack > 0], 99) if (stack > 0).any() else 1
    stack_norm = np.clip(stack / max(p99, 1), 0, 1)
    argmax_z = np.argmax(stack_norm, axis=0)
    mip_val = np.max(stack_norm, axis=0)
    h, w = argmax_z.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for zi in range(len(slices_list)):
        mask = argmax_z == zi
        r, g, b = depth_color(depths[zi])
        rgb[mask, 0] = mip_val[mask] * r
        rgb[mask, 1] = mip_val[mask] * g
        rgb[mask, 2] = mip_val[mask] * b
    rgb_u8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    resized = np.array(Image.fromarray(rgb_u8).resize((patch_sz, patch_sz), Image.LANCZOS))
    return resized

def make_gray_mip(slices_list, patch_sz, color=(0, 255, 0)):
    if not slices_list:
        return np.zeros((patch_sz, patch_sz, 3), dtype=np.uint8)
    mip = np.max(np.array(slices_list), axis=0)
    p99 = np.percentile(mip[mip > 0], 99) if (mip > 0).any() else 1
    mip_f = np.clip(mip / max(p99, 1), 0, 1)
    mip_resized = np.array(Image.fromarray((mip_f * 255).astype(np.uint8)).resize((patch_sz, patch_sz), Image.LANCZOS)).astype(np.float32) / 255.0
    # Apply color tint
    out = np.zeros((patch_sz, patch_sz, 3), dtype=np.uint8)
    for ch in range(3):
        out[:, :, ch] = np.clip(mip_resized * color[ch], 0, 255).astype(np.uint8)
    return out

VOL_CACHE = f'{BASE}/3d_viewer/vol_cache_v5.npz'
LM_CACHE  = f'{BASE}/3d_viewer/lm_cache_v5.pkl'

if os.path.exists(VOL_CACHE) and os.path.exists(LM_CACHE):
    print("Loading cached volumes...")
    _vc = np.load(VOL_CACHE)
    ex_vx, ex_vy, ex_vz, ex_vv = _vc['ex_vx'], _vc['ex_vy'], _vc['ex_vz'], _vc['ex_vv']
    iv_vx, iv_vy, iv_vz, iv_vv = _vc['iv_vx'], _vc['iv_vy'], _vc['iv_vz'], _vc['iv_vv']
    cal_vx, cal_vy, cal_vz, cal_vv = _vc['cal_vx'], _vc['cal_vy'], _vc['cal_vz'], _vc['cal_vv']
    merc_vx, merc_vy, merc_vz = _vc['merc_vx'], _vc['merc_vy'], _vc['merc_vz']
    merc_color_idx = _vc['merc_color_idx']
    merc_vv = np.ones(len(merc_vx), dtype=np.float32) * 0.8
    n_ex, n_iv, n_cal_vox, n_merc_total = len(ex_vx), len(iv_vx), len(cal_vx), len(merc_vx)
    span = float(_vc['span'])
    ex_cx, ex_cy, ex_cz = float(_vc['ex_centroid'][0]), float(_vc['ex_centroid'][1]), float(_vc['ex_centroid'][2])
    ex_nx, ex_ny, ex_nz = int(_vc['ex_shape'][0]), int(_vc['ex_shape'][1]), int(_vc['ex_shape'][2])
    gene_palette_js = json.dumps(GENE_PALETTE[:N_GENE_COLORS])
    with open(LM_CACHE, 'rb') as f:
        _lmc = pickle.load(f)
    all_landmarks = _lmc['all_landmarks']
    N_CELLS = _lmc['N_CELLS']
    N_FILTERED = _lmc['N_FILTERED']
    print(f"  Ex-vivo: {n_ex:,} | In-vivo: {n_iv:,} | Calcium: {n_cal_vox:,} | MERSCOPE: {n_merc_total:,}")
    print(f"  Landmarks: {N_CELLS} ({N_FILTERED} filtered)")
    _vol_cached = True
else:
    _vol_cached = False

# Always load stitch params (needed for patches + HTML)
print("Loading stitch params (v5)...")
with open(f'{BASE}/registration_video/stitch_v5_params.json') as f:
    params = json.load(f)
TILE_ORDER = params['tile_order']
tile_z_offsets = params['tile_z_offsets']
canvas_w = params['canvas_w']
canvas_h = params['canvas_h']
cum_iou = params['cumulative_iou']
STITCH_ELASTIX_DIR = params['elastix_dir']
total_z_native = max(tile_z_offsets.values()) + 12

with open(f'{BASE}/registration_video/auto_align_transforms_iou_v4.json') as f:
    iou_transforms = json.load(f)

# Always build tile_lm_files (needed for patches even when volumes are cached)
_lm_files = sorted(glob.glob(f'{BASE}/registration_video/landmarks_nd2_native_*.npz'))
_legacy = f'{BASE}/registration_video/landmarks_27_nd2_native.npz'
if os.path.exists(_legacy):
    _lm_files.append(_legacy)
tile_lm_files = {}
for lm_file in _lm_files:
    bn = os.path.basename(lm_file)
    if 'landmarks_27_nd2_native' in bn:
        tile = 'row2_1'
    else:
        tile = bn.replace('landmarks_nd2_native_', '').replace('.npz', '')
    if tile in TILE_ORDER:
        tile_lm_files[tile] = lm_file

if not _vol_cached:

    tile_idx_map = {t: i for i, t in enumerate(TILE_ORDER)}

    # ============================================================
    # 2. Load in-vivo
    # ============================================================
    print("Loading JY306 in-vivo...")
    iv_vol_raw = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
    nz_iv, ny_iv, nx_iv = iv_vol_raw.shape
    print(f"  In-vivo: {iv_vol_raw.shape}")

    # Per-z centroid drift correction for in-vivo stack
    # The JY306 stack has inter-slice tissue drift (~13µm Y, ~33µm X over z)
    # which propagates through the constant M2d into the 3D viewer
    print("  Computing per-z centroid drift...")
    iv_centroids = []
    for zi in range(nz_iv):
        sl = iv_vol_raw[zi]
        total = sl.sum()
        if total < 1e3:
            iv_centroids.append((np.nan, np.nan))
        else:
            ys, xs = np.mgrid[:sl.shape[0], :sl.shape[1]]
            iv_centroids.append(((ys * sl).sum() / total, (xs * sl).sum() / total))
    good = [(cy, cx) for z, (cy, cx) in enumerate(iv_centroids)
            if 2 <= z <= 13 and not np.isnan(cy)]
    iv_ref_cy = np.mean([c[0] for c in good])
    iv_ref_cx = np.mean([c[1] for c in good])
    iv_z_drift = {}  # z -> (dy, dx) offset from reference
    for zi in range(nz_iv):
        cy, cx = iv_centroids[zi]
        if np.isnan(cy):
            iv_z_drift[zi] = (0.0, 0.0)
        else:
            iv_z_drift[zi] = (cy - iv_ref_cy, cx - iv_ref_cx)
    print(f"    ref centroid: y={iv_ref_cy:.1f} x={iv_ref_cx:.1f}")
    print(f"    max drift: dy={max(abs(d[0]) for d in iv_z_drift.values()):.1f}px "
          f"dx={max(abs(d[1]) for d in iv_z_drift.values()):.1f}px")

    # ============================================================
    # 3. Load landmark files
    # ============================================================
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
    # 4. Per-tile: load pkl transform, warp in-vivo, build landmarks
    # ============================================================
    print("\nProcessing tiles (pkl direct)...")

    ds_w = canvas_w // DS
    ds_h = canvas_h // DS
    warped_vol = np.zeros((total_z_native, ds_h, ds_w), dtype=np.float32)

    all_landmarks = []

    for tile in sorted(tile_lm_files.keys()):
        print(f"\n  {tile}:")

        # Load pkl transform
        pkl_tfm_path = f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz'
        if not os.path.exists(pkl_tfm_path):
            print(f"    SKIP: no pkl transform")
            continue
        pkl_tfm = np.load(pkl_tfm_path)
        M2d = pkl_tfm['M2d_jy306_to_nd2']  # (2,3) maps JY306 xy → nd2 xy

        # Load landmarks
        d = np.load(tile_lm_files[tile])
        ev_nd2 = d['ev_nd2']
        iv_nd2 = d['iv_nd2']
        pcd_iv = d['pcd_invivo_jy306']
        N_LM = ev_nd2.shape[0]
        if N_LM < 3:
            print(f"    SKIP: only {N_LM} landmarks")
            continue

        # Load stitched v5 coords
        stitched_path = f'{BASE}/registration_video/landmarks_stitched_v5_{tile}.npz'
        if os.path.exists(stitched_path):
            sc = np.load(stitched_path)
            stitched_coords = sc['stitched_coords']  # (N,3) [z_um, y_um, x_um] in 1µm iso
        else:
            # Fallback: compute from cum_iou (less accurate but functional)
            print(f"    WARNING: no stitched_v5 coords, using cum_iou fallback")
            M_cum = np.array(cum_iou[tile])
            stitched_coords = np.zeros((N_LM, 3))
            z_offset = tile_z_offsets[tile]
            for i in range(N_LM):
                nd2_z_g = find_z_gaussian([0]*12)  # placeholder
                canvas_pt = M_cum[:2, :] @ np.array([ev_nd2[i, 0], ev_nd2[i, 1], 1])
                stitched_coords[i] = [z_offset * ND2_Z_UM, canvas_pt[1] * ND2_XY_UM, canvas_pt[0] * ND2_XY_UM]

        # Load nd2 slices for Gaussian z fitting
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

        # Gaussian z per landmark
        nd2_z_vals = []
        for i in range(N_LM):
            x, y = ev_nd2[i, 0], ev_nd2[i, 1]
            c = int(round(np.clip(x, 10, nd2_h - 11)))
            r = int(round(np.clip(y, 10, nd2_h - 11)))
            intensities = [nd2_slices[z][r-10:r+10, c-10:c+10].mean() for z in range(12)]
            nd2_z_vals.append(find_z_gaussian(intensities))

        # PKL 2D error
        pkl_dist_um = pkl_tfm['pkl_dist_um']
        print(f"    {N_LM} lm | pkl err: {pkl_dist_um.mean():.1f}µm")

        # Build z_nd2 → z_iv mapping from landmarks
        z_nd2_to_z_iv = {}
        for i in range(N_LM):
            z_nd2_i = int(round(np.clip(nd2_z_vals[i], 0, 11)))
            z_iv_i = int(round(pcd_iv[i, 0]))
            if z_nd2_i not in z_nd2_to_z_iv:
                z_nd2_to_z_iv[z_nd2_i] = []
            z_nd2_to_z_iv[z_nd2_i].append(z_iv_i)
        # Median z_iv per z_nd2
        z_mapping = {}
        for z_nd2, z_ivs in z_nd2_to_z_iv.items():
            z_mapping[z_nd2] = int(round(np.median(z_ivs)))

        z_offset = tile_z_offsets[tile]

        # Warp each nd2 z-slice using pkl 2D affine
        for z_nd2 in range(12):
            if z_nd2 not in z_mapping:
                # Interpolate from nearest mapped z
                nearest = min(z_mapping.keys(), key=lambda k: abs(k - z_nd2), default=None)
                if nearest is None:
                    continue
                z_iv = z_mapping[nearest]
            else:
                z_iv = z_mapping[z_nd2]
            z_iv = np.clip(z_iv, 0, nz_iv - 1)

            # Warp in-vivo using pkl M2d (JY306 → nd2, cv2 inverts internally)
            iv_warped = cv2.warpAffine(iv_vol_raw[z_iv], M2d, (nd2_w, nd2_h),
                                        flags=cv2.INTER_LINEAR, borderValue=0)

            # Full v5 stitching: pair IOU rigid + pair elastix + cumulative placement
            iv_stitched = stitch_tile_v5(iv_warped.astype(np.float32), tile,
                                          tile_idx_map, iou_transforms, cum_iou,
                                          STITCH_ELASTIX_DIR, TILE_ORDER, canvas_w, canvas_h)
            iv_ds = cv2.resize(iv_stitched, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
            z_out = z_offset + z_nd2
            warped_vol[z_out] = np.maximum(warped_vol[z_out], iv_ds)
            print(f"    pkl warp z_nd2={z_nd2} z_iv={z_iv}")

        # Blob-distance filter using pkl M2d
        n_kept = 0
        for i in range(N_LM):
            z_nd2_i = int(round(np.clip(nd2_z_vals[i], 0, 11)))

            # Create Gaussian blob at JY306 position
            bimg = np.zeros((ny_iv, nx_iv), dtype=np.float32)
            ly = int(round(pcd_iv[i, 1]))
            lx = int(round(pcd_iv[i, 2]))
            for dy in range(-8, 9):
                for dx in range(-8, 9):
                    yy, xx = ly + dy, lx + dx
                    if 0 <= yy < ny_iv and 0 <= xx < nx_iv:
                        bimg[yy, xx] = 255.0 * np.exp(-0.5 * (dy*dy + dx*dx) / 9.0)

            # Warp blob through pkl M2d
            blob_warped = cv2.warpAffine(bimg, M2d, (nd2_w, nd2_h),
                                          flags=cv2.INTER_LINEAR, borderValue=0)

            cx = ev_nd2[i, 0]
            cy = ev_nd2[i, 1]
            bmax = blob_warped.max()
            if bmax > 1:
                byy, bxx = np.where(blob_warped > bmax * 0.1)
                if len(bxx) > 0:
                    ww = blob_warped[byy, bxx]
                    blob_cx = float(np.average(bxx, weights=ww))
                    blob_cy = float(np.average(byy, weights=ww))
                    blob_dist_um = np.sqrt((blob_cx - cx)**2 + (blob_cy - cy)**2) * ND2_XY_UM
                else:
                    blob_dist_um = 999.0
            else:
                blob_dist_um = 999.0

            is_filtered = blob_dist_um <= WARP_ERR_MAX_UM
            if is_filtered:
                n_kept += 1

            st_z_um = stitched_coords[i, 0]
            st_y_um = stitched_coords[i, 1]
            st_x_um = stitched_coords[i, 2]

            all_landmarks.append({
                'tile': tile,
                'idx': i,
                'ev_nd2': ev_nd2[i],
                'iv_nd2': iv_nd2[i],
                'pcd_iv': pcd_iv[i],
                'z_nd2': z_nd2_i,
                'z_nd2_gauss': nd2_z_vals[i],
                'st_x_um': st_x_um,
                'st_y_um': st_y_um,
                'st_z_um': st_z_um,
                'err_um': float(pkl_dist_um[i]),
                'filtered': is_filtered,
            })
        print(f"    blob filter: {n_kept}/{N_LM} kept (threshold={WARP_ERR_MAX_UM}µm)")

        del nd2_slices

    N_CELLS = len(all_landmarks)
    N_FILTERED = sum(1 for lm in all_landmarks if lm['filtered'])
    print(f"\nTotal landmarks: {N_CELLS} ({N_FILTERED} filtered)")
    print(f"Warped volume: {warped_vol.shape}, non-zero slices: {np.sum(warped_vol.max(axis=(1,2)) > 0)}")

    # ============================================================
    # 5. Load stitched ex-vivo (downsampled)
    # ============================================================
    print("\nLoading stitched ex-vivo...")
    EX_TIFF = f"{BASE}/registration_video/stitched/stitched_gfp_fullres_v5_1um_isotropic.tif"
    with tifffile.TiffFile(EX_TIFF) as tif:
        ex_nz_full = len(tif.pages)
        ex_h_full, ex_w_full = tif.pages[0].shape
        ex_nz = ex_nz_full // DS_EX
        ex_ny = ex_h_full // DS_EX
        ex_nx = ex_w_full // DS_EX
        print(f"  Full: ({ex_nz_full}, {ex_h_full}, {ex_w_full}) -> DS{DS_EX}: ({ex_nz}, {ex_ny}, {ex_nx})")
        ex_vol = np.zeros((ex_nz, ex_ny, ex_nx), dtype=np.float32)
        for zi in range(ex_nz):
            sl = tif.pages[zi * DS_EX].asarray().astype(np.float32)
            ex_vol[zi] = sl[::DS_EX, ::DS_EX][:ex_ny, :ex_nx]

    ex_p99 = np.percentile(ex_vol[ex_vol > 0], 99.5)
    print(f"  Ex-vivo p99.5 = {ex_p99:.1f}")
    ex_u8 = np.clip(ex_vol / ex_p99 * 255, 0, 255).astype(np.uint8)
    del ex_vol

    # ============================================================
    # 6. Resample warped in-vivo to match ex-vivo grid
    # ============================================================
    print("Resampling warped in-vivo to ex-vivo grid...")
    iv_resized = np.zeros((ex_nz, ex_ny, ex_nx), dtype=np.float32)
    for z_ex in range(ex_nz):
        z_um = z_ex * DS_EX
        z_native = z_um / ND2_Z_UM
        z_int = int(round(z_native))
        if 0 <= z_int < total_z_native:
            sl = warped_vol[z_int]
            if sl.max() > 0:
                iv_resized[z_ex] = cv2.resize(sl, (ex_nx, ex_ny), interpolation=cv2.INTER_LINEAR)

    iv_pos = iv_resized[iv_resized > 0]
    if len(iv_pos) > 100:
        iv_p99 = np.percentile(iv_pos, 99)
        iv_norm = np.clip(iv_resized / max(iv_p99, 1) * 255, 0, 255)
    else:
        iv_norm = iv_resized.copy()

    print("  Median filter bg subtraction on warped in-vivo...")
    iv_sub = np.zeros_like(iv_norm)
    for z in range(ex_nz):
        if iv_norm[z].max() > 0:
            bg = median_filter(iv_norm[z], size=15)
            iv_sub[z] = np.clip(iv_norm[z] - bg, 0, 255)
    iv_u8 = iv_sub.astype(np.uint8)
    del warped_vol, iv_resized, iv_norm, iv_sub

    # ============================================================
    # 7. Extract sparse voxels
    # ============================================================
    print("Extracting sparse voxels...")
    MAX_EX_VOX = 800000
    ez, ey, exx = np.where(ex_u8 > VOXEL_THRESH_EX)
    ex_vals = ex_u8[ez, ey, exx]
    n_ex = len(ez)
    print(f"  Ex-vivo raw voxels: {n_ex:,}")
    if n_ex > MAX_EX_VOX:
        idx = np.argsort(ex_vals)[-MAX_EX_VOX:]
        ez, ey, exx, ex_vals = ez[idx], ey[idx], exx[idx], ex_vals[idx]
        n_ex = MAX_EX_VOX
        print(f"  Capped to brightest {MAX_EX_VOX:,}")

    ex_mask = ex_u8 > VOXEL_THRESH_EX
    ex_any_y = ex_mask.any(axis=(0, 2))
    ex_any_x = ex_mask.any(axis=(0, 1))
    ey_min, ey_max = np.where(ex_any_y)[0][[0, -1]]
    exx_min, exx_max = np.where(ex_any_x)[0][[0, -1]]
    ez_min, ez_max = np.where(ex_mask.any(axis=(1, 2)))[0][[0, -1]]
    PAD = 5
    ey_min, ey_max = max(0, ey_min - PAD), min(ex_ny - 1, ey_max + PAD)
    exx_min, exx_max = max(0, exx_min - PAD), min(ex_nx - 1, exx_max + PAD)
    ez_min, ez_max = max(0, ez_min - PAD), min(ex_nz - 1, ez_max + PAD)
    iv_u8[:ez_min] = 0
    iv_u8[ez_max+1:] = 0
    iv_u8[:, :ey_min, :] = 0
    iv_u8[:, ey_max+1:, :] = 0
    iv_u8[:, :, :exx_min] = 0
    iv_u8[:, :, exx_max+1:] = 0
    print(f"  IV bbox clip: z=[{ez_min},{ez_max}] y=[{ey_min},{ey_max}] x=[{exx_min},{exx_max}]")

    # ============================================================
    # 6b. Calcium volume (warped to stitched space)
    # ============================================================
    print("\nBuilding calcium volume in stitched space...")
    avi_path = os.path.join(BASE, 'movie_rolling_avg_win12_step3_short.avi')
    cap = cv2.VideoCapture(avi_path)
    cal_frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        cal_frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr)
    cap.release()
    cal_movie = np.array(cal_frames, dtype=np.float32)
    n_cal_frames = len(cal_movie)
    cal_std = cal_movie.std(axis=0)
    del cal_movie
    print(f"  {n_cal_frames} frames, cal_std shape: {cal_std.shape}")

    # Load movie→JY306 affine
    M_m2j = np.load(f'{BASE}/animation/movie_avi_to_jy306_affine.npz')['M_affine']  # 2×3

    # Warp cal_std per tile: movie→JY306→nd2→stitched
    cal_vol = np.zeros((total_z_native, ds_h, ds_w), dtype=np.float32)
    for tile in sorted(tile_lm_files.keys()):
        pkl_path = f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz'
        if not os.path.exists(pkl_path):
            continue
        M2d = np.load(pkl_path)['M2d_jy306_to_nd2']
        M_j2n = np.vstack([M2d, [0, 0, 1]])
        M_m2j_h = np.vstack([M_m2j, [0, 0, 1]])
        M_movie_to_nd2 = (M_j2n @ M_m2j_h)[:2, :]
        cal_nd2 = cv2.warpAffine(cal_std, M_movie_to_nd2, (4200, 4200), borderValue=0)
        z_offset = tile_z_offsets[tile]
        for z_nd2 in range(12):
            z_out = z_offset + z_nd2
            if z_out >= total_z_native:
                continue
            cal_stitched = stitch_tile_v5(cal_nd2.astype(np.float32), tile,
                                           tile_idx_map, iou_transforms, cum_iou,
                                           STITCH_ELASTIX_DIR, TILE_ORDER, canvas_w, canvas_h)
            cal_ds = cv2.resize(cal_stitched, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
            cal_vol[z_out] = np.maximum(cal_vol[z_out], cal_ds)
        print(f"  {tile} calcium stitched")

    # Resample to ex-vivo grid
    cal_resized = np.zeros((ex_nz, ex_ny, ex_nx), dtype=np.float32)
    for z_ex in range(ex_nz):
        z_um = z_ex * DS_EX
        z_native = z_um / ND2_Z_UM
        z_int = int(round(z_native))
        if 0 <= z_int < total_z_native:
            sl = cal_vol[z_int]
            if sl.max() > 0:
                cal_resized[z_ex] = cv2.resize(sl, (ex_nx, ex_ny), interpolation=cv2.INTER_LINEAR)
    cal_pos = cal_resized[cal_resized > 0]
    if len(cal_pos) > 100:
        cal_p99 = np.percentile(cal_pos, 99)
        cal_u8 = np.clip(cal_resized / max(cal_p99, 1) * 255, 0, 255).astype(np.uint8)
    else:
        cal_u8 = cal_resized.astype(np.uint8)
    cal_u8[:ez_min] = 0; cal_u8[ez_max+1:] = 0
    cal_u8[:, :ey_min, :] = 0; cal_u8[:, ey_max+1:, :] = 0
    cal_u8[:, :, :exx_min] = 0; cal_u8[:, :, exx_max+1:] = 0
    del cal_vol

    # ============================================================
    # 6c. MERSCOPE dots (warped to stitched space)
    # ============================================================
    print("\nLoading MERSCOPE transcripts for stitched space...")
    all_merc_x, all_merc_y, all_merc_z, all_merc_genes = [], [], [], []
    gene_counter = Counter()
    via_masks_path = f'{BASE}/registration_video/via_masks_v4.npz'
    via_masks = np.load(via_masks_path) if os.path.exists(via_masks_path) else {}

    for tile in sorted(TILE_TO_REGION.keys()):
        region_id = TILE_TO_REGION[tile]
        csv_path = f'{VAROL}/region_{region_id}_resegmentation/detected_transcripts.csv'
        mnf_path = f'{VAROL}/region_{region_id}_resegmentation/manifest.json'
        m2m_path = f'{VAROL}/region_{region_id}_resegmentation/micron_to_mosaic_pixel_transform.csv'
        pkl_f = None
        if os.path.isdir(PKL_MERC_DIR):
            for fname in os.listdir(PKL_MERC_DIR):
                if fname.endswith('.pkl'):
                    mm = re.match(r'[\d_]+_reg(\d+)_transformed.*\.pkl', fname)
                    if mm and int(mm.group(1)) == region_id:
                        if pkl_f is None or fname > pkl_f:
                            pkl_f = fname
        if not pkl_f or not all(os.path.exists(p) for p in [csv_path, mnf_path, m2m_path]):
            print(f"  SKIP {tile} (region {region_id}): missing files")
            continue
        m2m = np.loadtxt(m2m_path, delimiter=' ')
        sc_m, tx_m, ty_m = m2m[0, 0], m2m[0, 2], m2m[1, 2]
        with open(mnf_path) as f:
            mnf = json.load(f)
        W_mos = mnf['mosaic_width_pixels']
        with open(f'{PKL_MERC_DIR}/{pkl_f}', 'rb') as f:
            pdat = pickle.load(f)
        R3i, off3 = build_pkl_affine(pdat['transformations'])
        tif_sz = pdat['transformed'].shape[-1]
        nd2_sc = 4200 / tif_sz

        df = pd.read_csv(csv_path, usecols=['global_x', 'global_y', 'gene'])
        df = df[~df['gene'].str.startswith('Blank')]
        gx, gy = df.global_x.values, df.global_y.values
        xm = sc_m * gx + tx_m
        ym = sc_m * gy + ty_m
        mx = (W_mos - 1 - xm) * 0.108
        my = ym * 0.108
        ay = my - off3[1]; ax = mx - off3[2]
        nd2_x = (R3i[2, 1] * ay + R3i[2, 2] * ax) * nd2_sc
        nd2_y = (R3i[1, 1] * ay + R3i[1, 2] * ax) * nd2_sc

        # Apply tissue mask
        if hasattr(via_masks, '__contains__') and tile in via_masks:
            tile_mask = via_masks[tile]
        else:
            tile_mask = np.ones((4200, 4200), np.uint8)
        ix_int = np.clip(nd2_x.astype(int), 0, 4199)
        iy_int = np.clip(nd2_y.astype(int), 0, 4199)
        mask_vals = tile_mask[iy_int, ix_int]
        keep = mask_vals > 0
        nd2_x, nd2_y = nd2_x[keep], nd2_y[keep]
        genes_kept = df.gene.values[keep]

        # Transform nd2 coords to stitched canvas
        M_cum = np.array(cum_iou[tile])
        ones = np.ones(len(nd2_x))
        pts = np.vstack([nd2_x, nd2_y, ones])  # (3, N)
        canvas_pts = M_cum @ pts  # (3, N)
        # Convert from native canvas pixels to 1µm isotropic coords, then to DS4 grid
        canvas_x = canvas_pts[0] * ND2_XY_UM / DS_EX
        canvas_y = canvas_pts[1] * ND2_XY_UM / DS_EX

        z_offset = tile_z_offsets[tile]
        z_um = (z_offset + 6) * ND2_Z_UM  # middle of tile
        canvas_z = z_um / DS_EX  # match ex-vivo z scaling

        n_dots = len(canvas_x)
        gene_counter.update(genes_kept)
        all_merc_x.append(canvas_x.astype(np.float32))
        all_merc_y.append(canvas_y.astype(np.float32))
        all_merc_z.append(np.full(n_dots, canvas_z, dtype=np.float32))
        all_merc_genes.append(genes_kept)
        print(f"  {tile} (region {region_id}): {n_dots:,} dots")

    merc_x = np.concatenate(all_merc_x) if all_merc_x else np.zeros(0, dtype=np.float32)
    merc_y = np.concatenate(all_merc_y) if all_merc_y else np.zeros(0, dtype=np.float32)
    merc_z = np.concatenate(all_merc_z) if all_merc_z else np.zeros(0, dtype=np.float32)
    merc_genes = np.concatenate(all_merc_genes) if all_merc_genes else np.array([], dtype=object)
    n_merc_total = len(merc_x)
    print(f"  Total: {n_merc_total:,} dots, {len(gene_counter)} genes")

    # Subsample
    if n_merc_total > MAX_MERC:
        rng = np.random.default_rng(42)
        sel = rng.choice(n_merc_total, MAX_MERC, replace=False)
        sel.sort()
        merc_x, merc_y, merc_z = merc_x[sel], merc_y[sel], merc_z[sel]
        merc_genes = merc_genes[sel]
        n_merc_total = MAX_MERC
        print(f"  Subsampled to {n_merc_total:,}")

    # Per-gene consistent colors
    all_genes_sorted = [g for g, _ in gene_counter.most_common()]
    gene_to_color_idx = {g: i % N_GENE_COLORS for i, g in enumerate(all_genes_sorted)}
    if n_merc_total > 0:
        merc_color_idx = np.array([gene_to_color_idx[g] for g in merc_genes], dtype=np.float32)
    else:
        merc_color_idx = np.zeros(0, dtype=np.float32)
    merc_vv = np.ones(n_merc_total, dtype=np.float32) * 0.8

    gene_palette_js = json.dumps(GENE_PALETTE[:N_GENE_COLORS])
    del all_merc_x, all_merc_y, all_merc_z, all_merc_genes, merc_genes

    # Mask IV to ex-vivo tissue region: bbox + dilated tissue mask
    from scipy.ndimage import binary_dilation, binary_closing
    # Close small holes in ex-vivo mask, then dilate conservatively
    ex_tissue = binary_closing(ex_mask, iterations=3)
    ex_tissue = binary_dilation(ex_tissue, iterations=3)
    iv_u8[~ex_tissue] = 0
    iz, iy, ix = np.where(iv_u8 > VOXEL_THRESH_IV)
    iv_vals_arr = iv_u8[iz, iy, ix]
    n_iv = len(iz)
    print(f"  IV voxels after tissue mask: {n_iv:,}")

    VOXEL_THRESH_CAL = 15
    MAX_CAL_VOX = 400000
    cal_u8[~ex_tissue] = 0
    cz, cy_cal, cx_cal = np.where(cal_u8 > VOXEL_THRESH_CAL)
    cal_vals = cal_resized[cz, cy_cal, cx_cal]  # use float values, not u8
    if len(cz) > MAX_CAL_VOX:
        idx = np.argsort(cal_vals)[-MAX_CAL_VOX:]  # keep brightest
        cz, cy_cal, cx_cal, cal_vals = cz[idx], cy_cal[idx], cx_cal[idx], cal_vals[idx]
    n_cal_vox = len(cz)
    print(f"  Ex-vivo: {n_ex:,} | In-vivo warped: {n_iv:,} | Calcium: {n_cal_vox:,} | MERSCOPE: {n_merc_total:,}")

    span = float(max(ex_nx, ex_ny, ex_nz))
    ex_vx = exx.astype(np.float32) / span
    ex_vy = ey.astype(np.float32) / span
    ex_vz = ez.astype(np.float32) / span
    ex_cx, ex_cy, ex_cz = ex_vx.mean(), ex_vy.mean(), ex_vz.mean()

    iv_vx = ix.astype(np.float32) / span
    iv_vy = iy.astype(np.float32) / span
    iv_vz = iz.astype(np.float32) / span

    cal_vx = cx_cal.astype(np.float32) / span
    cal_vy = cy_cal.astype(np.float32) / span
    cal_vz = cz.astype(np.float32) / span

    ex_vx += (0.5 - ex_cx); ex_vy += (0.5 - ex_cy); ex_vz += (0.5 - ex_cz)
    iv_vx += (0.5 - ex_cx); iv_vy += (0.5 - ex_cy); iv_vz += (0.5 - ex_cz)
    cal_vx += (0.5 - ex_cx); cal_vy += (0.5 - ex_cy); cal_vz += (0.5 - ex_cz)

    # Normalize MERSCOPE to same space
    merc_vx = merc_x / span + (0.5 - ex_cx)
    merc_vy = merc_y / span + (0.5 - ex_cy)
    merc_vz = merc_z / span + (0.5 - ex_cz)

    ex_vv = ex_vals.astype(np.float32) / 255.0
    iv_vv = iv_vals_arr.astype(np.float32) / 255.0
    # cal_vals are already float from cal_resized, normalize to 0-1
    cal_vv = cal_vals.astype(np.float32)
    cal_vv_max = cal_vv.max() if cal_vv.max() > 0 else 1.0
    cal_vv = cal_vv / cal_vv_max
    del ex_u8, iv_u8, cal_u8

    # Save volume cache
    print("Saving volume cache...")
    np.savez_compressed(VOL_CACHE,
        ex_vx=ex_vx, ex_vy=ex_vy, ex_vz=ex_vz, ex_vv=ex_vv,
        iv_vx=iv_vx, iv_vy=iv_vy, iv_vz=iv_vz, iv_vv=iv_vv,
        cal_vx=cal_vx, cal_vy=cal_vy, cal_vz=cal_vz, cal_vv=cal_vv,
        merc_vx=merc_vx, merc_vy=merc_vy, merc_vz=merc_vz, merc_color_idx=merc_color_idx,
        span=np.array([span]), ex_centroid=np.array([ex_cx, ex_cy, ex_cz]),
        ex_shape=np.array([ex_nx, ex_ny, ex_nz]))
    with open(LM_CACHE, 'wb') as f:
        pickle.dump({
            'all_landmarks': all_landmarks,
            'N_CELLS': N_CELLS, 'N_FILTERED': N_FILTERED,
        }, f)
    print(f"  Cached to {VOL_CACHE}")

# ============================================================
# 8. Compute landmark positions in normalized 3D space
# ============================================================
print("Computing landmark 3D positions...")

landmarks_js = []
cell_tiles = []
cell_filtered = []
for lm in all_landmarks:
    tif_x = lm['st_x_um'] / DS_EX
    tif_y = lm['st_y_um'] / DS_EX
    tif_z = lm['st_z_um'] / DS_EX

    nx = tif_x / span + (0.5 - ex_cx)
    ny = tif_y / span + (0.5 - ex_cy)
    nz = tif_z / span + (0.5 - ex_cz)
    landmarks_js.append(f'[{nx:.5f},{ny:.5f},{nz:.5f}]')
    cell_tiles.append(lm['tile'])
    cell_filtered.append(bool(lm['filtered']))

# ============================================================
# 9. Generate patch strips (4 columns: EV magenta, IV warped green, Calcium magenta, MERSCOPE rainbow)
# ============================================================
PATCH_CACHE = f'{BASE}/3d_viewer/patch_strip_v5.png'
CELL_INFO_CACHE = f'{BASE}/3d_viewer/cell_info_v5.json'
_use_cache = os.path.exists(PATCH_CACHE) and os.path.exists(CELL_INFO_CACHE)

if _use_cache:
    print("Loading cached patch strip...")
    with open(PATCH_CACHE, 'rb') as f:
        patch_strip_b64 = base64.b64encode(f.read()).decode('ascii')
    with open(CELL_INFO_CACHE, 'r') as f:
        cell_info_js = json.load(f)
    print(f"  Cached strip: {len(patch_strip_b64)//1024}KB, {len(cell_info_js)} entries")

patch_tiles = set(lm['tile'] for lm in all_landmarks)

if not _use_cache:
    # Load raw data needed for patches if volumes were cached
    if _vol_cached:
        print("Loading raw data for patch generation...")
        iv_vol_raw = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
        nz_iv = iv_vol_raw.shape[0]
        M_m2j_data = np.load(f'{BASE}/animation/movie_avi_to_jy306_affine.npz')
        M_m2j = M_m2j_data['M_affine']
        cal_movie = cv2.VideoCapture(f'{BASE}/movie_rolling_avg_win12_step3_short.avi')
        frames = []
        while True:
            ret, fr = cal_movie.read()
            if not ret: break
            frames.append(fr[:,:,0].astype(np.float32))
        cal_movie.release()
        cal_std = np.std(np.array(frames), axis=0).astype(np.float32)
        del frames
        # Build all_genes_sorted from MERSCOPE CSVs
        from collections import Counter
        gene_counter = Counter()
        for tile, region_id in TILE_TO_REGION.items():
            csv_path = f'{VAROL}/region_{region_id}_resegmentation/detected_transcripts.csv'
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, usecols=['gene'])
                df = df[~df['gene'].str.startswith('Blank')]
                gene_counter.update(df['gene'].values)
        all_genes_sorted = [g for g, _ in gene_counter.most_common()]
        gene_to_color_idx = {g: i % N_GENE_COLORS for i, g in enumerate(all_genes_sorted)}
        print(f"  Loaded {len(all_genes_sorted)} genes, iv shape {iv_vol_raw.shape}")

    print("Generating patch strips...")

    # Pre-load nd2 pages for ex-vivo patches
    print("  Loading nd2 PNGs for patches...")
    nd2_pages = {}
    for tile in patch_tiles:
        tile_dir = f'{BASE}/png_exports/registration_video/{tile}'
        for zi in range(12):
            png_path = f'{tile_dir}/GFP_z{zi:03d}.png'
            nd2_pages[(tile, zi)] = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)

    # Load pkl M2d per tile for warped patches
    print("  Loading pkl transforms for warped patches...")
    tile_pkl_m2d = {}
    for tile in sorted(patch_tiles):
        pkl_path = f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz'
        if os.path.exists(pkl_path):
            td = np.load(pkl_path)
            tile_pkl_m2d[tile] = td['M2d_jy306_to_nd2']

    # Build z-mapping per tile for patches
    tile_z_mappings = {}
    for tile in sorted(patch_tiles):
        d = np.load(tile_lm_files[tile])
        ev = d['ev_nd2']
        pcd = d['pcd_invivo_jy306']
        n = ev.shape[0]
        if n < 3:
            continue
        z_map = {}
        for i in range(n):
            c = int(round(np.clip(ev[i, 0], 10, 4189)))
            r = int(round(np.clip(ev[i, 1], 10, 4189)))
            intensities = [nd2_pages[(tile, z)][r-10:r+10, c-10:c+10].astype(np.float32).mean() for z in range(12)]
            z_nd2_i = int(round(np.clip(find_z_gaussian(intensities), 0, 11)))
            z_iv_i = int(round(pcd[i, 0]))
            if z_nd2_i not in z_map:
                z_map[z_nd2_i] = []
            z_map[z_nd2_i].append(z_iv_i)
        tile_z_mappings[tile] = {k: int(round(np.median(v))) for k, v in z_map.items()}

    # Warped iv slice cache per (tile, z_nd2)
    warped_cache = {}

    def get_warped_iv(tile, z_nd2, nd2_y=2100, nd2_x=2100):
        """Get warped in-vivo slice using pkl M2d."""
        if tile not in tile_pkl_m2d:
            return None
        if tile not in tile_z_mappings:
            return None
        z_map = tile_z_mappings[tile]
        if z_nd2 not in z_map:
            nearest = min(z_map.keys(), key=lambda k: abs(k - z_nd2), default=None)
            if nearest is None:
                return None
            z_iv = z_map[nearest]
        else:
            z_iv = z_map[z_nd2]
        z_iv = np.clip(z_iv, 0, nz_iv - 1)
        key = (tile, z_nd2, z_iv)
        if key in warped_cache:
            return warped_cache[key]
        M2d = tile_pkl_m2d[tile]
        iv_w = cv2.warpAffine(iv_vol_raw[z_iv], M2d, (4200, 4200),
                               flags=cv2.INTER_LINEAR, borderValue=0)
        warped_cache[key] = iv_w
        return iv_w

    # Build patch strip: 4 columns
    # Col 0: EV (magenta) | Col 1: IV warped (green) | Col 2: Calcium (magenta) | Col 3: MERSCOPE dots
    print("  Building patch strip (3 columns: EV, IV, MERSCOPE)...")
    patch_strip_w = PATCH_SZ * 3
    patch_strip_h = PATCH_SZ * N_CELLS
    patch_strip = np.zeros((patch_strip_h, patch_strip_w, 3), dtype=np.uint8)

    cell_info_js = []

    # Pre-load MERSCOPE transform chains for patches
    print("  Pre-loading MERSCOPE data for patches...")
    merc_tile_data = {}
    for tile in patch_tiles:
        region_id = TILE_TO_REGION.get(tile)
        if region_id is None:
            continue
        csv_path = f'{VAROL}/region_{region_id}_resegmentation/detected_transcripts.csv'
        mnf_path = f'{VAROL}/region_{region_id}_resegmentation/manifest.json'
        m2m_path = f'{VAROL}/region_{region_id}_resegmentation/micron_to_mosaic_pixel_transform.csv'
        pkl_f = None
        if os.path.isdir(PKL_MERC_DIR):
            for fname in os.listdir(PKL_MERC_DIR):
                if fname.endswith('.pkl'):
                    mm = re.match(r'[\d_]+_reg(\d+)_transformed.*\.pkl', fname)
                    if mm and int(mm.group(1)) == region_id:
                        if pkl_f is None or fname > pkl_f:
                            pkl_f = fname
        if not pkl_f or not all(os.path.exists(p) for p in [csv_path, mnf_path, m2m_path]):
            continue
        m2m_data = np.loadtxt(m2m_path, delimiter=' ')
        sc_m_p, tx_m_p, ty_m_p = m2m_data[0, 0], m2m_data[0, 2], m2m_data[1, 2]
        with open(mnf_path) as f:
            mnf_data = json.load(f)
        W_mos_p = mnf_data['mosaic_width_pixels']
        with open(f'{PKL_MERC_DIR}/{pkl_f}', 'rb') as f:
            pdat_p = pickle.load(f)
        R3i_p, off3_p = build_pkl_affine(pdat_p['transformations'])
        tif_sz_p = pdat_p['transformed'].shape[-1]
        nd2_sc_p = 4200 / tif_sz_p
        merc_tile_data[tile] = {
            'csv': csv_path, 'sc_m': sc_m_p, 'tx_m': tx_m_p, 'ty_m': ty_m_p,
            'W_mos': W_mos_p, 'R3i': R3i_p, 'off3': off3_p, 'nd2_sc': nd2_sc_p,
        }

    # BGR color lookup for MERSCOPE dots
    gene_to_bgr = {}
    for gi, g in enumerate(all_genes_sorted):
        c = GENE_PALETTE[gi % N_GENE_COLORS]
        gene_to_bgr[g] = (c[2], c[1], c[0])  # RGB → BGR

    merc_nd2_cache = {}

    # Cache cal_std warped to nd2 per tile
    print("  Caching calcium→nd2 per tile...")
    cal_nd2_cache = {}
    for tile in sorted(patch_tiles):
        if tile in tile_pkl_m2d:
            M2d_t = tile_pkl_m2d[tile]
            M_j2n_t = np.vstack([M2d_t, [0, 0, 1]])
            M_m2j_h_t = np.vstack([M_m2j, [0, 0, 1]])
            M_movie_to_nd2_t = (M_j2n_t @ M_m2j_h_t)[:2, :]
            cal_nd2_cache[tile] = cv2.warpAffine(cal_std, M_movie_to_nd2_t, (4200, 4200), borderValue=0)
            print(f"    Cached calcium→nd2 for {tile}")

    for ci, lm in enumerate(all_landmarks):
        tile = lm['tile']
        col_c = int(round(lm['ev_nd2'][0]))
        row_c = int(round(lm['ev_nd2'][1]))
        z_nd2 = lm['z_nd2']

        # --- Col 0: Ex-vivo (magenta) — single z-slice ---
        row = ci * PATCH_SZ
        ex_patch = np.zeros((PATCH_SZ, PATCH_SZ, 3), dtype=np.uint8)
        if (tile, z_nd2) in nd2_pages and nd2_pages[(tile, z_nd2)] is not None:
            page = nd2_pages[(tile, z_nd2)]
            y0 = max(0, row_c - CROP_ND2); y1 = min(page.shape[0], row_c + CROP_ND2)
            x0 = max(0, col_c - CROP_ND2); x1 = min(page.shape[1], col_c + CROP_ND2)
            crop = page[y0:y1, x0:x1].astype(np.float32)
            p99 = np.percentile(crop[crop > 0], 99) if (crop > 0).any() else 1
            crop_n = np.clip(crop / max(p99, 1), 0, 1)
            crop_r = np.array(Image.fromarray((crop_n * 255).astype(np.uint8)).resize((PATCH_SZ, PATCH_SZ), Image.LANCZOS)).astype(np.float32) / 255.0
            ex_patch[:, :, 0] = np.clip(crop_r * 255, 0, 255).astype(np.uint8)
            ex_patch[:, :, 2] = np.clip(crop_r * 255, 0, 255).astype(np.uint8)
        patch_strip[row:row+PATCH_SZ, 0:PATCH_SZ] = ex_patch

        # --- Col 1: IV warped (green) — single z-slice ---
        iv_patch = np.zeros((PATCH_SZ, PATCH_SZ, 3), dtype=np.uint8)
        warp_sl = get_warped_iv(tile, z_nd2, nd2_y=row_c, nd2_x=col_c)
        if warp_sl is not None:
            y0 = max(0, row_c - CROP_ND2); y1 = min(4200, row_c + CROP_ND2)
            x0 = max(0, col_c - CROP_ND2); x1 = min(4200, col_c + CROP_ND2)
            crop = warp_sl[y0:y1, x0:x1].astype(np.float32)
            p99 = np.percentile(crop[crop > 0], 99) if (crop > 0).any() else 1
            crop_n = np.clip(crop / max(p99, 1), 0, 1)
            crop_r = np.array(Image.fromarray((crop_n * 255).astype(np.uint8)).resize((PATCH_SZ, PATCH_SZ), Image.LANCZOS)).astype(np.float32) / 255.0
            iv_patch[:, :, 1] = np.clip(crop_r * 255, 0, 255).astype(np.uint8)
        patch_strip[row:row+PATCH_SZ, PATCH_SZ:PATCH_SZ*2] = iv_patch

        # --- Col 2: MERSCOPE dots ---
        if tile in merc_tile_data:
            if tile not in merc_nd2_cache:
                td = merc_tile_data[tile]
                df_m = pd.read_csv(td['csv'], usecols=['global_x', 'global_y', 'gene'])
                df_m = df_m[~df_m['gene'].str.startswith('Blank')]
                gx_m, gy_m = df_m.global_x.values, df_m.global_y.values
                xm_m = td['sc_m'] * gx_m + td['tx_m']
                ym_m = td['sc_m'] * gy_m + td['ty_m']
                mx_v = (td['W_mos'] - 1 - xm_m) * 0.108
                my_v = ym_m * 0.108
                ay_m = my_v - td['off3'][1]; ax_m = mx_v - td['off3'][2]
                nd2_xd = (td['R3i'][2, 1] * ay_m + td['R3i'][2, 2] * ax_m) * td['nd2_sc']
                nd2_yd = (td['R3i'][1, 1] * ay_m + td['R3i'][1, 2] * ax_m) * td['nd2_sc']
                merc_nd2_cache[tile] = (nd2_xd, nd2_yd, df_m.gene.values)
            nd2_xd, nd2_yd, m_genes = merc_nd2_cache[tile]
            cr = CROP_ND2
            in_crop = ((nd2_xd >= col_c - cr) & (nd2_xd < col_c + cr) &
                       (nd2_yd >= row_c - cr) & (nd2_yd < row_c + cr))
            dot_canvas = np.zeros((cr * 2, cr * 2, 3), np.uint8)
            if in_crop.any():
                px = (nd2_xd[in_crop] - col_c + cr).astype(int)
                py = (nd2_yd[in_crop] - row_c + cr).astype(int)
                dot_g = m_genes[in_crop]
                valid = (px >= 0) & (px < cr*2) & (py >= 0) & (py < cr*2)
                for j in np.where(valid)[0]:
                    c_bgr = gene_to_bgr.get(dot_g[j], (255, 255, 255))
                    dot_canvas[py[j], px[j]] = (c_bgr[2], c_bgr[1], c_bgr[0])  # BGR→RGB
            gd_resized = cv2.resize(dot_canvas, (PATCH_SZ, PATCH_SZ), interpolation=cv2.INTER_NEAREST)
            patch_strip[row:row+PATCH_SZ, PATCH_SZ*2:PATCH_SZ*3] = gd_resized

        z_iv = int(round(lm['pcd_iv'][0]))
        ez_lo = max(0, z_nd2 - DZ_SLICES)
        ez_hi = min(11, z_nd2 + DZ_SLICES)
        ivz_lo = max(0, z_iv - DZ_SLICES)
        ivz_hi = min(15, z_iv + DZ_SLICES)
        CAL_BEST_Z = 3
        CAL_Z_TOL = 1
        cal_z_ok = 1 if abs(z_iv - CAL_BEST_Z) <= CAL_Z_TOL else 0
        cell_info_js.append(f'[{z_nd2},{ez_lo},{ez_hi},{z_iv},{ivz_lo},{ivz_hi},{cal_z_ok}]')

        if ci % 100 == 0:
            print(f"    patch {ci}/{N_CELLS}")

    warped_cache.clear()
    del nd2_pages

    print("  Encoding patch strip...")
    patch_img = Image.fromarray(patch_strip, 'RGB')
    buf = io.BytesIO()
    patch_img.save(buf, format='PNG', optimize=True)
    patch_strip_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    print(f"  Patch strip: {patch_strip_w}x{patch_strip_h}, {len(patch_strip_b64)//1024}KB")
    del patch_strip

    print('  Saving patch cache...')
    patch_img.save(PATCH_CACHE, format='PNG', optimize=True)
    with open(CELL_INFO_CACHE, 'w') as fc:
        json.dump(cell_info_js, fc)
    print(f'  Cached to {PATCH_CACHE}')


# ============================================================
# 9b. Calcium video strip (per-landmark, warped to nd2 space)
# ============================================================
CAL_STRIP_CACHE = f'{BASE}/3d_viewer/cal_vid_strip_v5.png'
if os.path.exists(CAL_STRIP_CACHE):
    print("Loading cached calcium video strip...")
    with open(CAL_STRIP_CACHE, 'rb') as f:
        calStripB64 = base64.b64encode(f.read()).decode('ascii')
    cap_tmp = cv2.VideoCapture(os.path.join(BASE, 'movie_rolling_avg_win12_step3_short.avi'))
    n_cal_frames_raw = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_tmp.release()
    CAL_STEP = 3
    n_cal_frames = len(range(0, n_cal_frames_raw, CAL_STEP))
    print(f"  Cached cal strip: {len(calStripB64)//1024}KB, {n_cal_frames} frames")
else:
    print("Building per-landmark calcium video strip (nd2 space)...")
    cap = cv2.VideoCapture(os.path.join(BASE, 'movie_rolling_avg_win12_step3_short.avi'))
    cal_mov_frames = []
    while True:
        ret, fr = cap.read()
        if not ret: break
        cal_mov_frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr)
    cap.release()
    n_cal_frames = len(cal_mov_frames)
    # Subsample to every 3rd frame to keep file size manageable
    CAL_STEP = 3
    cal_mov_frames = cal_mov_frames[::CAL_STEP]
    n_cal_frames = len(cal_mov_frames)
    cal_movie_u8 = np.array(cal_mov_frames, dtype=np.uint8)
    del cal_mov_frames
    print(f"  Subsampled to {n_cal_frames} frames (step={CAL_STEP})")

    # Build movie→nd2 transform per tile
    M_m2j_cal = np.load(f'{BASE}/animation/movie_avi_to_jy306_affine.npz')['M_affine']
    tile_M_m2nd2 = {}
    for tile in patch_tiles:
        pkl_path = f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz'
        if os.path.exists(pkl_path):
            M2d = np.load(pkl_path)['M2d_jy306_to_nd2']
            M_j2n = np.vstack([M2d, [0, 0, 1]])
            M_m2j_h = np.vstack([M_m2j_cal, [0, 0, 1]])
            tile_M_m2nd2[tile] = (M_j2n @ M_m2j_h)[:2, :]

    # Per-landmark: warp each frame to nd2, crop at ev_nd2 coords → CAL_PSZ×CAL_PSZ
    CAL_PSZ = 40  # smaller than PATCH_SZ to keep file size manageable
    cal_vid_strip = np.zeros((N_CELLS * n_cal_frames * CAL_PSZ, CAL_PSZ), dtype=np.uint8)

    # Cache warped frames per tile (avoid re-warping for same tile)
    warp_cache = {}
    for ci, lm in enumerate(all_landmarks):
        tile = lm['tile']
        col_c = int(round(lm['ev_nd2'][0]))
        row_c = int(round(lm['ev_nd2'][1]))
        if tile not in tile_M_m2nd2:
            continue
        M_t = tile_M_m2nd2[tile]
        # Warp all frames for this tile (cache)
        if tile not in warp_cache:
            print(f"  Warping {n_cal_frames} frames → nd2 for {tile}...")
            warped_frames = []
            for fi in range(n_cal_frames):
                w = cv2.warpAffine(cal_movie_u8[fi], M_t, (4200, 4200), borderValue=0)
                warped_frames.append(w)
            warp_cache[tile] = warped_frames
        warped_frames = warp_cache[tile]
        # Crop per frame
        y0 = max(0, row_c - CROP_ND2); y1 = min(4200, row_c + CROP_ND2)
        x0 = max(0, col_c - CROP_ND2); x1 = min(4200, col_c + CROP_ND2)
        for fi in range(n_cal_frames):
            crop = warped_frames[fi][y0:y1, x0:x1].astype(np.float32)
            if crop.size > 0 and crop.max() > 0:
                p99 = np.percentile(crop[crop > 0], 99) if (crop > 0).any() else 1
                crop_u8 = np.clip(crop / max(p99, 1) * 255, 0, 255).astype(np.uint8)
                resized = np.array(Image.fromarray(crop_u8).resize((CAL_PSZ, CAL_PSZ), Image.LANCZOS))
            else:
                resized = np.zeros((CAL_PSZ, CAL_PSZ), dtype=np.uint8)
            row_off = (ci * n_cal_frames + fi) * CAL_PSZ
            cal_vid_strip[row_off:row_off+CAL_PSZ, :] = resized
        if ci % 100 == 0:
            print(f"    cal patch {ci}/{N_CELLS}")
    del cal_movie_u8, warp_cache

    cal_vid_img = Image.fromarray(cal_vid_strip)
    buf_cal = io.BytesIO()
    cal_vid_img.save(buf_cal, format='PNG', optimize=True)
    calStripB64 = base64.b64encode(buf_cal.getvalue()).decode('ascii')
    cal_vid_img.save(CAL_STRIP_CACHE, format='PNG', optimize=True)
    del cal_vid_strip
    print(f"  Cal vid strip: {CAL_PSZ}x{N_CELLS * n_cal_frames * CAL_PSZ}, {len(calStripB64)//1024}KB")

# ============================================================
# 10. Build HTML
# ============================================================
print("\nBuilding HTML...")

unique_tiles = []
tile_ranges = {}
idx = 0
for tile in sorted(patch_tiles):
    count = sum(1 for lm in all_landmarks if lm['tile'] == tile)
    tile_ranges[tile] = [idx, idx + count]
    unique_tiles.append(tile)
    idx += count

tile_opts = '<option value="all">All ({0})</option>'.format(N_CELLS)
for t in unique_tiles:
    n = tile_ranges[t][1] - tile_ranges[t][0]
    tile_opts += f'<option value="{t}">{t} ({n})</option>'

SCALE = 4.0

html_parts = []
html_parts.append(f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>4-Modality 3D Viewer v5 — EV + IV + Calcium + MERSCOPE</title>
<style>
  body {{ margin:0; overflow:hidden; background:#000; color:#ddd; font-family:monospace; font-size:11px; }}
  #info {{ position:absolute; top:8px; left:8px; z-index:10; background:rgba(0,0,0,0.8); padding:10px; border-radius:6px; max-width:400px; }}
  #controls {{ position:absolute; top:8px; right:8px; z-index:10; background:rgba(0,0,0,0.8); padding:10px 14px; border-radius:6px; min-width:250px; max-height:90vh; overflow-y:auto; }}
  #controls label {{ display:block; margin:4px 0; }}
  #controls hr {{ border-color:#444; margin:8px 0; }}
  .sg {{ color:#ff00ff; font-weight:bold; }}
  .siv {{ color:#00ff00; font-weight:bold; }}
  .scal {{ color:#00ff00; font-weight:bold; }}
  .smerc {{ color:#ffaa00; font-weight:bold; }}
  #patchPanel {{ position:absolute; bottom:0; left:0; right:0; height:0; background:rgba(0,0,0,0.92);
                 z-index:20; transition:height 0.3s; overflow:hidden; }}
  #patchPanel.show {{ height:220px; }}
  #patchInner {{ display:flex; align-items:center; justify-content:center; gap:12px; height:100%; padding:0 20px; }}
  #patchPanel canvas {{ width:120px; height:120px; image-rendering:pixelated; }}
  .plabel {{ font-size:10px; text-align:center; margin-bottom:2px; }}
  .ppair {{ text-align:center; }}
  .ppair canvas {{ border:2px solid #555; }}
  #closeBtn {{ position:absolute; top:5px; right:15px; cursor:pointer; color:#f00; font-size:18px; font-weight:bold; z-index:21; }}
  .btn-toggle {{ display:inline-block; padding:4px 10px; margin:2px; border:1px solid #555; border-radius:4px;
    cursor:pointer; font-size:10px; color:#aaa; background:#222; }}
  .btn-toggle.active {{ background:#335; border-color:#88f; color:#fff; }}
</style>
</head><body>
<div id="viewToggle" style="position:fixed;top:8px;left:50%;transform:translateX(-50%);z-index:50;display:flex;gap:0;border-radius:6px;overflow:hidden;border:1px solid #555">
  <button onclick="window.location.href='dual_v5.html'" style="background:#222;color:#888;border:none;padding:6px 16px;cursor:pointer;font-size:11px;font-family:inherit">MODALITY</button>
  <button style="background:#444;color:#fff;border:none;border-left:1px solid #555;padding:6px 16px;cursor:pointer;font-size:11px;font-family:inherit;font-weight:bold">REGISTRATION</button>
</div>
<div id="info">
  <b>4-Modality 3D Viewer v5 (PKL direct)</b><br>
  {len(set(lm['tile'] for lm in all_landmarks))} tiles | <span style="color:#0ff">{N_FILTERED} filtered</span> / {N_CELLS} total landmarks<br>
  <span style="color:#ff00ff">Magenta</span> = ex-vivo &nbsp;
  <span style="color:#00ff00">Green</span> = in-vivo warped<br>
  <span style="color:#00ff00">Green</span> = calcium (std) &nbsp;
  <span style="color:#ffaa00">Rainbow</span> = MERSCOPE<br>
  <span style="color:#0ff">Cyan</span> = filtered lm &nbsp;
  <span style="color:#f80">Orange</span> = unfiltered lm<br>
  Drag: rotate | Scroll: zoom | Shift+drag: pan<br>
  <hr style="border-color:#444;margin:6px 0">
  <table style="font-size:10px;border-collapse:collapse;width:100%">
    <tr><td></td>
        <td style="color:#ff00ff"><b>EV</b></td>
        <td style="color:#00ff00"><b>IV warp</b></td>
        <td style="color:#00ff00"><b>Cal</b></td>
        <td style="color:#ffaa00"><b>MERC</b></td></tr>
    <tr><td>Pts</td>
        <td style="color:#ff00ff">{n_ex:,}</td>
        <td style="color:#00ff00">{n_iv:,}</td>
        <td style="color:#00ff00">{n_cal_vox:,}</td>
        <td style="color:#ffaa00">{n_merc_total:,}</td></tr>
    <tr><td>Grid</td><td colspan="4">({ex_nz},{ex_ny},{ex_nx}) @ DS{DS_EX}</td></tr>
  </table>
</div>
<div id="controls">
  <span class="sg">Ex-vivo (magenta)</span>
  <label>Thresh: <input type="range" id="exThresh" min="0" max="100" value="15" style="width:90px"><span id="exThVal">15</span></label>
  <label>Opacity: <input type="range" id="exOpac" min="0" max="100" value="50" style="width:90px"><span id="exOpVal">50</span></label>
  <label>Pt size: <input type="range" id="exPsize" min="1" max="30" value="1" style="width:90px"><span id="exPsVal">1</span></label>
  <hr>
  <span class="siv">In-vivo (green)</span>
  <label>Thresh: <input type="range" id="ivThresh" min="0" max="100" value="20" style="width:90px"><span id="ivThVal">20</span></label>
  <label>Opacity: <input type="range" id="ivOpac" min="0" max="100" value="70" style="width:90px"><span id="ivOpVal">70</span></label>
  <label>Pt size: <input type="range" id="ivPsize" min="1" max="30" value="1" style="width:90px"><span id="ivPsVal">1</span></label>
  <hr>
  <span class="scal">Calcium (green)</span>
  <label>Thresh: <input type="range" id="calThresh" min="0" max="100" value="10" style="width:90px"><span id="calThVal">10</span></label>
  <label>Opacity: <input type="range" id="calOpac" min="0" max="100" value="60" style="width:90px"><span id="calOpVal">60</span></label>
  <label>Pt size: <input type="range" id="calPsize" min="1" max="30" value="1" style="width:90px"><span id="calPsVal">1</span></label>
  <hr>
  <span class="smerc">MERSCOPE (rainbow)</span>
  <label>Opacity: <input type="range" id="mercOpac" min="0" max="100" value="60" style="width:90px"><span id="mercOpVal">60</span></label>
  <label>Pt size: <input type="range" id="mercPsize" min="1" max="30" value="1" style="width:90px"><span id="mercPsVal">1</span></label>
  <hr>
  <label>Tile: <select id="tileSelect">{tile_opts}</select></label>
  <label><input type="checkbox" id="autorot"> Auto-rotate</label>
  <label><b>Landmarks:</b></label>
  <div>
    <span class="btn-toggle active" id="btnFiltered" onclick="toggleLmMode('filtered')">Filtered ({N_FILTERED})</span>
    <span class="btn-toggle" id="btnAll" onclick="toggleLmMode('all')">All ({N_CELLS})</span>
    <span class="btn-toggle" id="btnNone" onclick="toggleLmMode('none')">None</span>
  </div>
  <label><input type="checkbox" id="showCross" checked> Show crosshairs</label>
</div>
<div id="patchPanel">
  <span id="closeBtn" onclick="stopCalAnim();document.getElementById('patchPanel').classList.remove('show')">&times;</span>
  <div id="patchInner">
    <div class="ppair"><div class="plabel" style="color:#00ff00">Calcium</div><canvas id="cvCAL" width="{PATCH_SZ}" height="{PATCH_SZ}"></canvas></div>
    <div class="ppair"><div class="plabel" style="color:#00ff00">In Vivo Warped</div><canvas id="cvIV" width="{PATCH_SZ}" height="{PATCH_SZ}"></canvas></div>
    <div class="ppair"><div class="plabel" style="color:#ff00ff">Ex Vivo</div><canvas id="cvEV" width="{PATCH_SZ}" height="{PATCH_SZ}"></canvas></div>
    <div class="ppair"><div class="plabel" style="color:#ffaa00">MERSCOPE</div><canvas id="cvMERC" width="{PATCH_SZ}" height="{PATCH_SZ}"></canvas></div>
    <div style="text-align:center;min-width:70px;font-size:10px" id="pairInfo"></div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const SCALE={SCALE:.1f}, N_CELLS={N_CELLS}, PATCH_SZ={PATCH_SZ};
const SPAN={span:.1f}, EX_CX={ex_cx:.6f}, EX_CY={ex_cy:.6f}, EX_CZ={ex_cz:.6f};
const ND2_Z_UM={ND2_Z_UM}, DS_EX={DS_EX};
const landmarks=[{",".join(landmarks_js)}];
const tileNames={json.dumps(cell_tiles)};
const isFiltered={json.dumps(cell_filtered)};
const tileRanges={json.dumps(tile_ranges)};
const tileZInfo={json.dumps({t: [tile_z_offsets[t], tile_z_offsets[t]+12] for t in unique_tiles})};
const cellInfo=[{",".join(cell_info_js)}];
const DZ={DZ_SLICES};
''')

html_parts.append(f'const exVox={{x:"{encode_f32(ex_vx)}",y:"{encode_f32(ex_vy)}",z:"{encode_f32(ex_vz)}",v:"{encode_f32(ex_vv)}",n:{n_ex}}};\n')
html_parts.append(f'const ivVox={{x:"{encode_f32(iv_vx)}",y:"{encode_f32(iv_vy)}",z:"{encode_f32(iv_vz)}",v:"{encode_f32(iv_vv)}",n:{n_iv}}};\n')
html_parts.append(f'const calVox={{x:"{encode_f32(cal_vx)}",y:"{encode_f32(cal_vy)}",z:"{encode_f32(cal_vz)}",v:"{encode_f32(cal_vv)}",n:{n_cal_vox}}};\n')
html_parts.append(f'const mercVox={{x:"{encode_f32(merc_vx)}",y:"{encode_f32(merc_vy)}",z:"{encode_f32(merc_vz)}",v:"{encode_f32(merc_vv)}",ci:"{encode_f32(merc_color_idx)}",n:{n_merc_total}}};\n')
html_parts.append(f'const genePalette={gene_palette_js};\n')
html_parts.append(f'const patchStripB64="{patch_strip_b64}";\n')
html_parts.append(f'const calStripB64="{calStripB64}";\n')
html_parts.append(f'const CAL_N_FRAMES={n_cal_frames}, CAL_PSZ=40;\n')

html_parts.append(f'''
let scene, camera, renderer, raycaster, mouse, pivotGroup;
let exPoints, ivPoints, calPoints, mercPoints, lmGroup;
let rotY=0, rotX=-0.3, zoom=6.0, panX=0, panY=0;
let dragging=false, lastX=0, lastY=0, startX=0, startY=0;
let autoRotate=false;
let calStripImg=null, calAnimId=null, calCurrentLm=-1, calFrameIdx=0;
let patchStripImg=null;
let hoveredIdx=-1, selectedIdx=-1;
let lmMode='filtered';
let visibleIndices=[];

function b64toF32(b64, n) {{
  const bin=atob(b64); const buf=new ArrayBuffer(n*4); const u8=new Uint8Array(buf);
  for(let i=0;i<bin.length;i++) u8[i]=bin.charCodeAt(i);
  return new Float32Array(buf);
}}

function colormap(v, name) {{
  if(name==='green') return [0,v,0];
  if(name==='magenta') return [v,0,v];
  if(name==='hot') return [Math.min(v*2,1),Math.max(v*2-1,0)*0.8,Math.max(v*3-2,0)];
  if(name==='cyan') return [0,v*0.8,v];
  return [v,v,v];
}}

function buildPoints(data, s, cmapName, zMin, zMax, vMin) {{
  const n=data.n;
  const xs=b64toF32(data.x,n), ys=b64toF32(data.y,n), zs=b64toF32(data.z,n), vs=b64toF32(data.v,n);
  // Auto-normalize values to 0-1 range
  let vlo=Infinity, vhi=-Infinity;
  for(let i=0;i<n;i++) {{ if(vs[i]<vlo) vlo=vs[i]; if(vs[i]>vhi) vhi=vs[i]; }}
  const vRange=vhi-vlo||1;
  const vn=new Float32Array(n);
  for(let i=0;i<n;i++) vn[i]=(vs[i]-vlo)/vRange;
  const vT=vMin||0;
  let cnt=0;
  for(let i=0;i<n;i++) if(vn[i]>=vT&&(zMin===undefined||zs[i]>=zMin&&zs[i]<=zMax)) cnt++;
  const pos=new Float32Array(cnt*3), col=new Float32Array(cnt*3);
  let j=0;
  for(let i=0;i<n;i++) {{
    if(vn[i]<vT) continue;
    if(zMin!==undefined&&(zs[i]<zMin||zs[i]>zMax)) continue;
    pos[j*3]=(xs[i]-0.5)*s*2; pos[j*3+1]=-(ys[i]-0.5)*s*2; pos[j*3+2]=(zs[i]-0.5)*s*2;
    const [r,g,b]=colormap(vn[i],cmapName);
    col[j*3]=r; col[j*3+1]=g; col[j*3+2]=b;
    j++;
  }}
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  geo.setAttribute('color',new THREE.BufferAttribute(col,3));
  geo._voxCount=cnt;
  return geo;
}}

function buildMercPoints(data, s, zMin, zMax) {{
  const n=data.n;
  const xs=b64toF32(data.x,n), ys=b64toF32(data.y,n), zs=b64toF32(data.z,n);
  const cis=b64toF32(data.ci,n);
  let cnt=0;
  for(let i=0;i<n;i++) if(zMin===undefined||zs[i]>=zMin&&zs[i]<=zMax) cnt++;
  const pos=new Float32Array(cnt*3), col=new Float32Array(cnt*3);
  let j=0;
  for(let i=0;i<n;i++) {{
    if(zMin!==undefined&&(zs[i]<zMin||zs[i]>zMax)) continue;
    pos[j*3]=(xs[i]-0.5)*s*2; pos[j*3+1]=-(ys[i]-0.5)*s*2; pos[j*3+2]=(zs[i]-0.5)*s*2;
    const c=genePalette[Math.round(cis[i]) % genePalette.length];
    col[j*3]=c[0]/255; col[j*3+1]=c[1]/255; col[j*3+2]=c[2]/255;
    j++;
  }}
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  geo.setAttribute('color',new THREE.BufferAttribute(col,3));
  return geo;
}}

function getVisibleIndices() {{
  if(lmMode==='none') return [];
  const sel=document.getElementById('tileSelect').value;
  const a=[];
  const start=sel==='all'?0:tileRanges[sel][0];
  const end=sel==='all'?N_CELLS:tileRanges[sel][1];
  for(let i=start;i<end;i++) {{
    if(lmMode==='filtered' && !isFiltered[i]) continue;
    a.push(i);
  }}
  return a;
}}

function lmPos(lm) {{
  return [(lm[0]-0.5)*SCALE*2, -(lm[1]-0.5)*SCALE*2, (lm[2]-0.5)*SCALE*2];
}}

function buildLandmarks() {{
  if(lmGroup) pivotGroup.remove(lmGroup);
  lmGroup=null;
  if(lmMode==='none') return;
  lmGroup=new THREE.Group();
  visibleIndices=getVisibleIndices();
  const sphereGeo=new THREE.SphereGeometry(0.025,10,10);
  for(const i of visibleIndices) {{
    const p=lmPos(landmarks[i]);
    const isHover=(i===hoveredIdx), isSel=(i===selectedIdx);
    let color;
    if(isHover) color=0xffff00;
    else if(isSel) color=0x00ffaa;
    else color=isFiltered[i]?0x00ffff:0xff8800;
    const mat=new THREE.MeshBasicMaterial({{color,transparent:true,opacity:0.95}});
    const s=new THREE.Mesh(sphereGeo,mat);
    s.position.set(p[0],p[1],p[2]);
    s.userData={{idx:i}};
    lmGroup.add(s);
  }}
  pivotGroup.add(lmGroup);
}}

function toggleLmMode(mode) {{
  lmMode=mode;
  document.getElementById('btnFiltered').classList.toggle('active',mode==='filtered');
  document.getElementById('btnAll').classList.toggle('active',mode==='all');
  document.getElementById('btnNone').classList.toggle('active',mode==='none');
  buildLandmarks();
}}

function rebuild() {{
  const exOpac=+document.getElementById('exOpac').value/100;
  const ivOpac=+document.getElementById('ivOpac').value/100;
  const calOpac=+document.getElementById('calOpac').value/100;
  const mercOpac=+document.getElementById('mercOpac').value/100;
  const exPs=+document.getElementById('exPsize').value;
  const ivPs=+document.getElementById('ivPsize').value;
  const calPs=+document.getElementById('calPsize').value;
  const mercPs=+document.getElementById('mercPsize').value;
  const exTh=+document.getElementById('exThresh').value/100;
  const ivTh=+document.getElementById('ivThresh').value/100;
  const calTh=+document.getElementById('calThresh').value/100;
  document.getElementById('exThVal').textContent=document.getElementById('exThresh').value;
  document.getElementById('ivThVal').textContent=document.getElementById('ivThresh').value;
  document.getElementById('calThVal').textContent=document.getElementById('calThresh').value;
  document.getElementById('exOpVal').textContent=document.getElementById('exOpac').value;
  document.getElementById('exPsVal').textContent=exPs;
  document.getElementById('ivOpVal').textContent=document.getElementById('ivOpac').value;
  document.getElementById('ivPsVal').textContent=ivPs;
  document.getElementById('calOpVal').textContent=document.getElementById('calOpac').value;
  document.getElementById('calPsVal').textContent=calPs;
  document.getElementById('mercOpVal').textContent=document.getElementById('mercOpac').value;
  document.getElementById('mercPsVal').textContent=mercPs;

  const sel=document.getElementById('tileSelect').value;
  let zMin, zMax;
  if(sel!=='all' && tileZInfo[sel]) {{
    const zOff=tileZInfo[sel];
    zMin = (zOff[0]*ND2_Z_UM/DS_EX)/SPAN + (0.5-EX_CZ);
    zMax = (zOff[1]*ND2_Z_UM/DS_EX)/SPAN + (0.5-EX_CZ);
  }}

  if(exPoints) pivotGroup.remove(exPoints);
  if(ivPoints) pivotGroup.remove(ivPoints);
  if(calPoints) pivotGroup.remove(calPoints);
  if(mercPoints) pivotGroup.remove(mercPoints);

  exPoints=new THREE.Points(buildPoints(exVox,SCALE,'magenta',zMin,zMax,exTh),new THREE.PointsMaterial({{
    size:exPs*0.02,vertexColors:true,transparent:true,opacity:exOpac,blending:THREE.AdditiveBlending,depthWrite:false}}));
  ivPoints=new THREE.Points(buildPoints(ivVox,SCALE,'green',zMin,zMax,ivTh),new THREE.PointsMaterial({{
    size:ivPs*0.02,vertexColors:true,transparent:true,opacity:ivOpac,blending:THREE.AdditiveBlending,depthWrite:false}}));
  calPoints=new THREE.Points(buildPoints(calVox,SCALE,'green',zMin,zMax,calTh),new THREE.PointsMaterial({{
    size:calPs*0.02,vertexColors:true,transparent:true,opacity:calOpac,blending:THREE.AdditiveBlending,depthWrite:false}}));
  mercPoints=new THREE.Points(buildMercPoints(mercVox,SCALE,zMin,zMax),new THREE.PointsMaterial({{
    size:mercPs*0.02,vertexColors:true,transparent:true,opacity:mercOpac,blending:THREE.AdditiveBlending,depthWrite:false}}));

  pivotGroup.add(exPoints);
  pivotGroup.add(ivPoints);
  pivotGroup.add(calPoints);
  pivotGroup.add(mercPoints);
  buildLandmarks();
}}

function findNearestLandmark(e) {{
  mouse.x=(e.clientX/innerWidth)*2-1; mouse.y=-(e.clientY/innerHeight)*2+1;
  raycaster.setFromCamera(mouse,camera);
  if(!lmGroup) return -1;
  const hits=raycaster.intersectObjects(lmGroup.children);
  if(hits.length>0) return hits[0].object.userData.idx;
  return -1;
}}

function drawCrosshair(ctx, color) {{
  const cx=PATCH_SZ/2, cy=PATCH_SZ/2;
  ctx.strokeStyle=color; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(cx-10,cy); ctx.lineTo(cx+10,cy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx,cy-10); ctx.lineTo(cx,cy+10); ctx.stroke();
}}

function drawCalFrame(idx, fi) {{
  if(!calStripImg||!calStripImg.complete) return;
  const cv=document.getElementById('cvCAL'), ctx=cv.getContext('2d');
  ctx.clearRect(0,0,PATCH_SZ,PATCH_SZ);
  const sy=(idx*CAL_N_FRAMES+fi)*CAL_PSZ;
  const off=document.createElement('canvas');off.width=CAL_PSZ;off.height=CAL_PSZ;
  const oCtx=off.getContext('2d');
  oCtx.drawImage(calStripImg,0,sy,CAL_PSZ,CAL_PSZ,0,0,CAL_PSZ,CAL_PSZ);
  const idata=oCtx.getImageData(0,0,CAL_PSZ,CAL_PSZ);
  const d=idata.data;
  for(let p=0;p<d.length;p+=4){{const v=d[p]; d[p]=0; d[p+1]=v; d[p+2]=0; d[p+3]=255;}}
  oCtx.putImageData(idata,0,0);
  ctx.imageSmoothingEnabled=true;
  ctx.drawImage(off,0,0,CAL_PSZ,CAL_PSZ,0,0,PATCH_SZ,PATCH_SZ);
  if(document.getElementById('showCross').checked) drawCrosshair(ctx,'#fff');
}}
function startCalAnim(idx) {{
  if(calAnimId) clearInterval(calAnimId);
  calCurrentLm=idx; calFrameIdx=0;
  const calZOk = cellInfo[idx][6];
  const cv=document.getElementById('cvCAL'), ctx=cv.getContext('2d');
  if(!calZOk) {{
    ctx.clearRect(0,0,PATCH_SZ,PATCH_SZ);
    ctx.fillStyle='#000'; ctx.fillRect(0,0,PATCH_SZ,PATCH_SZ);
    ctx.fillStyle='#666'; ctx.font='9px monospace'; ctx.textAlign='center';
    ctx.fillText('No calcium',PATCH_SZ/2,PATCH_SZ/2-6);
    ctx.fillText('at z='+cellInfo[idx][3],PATCH_SZ/2,PATCH_SZ/2+6);
    return;
  }}
  drawCalFrame(idx, 0);
  calAnimId=setInterval(function(){{
    calFrameIdx=(calFrameIdx+1)%CAL_N_FRAMES;
    drawCalFrame(calCurrentLm, calFrameIdx);
  }}, 100);
}}
function stopCalAnim() {{
  if(calAnimId){{clearInterval(calAnimId);calAnimId=null;}}
  calCurrentLm=-1;
}}

function showPatch(idx) {{
  if(!patchStripImg) return;
  const sy=idx*PATCH_SZ;
  const showCross=document.getElementById('showCross').checked;
  // 3 columns in strip: 0=EV, 1=IV, 2=MERC
  const panels=[
    {{id:'cvEV',  col:0, color:'#ff00ff'}},
    {{id:'cvIV',  col:1, color:'#00ff00'}},
    {{id:'cvMERC',col:2, color:'#ffaa00'}},
  ];
  for(const p of panels) {{
    const cv=document.getElementById(p.id), ctx=cv.getContext('2d');
    ctx.clearRect(0,0,PATCH_SZ,PATCH_SZ);
    ctx.drawImage(patchStripImg,PATCH_SZ*p.col,sy,PATCH_SZ,PATCH_SZ,0,0,PATCH_SZ,PATCH_SZ);
    if(showCross) drawCrosshair(ctx, p.color);
  }}
  // Start calcium animation
  startCalAnim(idx);
  document.getElementById('pairInfo').innerHTML=
    '<b>#'+idx+'</b><br><span style="font-size:10px;color:#aaa">'+tileNames[idx]+'</span>';
  document.getElementById('patchPanel').classList.add('show');
}}

function onClick(e) {{
  if(e.shiftKey) return;
  if(Math.abs(e.clientX-startX)>4 || Math.abs(e.clientY-startY)>4) return;
  const idx=findNearestLandmark(e);
  if(idx>=0) {{ selectedIdx=idx; showPatch(idx); buildLandmarks(); }}
  else {{ selectedIdx=-1; stopCalAnim(); document.getElementById('patchPanel').classList.remove('show'); buildLandmarks(); }}
}}

function onMove(e) {{
  if(dragging) {{
    const dx=e.clientX-lastX, dy=e.clientY-lastY;
    if(e.shiftKey){{ panX+=dx*0.003; panY-=dy*0.003; }} else {{ rotY+=dx*0.005; rotX+=dy*0.005; }}
    lastX=e.clientX; lastY=e.clientY; return;
  }}
  const idx=findNearestLandmark(e);
  if(idx!==hoveredIdx) {{ hoveredIdx=idx; buildLandmarks(); renderer.domElement.style.cursor=idx>=0?'pointer':'default'; }}
}}

function animate() {{
  requestAnimationFrame(animate);
  if(autoRotate&&!dragging) rotY+=0.002;
  pivotGroup.rotation.y=rotY; pivotGroup.rotation.x=rotX;
  pivotGroup.position.x=panX; pivotGroup.position.y=panY;
  camera.position.z=zoom;
  renderer.render(scene,camera);
}}

function init() {{
  scene=new THREE.Scene();
  camera=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.1,100);
  camera.position.z=zoom;
  renderer=new THREE.WebGLRenderer({{antialias:true}});
  renderer.setSize(innerWidth,innerHeight);
  renderer.setPixelRatio(devicePixelRatio);
  document.body.appendChild(renderer.domElement);
  raycaster=new THREE.Raycaster(); mouse=new THREE.Vector2();
  pivotGroup=new THREE.Group(); scene.add(pivotGroup);
  patchStripImg=new Image();
  patchStripImg.src='data:image/png;base64,'+patchStripB64;
  calStripImg=new Image();
  calStripImg.src='data:image/png;base64,'+calStripB64;
  rebuild();
  animate();
}}

document.addEventListener('mousedown',e=>{{ dragging=true; startX=lastX=e.clientX; startY=lastY=e.clientY; }});
document.addEventListener('mouseup',e=>{{ dragging=false; onClick(e); }});
document.addEventListener('mousemove',onMove);
document.addEventListener('wheel',e=>{{ zoom=Math.max(0.5,Math.min(20,zoom+e.deltaY*0.003)); }});
window.addEventListener('resize',()=>{{ camera.aspect=innerWidth/innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth,innerHeight); }});

let rt=null;
['exThresh','ivThresh','calThresh','exOpac','exPsize','ivOpac','ivPsize','calOpac','calPsize','mercOpac','mercPsize'].forEach(id=>{{
  document.getElementById(id).addEventListener('input',()=>{{ clearTimeout(rt); rt=setTimeout(rebuild,200); }});
}});
document.getElementById('tileSelect').addEventListener('change',rebuild);
document.getElementById('showCross').addEventListener('change',()=>{{ if(selectedIdx>=0) showPatch(selectedIdx); }});
document.getElementById('autorot').addEventListener('change',e=>autoRotate=e.target.checked);
init();
</script></body></html>
''')

html = ''.join(html_parts)

with open(OUT, 'w') as f:
    f.write(html)
fsize = os.path.getsize(OUT) / 1e6
print(f"\nDone! {OUT} ({fsize:.1f} MB)")
print(f"Ex-vivo: {n_ex:,} | In-vivo warped: {n_iv:,} | Calcium: {n_cal_vox:,} | MERSCOPE: {n_merc_total:,} | Landmarks: {N_CELLS} ({N_FILTERED} filtered)")