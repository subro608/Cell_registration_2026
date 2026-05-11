#!/usr/bin/env python3
"""
3D HTML viewer: ex-vivo stitched + in-vivo warped into same space.
Clickable landmark spheres → patch panel showing:
  ex-vivo MIP, ex-vivo depth, iv-warped MIP, iv-warped depth, iv-raw MIP, iv-raw depth
Based on viewer_dual_3d_v5.html pattern.
"""
import numpy as np
import cv2
import os
import base64
import json
import glob
import io
import tifffile
import SimpleITK as sitk
from PIL import Image
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from collections import defaultdict

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT = f'{BASE}/3d_viewer/viewer_warped_invivo_3d_v3.html'
os.makedirs(f'{BASE}/3d_viewer', exist_ok=True)

IV_XY_UM = 0.6835
IV_Z_UM = 3.0
ND2_XY_UM = 0.645
ND2_Z_UM = 2.0

DS = 4
DS_EX = 4
VOXEL_THRESH_EX = 8
VOXEL_THRESH_IV = 50
PATCH_SZ = 80
DZ_SLICES = 2
PHYS_RADIUS = 50  # 50µm half-width
CROP_ND2 = int(round(PHYS_RADIUS / ND2_XY_UM))  # ~78 px
CROP_JY = int(round(PHYS_RADIUS / IV_XY_UM))     # ~73 px

# ============================================================
# Helpers
# ============================================================
def norm_f(img):
    vals = img[img > 0]
    if len(vals) < 100:
        return img.copy()
    lo, hi = np.percentile(vals, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.float32)

def tps_remap_maps(h, w, W_inv, a_inv, src_inv, ds=4):
    """Build dense remap field for inverse TPS (for cv2.remap image warping)."""
    gy, gx = np.mgrid[0:h:ds, 0:w:ds].astype(np.float64)
    gh, gw = gy.shape
    qx = gx.ravel()[:, None]
    qy = gy.ravel()[:, None]
    sx = src_inv[:, 0][None, :]
    sy = src_inv[:, 1][None, :]
    r = np.sqrt((qx - sx)**2 + (qy - sy)**2)
    r = np.maximum(r, 1e-10)
    U = r**2 * np.log(r)
    map_x = a_inv[0, 0] + a_inv[1, 0] * qx.ravel() + a_inv[2, 0] * qy.ravel() + U @ W_inv[:, 0]
    map_y = a_inv[0, 1] + a_inv[1, 1] * qx.ravel() + a_inv[2, 1] * qy.ravel() + U @ W_inv[:, 1]
    map_x = map_x.reshape(gh, gw).astype(np.float32)
    map_y = map_y.reshape(gh, gw).astype(np.float32)
    if ds > 1:
        map_x = cv2.resize(map_x, (w, h), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(map_y, (w, h), interpolation=cv2.INTER_LINEAR)
    return map_x, map_y


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
        # First tile: just cumulative (identity + offset)
        M_cum = np.array(cum_iou[tile])[:2, :]
        return cv2.warpAffine(sl, M_cum, (canvas_w, canvas_h),
                               flags=cv2.INTER_LINEAR, borderValue=0)
    # Step 1: pair IOU rigid in 4200x4200
    prev_key = tile_order[tidx - 1]
    pair_key = f'{prev_key}_to_{tile}'
    if pair_key in iou_transforms:
        pair_warp = np.array(iou_transforms[pair_key]['warp_matrix'], dtype=np.float32)
        sl_rigid = cv2.warpAffine(sl, pair_warp, (4200, 4200),
                                   flags=cv2.INTER_LINEAR, borderValue=0)
    else:
        sl_rigid = sl

    # Step 2: pair elastix B-spline in 4200x4200
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
            # Recover original intensity scale
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

    # Step 3: previous tile's cumulative to place on canvas
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
    mip_u8 = np.clip(mip / max(p99, 1) * 255, 0, 255).astype(np.uint8)
    mip_resized = np.array(Image.fromarray(mip_u8).resize((patch_sz, patch_sz), Image.LANCZOS))
    mip_rgb = cv2.cvtColor(mip_resized, cv2.COLOR_GRAY2BGR)
    return mip_rgb

# ============================================================
# 1. Load stitch params
# ============================================================
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

# Load pair IOU rigid transforms for v5 stitching
with open(f'{BASE}/registration_video/auto_align_transforms_iou_v4.json') as f:
    iou_transforms = json.load(f)

# Build tile_idx lookup
tile_idx_map = {t: i for i, t in enumerate(TILE_ORDER)}

# ============================================================
# 2. Load in-vivo
# ============================================================
print("Loading JY306 in-vivo...")
iv_vol_raw = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv, ny_iv, nx_iv = iv_vol_raw.shape
print(f"  In-vivo: {iv_vol_raw.shape}")

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
# 4. Per-tile: compute affine, warp in-vivo, build patches
# ============================================================
print("\nProcessing tiles...")

ds_w = canvas_w // DS
ds_h = canvas_h // DS
warped_vol = np.zeros((total_z_native, ds_h, ds_w), dtype=np.float32)

# Collect all landmark data for patch generation + 3D positions
all_landmarks = []  # list of dicts per landmark

for tile in sorted(tile_lm_files.keys()):
    print(f"\n  {tile}:")

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

    d = np.load(tile_lm_files[tile])
    ev_nd2 = d['ev_nd2']
    pcd_iv = d['pcd_invivo_jy306']
    N_LM = ev_nd2.shape[0]
    if N_LM < 4:
        print(f"    SKIP: only {N_LM} landmarks")
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
    print(f"    {N_LM} lm | affine err: {errors.mean():.1f}µm")

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

    z_offset = tile_z_offsets[tile]

    # Load TPS transform for this tile
    tps_path = f'{BASE}/png_exports/registration_per_tile_tps/{tile}/tps_transform_{tile}.npz'
    tps_data = None
    if os.path.exists(tps_path):
        td_npz = np.load(tps_path)
        tps_data = {
            'W_inv': td_npz['W_inv'], 'a_inv': td_npz['a_inv'], 'src_inv': td_npz['src_inv'],
        }
        print(f"    TPS transform loaded ({len(tps_data['src_inv'])} control pts)")

    # Warp each nd2 z-slice (affine + TPS)
    for z_nd2 in range(12):
        # Sample z_iv at all landmark positions, use median of valid ones
        z_iv_candidates = []
        for i in range(N_LM):
            pt = np.array([z_nd2, ev_nd2[i, 1], ev_nd2[i, 0]])
            z_iv_cand = (M_inv @ pt + offset_inv)[0]
            if 0 <= z_iv_cand < nz_iv:
                z_iv_candidates.append(z_iv_cand)
        if not z_iv_candidates:
            continue
        z_iv = int(round(np.median(z_iv_candidates)))
        z_iv = np.clip(z_iv, 0, nz_iv - 1)

        M2d = np.array([
            [M_inv[2, 2], M_inv[2, 1], M_inv[2, 0] * z_nd2 + offset_inv[2]],
            [M_inv[1, 2], M_inv[1, 1], M_inv[1, 0] * z_nd2 + offset_inv[1]],
        ], dtype=np.float64)
        iv_affine_raw = cv2.warpAffine(iv_vol_raw[z_iv], M2d, (nd2_w, nd2_h),
                                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)

        # Apply TPS on top of affine
        if tps_data is not None:
            tps_maps = tps_remap_maps(nd2_h, nd2_w,
                                       tps_data['W_inv'], tps_data['a_inv'], tps_data['src_inv'])
            iv_warped = cv2.remap(iv_affine_raw, tps_maps[0], tps_maps[1],
                                   cv2.INTER_LINEAR, borderValue=0)
            print(f"    affine+TPS z_nd2={z_nd2} z_iv={z_iv}")
        else:
            iv_warped = iv_affine_raw
            print(f"    affine-only z_nd2={z_nd2} z_iv={z_iv}")

        # Full v5 stitching: pair IOU rigid + pair elastix + cumulative placement
        iv_stitched = stitch_tile_v5(iv_warped.astype(np.float32), tile,
                                      tile_idx_map, iou_transforms, cum_iou,
                                      STITCH_ELASTIX_DIR, TILE_ORDER, canvas_w, canvas_h)
        iv_ds = cv2.resize(iv_stitched, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
        z_out = z_offset + z_nd2
        warped_vol[z_out] = np.maximum(warped_vol[z_out], iv_ds)

    # Collect landmark info with blob-distance filtering
    WARP_ERR_MAX_UM = 5.0
    # Compute TPS remap once per tile (2D, same for all z)
    tile_tps_maps = None
    if tps_data is not None:
        tile_tps_maps = tps_remap_maps(nd2_h, nd2_w,
                                        tps_data['W_inv'], tps_data['a_inv'], tps_data['src_inv'])

    n_kept = 0
    for i in range(N_LM):
        z_nd2_i = int(round(np.clip(nd2_z_vals[i], 0, 11)))

        # Blob-distance filter: warp Gaussian blob through affine+TPS, measure centroid
        z_iv_i = int(round(pcd_iv[i, 0]))
        z_iv_i = max(0, min(nz_iv - 1, z_iv_i))
        M2d_i = np.array([
            [M_inv[2, 2], M_inv[2, 1], M_inv[2, 0] * z_nd2_i + offset_inv[2]],
            [M_inv[1, 2], M_inv[1, 1], M_inv[1, 0] * z_nd2_i + offset_inv[1]],
        ], dtype=np.float64)

        # Create Gaussian blob at in-vivo landmark position
        bimg = np.zeros((ny_iv, nx_iv), dtype=np.float32)
        ly = int(round(pcd_iv[i, 1]))
        lx = int(round(pcd_iv[i, 2]))
        for dy in range(-8, 9):
            for dx in range(-8, 9):
                yy, xx = ly + dy, lx + dx
                if 0 <= yy < ny_iv and 0 <= xx < nx_iv:
                    bimg[yy, xx] = 255.0 * np.exp(-0.5 * (dy*dy + dx*dx) / 9.0)

        # Warp blob through affine
        blob_affine = cv2.warpAffine(bimg, M2d_i, (nd2_w, nd2_h),
                                      flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
        # Warp through TPS
        if tile_tps_maps is not None:
            blob_warped = cv2.remap(blob_affine, tile_tps_maps[0], tile_tps_maps[1],
                                     cv2.INTER_LINEAR, borderValue=0)
        else:
            blob_warped = blob_affine

        # Measure blob centroid distance from ex-vivo landmark
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

        if blob_dist_um > WARP_ERR_MAX_UM:
            continue

        n_kept += 1

        # Stitched position for 3D display
        nd2_x = ev_nd2[i, 0]
        nd2_y = ev_nd2[i, 1]
        M_st = np.array(cum_iou[tile])
        canvas_x = M_st[0, 0] * nd2_x + M_st[0, 1] * nd2_y + M_st[0, 2]
        canvas_y = M_st[1, 0] * nd2_x + M_st[1, 1] * nd2_y + M_st[1, 2]
        canvas_z = (z_offset + z_nd2_i) * ND2_Z_UM  # physical z in µm

        all_landmarks.append({
            'tile': tile,
            'idx': i,
            'ev_nd2': ev_nd2[i],        # (x, y) in nd2 pixels
            'pcd_iv': pcd_iv[i],         # (z, y, x) in JY306 pixels
            'z_nd2': z_nd2_i,
            'z_nd2_gauss': nd2_z_vals[i],
            'canvas_x': canvas_x,
            'canvas_y': canvas_y,
            'canvas_z_um': canvas_z,
            'err_um': float(errors[i]),
        })
    print(f"    blob filter: {n_kept}/{N_LM} kept (threshold={WARP_ERR_MAX_UM}µm)")

    del nd2_slices

N_CELLS = len(all_landmarks)
print(f"\nTotal landmarks: {N_CELLS}")
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

ex_u8 = np.clip(ex_vol / 4000 * 255, 0, 255).astype(np.uint8)
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

# Normalize to p99 then median filter bg subtraction (same as v5)
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
ez, ey, exx = np.where(ex_u8 > VOXEL_THRESH_EX)
ex_vals = ex_u8[ez, ey, exx]
n_ex = len(ez)

# Mask in-vivo to ex-vivo bounding box (remove scattered edge artifacts)
ex_mask = ex_u8 > VOXEL_THRESH_EX
ex_any_z = ex_mask.any(axis=0)  # 2D projection
ex_any_y = ex_mask.any(axis=(0, 2))
ex_any_x = ex_mask.any(axis=(0, 1))
ey_min, ey_max = np.where(ex_any_y)[0][[0, -1]]
exx_min, exx_max = np.where(ex_any_x)[0][[0, -1]]
ez_min, ez_max = np.where(ex_mask.any(axis=(1, 2)))[0][[0, -1]]
# Pad slightly
PAD = 5
ey_min, ey_max = max(0, ey_min - PAD), min(ex_ny - 1, ey_max + PAD)
exx_min, exx_max = max(0, exx_min - PAD), min(ex_nx - 1, exx_max + PAD)
ez_min, ez_max = max(0, ez_min - PAD), min(ex_nz - 1, ez_max + PAD)
# Zero out iv outside bounding box
iv_u8[:ez_min] = 0
iv_u8[ez_max+1:] = 0
iv_u8[:, :ey_min, :] = 0
iv_u8[:, ey_max+1:, :] = 0
iv_u8[:, :, :exx_min] = 0
iv_u8[:, :, exx_max+1:] = 0
print(f"  IV bbox clip: z=[{ez_min},{ez_max}] y=[{ey_min},{ey_max}] x=[{exx_min},{exx_max}]")

iz, iy, ix = np.where(iv_u8 > VOXEL_THRESH_IV)
iv_vals_arr = iv_u8[iz, iy, ix]
n_iv = len(iz)
print(f"  Ex-vivo: {n_ex:,} | In-vivo warped: {n_iv:,}")

# Normalize - same span for both
span = float(max(ex_nx, ex_ny, ex_nz))
ex_vx = exx.astype(np.float32) / span
ex_vy = ey.astype(np.float32) / span
ex_vz = ez.astype(np.float32) / span
ex_cx, ex_cy, ex_cz = ex_vx.mean(), ex_vy.mean(), ex_vz.mean()

iv_vx = ix.astype(np.float32) / span
iv_vy = iy.astype(np.float32) / span
iv_vz = iz.astype(np.float32) / span

# Center both on ex-vivo centroid
ex_vx += (0.5 - ex_cx); ex_vy += (0.5 - ex_cy); ex_vz += (0.5 - ex_cz)
iv_vx += (0.5 - ex_cx); iv_vy += (0.5 - ex_cy); iv_vz += (0.5 - ex_cz)

ex_vv = ex_vals.astype(np.float32) / 255.0
iv_vv = iv_vals_arr.astype(np.float32) / 255.0
del ex_u8, iv_u8

# ============================================================
# 8. Compute landmark positions in normalized 3D space
# ============================================================
print("Computing landmark 3D positions...")
# Landmarks are in nd2 canvas pixel coords -> need to convert to same
# normalized space as the ex-vivo voxels.
# The ex-vivo tif is at 1µm iso. Canvas native is 0.645µm/px.
# Ex-vivo tif dims: (ex_nz_full, ex_h_full, ex_w_full) at 1µm iso
# Canvas physical: canvas_w * 0.645 µm wide = ex_w_full * 1.0 µm (approximately)
# So canvas_px * 0.645 ≈ tif_px * 1.0 -> tif_px = canvas_px * 0.645

landmarks_js = []
cell_tiles = []
for lm in all_landmarks:
    # Convert canvas coords to ex-vivo tif DS grid coordinates
    tif_x = lm['canvas_x'] * ND2_XY_UM / DS_EX  # canvas px -> µm -> DS_EX tif px
    tif_y = lm['canvas_y'] * ND2_XY_UM / DS_EX
    tif_z = lm['canvas_z_um'] / DS_EX

    # Normalize same as voxels
    nx = tif_x / span + (0.5 - ex_cx)
    ny = tif_y / span + (0.5 - ex_cy)
    nz = tif_z / span + (0.5 - ex_cz)
    landmarks_js.append(f'[{nx:.5f},{ny:.5f},{nz:.5f}]')
    cell_tiles.append(lm['tile'])

# ============================================================
# 9. Generate patch strips (6 columns per landmark)
# ============================================================
print("Generating patch strips...")
# Columns: ex-MIP, ex-depth, iv-warped-MIP, iv-warped-depth, iv-raw-MIP, iv-raw-depth

# For iv-warped patches, we need the warped in-vivo in nd2 tile space per landmark.
# Rather than re-warping, crop from the stitched warped volume... but we deleted it.
# Instead, let's re-do the warp per landmark z-slice (lightweight).

# Pre-load nd2 pages for ex-vivo patches
print("  Loading nd2 PNGs for patches...")
patch_tiles = set(lm['tile'] for lm in all_landmarks)
nd2_pages = {}
for tile in patch_tiles:
    tile_dir = f'{BASE}/png_exports/registration_video/{tile}'
    for zi in range(12):
        png_path = f'{tile_dir}/GFP_z{zi:03d}.png'
        nd2_pages[(tile, zi)] = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)

# Per-tile affine caches for warped patches
print("  Computing per-tile affines for warped patches...")
tile_affines = {}
for tile in sorted(patch_tiles):
    d = np.load(tile_lm_files[tile])
    ev = d['ev_nd2']
    pcd = d['pcd_invivo_jy306']
    n = ev.shape[0]
    if n < 4:
        continue
    # Gaussian z
    nd2_zv = []
    for i in range(n):
        x, y = ev[i, 0], ev[i, 1]
        c = int(round(np.clip(x, 10, 4189)))
        r = int(round(np.clip(y, 10, 4189)))
        intensities = [nd2_pages[(tile, z)][r-10:r+10, c-10:c+10].astype(np.float32).mean() for z in range(12)]
        nd2_zv.append(find_z_gaussian(intensities))
    src = np.column_stack([pcd[:, 2] * IV_XY_UM, pcd[:, 1] * IV_XY_UM, pcd[:, 0] * IV_Z_UM])
    dst = np.column_stack([ev[:, 0] * ND2_XY_UM, ev[:, 1] * ND2_XY_UM, np.array(nd2_zv) * ND2_Z_UM])
    src_h = np.hstack([src, np.ones((n, 1))])
    AT, _, _, _ = np.linalg.lstsq(src_h, dst, rcond=None)
    Af = AT.T
    Mf = np.array([
        [Af[2,2]*IV_Z_UM/ND2_Z_UM, Af[2,1]*IV_XY_UM/ND2_Z_UM, Af[2,0]*IV_XY_UM/ND2_Z_UM],
        [Af[1,2]*IV_Z_UM/ND2_XY_UM, Af[1,1]*IV_XY_UM/ND2_XY_UM, Af[1,0]*IV_XY_UM/ND2_XY_UM],
        [Af[0,2]*IV_Z_UM/ND2_XY_UM, Af[0,1]*IV_XY_UM/ND2_XY_UM, Af[0,0]*IV_XY_UM/ND2_XY_UM],
    ])
    tf = np.array([Af[2,3]/ND2_Z_UM, Af[1,3]/ND2_XY_UM, Af[0,3]/ND2_XY_UM])
    Mi = np.linalg.inv(Mf)
    oi = -Mi @ tf
    tile_affines[tile] = (Mf, tf, Mi, oi)

# Load TPS data for patch generation
tile_tps_data = {}
for tile in sorted(patch_tiles):
    tps_path = f'{BASE}/png_exports/registration_per_tile_tps/{tile}/tps_transform_{tile}.npz'
    if os.path.exists(tps_path):
        td = np.load(tps_path)
        tile_tps_data[tile] = {
            'W_inv': td['W_inv'], 'a_inv': td['a_inv'], 'src_inv': td['src_inv'],
        }

# Warped iv slice cache per (tile, z_nd2)
warped_cache = {}

def get_warped_iv(tile, z_nd2, nd2_y=2100, nd2_x=2100):
    """Get warped in-vivo slice using 3D affine + TPS.
    nd2_y, nd2_x: reference position for z_iv mapping (use landmark pos)."""
    if tile not in tile_affines:
        return None
    Mf, tf, Mi, oi = tile_affines[tile]
    ref_nd2 = np.array([z_nd2, nd2_y, nd2_x])
    ref_iv = Mi @ ref_nd2 + oi
    z_iv = int(round(ref_iv[0]))
    if z_iv < 0 or z_iv >= nz_iv:
        return None
    key = (tile, z_nd2, z_iv)
    if key in warped_cache:
        return warped_cache[key]
    M2d = np.array([
        [Mi[2, 2], Mi[2, 1], Mi[2, 0] * z_nd2 + oi[2]],
        [Mi[1, 2], Mi[1, 1], Mi[1, 0] * z_nd2 + oi[1]],
    ], dtype=np.float64)
    iv_affine_raw = cv2.warpAffine(iv_vol_raw[z_iv], M2d, (4200, 4200),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
    # Apply TPS on top of affine
    if tile in tile_tps_data:
        td = tile_tps_data[tile]
        tps_maps = tps_remap_maps(4200, 4200, td['W_inv'], td['a_inv'], td['src_inv'])
        iv_w = cv2.remap(iv_affine_raw, tps_maps[0], tps_maps[1],
                          cv2.INTER_LINEAR, borderValue=0)
    else:
        iv_w = iv_affine_raw
    warped_cache[key] = iv_w
    return iv_w

# Build patch strip: 6 columns
print("  Building patch strip...")
patch_strip_w = PATCH_SZ * 6
patch_strip_h = PATCH_SZ * N_CELLS
patch_strip = np.zeros((patch_strip_h, patch_strip_w, 3), dtype=np.uint8)

cell_info_js = []

for ci, lm in enumerate(all_landmarks):
    tile = lm['tile']
    col_c = int(round(lm['ev_nd2'][0]))
    row_c = int(round(lm['ev_nd2'][1]))
    z_nd2 = lm['z_nd2']

    # --- Col 0-1: Ex-vivo MIP + depth ---
    slices_ex = []
    slices_ex_depth = []
    for dz in range(-DZ_SLICES, DZ_SLICES + 1):
        zz = z_nd2 + dz
        if 0 <= zz < 12 and (tile, zz) in nd2_pages:
            page = nd2_pages[(tile, zz)]
            if page is not None:
                y0 = max(0, row_c - CROP_ND2)
                y1 = min(page.shape[0], row_c + CROP_ND2)
                x0 = max(0, col_c - CROP_ND2)
                x1 = min(page.shape[1], col_c + CROP_ND2)
                crop = page[y0:y1, x0:x1].astype(np.float32)
                slices_ex.append(crop)
                t = (dz + DZ_SLICES) / max(2 * DZ_SLICES, 1)
                slices_ex_depth.append((crop, t))

    row = ci * PATCH_SZ
    patch_strip[row:row+PATCH_SZ, 0:PATCH_SZ] = make_gray_mip(slices_ex, PATCH_SZ, (0, 255, 0))
    patch_strip[row:row+PATCH_SZ, PATCH_SZ:PATCH_SZ*2] = make_depth_mip(slices_ex_depth, PATCH_SZ)

    # --- Col 2-3: In-vivo warped MIP + depth ---
    slices_ivw = []
    slices_ivw_depth = []
    for dz in range(-DZ_SLICES, DZ_SLICES + 1):
        zz = z_nd2 + dz
        if 0 <= zz < 12:
            warp_sl = get_warped_iv(tile, zz, nd2_y=row_c, nd2_x=col_c)
            if warp_sl is not None:
                y0 = max(0, row_c - CROP_ND2)
                y1 = min(4200, row_c + CROP_ND2)
                x0 = max(0, col_c - CROP_ND2)
                x1 = min(4200, col_c + CROP_ND2)
                crop = warp_sl[y0:y1, x0:x1].astype(np.float32)
                slices_ivw.append(crop)
                t = (dz + DZ_SLICES) / max(2 * DZ_SLICES, 1)
                slices_ivw_depth.append((crop, t))

    patch_strip[row:row+PATCH_SZ, PATCH_SZ*2:PATCH_SZ*3] = make_gray_mip(slices_ivw, PATCH_SZ, (255, 0, 255))
    patch_strip[row:row+PATCH_SZ, PATCH_SZ*3:PATCH_SZ*4] = make_depth_mip(slices_ivw_depth, PATCH_SZ)

    # --- Col 4-5: In-vivo raw (native) MIP + depth ---
    z_iv = int(round(lm['pcd_iv'][0]))
    y_iv = int(round(lm['pcd_iv'][1]))
    x_iv = int(round(lm['pcd_iv'][2]))
    slices_ivr = []
    slices_ivr_depth = []
    for dz in range(-DZ_SLICES, DZ_SLICES + 1):
        zz = z_iv + dz
        if 0 <= zz < nz_iv:
            page = iv_vol_raw[zz]
            y0 = max(0, y_iv - CROP_JY)
            y1 = min(page.shape[0], y_iv + CROP_JY)
            x0 = max(0, x_iv - CROP_JY)
            x1 = min(page.shape[1], x_iv + CROP_JY)
            crop = page[y0:y1, x0:x1].astype(np.float32)
            slices_ivr.append(crop)
            t = (dz + DZ_SLICES) / max(2 * DZ_SLICES, 1)
            slices_ivr_depth.append((crop, t))

    patch_strip[row:row+PATCH_SZ, PATCH_SZ*4:PATCH_SZ*5] = make_gray_mip(slices_ivr, PATCH_SZ, (255, 0, 255))
    patch_strip[row:row+PATCH_SZ, PATCH_SZ*5:PATCH_SZ*6] = make_depth_mip(slices_ivr_depth, PATCH_SZ)

    ez_lo = max(0, z_nd2 - DZ_SLICES)
    ez_hi = min(11, z_nd2 + DZ_SLICES)
    ivz_lo = max(0, z_iv - DZ_SLICES)
    ivz_hi = min(15, z_iv + DZ_SLICES)
    cell_info_js.append(f'[{z_nd2},{ez_lo},{ez_hi},{z_iv},{ivz_lo},{ivz_hi}]')

    if ci % 100 == 0:
        print(f"    patch {ci}/{N_CELLS}")

# Clear caches
warped_cache.clear()
del nd2_pages

print("  Encoding patch strip...")
patch_img = Image.fromarray(patch_strip, 'RGB')
buf = io.BytesIO()
patch_img.save(buf, format='PNG', optimize=True)
patch_strip_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
print(f"  Patch strip: {patch_strip_w}x{patch_strip_h}, {len(patch_strip_b64)//1024}KB")
del patch_strip

# ============================================================
# 10. Build HTML
# ============================================================
print("\nBuilding HTML...")

# Tile ranges for dropdown filter
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

html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>In-vivo Warped into Ex-vivo Stitched — 3D + Patches</title>
<style>
  body {{ margin:0; overflow:hidden; background:#000; color:#ddd; font-family:monospace; font-size:11px; }}
  #info {{ position:absolute; top:8px; left:8px; z-index:10; background:rgba(0,0,0,0.8); padding:10px; border-radius:6px; max-width:380px; }}
  #controls {{ position:absolute; top:8px; right:8px; z-index:10; background:rgba(0,0,0,0.8); padding:10px 14px; border-radius:6px; min-width:240px; }}
  #controls label {{ display:block; margin:4px 0; }}
  #controls hr {{ border-color:#444; margin:8px 0; }}
  .sg {{ color:#0f0; font-weight:bold; }}
  .sm {{ color:#f0f; font-weight:bold; }}
  #patchPanel {{ position:absolute; bottom:0; left:0; right:0; height:0; background:rgba(0,0,0,0.92);
                 z-index:20; transition:height 0.3s; overflow:hidden; }}
  #patchPanel.show {{ height:220px; }}
  #patchInner {{ display:flex; align-items:center; justify-content:center; gap:10px; height:100%; }}
  #patchPanel canvas {{ width:120px; height:120px; image-rendering:pixelated; }}
  .plabel {{ font-size:10px; text-align:center; margin-bottom:2px; }}
  .ppair {{ text-align:center; }}
  .ppair canvas {{ border:2px solid #555; }}
  #closeBtn {{ position:absolute; top:5px; right:15px; cursor:pointer; color:#f00; font-size:18px; font-weight:bold; z-index:21; }}
  #depthLegend {{ display:inline-block; width:80px; height:10px; border-radius:3px;
    background:linear-gradient(to right, #00ff00, #ffff00, #ff0000); margin-top:4px; }}
  .depth-labels {{ display:flex; justify-content:space-between; font-size:9px; color:#aaa; width:80px; }}
</style>
</head><body>
<div id="info">
  <b>In-vivo warped into Ex-vivo stitched space</b><br>
  {len(tile_lm_files)} tiles, {N_CELLS} landmarks | Click landmark to see patches<br>
  <span style="color:#0f0">Green</span> = ex-vivo &nbsp;
  <span style="color:#f0f">Magenta</span> = in-vivo warped<br>
  Drag: rotate | Scroll: zoom | Shift+drag: pan<br>
  <hr style="border-color:#444;margin:6px 0">
  <table style="font-size:10px;border-collapse:collapse;width:100%">
    <tr><td></td><td style="color:#0f0"><b>Ex-vivo</b></td><td style="color:#f0f"><b>IV warped</b></td></tr>
    <tr><td>Voxels</td><td style="color:#0f0">{n_ex:,}</td><td style="color:#f0f">{n_iv:,}</td></tr>
    <tr><td>Grid</td><td colspan="2">({ex_nz}, {ex_ny}, {ex_nx}) @ DS{DS_EX}</td></tr>
  </table>
</div>
<div id="controls">
  <span class="sg">Ex-vivo (stitched)</span>
  <label>Opacity: <input type="range" id="exOpac" min="0" max="100" value="50" style="width:90px"><span id="exOpVal">50</span></label>
  <label>Pt size: <input type="range" id="exPsize" min="1" max="30" value="2" style="width:90px"><span id="exPsVal">2</span></label>
  <hr>
  <span class="sm">In-vivo (warped)</span>
  <label>Opacity: <input type="range" id="ivOpac" min="0" max="100" value="70" style="width:90px"><span id="ivOpVal">70</span></label>
  <label>Pt size: <input type="range" id="ivPsize" min="1" max="30" value="2" style="width:90px"><span id="ivPsVal">2</span></label>
  <hr>
  <label>Tile: <select id="tileSelect">{tile_opts}</select></label>
  <label><input type="checkbox" id="autorot"> Auto-rotate</label>
  <label><input type="checkbox" id="showLm" checked> Show landmarks</label>
  <label><input type="checkbox" id="showCross" checked> Show crosshairs</label>
  <label>Ex cmap: <select id="exCmap"><option value="green">Green</option><option value="hot">Hot</option><option value="cyan">Cyan</option><option value="gray">Gray</option></select></label>
  <label>IV cmap: <select id="ivCmap"><option value="magenta" selected>Magenta</option><option value="green">Green</option><option value="hot">Hot</option><option value="cyan">Cyan</option><option value="gray">Gray</option></select></label>
</div>
<div id="patchPanel">
  <span id="closeBtn" onclick="document.getElementById('patchPanel').classList.remove('show')">&times;</span>
  <div id="patchInner">
    <div class="ppair"><div class="plabel" style="color:#0f0">Ex-vivo MIP</div><canvas id="cv0" width="{PATCH_SZ}" height="{PATCH_SZ}"></canvas></div>
    <div class="ppair"><div class="plabel" style="color:#fc0">Ex-vivo Depth</div><canvas id="cv1" width="{PATCH_SZ}" height="{PATCH_SZ}"></canvas></div>
    <div class="ppair"><div class="plabel" style="color:#f0f">IV Warped MIP</div><canvas id="cv2" width="{PATCH_SZ}" height="{PATCH_SZ}"></canvas></div>
    <div class="ppair"><div class="plabel" style="color:#fc0">IV Warped Depth</div><canvas id="cv3" width="{PATCH_SZ}" height="{PATCH_SZ}"></canvas></div>
    <div style="text-align:center;min-width:60px" id="pairInfo"></div>
    <div class="ppair"><div class="plabel" style="color:#f0f">IV Raw MIP</div><canvas id="cv4" width="{PATCH_SZ}" height="{PATCH_SZ}"></canvas></div>
    <div class="ppair"><div class="plabel" style="color:#fc0">IV Raw Depth</div><canvas id="cv5" width="{PATCH_SZ}" height="{PATCH_SZ}"></canvas></div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const SCALE={SCALE:.1f}, N_CELLS={N_CELLS}, PATCH_SZ={PATCH_SZ};
const SPAN={span:.1f}, EX_CZ={ex_cz:.6f}, ND2_Z_UM={ND2_Z_UM}, DS_EX={DS_EX};
const landmarks=[{",".join(landmarks_js)}];
const tileNames={json.dumps(cell_tiles)};
const tileRanges={json.dumps(tile_ranges)};
const tileZInfo={json.dumps({t: [tile_z_offsets[t], tile_z_offsets[t]+12] for t in unique_tiles})};
const cellInfo=[{",".join(cell_info_js)}];
const DZ={DZ_SLICES};
const exVox={{x:"{encode_f32(ex_vx)}",y:"{encode_f32(ex_vy)}",z:"{encode_f32(ex_vz)}",v:"{encode_f32(ex_vv)}",n:{n_ex}}};
const ivVox={{x:"{encode_f32(iv_vx)}",y:"{encode_f32(iv_vy)}",z:"{encode_f32(iv_vz)}",v:"{encode_f32(iv_vv)}",n:{n_iv}}};
const patchStripB64="{patch_strip_b64}";

let scene, camera, renderer, raycaster, mouse, pivotGroup;
let exPoints, ivPoints, lmGroup;
let rotY=0, rotX=-0.3, zoom=6.0, panX=0, panY=0;
let dragging=false, lastX=0, lastY=0, startX=0, startY=0;
let autoRotate=false;
let patchStripImg=null;
let hoveredIdx=-1, selectedIdx=-1;
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

function buildPoints(data, s, cmapName, zMin, zMax) {{
  const n=data.n;
  const xs=b64toF32(data.x,n), ys=b64toF32(data.y,n), zs=b64toF32(data.z,n), vs=b64toF32(data.v,n);
  // Count filtered points first
  let cnt=0;
  for(let i=0;i<n;i++) if(zMin===undefined||zs[i]>=zMin&&zs[i]<=zMax) cnt++;
  const pos=new Float32Array(cnt*3), col=new Float32Array(cnt*3);
  let j=0;
  for(let i=0;i<n;i++) {{
    if(zMin!==undefined&&(zs[i]<zMin||zs[i]>zMax)) continue;
    pos[j*3]=(xs[i]-0.5)*s*2; pos[j*3+1]=-(ys[i]-0.5)*s*2; pos[j*3+2]=(zs[i]-0.5)*s*2;
    const [r,g,b]=colormap(vs[i],cmapName);
    col[j*3]=r; col[j*3+1]=g; col[j*3+2]=b;
    j++;
  }}
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  geo.setAttribute('color',new THREE.BufferAttribute(col,3));
  return geo;
}}

function getVisibleIndices() {{
  const sel=document.getElementById('tileSelect').value;
  if(sel==='all'){{ const a=[]; for(let i=0;i<N_CELLS;i++) a.push(i); return a; }}
  const r=tileRanges[sel]; const a=[]; for(let i=r[0];i<r[1];i++) a.push(i); return a;
}}

function lmPos(lm) {{
  return [(lm[0]-0.5)*SCALE*2, -(lm[1]-0.5)*SCALE*2, (lm[2]-0.5)*SCALE*2];
}}

function buildLandmarks() {{
  if(lmGroup) pivotGroup.remove(lmGroup);
  lmGroup=null;
  if(!document.getElementById('showLm').checked) return;
  lmGroup=new THREE.Group();
  visibleIndices=getVisibleIndices();
  const sphereGeo=new THREE.SphereGeometry(0.018,8,8);
  for(const i of visibleIndices) {{
    const p=lmPos(landmarks[i]);
    const isHover=(i===hoveredIdx), isSel=(i===selectedIdx);
    const color=isHover?0xffff00:isSel?0x00ffaa:0xff8800;
    const mat=new THREE.MeshBasicMaterial({{color,transparent:true,opacity:0.85}});
    const s=new THREE.Mesh(sphereGeo,mat);
    s.position.set(p[0],p[1],p[2]);
    s.userData={{idx:i}};
    lmGroup.add(s);
  }}
  pivotGroup.add(lmGroup);
}}

function rebuild() {{
  const exCmap=document.getElementById('exCmap').value;
  const ivCmap=document.getElementById('ivCmap').value;
  const exOpac=+document.getElementById('exOpac').value/100;
  const ivOpac=+document.getElementById('ivOpac').value/100;
  const exPs=+document.getElementById('exPsize').value;
  const ivPs=+document.getElementById('ivPsize').value;
  document.getElementById('exOpVal').textContent=document.getElementById('exOpac').value;
  document.getElementById('exPsVal').textContent=exPs;
  document.getElementById('ivOpVal').textContent=document.getElementById('ivOpac').value;
  document.getElementById('ivPsVal').textContent=ivPs;

  // Compute z-range filter when tile selected
  const sel=document.getElementById('tileSelect').value;
  let zMin, zMax;
  if(sel!=='all' && tileZInfo[sel]) {{
    const zOff=tileZInfo[sel];
    // Convert native z-offsets to normalized coords: z_native -> z_tif_ds -> z_norm
    zMin = (zOff[0]*ND2_Z_UM/DS_EX)/SPAN + (0.5-EX_CZ);
    zMax = (zOff[1]*ND2_Z_UM/DS_EX)/SPAN + (0.5-EX_CZ);
  }}

  if(exPoints) pivotGroup.remove(exPoints);
  if(ivPoints) pivotGroup.remove(ivPoints);
  exPoints=new THREE.Points(buildPoints(exVox,SCALE,exCmap,zMin,zMax),new THREE.PointsMaterial({{
    size:exPs*0.02,vertexColors:true,transparent:true,opacity:exOpac,blending:THREE.AdditiveBlending,depthWrite:false}}));
  ivPoints=new THREE.Points(buildPoints(ivVox,SCALE,ivCmap,zMin,zMax),new THREE.PointsMaterial({{
    size:ivPs*0.02,vertexColors:true,transparent:true,opacity:ivOpac,blending:THREE.AdditiveBlending,depthWrite:false}}));
  pivotGroup.add(exPoints);
  pivotGroup.add(ivPoints);
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

function showPatch(idx) {{
  if(!patchStripImg) return;
  const sy=idx*PATCH_SZ;
  const showCross=document.getElementById('showCross').checked;
  const crossColors=['#00ff00','#ffffff','#ff00ff','#ffffff','#ff00ff','#ffffff'];
  for(let c=0;c<6;c++) {{
    const cv=document.getElementById('cv'+c), ctx=cv.getContext('2d');
    ctx.clearRect(0,0,PATCH_SZ,PATCH_SZ);
    ctx.drawImage(patchStripImg,PATCH_SZ*c,sy,PATCH_SZ,PATCH_SZ,0,0,PATCH_SZ,PATCH_SZ);
    if(showCross) drawCrosshair(ctx, crossColors[c]);
  }}
  const ci=cellInfo[idx];
  document.getElementById('pairInfo').innerHTML=
    '<b>#'+idx+'</b><br><span style="font-size:10px;color:#aaa">'+tileNames[idx]+'</span><br>'+
    '<span style="color:#0f0;font-size:9px">nd2 z'+ci[1]+'-'+ci[2]+'</span><br>'+
    '<span style="color:#f0f;font-size:9px">iv z'+ci[4]+'-'+ci[5]+'</span><br>'+
    '<div id="depthLegend"></div><div class="depth-labels"><span>-{DZ_SLICES}</span><span>+{DZ_SLICES}</span></div>';
  document.getElementById('patchPanel').classList.add('show');
}}

function onClick(e) {{
  if(e.shiftKey) return;
  if(Math.abs(e.clientX-startX)>4 || Math.abs(e.clientY-startY)>4) return;
  const idx=findNearestLandmark(e);
  if(idx>=0) {{ selectedIdx=idx; showPatch(idx); buildLandmarks(); }}
  else {{ selectedIdx=-1; document.getElementById('patchPanel').classList.remove('show'); buildLandmarks(); }}
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
  rebuild();
  animate();
}}

document.addEventListener('mousedown',e=>{{ dragging=true; startX=lastX=e.clientX; startY=lastY=e.clientY; }});
document.addEventListener('mouseup',e=>{{ dragging=false; onClick(e); }});
document.addEventListener('mousemove',onMove);
document.addEventListener('wheel',e=>{{ zoom=Math.max(0.5,Math.min(20,zoom+e.deltaY*0.003)); }});
window.addEventListener('resize',()=>{{ camera.aspect=innerWidth/innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth,innerHeight); }});

let rt=null;
['exOpac','exPsize','ivOpac','ivPsize','exCmap','ivCmap'].forEach(id=>{{
  document.getElementById(id).addEventListener('input',()=>{{ clearTimeout(rt); rt=setTimeout(rebuild,200); }});
}});
document.getElementById('showLm').addEventListener('change',buildLandmarks);
document.getElementById('tileSelect').addEventListener('change',rebuild);
document.getElementById('showCross').addEventListener('change',()=>{{ if(selectedIdx>=0) showPatch(selectedIdx); }});
document.getElementById('autorot').addEventListener('change',e=>autoRotate=e.target.checked);
init();
</script></body></html>
'''

with open(OUT, 'w') as f:
    f.write(html)
fsize = os.path.getsize(OUT) / 1e6
print(f"\nDone! {OUT} ({fsize:.1f} MB)")
print(f"Ex-vivo: {n_ex:,} | In-vivo warped: {n_iv:,} | Landmarks: {N_CELLS}")