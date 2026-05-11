#!/usr/bin/env python3
"""
Landmark patch viewer: side-by-side ex-vivo vs in-vivo warped patches
around each landmark, with tile selection dropdown and crosshair toggle.
Uses raw (unfiltered) in-vivo for display, median-filtered for elastix registration.
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
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from collections import defaultdict

BASE = '/Users/neurolab/neuroinformatics/margaret'


def tps_transform_pt(W, a, src, qx, qy):
    """Transform a point through TPS."""
    result = a[0] + a[1] * qx + a[2] * qy
    for i in range(len(src)):
        r = np.sqrt((qx - src[i, 0])**2 + (qy - src[i, 1])**2)
        if r > 1e-10:
            result = result + W[i] * r**2 * np.log(r)
    return float(result[0]), float(result[1])


def tps_remap_maps(h, w, W_inv, a_inv, src_inv, ds=4):
    """Build TPS remap maps (map_x, map_y) at full resolution. Reusable for multiple images."""
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


def tps_warp_image(img, W_inv, a_inv, src_inv, ds=4, remap=None):
    """Warp an image using inverse TPS. Pass remap=(map_x,map_y) to reuse precomputed maps."""
    if remap is not None:
        map_x, map_y = remap
    else:
        map_x, map_y = tps_remap_maps(img.shape[0], img.shape[1], W_inv, a_inv, src_inv, ds)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)

IV_XY_UM = 0.6835
IV_Z_UM = 3.0
ND2_XY_UM = 0.645
ND2_Z_UM = 2.0

PATCH_R = 100  # patch half-size in full-res pixels (~65µm radius)

OUT_DIR = f'{BASE}/png_exports/registration_per_tile_elastix'
OUT_HTML = f'{OUT_DIR}/landmark_patches_elastix_warp_err.html'

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
    _, buf = cv2.imencode('.jpg', rgb_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode('ascii')


def run_elastix_with_raw(fixed_np, moving_filt, moving_raw, extra_images=None,
                          fixed_points=None, moving_points=None):
    """Run elastix on filtered images, apply same transform to raw image.
    extra_images: optional list of additional images to warp with the same transform.
    fixed_points/moving_points: lists of (x,y) tuples for landmark correspondence penalty.
    Returns (warped_raw, [warped_extras]) or (warped_raw, None) if no extras."""
    fixed_sitk = sitk.GetImageFromArray(norm_f(fixed_np))
    moving_sitk = sitk.GetImageFromArray(norm_f(moving_filt))
    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(fixed_sitk)
    elastix.SetMovingImage(moving_sitk)
    elastix.SetLogToConsole(False)
    pm = sitk.GetDefaultParameterMap('bspline')
    pm['NumberOfResolutions'] = ['4']
    pm['MaximumNumberOfIterations'] = ['1500']
    pm['FinalGridSpacingInPhysicalUnits'] = ['25']
    pm['NumberOfSpatialSamples'] = ['6000']
    pm['GridSpacingSchedule'] = ['8.0', '4.0', '2.0', '1.0']
    pm['ImagePyramidSchedule'] = ['16', '16', '8', '8', '4', '4', '2', '2']

    # Dual metric: MI + landmark correspondences if points provided
    pts_files = []
    if fixed_points and moving_points and len(fixed_points) > 0:
        pm['Registration'] = ['MultiMetricMultiResolutionRegistration']
        pm['Metric'] = ['AdvancedMattesMutualInformation',
                         'CorrespondingPointsEuclideanDistanceMetric']
        pm['Metric0Weight'] = ['1.0']
        pm['Metric1Weight'] = ['0.5']
        # Write point set files in elastix format
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
    else:
        pm['Metric'] = ['AdvancedMattesMutualInformation']

    elastix.SetParameterMap(pm)
    try:
        elastix.Execute()
        tp = elastix.GetTransformParameterMap()
        transformix = sitk.TransformixImageFilter()
        transformix.SetTransformParameterMap(tp)
        transformix.SetLogToConsole(False)
        transformix.SetMovingImage(sitk.GetImageFromArray(moving_raw))
        transformix.Execute()
        warped_raw = sitk.GetArrayFromImage(transformix.GetResultImage())
        warped_extras = []
        if extra_images:
            for img in extra_images:
                tx = sitk.TransformixImageFilter()
                tx.SetTransformParameterMap(tp)
                tx.SetLogToConsole(False)
                tx.SetMovingImage(sitk.GetImageFromArray(img))
                tx.Execute()
                warped_extras.append(sitk.GetArrayFromImage(tx.GetResultImage()))
        return warped_raw, warped_extras
    except Exception:
        return moving_raw, [img for img in (extra_images or [])]
    finally:
        for f in pts_files:
            try: os.unlink(f)
            except OSError: pass


def crop_patch(img, cx, cy, r):
    """Crop 2r x 2r patch from img centered at (cx, cy), with zero-padding."""
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
# Process each tile — only z-slices with landmarks
# Reuse saved elastix transforms from build_all_tiles_elastix_v2.py
# ============================================================
TRANSFORMS_BASE = f'{BASE}/png_exports/registration_per_tile_elastix'
all_tiles_data = {}

for tile in sorted(tile_lm_files.keys()):
    # Force elastix only (no TPS)
    use_tps = False
    tps_data = None
    tfm_dir = f'{TRANSFORMS_BASE}/{tile}/transforms'
    if not os.path.isdir(tfm_dir) or not glob.glob(f'{tfm_dir}/elastix_tp_*.txt'):
        print(f"\n  {tile}: no saved transforms, skipping")
        continue
    print(f"\n{'='*60}")
    print(f"  {tile}")
    print(f"{'='*60}")

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
        print(f"  SKIP: only {N_LM} landmarks")
        continue

    nd2_z_vals = []
    for i in range(N_LM):
        x, y = ev_nd2[i, 0], ev_nd2[i, 1]
        c = int(round(np.clip(x, 10, nd2_h - 11)))
        r = int(round(np.clip(y, 10, nd2_h - 11)))
        intensities = [nd2_slices[z][r-10:r+10, c-10:c+10].mean() for z in range(12)]
        nd2_z_vals.append(find_z_gaussian(intensities))

    src = np.column_stack([pcd_iv[:, 2] * IV_XY_UM, pcd_iv[:, 1] * IV_XY_UM, pcd_iv[:, 0] * IV_Z_UM])
    dst = np.column_stack([ev_nd2[:, 0] * ND2_XY_UM, ev_nd2[:, 1] * ND2_XY_UM, np.array(nd2_z_vals) * ND2_Z_UM])
    src_h = np.hstack([src, np.ones((N_LM, 1))])
    A_T, _, _, _ = np.linalg.lstsq(src_h, dst, rcond=None)
    A = A_T.T
    predicted = src_h @ A_T
    errors = np.sqrt(np.sum((predicted - dst) ** 2, axis=1))
    # 2D XY affine error in µm (what you see in patches)
    errors_xy_um = np.sqrt((predicted[:, 0] - dst[:, 0])**2 + (predicted[:, 1] - dst[:, 1])**2)
    print(f"  {N_LM} lm | affine: mean={errors.mean():.1f}µm (xy={errors_xy_um.mean():.1f}µm)")

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

    # No pre-filtering — filter by actual blob position after warping
    good_lm = set(range(N_LM))

    # Group landmarks by known (z_iv, z_nd2) pairs — no 3D affine for z mapping
    z_pair_to_lm = defaultdict(list)
    for i in range(N_LM):
        if i not in good_lm:
            continue
        z_iv = int(round(pcd_iv[i, 0]))
        z_iv = max(0, min(nz_iv - 1, z_iv))
        z_nd2 = int(round(np.clip(nd2_z_vals[i], 0, 11)))
        z_pair_to_lm[(z_iv, z_nd2)].append(i)

    tile_patches = []
    elastix_cache = {}

    for (z_iv, z_nd2), lm_indices in sorted(z_pair_to_lm.items()):

        if (z_iv, z_nd2) not in elastix_cache:
            nd2_sl = nd2_slices[z_nd2]

            # Load saved 2D affine from tile registration build
            slice_path = f'{tfm_dir}/slice_ziv{z_iv:02d}_znd2{z_nd2:02d}.npz'
            tp_path = f'{tfm_dir}/elastix_tp_ziv{z_iv:02d}_znd2{z_nd2:02d}.txt'

            if os.path.exists(slice_path):
                sd = np.load(slice_path)
                M2d = sd['M2d']
            else:
                M2d = np.array([
                    [M_inv[2, 2], M_inv[2, 1], M_inv[2, 0] * z_nd2 + offset_inv[2]],
                    [M_inv[1, 2], M_inv[1, 1], M_inv[1, 0] * z_nd2 + offset_inv[1]],
                ], dtype=np.float64)

            iv_affine_raw = cv2.warpAffine(iv_vol_raw[z_iv], M2d, (nd2_w, nd2_h),
                                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)

            if use_tps:
                # TPS: compute remap once, warp in-vivo image + binary blobs
                print(f"    Affine+TPS warp iv_z={z_iv} → nd2_z={z_nd2} ({len(lm_indices)} lm) ...", end="", flush=True)
                tps_maps = tps_remap_maps(nd2_h, nd2_w,
                                          tps_data['W_inv'], tps_data['a_inv'], tps_data['src_inv'])
                iv_tps_raw = tps_warp_image(iv_affine_raw, None, None, None, remap=tps_maps)
                # Warp binary landmark blobs through affine+TPS
                warped_binaries = {}
                for i in lm_indices:
                    bimg = np.zeros((ny_iv, nx_iv), dtype=np.float32)
                    ly = int(round(pcd_iv[i, 1]))
                    lx = int(round(pcd_iv[i, 2]))
                    for dy in range(-8, 9):
                        for dx in range(-8, 9):
                            yy, xx = ly + dy, lx + dx
                            if 0 <= yy < ny_iv and 0 <= xx < nx_iv:
                                bimg[yy, xx] = 255.0 * np.exp(-0.5 * (dy*dy + dx*dx) / 9.0)
                    bimg_affine = cv2.warpAffine(bimg, M2d, (nd2_w, nd2_h),
                                                  flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
                    warped_binaries[i] = cv2.remap(bimg_affine, tps_maps[0], tps_maps[1],
                                                    cv2.INTER_LINEAR, borderValue=0)
                print(" done")
                elastix_cache[(z_iv, z_nd2)] = (nd2_sl, iv_tps_raw, z_nd2, warped_binaries)
            elif not os.path.exists(tp_path):
                print(f"    SKIP iv_z={z_iv} → nd2_z={z_nd2}: no saved elastix")
                continue
            else:
                # Create binary landmark images for all landmarks at this z pair
                binary_imgs = {}
                for i in lm_indices:
                    bimg = np.zeros((ny_iv, nx_iv), dtype=np.float32)
                    ly = int(round(pcd_iv[i, 1]))
                    lx = int(round(pcd_iv[i, 2]))
                    for dy in range(-8, 9):
                        for dx in range(-8, 9):
                            yy, xx = ly + dy, lx + dx
                            if 0 <= yy < ny_iv and 0 <= xx < nx_iv:
                                bimg[yy, xx] = 255.0 * np.exp(-0.5 * (dy*dy + dx*dx) / 9.0)
                    bimg_affine = cv2.warpAffine(bimg, M2d, (nd2_w, nd2_h),
                                                  flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
                    binary_imgs[i] = bimg_affine

                # Load saved elastix transform and apply via transformix
                print(f"    Transformix iv_z={z_iv} → nd2_z={z_nd2} ({len(lm_indices)} lm) ...", end="", flush=True)
                tp = sitk.ReadParameterFile(tp_path)
                tx = sitk.TransformixImageFilter()
                tx.SetTransformParameterMap((tp,))
                tx.SetLogToConsole(False)

                # Warp raw in-vivo
                tx.SetMovingImage(sitk.GetImageFromArray(iv_affine_raw))
                tx.Execute()
                iv_elastix_raw = sitk.GetArrayFromImage(tx.GetResultImage())

                # Warp binary landmarks
                warped_binaries = {}
                for i in lm_indices:
                    tx2 = sitk.TransformixImageFilter()
                    tx2.SetTransformParameterMap((tp,))
                    tx2.SetLogToConsole(False)
                    tx2.SetMovingImage(sitk.GetImageFromArray(binary_imgs[i]))
                    tx2.Execute()
                    warped_binaries[i] = sitk.GetArrayFromImage(tx2.GetResultImage())

                print(" done")
                elastix_cache[(z_iv, z_nd2)] = (nd2_sl, iv_elastix_raw, z_nd2, warped_binaries)

        nd2_sl, iv_elastix_raw, z_nd2_cached, warped_binaries = elastix_cache[(z_iv, z_nd2)]

        ev_norm = norm8(nd2_sl)
        iv_norm = norm8(iv_elastix_raw)

        iv_raw_norm = norm8(iv_vol_raw[z_iv])

        for i in lm_indices:
            ex_x = int(round(ev_nd2[i, 0]))
            ex_y = int(round(ev_nd2[i, 1]))
            pr_x = int(round(predicted[i, 0] / ND2_XY_UM))
            pr_y = int(round(predicted[i, 1] / ND2_XY_UM))

            cx, cy = ex_x, ex_y

            ev_patch = crop_patch(ev_norm, cx, cy, PATCH_R)
            iv_patch = crop_patch(iv_norm, cx, cy, PATCH_R)

            # Raw in-vivo patch (unwarped) at original landmark position
            # pcd_iv is (z, y, x) in JY306 pixel coords
            iv_raw_x = int(round(pcd_iv[i, 2]))
            iv_raw_y = int(round(pcd_iv[i, 1]))
            iv_raw_patch = crop_patch(iv_raw_norm, iv_raw_x, iv_raw_y, PATCH_R)

            ev_rgb = cv2.cvtColor(ev_patch, cv2.COLOR_GRAY2BGR)
            iv_rgb = cv2.cvtColor(iv_patch, cv2.COLOR_GRAY2BGR)
            iv_raw_rgb = cv2.cvtColor(iv_raw_patch, cv2.COLOR_GRAY2BGR)

            ov_rgb = np.zeros((2*PATCH_R, 2*PATCH_R, 3), dtype=np.uint8)
            ov_rgb[:, :, 1] = ev_patch
            ov_rgb[:, :, 0] = iv_patch
            ov_rgb[:, :, 2] = iv_patch

            # Compute corrected crosshair position
            aff_dx, aff_dy = pr_x - cx, pr_y - cy
            aff_err_px = float(np.sqrt(aff_dx**2 + aff_dy**2))

            if use_tps and tps_data is not None:
                # TPS point transform: predicted → corrected
                tps_x, tps_y = tps_transform_pt(
                    tps_data['W'], tps_data['a'], tps_data['src'],
                    float(predicted[i, 0] / ND2_XY_UM),
                    float(predicted[i, 1] / ND2_XY_UM))
                tps_dx = tps_x - cx
                tps_dy = tps_y - cy
                tps_err_px = float(np.sqrt(tps_dx**2 + tps_dy**2))
                final_dx = tps_dx
                final_dy = tps_dy
                final_err_px = tps_err_px
                corr_method = f"tps={tps_err_px*ND2_XY_UM:.1f}µm"
            else:
                final_dx = aff_dx
                final_dy = aff_dy
                final_err_px = aff_err_px
                corr_method = "aff"

            # Warp diagnostic: warped in-vivo blob (affine+TPS) vs ex-vivo crosshair
            WARP_ERR_MAX_UM = 5.0
            if i in warped_binaries:
                blob_full = warped_binaries[i]
                # Measure actual blob centroid distance from ex-vivo landmark
                bmax_full = blob_full.max()
                if bmax_full > 1:
                    # Weighted centroid of blob in full image coords
                    yy, xx = np.where(blob_full > bmax_full * 0.1)
                    if len(xx) > 0:
                        ww = blob_full[yy, xx]
                        blob_cx = float(np.average(xx, weights=ww))
                        blob_cy = float(np.average(yy, weights=ww))
                        blob_dist_um = np.sqrt((blob_cx - cx)**2 + (blob_cy - cy)**2) * ND2_XY_UM
                    else:
                        blob_dist_um = 999.0
                else:
                    blob_dist_um = 999.0
                if blob_dist_um > WARP_ERR_MAX_UM:
                    continue
                blob_patch = crop_patch(blob_full, cx, cy, PATCH_R)
                bmax = blob_patch.max()
                if bmax > 1:
                    blob_norm = np.clip(blob_patch / bmax * 255, 0, 255).astype(np.uint8)
                else:
                    blob_norm = np.zeros((2*PATCH_R, 2*PATCH_R), dtype=np.uint8)
                bin_vis = np.zeros((2*PATCH_R, 2*PATCH_R, 3), dtype=np.uint8)
                bin_vis[:, :, 1] = blob_norm
            else:
                continue  # no blob = can't verify

            tile_patches.append({
                'idx': i,
                'z_iv': int(pcd_iv[i, 0]),
                'z_nd2': z_nd2,
                'z_nd2_gauss': nd2_z_vals[i],
                'ex_x': ex_x, 'ex_y': ex_y,
                'pr_x': pr_x, 'pr_y': pr_y,
                'iv_raw_x': iv_raw_x, 'iv_raw_y': iv_raw_y,
                'pred_dx': final_dx, 'pred_dy': final_dy,
                'err_um': float(errors[i]),
                'elx_err_px': final_err_px,
                'corr_method': corr_method,
                'ev_b64': to_b64(ev_rgb),
                'iv_b64': to_b64(iv_rgb),
                'iv_raw_b64': to_b64(iv_raw_rgb),
                'ov_b64': to_b64(ov_rgb),
                'bin_b64': to_b64(bin_vis),
            })

    all_tiles_data[tile] = {
        'n_lm': N_LM,
        'mean_err': float(errors.mean()),
        'patches': tile_patches,
    }
    print(f"  {len(tile_patches)} patches extracted")
    del nd2_slices, elastix_cache


# ============================================================
# Build HTML — crosshairs as SVG overlays with toggle
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
    tile_options += f'<option value="{tile}"{sel}>{tile} ({td["n_lm"]} lm, err={td["mean_err"]:.1f}µm)</option>\n'

    cards = ""
    for p in td['patches']:
        ecx, ecy = PATCH_R, PATCH_R
        pcx, pcy = PATCH_R + p['pred_dx'], PATCH_R + p['pred_dy']

        cards += f"""
        <div class="card">
          <div class="card-header">LM #{p['idx']} | iv_z={p['z_iv']}→nd2_z={p['z_nd2']} (g={p['z_nd2_gauss']:.1f}) | aff={p['err_um']:.1f}µm | {p['corr_method']}</div>
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
      <div class="tile-info">{tile}: {td['n_lm']} landmarks | Mean affine error: {td['mean_err']:.1f}µm</div>
      <div class="cards-grid">{cards}
      </div>
    </div>"""
    first = False

html = f"""<!DOCTYPE html>
<html>
<head>
<title>Landmark Patches — Ex-vivo vs In-vivo (Elastix, blob filter)</title>
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
<h1>Landmark Patches: Ex-vivo vs In-vivo (Elastix, blob filter &lt;5µm)</h1>
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
  <span class="legend">Blue=exvivo actual | Red=invivo predicted | Green/Magenta=overlay</span>
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
print(f"\nSaved: {OUT_HTML}")
print(f"Total landmarks across all tiles: {sum(len(td['patches']) for td in all_tiles_data.values())}")
