"""
Build assets for scene 5b: stitched 3D overlay (all tiles).

Same approach as row2_1's per-tile 3D + v4 viewer stitching:
  - Per tile: load raw nd2 PNGs (4200x4200)
  - For each invivo z: warp with M2d, NCC match to best nd2 z
  - Stitch using full v5 pipeline (IOU rigid + elastix + cumulative)
    — same as build_viewer_warped_invivo_3d_v4.py
  - Build hot(invivo) + green(exvivo) overlay

Output: animation/scene5b_assets_v2.npz
"""

import numpy as np, cv2, os, json, glob
import tifffile
import SimpleITK as sitk

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_NPZ = f'{BASE}/animation/scene5b_assets_v2.npz'

Z_STEP = 2  # every 2nd native z for the output

def norm8(img, lo=1, hi=99.5):
    v = img.ravel(); v = v[v > 0]
    if len(v) < 50: return np.zeros(img.shape, np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img.astype(np.float32) - p1) / max(p2 - p1, 1) * 255, 0, 255).astype(np.uint8)

def normalize_u8_f(img):
    vals = img[img > 0]
    if len(vals) < 100: return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(vals, [2, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.float32)

def make_hot(u8):
    hot = cv2.applyColorMap(u8, cv2.COLORMAP_HOT)
    hot[u8 < 3] = 0
    return hot

def ncc(a, b):
    mask = (a > 5) & (b > 5)
    if mask.sum() < 100: return -1
    af = a[mask].astype(np.float32); af -= af.mean()
    bf = b[mask].astype(np.float32); bf -= bf.mean()
    return float(np.sum(af * bf) / (np.sqrt(np.sum(af**2) * np.sum(bf**2)) + 1e-8))

def stitch_tile_v5(sl, tile, tile_idx_map, iou_transforms, cum_iou,
                   stitch_elastix_dir, tile_order, canvas_w, canvas_h):
    """Full v5 stitch: IOU rigid + elastix + cumulative (same as v4 viewer)."""
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
        sl_n = normalize_u8_f(sl_rigid)
        sl_itk = sitk.GetImageFromArray(sl_n)
        tfm = sitk.ReadParameterFile(tfm_file)
        transformix = sitk.TransformixImageFilter()
        transformix.SetTransformParameterMap(tfm)
        transformix.SetMovingImage(sl_itk)
        transformix.LogToConsoleOff()
        try:
            transformix.Execute()
            sl_deformed = sitk.GetArrayFromImage(transformix.GetResultImage())
        except:
            sl_deformed = sl_n
    else:
        sl_deformed = normalize_u8_f(sl_rigid) if sl_rigid.max() > 0 else sl_rigid
    M_cum = np.array(cum_iou[tile])[:2, :]
    return cv2.warpAffine(sl_deformed.astype(np.float32), M_cum, (canvas_w, canvas_h),
                           flags=cv2.INTER_LINEAR, borderValue=0)

# ── Load stitch params ──
print("Loading stitch params...")
with open(f'{BASE}/registration_video/stitch_v5_params.json') as f:
    params = json.load(f)
TILE_ORDER = params['tile_order']
tile_z_offsets = params['tile_z_offsets']
canvas_w = params['canvas_w']
canvas_h = params['canvas_h']
cum_iou = params['cumulative_iou']
stitch_elastix_dir = params['elastix_dir']
total_z_native = max(tile_z_offsets.values()) + 12

with open(f'{BASE}/registration_video/auto_align_transforms_iou_v4.json') as f:
    iou_transforms = json.load(f)
tile_idx_map = {t: i for i, t in enumerate(TILE_ORDER)}

# ── Load in-vivo ──
print("Loading JY306 in-vivo...")
iv_vol = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv = iv_vol.shape[0]
print(f"  {iv_vol.shape}")

# ── Load per-tile data ──
print("Loading per-tile nd2 stacks and transforms...")
tile_data = {}

for tile in TILE_ORDER:
    pkl_path = f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz'
    nd2_files = sorted(glob.glob(f'{BASE}/png_exports/registration_video/{tile}/GFP_z*.png'))
    if not os.path.exists(pkl_path) or not nd2_files:
        continue

    pkl = np.load(pkl_path)
    M2d = pkl['M2d_jy306_to_nd2']
    ev = pkl['ev_nd2']
    iv_lm = pkl['pcd_invivo_jy306']

    margin = 350
    crop_x0 = max(0, int(ev[:, 0].min() - margin))
    crop_y0 = max(0, int(ev[:, 1].min() - margin))
    crop_x1 = min(4200, int(ev[:, 0].max() + margin))
    crop_y1 = min(4200, int(ev[:, 1].max() + margin))

    iv_z_min = max(0, int(iv_lm[:, 0].min()) - 1)
    iv_z_max = min(nz_iv - 1, int(iv_lm[:, 0].max()) + 1)

    nd2_stack = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                          for f in nd2_files])

    tile_data[tile] = {
        'M2d': M2d, 'nd2_stack': nd2_stack,
        'crop': (crop_x0, crop_y0, crop_x1, crop_y1),
        'iv_z_range': (iv_z_min, iv_z_max),
        'z_offset': tile_z_offsets.get(tile, 0),
    }
    print(f"  {tile}: {len(nd2_files)} nd2 z, iv_z={iv_z_min}-{iv_z_max}, offset={tile_z_offsets.get(tile,0)}")

print(f"  {len(tile_data)} tiles loaded")

# ── Per-tile: warp, NCC match, stitch ──
print(f"\nBuilding per-tile overlays (NCC z-matching + full v5 stitch)...")

invivo_by_z = {}
exvivo_by_z = {}

for tile in sorted(tile_data.keys()):
    td = tile_data[tile]
    M2d = td['M2d']
    nd2_stack = td['nd2_stack']
    crop_x0, crop_y0, crop_x1, crop_y1 = td['crop']
    iv_z_min, iv_z_max = td['iv_z_range']
    z_offset = td['z_offset']
    nd2_nz = len(nd2_stack)

    print(f"  {tile}: ", end='', flush=True)
    count = 0

    for z_iv in range(iv_z_min, iv_z_max + 1):
        iv_u8 = norm8(iv_vol[z_iv])
        warped_iv = cv2.warpAffine(iv_u8, M2d, (4200, 4200),
                                    flags=cv2.INTER_LINEAR, borderValue=0)
        warped_crop = warped_iv[crop_y0:crop_y1, crop_x0:crop_x1]

        # NCC match
        best_ncc, best_nd2_z = -1, 0
        for zi in range(nd2_nz):
            nd2_z_img = nd2_stack[zi].astype(np.uint8)
            nd2_c = nd2_z_img[crop_y0:min(crop_y1, 4200), crop_x0:min(crop_x1, 4200)]
            wc = warped_crop[:nd2_c.shape[0], :nd2_c.shape[1]]
            score = ncc(norm8(wc), norm8(nd2_c))
            if score > best_ncc:
                best_ncc, best_nd2_z = score, zi

        native_z = z_offset + best_nd2_z

        # Full v5 stitch for invivo
        iv_stitched = stitch_tile_v5(
            warped_iv.astype(np.float32), tile, tile_idx_map,
            iou_transforms, cum_iou, stitch_elastix_dir, TILE_ORDER,
            canvas_w, canvas_h)

        if native_z not in invivo_by_z:
            invivo_by_z[native_z] = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        invivo_by_z[native_z] = np.maximum(invivo_by_z[native_z], iv_stitched)
        # Full v5 stitch for exvivo — masked to invivo coverage area
        nd2_best = nd2_stack[best_nd2_z].astype(np.float32)
        # Mask nd2 to only show where warped invivo has content
        iv_mask = (warped_iv > 5).astype(np.float32)
        # Dilate mask slightly to avoid hard edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        iv_mask = cv2.dilate(iv_mask, kernel)
        nd2_masked = nd2_best * iv_mask
        ev_stitched = stitch_tile_v5(
            nd2_masked, tile, tile_idx_map,
            iou_transforms, cum_iou, stitch_elastix_dir, TILE_ORDER,
            canvas_w, canvas_h)

        if native_z not in exvivo_by_z:
            exvivo_by_z[native_z] = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        exvivo_by_z[native_z] = np.maximum(exvivo_by_z[native_z], ev_stitched)

        count += 1

    print(f"{count} z-slices")

del iv_vol
print(f"Built {len(invivo_by_z)} unique native z-slices")

# ── Build overlay slices ──
all_z = sorted(invivo_by_z.keys())
sel_z = all_z[::Z_STEP]
print(f"\nBuilding {len(sel_z)} display overlays (from {len(all_z)} total, step={Z_STEP})...")

slice_w = canvas_w
slice_h = canvas_h

overlay_slices = []
z_indices = []

for zi, nz in enumerate(sel_z):
    if zi % 10 == 0:
        print(f"  {zi+1}/{len(sel_z)} (native_z={nz})...", flush=True)

    iv_canvas = invivo_by_z[nz]
    ev_canvas = exvivo_by_z.get(nz, np.zeros_like(iv_canvas))

    # Full resolution — no downsampling
    iv_u8 = norm8(iv_canvas)
    ev_u8 = norm8(ev_canvas)

    hot = make_hot(iv_u8)
    green = np.zeros((slice_h, slice_w, 3), np.uint8)
    green[:, :, 1] = ev_u8
    ov = cv2.addWeighted(green, 0.5, hot, 0.5, 0)

    overlay_slices.append(ov)
    z_indices.append(nz)

del invivo_by_z, exvivo_by_z

overlay_slices = np.array(overlay_slices)
print(f"  Shape: {overlay_slices.shape}")

# ── Save ──
print(f"Saving to {OUT_NPZ}...")
np.savez_compressed(OUT_NPZ,
    overlay_slices=overlay_slices,
    z_indices=np.array(z_indices),
    slice_w=slice_w,
    slice_h=slice_h,
    canvas_shape=np.array([canvas_h, canvas_w]),
)
fsize = os.path.getsize(OUT_NPZ) / 1e6
print(f"Done! {OUT_NPZ} ({fsize:.1f} MB)")
print(f"  {len(overlay_slices)} overlay slices, {slice_w}x{slice_h}")
