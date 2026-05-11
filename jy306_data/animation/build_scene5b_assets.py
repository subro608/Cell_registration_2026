"""
Build overlay slices for scene 5b: full stitched ex-vivo + warped in-vivo.

Uses the SAME pipeline as build_viewer_warped_invivo_3d_v4.py:
  - Per-tile PKL M2d warp + v5 stitching (IOU rigid + elastix + cumulative)
  - Full resolution (no downsampling during warp)
  - Ex-vivo: stitched_gfp_fullres_v5_1um_isotropic.tif

Saves: animation/scene5b_assets.npz
  - overlay_slices: (N, slice_h, slice_w, 3) uint8
  - z_ex_indices: which exvivo z each slice corresponds to
  - slice_w, slice_h: display dimensions
"""

import numpy as np, cv2, tifffile, os, json, glob
import SimpleITK as sitk

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_NPZ = f'{BASE}/animation/scene5b_assets.npz'

Z_STEP = 4  # every 4th exvivo z-slice (1µm isotropic → 4µm steps)
ND2_Z_UM = 2.0
MAX_SLICE = 600  # display size

def norm_u8(img, lo=2, hi=99.5):
    v = img.ravel(); v = v[v > 0]
    if len(v) < 50: return np.zeros(img.shape, np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img.astype(np.float32) - p1) / max(p2 - p1, 1) * 255, 0, 255).astype(np.uint8)

def normalize_u8_f(img):
    vals = img[img > 0]
    if len(vals) < 100: return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(vals, [2, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.float32)

def stitch_tile_v5(sl, tile, tile_idx_map, iou_transforms, cum_iou,
                    stitch_elastix_dir, tile_order, canvas_w, canvas_h):
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
    M_prev = np.array(cum_iou[tile])[:2, :]
    return cv2.warpAffine(sl_deformed.astype(np.float32), M_prev, (canvas_w, canvas_h),
                           flags=cv2.INTER_LINEAR, borderValue=0)

# ── Load stitch params ──
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
tile_idx_map = {t: i for i, t in enumerate(TILE_ORDER)}

# ── Load in-vivo ──
print("Loading JY306 in-vivo...")
iv_vol = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv = iv_vol.shape[0]
print(f"  {iv_vol.shape}")

# ── Load landmarks + PKL transforms ──
print("Loading landmarks and PKL transforms...")
lm_files = sorted(glob.glob(f'{BASE}/registration_video/landmarks_nd2_native_*.npz'))
legacy = f'{BASE}/registration_video/landmarks_27_nd2_native.npz'
if os.path.exists(legacy):
    lm_files.append(legacy)

tile_data = {}
for lm_file in lm_files:
    bn = os.path.basename(lm_file)
    tile = 'row2_1' if 'landmarks_27_nd2_native' in bn else bn.replace('landmarks_nd2_native_', '').replace('.npz', '')
    if tile not in TILE_ORDER: continue
    pkl_path = f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz'
    if not os.path.exists(pkl_path): continue

    pkl = np.load(pkl_path)
    M2d = pkl['M2d_jy306_to_nd2']
    d = np.load(lm_file)
    ev_nd2 = d['ev_nd2']
    pcd_iv = d['pcd_invivo_jy306']

    z_nd2_to_z_iv = {}
    for i in range(len(ev_nd2)):
        z_nd2_i = int(round(np.clip(ev_nd2[i, 2], 0, 11)))
        z_iv_i = int(round(pcd_iv[i, 0]))
        z_nd2_to_z_iv.setdefault(z_nd2_i, []).append(z_iv_i)
    z_mapping = {z: int(round(np.median(zivs))) for z, zivs in z_nd2_to_z_iv.items()}

    tile_data[tile] = {'M2d': M2d, 'z_mapping': z_mapping, 'z_offset': tile_z_offsets[tile]}

print(f"  {len(tile_data)} tiles ready")

# ── Determine needed z-slices ──
EX_TIFF = f"{BASE}/registration_video/stitched/stitched_gfp_fullres_v5_1um_isotropic.tif"
with tifffile.TiffFile(EX_TIFF) as tif:
    ex_nz = len(tif.pages)
    ex_h, ex_w = tif.pages[0].shape
print(f"ExVivo: ({ex_nz}, {ex_h}, {ex_w})")

z_ex_indices = list(range(0, ex_nz, Z_STEP))
n_slices = len(z_ex_indices)

z_native_needed = set()
for z_ex in z_ex_indices:
    z_native = int(round(z_ex / ND2_Z_UM))
    if 0 <= z_native < total_z_native:
        z_native_needed.add(z_native)

# ── Build warped invivo for needed native z-slices ──
print(f"\nBuilding warped invivo (full res {canvas_w}x{canvas_h})...")
print(f"  Need {len(z_native_needed)} native z-slices")
warped_by_native_z = {}

for tile in sorted(tile_data.keys()):
    td = tile_data[tile]
    M2d = td['M2d']
    z_mapping = td['z_mapping']
    z_offset = td['z_offset']
    print(f"  {tile} (z_offset={z_offset})...", end='', flush=True)
    count = 0

    for z_nd2 in range(12):
        z_out = z_offset + z_nd2
        if z_out not in z_native_needed:
            continue

        if z_nd2 in z_mapping:
            z_iv = z_mapping[z_nd2]
        else:
            nearest = min(z_mapping.keys(), key=lambda k: abs(k - z_nd2), default=None)
            if nearest is None: continue
            z_iv = z_mapping[nearest]
        z_iv = np.clip(z_iv, 0, nz_iv - 1)

        iv_warped = cv2.warpAffine(iv_vol[z_iv], M2d, (4200, 4200),
                                    flags=cv2.INTER_LINEAR, borderValue=0)
        iv_stitched = stitch_tile_v5(iv_warped, tile, tile_idx_map, iou_transforms,
                                      cum_iou, STITCH_ELASTIX_DIR, TILE_ORDER,
                                      canvas_w, canvas_h)
        if z_out not in warped_by_native_z:
            warped_by_native_z[z_out] = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        warped_by_native_z[z_out] = np.maximum(warped_by_native_z[z_out], iv_stitched)
        count += 1

    print(f" {count} slices")

del iv_vol
print(f"Built {len(warped_by_native_z)} warped z-slices")

# ── Build overlay slices ──
print(f"\nBuilding overlay slices (display {MAX_SLICE}px)...")
scale_fit = MAX_SLICE / max(ex_w, ex_h)
slice_w = int(ex_w * scale_fit)
slice_h = int(ex_h * scale_fit)

overlay_slices = []
with tifffile.TiffFile(EX_TIFF) as tif:
    for si, z_ex in enumerate(z_ex_indices):
        if si % 20 == 0:
            print(f"  {si+1}/{n_slices} (z_ex={z_ex})...", flush=True)

        ex_slice = tif.pages[z_ex].asarray().astype(np.float32)
        ex_u8 = norm_u8(ex_slice)

        z_native = int(round(z_ex / ND2_Z_UM))
        iv_u8 = np.zeros((ex_h, ex_w), dtype=np.uint8)
        if z_native in warped_by_native_z:
            wsl = warped_by_native_z[z_native]
            if wsl.max() > 0:
                iv_u8 = cv2.resize(norm_u8(wsl), (ex_w, ex_h), interpolation=cv2.INTER_LINEAR)

        ov = np.zeros((ex_h, ex_w, 3), np.uint8)
        ov[:,:,1] = ex_u8
        ov[:,:,2] = iv_u8
        ov_small = cv2.resize(ov, (slice_w, slice_h), interpolation=cv2.INTER_AREA)
        overlay_slices.append(ov_small)

del warped_by_native_z

overlay_slices = np.array(overlay_slices)
print(f"  Shape: {overlay_slices.shape}")

# ── Save ──
print(f"Saving to {OUT_NPZ}...")
np.savez_compressed(OUT_NPZ,
    overlay_slices=overlay_slices,
    z_ex_indices=np.array(z_ex_indices),
    slice_w=slice_w,
    slice_h=slice_h,
    ex_shape=np.array([ex_nz, ex_h, ex_w]),
)
fsize = os.path.getsize(OUT_NPZ) / 1e6
print(f"Done! Saved {OUT_NPZ} ({fsize:.1f} MB)")
print(f"  {n_slices} overlay slices, {slice_w}x{slice_h} each")
