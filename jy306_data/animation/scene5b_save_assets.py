"""
Save scene5b 3D tile assets (volumes, MERSCOPE, z-offsets, grid positions)
so rendering can reload instantly without NCC matching.
"""
import numpy as np, cv2, os, glob, tifffile, pickle

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT = f'{BASE}/animation/scene5b_assets_v3.npz'

TILES = ['row1_3',
         'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
         'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
         'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
         'row5_1']

CELL_H = 400
INTERP_PER_GAP = 2

MERSCOPE_MAP = {
    'row1_3': 'region_23_1_3.png', 'row2_1': 'region_17_2_1.png',
    'row2_2': 'region_18_2_2.png', 'row2_3': 'region_19_2_3.png',
    'row2_4': 'region_20_2_4.png', 'row2_5': 'region_21_2_5.png',
    'row3_1': 'region_16_3_1.png', 'row3_2': 'region_15_3_2.png',
    'row3_3': 'region_14_3_3.png', 'row3_4': 'region_13_3_4.png',
    'row3_5': 'region_12_3_5.png', 'row3_6': 'region_11_3_6.png',
    'row4_1': 'region_5_4_1.png',  'row4_2': 'region_6_4_2.png',
    'row4_3': 'region_7_4_3.png',  'row4_4': 'region_8_4_4.png',
    'row4_5': 'region_9_4_5.png',  'row4_6': 'region_10_4_6.png',
    'row5_1': 'region_4_5_1.png',
}
MERSCOPE_DIR = f'{BASE}/png_exports/merscope_overlay'


def norm8(img, lo=1, hi=99.5):
    v = img.ravel(); v = v[v > 0]
    if len(v) < 50: return np.zeros(img.shape, np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img - p1) / max(p2 - p1, 1) * 255, 0, 255).astype(np.uint8)


def ncc(a, b):
    mask = (a > 5) & (b > 5)
    if mask.sum() < 100: return -1
    af = a[mask].astype(np.float32); af -= af.mean()
    bf = b[mask].astype(np.float32); bf -= bf.mean()
    return float(np.sum(af * bf) / (np.sqrt(np.sum(af**2) * np.sum(bf**2)) + 1e-8))


print("Loading JY306 in-vivo volume...")
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_jy = jy306.shape[0]

print("Loading tissue masks...")
via_masks = np.load(f'{BASE}/registration_video/via_masks_v4.npz')

# Load stitch params for canvas positions
import json
with open(f'{BASE}/registration_video/stitch_v5_params.json') as f:
    stitch_params = json.load(f)

assets = {}

for tile in TILES:
    print(f"  Building 3D for {tile}...")
    nd2_files = sorted(glob.glob(f'{BASE}/png_exports/registration_video/{tile}/GFP_z*.png'))
    if not nd2_files: continue
    nd2_stack = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in nd2_files])

    pkl_path = f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz'
    if not os.path.exists(pkl_path): continue
    pkl = np.load(pkl_path)
    M2d = pkl['M2d_jy306_to_nd2']
    iv = pkl['pcd_invivo_jy306']
    ev = pkl['ev_nd2']

    MODE_Z = max(0, min(nz_jy - 1, int(round(np.median(iv[:, 0])))))

    margin_nd2 = 350
    crop_x0 = max(0, int(ev[:, 0].min() - margin_nd2))
    crop_y0 = max(0, int(ev[:, 1].min() - margin_nd2))
    crop_x1 = min(4200, int(ev[:, 0].max() + margin_nd2))
    crop_y1 = min(4200, int(ev[:, 1].max() + margin_nd2))

    scale_nd2 = CELL_H / (crop_y1 - crop_y0)
    cell_w = int((crop_x1 - crop_x0) * scale_nd2)
    cell_h = CELL_H
    crop_h_nd2 = crop_y1 - crop_y0

    iv_z_min = max(0, int(iv[:, 0].min()) - 1)
    iv_z_max = min(nz_jy - 1, int(iv[:, 0].max()) + 1)
    z_range = list(range(iv_z_min, iv_z_max + 1))

    # Load tissue mask and compute display-size version once
    tile_mask = via_masks[tile] if tile in via_masks else np.ones((4200, 4200), dtype=np.uint8)
    mask_crop = tile_mask[crop_y0:min(crop_y1, 4200), crop_x0:min(crop_x1, 4200)]
    mask_small = cv2.resize(mask_crop.astype(np.uint8), (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
    mask_small3 = np.stack([mask_small]*3, axis=-1)

    overlay_slices = []
    for z_iv in z_range:
        iv_u8 = norm8(jy306[z_iv])
        warped_iv = cv2.warpAffine(iv_u8, M2d, (4200, 4200), flags=cv2.INTER_LINEAR, borderValue=0)

        best_ncc, best_z = -1, 0
        for zi in range(len(nd2_stack)):
            nd2_c = nd2_stack[zi].astype(np.uint8)[crop_y0:min(crop_y1, nd2_stack.shape[1]),
                                                    crop_x0:min(crop_x1, nd2_stack.shape[2])]
            wc_crop = warped_iv[crop_y0:crop_y1, crop_x0:crop_x1]
            wc = wc_crop[:nd2_c.shape[0], :nd2_c.shape[1]]
            score = ncc(norm8(wc), norm8(nd2_c))
            if score > best_ncc: best_ncc, best_z = score, zi

        nd2_best = nd2_stack[best_z].astype(np.uint8)

        # Crop first, then normalize, then apply mask
        nd2_c = nd2_best[crop_y0:min(crop_y1, nd2_best.shape[0]),
                         crop_x0:min(crop_x1, nd2_best.shape[1])]
        wc = warped_iv[crop_y0:crop_y1, crop_x0:crop_x1]
        wc = wc[:nd2_c.shape[0], :nd2_c.shape[1]]

        # Normalize on full data (before masking)
        ev_u8 = norm8(nd2_c)
        iv_u8 = norm8(wc)[:nd2_c.shape[0], :nd2_c.shape[1]]

        # Apply mask after normalization
        mask_crop = tile_mask[crop_y0:min(crop_y1, tile_mask.shape[0]),
                              crop_x0:min(crop_x1, tile_mask.shape[1])]
        mc = mask_crop[:nd2_c.shape[0], :nd2_c.shape[1]]
        ev_u8 = ev_u8 * mc
        iv_u8 = iv_u8 * mc

        # Ex-vivo = magenta, In-vivo = green (full brightness)
        ov = np.zeros((nd2_c.shape[0], nd2_c.shape[1], 3), np.uint8)
        ov[:, :, 0] = ev_u8  # B (magenta)
        ov[:, :, 2] = ev_u8  # R (magenta)
        ov[:, :, 1] = iv_u8  # G (green)
        ov_small = cv2.resize(ov, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        overlay_slices.append(ov_small)

    n_slices = len(overlay_slices)

    # Correct z-spacing
    um_per_display_px = (crop_h_nd2 * 0.65) / CELL_H
    z_step_um = 3.0
    tile_z_spacing = z_step_um / um_per_display_px

    dense = []
    dense_z = []
    for i in range(n_slices):
        dense.append(overlay_slices[i])
        dense_z.append(i * tile_z_spacing)
        if i < n_slices - 1:
            for sub in range(1, INTERP_PER_GAP + 1):
                t_sub = sub / (INTERP_PER_GAP + 1)
                interp = (overlay_slices[i].astype(np.float32) * (1 - t_sub) +
                          overlay_slices[i + 1].astype(np.float32) * t_sub)
                dense.append(interp.astype(np.uint8))
                dense_z.append(i * tile_z_spacing + t_sub * tile_z_spacing)

    dense = np.array(dense)
    dense_z = np.array(dense_z, dtype=np.float64)
    center_z = (dense_z[-1] + dense_z[0]) / 2.0

    # Brightness normalize
    mx = dense.max()
    boost = 1.0
    if mx < 120:
        boost = min(255.0 / max(mx, 1), 3.0)
        dense = np.clip(dense.astype(np.float32) * boost, 0, 255).astype(np.uint8)
        print(f"    {tile}: boosted {boost:.1f}x")

    # MERSCOPE gene dots (right - left subtraction, masked, dilated)
    merscope_small = None
    if tile in MERSCOPE_MAP:
        mpath = f'{MERSCOPE_DIR}/{MERSCOPE_MAP[tile]}'
        if os.path.exists(mpath):
            mimg = cv2.imread(mpath)
            mw = mimg.shape[1]
            left = mimg[:, :mw // 2, :].astype(np.float32)
            right = mimg[:, mw // 2:, :].astype(np.float32)
            dots = np.clip(right - left, 0, 255).astype(np.uint8)
            # Apply same tissue mask as overlay
            dh, dw = min(dots.shape[0], tile_mask.shape[0]), min(dots.shape[1], tile_mask.shape[1])
            dots_masked = dots[:dh, :dw, :] * np.stack([tile_mask[:dh, :dw]]*3, axis=-1)
            # Crop
            dots_crop = dots_masked[crop_y0:min(crop_y1, dots_masked.shape[0]),
                                    crop_x0:min(crop_x1, dots_masked.shape[1])]
            # No dilation — keep dots small. Downscale directly.
            merscope_small = cv2.resize(dots_crop, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            # Apply display-size mask for pixel-perfect boundary match
            merscope_small = merscope_small * mask_small3

    # Positions from stitch params
    med_z = float(np.median(iv[:, 0]))
    M_canvas = np.array(stitch_params['cumulative_iou'][tile])
    canvas_center = M_canvas @ np.array([2100, 2100, 1])

    assets[tile] = {
        'dense': dense,
        'dense_z': dense_z,
        'center_z': center_z,
        'n_slices': n_slices,
        'cell_w': cell_w,
        'cell_h': cell_h,
        'crop_h_nd2': crop_h_nd2,
        'tile_z_spacing': tile_z_spacing,
        'med_z': med_z,
        'med_x': float(np.median(iv[:, 2])),
        'med_y': float(np.median(iv[:, 1])),
        'canvas_x': float(canvas_center[0]),
        'canvas_y': float(canvas_center[1]),
        'stitch_z_offset': stitch_params['tile_z_offsets'][tile],
        'boost': boost,
        'merscope': merscope_small,
    }
    print(f"    {len(dense)} dense slices, {cell_w}x{cell_h}, z_spacing={tile_z_spacing:.2f}")

del jy306

# Save as pickle (npz can't handle nested dicts with mixed types well)
out_pkl = f'{BASE}/animation/scene5b_assets_v3.pkl'
with open(out_pkl, 'wb') as f:
    pickle.dump(assets, f)

sz_mb = os.path.getsize(out_pkl) / 1e6
print(f"\nSaved {len(assets)} tiles to {out_pkl} ({sz_mb:.0f} MB)")
print(f"Tiles: {list(assets.keys())}")
print(f"Color convention: ex-vivo=magenta, in-vivo=green")
print(f"Z-step: 3.0 µm, XY: 0.645 µm/px")