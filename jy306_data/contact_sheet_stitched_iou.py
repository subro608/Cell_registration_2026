"""
Contact sheets: IOU-only stitched ex-vivo vs in-vivo JY306.

Propagates landmarks through IOU rigid chain only (no elastix),
then extracts patches from the IOU-only stitched 1um isotropic volume.

Left: stitched ex-vivo patch (1um isotropic)
Right: in-vivo JY306 patch

No SimpleITK needed.

Usage:
    python contact_sheet_stitched_iou.py
"""

import numpy as np
import cv2
import os
import json
import glob
import tifffile

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_DIR = f'{BASE}/png_exports/coarse_registration/contact_sheet/stitched_iou'
os.makedirs(OUT_DIR, exist_ok=True)

# Config
PATCH = 160
CROP_STITCHED = 100  # crop radius in 1um isotropic pixels
CROP_JY = 35
FULL_H = 400
MAX_COLS = 5

NATIVE_XY_UM = 0.645
NATIVE_Z_UM = 2.0

# ============================================================
# Load stitch params
# ============================================================
print("Loading stitch params...")
with open(f'{BASE}/registration_video/stitch_iou_only_params.json') as f:
    params = json.load(f)

TILE_ORDER = params['tile_order']
tile_z_offsets = {k: int(v) for k, v in params['tile_z_offsets'].items()}
canvas_w = params['canvas_w']
canvas_h = params['canvas_h']
offset_x = params['offset_x']
offset_y = params['offset_y']
cum_iou = {}
for k, v in params['cumulative_iou'].items():
    cum_iou[k] = np.array(v)

# Load IOU pair transforms
with open(f'{BASE}/registration_video/auto_align_transforms_iou_v4.json') as f:
    iou_transforms = json.load(f)

# ============================================================
# Load volumes
# ============================================================
print("Loading IOU-only stitched 1um isotropic volume...")
with tifffile.TiffFile(f'{BASE}/registration_video/stitched/stitched_gfp_iou_only_1um_isotropic.tif') as tif:
    n_pages = len(tif.pages)
    h, w = tif.pages[0].shape
    stitched = np.zeros((n_pages, h, w), dtype=np.uint16)
    for i, page in enumerate(tif.pages):
        stitched[i] = page.asarray()
print(f"  Shape: {stitched.shape}")

print("Loading JY306 in-vivo volume...")
jy306_vol = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
print(f"  Shape: {jy306_vol.shape}")

# ============================================================
# Helper functions
# ============================================================
def norm8(img, p_lo=1, p_hi=99.5):
    vals = img[img > 0]
    if len(vals) == 0:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    return np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


def propagate_point_to_stitched(x_nd2, y_nd2, z_nd2, tile_key):
    """
    Map a point from nd2 native (4200) space through cumulative IOU rigid
    to the stitched 1um isotropic volume. No elastix.
    """
    # Apply cumulative IOU directly
    M = cum_iou[tile_key]
    p = M @ np.array([x_nd2, y_nd2, 1.0])
    x_canvas, y_canvas = p[0], p[1]

    # Convert to 1um isotropic
    x_iso = x_canvas * NATIVE_XY_UM
    y_iso = y_canvas * NATIVE_XY_UM

    # Z: tile z-offset + slice z, converted to 1um
    z_native = tile_z_offsets[tile_key] + z_nd2
    z_iso = z_native * NATIVE_Z_UM

    return z_iso, y_iso, x_iso

# ============================================================
# Discover landmark files
# ============================================================
lm_files = sorted(glob.glob(f'{BASE}/registration_video/landmarks_nd2_native_*.npz'))
legacy = f'{BASE}/registration_video/landmarks_27_nd2_native.npz'
if os.path.exists(legacy):
    lm_files.append(legacy)

tile_to_idx = {k: i for i, k in enumerate(TILE_ORDER)}

print(f"\nFound {len(lm_files)} landmark files")

for lm_file in lm_files:
    bn = os.path.basename(lm_file)
    if 'landmarks_27_nd2_native' in bn:
        tile = 'row2_1'
    else:
        tile = bn.replace('landmarks_nd2_native_', '').replace('.npz', '')

    if tile not in tile_to_idx:
        print(f"  SKIP {tile}: not in tile order")
        continue

    tile_idx = tile_to_idx[tile]

    d = np.load(lm_file)
    ev_nd2 = d['ev_nd2']           # (N, 3) = (col, row, z_merc) in nd2 pixels
    pcd_iv = d['pcd_invivo_jy306']  # (N, 3) = (z, y, x) in JY306 pixels
    N_CELLS = ev_nd2.shape[0]

    print(f"\n{'='*50}")
    print(f"  {tile} ({N_CELLS} cells)")
    print(f"{'='*50}")

    # Find best nd2 z per cell (brightest z-slice)
    img_dir = f'{BASE}/png_exports/registration_video/{tile}'
    nd2_slices = []
    for zi in range(12):
        img = cv2.imread(f'{img_dir}/GFP_z{zi:03d}.png', cv2.IMREAD_UNCHANGED)
        if img is None:
            nd2_slices.append(np.zeros((4200, 4200), dtype=np.float32))
        else:
            nd2_slices.append(img.astype(np.float32))
    nd2_slices = np.array(nd2_slices)
    H_nd2 = nd2_slices.shape[1]

    cell_nd2_z = []
    for i in range(N_CELLS):
        c = int(round(np.clip(ev_nd2[i, 0], 10, H_nd2 - 11)))
        r = int(round(np.clip(ev_nd2[i, 1], 10, H_nd2 - 11)))
        intensities = [nd2_slices[z][r-10:r+10, c-10:c+10].mean() for z in range(12)]
        cell_nd2_z.append(np.argmax(intensities))
    cell_nd2_z = np.array(cell_nd2_z)
    jy_z_vals = pcd_iv[:, 0].astype(int)

    # Propagate landmarks to stitched space (IOU only)
    stitched_coords = []
    for i in range(N_CELLS):
        x_nd2 = ev_nd2[i, 0]
        y_nd2 = ev_nd2[i, 1]
        z_nd2 = cell_nd2_z[i]
        z_iso, y_iso, x_iso = propagate_point_to_stitched(x_nd2, y_nd2, z_nd2, tile)
        stitched_coords.append([z_iso, y_iso, x_iso])
    stitched_coords = np.array(stitched_coords)

    # ============================================================
    # Generate contact sheet
    # ============================================================
    ARROW_W = 20
    LABEL_H = 22
    PAIR_W = PATCH * 2 + ARROW_W
    PAIR_H = PATCH + LABEL_H

    # Overview: stitched MIP (small) + JY306 MIP
    z_lo_tile = int(round(tile_z_offsets[tile] * NATIVE_Z_UM))
    z_hi_tile = min(stitched.shape[0], int(round((tile_z_offsets[tile] + 12) * NATIVE_Z_UM)))
    stitched_tile_mip = np.max(stitched[z_lo_tile:z_hi_tile], axis=0).astype(np.float32)

    jy_mip = np.max(jy306_vol, axis=0).astype(np.float32)

    # Scale both to FULL_H
    st_h, st_w = stitched_tile_mip.shape
    st_scale = FULL_H / float(st_h)
    st_pw = int(st_w * st_scale)
    st_small = cv2.resize(norm8(stitched_tile_mip), (st_pw, FULL_H))

    jy_scale = FULL_H / 658.0
    jy_pw = int(629 * jy_scale)
    jy_small = cv2.resize(norm8(jy_mip), (jy_pw, FULL_H))

    arrow_gap = 30
    full_w_view = st_pw + arrow_gap + jy_pw
    full_view = np.zeros((FULL_H, full_w_view, 3), dtype=np.uint8)
    full_view[:, :st_pw] = cv2.cvtColor(st_small, cv2.COLOR_GRAY2BGR)
    full_view[:, st_pw + arrow_gap:] = cv2.cvtColor(jy_small, cv2.COLOR_GRAY2BGR)

    # Draw arrows
    for i in range(N_CELLS):
        lx = int(round(stitched_coords[i, 2] * st_scale))
        ly = int(round(stitched_coords[i, 1] * st_scale))
        rx = int(round(pcd_iv[i, 2] * jy_scale)) + st_pw + arrow_gap
        ry = int(round(pcd_iv[i, 1] * jy_scale))

        lx = np.clip(lx, 0, st_pw - 1)
        ly = np.clip(ly, 0, FULL_H - 1)
        rx = np.clip(rx, st_pw + arrow_gap, full_w_view - 1)
        ry = np.clip(ry, 0, FULL_H - 1)

        cv2.arrowedLine(full_view, (lx, ly), (rx, ry), (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.05)
        cv2.circle(full_view, (lx, ly), 3, (0, 255, 0), 1)
        cv2.circle(full_view, (rx, ry), 3, (0, 255, 0), 1)

    cv2.putText(full_view, f"{tile} stitched IOU-only MIP (1um iso, z={z_lo_tile}-{z_hi_tile})", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(full_view, "JY306 in-vivo MIP", (st_pw + arrow_gap + 8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    # Zoomed patches
    all_patches = []
    for idx in range(N_CELLS):
        z_st = int(round(stitched_coords[idx, 0]))
        y_st = int(round(stitched_coords[idx, 1]))
        x_st = int(round(stitched_coords[idx, 2]))
        jy_z = jy_z_vals[idx]

        # Stitched patch
        z_st = np.clip(z_st, 0, stitched.shape[0] - 1)
        st_slice = stitched[z_st].astype(np.float32)
        x1 = max(0, x_st - CROP_STITCHED)
        y1 = max(0, y_st - CROP_STITCHED)
        x2 = min(st_slice.shape[1], x_st + CROP_STITCHED)
        y2 = min(st_slice.shape[0], y_st + CROP_STITCHED)
        crop = st_slice[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((10, 10), dtype=np.float32)
        st_crop = cv2.cvtColor(cv2.resize(norm8(crop), (PATCH, PATCH)), cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(st_crop, (PATCH//2, PATCH//2), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
        cv2.rectangle(st_crop, (0, 0), (PATCH-1, PATCH-1), (0, 80, 0), 1)

        # JY306 patch
        jy_z_c = np.clip(jy_z, 0, jy306_vol.shape[0] - 1)
        jy_slice = jy306_vol[jy_z_c]
        jx = int(round(pcd_iv[idx, 2]))
        jy_c = int(round(pcd_iv[idx, 1]))
        jx1, jy1 = max(0, jx - CROP_JY), max(0, jy_c - CROP_JY)
        jx2, jy2 = min(629, jx + CROP_JY), min(658, jy_c + CROP_JY)
        jcrop = jy_slice[jy1:jy2, jx1:jx2]
        if jcrop.size == 0:
            jcrop = np.zeros((10, 10), dtype=np.float32)
        jy_crop = cv2.cvtColor(cv2.resize(norm8(jcrop), (PATCH, PATCH)), cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(jy_crop, (PATCH//2, PATCH//2), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
        cv2.rectangle(jy_crop, (0, 0), (PATCH-1, PATCH-1), (0, 80, 0), 1)

        # Arrow
        arrow = np.zeros((PATCH, ARROW_W, 3), dtype=np.uint8)
        cv2.arrowedLine(arrow, (2, PATCH//2), (ARROW_W-2, PATCH//2), (0, 255, 0), 2, cv2.LINE_AA)

        pair_img = np.hstack([st_crop, arrow, jy_crop])

        label = np.zeros((LABEL_H, PAIR_W, 3), dtype=np.uint8)
        cv2.putText(label, f"#{idx+1} stitch z{z_st}", (2, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
        cv2.putText(label, f"jy z{jy_z_c}", (PATCH + ARROW_W + 2, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

        all_patches.append(np.vstack([pair_img, label]))

    # Grid
    n_rows_g = int(np.ceil(N_CELLS / MAX_COLS))
    grid_w = MAX_COLS * PAIR_W
    grid = np.zeros((n_rows_g * PAIR_H, grid_w, 3), dtype=np.uint8)
    for j, pp in enumerate(all_patches):
        r, c = j // MAX_COLS, j % MAX_COLS
        grid[r * PAIR_H:(r+1) * PAIR_H, c * PAIR_W:(c+1) * PAIR_W] = pp

    ph = np.zeros((30, grid_w, 3), dtype=np.uint8)
    cv2.putText(ph, f"Zoomed patches: single z-slice, IOU-only stitched 1um iso ({N_CELLS} cells)",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # Assemble
    final_w = max(full_w_view, grid_w)

    def pad_w(img, w):
        if img.shape[1] < w:
            return np.hstack([img, np.zeros((img.shape[0], w - img.shape[1], 3), dtype=np.uint8)])
        return img

    title = np.zeros((40, final_w, 3), dtype=np.uint8)
    cv2.putText(title, f"{tile} -- {N_CELLS} cells: IOU-only stitched ex-vivo (1um iso) vs in-vivo (JY306)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    sep = np.ones((6, final_w, 3), dtype=np.uint8) * 50
    sheet = np.vstack([title, pad_w(full_view, final_w), sep, pad_w(ph, final_w), pad_w(grid, final_w)])

    out_path = f'{OUT_DIR}/{tile}_contact_sheet.png'
    cv2.imwrite(out_path, sheet)
    print(f"  {N_CELLS} cells -> {out_path}")

    # Save stitched coords
    np.savez(f'{BASE}/registration_video/landmarks_stitched_iou_{tile}.npz',
             stitched_coords=stitched_coords,
             pcd_invivo_jy306=pcd_iv,
             ev_nd2=ev_nd2,
             cell_nd2_z=cell_nd2_z)

    del nd2_slices

print(f"\nDone! All contact sheets saved to {OUT_DIR}/")
