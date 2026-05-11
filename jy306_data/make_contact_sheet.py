"""
Contact sheet: ex-vivo native (nd2) vs in-vivo native (JY306) matched cells.

For ALL tiles that have pkl transforms in the transformation/ folder (19 tiles).

Pipeline per tile:
  1. SIFT affine: nd2(4200) <-> MERSCOPE(1627) — pure scale ~0.3873
  2. Iterative pkl inverse: JY306 -> MERSCOPE (exact, 0px round-trip)
  3. SIFT inverse: MERSCOPE -> nd2 (just x2.58 scale)

Output per tile:
  - Top: full MIP view with green arrows connecting matched cell pairs
  - Bottom: zoomed patches using local MIP +/-2 z-slices around each cell
  - Saved to png_exports/coarse_registration/contact_sheet/{tile}_contact_sheet.png
  - Landmarks saved to registration_video/landmarks_nd2_native_{tile}.npz

Usage:
    python make_contact_sheet.py

Requires:
    - transformation/*.pkl (19 pkl files with pcd_invivo, pcd_exvivo, transformations)
    - png_exports/registration_video/{tile}/GFP_z*.png (nd2 z-slices per tile)
    - exvivo_merscope_combined/{tile}_merscope*.tif (MERSCOPE reference images)
    - JY306_in_Vivo_stack_flipped_s80.tif (in-vivo volume)
"""

import numpy as np
import cv2
import tifffile
import pickle
import os
from scipy.ndimage import map_coordinates

BASE = '/Users/neurolab/neuroinformatics/margaret'
REG_DIR = os.path.join(BASE, 'registration_video')
PNG_DIR = os.path.join(BASE, 'png_exports/registration_video')
OUT_DIR = os.path.join(BASE, 'png_exports/coarse_registration/contact_sheet')
os.makedirs(OUT_DIR, exist_ok=True)

# Config
Z_RADIUS = 2        # local MIP +/- z-slices for zoomed patches
PATCH = 160          # zoomed patch display size
CROP_ND2 = 150       # crop radius in nd2 pixels (4200 space)
CROP_JY = 35         # crop radius in JY306 pixels (658 space)
FULL_H = 800         # full view panel height
MAX_COLS = 5         # patches per row

# ============================================================
# Helper functions
# ============================================================

def norm8(img, p_lo=1, p_hi=99.5):
    """Percentile-based normalization to uint8."""
    vals = img[img > 0]
    if len(vals) == 0:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    return np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


def interp_vecfield(vf, pts):
    """Interpolate vector field at fractional points. pts: (N,3), vf: (Z,Y,X,3)."""
    out = np.zeros_like(pts)
    for c in range(3):
        out[:, c] = map_coordinates(vf[..., c], pts.T, order=1, mode='nearest')
    return out


def point_inverse_iterative(pts, transforms, max_iter=20, tol=1e-6):
    """
    Inverse pkl transform: JY306 -> MERSCOPE space.
    Applies stages in reverse order with iterative fixed-point for vec_fields.
    """
    p = pts.copy().astype(np.float64)
    for t in reversed(transforms):
        key = list(t.keys())[0]
        val = t[key]
        if key == 'scale':
            p = p / val
        elif key == 'bhat':
            R = val[:3].astype(np.float64)
            tv = val[3].astype(np.float64)
            p = (p - tv) @ np.linalg.inv(R)
        elif key == 'vec_field_total':
            p_out = p.copy()
            p_in = p_out.copy()
            for _ in range(max_iter):
                disp = interp_vecfield(val, p_in)
                p_new = p_out - disp
                if np.max(np.abs(p_new - p_in)) < tol:
                    break
                p_in = p_new
            p = p_in
    return p


def compute_sift_affine(nd2_mip, merc_mip):
    """
    Compute SIFT-based affine from nd2(4200) to MERSCOPE(1627).
    Returns 2x3 affine matrix mapping nd2(4200) coords to MERSCOPE(1627) coords.
    """
    nd2_ds = cv2.resize(nd2_mip, (1627, 1627), interpolation=cv2.INTER_AREA)
    nd2_u8 = norm8(nd2_ds)
    merc_u8 = norm8(merc_mip)

    sift = cv2.SIFT_create(nfeatures=5000)
    kp1, des1 = sift.detectAndCompute(nd2_u8, None)
    kp2, des2 = sift.detectAndCompute(merc_u8, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, 0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good) < 10:
        return None, 0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    if M is None:
        return None, 0

    inliers = int(mask.sum())
    # Convert from 1627-space to 4200-space
    scale_factor = 1627.0 / 4200.0
    M_4200 = M.copy()
    M_4200[:, :2] *= scale_factor
    return M_4200, inliers


# ============================================================
# Discover all tiles from transformation/ pkl files
# ============================================================
pkl_dir = os.path.join(BASE, 'transformation')
pkl_files = sorted([f for f in os.listdir(pkl_dir) if f.endswith('.pkl')])

TILES = {}
for pf in pkl_files:
    parts = pf.split('_merscope')
    tile_num = parts[0]
    merc_num = parts[1].split('transformed')[0]
    tile_key = f"row{tile_num}"
    merc_tif = f"{tile_num}_merscope{merc_num}.tif"
    TILES[tile_key] = (pf, merc_tif)

print(f"Found {len(TILES)} tiles with pkl transforms")

# ============================================================
# Load JY306 in-vivo volume (shared across all tiles)
# ============================================================
jy306_vol = tifffile.imread(os.path.join(BASE, 'JY306_in_Vivo_stack_flipped_s80.tif')).astype(np.float32)
jy_mip = np.max(jy306_vol, axis=0)
print(f"JY306 volume: {jy306_vol.shape}")

# ============================================================
# Process each tile
# ============================================================
for tile, (pkl_file, merc_file) in sorted(TILES.items()):
    print(f"\n{'='*50}")
    print(f"  {tile}")
    print(f"{'='*50}")

    # --- Load pkl ---
    with open(os.path.join(pkl_dir, pkl_file), 'rb') as f:
        d = pickle.load(f)
    transforms = d['transformations']
    pcd_iv = d['pcd_invivo']    # (N, 3) z, y, x in JY306
    pcd_ev = d['pcd_exvivo']    # (N, 3) z, y, x in JY306
    N_CELLS = len(pcd_iv)

    if N_CELLS == 0:
        print(f"  No cells, skipping")
        continue

    # --- SIFT affine (compute or load) ---
    if tile == 'row2_1':
        aff_path = os.path.join(REG_DIR, 'affine_nd2_to_merscope_ecc.npy')
    else:
        aff_path = os.path.join(REG_DIR, f'affine_nd2_to_merscope_ecc_{tile}.npy')

    if os.path.exists(aff_path):
        aff = np.load(aff_path)
        sx = np.sqrt(aff[0, 0]**2 + aff[1, 0]**2)
        if abs(sx - 0.3873) < 0.01:
            print(f"  SIFT affine loaded (sx={sx:.4f})")
        else:
            aff = None
    else:
        aff = None

    if aff is None:
        print(f"  Computing SIFT affine...")
        nd2_dir = os.path.join(PNG_DIR, tile)
        nd2_slices_for_sift = []
        for i in range(12):
            p = os.path.join(nd2_dir, f'GFP_z{i:03d}.png')
            if os.path.exists(p):
                nd2_slices_for_sift.append(cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32))
        nd2_mip_sift = np.max(np.stack(nd2_slices_for_sift), axis=0)

        merc_path = os.path.join(BASE, 'exvivo_merscope_combined', merc_file)
        merc_tif = tifffile.imread(merc_path)
        if merc_tif.ndim == 4:
            merc_mip_sift = np.max(merc_tif[:, :, :, 0], axis=0).astype(np.float32)
        else:
            merc_mip_sift = np.max(merc_tif, axis=0).astype(np.float32)

        aff, inliers = compute_sift_affine(nd2_mip_sift, merc_mip_sift)
        if aff is None:
            print(f"  SIFT FAILED, skipping tile")
            continue
        np.save(aff_path, aff)
        sx = np.sqrt(aff[0, 0]**2 + aff[1, 0]**2)
        print(f"  SIFT affine computed: {inliers} inliers, sx={sx:.4f}")

    M3_inv = np.linalg.inv(np.vstack([aff, [0, 0, 1]]))[:2, :]

    # --- Iterative pkl inverse: JY306 -> MERSCOPE -> nd2 ---
    merc_ev = point_inverse_iterative(pcd_ev, transforms)
    merc_iv = point_inverse_iterative(pcd_iv, transforms)

    ev_nd2 = np.zeros((N_CELLS, 3))
    iv_nd2 = np.zeros((N_CELLS, 3))
    for i in range(N_CELLS):
        ev_nd2[i] = [*(M3_inv @ [merc_ev[i, 2], merc_ev[i, 1], 1]), merc_ev[i, 0]]
        iv_nd2[i] = [*(M3_inv @ [merc_iv[i, 2], merc_iv[i, 1], 1]), merc_iv[i, 0]]

    # --- Save landmarks ---
    if tile == 'row2_1':
        lm_path = os.path.join(REG_DIR, 'landmarks_27_nd2_native.npz')
    else:
        lm_path = os.path.join(REG_DIR, f'landmarks_nd2_native_{tile}.npz')
    np.savez(lm_path,
             ev_nd2=ev_nd2, iv_nd2=iv_nd2,
             pcd_invivo_jy306=pcd_iv, pcd_exvivo_jy306=pcd_ev)

    # --- Load nd2 z-slices ---
    nd2_dir = os.path.join(PNG_DIR, tile)
    nd2_slices = np.stack([
        cv2.imread(os.path.join(nd2_dir, f'GFP_z{i:03d}.png'), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        for i in range(12)
    ])
    nd2_mip = np.max(nd2_slices, axis=0)
    H_nd2 = nd2_mip.shape[0]

    # --- Find best nd2 z per cell ---
    cell_nd2_z = []
    for i in range(N_CELLS):
        c = int(round(np.clip(ev_nd2[i, 0], 10, H_nd2 - 11)))
        r = int(round(np.clip(ev_nd2[i, 1], 10, H_nd2 - 11)))
        intensities = [nd2_slices[z][r-10:r+10, c-10:c+10].mean() for z in range(12)]
        cell_nd2_z.append(np.argmax(intensities))
    cell_nd2_z = np.array(cell_nd2_z)
    jy_z_vals = pcd_iv[:, 0].astype(int)

    # ============================================================
    # Part 1: Full view -- ex-vivo MIP + in-vivo MIP + green arrows
    # ============================================================
    nd2_scale = FULL_H / float(H_nd2)
    nd2_pw = FULL_H
    jy_scale = FULL_H / 658.0
    jy_pw = int(629 * jy_scale)
    ARROW_GAP = 100

    nd2_img = cv2.cvtColor(cv2.resize(norm8(nd2_mip), (nd2_pw, FULL_H)), cv2.COLOR_GRAY2BGR)
    jy_img = cv2.cvtColor(cv2.resize(norm8(jy_mip), (jy_pw, FULL_H)), cv2.COLOR_GRAY2BGR)
    full_view = np.hstack([nd2_img, np.zeros((FULL_H, ARROW_GAP, 3), dtype=np.uint8), jy_img])

    for idx in range(N_CELLS):
        lx = int(ev_nd2[idx, 0] * nd2_scale)
        ly = int(ev_nd2[idx, 1] * nd2_scale)
        rx = int(pcd_iv[idx, 2] * jy_scale) + nd2_pw + ARROW_GAP
        ry = int(pcd_iv[idx, 1] * jy_scale)
        cv2.arrowedLine(full_view, (lx, ly), (rx, ry), (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.015)
        cv2.circle(full_view, (lx, ly), 5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(full_view, (rx, ry), 5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(full_view, str(idx+1), (lx-14, ly-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1)
        cv2.putText(full_view, str(idx+1), (rx+7, ry-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1)

    cv2.putText(full_view, f"{tile} ex-vivo MIP (nd2 native)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(full_view, "JY306 in-vivo MIP", (nd2_pw + ARROW_GAP + 8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # ============================================================
    # Part 2: Zoomed patches — generate both single-z and MIP±2
    # ============================================================
    ARROW_W = 20
    LABEL_H = 22
    PAIR_W = PATCH * 2 + ARROW_W
    PAIR_H = PATCH + LABEL_H

    modes = [
        {'name': 'single_z', 'z_radius': 0, 'label': 'single z-slice', 'suffix': '_single_z'},
        {'name': 'mip_pm2',  'z_radius': 2, 'label': 'MIP +/-2 z-slices', 'suffix': '_mip_pm2'},
    ]

    for mode in modes:
        zr = mode['z_radius']
        out_dir_mode = os.path.join(OUT_DIR, mode['name'])
        os.makedirs(out_dir_mode, exist_ok=True)

        all_patches = []
        for idx in range(N_CELLS):
            nd2_z = cell_nd2_z[idx]
            jy_z = jy_z_vals[idx]

            # nd2 patch
            z_lo_nd2 = max(0, nd2_z - zr)
            z_hi_nd2 = min(12, nd2_z + zr + 1)
            if zr == 0:
                nd2_img = nd2_slices[np.clip(nd2_z, 0, 11)]
            else:
                nd2_img = np.max(nd2_slices[z_lo_nd2:z_hi_nd2], axis=0)

            cx = int(round(ev_nd2[idx, 0]))
            cy = int(round(ev_nd2[idx, 1]))
            x1, y1 = max(0, cx - CROP_ND2), max(0, cy - CROP_ND2)
            x2, y2 = min(H_nd2, cx + CROP_ND2), min(H_nd2, cy + CROP_ND2)
            crop = nd2_img[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((10, 10), dtype=np.float32)
            nd2_crop = cv2.cvtColor(cv2.resize(norm8(crop), (PATCH, PATCH)), cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(nd2_crop, (PATCH//2, PATCH//2), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
            cv2.rectangle(nd2_crop, (0, 0), (PATCH-1, PATCH-1), (0, 80, 0), 1)

            # JY306 patch
            z_lo_jy = max(0, jy_z - zr)
            z_hi_jy = min(jy306_vol.shape[0], jy_z + zr + 1)
            if zr == 0:
                jy_img = jy306_vol[np.clip(jy_z, 0, jy306_vol.shape[0]-1)]
            else:
                jy_img = np.max(jy306_vol[z_lo_jy:z_hi_jy], axis=0)

            jx = int(round(pcd_iv[idx, 2]))
            jy_c = int(round(pcd_iv[idx, 1]))
            jx1, jy1 = max(0, jx - CROP_JY), max(0, jy_c - CROP_JY)
            jx2, jy2 = min(629, jx + CROP_JY), min(658, jy_c + CROP_JY)
            jcrop = jy_img[jy1:jy2, jx1:jx2]
            if jcrop.size == 0:
                jcrop = np.zeros((10, 10), dtype=np.float32)
            jy_crop = cv2.cvtColor(cv2.resize(norm8(jcrop), (PATCH, PATCH)), cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(jy_crop, (PATCH//2, PATCH//2), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
            cv2.rectangle(jy_crop, (0, 0), (PATCH-1, PATCH-1), (0, 80, 0), 1)

            # Arrow between patches
            arrow = np.zeros((PATCH, ARROW_W, 3), dtype=np.uint8)
            cv2.arrowedLine(arrow, (2, PATCH//2), (ARROW_W-2, PATCH//2), (0, 255, 0), 2, cv2.LINE_AA)

            pair_img = np.hstack([nd2_crop, arrow, jy_crop])

            # Label
            label = np.zeros((LABEL_H, PAIR_W, 3), dtype=np.uint8)
            if zr == 0:
                nd2_zlabel = f"z{np.clip(nd2_z, 0, 11)}"
                jy_zlabel = f"z{np.clip(jy_z, 0, jy306_vol.shape[0]-1)}"
            else:
                nd2_zlabel = f"z{z_lo_nd2}-{z_hi_nd2-1}"
                jy_zlabel = f"z{z_lo_jy}-{z_hi_jy-1}"
            cv2.putText(label, f"#{idx+1} nd2 {nd2_zlabel}", (2, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
            cv2.putText(label, f"jy {jy_zlabel}", (PATCH + ARROW_W + 2, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

            all_patches.append(np.vstack([pair_img, label]))

        # Pack into tight grid
        n_rows = int(np.ceil(N_CELLS / MAX_COLS))
        grid_w = MAX_COLS * PAIR_W
        grid = np.zeros((n_rows * PAIR_H, grid_w, 3), dtype=np.uint8)
        for j, pp in enumerate(all_patches):
            r, c = j // MAX_COLS, j % MAX_COLS
            grid[r * PAIR_H:(r+1) * PAIR_H, c * PAIR_W:(c+1) * PAIR_W] = pp

        # Patch section header
        patch_header = np.zeros((30, grid_w, 3), dtype=np.uint8)
        cv2.putText(patch_header,
                    f"Zoomed patches: {mode['label']} ({N_CELLS} cells)",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        # ============================================================
        # Combine: title + full view + patches
        # ============================================================
        fv = full_view.copy()
        full_w = fv.shape[1]
        g = grid.copy()
        ph = patch_header.copy()
        if grid_w < full_w:
            g = np.hstack([g, np.zeros((g.shape[0], full_w - grid_w, 3), dtype=np.uint8)])
            ph = np.hstack([ph, np.zeros((30, full_w - grid_w, 3), dtype=np.uint8)])
        elif grid_w > full_w:
            fv = np.hstack([fv, np.zeros((FULL_H, grid_w - full_w, 3), dtype=np.uint8)])

        final_w = max(full_w, grid_w)
        title = np.zeros((40, final_w, 3), dtype=np.uint8)
        cv2.putText(title,
                    f"{tile} -- {N_CELLS} cells [{mode['label']}]: ex-vivo (nd2) vs in-vivo (JY306)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        sep = np.ones((6, final_w, 3), dtype=np.uint8) * 50
        sheet = np.vstack([title, fv, sep, ph, g])

        out_path = os.path.join(out_dir_mode, f'{tile}_contact_sheet.png')
        cv2.imwrite(out_path, sheet)
        print(f"  [{mode['name']}] {N_CELLS} cells -> {out_path}")

    # ============================================================
    # Part 3: Combined sheet — smaller overview + both patch grids
    # ============================================================
    COMBINED_DIR = os.path.join(OUT_DIR, 'combined')
    os.makedirs(COMBINED_DIR, exist_ok=True)

    SMALL_H = 400  # reduced overview height
    grids_by_mode = {}

    for mode in modes:
        zr = mode['z_radius']
        all_patches = []
        for idx in range(N_CELLS):
            nd2_z = cell_nd2_z[idx]
            jy_z = jy_z_vals[idx]

            z_lo_nd2 = max(0, nd2_z - zr)
            z_hi_nd2 = min(12, nd2_z + zr + 1)
            if zr == 0:
                nd2_img = nd2_slices[np.clip(nd2_z, 0, 11)]
            else:
                nd2_img = np.max(nd2_slices[z_lo_nd2:z_hi_nd2], axis=0)

            cx = int(round(ev_nd2[idx, 0]))
            cy = int(round(ev_nd2[idx, 1]))
            x1, y1 = max(0, cx - CROP_ND2), max(0, cy - CROP_ND2)
            x2, y2 = min(H_nd2, cx + CROP_ND2), min(H_nd2, cy + CROP_ND2)
            crop = nd2_img[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((10, 10), dtype=np.float32)
            nd2_crop = cv2.cvtColor(cv2.resize(norm8(crop), (PATCH, PATCH)), cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(nd2_crop, (PATCH//2, PATCH//2), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
            cv2.rectangle(nd2_crop, (0, 0), (PATCH-1, PATCH-1), (0, 80, 0), 1)

            z_lo_jy = max(0, jy_z - zr)
            z_hi_jy = min(jy306_vol.shape[0], jy_z + zr + 1)
            if zr == 0:
                jy_img = jy306_vol[np.clip(jy_z, 0, jy306_vol.shape[0]-1)]
            else:
                jy_img = np.max(jy306_vol[z_lo_jy:z_hi_jy], axis=0)

            jx = int(round(pcd_iv[idx, 2]))
            jy_c = int(round(pcd_iv[idx, 1]))
            jx1, jy1 = max(0, jx - CROP_JY), max(0, jy_c - CROP_JY)
            jx2, jy2 = min(629, jx + CROP_JY), min(658, jy_c + CROP_JY)
            jcrop = jy_img[jy1:jy2, jx1:jx2]
            if jcrop.size == 0:
                jcrop = np.zeros((10, 10), dtype=np.float32)
            jy_crop = cv2.cvtColor(cv2.resize(norm8(jcrop), (PATCH, PATCH)), cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(jy_crop, (PATCH//2, PATCH//2), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
            cv2.rectangle(jy_crop, (0, 0), (PATCH-1, PATCH-1), (0, 80, 0), 1)

            arrow = np.zeros((PATCH, ARROW_W, 3), dtype=np.uint8)
            cv2.arrowedLine(arrow, (2, PATCH//2), (ARROW_W-2, PATCH//2), (0, 255, 0), 2, cv2.LINE_AA)
            pair_img = np.hstack([nd2_crop, arrow, jy_crop])

            label = np.zeros((LABEL_H, PAIR_W, 3), dtype=np.uint8)
            if zr == 0:
                nd2_zlabel = f"z{np.clip(nd2_z, 0, 11)}"
                jy_zlabel = f"z{np.clip(jy_z, 0, jy306_vol.shape[0]-1)}"
            else:
                nd2_zlabel = f"z{z_lo_nd2}-{z_hi_nd2-1}"
                jy_zlabel = f"z{z_lo_jy}-{z_hi_jy-1}"
            cv2.putText(label, f"#{idx+1} nd2 {nd2_zlabel}", (2, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
            cv2.putText(label, f"jy {jy_zlabel}", (PATCH + ARROW_W + 2, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

            all_patches.append(np.vstack([pair_img, label]))

        n_rows_g = int(np.ceil(N_CELLS / MAX_COLS))
        grid_w = MAX_COLS * PAIR_W
        grid = np.zeros((n_rows_g * PAIR_H, grid_w, 3), dtype=np.uint8)
        for j, pp in enumerate(all_patches):
            r, c = j // MAX_COLS, j % MAX_COLS
            grid[r * PAIR_H:(r+1) * PAIR_H, c * PAIR_W:(c+1) * PAIR_W] = pp

        ph = np.zeros((30, grid_w, 3), dtype=np.uint8)
        cv2.putText(ph, f"Zoomed patches: {mode['label']} ({N_CELLS} cells)",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        grids_by_mode[mode['name']] = np.vstack([ph, grid])

    # Small overview
    small_scale = SMALL_H / float(H_nd2)
    small_pw = SMALL_H
    nd2_small = cv2.resize(norm8(nd2_mip), (small_pw, SMALL_H))
    jy_mip = np.max(jy306_vol, axis=0).astype(np.float32)
    jy_small = cv2.resize(norm8(jy_mip), (int(629 * SMALL_H / 658), SMALL_H))
    small_arrow_gap = 30
    small_fv = np.zeros((SMALL_H, small_pw + small_arrow_gap + jy_small.shape[1], 3), dtype=np.uint8)
    small_fv[:, :small_pw] = cv2.cvtColor(nd2_small, cv2.COLOR_GRAY2BGR)
    small_fv[:, small_pw + small_arrow_gap:] = cv2.cvtColor(jy_small, cv2.COLOR_GRAY2BGR)

    # Draw arrows on small overview
    for idx in range(N_CELLS):
        lx = int(round(ev_nd2[idx, 0] * small_scale))
        ly = int(round(ev_nd2[idx, 1] * small_scale))
        rx = int(round(pcd_iv[idx, 2] * SMALL_H / 658)) + small_pw + small_arrow_gap
        ry = int(round(pcd_iv[idx, 1] * SMALL_H / 658))
        cv2.arrowedLine(small_fv, (lx, ly), (rx, ry), (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.05)
        cv2.circle(small_fv, (lx, ly), 3, (0, 255, 0), 1)
        cv2.circle(small_fv, (rx, ry), 3, (0, 255, 0), 1)

    # Assemble combined
    grid_w = MAX_COLS * PAIR_W
    final_w = max(small_fv.shape[1], grid_w)

    title = np.zeros((40, final_w, 3), dtype=np.uint8)
    cv2.putText(title, f"{tile} -- {N_CELLS} cells: ex-vivo (nd2) vs in-vivo (JY306)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def pad_w(img, w):
        if img.shape[1] < w:
            return np.hstack([img, np.zeros((img.shape[0], w - img.shape[1], 3), dtype=np.uint8)])
        return img

    sep = np.ones((6, final_w, 3), dtype=np.uint8) * 50
    parts = [title, pad_w(small_fv, final_w), sep]
    for mname in ['single_z', 'mip_pm2']:
        parts.append(pad_w(grids_by_mode[mname], final_w))
        parts.append(sep.copy())

    combined = np.vstack(parts)
    out_path = os.path.join(COMBINED_DIR, f'{tile}_contact_sheet.png')
    cv2.imwrite(out_path, combined)
    print(f"  [combined] {N_CELLS} cells -> {out_path}")

print(f"\nDone! All {len(TILES)} tiles saved to:")
print(f"  {OUT_DIR}/single_z/")
print(f"  {OUT_DIR}/mip_pm2/")
print(f"  {OUT_DIR}/combined/")
