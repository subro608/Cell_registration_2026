"""
Fix ECC affine registration for 4 failed tiles (row3_1, row3_4, row3_5, row3_6).

Strategy:
1. Phase correlation to get (dx, dy) translation, build initial affine, run ECC
2. If ECC diverges (unreasonable scale/rotation), fall back to SIFT feature matching
3. SIFT: detect features in downsampled nd2 and MERSCOPE MIPs, estimate affine via RANSAC,
   convert to full nd2->MERSCOPE space
4. Validate result: scale should be ~0.387, rotation < 15deg
5. Save affines and regenerate contact sheets
"""

import numpy as np
import cv2
import tifffile
import os

BASE = '/Users/neurolab/neuroinformatics/margaret'
REG_DIR = os.path.join(BASE, 'registration_video')
MERSCOPE_DIR = os.path.join(BASE, 'exvivo_merscope_combined')
PNG_DIR = os.path.join(BASE, 'png_exports/registration_video')
OUT_DIR = os.path.join(BASE, 'png_exports/coarse_registration/contact_sheet')
os.makedirs(OUT_DIR, exist_ok=True)

TILES = {
    'row3_1': '3_1_merscope16.tif',
    'row3_4': '3_4_merscope13.tif',
    'row3_5': '3_5_merscope12.tif',
    'row3_6': '3_6_merscope11.tif',
}

SCALE = 1627.0 / 4200.0  # ~0.3874
ND2_SIZE = 4200
MERSCOPE_SIZE = 1627
ECC_ITERS = 5000
ECC_EPS = 1e-6


def norm8(img, p_lo=1, p_hi=99.5):
    vals = img[img > 0]
    if len(vals) == 0:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    return np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


def load_nd2_mip(tile):
    d = os.path.join(PNG_DIR, tile)
    slices = []
    for i in range(12):
        p = os.path.join(d, f'GFP_z{i:03d}.png')
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Missing: {p}")
        slices.append(img.astype(np.float32))
    return np.max(np.stack(slices), axis=0)


def load_merscope_mip(tif_name):
    path = os.path.join(MERSCOPE_DIR, tif_name)
    vol = tifffile.imread(path)  # (3, 1627, 1627, 3) -- (z, H, W, channels)
    gfp = vol[:, :, :, 0].astype(np.float32)
    return np.max(gfp, axis=0)


def validate_affine(warp, tol_scale=0.15, tol_rot=15.0):
    """Check if affine has reasonable scale (~0.387) and rotation (<15deg)."""
    sx = np.sqrt(warp[0, 0]**2 + warp[1, 0]**2)
    sy = np.sqrt(warp[0, 1]**2 + warp[1, 1]**2)
    rot = abs(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
    scale_ok = abs(sx - SCALE) < tol_scale and abs(sy - SCALE) < tol_scale
    rot_ok = rot < tol_rot
    return scale_ok and rot_ok, sx, sy, rot


def try_ecc(nd2_f32, merscope_f32, init_warp):
    """Try ECC, return (warp, success). Success means valid affine."""
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERS, ECC_EPS)
    warp = init_warp.copy()
    try:
        cc, warp = cv2.findTransformECC(
            merscope_f32, nd2_f32, warp, cv2.MOTION_AFFINE, criteria,
            inputMask=None, gaussFiltSize=5
        )
        valid, sx, sy, rot = validate_affine(warp)
        print(f"    ECC: cc={cc:.4f}, sx={sx:.4f}, sy={sy:.4f}, rot={rot:.1f}deg, "
              f"tx={warp[0,2]:.1f}, ty={warp[1,2]:.1f} -- {'VALID' if valid else 'INVALID'}")
        return warp, valid
    except cv2.error as e:
        print(f"    ECC failed: {e}")
        return warp, False


def sift_affine(nd2_mip, merscope_mip):
    """Use SIFT feature matching to estimate affine (in downsampled space), convert to full."""
    nd2_ds = cv2.resize(nd2_mip, (MERSCOPE_SIZE, MERSCOPE_SIZE), interpolation=cv2.INTER_AREA)
    nd2_u8 = norm8(nd2_ds)
    merc_u8 = norm8(merscope_mip)

    sift = cv2.SIFT_create(5000)
    kp1, des1 = sift.detectAndCompute(nd2_u8, None)
    kp2, des2 = sift.detectAndCompute(merc_u8, None)

    if des1 is None or des2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    print(f"    SIFT: {len(kp1)} kp1, {len(kp2)} kp2, {len(good)} good matches")

    if len(good) < 10:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if M is None:
        return None

    inliers = mask.sum() if mask is not None else 0
    print(f"    SIFT affine: {inliers}/{len(good)} inliers")

    # Convert: ds-space affine to full nd2->MERSCOPE affine
    # In ds space: pixel_merc = M @ [pixel_nd2_ds; 1]
    # pixel_nd2_ds = SCALE * pixel_nd2_full
    # So: pixel_merc = [[M00*S, M01*S, M02], [M10*S, M11*S, M12]] @ [pixel_nd2_full; 1]
    full_M = M.astype(np.float32)
    full_M[:, :2] *= SCALE

    valid, sx, sy, rot = validate_affine(full_M)
    print(f"    SIFT full: sx={sx:.4f}, sy={sy:.4f}, rot={rot:.1f}deg, "
          f"tx={full_M[0,2]:.1f}, ty={full_M[1,2]:.1f} -- {'VALID' if valid else 'INVALID'}")
    return full_M


# ============================================================
# Part 1: Fix ECC affines
# ============================================================
print("=" * 70)
print("PART 1: Fix ECC affine registration for failed tiles")
print("=" * 70)

results = {}

for tile, tif_name in TILES.items():
    print(f"\n--- {tile} ({tif_name}) ---")

    nd2_mip = load_nd2_mip(tile)
    merscope_mip = load_merscope_mip(tif_name)
    print(f"  nd2 MIP: {nd2_mip.shape}, MERSCOPE MIP: {merscope_mip.shape}")

    nd2_f32 = norm8(nd2_mip).astype(np.float32) / 255.0
    merc_f32 = norm8(merscope_mip).astype(np.float32) / 255.0

    # Phase correlation for initial translation
    nd2_ds = cv2.resize(nd2_mip, (MERSCOPE_SIZE, MERSCOPE_SIZE), interpolation=cv2.INTER_AREA)
    nd2_n = norm8(nd2_ds).astype(np.float32)
    merc_n = norm8(merscope_mip).astype(np.float32)
    (dx_pc, dy_pc), resp = cv2.phaseCorrelate(nd2_n, merc_n)
    print(f"  Phase correlation: dx={dx_pc:.2f}, dy={dy_pc:.2f}, response={resp:.4f}")

    init_warp = np.array([[SCALE, 0, dx_pc], [0, SCALE, dy_pc]], dtype=np.float32)

    # Try ECC
    print("  Trying ECC with phase-correlation init...")
    warp, success = try_ecc(nd2_f32, merc_f32, init_warp)

    if not success:
        # ECC diverged -- use SIFT feature matching
        print("  ECC produced invalid result. Falling back to SIFT feature matching...")
        sift_warp = sift_affine(nd2_mip, merscope_mip)

        if sift_warp is not None:
            # Try ECC refinement from SIFT init
            print("  Trying ECC refinement from SIFT init...")
            warp2, success2 = try_ecc(nd2_f32, merc_f32, sift_warp)

            if success2:
                warp = warp2
                print("  Using ECC-refined SIFT affine.")
            else:
                # ECC diverged again -- use SIFT directly
                warp = sift_warp
                print("  ECC diverged again. Using SIFT affine directly.")
        else:
            print("  SIFT also failed. Keeping scale-only affine.")
            warp = init_warp

    # Save
    out_path = os.path.join(REG_DIR, f'affine_nd2_to_merscope_ecc_{tile}.npy')
    np.save(out_path, warp)
    valid, sx, sy, rot = validate_affine(warp)
    print(f"  SAVED: {out_path}")
    print(f"  Final: sx={sx:.4f}, sy={sy:.4f}, rot={rot:.1f}deg, tx={warp[0,2]:.1f}, ty={warp[1,2]:.1f}")
    results[tile] = warp


# ============================================================
# Part 2: Regenerate contact sheets
# ============================================================
print("\n" + "=" * 70)
print("PART 2: Regenerate contact sheets for fixed tiles")
print("=" * 70)

jy306_vol = tifffile.imread(os.path.join(BASE, 'JY306_in_Vivo_stack_flipped_s80.tif')).astype(np.float32)
jy_mip = np.max(jy306_vol, axis=0)
print(f"JY306 volume: {jy306_vol.shape}")

Z_RADIUS = 2
PATCH = 160
CROP_ND2 = 150
CROP_JY = 35
FULL_H = 800
MAX_COLS = 5

for tile in TILES:
    print(f"\n--- Contact sheet for {tile} ---")

    lm_path = os.path.join(REG_DIR, f'landmarks_nd2_native_{tile}.npz')
    data = np.load(lm_path)
    ev_nd2 = data['ev_nd2']
    pcd_iv = data['pcd_invivo_jy306']
    N_CELLS = len(ev_nd2)
    print(f"  {N_CELLS} matched cells")

    nd2_dir = os.path.join(PNG_DIR, tile)
    nd2_slices = np.stack([
        cv2.imread(os.path.join(nd2_dir, f'GFP_z{i:03d}.png'), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        for i in range(12)
    ])
    nd2_mip = np.max(nd2_slices, axis=0)

    cell_nd2_z = []
    for i in range(N_CELLS):
        c = int(round(np.clip(ev_nd2[i, 0], 10, 4189)))
        r = int(round(np.clip(ev_nd2[i, 1], 10, 4189)))
        intensities = [nd2_slices[z][r-10:r+10, c-10:c+10].mean() for z in range(12)]
        cell_nd2_z.append(np.argmax(intensities))
    cell_nd2_z = np.array(cell_nd2_z)
    jy_z_vals = pcd_iv[:, 0].astype(int)

    # Full view
    nd2_scale = FULL_H / 4200.0
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
        cv2.putText(full_view, str(idx+1), (lx-14, ly-7), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1)
        cv2.putText(full_view, str(idx+1), (rx+7, ry-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1)

    cv2.putText(full_view, f"{tile} ex-vivo MIP (nd2 native)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(full_view, "JY306 in-vivo MIP", (nd2_pw + ARROW_GAP + 8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # Zoomed patches
    ARROW_W = 20
    LABEL_H = 22
    PAIR_W = PATCH * 2 + ARROW_W
    PAIR_H = PATCH + LABEL_H

    all_patches = []
    for idx in range(N_CELLS):
        nd2_z = cell_nd2_z[idx]
        jy_z = jy_z_vals[idx]

        z_lo_nd2 = max(0, nd2_z - Z_RADIUS)
        z_hi_nd2 = min(nd2_slices.shape[0], nd2_z + Z_RADIUS + 1)
        nd2_local_mip = np.max(nd2_slices[z_lo_nd2:z_hi_nd2], axis=0)

        cx, cy = int(round(ev_nd2[idx, 0])), int(round(ev_nd2[idx, 1]))
        x1, y1 = max(0, cx - CROP_ND2), max(0, cy - CROP_ND2)
        x2, y2 = min(4200, cx + CROP_ND2), min(4200, cy + CROP_ND2)
        nd2_crop = cv2.cvtColor(cv2.resize(norm8(nd2_local_mip[y1:y2, x1:x2]), (PATCH, PATCH)), cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(nd2_crop, (PATCH//2, PATCH//2), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
        cv2.rectangle(nd2_crop, (0, 0), (PATCH-1, PATCH-1), (0, 80, 0), 1)

        z_lo_jy = max(0, jy_z - Z_RADIUS)
        z_hi_jy = min(jy306_vol.shape[0], jy_z + Z_RADIUS + 1)
        jy_local_mip = np.max(jy306_vol[z_lo_jy:z_hi_jy], axis=0)

        jx = int(round(pcd_iv[idx, 2]))
        jy_c = int(round(pcd_iv[idx, 1]))
        jx1, jy1 = max(0, jx - CROP_JY), max(0, jy_c - CROP_JY)
        jx2, jy2 = min(629, jx + CROP_JY), min(658, jy_c + CROP_JY)
        jy_crop = cv2.cvtColor(cv2.resize(norm8(jy_local_mip[jy1:jy2, jx1:jx2]), (PATCH, PATCH)), cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(jy_crop, (PATCH//2, PATCH//2), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
        cv2.rectangle(jy_crop, (0, 0), (PATCH-1, PATCH-1), (0, 80, 0), 1)

        arrow = np.zeros((PATCH, ARROW_W, 3), dtype=np.uint8)
        cv2.arrowedLine(arrow, (2, PATCH//2), (ARROW_W-2, PATCH//2), (0, 255, 0), 2, cv2.LINE_AA)

        pair_img = np.hstack([nd2_crop, arrow, jy_crop])

        label = np.zeros((LABEL_H, PAIR_W, 3), dtype=np.uint8)
        cv2.putText(label, f"#{idx+1} nd2 z{z_lo_nd2}-{z_hi_nd2-1}", (2, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
        cv2.putText(label, f"jy z{z_lo_jy}-{z_hi_jy-1}", (PATCH + ARROW_W + 2, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

        all_patches.append(np.vstack([pair_img, label]))

    n_rows = int(np.ceil(N_CELLS / MAX_COLS))
    grid_w = MAX_COLS * PAIR_W
    grid = np.zeros((n_rows * PAIR_H, grid_w, 3), dtype=np.uint8)
    for j, pp in enumerate(all_patches):
        r, c = j // MAX_COLS, j % MAX_COLS
        grid[r * PAIR_H:(r+1) * PAIR_H, c * PAIR_W:(c+1) * PAIR_W] = pp

    patch_header = np.zeros((30, grid_w, 3), dtype=np.uint8)
    cv2.putText(patch_header, f"Zoomed patches: local MIP +/-{Z_RADIUS} z-slices around each cell",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    full_w = full_view.shape[1]
    if grid_w < full_w:
        grid = np.hstack([grid, np.zeros((grid.shape[0], full_w - grid_w, 3), dtype=np.uint8)])
        patch_header = np.hstack([patch_header, np.zeros((30, full_w - grid_w, 3), dtype=np.uint8)])
    elif grid_w > full_w:
        full_view = np.hstack([full_view, np.zeros((FULL_H, grid_w - full_w, 3), dtype=np.uint8)])

    final_w = max(full_w, grid_w)
    title = np.zeros((40, final_w, 3), dtype=np.uint8)
    cv2.putText(title, f"{tile} -- {N_CELLS} matched cells: ex-vivo native (nd2) vs in-vivo native (JY306)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    sep = np.ones((6, final_w, 3), dtype=np.uint8) * 50
    sheet = np.vstack([title, full_view, sep, patch_header, grid])

    out_path = os.path.join(OUT_DIR, f'{tile}_contact_sheet.png')
    cv2.imwrite(out_path, sheet)
    print(f"  Saved: {out_path} ({sheet.shape[1]}x{sheet.shape[0]})")

print("\nDone!")
