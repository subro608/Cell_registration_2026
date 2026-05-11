"""
Regenerate calcium patch videos for the website from movie_rolling_avg .avi.

For each of 878 cells:
  1. Warp each calcium frame: movie → JY306 → nd2 space
  2. Crop 80x80 patch around cell location in nd2
  3. Export as H264 mp4 at 10fps

Output: invivo-exvivo-cell-registration/patches/patch_{idx}.mp4
"""

import numpy as np, cv2, os, glob, time

BASE = '/Users/neurolab/neuroinformatics/margaret'
AVI_PATH = f'{BASE}/movie_rolling_avg_win12_step3_short.avi'
PKL_BASE = f'{BASE}/png_exports/registration_per_tile_pkl'
OUT_DIR = f'{BASE}/invivo-exvivo-cell-registration/patches'
M_AVI_PATH = f'{BASE}/animation/movie_avi_to_jy306_affine.npz'

CROP = 80  # patch size in pixels
FPS_OUT = 10

TILES = ['row1_3',
         'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
         'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
         'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
         'row5_1']


def crop_centered(img, cx, cy, sz):
    h, w = img.shape[:2]
    x0 = max(0, cx - sz // 2)
    y0 = max(0, cy - sz // 2)
    x1 = min(w, x0 + sz)
    y1 = min(h, y0 + sz)
    crop = img[y0:y1, x0:x1]
    if crop.shape[0] != sz or crop.shape[1] != sz:
        padded = np.zeros((sz, sz), dtype=crop.dtype)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        return padded
    return crop


# ── Load calcium movie ──
print("Loading calcium movie...")
cap = cv2.VideoCapture(AVI_PATH)
cal_frames = []
while True:
    ret, fr = cap.read()
    if not ret:
        break
    cal_frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr)
cap.release()
cal_movie = np.array(cal_frames, dtype=np.uint8)
n_cal = len(cal_movie)
print(f"  {n_cal} calcium frames, shape: {cal_movie[0].shape}")

# ── Load movie→JY306 affine ──
M_movie_to_jy306 = np.load(M_AVI_PATH)['M_affine']
M_m2j_3x3 = np.vstack([M_movie_to_jy306, [0, 0, 1]])

# ── Process each tile ──
os.makedirs(OUT_DIR, exist_ok=True)

t0 = time.time()
patch_idx = 0
total_cells = 878

for tile in TILES:
    npz_path = f'{PKL_BASE}/{tile}/pkl_transform_{tile}.npz'
    npz = np.load(npz_path, allow_pickle=True)
    ev_nd2 = npz['ev_nd2']
    M2d = npz['M2d_jy306_to_nd2']
    n_cells = ev_nd2.shape[0]

    # Compute composite transform: movie → JY306 → nd2
    M_j2n_3x3 = np.vstack([M2d, [0, 0, 1]])
    M_movie_to_nd2 = (M_j2n_3x3 @ M_m2j_3x3)[:2, :]

    # Pre-warp all calcium frames for this tile (same transform for all cells in tile)
    print(f"\n{tile}: warping {n_cal} frames for {n_cells} cells...")
    warped_frames = []
    for fi in range(n_cal):
        warped = cv2.warpAffine(cal_movie[fi], M_movie_to_nd2, (4200, 4200), borderValue=0)
        warped_frames.append(warped)

    # Generate patch video for each cell
    for ci in range(n_cells):
        x_nd2 = int(round(ev_nd2[ci, 0]))
        y_nd2 = int(round(ev_nd2[ci, 1]))

        out_path = f'{OUT_DIR}/patch_{patch_idx}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(out_path, fourcc, FPS_OUT, (CROP, CROP), isColor=True)

        for fi in range(n_cal):
            patch = crop_centered(warped_frames[fi], x_nd2, y_nd2, CROP)
            patch_bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
            writer.write(patch_bgr)

        writer.release()
        patch_idx += 1

        if patch_idx % 50 == 0:
            elapsed = time.time() - t0
            rate = patch_idx / elapsed
            remaining = (total_cells - patch_idx) / rate
            print(f"  {patch_idx}/{total_cells} patches ({rate:.1f}/s, ~{remaining:.0f}s remaining)")

    del warped_frames

elapsed = time.time() - t0
print(f"\nDone! {patch_idx} patches in {elapsed:.0f}s")
print(f"Output: {OUT_DIR}/")
