"""
Scene 5: All remaining tiles — cycle z-slices with matched landmarks, then warp.

For each tile:
  Phase 1 (3s): Cycle through z-slices of in-vivo and ex-vivo, drawing green
                 arrows for matched landmarks visible at each z-level.
  Phase 2 (2s): Warp in-vivo (mode z) onto ex-vivo.

17 tiles (row2_2 through row5_1, excluding row1_3 and row2_1).
~5s per tile + 2s final = ~87s total.

Output: animation/scene5_h264.mp4
"""

import numpy as np, cv2, tifffile, math, subprocess, os, glob
from collections import Counter, defaultdict

BASE = '/Users/neurolab/neuroinformatics/margaret'
TMP  = f'{BASE}/animation/scene5_raw.mp4'
OUT  = f'{BASE}/animation/scene5_h264.mp4'

W, H = 1920, 1080
FPS  = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREEN = (0, 220, 0)

ALL_TILES = ['row2_2', 'row2_3', 'row2_4', 'row2_5',
             'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
             'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
             'row5_1']

def ease(t):
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))

def norm_u8(img, lo=1, hi=99.5):
    v = img.ravel(); v = v[v > 0]
    if len(v) < 50: return np.zeros(img.shape, np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img.astype(np.float32) - p1) / max(p2 - p1, 1) * 255, 0, 255).astype(np.uint8)

def caption(frame, text, alpha=1.0):
    if alpha < 0.01: return
    ts, th2 = 0.72, 1
    (tw2, _), _ = cv2.getTextSize(text, FONT, ts, th2)
    x = (W - tw2) // 2; y = H - 42
    col = tuple(int(v * alpha) for v in WHITE)
    cv2.putText(frame, text, (x, y), FONT, ts, col, th2, cv2.LINE_AA)

def draw_arrow(frame, pt1, pt2, color, thickness=2, tip_length=0.025):
    cv2.arrowedLine(frame, pt1, pt2, color, thickness, cv2.LINE_AA, tipLength=tip_length)

# ── Load shared data ──
print("Loading JY306 z-stack...")
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz, hy, wx = jy306.shape

# Display layout
DISP_H = int(H * 0.72)
IMG_GAP = 100
scale_jy = DISP_H / hy
disp_jy_w = int(wx * scale_jy)
disp_jy_h = DISP_H

# ── Video writer ──
vw = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
total_frames = 0

# ── Process each tile ──
for ti, tile in enumerate(ALL_TILES):
    print(f"Tile {ti+1}/{len(ALL_TILES)}: {tile}...")

    # Load PKL transform
    pkl = np.load(f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz', allow_pickle=True)
    M2d = pkl['M2d_jy306_to_nd2']  # (x,y) convention
    M3 = np.vstack([M2d, [0, 0, 1]])
    iv = pkl['pcd_invivo_jy306']  # (z, y, x)
    ev = pkl['ev_nd2']            # (x, y, z)
    n_lm = len(iv)

    # Per-tile mode z
    z_counts = Counter(iv[:,0].astype(int))
    z_mode = z_counts.most_common(1)[0][0]
    z_mode = max(0, min(nz-1, z_mode))

    # Load nd2 stack
    nd2_files = sorted(glob.glob(f'{BASE}/png_exports/registration_video/{tile}/GFP_z*.png'))
    nd2_stack = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in nd2_files])

    # Crop nd2 to landmark region
    margin_nd2 = 350
    crop_x0 = max(0, int(ev[:,0].min() - margin_nd2))
    crop_y0 = max(0, int(ev[:,1].min() - margin_nd2))
    crop_x1 = min(4200, int(ev[:,0].max() + margin_nd2))
    crop_y1 = min(4200, int(ev[:,1].max() + margin_nd2))
    nd2_crop_shape = (crop_y1 - crop_y0, crop_x1 - crop_x0)
    scale_nd2 = DISP_H / nd2_crop_shape[0]
    disp_nd2_w = int(nd2_crop_shape[1] * scale_nd2)
    disp_nd2_h = DISP_H

    total_w = disp_jy_w + IMG_GAP + disp_nd2_w
    jy_x0 = (W - total_w) // 2
    jy_y0 = (H - DISP_H) // 2 - 20
    nd2_x0 = jy_x0 + disp_jy_w + IMG_GAP
    nd2_y0 = jy_y0

    # Group landmarks by in-vivo z
    lm_by_z = defaultdict(list)
    for i in range(n_lm):
        z_iv = int(round(iv[i, 0]))
        lm_by_z[z_iv].append(i)
    z_levels = sorted(lm_by_z.keys())
    n_z_levels = len(z_levels)

    # For each z-level, find best nd2 z
    best_nd2_z_per_level = {}
    for z_iv in z_levels:
        # Use landmarks at this z to find which nd2 z they map to
        nd2_zs = [int(round(ev[i, 2])) for i in lm_by_z[z_iv]]
        best_nd2_z_per_level[z_iv] = Counter(nd2_zs).most_common(1)[0][0]
        best_nd2_z_per_level[z_iv] = max(0, min(len(nd2_stack)-1, best_nd2_z_per_level[z_iv]))

    # ── Pick 3 best-contrast landmarks for zoom panels ──
    CROP_R_JY = 40; CROP_R_ND2 = 100; PANEL_SZ = 100
    jy_p1, jy_p2 = np.percentile(jy306[jy306 > 0], [1, 99.5])

    lm_scores = []
    for i in range(n_lm):
        z_lm = int(round(iv[i, 0])); y_lm = int(round(iv[i, 1])); x_lm = int(round(iv[i, 2]))
        r = CROP_R_JY
        y0c = max(0, y_lm-r); y1c = min(hy, y_lm+r)
        x0c = max(0, x_lm-r); x1c = min(wx, x_lm+r)
        crop = jy306[z_lm, y0c:y1c, x0c:x1c]
        lm_scores.append(float(crop.std()))
    selected_3 = sorted(range(n_lm), key=lambda i: -lm_scores[i])[:3]

    # Pre-build zoom panels for selected landmarks
    zoom_panels = {}  # idx -> (hot_panel, green_panel)
    for idx in selected_3:
        z_lm = int(round(iv[idx, 0])); y_lm = int(round(iv[idx, 1])); x_lm = int(round(iv[idx, 2]))
        # In-vivo MIP±2
        z_lo, z_hi = max(0, z_lm-2), min(nz, z_lm+3)
        mip_jy = np.max(jy306[z_lo:z_hi], axis=0)
        y0c = max(0, y_lm-CROP_R_JY); y1c = min(hy, y_lm+CROP_R_JY)
        x0c = max(0, x_lm-CROP_R_JY); x1c = min(wx, x_lm+CROP_R_JY)
        crop_jy = np.clip((mip_jy[y0c:y1c, x0c:x1c] - jy_p1) / max(jy_p2 - jy_p1, 1) * 255, 0, 255).astype(np.uint8)
        crop_jy = cv2.resize(crop_jy, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_LANCZOS4)
        hot_p = cv2.applyColorMap(crop_jy, cv2.COLORMAP_HOT); hot_p[crop_jy < 5] = 0

        # Ex-vivo patch
        x_nd2 = int(round(ev[idx, 0])); y_nd2 = int(round(ev[idx, 1]))
        z_nd2_lm = int(round(ev[idx, 2]))
        z_nd2_lm = max(0, min(len(nd2_stack)-1, z_nd2_lm))
        z_lo_n, z_hi_n = max(0, z_nd2_lm-2), min(len(nd2_stack), z_nd2_lm+3)
        mip_nd2 = np.max(nd2_stack[z_lo_n:z_hi_n], axis=0)
        yn0 = max(0, y_nd2-CROP_R_ND2); yn1 = min(4200, y_nd2+CROP_R_ND2)
        xn0 = max(0, x_nd2-CROP_R_ND2); xn1 = min(4200, x_nd2+CROP_R_ND2)
        crop_nd2 = norm_u8(mip_nd2[yn0:yn1, xn0:xn1].astype(np.uint8))
        crop_nd2 = cv2.resize(crop_nd2, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_LANCZOS4)
        green_p = np.zeros((PANEL_SZ, PANEL_SZ, 3), dtype=np.uint8); green_p[:,:,1] = crop_nd2

        zoom_panels[idx] = (hot_p, green_p)

    # Which z-level each selected landmark belongs to
    selected_z = {idx: int(round(iv[idx, 0])) for idx in selected_3}

    # ── Phase 1: Cycle through z-slices with green arrows (72 fr = 3s) ──
    frames_per_z = max(12, 72 // max(1, n_z_levels))  # at least 0.5s per z
    total_phase1 = frames_per_z * n_z_levels

    for fi in range(total_phase1):
        # Which z-level are we on?
        z_idx = min(fi // frames_per_z, n_z_levels - 1)
        z_iv = z_levels[z_idx]
        z_nd2 = best_nd2_z_per_level[z_iv]
        local_fi = fi - z_idx * frames_per_z

        # In-vivo display at this z
        z_u8 = norm_u8(jy306[z_iv])
        z_hot = cv2.applyColorMap(z_u8, cv2.COLORMAP_HOT)
        z_hot[z_u8 == 0] = 0
        jy_disp = cv2.resize(z_hot, (disp_jy_w, disp_jy_h), interpolation=cv2.INTER_LANCZOS4)

        # Ex-vivo display at corresponding nd2 z
        nd2_z_img = nd2_stack[z_nd2].astype(np.uint8)
        nd2_z_u8 = norm_u8(nd2_z_img)
        nd2_green = np.zeros((4200, 4200, 3), dtype=np.uint8)
        nd2_green[:,:,1] = nd2_z_u8
        nd2_crop = nd2_green[crop_y0:crop_y1, crop_x0:crop_x1]
        nd2_disp = cv2.resize(nd2_crop, (disp_nd2_w, disp_nd2_h), interpolation=cv2.INTER_LANCZOS4)

        frame = np.zeros((H, W, 3), np.uint8)
        frame[jy_y0:jy_y0+disp_jy_h, jy_x0:jy_x0+disp_jy_w] = jy_disp
        frame[nd2_y0:nd2_y0+disp_nd2_h, nd2_x0:nd2_x0+disp_nd2_w] = nd2_disp

        cv2.putText(frame, f'IN-VIVO  z = {z_iv}', (jy_x0+10, jy_y0-12),
                    FONT, 0.5, (100,180,255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'EX-VIVO  {tile}  z = {z_nd2}', (nd2_x0+10, nd2_y0-12),
                    FONT, 0.5, (100,255,100), 1, cv2.LINE_AA)

        # Draw green arrows for landmarks at this z-level
        lm_indices = lm_by_z[z_iv]
        # Animate arrows appearing
        arrows_per_step = max(1, len(lm_indices) // max(1, frames_per_z // 4))
        n_arrows_visible = min(len(lm_indices), 1 + int(local_fi * len(lm_indices) / max(1, frames_per_z - 4)))

        for ai, idx in enumerate(lm_indices[:n_arrows_visible]):
            # In-vivo position in display
            dx_jy = int(iv[idx, 2] * scale_jy) + jy_x0
            dy_jy = int(iv[idx, 1] * scale_jy) + jy_y0
            # Ex-vivo position in display
            dx_nd2 = int((ev[idx, 0] - crop_x0) * scale_nd2) + nd2_x0
            dy_nd2 = int((ev[idx, 1] - crop_y0) * scale_nd2) + nd2_y0

            cv2.circle(frame, (dx_jy, dy_jy), 6, GREEN, 2, cv2.LINE_AA)
            cv2.circle(frame, (dx_nd2, dy_nd2), 6, GREEN, 2, cv2.LINE_AA)
            draw_arrow(frame, (dx_jy, dy_jy), (dx_nd2, dy_nd2), GREEN, 1)

        # Also show previously seen z-levels' arrows faintly
        for prev_z_idx in range(z_idx):
            prev_z = z_levels[prev_z_idx]
            fade = 0.3
            fade_col = tuple(int(v * fade) for v in GREEN)
            for idx in lm_by_z[prev_z]:
                dx_jy = int(iv[idx, 2] * scale_jy) + jy_x0
                dy_jy = int(iv[idx, 1] * scale_jy) + jy_y0
                dx_nd2 = int((ev[idx, 0] - crop_x0) * scale_nd2) + nd2_x0
                dy_nd2 = int((ev[idx, 1] - crop_y0) * scale_nd2) + nd2_y0
                cv2.circle(frame, (dx_jy, dy_jy), 4, fade_col, 1, cv2.LINE_AA)
                cv2.circle(frame, (dx_nd2, dy_nd2), 4, fade_col, 1, cv2.LINE_AA)
                cv2.line(frame, (dx_jy, dy_jy), (dx_nd2, dy_nd2), fade_col, 1, cv2.LINE_AA)

        # Zoom panels for selected landmarks at current z-level
        active_panels = [idx for idx in selected_3 if selected_z[idx] == z_iv]
        if active_panels:
            pw = PANEL_SZ; gap = 30
            n_panels = len(active_panels)
            total_pw = n_panels * pw * 2 + n_panels * gap + (n_panels - 1) * 40
            px_start = (W - total_pw) // 2
            py_s = H - PANEL_SZ - 85
            p_alpha = ease(min(1.0, local_fi / 8.0))

            for pi, idx in enumerate(active_panels):
                hot_p, green_p = zoom_panels[idx]
                offset = pi * (pw * 2 + gap + 40)
                # Hot panel (in-vivo)
                hx = px_start + offset
                hp = (hot_p.astype(np.float32) * p_alpha).astype(np.uint8)
                frame[py_s:py_s+pw, hx:hx+pw] = hp
                cv2.rectangle(frame, (hx-2, py_s-2), (hx+pw+2, py_s+pw+2),
                              tuple(int(v*p_alpha) for v in (100,180,255)), 2)
                # Green panel (ex-vivo)
                gx = hx + pw + gap
                gp = (green_p.astype(np.float32) * p_alpha).astype(np.uint8)
                frame[py_s:py_s+pw, gx:gx+pw] = gp
                cv2.rectangle(frame, (gx-2, py_s-2), (gx+pw+2, py_s+pw+2),
                              tuple(int(v*p_alpha) for v in (100,255,100)), 2)
                # Lines from landmark to panels
                dx_jy = int(iv[idx, 2] * scale_jy) + jy_x0
                dy_jy = int(iv[idx, 1] * scale_jy) + jy_y0
                dx_nd2 = int((ev[idx, 0] - crop_x0) * scale_nd2) + nd2_x0
                dy_nd2 = int((ev[idx, 1] - crop_y0) * scale_nd2) + nd2_y0
                lcol = tuple(int(v*p_alpha*0.6) for v in WHITE)
                cv2.line(frame, (dx_jy, dy_jy), (hx+pw//2, py_s), lcol, 1, cv2.LINE_AA)
                cv2.line(frame, (dx_nd2, dy_nd2), (gx+pw//2, py_s), lcol, 1, cv2.LINE_AA)

        n_shown_total = sum(len(lm_by_z[z_levels[zi]]) for zi in range(z_idx)) + n_arrows_visible
        caption(frame, f'TILE  {tile.upper()}  --  MATCHED  LANDMARKS  ({n_shown_total}/{n_lm})')
        vw.write(frame)

    # ── Phase 2: Warp in-vivo (mode z) onto ex-vivo (48 fr = 2s) ──
    z_u8_mode = norm_u8(jy306[z_mode])
    z_hot_mode = cv2.applyColorMap(z_u8_mode, cv2.COLORMAP_HOT)
    z_hot_mode[z_u8_mode == 0] = 0
    jy_disp_mode = cv2.resize(z_hot_mode, (disp_jy_w, disp_jy_h), interpolation=cv2.INTER_LANCZOS4)

    # Best nd2 z for mode
    z_nd2_mode = best_nd2_z_per_level.get(z_mode, 0)
    nd2_mode_u8 = norm_u8(nd2_stack[z_nd2_mode].astype(np.uint8))
    nd2_green_mode = np.zeros((4200, 4200, 3), dtype=np.uint8)
    nd2_green_mode[:,:,1] = nd2_mode_u8
    nd2_crop_mode = nd2_green_mode[crop_y0:crop_y1, crop_x0:crop_x1]
    nd2_disp_mode = cv2.resize(nd2_crop_mode, (disp_nd2_w, disp_nd2_h), interpolation=cv2.INTER_LANCZOS4)

    M_start_3x3 = np.array([[scale_jy, 0, jy_x0], [0, scale_jy, jy_y0], [0, 0, 1]], dtype=np.float64)
    M_end_canvas = np.array([
        [scale_nd2, 0, nd2_x0 - crop_x0*scale_nd2],
        [0, scale_nd2, nd2_y0 - crop_y0*scale_nd2],
        [0, 0, 1]
    ], dtype=np.float64) @ M3

    nd2_bg = np.zeros((H, W, 3), np.uint8)
    nd2_bg[nd2_y0:nd2_y0+disp_nd2_h, nd2_x0:nd2_x0+disp_nd2_w] = nd2_disp_mode

    for fi in range(48):
        t = ease(fi / 40)
        frame = np.zeros((H, W, 3), np.uint8)

        M_t = M_start_3x3 * (1-t) + M_end_canvas * t
        warped_jy = cv2.warpAffine(z_u8_mode, M_t[:2].astype(np.float64), (W, H),
                                    flags=cv2.INTER_LANCZOS4, borderValue=0)
        frame[:,:,1] = nd2_bg[:,:,1]
        frame[:,:,2] = np.maximum(frame[:,:,2], warped_jy)

        a_old = 1 - ease(fi/15); a_new = ease((fi-15)/20)
        caption(frame, f'TILE  {tile.upper()}  --  {n_lm}  MATCHED  CELLS', alpha=max(0, a_old))
        caption(frame, f'WARPING  IN-VIVO  TO  {tile.upper()}', alpha=max(0, a_new))
        vw.write(frame)

    total_frames += total_phase1 + 48
    print(f"  {tile}: {n_lm} landmarks, {n_z_levels} z-levels ({z_levels}), z_mode={z_mode}")

# ── Final hold: all tiles done (48 fr = 2s) ──
print("Final hold...")
for fi in range(48):
    frame = np.zeros((H, W, 3), np.uint8)
    frame[:,:,1] = nd2_bg[:,:,1]
    warped_final = cv2.warpAffine(z_u8_mode, M_end_canvas[:2].astype(np.float64), (W, H),
                                   flags=cv2.INTER_LANCZOS4, borderValue=0)
    frame[:,:,2] = np.maximum(frame[:,:,2], warped_final)
    caption(frame, 'ALIGNED  TO  ALL  EX-VIVO  TILES')
    vw.write(frame)
total_frames += 48

# ── Finalize ──
vw.release()
print("Re-encoding to H.264...")
subprocess.run([
    'ffmpeg', '-y', '-i', TMP,
    '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
    OUT
], capture_output=True)
os.remove(TMP)
print(f"Done! {total_frames} frames, {total_frames/FPS:.1f}s @ {FPS}fps -- {OUT}")