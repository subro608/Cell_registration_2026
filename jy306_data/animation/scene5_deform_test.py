"""
Scene 5 deformation test — 1 tile (row2_1).

1. Side-by-side: invivo (hot) left, exvivo (green) right — same proportions as scene 4
2. Show matched landmarks with green arrows + zoom panels
3. Invivo moves on top of exvivo (centroid-based alignment)
4. Apply M2d affine warp — show cells registering
5. Hold final overlay

Uses only what's in pkl_transform_row2_1.npz: M2d + landmarks. No RBF/TPS.
"""

import numpy as np, cv2, tifffile, math, subprocess, os, glob
from collections import Counter

BASE = '/Users/neurolab/neuroinformatics/margaret'
TILE = 'row2_1'
TMP  = f'{BASE}/animation/scene5_deform_test_raw.mp4'
OUT  = f'{BASE}/animation/scene5_deform_test_h264.mp4'

W, H = 1920, 1080
FPS  = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREEN = (0, 220, 0)

# 3D rotation params (same as scene4b)
Z_SPACING = 4
INTERP_PER_GAP = 3
INIT_ROT_X = -0.3

def ease(t):
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))

def norm8(img, lo=1, hi=99.5):
    v = img.ravel(); v = v[v > 0]
    if len(v) < 50: return np.zeros(img.shape, np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img - p1) / max(p2 - p1, 1) * 255, 0, 255).astype(np.uint8)

def make_hot(u8):
    hot = cv2.applyColorMap(u8, cv2.COLORMAP_HOT)
    hot[u8 == 0] = 0
    return hot

def caption(frame, text, alpha=1.0):
    if alpha < 0.01: return
    ts, th = 0.72, 1
    (tw, _), _ = cv2.getTextSize(text, FONT, ts, th)
    x = (W - tw) // 2; y = H - 42
    col = tuple(int(v * alpha) for v in WHITE)
    cv2.putText(frame, text, (x, y), FONT, ts, col, th, cv2.LINE_AA)

def draw_arrow(frame, pt1, pt2, color, thickness=2, tip=0.025):
    cv2.arrowedLine(frame, pt1, pt2, color, thickness, cv2.LINE_AA, tipLength=tip)

# ── Load data ──
print("Loading data...")
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_jy, hy_jy, wx_jy = jy306.shape
MODE_Z = 3

nd2_files = sorted(glob.glob(f'{BASE}/png_exports/registration_video/{TILE}/GFP_z*.png'))
nd2_stack = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in nd2_files])

# Landmarks
pkl = np.load(f'{BASE}/png_exports/registration_per_tile_pkl/{TILE}/pkl_transform_{TILE}.npz')
M2d = pkl['M2d_jy306_to_nd2']
M3 = np.vstack([M2d, [0, 0, 1]])
iv = pkl['pcd_invivo_jy306']   # (N,3) z,y,x in JY306
ev = pkl['ev_nd2']             # (N,3) x,y,z in nd2 4200px
n_lm = len(iv)

# Best nd2 z for mode z=3
z3_lms = [i for i in range(n_lm) if int(round(iv[i, 0])) == MODE_Z]
nd2_z_mode = Counter([int(round(ev[i, 2])) for i in z3_lms]).most_common(1)[0][0]
nd2_z_mode = max(0, min(len(nd2_stack) - 1, nd2_z_mode))
print(f"  {n_lm} landmarks, mode_z={MODE_Z}, nd2_z={nd2_z_mode}")

# ── Display images ──
z3_u8 = norm8(jy306[MODE_Z])
z3_hot = make_hot(z3_u8)

nd2_u8 = norm8(nd2_stack[nd2_z_mode])
nd2_green_full = np.zeros((4200, 4200, 3), dtype=np.uint8)
nd2_green_full[:, :, 1] = nd2_u8

# ── Layout (same as scene 4) ──
DISP_H = int(H * 0.72)
IMG_GAP = 100

scale_jy = DISP_H / hy_jy
disp_jy_w = int(wx_jy * scale_jy)
disp_jy_h = DISP_H
jy_disp = cv2.resize(z3_hot, (disp_jy_w, disp_jy_h), interpolation=cv2.INTER_LANCZOS4)

# Crop nd2 around landmarks
margin_nd2 = 350
crop_x0 = max(0, int(ev[:, 0].min() - margin_nd2))
crop_y0 = max(0, int(ev[:, 1].min() - margin_nd2))
crop_x1 = min(4200, int(ev[:, 0].max() + margin_nd2))
crop_y1 = min(4200, int(ev[:, 1].max() + margin_nd2))
nd2_crop = nd2_green_full[crop_y0:crop_y1, crop_x0:crop_x1]
scale_nd2 = DISP_H / nd2_crop.shape[0]
disp_nd2_w = int(nd2_crop.shape[1] * scale_nd2)
disp_nd2_h = DISP_H
nd2_disp = cv2.resize(nd2_crop, (disp_nd2_w, disp_nd2_h), interpolation=cv2.INTER_LANCZOS4)

total_w = disp_jy_w + IMG_GAP + disp_nd2_w
jy_x0 = (W - total_w) // 2
jy_y0 = (H - DISP_H) // 2 - 20
nd2_x0 = jy_x0 + disp_jy_w + IMG_GAP
nd2_y0 = jy_y0

# Landmark display positions
lm_jy_disp = []
lm_nd2_disp = []
for i in range(n_lm):
    dx = int(iv[i, 2] * scale_jy) + jy_x0
    dy = int(iv[i, 1] * scale_jy) + jy_y0
    lm_jy_disp.append((dx, dy))
    dx2 = int((ev[i, 0] - crop_x0) * scale_nd2) + nd2_x0
    dy2 = int((ev[i, 1] - crop_y0) * scale_nd2) + nd2_y0
    lm_nd2_disp.append((dx2, dy2))

# ── Zoom panels (top 9 by contrast, like scene 4) ──
print("Preparing zoom panels...")
jy_p1, jy_p2 = np.percentile(jy306[jy306 > 0], [1, 99.5])
CROP_R_JY = 40; CROP_R_ND2 = 100; PANEL_SZ = 110

lm_scores = []
for i in range(n_lm):
    z_i = int(round(iv[i, 0])); y_i = int(round(iv[i, 1])); x_i = int(round(iv[i, 2]))
    r = CROP_R_JY
    crop = jy306[z_i, max(0,y_i-r):min(hy_jy,y_i+r), max(0,x_i-r):min(wx_jy,x_i+r)]
    lm_scores.append(float(crop.std()))
SELECTED = sorted(range(n_lm), key=lambda i: -lm_scores[i])[:9]

zoom_panels = []
for idx in SELECTED:
    z_lm = int(round(iv[idx, 0])); y_lm = int(round(iv[idx, 1])); x_lm = int(round(iv[idx, 2]))
    z_lo, z_hi = max(0, z_lm - 2), min(nz_jy, z_lm + 3)
    mip_jy = np.max(jy306[z_lo:z_hi], axis=0)
    y0 = max(0, y_lm - CROP_R_JY); y1 = min(hy_jy, y_lm + CROP_R_JY)
    x0 = max(0, x_lm - CROP_R_JY); x1 = min(wx_jy, x_lm + CROP_R_JY)
    crop_n = np.clip((mip_jy[y0:y1, x0:x1] - jy_p1) / max(jy_p2 - jy_p1, 1) * 255, 0, 255).astype(np.uint8)
    crop_n = cv2.resize(crop_n, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_LANCZOS4)
    hot_p = cv2.applyColorMap(crop_n, cv2.COLORMAP_HOT); hot_p[crop_n < 5] = 0

    x_nd2 = int(round(ev[idx, 0])); y_nd2 = int(round(ev[idx, 1]))
    z_nd2_lm = int(round(ev[idx, 2])); z_nd2_lm = max(0, min(len(nd2_stack) - 1, z_nd2_lm))
    mip_nd2 = np.max(nd2_stack[max(0, z_nd2_lm - 2):min(len(nd2_stack), z_nd2_lm + 3)], axis=0)
    yn0 = max(0, y_nd2 - CROP_R_ND2); yn1 = min(4200, y_nd2 + CROP_R_ND2)
    xn0 = max(0, x_nd2 - CROP_R_ND2); xn1 = min(4200, x_nd2 + CROP_R_ND2)
    crop_nd2 = norm8(mip_nd2[yn0:yn1, xn0:xn1])
    crop_nd2 = cv2.resize(crop_nd2, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_LANCZOS4)
    green_p = np.zeros((PANEL_SZ, PANEL_SZ, 3), dtype=np.uint8); green_p[:, :, 1] = crop_nd2

    zoom_panels.append((hot_p, green_p))

# ── Build multi-z overlay slices for 3D rotation (Phase G) ──
print("Building multi-z overlay slices for 3D rotation...")
iv_z_min = max(0, int(iv[:, 0].min()) - 1)
iv_z_max = min(nz_jy - 1, int(iv[:, 0].max()) + 1)
z_range_3d = list(range(iv_z_min, iv_z_max + 1))

crop_w_nd2 = crop_x1 - crop_x0
crop_h_nd2 = crop_y1 - crop_y0

overlay_slices_3d = []
overlay_z_labels_3d = []

for z_iv in z_range_3d:
    iv_u8_z = norm8(jy306[z_iv])
    warped_iv_z = cv2.warpAffine(iv_u8_z, M2d, (4200, 4200),
                                  flags=cv2.INTER_LINEAR, borderValue=0)
    warped_crop_z = warped_iv_z[crop_y0:crop_y1, crop_x0:crop_x1]

    # Find best nd2 z by NCC
    best_ncc_z, best_nd2_z = -1, 0
    for zi in range(len(nd2_stack)):
        nd2_full_z = nd2_stack[zi].astype(np.uint8)
        nd2_c_z = nd2_full_z[crop_y0:min(crop_y1, nd2_full_z.shape[0]),
                             crop_x0:min(crop_x1, nd2_full_z.shape[1])]
        wc_z = warped_crop_z[:nd2_c_z.shape[0], :nd2_c_z.shape[1]]
        wn_z = norm8(wc_z); nn_z = norm8(nd2_c_z)
        mask_z = (wn_z > 5) & (nn_z > 5)
        if mask_z.sum() < 100: continue
        a_v = wn_z[mask_z].astype(np.float32); a_v -= a_v.mean()
        b_v = nn_z[mask_z].astype(np.float32); b_v -= b_v.mean()
        ncc_v = float(np.sum(a_v * b_v) / (np.sqrt(np.sum(a_v**2) * np.sum(b_v**2)) + 1e-8))
        if ncc_v > best_ncc_z: best_ncc_z, best_nd2_z = ncc_v, zi

    nd2_best_z = nd2_stack[best_nd2_z].astype(np.uint8)
    nd2_c_best = nd2_best_z[crop_y0:min(crop_y1, nd2_best_z.shape[0]),
                            crop_x0:min(crop_x1, nd2_best_z.shape[1])]
    wc_best = warped_crop_z[:nd2_c_best.shape[0], :nd2_c_best.shape[1]]

    ov_3d = np.zeros((nd2_c_best.shape[0], nd2_c_best.shape[1], 3), np.uint8)
    ov_3d[:, :, 1] = norm8(nd2_c_best)   # green = exvivo
    ov_hot = make_hot(norm8(wc_best))      # hot = invivo
    ov_3d = cv2.addWeighted(ov_3d, 0.5, ov_hot[:nd2_c_best.shape[0], :nd2_c_best.shape[1]], 0.5, 0)

    ov_small = cv2.resize(ov_3d, (disp_nd2_w, disp_nd2_h), interpolation=cv2.INTER_AREA)
    overlay_slices_3d.append(ov_small)
    overlay_z_labels_3d.append((z_iv, best_nd2_z))
    print(f"  z_iv={z_iv} -> nd2_z={best_nd2_z}, NCC={best_ncc_z:.3f}")

n_slices_3d = len(overlay_slices_3d)
mid_idx_3d = z_range_3d.index(MODE_Z) if MODE_Z in z_range_3d else n_slices_3d // 2

# Gaussian interpolation between z-slices
print(f"Gaussian interpolation: {n_slices_3d} -> {n_slices_3d + (n_slices_3d - 1) * INTERP_PER_GAP} sub-slices...")
dense_slices = []
dense_z_pos = []
dense_real_idx = []

for i in range(n_slices_3d):
    dense_slices.append(overlay_slices_3d[i])
    dense_z_pos.append(i * Z_SPACING)
    dense_real_idx.append(i)
    if i < n_slices_3d - 1:
        for sub in range(1, INTERP_PER_GAP + 1):
            t_sub = sub / (INTERP_PER_GAP + 1)
            interp = (overlay_slices_3d[i].astype(np.float32) * (1 - t_sub) +
                      overlay_slices_3d[i + 1].astype(np.float32) * t_sub)
            dense_slices.append(interp.astype(np.uint8))
            dense_z_pos.append(i * Z_SPACING + t_sub * Z_SPACING)
            dense_real_idx.append(-1)

dense_slices = np.array(dense_slices)
dense_z_pos = np.array(dense_z_pos, dtype=np.float64)
n_dense = len(dense_slices)
STACK_CENTER_Z = (dense_z_pos[-1] + dense_z_pos[0]) / 2.0

# ── 3D rendering function ──
def render_3d_stack(rot_y, rot_x, slice_alphas, center=None):
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cx, cy = center if center else (W // 2, H // 2)

    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

    z_depths = []
    for i in range(n_dense):
        dz = dense_z_pos[i] - STACK_CENTER_Z
        rz = -sin_y * 0 + cos_y * dz
        rz2 = sin_x * 0 + cos_x * rz
        z_depths.append((rz2, i))
    z_depths.sort(key=lambda x: x[0])

    for depth, i in z_depths:
        real_idx = dense_real_idx[i]
        if real_idx >= 0:
            alpha = slice_alphas[real_idx] if real_idx < len(slice_alphas) else 0.5
        else:
            zp = dense_z_pos[i]
            z_below = int(zp / Z_SPACING)
            z_above = min(n_slices_3d - 1, z_below + 1)
            t_a = (zp - z_below * Z_SPACING) / Z_SPACING if Z_SPACING > 0 else 0
            a_b = slice_alphas[z_below] if z_below < len(slice_alphas) else 0.5
            a_a = slice_alphas[z_above] if z_above < len(slice_alphas) else 0.5
            alpha = a_b * (1 - t_a) + a_a * t_a

        if alpha < 0.01: continue

        sl = dense_slices[i].astype(np.float32) / 255.0
        sh, sw = sl.shape[:2]
        hw, hh = sw / 2, sh / 2
        dz = dense_z_pos[i] - STACK_CENTER_Z

        corners_3d = np.array([
            [-hw, -hh, dz], [hw, -hh, dz],
            [hw, hh, dz], [-hw, hh, dz],
        ], dtype=np.float64)

        rot_corners = []
        for c in corners_3d:
            rx = cos_y * c[0] + sin_y * c[2]
            ry = c[1]
            rz = -sin_y * c[0] + cos_y * c[2]
            ry2 = cos_x * ry - sin_x * rz
            rot_corners.append([rx + cx, ry2 + cy])

        rot_corners = np.array(rot_corners, dtype=np.float32)
        src_corners = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float32)

        M_persp = cv2.getPerspectiveTransform(src_corners, rot_corners)
        warped_sl = cv2.warpPerspective(sl, M_persp, (W, H))

        mask_sl = np.max(warped_sl, axis=2) > 0.01
        mask3 = np.stack([mask_sl] * 3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, warped_sl * alpha), canvas)

    return np.clip(canvas * 255, 0, 255).astype(np.uint8)

# ── Helper: draw side-by-side base ──
def draw_base(frame):
    frame[jy_y0:jy_y0 + disp_jy_h, jy_x0:jy_x0 + disp_jy_w] = jy_disp
    frame[nd2_y0:nd2_y0 + disp_nd2_h, nd2_x0:nd2_x0 + disp_nd2_w] = nd2_disp
    cv2.putText(frame, f'IN-VIVO  z={MODE_Z}', (jy_x0 + 10, jy_y0 - 12),
                FONT, 0.5, (100, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f'EX-VIVO  {TILE}  z={nd2_z_mode}', (nd2_x0 + 10, nd2_y0 - 12),
                FONT, 0.5, (100, 255, 100), 1, cv2.LINE_AA)

# ── Video ──
vw = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

# ═══════════════════════════════════════════════════════════
# PHASE A: Side-by-side appear (1.5s = 36 frames)
# ═══════════════════════════════════════════════════════════
print("Phase A: side-by-side appear (1.5s)...")
for fi in range(36):
    t = ease(fi / 28)
    frame = np.zeros((H, W, 3), np.uint8)
    jy_d = (jy_disp.astype(np.float32) * t).astype(np.uint8)
    nd2_d = (nd2_disp.astype(np.float32) * t).astype(np.uint8)
    frame[jy_y0:jy_y0 + disp_jy_h, jy_x0:jy_x0 + disp_jy_w] = jy_d
    slide = int((1 - t) * 200)
    rx = nd2_x0 + slide; rw = min(disp_nd2_w, W - rx)
    if rw > 0:
        frame[nd2_y0:nd2_y0 + disp_nd2_h, rx:rx + rw] = nd2_d[:, :rw]
    cv2.putText(frame, f'IN-VIVO  z={MODE_Z}', (jy_x0 + 10, jy_y0 - 12),
                FONT, 0.5, tuple(int(v * t) for v in (100, 180, 255)), 1, cv2.LINE_AA)
    cv2.putText(frame, f'EX-VIVO  {TILE}  z={nd2_z_mode}', (nd2_x0 + 10, nd2_y0 - 12),
                FONT, 0.5, tuple(int(v * t) for v in (100, 255, 100)), 1, cv2.LINE_AA)
    caption(frame, f'TILE  {TILE.upper()}  --  NATIVE  SPACES', alpha=t)
    vw.write(frame)

# ═══════════════════════════════════════════════════════════
# PHASE B: Green arrows one at a time + zoom panels (9s = 216 frames)
# ═══════════════════════════════════════════════════════════
print("Phase B: landmarks + zoom panels (9s)...")
frames_per_lm = 216 // len(SELECTED)

for fi in range(216):
    frame = np.zeros((H, W, 3), np.uint8)
    draw_base(frame)

    current_lm = -1
    for li in range(len(SELECTED)):
        idx = SELECTED[li]
        appear = li * frames_per_lm
        age = fi - appear
        if age < 0: continue

        jpt = lm_jy_disp[idx]; npt = lm_nd2_disp[idx]
        cv2.circle(frame, jpt, 8, GREEN, 2, cv2.LINE_AA)
        cv2.circle(frame, npt, 8, GREEN, 2, cv2.LINE_AA)

        progress = min(1.0, age / 14.0)
        if progress >= 1.0:
            draw_arrow(frame, jpt, npt, GREEN, 2, 0.025)
        else:
            mid_x = int(jpt[0] * (1 - progress) + npt[0] * progress)
            mid_y = int(jpt[1] * (1 - progress) + npt[1] * progress)
            cv2.line(frame, jpt, (mid_x, mid_y), GREEN, 2, cv2.LINE_AA)

        if age < frames_per_lm: current_lm = li

    # Zoom panel
    if current_lm >= 0:
        zh, zg = zoom_panels[current_lm]
        appear = current_lm * frames_per_lm
        age = fi - appear
        p_alpha = ease(min(1.0, age / 8.0)) * ease(max(0, 1 - (age - frames_per_lm + 8) / 8.0))

        if p_alpha > 0.01:
            pw = PANEL_SZ; gap = 30
            total_pw = pw * 2 + gap
            px_s = (W - total_pw) // 2
            py_s = H - PANEL_SZ - 85
            border = 2

            hp = (zh.astype(np.float32) * p_alpha).astype(np.uint8)
            frame[py_s:py_s + pw, px_s:px_s + pw] = hp
            cv2.rectangle(frame, (px_s - border, py_s - border),
                          (px_s + pw + border, py_s + pw + border), GREEN, border)

            gp = (zg.astype(np.float32) * p_alpha).astype(np.uint8)
            gx = px_s + pw + gap
            frame[py_s:py_s + pw, gx:gx + pw] = gp
            cv2.rectangle(frame, (gx - border, py_s - border),
                          (gx + pw + border, py_s + pw + border), GREEN, border)

            idx_sel = SELECTED[current_lm]
            jpt = lm_jy_disp[idx_sel]; npt = lm_nd2_disp[idx_sel]
            lcol = tuple(int(v * p_alpha) for v in GREEN)
            cv2.line(frame, jpt, (px_s + pw // 2, py_s), lcol, 1, cv2.LINE_AA)
            cv2.line(frame, npt, (gx + pw // 2, py_s), lcol, 1, cv2.LINE_AA)

            la = p_alpha
            cv2.putText(frame, 'IN-VIVO', (px_s, py_s - 8), FONT, 0.38,
                        tuple(int(v * la) for v in (100, 180, 255)), 1, cv2.LINE_AA)
            cv2.putText(frame, 'EX-VIVO', (gx, py_s - 8), FONT, 0.38,
                        tuple(int(v * la) for v in (100, 255, 100)), 1, cv2.LINE_AA)

    n_shown = sum(1 for li in range(len(SELECTED)) if fi >= li * frames_per_lm)
    caption(frame, f'MATCHED  LANDMARKS  ({n_shown}/{len(SELECTED)})')
    vw.write(frame)

# ═══════════════════════════════════════════════════════════
# PHASE C: Hold all arrows (1.5s = 36 frames)
# ═══════════════════════════════════════════════════════════
print("Phase C: hold arrows (1.5s)...")
for fi in range(36):
    frame = np.zeros((H, W, 3), np.uint8)
    draw_base(frame)
    for li in range(len(SELECTED)):
        idx = SELECTED[li]
        draw_arrow(frame, lm_jy_disp[idx], lm_nd2_disp[idx], GREEN, 2, 0.025)
        cv2.circle(frame, lm_jy_disp[idx], 8, GREEN, 2, cv2.LINE_AA)
        cv2.circle(frame, lm_nd2_disp[idx], 8, GREEN, 2, cv2.LINE_AA)
    caption(frame, f'{n_lm}  MATCHED  CELLS')
    vw.write(frame)

# ═══════════════════════════════════════════════════════════
# PHASE D: Invivo moves on top of exvivo — centroid alignment (2s = 48 frames)
# Invivo slides from left position to overlap exvivo (centroid-matched),
# BEFORE affine — just translation to roughly align centroids.
# ═══════════════════════════════════════════════════════════
print("Phase D: centroid overlap (2s)...")

# Centroid of invivo landmarks in JY306 display coords
iv_cx = np.mean([lm_jy_disp[i][0] for i in range(n_lm)])
iv_cy = np.mean([lm_jy_disp[i][1] for i in range(n_lm)])
# Centroid of exvivo landmarks in nd2 display coords
ev_cx = np.mean([lm_nd2_disp[i][0] for i in range(n_lm)])
ev_cy = np.mean([lm_nd2_disp[i][1] for i in range(n_lm)])

# M_start: invivo at its current position (left side)
M_start = np.array([[scale_jy, 0, jy_x0],
                     [0, scale_jy, jy_y0],
                     [0, 0, 1]], dtype=np.float64)

# M_centroid: invivo translated so its landmark centroid matches exvivo's
# Centroid of invivo landmarks in JY306 pixel coords
iv_cx_px = np.mean(iv[:, 2])  # x
iv_cy_px = np.mean(iv[:, 1])  # y
# Where this centroid ends up with M_start
cx_start = iv_cx_px * scale_jy + jy_x0
cy_start = iv_cy_px * scale_jy + jy_y0
# Shift needed
shift_x = ev_cx - cx_start
shift_y = ev_cy - cy_start

M_centroid = np.array([[scale_jy, 0, jy_x0 + shift_x],
                        [0, scale_jy, jy_y0 + shift_y],
                        [0, 0, 1]], dtype=np.float64)

# nd2 background at overlay position
nd2_bg = np.zeros((H, W, 3), np.uint8)
nd2_bg[nd2_y0:nd2_y0 + disp_nd2_h, nd2_x0:nd2_x0 + disp_nd2_w] = nd2_disp

for fi in range(48):
    t = ease(fi / 40)
    frame = np.zeros((H, W, 3), np.uint8)

    # Exvivo stays at nd2 position, full opacity
    frame[:, :, :] = nd2_bg

    # Invivo warps from left position to centroid-aligned position
    M_t = M_start * (1 - t) + M_centroid * t
    warped = cv2.warpAffine(z3_u8, M_t[:2].astype(np.float64), (W, H),
                             flags=cv2.INTER_LANCZOS4, borderValue=0)
    w_hot = make_hot(warped)
    mask = warped > 0
    frame[mask] = cv2.addWeighted(frame[mask], 0.5, w_hot[mask], 0.5, 0)

    # Arrows fade out
    arrow_alpha = 1 - ease(t * 3)
    if arrow_alpha > 0.01:
        for li in range(len(SELECTED)):
            idx = SELECTED[li]
            acol = tuple(int(v * arrow_alpha) for v in GREEN)
            cv2.circle(frame, lm_jy_disp[idx], 6, acol, 1, cv2.LINE_AA)
            draw_arrow(frame, lm_jy_disp[idx], lm_nd2_disp[idx], acol, 1, 0.025)

    a_old = 1 - ease(fi / 15)
    a_new = ease((fi - 15) / 20)
    caption(frame, f'{n_lm}  MATCHED  CELLS', alpha=max(0, a_old))
    caption(frame, 'CENTROID  ALIGNMENT', alpha=max(0, a_new))
    vw.write(frame)

# ═══════════════════════════════════════════════════════════
# PHASE E: Affine warp M2d — centroid to full affine (4s = 96 frames)
# Now apply the actual M2d affine (rotation + scale + translation)
# ═══════════════════════════════════════════════════════════
print("Phase E: affine warp M2d (4s)...")

# M_end: full M2d in display space (nd2 crop mapped to display)
M_end = np.array([
    [scale_nd2, 0, nd2_x0 - crop_x0 * scale_nd2],
    [0, scale_nd2, nd2_y0 - crop_y0 * scale_nd2],
    [0, 0, 1]
], dtype=np.float64) @ M3

for fi in range(96):
    t = ease(fi / 85)
    frame = np.zeros((H, W, 3), np.uint8)

    # Exvivo stays
    frame[:, :, :] = nd2_bg

    # Invivo: interpolate from centroid to full M2d
    M_t = M_centroid * (1 - t) + M_end * t
    warped = cv2.warpAffine(z3_u8, M_t[:2].astype(np.float64), (W, H),
                             flags=cv2.INTER_LANCZOS4, borderValue=0)
    w_hot = make_hot(warped)
    mask = warped > 0
    frame[mask] = cv2.addWeighted(frame[mask], 0.5, w_hot[mask], 0.5, 0)

    a_new = ease((fi - 10) / 30)
    caption(frame, 'AFFINE  REGISTRATION  (M2d)', alpha=max(0, a_new))
    vw.write(frame)

# ═══════════════════════════════════════════════════════════
# PHASE F: Hold final overlay + show landmark convergence (3s = 72 frames)
# Show green circles on matched cells to confirm registration
# ═══════════════════════════════════════════════════════════
print("Phase F: final overlay + cell confirmation (3s)...")

# Final warped invivo
warped_final = cv2.warpAffine(z3_u8, M_end[:2].astype(np.float64), (W, H),
                               flags=cv2.INTER_LANCZOS4, borderValue=0)
w_final_hot = make_hot(warped_final)

final_base = nd2_bg.copy()
mask_f = warped_final > 0
final_base[mask_f] = cv2.addWeighted(final_base[mask_f], 0.5, w_final_hot[mask_f], 0.5, 0)

# Landmark positions in display space after M2d warp
lm_registered = []
for i in range(n_lm):
    # Where M2d places this invivo landmark in nd2 space
    src = np.array([iv[i, 2], iv[i, 1], 1.0])
    dst_nd2 = M2d @ src  # (x, y) in nd2 4200px
    # Map to display
    dx = int((dst_nd2[0] - crop_x0) * scale_nd2) + nd2_x0
    dy = int((dst_nd2[1] - crop_y0) * scale_nd2) + nd2_y0
    lm_registered.append((dx, dy))

for fi in range(72):
    frame = final_base.copy()

    # Progressively show green circles at registered positions
    n_show = min(n_lm, 1 + int(fi * n_lm / 40))
    for i in range(n_show):
        rpt = lm_registered[i]
        ept = lm_nd2_disp[i]
        # Green circle at exvivo position
        cv2.circle(frame, ept, 8, GREEN, 2, cv2.LINE_AA)
        # Orange circle at registered invivo position
        cv2.circle(frame, rpt, 6, (0, 140, 255), 2, cv2.LINE_AA)

    caption(frame, 'HOT = IN-VIVO    GREEN = EX-VIVO    YELLOW = OVERLAP')
    vw.write(frame)

# ═══════════════════════════════════════════════════════════
# PHASE G0: Slide 2D overlay from nd2 panel position to center (1.5s = 36 frames)
# Smooth position transition before 3D emerge
# ═══════════════════════════════════════════════════════════
print("Phase G0: slide overlay to center (1.5s)...")

# Start center: nd2 panel position
start_cx = nd2_x0 + disp_nd2_w // 2
start_cy = nd2_y0 + disp_nd2_h // 2
# End center: screen center
end_cx = W // 2
end_cy = H // 2

# Extract the overlay content from the nd2 panel region for re-rendering
# We'll re-render the mid-z overlay slice moving from panel position to center
mid_overlay = overlay_slices_3d[mid_idx_3d]  # (disp_nd2_h, disp_nd2_w, 3)

for fi in range(36):
    t = ease(fi / 30)
    frame = np.zeros((H, W, 3), np.uint8)

    # Crossfade: (1-t) = full 2D overlay at panel pos, t = single centered slice
    # Fade out the parts of the 2D scene that aren't the overlay
    if t < 1.0:
        frame_2d = final_base.copy()
        # Fade non-overlay parts (invivo left side label, etc)
        frame = (frame_2d.astype(np.float32) * (1 - t)).astype(np.uint8)

    # Render mid-slice at interpolated position
    cur_cx = int(start_cx * (1 - t) + end_cx * t)
    cur_cy = int(start_cy * (1 - t) + end_cy * t)
    sh, sw = mid_overlay.shape[:2]
    px = cur_cx - sw // 2
    py = cur_cy - sh // 2
    # Clamp to canvas
    src_x0 = max(0, -px); src_y0 = max(0, -py)
    dst_x0 = max(0, px); dst_y0 = max(0, py)
    dst_x1 = min(W, px + sw); dst_y1 = min(H, py + sh)
    src_x1 = src_x0 + (dst_x1 - dst_x0); src_y1 = src_y0 + (dst_y1 - dst_y0)
    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        region = mid_overlay[src_y0:src_y1, src_x0:src_x1]
        # Blend: at t=0 show the 2D overlay, at t=1 show the 3D slice
        alpha_slice = max(0.5, t)
        existing = frame[dst_y0:dst_y1, dst_x0:dst_x1]
        blended = cv2.addWeighted(existing, 1 - alpha_slice, region, alpha_slice, 0)
        frame[dst_y0:dst_y1, dst_x0:dst_x1] = blended

    caption(frame, 'HOT = IN-VIVO    GREEN = EX-VIVO    YELLOW = OVERLAP')
    vw.write(frame)

# ═══════════════════════════════════════════════════════════
# PHASE G1: Other z-slices emerge + tilt (2s = 48 frames)
# Starting from centered mid-slice, other slices emerge with slight tilt
# ═══════════════════════════════════════════════════════════
print("Phase G1: 3D emerge from center (2s)...")

for fi in range(48):
    t = ease(fi / 40)

    alphas = np.zeros(n_slices_3d, dtype=np.float32)
    alphas[mid_idx_3d] = 0.8
    # Other slices emerge progressively
    for si in range(n_slices_3d):
        if si == mid_idx_3d: continue
        dist = abs(si - mid_idx_3d)
        max_dist = t * (n_slices_3d - 1)
        if dist <= max_dist:
            alphas[si] = min(0.7, (max_dist - dist + 1) / 2.0) * t

    rot_x = INIT_ROT_X * t
    frame = render_3d_stack(0.0, rot_x, alphas)

    a_new = ease((fi - 10) / 20)
    caption(frame, f'3D DEPTH:  IN-VIVO Z = {iv_z_min}  TO  Z = {iv_z_max}  ALIGNED', alpha=max(0, a_new))
    vw.write(frame)

# ═══════════════════════════════════════════════════════════
# PHASE G2: 3D rotation (5s = 120 frames)
# ═══════════════════════════════════════════════════════════
print("Phase G2: 3D rotation (5s)...")
alphas_full = np.ones(n_slices_3d, dtype=np.float32) * 0.7

for fi in range(120):
    t = fi / 119.0
    rot_y = t * math.pi * 1.5
    rot_x = INIT_ROT_X + 0.15 * math.sin(t * math.pi)

    frame = render_3d_stack(rot_y, rot_x, alphas_full)
    caption(frame, f'MULTI-SLICE  ALIGNMENT  --  IN-VIVO  Z = {iv_z_min}  TO  {iv_z_max}')
    vw.write(frame)

# ═══════════════════════════════════════════════════════════
# PHASE G3: Settle front-facing (1.5s = 36 frames)
# ═══════════════════════════════════════════════════════════
print("Phase G3: settle (1.5s)...")
final_rot_y = math.pi * 1.5

for fi in range(36):
    t = ease(fi / 30)
    rot_y = final_rot_y * (1 - t)
    rot_x = INIT_ROT_X

    frame = render_3d_stack(rot_y, rot_x, alphas_full)
    caption(frame, f'MULTI-SLICE  ALIGNMENT  --  IN-VIVO  Z = {iv_z_min}  TO  {iv_z_max}')
    vw.write(frame)

# ═══════════════════════════════════════════════════════════
# PHASE G4: Hold with z-level labels (1s = 24 frames)
# ═══════════════════════════════════════════════════════════
print("Phase G4: hold with z labels (1s)...")

for fi in range(24):
    frame = render_3d_stack(0.0, INIT_ROT_X, alphas_full)
    label_alpha = ease(fi / 12)
    for si, (z_iv_l, z_nd2_l) in enumerate(overlay_z_labels_3d):
        ly = H // 2 - int((si - n_slices_3d / 2) * 28)
        col = tuple(int(v * label_alpha) for v in WHITE)
        cv2.putText(frame, f'z={z_iv_l} -- nd2 z={z_nd2_l}', (W - 280, ly),
                    FONT, 0.38, col, 1, cv2.LINE_AA)
    caption(frame, f'MULTI-SLICE  ALIGNMENT  --  {TILE.upper()}')
    vw.write(frame)

# ── Finalize ──
vw.release()
total_frames = 36 + 216 + 36 + 48 + 96 + 72 + 36 + 48 + 120 + 36 + 24
print(f"\nRe-encoding to H.264... ({total_frames} frames, {total_frames / FPS:.1f}s)")
subprocess.run([
    'ffmpeg', '-y', '-i', TMP,
    '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
    OUT
], capture_output=True)
os.remove(TMP)
print(f"Done! {total_frames} frames, {total_frames / FPS:.1f}s @ {FPS}fps -- {OUT}")