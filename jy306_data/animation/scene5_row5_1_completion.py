"""
Generate the missing row5_1 frames (phases A-F) + final stitched 3D
to complete scene5. Reuses the exact same logic as scene5_all_tiles_v2.py.
"""
import numpy as np, cv2, tifffile, math, subprocess, os, glob
from collections import Counter

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_DIR = f'{BASE}/animation/frames_row5_1_completion'
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 1920, 1080
FPS  = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREEN = (0, 220, 0)
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

# ── Load shared data ──
print("Loading JY306 z-stack...")
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_jy, hy_jy, wx_jy = jy306.shape
jy_p1, jy_p2 = np.percentile(jy306[jy306 > 0], [1, 99.5])

tile = 'row5_1'
N_A, N_B, N_C, N_D, N_E, N_F = 12, 96, 12, 36, 72, 48
N_TOP_LM = 7

# ── Load tile data ──
print(f"Loading {tile} data...")
nd2_files = sorted(glob.glob(f'{BASE}/png_exports/registration_video/{tile}/GFP_z*.png'))
nd2_stack = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in nd2_files])

pkl = np.load(f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz')
M2d = pkl['M2d_jy306_to_nd2']
M3 = np.vstack([M2d, [0, 0, 1]])
iv = pkl['pcd_invivo_jy306']
ev = pkl['ev_nd2']
n_lm = len(iv)

MODE_Z = int(round(np.median(iv[:, 0])))
MODE_Z = max(0, min(nz_jy - 1, MODE_Z))

z_lms = [i for i in range(n_lm) if int(round(iv[i, 0])) == MODE_Z]
if z_lms:
    nd2_z_mode = Counter([int(round(ev[i, 2])) for i in z_lms]).most_common(1)[0][0]
else:
    nd2_z_mode = Counter([int(round(ev[i, 2])) for i in range(n_lm)]).most_common(1)[0][0]
nd2_z_mode = max(0, min(len(nd2_stack) - 1, nd2_z_mode))

# ── Display images ──
z_u8 = norm8(jy306[MODE_Z])
z_hot = make_hot(z_u8)

nd2_u8 = norm8(nd2_stack[nd2_z_mode])
nd2_green_full = np.zeros((4200, 4200, 3), dtype=np.uint8)
nd2_green_full[:, :, 1] = nd2_u8

# ── Layout ──
DISP_H = int(H * 0.72)
IMG_GAP = 100

scale_jy = DISP_H / hy_jy
disp_jy_w = int(wx_jy * scale_jy)
disp_jy_h = DISP_H
jy_disp = cv2.resize(z_hot, (disp_jy_w, disp_jy_h), interpolation=cv2.INTER_LANCZOS4)

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

# ── Zoom panels ──
CROP_R_JY = 40; CROP_R_ND2 = 100; PANEL_SZ = 110
lm_scores = []
for i in range(n_lm):
    z_i = int(round(iv[i, 0])); y_i = int(round(iv[i, 1])); x_i = int(round(iv[i, 2]))
    r = CROP_R_JY
    crop = jy306[max(0, z_i), max(0, y_i-r):min(hy_jy, y_i+r), max(0, x_i-r):min(wx_jy, x_i+r)]
    lm_scores.append(float(crop.std()))
SELECTED = sorted(range(n_lm), key=lambda i: -lm_scores[i])[:N_TOP_LM]

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

# ── Helpers ──
def draw_base(frame):
    frame[jy_y0:jy_y0 + disp_jy_h, jy_x0:jy_x0 + disp_jy_w] = jy_disp
    frame[nd2_y0:nd2_y0 + disp_nd2_h, nd2_x0:nd2_x0 + disp_nd2_w] = nd2_disp
    cv2.putText(frame, f'IN-VIVO  z={MODE_Z}', (jy_x0 + 10, jy_y0 - 12),
                FONT, 0.5, (100, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f'EX-VIVO  {tile}  z={nd2_z_mode}', (nd2_x0 + 10, nd2_y0 - 12),
                FONT, 0.5, (100, 255, 100), 1, cv2.LINE_AA)

frame_count = 0

# ═══ PHASE A: Side-by-side appear ═══
print("Phase A: appear...")
for fi in range(N_A):
    t = ease(fi / max(1, N_A - 8))
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
    cv2.putText(frame, f'EX-VIVO  {tile}  z={nd2_z_mode}', (nd2_x0 + 10, nd2_y0 - 12),
                FONT, 0.5, tuple(int(v * t) for v in (100, 255, 100)), 1, cv2.LINE_AA)
    caption(frame, f'TILE  {tile.upper()}  --  NATIVE  SPACES', alpha=t)
    cv2.imwrite(f'{OUT_DIR}/frame_{frame_count:05d}.png', frame)
    frame_count += 1

# ═══ PHASE B: Landmarks + zoom panels ═══
print("Phase B: landmarks...")
frames_per_lm = N_B // len(SELECTED)
for fi in range(N_B):
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
            jpt2 = lm_jy_disp[idx_sel]; npt2 = lm_nd2_disp[idx_sel]
            lcol = tuple(int(v * p_alpha) for v in GREEN)
            cv2.line(frame, jpt2, (px_s + pw // 2, py_s), lcol, 1, cv2.LINE_AA)
            cv2.line(frame, npt2, (gx + pw // 2, py_s), lcol, 1, cv2.LINE_AA)

            cv2.putText(frame, 'IN-VIVO', (px_s, py_s - 8), FONT, 0.38,
                        tuple(int(v * p_alpha) for v in (100, 180, 255)), 1, cv2.LINE_AA)
            cv2.putText(frame, 'EX-VIVO', (gx, py_s - 8), FONT, 0.38,
                        tuple(int(v * p_alpha) for v in (100, 255, 100)), 1, cv2.LINE_AA)

    n_shown = sum(1 for li in range(len(SELECTED)) if fi >= li * frames_per_lm)
    caption(frame, f'MATCHED  LANDMARKS  ({n_shown}/{len(SELECTED)})')
    cv2.imwrite(f'{OUT_DIR}/frame_{frame_count:05d}.png', frame)
    frame_count += 1

# ═══ PHASE C: Hold all arrows ═══
print("Phase C: hold arrows...")
for fi in range(N_C):
    frame = np.zeros((H, W, 3), np.uint8)
    draw_base(frame)
    for li in range(len(SELECTED)):
        idx = SELECTED[li]
        draw_arrow(frame, lm_jy_disp[idx], lm_nd2_disp[idx], GREEN, 2, 0.025)
        cv2.circle(frame, lm_jy_disp[idx], 8, GREEN, 2, cv2.LINE_AA)
        cv2.circle(frame, lm_nd2_disp[idx], 8, GREEN, 2, cv2.LINE_AA)
    caption(frame, f'{n_lm}  MATCHED  CELLS')
    cv2.imwrite(f'{OUT_DIR}/frame_{frame_count:05d}.png', frame)
    frame_count += 1

# ═══ PHASE D: Centroid alignment ═══
print("Phase D: centroid alignment...")
iv_cx_mean = np.mean([lm_jy_disp[i][0] for i in range(n_lm)])
iv_cy_mean = np.mean([lm_jy_disp[i][1] for i in range(n_lm)])
ev_cx_mean = np.mean([lm_nd2_disp[i][0] for i in range(n_lm)])
ev_cy_mean = np.mean([lm_nd2_disp[i][1] for i in range(n_lm)])

M_start = np.array([[scale_jy, 0, jy_x0],
                     [0, scale_jy, jy_y0],
                     [0, 0, 1]], dtype=np.float64)

iv_cx_px = np.mean(iv[:, 2])
iv_cy_px = np.mean(iv[:, 1])
cx_start = iv_cx_px * scale_jy + jy_x0
cy_start = iv_cy_px * scale_jy + jy_y0
shift_x = ev_cx_mean - cx_start
shift_y = ev_cy_mean - cy_start

M_centroid = np.array([[scale_jy, 0, jy_x0 + shift_x],
                        [0, scale_jy, jy_y0 + shift_y],
                        [0, 0, 1]], dtype=np.float64)

nd2_bg = np.zeros((H, W, 3), np.uint8)
nd2_bg[nd2_y0:nd2_y0 + disp_nd2_h, nd2_x0:nd2_x0 + disp_nd2_w] = nd2_disp

for fi in range(N_D):
    t = ease(fi / max(1, N_D - 8))
    frame = np.zeros((H, W, 3), np.uint8)
    frame[:, :, :] = nd2_bg

    M_t = M_start * (1 - t) + M_centroid * t
    warped = cv2.warpAffine(z_u8, M_t[:2].astype(np.float64), (W, H),
                             flags=cv2.INTER_LANCZOS4, borderValue=0)
    w_hot = make_hot(warped)
    mask = warped > 0
    frame[mask] = cv2.addWeighted(frame[mask], 0.5, w_hot[mask], 0.5, 0)

    arrow_alpha = 1 - ease(t * 3)
    if arrow_alpha > 0.01:
        for li in range(len(SELECTED)):
            idx = SELECTED[li]
            acol = tuple(int(v * arrow_alpha) for v in GREEN)
            cv2.circle(frame, lm_jy_disp[idx], 6, acol, 1, cv2.LINE_AA)
            draw_arrow(frame, lm_jy_disp[idx], lm_nd2_disp[idx], acol, 1, 0.025)

    a_old = 1 - ease(fi / max(1, N_D * 0.3))
    a_new = ease((fi - N_D * 0.3) / max(1, N_D * 0.4))
    caption(frame, f'{n_lm}  MATCHED  CELLS', alpha=max(0, a_old))
    caption(frame, 'CENTROID  ALIGNMENT', alpha=max(0, a_new))
    cv2.imwrite(f'{OUT_DIR}/frame_{frame_count:05d}.png', frame)
    frame_count += 1

# ═══ PHASE E: Affine warp M2d ═══
print("Phase E: affine warp...")
M_end = np.array([
    [scale_nd2, 0, nd2_x0 - crop_x0 * scale_nd2],
    [0, scale_nd2, nd2_y0 - crop_y0 * scale_nd2],
    [0, 0, 1]
], dtype=np.float64) @ M3

for fi in range(N_E):
    t = ease(fi / max(1, N_E - 11))
    frame = np.zeros((H, W, 3), np.uint8)
    frame[:, :, :] = nd2_bg

    M_t = M_centroid * (1 - t) + M_end * t
    warped = cv2.warpAffine(z_u8, M_t[:2].astype(np.float64), (W, H),
                             flags=cv2.INTER_LANCZOS4, borderValue=0)
    w_hot = make_hot(warped)
    mask = warped > 0
    frame[mask] = cv2.addWeighted(frame[mask], 0.5, w_hot[mask], 0.5, 0)

    a_new = ease((fi - 10) / 30)
    caption(frame, 'AFFINE  REGISTRATION  (M2d)', alpha=max(0, a_new))
    cv2.imwrite(f'{OUT_DIR}/frame_{frame_count:05d}.png', frame)
    frame_count += 1

# ═══ PHASE F: Final overlay + cell confirmation ═══
print("Phase F: cell confirmation...")
warped_final = cv2.warpAffine(z_u8, M_end[:2].astype(np.float64), (W, H),
                               flags=cv2.INTER_LANCZOS4, borderValue=0)
w_final_hot = make_hot(warped_final)

final_base = nd2_bg.copy()
mask_f = warped_final > 0
final_base[mask_f] = cv2.addWeighted(final_base[mask_f], 0.5, w_final_hot[mask_f], 0.5, 0)

lm_registered = []
for i in range(n_lm):
    src = np.array([iv[i, 2], iv[i, 1], 1.0])
    dst_nd2 = M2d @ src
    dx = int((dst_nd2[0] - crop_x0) * scale_nd2) + nd2_x0
    dy = int((dst_nd2[1] - crop_y0) * scale_nd2) + nd2_y0
    lm_registered.append((dx, dy))

for fi in range(N_F):
    frame = final_base.copy()
    n_show = min(n_lm, 1 + int(fi * n_lm / max(1, N_F * 0.55)))
    for i in range(n_show):
        rpt = lm_registered[i]
        ept = lm_nd2_disp[i]
        cv2.circle(frame, ept, 8, GREEN, 2, cv2.LINE_AA)
        cv2.circle(frame, rpt, 6, (0, 140, 255), 2, cv2.LINE_AA)
    caption(frame, 'HOT = IN-VIVO    GREEN = EX-VIVO    YELLOW = OVERLAP')
    cv2.imwrite(f'{OUT_DIR}/frame_{frame_count:05d}.png', frame)
    frame_count += 1

total_expected = N_A + N_B + N_C + N_D + N_E + N_F
print(f"\nDone! Generated {frame_count} frames ({frame_count/FPS:.1f}s) in {OUT_DIR}")
print(f"Expected: {total_expected} frames")
