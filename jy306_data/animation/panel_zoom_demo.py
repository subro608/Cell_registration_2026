#!/usr/bin/env python3
"""Demo: 4-panel joint zoom into center cell, then zoom out.
Shows one cell (row1_3 #6) — all panels zoom together focused on crosshair.
Output: animation/panel_zoom_demo.mp4
"""
import numpy as np, cv2, math, os, re
import tifffile

BASE = '/Users/neurolab/neuroinformatics/margaret'
W, H = 1920, 1080
FPS = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)

CROP_ND2 = 130  # initial crop radius in nd2 px
PATCH_SZ = 400  # display panel size
GAP = 20

def ease(t):
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))

def norm8(img, lo=1, hi=99.5):
    v = img[img > 0]
    if len(v) < 100: return np.zeros_like(img, dtype=np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img - p1) / max(p2 - p1, 1) * 255, 0, 255).astype(np.uint8)

def crop_patch(img, cx, cy, r):
    h, w = img.shape[:2]
    ndim = img.shape[2] if img.ndim == 3 else None
    out = np.zeros((2*r, 2*r, ndim) if ndim else (2*r, 2*r), dtype=img.dtype)
    sx0, sy0 = max(0, cx-r), max(0, cy-r)
    sx1, sy1 = min(w, cx+r), min(h, cy+r)
    dx0, dy0 = sx0-(cx-r), sy0-(cy-r)
    if sx1 > sx0 and sy1 > sy0:
        out[dy0:dy0+(sy1-sy0), dx0:dx0+(sx1-sx0)] = img[sy0:sy1, sx0:sx1]
    return out

# ── Load data for row1_3 LM#6 ──
tile, lm_idx = 'row1_3', 6
print(f"Loading {tile} LM#{lm_idx}...")

tfm = np.load(f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz')
ev_nd2 = tfm['ev_nd2']
pcd_iv = tfm['pcd_invivo_jy306']
M2d = tfm['M2d_jy306_to_nd2']

cx_nd2 = int(round(ev_nd2[lm_idx, 0]))
cy_nd2 = int(round(ev_nd2[lm_idx, 1]))
z_nd2 = int(round(ev_nd2[lm_idx, 2]))
z_iv = int(round(np.clip(pcd_iv[lm_idx, 0], 0, 15)))

iv_vol = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)

# Calcium warped to nd2
M_movie2jy = np.load(f'{BASE}/animation/movie_avi_to_jy306_affine.npz')['M_affine']
cap = cv2.VideoCapture(f'{BASE}/movie_rolling_avg_win12_step3_short.avi')
cal_frames_nd2 = []
M_m2j = np.vstack([M_movie2jy, [0, 0, 1]])
M_j2n = np.vstack([M2d, [0, 0, 1]])
M_movie2nd2 = (M_j2n @ M_m2j)[:2, :]
while True:
    ret, frm = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warped = cv2.warpAffine(gray, M_movie2nd2, (4200, 4200), borderValue=0)
    cal_frames_nd2.append(warped)
cap.release()
print(f"  {len(cal_frames_nd2)} calcium frames warped to nd2")

# In-vivo warped
iv_warped = cv2.warpAffine(norm8(iv_vol[z_iv]), M2d, (4200, 4200), borderValue=0)

# Ex-vivo
nd2_slice = cv2.imread(f'{BASE}/png_exports/registration_video/{tile}/GFP_z{z_nd2:03d}.png',
                        cv2.IMREAD_GRAYSCALE).astype(np.float32)

# MERSCOPE overlay
OVERLAY_DIR = f'{BASE}/png_exports/merscope_overlay'
EX_DIR = f'{BASE}/exvivo_merscope_combined'
tile_to_ms = {}
for fname in os.listdir(EX_DIR):
    m = re.match(r'(\d+_\d+)_merscope(\d+)\.tif', fname)
    if m:
        tile_to_ms[f'row{m.group(1)}'] = int(m.group(2))

ms_id = tile_to_ms.get(tile)
overlay_gcamp = None
overlay_dots = None
if ms_id is not None:
    row_col = tile.replace('row', '')
    ov_path = f'{OVERLAY_DIR}/region_{ms_id}_{row_col}.png'
    if os.path.exists(ov_path):
        full_ov = cv2.imread(ov_path)
        left = full_ov[32:, :4200]
        right = full_ov[32:, 4208:]
        overlay_gcamp = left[:, :, 1]
        overlay_dots = np.clip(right.astype(np.int16) - left.astype(np.int16), 0, 255).astype(np.uint8)
        print(f"  MERSCOPE overlay loaded")

del iv_vol

# ── Build full nd2-space images for each panel (large crop for zoom) ──
# Use a large crop so we can zoom in/out
BIG_R = 400  # big crop radius for zoom range

def get_panel_images(crop_r):
    """Get 4 panels at given crop radius, all centered on (cx_nd2, cy_nd2)."""
    # Calcium (use frame 0 avg for static)
    cal_avg = np.mean(cal_frames_nd2, axis=0).astype(np.float32)
    cal_p = crop_patch(norm8(cal_avg), cx_nd2, cy_nd2, crop_r)
    p1 = np.zeros((crop_r*2, crop_r*2, 3), np.uint8)
    p1[:,:,0] = cal_p; p1[:,:,1] = cal_p; p1[:,:,2] = cal_p

    # In-vivo warped
    iv_p_raw = crop_patch(iv_warped, cx_nd2, cy_nd2, crop_r)
    p2 = np.zeros((crop_r*2, crop_r*2, 3), np.uint8)
    p2[:,:,1] = iv_p_raw  # green

    # Ex-vivo
    ev_p_raw = crop_patch(norm8(nd2_slice), cx_nd2, cy_nd2, crop_r)
    p3 = np.zeros((crop_r*2, crop_r*2, 3), np.uint8)
    p3[:,:,2] = ev_p_raw; p3[:,:,0] = ev_p_raw  # magenta

    # MERSCOPE
    p4 = np.zeros((crop_r*2, crop_r*2, 3), np.uint8)
    if overlay_gcamp is not None:
        gc_crop = crop_patch(overlay_gcamp, cx_nd2, cy_nd2, crop_r)
        dots_crop = crop_patch(overlay_dots, cx_nd2, cy_nd2, crop_r)
        p4[:,:,0] = (gc_crop.astype(np.float32) * 0.25).astype(np.uint8)
        p4[:,:,2] = (gc_crop.astype(np.float32) * 0.25).astype(np.uint8)
        dot_px = dots_crop.max(axis=2) > 0
        p4[dot_px] = dots_crop[dot_px]

    return [p1, p2, p3, p4]

# Pre-render panels at big crop
panels_big = get_panel_images(BIG_R)
print("Panels ready")

LABELS = ['CALCIUM', 'IN-VIVO WARPED', 'EX-VIVO', 'GCaMP + GENE DOTS']

def draw_frame(panels, zoom, cal_fi=0):
    """Draw 4 panels at current zoom level. zoom=1.0 is full view, zoom>1 crops in."""
    canvas = np.zeros((H, W, 3), np.uint8)
    total_pw = PATCH_SZ * 4 + GAP * 3
    x_start = (W - total_pw) // 2
    y_start = (H - PATCH_SZ) // 2 - 20

    for i, (panel, label) in enumerate(zip(panels, LABELS)):
        ph, pw = panel.shape[:2]
        # Zoom: crop center portion
        crop_h = int(ph / zoom)
        crop_w = int(pw / zoom)
        y0 = (ph - crop_h) // 2
        x0 = (pw - crop_w) // 2
        cropped = panel[y0:y0+crop_h, x0:x0+crop_w]

        resized = cv2.resize(cropped, (PATCH_SZ, PATCH_SZ), interpolation=cv2.INTER_LANCZOS4)
        # Brightness boost
        resized = np.clip(resized.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)

        px = x_start + i * (PATCH_SZ + GAP)
        py = y_start
        canvas[py:py+PATCH_SZ, px:px+PATCH_SZ] = resized
        cv2.rectangle(canvas, (px, py), (px+PATCH_SZ-1, py+PATCH_SZ-1), (100,100,100), 1)

        # Crosshair at center
        cr = PATCH_SZ // 2
        color = (0, 255, 255)
        cv2.circle(canvas, (px+cr, py+cr), 12, color, 1, cv2.LINE_AA)
        cv2.line(canvas, (px+cr-18, py+cr), (px+cr-7, py+cr), color, 1)
        cv2.line(canvas, (px+cr+7, py+cr), (px+cr+18, py+cr), color, 1)
        cv2.line(canvas, (px+cr, py+cr-18), (px+cr, py+cr-7), color, 1)
        cv2.line(canvas, (px+cr, py+cr+7), (px+cr, py+cr+18), color, 1)

        # Label
        ts = 0.5
        (tw, _), _ = cv2.getTextSize(label, FONT, ts, 1)
        cv2.putText(canvas, label, (px + (PATCH_SZ-tw)//2, py + PATCH_SZ + 25),
                    FONT, ts, WHITE, 1, cv2.LINE_AA)

    # Title
    title = f'{tile.upper()}  LM#{lm_idx}'
    ts = 0.7
    (tw, _), _ = cv2.getTextSize(title, FONT, ts, 1)
    cv2.putText(canvas, title, ((W-tw)//2, y_start - 15), FONT, ts, WHITE, 1, cv2.LINE_AA)

    return canvas

# ── Render frames ──
OUT_DIR = f'{BASE}/animation/frames_panel_zoom_demo'
os.makedirs(OUT_DIR, exist_ok=True)

fi = 0
print("Rendering...")

# Phase 1: Hold at 1x (24 frames)
for f in range(24):
    frame = draw_frame(panels_big, 1.0)
    cv2.imwrite(f'{OUT_DIR}/frame_{fi:05d}.png', frame)
    fi += 1

# Phase 2: Zoom in 1x → 3x (48 frames)
for f in range(48):
    t = ease(f / 47)
    zoom = 1.0 + (3.0 - 1.0) * t
    frame = draw_frame(panels_big, zoom)
    cv2.imwrite(f'{OUT_DIR}/frame_{fi:05d}.png', frame)
    fi += 1

# Phase 3: Hold at 3x (36 frames)
for f in range(36):
    frame = draw_frame(panels_big, 3.0)
    cv2.imwrite(f'{OUT_DIR}/frame_{fi:05d}.png', frame)
    fi += 1

# Phase 4: Zoom out 3x → 1x (48 frames)
for f in range(48):
    t = ease(f / 47)
    zoom = 3.0 + (1.0 - 3.0) * t
    frame = draw_frame(panels_big, zoom)
    cv2.imwrite(f'{OUT_DIR}/frame_{fi:05d}.png', frame)
    fi += 1

# Phase 5: Hold at 1x (12 frames)
for f in range(12):
    frame = draw_frame(panels_big, 1.0)
    cv2.imwrite(f'{OUT_DIR}/frame_{fi:05d}.png', frame)
    fi += 1

print(f"Done! {fi} frames")

# Encode
os.system(f'ffmpeg -y -framerate {FPS} -i {OUT_DIR}/frame_%05d.png '
          f'-c:v libx264 -pix_fmt yuv420p -crf 18 '
          f'{BASE}/animation/panel_zoom_demo.mp4 2>/dev/null')
print("Saved panel_zoom_demo.mp4")
