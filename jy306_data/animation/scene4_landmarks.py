"""
Scene 4: Matching landmarks + zoom panels + slow warp to ex-vivo

Continues from scene 3 (z=3 highlighted, others faded).

Flow:
  Phase A (3s):   3D stack → isolate z=3, flatten, grow to left half
  Phase B (1.5s): Ex-vivo tile row2_1 z=11 appears on right
  Phase C (9s):   9 green arrows drawn one at a time with zoom panels
  Phase D (2s):   Hold all arrows + "27 MATCHED CELLS" caption
  Phase E (4s):   Slow dramatic warp — in-vivo deforms onto ex-vivo tile
  Phase F (2s):   Hold final overlay (red=invivo warped, green=exvivo native, yellow=match)

Output: animation/scene4_h264.mp4
"""

import numpy as np, cv2, tifffile, math, subprocess, os, glob

BASE = '/Users/neurolab/neuroinformatics/margaret'
TMP  = f'{BASE}/animation/scene4_raw.mp4'
OUT  = f'{BASE}/animation/scene4_h264.mp4'

W, H = 1920, 1080
FPS  = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)

SELECTED = [1, 2, 3, 4, 5, 6, 11, 13, 15]

# Pixel sizes from microscope metadata
IV_XY_UM = 0.82  # JY306 in-vivo (µm/px)

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

def draw_arrow(frame, pt1, pt2, color, thickness=2, tip_length=0.03):
    cv2.arrowedLine(frame, pt1, pt2, color, thickness, cv2.LINE_AA, tipLength=tip_length)

def draw_scale_bar(frame, um_per_disp_px, alpha=1.0, x_right=None, y_bottom=None):
    """Draw a scale bar with physical distance label."""
    if alpha < 0.01:
        return
    if x_right is None:
        x_right = W - 50
    if y_bottom is None:
        y_bottom = H - 75
    bar_px = 120
    bar_um = bar_px * um_per_disp_px
    if bar_um >= 10:
        bar_um_label = f'{int(round(bar_um))} um'
    else:
        bar_um_label = f'{bar_um:.1f} um'
    x_left = x_right - bar_px
    y_bar = y_bottom
    col = tuple(int(255 * alpha) for _ in range(3))
    bg = (0, 0, 0)
    cv2.rectangle(frame, (x_left - 1, y_bar - 4), (x_right + 1, y_bar + 4), bg, -1)
    cv2.rectangle(frame, (x_left, y_bar - 3), (x_right, y_bar + 3), col, -1)
    label = bar_um_label
    ts = 0.45
    (tw, _), _ = cv2.getTextSize(label, FONT, ts, 1)
    tx = x_left + (bar_px - tw) // 2
    ty = y_bar - 8
    cv2.putText(frame, label, (tx, ty), FONT, ts, bg, 3, cv2.LINE_AA)
    cv2.putText(frame, label, (tx, ty), FONT, ts, col, 1, cv2.LINE_AA)

# ── Load data ──
print("Loading data...")
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz, hy, wx = jy306.shape

# 3D rendering setup (copied from scene3)
slices_small = []
SLICE_W, SLICE_H = 400, 400
for z in range(nz):
    s = norm_u8(jy306[z])
    s = cv2.resize(s, (SLICE_W, SLICE_H), interpolation=cv2.INTER_AREA)
    slices_small.append(s)
slices_small = np.array(slices_small)

Z_SPACING = 18
STACK_CENTER_Z = (nz - 1) * Z_SPACING / 2.0
INIT_ROT_X = -0.3

def render_3d_stack(rot_y, rot_x, slice_alphas):
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cx, cy = W // 2, H // 2
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)
    z_depths = []
    for z in range(nz):
        dz = z * Z_SPACING - STACK_CENTER_Z
        rz = cos_y * dz; rz2 = cos_x * rz
        z_depths.append((rz2, z))
    z_depths.sort(key=lambda x: x[0])
    for depth, z in z_depths:
        alpha = slice_alphas[z]
        if alpha < 0.01: continue
        sl = slices_small[z].astype(np.float32) / 255.0
        sh, sw = sl.shape
        hw, hh = sw/2, sh/2
        dz = z * Z_SPACING - STACK_CENTER_Z
        corners_3d = np.array([[-hw,-hh,dz],[hw,-hh,dz],[hw,hh,dz],[-hw,hh,dz]], dtype=np.float64)
        rot_corners = []
        for c in corners_3d:
            rx = cos_y*c[0] + sin_y*c[2]; ry = c[1]; rz = -sin_y*c[0] + cos_y*c[2]
            ry2 = cos_x*ry - sin_x*rz
            rot_corners.append([rx+cx, ry2+cy])
        rot_corners = np.array(rot_corners, dtype=np.float32)
        src_corners = np.array([[0,0],[sw,0],[sw,sh],[0,sh]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_corners, rot_corners)
        warped = cv2.warpPerspective(sl, M, (W, H))
        green = np.zeros((H, W, 3), dtype=np.float32)
        green[:, :, 1] = warped  # in-vivo = green
        mask3 = np.stack([warped>0.01]*3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, green*alpha), canvas)
    return np.clip(canvas*255, 0, 255).astype(np.uint8)

# ── Load row2_1 ──
print("Loading row2_1...")
pkl = np.load(f'{BASE}/png_exports/registration_per_tile_pkl/row2_1/pkl_transform_row2_1.npz', allow_pickle=True)
M2d = pkl['M2d_jy306_to_nd2']  # jy306 (x,y) → nd2 (x,y)
M3_full = np.vstack([M2d, [0,0,1]])
M_inv = np.linalg.inv(M3_full)[:2]
iv = pkl['pcd_invivo_jy306']  # (z, y, x)
ev = pkl['ev_nd2']            # (x, y, z)
n_lm = len(iv)

# Best nd2 z-slice = 11
nd2_z11 = cv2.imread(f'{BASE}/png_exports/registration_video/row2_1/GFP_z011.png', cv2.IMREAD_GRAYSCALE)
nd2_files = sorted(glob.glob(f'{BASE}/png_exports/registration_video/row2_1/GFP_z*.png'))
nd2_stack = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in nd2_files])

# Display images
z3_u8 = norm_u8(jy306[3])
z3_green = np.zeros((*z3_u8.shape, 3), np.uint8)
z3_green[:, :, 1] = z3_u8  # in-vivo = green

nd2_u8 = norm_u8(nd2_z11)
nd2_magenta_full = np.zeros((4200, 4200, 3), dtype=np.uint8)
nd2_magenta_full[:,:,0] = nd2_u8  # B } ex-vivo = magenta
nd2_magenta_full[:,:,2] = nd2_u8  # R }

# ── Zoom panels ──
print("Preparing zoom panels...")
jy_p1, jy_p2 = np.percentile(jy306[jy306 > 0], [1, 99.5])
CROP_R_JY = 40; CROP_R_ND2 = 100; PANEL_SZ = 110

zoom_panels = []
for idx in SELECTED:
    # In-vivo MIP ±2z
    z_lm = int(round(iv[idx,0])); y_lm = int(round(iv[idx,1])); x_lm = int(round(iv[idx,2]))
    z_lo, z_hi = max(0, z_lm-2), min(nz, z_lm+3)
    mip_jy = np.max(jy306[z_lo:z_hi], axis=0)
    y0=max(0,y_lm-CROP_R_JY); y1=min(hy,y_lm+CROP_R_JY)
    x0=max(0,x_lm-CROP_R_JY); x1=min(wx,x_lm+CROP_R_JY)
    crop = mip_jy[y0:y1, x0:x1]
    crop_n = np.clip((crop-jy_p1)/max(jy_p2-jy_p1,1)*255, 0, 255).astype(np.uint8)
    crop_n = cv2.resize(crop_n, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_LANCZOS4)
    green_p_iv = np.zeros((PANEL_SZ, PANEL_SZ, 3), dtype=np.uint8)
    green_p_iv[:,:,1] = crop_n  # in-vivo = green

    # Ex-vivo MIP ±2z
    x_nd2=int(round(ev[idx,0])); y_nd2=int(round(ev[idx,1]))
    mip_nd2 = np.max(nd2_stack[max(0,9):12], axis=0)  # z9-11 for MIP around z=11
    yn0=max(0,y_nd2-CROP_R_ND2); yn1=min(4200,y_nd2+CROP_R_ND2)
    xn0=max(0,x_nd2-CROP_R_ND2); xn1=min(4200,x_nd2+CROP_R_ND2)
    crop_nd2 = mip_nd2[yn0:yn1, xn0:xn1]
    crop_nd2_n = norm_u8(crop_nd2)
    crop_nd2_n = cv2.resize(crop_nd2_n, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_LANCZOS4)
    magenta_p = np.zeros((PANEL_SZ, PANEL_SZ, 3), dtype=np.uint8)
    magenta_p[:,:,0] = crop_nd2_n  # B } ex-vivo = magenta
    magenta_p[:,:,2] = crop_nd2_n  # R }

    zoom_panels.append((green_p_iv, magenta_p))

# ── Layout ──
DISP_H = int(H * 0.72)
IMG_GAP = 100

# In-vivo display
scale_jy = DISP_H / hy
disp_jy_w = int(wx * scale_jy); disp_jy_h = DISP_H
jy_disp = cv2.resize(z3_green, (disp_jy_w, disp_jy_h), interpolation=cv2.INTER_LANCZOS4)

# Ex-vivo: crop to landmark region
lm_x_nd2 = ev[:, 0]; lm_y_nd2 = ev[:, 1]
margin_nd2 = 350
crop_x0 = max(0, int(lm_x_nd2.min()-margin_nd2)); crop_y0 = max(0, int(lm_y_nd2.min()-margin_nd2))
crop_x1 = min(4200, int(lm_x_nd2.max()+margin_nd2)); crop_y1 = min(4200, int(lm_y_nd2.max()+margin_nd2))
nd2_crop = nd2_magenta_full[crop_y0:crop_y1, crop_x0:crop_x1]
scale_nd2 = DISP_H / nd2_crop.shape[0]
disp_nd2_w = int(nd2_crop.shape[1] * scale_nd2); disp_nd2_h = DISP_H
nd2_disp = cv2.resize(nd2_crop, (disp_nd2_w, disp_nd2_h), interpolation=cv2.INTER_LANCZOS4)

total_w = disp_jy_w + IMG_GAP + disp_nd2_w
jy_x0 = (W - total_w) // 2
jy_y0 = (H - DISP_H) // 2 - 20
nd2_x0 = jy_x0 + disp_jy_w + IMG_GAP
nd2_y0 = jy_y0

# Landmark display positions
lm_jy_disp = []; lm_nd2_disp = []
for idx in SELECTED:
    dx = int(iv[idx,2] * scale_jy) + jy_x0
    dy = int(iv[idx,1] * scale_jy) + jy_y0
    lm_jy_disp.append((dx, dy))
    dx2 = int((ev[idx,0] - crop_x0) * scale_nd2) + nd2_x0
    dy2 = int((ev[idx,1] - crop_y0) * scale_nd2) + nd2_y0
    lm_nd2_disp.append((dx2, dy2))

# ── Video ──
vw = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

# ═══════════════════════════════════════════════════════════════
# PHASE A: 3D stack → isolate z=3, flatten, grow to left half (72 fr = 3s)
# ═══════════════════════════════════════════════════════════════
print("Phase A: 3D→z=3 flatten (3s)...")

z3_green_frames = []
for fi in range(72):
    t = ease(fi / 71)
    nw = int(round(SLICE_W*(1-t) + disp_jy_w*t))
    nh = int(round(SLICE_H*(1-t) + disp_jy_h*t))
    z3_green_frames.append(cv2.resize(z3_green, (max(1,nw), max(1,nh)), interpolation=cv2.INTER_LANCZOS4))

for fi in range(72):
    t = ease(fi / 71)
    if t < 0.4:
        t_fade = t / 0.4
        slice_alphas = np.ones(nz, np.float32) * 0.14 * (1 - ease(t_fade))
        slice_alphas[3] = 1.0
        rot_x_t = INIT_ROT_X * (1 - ease(t_fade) * 0.7)
        frame = render_3d_stack(0.0, rot_x_t, slice_alphas)
    else:
        t_grow = (t - 0.4) / 0.6; t_g = ease(t_grow)
        frame = np.zeros((H, W, 3), np.uint8)
        img = z3_green_frames[fi]; ih, iw = img.shape[:2]
        cx_s, cy_s = W//2, H//2
        cx_e = jy_x0 + disp_jy_w//2; cy_e = jy_y0 + disp_jy_h//2
        cx_t = int(cx_s*(1-t_g) + cx_e*t_g); cy_t = int(cy_s*(1-t_g) + cy_e*t_g)
        px, py = cx_t - iw//2, cy_t - ih//2
        sx, sy = max(0,-px), max(0,-py); fx, fy = max(0,px), max(0,py)
        fw, fh = min(iw-sx, W-fx), min(ih-sy, H-fy)
        if fw>0 and fh>0: frame[fy:fy+fh, fx:fx+fw] = img[sy:sy+fh, sx:sx+fw]

    a_old = 1 - ease(fi/15)
    caption(frame, 'BEST ALIGNMENT:  Z = 3  --  EX VIVO  TILE  ROW2_1', alpha=max(0, a_old))
    # Scale bar: fade in once z=3 is nearly at final left-half position (last 30% of animation)
    um_disp_jy = IV_XY_UM / scale_jy
    sb_alpha = max(0.0, ease((t - 0.75) / 0.25)) if t > 0.7 else 0.0
    draw_scale_bar(frame, um_disp_jy, alpha=sb_alpha)
    vw.write(frame)

# ── Finalize ──
vw.release()
print("Re-encoding to H.264...")
subprocess.run([
    'ffmpeg', '-y', '-i', TMP,
    '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
    OUT
], capture_output=True)
os.remove(TMP)

total_fr = 72
total_s = total_fr / FPS
print(f"Done! {total_fr} frames, {total_s:.1f}s @ {FPS}fps → {OUT}")
