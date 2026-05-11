"""
Test: 2D native → 3D → register in-vivo to ex-vivo.

Phase 1 (96fr, 4s):  Side-by-side 2D: IV MIP (left) | EV MIP (right) in native spaces
Phase 2 (120fr, 5s): 3D clouds emerge on top of each panel, 2D dims
Phase 3 (120fr, 5s): IV cloud slides + warps to align with EV cloud
Phase 4 (96fr, 4s):  Merged cloud rotates showing depth
Total: 432fr = 18s

Rendering: gaussian-splatted point clouds with additive blending (matches v4 viewer quality).
"""
import numpy as np, cv2, tifffile, math, subprocess, os

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT  = f'{BASE}/animation/invivo_3d_test.mp4'
TMP  = f'{BASE}/animation/invivo_3d_test_raw.mp4'
W, H = 1920, 1080; FPS = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX

def norm_u8(img, lo=1, hi=99.5):
    v = img.ravel(); v = v[v>0]
    if len(v) < 50: return np.zeros(img.shape, np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img.astype(np.float32)-p1)/max(p2-p1,1)*255, 0, 255).astype(np.uint8)

def ease(t): return float(0.5 - 0.5*math.cos(math.pi*max(0., min(1., t))))

def fit_into(img, tw, th):
    if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]; s = min(tw/w, th/h)
    nw, nh = int(w*s), int(h*s)
    rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    c = np.zeros((th, tw, 3), np.uint8)
    yo, xo = (th-nh)//2, (tw-nw)//2
    c[yo:yo+nh, xo:xo+nw] = rs
    return c

def place(frame, img, cy, cx):
    ih, iw = img.shape[:2]; y0, x0 = cy-ih//2, cx-iw//2
    fy0, fy1 = max(0,y0), min(H,y0+ih); fx0, fx1 = max(0,x0), min(W,x0+iw)
    if fy1>fy0 and fx1>fx0:
        frame[fy0:fy1, fx0:fx1] = img[fy0-y0:fy1-y0, fx0-x0:fx1-x0]

# ── Load 2D images ───────────────────────────────────────────────────────────
print("Loading 2D images...")
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
iv_mip_u8 = norm_u8(np.max(jy306, axis=0))
del jy306
print(f"  IV MIP: {iv_mip_u8.shape}")

print("Computing EV MIP from stitched volume...")
EV_TIF = f'{BASE}/registration_video/stitched/stitched_gfp_elastix_1um_isotropic.tif'
with tifffile.TiffFile(EV_TIF) as tif:
    ev_nz = len(tif.pages)
    ev_h, ev_w = tif.pages[0].shape
    ev_mip = np.zeros((ev_h, ev_w), np.float32)
    for i in range(0, ev_nz, 4):
        sl = tif.pages[i].asarray().astype(np.float32)
        np.maximum(ev_mip, sl, out=ev_mip)
ev_mip_u8 = norm_u8(ev_mip)
del ev_mip
print(f"  EV MIP: {ev_mip_u8.shape}")

# ── Load 3D point clouds (pre-aligned, from v4 viewer) ──────────────────────
print("Loading 3D voxels...")
vox = np.load('/tmp/viewer_v4_voxels.npz')
ex_vx = vox['ex_vx'].astype(np.float32); ex_vy = vox['ex_vy'].astype(np.float32)
ex_vz = vox['ex_vz'].astype(np.float32); ex_vv = vox['ex_vv'].astype(np.float32)
iv_vx = vox['iv_vx'].astype(np.float32); iv_vy = vox['iv_vy'].astype(np.float32)
iv_vz = vox['iv_vz'].astype(np.float32); iv_vv = vox['iv_vv'].astype(np.float32)

# Re-normalize each channel to full 0–1 range (raw values are compressed)
ex_vn = (ex_vv - ex_vv.min()) / (ex_vv.max() - ex_vv.min() + 1e-8)
iv_vn = (iv_vv - iv_vv.min()) / (iv_vv.max() - iv_vv.min() + 1e-8)

# Cut bottom noise: keep only brighter voxels
EX_FLOOR = 0.3  # drop bottom 30% of ex-vivo (noise)
IV_FLOOR = 0.2
ex_keep = ex_vn > EX_FLOOR
iv_keep = iv_vn > IV_FLOOR
print(f"  Filtering: ex {ex_keep.sum():,}/{len(ex_vn):,}, iv {iv_keep.sum():,}/{len(iv_vn):,}")
ex_x_all, ex_y_all, ex_z_all = ex_vx[ex_keep], ex_vy[ex_keep], ex_vz[ex_keep]
iv_x_all, iv_y_all, iv_z_all = iv_vx[iv_keep], iv_vy[iv_keep], iv_vz[iv_keep]
ex_vn = (ex_vn[ex_keep] - EX_FLOOR) / (1 - EX_FLOOR)  # re-range to 0–1
iv_vn = (iv_vn[iv_keep] - IV_FLOOR) / (1 - IV_FLOOR)

# Gamma to boost mid-tones
ex_vn = np.power(ex_vn, 0.5)
iv_vn = np.power(iv_vn, 0.4)

# Centroid-align to ex-vivo (same as v4 viewer) — using filtered arrays
cx_c = (ex_x_all.max()+ex_x_all.min())*0.5
cy_c = (ex_y_all.max()+ex_y_all.min())*0.5
cz_c = (ex_z_all.max()+ex_z_all.min())*0.5
ex_x = ex_x_all-cx_c; ex_y = ex_y_all-cy_c; ex_z = ex_z_all-cz_c
iv_x = iv_x_all-cx_c; iv_y = iv_y_all-cy_c; iv_z = iv_z_all-cz_c

# In-vivo in its own centroid (for unregistered view)
iv_cx = (iv_x_all.max()+iv_x_all.min())*0.5
iv_cy = (iv_y_all.max()+iv_y_all.min())*0.5
iv_cz = (iv_z_all.max()+iv_z_all.min())*0.5
iv_x_native = iv_x_all - iv_cx
iv_y_native = iv_y_all - iv_cy
iv_z_native = iv_z_all - iv_cz

data_span = max(ex_x_all.max()-ex_x_all.min(), ex_y_all.max()-ex_y_all.min())
SCALE_FULL = int(W*0.82/data_span)
SCALE_HALF = int((W//2)*0.80/data_span)
print(f"  ex={len(ex_x):,}  iv={len(iv_x):,}  scale_full={SCALE_FULL}")

# ── Colours (BGR for cv2) ───────────────────────────────────────────────────
# EV: green, IV: red/hot
EX_COL = np.array([0, 1.0, 0], np.float32)       # pure green
IV_COL = np.array([0.1, 0.3, 1.0], np.float32)   # red-dominant with warm tint

# ── Gaussian splat kernel ───────────────────────────────────────────────────
SPLAT_R = 2  # radius in pixels → 5x5 patch
kernel = np.zeros((2*SPLAT_R+1, 2*SPLAT_R+1), np.float32)
for dy in range(-SPLAT_R, SPLAT_R+1):
    for dx in range(-SPLAT_R, SPLAT_R+1):
        kernel[dy+SPLAT_R, dx+SPLAT_R] = math.exp(-(dx*dx+dy*dy)/(2*1.2*1.2))

def render_one(x, y, z, v, rot_y, rot_x, alpha, col, cw, ch, sc, cx_pos, cy_pos):
    """Render point cloud with additive gaussian splatting."""
    canvas = np.zeros((ch, cw, 3), np.float32)
    if alpha < 0.005: return canvas

    cr, sr = math.cos(rot_y), math.sin(rot_y)
    rx = cr*x + sr*z; rz = -sr*x + cr*z
    cr2, sr2 = math.cos(rot_x), math.sin(rot_x)
    ry2 = cr2*y - sr2*rz; rz2 = sr2*y + cr2*rz
    px = (rx*sc + cx_pos).astype(np.int32)
    py = (ry2*sc + cy_pos).astype(np.int32)

    # Margin for splat kernel
    R = SPLAT_R
    mask = (px>=R)&(px<cw-R)&(py>=R)&(py<ch-R)
    px, py, vv, dd = px[mask], py[mask], v[mask], rz2[mask]

    # Depth sort: back-to-front for additive blending (render bright on top)
    order = np.argsort(-dd)
    px, py, vv = px[order], py[order], vv[order]

    # Additive splatting using vectorized approach
    # For speed, use single-pixel scatter + gaussian blur (equivalent to splatting)
    for c in range(3):
        np.add.at(canvas[:,:,c], (py, px), vv * alpha * col[c])

    # Gaussian blur to spread points (simulates splatting)
    canvas = cv2.GaussianBlur(canvas, (7, 7), 1.5)

    # Second pass: add brighter core for high-value points
    bright_mask = vv > 0.5
    if np.any(bright_mask):
        pxb, pyb, vvb = px[bright_mask], py[bright_mask], vv[bright_mask]
        for c in range(3):
            np.add.at(canvas[:,:,c], (pyb, pxb), vvb * alpha * col[c] * 0.5)
        canvas = cv2.GaussianBlur(canvas, (5, 5), 1.0)

    return canvas


# ── Panel constants ──────────────────────────────────────────────────────────
PANEL_W, PANEL_H = 840, 840
IV_CX = W//4; EV_CX = 3*W//4; CY = H//2
iv_panel = fit_into(iv_mip_u8, PANEL_W, PANEL_H)
ev_panel = fit_into(ev_mip_u8, PANEL_W, PANEL_H)

BASE_ROT_X = math.radians(10); BASE_ROT_Y = math.radians(5)

# ═════════════════════════════════════════════════════════════════════════════
# VIDEO
# ═════════════════════════════════════════════════════════════════════════════
vw = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

# ── Phase 1: 2D side-by-side (96fr, 4s) ─────────────────────────────────────
print("Phase 1: 2D side-by-side...")
for fi in range(96):
    frame = np.zeros((H, W, 3), np.uint8)
    t_split = ease(fi/40)
    iv_cx_now = int(W//2 + t_split*(IV_CX - W//2))
    ev_cx_now = int(W//2 + t_split*(EV_CX - W//2))
    place(frame, iv_panel, CY, iv_cx_now)
    if t_split > 0.1: place(frame, ev_panel, CY, ev_cx_now)
    a = min(1, t_split*3)
    cv2.putText(frame, 'IN VIVO', (iv_cx_now-PANEL_W//2+4, CY-PANEL_H//2-12),
                FONT, 0.42, tuple(int(v*a) for v in (40,60,240)), 1, cv2.LINE_AA)
    cv2.putText(frame, 'EX VIVO', (ev_cx_now-PANEL_W//2+4, CY-PANEL_H//2-12),
                FONT, 0.42, tuple(int(v*a) for v in (0,200,0)), 1, cv2.LINE_AA)
    vw.write(frame)

# ── Phase 2: 3D clouds emerge (120fr, 5s) ────────────────────────────────────
print("Phase 2: 3D clouds emerge...")
for fi in range(120):
    frame = np.zeros((H, W, 3), np.uint8)
    t_cloud = ease(fi/90)
    dim_a = max(0.08, 1.0 - ease(fi/100)*0.92)

    place(frame, np.clip(iv_panel.astype(np.float32)*dim_a, 0, 255).astype(np.uint8), CY, IV_CX)
    place(frame, np.clip(ev_panel.astype(np.float32)*dim_a, 0, 255).astype(np.uint8), CY, EV_CX)

    c_ex = render_one(ex_x, ex_y, ex_z, ex_vn,
                      BASE_ROT_Y, BASE_ROT_X, t_cloud, EX_COL,
                      W, H, SCALE_HALF, EV_CX, CY)
    c_iv = render_one(iv_x_native, iv_y_native, iv_z_native, iv_vn,
                      BASE_ROT_Y, BASE_ROT_X, t_cloud, IV_COL,
                      W, H, SCALE_HALF, IV_CX, CY)
    # Additive blend: clamp to 255
    cloud = np.clip((c_ex + c_iv)*255, 0, 255).astype(np.uint8)
    frame = cv2.add(frame, cloud)

    cv2.putText(frame, 'IN VIVO  3D', (IV_CX-PANEL_W//2+4, CY-PANEL_H//2-12),
                FONT, 0.42, tuple(int(v*min(1,t_cloud*2)) for v in (40,60,240)), 1, cv2.LINE_AA)
    cv2.putText(frame, 'EX VIVO  3D', (EV_CX-PANEL_W//2+4, CY-PANEL_H//2-12),
                FONT, 0.42, tuple(int(v*min(1,t_cloud*2)) for v in (0,200,0)), 1, cv2.LINE_AA)
    vw.write(frame)
    if fi % 30 == 0: print(f"  {fi}/120")

# ── Phase 3: IV slides to align with EV (120fr, 5s) ──────────────────────────
print("Phase 3: registration...")
for fi in range(120):
    frame = np.zeros((H, W, 3), np.uint8)
    t = ease(fi/100)

    iv_cx_now = int(IV_CX + t*(W//2 - IV_CX))
    ev_cx_now = int(EV_CX + t*(W//2 - EV_CX))
    scale_now = int(SCALE_HALF + t*(SCALE_FULL - SCALE_HALF))

    iv_x_now = iv_x_native*(1-t) + iv_x*t
    iv_y_now = iv_y_native*(1-t) + iv_y*t
    iv_z_now = iv_z_native*(1-t) + iv_z*t

    c_ex = render_one(ex_x, ex_y, ex_z, ex_vn,
                      BASE_ROT_Y, BASE_ROT_X, 1.0, EX_COL,
                      W, H, scale_now, ev_cx_now, CY)
    c_iv = render_one(iv_x_now, iv_y_now, iv_z_now, iv_vn,
                      BASE_ROT_Y, BASE_ROT_X, 1.0, IV_COL,
                      W, H, scale_now, iv_cx_now, CY)
    cloud = np.clip((c_ex + c_iv)*255, 0, 255).astype(np.uint8)
    frame[:] = cloud

    if t < 0.3:
        cv2.putText(frame, 'REGISTERING...', (W//2-80, 36), FONT, 0.50, (200,200,200), 1, cv2.LINE_AA)
    if t > 0.7:
        a2 = min(1, (t-0.7)/0.2)
        cv2.putText(frame, 'REGISTERED', (W//2-60, 36), FONT, 0.50,
                    tuple(int(v*a2) for v in (200,200,200)), 1, cv2.LINE_AA)
    cv2.circle(frame, (28,28), 7, (0,200,0), -1, cv2.LINE_AA)
    cv2.putText(frame, 'EX VIVO', (42,32), FONT, 0.38, (0,200,0), 1, cv2.LINE_AA)
    cv2.circle(frame, (28,52), 7, (40,60,240), -1, cv2.LINE_AA)
    cv2.putText(frame, 'IN VIVO', (42,56), FONT, 0.38, (40,60,240), 1, cv2.LINE_AA)
    vw.write(frame)
    if fi % 30 == 0: print(f"  {fi}/120")

# ── Phase 4: merged rotation (96fr, 4s) ──────────────────────────────────────
print("Phase 4: merged rotation...")
for fi in range(96):
    frame = np.zeros((H, W, 3), np.uint8)
    t_rot = fi/95
    rot_y = math.radians(5 + 30*math.sin(t_rot*math.pi))
    rot_x = BASE_ROT_X + math.radians(8*math.sin(t_rot*math.pi*0.5))

    c_ex = render_one(ex_x, ex_y, ex_z, ex_vn,
                      rot_y, rot_x, 1.0, EX_COL, W, H, SCALE_FULL, W//2, CY)
    c_iv = render_one(iv_x, iv_y, iv_z, iv_vn,
                      rot_y, rot_x, 1.0, IV_COL, W, H, SCALE_FULL, W//2, CY)
    cloud = np.clip((c_ex + c_iv)*255, 0, 255).astype(np.uint8)
    frame[:] = cloud

    cv2.circle(frame, (28,28), 7, (0,200,0), -1, cv2.LINE_AA)
    cv2.putText(frame, 'EX VIVO  CONFOCAL', (42,32), FONT, 0.38, (0,200,0), 1, cv2.LINE_AA)
    cv2.circle(frame, (28,52), 7, (40,60,240), -1, cv2.LINE_AA)
    cv2.putText(frame, 'IN VIVO  CALCIUM', (42,56), FONT, 0.38, (40,60,240), 1, cv2.LINE_AA)
    cv2.putText(frame, 'MULTIMODAL  3D  REGISTRATION', (W//2-160, H-42), FONT, 0.60,
                (200,200,200), 1, cv2.LINE_AA)
    vw.write(frame)
    if fi % 24 == 0: print(f"  {fi}/96")

vw.release()
print("Encoding...")
subprocess.run(['ffmpeg','-y','-i',TMP,
                '-vcodec','libx264','-pix_fmt','yuv420p','-crf','18','-preset','fast',
                OUT], check=True, capture_output=True)
os.remove(TMP)
total = 96+120+120+96
print(f"Done -> {OUT}  ({total}fr = {total/FPS:.0f}s)")
