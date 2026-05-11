#!/usr/bin/env python3
"""
Registration animation v4 — same as v3 but Scene 4 uses the actual
3D stitched ex-vivo + warped in-vivo point clouds from
viewer_warped_invivo_3d_v4.html (extracted to /tmp/viewer_v4_voxels.npz).

  Scene 1 : Calcium movie playing                          ( 192 fr,  8s)
  Scene 2 : Freeze → max-proj + dashed cell circles        (  96 fr,  4s)
  Scene 3 : Side-by-side registration + connecting lines   ( 144 fr,  6s)
  Scene 4 : 3D point cloud (ex-vivo green → in-vivo mag.)  ( 336 fr, 14s)
  Scene 5 : Cell strip (ex-vivo top / in-vivo bottom)      ( 288 fr, 12s)
  Scene 6 : Per-cell panels (3 × 216 fr)                   ( 648 fr, 27s)
  Scene 7 : Summary three-panel                            ( 192 fr,  8s)
  Total ≈ 79s @ 24 fps
"""
import numpy as np, cv2, tifffile, json, os, glob, math, subprocess

BASE    = '/Users/neurolab/neuroinformatics/margaret'
W, H    = 1920, 1080
FPS     = 24
TMP     = f'{BASE}/png_exports/registration_animation_v4_raw.mp4'
OUT     = f'{BASE}/png_exports/registration_animation_v4.mp4'
os.makedirs(f'{BASE}/png_exports', exist_ok=True)

FONT   = cv2.FONT_HERSHEY_SIMPLEX
BLACK  = (  0,   0,   0)
WHITE  = (255, 255, 255)
GRAY   = (160, 160, 160)
RED_C  = (  0,   0, 220)   # BGR

CELL_COLS = [             # BGR, one per cell
    (  0,   0, 220),  # red
    (  0, 110, 255),  # orange
    (  0, 220, 220),  # yellow
    ( 60, 200,  60),  # green
    (220, 200,   0),  # cyan
    (220,  80,   0),  # blue
]

IV_XY_UM = 0.6835
SKIP_TILES = {'row3_1', 'row3_5'}

# ─── helpers ────────────────────────────────────────────────────────────────
def norm_u8(img, lo=1, hi=99.5):
    v = img.ravel(); v = v[v>0]
    if len(v)<50: return np.zeros(img.shape, np.uint8)
    p1,p2 = np.percentile(v,[lo,hi])
    return np.clip((img.astype(np.float32)-p1)/max(p2-p1,1)*255,0,255).astype(np.uint8)

def ease(t): return float(0.5-0.5*math.cos(math.pi*max(0.,min(1.,t))))

def blend(a,b,t):
    t=max(0.,min(1.,t))
    return np.clip((1-t)*a.astype(np.float32)+t*b.astype(np.float32),0,255).astype(np.uint8)

def caption(frame, text, alpha=1.0, y_off=0):
    if alpha < 0.01: return
    ts, th = 0.72, 1
    (tw,_),_ = cv2.getTextSize(text, FONT, ts, th)
    x = (W-tw)//2; y = H-42+y_off
    col = tuple(int(v*alpha) for v in WHITE)
    cv2.putText(frame, text, (x,y), FONT, ts, col, th, cv2.LINE_AA)

def small_label(frame, text, y, x, col=GRAY, alpha=1.0):
    col2 = tuple(int(v*alpha) for v in col)
    cv2.putText(frame, text, (x,y), FONT, 0.38, col2, 1, cv2.LINE_AA)

def dashed_circle(img, cx, cy, r, col, dash_deg=14):
    for d in range(0, 360, dash_deg*2):
        a1,a2 = math.radians(d), math.radians(d+dash_deg)
        p1 = (int(cx+r*math.cos(a1)), int(cy+r*math.sin(a1)))
        p2 = (int(cx+r*math.cos(a2)), int(cy+r*math.sin(a2)))
        cv2.line(img, p1, p2, col, 2, cv2.LINE_AA)

def fit_into(img, tw, th):
    if img.ndim==2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h,w = img.shape[:2]; s = min(tw/w, th/h)
    nw,nh = int(w*s), int(h*s)
    rs = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LANCZOS4)
    c  = np.zeros((th,tw,3),np.uint8)
    yo,xo = (th-nh)//2, (tw-nw)//2
    c[yo:yo+nh,xo:xo+nw] = rs
    return c, s, yo, xo

def place(frame, img, cy, cx):
    ih,iw = img.shape[:2]
    y0,x0 = cy-ih//2, cx-iw//2
    fy0=max(0,y0); fy1=min(H,y0+ih)
    fx0=max(0,x0); fx1=min(W,x0+iw)
    sy0=fy0-y0; sx0=fx0-x0
    if fy1>fy0 and fx1>fx0:
        frame[fy0:fy1,fx0:fx1]=img[sy0:sy0+(fy1-fy0),sx0:sx0+(fx1-fx0)]

def zoom_crop(img, scale, cx=None, cy=None):
    h,w = img.shape[:2]
    if cx is None: cx=w//2
    if cy is None: cy=h//2
    nw,nh = int(w/scale), int(h/scale)
    x0 = max(0,min(w-nw, cx-nw//2))
    y0 = max(0,min(h-nh, cy-nh//2))
    crop = img[y0:y0+nh, x0:x0+nw]
    return cv2.resize(crop,(w,h),interpolation=cv2.INTER_LANCZOS4)

# ─── calcium trace renderer ──────────────────────────────────────────────────
def render_trace(trace, playhead_fi, tw, th, line_col=WHITE, head_col=RED_C):
    img = np.zeros((th,tw,3),np.uint8)
    n   = len(trace)
    t_norm = (trace - trace.min()) / max(trace.max()-trace.min(), 1)
    pts = []
    for i,v in enumerate(t_norm):
        x = int(i/(n-1)*(tw-4)+2)
        y = int((1-v)*(th-10)+5)
        pts.append((x,y))
    for i in range(len(pts)-1):
        cv2.line(img, pts[i], pts[i+1], (180,180,180), 1, cv2.LINE_AA)
    if 0 <= playhead_fi < n:
        px = pts[playhead_fi][0]
        cv2.line(img,(px,0),(px,th),head_col,1,cv2.LINE_AA)
        cv2.circle(img,(px,pts[playhead_fi][1]),3,head_col,-1,cv2.LINE_AA)
    return img

# ─── 3D point cloud renderer ─────────────────────────────────────────────────
# Ex-vivo: green  (BGR 0,200,0)
# In-vivo: magenta (BGR 200,0,200)
EX_COL = np.array([0, 200, 0],   np.float32)
IV_COL = np.array([200, 0, 200], np.float32)

def _rot_y(x, y, z, angle):
    """Rotate points around Y axis."""
    c, s = np.cos(angle), np.sin(angle)
    return c*x + s*z, y, -s*x + c*z

def _rot_x(x, y, z, angle):
    """Rotate points around X axis."""
    c, s = np.cos(angle), np.sin(angle)
    return x, c*y - s*z, s*y + c*z

def render_clouds(ex_x, ex_y, ex_z, ex_v,
                  iv_x, iv_y, iv_z, iv_v,
                  rot_y_angle, rot_x_angle,
                  ex_alpha, iv_alpha,
                  canvas_w, canvas_h, scale_px):
    """
    Orthographic projection of both point clouds onto a black canvas.
    Points are already centred at 0.  scale_px maps 1 unit → pixels.
    Uses depth-sorted painter's algorithm (near overwrites far).
    """
    canvas = np.zeros((canvas_h, canvas_w, 3), np.float32)
    cx = canvas_w // 2
    cy = canvas_h // 2

    def paint(x, y, z, v_norm, col_f32, alpha):
        if alpha < 0.005: return
        rx, ry, rz = _rot_y(x, y, z, rot_y_angle)
        rx, ry, rz = _rot_x(rx, ry, rz, rot_x_angle)
        px = (rx * scale_px + cx).astype(np.int32)
        py = (ry * scale_px + cy).astype(np.int32)
        mask = (px >= 0) & (px < canvas_w) & (py >= 0) & (py < canvas_h)
        px, py = px[mask], py[mask]
        v2 = v_norm[mask]
        rz2 = rz[mask]
        # Sort far → near so near points overwrite
        order = np.argsort(-rz2)
        px, py, v2 = px[order], py[order], v2[order]
        intensity = (v2 * alpha).astype(np.float32)
        canvas[py, px, 0] = np.maximum(canvas[py, px, 0], intensity * col_f32[0])
        canvas[py, px, 1] = np.maximum(canvas[py, px, 1], intensity * col_f32[1])
        canvas[py, px, 2] = np.maximum(canvas[py, px, 2], intensity * col_f32[2])

    # ex-vivo first (green, farther layer visually)
    if ex_alpha > 0.005:
        paint(ex_x, ex_y, ex_z, ex_v, EX_COL / 255.0, ex_alpha)
    # in-vivo on top (magenta)
    if iv_alpha > 0.005:
        paint(iv_x, iv_y, iv_z, iv_v, IV_COL / 255.0, iv_alpha)

    out = np.clip(canvas * 255, 0, 255).astype(np.uint8)
    # Small blur to smooth sparse pixels
    out = cv2.GaussianBlur(out, (3, 3), 0.8)
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading 3D voxel clouds from viewer_v4_voxels.npz…")
vox = np.load('/tmp/viewer_v4_voxels.npz')

ex_vx = vox['ex_vx'].astype(np.float32)
ex_vy = vox['ex_vy'].astype(np.float32)
ex_vz = vox['ex_vz'].astype(np.float32)
ex_vv = vox['ex_vv'].astype(np.float32)

iv_vx = vox['iv_vx'].astype(np.float32)
iv_vy = vox['iv_vy'].astype(np.float32)
iv_vz = vox['iv_vz'].astype(np.float32)
iv_vv = vox['iv_vv'].astype(np.float32)

# Normalize intensities to [0,1]
ex_vn = (ex_vv - ex_vv.min()) / (ex_vv.max() - ex_vv.min() + 1e-8)
iv_vn = (iv_vv - iv_vv.min()) / (iv_vv.max() - iv_vv.min() + 1e-8)

# Centre both clouds on the ex-vivo centroid (they were already aligned this way)
cx_c = (ex_vx.max() + ex_vx.min()) * 0.5
cy_c = (ex_vy.max() + ex_vy.min()) * 0.5
cz_c = (ex_vz.max() + ex_vz.min()) * 0.5

ex_x = ex_vx - cx_c; ex_y = ex_vy - cy_c; ex_z = ex_vz - cz_c
iv_x = iv_vx - cx_c; iv_y = iv_vy - cy_c; iv_z = iv_vz - cz_c

# Scale so the data fills ~80 % of canvas width
data_span = max(ex_vx.max()-ex_vx.min(), ex_vy.max()-ex_vy.min())
SCALE_PX  = int(W * 0.82 / data_span)
print(f"  ex: {len(ex_x)} pts  iv: {len(iv_x)} pts  scale_px={SCALE_PX}")

print("Loading JY306 stack…")
jy306     = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv, ny_iv, nx_iv = jy306.shape
jy306_mip_u8 = norm_u8(np.max(jy306,axis=0))
jy306_z3_u8  = norm_u8(jy306[3])

print("Loading calcium movie…")
cap = cv2.VideoCapture(f'{BASE}/png_exports/native_invivo/movie_warped_h264.mp4')
movie_frames=[]
while True:
    ret,frm=cap.read()
    if not ret: break
    movie_frames.append(cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY))
cap.release()
movie_frames = np.array(movie_frames, np.uint8)
n_movie = len(movie_frames)
movie_max_u8 = norm_u8(np.max(movie_frames.astype(np.float32),axis=0))
print(f"  {n_movie} frames {movie_frames.shape[1:]}")

print("Loading nd2 row2_1 tile for Scene 3…")
nd2_slices=[]
for zi in range(12):
    p=cv2.imread(f'{BASE}/png_exports/registration_video/row2_1/GFP_z{zi:03d}.png',cv2.IMREAD_UNCHANGED)
    if p is not None: nd2_slices.append(p.astype(np.float32))
nd2_mip_u8 = norm_u8(np.max(nd2_slices,axis=0))

print("Loading landmarks…")
lm27      = np.load(f'{BASE}/registration_video/landmarks_27_nd2_native.npz')
ev_nd2_27 = lm27['ev_nd2']
pcd_iv_27 = lm27['pcd_invivo_jy306']

print("Loading patch strip + cell info…")
patch_strip = cv2.imread(f'{BASE}/3d_viewer/patch_strip_v4.png')
with open(f'{BASE}/3d_viewer/cell_info_v4.json') as f:
    cell_info=[json.loads(x) if isinstance(x,str) else x for x in json.load(f)]
PATCH_SZ = 80

tile_lm_map={}
for lf in sorted(glob.glob(f'{BASE}/registration_video/landmarks_nd2_native_*.npz')):
    tile=os.path.basename(lf).replace('landmarks_nd2_native_','').replace('.npz','')
    if tile not in SKIP_TILES: tile_lm_map[tile]=lf
legacy=f'{BASE}/registration_video/landmarks_27_nd2_native.npz'
if os.path.exists(legacy) and 'row2_1' not in tile_lm_map:
    tile_lm_map['row2_1']=legacy
all_lm_jy=[]
for tile in sorted(tile_lm_map):
    d=np.load(tile_lm_map[tile])
    for row in d['pcd_invivo_jy306']:
        all_lm_jy.append((int(round(row[1])),int(round(row[2]))))

strip_cells = [ci for ci,info in enumerate(cell_info) if 1<=info[3]<=5][:6]
dive_cells  = strip_cells[:3]
print(f"  Strip cells: {strip_cells}  Dive cells: {dive_cells}")

print("Computing calcium traces…")
traces={}
for ci in dive_cells:
    lm_y,lm_x = all_lm_jy[ci]
    r=8
    tr=[]
    for frm in movie_frames:
        y0=max(0,lm_y-r); y1=min(frm.shape[0],lm_y+r)
        x0=max(0,lm_x-r); x1=min(frm.shape[1],lm_x+r)
        tr.append(float(frm[y0:y1,x0:x1].mean()))
    traces[ci]=np.array(tr)

# Pre-crop cell strip patches
CELL_W = 130; CELL_H = 130
N_STRIP = len(strip_cells)
STRIP_W = N_STRIP*CELL_W + (N_STRIP-1)*10
STRIP_X0 = (W-STRIP_W)//2
ROW1_CY  = H//2 - CELL_H//2 - 30
ROW2_CY  = H//2 + CELL_H//2 + 30
ev_crops, iv_crops = [], []
for ci in strip_cells:
    ep = patch_strip[ci*PATCH_SZ:(ci+1)*PATCH_SZ, 0:PATCH_SZ]
    ev_crops.append(cv2.resize(ep,(CELL_W,CELL_H),interpolation=cv2.INTER_LANCZOS4))
    lm_y,lm_x = all_lm_jy[ci]
    r = 55
    y0=max(0,lm_y-r); y1=min(movie_max_u8.shape[0],lm_y+r)
    x0=max(0,lm_x-r); x1=min(movie_max_u8.shape[1],lm_x+r)
    crop = movie_max_u8[y0:y1,x0:x1]
    crop_sq = np.zeros((r*2,r*2),np.uint8)
    crop_sq[:crop.shape[0],:crop.shape[1]] = crop
    iv_crops.append(cv2.resize(crop_sq,(CELL_W,CELL_H),interpolation=cv2.INTER_LANCZOS4))

# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO WRITER
# ═══════════════════════════════════════════════════════════════════════════════
vw = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W,H))

# ─────────────────────────────────────────────────────────────────────────────
# SCENE 1  Calcium movie  (192 fr, 8s)
# ─────────────────────────────────────────────────────────────────────────────
print("Scene 1: calcium movie…")
MOV_STEP = max(1, n_movie // 192)
for fi in range(192):
    frame = np.zeros((H,W,3),np.uint8)
    idx   = (fi * MOV_STEP) % n_movie
    gray  = norm_u8(movie_frames[idx].astype(np.float32))
    zoom  = 1.15 - 0.15*(fi/191)
    zoomed = zoom_crop(gray, zoom)
    sq,_,_,_ = fit_into(zoomed, 860, 860)
    place(frame, sq, H//2, W//2)
    if fi >= 80:
        caption(frame,'IN VIVO  TWO-PHOTON  CALCIUM  IMAGING', alpha=ease((fi-80)/30))
    vw.write(frame)

# ─────────────────────────────────────────────────────────────────────────────
# SCENE 2  Freeze → max-proj + cell circles  (96 fr, 4s)
# ─────────────────────────────────────────────────────────────────────────────
print("Scene 2: freeze + circles…")
last_gray = norm_u8(movie_frames[(191*MOV_STEP)%n_movie].astype(np.float32))
circle_cells = [0,5,10,15,20]
for fi in range(96):
    t_max  = ease(fi/40)
    frame  = np.zeros((H,W,3),np.uint8)
    img_b  = blend(last_gray, movie_max_u8, t_max)
    sq,scale,yo,xo = fit_into(img_b, 860, 860)
    cy0,cx0 = H//2-430, W//2-430
    place(frame, sq, H//2, W//2)
    for k,ci in enumerate(circle_cells):
        a_circle = ease((fi - k*12 - 20) / 15)
        if a_circle <= 0: continue
        ry = int(pcd_iv_27[ci,1]*scale)+yo+cy0
        rx = int(pcd_iv_27[ci,2]*scale)+xo+cx0
        col = tuple(int(v*a_circle) for v in WHITE)
        dashed_circle(frame, rx, ry, 18, col)
    a_out = 1-ease(fi/20)
    a_in  = ease((fi-30)/25)
    caption(frame,'IN VIVO  TWO-PHOTON  CALCIUM  IMAGING', alpha=float(max(0,a_out)))
    caption(frame,'NEURONS  RECORDED  DURING  BEHAVIOUR',  alpha=float(max(0,a_in)))
    vw.write(frame)

# ─────────────────────────────────────────────────────────────────────────────
# SCENE 3  Side-by-side registration  (144 fr, 6s)
# ─────────────────────────────────────────────────────────────────────────────
print("Scene 3: side-by-side registration…")
SIDE_W, SIDE_H = 780, 780
IV_CX,  IV_CY  = W//4,   H//2
EV_CX,  EV_CY  = 3*W//4, H//2
HALF_W, HALF_H = 820, 820

iv_half, iv_sc2, iv_yo2, iv_xo2 = fit_into(movie_max_u8, HALF_W, HALF_H)
ev_half, ev_sc2, ev_yo2, ev_xo2 = fit_into(nd2_mip_u8,   HALF_W, HALF_H)

for fi in range(144):
    t_split  = ease(fi/45)
    t_lines  = ease((fi-50)/50)
    frame    = np.zeros((H,W,3),np.uint8)

    iv_cx = int(W//2 + t_split*(IV_CX - W//2))
    ev_cx = int(W//2 + t_split*(EV_CX - W//2))

    place(frame, iv_half, H//2, iv_cx)
    if t_split > 0.15:
        place(frame, ev_half, H//2, ev_cx)

    small_label(frame,'2P GFP',         H//2-HALF_H//2-18, iv_cx-HALF_W//2, GRAY, min(1,t_split*3))
    small_label(frame,'CONFOCAL GFP',   H//2-HALF_H//2-18, ev_cx-HALF_W//2, GRAY, min(1,(t_split-0.3)*3))

    if t_lines > 0:
        iv_tl_x = iv_cx-HALF_W//2; iv_tl_y = H//2-HALF_H//2
        ev_tl_x = ev_cx-HALF_W//2; ev_tl_y = H//2-HALF_H//2
        n_lines  = int(t_lines*7)
        for k in range(n_lines):
            ry_iv = int(pcd_iv_27[k,1]*iv_sc2)+iv_yo2+iv_tl_y
            rx_iv = int(pcd_iv_27[k,2]*iv_sc2)+iv_xo2+iv_tl_x
            rx_ev = int(ev_nd2_27[k,0]*ev_sc2)+ev_xo2+ev_tl_x
            ry_ev = int(ev_nd2_27[k,1]*ev_sc2)+ev_yo2+ev_tl_y
            a_k   = min(1.0, t_lines*7-k)
            col   = tuple(int(v*a_k*0.7) for v in WHITE)
            cv2.circle(frame,(rx_iv,ry_iv),5,tuple(int(v*a_k) for v in WHITE),-1,cv2.LINE_AA)
            cv2.circle(frame,(rx_ev,ry_ev),5,tuple(int(v*a_k) for v in WHITE),-1,cv2.LINE_AA)
            cv2.line(frame,(rx_iv,ry_iv),(rx_ev,ry_ev),col,1,cv2.LINE_AA)

    caption(frame,'NEURONS  RECORDED  DURING  BEHAVIOUR', alpha=max(0,1-ease(fi/15)))
    caption(frame,'RE-IDENTIFIED  IN  EX-VIVO  TISSUE',   alpha=ease((fi-20)/25))
    vw.write(frame)

# ─────────────────────────────────────────────────────────────────────────────
# SCENE 4  3D Point Cloud  (336 fr, 14s)
#   0–72    fr (3s) : ex-vivo green cloud fades in, slight tilt begins
#   72–144  fr (3s) : in-vivo magenta cloud fades in
#   144–336 fr (8s) : both clouds, slow Y-axis rotation ±30° + slight X tilt
#                     + zoom/pan to highlight overlap
# ─────────────────────────────────────────────────────────────────────────────
print("Scene 4: 3D point cloud…")

# Baseline rotation: tilt X by 10° to show depth (looking slightly down)
BASE_ROT_X = math.radians(10)

for fi in range(336):
    frame = np.zeros((H,W,3),np.uint8)

    if fi < 72:
        # Ex-vivo fades in; gentle Y rotation starts
        ex_a   = ease(fi / 55)
        iv_a   = 0.0
        rot_y  = math.radians(fi / 71 * 15)       # 0° → 15°
        rot_x  = BASE_ROT_X * ease(fi / 71)
        zoom_f = 1.0 + 0.1 * ease(fi / 71)        # slight zoom in
    elif fi < 144:
        # In-vivo magenta fades in
        t2     = (fi - 72) / 71
        ex_a   = 1.0
        iv_a   = ease(t2)
        rot_y  = math.radians(15 + t2 * 10)       # 15° → 25°
        rot_x  = BASE_ROT_X
        zoom_f = 1.1
    else:
        # Both visible; slow continuous rotation, back and forth
        t3     = (fi - 144) / 191
        ex_a   = 1.0
        iv_a   = 1.0
        # Oscillate Y between 25° and -5° (sweeps through 30°)
        rot_y  = math.radians(25 - 30 * ease(t3))
        rot_x  = BASE_ROT_X + math.radians(5 * math.sin(t3 * math.pi))
        zoom_f = 1.1 + 0.3 * ease(t3 * 0.8)      # slowly zoom in

    # Render clouds into a sub-canvas then zoom-centre onto frame
    cloud_w = int(W / zoom_f)
    cloud_h = int(H / zoom_f)
    # Clamp to canvas
    cloud_w = min(cloud_w, W); cloud_h = min(cloud_h, H)
    scale_adj = int(SCALE_PX / zoom_f)

    cloud = render_clouds(
        ex_x, ex_y, ex_z, ex_vn,
        iv_x, iv_y, iv_z, iv_vn,
        rot_y, rot_x,
        ex_a, iv_a,
        cloud_w, cloud_h, scale_adj
    )

    # Paste into full frame (centred)
    x0 = (W - cloud_w) // 2; y0 = (H - cloud_h) // 2
    frame[y0:y0+cloud_h, x0:x0+cloud_w] = cloud

    # Legend dots (top-left)
    if ex_a > 0.05:
        a_lab = min(1.0, ex_a)
        cv2.circle(frame, (28,28), 7, tuple(int(v*a_lab) for v in EX_COL), -1, cv2.LINE_AA)
        small_label(frame, 'EX VIVO  CONFOCAL', 32, 42, tuple(int(v*a_lab) for v in EX_COL), a_lab)
    if iv_a > 0.05:
        a_lab = min(1.0, iv_a)
        cv2.circle(frame, (28,52), 7, tuple(int(v*a_lab) for v in IV_COL), -1, cv2.LINE_AA)
        small_label(frame, 'IN VIVO  CALCIUM', 56, 42, tuple(int(v*a_lab) for v in IV_COL), a_lab)

    caption(frame,'RE-IDENTIFIED  IN  EX-VIVO  TISSUE',  alpha=max(0,1-ease(fi/20)))
    caption(frame,'MULTIMODAL  CELL  MATCHING',           alpha=ease((fi-15)/25))
    vw.write(frame)
    if fi % 48 == 0:
        print(f"  frame {fi}/336")

# ─────────────────────────────────────────────────────────────────────────────
# SCENE 5  Cell strip  (288 fr, 12s)
# ─────────────────────────────────────────────────────────────────────────────
print("Scene 5: cell strip…")
for fi in range(288):
    frame = np.zeros((H,W,3),np.uint8)
    for k,ci in enumerate(strip_cells):
        appear_t = ease((fi - k*28) / 22)
        if appear_t <= 0: continue
        x0 = STRIP_X0 + k*(CELL_W+10)
        col = CELL_COLS[k]

        ev_img = np.clip(ev_crops[k].astype(np.float32)*appear_t,0,255).astype(np.uint8)
        frame[ROW1_CY-CELL_H//2 : ROW1_CY+CELL_H//2,
              x0 : x0+CELL_W] = ev_img
        cv2.rectangle(frame,(x0,ROW1_CY-CELL_H//2),(x0+CELL_W-1,ROW1_CY+CELL_H//2-1),
                      tuple(int(v*appear_t) for v in col),2)

        iv_img = np.clip(cv2.cvtColor(iv_crops[k],cv2.COLOR_GRAY2BGR).astype(np.float32)*appear_t,
                         0,255).astype(np.uint8)
        frame[ROW2_CY-CELL_H//2 : ROW2_CY+CELL_H//2,
              x0 : x0+CELL_W] = iv_img
        cv2.rectangle(frame,(x0,ROW2_CY-CELL_H//2),(x0+CELL_W-1,ROW2_CY+CELL_H//2-1),
                      tuple(int(v*appear_t*0.6) for v in col),2)

        cv2.putText(frame,str(k+1),(x0+4,ROW1_CY-CELL_H//2-6),
                    FONT,0.5,tuple(int(v*appear_t) for v in col),1,cv2.LINE_AA)

    all_appeared = ease((fi - (N_STRIP-1)*28 - 10) / 20)
    small_label(frame,'EX VIVO  CONFOCAL', ROW1_CY-4, STRIP_X0-160, WHITE, all_appeared)
    small_label(frame,'IN VIVO  CALCIUM',  ROW2_CY-4, STRIP_X0-160, WHITE, all_appeared)
    caption(frame,'MULTIMODAL  CELL  MATCHING', alpha=max(0,1-ease(fi/20)))
    caption(frame,'MATCHED  NEURONS',            alpha=ease((fi - (N_STRIP-1)*28)/25)*all_appeared)
    vw.write(frame)

# ─────────────────────────────────────────────────────────────────────────────
# SCENE 6  Per-cell deep dive  (3 × 216 fr)
# ─────────────────────────────────────────────────────────────────────────────
print("Scene 6: per-cell panels…")
LEFT_W  = W//2 - 20
FIELD_H = H//2 - 10
TRACE_H = H//2 - 80
PATCH_DISP = 280

iv_mip_bgr = cv2.cvtColor(jy306_mip_u8, cv2.COLOR_GRAY2BGR)

for cell_idx, ci in enumerate(dive_cells):
    info     = cell_info[ci]
    lm_y,lm_x = all_lm_jy[ci]
    trace    = traces[ci]
    col      = CELL_COLS[cell_idx]

    field_rs, fsc, fyo, fxo = fit_into(iv_mip_bgr, LEFT_W, FIELD_H)
    dot_rx = int(lm_x*fsc)+fxo
    dot_ry = int(lm_y*fsc)+fyo

    ev_p = cv2.resize(patch_strip[ci*PATCH_SZ:(ci+1)*PATCH_SZ, 0:PATCH_SZ],
                       (PATCH_DISP,PATCH_DISP), interpolation=cv2.INTER_LANCZOS4)
    iv_p = cv2.resize(patch_strip[ci*PATCH_SZ:(ci+1)*PATCH_SZ, PATCH_SZ*4:PATCH_SZ*5],
                       (PATCH_DISP,PATCH_DISP), interpolation=cv2.INTER_LANCZOS4)

    for fi in range(216):
        frame = np.zeros((H,W,3),np.uint8)
        a_in  = ease(fi/20)

        left_img = field_rs.copy()
        pulse = int(14 + 5*math.sin(fi*0.18))
        dashed_circle(left_img, dot_rx, dot_ry, pulse,
                       tuple(int(v*a_in) for v in col))
        frame[0:FIELD_H, 0:LEFT_W] = left_img

        movie_fi = fi % n_movie
        tr_img = render_trace(trace, movie_fi, LEFT_W, TRACE_H)
        frame[FIELD_H+10:FIELD_H+10+TRACE_H, 0:LEFT_W] = \
            np.clip(tr_img.astype(np.float32)*a_in,0,255).astype(np.uint8)
        small_label(frame,'CALCIUM  ACTIVITY  TRACE', FIELD_H+14, 8, GRAY, a_in)

        ey0 = (H - 2*PATCH_DISP - 30) // 2
        frame[ey0:ey0+PATCH_DISP, LEFT_W+40:LEFT_W+40+PATCH_DISP] = \
            np.clip(ev_p.astype(np.float32)*a_in,0,255).astype(np.uint8)
        cv2.rectangle(frame,(LEFT_W+40, ey0),(LEFT_W+40+PATCH_DISP-1, ey0+PATCH_DISP-1),
                      tuple(int(v*a_in) for v in WHITE), 1)
        small_label(frame,'EX VIVO  CONFOCAL', ey0-10, LEFT_W+42, GRAY, a_in)

        ch = PATCH_DISP//2
        alpha_cross = int(120*a_in)
        cv2.line(frame,(LEFT_W+40+ch-12,ey0+ch),(LEFT_W+40+ch+12,ey0+ch),(alpha_cross,)*3,1)
        cv2.line(frame,(LEFT_W+40+ch,ey0+ch-12),(LEFT_W+40+ch,ey0+ch+12),(alpha_cross,)*3,1)

        iy0 = ey0 + PATCH_DISP + 30
        frame[iy0:iy0+PATCH_DISP, LEFT_W+40:LEFT_W+40+PATCH_DISP] = \
            np.clip(iv_p.astype(np.float32)*a_in,0,255).astype(np.uint8)
        cv2.rectangle(frame,(LEFT_W+40, iy0),(LEFT_W+40+PATCH_DISP-1, iy0+PATCH_DISP-1),
                      tuple(int(v*a_in*0.7) for v in WHITE), 1)
        small_label(frame,'IN VIVO  TWO-PHOTON', iy0-10, LEFT_W+42, GRAY, a_in)

        cv2.line(frame,(LEFT_W+40+ch-12,iy0+ch),(LEFT_W+40+ch+12,iy0+ch),(alpha_cross,)*3,1)
        cv2.line(frame,(LEFT_W+40+ch,iy0+ch-12),(LEFT_W+40+ch,iy0+ch+12),(alpha_cross,)*3,1)

        cv2.line(frame,(LEFT_W+20,30),(LEFT_W+20,H-30),(40,40,40),1)
        cv2.putText(frame,f'CELL  {cell_idx+1}',(14,32),FONT,0.55,
                    tuple(int(v*a_in) for v in col),1,cv2.LINE_AA)
        caption(frame,'FUNCTIONAL  /  STRUCTURAL  READOUT', alpha=a_in)
        vw.write(frame)

    for fi in range(12):
        vw.write(np.zeros((H,W,3),np.uint8))
    print(f"  Cell {cell_idx+1} done")

# ─────────────────────────────────────────────────────────────────────────────
# SCENE 7  Summary  (192 fr, 8s)
# ─────────────────────────────────────────────────────────────────────────────
print("Scene 7: summary…")
P1_W,P1_H = 480,480
P2_W,P2_H = 560,400
P3_W,P3_H = 380,380
CX1,CX2,CX3 = 280, W//2, W-280
CY_MID = H//2

# Pre-render cloud still for summary centre panel (rotation 20°, tilt 12°)
print("  Pre-rendering summary cloud still…")
cloud_still = render_clouds(
    ex_x, ex_y, ex_z, ex_vn,
    iv_x, iv_y, iv_z, iv_vn,
    rot_y_angle=math.radians(20), rot_x_angle=math.radians(12),
    ex_alpha=1.0, iv_alpha=1.0,
    canvas_w=P2_W, canvas_h=P2_H,
    scale_px=int(SCALE_PX * P2_W / W)
)

for fi in range(192):
    frame = np.zeros((H,W,3),np.uint8)

    # Left: calcium movie looping
    a1 = ease(fi/25)
    if a1 > 0:
        midx = (fi*3)%n_movie
        sq1,_,_,_ = fit_into(norm_u8(movie_frames[midx].astype(np.float32)), P1_W, P1_H)
        place(frame, np.clip(sq1.astype(np.float32)*a1,0,255).astype(np.uint8), CY_MID, CX1)
        small_label(frame,'IN VIVO  CALCIUM', CY_MID+P1_H//2+18, CX1-80, GRAY, a1)

    # Centre: 3D cloud still
    a2 = ease((fi-20)/25)
    if a2 > 0:
        place(frame, np.clip(cloud_still.astype(np.float32)*a2,0,255).astype(np.uint8), CY_MID, CX2)
        small_label(frame,'3D  REGISTRATION', CY_MID+P2_H//2+18, CX2-70, GRAY, a2)

    # Right: cell strip (3 pairs)
    a3 = ease((fi-40)/25)
    if a3 > 0:
        CS = 80; gap=8; pairs=3
        sx0 = CX3 - (pairs*(CS+gap))//2
        for k in range(pairs):
            xk  = sx0 + k*(CS+gap)
            ep  = cv2.resize(ev_crops[k],(CS,CS))
            ivp = cv2.cvtColor(cv2.resize(iv_crops[k],(CS,CS)),cv2.COLOR_GRAY2BGR)
            yk1 = CY_MID-CS-gap//2; yk2 = CY_MID+gap//2
            frame[yk1:yk1+CS,xk:xk+CS]=np.clip(ep.astype(np.float32)*a3,0,255).astype(np.uint8)
            frame[yk2:yk2+CS,xk:xk+CS]=np.clip(ivp.astype(np.float32)*a3,0,255).astype(np.uint8)
            cv2.rectangle(frame,(xk,yk1),(xk+CS-1,yk1+CS-1),
                           tuple(int(v*a3*0.7) for v in CELL_COLS[k]),1)
        small_label(frame,'MATCHED  NEURONS', CY_MID+CS+gap+18, sx0, GRAY, a3)

    a_title = ease((fi-60)/30)
    if a_title > 0:
        ts,th=0.62,1
        t1='MULTIMODAL  REGISTRATION'
        (tw,_),_=cv2.getTextSize(t1,FONT,ts,th)
        cv2.putText(frame,t1,((W-tw)//2,58),FONT,ts,
                    tuple(int(v*a_title) for v in WHITE),th,cv2.LINE_AA)
        t2='IN VIVO  ·  EX VIVO  ·  CALCIUM  ·  CONFOCAL'
        (tw2,_),_=cv2.getTextSize(t2,FONT,0.42,1)
        cv2.putText(frame,t2,((W-tw2)//2,86),FONT,0.42,
                    tuple(int(v*a_title*0.6) for v in WHITE),1,cv2.LINE_AA)

    caption(frame,'JY306  MOUSE  HIPPOCAMPUS', alpha=ease((fi-80)/30))
    vw.write(frame)

# Final black hold
for _ in range(24):
    vw.write(np.zeros((H,W,3),np.uint8))

vw.release()
print(f"Raw: {TMP}")

print("Re-encoding H.264…")
subprocess.run(['ffmpeg','-y','-i',TMP,'-vcodec','libx264','-pix_fmt','yuv420p',
                '-crf','18','-preset','fast',OUT],check=True)
total_fr = 192+96+144+336+288+3*216+3*12+192+24
print(f"\n✓  {OUT}  ({total_fr/FPS:.0f}s @ {FPS}fps)")
