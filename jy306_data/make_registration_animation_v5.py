#!/usr/bin/env python3
"""
Registration animation v5 — narrative:
  S1  ( 72fr,  3s) : Calcium movie playing
  S2  ( 48fr,  2s) : Freeze → max projection
  S3  ( 96fr,  4s) : 2D side-by-side  JY306 MIP (left) | nd2 MIP (right)
  S4  (144fr,  6s) : 3D point clouds emerge ON TOP of 2D panels
                     (magenta in-vivo cloud grows over left panel,
                      green ex-vivo cloud grows over right panel)
  S5  (168fr,  7s) : Both clouds slide to centre → registered, rotate to show depth
  S6  (192fr,  8s) : 8-cell strip (4 cols × 4 rows: EV/IV for cells 1-4, then 5-8)
  S7  (192fr,  8s) : Per-cell deep-dive  Cell 1  (calcium video | EV | IV-warp)
  S8  (192fr,  8s) : Per-cell deep-dive  Cell 2
  S9  (192fr,  8s) : Per-cell deep-dive  Cell 3
  Total ≈ 54s @ 24fps

QC frames → png_exports/registration_animation_v5_qc/
"""
import numpy as np, cv2, tifffile, json, os, glob, math, subprocess

BASE = '/Users/neurolab/neuroinformatics/margaret'
W, H = 1920, 1080
FPS  = 24
TMP  = f'{BASE}/png_exports/registration_animation_v5_raw.mp4'
OUT  = f'{BASE}/png_exports/registration_animation_v5.mp4'
QC   = f'{BASE}/png_exports/registration_animation_v5_qc'
os.makedirs(f'{BASE}/png_exports', exist_ok=True)
os.makedirs(QC, exist_ok=True)

FONT   = cv2.FONT_HERSHEY_SIMPLEX
WHITE  = (255, 255, 255)
GRAY   = (160, 160, 160)
EX_COL = np.array([0, 200, 0],   np.float32)   # green  (BGR)
IV_COL = np.array([200, 0, 200], np.float32)   # magenta (BGR)
CELL_COLS = [
    (  0,   0, 220), (  0, 110, 255), (  0, 220, 220),
    ( 60, 200,  60), (220, 200,   0), (220,  80,   0),
    (180,   0, 180), (255, 255,   0),
]
SKIP_TILES = {'row3_1', 'row3_5'}

# ─── helpers ─────────────────────────────────────────────────────────────────
def norm_u8(img, lo=1, hi=99.5):
    v = img.ravel(); v = v[v>0]
    if len(v)<50: return np.zeros(img.shape, np.uint8)
    p1,p2 = np.percentile(v,[lo,hi])
    return np.clip((img.astype(np.float32)-p1)/max(p2-p1,1)*255,0,255).astype(np.uint8)

def ease(t): return float(0.5-0.5*math.cos(math.pi*max(0.,min(1.,t))))

def blend(a, b, t):
    t = max(0., min(1., t))
    return np.clip((1-t)*a.astype(np.float32)+t*b.astype(np.float32),0,255).astype(np.uint8)

def caption(frame, text, alpha=1.0):
    if alpha < 0.01: return
    (tw,_),_ = cv2.getTextSize(text, FONT, 0.72, 1)
    cv2.putText(frame, text, ((W-tw)//2, H-42), FONT, 0.72,
                tuple(int(v*alpha) for v in WHITE), 1, cv2.LINE_AA)

def small_label(frame, text, y, x, col=GRAY, alpha=1.0, scale=0.40):
    cv2.putText(frame, text, (x, y), FONT, scale,
                tuple(int(v*alpha) for v in col), 1, cv2.LINE_AA)

def fit_into(img, tw, th):
    if img.ndim==2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h,w = img.shape[:2]; s = min(tw/w, th/h)
    nw,nh = int(w*s),int(h*s)
    rs = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LANCZOS4)
    c = np.zeros((th,tw,3),np.uint8)
    yo,xo = (th-nh)//2,(tw-nw)//2
    c[yo:yo+nh,xo:xo+nw] = rs
    return c

def place(frame, img, cy, cx):
    ih,iw = img.shape[:2]
    y0,x0 = cy-ih//2, cx-iw//2
    fy0=max(0,y0); fy1=min(H,y0+ih)
    fx0=max(0,x0); fx1=min(W,x0+iw)
    if fy1>fy0 and fx1>fx0:
        frame[fy0:fy1,fx0:fx1]=img[fy0-y0:fy1-y0,fx0-x0:fx1-x0]

def save_qc(frame, name):
    p = f'{QC}/{name}.png'
    cv2.imwrite(p, frame)
    print(f'  QC → {p}')

# ─── 3D point cloud renderer ─────────────────────────────────────────────────
def render_one_cloud(x, y, z, v, rot_y, rot_x, alpha, col_f32,
                     canvas_w, canvas_h, scale_px, cx, cy):
    """Render one cloud onto a float32 canvas, return canvas."""
    canvas = np.zeros((canvas_h, canvas_w, 3), np.float32)
    if alpha < 0.005: return canvas
    c_y, s_y = np.cos(rot_y), np.sin(rot_y)
    rx = c_y*x + s_y*z;  rz = -s_y*x + c_y*z
    c_x, s_x = np.cos(rot_x), np.sin(rot_x)
    ry2 = c_x*y - s_x*rz;  rz2 = s_x*y + c_x*rz
    px = (rx*scale_px + cx).astype(np.int32)
    py = (ry2*scale_px + cy).astype(np.int32)
    mask = (px>=0)&(px<canvas_w)&(py>=0)&(py<canvas_h)
    px,py,v2,d = px[mask],py[mask],v[mask],rz2[mask]
    order = np.argsort(-d)
    px,py,v2 = px[order],py[order],v2[order]
    iv_ = (v2*alpha).astype(np.float32)
    np.maximum(canvas[py,px,0], iv_*col_f32[0], out=canvas[py,px,0])
    np.maximum(canvas[py,px,1], iv_*col_f32[1], out=canvas[py,px,1])
    np.maximum(canvas[py,px,2], iv_*col_f32[2], out=canvas[py,px,2])
    return canvas

def render_clouds(ex_x,ex_y,ex_z,ex_v, iv_x,iv_y,iv_z,iv_v,
                  rot_y, rot_x, ex_alpha, iv_alpha,
                  canvas_w, canvas_h, scale_px,
                  ex_cx=None, ex_cy=None, iv_cx=None, iv_cy=None):
    """Composite ex-vivo (green) + in-vivo (magenta) point clouds.
    ex_cx/iv_cx allow positioning each cloud at a different screen centre."""
    if ex_cx is None: ex_cx = canvas_w//2
    if ex_cy is None: ex_cy = canvas_h//2
    if iv_cx is None: iv_cx = canvas_w//2
    if iv_cy is None: iv_cy = canvas_h//2
    canvas = np.zeros((canvas_h, canvas_w, 3), np.float32)
    if ex_alpha > 0.005:
        c = render_one_cloud(ex_x,ex_y,ex_z,ex_v,rot_y,rot_x,ex_alpha,
                             EX_COL/255.,canvas_w,canvas_h,scale_px,ex_cx,ex_cy)
        np.maximum(canvas, c, out=canvas)
    if iv_alpha > 0.005:
        c = render_one_cloud(iv_x,iv_y,iv_z,iv_v,rot_y,rot_x,iv_alpha,
                             IV_COL/255.,canvas_w,canvas_h,scale_px,iv_cx,iv_cy)
        np.maximum(canvas, c, out=canvas)
    out = np.clip(canvas*255, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(out, (3,3), 0.8)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
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
movie_max = norm_u8(np.max(movie_frames.astype(np.float32),axis=0))
print(f"  {n_movie} frames {movie_frames.shape[1:]}")

print("Loading JY306 stack…")
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
jy306_mip_u8 = norm_u8(np.max(jy306,axis=0))

print("Loading nd2 row2_1 tile…")
nd2_slices=[]
for zi in range(12):
    p=cv2.imread(f'{BASE}/png_exports/registration_video/row2_1/GFP_z{zi:03d}.png',cv2.IMREAD_UNCHANGED)
    if p is not None: nd2_slices.append(p.astype(np.float32))
nd2_mip_full = norm_u8(np.max(nd2_slices,axis=0))
nd2_crop_u8  = nd2_mip_full[1235:2856, 688:2377]   # region matching JY306 footprint

print("Loading 3D voxel clouds…")
vox = np.load('/tmp/viewer_v4_voxels.npz')
ex_vx=vox['ex_vx'].astype(np.float32); ex_vy=vox['ex_vy'].astype(np.float32)
ex_vz=vox['ex_vz'].astype(np.float32); ex_vv=vox['ex_vv'].astype(np.float32)
iv_vx=vox['iv_vx'].astype(np.float32); iv_vy=vox['iv_vy'].astype(np.float32)
iv_vz=vox['iv_vz'].astype(np.float32); iv_vv=vox['iv_vv'].astype(np.float32)
ex_vn = (ex_vv-ex_vv.min())/(ex_vv.max()-ex_vv.min()+1e-8)
iv_vn = (iv_vv-iv_vv.min())/(iv_vv.max()-iv_vv.min()+1e-8)
cx_c=(ex_vx.max()+ex_vx.min())*0.5; cy_c=(ex_vy.max()+ex_vy.min())*0.5
cz_c=(ex_vz.max()+ex_vz.min())*0.5
ex_x=ex_vx-cx_c; ex_y=ex_vy-cy_c; ex_z=ex_vz-cz_c
iv_x=iv_vx-cx_c; iv_y=iv_vy-cy_c; iv_z=iv_vz-cz_c
data_span=max(ex_vx.max()-ex_vx.min(), ex_vy.max()-ex_vy.min())
# Scale for full-width display and for half-width panels
SCALE_FULL = int(W * 0.82 / data_span)
SCALE_HALF = int((W//2) * 0.80 / data_span)
print(f"  ex={len(ex_x)} iv={len(iv_x)} scale_full={SCALE_FULL} scale_half={SCALE_HALF}")

print("Loading landmarks…")
lm27      = np.load(f'{BASE}/registration_video/landmarks_27_nd2_native.npz')
ev_nd2_27 = lm27['ev_nd2']           # (27,2) x,y in full nd2 space
pcd_iv_27 = lm27['pcd_invivo_jy306'] # (27,3) z,y,x JY306

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

# User-selected cells — col 0=EV MIP, col 1=IV Warped MIP
APPROVED_CELLS = [49, 105, 100, 108, 44, 23, 20, 0]
dive_cells     = APPROVED_CELLS[:3]
print(f"  Strip cells: {APPROVED_CELLS}  Dive cells: {dive_cells}")

# ─── Panel layout constants ───────────────────────────────────────────────────
PANEL_W, PANEL_H = 860, 860
IV_CX = W//4          # left panel centre-x
EV_CX = 3*W//4        # right panel centre-x
CY    = H//2

# Fit JY306 MIP and nd2 crop into panels
iv_panel = fit_into(jy306_mip_u8,  PANEL_W, PANEL_H)
ev_panel = fit_into(nd2_crop_u8,   PANEL_W, PANEL_H)

# Landmark projections into each panel for connecting lines
# JY306 (629×658) → iv_panel (PANEL_W×PANEL_H)
_iv_sc = min(PANEL_W/629, PANEL_H/658)
_iv_xo = (PANEL_W - int(629*_iv_sc))//2
_iv_yo = (PANEL_H - int(658*_iv_sc))//2
# nd2 crop (1689×1621) → ev_panel
_nd2_cw, _nd2_ch = nd2_crop_u8.shape[1], nd2_crop_u8.shape[0]
_ev_sc = min(PANEL_W/_nd2_cw, PANEL_H/_nd2_ch)
_ev_xo = (PANEL_W - int(_nd2_cw*_ev_sc))//2
_ev_yo = (PANEL_H - int(_nd2_ch*_ev_sc))//2

def lm_to_screen_iv(k):
    """JY306 (y,x) landmark → screen pixel (px, py) in iv_panel placed at (CY, IV_CX)."""
    ry = int(pcd_iv_27[k,1]*_iv_sc) + _iv_yo + (CY - PANEL_H//2)
    rx = int(pcd_iv_27[k,2]*_iv_sc) + _iv_xo + (IV_CX - PANEL_W//2)
    return rx, ry

def lm_to_screen_ev(k):
    """nd2 crop (x,y) landmark → screen pixel in ev_panel placed at (CY, EV_CX)."""
    # ev_nd2_27 are in full nd2 space; subtract crop origin
    rx = int((ev_nd2_27[k,0]-688)*_ev_sc) + _ev_xo + (EV_CX - PANEL_W//2)
    ry = int((ev_nd2_27[k,1]-1235)*_ev_sc) + _ev_yo + (CY - PANEL_H//2)
    return rx, ry

# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO WRITER
# ═══════════════════════════════════════════════════════════════════════════════
vw = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W,H))

# ─────────────────────────────────────────────────────────────────────────────
# SCENE 1  Calcium movie playing  (72fr, 3s)
# ─────────────────────────────────────────────────────────────────────────────
print("Scene 1: calcium movie…")
for fi in range(72):
    frame = np.zeros((H,W,3),np.uint8)
    idx   = int(fi/71*(n_movie-1))
    sq    = fit_into(norm_u8(movie_frames[idx].astype(np.float32)), 860, 860)
    place(frame, sq, H//2, W//2)
    if fi >= 48:
        caption(frame,'IN VIVO  TWO-PHOTON  CALCIUM  IMAGING', alpha=ease((fi-48)/20))
    vw.write(frame)

# ─────────────────────────────────────────────────────────────────────────────
# SCENE 2  Freeze → max projection  (48fr, 2s)
# ─────────────────────────────────────────────────────────────────────────────
print("Scene 2: freeze → max-proj…")
last_frame = norm_u8(movie_frames[-1].astype(np.float32))
for fi in range(48):
    frame = np.zeros((H,W,3),np.uint8)
    img = blend(last_frame, movie_max, ease(fi/35))
    place(frame, fit_into(img, 860, 860), H//2, W//2)
    caption(frame,'IN VIVO  MAX  PROJECTION', alpha=ease((fi-20)/20))
    vw.write(frame)
save_qc(frame,'s2_max_projection')

# ─────────────────────────────────────────────────────────────────────────────
# SCENE 3  2D side-by-side  (96fr, 4s)
# Left: JY306 MIP (in-vivo structural)   Right: nd2 MIP crop (ex-vivo confocal)
# Connecting landmark lines appear after split
# ─────────────────────────────────────────────────────────────────────────────
print("Scene 3: 2D side-by-side…")
for fi in range(96):
    t_split = ease(fi/50)
    t_lines = ease((fi-55)/30)
    frame   = np.zeros((H,W,3),np.uint8)

    # Positions slide from centre to left/right
    iv_cx = int(W//2 + t_split*(IV_CX - W//2))
    ev_cx = int(W//2 + t_split*(EV_CX - W//2))

    place(frame, np.clip(iv_panel.astype(np.float32)*max(0.05,1-ease(fi/30)*0.0),
                         0,255).astype(np.uint8), CY, iv_cx)
    if t_split > 0.1:
        place(frame, ev_panel, CY, ev_cx)

    # Labels
    small_label(frame,'IN VIVO  —  JY306',  CY-PANEL_H//2-18, iv_cx-PANEL_W//2+4,
                WHITE, min(1,t_split*3), 0.40)
    small_label(frame,'EX VIVO  —  CONFOCAL', CY-PANEL_H//2-18, ev_cx-PANEL_W//2+4,
                WHITE, min(1,(t_split-0.2)*3), 0.40)

    # Connecting landmark lines (7 landmarks)
    if t_lines > 0 and t_split > 0.9:
        n_lines = int(t_lines*7)
        for k in range(n_lines):
            # Correct for current panel positions (which may differ from final)
            rx_iv = int(pcd_iv_27[k,2]*_iv_sc)+_iv_xo+(iv_cx-PANEL_W//2)
            ry_iv = int(pcd_iv_27[k,1]*_iv_sc)+_iv_yo+(CY-PANEL_H//2)
            rx_ev = int((ev_nd2_27[k,0]-688)*_ev_sc)+_ev_xo+(ev_cx-PANEL_W//2)
            ry_ev = int((ev_nd2_27[k,1]-1235)*_ev_sc)+_ev_yo+(CY-PANEL_H//2)
            ak = min(1.0, t_lines*7-k)
            cv2.circle(frame,(rx_iv,ry_iv),5,tuple(int(v*ak) for v in WHITE),-1,cv2.LINE_AA)
            cv2.circle(frame,(rx_ev,ry_ev),5,tuple(int(v*ak) for v in WHITE),-1,cv2.LINE_AA)
            cv2.line(frame,(rx_iv,ry_iv),(rx_ev,ry_ev),
                     tuple(int(v*ak*0.6) for v in WHITE),1,cv2.LINE_AA)

    caption(frame,'IN VIVO  MAX  PROJECTION', alpha=max(0,1-ease(fi/15)))
    caption(frame,'REGISTERED  VOLUMES', alpha=ease((fi-40)/35))
    vw.write(frame)
save_qc(frame,'s3_2d_sidebyside')

# ─────────────────────────────────────────────────────────────────────────────
# SCENE 4  3D clouds emerge on top of 2D panels  (144fr, 6s)
# Magenta in-vivo cloud rises from left panel; green ex-vivo from right panel
# 2D images dim as clouds appear
# ─────────────────────────────────────────────────────────────────────────────
print("Scene 4: 3D clouds emerge…")
BASE_ROT_X = math.radians(10)
BASE_ROT_Y = math.radians(5)

for fi in range(144):
    t_cloud = ease(fi/100)       # clouds fade in
    t_dim   = ease(fi/120)       # 2D panels dim
    frame   = np.zeros((H,W,3),np.uint8)

    # Dimming 2D panels
    dim_a = max(0.12, 1.0 - t_dim*0.85)
    place(frame, np.clip(iv_panel.astype(np.float32)*dim_a,0,255).astype(np.uint8), CY, IV_CX)
    place(frame, np.clip(ev_panel.astype(np.float32)*dim_a,0,255).astype(np.uint8), CY, EV_CX)

    # 3D clouds — each centred on its own panel
    cloud = render_clouds(
        ex_x,ex_y,ex_z,ex_vn, iv_x,iv_y,iv_z,iv_vn,
        BASE_ROT_Y, BASE_ROT_X,
        t_cloud, t_cloud,
        W, H, SCALE_HALF,
        ex_cx=EV_CX, ex_cy=CY, iv_cx=IV_CX, iv_cy=CY
    )
    frame = cv2.add(frame, cloud)

    # Panel labels
    small_label(frame,'IN VIVO  3D  VOLUME',  CY-PANEL_H//2-18, IV_CX-PANEL_W//2+4,
                tuple(int(v) for v in IV_COL), min(1,t_cloud*2), 0.40)
    small_label(frame,'EX VIVO  3D  VOLUME',  CY-PANEL_H//2-18, EV_CX-PANEL_W//2+4,
                tuple(int(v) for v in EX_COL), min(1,t_cloud*2), 0.40)

    caption(frame,'REGISTERED  VOLUMES',  alpha=max(0,1-ease(fi/20)))
    caption(frame,'3D  TISSUE  VOLUMES',   alpha=ease((fi-30)/40))
    vw.write(frame)
    if fi % 48 == 0: print(f'  S4 frame {fi}/144')
save_qc(frame,'s4_clouds_on_2d')

# ─────────────────────────────────────────────────────────────────────────────
# SCENE 5  Both clouds slide to centre → register + rotate  (168fr, 7s)
# Phase A (0–80):   clouds slide from IV_CX/EV_CX toward W//2
# Phase B (80–168): full merged cloud, slow Y rotation to show depth
# ─────────────────────────────────────────────────────────────────────────────
print("Scene 5: merge + rotate…")
for fi in range(168):
    frame = np.zeros((H,W,3),np.uint8)

    if fi < 80:
        t_merge = ease(fi/75)
        iv_cx_now = int(IV_CX + t_merge*(W//2 - IV_CX))
        ev_cx_now = int(EV_CX + t_merge*(W//2 - EV_CX))
        scale_now  = int(SCALE_HALF + t_merge*(SCALE_FULL - SCALE_HALF))
        cloud = render_clouds(
            ex_x,ex_y,ex_z,ex_vn, iv_x,iv_y,iv_z,iv_vn,
            BASE_ROT_Y, BASE_ROT_X,
            1.0, 1.0,
            W, H, scale_now,
            ex_cx=ev_cx_now, ex_cy=CY, iv_cx=iv_cx_now, iv_cy=CY
        )
        frame[:] = cloud
        caption(frame,'3D  TISSUE  VOLUMES', alpha=max(0,1-ease(fi/25)))
        caption(frame,'MULTIMODAL  REGISTRATION', alpha=ease((fi-35)/30))
    else:
        # Full merged cloud, rotate
        t_rot = (fi-80)/87
        rot_y = BASE_ROT_Y + math.radians(-20*ease(t_rot) + 10*(1-ease(t_rot)))
        rot_x = BASE_ROT_X + math.radians(5*math.sin(t_rot*math.pi))
        cloud = render_clouds(
            ex_x,ex_y,ex_z,ex_vn, iv_x,iv_y,iv_z,iv_vn,
            rot_y, rot_x, 1.0, 1.0,
            W, H, SCALE_FULL
        )
        frame[:] = cloud
        # Legend
        cv2.circle(frame,(28,28),7,tuple(int(v) for v in EX_COL),-1,cv2.LINE_AA)
        small_label(frame,'EX VIVO  CONFOCAL',32,42,tuple(int(v) for v in EX_COL))
        cv2.circle(frame,(28,52),7,tuple(int(v) for v in IV_COL),-1,cv2.LINE_AA)
        small_label(frame,'IN VIVO  CALCIUM', 56,42,tuple(int(v) for v in IV_COL))
        caption(frame,'MULTIMODAL  REGISTRATION', alpha=1.0)

    vw.write(frame)
    if fi==79:  save_qc(frame,'s5a_clouds_merging')
save_qc(frame,'s5b_registered_rotated')

# ─────────────────────────────────────────────────────────────────────────────
# SCENE 6  8-cell strip  4×4  (192fr, 8s)
# Row 1: EV  cells 1-4   Row 2: IV-warp  cells 1-4
# Row 3: EV  cells 5-8   Row 4: IV-warp  cells 5-8
# ─────────────────────────────────────────────────────────────────────────────
print("Scene 6: 8-cell strip 4×4…")
CS_CELL = 160; CS_GAP = 14; CS_COLS = 4
CS_TOTAL_W = CS_COLS*(CS_CELL+CS_GAP) - CS_GAP
CS_TOTAL_H = 4*(CS_CELL+CS_GAP) - CS_GAP
CS_X0 = (W - CS_TOTAL_W)//2
CS_Y0 = (H - CS_TOTAL_H)//2 - 20

ev_strip, ivw_strip = [], []
for ci in APPROVED_CELLS:
    ev_strip.append(cv2.resize(
        patch_strip[ci*PATCH_SZ:(ci+1)*PATCH_SZ, 0:PATCH_SZ],
        (CS_CELL,CS_CELL), interpolation=cv2.INTER_LANCZOS4))
    ivw_strip.append(cv2.resize(
        patch_strip[ci*PATCH_SZ:(ci+1)*PATCH_SZ, PATCH_SZ:PATCH_SZ*2],
        (CS_CELL,CS_CELL), interpolation=cv2.INTER_LANCZOS4))

for fi in range(192):
    frame = np.zeros((H,W,3),np.uint8)
    for k in range(8):
        col_k = k % CS_COLS; group = k // CS_COLS
        x0  = CS_X0 + col_k*(CS_CELL+CS_GAP)
        ey0 = CS_Y0 + group*2*(CS_CELL+CS_GAP)
        iy0 = ey0 + CS_CELL + CS_GAP
        at  = ease((fi - k*16)/22)
        if at <= 0: continue
        col_c = CELL_COLS[k]
        frame[ey0:ey0+CS_CELL, x0:x0+CS_CELL] = np.clip(
            ev_strip[k].astype(np.float32)*at,0,255).astype(np.uint8)
        frame[iy0:iy0+CS_CELL, x0:x0+CS_CELL] = np.clip(
            ivw_strip[k].astype(np.float32)*at,0,255).astype(np.uint8)
        cv2.rectangle(frame,(x0,ey0),(x0+CS_CELL-1,ey0+CS_CELL-1),
                      tuple(int(v*at) for v in col_c),1)
        cv2.rectangle(frame,(x0,iy0),(x0+CS_CELL-1,iy0+CS_CELL-1),
                      tuple(int(v*at*0.5) for v in col_c),1)
        cv2.putText(frame,str(k+1),(x0+4,ey0-5),FONT,0.45,
                    tuple(int(v*at) for v in col_c),1,cv2.LINE_AA)

    all_t = ease((fi-7*16)/20)
    if all_t > 0:
        small_label(frame,'EX VIVO', CS_Y0+CS_CELL//2,     CS_X0-82, WHITE, all_t, 0.37)
        small_label(frame,'IN VIVO', CS_Y0+CS_CELL+CS_GAP+CS_CELL//2, CS_X0-82, WHITE, all_t, 0.37)
        small_label(frame,'EX VIVO', CS_Y0+2*(CS_CELL+CS_GAP)+CS_CELL//2, CS_X0-82, WHITE, all_t, 0.37)
        small_label(frame,'IN VIVO', CS_Y0+3*(CS_CELL+CS_GAP)+CS_CELL//2, CS_X0-82, WHITE, all_t, 0.37)
    caption(frame,'MULTIMODAL  REGISTRATION', alpha=max(0,1-ease(fi/20)))
    caption(frame,'MATCHED  NEURONS', alpha=ease((fi-130)/30))
    vw.write(frame)
save_qc(frame,'s6_cell_strip_4x4')

# ─────────────────────────────────────────────────────────────────────────────
# SCENE 7-9  Per-cell deep-dive  (3 × 192fr, 3 × 8s)
# 3 panels side-by-side:
#   Left  : Calcium movie live playback at cell location
#   Centre: Ex-vivo confocal patch (col 0)
#   Right : In-vivo warped patch   (col 1)
# ─────────────────────────────────────────────────────────────────────────────
print("Scene 7-9: per-cell panels…")
PANEL_SZ  = 320
PANEL_GAP = 40
PANELS_W  = 3*PANEL_SZ + 2*PANEL_GAP
PX1 = (W - PANELS_W)//2
PX2 = PX1 + PANEL_SZ + PANEL_GAP
PX3 = PX2 + PANEL_SZ + PANEL_GAP
PY  = (H - PANEL_SZ)//2 - 20
CROP_R = 45

for cell_idx, ci in enumerate(dive_cells):
    lm_y,lm_x = all_lm_jy[ci]
    col = CELL_COLS[cell_idx]

    ev_p = cv2.resize(patch_strip[ci*PATCH_SZ:(ci+1)*PATCH_SZ, 0:PATCH_SZ],
                       (PANEL_SZ,PANEL_SZ), interpolation=cv2.INTER_LANCZOS4)
    iv_p = cv2.resize(patch_strip[ci*PATCH_SZ:(ci+1)*PATCH_SZ, PATCH_SZ:PATCH_SZ*2],
                       (PANEL_SZ,PANEL_SZ), interpolation=cv2.INTER_LANCZOS4)

    for fi in range(192):
        frame = np.zeros((H,W,3),np.uint8)
        a_in  = ease(fi/25)

        # Calcium movie crop (live playback)
        mov_idx = int(fi/191*(n_movie-1))
        mov_f   = movie_frames[mov_idx]
        y0=max(0,lm_y-CROP_R); y1=min(mov_f.shape[0],lm_y+CROP_R)
        x0=max(0,lm_x-CROP_R); x1=min(mov_f.shape[1],lm_x+CROP_R)
        crop = mov_f[y0:y1,x0:x1]
        sq = np.zeros((CROP_R*2,CROP_R*2),np.uint8)
        sq[:crop.shape[0],:crop.shape[1]] = crop
        ca_panel = cv2.cvtColor(
            cv2.resize(norm_u8(sq.astype(np.float32)),(PANEL_SZ,PANEL_SZ),
                       interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_GRAY2BGR)
        cv2.line(ca_panel,(PANEL_SZ//2-12,PANEL_SZ//2),(PANEL_SZ//2+12,PANEL_SZ//2),(180,180,180),1)
        cv2.line(ca_panel,(PANEL_SZ//2,PANEL_SZ//2-12),(PANEL_SZ//2,PANEL_SZ//2+12),(180,180,180),1)

        for panel, px, label in [(ca_panel,PX1,'IN VIVO  CALCIUM  MOVIE'),
                                   (ev_p,    PX2,'EX VIVO  CONFOCAL'),
                                   (iv_p,    PX3,'IN VIVO  WARPED')]:
            frame[PY:PY+PANEL_SZ, px:px+PANEL_SZ] = \
                np.clip(panel.astype(np.float32)*a_in,0,255).astype(np.uint8)
            cv2.rectangle(frame,(px,PY),(px+PANEL_SZ-1,PY+PANEL_SZ-1),
                          tuple(int(v*a_in) for v in WHITE),1)
            small_label(frame,label,PY-18,px,GRAY,a_in,0.40)

        cv2.putText(frame,f'CELL  {cell_idx+1}',(14,34),FONT,0.55,
                    tuple(int(v*a_in) for v in col),1,cv2.LINE_AA)
        caption(frame,'FUNCTIONAL  /  STRUCTURAL  READOUT', alpha=a_in)
        vw.write(frame)

    for _ in range(12): vw.write(np.zeros((H,W,3),np.uint8))
    save_qc(frame,f's{7+cell_idx}_cell{cell_idx+1}')
    print(f"  Cell {cell_idx+1} done")

# Final hold
for _ in range(24): vw.write(np.zeros((H,W,3),np.uint8))
vw.release()
print(f"Raw: {TMP}")

print("Re-encoding H.264…")
subprocess.run(['ffmpeg','-y','-i',TMP,'-vcodec','libx264','-pix_fmt','yuv420p',
                '-crf','18','-preset','fast',OUT],check=True)
total = 72+48+96+144+168+192+3*192+3*12+24
print(f"\n✓  {OUT}  ({total/FPS:.0f}s @ {FPS}fps)")
print(f"   QC frames → {QC}/")
