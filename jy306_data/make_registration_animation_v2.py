#!/usr/bin/env python3
"""
Registration animation v2 — 3-D point-cloud style (same as HTML viewer).
  Phase 0 : Calcium movie plays in JY306 space               (150 fr, 5s)
  Phase 1 : Pause → temporal max-projection                  ( 60 fr, 2s)
  Phase 2 : Max-frame cross-fades with JY306 z=3 reference   ( 90 fr, 3s)
  Phase 3 : Z-scan through JY306 in-vivo stack               ( 72 fr, 2.4s)
  Phase 4 : 3-D point cloud: ex-vivo appears, in-vivo
            warps in, 30-degree rotation to show depth        (150 fr, 5s)
  Phase 5–7: 3 focal-plane cells — calcium / iv / ev panels  (3×180 fr, 18s)
  Total ≈ 36s @ 30 fps

3-D rendering: numpy orthographic projection (same thresholding as HTML viewer)
  Ex-vivo  : stitched_gfp_fullres_v5_1um_isotropic.tif  DS=4, /4000, thresh>8
  In-vivo  : JY306 stack → stitched space via global affine, thresh>50
"""
import numpy as np
import cv2, tifffile, json, os, glob, math, subprocess

BASE    = '/Users/neurolab/neuroinformatics/margaret'
W, H    = 1920, 1080
FPS     = 30
TMP_MP4 = f'{BASE}/png_exports/registration_animation_v2_raw.mp4'
OUT_MP4 = f'{BASE}/png_exports/registration_animation_v2.mp4'
os.makedirs(f'{BASE}/png_exports', exist_ok=True)

IV_XY_UM   = 0.6835
IV_Z_UM    = 3.0
SKIP_TILES = {'row3_1', 'row3_5'}

GREEN   = (  80, 255,  80)   # BGR
MAGENTA = ( 200,  60, 255)
CYAN    = ( 255, 220,   0)
ORANGE  = (  20, 140, 255)
WHITE   = ( 255, 255, 255)
YELLOW  = (  20, 220, 255)

# ───────────────────────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────────────────────
def norm_u8(img, lo=2, hi=99.5):
    v = img.ravel(); v = v[v > 0]
    if len(v) < 100: return np.zeros(img.shape, np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img.astype(np.float32)-p1)/max(p2-p1,1)*255,0,255).astype(np.uint8)

def ease(t): return 0.5 - 0.5*math.cos(math.pi*np.clip(t,0,1))

def blend_f(a, b, t):
    return np.clip((1-t)*a.astype(np.float32)+t*b.astype(np.float32),0,255).astype(np.uint8)

def lbl(frame, text, y, x, col=WHITE, sc=0.65, th=1):
    cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,sc,col,th,cv2.LINE_AA)

def to_bgr(gray, col_bgr):
    rgb = np.zeros(gray.shape+(3,), np.float32)
    norm = gray.astype(np.float32)/255.0
    for c,v in enumerate(col_bgr): rgb[:,:,c] = norm*v
    return np.clip(rgb,0,255).astype(np.uint8)

def fit_square(img, size):
    if img.ndim==2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h,w = img.shape[:2]; s = size/max(h,w)
    nh,nw = int(h*s),int(w*s)
    rs = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LANCZOS4)
    c  = np.zeros((size,size,3),np.uint8)
    yo,xo = (size-nh)//2,(size-nw)//2
    c[yo:yo+nh,xo:xo+nw] = rs
    return c, s, yo, xo

def place_center(frame,img,cy,cx):
    ih,iw = img.shape[:2]
    y0,x0 = cy-ih//2, cx-iw//2
    y1,x1 = y0+ih, x0+iw
    sy0,sx0 = max(0,-y0),max(0,-x0)
    fy0,fx0 = max(0,y0),max(0,x0)
    fy1 = min(H,y1); fx1 = min(W,x1)
    if fy1>fy0 and fx1>fx0:
        frame[fy0:fy1,fx0:fx1] = img[sy0:sy0+(fy1-fy0),sx0:sx0+(fx1-fx0)]

def title_bar(frame, text, sub='', alpha=1.0):
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(W,65),(0,0,0),-1)
    lbl(ov,text,32,30,YELLOW,0.85,2)
    if sub: lbl(ov,sub,56,32,(180,180,180),0.44,1)
    cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)

# ───────────────────────────────────────────────────────────────────────
# 3-D POINT CLOUD RENDERER  (numpy orthographic projection)
# ───────────────────────────────────────────────────────────────────────
def rot_y(a):
    c,s = math.cos(a),math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]],np.float32)

def rot_x(a):
    c,s = math.cos(a),math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]],np.float32)

def render_pc(ex_pts, ex_v, iv_pts, iv_v, angle_y, angle_x,
              canvas_w, canvas_h, scale=620, ex_alpha=1.0, iv_alpha=1.0):
    """
    Render two point clouds (ex-vivo green, in-vivo magenta) with rotation.
    pts: (N,3) normalized [0,1], v: (N,) brightness [0,1]
    """
    R = rot_x(angle_x) @ rot_y(angle_y)
    img = np.zeros((canvas_h, canvas_w, 3), np.float32)

    def add_cloud(pts, v, col_bgr, alpha):
        if len(pts) == 0 or alpha < 0.01: return
        c  = (pts - 0.5)                   # center around origin
        r  = (R @ c.T).T                   # rotate  (N,3)
        sx = (r[:,0]*scale + canvas_w//2).astype(np.int32)
        sy = (-r[:,1]*scale + canvas_h//2).astype(np.int32)
        depth = r[:,2]
        valid = (sx>=0)&(sx<canvas_w)&(sy>=0)&(sy<canvas_h)
        sx,sy,dv,vv = sx[valid],sy[valid],depth[valid],v[valid]
        # sort back-to-front
        order = np.argsort(-dv)
        sx,sy,vv = sx[order],sy[order],vv[order]
        brightness = vv * alpha
        for c_i,(bv,cv) in enumerate(zip([col_bgr[0],col_bgr[1],col_bgr[2]],
                                          [col_bgr[0],col_bgr[1],col_bgr[2]])):
            np.add.at(img[:,:,c_i], (sy,sx), brightness * (col_bgr[c_i]/255.0))
        # accumulate weight for normalisation per pixel
        # (simple overwrite is faster; use maximum for point-cloud feel)

    # Faster: use cv2 for dot drawing (avoids slow python loops)
    def draw_cloud(pts, v, col_bgr, alpha, pt_sz=2):
        if len(pts)==0 or alpha<0.01: return
        c  = (pts - 0.5).astype(np.float32)
        r  = (R @ c.T).T
        sx = (r[:,0]*scale + canvas_w//2).astype(np.int32)
        sy = (-r[:,1]*scale + canvas_h//2).astype(np.int32)
        depth = r[:,2]
        valid = (sx>=0)&(sx<canvas_w)&(sy>=0)&(sy<canvas_h)
        sx,sy,depth,vv = sx[valid],sy[valid],depth[valid],v[valid]
        order = np.argsort(-depth)   # back→front
        sx,sy,vv = sx[order],sy[order],vv[order]
        for i in range(len(sx)):
            bright = float(vv[i]) * alpha
            c_draw = tuple(int(col_bgr[k]*bright) for k in range(3))
            cv2.circle(img, (sx[i],sy[i]), pt_sz, c_draw, -1)

    # For speed: subsample to 300 K pts per cloud
    MAX_PTS = 300_000
    def subsample(pts, v, n=MAX_PTS):
        if len(pts) <= n: return pts, v
        idx = np.random.choice(len(pts), n, replace=False)
        return pts[idx], v[idx]

    ex_p, ex_vv = subsample(ex_pts, ex_v)
    iv_p, iv_vv = subsample(iv_pts, iv_v)

    draw_cloud(ex_p, ex_vv, GREEN,   ex_alpha, pt_sz=2)
    draw_cloud(iv_p, iv_vv, MAGENTA, iv_alpha, pt_sz=2)

    return np.clip(img, 0, 255).astype(np.uint8)

# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
print("Loading JY306 stack…")
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv, ny_iv, nx_iv = jy306.shape      # (16,658,629)
jy306_mip_u8 = norm_u8(np.max(jy306,axis=0))
jy306_z_u8   = [norm_u8(jy306[z]) for z in range(nz_iv)]
jy306_z3_u8  = jy306_z_u8[3]

print("Loading calcium movie (warped to JY306)…")
cap = cv2.VideoCapture(f'{BASE}/png_exports/native_invivo/movie_warped_h264.mp4')
movie_frames = []
while True:
    ret,frm = cap.read()
    if not ret: break
    movie_frames.append(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY))
cap.release()
movie_frames = np.array(movie_frames, np.uint8)      # (663, 658, 628)
n_movie = len(movie_frames)
movie_max_u8 = norm_u8(np.max(movie_frames.astype(np.float32),axis=0))
print(f"  {n_movie} frames {movie_frames.shape[1:]}")

print("Loading nd2 row2_1 tile…")
nd2_slices = []
for zi in range(12):
    p = cv2.imread(f'{BASE}/png_exports/registration_video/row2_1/GFP_z{zi:03d}.png',
                   cv2.IMREAD_UNCHANGED)
    if p is not None: nd2_slices.append(p.astype(np.float32))
nd2_mip_u8 = norm_u8(np.max(nd2_slices,axis=0))

lm27 = np.load(f'{BASE}/registration_video/landmarks_27_nd2_native.npz')
ev_nd2_27 = lm27['ev_nd2']
pcd_iv_27 = lm27['pcd_invivo_jy306']

M_nd2_to_jy = np.load(f'{BASE}/registration_video/affine_nd2_to_exvivo.npy')
M3   = np.vstack([M_nd2_to_jy,[0,0,1]])
M_jy_to_nd2 = np.linalg.inv(M3)[:2,:]

# SIFT affine: movie(512×512 flipped)→JY306(658×629)
theta = math.radians(0.045); s = 0.881
M_sift = np.array([[s*math.cos(theta),-s*math.sin(theta),61.7],
                    [s*math.sin(theta), s*math.cos(theta),89.3]],np.float64)

print("Loading patch strip + cell info…")
patch_strip = cv2.imread(f'{BASE}/3d_viewer/patch_strip_v4.png')
with open(f'{BASE}/3d_viewer/cell_info_v4.json') as f:
    raw_ci = json.load(f)
cell_info = [json.loads(x) if isinstance(x,str) else x for x in raw_ci]
PATCH_SZ = 80

print("Building landmark order list…")
tile_lm_map = {}
for lf in sorted(glob.glob(f'{BASE}/registration_video/landmarks_nd2_native_*.npz')):
    tile = os.path.basename(lf).replace('landmarks_nd2_native_','').replace('.npz','')
    if tile not in SKIP_TILES: tile_lm_map[tile] = lf
legacy = f'{BASE}/registration_video/landmarks_27_nd2_native.npz'
if os.path.exists(legacy) and 'row2_1' not in tile_lm_map:
    tile_lm_map['row2_1'] = legacy

all_lm_jy = []
for tile in sorted(tile_lm_map):
    d = np.load(tile_lm_map[tile])
    for row in d['pcd_invivo_jy306']:
        all_lm_jy.append((int(round(row[1])), int(round(row[2]))))

focal_cells = [ci for ci,info in enumerate(cell_info) if 1<=info[3]<=5][:3]
print(f"  Focal-plane cells: {focal_cells}")

# ───────────────────────────────────────────────────────────────────────
# BUILD 3-D POINT CLOUDS  (same logic as HTML viewer)
# ───────────────────────────────────────────────────────────────────────
DS_EX = 4
THRESH_EX = 8
THRESH_IV = 50

print("Building ex-vivo 3D point cloud (DS=4, thresh>8, /4000)…")
EX_TIFF = f'{BASE}/registration_video/stitched/stitched_gfp_fullres_v5_1um_isotropic.tif'
with tifffile.TiffFile(EX_TIFF) as tif:
    n_pages   = len(tif.pages)
    ex_h_full = tif.pages[0].shape[0]
    ex_w_full = tif.pages[0].shape[1]
    ex_nz = n_pages  // DS_EX
    ex_ny = ex_h_full // DS_EX
    ex_nx = ex_w_full // DS_EX
    print(f"  Full ({n_pages},{ex_h_full},{ex_w_full}) → DS4 ({ex_nz},{ex_ny},{ex_nx})")
    ex_vol = np.zeros((ex_nz, ex_ny, ex_nx), np.float32)
    for zi in range(ex_nz):
        sl = tif.pages[zi*DS_EX].asarray().astype(np.float32)
        ex_vol[zi] = sl[::DS_EX,:ex_w_full:DS_EX][:ex_ny,:ex_nx]

# Threshold exactly like HTML: /4000*255, then keep > 8
ex_u8 = np.clip(ex_vol / 4000 * 255, 0, 255).astype(np.uint8)
del ex_vol

ez, ey, exx = np.where(ex_u8 > THRESH_EX)
ex_vals_raw = ex_u8[ez, ey, exx].astype(np.float32) / 255.0
del ex_u8
print(f"  Ex-vivo voxels: {len(ez):,}")

# Normalise positions to [0,1] cube (same as HTML)
span  = float(max(ex_nx, ex_ny, ex_nz))
ex_pts_norm = np.column_stack([exx/span, ey/span, ez/span]).astype(np.float32)
ex_cx = exx.mean()/span; ex_cy = ey.mean()/span; ex_cz = ez.mean()/span
ex_pts_norm[:,0] += 0.5-ex_cx
ex_pts_norm[:,1] += 0.5-ex_cy
ex_pts_norm[:,2] += 0.5-ex_cz
del ez, ey, exx

# ── In-vivo: JY306 → stitched space via global affine from all landmarks ──
print("Computing JY306 → stitched affine from all landmarks…")
src_all, dst_all = [], []
for lf in sorted(glob.glob(f'{BASE}/registration_video/landmarks_stitched_v5_*.npz')):
    tile = os.path.basename(lf).replace('landmarks_stitched_v5_','').replace('.npz','')
    if tile in SKIP_TILES: continue
    sc  = np.load(lf)
    pcd = sc['pcd_invivo_jy306']                # (N,3) z,y,x JY306 px
    st  = sc['stitched_coords']                 # (N,3) z,y,x µm
    src = np.column_stack([pcd[:,0]*IV_Z_UM, pcd[:,1]*IV_XY_UM, pcd[:,2]*IV_XY_UM])
    src_all.append(src); dst_all.append(st)
src_all = np.vstack(src_all); dst_all = np.vstack(dst_all)
src_h   = np.hstack([src_all, np.ones((len(src_all),1))])
A_T, _, _, _ = np.linalg.lstsq(src_h, dst_all, rcond=None)   # (4,3)
pred = src_h @ A_T
err  = np.sqrt(((pred-dst_all)**2).sum(axis=1))
print(f"  JY306→stitched affine: mean error = {err.mean():.1f} µm")

print("Building in-vivo 3D point cloud…")
# Median-filter BG subtraction (same as HTML viewer)
from scipy.ndimage import median_filter
iv_sub = np.zeros_like(jy306)
for z in range(nz_iv):
    bg = median_filter(jy306[z], size=15)
    iv_sub[z] = np.clip(jy306[z] - bg, 0, None)

iv_p99 = np.percentile(iv_sub[iv_sub>0], 99) if (iv_sub>0).any() else 1
iv_u8  = np.clip(iv_sub / max(iv_p99,1) * 255, 0, 255).astype(np.uint8)
iz, iy, ix = np.where(iv_u8 > THRESH_IV)
iv_vals_raw = iv_u8[iz, iy, ix].astype(np.float32) / 255.0

# Transform JY306 voxels → stitched µm space
pts_jy  = np.column_stack([iz*IV_Z_UM, iy*IV_XY_UM, ix*IV_XY_UM])
pts_h   = np.hstack([pts_jy, np.ones((len(pts_jy),1))])
pts_st  = (pts_h @ A_T)                        # (N,3) in µm
# Convert µm → DS4 grid index (same normalisation as ex-vivo)
pts_px  = pts_st / DS_EX                       # divide by DS_EX (1µm/px ÷ 4)
iv_pts_norm = np.column_stack([pts_px[:,2]/span, pts_px[:,1]/span,
                                pts_px[:,0]/span]).astype(np.float32)
iv_pts_norm[:,0] += 0.5-ex_cx
iv_pts_norm[:,1] += 0.5-ex_cy
iv_pts_norm[:,2] += 0.5-ex_cz
print(f"  In-vivo voxels: {len(iz):,}")
del iv_sub, iv_u8, pts_jy, pts_h, pts_st, pts_px, iz, iy, ix

# ═══════════════════════════════════════════════════════════════════════
# OPEN VIDEO WRITER
# ═══════════════════════════════════════════════════════════════════════
print("Opening VideoWriter…")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vw = cv2.VideoWriter(TMP_MP4, fourcc, FPS, (W, H))

SZ = 680    # display size for 2-D phases

# ═══════════════════════════════════════════════════════════════════════
# PHASE 0  Calcium movie plays  (150 frames, 5s)
# ═══════════════════════════════════════════════════════════════════════
print("Phase 0: calcium movie…")
for fi in range(150):
    frame = np.zeros((H,W,3),np.uint8)
    idx   = fi*5 % n_movie
    sq,_,_,_ = fit_square(to_bgr(movie_frames[idx], MAGENTA), SZ)
    place_center(frame, sq, H//2, W//2)
    a = min(1.0, fi/20)
    title_bar(frame,'2-Photon Calcium Imaging  |  In Vivo',
              f'JY306 hippocampus  —  {n_movie} frames  —  warped to JY306 space',a)
    lbl(frame,f'Frame {idx}/{n_movie-1}',H-25,W-185,(120,120,120),0.4)
    vw.write(frame)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1  Pause → max-projection  (60 frames, 2s)
# ═══════════════════════════════════════════════════════════════════════
print("Phase 1: max projection…")
last_frame = movie_frames[(149*5) % n_movie]
for fi in range(60):
    t = ease(fi/59)
    frame = np.zeros((H,W,3),np.uint8)
    img_b = blend_f(last_frame, movie_max_u8, t)
    sq,_,_,_ = fit_square(to_bgr(img_b, MAGENTA), SZ)
    sq = np.clip(sq.astype(np.float32)*(1+0.4*t),0,255).astype(np.uint8)
    place_center(frame,sq,H//2,W//2)
    title_bar(frame,'Temporal Max-Projection  (663 frames)',alpha=float(t))
    if fi<20: lbl(frame,'[ PAUSED ]',H//2-SZ//2-30,W//2-65,
                  tuple(int(v*(1-fi/20)) for v in YELLOW),0.6)
    vw.write(frame)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2  Cross-fade → JY306 z=3  (90 frames, 3s)
# ═══════════════════════════════════════════════════════════════════════
print("Phase 2: align to JY306 z=3…")
mh = min(movie_max_u8.shape[0], jy306_z3_u8.shape[0])
mw = min(movie_max_u8.shape[1], jy306_z3_u8.shape[1])
for fi in range(90):
    t  = ease(fi/89)
    frame = np.zeros((H,W,3),np.uint8)
    mv = to_bgr(movie_max_u8[:mh,:mw], MAGENTA).astype(np.float32)*(1-t*0.7)
    jy = to_bgr(jy306_z3_u8[:mh,:mw],  GREEN).astype(np.float32)*t
    comp = np.clip(mv+jy,0,255).astype(np.uint8)
    sq, scale, yo, xo = fit_square(comp, SZ)
    place_center(frame,sq,H//2,W//2)
    if t > 0.3:
        a_lm = min(1.0,(t-0.3)/0.4)
        cy0,cx0 = H//2-SZ//2, W//2-SZ//2
        for i in range(len(pcd_iv_27)):
            ry = int(pcd_iv_27[i,1]*scale)+yo+cy0
            rx = int(pcd_iv_27[i,2]*scale)+xo+cx0
            col = tuple(int(v*a_lm) for v in CYAN)
            cv2.circle(frame,(rx,ry),5,col,-1,cv2.LINE_AA)
    title_bar(frame,'Calcium Max-Projection  →  In-Vivo z=3 Reference',
              'Magenta = calcium  |  Green = two-photon stack')
    lbl(frame,f'{int(t*100)}% aligned',H-30,W-160,(160,160,160),0.5)
    vw.write(frame)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 3  Z-scan through JY306 stack  (72 frames, 2.4s)
# ═══════════════════════════════════════════════════════════════════════
print("Phase 3: JY306 z-scan…")
BAR_X = W-75; BAR_H = SZ-40; BAR_Y0 = H//2-SZ//2+20
for fi in range(72):
    z = fi % nz_iv
    frame = np.zeros((H,W,3),np.uint8)
    sq,scale,yo,xo = fit_square(to_bgr(jy306_z_u8[z], GREEN), SZ)
    place_center(frame,sq,H//2,W//2)
    cy0,cx0 = H//2-SZ//2, W//2-SZ//2
    for i in range(len(pcd_iv_27)):
        dz = abs(pcd_iv_27[i,0]-z)
        if dz>2: continue
        a_lm = max(0.2,1.0-dz*0.4)
        ry = int(pcd_iv_27[i,1]*scale)+yo+cy0
        rx = int(pcd_iv_27[i,2]*scale)+xo+cx0
        cv2.circle(frame,(rx,ry),6,tuple(int(v*a_lm) for v in CYAN),-1,cv2.LINE_AA)
        cv2.circle(frame,(rx,ry),6,WHITE,1,cv2.LINE_AA)
    # Z-bar
    cv2.rectangle(frame,(BAR_X,BAR_Y0),(BAR_X+12,BAR_Y0+BAR_H),(55,55,55),-1)
    zp = int(z/(nz_iv-1)*BAR_H)
    cv2.rectangle(frame,(BAR_X,BAR_Y0),(BAR_X+12,BAR_Y0+zp),GREEN,-1)
    lbl(frame,f'z={z:2d}',BAR_Y0+BAR_H+14,BAR_X-4,(160,160,160),0.4)
    title_bar(frame,'In-Vivo Stack  (JY306)',
              f'z = {z:2d}/{nz_iv-1}  |  0.68 µm/px XY  |  3 µm z-step  |  27 landmarks shown')
    vw.write(frame)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 4  3-D point cloud: ex-vivo appears → in-vivo warps in → 30° rotation
#          (150 frames, 5s)
# ═══════════════════════════════════════════════════════════════════════
print("Phase 4: 3-D point cloud registration…")
# Rotation schedule:
#  0-30  fr : ex-vivo fades in, static view (angle_y=0.1 rad ≈ 6°)
#  30-75 fr : in-vivo warps in (fade alpha 0→1)
#  75-150 fr: slow 30° rotation around Y  (0.1 → 0.1+π/6 rad)
START_Y = 0.1       # ~6° initial tilt
END_Y   = START_Y + math.pi/6    # +30°
AX      = -0.25     # slight X tilt for depth perception (constant)

PC_W, PC_H = W, H   # full canvas for point cloud

for fi in range(150):
    frame = np.zeros((H,W,3),np.uint8)

    if fi < 30:
        ex_a  = ease(fi/29)
        iv_a  = 0.0
        ay    = START_Y
        sub   = 'Ex-vivo stitched volume  |  threshold /4000 > 8  |  Green'
    elif fi < 75:
        ex_a  = 1.0
        iv_a  = ease((fi-30)/44)
        ay    = START_Y
        sub   = 'In-vivo warping in  |  threshold > 50  |  Magenta'
    else:
        ex_a  = 1.0
        iv_a  = 1.0
        t_rot = (fi-75)/74
        ay    = START_Y + ease(t_rot)*(END_Y-START_Y)
        sub   = f'Rotating +30°  to show 3-D structure  (angle={math.degrees(ay):.0f}°)'

    pc_img = render_pc(ex_pts_norm, ex_vals_raw,
                       iv_pts_norm, iv_vals_raw,
                       ay, AX, PC_W, PC_H,
                       scale=580, ex_alpha=float(ex_a), iv_alpha=float(iv_a))

    frame[:] = pc_img

    # Legend
    cv2.circle(frame,(50,H-50),8,GREEN,-1,cv2.LINE_AA)
    lbl(frame,'Ex-Vivo (stitched)',H-44,65,GREEN,0.5)
    cv2.circle(frame,(280,H-50),8,MAGENTA,-1,cv2.LINE_AA)
    lbl(frame,'In-Vivo (warped)',H-44,295,MAGENTA,0.5)

    title_bar(frame,'3-D Registration  |  In-Vivo → Ex-Vivo', sub)
    vw.write(frame)

# ═══════════════════════════════════════════════════════════════════════
# PHASES 5–7  Per-cell panels  (3 × 180 frames, 6s each)
# ═══════════════════════════════════════════════════════════════════════
PANEL_SZ  = 200
PANEL_Y   = H - PANEL_SZ - 55
GAP       = 28
PX0       = (W - 3*PANEL_SZ - 2*GAP) // 2
MAIN_H_PC = PANEL_Y - 75    # height for point cloud in main area

# Pre-compute a static 3-D render at the final rotated angle for the background
static_pc = render_pc(ex_pts_norm, ex_vals_raw,
                      iv_pts_norm, iv_vals_raw,
                      END_Y+0.1, AX, W, MAIN_H_PC,
                      scale=460, ex_alpha=0.6, iv_alpha=0.6)

print("Phases 5–7: cell panels…")
for cell_idx, ci in enumerate(focal_cells):
    info     = cell_info[ci]          # [z_nd2,ez_lo,ez_hi,z_iv,ivz_lo,ivz_hi]
    lm_y, lm_x = all_lm_jy[ci]      # (y,x) in JY306 pixels

    ev_patch  = cv2.resize(patch_strip[ci*PATCH_SZ:(ci+1)*PATCH_SZ, 0:PATCH_SZ],
                            (PANEL_SZ,PANEL_SZ))
    piv_patch = cv2.resize(patch_strip[ci*PATCH_SZ:(ci+1)*PATCH_SZ, PATCH_SZ*4:PATCH_SZ*5],
                            (PANEL_SZ,PANEL_SZ))

    for fi in range(180):
        frame = np.zeros((H,W,3),np.uint8)

        # ── main area: 3-D point cloud (dim background) with pulsing dot
        frame[:MAIN_H_PC,:,:] = static_pc

        # Pulsing landmark sphere — find 3-D position on screen
        # Use iv_pts_norm for this landmark's approximate position
        lm_jy_um = np.array([[lm_y*IV_XY_UM, lm_x*IV_XY_UM, 3*IV_Z_UM]])  # z≈3
        lm_st    = (np.hstack([lm_jy_um[:,::-1], [[1]]]) @ A_T) / DS_EX / span
        lm_norm  = np.array([lm_st[0,2]+0.5-ex_cx,
                              lm_st[0,1]+0.5-ex_cy,
                              lm_st[0,0]+0.5-ex_cz], np.float32)
        R_static = rot_x(AX) @ rot_y(END_Y+0.1)
        lm_rot   = R_static @ (lm_norm - 0.5)
        lm_sx    = int(lm_rot[0]*460 + W//2)
        lm_sy    = int(-lm_rot[1]*460 + MAIN_H_PC//2)
        pulse    = 10 + int(5*math.sin(fi*0.25))
        if 0 < lm_sx < W and 0 < lm_sy < MAIN_H_PC:
            cv2.circle(frame,(lm_sx,lm_sy),pulse,YELLOW,2,cv2.LINE_AA)
            cv2.circle(frame,(lm_sx,lm_sy),5,YELLOW,-1,cv2.LINE_AA)

        # ── separator line
        cv2.line(frame,(0,MAIN_H_PC),(W,MAIN_H_PC),(60,60,60),1)

        # ── calcium playback crop from warped movie
        movie_fi   = fi % n_movie
        m_frame    = movie_frames[movie_fi]
        y0c = max(0,lm_y-PANEL_SZ//2); y1c = min(ny_iv,lm_y+PANEL_SZ//2)
        x0c = max(0,lm_x-PANEL_SZ//2); x1c = min(m_frame.shape[1],lm_x+PANEL_SZ//2)
        crop = m_frame[y0c:y1c,x0c:x1c]
        crop_sq = np.zeros((PANEL_SZ,PANEL_SZ),np.uint8)
        crop_sq[:crop.shape[0],:crop.shape[1]] = crop
        crop_rgb = to_bgr(norm_u8(crop_sq.astype(np.float32)), MAGENTA)

        # ── draw 3 panels
        panel_alpha = float(ease(min(1.0, fi/20)))
        panels = [(crop_rgb, MAGENTA, '2P Calcium (live)'),
                  (piv_patch,  GREEN,   'In-Vivo MIP'),
                  (ev_patch,   (80,220,80), 'Ex-Vivo MIP')]
        for pi,(pimg,pcol,pname) in enumerate(panels):
            x0p = PX0 + pi*(PANEL_SZ+GAP)
            y0p = PANEL_Y
            ov = frame.copy()
            ov[y0p:y0p+PANEL_SZ,x0p:x0p+PANEL_SZ] = pimg
            cv2.rectangle(ov,(x0p,y0p),(x0p+PANEL_SZ-1,y0p+PANEL_SZ-1),pcol,2)
            lbl(ov,pname,y0p+PANEL_SZ+18,x0p+4,pcol,0.48,1)
            cv2.addWeighted(ov,panel_alpha,frame,1-panel_alpha,0,frame)

        # ── calcium frame counter
        lbl(frame,f'▶ {movie_fi}/{n_movie-1}',PANEL_Y+PANEL_SZ+38,
            PX0+4,MAGENTA,0.38,1)

        title_bar(frame,
                  f'Cell {cell_idx+1} / {len(focal_cells)}  —  Landmark #{ci}',
                  f'JY306 z={info[3]}  |  nd2 z={info[0]}  |  '
                  f'Green dot = cell position in 3-D volume')
        vw.write(frame)

    print(f"  Cell {cell_idx+1} done")

vw.release()
print(f"\nRaw MP4: {TMP_MP4}")

print("Re-encoding to H.264…")
subprocess.run(['ffmpeg','-y','-i',TMP_MP4,
                '-vcodec','libx264','-pix_fmt','yuv420p',
                '-crf','20','-preset','fast',OUT_MP4], check=True)
print(f"\n✓  {OUT_MP4}")
total = (150+60+90+72+150+3*180)/FPS
print(f"   Duration ≈ {total:.1f}s  ({W}×{H} @ {FPS}fps)")
