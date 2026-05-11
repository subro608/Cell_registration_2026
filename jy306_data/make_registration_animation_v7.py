#!/usr/bin/env python3
"""
Registration animation v7 — follows Erdem's screenplay exactly.

  S1  (120fr,  5s) : Histmatch calcium movie (movie_histmatch_avg_contrast.mp4)
  S2  ( 96fr,  4s) : Freeze → max-proj → alignment overlay (IV magenta | JY306 z=3 green)
  S3  (300fr, 12s) : Green arrows one-by-one + zoom panels (PKL row4_1 landmarks)
  S4  (144fr,  6s) : Progressive PKL warp — ACTUAL deformation, NOT fade
  S4b ( 72fr,  3s) : Hold → red+green=yellow matched-cells overlay
  S5  (336fr, 14s) : 3D point-cloud zoom-out (all volumes registered)
  S6  (192fr,  8s) : 8-cell strip  4×4  (EV / IV-warp)
  S7  (216fr,  9s) : Per-cell  cell 1
  S8  (216fr,  9s) : Per-cell  cell 2
  S9  (216fr,  9s) : Per-cell  cell 3
  Total ≈ 93s @ 24fps

QC frames → png_exports/registration_animation_v7_qc/
"""
import numpy as np, cv2, tifffile, json, os, glob, math, subprocess
from scipy.interpolate import RBFInterpolator

BASE = '/Users/neurolab/neuroinformatics/margaret'
W, H = 1920, 1080
FPS  = 24
TMP  = f'{BASE}/png_exports/registration_animation_v7_raw.mp4'
OUT  = f'{BASE}/png_exports/registration_animation_v7.mp4'
QC   = f'{BASE}/png_exports/registration_animation_v7_qc'
os.makedirs(f'{BASE}/png_exports', exist_ok=True)
os.makedirs(QC, exist_ok=True)

FONT  = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255,255,255); GRAY = (160,160,160)
GREEN = (0,200,0); MAGENTA = (200,0,200)
EX_COL = np.array([0, 200, 0],   np.float32)   # green  BGR
IV_COL = np.array([200, 0, 200], np.float32)   # magenta BGR
CELL_COLS = [
    (  0,   0,220),(  0,110,255),(  0,220,220),(60,200,60),
    (220,200,  0),(220, 80,  0),(180,  0,180),(255,255,  0),
]
SKIP_TILES = {'row3_1','row3_5'}

# ── helpers ───────────────────────────────────────────────────────────────────
def norm_u8(img, lo=1, hi=99.5):
    v=img.ravel(); v=v[v>0]
    if len(v)<50: return np.zeros(img.shape,np.uint8)
    p1,p2=np.percentile(v,[lo,hi])
    return np.clip((img.astype(np.float32)-p1)/max(p2-p1,1)*255,0,255).astype(np.uint8)

def ease(t): return float(0.5-0.5*math.cos(math.pi*max(0.,min(1.,t))))

def blend(a,b,t):
    t=max(0.,min(1.,t))
    return np.clip((1-t)*a.astype(np.float32)+t*b.astype(np.float32),0,255).astype(np.uint8)

def caption(frame,text,alpha=1.0):
    if alpha<0.01: return
    (tw,_),_=cv2.getTextSize(text,FONT,0.72,1)
    cv2.putText(frame,text,((W-tw)//2,H-42),FONT,0.72,
                tuple(int(v*alpha) for v in WHITE),1,cv2.LINE_AA)

def small_label(frame,text,y,x,col=GRAY,alpha=1.0,scale=0.40):
    cv2.putText(frame,text,(x,y),FONT,scale,
                tuple(int(v*alpha) for v in col),1,cv2.LINE_AA)

def dashed_circle(img,cx,cy,r,col,dash=14):
    for d in range(0,360,dash*2):
        a1,a2=math.radians(d),math.radians(d+dash)
        cv2.line(img,(int(cx+r*math.cos(a1)),int(cy+r*math.sin(a1))),
                     (int(cx+r*math.cos(a2)),int(cy+r*math.sin(a2))),col,2,cv2.LINE_AA)

def fit_into(img,tw,th):
    if img.ndim==2: img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    h,w=img.shape[:2]; s=min(tw/w,th/h)
    nw,nh=int(w*s),int(h*s)
    rs=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LANCZOS4)
    c=np.zeros((th,tw,3),np.uint8)
    yo,xo=(th-nh)//2,(tw-nw)//2
    c[yo:yo+nh,xo:xo+nw]=rs
    return c

def place(frame,img,cy,cx):
    ih,iw=img.shape[:2]; y0,x0=cy-ih//2,cx-iw//2
    fy0=max(0,y0);fy1=min(H,y0+ih);fx0=max(0,x0);fx1=min(W,x0+iw)
    if fy1>fy0 and fx1>fx0:
        frame[fy0:fy1,fx0:fx1]=img[fy0-y0:fy1-y0,fx0-x0:fx1-x0]

def save_qc(frame,name):
    p=f'{QC}/{name}.png'; cv2.imwrite(p,frame); print(f'  QC → {p}')

def draw_arrow(frame, p1, p2, col, alpha, thickness=2, tip=10):
    """Draw a green arrow from p1 to p2 with arrowhead."""
    if alpha < 0.01: return
    c = tuple(int(v*alpha) for v in col)
    cv2.arrowedLine(frame, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])),
                    c, thickness, cv2.LINE_AA, tipLength=tip/max(1,
                    math.hypot(p2[0]-p1[0],p2[1]-p1[1])))

# ── 3D point cloud renderer ────────────────────────────────────────────────────
def render_one_cloud(x,y,z,v,rot_y,rot_x,alpha,col_f32,
                     canvas_w,canvas_h,scale_px,cx,cy):
    canvas=np.zeros((canvas_h,canvas_w,3),np.float32)
    if alpha<0.005: return canvas
    cy_r,sy_r=np.cos(rot_y),np.sin(rot_y)
    rx=cy_r*x+sy_r*z; rz=-sy_r*x+cy_r*z
    cx_r,sx_r=np.cos(rot_x),np.sin(rot_x)
    ry2=cx_r*y-sx_r*rz; rz2=sx_r*y+cx_r*rz
    px=(rx*scale_px+cx).astype(np.int32)
    py=(ry2*scale_px+cy).astype(np.int32)
    mask=(px>=0)&(px<canvas_w)&(py>=0)&(py<canvas_h)
    px,py,v2,d=px[mask],py[mask],v[mask],rz2[mask]
    order=np.argsort(-d); px,py,v2=px[order],py[order],v2[order]
    iv_=(v2*alpha).astype(np.float32)
    np.maximum(canvas[py,px,0],iv_*col_f32[0],out=canvas[py,px,0])
    np.maximum(canvas[py,px,1],iv_*col_f32[1],out=canvas[py,px,1])
    np.maximum(canvas[py,px,2],iv_*col_f32[2],out=canvas[py,px,2])
    return canvas

def render_clouds(ex_x,ex_y,ex_z,ex_v, iv_x,iv_y,iv_z,iv_v,
                  rot_y,rot_x, ex_alpha,iv_alpha,
                  canvas_w,canvas_h,scale_px,
                  ex_cx=None,ex_cy=None, iv_cx=None,iv_cy=None):
    if ex_cx is None: ex_cx=canvas_w//2
    if ex_cy is None: ex_cy=canvas_h//2
    if iv_cx is None: iv_cx=canvas_w//2
    if iv_cy is None: iv_cy=canvas_h//2
    canvas=np.zeros((canvas_h,canvas_w,3),np.float32)
    if ex_alpha>0.005:
        c=render_one_cloud(ex_x,ex_y,ex_z,ex_v,rot_y,rot_x,ex_alpha,
                           EX_COL/255.,canvas_w,canvas_h,scale_px,ex_cx,ex_cy)
        np.maximum(canvas,c,out=canvas)
    if iv_alpha>0.005:
        c=render_one_cloud(iv_x,iv_y,iv_z,iv_v,rot_y,rot_x,iv_alpha,
                           IV_COL/255.,canvas_w,canvas_h,scale_px,iv_cx,iv_cy)
        np.maximum(canvas,c,out=canvas)
    out=np.clip(canvas*255,0,255).astype(np.uint8)
    return cv2.GaussianBlur(out,(3,3),0.8)

# ═════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
print("Loading histmatch calcium movie…")
cap=cv2.VideoCapture(f'{BASE}/png_exports/native_invivo/movie_histmatch_avg_contrast.mp4')
movie_frames=[]
while True:
    ret,frm=cap.read()
    if not ret: break
    movie_frames.append(cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY))
cap.release()
movie_frames=np.array(movie_frames,np.uint8)
n_movie=len(movie_frames)
movie_max=norm_u8(np.max(movie_frames.astype(np.float32),axis=0))
print(f"  {n_movie} frames {movie_frames.shape[1:]}")

print("Loading JY306 stack…")
jy306=tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
jy306_mip_u8=norm_u8(np.max(jy306,axis=0))
jy306_z3_u8=norm_u8(jy306[3])   # z=3 best aligns to calcium movie

print("Loading nd2 row4_1 GFP slices…")
nd2_r4_slices=[]
for zi in range(12):
    p=cv2.imread(f'{BASE}/png_exports/registration_video/row4_1/GFP_z{zi:03d}.png',
                 cv2.IMREAD_UNCHANGED)
    if p is not None: nd2_r4_slices.append(p.astype(np.float32))
nd2_r4_mip_full=norm_u8(np.max(nd2_r4_slices,axis=0))
print(f"  nd2 r4 MIP: {nd2_r4_mip_full.shape}")

print("Loading PKL transform row4_1…")
pkl=np.load(f'{BASE}/png_exports/registration_per_tile_pkl/row4_1/pkl_transform_row4_1.npz')
M2d         = pkl['M2d_jy306_to_nd2']          # (2,3): JY306(y,x,1) → nd2(y,x)
src2d       = pkl['pcd_invivo_jy306'][:,1:3].astype(np.float32)   # (70,2) JY306 y,x
tgt2d       = pkl['ev_nd2'][:,0:2].astype(np.float32)             # (70,2) nd2   y,x
pkl_weights = pkl['weights'] if 'weights' in pkl else None

print("Loading 3D voxel clouds…")
vox=np.load('/tmp/viewer_v4_voxels.npz')
ex_vx=vox['ex_vx'].astype(np.float32); ex_vy=vox['ex_vy'].astype(np.float32)
ex_vz=vox['ex_vz'].astype(np.float32); ex_vv=vox['ex_vv'].astype(np.float32)
iv_vx=vox['iv_vx'].astype(np.float32); iv_vy=vox['iv_vy'].astype(np.float32)
iv_vz=vox['iv_vz'].astype(np.float32); iv_vv=vox['iv_vv'].astype(np.float32)
ex_vn=(ex_vv-ex_vv.min())/(ex_vv.max()-ex_vv.min()+1e-8)
iv_vn=(iv_vv-iv_vv.min())/(iv_vv.max()-iv_vv.min()+1e-8)
cx_c=(ex_vx.max()+ex_vx.min())*0.5; cy_c=(ex_vy.max()+ex_vy.min())*0.5
cz_c=(ex_vz.max()+ex_vz.min())*0.5
ex_x=ex_vx-cx_c; ex_y=ex_vy-cy_c; ex_z=ex_vz-cz_c
iv_x=iv_vx-cx_c; iv_y=iv_vy-cy_c; iv_z=iv_vz-cz_c
data_span=max(ex_vx.max()-ex_vx.min(),ex_vy.max()-ex_vy.min())
SCALE_FULL=int(W*0.82/data_span)
SCALE_HALF=int((W//2)*0.80/data_span)
print(f"  ex={len(ex_x)} iv={len(iv_x)}")

print("Loading patch strip + cell info…")
patch_strip=cv2.imread(f'{BASE}/3d_viewer/patch_strip_v4.png')
with open(f'{BASE}/3d_viewer/cell_info_v4.json') as f:
    cell_info=[json.loads(x) if isinstance(x,str) else x for x in json.load(f)]
PATCH_SZ=80

# All landmarks for per-cell calcium crops (use PKL row4_1 only here)
all_lm_jy=[(int(round(row[0])),int(round(row[1]))) for row in src2d]  # (y,x)

# Build per-tile landmark map for any future use
tile_lm_map={}
for lf in sorted(glob.glob(f'{BASE}/registration_video/landmarks_nd2_native_*.npz')):
    tile=os.path.basename(lf).replace('landmarks_nd2_native_','').replace('.npz','')
    if tile not in SKIP_TILES: tile_lm_map[tile]=lf

APPROVED_CELLS=[49,105,100,108,44,23,20,0]
dive_cells=APPROVED_CELLS[:3]
print(f"  Strip cells: {APPROVED_CELLS}  Dive: {dive_cells}")

# ═════════════════════════════════════════════════════════════════════════════
# PRECOMPUTE  ─ nd2 crop, panels, PKL warp maps
# ═════════════════════════════════════════════════════════════════════════════
PAD_ND2=150
y0_c=max(0,int(tgt2d[:,0].min())-PAD_ND2)
y1_c=min(nd2_r4_mip_full.shape[0],int(tgt2d[:,0].max())+PAD_ND2)
x0_c=max(0,int(tgt2d[:,1].min())-PAD_ND2)
x1_c=min(nd2_r4_mip_full.shape[1],int(tgt2d[:,1].max())+PAD_ND2)
nd2_crop=nd2_r4_mip_full[y0_c:y1_c, x0_c:x1_c]
print(f"  nd2 crop: {nd2_crop.shape}  y=[{y0_c},{y1_c}] x=[{x0_c},{x1_c}]")

# ── Panel layout (S3 arrows) ──────────────────────────────────────────────────
PANEL_W,PANEL_H=840,840
IV_CX=W//4; EV_CX=3*W//4; CY=H//2

iv_panel=fit_into(jy306_mip_u8,PANEL_W,PANEL_H)
ev_panel=fit_into(nd2_crop,     PANEL_W,PANEL_H)

# Pixel-coord transforms: JY306(y,x) → frame(x,y)
H_jy,W_jy=658,629
sc_iv=min(PANEL_W/W_jy,PANEL_H/H_jy)
nw_iv,nh_iv=int(W_jy*sc_iv),int(H_jy*sc_iv)
xo_iv=(PANEL_W-nw_iv)//2; yo_iv=(PANEL_H-nh_iv)//2
# frame origin of iv image top-left
iv_tl_x=IV_CX-PANEL_W//2+xo_iv; iv_tl_y=CY-PANEL_H//2+yo_iv

def jy_to_frame(y,x):
    return (x*sc_iv+iv_tl_x, y*sc_iv+iv_tl_y)

H_nd2c,W_nd2c=nd2_crop.shape[:2]
sc_ev=min(PANEL_W/W_nd2c,PANEL_H/H_nd2c)
nw_ev,nh_ev=int(W_nd2c*sc_ev),int(H_nd2c*sc_ev)
xo_ev=(PANEL_W-nw_ev)//2; yo_ev=(PANEL_H-nh_ev)//2
ev_tl_x=EV_CX-PANEL_W//2+xo_ev; ev_tl_y=CY-PANEL_H//2+yo_ev

def nd2_to_frame(y_nd2,x_nd2):
    yc=y_nd2-y0_c; xc=x_nd2-x0_c
    return (xc*sc_ev+ev_tl_x, yc*sc_ev+ev_tl_y)

# ── PKL warp maps (S4) ────────────────────────────────────────────────────────
# Working in nd2 crop display space (800×800)
WARP_SZ=800
sc_nd2w=WARP_SZ/max(H_nd2c,W_nd2c)
W_nd2w=int(W_nd2c*sc_nd2w); H_nd2w=int(H_nd2c*sc_nd2w)
nd2_warp_disp=cv2.resize(nd2_crop,(W_nd2w,H_nd2w),interpolation=cv2.INTER_LANCZOS4)

sc_jyw=WARP_SZ/max(H_jy,W_jy)
W_jyw=int(W_jy*sc_jyw); H_jyw=int(H_jy*sc_jyw)
jy_warp_disp=cv2.resize(jy306_mip_u8,(W_jyw,H_jyw),interpolation=cv2.INTER_LANCZOS4)

print("Precomputing PKL warp maps (RBF)…")
# Evaluate on H_nd2w × W_nd2w grid
ygrid_w=np.arange(H_nd2w,dtype=np.float32)
xgrid_w=np.arange(W_nd2w,dtype=np.float32)
yg_w,xg_w=np.meshgrid(ygrid_w,xgrid_w,indexing='ij')

# Convert display coords → nd2 native
y_nd2n=yg_w/sc_nd2w+y0_c
x_nd2n=xg_w/sc_nd2w+x0_c

# Affine inverse: nd2_native → JY306  (M2d maps JY306→nd2; invert)
# M2d @ [jy_y, jy_x, 1]^T = [nd2_y, nd2_x]
# Fit least-squares inverse from landmarks: tgt2d → src2d
tgt_h=np.concatenate([tgt2d,np.ones((len(tgt2d),1))],axis=1)
M_inv_aff,_,_,_=np.linalg.lstsq(tgt_h,src2d,rcond=None)
pts_flat=np.stack([y_nd2n.ravel(),x_nd2n.ravel(),np.ones(y_nd2n.size)],axis=1)
jy_aff=pts_flat@M_inv_aff      # (N,2) JY306 native
map_y_aff=(jy_aff[:,0]*sc_jyw).reshape(H_nd2w,W_nd2w).astype(np.float32)
map_x_aff=(jy_aff[:,1]*sc_jyw).reshape(H_nd2w,W_nd2w).astype(np.float32)

# RBF (thin-plate-spline): tgt2d → src2d (inverse, nd2 native → JY306 native)
rbf=RBFInterpolator(tgt2d,src2d,kernel='thin_plate_spline',smoothing=0)
pts_nd2_eval=np.stack([y_nd2n.ravel(),x_nd2n.ravel()],axis=1)
jy_rbf=rbf(pts_nd2_eval)       # (N,2) JY306 native
map_y_rbf=(jy_rbf[:,0]*sc_jyw).reshape(H_nd2w,W_nd2w).astype(np.float32)
map_x_rbf=(jy_rbf[:,1]*sc_jyw).reshape(H_nd2w,W_nd2w).astype(np.float32)

dy_warp=map_y_rbf-map_y_aff
dx_warp=map_x_rbf-map_x_aff
print(f"  warp maps: {H_nd2w}×{W_nd2w}  disp magnitude max={np.sqrt(dx_warp**2+dy_warp**2).max():.1f}px")

# Pre-render all 21 warp frames (t=0…1)
T_WARP=np.linspace(0,1,21)
warp_frames=[]
for t in T_WARP:
    mx=(map_x_aff+t*dx_warp).astype(np.float32)
    my=(map_y_aff+t*dy_warp).astype(np.float32)
    w=cv2.remap(jy_warp_disp,mx,my,cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    warp_frames.append(w)
print(f"  {len(warp_frames)} warp frames precomputed")

# Warp display position in full frame
WARP_CX=W//2; WARP_CY=H//2-20
WARP_X0=WARP_CX-W_nd2w//2; WARP_Y0=WARP_CY-H_nd2w//2

# ── Select N_ARROWS landmarks spaced across the image ─────────────────────────
N_ARROWS=10
# pick by spread: use the ones with highest spread (sort by y then x interleaved)
_order=np.argsort(src2d[:,0])  # sort by JY306 y
arrow_idx=_order[np.linspace(0,len(_order)-1,N_ARROWS).astype(int)]

# ── For zoom panels in S3: per-landmark crops ─────────────────────────────────
ZOOM_SZ=180   # each side of zoom panel
ZOOM_PAD=35   # neighbourhood radius in source pixels

def make_zoom_pair(lm_idx):
    """Return (jy_crop, nd2_crop) each at ZOOM_SZ×ZOOM_SZ BGR."""
    sy,sx=src2d[lm_idx]
    ty,tx=tgt2d[lm_idx]
    sy,sx=int(sy),int(sx)
    ty,tx=int(ty),int(tx)
    # JY306 crop
    jy_src=jy306_mip_u8
    r=ZOOM_PAD
    jy_y0=max(0,sy-r);jy_y1=min(jy_src.shape[0],sy+r)
    jy_x0=max(0,sx-r);jy_x1=min(jy_src.shape[1],sx+r)
    jy_crop=cv2.resize(cv2.cvtColor(jy_src[jy_y0:jy_y1,jy_x0:jy_x1],
                                    cv2.COLOR_GRAY2BGR),(ZOOM_SZ,ZOOM_SZ),
                       interpolation=cv2.INTER_LANCZOS4)
    cv2.circle(jy_crop,(ZOOM_SZ//2,ZOOM_SZ//2),6,(0,255,0),-1,cv2.LINE_AA)
    # nd2 crop (from full nd2 image)
    nd2_src=nd2_r4_mip_full
    nd2_y0=max(0,ty-r);nd2_y1=min(nd2_src.shape[0],ty+r)
    nd2_x0=max(0,tx-r);nd2_x1=min(nd2_src.shape[1],tx+r)
    nd2_crop_z=cv2.resize(cv2.cvtColor(nd2_src[nd2_y0:nd2_y1,nd2_x0:nd2_x1],
                                       cv2.COLOR_GRAY2BGR),(ZOOM_SZ,ZOOM_SZ),
                          interpolation=cv2.INTER_LANCZOS4)
    cv2.circle(nd2_crop_z,(ZOOM_SZ//2,ZOOM_SZ//2),6,(0,255,0),-1,cv2.LINE_AA)
    return jy_crop, nd2_crop_z

zoom_pairs=[make_zoom_pair(i) for i in arrow_idx]

# ═════════════════════════════════════════════════════════════════════════════
# VIDEO WRITER
# ═════════════════════════════════════════════════════════════════════════════
vw=cv2.VideoWriter(TMP,cv2.VideoWriter_fourcc(*'mp4v'),FPS,(W,H))

# ─────────────────────────────────────────────────────────────────────────────
# S1  Histmatch calcium movie  (120fr, 5s)
# ─────────────────────────────────────────────────────────────────────────────
print("S1: histmatch calcium movie…")
MOV_STEP=max(1,n_movie//120)
for fi in range(120):
    frame=np.zeros((H,W,3),np.uint8)
    idx=(fi*MOV_STEP)%n_movie
    gray=norm_u8(movie_frames[idx].astype(np.float32))
    zoom=1.15-0.15*(fi/119)
    h0,w0=gray.shape; nw0,nh0=int(w0/zoom),int(h0/zoom)
    x00,y00=(w0-nw0)//2,(h0-nh0)//2
    crop=cv2.resize(gray[y00:y00+nh0,x00:x00+nw0],(w0,h0),interpolation=cv2.INTER_LANCZOS4)
    place(frame,fit_into(crop,860,860),H//2,W//2)
    if fi>=70: caption(frame,'IN VIVO  TWO-PHOTON  CALCIUM  IMAGING',alpha=ease((fi-70)/25))
    vw.write(frame)
save_qc(frame,'s1_calcium_movie')

# ─────────────────────────────────────────────────────────────────────────────
# S2  Freeze → max-proj → alignment overlay (IV magenta | JY306 z=3 green)
# (96fr, 4s)
# ─────────────────────────────────────────────────────────────────────────────
print("S2: freeze → max-proj → alignment overlay…")
last_gray=norm_u8(movie_frames[(119*MOV_STEP)%n_movie].astype(np.float32))
jy_z3_disp=cv2.resize(jy306_z3_u8,(860,860),interpolation=cv2.INTER_LANCZOS4)
mov_max_disp=cv2.resize(movie_max,(860,860),interpolation=cv2.INTER_LANCZOS4)

for fi in range(96):
    frame=np.zeros((H,W,3),np.uint8)
    t_freeze=ease(fi/30)        # 0→1 : movie frame → max-proj
    t_overlay=ease((fi-45)/35)  # 0→1 : max-proj → overlay

    if t_overlay < 0.05:
        # show max-proj blending in
        img_b=blend(last_gray,movie_max,t_freeze)
        place(frame,fit_into(img_b,860,860),H//2,W//2)
    else:
        # overlay: IV in magenta channel, JY306 z=3 in green channel
        ov=np.zeros((860,860,3),np.uint8)
        ov[:,:,1]=np.clip(jy_z3_disp.astype(np.float32)*t_overlay,0,255).astype(np.uint8)
        ov[:,:,2]=np.clip(mov_max_disp.astype(np.float32)*t_overlay,0,255).astype(np.uint8)
        ov[:,:,0]=np.clip(mov_max_disp.astype(np.float32)*t_overlay,0,255).astype(np.uint8)
        place(frame,ov,H//2,W//2)

    caption(frame,'IN VIVO  TWO-PHOTON  CALCIUM  IMAGING',
            alpha=max(0,1-ease(fi/25)))
    if t_overlay>0.1:
        small_label(frame,'IN VIVO  CALCIUM  MAX-PROJ',CY+440,W//2-160,
                    (80,80,200),min(1,t_overlay*2),0.40)
        small_label(frame,'JY306  z=3  CONFOCAL',CY+460,W//2-160,
                    (80,200,80),min(1,t_overlay*2),0.40)
    caption(frame,'IN VIVO  ->  CONFOCAL  ALIGNMENT',
            alpha=ease((fi-55)/25))
    vw.write(frame)
save_qc(frame,'s2_alignment_overlay')

# ─────────────────────────────────────────────────────────────────────────────
# S3  Green arrows one-by-one + zoom panels  (300fr, 12.5s)
# ─────────────────────────────────────────────────────────────────────────────
print("S3: green arrows + zoom panels…")
FRAMES_PER_ARROW=300//N_ARROWS   # 30fr per arrow

for fi in range(300):
    frame=np.zeros((H,W,3),np.uint8)
    # draw base panels
    place(frame,iv_panel,CY,IV_CX)
    place(frame,ev_panel,CY,EV_CX)
    small_label(frame,'IN VIVO  JY306',CY-PANEL_H//2-18,IV_CX-PANEL_W//2+4,
                WHITE,1.0,0.40)
    small_label(frame,'EX VIVO  CONFOCAL',CY-PANEL_H//2-18,EV_CX-PANEL_W//2+4,
                WHITE,1.0,0.40)
    # divider line
    cv2.line(frame,(W//2,80),(W//2,H-60),(80,80,80),1,cv2.LINE_AA)

    n_shown=min(N_ARROWS,(fi//FRAMES_PER_ARROW)+1)
    for k in range(n_shown):
        lm_i=arrow_idx[k]
        jy_pt=jy_to_frame(src2d[lm_i,0],src2d[lm_i,1])
        ev_pt=nd2_to_frame(tgt2d[lm_i,0],tgt2d[lm_i,1])
        # alpha for this arrow
        if k < fi//FRAMES_PER_ARROW:
            ak=1.0
        else:
            ak=ease((fi%FRAMES_PER_ARROW)/15)
        # dot + arrow
        cv2.circle(frame,(int(jy_pt[0]),int(jy_pt[1])),5,
                   tuple(int(v*ak) for v in GREEN),-1,cv2.LINE_AA)
        cv2.circle(frame,(int(ev_pt[0]),int(ev_pt[1])),5,
                   tuple(int(v*ak) for v in GREEN),-1,cv2.LINE_AA)
        draw_arrow(frame,jy_pt,ev_pt,(0,200,0),ak,thickness=2,tip=8)

    # zoom panel for most recent arrow
    cur_k=min(N_ARROWS-1,fi//FRAMES_PER_ARROW)
    zoom_a=ease((fi%FRAMES_PER_ARROW)/12)
    jz,ez=zoom_pairs[cur_k]
    ZPAN_W=ZOOM_SZ*2+8; ZPAN_H=ZOOM_SZ+28
    zpan=np.zeros((ZPAN_H,ZPAN_W,3),np.uint8)
    zpan[24:24+ZOOM_SZ,0:ZOOM_SZ]=jz
    zpan[24:24+ZOOM_SZ,ZOOM_SZ+8:]=ez
    cv2.putText(zpan,'IN VIVO',(2,16),FONT,0.35,(200,200,200),1,cv2.LINE_AA)
    cv2.putText(zpan,'EX VIVO',(ZOOM_SZ+10,16),FONT,0.35,(200,200,200),1,cv2.LINE_AA)
    # place zoom panel bottom-centre
    zx=W//2-ZPAN_W//2; zy=H-ZPAN_H-14
    zpan_alpha=np.clip(zpan.astype(np.float32)*zoom_a,0,255).astype(np.uint8)
    frame[zy:zy+ZPAN_H,zx:zx+ZPAN_W]=zpan_alpha

    caption(frame,'LANDMARK  CORRESPONDENCE  (PKL)',
            alpha=min(1,fi/20))
    vw.write(frame)
    if fi==0:  save_qc(frame,'s3_arrows_start')
save_qc(frame,'s3_arrows_end')

# ─────────────────────────────────────────────────────────────────────────────
# S4  Progressive PKL warp  (144fr, 6s)
# Ramps from affine (t=0) to full RBF (t=1) over 96fr, then holds 48fr
# ─────────────────────────────────────────────────────────────────────────────
print("S4: progressive PKL warp…")

nd2_warp_bgr=cv2.cvtColor(nd2_warp_disp,cv2.COLOR_GRAY2BGR)

for fi in range(144):
    frame=np.zeros((H,W,3),np.uint8)

    if fi < 96:
        t=ease(fi/95)
    else:
        t=1.0

    # pick pre-rendered warp frame (index 0..20)
    wf_idx=int(round(t*20))
    warped=warp_frames[wf_idx]

    # Compose: nd2 in green, JY306 warped in red+blue (magenta)
    canvas=np.zeros((H_nd2w,W_nd2w,3),np.uint8)
    canvas[:,:,1]=np.clip(nd2_warp_disp.astype(np.float32)*0.80,0,255).astype(np.uint8)
    canvas[:,:,2]=np.clip(warped.astype(np.float32)*0.95,0,255).astype(np.uint8)
    canvas[:,:,0]=np.clip(warped.astype(np.float32)*0.95,0,255).astype(np.uint8)

    # Place in frame
    frame[WARP_Y0:WARP_Y0+H_nd2w,WARP_X0:WARP_X0+W_nd2w]=canvas

    # legend
    cv2.circle(frame,(28,28),7,(80,200,80),-1,cv2.LINE_AA)
    small_label(frame,'EX VIVO  CONFOCAL',32,42,(80,200,80),1.0,0.37)
    cv2.circle(frame,(28,52),7,(200,80,200),-1,cv2.LINE_AA)
    small_label(frame,'IN VIVO  CALCIUM  (WARPING)',56,42,(200,80,200),1.0,0.37)

    # warp progress bar
    bar_w=int(t*(W_nd2w-4)); bar_y=WARP_Y0+H_nd2w+8
    cv2.rectangle(frame,(WARP_X0,bar_y),(WARP_X0+W_nd2w-1,bar_y+6),(60,60,60),-1)
    if bar_w>0:
        cv2.rectangle(frame,(WARP_X0,bar_y),(WARP_X0+bar_w,bar_y+6),(0,200,0),-1)

    caption(frame,'PKL  NON-RIGID  REGISTRATION',alpha=1.0)
    if fi<10: caption(frame,'AFFINE  ALIGNMENT',alpha=ease((10-fi)/8))
    if fi>85: caption(frame,'FULLY  REGISTERED',alpha=ease((fi-85)/15))
    vw.write(frame)
    if fi==0:  save_qc(frame,'s4_warp_t0')
    if fi==47: save_qc(frame,'s4_warp_t05')
    if fi==95: save_qc(frame,'s4_warp_t1')

# ─────────────────────────────────────────────────────────────────────────────
# S4b  Red+green=yellow overlay  (72fr, 3s)
# ─────────────────────────────────────────────────────────────────────────────
print("S4b: yellow overlay…")
# Build overlay: nd2 in green, IV fully warped in red → yellow = matched
canvas_ov=np.zeros((H_nd2w,W_nd2w,3),np.uint8)
canvas_ov[:,:,1]=nd2_warp_disp.astype(np.uint8)
canvas_ov[:,:,2]=warp_frames[-1]
canvas_ov[:,:,0]=warp_frames[-1]

for fi in range(72):
    frame=np.zeros((H,W,3),np.uint8)
    a_yellow=ease(fi/30)
    # transition from warp-t1 to combined overlay
    warp_canvas=np.zeros((H_nd2w,W_nd2w,3),np.uint8)
    warp_canvas[:,:,1]=np.clip(nd2_warp_disp.astype(np.float32)*0.80,0,255).astype(np.uint8)
    warp_canvas[:,:,2]=np.clip(warp_frames[-1].astype(np.float32)*0.95,0,255).astype(np.uint8)
    warp_canvas[:,:,0]=np.clip(warp_frames[-1].astype(np.float32)*0.95,0,255).astype(np.uint8)
    blended=blend(warp_canvas,canvas_ov,a_yellow)
    frame[WARP_Y0:WARP_Y0+H_nd2w,WARP_X0:WARP_X0+W_nd2w]=blended

    cv2.circle(frame,(28,28),7,(80,200,80),-1,cv2.LINE_AA)
    small_label(frame,'EX VIVO',32,42,(80,200,80),1.0,0.37)
    cv2.circle(frame,(28,52),7,(200,80,200),-1,cv2.LINE_AA)
    small_label(frame,'IN VIVO',56,42,(200,80,200),1.0,0.37)
    cv2.circle(frame,(28,76),7,(100,200,200),-1,cv2.LINE_AA)
    small_label(frame,'MATCHED  CELLS',80,42,(100,200,200),a_yellow,0.37)

    caption(frame,'MATCHED  NEURONS  —  IN VIVO  +  EX VIVO',
            alpha=ease((fi-20)/25))
    vw.write(frame)
save_qc(frame,'s4b_yellow_overlay')

# ─────────────────────────────────────────────────────────────────────────────
# S5  3D transition (336fr, 14s)
#  0–60  : 2D panels dim, 3D clouds emerge on each panel
#  60–150: clouds fully materialise
#  150–210: 2D fades, clouds slide to centre
#  210–336: merged 3D cloud rotates to show depth
# ─────────────────────────────────────────────────────────────────────────────
print("S5: 3D transition…")
BASE_ROT_X=math.radians(10); BASE_ROT_Y=math.radians(5)

for fi in range(336):
    frame=np.zeros((H,W,3),np.uint8)

    if fi < 150:
        t_cloud=ease(fi/120); t_dim=ease(fi/140)
        dim_a=max(0.10,1.0-t_dim*0.90)
        place(frame,np.clip(iv_panel.astype(np.float32)*dim_a,0,255).astype(np.uint8),CY,IV_CX)
        place(frame,np.clip(ev_panel.astype(np.float32)*dim_a,0,255).astype(np.uint8),CY,EV_CX)
        cloud=render_clouds(
            ex_x,ex_y,ex_z,ex_vn, iv_x,iv_y,iv_z,iv_vn,
            BASE_ROT_Y,BASE_ROT_X, t_cloud,t_cloud,
            W,H,SCALE_HALF,
            ex_cx=EV_CX,ex_cy=CY, iv_cx=IV_CX,iv_cy=CY)
        frame=cv2.add(frame,cloud)
        small_label(frame,'IN VIVO  3D',CY-PANEL_H//2-18,IV_CX-PANEL_W//2+4,
                    tuple(int(v) for v in IV_COL),min(1,t_cloud*2))
        small_label(frame,'EX VIVO  3D',CY-PANEL_H//2-18,EV_CX-PANEL_W//2+4,
                    tuple(int(v) for v in EX_COL),min(1,t_cloud*2))
        caption(frame,'MATCHED  NEURONS  —  IN VIVO  +  EX VIVO',
                alpha=max(0,1-ease(fi/20)))
        caption(frame,'3D  TISSUE  VOLUMES',alpha=ease((fi-30)/40))

    elif fi < 210:
        t_merge=ease((fi-150)/55)
        iv_cx_now=int(IV_CX+t_merge*(W//2-IV_CX))
        ev_cx_now=int(EV_CX+t_merge*(W//2-EV_CX))
        scale_now=int(SCALE_HALF+t_merge*(SCALE_FULL-SCALE_HALF))
        cloud=render_clouds(
            ex_x,ex_y,ex_z,ex_vn, iv_x,iv_y,iv_z,iv_vn,
            BASE_ROT_Y,BASE_ROT_X, 1.0,1.0,
            W,H,scale_now,
            ex_cx=ev_cx_now,ex_cy=CY, iv_cx=iv_cx_now,iv_cy=CY)
        frame[:]=cloud
        caption(frame,'3D  TISSUE  VOLUMES',alpha=max(0,1-ease((fi-150)/25)))
        caption(frame,'MULTIMODAL  REGISTRATION',alpha=ease((fi-165)/30))

    else:
        t_rot=(fi-210)/125
        rot_y=math.radians(5+25*math.sin(t_rot*math.pi))
        rot_x=BASE_ROT_X+math.radians(8*math.sin(t_rot*math.pi*0.5))
        cloud=render_clouds(
            ex_x,ex_y,ex_z,ex_vn, iv_x,iv_y,iv_z,iv_vn,
            rot_y,rot_x, 1.0,1.0, W,H,SCALE_FULL)
        frame[:]=cloud
        cv2.circle(frame,(28,28),7,tuple(int(v) for v in EX_COL),-1,cv2.LINE_AA)
        small_label(frame,'EX VIVO  CONFOCAL',32,42,tuple(int(v) for v in EX_COL))
        cv2.circle(frame,(28,52),7,tuple(int(v) for v in IV_COL),-1,cv2.LINE_AA)
        small_label(frame,'IN VIVO  CALCIUM',56,42,tuple(int(v) for v in IV_COL))
        caption(frame,'MULTIMODAL  REGISTRATION',alpha=1.0)

    vw.write(frame)
    if fi==59:  save_qc(frame,'s5a_cloud_emerge')
    if fi==149: save_qc(frame,'s5b_cloud_full')
    if fi==209: save_qc(frame,'s5c_cloud_centre')
    if fi==335: save_qc(frame,'s5d_cloud_rotated')
    if fi%48==0: print(f'  S5 {fi}/336')

# ─────────────────────────────────────────────────────────────────────────────
# S6  8-cell strip  4×4  (192fr, 8s)
# ─────────────────────────────────────────────────────────────────────────────
print("S6: 8-cell strip 4×4…")
CS_CELL=160; CS_GAP=14; CS_COLS=4
CS_TW=CS_COLS*(CS_CELL+CS_GAP)-CS_GAP
CS_TH=4*(CS_CELL+CS_GAP)-CS_GAP
CS_X0=(W-CS_TW)//2; CS_Y0=(H-CS_TH)//2-20

ev_strip=[cv2.resize(patch_strip[ci*PATCH_SZ:(ci+1)*PATCH_SZ,0:PATCH_SZ],
                     (CS_CELL,CS_CELL),interpolation=cv2.INTER_LANCZOS4)
          for ci in APPROVED_CELLS]
ivw_strip=[cv2.resize(patch_strip[ci*PATCH_SZ:(ci+1)*PATCH_SZ,PATCH_SZ:PATCH_SZ*2],
                      (CS_CELL,CS_CELL),interpolation=cv2.INTER_LANCZOS4)
           for ci in APPROVED_CELLS]

for fi in range(192):
    frame=np.zeros((H,W,3),np.uint8)
    for k in range(8):
        col_k=k%CS_COLS; group=k//CS_COLS
        x0=CS_X0+col_k*(CS_CELL+CS_GAP)
        ey0=CS_Y0+group*2*(CS_CELL+CS_GAP)
        iy0=ey0+CS_CELL+CS_GAP
        at=ease((fi-k*16)/22)
        if at<=0: continue
        col_c=CELL_COLS[k]
        frame[ey0:ey0+CS_CELL,x0:x0+CS_CELL]=\
            np.clip(ev_strip[k].astype(np.float32)*at,0,255).astype(np.uint8)
        frame[iy0:iy0+CS_CELL,x0:x0+CS_CELL]=\
            np.clip(ivw_strip[k].astype(np.float32)*at,0,255).astype(np.uint8)
        cv2.rectangle(frame,(x0,ey0),(x0+CS_CELL-1,ey0+CS_CELL-1),
                      tuple(int(v*at) for v in col_c),1)
        cv2.rectangle(frame,(x0,iy0),(x0+CS_CELL-1,iy0+CS_CELL-1),
                      tuple(int(v*at*0.5) for v in col_c),1)
        cv2.putText(frame,str(k+1),(x0+4,ey0-5),FONT,0.45,
                    tuple(int(v*at) for v in col_c),1,cv2.LINE_AA)
    all_t=ease((fi-7*16)/20)
    if all_t>0:
        for row_i,(label,) in enumerate([('EX VIVO',),('IN VIVO',),('EX VIVO',),('IN VIVO',)]):
            small_label(frame,label,CS_Y0+row_i*(CS_CELL+CS_GAP)+CS_CELL//2,
                        CS_X0-82,WHITE,all_t,0.37)
    caption(frame,'MULTIMODAL  REGISTRATION',alpha=max(0,1-ease(fi/20)))
    caption(frame,'MATCHED  NEURONS',alpha=ease((fi-130)/30))
    vw.write(frame)
save_qc(frame,'s6_cell_strip')

# ─────────────────────────────────────────────────────────────────────────────
# S7-S9  Per-cell deep-dive  (3 × 216fr, 3 × 9s)
# 3 panels: calcium movie (live) | EV confocal (col 0) | IV warped (col 1)
# ─────────────────────────────────────────────────────────────────────────────
print("S7-S9: per-cell panels…")
PANEL_SZ=320; PANEL_GAP=40
PX1=(W-3*PANEL_SZ-2*PANEL_GAP)//2
PX2=PX1+PANEL_SZ+PANEL_GAP
PX3=PX2+PANEL_SZ+PANEL_GAP
PY=(H-PANEL_SZ)//2-20
CROP_R=45

for ci_idx,ci in enumerate(dive_cells):
    # calcium crop centred on the landmark for this cell
    lm_idx=min(ci,len(all_lm_jy)-1)
    lm_y,lm_x=all_lm_jy[lm_idx]
    col=CELL_COLS[ci_idx]
    ev_p=cv2.resize(patch_strip[ci*PATCH_SZ:(ci+1)*PATCH_SZ,0:PATCH_SZ],
                    (PANEL_SZ,PANEL_SZ),interpolation=cv2.INTER_LANCZOS4)
    iv_p=cv2.resize(patch_strip[ci*PATCH_SZ:(ci+1)*PATCH_SZ,PATCH_SZ:PATCH_SZ*2],
                    (PANEL_SZ,PANEL_SZ),interpolation=cv2.INTER_LANCZOS4)

    for fi in range(216):
        frame=np.zeros((H,W,3),np.uint8)
        a_in=ease(fi/25)
        mov_idx=int(fi/215*(n_movie-1))
        mov_f=movie_frames[mov_idx]
        y0m=max(0,lm_y-CROP_R); y1m=min(mov_f.shape[0],lm_y+CROP_R)
        x0m=max(0,lm_x-CROP_R); x1m=min(mov_f.shape[1],lm_x+CROP_R)
        crop=mov_f[y0m:y1m,x0m:x1m]
        sq=np.zeros((CROP_R*2,CROP_R*2),np.uint8)
        sq[:crop.shape[0],:crop.shape[1]]=crop
        ca_panel=cv2.cvtColor(cv2.resize(norm_u8(sq.astype(np.float32)),
                               (PANEL_SZ,PANEL_SZ),interpolation=cv2.INTER_LANCZOS4),
                              cv2.COLOR_GRAY2BGR)
        ch=PANEL_SZ//2
        cv2.line(ca_panel,(ch-14,ch),(ch+14,ch),(160,160,160),1)
        cv2.line(ca_panel,(ch,ch-14),(ch,ch+14),(160,160,160),1)

        for panel,px,label in [(ca_panel,PX1,'IN VIVO  CALCIUM  MOVIE'),
                                (ev_p,    PX2,'EX VIVO  CONFOCAL'),
                                (iv_p,    PX3,'IN VIVO  WARPED')]:
            frame[PY:PY+PANEL_SZ,px:px+PANEL_SZ]=\
                np.clip(panel.astype(np.float32)*a_in,0,255).astype(np.uint8)
            cv2.rectangle(frame,(px,PY),(px+PANEL_SZ-1,PY+PANEL_SZ-1),
                          tuple(int(v*a_in) for v in WHITE),1)
            small_label(frame,label,PY-18,px,GRAY,a_in,0.40)

        cv2.putText(frame,f'CELL  {ci_idx+1}',(14,34),FONT,0.55,
                    tuple(int(v*a_in) for v in col),1,cv2.LINE_AA)
        caption(frame,'FUNCTIONAL  /  STRUCTURAL  READOUT',alpha=a_in)
        vw.write(frame)

    for _ in range(12): vw.write(np.zeros((H,W,3),np.uint8))
    save_qc(frame,f's{7+ci_idx}_cell{ci_idx+1}')
    print(f"  Cell {ci_idx+1} done (ci={ci})")

# final hold
for _ in range(24): vw.write(np.zeros((H,W,3),np.uint8))
vw.release()
print(f"Raw: {TMP}")

print("Re-encoding H.264…")
subprocess.run(['ffmpeg','-y','-i',TMP,
                '-vcodec','libx264','-pix_fmt','yuv420p','-crf','18','-preset','fast',
                OUT],check=True)
total_fr=120+96+300+144+72+336+192+3*216+3*12+24
print(f"\n✓  {OUT}  ({total_fr/FPS:.0f}s @ {FPS}fps)")
