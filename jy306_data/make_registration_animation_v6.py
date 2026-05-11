#!/usr/bin/env python3
"""
Registration animation v6 — v4 as base, fixed transitions, new cells.

  S1  (192fr,  8s) : Calcium movie playing
  S2  ( 96fr,  4s) : Freeze → max-proj
  S3  (120fr,  5s) : 2D side-by-side: JY306 MIP (left) | nd2 MIP (right) + landmarks
  S4  (336fr, 14s) : 3D transition:
                       0–60   : 2D panels visible (dim)
                       60–150 : 3D clouds emerge on top of each panel
                       150–210: 2D fades, clouds slide to centre
                       210–336: merged 3D cloud rotates
  S5  (192fr,  8s) : 8-cell strip  4 cols × 4 rows (EV/IV)
  S6  (216fr,  9s) : Per-cell deep-dive  Cell 1 (calcium video | EV | IV-warp)
  S7  (216fr,  9s) : Per-cell deep-dive  Cell 2
  S8  (216fr,  9s) : Per-cell deep-dive  Cell 3
  Total ≈ 66s @ 24fps

QC frames → png_exports/registration_animation_v6_qc/
"""
import numpy as np, cv2, tifffile, json, os, glob, math, subprocess

BASE = '/Users/neurolab/neuroinformatics/margaret'
W, H = 1920, 1080
FPS  = 24
TMP  = f'{BASE}/png_exports/registration_animation_v6_raw.mp4'
OUT  = f'{BASE}/png_exports/registration_animation_v6.mp4'
QC   = f'{BASE}/png_exports/registration_animation_v6_qc'
os.makedirs(f'{BASE}/png_exports', exist_ok=True)
os.makedirs(QC, exist_ok=True)

FONT  = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255,255,255); GRAY = (160,160,160)
EX_COL = np.array([0, 200, 0],   np.float32)   # green  BGR
IV_COL = np.array([200, 0, 200], np.float32)   # magenta BGR
CELL_COLS = [
    (  0,   0,220),(  0,110,255),(  0,220,220),(60,200,60),
    (220,200,  0),(220, 80,  0),(180,  0,180),(255,255,  0),
]
SKIP_TILES = {'row3_1','row3_5'}

# ── helpers ──────────────────────────────────────────────────────────────────
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

# ── 3D point cloud renderer ───────────────────────────────────────────────────
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
print("Loading calcium movie…")
cap=cv2.VideoCapture(f'{BASE}/png_exports/native_invivo/movie_warped_h264.mp4')
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

print("Loading nd2 row2_1 tile…")
nd2_slices=[]
for zi in range(12):
    p=cv2.imread(f'{BASE}/png_exports/registration_video/row2_1/GFP_z{zi:03d}.png',cv2.IMREAD_UNCHANGED)
    if p is not None: nd2_slices.append(p.astype(np.float32))
nd2_mip_full=norm_u8(np.max(nd2_slices,axis=0))
nd2_crop_u8=nd2_mip_full[1235:2856,688:2377]

print("Loading 3D voxel clouds from viewer_v4…")
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
print(f"  ex={len(ex_x)} iv={len(iv_x)} scale_full={SCALE_FULL}")

print("Loading landmarks…")
lm27=np.load(f'{BASE}/registration_video/landmarks_27_nd2_native.npz')
ev_nd2_27=lm27['ev_nd2']; pcd_iv_27=lm27['pcd_invivo_jy306']

print("Loading patch strip + cell info…")
patch_strip=cv2.imread(f'{BASE}/3d_viewer/patch_strip_v4.png')
with open(f'{BASE}/3d_viewer/cell_info_v4.json') as f:
    cell_info=[json.loads(x) if isinstance(x,str) else x for x in json.load(f)]
PATCH_SZ=80

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

# Selected cells — col 0 = EV MIP, col 1 = IV Warped MIP
APPROVED_CELLS=[49,105,100,108,44,23,20,0]
dive_cells=APPROVED_CELLS[:3]
print(f"  Strip cells: {APPROVED_CELLS}  Dive: {dive_cells}")

# ── 2D panel constants ────────────────────────────────────────────────────────
PANEL_W,PANEL_H=840,840
IV_CX=W//4; EV_CX=3*W//4; CY=H//2
iv_panel=fit_into(jy306_mip_u8, PANEL_W,PANEL_H)
ev_panel=fit_into(nd2_crop_u8,  PANEL_W,PANEL_H)

# Landmark projections
_iv_sc=min(PANEL_W/629,PANEL_H/658)
_iv_xo=(PANEL_W-int(629*_iv_sc))//2; _iv_yo=(PANEL_H-int(658*_iv_sc))//2
_nd2_cw,_nd2_ch=nd2_crop_u8.shape[1],nd2_crop_u8.shape[0]
_ev_sc=min(PANEL_W/_nd2_cw,PANEL_H/_nd2_ch)
_ev_xo=(PANEL_W-int(_nd2_cw*_ev_sc))//2; _ev_yo=(PANEL_H-int(_nd2_ch*_ev_sc))//2

# ═════════════════════════════════════════════════════════════════════════════
# VIDEO WRITER
# ═════════════════════════════════════════════════════════════════════════════
vw=cv2.VideoWriter(TMP,cv2.VideoWriter_fourcc(*'mp4v'),FPS,(W,H))

# ─────────────────────────────────────────────────────────────────────────────
# S1  Calcium movie  (192fr, 8s)
# ─────────────────────────────────────────────────────────────────────────────
print("S1: calcium movie…")
MOV_STEP=max(1,n_movie//192)
for fi in range(192):
    frame=np.zeros((H,W,3),np.uint8)
    idx=(fi*MOV_STEP)%n_movie
    gray=norm_u8(movie_frames[idx].astype(np.float32))
    zoom=1.15-0.15*(fi/191)
    h,w=gray.shape; nw,nh=int(w/zoom),int(h/zoom)
    x0,y0=(w-nw)//2,(h-nh)//2
    crop=cv2.resize(gray[y0:y0+nh,x0:x0+nw],(w,h),interpolation=cv2.INTER_LANCZOS4)
    place(frame,fit_into(crop,860,860),H//2,W//2)
    if fi>=80: caption(frame,'IN VIVO  TWO-PHOTON  CALCIUM  IMAGING',alpha=ease((fi-80)/30))
    vw.write(frame)

# ─────────────────────────────────────────────────────────────────────────────
# S2  Freeze → max-proj  (96fr, 4s)
# ─────────────────────────────────────────────────────────────────────────────
print("S2: freeze → max-proj…")
last_gray=norm_u8(movie_frames[(191*MOV_STEP)%n_movie].astype(np.float32))
circle_cells=[0,5,10,15,20]
for fi in range(96):
    frame=np.zeros((H,W,3),np.uint8)
    img_b=blend(last_gray,movie_max,ease(fi/40))
    sq,sc,yo,xo=fit_into(img_b,860,860),*([None]*3)
    sq,sc,yo,xo=fit_into(img_b,860,860),min(860/movie_max.shape[1],860/movie_max.shape[0]),(860-int(movie_max.shape[0]*min(860/movie_max.shape[1],860/movie_max.shape[0])))//2,(860-int(movie_max.shape[1]*min(860/movie_max.shape[1],860/movie_max.shape[0])))//2
    place(frame,sq,H//2,W//2)
    for k,ci in enumerate(circle_cells):
        ac=ease((fi-k*12-20)/15)
        if ac<=0: continue
        ry=int(pcd_iv_27[ci,1]*sc)+yo+(H//2-430)
        rx=int(pcd_iv_27[ci,2]*sc)+xo+(W//2-430)
        dashed_circle(frame,rx,ry,18,tuple(int(v*ac) for v in WHITE))
    caption(frame,'IN VIVO  TWO-PHOTON  CALCIUM  IMAGING',alpha=max(0,1-ease(fi/20)))
    caption(frame,'NEURONS  RECORDED  DURING  BEHAVIOUR',alpha=ease((fi-30)/25))
    vw.write(frame)
save_qc(frame,'s2_max_proj')

# ─────────────────────────────────────────────────────────────────────────────
# S3  2D side-by-side  (120fr, 5s)
# ─────────────────────────────────────────────────────────────────────────────
print("S3: 2D side-by-side…")
for fi in range(120):
    t_split=ease(fi/50); t_lines=ease((fi-60)/40)
    frame=np.zeros((H,W,3),np.uint8)
    iv_cx=int(W//2+t_split*(IV_CX-W//2))
    ev_cx=int(W//2+t_split*(EV_CX-W//2))
    place(frame,iv_panel,CY,iv_cx)
    if t_split>0.1: place(frame,ev_panel,CY,ev_cx)
    small_label(frame,'IN VIVO  JY306',  CY-PANEL_H//2-18,iv_cx-PANEL_W//2+4,WHITE,min(1,t_split*3))
    small_label(frame,'EX VIVO  CONFOCAL',CY-PANEL_H//2-18,ev_cx-PANEL_W//2+4,WHITE,min(1,(t_split-0.2)*3))
    if t_lines>0 and t_split>0.85:
        for k in range(int(t_lines*7)):
            ry_iv=int(pcd_iv_27[k,1]*_iv_sc)+_iv_yo+(iv_cx-PANEL_W//2)+(CY-PANEL_H//2)
            rx_iv=int(pcd_iv_27[k,2]*_iv_sc)+_iv_xo+(iv_cx-PANEL_W//2)
            # fix: x and y separate
            rx_iv=int(pcd_iv_27[k,2]*_iv_sc)+_iv_xo+(iv_cx-PANEL_W//2)
            ry_iv=int(pcd_iv_27[k,1]*_iv_sc)+_iv_yo+(CY-PANEL_H//2)
            rx_ev=int((ev_nd2_27[k,0]-688)*_ev_sc)+_ev_xo+(ev_cx-PANEL_W//2)
            ry_ev=int((ev_nd2_27[k,1]-1235)*_ev_sc)+_ev_yo+(CY-PANEL_H//2)
            ak=min(1.,t_lines*7-k)
            cv2.circle(frame,(rx_iv,ry_iv),5,tuple(int(v*ak) for v in WHITE),-1,cv2.LINE_AA)
            cv2.circle(frame,(rx_ev,ry_ev),5,tuple(int(v*ak) for v in WHITE),-1,cv2.LINE_AA)
            cv2.line(frame,(rx_iv,ry_iv),(rx_ev,ry_ev),tuple(int(v*ak*0.6) for v in WHITE),1,cv2.LINE_AA)
    caption(frame,'NEURONS  RECORDED  DURING  BEHAVIOUR',alpha=max(0,1-ease(fi/15)))
    caption(frame,'RE-IDENTIFIED  IN  EX-VIVO  TISSUE',  alpha=ease((fi-20)/30))
    vw.write(frame)
save_qc(frame,'s3_2d_sidebyside')

# ─────────────────────────────────────────────────────────────────────────────
# S4  3D transition (336fr, 14s)
#  0–60  : 2D panels dimming, clouds begin to emerge (each on its own side)
#  60–150: clouds fully materialise on each panel
#  150–210: 2D fades to black, clouds slide to centre
#  210–336: full merged cloud, slow Y rotation to show depth
# ─────────────────────────────────────────────────────────────────────────────
print("S4: 3D transition…")
BASE_ROT_X=math.radians(10); BASE_ROT_Y=math.radians(5)

for fi in range(336):
    frame=np.zeros((H,W,3),np.uint8)

    if fi < 150:
        # Phase 1+2: 2D panels dim, 3D clouds emerge over them
        t_cloud=ease(fi/120)
        t_dim  =ease(fi/140)
        dim_a  =max(0.10, 1.0-t_dim*0.90)
        place(frame,np.clip(iv_panel.astype(np.float32)*dim_a,0,255).astype(np.uint8),CY,IV_CX)
        place(frame,np.clip(ev_panel.astype(np.float32)*dim_a,0,255).astype(np.uint8),CY,EV_CX)
        cloud=render_clouds(
            ex_x,ex_y,ex_z,ex_vn, iv_x,iv_y,iv_z,iv_vn,
            BASE_ROT_Y, BASE_ROT_X,
            t_cloud, t_cloud,
            W,H, SCALE_HALF,
            ex_cx=EV_CX, ex_cy=CY, iv_cx=IV_CX, iv_cy=CY)
        frame=cv2.add(frame,cloud)
        small_label(frame,'IN VIVO  3D',CY-PANEL_H//2-18,IV_CX-PANEL_W//2+4,
                    tuple(int(v) for v in IV_COL),min(1,t_cloud*2))
        small_label(frame,'EX VIVO  3D',CY-PANEL_H//2-18,EV_CX-PANEL_W//2+4,
                    tuple(int(v) for v in EX_COL),min(1,t_cloud*2))
        caption(frame,'RE-IDENTIFIED  IN  EX-VIVO  TISSUE',alpha=max(0,1-ease(fi/20)))
        caption(frame,'3D  TISSUE  VOLUMES',                alpha=ease((fi-30)/40))

    elif fi < 210:
        # Phase 3: clouds slide to centre
        t_merge=ease((fi-150)/55)
        iv_cx_now=int(IV_CX+t_merge*(W//2-IV_CX))
        ev_cx_now=int(EV_CX+t_merge*(W//2-EV_CX))
        scale_now =int(SCALE_HALF+t_merge*(SCALE_FULL-SCALE_HALF))
        cloud=render_clouds(
            ex_x,ex_y,ex_z,ex_vn, iv_x,iv_y,iv_z,iv_vn,
            BASE_ROT_Y, BASE_ROT_X,
            1.0, 1.0,
            W,H, scale_now,
            ex_cx=ev_cx_now, ex_cy=CY, iv_cx=iv_cx_now, iv_cy=CY)
        frame[:]=cloud
        caption(frame,'3D  TISSUE  VOLUMES',       alpha=max(0,1-ease((fi-150)/25)))
        caption(frame,'MULTIMODAL  REGISTRATION',   alpha=ease((fi-165)/30))

    else:
        # Phase 4: merged cloud rotates
        t_rot=(fi-210)/125
        rot_y=math.radians(5+25*math.sin(t_rot*math.pi))
        rot_x=BASE_ROT_X+math.radians(8*math.sin(t_rot*math.pi*0.5))
        cloud=render_clouds(
            ex_x,ex_y,ex_z,ex_vn, iv_x,iv_y,iv_z,iv_vn,
            rot_y,rot_x, 1.0,1.0,
            W,H,SCALE_FULL)
        frame[:]=cloud
        cv2.circle(frame,(28,28),7,tuple(int(v) for v in EX_COL),-1,cv2.LINE_AA)
        small_label(frame,'EX VIVO  CONFOCAL',32,42,tuple(int(v) for v in EX_COL))
        cv2.circle(frame,(28,52),7,tuple(int(v) for v in IV_COL),-1,cv2.LINE_AA)
        small_label(frame,'IN VIVO  CALCIUM', 56,42,tuple(int(v) for v in IV_COL))
        caption(frame,'MULTIMODAL  REGISTRATION',alpha=1.0)

    vw.write(frame)
    if fi==59:  save_qc(frame,'s4a_cloud_emerge')
    if fi==149: save_qc(frame,'s4b_cloud_full')
    if fi==209: save_qc(frame,'s4c_cloud_centre')
    if fi==335: save_qc(frame,'s4d_cloud_rotated')
    if fi%48==0: print(f'  S4 {fi}/336')

# ─────────────────────────────────────────────────────────────────────────────
# S5  8-cell strip  4×4  (192fr, 8s)
# ─────────────────────────────────────────────────────────────────────────────
print("S5: 8-cell strip 4×4…")
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
        frame[ey0:ey0+CS_CELL,x0:x0+CS_CELL]=np.clip(ev_strip[k].astype(np.float32)*at,0,255).astype(np.uint8)
        frame[iy0:iy0+CS_CELL,x0:x0+CS_CELL]=np.clip(ivw_strip[k].astype(np.float32)*at,0,255).astype(np.uint8)
        cv2.rectangle(frame,(x0,ey0),(x0+CS_CELL-1,ey0+CS_CELL-1),tuple(int(v*at) for v in col_c),1)
        cv2.rectangle(frame,(x0,iy0),(x0+CS_CELL-1,iy0+CS_CELL-1),tuple(int(v*at*0.5) for v in col_c),1)
        cv2.putText(frame,str(k+1),(x0+4,ey0-5),FONT,0.45,tuple(int(v*at) for v in col_c),1,cv2.LINE_AA)
    all_t=ease((fi-7*16)/20)
    if all_t>0:
        for row_i,(label,) in enumerate([('EX VIVO',),('IN VIVO',),('EX VIVO',),('IN VIVO',)]):
            small_label(frame,label,CS_Y0+row_i*(CS_CELL+CS_GAP)+CS_CELL//2,CS_X0-82,WHITE,all_t,0.37)
    caption(frame,'MULTIMODAL  REGISTRATION',alpha=max(0,1-ease(fi/20)))
    caption(frame,'MATCHED  NEURONS',alpha=ease((fi-130)/30))
    vw.write(frame)
save_qc(frame,'s5_cell_strip')

# ─────────────────────────────────────────────────────────────────────────────
# S6-S8  Per-cell deep-dive  (3 × 216fr, 3 × 9s)
# 3 panels: calcium movie (live) | EV confocal (col 0) | IV warped (col 1)
# ─────────────────────────────────────────────────────────────────────────────
print("S6-S8: per-cell panels…")
PANEL_SZ=320; PANEL_GAP=40
PX1=(W-3*PANEL_SZ-2*PANEL_GAP)//2
PX2=PX1+PANEL_SZ+PANEL_GAP
PX3=PX2+PANEL_SZ+PANEL_GAP
PY=(H-PANEL_SZ)//2-20
CROP_R=45

for ci_idx,ci in enumerate(dive_cells):
    lm_y,lm_x=all_lm_jy[ci]; col=CELL_COLS[ci_idx]
    ev_p=cv2.resize(patch_strip[ci*PATCH_SZ:(ci+1)*PATCH_SZ,0:PATCH_SZ],
                    (PANEL_SZ,PANEL_SZ),interpolation=cv2.INTER_LANCZOS4)
    iv_p=cv2.resize(patch_strip[ci*PATCH_SZ:(ci+1)*PATCH_SZ,PATCH_SZ:PATCH_SZ*2],
                    (PANEL_SZ,PANEL_SZ),interpolation=cv2.INTER_LANCZOS4)

    for fi in range(216):
        frame=np.zeros((H,W,3),np.uint8)
        a_in=ease(fi/25)
        mov_idx=int(fi/215*(n_movie-1))
        mov_f=movie_frames[mov_idx]
        y0=max(0,lm_y-CROP_R); y1=min(mov_f.shape[0],lm_y+CROP_R)
        x0=max(0,lm_x-CROP_R); x1=min(mov_f.shape[1],lm_x+CROP_R)
        crop=mov_f[y0:y1,x0:x1]
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
    save_qc(frame,f's{6+ci_idx}_cell{ci_idx+1}')
    print(f"  Cell {ci_idx+1} done (ci={ci})")

# final hold
for _ in range(24): vw.write(np.zeros((H,W,3),np.uint8))
vw.release()
print(f"Raw: {TMP}")

print("Re-encoding H.264…")
subprocess.run(['ffmpeg','-y','-i',TMP,'-vcodec','libx264','-pix_fmt','yuv420p',
                '-crf','18','-preset','fast',OUT],check=True)
total=192+96+120+336+192+3*216+3*12+24
print(f"\n✓  {OUT}  ({total/FPS:.0f}s @ {FPS}fps)")
print(f"   QC → {QC}/")
