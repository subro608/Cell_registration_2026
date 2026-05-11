"""
1. Resample JY306 in-vivo stack to 1µm isotropic (scipy.ndimage.zoom)
2. Save as invivo_jy306_1um_isotropic.tif
3. Make a quick 3D visualization: rotating max-intensity projection (60 frames, 2.5s)
"""
import numpy as np, cv2, tifffile, math, subprocess, os
from scipy.ndimage import zoom

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_TIF = f'{BASE}/registration_video/stitched/invivo_jy306_1um_isotropic.tif'
OUT_VID = f'{BASE}/animation/invivo_3d_test.mp4'
TMP_VID = f'{BASE}/animation/invivo_3d_test_raw.mp4'
W, H = 1280, 720
FPS = 24

# ── physical pixel sizes (JY306) ──────────────────────────────────────────────
# XY: 1.87 µm/px  Z: 12.6 µm/slice  (16 slices → 189 µm range)
Z_UM  = 12.6   # µm per z-slice
XY_UM = 1.87   # µm per xy-pixel

# ── load ──────────────────────────────────────────────────────────────────────
print("Loading JY306…")
vol = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
print(f"  native: {vol.shape}  range=[{vol.min():.0f},{vol.max():.0f}]")

# ── resample to 1µm isotropic ─────────────────────────────────────────────────
print("Resampling to 1µm isotropic…")
zoom_z  = Z_UM  / 1.0   # 12.6
zoom_xy = XY_UM / 1.0   # 1.87
print(f"  zoom: z×{zoom_z:.2f}  xy×{zoom_xy:.2f}")
vol_iso = zoom(vol, (zoom_z, zoom_xy, zoom_xy), order=1, prefilter=False)
print(f"  isotropic: {vol_iso.shape}")

# normalise for display + save
p1, p99 = np.percentile(vol_iso[vol_iso > 0], [1, 99.5])
vol_u16 = np.clip((vol_iso - p1) / (p99 - p1) * 65535, 0, 65535).astype(np.uint16)
print(f"  saving → {OUT_TIF}")
tifffile.imwrite(OUT_TIF, vol_u16)
print(f"  saved  {vol_u16.nbytes/1e6:.0f} MB")

# ── 3-view orthographic MIP for single QC frame ───────────────────────────────
vol_n = vol_u16.astype(np.float32) / 65535.0
mip_xy = np.max(vol_n, axis=0)   # Z projection → top view
mip_xz = np.max(vol_n, axis=1)   # Y projection → front view
mip_yz = np.max(vol_n, axis=2)   # X projection → side view

def to_u8(m):
    v = m[m>0]
    if len(v)==0: return np.zeros_like(m,np.uint8)
    lo,hi = np.percentile(v,[1,99])
    return np.clip((m-lo)/(hi-lo+1e-8)*255,0,255).astype(np.uint8)

mip_xy_u8 = to_u8(mip_xy)   # (nz_iso, ny_iso)→ no wait: axis0=z so shape (ny,nx)
mip_xz_u8 = to_u8(mip_xz)   # shape (nz_iso, nx_iso) — depth vs X
mip_yz_u8 = to_u8(mip_yz)   # shape (nz_iso, ny_iso) — depth vs Y

nz,ny,nx = vol_n.shape
print(f"  iso dims: z={nz} y={ny} x={nx}")

# ── rotating 3D MIP video ─────────────────────────────────────────────────────
# Orthographic MIP along a rotating horizontal axis in a centered 3D view.
# We render max-intensity projections from 36 angles around the Y axis.
N_ANGLES = 72   # 72 frames = 3s @ 24fps

print(f"Rendering {N_ANGLES} angle MIPs…")
# Downsample for speed
DS = 2
vol_ds = vol_n[::DS, ::DS, ::DS]
nzd,nyd,nxd = vol_ds.shape
print(f"  downsampled: {vol_ds.shape}")

# Precompute flattened coords centred
zc = (np.arange(nzd) - nzd/2).astype(np.float32)
yc = (np.arange(nyd) - nyd/2).astype(np.float32)
xc = (np.arange(nxd) - nxd/2).astype(np.float32)
ZZ,YY,XX = np.meshgrid(zc,yc,xc,indexing='ij')   # each (nzd,nyd,nxd)
ZZ=ZZ.ravel(); YY=YY.ravel(); XX=XX.ravel()
VV = vol_ds.ravel()

THUMB = 600
frames = []
for ai in range(N_ANGLES):
    angle = 2*math.pi * ai / N_ANGLES
    ca,sa = math.cos(angle), math.sin(angle)
    # Rotate around Y axis
    rx =  ca*XX + sa*ZZ
    rz = -sa*XX + ca*ZZ
    ry =  YY
    # Project onto view plane (rx → screen-x, ry → screen-y), depth = rz
    # Scale to thumbnail
    sc = THUMB / max(nxd, nyd, nzd) * 0.85
    px = (rx*sc + THUMB//2).astype(np.int32)
    py = (ry*sc + THUMB//2).astype(np.int32)
    mask = (px>=0)&(px<THUMB)&(py>=0)&(py<THUMB)
    px,py,vv,dd = px[mask],py[mask],VV[mask],rz[mask]
    order = np.argsort(-dd)
    px,py,vv = px[order],py[order],vv[order]
    canvas = np.zeros((THUMB,THUMB),np.float32)
    np.maximum(canvas[py,px], vv, out=canvas[py,px])
    canvas = np.clip(canvas*3, 0, 1)   # boost brightness
    img_u8 = (canvas*255).astype(np.uint8)
    img_u8 = cv2.GaussianBlur(img_u8,(3,3),0.8)
    frames.append(img_u8)
    if ai % 12 == 0: print(f"  angle {ai}/{N_ANGLES}")

print("Writing video…")
vw = cv2.VideoWriter(TMP_VID, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W,H))
FONT = cv2.FONT_HERSHEY_SIMPLEX

for fi,img_u8 in enumerate(frames):
    frame = np.zeros((H,W,3),np.uint8)
    # Place MIP rotation in left panel
    cy,cx = H//2, W//4
    y0,x0 = cy-THUMB//2, cx-THUMB//2
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    # Tint cyan
    bgr_tinted = bgr.copy()
    bgr_tinted[:,:,2] = (bgr[:,:,2].astype(np.float32)*0.6).astype(np.uint8)
    frame[y0:y0+THUMB,x0:x0+THUMB] = bgr_tinted

    # Right panel: 3 static MIP views
    PW,PH = 280,200
    def place_mip(m,cy,cx,label):
        h0,w0=m.shape
        sc=min(PW/w0,PH/h0)
        rs=cv2.resize(m,(int(w0*sc),int(h0*sc)),interpolation=cv2.INTER_LANCZOS4)
        rh,rw=rs.shape
        y1,x1=cy-rh//2,cx-rw//2
        frame[y1:y1+rh,x1:x1+rw]=cv2.cvtColor(rs,cv2.COLOR_GRAY2BGR)
        cv2.putText(frame,label,(x1,y1-6),FONT,0.38,(140,140,140),1,cv2.LINE_AA)

    RX = W//4*3
    place_mip(mip_xy_u8, H//5,   RX, 'XY  (top view)')
    place_mip(mip_xz_u8, H//2,   RX, 'XZ  (front view)')
    place_mip(mip_yz_u8, H//5*4, RX, 'YZ  (side view)')

    cv2.putText(frame,'JY306 IN VIVO  1um ISOTROPIC',(20,36),FONT,0.60,(200,200,200),1,cv2.LINE_AA)
    cv2.putText(frame,f'{nz}z x {ny}y x {nx}x  um',(20,60),FONT,0.38,(120,120,120),1,cv2.LINE_AA)
    vw.write(frame)

# hold last frame 1s
for _ in range(24): vw.write(frame)
vw.release()

subprocess.run(['ffmpeg','-y','-i',TMP_VID,
                '-vcodec','libx264','-pix_fmt','yuv420p','-crf','18','-preset','fast',
                OUT_VID],check=True,capture_output=True)
os.remove(TMP_VID)
print(f"\nDone!  {OUT_VID}")
