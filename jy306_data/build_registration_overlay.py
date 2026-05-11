#!/usr/bin/env python3
"""
Slice-by-slice registration overlay: 16 in-vivo slices, each 2D-warped
into ex-vivo stitched space. No z-interpolation.

For each of the 16 in-vivo z-slices:
  1. Compute which stitched z it maps to (center of slice)
  2. 2D affine warp the in-vivo slice into stitched XY coords at that z
  3. Extract the corresponding ex-vivo stitched slice
  4. Overlay: green=ex-vivo, magenta=in-vivo

HTML with z-slider (16 slices), landmarks, view toggles.
"""
import numpy as np
import tifffile
import cv2
from PIL import Image
from scipy.ndimage import median_filter
import io, base64

BASE = '/Users/neurolab/neuroinformatics/margaret'

IV_XY_UM = 0.6835
IV_Z_UM = 3.0

# ============================================================
# Load data
# ============================================================
print("Loading IOU-only stitched 1µm isotropic volume...")
vol_path = f'{BASE}/registration_video/stitched/stitched_gfp_iou_only_1um_isotropic.tif'
with tifffile.TiffFile(vol_path) as tif:
    n_pages = len(tif.pages)
    h, w = tif.pages[0].shape
    ev_vol = np.zeros((n_pages, h, w), dtype=np.uint16)
    for i, page in enumerate(tif.pages):
        ev_vol[i] = page.asarray()
print(f"  Ex-vivo: {ev_vol.shape}")  # (516, 3554, 3545)

print("Loading JY306 in-vivo volume...")
iv_vol_raw = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv, ny_iv, nx_iv = iv_vol_raw.shape
print(f"  In-vivo: {iv_vol_raw.shape}")  # (16, 658, 629)

print("Loading landmarks and affine...")
lm = np.load(f'{BASE}/registration_video/affine_3d_iou_results.npz', allow_pickle=True)
ev_um = lm['ev_stitched_um']   # (N, 3) x,y,z in µm
iv_um = lm['iv_um']            # (N, 3) x,y,z in µm
A = lm['affine_3x4']           # (3, 4) maps iv µm → stitched µm
N_LM = ev_um.shape[0]

# ============================================================
# Median filter BG subtraction
# ============================================================
print("Median filter BG subtraction on in-vivo...")
iv_vol = np.zeros_like(iv_vol_raw)
for z in range(nz_iv):
    bg = median_filter(iv_vol_raw[z], size=15)
    iv_vol[z] = np.clip(iv_vol_raw[z] - bg, 0, None)
del iv_vol_raw

# ============================================================
# Build 3D affine in (z,y,x) pixel convention
# A maps: [x_um, y_um, z_um, 1] → [x_st, y_st, z_st]
# where x_um = col * IV_XY_UM, etc. and stitched is 1µm/px
# ============================================================
sx, sy, sz = IV_XY_UM, IV_XY_UM, IV_Z_UM

# Forward in (z,y,x) pixel: p_out = M_fwd @ p_in + t_fwd
M_fwd = np.array([
    [A[2,2]*sz, A[2,1]*sy, A[2,0]*sx],
    [A[1,2]*sz, A[1,1]*sy, A[1,0]*sx],
    [A[0,2]*sz, A[0,1]*sy, A[0,0]*sx],
])
t_fwd = np.array([A[2,3], A[1,3], A[0,3]])

M_inv = np.linalg.inv(M_fwd)
offset_inv = -M_inv @ t_fwd

# ============================================================
# XY bounding box in stitched space
# ============================================================
corners_iv = np.array([
    [0,0,0],[0,0,nx_iv-1],[0,ny_iv-1,0],[0,ny_iv-1,nx_iv-1],
    [nz_iv-1,0,0],[nz_iv-1,0,nx_iv-1],[nz_iv-1,ny_iv-1,0],[nz_iv-1,ny_iv-1,nx_iv-1]
], dtype=np.float64)
corners_out = (M_fwd @ corners_iv.T).T + t_fwd
y_lo = max(0, int(np.floor(corners_out[:,1].min())))
y_hi = min(ev_vol.shape[1], int(np.ceil(corners_out[:,1].max())) + 1)
x_lo = max(0, int(np.floor(corners_out[:,2].min())))
x_hi = min(ev_vol.shape[2], int(np.ceil(corners_out[:,2].max())) + 1)
crop_h = y_hi - y_lo
crop_w = x_hi - x_lo
print(f"  XY crop: y=[{y_lo},{y_hi}], x=[{x_lo},{x_hi}] → {crop_h}x{crop_w}")

# ============================================================
# For each of the 16 in-vivo z-slices, compute target stitched z
# and build 2D warp
# ============================================================
DS = 2
out_h = crop_h // DS
out_w = crop_w // DS

print(f"\nProcessing {nz_iv} in-vivo slices, DS{DS} → ({out_h}x{out_w}) per slice...")

overlay_b64 = []
ev_only_b64 = []
iv_only_b64 = []
slice_z_stitched = []

# Normalize in-vivo globally
iv_vals = iv_vol[iv_vol > 0]
iv_p995 = np.percentile(iv_vals, 99.5) if len(iv_vals) > 0 else 1

for z_iv in range(nz_iv):
    # Center of this in-vivo slice in (z,y,x) pixel coords
    center_iv = np.array([z_iv, ny_iv/2, nx_iv/2])
    center_out = M_fwd @ center_iv + t_fwd
    z_st = int(round(center_out[0]))
    z_st = np.clip(z_st, 0, ev_vol.shape[0] - 1)
    slice_z_stitched.append(z_st)

    print(f"  iv z={z_iv} → stitched z={z_st}", end="")

    # 2D backward warp matrix at this z_out
    # For cv2.warpAffine with WARP_INVERSE_MAP:
    #   x_in = M2d[0,0]*x_out + M2d[0,1]*y_out + M2d[0,2]
    #   y_in = M2d[1,0]*x_out + M2d[1,1]*y_out + M2d[1,2]
    # From 3D inverse (z,y,x):
    #   x_in = M_inv[2,2]*x_out + M_inv[2,1]*y_out + (M_inv[2,0]*z_out + offset_inv[2])
    #   y_in = M_inv[1,2]*x_out + M_inv[1,1]*y_out + (M_inv[1,0]*z_out + offset_inv[1])
    # x_out/y_out are in the CROPPED stitched space, so add x_lo/y_lo offset
    M2d = np.array([
        [M_inv[2,2], M_inv[2,1], M_inv[2,0]*z_st + M_inv[2,1]*y_lo + M_inv[2,2]*x_lo + offset_inv[2]],
        [M_inv[1,2], M_inv[1,1], M_inv[1,0]*z_st + M_inv[1,1]*y_lo + M_inv[1,2]*x_lo + offset_inv[1]],
    ], dtype=np.float64)

    # Warp in-vivo slice into cropped stitched XY
    iv_slice = iv_vol[z_iv].astype(np.float32)
    iv_warped_2d = cv2.warpAffine(iv_slice, M2d, (crop_w, crop_h),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Extract corresponding ex-vivo slice
    ev_slice = ev_vol[z_st, y_lo:y_hi, x_lo:x_hi].astype(np.float32)

    # DS2
    ev_ds = cv2.resize(ev_slice, (out_w, out_h), interpolation=cv2.INTER_AREA)
    iv_ds = cv2.resize(iv_warped_2d, (out_w, out_h), interpolation=cv2.INTER_AREA)

    # Normalize ex-vivo: per-slice percentile
    ev_vals = ev_ds[ev_ds > 0]
    if len(ev_vals) > 100:
        ev_hi = np.percentile(ev_vals, 99.5)
        ev_n = np.clip(ev_ds / max(ev_hi, 1), 0, 1)
    else:
        ev_n = np.zeros_like(ev_ds)
    # Normalize in-vivo: per-slice percentile
    iv_vals = iv_ds[iv_ds > 0]
    if len(iv_vals) > 100:
        iv_hi = np.percentile(iv_vals, 99.5)
        iv_n = np.clip(iv_ds / max(iv_hi, 1), 0, 1)
    else:
        iv_n = np.zeros_like(iv_ds)

    ev_u8 = (ev_n * 255).astype(np.uint8)
    iv_u8 = (iv_n * 255).astype(np.uint8)

    # Overlay: green=ex-vivo, magenta=in-vivo
    ov = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    ov[:,:,1] = ev_u8
    ov[:,:,0] = iv_u8
    ov[:,:,2] = iv_u8

    buf = io.BytesIO(); Image.fromarray(ov).save(buf, format='PNG', optimize=True)
    overlay_b64.append(base64.b64encode(buf.getvalue()).decode('ascii'))

    buf2 = io.BytesIO(); Image.fromarray(ev_u8).save(buf2, format='PNG', optimize=True)
    ev_only_b64.append(base64.b64encode(buf2.getvalue()).decode('ascii'))

    buf3 = io.BytesIO(); Image.fromarray(iv_u8).save(buf3, format='PNG', optimize=True)
    iv_only_b64.append(base64.b64encode(buf3.getvalue()).decode('ascii'))

    nz_px = (iv_u8 > 10).sum()
    print(f"  iv_nonzero={nz_px}")

del iv_vol, ev_vol

# ============================================================
# Landmark positions in DS2 cropped px, per-slice
# ============================================================
iv_h = np.hstack([iv_um, np.ones((N_LM, 1))])
iv_pred = iv_h @ A.T  # predicted stitched µm
errors = np.sqrt(np.sum((iv_pred - ev_um)**2, axis=1))

lm_json = []
for i in range(N_LM):
    # Ex-vivo landmark in cropped DS2 coords
    ex_x = (ev_um[i, 0] - x_lo) / DS
    ex_y = (ev_um[i, 1] - y_lo) / DS
    ex_z_st = ev_um[i, 2]  # stitched z in µm (= pixel since 1µm)

    # In-vivo predicted in cropped DS2 coords
    iv_x = (iv_pred[i, 0] - x_lo) / DS
    iv_y = (iv_pred[i, 1] - y_lo) / DS
    iv_z_st = iv_pred[i, 2]

    # Find nearest in-vivo slice index for this landmark
    best_iv_slice = -1
    best_dist = 1e9
    for si, z_st in enumerate(slice_z_stitched):
        d = abs(z_st - ex_z_st)
        if d < best_dist:
            best_dist = d
            best_iv_slice = si

    lm_json.append({
        'ex': [round(float(ex_x),1), round(float(ex_y),1), round(float(ex_z_st),1)],
        'iv': [round(float(iv_x),1), round(float(iv_y),1), round(float(iv_z_st),1)],
        'err': round(float(errors[i]),1),
        'slice': best_iv_slice  # which of the 16 slices this landmark is nearest to
    })

ov_js = ",".join(f'"{s}"' for s in overlay_b64)
ev_js = ",".join(f'"{s}"' for s in ev_only_b64)
iv_js = ",".join(f'"{s}"' for s in iv_only_b64)
lm_js = str(lm_json).replace("'", '"')
zmap_js = str(slice_z_stitched)

# ============================================================
# Build HTML
# ============================================================
print("\nBuilding HTML...")

nz = nz_iv
html = '''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Registration Overlay: 16 In-vivo Slices in Ex-vivo Space</title>
<style>
  body { margin:0; background:#111; color:#ddd; font-family:sans-serif; font-size:13px;
         display:flex; flex-direction:column; height:100vh; overflow:hidden; }
  #top { padding:8px 12px; background:#1a1a1a; display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
  #top b { font-size:14px; }
  #canvas-wrap { flex:1; display:flex; justify-content:center; align-items:center; overflow:hidden; position:relative; }
  canvas { image-rendering:pixelated; cursor:crosshair; }
  #info { position:absolute; bottom:8px; left:12px; background:rgba(0,0,0,0.7); padding:6px 10px; border-radius:4px; font-size:11px; }
  #legend { position:absolute; top:8px; right:12px; background:rgba(0,0,0,0.7); padding:6px 10px; border-radius:4px; font-size:11px; }
  .btn { background:#333; border:1px solid #555; color:#ddd; padding:3px 10px; border-radius:3px; cursor:pointer; font-size:12px; }
  .btn.active { background:#0a5; border-color:#0c7; }
</style>
</head><body>

<div id="top">
  <b>Registration: 16 In-vivo Slices</b>
  <label>In-vivo Z: <input type="range" id="zslider" min="0" max="''' + str(nz-1) + '''" value="0" style="width:250px">
  <span id="zval">0</span>/''' + str(nz-1) + '''</label>
  <span id="zinfo" style="color:#0ff;font-size:11px"></span>
  <button class="btn active" id="btn_overlay" onclick="setMode('overlay')">Overlay (1)</button>
  <button class="btn" id="btn_ev" onclick="setMode('ev')">Ex-vivo (2)</button>
  <button class="btn" id="btn_iv" onclick="setMode('iv')">In-vivo (3)</button>
  <button class="btn active" id="btn_marks" onclick="toggleMarkers()">Landmarks (M)</button>
  <label>Zoom: <input type="range" id="zoomslider" min="50" max="400" value="100" style="width:80px"><span id="zoomval">100</span>%</label>
</div>

<div id="canvas-wrap">
  <canvas id="cv"></canvas>
  <div id="legend">
    <span style="color:#0f0">&#9632;</span> Ex-vivo stitched (green)<br>
    <span style="color:#f0f">&#9632;</span> In-vivo 2D-warped (magenta)<br>
    <span style="color:#0ff">+</span> Ex-vivo landmark<br>
    <span style="color:#ff0">+</span> In-vivo predicted<br>
    ''' + str(N_LM) + ''' landmarks | Mean err: ''' + f'{errors.mean():.1f}' + ''' um<br>
    No z-interpolation: each slice is a native in-vivo z
  </div>
  <div id="info">Scroll/arrows: z | 1/2/3: view | M: landmarks | Ctrl+scroll: zoom | Shift+drag: pan</div>
</div>

<script>
const NZ=''' + str(nz) + ''', NY=''' + str(out_h) + ''', NX=''' + str(out_w) + ''';
const zMap=''' + zmap_js + ''';
const ovSlices=[''' + ov_js + '''];
const evSlices=[''' + ev_js + '''];
const ivSlices=[''' + iv_js + '''];
const landmarks=''' + lm_js + ''';

const cv=document.getElementById('cv');
const ctx=cv.getContext('2d');
cv.width=NX; cv.height=NY;

let curZ=0, mode='overlay', showMarkers=true, zoomPct=100;
let panX=0, panY=0, dragging=false, dragSX=0, dragSY=0, panSX=0, panSY=0;

const imgCache={};
function getImg(z,m) {
  const k=m+'_'+z;
  if(imgCache[k]) return imgCache[k];
  const arr=m==='overlay'?ovSlices:m==='ev'?evSlices:ivSlices;
  const img=new window.Image();
  img.src='data:image/png;base64,'+arr[z];
  imgCache[k]=img;
  return img;
}

function draw() {
  const s=zoomPct/100;
  cv.style.width=Math.round(NX*s)+'px';
  cv.style.height=Math.round(NY*s)+'px';
  cv.style.transform='translate('+panX+'px,'+panY+'px)';
  ctx.clearRect(0,0,NX,NY);
  const img=getImg(curZ,mode);
  const doDraw=()=>{
    ctx.drawImage(img,0,0,NX,NY);
    if(showMarkers) drawMarkers();
  };
  if(img.complete) doDraw(); else img.onload=doDraw;
  document.getElementById('zinfo').textContent=
    'iv_z='+curZ+' → stitched_z='+zMap[curZ];
}

function drawMarkers() {
  const R=10;
  let idx=0;
  const nearby=[];
  for(const lm of landmarks) {
    idx++;
    if(lm.slice!==curZ) continue;
    nearby.push(idx);

    // White connecting line
    ctx.beginPath(); ctx.moveTo(lm.ex[0],lm.ex[1]); ctx.lineTo(lm.iv[0],lm.iv[1]);
    ctx.strokeStyle='rgba(255,255,255,0.7)'; ctx.lineWidth=2; ctx.stroke();

    // Cyan crosshair = ex-vivo
    ctx.strokeStyle='rgba(0,255,255,1)'; ctx.lineWidth=2;
    ctx.beginPath(); ctx.arc(lm.ex[0],lm.ex[1],R,0,Math.PI*2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(lm.ex[0]-R-4,lm.ex[1]); ctx.lineTo(lm.ex[0]+R+4,lm.ex[1]); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(lm.ex[0],lm.ex[1]-R-4); ctx.lineTo(lm.ex[0],lm.ex[1]+R+4); ctx.stroke();

    // Yellow crosshair = in-vivo predicted
    ctx.strokeStyle='rgba(255,255,0,1)'; ctx.lineWidth=2;
    ctx.beginPath(); ctx.arc(lm.iv[0],lm.iv[1],R,0,Math.PI*2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(lm.iv[0]-R-4,lm.iv[1]); ctx.lineTo(lm.iv[0]+R+4,lm.iv[1]); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(lm.iv[0],lm.iv[1]-R-4); ctx.lineTo(lm.iv[0],lm.iv[1]+R+4); ctx.stroke();

    // Label
    const lx=Math.max(lm.ex[0],lm.iv[0])+R+6;
    const ly=Math.min(lm.ex[1],lm.iv[1])-4;
    ctx.font='bold 11px sans-serif';
    ctx.fillStyle='rgba(255,255,255,0.9)';
    ctx.fillText('#'+idx+' '+lm.err+'um',lx,ly);
  }
  document.getElementById('info').textContent=nearby.length+' landmarks on this slice';
}

function setZ(z) {
  curZ=Math.max(0,Math.min(NZ-1,z));
  document.getElementById('zslider').value=curZ;
  document.getElementById('zval').textContent=curZ;
  draw();
}

function setMode(m) {
  mode=m;
  ['overlay','ev','iv'].forEach(k=>document.getElementById('btn_'+k).classList.toggle('active',k===m));
  draw();
}

function toggleMarkers() {
  showMarkers=!showMarkers;
  document.getElementById('btn_marks').textContent='Landmarks '+(showMarkers?'ON':'OFF')+' (M)';
  document.getElementById('btn_marks').classList.toggle('active',showMarkers);
  draw();
}

document.getElementById('zslider').addEventListener('input',e=>setZ(+e.target.value));
document.getElementById('zoomslider').addEventListener('input',e=>{
  zoomPct=+e.target.value;
  document.getElementById('zoomval').textContent=zoomPct;
  draw();
});

document.addEventListener('keydown',e=>{
  if(e.key==='ArrowRight'||e.key==='ArrowDown'||e.key==='d') setZ(curZ+1);
  else if(e.key==='ArrowLeft'||e.key==='ArrowUp'||e.key==='a') setZ(curZ-1);
  else if(e.key==='1') setMode('overlay');
  else if(e.key==='2') setMode('ev');
  else if(e.key==='3') setMode('iv');
  else if(e.key==='m'||e.key==='M') toggleMarkers();
});

cv.addEventListener('wheel',e=>{
  e.preventDefault();
  if(e.ctrlKey||e.metaKey) {
    zoomPct=Math.max(50,Math.min(400,zoomPct-(e.deltaY>0?10:-10)));
    document.getElementById('zoomslider').value=zoomPct;
    document.getElementById('zoomval').textContent=zoomPct;
    draw();
  } else { setZ(curZ+(e.deltaY>0?1:-1)); }
},{passive:false});

cv.addEventListener('mousedown',e=>{
  if(e.shiftKey||e.button===1) {
    dragging=true;dragSX=e.clientX;dragSY=e.clientY;panSX=panX;panSY=panY;e.preventDefault();
  }
});
document.addEventListener('mousemove',e=>{
  if(dragging) { panX=panSX+(e.clientX-dragSX); panY=panSY+(e.clientY-dragSY); draw(); }
});
document.addEventListener('mouseup',()=>dragging=false);

cv.addEventListener('click',e=>{
  if(dragging) return;
  const rect=cv.getBoundingClientRect(), s=zoomPct/100;
  const px=Math.round((e.clientX-rect.left)/s), py=Math.round((e.clientY-rect.top)/s);
  let best=-1, bestD=1e9;
  for(let i=0;i<landmarks.length;i++) {
    if(landmarks[i].slice!==curZ) continue;
    const dx=landmarks[i].ex[0]-px, dy=landmarks[i].ex[1]-py;
    const d=Math.sqrt(dx*dx+dy*dy);
    if(d<bestD) { bestD=d; best=i; }
  }
  let msg='('+px+','+py+') iv_z='+curZ+' stitch_z='+zMap[curZ];
  if(best>=0&&bestD<40) msg+=' | #'+(best+1)+': err='+landmarks[best].err+'um';
  document.getElementById('info').textContent=msg;
});

draw();
</script>
</body></html>
'''

out = f'{BASE}/registration_overlay.html'
with open(out, 'w') as f:
    f.write(html)
print(f"Done! {out} ({len(html)/1e6:.1f} MB)")
