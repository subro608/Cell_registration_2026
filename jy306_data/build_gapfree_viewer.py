#!/usr/bin/env python3
"""
Build a gap-free 3D viewer from the feathered stitched volume.
Strategy: keep ALL z-slices (504), downsample XY by 8x (→343×343).
This gives ~59M voxels — manageable for the browser.
No z-squishing. The dense z sampling fills the visual gaps.
"""
import numpy as np
import tifffile
from PIL import Image
import io, base64, json

# --- Load feathered volume ---
print("Loading feathered volume...")
vol = tifffile.imread("/Users/neurolab/neuroinformatics/margaret/registration_video/stitched/stitched_gfp_elastix_1um_isotropic_feathered.tif")
print(f"  Shape: {vol.shape}, dtype: {vol.dtype}, range: {vol.min()}-{vol.max()}")

NZ_orig, NY_orig, NX_orig = vol.shape

# --- Per-slice intensity equalization ---
print("Equalizing per-slice intensity...")
vol_f = vol.astype(np.float32)
global_mean = vol_f[vol_f > 0].mean()
print(f"  Global mean (nonzero): {global_mean:.1f}")
for z in range(NZ_orig):
    sl = vol_f[z]
    nz_mask = sl > 0
    if nz_mask.sum() > 1000:
        sl_mean = sl[nz_mask].mean()
        if sl_mean > 1:
            sl[nz_mask] *= (global_mean / sl_mean)
    vol_f[z] = sl

# Clip to uint16 range
vol_f = np.clip(vol_f, 0, 65535)

# --- Downsample XY by 8x ---
DS_XY = 8
ny = NY_orig // DS_XY  # 343
nx = NX_orig // DS_XY  # 343
nz = NZ_orig            # 504 (keep all z)

print(f"Downsampling XY by {DS_XY}x: ({nz}, {ny}, {nx})")
print(f"  Total voxels: {nz * ny * nx / 1e6:.1f}M")

# Use block averaging for XY downsample
vol_ds = np.zeros((nz, ny, nx), dtype=np.float32)
for z in range(nz):
    # Crop to exact multiple
    crop_y = ny * DS_XY
    crop_x = nx * DS_XY
    sl = vol_f[z, :crop_y, :crop_x]
    # Block average
    vol_ds[z] = sl.reshape(ny, DS_XY, nx, DS_XY).mean(axis=(1, 3))
    if z % 50 == 0:
        print(f"  z={z}/{nz}")

# --- Normalize to 0-255 for PNG encoding ---
p99 = np.percentile(vol_ds[vol_ds > 0], 99)
print(f"  p99 (nonzero) = {p99:.1f}")
vol_u8 = np.clip(vol_ds / p99 * 255, 0, 255).astype(np.uint8)
print(f"  Final uint8 range: {vol_u8.min()}-{vol_u8.max()}")

# --- Encode slices as base64 PNGs ---
print("Encoding slices as base64 PNG...")
b64_slices = []
for z in range(nz):
    img = Image.fromarray(vol_u8[z], mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    b64_slices.append(base64.b64encode(buf.getvalue()).decode('ascii'))
    if z % 50 == 0:
        print(f"  z={z}/{nz}")

total_b64_bytes = sum(len(s) for s in b64_slices)
print(f"  Total base64 size: {total_b64_bytes / 1e6:.1f} MB")

# --- Aspect ratio ---
# Original volume is 1µm isotropic, so aspect is 1:1:1
# After DS: z stays at 1µm, XY goes to 8µm per pixel
# So aspect_z = NZ_real_um / max(NY_real_um, NX_real_um)
# = 504 / (343*8) = 504/2744 ≈ 0.1837
aspect_z = NZ_orig / max(NY_orig, NX_orig)
print(f"  ASPECT_Z = {aspect_z:.6f}")

# --- Generate HTML ---
print("Generating HTML viewer...")

html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Ex-vivo Hippocampus — Gap-free (Full Z)</title>
<style>body{{margin:0;overflow:hidden;background:#000;font:12px sans-serif;color:#ccc}}
#controls{{position:fixed;top:8px;left:8px;z-index:10;background:rgba(0,0,0,0.7);padding:8px;border-radius:6px}}
label{{display:block;margin:4px 0}}canvas{{display:block}}</style></head>
<body>
<div id="controls">
  <b>Ex-vivo Hippocampus — Gap-free (Full Z)</b><br>
  <small>{nz}&times;{ny}&times;{nx} | XY {DS_XY}x DS | all {NZ_orig} z-slices</small><br>
  <label>Threshold: <input type="range" id="thresh" min="0" max="255" value="15" style="width:120px"><span id="threshVal">15</span></label>
  <label>Opacity: <input type="range" id="opacity" min="1" max="100" value="8" style="width:120px"><span id="opVal">8</span></label>
  <label>Point Size: <input type="range" id="psize" min="1" max="30" value="3" style="width:120px"><span id="psVal">3</span></label>
  <label>Colormap: <select id="cmap"><option value="green">Green</option><option value="gray">Gray</option><option value="hot">Hot</option><option value="cyan">Cyan</option></select></label>
  <label><input type="checkbox" id="autorot" checked> Auto-rotate</label>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const NZ={nz}, NY={ny}, NX={nx};
const ASPECT_Z = {aspect_z:.6f};
const sliceData=[
"""

# Embed slice data
for i, s in enumerate(b64_slices):
    html += f'"{s}"'
    if i < len(b64_slices) - 1:
        html += ","
    if i % 10 == 9:
        html += "\n"

html += f"""];
async function loadSlices(){{const voxels=[];for(let z=0;z<NZ;z++){{const img=new window.Image();await new Promise(r=>{{img.onload=r;img.src='data:image/png;base64,'+sliceData[z];}});const c=document.createElement('canvas');c.width=img.width;c.height=img.height;const ctx=c.getContext('2d');ctx.drawImage(img,0,0);const d=ctx.getImageData(0,0,img.width,img.height).data;for(let y=0;y<NY;y++)for(let x=0;x<NX;x++){{const val=d[(y*NX+x)*4];voxels.push({{x,y,z,val}});}}}}return voxels;}}

function colormap(t,name){{
  if(name==='green')return[0,t,0];
  if(name==='gray')return[t,t,t];
  if(name==='cyan')return[0,t,t];
  if(name==='hot')return[Math.min(t*3,1),Math.max(0,t*3-1),Math.max(0,t*3-2)];
  return[0,t,0];
}}

function buildPoints(voxels){{
  const thresh=+document.getElementById('thresh').value;
  const opa=+document.getElementById('opacity').value/100;
  const ps=+document.getElementById('psize').value;
  const cm=document.getElementById('cmap').value;
  const filt=voxels.filter(v=>v.val>=thresh);
  const n=filt.length;
  document.title='Gap-free ('+n.toLocaleString()+' pts)';
  const pos=new Float32Array(n*3);
  const col=new Float32Array(n*3);
  for(let i=0;i<n;i++){{
    const v=filt[i];
    pos[i*3]=(v.x/NX-0.5)*2;
    pos[i*3+1]=(v.y/NY-0.5)*2;
    pos[i*3+2]=(v.z/NZ-0.5)*2*ASPECT_Z;
    const t=v.val/255;
    const c=colormap(t,cm);
    col[i*3]=c[0];col[i*3+1]=c[1];col[i*3+2]=c[2];
  }}
  const geom=new THREE.BufferGeometry();
  geom.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  geom.setAttribute('color',new THREE.Float32BufferAttribute(col,3));
  const mat=new THREE.PointsMaterial({{size:ps*0.01,vertexColors:true,transparent:true,opacity:opa,
    blending:THREE.AdditiveBlending,depthWrite:false,sizeAttenuation:true}});
  if(window._pts)window._scene.remove(window._pts);
  window._pts=new THREE.Points(geom,mat);
  window._scene.add(window._pts);
}}

async function main(){{
  const scene=new THREE.Scene();window._scene=scene;
  const cam=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.01,100);
  cam.position.set(0,0,2.5);
  const ren=new THREE.WebGLRenderer({{antialias:true}});
  ren.setSize(innerWidth,innerHeight);ren.setPixelRatio(devicePixelRatio);
  document.body.appendChild(ren.domElement);

  // Mouse controls
  let isDrag=false,prevX=0,prevY=0,rotX=-0.3,rotY=0,dist=2.5;
  ren.domElement.addEventListener('mousedown',e=>{{isDrag=true;prevX=e.clientX;prevY=e.clientY;}});
  window.addEventListener('mouseup',()=>isDrag=false);
  window.addEventListener('mousemove',e=>{{
    if(!isDrag)return;
    rotY+=(e.clientX-prevX)*0.005;rotX+=(e.clientY-prevY)*0.005;
    prevX=e.clientX;prevY=e.clientY;
  }});
  ren.domElement.addEventListener('wheel',e=>{{dist=Math.max(0.5,Math.min(10,dist+e.deltaY*0.002));}});
  window.addEventListener('resize',()=>{{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();ren.setSize(innerWidth,innerHeight);}});

  const status=document.createElement('div');
  status.style.cssText='position:fixed;bottom:10px;left:10px;color:#0f0;font:14px monospace;z-index:10';
  document.body.appendChild(status);
  status.textContent='Loading slices...';

  const voxels=await loadSlices();
  window._voxels=voxels;
  status.textContent='Building point cloud ('+voxels.length.toLocaleString()+' voxels)...';
  buildPoints(voxels);
  status.textContent=voxels.filter(v=>v.val>=+document.getElementById('thresh').value).length.toLocaleString()+' points rendered';

  ['thresh','opacity','psize','cmap'].forEach(id=>{{
    document.getElementById(id).addEventListener('input',()=>{{
      document.getElementById('threshVal').textContent=document.getElementById('thresh').value;
      document.getElementById('opVal').textContent=document.getElementById('opacity').value;
      document.getElementById('psVal').textContent=document.getElementById('psize').value;
      if(window._voxels)buildPoints(window._voxels);
    }});
  }});

  function animate(){{
    requestAnimationFrame(animate);
    if(document.getElementById('autorot').checked)rotY+=0.003;
    cam.position.set(dist*Math.sin(rotY)*Math.cos(rotX),dist*Math.sin(rotX),dist*Math.cos(rotY)*Math.cos(rotX));
    cam.lookAt(0,0,0);
    ren.render(scene,cam);
  }}
  animate();
}}
main();
</script></body></html>
"""

out_path = "/Users/neurolab/neuroinformatics/margaret/3d_viewer/viewer_gapfree_fullz.html"
with open(out_path, 'w') as f:
    f.write(html)

fsize = len(html) / 1e6
print(f"\nDone! Wrote {out_path} ({fsize:.1f} MB)")
print(f"Open with: open {out_path}")
