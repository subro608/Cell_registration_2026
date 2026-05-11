#!/usr/bin/env python3
"""
3D HTML viewer for stitched ex-vivo v5 volume (1µm isotropic).
Per-slice equalization + GP RBF interpolation (same as viewer_equalized.html).
8x downsample → ~65×444×443.
"""
import numpy as np
import tifffile
from PIL import Image
import io, base64

TIFF = "/Users/neurolab/neuroinformatics/margaret/registration_video/stitched/stitched_gfp_fullres_v5_1um_isotropic.tif"
OUT = "/Users/neurolab/neuroinformatics/margaret/3d_viewer/viewer_stitched_v5.html"
DS = 4
NORM = 4000

print("Loading stitched v5 volume (page-by-page)...")
with tifffile.TiffFile(TIFF) as tif:
    n_pages = len(tif.pages)
    h, w = tif.pages[0].shape
    print(f"  {n_pages} pages, {h}x{w}")
    nz = n_pages // DS
    ny = h // DS
    nx = w // DS
    print(f"  {DS}x DS: ({nz}, {ny}, {nx})")

    vol_ds = np.zeros((nz, ny, nx), dtype=np.float32)
    for zi in range(nz):
        page_idx = zi * DS
        sl = tif.pages[page_idx].asarray().astype(np.float32)
        vol_ds[zi] = sl[::DS, ::DS][:ny, :nx]
        if zi % 10 == 0:
            print(f"  z={zi}/{nz}")

# Per-slice intensity equalization
print("Per-slice equalization...")
all_nz = vol_ds[vol_ds > 0]
global_mean = all_nz.mean()
print(f"  Global nonzero mean: {global_mean:.1f}")
for z in range(nz):
    sl = vol_ds[z]
    mask = sl > 0
    if mask.sum() > 100:
        sl_mean = sl[mask].mean()
        if sl_mean > 1:
            vol_ds[z][mask] *= (global_mean / sl_mean)

# Normalize by /4000
vol_u8 = np.clip(vol_ds / NORM * 255, 0, 255).astype(np.uint8)

print("Encoding PNGs...")
b64 = []
for z in range(nz):
    img = Image.fromarray(vol_u8[z])
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    b64.append(base64.b64encode(buf.getvalue()).decode('ascii'))

aspect_z = nz / max(ny, nx)
slice_js = ",".join(f'"{s}"' for s in b64)

# JS code with GP interpolation (same as viewer_equalized.html)
js_code = """
const NZ=__NZ__, NY=__NY__, NX=__NX__;
const ASPECT_Z = __AZ__;
const sliceData = [__SLICES__];
let scene, camera, renderer, points, rotY=0, rotX=-0.3;
let dragging=false, lastX=0, lastY=0, zoom=2.5, panX=0, panY=0;
let autoRotate=true;
let rawData = null;

function colormap(v, name) {
  if(name==='green') return [0, v, 0];
  if(name==='hot') return [Math.min(v*2,1), Math.max(v*2-1,0)*0.8, Math.max(v*3-2,0)];
  if(name==='cyan') return [0, v*0.8, v];
  return [v, v, v];
}

async function loadSlices() {
  rawData = new Uint8Array(NZ * NY * NX);
  for(let z=0; z<NZ; z++) {
    const img = new Image();
    await new Promise(r => { img.onload=r; img.src='data:image/png;base64,'+sliceData[z]; });
    const c = document.createElement('canvas');
    c.width=img.width; c.height=img.height;
    const ctx = c.getContext('2d');
    ctx.drawImage(img,0,0);
    const d = ctx.getImageData(0,0,img.width,img.height).data;
    const off = z * NY * NX;
    for(let y=0; y<NY; y++)
      for(let x=0; x<NX; x++)
        rawData[off + y * NX + x] = d[(y*NX+x)*4];
    if(z % 20 === 0) document.getElementById('ptCount').textContent = 'Loading slice ' + z + '/' + NZ;
  }
}

// ====== GP MATH ======

function rbfKernel(a, b, l) {
  const d = a - b;
  return Math.exp(-(d*d) / (2*l*l));
}

function buildK(N, l, noise) {
  const K = new Float64Array(N * N);
  for(let i=0; i<N; i++) {
    for(let j=0; j<N; j++) {
      K[i*N+j] = rbfKernel(i, j, l);
      if(i === j) K[i*N+j] += noise * noise;
    }
  }
  return K;
}

function cholesky(K, N) {
  const L = new Float64Array(N * N);
  for(let i=0; i<N; i++) {
    for(let j=0; j<=i; j++) {
      let s = 0;
      for(let k=0; k<j; k++) s += L[i*N+k] * L[j*N+k];
      if(i === j) {
        const diag = K[i*N+i] - s;
        L[i*N+j] = diag > 0 ? Math.sqrt(diag) : 1e-10;
      } else {
        L[i*N+j] = (K[i*N+j] - s) / L[j*N+j];
      }
    }
  }
  return L;
}

function solveL(L, b, N) {
  const x = new Float64Array(N);
  for(let i=0; i<N; i++) {
    let s = 0;
    for(let k=0; k<i; k++) s += L[i*N+k] * x[k];
    x[i] = (b[i] - s) / L[i*N+i];
  }
  return x;
}

function solveLT(L, b, N) {
  const x = new Float64Array(N);
  for(let i=N-1; i>=0; i--) {
    let s = 0;
    for(let k=i+1; k<N; k++) s += L[k*N+i] * x[k];
    x[i] = (b[i] - s) / L[i*N+i];
  }
  return x;
}

function solveChol(L, b, N) {
  return solveLT(L, solveL(L, b, N), N);
}

function gpWeights(NZ, targetZs, l, noise) {
  const K = buildK(NZ, l, noise);
  const L = cholesky(K, NZ);
  const W = [];
  for(let t=0; t<targetZs.length; t++) {
    const zstar = targetZs[t];
    const kstar = new Float64Array(NZ);
    for(let i=0; i<NZ; i++) kstar[i] = rbfKernel(zstar, i, l);
    W.push(solveChol(L, kstar, NZ));
  }
  return W;
}

// ====== BUILD POINTS ======

function buildPoints() {
  const thresh = +document.getElementById('thresh').value;
  const opac = +document.getElementById('opacity').value / 100;
  const ps = +document.getElementById('psize').value;
  const cmapName = document.getElementById('cmap').value;
  const l = +document.getElementById('gp_l').value / 10;
  const interp = +document.getElementById('gp_interp').value;
  const noise = +document.getElementById('gp_noise').value / 1000;

  if(points) scene.remove(points);

  const targetZs = [];
  for(let z=0; z<NZ; z++) {
    targetZs.push(z);
    if(interp > 1 && z < NZ-1) {
      for(let k=1; k<interp; k++) targetZs.push(z + k/interp);
    }
  }
  const nZout = targetZs.length;

  console.time('GP weights');
  const W = gpWeights(NZ, targetZs, l, noise);
  console.timeEnd('GP weights');

  console.time('GP interpolation');
  const ptsX = [], ptsY = [], ptsT = [], ptsV = [];
  const colBuf = new Float64Array(NZ);

  for(let y=0; y<NY; y++) {
    for(let x=0; x<NX; x++) {
      let hasData = false;
      for(let z=0; z<NZ; z++) {
        colBuf[z] = rawData[z * NY * NX + y * NX + x];
        if(colBuf[z] > 0) hasData = true;
      }
      if(!hasData) continue;

      for(let t=0; t<nZout; t++) {
        let val = 0;
        const w = W[t];
        for(let i=0; i<NZ; i++) val += w[i] * colBuf[i];
        val = Math.max(0, Math.min(255, val));
        if(val > thresh) {
          ptsX.push(x); ptsY.push(y); ptsT.push(t); ptsV.push(val);
        }
      }
    }
  }
  console.timeEnd('GP interpolation');

  const n = ptsX.length;
  document.getElementById('ptCount').textContent = n.toLocaleString() + ' pts | ' + nZout + ' z-levels';

  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  for(let i=0; i<n; i++) {
    const zReal = targetZs[ptsT[i]];
    pos[i*3]   = (ptsX[i]/NX - 0.5) * 2;
    pos[i*3+1] = -(ptsY[i]/NY - 0.5) * 2;
    pos[i*3+2] = (zReal/NZ - 0.5) * 2 * ASPECT_Z;
    const nv = ptsV[i] / 255;
    const [r,g,b] = colormap(nv, cmapName);
    col[i*3]=r; col[i*3+1]=g; col[i*3+2]=b;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const mat = new THREE.PointsMaterial({
    size: ps * 0.01, vertexColors: true, transparent: true, opacity: opac,
    blending: THREE.AdditiveBlending, depthWrite: false
  });
  points = new THREE.Points(geo, mat);
  scene.add(points);
}

function animate() {
  requestAnimationFrame(animate);
  if(autoRotate && !dragging) rotY += 0.003;
  if(points) {
    points.rotation.y = rotY;
    points.rotation.x = rotX;
    points.position.x = panX;
    points.position.y = panY;
  }
  camera.position.z = zoom;
  renderer.render(scene, camera);
}

async function init() {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.1, 100);
  camera.position.z = zoom;
  renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x000000);
  document.body.appendChild(renderer.domElement);
  await loadSlices();
  document.getElementById('ptCount').textContent = 'Computing GP...';
  setTimeout(() => { buildPoints(); animate(); }, 50);
}

document.addEventListener('mousedown', e => { dragging=true; lastX=e.clientX; lastY=e.clientY; });
document.addEventListener('mouseup', () => dragging=false);
document.addEventListener('mousemove', e => {
  if(!dragging) return;
  const dx = e.clientX-lastX, dy = e.clientY-lastY;
  if(e.shiftKey) { panX+=dx*0.002; panY-=dy*0.002; }
  else { rotY+=dx*0.005; rotX+=dy*0.005; }
  lastX=e.clientX; lastY=e.clientY;
});
document.addEventListener('wheel', e => { zoom=Math.max(0.5, Math.min(10, zoom+e.deltaY*0.002)); });
window.addEventListener('resize', () => {
  camera.aspect=window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

let rebuildTimer = null;
['thresh','opacity','psize','cmap','gp_l','gp_interp','gp_noise'].forEach(id => {
  document.getElementById(id).addEventListener('input', () => {
    document.getElementById('threshVal').textContent=document.getElementById('thresh').value;
    document.getElementById('opVal').textContent=document.getElementById('opacity').value;
    document.getElementById('psVal').textContent=document.getElementById('psize').value;
    document.getElementById('gpLVal').textContent=(+document.getElementById('gp_l').value/10).toFixed(1);
    document.getElementById('gpIVal').textContent=document.getElementById('gp_interp').value;
    document.getElementById('gpNVal').textContent=(+document.getElementById('gp_noise').value/1000).toFixed(3);
    clearTimeout(rebuildTimer);
    rebuildTimer = setTimeout(() => { if(rawData) { document.getElementById('ptCount').textContent='Recomputing GP...'; setTimeout(buildPoints, 30); } }, 300);
  });
});
document.getElementById('autorot').addEventListener('change', e => autoRotate=e.target.checked);
init();
"""

js_code = js_code.replace('__NZ__', str(nz))
js_code = js_code.replace('__NY__', str(ny))
js_code = js_code.replace('__NX__', str(nx))
js_code = js_code.replace('__AZ__', f'{aspect_z:.6f}')
js_code = js_code.replace('__SLICES__', slice_js)

html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Stitched Ex-vivo v5 3D — GP Equalized (1µm iso)</title>
<style>
  body {{ margin:0; overflow:hidden; background:#000; color:#ddd; font-family:sans-serif; font-size:12px; }}
  #info {{ position:absolute; top:10px; left:10px; z-index:10; background:rgba(0,0,0,0.7); padding:8px; border-radius:4px; }}
  #controls {{ position:absolute; top:10px; right:10px; z-index:10; background:rgba(0,0,0,0.7);
               padding:8px 12px; border-radius:4px; }}
  #controls label {{ display:block; margin:4px 0; }}
</style>
</head><body>
<div id="info">
  <b>Stitched Ex-vivo v5 (1µm iso) — GP Equalized</b><br>
  Full: 516&times;3554&times;3545 &rarr; Display: {nz}&times;{ny}&times;{nx} ({DS}x DS, equalized)<br>
  Drag: rotate | Scroll: zoom | Shift+drag: pan
</div>
<div id="controls">
  <label>Threshold: <input type="range" id="thresh" min="0" max="15" value="5" style="width:120px"><span id="threshVal">15</span></label>
  <label>Opacity: <input type="range" id="opacity" min="1" max="100" value="30" style="width:120px"><span id="opVal">30</span></label>
  <label>Point size: <input type="range" id="psize" min="1" max="10" value="2" style="width:120px"><span id="psVal">2</span></label>
  <label><input type="checkbox" id="autorot" checked> Auto-rotate</label>
  <label>Colormap: <select id="cmap"><option value="green">Green</option><option value="hot">Hot</option><option value="gray">Gray</option><option value="cyan">Cyan</option></select></label>
  <hr style="border-color:#555;margin:6px 0">
  <label style="color:#0f0;font-weight:bold">GP Lengthscale (l): <input type="range" id="gp_l" min="1" max="200" value="100" step="1" style="width:120px"><span id="gpLVal">10.0</span></label>
  <label style="color:#0f0">Interp factor: <input type="range" id="gp_interp" min="1" max="4" value="1" step="1" style="width:120px"><span id="gpIVal">1</span>&times;</label>
  <label style="color:#0f0">Noise &sigma;: <input type="range" id="gp_noise" min="1" max="100" value="10" step="1" style="width:120px"><span id="gpNVal">0.010</span></label>
  <div style="font-size:10px;color:#888;margin-top:4px">
    GP posterior: &mu;(z*)=k<sub>*</sub>K<sup>-1</sup>y<br>
    k(z,z')=exp(-|z-z'|&sup2;/2l&sup2;)<br>
    l=large: smooth | interp: z-slices between originals<br>
    <span id="ptCount"></span>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
{js_code}
</script></body></html>
'''

with open(OUT, 'w') as f:
    f.write(html)
print(f"Done! {OUT} ({len(html)/1e6:.1f} MB)")
