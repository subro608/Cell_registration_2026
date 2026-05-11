#!/usr/bin/python3
"""
Build 3D HTML viewer of IOU-stitched ex-vivo volume with masks applied.
Same rendering pipeline as viewer_dual_3d_v5: DS3, per-slice equalization,
/4000 normalization, GP 2x interpolation in z.
"""
import numpy as np
import cv2
import json
import base64

BASE = "/Users/neurolab/neuroinformatics/margaret"

TILE_ORDER = [
    'row1_1', 'row1_2', 'row1_3',
    'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
    'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
    'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
    'row5_1',
]
ROW1_4_INSERT_AFTER = 'row1_3'
ROW1_4_GAP = 6

DS = 3
NORM = 4000
VOXEL_THRESH = 8

# GP parameters (same as v5 viewer)
GP_LENGTHSCALE = 1.0
GP_INTERP = 2
GP_NOISE = 0.01

def gp_weight_matrix(nz, interp, lengthscale, noise):
    target_zs = []
    for z in range(nz):
        target_zs.append(z)
        if interp > 1 and z < nz - 1:
            for k in range(1, interp):
                target_zs.append(z + k / interp)
    target_zs = np.array(target_zs, dtype=np.float64)
    z_orig = np.arange(nz, dtype=np.float64)
    K = np.exp(-0.5 * (z_orig[:, None] - z_orig[None, :]) ** 2 / lengthscale ** 2)
    K += noise ** 2 * np.eye(nz)
    K_inv = np.linalg.solve(K, np.eye(nz))
    kstar = np.exp(-0.5 * (target_zs[:, None] - z_orig[None, :]) ** 2 / lengthscale ** 2)
    W = kstar @ K_inv
    return W, target_zs

# ============================================================
# 1. Load transforms and masks
# ============================================================
print("Loading transforms and masks...")
with open(f'{BASE}/registration_video/auto_align_transforms_iou_v4.json') as f:
    iou_transforms = json.load(f)

params = json.load(open(f"{BASE}/registration_video/stitch_v5_params.json"))
masks_data = np.load(f'{BASE}/registration_video/via_masks_v4.npz')
masks = {k: masks_data[k] for k in masks_data.files}

def to_3x3(M):
    return np.vstack([M, [0, 0, 1]])

cum_iou = {TILE_ORDER[0]: np.eye(3)}
for i in range(len(TILE_ORDER) - 1):
    a, b = TILE_ORDER[i], TILE_ORDER[i+1]
    warp = np.array(iou_transforms[f'{a}_to_{b}']['warp_matrix'], dtype=np.float64)
    cum_iou[b] = cum_iou[a] @ to_3x3(warp)

# Canvas bounds
corners_all = []
for k in TILE_ORDER:
    for c in [[0,0],[4200,0],[4200,4200],[0,4200]]:
        p = cum_iou[k] @ [c[0], c[1], 1]
        corners_all.append(p[:2])
corners_all = np.array(corners_all)
x_min, y_min = corners_all.min(axis=0)
for k in TILE_ORDER:
    cum_iou[k][0, 2] -= x_min
    cum_iou[k][1, 2] -= y_min

canvas_w = int(np.ceil(corners_all[:, 0].max() - x_min))
canvas_h = int(np.ceil(corners_all[:, 1].max() - y_min))

total_z = len(TILE_ORDER) * 12 + ROW1_4_GAP
print(f"  Canvas: {canvas_w}x{canvas_h}, {total_z} z-slices")

# DS3 dimensions
ds_w = canvas_w // DS
ds_h = canvas_h // DS
print(f"  DS{DS}: ({total_z}, {ds_h}, {ds_w})")

# ============================================================
# 2. Stitch into DS3 volume (masked, IOU-only)
# ============================================================
volume = np.zeros((total_z, ds_h, ds_w), dtype=np.float32)
weight = np.zeros((total_z, ds_h, ds_w), dtype=np.float32)

z_offset = 0
print(f"\nStitching {len(TILE_ORDER)} tiles...")
for tile_idx, key in enumerate(TILE_ORDER):
    print(f"  [{tile_idx+1}/{len(TILE_ORDER)}] {key} z={z_offset}", flush=True)

    mask = masks.get(key, np.ones((4200, 4200), dtype=np.uint8))

    if tile_idx > 0:
        prev_key = TILE_ORDER[tile_idx - 1]
        pair_warp = np.array(iou_transforms[f'{prev_key}_to_{key}']['warp_matrix'], dtype=np.float32)
        M_prev = cum_iou[prev_key][:2, :]
    else:
        M_cum = cum_iou[key][:2, :]

    for zi in range(12):
        img = cv2.imread(f'{BASE}/png_exports/registration_video/{key}/GFP_z{zi:03d}.png',
                         cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        sl = img.astype(np.float32) * mask.astype(np.float32)

        if tile_idx == 0:
            warped = cv2.warpAffine(sl, M_cum, (canvas_w, canvas_h),
                                    flags=cv2.INTER_LINEAR, borderValue=0)
            warped_m = cv2.warpAffine(mask.astype(np.float32), M_cum, (canvas_w, canvas_h),
                                      flags=cv2.INTER_LINEAR, borderValue=0)
        else:
            sl_r = cv2.warpAffine(sl, pair_warp, (4200, 4200),
                                  flags=cv2.INTER_LINEAR, borderValue=0)
            m_r = cv2.warpAffine(mask.astype(np.float32), pair_warp, (4200, 4200),
                                 flags=cv2.INTER_LINEAR, borderValue=0)
            warped = cv2.warpAffine(sl_r, M_prev, (canvas_w, canvas_h),
                                    flags=cv2.INTER_LINEAR, borderValue=0)
            warped_m = cv2.warpAffine(m_r, M_prev, (canvas_w, canvas_h),
                                      flags=cv2.INTER_LINEAR, borderValue=0)

        # DS3 subsample
        sl_ds = warped[::DS, ::DS][:ds_h, :ds_w]
        w_ds = warped_m[::DS, ::DS][:ds_h, :ds_w]

        volume[z_offset + zi] += sl_ds
        weight[z_offset + zi] += w_ds

    z_offset += 12
    if key == ROW1_4_INSERT_AFTER:
        z_offset += ROW1_4_GAP

# Normalize overlapping regions
valid = weight > 0
volume[valid] /= weight[valid]

# ============================================================
# 3. Per-slice equalization (same as v5 viewer)
# ============================================================
print("\nPer-slice equalization...")
all_vals = volume[volume > 0]
gmean = all_vals.mean()
for z in range(total_z):
    sl = volume[z]
    m = sl > 0
    if m.sum() > 100:
        smean = sl[m].mean()
        if smean > 1:
            volume[z][m] *= (gmean / smean)

# Normalize by /4000 (same as v5 viewer)
ex_u8 = np.clip(volume / NORM * 255, 0, 255).astype(np.uint8)
del volume

# ============================================================
# 4. GP interpolation in z (2x, same as v5 viewer)
# ============================================================
print(f"GP interpolation: {total_z} → ", end='')
W, target_zs = gp_weight_matrix(total_z, GP_INTERP, GP_LENGTHSCALE, GP_NOISE)
nz_out = len(target_zs)
print(f"{nz_out} z-levels")

flat = ex_u8.reshape(total_z, -1).astype(np.float64)
gp_flat = np.clip(W @ flat, 0, 255).astype(np.uint8)
gp_vol = gp_flat.reshape(nz_out, ds_h, ds_w)
del ex_u8, flat, gp_flat

# ============================================================
# 5. Extract sparse voxels
# ============================================================
print(f"Extracting voxels (threshold={VOXEL_THRESH})...")
ez, ey, exx = np.where(gp_vol > VOXEL_THRESH)
ex_vals = gp_vol[ez, ey, exx]
n_vox = len(ez)
print(f"  {n_vox:,} voxels")

# Subsample if too many (keep under ~2M for <80MB HTML)
MAX_VOXELS = 2000000
if n_vox > MAX_VOXELS:
    idx = np.random.RandomState(42).choice(n_vox, MAX_VOXELS, replace=False)
    ez, ey, exx, ex_vals = ez[idx], ey[idx], exx[idx], ex_vals[idx]
    n_vox = MAX_VOXELS
    print(f"  Subsampled to {n_vox:,}")

# Normalize coordinates (centroid-centered, same as v5)
ex_span = float(max(ds_w, ds_h, nz_out))
vx = exx.astype(np.float32) / ex_span
vy = ey.astype(np.float32) / ex_span
vz = np.array([target_zs[z] / ex_span for z in ez], dtype=np.float32)

cx, cy, cz = vx.mean(), vy.mean(), vz.mean()
vx -= cx
vy -= cy
vz -= cz
vv = ex_vals.astype(np.float32) / 255.0
del gp_vol

print(f"  Span: {ex_span:.0f}, centroid shift: ({0.5-cx:.3f}, {0.5-cy:.3f}, {0.5-cz:.3f})")

# ============================================================
# 6. Encode and build HTML
# ============================================================
print("Encoding...")
data = np.zeros(n_vox, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('i', 'f4')])
data['x'] = vx
data['y'] = -vy  # flip y
data['z'] = vz
data['i'] = vv
b64_data = base64.b64encode(data.tobytes()).decode('ascii')
print(f"  {n_vox:,} points, {len(b64_data)//1024} KB")

print("Building HTML...")

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>IOU-Stitched Ex-vivo 3D (Masked)</title>
<style>
  body {{ margin: 0; overflow: hidden; background: #000; font-family: -apple-system, sans-serif; }}
  #info {{ position: absolute; top: 10px; left: 10px; color: #eee; font-size: 13px;
    background: rgba(0,0,0,0.7); padding: 10px; border-radius: 6px; z-index: 10; max-width: 360px; }}
  #controls {{ position: absolute; top: 10px; right: 10px; color: #eee; font-size: 12px;
    background: rgba(0,0,0,0.7); padding: 10px; border-radius: 6px; z-index: 10; }}
  #controls label {{ display: block; margin: 5px 0; }}
  #controls input[type=range] {{ width: 120px; vertical-align: middle; }}
</style>
</head>
<body>
<div id="info">
  <b>IOU-Stitched Ex-vivo (Masked, No Elastix)</b><br>
  {len(TILE_ORDER)} tiles, {total_z} z &rarr; GP2x &rarr; {nz_out} z<br>
  DS{DS}: ({nz_out}, {ds_h}, {ds_w}), equalized, /4000<br>
  {n_vox:,} voxels (thresh {VOXEL_THRESH})<br>
  Drag to rotate, scroll to zoom
</div>
<div id="controls">
  <label>Threshold: <input type="range" id="threshSlider" min="1" max="60" value="{VOXEL_THRESH}" oninput="updateThresh()">
    <span id="threshVal">{VOXEL_THRESH}</span></label>
  <label>Point size: <input type="range" id="sizeSlider" min="1" max="40" value="10" oninput="updateSize()">
    <span id="sizeVal">0.010</span></label>
  <label>Opacity: <input type="range" id="opacSlider" min="5" max="100" value="60" oninput="updateOpac()">
    <span id="opacVal">0.60</span></label>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
var b64 = "{b64_data}";
var raw = Uint8Array.from(atob(b64), function(c){{ return c.charCodeAt(0); }});
var floats = new Float32Array(raw.buffer);
var nPts = floats.length / 4;

var allPos = new Float32Array(nPts * 3);
var allInt = new Float32Array(nPts);
for (var i = 0; i < nPts; i++) {{
  allPos[i*3]   = floats[i*4];
  allPos[i*3+1] = floats[i*4+1];
  allPos[i*3+2] = floats[i*4+2];
  allInt[i]     = floats[i*4+3];
}}

var scene = new THREE.Scene();
var camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.01, 100);
camera.position.set(0, 0, 1.5);
var renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);

var geometry, material, points;
var currentThresh = {VOXEL_THRESH} / 255.0;

function buildCloud(thresh) {{
  if (points) scene.remove(points);
  if (geometry) geometry.dispose();
  var pos = [], col = [];
  for (var i = 0; i < nPts; i++) {{
    if (allInt[i] >= thresh) {{
      pos.push(allPos[i*3], allPos[i*3+1], allPos[i*3+2]);
      var v = Math.min(allInt[i] * 1.5, 1.0);
      col.push(0.15*v, v, 0.25*v);
    }}
  }}
  geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  material = new THREE.PointsMaterial({{
    size: parseFloat(document.getElementById('sizeVal').textContent),
    vertexColors: true, transparent: true,
    opacity: parseFloat(document.getElementById('opacVal').textContent),
    sizeAttenuation: true
  }});
  points = new THREE.Points(geometry, material);
  scene.add(points);
}}

buildCloud(currentThresh);

var isDrag = false, ppx = 0, ppy = 0, rx = -0.3, ry = 0.4;
renderer.domElement.onmousedown = function(e) {{ isDrag = true; ppx = e.clientX; ppy = e.clientY; }};
renderer.domElement.onmouseup = function() {{ isDrag = false; }};
renderer.domElement.onmousemove = function(e) {{
  if (!isDrag) return;
  ry += (e.clientX - ppx) * 0.005;
  rx += (e.clientY - ppy) * 0.005;
  ppx = e.clientX; ppy = e.clientY;
}};
renderer.domElement.onwheel = function(e) {{
  camera.position.z *= e.deltaY > 0 ? 1.05 : 0.95;
  camera.position.z = Math.max(0.3, Math.min(5, camera.position.z));
}};

function updateThresh() {{
  var v = parseInt(document.getElementById('threshSlider').value);
  document.getElementById('threshVal').textContent = v;
  currentThresh = v / 255.0;
  buildCloud(currentThresh);
}}
function updateSize() {{
  var v = parseInt(document.getElementById('sizeSlider').value);
  var s = v / 1000;
  document.getElementById('sizeVal').textContent = s.toFixed(3);
  if (material) material.size = s;
}}
function updateOpac() {{
  var v = parseInt(document.getElementById('opacSlider').value);
  var o = v / 100;
  document.getElementById('opacVal').textContent = o.toFixed(2);
  if (material) material.opacity = o;
}}

function animate() {{
  requestAnimationFrame(animate);
  if (points) {{ points.rotation.x = rx; points.rotation.y = ry; }}
  renderer.render(scene, camera);
}}
animate();
onresize = function() {{
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}};
</script>
</body>
</html>"""

OUT = f"{BASE}/iou_stitched_3d.html"
with open(OUT, 'w') as f:
    f.write(html)
print(f"\nSaved: {OUT}")
print(f"Size: {len(html)/1024/1024:.1f} MB")
