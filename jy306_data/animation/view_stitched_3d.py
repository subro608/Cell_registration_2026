"""
Build an interactive 3D HTML viewer for the stitched volume.
Subsamples voxels, encodes as a Three.js point cloud with real colors.
"""
import numpy as np, pickle, json, os

BASE = '/Users/neurolab/neuroinformatics/margaret'
with open(f'{BASE}/animation/scene5b_assets_v3.pkl', 'rb') as f:
    assets = pickle.load(f)

vol = assets['_stitched']['volume']   # (nz, h, w, 3) uint8
z_vals = assets['_stitched']['z']
print(f"Volume: {vol.shape}, z range: {z_vals[0]:.1f} to {z_vals[-1]:.1f}")

# Subsample: take every 2nd slice, every 4th pixel
z_step, xy_step = 2, 4
brightness_thresh = 15  # skip dark voxels

positions = []
colors = []

for zi in range(0, vol.shape[0], z_step):
    sl = vol[zi]
    z_val = z_vals[zi]
    for y in range(0, sl.shape[0], xy_step):
        for x in range(0, sl.shape[1], xy_step):
            r, g, b = int(sl[y, x, 0]), int(sl[y, x, 1]), int(sl[y, x, 2])
            if max(r, g, b) < brightness_thresh:
                continue
            positions.append([float(x), float(-y), float(z_val * 3)])  # z exaggerated
            colors.append([r / 255, g / 255, b / 255])

print(f"Points: {len(positions)}")

html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Stitched 3D Volume</title>
<style>body {{ margin: 0; overflow: hidden; background: #000; }}</style>
</head><body>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const positions = {json.dumps(positions)};
const colors = {json.dumps(colors)};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 1, 10000);
camera.position.set(300, -200, 800);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// Build point cloud
const geo = new THREE.BufferGeometry();
const pos = new Float32Array(positions.length * 3);
const col = new Float32Array(colors.length * 3);
for (let i = 0; i < positions.length; i++) {{
    pos[i*3] = positions[i][0];
    pos[i*3+1] = positions[i][1];
    pos[i*3+2] = positions[i][2];
    col[i*3] = colors[i][0];
    col[i*3+1] = colors[i][1];
    col[i*3+2] = colors[i][2];
}}
geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
geo.setAttribute('color', new THREE.BufferAttribute(col, 3));

const mat = new THREE.PointsMaterial({{ size: 2.5, vertexColors: true, sizeAttenuation: true }});
const cloud = new THREE.Points(geo, mat);

// Center
geo.computeBoundingBox();
const center = new THREE.Vector3();
geo.boundingBox.getCenter(center);
cloud.position.sub(center);
scene.add(cloud);

controls.target.set(0, 0, 0);

window.addEventListener('resize', () => {{
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
}});

function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}}
animate();
</script></body></html>"""

out_path = f'{BASE}/animation/stitched_volume_3d.html'
with open(out_path, 'w') as f:
    f.write(html)
print(f"Saved: {out_path}")
sz = os.path.getsize(out_path) / 1e6
print(f"File size: {sz:.1f} MB")
