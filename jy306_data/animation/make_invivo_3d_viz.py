"""Quick 3D test: ex-vivo (green) + in-vivo (red/hot), same data as v4 HTML viewer."""
import numpy as np, cv2, math, subprocess, os

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT  = f'{BASE}/animation/invivo_3d_test.mp4'
TMP  = f'{BASE}/animation/invivo_3d_test_raw.mp4'
W, H = 1280, 720; FPS = 24; FONT = cv2.FONT_HERSHEY_SIMPLEX

print("Loading viewer_v4_voxels.npz…")
vox = np.load('/tmp/viewer_v4_voxels.npz')
ex_x = vox['ex_vx'].astype(np.float32); ex_y = vox['ex_vy'].astype(np.float32)
ex_z = vox['ex_vz'].astype(np.float32); ex_v = vox['ex_vv'].astype(np.float32)
iv_x = vox['iv_vx'].astype(np.float32); iv_y = vox['iv_vy'].astype(np.float32)
iv_z = vox['iv_vz'].astype(np.float32); iv_v = vox['iv_vv'].astype(np.float32)
print(f"  ex={len(ex_x):,}  iv={len(iv_x):,}")

# Normalise intensities
ex_v = (ex_v - ex_v.min()) / (ex_v.max() - ex_v.min() + 1e-8)
iv_v = (iv_v - iv_v.min()) / (iv_v.max() - iv_v.min() + 1e-8)

# Centre on ex-vivo centroid (same as v4 viewer)
cx = (ex_x.max()+ex_x.min())*0.5
cy = (ex_y.max()+ex_y.min())*0.5
cz = (ex_z.max()+ex_z.min())*0.5
ex_x -= cx; ex_y -= cy; ex_z -= cz
iv_x -= cx; iv_y -= cy; iv_z -= cz

span = max(ex_x.max()-ex_x.min(), ex_y.max()-ex_y.min())
THUMB = 640
scale = int(THUMB * 0.82 / span)

def render_cloud(x, y, z, v, rot_y, rot_x, alpha, col_f32, thumb):
    canvas = np.zeros((thumb, thumb, 3), np.float32)
    if alpha < 0.005: return canvas
    cy_r, sy_r = math.cos(rot_y), math.sin(rot_y)
    rx = cy_r*x + sy_r*z; rz = -sy_r*x + cy_r*z
    cx_r, sx_r = math.cos(rot_x), math.sin(rot_x)
    ry2 = cx_r*y - sx_r*rz; rz2 = sx_r*y + cx_r*rz
    px = (rx*scale + thumb//2).astype(np.int32)
    py = (ry2*scale + thumb//2).astype(np.int32)
    mask = (px>=0)&(px<thumb)&(py>=0)&(py<thumb)
    px, py, vv, dd = px[mask], py[mask], v[mask], rz2[mask]
    order = np.argsort(-dd)
    px, py, vv = px[order], py[order], vv[order]
    iv_ = (vv * alpha).astype(np.float32)
    for c in range(3):
        np.maximum.at(canvas[:,:,c], (py, px), iv_ * col_f32[c])
    return canvas

EX_COL = np.array([0, 0.78, 0], np.float32)     # green (BGR)
IV_COL = np.array([0.15, 0.25, 0.95], np.float32) # red-hot (BGR)

N = 120   # 5 seconds
TILT = math.radians(12)

print(f"Rendering {N} frames…")
vw = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

for fi in range(N):
    t = fi / N
    rot_y = math.radians(5 + 40 * math.sin(t * 2 * math.pi))
    rot_x = TILT + math.radians(6 * math.sin(t * math.pi))

    c_ex = render_cloud(ex_x, ex_y, ex_z, ex_v, rot_y, rot_x, 1.0, EX_COL, THUMB)
    c_iv = render_cloud(iv_x, iv_y, iv_z, iv_v, rot_y, rot_x, 1.0, IV_COL, THUMB)

    canvas = np.maximum(c_ex, c_iv)
    img = np.clip(canvas * 255, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3,3), 0.8)

    frame = np.zeros((H, W, 3), np.uint8)
    x0 = W//2 - THUMB//2; y0 = H//2 - THUMB//2
    frame[y0:y0+THUMB, x0:x0+THUMB] = img

    # Legend
    cv2.circle(frame, (28, 28), 7, (0,200,0), -1, cv2.LINE_AA)
    cv2.putText(frame, 'EX VIVO  CONFOCAL', (42, 32), FONT, 0.38, (0,200,0), 1, cv2.LINE_AA)
    cv2.circle(frame, (28, 52), 7, (40,60,240), -1, cv2.LINE_AA)
    cv2.putText(frame, 'IN VIVO  CALCIUM', (42, 56), FONT, 0.38, (40,60,240), 1, cv2.LINE_AA)

    vw.write(frame)
    if fi % 24 == 0: print(f"  {fi}/{N}")

vw.release()
subprocess.run(['ffmpeg','-y','-i',TMP,
                '-vcodec','libx264','-pix_fmt','yuv420p','-crf','16','-preset','fast',
                OUT], check=True, capture_output=True)
os.remove(TMP)
print(f"Done → {OUT}")
