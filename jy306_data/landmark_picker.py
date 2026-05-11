"""
Interactive landmark picker for z-slice registration.

SOURCE: png_exports/registration_video/nd2_1/ch1_z*.png  (6 slices, 4200x4200)
TARGET: png_exports/exvivo_combined_registered/z*.png     (16 slices, 658x629)

Controls:
  LEFT-CLICK  — place point (alternates: source, then target)
  A / D       — prev / next SOURCE z-slice
  J / L       — prev / next TARGET z-slice
  Z           — undo last point
  S           — save landmarks and quit
  Q           — quit without saving
"""

import numpy as np
import cv2
import os
import glob

# ============================================================
# Load z-slice images
# ============================================================
src_dir = '/Users/neurolab/neuroinformatics/margaret/png_exports/registration_video/nd2_1'
tgt_dir = '/Users/neurolab/neuroinformatics/margaret/png_exports/exvivo_combined_registered'

src_files = sorted(glob.glob(os.path.join(src_dir, 'ch1_z*.png')))
tgt_files = sorted([f for f in glob.glob(os.path.join(tgt_dir, 'z*.png')) if 'MIP' not in f])

print(f"Source slices: {len(src_files)}")
for f in src_files:
    print(f"  {os.path.basename(f)}")
print(f"Target slices: {len(tgt_files)}")
for f in tgt_files:
    print(f"  {os.path.basename(f)}")

# Load all images
src_imgs = []
for f in src_files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    src_imgs.append(img)
    print(f"  Loaded {os.path.basename(f)}: {img.shape}")

tgt_imgs = []
for f in tgt_files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    tgt_imgs.append(img)

print(f"\nSource image size: {src_imgs[0].shape}")
print(f"Target image size: {tgt_imgs[0].shape}")

# ============================================================
# Display scaling
# ============================================================
DISP_H = 800
src_h, src_w = src_imgs[0].shape
src_scale = DISP_H / src_h
src_disp_w = int(src_w * src_scale)

tgt_h, tgt_w = tgt_imgs[0].shape
tgt_scale = DISP_H / tgt_h
tgt_disp_w = int(tgt_w * tgt_scale)

GAP = 20

# ============================================================
# State
# ============================================================
src_z_idx = 0
tgt_z_idx = 0
src_points = []   # list of (col, row, z_idx) in ORIGINAL source coords
tgt_points = []   # list of (col, row, z_idx) in ORIGINAL target coords
next_is_src = True

COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
    (255, 128, 0), (128, 255, 0), (0, 255, 128), (255, 0, 128),
    (100, 100, 255), (100, 255, 100), (255, 100, 100), (200, 200, 0),
]

def build_display():
    # Source
    left = cv2.resize(src_imgs[src_z_idx], (src_disp_w, DISP_H))
    left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)

    # Target
    right = cv2.resize(tgt_imgs[tgt_z_idx], (tgt_disp_w, DISP_H))
    right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

    # Draw all source points (dim if different z, bright if same z)
    for i, (x, y, zi) in enumerate(src_points):
        dx, dy = int(x * src_scale), int(y * src_scale)
        col = COLORS[i % len(COLORS)]
        if zi == src_z_idx:
            cv2.circle(left, (dx, dy), 6, col, 2, cv2.LINE_AA)
            cv2.putText(left, str(i+1), (dx+8, dy-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        else:
            dim = tuple(int(v*0.3) for v in col)
            cv2.circle(left, (dx, dy), 4, dim, 1, cv2.LINE_AA)
            cv2.putText(left, str(i+1), (dx+6, dy-3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, dim, 1, cv2.LINE_AA)

    # Draw all target points
    for i, (x, y, zi) in enumerate(tgt_points):
        dx, dy = int(x * tgt_scale), int(y * tgt_scale)
        col = COLORS[i % len(COLORS)]
        if zi == tgt_z_idx:
            cv2.circle(right, (dx, dy), 6, col, 2, cv2.LINE_AA)
            cv2.putText(right, str(i+1), (dx+8, dy-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        else:
            dim = tuple(int(v*0.3) for v in col)
            cv2.circle(right, (dx, dy), 4, dim, 1, cv2.LINE_AA)
            cv2.putText(right, str(i+1), (dx+6, dy-3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, dim, 1, cv2.LINE_AA)

    # Slice labels
    src_name = os.path.basename(src_files[src_z_idx])
    tgt_name = os.path.basename(tgt_files[tgt_z_idx])
    cv2.putText(left, f"SOURCE: {src_name}  (A/D = prev/next)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(right, f"TARGET: {tgt_name}  (J/L = prev/next)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    gap = np.zeros((DISP_H, GAP, 3), dtype=np.uint8)
    combined = np.hstack([left, gap, right])

    # Status bar
    n_pairs = min(len(src_points), len(tgt_points))
    if next_is_src:
        status = f"Pairs: {n_pairs} | Click SOURCE (left) for point {n_pairs+1}"
    else:
        status = f"Pairs: {n_pairs} | Click TARGET (right) for point {n_pairs+1}"

    bar = np.zeros((35, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, "S=save  Z=undo  Q=quit", (combined.shape[1]-280, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    return np.vstack([combined, bar])


def on_mouse(event, x, y, flags, param):
    global next_is_src

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if x < src_disp_w:
        # Source click
        if not next_is_src:
            print("  -> Click TARGET (right) next!")
            return
        orig_x = x / src_scale
        orig_y = y / src_scale
        src_points.append((orig_x, orig_y, src_z_idx))
        print(f"  Source point {len(src_points)}: ({orig_x:.1f}, {orig_y:.1f}) z={src_z_idx} [{os.path.basename(src_files[src_z_idx])}]")
        next_is_src = False

    elif x > src_disp_w + GAP:
        # Target click
        if next_is_src:
            print("  -> Click SOURCE (left) first!")
            return
        rx = x - src_disp_w - GAP
        orig_x = rx / tgt_scale
        orig_y = y / tgt_scale
        tgt_points.append((orig_x, orig_y, tgt_z_idx))
        print(f"  Target point {len(tgt_points)}: ({orig_x:.1f}, {orig_y:.1f}) z={tgt_z_idx} [{os.path.basename(tgt_files[tgt_z_idx])}]")
        next_is_src = True


# ============================================================
# Main loop
# ============================================================
WIN = "Landmark Picker — z-slice registration"
cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(WIN, on_mouse)

print("\n=== LANDMARK PICKER ===")
print("Click matching cells: left first, then right")
print("A/D = source z-slice, J/L = target z-slice")
print("S=save, Z=undo, Q=quit\n")

while True:
    disp = build_display()
    cv2.imshow(WIN, disp)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('a') or key == ord('A'):
        src_z_idx = max(0, src_z_idx - 1)
        print(f"  Source z: {os.path.basename(src_files[src_z_idx])}")
    elif key == ord('d') or key == ord('D'):
        src_z_idx = min(len(src_imgs)-1, src_z_idx + 1)
        print(f"  Source z: {os.path.basename(src_files[src_z_idx])}")
    elif key == ord('j') or key == ord('J'):
        tgt_z_idx = max(0, tgt_z_idx - 1)
        print(f"  Target z: {os.path.basename(tgt_files[tgt_z_idx])}")
    elif key == ord('l') or key == ord('L'):
        tgt_z_idx = min(len(tgt_imgs)-1, tgt_z_idx + 1)
        print(f"  Target z: {os.path.basename(tgt_files[tgt_z_idx])}")

    elif key == ord('s') or key == ord('S'):
        n_pairs = min(len(src_points), len(tgt_points))
        if n_pairs < 3:
            print(f"Need at least 3 pairs, have {n_pairs}")
            continue

        src_pts = np.array(src_points[:n_pairs])
        tgt_pts = np.array(tgt_points[:n_pairs])

        out_path = '/Users/neurolab/neuroinformatics/margaret/registration_video/landmarks.npz'
        np.savez(out_path,
                 src_points=src_pts,    # (N, 3) col, row, z_idx
                 tgt_points=tgt_pts,    # (N, 3) col, row, z_idx
                 src_shape=(len(src_imgs), src_h, src_w),
                 tgt_shape=(len(tgt_imgs), tgt_h, tgt_w),
                 src_files=[os.path.basename(f) for f in src_files],
                 tgt_files=[os.path.basename(f) for f in tgt_files])
        print(f"\nSaved {n_pairs} landmark pairs to {out_path}")
        break

    elif key == ord('z') or key == ord('Z'):
        if not next_is_src and len(src_points) > len(tgt_points):
            removed = src_points.pop()
            next_is_src = True
            print(f"  Undid source point {len(src_points)+1}")
        elif next_is_src and len(tgt_points) > 0 and len(tgt_points) >= len(src_points):
            removed = tgt_points.pop()
            next_is_src = False
            print(f"  Undid target point {len(tgt_points)+1}")
        elif next_is_src and len(src_points) > 0:
            removed = src_points.pop()
            print(f"  Undid source point {len(src_points)+1}")

    elif key == ord('q') or key == ord('Q'):
        print("Quit without saving")
        break

cv2.destroyAllWindows()
print("Done!")
