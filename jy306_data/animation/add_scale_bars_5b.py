"""
Post-process scene5b_combined frames to add scale bars.

Phase breakdown (from combined folder):
  1-546:    multi_tile_3d (GRID_SCALE=1.0, 3D rotation) — ~3.2 um/disp_px
  547-582:  three_stacks split (3 col grids)             — ~10.7 um/disp_px (GRID_SCALE≈0.3)
  583-618:  three_stacks rotate split (3D)               — ~10.7 um/disp_px
  619-654:  three_stacks merge grids                     — varies (grid→merged)
  655-726:  tile merge (72fr, 3D)                        — ~3.2→1.77 um/disp_px
  727-762:  scale_up (36fr)                              — ~1.77 um/disp_px
  763-846:  rotation transition (84fr, VOLUME_SCALE=1.8) — 1.77 um/disp_px
  847-894:  split_channels (48fr)                        — 1.77→2.67 um/disp_px
  895-942:  hold_split_vols (48fr)                       — 2.67 um/disp_px
  943-978:  merge_channels (36fr)                        — 2.67→1.77 um/disp_px
  979-1013: hold_combined (35fr)                         — 1.77 um/disp_px
  1014-1025: fade to black                               — skip

For 3D volumes: scale bar represents the XY scale at the front face (no perspective correction).
This matches scene 7's convention.
"""
import cv2, numpy as np, os, glob

DIR = '/Users/neurolab/neuroinformatics/margaret/animation/frames_scene5b_combined'
W, H = 1920, 1080
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Scale values computed from assets (ex-vivo = 0.65 µm/px)
UM_GRID1 = 3.21      # multi_tile_3d at GRID_SCALE=1.0 (avg across tiles)
UM_GRID3 = 10.74     # three_stacks 3-column grid (GRID_SCALE≈0.3)
UM_VOL18 = 1.79      # VOLUME_SCALE=1.8 (3.21/1.8)
UM_VOLSPLIT = 2.69   # VOL_SCALE_SPLIT≈1.19 (3.21/1.19)
UM_MERGED = 5.59      # single merged grid (MERGED_SCALE≈0.575)


def draw_scale_bar(frame, um_per_disp_px, alpha=1.0, x_right=None, y_bottom=None):
    if alpha < 0.01:
        return
    if x_right is None:
        x_right = W - 50
    if y_bottom is None:
        y_bottom = H - 75
    for target_um in [10, 20, 50, 100, 200, 500, 1000]:
        bar_px = int(round(target_um / um_per_disp_px))
        if 80 <= bar_px <= 200:
            break
    if bar_px < 20 or bar_px > 400:
        return  # can't fit a reasonable bar
    x_left = x_right - bar_px
    y_bar = y_bottom
    col = tuple(int(255 * alpha) for _ in range(3))
    bg = (0, 0, 0)
    cv2.rectangle(frame, (x_left - 1, y_bar - 4), (x_right + 1, y_bar + 4), bg, -1)
    cv2.rectangle(frame, (x_left, y_bar - 3), (x_right, y_bar + 3), col, -1)
    label = f'{target_um} um'
    ts = 0.45
    (tw, _), _ = cv2.getTextSize(label, FONT, ts, 1)
    tx = x_left + (bar_px - tw) // 2
    ty = y_bar - 8
    cv2.putText(frame, label, (tx, ty), FONT, ts, bg, 3, cv2.LINE_AA)
    cv2.putText(frame, label, (tx, ty), FONT, ts, col, 1, cv2.LINE_AA)


def ease(t):
    import math
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))


frames = sorted(glob.glob(f'{DIR}/frame_*.png'))
print(f"Processing {len(frames)} frames...")

count = 0
for path in frames:
    fname = os.path.basename(path)
    fi = int(fname.split('_')[1].split('.')[0])

    if fi >= 1014:
        continue  # fade to black, skip

    # Determine um/disp_px for this frame
    if fi <= 546:
        # multi_tile_3d: all at GRID_SCALE=1.0
        # Phase A (1-36): zooming from FULL_SCALE to GRID_SCALE
        if fi <= 36:
            FULL_SCALE = 2.95
            t = ease((fi - 1) / 35)
            scale = FULL_SCALE * (1 - t) + 1.0 * t
            um = UM_GRID1 / scale  # larger display = smaller um/px
        else:
            um = UM_GRID1
        x_r, y_b = W - 50, H - 75
    elif fi <= 582:
        # three_stacks split: 3 column grids
        um = UM_GRID3
        x_r, y_b = W - 50, H - 75
    elif fi <= 618:
        # rotate split: 3D but still grid scale
        um = UM_GRID3
        x_r, y_b = W - 50, H - 75
    elif fi <= 654:
        # merge grids: 3 cols → 1 center, scale transitions
        t = ease((fi - 619) / 35)
        um = UM_GRID3 * (1 - t) + UM_MERGED * t
        x_r, y_b = W - 50, H - 75
    elif fi <= 726:
        # tile merge: grid → 3D volume
        t = (fi - 655) / 71
        um = UM_MERGED * (1 - t) + UM_VOL18 * t
        x_r, y_b = W - 50, H - 75
    elif fi <= 762:
        # scale_up
        um = UM_VOL18
        x_r, y_b = W - 50, H - 75
    elif fi <= 846:
        # rotation transition
        um = UM_VOL18
        x_r, y_b = W - 50, H - 75
    elif fi <= 894:
        # split_channels: center → 3 columns
        t = ease((fi - 847) / 47)
        um = UM_VOL18 * (1 - t) + UM_VOLSPLIT * t
        x_r, y_b = W - 50, H - 75
    elif fi <= 942:
        # hold_split_vols
        um = UM_VOLSPLIT
        x_r, y_b = W - 50, H - 75
    elif fi <= 978:
        # merge_channels
        t = ease((fi - 943) / 35)
        um = UM_VOLSPLIT * (1 - t) + UM_VOL18 * t
        x_r, y_b = W - 50, H - 75
    elif fi <= 1013:
        # hold_combined
        um = UM_VOL18
        x_r, y_b = W - 50, H - 75
    else:
        continue

    frame = cv2.imread(path)
    draw_scale_bar(frame, um, x_right=x_r, y_bottom=y_b)
    cv2.imwrite(path, frame)
    count += 1
    if count % 100 == 0:
        print(f"  {count}/{len(frames)}")

print(f"Done! Added scale bars to {count} frames.")
