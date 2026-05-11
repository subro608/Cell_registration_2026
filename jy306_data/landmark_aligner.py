"""
Interactive landmark-based slice aligner.

For each adjacent tile pair, shows the last z-slice of tile A and first z-slice
of tile B side-by-side. User clicks 3-4 matching cell landmarks in each image.
A rigid transform (rotation + translation) is computed from the landmark pairs
and applied. An overlay is shown for verification before moving to the next pair.

Usage:
    python landmark_aligner.py                  # start from first pair
    python landmark_aligner.py --start-pair 5   # resume from pair index 5
    python landmark_aligner.py --review          # review saved transforms
"""

import numpy as np
import nd2
import cv2
import os
import json
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from skimage.transform import estimate_transform

# ============================================================
# Tile order and paths
# ============================================================
TILE_ORDER = [
    'row1_1', 'row1_2', 'row1_3',
    # row1_4 excluded (experiment failure) — black slab inserted during stitching
    'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
    'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
    'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
    'row5_1',
]

BASE = '/Users/neurolab/neuroinformatics/margaret'

def nd2_path(key):
    row, tile = key.split('_')
    if row == 'row5':
        return f'{BASE}/registration_video/row5/Row5/{tile}.nd2'
    return f'{BASE}/registration_video/{row}/{tile}.nd2'

# ============================================================
# Load boundary slices
# ============================================================
def load_last_slice(key, masks):
    """Load last z-slice GFP from tile, masked."""
    path = nd2_path(key)
    with nd2.ND2File(path) as f:
        data = f.asarray()  # (12, 2, 4200, 4200)
    gfp = data[-1, 1].astype(np.float32)  # last z, GFP channel
    if key in masks:
        gfp *= masks[key].astype(np.float32)
    return gfp

def load_first_slice(key, masks):
    """Load first z-slice GFP from tile, masked."""
    path = nd2_path(key)
    with nd2.ND2File(path) as f:
        data = f.asarray()
    gfp = data[0, 1].astype(np.float32)  # first z, GFP channel
    if key in masks:
        gfp *= masks[key].astype(np.float32)
    return gfp

def normalize_for_display(img):
    """Normalize to 0-1 for display using robust percentile scaling."""
    vals = img[img > 0]
    if len(vals) == 0:
        return np.zeros_like(img)
    p2, p995 = np.percentile(vals, [2, 99.5])
    out = np.clip((img - p2) / max(p995 - p2, 1), 0, 1)
    return out

# ============================================================
# Landmark picker
# ============================================================
class LandmarkPicker:
    """Interactive landmark picker for two images."""

    def __init__(self, img_a, img_b, title_a, title_b, pair_label):
        self.img_a = normalize_for_display(img_a)
        self.img_b = normalize_for_display(img_b)
        self.title_a = title_a
        self.title_b = title_b
        self.pair_label = pair_label
        self.pts_a = []
        self.pts_b = []
        self.current_side = 'A'  # which image to add points to
        self.markers_a = []
        self.markers_b = []
        self.result = None  # 'accept', 'skip', 'quit'

    def run(self):
        self.fig, (self.ax_a, self.ax_b) = plt.subplots(1, 2, figsize=(18, 9))
        self.fig.suptitle(f'{self.pair_label}\nClick LEFT image first, then RIGHT image. '
                          f'Alternate: L1, R1, L2, R2, ... (min 3 pairs)\n'
                          f'Current: clicking on LEFT (green border)',
                          fontsize=11)

        self.ax_a.imshow(self.img_a, cmap='gray', vmin=0, vmax=1)
        self.ax_a.set_title(f'{self.title_a} (last z-slice)', color='lime')
        self.ax_a.axis('off')

        self.ax_b.imshow(self.img_b, cmap='gray', vmin=0, vmax=1)
        self.ax_b.set_title(f'{self.title_b} (first z-slice)', color='cyan')
        self.ax_b.axis('off')

        # Highlight active side
        for spine in self.ax_a.spines.values():
            spine.set_edgecolor('lime')
            spine.set_linewidth(3)
            spine.set_visible(True)

        # Buttons
        ax_compute = self.fig.add_axes([0.3, 0.02, 0.12, 0.04])
        ax_skip = self.fig.add_axes([0.45, 0.02, 0.1, 0.04])
        ax_undo = self.fig.add_axes([0.58, 0.02, 0.1, 0.04])
        ax_quit = self.fig.add_axes([0.71, 0.02, 0.1, 0.04])

        self.btn_compute = Button(ax_compute, 'Compute (Enter)')
        self.btn_skip = Button(ax_skip, 'Skip')
        self.btn_undo = Button(ax_undo, 'Undo')
        self.btn_quit = Button(ax_quit, 'Quit')

        self.btn_compute.on_clicked(self._on_compute)
        self.btn_skip.on_clicked(self._on_skip)
        self.btn_undo.on_clicked(self._on_undo)
        self.btn_quit.on_clicked(self._on_quit)

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        plt.tight_layout(rect=[0, 0.07, 1, 0.93])
        plt.show()

        return self.result, np.array(self.pts_a), np.array(self.pts_b)

    def _update_title(self):
        n = min(len(self.pts_a), len(self.pts_b))
        side_str = 'LEFT (green border)' if self.current_side == 'A' else 'RIGHT (cyan border)'
        self.fig.suptitle(
            f'{self.pair_label}\n'
            f'{n} landmark pairs placed. Current: clicking on {side_str}\n'
            f'Click matching cells alternately. Min 3 pairs, then press Enter/Compute.',
            fontsize=11)

        # Update borders
        for spine in self.ax_a.spines.values():
            spine.set_edgecolor('lime' if self.current_side == 'A' else 'gray')
            spine.set_linewidth(3 if self.current_side == 'A' else 1)
            spine.set_visible(True)
        for spine in self.ax_b.spines.values():
            spine.set_edgecolor('cyan' if self.current_side == 'B' else 'gray')
            spine.set_linewidth(3 if self.current_side == 'B' else 1)
            spine.set_visible(True)

        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes == self.ax_a and self.current_side == 'A':
            self.pts_a.append([event.xdata, event.ydata])
            idx = len(self.pts_a)
            m = self.ax_a.plot(event.xdata, event.ydata, 'o', color='lime',
                               markersize=8, markeredgecolor='white', markeredgewidth=1)
            self.ax_a.annotate(str(idx), (event.xdata, event.ydata),
                               color='yellow', fontsize=10, fontweight='bold',
                               xytext=(8, 8), textcoords='offset points')
            self.markers_a.append(m)
            self.current_side = 'B'
            self._update_title()

        elif event.inaxes == self.ax_b and self.current_side == 'B':
            self.pts_b.append([event.xdata, event.ydata])
            idx = len(self.pts_b)
            m = self.ax_b.plot(event.xdata, event.ydata, 'o', color='cyan',
                               markersize=8, markeredgecolor='white', markeredgewidth=1)
            self.ax_b.annotate(str(idx), (event.xdata, event.ydata),
                               color='yellow', fontsize=10, fontweight='bold',
                               xytext=(8, 8), textcoords='offset points')
            self.markers_b.append(m)
            self.current_side = 'A'
            self._update_title()

    def _on_key(self, event):
        if event.key == 'enter':
            self._on_compute(None)
        elif event.key == 'escape':
            self._on_quit(None)
        elif event.key == 'z':
            self._on_undo(None)

    def _on_compute(self, event):
        n = min(len(self.pts_a), len(self.pts_b))
        if n < 3:
            self.fig.suptitle(f'Need at least 3 landmark pairs! Currently have {n}.',
                              fontsize=12, color='red')
            self.fig.canvas.draw_idle()
            return
        self.result = 'accept'
        plt.close(self.fig)

    def _on_skip(self, event):
        self.result = 'skip'
        plt.close(self.fig)

    def _on_quit(self, event):
        self.result = 'quit'
        plt.close(self.fig)

    def _on_undo(self, event):
        # Undo last point
        if self.current_side == 'A' and len(self.pts_b) > 0:
            self.pts_b.pop()
            self.current_side = 'B'
        elif self.current_side == 'B' and len(self.pts_a) > 0:
            self.pts_a.pop()
            self.current_side = 'A'

        # Redraw
        self.ax_a.cla()
        self.ax_b.cla()
        self.ax_a.imshow(self.img_a, cmap='gray', vmin=0, vmax=1)
        self.ax_b.imshow(self.img_b, cmap='gray', vmin=0, vmax=1)
        self.ax_a.set_title(f'{self.title_a} (last z-slice)', color='lime')
        self.ax_b.set_title(f'{self.title_b} (first z-slice)', color='cyan')
        self.ax_a.axis('off')
        self.ax_b.axis('off')

        for i, pt in enumerate(self.pts_a):
            self.ax_a.plot(pt[0], pt[1], 'o', color='lime', markersize=8,
                           markeredgecolor='white', markeredgewidth=1)
            self.ax_a.annotate(str(i+1), (pt[0], pt[1]), color='yellow',
                               fontsize=10, fontweight='bold',
                               xytext=(8, 8), textcoords='offset points')
        for i, pt in enumerate(self.pts_b):
            self.ax_b.plot(pt[0], pt[1], 'o', color='cyan', markersize=8,
                           markeredgecolor='white', markeredgewidth=1)
            self.ax_b.annotate(str(i+1), (pt[0], pt[1]), color='yellow',
                               fontsize=10, fontweight='bold',
                               xytext=(8, 8), textcoords='offset points')

        self._update_title()

# ============================================================
# Transform computation and verification
# ============================================================
def compute_rigid_transform(pts_src, pts_dst):
    """Compute rigid (Euclidean) transform: rotation + translation.
    Maps pts_src -> pts_dst."""
    tform = estimate_transform('euclidean', pts_src, pts_dst)
    return tform

def show_verification(img_a, img_b, tform, title_a, title_b, pair_label):
    """Show green/magenta overlay of aligned images for verification.
    Returns 'accept' or 'redo'."""
    h, w = img_a.shape

    # Warp img_b to align with img_a
    M = np.array([[tform.params[0, 0], tform.params[0, 1], tform.params[0, 2]],
                   [tform.params[1, 0], tform.params[1, 1], tform.params[1, 2]]],
                  dtype=np.float64)
    img_b_warped = cv2.warpAffine(img_b, M, (w, h), flags=cv2.INTER_LINEAR)

    # Normalize for display
    a_norm = normalize_for_display(img_a)
    b_norm = normalize_for_display(img_b_warped)

    # Green/magenta overlay
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    overlay[:, :, 0] = b_norm  # R = moving (magenta)
    overlay[:, :, 1] = a_norm  # G = reference (green)
    overlay[:, :, 2] = b_norm  # B = moving (magenta)

    # Checkerboard
    block = 100
    checker = np.zeros((h, w), dtype=bool)
    for cy in range(0, h, block):
        for cx in range(0, w, block):
            if ((cy // block) + (cx // block)) % 2 == 0:
                checker[cy:cy+block, cx:cx+block] = True
    checker_img = np.where(checker[..., None],
                           np.stack([a_norm]*3, axis=-1),
                           np.stack([b_norm]*3, axis=-1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle(f'{pair_label} — VERIFICATION\n'
                 f'Rotation: {np.degrees(tform.rotation):.2f}°, '
                 f'Translation: ({tform.translation[0]:.1f}, {tform.translation[1]:.1f})\n'
                 f'Press Enter to ACCEPT, R to REDO landmarks',
                 fontsize=11)

    ax1.imshow(checker_img)
    ax1.set_title('Checkerboard (A vs warped B)')
    ax1.axis('off')

    ax2.imshow(overlay)
    ax2.set_title('Green=A (ref), Magenta=warped B')
    ax2.axis('off')

    result = {'value': None}

    def on_key(event):
        if event.key == 'enter':
            result['value'] = 'accept'
            plt.close(fig)
        elif event.key == 'r':
            result['value'] = 'redo'
            plt.close(fig)

    # Buttons
    ax_accept = fig.add_axes([0.35, 0.02, 0.12, 0.04])
    ax_redo = fig.add_axes([0.52, 0.02, 0.12, 0.04])
    btn_accept = Button(ax_accept, 'Accept (Enter)')
    btn_redo = Button(ax_redo, 'Redo (R)')
    btn_accept.on_clicked(lambda e: (result.update({'value': 'accept'}), plt.close(fig)))
    btn_redo.on_clicked(lambda e: (result.update({'value': 'redo'}), plt.close(fig)))

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout(rect=[0, 0.07, 1, 0.93])
    plt.show()

    return result['value']

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Interactive landmark-based slice aligner')
    parser.add_argument('--start-pair', type=int, default=0,
                        help='Start from this pair index (0-based)')
    parser.add_argument('--review', action='store_true',
                        help='Review saved transforms without picking')
    args = parser.parse_args()

    # Load masks
    masks_data = np.load(f'{BASE}/registration_video/via_masks_v2.npz')
    masks = {k: masks_data[k] for k in masks_data.files}
    print(f"Loaded {len(masks)} masks")

    # Build pair list
    pairs = []
    for i in range(len(TILE_ORDER) - 1):
        pairs.append((TILE_ORDER[i], TILE_ORDER[i + 1]))
    print(f"{len(pairs)} pairs to align")

    # Load existing transforms if any
    save_path = f'{BASE}/registration_video/landmark_transforms.json'
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            saved = json.load(f)
        print(f"Loaded {len(saved)} existing transforms from {save_path}")
    else:
        saved = {}

    if args.review:
        print("\n=== Saved transforms ===")
        for pair_key, info in sorted(saved.items()):
            angle = info['angle_deg']
            tx, ty = info['translation']
            n = info['n_landmarks']
            print(f"  {pair_key}: angle={angle:.2f}°, tx={tx:.1f}, ty={ty:.1f} ({n} landmarks)")
        return

    # Process each pair
    for idx in range(args.start_pair, len(pairs)):
        key_a, key_b = pairs[idx]
        pair_key = f'{key_a}_to_{key_b}'
        is_cross_row = key_a.split('_')[0] != key_b.split('_')[0]
        tag = ' [CROSS-ROW]' if is_cross_row else ''

        print(f"\n=== Pair {idx}/{len(pairs)-1}: {pair_key}{tag} ===")

        if pair_key in saved:
            info = saved[pair_key]
            print(f"  Already aligned: angle={info['angle_deg']:.2f}°, "
                  f"tx={info['translation'][0]:.1f}, ty={info['translation'][1]:.1f}")
            resp = input("  [s]kip / [r]edo / [q]uit? ").strip().lower()
            if resp == 'q':
                break
            if resp != 'r':
                continue

        print(f"  Loading {key_a} last slice...", flush=True)
        img_a = load_last_slice(key_a, masks)
        print(f"  Loading {key_b} first slice...", flush=True)
        img_b = load_first_slice(key_b, masks)

        while True:
            # Pick landmarks
            label = f'Pair {idx}: {key_a} → {key_b}{tag}'
            picker = LandmarkPicker(img_a, img_b, key_a, key_b, label)
            result, pts_a, pts_b = picker.run()

            if result == 'quit':
                print("Quitting. Progress saved.")
                with open(save_path, 'w') as f:
                    json.dump(saved, f, indent=2)
                return
            if result == 'skip':
                print(f"  Skipped {pair_key}")
                break

            # Compute transform (maps B points to A points)
            n = min(len(pts_a), len(pts_b))
            pts_a_use = pts_a[:n]
            pts_b_use = pts_b[:n]

            tform = compute_rigid_transform(pts_b_use, pts_a_use)
            angle_deg = np.degrees(tform.rotation)
            tx, ty = tform.translation

            print(f"  Transform: rotation={angle_deg:.2f}°, "
                  f"translation=({tx:.1f}, {ty:.1f})")

            # Show verification
            verify = show_verification(img_a, img_b, tform, key_a, key_b, label)

            if verify == 'accept':
                saved[pair_key] = {
                    'angle_deg': float(angle_deg),
                    'translation': [float(tx), float(ty)],
                    'rotation_matrix': tform.params[:2, :2].tolist(),
                    'n_landmarks': n,
                    'pts_a': pts_a_use.tolist(),
                    'pts_b': pts_b_use.tolist(),
                }
                # Save after each pair
                with open(save_path, 'w') as f:
                    json.dump(saved, f, indent=2)
                print(f"  Saved! ({len(saved)}/{len(pairs)} pairs done)")
                break
            else:
                print("  Redoing landmarks...")
                continue

    # Final save
    with open(save_path, 'w') as f:
        json.dump(saved, f, indent=2)
    print(f"\nDone! {len(saved)}/{len(pairs)} pairs aligned.")
    print(f"Transforms saved to: {save_path}")

if __name__ == '__main__':
    main()
