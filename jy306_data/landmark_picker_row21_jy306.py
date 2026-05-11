"""
Interactive landmark picker: ex-vivo tile ↔ JY306 in-vivo.
Pre-loaded with 27 predicted pkl cell positions (adjustable).

LEFT panel:  Selectable ex-vivo tile GFP z-slices (4200x4200)
RIGHT panel: JY306 in-vivo stack (16 slices, 658x629)

Controls:
  LEFT-CLICK  — adjust current cell position on that panel
  SCROLL      — zoom in/out (on whichever panel the mouse is over)
  SHIFT+DRAG  — pan the zoomed view
  W / E       — prev / next cell (auto-jumps to predicted z)
  A / D       — prev / next left z-slice
  J / L       — prev / next right z-slice
  1 / 2       — prev / next left tile
  9 / 0       — prev / next right set
  C / V       — left contrast +/-
  N / M       — right contrast +/-
  X           — delete current cell
  R           — reset zoom on both panels
  S           — save landmarks and quit
  Q           — quit without saving
"""

import numpy as np
import cv2
import os
import glob
import tifffile

BASE = '/Users/neurolab/neuroinformatics/margaret'

# ============================================================
# Image sets
# ============================================================
tile_dirs = sorted(glob.glob(os.path.join(BASE, 'png_exports/registration_video/row*')))
left_sets = []
for d in tile_dirs:
    gfp_files = sorted(glob.glob(os.path.join(d, 'GFP_z*.png')))
    if gfp_files:
        left_sets.append({'name': os.path.basename(d), 'files': gfp_files, 'imgs': None})

right_sets = []
jy306_vol = tifffile.imread(os.path.join(BASE, 'JY306_in_Vivo_stack_flipped_s80.tif'))
p99 = np.percentile(jy306_vol[jy306_vol > 0], 99)
jy306_u8 = np.clip(jy306_vol / p99 * 255, 0, 255).astype(np.uint8)
right_sets.append({
    'name': 'JY306_invivo',
    'imgs': [jy306_u8[z] for z in range(jy306_u8.shape[0])],
    'shape': jy306_u8.shape[1:]
})

exvivo_path = os.path.join(BASE, 'jy306_registered files/exvivo_combined.tif')
if os.path.exists(exvivo_path):
    ev_vol = tifffile.imread(exvivo_path)
    ev_p99 = np.percentile(ev_vol[ev_vol > 0], 99) if np.any(ev_vol > 0) else 1
    ev_u8 = np.clip(ev_vol / ev_p99 * 255, 0, 255).astype(np.uint8)
    right_sets.append({
        'name': 'exvivo_combined',
        'imgs': [ev_u8[z] for z in range(ev_u8.shape[0])],
        'shape': ev_u8.shape[1:]
    })

print(f"Left sets ({len(left_sets)}):")
for i, s in enumerate(left_sets):
    print(f"  [{i}] {s['name']} ({len(s['files'])} z)")
print(f"Right sets ({len(right_sets)}):")
for i, s in enumerate(right_sets):
    print(f"  [{i}] {s['name']} ({len(s['imgs'])} z, {s['shape']})")

def get_left_imgs(idx):
    s = left_sets[idx]
    if s['imgs'] is None:
        print(f"  Loading {s['name']}...")
        s['imgs'] = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in s['files']]
        print(f"  Loaded {len(s['imgs'])} slices")
    return s['imgs']

left_idx = next((i for i, s in enumerate(left_sets) if s['name'] == 'row2_1'), 0)
right_idx = 0
get_left_imgs(left_idx)

# ============================================================
# Pre-load 27 predicted cells
# ============================================================
M_nd2_to_jy = np.load(os.path.join(BASE, 'registration_video/affine_nd2_to_exvivo.npy'))
M_3x3 = np.vstack([M_nd2_to_jy, [0, 0, 1]])
M_jy_to_nd2 = np.linalg.inv(M_3x3)[:2, :]

mc = np.load(os.path.join(BASE, 'registration_video/matched_cells_3d.npz'), allow_pickle=True)
iv_3d = mc['iv_3d']
ev_3d = mc['ev_3d']

cells = []
for i in range(len(iv_3d)):
    nd2_pos = M_jy_to_nd2 @ np.array([ev_3d[i, 0], ev_3d[i, 1], 1.0])
    est_nd2_z = max(0, min(11, int(round(ev_3d[i, 2] - 2))))
    jy_z = max(0, min(15, int(round(iv_3d[i, 2]))))
    cells.append({
        'src_col': float(nd2_pos[0]), 'src_row': float(nd2_pos[1]),
        'src_z': est_nd2_z, 'src_set': 'row2_1',
        'tgt_col': float(iv_3d[i, 0]), 'tgt_row': float(iv_3d[i, 1]),
        'tgt_z': jy_z, 'tgt_set': 'JY306_invivo',
        'src_adjusted': False, 'tgt_adjusted': False, 'deleted': False,
    })

print(f"\nPre-loaded {len(cells)} cells")

# ============================================================
# Zoom/pan state per panel
# ============================================================
PANEL_W = 800
PANEL_H = 900
GAP = 20

class PanelView:
    def __init__(self, img_w, img_h):
        self.img_w = img_w
        self.img_h = img_h
        self.zoom = 1.0       # 1.0 = fit whole image in panel
        self.cx = img_w / 2   # center of view in image coords
        self.cy = img_h / 2
        self.dragging = False
        self.drag_start = None

    def reset(self):
        self.zoom = 1.0
        self.cx = self.img_w / 2
        self.cy = self.img_h / 2

    def zoom_at(self, img_x, img_y, factor):
        """Zoom in/out centered on (img_x, img_y)."""
        new_zoom = max(1.0, min(20.0, self.zoom * factor))
        # Adjust center so the point under cursor stays fixed
        self.cx = img_x + (self.cx - img_x) * (self.zoom / new_zoom)
        self.cy = img_y + (self.cy - img_y) * (self.zoom / new_zoom)
        self.zoom = new_zoom
        self._clamp()

    def pan(self, dx_panel, dy_panel):
        """Pan by pixel delta in panel coords."""
        scale = self.get_scale()
        self.cx -= dx_panel / scale
        self.cy -= dy_panel / scale
        self._clamp()

    def get_scale(self):
        """Pixels per image pixel at current zoom."""
        base_scale = min(PANEL_W / self.img_w, PANEL_H / self.img_h)
        return base_scale * self.zoom

    def _clamp(self):
        scale = self.get_scale()
        half_w = PANEL_W / (2 * scale)
        half_h = PANEL_H / (2 * scale)
        self.cx = np.clip(self.cx, half_w, self.img_w - half_w)
        self.cy = np.clip(self.cy, half_h, self.img_h - half_h)

    def panel_to_img(self, px, py):
        """Convert panel pixel (px, py) to image coords."""
        scale = self.get_scale()
        img_x = self.cx + (px - PANEL_W / 2) / scale
        img_y = self.cy + (py - PANEL_H / 2) / scale
        return img_x, img_y

    def img_to_panel(self, img_x, img_y):
        """Convert image coords to panel pixel."""
        scale = self.get_scale()
        px = (img_x - self.cx) * scale + PANEL_W / 2
        py = (img_y - self.cy) * scale + PANEL_H / 2
        return int(px), int(py)

    def render(self, img_gray, contrast=1.0):
        """Render the visible portion of img into a PANEL_W x PANEL_H BGR image."""
        scale = self.get_scale()
        half_w = PANEL_W / (2 * scale)
        half_h = PANEL_H / (2 * scale)

        # Source rect in image coords
        x0 = self.cx - half_w
        y0 = self.cy - half_h
        x1 = self.cx + half_w
        y1 = self.cy + half_h

        # Clip to image bounds
        sx0 = max(0, int(x0))
        sy0 = max(0, int(y0))
        sx1 = min(self.img_w, int(np.ceil(x1)))
        sy1 = min(self.img_h, int(np.ceil(y1)))

        crop = img_gray[sy0:sy1, sx0:sx1]
        if contrast != 1.0:
            crop = np.clip(crop.astype(np.float32) * contrast, 0, 255).astype(np.uint8)

        # Destination rect in panel coords
        dx0 = int((sx0 - x0) * scale)
        dy0 = int((sy0 - y0) * scale)
        dw = int((sx1 - sx0) * scale)
        dh = int((sy1 - sy0) * scale)

        panel = np.zeros((PANEL_H, PANEL_W), dtype=np.uint8)
        if dw > 0 and dh > 0 and crop.size > 0:
            resized = cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_LINEAR)
            # Clip to panel bounds
            pw = min(dw, PANEL_W - dx0)
            ph = min(dh, PANEL_H - dy0)
            if pw > 0 and ph > 0 and dx0 >= 0 and dy0 >= 0:
                panel[dy0:dy0+ph, dx0:dx0+pw] = resized[:ph, :pw]

        return cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)

# Init panel views
limgs = get_left_imgs(left_idx)
lh, lw = limgs[0].shape[:2]
left_view = PanelView(lw, lh)

rh, rw = right_sets[right_idx]['shape']
right_view = PanelView(rw, rh)

# ============================================================
# State
# ============================================================
src_z_idx = 0
tgt_z_idx = 0
left_contrast = 5.0
right_contrast = 1.0
current_cell = 0

COLORS = [
    (0, 255, 255), (0, 255, 0), (255, 0, 0), (0, 200, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
    (255, 128, 0), (128, 255, 0), (0, 255, 128), (255, 0, 128),
    (100, 100, 255), (100, 255, 100), (255, 100, 100), (200, 200, 0),
    (255, 200, 100), (100, 200, 255), (200, 100, 255), (255, 100, 200),
    (150, 255, 150), (150, 150, 255), (255, 150, 150), (200, 255, 100),
    (0, 255, 255), (0, 255, 0), (255, 0, 0),
]


def build_display():
    limgs = get_left_imgs(left_idx)
    left = left_view.render(limgs[src_z_idx], left_contrast)

    rimgs = right_sets[right_idx]['imgs']
    right = right_view.render(rimgs[tgt_z_idx], right_contrast)

    cur_lname = left_sets[left_idx]['name']
    cur_rname = right_sets[right_idx]['name']

    for i, c in enumerate(cells):
        if c['deleted']:
            continue
        col = COLORS[i % len(COLORS)]
        is_current = (i == current_cell)

        # Left panel
        if c['src_set'] == cur_lname:
            px, py = left_view.img_to_panel(c['src_col'], c['src_row'])
            if 0 <= px < PANEL_W and 0 <= py < PANEL_H:
                if c['src_z'] == src_z_idx:
                    r = 12 if is_current else 7
                    t = 2 if is_current else 1
                    cv2.circle(left, (px, py), r, col, t, cv2.LINE_AA)
                    if not c['src_adjusted']:
                        cv2.circle(left, (px, py), r + 4, (100, 100, 100), 1, cv2.LINE_AA)
                    cv2.putText(left, str(i+1), (px+r+4, py+4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
                else:
                    dim = tuple(int(v * 0.3) for v in col)
                    cv2.circle(left, (px, py), 3, dim, 1, cv2.LINE_AA)

        # Right panel
        if c['tgt_set'] == cur_rname:
            px, py = right_view.img_to_panel(c['tgt_col'], c['tgt_row'])
            if 0 <= px < PANEL_W and 0 <= py < PANEL_H:
                if c['tgt_z'] == tgt_z_idx:
                    r = 12 if is_current else 7
                    t = 2 if is_current else 1
                    cv2.circle(right, (px, py), r, col, t, cv2.LINE_AA)
                    if not c['tgt_adjusted']:
                        cv2.circle(right, (px, py), r + 4, (100, 100, 100), 1, cv2.LINE_AA)
                    cv2.putText(right, str(i+1), (px+r+4, py+4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
                else:
                    dim = tuple(int(v * 0.3) for v in col)
                    cv2.circle(right, (px, py), 3, dim, 1, cv2.LINE_AA)

    # Labels
    lname = left_sets[left_idx]['name']
    rname = right_sets[right_idx]['name']
    zl = f"z{src_z_idx:02d}"
    zr = f"z{tgt_z_idx:02d}"
    cv2.putText(left, f"{lname} {zl} x{left_contrast:.0f} zoom={left_view.zoom:.1f}x",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(right, f"{rname} {zr} x{right_contrast:.0f} zoom={right_view.zoom:.1f}x",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    gap = np.zeros((PANEL_H, GAP, 3), dtype=np.uint8)
    combined = np.hstack([left, gap, right])

    # Status bar
    alive = sum(1 for c2 in cells if not c2['deleted'])
    adj_both = sum(1 for c2 in cells if c2['src_adjusted'] and c2['tgt_adjusted'] and not c2['deleted'])
    bar = np.zeros((45, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, f"Cell {current_cell+1}/{alive} | Verified: {adj_both} | "
                f"W/E=prev/next | scroll=zoom | shift+drag=pan | R=reset zoom",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, f"click=adjust | X=delete | S=save | Q=quit | gray ring=unadjusted",
                (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)

    return np.vstack([combined, bar])


# Undo history: list of (cell_idx, side, old_col, old_row, old_z, old_adjusted)
undo_stack = []

# Mouse state
mouse_x, mouse_y = 0, 0
dragging = False
drag_start = None
drag_panel = None  # 'left' or 'right'


def get_panel(x):
    if x < PANEL_W:
        return 'left', x
    elif x > PANEL_W + GAP:
        return 'right', x - PANEL_W - GAP
    return None, 0


def on_mouse(event, x, y, flags, param):
    global mouse_x, mouse_y, dragging, drag_start, drag_panel, current_cell

    mouse_x, mouse_y = x, y

    panel, px = get_panel(x)

    if event == cv2.EVENT_MOUSEWHEEL:
        if panel == 'left':
            img_x, img_y = left_view.panel_to_img(px, y)
            factor = 1.3 if flags > 0 else 1/1.3
            left_view.zoom_at(img_x, img_y, factor)
        elif panel == 'right':
            img_x, img_y = right_view.panel_to_img(px, y)
            factor = 1.3 if flags > 0 else 1/1.3
            right_view.zoom_at(img_x, img_y, factor)

    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            # Shift+click = start pan
            dragging = True
            drag_start = (x, y)
            drag_panel = panel
        else:
            # Normal click = adjust cell position
            c = cells[current_cell]
            if panel == 'left':
                # Save undo state
                undo_stack.append((current_cell, 'src', c['src_col'], c['src_row'], c['src_z'], c['src_adjusted']))
                img_x, img_y = left_view.panel_to_img(px, y)
                c['src_col'] = img_x
                c['src_row'] = img_y
                c['src_z'] = src_z_idx
                c['src_set'] = left_sets[left_idx]['name']
                c['src_adjusted'] = True
                print(f"  Cell {current_cell+1} LEFT: ({img_x:.1f}, {img_y:.1f}) z={src_z_idx}")
            elif panel == 'right':
                # Save undo state
                undo_stack.append((current_cell, 'tgt', c['tgt_col'], c['tgt_row'], c['tgt_z'], c['tgt_adjusted']))
                img_x, img_y = right_view.panel_to_img(px, y)
                c['tgt_col'] = img_x
                c['tgt_row'] = img_y
                c['tgt_z'] = tgt_z_idx
                c['tgt_set'] = right_sets[right_idx]['name']
                c['tgt_adjusted'] = True
                print(f"  Cell {current_cell+1} RIGHT: ({img_x:.1f}, {img_y:.1f}) z={tgt_z_idx}")

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging and drag_start is not None:
            dx = x - drag_start[0]
            dy = y - drag_start[1]
            if drag_panel == 'left':
                left_view.pan(dx, dy)
            elif drag_panel == 'right':
                right_view.pan(dx, dy)
            drag_start = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        drag_start = None
        drag_panel = None


def jump_to_cell(idx):
    global src_z_idx, tgt_z_idx, current_cell
    current_cell = idx
    c = cells[idx]
    src_z_idx = c['src_z']
    tgt_z_idx = c['tgt_z']
    # Center views on cell
    left_view.cx = c['src_col']
    left_view.cy = c['src_row']
    right_view.cx = c['tgt_col']
    right_view.cy = c['tgt_row']
    status = "verified" if c['src_adjusted'] and c['tgt_adjusted'] else "predicted"
    print(f"  -> Cell {idx+1} ({status}): L z={src_z_idx} R z={tgt_z_idx}")


# ============================================================
# Main loop
# ============================================================
WIN = "Landmark Picker — 27 cells (scroll=zoom, shift+drag=pan)"
cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(WIN, on_mouse)

print(f"\n=== LANDMARK PICKER (27 pre-loaded, zoomable) ===")
print(f"SCROLL = zoom, SHIFT+DRAG = pan, R = reset zoom")
print(f"W/E = prev/next cell, click = adjust, X = delete")
print(f"A/D = left z, J/L = right z, C/V N/M = contrast")
print(f"S = save, Q = quit\n")

jump_to_cell(0)

while True:
    disp = build_display()
    cv2.imshow(WIN, disp)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('a') or key == ord('A'):
        src_z_idx = max(0, src_z_idx - 1)
        print(f"  Left z: {src_z_idx}")
    elif key == ord('d') or key == ord('D'):
        limgs = get_left_imgs(left_idx)
        src_z_idx = min(len(limgs) - 1, src_z_idx + 1)
        print(f"  Left z: {src_z_idx}")
    elif key == ord('j') or key == ord('J'):
        tgt_z_idx = max(0, tgt_z_idx - 1)
        print(f"  Right z: {tgt_z_idx}")
    elif key == ord('l') or key == ord('L'):
        tgt_z_idx = min(len(right_sets[right_idx]['imgs']) - 1, tgt_z_idx + 1)
        print(f"  Right z: {tgt_z_idx}")

    elif key == ord('w') or key == ord('W'):
        idx = current_cell - 1
        while idx >= 0 and cells[idx]['deleted']:
            idx -= 1
        if idx >= 0:
            jump_to_cell(idx)
    elif key == ord('e') or key == ord('E'):
        idx = current_cell + 1
        while idx < len(cells) and cells[idx]['deleted']:
            idx += 1
        if idx < len(cells):
            jump_to_cell(idx)

    elif key == ord('[') or key == ord('-'):
        # Zoom out on panel under mouse
        panel, px = get_panel(mouse_x)
        if panel == 'left':
            img_x, img_y = left_view.panel_to_img(PANEL_W//2, PANEL_H//2)
            left_view.zoom_at(img_x, img_y, 1/1.4)
            print(f"  Left zoom: {left_view.zoom:.1f}x")
        elif panel == 'right':
            img_x, img_y = right_view.panel_to_img(PANEL_W//2, PANEL_H//2)
            right_view.zoom_at(img_x, img_y, 1/1.4)
            print(f"  Right zoom: {right_view.zoom:.1f}x")
    elif key == ord(']') or key == ord('='):
        # Zoom in on panel under mouse
        panel, px = get_panel(mouse_x)
        if panel == 'left':
            img_x, img_y = left_view.panel_to_img(PANEL_W//2, PANEL_H//2)
            left_view.zoom_at(img_x, img_y, 1.4)
            print(f"  Left zoom: {left_view.zoom:.1f}x")
        elif panel == 'right':
            img_x, img_y = right_view.panel_to_img(PANEL_W//2, PANEL_H//2)
            right_view.zoom_at(img_x, img_y, 1.4)
            print(f"  Right zoom: {right_view.zoom:.1f}x")

    elif key == ord('r') or key == ord('R'):
        left_view.reset()
        right_view.reset()
        print("  Zoom reset")

    elif key == ord('z') or key == ord('Z'):
        if undo_stack:
            ci, side, old_col, old_row, old_z, old_adj = undo_stack.pop()
            c = cells[ci]
            if side == 'src':
                c['src_col'] = old_col
                c['src_row'] = old_row
                c['src_z'] = old_z
                c['src_adjusted'] = old_adj
                print(f"  Undo cell {ci+1} LEFT → ({old_col:.1f}, {old_row:.1f}, z{old_z})")
            else:
                c['tgt_col'] = old_col
                c['tgt_row'] = old_row
                c['tgt_z'] = old_z
                c['tgt_adjusted'] = old_adj
                print(f"  Undo cell {ci+1} RIGHT → ({old_col:.1f}, {old_row:.1f}, z{old_z})")
        else:
            print("  Nothing to undo")

    elif key == ord('x') or key == ord('X'):
        cells[current_cell]['deleted'] = True
        print(f"  Deleted cell {current_cell + 1}")
        idx = current_cell + 1
        while idx < len(cells) and cells[idx]['deleted']:
            idx += 1
        if idx < len(cells):
            jump_to_cell(idx)

    elif key == ord('1'):
        left_idx = max(0, left_idx - 1)
        limgs = get_left_imgs(left_idx)
        lh, lw = limgs[0].shape[:2]
        left_view = PanelView(lw, lh)
        src_z_idx = min(src_z_idx, len(limgs) - 1)
        print(f"  Left set: {left_sets[left_idx]['name']}")
    elif key == ord('2'):
        left_idx = min(len(left_sets) - 1, left_idx + 1)
        limgs = get_left_imgs(left_idx)
        lh, lw = limgs[0].shape[:2]
        left_view = PanelView(lw, lh)
        src_z_idx = min(src_z_idx, len(limgs) - 1)
        print(f"  Left set: {left_sets[left_idx]['name']}")

    elif key == ord('9'):
        right_idx = max(0, right_idx - 1)
        rh, rw = right_sets[right_idx]['shape']
        right_view = PanelView(rw, rh)
        tgt_z_idx = min(tgt_z_idx, len(right_sets[right_idx]['imgs']) - 1)
        print(f"  Right set: {right_sets[right_idx]['name']}")
    elif key == ord('0'):
        right_idx = min(len(right_sets) - 1, right_idx + 1)
        rh, rw = right_sets[right_idx]['shape']
        right_view = PanelView(rw, rh)
        tgt_z_idx = min(tgt_z_idx, len(right_sets[right_idx]['imgs']) - 1)
        print(f"  Right set: {right_sets[right_idx]['name']}")

    elif key == ord('c') or key == ord('C'):
        left_contrast = min(20.0, left_contrast + 1.0)
        print(f"  Left contrast: {left_contrast:.0f}x")
    elif key == ord('v') or key == ord('V'):
        left_contrast = max(1.0, left_contrast - 1.0)
        print(f"  Left contrast: {left_contrast:.0f}x")
    elif key == ord('n') or key == ord('N'):
        right_contrast = min(20.0, right_contrast + 1.0)
        print(f"  Right contrast: {right_contrast:.0f}x")
    elif key == ord('m') or key == ord('M'):
        right_contrast = max(1.0, right_contrast - 1.0)
        print(f"  Right contrast: {right_contrast:.0f}x")

    elif key == ord('s') or key == ord('S'):
        active = [c for c in cells if not c['deleted']]
        if len(active) < 1:
            print("No cells to save")
            continue

        src_pts = np.array([[c['src_col'], c['src_row'], c['src_z']] for c in active])
        tgt_pts = np.array([[c['tgt_col'], c['tgt_row'], c['tgt_z']] for c in active])
        src_names = [c['src_set'] for c in active]
        tgt_names = [c['tgt_set'] for c in active]
        src_adj = [c['src_adjusted'] for c in active]
        tgt_adj = [c['tgt_adjusted'] for c in active]

        out_path = os.path.join(BASE, 'registration_video/landmarks_row21_jy306.npz')
        np.savez(out_path,
                 src_points=src_pts, tgt_points=tgt_pts,
                 src_set_names=src_names, tgt_set_names=tgt_names,
                 src_adjusted=src_adj, tgt_adjusted=tgt_adj)

        n_both = sum(1 for c in active if c['src_adjusted'] and c['tgt_adjusted'])
        print(f"\nSaved {len(active)} cells ({n_both} fully verified) to {out_path}")
        for i, c in enumerate(active):
            s = "OK" if c['src_adjusted'] and c['tgt_adjusted'] else "pred"
            print(f"  {i+1}: ({c['src_col']:.1f},{c['src_row']:.1f},z{c['src_z']}) "
                  f"<-> ({c['tgt_col']:.1f},{c['tgt_row']:.1f},z{c['tgt_z']}) [{s}]")
        break

    elif key == ord('q') or key == ord('Q'):
        print("Quit without saving")
        break

cv2.destroyAllWindows()
print("Done!")
