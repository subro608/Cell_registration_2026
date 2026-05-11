"""
Cell zoom v2: Start from grid view with MERSCOPE dots, zoom into a tile,
zoom further into a cell, show 3-panel split (IV | EV | MERSCOPE).
Does 2 cells from different tiles.

Uses existing rendered grid frame as starting point, then progressively
crops and scales to zoom in. Transitions to 3-panel view at cell level.

Outputs to frames_cell_zoom_v2/
"""
import numpy as np, cv2, math, os, pickle, shutil

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_DIR = f'{BASE}/animation/frames_cell_zoom_v2'
FRAMES_DIR = f'{BASE}/animation/frames_multi_tile_3d'
W, H = 1920, 1080
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
MARGIN_ND2 = 350
CELL_H = 400

def ease(t):
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))

# ── Load assets ──
print("Loading assets...")
with open(f'{BASE}/animation/scene5b_assets_v3.pkl', 'rb') as f:
    assets = pickle.load(f)

# Grid frame with MERSCOPE dots visible (from fly-by phase)
grid_frame = cv2.imread(f'{FRAMES_DIR}/frame_00170.png')
print(f"Grid frame: {grid_frame.shape}")

# ── Grid positions (same computation as main script) ──
TILES = ['row1_3','row2_1','row2_2','row2_3','row2_4','row2_5',
         'row3_1','row3_2','row3_3','row3_4','row3_5','row3_6',
         'row4_1','row4_2','row4_3','row4_4','row4_5','row4_6','row5_1']
tile_list = [t for t in TILES if t in assets]
GRID_COLS = 5
avg_cw = sum(assets[t]['cell_w'] for t in tile_list) // len(tile_list)
avg_ch = sum(assets[t]['cell_h'] for t in tile_list) // len(tile_list)
GAP_X, GAP_Y = 20, 15
grid_w = GRID_COLS * avg_cw + 4 * GAP_X
grid_h = 4 * avg_ch + 3 * GAP_Y
margin = 80
fit_scale = min((W - margin) / grid_w, (H - margin) / grid_h)
fitted_w = grid_w * fit_scale
fitted_h = grid_h * fit_scale
gx0 = (W - fitted_w) / 2
gy0 = (H - fitted_h) / 2

grid_positions = {}
for idx, tn in enumerate(tile_list):
    row = idx // GRID_COLS
    col = idx % GRID_COLS
    cx = gx0 + (col * (avg_cw + GAP_X) + avg_cw / 2) * fit_scale
    cy = gy0 + (row * (avg_ch + GAP_Y) + avg_ch / 2) * fit_scale
    grid_positions[tn] = (int(cx), int(cy))

# ── Select 2 cells from different tiles ──
SELECTED = ['row3_3', 'row4_3']  # good central tiles with many landmarks

cell_info = []
for tile in SELECTED:
    a = assets[tile]
    dense = a['dense']
    ms = a.get('merscope')
    cell_w = a['cell_w']

    pkl_path = f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz'
    pkl = np.load(pkl_path)
    ev = pkl['ev_nd2']
    iv = pkl['pcd_invivo_jy306']

    # Crop bounds
    crop_x0 = max(0, int(ev[:, 0].min() - MARGIN_ND2))
    crop_y0 = max(0, int(ev[:, 1].min() - MARGIN_ND2))
    crop_x1 = min(4200, int(ev[:, 0].max() + MARGIN_ND2))
    crop_y1 = min(4200, int(ev[:, 1].max() + MARGIN_ND2))
    crop_h_nd2 = crop_y1 - crop_y0
    scale_nd2 = CELL_H / crop_h_nd2

    # Pick a landmark near center
    ev_cx = (ev[:, 0].min() + ev[:, 0].max()) / 2
    ev_cy = (ev[:, 1].min() + ev[:, 1].max()) / 2
    dists = np.sqrt((ev[:, 0] - ev_cx)**2 + (ev[:, 1] - ev_cy)**2)
    sorted_idx = np.argsort(dists)
    pick = sorted_idx[min(3, len(sorted_idx) - 1)]

    # Landmark in display coordinates
    lm_x_disp = (ev[pick, 0] - crop_x0) * scale_nd2
    lm_y_disp = (ev[pick, 1] - crop_y0) * scale_nd2
    lm_z = ev[pick, 2]
    dense_z = a['dense_z']
    best_zi = int(np.argmin(np.abs(dense_z - lm_z)))

    # Patch bounds in display coords
    PATCH_R = 80
    px, py = int(lm_x_disp), int(lm_y_disp)
    x0 = max(0, px - PATCH_R)
    y0 = max(0, py - PATCH_R)
    x1 = min(cell_w, px + PATCH_R)
    y1 = min(CELL_H, py + PATCH_R)

    sl = dense[best_zi]  # (h, w, 3)

    # Extract channel patches
    patch_merged = sl[y0:y1, x0:x1].copy()
    patch_iv = np.zeros_like(patch_merged)
    patch_iv[:, :, 1] = patch_merged[:, :, 1]
    patch_ev = np.zeros_like(patch_merged)
    patch_ev[:, :, 0] = patch_merged[:, :, 0]
    patch_ev[:, :, 2] = patch_merged[:, :, 2]
    patch_ms = ms[y0:y1, x0:x1].copy() if ms is not None else np.zeros_like(patch_merged)

    # Grid position of this tile
    gcx, gcy = grid_positions[tile]

    # Tile size in grid (display pixels)
    tile_half_w = int(cell_w * fit_scale / 2)
    tile_half_h = int(CELL_H * fit_scale / 2)

    # Cell position within grid frame (for zoom target)
    # lm position relative to tile center, scaled to grid
    cell_gx = gcx + (lm_x_disp - cell_w / 2) * fit_scale
    cell_gy = gcy + (lm_y_disp - CELL_H / 2) * fit_scale

    cell_info.append({
        'tile': tile,
        'gcx': gcx, 'gcy': gcy,
        'cell_gx': cell_gx, 'cell_gy': cell_gy,
        'tile_half_w': tile_half_w, 'tile_half_h': tile_half_h,
        'patch_iv': patch_iv,
        'patch_ev': patch_ev,
        'patch_ms': patch_ms,
        'patch_merged': patch_merged,
        'lm_px': px - x0, 'lm_py': py - y0,
        'patch_w': x1 - x0, 'patch_h': y1 - y0,
        'full_slice': sl,
        'cell_w': cell_w,
    })
    print(f"  {tile}: landmark at grid ({cell_gx:.0f},{cell_gy:.0f}), patch {x1-x0}x{y1-y0}")

print(f"\n{len(cell_info)} cells selected")


def zoom_crop(img, cx, cy, zoom, out_w=W, out_h=H):
    """Crop and scale image to simulate zoom centered at (cx, cy)."""
    h, w = img.shape[:2]
    crop_w = out_w / zoom
    crop_h = out_h / zoom
    x0 = max(0, min(w - crop_w, cx - crop_w / 2))
    y0 = max(0, min(h - crop_h, cy - crop_h / 2))
    x1 = x0 + crop_w
    y1 = y0 + crop_h
    cropped = img[int(y0):int(y1), int(x0):int(x1)]
    return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)


def draw_3panel(cell, panel_size=320, gap=30):
    """Draw 3-panel view: IV | EV | MERSCOPE with crosshairs."""
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    panels = [
        ('IN-VIVO', cell['patch_iv']),
        ('EX-VIVO', cell['patch_ev']),
        ('MERSCOPE', cell['patch_ms']),
    ]

    total_w = 3 * panel_size + 2 * gap
    x_start = (W - total_w) // 2
    y_start = (H - panel_size) // 2 - 20

    for i, (label, patch) in enumerate(panels):
        px = x_start + i * (panel_size + gap)
        py = y_start

        resized = cv2.resize(patch, (panel_size, panel_size), interpolation=cv2.INTER_LANCZOS4)
        # Brightness boost
        resized = np.clip(resized.astype(np.float32) * 2.0, 0, 255).astype(np.uint8)
        canvas[py:py + panel_size, px:px + panel_size] = resized

        # Border
        cv2.rectangle(canvas, (px, py), (px + panel_size - 1, py + panel_size - 1),
                       (100, 100, 100), 1)

        # Crosshair
        lm_x = int(cell['lm_px'] * panel_size / cell['patch_w'])
        lm_y = int(cell['lm_py'] * panel_size / cell['patch_h'])
        cr = 15
        color = (0, 255, 255)
        cv2.circle(canvas, (px + lm_x, py + lm_y), cr, color, 1, cv2.LINE_AA)
        cv2.line(canvas, (px + lm_x - cr - 5, py + lm_y), (px + lm_x - 5, py + lm_y), color, 1)
        cv2.line(canvas, (px + lm_x + 5, py + lm_y), (px + lm_x + cr + 5, py + lm_y), color, 1)
        cv2.line(canvas, (px + lm_x, py + lm_y - cr - 5), (px + lm_x, py + lm_y - 5), color, 1)
        cv2.line(canvas, (px + lm_x, py + lm_y + 5), (px + lm_x, py + lm_y + cr + 5), color, 1)

        # Label
        ts = 0.55
        (tw, _), _ = cv2.getTextSize(label, FONT, ts, 1)
        cv2.putText(canvas, label, (px + (panel_size - tw) // 2, py + panel_size + 25),
                    FONT, ts, WHITE, 1, cv2.LINE_AA)

    # Tile name
    tile_label = cell['tile'].upper().replace('_', ' ')
    ts = 0.6
    (tw, _), _ = cv2.getTextSize(tile_label, FONT, ts, 1)
    cv2.putText(canvas, tile_label, ((W - tw) // 2, y_start - 15),
                FONT, ts, WHITE, 1, cv2.LINE_AA)

    cap = 'SINGLE-CELL  MULTIMODAL  REGISTRATION'
    ts = 0.72
    (tw, _), _ = cv2.getTextSize(cap, FONT, ts, 1)
    cv2.putText(canvas, cap, ((W - tw) // 2, H - 42), FONT, ts, WHITE, 1, cv2.LINE_AA)

    return canvas


# ── Render frames ──
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)

fi = 0
print("\nRendering...")

for ci, cell in enumerate(cell_info):
    gcx, gcy = cell['gcx'], cell['gcy']
    cell_gx, cell_gy = cell['cell_gx'], cell['cell_gy']
    panel_img = draw_3panel(cell)

    # Determine start frame — first cell starts from grid, second from previous panel
    if ci == 0:
        start_img = grid_frame.copy()
    else:
        start_img = prev_panel  # from previous cell's panel

    # Phase 1: Zoom from grid into the tile (30 frames)
    ZOOM_TILE = 30
    for f in range(ZOOM_TILE):
        t = ease(f / max(1, ZOOM_TILE - 1))
        zoom = 1.0 + (4.0 - 1.0) * t  # zoom 1x → 4x
        # Pan toward the tile center
        target_x = gcx * (1 - t) + cell_gx * t
        target_y = gcy * (1 - t) + cell_gy * t
        # For first cell start from center, for second from wherever
        if ci == 0:
            cx = W / 2 * (1 - t) + target_x * t
            cy = H / 2 * (1 - t) + target_y * t
        else:
            cx = target_x
            cy = target_y
        frame = zoom_crop(start_img if ci == 0 else grid_frame, cx, cy, zoom)

        # Add caption
        cap = f'ZOOMING INTO {cell["tile"].upper().replace("_"," ")}'
        ts = 0.72
        (tw, _), _ = cv2.getTextSize(cap, FONT, ts, 1)
        cv2.putText(frame, cap, ((W - tw) // 2, H - 42), FONT, ts, WHITE, 1, cv2.LINE_AA)

        fi += 1
        cv2.imwrite(f'{OUT_DIR}/frame_{fi:05d}.png', frame)

    # Phase 2: Zoom further into cell + transition to 3-panel (24 frames)
    # Last zoomed frame
    last_zoom = zoom_crop(grid_frame, cell_gx, cell_gy, 4.0)
    TRANS = 24
    for f in range(TRANS):
        t = ease(f / max(1, TRANS - 1))
        # Continue zooming
        zoom = 4.0 + (8.0 - 4.0) * t
        zoomed = zoom_crop(grid_frame, cell_gx, cell_gy, zoom)
        # Blend with panel view
        frame = cv2.addWeighted(zoomed, 1 - t, panel_img, t, 0)
        fi += 1
        cv2.imwrite(f'{OUT_DIR}/frame_{fi:05d}.png', frame)

    # Phase 3: Hold 3-panel (48 frames)
    HOLD = 48
    for f in range(HOLD):
        fi += 1
        cv2.imwrite(f'{OUT_DIR}/frame_{fi:05d}.png', panel_img)

    prev_panel = panel_img

    # Phase 4: Transition to next cell or zoom out (18 frames)
    if ci < len(cell_info) - 1:
        # Zoom out back to grid
        ZOOM_OUT = 18
        for f in range(ZOOM_OUT):
            t = ease(f / max(1, ZOOM_OUT - 1))
            frame = cv2.addWeighted(panel_img, 1 - t, grid_frame, t, 0)
            fi += 1
            cv2.imwrite(f'{OUT_DIR}/frame_{fi:05d}.png', frame)
    else:
        # Last cell: fade out
        FADE = 18
        for f in range(FADE):
            t = ease(f / max(1, FADE - 1))
            frame = (panel_img.astype(np.float32) * (1 - t)).astype(np.uint8)
            fi += 1
            cv2.imwrite(f'{OUT_DIR}/frame_{fi:05d}.png', frame)

    print(f"  Cell {ci+1}/{len(cell_info)}: {cell['tile']} done")

print(f"\nDone! {fi} frames saved to {OUT_DIR}/")

# Encode
print("Encoding preview...")
os.system(f'ffmpeg -y -framerate 24 -i {OUT_DIR}/frame_%05d.png '
          f'-c:v libx264 -pix_fmt yuv420p -crf 18 '
          f'{BASE}/animation/cell_zoom_v2_preview.mp4 2>/dev/null')
print("Saved cell_zoom_v2_preview.mp4")
