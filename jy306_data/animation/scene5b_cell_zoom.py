"""
Cell-level zoom sequence: zoom into individual matched cells across tiles.
For each cell, show 4 panels: in-vivo | ex-vivo | gene dots | merged overlay.
Picks 5 cells from different tiles with good landmark matches.

Outputs to frames_cell_zoom/ folder.
"""
import numpy as np, cv2, math, os, pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_DIR = f'{BASE}/animation/frames_cell_zoom'
W, H = 1920, 1080
FPS = 24
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

# ── Select tiles and landmarks ──
SELECTED_TILES = ['row2_1', 'row2_3', 'row3_3', 'row4_3', 'row5_1']
cell_cards = []

for tile in SELECTED_TILES:
    if tile not in assets:
        continue
    a = assets[tile]
    dense = a['dense']  # (n, h, w, 3)
    ms = a.get('merscope')  # (h, w, 3) or None

    # Load landmarks
    pkl_path = f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz'
    if not os.path.exists(pkl_path):
        continue
    pkl = np.load(pkl_path)
    ev = pkl['ev_nd2']  # (n_lm, 3) — x, y, z in nd2 space
    iv = pkl['pcd_invivo_jy306']  # (n_lm, 3) — z, y, x in jy306

    # Reconstruct crop bounds (same as scene5b_save_assets.py)
    crop_x0 = max(0, int(ev[:, 0].min() - MARGIN_ND2))
    crop_y0 = max(0, int(ev[:, 1].min() - MARGIN_ND2))
    crop_x1 = min(4200, int(ev[:, 0].max() + MARGIN_ND2))
    crop_y1 = min(4200, int(ev[:, 1].max() + MARGIN_ND2))
    crop_h_nd2 = crop_y1 - crop_y0
    crop_w_nd2 = crop_x1 - crop_x0
    scale_nd2 = CELL_H / crop_h_nd2
    cell_w = a['cell_w']

    # Pick a landmark near the center of the tile (most visible)
    ev_cx = (ev[:, 0].min() + ev[:, 0].max()) / 2
    ev_cy = (ev[:, 1].min() + ev[:, 1].max()) / 2
    dists = np.sqrt((ev[:, 0] - ev_cx)**2 + (ev[:, 1] - ev_cy)**2)
    # Pick one close to center but not the closest (more interesting)
    sorted_idx = np.argsort(dists)
    pick = sorted_idx[min(2, len(sorted_idx) - 1)]

    # Landmark position in dense slice coordinates (display pixels)
    lm_x_nd2 = ev[pick, 0]
    lm_y_nd2 = ev[pick, 1]
    lm_x_disp = (lm_x_nd2 - crop_x0) * scale_nd2
    lm_y_disp = (lm_y_nd2 - crop_y0) * scale_nd2

    # Pick the dense slice closest to this landmark's z
    lm_z_nd2 = ev[pick, 2]
    dense_z = a['dense_z']
    best_zi = int(np.argmin(np.abs(dense_z - lm_z_nd2)))

    # Extract patch (200x200 display pixels around the landmark)
    PATCH_R = 100  # radius in display pixels
    px = int(lm_x_disp)
    py = int(lm_y_disp)
    x0 = max(0, px - PATCH_R)
    y0 = max(0, py - PATCH_R)
    x1 = min(cell_w, px + PATCH_R)
    y1 = min(CELL_H, py + PATCH_R)

    sl = dense[best_zi]  # (h, w, 3) BGR

    # Extract patches for each channel
    patch_merged = sl[y0:y1, x0:x1].copy()

    # In-vivo (green channel)
    patch_iv = np.zeros_like(patch_merged)
    patch_iv[:, :, 1] = patch_merged[:, :, 1]

    # Ex-vivo (magenta = R+B)
    patch_ev = np.zeros_like(patch_merged)
    patch_ev[:, :, 0] = patch_merged[:, :, 0]
    patch_ev[:, :, 2] = patch_merged[:, :, 2]

    # MERSCOPE dots
    if ms is not None:
        patch_ms = ms[y0:y1, x0:x1].copy()
    else:
        patch_ms = np.zeros_like(patch_merged)

    # Landmark position within patch
    lm_px = px - x0
    lm_py = py - y0

    cell_cards.append({
        'tile': tile,
        'patch_iv': patch_iv,
        'patch_ev': patch_ev,
        'patch_ms': patch_ms,
        'patch_merged': patch_merged,
        'lm_px': lm_px,
        'lm_py': lm_py,
        'dense_slice': sl,
        'cell_w': cell_w,
        'patch_x0': x0, 'patch_y0': y0,
        'patch_size': (y1 - y0, x1 - x0),
    })
    print(f"  {tile}: landmark at disp ({px},{py}), patch {x1-x0}x{y1-y0}, z-slice {best_zi}")

print(f"\n{len(cell_cards)} cell cards prepared")


def draw_cell_card(card, panel_size=350, gap=20):
    """Draw a 4-panel cell card: IV | EV | MERSCOPE | Merged."""
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    panels = [
        ('IN-VIVO', card['patch_iv']),
        ('EX-VIVO', card['patch_ev']),
        ('MERSCOPE', card['patch_ms']),
        ('MERGED', card['patch_merged']),
    ]

    total_w = 4 * panel_size + 3 * gap
    x_start = (W - total_w) // 2
    y_start = (H - panel_size) // 2 - 30

    for i, (label, patch) in enumerate(panels):
        px = x_start + i * (panel_size + gap)
        py = y_start

        # Resize patch to panel_size
        resized = cv2.resize(patch, (panel_size, panel_size), interpolation=cv2.INTER_LANCZOS4)

        # Brightness boost for visibility
        resized_f = resized.astype(np.float32)
        resized_f = np.clip(resized_f * 1.5, 0, 255)
        resized = resized_f.astype(np.uint8)

        canvas[py:py + panel_size, px:px + panel_size] = resized

        # Border
        cv2.rectangle(canvas, (px, py), (px + panel_size - 1, py + panel_size - 1),
                       (80, 80, 80), 1)

        # Crosshair at landmark position
        lm_x = int(card['lm_px'] * panel_size / card['patch_size'][1])
        lm_y = int(card['lm_py'] * panel_size / card['patch_size'][0])
        cr = 12
        color = (0, 255, 255)  # yellow
        cv2.circle(canvas, (px + lm_x, py + lm_y), cr, color, 1, cv2.LINE_AA)
        # Crosshair lines
        cv2.line(canvas, (px + lm_x - cr - 4, py + lm_y), (px + lm_x - 4, py + lm_y), color, 1)
        cv2.line(canvas, (px + lm_x + 4, py + lm_y), (px + lm_x + cr + 4, py + lm_y), color, 1)
        cv2.line(canvas, (px + lm_x, py + lm_y - cr - 4), (px + lm_x, py + lm_y - 4), color, 1)
        cv2.line(canvas, (px + lm_x, py + lm_y + 4), (px + lm_x, py + lm_y + cr + 4), color, 1)

        # Label
        ts = 0.5
        (tw, _), _ = cv2.getTextSize(label, FONT, ts, 1)
        cv2.putText(canvas, label, (px + (panel_size - tw) // 2, py + panel_size + 22),
                    FONT, ts, WHITE, 1, cv2.LINE_AA)

    # Tile name
    tile_label = card['tile'].upper().replace('_', ' ')
    ts = 0.6
    (tw, _), _ = cv2.getTextSize(tile_label, FONT, ts, 1)
    cv2.putText(canvas, tile_label, ((W - tw) // 2, y_start - 15),
                FONT, ts, WHITE, 1, cv2.LINE_AA)

    # Caption
    cap = 'SINGLE-CELL  MULTIMODAL  REGISTRATION'
    ts = 0.72
    (tw, _), _ = cv2.getTextSize(cap, FONT, ts, 1)
    cv2.putText(canvas, cap, ((W - tw) // 2, H - 42), FONT, ts, WHITE, 1, cv2.LINE_AA)

    return canvas


def draw_tile_overview(card, highlight_alpha=1.0):
    """Draw the full tile slice with the patch region highlighted."""
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    sl = card['dense_slice']
    sh, sw = sl.shape[:2]

    # Scale to fit nicely in frame
    disp_scale = min((W * 0.6) / sw, (H * 0.7) / sh)
    dw = int(sw * disp_scale)
    dh = int(sh * disp_scale)
    resized = cv2.resize(sl, (dw, dh), interpolation=cv2.INTER_LANCZOS4)

    ox = (W - dw) // 2
    oy = (H - dh) // 2 - 20
    canvas[oy:oy + dh, ox:ox + dw] = resized

    # Highlight patch region
    if highlight_alpha > 0:
        rx0 = ox + int(card['patch_x0'] * disp_scale)
        ry0 = oy + int(card['patch_y0'] * disp_scale)
        rx1 = rx0 + int(card['patch_size'][1] * disp_scale)
        ry1 = ry0 + int(card['patch_size'][0] * disp_scale)
        color = (0, 255, 255)
        thickness = max(1, int(2 * highlight_alpha))
        cv2.rectangle(canvas, (rx0, ry0), (rx1, ry1), color, thickness)

    # Tile label
    tile_label = card['tile'].upper().replace('_', ' ')
    ts = 0.6
    (tw, _), _ = cv2.getTextSize(tile_label, FONT, ts, 1)
    cv2.putText(canvas, tile_label, ((W - tw) // 2, oy - 10), FONT, ts, WHITE, 1, cv2.LINE_AA)

    cap = 'SINGLE-CELL  MULTIMODAL  REGISTRATION'
    ts = 0.72
    (tw, _), _ = cv2.getTextSize(cap, FONT, ts, 1)
    cv2.putText(canvas, cap, ((W - tw) // 2, H - 42), FONT, ts, WHITE, 1, cv2.LINE_AA)

    return canvas


# ── Build frames ──
import shutil, time
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)

print("\nRendering cell zoom frames...")
fi = 0

# For each cell card: zoom-in (18fr) → hold card (36fr) → transition (12fr)
ZOOM_IN = 18
HOLD = 36
TRANSITION = 12
FRAMES_PER_CELL = ZOOM_IN + HOLD + TRANSITION

for ci, card in enumerate(cell_cards):
    overview = draw_tile_overview(card, highlight_alpha=1.0)
    cell_card_img = draw_cell_card(card)

    # Zoom in: overview → cell card
    for f in range(ZOOM_IN):
        t = ease(f / max(1, ZOOM_IN - 1))
        frame = cv2.addWeighted(overview, 1 - t, cell_card_img, t, 0)
        fi += 1
        cv2.imwrite(f'{OUT_DIR}/frame_{fi:05d}.png', frame)

    # Hold: show cell card
    for f in range(HOLD):
        fi += 1
        cv2.imwrite(f'{OUT_DIR}/frame_{fi:05d}.png', cell_card_img)

    # Transition to next
    if ci < len(cell_cards) - 1:
        next_overview = draw_tile_overview(cell_cards[ci + 1], highlight_alpha=1.0)
        for f in range(TRANSITION):
            t = ease(f / max(1, TRANSITION - 1))
            frame = cv2.addWeighted(cell_card_img, 1 - t, next_overview, t, 0)
            fi += 1
            cv2.imwrite(f'{OUT_DIR}/frame_{fi:05d}.png', frame)
    else:
        # Last cell: fade to black
        for f in range(TRANSITION):
            t = ease(f / max(1, TRANSITION - 1))
            frame = (cell_card_img.astype(np.float32) * (1 - t)).astype(np.uint8)
            fi += 1
            cv2.imwrite(f'{OUT_DIR}/frame_{fi:05d}.png', frame)

    print(f"  Cell {ci+1}/{len(cell_cards)}: {card['tile']} done")

print(f"\nDone! {fi} frames saved to {OUT_DIR}/")
