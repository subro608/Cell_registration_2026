"""
Parallel scene5b renderer — loads from pre-computed assets pkl.
Uses ThreadPoolExecutor for parallel frame rendering.
"""

import numpy as np, cv2, math, os, shutil, pickle
from multiprocessing import cpu_count

BASE = '/Users/neurolab/neuroinformatics/margaret'
FRAMES_NEW = f'{BASE}/animation/frames_multi_tile_3d'

W, H = 1920, 1080
FPS = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)

INIT_ROT_X = -0.3
STATIC_ROT_Y = 0.25
STATIC_ROT_X = INIT_ROT_X
SPIN_AMP = 0.12    # gentle tilt amplitude (rad) to show 3D depth
SPIN_PERIOD = 120   # frames per oscillation cycle
VOLUME_SCALE = 1.8  # scale up the final stitched 3D volume

TILES = ['row1_3',
         'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
         'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
         'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
         'row5_1']

def ease(t):
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))

def caption(frame, text, alpha=1.0):
    if alpha < 0.01: return
    ts, th = 0.72, 1
    (tw, _), _ = cv2.getTextSize(text, FONT, ts, th)
    x = (W - tw) // 2; y = H - 42
    col = tuple(int(v * alpha) for v in WHITE)
    cv2.putText(frame, text, (x, y), FONT, ts, col, th, cv2.LINE_AA)


def draw_axes(frame, rot_y, rot_x, cx=120, cy=None, ax_len=70, alpha=1.0):
    """Draw 3D axis trident that rotates with the volume."""
    if alpha < 0.01:
        return
    if cy is None:
        cy = H - 120
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)
    axes_3d = [
        (1, 0, 0),     # ML — horizontal in slice
        (0, -1, 0),    # AP — vertical in slice (up)
        (0, 0, 1),     # DV — depth (z-stack, down/ventral)
    ]
    ax_colors = [(0, 0, 180), (40, 40, 40), (200, 80, 0)]
    ax_labels = ['ML', 'AP', 'DV']
    for ai, (ux, uy, uz) in enumerate(axes_3d):
        rx = cos_y * ux + sin_y * uz
        ry = uy
        rz = -sin_y * ux + cos_y * uz
        ry2 = cos_x * ry - sin_x * rz
        px, py = int(cx + rx * ax_len), int(cy + ry2 * ax_len)
        nx, ny = int(cx - rx * ax_len), int(cy - ry2 * ax_len)
        col = tuple(int(c * alpha) for c in ax_colors[ai])
        cv2.line(frame, (nx, ny), (cx, cy), col, 2, cv2.LINE_AA)
        cv2.arrowedLine(frame, (cx, cy), (px, py), col, 3, cv2.LINE_AA, tipLength=0.15)
        dx_tip = rx / max(abs(rx), abs(ry2), 0.01) * 18
        dy_tip = ry2 / max(abs(rx), abs(ry2), 0.01) * 18
        lx, ly = int(px + dx_tip), int(py + dy_tip)
        tcol = tuple(int(c * alpha) for c in ax_colors[ai])
        cv2.putText(frame, ax_labels[ai], (lx - 10, ly + 5), FONT, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, ax_labels[ai], (lx - 10, ly + 5), FONT, 0.55, tcol, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 4, tuple(int(200*alpha) for _ in range(3)), -1, cv2.LINE_AA)


# ── Global tile data (set after loading, shared via fork) ──
tile_data = {}
tile_list = []
grid_positions = {}
tile_final_z = {}       # exaggerated z for merge animation
tile_real_z = {}        # physically correct z for final view
tile_real_xy = {}       # real XY offsets in display pixels for stacked view
GRID_SCALE = 1.0
FULL_SCALE = 1.0


def render_tile_3d_at(td, rot_y, rot_x, cx, cy, scale=1.0, alpha_val=0.7, z_offset=0.0,
                      merscope_img=None, merscope_alpha=0.0):
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)
    dense = td['dense']
    dense_z = td['dense_z']
    center_z = td['center_z']

    z_depths = []
    for i in range(len(dense)):
        dz = dense_z[i] - center_z + z_offset
        rz2 = cos_x * (cos_y * dz)
        z_depths.append((rz2, i))
    z_depths.sort(key=lambda x: x[0])

    for depth, i in z_depths:
        sl = dense[i].astype(np.float32) / 255.0
        # Composite MERSCOPE gene dots (already clean from right-left subtraction)
        if merscope_img is not None and merscope_alpha > 0:
            ms = merscope_img.astype(np.float32) / 255.0
            ms_mask = np.max(ms, axis=2) > 0.05
            ms_mask3 = np.stack([ms_mask]*3, axis=-1)
            sl = np.where(ms_mask3, sl * (1 - merscope_alpha) + ms * merscope_alpha, sl)
        sh, sw = sl.shape[:2]
        hw, hh = sw * scale / 2, sh * scale / 2
        dz = dense_z[i] - center_z + z_offset

        corners_3d = np.array([
            [-hw, -hh, dz * scale], [hw, -hh, dz * scale],
            [hw, hh, dz * scale], [-hw, hh, dz * scale]
        ], dtype=np.float64)

        rot_corners = []
        for c in corners_3d:
            rx = cos_y * c[0] + sin_y * c[2]
            ry = c[1]
            rz = -sin_y * c[0] + cos_y * c[2]
            ry2 = cos_x * ry - sin_x * rz
            rot_corners.append([rx + cx, ry2 + cy])

        rot_corners = np.array(rot_corners, dtype=np.float32)
        src_corners = np.array([[0,0],[sw,0],[sw,sh],[0,sh]], dtype=np.float32)
        M_persp = cv2.getPerspectiveTransform(src_corners, rot_corners)
        warped = cv2.warpPerspective(sl, M_persp, (W, H))
        mask = np.max(warped, axis=2) > 0.01
        mask3 = np.stack([mask]*3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, warped * alpha_val), canvas)

    return canvas


def render_multi_tile(positions, rot_y, rot_x, scales=None, z_offsets=None, merscope_alpha=0.0):
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    for tile_name, (cx, cy) in positions.items():
        if tile_name not in tile_data: continue
        if cx < -500 or cx > W + 500 or cy < -500 or cy > H + 500: continue
        td = tile_data[tile_name]
        scale = scales.get(tile_name, 1.0) if scales else 1.0
        if scale < 0.05: continue  # skip tiles too small to render
        z_off = z_offsets.get(tile_name, 0.0) if z_offsets else 0.0
        ms_img = tile_merscope.get(tile_name) if merscope_alpha > 0 else None
        tc = render_tile_3d_at(td, rot_y, rot_x, cx, cy, scale=scale, z_offset=z_off,
                               merscope_img=ms_img, merscope_alpha=merscope_alpha)
        mask = np.max(tc, axis=2) > 0.01
        mask3 = np.stack([mask]*3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, tc), canvas)
    return np.clip(canvas * 255, 0, 255).astype(np.uint8)


def render_frame(job):
    """Render a single frame given its parameters. Called by worker processes."""
    frame_idx, params = job
    phase = params['phase']
    rot_y = params.get('rot_y', STATIC_ROT_Y)
    rot_x = params.get('rot_x', STATIC_ROT_X)
    positions = params['positions']
    scales = params['scales']
    z_offsets = params.get('z_offsets', None)
    cap_text = params.get('caption', '')
    labels = params.get('labels', None)

    ms_alpha = params.get('merscope_alpha', 0.0)
    use_stitched = params.get('use_stitched', False)
    stitch_scale = params.get('stitch_scale', VOLUME_SCALE)
    stitch_subsample = params.get('stitch_subsample', 2)

    # Phase A0: scene5 last frame handling
    s5_last = params.get('s5_last')
    s5_blend_t = params.get('s5_blend_t', None)
    if s5_last is not None and s5_blend_t is None:
        # Hold: just copy scene5's last frame
        frame = s5_last.copy()
    elif s5_last is not None and s5_blend_t is not None:
        # Mask step: blend from scene5 frame to scene5b rendering (applies mask, removes annotations)
        scene5b_frame = render_multi_tile(positions, rot_y, rot_x, scales, z_offsets, merscope_alpha=ms_alpha)
        frame = cv2.addWeighted(s5_last, 1 - s5_blend_t, scene5b_frame, s5_blend_t, 0)
    elif use_stitched and stitched_data:
        frame = render_stitched_3d(rot_y, rot_x, W // 2, H // 2,
                                    scale=stitch_scale, subsample=stitch_subsample)
    else:
        frame = render_multi_tile(positions, rot_y, rot_x, scales, z_offsets, merscope_alpha=ms_alpha)

    if labels:
        for tile_name, (lx, ly, font_scale, alpha) in labels.items():
            if alpha < 0.05: continue
            td = tile_data[tile_name]
            lbl = td['name']
            (tw, _), _ = cv2.getTextSize(lbl, FONT, font_scale, 1)
            col = tuple(int(v * alpha) for v in WHITE)
            cv2.putText(frame, lbl, (lx - tw // 2, ly), FONT, font_scale, col, 1, cv2.LINE_AA)

    # Draw 3D axis widget (rotates with volume)
    draw_axes(frame, rot_y, rot_x)

    if cap_text:
        caption(frame, cap_text)

    out_path = f'{FRAMES_NEW}/frame_{frame_idx:05d}.png'
    cv2.imwrite(out_path, frame)
    return frame_idx


# Global merscope overlays per tile
tile_merscope = {}
# Stitched volume (loaded from assets)
stitched_data = {}


def render_stitched_3d(rot_y, rot_x, cx, cy, scale=1.0, alpha_val=0.7, subsample=1):
    """Render the pre-built stitched volume as a single 3D object."""
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)
    volume = stitched_data['volume']
    z_vals = stitched_data['z']
    center_z = (z_vals[0] + z_vals[-1]) / 2
    # Z scale: convert native z-slices to display pixels
    avg_um_per_dpx = stitched_data['avg_um_per_dpx']
    z_px_per_slice = 2.0 / avg_um_per_dpx  # 2µm per native slice → display pixels

    n_slices = len(volume)
    indices = range(0, n_slices, subsample)

    z_depths = []
    for i in indices:
        dz = (z_vals[i] - center_z) * z_px_per_slice
        rz2 = cos_x * (cos_y * dz)
        z_depths.append((rz2, i, dz))
    z_depths.sort(key=lambda x: x[0])

    sh, sw = volume.shape[1], volume.shape[2]
    src_corners = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float32)

    for depth, i, dz in z_depths:
        sl = volume[i].astype(np.float32) / 255.0
        hw, hh = sw * scale / 2, sh * scale / 2

        corners_3d = np.array([
            [-hw, -hh, dz * scale], [hw, -hh, dz * scale],
            [hw, hh, dz * scale], [-hw, hh, dz * scale]
        ], dtype=np.float64)

        rot_corners = []
        for c in corners_3d:
            rx = cos_y * c[0] + sin_y * c[2]
            ry = c[1]
            rz = -sin_y * c[0] + cos_y * c[2]
            ry2 = cos_x * ry - sin_x * rz
            rot_corners.append([rx + cx, ry2 + cy])

        rot_corners = np.array(rot_corners, dtype=np.float32)
        M_persp = cv2.getPerspectiveTransform(src_corners, rot_corners)
        warped = cv2.warpPerspective(sl, M_persp, (W, H))
        mask = np.max(warped, axis=2) > 0.01
        mask3 = np.stack([mask] * 3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, warped * alpha_val), canvas)

    return np.clip(canvas * 255, 0, 255).astype(np.uint8)


def load_tiles():
    """Load all tile data from pre-computed assets pkl."""
    global tile_data, tile_list, grid_positions, tile_final_z, GRID_SCALE, FULL_SCALE

    print("Loading assets from scene5b_assets_v3.pkl...")
    with open(f'{BASE}/animation/scene5b_assets_v3.pkl', 'rb') as f:
        assets = pickle.load(f)

    # Compute per-tile ex-vivo p99 for normalization
    tile_ev_p99 = {}
    for tile in TILES:
        if tile not in assets: continue
        dense = assets[tile]['dense']
        ev_nz = dense[:, :, :, 0].ravel()
        ev_nz = ev_nz[ev_nz > 0]
        tile_ev_p99[tile] = float(np.percentile(ev_nz, 99)) if len(ev_nz) > 100 else 255.0
    target_p99 = float(np.percentile(list(tile_ev_p99.values()), 25))
    print(f"  Ex-vivo normalization target p99: {target_p99:.0f} (p25 — reduces bright dorsal tiles)")

    for tile in TILES:
        if tile not in assets: continue
        a = assets[tile]
        dense = a['dense'].copy()

        # Normalize ex-vivo brightness (B and R channels)
        ev_scale = target_p99 / max(tile_ev_p99[tile], 1.0)
        if abs(ev_scale - 1.0) > 0.05:
            for ch in [0, 2]:
                dense[:, :, :, ch] = np.clip(
                    dense[:, :, :, ch].astype(np.float32) * ev_scale, 0, 255
                ).astype(np.uint8)
            print(f"  {tile}: boosted ex-vivo {ev_scale:.2f}x (p99={tile_ev_p99[tile]:.0f})")

        tile_data[tile] = {
            'dense': dense, 'dense_z': a['dense_z'], 'center_z': a['center_z'],
            'n_slices': a['n_slices'], 'cell_w': a['cell_w'], 'cell_h': a['cell_h'],
            'crop_h_nd2': a['crop_h_nd2'],
            'name': tile.upper().replace('_', ' '),
        }
        if a['merscope'] is not None:
            tile_merscope[tile] = a['merscope']
        print(f"  {tile}: {len(dense)} slices, {a['cell_w']}x{a['cell_h']}")

    # Load stitched volume
    if '_stitched' in assets:
        s = assets['_stitched']
        avg_um_per_dpx = np.mean([(assets[t]['crop_h_nd2'] * 0.65) / assets[t]['cell_h']
                                   for t in TILES if t in assets])
        stitched_data['volume'] = s['volume']
        stitched_data['z'] = s['z']
        stitched_data['width'] = s['width']
        stitched_data['height'] = s['height']
        stitched_data['avg_um_per_dpx'] = avg_um_per_dpx
        print(f"  Stitched: {s['volume'].shape}, z=[{s['z'][0]:.0f},{s['z'][-1]:.0f}]")

    print(f"\nLoaded {len(tile_data)} tiles")

    # Grid layout
    tile_list.extend(list(tile_data.keys()))
    GRID_COLS = 5
    avg_cell_w = sum(td['cell_w'] for td in tile_data.values()) // len(tile_data)
    avg_cell_h = sum(td['cell_h'] for td in tile_data.values()) // len(tile_data)
    CELL_GAP_X, CELL_GAP_Y = 20, 15
    GRID_ROWS = 4
    grid_w = GRID_COLS * avg_cell_w + (GRID_COLS - 1) * CELL_GAP_X
    grid_h = GRID_ROWS * avg_cell_h + (GRID_ROWS - 1) * CELL_GAP_Y
    FULL_SCALE = 778 / tile_data['row5_1']['cell_h']
    # Scale grid to fit all 19 tiles in 1920x1080 with margin
    margin = 80
    fit_scale = min((W - margin) / grid_w, (H - margin) / grid_h)
    GRID_SCALE = fit_scale
    print(f"  Grid: {grid_w}x{grid_h}, fit_scale={fit_scale:.3f}")
    # Recompute grid positions at fitted scale
    fitted_grid_w = grid_w * fit_scale
    fitted_grid_h = grid_h * fit_scale
    grid_x0 = (W - fitted_grid_w) / 2
    grid_y0 = (H - fitted_grid_h) / 2
    for idx, tn in enumerate(tile_list):
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        cx = grid_x0 + (col * (avg_cell_w + CELL_GAP_X) + avg_cell_w / 2) * fit_scale
        cy = grid_y0 + (row * (avg_cell_h + CELL_GAP_Y) + avg_cell_h / 2) * fit_scale
        grid_positions[tn] = (int(cx), int(cy))

    # Z offsets for stacking — from stitched canvas z_offsets (nd2 space, 2µm per slice)
    # Canvas z-offsets are in native z-slices (each = 2.0 µm)
    avg_um_per_dpx = np.mean([(td['crop_h_nd2'] * 0.65) / td['cell_h']
                               for td in tile_data.values()])
    REAL_Z_PX_PER_SLICE = 2.0 / avg_um_per_dpx  # native z-step (2µm) in display pixels
    ANIM_Z_SCALE = 0.8  # exaggerated for visible merge animation

    z_offsets_raw = {tn: assets[tn]['stitch_z_offset'] for tn in tile_list}
    z_center = (min(z_offsets_raw.values()) + max(z_offsets_raw.values())) / 2
    for tn in tile_list:
        tile_final_z[tn] = (z_offsets_raw[tn] - z_center) * ANIM_Z_SCALE
        tile_real_z[tn] = (z_offsets_raw[tn] - z_center) * REAL_Z_PX_PER_SLICE
    print(f"  Z scaling: anim={ANIM_Z_SCALE}/slice, real={REAL_Z_PX_PER_SLICE:.3f} dpx/slice")

    # Real XY offsets: from stitched canvas positions (nd2 space, 0.65 µm/px)
    xy_scale = 1.0 / (avg_um_per_dpx / 0.65)  # canvas px → display px
    x_mean = np.mean([assets[t]['canvas_x'] for t in tile_list])
    y_mean = np.mean([assets[t]['canvas_y'] for t in tile_list])
    for tn in tile_list:
        dx = (assets[tn]['canvas_x'] - x_mean) * xy_scale
        dy = (assets[tn]['canvas_y'] - y_mean) * xy_scale
        tile_real_xy[tn] = (dx, dy)
    print(f"  XY scale: {xy_scale:.3f} display px per canvas px")
    print(f"  XY range: x=[{min(v[0] for v in tile_real_xy.values()):.0f}, {max(v[0] for v in tile_real_xy.values()):.0f}], "
          f"y=[{min(v[1] for v in tile_real_xy.values()):.0f}, {max(v[1] for v in tile_real_xy.values()):.0f}]")


def build_frame_jobs():
    """Pre-compute all frame parameters."""
    jobs = []
    fi_global = 0

    row5_1_grid = grid_positions.get('row5_1', (W // 2, H // 2))

    # Load scene5 last frame as literal starting frame
    import subprocess
    s5_last = None
    s5_path = f'{BASE}/animation/scene5_all_tiles_v2_h264.mp4'
    if os.path.exists(s5_path):
        tmp_img = '/tmp/_s5b_last_frame.png'
        subprocess.run(['ffmpeg', '-y', '-sseof', '-0.1', '-i', s5_path,
                        '-frames:v', '1', tmp_img], capture_output=True)
        if os.path.exists(tmp_img):
            s5_last = cv2.imread(tmp_img)
            print(f"  Loaded scene5 last frame: {s5_last.shape}")

    # Phase A0: multi-step transition from scene5's last frame
    # Step 1 (12fr): hold scene5 last frame
    # Step 2 (24fr): blend scene5 → scene5b flat rendering (removes landmarks/scalebar, applies mask)
    # Step 3 (24fr): move from right-half to center + introduce 3D tilt
    A0_HOLD = 12
    A0_MASK = 24
    A0_MOVE = 24
    S5_END_CX, S5_END_CY = 1200, 430  # scene5 tile center position

    # Step 1: hold scene5's actual last frame
    for fi in range(A0_HOLD):
        fi_global += 1
        jobs.append((fi_global, {
            'phase': 'A0_hold',
            'rot_y': 0.0, 'rot_x': 0.0,
            'positions': {}, 'scales': {},
            's5_last': s5_last,
            'caption': '', 'labels': {},
        }))

    # Step 2: blend from scene5 frame to scene5b's flat rendering at same position
    # This smoothly removes yellow circles, scale bar, and applies tissue mask
    for fi in range(A0_MASK):
        t = ease(fi / max(1, A0_MASK - 1))
        fi_global += 1
        jobs.append((fi_global, {
            'phase': 'A0_mask',
            'rot_y': 0.0, 'rot_x': 0.0,  # still flat
            'positions': {'row5_1': (S5_END_CX, S5_END_CY)},  # same position as scene5
            'scales': {'row5_1': FULL_SCALE},
            's5_last': s5_last,
            's5_blend_t': t,
            'caption': '', 'labels': {},
        }))

    # Step 3: move from right-half to center + introduce 3D tilt
    for fi in range(A0_MOVE):
        t = ease(fi / max(1, A0_MOVE - 1))
        cx = int(S5_END_CX * (1 - t) + (W // 2) * t)
        cy = int(S5_END_CY * (1 - t) + (H // 2) * t)
        rot_x = 0.0 * (1 - t) + (-0.3) * t  # flat → tilted

        fi_global += 1
        jobs.append((fi_global, {
            'phase': 'A0_move',
            'rot_y': 0.0, 'rot_x': rot_x,
            'positions': {'row5_1': (cx, cy)},
            'scales': {'row5_1': FULL_SCALE},
            'caption': 'GREEN = IN VIVO    MAGENTA = EX VIVO',
            'labels': {},
        }))

    # Phase A: zoom out from row5_1 (now centered after A0_move) to full grid (60 frames = 2.5s)
    A_FRAMES = 60
    for fi in range(A_FRAMES):
        t = ease(fi / (A_FRAMES - 4))
        # Rotation transitions from A0 end (rot_y=0, rot_x=-0.3) to static angle
        rot_y = 0.0 * (1 - t) + STATIC_ROT_Y * t
        rot_x = -0.3 * (1 - t) + STATIC_ROT_X * t

        positions = {}; scales = {}
        # Row5_1 moves from center to its grid slot
        positions['row5_1'] = (int(W // 2 * (1 - t) + row5_1_grid[0] * t),
                               int((H // 2) * (1 - t) + row5_1_grid[1] * t))
        scales['row5_1'] = FULL_SCALE * (1 - t) + GRID_SCALE * t

        # Other tiles emerge from row5_1's current position
        r5x, r5y = positions['row5_1']
        other_alpha = ease(max(0, (t - 0.15) / 0.85))
        for tn in tile_list:
            if tn == 'row5_1': continue
            gx, gy = grid_positions[tn]
            positions[tn] = (int(r5x * (1 - other_alpha) + gx * other_alpha),
                             int(r5y * (1 - other_alpha) + gy * other_alpha))
            scales[tn] = GRID_SCALE * other_alpha

        labels = {}
        if t > 0.7:
            la = ease((t - 0.7) / 0.3)
            for tn, (gx, gy) in grid_positions.items():
                if tn not in tile_data: continue
                td = tile_data[tn]
                labels[tn] = (gx, gy - int(td['cell_h'] * GRID_SCALE) // 2 - 8, 0.28, la)

        fi_global += 1
        jobs.append((fi_global, {
            'phase': 'A', 'rot_y': rot_y, 'rot_x': rot_x,
            'positions': positions, 'scales': scales,
            'caption': f'IN VIVO  --  EX VIVO  REGISTRATION  --  {len(tile_data)}  TILES',
            'labels': labels,
        }))

    # Phase B: hold with slow spin (24 frames)
    spin_fi = 0  # running spin counter across B/B1.5/B2/B3
    for fi in range(24):
        positions = {tn: grid_positions[tn] for tn in tile_list}
        scales = {tn: GRID_SCALE for tn in tile_list}
        labels = {}
        for tn, (gx, gy) in grid_positions.items():
            if tn not in tile_data: continue
            td = tile_data[tn]
            labels[tn] = (gx, gy - int(td['cell_h'] * GRID_SCALE) // 2 - 8, 0.28, 1.0)

        fi_global += 1
        jobs.append((fi_global, {
            'phase': 'B', 'rot_y': STATIC_ROT_Y + SPIN_AMP * math.sin(2 * math.pi * spin_fi / SPIN_PERIOD), 'rot_x': STATIC_ROT_X,
            'positions': positions, 'scales': scales,
            'caption': f'IN VIVO  --  EX VIVO  REGISTRATION  --  {len(tile_data)}  TILES',
            'labels': labels,
        }))
        spin_fi += 1

    # Phase B1.5: MERSCOPE gene dots fade in (36 frames = 1.5s)
    for fi in range(36):
        t = ease(fi / 30)
        positions = {tn: grid_positions[tn] for tn in tile_list}
        scales = {tn: GRID_SCALE for tn in tile_list}
        labels = {}
        for tn, (gx, gy) in grid_positions.items():
            if tn not in tile_data: continue
            td = tile_data[tn]
            labels[tn] = (gx, gy - int(td['cell_h'] * GRID_SCALE) // 2 - 8, 0.28, 1.0)

        fi_global += 1
        jobs.append((fi_global, {
            'phase': 'B1.5', 'rot_y': STATIC_ROT_Y + SPIN_AMP * math.sin(2 * math.pi * spin_fi / SPIN_PERIOD), 'rot_x': STATIC_ROT_X,
            'positions': positions, 'scales': scales,
            'merscope_alpha': t * 0.6,  # fade in to 60% opacity
            'caption': f'MERSCOPE  mRNA  EXPRESSION',
            'labels': labels,
        }))
        spin_fi += 1

    # Phase B2: fly-by (with MERSCOPE visible)
    row_ys = sorted(set(cy for _, cy in grid_positions.values()))
    n_rows_grid = len(row_ys)
    row_x_ranges = {}
    for tn, (gx, gy) in grid_positions.items():
        for ri, ry in enumerate(row_ys):
            if abs(gy - ry) < 5:
                if ri not in row_x_ranges: row_x_ranges[ri] = [gx, gx]
                else:
                    row_x_ranges[ri][0] = min(row_x_ranges[ri][0], gx)
                    row_x_ranges[ri][1] = max(row_x_ranges[ri][1], gx)

    all_xs = [cx for cx, _ in grid_positions.values()]
    x_min_grid, x_max_grid = min(all_xs), max(all_xs)
    x_center_grid = (x_min_grid + x_max_grid) / 2
    y_center_grid = (row_ys[0] + row_ys[-1]) / 2

    ZOOM_LEVEL = 2.5
    ZOOM_IN_FRAMES = 18; HOLD_FRAMES = 48; PAN_FRAMES = 24
    TRANS_FRAMES = 10; ZOOM_OUT_FRAMES = 18
    PER_ROW = HOLD_FRAMES + PAN_FRAMES + TRANS_FRAMES
    FLYBY_FRAMES = ZOOM_IN_FRAMES + n_rows_grid * PER_ROW - TRANS_FRAMES + ZOOM_OUT_FRAMES

    for fi in range(FLYBY_FRAMES):
        if fi < ZOOM_IN_FRAMES:
            zt = ease(fi / max(1, ZOOM_IN_FRAMES - 1))
            zoom = 1.0 + (ZOOM_LEVEL - 1.0) * zt
            cam_x = x_center_grid * (1 - zt) + row_x_ranges[0][0] * zt
            cam_y = y_center_grid * (1 - zt) + row_ys[0] * zt
        elif fi >= FLYBY_FRAMES - ZOOM_OUT_FRAMES:
            zoom_fi = fi - (FLYBY_FRAMES - ZOOM_OUT_FRAMES)
            zt = ease(zoom_fi / max(1, ZOOM_OUT_FRAMES - 1))
            zoom = ZOOM_LEVEL * (1 - zt) + 1.0 * zt
            cam_x = row_x_ranges[n_rows_grid-1][1] * (1 - zt) + x_center_grid * zt
            cam_y = row_ys[-1] * (1 - zt) + y_center_grid * zt
        else:
            zoom = ZOOM_LEVEL
            row_fi = fi - ZOOM_IN_FRAMES
            row_idx = min(row_fi // PER_ROW, n_rows_grid - 1)
            sub_fi = row_fi - row_idx * PER_ROW
            xl = row_x_ranges[row_idx][0]; xr = row_x_ranges[row_idx][1]
            if sub_fi < HOLD_FRAMES:
                cam_x = xl; cam_y = row_ys[row_idx]
            elif sub_fi < HOLD_FRAMES + PAN_FRAMES:
                pan_t = ease((sub_fi - HOLD_FRAMES) / max(1, PAN_FRAMES - 1))
                cam_x = xl + (xr - xl) * pan_t; cam_y = row_ys[row_idx]
            else:
                trans_t = ease((sub_fi - HOLD_FRAMES - PAN_FRAMES) / max(1, TRANS_FRAMES - 1))
                next_row = min(row_idx + 1, n_rows_grid - 1)
                cam_x = xr * (1 - trans_t) + row_x_ranges[next_row][0] * trans_t
                cam_y = row_ys[row_idx] * (1 - trans_t) + row_ys[next_row] * trans_t

        positions = {}; scales = {}
        for tn in tile_list:
            gx, gy = grid_positions[tn]
            positions[tn] = (int(W/2 + (gx - cam_x) * zoom), int(H/2 + (gy - cam_y) * zoom))
            scales[tn] = GRID_SCALE * zoom

        labels = {}
        font_scale = 0.32 * min(zoom, 2.0)
        for tn, (px, py) in positions.items():
            if tn not in tile_data: continue
            if px < -200 or px > W + 200 or py < -200 or py > H + 200: continue
            td = tile_data[tn]
            labels[tn] = (px, py - int(td['cell_h'] * zoom / 2) - 8, font_scale, 1.0)

        fi_global += 1
        jobs.append((fi_global, {
            'phase': 'B2', 'rot_y': STATIC_ROT_Y + SPIN_AMP * math.sin(2 * math.pi * spin_fi / SPIN_PERIOD), 'rot_x': STATIC_ROT_X,
            'positions': positions, 'scales': scales,
            'merscope_alpha': 0.6,  # gene dots visible during fly-by
            'caption': f'MERSCOPE  mRNA  EXPRESSION',
            'labels': labels,
        }))
        spin_fi += 1

    # Phase B3: hold (12 frames)
    for fi in range(12):
        positions = {tn: grid_positions[tn] for tn in tile_list}
        scales = {tn: GRID_SCALE for tn in tile_list}
        labels = {}
        for tn, (gx, gy) in grid_positions.items():
            if tn not in tile_data: continue
            td = tile_data[tn]
            labels[tn] = (gx, gy - int(td['cell_h'] * GRID_SCALE) // 2 - 8, 0.28, 1.0)

        fi_global += 1
        jobs.append((fi_global, {
            'phase': 'B3', 'rot_y': STATIC_ROT_Y + SPIN_AMP * math.sin(2 * math.pi * spin_fi / SPIN_PERIOD), 'rot_x': STATIC_ROT_X,
            'positions': positions, 'scales': scales,
            'merscope_alpha': 0.6,
            'caption': f'MERSCOPE  mRNA  EXPRESSION',
            'labels': labels,
        }))
        spin_fi += 1

    # Phase C: merge — tiles move from grid to center with z-depth (96 frames = 4s, smooth)
    C_FRAMES = 96
    for fi in range(C_FRAMES):
        t = ease(fi / max(1, C_FRAMES - 6))
        vol_s = 1.0 + (VOLUME_SCALE - 1.0) * t
        positions = {}; scales = {}; z_offsets = {}
        for tn in tile_list:
            gx, gy = grid_positions[tn]
            positions[tn] = (int(gx * (1-t) + W // 2 * t), int(gy * (1-t) + H // 2 * t))
            scales[tn] = GRID_SCALE * vol_s
            z_offsets[tn] = tile_final_z[tn] * t * VOLUME_SCALE

        labels = {}
        la = max(0, 1 - t * 3)
        if la > 0.05:
            for tn, (px, py) in positions.items():
                if tn not in tile_data: continue
                td = tile_data[tn]
                labels[tn] = (px, py - int(td['cell_h'] * vol_s) // 2 - 8, 0.28, la)

        # MERSCOPE fades out smoothly during merge
        ms_alpha = 0.6 * (1 - ease(max(0, (t - 0.3) / 0.7)))  # starts fading at t=0.3

        fi_global += 1
        jobs.append((fi_global, {
            'phase': 'C', 'rot_y': STATIC_ROT_Y + SPIN_AMP * math.sin(2 * math.pi * spin_fi / SPIN_PERIOD), 'rot_x': STATIC_ROT_X,
            'positions': positions, 'scales': scales, 'z_offsets': z_offsets,
            'merscope_alpha': ms_alpha,
            'caption': 'COMBINING  TILES  INTO  STITCHED  3D  VOLUME',
            'labels': labels,
        }))
        spin_fi += 1

    # Phase C2: scale-up transition — individual tiles grow to match stitched volume size (48 frames = 2s)
    # At end of C: tiles at center, scale = GRID_SCALE * VOLUME_SCALE, z_offsets = tile_final_z * VOLUME_SCALE
    # Need to reach: scale = VOLUME_SCALE (per-tile), same z structure as stitched volume
    # Also transition z_offsets from tile_final_z (anim-exaggerated) to tile_real_z (physically correct)
    C2_FRAMES = 48
    c_end_scale = GRID_SCALE * VOLUME_SCALE  # where Phase C left off
    c_target_scale = VOLUME_SCALE             # match stitched volume
    for fi in range(C2_FRAMES):
        t = ease(fi / max(1, C2_FRAMES - 1))
        cur_scale = c_end_scale + (c_target_scale - c_end_scale) * t
        positions = {}; scales = {}; z_offsets = {}
        for tn in tile_list:
            positions[tn] = (W // 2, H // 2)
            scales[tn] = cur_scale
            # Transition z from anim-exaggerated to physically correct (matching stitched volume)
            z_anim = tile_final_z[tn] * VOLUME_SCALE
            z_real = tile_real_z[tn] * VOLUME_SCALE
            z_offsets[tn] = z_anim + (z_real - z_anim) * t

        fi_global += 1
        jobs.append((fi_global, {
            'phase': 'C2',
            'rot_y': STATIC_ROT_Y + SPIN_AMP * math.sin(2 * math.pi * spin_fi / SPIN_PERIOD),
            'rot_x': STATIC_ROT_X,
            'positions': positions, 'scales': scales, 'z_offsets': z_offsets,
            'merscope_alpha': 0.0,
            'caption': 'COMBINING  TILES  INTO  STITCHED  3D  VOLUME',
            'labels': {},
        }))
        spin_fi += 1

    # Phase D: full 360° rotation of stitched volume (144 frames = 6s)
    D_FRAMES = 144
    d_start_rot = STATIC_ROT_Y
    for fi in range(D_FRAMES):
        d_t = fi / max(1, D_FRAMES - 1)
        d_rot = d_start_rot + d_t * 2 * math.pi  # full 360°

        fi_global += 1
        jobs.append((fi_global, {
            'phase': 'D', 'rot_y': d_rot, 'rot_x': STATIC_ROT_X,
            'positions': {}, 'scales': {},
            'use_stitched': True,
            'stitch_scale': VOLUME_SCALE,
            'stitch_subsample': 2,  # render every 2nd slice for speed
            'caption': 'STITCHED  3D  VOLUME  --  ALL  19  TILES',
            'labels': {},
        }))

    return jobs


if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Step 1: Load tiles (serial)
    load_tiles()

    # Step 2: Build all frame jobs
    print("\nBuilding frame jobs...")
    jobs = build_frame_jobs()
    print(f"  {len(jobs)} frames to render")

    # Step 3: Clean output dir
    if os.path.exists(FRAMES_NEW):
        shutil.rmtree(FRAMES_NEW)
    os.makedirs(FRAMES_NEW)

    # Step 4: Render in parallel using threads (cv2/numpy release GIL)
    N_WORKERS = min(cpu_count(), 12)
    print(f"\nRendering with {N_WORKERS} threads...")

    import time
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(render_frame, job): job[0] for job in jobs}
        for future in as_completed(futures):
            idx = future.result()
            done += 1
            if done % 20 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (len(jobs) - done) / rate
                print(f"  {done}/{len(jobs)} frames ({rate:.1f} fr/s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"\nDone! {len(jobs)} frames in {elapsed:.0f}s ({len(jobs)/elapsed:.1f} fr/s)")
    print(f"Saved to {FRAMES_NEW}/")
