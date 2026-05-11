"""
Pre-build all assets needed for three_stacks_v2 rendering:

1. Per-tile: dense slices with MERSCOPE composited (dense_with_ms)
2. Per-tile channel splits: invivo, exvivo, merscope (already in assets, just copy)
3. Stitched volume with MERSCOPE (load from stitched_with_merscope.pkl)
4. Stitched channel splits: invivo-only, exvivo-only volumes

Saves to scene5b_three_stacks_assets.pkl
"""
import numpy as np, cv2, pickle, os, time

BASE = '/Users/neurolab/neuroinformatics/margaret'

TILES = ['row1_3',
         'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
         'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
         'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
         'row5_1']

MERSCOPE_ALPHA = 0.6

print("Loading main assets...")
t0 = time.time()
with open(f'{BASE}/animation/scene5b_assets_v3.pkl', 'rb') as f:
    assets = pickle.load(f)
print(f"  Loaded in {time.time()-t0:.0f}s")

out = {}

# ── Compute per-tile ex-vivo histogram equalization ──
# Erdem/Jason: oriens (dorsal, top rows) is over-exposed / washed out.
# Reduce bright tiles to match the dimmer ventral levels, not the other way around.
print("\nComputing per-tile ex-vivo normalization...")
tile_ev_p99 = {}
for tile in TILES:
    if tile not in assets:
        continue
    dense = assets[tile]['dense']
    ev_all = dense[:, :, :, 0].ravel()  # B channel (ex-vivo)
    ev_nz = ev_all[ev_all > 0]
    if len(ev_nz) > 100:
        tile_ev_p99[tile] = float(np.percentile(ev_nz, 99))
    else:
        tile_ev_p99[tile] = 255.0
    print(f"  {tile}: ex-vivo p99 = {tile_ev_p99[tile]:.0f}")

# Target: p25 of p99 values — brings bright dorsal tiles down to match dimmer ventral
# This reduces oriens washed-out look while slightly boosting the dimmest tiles
target_p99 = float(np.percentile(list(tile_ev_p99.values()), 25))
print(f"  Target p99: {target_p99:.0f} (p25 — reduces bright dorsal tiles)")

# ── Per-tile assets ──
print("\nBuilding per-tile assets...")
for tile in TILES:
    if tile not in assets:
        continue
    a = assets[tile]
    dense = a['dense'].copy()  # (n, h, w, 3) uint8 BGR
    ms = a.get('merscope')  # (h, w, 3) or None
    n, h, w, _ = dense.shape

    # Per-tile ex-vivo normalization: scale B and R channels
    ev_scale = target_p99 / max(tile_ev_p99[tile], 1.0)
    if abs(ev_scale - 1.0) > 0.05:
        print(f"  {tile}: boosting ex-vivo {ev_scale:.2f}x")
        for ch in [0, 2]:  # B and R
            dense[:, :, :, ch] = np.clip(
                dense[:, :, :, ch].astype(np.float32) * ev_scale, 0, 255
            ).astype(np.uint8)

    # Channel splits (no MERSCOPE)
    invivo = np.zeros((n, h, w, 3), dtype=np.uint8)
    invivo[:, :, :, 1] = dense[:, :, :, 1]

    exvivo = np.zeros((n, h, w, 3), dtype=np.uint8)
    exvivo[:, :, :, 0] = dense[:, :, :, 0]
    exvivo[:, :, :, 2] = dense[:, :, :, 2]

    # Dense with MERSCOPE composited
    if ms is not None:
        dense_with_ms = dense.copy()
        ms_f = ms.astype(np.float32)
        ms_mask = np.max(ms, axis=2) > 10
        for i in range(n):
            sl = dense_with_ms[i].astype(np.float32)
            sl[ms_mask] = sl[ms_mask] * (1 - MERSCOPE_ALPHA) + ms_f[ms_mask] * MERSCOPE_ALPHA
            dense_with_ms[i] = np.clip(sl, 0, 255).astype(np.uint8)
    else:
        dense_with_ms = dense

    out[tile] = {
        'dense': dense,
        'dense_with_ms': dense_with_ms,
        'invivo': invivo,
        'exvivo': exvivo,
        'merscope': ms,
        'dense_z': a['dense_z'],
        'center_z': a['center_z'],
        'cell_w': a['cell_w'],
        'cell_h': a['cell_h'],
        'stitch_z_offset': a['stitch_z_offset'],
        'canvas_x': a['canvas_x'],
        'canvas_y': a['canvas_y'],
        'crop_h_nd2': a['crop_h_nd2'],
        'n_slices': a['n_slices'],
    }
    print(f"  {tile}: {n} slices, {w}x{h}, ms={'yes' if ms is not None else 'no'}")

# ── Stitched volumes ──
print("\nBuilding stitched channel volumes...")

# Load stitched with MERSCOPE
stitch_ms_path = f'{BASE}/animation/stitched_with_merscope.pkl'
if os.path.exists(stitch_ms_path):
    with open(stitch_ms_path, 'rb') as f:
        stitch_ms = pickle.load(f)
    stitch_combined = stitch_ms['volume']  # with MERSCOPE baked in
    print(f"  Loaded stitched+MERSCOPE: {stitch_combined.shape}")
else:
    print("  WARNING: stitched_with_merscope.pkl not found, using plain")
    stitch_combined = assets['_stitched']['volume']

stitch_plain = assets['_stitched']['volume']
stitch_z = assets['_stitched']['z']
vol_w = assets['_stitched']['width']
vol_h = assets['_stitched']['height']

# Channel splits from plain volume (no MERSCOPE mixed in)
# Also normalize stitched ex-vivo: per-z-slice normalization to match per-tile approach
stitch_invivo = np.zeros_like(stitch_plain)
stitch_invivo[:, :, :, 1] = stitch_plain[:, :, :, 1]

stitch_exvivo = np.zeros_like(stitch_plain)
stitch_exvivo[:, :, :, 0] = stitch_plain[:, :, :, 0]
stitch_exvivo[:, :, :, 2] = stitch_plain[:, :, :, 2]

# Normalize stitched ex-vivo brightness per z-chunk (tiles map to z-ranges)
print("  Normalizing stitched ex-vivo brightness...")
tile_list_tmp = [t for t in TILES if t in out]
for t in tile_list_tmp:
    td = out[t]
    z_off = td['stitch_z_offset']
    n_sl = td['n_slices']
    ev_scale = target_p99 / max(tile_ev_p99[t], 1.0)
    if abs(ev_scale - 1.0) > 0.05:
        z_start = max(0, z_off)
        z_end = min(stitch_exvivo.shape[0], z_off + n_sl)
        for ch in [0, 2]:
            chunk = stitch_exvivo[z_start:z_end, :, :, ch].astype(np.float32)
            stitch_exvivo[z_start:z_end, :, :, ch] = np.clip(chunk * ev_scale, 0, 255).astype(np.uint8)
        # Also normalize the combined volume
        for ch in [0, 2]:
            chunk = stitch_combined[z_start:z_end, :, :, ch].astype(np.float32)
            stitch_combined[z_start:z_end, :, :, ch] = np.clip(chunk * ev_scale, 0, 255).astype(np.uint8)
        print(f"    z[{z_start}:{z_end}] ({t}): {ev_scale:.2f}x")

print(f"  stitch_invivo: {stitch_invivo.shape}")
print(f"  stitch_exvivo: {stitch_exvivo.shape}")

# Compute avg_um_per_dpx for z scaling
tile_list = [t for t in TILES if t in out]
avg_um_per_dpx = np.mean([(out[t]['crop_h_nd2'] * 0.65) / out[t]['cell_h']
                           for t in tile_list])

out['_stitched'] = {
    'combined': stitch_combined,     # in-vivo + ex-vivo + MERSCOPE
    'invivo': stitch_invivo,         # green channel only
    'exvivo': stitch_exvivo,         # magenta channels only
    'z': stitch_z,
    'width': vol_w,
    'height': vol_h,
    'avg_um_per_dpx': avg_um_per_dpx,
}

# ── Save ──
out_path = f'{BASE}/animation/scene5b_three_stacks_assets.pkl'
print(f"\nSaving to {out_path}...")
t0 = time.time()
with open(out_path, 'wb') as f:
    pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

fsize = os.path.getsize(out_path) / 1e6
print(f"Saved: {fsize:.0f} MB in {time.time()-t0:.0f}s")
print(f"\nKeys per tile: {sorted(out[tile_list[0]].keys())}")
print(f"Stitched keys: {sorted(out['_stitched'].keys())}")