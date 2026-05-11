"""
Render MERSCOPE gene dots with rainbow colors per gene.
Uses per-region micron_to_mosaic + manifest + PKL affine (same as build_landmark_genedot_patches.py).
Updates scene5b_assets_v3.pkl with clean rainbow gene dot images.
"""
import numpy as np, cv2, os, pickle, json
import pandas as pd
from collections import Counter

BASE = '/Users/neurolab/neuroinformatics/margaret'
ND2_SIZE = 4200
VAROL = f'{BASE}/jy306_varol'
PKL_DIR = f'{BASE}/merscope_exvivo '

# Tile → MERSCOPE region mapping
TILE_TO_REGION = {
    'row1_3': 23, 'row2_1': 17, 'row2_2': 18, 'row2_3': 19,
    'row2_4': 20, 'row2_5': 21, 'row3_1': 16, 'row3_2': 15,
    'row3_3': 14, 'row3_4': 13, 'row3_5': 12, 'row3_6': 11,
    'row4_1': 5, 'row4_2': 6, 'row4_3': 7, 'row4_4': 8,
    'row4_5': 9, 'row4_6': 10, 'row5_1': 4,
}


def make_rainbow_palette(n):
    """Generate n maximally distinct bright colours cycling through HSV."""
    colors = []
    for i in range(n):
        h = int(180 * i / n)
        s = 200 + int(55 * ((i % 3) / 2))
        v = 200 + int(55 * ((i % 5) / 4))
        bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(c) for c in bgr))
    return colors

N_GENE_COLORS = 550
GENE_PALETTE = make_rainbow_palette(N_GENE_COLORS)


def build_pkl_affine(T_dict):
    """Build B matrix from PKL transform chain, return R_3_inv and offset_3."""
    B = np.eye(4)
    for step in T_dict:
        for k, v in step.items():
            if k == 'bhat':
                B = B @ np.c_[v, np.array((0, 0, 0, 1))]
            if k == 'scale':
                B[:, :3] *= v
    R_3 = np.linalg.inv(B[:3, :3]).T
    offset_3 = -B[-1, :-1] @ np.linalg.inv(B[:3, :3])
    R_3_inv = np.linalg.inv(R_3)
    return R_3_inv, offset_3


# Load assets
with open(f'{BASE}/animation/scene5b_assets_v3.pkl', 'rb') as f:
    assets = pickle.load(f)

# Load masks
via_masks = np.load(f'{BASE}/registration_video/via_masks_v4.npz')

# PKL files per region
pkl_files = {}
for fname in os.listdir(PKL_DIR):
    if fname.endswith('.pkl'):
        import re
        m = re.match(r'[\d_]+_reg(\d+)_transformed.*\.pkl', fname)
        if m:
            mid = int(m.group(1))
            if mid not in pkl_files or fname > pkl_files[mid]:
                pkl_files[mid] = fname

for tile, region_id in TILE_TO_REGION.items():
    if tile not in assets:
        continue
    print(f"\n  {tile} (region {region_id})...")

    # Per-region paths
    csv_path = f'{VAROL}/region_{region_id}_resegmentation/detected_transcripts.csv'
    mnf_path = f'{VAROL}/region_{region_id}_resegmentation/manifest.json'
    m2m_path = f'{VAROL}/region_{region_id}_resegmentation/micron_to_mosaic_pixel_transform.csv'

    if not all(os.path.exists(p) for p in [csv_path, mnf_path, m2m_path]):
        print(f"    Missing MERSCOPE data files")
        continue
    if region_id not in pkl_files:
        print(f"    No PKL found")
        continue

    # Load per-region transforms
    m2m = np.loadtxt(m2m_path, delimiter=' ')
    scale_m, tx_m, ty_m = m2m[0, 0], m2m[0, 2], m2m[1, 2]

    with open(mnf_path) as f:
        mnf = json.load(f)
    W_mosaic = mnf['mosaic_width_pixels']

    # PKL affine
    pkl_path = f'{PKL_DIR}/{pkl_files[region_id]}'
    with open(pkl_path, 'rb') as f:
        pdat = pickle.load(f)
    R_3_inv, offset_3 = build_pkl_affine(pdat['transformations'])
    tif_size = pdat['transformed'].shape[-1]
    nd2_scale = ND2_SIZE / tif_size

    print(f"    scale_m={scale_m:.4f}, W_mosaic={W_mosaic}, tif_size={tif_size}")

    # Load transcripts
    df = pd.read_csv(csv_path, usecols=['global_x', 'global_y', 'gene'])
    gx, gy = df.global_x.values, df.global_y.values

    # Transform: global microns → nd2 pixels (per-region, same as build_landmark_genedot_patches.py)
    x_mos = scale_m * gx + tx_m
    y_mos = scale_m * gy + ty_m
    merc_x = (W_mosaic - 1 - x_mos) * 0.108
    merc_y = y_mos * 0.108
    adj_y = merc_y - offset_3[1]
    adj_x = merc_x - offset_3[2]
    nd2_x = (R_3_inv[2, 1] * adj_y + R_3_inv[2, 2] * adj_x) * nd2_scale
    nd2_y = (R_3_inv[1, 1] * adj_y + R_3_inv[1, 2] * adj_x) * nd2_scale

    # Get crop bounds
    reg_pkl = np.load(f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz')
    ev = reg_pkl['ev_nd2']
    margin_nd2 = 350
    crop_x0 = max(0, int(ev[:, 0].min() - margin_nd2))
    crop_y0 = max(0, int(ev[:, 1].min() - margin_nd2))
    crop_x1 = min(4200, int(ev[:, 0].max() + margin_nd2))
    crop_y1 = min(4200, int(ev[:, 1].max() + margin_nd2))

    cell_w = assets[tile]['cell_w']
    cell_h = assets[tile]['cell_h']

    # Filter to crop region
    in_crop = (nd2_x >= crop_x0) & (nd2_x < crop_x1) & (nd2_y >= crop_y0) & (nd2_y < crop_y1)
    xc = nd2_x[in_crop] - crop_x0
    yc = nd2_y[in_crop] - crop_y0
    gc = df['gene'].values[in_crop]

    n_in = in_crop.sum()
    n_genes = len(set(gc))
    print(f"    {n_in} transcripts in crop, {n_genes} genes")

    # Assign rainbow colors sorted by frequency
    gene_counts = Counter(gc)
    all_genes_sorted = [g for g, _ in gene_counts.most_common()]
    gene_to_color = {g: GENE_PALETTE[i % N_GENE_COLORS] for i, g in enumerate(all_genes_sorted)}

    # Draw on full-res crop canvas
    crop_h = crop_y1 - crop_y0
    crop_w = crop_x1 - crop_x0
    canvas = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)

    for i in range(len(xc)):
        x_px = int(xc[i])
        y_px = int(yc[i])
        if 0 <= x_px < crop_w and 0 <= y_px < crop_h:
            canvas[y_px, x_px] = gene_to_color[gc[i]]

    # Apply tissue mask
    tile_mask = via_masks[tile] if tile in via_masks else np.ones((4200, 4200), dtype=np.uint8)
    mask_crop = tile_mask[crop_y0:min(crop_y1, 4200), crop_x0:min(crop_x1, 4200)]
    mc = mask_crop[:crop_h, :crop_w]
    canvas = canvas * np.stack([mc] * 3, axis=-1)

    # Downscale
    dots_small = cv2.resize(canvas, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

    # Boost brightness — single pixels get averaged away by 10x downscale
    dots_f = dots_small.astype(np.float32)
    dot_mask = np.max(dots_f, axis=2) > 1  # any non-zero pixel
    dots_f[dot_mask] = np.clip(dots_f[dot_mask] * 2.0, 0, 255)  # 2x boost
    dots_small = dots_f.astype(np.uint8)

    # Apply display-size mask
    mask_small = cv2.resize(mc.astype(np.uint8), (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
    dots_small = dots_small * np.stack([mask_small] * 3, axis=-1)

    assets[tile]['merscope'] = dots_small
    print(f"    Done: {dots_small.shape}, {n_genes} genes rainbow")

# Save
with open(f'{BASE}/animation/scene5b_assets_v3.pkl', 'wb') as f:
    pickle.dump(assets, f)
print(f"\nSaved ({os.path.getsize(f'{BASE}/animation/scene5b_assets_v3.pkl') / 1e6:.0f} MB)")
