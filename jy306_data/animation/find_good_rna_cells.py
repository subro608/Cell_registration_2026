#!/usr/bin/env python3
"""Find landmarks with bright ex-vivo signal AND high MERSCOPE gene dot density.
These are the 'good RNA' cells — magenta dots with rich transcriptomic data.
"""
import numpy as np, cv2, os, glob, pickle, json, re
import tifffile
import pandas as pd
from collections import Counter

BASE = '/Users/neurolab/neuroinformatics/margaret'
ND2_XY_UM = 0.645
CROP_ND2 = 120  # radius in nd2 pixels

EX_DIR = f'{BASE}/exvivo_merscope_combined'
PKL_DIR = f'{BASE}/merscope_exvivo '
VAROL = f'{BASE}/jy306_varol'

def build_pkl_affine(T_dict):
    B = np.eye(4)
    for step in T_dict:
        for k, v in step.items():
            if k == 'bhat':
                B = B @ np.c_[v, np.array((0,0,0,1))]
            if k == 'scale':
                B[:,:3] *= v
    R_3 = np.linalg.inv(B[:3,:3]).T
    offset_3 = -B[-1,:-1] @ np.linalg.inv(B[:3,:3])
    return np.linalg.inv(R_3), offset_3

# MERSCOPE region mapping
tile_to_ms = {}
for fname in os.listdir(EX_DIR):
    m = re.match(r'(\d+_\d+)_merscope(\d+)\.tif', fname)
    if m:
        tile_short = m.group(1)
        ms_id = int(m.group(2))
        row_tile = f'row{tile_short}'
        tile_to_ms[row_tile] = (ms_id, tile_short, f'{EX_DIR}/{fname}')

pkl_files = {}
for fname in os.listdir(PKL_DIR):
    m = re.match(r'[\d_]+_reg(\d+)_transformed.*\.pkl', fname)
    if m:
        mid = int(m.group(1))
        if mid not in pkl_files or fname > pkl_files[mid][0]:
            pkl_files[mid] = (fname, f'{PKL_DIR}/{fname}')

# All tiles with pkl transforms
TILES = ['row1_3','row2_1','row2_2','row2_3','row2_4','row2_5',
         'row3_1','row3_2','row3_3','row3_4','row3_5','row3_6',
         'row4_1','row4_2','row4_3','row4_4','row4_5']

results = []

for tile in TILES:
    pkl_tfm_path = f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz'
    if not os.path.exists(pkl_tfm_path):
        continue
    tfm = np.load(pkl_tfm_path)
    ev_nd2 = tfm['ev_nd2']
    pcd_iv = tfm['pcd_invivo_jy306']
    iv_nd2 = tfm.get('iv_nd2', None)
    if iv_nd2 is None and 'iv_nd2' in tfm:
        iv_nd2 = tfm['iv_nd2']
    N_LM = len(ev_nd2)

    # Warp errors
    if iv_nd2 is not None:
        pkl_dist_um = np.sqrt((iv_nd2[:,0]-ev_nd2[:,0])**2 + (iv_nd2[:,1]-ev_nd2[:,1])**2) * ND2_XY_UM
    else:
        pkl_dist_um = np.zeros(N_LM)

    # Check if tile has MERSCOPE data
    if tile not in tile_to_ms:
        continue
    ms_id, tile_short, ex_path = tile_to_ms[tile]
    csv_path = f'{VAROL}/region_{ms_id}_resegmentation/detected_transcripts.csv'
    mnf_path = f'{VAROL}/region_{ms_id}_resegmentation/manifest.json'
    m2m_path = f'{VAROL}/region_{ms_id}_resegmentation/micron_to_mosaic_pixel_transform.csv'

    if ms_id not in pkl_files or not all(os.path.exists(p) for p in [csv_path, mnf_path, m2m_path]):
        continue

    # Load gene dot coordinates
    try:
        exvivo_tif = tifffile.imread(ex_path).astype(np.float32)
        tif_size = exvivo_tif.shape[1]
        nd2_scale = 4200 / tif_size
        del exvivo_tif
    except:
        continue

    _, pkl_file_path = pkl_files[ms_id]
    with open(pkl_file_path, 'rb') as f:
        pdat = pickle.load(f)
    R_3_inv, offset_3 = build_pkl_affine(pdat['transformations'])

    with open(mnf_path) as f:
        mnf = json.load(f)
    W_mos = mnf['mosaic_width_pixels']
    m2m = np.loadtxt(m2m_path, delimiter=' ')
    scale_m, tx_m, ty_m = m2m[0,0], m2m[0,2], m2m[1,2]

    try:
        df = pd.read_csv(csv_path, usecols=['global_x','global_y','gene'])
        gx, gy = df.global_x.values, df.global_y.values
        x_mos = scale_m * gx + tx_m
        y_mos = scale_m * gy + ty_m
        merc_x = (W_mos - 1 - x_mos) * 0.108
        merc_y = y_mos * 0.108
        adj_y = merc_y - offset_3[1]
        adj_x = merc_x - offset_3[2]
        gene_nd2_x = (R_3_inv[2,1]*adj_y + R_3_inv[2,2]*adj_x) * nd2_scale
        gene_nd2_y = (R_3_inv[1,1]*adj_y + R_3_inv[1,2]*adj_x) * nd2_scale

        valid = (gene_nd2_x >= 0) & (gene_nd2_x < 4200) & \
                (gene_nd2_y >= 0) & (gene_nd2_y < 4200)
        gene_nd2_x = gene_nd2_x[valid]
        gene_nd2_y = gene_nd2_y[valid]
        gene_names_all = df.gene.values[valid]
        n_unique_genes = len(set(gene_names_all))
    except Exception as e:
        print(f"  {tile}: gene load error: {e}")
        continue

    # Load nd2 slices for brightness check
    tile_dir = f'{BASE}/png_exports/registration_video/{tile}'

    for i in range(N_LM):
        err = pkl_dist_um[i]
        if err > 5.0:  # skip high error
            continue

        cx = int(round(ev_nd2[i, 0]))
        cy = int(round(ev_nd2[i, 1]))
        z_nd2 = int(round(ev_nd2[i, 2]))
        z_iv = int(round(np.clip(pcd_iv[i, 0], 0, 15)))

        # Count gene dots in patch
        dx = gene_nd2_x - cx
        dy = gene_nd2_y - cy
        in_patch = (np.abs(dx) < CROP_ND2) & (np.abs(dy) < CROP_ND2)
        n_dots = int(in_patch.sum())
        if n_dots < 50:  # need reasonable gene coverage
            continue

        # Count unique genes in patch
        patch_genes = gene_names_all[in_patch]
        n_genes = len(set(patch_genes))

        # Check ex-vivo brightness
        nd2_path = f'{tile_dir}/GFP_z{z_nd2:03d}.png'
        if not os.path.exists(nd2_path):
            continue
        nd2_sl = cv2.imread(nd2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        # Brightness in patch
        y0 = max(0, cy - 30); y1 = min(4200, cy + 30)
        x0 = max(0, cx - 30); x1 = min(4200, cx + 30)
        patch_mean = nd2_sl[y0:y1, x0:x1].mean()

        results.append({
            'tile': tile, 'lm': i, 'z_iv': z_iv, 'z_nd2': z_nd2,
            'err_um': err, 'n_dots': n_dots, 'n_genes': n_genes,
            'ev_brightness': patch_mean,
            'cx': cx, 'cy': cy,
        })

    print(f"  {tile}: {N_LM} landmarks, found {sum(1 for r in results if r['tile']==tile)} good ones")

# Sort by combined score: high brightness + high gene dots + low error
print(f"\nTotal candidates: {len(results)}")
print(f"\n{'='*100}")
print(f"{'Tile':>8} {'LM':>4} {'z_iv':>5} {'z_nd2':>5} {'Err':>6} {'Dots':>6} {'Genes':>6} {'EV_bright':>10}")
print(f"{'='*100}")

# Rank by: high dots, high brightness, low error
for r in results:
    r['score'] = r['n_dots'] * 0.5 + r['ev_brightness'] * 2.0 - r['err_um'] * 100

results.sort(key=lambda x: -x['score'])

# Print top 20
for r in results[:30]:
    print(f"{r['tile']:>8} #{r['lm']:<3} z_iv={r['z_iv']:<3} z_nd2={r['z_nd2']:<3} "
          f"err={r['err_um']:.1f}µm  dots={r['n_dots']:<6} genes={r['n_genes']:<4} "
          f"bright={r['ev_brightness']:.0f}  score={r['score']:.0f}")

# Also show best per tile (to ensure diversity)
print(f"\n{'='*100}")
print("BEST PER TILE:")
print(f"{'='*100}")
seen = set()
for r in results:
    if r['tile'] not in seen:
        seen.add(r['tile'])
        print(f"{r['tile']:>8} #{r['lm']:<3} z_iv={r['z_iv']:<3} z_nd2={r['z_nd2']:<3} "
              f"err={r['err_um']:.1f}µm  dots={r['n_dots']:<6} genes={r['n_genes']:<4} "
              f"bright={r['ev_brightness']:.0f}  score={r['score']:.0f}")
