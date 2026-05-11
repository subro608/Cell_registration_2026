"""
Farneback optical flow deformable registration for all pairs (v4).
Starts from IOU rigid, applies Farneback v3 params.
Uses v4 masks + v4 IOU transforms + PNG loading.
"""
import numpy as np
import cv2
import os
import json

BASE = '/Users/neurolab/neuroinformatics/margaret'
PNG_DIR = f'{BASE}/png_exports/registration_video'
out_dir = f'{BASE}/png_exports/z_stitch_farneback_v4'
os.makedirs(out_dir, exist_ok=True)

# Farneback v3 params (best general)
FB_DS       = 2
FB_WINSIZE  = 76
FB_BLUR_K   = 13
FB_FSK      = 65
FB_LEVELS   = 5
FB_ITER     = 15

PAIRS = [
    ('row1_1','row1_2'), ('row1_2','row1_3'),
    ('row1_3','row2_1'),
    ('row2_1','row2_2'), ('row2_2','row2_3'), ('row2_3','row2_4'), ('row2_4','row2_5'),
    ('row2_5','row3_1'),
    ('row3_1','row3_2'), ('row3_2','row3_3'), ('row3_3','row3_4'), ('row3_4','row3_5'), ('row3_5','row3_6'),
    ('row3_6','row4_1'),
    ('row4_1','row4_2'), ('row4_2','row4_3'), ('row4_3','row4_4'), ('row4_4','row4_5'), ('row4_5','row4_6'),
    ('row4_6','row5_1'),
]

ROWS_CHANGED = {'row2', 'row3', 'row4'}

def normalize_float(img):
    vals = img[img > 0]
    if len(vals) == 0:
        return np.zeros_like(img, dtype=np.float32)
    p2, p995 = np.percentile(vals, [2, 99.5])
    return np.clip((img - p2) / max(p995 - p2, 1), 0, 1).astype(np.float32)

def normalize_8bit(img):
    vals = img[img > 0]
    if len(vals) == 0:
        return np.zeros_like(img, dtype=np.uint8)
    p2, p995 = np.percentile(vals, [2, 99.5])
    return np.clip((img - p2) / max(p995 - p2, 1) * 255, 0, 255).astype(np.uint8)

def compute_ncc(a, b, mask):
    a_m = a[mask].astype(np.float64); b_m = b[mask].astype(np.float64)
    if len(a_m) == 0: return 0.0
    a_m -= a_m.mean(); b_m -= b_m.mean()
    denom = np.sqrt(np.sum(a_m**2) * np.sum(b_m**2))
    return float(np.sum(a_m * b_m) / denom) if denom > 0 else 0.0

def apply_flow(img, flow):
    h, w = img.shape[:2]
    ys, xs = np.mgrid[:h, :w].astype(np.float32)
    return cv2.remap(img, xs + flow[:,:,0], ys + flow[:,:,1], cv2.INTER_LINEAR)

def run_farneback(ref, mov_rigid, h, w):
    ref_8 = normalize_8bit(ref)
    mov_8 = normalize_8bit(mov_rigid)
    sh, sw = h // FB_DS, w // FB_DS
    ref_s = cv2.GaussianBlur(cv2.resize(ref_8, (sw, sh)), (FB_BLUR_K, FB_BLUR_K), 0)
    mov_s = cv2.GaussianBlur(cv2.resize(mov_8, (sw, sh)), (FB_BLUR_K, FB_BLUR_K), 0)
    flow_small = cv2.calcOpticalFlowFarneback(
        ref_s, mov_s, flow=None, pyr_scale=0.5, levels=FB_LEVELS,
        winsize=FB_WINSIZE, iterations=FB_ITER, poly_n=7, poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[:,:,0] = cv2.resize(flow_small[:,:,0], (w, h)) * FB_DS
    flow[:,:,1] = cv2.resize(flow_small[:,:,1], (w, h)) * FB_DS
    flow[:,:,0] = cv2.GaussianBlur(flow[:,:,0], (FB_FSK, FB_FSK), 0)
    flow[:,:,1] = cv2.GaussianBlur(flow[:,:,1], (FB_FSK, FB_FSK), 0)
    return apply_flow(mov_rigid, flow)

def make_overlay(ref_n, mov_n, mask):
    h, w = ref_n.shape
    ov = np.zeros((h, w, 3), dtype=np.float32)
    r = ref_n * mask.astype(np.float32)
    m = mov_n * mask.astype(np.float32)
    ov[:,:,1] = r; ov[:,:,0] = m; ov[:,:,2] = m
    return (ov * 255).astype(np.uint8)

# Load masks and transforms
print("Loading masks and transforms...")
masks_data = np.load(f'{BASE}/registration_video/via_masks_v4.npz')
masks = {k: masks_data[k] for k in masks_data.files}

with open(f'{BASE}/registration_video/auto_align_transforms_iou_v4.json') as f:
    iou_transforms = json.load(f)

results = {}
contact_rows = []

for idx, (key_a, key_b) in enumerate(PAIRS):
    pair_key = f'{key_a}_to_{key_b}'
    tag = ' [CROSS]' if key_a.split('_')[0] != key_b.split('_')[0] else ''
    print(f"=== [{idx+1}/20] {pair_key}{tag} ===")

    ref = cv2.imread(f'{PNG_DIR}/{key_a}/GFP_z011.png', cv2.IMREAD_GRAYSCALE)
    mov = cv2.imread(f'{PNG_DIR}/{key_b}/GFP_z000.png', cv2.IMREAD_GRAYSCALE)
    if ref is None or mov is None:
        print(f"  SKIP: image not found"); continue

    ref = ref.astype(np.float32)
    mov = mov.astype(np.float32)
    h, w = ref.shape

    mask_a = masks.get(key_a, np.ones((h,w), dtype=np.uint8))
    mask_b = masks.get(key_b, np.ones((h,w), dtype=np.uint8))

    # IOU rigid
    tfm = iou_transforms[pair_key]
    warp = np.array(tfm['warp_matrix'], dtype=np.float32)
    mov_rigid = cv2.warpAffine(mov, warp, (w,h), flags=cv2.INTER_LINEAR)
    mask_b_w  = cv2.warpAffine(mask_b, warp, (w,h), flags=cv2.INTER_NEAREST)
    mask_both = (mask_a > 0) & (mask_b_w > 0)

    ref_n       = normalize_float(ref * mask_a)
    mov_rigid_n = normalize_float(mov_rigid * mask_b_w)
    ncc_rigid   = compute_ncc(ref_n, mov_rigid_n, mask_both)

    # Farneback
    mov_fb   = run_farneback(ref * mask_a, mov_rigid * mask_b_w, h, w)
    mov_fb_n = normalize_float(mov_fb)
    ncc_fb   = compute_ncc(ref_n, mov_fb_n, mask_both)

    print(f"  Rigid NCC={ncc_rigid:.4f}  Farneback NCC={ncc_fb:.4f}  Δ=+{ncc_fb-ncc_rigid:.4f}")

    # Save overlay
    pair_dir = f'{out_dir}/{pair_key}'
    os.makedirs(pair_dir, exist_ok=True)

    ov_rigid = make_overlay(ref_n, mov_rigid_n, mask_both)
    ov_fb    = make_overlay(ref_n, mov_fb_n, mask_both)

    gap = 20
    combined = np.zeros((h, w*2+gap, 3), dtype=np.uint8)
    combined[:, :w] = cv2.cvtColor(ov_rigid, cv2.COLOR_RGB2BGR)
    combined[:, w+gap:] = cv2.cvtColor(ov_fb, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, f'IOU RIGID (NCC={ncc_rigid:.4f})',     (20, 60),        font, 2.0, (0,255,0),   3, cv2.LINE_AA)
    cv2.putText(combined, f'FARNEBACK v3 (NCC={ncc_fb:.4f})',     (w+gap+20, 60),  font, 2.0, (255,200,0), 3, cv2.LINE_AA)
    cv2.putText(combined, f'winsize={FB_WINSIZE} blur={FB_BLUR_K} fsk={FB_FSK} DS={FB_DS} levels={FB_LEVELS} iter={FB_ITER}',
                (20, h-30), font, 0.9, (180,180,180), 2, cv2.LINE_AA)
    cv2.imwrite(f'{pair_dir}/overlay.png', combined)

    results[pair_key] = {'rigid_ncc': ncc_rigid, 'farneback_ncc': ncc_fb}

    # Contact sheet thumbnail
    th = 300
    tw = int(combined.shape[1] * th / combined.shape[0])
    thumb = cv2.resize(combined, (tw, th))
    label = np.zeros((30, tw, 3), dtype=np.uint8)
    cv2.putText(label, pair_key, (5, 22), font, 0.55, (255,255,255), 1)
    contact_rows.append(np.vstack([label, thumb]))

# Contact sheet
if contact_rows:
    sheet = np.vstack(contact_rows)
    cv2.imwrite(f'{out_dir}/contact_sheet.png', sheet)

with open(f'{out_dir}/farneback_results_v4.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n=== DONE ===")
for k, v in results.items():
    print(f"  {k}: rigid={v['rigid_ncc']:.4f} farneback={v['farneback_ncc']:.4f} Δ=+{v['farneback_ncc']-v['rigid_ncc']:.4f}")
print(f"\nSaved to: {out_dir}/")
