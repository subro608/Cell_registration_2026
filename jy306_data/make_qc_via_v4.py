"""
Regenerate z_stitch_qc_via with v4 masks.
3-panel per pair: raw green/magenta overlay | overlay + mask contours | mask-only
Centroid-based translation only (no rotation).
"""
import numpy as np
import cv2
import os

BASE = '/Users/neurolab/neuroinformatics/margaret'
PNG_DIR = f'{BASE}/png_exports/registration_video'
OUT_DIR = f'{BASE}/png_exports/z_stitch_qc_via_v4'
os.makedirs(OUT_DIR, exist_ok=True)

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

THUMB_W, THUMB_H = 1050, 1050
GAP = 8

print("Loading v4 masks...")
masks_data = np.load(f'{BASE}/registration_video/via_masks_v4.npz')
masks = {k: masks_data[k] for k in masks_data.files}

def load_img(key, fname):
    p = f'{PNG_DIR}/{key}/{fname}'
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img.astype(np.float32)

def normalize(img):
    vals = img[img > 0]
    if len(vals) == 0:
        return np.zeros_like(img, dtype=np.uint8)
    p2, p99 = np.percentile(vals, [2, 99.5])
    return np.clip((img - p2) / max(p99 - p2, 1) * 255, 0, 255).astype(np.uint8)

def centroid(mask):
    ys, xs = np.where(mask > 0)
    return float(xs.mean()), float(ys.mean())

def shift_image(img, tx, ty):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

def make_panel_raw(ref8, mov8_shifted):
    h, w = ref8.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:,:,1] = ref8        # green = ref (last slice of A)
    out[:,:,0] = mov8_shifted  # blue+red = magenta = mov (first slice of B)
    out[:,:,2] = mov8_shifted
    return out

def make_panel_contours(ref8, mov8_shifted, mask_a, mask_b_shifted):
    panel = make_panel_raw(ref8, mov8_shifted)
    cnt_a, _ = cv2.findContours((mask_a * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_b, _ = cv2.findContours((mask_b_shifted * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, cnt_a, -1, (0, 255, 0), 6)      # green = ref mask
    cv2.drawContours(panel, cnt_b, -1, (255, 0, 255), 6)    # magenta = mov mask
    return panel

def make_panel_masks(mask_a, mask_b_shifted):
    h, w = mask_a.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    overlap = (mask_a > 0) & (mask_b_shifted > 0)
    only_a  = (mask_a > 0) & ~overlap
    only_b  = (mask_b_shifted > 0) & ~overlap
    out[overlap] = (180, 180, 180)
    out[only_a]  = (0, 200, 0)
    out[only_b]  = (200, 0, 200)
    cnt_a, _ = cv2.findContours((mask_a * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_b, _ = cv2.findContours((mask_b_shifted * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnt_a, -1, (0, 255, 0), 6)
    cv2.drawContours(out, cnt_b, -1, (255, 0, 255), 6)
    return out

contact_thumbs = []

for key_a, key_b in PAIRS:
    pair_key = f'{key_a}_to_{key_b}'
    print(f"  {pair_key}...", end=' ', flush=True)

    try:
        ref_raw = load_img(key_a, 'GFP_z011.png')
        mov_raw = load_img(key_b, 'GFP_z000.png')
    except FileNotFoundError as e:
        print(f"SKIP: {e}")
        continue

    mask_a = masks.get(key_a, np.ones(ref_raw.shape, dtype=np.uint8))
    mask_b = masks.get(key_b, np.ones(mov_raw.shape, dtype=np.uint8))

    # Centroid-based shift
    cx_a, cy_a = centroid(mask_a)
    cx_b, cy_b = centroid(mask_b)
    tx, ty = cx_a - cx_b, cy_a - cy_b

    mov_shifted   = shift_image(mov_raw, tx, ty)
    mask_b_shifted = shift_image(mask_b.astype(np.float32), tx, ty)
    mask_b_shifted = (mask_b_shifted > 0.5).astype(np.uint8)

    ref8 = normalize(ref_raw * mask_a)
    mov8 = normalize(mov_shifted * mask_b_shifted)

    h, w = ref8.shape
    p1 = make_panel_raw(ref8, mov8)
    p2 = make_panel_contours(ref8, mov8, mask_a, mask_b_shifted)
    p3 = make_panel_masks(mask_a, mask_b_shifted)

    # Resize panels to THUMB size
    t1 = cv2.resize(p1, (THUMB_W, THUMB_H))
    t2 = cv2.resize(p2, (THUMB_W, THUMB_H))
    t3 = cv2.resize(p3, (THUMB_W, THUMB_H))

    divider = np.zeros((THUMB_H, GAP, 3), dtype=np.uint8)
    row_img = np.hstack([t1, divider, t2, divider, t3])

    out_path = f'{OUT_DIR}/{pair_key}.png'
    cv2.imwrite(out_path, row_img)

    # Label for contact sheet
    label = np.zeros((40, row_img.shape[1], 3), dtype=np.uint8)
    cv2.putText(label, pair_key, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    contact_thumbs.append(np.vstack([label, row_img]))
    print("done")

# Contact sheet
if contact_thumbs:
    sheet = np.vstack(contact_thumbs)
    cv2.imwrite(f'{OUT_DIR}/all_pairs_contact_sheet.png', sheet)
    print(f"\nContact sheet saved.")

print(f"\nAll saved to: {OUT_DIR}/")
