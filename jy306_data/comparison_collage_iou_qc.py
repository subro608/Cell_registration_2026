"""
2-column comparison collage — IOU rigid QC for all 20 pairs.
Col 1: Raw overlap + annotation contours (centroid-aligned, no masking)
Col 2: IOU Rigid
Saves one PNG per pair + final high-res contact sheet.
"""
import numpy as np
import cv2
import os
import json

BASE = '/Users/neurolab/neuroinformatics/margaret'
PNG_DIR = f'{BASE}/png_exports/registration_video'
OUT_DIR = f'{BASE}/png_exports/z_stitch_comparison_iou_qc'
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

METHODS  = ['Raw + Annotation', 'IOU Rigid']
HDR_COLS = [(220,220,220), (200,255,200)]
CELL_W, CELL_H = 1000, 1000
LABEL_H, GAP, HDR_H = 50, 8, 60
MAGENTA_BOOST = 1.0

def normalize_float(img):
    vals = img[img > 0]
    if len(vals) == 0: return np.zeros_like(img, dtype=np.float32)
    p2, p995 = np.percentile(vals, [2, 99.5])
    return np.clip((img - p2) / max(p995 - p2, 1), 0, 1).astype(np.float32)

def compute_ncc(a, b, mask):
    a_m = a[mask].astype(np.float64); b_m = b[mask].astype(np.float64)
    if len(a_m) == 0: return 0.0
    a_m -= a_m.mean(); b_m -= b_m.mean()
    denom = np.sqrt(np.sum(a_m**2) * np.sum(b_m**2))
    return float(np.sum(a_m * b_m) / denom) if denom > 0 else 0.0

def make_overlay_masked(ref_n, mov_n, mask):
    h, w = ref_n.shape
    ov = np.zeros((h, w, 3), dtype=np.float32)
    r = ref_n * mask.astype(np.float32)
    m = np.clip(mov_n * mask.astype(np.float32) * MAGENTA_BOOST, 0, 1)
    ov[:,:,1] = r; ov[:,:,0] = m; ov[:,:,2] = m
    return (ov * 255).astype(np.uint8)

def make_overlay_raw(ref_raw, mov_raw):
    ref_n = normalize_float(ref_raw)
    m = np.clip(normalize_float(mov_raw) * MAGENTA_BOOST, 0, 1)
    h, w = ref_n.shape
    ov = np.zeros((h, w, 3), dtype=np.float32)
    ov[:,:,1] = ref_n; ov[:,:,0] = m; ov[:,:,2] = m
    return (ov * 255).astype(np.uint8)

def draw_mask_contour(img, mask, color, thickness=12):
    cnts, _ = cv2.findContours((mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, color, thickness)

print("Loading masks and transforms...")
masks_data = np.load(f'{BASE}/registration_video/via_masks_v4.npz')
masks = {k: masks_data[k] for k in masks_data.files}
with open(f'{BASE}/registration_video/auto_align_transforms_iou_v4.json') as f:
    iou_transforms = json.load(f)

n_methods = len(METHODS)
sheet_w = n_methods * (CELL_W + GAP) + GAP
contact_rows = []
all_results = {}

for idx, (key_a, key_b) in enumerate(PAIRS):
    pair_key = f'{key_a}_to_{key_b}'
    tag = ' [CROSS]' if key_a.split('_')[0] != key_b.split('_')[0] else ''
    print(f"\n=== [{idx+1}/20] {pair_key}{tag} ===")

    ref_raw = cv2.imread(f'{PNG_DIR}/{key_a}/GFP_z011.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    mov_raw = cv2.imread(f'{PNG_DIR}/{key_b}/GFP_z000.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    h, w = ref_raw.shape

    mask_a = masks.get(key_a, np.ones((h,w), dtype=np.uint8))
    mask_b = masks.get(key_b, np.ones((h,w), dtype=np.uint8))

    tfm = iou_transforms[pair_key]
    warp = np.array(tfm['warp_matrix'], dtype=np.float32)
    mov_rigid  = cv2.warpAffine(mov_raw, warp, (w,h), flags=cv2.INTER_LINEAR)
    mask_b_w   = cv2.warpAffine(mask_b, warp, (w,h), flags=cv2.INTER_NEAREST)
    mask_both  = (mask_a > 0) & (mask_b_w > 0)

    ref_n       = normalize_float(ref_raw * mask_a)
    mov_rigid_n = normalize_float(mov_rigid * mask_b_w)
    ncc_rigid   = compute_ncc(ref_n, mov_rigid_n, mask_both)
    print(f"  IOU rigid: NCC={ncc_rigid:.4f}")

    # Raw panel (centroid shift)
    cx_a = float(np.where(mask_a>0)[1].mean()); cy_a = float(np.where(mask_a>0)[0].mean())
    cx_b = float(np.where(mask_b>0)[1].mean()); cy_b = float(np.where(mask_b>0)[0].mean())
    M_c = np.float32([[1,0,cx_a-cx_b],[0,1,cy_a-cy_b]])
    mov_cent    = cv2.warpAffine(mov_raw, M_c, (w,h))
    mask_b_cent = cv2.warpAffine(mask_b, M_c, (w,h), flags=cv2.INTER_NEAREST)
    ov_raw = make_overlay_raw(ref_raw, mov_cent)
    draw_mask_contour(ov_raw, mask_a,      (0,255,0),   12)
    draw_mask_contour(ov_raw, mask_b_cent, (255,0,255), 12)

    panels = [
        (ov_raw,                                                None),
        (make_overlay_masked(ref_n, mov_rigid_n, mask_both),    ncc_rigid),
    ]

    sheet_h = HDR_H + GAP + CELL_H + LABEL_H + GAP
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for mi, (mname, col) in enumerate(zip(METHODS, HDR_COLS)):
        x0 = GAP + mi * (CELL_W + GAP)
        cv2.putText(sheet, mname, (x0+8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2, cv2.LINE_AA)

    y0 = HDR_H + GAP
    for mi, (ov, ncc) in enumerate(panels):
        x0 = GAP + mi * (CELL_W + GAP)
        thumb = cv2.resize(cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), (CELL_W, CELL_H))
        sheet[y0:y0+CELL_H, x0:x0+CELL_W] = thumb
        if ncc is not None:
            cv2.putText(sheet, f'NCC={ncc:.4f}', (x0+5, y0+CELL_H-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,100), 1, cv2.LINE_AA)

    cv2.putText(sheet, pair_key, (GAP+5, y0+CELL_H+35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2, cv2.LINE_AA)

    cv2.imwrite(f'{OUT_DIR}/{pair_key}.png', sheet)
    print(f"  Saved: {pair_key}.png")

    label_bar = np.zeros((35, sheet_w, 3), dtype=np.uint8)
    cv2.putText(label_bar, f'[{idx+1}/20] {pair_key}{tag}', (8,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)
    contact_rows.append(np.vstack([label_bar, sheet]))
    all_results[pair_key] = {'ncc_rigid': ncc_rigid}

print("\nBuilding contact sheet...")
contact_sheet = np.vstack(contact_rows)
cv2.imwrite(f'{OUT_DIR}/ALL_PAIRS_contact_sheet.png', contact_sheet)
print(f"Contact sheet: {contact_sheet.shape[1]}x{contact_sheet.shape[0]}px")

with open(f'{OUT_DIR}/results_iou_qc.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nAll saved to: {OUT_DIR}/")
