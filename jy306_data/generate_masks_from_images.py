"""
Auto-generate tissue masks from GFP MIP images using thresholding + morphology.
Reads png_exports/registration_video/{tile}/GFP_MIP.png -> registration_video/via_masks_v4.npz
"""

import numpy as np
import cv2
import os

BASE = '/Users/neurolab/neuroinformatics/margaret'
IMG_DIR = f'{BASE}/png_exports/registration_video'
OUT_MASKS = f'{BASE}/registration_video/via_masks_v4.npz'
MASK_SIZE = 4200

TILE_ORDER = [
    'row1_1', 'row1_2', 'row1_3',
    'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
    'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
    'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
    'row5_1',
]

def make_mask_from_mip(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {img_path}")

    # Resize to 4200x4200 if needed
    if img.shape != (MASK_SIZE, MASK_SIZE):
        img = cv2.resize(img, (MASK_SIZE, MASK_SIZE), interpolation=cv2.INTER_LINEAR)

    # Otsu threshold to find tissue
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup: close small holes, remove small islands
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  kernel_open)

    # Keep only the largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        opened = (labels == largest).astype(np.uint8)
    else:
        opened = (opened > 0).astype(np.uint8)

    return opened

masks = {}
for key in TILE_ORDER:
    img_path = f'{IMG_DIR}/{key}/GFP_MIP.png'
    print(f"  {key}...", end=" ", flush=True)
    try:
        mask = make_mask_from_mip(img_path)
        masks[key] = mask
        coverage = mask.mean() * 100
        print(f"done ({coverage:.1f}% coverage)")
    except Exception as e:
        print(f"FAILED: {e}")

np.savez_compressed(OUT_MASKS, **masks)
print(f"\nSaved {len(masks)} masks -> {OUT_MASKS}")
