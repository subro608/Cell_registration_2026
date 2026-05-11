"""
Convert VIA polyline annotations (CSV) to binary masks NPZ.
Reads via_annotations/via_export_csv.csv -> registration_video/via_masks_v4.npz

Keeps existing masks from via_masks_v2.npz for tiles not present in new CSV,
so only updated tiles get replaced.
"""

import numpy as np
import cv2
import csv
import json
import os
import re

BASE = '/Users/neurolab/neuroinformatics/margaret'
CSV_PATH = f'{BASE}/via_annotations/via_export_csv.csv'
OLD_MASKS = f'{BASE}/registration_video/via_masks_v2.npz'
OUT_MASKS = f'{BASE}/registration_video/via_masks_v4.npz'
MASK_SIZE = 4200

def filename_to_key(fname):
    # e.g. row2_3_GFP_MIP.png -> row2_3
    m = re.match(r'(row\d+_\d+)_', fname)
    return m.group(1) if m else None

# Load existing masks as base
print("Loading existing masks from v2...")
old_data = np.load(OLD_MASKS)
masks = {k: old_data[k] for k in old_data.files}
print(f"  Loaded {len(masks)} existing masks")

# Parse CSV
print(f"\nParsing {CSV_PATH}...")
updated = {}

with open(CSV_PATH, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname = row['filename']
        key = filename_to_key(fname)
        if key is None:
            print(f"  Skipping unrecognised filename: {fname}")
            continue

        attrs = json.loads(row['region_shape_attributes'])
        shape = attrs.get('name', '')

        if shape in ('polygon', 'polyline'):
            xs = attrs['all_points_x']
            ys = attrs['all_points_y']
        elif shape == 'rect':
            x, y, w, h = attrs['x'], attrs['y'], attrs['width'], attrs['height']
            xs = [x, x+w, x+w, x]
            ys = [y, y, y+h, y+h]
        else:
            print(f"  Unsupported shape '{shape}' for {key}, skipping")
            continue

        pts = np.array(list(zip(xs, ys)), dtype=np.int32)

        if key not in updated:
            updated[key] = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.uint8)

        cv2.fillPoly(updated[key], [pts], 1)

print(f"  Found annotations for: {sorted(updated.keys())}")

# Merge: updated tiles override old masks
for key, mask in updated.items():
    if key in masks:
        print(f"  Updating {key} (was in v2)")
    else:
        print(f"  Adding new {key}")
    masks[key] = mask

# Save
np.savez_compressed(OUT_MASKS, **masks)
print(f"\nSaved {len(masks)} masks -> {OUT_MASKS}")

# Quick check
for key in sorted(updated.keys()):
    m = masks[key]
    print(f"  {key}: {m.sum()} foreground pixels ({m.mean()*100:.1f}% coverage)")
