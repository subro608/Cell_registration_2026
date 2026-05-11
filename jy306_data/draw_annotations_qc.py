"""
Draw VIA polyline annotations from CSV onto all GFP images (MIP + z000-z011) for visual QC.
Outputs to via_annotations/annotated_qc_new/{key}/
"""
import numpy as np
import cv2
import csv
import json
import re
import os
import glob

BASE = '/Users/neurolab/neuroinformatics/margaret'
CSV_PATH = f'{BASE}/via_annotations/via_export_csv.csv'
PNG_DIR = f'{BASE}/png_exports/registration_video'
OUT_DIR = f'{BASE}/via_annotations/annotated_qc_new'

def filename_to_key(fname):
    m = re.match(r'(row\d+_\d+)_', fname)
    return m.group(1) if m else None

def draw_regions(img, regions):
    for shape, pts in regions:
        arr = np.array(pts, dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [arr], (0, 100, 0))
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
        cv2.polylines(img, [arr], isClosed=True, color=(0, 255, 0), thickness=8)
        cv2.circle(img, tuple(arr[0]),  20, (0, 0, 255), -1)  # red = start
        cv2.circle(img, tuple(arr[-1]), 20, (255, 0, 0), -1)  # blue = end
    return img

# Parse CSV
regions_per_tile = {}
with open(CSV_PATH, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname = row['filename']
        key = filename_to_key(fname)
        if key is None:
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
        regions_per_tile.setdefault(key, []).append((shape, list(zip(xs, ys))))

print(f"Found annotations for: {sorted(regions_per_tile.keys())}\n")

for key, regions in sorted(regions_per_tile.items()):
    tile_dir = f'{PNG_DIR}/{key}'
    out_tile_dir = f'{OUT_DIR}/{key}'
    os.makedirs(out_tile_dir, exist_ok=True)

    # All GFP images: MIP + z000-z011
    images = sorted(glob.glob(f'{tile_dir}/GFP_*.png'))
    print(f"  {key}: {len(images)} images...", end=' ', flush=True)

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = draw_regions(img, regions)
        fname = os.path.basename(img_path).replace('.png', '_annotated.png')
        cv2.imwrite(f'{out_tile_dir}/{fname}', img)

    print("done")

print(f"\nSaved to {OUT_DIR}/")
