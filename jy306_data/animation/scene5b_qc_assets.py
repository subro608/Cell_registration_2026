"""
QC script: visualize overlay, MERSCOPE dots, and masks from scene5b assets.
Generates one image per tile showing 3 panels side by side:
  1. Overlay (magenta ex-vivo + green in-vivo) — middle z-slice
  2. MERSCOPE gene dots
  3. Overlay + dots composited
"""
import numpy as np, cv2, os, pickle

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_DIR = f'{BASE}/animation/scene5b_qc'
os.makedirs(OUT_DIR, exist_ok=True)

with open(f'{BASE}/animation/scene5b_assets_v3.pkl', 'rb') as f:
    assets = pickle.load(f)

for tile, a in assets.items():
    dense = a['dense']
    mid = len(dense) // 2
    overlay = dense[mid]  # middle z-slice

    h, w = overlay.shape[:2]
    ms = a.get('merscope')

    # Panel 1: overlay
    p1 = overlay.copy()

    # Panel 2: MERSCOPE dots (or black if none)
    if ms is not None:
        p2 = ms.copy()
    else:
        p2 = np.zeros_like(p1)

    # Panel 3: composite overlay + dots
    p3 = overlay.copy()
    if ms is not None:
        ms_f = ms.astype(np.float32) / 255.0
        ms_mask = np.max(ms_f, axis=2) > 0.05
        ms_mask3 = np.stack([ms_mask]*3, axis=-1)
        p3_f = p3.astype(np.float32) / 255.0
        p3_f = np.where(ms_mask3, p3_f * 0.4 + ms_f * 0.6, p3_f)
        p3 = np.clip(p3_f * 255, 0, 255).astype(np.uint8)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for img, label in [(p1, 'OVERLAY'), (p2, 'GENE DOTS'), (p3, 'COMPOSITE')]:
        cv2.putText(img, label, (5, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Combine side by side
    gap = np.zeros((h, 4, 3), dtype=np.uint8)
    combined = np.hstack([p1, gap, p2, gap, p3])

    # Add tile name at top
    cv2.putText(combined, tile.upper(), (combined.shape[1]//2 - 40, 15),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    out_path = f'{OUT_DIR}/{tile}_qc.png'
    cv2.imwrite(out_path, combined)
    print(f'  {tile}: {combined.shape}')

print(f'\nSaved QC images to {OUT_DIR}/')
