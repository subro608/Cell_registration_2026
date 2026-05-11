---
name: Mask Generation Method
description: Always generate masks from VIA CSV annotations, never from auto-thresholding images
type: feedback
---

Always generate masks from `via_annotations/via_export_csv.csv` using `generate_masks_from_csv.py`. Never use auto-thresholding (Otsu) on GFP MIP images to generate masks.

**Why:** User explicitly rejected `generate_masks_from_images.py` — the auto-threshold masks were wrong. The VIA polyline annotations represent careful manual tissue boundary decisions that auto-thresholding cannot reproduce.

**How to apply:** When regenerating `via_masks_v4.npz`, always run `generate_masks_from_csv.py` first. If asked to regenerate masks, default to the CSV method. The `generate_masks_from_images.py` script exists but should not be used.
