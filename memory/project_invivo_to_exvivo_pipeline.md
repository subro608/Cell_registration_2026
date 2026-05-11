---
name: In-Vivo to Ex-Vivo Cell Mapping Pipeline
description: 4-step plan to reverse pkl transforms, map in-vivo cell locations to ex-vivo stitched stack, do coarse+deformable registration, then animate
type: project
---

**Goal:** Map cell locations from JY306 in-vivo space → ex-vivo stitched stack, register them finely, and animate the transformation.

**Why:** Complete the multimodal cell-matching pipeline for the Nature Science submission.

**How to apply:** Work through steps 1→4 in order.

---

## Step 1 — Reverse pkl transform ✓ DONE
Map a cell position (z,y,x) in JY306 in-vivo space back to its position in the ex-vivo tile (1734 canvas) space by inverting the 14-stage pkl pipeline in reverse order:
- `scale`: p = p / val
- `bhat` (4×3 affine): p = (p - t) @ inv(R)
- `vec_field_total`: p = p - vf[p]  (approximate inverse, valid here)
Apply stages in **reverse order** (stage 13 → 0).

**Verified:**
- Round-trip error (pcd_invivo → inv → fwd → pcd_invivo): mean=0.98px, max=2.17px ✓
- Inverse landmarks land inside ex-vivo tissue on native MIP ✓
- pcd_invivo and pcd_exvivo give nearly identical inverse results (mean separation 1.72px) — expected because they are only 1.86px apart in JY306 input space; confirms code is correct
- Script: `inverse_pkl_transform.py`; verification: `verify_inverse_landmarks.py`
- To scale from 1734 canvas → 4200px native MIP: multiply coords by 4200/1734 ≈ 2.422

## Step 2 — Coarse rigid registration (IN PROGRESS)

### What was tried (row2_1 as test case):
- **Stored affine** `affine_nd2_to_exvivo.npy` (2×3, yx convention): maps nd2 4200px → JY306 (658×629). Inverse gives best overlay of JY306→nd2 (used in `row21_jy306_4pt_overlay.py`)
- **4 manual landmarks** `registration_video/landmarks.npz`: src_points (nd2 yx), tgt_points (JY306 yx). Only 4 pts clustered at tissue edge — too few and poorly spread to recompute affine reliably. Stored affine is better.
- **Cell-density NCC**: NCC=0.43, translation y=756, x=352 (1µm iso space)
- **Raw MIP NCC**: NCC=0.295 — worse
- **Backward mapping**: create grid in 1734px input space → forward-transform → sample JY306. Correct approach but JY306 offset from nd2 because in-vivo and ex-vivo look different (cell bodies vs axons/dendrites).
- **Farneback**: mean flow 2px — failed (no matching intensity patterns across modalities)
- **Elastix B-spline + MI**: attempted but not completed

### Key insight on pkl coordinate spaces:
- `pcd_invivo`, `pcd_exvivo` in pkl are BOTH in JY306 space (0-657, 0-628)
- `transformed` array (3,16,1704,1704): crop [:H,:W] (658×629) maps 1:1 to JY306 coords — pcd coords plot directly on it with no scaling
- Inverse transform gives coords in 1734px canvas space; scale to nd2 4200px: ×(4200/1734)=×2.422
- Backward image mapping: create 1734×1734 grid → forward transform → JY306 coords (y:500-1100, x:300-850 maps to valid JY306)
- INPUT_SCALE = 7 is WRONG for this pkl; actual input space is ~1734px not 600px

### Available pkls:
- row1: 1_3 (13 cells)
- row2: 2_1 (27), 2_2, 2_3, 2_4, 2_5
- row3: 3_1 through 3_6
- row4, row5: NO pkls exist

### Contact sheets:
- All 12 tiles have contact sheets at `png_exports/coarse_registration/contact_sheet/`
- Show matched cells with green lines: ex-vivo native (left) ↔ JY306 (right)
- row1_3 matches look suspicious to user

## Step 3 — Fine deformable registration ✓ DONE
- Elastix B-spline with Mattes MI (SimpleITK) on top of 3D affine (Gaussian z-fit)
- All 19 tiles with landmarks processed — 878 total landmarks
- Elastix consistently outperforms affine-only and affine+RBF
- Contact sheets (3-col: exvivo | invivo warped | overlay) + interactive patch viewer
- Scripts: `build_all_tiles_elastix_v2.py`, `build_landmark_patches_viewer.py`
- Outputs: `png_exports/registration_per_tile_elastix/`

## Step 4 — Animate
Show transformation from in-vivo → ex-vivo with highlighted cells:
- Similar to `invivo_to_exvivo_animation.py` (27 matched neurons, pink→green)
- Output: MP4 animation

---

## Status
- Step 1: COMPLETE ✓
- Step 2: COMPLETE ✓ (per-tile 3D affine with Gaussian z-fit, 5-11µm error)
- Step 3: COMPLETE ✓ (elastix B-spline on all 19 tiles, interactive patch viewer)
- Step 4: pending — animate

## 3D Viewer — DEPLOYED
- `3d_viewer/viewer_dual_3d.html` → deployed to Vercel
- GitHub: https://github.com/subro608/invivo-exvivo-cell-registration
- Live URL: https://invivo-exvivo-cell-registration.vercel.app
- Local repo: `/Users/neurolab/neuroinformatics/invivo-exvivo-cell-registration/`
