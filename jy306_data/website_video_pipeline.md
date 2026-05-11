# Margaret — Website & Video Pipeline

How the deployed 3D viewer (`invivo-exvivo-cell-registration/`) and the
Nature Science registration animation (`animation/`) are built. Lists the
**actual** Python files used, in the order they run, and what each produces.

---

## 1. Website — 4-modality 3D viewer

Deployed at `invivo-exvivo-cell-registration/` (Vercel project).

### Source data
- `JY306_in_Vivo_stack_flipped_s80.tif` — in-vivo two-photon volume
- `png_exports/registration_video/<tile>/GFP_z*.png` — 22 ex-vivo nd2 tiles
- `merscope_exvivo /` — 19 MERSCOPE→ex-vivo PKL transforms + transcript CSVs
- `jy306_varol/movie_rolling_avg_*.avi` — temporal-average calcium movie
- `png_exports/registration_per_tile_pkl/` — per-tile PKL-direct landmark NPZs

### Build chain

| Step | Script | Produces |
|---|---|---|
| 1. Generate per-tile landmark NPZs (PKL-direct chain) | [build_patches_pkl_direct_all.py](build_patches_pkl_direct_all.py) | `png_exports/registration_per_tile_pkl/<tile>.npz` (878 cells, 19 tiles) |
| 2. Stitch nd2 tiles → 1 µm isotropic volume | [build_stitched_fullres_v5.py](build_stitched_fullres_v5.py) + [elastix_fullres_v5.py](elastix_fullres_v5.py) | `registration_video/stitched/stitched_gfp_fullres_v5_1um_isotropic.tif` |
| 3. Warp in-vivo z-stack into nd2 space | [register_invivo_to_stitched_pkl_chain.py](register_invivo_to_stitched_pkl_chain.py) | warped in-vivo volume in nd2 coords |
| 4. **Build the 4-modality binary data + landmarks JSON** | **[build_v5_website.py](build_v5_website.py)** | `invivo-exvivo-cell-registration/data/*.bin`, `landmarks.json`, per-cell panel PNGs |
| 5. **Render the dual 3D HTML viewer** | **[build_viewer_dual_3d_v5.py](build_viewer_dual_3d_v5.py)** | `invivo-exvivo-cell-registration/dual_v5.html` (4×modality 2×2 grid) |
| 6. Render the warped in-vivo overlay viewer | [build_viewer_warped_invivo_3d_v5.py](build_viewer_warped_invivo_3d_v5.py) | `invivo-exvivo-cell-registration/viewer_warped_invivo_3d_v5.html` |
| 7. Externalize embedded base64 → CDN-friendly | [make_v4_deploy.py](make_v4_deploy.py) | `invivo-exvivo-cell-registration/viewer_v4.html` with external `patches/` + PNG strips |
| 8. Deploy | `vercel` CLI on `invivo-exvivo-cell-registration/` | preview URL → push to main after approval |

### The four modalities (in `build_v5_website.py`)
1. **Ex-vivo structural** — nd2 GFP tiles (magenta), p99 per-tile normalized
2. **In-vivo structural** — JY306 z-stack warped into nd2 space (green)
3. **In-vivo calcium** — temporal-average movie warped to nd2 space (green functional)
4. **MERSCOPE gene dots** — transcript positions in nd2 space (rainbow, 550 genes ranked by `most_common()`)

### Vercel config
- [invivo-exvivo-cell-registration/vercel.json](invivo-exvivo-cell-registration/vercel.json) — rewrites `/` → `/index.html`
- [invivo-exvivo-cell-registration/index.html](invivo-exvivo-cell-registration/index.html) — top-level landing page that links to the viewer HTMLs

### Historical viewer variants (kept for reference, not deployed)
- `build_viewer_dual_3d.py` → v2 → v3 → v4 → v4_masked → **v5** *(current)*
- `build_rbf_viewer.py`, `build_gapfree_viewer.py`, `build_viewer_equalized.py`, `build_viewer_copy.py` — early experiments per memory's "GP equalized / RBF / gapfree / copy" lineage
- `build_landmark_patches_viewer*.py` — per-landmark 5-panel QC viewers

---

## 2. Video — Registration animation (Nature Science screenplay)

Final composite: `animation/merged_scenes_1-3-5-7.mp4` (and `1-5b-7` variant).
The full screenplay is in [animation_script_v3.md](animation_script_v3.md) and
the v7 master script header documents the exact scene timing.

### Master orchestrator

**[make_registration_animation_v7.py](make_registration_animation_v7.py)** — follows Erdem's screenplay exactly:

| Scene | Frames | Content |
|---|---|---|
| S1  | 120 fr (5 s)  | Histmatch calcium movie |
| S2  | 96 fr (4 s)   | Freeze → max-proj → IV magenta / JY306 z=3 green |
| S3  | 300 fr (12 s) | Green arrows + zoom panels (PKL row4_1 landmarks) |
| S4  | 144 fr (6 s)  | Progressive PKL warp (actual deformation, not fade) |
| S4b | 72 fr (3 s)   | Hold → red+green=yellow matched-cells overlay |
| S5  | 336 fr (14 s) | 3D point-cloud zoom-out, all volumes registered |
| S6  | 192 fr (8 s)  | 8-cell strip 4×4 (EV / IV-warp) |
| S7-9 | 216 fr × 3 | Per-cell deep-dive, 3 cells |

Total ≈ 93 s @ 24 fps.

### Scene-specific asset builders (`animation/`)

| Scene | Script | Output |
|---|---|---|
| S1 (calcium intro) | [animation/scene1_2_movie_to_zstack.py](animation/scene1_2_movie_to_zstack.py) | `scene1_2.mp4` |
| S3 (z-stack 3D) | [animation/scene3_zstack_3d.py](animation/scene3_zstack_3d.py) | `scene3_h264.mp4` |
| S4 (PKL warp) | [animation/build_scene5_deform_assets.py](animation/build_scene5_deform_assets.py) | warp asset frames |
| S5b (tile-stack merge) | [animation/build_scene5b_assets.py](animation/build_scene5b_assets.py), [_v2](animation/build_scene5b_assets_v2.py) | per-tile rotation frames |
| S6/S7 (cell panels) | [animation/candidate_4panel_nd2.py](animation/candidate_4panel_nd2.py), [_v2](animation/candidate_4panel_v2.py) | 4-panel cell videos (CALCIUM \| IV \| EV \| MS) |
| Three-stack rotation | [animation/build_three_stacks_assets.py](animation/build_three_stacks_assets.py) | `3layer_stack_preview.mp4` |
| Stitched + MERSCOPE | [animation/build_stitched_with_merscope.py](animation/build_stitched_with_merscope.py) | composite stitched-mosaic frames |
| Rotation transitions | [animation/gen_rotate_transition.py](animation/gen_rotate_transition.py) | smooth scene-to-scene rotation frames |
| Final assembly | [animation/edit_combined_video.py](animation/edit_combined_video.py) | `merged_scenes_1-3-5-7.mp4` |
| Post: scale bars on S5b | [animation/add_scale_bars_5b.py](animation/add_scale_bars_5b.py) | µm scale bars added per-phase |
| Post: caption fixes | [animation/fix_captions.py](animation/fix_captions.py) | IN VIVO / EX VIVO / MERSCOPE mRNA labels |

### Standalone animations (older versions, kept for reference)
- [invivo_to_exvivo_animation.py](invivo_to_exvivo_animation.py) — pre-screenplay affine-fit demo
- [native_to_registered_animation.py](native_to_registered_animation.py) — nd2 native → registered overlay
- [make_registration_animation.py](make_registration_animation.py) → v2…v6 → **v7** *(current)*

### Twitter showcase (separate, simpler output)
- [build_twitter_final.py](build_twitter_final.py) — 3-panel layout (GIF + static + 3-cell column) for Jason's Twitter post
- [build_twitter_showcase.py](build_twitter_showcase.py), [build_twitter_tissue.py](build_twitter_tissue.py) — earlier variants

---

## 3. Supporting / cross-cutting scripts

These don't produce website or video output directly but are dependencies of
both pipelines.

| Purpose | Script |
|---|---|
| Apply PKL deformation (forward) | [apply_pkl_transform.py](apply_pkl_transform.py) |
| Apply PKL deformation (inverse, iterative) | [inverse_pkl_transform.py](inverse_pkl_transform.py) |
| Build per-tile landmark NPZs via PKL chain | [build_patches_pkl_direct_all.py](build_patches_pkl_direct_all.py) |
| MERSCOPE → ex-vivo overlay | [build_merscope_overlay.py](build_merscope_overlay.py) |
| MERSCOPE per-cell patch contact sheet | [build_merscope_contact_sheet.py](build_merscope_contact_sheet.py) |
| Stitching via elastix at full-res | [elastix_fullres_v5.py](elastix_fullres_v5.py) |
| Comparison collages (alignment QC) | [comparison_collage_v5.py](comparison_collage_v5.py), [comparison_collage_v5b.py](comparison_collage_v5b.py) |
| Auto-alignment (IoU + landmark-aware) | [auto_align_iou_v4.py](auto_align_iou_v4.py) |
| Landmark picking (interactive GUI) | [landmark_picker_3d.py](landmark_picker_3d.py) |

---

## 4. Common gotchas (from memory)

- `exvivo_total` is a **labels** array, not intensity — see [coord pitfalls memory](file:///Users/neurolab/.claude/projects/-Users-neurolab-neuroinformatics-margaret/memory/project_coord_pitfalls.md)
- `merscope_exvivo ` folder has a **trailing space** in the name (used as-is in all scripts)
- PKL transforms use rows 1–3 only (NOT rows 4–5)
- Ex-vivo display = **raw intensity** (no threshold); median filter is for registration *input* only, not display
- MERSCOPE gene colors must be ordered by `most_common()`, stored BGR, converted to RGB only at output

---

## 5. Output locations

- **Website data**: `invivo-exvivo-cell-registration/data/`, `/patches/`, `/*.html`
- **Animation MP4s**: `animation/merged_scenes_*.mp4`, `animation/scene*.mp4`
- **QC frames**: `png_exports/registration_animation_v7_qc/`
- **Stitched volumes**: `png_exports/registration_video/stitched/`
- **Per-tile landmarks**: `png_exports/registration_per_tile_pkl/`
