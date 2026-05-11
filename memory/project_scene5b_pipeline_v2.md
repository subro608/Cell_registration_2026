---
name: Scene 5b Pipeline v2
description: Current scene 5b rendering pipeline — per-tile channels (no stitched intermediate), rotation blends, caption conventions
type: project
---

## Scene 5b Combined Frame Layout (v2)
1. **Frames 1-26**: Hold last frame of scene 5 (smooth transition)
2. **Frames 27-543**: multi_tile_3d (flat grid with MERSCOPE, from scene5b_multi_tile_3d_parallel.py)
3. **Frames 544-963**: three_stacks_v2 (420 frames, from scene5b_three_stacks_v2.py)
4. **Frames 964-1059**: channel-to-stitched transition (96 frames, from gen_channel_to_stitched.py)
5. **Frames 1060-1083**: fade to black
Total: 1083 frames = 45.1s @ 24fps

## Three Stacks V2 Phases (420 frames)
1. Split (36fr): merged center → 3 columns (IN VIVO | EX VIVO | MERSCOPE)
2. Rotate split (36fr): rotate 3 grids
3. Merge grids (36fr): 3 columns → merged center
4. Tile merge (72fr): tiles slide to center with z-depth
5. Scale-up (36fr): grow to volume scale + z correction
6. **Rotation blend (72fr)**: 360° rotation, blend per-tile combined → per-tile channels at back-facing (20-55%). NO stitched volume here.
7. Split channels (48fr): per-tile channels → 3 side-by-side columns
8. Hold split (48fr): rotate 3 channel volumes
9. Merge channels (36fr): 3 channels → single combined

## Channel-to-Stitched Transition (96 frames)
- Blends from per-tile combined (dense_with_ms) → stitched combined volume during 360° rotation
- NO 3-channel per-tile intermediate (removed — was showing 3 overlapping volumes)
- Hold stitched for 24 frames at end

## Key Design Decisions
- **No stitched 3D intermediate** between per-tile combined and channel split (user request)
- **Per-tile channel rendering** for split/hold/merge phases (not stitched volumes)
- **Rotation hides data source swaps** — blend during back-facing portion
- **No scale bars** on scene 5b
- **3D axis widget** (AP/ML/DV) on all 3D phases
- **Per-tile ex-vivo normalization**: target p99=204 (p25 of all tiles)
- Pixel sizes: in-vivo=0.82 µm/px, ex-vivo=0.65 µm/px

## Caption Conventions (2026-04-05 update from Erdem/Jason)
- NO hyphens: "IN VIVO" not "IN-VIVO", "EX VIVO" not "EX-VIVO"
- NO "confocal" with in vivo imaging
- "MERSCOPE mRNA EXPRESSION" not "GENE EXPRESSION"
- Applied across ALL scene scripts (sed replacements done)

## Assets
- `scene5b_assets_v3.pkl` (212MB): per-tile dense slices, NOT modified (normalization at load time)
- `scene5b_three_stacks_assets.pkl` (1.5GB): all rendering assets including stitched volumes
- Output folders: frames_multi_tile_3d/, frames_three_stacks_v2/, frames_channel_to_stitched_v2/

## Full Combined Video
`scenes_1_3_5_5b_7.mp4`: 8827 frames = 6.1 min
- 1-5784: scenes 1-3-5 (from frames_merged_1_3_5)
- 5785-6867: scene 5b
- 6868-8827: scene 7
