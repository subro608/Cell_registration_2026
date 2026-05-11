# Registration Animation — Video Script v3
*Inspired by Science abl5981 Movie S1 — stops before MERSCOPE*

---

## Style Guide
- **Background**: pure black `#000000`
- **Text**: all-caps, white, bottom-center, Helvetica/sans-serif, fade-in/out
- **No UI chrome** — no sliders, no legends except minimal channel labels top-left
- **Transitions**: slow ease-in/out fades and zooms (no cuts)
- **Aspect ratio**: 1920×1080, 24 fps
- **Audio**: none

---

## Scene 1 — Calcium Movie  `0–8s  (192 fr)`

**Visuals**
- Show `movie_warped_h264.mp4` (663 frames, warped to JY306 space) playing in grayscale
- Start slightly zoomed in (scale 1.15×), centered
- Slow zoom out to 1.0× over the 8 seconds
- Render as grayscale (raw, no colour tint)

**Text**
- None for first 4s
- At 4s fade in bottom-center: `IN VIVO TWO-PHOTON CALCIUM IMAGING`

**Notes**
- Play at real speed (movie is 24fps equivalent — step every 3 frames of 663 over 192 output frames)
- Dark cell-body "holes" must be visible — use percentile stretch p1/p99

---

## Scene 2 — Freeze + Highlight  `8–12s  (96 fr)`

**Visuals**
- Movie slows to stop (last frame holds)
- Max-projection frame cross-fades in (still grayscale)
- 3–5 landmark cells get a **white dashed circle** drawn around them, fading in one by one
- Slight brightness boost on the max-proj

**Text**
- `IN VIVO TWO-PHOTON CALCIUM IMAGING` fades out at 9s
- `NEURONS RECORDED DURING BEHAVIOUR` fades in at 10s

---

## Scene 3 — Registration Transition  `12–18s  (144 fr)`

**Visuals**
- The 2P max-proj **shrinks and slides to left** half of frame (12→15s)
- On the right half: ex-vivo confocal MIP of the matching tile (row2_1, nd2 GFP MIP) fades in
- At 15s both are visible side-by-side (~equal size)
- Matching landmark circles appear on both sides with **thin white lines** connecting them (drawn one at a time)
- At 17s the two images gently **zoom toward each other** and begin to overlap (morph)

**Text**
- `NEURONS RECORDED DURING BEHAVIOUR` fades out at 13s
- `RE-IDENTIFIED IN EX-VIVO TISSUE` fades in at 14s, holds

**Channel labels** (top-left, small)
- Left panel: `2P GFP` in dim white
- Right panel: `CONFOCAL GFP` in dim white

---

## Scene 4 — 3D Slab Reveal  `18–32s  (336 fr)`

**Visuals**
- The ex-vivo tile (nd2 GFP MIP) is rendered as a **3D rectangular slab** using perspective projection
- Starts flat (face-on), then **tilts ~35° around X axis** over 4s → reveals depth/thickness
- The slab has a visible **top face** (the MIP) and **side edges** (white/gray lines with slight depth shading)
- For depth: stack 5–7 slightly offset semi-transparent copies of the GFP slices to simulate the z-stack volume
- After tilt, **slow pan** moves across the slab surface (like in the Science video), zooming into individual neurons
- At 28s: zoom into a central crop showing ~6 bright cell bodies clearly

**Text**
- `RE-IDENTIFIED IN EX-VIVO TISSUE` fades out at 19s
- `MULTIMODAL CELL MATCHING` fades in at 20s, holds through scene

**Notes**
- Use `cv2.warpPerspective` with a homography that tilts the image
- Add thin white border lines to show slab edges
- Slight vignette around edges

---

## Scene 5 — Cell Strip  `32–46s  (336 fr)`

**Visuals**
- Layout splits into **two rows**:
  - **Top row**: numbered ex-vivo cell crops (100×100 px each), 6 cells shown, sliding in from right one by one
  - **Bottom row**: corresponding in-vivo calcium max-proj crops (same cells, same order)
- Each cell numbered 1–6 in a distinct colour (matching Science video style: red, orange, yellow, green, cyan, blue)
- Cells appear one by one (every ~1.5s)
- After all 6 are shown, brief hold

**Text**
- Top row label: `EX VIVO CONFOCAL` (small, left edge)
- Bottom row label: `IN VIVO CALCIUM` (small, left edge)
- Bottom-center: `MATCHED NEURONS` fades in when all 6 shown

**Notes**
- Use filtered (blob-distance ≤5µm) landmarks for best matches
- Crop from nd2 GFP MIP for ex-vivo, from warped movie max-proj for in-vivo
- Both normalised independently

---

## Scene 6 — Per-Cell Deep Dive  `46–74s  (3 cells × ~9s = 648 fr)`

*Repeat for 3 cells (cells 1, 2, 3 from Scene 5)*

**Layout** (per cell, 216 frames / 9s each)

```
┌─────────────────────────┬─────────────────────────┐
│  2P field (full FOV)    │  Ex-vivo confocal patch  │
│  red circle on cell     │  (zoomed 200×200 px)     │
│                         │                          │
│  [calcium trace below]  │  In-vivo MIP patch       │
│  ───────────────────    │  (same cell, zoomed)     │
│  0s              28s    │                          │
└─────────────────────────┴─────────────────────────┘
```

**Left panel**
- Full 2P field (JY306 MIP or single z), grayscale
- White dashed circle pulses around the target cell
- Below it: calcium trace (intensity over 663 frames), white line on black
- A red vertical playhead scrubs along the trace as the movie plays
- The 2P field loops through the calcium movie for this cell (cropped ±50px)

**Right panel — top**
- Ex-vivo confocal crop (from nd2 GFP, patch_strip col 0), zoomed and contrast-stretched
- White crosshair at cell centre

**Right panel — bottom**
- In-vivo MIP crop (from patch_strip col 4)
- White crosshair

**Transition between cells**
- Current cell fades out (0.5s), black frame (0.25s), next cell number fades in

**Text (bottom-center)**
- `FUNCTIONAL / STRUCTURAL READOUT` shown for all 3 cells
- Cell number shown top-left: `CELL 1`, `CELL 2`, `CELL 3`

---

## Scene 7 — Summary  `74–82s  (192 fr)`

**Visuals**
- Fade to black (0.5s)
- Three panels arranged horizontally:
  1. **Left**: 2P calcium movie (looping, small)
  2. **Centre**: ex-vivo confocal slab (tilted, still)
  3. **Right**: matched cell strip (3 pairs, static)
- Each panel fades in sequentially (left → centre → right), 1s each
- Hold for 3s

**Text**
- `MULTIMODAL REGISTRATION` top-center, fades in
- `IN VIVO  ·  EX VIVO  ·  CALCIUM  ·  CONFOCAL` subtitle below, fades in
- Bottom-center: `JY306 MOUSE HIPPOCAMPUS`

---

## Data Sources

| Data | File | Usage |
|---|---|---|
| Calcium movie (warped) | `png_exports/native_invivo/movie_warped_h264.mp4` | Scenes 1, 2, 6 |
| JY306 in-vivo stack | `JY306_in_Vivo_stack_flipped_s80.tif` | Scene 3 left panel, cell trace |
| nd2 row2_1 tile (GFP PNGs) | `png_exports/registration_video/row2_1/GFP_z*.png` | Scenes 3, 4, 5 |
| Landmarks (nd2+JY306) | `registration_video/landmarks_27_nd2_native.npz` | Scenes 3, 5, 6 |
| Patch strip | `3d_viewer/patch_strip_v4.png` + `cell_info_v4.json` | Scenes 5, 6 |
| Filtered landmark list | `cell_info_v4.json` (filtered=True, z_iv=3±2) | Scenes 5, 6 |

---

## Key Parameters

```python
W, H    = 1920, 1080
FPS     = 24
FONT    = cv2.FONT_HERSHEY_SIMPLEX  # closest to sans-serif
TEXT_SCALE = 0.75    # main captions
TEXT_THICK = 2
CELL_COLOURS = [
    (  0,   0, 255),   # red    (BGR)
    (  0, 100, 255),   # orange
    (  0, 210, 255),   # yellow
    (  0, 200,  80),   # green
    (200, 200,   0),   # cyan
    (230,  80,   0),   # blue
]
SLAB_TILT_DEG  = 35    # X-axis tilt for 3D slab
SLAB_THICKNESS = 8     # number of z-slices stacked for depth
```

---

## Output

`png_exports/registration_animation_v3.mp4`  
H.264, CRF 18, 1920×1080, 24fps, ~82s
