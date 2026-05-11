---
name: Overlay Color Convention
description: Paper color scheme — in-vivo GCaMP green, ex-vivo GCaMP magenta. Must stay consistent across ALL scenes AND viewers.
type: feedback
---

In-vivo GCaMP = green (G channel only in BGR)
Ex-vivo GCaMP = magenta (B + R channels in BGR)
Calcium movie = grayscale (raw functional imaging, not colored)

**Why:** Jason requested colors match the paper figures (2026-04-04). Nature Science publication.

**How to apply:**
- ALL viewers (3D clouds, labels, patch images) must use this convention consistently
- Calcium movie playback is always grayscale — it's raw functional data, not a color-coded modality
- When calcium is registered/aligned and shown alongside confocal, the confocal (in-vivo) emerges as green on top of grayscale calcium
- MERSCOPE GCaMP background = magenta (it's ex-vivo tissue)
- Scene 1: grayscale calcium → green confocal fades in on top
- Scene 5b: magenta ex-vivo + green in-vivo tiles
- Scene 7: green in-vivo panel, magenta ex-vivo panel, magenta MERSCOPE GCaMP + multi-color gene dots
- JY306 confocal z-stack = in-vivo = green (NOT magenta)
- nd2 = ex-vivo = magenta
- dual_v5.html 3D clouds: ex-vivo=magenta, in-vivo=green, calcium=green (fixed 2026-04-07, was previously swapped)
