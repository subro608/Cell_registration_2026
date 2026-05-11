---
name: Animation Flow Feedback (2026-04-03)
description: Erdem's feedback on scenes_1_to_5b video — eliminate redundant loops, smoother transitions, keep forward momentum
type: feedback
---

## Key Principles
- **No going backwards**: Once a transition is done, don't undo it (e.g., don't split back to side-by-side after aligning, don't make in-vivo disappear then reappear)
- **One rotation loop is enough**: No second rotation of any 3D volume — one loop per volume
- **Keep forward momentum**: Every frame should advance the story, no redundant holds or repeats

## Specific Fixes (timestamps from scenes_1_to_5b.mp4)

1. **t=10s** (Scene 1-2 transition): After aligning in-vivo video to z-stack, DON'T split left-right again. Make in-vivo video disappear and go straight to next scene.
2. **t=29s** (Scene 3 start): Don't make in-vivo disappear then reappear. Keep continuous.
3. **t=59s** (Scene 3 rotation): Only ONE rotation loop of the in-vivo 3D volume, not two.
4. **t=4:36s** (Scene 5b): Only ONE rotation loop of the stitched 3D volume, not two.
5. **Scene 5→5b transition**: After last tile (row5_1) aligned, show ALL ex-vivo tiles with their aligned in-vivo slices in the original row×image grid format. Then smoothly merge/stack these together on top of each other. Then do the rotating 3D loop (once).

## Why:
Erdem wants a clean, publication-quality forward narrative. Redundancy and back-and-forth breaks the visual flow and wastes viewer attention.

## How to apply:
When building animation scenes, always check: does this frame move the story forward? If it repeats or reverses a previous transition, cut it.
