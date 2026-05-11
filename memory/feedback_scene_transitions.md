---
name: Scene Transition Continuity
description: No blackouts between scenes — each scene's first frame must match the previous scene's last frame visually.
type: feedback
---

No blackouts or fade-to-black between consecutive scenes. Each scene must start from where the previous one ended.

**Why:** User flagged blackout between scene 4→5 where in-vivo moved to left, then both images faded in from black.

**How to apply:** When scene N ends with an image on screen, scene N+1's first frame must show that same image already visible. Use `is_first_tile` flags or similar to skip fade-in for elements already present. Test by concatenating scenes and watching the transitions.
