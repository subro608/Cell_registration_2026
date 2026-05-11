---
name: No Assumptions — Nature Science Quality
description: Never use placeholder values, arbitrary constants, or assumptions. Everything must be computed from real data.
type: feedback
---

This is a Nature Science publication. Every value must be derived from actual data, never guessed or approximated.

**Examples of what NOT to do:**
- Z_SPACING = 4 (arbitrary) — compute from real pixel sizes
- Threshold = 0.05 (guessed) — derive from data statistics
- "~2 µm z-step" — look up the actual value from metadata
- Hardcoded positions — compute from transforms/registration data

**How to apply:** Before using any constant, ask: is this derived from real data or am I guessing? If guessing, find the real value from the data files, metadata, or registration transforms. If uncertain, ask the user rather than assume.
