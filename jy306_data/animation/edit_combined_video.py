"""
Edit combined video (scenes_1_to_5b.mp4) based on Erdem's feedback:

1. t≈12-15s: Don't split back to side-by-side after overlay. Crossfade directly to scene 3.
2. t≈27-30s: No black gap. Crossfade scene 3 end → scene 4 start.
3. t≈56-60s: Remove row2_1 second rotation (settle phase). Crossfade to next section.
4. Scene 5b: Remove stitched 3D settle (second loop). One rotation only.

Approach: Read all frames, cut/crossfade at specific frame ranges, write to new frames dir.
"""

import cv2, numpy as np, math, os, subprocess

BASE = '/Users/neurolab/neuroinformatics/margaret/animation'
INPUT = f'{BASE}/scenes_1_to_5b_backup.mp4'
FRAMES_OUT = f'{BASE}/frames_combined_edited'
OUTPUT = f'{BASE}/scenes_1_to_5b_edited.mp4'

FPS = 24
XFADE = 12  # crossfade length in frames

def ease(t):
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))

os.makedirs(FRAMES_OUT, exist_ok=True)

# Read ALL frames
print("Reading all frames from video...")
cap = cv2.VideoCapture(INPUT)
all_frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    all_frames.append(frame)
cap.release()
N = len(all_frames)
print(f"  {N} frames loaded ({N/FPS:.1f}s)")

# Define CUT regions (0-indexed frame numbers)
# Each cut: (start_frame, end_frame) — frames to REMOVE
# Plus crossfade frames at boundaries

# Issue 1: t≈12.5-15s — split back to side-by-side
# Overlay is good at frame ~288 (12s), split starts ~312 (13s), scene3 begins ~360 (15s)
CUT1_START = 300   # start cutting (overlay still visible)
CUT1_END = 372     # scene 3 is underway

# Issue 2: t≈27-30s — black gap between scene 3 and scene 4
# Scene 3 ends ~648 (27s), black at ~672 (28s), scene 4 starts ~720 (30s)
CUT2_START = 648
CUT2_END = 744

# Issue 3: t≈56-60s — row2_1 second rotation (settle/G3)
# First rotation ends ~1368 (57s), settle ~1368-1416, hold starts ~1416
# G2 rotation: 120 frames ending at some point, G3 settle: 36 frames, G4 hold: 24 frames
# The settle rotates back — cut it and crossfade from rotation end to hold start
CUT3_START = 1368
CUT3_END = 1416

# Issue 4: Scene 5b — stitched 3D settle (second loop)
# Scene 5b starts at frame 6432
# In scene5b_v2: 204 frames row5_1 + 36 crossfade + 276 stitched 3D
# Stitched 3D: 60 emerge + 144 rotate + 36 settle + 36 hold
# The settle is frames 6432+204+36+60+144 = 6876 to 6876+36 = 6912
# Let me be more precise from the frames_scene5b_v2 structure:
# frame 1-204: row5_1 3D
# frame 205-240: crossfade
# frame 241-300: stitched emerge (60fr)
# frame 301-444: stitched rotate (144fr)
# frame 445-480: stitched settle (36fr) ← REMOVE THIS
# frame 481-516: stitched hold (36fr)
# In combined: scene5b starts at frame 6432
S5B_START = 6432
CUT4_START = S5B_START + 444   # 6876 — settle start
CUT4_END = S5B_START + 480     # 6912 — settle end

print(f"Cuts:")
print(f"  Cut 1 (split-back):     frames {CUT1_START}-{CUT1_END} ({(CUT1_END-CUT1_START)/FPS:.1f}s)")
print(f"  Cut 2 (black gap):      frames {CUT2_START}-{CUT2_END} ({(CUT2_END-CUT2_START)/FPS:.1f}s)")
print(f"  Cut 3 (row2_1 settle):  frames {CUT3_START}-{CUT3_END} ({(CUT3_END-CUT3_START)/FPS:.1f}s)")
print(f"  Cut 4 (stitch settle):  frames {CUT4_START}-{CUT4_END} ({(CUT4_END-CUT4_START)/FPS:.1f}s)")

# Build output frame list
print("\nBuilding edited sequence...")
output_frames = []

cuts = [
    (CUT1_START, CUT1_END),
    (CUT2_START, CUT2_END),
    (CUT3_START, CUT3_END),
    (CUT4_START, CUT4_END),
]

i = 0
for cut_start, cut_end in cuts:
    # Add frames before this cut
    while i < cut_start and i < N:
        output_frames.append(all_frames[i])
        i += 1

    # Generate crossfade: last frame before cut → first frame after cut
    frame_a = all_frames[min(cut_start - 1, N - 1)]
    frame_b = all_frames[min(cut_end, N - 1)]
    for fi in range(XFADE):
        t = ease(fi / (XFADE - 1))
        blended = cv2.addWeighted(frame_a, 1 - t, frame_b, t, 0)
        output_frames.append(blended)

    # Skip the cut region
    i = cut_end

# Add remaining frames
while i < N:
    output_frames.append(all_frames[i])
    i += 1

n_out = len(output_frames)
print(f"  Input: {N} frames ({N/FPS:.1f}s)")
print(f"  Output: {n_out} frames ({n_out/FPS:.1f}s)")
print(f"  Removed: {N - n_out + len(cuts)*XFADE} frames, added {len(cuts)*XFADE} crossfade frames")

# Write frames
print(f"\nWriting {n_out} frames...")
for idx, frame in enumerate(output_frames):
    if idx % 500 == 0:
        print(f"  {idx}/{n_out}...")
    cv2.imwrite(f'{FRAMES_OUT}/frame_{idx+1:05d}.png', frame)

print(f"Done writing frames to {FRAMES_OUT}/")

# Encode
print(f"\nEncoding to H.264...")
subprocess.run([
    'ffmpeg', '-y', '-framerate', str(FPS),
    '-i', f'{FRAMES_OUT}/frame_%05d.png',
    '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
    OUTPUT
], capture_output=True)
print(f"Done! {n_out} frames, {n_out/FPS:.1f}s @ {FPS}fps -> {OUTPUT}")
