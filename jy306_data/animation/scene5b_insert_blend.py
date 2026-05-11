"""
Insert 12 blend frames between frame 690 (last C2) and 691 (first D).
Shifts existing frames 691+ by 12, then generates blend frames 691-702.
"""
import numpy as np, cv2, os, shutil

BASE = '/Users/neurolab/neuroinformatics/margaret'
FRAMES_DIR = f'{BASE}/animation/frames_multi_tile_3d'

# Frame 690 = last C2 (individual tiles), frame 691 = first D (stitched volume)
BLEND_START = 690  # last frame before blend
BLEND_AFTER = 691  # first frame to shift
N_BLEND = 12

# Step 1: Shift frames 691+ by N_BLEND
print("Shifting existing frames...")
all_frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.startswith('frame_') and f.endswith('.png')])
# Work backwards to avoid overwriting
frames_to_shift = [f for f in all_frames if int(f.split('_')[1].split('.')[0]) >= BLEND_AFTER]
frames_to_shift.sort(reverse=True)

for fname in frames_to_shift:
    idx = int(fname.split('_')[1].split('.')[0])
    new_idx = idx + N_BLEND
    old_path = f'{FRAMES_DIR}/{fname}'
    new_path = f'{FRAMES_DIR}/frame_{new_idx:05d}.png'
    os.rename(old_path, new_path)

print(f"  Shifted {len(frames_to_shift)} frames by +{N_BLEND}")

# Step 2: Load the two endpoint frames
frame_c2 = cv2.imread(f'{FRAMES_DIR}/frame_{BLEND_START:05d}.png')
frame_d0 = cv2.imread(f'{FRAMES_DIR}/frame_{BLEND_AFTER + N_BLEND:05d}.png')
print(f"  C2 last: frame_{BLEND_START:05d}.png, shape={frame_c2.shape}")
print(f"  D first: frame_{BLEND_AFTER + N_BLEND:05d}.png, shape={frame_d0.shape}")

# Step 3: Generate blend frames
def ease(t):
    import math
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))

for i in range(N_BLEND):
    t = ease(i / max(1, N_BLEND - 1))
    blended = cv2.addWeighted(frame_c2, 1 - t, frame_d0, t, 0)
    out_idx = BLEND_AFTER + i
    out_path = f'{FRAMES_DIR}/frame_{out_idx:05d}.png'
    cv2.imwrite(out_path, blended)

print(f"  Generated {N_BLEND} blend frames ({BLEND_AFTER} to {BLEND_AFTER + N_BLEND - 1})")

total = len([f for f in os.listdir(FRAMES_DIR) if f.startswith('frame_') and f.endswith('.png')])
print(f"\nTotal frames now: {total}")
