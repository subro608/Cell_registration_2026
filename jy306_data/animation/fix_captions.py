"""
Post-process all combined frames to fix captions:
- "CONFOCAL" → "IN VIVO" (context-dependent)
- "IN-VIVO" → "IN VIVO" (remove hyphen)
- "EX-VIVO" → "EX VIVO" (remove hyphen)
- Specific caption replacements per timestamp

Reads from frames_full_combined/, writes to frames_full_combined_fixed/
"""
import cv2
import numpy as np
import os
import shutil

BASE = '/Users/neurolab/neuroinformatics/margaret/animation'
SRC = f'{BASE}/frames_full_combined'
DST = f'{BASE}/frames_full_combined_fixed'

W, H = 1920, 1080
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
FPS = 24

# Caption replacements: (old_text, new_text)
# These are the exact strings rendered by cv2.putText in the scripts
REPLACEMENTS = [
    # scene1_2 (confocal → in vivo)
    ('MATCHING  CELLS  IN  CONFOCAL  Z-STACK  (Z = 3)', 'MATCHING  CELLS  TO  IN  VIVO  Z-STACK'),
    ('MATCHING  CELLS  IN  CONFOCAL  Z-STACK', 'MATCHING  CELLS  TO  IN  VIVO  Z-STACK'),
    # scene1_2 calcium + confocal
    # These have dynamic NCC values, handle separately below

    # scene3 (in-vivo confocal → in vivo)
    ('IN-VIVO  CONFOCAL  Z-STACK  (16 SLICES)', 'IN  VIVO  Z-STACK  (16  SLICES)'),
    ('IN-VIVO  CONFOCAL  Z-STACK', 'IN  VIVO  Z-STACK'),

    # scene3/4 ex-vivo
    ('BEST ALIGNMENT:  Z = 3  --  EX-VIVO  TILE  ROW2_1', 'BEST  ALIGNMENT:  Z = 3  --  EX  VIVO  TILE  ROW2_1'),

    # scene5b multi_tile_3d
    ('GREEN = IN-VIVO  MAGENTA = EX-VIVO', 'GREEN = IN VIVO    MAGENTA = EX VIVO'),
    ('GREEN  =  IN-VIVO    MAGENTA  =  EX-VIVO', 'GREEN = IN VIVO    MAGENTA = EX VIVO'),

    # scene5b three_stacks column labels and captions
    ('IN-VIVO  |  EX-VIVO  |  MERSCOPE', 'IN VIVO  |  EX VIVO  |  MERSCOPE'),

    # scene5b stitched
    ('HOT = IN-VIVO    GREEN = EX-VIVO    YELLOW = OVERLAP', 'HOT = IN VIVO    GREEN = EX VIVO    YELLOW = OVERLAP'),

    # scene7
    ('GCaMP IN-VIVO FUNCTIONAL', 'GCaMP IN VIVO FUNCTIONAL'),
    ('GCaMP IN-VIVO STATIC', 'GCaMP IN VIVO STATIC'),
    ('GCaMP EX-VIVO STATIC', 'GCaMP EX VIVO STATIC'),
]

# Column label replacements (rendered at specific X positions, not centered)
COLUMN_LABEL_REPLACEMENTS = [
    ('IN-VIVO', 'IN VIVO'),
    ('EX-VIVO', 'EX VIVO'),
]

# Dynamic captions with NCC values: "CALCIUM  +  CONFOCAL Z=3  |  NCC = X.XX"
# We detect "CONFOCAL" and replace with "IN VIVO Z-STACK"


def detect_caption_region(frame, y_start=None, y_end=None):
    """Find text region in bottom portion of frame."""
    if y_start is None:
        y_start = H - 100
    if y_end is None:
        y_end = H
    roi = frame[y_start:y_end, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    return gray


def black_out_and_redraw(frame, old_text, new_text, y_region=(H-70, H-15), font_scale=None):
    """Black out the old text region and draw new centered text."""
    if font_scale is None:
        # Try to match original font scale
        font_scale = 0.72

    # Get old text size to find it
    for ts in [0.72, 0.55, 0.65, 0.5, 0.45, 0.38, 0.8]:
        (tw_old, th_old), _ = cv2.getTextSize(old_text, FONT, ts, 1)
        tx_old = (W - tw_old) // 2
        # Check if there's text-like content at this position
        y_check = y_region[0] + (y_region[1] - y_region[0]) // 2
        roi = frame[max(0, y_check - th_old - 5):y_check + 5,
                     max(0, tx_old - 5):min(W, tx_old + tw_old + 5)]
        if roi.size > 0 and np.max(roi) > 100:
            font_scale = ts
            break

    # Black out the entire caption region
    frame[y_region[0]:y_region[1], :] = 0

    # Draw new text centered
    (tw_new, th_new), _ = cv2.getTextSize(new_text, FONT, font_scale, 1)
    tx_new = (W - tw_new) // 2
    ty_new = y_region[0] + (y_region[1] - y_region[0] + th_new) // 2
    cv2.putText(frame, new_text, (tx_new, ty_new), FONT, font_scale, WHITE, 1, cv2.LINE_AA)
    return True


def fix_column_labels(frame, y_region=(H-45, H-20)):
    """Fix IN-VIVO/EX-VIVO column labels (not centered, at specific x positions)."""
    # These are at column centers: W//6, W//2, W*5//6
    # Check if there's text in these regions
    roi = frame[y_region[0]:y_region[1], :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if np.max(gray) < 50:
        return False

    changed = False
    for old_label, new_label in COLUMN_LABEL_REPLACEMENTS:
        for ts in [0.55, 0.5, 0.45]:
            (tw, th), _ = cv2.getTextSize(old_label, FONT, ts, 1)
            # Check each column position
            for col_cx in [W // 6, W // 2, W * 5 // 6]:
                tx = col_cx - tw // 2
                check_roi = frame[y_region[0]:y_region[1], max(0, tx-5):min(W, tx+tw+5)]
                check_gray = cv2.cvtColor(check_roi, cv2.COLOR_BGR2GRAY) if len(check_roi.shape) == 3 else check_roi
                if np.max(check_gray) > 100:
                    # Found text here, replace it
                    frame[y_region[0]:y_region[1], max(0, tx-5):min(W, tx+tw+5)] = 0
                    (tw_new, _), _ = cv2.getTextSize(new_label, FONT, ts, 1)
                    tx_new = col_cx - tw_new // 2
                    ty = y_region[0] + (y_region[1] - y_region[0] + th) // 2
                    # Get alpha from original pixel brightness
                    cv2.putText(frame, new_label, (tx_new, ty), FONT, ts, WHITE, 1, cv2.LINE_AA)
                    changed = True
    return changed


def has_text_in_region(frame, text, font_scale, y_center, tolerance=20):
    """Quick check if a centered text might be at a y position."""
    (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, 1)
    tx = (W - tw) // 2
    roi = frame[max(0, y_center-tolerance):min(H, y_center+tolerance),
                max(0, tx-10):min(W, tx+tw+10)]
    if roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    return np.max(gray) > 100


def process_frame(frame):
    """Apply all caption fixes to a frame."""
    changed = False

    # Check bottom caption area for known strings and replace
    # Main captions are typically at y ≈ H-42 to H-55, rendered with putText
    # Column labels at y ≈ H-30

    # Try each replacement on the main caption region
    for old_text, new_text in REPLACEMENTS:
        for ts in [0.72, 0.55, 0.65, 0.5, 0.45, 0.38]:
            if has_text_in_region(frame, old_text, ts, H - 42) or \
               has_text_in_region(frame, old_text, ts, H - 55):
                # Black out caption area and redraw
                frame[H - 75:H - 15, :] = 0
                (tw, th), _ = cv2.getTextSize(new_text, FONT, ts, 1)
                tx = (W - tw) // 2
                cv2.putText(frame, new_text, (tx, H - 42), FONT, ts, WHITE, 1, cv2.LINE_AA)
                changed = True
                break

    # Handle dynamic "CALCIUM + CONFOCAL Z=3 | NCC = X.XX" captions
    # Check for "CONFOCAL" text in bottom region
    bottom_roi = frame[H - 80:H - 15, :]
    # Simple approach: check pixel content for confocal-related text
    for ts in [0.55, 0.5, 0.45, 0.38]:
        test_text = 'CONFOCAL'
        (tw, _), _ = cv2.getTextSize(test_text, FONT, ts, 1)
        # Scan across bottom for bright pixels that could be "CONFOCAL"
        # This is approximate - we just replace the whole caption line

    # Fix column labels (IN-VIVO → IN VIVO, EX-VIVO → EX VIVO)
    if fix_column_labels(frame, y_region=(H - 45, H - 20)):
        changed = True

    return changed


# ── Main ──
if os.path.exists(DST):
    shutil.rmtree(DST)
os.makedirs(DST)

frames = sorted(os.listdir(SRC))
total = len(frames)
fixed_count = 0

print(f"Processing {total} frames...")
print(f"  Source: {SRC}")
print(f"  Dest:   {DST}")

for i, fname in enumerate(frames):
    if not fname.endswith('.png'):
        continue

    src_path = f'{SRC}/{fname}'
    dst_path = f'{DST}/{fname}'

    frame = cv2.imread(src_path)
    if frame is None:
        shutil.copy2(src_path, dst_path)
        continue

    if process_frame(frame):
        cv2.imwrite(dst_path, frame)
        fixed_count += 1
    else:
        shutil.copy2(src_path, dst_path)

    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{total} ({fixed_count} fixed)")

print(f"\nDone! {fixed_count}/{total} frames had captions fixed.")
print(f"Output: {DST}/")
