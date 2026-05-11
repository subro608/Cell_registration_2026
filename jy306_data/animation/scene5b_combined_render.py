"""
Scene 5b Combined Renderer
===========================
Runs all 3 scene5b sub-scripts in sequence, then assembles the combined output.

Sub-scripts (run in order):
  1. scene5b_multi_tile_3d_parallel.py → frames_multi_tile_3d/
  2. scene5b_three_stacks_v2.py        → frames_three_stacks_v2/
  3. gen_channel_to_stitched.py         → frames_channel_to_stitched_v2/

Output: frames_scene5b_combined/ (sequential frames for the full scene 5b)

Structure:
  - Frames 1-6:   Hold frame (last frame of scene 5, with italic IN VIVO / EX VIVO caption)
  - Frames 7+:    multi_tile_3d frames
  - Then:          three_stacks_v2 frames
  - Then:          channel_to_stitched_v2 frames

Usage:
  python scene5b_combined_render.py              # render all 3 phases + assemble
  python scene5b_combined_render.py --assemble   # skip rendering, just assemble from existing frames
  python scene5b_combined_render.py --phase 1    # render only phase 1 (multi_tile_3d)
  python scene5b_combined_render.py --phase 2    # render only phase 2 (three_stacks_v2)
  python scene5b_combined_render.py --phase 3    # render only phase 3 (channel_to_stitched)
"""

import subprocess, sys, os, shutil, time, argparse
import cv2
import numpy as np

BASE = '/Users/neurolab/neuroinformatics/margaret/animation'
OUT_DIR = f'{BASE}/frames_scene5b_combined'

# Hold frame source: last frame of scene 5 (from merged video)
# This is generated separately and cached
HOLD_FRAME_PATH = f'{BASE}/scene5_last_frame_fixed.png'

PHASE_SCRIPTS = {
    1: ('scene5b_multi_tile_3d_parallel.py', 'frames_multi_tile_3d'),
    2: ('scene5b_three_stacks_v2.py', 'frames_three_stacks_v2'),
    3: ('gen_channel_to_stitched.py', 'frames_channel_to_stitched_v2'),
}

N_HOLD_FRAMES = 6  # hold frames at start


def generate_hold_frame():
    """Generate the hold frame with italic IN VIVO / EX VIVO caption."""
    if os.path.exists(HOLD_FRAME_PATH):
        print(f"  Hold frame exists: {HOLD_FRAME_PATH}")
        return

    # Try to extract from merged_scenes_1-3-5.mp4
    merged = f'{BASE}/merged_scenes_1-3-5.mp4'
    if os.path.exists(merged):
        cap = cv2.VideoCapture(merged)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
        ret, frame = cap.read()
        cap.release()
        if ret:
            # Fix caption: black out old text area and redraw with italic
            sys.path.insert(0, BASE)
            from text_utils import put_text_mixed, text_width_mixed

            FONT = cv2.FONT_HERSHEY_SIMPLEX
            WHITE = (255, 255, 255)
            W, H = 1920, 1080

            # Black out caption area
            frame[1010:1055, :] = 0

            # Redraw with italic
            caption_text = 'GREEN = IN VIVO    MAGENTA = EX VIVO'
            ts = 0.55
            tw = text_width_mixed(caption_text, ts)
            tx = (W - tw) // 2
            put_text_mixed(frame, caption_text, (tx, 1038), FONT, ts, WHITE, 1)

            cv2.imwrite(HOLD_FRAME_PATH, frame)
            print(f"  Generated hold frame: {HOLD_FRAME_PATH}")
            return

    print("  WARNING: Could not generate hold frame. No merged_scenes_1-3-5.mp4 found.")
    print("  The first 6 frames will be skipped.")


def run_phase(phase_num):
    """Run a rendering phase sub-script."""
    script, out_folder = PHASE_SCRIPTS[phase_num]
    script_path = f'{BASE}/{script}'
    print(f"\n{'='*60}")
    print(f"Phase {phase_num}: {script}")
    print(f"  Output: {out_folder}/")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE,
        capture_output=False,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR: Phase {phase_num} failed (exit code {result.returncode})")
        sys.exit(1)

    n_frames = len([f for f in os.listdir(f'{BASE}/{out_folder}') if f.endswith('.png')])
    print(f"  Phase {phase_num} done: {n_frames} frames in {elapsed:.0f}s")
    return n_frames


def assemble():
    """Assemble all phase outputs into frames_scene5b_combined/."""
    print(f"\n{'='*60}")
    print("Assembling combined output")
    print(f"{'='*60}")

    # Ensure output dir
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    idx = 1

    # Hold frames (1-6)
    if os.path.exists(HOLD_FRAME_PATH):
        for i in range(N_HOLD_FRAMES):
            shutil.copy2(HOLD_FRAME_PATH, f'{OUT_DIR}/frame_{idx:05d}.png')
            idx += 1
        print(f"  Hold frames: {N_HOLD_FRAMES} (frames 1-{N_HOLD_FRAMES})")
    else:
        print("  WARNING: No hold frame, skipping")

    # Phase 1: multi_tile_3d
    phase1_dir = f'{BASE}/{PHASE_SCRIPTS[1][1]}'
    if os.path.exists(phase1_dir):
        frames = sorted(f for f in os.listdir(phase1_dir) if f.endswith('.png'))
        start = idx
        for f in frames:
            shutil.copy2(f'{phase1_dir}/{f}', f'{OUT_DIR}/frame_{idx:05d}.png')
            idx += 1
        print(f"  multi_tile_3d: {len(frames)} frames ({start}-{idx-1})")

    # Phase 2: three_stacks_v2
    phase2_dir = f'{BASE}/{PHASE_SCRIPTS[2][1]}'
    if os.path.exists(phase2_dir):
        frames = sorted(f for f in os.listdir(phase2_dir) if f.endswith('.png'))
        start = idx
        for f in frames:
            shutil.copy2(f'{phase2_dir}/{f}', f'{OUT_DIR}/frame_{idx:05d}.png')
            idx += 1
        print(f"  three_stacks_v2: {len(frames)} frames ({start}-{idx-1})")

    # Phase 3: channel_to_stitched
    phase3_dir = f'{BASE}/{PHASE_SCRIPTS[3][1]}'
    if os.path.exists(phase3_dir):
        frames = sorted(f for f in os.listdir(phase3_dir) if f.endswith('.png'))
        start = idx
        for f in frames:
            shutil.copy2(f'{phase3_dir}/{f}', f'{OUT_DIR}/frame_{idx:05d}.png')
            idx += 1
        print(f"  channel_to_stitched: {len(frames)} frames ({start}-{idx-1})")

    total = idx - 1
    print(f"\n  Total: {total} frames = {total/24:.1f}s @ 24fps")
    print(f"  Output: {OUT_DIR}/")
    return total


def main():
    parser = argparse.ArgumentParser(description='Scene 5b Combined Renderer')
    parser.add_argument('--assemble', action='store_true',
                        help='Skip rendering, just assemble from existing frame folders')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                        help='Render only a specific phase (1=multi_tile, 2=three_stacks, 3=channel_to_stitched)')
    args = parser.parse_args()

    t0 = time.time()

    if not args.assemble:
        # Generate hold frame
        print("Generating hold frame...")
        generate_hold_frame()

        if args.phase:
            # Render single phase
            run_phase(args.phase)
        else:
            # Render all phases
            for phase in [1, 2, 3]:
                run_phase(phase)

    # Assemble
    total = assemble()

    elapsed = time.time() - t0
    print(f"\nAll done! {total} frames in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
