#!/usr/bin/env python3
"""
test_stimuli.py — Preview all stimuli one by one.

Plays each clip for a few seconds so you can verify it works.
Press SPACE to skip to next clip, ESCAPE to quit.

Usage:
    python test_stimuli.py
    python test_stimuli.py --full        (play entire clips, not just preview)
    python test_stimuli.py --category SAD  (only test SAD clips)
"""

import sys
import csv
import argparse
from pathlib import Path

try:
    from psychopy import visual, core, event, sound
except ImportError:
    print("ERROR: PsychoPy not installed. Run: pip install psychopy")
    sys.exit(1)

STIMULI_DIR = Path("stimuli_new")
STIMULUS_CSV = Path("stimulus_list.csv")
PREVIEW_DURATION = 5.0  # seconds per clip in preview mode
IMAGE_DURATION = 5.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Play entire clips instead of preview")
    parser.add_argument("--category", type=str, default=None,
                        help="Only test one category: HAP, SAD, ANG, SUR, NEU")
    args = parser.parse_args()

    # Load stimulus list
    stimuli = []
    with open(STIMULUS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stim_id = row["stimulus_id"]
            # Find the file (mp4, jpg, or png)
            filepath = None
            for ext in [".mp4", ".jpg", ".png"]:
                candidate = STIMULI_DIR / f"{stim_id}{ext}"
                if candidate.exists():
                    filepath = candidate
                    break
            if filepath is None:
                print(f"  MISSING: {stim_id}")
                continue
            row["filepath"] = filepath
            row["is_image"] = filepath.suffix.lower() in [".jpg", ".jpeg", ".png"]
            stimuli.append(row)

    # Filter by category if requested
    if args.category:
        cat = args.category.upper()
        stimuli = [s for s in stimuli if s["stimulus_id"].startswith(cat)]
        if not stimuli:
            print(f"No stimuli found for category '{cat}'")
            sys.exit(1)

    print(f"\nTesting {len(stimuli)} stimuli")
    print(f"Mode: {'FULL playback' if args.full else f'PREVIEW ({PREVIEW_DURATION}s each)'}")
    print(f"Controls: SPACE = next clip, ESCAPE = quit\n")

    # Create window
    win = visual.Window(
        size=(1280, 720),
        fullscr=False,
        color="black",
        units="norm",
        winType="pyglet"
    )

    info_text = visual.TextStim(win, height=0.06, color="white",
                                wrapWidth=1.6, pos=(0, 0))

    passed = 0
    failed = 0
    skipped = 0

    for i, stim in enumerate(stimuli):
        stim_id = stim["stimulus_id"]
        emotion = stim["target_emotion"]
        filepath = stim["filepath"]
        is_image = stim["is_image"]

        # Show info card first
        info_text.text = (
            f"[{i+1}/{len(stimuli)}]  {stim_id}\n\n"
            f"Emotion: {emotion}\n"
            f"File: {filepath.name}\n"
            f"Type: {'IMAGE' if is_image else 'VIDEO'}\n\n"
            f"Press SPACE to play, ESCAPE to quit"
        )
        info_text.draw()
        win.flip()

        # Wait for keypress
        keys = event.waitKeys(keyList=["space", "escape"])
        if "escape" in keys:
            print(f"  Quit at {stim_id}")
            break

        # Play the stimulus
        try:
            if is_image:
                img = visual.ImageStim(win, image=str(filepath))
                timer = core.Clock()
                while timer.getTime() < IMAGE_DURATION:
                    img.draw()
                    # Show overlay
                    overlay = visual.TextStim(
                        win, text=f"{stim_id} (IMAGE) - SPACE to skip",
                        pos=(0, -0.9), height=0.04, color="yellow"
                    )
                    overlay.draw()
                    win.flip()
                    keys = event.getKeys(keyList=["space", "escape"])
                    if "space" in keys:
                        break
                    if "escape" in keys:
                        win.close()
                        print_summary(passed, failed, skipped, len(stimuli))
                        return
                passed += 1
                status = "OK"

            else:
                mov = visual.MovieStim(
                    win,
                    filename=str(filepath),
                    flipVert=False,
                    loop=False,
                    noAudio=False
                )

                timer = core.Clock()
                while mov.status != visual.FINISHED:
                    mov.draw()
                    # Show overlay
                    elapsed = timer.getTime()
                    overlay = visual.TextStim(
                        win,
                        text=f"{stim_id} ({elapsed:.1f}s) - SPACE to skip",
                        pos=(0, -0.9), height=0.04, color="yellow"
                    )
                    overlay.draw()
                    win.flip()

                    keys = event.getKeys(keyList=["space", "escape"])
                    if "space" in keys:
                        mov.stop()
                        break
                    if "escape" in keys:
                        mov.stop()
                        win.close()
                        print_summary(passed, failed, skipped, len(stimuli))
                        return

                    # In preview mode, stop after PREVIEW_DURATION
                    if not args.full and elapsed >= PREVIEW_DURATION:
                        mov.stop()
                        break

                passed += 1
                status = "OK"

        except Exception as e:
            failed += 1
            status = f"FAILED: {e}"
            # Show error briefly
            info_text.text = f"{stim_id}\n\nERROR: {e}\n\nPress SPACE to continue"
            info_text.draw()
            win.flip()
            event.waitKeys(keyList=["space", "escape"])

        print(f"  [{i+1:2d}/{len(stimuli)}] {stim_id:8s} {emotion:10s} {status}")

    win.close()
    print_summary(passed, failed, skipped, len(stimuli))


def print_summary(passed, failed, skipped, total):
    print(f"\n{'='*40}")
    print(f"  RESULTS: {passed} OK / {failed} FAILED / {total} total")
    if failed == 0:
        print(f"  All stimuli working!")
    else:
        print(f"  {failed} stimuli need fixing.")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
