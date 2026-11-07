# LUMI FER Validation Experiment — Setup Guide

## Overview

This experiment tests the LUMI facial expression recognition system with 20 human participants. Each participant views 50 stimuli (10 per emotion) in a randomised order, with the FER system classifying their facial expressions in real time.

**Session duration:** ~40–45 minutes per participant  
**Stimulus phase:** ~27 minutes (50 stimuli + washout periods)


## Quick Start (Step by Step)

### Step 1: Install dependencies

```bash
pip install psychopy opencv-python pandas
```

Also install FFmpeg for stimulus preparation:
- **Windows:** Download from https://ffmpeg.org/download.html and add to PATH
- **Mac:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`


### Step 2: Download the stimulus databases

You need files from **three** sources:

| Database | What you need | Where to get it | Emotions covered |
|----------|--------------|-----------------|------------------|
| **FilmStim** | 70 film clips + ratings spreadsheet | https://sites.uclouvain.be/ipsp/FilmStim/ | Happy (10), Sad (10), Angry (5), Neutral (5) |
| **NAPS BE** | 510 emotion-rated images | https://lobi.nencki.edu.pl/research/8/ | Angry (5 images), Neutral (5 images) |
| **MagicCATs** | Magic trick video clips | Search https://osf.io/ for "MagicCATs" (Ozono et al.) | Surprised (10) |

**NAPS BE requires an academic request** — submit it immediately using your QMUL email. Mention:
> "Validation experiment for a facial expression recognition system developed as part of a BEng Biomedical Engineering final year project (LUMI social robot for paediatric clinics) at Queen Mary University of London."


### Step 3: Select and trim your stimuli

#### FilmStim clips
1. Open the FilmStim ratings spreadsheet
2. For each emotion, sort by **discreteness score** (highest = purest emotion)
3. Pick the top clips per emotion:
   - **Happy:** Top 10 amusement clips (look for physical comedy, animal humour)
   - **Sad:** Top 10 sadness clips (the "Champ" funeral scene is a classic)
   - **Angry:** Top 5 anger clips
   - **Neutral:** 5 clips closest to midpoint valence + low arousal (landscapes, weather)
4. Note down the emotional peak in each clip (e.g., "laugh at 1:32–2:05")
5. Trim using FFmpeg:
   ```bash
   # Example: extract 30 seconds starting at 1:32
   ffmpeg -i original_clip.mp4 -ss 00:01:32 -t 00:00:30 -c copy HAP_01_raw.mp4
   ```

#### NAPS BE images
1. Once you receive access, open the ratings spreadsheet
2. **Angry images (5):** Sort by anger rating, pick the 5 with highest anger + lowest blend
3. **Neutral images (5):** Sort by valence nearest midpoint + lowest arousal (household objects, landscapes)
4. Rename to `ANG_06_raw.jpg` through `ANG_10_raw.jpg` and `NEU_06_raw.jpg` through `NEU_10_raw.jpg`

#### MagicCATs clips
1. Download from OSF
2. Pick 10 clips with the highest surprise ratings
3. Trim to 15–25 seconds around the reveal moment
4. Rename to `SUR_01_raw.mp4` through `SUR_10_raw.mp4`

**Fallback for MagicCATs:** If you can't access the dataset, source 10 magic trick reveal clips from YouTube. Note this as a limitation in your report.


### Step 4: Place raw files in the correct folders

```
raw_stimuli/
├── filmstim/
│   ├── HAP_01_raw.mp4  ...  HAP_10_raw.mp4
│   ├── SAD_01_raw.mp4  ...  SAD_10_raw.mp4
│   ├── ANG_01_raw.mp4  ...  ANG_05_raw.mp4
│   └── NEU_01_raw.mp4  ...  NEU_05_raw.mp4
├── naps_be/
│   ├── ANG_06_raw.jpg  ...  ANG_10_raw.jpg
│   └── NEU_06_raw.jpg  ...  NEU_10_raw.jpg
└── magiccats/
    └── SUR_01_raw.mp4  ...  SUR_10_raw.mp4
```


### Step 5: Run the preparation script

```bash
chmod +x 01_prepare_stimuli.sh
./01_prepare_stimuli.sh
```

This normalises all clips to 1280×720 @ 30fps and converts images to 8-second video clips. Check the output — it will tell you if anything is missing.


### Step 6: Update the stimulus list CSV

Open `stimulus_list.csv` and fill in:
- `original_id` — the clip's original ID in its source database
- `duration_sec` — actual duration after trimming
- `discreteness_score` — from the database's ratings spreadsheet
- `notes` — brief description of the clip content

PsychoPy reads this file to know what to present.


### Step 7: Run the experiment

```bash
python 02_run_experiment.py
```

The script will:
1. Show a demographics dialog (participant ID, age, gender, Fitzpatrick skin type, etc.)
2. Display instructions
3. Record a 30-second neutral baseline
4. Present all 50 stimuli in randomised order with:
   - 10s washout (fixation cross) between each trial
   - 440 Hz beep at stimulus onset
   - Per-trial self-report (emotion + intensity)
5. Save all results automatically


## Output Files

After each session, you'll find:

```
results/
├── P001_demographics.csv       # participant info
├── P001_stimulus_order.csv     # randomised presentation order
└── P001_trial_results.csv      # per-trial results (self-report + timestamps)

data/
└── raw_video/
    └── P001/
        ├── P001_HAP_03.avi     # webcam recording during HAP_03
        ├── P001_SAD_07.avi     # etc.
        └── ...
```

The `_trial_results.csv` contains everything you need for analysis:
- `target_emotion` — what the stimulus was supposed to elicit
- `self_report_emotion` — what the participant said they felt
- `self_report_intensity` — how strongly (1–5)
- `onset_unix` / `offset_unix` — precise timestamps for syncing with FER logs
- `correct_match` — 1 if self-report matched target, 0 if not


## Running your FER system alongside

The PsychoPy script handles stimulus presentation and self-reports. Your FER system should run as a **separate process** reading from the same webcam (or a second camera). To sync:

1. Both systems log Unix timestamps
2. Use `onset_unix` and `offset_unix` from the trial results to slice the FER predictions for each trial
3. Compare FER predictions during each trial window against the `target_emotion`

If your FER system needs exclusive camera access, you can:
- Use a second USB webcam (one for PsychoPy recording, one for FER)
- Or disable PsychoPy's webcam recording (set `HAS_OPENCV = False` at the top of the script) and let only the FER system use the camera


## Troubleshooting

**PsychoPy can't play videos:** Make sure you have the right video backend. Try:
```bash
pip install python-vlc
```
Or use `MovieStim3` instead of `MovieStim` in the script (edit line ~262).

**No sound:** PsychoPy needs a working audio backend. On Linux: `sudo apt install pulseaudio`

**Webcam conflict:** If both PsychoPy and your FER system need the camera, use two cameras or disable PsychoPy recording.

**NAPS BE taking too long:** Start with FilmStim + MagicCATs first (40 stimuli). Run initial participants with those while waiting for NAPS BE. Add the 10 image-based stimuli later — PsychoPy will simply skip missing files if you tell it to continue.
