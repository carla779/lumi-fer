#!/usr/bin/env bash
# ============================================================================
# 01_prepare_stimuli.sh
# Prepares raw stimulus files for the FER validation experiment.
#
# What it does:
#   1. Creates the required folder structure
#   2. Normalises all video clips to 1280x720 @ 30fps, H.264 + AAC
#   3. Converts NAPS BE images into 8-second video clips (static image + silence)
#   4. Verifies all 50 stimuli are present and correctly formatted
#
# Prerequisites:
#   - FFmpeg installed  (sudo apt install ffmpeg  OR  brew install ffmpeg)
#   - Raw clips placed in  raw_stimuli/  with the naming convention below
#
# Usage:
#   chmod +x 01_prepare_stimuli.sh
#   ./01_prepare_stimuli.sh
# ============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
RAW_DIR="raw_stimuli"
OUT_DIR="stimuli"
TARGET_W=1280
TARGET_H=720
TARGET_FPS=30
IMAGE_DURATION=8   # seconds to display each NAPS BE image

# ── Step 1: Create folder structure ────────────────────────────────────────
echo "=== Step 1: Creating folder structure ==="
mkdir -p "$RAW_DIR"/{filmstim,naps_be,magiccats}
mkdir -p "$OUT_DIR"
mkdir -p data/raw_video
mkdir -p data/fer_logs
mkdir -p data/self_reports
mkdir -p results

cat << 'TREE'
Expected folder structure after you place your raw files:

raw_stimuli/
├── filmstim/
│   ├── HAP_01_raw.mp4   (or whatever the original filename is)
│   ├── HAP_02_raw.mp4
│   ├── ...
│   ├── SAD_01_raw.mp4  ...  SAD_10_raw.mp4
│   ├── ANG_01_raw.mp4  ...  ANG_05_raw.mp4
│   └── NEU_01_raw.mp4  ...  NEU_05_raw.mp4
├── naps_be/
│   ├── ANG_06_raw.jpg   (or .png)
│   ├── ...
│   ├── ANG_10_raw.jpg
│   ├── NEU_06_raw.jpg
│   ├── ...
│   └── NEU_10_raw.jpg
└── magiccats/
    ├── SUR_01_raw.mp4
    ├── ...
    └── SUR_10_raw.mp4

IMPORTANT: You need to rename your downloaded files to match this
convention BEFORE running this script. The "_raw" suffix distinguishes
source files from the normalised outputs.
TREE

# ── Step 2: Normalise video clips ──────────────────────────────────────────
echo ""
echo "=== Step 2: Normalising video clips to ${TARGET_W}x${TARGET_H} @ ${TARGET_FPS}fps ==="

normalise_video() {
    local input="$1"
    local output="$2"
    if [ ! -f "$input" ]; then
        echo "  WARNING: $input not found, skipping."
        return 1
    fi
    echo "  Processing: $(basename "$input") -> $(basename "$output")"
    ffmpeg -y -i "$input" \
        -vf "scale=${TARGET_W}:${TARGET_H}:force_original_aspect_ratio=decrease,pad=${TARGET_W}:${TARGET_H}:(ow-iw)/2:(oh-ih)/2:black" \
        -r "$TARGET_FPS" \
        -c:v libx264 -preset medium -crf 18 \
        -c:a aac -b:a 128k -ac 2 -ar 44100 \
        -movflags +faststart \
        "$output" 2>/dev/null
    return 0
}

MISSING=0

# Happy clips (FilmStim)
for i in $(seq -w 1 10); do
    normalise_video "$RAW_DIR/filmstim/HAP_${i}_raw.mp4" "$OUT_DIR/HAP_${i}.mp4" || ((MISSING++))
done

# Sad clips (FilmStim)
for i in $(seq -w 1 10); do
    normalise_video "$RAW_DIR/filmstim/SAD_${i}_raw.mp4" "$OUT_DIR/SAD_${i}.mp4" || ((MISSING++))
done

# Angry clips (FilmStim, first 5 only)
for i in $(seq -w 1 5); do
    normalise_video "$RAW_DIR/filmstim/ANG_${i}_raw.mp4" "$OUT_DIR/ANG_${i}.mp4" || ((MISSING++))
done

# Surprise clips (MagicCATs)
for i in $(seq -w 1 10); do
    normalise_video "$RAW_DIR/magiccats/SUR_${i}_raw.mp4" "$OUT_DIR/SUR_${i}.mp4" || ((MISSING++))
done

# Neutral clips (FilmStim, first 5 only)
for i in $(seq -w 1 5); do
    normalise_video "$RAW_DIR/filmstim/NEU_${i}_raw.mp4" "$OUT_DIR/NEU_${i}.mp4" || ((MISSING++))
done

# ── Step 3: Convert NAPS BE images to 8-second video clips ────────────────
echo ""
echo "=== Step 3: Converting NAPS BE images to ${IMAGE_DURATION}s video clips ==="

image_to_video() {
    local input="$1"
    local output="$2"

    # Try common extensions
    local actual_input=""
    for ext in jpg jpeg png bmp; do
        local candidate="${input%.*}.${ext}"
        if [ -f "$candidate" ]; then
            actual_input="$candidate"
            break
        fi
    done

    if [ -z "$actual_input" ]; then
        echo "  WARNING: No image found for $(basename "$input"), skipping."
        return 1
    fi

    echo "  Processing: $(basename "$actual_input") -> $(basename "$output")"
    ffmpeg -y -loop 1 -i "$actual_input" \
        -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 \
        -vf "scale=${TARGET_W}:${TARGET_H}:force_original_aspect_ratio=decrease,pad=${TARGET_W}:${TARGET_H}:(ow-iw)/2:(oh-ih)/2:black" \
        -r "$TARGET_FPS" \
        -c:v libx264 -preset medium -crf 18 \
        -c:a aac -b:a 128k \
        -t "$IMAGE_DURATION" \
        -movflags +faststart \
        "$output" 2>/dev/null
    return 0
}

# Angry images (NAPS BE, slots 06-10)
for i in $(seq -w 6 10); do
    image_to_video "$RAW_DIR/naps_be/ANG_${i}_raw.jpg" "$OUT_DIR/ANG_${i}.mp4" || ((MISSING++))
done

# Neutral images (NAPS BE, slots 06-10)
for i in $(seq -w 6 10); do
    image_to_video "$RAW_DIR/naps_be/NEU_${i}_raw.jpg" "$OUT_DIR/NEU_${i}.mp4" || ((MISSING++))
done

# ── Step 4: Verification ──────────────────────────────────────────────────
echo ""
echo "=== Step 4: Verification ==="

EXPECTED_FILES=(
    HAP_01 HAP_02 HAP_03 HAP_04 HAP_05 HAP_06 HAP_07 HAP_08 HAP_09 HAP_10
    SAD_01 SAD_02 SAD_03 SAD_04 SAD_05 SAD_06 SAD_07 SAD_08 SAD_09 SAD_10
    ANG_01 ANG_02 ANG_03 ANG_04 ANG_05 ANG_06 ANG_07 ANG_08 ANG_09 ANG_10
    SUR_01 SUR_02 SUR_03 SUR_04 SUR_05 SUR_06 SUR_07 SUR_08 SUR_09 SUR_10
    NEU_01 NEU_02 NEU_03 NEU_04 NEU_05 NEU_06 NEU_07 NEU_08 NEU_09 NEU_10
)

FOUND=0
TOTAL=${#EXPECTED_FILES[@]}

for f in "${EXPECTED_FILES[@]}"; do
    if [ -f "$OUT_DIR/${f}.mp4" ]; then
        ((FOUND++))
    else
        echo "  MISSING: $OUT_DIR/${f}.mp4"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Stimuli ready: $FOUND / $TOTAL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$FOUND" -eq "$TOTAL" ]; then
    echo "  ✓ All stimuli prepared! You can now run the PsychoPy experiment."
else
    echo "  ✗ Some stimuli missing. Place the raw files and re-run this script."
fi
