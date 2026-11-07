#!/bin/bash
# ============================================================
# MANUAL DOWNLOAD HELPER
# ============================================================
# Use this for clips that failed automatic download.
#
# Usage:
#   ./manual_download.sh CLIP_ID YOUTUBE_URL START END
#
# Example:
#   ./manual_download.sh HAP_01 "https://www.youtube.com/watch?v=abc123" 00:30 01:00
# ============================================================

if [ $# -lt 4 ]; then
    echo "Usage: $0 CLIP_ID YOUTUBE_URL START END"
    echo "Example: $0 HAP_01 \"https://www.youtube.com/watch?v=abc123\" 00:30 01:00"
    exit 1
fi

ID="$1"
URL="$2"
START="$3"
END="$4"

RAW_DIR="raw_stimuli"
NORM_DIR="stimuli"

mkdir -p "$RAW_DIR" "$NORM_DIR"

echo "Downloading $ID from $URL ($START → $END)..."

yt-dlp \
    -f "bestvideo[height>=720]+bestaudio/best[height>=720]" \
    --download-sections "*${START}-${END}" \
    --merge-output-format mp4 \
    --no-playlist \
    -o "${RAW_DIR}/${ID}_raw.mp4" \
    "$URL"

if [ $? -eq 0 ]; then
    echo "✓ Downloaded. Normalizing..."
    ffmpeg -y -i "${RAW_DIR}/${ID}_raw.mp4" \
        -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black" \
        -r 30 -c:v libx264 -preset medium -crf 18 \
        -c:a aac -b:a 128k -ac 2 -ar 44100 \
        -movflags +faststart \
        "${NORM_DIR}/${ID}.mp4"
    echo "✓ Done! Normalized clip: ${NORM_DIR}/${ID}.mp4"
else
    echo "✗ Download failed. Check the URL and try again."
fi
