#!/bin/bash
# ============================================================
# STIMULUS VIDEO DOWNLOADER — FER Validation Experiment
# ============================================================
# Run this on YOUR laptop (not in Cowork) where YouTube is accessible.
#
# Prerequisites:
#   pip install yt-dlp
#   brew install ffmpeg   (Mac)  or  sudo apt install ffmpeg (Linux)
#
# Usage:
#   chmod +x download_all_stimuli.sh
#   ./download_all_stimuli.sh
#
# The script will:
#   1. Download + trim each clip into raw_stimuli/
#   2. Normalize all clips to 1280×720 @ 30fps into stimuli/
#   3. Print a summary of successes/failures
# ============================================================

set -euo pipefail

RAW_DIR="raw_stimuli"
NORM_DIR="stimuli"
LOG_FILE="download_log.txt"
FAILED_FILE="failed_downloads.txt"

mkdir -p "$RAW_DIR" "$NORM_DIR"
> "$LOG_FILE"
> "$FAILED_FILE"

# Counters
SUCCESS=0
FAIL=0

# ──────────────────────────────────────────────
# Helper: download + trim a clip
# Args: CLIP_ID  SEARCH_QUERY  START  END  DESCRIPTION
# ──────────────────────────────────────────────
download_clip() {
    local ID="$1"
    local QUERY="$2"
    local START="$3"
    local END="$4"
    local DESC="$5"
    local OUTFILE="${RAW_DIR}/${ID}_raw.mp4"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Downloading: $ID — $DESC"
    echo "  Search:      $QUERY"
    echo "  Trim:        $START → $END"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ -f "$OUTFILE" ]; then
        echo "  ✓ Already exists, skipping."
        echo "$ID|SKIPPED|already exists" >> "$LOG_FILE"
        SUCCESS=$((SUCCESS + 1))
        return 0
    fi

    # Try ytsearch first, fall back to ytsearch3 (picks from top 3)
    if yt-dlp \
        -f "bestvideo[height>=720]+bestaudio/best[height>=720]" \
        --download-sections "*${START}-${END}" \
        --merge-output-format mp4 \
        --no-playlist \
        -o "$OUTFILE" \
        "ytsearch:${QUERY}" 2>&1 | tee -a "$LOG_FILE"; then
        echo "  ✓ Downloaded: $OUTFILE"
        echo "$ID|SUCCESS|$QUERY" >> "$LOG_FILE"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  ✗ FAILED: $ID"
        echo "$ID|FAILED|$QUERY" >> "$FAILED_FILE"
        echo "$ID|FAILED|$QUERY — try manual search: https://www.youtube.com/results?search_query=$(echo "$QUERY" | tr ' ' '+')" >> "$LOG_FILE"
        FAIL=$((FAIL + 1))
    fi
}

# ──────────────────────────────────────────────
# Helper: normalize a clip
# Args: CLIP_ID
# ──────────────────────────────────────────────
normalize_clip() {
    local ID="$1"
    local INFILE="${RAW_DIR}/${ID}_raw.mp4"
    local OUTFILE="${NORM_DIR}/${ID}.mp4"

    if [ ! -f "$INFILE" ]; then
        echo "  ⚠ Raw file missing for $ID, skipping normalization."
        return 1
    fi

    if [ -f "$OUTFILE" ]; then
        echo "  ✓ $ID already normalized."
        return 0
    fi

    echo "  Normalizing $ID..."
    ffmpeg -y -i "$INFILE" \
        -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black" \
        -r 30 -c:v libx264 -preset medium -crf 18 \
        -c:a aac -b:a 128k -ac 2 -ar 44100 \
        -movflags +faststart \
        "$OUTFILE" 2>/dev/null

    echo "  ✓ Normalized: $OUTFILE"
}


echo "╔══════════════════════════════════════════════╗"
echo "║   DOWNLOADING 50 STIMULUS CLIPS              ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ============================================================
# HAPPY — 10 clips (trim to 20–45s of funniest moment)
# ============================================================
echo ""
echo "▶ HAPPY CLIPS (HAP_01 – HAP_10)"
echo "────────────────────────────────"

download_clip "HAP_01" "Tom and Jerry funniest chase scene WB Kids official" \
    "00:30" "01:00" "Tom & Jerry slapstick chase"

download_clip "HAP_02" "SpongeBob SquarePants funniest moments Nickelodeon official" \
    "00:20" "00:50" "SpongeBob iconic funny moment"

download_clip "HAP_03" "Road Runner Wile E Coyote ACME fail Looney Tunes" \
    "00:15" "00:45" "Wile E. Coyote ACME backfire"

download_clip "HAP_04" "Oggy and the Cockroaches funniest moments compilation" \
    "00:30" "01:00" "Oggy slapstick gag"

download_clip "HAP_05" "Mr Bean animated funniest moments official" \
    "00:20" "00:50" "Mr Bean physical comedy"

download_clip "HAP_06" "funniest cats compilation try not to laugh" \
    "00:10" "00:40" "Cats doing silly things"

download_clip "HAP_07" "funniest dogs compilation fails zoomies" \
    "00:10" "00:40" "Dog fails and zoomies"

download_clip "HAP_08" "Phineas and Ferb Perry the Platypus funniest moments" \
    "00:30" "01:00" "Perry reveal / Doofenshmirtz fail"

download_clip "HAP_09" "Shaun the Sheep funniest moments slapstick" \
    "00:20" "00:50" "Shaun the Sheep silent slapstick"

download_clip "HAP_10" "try not to laugh clean compilation harmless fails" \
    "00:15" "00:45" "Harmless funny accidents"


# ============================================================
# SAD — 10 clips (trim to 30–60s of saddest moment)
# ============================================================
echo ""
echo "▶ SAD CLIPS (SAD_01 – SAD_10)"
echo "──────────────────────────────"

download_clip "SAD_01" "Up married life scene Pixar Carl Ellie" \
    "03:00" "04:00" "Up — Carl alone after Ellie's death"

download_clip "SAD_02" "Finding Nemo barracuda attack opening scene" \
    "00:30" "01:20" "Finding Nemo — barracuda attack"

download_clip "SAD_03" "Mufasa death scene Lion King Simba" \
    "01:30" "02:20" "Lion King — Mufasa's death"

download_clip "SAD_04" "Toy Story 3 incinerator scene holding hands" \
    "01:00" "01:50" "Toy Story 3 — toys hold hands"

download_clip "SAD_05" "Changing Batteries animated short film" \
    "04:00" "05:00" "Changing Batteries — ending"

download_clip "SAD_06" "Coco Remember Me Mama Coco scene Pixar" \
    "01:30" "02:20" "Coco — Mama Coco remembers"

download_clip "SAD_07" "Trois Petits Chats animated short film" \
    "03:00" "04:00" "Trois Petits Chats — ending"

download_clip "SAD_08" "Futurama Jurassic Bark Seymour waiting ending" \
    "01:00" "02:00" "Futurama — Seymour waiting for Fry"

download_clip "SAD_09" "Iron Giant Superman sacrifice scene" \
    "01:00" "01:50" "Iron Giant — Superman sacrifice"

download_clip "SAD_10" "Inside Out Bing Bong fading take her to the moon" \
    "01:00" "01:50" "Inside Out — Bing Bong fading"


# ============================================================
# ANGRY — 10 clips (trim to 20–40s of most anger-inducing)
# ============================================================
echo ""
echo "▶ ANGRY CLIPS (ANG_01 – ANG_10)"
echo "─────────────────────────────────"

download_clip "ANG_01" "worst road rage dashcam compilation dangerous driving" \
    "00:20" "00:50" "Dashcam road rage"

download_clip "ANG_02" "bystander stops bully caught on camera" \
    "00:30" "01:00" "Bullying with bystander intervention"

download_clip "ANG_03" "gaming rage compilation funny controller smash" \
    "00:15" "00:45" "Gaming rage moment"

download_clip "ANG_04" "Scar long live the king Lion King betrayal scene" \
    "00:30" "01:00" "Scar's betrayal — Long live the king"

download_clip "ANG_05" "worst referee decisions football soccer" \
    "00:20" "00:50" "Terrible referee call"

download_clip "ANG_06" "Lotso betrayal Toy Story 3 furnace scene" \
    "00:30" "01:00" "Lotso leaving toys to die"

download_clip "ANG_07" "prank gone too far destroying stuff reaction" \
    "00:20" "00:50" "Prank destroying belongings"

download_clip "ANG_08" "cheater caught gaming esports compilation" \
    "00:20" "00:50" "Cheater caught in competitive gaming"

download_clip "ANG_09" "Matilda Trunchbull bullying children scene" \
    "00:30" "01:00" "Trunchbull bullying children"

download_clip "ANG_10" "cutting in line social experiment reaction" \
    "00:30" "01:00" "Queue-cutting social experiment"


# ============================================================
# SURPRISED — 10 clips (trim to 15–25s around surprise/reveal)
# ============================================================
echo ""
echo "▶ SURPRISED CLIPS (SUR_01 – SUR_10)"
echo "─────────────────────────────────────"

download_clip "SUR_01" "best Rube Goldberg machine chain reaction finale" \
    "02:00" "02:25" "Rube Goldberg unexpected finale"

download_clip "SUR_02" "optical illusion reveal not what you think mind blowing" \
    "00:10" "00:30" "Optical illusion reveal"

download_clip "SUR_03" "Zach King best magic vines compilation object transformation" \
    "00:05" "00:25" "Zach King magic trick"

download_clip "SUR_04" "Zach King impossible tricks walking into poster" \
    "00:05" "00:25" "Zach King impossible trick"

download_clip "SUR_05" "wait for it animal video unexpected funny" \
    "00:10" "00:30" "Animal does something unexpected"

download_clip "SUR_06" "most unexpected sports moments goals catches impossible" \
    "00:10" "00:30" "Impossible sports moment"

download_clip "SUR_07" "surprise reaction hidden camera wholesome emotional" \
    "00:30" "00:50" "Wholesome surprise reaction"

download_clip "SUR_08" "animated short film unexpected plot twist ending" \
    "02:00" "02:25" "Animated short with plot twist"

download_clip "SUR_09" "science experiment unexpected result amazing" \
    "00:15" "00:35" "Science experiment surprise"

download_clip "SUR_10" "street performer unexpected talent reveal amazing" \
    "00:30" "00:50" "Street performer hidden talent"


# ============================================================
# NEUTRAL — 10 clips (trim to 15–25s, calm and boring)
# ============================================================
echo ""
echo "▶ NEUTRAL CLIPS (NEU_01 – NEU_10)"
echo "───────────────────────────────────"

download_clip "NEU_01" "drone footage farmland aerial peaceful" \
    "00:15" "00:40" "Drone footage of farmland"

download_clip "NEU_02" "cloud timelapse city skyline calm" \
    "00:10" "00:35" "Cloud timelapse over city"

download_clip "NEU_03" "how to fold towels tutorial simple" \
    "00:30" "00:55" "Folding towels"

download_clip "NEU_04" "highway traffic footage calm driving" \
    "00:15" "00:40" "Highway traffic"

download_clip "NEU_05" "pond fish swimming peaceful footage" \
    "00:10" "00:35" "Fish swimming in pond"

download_clip "NEU_06" "bread baking kneading dough close up" \
    "00:20" "00:45" "Kneading bread dough"

download_clip "NEU_07" "cake decorating satisfying no talking calm" \
    "00:30" "00:55" "Cake decorating"

download_clip "NEU_08" "weather forecast UK clip BBC" \
    "00:10" "00:35" "Weather forecast"

download_clip "NEU_09" "typing keyboard ASMR footage" \
    "00:10" "00:35" "Typing on keyboard"

download_clip "NEU_10" "walking POV city street footage calm" \
    "00:15" "00:40" "Walking down city street"


# ============================================================
# NORMALIZE ALL CLIPS
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   NORMALIZING ALL CLIPS (1280×720 @ 30fps)   ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

for RAW in "$RAW_DIR"/*_raw.mp4; do
    [ -f "$RAW" ] || continue
    BASENAME=$(basename "$RAW" _raw.mp4)
    normalize_clip "$BASENAME"
done


# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   DOWNLOAD SUMMARY                            ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  ✓ Succeeded: $SUCCESS / 50"
echo "  ✗ Failed:    $FAIL / 50"
echo ""

if [ -s "$FAILED_FILE" ]; then
    echo "  Failed clips (search manually on YouTube):"
    while IFS='|' read -r id status query; do
        echo "    - $id: search for \"$query\""
    done < "$FAILED_FILE"
    echo ""
    echo "  For failed clips, find the YouTube URL manually and run:"
    echo '    yt-dlp -f "bestvideo[height>=720]+bestaudio/best[height>=720]" \'
    echo '      --download-sections "*START-END" --merge-output-format mp4 \'
    echo "      -o \"raw_stimuli/CLIP_ID_raw.mp4\" \"YOUTUBE_URL\""
    echo ""
fi

echo "  Raw clips:        $RAW_DIR/"
echo "  Normalized clips:  $NORM_DIR/"
echo "  Download log:      $LOG_FILE"
echo ""
echo "  Done! Review the clips and re-trim any that need adjustment."
