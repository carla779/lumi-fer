# ============================================================
# STIMULUS VIDEO DOWNLOADER — FER Validation Experiment
# ============================================================
# Run this on your laptop in PowerShell.
#
# Prerequisites:
#   pip install yt-dlp
#   Install ffmpeg: https://ffmpeg.org/download.html (add to PATH)
#
# Usage:
#   cd "EMS690U - Integrated Design Project\INDIVIDUAL REPORT\Stimuli"
#   powershell -ExecutionPolicy Bypass -File download_all_stimuli.ps1
# ============================================================

# Auto-add tool paths (yt-dlp, ffmpeg, deno)
$extraPaths = @(
    "$env:LOCALAPPDATA\Programs\Python\Python313\Scripts",
    "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts",
    "$env:LOCALAPPDATA\Programs\Python\Python311\Scripts",
    "$env:USERPROFILE\.deno\bin",
    "$env:LOCALAPPDATA\Microsoft\WinGet\Links",
    "C:\ffmpeg\bin",
    "C:\ProgramData\chocolatey\bin"
)
foreach ($p in $extraPaths) {
    if (Test-Path $p) { $env:PATH += ";$p" }
}

# Also search for ffmpeg in common winget install locations
$ffmpegSearch = Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter 'ffmpeg.exe' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if ($ffmpegSearch) {
    $env:PATH += ";" + $ffmpegSearch.DirectoryName
}

# Verify required tools
$missing = @()
if (-not (Get-Command yt-dlp -ErrorAction SilentlyContinue)) { $missing += 'yt-dlp (py -m pip install yt-dlp)' }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { $missing += 'ffmpeg (winget install Gyan.FFmpeg)' }
if (-not (Get-Command deno -ErrorAction SilentlyContinue)) { $missing += 'deno (winget install DenoLand.Deno) - needed for YouTube extraction' }

if ($missing.Count -gt 0) {
    Write-Host 'Missing required tools:' -ForegroundColor Red
    foreach ($m in $missing) { Write-Host "  - $m" -ForegroundColor Yellow }
    Write-Host ''
    Write-Host 'Install them, then close and reopen PowerShell and re-run this script.' -ForegroundColor Red
    exit 1
}

Write-Host "Tools found: yt-dlp, ffmpeg, deno" -ForegroundColor Green

$ErrorActionPreference = "Continue"

$RAW_DIR = "raw_stimuli"
$NORM_DIR = "stimuli"
$LOG_FILE = "download_log.txt"

New-Item -ItemType Directory -Force -Path $RAW_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $NORM_DIR | Out-Null
"" | Set-Content $LOG_FILE

$success = 0
$fail = 0
$failed_list = @()

function Download-Clip {
    param(
        [string]$Id,
        [string]$Query,
        [string]$Start,
        [string]$End,
        [string]$Desc
    )

    $outfile = "$RAW_DIR\${Id}_raw.mp4"

    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "  $Id - $Desc" -ForegroundColor Cyan
    Write-Host "  Search: $Query" -ForegroundColor Gray
    Write-Host "  Trim:   $Start -> $End" -ForegroundColor Gray
    Write-Host "============================================" -ForegroundColor Cyan

    if (Test-Path $outfile) {
        Write-Host "  Already exists, skipping." -ForegroundColor Green
        $script:success++
        return
    }

    try {
        yt-dlp -f "bestvideo[height>=720]+bestaudio/best[height>=720]" `
            --remote-components ejs:github `
            --download-sections "*${Start}-${End}" `
            --merge-output-format mp4 `
            --no-playlist `
            -o "$outfile" `
            "ytsearch:${Query}" 2>&1 | Tee-Object -Append -FilePath $LOG_FILE

        if (Test-Path $outfile) {
            Write-Host "  Downloaded!" -ForegroundColor Green
            $script:success++
        } else {
            throw "File not created"
        }
    }
    catch {
        Write-Host "  FAILED: $Id" -ForegroundColor Red
        $script:fail++
        $script:failed_list += [PSCustomObject]@{Id=$Id; Query=$Query}
        "$Id|FAILED|$Query" | Add-Content $LOG_FILE
    }
}

function Normalize-Clip {
    param([string]$Id)

    $infile  = "$RAW_DIR\${Id}_raw.mp4"
    $outfile = "$NORM_DIR\${Id}.mp4"

    if (-not (Test-Path $infile)) {
        Write-Host "  Raw file missing for $Id" -ForegroundColor Yellow
        return
    }
    if (Test-Path $outfile) {
        Write-Host "  $Id already normalized." -ForegroundColor Green
        return
    }

    Write-Host "  Normalizing $Id..." -ForegroundColor Gray
    ffmpeg -y -i "$infile" `
        -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black" `
        -r 30 -c:v libx264 -preset medium -crf 18 `
        -c:a aac -b:a 128k -ac 2 -ar 44100 `
        -movflags +faststart `
        "$outfile" 2>$null

    if (Test-Path $outfile) {
        Write-Host "  Normalized: $outfile" -ForegroundColor Green
    }
}

# ============================================================
Write-Host ""
Write-Host "=========================================" -ForegroundColor Yellow
Write-Host "   DOWNLOADING 50 STIMULUS CLIPS" -ForegroundColor Yellow
Write-Host "=========================================" -ForegroundColor Yellow

# ── HAPPY (20-45s of funniest moment) ──
Write-Host "`n>> HAPPY CLIPS" -ForegroundColor Magenta

Download-Clip "HAP_01" "Tom and Jerry funniest chase scene WB Kids official" "00:30" "01:00" "Tom & Jerry slapstick chase"
Download-Clip "HAP_02" "SpongeBob SquarePants funniest moments Nickelodeon official" "00:20" "00:50" "SpongeBob funny moment"
Download-Clip "HAP_03" "Road Runner Wile E Coyote ACME fail Looney Tunes" "00:15" "00:45" "Wile E. Coyote ACME backfire"
Download-Clip "HAP_04" "Oggy and the Cockroaches funniest moments compilation" "00:30" "01:00" "Oggy slapstick gag"
Download-Clip "HAP_05" "Mr Bean animated funniest moments official" "00:20" "00:50" "Mr Bean physical comedy"
Download-Clip "HAP_06" "funniest cats compilation try not to laugh" "00:10" "00:40" "Cats doing silly things"
Download-Clip "HAP_07" "funniest dogs compilation fails zoomies" "00:10" "00:40" "Dog fails and zoomies"
Download-Clip "HAP_08" "Phineas and Ferb Perry the Platypus funniest moments" "00:30" "01:00" "Perry reveal / Doofenshmirtz fail"
Download-Clip "HAP_09" "Shaun the Sheep funniest moments slapstick" "00:20" "00:50" "Shaun the Sheep slapstick"
Download-Clip "HAP_10" "try not to laugh clean compilation harmless fails" "00:15" "00:45" "Harmless funny accidents"

# ── SAD (30-60s of saddest moment) ──
Write-Host "`n>> SAD CLIPS" -ForegroundColor Magenta

Download-Clip "SAD_01" "Up married life scene Pixar Carl Ellie" "03:00" "04:00" "Up - Carl alone after Ellie"
Download-Clip "SAD_02" "Finding Nemo barracuda attack opening scene" "00:30" "01:20" "Finding Nemo barracuda attack"
Download-Clip "SAD_03" "Mufasa death scene Lion King Simba" "01:30" "02:20" "Mufasa death"
Download-Clip "SAD_04" "Toy Story 3 incinerator scene holding hands" "01:00" "01:50" "Toy Story 3 incinerator"
Download-Clip "SAD_05" "Changing Batteries animated short film" "04:00" "05:00" "Changing Batteries ending"
Download-Clip "SAD_06" "Coco Remember Me Mama Coco scene Pixar" "01:30" "02:20" "Coco - Mama Coco remembers"
Download-Clip "SAD_07" "Trois Petits Chats animated short film" "03:00" "04:00" "Trois Petits Chats ending"
Download-Clip "SAD_08" "Futurama Jurassic Bark Seymour waiting ending" "01:00" "02:00" "Futurama - Seymour waiting"
Download-Clip "SAD_09" "Iron Giant Superman sacrifice scene" "01:00" "01:50" "Iron Giant Superman"
Download-Clip "SAD_10" "Inside Out Bing Bong fading take her to the moon" "01:00" "01:50" "Inside Out - Bing Bong fading"

# ── ANGRY (20-40s of most anger-inducing moment) ──
Write-Host "`n>> ANGRY CLIPS" -ForegroundColor Magenta

Download-Clip "ANG_01" "worst road rage dashcam compilation dangerous driving" "00:20" "00:50" "Dashcam road rage"
Download-Clip "ANG_02" "bystander stops bully caught on camera" "00:30" "01:00" "Bullying with bystander"
Download-Clip "ANG_03" "gaming rage compilation funny controller smash" "00:15" "00:45" "Gaming rage"
Download-Clip "ANG_04" "Scar long live the king Lion King betrayal scene" "00:30" "01:00" "Scar betrayal"
Download-Clip "ANG_05" "worst referee decisions football soccer" "00:20" "00:50" "Bad referee call"
Download-Clip "ANG_06" "Lotso betrayal Toy Story 3 furnace scene" "00:30" "01:00" "Lotso betrayal"
Download-Clip "ANG_07" "prank gone too far destroying stuff reaction" "00:20" "00:50" "Prank destroying stuff"
Download-Clip "ANG_08" "cheater caught gaming esports compilation" "00:20" "00:50" "Cheater caught gaming"
Download-Clip "ANG_09" "Matilda Trunchbull bullying children scene" "00:30" "01:00" "Trunchbull scene"
Download-Clip "ANG_10" "cutting in line social experiment reaction" "00:30" "01:00" "Queue cutting"

# ── SURPRISED (15-25s around surprise/reveal) ──
Write-Host "`n>> SURPRISED CLIPS" -ForegroundColor Magenta

Download-Clip "SUR_01" "best Rube Goldberg machine chain reaction finale" "02:00" "02:25" "Rube Goldberg finale"
Download-Clip "SUR_02" "optical illusion reveal not what you think mind blowing" "00:10" "00:30" "Optical illusion reveal"
Download-Clip "SUR_03" "Zach King best magic vines compilation object transformation" "00:05" "00:25" "Zach King magic trick"
Download-Clip "SUR_04" "Zach King impossible tricks walking into poster" "00:05" "00:25" "Zach King impossible trick"
Download-Clip "SUR_05" "wait for it animal video unexpected funny" "00:10" "00:30" "Animal unexpected"
Download-Clip "SUR_06" "most unexpected sports moments goals catches impossible" "00:10" "00:30" "Impossible sports moment"
Download-Clip "SUR_07" "surprise reaction hidden camera wholesome emotional" "00:30" "00:50" "Wholesome surprise"
Download-Clip "SUR_08" "animated short film unexpected plot twist ending" "02:00" "02:25" "Animated short plot twist"
Download-Clip "SUR_09" "science experiment unexpected result amazing" "00:15" "00:35" "Science experiment surprise"
Download-Clip "SUR_10" "street performer unexpected talent reveal amazing" "00:30" "00:50" "Street performer talent"

# ── NEUTRAL (15-25s, calm and boring) ──
Write-Host "`n>> NEUTRAL CLIPS" -ForegroundColor Magenta

Download-Clip "NEU_01" "drone footage farmland aerial peaceful" "00:15" "00:40" "Drone farmland"
Download-Clip "NEU_02" "cloud timelapse city skyline calm" "00:10" "00:35" "Cloud timelapse"
Download-Clip "NEU_03" "how to fold towels tutorial simple" "00:30" "00:55" "Folding towels"
Download-Clip "NEU_04" "highway traffic footage calm driving" "00:15" "00:40" "Highway traffic"
Download-Clip "NEU_05" "pond fish swimming peaceful footage" "00:10" "00:35" "Fish swimming"
Download-Clip "NEU_06" "bread baking kneading dough close up" "00:20" "00:45" "Kneading bread"
Download-Clip "NEU_07" "cake decorating satisfying no talking calm" "00:30" "00:55" "Cake decorating"
Download-Clip "NEU_08" "weather forecast UK clip BBC" "00:10" "00:35" "Weather forecast"
Download-Clip "NEU_09" "typing keyboard ASMR footage" "00:10" "00:35" "Keyboard typing"
Download-Clip "NEU_10" "walking POV city street footage calm" "00:15" "00:40" "Walking POV city"

# ============================================================
# NORMALIZE ALL
# ============================================================
Write-Host ""
Write-Host "=========================================" -ForegroundColor Yellow
Write-Host "   NORMALIZING ALL CLIPS" -ForegroundColor Yellow
Write-Host "=========================================" -ForegroundColor Yellow

Get-ChildItem "$RAW_DIR\*_raw.mp4" | ForEach-Object {
    $clipId = $_.BaseName -replace "_raw$", ""
    Normalize-Clip $clipId
}

# ============================================================
# SUMMARY
# ============================================================
Write-Host ""
Write-Host "=========================================" -ForegroundColor Yellow
Write-Host "   SUMMARY" -ForegroundColor Yellow
Write-Host "=========================================" -ForegroundColor Yellow
Write-Host "  Succeeded: $success / 50" -ForegroundColor Green
Write-Host "  Failed:    $fail / 50" -ForegroundColor Red

if ($failed_list.Count -gt 0) {
    Write-Host ''
    Write-Host '  Failed clips - search manually on YouTube:' -ForegroundColor Red
    foreach ($item in $failed_list) {
        $msg = '    - ' + $item.Id + ': search for ' + $item.Query
        Write-Host $msg -ForegroundColor Yellow
    }
    Write-Host ''
    Write-Host '  For manual downloads, use yt-dlp with the YouTube URL.' -ForegroundColor Gray
}

Write-Host ''
$rawMsg = '  Raw clips:        ' + $RAW_DIR
$normMsg = '  Normalized clips: ' + $NORM_DIR
Write-Host $rawMsg -ForegroundColor Cyan
Write-Host $normMsg -ForegroundColor Cyan
Write-Host ''
Write-Host '  Done! Review clips and re-trim any that miss the emotional peak.' -ForegroundColor Green
