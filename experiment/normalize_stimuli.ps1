# ============================================================
# NORMALIZE STIMULI — High-Quality for Screen Playback
# ============================================================
# Converts all clips in stimuli_new/ to a consistent format
# and saves them to stimuli/
#
# Output format:
#   - 1920x1080 (landscape, letterboxed if needed)
#   - H.264 High Profile (best compatibility with PsychoPy)
#   - CRF 15 (near-lossless visual quality)
#   - 30 fps (consistent timing)
#   - AAC 192kbps stereo audio
#   - faststart for instant playback
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File normalize_stimuli.ps1
# ============================================================

# Auto-add tool paths
$extraPaths = @(
    "$env:LOCALAPPDATA\Programs\Python\Python313\Scripts",
    "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts",
    "$env:USERPROFILE\.deno\bin",
    "$env:LOCALAPPDATA\Microsoft\WinGet\Links",
    "C:\ProgramData\chocolatey\bin"
)
foreach ($p in $extraPaths) {
    if (Test-Path $p) { $env:PATH += ";$p" }
}
$ffmpegSearch = Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter 'ffmpeg.exe' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if ($ffmpegSearch) { $env:PATH += ";" + $ffmpegSearch.DirectoryName }

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host 'ffmpeg not found. Install with: winget install Gyan.FFmpeg' -ForegroundColor Red
    exit 1
}

$SRC_DIR = 'stimuli_new'
$OUT_DIR = 'stimuli'
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null

$success = 0
$fail = 0
$skipped = 0

$files = Get-ChildItem -Path $SRC_DIR -Filter '*.mp4' | Sort-Object Name

Write-Host '=========================================' -ForegroundColor Yellow
Write-Host '  NORMALIZING STIMULI (max quality)' -ForegroundColor Yellow
Write-Host "  Source: $SRC_DIR\" -ForegroundColor Gray
Write-Host "  Output: $OUT_DIR\" -ForegroundColor Gray
Write-Host "  Found $($files.Count) mp4 files" -ForegroundColor Gray
Write-Host '=========================================' -ForegroundColor Yellow
Write-Host ''

foreach ($file in $files) {
    $outFile = Join-Path $OUT_DIR $file.Name

    Write-Host "  $($file.Name) " -NoNewline

    # Skip if already normalized
    if (Test-Path $outFile) {
        Write-Host '[already done]' -ForegroundColor DarkGray
        $skipped++
        continue
    }

    # Normalize: scale to fit 1920x1080, letterbox if needed, high quality
    try {
        $result = & ffmpeg -y -i $file.FullName `
            -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `
            -r 30 `
            -c:v libx264 -profile:v high -level 4.1 -crf 15 -preset medium `
            -pix_fmt yuv420p `
            -c:a aac -b:a 192k -ac 2 -ar 44100 `
            -movflags +faststart `
            $outFile 2>&1

        if (Test-Path $outFile) {
            $info = & ffprobe -v quiet -show_entries format=duration -of csv=p=0 $outFile 2>$null
            if ($info) {
                $dur = [math]::Round([double]$info, 1)
                Write-Host "[OK - ${dur}s]" -ForegroundColor Green
                $success++
            } else {
                Write-Host '[OK]' -ForegroundColor Green
                $success++
            }
        } else {
            Write-Host '[FAILED - no output]' -ForegroundColor Red
            $fail++
        }
    }
    catch {
        Write-Host "[FAILED - $($_.Exception.Message)]" -ForegroundColor Red
        $fail++
    }
}

Write-Host ''
Write-Host '=========================================' -ForegroundColor Yellow
Write-Host '  SUMMARY' -ForegroundColor Yellow
Write-Host '=========================================' -ForegroundColor Yellow
Write-Host "  Normalized: $success" -ForegroundColor Green
Write-Host "  Skipped:    $skipped" -ForegroundColor DarkGray
Write-Host "  Failed:     $fail" -ForegroundColor Red
Write-Host ''

# Check for missing clips
$expected = @()
foreach ($cat in @('HAP', 'SAD', 'ANG', 'SUR', 'NEU')) {
    1..10 | ForEach-Object { $expected += "${cat}_$($_.ToString('D2')).mp4" }
}
$existing = Get-ChildItem -Path $OUT_DIR -Filter '*.mp4' | Select-Object -ExpandProperty Name
$missing = $expected | Where-Object { $_ -notin $existing }
if ($missing.Count -gt 0) {
    Write-Host "  Missing clips ($($missing.Count)):" -ForegroundColor Yellow
    foreach ($m in $missing) { Write-Host "    - $m" -ForegroundColor Yellow }
}

Write-Host ''
Write-Host "  Output folder: $OUT_DIR\" -ForegroundColor Cyan
Write-Host '  Done!' -ForegroundColor Green
