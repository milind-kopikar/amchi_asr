# Recording Script for Sentence 2
# Run this in PowerShell to record: एकी गोम्टी काणी आय्कयाति!

Write-Host "🎤 Recording Sentence 2: एकी गोम्टी काणी आय्कयाति!"
Write-Host "Please speak clearly: एकी गोम्टी काणी आय्कयाति!"
Write-Host "Press Enter to start recording (5 seconds)..."

Read-Host

# Check if ffmpeg is available
try {
    $ffmpeg = Get-Command ffmpeg -ErrorAction Stop
    Write-Host "✅ FFmpeg found: $($ffmpeg.Source)"
} catch {
    Write-Host "❌ FFmpeg not found. Please install from https://ffmpeg.org/download.html"
    exit 1
}

# Start recording
Write-Host "🔴 Recording started... Speak now!"
& ffmpeg -f dshow -i audio="Microphone (Realtek(R) Audio)" -t 5 -acodec pcm_s16le -ar 16000 -ac 1 "data/audio/sentence_02.wav" -y

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Recording completed: data/audio/sentence_02.wav"
    $fileSize = (Get-Item "data/audio/sentence_02.wav").Length
    Write-Host "   File size: $($fileSize) bytes"
} else {
    Write-Host "❌ Recording failed"
}