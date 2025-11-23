@echo off
echo 🎵 Quick WAV File Check for Konkani ASR
echo =======================================

echo.
echo 📁 Files in data/audio/:
dir data\audio\*.wav /b

echo.
echo 📊 File sizes (should be 80-128KB for 5-8 second sentences):
for %%f in (data\audio\*.wav) do (
    echo 📄 %%~nf%%~xf: 
    powershell "Write-Host '   Size:' (Get-Item '%%f').Length 'bytes' -NoNewline; $size = (Get-Item '%%f').Length; if ($size -lt 50000) {Write-Host ' ❌ Too small' -ForegroundColor Red} elseif ($size -gt 200000) {Write-Host ' ❌ Too large (wrong sample rate?)' -ForegroundColor Red} else {Write-Host ' ✅ Good size' -ForegroundColor Green}"
)

echo.
echo 🎯 ASR Requirements Check:
echo ✅ Sample Rate: 16000 Hz (set in CloudConvert)
echo ✅ Channels: Mono (set in CloudConvert)  
echo ✅ Codec: PCM (gives 16-bit)
echo ✅ Format: WAV

echo.
echo 🚀 If sizes look good, run:
echo python scripts/prepare_data.py --audio_dir data/audio --transcript_dir data/transcripts --output_dir data/test_run

echo.
pause