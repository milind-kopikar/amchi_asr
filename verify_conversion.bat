@echo off
echo 🎵 Verifying WAV File Conversions for Konkani ASR
echo ==================================================

echo.
echo 📁 Checking data/audio/ directory...
dir data\audio\ /b

echo.
echo 🔍 Detailed file information:
for %%f in (data\audio\*.wav) do (
    echo.
    echo 📄 %%~nf%%~xf:
    powershell "Get-Item '%%f' | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize"
)

echo.
echo ✅ Conversion checklist:
echo - [ ] sentence_01.wav exists (16kHz mono)
echo - [ ] sentence_02.wav exists (16kHz mono)
echo - [ ] sentence_03.wav exists (16kHz mono)
echo.
echo 🚀 Next: Run data preparation
echo python scripts/prepare_data.py --audio_dir data/audio --transcript_dir data/transcripts --output_dir data/test_run

pause