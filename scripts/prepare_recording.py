#!/usr/bin/env python3
"""
Audio recording helper for Konkani ASR testing
Helps record and prepare test audio samples
"""

import os
import time
import argparse
from pathlib import Path

def create_recording_script(audio_name: str, duration: int = 60):
    """Create a PowerShell script for recording audio"""

    script_content = f'''
# Audio Recording Script for Konkani ASR Testing
# Run this in PowerShell to record {duration} seconds of audio

Write-Host "🎤 Recording {duration} seconds of Konkani speech..."
Write-Host "Please speak clearly in Konkani for {duration} seconds"
Write-Host "Press Enter to start recording..."

Read-Host

# Check if ffmpeg is available
try {{
    $ffmpeg = Get-Command ffmpeg -ErrorAction Stop
    Write-Host "✅ FFmpeg found: $($ffmpeg.Source)"
}} catch {{
    Write-Host "❌ FFmpeg not found. Please install from https://ffmpeg.org/download.html"
    exit 1
}}

# Start recording
Write-Host "🔴 Recording started... Speak now!"
Write-Host "Recording will stop automatically in {duration} seconds"

& ffmpeg -f dshow -i audio="Microphone (Realtek(R) Audio)" -t {duration} -acodec pcm_s16le -ar 16000 -ac 1 "{audio_name}" -y

if ($LASTEXITCODE -eq 0) {{
    Write-Host "✅ Recording completed: {audio_name}"
    Write-Host "File size: $((Get-Item "{audio_name}").Length) bytes"
}} else {{
    Write-Host "❌ Recording failed"
}}
'''

    script_name = f"record_{audio_name.replace('.wav', '')}.ps1"
    with open(script_name, 'w', encoding='utf-8') as f:
        f.write(script_content)

    print(f"✅ Created recording script: {script_name}")
    print("Run this script in PowerShell to record your audio")
    return script_name

def create_test_transcript_template(audio_name: str):
    """Create a template for the transcript"""

    template_content = f'''# Transcript Template for {audio_name}
#
# Copy the exact words you spoke during recording
# Write them below in Konkani script (Devanagari)
#
# Example:
# माझे नाव मिलिंद आहे. आमी कोकणात राहतो.
# आमची भाषा कोकणी आहे.
#
# Your transcript:

'''

    transcript_name = audio_name.replace('.wav', '_transcript.txt')
    with open(transcript_name, 'w', encoding='utf-8') as f:
        f.write(template_content)

    print(f"✅ Created transcript template: {transcript_name}")
    print("Edit this file with the exact text you spoke")
    return transcript_name

def prepare_test_package(audio_name: str, transcript_text: str, duration: int = 60):
    """Prepare a complete test package"""

    print("🎯 Preparing Konkani ASR Test Package")
    print("=" * 50)

    # Create recording script
    script_name = create_recording_script(audio_name, duration)

    # Create transcript template
    template_name = create_test_transcript_template(audio_name)

    print("\n📋 Next Steps:")
    print("1. Run the recording script in PowerShell:")
    print(f"   .\\{script_name}")
    print()
    print("2. Speak clearly in Konkani for", duration, "seconds")
    print()
    print("3. Edit the transcript file with exact words spoken:")
    print(f"   notepad {template_name}")
    print()
    print("4. Run the minimal test:")
    print(f"   python scripts/minimal_test.py --audio_file {audio_name} --transcript \"[your transcript text]\"")
    print()
    print("💡 Tips:")
    print("   - Speak at normal speed and volume")
    print("   - Use a quiet environment")
    print("   - Speak complete sentences")
    print("   - Include different Konkani words/phrases")

def validate_recording(audio_file: str):
    """Basic validation of recorded audio"""

    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return False

    file_size = os.path.getsize(audio_file)
    print(f"✅ Audio file found: {audio_file}")
    print(f"   Size: {file_size:,} bytes")

    # Rough check for 16kHz mono WAV
    expected_min_size = 16000 * 2 * 10  # 10 seconds minimum
    if file_size < expected_min_size:
        print(f"⚠️ File seems small. Expected at least {expected_min_size:,} bytes for 10 seconds")
        return False

    print("✅ Audio file looks valid")
    return True

def main():
    parser = argparse.ArgumentParser(description="Prepare audio recording for Konkani ASR testing")
    parser.add_argument("--audio_name", default="konkani_test.wav", help="Name for the audio file")
    parser.add_argument("--duration", type=int, default=60, help="Recording duration in seconds")
    parser.add_argument("--prepare_only", action="store_true", help="Only create scripts, don't validate")
    parser.add_argument("--validate", help="Validate existing audio file")

    args = parser.parse_args()

    if args.validate:
        # Just validate existing file
        validate_recording(args.validate)
    else:
        # Prepare test package
        prepare_test_package(args.audio_name, "", args.duration)

if __name__ == "__main__":
    main()