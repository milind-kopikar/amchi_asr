#!/usr/bin/env python3
"""
Data Verification Script
Display manifests with proper Devanagari text and allow audio playback
"""

import json
import os
import sys
import subprocess
from pathlib import Path

def display_manifest(manifest_path: str, title: str, max_samples: int = None):
    """Display manifest with Devanagari text"""
    print("="*100)
    print(f"{title}")
    print("="*100)
    
    if not os.path.exists(manifest_path):
        print(f"❌ Manifest not found: {manifest_path}")
        return
    
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    
    print(f"\nTotal samples: {len(samples)}\n")
    
    display_count = max_samples if max_samples else len(samples)
    
    for i, sample in enumerate(samples[:display_count], 1):
        audio_path = sample['audio_filepath']
        text = sample['text']
        duration = sample['duration']
        
        # Check if audio file exists
        audio_exists = "✓" if os.path.exists(audio_path) else "❌"
        
        print(f"{i}. {audio_exists} Audio: {audio_path}")
        print(f"   📝 Text: {text}")
        print(f"   ⏱️  Duration: {duration}s")
        print()
    
    if max_samples and len(samples) > max_samples:
        print(f"... and {len(samples) - max_samples} more samples")
    
    return samples

def play_audio(audio_path: str):
    """Play audio file using default system player"""
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    print(f"🔊 Playing: {audio_path}")
    
    # Windows: use start command
    if sys.platform == 'win32':
        os.startfile(audio_path)
    # macOS: use open
    elif sys.platform == 'darwin':
        subprocess.call(['open', audio_path])
    # Linux: use xdg-open
    else:
        subprocess.call(['xdg-open', audio_path])

def verify_sample(sample: dict):
    """Display sample and allow audio playback"""
    print("\n" + "="*100)
    print("📋 SAMPLE VERIFICATION")
    print("="*100)
    print(f"Audio: {sample['audio_filepath']}")
    print(f"Text:  {sample['text']}")
    print(f"Duration: {sample['duration']}s")
    print("="*100)
    
    # Play audio
    play_audio(sample['audio_filepath'])
    print("\n✓ Audio should be playing in your default audio player")
    print(f"📝 Verify the audio says: \"{sample['text']}\"")

def interactive_mode(samples: list):
    """Interactive verification mode"""
    print("\n" + "="*100)
    print("🔍 INTERACTIVE VERIFICATION MODE")
    print("="*100)
    print("Commands:")
    print("  1-N  : Play sample number N")
    print("  q    : Quit")
    print("="*100)
    
    while True:
        try:
            choice = input("\nEnter sample number (or 'q' to quit): ").strip()
            
            if choice.lower() == 'q':
                print("👋 Goodbye!")
                break
            
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(samples):
                    verify_sample(samples[idx])
                else:
                    print(f"❌ Invalid sample number. Choose 1-{len(samples)}")
            else:
                print("❌ Invalid input. Enter a number or 'q'")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def export_readable(manifest_path: str, output_path: str):
    """Export manifest to readable text file"""
    print(f"\n📄 Exporting to readable format: {output_path}")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line.strip()) for line in f]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("KONKANI ASR TRAINING DATA\n")
        f.write("="*100 + "\n\n")
        
        for i, sample in enumerate(samples, 1):
            f.write(f"{i}. Audio: {sample['audio_filepath']}\n")
            f.write(f"   Text: {sample['text']}\n")
            f.write(f"   Duration: {sample['duration']}s\n\n")
    
    print(f"✓ Exported {len(samples)} samples")
    print(f"📂 Open this file to see all text in Devanagari: {output_path}")

def main():
    print("\n🎯 KONKANI ASR DATA VERIFICATION TOOL\n")
    
    # Display train manifest
    train_samples = display_manifest(
        "data/train/manifest.jsonl",
        "📚 TRAIN MANIFEST (First 5 samples)",
        max_samples=5
    )
    
    # Display dev manifest
    dev_samples = display_manifest(
        "data/dev/manifest.jsonl",
        "📊 DEV MANIFEST (All samples)",
        max_samples=None
    )
    
    # Export readable versions
    print("\n" + "="*100)
    print("📝 EXPORTING READABLE TEXT FILES")
    print("="*100)
    
    export_readable("data/train/manifest.jsonl", "data/train/manifest_readable.txt")
    export_readable("data/dev/manifest.jsonl", "data/dev/manifest_readable.txt")
    
    # Ask if user wants interactive mode
    print("\n" + "="*100)
    print("Would you like to verify samples interactively? (y/n)")
    print("="*100)
    
    choice = input("Choice: ").strip().lower()
    
    if choice == 'y' and train_samples:
        print("\n🔊 Opening interactive mode for TRAIN samples...")
        print("💡 Tip: Play a sample and listen to verify the text matches")
        interactive_mode(train_samples)

if __name__ == "__main__":
    main()
