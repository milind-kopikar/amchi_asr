import json
import os
import sys
import argparse

def verify_manifest(manifest_path):
    print(f"Verifying audio files in manifest: {manifest_path}")
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found: {manifest_path}")
        return False

    missing_files = []
    total_files = 0
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_files += 1
                data = json.loads(line)
                audio_path = data.get('audio_filepath')
                
                if not audio_path:
                    print(f"Warning: Line {total_files} missing 'audio_filepath'")
                    continue
                    
                # Handle relative paths if necessary (assuming relative to manifest or CWD)
                # NeMo usually expects absolute paths or paths relative to CWD
                if not os.path.exists(audio_path):
                    # Try relative to manifest dir
                    manifest_dir = os.path.dirname(manifest_path)
                    rel_path = os.path.join(manifest_dir, audio_path)
                    if not os.path.exists(rel_path):
                         missing_files.append(audio_path)

    except Exception as e:
        print(f"Error reading manifest: {e}")
        return False

    if missing_files:
        print(f"Found {len(missing_files)} missing audio files.")
        for p in missing_files[:5]:
            print(f"  - {p}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more.")
        return False
    
    print(f"All {total_files} audio files verified successfully.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify audio files in NeMo manifest")
    parser.add_argument("manifests", nargs='+', help="Path(s) to manifest JSONL files")
    args = parser.parse_args()

    all_valid = True
    for manifest in args.manifests:
        if not verify_manifest(manifest):
            all_valid = False
    
    sys.exit(0 if all_valid else 1)
