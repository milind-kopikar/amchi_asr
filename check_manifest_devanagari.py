#!/usr/bin/env python3
"""Check manifests for UTF-8 encoding and presence of Devanagari characters"""
import argparse
import json
import re
from pathlib import Path

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def check_manifest(manifest_path: Path) -> int:
    with manifest_path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"ERROR: Line {i}: invalid JSON: {e}")
                return 1
            text = obj.get("text", "")
            if not text:
                print(f"ERROR: Line {i}: missing 'text' field")
                return 2
            if not DEVANAGARI_RE.search(text):
                print(f"WARNING: Line {i}: no Devanagari character found; text sample: {text[:40]!r}")
    print(f"Checked {manifest_path}: OK (lines={i})")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manifest UTF-8 + Devanagari validator")
    parser.add_argument("manifest", help="Path to manifest.jsonl")
    args = parser.parse_args()

    exit(check_manifest(Path(args.manifest)))
