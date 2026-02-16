#!/usr/bin/env bash
# Helper script to stop tracking large data/model files and commit .gitignore
# Run this from the repo root (after reviewing .gitignore changes)

set -e

echo "This script will remove large files from git index (not from disk) according to common patterns."
echo "Make sure you review .gitignore before running."

read -p "Proceed? [y/N] " yn
if [[ "$yn" != "y" && "$yn" != "Y" ]]; then
  echo "Aborting."
  exit 1
fi

# Ensure .gitignore is added
git add .gitignore || true

# Patterns to untrack
patterns=("data" "data/audio" "data/transcripts" "models" "*.nemo" "*.pt" "*.pth" "*.bin" "*.onnx" "*.ckpt" "ffmpeg" "ffmpeg.zip")

for p in "${patterns[@]}"; do
  echo "Removing from git index: $p"
  git rm -r --cached --ignore-unmatch "$p" || true
done

echo "Creating commit (if there are staged changes)..."
if git diff --cached --quiet; then
  echo "No changes to commit."
else
  git commit -m "chore: stop tracking large files (audio, models); update .gitignore"
  echo "Committed. You may now push: git push origin $(git rev-parse --abbrev-ref HEAD)"
fi

echo "Done."
