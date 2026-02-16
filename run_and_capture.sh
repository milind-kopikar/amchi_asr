#!/usr/bin/env bash
# Run the runpod test and capture both stdout and file output
python3 scripts/runpod_test.py > scripts/runpod_test_stdout.txt 2>&1
echo "Stdout written to scripts/runpod_test_stdout.txt"
if [ -f runpod_test_output.txt ]; then
  echo "runpod_test_output.txt exists and contains:"
  cat runpod_test_output.txt
else
  echo "runpod_test_output.txt not found"
fi
