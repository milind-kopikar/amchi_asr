#!/usr/bin/env python3
"""Remove lines containing Sanskrit markers from a text file.

By default this writes a cleaned file next to the input file (with suffix `.cleaned`).
Options:
  --input/-i     Input file path (default: data/extra_corpus.txt)
  --output/-o    Output file path (default: <input>.cleaned)
  --marker/-m    Marker string to detect Sanskrit (default: "[Sanskrit]")
  --bookended    Only remove lines that contain both a start and end marker (default: False)
  --ignore-case  Case-insensitive marker match
  --in-place     Overwrite input file (backup created: <input>.bak)
  --dry-run      Don't write anything, only report counts

Examples:
  python scripts/remove_sanskrit_lines.py -i data/extra_corpus.txt
  python scripts/remove_sanskrit_lines.py -i data/extra_corpus.txt --in-place
"""

from pathlib import Path
import argparse
import re
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Remove lines that contain Sanskrit markers")
    p.add_argument("--input", "-i", default="data/extra_corpus.txt", help="Input file")
    p.add_argument("--output", "-o", help="Output file (default: <input>.cleaned)")
    p.add_argument("--marker", "-m", default="[Sanskrit]", help="Marker string to match")
    p.add_argument("--bookended", action="store_true", help="Only remove lines that contain both a start and end marker")
    p.add_argument("--ignore-case", action="store_true", help="Case-insensitive marker match")
    p.add_argument("--remove-latin", action="store_true", help="Also remove lines that contain ASCII Latin letters (A-Z, a-z)")
    p.add_argument("--in-place", action="store_true", help="Overwrite the input file (creates a .bak backup)")
    p.add_argument("--dry-run", action="store_true", help="Show counts but do not write files")
    return p.parse_args()


def build_pattern(marker: str, bookended: bool, ignore_case: bool, remove_latin: bool=False):
    # Escape marker for regex
    esc = re.escape(marker)
    flags = re.IGNORECASE if ignore_case else 0
    latin_pat = r"[A-Za-z]"
    if bookended and remove_latin:
        # Match either bookended marker OR any Latin letter
        pattern = re.compile(r"(^.*" + esc + r".*" + esc + r".*$)|(" + latin_pat + r")", flags)
    elif bookended:
        # Looks for marker ... marker (anywhere in the line)
        pattern = re.compile(r"^.*" + esc + r".*" + esc + r".*$", flags)
    else:
        if remove_latin:
            # Match marker OR any Latin letter
            pattern = re.compile(r"(" + esc + r")|(" + latin_pat + r")", flags)
        else:
            pattern = re.compile(esc, flags)
    return pattern


def process(input_path: Path, output_path: Path, pattern: re.Pattern, dry_run=False):
    removed = 0
    total = 0
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 0, 0
    with input_path.open("r", encoding="utf-8") as inf:
        lines = inf.readlines()
    kept_lines = []
    for ln in lines:
        total += 1
        if pattern.search(ln):
            removed += 1
        else:
            kept_lines.append(ln)
    if dry_run:
        print(f"Dry-run: total lines = {total}, matched (would remove) = {removed}, kept = {total-removed}")
        return total, removed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outf:
        outf.writelines(kept_lines)
    print(f"Processed: total={total}, removed={removed}, written={len(kept_lines)} -> {output_path}")
    return total, removed


def main():
    args = parse_args()
    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.with_suffix(inp.suffix + ".cleaned")
    pattern = build_pattern(args.marker, args.bookended, args.ignore_case, remove_latin=args.remove_latin)

    # Dry-run first if requested
    if args.dry_run:
        process(inp, out, pattern, dry_run=True)
        return

    if args.in_place:
        bak = inp.with_suffix(inp.suffix + ".bak")
        inp.rename(bak)
        # After renaming, process from bak -> inp
        total, removed = process(bak, inp, pattern, dry_run=False)
        print(f"In-place: original backed up to {bak}")
    else:
        total, removed = process(inp, out, pattern, dry_run=False)


if __name__ == '__main__':
    main()
