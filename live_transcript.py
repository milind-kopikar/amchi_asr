#!/usr/bin/env python3
"""live_transcript.py

Simulate live transcription by streaming words from a text file.
This version lives in `konkani_asr/` and exposes a small programmatic API so
an inference loop can call it directly.

Public functions you may call from your model code:
- word_generator(path) -> yields words (str) from the file
- stream_words_from_file(path, delay=1.0, callback=None) -> streams words; if
  callback is provided it's called with each word, otherwise prints to console

CLI usage (same as the collector version):
  python live_transcript.py story5.txt            # console, 1s delay
  python live_transcript.py story5.txt --mode gui  # GUI mode

"""

from __future__ import annotations
import argparse
import os
import sys
import time
import shutil
from typing import Callable, Iterable, Iterator, Optional

try:
    import tkinter as tk
except Exception:
    tk = None


def read_words(path: str) -> list[str]:
    """Read text file and split on whitespace to produce words."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return text.split()


def word_generator(path: str) -> Iterator[str]:
    """Yield words from a file as an iterator (useful for model integration)."""
    for w in read_words(path):
        yield w


def stream_words(words: Iterable[str], delay: float = 1.0, callback: Optional[Callable[[str], None]] = None) -> None:
    """
    Stream words from an iterable. If callback is provided it will be invoked
    with each word, otherwise words are printed to stdout with spaces.
    """
    if callback is None:
        # default: console print with space-separated words
        for w in words:
            sys.stdout.write(w + " ")
            sys.stdout.flush()
            time.sleep(delay)
        sys.stdout.write("\n")
    else:
        for w in words:
            callback(w)
            time.sleep(delay)


def stream_words_from_file(path: str, delay: float = 1.0, callback: Optional[Callable[[str], None]] = None) -> None:
    """Convenience wrapper: read file and stream its words."""
    stream_words(word_generator(path), delay=delay, callback=callback)


def line_word_generator(path: str) -> Iterator[tuple[str, bool]]:
    """Yield (word, is_line_start) tuples from file.

    is_line_start is True when the word is the first word of a line.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                # yield a newline marker (no word)
                yield ("", True)
                continue
            parts = line.split()
            for i, w in enumerate(parts):
                yield (w, i == 0)


def stream_lines_from_file(path: str, delay: float = 1.0, callback: Optional[Callable[[str, bool], None]] = None) -> None:
    """Stream words preserving line boundaries.

    For each word, the callback (if provided) is invoked as callback(word, is_line_start).
    If callback is None, the function prints words and starts a new line when appropriate.
    """
    if callback is None:
        for w, is_start in line_word_generator(path):
            if w == "" and is_start:
                # blank line
                sys.stdout.write("\n")
                sys.stdout.flush()
                time.sleep(delay)
                continue
            if is_start:
                # start a new line
                sys.stdout.write("\n" + w)
            else:
                sys.stdout.write(" " + w)
            sys.stdout.flush()
            time.sleep(delay)
        sys.stdout.write("\n")
    else:
        for w, is_start in line_word_generator(path):
            callback(w, is_start)
            time.sleep(delay)


def console_single_word_mode(path: str, delay: float) -> None:
    words = read_words(path)
    width = shutil.get_terminal_size((80, 20)).columns
    for w in words:
        sys.stdout.write("\r" + " " * width + "\r")
        sys.stdout.write(w)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")


def gui_mode(path: str, delay: float, font_size: int, wrap: int) -> None:
    if tk is None:
        print("Tkinter not available on this system. Use console mode instead.")
        sys.exit(1)

    words = read_words(path)

    root = tk.Tk()
    root.title("Live transcript")

    lbl = tk.Label(root, text="", font=("Helvetica", font_size), wraplength=wrap, justify="left")
    lbl.pack(padx=20, pady=20)

    idx = 0

    def step():
        nonlocal idx
        if idx < len(words):
            curr = lbl.cget("text")
            if curr:
                new = curr + " " + words[idx]
            else:
                new = words[idx]
            lbl.config(text=new)
            idx += 1
            root.after(int(delay * 1000), step)
        else:
            return

    root.after(0, step)
    root.mainloop()


def main():
    p = argparse.ArgumentParser(description="Show words from a story file one-by-one to simulate live transcription")
    p.add_argument("file", help="Path to the text file (UTF-8)")
    p.add_argument("--delay", "-d", type=float, default=1.0, help="Delay between words in seconds (default: 1.0)")
    p.add_argument("--mode", "-m", choices=("console", "gui"), default="console", help="Display mode (console or gui)")
    p.add_argument("--single", action="store_true", help="In console mode: show only the current word (overwrite previous)")
    p.add_argument("--font-size", type=int, default=28, help="Font size for GUI mode")
    p.add_argument("--wrap", type=int, default=800, help="Wrap length in pixels for GUI mode")
    p.add_argument("--linewise", action="store_true", help="In console mode: preserve line boundaries and stream words line-by-line")
    args = p.parse_args()

    if not os.path.isfile(args.file):
        print(f"Error: file not found: {args.file}")
        sys.exit(2)

    if args.mode == "console":
        if args.single:
            console_single_word_mode(args.file, args.delay)
        elif args.linewise:
            stream_lines_from_file(args.file, delay=args.delay)
        else:
            stream_words_from_file(args.file, delay=args.delay)
    else:
        gui_mode(args.file, args.delay, args.font_size, args.wrap)


if __name__ == "__main__":
    main()
