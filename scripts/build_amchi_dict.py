#!/usr/bin/env python3
"""
Build the Amchi Konkani dictionary JSON used by the RunPod handler.

The dictionary is fetched from the konkani_dictionary Railway API (or read
from a local CSV fallback) and saved as a JSON array of objects, each with
at least a ``word_konkani_devanagari`` field — the format
``scripts/amchi_postprocess_asr.load_dictionary`` expects.

Usage
-----
Default (fetch live from Railway and write to data/amchi_konkani_dict.json)::

    python3 scripts/build_amchi_dict.py

From a local CSV instead (offline mode)::

    python3 scripts/build_amchi_dict.py --csv ../konkani_dictionary/konkani_dictionary_csv.csv

Run before building the Docker image so the dictionary gets baked into
``runpod/Dockerfile.serverless``.

Inputs
------
- ``--source railway`` (default): fetches from ``--url`` (the konkani_dictionary
  Railway endpoint).
- ``--source csv``: reads from ``--csv`` path.

Outputs
-------
A JSON file at ``--out`` (default ``data/amchi_konkani_dict.json``) containing
a JSON array of dict-entry objects. Each object has at minimum::

    {"word_konkani_devanagari": "<devanagari word>", ...}

Exit codes
----------
- 0 on success
- 1 on network / file errors
- 2 on malformed source data
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

# Public Railway endpoint (paginated). Page size 5000 covers the whole dictionary
# in a single request, which is what we want for a build-time script.
DEFAULT_RAILWAY_URL = "https://konkanicollector-production.up.railway.app/api/dictionary?limit=5000"
DEFAULT_OUT_PATH = "data/amchi_konkani_dict.json"

# Field name the post-processor reads.
DEV_FIELD = "word_konkani_devanagari"

# CSV header variants seen in konkani_dictionary_csv.csv.
CSV_HEADER_ALIASES = (
    "Konkani Word in Devnagiri",
    "konkani_word_in_devnagiri",
    "word_konkani_devanagari",
    "devanagari",
)


def fetch_from_railway(url: str, timeout_seconds: int = 30) -> list[dict]:
    """Fetch dictionary entries from a Railway JSON endpoint.

    Parameters
    ----------
    url
        Full URL to the dictionary endpoint. The endpoint must return either
        a JSON array or an object with a ``"data"`` or ``"entries"`` list.
    timeout_seconds
        HTTP timeout. Defaults to 30 seconds (Railway can be slow on cold start).

    Returns
    -------
    list[dict]
        Entries with at least a Devanagari-word field.

    Raises
    ------
    urllib.error.URLError, urllib.error.HTTPError
        On network failure.
    ValueError
        If the response is not JSON or has an unexpected shape.
    """
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        for key in ("data", "entries", "results", "rows"):
            if key in payload and isinstance(payload[key], list):
                entries = payload[key]
                break
        else:
            raise ValueError(
                f"Unexpected Railway response shape; top-level keys: "
                f"{sorted(payload.keys())}"
            )
    else:
        raise ValueError(f"Unexpected Railway response type: {type(payload).__name__}")

    return entries


def read_from_csv(csv_path: str) -> list[dict]:
    """Read dictionary entries from the konkani_dictionary CSV file.

    Parameters
    ----------
    csv_path
        Path to ``konkani_dictionary_csv.csv``.

    Returns
    -------
    list[dict]
        Entries normalised to have a ``word_konkani_devanagari`` field.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    ValueError
        If no Devanagari-word column is found in the CSV header.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")

        dev_col = _pick_devanagari_column(reader.fieldnames)
        if dev_col is None:
            raise ValueError(
                f"No Devanagari-word column found in CSV header: {reader.fieldnames}"
            )

        entries = []
        for row in reader:
            word = (row.get(dev_col) or "").strip()
            if not word:
                continue
            entries.append({DEV_FIELD: word, **{k: v for k, v in row.items() if k != dev_col}})
    return entries


def _pick_devanagari_column(field_names: Iterable[str]) -> str | None:
    """Choose which CSV column holds the Devanagari word (case-insensitive)."""
    lowered = {f.lower().strip(): f for f in field_names if f}
    for alias in CSV_HEADER_ALIASES:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def normalise_entries(entries: list[dict]) -> list[dict]:
    """Ensure every entry has a non-empty ``word_konkani_devanagari`` field.

    Drops entries that have no Devanagari word. Keeps any other fields intact
    (so future post-processor logic could use English meanings or POS tags).
    """
    out = []
    for entry in entries:
        word = (entry.get(DEV_FIELD) or "").strip()
        if not word:
            continue
        normalised = dict(entry)
        normalised[DEV_FIELD] = word
        out.append(normalised)
    return out


def write_json(entries: list[dict], out_path: str) -> None:
    """Write the entries as pretty-printed UTF-8 JSON. Creates parent dirs."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False, indent=2)


def build(source: str, url: str, csv_path: str | None, out_path: str) -> int:
    """End-to-end build. Returns the number of entries written."""
    if source == "railway":
        entries = fetch_from_railway(url)
    elif source == "csv":
        if not csv_path:
            raise ValueError("--csv path required when --source csv")
        entries = read_from_csv(csv_path)
    else:
        raise ValueError(f"Unknown --source: {source}")

    normalised = normalise_entries(entries)
    if not normalised:
        raise ValueError("No usable dictionary entries found (all rows missing Devanagari).")

    write_json(normalised, out_path)
    return len(normalised)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Amchi Konkani dictionary JSON for the RunPod handler.",
    )
    parser.add_argument("--source", choices=("railway", "csv"), default="railway",
                       help="Where to read dictionary entries from.")
    parser.add_argument("--url", default=DEFAULT_RAILWAY_URL,
                       help="Railway endpoint URL (used when --source railway).")
    parser.add_argument("--csv", default=None,
                       help="Path to konkani_dictionary_csv.csv (used when --source csv).")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH,
                       help="Where to write the resulting JSON file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        count = build(args.source, args.url, args.csv, args.out)
    except (urllib.error.URLError, urllib.error.HTTPError, FileNotFoundError) as exc:
        print(f"ERROR: I/O failure: {exc}", file=sys.stderr)
        return 1
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: malformed source: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {count} entries to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
