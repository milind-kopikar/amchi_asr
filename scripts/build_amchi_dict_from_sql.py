#!/usr/bin/env python3
"""Build amchi_konkani_dict.json from the in-repo SQL dump.

Fallback for `scripts/build_amchi_dict.py` when the upstream Railway
endpoint is 404 (which it has been since at least 2026-05). Reads
`konkani_dictionary_export.sql` from the repo root, extracts the
`word_konkani_devanagari` column from every INSERT tuple, and writes
the same JSON shape `scripts/amchi_postprocess_asr.load_dictionary`
expects.

Usage::

    python3 scripts/build_amchi_dict_from_sql.py

Or with explicit paths::

    python3 scripts/build_amchi_dict_from_sql.py \\
        --sql konkani_dictionary_export.sql \\
        --out data/amchi_konkani_dict.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def extract_devanagari_words(sql_text: str) -> list[str]:
    """Return unique word_konkani_devanagari values from the SQL dump.

    Parses each VALUES tuple character-by-character so SQL ``''`` escapes
    inside quoted strings are handled correctly. The 2nd column in each
    row is ``word_konkani_devanagari`` (per the dump's INSERT column list).
    """
    out: list[str] = []
    seen: set[str] = set()
    n = len(sql_text)
    for match in re.finditer(r"\bVALUES\b", sql_text):
        i = match.end()
        while i < n:
            while i < n and sql_text[i] in " \t\n\r,":
                i += 1
            if i >= n or sql_text[i] == ";":
                break
            if sql_text[i] != "(":
                break
            i += 1  # past '('
            fields: list[tuple[str, bool]] = []
            cur: list[str] = []
            in_str = False
            field_is_string = False
            while i < n:
                c = sql_text[i]
                if in_str:
                    if c == "'":
                        if i + 1 < n and sql_text[i + 1] == "'":
                            cur.append("'")
                            i += 2
                            continue
                        in_str = False
                        i += 1
                        continue
                    cur.append(c)
                    i += 1
                else:
                    if c == "'":
                        in_str = True
                        field_is_string = True
                        i += 1
                    elif c == ",":
                        fields.append(("".join(cur).strip(), field_is_string))
                        cur = []
                        field_is_string = False
                        i += 1
                    elif c == ")":
                        fields.append(("".join(cur).strip(), field_is_string))
                        i += 1
                        break
                    else:
                        cur.append(c)
                        i += 1
            if len(fields) >= 2:
                value, is_string = fields[1]
                if is_string and value and value not in seen:
                    seen.add(value)
                    out.append(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sql", default="konkani_dictionary_export.sql")
    parser.add_argument("--out", default="data/amchi_konkani_dict.json")
    args = parser.parse_args()

    sql_path = Path(args.sql)
    if not sql_path.exists():
        print(f"error: SQL dump not found at {sql_path}", file=sys.stderr)
        return 1

    words = extract_devanagari_words(sql_path.read_text())
    entries = [{"word_konkani_devanagari": w} for w in words]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2))
    print(f"wrote {len(entries)} entries to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
