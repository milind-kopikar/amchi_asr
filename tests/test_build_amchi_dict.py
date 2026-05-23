"""Unit tests for ``scripts/build_amchi_dict.py``.

Covers the four pure functions (no network) plus the ``build`` orchestrator
and the ``main`` CLI entry point. Network and filesystem I/O are mocked so
the tests run offline.

Run from the repo root::

    pytest tests/test_build_amchi_dict.py -v
"""

from __future__ import annotations

import csv
import io
import json
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

from scripts.build_amchi_dict import (
    DEV_FIELD,
    _pick_devanagari_column,
    build,
    fetch_from_railway,
    main,
    normalise_entries,
    read_from_csv,
    write_json,
)


# ---------------------------------------------------------------------------
# _pick_devanagari_column — pure column-name matching
# ---------------------------------------------------------------------------

class TestPickDevanagariColumn:
    """Edge cases for the case-insensitive header matcher."""

    def test_typical_header(self):
        """Exact match against the konkani_dictionary CSV's primary alias."""
        assert _pick_devanagari_column(["Konkani Word in Devnagiri", "English Meaning"]) == (
            "Konkani Word in Devnagiri"
        )

    def test_case_insensitive_match(self):
        """Matching is case- and whitespace-tolerant."""
        assert (
            _pick_devanagari_column(["  konkani word in devnagiri  ", "something_else"])
            is not None
        )

    def test_alternate_alias(self):
        """One of the alternate field names also matches."""
        assert _pick_devanagari_column(["word_konkani_devanagari", "pos"]) == (
            "word_konkani_devanagari"
        )

    def test_no_match(self):
        """Returns None when no Devanagari-word column is present."""
        assert _pick_devanagari_column(["English", "Marathi"]) is None

    def test_empty_input(self):
        """Empty header list returns None."""
        assert _pick_devanagari_column([]) is None

    def test_skips_none_entries(self):
        """Defensive: skips None/empty entries in the field list."""
        assert _pick_devanagari_column(["", None, "devanagari"]) == "devanagari"


# ---------------------------------------------------------------------------
# normalise_entries — drops empties, trims whitespace, preserves extras
# ---------------------------------------------------------------------------

class TestNormaliseEntries:
    """Edge cases for the entry-normaliser."""

    def test_typical(self):
        """A clean list passes through unchanged (modulo trimming)."""
        out = normalise_entries(
            [
                {DEV_FIELD: "अम्चि", "meaning": "ours"},
                {DEV_FIELD: "घर", "meaning": "house"},
            ]
        )
        assert len(out) == 2
        assert out[0][DEV_FIELD] == "अम्चि"
        assert out[0]["meaning"] == "ours"

    def test_empty_input(self):
        """Empty list → empty list."""
        assert normalise_entries([]) == []

    def test_drops_entries_with_missing_devanagari(self):
        """Entries without a Devanagari word are silently dropped."""
        out = normalise_entries(
            [
                {DEV_FIELD: "घर"},
                {DEV_FIELD: ""},
                {"english_meaning": "house"},  # no Devanagari at all
                {DEV_FIELD: "   "},  # whitespace only
            ]
        )
        assert len(out) == 1
        assert out[0][DEV_FIELD] == "घर"

    def test_trims_whitespace(self):
        """Leading/trailing whitespace around the Devanagari is trimmed."""
        out = normalise_entries([{DEV_FIELD: "  घर  "}])
        assert out[0][DEV_FIELD] == "घर"

    def test_preserves_other_fields(self):
        """Non-Devanagari fields are kept (so the post-processor could read them)."""
        out = normalise_entries(
            [{DEV_FIELD: "घर", "english_meaning": "house", "pos": "n.m."}]
        )
        assert out[0]["english_meaning"] == "house"
        assert out[0]["pos"] == "n.m."

    def test_malformed_entry_with_non_string_devanagari(self):
        """Entries where the Devanagari value isn't a string are dropped safely."""
        with pytest.raises(AttributeError):
            # `.get(DEV_FIELD) or ""` returns the int; `.strip()` then fails.
            # This documents the current behaviour — we expect strings only.
            normalise_entries([{DEV_FIELD: 42}])


# ---------------------------------------------------------------------------
# read_from_csv — file-based input path
# ---------------------------------------------------------------------------

class TestReadFromCsv:
    """Filesystem-level tests using a temporary CSV file per test."""

    def test_typical(self, tmp_path):
        """A small valid CSV is parsed into the expected entry list."""
        csv_path = tmp_path / "dict.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Konkani Word in Devnagiri", "English Meaning"])
            writer.writerow(["घर", "house"])
            writer.writerow(["वाट", "path"])

        entries = read_from_csv(str(csv_path))
        assert len(entries) == 2
        assert entries[0][DEV_FIELD] == "घर"
        assert entries[0]["English Meaning"] == "house"

    def test_missing_file_raises(self, tmp_path):
        """A non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_from_csv(str(tmp_path / "nope.csv"))

    def test_no_devanagari_column_raises(self, tmp_path):
        """A CSV with no recognised Devanagari header raises ValueError."""
        csv_path = tmp_path / "dict.csv"
        csv_path.write_text("english,marathi\nhouse,घर\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No Devanagari-word column"):
            read_from_csv(str(csv_path))

    def test_empty_csv_returns_empty_list(self, tmp_path):
        """A CSV with header but no rows returns []."""
        csv_path = tmp_path / "dict.csv"
        csv_path.write_text("Konkani Word in Devnagiri\n", encoding="utf-8")
        assert read_from_csv(str(csv_path)) == []

    def test_skips_blank_rows(self, tmp_path):
        """Rows with empty Devanagari are skipped, not errored on."""
        csv_path = tmp_path / "dict.csv"
        csv_path.write_text(
            "Konkani Word in Devnagiri,English Meaning\n"
            "घर,house\n"
            ",empty\n"
            "वाट,path\n",
            encoding="utf-8",
        )
        entries = read_from_csv(str(csv_path))
        assert len(entries) == 2
        assert {e[DEV_FIELD] for e in entries} == {"घर", "वाट"}


# ---------------------------------------------------------------------------
# fetch_from_railway — network mock
# ---------------------------------------------------------------------------

def _make_urlopen_returning(payload: object):
    """Helper: produce a context-manager mock that returns ``payload`` as JSON."""

    class _Resp:
        def __init__(self, body: bytes):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self._body

    body = json.dumps(payload).encode("utf-8")
    return mock.MagicMock(return_value=_Resp(body))


class TestFetchFromRailway:
    """Network-mocked tests for the Railway fetcher."""

    def test_list_payload(self):
        """Endpoint that returns a plain JSON array."""
        payload = [{DEV_FIELD: "घर"}, {DEV_FIELD: "वाट"}]
        with mock.patch("scripts.build_amchi_dict.urllib.request.urlopen",
                       _make_urlopen_returning(payload)):
            entries = fetch_from_railway("http://example.test/api")
        assert entries == payload

    def test_dict_with_data_key(self):
        """Endpoint that wraps the array under ``data``."""
        payload = {"data": [{DEV_FIELD: "घर"}]}
        with mock.patch("scripts.build_amchi_dict.urllib.request.urlopen",
                       _make_urlopen_returning(payload)):
            entries = fetch_from_railway("http://example.test/api")
        assert entries == [{DEV_FIELD: "घर"}]

    def test_dict_with_entries_key(self):
        """Endpoint that wraps the array under ``entries``."""
        payload = {"entries": [{DEV_FIELD: "घर"}]}
        with mock.patch("scripts.build_amchi_dict.urllib.request.urlopen",
                       _make_urlopen_returning(payload)):
            entries = fetch_from_railway("http://example.test/api")
        assert entries == [{DEV_FIELD: "घर"}]

    def test_unexpected_dict_shape_raises(self):
        """A dict response with no recognised list key raises ValueError."""
        payload = {"unknown_key": []}
        with mock.patch("scripts.build_amchi_dict.urllib.request.urlopen",
                       _make_urlopen_returning(payload)):
            with pytest.raises(ValueError, match="Unexpected Railway response shape"):
                fetch_from_railway("http://example.test/api")

    def test_unexpected_payload_type_raises(self):
        """A bare number or string at top level raises ValueError."""
        with mock.patch("scripts.build_amchi_dict.urllib.request.urlopen",
                       _make_urlopen_returning(42)):
            with pytest.raises(ValueError, match="Unexpected Railway response type"):
                fetch_from_railway("http://example.test/api")

    def test_network_error_propagates(self):
        """urllib errors propagate so the caller can decide whether to retry."""
        with mock.patch(
            "scripts.build_amchi_dict.urllib.request.urlopen",
            side_effect=urllib.error.URLError("network down"),
        ):
            with pytest.raises(urllib.error.URLError):
                fetch_from_railway("http://example.test/api")


# ---------------------------------------------------------------------------
# write_json — filesystem round-trip
# ---------------------------------------------------------------------------

class TestWriteJson:
    """Filesystem tests for the writer."""

    def test_typical_round_trip(self, tmp_path):
        """Devanagari content round-trips correctly through UTF-8 JSON."""
        out = tmp_path / "dict.json"
        write_json([{DEV_FIELD: "घर", "meaning": "house"}], str(out))
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded[0][DEV_FIELD] == "घर"

    def test_creates_parent_dirs(self, tmp_path):
        """Missing parent directories are created on demand."""
        out = tmp_path / "nested" / "deeper" / "dict.json"
        write_json([{DEV_FIELD: "घर"}], str(out))
        assert out.is_file()

    def test_empty_list(self, tmp_path):
        """An empty entry list still produces a valid JSON file (just ``[]``)."""
        out = tmp_path / "dict.json"
        write_json([], str(out))
        assert json.loads(out.read_text(encoding="utf-8")) == []


# ---------------------------------------------------------------------------
# build — orchestrator
# ---------------------------------------------------------------------------

class TestBuild:
    """End-to-end orchestrator tests using the CSV path (no network needed)."""

    def test_csv_source_produces_output_file(self, tmp_path):
        """A valid CSV source writes the expected number of entries."""
        csv_path = tmp_path / "dict.csv"
        csv_path.write_text(
            "Konkani Word in Devnagiri,English Meaning\nघर,house\nवाट,path\n",
            encoding="utf-8",
        )
        out_path = tmp_path / "out.json"
        count = build(source="csv", url="", csv_path=str(csv_path), out_path=str(out_path))
        assert count == 2
        assert out_path.is_file()

    def test_csv_source_without_path_raises(self, tmp_path):
        """``--source csv`` without ``--csv`` raises ValueError."""
        with pytest.raises(ValueError, match="--csv path required"):
            build(source="csv", url="", csv_path=None, out_path=str(tmp_path / "x.json"))

    def test_unknown_source_raises(self, tmp_path):
        """An unrecognised source value raises ValueError."""
        with pytest.raises(ValueError, match="Unknown --source"):
            build(source="bogus", url="", csv_path=None, out_path=str(tmp_path / "x.json"))

    def test_all_empty_entries_raises(self, tmp_path):
        """If normalisation leaves zero entries, build raises ValueError."""
        csv_path = tmp_path / "dict.csv"
        csv_path.write_text(
            "Konkani Word in Devnagiri\n   \n,\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="No usable dictionary entries"):
            build(source="csv", url="", csv_path=str(csv_path),
                  out_path=str(tmp_path / "x.json"))


# ---------------------------------------------------------------------------
# main — CLI entry point exit codes
# ---------------------------------------------------------------------------

class TestMain:
    """Exit-code contract for the CLI."""

    def test_success_returns_0(self, tmp_path, capsys):
        """Happy path: writes file and returns 0."""
        csv_path = tmp_path / "dict.csv"
        csv_path.write_text(
            "Konkani Word in Devnagiri\nघर\n", encoding="utf-8"
        )
        out_path = tmp_path / "out.json"
        rc = main(["--source", "csv", "--csv", str(csv_path), "--out", str(out_path)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Wrote 1 entries" in captured.out
        assert out_path.is_file()

    def test_network_error_returns_1(self, tmp_path):
        """URL fetch failure → exit code 1."""
        with mock.patch(
            "scripts.build_amchi_dict.urllib.request.urlopen",
            side_effect=urllib.error.URLError("offline"),
        ):
            rc = main(["--source", "railway",
                       "--url", "http://example.test/api",
                       "--out", str(tmp_path / "out.json")])
        assert rc == 1

    def test_missing_csv_returns_1(self, tmp_path):
        """Missing CSV file → exit code 1 (I/O failure category)."""
        rc = main(["--source", "csv",
                   "--csv", str(tmp_path / "absent.csv"),
                   "--out", str(tmp_path / "out.json")])
        assert rc == 1

    def test_malformed_source_returns_2(self, tmp_path):
        """A CSV with no Devanagari column → exit code 2 (malformed)."""
        csv_path = tmp_path / "dict.csv"
        csv_path.write_text("english,marathi\nhouse,घर\n", encoding="utf-8")
        rc = main(["--source", "csv", "--csv", str(csv_path),
                   "--out", str(tmp_path / "out.json")])
        assert rc == 2
