"""Unit tests for ``scripts/runpod_smoke.py``.

The smoke script runs in the Docker container against the real installed
dependencies (NeMo, the patch, the dictionary file). These unit tests
verify the script's *logic* — argument parsing, check dispatch, failure
modes, exit codes — using mocked imports and a temporary filesystem.

The actual ``check_imports`` and ``check_patch`` functions are
intentionally tested only through their ability to surface clear error
messages when the underlying module isn't there — we do NOT install NeMo
in CI just to satisfy these tests.

Run from the repo root::

    pytest tests/test_runpod_smoke.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

# The smoke script lives in scripts/. Make sure the repo root is on the
# import path so we can ``from scripts import runpod_smoke``.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import runpod_smoke


# ---------------------------------------------------------------------------
# get_check_function — dispatch
# ---------------------------------------------------------------------------

class TestGetCheckFunction:
    """The dispatcher maps (check_name, variant) → callable."""

    def test_imports_returns_check_imports(self):
        fn = runpod_smoke.get_check_function("imports", "amchi")
        assert fn is runpod_smoke.check_imports

    def test_patch_returns_check_patch(self):
        fn = runpod_smoke.get_check_function("patch", "amchi")
        assert fn is runpod_smoke.check_patch

    def test_dictionary_amchi_returns_check_dictionary(self):
        fn = runpod_smoke.get_check_function("dictionary", "amchi")
        assert fn is runpod_smoke.check_dictionary

    def test_dictionary_deaf_returns_no_op(self):
        """The deaf variant has no dictionary — the dispatcher returns a no-op."""
        fn = runpod_smoke.get_check_function("dictionary", "deaf")
        # Calling it should not raise.
        fn()

    def test_handler_returns_callable_for_variant(self):
        """The handler check is variant-bound."""
        fn = runpod_smoke.get_check_function("handler", "amchi")
        assert callable(fn)

    def test_unknown_check_raises(self):
        with pytest.raises(ValueError, match="Unknown check"):
            runpod_smoke.get_check_function("nope", "amchi")


# ---------------------------------------------------------------------------
# check_dictionary — file existence + parseability
# ---------------------------------------------------------------------------

class TestCheckDictionary:
    """Covers the dictionary-file validation logic."""

    def test_typical_valid_dictionary(self, tmp_path):
        """A normal JSON list with Devanagari words passes."""
        path = tmp_path / "dict.json"
        path.write_text(
            json.dumps([
                {"word_konkani_devanagari": "घर", "meaning": "house"},
                {"word_konkani_devanagari": "वाट", "meaning": "path"},
            ], ensure_ascii=False),
            encoding="utf-8",
        )
        runpod_smoke.check_dictionary(str(path))  # should not raise

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="Dictionary JSON not found"):
            runpod_smoke.check_dictionary(str(tmp_path / "missing.json"))

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / "dict.json"
        path.write_text("not valid json {{{", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Failed to parse dictionary"):
            runpod_smoke.check_dictionary(str(path))

    def test_non_list_top_level_raises(self, tmp_path):
        path = tmp_path / "dict.json"
        path.write_text('{"data": []}', encoding="utf-8")
        with pytest.raises(RuntimeError, match="must be a list"):
            runpod_smoke.check_dictionary(str(path))

    def test_empty_list_raises(self, tmp_path):
        path = tmp_path / "dict.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(RuntimeError, match="zero usable entries"):
            runpod_smoke.check_dictionary(str(path))

    def test_list_with_no_devanagari_field_raises(self, tmp_path):
        path = tmp_path / "dict.json"
        path.write_text(
            json.dumps([{"english_only": "house"}, {"foo": "bar"}]),
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="zero usable entries"):
            runpod_smoke.check_dictionary(str(path))

    def test_list_with_whitespace_only_devanagari_raises(self, tmp_path):
        path = tmp_path / "dict.json"
        path.write_text(
            json.dumps([{"word_konkani_devanagari": "   "}]),
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="zero usable entries"):
            runpod_smoke.check_dictionary(str(path))


# ---------------------------------------------------------------------------
# check_handler — module + handler() shape
# ---------------------------------------------------------------------------

class TestCheckHandler:
    """Covers the handler-import + graceful-error contract."""

    def test_typical_amchi_handler_empty_input(self):
        """The real amchi handler should error out on empty input, not raise."""
        # We DO import the real handler (the heavy deps are not invoked because
        # the handler short-circuits on missing audio_base64 / audio_url before
        # touching the model or dictionary).
        runpod_smoke.check_handler("amchi")  # should not raise

    def test_unknown_variant_raises(self):
        with pytest.raises(RuntimeError, match="Unknown variant"):
            runpod_smoke.check_handler("klingon")

    def _make_fake_variant(self, tmp_path, body: str, variant: str = "fake"):
        """Write a tiny handler.py to tmp_path and return a VARIANT_CONFIG patch dict."""
        handler_file = tmp_path / f"{variant}_handler.py"
        handler_file.write_text(body, encoding="utf-8")
        return {
            variant: {
                "handler_module": f"fake.{variant}",
                "handler_path": str(handler_file),
                "needs_dictionary": False,
            }
        }

    def test_handler_module_import_failure_raises(self, tmp_path):
        """If the handler source file is missing, the error names the path."""
        # Patch in a variant whose handler_path doesn't exist on disk.
        missing_path = str(tmp_path / "does_not_exist.py")
        with mock.patch.dict(
            runpod_smoke.VARIANT_CONFIG,
            {"bogus": {
                "handler_module": "this.does.not.exist",
                "handler_path": missing_path,
                "needs_dictionary": False,
            }},
            clear=False,
        ):
            with pytest.raises(RuntimeError, match="Handler source file not found"):
                runpod_smoke.check_handler("bogus")

    def test_handler_load_failure_raises(self, tmp_path):
        """If the handler source file is syntactically broken, the error names the module."""
        patch_cfg = self._make_fake_variant(
            tmp_path, "this is not valid python !!!", variant="brokensyntax"
        )
        with mock.patch.dict(runpod_smoke.VARIANT_CONFIG, patch_cfg, clear=False):
            with pytest.raises(RuntimeError, match="Failed to load fake.brokensyntax"):
                runpod_smoke.check_handler("brokensyntax")

    def test_handler_must_be_callable(self, tmp_path):
        """If the module loads but exports no callable 'handler', we fail."""
        patch_cfg = self._make_fake_variant(
            tmp_path, "handler = 'not callable'\n", variant="notcallable"
        )
        with mock.patch.dict(runpod_smoke.VARIANT_CONFIG, patch_cfg, clear=False):
            with pytest.raises(RuntimeError, match="does not export a callable"):
                runpod_smoke.check_handler("notcallable")

    def test_handler_must_return_dict(self, tmp_path):
        """A handler that returns a non-dict on empty input fails the check."""
        patch_cfg = self._make_fake_variant(
            tmp_path,
            "def handler(job):\n    return 'I am not a dict'\n",
            variant="notdict",
        )
        with mock.patch.dict(runpod_smoke.VARIANT_CONFIG, patch_cfg, clear=False):
            with pytest.raises(RuntimeError, match="expected dict"):
                runpod_smoke.check_handler("notdict")

    def test_handler_must_produce_error_field(self, tmp_path):
        """A handler that returns a dict without an 'error' key on empty input fails."""
        patch_cfg = self._make_fake_variant(
            tmp_path,
            "def handler(job):\n    return {'transcription': 'I made one up'}\n",
            variant="noerror",
        )
        with mock.patch.dict(runpod_smoke.VARIANT_CONFIG, patch_cfg, clear=False):
            with pytest.raises(RuntimeError, match="did not produce an 'error'"):
                runpod_smoke.check_handler("noerror")

    def test_handler_raising_exception_is_caught(self, tmp_path):
        """A handler that raises (rather than returning an error-dict) fails the check."""
        patch_cfg = self._make_fake_variant(
            tmp_path,
            "def handler(job):\n    raise ValueError('model not configured')\n",
            variant="raises",
        )
        with mock.patch.dict(runpod_smoke.VARIANT_CONFIG, patch_cfg, clear=False):
            with pytest.raises(RuntimeError, match="raised an exception on empty input"):
                runpod_smoke.check_handler("raises")


# ---------------------------------------------------------------------------
# run_checks — orchestration + exit code
# ---------------------------------------------------------------------------

class TestRunChecks:
    """The orchestrator should run checks in order, capture failures, and
    return the right exit code."""

    def test_all_pass_returns_zero(self, capsys, tmp_path, monkeypatch):
        """When every check succeeds, run_checks returns 0."""
        # Build a valid dictionary file so check_dictionary passes.
        d = tmp_path / "d.json"
        d.write_text(json.dumps([{"word_konkani_devanagari": "घर"}]), encoding="utf-8")
        monkeypatch.setenv("AMCHI_DICT_PATH", str(d))

        # Mock imports + patch so they pass without real NeMo
        with mock.patch.object(runpod_smoke, "check_imports", lambda: None), \
             mock.patch.object(runpod_smoke, "check_patch", lambda: None):
            rc = runpod_smoke.run_checks(
                ["imports", "patch", "dictionary"], "amchi", verbose=True
            )
        assert rc == 0
        out = capsys.readouterr().out
        assert "PASS" in out
        assert "FAIL" not in out

    def test_one_failure_returns_one(self, capsys, monkeypatch):
        """Even a single failure flips the exit code to 1."""
        with mock.patch.object(runpod_smoke, "check_imports", lambda: None), \
             mock.patch.object(
                 runpod_smoke, "check_patch",
                 side_effect=RuntimeError("patch missing"),
             ):
            rc = runpod_smoke.run_checks(["imports", "patch"], "amchi", verbose=True)
        assert rc == 1
        out = capsys.readouterr().out
        assert "FAIL: patch missing" in out

    def test_failures_listed_in_summary(self, capsys):
        """The per-check error messages appear in the final summary."""
        with mock.patch.object(
            runpod_smoke, "check_imports",
            side_effect=RuntimeError("nemo not found"),
        ), mock.patch.object(
            runpod_smoke, "check_patch",
            side_effect=RuntimeError("patch missing"),
        ):
            rc = runpod_smoke.run_checks(["imports", "patch"], "amchi", verbose=True)
        assert rc == 1
        out = capsys.readouterr().out
        assert "[imports] nemo not found" in out
        assert "[patch] patch missing" in out

    def test_quiet_mode_no_output(self, capsys):
        """``verbose=False`` suppresses PASS/FAIL output."""
        with mock.patch.object(runpod_smoke, "check_imports", lambda: None):
            rc = runpod_smoke.run_checks(["imports"], "amchi", verbose=False)
        assert rc == 0
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# main — CLI argument parsing and dispatch
# ---------------------------------------------------------------------------

class TestMain:
    """End-to-end CLI behaviour: --check, --variant, --quiet, exit codes."""

    def test_default_runs_all_checks(self):
        """Without --check, the CLI runs every check."""
        # Mock all checks to pass so we just exercise the dispatch path.
        with mock.patch.object(runpod_smoke, "check_imports", lambda: None), \
             mock.patch.object(runpod_smoke, "check_patch", lambda: None), \
             mock.patch.object(runpod_smoke, "check_dictionary", lambda *a, **k: None), \
             mock.patch.object(runpod_smoke, "check_handler", lambda *a, **k: None):
            rc = runpod_smoke.main(["--quiet"])
        assert rc == 0

    def test_single_check(self, capsys):
        """``--check imports`` runs only that check."""
        with mock.patch.object(runpod_smoke, "check_imports", lambda: None):
            rc = runpod_smoke.main(["--check", "imports"])
        assert rc == 0
        # Other checks should NOT have shown up in the output.
        out = capsys.readouterr().out
        assert "[1/1] imports" in out

    def test_failure_exits_one(self):
        with mock.patch.object(
            runpod_smoke, "check_imports",
            side_effect=RuntimeError("nemo missing"),
        ):
            rc = runpod_smoke.main(["--check", "imports", "--quiet"])
        assert rc == 1

    def test_invalid_check_argument_exits_two(self):
        """argparse rejects unknown --check values with exit code 2."""
        with pytest.raises(SystemExit) as exc_info:
            runpod_smoke.main(["--check", "bogus"])
        assert exc_info.value.code == 2

    def test_deaf_variant_skips_dictionary(self, capsys):
        """The deaf variant has no dictionary check."""
        with mock.patch.object(runpod_smoke, "check_imports", lambda: None), \
             mock.patch.object(runpod_smoke, "check_patch", lambda: None), \
             mock.patch.object(runpod_smoke, "check_handler", lambda *a, **k: None):
            rc = runpod_smoke.main(["--variant", "deaf"])
        assert rc == 0
        # The "dictionary" line still appears (it's a no-op) — that's fine.
        out = capsys.readouterr().out
        assert "[1/4] imports" in out
