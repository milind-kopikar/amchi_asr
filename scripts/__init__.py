"""Marker file: makes ``scripts/`` a regular package, not an implicit namespace package.

Without this, Python's path-based finder can produce inconsistent results in
edge cases — especially when a module is loaded via
``importlib.util.spec_from_file_location`` (as ``scripts/runpod_smoke.py``
does for the RunPod handlers). With a real ``__init__.py``, the import
system registers ``scripts`` as a regular package in ``sys.modules`` the
first time anything imports from it, and subsequent lookups are stable.

Intentionally empty — this is only a marker.
"""
