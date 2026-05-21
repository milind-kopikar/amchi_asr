"""Integration tests — heavyweight tests that require a GPU, the fine-tuned
checkpoint, and the held-out test audio.

Skipped by default. To run them locally::

    export RUN_GOLDEN_TESTS=1
    pytest tests/integration/ -v

Or run a specific variant::

    pytest tests/integration/test_amchi_golden.py -v
"""
