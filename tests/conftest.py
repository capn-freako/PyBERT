"""Shared fixtures across the pybert testing infrastructure."""

import pytest

from pybert.pybert import PyBERT


@pytest.fixture(scope="module")
def dut():
    """Return an initialized pybert object that has already run the initial simulation."""
    yield PyBERT(gui=False)
