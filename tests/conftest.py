"""Shared fixtures across the pybert testing infrastructure."""

import pytest

from pybert.pybert import PyBERT


@pytest.fixture(scope="module")
def dut():
    """Return an initialized pybert object that has already run the initial simulation."""
    yield PyBERT(gui=False)

@pytest.fixture(scope="module")
def dut_imp_len():
    """Return an initialized pybert object with manually controlled channel impulse response length."""
    dut = PyBERT(run_simulation=False, gui=False)
    dut.impulse_length = 10  # (ns)
    dut.simulate(initial_run=True)
    yield dut
