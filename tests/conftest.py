"""Shared fixtures across the pybert testing infrastructure."""

import pytest

from pybert.pybert import PyBERT
from pybert.gui.handler import MyHandler


class DummyInfo():
    """Dummy ``Info`` object, for making use of convenience functions in ``pybert.gui.handler.MyHandler``."""
    object: PyBERT = None

    def __init__(self, thePyBERT: PyBERT):
        self.object = thePyBERT


@pytest.fixture(scope="module")
def optimization_triplet():
    """
    Return a triplet for testing the optimizer, containing

    - an initialized PyBERT object that has NOT run the initial simulation,
    - an instance of ``pybert.gui.handler.MyHandler``, and
    - a dummy ``Info`` object pre-initialized with a pointer to the returned PyBERT object.
    """

    thePyBERT = PyBERT(run_simulation=False, gui=False)
    theHandler = MyHandler()
    theInfo = DummyInfo(thePyBERT)

    yield (thePyBERT, theHandler, theInfo)

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

@pytest.fixture(scope="module")
def dut_viterbi():
    """Return an initialized pybert object with Viterbi decoder enabled."""
    dut = PyBERT(run_simulation=False, gui=False)
    dut.rx_use_viterbi = True
    dut.simulate(initial_run=True)
    yield dut

@pytest.fixture(scope="module")
def dut_viterbi_stressed():
    """Return an initialized pybert object with Viterbi decoder enabled and eye stressed."""
    dut = PyBERT(run_simulation=False, gui=False)
    dut.rx_use_viterbi = True
    dut.l_ch = 2.0
    dut.simulate(initial_run=True)
    yield dut

@pytest.fixture(scope="module")
def ibisami_rx_init():
    """
    Return an initialized pybert object configured to use
    an Rx IBIS-AMI model in statistical mode.
    """
    dut = PyBERT(run_simulation=False, gui=False)
    dut.rx_ibis_file = "models/ibisami/example_rx.ibs"
    dut.rx_use_ibis = True
    dut.rx_use_ami = True
    dut.simulate(initial_run=True)
    yield dut

@pytest.fixture(scope="module")
def ibisami_rx_getwave():
    """
    Return an initialized pybert object configured to use
    an Rx IBIS-AMI model in bit-by-bit mode.
    """
    dut = PyBERT(run_simulation=False, gui=False)
    dut.rx_ibis_file = "models/ibisami/example_rx.ibs"
    dut.rx_use_ibis = True
    dut.rx_use_ami = True
    dut.rx_use_getwave = True
    dut.simulate(initial_run=True)
    yield dut

@pytest.fixture(scope="module")
def ibisami_rx_getwave_clocked():
    """
    Return an initialized pybert object configured to use
    an Rx IBIS-AMI model in bit-by-bit mode and making use of clock times.
    """
    dut = PyBERT(run_simulation=False, gui=False)
    dut.rx_ibis_file = "models/ibisami/example_rx.ibs"
    dut.rx_use_ibis = True
    dut.rx_use_ami = True
    dut.rx_use_getwave = True
    dut.rx_use_clocks = True
    dut.simulate(initial_run=True)
    yield dut
