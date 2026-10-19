"""The DFE slicer level (`decision_scaler`) is a user setting the simulation must not clobber."""
import pytest

from pybert.models.dfe import DFE
from pybert.pybert import PyBERT


def _dut() -> PyBERT:
    dut = PyBERT(run_simulation=False, gui=False)
    dut.nbits = 2000
    dut.eye_bits = 1000
    dut.decision_scaler = 0.37
    return dut


def test_level_unchanged_without_agc():
    """Without AGC, a simulation leaves the slicer level where the user set it."""
    dut = _dut()
    dut.use_agc = False
    dut.simulate(initial_run=True, update_plots=False)
    assert dut.decision_scaler == pytest.approx(0.37)


def test_level_survives_paths_that_dont_report_it(monkeypatch):
    """A run that never reports a slicer level back (e.g. IBIS-AMI `GetWave()` with its own clocks, or a failure
    in the DFE) leaves the user's level in place, rather than 0 V -- which a saved config, and every later run,
    would then inherit as a collapsed slicer."""
    def fail(*args, **kwargs):
        raise RuntimeError("DFE failed")

    dut = _dut()
    monkeypatch.setattr(DFE, "run_dfe", fail)
    with pytest.raises(RuntimeError):
        dut.simulate(initial_run=True, update_plots=False)
    assert dut.decision_scaler == pytest.approx(0.37)
