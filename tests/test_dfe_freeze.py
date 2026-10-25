"""Test freezing the DFE tap weights, once they've stabilized (``dfe_freeze``)."""
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from pybert.models.dfe import DFE
from pybert.pybert import PyBERT
from pybert.utility import lfsr_bits

UI = 100e-12
NSPUI = 8
PERIOD = 2 * (2**7 - 1)  # PRBS-7, doubled as `my_run_simulation()` does.
H = (0.3, 0.1)  # Post-cursor ISI, relative to the cursor.


def run_dfe(freeze_window: int, freeze_tol: float = 2e-3, n_taps: int = len(H),
            use_agc: bool = False, dbg: Optional[dict] = None, freeze_min_ui: int = 0) -> tuple[np.ndarray, int]:
    """Run an NRZ DFE, clocked at a fixed phase, over a noisy channel w/ known ISI.
    (The noise is heavy enough that the tolerance must be looser than the default, to freeze.)
    With AGC on, its level starts low (0.4 V; ~0.5 V is right).

    Returns:
        The tap weights at each UI, and the UI at which they were frozen (-1 if never).
    """
    gen = lfsr_bits([7, 6], 1)
    syms = np.array([next(gen) for _ in range(40 * PERIOD)]) * 2.0 - 1.0
    x = 0.5 * (syms + H[0] * np.roll(syms, 1) + H[1] * np.roll(syms, 2))
    x += np.random.default_rng(0).normal(0, 0.03, len(x))
    sig = np.repeat(x, NSPUI)
    t = np.arange(len(sig)) * UI / NSPUI
    dfe = DFE(n_taps, 0.1, 1e-12, 0.01, UI, NSPUI, 0.4 if use_agc else 0.5, n_ave=10, bandwidth=10e9,
              freeze_window=freeze_window, freeze_tol=freeze_tol, freeze_min_ui=freeze_min_ui)
    if dbg is None:
        dbg = {}
    tap_weights = dfe.run_dfe(t, sig, use_agc=use_agc, dbg_dict=dbg, clock_phase=UI / 2)[1]
    dbg["decision_scaler"] = dfe.decision_scaler
    return np.array(tap_weights), dbg["freeze_ui"]


def test_freezes_at_converged_solution():
    """Tap weights freeze near the channel's ISI and stay put thereafter."""
    tap_weights, freeze_ui = run_dfe(PERIOD)
    assert 0 <= freeze_ui < len(tap_weights) - PERIOD, "DFE never froze!"
    assert np.all(tap_weights[freeze_ui + 1:] == tap_weights[-1]), "Tap weights changed after freezing!"
    assert tap_weights[-1] == pytest.approx(H, abs=0.01)


def test_hunts_without_freezing():
    """Without freezing, the tap weights keep moving to the end of the run."""
    tap_weights, freeze_ui = run_dfe(0)
    assert freeze_ui == -1
    assert np.ptp(tap_weights[-PERIOD:], axis=0).max() > 0


def test_zero_tol_never_freezes():
    """A zero tolerance on a noisy channel never freezes, matching no freezing at all."""
    tap_weights, freeze_ui = run_dfe(PERIOD, freeze_tol=0.0)
    assert freeze_ui == -1
    assert np.array_equal(tap_weights, run_dfe(0)[0])


def test_min_ui_delays_freeze():
    """An already stable solution freezes just in time to equalize UI `freeze_min_ui`, at the same values."""
    tap_weights, freeze_ui = run_dfe(PERIOD)
    min_ui = freeze_ui + 5 * PERIOD + 7  # Mid-window, well after the unconstrained freeze.
    tap_weights_min, freeze_ui_min = run_dfe(PERIOD, freeze_min_ui=min_ui)
    assert freeze_ui_min == min_ui - 1
    assert np.all(tap_weights_min[min_ui:] == tap_weights_min[-1]), "Tap weights changed after freezing!"
    assert np.any(tap_weights_min[freeze_ui + 1:min_ui] != tap_weights_min[-1]), "Froze before `freeze_min_ui`!"
    assert tap_weights_min[-1] == pytest.approx(tap_weights[-1], abs=0.01)


def test_min_ui_beyond_run():
    """No freeze when `freeze_min_ui` lies beyond the end of the run."""
    tap_weights, freeze_ui = run_dfe(PERIOD, freeze_min_ui=50 * PERIOD)
    assert freeze_ui == -1


def test_agc_freezes_with_taps():
    """With AGC on, its level freezes along with the taps: no level changes after the freeze."""
    dbg: dict = {}
    tap_weights, freeze_ui = run_dfe(PERIOD, use_agc=True, dbg=dbg)
    assert 0 <= freeze_ui < len(tap_weights) - PERIOD, "DFE never froze!"
    assert np.all(tap_weights[freeze_ui + 1:] == tap_weights[-1]), "Tap weights changed after freezing!"
    # The AGC sets a new level each UI, once its averages fill (2 x 100 UI); the freeze sets one more.
    assert len(dbg["scalar_values"]) <= freeze_ui + 2, "AGC level changed after freezing!"
    assert dbg["scalar_values"][-1] == dbg["decision_scaler"] == pytest.approx(0.5, rel=0.02)


def test_agc_hunts_without_freezing():
    """Without freezing, the AGC keeps setting new levels to the end of the run."""
    dbg: dict = {}
    tap_weights = run_dfe(0, use_agc=True, dbg=dbg)[0]
    assert len(dbg["scalar_values"]) > len(tap_weights) - 2 * PERIOD
    assert np.ptp(dbg["scalar_values"][-PERIOD:]) > 0


def test_agc_not_frozen_without_taps():
    """With no DFE taps, nothing triggers a freeze, so the AGC keeps adapting."""
    dbg: dict = {}
    tap_weights, freeze_ui = run_dfe(PERIOD, n_taps=0, use_agc=True, dbg=dbg)
    assert freeze_ui == -1
    assert len(dbg["scalar_values"]) > len(tap_weights) - 2 * PERIOD


def test_config_round_trip(tmp_path: Path):
    """``dfe_freeze``, ``dfe_freeze_tol``, and ``dfe_freeze_nwin`` survive a YAML save/load round-trip."""
    dut = PyBERT(run_simulation=False, gui=False)
    dut.dfe_freeze = True
    dut.dfe_freeze_tol = 2.5e-3
    dut.dfe_freeze_nwin = 5
    save_file = tmp_path / "dfe_freeze.yaml"
    dut.save_configuration(save_file)

    dut2 = PyBERT(run_simulation=False, gui=False)
    dut2.load_configuration(save_file)
    assert dut2.dfe_freeze is True
    assert dut2.dfe_freeze_tol == 2.5e-3
    assert dut2.dfe_freeze_nwin == 5


@pytest.mark.parametrize("pdut", [{"dfe_freeze": True, "gain": 0.5, "nbits": 25000}], indirect=True)
class TestDfeFreezeSim:
    """Default configuration, simulated with DFE freezing enabled.
    (The default gain is too low, and the default pre-eye run too short, for the DFE to stabilize before the eye.)"""

    def test_ber(self, pdut):
        """No bit errors."""
        assert pdut.n_errs_dfe == 0, "Bit errors detected!"

    def test_frozen(self, pdut):
        """The DFE freezes, and neither its tap weights nor its AGC level change thereafter."""
        if not any(t.enabled for t in pdut.dfe_tap_tuners):
            pytest.skip("No DFE taps enabled in the default configuration.")
        assert pdut.dfe_freeze_ui >= 0, "DFE never froze!"
        tap_weights = np.array(pdut.adaptation)
        assert np.all(tap_weights[pdut.dfe_freeze_ui + 1:] == tap_weights[-1])
        assert len(pdut.dfe_scalar_values) <= pdut.dfe_freeze_ui + 2, "AGC level changed after freezing!"

    def test_no_gain_warning(self, pdut):
        """With ample adaptation time before the eye, there's no DFE gain warning."""
        assert pdut.dfe_tau_ui > 0
        assert pdut.dfe_warning == ""

    def test_frozen_for_eye(self, pdut):
        """Already stable, the DFE freezes just as eye data collection begins, holding steady across the eye."""
        if not any(t.enabled for t in pdut.dfe_tap_tuners):
            pytest.skip("No DFE taps enabled in the default configuration.")
        assert pdut.dfe_freeze_ui == pdut.nui - pdut.eye_uis - 1
        # `adaptation[k]` holds the tap weights used at clock `k`.
        n_clocks = len(pdut.adaptation) - 1
        assert pdut.dfe_freeze_ui + 1 <= n_clocks - pdut.eye_uis, "Tap weights changed during the eye window!"


def test_gain_warning():
    """With too little time to adapt before the eye, the user is warned."""
    dut = PyBERT(run_simulation=False, gui=False)
    dut.gain = 0.01
    dut.simulate(initial_run=True)
    # Its LMS update gives a time constant of `n_ave / (gain * level^2 * E[d^2])`; NRZ: E[d^2] = 1.
    # (Nearly: a few decisions, made before the signal arrives, are 0.)
    assert dut.dfe_tau_ui == pytest.approx(dut.n_ave / (dut.gain * dut.decision_scaler**2), rel=1e-3)
    assert "can't converge before eye data collection" in dut.dfe_warning
    assert dut.dfe_warning in dut.console_log


def test_alert_from_sim_thread(monkeypatch):
    """An alert raised off the main thread (e.g. - by the simulation thread) is deferred to the GUI thread."""
    import threading

    from pybert import pybert as pybert_mod

    calls: list = []
    monkeypatch.setattr(pybert_mod, "message", lambda *args, **kwargs: calls.append("direct"))
    monkeypatch.setattr(pybert_mod.GUI, "invoke_later", lambda *args, **kwargs: calls.append("deferred"))
    dut = PyBERT(run_simulation=False, gui=False)
    dut.GUI = True
    thread = threading.Thread(target=dut.log, args=("Test alert.",), kwargs={"alert": True})
    thread.start()
    thread.join()
    dut.log("Test alert.", alert=True)
    assert calls == ["deferred", "direct"]
