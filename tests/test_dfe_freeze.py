"""Test freezing the DFE tap weights, once they've stabilized (``dfe_freeze``)."""
from pathlib import Path

import numpy as np
import pytest

from pybert.models.dfe import DFE
from pybert.pybert import PyBERT
from pybert.utility import lfsr_bits

UI = 100e-12
NSPUI = 8
PERIOD = 2 * (2**7 - 1)  # PRBS-7, doubled as `my_run_simulation()` does.
H = (0.3, 0.1)  # Post-cursor ISI, relative to the cursor.


def run_dfe(freeze_window: int, freeze_tol: float = 2e-3) -> tuple[np.ndarray, int]:
    """Run a 2-tap NRZ DFE, clocked at a fixed phase, over a noisy channel w/ known ISI.
    (The noise is heavy enough that the tolerance must be looser than the default, to freeze.)

    Returns:
        The tap weights at each UI, and the UI at which they were frozen (-1 if never).
    """
    gen = lfsr_bits([7, 6], 1)
    syms = np.array([next(gen) for _ in range(40 * PERIOD)]) * 2.0 - 1.0
    x = 0.5 * (syms + H[0] * np.roll(syms, 1) + H[1] * np.roll(syms, 2))
    x += np.random.default_rng(0).normal(0, 0.03, len(x))
    sig = np.repeat(x, NSPUI)
    t = np.arange(len(sig)) * UI / NSPUI
    dfe = DFE(len(H), 0.1, 1e-12, 0.01, UI, NSPUI, 0.5, n_ave=10, bandwidth=10e9,
              freeze_window=freeze_window, freeze_tol=freeze_tol)
    dbg: dict = {}
    tap_weights = dfe.run_dfe(t, sig, dbg_dict=dbg, clock_phase=UI / 2)[1]
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


@pytest.mark.parametrize("pdut", [{"dfe_freeze": True, "gain": 0.5}], indirect=True)
class TestDfeFreezeSim:
    """Default configuration, simulated with DFE freezing enabled.
    (The default gain is too low for the DFE to stabilize within the default run length.)"""

    def test_ber(self, pdut):
        """No bit errors."""
        assert pdut.n_errs_dfe == 0, "Bit errors detected!"

    def test_frozen(self, pdut):
        """The DFE freezes, and its tap weights don't change thereafter."""
        if not any(t.enabled for t in pdut.dfe_tap_tuners):
            pytest.skip("No DFE taps enabled in the default configuration.")
        assert pdut.dfe_freeze_ui >= 0, "DFE never froze!"
        tap_weights = np.array(pdut.adaptation)
        assert np.all(tap_weights[pdut.dfe_freeze_ui + 1:] == tap_weights[-1])
