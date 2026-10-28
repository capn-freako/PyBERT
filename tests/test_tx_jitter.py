"""Test the Tx jitter (Issue #85): the IBIS-AMI Tx jitter Reserved Parameters (IBIS 8.0 Sec. 10.6.1),
and the native traits they set."""
from pathlib import Path

import numpy as np
import pytest

from pybert.pybert import PyBERT
from pybert.utility import jittered_waveform, tx_jitter_offsets

UI = 100e-12
N = 200_000  # Clock times, for statistics.

TX_AMI = """(my_tx
  (Reserved_Parameters
    (AMI_Version (Usage Info) (Type String) (Value "{version}"))
    (Init_Returns_Impulse (Usage Info) (Type Boolean) (Value True))
    (GetWave_Exists (Usage Info) (Type Boolean) (Value False))
{params}
  )
  (Model_Specific
    (foo (Usage In) (Type Integer) (List 1 2) (Default 1))
  )
)
"""


def load_tx_ami(tmp_path: Path, params: str, version: str = "7.0", dut: PyBERT | None = None) -> PyBERT:
    """Load a Tx AMI file declaring ``params`` (its Reserved Parameters' text) into ``dut``
    (a new ``PyBERT``, by default)."""
    ami_file = tmp_path / "my_tx.ami"
    ami_file.write_text(TX_AMI.format(version=version, params=params))
    if dut is None:
        dut = PyBERT(run_simulation=False, gui=False)
    dut.tx_ami_file = str(ami_file)
    assert dut.tx_ami_valid
    return dut


class TestOffsets:
    """``tx_jitter_offsets()`` follows the spec's ``Time(n)`` formula for each parameter."""

    def test_none(self):
        """No jitter, no displacements, and no random draws."""
        state = np.random.get_state()[1].copy()
        assert not tx_jitter_offsets(N, UI, sj=1e-12).any()  # Tx_Sj needs Tx_Sj_Frequency.
        assert np.array_equal(np.random.get_state()[1], state)

    def test_dcd(self):
        """``Tx_DCD * (-1)^n``."""
        offsets = tx_jitter_offsets(5, UI, dcd=2e-12)
        assert offsets == pytest.approx([2e-12, -2e-12, 2e-12, -2e-12, 2e-12])

    def test_dj(self):
        """``2 * Tx_Dj * rand()``, ``rand()`` uniform over [-0.5, +0.5]."""
        offsets = tx_jitter_offsets(N, UI, dj=3e-12, rng=np.random.default_rng(0))
        assert abs(offsets).max() <= 3e-12
        assert abs(offsets).max() == pytest.approx(3e-12, rel=1e-3)
        assert offsets.std() == pytest.approx(3e-12 / np.sqrt(3), rel=0.01)

    def test_rj(self):
        """``Tx_Rj * gaussian_rand()``."""
        offsets = tx_jitter_offsets(N, UI, rj=1e-12, rng=np.random.default_rng(0))
        assert offsets.mean() == pytest.approx(0, abs=0.01e-12)
        assert offsets.std() == pytest.approx(1e-12, rel=0.01)

    def test_sj(self):
        """``Tx_Sj * sin(2 * pi * Tx_Sj_Frequency * n * symbol_time)``."""
        offsets = tx_jitter_offsets(N, UI, sj=2e-12, sj_freq=1e9)
        n = np.arange(N)
        assert offsets == pytest.approx(2e-12 * np.sin(2 * np.pi * 1e9 * n * UI), abs=1e-20)


class TestWaveform:
    """``jittered_waveform()`` moves the symbol boundaries, to a fraction of a sample."""

    def test_no_offsets(self):
        """With no displacements, it's the plain over-sampled waveform."""
        symbols = np.array([-1.0, 1.0, 1.0, -1.0, 1 / 3, -1 / 3])
        waveform, n_clipped = jittered_waveform(symbols, 8, np.zeros(len(symbols) + 1))
        assert waveform == pytest.approx(np.repeat(symbols, 8), abs=1e-12)
        assert n_clipped == 0

    def test_fractional_edges(self):
        """An edge between samples gives the sample it falls in the time average of the levels either side."""
        symbols = np.array([-1.0, 1.0, -1.0])
        waveform = jittered_waveform(symbols, 4, np.array([0, 0.1, -0.3, 0]))[0]
        # Edge 1, at 1.1 UI = sample 4.4; edge 2, at 1.7 UI = sample 6.8.
        assert waveform == pytest.approx([-1, -1, -1, -1, 0.2, 1, 0.6, -1, -1, -1, -1, -1])

    def test_clipping(self):
        """Displacements beyond +/-0.5 UI are clipped, keeping the boundaries in order, and counted."""
        symbols = np.array([1.0, -1.0, 1.0, -1.0])
        waveform, n_clipped = jittered_waveform(symbols, 4, np.array([0, 0.7, -0.6, 0.2, 0]))
        assert n_clipped == 2
        assert waveform.sum() == pytest.approx((symbols * np.diff([0, 1.5, 1.5, 3.2, 4]) * 4).sum(), abs=1e-4)

    def test_offsets_length(self):
        """One displacement per symbol boundary, or it's an error."""
        with pytest.raises(ValueError):
            jittered_waveform(np.ones(4), 4, np.zeros(4))


class TestSimulation:
    """PyBERT's own Tx-stage jitter analysis reads back the injected jitter.
    (A short channel, and no voltage noise, isolate it: a lossy channel amplifies high-frequency jitter.)"""

    @staticmethod
    def simulate(**jitter) -> PyBERT:
        "Simulate the default configuration over a short channel, with no voltage noise and the given Tx jitter."
        dut = PyBERT(run_simulation=False, gui=False)
        dut.rn, dut.pn_mag, dut.l_ch = 0.0, 0.0, 0.02
        for name, value in jitter.items():
            setattr(dut, name, value)
        dut.simulate(initial_run=True, update_plots=True)
        return dut

    def test_rj(self):
        "Tx Rj reads back as Rj."
        dut = self.simulate(tx_rj=0.02)
        assert dut.rj_tx / dut.ui == pytest.approx(0.02, rel=0.1)
        assert dut.rj_chnl / dut.ui < 1e-6  # The channel's own output is jitter-free.
        assert dut.n_errs_dfe == 0

    def test_sj(self):
        """Tx Sj reads back as Pj (peak-to-peak). (Its frequency is placed on a bin of the jitter spectrum,
        whose periodic-jitter extraction is then exact; between bins, spectral leakage reduces its accuracy.)"""
        dut = PyBERT(run_simulation=False, gui=False)
        sj_freq = 100 / (dut.eye_uis * dut.ui) * 1e-6  # 100 bins (MHz)
        dut = self.simulate(tx_sj=0.02, tx_sj_freq=sj_freq)
        assert dut.pj_tx / dut.ui == pytest.approx(0.04, rel=0.05)
        assert dut.pj_chnl / dut.ui < 1e-6


class TestAmi:
    """A Tx IBIS-AMI model's jitter parameters set the Tx jitter."""

    def test_ui_and_seconds(self, tmp_path):
        """Type UI values are taken as is; Type Float values are seconds; Corner/Range give their typical value."""
        dut = load_tx_ami(tmp_path, """
    (Tx_Rj (Usage Info) (Type Float) (Value 1e-12))
    (Tx_Dj (Usage Info) (Type UI) (Value 0.03))
    (Tx_DCD (Usage Info) (Type UI) (Corner 0.01 0.02 0.005))
    (Tx_Sj (Usage Info) (Type UI) (Range 0.02 0.0 0.05))
    (Tx_Sj_Frequency (Usage Info) (Type Float) (Value 1e9))""")
        assert dut.tx_rj == pytest.approx(1e-12 / dut.ui)
        assert (dut.tx_dj, dut.tx_dcd, dut.tx_sj, dut.tx_sj_freq) == pytest.approx((0.03, 0.01, 0.02, 1000.0))
        assert "Tx jitter set from the IBIS-AMI model" in dut.console_log

    def test_undeclared_kept(self, tmp_path):
        """Parameters the model doesn't declare leave their traits alone; a lone Tx_Sj is flagged as ignored."""
        dut = PyBERT(run_simulation=False, gui=False)
        dut.tx_rj = 0.01
        load_tx_ami(tmp_path, "    (Tx_Sj (Usage Info) (Type UI) (Value 0.02))", dut=dut)
        assert dut.tx_rj == 0.01
        assert dut.tx_sj == 0.02
        assert "Tx_Sj is ignored" in dut.console_log

    def test_old_ami_version(self, tmp_path):
        """Tx_Rj/Tx_Dj/Tx_Sj/Tx_Sj_Frequency are illegal before AMI_Version 6.0 (Tx_DCD isn't): used, with a warning."""
        dut = load_tx_ami(tmp_path, """
    (Tx_Rj (Usage Info) (Type UI) (Value 0.01))
    (Tx_DCD (Usage Info) (Type UI) (Value 0.01))""", version="5.1")
        assert dut.tx_rj == 0.01
        assert "Tx_Rj are illegal before AMI_Version 6.0" in dut.console_log

    def test_no_jitter_params(self, tmp_path):
        "A model without any leaves the Tx jitter alone, and says nothing."
        dut = load_tx_ami(tmp_path, "")
        assert dut.tx_rj == dut.tx_dj == dut.tx_dcd == dut.tx_sj == dut.tx_sj_freq == 0
        assert "Tx jitter" not in dut.console_log


def test_config_round_trip(tmp_path):
    """The Tx jitter survives a YAML save/load round-trip, overriding what the config's Tx AMI model declares."""
    dut = load_tx_ami(tmp_path, "    (Tx_Rj (Usage Info) (Type UI) (Value 0.01))")
    dut.tx_rj, dut.tx_dj, dut.tx_dcd, dut.tx_sj, dut.tx_sj_freq = 0.005, 0.01, 0.015, 0.02, 25.0
    save_file = tmp_path / "tx_jitter.yaml"
    dut.save_configuration(save_file)

    dut2 = PyBERT(run_simulation=False, gui=False)
    dut2.load_configuration(save_file)
    assert dut2.tx_ami_file == dut.tx_ami_file
    assert (dut2.tx_rj, dut2.tx_dj, dut2.tx_dcd, dut2.tx_sj, dut2.tx_sj_freq) == (0.005, 0.01, 0.015, 0.02, 25.0)
