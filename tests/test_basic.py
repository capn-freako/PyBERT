"""Run some basic tests on a PyBERT instance."""
import numpy as np
import pytest

from pybert.pybert import PyBERT


@pytest.mark.usefixtures("dut")
class TestBasic(object):
    """Basic tests of a properly initialized PyBERT."""

    def test_status(self, dut):
        """Test post-simulation status."""
        assert dut.status == "Ready.", "Status not 'Ready.'!"

    def test_perf(self, dut):
        """Test simulation performance."""
        assert dut.total_perf > (1e6 / 60), "Performance dropped below 1 Msmpls/min.!"

    def test_ber(self, dut):
        """Test simulation bit errors."""
        assert dut.n_errs_dfe == 0, "Bit errors detected!"

    def test_dly(self, dut):
        """Test channel delay."""
        assert dut.chnl_dly > 1e-9 and dut.chnl_dly < 10e-9, "Channel delay is out of range!"

    def test_isi(self, dut):
        """Test ISI portion of jitter."""
        assert dut.isi_dfe < 50e-12, "ISI is too high!"

    def test_dcd(self, dut):
        """Test DCD portion of jitter."""
        assert dut.dcd_dfe < 20e-12, "DCD is too high!"

    def test_pj(self, dut):
        """Test periodic portion of jitter."""
        assert dut.pj_dfe < 20e-12, "Periodic jitter is too high!"

    def test_rj(self, dut):
        """Test random portion of jitter."""
        assert dut.rj_dfe < 20e-12, "Random jitter is too high!"

    def test_lock(self, dut):
        """Test CDR lock, by ensuring that last 20% of locked indication vector
        is all True."""
        _lockeds = dut.lockeds
        assert all(_lockeds[4 * len(_lockeds) // 5 :]), "CDR lock is unstable!"

    def test_adapt(self, dut):
        """Test DFE lock, by ensuring that last 20% of all coefficient vectors
        are stable to within +/-20% of their mean."""
        _weights = dut.adaptation  # rows = step; cols = tap
        _ws = np.array(list(zip(*_weights[4 * len(_weights) // 5 :])))  # zip(*x) = unzip(x)
        _means = list(map(lambda xs: sum(xs) / len(xs), _ws))
        assert all(
            list(map(lambda pr: pr[1] == 0 or all(abs(pr[0] - pr[1]) / pr[1] < 0.2), zip(_ws, _means)))
        ), f"DFE adaptation is unstable! {max(_ws[-1])} {min(_ws[-1])}"


def test_disabled_rx_ffe_taps_not_applied():
    """A disabled Rx FFE tap contributes nothing to the simulated FFE, whatever value it holds."""
    from pybert.pybert import PyBERT

    p = PyBERT(run_simulation=False, gui=False)
    p.nbits, p.eye_bits = 1000, 500
    p.rx_n_taps = 5  # disables taps 5 and up
    p.rx_taps[7].value = 0.5
    p.simulate(initial_run=True, update_plots=False)
    assert not p.rx_taps[7].enabled
    assert p.ffe_h[7 * p.nspui] == 0.0



def test_ctle_impulse_not_limited_by_sim_frequency_grid():
    """A 0 dB CTLE is a plain `rx_bw` low-pass; with `f_max` no higher than `rx_bw`, the simulation's
    CTLE impulse response must still not ring (it used to, from `irfft()` over the truncated grid)."""
    from pybert.pybert import PyBERT

    p = PyBERT(run_simulation=False, gui=False)
    p.nbits, p.eye_bits = 1000, 500
    p.f_max, p.rx_bw, p.peak_mag, p.ctle_enable = 40.0, 40.0, 0.0, True
    p.simulate(initial_run=True, update_plots=False)
    step = np.cumsum(p.ctle_h)
    assert step.max() / step[-1] - 1 < 1e-3
    assert p.ctle_h.min() > -1e-3 * p.ctle_h.max()


def test_gui_startup_without_simulation():
    """The GUI's plots exist before any simulation (none runs at start-up), and the first one fills them in."""
    from pybert.models.bert import my_run_simulation

    dut = PyBERT(run_simulation=False)
    assert dut.plots_eye is not None and dut.plot_h_tune is not None
    assert dut.plotdata.get_data("eye_dfe").shape == (2, 2)  # Placeholder
    my_run_simulation(dut)  # As run by the GUI's "Run" button (`RunSimThread`).
    assert dut.plotdata.get_data("eye_dfe").shape != (2, 2)
    assert dut.n_errs_dfe == 0


def test_plotdata_per_instance():
    """Each `PyBERT` instance has its own plot data, so one's results never show up in another's plots."""
    p1 = PyBERT(run_simulation=False, gui=False)
    p2 = PyBERT(run_simulation=False, gui=False)
    assert p1.plotdata is not p2.plotdata
    p1.plotdata.set_data("probe", np.zeros(3))
    assert p2.plotdata.get_data("probe") is None
