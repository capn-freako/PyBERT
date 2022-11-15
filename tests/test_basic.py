"""Run some basic tests on a PyBERT instance."""
import numpy as np

import pybert
from pybert.pybert import PyBERT


class TestBasic(object):
    """Basic tests of a properly initialized PyBERT."""

    dut = PyBERT(gui=False)

    def test_version(self):
        assert pybert.__version__ == "3.5.8"

    def test_status(self):
        """Test post-simulation status."""
        assert self.dut.status == "Ready.", "Status not 'Ready.'!"

    def test_perf(self):
        """Test simulation performance."""
        assert self.dut.total_perf > (1e6 / 60), "Performance dropped below 1 Msmpls/min.!"

    def test_ber(self):
        """Test simulation bit errors."""
        assert not self.dut.bit_errs, "Bit errors detected!"

    def test_dly(self):
        """Test channel delay."""
        assert self.dut.chnl_dly > 1e-9 and self.dut.chnl_dly < 10e-9, "Channel delay is out of range!"

    def test_isi(self):
        """Test ISI portion of jitter."""
        assert self.dut.isi_dfe < 50e-12, "ISI is too high!"

    def test_dcd(self):
        """Test DCD portion of jitter."""
        assert self.dut.dcd_dfe < 20e-12, "DCD is too high!"

    def test_pj(self):
        """Test periodic portion of jitter."""
        assert self.dut.pj_dfe < 20e-12, "Periodic jitter is too high!"

    def test_rj(self):
        """Test random portion of jitter."""
        assert self.dut.rj_dfe < 20e-12, "Random jitter is too high!"

    def test_lock(self):
        """Test CDR lock, by ensuring that last 20% of locked indication vector
        is all True."""
        _lockeds = self.dut.lockeds
        assert all(_lockeds[4 * len(_lockeds) // 5 :]), "CDR lock is unstable!"

    def test_adapt(self):
        """Test DFE lock, by ensuring that last 20% of all coefficient vectors
        are stable to within +/-20% of their mean."""
        _weights = self.dut.adaptation  # rows = step; cols = tap
        _ws = np.array(list(zip(*_weights[4 * len(_weights) // 5 :])))  # zip(*x) = unzip(x)
        _means = list(map(lambda xs: sum(xs) / len(xs), _ws))
        assert all(
            map(lambda pr: all(abs(pr[0] - pr[1]) / pr[1] < 0.2), zip(_ws, _means))
        ), "DFE adaptation is unstable!"
