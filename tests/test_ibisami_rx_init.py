"""
Run some basic tests on a PyBERT instance w/
an Rx IBIS-AMI model run in statistical mode.
"""

import numpy as np
import pytest


@pytest.mark.usefixtures("ibisami_rx_init")
class TestIbisAmiRxInit(object):
    """
    Basic tests of a properly initialized PyBERT w/
    user-defined channel impulse response length.
    """

    def test_status(self, dut):
        """Test post-simulation status."""
        assert dut.status == "Ready.", "Status not 'Ready.'!"

    def test_perf(self, dut):
        """Test simulation performance."""
        assert dut.total_perf > (1e6 / 60), "Performance dropped below 1 Msmpls/min.!"

    def test_ber(self, dut):
        """Test simulation bit errors."""
        assert not dut.bit_errs, "Bit errors detected!"

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
