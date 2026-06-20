"""
Run some basic tests on a PyBERT instance w/
an Rx IBIS-AMI model run in statistical mode.
"""

import numpy as np
import pytest


class TestIbisAmiRxInit(object):
    """
    Basic tests of a properly initialized PyBERT w/
    Rx IBIS-AMI model in statistical mode.
    """

    def test_rx_sel(self, ibisami_rx_init):
        """Confirm that the Rx IBIS path is actually active."""
        assert ibisami_rx_init.rx_sel == "ibis", "rx_sel was not set to 'ibis'!"

    def test_status(self, ibisami_rx_init):
        """Test post-simulation status."""
        assert ibisami_rx_init.status == "Ready.", "Status not 'Ready.'!"

    def test_perf(self, ibisami_rx_init):
        """Test simulation performance."""
        assert ibisami_rx_init.total_perf > (1e6 / 60), "Performance dropped below 1 Msmpls/min.!"

    def test_ber(self, ibisami_rx_init):
        """Test simulation bit errors."""
        assert ibisami_rx_init.n_errs_dfe == 0, "Bit errors detected!"

    def test_dly(self, ibisami_rx_init):
        """Test channel delay."""
        assert ibisami_rx_init.chnl_dly > 1e-9 and ibisami_rx_init.chnl_dly < 10e-9, "Channel delay is out of range!"

    def test_isi(self, ibisami_rx_init):
        """Test ISI portion of jitter."""
        assert ibisami_rx_init.isi_dfe < 50e-12, "ISI is too high!"

    def test_dcd(self, ibisami_rx_init):
        """Test DCD portion of jitter."""
        assert ibisami_rx_init.dcd_dfe < 20e-12, "DCD is too high!"

    def test_pj(self, ibisami_rx_init):
        """Test periodic portion of jitter."""
        assert ibisami_rx_init.pj_dfe < 40e-12, "Periodic jitter is too high!"

    def test_rj(self, ibisami_rx_init):
        """Test random portion of jitter."""
        assert ibisami_rx_init.rj_dfe < 20e-12, "Random jitter is too high!"

    def test_lock(self, ibisami_rx_init):
        """Test CDR lock, by ensuring that last 20% of locked indication vector
        is all True."""
        _lockeds = ibisami_rx_init.lockeds
        assert all(_lockeds[4 * len(_lockeds) // 5 :]), "CDR lock is unstable!"

    def test_adapt(self, ibisami_rx_init):
        """Test DFE lock, by ensuring that last 20% of all coefficient vectors
        are stable to within +/-20% of their mean."""
        _weights = ibisami_rx_init.adaptation  # rows = step; cols = tap
        _ws = np.array(list(zip(*_weights[4 * len(_weights) // 5 :])))  # zip(*x) = unzip(x)
        _means = list(map(lambda xs: sum(xs) / len(xs), _ws))
        assert all(
            list(map(lambda pr: pr[1] == 0 or all(abs(pr[0] - pr[1]) / pr[1] < 0.2), zip(_ws, _means)))
        ), f"DFE adaptation is unstable! {max(_ws[-1])} {min(_ws[-1])}"
