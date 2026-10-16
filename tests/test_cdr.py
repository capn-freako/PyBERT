"""Test the Mueller-Muller CDR option (``cdr_type``)."""
import numpy as np
import pytest

from pychopmarg.utility import loc_curs


@pytest.mark.parametrize("pdut", [{"cdr_type": "Mueller-Muller"}], indirect=True)
class TestMuellerMuller:
    """Default configuration, simulated with the Mueller-Muller CDR."""

    def test_ber(self, pdut):
        """No bit errors."""
        assert pdut.n_errs_dfe == 0, "Bit errors detected!"

    def test_clock_phase(self, pdut):
        """Clocks land on the pulse-response sample that meets the Mueller-Muller criterion, once per nominal UI."""
        limits = [(t.min_val, t.max_val) for t in pdut.dfe_tap_tuners if t.enabled][:1]
        dfe_max, dfe_min = (np.array([limits[0][1]]), np.array([limits[0][0]])) if limits else (np.zeros(1), np.zeros(1))
        curs_ix = loc_curs(pdut.ffe_out_p, pdut.nspui, dfe_max, dfe_min)
        clock_ixs = np.where(pdut.clocks)[0]
        assert all(clock_ixs % pdut.nspui == curs_ix % pdut.nspui), "Clock phase doesn't match the MM cursor!"
        assert all(np.diff(clock_ixs) == pdut.nspui), "Clock period isn't the nominal UI!"

    def test_dfe_start(self, pdut):
        """DFE tap 1 starts at the value the Mueller-Muller lock point anticipates."""
        limits = [(t.min_val, t.max_val) for t in pdut.dfe_tap_tuners if t.enabled][:1]
        if not limits:
            pytest.skip("No DFE taps enabled in the default configuration.")
        curs_ix = loc_curs(pdut.ffe_out_p, pdut.nspui, np.array([limits[0][1]]), np.array([limits[0][0]]))
        b_1 = min(limits[0][1], max(limits[0][0], pdut.ffe_out_p[curs_ix + pdut.nspui] / pdut.ffe_out_p[curs_ix]))
        assert pdut.adaptation[0][0] == pytest.approx(b_1), "DFE tap 1 didn't start at the anticipated value!"
