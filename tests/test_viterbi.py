"""Test Viterbi decoder of a PyBERT instance."""
from pathlib import Path
import numpy as np
import pytest

@pytest.mark.usefixtures("dut_viterbi")
class TestViterbi(object):
    """Test Viterbi decoder of a properly initialized PyBERT."""

    def test_status(self, dut_viterbi):
        """Test post-simulation status."""
        assert dut_viterbi.status == "Ready.", "Status not 'Ready.'!"

    def test_ber(self, dut_viterbi):
        """Test simulation bit errors."""
        n_errs = dut_viterbi.n_errs_viterbi
        assert n_errs == 0, f"{n_errs} bit errors from Viterbi decoder detected!"


dut_cfg_stressed = {
    "l_ch": 2.0,
}


@pytest.mark.parametrize("pdut",
    [dut_cfg_stressed,
    ], indirect=True)
class TestViterbiStressed(object):
    """Test Viterbi decoder of a properly initialized PyBERT w/ stressed eye."""

    def test_perf(self, pdut):
        """Test relative Viterbi decoder performance."""
        n_errs_dfe     = pdut.n_errs_dfe
        n_errs_viterbi = pdut.n_errs_viterbi
        assert n_errs_viterbi < n_errs_dfe, \
            f"No improvement from Viterbi decoder ({n_errs_viterbi} errors), relative to DFE ({n_errs_dfe} errors)!"


@pytest.mark.parametrize("cdut",
    [Path("misc", "ViterbiTesting", "chnl_1p75.yaml"),
    ], indirect=True)
class TestViterbi1p75mChannel(object):
    """Test Viterbi decoder on ``chnl_1p75.yaml`` configuration."""

    def test_perf(self, cdut):
        """Test relative Viterbi decoder performance."""
        n_errs_dfe     = cdut.n_errs_dfe
        n_errs_viterbi = cdut.n_errs_viterbi
        assert n_errs_viterbi < n_errs_dfe, \
            f"No improvement from Viterbi decoder ({n_errs_viterbi} errors), relative to DFE ({n_errs_dfe} errors)!"


dut_cfg_pam4_viterbi_2 = {
    "mod_type": "PAM-4",
    "rx_use_viterbi": True,
    "rx_viterbi_symbols": 2,
}


@pytest.mark.parametrize("pdut",
    [dut_cfg_pam4_viterbi_2,
    ], indirect=True)
class TestViterbiPAM4(object):
    """Test Viterbi decoder on PAM4 channel coding."""

    # Not allowed!
    # def __init__(self):
    #     dut = PyBERT(run_simulation=False, gui=False)
    #     dut.mod_type = "PAM-4"
    #     dut.rx_use_viterbi = True
    #     dut.rx_viterbi_symbols = 2
    #     dut.simulate(initial_run=True)
    #     self.dut = dut

    # def test_status(self, dut):
    def test_status(self, pdut):
        """Test post-simulation status."""
        assert pdut.status == "Ready.", "Status not 'Ready.'!"

    # def test_ber(self, dut):
    def test_ber(self, pdut):
        """Test simulation bit errors."""
        n_errs = pdut.n_errs_viterbi
        assert n_errs == 0, f"{n_errs} bit errors from Viterbi decoder detected!"
