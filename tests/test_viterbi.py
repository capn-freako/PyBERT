"""Test Viterbi decoder of a PyBERT instance."""
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
        n_errs = dut_viterbi.n_errs
        assert n_errs == 0, f"{n_errs} bit errors from Viterbi decoder detected!"


dut_cfg_stressed = {
    "l_ch": 2.0,
}


@pytest.mark.parametrize("pdut_viterbi_vs_dfe",
    [dut_cfg_stressed,
    ], indirect=True)
class TestViterbiStressed(object):
    """Test Viterbi decoder of a properly initialized PyBERT w/ stressed eye."""

    def test_perf(self, pdut_viterbi_vs_dfe):
        """Test relative Viterbi decoder performance."""
        n_errs_dfe     = pdut_viterbi_vs_dfe.n_errs_dfe
        n_errs_viterbi = pdut_viterbi_vs_dfe.n_errs_viterbi
        # assert n_errs_viterbi == 0 or n_errs_viterbi < n_errs_dfe, \
        assert n_errs_viterbi < n_errs_dfe, \
            f"No improvement from Viterbi decoder ({n_errs_viterbi} errors), relative to DFE ({n_errs_dfe} errors)!"


@pytest.mark.usefixtures("dut_viterbi_1p5mChannel")
class TestViterbi1p5mChannel(object):
    """Test Viterbi decoder on ``chnl_1p5.yaml`` configuration."""

    def test_perf(self, dut_viterbi_1p5mChannel):
        """Test relative Viterbi decoder performance."""
        n_errs_dfe     = dut_viterbi_1p5mChannel.n_errs_dfe
        n_errs_viterbi = dut_viterbi_1p5mChannel.n_errs_viterbi
        # assert n_errs_viterbi == 0 or n_errs_viterbi < n_errs_dfe, \
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
        n_errs = pdut.n_errs
        assert n_errs == 0, f"{n_errs} bit errors from Viterbi decoder detected!"
