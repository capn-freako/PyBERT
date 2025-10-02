"""Test Viterbi decoder of a PyBERT instance."""
import numpy as np
import pytest

dut_cfg_pam4_viterbi_2 = {
    "mod_type": "PAM-4",
    "rx_use_viterbi": True,
    "rx_viterbi_symbols": 2,
}


@pytest.mark.usefixtures("dut_viterbi")
class TestViterbi(object):
    """Test Viterbi decoder of a properly initialized PyBERT."""

    def test_status(self, dut):
        """Test post-simulation status."""
        assert dut.status == "Ready.", "Status not 'Ready.'!"

    def test_ber(self, dut):
        """Test simulation bit errors."""
        n_errs = dut.bit_errs
        assert n_errs == 0, f"{n_errs} bit errors from Viterbi decoder detected!"


@pytest.mark.usefixtures("dut_viterbi_stressed")
class TestViterbiStressed(object):
    """Test Viterbi decoder of a properly initialized PyBERT w/ stressed eye."""

    def test_perf(self, dut):
        """Test relative Viterbi decoder performance."""
        n_errs = dut.bit_errs
        n_errs_viterbi = dut.bit_errs
        assert n_errs_viterbi < n_errs, f"No improvement from Viterbi decoder ({n_errs_viterbi} errors), relative to DFE ({n_errs} errors)!"


@pytest.mark.usefixtures("dut_viterbi_1p5mChannel")
class TestViterbi1p5mChannel(object):
    """Test Viterbi decoder on ``chnl_1p5.yaml`` configuration."""

    def test_perf(self, dut):
        """Test relative Viterbi decoder performance."""
        n_errs = dut.bit_errs
        n_errs_viterbi = dut.bit_errs
        assert n_errs_viterbi < n_errs, f"No improvement from Viterbi decoder ({n_errs_viterbi} errors), relative to DFE ({n_errs} errors)!"


@pytest.mark.parametrize("pdut", [dut_cfg_pam4_viterbi_2], indirect=True)
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
        n_errs = pdut.bit_errs
        assert n_errs == 0, f"{n_errs} bit errors from Viterbi decoder detected!"
