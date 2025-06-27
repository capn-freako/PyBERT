"""Test Viterbi decoder of a PyBERT instance."""
import numpy as np
import pytest


@pytest.mark.usefixtures("dut_viterbi")
class TestViterbi(object):
    """Test Viterbi decoder of a properly initialized PyBERT."""

    def test_status(self, dut):
        """Test post-simulation status."""
        assert dut.status == "Ready.", "Status not 'Ready.'!"

    def test_ber(self, dut):
        """Test simulation bit errors."""
        n_errs = dut.bit_errs_viterbi
        assert n_errs <= 0, f"{n_errs} bit errors from Viterbi decoder detected!"


@pytest.mark.usefixtures("dut_viterbi_stressed")
class TestViterbiStressed(object):
    """Test Viterbi decoder of a properly initialized PyBERT w/ stressed eye."""

    def test_perf(self, dut):
        """Test relative Viterbi decoder performance."""
        n_errs = dut.bit_errs
        n_errs_viterbi = dut.bit_errs_viterbi
        assert n_errs_viterbi < n_errs, f"No improvement from Viterbi decoder ({n_errs_viterbi} errors), relative to DFE ({n_errs} errors)!"
