"""Tests for COM metric reporting via PyChOpMarg."""

import math
from pathlib import Path

import pytest

from pybert.models.com import calc_com
from pybert.pybert import PyBERT

EXAMPLE_THRU = Path(__file__).parent.parent / "PyChOpMarg" / "chnl_data" / "example2_THRU.s4p"


class TestComAttribute:
    """Verify that PyBERT exposes com_value and enable_com correctly."""

    def test_enable_com_default(self, dut):
        """COM computation is disabled by default."""
        assert dut.enable_com is False, "enable_com should default to False."

    def test_com_value_default(self, dut):
        """COM value defaults to sentinel when COM is disabled or native channel is used."""
        assert dut.com_value == -999.0, "com_value should be -999.0 (not computed) when COM is disabled."

    def test_com_info_not_computed(self, dut):
        """com_info reports 'Not computed' when COM was not run."""
        assert "Not computed" in dut.com_info

    def test_com_info_with_value(self):
        """com_info reports the numeric value when com_value has been set."""
        dut = PyBERT(run_simulation=False, gui=False)
        dut.com_value = 5.3
        assert "5.30" in dut.com_info


@pytest.mark.skipif(not EXAMPLE_THRU.exists(), reason="PyChOpMarg example data not available.")
@pytest.mark.slow
class TestCalcCom:
    """Integration tests that call the real COM calculation (slow)."""

    def test_calc_com_returns_float(self):
        """calc_com() returns a finite float for the example THRU channel."""
        result = calc_com(str(EXAMPLE_THRU))
        assert isinstance(result, float)
        assert math.isfinite(result), f"COM result is not finite: {result}"

    def test_calc_com_reasonable_range(self):
        """COM value for the example channel should be within a plausible range (-20 to 50 dB)."""
        result = calc_com(str(EXAMPLE_THRU))
        assert -20.0 <= result <= 50.0, f"COM = {result:.2f} dB is outside the expected range."
