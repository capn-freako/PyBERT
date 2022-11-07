"""Run some basic tests on a PyBERT instance.
"""
from pybert.pybert      import PyBERT
import pybert

class TestBasic(object):
    def test_version(self):
        assert pybert.__version__ == "3.5.8"

    def test_status(self):
        """Test post-simulation status."""
        dut = PyBERT(gui=False)
        assert dut.status == "Ready.", "Status not 'Ready.'!"

    def test_perf(self):
        """Test simulation performance."""
        dut = PyBERT(gui=False)
        assert dut.total_perf > (1e6/60), "Performance dropped below 1 Msmpls/min.!"
