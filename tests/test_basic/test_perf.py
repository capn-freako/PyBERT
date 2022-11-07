"""Test performance of default run."""
from pybert.pybert      import PyBERT

dut = PyBERT(gui=False)
assert dut.total_perf > (1e6/60), "Performance dropped below 1 Msmpls/min.!"
