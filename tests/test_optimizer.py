"""Run some tests on the PyBERT optimizer."""
from time import sleep, time
import numpy as np
import pytest

WATCHDOG_TIMEOUT = 120  # 2 minutes

@pytest.mark.usefixtures("optimization_triplet")
class TestOptimizer(object):
    """Basic tests of PyBERT optimizer."""

    def test_defaults(self, optimization_triplet):
        """Test optimization w/ default settings."""
        
        # Launch optimization.
        dut, handler, info = optimization_triplet
        handler.do_tune_eq(info)

        # Confirm successful optimization launch.
        assert dut.opt_thread and dut.opt_thread.is_alive(), "Optimizer failed to launch!"

        # Wait for optimizer to finish, or watchdog timer to expire.
        start_time = time()
        while dut.opt_thread.is_alive():
            sleep(1.0)
            assert (time() - start_time) < WATCHDOG_TIMEOUT, "Watchdog timed out!"

        # Confirm optimizer success.
        assert dut.status.startswith("Finished"), f"Optimization failed: {dut.status}."

    def test_no_ffe(self, optimization_triplet):
        """Test optimization w/ all Rx FFE taps disabled."""
        
        # Disable Rx FFE and launch optimization.
        dut, handler, info = optimization_triplet
        dut._btn_disable_ffe_fired()
        handler.do_tune_eq(info)

        # Confirm successful optimization launch.
        assert dut.opt_thread and dut.opt_thread.is_alive(), "Optimizer failed to launch!"

        # Wait for optimizer to finish, or watchdog timer to expire.
        start_time = time()
        while dut.opt_thread.is_alive():
            sleep(1.0)
            assert (time() - start_time) < WATCHDOG_TIMEOUT, "Watchdog timed out!"

        # Confirm optimizer success.
        assert dut.status.startswith("Finished"), f"Optimization failed: {dut.status}."

    def test_viterbi(self, optimization_triplet):
        """Test Viterbi decoder, post-optimization."""
        
        # Launch optimization.
        dut, handler, info = optimization_triplet
        handler.do_tune_eq(info)

        # Confirm successful optimization launch.
        assert dut.opt_thread and dut.opt_thread.is_alive(), "Optimizer failed to launch!"

        # Wait for optimizer to finish, or watchdog timer to expire.
        start_time = time()
        while dut.opt_thread.is_alive():
            sleep(1.0)
            assert (time() - start_time) < WATCHDOG_TIMEOUT, "Watchdog timed out!"

        # Confirm optimizer success.
        assert dut.status.startswith("Finished"), f"Optimization failed: {dut.status}."

        # Run optimized simulation w/ Viterbi decoder enabled.
        handler.do_use_eq(info)
        thePyBERT = info.object
        thePyBERT.rx_viterbi_symbols = 2  # Solely to speed up testing.
        thePyBERT.rx_use_viterbi = True
        thePyBERT.simulate(initial_run=True)
        assert thePyBERT.status == "Ready.", "Status not 'Ready.'!"
        n_errs = thePyBERT.bit_errs_viterbi
        assert n_errs <= 0, f"{n_errs} bit errors from Viterbi decoder detected!"
