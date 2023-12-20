from pybert.pybert import PyBERT
from pybert.threads.sim import RunSimThread


def test_simulation_can_abort():
    """Test that spawning a simulation thread can be aborted.

    The timeout on join does not kill the thread but if reached will stop
    blocking.  This just guards against pytest infinitely hanging up because
    of some event.  The simulation should abort within a second or two.
    """
    app = PyBERT(run_simulation=False, gui=False)

    sim = RunSimThread()
    sim.the_pybert = app
    sim.start()  # Start the thread
    sim.stop()  # Abort the thread
    sim.join(60)  # Join and wait until it ends or 10 seconds passed.

    assert "Aborted" in app.status_str
