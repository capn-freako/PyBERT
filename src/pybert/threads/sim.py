"Thread for running the PyBERT simulation."

from pybert.models.bert import my_run_simulation
from pybert.threads.stoppable import StoppableThread


class RunSimThread(StoppableThread):
    """Used to run the simulation in its own thread, in order to preserve GUI
    responsiveness."""

    def __init__(self):
        super().__init__()
        self.the_pybert = None

    def run(self):
        """Run the simulation(s)."""
        try:
            my_run_simulation(self.the_pybert, aborted_sim=self.stopped)
        except RuntimeError as err:
            print(f"Error in `pybert.threads.sim.RunSimThread`: {err}")
            raise
