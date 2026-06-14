"Thread for running the PyBERT simulation."

from pybert.models.bert import my_run_ch_sweep, my_run_simulation
from pybert.threads.stoppable import StoppableThread


class RunSimThread(StoppableThread):
    """Used to run the simulation in its own thread, in order to preserve GUI
    responsiveness."""

    def __init__(self):
        super().__init__()
        self.the_pybert = None

    def run(self):
        """Run a channel sweep or single simulation depending on configuration."""
        try:
            pybert = self.the_pybert
            if pybert.use_ch_files and pybert.ch_files:
                my_run_ch_sweep(pybert, aborted_sim=self.stopped)
            else:
                my_run_simulation(pybert, aborted_sim=self.stopped)
        except RuntimeError as err:
            print(f"Error in `pybert.threads.sim.RunSimThread`: {err}")
            raise
