from pybert.models.bert import my_run_sweeps
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
            my_run_sweeps(self.the_pybert, self.stopped)
        except RuntimeError:
            pass
