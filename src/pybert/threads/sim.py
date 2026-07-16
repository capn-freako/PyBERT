"Thread for running the PyBERT simulation."

from pybert.models.bert import my_run_simulation
from pybert.threads.stoppable import StoppableThread


class RunSimThread(StoppableThread):
    """Used to run the simulation in its own thread, in order to preserve GUI
    responsiveness."""

    def __init__(self, initial_run: bool = False, update_plots: bool = True):
        """
        Keyword Args:
            initial_run: If True, don't update the eye diagrams, since
                they haven't been created, yet.
                Default: False
            update_plots: If True, update the plots, after simulation
                completes. This option can be used by larger scripts, which
                import *pybert*, in order to avoid graphical back-end
                conflicts and speed up this function's execution time.
                Default: True
        """
        super().__init__()
        self.the_pybert = None
        self.initial_run = initial_run
        self.update_plots = update_plots

    def run(self):
        """Run the simulation(s)."""
        try:
            my_run_simulation(
                self.the_pybert,
                initial_run=self.initial_run,
                update_plots=self.update_plots,
                aborted_sim=self.stopped
            )
        except RuntimeError as err:
            print(f"Error in `pybert.threads.sim.RunSimThread`: {err}")
            raise
